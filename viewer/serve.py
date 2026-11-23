#!/usr/bin/env python3
"""Local viewer for our SGA (SparGuardianAgents) transcripts and the framework's three ladders (Cheating,
Coordination, Collusion).

    .venv/bin/python viewer/serve.py [--port 8770] [--rebuild]

It lists every run of our seven sets in one index and opens a run page with the three ladders per agent and per team,
the code's evidence, the judge replies (controls), the message board and each agent's transcript.

The runs are the relabelled run folders of data/new_ladders (runs/<set>/<run>, spar/<hub path>): processed/ holds
run.json, episodes.jsonl, trajectories/ and communication/; analysis/ holds signals.jsonl and labels/*.json. The run
list, the binary labels and the onsets come from hermeneutics/results/new_ladders/agents.csv (one row per agent of the
activation matrix, its run_dir naming the folder); the per-step levels from steps.csv.gz. The ladders are read by
viewer/levels.py (rungs from hermeneutics/new_ladders.py).

The server reads the data and never writes it. Its one output file is the index cache, viewer/.cache/index.json,
built on the first start and again with --rebuild. A transcript is read only when the browser asks for it, and it is
streamed from disk as stored.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import shutil
import signal
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import levels as three  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data"
NEW = Path(os.environ.get("HERM_NEW_LADDERS", DATA / "new_ladders"))
LABELS = REPO / "hermeneutics" / "results" / "new_ladders"
STATIC = HERE / "static"
CACHE = HERE / ".cache" / "index.json"
INDEX_VERSION = 9
SETS = ["board_seed", "solo_design", "controls_600", "spar_screen_team", "spar_mixed_41", "spar_exploration_single",
        "spar_screen_solo"]

# Our API judges (data/judge_*.jsonl): they judge cheating only, on the 600 controls agents (Sonnet on 464).
API_JUDGES = [
    ("haiku", "judge_anthropic_claude-haiku-5.5.jsonl"),
    ("gpt5m", "judge_openai_gpt-5-mini.jsonl"),
    ("sonnet", "judge_anthropic_claude-sonnet-5.5.jsonl"),
]

# Episode fields the run page shows; the rest of episodes.jsonl (rules, probes, ledger, access) stays on disk.
EPISODE_KEYS = ["episode", "orbit_agent", "agent", "role", "model", "topology", "n_turns", "n_tool_calls", "wall_time",
                "limit", "score", "ended_by", "ok", "failure", "error", "final_text", "usage", "seat", "piece",
                "reward", "team_reward", "defenses", "agent_version"]
RUN_JSON_KEYS = ["task", "variant", "model", "topology", "mode", "eval_log", "code", "orbit_version", "inspect_version",
                 "defenses", "team_submission", "shutdown_rate", "deviations"]
# agents.csv columns the run page shows per agent (the binary labels and the onsets of the three ladders)
LABEL_KEYS = ["uid", "cheat", "coord", "collude", "cheating_rung", "coordination_level", "collusion_rung",
              "cheat_onset_step", "cheat_onset_turn", "cheat_onset_source", "coord_onset_step", "coord_onset_turn",
              "coord_onset_source", "collude_onset_source", "n_steps", "n_turns_y", "old_cheat", "old_coord",
              "old_collude"]


def jsonl(path: Path):
    """Rows of a JSONL file, read line by line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def transcript_dirs(run_dir: Path) -> list[str]:
    """Trajectory folders that hold a transcript.json (names only; no transcript is opened)."""
    tdir = run_dir / "processed" / "trajectories"
    if not tdir.is_dir():
        return []
    return sorted(n for n in os.listdir(tdir) if (tdir / n / "transcript.json").is_file())


def agree_status(label: str | None, judges: dict) -> tuple[str, list[str]]:
    """Whether the Cheating ladder's cheated label and each judge's cheating verdict agree."""
    verdicts = {k: v.get("yes") for k, v in judges.items() if v.get("yes") is not None}
    if not judges:
        return "no judge", []
    if not verdicts:
        return "no answer", []
    if label not in ("cheat", "no cheat"):
        return "not comparable", []
    code_yes = label == "cheat"
    disagree = [k for k, yes in verdicts.items() if yes != code_yes]
    return ("disagree" if disagree else "agree"), disagree


def signals(run_dir: Path):
    """(team row, {agent: agent row}) of a run's analysis/signals.jsonl."""
    rows = list(jsonl(run_dir / "analysis" / "signals.jsonl"))
    return (next((r for r in rows if r.get("row") == "team"), None),
            {r["agent"]: r for r in rows if r.get("row") == "agent"})


def roles_of(sig_agents: dict, rj: dict | None = None) -> dict:
    roles = {a: (s.get("role") or "worker") for a, s in sig_agents.items()}
    for a, r in ((rj or {}).get("swarm") or {}).get("roles", {}).items():
        if r:
            roles[a] = r
    return roles


# ---------------------------------------------------------------------------------------------------------- index
def load_api_judges() -> dict[str, dict[tuple[str, str], dict]]:
    out = {}
    for key, name in API_JUDGES:
        rows = {}
        path = DATA / name
        if path.is_file():
            for r in jsonl(path):
                rows[(r["run"], r["agent"])] = r
        out[key] = rows
    return out


def load_agents() -> dict[str, list[dict]]:
    """agents.csv rows by run id (<set>/<run>), in file order."""
    by_run: dict[str, list[dict]] = {}
    with open(LABELS / "agents.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_run.setdefault(f"{r['source']}/{r['run']}", []).append(r)
    return by_run


def load_steps() -> dict[str, list[list]]:
    """steps.csv.gz: uid -> [[step, turn, cheat_level, coord_level], ...] in step order (None where empty)."""
    out: dict[str, list[list]] = {}
    num = lambda v: int(float(v)) if v not in ("", None) else None  # noqa: E731
    with gzip.open(LABELS / "steps.csv.gz", "rt", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["uid"], []).append([num(r["step"]), num(r["turn"]), num(r["cheat_level"]),
                                                 num(r["coord_level"])])
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return out


def build_index() -> dict:
    t0 = time.time()
    api = load_api_judges()
    rows = []
    for rid, ags in load_agents().items():
        a0 = ags[0]
        rd = NEW / a0["run_dir"]
        team, sa = signals(rd)
        roles = roles_of(sa)
        judges = {}
        if a0["source"] == "controls_600":
            for key, _ in API_JUDGES:
                j = api[key].get((a0["run"], a0["agent"]))
                if j is not None:
                    judges[key] = {"yes": j.get("yes"), "conf": j.get("confidence"), "finish": j.get("finish")}
        rung = (team or {}).get("cheating_rung")
        label = None if rung is None else ("cheat" if three.CHEATING.index(rung) >= three.CUT["cheat"] else "no cheat")
        status, disagree = agree_status(label, judges)
        rows.append({
            "id": rid, "col": a0["source"], "run": a0["run"], "task": a0["task"], "task_name": (team or {}).get("task"),
            "arm": a0["arm"], "author": a0["author_short"], "model_id": a0["author_raw"],
            "author_name": " + ".join(sorted({a["author"] for a in ags if a["author"]})), "n_agents": len(sa),
            "n_labelled": len(ags), "rung": rung, "label": label, "judges": judges, "agree": status,
            "disagree": disagree, "roles": roles, "dir": str(rd), "run_dir": a0["run_dir"],
            "n_transcripts": len(transcript_dirs(rd)), "levels": three.index_levels(team, sa, roles),
        })
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["col"]] = counts.get(r["col"], 0) + 1
    return {"version": INDEX_VERSION, "built_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "build_seconds": round(time.time() - t0, 2), "counts": counts, "sets": SETS, "rows": rows}


def load_or_build(rebuild: bool) -> dict:
    if not rebuild and CACHE.is_file():
        try:
            with open(CACHE, encoding="utf-8") as f:
                idx = json.load(f)
            if idx.get("version") == INDEX_VERSION:
                return idx
        except (OSError, ValueError):
            pass
    idx = build_index()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, separators=(",", ":"))
    os.replace(tmp, CACHE)
    return idx


# ------------------------------------------------------------------------------------------------------- run page
def episode_agent(ep: dict) -> str | None:
    return ep.get("orbit_agent") or (ep.get("agent") or "").split("/")[-1] or None


def folder_agents(run_dir: Path, episodes: list[dict]) -> dict[str, str]:
    """agent -> trajectory folder. The folder is the episode name: <sample> alone, <sample>@<agent> in a team."""
    by_episode = {ep.get("episode"): episode_agent(ep) for ep in episodes}
    out = {}
    for name in transcript_dirs(run_dir):
        agent = by_episode.get(name)
        if agent is None and "@" in name:
            agent = name.rsplit("@", 1)[1]
        if agent is None and len(episodes) == 1:
            agent = episode_agent(episodes[0])
        if agent:
            out[agent] = name
    return out


ROUND_RE = re.compile(r"^r(\d+)")
SUBMIT_RE = re.compile(r"handed in the team's work in round (\d+)")


def board_view(run_dir: Path, timeline: list[dict], roles: dict, seeded: bool) -> list[dict]:
    """The board of each sample in list order, with each post's number, kind, round and the tool call behind it.
    A system message is the seed post when the run placed one (it is message 1); submit notices name their round."""
    cdir = run_dir / "processed" / "communication"
    if not cdir.is_dir():  # an old-layout run folder (data/acts_hub/runs) keeps communication/ at its top
        cdir = run_dir / "communication"
    if not cdir.is_dir():
        return []
    when = {(e.get("turn"), e.get("agent")): e.get("step") for e in timeline}
    boards = []
    for path in sorted(cdir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        sent, reads = {}, {}
        for e in c.get("events") or []:
            for mid in e.get("message_ids") or []:
                if e.get("kind") == "sent":
                    sent[mid] = {"agent": e.get("agent_id"), "call_id": e.get("tool_call_id")}
                elif e.get("kind") == "read":
                    reads.setdefault(mid, []).append({"agent": e.get("agent_id"), "call_id": e.get("tool_call_id")})
        msgs, n_post = [], 0
        for i, m in enumerate(c.get("messages") or []):
            sender = m.get("sender")
            content = m.get("content") or ""
            if sender == "system":
                sm = SUBMIT_RE.search(content)
                kind = "submit" if sm else ("seed" if i == 0 and seeded else "system")
                rnd = int(sm.group(1)) if sm else (0 if kind == "seed" else None)
                post_n, step = None, None
            else:
                n_post += 1
                kind, post_n = "post", n_post
                step = when.get((m.get("turn"), sender))
                rm = ROUND_RE.match(step or "")
                rnd = int(rm.group(1)) if rm else None
            msgs.append({
                "board_no": i + 1, "post_n": post_n, "kind": kind, "sender": sender, "role": roles.get(sender),
                "recipients": m.get("recipients") or [], "content": content, "sequence": m.get("sequence"),
                "turn": m.get("turn"), "round": rnd, "round_step": step, "id": m.get("id"),
                "sent": sent.get(m.get("id")), "reads": reads.get(m.get("id"), []),
            })
        kinds: dict[str, int] = {}
        for e in c.get("events") or []:
            kinds[e.get("kind") or "?"] = kinds.get(e.get("kind") or "?", 0) + 1
        boards.append({"sample": path.stem, "messages": msgs, "event_counts": kinds})
    return boards


def run_detail(row: dict, api: dict, agents_csv: dict, steps: dict) -> dict:
    run_dir = Path(row["dir"])
    with open(run_dir / "processed" / "run.json", encoding="utf-8") as f:
        rj = json.load(f)
    rules = rj.get("rules") or {}
    timeline = rules.get("timeline") or []
    episodes = list(jsonl(run_dir / "processed" / "episodes.jsonl"))
    team, sig_agents = signals(run_dir)
    roles = roles_of(sig_agents, rj)
    for ep in episodes:
        if ep.get("role"):
            roles[episode_agent(ep)] = ep["role"]
    folders = folder_agents(run_dir, episodes)
    labelled = {r["agent"]: r for r in agents_csv.get(row["id"], [])}

    agents = []
    for ep in episodes:
        a = episode_agent(ep)
        folder = folders.get(a)
        tpath = run_dir / "processed" / "trajectories" / folder / "transcript.json" if folder else None
        lab = labelled.get(a)
        entry = {
            "agent": a, "role": roles.get(a) or "worker", "folder": folder,
            "transcript_bytes": tpath.stat().st_size if tpath else None,
            "episode": {k: ep.get(k) for k in EPISODE_KEYS if k in ep},
            "signals": sig_agents.get(a),
            "labels": {k: (lab.get(k) or None) for k in LABEL_KEYS} if lab else None,
            "steps": steps.get(lab["uid"]) if lab else None,
            "judges": {},
        }
        if row["col"] == "controls_600":
            for key, _ in API_JUDGES:
                j = api[key].get((row["run"], a))
                if j is not None:
                    entry["judges"][key] = j
        agents.append(entry)
    # Workers first, regulators last, as SGA orders its lanes; then by agent id.
    agents.sort(key=lambda e: (e["role"] in ("regulator", "disregulator"), e["agent"] or ""))

    meta = {k: rj.get(k) for k in RUN_JSON_KEYS if k in rj}
    meta["config"] = rj.get("config")
    meta["rounds"] = rules.get("rounds")
    meta["workers"] = rules.get("workers")
    meta["composition"] = (rj.get("swarm") or {}).get("composition")
    meta["run_dir"] = row["run_dir"]
    boards = board_view(run_dir, timeline, roles, bool((rj.get("config") or {}).get("board_seed")))
    return {"row": row, "meta": meta, "team": team, "agents": agents, "boards": boards,
            "levels": three.compute(team, sig_agents, roles, three.read_labels(run_dir))}


# ---------------------------------------------------------------------------------------------------------- server
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml"}


class Viewer:
    def __init__(self, idx: dict):
        self.index_bytes = json.dumps(idx, separators=(",", ":")).encode("utf-8")
        self.rows = {r["id"]: r for r in idx["rows"]}
        self.api = load_api_judges()
        self.agents = load_agents()
        self.steps = load_steps()


class Handler(BaseHTTPRequestHandler):
    viewer: Viewer = None  # set in main()
    server_version = "SGAViewer/1"

    def log_message(self, fmt, *args):  # quiet: one line per error only
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def send_bytes(self, body: bytes, ctype: str, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=HTTPStatus.OK):
        self.send_bytes(json.dumps(obj, separators=(",", ":")).encode("utf-8"), "application/json", status)

    def fail(self, status, msg):
        self.send_json({"error": msg}, status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self.static("index.html")
            if url.path.startswith("/static/"):
                return self.static(url.path[len("/static/"):])
            if url.path == "/api/index":
                return self.send_bytes(self.viewer.index_bytes, "application/json")
            if url.path == "/api/run":
                row = self.viewer.rows.get(q.get("id", ""))
                if row is None:
                    return self.fail(HTTPStatus.NOT_FOUND, f"no run {q.get('id')!r} in the index")
                return self.send_json(run_detail(row, self.viewer.api, self.viewer.agents, self.viewer.steps))
            if url.path == "/api/lenses":
                ldir = DATA / "lens"
                names = sorted(p.stem for p in ldir.glob("*.json")) if ldir.is_dir() else []
                return self.send_json({"lenses": names})
            if url.path == "/api/lens":
                name = q.get("name", "")
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not (DATA / "lens" / f"{name}.json").is_file():
                    return self.fail(HTTPStatus.NOT_FOUND, f"no lens {name!r}")
                return self.send_bytes((DATA / "lens" / f"{name}.json").read_bytes(), "application/json")
            if url.path == "/api/ladders":
                import ladders_data as ld
                return self.send_json({"domains": ld.DOMAINS, "rung_notes": ld.RUNG_NOTES,
                                       "evidence": ld.RUNGS_BY_DOMAIN, "task_column": ld.TASK_COLUMN,
                                       "ladders": three.LADDERS, "order": three.ORDER})
            if url.path == "/api/transcript":
                return self.transcript(q.get("id", ""), q.get("agent", ""))
            return self.fail(HTTPStatus.NOT_FOUND, "no such path")
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:  # noqa: BLE001 - shown to the local user as an error box
            return self.fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    def static(self, rel: str):
        path = (STATIC / rel).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            return self.fail(HTTPStatus.NOT_FOUND, "no such file")
        self.send_bytes(path.read_bytes(), TYPES.get(path.suffix, "application/octet-stream"))

    def transcript(self, run_id: str, agent: str):
        row = self.viewer.rows.get(run_id)
        if row is None:
            return self.fail(HTTPStatus.NOT_FOUND, f"no run {run_id!r} in the index")
        run_dir = Path(row["dir"])
        folder = folder_agents(run_dir, list(jsonl(run_dir / "processed" / "episodes.jsonl"))).get(agent)
        if folder is None:
            return self.fail(HTTPStatus.NOT_FOUND, f"no transcript for agent {agent!r} in {run_id}")
        path = run_dir / "processed" / "trajectories" / folder / "transcript.json"
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 256 * 1024)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rebuild", action="store_true", help="rebuild viewer/.cache/index.json from the data")
    args = ap.parse_args(argv)
    t0 = time.time()
    idx = load_or_build(args.rebuild)
    took = time.time() - t0
    counts = ", ".join(f"{k} {v}" for k, v in idx["counts"].items())
    print(f"index: {len(idx['rows'])} runs ({counts}); built {idx['built_at']} in {idx['build_seconds']} s; "
          f"loaded in {took:.2f} s", flush=True)
    Handler.viewer = Viewer(idx)
    # The page loads a dozen ES modules at once; the default listen backlog of 5 resets some of them.
    ThreadingHTTPServer.request_queue_size = 128
    srv = ThreadingHTTPServer((args.host, args.port), Handler)

    def stop(signum, frame):  # a background start ignores Ctrl-C, so SIGTERM stops the server cleanly too
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f"http://{args.host}:{args.port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
