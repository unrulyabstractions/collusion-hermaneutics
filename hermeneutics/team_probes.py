"""Hermeneutic probes on the team runs (board seed) and the sets that follow, trained on this machine from the Hub.

No model calls and no spend: the reader's activations come from the public Hub dataset unrulyabstractions/apart-swarm-align
(activations/<set>/<run>/<agent>/), the labels from the run dirs under runs/ through viewer/levels.py.

Stages:
    python -m hermeneutics.team_probes fetch [--sets board_seed,controls_600]   think tensors to data/acts_hub/think
    python -m hermeneutics.team_probes labels                                    agent, team and per-step labels
    python -m hermeneutics.team_probes run                                       every probe, tables, RESULTS.md
    python -m hermeneutics.team_probes watch                                     poll the Hub, fetch, re-run, status

Call index k in the ladders is the 0-based index over the agent's individual tool calls (signals.jsonl exposed_call,
tried_call, opened_call; checked by hand on pilot60_none_s0). An assistant step can hold several tool calls, so k maps
to the step that holds it; acts turn t is assistant step t (spans.jsonl call field).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
HUB = REPO / "data" / "acts_hub"
THINK = HUB / "think"
OUT = REPO / "hermeneutics" / "results" / "team_probes"
REPO_ID = "unrulyabstractions/apart-swarm-align"
SMALL = ["meta.json", "spans.jsonl", "rendered.txt", "source.json"]
LAYERS = [26, 38, 51]
SET_ORDER = ["board_seed", "controls_600", "spar_mixed_41", "spar_screen_team", "solo_design"]
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:.0f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ Hub listing, fetch
def hub_listing():
    """{set: {(run, agent): set(file names)}} for every activations dir on the Hub, and the full file list."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    out = defaultdict(lambda: defaultdict(set))
    for f in files:
        p = f.split("/")
        if p[0] == "activations" and len(p) == 5:
            out[p[1]][(p[2], p[3])].add(p[4])
    return out, files


def think_path(s, run, agent):
    return THINK / s / run / f"{agent}.npz"


def fetch_agent(s, run, agent):
    """Small files kept under data/acts_hub/activations; acts.safetensors read once for the think tensors, saved as
    fp16 npz, then deleted. Returns a dict for the fetch log."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    base = f"activations/{s}/{run}/{agent}"
    from huggingface_hub.errors import EntryNotFoundError
    for n in SMALL:
        if not (HUB / base / n).is_file():
            try:
                hf_hub_download(REPO_ID, f"{base}/{n}", repo_type="dataset", local_dir=str(HUB))
            except EntryNotFoundError:
                if n == "meta.json":
                    raise
    meta = json.load(open(HUB / base / "meta.json"))
    p = hf_hub_download(REPO_ID, f"{base}/acts.safetensors", repo_type="dataset", local_dir=str(HUB))
    arr = {}
    with safe_open(p, "np") as f:
        keys = set(f.keys())
        for L in LAYERS:
            arr[f"think_L{L}"] = f.get_tensor(f"think_L{L}")
        arr["think_pos"] = f.get_tensor("think_pos")
        arr["think_readable"] = f.get_tensor("think_readable") if "think_readable" in keys else np.ones(len(arr["think_pos"]), np.int8)
        ids = f.get_tensor("ids")
    tp = arr["think_pos"]
    tid = meta.get("think_ids") or [None, None]
    ok_open = bool((ids[tp[:, 0]] == tid[0]).all()) if len(tp) else True
    ok_close = bool((ids[tp[:, 1]] == tid[1]).all()) if len(tp) else True
    n = arr["think_L26"].shape[0]
    rec = {"set": s, "run": run, "agent": agent, "n_turns": int(n), "meta_n_turns": meta.get("n_turns"),
           "think_open_token_ok": ok_open, "think_close_token_ok": ok_close, "acts_bytes": os.path.getsize(p)}
    dst = think_path(s, run, agent)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, dst)
    os.remove(p)
    return rec


def fetch_turns(s, run, agent):
    """turns.safetensors (the five turn-boundary tokens of each turn) to data/acts_hub/boundary as fp16 npz, with meta.json
    fetched again for its turn_positions and turn_ids; checks the boundary positions sit before the turn's think-open."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    base = f"activations/{s}/{run}/{agent}"
    hf_hub_download(REPO_ID, f"{base}/meta.json", repo_type="dataset", local_dir=str(HUB), force_download=True)
    meta = json.load(open(HUB / base / "meta.json"))
    p = hf_hub_download(REPO_ID, f"{base}/turns.safetensors", repo_type="dataset", local_dir=str(HUB))
    arr = {}
    with safe_open(p, "np") as f:
        keys = set(f.keys())
        for L in LAYERS:
            arr[f"turn_L{L}"] = f.get_tensor(f"turn_L{L}")
        arr["turn_pos"] = f.get_tensor("turn_pos") if "turn_pos" in keys else np.zeros((0, 5), np.int32)
    tz = np.load(think_path(s, run, agent))
    tp = tz["think_pos"]
    bp = arr["turn_pos"]
    n_ok = arr["turn_L26"].shape[0] == tp.shape[0] and arr["turn_L26"].shape[1] == 5
    order_ok = bool(n_ok and len(bp) == len(tp) and (np.diff(bp, axis=1) > 0).all() and (bp[:, -1] < tp[:, 0]).all())
    gap = (tp[:, 0] - bp[:, -1]).tolist() if n_ok and len(bp) == len(tp) else []
    arr["turn_positions"] = np.array([str(x) for x in (meta.get("turn_positions") or [])])
    dst = boundary_path(s, run, agent)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, dst)
    os.remove(p)
    return {"set": s, "run": run, "agent": agent, "kind": "turns", "n_turns": int(arr["turn_L26"].shape[0]),
            "shape_ok": bool(n_ok), "positions_before_think_open": order_ok,
            "gap_think_open_minus_last_boundary": sorted(set(gap))[:5], "turn_positions": meta.get("turn_positions"),
            "turn_ids": meta.get("turn_ids")}


def fetch(sets, listing=None, jobs=4, timeout=300):
    """Each agent in its own subprocess with a time limit (a stalled transfer is killed and retried), jobs at once."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    if listing is None:
        listing, _ = hub_listing()
    logp = OUT / "fetch_log.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    todo = []
    for s in sets:
        for (run, agent), names in sorted(listing.get(s, {}).items()):
            if "acts.safetensors" not in names or "meta.json" not in names:
                continue
            nt = not think_path(s, run, agent).is_file()
            nb = "turns.safetensors" in names and not boundary_path(s, run, agent).is_file()
            if nt or nb:
                todo.append((s, run, agent, nt, nb))
    log(f"fetch: {sum(t[3] for t in todo)} think and {sum(t[4] for t in todo)} boundary files to fetch in {sets}")

    def sub(stage, s, run, agent, target):
        for attempt in range(3):
            try:
                r = subprocess.run([sys.executable, "-m", "hermeneutics.team_probes", stage, s, run, agent],
                                   cwd=str(REPO), capture_output=True, text=True, timeout=timeout)
                if r.returncode == 0 and target.is_file():
                    return json.loads(r.stdout.strip().splitlines()[-1])
                log(f"{stage} {s}/{run}/{agent} failed ({attempt}): {r.stderr.strip()[-300:]}")
            except subprocess.TimeoutExpired:
                log(f"{stage} {s}/{run}/{agent} timed out ({attempt})")
            for inc in (HUB / ".cache").rglob("*.incomplete"):
                if f"/{run}/{agent}/" in str(inc):
                    inc.unlink(missing_ok=True)
        return None

    def one(job):
        s, run, agent, nt, nb = job
        recs = []
        if nt:
            recs.append(sub("fetch-one", s, run, agent, think_path(s, run, agent)))
        if nb and think_path(s, run, agent).is_file():
            recs.append(sub("fetch-turns", s, run, agent, boundary_path(s, run, agent)))
        return [r for r in recs if r is not None]
    got = 0
    with ThreadPoolExecutor(jobs) as ex:
        for recs in ex.map(one, todo):
            for rec in recs:
                rec["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(logp, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            got += bool(recs)
            if got % 10 == 0:
                log(f"fetched {got} of {len(todo)} ({rec['set']} {rec['run']}/{rec['agent']})")
    log(f"fetch done: {got} agents updated")
    return got


def local_counts():
    return {s.name: sum(1 for _ in s.glob("*/*.npz")) for s in sorted(THINK.glob("*")) if s.is_dir()} if THINK.is_dir() else {}


def boundary_counts():
    B = HUB / "boundary"
    return {s.name: sum(1 for _ in s.glob("*/*.npz")) for s in sorted(B.glob("*")) if s.is_dir()} if B.is_dir() else {}


# ------------------------------------------------------------------------------------------------ labels
RUN_COLLECTIONS = ["pilot_tb25", "pilot_tb60", "dose_response_tb60"]
RUNGS = ["none", "exposed", "tried", "opened", "used", "tampered", "checked", "shipped"]
USED, OPENED, CROSSED, POSTED, DISCUSSED = 4, 3, 3, 2, 3


def _viewer():
    sys.path.insert(0, str(REPO / "viewer"))
    import levels as six  # noqa: E402
    import serve  # noqa: E402
    return six, serve


def agent_calls(source_path):
    """The agent's individual tool calls in order, [(step, tool_call_id, arguments, result text)], and its step count."""
    t = json.load(open(source_path))["trajectory"]
    steps = [s for s in t["steps"] if s.get("source") == "agent"]
    calls = []
    for si, x in enumerate(steps):
        obs = {o.get("source_call_id"): o.get("content") for o in ((x.get("observation") or {}).get("results") or [])}
        for c in x.get("tool_calls") or []:
            r = obs.get(c.get("tool_call_id"))
            calls.append((si, c.get("tool_call_id"), c.get("arguments") or {}, r if isinstance(r, str) else json.dumps(r)))
    return calls, len(steps)


def used_write_call(calls, k_open):
    """Estimate of the used onset: the first tool call at or after the opened call whose arguments contain an added
    line (a '+' diff line of 20 or more characters) of the opened call's result, the first write of planted content."""
    if k_open is None or k_open >= len(calls):
        return None
    plus = [ln[1:].strip() for ln in calls[k_open][3].split("\n")
            if ln.startswith("+") and not ln.startswith("+++") and len(ln[1:].strip()) >= 20]
    if not plus:
        return None
    for j in range(k_open, len(calls)):
        txt = " ".join(str(v) for v in calls[j][2].values())
        if any(p in txt for p in plus):
            return j
    return None


def labels():
    """Agent and team labels of the board-seed runs from viewer/levels.py, with the step of each per-call event."""
    import pandas as pd
    six, serve = _viewer()
    arows, trows, checks = [], [], []
    for coll in RUN_COLLECTIONS:
        for rd in sorted((HUB / "runs" / coll).glob("*")):
            rj = json.load(open(rd / "run.json"))
            episodes = list(serve.jsonl(rd / "episodes.jsonl"))
            sig = list(serve.jsonl(rd / "signals.jsonl"))
            team = next((s for s in sig if s.get("row") == "team"), None)
            sa = {s.get("agent"): s for s in sig if s.get("row") == "agent"}
            roles = dict((rj.get("swarm") or {}).get("roles") or {})
            for e in episodes:
                if e.get("role"):
                    roles[e.get("orbit_agent")] = e["role"]
            for e in episodes:
                roles.setdefault(e.get("orbit_agent"), "worker")
            boards = serve.board_view(rd, (rj.get("rules") or {}).get("timeline") or [], roles,
                                      bool((rj.get("config") or {}).get("board_seed")))
            r = six.compute(team, sa, episodes, boards, roles)
            budget = (rj.get("turn_budget") or {}).get("budget")
            arm = seed = None
            for a, c in r["agents"].items():
                adir = HUB / "activations" / "board_seed" / rd.name / a
                meta = json.load(open(adir / "meta.json")) if (adir / "meta.json").is_file() else {}
                arm, seed = meta.get("condition", arm), meta.get("seed", seed)
                s = sa.get(a) or {}
                calls, n_steps = agent_calls(adir / "source.json") if (adir / "source.json").is_file() else ([], None)
                id2step = {cid: st for st, cid, _, _ in calls}

                def step_of(k):
                    return calls[k][0] if k is not None and 0 <= k < len(calls) else None
                ev = r["events"][a]
                first = {}
                for e in ev:
                    if e["ladder"] == "coord" and e.get("call_id") and e["level"] not in first:
                        first[e["level"]] = id2step.get(e["call_id"])
                ci, ui, li = c["cheat"]["index"], c["coord"]["index"], c["collude"]["index"]
                opened_step = step_of(s.get("opened_call"))
                kw = used_write_call(calls, s.get("opened_call")) if ci >= USED else None
                used_write_step = step_of(kw)
                used_onset = used_write_step if used_write_step is not None else (opened_step if ci >= USED else None)
                board_step = min([x for x in (first.get("read the board"), first.get("posted")) if x is not None], default=None)
                coll_onset = None
                if li >= CROSSED:
                    base = used_onset if ci >= USED else (first.get("posted") if first.get("posted") is not None else board_step)
                    cands = [x for x in (base, board_step) if x is not None]
                    coll_onset = max(cands) if cands else None
                arows.append({"run": rd.name, "collection": coll, "agent": a, "role": roles.get(a) or "worker",
                              "arm": meta.get("condition"), "seed": meta.get("seed"), "budget": budget,
                              "n_turns": meta.get("n_turns"), "n_steps_source": n_steps, "n_calls_source": len(calls),
                              "n_calls_signals": s.get("calls"),
                              "cheat": ci, "cheat_level": c["cheat"]["level"], "coord": ui, "coord_level": c["coord"]["level"],
                              "coop": c["coop"]["index"], "collab": c["collab"]["index"], "collude": li,
                              "collude_level": c["collude"]["level"], "scheme": c["scheme"]["index"],
                              "exposed_step": step_of(s.get("exposed_call")), "tried_step": step_of(s.get("tried_call")),
                              "opened_call": s.get("opened_call"), "opened_step": opened_step,
                              "used_write_call": kw, "used_write_step": used_write_step, "used_onset_step": used_onset,
                              "used_onset_source": ("write" if used_write_step is not None else "opened" if ci >= USED else ""),
                              "first_read_step": first.get("read the board"), "first_post_step": first.get("posted"),
                              "collude_onset_step": coll_onset})
                checks.append((rd.name, a, meta.get("n_turns"), n_steps, len(calls), s.get("calls")))
            t = r["team"]
            trows.append({"run": rd.name, "collection": coll, "arm": arm, "seed": seed, "budget": budget,
                          "team_cheat": t["cheat"]["index"], "collaboration": (team or {}).get("collaboration"),
                          "floor": int(t["cheat"]["index"] >= USED and (team or {}).get("collaboration") in ("crossed", "assembled")),
                          "team_collude": t["collude"]["index"],
                          "n_workers": sum(1 for x in r["agents"] if (roles.get(x) or "worker") != "regulator")})
    A, T = pd.DataFrame(arows), pd.DataFrame(trows)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    A.to_csv(OUT / "labels" / "agents.csv", index=False)
    T.to_csv(OUT / "labels" / "teams.csv", index=False)
    bad = [x for x in checks if not (x[2] == x[3] and x[4] == x[5])]
    log(f"labels: {len(A)} agents ({(A.role != 'regulator').sum()} workers), {len(T)} runs, floor {T.floor.sum()}; "
        f"nesting violations {len(six.NESTING_VIOLATIONS)}; step/call count mismatches {len(bad)} {bad[:5]}")
    return A, T


# ------------------------------------------------------------------------------------------------ features and text
POS = {"open": [0], "close": [1], "think": [0, 1]}
BPOS = {f"b{i}": [i] for i in range(5)} | {"bcat": [0, 1, 2, 3, 4]}
# Setting A, the think pair (think-open and think-close concatenated, layers concatenated) on every agent, with each
# token and layer alone as sub-rows. Setting B, all seven tokens (the five turn-boundary tokens of turns.safetensors and
# the think pair) on the agents that already have turns.safetensors, with each token alone as sub-rows. Same folds.
CONFIGS = [("all", "think")] + [(L, p) for L in ("all", 26, 38, 51) for p in ("open", "close")]
MAIN = [("all", "think"), ("all", "open"), ("all", "close")]
BCONFIGS = ([("all", "both"), ("all", "think"), ("all", "open"), ("all", "close")] + [("all", f"b{i}") for i in range(5)]
            + [("all", "bcat")])
BNULL_STEP = [("all", "both"), ("all", "think")]
SETTING = {"A": "A: think pair, all agents", "B": "B: all seven tokens, agents with turns.safetensors"}
BNAMES = ["boundary 1 (previous <|im_end|>)", "boundary 2 (newline after it)", "boundary 3 (<|im_start|>)",
          "boundary 4 (assistant)", "boundary 5 (newline after assistant)"]


def fname(cfg):
    L, p = cfg
    if p in ("open", "close"):
        return f"probe L{L} think-{p}"
    if p == "think":
        return f"probe L{L} think-open+close"
    if p == "bcat":
        return f"probe L{L} boundary 1-5 concatenated"
    if p == "both":
        return f"probe L{L} all seven tokens (boundary 1-5 + think pair)"
    return f"probe L{L} {BNAMES[int(p[1])]}"


def is_b(cfg):
    return cfg[1] in BPOS or cfg[1] == "both"


def boundary_path(s, run, agent):
    return HUB / "boundary" / s / run / f"{agent}.npz"


EMPTY = set()  # (set, run, agent) left out for having zero turns


def load_think(s, rows):
    """{(run, agent): {L: think [n_turns, 2, 5120], ("b", L): boundary [n_turns, 5, 5120]}} (float16) for the rows
    with a local npz; the boundary entries only when that agent's turns.safetensors has been fetched."""
    out = {}
    for r in rows.itertuples():
        p = think_path(s, r.run, r.agent)
        if p.is_file():
            z = np.load(p)
            if z["think_L26"].shape[0] == 0:  # an agent with no assistant turn has no features
                EMPTY.add((s, r.run, r.agent))
                continue
            d = {L: z[f"think_L{L}"] for L in LAYERS}
            bp = boundary_path(s, r.run, r.agent)
            if bp.is_file():
                zb = np.load(bp)
                if zb["turn_L26"].shape[0] == d[26].shape[0] and zb["turn_L26"].shape[1] == 5:
                    d.update({("b", L): zb[f"turn_L{L}"] for L in LAYERS})
            out[(r.run, r.agent)] = d
    return out


def has_b(acts, keys):
    return np.array([("b", 26) in acts[k] for k in keys])


def _rows(a, L, pos):
    n = a[L].shape[0]
    if pos in POS:
        return a[L][:, POS[pos], :].reshape(n, -1)
    if pos in BPOS:
        return a[("b", L)][:, BPOS[pos], :].reshape(n, -1)
    return np.concatenate([a[L][:, [0, 1], :].reshape(n, -1), a[("b", L)].reshape(n, -1)], 1)  # both


def _width(pos):
    m = len(POS[pos]) if pos in POS else len(BPOS[pos]) if pos in BPOS else 7
    return m * 5120


def turn_matrix(acts, keys, layer, pos):
    """Rows: every turn of the agents in keys, float32 (preallocated). An agent without the boundary tensors gets zero
    rows for a boundary setting; callers mask those agents out. Also the agent index and turn index of each row."""
    Ls = LAYERS if layer == "all" else [int(layer)]
    ns = [acts[k][26].shape[0] for k in keys]
    X = np.zeros((sum(ns), _width(pos) * len(Ls)), np.float32)
    o = 0
    for k, n in zip(keys, ns):
        a = acts[k]
        if pos in POS or ("b", 26) in a:
            c = 0
            for L in Ls:
                blk = _rows(a, L, pos)
                X[o:o + n, c:c + blk.shape[1]] = blk
                c += blk.shape[1]
        o += n
    ai = np.concatenate([np.full(n, i) for i, n in enumerate(ns)])
    ti = np.concatenate([np.arange(n) for n in ns])
    return X, ai, ti


def agent_matrix(acts, keys, layer, pos):
    """Mean over the agent's turns of its turn rows (zeros for an agent without the boundary tensors)."""
    Ls = LAYERS if layer == "all" else [int(layer)]
    out = np.zeros((len(keys), _width(pos) * len(Ls)), np.float64)
    for i, k in enumerate(keys):
        a = acts[k]
        if pos in POS or ("b", 26) in a:
            out[i] = np.concatenate([_rows(a, L, pos).astype(np.float64).mean(0) for L in Ls])
    return out


def turn_texts(s, run, agent, n_turns):
    """Per turn t: the agent's own words before its turn-t tool call (turn t-1's tool calls and turn t's reasoning) and
    the rendered text between the previous turn's cut and this turn's (the reader's whole input up to its turn-t
    tool call). Cut = the first character of turn t's tool-call span."""
    base = HUB / "activations" / s / run / agent
    rendered = open(base / "rendered.txt", encoding="utf-8").read()
    cot, tc, cut = defaultdict(list), defaultdict(list), {}
    for line in open(base / "spans.jsonl", encoding="utf-8"):
        sp = json.loads(line)
        k, seg = sp["call"], sp["segment"]
        if seg == "cot":
            cot[k].append(sp["text"])
        elif seg == "tool_call":
            tc[k].append(sp["text"])
            cut[k] = min(cut.get(k, 10**12), sp["chars"][0])
    words, whole, prev = [], [], 0
    for t in range(n_turns):
        words.append(" ".join(tc.get(t - 1, []) + cot.get(t, [])))
        c = cut.get(t, prev)
        whole.append(rendered[prev:c])
        prev = max(prev, c)
    return words, whole


def agent_texts(s, run, agent):
    base = HUB / "activations" / s / run / agent
    words = []
    for line in open(base / "spans.jsonl", encoding="utf-8"):
        sp = json.loads(line)
        if sp["segment"] in ("cot", "tool_call"):
            words.append(sp["text"])
    return " ".join(words), open(base / "rendered.txt", encoding="utf-8").read()


# ------------------------------------------------------------------------------------------------ probe machinery
K = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
CS_TEXT = [1e-2, 1e-1, 1.0, 10.0, 100.0]
N_BOOT = 1000
N_SHUF = 100


def auc(s, y, w=None):
    """Weighted AUROC (Mann-Whitney with ties counted half), equal to sklearn's roc_auc_score; rows with a non-finite
    score or zero weight are left out."""
    s, y = np.asarray(s, float), np.asarray(y).astype(int)
    ok = np.isfinite(s)
    if w is not None:
        ok &= np.asarray(w) > 0
    s, y = s[ok], y[ok]
    w = np.ones(len(s)) if w is None else np.asarray(w, float)[ok]
    if len(s) == 0 or y.min() == y.max():
        return np.nan
    _, g = np.unique(s, return_inverse=True)
    P = np.bincount(g, w * (y == 1))
    N = np.bincount(g, w * (y == 0))
    below = np.cumsum(N) - N
    return float((P * (below + 0.5 * N)).sum() / (P.sum() * N.sum()))


def boot_ci(score, y, runs, B=N_BOOT, seed=0):
    """95% interval of the AUROC over runs resampled with replacement (each row weighted by its run's draw count)."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(score)
    score, y, runs = np.asarray(score)[ok], np.asarray(y)[ok], np.asarray(runs)[ok]
    ur, inv = np.unique(runs, return_inverse=True)
    vals = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ur), len(ur)), minlength=len(ur))[inv].astype(float)
        a = auc(score, y, w)
        if np.isfinite(a):
            vals.append(a)
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [np.nan, np.nan]


def onehot(vals):
    cats = sorted(set(map(str, vals)))
    M = np.zeros((len(vals), len(cats)))
    M[np.arange(len(vals)), [cats.index(str(v)) for v in vals]] = 1
    return M


def folds_for(y_strat, groups, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    f = np.zeros(len(y_strat), int)
    for i, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=seed).split(np.zeros(len(y_strat)), y_strat, groups)):
        f[te] = i
    return f


def pca_z(Xtr, Xte, kmax=128):
    """Training-fold PCA to min(kmax, half the training rows) components, then z-scoring with training statistics."""
    k = min(kmax, max(2, len(Xtr) // 2))
    c = Xtr.mean(0)
    Xc = Xtr
    Xc -= c  # in place: callers pass a fresh copy (X[tr])
    if len(Xtr) <= 1500:
        G = Xc @ Xc.T
        w, U = np.linalg.eigh(G.astype(np.float64))
        w, U = w[::-1][:k], U[:, ::-1][:, :k]
        V = (Xc.T @ U) / np.sqrt(np.maximum(w, 1e-12))
    else:
        from sklearn.utils.extmath import randomized_svd
        _, _, Vt = randomized_svd(Xc, k, n_iter=4, random_state=0)
        V = Vt.T
    Ptr = Xc @ V
    sd = Ptr.std(0) + 1e-9
    return (Ptr / sd).astype(np.float64), (((Xte - c) @ V) / sd).astype(np.float64)


def choose_C(Z, y, groups, Cs=CS):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    try:
        sp = list(StratifiedGroupKFold(3, shuffle=True, random_state=1).split(np.zeros(len(y)), y, groups))
    except ValueError:
        return Cs[len(Cs) // 2]
    best, bc = -1.0, Cs[len(Cs) // 2]
    for C in Cs:
        vals = []
        for tr, te in sp:
            if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
                continue
            lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Z[tr], y[tr])
            vals.append(auc(lr.decision_function(Z[te]), y[te]))
        if vals and np.nanmean(vals) > best:
            best, bc = float(np.nanmean(vals)), C
    return bc


def fit_score(Ztr, ytr, gtr, Zte, C=None, Cs=CS):
    from sklearn.linear_model import LogisticRegression
    if len(set(ytr.tolist())) < 2:
        return None, None
    if C is None:
        C = choose_C(Ztr, ytr, gtr, Cs)
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ztr, ytr)
    return lr.decision_function(Zte), C


def prep_folds(X, fold, rows=None):
    """Per fold: (train index, test index, PCA-z train, PCA-z test), restricted to rows (a mask) when given."""
    rows = np.ones(len(X), bool) if rows is None else rows
    P = []
    for f in range(K):
        tr, te = np.flatnonzero(rows & (fold != f)), np.flatnonzero(rows & (fold == f))
        if len(tr) < 6 or not len(te):
            P.append(None)
            continue
        Ztr, Zte = pca_z(X[tr], X[te])
        P.append((tr, te, Ztr, Zte))
    return P


def _blocks(layer, pos):
    """The (layer, token kind, token index) blocks of 5120 columns that make up a feature setting, in turn_matrix order."""
    Ls = LAYERS if layer == "all" else [int(layer)]
    toks = ([("t", i) for i in POS[pos]] if pos in POS else [("b", i) for i in BPOS[pos]] if pos in BPOS
            else [("t", 0), ("t", 1)] + [("b", i) for i in range(5)])
    return [(L, k, i) for L in Ls for k, i in toks]


def _block_rows(acts, keys, L, kind, i):
    """float32 [all turns, 5120] for one token at one layer; zero rows for an agent without the boundary tensors."""
    ns = [acts[k][26].shape[0] for k in keys]
    out = np.zeros((sum(ns), 5120), np.float32)
    o = 0
    for k, n in zip(keys, ns):
        a = acts[k]
        if kind == "t":
            out[o:o + n] = a[L][:, i, :]
        elif ("b", L) in a:
            out[o:o + n] = a[("b", L)][:, i, :]
        o += n
    return out


def prep_folds_stream(acts, keys, layer, pos, fold, rows=None, kmax=128):
    """prep_folds without the full turn matrix: per fold, the training Gram matrix and the test-by-training cross-Gram
    are summed over 5120-column blocks centred with training means, then the exact top-k PCA comes from the Gram's
    eigenvectors (training scores U S, test scores Gte U / S). Peak memory is one block and the two Gram matrices."""
    from scipy.linalg import eigh
    n = sum(acts[k][26].shape[0] for k in keys)
    rows = np.ones(n, bool) if rows is None else rows
    P = []
    for f in range(K):
        tr, te = np.flatnonzero(rows & (fold != f)), np.flatnonzero(rows & (fold == f))
        if len(tr) < 6 or not len(te):
            P.append(None)
            continue
        G = np.zeros((len(tr), len(tr)), np.float64)
        Gte = np.zeros((len(te), len(tr)), np.float64)
        for L, kind, i in _blocks(layer, pos):
            B = _block_rows(acts, keys, L, kind, i)
            Btr = B[tr]
            mu = Btr.mean(0)
            Btr -= mu
            Bte = B[te] - mu
            del B
            G += Btr @ Btr.T
            Gte += Bte @ Btr.T
            del Btr, Bte
        k = min(kmax, max(2, len(tr) // 2))
        w, U = eigh(G, subset_by_index=[len(tr) - k, len(tr) - 1])
        w, U = w[::-1], U[:, ::-1]
        S = np.sqrt(np.maximum(w, 1e-12))
        Ptr, Pte = U * S, (Gte @ U) / S
        sd = Ptr.std(0) + 1e-9
        P.append((tr, te, Ptr / sd, Pte / sd))
        del G, Gte
    return P


def cv_scores(P, y, groups, n, C=None):
    s = np.full(n, np.nan)
    Cs_used = []
    for v in P:
        if v is None:
            continue
        tr, te, Ztr, Zte = v
        sc, Cu = fit_score(Ztr, y[tr], groups[tr], Zte, C=C)
        if sc is not None:
            s[te] = sc
            Cs_used.append(Cu)
    return s, Cs_used


def null_aucs(P, perm_labels, groups, n, C, agg=None, agg_y=None):
    """AUROC of the probe refitted on each permuted label vector with C fixed (the median of the chosen Cs).
    agg(s) maps row scores to the scored unit; agg_y(yp) gives the unit labels of that permutation."""
    out = []
    for yp in perm_labels:
        s, _ = cv_scores(P, yp, groups, n, C=C)
        if agg is not None:
            out.append(auc(agg(s), agg_y(yp)))
        else:
            out.append(auc(s, yp))
    return np.array(out, float)


def perm_within(arms, y, B=N_SHUF, seed=1000):
    rng = np.random.default_rng(seed)
    out = []
    for b in range(B):
        yp = y.copy()
        for a in np.unique(arms):
            ii = np.flatnonzero(arms == a)
            yp[ii] = y[ii][rng.permutation(len(ii))]
        out.append(yp)
    return out


def meta_matrix(df):
    return np.column_stack([onehot(df.arm.tolist()), onehot(df.budget.tolist()), df.seed.astype(float).values])


def cv_dense(F, y, groups, fold, rows=None, Cs=(1.0,), standardize=True):
    """Logistic regression on a small feature matrix (metadata, turn position) on the same folds."""
    from sklearn.linear_model import LogisticRegression
    rows = np.ones(len(y), bool) if rows is None else rows
    s = np.full(len(y), np.nan)
    for f in range(K):
        tr, te = np.flatnonzero(rows & (fold != f)), np.flatnonzero(rows & (fold == f))
        if len(set(y[tr].tolist())) < 2 or not len(te):
            continue
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        Ftr, Fte = ((F[tr] - mu) / sd, (F[te] - mu) / sd) if standardize else (F[tr], F[te])
        s[te] = LogisticRegression(C=Cs[0], max_iter=5000, class_weight="balanced").fit(Ftr, y[tr]).decision_function(Fte)
    return s


def cv_tfidf_docs(docs, y, groups, fold, rows=None):
    """TF-IDF (word unigrams and bigrams, fitted on the training fold) then L2 logistic regression with C by inner CV."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    rows = np.ones(len(y), bool) if rows is None else rows
    s = np.full(len(y), np.nan)
    for f in range(K):
        tr, te = np.flatnonzero(rows & (fold != f)), np.flatnonzero(rows & (fold == f))
        if len(set(y[tr].tolist())) < 2 or not len(te):
            continue
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, ngram_range=(1, 2))
        Xtr = vec.fit_transform([docs[i] for i in tr])
        Xte = vec.transform([docs[i] for i in te])
        sc, _ = fit_score(Xtr, y[tr], groups[tr], Xte, Cs=CS_TEXT)
        s[te] = sc
    return s


def cv_tfidf_cumulative(incr, agent_idx, y, groups, fold, rows=None):
    """Per-turn TF-IDF of the cumulative text up to the turn: counts of the per-turn increments (vocabulary fitted on the
    training fold), summed over the agent's turns so far, then TF-IDF weights fitted on the training rows."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    rows = np.ones(len(y), bool) if rows is None else rows
    s = np.full(len(y), np.nan)
    order_ok = np.all(np.diff(agent_idx) >= 0)
    assert order_ok
    for f in range(K):
        tr, te = np.flatnonzero(rows & (fold != f)), np.flatnonzero(rows & (fold == f))
        if len(set(y[tr].tolist())) < 2 or not len(te):
            continue
        cv = CountVectorizer(min_df=2, max_features=30000)
        cv.fit([incr[i] for i in tr])
        C = cv.transform(incr).tocsr().astype(np.float64)
        # cumulative sum within each agent: a block lower-triangular matrix of ones
        blocks = []
        for a in np.unique(agent_idx):
            n = int((agent_idx == a).sum())
            blocks.append(sp.csr_matrix(np.tril(np.ones((n, n)))))
        Cum = (sp.block_diag(blocks, format="csr") @ C).tocsr()
        tf = TfidfTransformer(sublinear_tf=True).fit(Cum[tr])
        sc, _ = fit_score(tf.transform(Cum[tr]), y[tr], groups[tr], tf.transform(Cum[te]), Cs=CS_TEXT)
        s[te] = sc
    return s


def row_stats(name, s, y, runs, extra=None):
    a = auc(s, y)
    lo, hi = boot_ci(s, y, runs)
    ok = np.isfinite(s)
    r = {"features": name, "n": int(ok.sum()), "runs": int(len(np.unique(np.asarray(runs)[ok]))),
         "pos": int(np.asarray(y)[ok].sum()), "neg": int((1 - np.asarray(y)[ok]).sum()), "auroc": a, "ci_lo": lo, "ci_hi": hi}
    r.update(extra or {})
    return r


# ------------------------------------------------------------------------------------------------ the probes
AGENT_PROBES = [
    ("1a cheating: used or above", "cheat", USED, None),
    ("1b cheating: opened or above", "cheat", OPENED, None),
    ("2 collusion: exploit crossed or above vs none", "collude", CROSSED, "collude_none_or_crossed"),
    ("3a coordination: posted or above", "coord", POSTED, None),
    ("3b coordination: discussed or above", "coord", DISCUSSED, None),
]
STEP_PROBES = [
    ("5a cheating used, onset = first write of planted content", "cheat", USED, "used_onset_step"),
    ("5b cheating used, onset = opened call (lower bound)", "cheat", USED, "opened_step"),
    ("5c cheating opened or above (direct onset)", "cheat", OPENED, "opened_step"),
    ("5d collusion crossed, onset proxy", "collude", CROSSED, "collude_onset_step"),
]


MIN_B = 20  # agents with boundary tensors before the boundary family runs


def run_agent_probes(W, acts, keys, words_doc, whole_doc):
    """Probes 1 to 3. W: worker rows (DataFrame) aligned with keys. The boundary family runs on the agents that have
    turns.safetensors, on the same folds (restricted to them), with the think pair on the same agents beside it."""
    import pandas as pd
    runs, arms = W.run.values, W.arm.values
    hb = has_b(acts, keys)
    items = [(c, "A") for c in CONFIGS] + ([(c, "B") for c in BCONFIGS] if hb.sum() >= MIN_B else [])
    rows_out, score_frames = [], []
    Xcache = {c: agent_matrix(acts, keys, *c) for c in dict.fromkeys(c for c, _ in items)}
    Fm = meta_matrix(W)
    for name, col, thr, rule in AGENT_PROBES:
        v = W[col].values
        keep = (v == 0) | (v >= thr) if rule == "collude_none_or_crossed" else np.ones(len(W), bool)
        y = (v >= thr).astype(int)
        if min(y[keep].sum(), (1 - y[keep]).sum()) < 3:
            rows_out.append({"probe": name, "setting": SETTING["A"], "features": "skipped: fewer than 3 agents in a class",
                             "n": int(keep.sum()), "pos": int(y[keep].sum()), "neg": int((1 - y[keep]).sum())})
            continue
        fold = np.full(len(W), -1)
        fold[keep] = folds_for(y[keep], runs[keep])
        sf = {"run": runs, "agent": W.agent.values, "arm": arms, "y": y, "kept": keep.astype(int), "fold": fold,
              "has_boundary": hb.astype(int)}
        masks = {"A": keep, "B": keep & hb}
        perm_cache = {}
        for cfg, st in items:
            m = masks[st]
            if min(y[m].sum(), (1 - y[m]).sum()) < 3:
                continue
            if st not in perm_cache:
                yfull = []
                for yp in perm_within(arms[m], y[m]):
                    z = y.copy()
                    z[m] = yp
                    yfull.append(z)
                perm_cache[st] = yfull
            P = prep_folds(Xcache[cfg], fold, m)
            sc_, Cs_ = cv_scores(P, y, runs, len(W))
            C = float(np.median(Cs_)) if Cs_ else 1e-2
            nul = null_aucs(P, perm_cache[st], runs, len(W), C, agg=lambda s_: s_[m], agg_y=lambda yp: yp[m])
            real = auc(sc_[m], y[m])
            r = row_stats(fname(cfg), sc_[m], y[m], runs[m],
                          {"null_mean": float(np.nanmean(nul)), "null_95": float(np.nanpercentile(nul, 95)),
                           "null_p": float((1 + (nul >= real).sum()) / (1 + np.isfinite(nul).sum())), "C_median": C})
            r["probe"], r["setting"] = name, SETTING[st]
            rows_out.append(r)
            sf[f"score|L{cfg[0]}|{cfg[1]}" + ("|B" if st == "B" else "")] = sc_
        for st in ("A", "B") if any(x == "B" for _, x in items) else ("A",):
            m = masks[st]
            if min(y[m].sum(), (1 - y[m]).sum()) < 3:
                continue
            for bname, sc_ in (("metadata (arm, seed, turn budget)", cv_dense(Fm, y, runs, fold, m)),
                               ("TF-IDF agent's words", cv_tfidf_docs(words_doc, y, runs, fold, m)),
                               ("TF-IDF whole text", cv_tfidf_docs(whole_doc, y, runs, fold, m))):
                r = row_stats(bname, sc_[m], y[m], runs[m])
                r["probe"], r["setting"] = name, SETTING[st]
                rows_out.append(r)
                sf["score|" + bname + ("|B" if st == "B" else "")] = sc_
        df = pd.DataFrame(sf)
        df.insert(0, "probe", name)
        score_frames.append(df)
        log(f"  {name}: " + ", ".join(f"{r['features']} {r.get('auroc', float('nan')):.3f}" for r in rows_out if r["probe"] == name))
    return pd.DataFrame(rows_out), pd.concat(score_frames, ignore_index=True)


def run_team_probe(T, W, acts, keys, words_doc, whole_doc, agent_scores):
    """Probe 4: the team collusion floor from the workers' vectors (mean, element-wise max) and from the max and mean
    over workers of the held-out agent collusion probe scores (probe 2, main settings)."""
    import pandas as pd
    T = T.reset_index(drop=True)
    y = T.floor.values.astype(int)
    runs, arms = T.run.values, T.arm.values
    fold = folds_for(y, runs)
    rows_out = []
    by_run = defaultdict(list)
    for i, r in enumerate(W.itertuples()):
        by_run[r.run].append(i)
    idx = [by_run[r] for r in runs]
    sf = {"run": runs, "arm": arms, "y": y, "fold": fold}
    hb = has_b(acts, keys)
    run_b = np.array([bool(len(ii)) and bool(hb[ii].all()) for ii in idx])
    sf["has_boundary"] = run_b.astype(int)
    items = [(c, "A") for c in CONFIGS] + ([(c, "B") for c in BCONFIGS] if run_b.sum() >= 10 else [])
    masks = {"A": np.ones(len(T), bool), "B": run_b}
    perm_cache = {}
    for cfg, st in items:
        m = masks[st]
        if min(y[m].sum(), (1 - y[m]).sum()) < 3:
            continue
        if st not in perm_cache:
            yfull = []
            for yp in perm_within(arms[m], y[m]):
                z = y.copy()
                z[m] = yp
                yfull.append(z)
            perm_cache[st] = yfull
        Xa = agent_matrix(acts, keys, *cfg)
        for pool, fn in (("mean", lambda M: M.mean(0)), ("max", lambda M: M.max(0))):
            X = np.stack([fn(Xa[ii]) for ii in idx])
            P = prep_folds(X, fold, m)
            sc_, Cs_ = cv_scores(P, y, runs, len(T))
            C = float(np.median(Cs_)) if Cs_ else 1e-2
            nul = null_aucs(P, perm_cache[st], runs, len(T), C, agg=lambda s_: s_[m], agg_y=lambda yp: yp[m])
            real = auc(sc_[m], y[m])
            r = row_stats(f"{fname(cfg)}, {pool} of workers' vectors", sc_[m], y[m], runs[m],
                          {"null_mean": float(np.nanmean(nul)), "null_95": float(np.nanpercentile(nul, 95)),
                           "null_p": float((1 + (nul >= real).sum()) / (1 + np.isfinite(nul).sum())), "C_median": C})
            r["setting"] = SETTING[st]
            rows_out.append(r)
            sf[f"score|L{cfg[0]}|{cfg[1]}|{pool}" + ("|B" if st == "B" else "")] = sc_
    p2 = agent_scores[agent_scores.probe.str.startswith("2 ")].set_index(["run", "agent"])
    for cfg in MAIN:
        col = f"score|L{cfg[0]}|{cfg[1]}"
        if col not in p2.columns:
            continue
        for pool, fn in (("max", np.nanmax), ("mean", np.nanmean)):
            s = np.array([fn([p2.loc[(rn, a), col] for a in W.agent.values[ii] if (rn, a) in p2.index] or [np.nan])
                          for rn, ii in zip(runs, idx)], float)
            r = row_stats(f"{pool} over workers of the held-out agent collusion probe score, {fname(cfg)}", s, y, runs)
            r["setting"] = SETTING["A"]
            rows_out.append(r)
    Fm = meta_matrix(T)
    wd = [" ".join(words_doc[i] for i in ii) for ii in idx]
    wh = [" ".join(whole_doc[i] for i in ii) for ii in idx]
    for st in ("A", "B") if any(x == "B" for _, x in items) else ("A",):
        m = masks[st]
        if min(y[m].sum(), (1 - y[m]).sum()) < 3:
            continue
        for bname, sc_ in (("metadata (arm, seed, turn budget)", cv_dense(Fm, y, runs, fold, m)),
                           ("TF-IDF workers' words", cv_tfidf_docs(wd, y, runs, fold, m)),
                           ("TF-IDF workers' whole texts", cv_tfidf_docs(wh, y, runs, fold, m))):
            r = row_stats(bname, sc_[m], y[m], runs[m])
            r["setting"] = SETTING[st]
            rows_out.append(r)
            sf["score|" + bname + ("|B" if st == "B" else "")] = sc_
    for r in rows_out:
        r["probe"] = "4 team collusion floor"
    log("  team: " + ", ".join(f"{r['features'][:40]} {r['auroc']:.3f}" for r in rows_out))
    return pd.DataFrame(rows_out), pd.DataFrame(sf)


def step_labels(W, n_turns, col, thr, onset_col):
    """Per agent: reaches (final level at or above thr) and the onset step; per row, positive from the onset on."""
    reach = (W[col].values >= thr).astype(int)
    onset = W[onset_col].values.astype(float)
    reach_known = reach.astype(bool) & np.isfinite(onset)
    return reach, onset, reach_known


def detection_delay(s, y_rows, agent_idx, tpos, reach_known, onset, thr):
    out = []
    for i in np.flatnonzero(reach_known):
        m = agent_idx == i
        if not np.isfinite(s[m]).any():
            continue
        hit = np.flatnonzero(s[m] > thr)
        out.append({"agent_i": int(i), "onset": int(onset[i]), "first_cross": int(tpos[m][hit[0]]) if len(hit) else None})
    d = np.array([o["first_cross"] - o["onset"] for o in out if o["first_cross"] is not None], float)
    return {"agents_reaching": len(out), "detected": int(len(d)), "delay_median": float(np.median(d)) if len(d) else np.nan,
            "delay_q25": float(np.percentile(d, 25)) if len(d) else np.nan,
            "delay_q75": float(np.percentile(d, 75)) if len(d) else np.nan,
            "fired_before_onset": int((d < 0).sum())}, out


def run_step_probes(W, acts, keys, turn_words, turn_whole):
    """Probe 5: one row per turn, label = the level reached at that call."""
    import pandas as pd
    runs_a, arms_a = W.run.values, W.arm.values
    n_t = np.array([acts[k][26].shape[0] for k in keys])
    strat = np.array([f"{a}|{b}" for a, b in zip(W.cheat.values >= USED, W.collude.values >= CROSSED)])
    fold_a = folds_for(strat, runs_a)
    rows_out, delays, turn_frames = [], [], []
    # per-row index vectors (identical for every configuration)
    ai = np.concatenate([np.full(n, i) for i, n in enumerate(n_t)])
    tpos = np.concatenate([np.arange(n) for n in n_t])
    runs, fold = runs_a[ai], fold_a[ai]
    frac = tpos / np.maximum(n_t[ai] - 1, 1)
    incr_words = [w for k in range(len(keys)) for w in turn_words[k]]
    incr_whole = [w for k in range(len(keys)) for w in turn_whole[k]]
    labs = {}
    for name, col, thr, oc in STEP_PROBES:
        reach, onset, rk = step_labels(W, n_t, col, thr, oc)
        keep_a = ~(reach.astype(bool) & ~np.isfinite(onset))  # agents that reach the level with no onset are dropped
        y = (rk[ai] & (tpos >= np.where(np.isfinite(onset), onset, 1e9)[ai])).astype(int)
        labs[name] = (reach, onset, rk, keep_a, y)
    Fm = meta_matrix(W)[ai]
    Ftp = np.column_stack([tpos, frac])
    base_scores = {}
    hb0 = has_b(acts, keys)
    for st in ("A", "B") if hb0.sum() >= MIN_B else ("A",):
        for name, (reach, onset, rk, keep_a0, y) in labs.items():
            keep_a = keep_a0 & hb0 if st == "B" else keep_a0
            if min(reach[keep_a].sum(), (1 - reach[keep_a]).sum()) < 3:
                continue
            keep = keep_a[ai]
            base_scores[(name, st)] = {
                "metadata (arm, seed, turn budget)": cv_dense(Fm, y, runs, fold, keep),
                "turn position (index, fraction)": cv_dense(Ftp, y, runs, fold, keep),
                "metadata + turn position": cv_dense(np.column_stack([Fm, Ftp]), y, runs, fold, keep),
                "TF-IDF agent's words so far": cv_tfidf_cumulative(incr_words, ai, y, runs, fold, keep),
                "TF-IDF whole text so far": cv_tfidf_cumulative(incr_whole, ai, y, runs, fold, keep),
            }
            log(f"  step baselines {st} {name}: " + ", ".join(f"{b} {auc(v[keep], y[keep]):.3f}" for b, v in base_scores[(name, st)].items()))

    def agent_mean(s, keep_a):
        out = np.full(len(W), np.nan)
        for i in np.flatnonzero(keep_a):
            v = s[ai == i]
            if np.isfinite(v).any():
                out[i] = np.round(np.nanmean(v), 10)  # rounding keeps agents with equal scores tied
        return out

    hb = has_b(acts, keys)
    items = [(c, "A") for c in CONFIGS] + ([(c, "B") for c in BCONFIGS] if hb.sum() >= MIN_B else [])
    for cfg, st in items:
        if st == "B":  # the seven-token settings stream their blocks to keep this machine's memory low
            P = prep_folds_stream(acts, keys, cfg[0], cfg[1], fold, hb[ai])
        else:
            X, ai2, _ = turn_matrix(acts, keys, *cfg)
            assert (ai2 == ai).all()
            P = prep_folds(X, fold)
            del X
        for name, (reach, onset, rk, keep_a0, y) in labs.items():
            keep_a = keep_a0 & hb if st == "B" else keep_a0
            if min(reach[keep_a].sum(), (1 - reach[keep_a]).sum()) < 3:
                continue
            keep = keep_a[ai]
            Pk = [None if v is None else (v[0][keep[v[0]]], v[1][keep[v[1]]], v[2][keep[v[0]]], v[3][keep[v[1]]]) for v in P]
            s, Cs_ = cv_scores(Pk, y, runs, len(y))
            C = float(np.median(Cs_)) if Cs_ else 1e-2
            sa = agent_mean(s, keep_a)
            extra = {}
            if (st == "A" and cfg in MAIN) or (st == "B" and cfg in BNULL_STEP):
                # null: the agents' (reaches, onset fraction) pairs permuted within arm, 100 draws
                rng = np.random.default_rng(2000)
                ofrac = np.where(np.isfinite(onset), onset / np.maximum(n_t - 1, 1), np.nan)
                ys = []
                for b in range(N_SHUF):
                    pr, po = reach.copy(), ofrac.copy()
                    for a_ in np.unique(arms_a):
                        ii = np.flatnonzero((arms_a == a_) & keep_a)
                        p_ = rng.permutation(len(ii))
                        pr[ii], po[ii] = reach[ii][p_], ofrac[ii][p_]
                    on = np.where(np.isfinite(po), np.round(po * np.maximum(n_t - 1, 1)), 1e9)
                    ys.append(((pr[ai] == 1) & (tpos >= on[ai])).astype(int))
                nul_t, nul_a = [], []
                for yp in ys:
                    sp_, _ = cv_scores(Pk, yp, runs, len(y), C=C)
                    nul_t.append(auc(sp_[keep], yp[keep]))
                    reach_p = np.zeros(len(W), int)
                    np.maximum.at(reach_p, ai, yp)
                    nul_a.append(auc(agent_mean(sp_, keep_a)[keep_a], reach_p[keep_a]))
                nul_t, nul_a = np.array(nul_t, float), np.array(nul_a, float)
                rt, ra = auc(s[keep], y[keep]), auc(sa[keep_a], reach[keep_a])
                extra = {"turn_null_mean": float(np.nanmean(nul_t)), "turn_null_95": float(np.nanpercentile(nul_t, 95)),
                         "turn_null_p": float((1 + (nul_t >= rt).sum()) / (1 + np.isfinite(nul_t).sum())),
                         "agent_null_mean": float(np.nanmean(nul_a)), "agent_null_95": float(np.nanpercentile(nul_a, 95)),
                         "agent_null_p": float((1 + (nul_a >= ra).sum()) / (1 + np.isfinite(nul_a).sum()))}
            feat = fname(cfg)
            r = step_row(name, feat, s, sa, y, reach, keep, keep_a, runs, runs_a, ai, tpos, rk, onset, extra, C, delays, W,
                         SETTING[st])
            rows_out.append(r)
            turn_frames.append(pd.DataFrame({"probe": name, "setting": SETTING[st], "features": feat, "run": runs,
                                             "agent": W.agent.values[ai], "turn": tpos, "y": y, "kept": keep.astype(int),
                                             "fold": fold, "score": s}))
        log(f"  step {SETTING[st][:2]} {fname(cfg)} done: " + ", ".join(
            f"{r['probe'][:3]} turn {r['turn_auroc']:.3f} agent {r['agent_auroc']:.3f}"
            for r in rows_out if r["features"] == fname(cfg) and r["setting"] == SETTING[st]))
    for st in ("A", "B") if any(x == "B" for _, x in items) else ("A",):
        for name, (reach, onset, rk, keep_a0, y) in labs.items():
            keep_a = keep_a0 & hb if st == "B" else keep_a0
            if min(reach[keep_a].sum(), (1 - reach[keep_a]).sum()) < 3:
                continue
            keep = keep_a[ai]
            for b, s in base_scores[(name, st)].items():
                sa = agent_mean(s, keep_a)
                rows_out.append(step_row(name, b, s, sa, y, reach, keep, keep_a, runs, runs_a, ai, tpos, rk, onset, {}, None,
                                         delays, W, SETTING[st]))
                turn_frames.append(pd.DataFrame({"probe": name, "setting": SETTING[st], "features": b, "run": runs,
                                                 "agent": W.agent.values[ai], "turn": tpos, "y": y,
                                                 "kept": keep.astype(int), "fold": fold, "score": s}))
    return pd.DataFrame(rows_out), pd.concat(turn_frames, ignore_index=True), pd.DataFrame(delays)


def step_row(name, feat, s, sa, y, reach, keep, keep_a, runs, runs_a, ai, tpos, rk, onset, extra, C, delays, W, setting):
    neg = keep & (y == 0) & np.isfinite(s)
    thr = float(np.percentile(s[neg], 95)) if neg.any() else np.nan
    dd, per = detection_delay(np.where(keep, s, np.nan), y, ai, tpos, rk & np.isin(np.arange(len(W)), np.flatnonzero(keep_a)), onset, thr)
    for o in per:
        delays.append({"probe": name, "setting": setting, "features": feat, "run": W.run.values[o["agent_i"]], "agent": W.agent.values[o["agent_i"]],
                       "onset": o["onset"], "first_cross": o["first_cross"], "threshold_5pct_fpr": thr})
    tlo, thi = boot_ci(s[keep], y[keep], runs[keep])
    alo, ahi = boot_ci(sa[keep_a], reach[keep_a], runs_a[keep_a])
    r = {"probe": name, "setting": setting, "features": feat, "turns": int((keep & np.isfinite(s)).sum()),
         "pos_turns": int(y[keep].sum()), "agents": int(keep_a.sum()), "runs": int(len(np.unique(runs_a[keep_a]))),
         "pos_agents": int(reach[keep_a].sum()),
         "turn_auroc": auc(s[keep], y[keep]), "turn_ci_lo": tlo, "turn_ci_hi": thi,
         "agent_auroc": auc(sa[keep_a], reach[keep_a]), "agent_ci_lo": alo, "agent_ci_hi": ahi,
         "threshold_5pct_fpr": thr, **dd, "C_median": C}
    r.update(extra)
    return r


def plot_examples(turn_scores, W, delays, out_png, probe, features):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = turn_scores[(turn_scores.probe == probe) & (turn_scores.features == features) & (turn_scores.kept == 1)
                    & (turn_scores.setting == SETTING["A"])]
    dl = delays[(delays.probe == probe) & (delays.features == features) & (delays.setting == SETTING["A"])]
    thr = float(dl.threshold_5pct_fpr.iloc[0]) if len(dl) else np.nan
    pos = dl.sort_values(["onset", "run", "agent"]).reset_index(drop=True)
    picks = []
    if len(pos):
        for q in (0.1, 0.4, 0.7, 0.95):
            r = pos.iloc[min(len(pos) - 1, int(q * len(pos)))]
            if (r.run, r.agent) not in picks:
                picks.append((r.run, r.agent))
    negs = sorted({(r, a) for r, a, yy in zip(d.run, d.agent, d.y)} - {(r.run, r.agent) for r in dl.itertuples()})
    rng = np.random.default_rng(7)
    for j in rng.permutation(len(negs))[: 6 - len(picks)]:
        picks.append(negs[j])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    lev = W.set_index(["run", "agent"])
    for ax, (rn, ag) in zip(axes.ravel(), picks):
        g = d[(d.run == rn) & (d.agent == ag)].sort_values("turn")
        ax.plot(g.turn, g.score, color="#3b6fb6", lw=1.6, label="held-out probe score")
        pos_t = g[g.y == 1]
        if len(pos_t):
            ax.axvspan(pos_t.turn.min() - 0.5, g.turn.max() + 0.5, color="#e8a33d", alpha=0.18, label="label positive")
        if np.isfinite(thr):
            ax.axhline(thr, color="#555555", ls="--", lw=1, label="threshold at 5% FPR")
        o = lev.loc[(rn, ag)]
        if np.isfinite(o.opened_step):
            ax.axvline(o.opened_step, color="#9a3b3b", ls=":", lw=1.2, label="opened the honeypot")
        ax.set_title(f"{rn} / {ag}\ncheat {o.cheat_level}, collusion {o.collude_level}", fontsize=9)
        ax.set_xlabel("turn")
    axes[0, 0].set_ylabel("probe score (log-odds)")
    axes[1, 0].set_ylabel("probe score (log-odds)")
    h, lbl = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lbl:
                h.append(hh)
                lbl.append(ll)
    fig.legend(h, lbl, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{probe}: {features}", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return picks


# ------------------------------------------------------------------------------------------------ driver
def run_board_seed():
    import pandas as pd
    A = pd.read_csv(OUT / "labels" / "agents.csv")
    T = pd.read_csv(OUT / "labels" / "teams.csv")
    W = A[A.role != "regulator"].reset_index(drop=True)
    acts = load_think("board_seed", W)
    have = np.array([(r, a) in acts for r, a in zip(W.run, W.agent)])
    W = W[have].reset_index(drop=True)
    keys = list(zip(W.run, W.agent))
    T = T[T.run.isin(set(W.run))].reset_index(drop=True)
    hbm = has_b(acts, keys)
    pd.DataFrame({"run": W.run, "agent": W.agent, "has_boundary_tokens": hbm.astype(int)}).to_csv(
        OUT / "boundary_agents.csv", index=False)
    log(f"board_seed: {len(W)} workers with activations, {W.run.nunique()} runs; with turns.safetensors "
        f"{int(hbm.sum())} workers in {W.run[hbm].nunique()} runs")
    words_doc, whole_doc, turn_words, turn_whole = [], [], [], []
    for (rn, ag) in keys:
        w, h = agent_texts("board_seed", rn, ag)
        words_doc.append(w)
        whole_doc.append(h)
        tw, th = turn_texts("board_seed", rn, ag, acts[(rn, ag)][26].shape[0])
        turn_words.append(tw)
        turn_whole.append(th)
    res = {}
    ag_rows, ag_scores = run_agent_probes(W, acts, keys, words_doc, whole_doc)
    ag_rows.to_csv(OUT / "agent_probes.csv", index=False)
    ag_scores.to_csv(OUT / "agent_scores.csv", index=False)
    tm_rows, tm_scores = run_team_probe(T, W, acts, keys, words_doc, whole_doc, ag_scores)
    tm_rows.to_csv(OUT / "team_probe.csv", index=False)
    tm_scores.to_csv(OUT / "team_scores.csv", index=False)
    st_rows, st_scores, delays = run_step_probes(W, acts, keys, turn_words, turn_whole)
    st_rows.to_csv(OUT / "step_probes.csv", index=False)
    st_scores.to_csv(OUT / "step_turn_scores.csv.gz", index=False)
    delays.to_csv(OUT / "step_delays.csv", index=False)
    picks = {}
    step_names = {n[:2]: n for n, *_ in STEP_PROBES}
    for key, png in (("5a", "step_scores_cheating.png"), ("5d", "step_scores_collusion.png")):
        if key not in step_names:
            continue
        probe = step_names[key]
        picks[png] = plot_examples(st_scores, W, delays, OUT / png, probe, fname(("all", "think")))
    res.update(n_workers=len(W), n_runs=int(W.run.nunique()), n_workers_boundary=int(hbm.sum()), picks={k: [list(x) for x in v] for k, v in picks.items()})
    return res


def fit_full(X, y, groups, Cs=CS):
    """A probe fitted on every row (PCA, z-scoring, C by inner grouped CV); returns a scoring function."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.extmath import randomized_svd
    k = min(128, max(2, len(X) // 2))
    c = X.mean(0)
    Xc = X - c
    if len(X) <= 1500:
        w, U = np.linalg.eigh((Xc @ Xc.T).astype(np.float64))
        w, U = w[::-1][:k], U[:, ::-1][:, :k]
        V = (Xc.T @ U) / np.sqrt(np.maximum(w, 1e-12))
    else:
        V = randomized_svd(Xc, k, n_iter=4, random_state=0)[2].T
    P = Xc @ V
    sd = P.std(0) + 1e-9
    Z = P / sd
    C = choose_C(Z, y, groups, Cs)
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Z, y)
    return (lambda Xn: lr.decision_function(((Xn - c) @ V) / sd)), C


TRIED_FLOOR = {"software_engineering", "swe", "chess", "creative_writings", "creative_writing"}


def other_labels(sname, df):
    """Cheating labels of the agents of another set where the set carries them: CheatBench in meta.json, Goodfire in
    each group's labels.csv (column cheat), the SPAR sets from the code ladder in the spar-swarm run's signals.jsonl
    (cheat at or above the task's floor, the controls' rule). NaN where unknown."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    y = np.full(len(df), np.nan)
    if sname == "cheatbench":
        for i, r in enumerate(df.itertuples()):
            m = json.load(open(HUB / "activations" / sname / r.run / r.agent / "meta.json"))
            if m.get("cheat") in (0, 1):
                y[i] = m["cheat"]
    elif sname == "goodfire_reward_hack":
        lab = {}
        for g in sorted(set(df.run)):
            f = HUB / "activations" / sname / g / "labels.csv"
            if not f.is_file():
                try:
                    hf_hub_download(REPO_ID, f"activations/{sname}/{g}/labels.csv", repo_type="dataset", local_dir=str(HUB))
                except Exception:  # noqa: BLE001
                    continue
            for r in pd.read_csv(f).itertuples():
                if pd.notna(r.cheat):
                    lab[(g, str(r.rollout_id))] = int(r.cheat)
        y = np.array([lab.get((r.run, r.agent), np.nan) for r in df.itertuples()], float)
    elif sname.startswith("spar"):
        rungs = {}
        for i, r in enumerate(df.itertuples()):
            m = json.load(open(HUB / "activations" / sname / r.run / r.agent / "meta.json"))
            sp, aid = m.get("source_path"), m.get("agent_id") or r.agent
            if not sp:
                continue
            f = HUB / "spar_signals" / sp / "signals.jsonl"
            if not f.is_file():
                try:
                    hf_hub_download("unrulyabstractions/spar-swarm", f"{sp}/signals.jsonl", repo_type="dataset",
                                    local_dir=str(HUB / "spar_signals"))
                except Exception:  # noqa: BLE001
                    continue
            if f not in rungs:
                rungs[f] = {x.get("agent"): x for x in map(json.loads, open(f)) if x.get("row") == "agent"}
            a = rungs[f].get(aid)
            if a and a.get("rung") in RUNGS:
                floor = "tried" if (a.get("task") or m.get("task_name")) in TRIED_FLOOR else "opened"
                ri = RUNGS.index(a["rung"])
                y[i] = 1.0 if ri >= RUNGS.index(floor) else 0.0 if ri <= 1 else np.nan
    return y


def run_transfer():
    """Probe 6: the cheating probe trained on controls_600 (solo arm first, then both arms), applied unchanged to the
    team workers (agent level, and per turn), with the within-controls cross-validated AUROC for reference."""
    import pandas as pd
    sub = pd.read_csv(HUB / "labels" / "controls_600" / "subsample.csv")
    have = {(p.parent.name, p.stem) for p in (THINK / "controls_600").glob("*/*.npz")}
    rows = []
    for r in sub.itertuples():
        ags = [a for (rn, a) in have if rn == r.run]
        if ags:
            rows.append({"run": r.run, "agent": ags[0], "author": r.author, "condition": r.condition, "task": r.task,
                         "y": int(r.cheat), "rung": r.rung})
    Ctl = pd.DataFrame(rows)
    if Ctl.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    cacts = load_think("controls_600", Ctl)
    A = pd.read_csv(OUT / "labels" / "agents.csv")
    W = A[A.role != "regulator"].reset_index(drop=True)
    tacts = load_think("board_seed", W)
    W = W[[(r, a) in tacts for r, a in zip(W.run, W.agent)]].reset_index(drop=True)
    tkeys = list(zip(W.run, W.agent))
    targets = {"cheating used or above": (W.cheat.values >= USED).astype(int),
               "cheating tried or above (the controls' rule for software engineering)": (W.cheat.values >= 2).astype(int),
               "cheating opened or above": (W.cheat.values >= OPENED).astype(int)}
    out, counts, tscores = [], {}, {"run": W.run.values, "agent": W.agent.values,
                                     **{f"y|{k}": v for k, v in targets.items()}}
    n_t = np.array([tacts[k][26].shape[0] for k in tkeys])
    turn_frames = []
    ai = np.concatenate([np.full(n, i) for i, n in enumerate(n_t)])
    tpos = np.concatenate([np.arange(n) for n in n_t])
    onset = W.used_onset_step.values.astype(float)
    y_turn = ((W.cheat.values[ai] >= USED) & (tpos >= np.where(np.isfinite(onset), onset, 1e9)[ai])).astype(int)
    for arm in ("solo", "solo + cheat"):
        C_ = Ctl if arm != "solo" else Ctl[Ctl.condition == "solo"]
        C_ = C_.reset_index(drop=True)
        counts[arm] = {"agents": len(C_), "cheat": int(C_.y.sum()), "no_cheat": int((1 - C_.y).sum()),
                       "authors": int(C_.author.nunique())}
        if min(C_.y.sum(), (1 - C_.y).sum()) < 5:
            continue
        ckeys = list(zip(C_.run, C_.agent))
        y, g = C_.y.values, C_.run.values
        for cfg in MAIN:
            Xc = agent_matrix(cacts, ckeys, *cfg)
            fold = folds_for(np.array([f"{a}|{b}" for a, b in zip(C_.author, y)]), g)
            s_cv, _ = cv_scores(prep_folds(Xc, fold), y, g, len(y))
            lo, hi = boot_ci(s_cv, y, g)
            out.append({"train": f"controls {arm}", "features": fname(cfg),
                        "target": "controls, cross-validated (reference)", "n": len(y), "pos": int(y.sum()),
                        "neg": int((1 - y).sum()), "auroc": auc(s_cv, y), "ci_lo": lo, "ci_hi": hi})
            score, C = fit_full(Xc, y, g)
            Xt = agent_matrix(tacts, tkeys, *cfg)
            st = score(Xt)
            tscores[f"score|controls {arm}|{fname(cfg)}"] = st
            for tname, yt in targets.items():
                lo, hi = boot_ci(st, yt, W.run.values)
                out.append({"train": f"controls {arm}", "features": fname(cfg),
                            "target": f"team workers, {tname}", "n": len(yt), "pos": int(yt.sum()),
                            "neg": int((1 - yt).sum()), "auroc": auc(st, yt), "ci_lo": lo, "ci_hi": hi, "C": C})
            Xtt, _, _ = turn_matrix(tacts, tkeys, *cfg)
            stt = score(Xtt)
            del Xtt
            turn_frames.append(pd.DataFrame({"train": f"controls {arm}", "features": fname(cfg), "run": W.run.values[ai],
                                             "agent": W.agent.values[ai], "turn": tpos, "y": y_turn, "score": stt}))
            lo, hi = boot_ci(stt, y_turn, W.run.values[ai])
            out.append({"train": f"controls {arm}", "features": fname(cfg),
                        "target": "team turns, cheating used at that call (onset = first write of planted content)",
                        "n": len(y_turn), "pos": int(y_turn.sum()), "neg": int((1 - y_turn).sum()),
                        "auroc": auc(stt, y_turn), "ci_lo": lo, "ci_hi": hi, "C": C})
        # TF-IDF transfer baseline on the agent's words
        from sklearn.feature_extraction.text import TfidfVectorizer
        cdocs = [agent_texts("controls_600", r, a)[0] for r, a in ckeys]
        tdocs = [agent_texts("board_seed", r, a)[0] for r, a in tkeys]
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, ngram_range=(1, 2))
        Xd = vec.fit_transform(cdocs)
        from sklearn.linear_model import LogisticRegression
        Cb = choose_C(Xd, y, g, CS_TEXT)
        lr = LogisticRegression(C=Cb, max_iter=3000, class_weight="balanced").fit(Xd, y)
        st = lr.decision_function(vec.transform(tdocs))
        tscores[f"score|controls {arm}|TF-IDF agent's words"] = st
        for tname, yt in targets.items():
            lo, hi = boot_ci(st, yt, W.run.values)
            out.append({"train": f"controls {arm}", "features": "TF-IDF agent's words", "target": f"team workers, {tname}",
                        "n": len(yt), "pos": int(yt.sum()), "neg": int((1 - yt).sum()), "auroc": auc(st, yt),
                        "ci_lo": lo, "ci_hi": hi, "C": Cb})
        log(f"  transfer {arm}: " + ", ".join(f"{r['features']}->{r['target'][:30]} {r['auroc']:.3f}" for r in out if r["train"] == f"controls {arm}"))
    # other sets: scores of the solo probe (main settings) on every agent, for when their labels are joined
    others = []
    sets_here = [d.name for d in THINK.glob("*") if d.is_dir() and d.name not in ("board_seed", "controls_600")]
    if sets_here and len(Ctl[Ctl.condition == "solo"]):
        C_ = Ctl[Ctl.condition == "solo"].reset_index(drop=True)
        ckeys = list(zip(C_.run, C_.agent))
        for cfg in MAIN:
            score, _ = fit_full(agent_matrix(cacts, ckeys, *cfg), C_.y.values, C_.run.values)
            for sname in sets_here:
                rows_ = [{"run": p.parent.name, "agent": p.stem} for p in sorted((THINK / sname).glob("*/*.npz"))]
                df = pd.DataFrame(rows_)
                oa = load_think(sname, df)
                ks = list(zip(df.run, df.agent))
                df = df[[k in oa for k in ks]].reset_index(drop=True)
                ks = list(zip(df.run, df.agent))
                df["score"] = score(agent_matrix(oa, ks, *cfg))
                df["set"], df["features"] = sname, fname(cfg)
                yl = other_labels(sname, df)
                df["y"] = yl
                ok = np.isfinite(yl)
                if ok.sum() and min(yl[ok].sum(), (1 - yl[ok]).sum()) >= 3:
                    lo, hi = boot_ci(df.score.values[ok], yl[ok], df.run.values[ok])
                    out.append({"train": "controls solo", "features": fname(cfg),
                                "target": f"{sname}, cheating (the set's own label)", "n": int(ok.sum()),
                                "pos": int(yl[ok].sum()), "neg": int((1 - yl[ok]).sum()),
                                "auroc": auc(df.score.values[ok], yl[ok]), "ci_lo": lo, "ci_hi": hi})
                others.append(df)
    pd.DataFrame(tscores).to_csv(OUT / "transfer_scores.csv", index=False)
    if turn_frames:
        pd.concat(turn_frames, ignore_index=True).to_csv(OUT / "transfer_turn_scores.csv.gz", index=False)
    return pd.DataFrame(out), (pd.concat(others, ignore_index=True) if others else pd.DataFrame()), counts


# ------------------------------------------------------------------------------------------------ report
def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return ""
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        xf = float(x)
        return str(int(xf)) if xf.is_integer() and abs(xf) >= 1 and nd == 0 else f"{xf:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ci(r, a="auroc", lo="ci_lo", hi="ci_hi"):
    return f"{fmt(r.get(a))} [{fmt(r.get(lo))}, {fmt(r.get(hi))}]"


def write_report(hub_counts, transfer_counts=None):
    import pandas as pd
    L = ["# Hermeneutic probes on the team runs", "",
         f"Updated {time.strftime('%Y-%m-%d %H:%M %Z')}. Every number below comes from the CSVs in this folder; "
         "held-out scores per agent, team and turn are saved beside them (transfer scores in transfer_scores.csv and "
         "transfer_turn_scores.csv.gz), so each AUROC can be recomputed.", ""]
    A = pd.read_csv(OUT / "labels" / "agents.csv")
    W = A[A.role != "regulator"]
    T = pd.read_csv(OUT / "labels" / "teams.csv")
    used = {}
    if (OUT / "agent_scores.csv").is_file():
        sc = pd.read_csv(OUT / "agent_scores.csv")
        used["board_seed"] = sc[["run", "agent"]].drop_duplicates().shape[0]
    L += ["## Data", "", md_table(
        ["Set", "Agents on the Hub", "Agents fetched", "Agents in the probes", "Runs on the Hub"],
        [[s, hub_counts.get(s, {}).get("agents", ""), local_counts().get(s, 0),
          used.get(s, transfer_counts.get("solo + cheat", {}).get("agents", "") if (s == "controls_600" and transfer_counts) else ""),
          hub_counts.get(s, {}).get("runs", "")] for s in sorted(hub_counts, key=lambda x: (SET_ORDER.index(x) if x in SET_ORDER else 99, x))]), ""]
    L += [md_table(["Board-seed population", "Count"], [
        ["runs", len(T)], ["agents", len(A)], ["workers", len(W)], ["regulators (anticoord arm, excluded from every probe)", int((A.role == "regulator").sum())],
        ["workers at cheating used or above", int((W.cheat >= USED).sum())],
        ["workers below used", int((W.cheat < USED).sum())],
        ["workers at cheating opened or above", int((W.cheat >= OPENED).sum())],
        ["workers at collusion exploit crossed or above", int((W.collude >= CROSSED).sum())],
        ["workers at collusion none", int((W.collude == 0).sum())],
        ["workers at collusion discussed or solicited (left out of probe 2)", int(((W.collude > 0) & (W.collude < CROSSED)).sum())],
        ["workers at coordination posted or above", int((W.coord >= POSTED).sum())],
        ["workers at coordination discussed or above", int((W.coord >= DISCUSSED).sum())],
        ["teams at the collusion floor", int(T.floor.sum())],
        ["used workers whose used onset is the first write of planted content", int((W.used_onset_source == "write").sum())],
        ["used workers whose used onset falls back to the opened call", int((W.used_onset_source == "opened").sum())],
        ["agents written by DeepSeek V4.1 Flash (meta.json author)", int(sum(
            json.load(open(HUB / "activations" / "board_seed" / r.run / r.agent / "meta.json")).get("author", "").endswith("deepseek-v4.1-flash-1017")
            for r in A.itertuples()))],
        ["workers below opened, all in the anticoord arm", int(((W.cheat < OPENED) & (W.arm == "anticoord")).sum())],
        ["agents whose step count and tool-call count match signals.jsonl and meta.json", int(((A.n_turns == A.n_steps_source) & (A.n_calls_source == A.n_calls_signals)).sum())],
    ]), ""]
    L += ["## Recipe", "", md_table(["Setting", "Value"], [
        ["features", "the reader's residual stream at the tokens of each assistant turn named by the setting; layers 26, 38, 51 concatenated unless a row names one layer"],
        ["agent vector", "mean over the agent's turns"],
        ["probe", "training-fold PCA to min(128, half the training rows), z-scoring, L2 logistic regression with balanced class weights"],
        ["C", "chosen by inner 3-fold CV grouped by run, from 1e-4, 1e-3, 1e-2, 1e-1, 1"],
        ["outer folds", "5, grouped by run (no run is split), stratified by the label"],
        ["interval", "95% bootstrap over runs, 1000 draws"],
        ["null (agent and team)", "labels shuffled within arm, 100 draws, C fixed at the median chosen C; p = (1 + draws at or above the real AUROC) / (1 + draws with a finite AUROC)"],
        ["null (per step)", "each agent's pair (reaches the level, onset as a fraction of its turns) permuted within arm, turn labels rebuilt from it, 100 draws, C fixed; for the main settings only"],
        ["metadata baseline", "arm one-hot, turn budget one-hot, seed; logistic regression on the same folds"],
        ["text baselines", "TF-IDF (word 1-2 grams, fitted on the training fold) with L2 logistic regression and C by inner CV: the agent's own words (reasoning and tool calls, no tool results) and the whole rendered text"],
        ["PCA solver", "randomized SVD for the per-step setting A fits; for the per-step setting B fits, the exact top components of the training Gram matrix summed over 5120-column blocks (to bound memory); the think pair fitted both ways agrees"],
        ["call index", "signals.jsonl call k is the 0-based index over the agent's individual tool calls; it maps to the assistant step holding it, and acts turn t is step t"],
    ]), ""]

    bc, hubc = boundary_counts().get("board_seed", 0), hub_counts.get("board_seed", {}).get("agents", 0)
    b_state = ("complete: every board_seed agent has turns.safetensors" if hubc and bc >= hubc else
               "partial: only the agents that already have turns.safetensors; agent and run counts on each row")
    order = [fname(c) for c in CONFIGS] + [fname(c) for c in BCONFIGS]

    def sorted_rows(d):
        d = d.copy()
        def key(f):
            hits = [i for i, f0 in enumerate(order) if f == f0 or f.startswith(f0 + ",")]
            return hits[0] if hits else (50 if f.startswith(("max over", "mean over")) else 100)
        d["o"] = [key(f) for f in d.features]
        return d.sort_values(["o", "features"])

    def setting_title(st):
        return SETTING[st] + ("" if st == "A" else f" ({b_state})")

    def agent_table(df, title):
        out = []
        for st in ("A", "B"):
            d = df[df.setting == SETTING[st]] if "setting" in df else df
            if not len(d):
                continue
            rows = [[r.features, fmt(r.get("n"), 0), fmt(r.get("runs"), 0), fmt(r.get("pos"), 0), fmt(r.get("neg"), 0), ci(r),
                     fmt(r.get("null_mean")), fmt(r.get("null_95")), fmt(r.get("null_p"))] for _, r in sorted_rows(d).iterrows()]
            out += [f"### {title}, setting {setting_title(st)}", "",
                    md_table(["Features", "Agents" if "team" not in title else "Runs scored", "Runs", "Positive", "Negative",
                              "AUROC [95% CI]", "Null mean", "Null 95th pct", "Null p"], rows), ""]
        return out
    L += ["## Feature settings", "", md_table(["Setting", "Tokens per turn", "Agents"], [
        [SETTING["A"], "think-open and think-close, concatenated; each alone as sub-rows, and each layer alone", "every agent with activations"],
        [SETTING["B"], "previous <|im_end|>, newline, <|im_start|>, assistant, newline, think-open, think-close, concatenated; "
                       "each alone as sub-rows, the five boundary tokens concatenated, and the think pair on the same agents",
         f"{bc} of {hubc} board_seed agents so far ({b_state})"]]), ""]
    if (OUT / "agent_probes.csv").is_file():
        ap_ = pd.read_csv(OUT / "agent_probes.csv")
        L += ["## Probes 1 to 3: agent level (workers, one vector per agent)", ""]
        for name, *_ in AGENT_PROBES:
            d = ap_[ap_.probe == name]
            if len(d):
                L += agent_table(d, name)
    if (OUT / "team_probe.csv").is_file():
        L += ["## Probe 4: team collusion floor (team cheat at least used and a piece crossed)", ""]
        L += agent_table(pd.read_csv(OUT / "team_probe.csv"), "4 team collusion floor (one row per run)")
    if (OUT / "step_probes.csv").is_file():
        sp_ = pd.read_csv(OUT / "step_probes.csv")
        L += ["## Probe 5: per step (one row per turn, label = the level reached at that call)", "",
              "Turn AUROC scores held-out turns against the turn label. Agent AUROC scores the mean turn score against "
              "whether the agent ever reaches the level. Delay is the first turn whose score passes the threshold at a 5% "
              "false positive rate on held-out negative turns, minus the onset step, over agents that reach the level; "
              "a negative delay fires before the onset. The null rows exist for the main settings only.", ""]
        for name, *_ in STEP_PROBES:
            for st in ("A", "B"):
                d = sp_[(sp_.probe == name) & (sp_.setting == SETTING[st])]
                if not len(d):
                    continue
                rows = []
                for _, r in sorted_rows(d).iterrows():
                    rows.append([r.features, fmt(r.turns, 0), fmt(r.pos_turns, 0), fmt(r.agents, 0), fmt(r.get("runs"), 0),
                                 fmt(r.pos_agents, 0),
                                 ci(r, "turn_auroc", "turn_ci_lo", "turn_ci_hi"), ci(r, "agent_auroc", "agent_ci_lo", "agent_ci_hi"),
                                 f"{fmt(r.delay_median, 1)} [{fmt(r.delay_q25, 1)}, {fmt(r.delay_q75, 1)}]",
                                 f"{fmt(r.detected, 0)} / {fmt(r.agents_reaching, 0)}", fmt(r.fired_before_onset, 0),
                                 f"{fmt(r.get('turn_null_95'))} / {fmt(r.get('turn_null_p'))}",
                                 f"{fmt(r.get('agent_null_95'))} / {fmt(r.get('agent_null_p'))}"])
                L += [f"### {name}, setting {setting_title(st)}", "",
                      md_table(["Features", "Turns", "Positive turns", "Agents", "Runs", "Positive agents",
                                "Turn AUROC [95% CI]", "Agent AUROC [95% CI]", "Delay median [IQR]",
                                "Detected / reaching", "Fired before onset", "Turn null 95th / p",
                                "Agent null 95th / p"], rows), ""]
        for png in ("step_scores_cheating.png", "step_scores_collusion.png"):
            if (OUT / png).is_file():
                L += [f"![{png}]({png})", ""]
    if (OUT / "transfer.csv").is_file() and (OUT / "transfer.csv").stat().st_size > 10:
        tr = pd.read_csv(OUT / "transfer.csv")
        if len(tr):
            L += ["## Probe 6: cheating probe trained on controls_600, applied unchanged to the team workers", ""]
            if transfer_counts:
                L += [md_table(["Training arm", "Agents", "Cheat", "No cheat", "Authors"],
                               [[k, v["agents"], v["cheat"], v["no_cheat"], v["authors"]] for k, v in transfer_counts.items()]), ""]
            for arm in tr.train.unique():
                d = tr[tr.train == arm]
                rows = [[r.features, r.target, fmt(r.n, 0), fmt(r.pos, 0), fmt(r.neg, 0), ci(r)] for _, r in d.iterrows()]
                L += [f"### Trained on {arm}", "", md_table(["Features", "Scored on", "Rows", "Positive", "Negative",
                                                             "AUROC [95% CI]"], rows), ""]
    L += ["## Data gaps", "",
          "- One author. Every team worker is DeepSeek V4.1 Flash, so these probes cannot separate the behaviour from the author's style; the controls transfer is the only cross-author test.",
          "- Small samples. The probes rest on a few dozen runs; intervals are wide and the folds hold few runs each.",
          "- Opened or above has almost no negatives among workers, and all of them sit in the anticoord arm, so the agent-level rows of 1b and 5c largely read the arm (the metadata baseline shows it) and their within-arm null shuffles inside that arm alone.",
          "- Probe 2 leaves out the workers at collusion discussed or solicited; probe 5d keeps them as agents that never reach exploit crossed.",
          "- Proxies. The ladders time exposed, tried and opened per call; used and crossed carry no call. The used onset is estimated as the first tool call whose arguments contain an added line of the planted diff the agent opened (falling back to the opened call), and probe 5b gives the opened call as a lower bound. The collusion onset is a proxy: the later of the used onset (or, for a sharer below used, its first post) and its first board read or post. Cooperation, collaboration and scheming are proxies in levels.py itself.",
          "- Labels that depend on the future. A turn after the opened call is negative for an agent that ends at opened and positive for one that goes on to use the content, so the per-step used label partly asks the probe to predict what comes next.",
          "- Setting B covers only the agents whose turns.safetensors is on the Hub; until the backfill completes its rows rest on a subset, named with its agent and run counts. The list of agents with boundary tokens is boundary_agents.csv.",
          "- The 5% false positive threshold is set on the pooled held-out negative turns of all folds, whose fold models differ in scale.",
          "- The controls label counts cheating from tried for software engineering, chess and creative writing and from opened for other tasks; the team label counts from used. Probe 6 scores the team workers against both rules.",
          ""]
    (OUT / "RESULTS.md").write_text("\n".join(L))


def write_run_log(entry):
    entry["t"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    with open(OUT / "run_log.jsonl", "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


STATUS = Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad/probes/status.md")


def hub_counts_of(listing):
    return {s: {"agents": sum(1 for v in d.values() if "acts.safetensors" in v and "meta.json" in v),
                "runs": len({r for r, _ in d})} for s, d in listing.items()}


def status_line(hc, ran):
    import pandas as pd
    head = []
    for f, col in (("agent_probes.csv", "auroc"), ("step_probes.csv", "turn_auroc"), ("transfer.csv", "auroc")):
        if (OUT / f).is_file():
            d = pd.read_csv(OUT / f)
            if f == "transfer.csv":
                d = d[(d.features == fname(("all", "think"))) & d.target.str.startswith("team workers, cheating used")]
            else:
                d = d[(d.features == fname(("all", "think"))) & ((d.setting == SETTING["A"]) if "setting" in d else True)]
            for _, r in d.iterrows():
                head.append(f"{str(r.get('probe', r.get('train')))[:14]} {fmt(r[col])}")
    line = (f"- {time.strftime('%Y-%m-%d %H:%M %Z')} | Hub agents " + ", ".join(f"{s} {v['agents']}" for s, v in hc.items())
            + " | fetched " + ", ".join(f"{s} {n}" for s, n in local_counts().items())
            + " | boundary (turns.safetensors) fetched: " + ", ".join(f"{s} {n}" for s, n in boundary_counts().items())
            + f" | ran: {', '.join(ran) or 'none yet'} | A think pair AUROC: " + "; ".join(head))
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")
    return line


def run_board(hc):
    labels()
    res = run_board_seed()
    write_run_log({"stage": "board_seed probes 1-5", "hub_agents": {k: v["agents"] for k, v in hc.items()},
                   "local_agents": local_counts(), "boundary_agents": boundary_counts(), **res})
    return res


def run_transfer_stage(hc):
    tr, others, tcounts = run_transfer()
    if len(tr):
        tr.to_csv(OUT / "transfer.csv", index=False)
        json.dump(tcounts, open(OUT / "transfer_counts.json", "w"), indent=1)
    if len(others):
        others.to_csv(OUT / "transfer_other_sets_scores.csv", index=False)
    write_run_log({"stage": "transfer 6", "hub_agents": {k: v["agents"] for k, v in hc.items()},
                   "local_agents": local_counts(), "transfer_counts": tcounts,
                   "other_sets_scored": sorted(set(others.set)) if len(others) else []})
    return tcounts


def saved_transfer_counts():
    f = OUT / "transfer_counts.json"
    return json.load(open(f)) if f.is_file() else None


def run_all(hc):
    run_board(hc)
    run_transfer_stage(hc)
    write_report(hc, saved_transfer_counts())
    return ["1-5 board_seed", "6 transfer"]


def watch(poll=600, status_every=900, controls_step=100, resume=True):
    """Every poll seconds: list the Hub; fetch board_seed (think and boundary files) and re-run probes 1-5 when its
    counts change; fetch controls_600 and re-run the transfer when it grows by controls_step agents or completes; then
    the other sets one by one, re-running the transfer scoring when one changes. A thread writes the status line every
    status_every seconds."""
    import threading
    state = {"hc": {}, "ran": [], "last_board": None, "last_ctl": 0, "last_other": {}}
    if resume and (OUT / "run_log.jsonl").is_file():
        # start from the last completed runs: board_seed counts of the last probe run, controls of the last transfer
        rl = [json.loads(x) for x in open(OUT / "run_log.jsonl")]
        b = [r for r in rl if r.get("stage", "").startswith(("board_seed", "run"))]
        t = [r for r in rl if r.get("stage", "").startswith(("transfer", "run"))]
        if b:
            state["last_board"] = (b[-1]["local_agents"].get("board_seed", 0), (b[-1].get("boundary_agents") or {}).get("board_seed", 0))
            state["ran"].append("1-5 board_seed")
        if t:
            state["last_ctl"] = t[-1]["local_agents"].get("controls_600", 0)
            state["ran"].append("6 transfer")
        log(f"watch resumes from {state}")

    def status_loop():
        while True:
            try:
                if state["hc"]:
                    status_line(state["hc"], state["ran"])
            except Exception as e:  # noqa: BLE001
                log("status error", repr(e))
            time.sleep(status_every)
    threading.Thread(target=status_loop, daemon=True).start()
    while True:
        try:
            listing, _ = hub_listing()
            hc = hub_counts_of(listing)
            state["hc"] = hc
            fetch(["board_seed"], listing)
            bkey = (local_counts().get("board_seed", 0), boundary_counts().get("board_seed", 0))
            if bkey != state["last_board"]:
                run_board(hc)
                state["last_board"] = bkey
                state["ran"] = sorted(set(state["ran"]) | {"1-5 board_seed"})
                write_report(hc, saved_transfer_counts())
            fetch(["controls_600"], listing)
            nc, hub_c = local_counts().get("controls_600", 0), hc.get("controls_600", {}).get("agents", 0)
            if nc and (nc - state["last_ctl"] >= controls_step or (nc >= hub_c and nc != state["last_ctl"])):
                run_transfer_stage(hc)
                state["last_ctl"] = nc
                state["ran"] = sorted(set(state["ran"]) | {"6 transfer"})
                write_report(hc, saved_transfer_counts())
            others = [x for x in SET_ORDER[2:] if x in listing] + sorted(x for x in listing if x not in SET_ORDER)
            for o in others:
                fetch([o], listing)
                n = local_counts().get(o, 0)
                if n and n != state["last_other"].get(o):
                    run_transfer_stage(hc)
                    state["last_other"][o] = n
                    write_report(hc, saved_transfer_counts())
        except Exception as e:  # noqa: BLE001
            import traceback
            log("watch error", repr(e), traceback.format_exc()[-1500:])
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    if len(sys.argv) == 5 and sys.argv[1] == "fetch-one":
        print(json.dumps(fetch_agent(*sys.argv[2:5])))
        return
    if len(sys.argv) == 5 and sys.argv[1] == "fetch-turns":
        print(json.dumps(fetch_turns(*sys.argv[2:5])))
        return
    ap.add_argument("stage", choices=["fetch", "labels", "run", "watch", "report"])
    ap.add_argument("--sets", default="board_seed")
    args = ap.parse_args()
    if args.stage == "fetch":
        fetch(args.sets.split(","))
        print(local_counts())
    elif args.stage == "labels":
        labels()
    elif args.stage == "run":
        listing, _ = hub_listing()
        hc = hub_counts_of(listing)
        ran = run_all(hc)
        log(status_line(hc, ran))
    elif args.stage == "report":
        listing, _ = hub_listing()
        write_report(hub_counts_of(listing), saved_transfer_counts())
    elif args.stage == "watch":
        watch()


if __name__ == "__main__":
    main()
