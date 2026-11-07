"""Run list over the controls data: one row per run_name, the canonical copy chosen among the hosts.

Layout (read-only): <DATA_ROOT>/<host>/runs/<run_name>/{run.json, episodes.jsonl, signals.jsonl, trajectories/*/transcript.json}
and <DATA_ROOT>/<host>/logs/<run_name>/*.eval. run_name = <task>_<cell>_<model>_s<seed>, cell in {solo, cheat}, model in
{ds, gpt5m, haiku, luna, qwen}. A run_name can have copies on several hosts (a concurrent-orchestrator restart bug).

Validity and the canonical choice follow the run driver's rules (arena-capstone publish_apart_swarm/build_snapshot.py):
  valid      episodes.jsonl line 0 parses, ok == true, trajectories/ holds bytes, signals.jsonl has a team row
  canonical  among the valid copies: ended_by submit > end > other, then the earliest .eval launch time, then host name

    python -m hermeneutics.runs [--data-root D] [--out data/runs.csv]
"""
import argparse
import glob
import json
import os
from datetime import datetime

import pandas as pd

from . import DATA_ROOT

AUTHORS = ("ds", "gpt5m", "haiku", "luna", "qwen")
CELLS = ("solo", "cheat")
AUTHOR_MODEL = {"ds": "deepseek/deepseek-v4.1-flash-1017", "gpt5m": "openai/gpt-5-mini", "haiku": "anthropic/claude-haiku-5.5",
                "luna": "openai/gpt-6-luna", "qwen": "qwen/qwen3.8-27b"}
AUTHOR_NAME = {"ds": "DeepSeek V4.1 Flash", "gpt5m": "GPT-5 mini", "haiku": "Claude Haiku 5.5", "luna": "GPT-6 Luna",
               "qwen": "Qwen3.8-27B"}
# run_name task prefix -> the task name run.json carries (the cheat ladder's threshold is per task name, labels.py)
TASK_NAME = {"binder": "protein_design", "factor": "prime_factorization", "flipt": "software_engineering", "game": "chess",
             "poem": "creative_writings", "riemann": "openmath_agent", "subset": "subset_sum", "vuls": "software_engineering",
             "worktime": "knowledge_work"}


def parse(run_name):
    """(task, cell, author, seed) or None."""
    p = run_name.split("_")
    if len(p) < 4 or not p[-1].startswith("s") or not p[-1][1:].isdigit() or p[-2] not in AUTHORS or p[-3] not in CELLS:
        return None
    return "_".join(p[:-3]), p[-3], p[-2], int(p[-1][1:])


def _launch_time(eval_log, run_dir):
    if eval_log:
        try:
            return datetime.strptime(os.path.basename(eval_log)[:19], "%Y-%m-%dT%H-%M-%S").timestamp()
        except ValueError:
            pass
    mts = [os.path.getmtime(p) for p in glob.glob(os.path.join(run_dir, "**", "*"), recursive=True) if os.path.isfile(p)]
    return min(mts) if mts else 0.0


def eval_log_of(run_dir, logs_dir):
    """The run's .eval: the one run.json names when it is there, else the largest in the logs folder."""
    try:
        name = os.path.basename(json.load(open(os.path.join(run_dir, "run.json"))).get("eval_log") or "")
    except (OSError, ValueError):
        name = ""
    if name and os.path.exists(os.path.join(logs_dir, name)):
        return os.path.join(logs_dir, name)
    logs = glob.glob(os.path.join(logs_dir, "*.eval"))
    return max(logs, key=os.path.getsize) if logs else None


def inspect_copy(run_dir, data_root):
    """One copy of a run: validity gates and what the label table needs."""
    host = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    run = os.path.basename(run_dir)
    d = {"run": run, "host": host, "run_dir": run_dir, "valid": False, "reason": "", "ended_by": None, "ok": None,
         "task_name": None, "model_id": None, "n_turns": None, "n_tool_calls": None, "score": None, "agent_id": None,
         "eval_log": eval_log_of(run_dir, os.path.join(data_root, host, "logs", run)), "n_trajectories": 0}
    try:
        lines = [ln for ln in open(os.path.join(run_dir, "episodes.jsonl")).read().splitlines() if ln.strip()]
        e0 = json.loads(lines[0])
    except (OSError, ValueError, IndexError) as ex:
        d["reason"] = f"episodes unreadable: {type(ex).__name__}"
        return d
    d.update(ok=e0.get("ok"), ended_by=e0.get("ended_by"), task_name=e0.get("task"), model_id=e0.get("model"),
             n_turns=e0.get("n_turns"), n_tool_calls=e0.get("n_tool_calls"), score=json.dumps(e0.get("score")),
             agent_id=e0.get("orbit_agent"))
    trajs = glob.glob(os.path.join(run_dir, "trajectories", "*", "transcript.json"))
    d["n_trajectories"] = len(trajs)
    d["launch_time"] = _launch_time(d["eval_log"], run_dir)
    sig = os.path.join(run_dir, "signals.jsonl")
    team = agent = None
    if os.path.exists(sig):
        for ln in open(sig):
            if ln.strip():
                row = json.loads(ln)
                if row.get("row") == "team" and team is None:
                    team = row
                elif row.get("row") == "agent" and agent is None:
                    agent = row
    d["team_row"], d["agent_row"] = team, agent
    if not e0.get("ok"):
        d["reason"] = f"ok={e0.get('ok')} ended_by={e0.get('ended_by')}"
    elif sum(os.path.getsize(p) for p in trajs) == 0:
        d["reason"] = "empty trajectories"
    elif team is None:
        d["reason"] = "no team row in signals.jsonl"
    elif d["eval_log"] is None:
        d["reason"] = "no .eval log"
    else:
        d["valid"] = True
    return d


def _canon_key(c):
    return ({"submit": 0, "end": 1}.get(c["ended_by"], 2), c.get("launch_time", 0.0), c["host"])


def run_list(data_root=DATA_ROOT, with_rows=False):
    """One dict per run_name: the canonical copy's fields plus n_copies, n_valid_copies, hosts. Invalid-only run_names
    are included with valid=False (their first copy). with_rows keeps the signals rows (labels.py)."""
    groups = {}
    for rd in sorted(glob.glob(os.path.join(data_root, "*", "runs", "*"))):
        run = os.path.basename(rd)
        if not os.path.isdir(rd) or parse(run) is None:
            continue
        groups.setdefault(run, []).append(inspect_copy(rd, data_root))
    out = []
    for run in sorted(groups):
        copies = groups[run]
        valid = [c for c in copies if c["valid"]]
        c = min(valid, key=_canon_key) if valid else copies[0]
        task, cell, author, seed = parse(run)
        row = {"run": run, "task": task, "task_name": c["task_name"] or TASK_NAME.get(task), "condition": cell,
               "author": author, "seed": seed, **{k: v for k, v in c.items() if k not in ("run", "team_row", "agent_row")},
               "n_copies": len(copies), "n_valid_copies": len(valid), "hosts": "|".join(x["host"] for x in copies)}
        if with_rows:
            row["team_row"], row["agent_row"] = c["team_row"], c["agent_row"]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = run_list(args.data_root)
    d = pd.DataFrame(rows)
    print(f"{len(d)} run_names, {d.n_copies.sum()} copies, {int(d.valid.sum())} valid; duplicated names {(d.n_copies > 1).sum()}")
    print(d.groupby(["author", "condition"]).valid.agg(["size", "sum"]).rename(columns={"size": "runs", "sum": "valid"}))
    print("invalid reasons:", d[~d.valid].reason.value_counts().to_dict())
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        d.to_csv(args.out, index=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
