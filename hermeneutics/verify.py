"""Check an activation cache: every agent folder is opened, its tensors read, and the invariants tested.

Per agent: meta.json present; acts.safetensors holds think_L<L> [n_turns, 2, 5120] and mean_L<L> [n_units, 5120] fp16
for every layer in meta; every value finite; ids[think_pos[:, 0]] == 248068 and ids[think_pos[:, 1]] == 248069; n_turns
equals meta's turns (minus turns_dropped by truncation); n_units equals the spans.jsonl line count. Exit 1 on any
failure; prints a per-author table of agents, turns and tokens.

    python -m hermeneutics.verify <cache_dir> [--expect-runs data/subsample.csv]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from safetensors import safe_open

from .think import CLOSE, OPEN


def check_agent(d):
    """List of problems (empty when the folder is sound) and a summary dict."""
    problems = []
    meta = json.load(open(os.path.join(d, "meta.json")))
    n_units = sum(1 for _ in open(os.path.join(d, "spans.jsonl")))
    with safe_open(os.path.join(d, "acts.safetensors"), framework="np") as f:
        keys = set(f.keys())
        ids = f.get_tensor("ids")
        pos = f.get_tensor("think_pos")
        readable = f.get_tensor("think_readable")
        n_turns = pos.shape[0]
        if pos.shape != (n_turns, 2):
            problems.append(f"think_pos shape {pos.shape}")
        if n_turns and not (np.all(ids[pos[:, 0]] == OPEN) and np.all(ids[pos[:, 1]] == CLOSE)):
            problems.append("think_pos does not point at <think>/</think> ids")
        if readable.shape != (n_turns,):
            problems.append(f"think_readable shape {readable.shape}")
        want = meta["turns"] - meta.get("turns_dropped", 0)
        if n_turns != want:
            problems.append(f"n_turns {n_turns} != turns {want}")
        if len(ids) != meta["n_tokens"]:
            problems.append(f"ids {len(ids)} != n_tokens {meta['n_tokens']}")
        if n_units != meta["n_units"]:
            problems.append(f"spans {n_units} != n_units {meta['n_units']}")
        for L in meta["layers"]:
            for name, shape in ((f"think_L{L}", (n_turns, 2, 5120)), (f"mean_L{L}", (n_units, 5120))):
                if name not in keys:
                    problems.append(f"missing {name}")
                    continue
                x = f.get_tensor(name)
                if x.dtype != np.float16:
                    problems.append(f"{name} dtype {x.dtype}")
                if x.shape != shape:
                    problems.append(f"{name} shape {x.shape} != {shape}")
                if not np.isfinite(x.astype(np.float32)).all():
                    problems.append(f"{name} non-finite")
                if x.size and float(np.abs(x.astype(np.float32)).mean()) == 0.0:
                    problems.append(f"{name} all zeros")
    return problems, {"run": meta["run"], "author": meta["author"], "condition": meta["condition"], "n_turns": n_turns,
                      "n_tokens": meta["n_tokens"], "n_readable": int(readable.sum()), "truncated": "truncated_at" in meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--expect-runs", default=None, help="CSV with a run column: report which are not yet in the cache")
    args = ap.parse_args()
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.cache, "*", "*", "meta.json")))
    bad, rows = 0, []
    for d in dirs:
        try:
            problems, s = check_agent(d)
        except Exception as e:  # noqa: BLE001
            problems, s = [f"{type(e).__name__}: {e}"], None
        if problems:
            bad += 1
            print("BROKEN", d, "; ".join(problems))
        if s:
            rows.append(s)
    import pandas as pd
    t = pd.DataFrame(rows)
    if len(t):
        g = t.groupby(["author", "condition"]).agg(agents=("run", "count"), turns=("n_turns", "sum"), tokens=("n_tokens", "sum"),
                                                  readable=("n_readable", "sum"), truncated=("truncated", "sum"))
        print(g)
    print(f"{len(dirs)} agent folders, {bad} broken, {len(t)} sound; turns {int(t.n_turns.sum()) if len(t) else 0}, "
          f"tokens {int(t.n_tokens.sum()) if len(t) else 0}")
    if args.expect_runs:
        want = set(pd.read_csv(args.expect_runs).run)
        have = set(t.run) if len(t) else set()
        print(f"expected {len(want)} runs: {len(want & have)} present, {len(want - have)} missing, {len(have - want)} extra")
    half = [d for d in sorted(glob.glob(os.path.join(args.cache, "*", "*"))) if os.path.isdir(d) and not os.path.exists(os.path.join(d, "meta.json"))]
    if half:
        print(f"{len(half)} folders without meta.json (interrupted writes): {half[:3]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
