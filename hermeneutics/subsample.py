"""Stratified subsample of runs for extraction: about N per author, balanced over condition x binary label.

Strata are author x condition x label (cheat / no cheat). Each author gets N runs: every stratum is filled to N/4, and
what a short stratum cannot fill goes first to the solo arm (the clean test: no cheating instruction in the prompt),
then to the cheat arm. Inside a stratum runs are taken round-robin over tasks, in a seeded random order inside each
task. Runs longer than --max-tokens are left out. The output order interleaves the strata so that every extraction
batch holds all of them.

    python -m hermeneutics.subsample [--per-author 120] [--max-tokens 100000] [--out data/subsample.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd

from . import REPO


def select(labels, summary, per_author=120, max_tokens=100_000, seed=0):
    d = labels[labels.valid & labels.label.isin(["cheat", "no cheat"])].merge(summary[["run", "n_tokens", "turns", "n_readable"]], on="run")
    d = d[d.n_tokens <= max_tokens].copy()
    rng = np.random.default_rng(seed)
    d["r"] = rng.random(len(d))
    d = d.sort_values(["author", "condition", "label", "task", "r"])
    chosen = []
    for author, da in d.groupby("author"):
        pools = {}
        for (cond, lab), g in da.groupby(["condition", "label"]):
            # round-robin over tasks
            by_task = [list(t.run) for _, t in g.groupby("task")]
            order = []
            for i in range(max(map(len, by_task))):
                order += [t[i] for t in by_task if i < len(t)]
            pools[(cond, lab)] = order
        quota = {k: min(len(v), per_author // 4) for k, v in pools.items()}
        left = per_author - sum(quota.values())
        for cond in ("solo", "cheat"):  # the solo arm first, its two labels alternately
            while left > 0 and any(quota[k] < len(pools[k]) for k in pools if k[0] == cond):
                for lab in ("no cheat", "cheat"):
                    k = (cond, lab)
                    if left > 0 and k in pools and quota[k] < len(pools[k]):
                        quota[k] += 1
                        left -= 1
        for k, q in quota.items():
            for run in pools[k][:q]:
                chosen.append((author, k[0], k[1], run))
    sel = pd.DataFrame(chosen, columns=["author", "condition", "label", "run"])
    # interleave strata: the i-th run of every stratum before the (i+1)-th of any
    sel["rank"] = sel.groupby(["author", "condition", "label"]).cumcount()
    sel = sel.sort_values(["rank", "author", "condition", "label"]).drop(columns="rank")
    return sel.merge(d[["run", "task", "seed", "n_tokens", "turns", "n_readable", "rung", "cheat"]], on="run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(REPO, "data/labels.csv"))
    ap.add_argument("--summary", default=os.path.join(REPO, "data/render_summary.csv"))
    ap.add_argument("--per-author", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=1500, help="tokens/s for the time estimate")
    ap.add_argument("--out", default=os.path.join(REPO, "data/subsample.csv"))
    args = ap.parse_args()
    sel = select(pd.read_csv(args.labels), pd.read_csv(args.summary), args.per_author, args.max_tokens, args.seed)
    sel.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    t = sel.pivot_table(index=["author", "condition"], columns="label", values="run", aggfunc="count", fill_value=0)
    t["tokens"] = sel.groupby(["author", "condition"]).n_tokens.sum()
    t["tasks"] = sel.groupby(["author", "condition"]).task.nunique()
    print(t)
    print(f"{len(sel)} runs, {sel.n_tokens.sum():,} tokens, {sel.turns.sum():,} turns; "
          f"about {sel.n_tokens.sum() / args.rate / 3600:.1f} h at {args.rate:.0f} tokens/s; longest {sel.n_tokens.max():,}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
