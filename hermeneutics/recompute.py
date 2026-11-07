"""Recompute every AUROC and bootstrap interval of a results tree from the stored per-agent scores, and compare.

For each <arm>/<config>/results.json with a scores.csv beside it: every "cheat" row's AUROC is recomputed from the
column score|<features>|<scope> over the rows with a finite score, and its 95% bootstrap over runs with the same seed
as probes.boot_ci. Exit 1 on any mismatch beyond 1e-9 (AUROC) or 1e-9 (CI; the bootstrap is seeded).

    python -m hermeneutics.recompute hermeneutics/results/pca_agent [more trees ...]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from .probes import auc, boot_ci


def check_tree(root):
    n, bad = 0, 0
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "results.json"))):
        sp = os.path.join(os.path.dirname(p), "scores.csv")
        if not os.path.exists(sp):
            print("NO SCORES", p)
            bad += 1
            continue
        res = json.load(open(p))
        sc = pd.read_csv(sp)
        y, run = sc.y.values.astype(int), sc.run.values
        for r in res["cheat"]:
            col = f"score|{r['features']}|{r['scope']}"
            if col not in sc.columns:
                print("MISSING COLUMN", p, col)
                bad += 1
                continue
            s = sc[col].values.astype(float)
            ok = np.isfinite(s)
            n += 1
            a = auc(s[ok], y[ok]) if ok.any() else np.nan
            same = (np.isnan(a) and np.isnan(r["auroc"])) or abs(a - r["auroc"]) < 1e-9
            counts = ok.sum() == r["agents"] and y[ok].sum() == r["cheat"] and (1 - y[ok]).sum() == r["no_cheat"]
            ci = boot_ci(s[ok], y[ok], run[ok]) if ok.any() else [np.nan, np.nan]
            ci_same = all((np.isnan(x) and np.isnan(z)) or abs(x - z) < 1e-9 for x, z in zip(ci, r["ci"]))
            if not (same and counts and ci_same):
                bad += 1
                print(f"MISMATCH {p} {r['features']} | {r['scope']}: stored {r['auroc']:.6f} {r['ci']} n={r['agents']}; "
                      f"recomputed {a:.6f} {ci} n={int(ok.sum())}")
    return n, bad


def main():
    tot_n, tot_bad = 0, 0
    for root in sys.argv[1:]:
        n, bad = check_tree(root)
        print(f"{root}: {n} AUROC cells recomputed from scores, {bad} problems")
        tot_n += n
        tot_bad += bad
    print(f"total {tot_n} cells, {tot_bad} problems")
    sys.exit(1 if tot_bad or not tot_n else 0)


if __name__ == "__main__":
    main()
