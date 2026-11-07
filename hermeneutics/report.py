"""RESULTS.md from the probe outputs (auroc.csv, author_probe.csv, agents.csv) of hermeneutics/results/.

    python -m hermeneutics.report hermeneutics/results
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from .runs import AUTHOR_NAME

ERASED = ["raw", "author erased", "author, task, condition erased"]


def f3(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"


def cell(r):
    return f"{f3(r.auroc)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]"


def auroc_table(d, arm, pooling):
    """Scope x features table for one arm and pooling, at the best layer/position by pooled raw AUROC."""
    x = d[(d.arm == arm) & (d.pooling == pooling)]
    if x.empty:
        return [], None
    best = x[(x.features == "raw") & (x.scope == "pooled")].sort_values("auroc", ascending=False).iloc[0]
    L, pos = best.layer, best.position
    x = x[(x.layer == L) & (x.position == pos)]
    feats = [f for f in ERASED if f in set(x.features)]
    out = [f"Layer {L}, position `{pos}` (the configuration with the highest pooled raw AUROC in this arm; every "
           f"configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].\n",
           "| scope | cheat / no cheat | " + " | ".join(feats) + " |", "|---|---|" + "---|" * len(feats)]
    scopes = ["pooled"] + sorted(s for s in set(x.scope) if not s.startswith("LOTO") and s != "pooled") + \
             sorted(s for s in set(x.scope) if s.startswith("LOTO"))
    for s in scopes:
        rs = {f: x[(x.scope == s) & (x.features == f)] for f in feats}
        r0 = rs[feats[0]]
        if r0.empty:
            continue
        r0 = r0.iloc[0]
        name = {"pooled": "All authors, pooled probe (B.I)"}.get(s, f"Leave out: {AUTHOR_NAME.get(s[5:], s[5:])} (B.II)" if s.startswith("LOTO")
                                                                 else f"Within: {AUTHOR_NAME.get(s, s)} (A)")
        cells = [cell(rs[f].iloc[0]) if not rs[f].empty else "n/a" for f in feats]
        extra = f" turns {f3(r0.auroc_turns)}" if pooling == "turn" and pd.notna(r0.auroc_turns) else ""
        out.append(f"| {name}{extra} | {r0.cheat} / {r0.no_cheat} | " + " | ".join(cells) + " |")
    return out, (L, pos)


def sweep_table(d, arm, pooling):
    """Pooled raw AUROC and the within-author mean for every layer x position."""
    x = d[(d.arm == arm) & (d.pooling == pooling) & (d.features == "raw") & (d.scope == "pooled")]
    if x.empty:
        return []
    out = ["| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |", "|---|---|---|---|---|---|"]
    for _, r in x.sort_values(["layer", "position"]).iterrows():
        out.append(f"| {r.layer} | {r.position} | {cell(r)} | {f3(r.within_mean)} | {f3(r.null_95)} | "
                   f"{f3(r.null_p) if pd.notna(r.null_p) else 'n/a'} |")
    return out


def author_table(ap, arm, L, pos):
    x = ap[(ap.arm == arm) & (ap.pooling == "agent") & (ap.layer.astype(str) == str(L)) & (ap.position == pos)]
    if x.empty:
        return []
    out = [f"Majority-class rate {f3(x.iloc[0].majority_rate)}.\n", "| features | accuracy | balanced accuracy |", "|---|---|---|"]
    for _, r in x.iterrows():
        out.append(f"| {r.features} | {f3(r.accuracy)} | {f3(r.balanced_accuracy)} |")
    raw = x[x.features == "raw"]
    if not raw.empty:
        cm = json.loads(raw.iloc[0].confusion)
        res = json.load(open(os.path.join(os.path.dirname(ap.attrs["path"]), arm, f"L{L}_{pos}_agent", "results.json")))
        T = [AUTHOR_NAME.get(t, t) for t in res["authors"]]
        out += ["", "Confusion matrix, raw features (rows true, columns predicted):\n", "| true \\ predicted | " + " | ".join(T) + " |",
                "|---|" + "---|" * len(T)]
        for t, row in zip(T, cm):
            out.append(f"| {t} | " + " | ".join(str(v) for v in row) + " |")
    return out


def main(path):
    d = pd.read_csv(os.path.join(path, "auroc.csv"))
    ap = pd.read_csv(os.path.join(path, "author_probe.csv"))
    ap.attrs["path"] = os.path.join(path, "author_probe.csv")
    ag = pd.read_csv(os.path.join(path, "agents.csv"))
    out = ["# Results: a hermeneutic cheating probe on the controls subsample\n",
           "Reader: Qwen3.8-27B, teacher-forced over each author's transcript. Features: the reader's residual stream at the "
           "`<think>` (open) and `</think>` (close) token of every assistant turn, at layers 26, 38 and 51 (HF "
           "`hidden_states` numbering, 64 blocks), averaged over the agent's turns (pooling `agent`) or one row per turn "
           "(pooling `turn`, scored per turn and as the mean turn score per agent). Probe: training-fold PCA to 256 "
           "components, z-scoring, L2 logistic regression; 5 folds grouped by run and stratified by author x label.\n",
           "The **solo arm** is the primary test: no cheating instruction in the prompt, and both classes occur for every "
           "author. In the **cheat arm** the prompt asks for the shortcut and nearly every author takes it, so a probe "
           "pooled over both arms can read the condition from the prompt instead of the cheating; the pooled arm is "
           "reported with the condition erased for that reason. The author-identity probe and the LEACE erasure expose "
           "the author signature that think coverage alone creates (qwen 1.00, ds 0.88, gpt5m 0.48, luna 0.34, haiku 0.00 "
           "of turns carry readable reasoning; Haiku's think blocks are all empty).\n",
           "## Agents in the cache\n", "| author | condition | cheat | no cheat | turns | readable turns |", "|---|---|---|---|---|---|"]
    for (a, c), g in ag.groupby(["author", "condition"]):
        out.append(f"| {AUTHOR_NAME.get(a, a)} | {c} | {int((g.y == 1).sum())} | {int((g.y == 0).sum())} | {int(g.n_turns.sum())} | {int(g.n_readable.sum())} |")
    for arm, title in (("solo", "Solo arm (primary)"), ("pooled", "Both arms pooled (secondary)"), ("cheat", "Cheat arm (secondary)")):
        if arm not in set(d.arm):
            continue
        out += ["", f"## {title}\n"]
        for pooling in ("agent", "turn"):
            t, lp = auroc_table(d, arm, pooling)
            if not t:
                continue
            out += [f"### Cheating AUROC, pooling `{pooling}`\n"] + t + ["", f"Sweep over layers and positions, pooling `{pooling}`, raw features:\n"] + sweep_table(d, arm, pooling)
            if pooling == "agent" and lp is not None:
                out += ["", "### Author-identity probe\n"] + author_table(ap, arm, *lp)
                cp = os.path.join(path, "condition_probe.csv")
                if arm == "pooled" and os.path.exists(cp):
                    c = pd.read_csv(cp)
                    c = c[(c.arm == arm) & (c.pooling == "agent") & (c.layer.astype(str) == str(lp[0])) & (c.position == lp[1])]
                    if not c.empty:
                        out += ["", "### Condition readout (cheat-arm prompt vs solo prompt)\n",
                                f"Majority-class rate {f3(c.iloc[0].majority_rate)}.\n", "| features | accuracy |", "|---|---|"]
                        out += [f"| {r.features} | {f3(r.accuracy)} |" for _, r in c.iterrows()]
                sh = d[(d.arm == arm) & (d.pooling == "agent") & (d.layer.astype(str) == str(lp[0])) & (d.position == lp[1]) & d.null_p.notna()]
                if not sh.empty:
                    out += ["", "### Label-permutation null (labels shuffled within author)\n",
                            "| features | scope | real AUROC | null mean | null 95th pct | p |", "|---|---|---|---|---|---|"]
                    for _, r in sh.iterrows():
                        out.append(f"| {r.features} | {r.scope} | {f3(r.auroc)} | {f3(r.null_mean)} | {f3(r.null_95)} | {r.null_p:.3f} |")
            out.append("")
    open(os.path.join(path, "RESULTS.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
