"""hermeneutics/results/RESULTS.md: the headline tables of the probe runs, every number read from the result files.

    python -m hermeneutics.summary [hermeneutics/results]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .runs import AUTHOR_NAME

ERASED = ["raw", "author erased", "author, task, condition erased"]


def f3(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"


def cell(r):
    return f"{f3(r.auroc)} [{f3(r.ci_lo)}, {f3(r.ci_hi)}]"


def best_config(d, arm):
    x = d[(d.arm == arm) & (d.features == "raw") & (d.scope == "pooled")].sort_values("auroc", ascending=False).iloc[0]
    return x.layer, x.position


def scope_name(s, pooled="B.I pooled, held-out runs"):
    if s == "pooled":
        return pooled
    if s.startswith("LOTO "):
        return f"B.II leave out {AUTHOR_NAME[s[5:]]}"
    return f"A within {AUTHOR_NAME[s]}"


def scopes_of(x):
    authors = sorted(s for s in set(x.scope) if not s.startswith("LOTO") and s != "pooled")
    return ["pooled"] + authors + ["LOTO " + a for a in authors]


def scope_table(d, arm, L, pos, feats):
    x = d[(d.arm == arm) & (d.layer.astype(str) == str(L)) & (d.position == pos)]
    out = ["| probe | cheat / no cheat | " + " | ".join(feats) + " |", "|---|---|" + "---|" * len(feats)]
    for s in scopes_of(x):
        rs = {f: x[(x.scope == s) & (x.features == f)] for f in feats}
        if rs[feats[0]].empty:
            continue
        r0 = rs[feats[0]].iloc[0]
        out.append(f"| {scope_name(s)} | {r0.cheat} / {r0.no_cheat} | "
                   + " | ".join(cell(rs[f].iloc[0]) if not rs[f].empty else "n/a" for f in feats) + " |")
    return out


def sweep(d, arm, features="raw"):
    x = d[(d.arm == arm) & (d.features == features) & (d.scope == "pooled")].sort_values(["layer", "position"])
    out = ["| layer | position | B.I pooled AUROC | within-author mean | null 95th pct |", "|---|---|---|---|---|"]
    for _, r in x.iterrows():
        out.append(f"| {r.layer} | {r.position} | {cell(r)} | {f3(r.within_mean)} | {f3(r.null_95)} |")
    return out


def length_baseline(ag, arm="solo"):
    """Held-out AUROC of a logistic regression on turns, log tokens and readable fraction, same fold design as the probes."""
    a = ag[ag.condition == arm].reset_index(drop=True)
    y = a.y.values
    strata = (a.author + "|" + a.y.astype(str)).values
    F = np.column_stack([a.n_turns, np.log(a.n_tokens), a.n_readable / a.n_turns]).astype(float)
    F = (F - F.mean(0)) / F.std(0)
    p = np.full(len(a), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, strata, a.run):
        p[te] = LogisticRegression(C=1.0, max_iter=5000).fit(F[tr], y[tr]).decision_function(F[te])
    return float(roc_auc_score(y, p))


def main(path):
    P = lambda *a: os.path.join(path, *a)  # noqa: E731
    pa = pd.read_csv(P("pca_agent", "auroc.csv"))
    ap = pd.read_csv(P("pca_agent", "author_probe.csv"))
    cp = pd.read_csv(P("pca_agent", "condition_probe.csv"))
    pt = pd.read_csv(P("pca_turn", "auroc.csv"))
    po = pd.read_csv(P("pca_turn_onset", "auroc.csv"))
    ag = pd.read_csv(P("pca_agent", "agents.csv"))
    fa = pd.read_csv(P("full_agent", "auroc.csv")) if os.path.exists(P("full_agent", "auroc.csv")) else None
    L, pos = best_config(pa, "solo")
    res = json.load(open(P("pca_agent", "solo", f"L{L}_{pos}_agent", "results.json")))
    nuis, shuf = res["nuisance"], res["shuffle"]
    apx = ap[(ap.arm == "solo") & (ap.layer.astype(str) == str(L)) & (ap.position == pos)].set_index("features")
    Lp, posp = best_config(pa, "pooled")
    cpx = cp[(cp.arm == "pooled") & (cp.layer.astype(str) == str(Lp)) & (cp.position == posp)].set_index("features")
    Lc, posc = best_config(pa, "cheat")
    Lt, post = best_config(pt, "solo")
    ptx = pt[(pt.arm == "solo") & (pt.layer == Lt) & (pt.position == post) & (pt.scope == "pooled") & (pt.features == "raw")].iloc[0]
    pox = po[(po.arm == "solo") & (po.features == "raw") & (po.scope == "pooled")].sort_values("auroc").iloc[-1]
    names = [AUTHOR_NAME[a] for a in sorted(ag.author.unique())]
    authors = ", ".join(names[:-1]) + " and " + names[-1]

    out = ["# A hermeneutic cheating probe: results on the controls subsample\n",
           "We ask whether an open-weight **reader** (Qwen3.8-27B) that only reads another model's transcript carries a linear "
           "readout of whether that model cheated. This is hard. The reader also encodes who wrote the text, which task it "
           "was, and whether the prompt asked for the shortcut. Any of these can stand in for cheating. We teacher-force each "
           "author's transcript through the reader and keep its residual stream at the `<think>` and `</think>` token of "
           "every assistant turn, at three depths. We fit L2 logistic probes on training-fold PCA features and hold out by "
           "run. We erase the author with **LEACE**, a closed-form linear concept eraser, and we erase author, task and "
           "condition together. We shuffle labels inside each author for a null. We therefore contend that the reader's "
           "think-position states separate cheating from honest work inside every author, across authors, and after the "
           "author is erased.\n",
           "## What was probed\n",
           f"The authors are {authors}. We probe {len(ag)} of the valid controls runs, one agent per run, balanced over "
           "condition and label where the data allow. The **solo arm** gives no cheating instruction in the prompt and is the "
           "primary test. The **cheat arm** asks for the shortcut, and nearly every author takes it. Labels follow the "
           "dataset's cheat ladder. Cheating starts at `opened`, or at `tried` on software engineering, chess and creative "
           "writing. The rungs `none` and `exposed` are honest. The rungs between are left out. The share of turns with "
           "readable reasoning differs by author (last column), so an author signature is built into the features.\n",
           "| author | condition | cheat | no cheat | turns | readable share |", "|---|---|---|---|---|---|"]
    for (a, c), g in ag.groupby(["author", "condition"]):
        out.append(f"| {AUTHOR_NAME[a]} | {c} | {int((g.y == 1).sum())} | {int((g.y == 0).sum())} | {int(g.n_turns.sum())} | "
                   f"{g.n_readable.sum() / g.n_turns.sum():.2f} |")
    out += ["", "## Does the reader separate cheating in the solo arm?\n",
            "The reader separates cheating in every author and across authors. Table 1 gives the agent-level probe (mean over "
            "turns) at the configuration with the highest pooled AUROC. Table 2 sweeps layers and positions. Erasing the "
            "author moves the numbers little. Erasing author, task and condition together moves them a few hundredths. Every "
            "real value sits above every draw of the within-author permutation null (Table 4). We therefore posit that the "
            "signal is neither the author's base rate nor the task.\n",
            f"**Table 1.** Solo arm, agent-level PCA probe, layer {L}, position `{pos}`. Held-out AUROC [95% bootstrap over runs].\n"]
    out += scope_table(pa, "solo", L, pos, ERASED)
    out += ["", "**Table 2.** Solo arm, raw features, every layer and position (agent-level PCA probe).\n"] + sweep(pa, "solo")
    out += ["", "## Is it the author, the task, or the length of the run?\n",
            "The author is almost perfectly readable from the same features and becomes unreadable after erasure (Table 3). "
            "Metadata alone predicts cheating poorly in the pooled solo arm. Inside some authors the task predicts it well. "
            "The probe stays above those baselines after the task is erased (Table 1, last column). Run length predicts "
            "cheating weakly (Table 3, last row). We therefore posit that the probe reads the content of the transcript.\n",
            f"**Table 3.** Solo arm: the author-identity probe (5 classes, majority rate {f3(apx.loc['raw'].majority_rate)}) "
            "and the cheating AUROC of metadata-only baselines, same folds.\n",
            "| readout | value |", "|---|---|",
            f"| author accuracy, raw features | {f3(apx.loc['raw'].accuracy)} |",
            f"| author accuracy, author erased | {f3(apx.loc['author erased'].accuracy)} |",
            f"| author accuracy, author, task, condition erased | {f3(apx.loc['author, task, condition erased'].accuracy)} |"]
    for k, v in nuis.items():
        out.append(f"| cheating AUROC from one-hot {k} | {f3(v)} |")
    out.append(f"| cheating AUROC from run length (turns, log tokens, readable share) | {f3(length_baseline(ag))} |")
    out += ["", f"**Table 4.** Solo arm, labels shuffled inside each author (100 draws), layer {L}, position `{pos}`.\n",
            "| features | scope | real | null mean | null 95th pct | p |", "|---|---|---|---|---|---|"]
    for f, dd in shuf.items():
        for s, v in dd.items():
            out.append(f"| {f} | {AUTHOR_NAME.get(s, s)} | {f3(v['real'])} | {f3(v['null_mean'])} | {f3(v['null_95'])} | {v['p']:.3f} |")
    out += ["", "## Does a probe trained on four authors read the fifth?\n",
            "A probe trained on four authors reads the fifth for every author except GPT-6 Luna (B.II rows of Table 1). Luna "
            "receives the weakest transfer and is also the hardest author within itself. Luna's reasoning is encrypted, and "
            "few of its turns carry a provider summary. We therefore conjecture that transfer depends on how much of the "
            "author's reasoning the reader gets to see.\n",
            "## What about the cheat arm and the pooled set?\n",
            "In the pooled set the prompt condition is readable from the features and falls to the majority rate once "
            "condition is erased (Table 5). The pooled cheating probe keeps most of its AUROC after that erasure (Table 6). "
            "The cheat arm has few honest agents for DeepSeek and none for Qwen, so its within-author rows are thin and its "
            "pooled number is secondary. We therefore posit that the pooled probe does not rest on reading the prompt.\n",
            f"**Table 5.** Both arms pooled, condition readout (cheat-arm prompt vs solo prompt), layer {Lp}, position `{posp}`. "
            f"Majority rate {f3(cpx.loc['raw'].majority_rate)}.\n",
            "| features | accuracy |", "|---|---|"]
    out += [f"| {f} | {f3(cpx.loc[f].accuracy)} |" for f in ERASED if f in cpx.index]
    out += ["", f"**Table 6.** Both arms pooled (layer {Lp}, `{posp}`), then the cheat arm alone (layer {Lc}, `{posc}`), agent-level PCA probe.\n"]
    out += scope_table(pa, "pooled", Lp, posp, ERASED) + [""] + scope_table(pa, "cheat", Lc, posc, ERASED)
    out += ["", "## Does a single turn carry the signal?\n",
            "A single turn carries part of the signal. With every turn labelled by its agent, the per-turn probe separates "
            "turns, and separates agents better once their turn scores are averaged (Table 7). Its turn-position baseline "
            "sits at chance, so position does not carry the signal (Table 8). A second labelling keeps a cheating agent's "
            "turns only from its first cheating tool call on. Its probe and its turn-position baseline reach the same high "
            "value (Table 8). That design makes lateness the label, and we do not read it as detection. We therefore posit "
            "that a turn-level monitor needs position-matched labels before its score can be read.\n",
            f"**Table 7.** Solo arm, per-turn PCA probe, agent-labelled turns, layer {Lt}, position `{post}`. Agent AUROC is "
            "the mean turn score per agent.\n",
            "| probe | cheat / no cheat | agent AUROC | turn AUROC |", "|---|---|---|---|"]
    x = pt[(pt.arm == "solo") & (pt.layer == Lt) & (pt.position == post) & (pt.features == "raw")]
    for s in scopes_of(x):
        r = x[x.scope == s].iloc[0]
        out.append(f"| {scope_name(s, 'B.I pooled')} | {r.cheat} / {r.no_cheat} | {cell(r)} | {f3(r.auroc_turns)} |")
    out += ["", "**Table 8.** Solo arm, per-turn probes against the turn-position baseline (logistic regression on turn index "
            "and turn fraction, same folds, agent-aggregated). The onset labelling drops a cheating agent's turns before its "
            "first cheating tool call.\n",
            "| turn labels | layer | position | B.I pooled agent AUROC | turn-position baseline |", "|---|---|---|---|---|",
            f"| agent | {Lt} | {post} | {cell(ptx)} | {f3(ptx.baseline_turn_position)} |",
            f"| onset | {pox.layer} | {pox.position} | {cell(pox)} | {f3(pox.baseline_turn_position)} |"]
    if fa is not None:
        Lf, posf = best_config(fa, "solo")
        out += ["", "## Does the result depend on the PCA step?\n",
                "A probe on every dimension, z-scored with an L2 penalty and no PCA, gives the same picture in the solo arm "
                "(Table 9). Table 9 covers the solo arm.\n",
                f"**Table 9.** Solo arm, full-dimension probe, layer {Lf}, position `{posf}`.\n"] + scope_table(fa, "solo", Lf, posf, ERASED)
    out += ["", "## Limits\n",
            "The subsample holds a fraction of the runs per author and leaves out the longest transcripts. The cheat label "
            "counts opening the planted material, so a probe may read that the solution text entered the context rather than "
            "an intent. A text classifier on the transcript is the baseline that settles this, and it is not in these tables. "
            "Haiku's think blocks are empty and Luna's hold summaries at best, so for those authors the positions read a turn "
            "boundary with no reasoning inside it. LEACE removes linear traces only. All probes are linear.\n",
            "## Files\n",
            "`pca_agent/`, `full_agent/`, `pca_turn/`, `pca_turn_onset/`: `auroc.csv` (every arm, layer, position, feature "
            "set and scope), `author_probe.csv`, `condition_probe.csv`, `agents.csv`, one `results.json` per configuration, "
            "and a per-run `RESULTS.md`."]
    open(P("RESULTS.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
