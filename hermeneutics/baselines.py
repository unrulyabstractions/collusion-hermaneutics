"""Text baselines for the cheating probe, on the same agents, folds, scopes and statistics as probes.py.

Three views of each agent's text, all taken from the rendered sequence the reader saw (render.py):
  transcript  the whole rendered sequence: system prompt, task, tool outputs and the agent's turns
  agent       the agent's own words: reasoning, message text and tool-call arguments of its assistant turns
  think       the think-window text only: the reasoning inside <think>...</think> (empty for authors without readable reasoning)
Each view gets a TF-IDF (word 1-2 grams, fitted on the training rows of each fold) and an L2 logistic regression with C
by grouped inner CV. The "transcript" view is the text-classifier baseline of PROJECT_DESCRIPTION.md; "think" is the lexical-only reader null
(a reader with no language model, at the same positions). Scopes: B.I pooled, A within author, B.II leave one author out;
AUROC with a 95% bootstrap over runs; within-author label shuffle null (pooled and within scopes).

Per-agent held-out scores are written beside the tables (scores.csv), so every AUROC is recomputable.

    python -m hermeneutics.baselines --agents hermeneutics/results/pca_agent/agents.csv --out hermeneutics/results/tfidf --jobs 40
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from . import DATA_ROOT, MODEL_DIR, REPO
from .probes import K, MIN_CLASS, auc, boot_ci, log, pmap
from .runs import run_list

VIEWS = ("transcript", "agent", "think")
CS = [0.1, 1.0, 10.0, 100.0]
_TOK = None


def texts_of(run_row):
    """The three text views of one run's agent."""
    global _TOK
    from transformers import AutoTokenizer
    from .render import sequence
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL_DIR)
    out = {}
    for agent, q in sequence(run_row["eval_log"], _TOK, with_units=False).items():
        own, think = [], []
        for m in q["messages"]:
            if m["role"] != "assistant":
                continue
            r = (m.get("reasoning") or "").strip()
            if r:
                think.append(r)
                own.append(r)
            if (m.get("content") or "").strip():
                own.append(m["content"].strip())
            for c in m.get("tool_calls") or []:
                own.append(c["function"]["arguments"])
        out = {"run": run_row["run"], "agent": agent, "transcript": q["text"], "agent_text": "\n\n".join(own),
               "think": "\n\n".join(think), "n_tokens": len(q["ids"])}
    return out


def build_texts(agents, data_root, jobs, cache):
    """texts of every agent in agents.csv order; cached as JSONL."""
    if os.path.exists(cache):
        rows = {json.loads(l)["run"]: json.loads(l) for l in open(cache)}
        if set(rows) >= set(agents.run):
            return [rows[r] for r in agents.run]
    valid = {r["run"]: r for r in run_list(data_root) if r["valid"]}
    rows = pmap(texts_of, [(valid[r],) for r in agents.run], jobs)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def fit_lr(Xtr, ytr, gtr, Xte, C=None):
    """Logistic regression on sparse TF-IDF rows: held-out decision values and the C used (grouped inner CV when C is None)."""
    if len(set(ytr.tolist())) < 2:
        return None, None
    if C is None:
        best, C = -1, 1.0
        try:
            sp = list(StratifiedGroupKFold(3, shuffle=True, random_state=1).split(Xtr, ytr, gtr))
        except Exception:  # noqa: BLE001
            sp = []
        for c in CS:
            vals = []
            for tr, te in sp:
                if len(set(ytr[tr].tolist())) < 2 or len(set(ytr[te].tolist())) < 2:
                    continue
                lr = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(Xtr[tr], ytr[tr])
                vals.append(auc(lr.decision_function(Xtr[te]), ytr[te]))
            if vals and np.nanmean(vals) > best:
                best, C = np.nanmean(vals), c
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    return lr.decision_function(Xte), C


def analyze_text(docs, a, n_shuf=100, jobs=8):
    """docs: list of strings, one per agent row of a (DataFrame with run, author, condition, task, y)."""
    y = a.y.values.astype(int)
    auth = a.author.values
    run = a.run.values
    T = sorted(set(auth.tolist()))
    strata = np.array([f"{t}|{v}" for t, v in zip(auth, y)])
    fold = np.zeros(len(a), int)
    for f, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=0).split(np.zeros(len(a)), strata, run)):
        fold[te] = f
    both = [t for t in T if min(((auth == t) & (y == 1)).sum(), ((auth == t) & (y == 0)).sum()) >= MIN_CLASS]
    scopes = ["pooled"] + both
    docs = np.array(docs, dtype=object)

    def prep(scope, f):
        tr, te = fold != f, fold == f
        if scope != "pooled":
            tr, te = tr & (auth == scope), te & (auth == scope)
        if tr.sum() < 2 * MIN_CLASS or not te.any() or not any(d.strip() for d in docs[tr]):
            return scope, f, None
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True, dtype=np.float32)
        try:
            Xtr = vec.fit_transform(docs[tr])
        except ValueError:  # empty vocabulary (e.g. no readable reasoning in this author)
            return scope, f, None
        return scope, f, (np.flatnonzero(tr), np.flatnonzero(te), Xtr, vec.transform(docs[te]))
    PREP = {(s, f): v for s, f, v in pmap(prep, [(s, f) for s in scopes for f in range(K)], jobs)}

    def job(scope, f):
        if scope.startswith("LOTO "):
            t = scope[5:]
            tr, te = auth != t, auth == t
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True, dtype=np.float32)
            try:
                Xtr = vec.fit_transform(docs[tr])
            except ValueError:
                return scope, f, None, None, None
            s, C = fit_lr(Xtr, y[tr], run[tr], vec.transform(docs[te]))
            return scope, f, np.flatnonzero(te), s, C
        v = PREP[(scope, f)]
        if v is None:
            return scope, f, None, None, None
        tr, te, Xtr, Xte = v
        s, C = fit_lr(Xtr, y[tr], run[tr], Xte)
        return scope, f, te, s, C
    jobs_ = [(s, f) for f in range(K) for s in scopes] + [("LOTO " + t, None) for t in both]
    SC, CC = {}, {}
    for scope, f, idx, s, C in pmap(job, jobs_, jobs):
        SC.setdefault(scope, np.full(len(a), np.nan))
        if s is not None:
            SC[scope][idx] = s
            CC.setdefault(scope, []).append(C)
    rows = []
    for scope in scopes + ["LOTO " + t for t in both]:
        s = SC[scope]
        ok = np.isfinite(s)
        r = {"scope": scope, "agents": int(ok.sum()), "cheat": int(y[ok].sum()), "no_cheat": int((1 - y[ok]).sum()),
             "auroc": auc(s[ok], y[ok]) if ok.any() else np.nan, "ci": boot_ci(s[ok], y[ok], run[ok]) if ok.any() else [np.nan, np.nan],
             "C": CC.get(scope)}
        if scope == "pooled":
            r["within"] = {t: auc(s[(auth == t) & ok], y[(auth == t) & ok]) for t in both}
            r["within_mean"] = float(np.nanmean(list(r["within"].values()))) if both else np.nan
        rows.append(r)
        log(f"  {scope}: {r['auroc']:.3f} {r['ci']}")

    def shuf(b):
        rng = np.random.default_rng(1000 + b)
        yp = y.copy()
        for t in T:
            ii = np.flatnonzero(auth == t)
            yp[ii] = y[ii][rng.permutation(len(ii))]
        out = {}
        for scope in scopes:
            Cs = CC.get(scope)
            if not Cs:
                out[scope] = np.nan
                continue
            Cp = float(np.median(Cs))
            s = np.full(len(a), np.nan)
            for f in range(K):
                v = PREP[(scope, f)]
                if v is None:
                    continue
                tr, te, Xtr, Xte = v
                r_ = fit_lr(Xtr, yp[tr], None, Xte, C=Cp)[0]
                if r_ is not None:
                    s[te] = r_
            ok = np.isfinite(s)
            out[scope] = auc(s[ok], yp[ok]) if ok.any() else np.nan
        return out
    shuffle = {}
    if n_shuf:
        draws = pmap(shuf, [(b,) for b in range(n_shuf)], jobs)
        for scope in scopes:
            pv = np.array([d[scope] for d in draws], float)
            real = next(r for r in rows if r["scope"] == scope)["auroc"]
            shuffle[scope] = {"real": real, "null_mean": float(np.nanmean(pv)), "null_95": float(np.nanpercentile(pv, 95)),
                              "p": float((1 + (pv >= real).sum()) / (1 + np.isfinite(pv).sum()))}
    scores = pd.DataFrame({"run": run, "author": auth, "condition": a.condition.values, "y": y, "fold": fold,
                           **{f"score|{s}": SC[s] for s in SC}})
    return {"cheat": rows, "shuffle": shuffle, "authors": T, "with_both_classes": both, "n_agents": int(len(a)),
            "counts": {t: {"cheat": int(((auth == t) & (y == 1)).sum()), "no_cheat": int(((auth == t) & (y == 0)).sum())} for t in T}}, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", required=True, help="agents.csv of a probe run (the same agents and order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--texts", default=os.path.join(REPO, "data", "texts_subsample.jsonl"))
    ap.add_argument("--arms", default="solo,pooled,cheat")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--shuf", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    agents = pd.read_csv(args.agents)
    t0 = time.time()
    texts = build_texts(agents, args.data_root, args.jobs, args.texts)
    assert [t["run"] for t in texts] == list(agents.run)
    log(f"{len(texts)} texts ready ({time.time() - t0:.0f}s); empty think: {sum(not t['think'].strip() for t in texts)}")
    table = []
    for arm in args.arms.split(","):
        sel = np.ones(len(agents), bool) if arm == "pooled" else (agents.condition == arm).values
        a = agents[sel].reset_index(drop=True)
        tx = [t for t, s in zip(texts, sel) if s]
        for view in args.views.split(","):
            key = {"transcript": "transcript", "agent": "agent_text", "think": "think"}[view]
            log(f"== arm {arm} | view {view} | {len(a)} agents")
            res, scores = analyze_text([t[key] for t in tx], a, n_shuf=args.shuf, jobs=args.jobs)
            d = os.path.join(args.out, arm, view)
            os.makedirs(d, exist_ok=True)
            json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1, default=float)
            scores.to_csv(os.path.join(d, "scores.csv"), index=False)
            for r in res["cheat"]:
                s = res["shuffle"].get(r["scope"], {})
                table.append({"arm": arm, "probe": f"tfidf", "layer": "text", "position": view, "pooling": "agent", "features": "raw",
                              "scope": r["scope"], "agents": r["agents"], "cheat": r["cheat"], "no_cheat": r["no_cheat"],
                              "auroc": r["auroc"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "within_mean": r.get("within_mean"),
                              "null_mean": s.get("null_mean"), "null_95": s.get("null_95"), "null_p": s.get("p")})
            pd.DataFrame(table).to_csv(os.path.join(args.out, "auroc.csv"), index=False)
    agents.to_csv(os.path.join(args.out, "agents.csv"), index=False)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
