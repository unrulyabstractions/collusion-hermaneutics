"""Linear probes for cheating on the reader's think-position activations, with the author controls.

Adapted from arena-capstone OLD/analysis/teacher_confound/run.py (PREEXISTING.md). Samples are agents (one per run).
Features: the reader's residual stream at a think position of each assistant turn, at one layer (or the three
concatenated), averaged over the agent's turns (pooling "agent"), or one row per turn (pooling "turn", scored per turn
and as the mean turn score per agent). Arms: solo (no cheating instruction in the prompt: the primary test), cheat
(the prompt asks for the shortcut; nearly every author takes it), pooled (both arms, where a probe can read the
condition instead of the cheating).

Per arm and feature configuration:
  A      a probe trained and tested inside one author (5 folds grouped by run, stratified by label)
  B.I    one pooled probe over the authors, scored on held-out runs; its scores also ranked inside each author
  B.II   leave one author out: trained on four authors, scored on the fifth
  author identity probe (multinomial) with its confusion matrix, before and after erasure
  LEACE erasure (Belrose et al. 2023) of the author, and of author + task + condition, fitted on training rows only
  label-permutation null: labels shuffled inside each author, N_SHUF times, pooled and within-author
  nuisance baseline: logistic regression on one-hot author + task + condition
Probe: training-fold PCA to 256 components, z-scoring, L2 logistic regression with C by grouped inner CV. AUROC with a
95% bootstrap over runs.

    python -m hermeneutics.probes --cache /root/work_hermeneutics/acts --labels data/labels.csv --out hermeneutics/results
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from safetensors import safe_open  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from .runs import AUTHOR_NAME  # noqa: E402

K = 5
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_BOOT = 1000
PCA_DIM = 256
MIN_CLASS = 3  # agents per class for a within-author or leave-one-out probe
T0 = time.time()
REG = {}


def log(*a):
    print(f"[{time.time() - T0:.0f}s]", *a, flush=True)


def _call(*a):
    return REG["f"](*a)


def pmap(fn, arglist, jobs):
    """Forked workers inherit the data; only the small arguments and results are pickled."""
    REG["f"] = fn
    if jobs <= 1:
        return [fn(*a) for a in arglist]
    with mp.get_context("fork").Pool(jobs) as p:
        return p.starmap(_call, arglist)


def auc(s, y, w=None):
    ok = np.isfinite(s)
    if w is not None:
        ok &= w > 0
    if len(set(y[ok].tolist())) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok], sample_weight=None if w is None else w[ok]))


def onehot(vals):
    cats = sorted(set(vals))
    M = np.zeros((len(vals), len(cats)))
    M[np.arange(len(vals)), [cats.index(v) for v in vals]] = 1
    return M


def varying(Z):
    """The one-hot columns that are not constant (a constant concept has nothing to erase)."""
    return Z[:, (Z.std(0) > 0)]


# ---------------------------------------------------------------- LEACE (OLD/analysis/teacher_confound/run.py)
class Leace:
    """x' = x - W+ P W (x - mu): W the (pseudo) whitening of the training covariance, P the projection on the span of
    W Cov(x, z), z the one-hot concept. Removes every linear trace of z from the training rows."""

    def __init__(self, X, Z):
        n = len(X)
        self.mu = X.mean(0)
        Xc = X - self.mu
        Zc = Z - Z.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        keep = S > S[0] * 1e-6
        S, V = S[keep], Vt[keep].T
        Sxz = Xc.T @ Zc / n
        A = V @ ((np.sqrt(n) / S)[:, None] * (V.T @ Sxz))
        Ua, Sa, _ = np.linalg.svd(A, full_matrices=False)
        Q = Ua[:, Sa > Sa[0] * 1e-8] if Sa.size and Sa[0] > 0 else Ua[:, :0]
        self.B1 = V @ ((np.sqrt(n) / S)[:, None] * (V.T @ Q))
        self.B2 = V @ ((S / np.sqrt(n))[:, None] * (V.T @ Q))
        self.rank = Q.shape[1]

    def __call__(self, X):
        return X - ((X - self.mu) @ self.B1) @ self.B2.T


# ---------------------------------------------------------------- probe pieces
PROBE = os.environ.get("HERM_PROBE", "pca")  # pca: PCA-256 then z-score; full: z-score every dimension, no PCA


def zs(Xtr, Xte):
    """Probe features from raw rows with training statistics: PCA to min(PCA_DIM, n_train / 2) components then z-scoring
    (PROBE "pca"), or every dimension z-scored (PROBE "full"; the L2 penalty alone regularises)."""
    c = Xtr.mean(0)
    if PROBE == "full":
        s = Xtr.std(0) + 1e-6
        return (Xtr - c) / s, (Xte - c) / s
    k = min(PCA_DIM, max(10, len(Xtr) // 2))
    Xc = Xtr - c
    if len(Xtr) > Xtr.shape[1]:
        w, V = np.linalg.eigh(Xc.T @ Xc)
        V = V[:, ::-1][:, :k]
    else:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:k].T
    Ptr = Xc @ V
    s = Ptr.std(0) + 1e-9
    return Ptr / s, (Xte - c) @ V / s


def choose_C(Z, y, groups, multi):
    try:
        sp = list(StratifiedGroupKFold(3, shuffle=True, random_state=1).split(Z, y, groups))
    except Exception:  # noqa: BLE001
        return 1e-2
    best, bc = -1, 1e-2
    for C in CS:
        vals = []
        for tr, te in sp:
            if len(set(y[tr].tolist())) < len(set(y.tolist())) or (not multi and len(set(y[te].tolist())) < 2):
                continue
            lr = LogisticRegression(C=C, max_iter=3000, class_weight=None if multi else "balanced").fit(Z[tr], y[tr])
            vals.append((lr.predict(Z[te]) == y[te]).mean() if multi else auc(lr.decision_function(Z[te]), y[te]))
        if vals and np.nanmean(vals) > best:
            best, bc = np.nanmean(vals), C
    return bc


def fit_z(Ztr, ytr, gtr, Zte, multi=False, C=None):
    """Logistic regression on z-scored rows: scores (or predicted classes) for Zte and the C used."""
    if len(set(ytr.tolist())) < 2:
        return None, None
    if C is None:
        C = choose_C(Ztr, ytr, gtr, multi)
    lr = LogisticRegression(C=C, max_iter=3000, class_weight=None if multi else "balanced").fit(Ztr, ytr)
    return (lr.predict(Zte) if multi else lr.decision_function(Zte)), C


def boot_ci(score, y, runs, B=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    ur, inv = np.unique(runs, return_inverse=True)
    vals = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ur), len(ur)), minlength=len(ur))[inv].astype(float)
        a = auc(score, y, w)
        if np.isfinite(a):
            vals.append(a)
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [np.nan, np.nan]


# ---------------------------------------------------------------- data
def load(cache, labels_csv):
    """agents (DataFrame: run, author, condition, task, y, n_turns, n_readable) and acts {layer: [per agent [n_turns, 2, d]]}."""
    lab = pd.read_csv(labels_csv).set_index("run")
    rows, acts = [], {}
    for p in sorted(glob.glob(os.path.join(cache, "*", "*", "meta.json"))):
        m = json.load(open(p))
        if m["n_turns"] == 0 or m["run"] not in lab.index or pd.isna(lab.loc[m["run"], "cheat"]):
            continue
        with safe_open(os.path.join(os.path.dirname(p), "acts.safetensors"), framework="np") as f:
            for L in m["layers"]:
                acts.setdefault(L, []).append(f.get_tensor(f"think_L{L}").astype(np.float32))
            readable = f.get_tensor("think_readable")
        onset = lab.loc[m["run"], "onset_turn"] if "onset_turn" in lab.columns else np.nan
        rows.append({"run": m["run"], "author": m["author"], "condition": m["condition"], "task": m["task"],
                     "y": int(lab.loc[m["run"], "cheat"]), "rung": lab.loc[m["run"], "rung"], "n_turns": m["n_turns"],
                     "n_readable": int(readable.sum()), "n_tokens": m["n_tokens"],
                     "onset_turn": None if pd.isna(onset) else int(onset)})
    return pd.DataFrame(rows), acts


POS = {"open": [0], "close": [1], "both": [0, 1]}


def features(agents, acts, layer, pos, pooling):
    """X and, for pooling "turn", the agent index and the within-agent turn index of every row (None, None for
    pooling "agent"). layer: 26 | 38 | 51 | "all"."""
    layers = sorted(acts) if layer == "all" else [int(layer)]
    per_agent = [np.concatenate([acts[L][i][:, POS[pos], :].reshape(acts[L][i].shape[0], -1) for L in layers], 1)
                 for i in range(len(agents))]
    if pooling == "agent":
        return np.stack([a.mean(0) for a in per_agent]).astype(np.float64), None, None
    idx = np.concatenate([np.full(len(a), i) for i, a in enumerate(per_agent)])
    tpos = np.concatenate([np.arange(len(a)) for a in per_agent])
    return np.concatenate(per_agent).astype(np.float64), idx, tpos


def onset_rows(agents, turn_of, tpos):
    """Row mask for the "onset" turn labels: every turn of a no-cheat agent; of a cheating agent, the turns from its
    first cheating tool call on (earlier turns are ambiguous and dropped; agents without a known onset are dropped)."""
    y = agents.y.values[turn_of]
    onset = np.array([np.nan if o is None else o for o in agents.onset_turn.values], float)[turn_of]
    return (y == 0) | ((y == 1) & np.isfinite(onset) & (tpos >= onset))


# ---------------------------------------------------------------- analysis of one (arm, config)
def analyze(X, a, turn_of=None, n_shuf=100, jobs=8, full=True, turn_pos=None):
    """X rows are agents (turn_of None) or turns (turn_of = agent index per row); a: the agents' DataFrame.
    turn_pos: per-row [turn index, turn fraction] for the turn-position baseline (turn pooling)."""
    y_a = a.y.values.astype(int)
    auth_a = a.author.values
    run_a = a.run.values
    rows_of = np.arange(len(a)) if turn_of is None else turn_of
    y, auth, run = y_a[rows_of], auth_a[rows_of], run_a[rows_of]
    task, cond = a.task.values[rows_of], a.condition.values[rows_of]
    T = sorted(set(auth_a.tolist()))
    tix = np.array([T.index(t) for t in auth])
    Zt = onehot(auth.tolist())
    Zall = varying(np.concatenate([Zt, onehot(task.tolist()), onehot(cond.tolist())], 1))
    strata = np.array([f"{t}|{v}" for t, v in zip(auth_a, y_a)])
    fold_a = np.zeros(len(a), int)
    for f, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=0).split(np.zeros(len(a)), strata, run_a)):
        fold_a[te] = f
    fold = fold_a[rows_of]
    both = [t for t in T if min(((auth_a == t) & (y_a == 1)).sum(), ((auth_a == t) & (y_a == 0)).sum()) >= MIN_CLASS]
    counts = {t: {"agents": int((auth_a == t).sum()), "cheat": int(((auth_a == t) & (y_a == 1)).sum()),
                  "no_cheat": int(((auth_a == t) & (y_a == 0)).sum())} for t in T}
    res = {"counts": counts, "authors": T, "with_both_classes": both, "n_rows": int(len(y)), "n_agents": int(len(a)),
           "pooling": "agent" if turn_of is None else "turn"}

    # ---- feature sets per fold: raw, author erased, author + task + condition erased (fitted on training rows)
    FEAT = {"raw": [X] * K}
    if full:
        FEAT["author erased"] = [Leace(X[fold != f], Zt[fold != f])(X) for f in range(K)]
        FEAT["author, task, condition erased"] = [Leace(X[fold != f], Zall[fold != f])(X) for f in range(K)]
        tr = fold != 0
        res["erasure_check"] = {}
        for name, Z in (("author erased", Zt), ("author, task, condition erased", Zall)):
            Xe = FEAT[name][0][tr]
            cov = (Xe - Xe.mean(0)).T @ (Z[tr] - Z[tr].mean(0)) / tr.sum()
            cov0 = (X[tr] - X[tr].mean(0)).T @ (Z[tr] - Z[tr].mean(0)) / tr.sum()
            res["erasure_check"][name] = {"max_abs_cov_after": float(np.abs(cov).max()), "max_abs_cov_before": float(np.abs(cov0).max())}

    def agent_scores(s):
        """Mean turn score per agent (turn pooling) or the scores themselves."""
        if turn_of is None:
            return s
        out = np.full(len(a), np.nan)
        for i in np.unique(turn_of[np.isfinite(s)]):
            out[i] = np.nanmean(s[turn_of == i])
        return out

    # ---- z-scored train/test rows per (feature set, scope, fold), computed once and shared by the real and shuffled fits
    scopes = ["pooled"] + both
    def prep(name, scope, f):
        F = FEAT[name][f]
        tr, te = fold != f, fold == f
        if scope != "pooled":
            tr, te = tr & (auth == scope), te & (auth == scope)
        if tr.sum() < 2 * MIN_CLASS or not te.any():
            return name, scope, f, None
        Ztr, Zte = zs(F[tr], F[te])
        return name, scope, f, (np.flatnonzero(tr), np.flatnonzero(te), Ztr, Zte)
    PREP = {(n, s, f): v for n, s, f, v in pmap(prep, [(n, s, f) for n in FEAT for s in scopes for f in range(K)], jobs)}
    log("  z-scored folds ready")

    # ---- 1. author identity probe (multinomial), before and after erasure
    def author_probe(name, f):
        v = PREP[(name, "pooled", f)]
        if v is None:
            return name, f, None, None, None
        tr, te, Ztr, Zte = v
        p, C = fit_z(Ztr, tix[tr], run[tr], Zte, multi=True)
        return name, f, te, p, C
    res["author_probe"] = {}
    for name in FEAT:
        pred = np.full(len(y), -1)
        Cs = []
        for n, f, te, p, C in pmap(author_probe, [(name, f) for f in range(K)], jobs):
            if n == name and p is not None:
                pred[te] = p
                Cs.append(C)
        ok = pred >= 0
        cmx = confusion_matrix(tix[ok], pred[ok], labels=range(len(T)))
        res["author_probe"][name] = {"accuracy": float((pred[ok] == tix[ok]).mean()), "C": Cs, "confusion": cmx.tolist(),
                                     "balanced_accuracy": float(np.mean([(pred[ok & (tix == k)] == k).mean() for k in range(len(T)) if (ok & (tix == k)).any()])),
                                     "majority_rate": float(np.bincount(tix).max() / len(tix))}
    log("  author probe", {n: round(v["accuracy"], 3) for n, v in res["author_probe"].items()})

    # ---- 1b. condition readout (pooled arm): can the features tell the cheat-arm prompt from the solo one?
    cix = (cond == "cheat").astype(int)
    if len(set(cix.tolist())) > 1:
        def cond_probe(name, f):
            v = PREP[(name, "pooled", f)]
            if v is None:
                return name, f, None, None
            tr, te, Ztr, Zte = v
            p, _ = fit_z(Ztr, cix[tr], run[tr], Zte, multi=True)
            return name, f, te, p
        res["condition_probe"] = {}
        for name in FEAT:
            pred = np.full(len(y), -1)
            for n, f, te, p in pmap(cond_probe, [(name, f) for f in range(K)], jobs):
                if n == name and p is not None:
                    pred[te] = p
            ok = pred >= 0
            res["condition_probe"][name] = {"accuracy": float((pred[ok] == cix[ok]).mean()),
                                            "majority_rate": float(max(cix.mean(), 1 - cix.mean()))}
        log("  condition probe", {n: round(v["accuracy"], 3) for n, v in res["condition_probe"].items()})

    # ---- 2. cheat probes: pooled (B.I) and within each author (A); 3. leave one author out (B.II)
    def cheat_job(name, scope, f):
        if scope.startswith("LOTO "):
            t = scope[5:]
            tr, te = auth != t, auth == t
            if name == "raw":
                Ftr, Fte = X[tr], X[te]
            else:
                Z = Zt if name == "author erased" else Zall
                e = Leace(X[tr], varying(Z[tr]))
                Ftr, Fte = e(X[tr]), e(X[te])
            Ztr, Zte = zs(Ftr, Fte)
            s, C = fit_z(Ztr, y[tr], run[tr], Zte)
            return name, scope, f, np.flatnonzero(te), s, C
        v = PREP[(name, scope, f)]
        if v is None:
            return name, scope, f, None, None, None
        tr, te, Ztr, Zte = v
        s, C = fit_z(Ztr, y[tr], run[tr], Zte)
        return name, scope, f, te, s, C
    jobs_ = [(n, s, f) for n in FEAT for f in range(K) for s in scopes] + [(n, "LOTO " + t, None) for n in FEAT for t in both]
    SC, CC = {}, {}
    for name, scope, f, idx, s, C in pmap(cheat_job, jobs_, jobs):
        SC.setdefault((name, scope), np.full(len(y), np.nan))
        if s is not None:
            SC[(name, scope)][idx] = s
            CC.setdefault((name, scope), []).append(C)
    rows = []
    for name in FEAT:
        for scope in scopes + ["LOTO " + t for t in both]:
            s = SC[(name, scope)]
            sa = agent_scores(s)
            ok = np.isfinite(sa)
            r = {"features": name, "scope": scope, "agents": int(ok.sum()), "cheat": int(y_a[ok].sum()),
                 "no_cheat": int((1 - y_a[ok]).sum()), "auroc": auc(sa[ok], y_a[ok]), "ci": None,
                 "C": CC.get((name, scope))}
            if turn_of is not None:
                okt = np.isfinite(s)
                r["auroc_turns"] = auc(s[okt], y[okt])
            if scope == "pooled":
                r["within"] = {t: auc(sa[(auth_a == t) & ok], y_a[(auth_a == t) & ok]) for t in both}
                r["within_mean"] = float(np.nanmean(list(r["within"].values()))) if both else np.nan
            rows.append(r)

    def ci_job(i):
        r = rows[i]
        sa = agent_scores(SC[(r["features"], r["scope"])])
        ok = np.isfinite(sa)
        return i, boot_ci(sa[ok], y_a[ok], run_a[ok])
    for i, ci in pmap(ci_job, [(i,) for i in range(len(rows))], jobs):
        rows[i]["ci"] = ci
    for r in rows:
        log(f"  {r['features']} | {r['scope']}: {r['auroc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
    res["cheat"] = rows
    # held-out scores per agent (and per row for turn pooling), so every AUROC above is recomputable from files
    res["_scores"] = pd.DataFrame({"run": run_a, "author": auth_a, "condition": a.condition.values, "y": y_a, "fold": fold_a,
                                   **{f"score|{n}|{s}": agent_scores(SC[(n, s)]) for (n, s) in SC}})
    if turn_of is not None:
        res["_turn_scores"] = pd.DataFrame({"run": run, "turn": turn_pos[:, 0].astype(int) if turn_pos is not None else -1,
                                            "y": y, "fold": fold, **{f"score|{n}|{s}": SC[(n, s)] for (n, s) in SC}})

    # ---- nuisance baselines (metadata only, same folds)
    def nuis(Fn, idx):
        p = np.full(len(y), np.nan)
        for f in range(K):
            tr, te = idx & (fold != f), idx & (fold == f)
            if len(set(y[tr].tolist())) < 2 or not te.any():
                continue
            p[te] = LogisticRegression(C=1.0, max_iter=5000).fit(Fn[tr], y[tr]).decision_function(Fn[te])
        pa = agent_scores(p)
        ok = np.isfinite(pa)
        return auc(pa[ok], y_a[ok])
    allidx = np.ones(len(y), bool)
    res["nuisance"] = {}
    if turn_pos is not None:
        res["nuisance"]["turn position (index, fraction)"] = nuis(turn_pos, allidx)
        log("  turn-position baseline", round(res["nuisance"]["turn position (index, fraction)"], 3))
    if not full:
        return res
    res["nuisance"].update({"pooled (author + task + condition)": nuis(Zall, allidx), "pooled (author only)": nuis(Zt, allidx)})
    Fe = varying(np.concatenate([onehot(task.tolist()), onehot(cond.tolist())], 1))
    for t in both:
        res["nuisance"][f"{t} (task + condition)"] = nuis(Fe, auth == t)

    # ---- label-permutation null: labels shuffled inside each author (agent level), fixed C (the median chosen C)
    def shuf(name, b):
        rng = np.random.default_rng(1000 + b)
        yp_a = y_a.copy()
        for t in T:
            ii = np.flatnonzero(auth_a == t)
            yp_a[ii] = y_a[ii][rng.permutation(len(ii))]
        yp = yp_a[rows_of]
        out = {}
        for scope in scopes:
            Cs = CC.get((name, scope))
            if not Cs:
                out[scope] = np.nan
                continue
            Cp = float(np.median(Cs))
            s = np.full(len(y), np.nan)
            for f in range(K):
                v = PREP[(name, scope, f)]
                if v is None:
                    continue
                tr, te, Ztr, Zte = v
                r_ = fit_z(Ztr, yp[tr], None, Zte, C=Cp)[0]
                if r_ is not None:
                    s[te] = r_
            sa = agent_scores(s)
            ok = np.isfinite(sa)
            out[scope] = auc(sa[ok], yp_a[ok])
        return name, out
    shuf_names = [n for n in ("raw", "author erased") if n in FEAT]
    out = pmap(shuf, [(n, b) for n in shuf_names for b in range(n_shuf)], jobs)
    res["shuffle"] = {}
    for name in shuf_names:
        d = {}
        for scope in scopes:
            pv = np.array([o[1][scope] for o in out if o[0] == name], float)
            real = next(r for r in rows if r["features"] == name and r["scope"] == scope)["auroc"]
            d[scope] = {"real": real, "null_mean": float(np.nanmean(pv)), "null_95": float(np.nanpercentile(pv, 95)),
                        "p": float((1 + (pv >= real).sum()) / (1 + np.isfinite(pv).sum()))}
        res["shuffle"][name] = d
    log("  shuffle done")
    return res


# ---------------------------------------------------------------- driver and tables
def flat_rows(res, arm, layer, pos, pooling):
    out = []
    sh = res.get("shuffle", {})
    for r in res["cheat"]:
        s = sh.get(r["features"], {}).get(r["scope"], {})
        out.append({"arm": arm, "probe": res.get("probe", PROBE), "turn_labels": res.get("turn_labels", "agent"),
                    "layer": layer, "position": pos, "pooling": pooling, "features": r["features"],
                    "baseline_turn_position": res.get("nuisance", {}).get("turn position (index, fraction)"),
                    "scope": r["scope"], "agents": r["agents"], "cheat": r["cheat"], "no_cheat": r["no_cheat"],
                    "auroc": r["auroc"], "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "auroc_turns": r.get("auroc_turns"),
                    "within_mean": r.get("within_mean"), "null_mean": s.get("null_mean"), "null_95": s.get("null_95"),
                    "null_p": s.get("p")})
    return out


def main():
    global PROBE
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="solo,cheat,pooled")
    ap.add_argument("--layers", default="26,38,51")
    ap.add_argument("--positions", default="open,close")
    ap.add_argument("--poolings", default="agent,turn")
    ap.add_argument("--shuf", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="raw features only, no erasure or null")
    ap.add_argument("--probe", default=PROBE, choices=["pca", "full"])
    ap.add_argument("--turn-labels", default="agent", choices=["agent", "onset"],
                    help="turn pooling: every turn carries its agent's label (agent), or a cheating agent's turns count "
                         "from its first cheating tool call on and its earlier turns are dropped (onset)")
    args = ap.parse_args()
    PROBE = args.probe
    os.makedirs(args.out, exist_ok=True)
    agents, acts = load(args.cache, args.labels)
    log(f"{len(agents)} agents; layers {sorted(acts)}")
    print(agents.pivot_table(index=["author", "condition"], columns="y", values="run", aggfunc="count", fill_value=0))
    agents.to_csv(os.path.join(args.out, "agents.csv"), index=False)
    layers = [x if x == "all" else int(x) for x in args.layers.split(",")]
    table, authors, conds = [], [], []
    for arm in args.arms.split(","):
        sel = np.ones(len(agents), bool) if arm == "pooled" else (agents.condition == arm).values
        a = agents[sel].reset_index(drop=True)
        sub = {L: [acts[L][i] for i in np.flatnonzero(sel)] for L in acts}
        if min(a.y.sum(), (1 - a.y).sum()) < MIN_CLASS:
            log(f"{arm}: too few agents of one class, skipped")
            continue
        for L, pos, pooling in itertools.product(layers, args.positions.split(","), args.poolings.split(",")):
            full = not args.quick and pooling == "agent"
            log(f"== arm {arm} | layer {L} | position {pos} | pooling {pooling} | {len(a)} agents")
            X, turn_of, tpos = features(a, sub, L, pos, pooling)
            turn_pos = None
            if pooling == "turn":
                if args.turn_labels == "onset":
                    keep = onset_rows(a, turn_of, tpos)
                    X, turn_of, tpos = X[keep], turn_of[keep], tpos[keep]
                    log(f"  onset labels: {int(keep.sum())} of {len(keep)} turns kept, "
                        f"{len(np.unique(turn_of))} agents")
                n_of = a.n_turns.values[turn_of]
                turn_pos = np.column_stack([tpos, tpos / np.maximum(n_of - 1, 1)]).astype(float)
            res = analyze(X, a, turn_of, n_shuf=args.shuf, jobs=args.jobs, full=full, turn_pos=turn_pos)
            res.update(arm=arm, layer=L, position=pos, probe=PROBE, turn_labels=args.turn_labels if pooling == "turn" else "agent")
            d = os.path.join(args.out, arm, f"L{L}_{pos}_{pooling}")
            os.makedirs(d, exist_ok=True)
            res.pop("_scores").to_csv(os.path.join(d, "scores.csv"), index=False)
            if "_turn_scores" in res:
                res.pop("_turn_scores").to_csv(os.path.join(d, "turn_scores.csv"), index=False)
            json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=1, default=float)
            table += flat_rows(res, arm, L, pos, pooling)
            for name, v in res["author_probe"].items():
                authors.append({"arm": arm, "layer": L, "position": pos, "pooling": pooling, "features": name,
                                "accuracy": v["accuracy"], "balanced_accuracy": v["balanced_accuracy"],
                                "majority_rate": v["majority_rate"], "confusion": json.dumps(v["confusion"])})
            for name, v in res.get("condition_probe", {}).items():
                conds.append({"arm": arm, "layer": L, "position": pos, "pooling": pooling, "features": name,
                              "accuracy": v["accuracy"], "majority_rate": v["majority_rate"]})
            pd.DataFrame(table).to_csv(os.path.join(args.out, "auroc.csv"), index=False)
            pd.DataFrame(authors).to_csv(os.path.join(args.out, "author_probe.csv"), index=False)
            if conds:
                pd.DataFrame(conds).to_csv(os.path.join(args.out, "condition_probe.csv"), index=False)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
