"""Controls and nulls for the hermeneutic probes, on every labelled set, on this machine.

No model calls and no spend. Activations come from the local copy of the public Hub dataset
unrulyabstractions/apart-swarm-align (data/acts_hub, fetched by team_probes) and the pooled caches of matrix_probes
(data/matrix_probes_cache, read only); the prompt sentence means are read from the Hub's acts.safetensors by HTTP range
requests over the prompt rows only (nothing is written to disk but the reduced vectors). Folds, labels, schemes and the
probe recipe (training-fold PCA to 128 components, z-scoring, balanced L2 logistic regression with C by grouped inner
CV) are those of matrix_probes and team_probes, imported.

Stages:
    python -m hermeneutics.controls_nulls feats      turn-0 boundary tokens and think-open state of every agent
    python -m hermeneutics.controls_nulls texts      think-window text of every agent, hashed counts
    python -m hermeneutics.controls_nulls means      prompt sentence means (range reads), agents of our sets
    python -m hermeneutics.controls_nulls prompt     1. prompt-only probe, pooled and within each prompt condition
    python -m hermeneutics.controls_nulls author     2. author-identity control with LEACE
    python -m hermeneutics.controls_nulls lexical    3. lexical-only reader null (think-window TF-IDF), every scheme
    python -m hermeneutics.controls_nulls nulls      4. label-permutation nulls for rows that had none
    python -m hermeneutics.controls_nulls plan       5. RANDOM_READER_PLAN.md
    python -m hermeneutics.controls_nulls report     RESULTS.md
    python -m hermeneutics.controls_nulls verify     VERIFICATION.md
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import struct
import sys
import zlib
import time
from collections import Counter, defaultdict
from pathlib import Path

from hermeneutics import team_probes as tp  # sets the BLAS thread counts before numpy loads
from hermeneutics import matrix_probes as mp

import numpy as np  # noqa: E402

REPO = tp.REPO
HUB = tp.HUB
OUT = REPO / "hermeneutics" / "results" / "controls_nulls"
CACHE = REPO / "data" / "controls_nulls_cache"
MPOUT = mp.OUT
STATUS = Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/"
              "62859942-35bf-474f-b418-4803b6079982/scratchpad/controls_nulls/status.md")
LAYERS = tp.LAYERS
D = 5120
N_NULL = 50
DESIGN_RUNS = REPO / "records" / "2026-11-09" / "design_runs" / "runs.csv"
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:.0f}s]", *a, flush=True)


def status(stage, msg):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} [{stage}] {msg}\n"
    head = "# controls_nulls status\n\n"
    old = STATUS.read_text().splitlines(keepends=True)[2:] if STATUS.is_file() else []
    STATUS.write_text(head + "".join(([line] + old)[:80]))


def agents():
    """The matrix_probes agents table joined to its feature index (fi, bi, ok), plus the design-set prompt fields."""
    import pandas as pd
    A = mp.load_agents()
    I = pd.read_csv(mp.CACHE / "feature_index.csv")
    assert (I.uid.values == A.uid.values).all()
    A = A.join(I.drop(columns="uid"))
    R = pd.read_csv(DESIGN_RUNS)[["run", "agent", "seed_text", "level"]]
    A = A.merge(R, on=["run", "agent"], how="left")
    assert (A.uid.values == I.uid.values).all()
    assert A[A.source == "solo_design"].seed_text.notna().all()
    return A


# ------------------------------------------------------------------------------------------------ caches
def feats():
    """Per agent (one at a time): the turn-0 think-open state at layers 26, 38, 51 ([N, 15360], every agent with a
    think file) and, for agents with turns.safetensors, the five turn-0 boundary tokens ([Nb, 76800], order layer then
    token). Both sit after the prompt and before any reasoning: the boundary tokens close the last prompt message and
    open the assistant turn, the think-open token follows them. Checks: boundary positions contiguous and ending one
    token before think-open."""
    A = agents()
    N = len(A)
    CACHE.mkdir(parents=True, exist_ok=True)
    lib = np.lib.format
    O = lib.open_memmap(CACHE / "open0.tmp.npy", mode="w+", dtype=np.float16, shape=(N, 3 * D))
    hb = A.bi.values >= 0
    B = lib.open_memmap(CACHE / "bound0.tmp.npy", mode="w+", dtype=np.float16, shape=(int(hb.sum()), 15 * D))
    ok = np.zeros(N, int)
    chk = Counter()
    for i, r in enumerate(A.itertuples()):
        if r.ok != 1:
            continue
        z = np.load(tp.think_path(r.source, r.run, r.agent))
        O[i] = np.concatenate([z[f"think_L{L}"][0, 0, :] for L in LAYERS])
        if r.bi >= 0:
            zb = np.load(tp.boundary_path(r.source, r.run, r.agent))
            B[r.bi] = np.concatenate([zb[f"turn_L{L}"][0].reshape(-1) for L in LAYERS])
            bp, t0 = zb["turn_pos"][0], z["think_pos"][0, 0]
            chk["boundary_contiguous_and_adjacent" if (np.diff(bp) == 1).all() and t0 == bp[-1] + 1 else "boundary_gap"] += 1
        ok[i] = 1
        if i % 1000 == 0:
            log(f"feats {i}/{N}")
            status("feats", f"{i}/{N} agents")
    O.flush()
    B.flush()
    del O, B
    for nm in ("open0", "bound0"):
        os.replace(CACHE / f"{nm}.tmp.npy", CACHE / f"{nm}.npy")
    np.save(CACHE / "feats_ok.npy", ok)
    json.dump(dict(chk), open(CACHE / "feats_checks.json", "w"), indent=1)
    log(f"feats done: {ok.sum()} agents; {dict(chk)}")


THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
MSG_RE = re.compile(r"<\|im_start\|>(\w+)")


def think_windows(text):
    return THINK_RE.findall(text)


def texts():
    """Hashed counts of each agent's think-window text (every span between <think> and </think> in rendered.txt, the
    reasoning of every turn, joined), with the same hashing as matrix_probes.tfidf_counts. Checks per agent: windows
    found against meta n_turns, and the roles of the messages before the first <think> (prompt-only position check for
    the sets without spans)."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import HashingVectorizer
    A = agents()
    hv = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 22, alternate_sign=False, norm=None, dtype=np.float32)
    mats, rec = [], []
    for i0 in range(0, len(A), 200):
        docs = []
        for r in A.iloc[i0:i0 + 200].itertuples():
            t = open(HUB / "activations" / r.source / r.run / r.agent / "rendered.txt", encoding="utf-8").read()
            w = think_windows(t)
            first = t.find("<think>")
            roles = MSG_RE.findall(t[:first]) if first >= 0 else []
            rec.append({"uid": r.uid, "windows": len(w), "n_turns": r.n_turns, "chars_think": sum(map(len, w)),
                        "chars_all": len(t), "roles_before_first_think": "/".join(roles)})
            docs.append("\n".join(w))
        mats.append(hv.transform(docs).tocsr())
        if i0 % 1000 == 0:
            log(f"texts {i0}/{len(A)}")
            status("texts", f"{i0}/{len(A)} rendered texts")
    H = sp.vstack(mats).tocsr()
    CACHE.mkdir(parents=True, exist_ok=True)
    sp.save_npz(CACHE / "think_counts.npz", H, compressed=False)
    import pandas as pd
    pd.DataFrame(rec).to_csv(CACHE / "think_text_checks.csv", index=False)
    log(f"texts done: {H.shape}, nnz {H.nnz}")


def _hub_url(s, run, agent):
    from huggingface_hub import hf_hub_url
    return hf_hub_url(tp.REPO_ID, f"activations/{s}/{run}/{agent}/acts.safetensors", repo_type="dataset")


def _get(sess, url, a, b, tries=5):
    for k in range(tries):
        try:
            r = sess.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == b - a + 1:
                return r.content
            err = f"status {r.status_code}, {len(r.content)} bytes"
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"range {a}-{b} of {url}: {err}")


def prompt_units(s, run, agent):
    """The leading units of spans.jsonl whose segment is system or user: the prompt, before the first assistant turn."""
    sp_ = [json.loads(l) for l in open(HUB / "activations" / s / run / agent / "spans.jsonl", encoding="utf-8")]
    p = 0
    while p < len(sp_) and sp_[p]["segment"] in ("system", "user"):
        p += 1
    return sp_, p


def mean_one(sess, r):
    """Token-weighted mean over the prompt units of the sentence means mean_L26/38/51 (float32 [15360]), read by HTTP
    range requests over rows 0..p-1 of each tensor. Checks: header row count equals spans.jsonl, the prompt units end
    before the turn-0 boundary tokens."""
    url = _hub_url(r.source, r.run, r.agent)
    head = _get(sess, url, 0, 65535)
    n = struct.unpack("<Q", head[:8])[0]
    if n + 8 > len(head):
        head = _get(sess, url, 0, n + 7)
    hd = json.loads(head[8:8 + n])
    sp_, p = prompt_units(r.source, r.run, r.agent)
    rec = {"uid": r.uid, "n_units_header": hd["mean_L26"]["shape"][0], "n_units_spans": len(sp_), "prompt_units": p}
    if p == 0 or rec["n_units_header"] != len(sp_):
        rec["problem"] = "no prompt units" if p == 0 else "unit count differs"
        return None, rec
    w = np.array([max(1, x["tokens"][1] - x["tokens"][0]) for x in sp_[:p]], np.float64)
    rec["prompt_tokens_last"] = int(sp_[p - 1]["tokens"][1])
    zb = np.load(tp.boundary_path(r.source, r.run, r.agent))
    rec["turn0_boundary_first"] = int(zb["turn_pos"][0][0])
    rec["first_assistant_segment"] = sp_[p]["segment"] if p < len(sp_) else None
    out = []
    for L in LAYERS:
        t = hd[f"mean_L{L}"]
        assert t["dtype"] == "F16" and t["shape"][1] == D
        a0 = 8 + n + t["data_offsets"][0]
        buf = _get(sess, url, a0, a0 + p * D * 2 - 1)
        M = np.frombuffer(buf, np.float16).reshape(p, D).astype(np.float32)
        out.append((M * w[:, None]).sum(0) / w.sum())
    return np.concatenate(out), rec


def means(jobs=8):
    """Prompt sentence means of every agent of our sets (the agents with turns.safetensors), eight range readers at a
    time; resumable (rows already filled are kept)."""
    import pandas as pd
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import get_token
    A = agents()
    rows = A[(A.bi >= 0) & (A.ok == 1)]
    nb = int((A.bi >= 0).sum())
    p = CACHE / "prompt_means.npy"
    M = np.lib.format.open_memmap(p, mode="r+" if p.is_file() else "w+", dtype=np.float32, shape=(nb, 3 * D))
    done_p = CACHE / "prompt_means_checks.jsonl"
    done = {json.loads(l)["uid"] for l in open(done_p)} if done_p.is_file() else set()
    todo = [r for r in rows.itertuples() if r.uid not in done]
    log(f"means: {len(todo)} of {len(rows)} to read")
    tok = get_token()

    def work(r):
        sess = requests.Session()
        if tok:
            sess.headers["Authorization"] = f"Bearer {tok}"
        return r, *mean_one(sess, r)
    fh = open(done_p, "a")
    with ThreadPoolExecutor(jobs) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for k, f in enumerate(as_completed(futs)):
            try:
                r, v, rec = f.result()
            except Exception as e:  # noqa: BLE001
                log(f"means failed: {e!r}"[:300])
                continue
            if v is not None:
                M[r.bi] = v
                rec["ok"] = 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 100 == 0:
                M.flush()
                log(f"means {k}/{len(todo)}")
                status("means", f"{k + len(done)}/{len(rows)} agents read")
    M.flush()
    fh.close()
    C = pd.read_json(done_p, lines=True)
    log(f"means done: {int(C.get('ok', pd.Series(dtype=float)).fillna(0).sum())} ok of {len(rows)}")


# ------------------------------------------------------------------------------------------------ shared
PFEATS = {"first": "turn-0 boundary tokens (5) + think-open, layers 26+38+51",
          "bound0": "turn-0 boundary tokens (5), layers 26+38+51",
          "open0": "turn-0 think-open, layers 26+38+51",
          "means": "prompt sentence means (token-weighted, system and user units), layers 26+38+51"}
REFS = {"think": "full probe (think pair over all turns, layers 26+38+51)", "seven": "full probe, seven tokens",
        "meta": "metadata baseline", "tfidf": "TF-IDF rendered text"}


class Feats:
    def __init__(self, A):
        self.A = A
        self.O = np.load(CACHE / "open0.npy", mmap_mode="r")
        self.B = np.load(CACHE / "bound0.npy", mmap_mode="r")
        self.M = np.load(CACHE / "prompt_means.npy", mmap_mode="r") if (CACHE / "prompt_means.npy").is_file() else None
        self.mean_ok = set()
        p = CACHE / "prompt_means_checks.jsonl"
        if p.is_file():
            self.mean_ok = {x["uid"] for x in map(json.loads, open(p)) if x.get("ok") == 1}

    def has(self, feat, idx):
        A = self.A
        if feat == "open0":
            return True
        if feat in ("first", "bound0"):
            return bool((A.bi.values[idx] >= 0).all())
        if feat == "means":
            return self.M is not None and set(A.uid.values[idx]) <= self.mean_ok
        raise KeyError(feat)

    def X(self, feat, idx):
        idx = np.asarray(idx)
        bi = self.A.bi.values[idx]
        if feat == "open0":
            return np.asarray(self.O[idx], np.float32)
        if feat == "bound0":
            return np.asarray(self.B[bi], np.float32)
        if feat == "first":
            return np.concatenate([np.asarray(self.B[bi], np.float32), np.asarray(self.O[idx], np.float32)], 1)
        if feat == "means":
            return np.asarray(self.M[bi], np.float32)
        raise KeyError(feat)


def plain_perms(y, seed, B=N_NULL):
    rng = np.random.default_rng(seed)
    return [y[rng.permutation(len(y))] for _ in range(B)]


def mp_scores(label, scheme, ptype="B.I", unit="pooled"):
    """matrix_probes' saved out-of-fold scores of one cell: {feature: DataFrame(uid, y, fold, score)}."""
    import pandas as pd
    p = MPOUT / "scores" / label / f"{mp.slug(scheme)}.csv.gz"
    S = pd.read_csv(p)
    S = S[(S.ptype == ptype) & (S.unit == unit)]
    return {f: d.set_index("uid") for f, d in S.groupby("feature")}


def mp_cell(label, scheme, ptype, unit, feature):
    import pandas as pd
    c = pd.read_csv(MPOUT / "cells.csv")
    r = c[(c.label == label) & (c.scheme == scheme) & (c.ptype == ptype) & (c.unit == unit) & (c.feature == feature)]
    return r.iloc[0].to_dict() if len(r) else {}


def scheme_rows(A, label, srcs):
    """The rows run_cv_scheme uses: the sources, features ok, label present (A order)."""
    return np.flatnonzero((A.source.isin(srcs) & (A.ok == 1) & A[label].notna()).values)


def probe_cv(X, y, g, fold, null_seed=None, B=N_NULL):
    """matrix_probes' recipe on one fold vector; optional plain label-permutation null with C fixed."""
    P = tp.prep_folds(X, fold)
    s, Cs = tp.cv_scores(P, y, g, len(y))
    C = float(np.median(Cs)) if Cs else 1e-2
    nul = tp.null_aucs(P, plain_perms(y, null_seed, B), g, len(y), C) if null_seed is not None else None
    return s, C, nul


def meta_cv(A, idx, y, fold):
    F = mp.meta_frame(A, idx)
    s = np.full(len(idx), np.nan)
    for k in range(tp.K):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if len(te) and len(set(y[tr].tolist())) > 1:
            s[te] = mp.meta_fit(F[tr], y[tr], F[te])
    return s


def within_folds(y, g, au):
    try:
        return tp.folds_for(np.array([f"{a}|{v}" for a, v in zip(au, y)]), g, seed=0)
    except ValueError:
        return tp.folds_for(y, g, seed=0)


def conditions(s):
    """Prompt conditions per set: the arm (controls solo vs cheat; design level|post; board seed arm; SPAR condition;
    external benchmark, harness or condition); for solo_design also level alone and post (team hint vs clean) alone."""
    return ["arm", "level", "seed_text"] if s == "solo_design" else ["arm"]


def save_frame(rows, p):
    import pandas as pd
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False, compression="gzip" if str(p).endswith(".gz") else None, float_format="%.6g")


def prompt_hashes(A):
    """sha1 of each agent's prompt text: rendered.txt up to the first '<|im_start|>assistant' (cached)."""
    import hashlib
    import pandas as pd
    p = CACHE / "prompt_hash.csv"
    if p.is_file():
        H = pd.read_csv(p)
        if len(H) == len(A) and (H.uid.values == A.uid.values).all():
            return H.prompt_sha1.values
    out = []
    for r in A.itertuples():
        t = open(HUB / "activations" / r.source / r.run / r.agent / "rendered.txt", encoding="utf-8").read()
        k = t.find("<|im_start|>assistant")
        out.append(hashlib.sha1(t[:k if k >= 0 else len(t)].encode()).hexdigest())
    pd.DataFrame({"uid": A.uid, "prompt_sha1": out}).to_csv(p, index=False)
    return np.array(out)


def rate_cv(keys_list, y, fold):
    """Out-of-fold score: the training fold's label rate among agents with the same key, trying each key in order
    (the last key falls back to the training fold's overall rate)."""
    s = np.full(len(y), np.nan)
    for k in range(tp.K):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if not len(te):
            continue
        tabs = []
        for keys in keys_list:
            d = defaultdict(list)
            for i in tr:
                d[keys[i]].append(y[i])
            tabs.append({a: float(np.mean(v)) for a, v in d.items()})
        for i in te:
            v = next((t[keys[i]] for keys, t in zip(keys_list, tabs) if keys[i] in t), float(y[tr].mean()))
            s[i] = v
    return s


IDENT = {"arm_rate": "condition identity: training-fold label rate of the agent's arm",
         "task_arm_rate": "task and condition identity: training-fold label rate of the agent's (task, arm), else arm",
         "prompt_rate": "prompt-text identity: training-fold label rate of agents with the identical prompt text"}


# ------------------------------------------------------------------------------------------------ 1. prompt-only probe
def prompt(only=None):
    """Per set and label ('alone' scheme of matrix_probes): the prompt-only probe on matrix_probes' own B.I folds,
    with a 50-draw plain label-permutation null; beside it the full probe, the seven-token probe, TF-IDF and the
    metadata baseline from matrix_probes' saved scores on the same agents. Then, within each prompt condition: the
    AUROC of those pooled out-of-fold scores restricted to the condition, and a probe refitted inside the condition
    (prompt-only, full think probe, metadata) on new grouped folds, with a 50-draw null for the prompt-only refit."""
    import pandas as pd
    A = agents()
    F = Feats(A)
    D_ = mp.Data(need_tfidf=False)
    assert (D_.A.uid.values == A.uid.values).all()
    pooled, within, scores = [], [], []
    PH = prompt_hashes(A)
    for li, label in enumerate(mp.LABELS):
        for name, kind, srcs, _ in mp.schemes(A, label):
            if kind != "alone" or (only and only not in name):
                continue
            s = srcs[0]
            ii = scheme_rows(A, label, srcs)
            y = A[label].values[ii].astype(int)
            g = A.group.values[ii]
            au = A.author.values[ii]
            uids = A.uid.values[ii]
            ref = mp_scores(label, name)
            th = ref["think"]
            assert set(th.index) == set(uids) and len(th) == len(uids), (label, name)
            fold = th.loc[uids, "fold"].values.astype(int)
            assert (th.loc[uids, "y"].values == y).all()
            feats = [f for f in PFEATS if F.has(f, ii)]
            base = {"label": label, "set": s, "scheme": name}
            sc = {}
            for j, f in enumerate(feats):
                X = F.X(f, ii)
                sc[f], C, nul = probe_cv(X, y, g, fold, null_seed=5000 + 100 * li + j)
                del X
                pooled.append(mp.stat_row(dict(base, feature=f, kind="prompt-only"), sc[f], y, g, au, nul, C))
            arm_k = A.arm.astype(str).values[ii]
            ta_k = np.array([f"{a}|{b}" for a, b in zip(A.task.astype(str).values[ii], arm_k)])
            ph = PH[ii]
            for f, keys in (("arm_rate", [arm_k]), ("task_arm_rate", [ta_k, arm_k]), ("prompt_rate", [ph])):
                sc[f] = rate_cv(keys, y, fold)
                pooled.append(mp.stat_row(dict(base, feature=f, kind="reference (prompt identity)",
                                               distinct_keys=len(set(keys[0]))), sc[f], y, g, au))
            for f in REFS:
                if f in ref:
                    sc[f] = ref[f].loc[uids, "score"].values
                    c = mp_cell(label, name, "B.I", "pooled", f)
                    r = mp.stat_row(dict(base, feature=f, kind="reference (matrix_probes)"), sc[f], y, g, au)
                    r.update({k: c.get(k) for k in ("null_mean", "null_95", "null_p")})
                    r["matrix_auroc"] = c.get("auroc")
                    pooled.append(r)
            for f, v in sc.items():
                scores.append(pd.DataFrame({"label": label, "set": s, "feature": f, "uid": uids, "y": y, "fold": fold,
                                            "score": v}))
            log(f"prompt {label} {name}: " + ", ".join(f"{f} {tp.auc(v, y):.3f}" for f, v in sc.items()))
            status("prompt", f"{label} {name} pooled done")
            # within each prompt condition
            main = "first" if "first" in feats else "open0"
            for cc in conditions(s):
                vals = A[cc].values[ii].astype(str)
                for c in sorted(set(vals)):
                    sub = np.flatnonzero(vals == c)
                    ys, gs = y[sub], g[sub]
                    cb = dict(base, condition=cc, value=c, n=len(sub), pos=int(ys.sum()), runs=len(np.unique(gs)))
                    for f, v in sc.items():
                        a = tp.auc(v[sub], ys)
                        lo, hi = tp.boot_ci(v[sub], ys, gs) if np.isfinite(a) else (np.nan, np.nan)
                        within.append(dict(cb, mode="pooled scores restricted", feature=f, auroc=a, ci_lo=lo, ci_hi=hi))
                    if not mp.feasible(ys, gs):
                        within.append(dict(cb, mode="refit within condition", feature=main,
                                           note="too few agents or runs in a class"))
                        continue
                    fw = within_folds(ys, gs, au[sub])
                    for j, f in enumerate([x for x in (main, "means") if x in feats] + ["think", "meta"]):
                        nul, C = None, None
                        if f == "think":
                            v, C, _ = probe_cv(D_.X("think", ii[sub]), ys, gs, fw)
                        elif f == "meta":
                            v = meta_cv(A, ii[sub], ys, fw)
                        else:
                            v, C, nul = probe_cv(F.X(f, ii[sub]), ys, gs, fw, null_seed=7000 + 100 * li + j)
                        r = mp.stat_row(dict(cb, mode="refit within condition", feature=f), v, ys, gs, au[sub], nul, C)
                        if not np.isfinite(r["auroc"]):  # every held-out fold held one class: nothing scored
                            r.update(null_mean=np.nan, null_95=np.nan, null_p=np.nan,
                                     note="not scorable: each test fold held one class")
                        within.append(r)
                        scores.append(pd.DataFrame({"label": label, "set": s, "feature": f"within:{cc}={c}:{f}",
                                                    "uid": uids[sub], "y": ys, "fold": fw, "score": v}))
                    log(f"  within {cc}={c}: " + ", ".join(f"{r['feature']} {r.get('auroc', np.nan):.3f}" for r in within[-4:]
                                                         if r.get("mode") == "refit within condition"))
    pd.DataFrame(pooled).to_csv(OUT / "prompt_pooled.csv", index=False)
    pd.DataFrame(within).to_csv(OUT / "prompt_within.csv", index=False)
    (OUT / "scores").mkdir(exist_ok=True)
    pd.concat(scores, ignore_index=True).to_csv(OUT / "scores" / "prompt_scores.csv.gz", index=False, compression="gzip",
                                                float_format="%.6g")
    log("prompt done")


# ------------------------------------------------------------------------------------------------ 2. author control
AUTHOR_SCHEMES = ["alone: controls", "alone: solo_design", "alone: spar_team", "alone: spar_mixed", "alone: goodfire",
                  "ours", "all"]
ERASE = {"raw": [], "author": ["author"], "author+task+condition": ["author", "task", "arm"]}


def leace_fit(X, Z, tol=1e-8):
    """LEACE (Belrose et al. 2023) on training rows: x' = x - W+ P W (x - mu), W = Sxx^(-1/2), P the orthogonal
    projection onto the column space of W Sxz. Returns (mu, E) with x' = x - (x - mu) @ E.T."""
    from scipy.linalg import eigh
    mu = X.mean(0)
    Xc = X - mu
    Zc = Z - Z.mean(0)
    n = len(X)
    w, V = eigh(Xc.T @ Xc / n)
    keep = w > w.max() * tol
    V, w = V[:, keep], w[keep]
    W = (V / np.sqrt(w)) @ V.T
    Wp = (V * np.sqrt(w)) @ V.T
    U, s_, _ = np.linalg.svd(W @ (Xc.T @ Zc / n), full_matrices=False)
    U = U[:, s_ > s_.max() * tol] if s_.size and s_.max() > 0 else U[:, :0]
    E = Wp @ U @ U.T @ W
    return mu, E


def concept(A, idx, cols):
    return np.column_stack([tp.onehot(A[c].astype(str).values[idx].tolist()) for c in cols])


def erase_pair(A, Ztr, Zte, tr_idx, cols):
    if not cols:
        return Ztr, Zte, 0.0
    mu, E = leace_fit(Ztr, concept(A, tr_idx, cols))
    Etr, Ete = Ztr - (Ztr - mu) @ E.T, Zte - (Zte - mu) @ E.T
    Zc = concept(A, tr_idx, cols)
    resid = float(np.abs((Etr - Etr.mean(0)).T @ (Zc - Zc.mean(0)) / len(Etr)).max())  # cross-covariance after erasure
    return Etr, Ete, resid


def author_acc(Etr, atr, Ete, ate):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(Etr, atr)
    pred = lr.predict(Ete)
    vals, cnt = np.unique(atr, return_counts=True)
    maj = vals[np.argmax(cnt)]
    return int((pred == ate).sum()), int((ate == maj).sum())


def author(only=None):
    """Author-identity control. On every pooled scheme with more than one author: per B.I fold of matrix_probes (the
    think-pair features, the same training-fold PCA-z), LEACE fitted on the training fold erases author (one-hot), or
    author + task + condition (arm), from the 128 components the probe reads; the author probe (multinomial logistic
    regression, C 1) and B.I are refitted on the erased components. B.II (leave one author out) likewise, with the
    eraser fitted on the training authors. Raw rows recompute matrix_probes' B.I and B.II and must equal them."""
    import pandas as pd
    A = agents()
    D_ = mp.Data(need_tfidf=False)
    assert (D_.A.uid.values == A.uid.values).all()
    p_rows = OUT / "author_rows.csv"
    old = pd.read_csv(p_rows) if p_rows.is_file() else pd.DataFrame(columns=["label", "scheme"])
    rows = [old] if len(old) else []
    for label in mp.LABELS:
        seen = {}
        for name, kind, srcs, te_ in mp.schemes(A, label):
            if name not in AUTHOR_SCHEMES or te_ is not None or (only and only not in name):
                continue
            if tuple(sorted(srcs)) in seen:  # matrix_probes skips a scheme whose sources equal an earlier one's
                log(f"author {label} {name}: identical source set to '{seen[tuple(sorted(srcs))]}'")
                continue
            seen[tuple(sorted(srcs))] = name
            if len(old) and ((old.label == label) & (old.scheme == name)).any():
                continue
            ii = scheme_rows(A, label, srcs)
            au = A.author.values[ii]
            if len(np.unique(au)) < 2:
                continue
            t0 = time.time()
            status("author", f"{label} {name}: {len(ii)} agents, {len(np.unique(au))} authors")
            y = A[label].values[ii].astype(int)
            g = A.group.values[ii]
            uids = A.uid.values[ii]
            th = mp_scores(label, name)["think"]
            assert set(th.index) == set(uids)
            fold = th.loc[uids, "fold"].values.astype(int)
            X = D_.X("think", ii)
            out, sc = [], {v: np.full(len(ii), np.nan) for v in ERASE}
            acc = {v: [0, 0, 0] for v in ERASE}
            resid = defaultdict(float)
            for k in range(tp.K):
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                if len(tr) < 6 or not len(te):
                    continue
                Ztr, Zte = tp.pca_z(X[tr], X[te])
                for v, cols in ERASE.items():
                    Etr, Ete, rs = erase_pair(A, Ztr, Zte, ii[tr], cols)
                    resid[v] = max(resid[v], rs)
                    s_, _ = tp.fit_score(Etr, y[tr], g[tr], Ete)
                    if s_ is not None:
                        sc[v][te] = s_
                    h, m = author_acc(Etr, au[tr], Ete, au[te])
                    acc[v][0] += h
                    acc[v][1] += m
                    acc[v][2] += len(te)
            base = {"label": label, "scheme": name, "authors": len(np.unique(au))}
            for v in ERASE:
                r = mp.stat_row(dict(base, ptype="B.I", unit="pooled", erase=v), sc[v], y, g, au)
                r.update(author_acc=acc[v][0] / acc[v][2], author_chance=acc[v][1] / acc[v][2], n_author_test=acc[v][2],
                         max_cross_cov_after=resid[v])
                if v == "raw":
                    r["matrix_auroc"] = mp_cell(label, name, "B.I", "pooled", "think").get("auroc")
                out.append(r)
                frames = pd.DataFrame({"label": label, "scheme": name, "ptype": "B.I", "unit": "pooled", "erase": v,
                                       "uid": uids, "y": y, "fold": fold, "score": sc[v]})
                save_frame(frames.to_dict("list"), OUT / "scores" / "author" / f"{label}_{mp.slug(name)}_BI_{mp.slug(v)}.csv.gz")
            log(f"author {label} {name} B.I: " + ", ".join(f"{r['erase']} {r['auroc']:.3f} (author {r['author_acc']:.2f}/"
                                                         f"{r['author_chance']:.2f})" for r in out))
            # B.II
            per = {v: {} for v in ERASE}
            bii_frames = []
            for a in sorted(np.unique(au)):
                te = np.flatnonzero(au == a)
                tr = np.flatnonzero((au != a) & ~np.isin(g, list(set(g[te]))))
                if min(int(y[te].sum()), int((1 - y[te]).sum())) < mp.MIN_CLASS or \
                        min(int(y[tr].sum()), int((1 - y[tr]).sum())) < mp.MIN_CLASS:
                    continue
                Ztr, Zte = tp.pca_z(X[tr], X[te])
                for v, cols in ERASE.items():
                    Etr, Ete, _ = erase_pair(A, Ztr, Zte, ii[tr], cols)
                    s_, C = tp.fit_score(Etr, y[tr], g[tr], Ete)
                    r = mp.stat_row(dict(base, ptype="B.II", unit=a, erase=v), s_, y[te], g[te], au[te], C=C,
                                    extra={"n_train": len(tr)})
                    if v == "raw":
                        r["matrix_auroc"] = mp_cell(label, name, "B.II", a, "think").get("auroc")
                    out.append(r)
                    per[v][a] = {"think": (s_, y[te], g[te], None)}
                    bii_frames.append(pd.DataFrame({"erase": v, "unit": a, "uid": uids[te], "y": y[te], "score": s_}))
            for v in ERASE:
                for r in mp.mean_rows(dict(base, ptype="B.II", erase=v), per[v], ["think"]):
                    if v == "raw":
                        r["matrix_auroc"] = mp_cell(label, name, "B.II", "mean over authors", "think").get("auroc")
                    out.append(r)
            if bii_frames:
                pd.concat(bii_frames).to_csv(OUT / "scores" / "author" / f"{label}_{mp.slug(name)}_BII.csv.gz", index=False,
                                             compression="gzip", float_format="%.6g")
            del X
            rows.append(pd.DataFrame(out))
            pd.concat(rows, ignore_index=True).to_csv(p_rows, index=False)
            log(f"author {label} {name} done in {time.time() - t0:.0f}s")
    log("author done")


# ------------------------------------------------------------------------------------------------ 3. lexical-only null
def lexical(only=None):
    """matrix_probes' own scheme runner (B.I, A, B.II with the same folds and seeds) with the probe features switched
    off and the TF-IDF baseline reading the think-window text instead of the whole rendered text; the metadata rows it
    recomputes must equal matrix_probes' (a check that the folds are the same)."""
    import pandas as pd
    import scipy.sparse as sp
    D_ = mp.Data(need_tfidf=False)
    D_.H = sp.load_npz(CACHE / "think_counts.npz").tocsr()
    assert D_.H.shape[0] == len(D_.A)
    mp.scheme_feats = lambda A, srcs: []
    fit0 = mp.tfidf_fit

    def tfidf_fit_guarded(Htr, ytr, gtr, Hte):
        # an agent whose think windows hold no word (Claude Haiku 5.5's reasoning is not exposed; several external
        # authors' windows are empty too) has nothing to read: it is left out of training and gets no lexical score,
        # so a held-out author made of such agents gets no AUROC instead of a constant-score 0.500
        out = np.full(Hte.shape[0], np.nan)
        ktr, kte = Htr.getnnz(1) > 0, Hte.getnnz(1) > 0
        if not kte.any() or len(set(ytr[ktr].tolist())) < 2 or \
                not (np.asarray((Htr[ktr] > 0).sum(0)).ravel() >= 2).any():
            return out, None
        s_, C = fit0(Htr[ktr], ytr[ktr], gtr[ktr], Hte[kte])
        if s_ is not None:
            out[kte] = s_
        return out, C
    mp.tfidf_fit = tfidf_fit_guarded
    A = D_.A
    p_rows = OUT / "lexical_rows.csv"
    old = pd.read_csv(p_rows) if p_rows.is_file() else pd.DataFrame(columns=["label", "scheme"])
    allrows = [old] if len(old) else []
    seen = {}
    plan = [(lab, sc) for lab in mp.LABELS for sc in mp.schemes(A, lab)]
    for k, (label, (name, kind, tr, te)) in enumerate(plan):
        key = (label, tuple(sorted(tr)), None if te is None else tuple(sorted(te)))
        if key in seen or (only and only not in name):
            continue
        seen[key] = name
        if len(old) and ((old.label == label) & (old.scheme == name)).any():
            continue
        if (te is None and not tr) or (te is not None and (not tr or not te)):
            continue
        t0 = time.time()
        status("lexical", f"cell {k + 1}/{len(plan)}: {label} {name}")
        if te is None:
            rows, frames, _ = mp.run_cv_scheme(D_, label, name, kind, tr, 1000 + 10 * k)
        else:
            rows, frames, _ = mp.run_split_scheme(D_, label, name, kind, tr, te, 1000 + 10 * k)
        R = pd.DataFrame(rows)
        R["feature"] = R.feature.replace({"tfidf": "tfidf_think"})
        allrows.append(R)
        for f in frames:
            f["feature"] = "tfidf_think" if f["feature"] == "tfidf" else f["feature"]
        mp.save_frames(frames, OUT / "scores" / "lexical" / label / f"{mp.slug(name)}.csv.gz")
        pd.concat(allrows, ignore_index=True).to_csv(p_rows, index=False)
        log(f"lexical {label} {name} done in {time.time() - t0:.0f}s")
    log("lexical done")


# ------------------------------------------------------------------------------------------------ 4. permutation nulls
RES = REPO / "hermeneutics" / "results"
AUDIT = OUT / "null_audit.csv"


def fixed_null(score, y, seed, B=N_NULL):
    """AUROC of fixed held-out scores against B random permutations of the labels (the scores were made without the
    test labels, so this is the permutation test of AUROC = 0.5 for those scores)."""
    ok = np.isfinite(np.asarray(score, float))
    s, yy = np.asarray(score, float)[ok], np.asarray(y).astype(int)[ok]
    rng = np.random.default_rng(seed)
    return np.array([tp.auc(s, yy[rng.permutation(len(yy))]) for _ in range(B)])


def null_stats(a, nl):
    nl = np.asarray(nl, float)
    return {"null_mean": float(np.nanmean(nl)), "null_95": float(np.nanpercentile(nl, 95)),
            "null_p": float((1 + (nl >= a).sum()) / (1 + np.isfinite(nl).sum())), "draws": int(np.isfinite(nl).sum())}


def refit_baselines(D_, label, name, srcs, seed, jobs=4):
    """Refit nulls for matrix_probes' TF-IDF and metadata B.I pooled rows: the label permuted within (source, author, arm)
    strata as matrix_probes does for the probe, 50 draws, the baseline refitted on each draw on the same folds (TF-IDF
    with C fixed at the median of the per-fold C chosen on the real labels). The real-label AUROC is recomputed and
    must equal cells.csv."""
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    A = D_.A
    ii = scheme_rows(A, label, srcs)
    y = A[label].values[ii].astype(int)
    g = A.group.values[ii]
    uids = A.uid.values[ii]
    strata = np.array([f"{a}|{b}|{c}" for a, b, c in zip(A.source.values[ii], A.author.values[ii], A.arm.astype(str).values[ii])])
    th = mp_scores(label, name)["think"]
    fold = th.loc[uids, "fold"].values.astype(int)
    pl = tp.perm_within(strata, y, B=N_NULL, seed=seed)
    out = {}
    # metadata
    F = mp.meta_frame(A, ii)

    def meta_s(yv):
        s = np.full(len(y), np.nan)
        for k in range(tp.K):
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            if len(te) and len(set(yv[tr].tolist())) > 1:
                s[te] = mp.meta_fit(F[tr], yv[tr], F[te])
        return s
    out["meta"] = (tp.auc(meta_s(y), y), np.array([tp.auc(meta_s(yp), yp) for yp in pl]))
    # TF-IDF
    H = D_.H[A.fi.values[ii]]
    P, Cs, s = [], [], np.full(len(y), np.nan)
    for k in range(tp.K):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if not len(te) or len(set(y[tr].tolist())) < 2:
            continue
        Xtr, Xte = mp.tfidf_pair(H[tr], H[te])
        sc, C = tp.fit_score(Xtr, y[tr], g[tr], Xte, Cs=tp.CS_TEXT)
        s[te] = sc
        Cs.append(C)
        P.append((tr, te, Xtr, Xte))
    C = float(np.median(Cs))

    def one(yp):
        v = np.full(len(y), np.nan)
        for tr, te, Xtr, Xte in P:
            if len(set(yp[tr].tolist())) > 1:
                v[te] = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, yp[tr]).decision_function(Xte)
        return tp.auc(v, yp)
    nl = Parallel(n_jobs=jobs, backend="threading")(delayed(one)(yp) for yp in pl)
    out["tfidf"] = (tp.auc(s, y), np.array(nl), C)
    return out, len(ii), int(y.sum())


def _mp_identical(C):
    """scheme -> the scheme whose scores it copies (cells.csv note 'identical source set to ...')."""
    m = {}
    for r in C.dropna(subset=["note"]).itertuples():
        mm = re.match(r"identical source set to '(.+)'", str(r.note))
        if mm:
            m[(r.label, r.scheme)] = mm.group(1)
    return m


def nulls():
    """Item 4. (a) Refit nulls for matrix_probes TF-IDF and metadata B.I pooled rows of every cross-validated scheme.
    (b) Fixed-score nulls (50 draws) for every other row without a null whose held-out scores are saved and reproduce
    the table's AUROC: matrix_probes TF-IDF and metadata rows (A, B.II, transfer and leave-one-out cells; mean-over-
    authors rows average the per-author draws), team_probes agent and team baselines and team aggregates (board_seed and
    the three sets), external_probes trajectory baselines. Every row is matched to its table by recomputing the AUROC."""
    import pandas as pd
    import scipy.sparse as sp
    rows = []
    p_out = OUT / "nulls_rows.csv"
    done = pd.read_csv(p_out) if p_out.is_file() else pd.DataFrame(columns=["part", "key"])
    have = set(zip(done.part, done.key))
    rows = done.to_dict("records")

    def add(r):
        rows.append(r)
        pd.DataFrame(rows).to_csv(p_out, index=False)
    C = pd.read_csv(MPOUT / "cells.csv")
    ident = _mp_identical(C)
    # ---- (a)
    D_ = mp.Data(need_tfidf=False)
    D_.H = sp.load_npz(mp.CACHE / "tfidf_counts.npz").tocsr()
    for li, label in enumerate(mp.LABELS):
        for k, (name, kind, srcs, te_) in enumerate(mp.schemes(D_.A, label)):
            if te_ is not None or not srcs or (label, name) in ident:
                continue
            key = f"{label}|{name}|B.I|pooled"
            if ("refit", key) in have:
                continue
            t0 = time.time()
            status("nulls", f"refit {key}")
            o, n, pos = refit_baselines(D_, label, name, srcs, seed=9000 + 100 * li + k)
            for f in ("meta", "tfidf"):
                c = C[(C.label == label) & (C.scheme == name) & (C.ptype == "B.I") & (C.unit == "pooled") & (C.feature == f)]
                a = o[f][0]
                add(dict(part="refit", key=key, dir="matrix_probes", source_csv="cells.csv", label=label, scheme=name,
                         ptype="B.I", unit="pooled", feature=f, n=n, pos=pos, auroc_table=c.auroc.iloc[0] if len(c) else np.nan,
                         auroc_recomputed=a, null_kind="refit, label permuted within source|author|arm",
                         **null_stats(a, o[f][1]), C=o[f][2] if f == "tfidf" else np.nan))
            log(f"nulls refit {key}: meta {o['meta'][0]:.3f} null95 {np.percentile(o['meta'][1], 95):.3f}; tfidf "
                f"{o['tfidf'][0]:.3f} null95 {np.percentile(o['tfidf'][1], 95):.3f} ({time.time() - t0:.0f}s)")
    del D_
    # ---- (b) matrix_probes: every other TF-IDF and metadata row
    if ("fixed", "matrix_probes") not in have:
        refit_keys = {(r["label"], r["scheme"], r["ptype"], r["unit"], r["feature"]) for r in rows if r["part"] == "refit"}
        for (label, scheme), cc in C[C.feature.isin(["tfidf", "meta"]) & C.auroc.notna()].groupby(["label", "scheme"], sort=False):
            src = ident.get((label, scheme), scheme)
            p = MPOUT / "scores" / label / f"{mp.slug(src)}.csv.gz"
            S = pd.read_csv(p)
            for r in cc.itertuples():
                if (label, scheme, r.ptype, r.unit, r.feature) in refit_keys:
                    continue
                seed = zlib.crc32(f"{label}|{scheme}|{r.ptype}|{r.unit}|{r.feature}".encode())
                if r.unit == "mean over authors":
                    per = S[(S.ptype == r.ptype) & (S.feature == r.feature) & (S.unit != "pooled")]
                    parts = [(d.score.values, d.y.values) for _, d in per.groupby("unit")
                             if np.isfinite(tp.auc(d.score.values, d.y.values))]
                    a = float(np.mean([tp.auc(s_, y_) for s_, y_ in parts]))
                    nl = np.mean([fixed_null(s_, y_, seed + j) for j, (s_, y_) in enumerate(parts)], 0)
                    n = int(sum(np.isfinite(s_).sum() for s_, _ in parts))
                    pos = int(sum(y_[np.isfinite(s_)].sum() for s_, y_ in parts))
                else:
                    d = S[(S.ptype == r.ptype) & (S.feature == r.feature) & (S.unit == r.unit)]
                    a = tp.auc(d.score.values, d.y.values)
                    nl = fixed_null(d.score.values, d.y.values, seed)
                    n, pos = int(np.isfinite(d.score.values).sum()), int(d.y.values[np.isfinite(d.score.values)].sum())
                rows.append(dict(part="fixed", key=f"{label}|{scheme}|{r.ptype}|{r.unit}|{r.feature}", dir="matrix_probes",
                                 source_csv="cells.csv", label=label, scheme=scheme, ptype=r.ptype, unit=r.unit,
                                 feature=r.feature, n=n, pos=pos, auroc_table=r.auroc, auroc_recomputed=a,
                                 null_kind="fixed held-out scores, labels permuted", seed=seed, **null_stats(a, nl)))
        rows.append(dict(part="fixed", key="matrix_probes", dir="marker"))
        pd.DataFrame(rows).to_csv(p_out, index=False)
        status("nulls", "matrix_probes fixed-score nulls done")
    # ---- (b) team_probes agent and team tables
    if ("fixed", "team_probes") not in have:
        for sub in ["", "sets/solo_design/", "sets/spar_mixed_41/", "sets/spar_screen_team/"]:
            for tab, sco in (("agent_probes.csv", "agent_scores.csv"), ("team_probe.csv", "team_scores.csv")):
                pt, ps = RES / "team_probes" / f"{sub}{tab}", RES / "team_probes" / f"{sub}{sco}"
                if not pt.is_file() or not ps.is_file():
                    continue
                T, S = pd.read_csv(pt), pd.read_csv(ps)
                for i, r in T[T.null_mean.isna() & T.auroc.notna()].iterrows():
                    B = "|B" if str(r.setting).startswith("B") else ""
                    col = f"score|{r.features}{B}"
                    d = S[S.probe == r.probe] if "probe" in S else S
                    if "subset" in T and "subset" in d:
                        d = d[d.subset == r.subset]
                    mm = re.match(r"(max|mean) over workers of the held-out agent collusion probe score, probe (L\w+) (.+)$",
                                  str(r.features))
                    if mm:  # team aggregate: per run, max or mean of the agent collusion probe's held-out scores
                        pos_ = {"think-open+close": "think", "think-open": "open", "think-close": "close"}[mm.group(3)]
                        Sa = pd.read_csv(RES / "team_probes" / f"{sub}agent_scores.csv")
                        Sa = Sa[Sa.probe == "2 collusion: exploit crossed or above vs none"]
                        agg = Sa.groupby("run")[f"score|{mm.group(2)}|{pos_}{B}"].agg(mm.group(1))
                        yt = S.set_index("run").y
                        agg = agg[agg.index.isin(yt.index)]
                        col = "aggregate"
                        d = pd.DataFrame({"aggregate": agg.values, "y": yt.loc[agg.index].values})
                    if col not in d:
                        rows.append(dict(part="fixed", key=f"team_probes/{sub}{tab}#{i}", dir="team_probes",
                                         source_csv=f"{sub}{tab}", label=r.probe, feature=r.features, unit=r.setting,
                                         auroc_table=r.auroc, note=f"score column {col!r} not found"))
                        continue
                    s_, y_ = d[col].values.astype(float), d["y"].values
                    a = tp.auc(s_, y_)
                    rows.append(dict(part="fixed", key=f"team_probes/{sub}{tab}#{i}", dir="team_probes",
                                     source_csv=f"{sub}{tab}", label=r.probe, feature=r.features, unit=r.setting,
                                     scheme=(r.subset if "subset" in T else np.nan), n=int(np.isfinite(s_).sum()), pos=int(y_[np.isfinite(s_)].sum()), auroc_table=r.auroc,
                                     auroc_recomputed=a, null_kind="fixed held-out scores, labels permuted",
                                     seed=4000 + i, **null_stats(a, fixed_null(s_, y_, 4000 + i))))
        rows.append(dict(part="fixed", key="team_probes", dir="marker"))
        pd.DataFrame(rows).to_csv(p_out, index=False)
        status("nulls", "team_probes fixed-score nulls done")
    # ---- (b) external_probes trajectory baselines
    if ("fixed", "external_probes") not in have:
        TJ = json.load(open(RES / "external_probes" / "traj.json"))
        SC = {k: pd.read_csv(RES / "external_probes" / f"traj_scores_{k}.csv.gz") for k in ("gf", "gfe", "cb", "eg")}
        for key, lst in TJ.items():
            if not isinstance(lst, list):
                continue
            for j, r in enumerate(lst):
                if not isinstance(r, dict) or "auroc" not in r or r.get("null_mean") is not None:
                    continue
                if any(t in key for t in ("_by", "_within", "_counts", "all13")):
                    continue  # subgroup and diagnostic rows: listed in the audit, not here
                f = r.get("features")
                hit = None
                for k, S in SC.items():
                    norm = lambda x: str(x).replace("|", "_").replace(" ", "_")  # noqa: E731
                    cand = S[S["eval"].map(norm) == norm(key)]
                    if len(cand) and f in cand:
                        hit = cand
                        break
                if hit is None:
                    rows.append(dict(part="fixed", key=f"external_probes/traj.json#{key}#{j}", dir="external_probes",
                                     source_csv="traj.json", label=key, feature=f, auroc_table=r["auroc"],
                                     note="score subset not found"))
                    continue
                s_, y_ = hit[f].values.astype(float), hit.y.values
                a = tp.auc(s_, y_)
                rows.append(dict(part="fixed", key=f"external_probes/traj.json#{key}#{j}", dir="external_probes",
                                 source_csv="traj.json", label=key, feature=f, n=int(np.isfinite(s_).sum()),
                                 pos=int(y_[np.isfinite(s_)].sum()), auroc_table=r["auroc"], auroc_recomputed=a,
                                 null_kind="fixed held-out scores, labels permuted", seed=6000 + j, **null_stats(a, fixed_null(s_, y_, 6000 + j))))
        rows.append(dict(part="fixed", key="external_probes", dir="marker"))
        pd.DataFrame(rows).to_csv(p_out, index=False)
    log("nulls done")


# ------------------------------------------------------------------------------------------------ report
f3, fint = mp.f3, mp.fint


def md_table(header, rows):
    """matrix_probes' markdown table with '|' inside a cell escaped (design arms are named like 'L1|clean')."""
    esc = lambda c: str(c).replace("|", "\\|")  # noqa: E731
    return mp.md_table([esc(h) for h in header], [[esc(c) for c in r] for r in rows])
SET_NAMES = {s: s for s in mp.SOURCES}


def fci(r):
    return mp.fci(r) or "n/a"


def fnull(r):
    m, q = r.get("null_mean"), r.get("null_95")
    return f"{m:.3f} / {q:.3f}" if m is not None and np.isfinite(m) else "n/a"


def verdict(main, full):
    """Plain call on the prompt-only probe: yes when it beats every one of the 50 permuted draws and reaches 0.60;
    weakly when it beats the null's 95th percentile only; no otherwise. Share = its margin over 0.5 as a fraction of
    the full probe's."""
    a, q, p = main.get("auroc"), main.get("null_95"), main.get("null_p")
    if a is None or not np.isfinite(a):
        return "n/a", "n/a"
    sh = (a - 0.5) / (full - 0.5) if full is not None and np.isfinite(full) and full > 0.5 else np.nan
    v = ("yes" if p is not None and p <= 1 / (N_NULL + 1) + 1e-9 and a >= 0.6 else
         "weakly" if q is not None and a > q else "no")
    return v, (f"{sh:.2f}" if np.isfinite(sh) else "n/a")


def report():
    import pandas as pd
    L = []
    w = L.append
    w("# Controls and nulls for the hermeneutic probes\n")
    w("Every number below was computed on this machine from the public Hub dataset unrulyabstractions/apart-swarm-align "
      "and the matrix_probes caches, with no model call and no spend. Folds, labels, schemes and the probe recipe "
      "(training-fold PCA to 128 components, z-scoring, balanced L2 logistic regression, C by grouped inner CV) are those "
      "of `hermeneutics/matrix_probes.py`, imported by `hermeneutics/controls_nulls.py`. AUROC intervals are 95% run "
      "bootstraps (1,000 draws). Nulls are label-permutation nulls: mean / 95th percentile of the AUROC over the draws, "
      "the probe refitted with C fixed at the median of the chosen Cs.\n")
    # ---------------- 1
    P = pd.read_csv(OUT / "prompt_pooled.csv")
    W_ = pd.read_csv(OUT / "prompt_within.csv")
    w("## 1. Is the label predictable from the prompt alone?\n")
    w("**Prompt-only probe.** The probe reads only states the reader has before the first assistant token. The main "
      "feature for our sets is the first assistant turn's five boundary tokens (`<|im_end|>` of the last prompt message, "
      "newline, `<|im_start|>`, `assistant`, newline) and the think-open state that follows them, at layers 26, 38, 51 "
      "(92,160 dims). The external sets have no turns.safetensors, so their main feature is the turn-0 think-open state "
      "alone (15,360 dims). The check feature is the token-weighted mean of the sentence means over the prompt units "
      "(system and user spans before the first assistant span), read for every agent of our sets by HTTP range requests "
      "over the prompt rows of acts.safetensors. The folds are matrix_probes' own B.I folds of each set alone. The null "
      "permutes the label over all agents of the set (50 draws).\n")
    w("Prompt identity, beside it: out-of-fold scores equal to the training fold's label rate among agents of the same "
      "arm, of the same (task, arm), or with the byte-identical prompt text (rendered input up to the first "
      "`<|im_start|>assistant`); an unseen key falls back to the next key, then to the overall rate. They show how much "
      "of the label the prompt condition carries without any activation. In controls_600, board_seed, solo_design, "
      "spar_mixed_41 and spar_exploration_single every prompt is unique (it names the agent, seed or files), so prompt-text "
      "identity falls back to the overall rate; the distinct count is beside it. Out-of-fold rates of rare keys are biased "
      "below 0.5.\n")
    w("Verdict: **yes** when the prompt-only AUROC beats all 50 permuted draws and reaches 0.60, **weakly** when it beats "
      "the null's 95th percentile only, **no** otherwise. Share: the prompt-only margin over 0.5 as a fraction of the "
      "full probe's margin.\n")
    hdr = ["Label", "Set", "Agents", "Positive", "Runs", "Prompt-only main feature", "Prompt-only AUROC [95% CI]",
           "Null mean / 95th", "Think-open only", "Boundary only", "Prompt sentence means", "Arm identity",
           "Task + arm identity", "Prompt-text identity (distinct prompts)", "Full probe (think pair)",
           "Full probe (seven tokens)", "Metadata", "TF-IDF rendered text", "Predictable from the prompt?", "Share of full margin"]
    rows = []
    for (lab, s), d in P.groupby(["label", "set"], sort=False):
        g = {r.feature: r._asdict() for r in d.itertuples()}
        main = "first" if "first" in g else "open0"
        m = g[main]
        v, sh = verdict(m, g.get("think", {}).get("auroc"))
        rows.append([mp.LABELS[lab], s, fint(m["n"]), fint(m["pos"]), fint(m["runs"]), PFEATS[main].split(",")[0], fci(m),
                     fnull(m), f3(g["open0"]["auroc"]) if main != "open0" else "(main)",
                     f3(g["bound0"]["auroc"]) if "bound0" in g else "n/a",
                     f3(g["means"]["auroc"]) if "means" in g else "n/a", f3(g["arm_rate"]["auroc"]),
                     f3(g["task_arm_rate"]["auroc"]),
                     f"{f3(g['prompt_rate']['auroc'])} ({fint(g['prompt_rate']['distinct_keys'])})",
                     fci(g["think"]) if "think" in g else "n/a", f3(g["seven"]["auroc"]) if "seven" in g else "n/a",
                     f3(g["meta"]["auroc"]) if "meta" in g else "n/a", f3(g["tfidf"]["auroc"]) if "tfidf" in g else "n/a", v, sh])
    w(md_table(hdr, rows))
    w("\nProvenance: `results/controls_nulls/prompt_pooled.csv` (prompt-only rows, this script) and matrix_probes "
      "`scores/<label>/alone_<set>.csv.gz` B.I pooled (full probe, seven tokens, metadata, TF-IDF on the same agents and "
      "folds); out-of-fold scores in `results/controls_nulls/scores/prompt_scores.csv.gz`.\n")
    w("**Within each prompt condition.** Two numbers per condition. *Restricted* is the AUROC of the pooled out-of-fold "
      "scores above on the agents of that condition only. *Refit* is a probe trained and tested inside the condition on "
      "new grouped folds (stratified by author and label, grouped by run), with a 50-draw null for the prompt-only refit. "
      "Conditions: controls solo vs cheat arm; design level (L0, L1, L2) and post (team hint vs clean) and their cross "
      "(the arm); team runs by seed arm; SPAR sets by condition; external sets by benchmark, harness or condition. In "
      "board_seed and solo_design the seed post is planted on the shared board, which the agent reads through a tool "
      "call, so it is never in the prompt; the design L2 arm names the board in the system prompt. *Restricted: task + arm "
      "identity* is shown only where the condition holds more than one task: it then says how much of the within-condition "
      "ranking the task alone gives. Out-of-fold label rates of small cells are biased below 0.5 (a held-out positive "
      "lowers its own cell's training rate), so values under 0.5 there mean no signal.\n")
    hdr = ["Label", "Set", "Condition", "Value", "Agents", "Positive", "Runs", "Restricted: prompt-only", "Restricted: full",
           "Restricted: metadata", "Restricted: task + arm identity", "Refit: prompt-only [95% CI]", "Refit prompt-only null 95th", "Refit: full",
           "Refit: metadata", "Refit: prompt sentence means"]
    rows = []
    Ag = agents()
    ntask = {(s, cc, str(v)): g.task.nunique() for s in W_.set.unique() for cc in ("arm", "level", "seed_text")
             for v, g in Ag[Ag.source == s].groupby(Ag[Ag.source == s][cc].astype(str))}
    for (lab, s, cc, val), d in W_.groupby(["label", "set", "condition", "value"], sort=False):
        R = {(r.mode, r.feature): r._asdict() for r in d.itertuples()}
        main = "first" if ("pooled scores restricted", "first") in R else "open0"
        rr = lambda f: R.get(("pooled scores restricted", f), {})  # noqa: E731
        fr = lambda f: R.get(("refit within condition", f), {})  # noqa: E731
        any_ = d.iloc[0]
        note = fr(main).get("note") if isinstance(fr(main).get("note"), str) else None
        if note is None and not np.isfinite(float(fr(main).get("auroc", np.nan))):
            note = "not scorable: each test fold held one class"
        rows.append([mp.LABELS[lab], s, {"arm": "arm", "level": "level", "seed_text": "post"}[cc], val, fint(any_.n),
                     fint(any_.pos), fint(any_.runs), f3(rr(main).get("auroc")), f3(rr("think").get("auroc")),
                     f3(rr("meta").get("auroc")), (f3(rr("task_arm_rate").get("auroc")) if ntask.get((s, cc, str(val)), 1) > 1 else "one task"),
                     note or fci(fr(main)), "" if note else f3(fr(main).get("null_95")),
                     "" if note else f3(fr("think").get("auroc")), "" if note else f3(fr("meta").get("auroc")),
                     "" if note else f3(fr("means").get("auroc"))])
    w(md_table(hdr, rows))
    w("\nProvenance: `results/controls_nulls/prompt_within.csv`; refit scores in `scores/prompt_scores.csv.gz` "
      "(feature `within:<condition>=<value>:<feature>`). An empty restricted cell: the condition holds one class only.\n")
    # ---------------- 2
    if (OUT / "author_rows.csv").is_file():
        R = pd.read_csv(OUT / "author_rows.csv")
        w("## 2. Does the type B probe read author identity?\n")
        w("On every pooled scheme with more than one author, per training fold: the training-fold PCA-z components the "
          "probe reads (think pair, layers 26+38+51, 128 components), an author-identity probe on them (multinomial "
          "logistic regression, C 1; chance is predicting the training fold's most frequent author), then LEACE (Belrose "
          "et al. 2023) fitted on the training fold erases author (one-hot), or author + task + condition (arm), from "
          "those components, and B.I and the author probe are refitted on the erased components. B.II (leave one author "
          "out) likewise, with the eraser fitted on the training authors. The raw rows recompute matrix_probes' B.I and "
          "B.II and equal them on every row (see VERIFICATION.md). coord and collude 'all' equal 'ours' (no external "
          "source carries those labels). Author accuracy at chance after erasure follows from LEACE by construction on the "
          "training fold: it confirms the eraser ran, and it does not rule out author information a nonlinear reader "
          "could still find.\n")
        hdr = ["Label", "Scheme", "Authors", "Agents", "Positive", "Author probe: raw", "Author: after author erasure",
               "Author: after author+task+condition erasure", "Chance", "B.I raw", "B.I author-erased",
               "B.I author+task+condition-erased", "B.II mean raw", "B.II mean author-erased",
               "B.II mean author+task+condition-erased", "B.II authors"]
        rows = []
        for (lab, sc), d in R.groupby(["label", "scheme"], sort=False):
            bi = {r.erase: r._asdict() for r in d[d.ptype == "B.I"].itertuples()}
            bm = {r.erase: r._asdict() for r in d[(d.ptype == "B.II") & (d.unit == "mean over authors")].itertuples()}
            rr = bi["raw"]
            rows.append([mp.LABELS[lab], sc, fint(rr["authors"]), fint(rr["n"]), fint(rr["pos"]), f3(rr["author_acc"]),
                         f3(bi["author"]["author_acc"]), f3(bi["author+task+condition"]["author_acc"]), f3(rr["author_chance"]),
                         fci(rr), fci(bi["author"]), fci(bi["author+task+condition"]),
                         fci(bm.get("raw", {})), fci(bm.get("author", {})), fci(bm.get("author+task+condition", {})),
                         fint(bm.get("raw", {}).get("authors"))])
        w(md_table(hdr, rows))
        w("\nProvenance: `results/controls_nulls/author_rows.csv` (per-author B.II rows there too); scores in "
          "`scores/author/`. Author-probe accuracy is over all held-out B.I folds.\n")
    # ---------------- 3
    if (OUT / "lexical_rows.csv").is_file():
        X = pd.read_csv(OUT / "lexical_rows.csv")
        C = pd.read_csv(MPOUT / "cells.csv")
        w("## 3. Lexical-only reader null: how much does the reasoning text alone give?\n")
        w("TF-IDF (word unigrams and bigrams, vocabulary, idf and C fitted on the training side) with L2 logistic "
          "regression on the think-window text only: every span between `<think>` and `</think>` in the reader's "
          "rendered input, all turns joined per agent. It runs on matrix_probes' own scheme runner, with the same folds "
          "and splits; the metadata rows that runner recomputes equal matrix_probes' on every row (VERIFICATION.md). "
          "Beside it: the TF-IDF of the whole rendered text and the full probe (think pair) from matrix_probes. Claude "
          "Haiku 5.5's reasoning is not exposed (its think windows are empty), and several external authors' windows are empty "
          "too: an agent with an empty think window is left out of training and gets no lexical score, so a held-out author "
          "made only of such agents gets no AUROC (n/a) and drops out of the mean over authors.\n")
        hdr = ["Label", "Scheme", "Probe type", "Unit", "Agents", "Positive", "Think-window TF-IDF",
               "Agents with a think-window score",
               "Whole-text TF-IDF", "Full probe (think pair)", "Probe minus think-window"]
        rows = []
        X = X[X.feature == "tfidf_think"]
        for r in X.itertuples():
            if r.unit not in ("pooled", "mean over authors"):
                continue
            c = C[(C.label == r.label) & (C.scheme == r.scheme) & (C.ptype == r.ptype) & (C.unit == r.unit)]
            ct = c[c.feature == "tfidf"]
            cp = c[c.feature == "think"]
            pa = cp.auroc.iloc[0] if len(cp) else np.nan
            rows.append([mp.LABELS[r.label], r.scheme, r.ptype, r.unit, fint(ct.n.iloc[0]) if len(ct) else "",
                         fint(ct.pos.iloc[0]) if len(ct) else "", fci(r._asdict()), fint(r.n),
                         fci(ct.iloc[0].to_dict()) if len(ct) else "n/a", fci(cp.iloc[0].to_dict()) if len(cp) else "n/a",
                         f3(pa - r.auroc) if np.isfinite(pa) and np.isfinite(r.auroc) else "n/a"])
        w(md_table(hdr, rows))
        w("\nProvenance: `results/controls_nulls/lexical_rows.csv` (per-author A and B.II rows there too), scores in "
          "`scores/lexical/`; whole-text TF-IDF and full probe from matrix_probes `cells.csv`.\n")
    # ---------------- 4
    p4 = OUT / "nulls_rows.csv"
    if p4.is_file():
        report_nulls(w)
    # ---------------- 5
    w("## 5. Random-weights reader\n")
    w("Not run: it needs a GPU pass. The plan (subsample of 911 agents, 31.1M tokens, the extractor change, box time, "
      "files and the probes to rerun) is in `RANDOM_READER_PLAN.md`.\n")
    (OUT / "RESULTS.md").write_text("\n".join(L))
    log(f"report: {len(L)} blocks")


COMPUTED_PATTERNS = [  # (dir, source_csv, feature_or_method) of null_audit.csv lines whose rows nulls() fills
    ("matrix_probes", "matrix_probes/cells.csv", "baseline: TF-IDF text"),
    ("matrix_probes", "matrix_probes/cells.csv", "baseline: metadata"),
    ("team_probes", "team_probes/agent_probes.csv", "baseline: tfidf"),
    ("team_probes", "team_probes/agent_probes.csv", "baseline: meta"),
    ("team_probes", "team_probes/team_probe.csv", "aggregate of held-out agent probe scores"),
    ("team_probes", "team_probes/team_probe.csv", "baseline: tfidf"),
    ("team_probes", "team_probes/team_probe.csv", "baseline: meta"),
] + [("team_probes", f"team_probes/sets/{s}/{t}", m) for s in ("solo_design", "spar_mixed_41", "spar_screen_team")
     for t, m in (("agent_probes.csv", "baselines: TF-IDF, metadata"), ("team_probe.csv", "aggregate of held-out agent probe scores"),
                  ("team_probe.csv", "baselines: TF-IDF, metadata"))]


def _computed(r):
    if r.dir == "external_probes" and str(r.feature_or_method) == "baselines: metadata, TF-IDF" and \
            str(r.source_csv).startswith("external_probes/traj.json keys gf_5fold,"):
        return True
    if r.dir == "external_probes" and str(r.source_csv).startswith("external_probes/traj.json key gf_5fold_task"):
        return True
    return (r.dir, r.source_csv, r.feature_or_method) in COMPUTED_PATTERNS


def report_nulls(w):
    import pandas as pd
    N = pd.read_csv(OUT / "nulls_rows.csv")
    N = N[N.dir != "marker"]
    Au = pd.read_csv(AUDIT)
    w("## 4. Does every main table carry a label-permutation null?\n")
    w("No. The audit read every main table of the four RESULTS.md files and matched each row to its CSV. The rows "
      "without a null are the baselines (TF-IDF, metadata), the team aggregates, the transfer cells, and most of the "
      "per-step and attention-monitor rows.\n")
    rows = []
    for d, g in Au.groupby("dir"):
        rows.append([d, fint(g[g.has_null == "yes"].n_rows_represented.sum()), fint(g[g.has_null == "no"].n_rows_represented.sum()),
                     fint(g[(g.has_null == "no") & g.apply(_computed, axis=1)].n_rows_represented.sum())])
    w(md_table(["Results dir", "Rows with a null", "Rows without", "Rows without, null computed here"], rows))
    w("\nProvenance: `results/controls_nulls/null_audit.csv`, one line per table pattern, counts from the CSVs.\n")
    w("**Refit nulls (the probe's own protocol).** These are matrix_probes' TF-IDF and metadata B.I pooled rows of every "
      "cross-validated scheme. The label is permuted within (source, author, arm) strata, as for the probe. There are 50 "
      "draws, and the baseline is refitted on each draw on the same folds. TF-IDF uses C fixed at the median of the "
      "per-fold C chosen on the real labels. Every real-label AUROC was recomputed and equals cells.csv.\n")
    R = N[N.part == "refit"]
    rows = [[mp.LABELS[r.label], r.scheme, BASES_NAME[r.feature], fint(r.n), fint(r.pos), f3(r.auroc_recomputed),
             f"{r.null_mean:.3f} / {r.null_95:.3f}", f"{r.null_p:.3f}", "yes" if r.null_p <= 0.05 else "no"]
            for r in R.itertuples()]
    w(md_table(["Label", "Scheme", "Baseline", "Agents", "Positive", "AUROC", "Null mean / 95th", "p (50 draws)",
                "Beats the null (p <= 0.05)?"], rows))
    w("\nProvenance: `results/controls_nulls/nulls_rows.csv` (part refit).\n")
    F = N[N.part == "fixed"].copy()
    F["match"] = (F.auroc_recomputed - F.auroc_table).abs() < TOL
    F["above"] = F.null_p <= 0.05
    w("**Fixed-score nulls.** These cover every other row without a null whose held-out scores are saved: matrix_probes "
      "TF-IDF and metadata rows (types A and B.II, leave-one-out and transfer cells), the team_probes agent and team "
      "baselines and team aggregates, and the external_probes trajectory baselines. The saved out-of-fold scores are "
      "kept, and the labels are permuted 50 times (for a mean-over-authors row, the per-author draws are averaged). This "
      "tests that the held-out AUROC beats chance. It does not refit, so it is weaker than the refit null. A row is used "
      "only when the AUROC recomputed from its scores equals the table's within 0.001 (the matrix_probes score files round "
      "scores to 6 significant digits). The team aggregates are rebuilt per run, as the max or mean of the agent collusion "
      "probe's held-out scores, and the team labels are permuted.\n")
    F["what"] = np.where(F.dir == "matrix_probes", F.feature.map(lambda x: BASES_NAME.get(x, x)),
                         np.where(F.feature.astype(str).str.contains("over workers"), "team aggregate of agent probe",
                                  np.where(F.feature.astype(str).str.contains("TF-IDF"), "TF-IDF",
                                           np.where(F.feature.astype(str).str.contains("think pair"), "linear probe",
                                                    "metadata"))))
    rows = []
    for (d, src, wh), g in F.groupby(["dir", "source_csv", "what"], sort=False):
        ok = g[g.match]
        rows.append([d, src, wh, fint(len(g)), fint(len(ok)), fint(int(ok.above.sum())), fint(int((~ok.above).sum())),
                     f3(ok.null_95.max()) if len(ok) else ""])
    w(md_table(["Results dir", "Table", "Row kind", "Rows", "Scores reproduce the table", "Beats the null (p <= 0.05)",
                "Does not", "Largest null 95th"], rows))
    nm = F[~F.match]
    w(f"\nRows whose saved scores did not reproduce the table's AUROC (no null attached): {len(nm)}"
      + ("." if not len(nm) else ": " + "; ".join(f"{r.dir} {r.source_csv} {r.label} {r.feature} {r.unit} (table "
                                                 f"{f3(r.auroc_table)}, scores {f3(r.auroc_recomputed)}{', ' + r.note if isinstance(r.note, str) else ''})"
                                                 for r in nm.itertuples())))
    na = F[F.match & ~F.above]
    w("\nRows that do not beat their null (p > 0.05):\n")
    w(md_table(["Results dir", "Table", "Row", "Feature", "Unit", "Agents", "AUROC", "Null 95th", "p"],
               [[r.dir, r.source_csv, f"{r.label} {r.scheme if isinstance(r.scheme, str) else ''} {r.ptype if isinstance(r.ptype, str) else ''}",
                 r.feature, r.unit, fint(r.n), f3(r.auroc_recomputed), f3(r.null_95), f"{r.null_p:.3f}"] for r in na.itertuples()]))
    w("\nProvenance: `results/controls_nulls/nulls_rows.csv` (part fixed, one line per row, with the seed).\n")
    w("**Rows still without a null.** These patterns were not computed here, for the reason given.\n")
    rest = Au[(Au.has_null == "no") & ~Au.apply(_computed, axis=1)]
    rows = []
    for r in rest.itertuples():
        sf = str(r.score_file_for_recompute)
        why = ("no per-row scores saved" if sf.lower().startswith("none") or "not saved" in sf.lower() or "none saved" in sf.lower()
               else "per-step or per-group scores: a valid permutation must keep each agent's turns together (labels "
               "permuted by agent, then the step metric recomputed); not done")
        rows.append([r.dir, str(r.results_section)[:80], str(r.feature_or_method)[:70], fint(r.n_rows_represented), why])
    w(md_table(["Results dir", "Section", "Rows", "Count", "Why not computed"], rows))
    w(f"\nTotal still without a null: {fint(rest.n_rows_represented.sum())} rows.\n")


BASES_NAME = {"tfidf": "TF-IDF rendered text", "meta": "metadata"}
TOL = 1e-3  # recomputed vs table AUROC: matrix_probes' score files round scores to 6 significant digits


def verify_nulls(w, chk):
    import pandas as pd
    N = pd.read_csv(OUT / "nulls_rows.csv")
    N = N[N.dir != "marker"]
    w("\n## 4. Label-permutation nulls\n")
    R = N[N.part == "refit"]
    d = (R.auroc_recomputed - R.auroc_table).abs()
    chk("refit nulls: the real-label AUROC recomputed by the null code equals cells.csv for every TF-IDF and metadata "
        "B.I pooled row; at least 45 of the 50 draws finite", d.max() < 1e-9 and bool((R.draws >= 45).all()),
        f"{len(R)} rows, max |difference| {d.max():.2e}, finite draws per row {sorted(set(R.draws.astype(int)))} (a draw is "
        f"undefined when a permutation leaves a training fold with one class: "
        f"{'; '.join(f'{r.label} {r.scheme} {r.feature}: {int(r.draws)}' for r in R[R.draws < N_NULL].itertuples()) or 'none'})")
    F = N[N.part == "fixed"]
    dd = (F.auroc_recomputed - F.auroc_table).abs()
    ok = dd < TOL
    chk("fixed-score nulls: saved scores reproduce the table AUROC", bool(ok.all()),
        f"{int(ok.sum())} of {len(F)} rows reproduce within {TOL:g} (matrix_probes saves scores to 6 significant digits); "
        f"max |difference| among them {dd[ok].max():.1e}; rows that do not reproduce carry no null and are listed in RESULTS.md")
    r = F[ok & (F.dir == "matrix_probes") & (F.unit != "mean over authors")].iloc[0]
    S = pd.read_csv(MPOUT / "scores" / r.label / f"{mp.slug(r.scheme)}.csv.gz")
    e = S[(S.ptype == r.ptype) & (S.unit == r.unit) & (S.feature == r.feature)]
    st = null_stats(tp.auc(e.score.values, e.y.values), fixed_null(e.score.values, e.y.values, int(r.seed)))
    chk("one fixed-score null recomputed from its seed", abs(st["null_95"] - r.null_95) < 1e-12,
        f"{r.label} {r.scheme} {r.ptype} {r.unit} {r.feature}: null 95th {r.null_95:.4f} saved, {st['null_95']:.4f} recomputed")


# ------------------------------------------------------------------------------------------------ verify
def sp_load_nnz():
    """uid -> number of hashed think-window buckets (0 for an empty think window)."""
    import scipy.sparse as sp
    A = agents()
    H = sp.load_npz(CACHE / "think_counts.npz").tocsr()
    return dict(zip(A.uid, H.getnnz(1)))


def _grouped_ok(uids, fold, group_of):
    """True when no run (group) has agents in two folds."""
    g = [group_of[u] for u in uids]
    seen = {}
    for gg, f in zip(g, fold):
        if seen.setdefault(gg, f) != f:
            return False
    return True


def verify():
    """Recomputes AUROCs from the saved scores (this module's auc and sklearn's roc_auc_score), checks the folds, and
    checks LEACE (author accuracy after erasure against chance on held-out folds, training cross-covariance)."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    A = agents()
    group_of = dict(zip(A.uid, A.group))
    L, bad = [], []
    w = L.append
    w("# Verification of the controls and nulls\n")
    w("Every check below was run by `python -m hermeneutics.controls_nulls verify` on the saved files, "
      f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.\n")

    def chk(name, ok, detail):
        w(f"- {'PASS' if ok else 'FAIL'}: {name}. {detail}")
        if not ok:
            bad.append(name)
    # ---- 1
    w("## 1. Prompt-only probe\n")
    P = pd.read_csv(OUT / "prompt_pooled.csv")
    S = pd.read_csv(OUT / "scores" / "prompt_scores.csv.gz")
    diffs, n = [], 0
    for r in P.itertuples():
        d = S[(S.label == r.label) & (S.set == r.set) & (S.feature == r.feature)]
        a = tp.auc(d.score.values, d.y.values)
        diffs.append(abs(a - r.auroc))
        n += 1
    chk("every pooled AUROC of table 1 recomputed from scores/prompt_scores.csv.gz", max(diffs) < 1e-6,
        f"{n} rows, max |difference| {max(diffs):.2e}")
    r0 = P[(P.set == "controls_600") & (P.feature == "first")].iloc[0]
    d = S[(S.label == "cheat") & (S.set == "controls_600") & (S.feature == "first")]
    ska = roc_auc_score(d.y, d.score)
    chk("one AUROC by sklearn: cheating, controls_600, prompt-only main feature", abs(ska - r0.auroc) < 1e-6,
        f"table {r0.auroc:.6f}, sklearn roc_auc_score {ska:.6f}, {len(d)} agents")
    # folds equal matrix_probes' and grouped
    fd, gd, nf = 0, 0, 0
    for (lab, s), d in S[~S.feature.str.startswith("within:")].groupby(["label", "set"]):
        ref = mp_scores(lab, f"alone: {mp.SHORT[s]}")["think"]
        for f, e in d.groupby("feature"):
            nf += 1
            fd += int((ref.loc[e.uid, "fold"].values != e.fold.values).any())
            gd += int(not _grouped_ok(e.uid.values, e.fold.values, group_of))
    chk("pooled prompt-only folds equal matrix_probes' B.I folds (by agent) and keep every run in one fold", fd == 0 and gd == 0,
        f"{nf} (label, set, feature) score sets, {fd} with a different fold, {gd} with a run split across folds")
    W_ = pd.read_csv(OUT / "prompt_within.csv")
    diffs, gd, nw = [], 0, 0
    for r in W_[W_["mode"] == "refit within condition"].dropna(subset=["auroc"]).itertuples():
        d = S[(S.label == r.label) & (S.set == r.set) & (S.feature == f"within:{r.condition}={r.value}:{r.feature}")]
        diffs.append(abs(tp.auc(d.score.values, d.y.values) - r.auroc))
        gd += int(not _grouped_ok(d.uid.values, d.fold.values, group_of))
        nw += 1
    chk("every within-condition refit AUROC recomputed from its saved scores; refit folds grouped by run",
        max(diffs) < 1e-6 and gd == 0, f"{nw} rows, max |difference| {max(diffs):.2e}, {gd} with a run split across folds")
    diffs = []
    for r in W_[W_["mode"] == "pooled scores restricted"].dropna(subset=["auroc"]).itertuples():
        d = S[(S.label == r.label) & (S.set == r.set) & (S.feature == r.feature)]
        d = d[A.set_index("uid").loc[d.uid, r.condition].astype(str).values == str(r.value)]
        diffs.append(abs(tp.auc(d.score.values, d.y.values) - r.auroc))
    chk("every restricted AUROC recomputed by filtering the pooled scores to the condition", max(diffs) < 1e-6,
        f"{len(diffs)} rows, max |difference| {max(diffs):.2e}")
    C = json.load(open(CACHE / "feats_checks.json"))
    chk("turn-0 boundary tokens contiguous and ending one token before the turn-0 think-open, every agent with "
        "turns.safetensors", set(C) == {"boundary_contiguous_and_adjacent"}, json.dumps(C))
    M = pd.read_json(CACHE / "prompt_means_checks.jsonl", lines=True)
    okm = M[M.get("ok", 0) == 1]
    adj = int((okm.prompt_tokens_last == okm.turn0_boundary_first).sum())
    chk("prompt sentence means: header unit count equals spans.jsonl, prompt units end at the turn-0 boundary token",
        len(okm) == len(M) and adj == len(okm), f"{len(okm)} of {len(M)} agents read; last prompt token end equals the "
        f"first boundary position on {adj}; first assistant span: {dict(Counter(okm.first_assistant_segment))}")
    T = pd.read_csv(CACHE / "think_text_checks.csv")
    T["src"] = T.uid.str.split("/").str[0]
    okuid = set(A.uid[A.ok == 1])
    n_out = int((~T.uid.isin(okuid)).sum())
    T = T[T.uid.isin(okuid)]  # the agents with zero turns have no features and no <think>
    roles = T.groupby("src").roles_before_first_think.agg(lambda v: dict(Counter(v)))
    chk("external sets: only system and user messages precede the first <think> (turn-0 think-open is prompt-only)",
        all(set(r) <= {"system/user/assistant", "user/assistant", "system/assistant"} for r in roles),
        "; ".join(f"{k}: {v}" for k, v in roles.items()) + f" ({n_out} agents with zero turns left out, as in matrix_probes)")
    # ---- 2
    if (OUT / "author_rows.csv").is_file():
        w("\n## 2. Author-identity control\n")
        R = pd.read_csv(OUT / "author_rows.csv")
        raw = R[R.erase == "raw"]
        dd = (raw.auroc - raw.matrix_auroc).abs()
        chk("raw B.I and B.II rows (pooled, every held-out author, mean over authors) equal matrix_probes cells.csv think",
            bool(raw.matrix_auroc.notna().all()) and dd.max() < 1e-9, f"{len(raw)} rows, max |difference| {dd.max():.2e}")
        diffs, nn = [], 0
        for r in R[(R.ptype == "B.I")].itertuples():
            f = OUT / "scores" / "author" / f"{r.label}_{mp.slug(r.scheme)}_BI_{mp.slug(r.erase)}.csv.gz"
            d = pd.read_csv(f)
            diffs.append(abs(tp.auc(d.score.values, d.y.values) - r.auroc))
            nn += 1
        for (lab, sc), d0 in R[(R.ptype == "B.II") & (R.unit != "mean over authors")].groupby(["label", "scheme"]):
            d = pd.read_csv(OUT / "scores" / "author" / f"{lab}_{mp.slug(sc)}_BII.csv.gz")
            for r in d0.itertuples():
                e = d[(d.erase == r.erase) & (d.unit == r.unit)]
                diffs.append(abs(tp.auc(e.score.values, e.y.values) - r.auroc))
                nn += 1
        chk("every B.I and per-author B.II AUROC of table 2 recomputed from scores/author (scores saved to 6 significant "
            "digits)", max(diffs) < TOL, f"{nn} rows, max |difference| {max(diffs):.2e}")
        er = R[(R.ptype == "B.I") & (R.erase != "raw")]
        gap = (er.author_acc - er.author_chance)
        cc = er.max_cross_cov_after.max()
        chk("LEACE: author-probe accuracy on held-out B.I folds after erasure within 0.02 of chance (training fold's most "
            "frequent author), and training cross-covariance of the erased components with the concept below 1e-6",
            bool((gap.abs() <= 0.02).all()) and cc < 1e-6,
            f"{len(er)} erased rows; accuracy minus chance from {gap.min():+.3f} to {gap.max():+.3f}; raw accuracy from "
            f"{R[(R.ptype == 'B.I') & (R.erase == 'raw')].author_acc.min():.3f} to "
            f"{R[(R.ptype == 'B.I') & (R.erase == 'raw')].author_acc.max():.3f}; max cross-covariance {cc:.1e}")
        for r in er.itertuples():
            w(f"    - {r.label}, {r.scheme}, {r.erase}: author accuracy {r.author_acc:.3f} vs chance {r.author_chance:.3f} "
              f"over {r.n_author_test} held-out agents")
    # ---- 3
    if (OUT / "lexical_rows.csv").is_file():
        w("\n## 3. Lexical-only reader null\n")
        X = pd.read_csv(OUT / "lexical_rows.csv")
        Cc = pd.read_csv(MPOUT / "cells.csv")
        mm = X[X.feature == "meta"].merge(Cc[Cc.feature == "meta"], on=["label", "scheme", "ptype", "unit"], suffixes=("", "_mp"))
        dd = (mm.auroc - mm.auroc_mp).abs()
        chk("metadata rows recomputed by the same runner equal matrix_probes cells.csv on every (label, scheme, type, unit): "
            "same folds and splits", len(mm) > 0 and dd.max() < 1e-9, f"{len(mm)} rows matched, max |difference| {dd.max():.2e}")
        diffs, nn = [], 0
        for (lab, sc), d0 in X[(X.feature == "tfidf_think") & ~X.unit.isin(["mean over authors"])].dropna(subset=["auroc"]).groupby(["label", "scheme"]):
            f = OUT / "scores" / "lexical" / lab / f"{mp.slug(sc)}.csv.gz"
            d = pd.read_csv(f)
            d = d[d.feature == "tfidf_think"]
            for r in d0.itertuples():
                e = d[(d.ptype == r.ptype) & (d.unit == r.unit)]
                diffs.append(abs(tp.auc(e.score.values, e.y.values) - r.auroc))
                nn += 1
        chk("every think-window TF-IDF AUROC (pooled and per author) recomputed from scores/lexical (matrix_probes' "
            "save_frames writes 6 significant digits)", max(diffs) < TOL, f"{nn} rows, max |difference| {max(diffs):.2e}")
        const, units = [], 0
        for f in sorted((OUT / "scores" / "lexical").glob("*/*.csv.gz")):
            d = pd.read_csv(f)
            for (pt, un), e in d[d.feature == "tfidf_think"].groupby(["ptype", "unit"]):
                v = e.score.values[np.isfinite(e.score.values)]
                units += 1
                if len(v) > 1 and np.ptp(v) == 0:
                    const.append(f"{f.parent.name}/{f.name} {pt} {un}")
        Hn = sp_load_nnz()
        empty_scored = 0
        for f in sorted((OUT / "scores" / "lexical").glob("*/*.csv.gz")):
            d = pd.read_csv(f)
            d = d[(d.feature == "tfidf_think") & np.isfinite(d.score)]
            empty_scored += int((d.uid.map(Hn) == 0).sum())
        chk("no think-window unit scores all its agents alike, and no agent with an empty think window gets a score",
            not const and empty_scored == 0, f"{units} (file, type, unit) score sets; {len(const)} constant "
            f"{const[:5]}; {empty_scored} scored agents with an empty window")
        mis = T[T.windows != T.n_turns]
        w(f"- NOTE: think windows found equal meta n_turns on {len(T) - len(mis)} of {len(T)} agents; the {len(mis)} others "
          f"({', '.join(sorted(set(mis.src)))}) hold more `<think>` tags than turns (tags inside tool output or board "
          "posts); their text between those tags is kept.")
    # ---- 4
    if (OUT / "nulls_rows.csv").is_file():
        verify_nulls(w, chk)
    w(f"\n**Summary: {'all checks PASS' if not bad else 'FAILED: ' + '; '.join(bad)}.**\n")
    (OUT / "VERIFICATION.md").write_text("\n".join(L))
    log(f"verify: {len(bad)} failures {bad}")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--only", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fn = globals()[a.stage]
    status(a.stage, "start")
    if a.stage == "means":
        fn(jobs=a.jobs)
    elif a.only is not None:
        fn(only=a.only)
    else:
        fn()
    status(a.stage, "done")


if __name__ == "__main__":
    main()
