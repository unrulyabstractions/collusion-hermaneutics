"""Hermeneutic probes on three externally sourced activation sets: Goodfire reward-hack rollouts, CheatBench, EvilGenie.

The transcripts were written by other labs' agents and labelled by their authors (Goodfire/reward-hack-data,
steinad/CheatBench, WiserHumanExperimental/evilgenie-escalation). Another agent rendered them in Qwen3.8-27B's chat
template, teacher-forced them through the reader and put the think-token activations on the public Hub dataset
unrulyabstractions/apart-swarm-align (activations/<set>/<run_dir>/<rollout>/). team_probes.py fetched the think tensors
to data/acts_hub/think/<set>/<run_dir>/<rollout>.npz. No model calls, no spend, this machine only.

Labels.
  Goodfire: the source dataset at revision GF_REV, per run dir labels/judge_rollouts.jsonl and judge_passages.jsonl
  (SWE-bench Verified, DeepSWE, ImpossibleBench: a 3-vote consensus judge, the source's ground truth) or
  generic_monitor/monitor_rollouts.jsonl and monitor_passages.jsonl (tau3-bench, ShoppingBench: a single-pass generic
  monitor; the source says no ground truth exists there). Rollout = traces[0].id = the folder name. A passage's
  node_index is joined to the reader turn through meta.json turn_nodes (the sampled assistant node of each turn), then
  turn_path_nodes (the path node the renderer replaced by its sampled sibling).
  CheatBench: meta.json label (cheating / attempt / benign) and cheat (1 for cheating or attempt). EvilGenie: meta.json
  sel_label (hack / legitimate), cheat, and the matched pair (sel_pair_with).

Probes: team_probes' recipe (training-fold PCA to min(128, half the training rows), z-score, L2 logistic regression with
C by inner grouped CV, class-balanced), on the mean over turns of the think pair (think-open and think-close) at layers
26, 38, 51 and the three concatenated. Bootstrap intervals over rollouts (runs for the team sets), a permutation null of
50 draws (labels permuted within run dir, PCA fixed, C fixed at the median chosen C).

    python -m hermeneutics.external_probes labels      label tables, joins and checks
    python -m hermeneutics.external_probes cache       per-rollout means and rendered-text words (data/external_probes)
    python -m hermeneutics.external_probes traj        trajectory-level probes on the three sets
    python -m hermeneutics.external_probes step        Goodfire per-step linear probe and attention monitor
    python -m hermeneutics.external_probes transfer    controls <-> Goodfire, Goodfire -> team runs and the other sets
    python -m hermeneutics.external_probes report      RESULTS.md
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import numpy as np  # noqa: E402

from hermeneutics import team_probes as tp  # noqa: E402

REPO = tp.REPO
HUB = tp.HUB
THINK = tp.THINK
ACTS = HUB / "activations"
OUT = REPO / "hermeneutics" / "results" / "external_probes"
CACHE = REPO / "data" / "external_probes"
SRC = REPO / "data" / "goodfire_source_labels"
GF, CB, EG = "goodfire_reward_hack", "cheatbench", "evilgenie"
GF_REPO, GF_REV = "Goodfire/reward-hack-data", "bd8fb473568b8ef1d12fc2325a5c8af939c15ad9"
STATUS = Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad/external_probes/status.md")
LAYERS = tp.LAYERS
CFGS = ["all", 26, 38, 51]
N_NULL = 50
N_BOOT = 1000
JUDGE_ENVS = ["swebench-verified", "deepswe", "impossiblebench"]
MONITOR_ENVS = ["tau3bench", "shoppingbench"]
AUTHORS = {"glm": "GLM 5.2", "kimi": "Kimi K3", "qwenmax": "Qwen 3.8 Max"}
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:.0f}s]", *a, flush=True)


def status(msg):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a") as f:
        f.write(f"- {time.strftime('%H:%M', time.gmtime())} UTC: {msg}\n")


def lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x or [])


def cfg_name(c):
    return "think pair, layers 26+38+51" if c == "all" else f"think pair, layer {c}"


# ------------------------------------------------------------------------------------------------ labels
def gf_source_rollouts(rd):
    """{trace_id: row} from the source's judge_rollouts.jsonl or monitor_rollouts.jsonl, and the label kind."""
    j = SRC / rd / "labels" / "judge_rollouts.jsonl"
    m = SRC / rd / "generic_monitor" / "monitor_rollouts.jsonl"
    p, kind = (j, "judge") if j.is_file() else (m, "monitor")
    return {r["trace_id"]: r for r in map(json.loads, open(p)) if r}, kind


def gf_source_passages(rd):
    j = SRC / rd / "labels" / "judge_passages.jsonl"
    m = SRC / rd / "generic_monitor" / "monitor_passages.jsonl"
    if j.is_file():
        for r in map(json.loads, open(j)):
            yield r["trace_id"], r["node_index"], r["step"], r["channel"], {"positive": 1, "ambiguous": 2}.get(r["consensus"], 0)
    else:
        for r in map(json.loads, open(m)):
            yield r["trace_id"], r["node_index"], r["step"], r["channel"], {"certain": 1, "ambiguous": 2}.get(r["evidence"], 0)


def labels_goodfire():
    """One row per extracted rollout: author, environment, task, n_turns, the verdict three ways (our meta.json, the
    Hub labels.csv, the source file at GF_REV), and per turn the passage status (0 clean, 1 flagged, 2 ambiguous only)."""
    import pandas as pd
    rows, turns = [], {}
    for rdp in sorted((ACTS / GF).iterdir()):
        if not rdp.is_dir():
            continue
        rd = rdp.name
        src, kind = gf_source_rollouts(rd)
        hub = pd.read_csv(rdp / "labels.csv", dtype={"rollout_id": str}).set_index("rollout_id")
        metas = {}
        for d in sorted(p for p in rdp.iterdir() if p.is_dir()):
            m = json.load(open(d / "meta.json"))
            metas[d.name] = m
            assert m["rollout_id"] == d.name and m["run_dir"] == rd
            z = np.load(tp.think_path(GF, rd, d.name))
            n = int(z["think_L26"].shape[0])
            s = src.get(d.name, {})
            h = hub.loc[d.name] if d.name in hub.index else None
            tn, tpn = lit(m.get("turn_nodes")), lit(m.get("turn_path_nodes"))
            rows.append({"set": GF, "run": rd, "agent": d.name, "author": m["author"], "env": m["benchmark"],
                         "task": m.get("task"), "label_kind": kind, "n_turns": n, "meta_n_turns": m["n_turns"],
                         "len_turn_nodes": len(tn), "verdict": m.get("verdict"), "verdict_enacted": m.get("verdict_enacted"),
                         "source_verdict": s.get("verdict"), "source_judged": s.get("judged"),
                         "source_row_index": s.get("row_index"), "meta_row_index": m.get("row_index"),
                         "hub_verdict": None if h is None else h["verdict"],
                         "n_readable": m.get("n_readable"), "n_tokens": m.get("n_tokens")})
            turns[(rd, d.name)] = {"tn": {x: i for i, x in enumerate(tn)}, "tpn": {x: i for i, x in enumerate(tpn)},
                                   "st": np.zeros(n, np.int8), "unmapped": 0, "unmapped_flag": 0, "step_off": Counter(),
                                   "n_pass": 0}
        for tid, node, step, ch, lab in gf_source_passages(rd):
            t = turns.get((rd, tid))
            if t is None:
                continue
            t["n_pass"] += 1
            i = t["tn"].get(node, t["tpn"].get(node))
            if i is None or i >= len(t["st"]):
                t["unmapped"] += 1
                t["unmapped_flag"] += int(lab == 1)
                continue
            t["step_off"][step - (i + 1)] += 1
            if lab == 1:
                t["st"][i] = 1
            elif lab == 2 and t["st"][i] == 0:
                t["st"][i] = 2
        log(f"labels {rd}: {len(metas)} rollouts, {kind}")
    df = pd.DataFrame(rows)
    st_rows = []
    for i, r in df.iterrows():
        t = turns[(r.run, r.agent)]
        st = t["st"]
        pos = np.flatnonzero(st == 1)
        st_rows.append({"n_passages_joined": t["n_pass"], "unmapped_passages": t["unmapped"],
                        "unmapped_flagged": t["unmapped_flag"], "step_offsets": json.dumps(dict(t["step_off"])),
                        "flagged_turns": int((st == 1).sum()), "ambiguous_turns": int((st == 2).sum()),
                        "first_flagged_turn": int(pos[0]) if len(pos) else None,
                        "turn_status": "".join(map(str, st.tolist()))})
    df = pd.concat([df, pd.DataFrame(st_rows)], axis=1)
    df["y"] = df.verdict.map({"yes": 1, "no": 0})
    return df


def labels_cheatbench():
    """meta.json label and cheat; the reader turn of the localised cheating snippet where it is found in rendered.txt
    (the turn whose think-open follows the snippet's first character, i.e. the first turn that has read it)."""
    import pandas as pd
    rows = []
    for d in sorted(ACTS.joinpath(CB).glob("*/*/meta.json")):
        m = json.load(open(d))
        run, agent = d.parent.parent.name, d.parent.name
        assert m["run"] == run and m["agent"] == agent
        z = np.load(tp.think_path(CB, run, agent))
        n = int(z["think_L26"].shape[0])
        onset, found = None, None
        if m.get("cheat") == 1 and m.get("loc_snippet"):
            txt = open(d.parent / "rendered.txt", encoding="utf-8").read()
            k = txt.find(m["loc_snippet"][:80])
            found = k >= 0
            if found:
                heads = [i for i in _find_all(txt, "<|im_start|>assistant\n<think>")]
                after = [j for j, h in enumerate(heads) if h > k]
                onset = after[0] if after else None
        rows.append({"set": CB, "run": run, "agent": agent, "author": m.get("author"), "env": m.get("benchmark"),
                     "harness": m.get("harness"), "label": m.get("label"), "cheat": m.get("cheat"),
                     "cheating_type": m.get("cheating_type"), "n_turns": n, "meta_n_turns": m.get("n_turns"),
                     "n_readable": m.get("n_readable"), "snippet_found": found, "onset_turn": onset,
                     "label_source": m.get("label_source")})
    df = pd.DataFrame(rows)
    df["y"] = df.cheat.astype(float)
    return df


def _find_all(s, sub):
    i = s.find(sub)
    while i >= 0:
        yield i
        i = s.find(sub, i + 1)


def labels_evilgenie():
    import pandas as pd
    rows = []
    for d in sorted(ACTS.joinpath(EG).glob("*/*/meta.json")):
        m = json.load(open(d))
        run, agent = d.parent.parent.name, d.parent.name
        z = np.load(tp.think_path(EG, run, agent))
        n = int(z["think_L26"].shape[0])
        rows.append({"set": EG, "run": run, "agent": agent, "author": m.get("author"), "env": m.get("sel_condition"),
                     "problem": m.get("sel_problem_id"), "label": m.get("sel_label"), "judge_class": m.get("sel_judge_class"),
                     "reward_hacked": m.get("sel_reward_hacked"), "tamper_hack": m.get("sel_tamper_hack"),
                     "cheat": m.get("cheat"), "role": m.get("sel_role"), "run_key": m.get("sel_run_key"),
                     "pair_with": m.get("sel_pair_with"), "match": m.get("sel_match"), "n_turns": n,
                     "meta_n_turns": m.get("n_turns"), "n_readable": m.get("n_readable"), "reasoning": m.get("sel_reasoning")})
    df = pd.DataFrame(rows)
    # a negative's sel_pair_with names its positive's run key; a positive's is empty: the pair id is the positive's key
    df["pair"] = [b if isinstance(b, str) and b else a for a, b in zip(df.run_key, df.pair_with)]
    df["y"] = df.cheat.astype(float)
    return df


def stage_labels():
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    g = labels_goodfire()
    g.to_csv(OUT / "labels" / "goodfire_rollouts.csv.gz", index=False)
    c = labels_cheatbench()
    c.to_csv(OUT / "labels" / "cheatbench.csv", index=False)
    e = labels_evilgenie()
    e.to_csv(OUT / "labels" / "evilgenie.csv", index=False)
    chk = {
        "goodfire_rollouts": len(g),
        "goodfire_npz_turns_equal_meta": int((g.n_turns == g.meta_n_turns).sum()),
        "goodfire_turn_nodes_equal_turns": int((g.len_turn_nodes == g.n_turns).sum()),
        "goodfire_meta_verdict_equal_source": int((g.verdict == g.source_verdict).sum()),
        "goodfire_hub_verdict_equal_source": int((g.hub_verdict == g.source_verdict).sum()),
        "goodfire_row_index_equal_source": int((g.meta_row_index == g.source_row_index).sum()),
        "goodfire_rollouts_with_unmapped_passages": int((g.unmapped_passages > 0).sum()),
        "goodfire_unmapped_flagged_passages": int(g.unmapped_flagged.sum()),
        "goodfire_verdict_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in g.groupby(["label_kind", "verdict"]).size().items()},
        "goodfire_yes_without_flagged_turn": int(((g.verdict == "yes") & (g.flagged_turns == 0)).sum()),
        "goodfire_no_with_flagged_turn": int(((g.verdict == "no") & (g.flagged_turns > 0)).sum()),
        "cheatbench_rollouts": len(c), "cheatbench_zero_turns": int((c.n_turns == 0).sum()),
        "cheatbench_label_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in c.groupby(["label", "cheat"]).size().items()},
        "cheatbench_onset_found": int(c.onset_turn.notna().sum()),
        "evilgenie_rollouts": len(e), "evilgenie_pairs": int(e.pair.nunique()),
        "evilgenie_pairs_one_pos_one_neg": int(((e.groupby("pair").y.sum() == 1) & (e.groupby("pair").y.size() == 2)).sum()),
        "evilgenie_pairs_same_author": int((e.groupby("pair").author.nunique() == 1).sum()),
        "evilgenie_label_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in e.groupby(["label", "cheat"]).size().items()},
    }
    json.dump(chk, open(OUT / "labels" / "checks.json", "w"), indent=1)
    log(json.dumps(chk, indent=1))
    return chk


def load_labels(s):
    import pandas as pd
    f = {GF: "goodfire_rollouts.csv.gz", CB: "cheatbench.csv", EG: "evilgenie.csv"}[s]
    return pd.read_csv(OUT / "labels" / f, dtype={"agent": str, "turn_status": str})


# ------------------------------------------------------------------------------------------------ cache
def rendered_turns(path):
    """Per reader turn of a rendered.txt: (reasoning inside <think>...</think>, the text after </think> up to
    <|im_end|>: the visible reply and the tool calls). A reader turn opens at '<|im_start|>assistant\\n<think>'."""
    txt = open(path, encoding="utf-8").read()
    out = []
    for i in _find_all(txt, "<|im_start|>assistant\n<think>"):
        body = txt[i + len("<|im_start|>assistant\n<think>"):]
        e = body.find("<|im_end|>")
        body = body if e < 0 else body[:e]
        j = body.find("</think>")
        out.append((body[:j].strip(), body[j + len("</think>"):].strip()) if j >= 0 else (body.strip(), ""))
    return out


def means_path(s):
    return CACHE / f"means_{s}.npy"


def words_path(s):
    return CACHE / f"words_{s}.jsonl.gz"


def build_cache(s, df):
    """means_<s>.npy: float32 [rollouts, 3 layers, 2 (open, close), 5120], the mean over turns (NaN for zero turns);
    words_<s>.jsonl.gz: per rollout the rendered turns (reasoning, after-think text). One rollout loaded at a time."""
    CACHE.mkdir(parents=True, exist_ok=True)
    M = np.lib.format.open_memmap(str(means_path(s)) + ".tmp", mode="w+", dtype=np.float32, shape=(len(df), 3, 2, 5120))
    bad = 0
    with gzip.open(str(words_path(s)) + ".tmp", "wt", encoding="utf-8") as fw:
        for i, r in enumerate(df.itertuples()):
            z = np.load(tp.think_path(s, r.run, r.agent))
            n = z["think_L26"].shape[0]
            for j, L in enumerate(LAYERS):
                M[i, j] = z[f"think_L{L}"].astype(np.float32).mean(0) if n else np.nan
            rt = rendered_turns(ACTS / s / r.run / r.agent / "rendered.txt")
            bad += len(rt) != n
            fw.write(json.dumps({"run": r.run, "agent": r.agent, "n_turns": int(n), "n_rendered": len(rt),
                                 "think": [a for a, _ in rt], "after": [b for _, b in rt]}) + "\n")
            if i % 500 == 0:
                log(f"cache {s}: {i}/{len(df)}")
    M.flush()
    del M
    os.replace(str(means_path(s)) + ".tmp", means_path(s))
    os.replace(str(words_path(s)) + ".tmp", words_path(s))
    log(f"cache {s}: done, {bad} rollouts whose rendered turn count differs from the think pairs")
    return bad


def stage_cache(sets=(CB, EG, GF)):
    res = {}
    for s in sets:
        res[s] = build_cache(s, load_labels(s))
    json.dump(res, open(CACHE / "cache_checks.json", "w"))
    return res


def load_means(s):
    return np.load(means_path(s), mmap_mode="r")


def feat(M, rows, cfg):
    """float32 [len(rows), d]: the think pair at one layer or the three concatenated (layer, token, dim order, the same
    as team_probes.agent_matrix(..., 'all', 'think'))."""
    js = list(range(3)) if cfg == "all" else [LAYERS.index(int(cfg))]
    return np.ascontiguousarray(np.asarray(M[rows][:, js]).reshape(len(rows), -1), dtype=np.float32)


def load_words(s):
    return [json.loads(x) for x in gzip.open(words_path(s), "rt", encoding="utf-8")]


def words_doc(w):
    return " ".join(a + " " + b for a, b in zip(w["think"], w["after"]))


# ------------------------------------------------------------------------------------------------ probe machinery
def pca_fit(Xtr, kmax=128):
    """Training PCA to min(kmax, half the training rows) and z-scoring; returns a transform."""
    k = min(kmax, max(2, len(Xtr) // 2))
    c = Xtr.mean(0)
    Xc = Xtr - c
    if len(Xtr) <= 1500:
        w, U = np.linalg.eigh((Xc @ Xc.T).astype(np.float64))
        w, U = w[::-1][:k], U[:, ::-1][:, :k]
        V = ((Xc.T @ U) / np.sqrt(np.maximum(w, 1e-12))).astype(np.float32)
    else:
        from sklearn.utils.extmath import randomized_svd
        V = randomized_svd(Xc, k, n_iter=4, random_state=0)[2].T.astype(np.float32)
    P = Xc @ V
    sd = P.std(0) + 1e-9
    return lambda X: (((X - c) @ V) / sd).astype(np.float64)


def split_probe(X, y, splits, groups, perms=None):
    """Each split (name, train idx, test idx): PCA-z fitted on the training rows, L2 logistic regression with C by inner
    grouped CV (team_probes.choose_C), scores on the test rows. perms: label vectors for the null, refitted with C fixed
    at the median chosen C on the same PCA. Returns (scores, {split: C}, null AUROCs [n_perm], prepared splits)."""
    s = np.full(len(y), np.nan)
    prep, Cs = [], {}
    for name, tr, te in splits:
        tr = tr[np.isfinite(y[tr])]
        if len(set(y[tr].tolist())) < 2 or not len(te):
            continue
        f = pca_fit(X[tr])
        Ztr, Zte = f(X[tr]), f(X[te])
        sc, C = tp.fit_score(Ztr, y[tr].astype(int), groups[tr], Zte)
        s[te] = sc
        Cs[name] = C
        prep.append((name, tr, te, Ztr, Zte))
    nul = None
    if perms is not None and prep:
        from sklearn.linear_model import LogisticRegression
        C = float(np.median(list(Cs.values())))
        nul = []
        for yp in perms:
            sp = np.full(len(y), np.nan)
            for name, tr, te, Ztr, Zte in prep:
                if len(set(yp[tr].tolist())) < 2:
                    continue
                sp[te] = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ztr, yp[tr].astype(int)).decision_function(Zte)
            nul.append(sp)
    return s, Cs, nul, prep


def perms_within(y, strata, B=N_NULL, seed=1000):
    """Label vectors permuted within each stratum (run dir); NaN labels stay in place."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        yp = y.copy()
        for a in np.unique(strata):
            ii = np.flatnonzero((strata == a) & np.isfinite(y))
            yp[ii] = y[ii][rng.permutation(len(ii))]
        out.append(yp)
    return out


def split_dense(F, y, splits, C=1.0):
    from sklearn.linear_model import LogisticRegression
    s = np.full(len(y), np.nan)
    for name, tr, te in splits:
        tr = tr[np.isfinite(y[tr])]
        if len(set(y[tr].tolist())) < 2 or not len(te):
            continue
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        s[te] = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit((F[tr] - mu) / sd, y[tr].astype(int)).decision_function((F[te] - mu) / sd)
    return s


_HASHED = {}


def hashed_counts(docs):
    """Word unigram and bigram counts of each document (sklearn's default tokenizer, lowercased), hashed to 2^20
    columns: label-free and per document, so it is computed once per document list."""
    from sklearn.feature_extraction.text import HashingVectorizer
    k = id(docs)
    if k not in _HASHED:
        _HASHED.clear()
        _HASHED[k] = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 20, alternate_sign=False, norm=None,
                                       dtype=np.float32).transform(docs).tocsc()
    return _HASHED[k]


def split_tfidf(docs, y, splits, groups, max_features=50000):
    """TF-IDF of word unigrams and bigrams: per split the columns are those in at least 2 training documents, the
    max_features most frequent in the training documents, with sublinear tf and IDF from the training documents (as
    TfidfVectorizer(sublinear_tf, min_df=2, max_features, ngram_range=(1, 2)) but on hashed counts, which keeps the
    Goodfire vocabulary out of memory); then L2 logistic regression with C by inner grouped CV."""
    from sklearn.feature_extraction.text import TfidfTransformer
    H = hashed_counts(docs)
    s = np.full(len(y), np.nan)
    for name, tr, te in splits:
        tr = tr[np.isfinite(y[tr])]
        if len(set(y[tr].tolist())) < 2 or not len(te):
            continue
        Htr = H[tr]
        df_ = np.asarray((Htr > 0).sum(0)).ravel()
        tot = np.asarray(Htr.sum(0)).ravel()
        cand = np.flatnonzero(df_ >= 2)
        cols = cand[np.argsort(-tot[cand], kind="stable")[:max_features]]
        Xtr, Xte = Htr[:, cols].tocsr(), H[te][:, cols].tocsr()
        tf = TfidfTransformer(sublinear_tf=True).fit(Xtr)
        sc, _ = tp.fit_score(tf.transform(Xtr), y[tr].astype(int), groups[tr], tf.transform(Xte), Cs=tp.CS_TEXT)
        s[te] = sc
    return s


def kfold_splits(y, strata, groups, k=5, seed=0):
    from sklearn.model_selection import StratifiedGroupKFold
    idx = np.flatnonzero(np.isfinite(y))
    st = np.array([f"{a}|{b}" for a, b in zip(strata[idx], y[idx])])
    out, fold = [], np.full(len(y), -1)
    for i, (tr, te) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(idx, st, groups[idx])):
        out.append((f"fold {i}", idx[tr], idx[te]))
        fold[idx[te]] = i
    return out, fold


def group_splits(y, by, train_mask=None, test_mask=None):
    """One split per value of `by`: test = rows with that value (and test_mask), train = the other rows (and train_mask)."""
    ok = np.isfinite(y)
    trm = ok if train_mask is None else ok & train_mask
    tem = ok if test_mask is None else ok & test_mask
    return [(str(v), np.flatnonzero(trm & (by != v)), np.flatnonzero(tem & (by == v))) for v in sorted(set(by[tem].tolist()))]


def stats(score, y, clusters, extra=None, B=N_BOOT):
    ok = np.isfinite(score) & np.isfinite(y)
    s, yy, cl = score[ok], y[ok].astype(int), np.asarray(clusters)[ok]
    a = tp.auc(s, yy)
    lo, hi = tp.boot_ci(s, yy, cl, B=B) if len(set(yy.tolist())) == 2 else (np.nan, np.nan)
    r = {"n": int(ok.sum()), "pos": int(yy.sum()), "neg": int((1 - yy).sum()), "auroc": a, "ci_lo": lo, "ci_hi": hi}
    r.update(extra or {})
    return r


def null_extra(nul, real):
    if nul is None or not len(nul):
        return {}
    return {"null_mean": float(np.nanmean(nul)), "null_95": float(np.nanpercentile(nul, 95)),
            "null_p": float((1 + (nul >= real).sum()) / (1 + np.isfinite(nul).sum()))}


def within_auroc(score, y, strata):
    """AUROC over positive-negative pairs that share a stratum (pairs weighted equally) and the per-stratum AUROCs."""
    num = den = 0.0
    per = {}
    for v in sorted(set(strata.tolist())):
        m = (strata == v) & np.isfinite(score) & np.isfinite(y)
        p, n = int(y[m].sum()), int((1 - y[m]).sum())
        if p and n:
            a = tp.auc(score[m], y[m])
            per[v] = (a, p, n)
            num += a * p * n
            den += p * n
    return (num / den if den else np.nan), per


def onehot_cols(df, cols):
    return np.column_stack([tp.onehot(df[c].astype(str).tolist()) for c in cols])


# ------------------------------------------------------------------------------------------------ trajectory level
def within_boot(score, y, strata, clusters, B=N_BOOT, seed=0):
    """95% interval of within_auroc over clusters resampled with replacement (cluster draw counts as weights)."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(score) & np.isfinite(y)
    score, y, strata, cl = score[ok], y[ok], np.asarray(strata)[ok], np.asarray(clusters)[ok]
    uc, inv = np.unique(cl, return_inverse=True)
    vals = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))[inv].astype(float)
        num = den = 0.0
        for v in np.unique(strata):
            m = strata == v
            p, n = (w[m] * y[m]).sum(), (w[m] * (1 - y[m])).sum()
            if p and n:
                a = tp.auc(score[m], y[m], w[m])
                num += a * p * n
                den += p * n
        if den:
            vals.append(num / den)
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [np.nan, np.nan]


def add_within(r, s, y, within, clusters, nul=None):
    """Within-stratum AUROC (positive-negative pairs that share a stratum), its interval and its null, per stratum kind."""
    for wname, strata in (within or {}).items():
        a, per = within_auroc(s, y, strata)
        r[f"w_{wname}"] = a
        r[f"w_{wname}_ci"] = within_boot(s, y, strata, clusters, B=300)
        r[f"w_{wname}_per"] = json.dumps({str(k): [round(v[0], 4), v[1], v[2]] for k, v in per.items()})
        if nul is not None:
            na = np.array([within_auroc(sp, yp, strata)[0] for sp, yp in nul], float)
            r.update({f"w_{wname}_{k}": v for k, v in null_extra(na, a).items()})


def traj_block(tag, X_of, y, splits, groups, strata, clusters, docs, meta_F, within=None, null=True, cfgs=CFGS):
    """Probe rows for one evaluation (a set of splits) over the configurations, and the baselines on the same splits.
    auroc pools every test row; w_<name> keeps only positive-negative pairs inside one stratum (one run dir, harness,
    author or held-out group), the number to read when strata differ in base rate or when each held-out group has its
    own model. Null: labels permuted within `strata`, refitted, C fixed. Returns (rows, scores dict)."""
    rows, scores = [], {}
    perms = perms_within(y, strata) if null else None
    for cfg in cfgs:
        X = X_of(cfg)
        s, Cs, nul, _ = split_probe(X, y, splits, groups, perms)
        del X
        r = stats(s, y, clusters, {"eval": tag, "features": cfg_name(cfg), "C_median": float(np.median(list(Cs.values()))) if Cs else np.nan})
        pairs = None
        if nul is not None:
            pairs = list(zip(nul, perms))
            r.update(null_extra(np.array([tp.auc(sp[np.isfinite(sp) & np.isfinite(yp)], yp[np.isfinite(sp) & np.isfinite(yp)])
                                          for sp, yp in pairs], float), r["auroc"]))
        add_within(r, s, y, within, clusters, pairs)
        rows.append(r)
        scores[cfg_name(cfg)] = s
        log(f"  {tag} {cfg_name(cfg)}: {r['auroc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            + (f" null {r['null_mean']:.3f}/{r['null_95']:.3f}" if "null_mean" in r else "")
            + "".join(f" | within {w} {r[f'w_{w}']:.3f}" + (f" null {r[f'w_{w}_null_mean']:.3f}/{r[f'w_{w}_null_95']:.3f}" if f"w_{w}_null_mean" in r else "")
                      for w in (within or {})))
    for bname, sc in (("metadata", split_dense(meta_F, y, splits)) if meta_F is not None else (None, None),
                      ("TF-IDF agent's words (rendered text)", split_tfidf(docs, y, splits, groups)) if docs is not None else (None, None)):
        if bname is None:
            continue
        r = stats(sc, y, clusters, {"eval": tag, "features": bname})
        add_within(r, sc, y, within, clusters)
        rows.append(r)
        scores[bname] = sc
        log(f"  {tag} {bname}: {r['auroc']:.3f}" + "".join(f" | within {w} {r[f'w_{w}']:.3f}" for w in (within or {})))
    return rows, scores


def per_group_rows(tag, scores, y, by, clusters, names=None):
    """AUROC of the held-out scores inside each value of `by` (each held-out author, environment or run dir)."""
    out = []
    for feat_name, s in scores.items():
        for v in sorted(set(by[np.isfinite(s)].tolist())):
            m = (by == v) & np.isfinite(s) & np.isfinite(y)
            if not m.any():
                continue
            r = stats(np.where(m, s, np.nan), np.where(m, y, np.nan), clusters, {"eval": tag, "features": feat_name, "group": (names or {}).get(v, v)})
            out.append(r)
    return out


def traj_goodfire(res, sc_frames):
    import pandas as pd
    df = load_labels(GF)
    M = load_means(GF)
    W = load_words(GF)
    assert all(w["run"] == r.run and w["agent"] == r.agent for w, r in zip(W, df.itertuples()))
    docs = [words_doc(w) for w in W]
    del W
    judge = (df.label_kind == "judge").values
    y_all = df.y.values.astype(float)
    runs, agents = df.run.values, df.agent.values
    rid = np.array([f"{a}/{b}" for a, b in zip(runs, agents)])
    task = np.array([f"{e}|{t}" for e, t in zip(df.env, df.task)])
    env, author = df.env.values, df.author.values
    y = np.where(judge, y_all, np.nan)  # judge-labelled rollouts, verdict yes / no (split and unjudged dropped)
    meta_F = onehot_cols(df, ["author", "env"])

    def X_of(cfg):
        return feat(M, np.arange(len(df)), cfg)
    # 1a: five folds, grouped by rollout, stratified by run dir x verdict
    splits, fold = kfold_splits(y, runs, rid)
    rows, sc = traj_block("Goodfire judge sets, 5 folds (grouped by rollout)", X_of, y, splits, rid, runs, rid, docs, meta_F,
                          within={"run dir": runs})
    res["gf_5fold"] = rows
    res["gf_5fold_by_run"] = per_group_rows("Goodfire judge sets, 5 folds (grouped by rollout)", sc, y, runs, rid)
    w = {k: within_auroc(v, y, runs) for k, v in sc.items()}
    res["gf_5fold_within_run"] = [{"features": k, "within_run_dir_auroc": v[0]} for k, v in w.items()]
    sc_frames.append(pd.DataFrame({"eval": "gf_5fold", "run": runs, "agent": agents, "y": y, "fold": fold, **sc}))
    # 1a': five folds grouped by task (the same SWE-bench or DeepSWE instance never in train and test)
    splits_t, fold_t = kfold_splits(y, runs, task)
    rows, sc = traj_block("Goodfire judge sets, 5 folds (grouped by task)", X_of, y, splits_t, task, runs, rid, docs, meta_F,
                          within={"run dir": runs}, null=False, cfgs=["all"])
    res["gf_5fold_task"] = rows
    sc_frames.append(pd.DataFrame({"eval": "gf_5fold_task", "run": runs, "agent": agents, "y": y, "fold": fold_t, **sc}))
    # 1a'': all 13 run dirs, judge and monitor verdicts pooled (the label the earlier transfer used)
    splits_a, fold_a = kfold_splits(y_all, runs, rid)
    rows, sc = traj_block("Goodfire all 13 run dirs (judge + monitor verdicts), 5 folds", X_of, y_all, splits_a, rid, runs, rid, None,
                          meta_F, within={"run dir": runs}, null=False, cfgs=["all"])
    res["gf_5fold_all13"] = rows
    res["gf_5fold_all13_by_kind"] = per_group_rows("Goodfire all 13 run dirs, 5 folds", sc, y_all, df.label_kind.values, rid)
    # 1b: held-out author (judge sets: train on two authors, test on the third)
    splits_h = group_splits(y, author)
    rows, sc = traj_block("Goodfire judge sets, held-out author", X_of, y, splits_h, rid, runs, rid, docs, onehot_cols(df, ["env"]),
                          within={"held-out author": author, "run dir": runs})
    res["gf_author"] = rows
    res["gf_author_by"] = per_group_rows("Goodfire judge sets, held-out author", sc, y, author, rid, AUTHORS)
    res["gf_author_by_run"] = per_group_rows("Goodfire judge sets, held-out author", sc, y, runs, rid)
    sc_frames.append(pd.DataFrame({"eval": "gf_author", "run": runs, "agent": agents, "y": y, "fold": author, **sc}))
    # 1c: held-out environment: train on the judge rollouts of the other environments; test on one (the two monitor
    # environments are test only, scored against the monitor verdict)
    splits_e = [(e, np.flatnonzero(np.isfinite(y) & (env != e)), np.flatnonzero(np.isfinite(y_all) & (env == e)))
                for e in JUDGE_ENVS + MONITOR_ENVS]
    rows, sc = traj_block("Goodfire held-out environment", X_of, y_all, splits_e, rid, runs, rid, docs, onehot_cols(df, ["author"]),
                          within={"held-out environment": env, "run dir": runs})
    res["gf_env"] = rows
    res["gf_env_by"] = per_group_rows("Goodfire held-out environment", sc, y_all, env, rid)
    sc_frames.append(pd.DataFrame({"eval": "gf_env", "run": runs, "agent": agents, "y": y_all, "fold": env, **sc}))
    json.dump({"rid_folds": {r: int(f) for r, f in zip(rid, fold) if f >= 0}}, gzip.open(OUT / "folds_goodfire.json.gz", "wt"))


def traj_goodfire_enacted(res, sc_frames):
    """Sensitivity: the source's verdict_enacted (attempted or enacted hacks only; a hack the agent only considered is
    not positive) instead of verdict, judge run dirs, five folds, all configurations and baselines."""
    import pandas as pd
    df = load_labels(GF)
    M = load_means(GF)
    docs = [words_doc(w) for w in load_words(GF)]
    judge = (df.label_kind == "judge").values
    y = np.where(judge, df.verdict_enacted.map({"yes": 1, "no": 0}).values.astype(float), np.nan)
    runs, agents = df.run.values, df.agent.values
    rid = np.array([f"{a}/{b}" for a, b in zip(runs, agents)])
    splits, fold = kfold_splits(y, runs, rid)
    rows, sc = traj_block("Goodfire judge sets, verdict_enacted yes vs no, 5 folds", lambda c: feat(M, np.arange(len(df)), c), y,
                          splits, rid, runs, rid, docs, onehot_cols(df, ["author", "env"]), within={"run dir": runs})
    res["gf_5fold_enacted"] = rows
    res["gf_5fold_enacted_counts"] = {"considered_only": int(((df.verdict == "yes") & (df.verdict_enacted == "no") & judge).sum()),
                                      "enacted_split": int(((df.verdict_enacted == "split") & judge).sum())}
    sc_frames.append(pd.DataFrame({"eval": "gf_5fold_enacted", "run": runs, "agent": agents, "y": y, "fold": fold, **sc}))


def traj_simple(s, res, sc_frames):
    """CheatBench and EvilGenie: five folds and held-out model (author); CheatBench also held-out harness and the
    within-harness AUROC. EvilGenie folds keep each matched pair together."""
    import pandas as pd
    df = load_labels(s)
    M = load_means(s)
    W = load_words(s)
    docs = [words_doc(w) for w in W]
    keep = (df.n_turns > 0).values
    y = np.where(keep, df.y.values.astype(float), np.nan)
    runs, agents = df.run.values, df.agent.values
    rid = np.array([f"{a}/{b}" for a, b in zip(runs, agents)])
    author = df.author.values
    idx = np.arange(len(df))

    def X_of(cfg):
        X = np.zeros((len(df), 30720 if cfg == "all" else 10240), np.float32)
        X[keep] = feat(M, idx[keep], cfg)
        return X
    if s == CB:
        grp, strata = rid, runs
        meta_F = onehot_cols(df, ["author", "harness", "env"])
        variants = [("cheating or attempt vs benign", y),
                    ("cheating vs benign (attempt dropped)", np.where(df.label.values == "attempt", np.nan, y))]
    else:
        grp, strata = df.pair.values, author
        # EvilGenie has eight problems, some almost all one label: the problem is metadata, and a stratum
        meta_F = onehot_cols(df, ["author", "env", "problem"])
        prob = df.problem.values.astype(str)
        ap_ = np.array([f"{a}|{b}" for a, b in zip(author, prob)])
        variants = [("hack vs legitimate", y)]
    for vname, yv in variants:
        tag = f"{s}: {vname}"
        splits, fold = kfold_splits(yv, strata, grp)
        rows, sc = traj_block(f"{tag}, 5 folds", X_of, yv, splits, grp, strata, grp, docs, meta_F,
                              within={"harness": runs} if s == CB else {"author": author, "problem": prob, "author and problem": ap_})
        res[f"{s}_{vname}_5fold"] = rows
        res[f"{s}_{vname}_5fold_by"] = per_group_rows(f"{tag}, 5 folds", sc, yv, runs if s == CB else author, grp)
        res[f"{s}_{vname}_5fold_within"] = [{"features": k, "within_auroc": within_auroc(v, yv, runs if s == CB else author)[0]}
                                            for k, v in sc.items()]
        sc_frames.append(pd.DataFrame({"eval": f"{s}|{vname}|5fold", "run": runs, "agent": agents, "y": yv, "fold": fold, **sc}))
        hold = [("author", author)] + ([("harness", df.harness.values)] if s == CB else [("problem", prob)])
        for hname, by in hold:
            splits_h = group_splits(yv, by)
            if s == EG:
                mF = onehot_cols(df, ["env", "problem"] if hname == "author" else ["author", "env"])
                wb = {f"held-out {hname}": by} | ({"problem": prob} if hname == "author" else {"author": author})
            else:
                mF = onehot_cols(df, ["env"]) if hname == "author" else None
                wb = {f"held-out {hname}": by} | ({"harness": runs} if hname == "author" else {})
            rows, sc = traj_block(f"{tag}, held-out {hname}", X_of, yv, splits_h, grp, strata, grp, docs, mF, within=wb)
            res[f"{s}_{vname}_held_{hname}"] = rows
            res[f"{s}_{vname}_held_{hname}_by"] = per_group_rows(f"{tag}, held-out {hname}", sc, yv, by, grp)
            sc_frames.append(pd.DataFrame({"eval": f"{s}|{vname}|held {hname}", "run": runs, "agent": agents, "y": yv, "fold": by, **sc}))


def stage_traj(which=("cb", "eg", "gf")):
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / "traj.json"
    res = json.load(open(rp)) if rp.is_file() else {}
    for w in which:
        frames = []
        r = {}
        if w == "gf":
            traj_goodfire(r, frames)
        elif w == "gfe":
            traj_goodfire_enacted(r, frames)
        else:
            traj_simple({"cb": CB, "eg": EG}[w], r, frames)
        res.update(r)
        json.dump(res, open(rp, "w"), indent=1, default=float)
        pd.concat(frames, ignore_index=True).to_csv(OUT / f"traj_scores_{w}.csv.gz", index=False)
        status(f"trajectory stage {w} done")


# ------------------------------------------------------------------------------------------------ per step (Goodfire)
STEP_N_SAMPLE = 3000  # PCA rows sampled per fold for the per-step bases


def step_frame():
    """Goodfire judge rollouts with verdict yes or no, every passage joined to a turn, and for a yes rollout at least one
    flagged turn (the onset is known). Rollout folds are the trajectory-level five folds."""
    df = load_labels(GF)
    f = json.load(gzip.open(OUT / "folds_goodfire.json.gz", "rt"))["rid_folds"]
    rid = [f"{a}/{b}" for a, b in zip(df.run, df.agent)]
    ok = (df.label_kind == "judge") & df.y.notna() & (df.unmapped_passages == 0) & ~((df.y == 1) & (df.flagged_turns == 0))
    S = df[ok.values].reset_index(drop=True)
    S["fold"] = [f[r] for r, k in zip(rid, ok.values) if k]
    drop = {"split_or_unjudged": int(((df.label_kind == "judge") & df.y.isna()).sum()),
            "unmapped_passages": int(((df.label_kind == "judge") & df.y.notna() & (df.unmapped_passages > 0)).sum()),
            "yes_without_flagged_turn": int(((df.label_kind == "judge") & (df.y == 1) & (df.flagged_turns == 0) & (df.unmapped_passages == 0)).sum())}
    return S, drop


def step_arrays(S):
    n_t = S.n_turns.values.astype(int)
    ai = np.concatenate([np.full(n, i) for i, n in enumerate(n_t)])
    tpos = np.concatenate([np.arange(n) for n in n_t])
    st = np.concatenate([np.array([int(c) for c in s], np.int8) if isinstance(s, str) else np.zeros(n, np.int8)
                         for s, n in zip(S.turn_status.values, n_t)])
    assert len(st) == len(ai)
    reach = S.y.values.astype(int)
    onset = np.array([float(x) if reach[i] and np.isfinite(x) else np.nan for i, x in enumerate(S.first_flagged_turn.values.astype(float))])
    # definition A: a turn is positive when a flagged passage falls in it (turns with only ambiguous passages left out)
    yA = (st == 1).astype(int)
    evalA = st != 2
    # definition B: positive from the first flagged passage's turn on (ambiguous-only turns before the onset left out)
    on_r = np.where(np.isfinite(onset), onset, 1e9)[ai]
    yB = (tpos >= on_r).astype(int)
    evalB = ~((st == 2) & (tpos < on_r))
    return n_t, ai, tpos, reach, onset, {"A": (yA, evalA), "B": (yB, evalB)}


def step_project(S, n_t, ai):
    """Per fold and configuration, a PCA basis (128) fitted on STEP_N_SAMPLE turns sampled from the fold's training
    rollouts; every turn projected (one rollout loaded at a time). Also the layer-38 think pair of every turn to a
    float16 memmap for the attention monitor. Returns {(cfg, fold): [n_rows, 128] float32}."""
    from sklearn.utils.extmath import randomized_svd
    folds = S.fold.values
    rng = np.random.default_rng(7)
    offs = np.concatenate([[0], np.cumsum(n_t)])
    samp = {}
    for f in range(5):
        cand = np.flatnonzero(folds[ai] == f)  # rows of fold f; sampled for the bases of the other folds
        samp[f] = np.sort(rng.choice(cand, min(STEP_N_SAMPLE, len(cand)), replace=False))
    allsamp = np.sort(np.concatenate(list(samp.values())))
    pos_of = {r: i for i, r in enumerate(allsamp)}
    Xs = np.zeros((len(allsamp), 30720), np.float16)
    l38 = np.lib.format.open_memmap(str(CACHE / "step_l38.npy"), mode="w+", dtype=np.float16, shape=(int(offs[-1]), 10240))
    need = defaultdict(list)
    for r in allsamp:
        need[int(ai[r])].append(int(r))
    for i, r in enumerate(S.itertuples()):
        z = np.load(tp.think_path(GF, r.run, r.agent))
        A = np.concatenate([z[f"think_L{L}"].reshape(z["think_L26"].shape[0], -1) for L in LAYERS], 1)
        assert A.shape[0] == n_t[i]
        l38[offs[i]:offs[i + 1]] = A[:, 10240:20480]
        for row in need.get(i, []):
            Xs[pos_of[row]] = A[row - offs[i]]
        if i % 500 == 0:
            log(f"  step sample pass {i}/{len(S)}")
    l38.flush()
    del l38
    bases = {}
    sfold = folds[ai[allsamp]]
    for cfg in CFGS:
        cols = slice(0, 30720) if cfg == "all" else slice(10240 * LAYERS.index(cfg), 10240 * (LAYERS.index(cfg) + 1))
        for f in range(5):
            Xtr = Xs[sfold != f][:, cols].astype(np.float32)
            c = Xtr.mean(0)
            Xtr -= c
            V = randomized_svd(Xtr, 128, n_iter=4, random_state=0)[2].T.astype(np.float32)
            bases[(cfg, f)] = (cols, c, V)
            del Xtr
        log(f"  step bases {cfg} done")
    del Xs
    P = {k: np.zeros((int(offs[-1]), 128), np.float32) for k in bases}
    for i, r in enumerate(S.itertuples()):
        z = np.load(tp.think_path(GF, r.run, r.agent))
        A = np.concatenate([z[f"think_L{L}"].reshape(z["think_L26"].shape[0], -1) for L in LAYERS], 1).astype(np.float32)
        for k, (cols, c, V) in bases.items():
            P[k][offs[i]:offs[i + 1]] = (A[:, cols] - c) @ V
        if i % 500 == 0:
            log(f"  step projection pass {i}/{len(S)}")
    return P


def step_linear(P, cfg, y, ev, folds_r, rid_r, C=None, keep_C=None):
    """Per fold: z-score the projections with the training rows' statistics, L2 logistic regression (C by inner grouped
    CV over rollouts unless given), scores on the test rows. Training uses the evaluated rows only."""
    from sklearn.linear_model import LogisticRegression
    s = np.full(len(y), np.nan)
    Cs = []
    for f in range(5):
        Z = P[(cfg, f)]
        tr = np.flatnonzero((folds_r != f) & ev)
        te = np.flatnonzero(folds_r == f)
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        Ztr, Zte = ((Z[tr] - mu) / sd).astype(np.float64), ((Z[te] - mu) / sd).astype(np.float64)
        Cf = C if C is not None else tp.choose_C(Ztr, y[tr], rid_r[tr])
        s[te] = LogisticRegression(C=Cf, max_iter=3000, class_weight="balanced").fit(Ztr, y[tr]).decision_function(Zte)
        Cs.append(Cf)
    return s, Cs


def step_metrics(s, y, ev, ai, tpos, reach, onset, rid_a, clusters_r, B=500):
    """Per-step AUROC on the evaluated turns; rollout AUROC of the last and the max score; detection delay at the
    threshold that flags 5% of the evaluated negative turns (first crossing at or after the onset, minus the onset)."""
    ok = ev & np.isfinite(s)
    r = {"turns": int(ok.sum()), "pos_turns": int(y[ok].sum()), "rollouts": int(len(np.unique(ai[ok]))),
         "pos_rollouts": int(reach.sum()), "step_auroc": tp.auc(s[ok], y[ok])}
    r["step_ci_lo"], r["step_ci_hi"] = tp.boot_ci(s[ok], y[ok], clusters_r[ok], B=B)
    last = np.full(len(reach), np.nan)
    mx = np.full(len(reach), np.nan)
    order = np.argsort(ai, kind="stable")
    bounds = np.searchsorted(ai[order], np.arange(len(reach) + 1))
    for i in range(len(reach)):
        v = s[order[bounds[i]:bounds[i + 1]]]
        v = v[np.isfinite(v)]
        if len(v):
            last[i], mx[i] = v[-1], v.max()
    r["rollout_auroc_last"] = tp.auc(last, reach)
    r["rollout_auroc_max"] = tp.auc(mx, reach)
    r["rollout_max_ci_lo"], r["rollout_max_ci_hi"] = tp.boot_ci(mx, reach, rid_a, B=B)
    neg = ok & (y == 0)
    thr = float(np.percentile(s[neg], 95))
    d, before, fired_neg = [], 0, 0
    for i in range(len(reach)):
        rows = order[bounds[i]:bounds[i + 1]]
        hit = rows[s[rows] > thr]
        if reach[i] and np.isfinite(onset[i]):
            after = hit[tpos[hit] >= onset[i]]
            if len(after):
                d.append(tpos[after[0]] - onset[i])
            before += int((tpos[hit] < onset[i]).any())
        elif not reach[i]:
            fired_neg += int(len(hit) > 0)
    d = np.array(d, float)
    r.update({"thr_5pct_fpr": thr, "detected": int(len(d)), "reaching": int(np.isfinite(onset).sum()),
              "delay_median": float(np.median(d)) if len(d) else np.nan,
              "delay_q25": float(np.percentile(d, 25)) if len(d) else np.nan,
              "delay_q75": float(np.percentile(d, 75)) if len(d) else np.nan,
              "fired_before_onset": int(before), "neg_rollouts_fired": int(fired_neg), "neg_rollouts": int((reach == 0).sum())})
    return r


def step_tfidf_cum(incr, ai, y, ev, folds_r, rid_r, max_features=10000, n_train=40000, seed=3):
    """Per-turn TF-IDF of the agent's words so far (this turn's reasoning and everything the agent wrote before it):
    word counts with the vocabulary (min_df 5, max_features) fitted on the training fold's turns, summed over the
    rollout's turns so far, sublinear tf and IDF from the training rows; L2 logistic regression with C by inner grouped
    CV. The cumulative matrix of every training turn does not fit this machine's memory, so the regression trains on
    n_train evaluated training turns drawn at random, and the test turns are scored one rollout at a time."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    s = np.full(len(y), np.nan)
    assert np.all(np.diff(ai) >= 0)
    bounds = np.searchsorted(ai, np.arange(ai.max() + 2))
    rng = np.random.default_rng(seed)

    def cum_rows(Cm, a, rows=None):
        blk = np.cumsum(Cm[bounds[a]:bounds[a + 1]].toarray(), 0)
        return sp.csr_matrix(blk if rows is None else blk[rows])
    for f in range(5):
        tr_all = np.flatnonzero((folds_r != f) & ev)
        tr = np.sort(rng.choice(tr_all, min(n_train, len(tr_all)), replace=False))
        te = np.flatnonzero(folds_r == f)
        cv = CountVectorizer(min_df=5, max_features=max_features, dtype=np.float32)
        cv.fit([incr[i] for i in np.flatnonzero(folds_r != f)])
        Cm = cv.transform(incr).tocsr()
        by_a = defaultdict(list)
        for i in tr:
            by_a[int(ai[i])].append(int(i - bounds[ai[i]]))
        Xtr = sp.vstack([cum_rows(Cm, a, np.array(r)) for a, r in sorted(by_a.items())], format="csr")
        order_tr = np.concatenate([np.array(r) + bounds[a] for a, r in sorted(by_a.items())])
        tf = TfidfTransformer(sublinear_tf=True).fit(Xtr)
        from sklearn.linear_model import LogisticRegression
        Ztr = tf.transform(Xtr)
        C = tp.choose_C(Ztr, y[order_tr], rid_r[order_tr], tp.CS_TEXT)
        lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ztr, y[order_tr])
        del Xtr, Ztr
        for a in np.unique(ai[te]):
            s[bounds[a]:bounds[a + 1]] = lr.decision_function(tf.transform(cum_rows(Cm, a)))
        del Cm
        log(f"   tfidf fold {f} done (C {C})")
    return s


# ---- attention monitor (attn_probe.py's simplified encoder: layer-38 think pair standardised per dimension, MLP to 128,
# learned positional encoding, two pre-norm causal self-attention layers, MLP head; here one binary head per step)
D_ATT, MAX_STEP = 128, 160


def build_monitor(n_in, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(D_ATT), nn.LayerNorm(D_ATT)
            self.att = nn.MultiheadAttention(D_ATT, 4, dropout=0.3, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(D_ATT, 4 * D_ATT), nn.GELU(), nn.Dropout(0.3), nn.Linear(4 * D_ATT, D_ATT), nn.Dropout(0.3))
            self.drop = nn.Dropout(0.3)

        def forward(self, x, causal, pad):
            h = self.n1(x)
            a, _ = self.att(h, h, h, attn_mask=causal, key_padding_mask=pad, need_weights=False)
            x = x + self.drop(a)
            return x + self.ff(self.n2(x))

    class Monitor(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(n_in, D_ATT), nn.GELU(), nn.Dropout(0.3), nn.Linear(D_ATT, D_ATT))
            self.pos = nn.Embedding(MAX_STEP, D_ATT)
            self.blocks = nn.ModuleList([Block(), Block()])
            self.norm = nn.LayerNorm(D_ATT)
            self.head = nn.Sequential(nn.Linear(D_ATT, D_ATT), nn.GELU(), nn.Dropout(0.3), nn.Linear(D_ATT, 1))

        def forward(self, z, steps, pad):
            h = self.enc(z) + self.pos(steps.clamp(max=MAX_STEP - 1))
            n = h.shape[1]
            causal = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
            for b in self.blocks:
                h = b(h, causal, pad)
            return self.head(self.norm(h))[..., 0]
    return Monitor()


def attention_cv(S, n_t, ai, y, ev, reach, seed=0, max_epochs=30, patience=5, bs=8, lr=1e-3):
    """Train on each fold's training rollouts (20% of them held out for early stopping), score the test rollouts.
    Loss: per-step binary cross-entropy on the evaluated steps plus the trajectory label at the last step."""
    import torch
    torch.set_num_threads(6)
    X = np.load(CACHE / "step_l38.npy", mmap_mode="r")
    offs = np.concatenate([[0], np.cumsum(n_t)])
    folds = S.fold.values
    s_out = np.full(len(y), np.nan)
    hist = {}
    for f in range(5):
        tr_all = np.flatnonzero(folds != f)
        te = np.flatnonzero(folds == f)
        rng = np.random.default_rng(100 + seed + f)
        va = np.sort(rng.choice(tr_all, int(round(0.2 * len(tr_all))), replace=False))
        tr = np.setdiff1d(tr_all, va)
        # per-dimension standardisation from the training rollouts' steps (streamed in chunks)
        s1 = np.zeros(10240)
        s2 = np.zeros(10240)
        cnt = 0
        for i in tr_all:
            x = np.asarray(X[offs[i]:offs[i + 1]], np.float64)
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
            cnt += len(x)
        mu = (s1 / cnt).astype(np.float32)
        sd = (np.sqrt(np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0)) + 1e-3).astype(np.float32)

        def batch(idx):
            n = max(int(n_t[i]) for i in idx)
            z = np.zeros((len(idx), n, 10240), np.float32)
            yy = np.zeros((len(idx), n), np.float32)
            w = np.zeros((len(idx), n), np.float32)
            pad = np.ones((len(idx), n), bool)
            fin = np.zeros(len(idx), np.float32)
            last = np.zeros(len(idx), np.int64)
            for b, i in enumerate(idx):
                m = int(n_t[i])
                z[b, :m] = (np.asarray(X[offs[i]:offs[i + 1]], np.float32) - mu) / sd
                yy[b, :m] = y[offs[i]:offs[i + 1]]
                w[b, :m] = ev[offs[i]:offs[i + 1]]
                pad[b, :m] = False
                fin[b], last[b] = reach[i], m - 1
            st = torch.arange(n)[None].repeat(len(idx), 1)
            return (torch.from_numpy(z), st, torch.from_numpy(pad), torch.from_numpy(yy), torch.from_numpy(w),
                    torch.from_numpy(fin), torch.from_numpy(last))

        def loss_of(model, bt):
            z, st, pad, yy, w, fin, last = bt
            o = model(z, st, pad)
            l1 = (torch.nn.functional.binary_cross_entropy_with_logits(o, yy, reduction="none") * w).sum()
            ar = torch.arange(len(fin))
            l2 = torch.nn.functional.binary_cross_entropy_with_logits(o[ar, last], fin, reduction="sum")
            return (l1 + l2) / (w.sum() + len(fin))
        model = build_monitor(10240, seed + f)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        # rollouts sorted by length into batches (less padding), batch order shuffled each epoch
        def batches(idx, shuffle):
            o = idx[np.argsort(n_t[idx], kind="stable")]
            bl = [o[k:k + bs] for k in range(0, len(o), bs)]
            if shuffle:
                rng.shuffle(bl)
            return bl
        best, best_state, bad, h = np.inf, None, 0, []
        for ep in range(max_epochs):
            model.train()
            tl = []
            for b in batches(tr, True):
                loss = loss_of(model, batch(b))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(loss_of(model, batch(b))) for b in batches(va, False)]))
            h.append((ep, float(np.mean(tl)), vl))
            log(f"   attention fold {f} epoch {ep}: train {np.mean(tl):.4f} val {vl:.4f}")
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for b in batches(te, False):
                bt = batch(b)
                o = model(bt[0], bt[1], bt[2]).numpy()
                for k, i in enumerate(b):
                    s_out[offs[i]:offs[i + 1]] = o[k, :n_t[i]]
        hist[f] = h
        status(f"attention fold {f} done ({len(h)} epochs, best val {best:.4f})")
    return s_out, hist


def stage_step(attention=True, null=True):
    import pandas as pd
    S, drop = step_frame()
    n_t, ai, tpos, reach, onset, defs = step_arrays(S)
    folds_r = S.fold.values[ai]
    rid_a = np.array([f"{a}/{b}" for a, b in zip(S.run, S.agent)])
    rid_r = rid_a[ai]
    runs_a = S.run.values
    log(f"step: {len(S)} rollouts, {len(ai)} turns, {reach.sum()} positive rollouts; dropped {drop}")
    pp = CACHE / "step_proj.npz"
    if pp.is_file():
        z = np.load(pp)
        P = {(c if c == "all" else int(c), int(f)): z[f"{c}|{f}"] for c in CFGS for f in range(5)}
    else:
        P = step_project(S, n_t, ai)
        np.savez(pp, **{f"{c}|{f}": v for (c, f), v in P.items()})
    rows, frames = [], {"run": S.run.values[ai], "agent": S.agent.values[ai], "turn": tpos, "fold": folds_r,
                        "status": np.concatenate([np.array([int(c) for c in s], np.int8) for s in S.turn_status.values]),
                        "yA": defs["A"][0], "yB": defs["B"][0], "evalA": defs["A"][1].astype(int), "evalB": defs["B"][1].astype(int)}
    meta_F = onehot_cols(S, ["author", "env"])[ai]
    frac = tpos / np.maximum(n_t[ai] - 1, 1)
    # the agent's words visible at each turn's think tokens: the previous turn's reply and tool calls, this turn's reasoning
    Wd = {(w["run"], w["agent"]): w for w in load_words(GF)}
    incr, txt_bad = [], 0
    for r in S.itertuples():
        w = Wd[(r.run, r.agent)]
        txt_bad += w["n_rendered"] != r.n_turns
        th, af = w["think"], w["after"]
        for t in range(r.n_turns):
            incr.append(((af[t - 1] if 0 < t <= len(af) else "") + " " + (th[t] if t < len(th) else "")))
    del Wd
    log(f"step: text built, {txt_bad} rollouts with a rendered turn count unlike the think pairs")
    for dname, (y, ev) in defs.items():
        for cfg in CFGS:
            s, Cs = step_linear(P, cfg, y, ev, folds_r, rid_r)
            r = step_metrics(s, y, ev, ai, tpos, reach, onset, rid_a, rid_r)
            r.update({"definition": dname, "features": "linear probe, " + cfg_name(cfg), "C": json.dumps(Cs)})
            if null and cfg == "all" and dname == "B":
                # null: the rollouts' (verdict, onset fraction) permuted within run dir, the probe refitted (C fixed)
                rng = np.random.default_rng(2000)
                ofrac = np.where(np.isfinite(onset), onset / np.maximum(n_t - 1, 1), np.nan)
                C = float(np.median(Cs))
                nul_t, nul_a = [], []
                for b in range(N_NULL):
                    pr, po = reach.copy(), ofrac.copy()
                    for rd in np.unique(runs_a):
                        ii = np.flatnonzero(runs_a == rd)
                        p_ = rng.permutation(len(ii))
                        pr[ii], po[ii] = reach[ii][p_], ofrac[ii][p_]
                    on = np.where(np.isfinite(po) & (pr == 1), np.round(po * np.maximum(n_t - 1, 1)), 1e9)
                    yp = (tpos >= on[ai]).astype(int)
                    sp_, _ = step_linear(P, cfg, yp, ev, folds_r, rid_r, C=C)
                    nul_t.append(tp.auc(sp_[ev], yp[ev]))
                    mx = pd.Series(sp_).groupby(ai).max().values
                    nul_a.append(tp.auc(mx, pr))
                    if b % 10 == 0:
                        log(f"   step null draw {b}: {nul_t[-1]:.3f} / {nul_a[-1]:.3f}")
                nul_t, nul_a = np.array(nul_t), np.array(nul_a)
                r.update({"step_null_mean": float(nul_t.mean()), "step_null_95": float(np.percentile(nul_t, 95)),
                          "step_null_p": float((1 + (nul_t >= r["step_auroc"]).sum()) / (1 + len(nul_t))),
                          "rollout_null_mean": float(nul_a.mean()), "rollout_null_95": float(np.percentile(nul_a, 95)),
                          "rollout_null_p": float((1 + (nul_a >= r["rollout_auroc_max"]).sum()) / (1 + len(nul_a)))})
            rows.append(r)
            frames[f"{dname}|linear {cfg}"] = s
            log(f"  step {dname} {cfg_name(cfg)}: step {r['step_auroc']:.3f} rollout(max) {r['rollout_auroc_max']:.3f} delay {r['delay_median']}")
        bases = {"turn index": tpos[:, None].astype(float), "turn index and fraction of the rollout": np.column_stack([tpos, frac]),
                 "metadata (author, environment) and turn index": np.column_stack([meta_F, tpos])}
        for bname, F in bases.items():
            s = np.full(len(y), np.nan)
            from sklearn.linear_model import LogisticRegression
            for f in range(5):
                tr, te = np.flatnonzero((folds_r != f) & ev), np.flatnonzero(folds_r == f)
                mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
                s[te] = LogisticRegression(max_iter=5000, class_weight="balanced").fit((F[tr] - mu) / sd, y[tr]).decision_function((F[te] - mu) / sd)
            r = step_metrics(s, y, ev, ai, tpos, reach, onset, rid_a, rid_r)
            r.update({"definition": dname, "features": bname})
            rows.append(r)
            frames[f"{dname}|{bname}"] = s
        s = step_tfidf_cum(incr, ai, y, ev, folds_r, rid_r)
        r = step_metrics(s, y, ev, ai, tpos, reach, onset, rid_a, rid_r)
        r.update({"definition": dname, "features": "TF-IDF agent's words so far"})
        rows.append(r)
        frames[f"{dname}|tfidf"] = s
        json.dump({"rows": rows, "drop": drop}, open(OUT / "step.json", "w"), indent=1, default=float)
        status(f"per-step linear and baselines, definition {dname}, done")
    if attention:
        yB, evB = defs["B"]
        s, hist = attention_cv(S, n_t, ai, yB, evB, reach)
        for dname, (y, ev) in defs.items():
            r = step_metrics(s, y, ev, ai, tpos, reach, onset, rid_a, rid_r)
            r.update({"definition": dname, "features": "attention monitor (layer-38 think pair, trained on definition B)"})
            rows.append(r)
        frames["attention"] = s
        json.dump(hist, open(OUT / "attention_history.json", "w"))
    json.dump({"rows": rows, "drop": drop}, open(OUT / "step.json", "w"), indent=1, default=float)
    pd.DataFrame(frames).to_csv(OUT / "step_scores.csv.gz", index=False)
    status("per-step stage done")


# ------------------------------------------------------------------------------------------------ transfer
def fit_probe(X, y, groups, perms=None):
    """A probe fitted on every row (PCA-z, C by inner grouped CV); returns (scorer, C, null scorers refitted on permuted
    labels with the same PCA and C)."""
    from sklearn.linear_model import LogisticRegression
    f = pca_fit(X)
    Z = f(X)
    C = tp.choose_C(Z, y, groups)
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Z, y)
    nulls = []
    for yp in perms or []:
        lp = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Z, yp.astype(int))
        nulls.append(lp)
    return (lambda Xn: lr.decision_function(f(Xn))), C, (lambda Xn: [m.decision_function(f(Xn)) for m in nulls])


def controls_frame():
    import pandas as pd
    sub = pd.read_csv(HUB / "labels" / "controls_600" / "subsample.csv")
    have = {(p.parent.name, p.stem) for p in (THINK / "controls_600").glob("*/*.npz")}
    rows = []
    for r in sub.itertuples():
        ags = sorted(a for (rn, a) in have if rn == r.run)
        if ags:
            rows.append({"run": r.run, "agent": ags[0], "author": r.author, "condition": r.condition, "task": r.task,
                         "y": int(r.cheat)})
    return pd.DataFrame(rows)


def team_frame():
    import pandas as pd
    A = pd.read_csv(tp.OUT / "labels" / "agents.csv")
    W = A[A.role != "regulator"].reset_index(drop=True)
    W = W[[tp.think_path("board_seed", r, a).is_file() for r, a in zip(W.run, W.agent)]].reset_index(drop=True)
    return W


def means_of(s, df):
    """team_probes.load_means + agent_matrix rows (the same layout as feat()) for a team_probes set."""
    acts = tp.load_means(s, df)
    keys = list(zip(df.run, df.agent))
    ok = np.array([k in acts for k in keys])
    out = {}
    for cfg in CFGS:
        X = np.zeros((len(df), 30720 if cfg == "all" else 10240), np.float32)
        X[ok] = tp.agent_matrix(acts, [k for k, o in zip(keys, ok) if o], "all" if cfg == "all" else cfg, "think").astype(np.float32)
        out[cfg] = X
    return out, ok


def stage_transfer():
    import pandas as pd
    rows, score_frames = [], []
    gf = load_labels(GF)
    Mg = load_means(GF)
    judge = (gf.label_kind == "judge").values
    gid = np.array([f"{a}/{b}" for a, b in zip(gf.run, gf.agent)])
    yg_all = gf.y.values.astype(float)
    cb, eg = load_labels(CB), load_labels(EG)
    Mc, Me = load_means(CB), load_means(EG)
    ctl = controls_frame()
    cX, cok = means_of("controls_600", ctl)
    team = team_frame()
    tX, tok_ = means_of("board_seed", team)
    targets_team = {"cheating used or above": (team.cheat.values >= tp.USED).astype(float),
                    "cheating opened or above": (team.cheat.values >= tp.OPENED).astype(float)}
    log(f"transfer: controls {len(ctl)} ({cok.sum()} with tensors), team workers {len(team)} ({tok_.sum()}), goodfire {len(gf)}")

    def add(train, cfg, target, group, s, y, clusters, nul=None):
        r = stats(s, y, clusters, {"train": train, "features": cfg_name(cfg), "target": target, "group": group})
        if nul is not None:
            na = np.array([tp.auc(v[np.isfinite(y) & np.isfinite(v)], y[np.isfinite(y) & np.isfinite(v)]) for v in nul])
            r.update(null_extra(na, r["auroc"]))
        rows.append(r)
        return r

    def score_targets(train, cfg, scorer, nscorer):
        # Goodfire: judge pool, each author, each environment (monitor environments against the monitor verdict)
        Xg = feat(Mg, np.arange(len(gf)), cfg)
        sg, ng = scorer(Xg), nscorer(Xg)
        del Xg
        score_frames.append(pd.DataFrame({"train": train, "features": cfg_name(cfg), "set": GF, "run": gf.run, "agent": gf.agent,
                                          "y": yg_all, "score": sg}))
        yj = np.where(judge, yg_all, np.nan)
        add(train, cfg, "Goodfire judge sets (verdict yes vs no)", "all", sg, yj, gid, ng)
        for a in AUTHORS:
            m = gf.author.values == a
            add(train, cfg, "Goodfire judge sets (verdict yes vs no)", f"author {AUTHORS[a]}", sg, np.where(m, yj, np.nan), gid, ng)
        for e in JUDGE_ENVS + MONITOR_ENVS:
            m = gf.env.values == e
            add(train, cfg, "Goodfire " + ("judge" if e in JUDGE_ENVS else "monitor") + " verdict", f"environment {e}",
                sg, np.where(m, yg_all, np.nan), gid, ng)
        w, _ = within_auroc(sg, yj, gf.run.values)
        rows.append({"train": train, "features": cfg_name(cfg), "target": "Goodfire judge sets (verdict yes vs no)",
                     "group": "within run dir (pairs inside one run dir)", "auroc": w})
        for sname, df, M in ((CB, cb, Mc), (EG, eg, Me)):
            k = (df.n_turns > 0).values
            X = np.zeros((len(df), 30720 if cfg == "all" else 10240), np.float32)
            X[k] = feat(M, np.flatnonzero(k), cfg)
            sv = np.where(k, scorer(X), np.nan)
            nv = [np.where(k, v, np.nan) for v in nscorer(X)]
            yv = np.where(k, df.y.values.astype(float), np.nan)
            cl = df.pair.values if sname == EG else np.array([f"{a}/{b}" for a, b in zip(df.run, df.agent)])
            add(train, cfg, f"{sname} (its own label)", "all", sv, yv, cl, nv)
            strata = df.run.values if sname == CB else df.author.values
            w, per = within_auroc(sv, yv, strata)
            rows.append({"train": train, "features": cfg_name(cfg), "target": f"{sname} (its own label)",
                         "group": "within " + ("harness" if sname == CB else "author"), "auroc": w,
                         "per": json.dumps({k_: [round(a_, 3), p_, n_] for k_, (a_, p_, n_) in per.items()})})
            score_frames.append(pd.DataFrame({"train": train, "features": cfg_name(cfg), "set": sname, "run": df.run, "agent": df.agent,
                                              "y": yv, "score": sv}))
    # (a) controls-trained cheating probe (team_probes' transfer recipe: solo arm, and solo + cheat) -> Goodfire and others
    for arm in ("solo", "solo + cheat"):
        C_ = (ctl[ctl.condition == "solo"] if arm == "solo" else ctl)
        ii = np.flatnonzero(cok & ctl.index.isin(C_.index))
        y, g = ctl.y.values[ii], ctl.run.values[ii]
        perms = perms_within(y.astype(float), ctl.author.values[ii])
        for cfg in CFGS:
            scorer, C, nscorer = fit_probe(cX[cfg][ii], y, g, perms)
            score_targets(f"controls {arm} ({len(ii)} agents, {int(y.sum())} cheat)", cfg, scorer, nscorer)
            log(f"  transfer controls {arm} {cfg}: " + ", ".join(f"{r['group']} {r['auroc']:.3f}" for r in rows[-14:-8]))
    # (b) Goodfire-trained probe (all judge rollouts, verdict yes vs no) -> controls, team workers, cheatbench, evilgenie
    ii = np.flatnonzero(judge & np.isfinite(yg_all))
    y, g = yg_all[ii].astype(int), gid[ii]
    perms = [p.astype(int) for p in perms_within(y.astype(float), gf.run.values[ii])]
    for cfg in CFGS:
        Xg = feat(Mg, ii, cfg)
        scorer, C, nscorer = fit_probe(Xg, y, g, perms)
        del Xg
        train = f"Goodfire judge sets ({len(ii)} rollouts, {int(y.sum())} yes)"
        for arm in ("solo", "solo + cheat"):
            C_ = (ctl[ctl.condition == "solo"] if arm == "solo" else ctl)
            m = cok & ctl.index.isin(C_.index)
            sc = scorer(cX[cfg])
            nv = nscorer(cX[cfg])
            yc = np.where(m, ctl.y.values.astype(float), np.nan)
            add(train, cfg, f"controls {arm}", "all", np.where(m, sc, np.nan), yc, ctl.run.values, [np.where(m, v, np.nan) for v in nv])
            w, per = within_auroc(np.where(m, sc, np.nan), yc, ctl.author.values)
            rows.append({"train": train, "features": cfg_name(cfg), "target": f"controls {arm}", "group": "within author",
                         "auroc": w, "per": json.dumps({k_: [round(a_, 3), p_, n_] for k_, (a_, p_, n_) in per.items()})})
            if arm == "solo + cheat":
                score_frames.append(pd.DataFrame({"train": train, "features": cfg_name(cfg), "set": "controls_600", "run": ctl.run,
                                                  "agent": ctl.agent, "y": yc, "score": sc}))
        st = scorer(tX[cfg])
        nv = nscorer(tX[cfg])
        for tname, yt in targets_team.items():
            add(train, cfg, f"team workers (board seed), {tname}", "all", np.where(tok_, st, np.nan), np.where(tok_, yt, np.nan),
                team.run.values, [np.where(tok_, v, np.nan) for v in nv])
        score_frames.append(pd.DataFrame({"train": train, "features": cfg_name(cfg), "set": "board_seed", "run": team.run,
                                          "agent": team.agent, "y": targets_team["cheating used or above"], "score": st}))
        for sname, df, M in ((CB, cb, Mc), (EG, eg, Me)):
            k = (df.n_turns > 0).values
            X = np.zeros((len(df), 30720 if cfg == "all" else 10240), np.float32)
            X[k] = feat(M, np.flatnonzero(k), cfg)
            sv = np.where(k, scorer(X), np.nan)
            nv = [np.where(k, v, np.nan) for v in nscorer(X)]
            yv = np.where(k, df.y.values.astype(float), np.nan)
            cl = df.pair.values if sname == EG else np.array([f"{a}/{b}" for a, b in zip(df.run, df.agent)])
            add(train, cfg, f"{sname} (its own label)", "all", sv, yv, cl, nv)
            w, per = within_auroc(sv, yv, df.run.values if sname == CB else df.author.values)
            rows.append({"train": train, "features": cfg_name(cfg), "target": f"{sname} (its own label)",
                         "group": "within " + ("harness" if sname == CB else "author"), "auroc": w,
                         "per": json.dumps({k_: [round(a_, 3), p_, n_] for k_, (a_, p_, n_) in per.items()})})
            score_frames.append(pd.DataFrame({"train": train, "features": cfg_name(cfg), "set": sname, "run": df.run,
                                              "agent": df.agent, "y": yv, "score": sv}))
        log(f"  transfer goodfire {cfg} done")
    pd.DataFrame(rows).to_csv(OUT / "transfer.csv", index=False)
    pd.concat(score_frames, ignore_index=True).to_csv(OUT / "transfer_scores.csv.gz", index=False)
    status("transfer stage done")


def stage_transfer_text():
    """The TF-IDF baseline of the transfer table: the agent's words of the training set (rendered text), columns in at
    least 2 training documents and the 50000 most frequent there, sublinear tf and IDF from the training documents,
    L2 logistic regression with C by inner grouped CV; applied unchanged to each target set."""
    import pandas as pd
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    hv = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 20, alternate_sign=False, norm=None, dtype=np.float32)

    def docs_of(s, df):
        return [" ".join(a + " " + b for a, b in rendered_turns(ACTS / s / r / a_ / "rendered.txt")) for r, a_ in zip(df.run, df.agent)]
    gf, cb, eg = load_labels(GF), load_labels(CB), load_labels(EG)
    ctl, team = controls_frame(), team_frame()
    H = {GF: hv.transform([words_doc(w) for w in load_words(GF)]).tocsc(), CB: hv.transform([words_doc(w) for w in load_words(CB)]).tocsc(),
         EG: hv.transform([words_doc(w) for w in load_words(EG)]).tocsc(), "controls_600": hv.transform(docs_of("controls_600", ctl)).tocsc(),
         "board_seed": hv.transform(docs_of("board_seed", team)).tocsc()}
    judge = (gf.label_kind == "judge").values
    yg = gf.y.values.astype(float)
    gid = np.array([f"{a}/{b}" for a, b in zip(gf.run, gf.agent)])
    rows = []

    def fit(Hs, y, g):
        df_ = np.asarray((Hs > 0).sum(0)).ravel()
        tot = np.asarray(Hs.sum(0)).ravel()
        cand = np.flatnonzero(df_ >= 2)
        cols = cand[np.argsort(-tot[cand], kind="stable")[:50000]]
        X = Hs[:, cols].tocsr()
        tf = TfidfTransformer(sublinear_tf=True).fit(X)
        Z = tf.transform(X)
        C = tp.choose_C(Z, y, g, tp.CS_TEXT)
        lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Z, y)
        return lambda Ht: lr.decision_function(tf.transform(Ht[:, cols].tocsr()))

    def add(train, target, group, s, y, cl):
        r = stats(s, y, cl, {"train": train, "features": "TF-IDF agent's words (rendered text)", "target": target, "group": group})
        rows.append(r)
    nk = (cb.n_turns > 0).values
    ycb = np.where(nk, cb.y.values.astype(float), np.nan)
    cbid = np.array([f"{a}/{b}" for a, b in zip(cb.run, cb.agent)])
    for arm in ("solo", "solo + cheat"):
        ii = np.flatnonzero((ctl.condition == "solo").values) if arm == "solo" else np.arange(len(ctl))
        sc = fit(H["controls_600"][ii], ctl.y.values[ii], ctl.run.values[ii])
        train = f"controls {arm} ({len(ii)} agents, {int(ctl.y.values[ii].sum())} cheat)"
        sg = sc(H[GF])
        yj = np.where(judge, yg, np.nan)
        add(train, "Goodfire judge sets (verdict yes vs no)", "all", sg, yj, gid)
        for a in AUTHORS:
            add(train, "Goodfire judge sets (verdict yes vs no)", f"author {AUTHORS[a]}", sg, np.where(gf.author.values == a, yj, np.nan), gid)
        for e in JUDGE_ENVS + MONITOR_ENVS:
            add(train, "Goodfire " + ("judge" if e in JUDGE_ENVS else "monitor") + " verdict", f"environment {e}", sg,
                np.where(gf.env.values == e, yg, np.nan), gid)
        rows.append({"train": train, "features": "TF-IDF agent's words (rendered text)", "target": "Goodfire judge sets (verdict yes vs no)",
                     "group": "within run dir (pairs inside one run dir)", "auroc": within_auroc(sg, yj, gf.run.values)[0]})
        s_ = sc(H[CB])
        add(train, "cheatbench (its own label)", "all", s_, ycb, cbid)
        rows.append({"train": train, "features": "TF-IDF agent's words (rendered text)", "target": "cheatbench (its own label)",
                     "group": "within harness", "auroc": within_auroc(s_, ycb, cb.run.values)[0]})
        s_ = sc(H[EG])
        add(train, "evilgenie (its own label)", "all", s_, eg.y.values.astype(float), eg.pair.values)
        rows.append({"train": train, "features": "TF-IDF agent's words (rendered text)", "target": "evilgenie (its own label)",
                     "group": "within author", "auroc": within_auroc(s_, eg.y.values.astype(float), eg.author.values)[0]})
    ii = np.flatnonzero(judge & np.isfinite(yg))
    sc = fit(H[GF][ii], yg[ii].astype(int), gid[ii])
    train = f"Goodfire judge sets ({len(ii)} rollouts, {int(yg[ii].sum())} yes)"
    s_ = sc(H["controls_600"])
    for arm in ("solo", "solo + cheat"):
        m = (ctl.condition == "solo").values if arm == "solo" else np.ones(len(ctl), bool)
        yc = np.where(m, ctl.y.values.astype(float), np.nan)
        add(train, f"controls {arm}", "all", s_, yc, ctl.run.values)
        rows.append({"train": train, "features": "TF-IDF agent's words (rendered text)", "target": f"controls {arm}",
                     "group": "within author", "auroc": within_auroc(s_, yc, ctl.author.values)[0]})
    st = sc(H["board_seed"])
    for tname, yt in (("cheating used or above", team.cheat.values >= tp.USED), ("cheating opened or above", team.cheat.values >= tp.OPENED)):
        add(train, f"team workers (board seed), {tname}", "all", st, yt.astype(float), team.run.values)
    s_ = sc(H[CB])
    add(train, "cheatbench (its own label)", "all", s_, ycb, cbid)
    rows.append({"train": train, "features": "TF-IDF agent's words (rendered text)", "target": "cheatbench (its own label)",
                 "group": "within harness", "auroc": within_auroc(s_, ycb, cb.run.values)[0]})
    s_ = sc(H[EG])
    add(train, "evilgenie (its own label)", "all", s_, eg.y.values.astype(float), eg.pair.values)
    rows.append({"train": train, "features": "TF-IDF agent's words (rendered text)", "target": "evilgenie (its own label)",
                 "group": "within author", "auroc": within_auroc(s_, eg.y.values.astype(float), eg.author.values)[0]})
    pd.DataFrame(rows).to_csv(OUT / "transfer_text.csv", index=False)
    log(pd.DataFrame(rows)[["train", "target", "group", "auroc"]].round(3).to_string())


# ------------------------------------------------------------------------------------------------ cheatbench diagnosis
def stage_diag():
    """The 0.279 transfer AUROC on CheatBench in team_probes' RESULTS.md: its scores file read back, the label polarity
    and the join checked against meta.json, the AUROC split into pairs inside one harness and pairs across harnesses."""
    import pandas as pd
    d = pd.read_csv(tp.OUT / "transfer_other_sets_scores.csv")
    d = d[d.set == CB].copy()
    lab = load_labels(CB).set_index(["run", "agent"])
    out = {"rows_in_file": int(len(d)), "features": {}}
    j = lab.loc[list(zip(d.run, d.agent))]
    out["join_rows_found_in_meta"] = int(len(j))
    out["y_equals_meta_cheat"] = int((j.cheat.values.astype(float) == d.y.values).sum())
    out["label_to_cheat"] = {f"{a}->{b}": int(n) for (a, b), n in lab.groupby(["label", "cheat"]).size().items()}
    out["rollouts_left_out_zero_turns"] = {f"{r}|{l}": int(n) for (r, l), n in lab[lab.n_turns == 0].groupby(["run", "label"]).size().items()}
    out["bootstrap_clusters"] = int(d.run.nunique())
    for fname_, g in d.groupby("features"):
        g = g[np.isfinite(g.y)]
        s, y, h = g.score.values.astype(float), g.y.values.astype(int), np.asarray(g.run.values, dtype=str)
        P, N = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        same = h[P][:, None] == h[N][None, :]
        gt = (s[P][:, None] > s[N][None, :]) + 0.5 * (s[P][:, None] == s[N][None, :])
        per = {}
        for v in sorted(set(h)):
            m = h == v
            per[v] = {"n": int(m.sum()), "pos": int(y[m].sum()), "neg": int((1 - y[m]).sum()),
                      "auroc": tp.auc(s[m], y[m]), "mean_score_pos": float(s[m & (y == 1)].mean()),
                      "mean_score_neg": float(s[m & (y == 0)].mean()) if (m & (y == 0)).any() else None,
                      "mean_score": float(s[m].mean())}
        out["features"][fname_] = {
            "n": int(len(g)), "pos": int(y.sum()), "neg": int((1 - y).sum()), "pooled_auroc": tp.auc(s, y),
            "pairs": int(same.size), "pairs_same_harness": int(same.sum()),
            "auroc_same_harness_pairs": float(gt[same].mean()), "auroc_cross_harness_pairs": float(gt[~same].mean()),
            "auroc_harness_mean_only": tp.auc(np.array([per[v]["mean_score"] for v in h]), y),
            "per_harness": per}
    json.dump(out, open(OUT / "cheatbench_diagnosis.json", "w"), indent=1, default=float)
    log(json.dumps(out, indent=1, default=float))
    return out


# ------------------------------------------------------------------------------------------------ report
def f3(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(x) else f"{x:.3f}"


def fci(a, lo, hi):
    return f"{f3(a)} [{f3(lo)}, {f3(hi)}]" if np.isfinite(float(lo if lo is not None else np.nan)) else f3(a)


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def write_report():
    import pandas as pd
    T = json.load(open(OUT / "traj.json"))
    chk = json.load(open(OUT / "labels" / "checks.json"))
    dg = json.load(open(OUT / "cheatbench_diagnosis.json"))
    st = json.load(open(OUT / "step.json")) if (OUT / "step.json").is_file() else None
    tr = pd.read_csv(OUT / "transfer.csv")
    tt = pd.read_csv(OUT / "transfer_text.csv")
    L = []
    w = L.append
    w("# Hermeneutic probes on externally sourced transcripts: Goodfire, CheatBench, EvilGenie\n")
    w("Generated by `python -m hermeneutics.external_probes report` from the files in this folder. No model calls and no "
      "spend: every number comes from the reader's think-token activations already on the Hub and labels the source "
      "datasets ship.\n")
    w("**Provenance.** The transcripts were written by other labs' agents and labelled by the datasets' authors: "
      f"[{GF_REPO}](https://huggingface.co/datasets/{GF_REPO}) (revision `{GF_REV[:7]}`; agents Kimi K3, GLM 5.2, "
      "Qwen 3.8 Max), steinad/CheatBench (agents claude-3-7-sonnet, claude-opus-4-6, gpt-5.3-codex, gpt-5.4) and "
      "WiserHumanExperimental/evilgenie-escalation (ten agents). Another agent rendered them in Qwen3.8-27B's chat template, "
      "teacher-forced them through the reader and extracted the think pair (think-open and think-close residuals at layers "
      "26, 38 and 51, one pair per assistant turn) into the public Hub dataset unrulyabstractions/apart-swarm-align. None of "
      "these authors, tasks or labels were used to build any earlier probe, prompt or label in this project.\n")
    w("**Recipe.** As in the sprint: the mean over turns of the think pair, training-fold PCA to min(128, half the "
      "training rows), z-scoring, L2 logistic regression (class-balanced) with C chosen by an inner 3-fold grouped CV. "
      "Intervals: bootstrap over rollouts (EvilGenie: over matched pairs; controls and team workers: over runs), 1000 "
      "draws for pooled AUROCs, 300 for within AUROCs, 500 for the per-step tables. Null: 50 draws with the labels permuted inside each run dir (CheatBench: harness; EvilGenie: author), PCA "
      "kept, C fixed at the median chosen C. TF-IDF: word unigrams and bigrams of the agent's words in the rendered text "
      "(reasoning, replies and tool calls), 50,000 columns, fitted on the training rows. Metadata: author and environment "
      "one-hot (only what varies inside the training rows; EvilGenie adds its condition and problem, CheatBench its "
      "harness and benchmark).\n")
    w("**Two AUROCs.** *Pooled* ranks every positive against every negative. *Within* keeps only the positive-negative "
      "pairs that share a stratum (one run dir, one harness, one author, or one held-out group) and weights them equally. "
      "Hack rates differ by run dir from one half to over nine in ten, so a probe that only recognises the run dir already scores a pooled "
      "AUROC well above 0.5; the pooled null shows how far. The within AUROC is the number that measures cheating "
      "detection itself, and with held-out splits, where every group gets its own model, it is the only fair one.\n")
    # data table
    vc = chk["goodfire_verdict_counts"]
    w("## Data\n")
    w(md(["Set", "Rollouts extracted", "Used", "Positive", "Negative", "Left out", "Label"], [
        ["Goodfire, judge run dirs (SWE-bench Verified, DeepSWE, ImpossibleBench; 9 run dirs)",
         vc.get("judge|yes", 0) + vc.get("judge|no", 0) + vc.get("judge|split", 0) + 6,
         vc.get("judge|yes", 0) + vc.get("judge|no", 0), vc.get("judge|yes", 0), vc.get("judge|no", 0),
         f"{vc.get('judge|split', 0)} split verdicts, 6 not judged",
         "3-vote consensus judge verdict (`labels/judge_rollouts.jsonl`), yes vs no"],
        ["Goodfire, monitor run dirs (tau3-bench, ShoppingBench; 4 run dirs)",
         vc.get("monitor|yes", 0) + vc.get("monitor|no", 0) + vc.get("monitor|unclear", 0),
         vc.get("monitor|yes", 0) + vc.get("monitor|no", 0), vc.get("monitor|yes", 0), vc.get("monitor|no", 0),
         f"{vc.get('monitor|unclear', 0)} unclear",
         "single-pass generic monitor (`generic_monitor/monitor_rollouts.jsonl`); the source says no ground truth exists here"],
        ["CheatBench (3 harnesses, 4 authors)", chk["cheatbench_rollouts"], chk["cheatbench_rollouts"] - chk["cheatbench_zero_turns"],
         134, 90, f"{chk['cheatbench_zero_turns']} with zero assistant turns (all Pilot, benign)",
         "meta.json `label`: cheating or attempt = 1, benign = 0 (attempt-dropped variant too)"],
        ["EvilGenie (10 authors)", chk["evilgenie_rollouts"], chk["evilgenie_rollouts"], 133, 133, "none",
         f"meta.json `sel_label` hack vs legitimate; {chk['evilgenie_pairs']} matched pairs, one of each, same author"]]))
    w(f"\nGoodfire joins: the verdict in our meta.json equals the source file's for every judged rollout "
      f"({chk['goodfire_meta_verdict_equal_source']} of {chk['goodfire_rollouts']}, the rest unjudged on both sides), and the "
      f"Hub labels.csv agrees ({chk['goodfire_hub_verdict_equal_source']}); the tensors' turn count equals meta.json's and the "
      f"turn-node list's for all {chk['goodfire_npz_turns_equal_meta']}. Passages: every passage's node_index lands on a reader "
      f"turn except in {chk['goodfire_rollouts_with_unmapped_passages']} rollouts (branching traces), which the per-step "
      "analysis drops.\n")
    # 0 cheatbench
    w("## Question 0. Why did the controls-trained probe read 0.279 on CheatBench?\n")
    F = dg["features"]["probe Lall think-open+close"]
    rows = []
    for h, v in F["per_harness"].items():
        rows.append([h, v["n"], v["pos"], v["neg"], f3(v["auroc"]), f3(v["mean_score_pos"]), f3(v["mean_score_neg"])])
    rows.append(["all three, pooled", F["n"], F["pos"], F["neg"], f3(F["pooled_auroc"]), "", ""])
    w(md(["Harness", "Rollouts", "Cheat", "Benign", "AUROC", "Mean score, cheat", "Mean score, benign"], rows))
    w("")
    w(md(["Pairs", "Count", "AUROC over these pairs"], [
        ["positive and negative in the same harness", F["pairs_same_harness"], f3(F["auroc_same_harness_pairs"])],
        ["positive and negative in different harnesses", F["pairs"] - F["pairs_same_harness"], f3(F["auroc_cross_harness_pairs"])],
        ["all (the reported number)", F["pairs"], f3(F["pooled_auroc"])],
        ["score replaced by its harness's mean score", F["pairs"], f3(F["auroc_harness_mean_only"])]]))
    w(f"\nThe label polarity is right and the join is right: `label` maps cheating and attempt to 1 and benign to 0 in "
      f"every meta.json ({', '.join(f'{k} {v}' for k, v in dg['label_to_cheat'].items())}), and the y column of team_probes' "
      f"scores file equals meta.json's `cheat` on all {dg['y_equals_meta_cheat']} of {dg['join_rows_found_in_meta']} rows. "
      "The cause is a harness confound. Inside each harness the probe ranks cheating at or above benign. Across harnesses it "
      "ranks the other way, because the Pilot rollouts (claude-opus-4-6 on terminal-bench-2) score far lower than the "
      "rest, and Pilot holds 106 of the 134 positives but only 5 of the 90 negatives: its other 20 benign rollouts have "
      "zero assistant turns in the converted trace and were left out. Six pairs in seven cross harnesses, so the pooled "
      "AUROC is mostly a comparison of harnesses. The interval in that table also resampled only the three harnesses "
      f"({dg['bootstrap_clusters']} clusters), which is why it was so wide; resampled over rollouts it is narrow and still "
      "below 0.5 (the transfer table below reproduces 0.279). The within-harness AUROC is the number to read.\n")
    # 1 goodfire trajectory
    w("## Question 1. Does the probe detect Goodfire's reward hacking at the trajectory level?\n")

    def tbl(rows_, within_keys, null=True):
        out = []
        for r in rows_:
            line = [r["features"], r["n"], r["pos"], r["neg"], fci(r["auroc"], r["ci_lo"], r["ci_hi"])]
            if null:
                line.append(f"{f3(r.get('null_mean'))} / {f3(r.get('null_95'))}" if "null_mean" in r else "")
            for k in within_keys:
                c = r.get(f"w_{k}_ci") or [np.nan, np.nan]
                line.append(fci(r.get(f"w_{k}"), c[0], c[1]))
                if null:
                    line.append(f"{f3(r.get(f'w_{k}_null_mean'))} / {f3(r.get(f'w_{k}_null_95'))}" if f"w_{k}_null_mean" in r else "")
            out.append(line)
        hdr = ["Features", "Rollouts", "Positive", "Negative", "Pooled AUROC [95% CI]"] + (["Pooled null mean / 95th"] if null else [])
        for k in within_keys:
            hdr += [f"Within {k} AUROC [95% CI]"] + (["Within null mean / 95th"] if null else [])
        return md(hdr, out)
    w("### 1a. Five folds (grouped by rollout, stratified by run dir and verdict), judge run dirs, verdict yes vs no\n")
    w(tbl(T["gf_5fold"], ["run dir"]))
    w("\nPer run dir (the same held-out scores, AUROC inside each run dir):\n")
    by = defaultdict(dict)
    cnt = {}
    for r in T["gf_5fold_by_run"]:
        by[r["group"]][r["features"]] = r["auroc"]
        cnt[r["group"]] = (r["n"], r["pos"], r["neg"])
    fe = [cfg_name(c) for c in CFGS] + ["TF-IDF agent's words (rendered text)", "metadata"]
    w(md(["Run dir", "Rollouts", "Yes", "No"] + ["probe " + x.replace("think pair, ", "") for x in fe[:4]] + ["TF-IDF", "metadata"],
         [[g, *cnt[g]] + [f3(by[g].get(x)) for x in fe] for g in sorted(by)]))
    w("\nRobustness (concatenated layers only). Folds grouped by task keep every SWE-bench or DeepSWE instance on one "
      "side of the split, across authors too. The all-13 rows pool the four monitor run dirs in, with the monitor verdict "
      "as label (the label team_probes' transfer used):\n")
    out = []
    for nm, rr in (("folds grouped by task, judge run dirs", T["gf_5fold_task"]), ("all 13 run dirs, judge + monitor verdicts", T["gf_5fold_all13"])):
        for r in rr:
            c = r.get("w_run dir_ci") or [np.nan, np.nan]
            out.append([nm, r["features"], r["n"], r["pos"], r["neg"], fci(r["auroc"], r["ci_lo"], r["ci_hi"]), fci(r.get("w_run dir"), c[0], c[1])])
    w(md(["Evaluation", "Features", "Rollouts", "Positive", "Negative", "Pooled AUROC [95% CI]", "Within run dir AUROC [95% CI]"], out))
    w("")
    if "gf_5fold_enacted" in T:
        c_ = T["gf_5fold_enacted_counts"]
        w(f"\nSensitivity: the source's `verdict_enacted` (attempted or enacted hacks only) as the label. A verdict-yes "
          f"rollout whose hacks were only considered ({c_['considered_only']} rollouts) becomes negative; "
          f"{c_['enacted_split']} rollouts with a split enacted verdict are left out. Same folds recipe:\n")
        w(tbl(T["gf_5fold_enacted"], ["run dir"]))
        w("")
    w("**Reading.** Inside a run dir the probe separates hacked from clean rollouts well on SWE-bench Verified and "
      "ImpossibleBench and only weakly on DeepSWE, where layer 26 alone does clearly better than the concatenation. The "
      "pooled AUROC sits well above its run-dir null, and metadata alone reaches a pooled AUROC far from 0.5 with a "
      "within AUROC of 0.5, so the pooled number carries run-dir composition. Grouping folds by task changes nothing. "
      "TF-IDF on the agent's own words beats every probe in seven of the nine run dirs; the layer-26 probe beats it on "
      "Kimi K3's and Qwen 3.8 Max's DeepSWE runs.\n")
    w("### 1b. Held-out author (train on two authors' judge rollouts, test on the third)\n")
    w(tbl(T["gf_author"], ["held-out author", "run dir"]))
    w("\nPer held-out author:\n")
    by, cnt = defaultdict(dict), {}
    for r in T["gf_author_by"]:
        by[r["group"]][r["features"]] = (r["auroc"], r["ci_lo"], r["ci_hi"])
        cnt[r["group"]] = (r["n"], r["pos"], r["neg"])
    w(md(["Held-out author", "Rollouts", "Yes", "No"] + ["probe " + x.replace("think pair, ", "") for x in fe[:4]] + ["TF-IDF", "metadata (environment)"],
         [[g, *cnt[g]] + [fci(*by[g][x]) if x in by[g] else "-" for x in fe] for g in sorted(by)]))
    w("\n**Reading.** The probe keeps most of its within-run-dir separation on an author it never saw, best at layer "
      "26. TF-IDF transfers across authors at least as well.\n")
    w("\n### 1c. Held-out environment (train on the judge rollouts of the other environments, test on one)\n")
    w(tbl(T["gf_env"], ["held-out environment", "run dir"]))
    w("\nPer held-out environment (tau3-bench and ShoppingBench are test only, scored against the monitor verdict):\n")
    by, cnt = defaultdict(dict), {}
    for r in T["gf_env_by"]:
        by[r["group"]][r["features"]] = (r["auroc"], r["ci_lo"], r["ci_hi"])
        cnt[r["group"]] = (r["n"], r["pos"], r["neg"])
    w(md(["Held-out environment", "Rollouts", "Positive", "Negative"] + ["probe " + x.replace("think pair, ", "") for x in fe[:4]] + ["TF-IDF", "metadata (author)"],
         [[g, *cnt[g]] + [fci(*by[g][x]) if x in by[g] else "-" for x in fe] for g in JUDGE_ENVS + MONITOR_ENVS if g in by]))
    w("\n**Reading.** A probe trained on two environments still ranks the third's hacks well on ImpossibleBench and "
      "SWE-bench Verified, less well on DeepSWE. Against the generic monitor's verdict it transfers to ShoppingBench and "
      "not to tau3-bench. The pooled AUROC of a held-out environment split mixes models fitted on different training "
      "sets (layer 38 pools below 0.5 while every held-out environment is above it), so the within columns are the ones "
      "to read. TF-IDF again does as well or better.\n")
    # 2 per step
    if st:
        w("## Question 2. Does a per-step monitor find the flagged step in Goodfire rollouts?\n")
        d = st["drop"]
        rows_ = st["rows"]
        r0 = rows_[0]
        w(f"Rollouts: judge run dirs, verdict yes or no, every passage joined to a reader turn, and for a yes rollout at "
          f"least one flagged passage (the onset is known): {r0['rollouts']} rollouts ({r0['pos_rollouts']} yes); "
          f"{min(x['turns'] for x in rows_)} evaluated turns under definition A and {max(x['turns'] for x in rows_)} under B. Left out: {d['split_or_unjudged']} split or unjudged, "
          f"{d['unmapped_passages']} with a passage on no reader turn, {d['yes_without_flagged_turn']} yes without a flagged "
          "passage. A flagged passage is a `positive` consensus passage (all three votes, verified quote); `ambiguous` "
          "passages are neither. **Definition A**: a turn is positive when a flagged passage falls in it (turns whose only "
          "flags are ambiguous are left out). **Definition B**: positive from the turn of the first flagged passage on "
          "(ambiguous-only turns before the onset left out). The think-open of turn t sees everything before turn t's "
          "reasoning, the think-close also sees that reasoning; a flag on turn t's reply or tool call is first visible "
          "at turn t+1.\n")
        w("The linear per-step probe uses, per fold, a PCA basis of 128 components fitted on 12,000 turns drawn from the "
          "training rollouts (all turns do not fit in memory), z-scoring on all training turns, then L2 logistic "
          "regression with C by inner grouped CV over rollouts. The attention monitor is attn_probe.py's simplified "
          "encoder (layer-38 think pair standardised per dimension, MLP to 128, learned positional encoding, two pre-norm "
          "causal self-attention layers, MLP head) with one binary head, trained on definition B plus the verdict at the "
          "last step, 20% of training rollouts held out for early stopping. Delay: threshold at 5% of the evaluated "
          "negative turns; delay = first crossing at or after the onset minus the onset, in turns. Folds are question 1a's.\n")
        for dname in ("A", "B"):
            rr = [r for r in rows_ if r["definition"] == dname]
            w(f"### Definition {dname}\n")
            out = []
            for r in rr:
                out.append([r["features"], r["turns"], r["pos_turns"], r["rollouts"], r["pos_rollouts"],
                            fci(r["step_auroc"], r["step_ci_lo"], r["step_ci_hi"]),
                            (f"{f3(r['step_null_mean'])} / {f3(r['step_null_95'])}" if "step_null_mean" in r else ""),
                            fci(r["rollout_auroc_max"], r["rollout_max_ci_lo"], r["rollout_max_ci_hi"]), f3(r["rollout_auroc_last"]),
                            f"{r['detected']} of {r['reaching']}",
                            f"{r['delay_median']:.1f} [{r['delay_q25']:.1f}, {r['delay_q75']:.1f}]" if np.isfinite(r["delay_median"]) else "-",
                            r["fired_before_onset"], f"{r['neg_rollouts_fired']} of {r['neg_rollouts']}"])
            w(md(["Features", "Turns", "Positive turns", "Rollouts", "Yes rollouts", "Step AUROC [95% CI]", "Step null mean / 95th",
                  "Rollout AUROC, max score [95% CI]", "Rollout AUROC, last score", "Yes rollouts detected",
                  "Delay, turns: median [q25, q75]", "Yes rollouts fired before onset", "No rollouts ever fired"], out))
            w("")
        w("**Reading.** Under definition A the linear probe finds the flagged turn itself far better than the turn index "
          "and better than the cumulative TF-IDF, which smears a turn's words into every later turn. Under definition B the "
          "attention monitor and the cumulative TF-IDF lead and the linear probe trails them slightly. The definition-B null "
          "is high because a permuted onset still marks late turns positive and the think pair encodes position; the "
          "probe sits clearly above it. At the 5% threshold a monitor detects most hacked rollouts within a few turns of "
          "the first flagged passage, but it also fires before that passage in a share of them, a large share under "
          "definition A. Either the judges flag later than the hack begins or the monitor responds to what precedes it; "
          "these tables cannot tell which. The attention monitor and the TF-IDF baseline fire on far fewer clean rollouts "
          "than the linear probe. The attention monitor's definition-A row is not a fair comparison, since it was trained "
          "for definition B. Its code is a copy of attn_probe.py's architecture with "
          "one binary head in place of the six ladder heads (attn_probe.py was being edited by another agent).\n")
    # 3 transfer
    w("## Question 3. Does a cheating probe transfer between the controls, Goodfire and the team runs?\n")
    w("Each probe is fitted once on every row of its training set (PCA, z-score, C by inner grouped CV) and applied "
      "unchanged. Null: the training labels permuted 50 times (inside author for the controls, inside run dir for "
      "Goodfire), refitted with the same PCA and C, applied to the same targets. TF-IDF: the same transfer with the "
      "agent's words.\n")
    TT = pd.concat([tr, tt], ignore_index=True)
    for train in TT.train.unique():
        sub_ = TT[TT.train == train]
        w(f"### Trained on {train}\n")
        out = []
        for (tg, gp), g in sub_.groupby(["target", "group"], sort=False):
            def cell(fn):
                x = g[g.features == fn]
                if not len(x):
                    return "-"
                x = x.iloc[0]
                return fci(x.auroc, x.get("ci_lo"), x.get("ci_hi")) if pd.notna(x.get("ci_lo")) else f3(x.auroc)
            x0 = g.iloc[0]
            m = g[g.features == "think pair, layers 26+38+51"]
            nul = (f"{f3(m.iloc[0].null_mean)} / {f3(m.iloc[0].null_95)}" if len(m) and pd.notna(m.iloc[0].get("null_mean")) else "")
            out.append([tg, gp, "" if pd.isna(x0.get("n")) else int(x0.n), "" if pd.isna(x0.get("pos")) else int(x0.pos),
                        "" if pd.isna(x0.get("neg")) else int(x0.neg)] + [cell(cfg_name(c)) for c in CFGS] + [nul, cell("TF-IDF agent's words (rendered text)")])
        w(md(["Target", "Group", "Rows", "Positive", "Negative"] + ["probe " + cfg_name(c).replace("think pair, ", "") for c in CFGS]
             + ["Null (layers 26+38+51) mean / 95th", "TF-IDF"], out))
        w("")
    w("**Reading.** The controls-trained probe ranks Goodfire's hacked rollouts above clean ones, above its null on the "
      "judge pool, and keeps part of that inside a run dir, best at layer 26. On the monitor-labelled environments the "
      "concatenated layers sit near chance and layer 26 alone does better. The other "
      "way round, the Goodfire-trained probe barely separates the controls' cheaters (only layer 26 clears chance), and on the team workers it "
      "sits at the edge of its null. The opened-or-above target has three negatives and says nothing. TF-IDF trained on "
      "the controls transfers to Goodfire about as well as the probe on the pooled judge sets and better inside a run dir, "
      "and the Goodfire TF-IDF transfers to "
      "CheatBench far better than any probe does.\n")
    # 4 cheatbench, evilgenie
    w("## Question 4. Does the probe detect cheating in CheatBench and EvilGenie?\n")
    w("CheatBench folds keep each rollout whole and stratify by harness and label; EvilGenie folds keep each matched pair "
      "whole and stratify by author. Held-out model: one split per author (CheatBench also per harness). Per-step: "
      "EvilGenie has no step-level label, so it has no per-step analysis. CheatBench localises the cheating moment, but "
      f"the localised snippet is found in the rendered text for only {chk['cheatbench_onset_found']} of 134 positives "
      "(the Pilot snippets quote raw JSON stream lines that the converter rewrote), too few to train or test a per-step "
      "probe, so it has none either.\n")
    for key, title, wk in [("cheatbench_cheating or attempt vs benign", "CheatBench, cheating or attempt vs benign", "harness"),
                           ("cheatbench_cheating vs benign (attempt dropped)", "CheatBench, cheating vs benign (the 9 attempts dropped)", "harness"),
                           ("evilgenie_hack vs legitimate", "EvilGenie, hack vs legitimate", "author")]:
        w(f"### {title}: five folds\n")
        w(tbl(T[f"{key}_5fold"], [wk] if key.startswith("cheatbench") else ["author", "problem", "author and problem"]))
        w("")
        holds = ["author", "harness"] if key.startswith("cheatbench") else ["author", "problem"]
        for h in holds:
            w(f"### {title}: held-out {h}\n")
            wk2 = [f"held-out {h}"] + ((["harness"] if h == "author" else []) if key.startswith("cheatbench")
                                       else (["problem"] if h == "author" else ["author"]))
            w(tbl(T[f"{key}_held_{h}"], wk2))
            by, cnt = defaultdict(dict), {}
            for r in T[f"{key}_held_{h}_by"]:
                by[r["group"]][r["features"]] = (r["auroc"], r["ci_lo"], r["ci_hi"])
                cnt[r["group"]] = (r["n"], r["pos"], r["neg"])
            w("")
            w(md([f"Held-out {h}", "Rollouts", "Positive", "Negative"] + ["probe " + x.replace("think pair, ", "") for x in fe[:4]] + ["TF-IDF", "metadata"],
                 [[g, *cnt[g]] + [fci(*by[g][x]) if x in by[g] else "-" for x in fe] for g in sorted(by)]))
            w("")
    w("**Reading.** Inside one harness, the CheatBench probe separates cheating from benign rollouts well above its "
      "null under five folds; the pooled AUROC above it is inflated by the harness confound of question 0. On a held-out "
      "author or harness the separation drops and the intervals widen; the Pilot split has five negatives, and its high "
      "AUROC rests on them. In CheatBench author and harness nearly coincide, so the held-out-author and held-out-harness "
      "splits largely repeat each other, and tasks rarely repeat, so task identity cannot carry the five-fold result. "
      "EvilGenie's pairs share an author but mostly not a problem, and its eight problems are close to one label each. "
      "Its probe result is mostly problem identity: metadata that knows the problem beats the probe, the probe's AUROC "
      "falls sharply inside one problem and inside one author and problem, and on a held-out problem it is at or below "
      "chance, as are metadata and TF-IDF. Inside one problem the five-fold probe stays above its null but well below "
      "TF-IDF, so EvilGenie gives at most weak evidence that the probe reads hacking itself. A held-out group with a handful of one class gives an unstable AUROC; the counts sit "
      "beside every number.\n")
    open(OUT / "RESULTS.md", "w").write("\n".join(L) + "\n")
    log(f"wrote {OUT / 'RESULTS.md'}")


# ------------------------------------------------------------------------------------------------ verification
def stage_verify():
    """Independent re-computation for VERIFICATION.md: one AUROC per results table with sklearn's roc_auc_score (and a
    brute-force pair count for a within AUROC) from the saved score files, the fold checks, the label join read back
    from the source files for five Goodfire rollouts, and three local think tensors compared with the Hub originals."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    T = json.load(open(OUT / "traj.json"))
    rep = {"auroc": [], "folds": {}, "join": [], "tensors": []}

    def chk(name, y, s, reported):
        ok = np.isfinite(y) & np.isfinite(s)
        v = float(roc_auc_score(y[ok].astype(int), s[ok]))
        rep["auroc"].append({"table": name, "rows": int(ok.sum()), "pos": int(y[ok].sum()), "sklearn": v,
                             "reported": float(reported), "match": abs(v - float(reported)) < 1e-9})
    # Q0: team_probes' own scores file
    d = pd.read_csv(tp.OUT / "transfer_other_sets_scores.csv")
    d = d[(d.set == CB) & (d.features == "probe Lall think-open+close")]
    chk("Q0 controls solo -> CheatBench (team_probes' scores file)", d.y.values, d.score.values,
        json.load(open(OUT / "cheatbench_diagnosis.json"))["features"]["probe Lall think-open+close"]["pooled_auroc"])
    g = pd.read_csv(OUT / "traj_scores_gf.csv.gz", low_memory=False)
    a = g[g["eval"] == "gf_5fold"]
    chk("Q1a Goodfire 5 folds, layers 26+38+51", a.y.values, a["think pair, layers 26+38+51"].values, T["gf_5fold"][0]["auroc"])
    m = a.run == "deepswe-glm-5.2-fp8"
    chk("Q1a per run dir deepswe-glm-5.2-fp8, layer 26", a.y.values[m], a["think pair, layer 26"].values[m],
        [r for r in T["gf_5fold_by_run"] if r["group"] == "deepswe-glm-5.2-fp8" and r["features"] == "think pair, layer 26"][0]["auroc"])
    # brute-force within-run-dir AUROC
    num = den = 0.0
    for rd, h in a[np.isfinite(a.y)].groupby("run"):
        sp_, sn_ = h[h.y == 1]["think pair, layers 26+38+51"].values, h[h.y == 0]["think pair, layers 26+38+51"].values
        cmp_ = (sp_[:, None] > sn_[None, :]) + 0.5 * (sp_[:, None] == sn_[None, :])
        num += cmp_.sum()
        den += cmp_.size
    rep["auroc"].append({"table": "Q1a within run dir, layers 26+38+51 (brute-force pair count)", "sklearn": num / den,
                         "reported": T["gf_5fold"][0]["w_run dir"], "match": abs(num / den - T["gf_5fold"][0]["w_run dir"]) < 1e-9})
    a = g[g["eval"] == "gf_5fold_task"]
    chk("Q1a robustness, folds grouped by task", a.y.values, a["think pair, layers 26+38+51"].values, T["gf_5fold_task"][0]["auroc"])
    a = g[g["eval"] == "gf_author"]
    m = (a.fold == "kimi").values
    chk("Q1b held-out author Kimi K3, layers 26+38+51", a.y.values[m], a["think pair, layers 26+38+51"].values[m],
        [r for r in T["gf_author_by"] if r["group"] == "Kimi K3" and r["features"] == "think pair, layers 26+38+51"][0]["auroc"])
    a = g[g["eval"] == "gf_env"]
    m = (a.fold == "tau3bench").values
    chk("Q1c held-out environment tau3bench, layer 26", a.y.values[m], a["think pair, layer 26"].values[m],
        [r for r in T["gf_env_by"] if r["group"] == "tau3bench" and r["features"] == "think pair, layer 26"][0]["auroc"])
    if (OUT / "step_scores.csv.gz").is_file():
        st = json.load(open(OUT / "step.json"))["rows"]
        z = pd.read_csv(OUT / "step_scores.csv.gz")
        for dn in ("A", "B"):
            e = z[f"eval{dn}"].values.astype(bool)
            chk(f"Q2 definition {dn}, linear layers 26+38+51, step AUROC", z[f"y{dn}"].values[e], z[f"{dn}|linear all"].values[e],
                [r for r in st if r["definition"] == dn and r["features"] == "linear probe, think pair, layers 26+38+51"][0]["step_auroc"])
        if "attention" in z:
            e = z.evalB.values.astype(bool)
            chk("Q2 definition B, attention monitor, step AUROC", z.yB.values[e], z.attention.values[e],
                [r for r in st if r["definition"] == "B" and r["features"].startswith("attention")][0]["step_auroc"])
            mx = z.groupby(["run", "agent"]).attention.max()
            yy = z.groupby(["run", "agent"]).yB.max()
            chk("Q2 definition B, attention monitor, rollout AUROC (max)", yy.values.astype(float), mx.values,
                [r for r in st if r["definition"] == "B" and r["features"].startswith("attention")][0]["rollout_auroc_max"])
    ts = pd.read_csv(OUT / "transfer_scores.csv.gz", low_memory=False)
    tr = pd.read_csv(OUT / "transfer.csv")
    x = ts[(ts.train.str.startswith("controls solo (")) & (ts.features == "think pair, layers 26+38+51") & (ts.set == GF)]
    lab = load_labels(GF).set_index(["run", "agent"])
    yj = np.where(lab.loc[list(zip(x.run, x.agent))].label_kind.values == "judge", x.y.values, np.nan)
    chk("Q3 controls solo -> Goodfire judge sets", yj, x.score.values,
        tr[(tr.train.str.startswith("controls solo (")) & (tr.features == "think pair, layers 26+38+51") & (tr.target.str.startswith("Goodfire judge sets")) & (tr.group == "all")].auroc.iloc[0])
    x = ts[(ts.train.str.startswith("Goodfire")) & (ts.features == "think pair, layers 26+38+51") & (ts.set == "board_seed")]
    chk("Q3 Goodfire -> team workers, cheating used or above", x.y.values.astype(float), x.score.values,
        tr[(tr.train.str.startswith("Goodfire")) & (tr.features == "think pair, layers 26+38+51") & (tr.target.str.contains("used or above"))].auroc.iloc[0])
    for w_, key, ev_ in (("cb", "cheatbench_cheating or attempt vs benign", "cheatbench|cheating or attempt vs benign|5fold"),
                         ("eg", "evilgenie_hack vs legitimate", "evilgenie|hack vs legitimate|5fold")):
        c = pd.read_csv(OUT / f"traj_scores_{w_}.csv.gz")
        a = c[c["eval"] == ev_]
        chk(f"Q4 {key} 5 folds, layers 26+38+51", a.y.values, a["think pair, layers 26+38+51"].values, T[f"{key}_5fold"][0]["auroc"])
    c = pd.read_csv(OUT / "traj_scores_eg.csv.gz")
    a = c[c["eval"] == "evilgenie|hack vs legitimate|held author"]
    m = (a.fold == "grok-4.5").values
    chk("Q4 EvilGenie held-out author grok-4.5, layer 38", a.y.values[m], a["think pair, layer 38"].values[m],
        [r for r in T["evilgenie_hack vs legitimate_held_author_by"] if r["group"] == "grok-4.5" and r["features"] == "think pair, layer 38"][0]["auroc"])
    # folds
    gfl = load_labels(GF)
    f = json.load(gzip.open(OUT / "folds_goodfire.json.gz", "rt"))["rid_folds"]
    rid = np.array([f"{a_}/{b_}" for a_, b_ in zip(gfl.run, gfl.agent)])
    lab_rows = np.flatnonzero((gfl.label_kind == "judge").values & gfl.y.notna().values)
    rep["folds"]["goodfire_5fold_rollouts_with_a_fold"] = int(sum(r in f for r in rid[lab_rows]))
    rep["folds"]["goodfire_5fold_labelled_rollouts"] = int(len(lab_rows))
    rep["folds"]["goodfire_5fold_unlabelled_with_a_fold"] = int(sum(r in f for r in np.delete(rid, lab_rows)))
    fl = pd.Series([f[r] for r in rid[lab_rows]])
    rep["folds"]["goodfire_5fold_sizes"] = fl.value_counts().sort_index().tolist()
    rep["folds"]["goodfire_5fold_yes_rate_per_fold"] = [round(float(gfl.y.values[lab_rows][fl.values == k].mean()), 4) for k in range(5)]
    rep["folds"]["goodfire_5fold_run_dirs_per_fold"] = [len(set(gfl.run.values[lab_rows][fl.values == k])) for k in range(5)]
    a = g[g["eval"] == "gf_5fold_task"]
    task = (gfl.env + "|" + gfl.task.astype(str)).values
    tf_ = pd.Series(a.fold.values, index=rid)
    ok = np.isfinite(a.y.values)
    rep["folds"]["goodfire_task_folds_tasks_in_more_than_one_fold"] = int((pd.DataFrame({"t": task[ok], "f": tf_.values[ok]}).groupby("t").f.nunique() > 1).sum())
    rep["folds"]["goodfire_task_folds_tasks"] = int(len(set(task[ok])))
    # held-out author / environment splits rebuilt: the test group never in training
    y = np.where((gfl.label_kind == "judge").values, gfl.y.values.astype(float), np.nan)
    for nm, sp in (("author", group_splits(y, gfl.author.values)),):
        rep["folds"][f"goodfire_held_{nm}"] = {n: {"train": int(len(trn)), "test": int(len(te)),
                                                   "test_group_in_train": int((gfl.author.values[trn] == n).sum())} for n, trn, te in sp}
    env = gfl.env.values
    rep["folds"]["goodfire_held_env"] = {e: {"train": int((np.isfinite(y) & (env != e)).sum()),
                                             "train_rows_of_test_env": int((np.isfinite(y) & (env != e) & (env == e)).sum()),
                                             "train_monitor_rows": int((np.isfinite(y) & (env != e) & (gfl.label_kind.values == "monitor")).sum())}
                                         for e in JUDGE_ENVS + MONITOR_ENVS}
    eg = load_labels(EG)
    c = pd.read_csv(OUT / "traj_scores_eg.csv.gz")
    a = c[c["eval"] == "evilgenie|hack vs legitimate|5fold"].reset_index(drop=True)
    rep["folds"]["evilgenie_pairs_split_across_folds"] = int((pd.DataFrame({"p": eg.pair.values, "f": a.fold.values}).groupby("p").f.nunique() > 1).sum())
    cb = load_labels(CB)
    c = pd.read_csv(OUT / "traj_scores_cb.csv.gz")
    a = c[c["eval"] == "cheatbench|cheating or attempt vs benign|5fold"].reset_index(drop=True)
    rep["folds"]["cheatbench_fold_by_harness"] = {str(k): v for k, v in pd.crosstab(cb.run, a.fold).to_dict().items()}
    import hashlib
    tk = []
    for r in cb.itertuples():
        t_ = open(ACTS / CB / r.run / r.agent / "rendered.txt", encoding="utf-8").read()
        u = t_.split("<|im_start|>user\n", 1)[1][:1500] if "<|im_start|>user\n" in t_ else t_[:1500]
        tk.append(hashlib.sha1(u.encode()).hexdigest()[:12])
    ok_ = (cb.n_turns > 0).values
    tdf = pd.DataFrame({"task": np.array(tk)[ok_], "y": cb.y.values[ok_]})
    rep["folds"]["cheatbench_tasks_by_first_prompt"] = {"rollouts": int(ok_.sum()), "tasks": int(tdf.task.nunique()),
                                                        "tasks_with_more_than_one_rollout": int((tdf.groupby("task").size() > 1).sum())}
    if (OUT / "step_scores.csv.gz").is_file():
        z = pd.read_csv(OUT / "step_scores.csv.gz")
        rep["folds"]["step_rollouts_in_more_than_one_fold"] = int((z.groupby(["run", "agent"]).fold.nunique() > 1).sum())
        zf = z.groupby(["run", "agent"]).fold.first()
        rep["folds"]["step_fold_equals_trajectory_fold"] = int(sum(f[f"{r}/{a_}"] == v for (r, a_), v in zf.items()))
        rep["folds"]["step_rollouts"] = int(len(zf))
    # label join, five Goodfire rollouts read back from the source files
    rng = np.random.default_rng(11)
    picks = []
    for rd, want in (("deepswe-glm-5.2-fp8", 1), ("impossiblebench-kimi-k3", 1), ("swebench-verified-qwen3.8-2.4t-a95b-fp8-nohack", 1),
                     ("swebench-verified-glm-5.2-nohack", 0), ("tau3bench-glm-5.2-fp8", 1)):
        cand = gfl[(gfl.run == rd) & (gfl.y == want) & ((gfl.flagged_turns > 0) if want else True) & (gfl.unmapped_passages == 0)]
        picks.append(cand.iloc[int(rng.integers(len(cand)))])
    for r in picks:
        src, kind = gf_source_rollouts(r.run)
        sr = src[r.agent]
        meta = json.load(open(ACTS / GF / r.run / r.agent / "meta.json"))
        tn, tpn = lit(meta["turn_nodes"]), lit(meta["turn_path_nodes"])
        rt = rendered_turns(ACTS / GF / r.run / r.agent / "rendered.txt")
        pf = SRC / r.run / ("labels/judge_passages.jsonl" if kind == "judge" else "generic_monitor/monitor_passages.jsonl")
        flagged = []
        for line in open(pf):
            if r.agent not in line:
                continue
            p_ = json.loads(line)
            if p_["trace_id"] != r.agent:
                continue
            pos = p_.get("consensus") == "positive" if kind == "judge" else p_.get("evidence") == "certain"
            if pos:
                q = (p_.get("carrier_quotes") or [None])[0] if kind == "judge" else p_.get("quote")
                t = tn.index(p_["node_index"]) if p_["node_index"] in tn else tpn.index(p_["node_index"])
                think, after = rt[t]
                norm = lambda x: " ".join(str(x).split())  # noqa: E731
                where = ("reasoning" if q and norm(q) in norm(think) else "reply or tool call" if q and norm(q) in norm(after) else
                         "elsewhere in the rendered text" if q and norm(q) in norm(open(ACTS / GF / r.run / r.agent / "rendered.txt").read()) else "not found")
                flagged.append({"node_index": p_["node_index"], "step": p_["step"], "channel": p_["channel"], "turn": t,
                                "quote": (q or "")[:160], "quote_found_in_turn": where})
        rep["join"].append({"run": r.run, "rollout": r.agent, "label_kind": kind, "source_row_index": sr.get("row_index"),
                            "meta_row_index": meta.get("row_index"), "source_verdict": sr.get("verdict"), "meta_verdict": meta.get("verdict"),
                            "our_y": r.y, "source_task": sr.get("task"), "meta_task": meta.get("task"), "n_turns": int(r.n_turns),
                            "our_first_flagged_turn": r.first_flagged_turn, "our_flagged_turns": int(r.flagged_turns),
                            "flagged_passages": flagged, "rendered_turns": len(rt)})
    # local think tensors vs the Hub originals (one rollout per set)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import tempfile
    for s_, df_ in ((GF, gfl), (CB, cb), (EG, eg)):
        r = df_[df_.n_turns > 0].iloc[int(rng.integers((df_.n_turns > 0).sum()))]
        with tempfile.TemporaryDirectory() as td:
            pth = hf_hub_download(tp.REPO_ID, f"activations/{s_}/{r.run}/{r.agent}/acts.safetensors", repo_type="dataset", local_dir=td)
            z = np.load(tp.think_path(s_, r.run, r.agent))
            with safe_open(pth, "np") as fh:
                eq = {f"L{L}": bool(np.array_equal(fh.get_tensor(f"think_L{L}"), z[f"think_L{L}"])) for L in LAYERS}
                shp = list(fh.get_tensor("think_L26").shape)
        M = load_means(s_)
        i = int(np.flatnonzero((df_.run == r.run).values & (df_.agent == r.agent).values)[0])
        mean_ok = bool(np.allclose(M[i, 1], z["think_L38"].astype(np.float32).mean(0)))
        rep["tensors"].append({"set": s_, "run": r.run, "rollout": r.agent, "shape": shp, "equal_to_hub": eq, "cached_mean_L38_ok": mean_ok})
    json.dump(rep, open(OUT / "verification.json", "w"), indent=1, default=str)
    log(json.dumps(rep, indent=1, default=str)[:6000])
    return rep


def write_verification():
    V = json.load(open(OUT / "verification.json"))
    chk = json.load(open(OUT / "labels" / "checks.json"))
    L = ["# Verification of the external-set probes\n",
         f"Written by `python -m hermeneutics.external_probes verify` then `report` on {time.strftime('%Y-%m-%d')}. Every check "
         "below reads a saved output or a source file again and recomputes it a second way.\n",
         "## One AUROC per table, recomputed with sklearn\n",
         "Each row reloads the saved per-rollout (or per-turn) scores and labels and recomputes the AUROC with "
         "`sklearn.metrics.roc_auc_score`, an implementation separate from team_probes' `auc`. The within-run-dir row "
         "counts every positive-negative pair inside each run dir by brute force.\n",
         md(["Table", "Rows", "Positive", "Recomputed", "Reported", "Equal (1e-9)"],
            [[a["table"], a.get("rows", ""), a.get("pos", ""), f"{float(a['sklearn']):.6f}", f"{float(a['reported']):.6f}", a["match"]]
             for a in V["auroc"]]),
         "\n## Folds\n"]
    F = V["folds"]
    L.append(md(["Check", "Result"], [
        ["Goodfire five folds: labelled judge rollouts with a fold / labelled", f"{F['goodfire_5fold_rollouts_with_a_fold']} / {F['goodfire_5fold_labelled_rollouts']}"],
        ["Goodfire five folds: unlabelled rollouts given a fold", F["goodfire_5fold_unlabelled_with_a_fold"]],
        ["Goodfire five folds: rollouts per fold", F["goodfire_5fold_sizes"]],
        ["Goodfire five folds: yes rate per fold", F["goodfire_5fold_yes_rate_per_fold"]],
        ["Goodfire five folds: run dirs present in each fold", F["goodfire_5fold_run_dirs_per_fold"]],
        ["Goodfire task-grouped folds: tasks seen in more than one fold (of all tasks)", f"{F['goodfire_task_folds_tasks_in_more_than_one_fold']} (of {F['goodfire_task_folds_tasks']})"],
        ["Goodfire held-out author: training rows of the test author", {k: v["test_group_in_train"] for k, v in F["goodfire_held_author"].items()}],
        ["Goodfire held-out author: train / test rollouts", {k: f"{v['train']} / {v['test']}" for k, v in F["goodfire_held_author"].items()}],
        ["Goodfire held-out environment: training rows of the test environment; monitor-labelled training rows",
         {k: f"{v['train_rows_of_test_env']}; {v['train_monitor_rows']}" for k, v in F["goodfire_held_env"].items()}],
        ["EvilGenie: matched pairs split across folds", F["evilgenie_pairs_split_across_folds"]],
        ["CheatBench: rollouts, distinct tasks (keyed by the first 1500 characters of the task prompt), tasks with more than one rollout",
         F.get("cheatbench_tasks_by_first_prompt", "-")],
        ["CheatBench: rollouts per harness and fold (-1 = left out, zero turns)", F["cheatbench_fold_by_harness"]],
        ["Per-step: rollouts whose turns fall in more than one fold", F.get("step_rollouts_in_more_than_one_fold", "not run")],
        ["Per-step: rollouts whose fold equals the trajectory-level fold", f"{F.get('step_fold_equals_trajectory_fold', '-')} of {F.get('step_rollouts', '-')}"],
    ]))
    L.append("\n## Label join, five Goodfire rollouts read back from the source files\n")
    L.append(f"The joins for all rollouts are in labels/checks.json: meta.json verdict equal to the source's for "
             f"{chk['goodfire_meta_verdict_equal_source']} of {chk['goodfire_rollouts']} (the other 6 are unjudged in both), "
             f"source row_index equal to meta.json's for {chk['goodfire_row_index_equal_source']}. For five rollouts the "
             "verify stage reread the source judge or monitor files at the pinned revision, mapped each flagged passage's "
             "node_index to a reader turn through meta.json, and searched the passage's quote in that turn of rendered.txt "
             "(whitespace normalised). One quote was also read by eye (impossiblebench-kimi-k3/1ef98975..., turn 2: the "
             "agent names special-casing \"11\" k=3 as a hack and rejects it; the judge flags it as a considered "
             "hardcoding hack, verdict yes, verdict_enacted no).\n")
    rows = []
    for j in V["join"]:
        fl = j["flagged_passages"]
        found = Counter(p["quote_found_in_turn"] for p in fl)
        rows.append([f"{j['run']}/{j['rollout'][:12]}", j["label_kind"], f"{j['source_row_index']} / {j['meta_row_index']}",
                     f"{j['source_verdict']} / {j['meta_verdict']} / {j['our_y']}", f"{j['source_task']} / {j['meta_task']}",
                     j["n_turns"], j["rendered_turns"], len(fl), j["our_flagged_turns"], j["our_first_flagged_turn"],
                     ", ".join(f"{k}: {v}" for k, v in found.items()) or "-"])
    L.append(md(["Rollout", "Label file", "Row index source / meta", "Verdict source / meta / y", "Task source / meta",
                 "Reader turns", "Rendered turns", "Flagged passages", "Flagged turns (ours)", "First flagged turn (ours)",
                 "Quote found in the mapped turn"], rows))
    L.append("\nThe one quote not found is a tau3-bench tool call: the monitor quotes tool calls in its compact JSON form, "
             "which the chat template renders differently; the reasoning quote of the same passage step is found in the "
             "same turn. Flagged passages in detail:\n")
    for j in V["join"]:
        for p_ in j["flagged_passages"][:4]:
            L.append(f"- {j['run']}/{j['rollout'][:12]} node {p_['node_index']} (step {p_['step']}, {p_['channel']}) -> turn "
                     f"{p_['turn']}: \"{p_['quote'][:120]}\" ({p_['quote_found_in_turn']})")
    L.append("\n## Think tensors against the Hub originals\n")
    L.append(md(["Set", "Rollout", "Shape", "Equal to the Hub acts.safetensors (L26, L38, L51)", "Cached mean at L38 equal"],
                [[t["set"], f"{t['run']}/{t['rollout'][:14]}", t["shape"], t["equal_to_hub"], t["cached_mean_L38_ok"]] for t in V["tensors"]]))
    L.append("\nAlso checked: the per-rollout means built here equal team_probes.agent_matrix of the same tensors exactly "
             "(max absolute difference 0.0 on three CheatBench rollouts, every configuration); the rendered turn count "
             "equals the think-pair count for every rollout of the three sets; controls-solo-to-CheatBench recomputed "
             "here gives the same 0.279 as team_probes' RESULTS.md.\n")
    L.append("## Independent verifier agent (2026-11-09)\n")
    L.append("A separate verifier agent reread the code and every output without writing in the repository. It found "
             "every one of the 2,133 table cells equal to its json or csv source; recomputed with its own code the "
             "Goodfire five-fold pooled and within-run-dir AUROC at layer 26, three held-out-author AUROCs, the layer-38 "
             "held-out-environment pooled AUROC, three per-step rows (step, rollout and delay figures), three transfer "
             "rows, the CheatBench within-harness and the EvilGenie held-out-author within AUROC, all equal to 1e-9; "
             "rebuilt the turn_status of all 5,553 Goodfire rollouts from the source passages with no mismatch; rebuilt "
             "the folds with its own call and found them identical and leak-free; and re-derived the question-0 pair "
             "split. It flagged four sentences of the prose that the tables did not support (the EvilGenie problem "
             "confound, two claims about the Goodfire-trained probe on the controls and the monitor environments, and a "
             "null claimed within run dir where none was computed); those sentences were rewritten, and the EvilGenie "
             "problem became a metadata column, a stratum and a held-out split.\n")
    L.append("**UNVERIFIED**: the bootstrap intervals and permutation nulls were checked only as copied, not recomputed "
             "independently; the attention monitor was not retrained by a second party.\n")
    open(OUT / "VERIFICATION.md", "w").write("\n".join(L) + "\n")
    log(f"wrote {OUT / 'VERIFICATION.md'}")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["labels", "cache", "traj", "step", "transfer", "transfer_text", "diag", "report", "verify"])
    ap.add_argument("--which", default="cb,eg,gf")
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--no-null", action="store_true")
    a = ap.parse_args()
    if a.stage == "labels":
        stage_labels()
    elif a.stage == "cache":
        stage_cache(tuple({"cb": CB, "eg": EG, "gf": GF}[w] for w in a.which.split(",")))
    elif a.stage == "traj":
        stage_traj(tuple(a.which.split(",")))
    elif a.stage == "step":
        stage_step(attention=not a.no_attention, null=not a.no_null)
    elif a.stage == "transfer":
        stage_transfer()
    elif a.stage == "transfer_text":
        stage_transfer_text()
    elif a.stage == "verify":
        stage_verify()
    elif a.stage == "diag":
        stage_diag()
    elif a.stage == "report":
        write_report()
        if (OUT / "verification.json").is_file():
            write_verification()


if __name__ == "__main__":
    main()
