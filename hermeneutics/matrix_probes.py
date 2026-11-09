"""Probe matrix: every probe type on every way of including or excluding the labelled sources.

Probe types. A: one probe per author (folds grouped by run within that author). B.I: one probe on every author of the
scheme (grouped folds, held-out runs). B.II: leave one author out (train on the other authors, test on the held-out
one; runs holding the held-out author are dropped from training).
Inclusion schemes. Each source alone; ours (controls_600, board_seed, solo_design and the four SPAR sets); external
(goodfire_reward_hack, plus cheatbench and evilgenie when they carry a valid label); all; leave one source out; ours to
external and external to ours. For the transfer schemes, B.I trains on the training sources and scores the test source,
A trains per author on the training sources and scores that author in the test source, B.II trains on the training
sources without the test author and scores the test author.

Features (per agent, the mean over its turns): the think pair (think-open and think-close states) at layers 26, 38, 51
concatenated (main, 30,720 dims); the think pair at layer 38 alone; the seven tokens (five turn-boundary tokens plus
the think pair, all layers) where every agent of the scheme has turns.safetensors. Per step: the think pair at layer 38
of every turn.

No model calls and no spend: activations from the local copy of the public Hub dataset unrulyabstractions/apart-swarm-
align (data/acts_hub, fetched by team_probes); run dirs of the two SPAR sets that team_probes had not labelled are
fetched from the public dataset unrulyabstractions/spar-swarm into data/matrix_probes_cache.

Stages:
    python -m hermeneutics.matrix_probes labels      one table of every agent: source, run, author, labels, onsets
    python -m hermeneutics.matrix_probes features    pooled vectors and layer-38 turn rows to data/matrix_probes_cache
    python -m hermeneutics.matrix_probes tfidf       hashed word unigram and bigram counts of every rendered text
    python -m hermeneutics.matrix_probes matrix      every agent-level cell
    python -m hermeneutics.matrix_probes heatmap     source-to-source transfer
    python -m hermeneutics.matrix_probes steps       per-step probe at layer 38 on ours and all
    python -m hermeneutics.matrix_probes report      RESULTS.md
    python -m hermeneutics.matrix_probes verify      VERIFICATION.md
    python -m hermeneutics.matrix_probes all         every stage in order
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from hermeneutics import team_probes as tp  # sets the BLAS thread counts before numpy loads

import numpy as np  # noqa: E402

REPO = tp.REPO
HUB = tp.HUB
OUT = REPO / "hermeneutics" / "results" / "matrix_probes"
CACHE = REPO / "data" / "matrix_probes_cache"
STATUS = Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/"
              "62859942-35bf-474f-b418-4803b6079982/scratchpad/matrix_probes/status.md")
LAYERS = tp.LAYERS
D = 5120
OURS = ["controls_600", "board_seed", "solo_design", "spar_screen_team", "spar_mixed_41", "spar_exploration_single",
        "spar_screen_solo"]
EXTERNAL = ["goodfire_reward_hack", "cheatbench", "evilgenie"]
SOURCES = OURS + EXTERNAL
BOARD = ["board_seed", "solo_design", "spar_screen_team", "spar_mixed_41"]
SHORT = {"controls_600": "controls", "board_seed": "board_seed", "solo_design": "solo_design",
         "spar_screen_team": "spar_team", "spar_mixed_41": "spar_mixed", "spar_exploration_single": "spar_expl",
         "spar_screen_solo": "spar_solo", "goodfire_reward_hack": "goodfire", "cheatbench": "cheatbench",
         "evilgenie": "evilgenie"}
LABELS = {"cheat": "cheating", "coord": "coordination posted", "collude": "collusion crossed"}
FEATS = {"think": "think pair, layers 26+38+51", "think38": "think pair, layer 38",
         "seven": "seven tokens, layers 26+38+51"}
BASES = {"tfidf": "TF-IDF rendered text", "meta": "metadata"}
N_NULL = 20
MIN_CLASS = 3
T0 = time.time()

# Raw model ids (lower case, a leading "swarm/" removed) to one author name. Every id met is listed; an unknown id stops
# the labels stage. Short names are resolved per source, never globally: "qwen" is Qwen3.8 27B in controls_600 and
# "qwenmax" is the Goodfire Qwen run.
AUTHOR_MAP = {
    "openrouter/deepseek/deepseek-v4.1-flash-1017": "DeepSeek V4.1 Flash",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "openrouter/openai/gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "gpt-6-luna-openrouter": "GPT-6 Luna",
    "openrouter/openai/gpt-5-mini": "GPT-5 mini",
    "openrouter/anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
    "anthropic/claude-haiku-5-5": "Claude Haiku 5.5",
    "openrouter/anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
    "openrouter/anthropic/claude-opus-5.5": "Claude Opus 5.5",
    "openrouter/google/gemini-4-argon": "Gemini 4 Argon",
    "openrouter/qwen/qwen3.8-27b": "Qwen3.8 27B",
    # solo_design's OpenRouter id and Goodfire's checkpoint are both called "Qwen 3.8 Max" by their sources
    # (Goodfire meta author_name "Qwen 3.8 Max", author "qwenmax"); treated as one author
    "openrouter/qwen/qwen3.8-max-0902": "Qwen3.8 Max",
    "qwen/qwen3.8-2.4t-a95b-fp8": "Qwen3.8 Max",
    "zai-org/glm-5.2-fp8": "GLM 5.2",
    "zai-org/glm-5.2": "GLM 5.2",
    "z-ai/glm-5.2": "GLM 5.2",
    "moonshotai/kimi-k3": "Kimi K3",
    # cheatbench and evilgenie (used only if a valid label exists)
    "claude-opus-4-6": "Claude Opus 4.6", "claude-3-7-sonnet": "Claude 3.7 Sonnet", "gpt-5.3-codex": "GPT-5.3 Codex",
    "gpt-5.4": "GPT-5.4", "gemini-3.5-flash": "Gemini 3.5 Flash", "grok-4.5": "Grok 4.5",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-sol": "GPT-5.6 Sol",
    "kimi-k2.7-code": "Kimi K2.7 Code", "claude-fable-5": "Claude Fable 5", "kimi-k3": "Kimi K3",
    "claude-opus-4.8": "Claude Opus 4.8",
}
CONTROLS_SHORT = {"ds": "openrouter/deepseek/deepseek-v4.1-flash-1017", "gpt5m": "openrouter/openai/gpt-5-mini",
                  "haiku": "openrouter/anthropic/claude-haiku-5.5", "luna": "openrouter/openai/gpt-6-luna",
                  "qwen": "openrouter/qwen/qwen3.8-27b"}


def canon(raw):
    k = (raw or "").strip().lower()
    k = k[len("swarm/"):] if k.startswith("swarm/") else k
    if k not in AUTHOR_MAP:
        raise KeyError(f"unknown author id {raw!r}: add it to AUTHOR_MAP")
    return AUTHOR_MAP[k]


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:.0f}s]", *a, flush=True)


def status(stage, msg):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} [{stage}] {msg}\n"
    head = "# matrix_probes status\n\n"
    old = STATUS.read_text().splitlines(keepends=True)[2:] if STATUS.is_file() else []
    STATUS.write_text(head + "".join(([line] + old)[:60]))


# ------------------------------------------------------------------------------------------------ labels
def _meta(s, run, agent):
    return json.load(open(HUB / "activations" / s / run / agent / "meta.json"))


def _agents_of(s, run):
    d = HUB / "activations" / s / run
    return sorted(a.name for a in d.iterdir() if (a / "meta.json").is_file())


def _run_dir(s, sp):
    if s in ("board_seed", "solo_design"):
        return HUB / sp
    loc = HUB / "spar_signals" / sp
    if all((loc / f).is_file() for f in ("run.json", "episodes.jsonl", "signals.jsonl")):
        return loc
    return tp.fetch_run_dir(tp.SPAR_REPO, sp, CACHE / "spar_runs")


def _arm(s, m, run):
    if s == "solo_design":
        batch = "clean" if re.match(r"solo_L\dc_", run) else "batch 1"
        lv = re.match(r"solo_(L\d)", run)
        return f"{lv.group(1) if lv else '?'}|{batch}"
    return str(m.get("condition"))


def team_rows(s):
    """levels.py (through team_probes.ladder_rows) on each run dir of a team or SPAR set; one row per agent that has
    activations. Labels: cheating used or above (cheat index >= 4), coordination posted or above (>= 2), collusion
    crossed or above (>= 3); onsets are activation-turn indices (None when the step cannot be mapped)."""
    out, missing = [], []
    for run in sorted(os.listdir(HUB / "activations" / s)):
        ags = _agents_of(s, run)
        metas = {a: _meta(s, run, a) for a in ags}
        sp = metas[ags[0]]["source_path"]
        rd = _run_dir(s, sp)
        id2dir = {(metas[a].get("agent_id") or a): a for a in ags}
        ar, _ = tp.ladder_rows(rd, lambda aid, run=run: (HUB / "activations" / s / run / id2dir[aid]) if aid in id2dir else None)
        seen = set()
        for x in ar:
            if x["agent_id"] not in id2dir:
                continue
            a = id2dir[x["agent_id"]]
            m = metas[a]
            seen.add(a)
            board = s in BOARD
            ci, ui, li = int(x["cheat"]), int(x["coord"] or 0), int(x["collude"] or 0)
            out.append({"source": s, "run": run, "agent": a, "author_raw": x.get("model") or m.get("source_model"),
                        "role": x.get("role") or "worker", "arm": _arm(s, m, run), "task": m.get("task_name") or m.get("task"),
                        "n_turns": m.get("n_turns"), "n_tokens": m.get("n_tokens"),
                        "cheat_index": ci, "coord_index": ui, "collude_index": li,
                        "cheat": float(ci >= tp.USED), "coord": float(ui >= tp.POSTED) if board else np.nan,
                        "collude": float(li >= tp.CROSSED) if board else np.nan,
                        "cheat_onset": x.get("used_onset_step"), "coord_onset": x.get("first_post_step") if board else None,
                        "collude_onset": x.get("collude_onset_step") if board else None,
                        "turn_map": x.get("turn_map"), "label_source": f"viewer/levels.py on {sp}"})
        for a in sorted(set(ags) - seen):
            missing.append((s, run, a))
    return out, missing


def controls_rows():
    """controls_600: the set's own label (data/acts_hub/labels/controls_600/subsample.csv cheat, equal to data/labels.csv
    on all 600 runs: cheating from tried for software engineering, chess and creative writing, from opened otherwise)
    and its onset turn (labels.csv onset_turn, checked against the onset call's step in source.json)."""
    import pandas as pd
    sub = pd.read_csv(HUB / "labels" / "controls_600" / "subsample.csv")
    lab = pd.read_csv(REPO / "data" / "labels.csv").set_index("run")
    out, checks = [], {"onset_match": 0, "onset_mismatch": 0, "label_match": 0}
    for r in sub.itertuples():
        ags = _agents_of("controls_600", r.run)
        assert len(ags) == 1, (r.run, ags)
        a = ags[0]
        m = _meta("controls_600", r.run, a)
        L = lab.loc[r.run]
        checks["label_match"] += int(float(L.cheat) == float(r.cheat))
        onset = int(L.onset_turn) if pd.notna(L.onset_turn) else None
        if onset is not None and pd.notna(L.onset_call):
            calls, _ = tp.agent_calls(HUB / "activations" / "controls_600" / r.run / a / "source.json")
            st = calls[int(L.onset_call)][0] if int(L.onset_call) < len(calls) else None
            checks["onset_match" if st == onset else "onset_mismatch"] += 1
        out.append({"source": "controls_600", "run": r.run, "agent": a, "author_raw": L.model_id, "author_short": r.author,
                    "role": "worker", "arm": r.condition, "task": r.task, "n_turns": m.get("n_turns"),
                    "n_tokens": m.get("n_tokens"), "cheat_index": tp.RUNGS.index(r.rung) if r.rung in tp.RUNGS else np.nan,
                    "cheat": float(r.cheat), "cheat_onset": onset if r.cheat == 1 else None,
                    "turn_map": "step", "label_source": "data/labels.csv cheat (subsample.csv)"})
    assert all(CONTROLS_SHORT[x["author_short"]] == x["author_raw"] for x in out)
    return out, checks


def goodfire_rows():
    """goodfire_reward_hack: verdict yes / no of labels/judge_rollouts.jsonl (joined on trace_id = rollout id), split
    dropped; groups without judge labels (tau3bench, shoppingbench: generic monitor only) carry no label. Onset turn:
    the first turn whose node is at or after the first positive passage of labels/judge_passages.jsonl."""
    src = REPO / "data" / "goodfire_source_labels"
    out, checks = [], defaultdict(int)
    for g in sorted(os.listdir(HUB / "activations" / "goodfire_reward_hack")):
        jr = src / g / "labels" / "judge_rollouts.jsonl"
        verdict, first_pos = {}, {}
        if jr.is_file():
            for x in map(json.loads, open(jr)):
                verdict[x["trace_id"]] = x.get("verdict")
            for x in map(json.loads, open(src / g / "labels" / "judge_passages.jsonl")):
                if x.get("consensus") == "positive":
                    t = x["trace_id"]
                    first_pos[t] = min(first_pos.get(t, 10**9), int(x["node_index"]))
        for a in _agents_of("goodfire_reward_hack", g):
            m = _meta("goodfire_reward_hack", g, a)
            rid = m.get("rollout_id") or a
            v = verdict.get(rid) if jr.is_file() else None
            if jr.is_file():
                checks["meta_verdict_match" if v == m.get("verdict") else "meta_verdict_mismatch"] += 1
            y = 1.0 if v == "yes" else 0.0 if v == "no" else np.nan
            onset = None
            tn = m.get("turn_nodes") or []
            if isinstance(tn, str):
                tn = json.loads(tn)
            if y == 1.0 and rid in first_pos:
                n0 = first_pos[rid]
                hit = [t for t, nd in enumerate(tn) if nd >= n0]
                if hit and hit[0] < (m.get("n_turns") or 0):
                    onset = hit[0]
                    checks["onset_exact_node" if tn[hit[0]] == n0 else "onset_next_node"] += 1
                else:
                    checks["onset_after_last_turn"] += 1
            elif y == 1.0:
                checks["yes_without_positive_passage"] += 1
            checks[f"verdict_{v}"] += 1
            out.append({"source": "goodfire_reward_hack", "run": g, "agent": a, "author_raw": m.get("author_model"),
                        "author_short": m.get("author"), "role": "worker", "arm": m.get("benchmark"),
                        "task": m.get("benchmark"), "n_turns": m.get("n_turns"), "n_tokens": m.get("n_tokens"),
                        "cheat": y, "cheat_onset": onset, "verdict": v, "turn_map": "turn_nodes",
                        "label_source": (f"Goodfire labels/judge_rollouts.jsonl verdict ({g})" if jr.is_file()
                                         else f"no judge label ({g}: generic monitor only)")})
    return out, dict(checks)


EXT_LABELS = REPO / "hermeneutics" / "results" / "external_probes" / "labels"


def external_label_file():
    """The CheatBench and EvilGenie label tables of the external_probes agent (labels/cheatbench.csv: meta.json label,
    cheating or attempt = 1, benign = 0; labels/evilgenie.csv: sel_label hack = 1, legitimate = 0, with the matched
    pair id), when both exist; read, never written."""
    f = [EXT_LABELS / "cheatbench.csv", EXT_LABELS / "evilgenie.csv"]
    return f if all(x.is_file() for x in f) else None


def other_external_rows(s):
    import pandas as pd
    ext = external_label_file()
    lab = None
    if ext:
        lab = pd.read_csv(ext[0] if s == "cheatbench" else ext[1]).set_index(["run", "agent"])
    out = []
    for run in sorted(os.listdir(HUB / "activations" / s)):
        for a in _agents_of(s, run):
            m = _meta(s, run, a)
            r = lab.loc[(run, a)] if lab is not None and (run, a) in lab.index else None
            y = float(r.y) if r is not None and pd.notna(r.y) else np.nan
            onset = (r.onset_turn if s == "cheatbench" and r is not None and y == 1 and pd.notna(r.onset_turn) else None)
            out.append({"source": s, "run": run, "agent": a, "author_raw": m.get("author"), "role": "worker",
                        "arm": m.get("harness") if s == "cheatbench" else m.get("sel_condition"),
                        "task": m.get("benchmark") if s == "cheatbench" else m.get("sel_problem_set"),
                        "n_turns": m.get("n_turns"), "n_tokens": m.get("n_tokens"), "cheat": y, "cheat_onset": onset,
                        "pair": (r.pair if s == "evilgenie" and r is not None else None),
                        "turn_map": "", "label_source": (f"external_probes labels/{s}.csv y" if r is not None
                                                         else "no label")})
    return out


def labels():
    import pandas as pd
    rows, notes = [], {}
    c, notes["controls_600"] = controls_rows()
    rows += c
    for s in ["board_seed", "solo_design", "spar_screen_team", "spar_mixed_41", "spar_exploration_single", "spar_screen_solo"]:
        r, miss = team_rows(s)
        rows += r
        notes[s] = {"agents": len(r), "agents_with_activations_not_in_levels": miss}
        log(f"labels {s}: {len(r)} agents, {len(miss)} not in levels.py output")
    g, notes["goodfire_reward_hack"] = goodfire_rows()
    rows += g
    ext = external_label_file()
    notes["external_label_file"] = str(ext) if ext else None
    for s in ("cheatbench", "evilgenie"):
        rows += other_external_rows(s)
    A = pd.DataFrame(rows)
    A["author"] = [canon(x) for x in A.author_raw]
    A["uid"] = A.source + "/" + A.run + "/" + A.agent
    ext_src = A.source.isin(EXTERNAL)
    # a Goodfire or CheatBench rollout is its own run; an EvilGenie matched pair (one hack, one legitimate) is one group
    A["group"] = np.where(ext_src, A.uid, A.source + "/" + A.run)
    eg = (A.source == "evilgenie") & A["pair"].notna() if "pair" in A else np.zeros(len(A), bool)
    A.loc[eg, "group"] = "evilgenie/pair/" + A.loc[eg, "pair"].astype(str)
    A["has_turns"] = [int(tp.boundary_path(s, r, a).is_file()) for s, r, a in zip(A.source, A.run, A.agent)]
    for col in ("cheat_onset", "coord_onset", "collude_onset"):
        A[col] = pd.to_numeric(A[col], errors="coerce")
    assert A.uid.is_unique
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    A.to_csv(OUT / "labels" / "agents.csv", index=False)
    json.dump(notes, open(OUT / "labels" / "label_notes.json", "w"), indent=1, default=str)
    log(f"labels: {len(A)} agents; per source:\n" + A.groupby("source").agg(
        n=("uid", "size"), cheat_lab=("cheat", lambda v: int(v.notna().sum())), cheat_pos=("cheat", "sum"),
        coord_pos=("coord", "sum"), collude_pos=("collude", "sum"), authors=("author", "nunique")).to_string())
    log(f"notes: {json.dumps({k: v for k, v in notes.items() if k != 'board_seed'}, default=str)[:2000]}")
    return A


def load_agents():
    import pandas as pd
    return pd.read_csv(OUT / "labels" / "agents.csv", low_memory=False)


# ------------------------------------------------------------------------------------------------ features
def step_eligible(A):
    return (A.source.isin(OURS) | (A.source.isin(EXTERNAL) & A.cheat.notna())).values


def features():
    """One agent at a time: the mean over turns of the think pair at every layer (fp16, [N, 30720], columns L26 open,
    L26 close, L38 open, L38 close, L51 open, L51 close, the team_probes agent_matrix order); for agents with
    turns.safetensors the mean of the five boundary tokens at every layer ([Nb, 76800]); for the per-step sources the
    layer-38 think pair of every turn ([turns, 10240])."""
    import pandas as pd
    A = load_agents()
    N = len(A)
    CACHE.mkdir(parents=True, exist_ok=True)
    nt = A.n_turns.fillna(0).astype(int).values
    elig = step_eligible(A)
    hb = A.has_turns.values.astype(bool)
    bi = np.full(N, -1)
    bi[hb] = np.arange(hb.sum())
    s0 = np.full(N, -1)
    s1 = np.full(N, -1)
    o = 0
    for i in np.flatnonzero(elig):
        s0[i], s1[i] = o, o + nt[i]
        o += int(nt[i])
    lib = np.lib.format
    T = lib.open_memmap(CACHE / "think_pooled.tmp.npy", mode="w+", dtype=np.float16, shape=(N, 6 * D))
    B = lib.open_memmap(CACHE / "boundary_pooled.tmp.npy", mode="w+", dtype=np.float16, shape=(int(hb.sum()), 15 * D))
    S = lib.open_memmap(CACHE / "step_L38.tmp.npy", mode="w+", dtype=np.float16, shape=(int(o), 2 * D))
    ok = np.zeros(N, int)
    bad = []
    for i, r in enumerate(A.itertuples()):
        p = tp.think_path(r.source, r.run, r.agent)
        if not p.is_file():
            bad.append((r.uid, "no think file"))
            continue
        z = np.load(p)
        n = z["think_L26"].shape[0]
        if n != nt[i]:
            bad.append((r.uid, f"think turns {n} != meta n_turns {nt[i]}"))
            continue
        if n == 0:
            bad.append((r.uid, "zero turns"))
            continue
        T[i] = np.concatenate([z[f"think_L{L}"][:, :2, :].astype(np.float32).mean(0).reshape(-1) for L in LAYERS]).astype(np.float16)
        if elig[i]:
            S[s0[i]:s1[i]] = z["think_L38"][:, :2, :].reshape(n, 2 * D)
        if hb[i]:
            zb = np.load(tp.boundary_path(r.source, r.run, r.agent))
            if zb["turn_L26"].shape[:2] != (n, 5):
                bad.append((r.uid, f"boundary shape {zb['turn_L26'].shape}"))
                continue
            B[bi[i]] = np.concatenate([zb[f"turn_L{L}"].astype(np.float32).mean(0).reshape(-1) for L in LAYERS]).astype(np.float16)
        ok[i] = 1
        if i % 500 == 0:
            log(f"features {i}/{N} ({r.source})")
            status("features", f"{i}/{N} agents pooled")
    T.flush()
    B.flush()
    S.flush()
    del T, B, S
    for nm in ("think_pooled", "boundary_pooled", "step_L38"):  # a process that has the old file mapped keeps its inode
        os.replace(CACHE / f"{nm}.tmp.npy", CACHE / f"{nm}.npy")
    idx = pd.DataFrame({"uid": A.uid, "fi": np.arange(N), "bi": bi, "s0": s0, "s1": s1, "ok": ok})
    idx.to_csv(CACHE / "feature_index.tmp.csv", index=False)
    os.replace(CACHE / "feature_index.tmp.csv", CACHE / "feature_index.csv")
    json.dump(bad, open(OUT / "labels" / "feature_problems.json", "w"), indent=1)
    log(f"features done: {ok.sum()} of {N} agents ok; {len(bad)} problems {bad[:5]}; step rows {o}")


def tfidf_counts():
    """Hashed counts (2^22 buckets, word unigrams and bigrams, TfidfVectorizer's tokenizer and lower-casing) of every
    agent's rendered.txt, the reader's whole input. Per fold the training rows then give the vocabulary (min_df 2, the
    50,000 most frequent), sublinear tf and the idf, as TfidfVectorizer fitted on the training fold would."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import HashingVectorizer
    A = load_agents()
    hv = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 22, alternate_sign=False, norm=None, dtype=np.float32)
    mats = []
    for i0 in range(0, len(A), 200):
        docs = [open(HUB / "activations" / r.source / r.run / r.agent / "rendered.txt", encoding="utf-8").read()
                for r in A.iloc[i0:i0 + 200].itertuples()]
        mats.append(hv.transform(docs).tocsr())
        if i0 % 1000 == 0:
            log(f"tfidf counts {i0}/{len(A)}")
            status("tfidf", f"{i0}/{len(A)} rendered texts hashed")
    H = sp.vstack(mats).tocsr()
    sp.save_npz(CACHE / "tfidf_counts.npz", H, compressed=False)
    log(f"tfidf counts done: {H.shape}, nnz {H.nnz}")


# ------------------------------------------------------------------------------------------------ data access
class Data:
    """The agents table joined to the feature index, the pooled feature arrays (memory-mapped) and the hashed counts."""

    def __init__(self, need_tfidf=True):
        import pandas as pd
        import scipy.sparse as sp
        A = load_agents()
        I = pd.read_csv(CACHE / "feature_index.csv")
        assert (I.uid.values == A.uid.values).all()
        A = A.join(I.drop(columns="uid"))
        self.A = A
        self.T = np.load(CACHE / "think_pooled.npy", mmap_mode="r")
        self.B = np.load(CACHE / "boundary_pooled.npy", mmap_mode="r")
        self.H = sp.load_npz(CACHE / "tfidf_counts.npz").tocsr() if need_tfidf else None

    def X(self, feat, idx):
        idx = np.asarray(idx)
        fi = self.A.fi.values[idx]
        if feat == "think":
            return np.asarray(self.T[fi], np.float32)
        if feat == "think38":
            return np.asarray(self.T[fi][:, 2 * D:4 * D], np.float32)
        if feat == "seven":
            bi = self.A.bi.values[idx]
            assert (bi >= 0).all()
            return np.concatenate([np.asarray(self.T[fi], np.float32), np.asarray(self.B[bi], np.float32)], 1)
        raise KeyError(feat)


# ------------------------------------------------------------------------------------------------ baselines
def tfidf_pair(Htr, Hte, max_features=50000):
    """TF-IDF weights fitted on the training rows' hashed counts (min_df 2, the max_features most frequent buckets,
    sublinear tf, smooth idf, L2 rows), applied to both."""
    from sklearn.preprocessing import normalize
    df = np.asarray((Htr > 0).sum(0)).ravel()
    tf = np.asarray(Htr.sum(0)).ravel()
    cand = np.flatnonzero(df >= 2)
    if len(cand) > max_features:
        cand = np.sort(cand[np.argsort(-tf[cand], kind="stable")[:max_features]])
    n = Htr.shape[0]
    idf = np.log((1 + n) / (1 + df[cand])) + 1

    def tr(M):
        M = M[:, cand].astype(np.float64).tocsr()
        M.data = 1 + np.log(M.data)
        return normalize(M.multiply(idf).tocsr())
    return tr(Htr), tr(Hte)


def meta_frame(A, idx):
    """Metadata known before reading any text: source, author, arm, task, role (one-hot), log turns and log tokens."""
    sub = A.iloc[idx]
    cols = [tp.onehot(sub[c].astype(str).tolist()) for c in ("source", "author", "arm", "task", "role")]
    cols += [np.log1p(sub.n_turns.fillna(0).values.astype(float))[:, None],
             np.log1p(sub.n_tokens.fillna(0).values.astype(float))[:, None]]
    return np.column_stack(cols)


def meta_fit(Ftr, ytr, Fte):
    from sklearn.linear_model import LogisticRegression
    keep = Ftr.std(0) > 0
    if not keep.any() or len(set(ytr.tolist())) < 2:
        return np.full(len(Fte), np.nan)
    mu, sd = Ftr[:, keep].mean(0), Ftr[:, keep].std(0)
    lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit((Ftr[:, keep] - mu) / sd, ytr)
    return lr.decision_function((Fte[:, keep] - mu) / sd)


def tfidf_fit(Htr, ytr, gtr, Hte):
    Xtr, Xte = tfidf_pair(Htr, Hte)
    s, C = tp.fit_score(Xtr, ytr, gtr, Xte, Cs=tp.CS_TEXT)
    return (np.full(Hte.shape[0], np.nan) if s is None else s), C


# ------------------------------------------------------------------------------------------------ probe evaluation
def perms(strata, y, seed):
    return tp.perm_within(np.asarray(strata), np.asarray(y), B=N_NULL, seed=seed)


def cv_eval(D_, idx, y, g, strata, fold, feats, seed):
    """Every feature setting and both baselines on one fold vector over the rows idx (local arrays y, g, fold).
    Null: the label vector permuted within strata (source, author, arm), the probe refitted with C fixed at the median
    of the chosen Cs, AUROC against the permuted labels (team_probes.null_aucs)."""
    n = len(idx)
    out = {}
    pl = perms(strata, y, seed)
    for f in feats:
        X = D_.X(f, idx)
        P = tp.prep_folds(X, fold)
        del X
        s, Cs = tp.cv_scores(P, y, g, n)
        C = float(np.median(Cs)) if Cs else 1e-2
        nul = tp.null_aucs(P, pl, g, n, C)
        out[f] = {"score": s, "C": C, "null": nul}
        del P
    st = np.full(n, np.nan)
    sm = np.full(n, np.nan)
    F = meta_frame(D_.A, idx)
    H = D_.H[D_.A.fi.values[idx]]
    for k in range(tp.K):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if not len(te) or len(set(y[tr].tolist())) < 2:
            continue
        st[te], _ = tfidf_fit(H[tr], y[tr], g[tr], H[te])
        sm[te] = meta_fit(F[tr], y[tr], F[te])
    out["tfidf"] = {"score": st}
    out["meta"] = {"score": sm}
    return out


def split_eval(D_, tr_idx, te_idx, ytr, gtr, strata_tr, yte, feats, seed):
    """Train on tr_idx, score te_idx. Null: the training labels permuted within strata, the probe refitted with C fixed,
    AUROC against the real test labels."""
    out = {}
    pl = perms(strata_tr, ytr, seed)
    for f in feats:
        Xtr, Xte = D_.X(f, tr_idx), D_.X(f, te_idx)
        Ztr, Zte = tp.pca_z(Xtr, Xte)
        del Xtr, Xte
        s, C = tp.fit_score(Ztr, ytr, gtr, Zte)
        nul = np.array([tp.auc(tp.fit_score(Ztr, yp, gtr, Zte, C=C)[0], yte) for yp in pl], float)
        out[f] = {"score": s, "C": C, "null": nul}
    H = D_.H
    fi = D_.A.fi.values
    out["tfidf"] = {"score": tfidf_fit(H[fi[tr_idx]], ytr, gtr, H[fi[te_idx]])[0]}
    F = meta_frame(D_.A, np.concatenate([tr_idx, te_idx]))  # one-hot columns from both sides, so they line up
    out["meta"] = {"score": meta_fit(F[:len(tr_idx)], ytr, F[len(tr_idx):])}
    return out


def boot_mean(parts, B=tp.N_BOOT, seed=0):
    """95% interval of the mean over authors of the per-author AUROC, runs resampled with replacement (one draw of the
    runs shared by every author)."""
    rng = np.random.default_rng(seed)
    allr = np.unique(np.concatenate([p[2] for p in parts]))
    pos = {r: i for i, r in enumerate(allr)}
    inv = [np.array([pos[r] for r in p[2]]) for p in parts]
    vals = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(allr), len(allr)), minlength=len(allr)).astype(float)
        a = [tp.auc(p[0], p[1], w[iv]) for p, iv in zip(parts, inv)]
        if np.isfinite(a).all():
            vals.append(np.mean(a))
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [np.nan, np.nan]


def stat_row(base, s, y, runs, authors, null=None, C=None, extra=None):
    ok = np.isfinite(s)
    s, y, runs, authors = s[ok], y[ok], np.asarray(runs)[ok], np.asarray(authors)[ok]
    a = tp.auc(s, y)
    lo, hi = tp.boot_ci(s, y, runs) if np.isfinite(a) else (np.nan, np.nan)
    r = dict(base, auroc=a, ci_lo=lo, ci_hi=hi, n=int(len(y)), pos=int(y.sum()), runs=int(len(np.unique(runs))),
             authors=int(len(np.unique(authors))), C=C)
    if null is not None:
        nl = np.asarray(null, float)
        r.update(null_mean=float(np.nanmean(nl)), null_95=float(np.nanpercentile(nl, 95)),
                 null_p=float((1 + (nl >= a).sum()) / (1 + np.isfinite(nl).sum())))
    r.update(extra or {})
    return r


def mean_rows(base, per, feats_all):
    """The mean over authors row of each feature, from per = {author: {feature: (score, y, runs, null or None)}}."""
    out = []
    for f in feats_all:
        parts = [(v[f][0], v[f][1], v[f][2]) for v in per.values() if f in v and np.isfinite(tp.auc(v[f][0], v[f][1]))]
        nulls = [v[f][3] for v in per.values() if f in v and v[f][3] is not None and np.isfinite(tp.auc(v[f][0], v[f][1]))]
        if not parts:
            continue
        aucs = [tp.auc(p[0], p[1]) for p in parts]
        lo, hi = boot_mean([(p[0][np.isfinite(p[0])], p[1][np.isfinite(p[0])], np.asarray(p[2])[np.isfinite(p[0])]) for p in parts])
        r = dict(base, feature=f, unit="mean over authors", auroc=float(np.mean(aucs)), ci_lo=lo, ci_hi=hi,
                 n=int(sum(np.isfinite(p[0]).sum() for p in parts)), pos=int(sum(p[1][np.isfinite(p[0])].sum() for p in parts)),
                 runs=int(len(np.unique(np.concatenate([np.asarray(p[2])[np.isfinite(p[0])] for p in parts])))),
                 authors=len(parts))
        if nulls and len(nulls) == len(parts):
            nm = np.nanmean(np.vstack(nulls), 0)
            r.update(null_mean=float(np.nanmean(nm)), null_95=float(np.nanpercentile(nm, 95)),
                     null_p=float((1 + (nm >= r["auroc"]).sum()) / (1 + np.isfinite(nm).sum())))
        out.append(r)
    return out


def feasible(y, g):
    return min(int(y.sum()), int((1 - y).sum())) >= MIN_CLASS and len(np.unique(g)) >= tp.K


# ------------------------------------------------------------------------------------------------ schemes
def labelled_sources(A, label):
    out = []
    for s in SOURCES:
        y = A.loc[(A.source == s) & (A.ok == 1), label].dropna()
        if len(y) and min(int(y.sum()), int((1 - y).sum())) >= MIN_CLASS:
            out.append(s)
    return out


def schemes(A, label):
    """[(name, kind, train sources, test sources or None for cross-validation within the train sources)]."""
    L = labelled_sources(A, label)
    ours = [s for s in OURS if s in L]
    ext = [s for s in EXTERNAL if s in L]
    out = [(f"alone: {SHORT[s]}", "alone", [s], None) for s in L]
    out.append(("ours", "ours", ours, None))
    out.append(("external", "external", ext, None) if ext else ("external", "external", [], None))
    out.append(("all", "all", L, None))
    out += [(f"leave out: {SHORT[s]}", "loso", [x for x in L if x != s], [s]) for s in L if len(L) > 1]
    out.append(("ours to external", "transfer", ours, ext) if ext else ("ours to external", "transfer", ours, []))
    out.append(("external to ours", "transfer", ext, ours) if ext else ("external to ours", "transfer", [], ours))
    return out


def scheme_feats(A, srcs):
    have = A[A.source.isin(srcs) & (A.ok == 1)].has_turns
    return ["think", "think38"] + (["seven"] if len(have) and bool((have == 1).all()) else [])


def run_cv_scheme(D_, label, name, kind, srcs, seed0):
    """B.I, A and B.II within the sources srcs."""
    A = D_.A
    ii = np.flatnonzero((A.source.isin(srcs) & (A.ok == 1) & A[label].notna()).values)
    y = A[label].values[ii].astype(int)
    g = A.group.values[ii]
    au = A.author.values[ii]
    src = A.source.values[ii]
    strata = np.array([f"{a}|{b}|{c}" for a, b, c in zip(src, au, A.arm.astype(str).values[ii])])
    feats = scheme_feats(A, srcs)
    base = {"label": label, "scheme": name, "kind": kind, "train": "+".join(SHORT[s] for s in srcs), "test": "held-out runs"}
    rows, frames, bii_folds = [], [], {}

    def keep_scores(ptype, unit, sub, res, fold):
        for f, v in res.items():
            frames.append({"label": label, "scheme": name, "ptype": ptype, "unit": unit, "feature": f,
                           "uid": A.uid.values[ii[sub]], "y": y[sub], "fold": fold, "score": v["score"]})

    # B.I
    if feasible(y, g):
        fold = tp.folds_for(np.array([f"{a}|{b}|{c}" for a, b, c in zip(src, au, y)]), g, seed=0)
        res = cv_eval(D_, ii, y, g, strata, fold, feats, seed0)
        for f, v in res.items():
            rows.append(stat_row(dict(base, ptype="B.I", unit="pooled", feature=f), v["score"], y, g, au, v.get("null"), v.get("C")))
        keep_scores("B.I", "pooled", np.arange(len(ii)), res, fold)
        log(f"  {label} {name} B.I: " + ", ".join(f"{f} {tp.auc(v['score'], y):.3f}" for f, v in res.items()))
    else:
        rows.append(dict(base, ptype="B.I", unit="pooled", feature="think", note="too few agents or runs in a class",
                         n=len(ii), pos=int(y.sum()), runs=len(np.unique(g)), authors=len(np.unique(au))))
    # A
    per = {}
    for a in sorted(np.unique(au)):
        sub = np.flatnonzero(au == a)
        ya, ga = y[sub], g[sub]
        if not feasible(ya, ga):
            rows.append(dict(base, ptype="A", unit=a, feature="think", note="too few agents or runs in a class",
                             n=len(sub), pos=int(ya.sum()), runs=len(np.unique(ga)), authors=1))
            continue
        try:
            fold = tp.folds_for(np.array([f"{s_}|{v}" for s_, v in zip(src[sub], ya)]), ga, seed=0)
        except ValueError as e:
            rows.append(dict(base, ptype="A", unit=a, feature="think", note=f"folds failed: {e}"[:120], n=len(sub)))
            continue
        res = cv_eval(D_, ii[sub], ya, ga, strata[sub], fold, feats, seed0 + 1)
        per[a] = {}
        for f, v in res.items():
            rows.append(stat_row(dict(base, ptype="A", unit=a, feature=f), v["score"], ya, ga, au[sub], v.get("null"), v.get("C")))
            per[a][f] = (v["score"], ya, ga, v.get("null"))
        keep_scores("A", a, sub, res, fold)
    rows += mean_rows(dict(base, ptype="A"), per, feats + list(BASES))
    # B.II
    per = {}
    for a in sorted(np.unique(au)):
        te = np.flatnonzero(au == a)
        bad_runs = set(g[te])
        tr = np.flatnonzero((au != a) & ~np.isin(g, list(bad_runs)))
        if len(np.unique(au)) < 2 or min(int(y[te].sum()), int((1 - y[te]).sum())) < MIN_CLASS or \
                min(int(y[tr].sum()), int((1 - y[tr]).sum())) < MIN_CLASS:
            why = ("one author only: nothing to hold out" if len(np.unique(au)) < 2 else
                   "held-out author or training set has too few in a class")
            rows.append(dict(base, ptype="B.II", unit=a, feature="think", note=why,
                             n=len(te), pos=int(y[te].sum()), runs=len(np.unique(g[te])), authors=1))
            continue
        res = split_eval(D_, ii[tr], ii[te], y[tr], g[tr], strata[tr], y[te], feats, seed0 + 2)
        per[a] = {}
        for f, v in res.items():
            rows.append(stat_row(dict(base, ptype="B.II", unit=a, feature=f), v["score"], y[te], g[te], au[te], v.get("null"),
                                 v.get("C"), {"n_train": len(tr), "train_authors": len(np.unique(au[tr]))}))
            per[a][f] = (v["score"], y[te], g[te], v.get("null"))
        keep_scores("B.II", a, te, res, np.full(len(te), -1))
        bii_folds[a] = {"test_uids": A.uid.values[ii[te]].tolist(), "train_runs": sorted(set(g[tr])),
                        "train_authors": sorted(set(au[tr]))}
    rows += mean_rows(dict(base, ptype="B.II"), per, feats + list(BASES))
    return rows, frames, bii_folds


def run_split_scheme(D_, label, name, kind, tr_srcs, te_srcs, seed0):
    """B.I, A and B.II from the training sources to the test sources (disjoint, so no run crosses)."""
    A = D_.A
    lab = (A.ok == 1) & A[label].notna()
    tr_all = np.flatnonzero((A.source.isin(tr_srcs) & lab).values)
    te_all = np.flatnonzero((A.source.isin(te_srcs) & lab).values)
    base = {"label": label, "scheme": name, "kind": kind, "train": "+".join(SHORT[s] for s in tr_srcs),
            "test": "+".join(SHORT[s] for s in te_srcs)}
    rows, frames, bii_folds = [], [], {}
    if not len(tr_all) or not len(te_all):
        return [dict(base, ptype=p, unit="pooled", feature="think", note="no labelled source on one side") for p in ("A", "B.I", "B.II")], [], {}
    feats = scheme_feats(A, list(tr_srcs) + list(te_srcs))
    y = A[label].values.astype(float)
    g = A.group.values
    au = A.author.values
    strata = np.array([f"{a}|{b}|{c}" for a, b, c in zip(A.source.values, au, A.arm.astype(str).values)])

    def one(ptype, unit, tr, te, seed):
        ytr, yte = y[tr].astype(int), y[te].astype(int)
        if min(ytr.sum(), (1 - ytr).sum()) < MIN_CLASS or min(yte.sum(), (1 - yte).sum()) < MIN_CLASS:
            rows.append(dict(base, ptype=ptype, unit=unit, feature="think", note="too few in a class on one side",
                             n=len(te), pos=int(yte.sum()), runs=len(np.unique(g[te])), authors=len(np.unique(au[te]))))
            return None
        res = split_eval(D_, tr, te, ytr, g[tr], strata[tr], yte, feats, seed)
        out = {}
        for f, v in res.items():
            rows.append(stat_row(dict(base, ptype=ptype, unit=unit, feature=f), v["score"], yte, g[te], au[te], v.get("null"),
                                 v.get("C"), {"n_train": len(tr), "train_authors": len(np.unique(au[tr]))}))
            frames.append({"label": label, "scheme": name, "ptype": ptype, "unit": unit, "feature": f, "uid": A.uid.values[te],
                           "y": yte, "fold": np.full(len(te), -1), "score": v["score"]})
            out[f] = (v["score"], yte, g[te], v.get("null"))
        return out
    one("B.I", "pooled", tr_all, te_all, seed0)
    log(f"  {label} {name} B.I: " + ", ".join(f"{r['feature']} {r.get('auroc', np.nan):.3f}" for r in rows if r["ptype"] == "B.I"))
    per = {}
    for a in sorted(set(au[te_all]) & set(au[tr_all])):
        r_ = one("A", a, tr_all[au[tr_all] == a], te_all[au[te_all] == a], seed0 + 1)
        if r_:
            per[a] = r_
    for a in sorted(set(au[te_all]) - set(au[tr_all])):
        rows.append(dict(base, ptype="A", unit=a, feature="think", note="author absent from the training sources",
                         n=int((au[te_all] == a).sum()), authors=1))
    rows += mean_rows(dict(base, ptype="A"), per, feats + list(BASES))
    per = {}
    for a in sorted(set(au[te_all])):
        # as within a scheme, a training run that holds any agent of the held-out author leaves training whole (its
        # other agents share a board or transcript with that author)
        a_runs = set(g[A.source.isin(tr_srcs).values & (A.ok == 1).values & (au == a)])
        tr = tr_all[(au[tr_all] != a) & ~np.isin(g[tr_all], list(a_runs))]
        te = te_all[au[te_all] == a]
        r_ = one("B.II", a, tr, te, seed0 + 2)
        if r_:
            per[a] = r_
            bii_folds[a] = {"test_uids": A.uid.values[te].tolist(), "train_runs": sorted(set(g[tr])),
                            "train_authors": sorted(set(au[tr]))}
    rows += mean_rows(dict(base, ptype="B.II"), per, feats + list(BASES))
    return rows, frames, bii_folds


# ------------------------------------------------------------------------------------------------ the matrix
def slug(x):
    return re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_")


def save_frames(frames, p):
    import pandas as pd
    if not frames:
        return
    df = pd.concat([pd.DataFrame(f) for f in frames], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, compression="gzip", float_format="%.6g")


def matrix(only=None, force=False):
    import pandas as pd
    D_ = Data()
    A = D_.A
    cells_p = OUT / "cells.csv"
    done = pd.read_csv(cells_p) if cells_p.is_file() and not force else pd.DataFrame(columns=["label", "scheme"])
    allrows = [done] if len(done) else []
    seen = {}
    plan = [(lab, sc) for lab in LABELS for sc in schemes(A, lab)]
    json.dump([{"label": l, "scheme": s[0], "kind": s[1], "train": s[2], "test": s[3]} for l, s in plan],
              open(OUT / "schemes.json", "w"), indent=1)
    for k, (label, (name, kind, tr, te)) in enumerate(plan):
        if only and only not in name:
            continue
        key = (label, tuple(sorted(tr)), None if te is None else tuple(sorted(te)))
        if key in seen:
            prev = done[(done.label == label) & (done.scheme == seen[key])].copy() if len(done) else pd.DataFrame()
            if not len(prev):
                prev = pd.concat(allrows, ignore_index=True)
                prev = prev[(prev.label == label) & (prev.scheme == seen[key])].copy()
            prev["scheme"], prev["kind"], prev["note"] = name, kind, f"identical source set to '{seen[key]}'"
            # drop only this scheme's rows (dropping every frame that holds one would drop the resumed cells.csv)
            allrows = [x[~((x.label == label) & (x.scheme == name))] for x in allrows] + [prev]
            log(f"{label} {name}: identical to {seen[key]}")
            continue
        seen[key] = name
        if len(done) and ((done.label == label) & (done.scheme == name)).any():
            continue
        t0 = time.time()
        status("matrix", f"cell {k + 1}/{len(plan)}: {LABELS[label]}, {name} (train {tr}, test {te or 'CV'})")
        log(f"== {label} {name}: train {tr} test {te}")
        if (te is None and not tr) or (te is not None and (not tr or not te)):
            rows = [dict(label=label, scheme=name, kind=kind, train="+".join(SHORT[s] for s in tr),
                         test="+".join(SHORT[s] for s in (te or [])), ptype=p, unit="pooled", feature="think",
                         note="no labelled source") for p in ("A", "B.I", "B.II")]
            frames, bii = [], {}
        elif te is None:
            rows, frames, bii = run_cv_scheme(D_, label, name, kind, tr, 1000 + 10 * k)
        else:
            rows, frames, bii = run_split_scheme(D_, label, name, kind, tr, te, 1000 + 10 * k)
        save_frames(frames, OUT / "scores" / label / f"{slug(name)}.csv.gz")
        if bii:
            p = OUT / "folds" / label / f"{slug(name)}_BII.json.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, "wt") as fh:
                json.dump(bii, fh)
        allrows.append(pd.DataFrame(rows))
        pd.concat(allrows, ignore_index=True).to_csv(cells_p, index=False)
        log(f"   done in {time.time() - t0:.0f}s")
    pd.concat(allrows, ignore_index=True).to_csv(cells_p, index=False)
    status("matrix", "all cells done")


# ------------------------------------------------------------------------------------------------ transfer heatmap
def heatmap():
    """Train on source i (every labelled agent), score source j; the diagonal is B.I within source i (the 'alone' cell)."""
    import pandas as pd
    D_ = Data(need_tfidf=False)
    A = D_.A
    cells = pd.read_csv(OUT / "cells.csv")
    out = []
    for label in LABELS:
        L = labelled_sources(A, label)
        for i in L:
            tri = np.flatnonzero(((A.source == i) & (A.ok == 1) & A[label].notna()).values)
            ytr = A[label].values[tri].astype(int)
            Xtr0 = D_.X("think", tri)
            for j in L:
                if i == j:
                    c = cells[(cells.label == label) & (cells.scheme == f"alone: {SHORT[i]}") & (cells.ptype == "B.I") &
                              (cells.feature == "think")]
                    r = c.iloc[0].to_dict() if len(c) else {}
                    out.append({"label": label, "train": SHORT[i], "test": SHORT[j], "kind": "B.I within source (CV)",
                                **{k: r.get(k) for k in ("auroc", "ci_lo", "ci_hi", "n", "pos", "runs")}})
                    continue
                tej = np.flatnonzero(((A.source == j) & (A.ok == 1) & A[label].notna()).values)
                yte = A[label].values[tej].astype(int)
                Ztr, Zte = tp.pca_z(Xtr0.copy(), D_.X("think", tej))
                s, C = tp.fit_score(Ztr, ytr, A.group.values[tri], Zte)
                lo, hi = tp.boot_ci(s, yte, A.group.values[tej])
                out.append({"label": label, "train": SHORT[i], "test": SHORT[j], "kind": "transfer", "auroc": tp.auc(s, yte),
                            "ci_lo": lo, "ci_hi": hi, "n": len(tej), "pos": int(yte.sum()),
                            "runs": len(np.unique(A.group.values[tej])), "C": C})
                log(f"heatmap {label} {SHORT[i]} -> {SHORT[j]}: {out[-1]['auroc']:.3f}")
            status("heatmap", f"{LABELS[label]}: rows of {SHORT[i]} done")
    H = pd.DataFrame(out)
    H.to_csv(OUT / "heatmap.csv", index=False)
    plot_heatmap(H)


def plot_heatmap(H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    for label, fn in (("cheat", "heatmap_cheating.png"), (None, "heatmap_board.png")):
        labs = [label] if label else ["coord", "collude"]
        fig, axes = plt.subplots(1, len(labs), figsize=(7.6 * len(labs) if label else 12.5, 6.6 if label else 5.2), squeeze=False,
                                 gridspec_kw={"wspace": 0.45})
        for ax, lab in zip(axes[0], labs):
            h = H[H.label == lab]
            src = list(dict.fromkeys(h.train))
            M = np.full((len(src), len(src)), np.nan)
            for r in h.itertuples():
                M[src.index(r.train), src.index(r.test)] = r.auroc
            im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0))
            for a in range(len(src)):
                for b in range(len(src)):
                    if np.isfinite(M[a, b]):
                        ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=9,
                                color="white" if abs(M[a, b] - 0.5) > 0.32 else "#222222",
                                fontweight="bold" if a == b else "normal")
            ax.set_xticks(range(len(src)), src, rotation=40, ha="right")
            ax.set_yticks(range(len(src)), src)
            ax.set_xlabel("test source")
            ax.set_ylabel("train source")
            ax.set_title(f"{LABELS[lab]}: held-out AUROC", fontsize=11)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="AUROC (diagonal: B.I within source)")
        fig.savefig(OUT / fn, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------------------------------------ per step
def step_pca(S, R, fold, k=128, chunk=8192, which=None):
    """Exact training-fold PCA of the turn rows S[R] without holding them: per fold the training covariance is summed
    over row chunks (shifted by the global mean for precision), its top-k eigenvectors give the projection, and the
    scores are z-scored with training statistics, as team_probes.pca_z does. Returns prep_folds-style tuples."""
    from sklearn.utils.extmath import randomized_svd
    d = S.shape[1]
    nR = len(R)
    chunks = [np.arange(i, min(i + chunk, nR)) for i in range(0, nR, chunk)]
    m0 = np.zeros(d)
    for c in chunks:
        m0 += np.asarray(S[R[c]], np.float64).sum(0)
    m0 /= nR
    m0f = m0.astype(np.float32)
    which = list(range(tp.K)) if which is None else list(which)
    proj = []
    for f in which:
        trm = fold != f
        G = np.zeros((d, d))
        s = np.zeros(d)
        n = 0
        for c in chunks:
            cc = c[trm[c]]
            if not len(cc):
                continue
            Xb = np.asarray(S[R[cc]], np.float32) - m0f
            G += Xb.T @ Xb
            s += Xb.sum(0, dtype=np.float64)
            n += len(cc)
            del Xb
        mu_d = s / n
        G /= n
        G -= np.outer(mu_d, mu_d)
        # top-k eigenvectors of the symmetric covariance by randomized SVD (the method team_probes.pca_z uses above
        # 1,500 rows), with more power iterations since the matrix is cheap to multiply
        U, w, _ = randomized_svd(G, k, n_oversamples=32, n_iter=8, random_state=0)
        del G
        proj.append(((m0 + mu_d).astype(np.float32), U.astype(np.float32)))
        log(f"   step PCA fold {f}: {n} training turns, top eigenvalue {w[0]:.3g}")
    Pf = [np.zeros((nR, k), np.float32) for _ in which]
    for c in chunks:
        Xb = np.asarray(S[R[c]], np.float32)
        for j, (mu, V) in enumerate(proj):
            Pf[j][c] = (Xb - mu) @ V
    out = []
    for j, f in enumerate(which):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        Ptr = Pf[j][tr].astype(np.float64)
        sd = Ptr.std(0) + 1e-9
        out.append((tr, te, Ptr / sd, Pf[j][te].astype(np.float64) / sd))
    return out


def run_steps(D_, S, label, name, srcs, seed=3000):
    import pandas as pd
    A = D_.A
    on = A[f"{label}_onset"].values.astype(float)
    nt = A.n_turns.fillna(0).values.astype(int)
    reach = A[label].values
    elig = (A.source.isin(srcs) & (A.ok == 1) & A[label].notna() & (A.s0 >= 0)).values
    drop = elig & (reach == 1) & (~np.isfinite(on) | (on >= nt))
    ii = np.flatnonzero(elig & ~drop)
    n_t = nt[ii]
    reach_i = reach[ii].astype(int)
    onset_i = on[ii]
    ai = np.repeat(np.arange(len(ii)), n_t)
    tpos = np.concatenate([np.arange(n) for n in n_t])
    frac = tpos / np.maximum(n_t[ai] - 1, 1)
    R = np.concatenate([np.arange(A.s0.values[i], A.s1.values[i]) for i in ii])
    assert len(R) == len(ai)
    y = ((reach_i[ai] == 1) & (tpos >= np.where(np.isfinite(onset_i), onset_i, 1e9)[ai])).astype(int)
    g_a = A.group.values[ii]
    src_a = A.source.values[ii]
    au_a = A.author.values[ii]
    strata_a = np.array([f"{a}|{b}|{c}" for a, b, c in zip(src_a, au_a, A.arm.astype(str).values[ii])])
    fold_a = tp.folds_for(np.array([f"{a}|{b}|{c}" for a, b, c in zip(src_a, au_a, reach_i)]), g_a, seed=0)
    fold = fold_a[ai]
    runs = g_a[ai]
    rk = (reach_i == 1) & np.isfinite(onset_i)
    log(f"== steps {label} {name}: {len(ii)} agents ({int(drop.sum())} dropped: positive with no usable onset), "
        f"{len(R)} turns, {int(y.sum())} positive turns")
    status("steps", f"{LABELS[label]} {name}: PCA over {len(R)} turns")
    P = step_pca(S, R, fold)
    s, Cs = tp.cv_scores(P, y, runs, len(y))
    C = float(np.median(Cs)) if Cs else 1e-2

    def agent_mean(sc):
        tot = np.bincount(ai, np.nan_to_num(sc), minlength=len(ii))
        cnt = np.bincount(ai, np.isfinite(sc).astype(float), minlength=len(ii))
        return np.round(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), 10)
    # null: the agents' (reaches, onset as a fraction of turns) permuted within strata, 20 draws
    rng = np.random.default_rng(seed)
    ofrac = np.where(np.isfinite(onset_i), onset_i / np.maximum(n_t - 1, 1), np.nan)
    nul_t, nul_a = [], []
    for b in range(N_NULL):
        pr, po = reach_i.copy(), ofrac.copy()
        for st in np.unique(strata_a):
            j = np.flatnonzero(strata_a == st)
            p_ = rng.permutation(len(j))
            pr[j], po[j] = reach_i[j][p_], ofrac[j][p_]
        onp = np.where(np.isfinite(po), np.round(po * np.maximum(n_t - 1, 1)), 1e9)
        yp = ((pr[ai] == 1) & (tpos >= onp[ai])).astype(int)
        sp_, _ = tp.cv_scores(P, yp, runs, len(y), C=C)
        nul_t.append(tp.auc(sp_, yp))
        nul_a.append(tp.auc(agent_mean(sp_), pr))
        status("steps", f"{LABELS[label]} {name}: null draw {b + 1}/{N_NULL}")
    del P
    Fm = meta_frame(A, ii)[ai]
    Ftp = np.column_stack([tpos, frac])
    base = {}
    for bname, F in (("turn index", tpos[:, None].astype(float)), ("turn index + fraction", Ftp),
                     ("metadata + turn index + fraction", np.column_stack([Fm, Ftp]))):
        sb = np.full(len(y), np.nan)
        for k in range(tp.K):
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            sb[te] = meta_fit(F[tr], y[tr], F[te])
        base[bname] = sb
    rows, by_src, delays = [], [], []
    for feat, sc, nl in [("think pair, layer 38 (per turn)", s, (nul_t, nul_a))] + [(b, v, None) for b, v in base.items()]:
        sa = agent_mean(sc)
        neg = (y == 0) & np.isfinite(sc)
        thr = float(np.percentile(sc[neg], 95))
        dd, per = tp.detection_delay(sc, y, ai, tpos, rk, onset_i, thr)
        for o in per:
            delays.append({"label": label, "scheme": name, "feature": feat, "uid": A.uid.values[ii[o["agent_i"]]],
                           "onset": o["onset"], "first_cross": o["first_cross"]})
        tlo, thi = tp.boot_ci(sc, y, runs)
        alo, ahi = tp.boot_ci(sa, reach_i, g_a)
        r = {"label": label, "scheme": name, "sources": "+".join(SHORT[x] for x in srcs), "feature": feat,
             "turns": int(np.isfinite(sc).sum()), "pos_turns": int(y.sum()), "agents": len(ii), "pos_agents": int(reach_i.sum()),
             "dropped_no_onset": int(drop.sum()), "runs": len(np.unique(g_a)), "authors": len(np.unique(au_a)),
             "turn_auroc": tp.auc(sc, y), "turn_ci_lo": tlo, "turn_ci_hi": thi,
             "agent_auroc": tp.auc(sa, reach_i), "agent_ci_lo": alo, "agent_ci_hi": ahi, "threshold_5pct_fpr": thr, **dd,
             "C": C if nl else None}
        if nl:
            nt_, na_ = np.array(nl[0], float), np.array(nl[1], float)
            r.update(turn_null_95=float(np.nanpercentile(nt_, 95)), turn_null_mean=float(np.nanmean(nt_)),
                     agent_null_95=float(np.nanpercentile(na_, 95)), agent_null_mean=float(np.nanmean(na_)))
        rows.append(r)
        for x in np.unique(src_a):
            m = src_a[ai] == x
            ma = src_a == x
            by_src.append({"label": label, "scheme": name, "feature": feat, "source": SHORT[x], "turns": int(m.sum()),
                           "pos_turns": int(y[m].sum()), "agents": int(ma.sum()), "pos_agents": int(reach_i[ma].sum()),
                           "turn_auroc": tp.auc(sc[m], y[m]), "agent_auroc": tp.auc(sa[ma], reach_i[ma])})
        log(f"   steps {label} {name} {feat}: turn {r['turn_auroc']:.3f} agent {r['agent_auroc']:.3f} delay {r['delay_median']}")
    sc_df = pd.DataFrame({"label": label, "scheme": name, "uid": A.uid.values[ii][ai], "turn": tpos, "y": y, "fold": fold,
                          "score": s, **{f"score|{b}": v for b, v in base.items()}})
    return rows, by_src, delays, sc_df


def steps():
    import pandas as pd
    D_ = Data(need_tfidf=False)
    S = np.load(CACHE / "step_L38.npy", mmap_mode="r")
    A = D_.A
    plan = []
    for label in LABELS:
        L = labelled_sources(A, label)
        ours = [s for s in OURS if s in L]
        if label == "cheat":
            # CheatBench has an onset for 21 of 134 positives and EvilGenie none, so 'all' per step keeps the sources
            # with per-call labels: ours and Goodfire
            plan += [(label, "ours", ours), (label, "all (ours + goodfire: the sources with per-call labels)",
                                                [x for x in L if x in OURS or x == "goodfire_reward_hack"])]
        else:
            plan.append((label, "ours (= all: no external source has a board)", ours))
    R, BS, DL = [], [], []
    for label, name, srcs in plan:
        rows, by_src, delays, sc = run_steps(D_, S, label, name, srcs)
        R += rows
        BS += by_src
        DL += delays
        p = OUT / "steps" / f"{label}_{slug(name.split(' (')[0])}_turn_scores.csv.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        sc.to_csv(p, index=False, compression="gzip", float_format="%.6g")
        pd.DataFrame(R).to_csv(OUT / "steps" / "steps.csv", index=False)
        pd.DataFrame(BS).to_csv(OUT / "steps" / "steps_by_source.csv", index=False)
        pd.DataFrame(DL).to_csv(OUT / "steps" / "delays.csv", index=False)
    status("steps", "done")


# ------------------------------------------------------------------------------------------------ report
def f3(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(x) else f"{x:.3f}"


def fci(r):
    a = f3(r.get("auroc"))
    if not a:
        return ""
    lo, hi = f3(r.get("ci_lo")), f3(r.get("ci_hi"))
    return f"{a} [{lo}, {hi}]" if lo else a


def fint(x):
    try:
        x = float(x)
        return "" if not np.isfinite(x) else str(int(x))
    except (TypeError, ValueError):
        return ""


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def scheme_table(c):
    """One scheme: rows (probe type, unit); the main probe with its interval and null, then the other features and
    the baselines on the same folds, then the counts."""
    rows = []
    order = {"B.I": 0, "A": 1, "B.II": 2}
    keys = sorted({(r.ptype, r.unit) for r in c.itertuples()},
                  key=lambda k: (order[k[0]], k[1] == "mean over authors", k[1]))
    for pt, unit in keys:
        sub = c[(c.ptype == pt) & (c.unit == unit)]
        by = {r["feature"]: r for r in sub.to_dict("records")}
        m = by.get("think", {})
        note = next((str(r.get("note")) for r in by.values() if isinstance(r.get("note"), str) and r.get("note")), "")
        if "identical source set" in note:
            note = ""
        if not f3(m.get("auroc")) and note:
            rows.append([pt, unit, "", "", "", "", "", "", fint(m.get("n")), fint(m.get("pos")), fint(m.get("runs")),
                         fint(m.get("authors")), note])
            continue
        rows.append([pt, f"**{unit}**" if unit == "mean over authors" else unit, fci(m), f3(m.get("null_95")),
                     f3(by.get("think38", {}).get("auroc")), f3(by.get("seven", {}).get("auroc")),
                     f3(by.get("tfidf", {}).get("auroc")), f3(by.get("meta", {}).get("auroc")),
                     fint(m.get("n")), fint(m.get("pos")), fint(m.get("runs")), fint(m.get("authors")),
                     (f"train {fint(m.get('n_train'))}" if fint(m.get("n_train")) else "")])
    return md_table(["type", "unit", "think pair, all layers [95% CI]", "null 95th", "layer 38", "seven tokens",
                     "TF-IDF text", "metadata", "agents", "pos", "runs", "authors", "note"], rows)


def write_report():
    import pandas as pd
    A = load_agents()
    I = pd.read_csv(CACHE / "feature_index.csv")
    A = A.join(I.drop(columns="uid"))
    cells = pd.read_csv(OUT / "cells.csv")
    sch = json.load(open(OUT / "schemes.json"))
    H = pd.read_csv(OUT / "heatmap.csv") if (OUT / "heatmap.csv").is_file() else None
    ST = pd.read_csv(OUT / "steps" / "steps.csv") if (OUT / "steps" / "steps.csv").is_file() else None
    SB = pd.read_csv(OUT / "steps" / "steps_by_source.csv") if (OUT / "steps" / "steps_by_source.csv").is_file() else None
    notes = json.load(open(OUT / "labels" / "label_notes.json"))
    L = []
    w = L.append
    w("# Probe matrix: probe types by inclusion schemes\n")
    w("Every probe type crossed with every way of including or excluding the labelled sources, on the reader's "
      "(Qwen3.8-27B) think-token states. Built on this Mac from the local copy of the public Hub dataset, with no "
      "model calls and no spend. Code: `hermeneutics/matrix_probes.py`, which imports the PCA, z-scoring, logistic "
      "regression, inner-CV choice of C, grouped folds, bootstrap and permutation helpers of `hermeneutics/team_probes.py`.\n")
    w("## Design\n")
    w("**Probe types.** **A** is one probe per author, cross-validated within that author with folds grouped by run; "
      "the table gives each author and the mean over authors. **B.I** is one probe on every author of the scheme, "
      "cross-validated with 5 folds grouped by run (stratified by source, author and label). **B.II** leaves one author "
      "out: the probe trains on every other author and scores the held-out author; runs that hold any agent of the "
      "held-out author are removed from training. In the transfer schemes (leave one source out, ours to external, "
      "external to ours) B.I trains on the training sources and scores the test source, A trains per author on the "
      "training sources and scores the same author in the test source, and B.II trains on the training sources "
      "without the test author, again removing every training run that holds any agent of that author.\n")
    w("**Probe.** Each agent is the mean over its turns of the think-open and think-close states at layers 26, 38 and 51 "
      "(30,720 dimensions). Training-fold PCA to at most 128 components, z-scoring, L2 logistic regression with balanced "
      "class weights, C chosen by an inner 3-fold CV grouped by run. Feature rows: the think pair at all layers (main), "
      "the think pair at layer 38 alone, and the seven tokens (five turn-boundary tokens plus the think pair, all "
      "layers) where every agent of the scheme has turns.safetensors.\n")
    w("**Each cell.** Held-out AUROC with a 95% bootstrap interval over runs (1,000 draws; for a mean over authors, "
      "one run draw shared by every author). Null: labels permuted within source, author and arm (20 draws), probe "
      "refitted with C fixed at the median chosen C; cross-validated cells score against the permuted labels "
      "(team_probes.null_aucs), split cells permute the training labels and score the real test labels. The null keeps "
      "each stratum's base rate, so it measures what the probe adds beyond source, author and arm. Baselines on the "
      "same folds: **TF-IDF text** (word unigrams and bigrams of the rendered transcript, the reader's whole input; "
      "vocabulary of the 50,000 most frequent hashed terms with min_df 2, sublinear tf and idf fitted on the training "
      "rows; 2^22 hash buckets, so rare terms can collide; C by inner CV) and **metadata** (one-hot source, author, "
      "arm, task, role, log turns, log tokens; logistic regression, C = 1).\n")
    w("**Labels.** Cheating: controls_600 uses the set's own label (data/labels.csv `cheat`: from `tried` for "
      "software engineering, chess and creative writing, from `opened` otherwise); the team and SPAR sets use the "
      "code ladder of viewer/levels.py, used or above; Goodfire uses the judge verdict of labels/judge_rollouts.jsonl, "
      "yes against no, split dropped; CheatBench and EvilGenie use the external_probes agent's label tables (CheatBench "
      "meta.json label, cheating or attempt against benign; EvilGenie sel_label, hack against legitimate, 133 matched "
      "pairs of one author, each pair one group in every fold). Coordination: posted or above. Collusion: crossed or above. Coordination and "
      "collusion only on board_seed, solo_design, spar_screen_team and spar_mixed_41, as asked; spar_exploration_single "
      "also has a board but is left out of those two labels. A source enters a label when it has at least 3 agents "
      "in each class; a cell runs when each class has at least 3 agents and there are at least 5 runs.\n")
    w("## Sources, counts and provenance\n")
    ext = notes.get("external_label_file")
    prov = {
        "controls_600": "ours; SGA-adapted single-agent controls run 2026-11-06 22:20 to 2026-11-07 UTC, during the sprint (opened 2026-11-06 21:30 UTC)",
        "board_seed": "ours; three-worker team runs made 2026-11-07, during the sprint",
        "solo_design": "ours; runs made 2026-11-09, during the sprint",
        "spar_screen_team": "SPAR team's runs (SparGuardianAgents, dataset unrulyabstractions/spar-swarm), 2026-10-30 to 10-31; predate the sprint",
        "spar_mixed_41": "SPAR team's runs, 2026-11-01 to 11-02; predate the sprint",
        "spar_exploration_single": "SPAR team's runs, 2026-11-01 to 11-03; predate the sprint",
        "spar_screen_solo": "SPAR team's runs, 2026-10-30; predate the sprint",
        "goodfire_reward_hack": "external (Goodfire/reward-hack-data, arXiv:2609.19101); predates the sprint",
        "cheatbench": "external (steinad/CheatBench); predates the sprint",
        "evilgenie": "external (WiserHumanExperimental/evilgenie-escalation); predates the sprint",
    }
    extr = {s: "another agent (Plan B extraction on rented boxes, 2026-11-09)" for s in OURS}
    extr.update({"goodfire_reward_hack": "another agent (hermeneutics/extract_goodfire.py, 2026-11-09)",
                 "cheatbench": "another agent", "evilgenie": "another agent"})
    rows = []
    for s in SOURCES:
        a = A[A.source == s]
        ok = a[a.ok == 1]
        lab = ok[ok.cheat.notna()]
        rows.append([s, len(a), int(ok.ok.sum()), len(lab), int(lab.cheat.sum()) if len(lab) else 0,
                     int(ok.coord.sum()) if s in BOARD else "", int(ok.collude.sum()) if s in BOARD else "",
                     ok.group.nunique(), ok.author.nunique(), prov[s], extr[s]])
    w(md_table(["source", "agents", "with features", "cheat labelled", "cheat pos", "coord pos", "collude pos", "runs",
                "authors", "data", "activations extracted by"], rows))
    w("")
    w("Labels of every source were built here; for board_seed, solo_design, spar_screen_team and spar_mixed_41 they "
      "equal the team_probes agent's label CSVs on every agent (cheat, coordination and collusion indices, onsets, "
      "roles). For spar_exploration_single and spar_screen_solo the run dirs were fetched from the public "
      "unrulyabstractions/spar-swarm dataset and run through the same levels.py code. Goodfire: "
      f"{notes['goodfire_reward_hack'].get('verdict_yes', 0)} yes, {notes['goodfire_reward_hack'].get('verdict_no', 0)} no, "
      f"{notes['goodfire_reward_hack'].get('verdict_split', 0)} split (dropped), "
      f"{notes['goodfire_reward_hack'].get('verdict_None', 0)} without a judge verdict (tau3bench and shoppingbench carry "
      "only the generic monitor, and 6 DeepSWE rollouts were not judged). A Goodfire rollout is its own run. "
      + ("CheatBench and EvilGenie: no valid label from the external_probes agent was found "
         "(no label file under hermeneutics/results/external_probes), so both are listed as no label and left out." if not ext
         else "CheatBench and EvilGenie: labels read from the external_probes agent's `labels/cheatbench.csv` and "
              "`labels/evilgenie.csv` (that agent checked the joins, the counts and the matched pairs in its own "
              "VERIFICATION.md); 20 CheatBench rollouts with zero assistant turns have no features and are left out. "
              "Neither set has enough per-call onsets for the per-step probe (CheatBench 21 of 134 positives, EvilGenie "
              "none), so the per-step 'all' scheme keeps ours and Goodfire.") + "\n")
    w("**Authors.** Names are normalised from each agent's own model id (the episode's model for team runs, so "
      "regulators of another model are their own author), never from a short name: `qwen` in controls_600 is Qwen3.8 "
      "27B, `qwenmax` in Goodfire is Qwen3.8 Max. One judgement call: solo_design's `openrouter/qwen/qwen3.8-max-0902` "
      "and Goodfire's `Qwen/Qwen3.8-2.4T-A95B-FP8` are both named Qwen 3.8 Max by their sources and are one author here.\n")
    am = A.groupby(["author_raw", "author"]).agg(sources=("source", lambda v: ", ".join(sorted(set(SHORT[x] for x in v)))),
                                                 agents=("uid", "size")).reset_index()
    w(md_table(["model id in the metadata", "author", "sources", "agents"], am.values.tolist()))
    w("")
    # compact matrix
    w("## The matrix (main feature: think pair, all layers)\n")
    w("AUROC [95% CI]; counts are agents / positives / runs / authors of the scored test rows. A and B.II give the mean "
      "over the authors that could be scored. `null` is the 95th percentile of the 20-draw permutation null of the same "
      "cell. TF-IDF and metadata are the B.I baselines on the same folds.\n")
    for label in LABELS:
        w(f"### {LABELS[label]}\n")
        rows = []
        for x in [s for s in sch if s["label"] == label]:
            c = cells[(cells.label == label) & (cells.scheme == x["scheme"])]

            def pick(pt, unit):
                r = c[(c.ptype == pt) & (c.unit == unit) & (c.feature == "think")]
                return r.iloc[0].to_dict() if len(r) else {}
            bi, am_, bm = pick("B.I", "pooled"), pick("A", "mean over authors"), pick("B.II", "mean over authors")
            cnt = (lambda r: f"{fint(r.get('n'))}/{fint(r.get('pos'))}/{fint(r.get('runs'))}/{fint(r.get('authors'))}" if fint(r.get("n")) else "")
            tb = c[(c.ptype == "B.I") & (c.unit == "pooled")]
            bb = {r["feature"]: r for r in tb.to_dict("records")}
            note = c.note.dropna().iloc[0] if c.note.notna().any() else ""
            rows.append([x["scheme"], fci(am_) or "n/a", cnt(am_), fci(bi) or "n/a", cnt(bi), f3(bi.get("null_95")),
                         f3(bb.get("tfidf", {}).get("auroc")), f3(bb.get("meta", {}).get("auroc")),
                         fci(bm) or "n/a", cnt(bm), note if isinstance(note, str) else ""])
        w(md_table(["scheme", "A mean", "A n/pos/runs/auth", "B.I", "B.I n/pos/runs/auth", "B.I null 95th",
                    "B.I TF-IDF", "B.I metadata", "B.II mean", "B.II n/pos/runs/auth", "note"], rows))
        w("")
    if H is not None:
        w("## Source-to-source transfer\n")
        w("Train on every labelled agent of the row source, score every labelled agent of the column source (think "
          "pair, all layers). The diagonal is B.I within the source. Figures: `heatmap_cheating.png`, `heatmap_board.png`; "
          "numbers with intervals in `heatmap.csv`.\n")
        w("![cheating transfer](heatmap_cheating.png)\n")
        w("![coordination and collusion transfer](heatmap_board.png)\n")
        for label in LABELS:
            h = H[H.label == label]
            src = list(dict.fromkeys(h.train))
            rows = []
            for i in src:
                rr = [i]
                for j in src:
                    r = h[(h.train == i) & (h.test == j)]
                    rr.append(f3(r.auroc.iloc[0]) if len(r) else "")
                rows.append(rr)
            w(f"**{LABELS[label]}** (rows train, columns test)\n")
            w(md_table(["train \\ test"] + src, rows))
            w("")
    if ST is not None:
        w("## Per step (layer 38, every turn)\n")
        w("One row per turn; a turn is positive from the agent's onset turn on (levels.py per-call onsets for our sets: "
          "cheating at the first write of planted content, else the opened call; coordination at the first post; "
          "collusion at team_probes' onset proxy. controls_600: labels.csv onset_turn. Goodfire: the first turn at or "
          "after the first positive passage of judge_passages.jsonl). Agents that reach the label with no usable onset "
          "are dropped. Probe: training-fold PCA (exact, from the streamed training covariance) to 128 components, "
          "z-scoring, logistic regression, 5 folds grouped by run. Agent AUROC uses the mean of the agent's turn "
          "scores. Detection delay: first turn above the 5% false-positive threshold of negative turns, minus the onset "
          "turn. Null: agents' (reaches, onset fraction) permuted within source, author and arm, 20 draws.\n")
        rows = []
        for r in ST.to_dict("records"):
            rows.append([LABELS[r["label"]], r["scheme"], r["feature"],
                         f"{f3(r['turn_auroc'])} [{f3(r['turn_ci_lo'])}, {f3(r['turn_ci_hi'])}]", f3(r.get("turn_null_95")),
                         f"{f3(r['agent_auroc'])} [{f3(r['agent_ci_lo'])}, {f3(r['agent_ci_hi'])}]", f3(r.get("agent_null_95")),
                         f"{fint(r['detected'])}/{fint(r['agents_reaching'])}",
                         f"{r['delay_median']:.0f} ({r['delay_q25']:.0f} to {r['delay_q75']:.0f})" if np.isfinite(r["delay_median"]) else "",
                         fint(r["fired_before_onset"]), f"{fint(r['turns'])}/{fint(r['pos_turns'])}",
                         f"{fint(r['agents'])}/{fint(r['pos_agents'])}", fint(r["dropped_no_onset"]), fint(r["runs"]), fint(r["authors"])])
        w(md_table(["label", "scheme", "features", "turn AUROC", "turn null 95th", "agent AUROC", "agent null 95th",
                    "detected / reaching", "delay median (IQR)", "fired before onset", "turns / pos", "agents / pos",
                    "dropped (no onset)", "runs", "authors"], rows))
        w("")
        if SB is not None:
            w("Turn AUROC within each source of the scheme (same scores):\n")
            sb = SB[SB.feature.str.startswith("think")]
            rows = [[LABELS[r["label"]], r["scheme"], r["source"], f3(r["turn_auroc"]), f3(r["agent_auroc"]),
                     f"{r['turns']}/{r['pos_turns']}", f"{r['agents']}/{r['pos_agents']}"] for r in sb.to_dict("records")]
            tb = SB[SB.feature == "turn index + fraction"].set_index(["label", "scheme", "source"]).turn_auroc
            rows = [r + [f3(tb.get((lab_k, r[1], r[2]), np.nan))] for r, lab_k in zip(rows, sb.label)]
            w(md_table(["label", "scheme", "source", "turn AUROC", "agent AUROC", "turns / pos", "agents / pos",
                        "turn index + fraction, turn AUROC"], rows))
            w("")
    L += arch_report_lines(cells)
    w("## One table per scheme\n")
    AC = pd.read_csv(ARCH_OUT / "arch_cells.csv") if (ARCH_OUT / "arch_cells.csv").is_file() else pd.DataFrame(
        columns=["label", "scheme", "ptype", "arch", "traj_auroc", "note"])
    plan_keys = ({(x["label"], x["scheme"], x["ptype"]) for x in json.load(open(ARCH_OUT / "arch_plan.json"))}
                 if (ARCH_OUT / "arch_plan.json").is_file() else set())
    w("Each scheme has two tables. The first has architecture rows and probe-type columns: held-out AUROC of the "
      "agent-level (trajectory) score, the mean over held-out authors for A and B.II, the mean over two seeds with "
      "the seed range for trained architectures; 'not planned' marks a probe type outside the architecture plan, "
      "'not reached' one the time budget did not reach. The second table gives the linear probe in full. "
      "Columns: the main probe with its interval and null; the same probe on layer 38 alone and on the seven tokens "
      "(blank where not every agent has turns.safetensors); the TF-IDF and metadata baselines on the same folds or "
      "split; the counts of the scored rows (B.II and transfer rows also give the training size).\n")
    for label in LABELS:
        w(f"### {LABELS[label]}\n")
        for x in [s for s in sch if s["label"] == label]:
            c = cells[(cells.label == label) & (cells.scheme == x["scheme"])]
            te = "cross-validated within the training sources" if x["test"] is None else f"test {', '.join(x['test']) or 'none'}"
            w(f"#### {LABELS[label]}: {x['scheme']}\n")
            w(f"Train {', '.join(x['train']) or 'none'}; {te}.\n")
            if c.note.astype(str).str.contains("identical source set").any():
                w(f"{c.note.dropna().iloc[0]}.\n")
            w(arch_ptype_table(AC, cells, plan_keys, label, x["scheme"]))
            w("")
            w(scheme_table(c))
            w("")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    log(f"RESULTS.md written ({len(L)} blocks)")


# ------------------------------------------------------------------------------------------------ verification
def tie_bound(y, s):
    """Largest AUROC change that ties among the saved scores can cause: the saved CSVs keep 6 significant digits, so
    scores that differed in the run can tie on disk, and each tied positive-negative pair moves the AUROC by at most
    0.5 / (positives x negatives)."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    v, inv = np.unique(s, return_inverse=True)
    P, N = np.bincount(inv, y == 1, len(v)), np.bincount(inv, y == 0, len(v))
    return float(0.5 * (P * N).sum() / max(1.0, float((y == 1).sum()) * float((y == 0).sum())))


def verify():
    """Checks written to VERIFICATION.md: one AUROC per scheme recomputed by hand (sklearn) from the saved scores; no run
    in two folds of any cross-validated cell; each B.II fold holds one author and no training run of that author;
    five author normalisations against the raw source files; the pooled features against team_probes."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    A = load_agents()
    I = pd.read_csv(CACHE / "feature_index.csv")
    A = A.join(I.drop(columns="uid"))
    grp = dict(zip(A.uid, A.group))
    auth = dict(zip(A.uid, A.author))
    cells = pd.read_csv(OUT / "cells.csv")
    sch = json.load(open(OUT / "schemes.json"))
    L, bad = [], []
    w = L.append
    w("# Verification of the probe matrix\n")
    w(f"Run {time.strftime('%Y-%m-%d %H:%M %Z')} by `python -m hermeneutics.matrix_probes verify`.\n")
    # 1. hand recomputation
    w("## 1. One AUROC per scheme recomputed from the saved scores\n")
    w("For each scheme the B.I main-probe row (or, when B.I did not run, the first A or B.II row with an AUROC) is "
      "recomputed twice from `scores/<label>/<scheme>.csv.gz` and compared with cells.csv: with "
      "sklearn.metrics.roc_auc_score, and by hand as the share of positive-negative pairs the probe orders correctly "
      "(ties count half), counted pair by pair with no library AUROC code. "
      "The saved scores keep 6 significant digits, so scores can tie on disk that did not tie in the run; a row matches "
      "when the difference is within the largest change those ties can cause (0.5 per tied positive-negative pair over "
      "positives x negatives; 0 when no score ties) and the counts are equal.\n")
    rows = []
    for x in sch:
        c = cells[(cells.label == x["label"]) & (cells.scheme == x["scheme"]) & cells.auroc.notna()]
        if c.note.astype(str).str.contains("identical source set").any() or not len(c):
            rows.append([x["label"], x["scheme"], "", "", "", "", "no own scores (identical to another scheme or no cell)"])
            continue
        c = c.assign(o=c.ptype.map({"B.I": 0, "A": 1, "B.II": 2}) + (c.feature != "think") * 10).sort_values("o")
        c = c[c.unit != "mean over authors"]
        r = c.iloc[0]
        S = pd.read_csv(OUT / "scores" / x["label"] / f"{slug(x['scheme'])}.csv.gz")
        s = S[(S.ptype == r.ptype) & (S.unit == r.unit) & (S.feature == r.feature)]
        s = s[np.isfinite(s.score)]
        a = roc_auc_score(s.y, s.score)
        tb = tie_bound(s.y, s.score)
        sp_, sn_ = s.score.values[s.y.values == 1], s.score.values[s.y.values == 0]
        hand = sum(float((p_ > sn_).sum()) + 0.5 * float((p_ == sn_).sum()) for p_ in sp_) / (len(sp_) * len(sn_))
        ok = (abs(a - r.auroc) <= tb + 1e-9 and abs(hand - a) < 1e-12 and len(s) == r.n and int(s.y.sum()) == r.pos)
        bad += [] if ok else [f"recompute {x['label']} {x['scheme']}"]
        rows.append([x["label"], x["scheme"], f"{r.ptype} {r.unit} {r.feature}", f"{r.auroc:.6f}", f"{a:.6f}", f"{hand:.6f}",
                     f"{abs(a - r.auroc):.1e}", f"{tb:.1e}", f"{len(s)}/{int(s.y.sum())}", "match" if ok else "MISMATCH"])
    w(md_table(["label", "scheme", "row", "cells.csv", "sklearn", "pair count", "difference", "tie bound", "n/pos", "result"], rows))
    w("")
    # 2. folds
    w("## 2. Runs and folds\n")
    rows = []
    n_cv = n_bad = n_bii = n_bii_bad = 0
    for p in sorted((OUT / "scores").rglob("*.csv.gz")):
        S = pd.read_csv(p, usecols=["label", "scheme", "ptype", "unit", "feature", "uid", "fold"])
        S = S[S.feature == "think"]
        S["run"] = S.uid.map(grp)
        for (pt, unit), s in S.groupby(["ptype", "unit"]):
            if (s.fold >= 0).all():
                n_cv += 1
                k = int((s.groupby("run").fold.nunique() > 1).sum())
                n_bad += int(k > 0)
                if k:
                    bad.append(f"run in two folds: {p.name} {pt} {unit}")
            elif pt == "B.II":
                n_bii += 1
                au = set(s.uid.map(auth))
                n_bii_bad += int(len(au) != 1 or au != {unit})
    runs_of = defaultdict(set)
    for u_, a_ in auth.items():
        runs_of[a_].add(grp[u_])
    for p in sorted((OUT / "folds").rglob("*_BII.json.gz")):
        fo = json.load(gzip.open(p, "rt"))
        for a, v in fo.items():
            te_runs = {grp[u] for u in v["test_uids"]}
            te_auth = {auth[u] for u in v["test_uids"]}
            over = te_runs & set(v["train_runs"])
            mates = set(v["train_runs"]) & runs_of.get(a, set())  # training runs holding any agent of the author
            okk = te_auth == {a} and not over and not mates and a not in v["train_authors"]
            rows.append([f"{p.parent.name}/{p.name.replace('_BII.json.gz', '')}", a, len(v["test_uids"]), len(te_runs), len(v["train_runs"]),
                         len(over), len(mates), ", ".join(sorted(te_auth)), "ok" if okk else "BAD"])
            if not okk:
                bad.append(f"B.II fold {p.name} {a}")
    w(f"Cross-validated score sets (B.I and A, main probe): {n_cv}; with a run in more than one fold: {n_bad}. "
      f"B.II score sets: {n_bii}; holding other than exactly the one held-out author: {n_bii_bad}.\n")
    w("Every B.II fold (from `folds/<label>/<scheme>_BII.json.gz`): test agents, test runs, training runs, runs in "
      "both, training runs that hold any agent of the held-out author (must be 0, so no run-mate of that author trains), "
      "authors in the test set.\n")
    w(md_table(["label/scheme", "held-out author", "test agents", "test runs", "train runs", "runs in both",
                "train runs holding the author", "test authors", "result"], rows))
    w("")
    # 3. author normalisation
    w("## 3. Five author normalisations, checked against the raw files\n")
    picks = [("controls_600", "Qwen3.8 27B"), ("goodfire_reward_hack", "Qwen3.8 Max"), ("spar_mixed_41", "Gemini 4 Argon"),
             ("spar_screen_team", "Claude Haiku 5.5"), ("solo_design", "Qwen3.8 Max")]
    rows = []
    for s, a in picks:
        r = A[(A.source == s) & (A.author == a)].iloc[0]
        m = _meta(s, r.run, r.agent)
        if s == "controls_600":
            raw = pd.read_csv(REPO / "data" / "labels.csv").set_index("run").loc[r.run, "model_id"]
            where = f"data/labels.csv model_id; meta author '{m.get('author')}'"
        elif s == "goodfire_reward_hack":
            raw = m.get("author_model")
            where = f"meta author_model; meta author '{m.get('author')}', author_name '{m.get('author_name')}'"
        else:
            rd = _run_dir(s, m["source_path"])
            ep = [json.loads(x) for x in open(rd / "episodes.jsonl")]
            aid = m.get("agent_id") or r.agent
            raw = next((e.get("model") for e in ep if e.get("orbit_agent") == aid), None)
            where = f"episodes.jsonl model of agent {aid}; meta author '{m.get('author')}'"
        okk = canon(raw) == a == r.author
        bad += [] if okk else [f"author {r.uid}"]
        rows.append([r.uid, where, raw, r.author, "ok" if okk else "BAD"])
    w(md_table(["agent", "raw field read", "raw model id", "normalised author", "result"], rows))
    w("")
    w("In spar_mixed_41 the meta.json `author` field names the run's worker model even for regulators; the episode's "
      "model is used instead, so regulators count as their own author.\n")
    # 4. features
    w("## 4. Pooled features against team_probes\n")
    D_ = Data(need_tfidf=False)
    rows = []
    rng = np.random.default_rng(11)
    for s in OURS + ["goodfire_reward_hack"]:
        i = int(rng.choice(np.flatnonzero(((D_.A.source == s) & (D_.A.ok == 1)).values)))
        r = D_.A.iloc[i]
        acts = tp.load_means(s, pd.DataFrame([{"run": r.run, "agent": r.agent}]))
        ref = tp.agent_matrix(acts, [(r.run, r.agent)], "all", "think")[0]
        err = float(np.abs(ref - D_.X("think", [i])[0]).max() / np.abs(ref).max())
        bad += [] if err < 2e-3 else [f"features {r.uid}"]
        rows.append([r.uid, f"{err:.1e}", "ok (fp16 rounding)" if err < 2e-3 else "BAD"])
    w(md_table(["agent", "max abs difference / max abs value vs team_probes.agent_matrix", "result"], rows))
    w("")
    # 5. labels
    w("## 5. Labels against the team_probes agent's CSVs\n")
    rows = []
    for s, f in [("board_seed", "agents.csv"), ("solo_design", "solo_design_agents.csv"),
                 ("spar_mixed_41", "spar_mixed_41_agents.csv"), ("spar_screen_team", "spar_screen_team_agents.csv")]:
        t = pd.read_csv(tp.OUT / "labels" / f)
        m = A[A.source == s].merge(t, on=["run", "agent"], suffixes=("", "_tp"))
        eq = [(m.cheat_index == m.cheat_tp).sum(), (m.coord_index == m.coord_tp.fillna(0)).sum(),
              (m.collude_index == m.collude_tp.fillna(0)).sum(),
              (m.cheat_onset.fillna(-1) == m.used_onset_step.fillna(-1)).sum()]
        okk = all(e == len(m) == (A.source == s).sum() for e in eq)
        bad += [] if okk else [f"labels {s}"]
        rows.append([s, (A.source == s).sum(), len(m)] + [int(e) for e in eq] + ["ok" if okk else "BAD"])
    w(md_table(["source", "agents here", "joined", "cheat index equal", "coord index equal", "collude index equal",
                "cheat onset equal", "result"], rows))
    w("")
    notes = json.load(open(OUT / "labels" / "label_notes.json"))
    w(f"controls_600: label equal to data/labels.csv on {notes['controls_600']['label_match']} of 600 runs; onset_turn "
      f"equal to the step of onset_call in source.json on {notes['controls_600']['onset_match']} positives "
      f"({notes['controls_600']['onset_mismatch']} mismatches). Goodfire: judge_rollouts verdict equal to meta.json "
      f"verdict on {notes['goodfire_reward_hack'].get('meta_verdict_match', 0)} rollouts "
      f"({notes['goodfire_reward_hack'].get('meta_verdict_mismatch', 0)} mismatches).\n")
    # 6. streamed per-step PCA against team_probes.prep_folds on the board_seed turns
    w("## 6. Streamed per-step PCA against team_probes.prep_folds\n")
    S_ = np.load(CACHE / "step_L38.npy", mmap_mode="r")
    ii = np.flatnonzero(((D_.A.source == "board_seed") & (D_.A.ok == 1)).values)
    R = np.concatenate([np.arange(D_.A.s0.values[i], D_.A.s1.values[i]) for i in ii])
    ai = np.repeat(np.arange(len(ii)), D_.A.n_turns.values[ii].astype(int))
    fold = tp.folds_for(D_.A.cheat.values[ii].astype(int), D_.A.group.values[ii])[ai]
    P1 = step_pca(S_, R, fold, chunk=2000)
    P2 = tp.prep_folds(np.asarray(S_[R], np.float32), fold)
    rows = []
    for f in range(tp.K):
        c = [abs(np.corrcoef(P1[f][2][:, j], P2[f][2][:, j])[0, 1]) for j in range(P1[f][2].shape[1])]
        rows.append([f, len(P1[f][0]), f"{min(c[:20]):.5f}", f"{np.median(c):.5f}"])
    y_ = (D_.A.cheat.values[ii][ai] == 1).astype(int)
    g_ = D_.A.group.values[ii][ai]
    s1, _ = tp.cv_scores(P1, y_, g_, len(y_), C=0.01)
    s2, _ = tp.cv_scores(P2, y_, g_, len(y_), C=0.01)
    a1, a2 = tp.auc(s1, y_), tp.auc(s2, y_)
    bad += [] if abs(a1 - a2) < 0.01 else ["step PCA"]
    w(md_table(["fold", "training turns", "min abs corr, top 20 components", "median abs corr, 128 components"], rows))
    w(f"\nTurn AUROC (board_seed cheating, C = 0.01): streamed {a1:.4f}, prep_folds {a2:.4f}; score correlation "
      f"{np.corrcoef(s1, s2)[0, 1]:.4f}. prep_folds uses a 4-iteration randomized SVD above 1,500 rows, so the two "
      "agree to that solver's precision.\n")
    # 7. architectures
    ac_p = ARCH_OUT / "arch_cells.csv"
    if ac_p.is_file():
        w("## 7. Architectures\n")
        AC = pd.read_csv(ac_p)
        AC = AC[AC.traj_auroc.notna()] if "traj_auroc" in AC else AC.iloc[0:0]
        rows, frows = [], []
        n_rows = n_ok = 0
        for (label, scheme, ptype), c in AC.groupby(["label", "scheme", "ptype"], sort=False):
            k_ok = 0
            for arch, ca in c.groupby("arch", sort=False):
                f = ARCH_OUT / "scores" / label / f"{slug(scheme)}__{ptype.replace('.', '')}__{arch}.csv.gz"
                S = pd.read_csv(f)
                for r in ca.itertuples():
                    sc = S[(S.unit == r.unit) & (S.seed.astype(float) == float(r.seed))]
                    a = roc_auc_score(sc.y, sc.score) if len(sc) and sc.y.nunique() == 2 else np.nan
                    okk = abs(a - r.traj_auroc) <= tie_bound(sc.y, sc.score) + 1e-9 and len(sc) == r.n
                    n_rows += 1
                    n_ok += int(okk)
                    k_ok += int(okk)
                    bad += [] if okk else [f"arch recompute {label} {scheme} {ptype} {arch} {r.unit} seed {r.seed}"]
                # folds: B.I arch folds equal the linear matrix's B.I folds; B.II test sets hold one author
                if ptype == "B.I" and S.fold.max() >= 0:
                    m = OUT / "scores" / label / f"{slug(scheme)}.csv.gz"
                    if m.is_file():
                        M = pd.read_csv(m, usecols=["ptype", "unit", "feature", "uid", "fold"])
                        M = M[(M.ptype == "B.I") & (M.feature == "think")].set_index("uid").fold
                        for sd_v, sc in S.groupby("seed"):
                            j_ = sc.set_index("uid").fold
                            same = int((M.reindex(j_.index).values == j_.values).sum())
                            split_runs = int((sc.assign(run=sc.uid.map(grp)).groupby("run").fold.nunique() > 1).sum())
                            okf = same == len(j_) and split_runs == 0
                            bad += [] if okf else [f"arch folds {label} {scheme} {arch} seed {sd_v}"]
                            frows.append([label, scheme, "B.I", f"{arch} seed {sd_v}", f"{same}/{len(j_)} agents in the "
                                          f"same fold as the linear matrix; runs in two folds: {split_runs}", "ok" if okf else "BAD"])
                if ptype == "B.II":
                    n1 = sum(len(set(x.uid.map(auth))) == 1 and set(x.uid.map(auth)) == {u} for u, x in S.groupby("unit"))
                    okb = n1 == S.unit.nunique()
                    bad += [] if okb else [f"arch B.II {label} {scheme} {arch}"]
                    frows.append([label, scheme, "B.II", arch, f"{n1}/{S.unit.nunique()} held-out author sets of exactly "
                                  "that author", "ok" if okb else "BAD"])
            r = c.iloc[0]
            f = ARCH_OUT / "scores" / label / f"{slug(scheme)}__{ptype.replace('.', '')}__{r.arch}.csv.gz"
            S = pd.read_csv(f)
            sc = S[(S.unit == r.unit) & (S.seed.astype(float) == float(r.seed))]
            a = roc_auc_score(sc.y, sc.score)
            rows.append([label, scheme, ptype, f"{r.arch} {r.unit} seed {int(r.seed)}", f"{r.traj_auroc:.6f}", f"{a:.6f}",
                         f"{k_ok}/{len(c)} rows of the cell match"])
        w(f"Every architecture row with an AUROC ({n_rows}) was recomputed with sklearn from `arch/scores/`; "
          f"{n_ok} match within the tie bound of section 1 with the same count. One row per cell is shown:\n")
        w(md_table(["label", "scheme", "type", "row", "arch_cells.csv", "recomputed", "result"], rows))
        w("")
        w(md_table(["label", "scheme", "type", "architecture", "check", "result"], frows))
        w("")
        w("The stack's components on the outer training agents come from an inner 3-fold split of those agents only "
          "(StratifiedGroupKFold over their runs); the outer test agents enter only through the four component scores "
          "of models fitted on the whole outer training set.\n")
    w("## Result\n")
    w("All checks passed." if not bad else "FAILED: " + "; ".join(bad))
    (OUT / "VERIFICATION.md").write_text("\n".join(L) + "\n")
    log(f"verify: {len(bad)} failures {bad[:10]}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["labels", "features", "tfidf", "matrix", "heatmap", "steps", "archs", "report", "verify",
                                      "all"])
    ap.add_argument("--hours", type=float, default=6.0, help="archs: time budget from the start of this process")
    ap.add_argument("--only", default=None, help="matrix: run only schemes whose name contains this")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = a.stage
    if st in ("labels", "all"):
        labels()
    if st in ("features", "all"):
        features()
    if st in ("tfidf", "all"):
        tfidf_counts()
    if st in ("matrix", "all"):
        matrix(a.only, a.force)
    if st in ("heatmap", "all"):
        heatmap()
    if st in ("steps", "all"):
        steps()
    if st in ("archs", "all"):
        dl = Path(STATUS.parent / "arch_deadline")
        archs(a.hours, a.only, deadline=float(dl.read_text()) if dl.is_file() else None)
    if st in ("report", "all"):
        write_report()
    if st in ("verify", "all"):
        verify()



# ================================================================================================ architectures
# A third axis: the probe architecture, on the main cells, with the same folds, labels and baselines as the linear
# matrix. Per-step input: the layer-38 think pair of every turn (data/matrix_probes_cache/step_L38.npy, built once),
# standardised with the training turns. Trajectory-level input (architecture 8): the pooled all-layers vector.
ARCHS = ["linear", "mlp_mean", "mlp_max", "attn_pool", "attn_monitor", "gru", "linear_step", "mlp_traj", "stack"]
ARCH_NAMES = {
    "linear": "1 linear: PCA + z + L2 logistic, mean-pooled think pair (all layers)",
    "mlp_mean": "2 MLP-pool: per-step MLP, causal running mean, 2-layer MLP",
    "mlp_max": "3 MLP-max: per-step MLP, causal running max, 2-layer MLP",
    "attn_pool": "4 attention pooling: per-step MLP, one learned query over the steps so far, MLP",
    "attn_monitor": "5 attention monitor (attn_probe.build_model: 2 causal self-attention layers, 4 heads, d 128)",
    "gru": "6 GRU: per-step MLP, one-layer GRU (hidden 128), MLP",
    "linear_step": "7 linear per step (PCA + z + L2 logistic on turns), running mean of its scores",
    "mlp_traj": "8 MLP on the mean-pooled all-layers vector (PCA 256 first)",
    "stack": "9 stacked: logistic regression over held-out scores of 1, 2, 5 and TF-IDF (inner folds)",
}
ARCH_SHORT = {"linear": "1 linear (pooled)", "mlp_mean": "2 MLP, running mean", "mlp_max": "3 MLP, running max",
              "attn_pool": "4 attention pooling", "attn_monitor": "5 attention monitor", "gru": "6 GRU",
              "linear_step": "7 linear per step", "mlp_traj": "8 MLP (pooled)", "stack": "9 stack"}
SEQ = ["mlp_mean", "mlp_max", "attn_pool", "attn_monitor", "gru"]
SEEDS = (0, 1)
H_ = 128
ARCH_OUT = OUT / "arch"


def _torch():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    return torch


def arch_device():
    torch = _torch()
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_net(kind, n_in, seed, label):
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    if kind == "attn_monitor":
        from hermeneutics import attn_probe as ap
        m = ap.build_model(n_in, seed)
        L = {"cheat": "cheat", "coord": "coord", "collude": "collude"}[label]

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = m

            def forward(self, z, pad):
                st = torch.arange(z.shape[1], device=z.device)[None].expand(z.shape[0], -1)
                return self.m(z, st, None, pad)[L]["bin"]
        return Wrap()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(n_in, H_), nn.GELU(), nn.Dropout(0.3))
            if kind == "gru":
                self.gru = nn.GRU(H_, H_, batch_first=True)
            if kind == "attn_pool":
                self.q = nn.Parameter(torch.randn(H_) / H_ ** 0.5)
                self.k = nn.Linear(H_, H_)
                self.v = nn.Linear(H_, H_)
            self.head = nn.Sequential(nn.Linear(H_, H_), nn.GELU(), nn.Dropout(0.3), nn.Linear(H_, 1))

        def forward(self, z, pad):
            # padding sits after each agent's last step, and every pooling here is causal, so it never reaches a real step
            h = self.enc(z)
            if kind == "mlp_mean":
                c = torch.cumsum(h, 1) / torch.arange(1, h.shape[1] + 1, device=h.device)[None, :, None]
            elif kind == "mlp_max":
                c = torch.cummax(h, 1).values
            elif kind == "gru":
                c, _ = self.gru(h)
            elif kind == "attn_pool":
                n = h.shape[1]
                s = self.k(h) @ self.q / H_ ** 0.5  # [B, n]
                fut = torch.triu(torch.ones(n, n, dtype=torch.bool, device=h.device), 1)
                att = torch.softmax(s[:, None, :].expand(-1, n, -1).masked_fill(fut, float("-inf")), -1)
                c = att @ self.v(h)
            return self.head(c).squeeze(-1)
    return Net()


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


class SeqData:
    """The turns of the scheme's agents on the device (float16) with per-turn targets: 0 before the onset and 1 from
    it on (weight 1 and 2, attn_probe's weighting), weight 0 on every turn of an agent that reaches the label with no
    usable onset (only its trajectory label trains); the trajectory label at the last step."""

    def __init__(self, D_, S, ii, label, dev):
        torch = _torch()
        A = D_.A
        self.ii, self.dev = ii, dev
        self.n = A.n_turns.values[ii].astype(int)
        self.start = np.concatenate([[0], np.cumsum(self.n)[:-1]]).astype(int)
        R = np.concatenate([np.arange(A.s0.values[i], A.s1.values[i]) for i in ii])
        assert len(R) == self.n.sum() and (np.diff(R) > 0).all()
        self.N = len(R)
        self.R = R
        X = torch.zeros((self.N + 1, S.shape[1]), dtype=torch.float16, device=dev)
        for c in range(0, self.N, 20000):
            X[c:min(c + 20000, self.N)] = torch.from_numpy(np.ascontiguousarray(S[R[c:c + 20000]])).to(dev)
        self.X = X
        self.y = A[label].values[ii].astype(int)
        on = A[f"{label}_onset"].values[ii].astype(float)
        self.onset_ok = (self.y == 0) | (np.isfinite(on) & (on < self.n))
        self.onset = np.where(self.y == 1, on, np.nan)
        ai = np.repeat(np.arange(len(ii)), self.n)
        tpos = np.concatenate([np.arange(k) for k in self.n])
        yt = ((self.y[ai] == 1) & (tpos >= np.where(np.isfinite(self.onset), self.onset, 1e9)[ai])).astype(np.float32)
        wt = np.where(self.onset_ok[ai], np.where(yt > 0, 2.0, 1.0), 0.0).astype(np.float32)
        self.ai, self.tpos, self.yt_np, self.wt_np = ai, tpos, yt, wt
        self.yt = torch.from_numpy(np.append(yt, 0).astype(np.float32)).to(dev)
        self.wt = torch.from_numpy(np.append(wt, 0).astype(np.float32)).to(dev)
        self.mu = self.sd = None

    def standardise(self, tr):
        """Per-dimension mean and sd of the training agents' turns (float64 on the CPU)."""
        torch = _torch()
        rows = np.concatenate([np.arange(self.start[i], self.start[i] + self.n[i]) for i in tr])
        s1 = np.zeros(self.X.shape[1])
        s2 = np.zeros(self.X.shape[1])
        for c in range(0, len(rows), 20000):
            b = self.X[torch.from_numpy(rows[c:c + 20000]).to(self.dev)].float().cpu().numpy().astype(np.float64)
            s1 += b.sum(0)
            s2 += (b * b).sum(0)
        mu = s1 / len(rows)
        sd = np.sqrt(np.maximum(s2 / len(rows) - mu * mu, 0)) + 1e-3
        self.mu = torch.from_numpy(mu.astype(np.float32)).to(self.dev)
        self.sd = torch.from_numpy(sd.astype(np.float32)).to(self.dev)

    def batch(self, b):
        torch = _torch()
        n = self.n[b]
        P = np.full((len(b), n.max()), self.N, np.int64)
        for j, i in enumerate(b):
            P[j, :n[j]] = self.start[i] + np.arange(n[j])
        Pt = torch.from_numpy(P).to(self.dev)
        z = (self.X[Pt].float() - self.mu) / self.sd
        return z, Pt == self.N, self.yt[Pt], self.wt[Pt], torch.from_numpy(n - 1).to(self.dev), \
            torch.from_numpy(self.y[b].astype(np.float32)).to(self.dev)


def length_batches(idx, n, bs, rng):
    """Batches of agents of similar length (sorted by turn count, random tie order, batch order shuffled when rng is
    given), so a batch is padded to about its own length rather than to the longest agent of the set."""
    idx = np.asarray(idx)
    key = rng.random(len(idx)) if rng is not None else np.zeros(len(idx))
    o = idx[np.lexsort((key, n[idx]))]
    bt = [o[s:s + bs] for s in range(0, len(o), bs)]
    if rng is not None:
        bt = [bt[i] for i in rng.permutation(len(bt))]
    return bt


def seq_loss(logit, pad, yt, wt, last, yfin):
    torch = _torch()
    F = torch.nn.functional
    ar = torch.arange(logit.shape[0], device=logit.device)
    lt = (F.binary_cross_entropy_with_logits(logit, yt, reduction="none") * wt).sum()
    lf = F.binary_cross_entropy_with_logits(logit[ar, last], yfin, reduction="sum")
    return (lt + lf) / (wt.sum() + len(yfin))


def train_seq(kind, sd_, tr, va, seed, label, max_epochs=30, patience=5, bs=32, lr=1e-3, wd=0.1):
    torch = _torch()
    model = build_net(kind, sd_.X.shape[1], seed, label).to(sd_.dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep_best = np.inf, None, 0, 0
    for ep in range(max_epochs):
        model.train()
        for b_ in length_batches(tr, sd_.n, bs, rng):
            z, pad, yt, wt, last, yfin = sd_.batch(b_)
            loss = seq_loss(model(z, pad), pad, yt, wt, last, yfin)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0.0
            for b_ in length_batches(va, sd_.n, 32, None):
                z, pad, yt, wt, last, yfin = sd_.batch(b_)
                vl += float(seq_loss(model(z, pad), pad, yt, wt, last, yfin)) * (float(wt.sum()) + len(yfin))
                vn += float(wt.sum()) + len(yfin)
        vl /= max(vn, 1e-9)
        if vl < best - 1e-4:
            best, bad, ep_best = vl, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, ep_best + 1


def predict_seq(model, sd_, idx):
    """Per-turn logits of the agents idx, concatenated in sd_ row order of those agents."""
    torch = _torch()
    out = {}
    with torch.no_grad():
        for b in length_batches(np.asarray(idx), sd_.n, 32, None):
            z, pad, *_ = sd_.batch(b)
            lg = model(z, pad).float().cpu().numpy()
            for j, i in enumerate(b):
                out[i] = lg[j, :sd_.n[i]].astype(np.float64)
    return out


def val_split_runs(tr, groups, seed):
    ur = np.unique(groups[tr])
    rng = np.random.default_rng(100 + seed)
    va_runs = set(rng.choice(ur, max(1, int(round(0.2 * len(ur)))), replace=False))
    va = np.array([i for i in tr if groups[i] in va_runs])
    return np.array([i for i in tr if groups[i] not in va_runs]), va


def step_metrics(turn_scores, sd_, te):
    """Turn AUROC on the held-out agents with a usable onset, the 5%-FPR threshold from their negative turns, and the
    detection delay (first turn above it minus the onset turn) of the held-out agents that reach the label."""
    keep = [i for i in te if sd_.onset_ok[i]]
    if not keep:
        return {}
    s = np.concatenate([turn_scores[i] for i in keep])
    y = np.concatenate([sd_.yt_np[sd_.start[i]:sd_.start[i] + sd_.n[i]] for i in keep])
    if y.min() == y.max():
        return {"turn_auroc": np.nan}
    thr = float(np.percentile(s[y == 0], 95))
    d = []
    for i in keep:
        if sd_.y[i] == 1:
            hit = np.flatnonzero(turn_scores[i] > thr)
            if len(hit):
                d.append(hit[0] - sd_.onset[i])
    d = np.array(d, float)
    return {"turn_auroc": tp.auc(s, y), "turns": len(y), "pos_turns": int(y.sum()),
            "reaching": int(sum(sd_.y[i] == 1 for i in keep)), "detected": len(d),
            "delay_median": float(np.median(d)) if len(d) else np.nan,
            "fired_before_onset": int((d < 0).sum()) if len(d) else 0}


def fit_linear_step(D_, S, sd_, tr, te):
    """Architecture 7: training-turn PCA (streamed), z-scoring, L2 logistic on the turns of training agents with a usable
    onset (C by inner CV grouped by run); per-step score = running mean of the turn scores; trajectory = last value."""
    gr = D_.A.group.values[sd_.ii]
    trk = np.array([i for i in tr if sd_.onset_ok[i]])
    rows_tr = np.concatenate([np.arange(sd_.start[i], sd_.start[i] + sd_.n[i]) for i in trk])
    rows_te = np.concatenate([np.arange(sd_.start[i], sd_.start[i] + sd_.n[i]) for i in te])
    fold = np.r_[np.ones(len(rows_tr), int), np.zeros(len(rows_te), int)]
    P = step_pca(S, sd_.R[np.concatenate([rows_tr, rows_te])], fold, which=[0])[0]
    assert (P[0] == np.arange(len(rows_tr))).all() and (P[1] == len(rows_tr) + np.arange(len(rows_te))).all()
    ytr = sd_.yt_np[rows_tr].astype(int)
    s, C = tp.fit_score(P[2], ytr, gr[sd_.ai[rows_tr]], np.vstack([P[3], P[2]]))
    if s is None:  # every training turn in one class: the positives of this split carry no per-turn onset
        return None
    st, str_ = s[:len(rows_te)], s[len(rows_te):]
    out, out_tr = {}, {}
    for src_, idx, dst in ((st, te, out), (str_, trk, out_tr)):
        o = 0
        for i in idx:
            v = src_[o:o + sd_.n[i]]
            dst[i] = np.cumsum(v) / np.arange(1, len(v) + 1)
            o += sd_.n[i]
    return out, out_tr, C, int(P[2].shape[1] + 1)


def fit_mlp_traj(D_, ii, tr, te, y, groups, seed, max_epochs=300, patience=20):
    """Architecture 8: training-fold PCA to 256 of the pooled all-layers vector, z-scoring, 2-layer MLP (256-128-1,
    GELU, dropout 0.3, AdamW weight decay 0.1), early stopping on held-out training runs."""
    torch = _torch()
    nn = torch.nn
    trn, va = val_split_runs(tr, groups, seed)
    Ztr, Zrest = tp.pca_z(D_.X("think", ii[trn]), D_.X("think", ii[np.concatenate([va, te, trn])]), kmax=256)
    Zva, Zte, Ztr2 = Zrest[:len(va)], Zrest[len(va):len(va) + len(te)], Zrest[len(va) + len(te):]
    torch.manual_seed(seed)
    k = Ztr.shape[1]
    m = nn.Sequential(nn.Linear(k, H_), nn.GELU(), nn.Dropout(0.3), nn.Linear(H_, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
    T = lambda a: torch.from_numpy(np.asarray(a, np.float32))  # noqa: E731
    Xt, yt_, Xv, yv = T(Ztr), T(y[trn]), T(Zva), T(y[va])
    pw = torch.tensor(float((1 - y[trn]).sum() / max(y[trn].sum(), 1)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(seed)
    best, bs_, bad, epb = np.inf, None, 0, 0
    for ep in range(max_epochs):
        m.train()
        p = rng.permutation(len(trn))
        for s in range(0, len(p), 64):
            j = p[s:s + 64]
            loss = lossf(m(Xt[j]).squeeze(-1), yt_[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lossf(m(Xv).squeeze(-1), yv))
        if vl < best - 1e-4:
            best, bad, epb = vl, 0, ep
            bs_ = {kk: v.clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(bs_)
    m.eval()
    with torch.no_grad():
        ste = m(T(Zte)).squeeze(-1).numpy().astype(np.float64)
        sttr = m(T(Ztr2)).squeeze(-1).numpy().astype(np.float64)
    return ste, sttr, trn, n_params(m), epb + 1


def arch_scheme_index(A, label, srcs):
    return np.flatnonzero((A.source.isin(srcs) & (A.ok == 1) & A[label].notna()).values)


def arch_splits(A, label, ii, ptype, te_srcs=None):
    """The same splits as the linear matrix: B.I = team_probes folds (stratified by source, author, label; grouped by
    run; seed 0); B.II = one split per held-out author with that author's runs out of training; transfer = sources."""
    y = A[label].values[ii].astype(int)
    g = A.group.values[ii]
    au = A.author.values[ii]
    src = A.source.values[ii]
    if te_srcs is not None:
        te = np.flatnonzero(np.isin(src, te_srcs))
        return [("pooled", [(np.flatnonzero(~np.isin(src, te_srcs)), te)])]
    if ptype == "B.I":
        if not feasible(y, g):
            return []
        fold = tp.folds_for(np.array([f"{a}|{b}|{c}" for a, b, c in zip(src, au, y)]), g, seed=0)
        return [("pooled", [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(tp.K)])]
    out = []
    if len(np.unique(au)) < 2:
        return out
    for a in sorted(np.unique(au)):
        te = np.flatnonzero(au == a)
        tr = np.flatnonzero((au != a) & ~np.isin(g, list(set(g[te]))))
        if min(int(y[te].sum()), int((1 - y[te]).sum())) >= MIN_CLASS and min(int(y[tr].sum()), int((1 - y[tr]).sum())) >= MIN_CLASS:
            out.append((a, [(tr, te)]))
    return out


def stack_scores(D_, S, sd_, ii, tr, te, y, g, label, seq_te):
    """Architecture 9 on one outer split: inner 3-fold CV (grouped by run) on the outer training agents gives
    out-of-fold scores of 1 (linear pooled), 2 (MLP-pool, seed 0), 5 (attention monitor, seed 0) and TF-IDF; a logistic
    regression on those four standardised columns is applied to the four outer test scores (each component fitted on
    the whole outer training set). The outer test agents are never seen by the stacker."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    H = D_.H
    fi = D_.A.fi.values

    def comps(a_tr, a_te, have=None):
        Ztr, Zte = tp.pca_z(D_.X("think", ii[a_tr]), D_.X("think", ii[a_te]))
        c1, _ = tp.fit_score(Ztr, y[a_tr], g[a_tr], Zte)
        c4, _ = tfidf_fit(H[fi[ii[a_tr]]], y[a_tr], g[a_tr], H[fi[ii[a_te]]])
        if have is not None:
            c2, c5 = have
        else:
            cs = []
            for kind in ("mlp_mean", "attn_monitor"):
                trn, va = val_split_runs(a_tr, g, 0)
                sd_.standardise(trn)
                m, _ = train_seq(kind, sd_, trn, va, 0, label)
                p = predict_seq(m, sd_, a_te)
                cs.append(np.array([p[i][-1] for i in a_te]))
            c2, c5 = cs
        return np.column_stack([c1, c2, c5, c4])
    oof = np.full((len(tr), 4), np.nan)
    sp = StratifiedGroupKFold(3, shuffle=True, random_state=2).split(np.zeros(len(tr)), y[tr], g[tr])
    for itr, ite in sp:
        oof[ite] = comps(tr[itr], tr[ite])
    outer = comps(tr, te, have=seq_te)
    mu, sd = oof.mean(0), oof.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit((oof - mu) / sd, y[tr])
    return lr.decision_function((outer - mu) / sd), lr.decision_function((oof - mu) / sd), lr.coef_[0].tolist()


def run_arch_one(D_, S, sd_, label, scheme, ptype, unit, splits, ii, arch, seq_seed0=None):
    """One architecture (every seed) on one unit of a cell: a list of train/test splits over the scheme's local index.
    seq_seed0, for the stack only: {local index: seed-0 trajectory score} of architectures 2 and 5 on the test agents."""
    A = D_.A
    y = A[label].values[ii].astype(int)
    g = A.group.values[ii]
    au = A.author.values[ii]
    te_all = np.concatenate([te for _, te in splits])
    fold = np.full(len(ii), -1)
    for k, (_, te) in enumerate(splits):
        fold[te] = k if len(splits) > 1 else -1
    rows, frames = [], []
    base = {"label": label, "scheme": scheme, "ptype": ptype, "unit": unit, "arch": arch}

    def add(seed, traj, turn_sc, train_aucs, params, epochs, secs, extra=None):
        sc = np.array([traj[i] for i in te_all])
        yy, gg = y[te_all], g[te_all]
        a = tp.auc(sc, yy)
        lo, hi = tp.boot_ci(sc, yy, gg) if np.isfinite(a) else (np.nan, np.nan)
        r = dict(base, seed=str(seed), traj_auroc=a, ci_lo=lo, ci_hi=hi, n=len(te_all), pos=int(yy.sum()),
                 runs=len(np.unique(gg)), authors=len(np.unique(au[te_all])), train_auroc=float(np.nanmean(train_aucs)),
                 gap=float(np.nanmean(train_aucs)) - a, params=params, epochs=epochs, seconds=round(secs, 1))
        if turn_sc is not None:
            r.update(step_metrics(turn_sc, sd_, te_all))
        r.update(extra or {})
        rows.append(r)
        frames.append({"label": label, "scheme": scheme, "ptype": ptype, "unit": unit, "arch": arch, "seed": str(seed),
                       "uid": A.uid.values[ii[te_all]], "y": yy, "fold": fold[te_all], "score": sc})
        log(f"   {label} {scheme} {ptype} {unit} {arch} seed {seed}: traj {a:.3f} turn {r.get('turn_auroc', np.nan):.3f} "
            f"gap {r['gap']:.3f} ({secs:.0f}s)")
    if arch == "stack":
        t0 = time.time()
        traj, tr_auc, coefs = {}, [], []
        for tr, te in splits:
            st, s_oof, cf = stack_scores(D_, S, sd_, ii, tr, te, y, g, label,
                                         (np.array([seq_seed0["mlp_mean"][int(i)] for i in te]),
                                          np.array([seq_seed0["attn_monitor"][int(i)] for i in te])))
            for i, v in zip(te, st):
                traj[i] = v
            tr_auc.append(tp.auc(s_oof, y[tr]))
            coefs.append(cf)
        add(0, traj, None, tr_auc, 5, None, time.time() - t0,
            {"note": "train AUROC = stacker on its inner out-of-fold scores; weights (linear, MLP-pool, attention, TF-IDF) "
                     + json.dumps(np.round(np.mean(coefs, 0), 3).tolist())})
        return rows, frames
    seeds = SEEDS if arch in SEQ or arch == "mlp_traj" else (0,)
    for seed in seeds:
        t0 = time.time()
        traj, turns, tr_auc, eps, params = {}, {}, [], [], None
        for tr, te in splits:
            if arch in SEQ:
                trn, va = val_split_runs(tr, g, seed)
                sd_.standardise(trn)
                m, ep = train_seq(arch, sd_, trn, va, seed, label)
                params = n_params(m)
                pr = predict_seq(m, sd_, np.concatenate([te, trn]))
                for i in te:
                    turns[i], traj[i] = pr[i], pr[i][-1]
                tr_auc.append(tp.auc(np.array([pr[i][-1] for i in trn]), y[trn]))
                eps.append(ep)
                del m
            elif arch == "linear_step":
                res_ = fit_linear_step(D_, S, sd_, tr, te)
                if res_ is None:
                    rows.append(dict(base, seed=str(seed), n=len(te_all), pos=int(y[te_all].sum()),
                                     note="no feasible split: the training turns carry one class (positives without a "
                                          "per-turn onset)"))
                    log(f"   {label} {scheme} {ptype} {unit} linear_step: training turns carry one class, skipped")
                    return rows, frames
                o, o_tr, C, params = res_
                for i in te:
                    turns[i], traj[i] = o[i], o[i][-1]
                tr_auc.append(tp.auc(np.array([v[-1] for v in o_tr.values()]), y[list(o_tr)]))
            elif arch == "mlp_traj":
                ste, sttr, trn, params, ep = fit_mlp_traj(D_, ii, tr, te, y, g, seed)
                for i, v in zip(te, ste):
                    traj[i] = v
                tr_auc.append(tp.auc(sttr, y[trn]))
                eps.append(ep)
        add(seed, traj, turns if turns else None, tr_auc, params, float(np.mean(eps)) if eps else None, time.time() - t0)
    return rows, frames


def arch_plan(A):
    """(tier, label, scheme name, ptype, train sources, test sources or None, stack?) in the order of the fallback:
    ours and all (B.I and B.II, with the stack), the per-source B.II cells, then each source alone B.I, then the
    leave-one-source-out B.I cells."""
    plan = []
    for label in LABELS:
        sch = {s[0]: s for s in schemes(A, label)}
        for nm in ("ours", "all"):
            s = sch[nm]
            if label != "cheat" and nm == "all":
                continue  # identical to ours for the board labels
            for pt in ("B.I", "B.II"):
                plan.append((1, label, nm, pt, s[2], None, pt == "B.I"))  # the stack on the B.I cells only
    for label in LABELS:
        for s in schemes(A, label):
            if s[1] == "alone":
                plan.append((2, label, s[0], "B.II", s[2], None, False))
    for label in LABELS:
        for s in schemes(A, label):
            if s[1] == "alone":
                plan.append((3, label, s[0], "B.I", s[2], None, False))
    for label in LABELS:
        for s in schemes(A, label):
            if s[1] == "loso":
                plan.append((4, label, s[0], "B.I", s[2], s[3], False))
    return plan


def archs(hours=6.0, only=None, deadline=None):
    """Architecture-major: each architecture in the order 2 to 9 on the priority cells (tier 1: ours and all, B.I and
    B.II; tier 2: each source alone, B.II), then each architecture on the remaining cells (tier 3: each source alone,
    B.I; tier 4: leave one source out, B.I). A (cell, architecture) starts only while the time budget lasts; resumable."""
    import gc
    import pandas as pd
    D_ = Data()
    A = D_.A
    S = np.load(CACHE / "step_L38.npy", mmap_mode="r")
    dev = arch_device()
    budget_end = deadline if deadline else T0 + hours * 3600
    ARCH_OUT.mkdir(parents=True, exist_ok=True)
    cp = ARCH_OUT / "arch_cells.csv"
    done = pd.read_csv(cp) if cp.is_file() else pd.DataFrame(columns=["label", "scheme", "ptype", "arch"])
    done = done[~done.note.astype(str).str.startswith("not run")] if "note" in done else done
    allrows = [done] if len(done) else []
    have = set(zip(done.label, done.scheme, done.ptype, done.arch))
    plan = arch_plan(A)
    json.dump([dict(zip(["tier", "label", "scheme", "ptype", "train", "test", "stack"], p)) for p in plan],
              open(ARCH_OUT / "arch_plan.json", "w"), indent=1)
    order = [a for a in ARCHS if a != "linear"]
    groups = [[p for p in plan if p[0] in (1, 2)], [p for p in plan if p[0] in (3, 4)]]
    cur, sd_ = None, None
    n_all = sum(len(gp) for gp in groups) * len(order)
    k = 0
    for gp in groups:
        for arch in order:
            for tier, label, scheme, ptype, tr_s, te_s, stk in gp:
                k += 1
                if (only and only not in scheme) or (arch == "stack" and not stk) or (label, scheme, ptype, arch) in have:
                    continue
                base = {"label": label, "scheme": scheme, "ptype": ptype, "arch": arch, "tier": tier}
                if time.time() > budget_end:
                    allrows.append(pd.DataFrame([dict(base, unit="all", note="not run: time budget reached")]))
                    pd.concat(allrows, ignore_index=True).to_csv(cp, index=False)
                    continue
                srcs = list(tr_s) + list(te_s or [])
                ii = arch_scheme_index(A, label, srcs)
                assert (A.s0.values[ii] >= 0).all()
                units = arch_splits(A, label, ii, ptype, te_s)
                if not units:
                    allrows.append(pd.DataFrame([dict(base, unit="", note="no feasible split (one author, or too few in a class)")]))
                    pd.concat(allrows, ignore_index=True).to_csv(cp, index=False)
                    continue
                seq0 = None
                if arch == "stack":
                    seq0 = {}
                    for comp in ("mlp_mean", "attn_monitor"):
                        f = ARCH_OUT / "scores" / label / f"{slug(scheme)}__{ptype.replace('.', '')}__{comp}.csv.gz"
                        if not f.is_file():
                            break
                        sc = pd.read_csv(f)
                        sc = sc[sc.seed.astype(str) == "0"]
                        pos = {u: j for j, u in enumerate(A.uid.values[ii])}
                        seq0[comp] = {pos[u]: v for u, v in zip(sc.uid, sc.score)}
                    if len(seq0) < 2:
                        allrows.append(pd.DataFrame([dict(base, unit="all", note="not run: components 2 and 5 missing")]))
                        continue
                if cur != (label, tuple(srcs)):
                    sd_ = None
                    gc.collect()
                    sd_ = SeqData(D_, S, ii, label, dev)
                    cur = (label, tuple(srcs))
                status("archs", f"{k}/{n_all} (tier {tier}) {arch}: {LABELS[label]}, {scheme}, {ptype}; {len(ii)} agents, "
                                f"{sd_.N} turns; {max(0, budget_end - time.time()) / 3600:.2f} h of budget left")
                log(f"== archs {arch} tier {tier}: {label} {scheme} {ptype} ({len(ii)} agents, {sd_.N} turns)")
                rows, frames = [], []
                for unit, splits in units:
                    r_, f_ = run_arch_one(D_, S, sd_, label, scheme, ptype, unit, splits, ii, arch, seq0)
                    rows += r_
                    frames += f_
                for r in rows:
                    r["tier"] = tier
                allrows.append(pd.DataFrame(rows))
                pd.concat(allrows, ignore_index=True).to_csv(cp, index=False)
                save_frames(frames, ARCH_OUT / "scores" / label / f"{slug(scheme)}__{ptype.replace('.', '')}__{arch}.csv.gz")
    status("archs", "done (or budget reached; see arch_cells.csv notes)")


def arch_summary(AC, cells, label, scheme, ptype):
    """{arch: dict} for one (label, scheme, probe type): trajectory AUROC as the mean over seeds with the seed range
    (for B.II the per-seed mean over held-out authors first), turn AUROC, delay, gap, parameters."""
    out = {}
    c = AC[(AC.label == label) & (AC.scheme == scheme) & (AC.ptype == ptype)]
    for arch in ARCHS:
        if arch == "linear":
            r = cells[(cells.label == label) & (cells.scheme == scheme) & (cells.ptype == ptype) & (cells.feature == "think") &
                      (cells.unit == ("pooled" if ptype == "B.I" else "mean over authors"))]
            if len(r) and np.isfinite(r.auroc.iloc[0]):
                out[arch] = {"auc": r.auroc.iloc[0], "lo": r.auroc.iloc[0], "hi": r.auroc.iloc[0], "ci": (r.ci_lo.iloc[0], r.ci_hi.iloc[0]),
                             "n": r.n.iloc[0], "pos": r.pos.iloc[0], "authors": r.authors.iloc[0], "seeds": 1, "turn": np.nan,
                             "delay": np.nan, "gap": np.nan, "params": np.nan}
            continue
        a = c[(c.arch == arch) & c.traj_auroc.notna()] if "traj_auroc" in c else c.iloc[0:0]
        if not len(a):
            continue
        per_seed = a.groupby("seed").agg(auc=("traj_auroc", "mean"), turn=("turn_auroc", "mean"), gap=("gap", "mean"),
                                         delay=("delay_median", "mean"), n=("n", "sum"), pos=("pos", "sum"),
                                         units=("unit", "nunique"))
        out[arch] = {"auc": per_seed.auc.mean(), "lo": per_seed.auc.min(), "hi": per_seed.auc.max(),
                     "ci": (a.ci_lo.iloc[0], a.ci_hi.iloc[0]) if ptype == "B.I" else (np.nan, np.nan),
                     "n": per_seed.n.iloc[0], "pos": per_seed.pos.iloc[0], "authors": int(per_seed.units.iloc[0]),
                     "seeds": len(per_seed), "turn": per_seed.turn.mean(), "delay": per_seed.delay.mean(),
                     "gap": per_seed.gap.mean(), "params": a.params.dropna().iloc[0] if a.params.notna().any() else np.nan}
    return out


def arch_ptype_table(AC, cells, plan_keys, label, scheme):
    """One scheme: architecture rows by probe-type columns (A, B.I, B.II), the linear null and the two baselines below."""
    by = {pt: arch_summary(AC, cells, label, scheme, pt) for pt in ("B.I", "B.II")}
    c = cells[(cells.label == label) & (cells.scheme == scheme)]

    def lin(pt, feat="think"):
        r = c[(c.ptype == pt) & (c.unit == ("pooled" if pt == "B.I" else "mean over authors")) & (c.feature == feat)]
        return r.iloc[0].to_dict() if len(r) else {}
    rows = []
    for a in ARCHS:
        if a == "linear":
            rows.append([ARCH_SHORT[a]] + [fci(lin(pt)) or "n/a" for pt in ("A", "B.I", "B.II")])
            continue
        cols = ["not planned"]
        for pt in ("B.I", "B.II"):
            d = by[pt].get(a)
            planned = (label, scheme, pt) in plan_keys and (a != "stack" or pt == "B.I")
            if a == "stack" and planned:
                planned = scheme in ("ours", "all")
            if d:
                s = f3(d["auc"]) + (f" ({f3(d['lo'])} to {f3(d['hi'])})" if d.get("seeds", 1) > 1 else "")
            elif not planned:
                s = "not planned"
            else:
                r = AC[(AC.label == label) & (AC.scheme == scheme) & (AC.ptype == pt) & (AC.arch == a)] if len(AC) else AC
                nt = str(r.note.dropna().iloc[0]) if len(r) and "note" in r and r.note.notna().any() else ""
                s = "no feasible split" if nt.startswith("no feasible") else "not reached"
            cols.append(s)
        if all(x == "not planned" for x in cols):
            continue
        rows.append([ARCH_SHORT[a]] + cols)
    rows.append(["linear: permutation null 95th"] + [f3(lin(pt).get("null_95")) for pt in ("A", "B.I", "B.II")])
    for f in ("tfidf", "meta"):
        rows.append([f"baseline: {BASES[f]}"] + [f3(lin(pt, f).get("auroc")) for pt in ("A", "B.I", "B.II")])
    return md_table(["architecture", "A (mean over authors)", "B.I", "B.II (mean over held-out authors)"], rows)


def arch_report_lines(cells):
    import pandas as pd
    p = ARCH_OUT / "arch_cells.csv"
    if not p.is_file():
        return []
    AC = pd.read_csv(p)
    plan = json.load(open(ARCH_OUT / "arch_plan.json"))
    L = []
    w = L.append
    w("## Architectures\n")
    w("A third axis: the probe architecture, on the main cells, with the same folds and held-out-author splits as the "
      "linear matrix (so the TF-IDF and metadata baselines of the linear tables apply unchanged). Per-step input: the "
      "layer-38 think pair of every turn, standardised with the training turns. Architectures 2 to 6 share one "
      "training objective, the binary part of the attention monitor's: per-turn cross-entropy (0 before the onset, 1 "
      "from it on, weight 1 and 2; weight 0 on the turns of an agent that reaches the label with no usable onset) plus "
      "the trajectory label at the last step; AdamW (lr 1e-3, weight decay 0.1), batch 32, at most 30 epochs, early "
      "stopping (patience 5) on 20% of the training runs held out. Their trajectory score is the last step's output, "
      "their per-step score the output at each step. Architecture 7 trains on the turns of training agents with a usable "
      "onset, so positives without one (most of CheatBench, all of EvilGenie) leave its training set; where no "
      "training positive has an onset it cannot run. "
      "Architecture 8 trains on agents (positive class weighted to balance). Trained models (2 to 6, 8) ran with two "
      "seeds; the table gives the mean over seeds and the seed range. Gap: AUROC on the training agents (fit set) "
      "minus held-out AUROC. Turn AUROC and the detection delay (median turns from onset to the first score above the "
      "5% false-positive threshold of held-out negative turns) are for per-step models, on held-out agents with a "
      "usable onset. Architecture 9 stacks 1, 2, 5 and TF-IDF with a logistic regression fitted on inner 3-fold "
      "out-of-fold scores of the outer training agents only. No permutation null was run for architectures 2 to 9.\n")
    for k, v in ARCH_NAMES.items():
        w(f"- **{k}**: {v}")
    w("")
    keys = list(dict.fromkeys((x["label"], x["scheme"]) for x in plan))
    summ = []
    for label, scheme in keys:
        bi = arch_summary(AC, cells, label, scheme, "B.I")
        b2 = arch_summary(AC, cells, label, scheme, "B.II")
        if not bi and not b2:
            continue
        order = sorted(ARCHS, key=lambda a: -(bi.get(a, {}).get("auc", np.nan) if np.isfinite(bi.get(a, {}).get("auc", np.nan))
                                              else b2.get(a, {}).get("auc", -1) - 1))
        rows = []
        for a in order:
            x, z = bi.get(a, {}), b2.get(a, {})
            if not x and not z:
                continue

            def rng(d):
                if not d:
                    return "not run"
                s = f3(d["auc"])
                return s + (f" ({f3(d['lo'])} to {f3(d['hi'])})" if d.get("seeds", 1) > 1 else "")
            rows.append([a, rng(x), f"[{f3(x['ci'][0])}, {f3(x['ci'][1])}]" if x else "", f3(x.get("turn")),
                         f"{x['delay']:.1f}" if x and np.isfinite(x.get("delay", np.nan)) else "", f3(x.get("gap")),
                         rng(z), f3(z.get("turn")), f3(z.get("gap")), fint(z.get("authors")),
                         fint(x.get("params") if x else z.get("params"))])
            summ.append({"label": label, "scheme": scheme, "arch": a, "B.I": x.get("auc", np.nan), "B.II": z.get("auc", np.nan)})
        c = cells[(cells.label == label) & (cells.scheme == scheme) & (cells.ptype == "B.I") & (cells.unit == "pooled")]
        bb = {r["feature"]: r for r in c.to_dict("records")}
        c2 = cells[(cells.label == label) & (cells.scheme == scheme) & (cells.ptype == "B.II") & (cells.unit == "mean over authors")]
        b2b = {r["feature"]: r for r in c2.to_dict("records")}
        for f in ("tfidf", "meta"):
            rows.append([f"baseline: {BASES[f]}", f3(bb.get(f, {}).get("auroc")), "", "", "", "", f3(b2b.get(f, {}).get("auroc")),
                         "", "", "", ""])
        n_ = bb.get("think", {})
        w(f"### {LABELS[label]}: {scheme}\n")
        w(f"B.I counts: {fint(n_.get('n'))} agents, {fint(n_.get('pos'))} positive, {fint(n_.get('runs'))} runs, "
          f"{fint(n_.get('authors'))} authors. Rows ranked by B.I trajectory AUROC.\n")
        w(md_table(["architecture", "B.I trajectory AUROC (seed range)", "B.I 95% CI (seed 0)", "B.I turn AUROC",
                    "B.I delay (turns)", "B.I gap", "B.II mean trajectory AUROC (seed range)", "B.II turn AUROC",
                    "B.II gap", "B.II authors", "parameters"], rows))
        w("")
    if summ:
        S_ = pd.DataFrame(summ)
        S_["cell"] = [f"{LABELS[a][:5]}: {b}" for a, b in zip(S_.label, S_.scheme)]
        for pt in ("B.I", "B.II"):
            pv = S_.pivot_table(index="arch", columns="cell", values=pt, aggfunc="first")
            pv = pv.reindex([a for a in ARCHS if a in pv.index])
            pv = pv[[c for c in dict.fromkeys(S_.cell) if c in pv.columns and pv[c].notna().any()]]
            if pv.shape[1]:
                w(f"### Summary across schemes: {pt} trajectory AUROC (mean over seeds{', mean over held-out authors' if pt == 'B.II' else ''})\n")
                w(md_table(["architecture"] + list(pv.columns), [[a] + [f3(v) for v in pv.loc[a]] for a in pv.index]))
                w("")
    nr = AC[AC.note.astype(str).str.startswith("not run") | AC.note.astype(str).str.startswith("no feasible")] if "note" in AC else AC.iloc[0:0]
    ran = set(zip(AC.label, AC.scheme, AC.ptype))
    missing = [x for x in plan if (x["label"], x["scheme"], x["ptype"]) not in ran]
    lines = [f"- {LABELS[r.label]}, {r.scheme}, {r.ptype}, " + (f"held-out {r.unit}, " if isinstance(r.unit, str) and
             r.unit not in ("pooled", "all") else "") + f"{ARCH_SHORT[r.arch]}: {r.note}" for r in nr.itertuples()]
    n_budget = int(AC.note.astype(str).str.startswith("not run").sum()) if "note" in AC else 0
    w("### Architecture cells with no result\n")
    w(f"Cells cut by the time budget: {n_budget}. Every other planned cell ran or is listed here with the reason it "
      "could not run.\n")
    lines += [f"- {LABELS[x['label']]}, {x['scheme']}, {x['ptype']}: not reached" for x in missing]
    w("\n".join(lines) if lines else "None: every planned architecture cell ran.")
    w("\nNot planned for the architectures: the A probe type, the external and transfer schemes (ours to external, "
      "external to ours), B.II of the leave-one-source-out schemes, and the stack outside the B.I cells of ours and all "
      "(cheating) and ours (coordination, collusion).\n")
    return L


if __name__ == "__main__":
    main()
