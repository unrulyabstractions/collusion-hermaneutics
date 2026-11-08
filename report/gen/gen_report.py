"""Build report/REPORT_DRAFT.md, report/NUMBERS.md and report/ABSTRACT.txt from the result files.

Every number in a table is read from a file and written to the ledger with its selector. Prose numbers are ledgered
by hand at the bottom of this script.
"""
import csv
import glob
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # report/gen -> repo root
SUB = f"{REPO}/submodule/spar-guardian-agents"  # the repo submodule on branch board-seed-injection: read for the board-seed texts (Appendix C) and cited in the ledger
PILOT = f"{REPO}/data/board_seed_pilot"
HF_README = f"{REPO}/report/gen/sources/hf_README.md"  # tracked copy of the published dataset README (byte-identical to the HF upload)
OUT = f"{REPO}/report"

LEDGER = []


def led(value, where, source, selector):
    LEDGER.append(f"| {value} | {where} | `{source}` | {selector} |")


def rel(p):
    return os.path.relpath(p, REPO) if p.startswith(REPO) else p


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f3(x):
    return f"{float(x):.3f}"


AUTH = ["ds", "gpt5m", "haiku", "luna", "qwen"]
NAME = {"ds": "DeepSeek V4.1 Flash", "gpt5m": "GPT-5 mini", "haiku": "Claude Haiku 5.5", "luna": "GPT-6 Luna",
        "qwen": "Qwen3.8-27B"}

# ----------------------------------------------------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------------------------------------------------
PCA = f"{REPO}/hermeneutics/results/pca_agent/auroc.csv"
FULL = f"{REPO}/hermeneutics/results/full_agent/auroc.csv"
TURN = f"{REPO}/hermeneutics/results/pca_turn/auroc.csv"
ONSET = f"{REPO}/hermeneutics/results/pca_turn_onset/auroc.csv"
AUTHP = f"{REPO}/hermeneutics/results/pca_agent/author_probe.csv"
CONDP = f"{REPO}/hermeneutics/results/pca_agent/condition_probe.csv"
AGENTS = f"{REPO}/hermeneutics/results/pca_agent/agents.csv"
RESJ = f"{REPO}/hermeneutics/results/pca_agent/solo/Lall_both_agent/results.json"
RESMD = f"{REPO}/hermeneutics/results/RESULTS.md"
RENDER = f"{REPO}/data/render_summary.csv"
RUNS = f"{REPO}/data/runs.csv"
LABELS = f"{REPO}/data/labels.csv"
TB25 = f"{PILOT}/RESULTS.tsv"
TB60 = f"{PILOT}/RESULTS_tb60.tsv"
DOSE = f"{PILOT}/confirm_doseresponse/RESULTS_doseresponse.tsv"

pca = rows(PCA)
full = rows(FULL)
turn = rows(TURN)
onset = rows(ONSET)
authp = rows(AUTHP)
condp = rows(CONDP)
agents = rows(AGENTS)
resj = json.load(open(RESJ))
render = rows(RENDER)
runs = rows(RUNS)
labels = rows(LABELS)


def pick(tab, **kw):
    out = [r for r in tab if all(str(r[k]) == str(v) for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


def sel(**kw):
    return ",".join(f"{k}={v}" for k, v in kw.items())


def auc_cell(tab, path, where, ci=True, **kw):
    r = pick(tab, **kw)
    v = f3(r["auroc"])
    if ci:
        v = f"{v} [{f3(r['ci_lo'])}, {f3(r['ci_hi'])}]"
        led(v, where, rel(path), sel(**kw) + " -> auroc, ci_lo, ci_hi")
    else:
        led(v, where, rel(path), sel(**kw) + " -> auroc")
    return v, r


# ----------------------------------------------------------------------------------------------------------------
# Table M1: authors and reader
# ----------------------------------------------------------------------------------------------------------------
model_id = {}
for r in runs:
    model_id.setdefault(r["author"], set()).add(r["model_id"])
cov = {}
for a in AUTH:
    rs = [r for r in render if r["author"] == a]
    t = sum(int(r["turns"]) for r in rs)
    rd = sum(int(r["n_readable"]) for r in rs)
    cov[a] = (len(rs), t, rd)
valid = {}
for a in AUTH:
    v = [r for r in labels if r["author"] == a and r["valid"] == "True"]
    valid[a] = len(v)

WEIGHTS = {"ds": "closed", "gpt5m": "closed", "haiku": "closed", "luna": "closed", "qwen": "open"}
TM1 = ["| author | API model id | weights | valid runs | think coverage (share of turns with readable reasoning) |",
       "|---|---|---|---|---|"]
for a in AUTH:
    (mid,) = model_id[a]
    n, t, rd = cov[a]
    share = f"{rd / t:.3f}"
    led(mid, f"Table M1 row {NAME[a]} col API model id", rel(RUNS), f"distinct model_id where author={a}")
    led(valid[a], f"Table M1 row {NAME[a]} col valid runs", rel(LABELS), f"count rows where author={a} and valid=True")
    led(share, f"Table M1 row {NAME[a]} col think coverage", rel(RENDER),
        f"sum(n_readable)/sum(turns) over rows where author={a} ({rd}/{t})")
    TM1.append(f"| {NAME[a]} | `{mid}` | {WEIGHTS[a]} | {valid[a]} | {share} |")
TM1.append("| Reader: Qwen3.8-27B | `openrouter/qwen/qwen3.8-27b` (same model, run locally in bf16) | open | | |")
TM1 = "\n".join(TM1)

# ----------------------------------------------------------------------------------------------------------------
# Table M2: data per author x arm (full valid set and the probed subsample)
# ----------------------------------------------------------------------------------------------------------------
TM2 = ["| author | arm | valid runs: cheat / no cheat / other | probed agents: cheat / no cheat | probed turns | probed turns with readable reasoning |",
       "|---|---|---|---|---|---|"]
for a in AUTH:
    for cond in ("solo", "cheat"):
        v = [r for r in labels if r["author"] == a and r["valid"] == "True" and r["condition"] == cond]
        c = sum(1 for r in v if r["label"] == "cheat")
        nc = sum(1 for r in v if r["label"] == "no cheat")
        ot = sum(1 for r in v if r["label"] == "other")
        s = [r for r in agents if r["author"] == a and r["condition"] == cond]
        sc = sum(1 for r in s if r["y"] == "1")
        snc = sum(1 for r in s if r["y"] == "0")
        st = sum(int(r["n_turns"]) for r in s)
        sr = sum(int(r["n_readable"]) for r in s)
        where = f"Table M2 row {NAME[a]}/{cond}"
        led(f"{c} / {nc} / {ot}", where + " col valid runs", rel(LABELS),
            f"count label in (cheat, no cheat, other) where author={a}, condition={cond}, valid=True")
        led(f"{sc} / {snc}", where + " col probed agents", rel(AGENTS), f"count y==1 / y==0 where author={a}, condition={cond}")
        led(st, where + " col probed turns", rel(AGENTS), f"sum n_turns where author={a}, condition={cond}")
        led(sr, where + " col readable", rel(AGENTS), f"sum n_readable where author={a}, condition={cond}")
        TM2.append(f"| {NAME[a]} | {cond} | {c} / {nc} / {ot} | {sc} / {snc} | {st} | {sr} |")
TM2 = "\n".join(TM2)
n_agents = len(agents)
led(n_agents, "Table M3 probed runs", rel(AGENTS), "row count")
assert n_agents == 600

# ----------------------------------------------------------------------------------------------------------------
# Table M3: probe setup constants (from code and logs)
# ----------------------------------------------------------------------------------------------------------------
TM3 = """| setting | value |
|---|---|
| reader | Qwen3.8-27B, bf16, decoder blocks loaded one at a time |
| positions | residual stream at the think-open (id 248068) and think-close (id 248069) token of every assistant turn |
| layers | 26, 38, 51 of 64 blocks (0.41, 0.59, 0.80 of depth), and the three concatenated |
| pooling | mean over the agent's turns (agent level), or one row per turn (turn level) |
| probe | training-fold PCA to 256 components, z-scoring, L2 logistic regression, C by grouped inner cross-validation |
| variant | every dimension z-scored, no PCA |
| folds | 5, grouped by run, stratified by author x label |
| intervals | 95% bootstrap over runs, 1000 draws |
| null | labels shuffled inside each author, 100 draws |
| erasure | linear concept erasure (LEACE) of author; of author, task and condition; fitted on training rows |
| probed runs | 600 (about 120 per author, balanced over arm x label, runs over 100k tokens left out) |
| probed tokens / turns | 14,182,746 / 10,985 |
| extraction | one L40S GPU, 1.94 h, on a free host |"""
for v, w, s, q in [
    ("248068 / 248069", "Table M3 positions", "VERIFICATION_LOG.md", "07:20 entry: token ids checked against the tokenizer"),
    ("26, 38, 51 of 64", "Table M3 layers", "hermeneutics/results/full_agent/RESULTS.md", "header: HF hidden_states numbering, 64 blocks"),
    ("256", "Table M3 probe", "hermeneutics/probes.py", "PCA_DIM = 256 (line 48)"),
    ("5", "Table M3 folds", "hermeneutics/probes.py", "K = 5 (line 45)"),
    ("1000", "Table M3 intervals", "hermeneutics/probes.py", "N_BOOT = 1000 (line 47)"),
    ("100", "Table M3 null", "hermeneutics/results/RESULTS.md", "Table 4 caption '100 draws'; VERIFICATION_LOG 09:13 '100 shuffles'"),
    ("600 / 120 / 100k", "Table M3 probed runs", "VERIFICATION_LOG.md", "07:45 entry: --per-author 120 --max-tokens 100000 -> 600 runs"),
    ("14,182,746 / 10,985", "Table M3 probed tokens / turns", "VERIFICATION_LOG.md", "09:13 entry"),
    ("1.94 h, L40S", "Table M3 extraction", "VERIFICATION_LOG.md", "09:13 entry (elapsed 1.94 h); cloud-jobs memory: wild = L40S"),
]:
    led(v, w, s, q)

# ----------------------------------------------------------------------------------------------------------------
# Table 1: solo arm, PCA agent probe, layer all / both: B.I and A, erasures, null, metadata baseline
# ----------------------------------------------------------------------------------------------------------------
FEATS = ["raw", "author erased", "author, task, condition erased"]
base = dict(arm="solo", probe="pca", turn_labels="agent", layer="all", position="both", pooling="agent")
nuis = {k: v for k, v in resj["nuisance"].items()}


def nuis_val(key):
    v = nuis[key]
    return v["auroc"] if isinstance(v, dict) else v


TFIDF_EARLY = rows(f"{REPO}/hermeneutics/results/tfidf/auroc.csv")
def tfidf_solo(scope):
    r = [x for x in TFIDF_EARLY if x["arm"] == "solo" and x["position"] == "transcript" and x["scope"] == scope]
    assert len(r) == 1, scope
    return r[0]
T1 = ["| probe | cheat / no cheat | raw | author erased | author, task, condition erased | null 95th pct (raw) | null p | metadata-only baseline | TF-IDF, whole transcript, same folds |",
      "|---|---|---|---|---|---|---|---|---|"]
for scope, label, nk in [("pooled", "B.I pooled, held-out runs", "pooled (author + task + condition)")] + \
        [(a, f"A within {NAME[a]}", f"{a} (task + condition)") for a in AUTH]:
    cells = []
    for ft in FEATS:
        v, r = auc_cell(pca, PCA, f"Table 1 row '{label}' col '{ft}'", **base, features=ft, scope=scope)
        cells.append(v)
    r = pick(pca, **base, features="raw", scope=scope)
    n95, p = f3(r["null_95"]), f3(r["null_p"])
    led(n95, f"Table 1 row '{label}' col null 95th", rel(PCA), sel(**base, features="raw", scope=scope) + " -> null_95")
    led(p, f"Table 1 row '{label}' col null p", rel(PCA), sel(**base, features="raw", scope=scope) + " -> null_p")
    nb = f3(nuis_val(nk))
    led(nb, f"Table 1 row '{label}' col metadata baseline", rel(RESJ), f"nuisance['{nk}']")
    tf = tfidf_solo(scope)
    tfv = f"{f3(tf['auroc'])} [{f3(tf['ci_lo'])}, {f3(tf['ci_hi'])}]" if scope == "pooled" else f3(tf["auroc"])
    led(tfv, f"Table 1 row '{label}' col TF-IDF whole transcript", "hermeneutics/results/tfidf/auroc.csv", f"arm=solo,position=transcript,scope={scope} -> auroc" + (", ci" if scope == "pooled" else ""))
    T1.append(f"| {label} | {r['cheat']} / {r['no_cheat']} | {cells[0]} | {cells[1]} | {cells[2]} | {n95} | {p} | {nb} | {tfv} |")
    led(f"{r['cheat']} / {r['no_cheat']}", f"Table 1 row '{label}' col cheat / no cheat", rel(PCA),
        sel(**base, features="raw", scope=scope) + " -> cheat, no_cheat")
# full-dimension variant row and run-length baseline row
v, r = auc_cell(full, FULL, "Table 1 row 'B.I pooled, full-dimension variant' col raw", arm="solo", probe="full",
                turn_labels="agent", layer="all", position="close", pooling="agent", features="raw", scope="pooled")
v2, _ = auc_cell(full, FULL, "Table 1 row 'B.I pooled, full-dimension variant' col author erased", arm="solo", probe="full",
                 turn_labels="agent", layer="all", position="close", pooling="agent", features="author erased", scope="pooled")
v3, _ = auc_cell(full, FULL, "Table 1 row 'B.I pooled, full-dimension variant' col all erased", arm="solo", probe="full",
                 turn_labels="agent", layer="all", position="close", pooling="agent",
                 features="author, task, condition erased", scope="pooled")
T1.append(f"| B.I pooled, full-dimension variant (layer all, position close) | {r['cheat']} / {r['no_cheat']} | {v} | {v2} | {v3} | {f3(r['null_95'])} | {f3(r['null_p'])} | | |")
led(f3(r["null_95"]), "Table 1 full-dimension row col null 95th", rel(FULL),
    "arm=solo,probe=full,layer=all,position=close,features=raw,scope=pooled -> null_95")
led(f3(r["null_p"]), "Table 1 full-dimension row col null p", rel(FULL),
    "arm=solo,probe=full,layer=all,position=close,features=raw,scope=pooled -> null_p")
# run-length baseline from RESULTS.md Table 3 (computed by summary.py at build time)
md = open(RESMD).read()
m = re.search(r"cheating AUROC from run length \(turns, log tokens, readable share\) \| ([0-9.]+) \|", md)
runlen = m.group(1)
led(runlen, "Table 1 row 'run-length baseline'", rel(RESMD), "Table 3 last row (summary.py recomputes it from agents.csv at build time)")
m = re.search(r"cheating AUROC from one-hot pooled \(author only\) \| ([0-9.]+) \|", md)
authonly = m.group(1)
led(authonly, "Table 1 row 'author-only one-hot baseline'", rel(RESJ), "nuisance['pooled (author only)'] (also RESULTS.md Table 3)")
assert abs(float(authonly) - float(nuis_val("pooled (author only)"))) < 5e-4
T1.append(f"| baseline: one-hot author only, pooled | 193 / 180 | {authonly} | | | | | | |")
T1.append(f"| baseline: run length (turns, log tokens, readable share), pooled | 193 / 180 | {runlen} | | | | | | |")
T1 = "\n".join(T1)

# ----------------------------------------------------------------------------------------------------------------
# Table 2: B.II leave-one-author-out
# ----------------------------------------------------------------------------------------------------------------
T2 = ["| held-out author | think coverage | cheat / no cheat | PCA probe, raw | PCA, author erased | PCA, author, task, condition erased | full-dimension probe, raw | per-turn probe, agent AUROC | TF-IDF, whole transcript |",
      "|---|---|---|---|---|---|---|---|---|"]
for a in AUTH:
    cells = []
    for ft in FEATS:
        v, r = auc_cell(pca, PCA, f"Table 2 row {NAME[a]} col PCA '{ft}'", **base, features=ft, scope=f"LOTO {a}")
        cells.append(v)
    vf, _ = auc_cell(full, FULL, f"Table 2 row {NAME[a]} col full-dimension raw", arm="solo", probe="full", turn_labels="agent",
                     layer="all", position="close", pooling="agent", features="raw", scope=f"LOTO {a}")
    vt, _ = auc_cell(turn, TURN, f"Table 2 row {NAME[a]} col per-turn agent AUROC", arm="solo", probe="pca", turn_labels="agent",
                     layer="38", position="close", pooling="turn", features="raw", scope=f"LOTO {a}")
    n, t, rd = cov[a]
    tf = tfidf_solo(f"LOTO {a}")
    tfv = f"{f3(tf['auroc'])} [{f3(tf['ci_lo'])}, {f3(tf['ci_hi'])}]"
    led(tfv, f"Table 2 row {NAME[a]} col TF-IDF whole transcript", "hermeneutics/results/tfidf/auroc.csv", f"arm=solo,position=transcript,scope=LOTO {a} -> auroc, ci")
    T2.append(f"| {NAME[a]} | {rd / t:.3f} | {r['cheat']} / {r['no_cheat']} | {cells[0]} | {cells[1]} | {cells[2]} | {vf} | {vt} | {tfv} |")
T2 = "\n".join(T2)

# ----------------------------------------------------------------------------------------------------------------
# Table 3: author, task, instruction
# ----------------------------------------------------------------------------------------------------------------
T3 = ["| readout | arm | raw | author erased | author, task, condition erased | chance / majority rate |", "|---|---|---|---|---|---|"]
ap = {ft: pick(authp, arm="solo", layer="all", position="both", pooling="agent", features=ft) for ft in FEATS}
for ft in FEATS:
    led(f3(ap[ft]["accuracy"]), f"Table 3 row 'author identity' col '{ft}'", rel(AUTHP),
        f"arm=solo,layer=all,position=both,pooling=agent,features={ft} -> accuracy")
led(f3(ap["raw"]["majority_rate"]), "Table 3 row 'author identity' col majority", rel(AUTHP), "same row -> majority_rate")
T3.append(f"| author identity (5 classes), accuracy | solo | {f3(ap['raw']['accuracy'])} | {f3(ap['author erased']['accuracy'])} | {f3(ap['author, task, condition erased']['accuracy'])} | {f3(ap['raw']['majority_rate'])} |")
cp = {ft: pick(condp, arm="pooled", layer="38", position="close", pooling="agent", features=ft) for ft in FEATS}
for ft in FEATS:
    led(f3(cp[ft]["accuracy"]), f"Table 3 row 'prompt condition' col '{ft}'", rel(CONDP),
        f"arm=pooled,layer=38,position=close,pooling=agent,features={ft} -> accuracy")
led(f3(cp["raw"]["majority_rate"]), "Table 3 row 'prompt condition' col majority", rel(CONDP), "same row -> majority_rate")
T3.append(f"| prompt condition (cheat arm vs solo arm), accuracy | both arms pooled | {f3(cp['raw']['accuracy'])} | {f3(cp['author erased']['accuracy'])} | {f3(cp['author, task, condition erased']['accuracy'])} | {f3(cp['raw']['majority_rate'])} |")
cells = []
for ft in FEATS:
    v, r = auc_cell(pca, PCA, f"Table 3 row 'cheating B.I, both arms pooled' col '{ft}'", arm="pooled", probe="pca", turn_labels="agent",
                    layer="38", position="close", pooling="agent", features=ft, scope="pooled")
    cells.append(v)
T3.append(f"| cheating, B.I pooled AUROC ({r['cheat']} / {r['no_cheat']}) | both arms pooled | {cells[0]} | {cells[1]} | {cells[2]} | 0.5 |")
cells = []
for ft in FEATS:
    v, r = auc_cell(pca, PCA, f"Table 3 row 'cheating B.I, cheat arm' col '{ft}'", arm="cheat", probe="pca", turn_labels="agent",
                    layer="38", position="close", pooling="agent", features=ft, scope="pooled")
    cells.append(v)
T3.append(f"| cheating, B.I pooled AUROC ({r['cheat']} / {r['no_cheat']}) | cheat arm only | {cells[0]} | {cells[1]} | {cells[2]} | 0.5 |")
# sweep range over the 12 solo configurations
solo_pooled = [float(r["auroc"]) for r in pca if r["arm"] == "solo" and r["features"] == "raw" and r["scope"] == "pooled"
               and r["pooling"] == "agent" and r["turn_labels"] == "agent"]
assert len(solo_pooled) == 12, len(solo_pooled)
rng = f"{min(solo_pooled):.3f} to {max(solo_pooled):.3f}"
led(rng, "Table 3 row 'sweep' col raw", rel(PCA), "min/max auroc over the 12 rows arm=solo,features=raw,scope=pooled,pooling=agent,turn_labels=agent")
T3.append(f"| cheating, B.I pooled AUROC, range over the 12 layer x position configurations | solo | {rng} | | | 0.5 |")
# per-turn against position baseline
rt = pick(turn, arm="solo", probe="pca", turn_labels="agent", layer="38", position="close", pooling="turn", features="raw", scope="pooled")
ro = pick(onset, arm="solo", probe="pca", turn_labels="onset", layer="26", position="open", pooling="turn", features="raw", scope="pooled")
for r, p, nm in [(rt, TURN, "agent labels"), (ro, ONSET, "onset labels")]:
    led(f"{f3(r['auroc'])} [{f3(r['ci_lo'])}, {f3(r['ci_hi'])}]", f"Table 3 row 'per-turn, {nm}' col raw", rel(p),
        f"arm=solo,turn_labels={r['turn_labels']},layer={r['layer']},position={r['position']},pooling=turn,features=raw,scope=pooled -> auroc, ci")
    led(f3(r["baseline_turn_position"]), f"Table 3 row 'per-turn, {nm}' col baseline", rel(p), "same row -> baseline_turn_position")
    led(f3(r["auroc_turns"]), f"Table 3 row 'per-turn, {nm}' turn-level AUROC", rel(p), "same row -> auroc_turns")
T3.append(f"| cheating, per-turn probe with the agent's label (layer 38, close), agent AUROC; turn AUROC {f3(rt['auroc_turns'])} | solo | {f3(rt['auroc'])} [{f3(rt['ci_lo'])}, {f3(rt['ci_hi'])}] | | | turn-position baseline {f3(rt['baseline_turn_position'])} |")
T3.append(f"| cheating, per-turn probe with onset labels (layer 26, open), agent AUROC | solo | {f3(ro['auroc'])} [{f3(ro['ci_lo'])}, {f3(ro['ci_hi'])}] | | | turn-position baseline {f3(ro['baseline_turn_position'])} |")
T3 = "\n".join(T3)

# ----------------------------------------------------------------------------------------------------------------
# Table 4: the probe against the text classifiers, the lexical-only null and the judges
# ----------------------------------------------------------------------------------------------------------------
TFIDF = f"{REPO}/hermeneutics/results/tfidf/auroc.csv"      # solo, whole transcript
TFIDF_B = f"{REPO}/hermeneutics/results/tfidf_b/auroc.csv"  # solo, agent's own words; think windows
TFIDF_C = f"{REPO}/hermeneutics/results/tfidf_c/auroc.csv"  # pooled and cheat arms, three views
JUDGES = f"{REPO}/hermeneutics/results/judges/auroc.csv"
tfidf_all = rows(TFIDF) + rows(TFIDF_B) + rows(TFIDF_C)
judges = rows(JUDGES)
# the lexical null's within-author rows: haiku has no think text (agents=0, auroc empty)


def scope_rows(tab, **kw):
    return [r for r in tab if all(str(r[k]) == str(v) for k, v in kw.items())]


def mean_of(tab, scopes, **kw):
    vals = []
    for sc in scopes:
        rr = scope_rows(tab, scope=sc, **kw)
        assert len(rr) == 1, (sc, kw, len(rr))
        if rr[0]["auroc"] not in ("", None):
            vals.append(float(rr[0]["auroc"]))
    return vals


def t4_row(label, tab, path, selector_base, label_short, has_loto=True, n_note=""):
    """One Table 4 row: solo B.I [CI], within-author mean, B.II mean, pooled-arm B.I, cheat-arm B.I."""
    cells = []
    # solo B.I
    rr = scope_rows(tab, arm="solo", scope="pooled", **selector_base)
    assert len(rr) == 1, (label, "solo pooled", len(rr))
    r = rr[0]
    v = f"{f3(r['auroc'])} [{f3(r['ci_lo'])}, {f3(r['ci_hi'])}]"
    led(v, f"Table 4 row '{label_short}' col solo B.I", rel(path), sel(arm="solo", scope="pooled", **selector_base) + " -> auroc, ci")
    agents_note = f"{r['agents']}" + (n_note or "")
    cells.append(v)
    # within-author mean (mean over the type-A rows with a value)
    w = mean_of(tab, AUTH, arm="solo", **selector_base)
    wm = f"{sum(w) / len(w):.3f}" + ("" if len(w) == 5 else f" ({len(w)} authors)")
    led(wm, f"Table 4 row '{label_short}' col within-author mean", rel(path),
        f"mean of auroc over scope in {AUTH} ({len(w)} defined), arm=solo, " + sel(**selector_base))
    cells.append(wm)
    # B.II mean
    if has_loto:
        l = mean_of(tab, [f"LOTO {a}" for a in AUTH], arm="solo", **selector_base)
        lm = f"{sum(l) / len(l):.3f}"
        led(lm, f"Table 4 row '{label_short}' col B.II mean", rel(path), "mean of auroc over scope in LOTO ds..qwen, arm=solo, " + sel(**selector_base))
    else:
        lm = "n/a (no training)"
    cells.append(lm)
    # pooled and cheat arms B.I
    for arm in ("pooled", "cheat"):
        rr = scope_rows(tab, arm=arm, scope="pooled", **selector_base)
        if len(rr) == 1 and rr[0]["auroc"]:
            x = f3(rr[0]["auroc"])
            led(x, f"Table 4 row '{label_short}' col {arm}-arm B.I", rel(path), sel(arm=arm, scope="pooled", **selector_base) + " -> auroc")
        else:
            x = "n/a"
        cells.append(x)
    return f"| {label} | {agents_note} | " + " | ".join(cells) + " |"


T4 = ["| method | solo agents scored | solo arm, B.I pooled AUROC [95% CI] | solo, within-author mean (A) | solo, B.II mean (unseen author) | both arms pooled, B.I | cheat arm, B.I |",
      "|---|---|---|---|---|---|---|"]
# probe rows: solo at layer all; pooled and cheat arms at layer 38 close (the configurations tabled in RESULTS.md)
def probe_row(label, tab, path, probe, solo_layer, solo_pos, label_short):
    cells = []
    r = pick(tab, arm="solo", probe=probe, turn_labels="agent", layer=solo_layer, position=solo_pos, pooling="agent", features="raw", scope="pooled")
    v = f"{f3(r['auroc'])} [{f3(r['ci_lo'])}, {f3(r['ci_hi'])}]"
    led(v, f"Table 4 row '{label_short}' col solo B.I", rel(path), sel(arm="solo", probe=probe, layer=solo_layer, position=solo_pos, features="raw", scope="pooled") + " -> auroc, ci")
    cells.append(v)
    w = [float(pick(tab, arm="solo", probe=probe, turn_labels="agent", layer=solo_layer, position=solo_pos, pooling="agent", features="raw", scope=a)["auroc"]) for a in AUTH]
    wm = f"{sum(w) / 5:.3f}"
    led(wm, f"Table 4 row '{label_short}' col within-author mean", rel(path), f"mean of auroc over scope in {AUTH}, arm=solo,probe={probe},layer={solo_layer},position={solo_pos},features=raw")
    cells.append(wm)
    l = [float(pick(tab, arm="solo", probe=probe, turn_labels="agent", layer=solo_layer, position=solo_pos, pooling="agent", features="raw", scope=f"LOTO {a}")["auroc"]) for a in AUTH]
    lm = f"{sum(l) / 5:.3f}"
    led(lm, f"Table 4 row '{label_short}' col B.II mean", rel(path), f"mean of auroc over LOTO ds..qwen, arm=solo,probe={probe},layer={solo_layer},position={solo_pos},features=raw")
    cells.append(lm)
    for arm in ("pooled", "cheat"):
        rr = [x for x in tab if x["arm"] == arm and x["probe"] == probe and x.get("turn_labels", "") in ("agent", "") and x["layer"] == "38" and x["position"] == "close" and x["pooling"] == "agent" and x["features"] == "raw" and x["scope"] == "pooled"]
        if len(rr) == 1:
            x = f3(rr[0]["auroc"])
            led(x, f"Table 4 row '{label_short}' col {arm}-arm B.I", rel(path), sel(arm=arm, probe=probe, layer=38, position="close", features="raw", scope="pooled") + " -> auroc")
        else:
            x = "n/a (layer 38 not run)"
        cells.append(x)
    return f"| {label} | {r['agents']} | " + " | ".join(cells) + " |"


T4.append(probe_row("activation probe, reader (PCA, agent means; solo at layers concatenated, both positions; other arms at layer 38, close)", pca, PCA, "pca", "all", "both", "probe PCA"))
T4.append(probe_row("activation probe, reader (every dimension; solo at layers concatenated, close position; other arms at layer 38, close)", full, FULL, "full", "all", "close", "probe full-dim"))
T4.append(t4_row("TF-IDF + logistic regression, whole rendered transcript", tfidf_all, f"{REPO}/hermeneutics/results/tfidf*/auroc.csv", dict(probe="tfidf", position="transcript"), "TF-IDF transcript"))
T4.append(t4_row("TF-IDF + logistic regression, the agent's own words only", tfidf_all, f"{REPO}/hermeneutics/results/tfidf*/auroc.csv", dict(probe="tfidf", position="agent"), "TF-IDF own words"))
T4.append(t4_row("lexical-only reader null: TF-IDF on the think windows", tfidf_all, f"{REPO}/hermeneutics/results/tfidf*/auroc.csv", dict(probe="tfidf", position="think"), "lexical null"))
T4.append(t4_row("judge: Claude Haiku 5.5 (author-level; reasoning off)", judges, JUDGES, dict(probe="judge:anthropic/claude-haiku-5.5"), "judge Haiku", has_loto=False))
T4.append(t4_row("judge: Claude Sonnet 5.5 (above the authors; low reasoning effort; stopped by the spend cap)", judges, JUDGES, dict(probe="judge:anthropic/claude-sonnet-5.5"), "judge Sonnet", has_loto=False, n_note=" of 373"))
T4.append(t4_row("judge: GPT-5 mini (author-level; minimal reasoning)", judges, JUDGES, dict(probe="judge:openai/gpt-5-mini"), "judge GPT-5 mini", has_loto=False))
T4.append("| random-weights reader (same architecture, untrained) | not run | not run | not run | not run | not run | not run |")
T4.append("| reader interrogation (ask the reader) | not run | not run | not run | not run | not run | not run |")
T4 = "\n".join(T4)
# judge coverage and cost from the judges file
son = pick(judges, arm="solo", probe="judge:anthropic/claude-sonnet-5.5", scope="pooled")
led(f"{son['agents']} of 373 solo agents; coverage {float(son['coverage']):.3f}", "Table 4 Sonnet row col agents; section 4.4 prose", rel(JUDGES), "arm=solo,probe=judge:anthropic/claude-sonnet-5.5,scope=pooled -> agents, coverage")
for m in ("openai/gpt-5-mini", "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"):
    r = pick(judges, arm="solo", probe=f"judge:{m}", scope="pooled")
    led(f"${float(r['cost_usd']):.2f}", f"NUMBERS only: judge {m} cost of every call in the result file", rel(JUDGES), "cost_usd (all calls; equals spend.json per_model_all_calls)")
    led(f"${float(r['cost_scored_usd']):.2f}", f"NUMBERS only: judge {m} cost of the calls that parsed to a score", rel(JUDGES), "cost_scored_usd")
loto = [float(pick(pca, **base, features="raw", scope=f"LOTO {a}")["auroc"]) for a in AUTH]
led(f"{min(loto):.3f} to {max(loto):.3f}", "Abstract and 4.4: B.II range of the PCA probe", rel(PCA), "min/max auroc over scope in LOTO ds..qwen, arm=solo,layer=all,position=both,features=raw")
jud_pooled = [float(pick(judges, arm="solo", probe=f"judge:{m}", scope="pooled")["auroc"]) for m in ("openai/gpt-5-mini", "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5")]
ABS_JUDGE = f"{min(jud_pooled):.2f} to {max(jud_pooled):.2f}"
led(ABS_JUDGE, "Abstract: judge AUROC range", rel(JUDGES), "min/max auroc over the three judges, arm=solo, scope=pooled")
ABS_TFIDF = f3(pick(tfidf_all, arm="solo", probe="tfidf", position="transcript", scope="pooled")["auroc"])
led(ABS_TFIDF, "Abstract: TF-IDF whole-transcript solo B.I", rel(TFIDF), "arm=solo,position=transcript,scope=pooled -> auroc")

# ----------------------------------------------------------------------------------------------------------------
# Table 5: board seed
# ----------------------------------------------------------------------------------------------------------------
def tsv_rows(path, keep_arm_header=False):
    out = []
    for line in open(path):
        if not line.strip() or line.startswith(("TOTAL", "NOTE")):
            continue
        if line.startswith("arm ") and not keep_arm_header:
            continue
        parts = line.split()
        out.append(parts)
    return out


tb25 = tsv_rows(TB25)
hdr25 = tb25[0]
tb25 = [r for r in tb25[1:] if r[0].startswith("pilot_")]
assert hdr25[:2] == ["run", "collab"]
arms25 = {}
for r in tb25:
    arm = r[0].split("_")[1]
    d = arms25.setdefault(arm, {"n": 0, "crossed+": 0})
    d["n"] += 1
    d["crossed+"] += r[1] in ("crossed", "assembled")
tb60 = tsv_rows(TB60)
hdr60 = tb60[0]
tb60 = [r for r in tb60[1:] if r[0].startswith("pilot60_")]
assert hdr60 == ["run", "collab", "dis", "sol", "crs", "asm", "passed", "posts", "cost$"], hdr60


def agent_posts(run):
    """Board posts by agents in a run: messages in the board channel file whose sender is not 'system'. The pilot TSV's
    posts column counted a system hand-in notice in pilot60_none_s1; the dose-response table excludes system messages,
    and this count follows the dose-response rule."""
    fs = glob.glob(os.path.join(PILOT, run, "communication", "*.json"))
    assert len(fs) == 1, (run, fs)
    msgs = json.load(open(fs[0]))["messages"]
    return sum(1 for m in msgs if m.get("sender") != "system"), len(msgs)


arms60 = {}
for r in tb60:
    arm = r[0].split("_")[1]
    d = arms60.setdefault(arm, {"n": 0, "crossed+": 0, "pass": 0, "posts": [], "dis": [], "sol": [], "crs": [], "asm": []})
    d["n"] += 1
    d["crossed+"] += r[1] in ("crossed", "assembled")
    d["pass"] += r[6] == "True"
    for k, i in (("dis", 2), ("sol", 3), ("crs", 4), ("asm", 5)):
        d[k].append(float(r[i]))
    n_agent, n_all = agent_posts(r[0])
    d["posts"].append(float(n_agent))
    led(f"{n_agent} agent posts ({n_all} messages incl. system; TSV posts column {r[7]})", f"Table 5 tb=60 pilot row {arm}: posts input for {r[0]}",
        rel(os.path.join(PILOT, r[0], "communication", "flipt-02e2#1.json")), "count of messages with sender != 'system'")
dose = tsv_rows(DOSE, keep_arm_header=True)
hdrd = dose[0]
assert hdrd == ["arm", "n", "crossed+", "pass", "mPosts", "mWPosts", "mDisc", "mSol", "mCrs", "mAsm"], hdrd
dose = {r[0]: r for r in dose[1:]}

ARMNAME = {"none": "none (empty board)", "neutral": "neutral post", "neutralL0": "neutral post", "procoordL1": "pro-coordination, mild",
           "procoordL2": "pro-coordination, explicit", "anticoord": "anti-coordination (hidden regulator)"}
T5 = ["| set | turn budget | arm | n | crossed or assembled | passed hidden tests | mean board posts (workers only) | mean discussed | mean solicited | mean crossed | mean assembled |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for arm in ("none", "neutral", "procoordL2"):
    d = arms25[arm]
    led(f"{d['crossed+']}/{d['n']}", f"Table 5 tb=25 row {arm} col crossed or assembled", rel(TB25),
        f"count of collab in (crossed, assembled) over rows pilot_{arm}_s*; n = row count")
    posts25 = "0.4 (one team, two posts)" if arm == "procoordL2" else "0"
    T5.append(f"| pilot | 25 | {ARMNAME[arm]} | {d['n']} | {d['crossed+']}/{d['n']} | 0/{d['n']} | {posts25} | | | 0 | 0 |")
led("passed False in all 12 rows; crs 0, asm 0", "Table 5 tb=25 rows cols passed, crossed, assembled", rel(TB25), "columns passed, crs, asm of every pilot_* row")
led("agent posts 0 in neutral 5/5, none 2/2, pro-coord 4/5; s4 posted 2", "Table 5 tb=25 rows col mean posts", rel(f"{SUB}/VERIFICATION_LOG.md"),
    "entry '2026-11-07 Board-seed LOCAL pilot', tb=25 bullet")
for arm in ("none", "procoordL2"):
    d = arms60[arm]
    mp = lambda k: f"{sum(d[k]) / len(d[k]):.1f}"
    led(f"{d['crossed+']}/{d['n']}, {d['pass']}/{d['n']}, posts {mp('posts')}, dis {mp('dis')}, sol {mp('sol')}, crs {mp('crs')}, asm {mp('asm')}",
        f"Table 5 tb=60 pilot row {arm}", rel(TB60), f"computed from the three pilot60_{arm}_s* rows (collab, passed, dis, sol, crs, asm); posts recounted from the board JSON files, system sender excluded")
    T5.append(f"| pilot | 60 | {ARMNAME[arm]} | {d['n']} | {d['crossed+']}/{d['n']} | {d['pass']}/{d['n']} | {mp('posts')} | {mp('dis')} | {mp('sol')} | {mp('crs')} | {mp('asm')} |")
for arm in ("none", "neutralL0", "procoordL1", "procoordL2", "anticoord"):
    r = dose[arm]
    led(" | ".join(r[1:]), f"Table 5 dose-response row {arm} (n, crossed+, pass, mPosts, mWPosts, mDisc, mSol, mCrs, mAsm)", rel(DOSE), f"row arm={arm}")
    posts = r[5] if arm == "anticoord" else r[4]
    extra = f" (all posts {r[4]}, the regulator included)" if arm == "anticoord" else ""
    T5.append(f"| dose-response | 60 | {ARMNAME[arm]} | {r[1]} | {r[2]} | {r[3]} | {posts}{extra} | {r[6]} | {r[7]} | {r[8]} | {r[9]} |")
T5 = "\n".join(T5)
# spend lines for the pilot
for path, pat, where in [(TB25, r"TOTAL billed: \$([0-9.]+) over (\d+) runs", "pilot tb=25 spend"),
                         (TB60, r"TOTAL tb=60 billed: \$([0-9.]+) over (\d+) runs", "pilot tb=60 spend"),
                         (DOSE, r"TOTAL billed: \$([0-9.]+) over (\d+) runs", "dose-response spend")]:
    m = re.search(pat, open(path).read())
    led(f"${m.group(1)} over {m.group(2)} runs", f"Table S (spend) row {where}", rel(path), "TOTAL line")
SPEND = {}
for path, pat, key in [(TB25, r"TOTAL billed: \$([0-9.]+)", "tb25"), (TB60, r"TOTAL tb=60 billed: \$([0-9.]+)", "tb60"),
                       (DOSE, r"TOTAL billed: \$([0-9.]+)", "dose")]:
    SPEND[key] = float(re.search(pat, open(path).read()).group(1))

# ----------------------------------------------------------------------------------------------------------------
# HF README facts
# ----------------------------------------------------------------------------------------------------------------
hf = open(HF_README).read()
hf_total = re.search(r"\*\*(4,157)\*\* unique completed, validated runs", hf).group(1)
hf_matrix = re.search(r"designed matrix of \*\*(4,424)\*\* runs", hf).group(1)
hf_extra = re.search(r"plus \*\*(309)\*\* extra duplicate samples \((4,466) transcripts", hf)
hf_spend = re.search(r"Total billed for this study: \$([0-9.]+)", hf).group(1)
hf_canon = re.search(r"\*\*\$([0-9.]+)\*\* for the 4,157 valid deduplicated runs", hf).group(1)
hf_built = re.search(r"Built (2026-11-07T10:06:53Z)", hf).group(1)
HF_URL = "https://huggingface.co/datasets/unrulyabstractions/apart-swarm"
for v, w, q in [(hf_total, "Table S row 'controls dataset' col runs; Appendix A", "line 3: 'with **4,157** unique completed, validated runs'"),
                (hf_matrix, "Table S row 'controls dataset' col designed matrix", "line 3: 'designed matrix of **4,424** runs'"),
                (f"{hf_extra.group(1)} / {hf_extra.group(2)}", "Table S row 'controls dataset' col duplicate copies / transcripts", "line 3"),
                (f"${hf_spend}", "Table S row 'controls dataset' col spend", "Spend: 'Total billed for this study: $192.85'"),
                (f"${hf_canon}", "Table S row 'controls dataset' col spend on canonical runs", "Spend bullet 1"),
                (hf_built, "Appendix A: publish time", "line 3: 'Built 2026-11-07T10:06:53Z'"),
                ("n_agents = 1 in every run; collaboration and engagement null 0 of 4466", "Method: single-agent scope", "section 'Single-agent scope'"),
                ("judged false in every run", "Method: no LLM judge", "section 'No LLM judge'"),
                ("label = cheat if used or tampered; honest if shipped; else other", "Method: coarse label", "section 'Labels'"),
                ("rung order none < exposed < tried < opened < used < tampered < checked < shipped", "Method: cheat ladder", "section 'Labels'")]:
    led(v, w, f"{HF_URL}/raw/main/README.md (fetched 2026-11-07 with the account token)", q)

# collection window
import datetime
lt = [float(r["launch_time"]) for r in runs if r["launch_time"]]
iso = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
first, last = iso(min(lt)), iso(max(lt))
led(first, "Appendix A: first controls launch", rel(RUNS), "min(launch_time) over all rows")
led(last, "Appendix A: last controls launch", rel(RUNS), "max(launch_time) over all rows")
led("Nov 6 to 8, 2026; submissions close Nov 8 11:59 PM AoE", "Appendix A: sprint window",
    "https://apartresearch.com/sprints/swarm-alignment-2026 (fetched 2026-11-07)", "Overview and 'Submissions close' lines")
led("2026-11-07 (mechanism commit 2c72d93; pilots the same day)", "Appendix A: board-seed dates", rel(f"{SUB}/VERIFICATION_LOG.md"),
    "entries '2026-11-07 — Board seed injection mechanism' and '2026-11-07 — Board-seed LOCAL pilot'")
led("2026-10-09", "Appendix A: framework log begins", rel(f"{SUB}/VERIFICATION_LOG.md"), "first heading")
led("2026-11-07T02-37-35 to 05-38-26 (eval log names)", "Appendix A: pilot run times", rel(f"{PILOT}") + "/*/run.json, confirm_doseresponse/*/run.json", "eval_log file names")

# spend table
SPJ = json.load(open(f"{REPO}/hermeneutics/results/judges/spend.json"))
TS = f"""| item | runs | spend (USD) | source |
|---|---|---|---|
| controls dataset, canonical valid runs | {hf_total} | {hf_canon} | dataset README |
| controls dataset, whole study (duplicates, invalid copies, attempts) | 4,737 executions | {hf_spend} | dataset README |
| board-seed pilot, turn budget 25 | 12 | {SPEND['tb25']:.2f} | pilot table |
| board-seed pilot, turn budget 60 | 6 | {SPEND['tb60']:.2f} | pilot table |
| board-seed dose-response | 25 | {SPEND['dose']:.2f} | dose-response table |
| judge baseline, GPT-5 mini | 600 agents | {SPJ['per_model_all_calls']['openai/gpt-5-mini']:.2f} | judge spend ledger |
| judge baseline, Claude Haiku 5.5 | 600 agents | {SPJ['per_model_all_calls']['anthropic/claude-haiku-5.5']:.2f} | judge spend ledger |
| judge baseline, Claude Sonnet 5.5 (stopped by the in-code cap) | 460 of 600 agents | {SPJ['per_model_all_calls']['anthropic/claude-sonnet-5.5']:.2f} | judge spend ledger |
| judge baseline, discarded pricing round (empty replies) | 20 calls per model | {SPJ['discarded_pricing_round']:.2f} | judge spend ledger |
| judge baselines, total as the provider ledger moved | | {SPJ['provider_delta']:.2f} | judge spend ledger |
| reader forward passes and probes | 600 runs probed | 0 (free GPU host) | verification log |"""
SPEND_JSON = f"{REPO}/hermeneutics/results/judges/spend.json"
for k, w in [("per_model_all_calls", "Table S judge rows per model"), ("discarded_pricing_round", "Table S discarded pricing round"),
             ("provider_delta", "Table S judge total"), ("provider_before", "NUMBERS only"), ("provider_after", "NUMBERS only"),
             ("score_files_all_calls", "NUMBERS only"), ("score_files_scored_calls", "NUMBERS only")]:
    led(SPJ[k], w, rel(SPEND_JSON), k)
led("$12.34 (12.3404)", "Table S judge total; supersedes the 11:05 log figure $12.36", "VERIFICATION_LOG.md", "2026-11-08 ~00:40 entry: judge spend re-read from the on-disk ledger; RESULTS.md Table 11b")
led("460/600 scored; 262 solo", "Table S Sonnet row; Table 4", "VERIFICATION_LOG.md", "11:05 entry: 'Claude Sonnet 5.5 0.796 on 262 solo agents (460/600 scored before the in-code $10 cap stopped it)'")
led("reasoning disabled (Haiku), minimal (gpt-5-mini), low (Sonnet); tool results cut to 500 chars; transcript tail cut to 12,000 tokens; mean 8,323 prompt tokens", "Table 4 caption", "VERIFICATION_LOG.md", "11:05 entry, 'API judges' bullet")
led("95%", "Table M3 intervals; captions of Tables 1, 2, 4, D1, D2: the bootstrap interval level", "hermeneutics/probes.py", "boot_ci: np.percentile(vals, [2.5, 97.5]); docstring 'AUROC with a 95% bootstrap over runs'")
led("4,737 executions", "Table S row whole study col runs", f"{HF_URL}/raw/main/README.md", "Spend: 'per-run ledger of every execution on the collection disk (4,737 executions)'")

# ----------------------------------------------------------------------------------------------------------------
# Table 6: label-gap map (DATASETS.md, datasets/ai-village.md)
# ----------------------------------------------------------------------------------------------------------------
T6 = """| output | controls dataset (own, single-agent) | board-seed environment (own, three agents) | AI Village (external, eval-only) |
|---|---|---|---|
| cheating | direct (cheat-ladder rung) | direct (team cheat-ladder rung) | judge only (no task, no grader) |
| coordination | zero by construction (one agent) | direct (collaboration ladder, board counts) | derivable (chat graph) |
| cooperation | zero by construction | not separable from collaboration (one shared goal) | derivable, weak (coordination serving an agent's own goal) |
| collaboration | zero by construction | direct for the shared exploit (crossed, assembled), under one shared goal | derivable (coordination under an active village-wide goal) |
| collusion | zero by construction | derivable (rung at least used, and collaboration at least crossed) | judge only (needs cheating) |
| scheming | zero by construction | gap (no covertness signal) | judge only |"""
led("label map cells", "Table 6", "DATASETS.md (commit 45f8119); DATASET_DETAILS.md; datasets/ai-village.md",
    "DATASET_DETAILS.md 'Notation' (hierarchy, zeros by construction), section 7 AI Village entry, section 8 own data; ai-village.md 'Labels for our six outputs'")
n_sources = sum(1 for l in open(f"{REPO}/DATASETS.md") if l.startswith("- **["))
led(f"about sixty ({n_sources} graded entries)", "Section 4 Q6: size of the survey", rel(f"{REPO}/DATASETS.md"), "count of lines starting '- **['")
led("hierarchy Sch => Col => (Ch and Cb); Cb => Cp => Cd", "Section 4 Q6 prose", rel(f"{REPO}/DATASET_DETAILS.md"), "'Notation' section, label hierarchy")
led("no external dataset uses our authors", "Section 4 Q6 prose", rel(f"{REPO}/DATASET_DETAILS.md"), "'Key takeaways', first bullet")
led("ranked next: SALT-NLP agent-collusion, NARCBench, Colosseum, MAST-Data, Terminal Wrench", "Section 4 Q6 prose", rel(f"{REPO}/DATASETS.md"), "Ranking rows 2 to 6; DATASET_DETAILS.md 'Plan by priority'")
led("AI Village grade B, gated with access granted, eval-only licence", "Section 4 Q6, Table 6", rel(f"{REPO}/DATASETS.md"), "Ranking row 16; section 7 entry")

# ----------------------------------------------------------------------------------------------------------------
# Appendix B: full sweep
# ----------------------------------------------------------------------------------------------------------------
def sweep(tab, path, probe, name):
    out = [f"**{name}.** Solo arm, raw features, agent pooling. Columns: B.I pooled AUROC [95% CI]; A within each author; B.II leave-one-author-out; permutation null 95th percentile (pooled).", "",
           "| layer | position | B.I pooled | A ds | A gpt5m | A haiku | A luna | A qwen | B.II ds | B.II gpt5m | B.II haiku | B.II luna | B.II qwen | null 95th |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"26": 0, "38": 1, "51": 2, "all": 3}
    keys = sorted({(r["layer"], r["position"]) for r in tab if r["arm"] == "solo" and r["pooling"] == "agent"}, key=lambda k: (order[k[0]], k[1]))
    for L, P in keys:
        g = lambda s: pick(tab, arm="solo", probe=probe, turn_labels="agent", layer=L, position=P, pooling="agent", features="raw", scope=s)
        p = g("pooled")
        cells = [f"{f3(p['auroc'])} [{f3(p['ci_lo'])}, {f3(p['ci_hi'])}]"] + [f3(g(a)["auroc"]) for a in AUTH] + [f3(g(f"LOTO {a}")["auroc"]) for a in AUTH] + [f3(p["null_95"])]
        led(" | ".join(cells), f"Appendix D {name} row layer={L} position={P}", rel(path),
            f"arm=solo,probe={probe},layer={L},position={P},features=raw; scope=pooled (auroc, ci, null_95), ds..qwen, LOTO ds..qwen -> auroc")
        out.append(f"| {L} | {P} | " + " | ".join(cells) + " |")
    return "\n".join(out)


TB1 = sweep(pca, PCA, "pca", "Table D1: PCA probe")
TB2 = sweep(full, FULL, "full", "Table D2: full-dimension probe")

# ----------------------------------------------------------------------------------------------------------------
# Appendix C: board-seed texts, read from the source module with ast
# ----------------------------------------------------------------------------------------------------------------
import ast
BS = f"{SUB}/src/swarm/board_seed.py"
mod = ast.parse(open(BS).read())
seed_table = {}
for node in mod.body:
    if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "SEED_TABLE":
        for k, v in zip(node.value.keys, node.value.values):
            key = ast.literal_eval(k)
            seed_table[key] = ast.literal_eval(v) if isinstance(v, ast.Constant) else getattr(v, "id", "?")
assert seed_table, "SEED_TABLE not found"
LABEL = {("neutral", "0"): "neutral post (L0)", ("pro-coord", "1"): "pro-coordination, mild (L1)", ("pro-coord", "2"): "pro-coordination, explicit (L2)"}
TC = ["| condition | text posted by the system |", "|---|---|"]
for key, text in seed_table.items():
    if text == "USE_REGULATOR":
        continue
    TC.append(f"| {LABEL[key]} | {text} |")
    led(f"'{text[:60]}...'", f"Appendix C row {LABEL[key]}", rel(BS), f"SEED_TABLE[{key}]")
TC = "\n".join(TC)

# ----------------------------------------------------------------------------------------------------------------
# prose numbers (abstract) ledger
# ----------------------------------------------------------------------------------------------------------------
r = pick(pca, **base, features="raw", scope="pooled")
ABS_BI = f3(r["auroc"])
led(ABS_BI, "Abstract: solo-arm B.I AUROC", rel(PCA), sel(**base, features="raw", scope="pooled") + " -> auroc")
ABS_LOTO = f"{min(loto):.2f} to {max(loto):.2f}"
led(ABS_LOTO, "Abstract: B.II range", rel(PCA), "min/max over LOTO scopes, same configuration")
r2 = pick(pca, **base, features="author, task, condition erased", scope="pooled")
ABS_ERASED = f3(r2["auroc"])
led(ABS_ERASED, "Abstract: B.I after erasure", rel(PCA), sel(**base, features="author, task, condition erased", scope="pooled") + " -> auroc")

# ----------------------------------------------------------------------------------------------------------------
# assemble
# ----------------------------------------------------------------------------------------------------------------
tpl = open(os.path.join(os.path.dirname(__file__), "report_template.md")).read()
for k, v in {"<<TM1>>": TM1, "<<TM2>>": TM2, "<<TM3>>": TM3, "<<T1>>": T1, "<<T2>>": T2, "<<T3>>": T3, "<<T4>>": T4, "<<T5>>": T5,
             "<<T6>>": T6, "<<TS>>": TS, "<<TB1>>": TB1, "<<TB2>>": TB2, "<<ABS_BI>>": ABS_BI, "<<ABS_LOTO>>": ABS_LOTO,
             "<<ABS_ERASED>>": ABS_ERASED, "<<TC>>": TC, "<<ABS_JUDGE>>": ABS_JUDGE, "<<ABS_TFIDF>>": ABS_TFIDF, "<<HF_TOTAL>>": hf_total, "<<FIRST>>": first, "<<LAST>>": last, "<<HF_BUILT>>": hf_built,
             "<<HF_SPEND>>": hf_spend}.items():
    assert k in tpl, k
    tpl = tpl.replace(k, v)
assert "<<" not in tpl, re.findall(r"<<[A-Z_0-9]+>>", tpl)
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/REPORT_DRAFT.md", "w").write(tpl)

abstract = tpl.split("## Abstract\n", 1)[1].split("\n## ", 1)[0].strip()
wc = len(abstract.split())
open(f"{OUT}/ABSTRACT.txt", "w").write(abstract + f"\n\n[{wc} words]\n")

hdr = ["# Number ledger for report/REPORT_DRAFT.md", "",
       "Every number in the draft traces to a file that was opened while drafting (2026-11-07). One line per table cell or prose value. "
       "Paths are relative to the repository root unless absolute. The dataset README was fetched from the Hub with the account token.", "",
       "| value | where in the draft | source | field / selector |", "|---|---|---|---|"]
open(f"{OUT}/NUMBERS.md", "w").write("\n".join(hdr + LEDGER) + "\n")

body = tpl.split("## Abstract\n", 1)[1].split("\n## References", 1)[0]
body_wo_tables = "\n".join(l for l in body.splitlines() if not l.startswith("|"))
print("abstract words", wc)
print("body words incl. tables", len(body.split()))
print("body words excl. table rows", len(body_wo_tables.split()))
print("ledger lines", len(LEDGER))
