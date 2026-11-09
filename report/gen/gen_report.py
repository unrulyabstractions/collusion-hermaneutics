"""Build report/REPORT_DRAFT.md, report/NUMBERS.md and report/ABSTRACT.txt from the result files.

Every number in a table is read from a file and written to the ledger with its selector. Prose numbers are ledgered
by hand at the bottom of this script.
"""
import collections
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


RENUM = {"2": "Table D4", "3": "Table 2", "4": "Table 3", "5": "Table 4"}


def renum(where):
    return re.sub(r"\bTable ([2-5])\b", lambda m: RENUM[m.group(1)], where)


def led(value, where, source, selector, raw_where=False):
    """raw_where: the location is already written in the final table numbering, so renum() must not touch it."""
    cell = lambda x: str(x).replace(" | ", ", ").replace("|", "/")
    LEDGER.append(f"| {cell(value)} | {cell(where if raw_where else renum(where))} | `{cell(source)}` | {cell(selector)} |")


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
# Table E1: authors and reader
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

WEIGHTS = {"ds": "open (MIT); API in our runs", "gpt5m": "closed", "haiku": "closed", "luna": "closed", "qwen": "open"}
DS_HUB = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash (Hub API read 2026-11-08)"
led("open (MIT); API in our runs", "Table E1 row DeepSeek V4.1 Flash col weights; Methods, Authors and reader", DS_HUB, "cardData.license = mit; gated = false")
led("763,205,315,794 parameters", "NUMBERS only: basis for 'far more parameters than the reader' (Methods, Authors and reader)", DS_HUB, "safetensors.total")
N_CLOSED = sum(1 for a in AUTH if WEIGHTS[a] == "closed")
assert N_CLOSED == 3, N_CLOSED
led(f"{N_CLOSED} of {len(AUTH)} authors closed-weight", "Related Work; Methods, Authors and reader; Conclusion ('three of our five authors', 'three of them closed-weight')",
    "report/gen/gen_report.py WEIGHTS (Table E1 weights column)", "authors whose weights cell reads 'closed'; DeepSeek V4.1 Flash open per its Hub page")
TM1 = ["| author | API model id | weights | valid runs | think coverage (share of turns with readable reasoning) |",
       "|---|---|---|---|---|"]
for a in AUTH:
    (mid,) = model_id[a]
    n, t, rd = cov[a]
    share = f"{rd / t:.3f}"
    led(mid, f"Table E1 row {NAME[a]} col API model id", rel(RUNS), f"distinct model_id where author={a}")
    led(valid[a], f"Table E1 row {NAME[a]} col valid runs", rel(LABELS), f"count rows where author={a} and valid=True")
    led(share, f"Table E1 row {NAME[a]} col think coverage", rel(RENDER),
        f"sum(n_readable)/sum(turns) over rows where author={a} ({rd}/{t})")
    TM1.append(f"| {NAME[a]} | `{mid}` | {WEIGHTS[a]} | {valid[a]} | {share} |")
TM1.append("| Reader: Qwen3.8-27B | open weights, run locally in bf16 (same model as the author) | open | | |")
TM1 = "\n".join(TM1)

# ----------------------------------------------------------------------------------------------------------------
# Table E2: data per author x arm (full valid set and the probed subsample)
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
        where = f"Table E2 row {NAME[a]}/{cond}"
        led(f"{c} / {nc} / {ot}", where + " col valid runs", rel(LABELS),
            f"count label in (cheat, no cheat, other) where author={a}, condition={cond}, valid=True")
        led(f"{sc} / {snc}", where + " col probed agents", rel(AGENTS), f"count y==1 / y==0 where author={a}, condition={cond}")
        led(st, where + " col probed turns", rel(AGENTS), f"sum n_turns where author={a}, condition={cond}")
        led(sr, where + " col readable", rel(AGENTS), f"sum n_readable where author={a}, condition={cond}")
        TM2.append(f"| {NAME[a]} | {cond} | {c} / {nc} / {ot} | {sc} / {snc} | {st} | {sr} |")
TM2 = "\n".join(TM2)
n_agents = len(agents)
led(n_agents, "Table E3 probed runs", rel(AGENTS), "row count")
assert n_agents == 600

# ----------------------------------------------------------------------------------------------------------------
# Table E3: probe setup constants (from code and logs)
# ----------------------------------------------------------------------------------------------------------------
# PCA width in the solo agent-level B.I fit: probes.py zs() takes k = min(PCA_DIM, max(10, n_train // 2)) on the outer
# training rows; the outer folds are rebuilt here as probes.py analyze() builds them
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
_solo = [r for r in agents if r["condition"] == "solo"]
_y = np.array([int(r["y"]) for r in _solo]); _au = np.array([r["author"] for r in _solo]); _run = np.array([r["run"] for r in _solo])
_strata = np.array([f"{t}|{v}" for t, v in zip(_au, _y)])
_fold = np.zeros(len(_solo), int)
for _f, (_, _te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(_solo)), _strata, _run)):
    _fold[_te] = _f
_ntr = [int((_fold != f).sum()) for f in range(5)]
_ks = sorted({min(256, max(10, n // 2)) for n in _ntr})
K_PCA = str(_ks[0]) if len(_ks) == 1 else f"{_ks[0]} to {_ks[-1]}"
led(K_PCA, "Table E3 probe row: PCA components in the solo agent-level fit", "hermeneutics/probes.py; hermeneutics/results/pca_agent/agents.csv",
    f"k = min(PCA_DIM, max(10, n_train // 2)) (probes.py lines 48 and 128); outer training sizes {_ntr} from StratifiedGroupKFold(5, shuffle=True, random_state=0) over the solo rows, strata author|label, groups run (probes.py line 242)")
DEPTH = ", ".join(f"{L / 64:.2f}" for L in (26, 38, 51))
led(DEPTH, "Table E3 layers row: depth fractions", "hermeneutics/results/full_agent/RESULTS.md", "26/64, 38/64, 51/64 (64 blocks, HF hidden_states numbering)")
TM3 = f"""| setting | value |
|---|---|
| reader | Qwen3.8-27B, bf16 |
| positions | residual stream at the think-open (id 248068) and think-close (id 248069) token of every assistant turn |
| layers | 26, 38, 51 of 64 blocks ({DEPTH} of depth), and the three concatenated |
| pooling | mean over the agent's turns (agent level), or one row per turn (turn level) |
| probe | training-fold PCA to min(256, half the training rows) components ({K_PCA} in the solo agent-level fit), z-scoring, L2 logistic regression, C by grouped inner cross-validation |
| variant | every dimension z-scored, no PCA |
| folds | 5, grouped by run, stratified by author x label |
| intervals | 95% bootstrap over runs, 1000 draws |
| null | labels shuffled inside each author, 100 draws |
| erasure | linear concept erasure (LEACE) of author; of author, task and condition; fitted on training rows |
| probed runs | 600 (about 120 per author, balanced over arm x label, runs over 100k tokens left out) |
| probed tokens / turns | 14,182,746 / 10,985 |"""
for v, w, s, q in [
    ("248068 / 248069", "Table E3 positions; Methods, Positions: 'one think-open and think-close token pair', 'the two tokens'", "VERIFICATION_LOG.md",
     "07:20 entry: token ids checked against the tokenizer; think pairs = assistant turns = completed calls in every run"),
    ("26, 38, 51 of 64", "Table E3 layers", "hermeneutics/results/full_agent/RESULTS.md", "header: HF hidden_states numbering, 64 blocks"),
    ("256", "Table E3 probe: the cap on PCA components", "hermeneutics/probes.py", "PCA_DIM = 256 (line 48); k = min(PCA_DIM, max(10, len(Xtr) // 2)) (line 128)"),
    ("5", "Table E3 folds", "hermeneutics/probes.py", "K = 5 (line 45)"),
    ("1000", "Table E3 intervals", "hermeneutics/probes.py", "N_BOOT = 1000 (line 47)"),
    ("100", "Table E3 null", "hermeneutics/results/RESULTS.md", "Table 4 caption '100 draws'; VERIFICATION_LOG 09:13 '100 shuffles'"),
    ("600 / 120 / 100k", "Table E3 probed runs", "VERIFICATION_LOG.md", "07:45 entry: --per-author 120 --max-tokens 100000 -> 600 runs"),
    ("14,182,746 / 10,985", "Table E3 probed tokens / turns", "VERIFICATION_LOG.md", "09:13 entry"),
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
T1 = ["| probe | cheat / no cheat | raw | author erased | author, task, condition erased | metadata-only baseline | TF-IDF, whole transcript |",
      "|---|---|---|---|---|---|---|"]
NULL95 = []
for scope, label, nk in [("pooled", "B.I pooled", "pooled (author + task + condition)")] + \
        [(a, f"A: {NAME[a]}", f"{a} (task + condition)") for a in AUTH]:
    cells = []
    for ft in FEATS:
        v, r = auc_cell(pca, PCA, f"Table 1 row '{label}' col '{ft}'", **base, features=ft, scope=scope)
        cells.append(v)
    r = pick(pca, **base, features="raw", scope=scope)
    n95, pv = f3(r["null_95"]), f3(r["null_p"])
    NULL95.append(float(r["null_95"]))
    led(n95, f"NUMBERS only: Table 1 row '{label}' null 95th percentile (the caption gives the range over rows)", rel(PCA), sel(**base, features="raw", scope=scope) + " -> null_95")
    led(pv, f"NUMBERS only: Table 1 row '{label}' null p (the caption states the 1/101 floor)", rel(PCA), sel(**base, features="raw", scope=scope) + " -> null_p")
    assert abs(float(r["null_p"]) - 1 / 101) < 1e-6, (label, r["null_p"])
    nb = f3(nuis_val(nk))
    led(nb, f"Table 1 row '{label}' col metadata baseline", rel(RESJ), f"nuisance['{nk}']")
    tf = tfidf_solo(scope)
    tfv = f"{f3(tf['auroc'])} [{f3(tf['ci_lo'])}, {f3(tf['ci_hi'])}]" if scope == "pooled" else f3(tf["auroc"])
    led(tfv, f"Table 1 row '{label}' col TF-IDF whole transcript", "hermeneutics/results/tfidf/auroc.csv", f"arm=solo,position=transcript,scope={scope} -> auroc" + (", ci" if scope == "pooled" else ""))
    led(f"{r['cheat']} / {r['no_cheat']}", f"Table 1 row '{label}' col cheat / no cheat", rel(PCA), sel(**base, features="raw", scope=scope) + " -> cheat, no_cheat")
    T1.append(f"| {label} | {r['cheat']} / {r['no_cheat']} | {cells[0]} | {cells[1]} | {cells[2]} | {nb} | {tfv} |")
v, r = auc_cell(full, FULL, "Table 1 row 'B.I pooled, every dimension' col raw", arm="solo", probe="full",
                turn_labels="agent", layer="all", position="close", pooling="agent", features="raw", scope="pooled")
v2, _ = auc_cell(full, FULL, "Table 1 row 'B.I pooled, every dimension' col author erased", arm="solo", probe="full",
                 turn_labels="agent", layer="all", position="close", pooling="agent", features="author erased", scope="pooled")
v3, _ = auc_cell(full, FULL, "Table 1 row 'B.I pooled, every dimension' col all erased", arm="solo", probe="full",
                 turn_labels="agent", layer="all", position="close", pooling="agent",
                 features="author, task, condition erased", scope="pooled")
NULL95.append(float(r["null_95"]))
led(f3(r["null_95"]), "NUMBERS only: Table 1 every-dimension row null 95th percentile (the caption gives the range over rows)", rel(FULL),
    "arm=solo,probe=full,layer=all,position=close,features=raw,scope=pooled -> null_95")
led(f3(r["null_p"]), "NUMBERS only: Table 1 every-dimension row null p (the caption states the 1/101 floor)", rel(FULL),
    "arm=solo,probe=full,layer=all,position=close,features=raw,scope=pooled -> null_p")
assert abs(float(r["null_p"]) - 1 / 101) < 1e-6
T1.append(f"| B.I pooled, every dimension | {r['cheat']} / {r['no_cheat']} | {v} | {v2} | {v3} | | |")
led(f"{r['cheat']} / {r['no_cheat']}", "Table 1 row 'B.I pooled, every dimension' col cheat / no cheat", rel(FULL),
    "arm=solo,probe=full,layer=all,position=close,features=raw,scope=pooled -> cheat, no_cheat")
NULL_LO, NULL_HI = f"{min(NULL95):.3f}", f"{max(NULL95):.3f}"
led(f"{NULL_LO} to {NULL_HI}", "Table 1 caption: range of the within-author label-shuffle null's 95th percentile over the probe rows",
    rel(PCA) + "; " + rel(FULL), "min and max of null_95 over the seven probe rows of Table 1 (solo arm, raw features)")
led("1/101", "Table 1 caption: the shuffle p floor", rel(PCA), "null_p = 0.0099 in every Table 1 probe row (100 shuffles, p = (k+1)/(n+1))")
md = open(RESMD).read()
m = re.search(r"cheating AUROC from run length \(turns, log tokens, readable share\) \| ([0-9.]+) \|", md)
runlen = m.group(1)
led(runlen, "Table 1 row 'baseline: run length'", rel(RESMD), "Table 3 last row (summary.py recomputes it from agents.csv at build time)")
m = re.search(r"cheating AUROC from one-hot pooled \(author only\) \| ([0-9.]+) \|", md)
authonly = m.group(1)
led(authonly, "Table 1 row 'baseline: author one-hot'", rel(RESJ), "nuisance['pooled (author only)'] (also RESULTS.md Table 3)")
assert abs(float(authonly) - float(nuis_val("pooled (author only)"))) < 5e-4
_bp = pick(pca, **base, features="raw", scope="pooled")
_bpc = f"{_bp['cheat']} / {_bp['no_cheat']}"
led(_bpc, "Table 1 rows 'baseline: author one-hot' and 'baseline: run length' col cheat / no cheat (the solo agents of the pooled fit)", rel(PCA),
    sel(**base, features="raw", scope="pooled") + " -> cheat, no_cheat")
T1.append(f"| baseline: author one-hot | {_bpc} | {authonly} | | | | |")
T1.append(f"| baseline: run length | {_bpc} | {runlen} | | | | |")
T1 = "\n".join(T1)

# ----------------------------------------------------------------------------------------------------------------
# Table 2: B.II leave-one-author-out
# ----------------------------------------------------------------------------------------------------------------
T2 = ["| held-out author | think coverage | cheat / no cheat | PCA probe, raw | PCA, author erased | PCA, author, task, condition erased | every-dimension probe, raw | per-turn probe, agent AUROC | TF-IDF, whole transcript |",
      "|---|---|---|---|---|---|---|---|---|"]
for a in AUTH:
    cells = []
    for ft in FEATS:
        v, r = auc_cell(pca, PCA, f"Table 2 row {NAME[a]} col PCA '{ft}'", **base, features=ft, scope=f"LOTO {a}")
        cells.append(v)
    vf, _ = auc_cell(full, FULL, f"Table 2 row {NAME[a]} col every-dimension raw", arm="solo", probe="full", turn_labels="agent",
                     layer="all", position="close", pooling="agent", features="raw", scope=f"LOTO {a}")
    vt, _ = auc_cell(turn, TURN, f"Table 2 row {NAME[a]} col per-turn agent AUROC", arm="solo", probe="pca", turn_labels="agent",
                     layer="38", position="close", pooling="turn", features="raw", scope=f"LOTO {a}")
    n, t, rd = cov[a]
    tf = tfidf_solo(f"LOTO {a}")
    tfv = f"{f3(tf['auroc'])} [{f3(tf['ci_lo'])}, {f3(tf['ci_hi'])}]"
    led(tfv, f"Table 2 row {NAME[a]} col TF-IDF whole transcript", "hermeneutics/results/tfidf/auroc.csv", f"arm=solo,position=transcript,scope=LOTO {a} -> auroc, ci")
    led(f"{r['cheat']} / {r['no_cheat']}", f"Table 2 row {NAME[a]} col cheat / no cheat", rel(PCA), sel(**base, features="raw", scope=f"LOTO {a}") + " -> cheat, no_cheat")
    led(f"{rd / t:.3f}", f"Table 2 row {NAME[a]} col think coverage", rel(RENDER), f"sum(n_readable)/sum(turns) over rows where author={a} ({rd}/{t})")
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
led(f"{r['cheat']} / {r['no_cheat']}", "Table 3 row 'cheating B.I, both arms pooled': cheat / no cheat", rel(PCA), "arm=pooled,layer=38,position=close,features=raw,scope=pooled -> cheat, no_cheat")
cells = []
for ft in FEATS:
    v, r = auc_cell(pca, PCA, f"Table 3 row 'cheating B.I, cheat arm' col '{ft}'", arm="cheat", probe="pca", turn_labels="agent",
                    layer="38", position="close", pooling="agent", features=ft, scope="pooled")
    cells.append(v)
T3.append(f"| cheating, B.I pooled AUROC ({r['cheat']} / {r['no_cheat']}) | cheat arm only | {cells[0]} | {cells[1]} | {cells[2]} | 0.5 |")
led(f"{r['cheat']} / {r['no_cheat']}", "Table 3 row 'cheating B.I, cheat arm': cheat / no cheat", rel(PCA), "arm=cheat,layer=38,position=close,features=raw,scope=pooled -> cheat, no_cheat")
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
    if nm == "agent labels":
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
            led(x, f"Table D3 row '{label_short}' col {arm}-arm B.I", rel(path), sel(arm=arm, scope="pooled", **selector_base) + " -> auroc")
        else:
            x = "n/a"
        cells.append(x)
    return (f"| {label} | {agents_note} | " + " | ".join(cells[:3]) + " |", f"| {label} | " + " | ".join(cells[3:]) + " |")


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
            led(x, f"Table D3 row '{label_short}' col {arm}-arm B.I", rel(path), sel(arm=arm, probe=probe, layer=38, position="close", features="raw", scope="pooled") + " -> auroc")
        else:
            x = "n/a (layer 38 not run)"
        cells.append(x)
    return (f"| {label} | {r['agents']} | " + " | ".join(cells[:3]) + " |", f"| {label} | " + " | ".join(cells[3:]) + " |")

T4 = ["| method | solo agents scored | solo B.I pooled AUROC [95% CI] | within-author mean (A) | B.II mean (unseen author) |",
      "|---|---|---|---|---|"]
TD3 = ["| method | both arms pooled, B.I | cheat arm, B.I |", "|---|---|---|"]
def add(pair):
    T4.append(pair[0]); TD3.append(pair[1])
add(probe_row("probe, PCA", pca, PCA, "pca", "all", "both", "probe PCA"))
add(probe_row("probe, every dimension", full, FULL, "full", "all", "close", "probe every-dim"))
add(t4_row("TF-IDF, whole transcript", tfidf_all, f"{REPO}/hermeneutics/results/tfidf*/auroc.csv", dict(probe="tfidf", position="transcript"), "TF-IDF transcript"))
add(t4_row("TF-IDF, agent's words", tfidf_all, f"{REPO}/hermeneutics/results/tfidf*/auroc.csv", dict(probe="tfidf", position="agent"), "TF-IDF own words"))
add(t4_row("lexical null (think windows)", tfidf_all, f"{REPO}/hermeneutics/results/tfidf*/auroc.csv", dict(probe="tfidf", position="think"), "lexical null"))
add(t4_row("judge: Claude Haiku 5.5", judges, JUDGES, dict(probe="judge:anthropic/claude-haiku-5.5"), "judge Haiku", has_loto=False))
add(t4_row("judge: Claude Sonnet 5.5", judges, JUDGES, dict(probe="judge:anthropic/claude-sonnet-5.5"), "judge Sonnet", has_loto=False, n_note=" of 373"))
add(t4_row("judge: GPT-5 mini", judges, JUDGES, dict(probe="judge:openai/gpt-5-mini"), "judge GPT-5 mini", has_loto=False))
T4.append("| random-weights reader | not run | not run | not run | not run |")
T4.append("| reader interrogation | not run | not run | not run | not run |")
TD3.append("| random-weights reader | not run | not run |")
TD3.append("| reader interrogation | not run | not run |")
T4 = "\n".join(T4)
TD3 = "\n".join(TD3)
son_p = pick(judges, arm="pooled", probe="judge:anthropic/claude-sonnet-5.5", scope="pooled")
son_c = pick(judges, arm="cheat", probe="judge:anthropic/claude-sonnet-5.5", scope="pooled")
SON_D3 = f"{son_p['agents']} of 600 and {son_c['agents']} of 227"
led(SON_D3, "Table D3 caption: the Sonnet judge's agents in the pooled and cheat arms", rel(JUDGES),
    "arm=pooled and arm=cheat, probe=judge:anthropic/claude-sonnet-5.5, scope=pooled -> agents")
# judge coverage and cost from the judges file
son = pick(judges, arm="solo", probe="judge:anthropic/claude-sonnet-5.5", scope="pooled")
led(f"{son['agents']} of 373", "Table 4 Sonnet row col solo agents scored", rel(JUDGES), f"arm=solo,probe=judge:anthropic/claude-sonnet-5.5,scope=pooled -> agents (coverage {float(son['coverage']):.3f} of all 600)")
loto =[float(pick(pca, **base, features="raw", scope=f"LOTO {a}")["auroc"]) for a in AUTH]
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
    led(f"{n_agent} agent posts ({n_all} messages incl. system; TSV posts column {r[7]})", f"NUMBERS only: basis for Table 5 tb=60 pilot row {arm} (posts input for {r[0]})",
        rel(os.path.join(PILOT, r[0], "communication", "flipt-02e2#1.json")), "count of messages with sender != 'system'")
dose = tsv_rows(DOSE, keep_arm_header=True)
hdrd = dose[0]
assert hdrd == ["arm", "n", "crossed+", "pass", "mPosts", "mWPosts", "mDisc", "mSol", "mCrs", "mAsm"], hdrd
dose = {r[0]: r for r in dose[1:]}

ARMNAME = {"none": "none (empty board)", "neutral": "neutral post", "neutralL0": "neutral post", "procoordL1": "pro-coordination, mild",
           "procoordL2": "pro-coordination, explicit", "anticoord": "anti-coordination (hidden regulator)"}
T5 = ["| set | turn budget | arm | n | crossed or assembled | passed hidden tests | mean board posts (workers only) | mean discussed | mean solicited | mean crossed | mean assembled |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def agent_posts25(run):
    """Agent posts on the board of a turn-budget-25 pilot run (sender != 'system'); a run with no board file has none."""
    fs = glob.glob(os.path.join(PILOT, run, "communication", "*.json"))
    assert len(fs) <= 1, (run, fs)
    return sum(1 for m in json.load(open(fs[0]))["messages"] if m.get("sender") != "system") if fs else 0


for arm in ("none", "neutral", "procoordL2"):
    d = arms25[arm]
    led(f"{d['crossed+']}/{d['n']}", f"Table 5 tb=25 row {arm} col crossed or assembled", rel(TB25),
        f"count of collab in (crossed, assembled) over rows pilot_{arm}_s*; n = row count")
    runs25 = sorted(os.path.basename(x) for x in glob.glob(os.path.join(PILOT, f"pilot_{arm}_s*")))
    assert len(runs25) == d["n"], (arm, runs25)
    per = [agent_posts25(x) for x in runs25]
    mean25 = sum(per) / len(per)
    k_post = sum(1 for x in per if x > 0)
    posts25 = "0" if mean25 == 0 else f"{mean25:.1f} ({WORD[k_post]} team{'s' if k_post > 1 else ''}, {WORD[sum(per)]} posts)"
    led(posts25, f"Table 5 tb=25 row {arm} col mean board posts", rel(PILOT) + f"/pilot_{arm}_s*/communication/*.json",
        f"agent posts per run (sender != 'system'; no board file = 0): {dict(zip(runs25, per))}")
    T5.append(f"| pilot | 25 | {ARMNAME[arm]} | {d['n']} | {d['crossed+']}/{d['n']} | 0/{d['n']} | {posts25} | | | 0 | 0 |")
led("passed False in all 12 rows; crs 0, asm 0", "Table 5 tb=25 rows cols passed, crossed, assembled", rel(TB25), "columns passed, crs, asm of every pilot_* row")
led("agent posts 0 in neutral 5/5, none 2/2, pro-coord 4/5; s4 posted 2", "NUMBERS only: corroboration of the tb=25 posts cells", rel(f"{SUB}/VERIFICATION_LOG.md"),
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
# turn budgets, read from every run.json of the pilots and the dose-response matrix
def budget(run_dir):
    tb = json.load(open(os.path.join(run_dir, "run.json")))["config"]["turn_budget"]
    return tb["budget"] if isinstance(tb, dict) else tb
tb25_runs = sorted(glob.glob(os.path.join(PILOT, "pilot_*")))
tb60_runs = sorted(glob.glob(os.path.join(PILOT, "pilot60_*"))) + sorted(glob.glob(os.path.join(PILOT, "confirm_doseresponse", "cm_*")))
assert len(tb25_runs) == 12 and len(tb60_runs) == 6 + 25, (len(tb25_runs), len(tb60_runs))
assert {budget(r) for r in tb25_runs} == {25} and {budget(r) for r in tb60_runs} == {60}
led("25 / 60", "Table 5 col turn budget and caption; Methods, Data (two turn budgets)", rel(PILOT) + "/*/run.json, confirm_doseresponse/*/run.json",
    f"config.turn_budget: 25 in the {len(tb25_runs)} pilot_* runs; 60 in the {len(tb60_runs)} pilot60_* and cm_* runs")

# ----------------------------------------------------------------------------------------------------------------
# HF README facts
# ----------------------------------------------------------------------------------------------------------------
hf = open(HF_README).read()
hf_total = re.search(r"\*\*(4,157)\*\* unique completed, validated runs", hf).group(1)
HF_URL = "https://huggingface.co/datasets/unrulyabstractions/apart-swarm"
for v, w, q in [(hf_total, "Appendix B: the controls dataset's valid runs", "line 3: 'with **4,157** unique completed, validated runs'"),
                ("n_agents = 1 in every run (single-agent scope)", "Methods, Data (one agent per run); Table F1 column header", "section 'Single-agent scope'"),
                ("rung order none < exposed < tried < opened < used < tampered < checked < shipped", "Methods, Data: the cheat ladder", "section 'Labels'")]:
    led(v, w, f"{HF_URL}/raw/main/README.md (fetched 2026-11-07 with the account token)", q)
assert sum(valid.values()) == int(hf_total.replace(",", "")), (sum(valid.values()), hf_total)
led(hf_total, "Appendix B: the controls dataset's valid runs equal the Table E1 valid-run column summed", rel(LABELS), f"sum of valid runs over authors: {valid}")

# the sprint window: every run was made inside it
import datetime
lt = [float(r["launch_time"]) for r in runs if r["launch_time"]]
SPRINT = "https://apartresearch.com/sprints/swarm-alignment-2026 (read 2026-11-08)"
led("November 6 to 8, 2026", "Title block: sprint window", SPRINT, "header 'Nov 6 - 8, 2026'; 'Submissions close Nov 8, 2026 at 11:59 PM Anywhere on Earth'")
led("Track 4; Track 2", "Title block: tracks", SPRINT, "submit form 'Tracks': 'Track 2: The Science of Agent Networks', 'Track 4: Multi-Agent Oversight and Control'")
assert min(lt) > datetime.datetime(2026, 11, 6, 21, 30, tzinfo=datetime.timezone.utc).timestamp(), "first controls launch precedes the kick-off"
led("every controls launch after the kick-off keynote (2026-11-06 21:30 UTC)", "Introduction, Pre-existing work; Appendix B: the controls dataset was produced inside the sprint window",
    rel(RUNS) + "; " + SPRINT, "min(launch_time) over all rows against the schedule's kick-off time")
_ev = sorted(os.path.basename(json.load(open(os.path.join(x, "run.json")))["eval_log"])[:19] for x in tb25_runs + tb60_runs)
assert all(x.startswith("2026-11-07") for x in _ev), _ev
led("every board-seed run inside the sprint window", "Introduction, Pre-existing work; Appendix B: the board-seed pilots and dose-response matrix were made inside the sprint",
    rel(f"{PILOT}") + "/*/run.json, confirm_doseresponse/*/run.json", f"eval_log file names over all {len(_ev)} run.json files: {_ev[0]} to {_ev[-1]}")

# judge facts
led("460/600 scored; 262 solo", "Table 4 caption; Table D3 caption; Appendix E judge setup note", "VERIFICATION_LOG.md", "11:05 entry: 'Claude Sonnet 5.5 0.796 on 262 solo agents (460/600 scored)'")
led("reasoning disabled (Haiku), minimal (gpt-5-mini), low (Sonnet); tool results cut to 500 chars; transcript tail cut to 12,000 tokens", "Appendix E, judge setup note", "VERIFICATION_LOG.md", "11:05 entry, 'API judges' bullet")
led("95%", "Table E3 intervals; captions of Tables 1, 2, 3, D1, D2, D4 and Figure 1: the bootstrap interval level", "hermeneutics/probes.py", "boot_ci: np.percentile(vals, [2.5, 97.5]); docstring 'AUROC with a 95% bootstrap over runs'")

# ----------------------------------------------------------------------------------------------------------------
# Table F1: label-gap map (DATASETS.md, datasets/ai-village.md)
# ----------------------------------------------------------------------------------------------------------------
T6 = """| output | controls dataset (own, single-agent) | board-seed environment (own, three agents) | AI Village (external, eval-only) |
|---|---|---|---|
| cheating | direct (cheat-ladder rung) | direct (team cheat-ladder rung) | gap (no task or grader; judge only) |
| coordination | zero by construction (one agent) | direct (collaboration ladder, board counts) | derivable (chat graph) |
| cooperation | zero by construction | not separable from collaboration (one shared goal) | derivable, weak (coordination serving an agent's own goal) |
| collaboration | zero by construction | direct for the shared exploit (crossed, assembled), under one shared goal | derivable (coordination under an active village-wide goal) |
| collusion | zero by construction | derivable (rung at least used, and collaboration at least crossed) | gap (needs cheating; judge only) |
| scheming | zero by construction | gap (no covertness signal) | gap (no covertness signal; judge only) |"""
led("label map cells", "Table F1", "DATASETS.md (commit 45f8119); DATASET_DETAILS.md; datasets/ai-village.md",
    "DATASET_DETAILS.md 'Notation' (hierarchy, zeros by construction), section 7 AI Village entry, section 8 own data; ai-village.md 'Labels for our six outputs'")
led("hierarchy Sch => Col => (Ch and Cb); Cb => Cp => Cd", "Section 4 Q6 prose", rel(f"{REPO}/DATASET_DETAILS.md"), "'Notation' section, label hierarchy")
led("no external dataset uses our authors", "Section 4 Q6 prose", rel(f"{REPO}/DATASET_DETAILS.md"), "'Key takeaways', first bullet")
led("closest external sets: SALT-NLP agent-collusion, NARCBench, Colosseum, MAST-Data, Terminal Wrench", "Section 4 Q6 prose", rel(f"{REPO}/DATASETS.md"), "Ranking rows 2 to 6; DATASET_DETAILS.md 'Plan by priority'")
led("licence allows evaluation only", "Results, label map; Discussion; Table F1 header 'eval-only'", "datasets/ai-village.md; DATASETS.md",
    "ai-village.md 'Constraints': research and analysis only, no training or fine-tuning without written permission; DATASETS.md ranking row 'AI Village', role 'eval-only licence'")

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
TB2 = sweep(full, FULL, "full", "Table D2: every-dimension probe")

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
# Figure 1 (unseen-author transfer): the plotted values must equal the source rows that Table D4 prints
# ----------------------------------------------------------------------------------------------------------------
FIGV = f"{REPO}/report/figures/fig1_transfer_values.json"
_fig = json.load(open(FIGV))
_src = {"hermeneutics/results/pca_agent/auroc.csv": pca, "hermeneutics/results/full_agent/auroc.csv": full,
        "hermeneutics/results/pca_turn/auroc.csv": turn, "hermeneutics/results/tfidf/auroc.csv": tfidf_all}
_name2a = {NAME[a]: a for a in AUTH}
assert len(_fig) == 20, len(_fig)
for e in _fig:
    kw = dict(kv.split("=", 1) for kv in e["selector"].split(","))
    tab = _src[e["source_file"]]
    if e["source_file"].endswith("tfidf/auroc.csv"):
        kw["probe"] = "tfidf"
    r = pick(tab, **kw)
    assert abs(float(r["auroc"]) - e["auroc"]) < 1e-12 and abs(float(r["ci_lo"]) - e["lo"]) < 1e-12 and abs(float(r["ci_hi"]) - e["hi"]) < 1e-12, e
    n_, t_, rd_ = cov[_name2a[e["author"]]]
    assert abs(rd_ / t_ - e["think_coverage"]) < 1e-12, e
led("20 plotted values (AUROC and interval) and 5 think-coverage labels", "Figure 1 markers, whiskers and author labels", rel(FIGV),
    "each entry equals its source row (auroc, ci_lo, ci_hi) and the Table E1 coverage; the same cells print in Table D4")
FIG_CI = "95%"
assert "np.percentile(vals, [2.5, 97.5])" in open(f"{REPO}/hermeneutics/probes.py").read()
led(FIG_CI, "Figure 1 caption: interval level", "hermeneutics/probes.py", "boot_ci: np.percentile(vals, [2.5, 97.5])")
_tl = {e["selector"].split("layer=")[1].split(",")[0] for e in _fig if e["method"] == "Per-turn probe"}
assert len(_tl) == 1, _tl
FIG_TURN_LAYER = _tl.pop()
led(FIG_TURN_LAYER, "Figure 1 caption: the per-turn probe's layer", rel(FIGV), "selector of the 'Per-turn probe' entries (layer=38)")

# ----------------------------------------------------------------------------------------------------------------
# prose numbers added in the final verification round
# ----------------------------------------------------------------------------------------------------------------
N_PROBED = str(len(agents))
led(N_PROBED, "Abstract: the stratified subsample of transcripts teacher-forced through the reader", rel(AGENTS), "row count (one agent per run)")
_bi = pick(pca, **base, features="raw", scope="pooled")
N_SOLO = _bi["agents"]
led(N_SOLO, "Table 3 caption and col 'solo agents scored'", rel(PCA), sel(**base, features="raw", scope="pooled") + " -> agents", raw_where=True)
SON_SOLO = son["agents"]
led(SON_SOLO, "Table 4 caption: solo agents the Sonnet judge scored", rel(JUDGES), "arm=solo,probe=judge:anthropic/claude-sonnet-5.5,scope=pooled -> agents")
N_BOARD = str(len(tb25_runs) + len(tb60_runs))
led(N_BOARD, "Results, label map: our board-seed runs (pilots and dose-response matrix)", rel(PILOT) + "/pilot_*, pilot60_*, confirm_doseresponse/cm_*", f"run directories: {len(tb25_runs)} + {len(tb60_runs)}")
assert all(dose[a][1] == "5" for a in dose), {a: dose[a][1] for a in dose}
led("5 per arm", "Abstract; Discussion; Conclusion: 'five seeds per arm'", rel(DOSE), "column n = 5 in every arm row")
SH = {}
for a in AUTH:
    for cond in ("solo", "cheat"):
        v = [r for r in labels if r["author"] == a and r["valid"] == "True" and r["condition"] == cond]
        SH[a, cond] = sum(r["label"] == "cheat" for r in v) / len(v)
assert all(SH[a, "cheat"] > SH[a, "solo"] for a in AUTH)
led("5 of 5 authors cheat more often in the cheat arm", "Methods, Controls: 'every author cheats more often there than in the solo arm'", rel(LABELS),
    "share of label == 'cheat' among valid runs per author, solo -> cheat arm: "
    + ", ".join(a + " " + format(SH[a, "solo"], ".3f") + " -> " + format(SH[a, "cheat"], ".3f") for a in AUTH))
_lo = {a: float(pick(pca, **base, features="raw", scope=f"LOTO {a}")["ci_lo"]) for a in AUTH}
assert all(v > 0.5 for v in _lo.values()), _lo
_amin = min(_lo, key=_lo.get)
led(f"5 of 5; lowest lower bound {_amin} {_lo[_amin]:.3f}", "Results, transfer; Conclusion: every unseen author read above chance", rel(PCA),
    "ci_lo > 0.5 in every LOTO row, arm=solo,layer=all,position=both,features=raw")
_tq = float(pick(tfidf_all, arm="solo", probe="tfidf", position="transcript", scope="LOTO qwen")["auroc"])
_pq = float(pick(pca, **base, features="raw", scope="LOTO qwen")["auroc"])
assert _tq > _pq
led(f"{_tq:.3f} > {_pq:.3f}", "Results, text classifier: on unseen authors the classifier leads for Qwen3.8-27B", "hermeneutics/results/tfidf/auroc.csv; " + rel(PCA), "LOTO qwen: TF-IDF whole transcript vs PCA probe")
for ref, yr, src, q in [("[1]", "2026 (authors as displayed: conti, ShivamR, Jacob Sharf, jain18ayush, johnyan)", "https://www.lesswrong.com/posts/CZLhfNGaDqtYxvGJD/model-hermeneutics-monitoring-closed-weight-models-with-open", "postedAt 2026-09-30T20:43:51Z"),
                        ("[3]", "2026 (arXiv:2606.32008)", "https://arxiv.org/abs/2606.32008", "citation_date 2026/06/30"),
                        ("[4]", "2026 (arXiv:2604.01151)", "https://arxiv.org/abs/2604.01151", "citation_date 2026/04/01 (v5 revised 15 Oct 2026)"),
                        ("[8]", "2026 (arXiv:2609.36308)", "https://arxiv.org/abs/2609.36308 (read 2026-11-08)", "citation_date 2026/09/28; title 'CheatBench: Measuring Reward Gaming in AI Agents'; 13 citation_author entries, first Phan, Long"),
                        ("[9]", "2026 (arXiv:2609.33102)", "https://arxiv.org/abs/2609.33102 (read 2026-11-08)", "citation_date 2026/09/27; title 'ORBIT: A Framework for Multi-Agent Safety and Security Evaluations'; 4 citation_author entries: Hagag, Anderson, Chakraborty, Schroeder de Witt"),
                        ("[10]", "2026 (November)", "https://docs.google.com/document/d/1-Ph3v3OPblL2tBs2lxPLk3oPMDhVh686zzhywludDgc (public text export read 2026-11-09)", "title block: 'SPAR', 'November, 2026'; authors Rigel Cierniak, Lucy Kappai, Ian Rios-Sialer; mentor Ben Maltbie; Code: github.com/unrulyabstractions/spar-guardian-agents"),
                        ("[5]", "2025 (arXiv:2502.14143; Technical Report 1)", "https://arxiv.org/abs/2502.14143", "citation_date 2025/02/19; comments 'Cooperative AI Foundation, Technical Report #1'"),
                        ("[6]", "2026", "https://huggingface.co/datasets/aidigestorg/ai-village (card read 2026-11-08 with the account token)", "card citation line: 'AI Digest, \"AI Village dataset\", 2026.'"),
                        ("[7]", "2026", HF_URL, "README 'Built 2026-11-07T10:06:53Z'"),
                        ("[11]", "2026", "https://huggingface.co/datasets/unrulyabstractions/apart-swarm-align", "VERIFICATION_LOG.md entry '2026-11-09 01:58 UTC: board-seed runs published to the new private dataset'")]:
    led(yr, f"References {ref}: year", src, q)
led("n.d.", "References [2]: no year on the page", "https://aryafayyazi.github.io/GreyBox/paper.html (read 2026-11-08)", "'Grey Box: Reading Closed Agents Through Their APIs Alone.' 'Arya Fayyazi · preprint in preparation.'")

# counts that the prose writes as words
_val = [r for r in labels if r["valid"] == "True"]
_tasks = sorted({r["task"] for r in _val}); _swe = sorted({r["task"] for r in _val if r["task_name"] == "software_engineering"})
_conds = sorted({r["condition"] for r in _val}); _auths = sorted({r["author"] for r in _val})
assert (len(_tasks), len(_swe), len(_conds), len(_auths)) == (9, 2, 2, 5), (_tasks, _swe, _conds, _auths)
led(f"{len(_tasks)} tasks", "Abstract; Methods, Data; Appendix E: 'nine tasks'", rel(LABELS), f"distinct task over valid rows: {_tasks}")
led(f"{len(_swe)} software-engineering task instances", "Appendix E: 'software engineering (two issues of one task)'", rel(LABELS), f"distinct task where task_name == software_engineering: {_swe}")
_tnames = sorted({r["task_name"] for r in _val})
assert len(_tnames) == 8, _tnames
led(f"{len(_tnames)} CheatBench tasks", "Appendix E: 'They cover eight CheatBench tasks'", rel(LABELS), f"distinct task_name over valid rows: {_tnames}")
led("Apache-2.0 (fork of wlanderson0/orbit, created 2026-10-09, before the sprint)", "Appendix B: 'ORBIT [9] (Apache-2.0)' under 'Before the sprint'",
    "GitHub API repos/unrulyabstractions/orbit (read 2026-11-08)", "created_at 2026-10-09T19:43:21Z; fork true; parent wlanderson0/orbit; license Apache-2.0", raw_where=True)
led("MIT (fork of centerforaisafety/cheatbench, created 2026-10-09, before the sprint)", "Appendix B: 'CheatBench [8] (Center for AI Safety, MIT licence)' under 'Before the sprint'",
    "GitHub API repos/unrulyabstractions/cheatbench (read 2026-11-08)", "created_at 2026-10-09T14:25:09Z; fork true; parent centerforaisafety/cheatbench; license MIT", raw_where=True)
led(f"{len(_conds)} arms", "Methods, Data: 'one of two arms'", rel(LABELS), f"distinct condition over valid rows: {_conds}")
led(f"{len(_auths)} authors (B.II trains on {len(_auths) - 1} and scores the fifth)", "Abstract; Introduction; Methods; Results; Conclusion: 'five authors', 'four authors', 'the fifth'; Table 2 row 'author identity (5 classes)'", rel(LABELS), f"distinct author over valid rows: {_auths}")
_cfg = [json.load(open(os.path.join(x, "run.json")))["config"] for x in tb25_runs + tb60_runs]
_n3 = sum(1 for c in _cfg if c["n_agents"] == 3 and c["regulators"] == 0); _n4 = sum(1 for c in _cfg if c["n_agents"] == 4 and c["regulators"] == 1)
assert _n3 + _n4 == len(_cfg) and all(c["access"] == "split" for c in _cfg), (_n3, _n4)
_held = set()
for x in tb25_runs + tb60_runs:
    h = set()
    for l in open(os.path.join(x, "signals.jsonl")):
        j = json.loads(l)
        if j.get("row") == "agent" and j.get("holds"):
            h |= set(j["holds"])
    _held.add(tuple(sorted(h)))
assert _held == {("A", "B", "C")}, _held
led(f"3 worker agents in all {len(_cfg)} runs ({_n3} with no regulator; {_n4} anti-coordination runs add 1 regulator); 3 pieces, one per agent",
    "Abstract; Methods, Data; Table 4 caption (Results, board signal); Conclusion; Table F1 header: 'three agents', 'three worker agents', 'three-agent', 'three pieces, one per agent'",
    rel(PILOT) + "/*/run.json, confirm_doseresponse/*/run.json, */signals.jsonl", "config.n_agents and config.regulators; config.access == split in every run; agent rows' holds = {A, B, C} in every run",
    raw_where=True)
_views = set()
for f in glob.glob(f"{REPO}/hermeneutics/results/tfidf*/auroc.csv"):
    _views |= {r["position"] for r in rows(f)}
assert _views == {"transcript", "agent", "think"}, _views
led(f"{len(_views)} views", "Methods, Controls: the text classifier 'in three views'", "hermeneutics/results/tfidf*/auroc.csv", f"distinct position: {sorted(_views)}")
_jud = sorted({r["probe"] for r in judges})
assert len(_jud) == 3, _jud
led(f"{len(_jud)} judges", "Abstract; Introduction; Methods; Discussion; Conclusion: 'three (cheap) LLM judges'", rel(JUDGES), f"distinct probe: {_jud}")
_vis = open(f"{REPO}/PROJECT_DESCRIPTION.md").read().split("#Probe Outputs", 1)[1].split("\n#", 1)[0]
_outs = [l for l in _vis.splitlines() if l.startswith("- ") and ":" in l]
assert len(_outs) == 6, _outs
led(f"{len(_outs)} outputs", "Introduction contribution 3; Related Work; Results, label map: 'six collusion-related concepts', 'six probe outputs'", "PROJECT_DESCRIPTION.md", "bullets under '#Probe Outputs': " + ", ".join(l[2:].split(":")[0] for l in _outs))
led("three failure modes", "Related Work: '[5] names collusion as one of three failure modes'", "https://arxiv.org/abs/2502.14143 (read 2026-11-08)",
    "abstract: 'identifying three key failure modes (miscoordination, conflict, and collusion)'")

# number words of round 3
_rj = [json.load(open(os.path.join(x, "run.json"))) for x in tb25_runs + tb60_runs]
_models = {r["model"] for r in _rj}; _tasks_b = {r["task"] for r in _rj}
_issues = set()
for x in tb25_runs + tb60_runs:
    for l in open(os.path.join(x, "episodes.jsonl")):
        _issues.add(json.loads(l)["id"])
assert len(_models) == 1 and len(_tasks_b) == 1 and len(_issues) == 1, (_models, _tasks_b, _issues)
led(f"1 author ({_models.pop()}), 1 task ({_tasks_b.pop()}), 1 issue ({_issues.pop()}) over all {len(_rj)} board-seed runs",
    "Methods, Data; Results, label map; Discussion; Limitations; Appendix A: 'one author', 'one task', 'one software-engineering issue'",
    rel(PILOT) + "/*/run.json and */episodes.jsonl", "run.json model and task; episodes.jsonl id", raw_where=True)
_seed_texts = {v for v in seed_table.values() if v != "USE_REGULATOR"}
_posts = collections.defaultdict(list)
for x in tb25_runs + tb60_runs:
    cfg = json.load(open(os.path.join(x, "run.json")))["config"]
    fs = glob.glob(os.path.join(x, "communication", "*.json"))
    n = sum(1 for m in json.load(open(fs[0]))["messages"] if m.get("sender") == "system" and m.get("content") in _seed_texts) if fs else 0
    _posts["seeded" if cfg.get("board_seed") else "unseeded"].append(n)
assert set(_posts["seeded"]) == {1} and set(_posts["unseeded"]) == {0}, dict(_posts)
led(f"1 seed post in each of {len(_posts['seeded'])} seeded runs; 0 in the {len(_posts['unseeded'])} empty-board and regulator runs",
    "Abstract; contribution 2; Methods, Data; Results, board signal; Discussion; Conclusion: 'one message', 'one post', 'one planted board message'",
    rel(PILOT) + "/*/communication/*.json", "system messages whose content is a SEED_TABLE text (board_seed.py)", raw_where=True)
_d = {a: dose[a] for a in dose}
_rd1 = float(_d["procoordL1"][6]) / float(_d["neutralL0"][6]); _rd2 = float(_d["procoordL2"][6]) / float(_d["neutralL0"][6])
_ra1 = float(_d["procoordL1"][9]) / float(_d["neutralL0"][9])
assert all(1.5 <= x <= 2.5 for x in (_rd1, _rd2, _ra1)), (_rd1, _rd2, _ra1)
led(f"discussion x{_rd1:.2f} (mild) and x{_rd2:.2f} (explicit) over neutral; assembly x{_ra1:.2f} (mild) over neutral",
    "NUMBERS only: basis for 'roughly double discussion' and 'the mild post doubles assembly' (Results, board signal)", rel(DOSE),
    "mDisc procoordL1 and procoordL2 over neutralL0; mAsm procoordL1 over neutralL0; each ratio between 1.5 and 2.5", raw_where=True)
_drops = {}
for scope in ["pooled"] + AUTH:
    raw_ = float(pick(pca, **base, features="raw", scope=scope)["auroc"])
    for ft in FEATS[1:]:
        _drops[(scope, ft)] = raw_ - float(pick(pca, **base, features=ft, scope=scope)["auroc"])
_fr = float(pick(full, arm="solo", probe="full", turn_labels="agent", layer="all", position="close", pooling="agent", features="raw", scope="pooled")["auroc"])
for ft in FEATS[1:]:
    _drops[("every dimension", ft)] = _fr - float(pick(full, arm="solo", probe="full", turn_labels="agent", layer="all", position="close", pooling="agent", features=ft, scope="pooled")["auroc"])
_pa = _drops[("pooled", "author, task, condition erased")]
assert max(_drops.values()) < 0.05 and 0.01 <= _pa < 0.05, (_drops, _pa)
led(f"pooled drop {_drops[('pooled', 'author erased')]:.3f} (author erased) and {_pa:.3f} (author, task, condition erased); largest drop in Table 1 {max(_drops.values()):.3f}",
    "NUMBERS only: basis for 'moves it a few hundredths' (Results, ambiguous arm)", rel(PCA) + "; " + rel(FULL), "raw minus erased AUROC over the Table 1 probe rows; all below 0.05", raw_where=True)
_calls = {}
for f in sorted(glob.glob(f"{REPO}/data/judge_*.jsonl")):
    if f.endswith("judge_prompts.jsonl"):
        continue
    rr = [json.loads(l)["run"] for l in open(f)]
    assert len(rr) == len(set(rr)), f
    _calls[os.path.basename(f)] = len(rr)
assert len(_calls) == 3, _calls
led("1 call per agent per judge",
    "Methods, Controls ('read each transcript once'); Appendix E judge setup ('one prompt per agent')", "data/judge_*.jsonl",
    "one line per run, no run twice: " + ", ".join(f"{k} {v} lines" for k, v in _calls.items()) + "; Sonnet's file stops at its in-code spend cap", raw_where=True)

# ----------------------------------------------------------------------------------------------------------------
# assemble
# ----------------------------------------------------------------------------------------------------------------
tpl = open(os.path.join(os.path.dirname(__file__), "report_template.md")).read()
# Table B1: the number of adapted files, counted from the table itself
_b1 = tpl.split("**Table B1.**", 1)[1].split("\n\n", 2)[1]
N_B1 = sum(1 for l in _b1.splitlines() if l.startswith("| ") and not l.startswith("| file in this project") and not l.startswith("|---"))
_pre = [l for l in open(f"{REPO}/hermeneutics/PREEXISTING.md") if l.startswith("| `hermeneutics/")]
led(N_B1, "Table B1 caption: adapted files", "report/gen/report_template.md Table B1 rows; hermeneutics/PREEXISTING.md", f"rows in Table B1; PREEXISTING.md lists {len(_pre)} at build time")
if len(_pre) != N_B1:
    print(f"NOTE: Table B1 has {N_B1} rows and hermeneutics/PREEXISTING.md lists {len(_pre)} files")
for k, v in {"<<TM1>>": TM1, "<<TM2>>": TM2, "<<TM3>>": TM3, "<<T1>>": T1, "<<T2>>": T2, "<<T3>>": T3, "<<T4>>": T4, "<<T5>>": T5,
             "<<T6>>": T6, "<<TB1>>": TB1, "<<TB2>>": TB2, "<<ABS_BI>>": ABS_BI, "<<ABS_LOTO>>": ABS_LOTO,
             "<<ABS_ERASED>>": ABS_ERASED, "<<TC>>": TC, "<<ABS_JUDGE>>": ABS_JUDGE, "<<ABS_TFIDF>>": ABS_TFIDF, "<<TD3>>": TD3, "<<SON_D3>>": SON_D3, "<<NULL_LO>>": NULL_LO, "<<NULL_HI>>": NULL_HI, "<<HF_TOTAL>>": hf_total,
             "<<N_PROBED>>": N_PROBED, "<<N_SOLO>>": N_SOLO, "<<SON_SOLO>>": SON_SOLO, "<<N_BOARD>>": N_BOARD, "<<N_B1>>": WORD[N_B1], "<<FIG_CI>>": FIG_CI, "<<FIG_TURN_LAYER>>": FIG_TURN_LAYER}.items():
    assert k in tpl, k
    tpl = tpl.replace(k, v)
assert "<<" not in tpl, re.findall(r"<<[A-Z_0-9]+>>", tpl)
led("0.5", "Table 2 col 'chance / majority rate': the three AUROC rows", "hermeneutics/probes.py", "auc() wraps sklearn roc_auc_score (lines 71-77); a score unrelated to the label has AUROC 0.5", raw_where=True)
assert "torch.bfloat16" in open(f"{REPO}/hermeneutics/extract.py").read()
led("bf16", "Table E1 reader row; Table E3 reader row", "hermeneutics/extract.py", "layer.to(torch.bfloat16) (line 124); docstring '52 GB bf16'", raw_where=True)
assert "L2 logistic regression" in open(f"{REPO}/hermeneutics/probes.py").read()
led("L2", "Table E3 probe row: 'L2 logistic regression'", "hermeneutics/probes.py", "docstring line 18 'L2 logistic regression'; LogisticRegression with its default l2 penalty", raw_where=True)
led("min(256, half the training rows)", "Table E3 probe row", "hermeneutics/probes.py", "k = min(PCA_DIM, max(10, len(Xtr) // 2)) (line 128), PCA_DIM = 256 (line 48)", raw_where=True)
assert "np.nanpercentile(pv, 95)" in open(f"{REPO}/hermeneutics/probes.py").read()
led("95th percentile", "Table 1 caption; Appendix D Tables D1 and D2 col 'null 95th'", "hermeneutics/probes.py", "null_95 = np.nanpercentile(pv, 95) over the shuffled fits (line 451)", raw_where=True)
_lv = sorted("L" + k[1] for k in seed_table if seed_table[k] != "USE_REGULATOR")
assert _lv == ["L0", "L1", "L2"], _lv
led(", ".join(_lv), "Appendix C, Table C1 condition labels", BS.replace(REPO + "/", ""), "SEED_TABLE keys with a text: " + ", ".join(str(k) for k in seed_table if seed_table[k] != "USE_REGULATOR"), raw_where=True)
_names = "; ".join(f"{NAME[a]} ({a})" for a in AUTH) + "; judges " + ", ".join(j.split("/")[-1] for j in sorted({r["probe"] for r in judges}))
led(_names, "author and judge names throughout; Appendix D column keys", "data/runs.csv; hermeneutics/results/judges/auroc.csv",
    "model_id per author (Table E1) and judge probe names", raw_where=True)
led("layers 26, 38, 51", "Table B1 row 'extract'", "hermeneutics/PREEXISTING.md; hermeneutics/extract.py", "PREEXISTING.md extract row: 'layers 26/38/51'; the extraction log entries 07:40 and 07:45", raw_where=True)
led("500-character", "Table B1 row 'judge'; Appendix E judge setup", "hermeneutics/PREEXISTING.md; VERIFICATION_LOG.md", "PREEXISTING.md judge row: 'the 500-character cut per tool result'; 11:05 entry: 'tool results cut to 500 chars'", raw_where=True)
# number words of round 4, read from the assembled draft and its sources
_ahead = [a for a in AUTH if float(pick(pca, **base, features="raw", scope=a)["auroc"]) > float(tfidf_solo(a)["auroc"])]
assert _ahead == ["qwen"], _ahead
led("1 author (Qwen3.8-27B)", "Results, text classifier: 'the one author where the probe stays ahead within itself'", rel(PCA) + "; hermeneutics/results/tfidf/auroc.csv",
    "type-A raw PCA AUROC above the TF-IDF whole-transcript AUROC only for qwen: "
    + ", ".join(a + " " + f3(pick(pca, **base, features="raw", scope=a)["auroc"]) + " vs " + f3(tfidf_solo(a)["auroc"]) for a in AUTH), raw_where=True)
_arms = sorted({r["arm"] for r in pca})
assert _arms == ["cheat", "pooled", "solo"], _arms
led("2 other arms (pooled, cheat)", "Results, text classifier; Table 3 caption: 'the other two arms'", rel(PCA), f"distinct arm: {_arms}; the solo arm is primary", raw_where=True)
_notrun = [l for l in T4.splitlines() if "not run" in l]
assert len(_notrun) == 2, _notrun
led("2 not-run rows (random-weights reader, reader interrogation)", "Table 3 rows; Methods, Controls; Limitations; Appendix A: 'the two baselines that were not run'",
    "report/gen/gen_report.py Table 3 rows", "rows of Table 3 whose cells read 'not run'", raw_where=True)
led("1 probe, scope pooled (B.I)", "Abstract: 'one probe pooled over authors'", rel(PCA), sel(**base, features="raw", scope="pooled"), raw_where=True)
led("1 reader (Qwen3.8-27B)", "Methods, Authors and reader: 'five authors through one reader'; Table E1 reader row", "VERIFICATION_LOG.md; report/gen/gen_report.py Table E1; hermeneutics/extract.py; hermeneutics/__init__.py",
    "07:20 entry: every transcript rendered through the Qwen3.8-27B chat template; the extraction loads that model alone", raw_where=True)
_q5 = tpl.split("### Can a planted board signal move coordination?", 1)[1].split("\n\n", 2)[1]
_ord = [w for w in ("First,", "Second,", "Third,", "Fourth,", "Fifth,") if w in _q5]
assert _ord == ["First,", "Second,", "Third,", "Fourth,"], _ord
led(f"{len(_ord)} readings", "Results, board signal: 'four readings'", "report/REPORT_DRAFT.md", "the paragraph's ordinals First, Second, Third, Fourth", raw_where=True)
_outrows = [l for l in T6.splitlines()[2:] if l.startswith("| ")]
assert len(_outrows) == 6, _outrows
led(f"{len(_outrows) - 1} other outputs", "Results, label map: 'the other five as zeros by construction'", "Table F1 rows; PROJECT_DESCRIPTION.md",
    f"{len(_outrows)} outputs in Table F1 less cheating", raw_where=True)
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/REPORT_DRAFT.md", "w").write(tpl)

abstract = tpl.split("## Abstract\n", 1)[1].split("\n## ", 1)[0].strip()
plain = abstract.replace("**", "")
wc = len(plain.split())
wc_split = len([w for w in re.split(r"[\s\-]+", plain) if w])  # word processors may count hyphenated parts apart
assert 150 <= wc <= 238 and wc_split <= 248, (wc, wc_split)  # the form allows 150 to 250; keep a margin for either count
open(f"{OUT}/ABSTRACT.txt", "w").write(plain + "\n")
_sub = f"{OUT}/SUBMISSION.md"
if os.path.exists(_sub):
    _st = open(_sub).read()
    _new, _n = re.subn(r"the text of `report/ABSTRACT.txt` \(\d+ words(?:; \d+ with hyphenated words split)?\)",
                       f"the text of `report/ABSTRACT.txt` ({wc} words; {wc_split} with hyphenated words split)", _st)
    assert _n == 1, _n
    if _new != _st:
        open(_sub, "w").write(_new)

hdr = ["# Number ledger for report/REPORT_DRAFT.md", "",
       "Every number in the tables and every number written in digits in the text of the draft traces to a cited source: a result file, a run log, a code constant, a repository document or a public page, read 2026-11-07 to 2026-11-08. One line per number; text cells (setup descriptions, 'n/a' and 'not run' markers) are not numbers. "
       "Paths are relative to the repository root unless absolute. The dataset README was fetched from the Hub with the account token.", "",
       "| value | where in the draft | source | field / selector |", "|---|---|---|---|"]
open(f"{OUT}/NUMBERS.md", "w").write("\n".join(hdr + LEDGER) + "\n")

body = tpl.split("## Abstract\n", 1)[1].split("\n## References", 1)[0]
body_wo_tables = "\n".join(l for l in body.splitlines() if not l.startswith("|"))
print("abstract words", wc, "| with hyphenated words split", wc_split)
print("body words incl. tables", len(body.split()))
print("body words excl. table rows", len(body_wo_tables.split()))
print("ledger lines", len(LEDGER))
