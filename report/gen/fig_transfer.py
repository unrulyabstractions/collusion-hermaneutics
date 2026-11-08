"""Figure 1: unseen-author transfer (type B.II, leave one author out), solo arm.

Reads the same result files and the same selectors as Table 2 in report/gen/gen_report.py, plots one row group per
held-out author with four methods (three reader probes and the TF-IDF text baseline), and cross-checks every plotted
value against the Table 2 cells in report/REPORT_DRAFT.md at 3 decimals. Any mismatch exits non-zero.

Writes report/figures/fig1_transfer.{pdf,png} and report/figures/fig1_transfer_values.json.
Run: .venv/bin/python report/gen/fig_transfer.py
"""
import csv
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # report/gen -> repo root
OUT = f"{REPO}/report/figures"
DRAFT = f"{REPO}/report/REPORT_DRAFT.md"
RENDER = "data/render_summary.csv"

# Table 2 row order and display names (gen_report.py AUTH / NAME)
AUTH = ["ds", "gpt5m", "haiku", "luna", "qwen"]
NAME = {"ds": "DeepSeek V4.1 Flash", "gpt5m": "GPT-5 mini", "haiku": "Claude Haiku 5.5", "luna": "GPT-6 Luna",
        "qwen": "Qwen3.8-27B"}

# Each method: the Table 2 selector (gen_report.py, Table 2 block), the Table 2 column it must equal, and its marks.
# Colours: categorical slots 1-3 of the dataviz reference palette for the three reader probes; the text baseline is
# the neutral ink with a hollow marker. Shapes carry identity a second time, so colour is never the only channel.
METHODS = [
    dict(key="pca", legend="PCA probe", source="hermeneutics/results/pca_agent/auroc.csv",
         sel=dict(arm="solo", probe="pca", turn_labels="agent", layer="all", position="both", pooling="agent",
                  features="raw"),
         columns=["PCA probe, raw"], color="#2a78d6", marker="o", ms=5.6, hollow=False),
    dict(key="every_dim", legend="Every-dimension probe", source="hermeneutics/results/full_agent/auroc.csv",
         sel=dict(arm="solo", probe="full", turn_labels="agent", layer="all", position="close", pooling="agent",
                  features="raw"),
         columns=["full-dimension probe, raw", "every-dimension probe, raw"], color="#eb6834", marker="s", ms=5.0,
         hollow=False),
    dict(key="per_turn", legend="Per-turn probe", source="hermeneutics/results/pca_turn/auroc.csv",
         sel=dict(arm="solo", probe="pca", turn_labels="agent", layer="38", position="close", pooling="turn",
                  features="raw"),
         columns=["per-turn probe, agent AUROC"], color="#1baf7a", marker="D", ms=4.6, hollow=False),
    dict(key="tfidf", legend="TF-IDF text baseline", source="hermeneutics/results/tfidf/auroc.csv",
         sel=dict(arm="solo", position="transcript"),
         columns=["TF-IDF, whole transcript"], color="#52514e", marker="o", ms=5.6, hollow=True),
]

INK = "#0b0b0b"         # primary ink
INK2 = "#52514e"        # secondary ink
MUTED = "#898781"       # muted ink (axis text, chance line)
GRID = "#e1e0d9"        # gridline hairline
AXIS = "#c3c2b7"        # baseline / axis
SURFACE = "#ffffff"     # the page


def rows(rel_path):
    with open(f"{REPO}/{rel_path}") as f:
        return list(csv.DictReader(f))


def pick(tab, **kw):
    out = [r for r in tab if all(str(r[k]) == str(v) for k, v in kw.items())]
    if len(out) != 1:
        sys.exit(f"FAIL: selector {kw} matched {len(out)} rows, expected exactly 1")
    return out[0]


def sel_str(kw):
    return ",".join(f"{k}={v}" for k, v in kw.items())


def f3(x):
    return f"{float(x):.3f}"


# ---------------------------------------------------------------------------------------------------------------
# values, by the Table 2 selectors
# ---------------------------------------------------------------------------------------------------------------
render = rows(RENDER)
cover = {}
for a in AUTH:
    rs = [r for r in render if r["author"] == a]
    t = sum(int(r["turns"]) for r in rs)
    rd = sum(int(r["n_readable"]) for r in rs)
    cover[a] = dict(value=rd / t, selector=f"sum(n_readable)/sum(turns) over rows where author={a} ({rd}/{t})")

tables = {m["key"]: rows(m["source"]) for m in METHODS}
points = []
for a in AUTH:
    for m in METHODS:
        kw = dict(m["sel"], scope=f"LOTO {a}")
        r = pick(tables[m["key"]], **kw)
        points.append(dict(author=NAME[a], method=m["legend"], auroc=float(r["auroc"]), lo=float(r["ci_lo"]),
                           hi=float(r["ci_hi"]), source_file=m["source"], selector=sel_str(kw),
                           think_coverage=cover[a]["value"], think_coverage_source=f"{RENDER}: {cover[a]['selector']}"))

# ---------------------------------------------------------------------------------------------------------------
# cross-check against the Table 2 cells of REPORT_DRAFT.md
# ---------------------------------------------------------------------------------------------------------------
lines = open(DRAFT).read().splitlines()
start = [i for i, ln in enumerate(lines) if ln.startswith("**Table 2.**")]
if len(start) != 1:
    sys.exit(f"FAIL: found {len(start)} '**Table 2.**' lines in {DRAFT}")
tab = []
for ln in lines[start[0] + 1:]:
    if ln.startswith("|"):
        tab.append(ln)
    elif tab:
        break
header = [c.strip() for c in tab[0].strip("|").split("|")]
body = {}
for ln in tab[2:]:
    cells = [c.strip() for c in ln.strip("|").split("|")]
    body[cells[0]] = dict(zip(header, cells))


def column(names):
    hit = [h for h in header if h in names]
    if len(hit) != 1:
        sys.exit(f"FAIL: Table 2 header {header} has no unique column among {names}")
    return hit[0]


CELL = re.compile(r"^([0-9.]+) \[([0-9.]+), ([0-9.]+)\]$")
bad = []
print(f"{'held-out author':<20} {'method':<22} {'plotted':<24} {'Table 2 cell':<24} match")
for a in AUTH:
    row = body.get(NAME[a])
    if row is None:
        sys.exit(f"FAIL: Table 2 has no row '{NAME[a]}'")
    want = row[column(["think coverage"])]
    got = f3(cover[a]["value"])
    ok = got == want
    bad += [] if ok else [(NAME[a], "think coverage", got, want)]
    print(f"{NAME[a]:<20} {'think coverage':<22} {got:<24} {want:<24} {'OK' if ok else 'MISMATCH'}")
    for m in METHODS:
        p = next(q for q in points if q["author"] == NAME[a] and q["method"] == m["legend"])
        cell = row[column(m["columns"])]
        mm = CELL.match(cell)
        got = f"{f3(p['auroc'])} [{f3(p['lo'])}, {f3(p['hi'])}]"
        ok = bool(mm) and (f3(p["auroc"]), f3(p["lo"]), f3(p["hi"])) == mm.groups()
        bad += [] if ok else [(NAME[a], m["legend"], got, cell)]
        print(f"{NAME[a]:<20} {m['legend']:<22} {got:<24} {cell:<24} {'OK' if ok else 'MISMATCH'}")
if bad:
    for b in bad:
        print("MISMATCH:", b, file=sys.stderr)
    sys.exit(f"FAIL: {len(bad)} plotted values disagree with Table 2 at 3 decimals")
print(f"cross-check: all {len(AUTH) * (len(METHODS) + 1)} values equal Table 2 at 3 decimals")

# ---------------------------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})
FS_LABEL, FS_TICK, FS_SMALL = 9, 8.5, 8

W, H = 6.5, 2.8
fig = plt.figure(figsize=(W, H))
LEFT, RIGHT, BOTTOM, TOP = 1.30, 0.10, 0.42, 0.26   # inches
ax = fig.add_axes([LEFT / W, BOTTOM / H, 1 - (LEFT + RIGHT) / W, 1 - (BOTTOM + TOP) / H])

GAP = 1.3                       # blank rows between author groups
n_m = len(METHODS)
ypos = {}
for g, a in enumerate(AUTH):
    for j, m in enumerate(METHODS):
        ypos[(a, m["key"])] = -(g * (n_m + GAP) + j)
y_min = -((len(AUTH) - 1) * (n_m + GAP) + n_m - 1)
ax.set_ylim(y_min - 0.75, 0.75)

X0, X1 = 0.5, 1.0
ax.set_xlim(X0 - 0.012, X1 + 0.008)

# recessive chrome: solid hairline grid at the ticks, axis baseline, no frame
for x in (0.6, 0.7, 0.8, 0.9, 1.0):
    ax.axvline(x, color=GRID, lw=0.5, zorder=0)
ax.axvline(X0, color=MUTED, lw=0.8, ls=(0, (3, 2.5)), zorder=1)          # chance
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.spines["bottom"].set_linewidth(0.6)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
ax.tick_params(axis="x", colors=AXIS, labelcolor=INK2, labelsize=FS_TICK, width=0.6, length=3, pad=2)
ax.set_yticks([])
ax.set_xlabel("Held-out AUROC", fontsize=FS_LABEL, color=INK, labelpad=3)

# group separators: hairlines midway between groups
for g in range(1, len(AUTH)):
    yb = -(g * (n_m + GAP)) + (GAP + 1) / 2
    ax.axhline(yb, color=GRID, lw=0.5, zorder=0)

# "chance" tag beside the reference line, in the empty band above the first group
ax.text(X0 + 0.006, 0.55, "chance", fontsize=FS_SMALL, color=MUTED, ha="left", va="center")

# whiskers then dots
for p in points:
    a = next(k for k, v in NAME.items() if v == p["author"])
    m = next(mm for mm in METHODS if mm["legend"] == p["method"])
    y = ypos[(a, m["key"])]
    # the baseline whisker wears the same secondary ink as its marker: the lighter muted grey failed the
    # validator against the aqua slot (deutan dE 5.1, normal-vision dE 14.5)
    ax.plot([p["lo"], p["hi"]], [y, y], color=m["color"], lw=1.3, solid_capstyle="round", zorder=2, clip_on=False)
    if m["hollow"]:
        ax.plot([p["auroc"]], [y], marker=m["marker"], ms=m["ms"], mfc=SURFACE, mec=m["color"], mew=1.1,
                zorder=4, clip_on=False, ls="none")
    else:
        ax.plot([p["auroc"]], [y], marker=m["marker"], ms=m["ms"], mfc=m["color"], mec=SURFACE, mew=0.8,
                zorder=3, clip_on=False, ls="none")

# row-group labels: author name, and the think coverage on the line below (Table 2 shows it at 3 decimals)
tr = blended_transform_factory(ax.transAxes, ax.transData)
for g, a in enumerate(AUTH):
    yc = -(g * (n_m + GAP) + (n_m - 1) / 2)
    ax.text(-0.015, yc + 0.62, NAME[a], transform=tr, ha="right", va="center", fontsize=FS_LABEL, color=INK)
    ax.text(-0.015, yc - 0.78, f"(think {cover[a]['value']:.3f})", transform=tr, ha="right", va="center",
            fontsize=FS_SMALL, color=INK2)

# legend: one row above the plot, in row order
handles = []
for m in METHODS:
    if m["hollow"]:
        h = Line2D([0], [0], color=m["color"], lw=1.3, marker=m["marker"], ms=m["ms"], mfc=SURFACE, mec=m["color"],
                   mew=1.1)
    else:
        h = Line2D([0], [0], color=m["color"], lw=1.3, marker=m["marker"], ms=m["ms"], mfc=m["color"],
                   mec=SURFACE, mew=0.8)
    handles.append(h)
leg = fig.legend(handles, [m["legend"] for m in METHODS], loc="upper center", bbox_to_anchor=(0.5, 1.0),
                 ncol=n_m, frameon=False, fontsize=FS_TICK, handlelength=2.2, handletextpad=0.5, columnspacing=1.5,
                 borderaxespad=0.15)

# layout check: every text at least 8 pt, inside the figure, and no two texts overlap; the legend clears the data
fig.canvas.draw()
rend = fig.canvas.get_renderer()
fbox = fig.bbox
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
problems = []
for t in texts:
    if t.get_fontsize() < 8:
        problems.append(f"text '{t.get_text()}' is {t.get_fontsize()} pt")
    b = t.get_window_extent(rend)
    if b.x0 < fbox.x0 - 0.5 or b.x1 > fbox.x1 + 0.5 or b.y0 < fbox.y0 - 0.5 or b.y1 > fbox.y1 + 0.5:
        problems.append(f"text '{t.get_text()}' leaves the figure: {b}")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        if texts[i].get_window_extent(rend).overlaps(texts[j].get_window_extent(rend)):
            problems.append(f"texts overlap: '{texts[i].get_text()}' and '{texts[j].get_text()}'")
lb = leg.get_window_extent(rend)
data_top = ax.transData.transform((0, max(ypos.values())))[1]
if lb.y0 < data_top + 4:
    problems.append(f"legend bottom {lb.y0:.1f} px reaches the top data row at {data_top:.1f} px")
if problems:
    sys.exit("FAIL: layout\n" + "\n".join(problems))
print(f"layout check: {len(texts)} texts, all >= 8 pt, inside the figure, no overlaps; legend clears the data; "
      f"figure {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in")

os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/fig1_transfer.pdf")
fig.savefig(f"{OUT}/fig1_transfer.png", dpi=200)
with open(f"{OUT}/fig1_transfer_values.json", "w") as f:
    json.dump(points, f, indent=1)
    f.write("\n")
print(f"wrote {OUT}/fig1_transfer.pdf, fig1_transfer.png, fig1_transfer_values.json ({len(points)} points)")
