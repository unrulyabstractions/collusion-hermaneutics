"""Tables and figures from the probe matrix (hermeneutics/results/matrix_probes at d38598d), read through pinned.py.

build(led, ctx) returns {placeholder: markdown}; it also copies the two transfer heatmaps, byte for byte from the
pinned commit, into report/figures.
"""
import os
import re

import pinned as P

MR = "hermeneutics/results/matrix_probes/RESULTS.md"
FIG = os.path.join(P.REPO, "report", "figures")
LAB = {"cheating": "cheating", "coordination posted": "coordination", "collusion crossed": "collusion"}
SRC = {"controls": "controls", "board_seed": "board-seed teams", "solo_design": "design runs", "spar_team": "SPAR screen teams",
       "spar_mixed": "SPAR mixed teams", "spar_expl": "SPAR exploration", "spar_solo": "SPAR screen solo", "goodfire": "Goodfire",
       "cheatbench": "CheatBench", "evilgenie": "EvilGenie"}
ARCH = [("linear", "linear, pooled"), ("mlp_mean", "MLP per step, running mean"), ("mlp_max", "MLP per step, running max"),
        ("attn_pool", "attention pooling"), ("attn_monitor", "attention monitor"), ("gru", "GRU"),
        ("linear_step", "linear per step, running mean"), ("mlp_traj", "MLP on the pooled vector"), ("stack", "stack with TF-IDF")]
ANAME = dict(ARCH)
TRAINED = [a for a, _ in ARCH if a not in ("linear", "stack")]
SHORT = {"mlp_mean": "MLP, mean", "mlp_max": "MLP, max", "attn_pool": "attention pooling", "attn_monitor": "attention monitor",
         "gru": "GRU", "linear_step": "linear per step", "mlp_traj": "pooled MLP"}   # Table 4 cells; Appendix L uses the full names   # the "best architecture" columns choose among these


def scheme_name(s):
    if s.startswith("alone: "):
        return SRC[s[7:]] + " alone"
    if s.startswith("leave out: "):
        return "all but " + SRC[s[11:]]
    return {"ours": "our sets", "external": "external sets", "all": "all sets", "ours to external": "our sets to external",
            "external to ours": "external to our sets"}[s]


def v(cell):
    return cell.split(" [")[0].split(" (")[0].strip()


def build(led, ctx):
    out = {}
    where = lambda t: P.where(t, None)
    # ---------------------------------------------------------------- the linear matrix (three labels)
    mat = {}
    for label in ("cheating", "coordination posted", "collusion crossed"):
        t = P.table(P.MX, MR, "The matrix (main feature: think pair, all layers)", first_col="scheme", last=label)
        mat[label] = {r[0]: dict(zip(t["header"], r)) for r in t["rows"]}
        ctx.setdefault("mx_where", where(t))
    ctx["mx"] = mat

    # architecture summaries (B.I and B.II), transposed to scheme -> arch -> value
    def summary(head):
        t = P.table(P.MX, MR, head, first_col="architecture")
        hdr = t["header"]
        d = {}
        for r in t["rows"]:
            for k in range(1, len(hdr)):
                if k < len(r) and r[k].strip():
                    d.setdefault(hdr[k], {})[r[0]] = r[k].strip()
        return d, t
    bi, tbi = summary("Summary across schemes: B.I trajectory AUROC (mean over seeds)")
    bii, tbii = summary("Summary across schemes: B.II trajectory AUROC (mean over seeds, mean over held-out authors)")
    key = lambda label, scheme: {"cheating": "cheat", "coordination posted": "coord", "collusion crossed": "collu"}[label] + ": " + scheme

    # B.II TF-IDF baseline per scheme, from the per-scheme architecture tables ("### <label>: <scheme>")
    def tfidf_bii(label, scheme):
        hits = [t for t in P.md_tables(P.MX, MR) if t["heads"][-1] == f"{label}: {scheme}" and t["header"][0] == "architecture"]
        for t in hits:    # the "###" architecture tables (B.II in column 6), then the "####" per-scheme tables (A, B.I, B.II)
            r = [x for x in t["rows"] if x[0] == "baseline: TF-IDF rendered text"]
            if not r:
                continue
            col = t["header"].index("B.II mean trajectory AUROC (seed range)") if "B.II mean trajectory AUROC (seed range)" in t["header"] else (
                t["header"].index("B.II (mean over held-out authors)") if "B.II (mean over held-out authors)" in t["header"] else None)
            if col is not None and len(r[0]) > col and r[0][col].strip() and r[0][col].strip()[0].isdigit():
                return r[0][col].strip()
        return None

    def best(d, k):
        cand = [(float(d[k][a]), a) for a in TRAINED if a in d.get(k, {})]
        return max(cand) if cand else None

    def best_cell(d, k):
        """The best trained architecture's AUROC and every architecture tied with it at three decimals."""
        b = best(d, k)
        if not b:
            return "not run"
        tied = [SHORT[a] for a in TRAINED if a in d[k] and f"{float(d[k][a]):.3f}" == f"{b[0]:.3f}"]
        return f"{b[0]:.3f} ({'; '.join(tied)})"

    # ---------------------------------------------------------------- main summary table
    ROWS = [("cheating", "ours"), ("cheating", "all"), ("coordination posted", "ours"), ("collusion crossed", "ours"),
            ("cheating", "leave out: controls"), ("cheating", "leave out: cheatbench"), ("cheating", "leave out: evilgenie"),
            ("coordination posted", "leave out: solo_design"), ("coordination posted", "leave out: spar_team"),
            ("coordination posted", "leave out: spar_mixed"), ("collusion crossed", "leave out: spar_team"),
            ("cheating", "ours to external"), ("cheating", "external to ours")]
    T = ["| label and scheme | positive / agents | linear probe | null 95th | TF-IDF | best architecture | B.II: linear probe | best architecture | TF-IDF |",
         "|---|---|---|---|---|---|---|---|---|"]
    for label, scheme in ROWS:
        m = mat[label][scheme]
        n, pos = m["B.I n/pos/runs/auth"].split("/")[:2]
        k = key(label, scheme)
        b1 = best(bi, k); b2 = best(bii, k)
        tb2 = tfidf_bii(label, scheme)
        cells = [f"{LAB[label]}, {scheme_name(scheme)}", f"{pos} / {n}", v(m["B.I"]), m["B.I null 95th"], m["B.I TF-IDF"],
                 best_cell(bi, k), v(m["B.II mean"]) if m["B.II mean"] != "n/a" else "n/a",
                 best_cell(bii, k), tb2 or "n/a"]
        line = "| " + " | ".join(cells) + " |"
        led(line, f"Table 4 row {label}/{scheme}", f"{MR}@{P.MX}", "matrix table (B.I, null 95th, TF-IDF, B.II mean); architecture summaries (max over architectures 2 to 8); per-scheme baseline TF-IDF B.II", raw_where=True)
        T.append(line)
    out["<<T_MATRIX_MAIN>>"] = "\n".join(T)

    # claims of Results ("Which probe type and which architecture hold up across sources?"), asserted here
    f = lambda label, scheme, col: float(v(mat[label][scheme][col]))
    inside = [(l, s) for l in mat for s in mat[l] if (s.startswith("alone") or s in ("ours", "external", "all")) and mat[l][s]["B.I"] != "n/a"]
    probe_wins_inside = sorted((l, s) for l, s in inside if f(l, s, "B.I") > f(l, s, "B.I TF-IDF"))
    assert probe_wins_inside == [("collusion crossed", "alone: spar_mixed"), ("coordination posted", "all"), ("coordination posted", "alone: spar_team"),
                                 ("coordination posted", "ours")], probe_wins_inside
    loso = [(l, s) for l in mat for s in mat[l] if s.startswith("leave out") and mat[l][s]["B.I"] != "n/a"]
    probe_wins_loso = sorted((l, s) for l, s in loso if f(l, s, "B.I") > f(l, s, "B.I TF-IDF"))
    assert probe_wins_loso == [("cheating", "leave out: cheatbench"), ("cheating", "leave out: evilgenie"), ("collusion crossed", "leave out: spar_team"),
                               ("coordination posted", "leave out: board_seed"), ("coordination posted", "leave out: solo_design"),
                               ("coordination posted", "leave out: spar_mixed"), ("coordination posted", "leave out: spar_team")], probe_wins_loso
    for l, s in [("cheating", "ours to external"), ("cheating", "external to ours")]:
        assert f(l, s, "B.I") < f(l, s, "B.I TF-IDF"), (l, s)
    for l, s in [("cheating", "ours"), ("cheating", "all")]:   # unseen authors cost the linear probe about a tenth
        assert 0.07 <= f(l, s, "B.I") - f(l, s, "B.II mean") <= 0.15, (l, s)
    # mlp_max: highest B.II among the trained architectures on cheating ours and coordination ours
    for k in ("cheat: ours", "coord: ours"):
        assert best(bii, k)[1] == "mlp_max", (k, best(bii, k))
    assert float(bii["collu: ours"]["attn_monitor"]) < 0.5, bii["collu: ours"]   # the monitor is weak on collusion B.II
    # ...but not uniformly: on unseen authors the linear probe matches or beats every trained architecture for collusion on our
    # sets, and beats MLP running max for cheating on all sets; with CheatBench held out every trained architecture trails it
    assert all(float(bii["collu: ours"][a]) <= float(bii["collu: ours"]["linear"]) for a in TRAINED), bii["collu: ours"]
    assert float(bii["cheat: all"]["mlp_max"]) < float(bii["cheat: all"]["linear"]), bii["cheat: all"]
    assert all(float(bi["cheat: leave out: cheatbench"][a]) < float(bi["cheat: leave out: cheatbench"]["linear"]) for a in TRAINED if a in bi["cheat: leave out: cheatbench"])
    # on held-out sources the best trained architecture beats the linear probe for coordination by a wide margin
    for sch in ("leave out: solo_design", "leave out: spar_team", "leave out: spar_mixed"):
        assert best(bi, "coord: " + sch)[0] - float(bi["coord: " + sch]["linear"]) >= 0.05, sch
    # the stack (which includes TF-IDF) sits within 0.035 of TF-IDF where both ran
    for k, lab, sch in (("cheat: ours", "cheating", "ours"), ("cheat: all", "cheating", "all"), ("coord: ours", "coordination posted", "ours"), ("collu: ours", "collusion crossed", "ours")):
        assert abs(float(bi[k]["stack"]) - float(mat[lab][sch]["B.I TF-IDF"])) <= 0.035, (k, bi[k]["stack"])
    led("asserted: probe beats TF-IDF inside a distribution only on 4 coordination/collusion cells; on unseen sources for every coordination cell, cheating without CheatBench or EvilGenie, collusion without the SPAR screen teams; never on ours<->external; B.I minus B.II 0.07-0.15 on cheating ours and all; MLP running max best B.II on cheating and coordination ours; attention monitor below 0.5 on collusion ours B.II",
        "Results, probe types and architectures; Discussion", f"{MR}@{P.MX}", "tables_matrix.py assertions", raw_where=True)

    # cells at or below their null (B.I)
    low = [(l, s) for l in mat for s in mat[l] if mat[l][s]["B.I"] != "n/a" and mat[l][s]["B.I null 95th"] and f(l, s, "B.I") <= float(mat[l][s]["B.I null 95th"])]
    out["<<MX_AT_NULL>>"] = "; ".join(f"{LAB[l]}, {scheme_name(s)}" for l, s in low)
    led(out["<<MX_AT_NULL>>"], "Appendix M: matrix cells at or below their null", f"{MR}@{P.MX}", "B.I <= B.I null 95th", raw_where=True)

    # ---------------------------------------------------------------- appendix: the full linear matrix per label
    for label, ph in (("cheating", "<<T_MX_CHEAT>>"), ("coordination posted", "<<T_MX_COORD>>"), ("collusion crossed", "<<T_MX_COLL>>")):
        L = ["| scheme | A: mean over authors | B.I [95% CI] | B.I agents / positive / runs / authors | null 95th | TF-IDF | metadata | B.II: mean over authors | B.II agents / positive / runs / authors |",
             "|---|---|---|---|---|---|---|---|---|"]
        for scheme, m in mat[label].items():
            cnt = lambda c: m[c].replace("/", " / ") if m[c] else ""
            line = (f"| {scheme_name(scheme)} | {v(m['A mean']) if m['A mean'] != 'n/a' else 'n/a'} | {m['B.I']} | {cnt('B.I n/pos/runs/auth')} | {m['B.I null 95th'] or 'n/a'} | "
                    f"{m['B.I TF-IDF'] or 'n/a'} | {m['B.I metadata'] or 'n/a'} | {v(m['B.II mean']) if m['B.II mean'] != 'n/a' else 'n/a'} | {cnt('B.II n/pos/runs/auth')} |")
            led(line, f"Appendix M matrix {label}/{scheme}", ctx["mx_where"], "row as printed", raw_where=True)
            L.append(line)
        out[ph] = "\n".join(L)

    # appendix: architectures by scheme (B.I and B.II)
    for d, ph, t, cap in ((bi, "<<T_MX_ARCH_BI>>", tbi, "B.I"), (bii, "<<T_MX_ARCH_BII>>", tbii, "B.II")):
        archs = [a for a, _ in ARCH if any(a in d[k] for k in d)]
        L = ["| label and scheme | " + " | ".join(ANAME[a] for a in archs) + " |", "|" + "---|" * (len(archs) + 1)]
        for k in d:
            lab, sch = k.split(": ", 1)
            lab = {"cheat": "cheating", "coord": "coordination", "collu": "collusion"}[lab]
            line = f"| {lab}, {scheme_name(sch)} | " + " | ".join(d[k].get(a, "") or "" for a in archs) + " |"
            led(line, f"Appendix M architectures {cap} {k}", where(t), "summary row transposed", raw_where=True)
            L.append(line)
        out[ph] = "\n".join(L)

    # appendix: per step
    t = P.table(P.MX, MR, "Per step (layer 38, every turn)", first_col="label", nth=0)  # the per-source table follows it
    S = ["| label and scheme | features | turn AUROC | turn null 95th | agent AUROC | detected / reaching | delay, median (IQR) | turns / positive |", "|---|---|---|---|---|---|---|---|"]
    FEAT = {"think pair, layer 38 (per turn)": "linear probe, think pair", "turn index": "turn index", "turn index + fraction": "turn index and fraction",
            "metadata + turn index + fraction": "metadata, turn index and fraction"}
    steps = {}
    for r in t["rows"]:
        d = dict(zip(t["header"], r))
        sch = "our sets" if d["scheme"].startswith("ours") else "our sets and Goodfire"
        line = (f"| {LAB[d['label']]}, {sch} | {FEAT[d['features']]} | {d['turn AUROC']} | {d['turn null 95th'] or 'n/a'} | {v(d['agent AUROC'])} | "
                f"{d['detected / reaching'].replace('/', ' / ')} | {d['delay median (IQR)']} | {d['turns / pos'].replace('/', ' / ')} |")
        led(line, f"Appendix M per step {d['label']}/{d['scheme']}/{d['features']}", P.where(t, r), "row as printed", raw_where=True)
        S.append(line)
        steps[(d["label"], sch, d["features"])] = d
    out["<<T_MX_STEPS>>"] = "\n".join(S)
    for lab in ("cheating", "coordination posted", "collusion crossed"):     # beats its null; and metadata with turn index
        pr = steps[(lab, "our sets", "think pair, layer 38 (per turn)")]; mt = steps[(lab, "our sets", "metadata + turn index + fraction")]
        assert float(v(pr["turn AUROC"])) > float(pr["turn null 95th"]), lab
        assert float(v(pr["turn AUROC"])) > float(v(mt["turn AUROC"])), lab
    for lab in ("cheating", "collusion crossed"):
        pr = steps[(lab, "our sets", "think pair, layer 38 (per turn)")]; mt = steps[(lab, "our sets", "metadata + turn index + fraction")]
        assert float(v(pr["turn AUROC"])) - float(v(mt["turn AUROC"])) < 0.02, lab   # "barely"

    # heatmap: off-diagonal mean (prose has no number; the caption gives it) and the two figures
    t = P.table(P.MX, MR, "Source-to-source transfer", first_col="train \\ test", nth=0)
    vals = [float(c) for r in t["rows"] for k, c in enumerate(r[1:]) if t["header"][k + 1] != r[0]]
    out["<<MX_OFFDIAG>>"] = f"{sum(vals) / len(vals):.2f}"
    led(out["<<MX_OFFDIAG>>"], "Figure M1 caption: mean off-diagonal cheating transfer AUROC", P.where(t, None), f"mean over {len(vals)} off-diagonal cells", raw_where=True)
    os.makedirs(FIG, exist_ok=True)
    for name in ("heatmap_cheating.png", "heatmap_board.png"):
        data = P.show(P.MX, f"hermeneutics/results/matrix_probes/{name}", binary=True)
        with open(os.path.join(FIG, "matrix_" + name), "wb") as fh:
            fh.write(data)
    # architecture cells with no result
    txt = P.show(P.MX, MR).split("### Architecture cells with no result", 1)[1].split("\n## ", 1)[0]
    cells = {re.sub(r", held-out [^,]+,", ",", l.split(": no feasible split")[0]) for l in txt.splitlines() if l.startswith("- ")}
    out["<<MX_INFEASIBLE>>"] = str(len(cells))   # one cell per (label, scheme, type, architecture); EvilGenie lists each held-out author
    led(out["<<MX_INFEASIBLE>>"], "Appendix M: architecture cells with no feasible split", f"{MR}@{P.MX}", "bullets under 'Architecture cells with no result'", raw_where=True)
    return out
