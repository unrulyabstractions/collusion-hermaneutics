"""The probe matrix on the old labels against the same cells on the new ladders' labels (hermeneutics/new_ladders.py):
the linear probe (think pair, all layers) and whole-text TF-IDF, B.I pooled and B.II mean over authors, per label and
scheme, and which of the two leads in each.

    python -m hermeneutics.new_ladders_compare        results/new_ladders/matrix_compare.csv and a printed summary
"""
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
KEYS = ["label", "scheme", "ptype", "unit"]


def cells(p):
    c = pd.read_csv(p)
    c = c[c.unit.isin(["pooled", "mean over authors"]) & c.feature.isin(["think", "tfidf", "meta"])]
    w = c.pivot_table(index=KEYS, columns="feature", values="auroc", aggfunc="first")
    n = c[c.feature == "think"].set_index(KEYS)[["n", "pos", "null_95"]]
    return w.join(n)


def main():
    old, new = cells(R / "matrix_probes" / "cells.csv"), cells(R / "matrix_probes_new_ladders" / "cells.csv")
    J = old.reset_index().merge(new.reset_index(), on=KEYS, how="outer", suffixes=("_old", "_new"))
    for s in ("old", "new"):
        J[f"margin_{s}"] = J[f"think_{s}"] - J[f"tfidf_{s}"]
        J[f"lead_{s}"] = J[f"margin_{s}"].map(lambda m: "" if pd.isna(m) else "probe" if m > 0 else "tfidf")
    J["d_think"] = J.think_new - J.think_old
    J["lead_flip"] = (J.lead_old != J.lead_new) & (J.lead_old != "") & (J.lead_new != "")
    J.to_csv(R / "new_ladders" / "matrix_compare.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    cols = ["label", "scheme", "ptype", "pos_old", "pos_new", "think_old", "think_new", "tfidf_old", "tfidf_new",
            "lead_old", "lead_new"]
    for ptype in ("B.I", "B.II"):
        print(f"== {ptype}")
        print(J[J.ptype == ptype][cols].round(3).to_string(index=False))
    f = J[J.lead_flip]
    print(f"\nlead flips: {len(f)}")
    print(f[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
