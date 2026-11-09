#!/usr/bin/env bash
# Build report/tex/report.pdf (ICML 2026 preprint, two columns) from report/REPORT_DRAFT.md.
#
#   report/build_pdf.sh            # uses report/REPORT_DRAFT.md
#   report/build_pdf.sh other.md   # any markdown file
#
# Steps: md2tex.py convert -> pdflatex, bibtex, pdflatex x2 (nonstopmode, halt-on-error)
# -> page summary -> check_report.py (overfull boxes, em dashes, paths, dollar figures,
# citations, main-text pages). The ICML style files come from report/icml (the official
# ICML 2026 author kit, unchanged). Exit status is non-zero when any step or check fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
TEXBIN="${TEXBIN:-/Library/TeX/texbin}"
PDFLATEX="$TEXBIN/pdflatex"; [ -x "$PDFLATEX" ] || PDFLATEX="pdflatex"
BIBTEX="$TEXBIN/bibtex"; [ -x "$BIBTEX" ] || BIBTEX="bibtex"
MD="${1:-$HERE/REPORT_DRAFT.md}"
TEXDIR="$HERE/tex"
TEX="$TEXDIR/report.tex"

[ -f "$MD" ] || { echo "build_pdf: markdown not found: $MD" >&2; exit 1; }
mkdir -p "$TEXDIR"
rm -f "$TEXDIR"/report.{aux,bbl,blg,log,out,pdf}

"$PY" "$HERE/gen/md2tex.py" convert "$MD" "$TEX"

cd "$TEXDIR"
export TEXINPUTS="$HERE/icml//:"
export BSTINPUTS="$HERE/icml//:"
run_latex() {
  if ! "$PDFLATEX" -interaction=nonstopmode -halt-on-error -file-line-error report.tex > "report.pass$1.out" 2>&1; then
    echo "build_pdf: pdflatex failed on pass $1; last lines of report.pass$1.out:" >&2
    tail -n 40 "report.pass$1.out" >&2
    exit 1
  fi
}
run_latex 1
if ! "$BIBTEX" report > report.bibtex.out 2>&1; then
  echo "build_pdf: bibtex failed:" >&2; cat report.bibtex.out >&2; exit 1
fi
run_latex 2
run_latex 3
[ -s report.pdf ] || { echo "build_pdf: report.pdf missing or empty" >&2; exit 1; }

"$PY" "$HERE/gen/md2tex.py" pages "$TEX"
"$PY" "$HERE/gen/check_report.py" "$TEX"
echo "build_pdf: wrote $TEXDIR/report.pdf"
