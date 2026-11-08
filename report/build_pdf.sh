#!/usr/bin/env bash
# Build report/tex/report.pdf from report/REPORT_DRAFT.md.
#
#   report/build_pdf.sh            # uses report/REPORT_DRAFT.md
#   report/build_pdf.sh other.md   # any markdown file
#
# Steps: md2tex.py convert -> xelatex x2 (nonstopmode, halt-on-error) -> page summary.
# Idempotent: every run regenerates report.tex and the PDF from the markdown.
# Exit status is non-zero when any step fails.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
XELATEX="${XELATEX:-/Library/TeX/texbin/xelatex}"
[ -x "$XELATEX" ] || XELATEX="xelatex"
MD="${1:-$HERE/REPORT_DRAFT.md}"
TEXDIR="$HERE/tex"
TEX="$TEXDIR/report.tex"

[ -f "$MD" ] || { echo "build_pdf: markdown not found: $MD" >&2; exit 1; }
mkdir -p "$TEXDIR"

"$PY" "$HERE/gen/md2tex.py" convert "$MD" "$TEX"

cd "$TEXDIR"
for pass in 1 2; do
  if ! "$XELATEX" -interaction=nonstopmode -halt-on-error -file-line-error report.tex > "report.pass$pass.out" 2>&1; then
    echo "build_pdf: xelatex failed on pass $pass; last lines of report.pass$pass.out:" >&2
    tail -n 40 "report.pass$pass.out" >&2
    exit 1
  fi
done
[ -s report.pdf ] || { echo "build_pdf: report.pdf missing or empty" >&2; exit 1; }

"$PY" "$HERE/gen/md2tex.py" pages "$TEX"
echo "build_pdf: wrote $TEXDIR/report.pdf"
