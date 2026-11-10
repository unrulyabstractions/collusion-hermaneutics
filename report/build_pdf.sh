#!/usr/bin/env bash
# Build report/tex/report.pdf (ICML 2026 preprint, two columns) from report/REPORT_DRAFT.md, and
# report/tex/supplement.pdf ("Supplementary tables") from report/SUPPLEMENT_DRAFT.md.
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
SUP_MD="$HERE/SUPPLEMENT_DRAFT.md"
mkdir -p "$TEXDIR"
export TEXINPUTS="$HERE/icml//:"
export BSTINPUTS="$HERE/icml//:"

# build_one <markdown> <stem>: md2tex, pdflatex, BibTeX when the document cites, pdflatex twice
build_one() {
  local md="$1" stem="$2"
  rm -f "$TEXDIR/$stem".{aux,bbl,blg,log,out,pdf}
  "$PY" "$HERE/gen/md2tex.py" convert "$md" "$TEXDIR/$stem.tex"
  ( cd "$TEXDIR"
    run_latex() {
      if ! "$PDFLATEX" -interaction=nonstopmode -halt-on-error -file-line-error "$stem.tex" > "$stem.pass$1.out" 2>&1; then
        echo "build_pdf: pdflatex failed on $stem, pass $1; last lines:" >&2
        tail -n 40 "$stem.pass$1.out" >&2
        exit 1
      fi
    }
    run_latex 1
    if grep -q '\\citation' "$stem.aux"; then
      if ! "$BIBTEX" "$stem" > "$stem.bibtex.out" 2>&1; then
        echo "build_pdf: bibtex failed on $stem:" >&2; cat "$stem.bibtex.out" >&2; exit 1
      fi
    fi
    run_latex 2
    run_latex 3
    [ -s "$stem.pdf" ] || { echo "build_pdf: $stem.pdf missing or empty" >&2; exit 1; }
  )
}

build_one "$MD" report
"$PY" "$HERE/gen/md2tex.py" pages "$TEXDIR/report.tex"
"$PY" "$HERE/gen/check_report.py" "$TEXDIR/report.tex" --md "$MD"
echo "build_pdf: wrote $TEXDIR/report.pdf"
if [ "$MD" = "$HERE/REPORT_DRAFT.md" ] && [ -f "$SUP_MD" ]; then
  build_one "$SUP_MD" supplement
  "$PY" "$HERE/gen/md2tex.py" pages "$TEXDIR/supplement.tex" | head -n 1
  "$PY" "$HERE/gen/check_report.py" "$TEXDIR/supplement.tex" --md "$SUP_MD" --supplement
  echo "build_pdf: wrote $TEXDIR/supplement.pdf"
fi
