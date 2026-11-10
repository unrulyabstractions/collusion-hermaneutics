#!/usr/bin/env bash
# Build paper/build/main.pdf (ICML 2026 preprint, two columns) from paper/build/main.md, and
# paper/build/supplement.pdf ("Supplementary tables") from paper/build/supplement.md.
# Generate the markdown first:  .venv/bin/python paper/src/gen_report.py
#
#   paper/src/build.sh            # builds both documents
#
# Steps per document: md2tex.py convert -> pdflatex, bibtex (when it cites), pdflatex x2
# (nonstopmode, halt-on-error) -> page summary -> check_report.py (overfull boxes, em dashes,
# paths, dollar figures, citations, main-text pages). The ICML style files sit in paper/src
# (the official ICML 2026 author kit, unchanged). Exit status is non-zero when any step or check fails.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER="$(cd "$SRC/.." && pwd)"
ROOT="$(cd "$PAPER/.." && pwd)"
BUILD="$PAPER/build"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
TEXBIN="${TEXBIN:-/Library/TeX/texbin}"
PDFLATEX="$TEXBIN/pdflatex"; [ -x "$PDFLATEX" ] || PDFLATEX="pdflatex"
BIBTEX="$TEXBIN/bibtex"; [ -x "$BIBTEX" ] || BIBTEX="bibtex"
export TEXINPUTS="$SRC//:"
export BSTINPUTS="$SRC//:"
mkdir -p "$BUILD"

# build_one <stem>: paper/build/<stem>.md -> <stem>.tex -> <stem>.pdf
build_one() {
  local stem="$1"
  [ -f "$BUILD/$stem.md" ] || { echo "build: $BUILD/$stem.md not found; run gen_report.py first" >&2; exit 1; }
  rm -f "$BUILD/$stem".{aux,bbl,blg,log,out,toc,pdf}
  "$PY" "$SRC/md2tex.py" convert "$BUILD/$stem.md" "$BUILD/$stem.tex"
  ( cd "$BUILD"
    run_latex() {
      if ! "$PDFLATEX" -interaction=nonstopmode -halt-on-error -file-line-error "$stem.tex" > "$stem.pass$1.out" 2>&1; then
        echo "build: pdflatex failed on $stem, pass $1; last lines:" >&2
        tail -n 40 "$stem.pass$1.out" >&2
        exit 1
      fi
    }
    run_latex 1
    if grep -q '\\citation' "$stem.aux"; then
      if ! "$BIBTEX" "$stem" > "$stem.bibtex.out" 2>&1; then
        echo "build: bibtex failed on $stem:" >&2; cat "$stem.bibtex.out" >&2; exit 1
      fi
    fi
    run_latex 2
    run_latex 3
    [ -s "$stem.pdf" ] || { echo "build: $stem.pdf missing or empty" >&2; exit 1; }
  )
}

build_one main
"$PY" "$SRC/md2tex.py" pages "$BUILD/main.tex"
"$PY" "$SRC/check_report.py" "$BUILD/main.tex"
echo "build: wrote $BUILD/main.pdf"
build_one supplement
"$PY" "$SRC/md2tex.py" pages "$BUILD/supplement.tex" | head -n 1
"$PY" "$SRC/check_report.py" "$BUILD/supplement.tex" --supplement
echo "build: wrote $BUILD/supplement.pdf"
