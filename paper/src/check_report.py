#!/usr/bin/env python3
"""Checks on the built report. Exit status 1 when any check fails.

  check_report.py paper/build/main.tex [--md paper/build/main.md] [--supplement]

1. overfull boxes: 0 Overfull \\hbox lines in report.log
2. em dashes: 0 U+2014 in the PDF text and 0 "---" in the tex body
3. paths: 0 file paths outside URLs in the main text and in the appendices (a token with a slash
   and a letter on both sides, or a file extension such as .py .csv .md .json .safetensors); the
   task file names that the framework's ladder tables quote verbatim (Appendix K) are counted apart
4. dollar figures: 0 anywhere in the PDF (main text and appendices counted apart)
5. citations: every \\citep key is in the .bib, every .bib entry is cited, no undefined citation
6. main text: the References heading starts on page 8 or earlier, or at the head of page 9
"""
import re
import subprocess
import sys
from pathlib import Path

PATH_RE = re.compile(r"(?<![\w@/])([A-Za-z_][\w.-]*[A-Za-z_]/[\w.-]*[A-Za-z][\w./-]*)")
EXT_RE = re.compile(r"\b[\w-]+\.(py|csv|md|json|jsonl|safetensors|txt|tsv|gz|sh|tex|eval)\b")
URL_RE = re.compile(r"https?://\S+|\b(?:github\.com|huggingface\.co)/\S+")
DOLLAR_RE = re.compile(r"\$\s?\d")
ALLOWED_SLASH = {"and/or"}


def pdf_pages(pdf: Path) -> list[str]:
    out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    return out.split("\f")


def main(argv: list[str]) -> int:
    tex = Path(argv[1])
    md = Path(argv[argv.index("--md") + 1]) if "--md" in argv else tex.with_suffix(".md")
    supplement = "--supplement" in argv   # the supplementary tables: no page limit, and the bib need not be cited in full
    log = tex.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    tex_text = tex.read_text(encoding="utf-8")
    pages = pdf_pages(tex.with_suffix(".pdf"))
    fails = []

    over = re.findall(r"^Overfull \\[hv]box.*$", log, re.M)
    print(f"check: overfull boxes {len(over)}")
    if over:
        fails.append("overfull boxes")

    body = tex_text.split("\\begin{document}", 1)[1]
    n_em = sum(p.count("\u2014") for p in pages) + body.count("---")
    print(f"check: em dashes {n_em}")
    if n_em:
        fails.append("em dashes")

    md_full = md.read_text(encoding="utf-8")
    if supplement:   # all of the supplement is checked like the appendices
        md_full = "\n## References\n\n## Appendix\n" + md_full.split("\n## References", 1)[0]
    main_md = md_full.split("\n## References", 1)[0]
    app_md = md_full.split("\n## Appendix", 1)[1] if "\n## Appendix" in md_full else ""
    # sections whose heading names the ladders quote the framework's evidence text verbatim (task file names)
    sections = re.split(r"\n(?=### )", app_md)
    ladder_md = "\n".join(sec for sec in sections if re.match(r"### [^\n]*ladders", sec))
    other_app = "\n".join(sec for sec in sections if not re.match(r"### [^\n]*ladders", sec))

    def find_paths(text):
        hits = []
        for line in text.splitlines():
            if re.match(r"^!\[[^\]]*\]\([^)]*\)\s*$", line.strip()):
                continue          # an image's source file is not printed
            line = URL_RE.sub(" ", line)
            for m in PATH_RE.finditer(line):
                if m.group(1) not in ALLOWED_SLASH:
                    hits.append(m.group(1))
            hits += [m.group(0) for m in EXT_RE.finditer(line)]
        return hits
    p_main, p_app, p_lad = find_paths(main_md), find_paths(other_app), find_paths(ladder_md)
    print(f"check: paths in the main text {len(p_main)}" + (": " + ", ".join(sorted(set(p_main))) if p_main else ""))
    print(f"check: paths in the appendices outside the ladder tables {len(p_app)}" + (": " + ", ".join(sorted(set(p_app))) if p_app else ""))
    print(f"check: task file names quoted verbatim in the ladder tables (the ladder sections, the framework's evidence text) {len(p_lad)}: " + ", ".join(sorted(set(p_lad))))
    if p_main or p_app:
        fails.append("paths")

    ref_page = next((i for i, p in enumerate(pages) if re.search(r"^\s*References\s*$", p, re.M)), None)
    if ref_page is None:
        print("check: References heading not found in the PDF text")
        if not supplement:
            fails.append("references heading")
        main_text, rest = "".join(pages), ""
    else:
        head = re.split(r"^\s*References\s*$", pages[ref_page], maxsplit=1, flags=re.M)
        main_text = "".join(pages[:ref_page]) + head[0]
        rest = head[1] + "".join(pages[ref_page + 1:])
    n_main = len(DOLLAR_RE.findall(main_text))
    print(f"check: dollar figures in the main text {n_main}; after it {len(DOLLAR_RE.findall(rest))}")
    if n_main or DOLLAR_RE.search(rest):
        fails.append("dollar figures")      # no spend figure anywhere in the report

    bib = Path(tex.parent, "../src/references.bib").resolve().read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    cited = set(re.findall(r"\\citep\{([^}]*)\}", body))
    cited = {k.strip() for c in cited for k in c.split(",")}
    undefined = re.findall(r"Citation `([^']+)' .*undefined", log)
    missing, uncited = sorted(cited - keys), sorted(keys - cited)
    print(f"check: citations {len(cited)} keys cited, {len(keys)} in the bib; missing {missing}; uncited {uncited}; undefined in the log {sorted(set(undefined))}")
    if missing or (uncited and not supplement) or undefined or "There were undefined references" in log:
        fails.append("citations")

    if ref_page is not None and not supplement:
        page_no = ref_page + 1
        before = re.split(r"^\s*References\s*$", pages[ref_page], maxsplit=1, flags=re.M)[0]
        lead = len(re.sub(r"\s+", "", before))
        print(f"check: References heading on page {page_no} ({lead} characters of main text before it on that page)")
        ok = page_no <= 8 or (page_no == 9 and lead < 80)
        if not ok:
            fails.append("main text longer than 8 pages")
    print(f"check: pages in the PDF {len([p for p in pages if p.strip()])}")
    if fails:
        print("check: FAILED: " + "; ".join(fails))
        return 1
    print("check: all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
