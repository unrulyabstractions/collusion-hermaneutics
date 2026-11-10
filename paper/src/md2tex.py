#!/usr/bin/env python3
"""Markdown (GitHub flavour) to ICML 2026 LaTeX (preprint, two columns) for the report, plus a page report.

Usage
  md2tex.py convert main.md main.tex         write main.tex (and nothing else)
  md2tex.py pages main.tex                   read main.log / main.aux after pdflatex
                                             and print the page summary

Standard library only. The converter is lossless for text: it escapes LaTeX
specials, converts **bold**, *italic*, `code`, links, URLs, headings, lists,
pipe tables (with "**Table X.**" caption lines) and images. It never rewords.

Headings: "# " -> the ICML title; the paragraph after it is "Author, Affiliation", and a
"Correspondence: <address>" paragraph sets the corresponding author. "## Abstract" -> the abstract
environment; other "## " -> \\section (numbered; Code and data, Acknowledgements and similar unnumbered),
"### " -> \\subsection. "## References" -> natbib bibliography from ../src/references.bib with icml2026.bst;
"## Appendix" -> \\appendix in one column, where "### Appendix X: Title" -> \\section{Title} lettered X.
Citations: [@key] or [@a; @b] -> \\citep.

Tables: booktabs at one font size (9pt). A table whose cells fit on one line is set at its natural width;
otherwise tabularx fills the column (or both columns, as table*, when it does not fit one column; a
caption ending in {wide} or {narrow} forces the choice). Appendix tables longer than LONGTABLE_ROWS
break across pages. Captions above, unnumbered by LaTeX: the markdown carries the numbers.

Every heading carries a position mark that is written to the .aux at shipout, so `pages` can say what
starts on which page.
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------- settings --
COLUMN_PT = 234.87             # ICML two-column: 3.25in column
TEXT_PT = 487.82               # ICML text width: 6.75in (both columns, and the one-column appendix)
TABCOLSEP_PT = 3.0
LONGTABLE_ROWS = 20            # appendix tables with more body rows than this break across pages
TABLE_SIZES = [("\\small", 9.0)]          # one table font size throughout (icml2026.sty: \\small = 9pt)
FALLBACK_SIZE = ("\\scriptsize", 7.0)     # used only when a table cannot fit at the main size (warned)
SHORT_CELL_EM = 12.0           # a column whose body cells are all shorter than this
                               # is "numeric": prefer a size where its cells wrap
                               # to at most two lines
CELL_TARGET_EM = 45.0          # a column never asks for more than this per line
FIRST_COL_BONUS = 1.3
HEADING_TOP_SKIP_PT = 25.0     # estimated gap from a heading's baseline to the
                               # top of its before-skip (for the page fraction)
UNNUMBERED = {"code and data", "acknowledgements", "llm usage statement", "impact statement"}

CAPTION_RE = re.compile(r"^\*\*(Table|Figure)\b[^*]*\*\*")
WIDTH_TAG_RE = re.compile(r"\s*\{(wide|narrow)\}\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
RULE_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
IMAGE_LINE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)\)\s*$")
REF_ITEM_RE = re.compile(r"^\[\d+\]\s")


# ------------------------------------------------------------------- blocks --
@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Para:
    text: str


@dataclass
class ListItem:
    text: str
    subs: list = field(default_factory=list)   # nested ListBlocks


@dataclass
class ListBlock:
    ordered: bool
    start: int
    items: list[ListItem] = field(default_factory=list)


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]
    caption: str | None = None


@dataclass
class Image:
    alt: str
    path: str
    caption: str | None = None


@dataclass
class Rule:
    pass


# ------------------------------------------------------------------ parsing --
def split_row(line: str) -> list[str]:
    """Split a pipe-table row into cells; pipes inside backticks do not split."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, in_code, i = [], [], False, 0
    while i < len(s):
        c = s[i]
        if c == "`":
            in_code = not in_code
            cur.append(c)
        elif c == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("|")
            i += 1
        elif c == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def is_table_start(lines: list[str], i: int) -> bool:
    return (
        i + 1 < len(lines)
        and lines[i].lstrip().startswith("|")
        and TABLE_SEP_RE.match(lines[i + 1]) is not None
    )


def parse_list(lines: list[str], i: int) -> tuple[ListBlock, int]:
    """Collect list lines from i; return the nested list and the next index."""
    raw: list[tuple[int, bool, str, str]] = []  # indent, ordered, marker, text
    n = len(lines)
    while i < n:
        line = lines[i]
        m = LIST_RE.match(line)
        if m:
            indent = len(m.group(1).expandtabs(4))
            marker = m.group(2)
            ordered = marker[0].isdigit()
            if raw and indent <= raw[0][0] and ordered != raw[0][1]:
                break   # a change of marker type at the top level starts a new list
            raw.append((indent, ordered, marker, m.group(3)))
            i += 1
            continue
        if line.strip() == "":
            # a blank line ends the list unless another item follows
            if i + 1 < n and LIST_RE.match(lines[i + 1]):
                i += 1
                continue
            break
        if HEADING_RE.match(line) or is_table_start(lines, i):
            break
        if line.startswith((" ", "\t")) and raw:
            ind, o, mk, tx = raw[-1]
            raw[-1] = (ind, o, mk, tx + " " + line.strip())
            i += 1
            continue
        if raw and not line.startswith((" ", "\t")):
            # lazy continuation of the last item
            ind, o, mk, tx = raw[-1]
            raw[-1] = (ind, o, mk, tx + " " + line.strip())
            i += 1
            continue
        break

    def build(pos: int, indent: int) -> tuple[ListBlock, int]:
        first = raw[pos]
        start = int(re.match(r"\d+", first[2]).group(0)) if first[1] else 1
        block = ListBlock(ordered=first[1], start=start)
        while pos < len(raw):
            ind, ordered, _marker, text = raw[pos]
            if ind < indent:
                break
            if ind > indent and block.items:
                sub, pos = build(pos, ind)
                block.items[-1].subs.append(sub)
                continue
            if ordered != block.ordered:
                break   # a new list of the other kind at this level
            block.items.append(ListItem(text=text))
            pos += 1
        return block, pos

    block, _ = build(0, raw[0][0]) if raw else (ListBlock(False, 1), 0)
    return block, i


def parse(md: str) -> list:
    lines = md.splitlines()
    blocks: list = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        m = HEADING_RE.match(line)
        if m:
            blocks.append(Heading(len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        if RULE_RE.match(line) and not (i + 1 < n and lines[i + 1].lstrip().startswith("|")):
            blocks.append(Rule())
            i += 1
            continue
        if is_table_start(lines, i):
            header = split_row(lines[i])
            i += 2
            rows = []
            while i < n and lines[i].lstrip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            blocks.append(Table(header, rows))
            continue
        if LIST_RE.match(line):
            block, i = parse_list(lines, i)
            blocks.append(block)
            continue
        m = IMAGE_LINE_RE.match(line.strip())
        if m:
            blocks.append(Image(m.group(1), m.group(2)))
            i += 1
            continue
        # paragraph: run until a blank line or the start of another block
        buf = [line.strip()]
        i += 1
        while i < n:
            nxt = lines[i]
            if nxt.strip() == "" or HEADING_RE.match(nxt) or is_table_start(lines, i):
                break
            if LIST_RE.match(nxt) or IMAGE_LINE_RE.match(nxt.strip()):
                break
            buf.append(nxt.strip())
            i += 1
        blocks.append(Para(" ".join(buf)))
    return attach_captions(blocks)


def attach_captions(blocks: list) -> list:
    """Bind a "**Table X.**" / "**Figure X.**" paragraph to the adjacent table or image."""
    out: list = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if isinstance(b, Para) and CAPTION_RE.match(b.text) and i + 1 < len(blocks) \
                and isinstance(blocks[i + 1], (Table, Image)) and blocks[i + 1].caption is None:
            blocks[i + 1].caption = b.text
            i += 1
            continue
        if isinstance(b, (Table, Image)) and b.caption is None and i + 1 < len(blocks) \
                and isinstance(blocks[i + 1], Para) and CAPTION_RE.match(blocks[i + 1].text):
            b.caption = blocks[i + 1].text
            out.append(b)
            i += 2
            continue
        out.append(b)
        i += 1
    return out


# ------------------------------------------------------------------- inline --
SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
}


def esc(s: str) -> str:
    return "".join(SPECIALS.get(c, c) for c in s)


def text_run(s: str, cell: bool = False) -> str:
    out = esc(s)
    # paired straight double quotes become typographic quotes
    out = re.sub(r'"([^"]*?)"', r"``\1''", out)
    if cell:
        # inside a table cell a slash may take a line break ("a/b"); an interval "v [lo, hi]" breaks only before "["
        out = out.replace("/", "\\slash{}")
        out = INTERVAL_RE.sub(lambda m: m.group(1) + " \\mbox{[" + m.group(2) + ", " + m.group(3) + "]}", out)
    return out


INTERVAL_RE = re.compile(r"(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\]")


def code_run(s: str) -> str:
    parts = []
    for c in s:
        parts.append(SPECIALS.get(c, c))
        if c in "/-_":
            parts.append(r"\allowbreak{}")
    return r"\texttt{" + "".join(parts) + "}"


INLINE_RE = re.compile(
    r"""
      (?P<code>`+)(?P<ctext>.+?)(?P=code)
    | \[(?P<cites>@[A-Za-z0-9_:-]+(?:\s*;\s*@[A-Za-z0-9_:-]+)*)\]
    | !\[(?P<ialt>[^\]]*)\]\((?P<ipath>[^)\s]+)\)
    | \[(?P<ltext>[^\]]+)\]\((?P<lhref>[^)\s]+)\)
    | (?P<url>https?://[^\s<>"'`]+)
    | \*\*(?P<bold>(?:[^*]|\*(?!\*))+?)\*\*
    | (?<![\w*])\*(?P<ital>[^*\s](?:[^*]*?[^*\s])?)\*(?![\w*])
    """,
    re.X,
)


def split_url(url: str) -> tuple[str, str]:
    tail = ""
    while url and url[-1] in ".,;:!?":
        tail = url[-1] + tail
        url = url[:-1]
    while url.endswith(")") and url.count(")") > url.count("("):
        tail = ")" + tail
        url = url[:-1]
    return url, tail


CITED: list[str] = []          # citation keys in order of use (checked against the .bib by check_report.py)


def inline(s: str, tex_dir: Path | None = None, md_dir: Path | None = None, cell: bool = False) -> str:
    out: list[str] = []
    pos = 0
    for m in INLINE_RE.finditer(s):
        out.append(text_run(s[pos:m.start()], cell))
        pos = m.end()
        if m.group("code") is not None:
            out.append(code_run(m.group("ctext")))
        elif m.group("cites") is not None:
            keys = [k.strip().lstrip("@") for k in m.group("cites").split(";")]
            CITED.extend(keys)
            out.append(r"\citep{" + ",".join(keys) + "}")
        elif m.group("ipath") is not None:
            out.append(r"\includegraphics[width=\linewidth]{" + image_path(m.group("ipath"), tex_dir, md_dir) + "}")
        elif m.group("lhref") is not None:
            # escape # so the link survives inside a caption's argument
            out.append(r"\href{" + m.group("lhref").replace("#", r"\#") + "}{" + inline(m.group("ltext"), tex_dir, md_dir, cell) + "}")
        elif m.group("url") is not None:
            url, tail = split_url(m.group("url"))
            out.append(r"\url{" + url + "}" + text_run(tail, cell))
        elif m.group("bold") is not None:
            out.append(r"\textbf{" + inline(m.group("bold"), tex_dir, md_dir, cell) + "}")
        elif m.group("ital") is not None:
            out.append(r"\emph{" + inline(m.group("ital"), tex_dir, md_dir, cell) + "}")
    out.append(text_run(s[pos:], cell))
    return "".join(out)


def image_path(p: str, tex_dir: Path | None, md_dir: Path | None) -> str:
    if tex_dir is None or md_dir is None or re.match(r"^[a-z]+://", p):
        return p
    absp = (md_dir / p).resolve()
    rel = os.path.relpath(absp, tex_dir.resolve())
    return rel.replace("\\", "/")


def guard(s: str) -> str:
    """A leading [ would be read as an optional argument after \\\\ or \\item."""
    return "{}" + s if s.startswith("[") else s


# ------------------------------------------------------------ table widths --
def em_width(tok: str, bold: bool = False, mono: bool = False) -> float:
    if mono:
        return 0.6 * len(tok)
    w = 0.0
    for c in tok:
        if c.isupper():
            w += 0.70
        elif c.isdigit():
            w += 0.50
        elif c.isalpha():
            w += 0.46
        elif c in " \x00":
            w += 0.25
        elif c in ".,;:'`|!":
            w += 0.28
        elif c in "()[]{}/-":
            w += 0.34
        else:
            w += 0.55
    return w * (1.05 if bold else 1.0)


def cell_tokens(cell: str, bold: bool = False) -> list[float]:
    """Widths (em) of the unbreakable pieces of a cell's visible text.

    Text breaks at spaces and after a slash (cells get \\slash); code breaks
    after / - _ (code_run inserts \\allowbreak there).
    """
    widths: list[float] = []

    def text_pieces(s: str) -> None:
        s = INTERVAL_RE.sub(lambda m: m.group(1) + " [" + m.group(2) + ",\x00" + m.group(3) + "]", s)   # "[lo, hi]" is one piece
        for tok in re.sub(r"\*+", "", s).split():
            for piece in re.split(r"(?<=[/-])(?=[A-Za-z])", tok):   # TeX may break after a slash or a hyphen
                if piece:
                    widths.append(em_width(piece, bold))

    pos = 0
    for m in re.finditer(r"`+(.+?)`+", cell):
        text_pieces(cell[pos:m.start()])
        for piece in re.split(r"(?<=[/_-])", m.group(1)):
            if piece:
                widths.append(em_width(piece, mono=True))
        pos = m.end()
    text_pieces(cell[pos:])
    return widths


def two_line_width(toks: list[float]) -> float:
    """The narrowest width (em) at which these pieces wrap to at most two lines."""
    if not toks:
        return 0.0
    if len(toks) == 1:
        return toks[0]
    best = math.inf
    for k in range(1, len(toks)):
        a = sum(toks[:k]) + 0.25 * (k - 1)
        b = sum(toks[k:]) + 0.25 * (len(toks) - k - 1)
        best = min(best, max(a, b))
    return best


def column_plan(header: list[str], rows: list[list[str]], width_pt: float) -> tuple[str, float, list[float], bool]:
    """Column widths (pt) for a table of total width width_pt at the one table font size.

    Returns (size cmd, size pt, widths, natural). natural is True when every cell fits on one
    line inside width_pt: the table is then set at its natural width (plain tabular), never
    stretched. Otherwise the widths fill width_pt (tabularx) and long cells wrap; numeric
    columns get room to wrap to at most two lines first. Only when the unbreakable pieces do
    not fit at the main size is the fallback size used, with a warning.
    """
    ncols = len(header)
    avail = width_pt - ncols * 2 * TABCOLSEP_PT
    longest_tok = [0.0] * ncols
    longest_any = [0.0] * ncols
    longest_body = [0.0] * ncols
    two_line = [0.0] * ncols
    for r, row in enumerate([header] + rows):
        for j in range(ncols):
            cell = row[j] if j < len(row) else ""
            toks = cell_tokens(cell, bold=(r == 0))
            if not toks:
                continue
            total = sum(toks) + 0.25 * (len(toks) - 1)
            longest_tok[j] = max(longest_tok[j], max(toks))
            longest_any[j] = max(longest_any[j], total)
            if r > 0:
                longest_body[j] = max(longest_body[j], total)
                two_line[j] = max(two_line[j], two_line_width(toks))
    short = [0 < longest_body[j] <= SHORT_CELL_EM for j in range(ncols)]
    for cmd, pt in TABLE_SIZES + [FALLBACK_SIZE]:
        natural = [(t + 0.4) * pt for t in longest_any]
        if sum(natural) <= avail:
            return cmd, pt, natural, True
        minw = [(t + 0.3) * pt for t in longest_tok]
        if sum(minw) > avail:
            continue
        if (cmd, pt) == FALLBACK_SIZE:
            print(f"md2tex: warning: table with {ncols} columns needs the fallback size {cmd}", file=sys.stderr)
        return cmd, pt, fill(minw, two_line, longest_any, short, pt, avail), False
    cmd, pt = FALLBACK_SIZE
    minw = [(t + 0.3) * pt for t in longest_tok]
    scale = avail / sum(minw)
    print(f"md2tex: warning: table with {ncols} columns does not fit at {cmd}; columns scaled by {scale:.2f}", file=sys.stderr)
    return cmd, pt, [w * scale for w in minw], False


def fits_column(header: list[str], rows: list[list[str]]) -> bool:
    """A main-text table goes in one column when, at the main size, every numeric column wraps
    to at most two lines and every text column to at most four."""
    cmd, pt, widths, natural = column_plan(header, rows, COLUMN_PT)
    if (cmd, pt) != TABLE_SIZES[0]:
        return False
    if natural:
        return True
    for j in range(len(header)):
        body = [cell_tokens(r[j] if j < len(r) else "") for r in rows]
        need = max((sum(t) + 0.25 * (len(t) - 1) for t in body if t), default=0.0) * pt
        lines = math.ceil(need / max(widths[j] - 2.0, 1.0)) if need else 1
        numeric = max((sum(t) for t in body if t), default=0.0) <= SHORT_CELL_EM
        if lines > (2 if numeric else 4):
            return False
    return True


def fill(minw: list[float], two_line: list[float], longest_any: list[float],
         short: list[bool], pt: float, avail: float) -> list[float]:
    """Spread the spare width: first let numeric columns reach two lines,
    then water-fill toward each column's one-line need, the first column favoured."""
    ncols = len(minw)
    w = list(minw)
    extra = avail - sum(w)
    grants = sorted((max(0.0, two_line[j] * pt + 1.0 - w[j]), j) for j in range(ncols) if short[j])
    for need, j in grants:
        if need <= 0.0:
            continue
        if need > extra:
            break
        w[j] += need
        extra -= need
    target = [max(w[j], min(longest_any[j], CELL_TARGET_EM) * pt + 2.0) for j in range(ncols)]
    for _ in range(20):
        if extra <= 0.01:
            break
        need = [max(0.0, target[j] - w[j]) * (FIRST_COL_BONUS if j == 0 else 1.0) for j in range(ncols)]
        if sum(need) <= 0.01:
            break
        total = sum(need)
        for j in range(ncols):
            give = min(extra * need[j] / total, target[j] - w[j])
            w[j] += max(0.0, give)
        extra = avail - sum(w)
    if extra > 0.01:
        for j in range(ncols):
            w[j] += extra / ncols
    return w


# ----------------------------------------------------------------- emitting --
class Emitter:
    def __init__(self, md_path: Path, tex_path: Path):
        self.md_dir = md_path.parent
        self.tex_dir = tex_path.parent
        self.marks: list[tuple[int, str]] = []   # (level, title) in mark order
        self.tables = 0
        self.in_appendix = False
        self.appendix_letters = 0
        self.sup_sections = 0
        self.supplement = False

    def inl(self, s: str) -> str:
        return inline(s, self.tex_dir, self.md_dir)

    def mark(self, level: int, title: str) -> str:
        self.marks.append((level, title))
        return r"\mdtexmark{%d}" % (len(self.marks) - 1)

    # -- blocks
    def title_page_supplement(self, title: str, meta: list[str], intro: list[str]) -> str:
        """The supplement's first page: title, author, a short note, and the contents (sections, tables, figures)."""
        author, _, affil = (meta[0] if meta else "").partition(",")
        corr = next((m.split(":", 1)[1].strip() for m in meta[1:] if m.lower().startswith("correspondence:")), None)
        out = ["\\onecolumn", "\\raggedbottom",   # tables stay in place, so a short page ends short instead of stretching its gaps
               "\\icmltitle{" + self.inl(title) + "}",
               "\\begin{icmlauthorlist}", "\\icmlauthor{" + self.inl(author.strip()) + "}{aff1}", "\\end{icmlauthorlist}",
               "\\icmlaffiliation{aff1}{" + self.inl(affil.strip()) + "}"]
        if corr:
            out.append("\\icmlcorrespondingauthor{" + self.inl(author.strip()) + "}{" + esc(corr) + "}")
        out += ["\\printAffiliationsAndNotice{}", "\\vspace{1ex}"]
        out += ["\\noindent " + self.inl(t) + "\\par\\medskip" for t in intro]
        out += ["\\let\\addcontentsline\\mdtexaddcontentsline", "\\setcounter{tocdepth}{2}", "{\\small\\tableofcontents}"]
        return "\n".join(out)

    def toc_entry(self, cap: str) -> str:
        """In the supplement, a contents line for a table or figure: its number and its first sentence."""
        if not self.supplement:
            return ""
        m = re.match(r"^\*\*(Table|Figure) (S\d+)\.\*\*\s*(.*)$", cap)
        if not m:
            return ""
        text = m.group(3).split(" Rows:", 1)[0].split(" Rows and columns", 1)[0]
        ends = [x.end() for x in [re.search(r"\?(?=\s|$)", text)] if x] + [x.start() for x in [re.search(r"(?<=[a-z0-9)])\.(?=\s|$)", text)] if x]
        first = text[:min(ends)] if ends else text
        first = first.strip().rstrip(".")
        first = re.sub(r"\*\*|`", "", first)
        if len(first) > 95:
            first = first[:92].rsplit(" ", 1)[0] + " ..."
        return "\\addcontentsline{toc}{subsection}{" + m.group(1) + " " + m.group(2) + ". " + inline(first) + "}"

    def title_block(self, title: str, meta: list[str]) -> str:
        """meta[0] is "Author, Affiliation"; ICML preprint byline with the affiliation footnote."""
        author, _, affil = (meta[0] if meta else "").partition(",")
        author, affil = author.strip(), affil.strip()
        out = ["\\twocolumn[",
               "\\icmltitle{" + self.inl(title) + "}",
               "\\begin{icmlauthorlist}",
               "\\icmlauthor{" + self.inl(author) + "}{aff1}",
               "\\end{icmlauthorlist}",
               "\\icmlaffiliation{aff1}{" + self.inl(affil) + "}"]
        corr = next((m.split(":", 1)[1].strip() for m in meta[1:] if m.lower().startswith("correspondence:")), None)
        if corr:
            out.append("\\icmlcorrespondingauthor{" + self.inl(author) + "}{" + esc(corr) + "}")
        out += [
               "\\icmlkeywords{multi-agent systems, collusion, monitoring, interpretability, probes}",
               "\\vskip 0.3in",
               "]",
               "\\printAffiliationsAndNotice{}"]
        return "\n".join(out)

    def abstract(self, blocks: list) -> str:
        out = ["\\begin{abstract}" + self.mark(1, "Abstract")]
        for b in blocks:
            out.append(self.block(b, noindent=True))
        out.append("\\end{abstract}")
        return "\n".join(out)

    def heading(self, h: Heading) -> str:
        text = h.text.strip()
        low = text.lower()
        if h.level == 2 and low == "references":
            return ("\\FloatBarrier\n" + self.mark(2, "References") +
                    "\n\\bibliographystyle{icml2026}\n\\bibliography{../src/references}")
        if h.level == 2 and low == "supplement":
            self.in_appendix = True
            self.supplement = True
            return "\\FloatBarrier\n\\clearpage" + self.mark(2, "Supplement")
        if self.supplement and h.level == 3:
            # a numbered section (hyperref names it section.N, so the paper can link to it); the running header names it
            self.sup_sections += 1
            head = "Supplementary tables, Section %d: %s" % (self.sup_sections, esc(text))
            return ("\\FloatBarrier\n" + ("\\clearpage\n" if self.sup_sections > 1 else "")
                    + "\\makeatletter\\gdef\\@icmltitlerunning{" + head + "}\\makeatother\n"
                    + "\\section[" + self.inl(text) + "]{" + self.inl(text) + self.mark(2, text) + "}")   # the short title keeps the mark out of the contents
        if h.level == 2 and low == "appendix":
            self.in_appendix = True
            return "\\clearpage\n\\appendix\n\\onecolumn" + self.mark(2, "Appendix")
        if self.in_appendix and h.level == 3:
            m = re.match(r"^Appendix ([A-Z]):\s*(.*)$", text)
            if m:
                expect = chr(ord("A") + self.appendix_letters)
                if m.group(1) != expect:
                    raise SystemExit(f"md2tex: appendix '{text}' is out of order; LaTeX will letter it {expect}")
                self.appendix_letters += 1
                barrier = "\\FloatBarrier\n"
                return barrier + "\\section{" + self.inl(m.group(2)) + self.mark(2, text) + "}"
        if self.in_appendix and h.level == 3:
            # a supplement section (no "Appendix X:" letter): unnumbered, each on a new page after the first
            barrier = "\\FloatBarrier\n" + ("\\clearpage\n" if self.appendix_letters or self.sup_sections else "")
            self.sup_sections += 1
            return barrier + "\\section*{" + self.inl(text) + self.mark(2, text) + "}"
        if self.in_appendix:
            cmd = {2: "\\section", 3: "\\section", 4: "\\subsection"}.get(h.level, "\\subsubsection")
        else:
            cmd = {2: "\\section", 3: "\\subsection"}.get(h.level, "\\subsubsection")
        if low in UNNUMBERED:
            cmd += "*"
        barrier = "\\FloatBarrier\n" if (self.in_appendix and h.level == 2) else ""
        return barrier + cmd + "{" + self.inl(text) + self.mark(h.level, text) + "}"

    def para(self, p: Para, noindent: bool = False) -> str:
        body = self.inl(p.text)
        return ("\\noindent " if noindent else "") + guard(body) + "\n"

    def list_block(self, lb: ListBlock) -> str:
        env = "enumerate" if lb.ordered else "itemize"
        opts = "leftmargin=*, itemsep=1pt, topsep=2pt, parsep=0pt"
        if lb.ordered and lb.start != 1:
            opts += f", start={lb.start}"
        out = [f"\\begin{{{env}}}[{opts}]"]
        for it in lb.items:
            out.append("\\item\\relax " + self.inl(it.text))
            for sub in it.subs:
                out.append(self.list_block(sub))
        out.append(f"\\end{{{env}}}")
        return "\n".join(out)

    def image(self, im: Image) -> str:
        cap = im.caption or ""
        tag = WIDTH_TAG_RE.search(cap)
        cap = WIDTH_TAG_RE.sub("", cap)
        wide = (tag is not None and tag.group(1) == "wide") and not self.in_appendix
        env = "figure*" if wide else "figure"
        width = "0.72\\linewidth" if self.in_appendix else "\\linewidth"
        place = "H" if self.supplement else ("!htbp" if self.in_appendix else "!t")   # the supplement keeps its order: no floating
        out = ["\\begin{" + env + "}[" + place + "]", "\\centering",
               "\\includegraphics[width=" + width + "]{" + image_path(im.path, self.tex_dir, self.md_dir) + "}"]
        if cap:
            out.append("\\caption*{" + self.inl(cap) + "}" + self.toc_entry(cap))
        out.append("\\end{" + env + "}")
        return "\n".join(out)

    def table(self, t: Table) -> str:
        self.tables += 1
        ncols = len(t.header)
        rows = []
        for r in t.rows:
            if len(r) > ncols:
                print(f"md2tex: warning: table row has {len(r)} cells, header has {ncols}; "
                      f"extra cells joined into the last column", file=sys.stderr)
                r = r[:ncols - 1] + [" ".join(r[ncols - 1:])]
            rows.append(r + [""] * (ncols - len(r)))
        cap = t.caption or ""
        tag = WIDTH_TAG_RE.search(cap)
        cap = WIDTH_TAG_RE.sub("", cap)
        if self.in_appendix:
            width, env = TEXT_PT, "table"
        elif tag is not None:
            width, env = (TEXT_PT, "table*") if tag.group(1) == "wide" else (COLUMN_PT, "table")
        else:
            width, env = (COLUMN_PT, "table") if fits_column(t.header, rows) else (TEXT_PT, "table*")
        size_cmd, size_pt, widths, natural = column_plan(t.header, rows, width)
        avail = width - ncols * 2 * TABCOLSEP_PT
        cell = lambda c: inline(c, self.tex_dir, self.md_dir, cell=True)
        cells = lambda row: " & ".join(guard(cell(c)) for c in row) + " \\\\"
        head = " & ".join(guard("\\textbf{" + cell(c) + "}") if c else "" for c in t.header) + " \\\\"
        out: list[str] = []
        long = self.in_appendix and len(rows) > LONGTABLE_ROWS
        if natural:
            spec = "l" * ncols
        elif long:
            spec = "".join(">{\\RaggedRight\\arraybackslash}p{%.2fpt}" % (w - 0.01) for w in widths)
        else:
            spec = "".join(">{\\hsize=%.4f\\hsize\\RaggedRight\\arraybackslash}X" % (w / (avail / ncols)) for w in widths)
        if long:
            out += ["\\begingroup", size_cmd, "\\setlength{\\tabcolsep}{%gpt}" % TABCOLSEP_PT,
                    "\\begin{longtable}{" + spec + "}"]
            if cap:
                out.append("\\caption*{" + self.inl(cap) + "}" + self.toc_entry(cap) + "\\\\")
            out += ["\\toprule", head, "\\midrule", "\\endfirsthead",
                    "\\toprule", head, "\\midrule", "\\endhead",
                    "\\bottomrule", "\\endlastfoot"]
            out += [cells(r) for r in rows]
            out += ["\\end{longtable}", "\\endgroup"]
            return "\n".join(out)
        place = "H" if self.supplement else ("!htbp" if self.in_appendix else "!t")   # the supplement keeps its order: no floating
        out += ["\\begin{" + env + "}[" + place + "]", "\\centering", size_cmd,
                "\\setlength{\\tabcolsep}{%gpt}" % TABCOLSEP_PT]
        if cap:
            out.append("\\caption*{" + self.inl(cap) + "}" + self.toc_entry(cap))
        if natural:
            out.append("\\begin{tabular}{" + spec + "}")
        else:
            out.append("\\begin{tabularx}{%.2fpt}{" % width + spec + "}")
        out += ["\\toprule", head, "\\midrule"]
        out += [cells(r) for r in rows]
        out += ["\\bottomrule", "\\end{tabular}" if natural else "\\end{tabularx}", "\\end{" + env + "}"]
        return "\n".join(out)

    def block(self, b, noindent: bool = False) -> str:
        if isinstance(b, Heading):
            return self.heading(b)
        if isinstance(b, Para):
            return self.para(b, noindent)
        if isinstance(b, ListBlock):
            return self.list_block(b)
        if isinstance(b, Table):
            return self.table(b)
        if isinstance(b, Image):
            return self.image(b)
        if isinstance(b, Rule):
            return "\\bigskip"
        raise TypeError(type(b))


PREAMBLE = r"""\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{placeins}
\usepackage{float}
\usepackage{enumitem}
\usepackage{ragged2e}
\PassOptionsToPackage{hyphens}{url}
\usepackage{hyperref}
\makeatletter\let\mdtexaddcontentsline\addcontentsline\makeatother % icml2026.sty empties it; the supplement restores it for its contents
\usepackage[preprint]{icml2026}
\definecolor{darkblue}{RGB}{0,0,139}
\hypersetup{colorlinks=true,linkcolor=darkblue,urlcolor=darkblue,citecolor=darkblue,breaklinks=true}
\urlstyle{same}
\captionsetup{font=small,justification=justified,singlelinecheck=false,skip=3pt}
\setlength{\emergencystretch}{2em}
\clubpenalty=10000 \widowpenalty=10000
\renewcommand{\arraystretch}{1.08}
\makeatletter
% a page of floats starts at the top of the page, not in its middle
\setlength{\@fptop}{0pt}\setlength{\@fpsep}{14pt plus 2fil}\setlength{\@fpbot}{0pt plus 1fil}
% wide floats stacked on one page may shrink their gaps a little instead of overflowing the page
\setlength{\dblfloatsep}{12pt plus 2pt minus 4pt}\setlength{\dbltextfloatsep}{16pt plus 2pt minus 6pt}
% heading position marks: \mdtexpos{n}{page}{y in sp}, written at shipout
\newcommand{\mdtexmark}[1]{\pdfsavepos\protected@write\@auxout{}{\string\mdtexpos{#1}{\thepage}{\noexpand\the\pdflastypos}}}
\providecommand{\mdtexpos}[3]{}
\providecommand{\mdtexgeom}[5]{}
\AtBeginDocument{\immediate\write\@auxout{\string\mdtexgeom{\the\paperheight}{\the\topmargin}{\the\headheight}{\the\headsep}{\the\textheight}}}
\makeatother
"""


def convert(md_path: Path, tex_path: Path) -> None:
    md = md_path.read_text(encoding="utf-8")
    blocks = parse(md)
    em = Emitter(md_path, tex_path)
    body: list[str] = []

    i = 0
    title = None
    meta: list[str] = []
    if blocks and isinstance(blocks[0], Heading) and blocks[0].level == 1:
        title = blocks[0].text
        i = 1
        while i < len(blocks) and not (isinstance(blocks[i], Heading) and blocks[i].level <= 2):
            b = blocks[i]
            meta.append(b.text if hasattr(b, "text") else "")
            i += 1
        if any(isinstance(x, Heading) and x.level == 2 and x.text.strip().lower() == "supplement" for x in blocks):
            # the paragraphs after the author line (other than the correspondence line) are the title page's note
            intro = [m for m in meta[1:] if m and not m.lower().startswith("correspondence:")]
            body.append(em.title_page_supplement(title, meta, intro))
        else:
            body.append(em.title_block(title, meta))

    while i < len(blocks):
        b = blocks[i]
        if isinstance(b, Heading) and b.level == 2 and b.text.strip().lower() == "abstract":
            j = i + 1
            inner = []
            while j < len(blocks) and not isinstance(blocks[j], Heading):
                inner.append(blocks[j])
                j += 1
            body.append(em.abstract(inner))
            i = j
            continue
        if isinstance(b, Heading) and b.text.strip().lower() == "references":
            body.append(em.heading(b))
            i += 1
            while i < len(blocks) and not isinstance(blocks[i], Heading):
                i += 1          # any reference list typed under the heading is replaced by the .bib
            continue
        body.append(em.block(b))
        i += 1

    pdf_title = esc(title or md_path.stem)
    head = PREAMBLE + "\\icmltitlerunning{%s}\n\\hypersetup{pdftitle={%s}}\n" % (esc(title or ""), pdf_title)
    marks = "\n".join("%%%% SECMARK %d %d %s" % (k, lvl, ttl.replace("\n", " "))
                      for k, (lvl, ttl) in enumerate(em.marks))
    cites = "%% CITED " + " ".join(dict.fromkeys(CITED)) + "\n"
    doc = (
        "%% Generated by paper/src/md2tex.py from %s. Do not edit; edit the template in paper/src.\n" % md_path.name
        + head + marks + "\n" + cites + "\\begin{document}\n" + "\n\n".join(body) + "\n\\end{document}\n"
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(doc, encoding="utf-8")

    md_tables = sum(1 for b in blocks if isinstance(b, Table))
    tex_tables = len(re.findall(r"\\begin\{(tabular|tabularx|longtable)\}", doc))
    print(f"md2tex: wrote {tex_path} ({len(blocks)} blocks, {len(em.marks)} headings, "
          f"tables: {md_tables} in markdown, {tex_tables} in tex; "
          f"{doc.count(chr(92) + 'begin{table*}')} across both columns)")
    if md_tables != tex_tables:
        print("md2tex: ERROR: table count mismatch", file=sys.stderr)
        sys.exit(2)


# -------------------------------------------------------------- page report --
def pt(s: str) -> float:
    return float(s.strip().replace("pt", ""))


def pages(tex_path: Path) -> int:
    stem = tex_path.with_suffix("")
    log = stem.with_suffix(".log")
    aux = stem.with_suffix(".aux")
    if not log.exists() or not aux.exists():
        print("md2tex: no .log/.aux next to the tex; compile first", file=sys.stderr)
        return 1
    log_text = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Output written on .*?\((\d+) pages?", log_text)
    total = int(m.group(1)) if m else None

    marks: dict[int, tuple[int, str]] = {}
    for mm in re.finditer(r"^%% SECMARK (\d+) (\d+) (.*)$", tex_path.read_text(encoding="utf-8"), re.M):
        marks[int(mm.group(1))] = (int(mm.group(2)), mm.group(3))
    aux_text = aux.read_text(encoding="utf-8", errors="replace")
    pos: dict[int, tuple[int, float]] = {}
    for mm in re.finditer(r"\\mdtexpos\{(\d+)\}\{(\d+)\}\{(-?\d+)\}", aux_text):
        pos[int(mm.group(1))] = (int(mm.group(2)), int(mm.group(3)) / 65536.0)

    print(f"pages: total = {total}")
    for name in ("References", "Appendix"):
        k = next((k for k, (lvl, ttl) in marks.items() if ttl.strip() == name), None)
        if k is not None and k in pos:
            print(f"pages: {name} mark on page {pos[k][0]}")
    ref = next((k for k, (lvl, ttl) in marks.items() if ttl.strip() == "References"), None)
    if ref is not None and ref in pos:
        print(f"pages: main text = pages 1 to {pos[ref][0]} (References begin on page {pos[ref][0]}; "
              f"see check_report.py for the measured fill of that page)")
    by_page: dict[int, list[str]] = {}
    for k in sorted(marks):
        if k in pos:
            page, y = pos[k]
            lvl, ttl = marks[k]
            by_page.setdefault(page, []).append(("  " if lvl >= 3 else "") + ttl)
    for page in sorted(by_page):
        print(f"page {page}: " + " | ".join(by_page[page]))
    over = re.findall(r"^Overfull \\hbox \(([\d.]+)pt too wide\)(.*)$", log_text, re.M)
    print(f"layout: {len(over)} overfull hbox(es)" + (
        "; worst: " + ", ".join(f"{float(a):.1f}pt" for a, _ in sorted(over, key=lambda t: -float(t[0]))[:5]) if over else ""))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "convert":
        convert(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) >= 3 and argv[1] == "pages":
        return pages(Path(argv[2]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
