#!/usr/bin/env python3
"""Markdown (GitHub flavour) to LaTeX for the hackathon report, plus a page report.

Usage
  md2tex.py convert REPORT.md report.tex     write report.tex (and nothing else)
  md2tex.py pages report.tex                 read report.log / report.aux after xelatex
                                             and print the page summary

Standard library only. The converter is lossless for text: it escapes LaTeX
specials, converts **bold**, *italic*, `code`, links, URLs, headings, lists,
pipe tables (with "**Table X.**" caption lines) and images. It never rewords.

Headings: "# " -> title block (the paragraphs that follow it, up to the first
"## ", are the author / affiliation / meta block); "## Abstract" -> bold
"Abstract" heading with the text in \\small; other "## " -> \\section*,
"### " -> \\subsection*, "#### " -> \\subsubsection*. Numbers stay as written.

Tables: tabularx (longtable above LONGTABLE_ROWS body rows) with column widths
estimated from the cell text, the first column favoured, text wrapped and
ragged right, caption above in a [H] float (set MD2TEX_PLACEMENT=ht to let a
table that does not fit move to the top of the next page while the text flows
on). Nothing is auto-numbered.

Every heading carries a position mark that is written to the .aux at shipout
(page and vertical position), so `pages` can say what starts on which page.
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------- settings --
TEXTWIDTH_PT = 469.75          # letterpaper, 1in margins: 6.5in
TABCOLSEP_PT = 3.0
LONGTABLE_ROWS = 35            # body rows above which longtable is used
TABLE_SIZES = [("\\small", 10.0), ("\\footnotesize", 9.0), ("\\scriptsize", 8.0)]
SHORT_CELL_EM = 12.0           # a column whose body cells are all shorter than this
                               # is "numeric": prefer a size where its cells wrap
                               # to at most two lines
CELL_TARGET_EM = 45.0          # a column never asks for more than this per line
FIRST_COL_BONUS = 1.3
TABLE_PLACEMENT = os.environ.get("MD2TEX_PLACEMENT", "!htbp")   # !htbp (default): float within the
                               # section, fenced by \FloatBarrier at every heading; H: strictly here
HEADING_TOP_SKIP_PT = 25.0     # estimated gap from a heading's baseline to the
                               # top of its before-skip (for the page fraction)

CAPTION_RE = re.compile(r"^\*\*(Table|Figure)\b[^*]*\*\*")
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
        # inside a table cell a slash may take a line break (paths, "a/b")
        out = out.replace("/", "\\slash{}")
    return out


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


def inline(s: str, tex_dir: Path | None = None, md_dir: Path | None = None, cell: bool = False) -> str:
    out: list[str] = []
    pos = 0
    for m in INLINE_RE.finditer(s):
        out.append(text_run(s[pos:m.start()], cell))
        pos = m.end()
        if m.group("code") is not None:
            out.append(code_run(m.group("ctext")))
        elif m.group("ipath") is not None:
            out.append(r"\includegraphics[width=\linewidth]{" + image_path(m.group("ipath"), tex_dir, md_dir) + "}")
        elif m.group("lhref") is not None:
            out.append(r"\href{" + m.group("lhref") + "}{" + inline(m.group("ltext"), tex_dir, md_dir, cell) + "}")
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
        elif c == " ":
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
        for tok in re.sub(r"\*+", "", s).split():
            for piece in re.split(r"(?<=/)", tok):
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


def column_plan(header: list[str], rows: list[list[str]]) -> tuple[str, float, list[float]]:
    """Pick a font size and column widths (pt). Returns (size cmd, size pt, widths).

    The largest size whose unbreakable pieces fit is tried first. For every
    size the slack is spread (fill); a size is accepted when every numeric
    column (short body cells) wraps to at most two lines, else the next
    smaller size is tried; if none satisfies that, the size with the fewest
    wrapped lines wins, the larger size on ties.
    """
    ncols = len(header)
    avail = TEXTWIDTH_PT - ncols * 2 * TABCOLSEP_PT
    longest_tok = [0.0] * ncols      # header and body
    longest_any = [0.0] * ncols      # header and body, whole cell
    longest_body = [0.0] * ncols     # body only, whole cell
    two_line = [0.0] * ncols         # body only
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

    candidates = []
    for cmd, pt in TABLE_SIZES:
        minw = [(t + 0.3) * pt for t in longest_tok]
        if sum(minw) > avail:
            continue
        widths = fill(minw, two_line, longest_any, short, pt, avail)
        lines = max((math.ceil(longest_body[j] * pt / widths[j]) for j in range(ncols) if short[j]), default=1)
        candidates.append((lines, -pt, cmd, widths))
        if lines <= 2:
            break
    if not candidates:
        cmd, pt = TABLE_SIZES[-1]
        minw = [(t + 0.3) * pt for t in longest_tok]
        scale = avail / sum(minw)
        print(f"md2tex: warning: table with {ncols} columns does not fit at {cmd}; "
              f"columns scaled by {scale:.2f}", file=sys.stderr)
        return cmd, pt, [w * scale for w in minw]
    lines, neg_pt, cmd, widths = min(candidates, key=lambda c: (c[0], c[1]))
    return cmd, -neg_pt, widths


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

    def inl(self, s: str) -> str:
        return inline(s, self.tex_dir, self.md_dir)

    def mark(self, level: int, title: str) -> str:
        self.marks.append((level, title))
        return r"\mdtexmark{%d}" % (len(self.marks) - 1)

    # -- blocks
    def title_block(self, title: str, meta: list[str]) -> str:
        out = ["\\begin{center}",
               "{\\LARGE\\bfseries " + self.inl(title) + self.mark(0, "Title block") + "\\par}",
               "\\vspace{1.5ex}"]
        for k, line in enumerate(meta):
            size = "\\large " if k == 0 else ""
            out.append("{" + size + self.inl(line) + "\\par}")
            if k == 0:
                out.append("\\vspace{0.5ex}")
        out.append("\\end{center}")
        out.append("\\vspace{0.5ex}")
        return "\n".join(out)

    def abstract(self, blocks: list) -> str:
        out = ["\\noindent{\\bfseries Abstract" + self.mark(1, "Abstract") + "}\\par\\vspace{0.4ex}",
               "\\begingroup\\small"]
        for b in blocks:
            out.append(self.block(b, noindent=True))
        out.append("\\endgroup\\vspace{1ex}")
        return "\n".join(out)

    def heading(self, h: Heading) -> str:
        cmd = {2: "\\section*", 3: "\\subsection*"}.get(h.level, "\\subsubsection*")
        # fence floats at section level only: a table may float to the next page top while the
        # following subsection fills the current page; it can never leave its section
        barrier = "" if (TABLE_PLACEMENT == "H" or h.level != 2) else "\\FloatBarrier\n"
        return barrier + cmd + "{" + self.inl(h.text) + self.mark(h.level, h.text) + "}"

    def para(self, p: Para, noindent: bool = False) -> str:
        body = self.inl(p.text)
        if REF_ITEM_RE.match(p.text):
            return "\\noindent\\hangindent=1.5em\\hangafter=1 " + guard(body) + "\\par"
        return ("\\noindent " if noindent else "") + guard(body) + "\n"

    def list_block(self, lb: ListBlock) -> str:
        env = "enumerate" if lb.ordered else "itemize"
        opts = "leftmargin=*, itemsep=2pt, topsep=3pt"
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
        out = ["\\begin{figure}[" + TABLE_PLACEMENT + "]", "\\centering",
               "\\includegraphics[width=\\linewidth]{" + image_path(im.path, self.tex_dir, self.md_dir) + "}"]
        if im.caption:
            out.append("\\caption*{" + self.inl(im.caption) + "}")
        out.append("\\end{figure}")
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
        size_cmd, size_pt, widths = column_plan(t.header, rows)
        avail = TEXTWIDTH_PT - ncols * 2 * TABCOLSEP_PT
        cell = lambda c: inline(c, self.tex_dir, self.md_dir, cell=True)
        cells = lambda row: " & ".join(guard(cell(c)) for c in row) + " \\\\"
        head = " & ".join(guard("\\textbf{" + cell(c) + "}") if c else "" for c in t.header) + " \\\\"
        out: list[str] = []
        if len(rows) > LONGTABLE_ROWS:
            spec = "".join(
                ">{\\RaggedRight\\arraybackslash}p{\\dimexpr %.5f\\textwidth-2\\tabcolsep\\relax}" % (w / TEXTWIDTH_PT)
                for w in widths)
            out += ["\\begingroup", size_cmd, "\\setlength{\\tabcolsep}{%gpt}" % TABCOLSEP_PT,
                    "\\begin{longtable}{" + spec + "}"]
            if t.caption:
                out.append("\\caption*{" + self.inl(t.caption) + "}\\\\")
            out += ["\\toprule", head, "\\midrule", "\\endfirsthead",
                    "\\toprule", head, "\\midrule", "\\endhead",
                    "\\bottomrule", "\\endlastfoot"]
            out += [cells(r) for r in rows]
            out += ["\\end{longtable}", "\\endgroup"]
        else:
            spec = "".join(
                ">{\\hsize=%.4f\\hsize\\RaggedRight\\arraybackslash}X" % (w / (avail / ncols))
                for w in widths)
            out += ["\\begin{table}[" + TABLE_PLACEMENT + "]", "\\centering", size_cmd,
                    "\\setlength{\\tabcolsep}{%gpt}" % TABCOLSEP_PT]
            if t.caption:
                out.append("\\caption*{" + self.inl(t.caption) + "}")
            out += ["\\begin{tabularx}{\\textwidth}{" + spec + "}", "\\toprule", head, "\\midrule"]
            out += [cells(r) for r in rows]
            out += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
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


PREAMBLE = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\IfFontExistsTF{texgyretermes-regular.otf}{%
  \setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
    ItalicFont=*-italic, BoldItalicFont=*-bolditalic, Ligatures=TeX]}{%
  \IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}[Ligatures=TeX]}{}}
\IfFontExistsTF{texgyrecursor-regular.otf}{%
  \setmonofont{texgyrecursor}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
    ItalicFont=*-italic, BoldItalicFont=*-bolditalic, Scale=MatchLowercase]}{}
\usepackage{microtype}
\usepackage{xcolor}
\definecolor{darkblue}{RGB}{0,0,139}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{float}
\usepackage{placeins}
\usepackage{enumitem}
\usepackage{ragged2e}
\usepackage{graphicx}
\PassOptionsToPackage{hyphens}{url}
\usepackage[colorlinks=true,linkcolor=darkblue,urlcolor=darkblue,citecolor=darkblue,breaklinks=true]{hyperref}
\urlstyle{same}
\usepackage{caption}
\captionsetup{font=small,justification=justified,singlelinecheck=false,skip=4pt}
\captionsetup[table]{position=above}
\setlength{\emergencystretch}{2em}
\clubpenalty=10000 \widowpenalty=10000
\renewcommand{\arraystretch}{1.12}
\renewcommand{\topfraction}{0.9}\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.07}\renewcommand{\floatpagefraction}{0.75}
\makeatletter
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
            if isinstance(b, Para):
                meta.append(b.text)
            else:
                meta.append(b.text if hasattr(b, "text") else "")
            i += 1
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
        if isinstance(b, Heading) and b.level == 1:
            body.append(em.heading(Heading(2, b.text)))
        else:
            body.append(em.block(b))
        i += 1

    pdf_title = esc(title or md_path.stem)
    pdf_author = esc(meta[0]) if meta else ""
    head = PREAMBLE + "\\hypersetup{pdftitle={%s},pdfauthor={%s}}\n" % (pdf_title, pdf_author)
    marks = "\n".join("%%%% SECMARK %d %d %s" % (k, lvl, ttl.replace("\n", " "))
                      for k, (lvl, ttl) in enumerate(em.marks))
    doc = (
        "%% Generated by report/gen/md2tex.py from %s. Do not edit; edit the markdown.\n" % md_path.name
        + head + marks + "\n\\begin{document}\n" + "\n\n".join(body) + "\n\\end{document}\n"
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(doc, encoding="utf-8")

    md_tables = sum(1 for b in blocks if isinstance(b, Table))
    tex_tables = doc.count("\\begin{tabularx}") + doc.count("\\begin{longtable}")
    print(f"md2tex: wrote {tex_path} ({len(blocks)} blocks, {len(em.marks)} headings, "
          f"tables: {md_tables} in markdown, {tex_tables} in tex)")
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
    pdf = stem.with_suffix(".pdf")
    if not log.exists() or not aux.exists():
        print("md2tex: no .log/.aux next to the tex; compile first", file=sys.stderr)
        return 1
    log_text = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Output written on .*?\((\d+) pages?", log_text)
    total = int(m.group(1)) if m else None
    if total is None and pdf.exists():
        import shutil
        import subprocess
        if shutil.which("pdftotext"):
            txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
            total = txt.count("\f") or None

    marks: dict[int, tuple[int, str]] = {}
    for mm in re.finditer(r"^%% SECMARK (\d+) (\d+) (.*)$", tex_path.read_text(encoding="utf-8"), re.M):
        marks[int(mm.group(1))] = (int(mm.group(2)), mm.group(3))
    aux_text = aux.read_text(encoding="utf-8", errors="replace")
    pos: dict[int, tuple[int, float]] = {}
    for mm in re.finditer(r"\\mdtexpos\{(\d+)\}\{(\d+)\}\{(-?\d+)\}", aux_text):
        pos[int(mm.group(1))] = (int(mm.group(2)), int(mm.group(3)) / 65536.0)
    g = re.search(r"\\mdtexgeom\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", aux_text)
    if g:
        paperheight, topmargin, headheight, headsep, textheight = (pt(x) for x in g.groups())
        text_top = paperheight - (72.27 + topmargin + headheight + headsep)
    else:
        text_top, textheight = 722.7, 650.43

    def frac_before(y: float) -> float:
        return min(1.0, max(0.0, (text_top - y - HEADING_TOP_SKIP_PT) / textheight))

    print(f"pages: total = {total}")
    ref = next((k for k, (lvl, ttl) in marks.items() if ttl.strip().lower().startswith("references")), None)
    if ref is not None and ref in pos:
        page, y = pos[ref]
        f = frac_before(y)
        print(f"pages: References heading starts on page {page}, {f:.2f} of the way down")
        print(f"pages: body pages (title through the section before References) = {page - 1 + f:.2f}")
    else:
        print("pages: References heading not found in the aux")
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
