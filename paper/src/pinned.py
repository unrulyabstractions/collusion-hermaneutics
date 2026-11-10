"""Read result files at pinned commits (git show), and parse the markdown tables of RESULTS.md files.

Every table the report builds from the 2026-11-09 results reads its source through this module, so a
number in the report traces to one file at one commit, whatever other sessions write to the working tree.
"""
import csv
import gzip
import io
import os
import re
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEAM = "195493d"     # team_probes and attn_probe RESULTS.md (gaps closed)
EXT = "5ecc85b"      # external_probes
CN = "fe5f66c"       # controls_nulls
JA = "4504c63"       # judges_all
MX = "d38598d"       # matrix_probes

_cache = {}


def show(commit, path, binary=False):
    key = (commit, path, binary)
    if key not in _cache:
        out = subprocess.run(["git", "-C", REPO, "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
        _cache[key] = out if binary else out.decode("utf-8")
    return _cache[key]


def csv_rows(commit, path):
    raw = show(commit, path, binary=True)
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def split_cells(line):
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return [c.strip() for c in s.split(" | ")]


def md_tables(commit, path):
    """Every pipe table of a markdown file: dicts with the heading stack above it, its header and rows."""
    text = show(commit, path)
    lines = text.splitlines()
    heads = {}
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            heads = {k: v for k, v in heads.items() if k < lvl}
            heads[lvl] = m.group(2).strip()
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$", lines[i + 1]):
            header = split_cells(line)
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(split_cells(lines[j]))
                j += 1
            out.append({"heads": [heads[k] for k in sorted(heads)], "header": header, "rows": rows,
                        "commit": commit, "path": path, "line": i + 1})
            i = j
            continue
        i += 1
    return out


def table(commit, path, *heading_parts, first_col=None, last=None, nth=None):
    """The one table whose heading stack contains every part (and whose first header cell is first_col,
    and whose nearest heading is exactly last)."""
    hits = [t for t in md_tables(commit, path)
            if all(any(p in h for h in t["heads"]) for p in heading_parts)
            and (first_col is None or t["header"][0] == first_col)
            and (last is None or t["heads"][-1] == last)]
    if nth is not None:      # several tables under one heading: take them in file order
        return hits[nth]
    assert len(hits) == 1, (path, heading_parts, first_col, [(t["heads"], t["header"][:2]) for t in hits])
    return hits[0]


def row(tab, **match):
    """The one row whose named columns equal the given values (column names as in the header)."""
    idx = {h: k for k, h in enumerate(tab["header"])}
    hits = [r for r in tab["rows"] if all(r[idx[c]] == v for c, v in match.items())]
    assert len(hits) == 1, (tab["heads"], match, len(hits))
    return dict(zip(tab["header"], hits[0]))


def where(tab, r):
    return f"{tab['path']}@{tab['commit']} line {tab['line']} ({' > '.join(tab['heads'][-2:])})"
