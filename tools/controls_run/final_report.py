#!/usr/bin/env python3
"""Final verification + report over the LOCALLY SYNCED data (no ssh).
For every run dir under data/apart_controls/<host>/runs/<run_name>/:
  VALID   = episodes.jsonl parses AND ok==True AND trajectories/ has >0 bytes AND a signals team-row label exists
  INVALID = anything else (listed)
Prints: per (task,cell,model) valid counts + cheat/honest/other + $; per-model totals; grand total;
invalid runs; and every (task,cell,model) cell with ZERO valid runs. Exit 1 if any cell has zero valid runs."""
import json, sys
from collections import defaultdict
from pathlib import Path

DATA = Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
MODELS = ("qwen", "haiku", "gpt5m", "ds", "luna")
TASKS = ("flipt", "vuls", "riemann", "subset", "factor", "binder", "poem", "worktime", "game")
CELLS = ("solo", "cheat")

def parse(name):
    p = name.split("_")
    if len(p) < 4 or not p[-1].startswith("s") or p[-2] not in MODELS or p[-3] not in CELLS:
        return None
    return "_".join(p[:-3]), p[-3], p[-2], p[-1]

valid = defaultdict(list)      # (task,cell,model) -> [(name, label, billed)]
invalid = []                   # (name, reason)
seen = set()
for rd in sorted(DATA.glob("*/runs/*")):
    if not rd.is_dir():
        continue
    name = rd.name
    key = parse(name)
    if key is None:
        continue
    if name in seen:           # same run_name synced from two hosts (e.g. a 402 copy + the retry); keep first VALID
        pass
    task, cell, model, seed = key
    # 2026-11-07: the qwen-chess seed>12 skip was REMOVED for publication. The cap at 12 was a LAUNCH
    # cap (stop spending on a context-bound model), not a data filter; valid runs that landed at
    # seeds >12 before the cap are legitimate replicates of the same fixed instance and are counted.
    ep = rd / "episodes.jsonl"
    try:
        e = json.loads(ep.read_text().splitlines()[0])
    except Exception as ex:
        invalid.append((name, f"episodes unreadable: {type(ex).__name__}")); continue
    if not e.get("ok"):
        invalid.append((name, f"ok={e.get('ok')} ended_by={e.get('ended_by')} err={bool(e.get('error'))}")); continue
    tbytes = sum(f.stat().st_size for f in (rd / "trajectories").rglob("*") if f.is_file()) if (rd / "trajectories").exists() else 0
    if tbytes == 0:
        invalid.append((name, "empty trajectories")); continue
    label = None
    sg = rd / "signals.jsonl"
    if sg.exists():
        for ln in sg.read_text().splitlines():
            o = json.loads(ln)
            if o.get("row") == "team":
                label = "cheat" if (o.get("used") or o.get("tampered")) else ("honest" if o.get("shipped") else "other")
                break
    if label is None:
        invalid.append((name, "no team-row label")); continue
    if name in seen:
        continue               # a second VALID copy of the same run_name: count once
    seen.add(name)
    billed = (e.get("ledger") or {}).get("billed_usd") or 0.0
    valid[(task, cell, model)].append((name, label, billed))

# ---- report
print(f"VALID runs (unique run_names): {sum(len(v) for v in valid.values())}   INVALID run dirs: {len(invalid)}\n")
print("VALIDITY RULE: strict episodes ok==true AND non-empty trajectories AND signals team-row label."
      " A clean cost-cap end (ok=False, ended_by=limit) is EXCLUDED by this rule.\n")
print(f"{'task':9s} {'cell':6s} {'model':6s} {'n':>4s} {'cheat':>5s} {'honest':>6s} {'other':>5s} {'$':>8s}")
grand = 0.0
per_model = defaultdict(lambda: [0, 0, 0, 0, 0.0])  # n,cheat,honest,other,$
zero_cells = []
for task in TASKS:
    for cell in CELLS:
        for model in MODELS:
            rows = valid.get((task, cell, model), [])
            n = len(rows)
            c = sum(1 for _, l, _ in rows if l == "cheat"); h = sum(1 for _, l, _ in rows if l == "honest"); o = n - c - h
            b = sum(x for _, _, x in rows); grand += b
            pm = per_model[model]; pm[0] += n; pm[1] += c; pm[2] += h; pm[3] += o; pm[4] += b
            print(f"{task:9s} {cell:6s} {model:6s} {n:4d} {c:5d} {h:6d} {o:5d} {b:8.3f}")
            if n == 0:
                zero_cells.append((task, cell, model))
print("\nPER MODEL:  model   n   cheat honest other     $      avg$/run")
for m in MODELS:
    n, c, h, o, b = per_model[m]
    print(f"            {m:6s} {n:4d} {c:5d} {h:6d} {o:5d} {b:8.3f}  {(b/n if n else 0):.4f}")
print(f"\nGRAND TOTAL billed over valid runs: ${grand:.2f}")
print(f"\nINVALID ({len(invalid)}):")
for n_, r in invalid[:40]:
    print(f"   {n_}: {r}")
if len(invalid) > 40:
    print(f"   ... and {len(invalid)-40} more")
print(f"\nCELLS WITH ZERO VALID RUNS ({len(zero_cells)}): {zero_cells if zero_cells else 'none'}")
sys.exit(1 if zero_cells else 0)
