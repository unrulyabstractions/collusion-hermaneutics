"""Early-stop completion gate (DELIBERATE stop to lock spend; NOT a failure).

The run was stopped on purpose. So:
  - "never-run" seeds below target are UNCOLLECTED BY DESIGN -> reported as the deliberate remainder, NOT gaps, NOT re-queued.
  - We verify every KEPT run is valid (ok==true, or clean cost-cap), dedup across host copies.
  - We report per-model PRESENT and VALID counts and check BALANCE across the 5 models (the study compares them).
  - qwen game (chess) target=12 (frozen 262k-context reduced-n); every other cell target=50.
  - A small residue of ATTEMPTED-but-FAILED cells (402/timeout/env/context) is reported SEPARATELY as an optional
    tiny re-queue — the user chose to lock spend, so it is the coordinator's call, not auto-requeued.

Run from data/apart_controls/.
"""
import json, glob, re, collections

PAT = re.compile(r"^(.+?)_(solo|cheat)_(qwen|haiku|gpt5m|ds|luna)_s(\d+)$")
MODELS = ["ds", "luna", "qwen", "haiku", "gpt5m"]
def target(m, t): return 12 if (m == "qwen" and t == "game") else 50

def estate(e):
    err = str(e.get("error") or "")
    if e.get("ok") is True: return "valid"
    if "maximum context length" in err or "Upstream error from DeepInfra" in err: return "context400"
    if e.get("failure") == "timeout" or ("time limit" in err and (e.get("n_turns") or 0) > 0): return "timeout"
    if "402" in err: return "402"
    if "No space left" in err or "No services started" in err: return "env"
    if e.get("ended_by") == "limit" and e.get("failure") is None and (e.get("n_turns") or 0) > 0: return "cost_cap"
    return "other"

def rstate(epf):
    st = [estate(json.loads(l)) for l in open(epf) if l.strip()]
    if not st: return "empty"
    if all(s in ("valid", "cost_cap") for s in st): return "valid" if "valid" in st else "cost_cap"
    for p in ("timeout", "context400", "402", "env", "other"):
        if p in st: return p
    return "other"

cells = collections.defaultdict(lambda: collections.defaultdict(set))
taskcells = set()
for f in glob.glob("*/runs/*/episodes.jsonl"):
    mo = PAT.match(f.split("/")[2])
    if not mo: continue
    t, c, m, s = mo.group(1), mo.group(2), mo.group(3), int(mo.group(4))
    cells[(m, t, c)][s].add(rstate(f))
    taskcells.add((t, c))

pm = {m: {"present": 0, "valid": 0, "target": 0} for m in MODELS}
failed_attempted = collections.defaultdict(list)   # reason -> [cell]
per_cell_valid = collections.defaultdict(dict)      # (task,cell) -> {model: valid_n}
for m in MODELS:
    for (t, c) in sorted(taskcells):
        T = target(m, t); pm[m]["target"] += T
        seeds = cells.get((m, t, c), {})
        vseeds = [s for s in range(1, T + 1) if {"valid", "cost_cap"} & seeds.get(s, set())]
        pseeds = [s for s in range(1, T + 1) if seeds.get(s)]
        pm[m]["present"] += len(pseeds); pm[m]["valid"] += len(vseeds)
        per_cell_valid[(t, c)][m] = len(vseeds)
        # attempted-but-failed (present, no valid) within target
        for s in pseeds:
            if s in vseeds: continue
            st = seeds[s]
            r = ("timeout" if "timeout" in st else "context400" if "context400" in st else
                 "402" if "402" in st else "env" if "env" in st else "other")
            failed_attempted[r].append(f"{t}_{c}_{m}_s{s}")

print("=== PER-MODEL present / valid / target (balance check) ===")
for m in MODELS:
    d = pm[m]; print(f"  {m:6} valid {d['valid']:4} / present {d['present']:4} / target {d['target']:4}")
vals = [pm[m]["valid"] for m in MODELS]
print(f"  balance: valid min={min(vals)} max={max(vals)} spread={max(vals)-min(vals)}")

print("\n=== BALANCE per (task,cell) across models (valid n; flag spread>6) ===")
for (t, c) in sorted(taskcells):
    row = per_cell_valid[(t, c)]
    vv = [row[m] for m in MODELS]
    # qwen-chess has target 12 not 50, so exclude it from the spread flag
    cmp = [row[m] for m in MODELS if not (m == "qwen" and t == "game")]
    flag = "  <<< IMBALANCE" if (max(cmp) - min(cmp) > 6) else ""
    print(f"  {t:9} {c:5}  " + " ".join(f"{m[:2]}={row[m]}" for m in MODELS) + flag)

print("\n=== ATTEMPTED-BUT-FAILED residue (optional tiny re-queue; coordinator's call) ===")
for r in ("timeout", "context400", "402", "env", "other"):
    lst = sorted(failed_attempted.get(r, []))
    # qwen-chess timeouts/context are frozen reduced-n, not residue to requeue
    lst = [x for x in lst if not (x.startswith("game_") and "_qwen_" in x)]
    print(f"  {r}: {len(lst)}" + ("  " + ", ".join(lst) if 0 < len(lst) <= 25 else ""))
