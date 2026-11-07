"""Completion-gate coverage, intent-correct per the driver's manifest.

Target seeds per (model, task, cell):  12 if (model=='qwen' and task=='game')  else 50.  (only cap = qwen chess)
A seed is COVERED iff some host copy is ok==true OR a clean cost-cap end (real turns, failure=None, ended_by=limit).
Timeouts exit rc=0 but ok=false -> NOT covered. Dedupe across host copies (duplicate orchestrators made copies).

For each cell, over seeds 1..target, every seed is either covered or a gap. Gaps are split so the driver
re-queues only what will progress:
  NEVER_RAN      no run dir for this seed            -> re-queue
  TIMEOUT        launched, every copy timed out       -> re-queue (at raised limit) or freeze if it recurs
  402            launched, every copy hit key 402     -> re-queue (healthy key)
  ENV            launched, compose/disk host failure  -> re-queue
  MIXED_RETRY    launched, mix of the above           -> re-queue
  OTHER          launched, some other non-valid       -> inspect
COST_CAP-only seeds are COVERED (re-running just caps again); counted, not re-queued.

Run from data/apart_controls/.  `--requeue` prints only the actionable list (what to hand the driver).
"""
import json, glob, os, re, sys, collections

PAT = re.compile(r"^(.+?)_(solo|cheat)_(qwen|haiku|gpt5m|ds|luna)_s(\d+)$")
MODELS = {"ds", "luna", "qwen", "haiku", "gpt5m"}
REQUEUE_ONLY = "--requeue" in sys.argv

def target(model, task):
    return 12 if (model == "qwen" and task == "game") else 50

def episode_state(e):
    err = str(e.get("error") or "")
    if e.get("ok") is True: return "valid"
    if e.get("failure") == "timeout" or ("time limit" in err and (e.get("n_turns") or 0) > 0): return "timeout"
    if "time limit" in err and (e.get("n_turns") or 0) == 0: return "harness"
    if "402" in err: return "402"
    if "No space left" in err or "No services started" in err: return "env"
    if e.get("ended_by") == "limit" and e.get("failure") is None and (e.get("n_turns") or 0) > 0: return "cost_cap"
    return "other"

def run_state(epf):
    states = [episode_state(json.loads(l)) for l in open(epf) if l.strip()]
    if not states: return "empty"
    if all(s in ("valid", "cost_cap") for s in states):
        return "valid" if "valid" in states else "cost_cap"
    for pref in ("timeout", "402", "env", "harness", "other"):
        if pref in states: return pref
    return "other"

# gather: (model,task,cell) -> {seed: set(states)}   and the universe of (task,cell) seen anywhere
cells = collections.defaultdict(lambda: collections.defaultdict(set))
taskcells = set()
for f in glob.glob("*/runs/*/episodes.jsonl"):
    m = PAT.match(f.split("/")[2])
    if not m: continue
    task, cell, mdl, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
    cells[(mdl, task, cell)][seed].add(run_state(f))
    taskcells.add((task, cell))

# qwen-game (chess) is FROZEN reduced-n per the driver/main ruling: its gaps are NOT force-requeued;
# we report its final valid-n instead. All other cells' gaps are force-requeue targets.
def is_frozen(mdl, task):
    return mdl == "qwen" and task == "game"

requeue = collections.defaultdict(list)
cov_rows = []
frozen_n = {}   # (model,task,cell) -> (valid_count, target) for the reduced-n note
for mdl in sorted(MODELS):
    for (task, cell) in sorted(taskcells):
        tgt = target(mdl, task)
        seeds = cells.get((mdl, task, cell), {})
        covered = sum(1 for s in range(1, tgt + 1)
                      if ("valid" in seeds.get(s, ())) or ("cost_cap" in seeds.get(s, ())))
        gaps = []
        for s in range(1, tgt + 1):
            st = seeds.get(s, set())
            if "valid" in st or "cost_cap" in st: continue
            gaps.append(s)
            if is_frozen(mdl, task):
                continue   # frozen cell: do NOT add to the force-requeue list
            if not st: reason = "NEVER_RAN"
            elif st <= {"timeout"}: reason = "TIMEOUT"
            elif st <= {"402"}: reason = "402"
            elif st <= {"env"}: reason = "ENV"
            elif st <= {"timeout", "402", "env", "harness"}: reason = "MIXED_RETRY"
            else: reason = "OTHER"
            requeue[reason].append(f"{task}_{cell}_{mdl}_s{s}")
        cov_rows.append((mdl, task, cell, covered, tgt, len(gaps)))
        if is_frozen(mdl, task):
            frozen_n[(mdl, task, cell)] = (covered, tgt)

if not REQUEUE_ONLY:
    total_tgt = sum(t for _, _, _, _, t, _ in cov_rows)
    total_cov = sum(c for _, _, _, c, _, _ in cov_rows)
    print(f"COVERAGE {total_cov}/{total_tgt} target cells filled; {total_tgt-total_cov} gaps")
    print("incomplete cells (covered<target):")
    for mdl, task, cell, c, t, g in cov_rows:
        if c < t:
            tag = "  [FROZEN reduced-n]" if is_frozen(mdl, task) else ""
            print(f"   {mdl:6} {task:9} {cell:5} {c}/{t}  (gaps={g}){tag}")
    print()
    print("=== QWEN-CHESS FROZEN reduced-n (NOT re-queued; document n as-is) ===")
    for (mdl, task, cell), (c, t) in sorted(frozen_n.items()):
        print(f"   {mdl} {task} {cell}: final valid-n = {c}/{t}")
    print()
req_total = sum(len(v) for v in requeue.values())
print(f"=== FORCE-REQUEUE LIST (non-chess gaps only; qwen-chess frozen & excluded): {req_total} ===")
for reason in ("NEVER_RAN", "TIMEOUT", "402", "ENV", "MIXED_RETRY", "OTHER"):
    lst = sorted(requeue.get(reason, []), key=lambda s: (s.split("_s")[0], int(s.split("_s")[-1])))
    print(f"-- {reason}: {len(lst)}")
    if REQUEUE_ONLY or len(lst) <= 30:
        for r in lst: print(f"     {r}")
