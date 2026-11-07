"""Per-seed coverage: for every (model, task, cell), which seeds have >=1 VALID run, which exist only as broken,
and the max seed seen (the frontier). A seed absent from both lists never ran at all -- a silent gap the
per-cell valid/broken counts cannot show. Run from data/apart_controls/."""
import json, glob, os, re, sys, collections

MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ds", "luna", "qwen", "haiku", "gpt5m"]
PAT = re.compile(r"^(.+?)_(solo|cheat)_(qwen|haiku|gpt5m|ds|luna)_s(\d+)$")

valid = collections.defaultdict(set)    # (model,task,cell) -> seeds with a valid run
broken_only = collections.defaultdict(set)
for f in glob.glob("*/runs/*/episodes.jsonl"):
    run = f.split("/")[2]
    m = PAT.match(run)
    if not m: continue
    task, cell, mdl, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
    if mdl not in MODELS: continue
    try: e = json.loads(open(f).readline())
    except Exception: continue
    key = (mdl, task, cell)
    # a clean cap end (cost or time limit, no harness failure, real turns) is a VALID end per spec, not broken
    clean_cap = (e.get("ended_by") == "limit" and e.get("failure") is None and (e.get("n_turns") or 0) > 0)
    if e.get("ok") or clean_cap: valid[key].add(seed)
    else: broken_only[key].add(seed)
for k in broken_only: broken_only[k] -= valid[k]   # a seed with any valid copy is covered

def rng(s):
    if not s: return "-"
    s = sorted(s); out = []; a = b = s[0]
    for x in s[1:]:
        if x == b + 1: b = x
        else: out.append(f"{a}" if a == b else f"{a}-{b}"); a = b = x
    out.append(f"{a}" if a == b else f"{a}-{b}")
    return ",".join(out)

print(f"{'model':6} {'task':9} {'cell':5} {'valid#':>6} {'frontier':>8}  valid seeds            missing<frontier  broken-only")
for mdl in MODELS:
    keys = sorted(k for k in set(valid) | set(broken_only) if k[0] == mdl)
    for k in keys:
        v = valid[k]; b = broken_only[k]
        front = max(v | b) if (v | b) else 0
        missing = sorted(set(range(1, front + 1)) - v - b)   # never ran at all, below the frontier
        flag = "  <<< GAP" if missing or b else ""
        print(f"{k[0]:6} {k[1]:9} {k[2]:5} {len(v):6} {front:8}  {rng(v):22} {rng(missing):16} {rng(b):10}{flag}")
    print()
