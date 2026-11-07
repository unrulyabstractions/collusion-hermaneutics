#!/usr/bin/env python3
"""Build a STABLE, deduped, valid-only HF staging snapshot of data/apart_controls.

Reuses the driver's validity+dedup logic (final_report.py):
  VALID = episodes.jsonl line0 parses AND ok==True AND trajectories/ > 0 bytes AND a signals team-row label exists.
Dedup by run_name ({task}_{cell}_{model}_s{seed}); keep the first VALID copy (host-alpha order).
STABLE = every file in the run dir (and matched eval) has mtime older than STABLE_AGE seconds,
so we never copy a dir mid-rsync. Unstable valid runs are deferred to the next snapshot.

Layout written to STAGE/apart_controls/<model>/<task>/<cell>/s<seed>/ :
  run.json episodes.jsonl signals.jsonl access_log.jsonl defense/** trajectories/** eval/<name>.eval
Also writes STAGE/manifest.jsonl and STAGE/_build_stats.json (consumed by the README builder).
Files are copied byte-for-byte (shutil.copy2); contents are not transformed.
"""
import json, os, shutil, time, sys
from collections import defaultdict
from pathlib import Path

DATA = Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
STAGE = Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad/hf_staging")
OUT = STAGE / "apart_controls"
STABLE_AGE = 60.0
NOW = time.time()

MODELS = ("qwen", "haiku", "gpt5m", "ds", "luna")
TASKS = ("flipt", "vuls", "riemann", "subset", "factor", "binder", "poem", "worktime", "game")
CELLS = ("solo", "cheat")

def parse(name):
    p = name.split("_")
    if len(p) < 4 or not p[-1].startswith("s") or p[-2] not in MODELS or p[-3] not in CELLS:
        return None
    return "_".join(p[:-3]), p[-3], p[-2], p[-1]  # task, cell, model, seedstr

def dir_max_mtime(d):
    m = 0.0
    for f in d.rglob("*"):
        if f.is_file():
            t = f.stat().st_mtime
            if t > m:
                m = t
    return m

if STAGE.exists():
    shutil.rmtree(STAGE)
OUT.mkdir(parents=True)

seen = {}            # run_name -> host (the copy we kept)
invalid = []         # (run_name, host, reason)
deferred = []        # (run_name, host) valid but unstable -> next snapshot
manifest = []        # dict rows
dup_valid = []       # (run_name, host) extra valid copies skipped
coverage = defaultdict(lambda: [0,0,0,0,0.0])  # (task,cell,model)->[n,cheat,honest,other,$]

for rd in sorted(DATA.glob("*/runs/*")):
    if not rd.is_dir():
        continue
    name = rd.name
    if name.startswith("_"):
        continue
    key = parse(name)
    if key is None:
        continue
    task, cell, model, seedstr = key
    host = rd.parts[-3]
    ep = rd / "episodes.jsonl"
    try:
        lines = [l for l in ep.read_text().splitlines() if l.strip()]
        e0 = json.loads(lines[0])
    except Exception as ex:
        invalid.append((name, host, f"episodes unreadable: {type(ex).__name__}")); continue
    if not e0.get("ok"):
        invalid.append((name, host, f"ok={e0.get('ok')} ended_by={e0.get('ended_by')}")); continue
    traj = rd / "trajectories"
    tbytes = sum(f.stat().st_size for f in traj.rglob("*") if f.is_file()) if traj.exists() else 0
    if tbytes == 0:
        invalid.append((name, host, "empty trajectories")); continue
    label = None
    sg = rd / "signals.jsonl"
    if sg.exists():
        for ln in sg.read_text().splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            if o.get("row") == "team":
                label = "cheat" if (o.get("used") or o.get("tampered")) else ("honest" if o.get("shipped") else "other")
                rung = o.get("rung")
                break
    if label is None:
        invalid.append((name, host, "no team-row label")); continue
    # dedup: keep first valid copy
    if name in seen:
        dup_valid.append((name, host)); continue

    # locate matched eval on this host (exact basename from run.json eval_log)
    eval_src = None
    run_json = rd / "run.json"
    try:
        rj = json.loads(run_json.read_text())
    except Exception:
        rj = {}
    evname = os.path.basename(rj.get("eval_log") or "")
    logdir = DATA / host / "logs" / name
    if evname and (logdir / evname).exists():
        eval_src = logdir / evname
    elif logdir.exists():
        evs = sorted(logdir.glob("*.eval"))
        eval_src = evs[0] if evs else None

    # stability: all files (run dir + eval) must be older than STABLE_AGE
    mtime = dir_max_mtime(rd)
    if eval_src is not None:
        mtime = max(mtime, eval_src.stat().st_mtime)
    if (NOW - mtime) <= STABLE_AGE:
        deferred.append((name, host)); continue

    seen[name] = host
    seed = int(seedstr[1:])
    billed = (rj.get("ledger") or {}).get("billed_usd")
    if billed is None:
        billed = (e0.get("ledger") or {}).get("billed_usd") or 0.0
    n_turns = sum(json.loads(l).get("n_turns") or 0 for l in lines)
    n_tool = sum(json.loads(l).get("n_tool_calls") or 0 for l in lines)
    score = e0.get("score")
    model_id = e0.get("model")

    # copy into layout
    dst = OUT / model / task / cell / seedstr
    dst.mkdir(parents=True, exist_ok=True)
    for f in rd.rglob("*"):
        if f.is_file():
            rel = f.relative_to(rd)
            tgt = dst / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, tgt)
    if eval_src is not None:
        ed = dst / "eval"
        ed.mkdir(exist_ok=True)
        shutil.copy2(eval_src, ed / eval_src.name)

    manifest.append({
        "run_name": name, "model": model, "model_id": model_id,
        "task": task, "cell": cell, "seed": seed, "host": host,
        "label": label, "rung": rung, "billed_usd": round(float(billed), 6),
        "n_turns": n_turns, "n_tool_calls": n_tool,
        "score": score, "n_episodes": len(lines),
        "has_eval": eval_src is not None, "valid": True,
    })
    c = coverage[(task, cell, model)]
    c[0]+=1; c[1]+= (label=="cheat"); c[2]+=(label=="honest"); c[3]+=(label=="other"); c[4]+=float(billed)

# write manifest (sorted for determinism)
manifest.sort(key=lambda r: (r["model"], r["task"], r["cell"], r["seed"]))
with open(STAGE / "manifest.jsonl", "w") as fh:
    for r in manifest:
        fh.write(json.dumps(r) + "\n")

zero_cells = [(t,c,m) for t in TASKS for c in CELLS for m in MODELS if coverage[(t,c,m)][0]==0]
stats = {
    "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "n_published": len(manifest),
    "n_invalid_dirs": len(invalid),
    "n_dup_valid_skipped": len(dup_valid),
    "n_deferred_unstable": len(deferred),
    "n_with_eval": sum(1 for r in manifest if r["has_eval"]),
    "n_cheat": sum(1 for r in manifest if r["label"]=="cheat"),
    "n_honest": sum(1 for r in manifest if r["label"]=="honest"),
    "n_other": sum(1 for r in manifest if r["label"]=="other"),
    "billed_usd_total": round(sum(r["billed_usd"] for r in manifest), 2),
    "models": MODELS, "tasks": TASKS, "cells": CELLS,
    "full_matrix": len(TASKS)*len(CELLS)*len(MODELS)*50,
    "per_model": {},
    "zero_cells": [list(z) for z in zero_cells],
    "deferred_sample": deferred[:20],
    "invalid_sample": [(n,h,r) for (n,h,r) in invalid[:20]],
}
# per-model totals (compute plainly)
pm = {}
for m in MODELS:
    n=cc=hh=oo=0; bb=0.0
    for (t,c,mm),v in coverage.items():
        if mm==m:
            n+=v[0]; cc+=v[1]; hh+=v[2]; oo+=v[3]; bb+=v[4]
    pm[m]={"n":n,"cheat":cc,"honest":hh,"other":oo,"billed_usd":round(bb,3)}
stats["per_model"]=pm
# per-cell coverage grid (n only) for README
grid={}
for t in TASKS:
    for c in CELLS:
        for m in MODELS:
            grid[f"{t}/{c}/{m}"]=coverage[(t,c,m)][0]
stats["coverage_grid"]=grid
with open(STAGE / "_build_stats.json", "w") as fh:
    json.dump(stats, fh, indent=1)

print(json.dumps({k:v for k,v in stats.items() if k not in ("coverage_grid","deferred_sample","invalid_sample")}, indent=1))
print("\nZERO CELLS:", zero_cells)
print("DEFERRED (unstable, next snapshot):", deferred)
print("INVALID count:", len(invalid), " DUP valid skipped:", len(dup_valid))
