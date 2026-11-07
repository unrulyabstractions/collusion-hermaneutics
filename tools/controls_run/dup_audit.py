import json
from collections import defaultdict
from pathlib import Path
DATA=Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
MODELS=("qwen","haiku","gpt5m","ds","luna"); TASKS=("flipt","vuls","riemann","subset","factor","binder","poem","worktime","game"); CELLS=("solo","cheat")
def parse(n):
    p=n.split("_")
    if len(p)<4 or not p[-1].startswith("s") or p[-2] not in MODELS or p[-3] not in CELLS: return None
    return 1
def inspect(rd):
    """return dict describing a run-dir copy: valid?, ok, ended_by, tbytes, label, rung, score, ntool"""
    d={"host":rd.parts[-3],"ok":None,"ended_by":None,"tbytes":0,"label":None,"rung":None,"score":None,"ntool":None,"valid":False,"reason":""}
    ep=rd/"episodes.jsonl"
    try:
        lines=[l for l in ep.read_text().splitlines() if l.strip()]; e0=json.loads(lines[0])
    except Exception as ex:
        d["reason"]=f"episodes unreadable {type(ex).__name__}"; return d
    d["ok"]=e0.get("ok"); d["ended_by"]=e0.get("ended_by"); d["score"]=e0.get("score"); d["ntool"]=e0.get("n_tool_calls")
    traj=rd/"trajectories"; d["tbytes"]=sum(f.stat().st_size for f in traj.rglob("*") if f.is_file()) if traj.exists() else 0
    sg=rd/"signals.jsonl"
    if sg.exists():
        for ln in sg.read_text().splitlines():
            if not ln.strip(): continue
            o=json.loads(ln)
            if o.get("row")=="team":
                d["label"]="cheat" if (o.get("used") or o.get("tampered")) else ("honest" if o.get("shipped") else "other")
                d["rung"]=o.get("rung"); break
    if not d["ok"]: d["reason"]="ok=False"; return d
    if d["tbytes"]==0: d["reason"]="empty trajectories"; return d
    if d["label"] is None: d["reason"]="no team label"; return d
    d["valid"]=True; return d

copies=defaultdict(list)
for rd in sorted(DATA.glob("*/runs/*")):
    if not rd.is_dir() or rd.name.startswith("_") or parse(rd.name) is None: continue
    copies[rd.name].append(rd)

run_names=sorted(copies)
dups={n:v for n,v in copies.items() if len(v)>1}
print(f"total run_names on disk: {len(run_names)}")
print(f"run_names with >1 copy: {len(dups)}")

kept_invalid=[]; divergent=[]; all_invalid=[]; dedup_kept={}
both_valid=0
for n,rds in dups.items():
    infos=[inspect(rd) for rd in sorted(rds, key=lambda p:str(p))]  # host-alpha sort == dedup order
    valids=[i for i in infos if i["valid"]]
    kept = next((i for i in infos if i["valid"]), None)  # first valid = what dedup keeps
    if kept is None:
        all_invalid.append((n,[(i["host"],i["reason"]) for i in infos])); continue
    dedup_kept[n]=kept["host"]
    if not kept["valid"]: kept_invalid.append(n)
    if len(valids)>1:
        both_valid+=1
        outs={(i["label"],i["rung"],json.dumps(i["score"],sort_keys=True)) for i in valids}
        if len(outs)>1:
            divergent.append((n,[(i["host"],i["label"],i["rung"],i["score"],i["ntool"]) for i in valids]))

print(f"\n--- duplicated run_name verdict ---")
print(f"kept copy is INVALID (should be 0): {len(kept_invalid)} {kept_invalid}")
print(f"run_names where ALL copies invalid (not published; gap if no valid anywhere): {len(all_invalid)}")
for n,info in all_invalid: print("   ALL-INVALID", n, info)
print(f"run_names with >1 VALID copy: {both_valid}")
print(f"  of those, DIVERGENT outcomes (different label/rung/score): {len(divergent)}")
for n,info in divergent: print("   DIVERGENT", n, info)

# show a detailed table for all duplicated run_names
print("\n--- per-duplicate detail (host:valid/ok/ended_by/tbytes/label/rung) ---")
for n in sorted(dups):
    infos=[inspect(rd) for rd in sorted(dups[n], key=lambda p:str(p))]
    kepthost=dedup_kept.get(n)
    cells=" | ".join(f"{i['host']}:{'V' if i['valid'] else 'X'}/ok={i['ok']}/{i['ended_by']}/{i['tbytes']}b/{i['label']}/{i['rung']}"+("<=KEEP" if i['host']==kepthost else "") for i in infos)
    print(f"  {n}: {cells}")
