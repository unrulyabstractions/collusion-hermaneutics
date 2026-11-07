import json, glob, os
base = "/root/arena-capstone/submodule/spar-guardian-agents/runs/study_apart_controls"
for d in sorted(glob.glob(base + "/*")):
    name = os.path.basename(d)
    ep = os.path.join(d, "episodes.jsonl")
    if not os.path.exists(ep):
        continue
    try:
        e = json.loads(open(ep).readline())
    except Exception as ex:
        print(json.dumps({"name": name, "ERR": str(ex)})); continue
    trajs = glob.glob(os.path.join(d, "trajectories", "**", "*"), recursive=True)
    tfiles = [f for f in trajs if os.path.isfile(f)]
    tbytes = sum(os.path.getsize(f) for f in tfiles)
    label = None
    sg = os.path.join(d, "signals.jsonl")
    if os.path.exists(sg):
        for ln in open(sg):
            o = json.loads(ln)
            if o.get("row") == "team":
                label = "cheat" if (o.get("used") or o.get("tampered")) else ("honest" if o.get("shipped") else "other")
                break
    print(json.dumps({"name": name, "turns": e.get("n_turns"), "tool_calls": e.get("n_tool_calls"),
        "final_len": len(e.get("final_text") or ""), "ok": e.get("ok"), "ended": e.get("ended_by"),
        "err": bool(e.get("error")), "traj_files": len(tfiles), "traj_bytes": tbytes, "label": label,
        "billed": round((e.get("ledger") or {}).get("billed_usd") or 0, 4)}))
