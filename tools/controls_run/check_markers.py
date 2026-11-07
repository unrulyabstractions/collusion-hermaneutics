import glob, json, collections, sys
SP = sys.argv[1]
tot = collections.Counter(); one = collections.Counter(); lab = collections.Counter(); ncalls_ok = ncalls_bad = 0
bad_calls = []
for st in sorted(glob.glob(f"{SP}/swarm_runs/*/*/steps.jsonl")):
    rd = st.rsplit("/", 1)[0]
    rows = [json.loads(l) for l in open(st)]
    steps = [r for r in rows if r["row"] == "step"]
    sig = [json.loads(l) for l in open(rd + "/signals.jsonl") if l.strip()]
    for s in sig:
        if s.get("row") != "agent": continue
        mine = [r for r in steps if r["agent"] == s["agent"]]
        n = sum(len(r["calls"]) for r in mine)
        if s.get("calls") is not None:
            if n == s["calls"]: ncalls_ok += 1
            else: ncalls_bad += 1; bad_calls.append((rd.split("/")[-1], s["agent"], n, s["calls"]))
        for m, need in (("exposed_call", None), ("tried_call", "aimed"), ("opened_call", "shown"), ("joined_call", None)):
            at = s.get(m)
            if at is None: continue
            tot[m] += 1
            hit = [r for r in mine if at in r["calls"]]
            one[m] += len(hit) == 1
            if need and len(hit) == 1:
                ok = any(i["call"] == at for i in hit[0][need]) or (m == "opened_call" and any(i["call"] == at for i in hit[0]["aimed"]))
                lab[m] += ok
for m in tot: print(m, "markers", tot[m], "in exactly one step", one[m], "carrying label", lab[m] if m in ("tried_call","opened_call") else "-")
print("agents whose tool-call count equals signals calls", ncalls_ok, "differs", ncalls_bad, bad_calls[:10])
