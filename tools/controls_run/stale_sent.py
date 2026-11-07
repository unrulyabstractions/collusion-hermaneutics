"""List activation-map sentence sets the current code never reads: the file's own agent list, rehashed with each
standard agent set, layer and scope, gives its fingerprint back (so we know which view it was built for), but that
view's current agent list differs, so the current code asks for another fingerprint. Files no view explains are kept."""
import glob, json, os, sys, zipfile
sys.path.insert(0, "script")
import numpy as np
import visualize_in_persona_space as p
import embed_acts as ea
import scopes
data = "data/hf"
st = p.Store(data, "qwen3.8-27b", False)
scan = st.scan()
layers = sorted({L for e in scan.values() for L in e["files"]})
d = os.path.join(data, ".viewer_cache", "activations", "embed")
cur = {}
for agents in ea.AGENT_SETS:
    for L in layers:
        es = ea._set_agents(scan, L, agents, data)
        for sc in list(scopes.ORDER) + ["own"]:
            cur[ea._sent_fp(es, L, sc, agents)] = (agents, L, sc, [e["key"] for e in es])
stale, kept, unknown = [], [], []
for f in sorted(glob.glob(os.path.join(d, "sent_*_agent.npz")) + glob.glob(os.path.join(d, "sent_*_global.npz"))):
    fp = os.path.basename(f).split("_")[1]
    if fp in cur:
        kept.append(f); continue
    try:
        with zipfile.ZipFile(f) as z:
            meta = json.loads(str(np.load(z.open("meta.npy"), allow_pickle=False)))
    except Exception as ex:
        unknown.append((f, f"unreadable {type(ex).__name__}")); continue
    keys = meta["keys"]
    if not all(k in scan for k in keys):
        stale.append((f, "holds agents that no longer exist")); continue
    es = [scan[k] for k in keys]
    hit = None
    for agents in ea.AGENT_SETS:
        for L in layers:
            if not all(L in e["files"] for e in es):
                continue
            for sc in list(scopes.ORDER) + ["own"]:
                if ea._sent_fp(es, L, sc, agents) == fp:
                    hit = (agents, L, sc)
    if hit:
        now = [e["key"] for e in ea._set_agents(scan, hit[1], hit[0], data)]
        stale.append((f, f"built for {hit} with {len(keys)} agents; that view now has {len(now)}") if now != keys else None) if now != keys else unknown.append((f, "matches current list?"))
    else:
        unknown.append((f, "no standard view explains it (a chip-filtered set, or skipped agents): kept"))
stale = [s for s in stale if s]
tot = 0
for f, why in stale:
    fp = os.path.basename(f).split("_")[1]
    fits = glob.glob(os.path.join(d, f"sentfit_{fp}_*"))
    sz = os.path.getsize(f) + sum(os.path.getsize(x) for x in fits)
    tot += sz
    print(f"STALE {sz/1e9:6.2f} GB {os.path.basename(f)} (+{len(fits)} fits): {why}")
print(f"current {len(kept)}, stale {len(stale)} ({tot/1e9:.1f} GB with their fits), kept as unknown {len(unknown)}")
for f, why in unknown[:8]:
    print("UNKNOWN", os.path.basename(f), why)
json.dump([f for f, _ in stale], open(sys.argv[1], "w"))
