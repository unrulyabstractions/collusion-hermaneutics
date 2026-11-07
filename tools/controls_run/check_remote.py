import os, json, random, sys
import numpy as np
from safetensors.numpy import load_file
root = "/root/work_hermeneutics/acts"
runs = sorted(os.listdir(root))
print("RUN_FOLDERS", len(runs))
complete = 0; missing = []
for r in runs:
    ags = os.listdir(os.path.join(root, r))
    for a in ags:
        d = os.path.join(root, r, a)
        need = ["acts.safetensors","meta.json","spans.jsonl"]
        have = [os.path.exists(os.path.join(d,f)) for f in need]
        if all(have): complete += 1
        else: missing.append((r,a,have))
print("AGENT_FOLDERS_COMPLETE", complete, "MISSING", missing[:10])
agents_per_run = {r: len(os.listdir(os.path.join(root,r))) for r in runs}
from collections import Counter
print("AGENTS_PER_RUN", Counter(agents_per_run.values()))
random.seed(20261107)
pick = random.sample(runs, 12)
print("PICKED", pick)
allok = True
for r in pick:
    for a in sorted(os.listdir(os.path.join(root, r))):
        d = os.path.join(root, r, a)
        t = load_file(os.path.join(d, "acts.safetensors"))
        meta = json.load(open(os.path.join(d, "meta.json")))
        nsp = sum(1 for _ in open(os.path.join(d, "spans.jsonl")))
        keys = sorted(t.keys())
        ids = t["ids"]; tp = t["think_pos"]
        n = tp.shape[0]
        probs = []
        for L in (26, 38, 51):
            x = t[f"think_L{L}"]
            if x.shape != (n, 2, 5120): probs.append(f"shape {L} {x.shape}")
            if x.dtype != np.float16: probs.append(f"dtype {L} {x.dtype}")
            xf = x.astype(np.float32)
            if not np.isfinite(xf).all(): probs.append(f"nonfinite {L}")
            if np.abs(xf).max() == 0: probs.append(f"allzero {L}")
            zero_rows = int((np.abs(xf).reshape(n*2, -1).max(1) == 0).sum())
            if zero_rows: probs.append(f"zero_rows L{L}={zero_rows}")
        o = ids[tp[:,0]]; c = ids[tp[:,1]]
        if not (o == 248068).all(): probs.append(f"open ids {np.unique(o)}")
        if not (c == 248069).all(): probs.append(f"close ids {np.unique(c)}")
        if meta.get("turns") != n: probs.append(f"meta turns {meta.get('turns')} != {n}")
        if not (tp[:,0] < tp[:,1]).all(): probs.append("open>=close")
        if nsp != n: probs.append(f"spans {nsp} != n {n}")
        stats = {L: (float(t[f"think_L{L}"].astype(np.float32).mean()), float(t[f"think_L{L}"].astype(np.float32).std()), float(np.abs(t[f"think_L{L}"].astype(np.float32)).max())) for L in (26,38,51)}
        print(f"{r}/{a}: keys={keys} n_turns={n} ids_len={len(ids)} meta_turns={meta.get('turns')} meta_keys={sorted(meta.keys())} spans={nsp} stats(mean,std,maxabs)={ {k:tuple(round(v,3) for v in s) for k,s in stats.items()} } PROBLEMS={probs or 'none'}")
        if probs: allok = False
print("ALL_12_OK", allok)
print("SAMPLE_META", json.dumps(meta)[:600])
