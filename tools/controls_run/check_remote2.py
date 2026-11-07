import os, json, random, hashlib
import numpy as np
from safetensors.numpy import load_file
root = "/root/work_hermeneutics/acts"
runs = sorted(os.listdir(root))
random.seed(20261107); pick = random.sample(runs, 12)
random.seed(7); pick += random.sample([r for r in runs if "_luna_" in r], 3)
for r in pick:
    for a in os.listdir(os.path.join(root, r)):
        d = os.path.join(root, r, a)
        t = load_file(os.path.join(d, "acts.safetensors")); meta = json.load(open(os.path.join(d, "meta.json")))
        nsp = sum(1 for _ in open(os.path.join(d, "spans.jsonl")))
        tp = t["think_pos"]; n = tp.shape[0]; ids = t["ids"]
        probs = []
        for L in (26,38,51):
            x = t[f"think_L{L}"]
            if x.shape != (n,2,5120) or x.dtype != np.float16: probs.append(f"L{L} {x.shape} {x.dtype}")
            xf = x.astype(np.float32)
            if not np.isfinite(xf).all() or np.abs(xf).max()==0: probs.append(f"L{L} finite/zero")
        if not (ids[tp[:,0]]==248068).all() or not (ids[tp[:,1]]==248069).all(): probs.append("ids")
        if meta["turns"] != n or meta["n_turns"] != n: probs.append(f"meta turns {meta['turns']} {meta['n_turns']}")
        if nsp != meta["n_units"]: probs.append(f"spans {nsp} != n_units {meta['n_units']}")
        if len(ids) != meta["n_tokens"]: probs.append(f"ids {len(ids)} != n_tokens {meta['n_tokens']}")
        print(f"{r}/{a}: author={meta['author']} cond={meta['condition']} n_turns={n} n_units={meta['n_units']} spans={nsp} n_tokens={meta['n_tokens']} ids={len(ids)} readable={int(t['think_readable'].sum()) if 'think_readable' in t else None}/{meta['n_readable']} PROBLEMS={probs or 'none'}")
print("MD5")
for f in ["pca_agent/auroc.csv","pca_agent/author_probe.csv","pca_agent/condition_probe.csv","pca_agent/agents.csv","pca_turn/auroc.csv","pca_turn_onset/auroc.csv","full_agent/auroc.csv","pca_agent/solo/Lall_both_agent/results.json","pca_agent/pooled/L38_close_agent/results.json","pca_agent/cheat/L38_close_agent/results.json"]:
    p = "/root/work_hermeneutics/results/"+f
    print(f, hashlib.md5(open(p,"rb").read()).hexdigest() if os.path.exists(p) else "MISSING")
