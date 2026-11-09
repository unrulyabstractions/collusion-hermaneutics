"""Balanced extraction rounds over the 13 Goodfire runs: round r holds, for every run, the next K rollouts of each verdict
class (yes, no) in sha1(rollout_id) order, skipping rollouts already extracted; K = 6 (2 for DeepSWE, ~150k tokens each).
With n pods, pod i takes the items whose sha1 % n == i. Every run gets an ids file each round (empty when nothing is
left), because goodfire_resident.py extracts ALL rollouts of a run that has no ids file.
    python rounds.py N_SHARDS [N_ROUNDS]   -> /workspace/rounds/s<i>of<n>/r<NN>/<run_dir>.ids
"""
import csv, glob, hashlib, os, shutil, sys
sys.path.insert(0, "/root/code/ch")
from hermeneutics.goodfire import label_rows
n_shards = int(sys.argv[1]); n_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
h = lambda s: int(hashlib.sha1(s.encode()).hexdigest(), 16)
cells = {}
for p in sorted(glob.glob("/workspace/plan_v3/*.csv")):
    rd = os.path.basename(p)[:-4]
    lab = label_rows(f"/workspace/data/goodfire/{rd}")
    for r in csv.DictReader(open(p)):
        if r["included"] != "True" or os.path.exists(f"/workspace/acts/{rd}/{r['rollout_id']}/meta.json"):
            continue
        v = (lab.get(r["rollout_id"]) or {}).get("verdict")
        if v in ("yes", "no"):
            cells.setdefault((rd, v), []).append(r["rollout_id"])
for k in cells:
    cells[k].sort(key=h)
root = f"/workspace/rounds/s{{}}of{n_shards}"
for i in range(n_shards):
    shutil.rmtree(root.format(i), ignore_errors=True)
runs = sorted({rd for rd, _ in cells} | {os.path.basename(p)[:-4] for p in glob.glob("/workspace/plan_v3/*.csv")})
cursor = {k: 0 for k in cells}
for r in range(n_rounds):
    per = {i: {rd: [] for rd in runs} for i in range(n_shards)}
    for (rd, v), ids in cells.items():
        k = 2 if rd.startswith("deepswe") else 6
        take = ids[cursor[(rd, v)]:cursor[(rd, v)] + k]
        cursor[(rd, v)] += k
        for rid in take:
            per[h(rid) % n_shards][rd].append(rid)
    for i in range(n_shards):
        d = os.path.join(root.format(i), f"r{r:02d}")
        os.makedirs(d, exist_ok=True)
        for rd, ids in per[i].items():
            open(os.path.join(d, f"{rd}.ids"), "w").write("".join(x + "\n" for x in ids))
left = {k: len(v) for k, v in cells.items()}
print(f"{n_rounds} rounds x {n_shards} shards written; pending per cell before round 0:",
      ", ".join(f"{rd.split('-')[0]}/{rd.split('-')[1]}/{v}={n}" for (rd, v), n in sorted(left.items())))
