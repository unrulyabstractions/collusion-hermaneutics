"""Independent Hub check of the extras: every planned agent has acts.safetensors, spans.jsonl, meta.json under
activations/ and activations_extra/; nothing else; labels files unchanged (blob ids vs a saved snapshot)."""
import csv, json, os, sys, collections
from huggingface_hub import HfApi
api = HfApi(token=os.environ["PERSONAL_HF_TOKEN"])
REPO = "unrulyabstractions/arena-capstone"
STUDY = {"study_exploration_1:incomplete": "swarm_exploration_1_incomplete",
         "study_exploration_1:swe_before_fix": "swarm_exploration_1_swe_before_fix",
         "study_exploration_1:regulator_refused": "swarm_exploration_1_regulator_refused"}
summary, snap, mode = sys.argv[1], sys.argv[2], sys.argv[3]   # mode: snapshot | check
want = {(STUDY[r["study"]], r["run"], r["agent"]) for r in csv.DictReader(open(summary))}
files = {}
for s in sorted(set(STUDY.values())):
    for root in ("activations", "activations_extra"):
        try:
            for i in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"{root}/{s}", recursive=True, expand=True):
                if hasattr(i, "size"):
                    files[i.path] = (i.size, i.lfs.sha256 if i.lfs else i.blob_id)
        except Exception as e:
            print(f"{root}/{s}: {type(e).__name__}")
labels = {p: v for p, v in files.items() if p.endswith(("agents.csv", "turns.csv"))}
if mode == "snapshot":
    json.dump(labels, open(snap, "w"), indent=1)
    print(len(labels), "label files saved;", len(files) - len(labels), "other files on the Hub now")
    sys.exit(0)
old = json.load(open(snap))
bad = []
for p, v in old.items():
    if p not in labels or list(labels[p]) != list(v):
        bad.append(f"label file changed or missing: {p}")
done = collections.Counter()
for s, r, a in sorted(want):
    for root in ("activations", "activations_extra"):
        for f in ("acts.safetensors", "spans.jsonl", "meta.json"):
            p = f"{root}/{s}/{r}/{a}/{f}"
            if p in files and files[p][0] > 0:
                done[root] += 1
            else:
                bad.append(f"missing: {p}")
expected = {f"{root}/{s}/{r}/{a}/{f}" for s, r, a in want for root in ("activations", "activations_extra")
            for f in ("acts.safetensors", "spans.jsonl", "meta.json")} | set(old)
extra = sorted(set(files) - expected)
print(f"{len(want)} agents planned; files present: {dict(done)} (expected {3 * len(want)} per root); "
      f"label files {len(old)} unchanged: {not any(b.startswith('label') for b in bad)}; unexpected files {len(extra)}")
for b in bad[:10] + extra[:10]:
    print("  ", b)
sys.exit(1 if bad or extra else 0)
