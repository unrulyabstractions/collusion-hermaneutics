import json, hashlib, random, zipfile
from pathlib import Path
DATA = Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
STAGE = Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad/hf_staging")
rows = [json.loads(l) for l in (STAGE/"manifest.jsonl").read_text().splitlines()]
# sample: cover every model and every task at least once, plus random fill to 30
random.seed(1)
by_model = {}; by_task = {}
for r in rows:
    by_model.setdefault(r["model"], r); by_task.setdefault(r["task"], r)
sample = list({r["run_name"]:r for r in list(by_model.values())+list(by_task.values())+random.sample(rows, 24)}.values())
def sha(p):
    h=hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()
ok=0; bad=[]
for r in sample:
    seeddir = STAGE/"apart_controls"/r["model"]/r["task"]/r["cell"]/f"s{r['seed']}"
    srcdir = DATA/r["host"]/"runs"/r["run_name"]
    # compare run.json, episodes.jsonl, signals.jsonl, one transcript, the eval
    checks=["run.json","episodes.jsonl","signals.jsonl"]
    for rel in checks:
        if sha(seeddir/rel)!=sha(srcdir/rel):
            bad.append((r["run_name"],rel,"sha mismatch")); continue
    # transcript parse + identity
    tr = list((seeddir/"trajectories").rglob("transcript.json"))
    if not tr: bad.append((r["run_name"],"transcript","missing")); 
    else:
        srctr = srcdir/tr[0].relative_to(seeddir)
        if sha(tr[0])!=sha(srctr): bad.append((r["run_name"],"transcript","sha mismatch"))
        try: json.loads(tr[0].read_text())
        except Exception as e: bad.append((r["run_name"],"transcript",f"unparseable {e}"))
    # parse episodes/signals/run.json
    for rel in checks:
        try:
            if rel.endswith(".jsonl"):
                for ln in (seeddir/rel).read_text().splitlines():
                    if ln.strip(): json.loads(ln)
            else: json.loads((seeddir/rel).read_text())
        except Exception as e: bad.append((r["run_name"],rel,f"unparseable {e}"))
    # eval zip + identity
    evs=list((seeddir/"eval").glob("*.eval"))
    if evs:
        src_ev = DATA/r["host"]/"logs"/r["run_name"]/evs[0].name
        if src_ev.exists() and sha(evs[0])!=sha(src_ev): bad.append((r["run_name"],"eval","sha mismatch"))
        if not zipfile.is_zipfile(evs[0]): bad.append((r["run_name"],"eval","not a zip"))
    ok+=1
print(f"sampled {len(sample)} runs across models={sorted(set(r['model'] for r in sample))} tasks={sorted(set(r['task'] for r in sample))} cells={sorted(set(r['cell'] for r in sample))}")
print(f"PROBLEMS: {len(bad)}")
for b in bad[:30]: print("  ", b)
