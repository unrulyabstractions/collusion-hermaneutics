#!/usr/bin/env bash
# Verify the HF upload: count repo files, then download a random sample and
# sha256-compare against the ORIGINAL source data (data/apart_controls), not staging.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCRATCH="/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad"
REPO="unrulyabstractions/apart-swarm"
cd "$SCRATCH"

echo "=== 1. repo file listing (dry-run, no bytes) ==="
hf download "$REPO" --type dataset --dry-run --quiet 2>/dev/null > repo_files.txt
# dry-run prints lines; keep only path-like lines
grep -E '\.(json|jsonl|txt|eval|md|gitattributes)$' repo_files.txt | sort -u > repo_files_clean.txt
echo "repo files listed (filtered): $(wc -l < repo_files_clean.txt)"
echo "local staged files: $(find hf_staging -type f ! -path '*/.cache/*' | wc -l | tr -d ' ')  (+ pre-existing .gitattributes in repo)"
echo "--- sample of repo listing ---"; head -5 repo_files_clean.txt; echo "..."; tail -3 repo_files_clean.txt

echo
echo "=== 2. sample-download + sha256 vs ORIGINAL source data ==="
python3 -I - <<'PY'
import json, random, subprocess, hashlib, os, tempfile
from pathlib import Path
SCRATCH=Path("/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad")
DATA=Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
REPO="unrulyabstractions/apart-swarm"
rows=[json.loads(l) for l in (SCRATCH/"hf_staging/manifest.jsonl").read_text().splitlines()]
random.seed(7)
# ensure every model+task covered, plus random fill
by={}
for r in rows:
    by.setdefault(("m",r["model"]),r); by.setdefault(("t",r["task"]),r)
sample={r["run_name"]:r for r in list(by.values())+random.sample(rows,10)}.values()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
dl=Path(tempfile.mkdtemp(prefix="hfverify_"))
bad=[]; checked=0
for r in sample:
    base=f"apart_controls/{r['model']}/{r['task']}/{r['cell']}/s{r['seed']}"
    # verify run.json, signals.jsonl, and one transcript against SOURCE data
    srcdir=DATA/r["host"]/"runs"/r["run_name"]
    rels=["run.json","signals.jsonl","episodes.jsonl"]
    tr=list((srcdir/"trajectories").rglob("transcript.json"))
    if tr: rels.append(str(tr[0].relative_to(srcdir)))
    for rel in rels:
        repopath=f"{base}/{rel}"
        rc=subprocess.run(["hf","download",REPO,repopath,"--type","dataset",
                           "--local-dir",str(dl),"--quiet"],capture_output=True,text=True)
        got=dl/repopath
        if rc.returncode!=0 or not got.exists():
            bad.append((repopath,"download failed",rc.stderr[:120])); continue
        if sha(got)!=sha(srcdir/rel):
            bad.append((repopath,"sha mismatch vs source","")); continue
        checked+=1
# manifest + README round-trip vs staging
for top in ["manifest.jsonl","README.md"]:
    rc=subprocess.run(["hf","download",REPO,top,"--type","dataset","--local-dir",str(dl),"--quiet"],
                      capture_output=True,text=True)
    if (dl/top).exists() and sha(dl/top)==sha(SCRATCH/"hf_staging"/top): checked+=1
    else: bad.append((top,"mismatch or missing vs staging",rc.stderr[:120]))
print(f"files sha256-checked against source/staging: {checked}")
print(f"PROBLEMS: {len(bad)}")
for b in bad: print("   ",b)
import shutil; shutil.rmtree(dl,ignore_errors=True)
PY
