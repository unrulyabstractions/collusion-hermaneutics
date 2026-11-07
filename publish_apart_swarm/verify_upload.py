#!/usr/bin/env python3
"""Verify the HF dataset upload against the local staging + original source data.

Checks (prints PASS/FAIL per check; exit 1 if any FAIL):
 1. Repo file count == local staged file count + 1 (the repo-generated .gitattributes).
 2. Per-path size diff: every staged file exists in the repo with identical byte size;
    the only repo-extra file is .gitattributes; no local file is missing from the repo.
 3. Sample sha256: download a random sample spanning every model+task and sha256-compare
    each against the ORIGINAL source file under data/apart_controls (not staging).
 4. manifest.jsonl + README.md round-trip identical to staging.

Auth: reads HF_TOKEN from env (never printed). Requires ~/.local/bin/hf on PATH for downloads.
Usage: python3 -I verify_upload.py [STAGE_DIR]
"""
import json, os, re, sys, hashlib, tempfile, subprocess, urllib.request
from pathlib import Path

REPO = "unrulyabstractions/apart-swarm"
DATA = Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
STAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA.parent / ".hf_staging_apart_swarm"
TOK = os.environ["HF_TOKEN"]
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")

def repo_tree():
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true&expand=true"
    sizes = {}
    while url:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOK}"})
        with urllib.request.urlopen(req) as r:
            data = json.load(r); link = r.headers.get("Link", "")
        for x in data:
            if x.get("type") == "file":
                s = x.get("size")
                if s is None:
                    s = (x.get("lfs") or {}).get("size")
                sizes[x["path"]] = s
        m = re.search(r'<([^>]+)>;\s*rel="next"', link); url = m.group(1) if m else None
    return sizes

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def local_sizes():
    out = {}
    for p in STAGE.rglob("*"):
        if p.is_file() and ".cache" not in p.parts:
            out[str(p.relative_to(STAGE))] = p.stat().st_size
    return out

fails = []
repo = repo_tree()
local = local_sizes()

# check 1: counts
n_repo, n_local = len(repo), len(local)
ok1 = (n_repo == n_local + 1) and (".gitattributes" in repo)
print(f"[{'PASS' if ok1 else 'FAIL'}] count: repo={n_repo} local={n_local} (expect repo==local+1 for .gitattributes)")
if not ok1: fails.append("count")

# check 2: per-path size diff
only_repo = set(repo) - set(local)
only_local = set(local) - set(repo)
mism = [(k, local[k], repo[k]) for k in (set(repo) & set(local)) if local[k] != repo[k]]
ok2 = (only_repo == {".gitattributes"}) and (not only_local) and (not mism)
print(f"[{'PASS' if ok2 else 'FAIL'}] size-diff: only_repo={sorted(only_repo)} only_local={len(only_local)} mismatches={len(mism)}")
for k, l, r in mism[:10]: print("      mismatch", k, "local", l, "repo", r)
if not ok2: fails.append("size-diff")

# check 3: sample sha256 vs source
rows = [json.loads(l) for l in (STAGE / "manifest.jsonl").read_text().splitlines()]
import random; random.seed(7)
by = {}
for r in rows:
    by.setdefault(("m", r["model"]), r); by.setdefault(("t", r["task"]), r)
sample = list({r.get("key", r["run_name"]): r for r in list(by.values()) + random.sample(rows, min(10, len(rows)))}.values())
dl = Path(tempfile.mkdtemp(prefix="hfverify_"))
checked, bad = 0, []
for r in sample:
    if r.get("canonical", True):
        base = f"apart_controls/{r['model']}/{r['task']}/{r['cell']}/s{r['seed']}"
    else:
        base = f"extra_duplicate_samples/{r['model']}/{r['task']}/{r['cell']}/s{r['seed']}__{r['host']}"
    srcdir = DATA / r["host"] / "runs" / r["run_name"]
    rels = ["run.json", "signals.jsonl", "episodes.jsonl"]
    tr = list((srcdir / "trajectories").rglob("transcript.json"))
    if tr: rels.append(str(tr[0].relative_to(srcdir)))
    for rel in rels:
        rp = f"{base}/{rel}"
        rc = subprocess.run(["hf", "download", REPO, rp, "--type", "dataset",
                             "--local-dir", str(dl), "--quiet"], capture_output=True, text=True)
        got = dl / rp
        if rc.returncode != 0 or not got.exists():
            bad.append((rp, "download failed")); continue
        if sha(got) != sha(srcdir / rel):
            bad.append((rp, "sha mismatch vs source")); continue
        checked += 1
ok3 = not bad
print(f"[{'PASS' if ok3 else 'FAIL'}] sample sha256 vs source: checked={checked} problems={len(bad)}")
for b in bad[:10]: print("      ", b)
if not ok3: fails.append("sha-sample")

# check 4: manifest + README round-trip
ok4 = True
for top in ["manifest.jsonl", "README.md"]:
    subprocess.run(["hf", "download", REPO, top, "--type", "dataset", "--local-dir", str(dl), "--quiet"],
                   capture_output=True, text=True)
    if not (dl / top).exists() or sha(dl / top) != sha(STAGE / top):
        ok4 = False; print(f"      {top} mismatch/missing")
print(f"[{'PASS' if ok4 else 'FAIL'}] manifest+README round-trip")
if not ok4: fails.append("roundtrip")

import shutil; shutil.rmtree(dl, ignore_errors=True)
print("\nRESULT:", "VERIFIED" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
