#!/usr/bin/env python3
"""ONE-SHOT full sync (mirrors sync_data.py's rsync exactly, but runs once and exits).
rsync -az --partial each host's runs/logs/out/<STUDY>/ -> data/apart_controls/<host>/<sub>/.
Exit 1 if any host/subdir failed, so a caller can retry before sweeping."""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
CFG = Path(__file__).resolve().parent / "orch_config.json"
DEFAULT_REPO = "/root/arena-capstone/submodule/spar-guardian-agents"
STUDY = "study_apart_controls"
DATA = Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
def sync_host(item):
    h, hcfg = item
    toks = hcfg.get("ssh", f"arena9-{h}").split(); target, opts = toks[-1], toks[:-1]
    repo = hcfg.get("repo", DEFAULT_REPO)
    ssh_cmd = "ssh -o ConnectTimeout=15 -o BatchMode=yes " + " ".join(opts)
    fails = []
    for sub in ("runs", "logs", "out"):
        dest = DATA / h / sub; dest.mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(["rsync", "-az", "--partial", "--timeout=120", "-e", ssh_cmd,
                                f"{target}:{repo}/{sub}/{STUDY}/", str(dest) + "/"],
                               capture_output=True, timeout=900)
            if r.returncode != 0:
                fails.append(f"{sub}:rc={r.returncode}:{r.stderr.decode(errors='replace').strip()[-120:]}")
        except Exception as e:
            fails.append(f"{sub}:{type(e).__name__}")
    return h, fails
hosts = json.loads(CFG.read_text())["hosts"]
t0 = time.time()
with ThreadPoolExecutor(max_workers=min(24, len(hosts))) as ex:
    res = list(ex.map(sync_host, hosts.items()))
bad = {h: f for h, f in res if f}
n = sum(1 for _ in DATA.glob("*/runs/*/episodes.jsonl"))
print(f"[{time.strftime('%H:%M:%S')}] one-shot sync: {len(hosts)} hosts in {time.time()-t0:.0f}s; local episodes={n}; failed_hosts={len(bad)}")
for h, f in bad.items(): print(f"  FAIL {h}: {f}")
print("FORCE_SYNC_" + ("OK" if not bad else "FAILED"))
sys.exit(1 if bad else 0)
