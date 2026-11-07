#!/usr/bin/env python3
"""Incremental transcript sync. Every INTERVAL s, rsync each configured host's
runs/<study>/ and logs/<study>/ into the Mac data dir (gitignored). Per-host ssh target and repo
path come from orch_config.json (ssh may be "arena9-apple" or "-F <ssh_config> box"; repo defaults
to the arena9 path). rsync is incremental + streaming. Per-host failures retry next cycle."""
import json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / "orch_config.json"
DEFAULT_REPO = "/root/arena-capstone/submodule/spar-guardian-agents"
STUDY = "study_apart_controls"
DATA = Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
INTERVAL = 600

def sync_host(item):
    h, hcfg = item
    toks = hcfg.get("ssh", f"arena9-{h}").split()
    target, opts = toks[-1], toks[:-1]
    repo = hcfg.get("repo", DEFAULT_REPO)
    ssh_cmd = "ssh -o ConnectTimeout=15 -o BatchMode=yes " + " ".join(opts)
    ok = True
    for sub in ("runs", "logs", "out"):
        dest = DATA / h / sub
        dest.mkdir(parents=True, exist_ok=True)
        try:
            r = subprocess.run(["rsync", "-az", "--partial", "--timeout=90", "-e", ssh_cmd,
                                f"{target}:{repo}/{sub}/{STUDY}/", str(dest) + "/"],
                               capture_output=True, timeout=300)
            if r.returncode != 0:
                ok = False
        except Exception:
            ok = False   # rsync/timeout: retry next cycle
    return h, ok

while True:
    try:
        hosts = json.loads(CFG.read_text())["hosts"]
    except Exception:
        hosts = {}
    if hosts:
        with ThreadPoolExecutor(max_workers=min(24, len(hosts))) as ex:
            res = list(ex.map(sync_host, hosts.items()))
        bad = [h for h, ok in res if not ok]
        n = sum(1 for _ in DATA.glob("*/runs/*/episodes.jsonl"))
        print(f"[{time.strftime('%H:%M:%S')}] synced {len(hosts)} hosts; local episodes={n}; failures={bad}", flush=True)
    time.sleep(INTERVAL)
