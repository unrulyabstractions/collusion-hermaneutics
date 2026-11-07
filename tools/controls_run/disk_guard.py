#!/usr/bin/env python3
"""Disk guard. Every cycle, on each configured host (per-host ssh target from orch_config.json), prune
stopped containers + dangling images, then read df. If < MIN_FREE_GB free or > MAX_PCT used, set its
conc_total=0 (auto-exclude). Only edits conc_total."""
import json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CFG = Path(__file__).resolve().parent / "orch_config.json"
MIN_FREE_GB = 6.0
MAX_PCT = 92

def check_host(item):
    h, hcfg = item
    toks = hcfg.get("ssh", f"arena9-{h}").split()
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", *toks,
                            "docker container prune -f >/dev/null 2>&1; docker image prune -f >/dev/null 2>&1; "
                            "df --output=avail,pcent / | tail -1"],
                           capture_output=True, timeout=60)
        out = r.stdout.decode("utf-8", "replace").split()
        return h, (int(out[0]) / 1024 / 1024, int(out[1].rstrip("%"))) if len(out) >= 2 else None
    except Exception:
        return h, None   # ssh/timeout: skip this host this cycle

while True:
    try:
        c = json.loads(CFG.read_text())
    except Exception:
        time.sleep(60); continue
    items = list(c["hosts"].items())
    with ThreadPoolExecutor(max_workers=min(24, len(items) or 1)) as ex:
        results = dict(ex.map(check_host, items))
    c = json.loads(CFG.read_text())  # re-read before writing
    changed = False
    for h, res in results.items():
        if res is None or h not in c["hosts"]:
            continue
        free_gb, pct = res
        if (free_gb < MIN_FREE_GB or pct > MAX_PCT) and c["hosts"][h].get("conc_total", 0) > 0:
            c["hosts"][h]["conc_total"] = 0
            changed = True
            print(f"EXCLUDE {h} free={free_gb:.1f}G pct={pct}% (conc_total->0)", flush=True)
    if changed:
        CFG.write_text(json.dumps(c, indent=2))
    time.sleep(240)
