#!/usr/bin/env python3
"""Key guard. Every cycle, query each pool key's live remaining via OpenRouter /api/v1/key and
RETIRE (drop from orch_config.json key_hosts) any key with remaining < MIN_REMAIN (OpenRouter
pre-reserves credit for max_tokens, so a near-empty key 402s every call). Keys are pulled from each
host's /root/.zshrc and NEVER printed. Prints retirements + pool total; WARNs if the pool runs low."""
import json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CFG = Path(__file__).resolve().parent / "orch_config.json"
MIN_REMAIN = 1.50   # raised: lag lets $0.80-$1.50 keys 402; prune churners proactively (persistent in config)
                    # qwen request reserves ~$0.25 (131072 max_tokens x $1.875/M); $0.80 margin avoids 402s
LOW_POOL = 25.0

def remaining(kh):
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", f"arena9-{kh}",
                            "zsh -lic 'printf %s \"$OPENROUTER_API_KEY\"'"], capture_output=True, timeout=30)
        tok = ""
        for p in r.stdout.decode("utf-8", "replace").replace("\r", "\n").split():
            if p.startswith("sk-or-v1-"):
                tok = p
        if not tok:
            return None
        r2 = subprocess.run(["curl", "-s", "--max-time", "15", "-H", f"Authorization: Bearer {tok}",
                             "https://openrouter.ai/api/v1/key"], capture_output=True, timeout=25)
        d = json.loads(r2.stdout.decode("utf-8", "replace"))["data"]
        lim = d.get("limit")
        return (lim - d.get("usage", 0)) if lim is not None else 999.0
    except Exception:
        return None   # ssh/curl timeout or parse error: skip this key this cycle (re-checked next cycle)

while True:
    try:
        c = json.loads(CFG.read_text())
    except Exception:
        time.sleep(60); continue
    pool = [kh[0] for kh in c.get("key_hosts", [])]
    with ThreadPoolExecutor(max_workers=min(24, len(pool) or 1)) as ex:
        rem = dict(zip(pool, ex.map(remaining, pool)))
    c = json.loads(CFG.read_text())  # re-read before writing
    keep, retired = [], []
    for kh in c.get("key_hosts", []):
        r = rem.get(kh[0])
        if r is not None and r < MIN_REMAIN:
            retired.append((kh[0], r))
        else:
            keep.append(kh)
    if retired:
        c["key_hosts"] = keep
        CFG.write_text(json.dumps(c, indent=2))
    total = sum(v for v in rem.values() if v is not None and v < 900)
    print(f"[{time.strftime('%H:%M:%S')}] pool={len(keep)} keys total=${total:.2f} "
          f"retired={[(h, round(x,2)) for h,x in retired]}", flush=True)
    if total < LOW_POOL:
        print(f"WARN key pool low: ${total:.2f} remaining across {len(keep)} keys", flush=True)
    time.sleep(420)
