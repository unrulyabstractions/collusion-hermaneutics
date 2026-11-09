"""One status line for the Goodfire extraction: rollouts and tokens done (per run dir in the detail file), tokens per second
since the previous line, ETA for the planned set, uploaded and hub-verified counts. Appends to <log> and rewrites <now>.

    python -m hermeneutics.goodfire_status --acts /workspace/acts --plan /workspace/plan_v3 --ledger /workspace/uploaded.jsonl \
        --verify /workspace/hub_verify.json --log /workspace/logs/status.log --now /workspace/logs/status_now.txt
"""
import argparse
import collections
import csv
import glob
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    for k in ("acts", "plan", "ledger", "verify", "log", "now"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    planned = sum(int(r["n_tokens"]) for p in glob.glob(os.path.join(a.plan, "*.csv")) for r in csv.DictReader(open(p))
                  if r["included"] == "True")
    n_plan = sum(1 for p in glob.glob(os.path.join(a.plan, "*.csv")) for r in csv.DictReader(open(p)) if r["included"] == "True")
    per, tokens, n = collections.Counter(), 0, 0
    for m in glob.glob(os.path.join(a.acts, "*", "*", "meta.json")):
        try:
            meta = json.load(open(m))
        except ValueError:
            continue
        n += 1
        tokens += meta["n_tokens"]
        per[meta["run_dir"]] += 1
    now = time.time()
    state = os.path.join(os.path.dirname(a.log), ".status_state.json")
    prev = json.load(open(state)) if os.path.exists(state) else None
    rate = (tokens - prev["tokens"]) / (now - prev["time"]) if prev and now > prev["time"] else 0.0
    json.dump({"time": now, "tokens": tokens}, open(state, "w"))
    eta = (planned - tokens) / rate / 3600 if rate > 0 else float("nan")
    up = sum(1 for _ in open(a.ledger)) if os.path.exists(a.ledger) else 0
    ver = json.load(open(a.verify)) if os.path.exists(a.verify) else {}
    line = (f"{time.strftime('%FT%TZ', time.gmtime(now))} rollouts {n}/{n_plan} | tokens {tokens / 1e6:.1f}M/{planned / 1e6:.1f}M | "
            f"{rate:.0f} tok/s (2 pods) | ETA full set {eta:.1f} h | uploaded {up} | hub-verified {ver.get('verified', 0)} "
            f"bad {ver.get('bad', 0)}")
    with open(a.log, "a") as f:
        f.write(line + "\n")
    detail = line + "\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(per.items())) + "\n"
    with open(a.now + ".tmp", "w") as f:
        f.write(detail)
    os.replace(a.now + ".tmp", a.now)
    print(line, flush=True)


if __name__ == "__main__":
    main()
