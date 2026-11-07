"""Run ON an arena9 host (via its repo venv). Prices every completed run of a study from its eval log.
Prints one JSON object per line: {name, model, status, input, cache_read, cache_write, output, cost_usd}.
Prices come from src/models/model_registry.yaml (usd_per_million), standard (not long-prompt) rates.
"""
import glob, json, os, sys
import yaml
from inspect_ai.log import read_eval_log

study = sys.argv[1] if len(sys.argv) > 1 else "study_apart_controls"
reg = yaml.safe_load(open("src/models/model_registry.yaml"))
# build inspect-string -> rates
rates = {}
def walk(d):
    if isinstance(d, dict):
        if "inspect" in d and "usd_per_million" in d:
            rates[d["inspect"]] = d["usd_per_million"]
        for v in d.values():
            walk(v)
    elif isinstance(d, list):
        for v in d:
            walk(v)
walk(reg)

base = f"logs/{study}"
for d in sorted(glob.glob(f"{base}/*")):
    name = os.path.basename(d)
    fs = glob.glob(f"{d}/*.eval")
    if not fs:
        continue
    f = max(fs, key=os.path.getmtime)
    try:
        log = read_eval_log(f)
    except Exception as e:
        print(json.dumps({"name": name, "error": str(e)})); continue
    mu = log.stats.model_usage or {}
    total = 0.0
    rows = []
    for model, u in mu.items():
        r = rates.get(model, {})
        inp = u.input_tokens or 0
        out = u.output_tokens or 0
        cr = getattr(u, "input_tokens_cache_read", 0) or 0
        cw = getattr(u, "input_tokens_cache_write", 0) or 0
        # Inspect's input_tokens already EXCLUDES cache reads (verified: input+cache_read+output=total)
        non_cached = inp
        cost = (non_cached * r.get("input", 0) + cr * r.get("cached_input", r.get("input", 0))
                + cw * r.get("cache_write", 0) + out * r.get("output", 0)) / 1e6
        total += cost
        rows.append({"model": model, "input": inp, "cache_read": cr, "cache_write": cw,
                     "output": out, "cost_usd": round(cost, 5)})
    print(json.dumps({"name": name, "status": str(log.status), "cost_usd": round(total, 5),
                      "rows": rows}))
