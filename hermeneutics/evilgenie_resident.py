"""WiserHumanExperimental/evilgenie-escalation runs teacher-forced through Qwen3.8-27B: natural reward hacking by frontier
authors (Claude Opus 4.8, claude-fable-5, GPT-5.6, Gemini 3.x, Grok 4.5, Kimi, DeepSeek V4, Qwen 3.7), a held-out test.

Rows are the release's transcripts/*.jsonl lines (one Inspect sample each, keyed by run_key); the selection CSV (from
the evilgenie track: label hack/legitimate, role, judge fields, ...) names the runs. Conversion: cheatbench_convert/
convert_evilgenie.py (reasoning from the plaintext reasoning-details items; provider summaries kept, encrypted items
dropped). Rendering, checks and the forward pass are cheatbench_resident.prepare and goodfire_resident.Resident; the
layout is Goodfire's: <out>/evilgenie/<author>/<run_id>/{acts.safetensors, meta.json, rendered.txt, spans.jsonl}, where
run_id is sha1(run_key)[:20] (run_key holds '/' and '::'); meta.json keeps run_key and every selection column.

    python -m hermeneutics.evilgenie_resident --data DIR/transcripts --select evilgenie_selection.csv --model DIR --out OUT
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time

import torch
from transformers import AutoTokenizer

from .cheatbench_resident import prepare
from .extract import LAYERS, model_dir_of
from .goodfire_resident import FP16_MAX, Resident, versions
from .render import TEMPLATE_KWARGS
from .think import CLOSE, OPEN

SOURCE_DATASET = "WiserHumanExperimental/evilgenie-escalation"
SOURCE_REVISION = "1bb777b666b0f76801f9b5b9a28eb3f7786ea70f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the release's transcripts/ folder")
    ap.add_argument("--select", required=True)
    ap.add_argument("--converters", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=262_144)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("convert_evilgenie", os.path.join(a.converters, "convert_evilgenie.py"))
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    sel = list(csv.DictReader(open(a.select)))
    want = {r["run_key"]: r for r in sel}
    rows = {}
    for fn in sorted({r["file"] for r in sel}):
        with open(os.path.join(a.data, fn), "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                x = json.loads(line)
                if x.get("run_key") in want:
                    rows[x["run_key"]] = (x, hashlib.sha256(line.rstrip(b"\n")).hexdigest())
    mdir = model_dir_of(a.model)
    tok = AutoTokenizer.from_pretrained(mdir)
    jobs, skipped = [], []
    for s in sel:
        rid = hashlib.sha1(s["run_key"].encode()).hexdigest()[:20]
        d = os.path.join(a.out, "evilgenie", s["author"], rid)
        if os.path.exists(os.path.join(d, "meta.json")):
            continue
        if s["run_key"] not in rows:
            skipped.append({"run_key": s["run_key"], "reason": "not in transcripts"})
            continue
        x, sha = rows[s["run_key"]]
        try:
            prep, why = prepare(x, conv.convert, tok, a.max_tokens)
        except Exception as e:  # noqa: BLE001
            prep, why = None, f"convert/render error {type(e).__name__}: {str(e)[:200]}"
        if prep is None:
            skipped.append({"run_key": s["run_key"], "author": s["author"], "label": s["label"], "reason": why})
            continue
        jobs.append((s, rid, d, sha, *prep))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "_skipped_evilgenie.jsonl"), "a") as f:
        for k in skipped:
            f.write(json.dumps(k) + "\n")
    ntot = sum(len(j[4]) for j in jobs)
    print(f"{len(jobs)} runs to extract, {len(skipped)} skipped, {ntot} tokens", flush=True)
    for k in skipped[:20]:
        print("  skipped", k, flush=True)
    if a.plan_only:
        return
    from safetensors.torch import save_file
    dev = torch.device("cuda")
    env = versions(dev)
    if not env["code_sha"]:
        raise SystemExit("set HERM_CODE_SHA")
    weights = None
    if a.stream:
        from .goodfire_stream import RemoteWeights
        weights = RemoteWeights(mdir)
        weights.stop_block = max(LAYERS)
    res = Resident(mdir, list(LAYERS), dev, weights=weights)
    t0, done = time.time(), 0
    for k, (s, rid, d, sha, ids, think, readable, info) in enumerate(jobs):
        try:
            acts, absmax = res.forward(ids, think)
            for name, t in acts.items():
                if t.is_floating_point() and not torch.isfinite(t.float()).all():
                    raise RuntimeError(f"non-finite {name}")
            os.makedirs(d, exist_ok=True)
            t = dict(acts)
            t["ids"] = torch.tensor(ids, dtype=torch.int32)
            t["think_pos"] = torch.tensor(think, dtype=torch.int32).reshape(len(think), 2)
            t["think_readable"] = torch.tensor(readable, dtype=torch.int8)
            save_file(t, os.path.join(d, "acts.safetensors.tmp"))
            os.replace(os.path.join(d, "acts.safetensors.tmp"), os.path.join(d, "acts.safetensors"))
            text = tok.decode(ids, skip_special_tokens=False)
            with open(os.path.join(d, "rendered.txt.tmp"), "w") as f:
                f.write(text)
            os.replace(os.path.join(d, "rendered.txt.tmp"), os.path.join(d, "rendered.txt"))
            open(os.path.join(d, "spans.jsonl"), "w").close()
            meta = {"dataset": SOURCE_DATASET, "source_dataset": SOURCE_DATASET, "source_revision": SOURCE_REVISION,
                    "source_path": f"transcripts/{s['file']}", "source_row": s["run_key"], "source_sha256": sha,
                    "run_dir": f"evilgenie/{s['author']}", "rollout_id": rid, "author": s["author"], "condition": "evilgenie",
                    **{f"sel_{c}": v for c, v in s.items()},
                    "cheat": {"hack": 1, "legitimate": 0}.get(s.get("label")),
                    "model": os.path.basename(os.path.normpath(mdir)), "layers": list(LAYERS), "n_tokens": len(ids),
                    "n_units": 0, "n_turns": len(think), "n_readable": int(sum(readable)), "units": False, "dtype": "float16",
                    "think_ids": [OPEN, CLOSE], "template": TEMPLATE_KWARGS,
                    "layer_convention": "HF hidden_states index: residual stream after L blocks", "teacher_forced": True,
                    "turn_vectors": True, "rendered_chars": len(text), **info, **env,
                    **{f"absmax_L{L}": v for L, v in absmax.items()},
                    "fp16_overflow": any(v > FP16_MAX for v in absmax.values())}
            with open(os.path.join(d, "meta.json.tmp"), "w") as f:
                json.dump(meta, f, indent=1)
            os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))
        except (torch.OutOfMemoryError, RuntimeError) as e:
            with open(os.path.join(a.out, "_failed_evilgenie.jsonl"), "a") as f:
                f.write(json.dumps({"run_key": s["run_key"], "n_tokens": len(ids), "error": f"{type(e).__name__}: {str(e)[:300]}"}) + "\n")
            torch.cuda.empty_cache()
            continue
        done += len(ids)
        if k % 20 == 0 or k + 1 == len(jobs):
            el = time.time() - t0
            print(f"  {k + 1}/{len(jobs)}: {done} tokens, {done / el:.0f} tok/s", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
