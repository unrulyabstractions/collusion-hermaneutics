"""steinad/CheatBench traces teacher-forced through Qwen3.8-27B, in the Goodfire layout, for a held-out test of cheating
probes on large closed authors (Claude Opus 4.6, GPT-5.4, GPT-5.3-codex, Claude 3.7 Sonnet, Qwen3-Coder-480B).

Each selected trace is converted to chat messages by the verified converter for its harness (cheatbench_convert/), rendered
in Qwen3.8's template (render.render), checked like Goodfire (one <think>/</think> pair per assistant message, decoding to
its reasoning under NFC; secrets redacted), and run through goodfire_resident.Resident, with the weights read from disk or,
with --stream, fetched from the Hub into GPU memory (goodfire_stream.RemoteWeights). CheatBench traces carry no reasoning,
so most think blocks are empty: the vectors are read at the turn boundaries.

Per trace, under <out>/<harness>/<trace_id>/: acts.safetensors (think_L26/38/51 [n_turns, 2, 5120] fp16, think_pos,
think_readable, ids), meta.json (labels, loc_* fields, converter info, provenance, versions; written last), rendered.txt
(the decoded ids, secrets already redacted), spans.jsonl (empty). No raw trace is written: the public traces hold leaked
credentials; meta.json records the source row id and the sha256 of its raw trace string instead.

    python -m hermeneutics.cheatbench_resident --parquet full.parquet --select select.csv --model DIR --out OUT [--stream]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time
import unicodedata

import torch
from transformers import AutoTokenizer

from .extract import LAYERS, model_dir_of
from .goodfire_resident import FP16_MAX, Resident, versions
from .render import TEMPLATE_KWARGS, render
from .think import CLOSE, OPEN, think_positions

SOURCE_DATASET = "steinad/CheatBench"
SOURCE_REVISION = "461360b34e93644955563dcc9243c6cc1a53bd0f"
CONVERTER = {"SWE-agent": "swe_agent", "Pilot": "pilot", "BoxPwnr": "boxpwnr", "ForgeCode": "forgecode", "OpenHands": "openhands"}
ROW_FIELDS = ("trace_id", "benchmark", "harness", "model", "label", "cheating_type", "cheating_mechanism", "split", "rationale",
              "loc_kind", "loc_pattern", "loc_blatancy", "loc_field_path", "loc_line_start", "loc_line_end", "loc_char_start",
              "loc_char_end", "loc_snippet", "loc_confidence", "annotator_id", "label_source")


def load_converter(conv_dir, harness):
    name = CONVERTER[harness]
    spec = importlib.util.spec_from_file_location(f"convert_{name}", os.path.join(conv_dir, f"convert_{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.convert


def prepare(row, convert, tok, max_tokens):
    """(ids, think, readable, info) or (None, reason)."""
    out = convert(row)
    msgs, tools, cinfo = out["messages"], out.get("tools") or [], out.get("info") or {}
    text = render(tok, msgs, tools, False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_tokens:
        return None, f"over_max_tokens {len(ids)}"
    think = think_positions(ids)
    asst = [m for m in msgs if m["role"] == "assistant"]
    if len(think) != len(asst):
        return None, f"think pairs {len(think)} != assistant messages {len(asst)}"
    nfc = lambda s: unicodedata.normalize("NFC", s).strip()  # noqa: E731
    bad = [k for k, ((a, b), m) in enumerate(zip(think, asst)) if nfc(tok.decode(ids[a + 1:b])) != nfc(m.get("reasoning") or "")]
    if bad:
        return None, f"think_mismatch {len(bad)} first {bad[0]}"
    readable = [bool((m.get("reasoning") or "").strip()) for m in asst]
    return (ids, think, readable, {"converter": cinfo, "n_messages": len(msgs), "turns": len(asst)}), None


def write(out, row, ids, think, readable, acts, info, layers, model, tok):
    from safetensors.torch import save_file
    d = os.path.join(out, row["harness"], row["trace_id"])
    os.makedirs(d, exist_ok=True)
    t = dict(acts)
    for name, x in t.items():
        if not torch.isfinite(x).all():
            raise RuntimeError(f"non-finite values in {row['trace_id']} {name}")
    t["ids"] = torch.tensor(ids, dtype=torch.int32)
    t["think_pos"] = torch.tensor(think, dtype=torch.int32).reshape(len(think), 2)
    t["think_readable"] = torch.tensor(readable, dtype=torch.int8)

    def atomic(path, write_fn):
        with open(path + ".tmp", "wb" if path.endswith(".safetensors") else "w") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path + ".tmp", path)

    save_file(t, os.path.join(d, "acts.safetensors.tmp"))
    os.replace(os.path.join(d, "acts.safetensors.tmp"), os.path.join(d, "acts.safetensors"))
    text = tok.decode(ids, skip_special_tokens=False)
    atomic(os.path.join(d, "rendered.txt"), lambda f: f.write(text))
    open(os.path.join(d, "spans.jsonl"), "w").close()
    meta = {"dataset": SOURCE_DATASET, "source_dataset": SOURCE_DATASET, "source_revision": SOURCE_REVISION,
            "source_path": "data/processed/parquet/full.parquet", "source_row": row["trace_id"],
            "source_sha256": row["_trace_sha256"], "run_dir": row["harness"], "rollout_id": row["trace_id"],
            "run": row["harness"], "agent": row["trace_id"], "author": row["model"], "condition": "cheatbench",
            **{f: row.get(f) for f in ROW_FIELDS},
            "cheat": {"cheating": 1, "attempt": 1, "benign": 0}.get(row.get("label")),
            "model": model, "layers": layers, "n_tokens": len(ids), "n_units": 0, "n_turns": len(think),
            "n_readable": int(sum(readable)), "units": False, "dtype": "float16", "think_ids": [OPEN, CLOSE],
            "template": TEMPLATE_KWARGS, "layer_convention": "HF hidden_states index: residual stream after L blocks",
            "teacher_forced": True, "rendered_chars": len(text), **info}
    atomic(os.path.join(d, "meta.json"), lambda f: f.write(json.dumps(meta, indent=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--select", required=True, help="CSV with a trace_id column: the traces to extract, in order")
    ap.add_argument("--converters", required=True, help="directory holding convert_<format>.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream", action="store_true", help="fetch the weights from the Hub instead of reading shards on disk")
    ap.add_argument("--max-tokens", type=int, default=262_144)
    ap.add_argument("--plan-only", action="store_true", help="convert, render and check every trace; no GPU")
    args = ap.parse_args()
    import pandas as pd
    want = [r["trace_id"] for r in csv.DictReader(open(args.select))]
    df = pd.read_parquet(args.parquet)
    df = df[df.trace_id.isin(set(want))].set_index("trace_id").loc[[w for w in want if w in set(df.trace_id)]].reset_index()
    mdir = model_dir_of(args.model) if not args.stream else args.model
    tok = AutoTokenizer.from_pretrained(mdir)
    layers = list(LAYERS)
    convs, jobs, skipped = {}, [], []
    for _, r in df.iterrows():
        row = {k: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for k, v in r.items() if k != "trace"}
        row["_trace_sha256"] = hashlib.sha256(r.trace.encode()).hexdigest()
        if os.path.exists(os.path.join(args.out, row["harness"], row["trace_id"], "meta.json")):
            continue
        tr = json.loads(r.trace)
        tr.pop("config", None)
        row["trace"] = tr
        conv = convs.setdefault(row["harness"], load_converter(args.converters, row["harness"]))
        try:
            prep, why = prepare(row, conv, tok, args.max_tokens)
        except Exception as e:  # noqa: BLE001  (record and skip)
            prep, why = None, f"convert/render error {type(e).__name__}: {str(e)[:200]}"
        row.pop("trace")
        if prep is None:
            skipped.append({"trace_id": row["trace_id"], "harness": row["harness"], "label": row["label"], "reason": why})
            continue
        jobs.append((row, *prep))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "_skipped.jsonl"), "a") as f:
        for s in skipped:
            f.write(json.dumps(s) + "\n")
    ntot = sum(len(j[1]) for j in jobs)
    print(f"{len(jobs)} traces to extract, {len(skipped)} skipped, {ntot} tokens", flush=True)
    for s in skipped[:20]:
        print("  skipped", s, flush=True)
    if args.plan_only:
        return
    dev = torch.device("cuda")
    env = versions(dev)
    if not env["code_sha"]:
        raise SystemExit("set HERM_CODE_SHA")
    weights = None
    if args.stream:
        from .goodfire_stream import RemoteWeights
        weights = RemoteWeights(mdir)
        weights.stop_block = max(layers)
    res = Resident(mdir, layers, dev, weights=weights)
    t0, done = time.time(), 0
    for k, (row, ids, think, readable, info) in enumerate(jobs):
        try:
            acts, absmax = res.forward(ids, think)
            info = {**info, **env, **{f"absmax_L{L}": v for L, v in absmax.items()},
                    "fp16_overflow": any(v > FP16_MAX for v in absmax.values())}
            write(args.out, row, ids, think, readable, acts, info, layers, os.path.basename(os.path.normpath(mdir)), tok)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            with open(os.path.join(args.out, "_failed.jsonl"), "a") as f:
                f.write(json.dumps({"trace_id": row["trace_id"], "n_tokens": len(ids), "error": f"{type(e).__name__}: {str(e)[:300]}"}) + "\n")
            torch.cuda.empty_cache()
            continue
        done += len(ids)
        if k % 10 == 0 or k + 1 == len(jobs):
            el = time.time() - t0
            print(f"  {k + 1}/{len(jobs)}: {done} tokens, {done / el:.0f} tok/s, remaining about {(ntot - done) / max(done / el, 1) / 3600:.2f} h",
                  flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
