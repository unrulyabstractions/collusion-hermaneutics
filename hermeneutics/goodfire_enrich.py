"""Add provenance files to finished Goodfire extraction folders, without touching the vectors.

Per <acts>/<run_dir>/<rollout_id>/ that has meta.json and acts.safetensors and is not yet enriched:
  rendered.txt  tokenizer.decode(ids, skip_special_tokens=False): the exact text the reader saw
  source.json   the rollout's own line from <data>/<run_dir>/traces.jsonl, byte for byte (its traces[0].id is checked)
  meta.json     gains source_dataset "Goodfire/reward-hack-data", source_revision, source_path "<run_dir>/traces.jsonl",
                rollout_id (already there), source_sha256 and rendered_chars; rewritten last, atomically, so a folder with
                meta.json["source_dataset"] is complete.

    python -m hermeneutics.goodfire_enrich --acts /workspace/acts --data /workspace/data/goodfire --model /workspace/models/Qwen3.8-27B
"""
import argparse
import glob
import hashlib
import json
import os
import time

SOURCE_DATASET = "Goodfire/reward-hack-data"
SOURCE_REVISION = "bd8fb473568b8ef1d12fc2325a5c8af939c15ad9"


def _atomic(path, data, mode="w"):
    tmp = path + ".tmp"
    with open(tmp, mode) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def line_offsets(traces):
    """{line_index: (offset, length)} for every line of traces.jsonl (line_index as goodfire.load_rollouts counts it)."""
    out, off = {}, 0
    with open(traces, "rb") as f:
        for i, line in enumerate(f):
            out[i] = (off, len(line))
            off += len(line)
    return out


def enrich(acts, data, tok):
    from safetensors import safe_open
    todo = []
    for m in glob.glob(os.path.join(acts, "*", "*", "meta.json")):
        meta = json.load(open(m))
        if meta.get("source_dataset") != SOURCE_DATASET:
            todo.append((m, meta))
    offsets, done = {}, 0
    for m, meta in sorted(todo):
        d = os.path.dirname(m)
        rd, rid = meta["run_dir"], meta["rollout_id"]
        traces = os.path.join(data, rd, "traces.jsonl")
        if rd not in offsets:
            offsets[rd] = line_offsets(traces)
        off, n = offsets[rd][meta["row_index"]]
        with open(traces, "rb") as f:
            f.seek(off)
            line = f.read(n)
        src = json.loads(line)
        if src["traces"][0]["id"] != rid:
            raise RuntimeError(f"{rd}/{rid}: line {meta['row_index']} holds rollout {src['traces'][0]['id']}")
        with safe_open(os.path.join(d, "acts.safetensors"), framework="np") as f:
            ids = f.get_tensor("ids").tolist()
        if len(ids) != meta["n_tokens"]:
            raise RuntimeError(f"{rd}/{rid}: {len(ids)} ids vs n_tokens {meta['n_tokens']}")
        text = tok.decode(ids, skip_special_tokens=False)
        body = line.rstrip(b"\n")
        _atomic(os.path.join(d, "source.json"), body, "wb")
        _atomic(os.path.join(d, "rendered.txt"), text)
        meta.update(source_dataset=SOURCE_DATASET, source_revision=SOURCE_REVISION, source_path=f"{rd}/traces.jsonl",
                    rollout_id=rid, source_sha256=hashlib.sha256(body).hexdigest(), rendered_chars=len(text))
        _atomic(m, json.dumps(meta, indent=1))
        done += 1
    return done, len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    done, todo = enrich(a.acts, a.data, tok)
    print(f"{time.strftime('%FT%TZ', time.gmtime())} enriched {done} of {todo} folders in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
