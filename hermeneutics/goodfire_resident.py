"""Goodfire/reward-hack-data extraction with Qwen3.8-27B resident on one large GPU (96 GB): the 51 decoder blocks the
three layers need (about 41 GB bf16) and the embeddings are loaded once, then every rollout is teacher-forced on its own
through the same extract.apply_layer the streaming pass uses. Same rendering, job list, output layout, meta and resume
rule as extract_goodfire.py; the only differences are where the weights live and that one rollout's failure (out of
memory, non-finite values) is recorded in <out>/<run_dir>/_failed.jsonl and the run moves on.

Per rollout, under <out>/<run_dir>/<rollout_id>/: acts.safetensors (think_L<L> [n_turns, 2, 5120] fp16, think_pos,
think_readable, ids), spans.jsonl (empty), meta.json (written last: the done marker; adds absmax_L<L>, the largest
|value| of the bf16 states at the think positions before the fp16 cast, and the software versions).

    python -m hermeneutics.goodfire_resident --model /workspace/models/Qwen3.8-27B --out /workspace/acts \\
        --traces /workspace/data/goodfire/*/traces.jsonl [--max-tokens 262144] [--limit 2]
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from . import extract_goodfire
from .extract import LAYERS, LINEAR_CHUNK, PREFIX, Qwen3_5TextRotaryEmbedding, Weights, apply_layer, load_layer, model_dir_of
from .goodfire import label_rows, read_ids, run_dir_of

FP16_MAX = 65504.0


def versions(dev):
    """Software and hardware the vectors came from; gdn_kernel names the Gated DeltaNet implementation transformers
    bound at import (the fla package when importable, else its torch fallback)."""
    import importlib.metadata as md
    import transformers
    try:
        fla = md.version("fla-core")  # the package that ships the `fla` module (flash-linear-attention is a shell)
    except md.PackageNotFoundError:
        fla = None
    try:
        import fla as _fla  # noqa: F401
        gdn = f"fla {fla}"
    except Exception:  # noqa: BLE001
        gdn = "torch fallback"
    return {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(dev), "gdn_kernel": gdn, "code_sha": os.environ.get("HERM_CODE_SHA", "")}


class Resident:
    """The embeddings and blocks 0 .. max(layers)-1 on the GPU, built once."""

    def __init__(self, mdir, layers, dev):
        self.cfg = AutoConfig.from_pretrained(mdir).text_config
        self.cfg._attn_implementation = "sdpa"
        w = Weights(mdir)
        t0 = time.time()
        self.emb = w.get([f"{PREFIX}embed_tokens.weight"], dev)[f"{PREFIX}embed_tokens.weight"]
        self.blocks = []
        for i in range(max(layers)):
            self.blocks.append(load_layer(self.cfg, w, i, dev))
        self.rotary = Qwen3_5TextRotaryEmbedding(config=self.cfg).to(dev)
        self.layers, self.dev = layers, dev
        print(f"resident: {len(self.blocks)} blocks + embeddings in {time.time() - t0:.0f}s, "
              f"GPU {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB", flush=True)

    @torch.no_grad()
    def forward(self, ids, think, linear_chunk=LINEAR_CHUNK):
        """{think_L<L>: [n_turns, 2, d] fp16} and {L: absmax} for one rollout."""
        h = self.emb[torch.tensor(ids, device=self.dev)][None]
        pos = torch.tensor([p for pair in think for p in pair], dtype=torch.long, device=self.dev)
        acts, absmax = {}, {}
        for i, block in enumerate(self.blocks):
            h = apply_layer(block, self.cfg, h, self.rotary, linear_chunk)
            if i + 1 in self.layers:
                x = h[0][pos].view(len(think), 2, h.shape[-1])
                absmax[i + 1] = float(x.float().abs().max()) if x.numel() else 0.0
                acts[f"think_L{i + 1}"] = x.half().cpu()
        del h
        return acts, absmax


def append_jsonl(path, rows, key="rollout_id"):
    """Append rows whose key is not already in the file (restarts do not duplicate)."""
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            seen = {json.loads(line).get(key) for line in f if line.strip()}
    with open(path, "a") as f:
        for r in rows:
            if r.get(key) not in seen:
                f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True, help="<run_dir>/traces.jsonl files, extracted in this order")
    ap.add_argument("--ids-dir", default=None, help="optional: <ids-dir>/<run_dir>.ids restricts each run's rollouts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--max-tokens", type=int, default=262_144, help="skip (or with --truncate cut) longer rollouts")
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--linear-chunk", type=int, default=LINEAR_CHUNK)
    ap.add_argument("--limit", type=int, default=0, help="at most N rollouts per traces file (pilots)")
    args = ap.parse_args()
    layers = sorted(int(x) for x in args.layers.split(","))
    dev = torch.device("cuda")
    mdir = model_dir_of(args.model)
    model = os.path.basename(os.path.normpath(mdir))
    tok = AutoTokenizer.from_pretrained(mdir)
    env = versions(dev)
    print("env", json.dumps(env), flush=True)
    res = Resident(mdir, layers, dev)
    for traces in args.traces:
        run_dir = run_dir_of(traces)
        os.makedirs(os.path.join(args.out, run_dir), exist_ok=True)
        ids = None
        if args.ids_dir:
            p = os.path.join(args.ids_dir, f"{run_dir}.ids")
            ids = read_ids(p) if os.path.exists(p) else None
        labels = label_rows(os.path.dirname(os.path.abspath(traces)))
        t0 = time.time()
        jobs, skipped = extract_goodfire.jobs_of(traces, ids, args.out, run_dir, tok, args.max_tokens, False,
                                                 args.truncate, labels)
        append_jsonl(os.path.join(args.out, run_dir, "_skipped.jsonl"),
                     [{"rollout_id": r, "n_tokens": n, "reason": why} for r, n, why in skipped])
        if args.limit:
            jobs = jobs[:args.limit]
        ntot = sum(len(j[2]) for j in jobs)
        print(f"=== {run_dir}: {len(jobs)} rollouts to extract, {len(skipped)} skipped, {ntot} tokens, "
              f"labels for {sum(1 for j in jobs if j[6].get('label_source'))}, rendered in {time.time() - t0:.0f}s", flush=True)
        t_run, done_tok, failed, streak = time.time(), 0, [], 0
        for k, job in enumerate(jobs):
            row, rid, toks, _, think, _, info = job
            try:
                acts, absmax = res.forward(toks, think, args.linear_chunk)
                info = {**info, **env, **{f"absmax_L{L}": v for L, v in absmax.items()},
                        "fp16_overflow": any(v > FP16_MAX for v in absmax.values())}
                extract_goodfire.write(args.out, (row, rid, toks, None, think, job[5], info), acts, layers, model)
            except (torch.OutOfMemoryError, RuntimeError) as e:
                failed.append({"rollout_id": rid, "n_tokens": len(toks), "error": f"{type(e).__name__}: {str(e)[:300]}"})
                append_jsonl(os.path.join(args.out, run_dir, "_failed.jsonl"), failed[-1:])
                print(f"  FAILED {rid} ({len(toks)} tokens): {failed[-1]['error'][:160]}", flush=True)
                torch.cuda.empty_cache()
                streak += 1
                if streak >= 5:  # a broken CUDA context fails every rollout after it: stop instead of skipping them all
                    raise SystemExit(f"5 failures in a row in {run_dir}; stopping")
                continue
            streak = 0
            done_tok += len(toks)
            if k % 25 == 0 or k + 1 == len(jobs):
                el = time.time() - t_run
                rate = done_tok / el if el else 0
                print(f"  {run_dir} {k + 1}/{len(jobs)}: {done_tok} tokens, {rate:.0f} tok/s, "
                      f"remaining about {(ntot - done_tok) / rate / 3600 if rate else 0:.2f} h, "
                      f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        print(f"=== {run_dir} done: {len(jobs) - len(failed)} written, {len(failed)} failed, "
              f"{(time.time() - t_run) / 3600:.2f} h", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
