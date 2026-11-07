"""Residual-stream activations of Qwen3.8-27B teacher-forced over the authors' transcripts: the reader's states at the
<think> and </think> tokens of every assistant turn, at three layers, plus sentence means as a fallback.

Adapted from arena-capstone OLD/extraction/extract.py and OLD/swarm/acts.py (PREEXISTING.md). One forward pass per
agent over its rendered sequence (render.py), run block by block: every decoder block is read from the safetensors
shards, moved to the GPU, applied to the hidden states of every agent of the batch, and dropped, so the 52 GB bf16
model runs on a 44 GB card. Layers are counted as in HF's hidden_states: layer L is the residual stream after L blocks.
The pass stops after the deepest requested layer (51 of 64).

Per agent, under <out>/<run>/<agent>/:
  acts.safetensors  think_L<L>: [n_turns, 2, 5120] fp16, the states at (<think>, </think>) of each assistant turn
                    mean_L<L>:  [n_units, 5120] fp16, one row per sentence unit of spans.jsonl (fallback features)
                    think_pos [n_turns, 2] int32 token indices; think_readable [n_turns] int8; ids [n_tokens] int32
  spans.jsonl       one line per unit: segment, call, name, chars, tokens=[start, end), text
  meta.json         run, agent, author, condition, task, source model, layers, token count ... (written last: done marker)

Resumable: agents with a meta.json are skipped. Batches hold at most --batch-tokens tokens (hidden states wait in CPU RAM,
10 KiB per token); each batch reads the weights once.

    python -m hermeneutics.extract --runs-file data/subsample.csv --out /root/work_hermeneutics/acts
    python -m hermeneutics.extract --runs RUN ... --out /root/work_hermeneutics/acts_test
"""
import argparse
import json
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
                                                       Qwen3_5TextRotaryEmbedding)

from . import DATA_ROOT, MODEL_DIR
from .render import TEMPLATE_KWARGS, sequence
from .runs import run_list
from .think import CLOSE, OPEN

PREFIX = "model.language_model."
LAYERS = [26, 38, 51]  # 0.4, 0.6, 0.8 of 64 blocks, in hidden_states numbering
MLP_CHUNK = 8192  # tokens per MLP call; bounds the 17408-wide intermediate
LINEAR_CHUNK = 32768  # tokens per Gated DeltaNet call; its conv and recurrent state carry over between chunks
NORM_CHUNK = 32768
_norm_forward = Qwen3_5RMSNorm.forward


def _chunked_norm(self, x):
    """RMSNorm upcasts to fp32; over a 200k-token sequence the attention's q_norm alone needs gigabytes of temporaries.
    The norm is row-wise, so chunking along the sequence is exact."""
    if x.dim() >= 3 and x.shape[1] > NORM_CHUNK:
        return torch.cat([_norm_forward(self, c) for c in x.split(NORM_CHUNK, dim=1)], dim=1)
    return _norm_forward(self, x)


Qwen3_5RMSNorm.forward = _chunked_norm


class Weights:
    """Lazy reader of the checkpoint's tensors."""

    def __init__(self, model_dir):
        self.dir = model_dir
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]

    def get(self, names, device):
        out = {}
        by_file = {}
        for n in names:
            by_file.setdefault(self.map[n], []).append(n)
        for f, ns in by_file.items():
            with safe_open(os.path.join(self.dir, f), framework="pt", device="cpu") as h:
                for n in ns:
                    out[n] = h.get_tensor(n).to(device)
            # drop the shard's pages from the page cache: the container's memory limit counts them
            fd = os.open(os.path.join(self.dir, f), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        return out

    def layer_names(self, i):
        p = f"{PREFIX}layers.{i}."
        return [n for n in self.map if n.startswith(p)]


def init_random(layer, gen):
    """Weights drawn as Qwen3_5PreTrainedModel._init_weights draws them, with no trained values: Linear and Conv1d
    N(0, 0.02) with zero bias, RMSNorm weight 0 (the norm scales by 1 + weight), gated RMSNorm weight 1, dt_bias 1,
    A_log = log U(0.01, 16). Every parameter must be covered, or this raises."""
    from torch import nn
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5RMSNormGated
    done = set()
    for m in layer.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            m.weight.data.normal_(0, 0.02, generator=gen)
            done.add(id(m.weight))
            if m.bias is not None:
                m.bias.data.zero_()
                done.add(id(m.bias))
        elif isinstance(m, Qwen3_5RMSNorm):
            m.weight.data.zero_()
            done.add(id(m.weight))
        elif isinstance(m, Qwen3_5RMSNormGated):
            m.weight.data.fill_(1.0)
            done.add(id(m.weight))
        elif isinstance(m, Qwen3_5GatedDeltaNet):
            m.dt_bias.data.fill_(1.0)
            m.A_log.data.copy_(torch.empty_like(m.A_log).uniform_(0.01, 16, generator=gen).log_())
            done.update((id(m.dt_bias), id(m.A_log)))
    missing = [n for n, p in layer.named_parameters() if id(p) not in done]
    if missing:
        raise RuntimeError(f"parameters without an init rule: {missing}")


def load_layer(cfg, weights, i, device, random_seed=None):
    """Decoder block i with its trained weights, or (random_seed given) with random weights: the random-weights reader."""
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(cfg, i)
    if random_seed is not None:
        layer.to_empty(device=device)
        init_random(layer, torch.Generator(device=device).manual_seed(random_seed * 1000 + i))
        layer.to(torch.bfloat16)
    else:
        p = f"{PREFIX}layers.{i}."
        sd = {n[len(p):]: t for n, t in weights.get(weights.layer_names(i), device).items()}
        layer.load_state_dict(sd, strict=True, assign=True)
    mlp = layer.mlp
    plain = mlp.forward

    def chunked(x):
        return torch.cat([plain(c) for c in x.split(MLP_CHUNK, dim=1)], dim=1)

    mlp.forward = chunked
    return layer.eval()


def apply_layer(layer, cfg, h, rotary, linear_chunk):
    """One decoder block over a whole sequence h [1, L, d]: Gated DeltaNet in chunks through a cache that carries the
    conv and recurrent state (HF's chunked prefill); full attention over the whole sequence."""
    x = layer.input_layernorm(h)
    if layer.block_type == "linear_attention":
        cache = DynamicCache(config=cfg)
        a = torch.cat([layer.linear_attn(hidden_states=c, cache_params=cache, attention_mask=None)
                       for c in x.split(linear_chunk, dim=1)], dim=1)
    else:
        pos = torch.arange(h.shape[1], device=h.device).view(1, 1, -1).expand(3, 1, -1)
        a, _ = layer.self_attn(hidden_states=x, position_embeddings=rotary(h, pos), attention_mask=None,
                               position_ids=pos[0])
    h = h + a
    return h + layer.mlp(layer.post_attention_layernorm(h))


def pool(h, tokens, n_units):
    """Mean (fp32 sum) vector per unit. h: [L, d]; tokens: [start, end) per unit."""
    unit_of = torch.full((h.shape[0],), -1, dtype=torch.long)
    for u, (a, b) in enumerate(tokens):
        unit_of[a:b] = u
    unit_of = unit_of.to(h.device)
    keep = unit_of >= 0
    sums = torch.zeros(n_units, h.shape[1], dtype=torch.float32, device=h.device)
    for k in range(0, h.shape[0], 16384):
        sel = keep[k:k + 16384]
        sums.index_add_(0, unit_of[k:k + 16384][sel], h[k:k + 16384][sel].float())
    counts = torch.bincount(unit_of[keep], minlength=n_units).clamp(min=1)
    return (sums / counts[:, None]).half()


# ---------------------------------------------------------------- jobs
def jobs_of(rows, tok, out, max_tokens):
    """[(row, agent, ids, units, think, readable, info)] for the runs not yet extracted under out."""
    jobs = []
    for r in rows:
        for agent, q in sequence(r["eval_log"], tok).items():
            if os.path.exists(os.path.join(out, r["run"], agent, "meta.json")):
                continue
            ids, units, think, readable = q["ids"], q["units"], q["think"], q["readable"]
            info = {"source_model": q["model"], "calls": q["calls"], "turns": q["turns"], "append_only": q["append_only"],
                    "full_tokens": len(ids)}
            if len(ids) > max_tokens:
                ids = ids[:max_tokens]
                kept = [(a, b) for a, b in think if b < max_tokens]
                info.update(truncated_at=max_tokens, units_dropped=len(units) - sum(u["tokens"][1] <= max_tokens for u in units),
                            turns_dropped=len(think) - len(kept))
                units = [u for u in units if u["tokens"][1] <= max_tokens]
                think, readable = kept, readable[:len(kept)]
            for a, b in think:
                assert ids[a] == OPEN and ids[b] == CLOSE, (r["run"], agent, a, b)
            jobs.append((r, agent, ids, units, think, readable, info))
    return jobs


def plan(jobs, limit):
    """Batches of jobs, each at most `limit` tokens (a single longer job is its own batch)."""
    out, cur, n = [], [], 0
    for j in jobs:
        k = len(j[2])
        if cur and n + k > limit:
            out.append(cur)
            cur, n = [], 0
        cur.append(j)
        n += k
    return out + ([cur] if cur else [])


def run_batch(batch, layers, cfg, weights, dev, linear_chunk, random_seed=None):
    """{f"think_L{L}": [n_turns, 2, d], f"mean_L{L}": [n_units, d]} fp16 per job."""
    if random_seed is None:
        emb = weights.get([f"{PREFIX}embed_tokens.weight"], dev)[f"{PREFIX}embed_tokens.weight"]
    else:  # random embeddings too: N(0, 0.02), the model's own initialiser
        g = torch.Generator(device=dev).manual_seed(random_seed * 1000 + 999)
        emb = (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    hidden = [emb[torch.tensor(j[2], device=dev)][None].cpu() for j in batch]  # pageable: pinned cache grows unbounded
    del emb
    torch.cuda.empty_cache()
    rotary = Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)
    acts = [{} for _ in batch]
    t0 = time.time()
    for i in range(max(layers)):
        layer = load_layer(cfg, weights, i, dev, random_seed)
        with torch.no_grad():
            for k, j in enumerate(batch):
                h = hidden[k].to(dev, non_blocking=True)
                h = apply_layer(layer, cfg, h, rotary, linear_chunk)
                if i + 1 in layers:
                    hk = h[0]
                    _, _, ids, units, think, _, _ = j
                    pos = torch.tensor([p for pair in think for p in pair], dtype=torch.long, device=dev)
                    acts[k][f"think_L{i + 1}"] = hk[pos].view(len(think), 2, hk.shape[1]).half().cpu()
                    acts[k][f"mean_L{i + 1}"] = pool(hk, [u["tokens"] for u in units], len(units)).cpu()
                hidden[k].copy_(h)
                del h
        del layer
        torch.cuda.empty_cache()
        if i % 8 == 7 or i + 1 == max(layers):
            print(f"  block {i} done, {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                  flush=True)
    del hidden
    return acts


def write(out, job, tensors, layers, model):
    r, agent, ids, units, think, readable, info = job
    d = os.path.join(out, r["run"], agent)
    os.makedirs(d, exist_ok=True)
    t = dict(tensors)
    for name, x in t.items():
        if not torch.isfinite(x).all():
            raise RuntimeError(f"non-finite values in {r['run']}/{agent} {name}")
    t["ids"] = torch.tensor(ids, dtype=torch.int32)
    t["think_pos"] = torch.tensor(think, dtype=torch.int32).reshape(len(think), 2)
    t["think_readable"] = torch.tensor(readable, dtype=torch.int8)
    save_file(t, os.path.join(d, "acts.safetensors"))
    with open(os.path.join(d, "spans.jsonl"), "w") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")
    meta = {"run": r["run"], "agent": agent, "author": r["author"], "condition": r["condition"], "task": r["task"],
            "task_name": r["task_name"], "seed": r["seed"], "host": r["host"], "eval_log": os.path.basename(r["eval_log"]),
            "model": model, "layers": layers, "n_tokens": len(ids), "n_units": len(units), "n_turns": len(think),
            "n_readable": int(sum(readable)), "dtype": "float16", "think_ids": [OPEN, CLOSE], "template": TEMPLATE_KWARGS,
            "layer_convention": "HF hidden_states index: residual stream after L blocks", "teacher_forced": True, **info}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)


def model_dir_of(path):
    """A snapshot directory that holds the weights (the HF cache can hold a tokenizer-only snapshot beside it)."""
    if os.path.exists(os.path.join(path, "model.safetensors.index.json")):
        return path
    import glob
    for s in sorted(glob.glob(os.path.join(path, "snapshots", "*"))):
        if os.path.exists(os.path.join(s, "model.safetensors.index.json")):
            return s
    raise FileNotFoundError(f"no model.safetensors.index.json under {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--runs-file", default=None, help="CSV with a run column; extracted in file order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--batch-tokens", type=int, default=2_000_000)
    ap.add_argument("--max-tokens", type=int, default=262_144, help="cut longer sequences (Qwen3.8's context)")
    ap.add_argument("--linear-chunk", type=int, default=LINEAR_CHUNK)
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--first", type=int, default=0, help="only the first N runs of the list")
    ap.add_argument("--random-init", type=int, default=None, metavar="SEED",
                    help="random-weights reader: the same architecture with random blocks and embeddings (a null)")
    args = ap.parse_args()
    layers = sorted(int(x) for x in args.layers.split(","))
    os.makedirs(args.out, exist_ok=True)
    mdir = model_dir_of(args.model)
    model = os.path.basename(mdir)
    rows = {r["run"]: r for r in run_list(args.data_root) if r["valid"]}
    if args.runs_file:
        import pandas as pd
        want = list(pd.read_csv(args.runs_file).run)
    elif args.runs:
        want = args.runs
    else:
        ap.error("--runs or --runs-file")
    missing = [w for w in want if w not in rows]
    if missing:
        print(f"{len(missing)} requested runs not valid or not found: {missing[:5]} ...", flush=True)
    rows = [rows[w] for w in want if w in rows]
    if args.first:
        rows = rows[:args.first]
    if args.random_init is not None:
        model = f"random-init-seed{args.random_init}"

    dev = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(mdir).text_config
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(mdir)
    weights = Weights(mdir)
    t0 = time.time()
    jobs = jobs_of(rows, tok, args.out, args.max_tokens)
    batches = plan(jobs, args.batch_tokens)
    ntot = sum(len(j[2]) for j in jobs)
    print(f"{len(jobs)} agents to extract ({len(rows) - len(jobs)} already done), {ntot} tokens, "
          f"{sum(len(j[4]) for j in jobs)} turns, {len(batches)} batches, rendered in {time.time() - t0:.0f}s", flush=True)
    t_all = time.time()
    done_tokens = 0
    for b, batch in enumerate(batches):
        if args.limit_batches and b >= args.limit_batches:
            break
        t0 = time.time()
        ntok = sum(len(j[2]) for j in batch)
        print(f"batch {b + 1}/{len(batches)}: {len(batch)} agents, {ntok} tokens, longest {max(len(j[2]) for j in batch)}", flush=True)
        acts = run_batch(batch, layers, cfg, weights, dev, args.linear_chunk, args.random_init)
        for job, t in zip(batch, acts):
            write(args.out, job, t, layers, model)
        del acts
        dt = time.time() - t0
        done_tokens += ntok
        left = (ntot - done_tokens) / (done_tokens / (time.time() - t_all))
        print(f"batch {b + 1} written: {dt:.0f}s, {ntok / dt:.0f} tokens/s, elapsed {(time.time() - t_all) / 3600:.2f} h, "
              f"remaining about {left / 3600:.2f} h", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
