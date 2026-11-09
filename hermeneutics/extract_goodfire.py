"""Goodfire/reward-hack-data rollouts through extract.py's layer-streaming pass: Qwen3.8-27B's residual stream at the
<think> and </think> tokens of every assistant turn, at layers 26/38/51, teacher-forced over the author's transcript.

The rollouts come from goodfire.py (render: the same shape render.sequence() gives an SGA agent); the forward pass,
batching and weights reader are extract.py's. Per rollout, under <out>/<run_dir>/<rollout_id>/:
  acts.safetensors  think_L<L> [n_turns, 2, 5120] fp16; think_pos [n_turns, 2] int32; think_readable [n_turns] int8;
                    ids [n_tokens] int32; with --units also mean_L<L> [n_units, 5120] fp16 (off by default: disk)
  spans.jsonl       the sentence units (empty unless --units)
  meta.json         dataset, run_dir, rollout_id, author, benchmark, author_model, n_tokens, n_turns, turn_nodes (the node
                    index of every assistant turn, for judge_passages joins), the label fields of judge_rollouts.jsonl or
                    monitor_rollouts.jsonl (label_source, verdict, verdict_enacted, categories, ...), written last (done marker)
Rollouts longer than --max-tokens are skipped and listed in <out>/<run_dir>/_skipped.jsonl (--truncate cuts them instead,
as extract.py does). Resumable: rollouts with a meta.json are skipped. Batches hold at most --batch-tokens tokens
(hidden states wait in CPU RAM at 10 KiB per token).

    python -m hermeneutics.extract_goodfire --traces DATA/impossiblebench-kimi-k3/traces.jsonl \\
        --ids-file plan/ids/impossiblebench-kimi-k3.shard0.ids --out /root/work_goodfire/acts
"""
import argparse
import json
import os
import time

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from . import MODEL_DIR
from .extract import LAYERS, LINEAR_CHUNK, Weights, model_dir_of, plan, run_batch
from .goodfire import DATASET, label_rows, load_rollouts, read_ids, render, run_dir_of, run_info
from .render import TEMPLATE_KWARGS
from .think import CLOSE, OPEN


def jobs_of(traces, ids, out, run_dir, tok, max_tokens, with_units=False, truncate=False, labels=None):
    """([(row, rollout_id, ids, units, think, readable, info)], skipped) for the rollouts of `ids` (None: all) not yet
    extracted under out, in traces.jsonl order. skipped: [(rollout_id, n_tokens, reason)]."""
    labels = labels or {}
    row = {"run_dir": run_dir, **run_info(run_dir)}
    jobs, skipped, seen = [], [], set()
    for i, rollout in load_rollouts(traces, ids):
        rid = rollout["traces"][0]["id"]
        seen.add(rid)
        if os.path.exists(os.path.join(out, run_dir, rid, "meta.json")):
            continue
        q = render(rollout, tok, with_units)
        tokens, units, think, readable = q["ids"], q["units"], q["think"], q["readable"]
        lab = {("label_" + k if k in ("row_index", "task") else k): v for k, v in labels.get(rid, {}).items()}
        info = {**q["info"], "row_index": i, "calls": q["calls"], "turns": q["turns"], "turn_nodes": q["turn_nodes"],
                "full_tokens": len(tokens), **lab}
        if len(think) != q["turns"]:
            skipped.append((rid, len(tokens), f"think pairs {len(think)} != turns {q['turns']}"))
            continue
        if q["info"].get("think_mismatch"):
            skipped.append((rid, len(tokens), f"think_mismatch: {q['info']['think_mismatch']} turns, first "
                                              f"{q['info'].get('think_mismatch_first')}"))
            continue
        if len(tokens) > max_tokens:
            if not truncate:
                skipped.append((rid, len(tokens), "over_max_tokens"))
                continue
            tokens = tokens[:max_tokens]
            kept = [(a, b) for a, b in think if b < max_tokens]
            info.update(truncated_at=max_tokens, turns_dropped=len(think) - len(kept), turn_nodes=q["turn_nodes"][:len(kept)])
            if units is not None:
                info["units_dropped"] = len(units) - sum(u["tokens"][1] <= max_tokens for u in units)
                units = [u for u in units if u["tokens"][1] <= max_tokens]
            think, readable = kept, readable[:len(kept)]
        for a, b in think:
            assert tokens[a] == OPEN and tokens[b] == CLOSE, (rid, a, b)
        jobs.append((row, rid, tokens, units, think, readable, info))
    if ids is not None:
        for rid in ids:
            if rid not in seen:
                skipped.append((rid, None, "not in traces.jsonl"))
    return jobs, skipped


def write(out, job, tensors, layers, model):
    row, rid, ids, units, think, readable, info = job
    d = os.path.join(out, row["run_dir"], rid)
    os.makedirs(d, exist_ok=True)
    t = dict(tensors)
    for name, x in t.items():
        if not torch.isfinite(x).all():
            raise RuntimeError(f"non-finite values in {row['run_dir']}/{rid} {name}")
    t["ids"] = torch.tensor(ids, dtype=torch.int32)
    t["think_pos"] = torch.tensor(think, dtype=torch.int32).reshape(len(think), 2)
    t["think_readable"] = torch.tensor(readable, dtype=torch.int8)
    save_file(t, os.path.join(d, "acts.safetensors"))
    with open(os.path.join(d, "spans.jsonl"), "w") as f:
        for u in units or []:
            f.write(json.dumps(u) + "\n")
    meta = {"dataset": DATASET, "run_dir": row["run_dir"], "rollout_id": rid, "run": row["run_dir"], "agent": rid,
            "author": row["author"], "author_name": row["author_name"], "benchmark": row["benchmark"], "condition": "goodfire",
            "model": model, "layers": layers, "n_tokens": len(ids), "n_units": len(units or []), "n_turns": len(think),
            "n_readable": int(sum(readable)), "units": units is not None, "dtype": "float16", "think_ids": [OPEN, CLOSE],
            "template": TEMPLATE_KWARGS, "layer_convention": "HF hidden_states index: residual stream after L blocks",
            "teacher_forced": True, **info}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="<run_dir>/traces.jsonl of the dataset")
    ap.add_argument("--ids-file", default=None, help="rollout ids, one per line (or a CSV with a rollout_id column); default all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", default=None, help="output folder name; default the traces file's directory name")
    ap.add_argument("--labels-dir", default=None, help="run directory holding labels/ or generic_monitor/; default the traces file's")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--batch-tokens", type=int, default=2_000_000)
    ap.add_argument("--max-tokens", type=int, default=100_000, help="skip (or with --truncate cut) longer rollouts")
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--units", action="store_true", help="also the sentence-mean fallback (mean_L<L>, spans.jsonl); large")
    ap.add_argument("--linear-chunk", type=int, default=LINEAR_CHUNK)
    ap.add_argument("--limit-batches", type=int, default=0)
    ap.add_argument("--first", type=int, default=0, help="only the first N jobs")
    ap.add_argument("--random-init", type=int, default=None, metavar="SEED", help="random-weights reader (a null)")
    args = ap.parse_args()
    layers = sorted(int(x) for x in args.layers.split(","))
    run_dir = args.run_dir or run_dir_of(args.traces)
    labels_dir = args.labels_dir or os.path.dirname(os.path.abspath(args.traces))
    os.makedirs(os.path.join(args.out, run_dir), exist_ok=True)
    mdir = model_dir_of(args.model)
    model = os.path.basename(mdir) if args.random_init is None else f"random-init-seed{args.random_init}"
    ids = read_ids(args.ids_file) if args.ids_file else None

    dev = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(mdir).text_config
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(mdir)
    weights = Weights(mdir)
    labels = label_rows(labels_dir)
    t0 = time.time()
    jobs, skipped = jobs_of(args.traces, ids, args.out, run_dir, tok, args.max_tokens, args.units, args.truncate, labels)
    if args.first:
        jobs = jobs[:args.first]
    if skipped:
        with open(os.path.join(args.out, run_dir, "_skipped.jsonl"), "a") as f:
            for rid, n, reason in skipped:
                f.write(json.dumps({"rollout_id": rid, "n_tokens": n, "reason": reason, "time": time.time()}) + "\n")
    batches = plan(jobs, args.batch_tokens)
    ntot = sum(len(j[2]) for j in jobs)
    print(f"{run_dir}: {len(jobs)} rollouts to extract, {len(skipped)} skipped "
          f"({sum(1 for s in skipped if s[2] == 'over_max_tokens')} over {args.max_tokens} tokens), labels for "
          f"{sum(1 for j in jobs if j[6].get('label_source'))}, {ntot} tokens, {sum(len(j[4]) for j in jobs)} turns, "
          f"{len(batches)} batches, rendered in {time.time() - t0:.0f}s", flush=True)
    t_all = time.time()
    done_tokens = 0
    for b, batch in enumerate(batches):
        if args.limit_batches and b >= args.limit_batches:
            break
        t0 = time.time()
        ntok = sum(len(j[2]) for j in batch)
        print(f"batch {b + 1}/{len(batches)}: {len(batch)} rollouts, {ntok} tokens, longest {max(len(j[2]) for j in batch)}", flush=True)
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
