"""Reference check of the block-wise teacher-forced pass (extract.apply_layer, as goodfire_resident.py runs it): HF's own
Qwen3_5TextModel.forward over each whole sequence in one call (its causal mask, its rotary positions, Gated DeltaNet with
no chunking, no cache), with forward hooks on blocks L-1 so the captured state is the residual stream after L blocks,
compared with extraction folders' think_L<L> vectors at the same positions.

    python -m hermeneutics.check_forward --model /workspace/models/Qwen3.8-27B --acts DIR [DIR ...] [--out report.json]

Prints, per folder and layer: max |difference|, relative difference ||ref - ext|| / ||ref|| and the smallest cosine over
the rows. Exits 1 if any layer of any folder has min cosine < 0.999 or relative difference > 0.02.
"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5TextRotaryEmbedding

from .extract import PREFIX, model_dir_of


def load_reference(mdir, n_blocks, dev):
    cfg = AutoConfig.from_pretrained(mdir).text_config
    cfg._attn_implementation = "sdpa"
    cfg.num_hidden_layers = n_blocks
    cfg.layer_types = list(cfg.layer_types[:n_blocks])
    with torch.device("meta"):
        m = Qwen3_5TextModel(cfg)
    index = json.load(open(os.path.join(mdir, "model.safetensors.index.json")))["weight_map"]
    keep = [n for n in index if n.startswith(PREFIX) and (
        n.startswith(f"{PREFIX}embed_tokens.") or n.startswith(f"{PREFIX}norm.") or
        (n.startswith(f"{PREFIX}layers.") and int(n[len(f"{PREFIX}layers."):].split(".")[0]) < n_blocks))]
    sd, by_file = {}, {}
    for n in keep:
        by_file.setdefault(index[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(os.path.join(mdir, f), framework="pt", device="cpu") as h:
            for n in ns:
                sd[n[len(PREFIX):]] = h.get_tensor(n).to(dev)
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    if unexpected or [k for k in missing if "rotary" not in k]:
        raise RuntimeError(f"state dict mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
    m.rotary_emb = Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)  # non-persistent buffers: rebuilt off the meta device
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--acts", nargs="+", required=True, help="extraction folders holding acts.safetensors and meta.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    mdir = model_dir_of(args.model)
    metas = [json.load(open(os.path.join(d, "meta.json"))) for d in args.acts]
    layers = sorted(set(L for m in metas for L in m["layers"]))
    model, cfg = load_reference(mdir, max(layers), dev)
    store = {}
    for L in layers:
        model.layers[L - 1].register_forward_hook(lambda mod, inp, out, L=L: store.__setitem__(L, out))
    report, bad = [], False
    for d, meta in zip(args.acts, metas):
        with safe_open(os.path.join(d, "acts.safetensors"), framework="pt") as f:
            ids, pos = f.get_tensor("ids").long(), f.get_tensor("think_pos").long()
            ext = {L: f.get_tensor(f"think_L{L}").float() for L in meta["layers"]}
        store.clear()
        with torch.no_grad():
            model(input_ids=ids[None].to(dev), use_cache=False)
        flat = pos.reshape(-1).to(dev)
        for L in meta["layers"]:
            ref = store[L][0][flat].float().cpu().view(ext[L].shape)
            diff = ref - ext[L]
            cos = torch.nn.functional.cosine_similarity(ref.view(-1, ref.shape[-1]), ext[L].view(-1, ref.shape[-1]), dim=-1)
            row = {"folder": d, "layer": L, "n_tokens": len(ids), "n_turns": int(pos.shape[0]),
                   "max_abs_diff": float(diff.abs().max()), "rel_diff": float(diff.norm() / ref.norm()),
                   "min_cos": float(cos.min()), "ref_absmax": float(ref.abs().max())}
            row["ok"] = row["min_cos"] >= 0.999 and row["rel_diff"] <= 0.02
            bad |= not row["ok"]
            report.append(row)
            print(json.dumps(row), flush=True)
        store.clear()
        torch.cuda.empty_cache()
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
    print("REFERENCE CHECK", "FAILED" if bad else "PASSED", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
