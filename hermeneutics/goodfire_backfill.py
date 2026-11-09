"""Backfill the turn-change vectors (turn_L26/38/51, turn_pos) into Goodfire rollouts extracted before they existed.

The token ids and think positions of every rollout are already in its acts.safetensors, so nothing is re-rendered. Per
rollout: download acts.safetensors and meta.json from the Hub, run the forward pass once (goodfire_resident.Resident),
keep every original tensor (think_L stays the vector from the original host) and add turn_L and turn_pos, record in
meta.json how the recomputed think_L compares with the original (min cosine, max |difference| per layer: the hardware gap
when the backfill host's GPU differs), upload acts.safetensors and meta.json, and check both files' bytes on the Hub
before the rollout enters the ledger. rendered.txt, source.json and spans.jsonl do not change.

    python -m hermeneutics.goodfire_backfill --list backfill.csv --model DIR --token-file F --work /root/backfill \
        --ledger /root/backfill_done.jsonl
"""
import argparse
import csv
import hashlib
import json
import os
import time

import torch

from .goodfire_resident import TURN_IDS, Resident, versions
from .goodfire_upload import REPO, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="CSV with run_dir,rollout_id")
    ap.add_argument("--model", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--batch", type=int, default=20, help="rollouts per Hub commit")
    ap.add_argument("--reverse", action="store_true",
                    help="walk the list from the end (a second host; rollouts the other host already did are skipped "
                         "because their Hub meta.json has turn_vectors)")
    ap.add_argument("--stream", action="store_true", help="fetch the weights from the Hub (host without the checkpoint)")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile
    from safetensors import safe_open
    from safetensors.torch import save_file
    api = HfApi(token=open(a.token_file).read().strip())
    todo = [(r["run_dir"], r["rollout_id"]) for r in csv.DictReader(open(a.list))]
    done = {(r["run_dir"], r["rollout_id"]) for r in map(json.loads, open(a.ledger))} if os.path.exists(a.ledger) else set()
    todo = [x for x in todo if x not in done]
    if a.reverse:
        todo.reverse()
    dev = torch.device("cuda")
    env = versions(dev)
    if not env["code_sha"]:
        raise SystemExit("set HERM_CODE_SHA")
    weights = None
    if a.stream:
        from .goodfire_stream import RemoteWeights
        weights = RemoteWeights(a.model)
        weights.stop_block = 51
    res = Resident(a.model, [26, 38, 51], dev, weights=weights)
    print(f"{len(todo)} rollouts to backfill, {len(done)} already in the ledger", flush=True)
    pending, t0, ntok = [], time.time(), 0

    def flush():
        if not pending:
            return
        for attempt in range(6):
            try:
                api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=a.work, path_in_repo=ROOT,
                                  allow_patterns=[f"{rd}/{rid}/acts.safetensors" for rd, rid in pending] +
                                                 [f"{rd}/{rid}/meta.json" for rd, rid in pending],
                                  commit_message=f"turn-change vectors backfilled into {len(pending)} goodfire rollouts")
                break
            except Exception as e:  # noqa: BLE001
                print(f"upload retry {attempt + 1}: {type(e).__name__} {str(e)[:200]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            print(f"upload FAILED for {len(pending)} rollouts; they stay pending", flush=True)
            return
        remote = {}
        for rd in {rd for rd, _ in pending}:
            remote.update({e.path: e for e in api.list_repo_tree(REPO, path_in_repo=f"{ROOT}/{rd}", repo_type="dataset",
                                                                  recursive=True, expand=True) if isinstance(e, RepoFile)})
        ok = 0
        with open(a.ledger, "a") as f:
            for rd, rid in pending:
                good = True
                for fn in ("acts.safetensors", "meta.json"):
                    data = open(os.path.join(a.work, rd, rid, fn), "rb").read()
                    e = remote.get(f"{ROOT}/{rd}/{rid}/{fn}")
                    good &= e is not None and (e.lfs.sha256 == hashlib.sha256(data).hexdigest() if e.lfs
                                               else e.blob_id == hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest())
                if good:
                    f.write(json.dumps({"run_dir": rd, "rollout_id": rid, "time": time.time()}) + "\n")
                    ok += 1
        print(f"{time.strftime('%FT%TZ', time.gmtime())} committed {len(pending)}, byte-verified {ok}", flush=True)
        pending.clear()

    for k, (rd, rid) in enumerate(todo):
        d = os.path.join(a.work, rd, rid)
        os.makedirs(d, exist_ok=True)
        try:
            pa = hf_hub_download(REPO, f"{ROOT}/{rd}/{rid}/acts.safetensors", repo_type="dataset", token=api.token)
            pm = hf_hub_download(REPO, f"{ROOT}/{rd}/{rid}/meta.json", repo_type="dataset", token=api.token)
            meta = json.load(open(pm))
            if meta.get("turn_vectors"):
                continue
            with safe_open(pa, framework="pt") as f:
                orig = {n: f.get_tensor(n) for n in f.keys()}
            ids, think = orig["ids"].tolist(), [tuple(x) for x in orig["think_pos"].tolist()]
            new, _ = res.forward(ids, think)
            cmp = {}
            for L in (26, 38, 51):
                o, n = orig[f"think_L{L}"].float(), new[f"think_L{L}"].float()
                cos = torch.nn.functional.cosine_similarity(o.view(-1, o.shape[-1]), n.view(-1, n.shape[-1]), dim=-1)
                cmp[f"L{L}"] = {"min_cos": float(cos.min()) if cos.numel() else None,
                                "max_abs_diff": float((o - n).abs().max()) if o.numel() else None}
            out = dict(orig)
            for L in (26, 38, 51):
                out[f"turn_L{L}"] = new[f"turn_L{L}"]
                if not torch.isfinite(out[f"turn_L{L}"].float()).all():
                    raise RuntimeError(f"non-finite turn_L{L}")
            out["turn_pos"] = new["turn_pos"]
            assert all(tuple(ids[p] for p in row) == TURN_IDS for row in out["turn_pos"].tolist())
            save_file(out, os.path.join(d, "acts.safetensors.tmp"))
            os.replace(os.path.join(d, "acts.safetensors.tmp"), os.path.join(d, "acts.safetensors"))
            meta.update(turn_vectors=True, turn_ids=list(TURN_IDS), turn_tokens=["<|im_end|>", "\\n", "<|im_start|>", "assistant", "\\n"],
                        turn_backfill={"gpu": env["gpu"], "code_sha": env["code_sha"], "torch": env["torch"],
                                       "transformers": env["transformers"], "gdn_kernel": env["gdn_kernel"],
                                       "think_L_recomputed_vs_original": cmp, "think_L_kept": "original",
                                       "time": time.strftime("%FT%TZ", time.gmtime())})
            with open(os.path.join(d, "meta.json.tmp"), "w") as f:
                json.dump(meta, f, indent=1)
            os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))
            for p in (pa, pm):
                os.remove(os.path.realpath(p))  # the cached Hub copies; the work dir keeps the new files
            pending.append((rd, rid))
            ntok += len(ids)
        except (torch.OutOfMemoryError, RuntimeError, AssertionError, OSError) as e:
            print(f"  FAILED {rd}/{rid}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            torch.cuda.empty_cache()
            continue
        if len(pending) >= a.batch:
            flush()
        if k % 20 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(todo)}: {ntok} tokens, {ntok / el:.0f} tok/s; last {rd}/{rid} L51 min cos {cmp['L51']['min_cos']}",
                  flush=True)
    flush()
    print("backfill done", flush=True)


if __name__ == "__main__":
    main()
