"""Workable snapshots of the Goodfire extraction while it runs: every finished rollout folder (meta.json present) that is
not yet packed goes into a new immutable part, so parts can be copied off the GPU host and used before the run ends.
Standalone (numpy + safetensors only): runs on the GPU host beside the extraction and anywhere the parts are read.

A part, <snap>/part_<NNNN>/:
  X_L26.npy, X_L38.npy, X_L51.npy   [n_turns_in_part, 2, 5120] float16: the residual stream at (<think>, </think>) of every
                                    assistant turn, after 26/38/51 blocks (HF hidden_states numbering)
  turns.csv      one row per turn, in array order: rollout key, turn index, think positions, readable, sampled, node ids
  rollouts.csv   one row per rollout: run_dir, author, benchmark, task, label fields (verdict yes/no/split/unclear, cheat 1/0/
                 empty, verdict_enacted, cheat_enacted, categories, ...), token and turn counts, first_row/last_row into X
  manifest.json  code_sha set, model, layers, counts, sha256 and bytes of every file (written last: the part's done marker)
<snap>/index.json lists every packed run_dir/rollout_id.

    python goodfire_snapshot.py build --acts /workspace/acts --snap /workspace/snapshots        # on the GPU host
    python goodfire_snapshot.py verify --snap DIR                                               # checksums + shapes
    from goodfire_snapshot import load; turns, rollouts, X = load(DIR, layers=(51,))           # anywhere (memmap)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np

LAYERS = (26, 38, 51)
ROLLOUT_FIELDS = ["key", "run_dir", "rollout_id", "author", "author_name", "author_model", "benchmark", "task", "label_source",
                  "verdict", "cheat", "verdict_enacted", "cheat_enacted", "categories", "split", "reward", "shortcut_signal",
                  "status", "any_certain_considered", "any_certain_attempted", "any_certain_enacted", "n_passages",
                  "n_passages_positive", "n_passages_certain", "n_passages_ambiguous", "n_tokens", "n_turns", "n_readable",
                  "n_unsampled_turns", "think_split_from_reasoning", "think_from_content", "ok", "stop_condition",
                  "first_row", "last_row", "code_sha", "gdn_kernel"]
TURN_FIELDS = ["row", "key", "run_dir", "rollout_id", "turn", "open_pos", "close_pos", "readable", "sampled", "turn_node",
               "turn_path_node"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _binary(v):
    return {"yes": 1, "no": 0}.get(v, "")


def _write_atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build(acts, snap, max_rollouts=0):
    """Pack the finished, not yet packed rollouts into the next part. Returns the part directory or None."""
    from safetensors import safe_open
    os.makedirs(snap, exist_ok=True)
    index_path = os.path.join(snap, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {"packed": [], "parts": []}
    packed = set(index["packed"])
    new = sorted(os.path.relpath(os.path.dirname(p), acts) for p in glob.glob(os.path.join(acts, "*", "*", "meta.json")))
    new = [k for k in new if k not in packed]
    if max_rollouts:
        new = new[:max_rollouts]
    if not new:
        return None
    part = os.path.join(snap, f"part_{len(index['parts']):04d}")
    os.makedirs(part, exist_ok=False)
    xs = {L: [] for L in LAYERS}
    rollouts, turns, row = [], [], 0
    shas, models, kernels = set(), set(), set()
    for key in new:
        d = os.path.join(acts, key)
        meta = json.load(open(os.path.join(d, "meta.json")))
        if tuple(meta["layers"]) != LAYERS:
            raise RuntimeError(f"{key}: layers {meta['layers']}")
        with safe_open(os.path.join(d, "acts.safetensors"), framework="np") as f:
            pos = f.get_tensor("think_pos")
            readable = f.get_tensor("think_readable")
            arr = {L: f.get_tensor(f"think_L{L}") for L in LAYERS}
        n = pos.shape[0]
        for L in LAYERS:
            if arr[L].shape != (n, 2, 5120) or arr[L].dtype != np.float16 or not np.isfinite(arr[L]).all():
                raise RuntimeError(f"{key}: think_L{L} {arr[L].shape} {arr[L].dtype}")
            xs[L].append(arr[L])
        tn, tpn, ts = meta.get("turn_nodes") or [], meta.get("turn_path_nodes") or [], meta.get("turn_sampled") or []
        if not (len(tn) == len(tpn) == len(ts) == n == meta["n_turns"]):
            raise RuntimeError(f"{key}: turn lists {len(tn)}/{len(tpn)}/{len(ts)} vs {n} turns")
        for k in range(n):
            turns.append({"row": row + k, "key": key, "run_dir": meta["run_dir"], "rollout_id": meta["rollout_id"], "turn": k,
                          "open_pos": int(pos[k, 0]), "close_pos": int(pos[k, 1]), "readable": int(readable[k]),
                          "sampled": int(bool(ts[k])), "turn_node": tn[k], "turn_path_node": tpn[k]})
        r = {f: meta.get(f, "") for f in ROLLOUT_FIELDS}
        r.update(key=key, cheat=_binary(meta.get("verdict")), cheat_enacted=_binary(meta.get("verdict_enacted")),
                 categories="|".join(meta.get("categories") or []), n_unsampled_turns=sum(1 for s in ts if not s),
                 first_row=row, last_row=row + n - 1 if n else row - 1)
        rollouts.append({k: ("" if v is None else v) for k, v in r.items()})
        row += n
        shas.add(meta.get("code_sha"))
        models.add(meta.get("model"))
        kernels.add(meta.get("gdn_kernel"))
    think_compatible = {"f4672813f425c6a5ba627198b720e7eb810883ba",  # later commits add turn_L only; think_L identical
                        "5724bb3f32272197c0a6a3e0c7456ebf85278618", "92d1e1426c75e283e59050402a7783b39d09f11e"}
    if len(shas - think_compatible) > 1 or len(kernels) != 1:
        raise RuntimeError(f"mixed provenance in one part: code_sha {shas} gdn_kernel {kernels}")
    files = {}
    for L in LAYERS:
        p = os.path.join(part, f"X_L{L}.npy")
        np.save(p, np.concatenate(xs[L]) if xs[L] else np.zeros((0, 2, 5120), np.float16))
    for name, fields, rows in (("turns.csv", TURN_FIELDS, turns), ("rollouts.csv", ROLLOUT_FIELDS, rollouts)):
        with open(os.path.join(part, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    for fn in sorted(os.listdir(part)):
        p = os.path.join(part, fn)
        files[fn] = {"bytes": os.path.getsize(p), "sha256": _sha256(p)}
    _write_atomic_json(os.path.join(part, "manifest.json"),
                       {"dataset": "Goodfire/reward-hack-data", "code_sha": shas.pop(), "model": sorted(models),
                        "gdn_kernel": kernels.pop(), "layers": list(LAYERS), "n_rollouts": len(rollouts), "n_turns": row,
                        "dtype": "float16", "files": files})
    index["packed"] += new
    index["parts"].append(os.path.basename(part))
    _write_atomic_json(index_path, index)
    return part


def verify(snap):
    """Checksums, shapes and row bookkeeping of every part. Returns the number of broken parts."""
    bad = 0
    for part in sorted(glob.glob(os.path.join(snap, "part_*"))):
        problems = []
        mpath = os.path.join(part, "manifest.json")
        if not os.path.exists(mpath):
            print("INCOMPLETE", part)
            bad += 1
            continue
        m = json.load(open(mpath))
        for fn, info in m["files"].items():
            p = os.path.join(part, fn)
            if not os.path.exists(p) or os.path.getsize(p) != info["bytes"] or _sha256(p) != info["sha256"]:
                problems.append(f"{fn} checksum")
        turns = list(csv.DictReader(open(os.path.join(part, "turns.csv"))))
        rollouts = list(csv.DictReader(open(os.path.join(part, "rollouts.csv"))))
        for L in m["layers"]:
            x = np.load(os.path.join(part, f"X_L{L}.npy"), mmap_mode="r")
            if x.shape != (m["n_turns"], 2, 5120):
                problems.append(f"X_L{L} shape {x.shape}")
        if len(turns) != m["n_turns"] or len(rollouts) != m["n_rollouts"]:
            problems.append("row counts")
        if sum(int(r["n_turns"]) for r in rollouts) != m["n_turns"]:
            problems.append("rollout n_turns do not sum to the turn rows")
        print(("BROKEN " + "; ".join(problems)) if problems else "ok", os.path.basename(part), m["n_rollouts"], "rollouts",
              m["n_turns"], "turns", m["code_sha"][:8])
        bad += bool(problems)
    return bad


def load(snap, layers=LAYERS):
    """(turns, rollouts, {L: X}) over every complete part: lists of dicts and float16 arrays [n_turns, 2, 5120]
    (memory-mapped per part, concatenated). turns[i] describes X[L][i]."""
    turns, rollouts, xs = [], [], {L: [] for L in layers}
    for part in sorted(glob.glob(os.path.join(snap, "part_*"))):
        if not os.path.exists(os.path.join(part, "manifest.json")):
            continue
        base = len(turns)
        t = list(csv.DictReader(open(os.path.join(part, "turns.csv"))))
        for r in t:
            r["row"] = int(r["row"]) + base
        r_ = list(csv.DictReader(open(os.path.join(part, "rollouts.csv"))))
        for r in r_:
            r["first_row"], r["last_row"] = int(r["first_row"]) + base, int(r["last_row"]) + base
        turns += t
        rollouts += r_
        for L in layers:
            xs[L].append(np.load(os.path.join(part, f"X_L{L}.npy"), mmap_mode="r"))
    return turns, rollouts, {L: (np.concatenate(v) if v else np.zeros((0, 2, 5120), np.float16)) for L, v in xs.items()}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--acts", required=True)
    b.add_argument("--snap", required=True)
    b.add_argument("--max-rollouts", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--snap", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        p = build(a.acts, a.snap, a.max_rollouts)
        print("nothing new" if p is None else f"wrote {p}: {json.load(open(os.path.join(p, 'manifest.json')))['n_rollouts']} rollouts")
    else:
        sys.exit(1 if verify(a.snap) else 0)


if __name__ == "__main__":
    main()
