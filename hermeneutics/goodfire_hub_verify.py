"""Content check of the uploaded Goodfire folders against the Hub: every local file of every ledger rollout must exist at
activations/goodfire_reward_hack/<run_dir>/<rollout_id>/ with the same bytes (LFS files: sha256; plain files: git blob
sha1). A size check alone cannot tell new vectors from an old file of the same shape at the same path.

    python -m hermeneutics.goodfire_hub_verify --acts /workspace/acts --ledger /workspace/uploaded.jsonl --token-file F --out R.json
"""
import argparse
import collections
import hashlib
import json
import os

from .goodfire_upload import REPO, ROOT


def git_blob_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    api = HfApi(token=open(a.token_file).read().strip())
    remote = {e.path: e for e in api.list_repo_tree(REPO, path_in_repo=ROOT, repo_type="dataset", recursive=True, expand=True)
              if isinstance(e, RepoFile)}
    ledger = {(r["run_dir"], r["rollout_id"]) for r in map(json.loads, open(a.ledger)) if r}
    cache_path = a.out + ".verified"  # rollouts already verified in an earlier cycle (files are never rewritten after upload)
    cached = {tuple(x) for x in json.load(open(cache_path))} if os.path.exists(cache_path) else set()
    c, bad = collections.Counter(verified=len(cached & ledger)), []
    for rd, rid in sorted(ledger - cached):
        d = os.path.join(a.acts, rd, rid)
        ok = True
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".tmp"):
                continue
            p, e = os.path.join(d, fn), remote.get(f"{ROOT}/{rd}/{rid}/{fn}")
            if e is None:
                ok = False
                bad.append((rd, rid, fn, "missing on hub"))
            elif e.lfs is not None and e.lfs.sha256 != sha256(p):
                ok = False
                bad.append((rd, rid, fn, "sha256 differs"))
            elif e.lfs is None and e.blob_id != git_blob_sha1(p):
                ok = False
                bad.append((rd, rid, fn, "git blob differs"))
        c["verified" if ok else "bad"] += 1
        if ok:
            cached.add((rd, rid))
    json.dump(sorted(cached), open(cache_path + ".tmp", "w"))
    os.replace(cache_path + ".tmp", cache_path)
    json.dump({"repo": REPO, "ledger_rollouts": len(ledger), **c, "bad": bad[:200]}, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)
    print(f"hub verify: {len(ledger)} ledger rollouts, verified {c['verified']}, bad {c['bad']}", flush=True)


if __name__ == "__main__":
    main()
