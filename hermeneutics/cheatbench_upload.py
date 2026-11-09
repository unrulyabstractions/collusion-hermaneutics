"""Sync finished CheatBench extraction folders to the Hub and check every uploaded file's bytes there.

Each cycle: for every folder with meta.json that is not in the ledger, redact secret-like strings in meta.json's text
fields (the public traces carry leaked credentials; rendered.txt is already redacted by the converters and is only
checked), upload the folders in one commit, then compare every local file with the Hub (LFS sha256, else git blob sha1)
and append the rollouts whose files all match to the ledger. Local files are kept.

    python -m hermeneutics.cheatbench_upload --acts /root/out/cheatbench --token-file /root/.hf_token \
        --ledger /root/cb_uploaded.jsonl --repo unrulyabstractions/apart-swarm-align --root activations/cheatbench --loop 300
"""
import argparse
import hashlib
import json
import os
import re
import time

SECRET = re.compile(r"sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|xox[abp]-[A-Za-z0-9-]+")


def redact(x):
    """(x with secret matches replaced in every string, count)."""
    if isinstance(x, str):
        y, n = SECRET.subn("[REDACTED]", x)
        return y, n
    if isinstance(x, list):
        out, n = [], 0
        for v in x:
            v, k = redact(v)
            out.append(v)
            n += k
        return out, n
    if isinstance(x, dict):
        out, n = {}, 0
        for key, v in x.items():
            v, k = redact(v)
            out[key] = v
            n += k
        return out, n
    return x, 0


def digest(path):
    data = open(path, "rb").read()
    return hashlib.sha256(data).hexdigest(), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def cycle(api, a, done):
    from huggingface_hub.hf_api import RepoFile
    todo = []
    for h in sorted(os.listdir(a.acts)):
        hd = os.path.join(a.acts, h)
        if not os.path.isdir(hd):
            continue
        for t in sorted(os.listdir(hd)):
            d = os.path.join(hd, t)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")) and (h, t) not in done:
                todo.append((h, t))
    if not todo:
        return 0, 0
    for h, t in todo:
        d = os.path.join(a.acts, h, t)
        if SECRET.search(open(os.path.join(d, "rendered.txt"), errors="replace").read()):
            raise SystemExit(f"{h}/{t}: rendered.txt holds a secret-like string; not uploading")
        meta = json.load(open(os.path.join(d, "meta.json")))
        meta2, n = redact(meta)
        if n:
            meta2["secrets_redacted_in_meta"] = meta.get("secrets_redacted_in_meta", 0) + n
            with open(os.path.join(d, "meta.json.tmp"), "w") as f:
                json.dump(meta2, f, indent=1)
            os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))
    for attempt in range(6):
        try:
            api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=a.acts, path_in_repo=a.root,
                              allow_patterns=[f"{h}/{t}/*" for h, t in todo] + ["_skipped.jsonl", "_failed.jsonl"],
                              ignore_patterns=["*.tmp"], commit_message=f"cheatbench activations: {len(todo)} traces")
            break
        except Exception as e:  # noqa: BLE001
            print(f"upload retry {attempt + 1}: {type(e).__name__} {str(e)[:200]}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        return 0, len(todo)
    remote = {e.path: e for e in api.list_repo_tree(a.repo, path_in_repo=a.root, repo_type="dataset", recursive=True, expand=True)
              if isinstance(e, RepoFile)}
    ok = 0
    with open(a.ledger, "a") as f:
        for h, t in todo:
            d = os.path.join(a.acts, h, t)
            good = True
            for fn in os.listdir(d):
                if fn.endswith(".tmp"):
                    continue
                e = remote.get(f"{a.root}/{h}/{t}/{fn}")
                s256, s1 = digest(os.path.join(d, fn))
                good &= e is not None and (e.lfs.sha256 == s256 if e.lfs else e.blob_id == s1)
            if good:
                f.write(json.dumps({"harness": h, "trace_id": t, "time": time.time()}) + "\n")
                done.add((h, t))
                ok += 1
    return ok, len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=open(a.token_file).read().strip())
    done = set()
    if os.path.exists(a.ledger):
        done = {(r["harness"], r["trace_id"]) for r in map(json.loads, open(a.ledger))}
    while True:
        if os.path.isdir(a.acts):
            ok, n = cycle(api, a, done)
            print(f"{time.strftime('%FT%TZ', time.gmtime())} uploaded {n}, byte-verified {ok}, ledger {len(done)}", flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
