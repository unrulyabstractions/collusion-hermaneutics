"""Upload finished goodfire rollout folders (meta.json present) to the private HF dataset as they land, check every
uploaded file's size against the repo tree, and only then delete the local acts.safetensors (meta.json and spans.jsonl
stay: they are the resume markers). The token is read from a file (mode 600), never from argv or printed.

    python -m hermeneutics.goodfire_upload --acts /root/work_goodfire/acts --token-file /root/work_goodfire/.hf_token \\
        --ledger /root/work_goodfire/uploaded.jsonl --loop 300
"""
import argparse
import json
import os
import time

REPO = "unrulyabstractions/apart-swarm-align"
ROOT = "activations/goodfire_reward_hack"


def pending(acts, done):
    """{run_dir: [rollout_id]} of finished folders not yet verified in the ledger."""
    out = {}
    for rd in sorted(os.listdir(acts)):
        d = os.path.join(acts, rd)
        if not os.path.isdir(d):
            continue
        for rid in sorted(os.listdir(d)):
            f = os.path.join(d, rid)
            if os.path.isdir(f) and os.path.exists(os.path.join(f, "meta.json")) and (rd, rid) not in done \
                    and os.path.exists(os.path.join(f, "acts.safetensors")):
                out.setdefault(rd, []).append(rid)
    return out


def remote_sizes(api, path):
    from huggingface_hub.hf_api import RepoFile
    return {e.path: e.size for e in api.list_repo_tree(REPO, path_in_repo=path, repo_type="dataset", recursive=True)
            if isinstance(e, RepoFile)}


def cycle(api, acts, ledger, done, delete, max_per_commit):
    for rd, rids in pending(acts, done).items():
        for k in range(0, len(rids), max_per_commit):
            chunk = rids[k:k + max_per_commit]
            local = {}
            for rid in chunk:
                for fn in sorted(os.listdir(os.path.join(acts, rd, rid))):
                    local[f"{ROOT}/{rd}/{rid}/{fn}"] = os.path.getsize(os.path.join(acts, rd, rid, fn))
            extra = [f for f in ("_skipped.jsonl",) if os.path.exists(os.path.join(acts, rd, f))]
            for attempt in range(6):
                try:
                    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=os.path.join(acts, rd),
                                      path_in_repo=f"{ROOT}/{rd}", allow_patterns=[f"{r}/*" for r in chunk] + extra,
                                      commit_message=f"goodfire activations: {rd}, {len(chunk)} rollouts")
                    break
                except Exception as e:
                    print(f"{time.strftime('%FT%TZ', time.gmtime())} upload retry {attempt + 1} {rd}: {type(e).__name__} {str(e)[:200]}", flush=True)
                    time.sleep(30 * (attempt + 1))
            else:
                print(f"upload FAILED for {rd} chunk of {len(chunk)}; will retry next cycle", flush=True)
                continue
            remote = remote_sizes(api, f"{ROOT}/{rd}")
            bad = [p for p, s in local.items() if remote.get(p) != s]
            ok = [rid for rid in chunk if not any(p.startswith(f"{ROOT}/{rd}/{rid}/") for p in bad)]
            with open(ledger, "a") as f:
                for rid in ok:
                    sizes = {p.rsplit("/", 1)[1]: s for p, s in local.items() if p.startswith(f"{ROOT}/{rd}/{rid}/")}
                    f.write(json.dumps({"run_dir": rd, "rollout_id": rid, "sizes": sizes, "time": time.time()}) + "\n")
                    done.add((rd, rid))
                    if delete:
                        os.remove(os.path.join(acts, rd, rid, "acts.safetensors"))
            print(f"{time.strftime('%FT%TZ', time.gmtime())} {rd}: uploaded {len(chunk)}, size-verified {len(ok)}, "
                  f"mismatched {len(chunk) - len(ok)} {bad[:3]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--loop", type=int, default=0, help="seconds between cycles; 0 = one cycle")
    ap.add_argument("--keep", action="store_true", help="do not delete verified acts.safetensors")
    ap.add_argument("--max-per-commit", type=int, default=100)
    ap.add_argument("--stop-file", default=None, help="exit after a cycle once this file exists")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=open(a.token_file).read().strip())
    done = set()
    if os.path.exists(a.ledger):
        for line in open(a.ledger):
            r = json.loads(line)
            done.add((r["run_dir"], r["rollout_id"]))
    while True:
        try:
            cycle(api, a.acts, a.ledger, done, not a.keep, a.max_per_commit)
        except Exception as e:
            print(f"{time.strftime('%FT%TZ', time.gmtime())} cycle error {type(e).__name__}: {str(e)[:300]}", flush=True)
        if not a.loop or (a.stop_file and os.path.exists(a.stop_file)):
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
