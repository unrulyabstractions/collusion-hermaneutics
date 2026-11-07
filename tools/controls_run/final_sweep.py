#!/usr/bin/env python3
"""BYTE-LEVEL per-host completeness sweep for the published dataset.
For every host in orch_config, `find` every file under the study's produced dirs (runs/, logs/, out/
of study_apart_controls) with its SIZE, and compare bytes against the local copy under
data/apart_controls/<host>/. Reports per host: files swept, missing (only-on-host), size-mismatch.
Exits 0 ONLY if total missing==0 AND size-mismatch==0. (Row/line counts can survive truncation; bytes
cannot — so we compare sizes.) Other host roots (/root, repo, venv, docker images) hold only
reproducible code/caches, not produced data, so the produced data lives entirely in these 3 dirs.
Usage: python3 final_sweep.py            # sweep + print; also appends a VERIFICATION_LOG.md entry
"""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / "orch_config.json"
DATA = Path("/Users/unrulyabstractions/work/arena-capstone/data/apart_controls")
VLOG = Path("/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents/VERIFICATION_LOG.md")
REPO_DEFAULT = "/root/arena-capstone/submodule/spar-guardian-agents"
STUDY = "study_apart_controls"

def remote_list(item):
    h, hcfg = item
    toks = hcfg.get("ssh", f"arena9-{h}").split()
    repo = hcfg.get("repo", REPO_DEFAULT)
    cmd = (f"for s in runs logs out; do find {repo}/$s/{STUDY} -type f -printf '%s\\t'$s'/%P\\n' "
           f"2>/dev/null; done")
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", *toks, cmd],
                           capture_output=True, timeout=180)
    except Exception as e:
        return h, None, [("<ssh error>", str(e))], []
    remote = {}
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        if "\t" in line:
            sz, rel = line.split("\t", 1)
            try:
                remote[rel] = int(sz)
            except ValueError:
                pass
    missing, mism = [], []
    for rel, sz in remote.items():
        lp = DATA / h / rel
        if not lp.exists():
            missing.append(rel)
        elif lp.stat().st_size != sz:
            mism.append((rel, sz, lp.stat().st_size))
    return h, len(remote), missing, mism

def main():
    hosts = json.loads(CFG.read_text())["hosts"]
    with ThreadPoolExecutor(max_workers=min(24, len(hosts))) as ex:
        results = list(ex.map(remote_list, hosts.items()))
    tot_files = tot_missing = tot_mism = 0
    lines = []
    for h, n, missing, mism in sorted(results):
        nf = n or 0
        tot_files += nf; tot_missing += len(missing); tot_mism += len(mism)
        flag = "OK" if (n is not None and not missing and not mism) else "**CHECK**"
        lines.append(f"  {h:10s} swept={nf:5d} missing={len(missing):3d} size_mismatch={len(mism):3d} {flag}")
        for m in missing[:3]:
            lines.append(f"       only-on-host: {m}")
        for rel, rs, ls in mism[:3]:
            lines.append(f"       size-mismatch: {rel} host={rs} local={ls}")
    ok = (tot_missing == 0 and tot_mism == 0 and all(r[1] is not None for r in results))
    hdr = (f"SWEEP {time.strftime('%Y-%m-%d %H:%M:%S')}  hosts={len(hosts)} files_swept={tot_files} "
           f"missing={tot_missing} size_mismatch={tot_mism}  => {'PASS 0/0' if ok else 'FAIL'}")
    print(hdr)
    print("\n".join(lines))
    if "--log" in sys.argv:
        with open(VLOG, "a") as f:
            f.write(f"\n## apart_controls host completeness sweep\n{hdr}\n" + "\n".join(lines) + "\n")
        print("appended to VERIFICATION_LOG.md")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
