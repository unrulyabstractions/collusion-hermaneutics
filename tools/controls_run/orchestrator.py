#!/usr/bin/env python3
"""Maximally-parallel elicitation orchestrator for study_apart_controls.

Launches detached remote runs on arena9 hosts (free, already rented). Keys are pulled from each
key-host's /root/.zshrc over ssh, validated against the known last4, fed to the run ONLY via ssh
stdin, cached in memory, and NEVER logged, written to disk, or placed in argv. Each run is capped in
code at cost_limit_usd=0.60 (turn_budget=25). Completion is read from the remote MARKER_DONE line.

State is written to apart_state.json each tick so the parent agent can observe progress.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / "apart_state.json"
REPO = "/root/arena-capstone/submodule/spar-guardian-agents"
STUDY = "study_apart_controls"
COST_CAP = 0.60
TURN_BUDGET = 25
LOGDIR = "/root/apart_logs"
CEILING_STOP = 55.0   # stop launching new runs past this (reserving 0.60/inflight)
CEILING_HARD = 60.0

HOSTS = {
    "apple": {"ssh": "arena9-apple", "conc": 3},
    "gabby": {"ssh": "arena9-gabby", "conc": 3},
}
# image_class -> docker repo:tag substrings that must be present on a host to run that class
CLASS_IMAGES = {
    "flipt": ["cbswe-flipt-02e2"],
    "python311": ["cbtask:python311"],
    "kw": ["cbkw:office", "cbkw:agent"],
    "cw": ["cbcw:agent", "cbcw:review"],
}
MODELS = {
    "ds": "deepseek-flash-openrouter",
    "luna": "gpt-6-luna-openrouter",
}
# task -> (module_dir, module_stem_solo, image_class)
TASKS = {
    "flipt":   ("examples.swe", "flipt_solo", "flipt"),
    "riemann": ("examples.openmath", "riemann_solo", "python311"),
    "subset":  ("examples.subset_sum", "subset_solo", "python311"),
    "factor":  ("examples.prime_factorization", "factor_solo", "python311"),
    "binder":  ("examples.protein_design", "binder_solo", "python311"),
    "game":    ("examples.chess", "game_solo", "python311"),
    "poem":    ("examples.creative_writing", "poem_solo", "cw"),
    "worktime":("examples.knowledge_work", "worktime_solo", "kw"),
}
# key-host -> expected last4 (from openrouter_key_inventory.csv, remaining >= $2.5)
KEY_HOSTS = [
    ("cloud", "091b"), ("kind", "39de"), ("sunset", "4df8"), ("twilight", "d028"),
    ("orchard", "e5a5"), ("vinyl", "50fb"), ("button", "65d7"), ("bonbon", "ff09"),
    ("gabby", "e2bc"), ("scoot", "eae7"), ("nurse", "2def"), ("cake", "3473"),
]

def sh(args, input_bytes=None, timeout=60):
    return subprocess.run(args, input=input_bytes, capture_output=True, timeout=timeout)

def ssh_out(alias, cmd, timeout=60):
    r = sh(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", alias, cmd], timeout=timeout)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")

_KEY_CACHE: dict[str, bytes] = {}
def get_key(key_host, last4):
    """Return key bytes for key_host, validated against last4. Cached. Never logged."""
    if key_host in _KEY_CACHE:
        return _KEY_CACHE[key_host]
    r = sh(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", f"arena9-{key_host}",
            "zsh -lic 'printf %s \"$OPENROUTER_API_KEY\"'"], timeout=30)
    key = r.stdout.decode("utf-8", "replace").strip()
    # a login shell may emit a banner; take the sk-or-v1 token
    tok = ""
    for part in key.replace("\r", "\n").split():
        if part.startswith("sk-or-v1-"):
            tok = part
    if not (tok.startswith("sk-or-v1-") and tok.endswith(last4)):
        return None
    kb = tok.encode()
    _KEY_CACHE[key_host] = kb
    return kb

def host_image_classes(alias):
    rc, out, _ = ssh_out(alias, "docker images --format '{{.Repository}}:{{.Tag}}'", timeout=40)
    present = set(out.splitlines())
    classes = set()
    for cls, imgs in CLASS_IMAGES.items():
        if all(any(i in p for p in present) for i in imgs):
            classes.add(cls)
    return classes

def host_done_markers(alias):
    rc, out, _ = ssh_out(alias, f"grep -h MARKER_DONE {LOGDIR}/*.log 2>/dev/null", timeout=40)
    done = {}
    for line in out.splitlines():
        # MARKER_DONE name=<name> rc=<n> wall_s=<n>
        parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
        if "name" in parts:
            done[parts["name"]] = {"rc": int(parts.get("rc", -1)), "wall_s": int(parts.get("wall_s", 0))}
    return done

def launch(alias, name, module, model, key_bytes):
    remote = f"""
IFS= read -r K
export OPENROUTER_API_KEY="$K"
unset K
cd {REPO} || exit 9
export STUDY={STUDY}
mkdir -p {LOGDIR}
setsid bash -c 't0=$(date +%s); .venv/bin/python -m {module} --model {model} --no-judge --set cost_limit_usd={COST_CAP} --set turn_budget={TURN_BUDGET} --name {name} >> {LOGDIR}/{name}.log 2>&1; RC=$?; t1=$(date +%s); echo "MARKER_DONE name={name} rc=$RC wall_s=$((t1-t0))" >> {LOGDIR}/{name}.log' </dev/null >/dev/null 2>&1 &
echo LAUNCHED {name}
"""
    r = sh(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", alias, remote],
           input_bytes=key_bytes + b"\n", timeout=60)
    return r.returncode == 0 and b"LAUNCHED" in r.stdout

def build_manifest():
    runs = {}
    for task, (mdir, stem, cls) in TASKS.items():
        for cell in ("solo", "cheat"):
            stem_cell = stem if cell == "solo" else stem + "_cheat"
            module = f"{mdir}.{stem_cell}"
            for mk, mins in MODELS.items():
                name = f"{task}_{cell}_{mk}_s1"
                runs[name] = {"task": task, "cell": cell, "model_key": mk, "model": mins,
                              "module": module, "image_class": cls, "status": "queued",
                              "host": None, "key_host": None, "rc": None, "wall_s": None}
    return runs

def main():
    if STATE.exists():
        state = json.loads(STATE.read_text())
    else:
        state = {"runs": build_manifest(), "key_idx": 0, "started": time.time()}
    runs = state["runs"]

    while True:
        # refresh completion + host image classes
        host_classes = {}
        host_inflight = {h: 0 for h in HOSTS}
        for h, cfg in HOSTS.items():
            host_classes[h] = host_image_classes(cfg["ssh"])
            done = host_done_markers(cfg["ssh"])
            for name, info in runs.items():
                if info["host"] == h and info["status"] == "launched" and name in done:
                    info["status"] = "done"
                    info["rc"] = done[name]["rc"]
                    info["wall_s"] = done[name]["wall_s"]
            for name, info in runs.items():
                if info["host"] == h and info["status"] == "launched":
                    host_inflight[h] += 1

        launched_total = sum(1 for i in runs.values() if i["status"] in ("launched", "done"))
        reserved = launched_total * COST_CAP  # upper bound (code-capped)

        # launch loop
        progressed = True
        while progressed:
            progressed = False
            if reserved >= CEILING_STOP:
                break
            for name, info in runs.items():
                if info["status"] != "queued":
                    continue
                cls = info["image_class"]
                # pick a capable host with a free slot
                for h, cfg in HOSTS.items():
                    if cls not in host_classes.get(h, set()):
                        continue
                    if host_inflight[h] >= cfg["conc"]:
                        continue
                    # pick key (round-robin, validated)
                    key_bytes = None
                    key_host = None
                    for _ in range(len(KEY_HOSTS)):
                        kh, last4 = KEY_HOSTS[state["key_idx"] % len(KEY_HOSTS)]
                        state["key_idx"] += 1
                        kb = get_key(kh, last4)
                        if kb is not None:
                            key_bytes, key_host = kb, kh
                            break
                    if key_bytes is None:
                        break  # no usable key; try next tick
                    if launch(cfg["ssh"], name, info["module"], info["model"], key_bytes):
                        info["status"] = "launched"
                        info["host"] = h
                        info["key_host"] = key_host
                        info["launched_at"] = time.time()
                        host_inflight[h] += 1
                        launched_total += 1
                        reserved = launched_total * COST_CAP
                        progressed = True
                    break  # one host attempt per run per pass

        STATE.write_text(json.dumps(state, indent=2))

        remaining = [n for n, i in runs.items() if i["status"] == "queued"]
        inflight = [n for n, i in runs.items() if i["status"] == "launched"]
        done = [n for n, i in runs.items() if i["status"] == "done"]
        print(f"[{time.strftime('%H:%M:%S')}] done={len(done)} inflight={len(inflight)} "
              f"queued={len(remaining)} reserved_upper=${reserved:.2f}", flush=True)
        if not remaining and not inflight:
            print("ALL_RUNS_COMPLETE", flush=True)
            break
        time.sleep(25)

if __name__ == "__main__":
    main()
