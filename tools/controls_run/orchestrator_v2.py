#!/usr/bin/env python3
"""Multi-seed breadth-first elicitation orchestrator for study_apart_controls.

Up to SEEDS seeds per cell; breadth-first (seed k for all schedulable cells before seed k+1).
Each run code-capped at cost_limit_usd=0.60. HARD STOP on real cumulative billed_usd >= $60
(stop launching at $55 reserving 0.60/inflight). Keys pulled via ssh stdin, validated by last4,
never logged. Failures retried once, then dropped. vuls is folded in automatically once its image
exists on a host. Per-seed completion prints SEED_COMPLETE k; terminal prints ALL_COMPLETE.
"""
from __future__ import annotations
import json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

def pmap(fn, items, workers=32):
    """Run fn over items in parallel threads; return {item: result}. Keeps ticks fast at many hosts."""
    items = list(items)
    if not items:
        return {}
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return dict(zip(items, ex.map(fn, items)))
STATE = HERE / "apart_state.json"
REPO = "/root/arena-capstone/submodule/spar-guardian-agents"
STUDY = "study_apart_controls"
COST_CAP = 0.60
TURN_BUDGET = 25
LOGDIR = "/root/apart_logs"
SEEDS = 50
MAX_ATTEMPTS = 2
CEILING_STOP = 190.0   # stop launching new runs past this (true-$ soft target)
CEILING_HARD = 200.0   # absolute hard stop (true $)
DUP_OFFSET = 17.5      # one-time duplicate-copy spend from the 4-orchestrator window, absent from state
RESERVE_PER_INFLIGHT = 0.10  # realistic aggregate reserve/inflight (avg ~$0.04, cap $0.60); NOT the cap
# NOTE: reserving the full $0.60 cap per inflight run was a phantom worst case that pinned inflight to
# ~(CEILING-cumulative)/0.60 and starved hosts. Real avg is ~$0.04; a $0.10 reserve is safely conservative.

CONFIG_FILE = HERE / "orch_config.json"
FORCE_RQ_FILE = HERE / "force_requeue.json"   # dedicated: only the operator writes it, orchestrator clears it
# conc_total: max concurrent runs per host; swe_cap: max concurrent CPU-heavy SWE runs (flipt/vuls);
# min_free_mb: skip launching on a host whose MemAvailable is below this.
# Live config is read from CONFIG_FILE each tick so hosts/keys can be added WITHOUT a restart.
DEFAULT_HOSTS = {
    "apple": {"ssh": "arena9-apple", "conc_total": 6, "swe_cap": 3, "min_free_mb": 3000},
    "gabby": {"ssh": "arena9-gabby", "conc_total": 6, "swe_cap": 3, "min_free_mb": 3000},
    "bloom": {"ssh": "arena9-bloom", "conc_total": 7, "swe_cap": 0, "min_free_mb": 3000},
    "flash": {"ssh": "arena9-flash", "conc_total": 7, "swe_cap": 0, "min_free_mb": 3000},
    "iron":  {"ssh": "arena9-iron", "conc_total": 7, "swe_cap": 0, "min_free_mb": 3000},
}
SWE_CLASSES = {"flipt", "vuls"}
CLASS_IMAGES = {
    "flipt": ["cbswe-flipt-02e2"], "python311": ["cbtask:python311"],
    "kw": ["cbkw:office", "cbkw:agent"], "cw": ["cbcw:agent", "cbcw:review"],
    "vuls": ["cbswe-vuls-1832:native"],
}
# qwen first = launch priority within each cell (the target on-policy model)
MODELS = {"qwen": "qwen3.8-27b-openrouter",
          "haiku": "claude-haiku-5-5-openrouter",
          "gpt5m": "gpt-5-mini-openrouter",
          "ds": "deepseek-flash-openrouter",
          "luna": "gpt-6-luna-openrouter"}
TASKS = {
    "flipt":   ("examples.swe", "flipt_solo", "flipt"),
    "vuls":    ("examples.swe", "vuls_solo", "vuls"),
    "riemann": ("examples.openmath", "riemann_solo", "python311"),
    "subset":  ("examples.subset_sum", "subset_solo", "python311"),
    "factor":  ("examples.prime_factorization", "factor_solo", "python311"),
    "binder":  ("examples.protein_design", "binder_solo", "python311"),
    "game":    ("examples.chess", "game_solo", "python311"),
    "poem":    ("examples.creative_writing", "poem_solo", "cw"),
    "worktime":("examples.knowledge_work", "worktime_solo", "kw"),
}
# Vetted IDLE-only borrow pool (Tier 1), larger balance first. Busy run hosts and busy boxes
# (apple/gabby/wild/luna/queen/trouble/silver/autumn) are excluded per the idle-instance check.
DEFAULT_KEY_HOSTS = [
    ("sunset", "4df8"), ("kind", "39de"), ("cloud", "091b"), ("vinyl", "50fb"), ("bonbon", "ff09"),
    ("nurse", "2def"), ("ivy", "72b3"), ("bulk", "99ea"), ("diamond", "3a6e"),
    ("discord", "3daa"), ("derpy", "f073"), ("coco", "f70c"), ("entropy", "69e3"),
    ("gustav", "c8a8"), ("jack", "449a"), ("harmony", "96bc"), ("lyra", "bd8c"),
    ("marble", "8cc2"), ("nova", "0757"), ("octavia", "2e43"), ("unity", "7b0b"), ("wish", "11fd"),
    ("yarn", "3eb2"), ("spike", "87bd"), ("photo", "0da7"), ("lightning", "1573"), ("flutter", "a68a"),
    ("shining", "8468"),
]

def load_config():
    """Read hosts, key pool, and lookahead from CONFIG_FILE each tick (changeable without restart).
    lookahead: how many seeds past the strict breadth-first frontier a host may pull work from when it
    would otherwise idle. Bounds cross-task imbalance while keeping every host saturated."""
    try:
        c = json.loads(CONFIG_FILE.read_text())
        hosts = c.get("hosts") or DEFAULT_HOSTS
        kh = c.get("key_hosts")
        key_hosts = [tuple(x) for x in kh] if kh else DEFAULT_KEY_HOSTS
        lookahead = int(c.get("lookahead", 6))
        pause_models = list(c.get("pause_models", []))
        # force_requeue lives in its OWN file so the key/disk guards (which read-modify-write
        # orch_config.json) can never clobber it in a race.
        force_rq = []
        if FORCE_RQ_FILE.exists():
            try:
                force_rq = list(json.loads(FORCE_RQ_FILE.read_text()))
            except Exception:
                force_rq = []
        return hosts, key_hosts, lookahead, force_rq, pause_models
    except Exception:
        return DEFAULT_HOSTS, DEFAULT_KEY_HOSTS, 6, [], []

def clear_force_requeue():
    try:
        FORCE_RQ_FILE.write_text("[]")
    except Exception:
        pass

def sh(args, input_bytes=None, timeout=60):
    try:
        return subprocess.run(args, input=input_bytes, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 1, b"", b"timeout")

def ssh_out(alias, cmd, timeout=60):
    # alias may be a plain Host alias ("arena9-apple") or full target tokens ("-p 30738 root@ssh9.vast.ai")
    r = sh(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", *alias.split(), cmd], timeout=timeout)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")

_KEY_CACHE: dict[str, bytes] = {}
def get_key(key_host, last4):
    if key_host in _KEY_CACHE:
        return _KEY_CACHE[key_host]
    r = sh(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", f"arena9-{key_host}",
            "zsh -lic 'printf %s \"$OPENROUTER_API_KEY\"'"], timeout=30)
    tok = ""
    for part in r.stdout.decode("utf-8", "replace").replace("\r", "\n").split():
        if part.startswith("sk-or-v1-"):
            tok = part
    if not (tok.startswith("sk-or-v1-") and tok.endswith(last4)):
        return None
    _KEY_CACHE[key_host] = tok.encode()
    return _KEY_CACHE[key_host]

def host_image_classes(alias):
    rc, out, _ = ssh_out(alias, "docker images --format '{{.Repository}}:{{.Tag}}'", timeout=40)
    present = set(out.splitlines())
    return {cls for cls, imgs in CLASS_IMAGES.items()
            if all(any(i in p for p in present) for i in imgs)}

def host_done_markers(alias):
    rc, out, _ = ssh_out(alias, f"grep -h MARKER_DONE {LOGDIR}/*.log 2>/dev/null", timeout=40)
    done = {}
    for line in out.splitlines():
        parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
        if "name" in parts:
            done[parts["name"]] = {"rc": int(parts.get("rc", -1)), "wall_s": int(parts.get("wall_s", 0))}
    return done

def is_credit_err(alias, name):
    """True if the run's log shows an OpenRouter credit/402 failure (key depleted). Empirical key-death
    signal, independent of the laggy /api/v1/key balance."""
    rc, out, _ = ssh_out(alias, f"grep -hiE '402|requires more credits|openrouter_key_limit|insufficient' "
                                f"{LOGDIR}/{name}.log 2>/dev/null | head -1", timeout=20)
    return bool(out.strip())

def host_free_mb(alias):
    rc, out, _ = ssh_out(alias, "awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo", timeout=30)
    try:
        return int(out.strip())
    except ValueError:
        return 0

def fetch_billed(alias, name, repo=REPO):
    cmd = (f"cd {repo} && .venv/bin/python -c \""
           f"import json,glob;"
           f"fs=glob.glob('runs/{STUDY}/{name}/episodes.jsonl');"
           f"print(json.loads(open(fs[0]).readline())['ledger']['billed_usd']) if fs else print('')\" 2>/dev/null")
    rc, out, _ = ssh_out(alias, cmd, timeout=30)
    out = out.strip()
    try:
        return float(out)
    except ValueError:
        return None

def launch(alias, name, module, model, key_bytes, replace, repo=REPO):
    rep = " --replace" if replace else ""
    # chess (game) and openmath (riemann) are long-horizon; raise the per-sample time limit UNIFORMLY
    # for all 5 models so qwen's slower play finishes (only qwen ever hit the 3600 default). Keeps
    # cross-model comparability (fast models finish <2500s either way).
    tl = " --set time_limit_s=5400" if (".chess.game" in module or ".openmath.riemann" in module) else ""
    remote = f"""
IFS= read -r K
export OPENROUTER_API_KEY="$K"
unset K
cd {repo} || exit 9
export STUDY={STUDY}
export RH_PY311={repo}/.venv/bin/python
mkdir -p {LOGDIR}
setsid bash -c 't0=$(date +%s); .venv/bin/python -m {module} --model {model} --no-judge --set cost_limit_usd={COST_CAP} --set turn_budget={TURN_BUDGET}{tl} --name {name}{rep} >> {LOGDIR}/{name}.log 2>&1; RC=$?; t1=$(date +%s); echo "MARKER_DONE name={name} rc=$RC wall_s=$((t1-t0))" >> {LOGDIR}/{name}.log' </dev/null >/dev/null 2>&1 &
echo LAUNCHED {name}
"""
    r = sh(["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", *alias.split(), remote],
           input_bytes=key_bytes + b"\n", timeout=60)
    return r.returncode == 0 and b"LAUNCHED" in r.stdout

def build_manifest():
    runs = {}
    for seed in range(1, SEEDS + 1):
        for task, (mdir, stem, cls) in TASKS.items():
            for cell in ("solo", "cheat"):
                stem_cell = stem if cell == "solo" else stem + "_cheat"
                module = f"{mdir}.{stem_cell}"
                for mk, mins in MODELS.items():
                    # qwen chess is ~2x slower (deepinfra/bf16) and long-horizon; cap its depth so it
                    # never throttles the main matrix. Other models/tasks run the full seed range.
                    if task == "game" and mk == "qwen" and seed > 12:
                        continue
                    name = f"{task}_{cell}_{mk}_s{seed}"
                    runs[name] = {"task": task, "cell": cell, "model_key": mk, "model": mins,
                                  "module": module, "image_class": cls, "seed": seed,
                                  "status": "queued", "host": None, "key_host": None,
                                  "rc": None, "wall_s": None, "attempts": 0, "billed_usd": None}
    return runs

def merge_old(runs):
    """Carry seed-1 progress from a prior single-seed state into the new manifest."""
    if not STATE.exists():
        return 6
    old = json.loads(STATE.read_text())
    for name, oi in old.get("runs", {}).items():
        if name in runs:
            for f in ("status", "host", "key_host", "rc", "wall_s", "launched_at", "billed_usd", "attempts"):
                if f in oi:
                    runs[name][f] = oi[f]
            runs[name].setdefault("attempts", 0)
            runs[name]["seed"] = runs[name].get("seed", 1)
    return old.get("key_idx", 6)

TERMINAL = {"done", "dropped"}

def main():
    runs = build_manifest()
    key_idx = merge_old(runs)
    state = {"runs": runs, "key_idx": key_idx, "started": time.time(), "seeds_announced": []}
    bad_keys = set()   # in-memory: key-hosts that actually 402'd this orchestrator's lifetime; never re-handed out
    hard_stopped = False

    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps({"hosts": DEFAULT_HOSTS, "key_hosts": DEFAULT_KEY_HOSTS}, indent=2))

    # startup: fetch billed for already-done runs
    for name, info in runs.items():
        if info["status"] == "done" and info["billed_usd"] is None and info["host"]:
            _h0, _k0, _l0, _f0, _p0 = load_config()
            _cfg0 = _h0.get(info["host"], {})
            info["billed_usd"] = fetch_billed(_cfg0.get("ssh", f"arena9-{info['host']}"), name,
                                              _cfg0.get("repo", REPO))

    while True:
        hosts, key_hosts, lookahead, force_rq, pause_models = load_config()
        aliases = {h: cfg["ssh"] for h, cfg in hosts.items()}
        # operator-driven force re-queue of DROPPED runs (config "force_requeue": [run names]).
        # attempts=1 so the relaunch uses --replace (a partial dir exists) and keeps one more retry.
        if force_rq:
            for name in force_rq:
                info = runs.get(name)
                # explicit operator list: requeue regardless of status (incl "done" runs that exited
                # rc=0 but whose SAMPLE actually timed out / was invalid — rc=0 != valid).
                if info:
                    h = info.get("host")
                    if h in hosts:
                        ssh_out(aliases[h], f": > {LOGDIR}/{name}.log", timeout=20)  # drop stale marker
                    info.update(status="queued", attempts=1, rc=None, wall_s=None)
                    print(f"FORCE_REQUEUE {name}", flush=True)
            clear_force_requeue()
        repos = {h: cfg.get("repo", REPO) for h, cfg in hosts.items()}
        # parallel per-host probes (keeps ticks fast at many hosts)
        host_classes = pmap(lambda h: host_image_classes(aliases[h]), hosts)
        markers_by_host = pmap(lambda h: host_done_markers(aliases[h]), hosts)
        host_free = pmap(lambda h: host_free_mb(aliases[h]), hosts)
        all_caps = set().union(*host_classes.values()) if host_classes else set()

        # collect newly-finished runs, fetch their billed in parallel, then apply status
        newly = [name for name, info in runs.items()
                 if info["status"] == "launched" and info["host"] in markers_by_host
                 and name in markers_by_host[info["host"]]]
        billed = pmap(lambda n: fetch_billed(aliases[runs[n]["host"]], n, repos[runs[n]["host"]]), newly)
        fails = [n for n in newly if markers_by_host[runs[n]["host"]][n]["rc"] != 0]
        credit = pmap(lambda n: is_credit_err(aliases[runs[n]["host"]], n), fails)
        for name in newly:
            info = runs[name]
            m = markers_by_host[info["host"]][name]
            info["rc"] = m["rc"]; info["wall_s"] = m["wall_s"]; info["billed_usd"] = billed.get(name)
            if m["rc"] == 0:
                info["status"] = "done"
            elif credit.get(name):
                # 402 / key-credit failure: $0 billed, not a real attempt. Blacklist the key immediately
                # (empirical, lag-proof) and requeue WITHOUT counting toward the drop limit.
                if info.get("key_host"):
                    bad_keys.add(info["key_host"])
                info["status"] = "queued"
                ssh_out(aliases[info["host"]], f": > {LOGDIR}/{name}.log", timeout=20)
                print(f"KEY_402 {name} key={info.get('key_host')} blacklisted; requeued", flush=True)
            else:
                info["attempts"] = info.get("attempts", 0) + 1
                if info["attempts"] >= MAX_ATTEMPTS:
                    info["status"] = "dropped"
                else:
                    info["status"] = "queued"
                    ssh_out(aliases[info["host"]], f": > {LOGDIR}/{name}.log", timeout=20)
        # stuck-launched liveness guard (OPERATIONS 8.2): a run launched > STUCK_S ago with no marker
        # is orphaned (dead host / lost marker, past the 90-min Inspect time_limit) -> requeue it so
        # it re-runs elsewhere and ALL_COMPLETE can be reached. Only when its host is reachable this tick.
        _now = time.time()
        for name, info in runs.items():
            if info["status"] != "launched":
                continue
            h = info.get("host")
            if h not in markers_by_host:   # host unreachable this tick: don't misjudge; wait
                continue
            if name in markers_by_host[h]:  # has a (terminal) marker: handled above
                continue
            stuck_s = (140 * 60) if info.get("task") == "game" else (105 * 60)  # chess runs legitimately longer
            if (_now - info.get("launched_at", _now)) > stuck_s:
                info["attempts"] = info.get("attempts", 0) + 1
                ssh_out(aliases[h], f": > {LOGDIR}/{name}.log", timeout=20)
                info["status"] = "dropped" if info["attempts"] >= MAX_ATTEMPTS else "queued"
                print(f"STUCK_REQUEUE {name} host={h} attempts={info['attempts']} -> {info['status']}", flush=True)
        host_inflight = {h: 0 for h in hosts}
        host_swe = {h: 0 for h in hosts}
        for h in hosts:
            host_inflight[h] = sum(1 for i in runs.values() if i["host"] == h and i["status"] == "launched")
            host_swe[h] = sum(1 for i in runs.values() if i["host"] == h and i["status"] == "launched"
                              and i["image_class"] in SWE_CLASSES)

        cumulative = sum(i["billed_usd"] or 0.0 for i in runs.values() if i["billed_usd"] is not None)
        inflight_total = sum(host_inflight.values())
        if cumulative + DUP_OFFSET >= CEILING_HARD:
            hard_stopped = True
            print(f"HARD_STOP cumulative=${cumulative:.3f}", flush=True)

        def schedulable(info):
            return info["image_class"] in all_caps
        # launch frontier: a seed is "open" once it has no queued schedulable runs left (all launched
        # or terminal). Keeps every host slot filled instead of draining between seeds, while
        # preserving breadth-first: every cell STARTS seed k before any cell starts seed k+1.
        # frontier EXCLUDES qwen-chess: it's capped+slow+deprioritized, so it must not hold active_seed
        # back and throttle the fast main matrix. qwen-chess fills opportunistically within the window.
        def in_matrix(i):
            return not (i["task"] == "game" and i["model_key"] == "qwen")
        active_seed = 1
        for s in range(1, SEEDS + 1):
            sched = [i for i in runs.values() if i["seed"] == s and schedulable(i) and in_matrix(i)
                     and i["model_key"] not in pause_models]
            if sched and any(i["status"] == "queued" for i in sched):
                active_seed = s
                break
            active_seed = s + 1
        active_seed = min(active_seed, SEEDS)
        # completion announcements: contiguous fully-terminal seeds (for per-seed sweep notes)
        for s in range(1, SEEDS + 1):
            sched = [i for i in runs.values() if i["seed"] == s and schedulable(i)]
            if sched and all(i["status"] in TERMINAL for i in sched):
                if s not in state["seeds_announced"]:
                    state["seeds_announced"].append(s)
                    print(f"SEED_COMPLETE {s}", flush=True)
            elif sched:
                break

        if not hard_stopped:
            progressed = True
            # qwen chess runs LAST (stable sort keeps breadth-first order otherwise), so it fills slots
            # only opportunistically and never throttles the main matrix.
            ordered = sorted(runs.items(), key=lambda kv: (kv[1]["task"] == "game" and kv[1]["model_key"] == "qwen"))
            while progressed:
                progressed = False
                for name, info in ordered:
                    if info["status"] != "queued" or info["seed"] > active_seed + lookahead:
                        continue
                    if info["model_key"] in pause_models:   # model paused pending spend authorization
                        continue
                    cls = info["image_class"]
                    for h, cfg in hosts.items():
                        if cls not in host_classes.get(h, set()):
                            continue
                        if host_inflight[h] >= cfg["conc_total"]:
                            continue
                        if cls in SWE_CLASSES and host_swe[h] >= cfg["swe_cap"]:
                            continue
                        if host_free[h] < cfg["min_free_mb"]:
                            continue
                        if cumulative + DUP_OFFSET + RESERVE_PER_INFLIGHT * (inflight_total + 1) >= CEILING_STOP:
                            progressed = False
                            break
                        key_bytes = key_host = None
                        for _ in range(len(key_hosts)):
                            kh, last4 = key_hosts[state["key_idx"] % len(key_hosts)]
                            state["key_idx"] += 1
                            if kh in bad_keys:   # key that actually 402'd: never re-hand out
                                continue
                            kb = get_key(kh, last4)
                            if kb is not None:
                                key_bytes, key_host = kb, kh
                                break
                        if key_bytes is None:
                            break
                        if launch(cfg["ssh"], name, info["module"], info["model"], key_bytes,
                                  info.get("attempts", 0) > 0, cfg.get("repo", REPO)):
                            info["status"] = "launched"
                            info["host"] = h
                            info["key_host"] = key_host
                            info["launched_at"] = time.time()
                            host_inflight[h] += 1
                            if cls in SWE_CLASSES:
                                host_swe[h] += 1
                            inflight_total += 1
                            progressed = True
                        break
                    if cumulative + DUP_OFFSET + RESERVE_PER_INFLIGHT * (inflight_total + 1) >= CEILING_STOP:
                        break

        STATE.write_text(json.dumps(state, indent=2))
        done = sum(1 for i in runs.values() if i["status"] == "done")
        dropped = sum(1 for i in runs.values() if i["status"] == "dropped")
        inflight = sum(1 for i in runs.values() if i["status"] == "launched")
        queued = sum(1 for i in runs.values() if i["status"] == "queued")
        print(f"[{time.strftime('%H:%M:%S')}] seed<={active_seed} done={done} dropped={dropped} "
              f"inflight={inflight} queued={queued} cumulative=${cumulative:.3f}", flush=True)

        if inflight == 0 and (queued == 0 or hard_stopped):
            print(f"ALL_COMPLETE done={done} dropped={dropped} cumulative=${cumulative:.3f}", flush=True)
            break
        time.sleep(25)

if __name__ == "__main__":
    main()
