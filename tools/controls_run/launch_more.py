"""Apple, 2026-11-06: the Qwen solo controls (study_elicitation_6_controls), then more D, A, B, C cell runs
(study_elicitation_6_cells, seeds from 71), all at call_cap 25 and $0.60, one launch at a time under guards:
no launch after 13:30 UTC, total runs in flight <= MAX_TOTAL, MemAvailable >= MIN_AVAIL_MB, and the key's
limit_remaining - 0.60 x (every run in flight + 1) >= $1. Each finished run gets a row with its call_cap in
out/<study>/driver/runs_cap25.jsonl. The key comes on stdin. STOP_MORE next to this file stops launches."""
import calendar, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

SGA = Path("/root/work_dontpressure/sga6")
sys.path.insert(0, str(SGA))
from elicitation import drive_selfhost as d   # noqa: E402

PY = "/root/work_dontpressure/sga/.venv/bin/python"
MODEL = d.OPENROUTER
d.MODEL = MODEL
CAP, CALL_CAP, HEADROOM = 0.60, 25, 1.0
MIN_AVAIL_MB, MAX_TOTAL = 3584, 9
now = time.gmtime(); DEADLINE = calendar.timegm((now.tm_year, now.tm_mon, now.tm_mday, 13, 30, 0, 0, 0, 0))
CONTROLS = [("examples.swe.flipt_solo", "flipt_solo", 61), ("examples.swe.flipt_solo_cheat", "flipt_solo_cheat", 61),
            ("examples.swe.flipt_solo", "flipt_solo", 62), ("examples.swe.flipt_solo_cheat", "flipt_solo_cheat", 62)]
CELLS = [(c, s) for s in range(71, 83) for c in "DABC"]


def log(m):
    print(time.strftime("%H:%M:%S", time.gmtime()), m, flush=True)


def avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024


def in_flight():
    n = 0
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                c = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if MODEL.encode() in c and any(x == b"elicitation.swe" or x.startswith(b"examples.") for x in c):
                n += 1
    return n


def remaining(key):
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return (json.loads(r.read()).get("data") or {}).get("limit_remaining")


def tokens(study, name):
    p = SGA / "runs" / study / name / "run.json"
    if not p.exists():
        return {}
    total = (json.loads(p.read_text()).get("ledger") or {}).get("total") or {}
    t = total.get("tokens") or {}
    return {"calls": total.get("calls"), "input_tokens": t.get("input"), "cache_read_tokens": t.get("cache_read"),
            "output_tokens": t.get("output"), "cap_usd_counted": total.get("cost_usd")}


key = sys.stdin.readline().strip()
if not key.startswith("sk-or-"):
    log("refused: no key on stdin"); sys.exit(2)
base = {k: v for k, v in os.environ.items() if not re.search(r"(KEY|TOKEN|SECRET|^OPENROUTER)", k)}
base["OPENROUTER_API_KEY"] = key
running = {}   # name -> (proc, study, started, extra)


def reap():
    for name, (proc, study, started, extra) in list(running.items()):
        if proc.poll() is None:
            continue
        del running[name]
        row = {"run": name, "study": study, "call_cap": CALL_CAP, "cost_limit_usd": CAP, "rc": proc.returncode,
               "wall_s": round(time.time() - started), **extra}
        if "cell" in extra:
            row.update(d.summary(study, extra["cell"], extra["seed"], proc.returncode, time.time() - started))
            row["call_cap"] = CALL_CAP
        else:
            row.update(tokens(study, name))
        if row.get("input_tokens") is not None:
            row["priced_usd"] = d.priced_usd({k: row.get(k) for k in ("input_tokens", "cache_read_tokens", "output_tokens")})
        out = SGA / "out" / study / "driver"; out.mkdir(parents=True, exist_ok=True)
        with (out / "runs_cap25.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"{name} finished rc={proc.returncode} calls={row.get('calls')} priced=${row.get('priced_usd')}")


def wait_for_slot(name):
    while True:
        reap()
        if time.time() > DEADLINE:
            log(f"past 13:30 UTC: {name} and the rest not launched"); return False
        if (SGA.parent / "STOP_MORE").exists():
            log("STOP_MORE: no new launch"); return False
        try:
            left = remaining(key)
        except Exception as e:
            left = None; log(f"key check failed: {type(e).__name__}")
        a, n = avail_mb(), in_flight()
        ok = left is not None and left - CAP * (n + 1) >= HEADROOM and a >= MIN_AVAIL_MB and n < MAX_TOTAL
        log(f"{name}: avail {a} MB, {n} runs in flight, key limit_remaining ${left}: {'launch' if ok else 'wait'}")
        if ok:
            return True
        time.sleep(60)


def launch(name, argv, study, extra):
    env = {**base, "STUDY": study}
    screen = SGA / "logs" / study / "screen"; screen.mkdir(parents=True, exist_ok=True)
    out = open(screen / f"{name}.log", "w"); out.write(" ".join(argv[1:]) + "\n"); out.flush()
    proc = subprocess.Popen(argv, cwd=SGA, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    running[name] = (proc, study, time.time(), extra)
    log(f"{name} launched pid {proc.pid} ({study}, call_cap {CALL_CAP})")
    time.sleep(45)
    if proc.poll() is not None and proc.returncode != 0:
        reap(); log(f"{name} EXITED rc={proc.returncode} within 45 s: stopping launches"); return False
    return True


go = True
for module, stem, seed in CONTROLS:
    name = f"{stem}_{MODEL}_s{seed}"
    if (SGA / "runs" / "study_elicitation_6_controls" / name).exists():
        continue
    if not (go := wait_for_slot(name)):
        break
    argv = [PY, "-m", module, "--model", MODEL, "--no-judge", "--name", name, "--set", "time_limit_s=7200",
            "--set", f"cost_limit_usd={CAP}", "--set", f"call_cap={CALL_CAP}", "--set", f"seed={seed}"]
    if not (go := launch(name, argv, "study_elicitation_6_controls", {"control": stem, "seed": seed})):
        break
if go:
    for cell, seed in CELLS:
        name = d.run_name(cell, seed)
        if (SGA / "runs" / "study_elicitation_6_cells" / name).exists():
            continue
        if not wait_for_slot(name):
            break
        argv = [a.replace("call_cap=50", f"call_cap={CALL_CAP}") for a in d.command(cell, seed)]
        argv[0] = PY
        assert f"call_cap={CALL_CAP}" in argv and f"cost_limit_usd={CAP}" in argv, argv
        if not launch(name, argv, "study_elicitation_6_cells", {"cell": cell, "seed": seed}):
            break
while running:
    reap(); time.sleep(30)
log("launcher done")
