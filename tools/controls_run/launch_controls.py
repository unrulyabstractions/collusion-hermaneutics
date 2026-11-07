"""Launch the Qwen solo controls on apple, one at a time, under RAM, key and deadline guards. Key on stdin."""
import calendar, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

SGA = Path("/root/work_dontpressure/sga6")
PY = "/root/work_dontpressure/sga/.venv/bin/python"
STUDY = "study_elicitation_6_controls"
MODEL = "qwen3.8-27b-openrouter"
CAP = 0.60
MIN_AVAIL_MB = 4096          # a solo run takes ~1-1.5 GB; the floor stays above 2 GB
HEADROOM = 1.0
now = time.gmtime(); DEADLINE = calendar.timegm((now.tm_year, now.tm_mon, now.tm_mday, 13, 15, 0, 0, 0, 0))
JOBS = [("examples.subset_sum.subset_solo", "subset_solo", 61), ("examples.subset_sum.subset_solo_cheat", "subset_solo_cheat", 61),
        ("examples.swe.flipt_solo", "flipt_solo", 61), ("examples.swe.flipt_solo_cheat", "flipt_solo_cheat", 61),
        ("examples.swe.flipt_solo", "flipt_solo", 62), ("examples.swe.flipt_solo_cheat", "flipt_solo_cheat", 62)]


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
            if b"--model" in c and MODEL.encode() in c and any(x in (b"elicitation.swe",) or x.startswith(b"examples.") for x in c):
                n += 1
    return n


def remaining(key):
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return (json.loads(r.read()).get("data") or {}).get("limit_remaining")


key = sys.stdin.readline().strip()
if not key.startswith("sk-or-"):
    log("refused: no key on stdin"); sys.exit(2)
env = {k: v for k, v in os.environ.items() if not re.search(r"(KEY|TOKEN|SECRET|^OPENROUTER)", k)}
env.update({"OPENROUTER_API_KEY": key, "STUDY": STUDY})
screen = SGA / "logs" / STUDY / "screen"; screen.mkdir(parents=True, exist_ok=True)
for module, stem, seed in JOBS:
    name = f"{stem}_{MODEL}_s{seed}"
    while True:
        if time.time() > DEADLINE:
            log(f"past 13:15 UTC: {name} and the rest not launched"); sys.exit(0)
        if (SGA.parent / "STOP_CONTROLS").exists():
            log("STOP_CONTROLS: no new launch"); sys.exit(0)
        try:
            left = remaining(key)
        except Exception as e:
            left = None; log(f"key check failed: {type(e).__name__}")
        a, n = avail_mb(), in_flight()
        ok = left is not None and left - CAP * (n + 1) >= HEADROOM and a >= MIN_AVAIL_MB
        log(f"{name}: avail {a} MB, {n} runs in flight, key limit_remaining ${left}: {'launch' if ok else 'wait'}")
        if ok:
            break
        time.sleep(60)
    argv = [PY, "-m", module, "--model", MODEL, "--no-judge", "--name", name, "--set", "time_limit_s=7200",
            "--set", f"cost_limit_usd={CAP}", "--set", "call_cap=50", "--set", f"seed={seed}"]
    out = open(screen / f"{name}.log", "w"); out.write(" ".join(argv[1:]) + "\n"); out.flush()
    proc = subprocess.Popen(argv, cwd=SGA, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    log(f"{name} launched pid {proc.pid}")
    time.sleep(60)
    if proc.poll() is not None:
        log(f"{name} EXITED rc={proc.returncode} within 60 s: stopping launches"); sys.exit(1)
log("all controls launched")
