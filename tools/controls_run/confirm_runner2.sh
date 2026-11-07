#!/usr/bin/env bash
# Compressed dose-response: n=5 (seeds 0-4) x 5 arms = 25 runs. Skips completed runs (run.json present).
# flipt tb=60, DeepSeek-Flash, in-code $0.60/run cap, stop if cumulative billed >$13 (ceiling $15).
# Concurrency 4, memory back-off if free <20%. Seed-major order spreads the heavy anti-coord arm across waves.
set -u
REPO=/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents
SCR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
cd "$REPO"; PY=.venv/bin/python; MAXJOBS=4; MINFREE=20

freep(){ memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2);print $2+0}'; }
cumcost(){ $PY -c '
import json,glob
t=0.0
for f in glob.glob("runs/study_exploration_1/cm_*/run.json"):
    try: t+=json.load(open(f))["ledger"]["total"]["billed_usd"] or 0
    except Exception: pass
print("%.4f"%t)'; }

run_one(){
  arm="$1"; seed="$2"; name="cm_${arm}_s${seed}"
  if [ -f "runs/study_exploration_1/$name/run.json" ]; then echo "SKIP $name (done)"; return; fi
  case "$arm" in
    none)       flags="" ;;
    neutralL0)  flags="--set board_seed=neutral --set board_level=0" ;;
    procoordL1) flags="--set board_seed=pro-coord --set board_level=1" ;;
    procoordL2) flags="--set board_seed=pro-coord --set board_level=2" ;;
    anticoord)  flags="--set n_agents=4 --set regulators=1 --set regulator_level=explicit --set max_concurrency=4" ;;
  esac
  # shellcheck disable=SC2086
  $PY -m examples.swe.flipt_minimal --name "$name" --replace --no-judge $flags \
      --set turn_budget=60 --set cost_limit_usd=0.60 --set seed="$seed" > "$SCR/$name.log" 2>&1
  echo "FINISHED $name rc=$?"
}

jobs=()
for s in 0 1 2 3 4; do for arm in none neutralL0 procoordL1 procoordL2 anticoord; do jobs+=("$arm $s"); done; done
echo "START confirm2: ${#jobs[@]} jobs, MAXJOBS=$MAXJOBS, free=$(freep)%"
for job in "${jobs[@]}"; do
  cc=$(cumcost); awk "BEGIN{exit !($cc>13)}" && { echo "STOP cumulative \$$cc near \$15"; break; }
  while :; do fp=$(freep); awk "BEGIN{exit !($fp<$MINFREE)}" && { echo "MEMLOW ${fp}% waiting"; sleep 30; } || break; done
  run_one $job &
  while [ "$(jobs -r|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
done
wait
echo "CUMCOST $(cumcost)"
echo "ALL RUNS DONE"
