#!/usr/bin/env bash
# Dose-response confirm matrix: flipt_minimal (access=split), DeepSeek-Flash, tb=60.
# Arms: none / neutralL0 / procoordL1 / procoordL2 / anticoord(regulator). n=8 each = 40 runs.
# Caps: in-code $0.60/run; abort if first run >$0.40; stop if cumulative billed >$13 (ceiling $15).
# Concurrency 2, memory-gated (<12% free -> wait). Mac OPENROUTER_API_KEY from env (run under zsh -lic).
set -u
REPO=/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents
SCR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
cd "$REPO"; PY=.venv/bin/python; MAXJOBS=2

freep(){ memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2);print $2+0}'; }
cumcost(){ $PY -c '
import json,glob
t=0.0
for f in glob.glob("runs/study_exploration_1/cm_*/run.json"):
    try: t+=json.load(open(f))["ledger"]["total"]["billed_usd"] or 0
    except Exception: pass
print("%.4f"%t)'; }

run_one(){
  arm="$1"; seed="$2"
  case "$arm" in
    none)       flags="" ;;
    neutralL0)  flags="--set board_seed=neutral --set board_level=0" ;;
    procoordL1) flags="--set board_seed=pro-coord --set board_level=1" ;;
    procoordL2) flags="--set board_seed=pro-coord --set board_level=2" ;;
    anticoord)  flags="--set n_agents=4 --set regulators=1 --set regulator_level=explicit --set max_concurrency=4" ;;
  esac
  name="cm_${arm}_s${seed}"
  # shellcheck disable=SC2086
  $PY -m examples.swe.flipt_minimal --name "$name" --replace --no-judge $flags \
      --set turn_budget=60 --set cost_limit_usd=0.60 --set seed="$seed" > "$SCR/$name.log" 2>&1
  echo "FINISHED $name rc=$?"
}

jobs=()
for arm in none neutralL0 procoordL1 procoordL2 anticoord; do
  for s in 0 1 2 3 4 5 6 7; do jobs+=("$arm $s"); done
done
echo "START confirm: ${#jobs[@]} runs"

# price the first run alone, then gate on it
echo "PRICING ${jobs[0]}"
run_one ${jobs[0]}
fc=$(cumcost); echo "FIRST_COST $fc"
awk "BEGIN{exit !($fc>0.40)}" && { echo "ABORT first run \$$fc exceeds \$0.40"; exit 1; }

for job in "${jobs[@]:1}"; do
  cc=$(cumcost); awk "BEGIN{exit !($cc>13)}" && { echo "STOP cumulative \$$cc near \$15 ceiling"; break; }
  while :; do fp=$(freep); awk "BEGIN{exit !($fp<12)}" && { echo "MEMLOW ${fp}% waiting"; sleep 30; } || break; done
  run_one $job &
  while [ "$(jobs -r|wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
done
wait
echo "CUMCOST $(cumcost)"
echo "ALL RUNS DONE"
