#!/usr/bin/env bash
# Board-seed validation pilot: flipt_minimal, DeepSeek-Flash (OpenRouter), cost cap $0.60/run, turn_budget=25.
# Concurrency 2, gated on >6GB free. Logs to scratchpad. OPENROUTER_API_KEY must be in env (run under zsh -lic).
set -u
REPO=/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents
SCR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
cd "$REPO"
PY=.venv/bin/python
MAXJOBS=2

run_one() {
  arm="$1"; seed="$2"
  case "$arm" in
    neutral)    flags="--set board_seed=neutral --set board_level=0" ;;
    procoordL2) flags="--set board_seed=pro-coord --set board_level=2" ;;
    none)       flags="" ;;
  esac
  name="pilot_${arm}_s${seed}"
  # shellcheck disable=SC2086
  $PY -m examples.swe.flipt_minimal --name "$name" --replace --no-judge \
      $flags --set turn_budget=25 --set cost_limit_usd=0.60 --set seed="$seed" \
      > "$SCR/$name.log" 2>&1
  echo "FINISHED $name rc=$?"
}

# free as a system-wide percentage (macOS keeps most RAM in reclaimable 'inactive', so Pages-free is misleading)
freegb() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2); print $2+0}'; }

# jobs (neutral s0 already done; keep it)
jobs=()
for s in 1 2 3 4; do jobs+=("neutral $s"); done
for s in 0 1 2 3 4; do jobs+=("procoordL2 $s"); done
for s in 0 1; do jobs+=("none $s"); done

echo "START pilot: ${#jobs[@]} runs, MAXJOBS=$MAXJOBS, free=$(freegb)%"
for job in "${jobs[@]}"; do
  # memory gate: wait if system-wide free drops under 12%
  while :; do fg=$(freegb); awk "BEGIN{exit !($fg < 12)}" && { echo "MEMLOW ${fg}% waiting"; sleep 20; } || break; done
  echo "LAUNCH $job (free=$(freegb)%)"
  run_one $job &
  while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
done
wait
echo "ALL RUNS DONE"
