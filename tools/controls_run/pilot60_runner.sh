#!/usr/bin/env bash
# tb=60 iteration: flipt_minimal (access=split), none vs pro-coord/L2, 3 seeds each.
# pilot60_procoordL2_s0 is run separately for pricing; this does the remaining 5.
set -u
REPO=/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents
SCR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
cd "$REPO"; PY=.venv/bin/python; MAXJOBS=2
freegb() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2); print $2+0}'; }
run_one() {
  arm="$1"; seed="$2"
  case "$arm" in
    procoordL2) flags="--set board_seed=pro-coord --set board_level=2" ;;
    none)       flags="" ;;
  esac
  name="pilot60_${arm}_s${seed}"
  # shellcheck disable=SC2086
  $PY -m examples.swe.flipt_minimal --name "$name" --replace --no-judge \
      $flags --set turn_budget=60 --set cost_limit_usd=0.60 --set seed="$seed" \
      > "$SCR/$name.log" 2>&1
  echo "FINISHED $name rc=$?"
}
jobs=("none 0" "none 1" "none 2" "procoordL2 1" "procoordL2 2")
echo "START pilot60: ${#jobs[@]} runs, free=$(freegb)%"
for job in "${jobs[@]}"; do
  while :; do fg=$(freegb); awk "BEGIN{exit !($fg < 12)}" && { echo "MEMLOW ${fg}% waiting"; sleep 20; } || break; done
  echo "LAUNCH $job (free=$(freegb)%)"
  run_one $job &
  while [ "$(jobs -r | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
done
wait
echo "ALL RUNS DONE"
