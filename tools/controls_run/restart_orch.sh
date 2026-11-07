#!/bin/bash
# Safely (re)start EXACTLY ONE orchestrator. Kills ALL existing orchestrator python processes by pid
# (not head -1, not a shell-mangled $2) and refuses to start a second if any survive. Run via `bash
# restart_orch.sh` so there is no zsh -c quoting of $2.
set -u
DIR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
cd "$DIR"
python3 -c "import ast; ast.parse(open('orchestrator_v2.py').read())" || { echo "SYNTAX FAIL - abort"; exit 1; }
pids=$(ps -eo pid,args | grep orchestrator_v2.py | grep -iE 'Python|python3' | grep -v zsh | grep -v grep | awk '{print $1}')
for p in $pids; do kill -9 "$p" 2>/dev/null && echo "killed orchestrator $p"; done
sleep 2
left=$(ps -eo args | grep orchestrator_v2.py | grep -iE 'Python|python3' | grep -v zsh | grep -v grep | wc -l | tr -d ' ')
if [ "$left" != "0" ]; then echo "ABORT: $left orchestrator(s) still alive"; exit 1; fi
nohup python3 orchestrator_v2.py >> orch.log 2>&1 &
echo "STARTED single orchestrator pid $!"
