#!/bin/zsh
# pull_run.sh <study> <run>: after PIPE_DONE, pull the run's activations_extra to the Mac and check it against the Hub and wild's sha256
SCR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
S=$1; R=$2
ssh -o LogLevel=ERROR arena9-wild "grep PIPE_ /root/work_dontpressure/pipe_$R.log; cat /root/work_dontpressure/sha_$R.txt" > $SCR/pipe_$R.txt
grep -q PIPE_DONE $SCR/pipe_$R.txt || { echo "not done: $(head -2 $SCR/pipe_$R.txt)"; exit 1; }
grep "^[0-9a-f]\{64\}" $SCR/pipe_$R.txt > $SCR/sha_$R.txt
cd /Users/unrulyabstractions/work/arena-capstone
df -h ~ | tail -1
hf download unrulyabstractions/arena-capstone --repo-type dataset --include "activations_extra/$S/$R/*" --local-dir data/hf --max-workers 4 > /dev/null 2>&1 || { echo "download failed"; exit 1; }
uv run python $SCR/hubcheck.py activations_extra/$S $SCR/sha_$R.txt
