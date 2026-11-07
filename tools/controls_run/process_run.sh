#!/bin/zsh
# process_run.sh <apple repo dir> <study> <run>: harvest to Mac and wild, publish the run's files, start wild's pipeline
SCR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
SRC=$1; S=$2; R=$3
ssh -o LogLevel=ERROR arena9-apple "bash -c 'pgrep -f -- \"--name $R\" || pgrep -f \"seed=${R##*_s}\" >/dev/null; true'" >/dev/null
$SCR/harvest.sh $SRC $S $R || exit 1
cd /Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents
K=$(ssh -o LogLevel=ERROR arena9-wild 'sed -n "s/^export OPENROUTER_API_KEY=//p" /root/.bashrc | tr -d "\"'"'"'" | head -1'); export WILD_OPENROUTER_KEY="$K"; unset K
uv run python -m elicitation.publish $S 2>&1 | tail -1
ssh -o LogLevel=ERROR arena9-wild "cd /root/work_dontpressure && nohup ./pipeline_run.sh $S $R > pipe_$R.log 2>&1 & echo pipeline started for $R"
