#!/bin/bash
# run_rounds.sh SHARD N_SHARDS : extract the balanced rounds of this pod's shard, round after round, until stopped.
I=$1; N=$2
cd /root/code/ch
export HERM_CODE_SHA=$(git rev-parse HEAD) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/workspace/data/goodfire
T=""; for rd in impossiblebench-kimi-k3 impossiblebench-glm-5.2-fp8 impossiblebench-qwen3.8-2.4t-a95b-fp8 swebench-verified-kimi-k3-nohack swebench-verified-glm-5.2-nohack swebench-verified-qwen3.8-2.4t-a95b-fp8-nohack shoppingbench-kimi-k3-20260828 shoppingbench-glm-5.2-20260828 tau3bench-kimi-k3 tau3bench-glm-5.2-fp8 deepswe-kimi-k3 deepswe-glm-5.2-fp8 deepswe-qwen3.8-2.4t-a95b-fp8; do T="$T $G/$rd/traces.jsonl"; done
for R in /workspace/rounds/s${I}of${N}/r*; do
  [ -f "$R/.done" ] && continue
  echo "##### $(date -u +%FT%TZ) shard $I/$N round $(basename $R): $(cat $R/*.ids | wc -l) rollouts"
  /root/venv/bin/python -m hermeneutics.goodfire_resident --model /workspace/models/Qwen3.8-27B --out /workspace/acts \
      --max-tokens 262144 --ids-dir "$R" --traces $T || { echo "round $(basename $R) exited $?"; sleep 30; continue; }
  touch "$R/.done"
done
echo "##### all rounds done $(date -u +%FT%TZ)"
