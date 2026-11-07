#!/bin/zsh
OUT=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad/out
rm -rf "$OUT"; mkdir -p "$OUT"

RUN_HOSTS=(apple gabby bloom flash iron cake twilight scoot diamond flutter discord entropy bulk gustav harmony bonbon joy kind jack ember)
GPU_HOSTS=(spitfire zebra moon wild)

RUNCMD='echo "LOAD:$(cat /proc/loadavg 2>/dev/null | awk "{print \$1,\$2,\$3}")"; echo "CORES:$(nproc 2>/dev/null)"; echo "MEM:$(free -g 2>/dev/null | awk "/Mem:/{print \$7}")"; echo "DISK:$(df -h /root 2>/dev/null | awk "NR==2{print \$5\" \"\$4}")"; echo "CTR:$(timeout 15 docker ps --filter name=inspect- -q 2>/dev/null | wc -l | tr -d " ")"; echo "LOG:$(timeout 15 find /root /home -maxdepth 7 -path "*study_apart_controls*" -type f -printf "%T@\n" 2>/dev/null | sort -n | tail -1)"; echo "NOW:$(date +%s)"'

GPUCMD='echo "GPU:$(timeout 15 nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr "\n" ";")"; echo "DISK:$(df -h /root 2>/dev/null | awk "NR==2{print \$5\" \"\$4}")"; echo "CTR:$(timeout 15 docker ps -q 2>/dev/null | wc -l | tr -d " ")"; echo "PROC:$(timeout 15 ps aux 2>/dev/null | grep -iE "inspect|python|vllm|study_apart" | grep -v grep | wc -l | tr -d " ")"; echo "LOAD:$(cat /proc/loadavg 2>/dev/null|awk "{print \$1,\$2,\$3}")"; echo "NOW:$(date +%s)"'

run_one() {
  local h=$1 cmd=$2 tag=$3
  local res
  res=$(perl -e 'alarm 25; exec @ARGV' ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "arena9-$h" "$cmd" 2>"$OUT/$tag-$h.err")
  local rc=$?
  if [[ $rc -ne 0 && -z "$res" ]]; then
    echo "UNREACHABLE rc=$rc $(head -c 200 $OUT/$tag-$h.err | tr '\n' ' ')" > "$OUT/$tag-$h.out"
  else
    echo "$res" > "$OUT/$tag-$h.out"
  fi
}

i=0
for h in $RUN_HOSTS; do
  run_one "$h" "$RUNCMD" run &
  ((i++))
  if (( i % 10 == 0 )); then wait; fi
done
wait
for h in $GPU_HOSTS; do
  run_one "$h" "$GPUCMD" gpu &
done
wait
echo "DONE"
