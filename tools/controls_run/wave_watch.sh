#!/bin/zsh
# Start fast_run.sh for each s71/s72 run the apple launcher reports finished, until 14:08 UTC.
SCR=/private/tmp/claude-501/-Users-unrulyabstractions-work-arena-capstone/62859942-35bf-474f-b418-4803b6079982/scratchpad
started=""
while [ $(date -u +%H%M) -lt 1408 ]; do
  for R in $(ssh -o LogLevel=ERROR -o ConnectTimeout=20 arena9-apple "grep -o 'flipt_duo_[^ ]*_s7[12] finished rc=0' /root/work_dontpressure/more2.log | cut -d' ' -f1" 2>/dev/null); do
    case " $started " in *" $R "*) ;; *) started="$started $R"; echo "$(date -u +%T) launching fast_run $R"; nohup $SCR/fast_run.sh $R > $SCR/fast_$R.log 2>&1 &;; esac
  done
  sleep 20
done
echo "$(date -u +%T) watch ended; started:$started"
