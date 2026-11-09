#!/bin/bash
# sync_parts.sh : every 30 min, pack a new snapshot part on the pod, pull it to the Mac with runpodctl, unpack, verify.
S=/private/tmp/claude-501/-Users-unrulyabstractions-work-collusion-hermaneutics/f66b8846-aba6-4751-8519-31c57899b754/scratchpad/pod
DST=/Users/unrulyabstractions/work/collusion-hermaneutics/data/goodfire_v2/snapshots
PY=/Users/unrulyabstractions/work/collusion-hermaneutics/.venv/bin/python
SNAP=/Users/unrulyabstractions/work/collusion-hermaneutics/worktree/goodfire-runpod/hermeneutics/goodfire_snapshot.py
while true; do
  free_gb=$(df -g "$DST" | awk 'NR==2{print $4}')
  if [ "$free_gb" -lt 5 ]; then echo "$(date -u +%T) Mac has ${free_gb} GB free: not pulling"; sleep 1800; continue; fi
  out=$($S/podrun.sh 'cd /workspace && /root/venv/bin/python goodfire_snapshot.py build --acts /workspace/acts --snap /workspace/snapshots 2>&1 | tail -1')
  echo "$(date -u +%T) build: $out"
  p=$(echo "$out" | grep -o 'part_[0-9]*' | head -1)
  if [ -n "$p" ]; then
    code=$($S/podrun.sh "cd /workspace/snapshots && tar cf $p.tar $p && tmux new-session -d -s send_$p 'cd /workspace/snapshots && runpodctl send $p.tar > /workspace/logs/send_$p.log 2>&1' && sleep 10 && grep -o 'runpodctl receive [0-9a-z-]*' /workspace/logs/send_$p.log | head -1 | cut -d' ' -f3")
    echo "$(date -u +%T) $p code $code"
    if [ -n "$code" ]; then
      (cd "$DST" && runpodctl receive "$code" > /dev/null 2>&1 && tar xf "$p.tar" && rm -f "$p.tar")
      $PY "$SNAP" verify --snap "$DST" | grep "$p"
      $S/podrun.sh "rm -f /workspace/snapshots/$p.tar" > /dev/null
    fi
  fi
  $S/podrun.sh 'tail -1 /workspace/logs/rounds.log | cut -c1-160; grep -c "^#####" /workspace/logs/rounds.log; ls -d /workspace/acts/*/*/ 2>/dev/null | wc -l'
  sleep 1800
done
