#!/bin/bash
# Light-tier bootstrap: uv py3.11 + repo tree + cbtask:python311 image. gabby->Mac->target.
# Producer-side ssh uses -n so it never eats a shared (xargs) stdin. Consumer ssh keeps the pipe.
h="$1"; log="probe/boot_$h.log"; : > "$log"
G="arena9-gabby"; H="arena9-$h"
SN() { perl -e 'alarm 120; exec @ARGV' ssh -n -o ConnectTimeout=10 -o BatchMode=yes "$@"; }  # short cmd, -n
echo "[$h] start $(date +%T)" >>"$log"
SN "$H" 'mkdir -p /root/arena-capstone/submodule /root/.local/share/uv/python; for b in tar gzip docker; do command -v $b >/dev/null||echo MISSING:$b; done' >>"$log" 2>&1
# clean any partial prior tree
SN "$H" 'rm -rf /root/arena-capstone/submodule/spar-guardian-agents' >>"$log" 2>&1
echo "[$h] uv-py $(date +%T)" >>"$log"
perl -e 'alarm 1800; exec @ARGV' ssh -n -o BatchMode=yes "$G" \
  'tar -C /root/.local/share/uv/python -cf - cpython-3.11.18-linux-x86_64-gnu cpython-3.11-linux-x86_64-gnu | gzip -1' \
  | perl -e 'alarm 1800; exec @ARGV' ssh -o BatchMode=yes "$H" \
  'cd /root/.local/share/uv/python && tar -xzf -' >>"$log" 2>&1
echo "[$h] uv rc=$? $(date +%T)" >>"$log"
echo "[$h] repo $(date +%T)" >>"$log"
perl -e 'alarm 1800; exec @ARGV' ssh -n -o BatchMode=yes "$G" \
  'cd /root/arena-capstone/submodule && tar -cf - --exclude=spar-guardian-agents/logs --exclude=spar-guardian-agents/runs --exclude=spar-guardian-agents/out spar-guardian-agents | gzip -1' \
  | perl -e 'alarm 1800; exec @ARGV' ssh -o BatchMode=yes "$H" \
  'cd /root/arena-capstone/submodule && tar -xzf -' >>"$log" 2>&1
echo "[$h] repo rc=$? $(date +%T)" >>"$log"
echo "[$h] gitmeta $(date +%T)" >>"$log"
perl -e 'alarm 600; exec @ARGV' ssh -n -o BatchMode=yes "$G" \
  'cd / && tar czf - root/arena-capstone/.git/modules/submodule/spar-guardian-agents/modules/submodules' \
  | perl -e 'alarm 600; exec @ARGV' ssh -o BatchMode=yes "$H" 'cd / && tar xzf -' >>"$log" 2>&1
echo "[$h] gitmeta rc=$? $(date +%T)" >>"$log"
echo "[$h] img $(date +%T)" >>"$log"
perl -e 'alarm 1800; exec @ARGV' ssh -n -o BatchMode=yes "$G" 'docker save cbtask:python311 | gzip -1' \
  | perl -e 'alarm 1800; exec @ARGV' ssh -o BatchMode=yes "$H" 'gunzip -c | docker load' >>"$log" 2>&1
echo "[$h] img rc=$? $(date +%T)" >>"$log"
echo "[$h] DONE $(date +%T)" >>"$log"
