#!/bin/bash
# Direct host->host LIGHT bootstrap via agent-forwarded ssh (bypasses Mac uplink).
# Usage: boot_light_direct.sh <source-host> <target-host>
# Payload: uv py3.11 + repo tree + submodule git-dirs + cbtask:python311 image.
src="$1"; tgt="$2"; log="probe/boot_$tgt.log"; : > "$log"
tport=$(perl -e 'alarm 10; exec @ARGV' ssh -G "arena9-$tgt" 2>/dev/null | awk '/^port /{print $2}')
echo "[$tgt] DIRECT from=$src start=$(date +%T) tport=$tport" >> "$log"
if [ -z "$tport" ]; then echo "[$tgt] NO PORT - abort" >> "$log"; exit 2; fi
RT="ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=30 -p $tport root@ARENA9_GATEWAY"
A() { perl -e 'alarm 3600; exec @ARGV' ssh -A -n -o ConnectTimeout=10 -o BatchMode=yes "arena9-$src" "$1"; }
echo "[$tgt] prep $(date +%T)" >> "$log"
A "$RT 'mkdir -p /root/arena-capstone/submodule /root/.local/share/uv/python; rm -rf /root/arena-capstone/submodule/spar-guardian-agents'" >> "$log" 2>&1
echo "[$tgt] uv $(date +%T)" >> "$log"
A "tar -C /root/.local/share/uv/python -cf - cpython-3.11.18-linux-x86_64-gnu cpython-3.11-linux-x86_64-gnu | $RT 'cd /root/.local/share/uv/python && tar -xf -'" >> "$log" 2>&1
echo "[$tgt] uv rc=$? repo $(date +%T)" >> "$log"
A "cd /root/arena-capstone/submodule && tar -cf - --exclude=spar-guardian-agents/logs --exclude=spar-guardian-agents/runs --exclude=spar-guardian-agents/out spar-guardian-agents | $RT 'cd /root/arena-capstone/submodule && tar -xf -'" >> "$log" 2>&1
echo "[$tgt] repo rc=$? gitmeta $(date +%T)" >> "$log"
A "cd / && tar -cf - root/arena-capstone/.git/modules/submodule/spar-guardian-agents/modules/submodules | $RT 'cd / && tar -xf -'" >> "$log" 2>&1
echo "[$tgt] gitmeta rc=$? img $(date +%T)" >> "$log"
A "docker save cbtask:python311 | $RT 'docker load'" >> "$log" 2>&1
echo "[$tgt] img rc=$? DONE $(date +%T)" >> "$log"
res=$(bash verify2.sh "$tgt")
echo "$res" > "probe/ready_$tgt.txt"
echo "[$tgt] VERIFY: $res" >> "$log"
