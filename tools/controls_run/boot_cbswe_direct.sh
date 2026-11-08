#!/bin/bash
# Direct host->host cbswe transfer via agent-forwarded ssh (bypasses the Mac uplink).
# Usage: boot_cbswe_direct.sh <source-host> <target-host>
# No gzip: docker layers are already compressed; saves source CPU (live run hosts).
src="$1"; tgt="$2"; log="probe/cbswe_${tgt}.log"; : > "$log"
tport=$(perl -e 'alarm 10; exec @ARGV' ssh -G "arena9-$tgt" 2>/dev/null | awk '/^port /{print $2}')
echo "[$tgt] cbswe from=$src start=$(date +%T) tport=$tport" >> "$log"
if [ -z "$tport" ]; then echo "[$tgt] NO PORT - abort" >> "$log"; exit 2; fi
perl -e 'alarm 14400; exec @ARGV' ssh -A -o ConnectTimeout=10 -o BatchMode=yes "arena9-$src" \
  "docker save cbswe-flipt-02e2:native cbswe-vuls-1832:native | ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ServerAliveInterval=30 -p $tport root@ARENA9_GATEWAY 'docker load'" >> "$log" 2>&1
echo "[$tgt] cbswe rc=$? end=$(date +%T)" >> "$log"
# quick verify of images on target
res=$(bash verify2.sh "$tgt")
echo "$res" > "probe/cbswe_ready_${tgt}.txt"
echo "[$tgt] VERIFY: $res" >> "$log"
