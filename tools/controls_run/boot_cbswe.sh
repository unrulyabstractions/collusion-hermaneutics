#!/bin/bash
# SWE-tier: transfer flipt + vuls images (all tags, deduped) gabby->Mac->target.
h="$1"; log="probe/cbswe_$h.log"; : > "$log"
echo "[$h] cbswe transfer start $(date +%T)" >>"$log"
perl -e 'alarm 7200; exec @ARGV' ssh -n -o BatchMode=yes arena9-gabby \
  'docker save cbswe-flipt-02e2:native cbswe-flipt-02e2:amd64 cbswe-vuls-1832:native cbswe-vuls-1832:amd64 | gzip -1' \
  | perl -e 'alarm 7200; exec @ARGV' ssh -o BatchMode=yes arena9-$h 'gunzip -c | docker load' >>"$log" 2>&1
echo "[$h] cbswe rc=$? $(date +%T)" >>"$log"
echo "[$h] CBSWE DONE $(date +%T)" >>"$log"
