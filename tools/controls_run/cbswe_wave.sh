#!/bin/bash
# cbswe_wave.sh src:tgt src:tgt ...  -- runs each direct cbswe transfer in parallel
for pair in "$@"; do
  src=${pair%%:*}; tgt=${pair##*:}
  bash boot_cbswe_direct.sh "$src" "$tgt" &
done
wait
echo "CBSWE WAVE DONE $(date +%T)"
