#!/bin/bash
# bootstrap light tier then verify; write probe/ready_<h>.txt with final status
h="$1"
bash boot_light.sh "$h"
res=$(bash verify_host.sh "$h")
echo "$res" > "probe/ready_$h.txt"
echo "[$h] VERIFY: $res" >> "probe/boot_$h.log"
