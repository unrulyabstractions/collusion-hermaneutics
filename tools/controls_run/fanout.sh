#!/bin/bash
# fanout.sh <script> <conc> <host...>
sc="$1"; conc="$2"; shift 2
printf '%s\n' "$@" | xargs -P "$conc" -I {} bash "$sc" {}
echo "FANOUT DONE $(date +%T)"
