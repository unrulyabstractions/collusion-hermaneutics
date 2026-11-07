#!/bin/bash
# Verify a bootstrapped host. Prints a single RESULT line.
h="$1"
perl -e 'alarm 120; exec @ARGV' ssh -o ConnectTimeout=10 -o BatchMode=yes arena9-$h '
  P=/root/arena-capstone/submodule/spar-guardian-agents
  tree=$([ -d "$P/src" ] && [ -d "$P/submodules" ] && [ -x "$P/.venv/bin/python" ] && echo ok || echo bad)
  pyver=$("$P/.venv/bin/python" --version 2>&1)
  d311=$(docker run --rm cbtask:python311 python -c "print(1)" 2>/dev/null)
  imgs=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -iE "cbswe-flipt-02e2|cbswe-vuls|cbtask:python311" | tr "\n" ",")
  flipt=$(docker image inspect cbswe-flipt-02e2:amd64 >/dev/null 2>&1 && echo y || echo n)
  vuls=$(docker image inspect cbswe-vuls-1832:amd64 >/dev/null 2>&1 && echo y || echo n)
  echo "RESULT tree=$tree py=[$pyver] run311=[$d311] flipt=$flipt vuls=$vuls imgs=[$imgs]"
' 2>/dev/null
