#!/bin/bash
# Bootstrap an idle arena9 box into a run host for study_apart_controls.
# Usage: bootstrap_host.sh arena9-<name>
# Copies the repo (+venv +submodule git-dirs) from apple to the identical path, prebuilds the
# python311 base image, and verifies. New hosts run the light python311-class tasks; flipt/vuls
# (SWE) and cw/kw stay on hosts that already hold those images.
set -u
H="$1"
SRC="arena9-apple"
REPO="/root/arena-capstone/submodule/spar-guardian-agents"
SSH="ssh -o ConnectTimeout=20 -o BatchMode=yes"

echo "[$H] docker + free check"
$SSH "$H" 'command -v docker >/dev/null 2>&1 || { echo NO_DOCKER; exit 3; }; awk "/MemAvailable/{print int(\$2/1024)\" MB avail\"}" /proc/meminfo; uname -m' || { echo "[$H] UNREACHABLE/NO_DOCKER"; exit 3; }

echo "[$H] copying repo from $SRC (no .git/logs/runs/out)"
$SSH "$SRC" "cd /root/arena-capstone/submodule && tar czf - --exclude=spar-guardian-agents/logs --exclude=spar-guardian-agents/runs --exclude=spar-guardian-agents/out --exclude=spar-guardian-agents/worktrees --exclude=spar-guardian-agents/.git --exclude=spar-guardian-agents/.pytest_cache --exclude=spar-guardian-agents/.ruff_cache spar-guardian-agents" \
  | $SSH "$H" 'mkdir -p /root/arena-capstone/submodule && tar xzf - -C /root/arena-capstone/submodule && echo REPO_OK'

echo "[$H] copying submodule git-dirs"
$SSH "$SRC" 'cd / && tar czf - root/arena-capstone/.git/modules/submodule/spar-guardian-agents/modules/submodules 2>/dev/null' \
  | $SSH "$H" 'cd / && tar xzf - && echo GITDIR_OK'

echo "[$H] verify venv + submodule heads"
$SSH "$H" "cd $REPO && (.venv/bin/python -c 'import inspect_ai; from src import run' 2>/dev/null && echo VENV_OK || { echo VENV_BAD; uv sync 2>&1 | tail -3; }); for s in cheatbench orbit lucy; do echo \"\$s \$(git -C submodules/\$s rev-parse --short HEAD 2>&1)\"; done"

echo "[$H] prebuild python311 base image"
$SSH "$H" "cd $REPO && mkdir -p /root/apart_logs && .venv/bin/python -c 'from src.sandbox.machines import agent_machines as m; print(\"PY311:\", m.ensure_image())'"

echo "[$H] images present:"
$SSH "$H" 'docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "cbtask|cbswe|cbkw|cbcw" || echo "(only base expected)"'
echo "[$H] BOOTSTRAP_DONE"
