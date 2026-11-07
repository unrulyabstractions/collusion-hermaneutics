#!/usr/bin/env python3
"""Tabulate the board-seed pilot: per run the team collaboration level, crossed/assembled counts, passed, cheat rung
and billed cost; then crossed+ (crossed or assembled) fraction per arm."""
import json
import sys
from pathlib import Path

RUNS = Path("/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents/runs/study_exploration_1")
ORDER = ("isolated", "discussed", "solicited", "crossed", "assembled")


def arm_of(name: str) -> str:
    # pilot_<arm>_s<seed>
    return name[len("pilot_"):].rsplit("_s", 1)[0]


def seed_of(name: str) -> str:
    return name.rsplit("_s", 1)[-1]


rows = []
for d in sorted(RUNS.glob("pilot_*")):
    sig = d / "signals.jsonl"
    rj = d / "run.json"
    if not sig.exists():
        rows.append((d.name, "NO_SIGNALS", 0, 0, "?", "?", None))
        continue
    team = None
    for ln in sig.read_text().splitlines():
        r = json.loads(ln)
        if r.get("row") == "team":
            team = r
            break
    cost = None
    if rj.exists():
        try:
            cost = json.load(open(rj)).get("ledger", {}).get("total", {}).get("billed_usd")
        except Exception:
            pass
    if team is None:
        rows.append((d.name, "NO_TEAM", 0, 0, "?", "?", cost))
        continue
    ev = team.get("collaboration_evidence") or {}
    rows.append((d.name, team.get("collaboration"), len(ev.get("crossed") or []),
                 len(ev.get("assembled") or []), str(team.get("passed")),
                 str(team.get("rung") or team.get("rung_by_code")), cost))

print(f"{'run':28} {'collab':11} {'crs':>3} {'asm':>3} {'passed':6} {'rung':10} {'cost$':>7}")
total = 0.0
by_arm = {}
for name, collab, crs, asm, passed, rung, cost in rows:
    c = f"{cost:.4f}" if isinstance(cost, (int, float)) else "--"
    if isinstance(cost, (int, float)):
        total += cost
    print(f"{name:28} {str(collab):11} {crs:>3} {asm:>3} {passed:6} {str(rung):10} {c:>7}")
    arm = arm_of(name)
    crossed_plus = str(collab) in ("crossed", "assembled")
    by_arm.setdefault(arm, []).append(crossed_plus)

print(f"\nTOTAL billed: ${total:.4f} over {sum(1 for r in rows if isinstance(r[6],(int,float)))} runs")
print("\narm            n   crossed+   frac")
for arm, flags in sorted(by_arm.items()):
    n = len(flags)
    cp = sum(flags)
    print(f"{arm:14} {n:>2}   {cp:>6}    {cp/n:.2f}" if n else f"{arm}: n=0")
