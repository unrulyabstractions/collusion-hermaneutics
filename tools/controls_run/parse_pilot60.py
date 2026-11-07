#!/usr/bin/env python3
"""tb=60 iteration: per run + per arm, the collaboration level, crossed+/passed, board activity and cost."""
import json
import os
from pathlib import Path

RUNS = Path("/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents/runs/study_exploration_1")


def board_n(d):
    p = d / "communication" / "flipt-02e2#1.json"
    if not p.exists():
        return 0
    c = json.load(open(p))
    return len(c if isinstance(c, list) else c.get("messages", []))


rows = []
for d in sorted(RUNS.glob("pilot60_*")):
    arm = d.name[len("pilot60_"):].rsplit("_s", 1)[0]
    team = None
    for ln in (d / "signals.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("row") == "team":
            team = r
            break
    ev = team.get("collaboration_evidence") or {}
    cost = json.load(open(d / "run.json")).get("ledger", {}).get("total", {}).get("billed_usd")
    bn = board_n(d)
    rows.append((d.name, arm, team.get("collaboration"), len(ev.get("discussed") or []),
                 len(ev.get("solicited") or []), len(ev.get("crossed") or []), len(ev.get("assembled") or []),
                 str(team.get("passed")), max(0, bn - (0 if arm == "none" else 1)), cost))

print(f"{'run':26} {'collab':10} {'dis':>3} {'sol':>3} {'crs':>3} {'asm':>3} {'passed':6} {'posts':>5} {'cost$':>7}")
from collections import defaultdict
agg = defaultdict(list)
total = 0.0
for name, arm, collab, dis, sol, crs, asm, passed, posts, cost in rows:
    c = f"{cost:.4f}" if isinstance(cost, (int, float)) else "--"
    total += cost if isinstance(cost, (int, float)) else 0
    print(f"{name:26} {str(collab):10} {dis:>3} {sol:>3} {crs:>3} {asm:>3} {passed:6} {posts:>5} {c:>7}")
    agg[arm].append((collab, crs, asm, passed == "True", posts, cost))

print(f"\nTOTAL tb=60 billed: ${total:.4f} over {len(rows)} runs\n")
print(f"{'arm':12} {'n':>2} {'crossed+':>8} {'passed':>7} {'mean_posts':>10} {'mean_crs':>8}")
for arm, v in sorted(agg.items()):
    n = len(v)
    cp = sum(1 for c, crs, asm, p, posts, cost in v if c in ("crossed", "assembled"))
    pa = sum(1 for *_, p, posts, cost in [(x[0], x[1], x[2], x[3], x[4], x[5]) for x in v] if x[3])
    mp = sum(x[4] for x in v) / n
    mc = sum(x[1] for x in v) / n
    print(f"{arm:12} {n:>2} {cp}/{n:>6} {pa}/{n:>5} {mp:>10.1f} {mc:>8.1f}")
