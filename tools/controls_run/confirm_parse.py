#!/usr/bin/env python3
"""Dose-response confirm matrix: per-arm intensity means + crossed+/pass-rate over cm_* runs.

mPosts  = all agent board posts (seed excluded). For anti-coord this includes the hidden regulator's posts.
mWPosts = WORKER-only posts (regulator excluded, from episodes.jsonl roles): the fair comparison against none.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

RUNS = Path("/Users/unrulyabstractions/work/arena-capstone/submodule/spar-guardian-agents/runs/study_exploration_1")
ARMS = ["none", "neutralL0", "procoordL1", "procoordL2", "anticoord"]
SEEDED = ("neutralL0", "procoordL1", "procoordL2")


def board(d):
    p = d / "communication" / "flipt-02e2#1.json"
    if not p.exists():
        return []
    c = json.load(open(p))
    return c if isinstance(c, list) else c.get("messages", [])


def regulators(d):
    out = set()
    for ln in (d / "episodes.jsonl").read_text().splitlines():
        e = json.loads(ln)
        if e.get("role") == "regulator":
            out.add(e.get("orbit_agent"))
    return out


agg = defaultdict(list)
total = 0.0
for d in sorted(RUNS.glob("cm_*")):
    arm = d.name[len("cm_"):].rsplit("_s", 1)[0]
    if not (d / "signals.jsonl").exists() or not (d / "run.json").exists():
        continue
    team = None
    for ln in (d / "signals.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r.get("row") == "team":
            team = r
            break
    if team is None:
        continue
    ev = team.get("collaboration_evidence") or {}
    try:
        cost = json.load(open(d / "run.json"))["ledger"]["total"]["billed_usd"] or 0
    except Exception:
        cost = 0.0
    total += cost
    b = board(d)
    regs = regulators(d)
    posts = sum(1 for m in b if m["sender"] != "system")          # all agent posts, seed excluded
    wposts = sum(1 for m in b if m["sender"] != "system" and m["sender"] not in regs)
    agg[arm].append(dict(collab=team.get("collaboration"),
                         dis=len(ev.get("discussed") or []), sol=len(ev.get("solicited") or []),
                         crs=len(ev.get("crossed") or []), asm=len(ev.get("assembled") or []),
                         passed=team.get("passed") is True, posts=posts, wposts=wposts, cost=cost))

print(f"{'arm':12} {'n':>2} {'crossed+':>8} {'pass':>5} {'mPosts':>7} {'mWPosts':>8} {'mDisc':>6} {'mSol':>6} {'mCrs':>6} {'mAsm':>6}")
for arm in ARMS:
    v = agg.get(arm, [])
    n = len(v)
    if not n:
        print(f"{arm:12}  0   (no runs)")
        continue
    cp = sum(1 for x in v if x["collab"] in ("crossed", "assembled"))
    pa = sum(1 for x in v if x["passed"])
    m = lambda k: sum(x[k] for x in v) / n
    print(f"{arm:12} {n:>2} {cp}/{n:<6} {pa}/{n:<3} {m('posts'):>7.1f} {m('wposts'):>8.1f} {m('dis'):>6.1f} "
          f"{m('sol'):>6.1f} {m('crs'):>6.1f} {m('asm'):>6.1f}")
print(f"\nTOTAL billed: ${total:.4f} over {sum(len(v) for v in agg.values())} runs")
print("NOTE: anti-coord's hidden regulator posts heavily -> its mPosts/discussed/solicited are inflated. Judge "
      "anti-coord by mWPosts (worker-only), crossed-count (mCrs) and assembled (mAsm), against none.")
