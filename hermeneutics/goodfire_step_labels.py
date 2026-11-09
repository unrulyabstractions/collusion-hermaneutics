"""Per-turn (step) labels for the Goodfire rollouts: the dataset's passage labels joined to the turns the extraction uses.

Each assistant turn k of a rollout (goodfire.messages_of order, the row order of think_L<L>) is joined to the passages whose
node_index is its node (turn_node, the sampled node used for rendering) or the path node it replaced (turn_path_node):
the judge files address either sibling (dataset README: "join on node_index"). Passage labels are mapped to one scale:
judge consensus positive / ambiguous / negative; monitor evidence certain -> positive, ambiguous -> ambiguous, none ->
negative. A turn's label is positive if any of its passages is, else ambiguous if any is, else negative; empty when the
turn has no passage (e.g. tau3-bench's scripted greeting). Per channel too (reasoning, content, tool). The README: exclude
ambiguous from both pools. Passages whose node is on no main-path turn (retry branches) are counted per run, not joined.

    python -m hermeneutics.goodfire_step_labels --data /workspace/data/goodfire --out /workspace/step_labels.csv
"""
import argparse
import collections
import csv
import glob
import json
import os

from .goodfire import load_rollouts, main_path

SCALE = {"positive": "positive", "negative": "negative", "ambiguous": "ambiguous",
         "certain": "positive", "none": "negative"}
RANK = {"positive": 2, "ambiguous": 1, "negative": 0}
FIELDS = ["run_dir", "rollout_id", "turn", "turn_node", "turn_path_node", "sampled", "step", "n_passages", "label",
          "label_reasoning", "label_content", "label_tool", "categories", "statuses", "label_source"]


def worst(labels):
    labels = [x for x in labels if x]
    return max(labels, key=RANK.get) if labels else ""


def passages(run_path):
    """({trace_id: {node_index: [passage]}}, source) from judge_passages.jsonl or monitor_passages.jsonl."""
    for src, rel in (("judge", "labels/judge_passages.jsonl"), ("monitor", "generic_monitor/monitor_passages.jsonl")):
        p = os.path.join(run_path, rel)
        if os.path.exists(p):
            out = collections.defaultdict(lambda: collections.defaultdict(list))
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        out[r["trace_id"]][r["node_index"]].append(r)
            return out, src
    return {}, None


def turn_row(ps, src):
    by_channel = collections.defaultdict(list)
    cats, stats = set(), set()
    for p in ps:
        lab = SCALE.get(p.get("consensus") if src == "judge" else p.get("evidence"), "")
        ch = p.get("channel") or ""
        by_channel["tool" if ch.startswith("tool") else ch].append(lab)
        if src == "judge":
            cats.update(p.get("carrier_categories") or [])
            stats.update(p.get("carrier_statuses") or [])
        else:
            if p.get("category"):
                cats.add(p["category"])
            if p.get("status"):
                stats.add(p["status"])
    return {"step": ps[0].get("step") if ps else "", "n_passages": len(ps), "label": worst(sum(by_channel.values(), [])),
            "label_reasoning": worst(by_channel["reasoning"]), "label_content": worst(by_channel["content"]),
            "label_tool": worst(by_channel["tool"]), "categories": "|".join(sorted(cats)), "statuses": "|".join(sorted(stats))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the dataset root holding the run directories")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, summary = [], []
    for traces in sorted(glob.glob(os.path.join(a.data, "*", "traces.jsonl"))):
        run_path = os.path.dirname(traces)
        rd = os.path.basename(run_path)
        pmap, src = passages(run_path)
        c = collections.Counter()
        for _, rollout in load_rollouts(traces):
            t = rollout["traces"][0]
            nodes = t["nodes"]
            path, _, original = main_path(nodes, with_original=True)
            node_pass = pmap.get(t["id"], {})
            used = set()
            k = 0
            for i, i0 in zip(path, original):
                if nodes[i]["message"]["role"] != "assistant":
                    continue
                ps = list(node_pass.get(i, [])) + (list(node_pass.get(i0, [])) if i0 != i else [])
                used.update({i, i0})
                r = {"run_dir": rd, "rollout_id": t["id"], "turn": k, "turn_node": i, "turn_path_node": i0,
                     "sampled": int(bool(nodes[i].get("sampled"))), "label_source": src or "", **turn_row(ps, src)}
                rows.append(r)
                c["turns"] += 1
                c["turns_with_passages"] += bool(ps)
                c[f"turn_{r['label'] or 'unlabelled'}"] += 1
                k += 1
            c["passages"] += sum(len(v) for v in node_pass.values())
            c["passages_unjoined"] += sum(len(v) for n, v in node_pass.items() if n not in used)
        summary.append((rd, src, dict(c)))
    with open(a.out + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(a.out + ".tmp", a.out)
    for rd, src, c in summary:
        print(f"{rd:48s} {src}: {c}")
    print(f"wrote {a.out}: {len(rows)} turns")


if __name__ == "__main__":
    main()
