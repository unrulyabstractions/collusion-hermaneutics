"""Goodfire/reward-hack-data as input to the teacher-forcing extraction: other labs' agent rollouts (Kimi K3, GLM 5.2,
Qwen 3.8 Max on SWE-bench Verified, DeepSWE, ImpossibleBench, tau3-bench, ShoppingBench) rendered in Qwen3.8-27B's chat
template, in the same shape render.sequence() gives an SGA agent, so extract_goodfire.py can run extract.py's pass on them.

Dataset layout (public, https://huggingface.co/datasets/Goodfire/reward-hack-data): one directory per run (RUN_DIRS),
each with traces.jsonl (one rollout per line: {"id", "env", "ok", "errors", "traces": [trace]}; traces[0].id is the
rollout id every label file joins on), labels/judge_rollouts.jsonl + judge_passages.jsonl (SWE-bench, DeepSWE,
ImpossibleBench) or generic_monitor/monitor_rollouts.jsonl + monitor_passages.jsonl (tau3-bench, ShoppingBench).

Conversation. traces[0].nodes is a tree: one node per message, linked by `parent` (index into nodes; one root per
rollout, the system message; a parent always precedes its children). Two shapes were seen:
  * mini-swe-agent runs (SWE-bench): a chain. Every assistant node is sampled and carries reasoning_content.
  * Inspect-harness runs (ImpossibleBench, DeepSWE): at every assistant step with reasoning the harness re-serialised the
    model's message before feeding it back, so the step appears twice under the same parent: the SAMPLED node (the
    model's raw output: content, reasoning_content, tool_calls; a leaf) and an UNSAMPLED sibling whose content is
    "\\n<think>\\n{reasoning}\\n</think>\\n\\n{content}" and which carries the tool results and the rest of the conversation.
    Tool-call ids and JSON-parsed arguments agree between the two (checked over impossiblebench-kimi-k3: 3244 of 3244).
We take the path from the root to the deepest leaf (ties: the last node in file order) and, for every unsampled
assistant node on it, substitute its sampled sibling (same parent, sampled, role assistant) when there is one, so the
reasoning comes from reasoning_content and not from a text parse. Nodes off that path are counted (info.n_off_path);
any off-path node that is not a sampled sibling of a path node is counted apart (info.n_off_path_unexplained: 0 in the
runs checked). calls[].node names the sampled nodes, so the j-th assistant turn is the author's j-th model call.

Reasoning. reasoning_content fills the <think> block (render.to_template). When an assistant node has no
reasoning_content and its content opens with a literal "<think>", the model wrote its thinking as text (Kimi K3 does this
after the harness showed it its own earlier turns in that form): the block up to "</think>" becomes the reasoning and
the rest the content; with no closing tag the whole remainder is reasoning and the content is empty
(info.think_from_content, info.think_unclosed). A literal "<think>" or "</think>" left inside the reasoning text is
rewritten "<think >" / "</think >" (info.think_tags_neutralized), because the tokenizer maps the literal to the same
special id as the template's tag and think.think_positions would close the turn at it; content, user and tool text are
left alone (think.py's prefix guard already ignores them). Tool results become role "tool" with their tool_call_id
(the template folds them into a user turn as <tool_response> blocks); the trace's tool definitions (traces[0].tools,
empty for mini-swe-agent) go to the template's tools block. TEMPLATE_KWARGS as for the SGA data.

Author and benchmark come from the run directory name (RUN_DIRS); the author model string is traces[0].agent.model
(mini-swe-agent) or .agent.config.model (Inspect).

    python -m hermeneutics.goodfire render --traces DIR/traces.jsonl [--first 20] [--show 3] [--units]   # tokenizer only
    python -m hermeneutics.goodfire plan --traces DIR/traces.jsonl --out plan.csv [--max-tokens 100000] [--n-shards 3]
                                        [--ids-dir DIR] [--jobs 8]
    python -m hermeneutics.goodfire labels --run-dir DIR --out labels.csv
    python -m hermeneutics.goodfire verify CACHE_DIR
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import time

from . import MODEL_DIR
from .render import TEMPLATE_KWARGS, render as render_text
from .think import CLOSE, OPEN, think_positions

DATASET = "Goodfire/reward-hack-data"
AUTHOR_NAME = {"kimi": "Kimi K3", "glm": "GLM 5.2", "qwenmax": "Qwen 3.8 Max"}
# run directory -> (benchmark, author key, label source)
RUN_DIRS = {
    "swebench-verified-kimi-k3-nohack": ("swebench-verified", "kimi", "judge"),
    "swebench-verified-glm-5.2-nohack": ("swebench-verified", "glm", "judge"),
    "swebench-verified-qwen3.8-2.4t-a95b-fp8-nohack": ("swebench-verified", "qwenmax", "judge"),
    "deepswe-kimi-k3": ("deepswe", "kimi", "judge"),
    "deepswe-glm-5.2-fp8": ("deepswe", "glm", "judge"),
    "deepswe-qwen3.8-2.4t-a95b-fp8": ("deepswe", "qwenmax", "judge"),
    "impossiblebench-kimi-k3": ("impossiblebench", "kimi", "judge"),
    "impossiblebench-glm-5.2-fp8": ("impossiblebench", "glm", "judge"),
    "impossiblebench-qwen3.8-2.4t-a95b-fp8": ("impossiblebench", "qwenmax", "judge"),
    "tau3bench-kimi-k3": ("tau3bench", "kimi", "monitor"),
    "tau3bench-glm-5.2-fp8": ("tau3bench", "glm", "monitor"),
    "shoppingbench-kimi-k3-20260828": ("shoppingbench", "kimi", "monitor"),
    "shoppingbench-glm-5.2-20260828": ("shoppingbench", "glm", "monitor"),
}
LEADING_THINK = re.compile(r"\A\s*<think>\n?", re.S)
_TOK = None


def tokenizer(model_dir=MODEL_DIR):
    """The Qwen3.8 tokenizer, loaded once per process (a forked worker inherits it)."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(model_dir)
    return _TOK


def run_info(run_dir):
    """{run_dir, benchmark, author, author_name, label_source} from a run directory name (unknown names are parsed)."""
    name = os.path.basename(os.path.normpath(run_dir))
    if name in RUN_DIRS:
        bench, author, src = RUN_DIRS[name]
    else:
        author = "kimi" if "kimi" in name else "glm" if "glm" in name else "qwenmax" if "qwen" in name else "unknown"
        bench = name.split("-" + {"kimi": "kimi", "glm": "glm", "qwenmax": "qwen"}.get(author, "?"))[0]
        src = "monitor" if bench.startswith(("tau3", "shopping")) else "judge"
    return {"run_dir": name, "benchmark": bench, "author": author, "author_name": AUTHOR_NAME.get(author, author), "label_source": src}


def run_dir_of(traces_path):
    return os.path.basename(os.path.dirname(os.path.abspath(traces_path)))


# ---------------------------------------------------------------- rollouts
def load_rollouts(traces_path, ids=None, first=0):
    """Yield (row_index, rollout) from traces.jsonl in file order; only the rollout ids in `ids` when given; at most
    `first` rows when first > 0. The rollout id is traces[0].id."""
    want = set(ids) if ids is not None else None
    n = 0
    with open(traces_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if first and n >= first:
                break
            r = json.loads(line)
            if want is not None and r["traces"][0]["id"] not in want:
                continue
            n += 1
            yield i, r


def main_path(nodes, with_original=False):
    """(path, off_path_unexplained): the node indices from the root to the deepest leaf (ties: the last in file order),
    unsampled assistant nodes replaced by their sampled sibling; and the off-path nodes that are not such siblings.
    with_original: also the path before substitution, index for index (judge files address either sibling)."""
    children = collections.defaultdict(list)
    roots = []
    for i, nd in enumerate(nodes):
        p = nd.get("parent")
        if p is None:
            roots.append(i)
        else:
            children[p].append(i)
    if len(roots) != 1:
        raise ValueError(f"{len(roots)} roots")
    depth = {roots[0]: 0}
    for i, nd in enumerate(nodes):
        p = nd.get("parent")
        if p is not None:
            if p >= i or p not in depth:
                raise ValueError(f"node {i}: parent {p} does not precede it")
            depth[i] = depth[p] + 1
    leaves = [i for i in range(len(nodes)) if not children[i]]
    leaf = max(leaves, key=lambda i: (depth[i], i))
    path = []
    k = leaf
    while k is not None:
        path.append(k)
        k = nodes[k].get("parent")
    path.reverse()
    out, explained = [], set(path)
    for i in path:
        nd = nodes[i]
        if nd["message"]["role"] == "assistant" and not nd.get("sampled"):
            sibs = [j for j in children[nd["parent"]] if j != i and nodes[j].get("sampled") and nodes[j]["message"]["role"] == "assistant"]
            if sibs:
                i = sibs[-1]
                explained.add(i)
        out.append(i)
    unexplained = [i for i in range(len(nodes)) if i not in explained]
    if with_original:
        return out, unexplained, path
    return out, unexplained


def _text(content):
    """Message content as text: a string, or the text items of a content list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
        return "".join(parts)
    return str(content)


def split_think(content):
    """(reasoning, content, closed) when content opens with a literal <think>, else (None, content, None)."""
    m = LEADING_THINK.match(content)
    if not m:
        return None, content, None
    rest = content[m.end():]
    j = rest.find("</think>")
    if j < 0:
        return rest.strip(), "", False
    return rest[:j].strip(), rest[j + len("</think>"):].lstrip("\n"), True


def neutralize(text):
    """(text with literal think tags rewritten, count)."""
    n = text.count("<think>") + text.count("</think>")
    if n:
        text = text.replace("</think>", "</think >").replace("<think>", "<think >")
    return text, n


def to_openai(message, info):
    """A trace node's message -> OpenAI-style chat message (what render.to_template expects). info counts the repairs."""
    role = message["role"]
    content = _text(message.get("content"))
    if role == "assistant":
        reasoning = message.get("reasoning_content")
        reasoning = reasoning if isinstance(reasoning, str) else _text(reasoning) if reasoning is not None else None
        if reasoning is None or not reasoning.strip():
            r2, content, closed = split_think(content)
            if r2 is not None:
                reasoning = r2
                info["think_from_content"] += 1
                if closed is False:
                    info["think_unclosed"] += 1
        else:
            # The provider failed to split the output: reasoning_content holds "REASONING</think>VISIBLE ANSWER" (Kimi K3 on
            # ImpossibleBench, 307 turns). The answer goes back to the content, so the template's </think> follows the
            # reasoning only.
            j = reasoning.find("</think>")
            if j >= 0:
                after = reasoning[j + len("</think>"):].strip("\n")
                reasoning = reasoning[:j]
                content = after + ("\n\n" + content if content.strip() and after.strip() else content)
                info["think_split_from_reasoning"] += 1
            # A second think block opening the content while reasoning is present: its text is reasoning too.
            r2, c2, closed = split_think(content)
            if r2 is not None:
                reasoning = reasoning.rstrip("\n") + "\n\n" + r2
                content = c2
                info["think_extra_from_content"] += 1
        reasoning, n = neutralize(reasoning or "")
        info["think_tags_neutralized"] += n
        content, n = neutralize(content)  # literal tags in the visible answer must not become think ids outside the block
        info["content_tags_neutralized"] += n
        calls = []
        for c in message.get("tool_calls") or []:
            fn = c.get("function") if isinstance(c.get("function"), dict) else c
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args if args is not None else {})
            else:
                try:
                    json.loads(args)
                except ValueError:
                    info["tool_args_not_json"] += 1
                    args = json.dumps({"arguments": args})
            calls.append({"id": c.get("id"), "type": "function", "function": {"name": fn.get("name"), "arguments": args}})
        return {"role": "assistant", "content": content, "reasoning": reasoning, "tool_calls": calls or None}
    if role == "tool":
        return {"role": "tool", "content": content, "tool_call_id": message.get("tool_call_id"), "name": message.get("name")}
    if role in ("system", "user"):
        return {"role": role, "content": content}
    raise ValueError(f"unexpected role {role}")


def to_tool(t):
    return {"type": "function", "function": {"name": t.get("name"), "description": t.get("description") or "",
                                             "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}


def messages_of(rollout):
    """(messages, tools, turn_nodes, info) for one rollout: the main path as OpenAI-style messages, the tool
    definitions, the node index of every assistant turn, and the reconstruction counts."""
    t = rollout["traces"][0]
    nodes = t["nodes"]
    path, unexplained, original = main_path(nodes, with_original=True)
    info = collections.Counter()
    msgs, turn_nodes, turn_path_nodes, turn_sampled = [], [], [], []
    for i, i0 in zip(path, original):
        m = to_openai(nodes[i]["message"], info)
        if m["role"] == "assistant":
            turn_nodes.append(i)
            turn_path_nodes.append(i0)
            # False for assistant text the author never generated, e.g. tau3-bench's scripted opening greeting
            turn_sampled.append(bool(nodes[i].get("sampled")))
        msgs.append(m)
    agent = t.get("agent") or {}
    model = agent.get("model") or (agent.get("config") or {}).get("model")
    if not model and t.get("calls"):
        model = t["calls"][0].get("model")
    task = t.get("task") if isinstance(t.get("task"), dict) else {}
    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    info = dict(info)
    info.update(rollout_id=t["id"], outer_id=rollout.get("id"), env=(rollout.get("env") or {}).get("id"), ok=rollout.get("ok"),
                stop_condition=t.get("stop_condition"), task=data.get("name"), author_model=model, n_nodes=len(nodes),
                n_sampled=sum(1 for nd in nodes if nd.get("sampled")), n_calls=len(t.get("calls") or []), n_path=len(path),
                n_off_path=len(nodes) - len(path), n_off_path_unexplained=len(unexplained),
                turn_path_nodes=turn_path_nodes, turn_sampled=turn_sampled)
    return msgs, [to_tool(x) for x in (t.get("tools") or [])], turn_nodes, info


def render(rollout, tok=None, with_units=False):
    """The rendered sequence of one rollout in render.sequence()'s per-agent shape: text, ids, units (None unless
    with_units), think [(open, close)] per assistant turn, readable per turn, messages, tools, calls, turns, model,
    append_only ({}), plus turn_nodes and info."""
    tok = tok or tokenizer()
    msgs, tools, turn_nodes, info = messages_of(rollout)
    text = render_text(tok, msgs, tools, False)
    if with_units:
        from . import spans
        ids, units = spans.build(text, msgs, tok)
    else:
        ids, units = tok(text, add_special_tokens=False)["input_ids"], None
    think = think_positions(ids)
    asst = [m for m in msgs if m["role"] == "assistant"]
    readable = [bool((m.get("reasoning") or "").strip()) for m in asst]
    # Every assistant turn renders as "<think>\n{reasoning|trim}\n</think>\n\n{content}" (chat_template.jinja, preserve_thinking
    # defaults to true), so the tokens strictly between a turn's pair decode to its trimmed reasoning. A pair that does not
    # is misplaced (e.g. a literal "<|im_start|>assistant\n<think>" in a tool output fooling think_positions' guard).
    mism = [k for k, ((a, b), m) in enumerate(zip(think, asst))
            if tok.decode(ids[a + 1:b]).strip() != (m.get("reasoning") or "").strip()]
    info["think_mismatch"] = len(mism) + abs(len(think) - len(asst))
    info["think_mismatch_first"] = mism[0] if mism else None
    return {"text": text, "ids": ids, "units": units, "think": think, "readable": readable, "messages": msgs, "tools": tools,
            "calls": info["n_calls"] or len(turn_nodes), "turns": len(turn_nodes), "model": info["author_model"],
            "append_only": {}, "turn_nodes": turn_nodes, "info": info}


# ---------------------------------------------------------------- planning
def shard_of(rollout_id, n_shards):
    """A stable shard index from the rollout id (sha1), so hosts can split one ids list without coordination."""
    return int(hashlib.sha1(rollout_id.encode()).hexdigest(), 16) % max(1, n_shards)


def _plan_one(args):
    row_index, rollout, run_dir, max_tokens, n_shards = args
    rid = rollout["traces"][0]["id"]
    row = {"run_dir": run_dir, "rollout_id": rid, "row_index": row_index, "shard": shard_of(rid, n_shards)}
    try:
        q = render(rollout, tokenizer())
    except Exception as e:  # noqa: BLE001  (record and exclude)
        row.update(n_tokens=None, n_turns=None, n_readable=None, n_messages=None, think_pairs=None, included=False,
                   reason=f"render_error: {type(e).__name__}: {str(e)[:160]}")
        return row
    n = len(q["ids"])
    mism = q["info"].get("think_mismatch", 0)
    row.update(n_tokens=n, n_turns=q["turns"], n_readable=sum(q["readable"]), n_messages=len(q["messages"]),
               think_pairs=len(q["think"]), included=n <= max_tokens and len(q["think"]) == q["turns"] and not mism,
               reason="" if n <= max_tokens else "over_max_tokens")
    if len(q["think"]) != q["turns"]:
        row["reason"] = (row["reason"] + ";" if row["reason"] else "") + "think_pairs!=turns"
    if mism:
        row["reason"] = (row["reason"] + ";" if row["reason"] else "") + "think_mismatch"
    for k in PLAN_COUNTS:
        row[k] = q["info"].get(k, 0)
    row["n_unsampled_turns"] = sum(1 for s in q["info"].get("turn_sampled", []) if not s)
    return row


PLAN_COUNTS = ("task", "author_model", "n_off_path_unexplained", "think_from_content", "think_unclosed", "think_tags_neutralized",
               "think_split_from_reasoning", "think_extra_from_content", "content_tags_neutralized", "think_mismatch")


def plan_shards(traces_path, max_tokens=100_000, n_shards=1, jobs=1, first=0, run_dir=None):
    """One row per rollout: rollout_id, n_tokens, n_turns, included (n_tokens <= max_tokens and every turn has its think
    pair), reason, shard (sha1 of the id mod n_shards), and the reconstruction counts. Rollouts the renderer rejects are
    kept with included=False and the error as reason."""
    run_dir = run_dir or run_dir_of(traces_path)
    items = ((i, r, run_dir, max_tokens, n_shards) for i, r in load_rollouts(traces_path, first=first))
    if jobs > 1:
        import multiprocessing as mp
        tokenizer()  # loaded before the fork, shared by the workers
        with mp.get_context("fork").Pool(jobs) as p:
            return list(p.imap(_plan_one, items, chunksize=2))
    return [_plan_one(x) for x in items]


def write_plan(rows, out_csv, ids_dir=None):
    fields = ["run_dir", "rollout_id", "row_index", "shard", "n_tokens", "n_turns", "n_readable", "n_messages", "think_pairs",
              "included", "reason", *PLAN_COUNTS, "n_unsampled_turns"]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    if ids_dir:
        os.makedirs(ids_dir, exist_ok=True)
        by_shard = collections.defaultdict(list)
        for r in rows:
            if r["included"]:
                by_shard[r["shard"]].append(r["rollout_id"])
        for s, ids in sorted(by_shard.items()):
            with open(os.path.join(ids_dir, f"{rows[0]['run_dir']}.shard{s}.ids"), "w") as f:
                f.write("\n".join(ids) + "\n")


def read_ids(path):
    """Rollout ids from a file: one per line, or the rollout_id column of a CSV."""
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return [r["rollout_id"] for r in csv.DictReader(f)]
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
    return {"min": xs[0], "median": q(0.5), "p90": q(0.9), "max": xs[-1], "sum": sum(xs)}


def summarize(rows, max_tokens):
    ok = [r for r in rows if r["n_tokens"] is not None]
    q = _quantiles([r["n_tokens"] for r in ok])
    lines = [f"{len(rows)} rollouts, {len(rows) - len(ok)} render errors, {sum(r['included'] for r in rows)} included",
             f"tokens: min {q.get('min')} median {q.get('median')} p90 {q.get('p90')} max {q.get('max')} total {q.get('sum')}",
             f"over {max_tokens} tokens: {sum(1 for r in ok if r['n_tokens'] > max_tokens)}",
             f"think pairs == assistant turns: {sum(1 for r in ok if r['think_pairs'] == r['n_turns'])} of {len(ok)}; "
             f"turns {sum(r['n_turns'] for r in ok)}, readable {sum(r['n_readable'] for r in ok)}",
             f"off-path nodes unexplained: {sum(r['n_off_path_unexplained'] for r in ok)}; think from content "
             f"{sum(r['think_from_content'] for r in ok)} (unclosed {sum(r['think_unclosed'] for r in ok)}); "
             f"think tags neutralized {sum(r['think_tags_neutralized'] for r in ok)}",
             f"think split from reasoning {sum(r.get('think_split_from_reasoning') or 0 for r in ok)}; extra think from content "
             f"{sum(r.get('think_extra_from_content') or 0 for r in ok)}; content tags neutralized "
             f"{sum(r.get('content_tags_neutralized') or 0 for r in ok)}; THINK MISMATCH rollouts "
             f"{sum(1 for r in ok if r.get('think_mismatch'))}; unsampled turns {sum(r.get('n_unsampled_turns') or 0 for r in ok)}"]
    shards = collections.Counter(r["shard"] for r in rows if r["included"])
    if len(shards) > 1:
        lines.append("included per shard: " + ", ".join(f"{s}: {n}" for s, n in sorted(shards.items())))
    return "\n".join(lines)


# ---------------------------------------------------------------- labels
JUDGE_FIELDS = ("verdict", "verdict_enacted", "categories", "task", "split", "reward", "shortcut_signal", "judged",
                "n_passages", "n_passages_positive", "n_passages_ambiguous")
MONITOR_FIELDS = ("verdict", "status", "judged", "any_certain_considered", "any_certain_attempted", "any_certain_enacted",
                  "n_passages", "n_passages_certain", "n_passages_ambiguous")


def label_rows(run_path):
    """{rollout_id: label dict} for one run directory: judge_rollouts.jsonl (verdict yes/no/split, verdict_enacted,
    categories, ...) or monitor_rollouts.jsonl (verdict yes/unclear/no; no rollout-level categories). For monitor rows
    verdict_enacted is derived to mean what the judge's field means, attempted or enacted: "yes" when any_certain_attempted
    or any_certain_enacted, "no" when the verdict is "no", else None (undecided: the monitor README says unclear is not a
    negative). The raw any_certain_* fields are kept. Every dict carries label_source and row_index."""
    out = {}
    judge = os.path.join(run_path, "labels", "judge_rollouts.jsonl")
    monitor = os.path.join(run_path, "generic_monitor", "monitor_rollouts.jsonl")
    if os.path.exists(judge):
        with open(judge) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["trace_id"]] = {"label_source": "judge", "row_index": r.get("row_index"),
                                          **{k: r.get(k) for k in JUDGE_FIELDS}}
    elif os.path.exists(monitor):
        with open(monitor) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["trace_id"]] = {"label_source": "monitor", "row_index": r.get("row_index"),
                                          **{k: r.get(k) for k in MONITOR_FIELDS},
                                          "verdict_enacted": "yes" if (r.get("any_certain_attempted") or r.get("any_certain_enacted"))
                                          else "no" if r.get("verdict") == "no" else None, "categories": []}
    return out


def binary(verdict):
    return {"yes": 1, "no": 0}.get(verdict)


def write_labels(run_path, out_csv):
    """One CSV row per rollout of run_path/traces.jsonl, with the label fields of label_rows joined on traces[0].id and the
    binary columns cheat (verdict yes -> 1, no -> 0, else empty) and cheat_enacted. Returns the rows."""
    info = run_info(run_path)
    labels = label_rows(run_path)
    rows = []
    for i, r in load_rollouts(os.path.join(run_path, "traces.jsonl")):
        t = r["traces"][0]
        try:
            path, _ = main_path(t["nodes"])
            n_turns = sum(1 for k in path if t["nodes"][k]["message"]["role"] == "assistant")
        except ValueError:
            n_turns = None
        task = t.get("task") if isinstance(t.get("task"), dict) else {}
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        lab = labels.get(t["id"], {})
        rows.append({"run_dir": info["run_dir"], "benchmark": info["benchmark"], "author": info["author"], "rollout_id": t["id"],
                     "row_index": i, "task": data.get("name"), "n_nodes": len(t["nodes"]), "n_turns": n_turns,
                     "label_source": lab.get("label_source"), "verdict": lab.get("verdict"), "verdict_enacted": lab.get("verdict_enacted"),
                     "cheat": binary(lab.get("verdict")), "cheat_enacted": binary(lab.get("verdict_enacted")),
                     "categories": "|".join(lab.get("categories") or []), "split": lab.get("split"), "reward": lab.get("reward"),
                     "status": lab.get("status"), "n_passages": lab.get("n_passages"),
                     "n_passages_positive": lab.get("n_passages_positive", lab.get("n_passages_certain")),
                     "n_passages_ambiguous": lab.get("n_passages_ambiguous")})
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rollout_id"])
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------- verification of a cache
def check_rollout(d):
    """Problems (empty when sound) and a summary for one <out>/<run_dir>/<rollout_id>/ folder: meta.json present, think_L<L>
    [n_turns, 2, 5120] fp16 finite and not all zero for every layer in meta, ids[think_pos] at the think ids, n_turns and
    n_tokens as meta says, spans.jsonl lines == n_units (0 unless the sentence means were on)."""
    import numpy as np
    from safetensors import safe_open
    problems = []
    meta = json.load(open(os.path.join(d, "meta.json")))
    n_units = sum(1 for _ in open(os.path.join(d, "spans.jsonl")))
    with safe_open(os.path.join(d, "acts.safetensors"), framework="np") as f:
        keys = set(f.keys())
        ids, pos, readable = f.get_tensor("ids"), f.get_tensor("think_pos"), f.get_tensor("think_readable")
        n_turns = pos.shape[0]
        if n_turns and not (np.all(ids[pos[:, 0]] == OPEN) and np.all(ids[pos[:, 1]] == CLOSE)):
            problems.append("think_pos does not point at <think>/</think> ids")
        if readable.shape != (n_turns,):
            problems.append(f"think_readable shape {readable.shape}")
        if n_turns != meta["n_turns"]:
            problems.append(f"think_pos rows {n_turns} != n_turns {meta['n_turns']}")
        if len(ids) != meta["n_tokens"]:
            problems.append(f"ids {len(ids)} != n_tokens {meta['n_tokens']}")
        if n_units != meta["n_units"]:
            problems.append(f"spans {n_units} != n_units {meta['n_units']}")
        if len(meta.get("turn_nodes") or []) != n_turns:
            problems.append(f"turn_nodes {len(meta.get('turn_nodes') or [])} != n_turns {n_turns}")
        if n_turns != meta["turns"] - meta.get("turns_dropped", 0):
            problems.append(f"n_turns {n_turns} != turns {meta['turns']} - dropped {meta.get('turns_dropped', 0)}")
        for L in meta["layers"]:
            want = [(f"think_L{L}", (n_turns, 2, 5120))] + ([(f"mean_L{L}", (n_units, 5120))] if meta.get("units") else [])
            for name, shape in want:
                if name not in keys:
                    problems.append(f"missing {name}")
                    continue
                x = f.get_tensor(name)
                if x.dtype != np.float16:
                    problems.append(f"{name} dtype {x.dtype}")
                if x.shape != shape:
                    problems.append(f"{name} shape {x.shape} != {shape}")
                if not np.isfinite(x.astype(np.float32)).all():
                    problems.append(f"{name} non-finite")
                if x.size and float(np.abs(x.astype(np.float32)).mean()) == 0.0:
                    problems.append(f"{name} all zeros")
    return problems, {"run_dir": meta["run_dir"], "rollout_id": meta["rollout_id"], "author": meta["author"], "verdict": meta.get("verdict"),
                      "n_turns": n_turns, "n_tokens": meta["n_tokens"], "n_readable": int(readable.sum())}


def verify_cache(cache):
    import glob
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(cache, "*", "*", "meta.json")))
    bad, rows = 0, []
    for d in dirs:
        try:
            problems, s = check_rollout(d)
        except Exception as e:  # noqa: BLE001
            problems, s = [f"{type(e).__name__}: {e}"], None
        if problems:
            bad += 1
            print("BROKEN", d, "; ".join(problems))
        if s:
            rows.append(s)
    by = collections.defaultdict(lambda: collections.Counter())
    for s in rows:
        c = by[(s["run_dir"], s["author"])]
        c["rollouts"] += 1
        c["turns"] += s["n_turns"]
        c["tokens"] += s["n_tokens"]
        c["readable"] += s["n_readable"]
        c[f"verdict_{s['verdict']}"] += 1
    for k, c in sorted(by.items()):
        print(k[0], k[1], dict(c))
    half = [d for d in sorted(glob.glob(os.path.join(cache, "*", "*"))) if os.path.isdir(d) and not os.path.exists(os.path.join(d, "meta.json"))]
    print(f"{len(dirs)} rollout folders, {bad} broken, {len(rows)} sound; turns {sum(s['n_turns'] for s in rows)}, "
          f"tokens {sum(s['n_tokens'] for s in rows)}; {len(half)} folders without meta.json")
    return bad


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="tokenizer-only check: render rollouts, print counts and examples")
    r.add_argument("--traces", required=True)
    r.add_argument("--first", type=int, default=0)
    r.add_argument("--ids-file", default=None)
    r.add_argument("--show", type=int, default=3, help="print the first 600 characters of this many renders")
    r.add_argument("--units", action="store_true", help="also build the sentence units (spans.py)")
    r.add_argument("--max-tokens", type=int, default=100_000)
    r.add_argument("--model", default=MODEL_DIR)
    r.add_argument("--out", default=None, help="CSV of the per-rollout rows")
    p = sub.add_parser("plan", help="per-rollout token counts and shards; ids files per shard")
    p.add_argument("--traces", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=100_000)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--ids-dir", default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--model", default=MODEL_DIR)
    lb = sub.add_parser("labels", help="label CSV for one run directory")
    lb.add_argument("--run-dir", required=True)
    lb.add_argument("--out", required=True)
    v = sub.add_parser("verify", help="check an extraction cache")
    v.add_argument("cache")
    args = ap.parse_args()

    if args.cmd == "labels":
        rows = write_labels(args.run_dir, args.out)
        c = collections.Counter(r["verdict"] for r in rows)
        print(f"{len(rows)} rollouts, labels for {sum(1 for r in rows if r['label_source'])}; verdict {dict(c)}; wrote {args.out}")
        return
    if args.cmd == "verify":
        sys.exit(1 if verify_cache(args.cache) else 0)

    t0 = time.time()
    tok = tokenizer(args.model)
    run_dir = run_dir_of(args.traces)
    if args.cmd == "plan":
        rows = plan_shards(args.traces, args.max_tokens, args.n_shards, args.jobs, args.first, run_dir)
        write_plan(rows, args.out, args.ids_dir)
        print(summarize(rows, args.max_tokens))
        print(f"wrote {args.out}" + (f" and ids files under {args.ids_dir}" if args.ids_dir else "") + f", {time.time() - t0:.0f}s")
        return

    ids = read_ids(args.ids_file) if args.ids_file else None
    rows, shown = [], 0
    for i, rollout in load_rollouts(args.traces, ids, args.first):
        q = render(rollout, tok, args.units)
        n = len(q["ids"])
        rows.append({"run_dir": run_dir, "rollout_id": q["info"]["rollout_id"], "row_index": i, "shard": 0, "n_tokens": n,
                     "n_turns": q["turns"], "n_readable": sum(q["readable"]), "n_messages": len(q["messages"]),
                     "think_pairs": len(q["think"]), "included": n <= args.max_tokens and len(q["think"]) == q["turns"],
                     "reason": "" if n <= args.max_tokens else "over_max_tokens", "n_units": len(q["units"]) if q["units"] is not None else None,
                     **{k: q["info"].get(k, 0) for k in ("task", "author_model", "n_off_path_unexplained", "think_from_content",
                                                          "think_unclosed", "think_tags_neutralized", "n_nodes", "n_calls")}})
        for a, b in q["think"]:
            assert q["ids"][a] == OPEN and q["ids"][b] == CLOSE
        if shown < args.show:
            shown += 1
            print(f"=== {run_dir}/{q['info']['rollout_id']} (row {i}, task {q['info']['task']}, {q['model']}): {n} tokens, "
                  f"{q['turns']} turns, {len(q['think'])} think pairs, readable {sum(q['readable'])}, messages {len(q['messages'])}, "
                  f"tools {[t['function']['name'] for t in q['tools']]}")
            print(q["text"][:600])
            print("...")
            # the first assistant turn's rendered head, to show the think block
            k = q["text"].find("<|im_start|>assistant")
            if k >= 0:
                print("[first assistant turn]", json.dumps(q["text"][k:k + 300]))
    print(summarize(rows, args.max_tokens))
    if args.units:
        print("units:", _quantiles([r["n_units"] for r in rows]))
    print(f"{time.time() - t0:.0f}s")
    if args.out:
        write_plan(rows, args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
