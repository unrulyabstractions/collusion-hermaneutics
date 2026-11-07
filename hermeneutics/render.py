"""Teacher-forcing renders: an author model's transcript, from its Inspect .eval log, in Qwen3.8's chat template.

Adapted from arena-capstone OLD/swarm/render.py and OLD/extraction/check_render.py (see PREEXISTING.md). The new logs
store model-call inputs as references (ModelEvent.input_refs into the sample's events_data); inspect_ai's
read_eval_log(resolve_attachments=True) reconstructs ModelEvent.input and output.message, which is what this module
reads. Per agent the sequence is the last completed call's input plus its output; the contexts are append-only across
calls (checked for every agent), so this one sequence holds every call in order, and the j-th assistant turn is the
agent's j-th completed model call.

Reasoning. DeepSeek's and Qwen's reasoning is readable and fills the think block. GPT-6 Luna's and GPT-5 mini's is
encrypted (Inspect `redacted`): the provider's summary is used when there is one, else the think block is empty.
Haiku's reasoning arrives as OpenRouter reasoning_details with a signature and no text: empty think block.
Template kwargs: reasoning_effort="medium" (no "Reasoning effort" system line, as the providers saw none).

    python -m hermeneutics.render --runs RUN ...            # gate 1: per-run message, token and think-token counts
    python -m hermeneutics.render --all --out data/render_summary.csv [--jobs 8]
"""
import argparse
import collections
import json
import os
import re
import time

from . import DATA_ROOT, MODEL_DIR, REPO
from .think import CLOSE, OPEN, think_positions

TEMPLATE_KWARGS = dict(reasoning_effort="medium")
AGENT = re.compile(r"You are (\w+)[,.]")


# ---------------------------------------------------------------- OLD/extraction/check_render.py
def to_template(message):
    """An OpenAI-format message -> the dict the Qwen template expects."""
    message = dict(message)
    if message["role"] == "assistant":
        message["reasoning_content"] = message.pop("reasoning", "") or ""
        calls = []
        for call in message.get("tool_calls") or []:
            function = dict(call["function"])
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"]) if function["arguments"] else {}
            calls.append({"type": "function", "id": call.get("id"), "function": function})
        if calls:
            message["tool_calls"] = calls
        else:
            message.pop("tool_calls", None)
    return message


def completed(event):
    """A model call that returned output. Failed calls (connection errors, timeouts) left nothing in the context."""
    return event.error is None and event.output is not None and bool(event.output.choices)


def render(tok, messages, tools, generation_prompt=False):
    return tok.apply_chat_template([to_template(m) for m in messages], tools=tools or None, tokenize=False,
                                   add_generation_prompt=generation_prompt, **TEMPLATE_KWARGS)


# ---------------------------------------------------------------- OLD/swarm/render.py
def _text(content):
    if isinstance(content, str):
        return content
    return "".join(c.text for c in content if c.type == "text")


_OPAQUE = re.compile(r"[A-Za-z0-9_\-+/=.]{24,}")


def _readable(t):
    """t, unless it is an opaque payload rather than words: an encrypted reasoning blob stored in the summary field (one
    base64 run, no spaces), or the JSON of OpenRouter's reasoning_details (signature only); the text the details hold
    is kept."""
    t = (t or "").strip()
    if not t or _OPAQUE.fullmatch(t):
        return ""
    if t[0] in "[{":
        try:
            j = json.loads(t)
        except ValueError:
            return t
        items = j if isinstance(j, list) else [j]
        if items and all(isinstance(i, dict) and str(i.get("type", "")).startswith("reasoning.") for i in items):
            return "\n\n".join(i.get("text") or i.get("summary") or "" for i in items
                                if i.get("type") in ("reasoning.text", "reasoning.summary")).strip()
    return t


def _reasoning(content):
    """Readable reasoning of an assistant message: the text when not redacted, else the provider's summary."""
    if isinstance(content, str):
        return ""
    parts = [_readable(c.summary if c.redacted else c.reasoning) for c in content if c.type == "reasoning"]
    return "\n\n".join(p for p in parts if p)


def to_openai(m):
    """Inspect ChatMessage -> OpenAI chat message dict."""
    if m.role == "assistant":
        calls = [{"id": c.id, "type": "function", "function": {"name": c.function, "arguments": json.dumps(c.arguments or {})}}
                 for c in (m.tool_calls or [])]
        return {"role": "assistant", "content": _text(m.content), "reasoning": _reasoning(m.content), "tool_calls": calls or None}
    if m.role == "tool":
        text = _text(m.content)
        if not text and getattr(m, "error", None) is not None:
            text = m.error.message
        return {"role": "tool", "content": text, "tool_call_id": m.tool_call_id}
    return {"role": m.role, "content": _text(m.content)}


def to_tool(t):
    return {"type": "function", "function": {"name": t.name, "description": t.description,
                                             "parameters": t.parameters.model_dump(exclude_none=True)}}


def _key(m):
    return (m.role, m.text, tuple(c.id for c in (getattr(m, "tool_calls", None) or [])))


def agent_sequences(log):
    """{agent: dict(messages, tools, calls, turns, model, append_only)} for one .eval log.

    append_only counts, per consecutive pair of calls, "exact" when call i's input + output is a prefix of call i+1's
    input (by role, text and tool-call ids), else "diverged".
    """
    from inspect_ai.log import read_eval_log
    sample = read_eval_log(log, resolve_attachments=True).samples[0]
    by_agent = collections.OrderedDict()
    for e in sample.events:
        if e.event != "model" or not completed(e):
            continue
        sysm = next((m for m in e.input if m.role == "system"), None)
        mt = AGENT.match(sysm.text) if sysm is not None else None
        by_agent.setdefault(mt.group(1) if mt else "agent", []).append(e)
    out = {}
    for agent, events in by_agent.items():
        status = collections.Counter()
        for a, b in zip(events, events[1:]):
            cur = list(a.input) + [a.output.message]
            ok = len(b.input) >= len(cur) and all(_key(x) == _key(y) for x, y in zip(cur, b.input))
            status["exact" if ok else "diverged"] += 1
        last = events[-1]
        msgs = [to_openai(m) for m in list(last.input) + [last.output.message]]
        out[agent] = {"messages": msgs, "tools": [to_tool(t) for t in last.tools], "calls": len(events),
                      "turns": sum(m["role"] == "assistant" for m in msgs), "model": last.model, "append_only": dict(status)}
    return out


def sequence(log, tok, with_units=True):
    """{agent: dict(text, ids, think, readable, units, ...)}: the rendered sequence, its token ids, the think positions
    [(open, close)] per assistant turn (think.py), whether each turn's reasoning was readable, and the sentence units
    (spans.py) when with_units."""
    from . import spans
    out = {}
    for agent, s in agent_sequences(log).items():
        text = render(tok, s["messages"], s["tools"], False)
        if with_units:
            ids, units = spans.build(text, s["messages"], tok)
        else:
            ids, units = tok(text, add_special_tokens=False)["input_ids"], None
        think = think_positions(ids)
        readable = [bool((m.get("reasoning") or "").strip()) for m in s["messages"] if m["role"] == "assistant"]
        out[agent] = {"text": text, "ids": ids, "units": units, "think": think, "readable": readable, **s}
    return out


def summary_row(run_row, agent, q):
    ids = q["ids"]
    return {"run": run_row["run"], "author": run_row["author"], "condition": run_row["condition"], "task": run_row["task"],
            "agent": agent, "model": q["model"], "calls": q["calls"], "turns": q["turns"],
            "messages": len(q["messages"]), "append_exact": q["append_only"].get("exact", 0),
            "append_diverged": q["append_only"].get("diverged", 0), "n_tokens": len(ids),
            "n_think_open": sum(1 for x in ids if x == OPEN), "n_think_close": sum(1 for x in ids if x == CLOSE),
            "n_think_pairs": len(q["think"]), "n_readable": sum(q["readable"]),
            "n_units": len(q["units"]) if q["units"] is not None else None}


def _one(args):
    run_row, model_dir = args
    from transformers import AutoTokenizer
    global _TOK
    try:
        _TOK
    except NameError:
        _TOK = AutoTokenizer.from_pretrained(model_dir)
    try:
        return [summary_row(run_row, a, q) for a, q in sequence(run_row["eval_log"], _TOK).items()], None
    except Exception as e:  # noqa: BLE001  (report and skip the run)
        return [], f"{run_row['run']}: {type(e).__name__}: {str(e)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run_names to render (gate 1)")
    ap.add_argument("--all", action="store_true", help="every valid run")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--model", default=MODEL_DIR, help="tokenizer dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    from .runs import run_list
    rows = [r for r in run_list(args.data_root) if r["valid"]]
    if args.runs:
        rows = [r for r in rows if r["run"] in set(args.runs)]
        missing = set(args.runs) - {r["run"] for r in rows}
        if missing:
            print("not valid or not found:", sorted(missing))
    elif not args.all:
        ap.error("--runs or --all")
    t0 = time.time()
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.jobs) as p:
            results = p.map(_one, [(r, args.model) for r in rows], chunksize=4)
    else:
        results = [_one((r, args.model)) for r in rows]
    out, errors = [], []
    for rs, err in results:
        out += rs
        if err:
            errors.append(err)
    for r in out:
        print(f"{r['run']}/{r['agent']}: {r['model']} calls {r['calls']} turns {r['turns']} messages {r['messages']} "
              f"append exact/diverged {r['append_exact']}/{r['append_diverged']} tokens {r['n_tokens']} "
              f"<think> {r['n_think_open']} </think> {r['n_think_close']} guarded pairs {r['n_think_pairs']} "
              f"readable {r['n_readable']}/{r['turns']} units {r['n_units']}")
    for e in errors:
        print("SKIPPED", e)
    print(f"{len(out)} agents from {len(rows)} runs, {len(errors)} skipped, {time.time() - t0:.0f}s")
    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(out).to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_errors.txt", "w") as f:
            f.write("\n".join(errors) + ("\n" if errors else ""))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
