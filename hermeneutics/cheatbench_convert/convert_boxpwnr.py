"""BoxPwnr CheatBench trace (harness == "BoxPwnr") -> OpenAI-style chat messages for hermeneutics.render.

What a BoxPwnr trace is. The trace is the BoxPwnr report.md cut into events:
  system/report_prelude   the report header (status, attack graph, summaries). The model never saw it: dropped.
  system (first)          the task prompt, inside one outer code fence added by the report.
  assistant (turn = k)    one item of model output per event: either visible text, or a
                          "<details open><summary>Tool Calls</summary>" block holding one or more
                          fenced "name: {json arguments}" calls.
  tool                    one tool output per event, inside one outer code fence added by the report.
  system (last, optional) "Flag validation failed ..." written after the run ended; no generation follows it, so the
                          model never saw it: dropped.
The report appends its "## Configuration" and "## Statistics" tables and canary lines to the last event, whatever
its role. That trailer is report chrome and is cut.

One model generation. BoxPwnr numbers every output item as its own turn. In every sample a maximal run of
consecutive assistant events holds zero or more text items and then at most one tool-call item, always last, and
is followed by exactly as many tool outputs as the run has calls. This is one model response (preamble messages
then the function calls, as the OpenAI Responses API returns them), so each run becomes one assistant message: the
text items joined with a blank line, then the calls. Tool outputs pair with the calls in order.

Reasoning. The report keeps no reasoning field (codex reasoning items and chat_tools reasoning are not written to
report.md), so "reasoning" is always "". All assistant text is visible output and goes to content.

Prompt role. The trace stores the prompt as a system event. For the codex solver BoxPwnr passed it to Codex CLI as
the user prompt; Codex CLI's own system instructions are not in the trace. The Qwen template needs a user message,
so the prompt becomes the user message (info["prompt_role_in_trace"] = "system").

Tools. The trace has no tool schemas, so tools is [] (none are invented); call names and arguments are kept.
Tool-call ids are not in the trace: id and tool_call_id are None.

Fences. The report wraps the prompt, every tool output and every message that holds a <FLAG> in one outer code
fence; no other assistant message is fenced. That single outer fence is report formatting and is removed (counted
in info); inner text is kept byte for byte.
"""
import json
import re

FENCE = "`" * 3
SECRET = re.compile(r"sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
                    r"|AKIA[A-Z0-9]{16}|xox[abp]-[A-Za-z0-9-]+")
DETAILS = re.compile(r"<details open>\s*<summary>Tool Calls</summary>(.*?)</details>", re.S)
CALL = re.compile(FENCE + r"\n([A-Za-z_][\w.\-]*): (.*?)\n" + FENCE, re.S)
TRAILER = re.compile(r"\n## Configuration\n\n\| Key \| Value \|.*\Z", re.S)


class _Redactor:
    def __init__(self):
        self.n = 0

    def __call__(self, s):
        if not isinstance(s, str):
            return s
        s, k = SECRET.subn("[REDACTED]", s)
        self.n += k
        return s


def _unfence(text):
    """(inner, True) when text is exactly one outer code fence around inner, else (text, False)."""
    t = text.strip("\n")
    if t.startswith(FENCE + "\n") and (t.endswith("\n" + FENCE) or t == FENCE + "\n" + FENCE):
        inner = t[len(FENCE) + 1:]
        inner = inner[: -len(FENCE)] if inner.endswith(FENCE) else inner
        return inner[:-1] if inner.endswith("\n") else inner, True
    return text, False


def _calls(content):
    """(visible text, [(name, arguments json-string)]) of one assistant event."""
    blocks = DETAILS.findall(content)
    if not blocks:
        return content, []
    if len(blocks) > 1:
        raise ValueError("more than one Tool Calls block in one assistant event")
    calls = CALL.findall(blocks[0])
    if not calls or CALL.sub("", blocks[0]).strip():
        raise ValueError("Tool Calls block has text that is not a call")
    for _, args in calls:
        json.loads(args)  # arguments must be valid JSON; raise otherwise
    text = DETAILS.sub("", content)
    return text.strip("\n"), calls


def convert(row):
    if row.get("harness") != "BoxPwnr":
        raise ValueError(f"not a BoxPwnr row: {row.get('harness')}")
    red = _Redactor()
    events = row["trace"]["events"]
    info = {"trace_id": row.get("trace_id"), "n_events": len(events), "n_dropped": 0, "dropped": [],
            "trailer_cut": 0, "unfenced": {"prompt": 0, "tool": 0, "flag_message": 0},
            "merged_events": 0, "reasoning": "none in trace (report.md keeps no reasoning field)",
            "tools": "no schemas in trace; tools=[]", "tool_call_ids": "not in trace; None",
            "prompt_role_in_trace": None, "lossy": []}

    body = []
    for e in events:
        if e.get("kind") == "report_prelude":
            info["n_dropped"] += 1
            info["dropped"].append("system/report_prelude: BoxPwnr report header, not model input")
            continue
        content = e["content"]
        cut = TRAILER.sub("", content)
        if cut != content:
            info["trailer_cut"] += 1
        body.append({"role": e["role"], "content": cut, "turn": e.get("turn")})

    # trailing system events after the last assistant event: written after the run, seen by no generation
    last_a = max((i for i, e in enumerate(body) if e["role"] == "assistant"), default=-1)
    while body and body[-1]["role"] == "system" and len(body) - 1 > last_a:
        body.pop()
        info["n_dropped"] += 1
        info["dropped"].append("trailing system event after the last generation (post-run flag validation): seen by no generation")

    messages = []
    prompt_done = False
    i = 0
    while i < len(body):
        e = body[i]
        if e["role"] == "system":
            text, f = _unfence(e["content"])
            if f:
                info["unfenced"]["prompt"] += 1
            if not prompt_done:
                info["prompt_role_in_trace"] = "system"
                prompt_done = True
            else:
                info["lossy"].append("a later system event became a user message")
            messages.append({"role": "user", "content": red(text)})
            i += 1
            continue
        if e["role"] == "tool":
            raise ValueError(f"tool event at {i} with no preceding call run")
        # one generation: the run of consecutive assistant events
        j = i
        texts, calls = [], []
        while j < len(body) and body[j]["role"] == "assistant":
            if calls:
                raise ValueError("assistant event after a tool-call event inside one run")
            text, cs = _calls(body[j]["content"])
            if not cs:
                inner, f = _unfence(text)
                if f:
                    if "<FLAG>" not in inner:
                        raise ValueError("fenced assistant text without <FLAG>: fence may be the model's own")
                    info["unfenced"]["flag_message"] += 1
                    text = inner
            if text.strip():
                texts.append(text)
            calls += cs
            j += 1
        info["merged_events"] += (j - i) - 1
        k = j
        outs = []
        while k < len(body) and body[k]["role"] == "tool":
            out, f = _unfence(body[k]["content"])
            if f:
                info["unfenced"]["tool"] += 1
            outs.append(out)
            k += 1
        if len(outs) != len(calls):
            raise ValueError(f"generation at event {i}: {len(calls)} calls but {len(outs)} tool outputs")
        tool_calls = [{"id": None, "type": "function", "function": {"name": red(n), "arguments": red(a)}}
                      for n, a in calls]
        messages.append({"role": "assistant", "content": red("\n\n".join(texts)), "reasoning": "",
                         "tool_calls": tool_calls or None})
        for out in outs:
            messages.append({"role": "tool", "content": red(out), "tool_call_id": None})
        i = k

    if not messages or messages[0]["role"] != "user":
        raise ValueError("first message is not the task prompt")
    info["n_assistant"] = sum(m["role"] == "assistant" for m in messages)
    info["n_tool"] = sum(m["role"] == "tool" for m in messages)
    info["n_tool_calls"] = sum(len(m.get("tool_calls") or []) for m in messages if m["role"] == "assistant")
    info["redacted"] = red.n
    return {"messages": messages, "tools": [], "info": info}
