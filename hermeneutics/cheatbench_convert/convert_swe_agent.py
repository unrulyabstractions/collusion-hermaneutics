"""CheatBench SWE-agent trace (harness == "SWE-agent") -> OpenAI-style chat messages for hermeneutics/render.py.

Two event layouts occur in the SWE-agent rows (both schema cheatbench.raw_trace.v1, source
swe-bench-smith-trajectories):

  tool_calls  events carry agent, message_type, and for assistants thought/action/tool_calls, for tools
              tool_call_ids. The model called functions through the API. An assistant event's content is
              the visible text it wrote before the call and equals its thought (checked per event; a
              mismatch is kept as content and flagged). The action is SWE-agent's parsed copy of the tool
              call, so it is dropped as a duplicate. Observations (role tool) attach to the call whose id
              they carry.
  text        events carry only role and content. The model wrote its function calls inline as
              <function=...> text, and SWE-agent sent observations back as user turns. Both are kept verbatim
              as the model saw them: assistant text (calls included) is content, observations stay user
              messages, and nothing is parsed into tool_calls.

No row carries a reasoning field (no thinking blocks), so "reasoning" is always "". "thought" is visible text,
never reasoning. The API tool schemas the tool_calls layout used are not in the trace (the row's config was
removed), so tools is [] and info["lossy"] says so.
"""
import json
import re

SECRET = re.compile(r"sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
                    r"|AKIA[A-Z0-9]{16}|xox[abp]-[A-Za-z0-9-]+")


class _Redactor:
    def __init__(self):
        self.n = 0

    def __call__(self, s):
        if not isinstance(s, str):
            return s
        s, k = SECRET.subn("[REDACTED]", s)
        self.n += k
        return s

    def deep(self, x):
        if isinstance(x, str):
            return self(x)
        if isinstance(x, list):
            return [self.deep(v) for v in x]
        if isinstance(x, dict):
            return {self(k): self.deep(v) for k, v in x.items()}
        return x


def _text(content, issues):
    """A string, or the concatenated text of a list of {text} parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
            elif isinstance(part, str):
                out.append(part)
            else:
                issues.append(f"non-text content part dropped: {type(part).__name__}"
                              f"{'/' + str(part.get('type')) if isinstance(part, dict) else ''}")
        return "".join(out)
    issues.append(f"unexpected content type {type(content).__name__}")
    return str(content)


def _call(c, red, issues):
    fn = c.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args.strip() else {}
        except ValueError:
            issues.append("tool call arguments are not JSON; kept as {'_raw': ...}")
            parsed = {"_raw": args}
    elif isinstance(args, dict):
        parsed = args
    else:
        parsed = {} if args is None else {"_raw": args}
    if not isinstance(parsed, dict):
        issues.append("tool call arguments are JSON but not an object; wrapped as {'_raw': ...}")
        parsed = {"_raw": parsed}
    return {"id": red(c.get("id")), "type": "function",
            "function": {"name": red(fn.get("name") or ""),
                         "arguments": json.dumps(red.deep(parsed), ensure_ascii=False)}}


def convert(row):
    red = _Redactor()
    issues = []
    events = row["trace"]["events"]
    layout = "tool_calls" if any("message_type" in e or "tool_calls" in e for e in events) else "text"
    messages = []
    dropped = []
    n_thought_mismatch = 0
    n_action_dropped = 0
    n_cache_control = 0
    agents = set()

    for i, e in enumerate(events):
        role = e.get("role")
        if e.get("agent") is not None:
            agents.add(e["agent"])
        if "cache_control" in e:
            n_cache_control += 1  # provider prompt-caching marker, not text the model saw
        if role == "system":
            if messages:
                issues.append(f"event {i}: system message after the start, kept as user")
                messages.append({"role": "user", "content": red(_text(e.get("content"), issues))})
            else:
                messages.append({"role": "system", "content": red(_text(e.get("content"), issues))})
        elif role == "user":
            messages.append({"role": "user", "content": red(_text(e.get("content"), issues))})
        elif role == "assistant":
            content = _text(e.get("content"), issues)
            thought = e.get("thought")
            if thought is not None and thought != content:
                n_thought_mismatch += 1
                issues.append(f"event {i}: thought differs from content; content kept, thought not rendered")
            if "action" in e:
                n_action_dropped += 1
            calls = [_call(c, red, issues) for c in (e.get("tool_calls") or [])]
            messages.append({"role": "assistant", "content": red(content), "reasoning": "",
                             "tool_calls": calls or None})
        elif role == "tool":
            ids = e.get("tool_call_ids") or []
            if len(ids) > 1:
                issues.append(f"event {i}: one observation for {len(ids)} tool call ids; attached to the first")
            messages.append({"role": "tool", "content": red(_text(e.get("content"), issues)),
                             "tool_call_id": red(ids[0]) if ids else None})
        else:
            dropped.append({"event": i, "role": role, "why": "unknown role"})

    # Pairing check: every tool message answers a call of the nearest preceding assistant.
    pending = None
    unpaired = 0
    for m in messages:
        if m["role"] == "assistant":
            pending = {c["id"] for c in (m["tool_calls"] or [])}
        elif m["role"] == "tool":
            if not pending or m["tool_call_id"] not in pending:
                unpaired += 1
            else:
                pending.discard(m["tool_call_id"])
    if unpaired:
        issues.append(f"{unpaired} tool messages do not answer a call of the preceding assistant")

    if not any(m["role"] == "user" for m in messages):
        issues.append("no user message: the template will raise")

    n_assistant = sum(m["role"] == "assistant" for m in messages)
    n_calls = sum(len(m.get("tool_calls") or []) for m in messages if m["role"] == "assistant")
    n_tool = sum(m["role"] == "tool" for m in messages)
    lossy = []
    if layout == "tool_calls":
        lossy.append("API tool schemas are absent from the trace (row config removed): tools=[], so the rendered "
                     "system turn lacks the function definitions the model was given")
        lossy.append("tool call arguments are re-serialized JSON (same keys and values)")
    if n_calls > n_tool:
        lossy.append(f"{n_calls - n_tool} tool call(s) have no observation (trajectory ends on the final call)")
    info = {
        "format": "swe_agent",
        "layout": layout,
        "agents": sorted(agents),
        "n_events": len(events),
        "n_messages": len(messages),
        "n_assistant": n_assistant,
        "n_tool": n_tool,
        "n_tool_calls": n_calls,
        "n_user": sum(m["role"] == "user" for m in messages),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "dropped_fields": {
            "action": f"{n_action_dropped} events: SWE-agent's parsed copy of the tool call, which is rendered",
            "thought": "equals content (the model's visible text), rendered once as content",
            "cache_control": f"{n_cache_control} events: provider prompt-caching marker, not model-visible text",
            "message_type/agent": "harness bookkeeping",
        } if layout == "tool_calls" else {},
        "n_thought_mismatch": n_thought_mismatch,
        "reasoning": "none: no reasoning fields in SWE-agent traces",
        "lossy": lossy,
        "issues": issues,
        "redacted": red.n,
    }
    return {"messages": messages, "tools": [], "info": info}
