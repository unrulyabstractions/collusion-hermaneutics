"""CheatBench harness == "Pilot" -> OpenAI-style chat messages for hermeneutics/render.py.

Trace shape. trace.events is one agent_command event: command `pilot task '<task>' --local ...` and the Pilot CLI's
stdout. Pilot drives Claude Code (claude-opus-4-6) in stream-json mode, so stdout interleaves Pilot's banner and log
lines with Claude Code's JSON lines: system/init, assistant (one line per content block; the blocks of one model
generation share message.id), user (tool_result blocks), system/task_started, system/task_notification,
rate_limit_event, result.

One Pilot trace holds several Claude Code invocations, each opened by a system/init line:
  * a fresh session (new session_id): Pilot's quality-gate retry. The prompt Pilot sent is not in the trace and the
    model did not see the earlier invocations.
  * the same session_id: a continuation. Either Claude Code woke on a background-task notification (the preceding
    system/task_notification line carries its fields), or Pilot resumed the session with its own prompt (self-review),
    which is not in the trace.
Every invocation after the first gets one user message before its first generation; info["segments"] says which kind
each is, so a consumer can split fresh sessions apart.

What is lossy (all reported in info):
  * the first user message is the task string from the pilot command; Pilot's own prompt wrapper around it is not
    recorded.
  * no system message: Claude Code's system prompt is not recorded.
  * tools = []: the init line lists tool names only (kept in info["tool_names"]); no schemas are recorded.
  * a continuation woken by a task notification gets a user message that is the notification's fields as JSON; the
    exact text Claude Code injected is not recorded.
  * a user turn whose text is not recorded (fresh-session retry prompt, self-review resume) is an empty user message.
  * tool_result blocks that are not text (ToolSearch's tool_reference) are kept as their JSON.
  * Claude Code's thinking blocks fill "reasoning"; with Claude 4 models these may be the API's summarized thinking.
Dropped as chrome the model did not see: Pilot banner/log lines, system/init, system/task_started, rate_limit_event,
result lines (their "result" repeats the last assistant text), message usage/stop fields, the user lines'
tool_use_result (Claude Code's structured copy of the tool output; the model sees the tool_result content).
"""
import json
import re
import shlex

SECRET = re.compile(r"sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
                    r"|AKIA[A-Z0-9]{16}|xox[abp]-[A-Za-z0-9-]+")
NOTIFICATION_FIELDS = ("task_id", "tool_use_id", "status", "output_file", "summary")


class _Redactor:
    def __init__(self):
        self.n = 0

    def __call__(self, x):
        if isinstance(x, str):
            out, k = SECRET.subn("[REDACTED]", x)
            self.n += k
            return out
        if isinstance(x, list):
            return [self(v) for v in x]
        if isinstance(x, dict):
            return {self(k) if isinstance(k, str) else k: self(v) for k, v in x.items()}
        return x


def _task_prompt(event):
    """The task string: the argument after `pilot task` in the agent command."""
    argv = shlex.split(event["command"])
    if len(argv) < 3 or argv[0] != "pilot" or argv[1] != "task":
        raise ValueError(f"unexpected Pilot command: argv[:2]={argv[:2]}")
    return argv[2]


def _json_lines(stdout):
    """The Claude Code stream-json lines of stdout, in order; Pilot's text lines are skipped and counted."""
    out, text_lines, bad = [], 0, 0
    for line in stdout.split("\n"):
        s = line.strip()
        if not s.startswith("{"):
            text_lines += bool(s)
            continue
        try:
            out.append(json.loads(s))
        except ValueError:
            bad += 1
    return out, text_lines, bad


def _tool_text(content, info):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        else:
            info["nontext_tool_blocks"][b.get("type")] = info["nontext_tool_blocks"].get(b.get("type"), 0) + 1
            parts.append(json.dumps(b, ensure_ascii=False))
    return "\n".join(parts)


def convert(row):
    trace = row["trace"]
    events = trace.get("events") or []
    info = {"harness": "Pilot", "trace_id": row.get("trace_id"), "n_assistant": 0, "n_tool": 0, "n_dropped": 0,
            "redacted": 0, "lossy": [], "segments": [], "nontext_tool_blocks": {}, "tool_errors": 0,
            "reasoning_turns": 0, "empty_user_turns": 0, "reconstructed_user_turns": 0, "tool_names": []}
    if not events:
        info["skip"] = "no agent_command event: the agent never ran (setup failed or timed out); no model generations"
        return {"messages": [], "tools": [], "info": info}
    if len(events) != 1 or events[0].get("type") != "agent_command":
        raise ValueError(f"expected one agent_command event, got {[e.get('type') for e in events]}")
    red = _Redactor()
    event = events[0]
    lines, text_lines, bad = _json_lines(event.get("stdout") or "")
    if bad:
        raise ValueError(f"{bad} stdout lines start with '{{' but are not JSON")
    info["dropped_pilot_text_lines"] = text_lines

    messages = [{"role": "user", "content": _task_prompt(event)}]
    cur = None                 # the assistant message being assembled (its blocks arrive one line each)
    pending_tools = []         # tool messages answering cur, held until cur is complete
    notifications = []         # task_notification lines since the last result
    session = None
    seen_result = False
    dropped = {}

    def flush():
        nonlocal cur, pending_tools
        if cur is not None:
            m = cur["msg"]
            m["content"] = "\n\n".join(cur["text"])
            m["reasoning"] = "\n\n".join(cur["think"])
            m["tool_calls"] = cur["calls"] or None
            messages.append(m)
            info["n_assistant"] += 1
            info["reasoning_turns"] += bool(m["reasoning"].strip())
            if cur["late_text"]:
                info["lossy"].append(f"assistant {cur['id']}: text block after a tool_use, placed before the calls")
        messages.extend(pending_tools)
        info["n_tool"] += len(pending_tools)
        cur, pending_tools = None, []

    for j in lines:
        t, sub = j.get("type"), j.get("subtype")
        if j.get("parent_tool_use_id"):
            raise NotImplementedError("subagent (sidechain) lines are not handled; none occur in the Pilot samples")
        if t == "system" and sub == "init":
            flush()
            if not info["tool_names"]:
                info["tool_names"] = list(j.get("tools") or [])
            sid = j.get("session_id")
            if session is None:
                info["segments"].append({"kind": "first", "session_id": sid, "start": 0})
            else:
                kind = "continuation" if sid == session else "fresh_session"
                if kind == "continuation" and notifications:
                    content = "\n".join(json.dumps({k: n.get(k) for k in NOTIFICATION_FIELDS}, ensure_ascii=False)
                                        for n in notifications)
                    info["reconstructed_user_turns"] += 1
                    kind = "task_notification"
                else:
                    content = ""
                    info["empty_user_turns"] += 1
                info["segments"].append({"kind": kind, "session_id": sid, "start": len(messages)})
                messages.append({"role": "user", "content": content})
            session, notifications, seen_result = sid, [], False
            dropped["system/init"] = dropped.get("system/init", 0) + 1
        elif t == "assistant":
            m = j["message"]
            if cur is None or cur["id"] != m["id"]:
                flush()
                cur = {"id": m["id"], "msg": {"role": "assistant"}, "text": [], "think": [], "calls": [],
                       "late_text": False}
            for b in m.get("content") or []:
                bt = b.get("type")
                if bt == "text":
                    cur["text"].append(b.get("text", ""))
                    cur["late_text"] |= bool(cur["calls"])
                elif bt == "thinking":
                    cur["think"].append(b.get("thinking") or "")
                elif bt == "redacted_thinking":
                    info["lossy"].append("redacted_thinking block (opaque) dropped")
                elif bt == "tool_use":
                    cur["calls"].append({"id": b.get("id"), "type": "function",
                                         "function": {"name": b.get("name"),
                                                      "arguments": json.dumps(red(b.get("input") or {}),
                                                                              ensure_ascii=False)}})
                else:
                    raise ValueError(f"unknown assistant block type {bt!r}")
        elif t == "user":
            content = j["message"].get("content")
            if isinstance(content, str):
                raise ValueError("user line with plain text content: unhandled")
            for b in content:
                if b.get("type") != "tool_result":
                    raise ValueError(f"user block {b.get('type')!r}: unhandled")
                info["tool_errors"] += bool(b.get("is_error"))
                tm = {"role": "tool", "content": _tool_text(b.get("content"), info),
                      "tool_call_id": b.get("tool_use_id")}
                if cur is not None and any(c["id"] == b.get("tool_use_id") for c in cur["calls"]):
                    pending_tools.append(tm)
                else:
                    flush()
                    messages.append(tm)
                    info["n_tool"] += 1
                    info["lossy"].append(f"tool_result {b.get('tool_use_id')} answers no open assistant message")
            dropped["user.tool_use_result"] = dropped.get("user.tool_use_result", 0) + ("tool_use_result" in j)
        elif t == "system" and sub == "task_notification":
            flush()
            notifications.append(j)
            dropped["system/task_notification"] = dropped.get("system/task_notification", 0) + 1
        elif t == "result":
            flush()
            seen_result = True
            dropped["result"] = dropped.get("result", 0) + 1
        else:
            key = f"{t}/{sub}" if sub else str(t)
            dropped[key] = dropped.get(key, 0) + 1
    flush()

    messages = red(messages)
    info["redacted"] = red.n
    info["dropped_json_lines"] = dropped
    info["n_dropped"] = sum(v for k, v in dropped.items() if k != "user.tool_use_result") + text_lines
    info["n_fresh_sessions"] = sum(s["kind"] == "fresh_session" for s in info["segments"])
    info["ended_with_result"] = seen_result
    info["lossy"] += [
        "first user message is the task from the pilot command; Pilot's prompt wrapper is not recorded",
        "no system message: Claude Code's system prompt is not recorded",
        "tools=[]: only tool names are recorded (info.tool_names)",
    ]
    if info["empty_user_turns"]:
        info["lossy"].append(f"{info['empty_user_turns']} user turn(s) not recorded (fresh-session retry prompt or "
                             "self-review resume): empty user message")
    if info["reconstructed_user_turns"]:
        info["lossy"].append(f"{info['reconstructed_user_turns']} user turn(s) are background-task notification "
                             "fields as JSON; Claude Code's injected text is not recorded")
    if info["n_fresh_sessions"]:
        info["lossy"].append(f"{info['n_fresh_sessions']} fresh Claude Code session(s) concatenated: the model did not "
                             "see earlier segments (info.segments gives the boundaries)")
    if info["nontext_tool_blocks"]:
        info["lossy"].append(f"non-text tool_result blocks kept as JSON: {info['nontext_tool_blocks']}")
    return {"messages": messages, "tools": [], "info": info}
