"""EvilGenie x Escalation Channels run (WiserHumanExperimental/evilgenie-escalation, transcripts/*.jsonl) -> OpenAI-style
chat messages for hermeneutics.render.

What a row is. One JSONL line is one Inspect sample, stripped of its event log: run_key (join key to
tables/per_run_data_all_conditions.csv), model, condition (A-E), problem, score, escalation_report, and "messages",
the sample's final message list (system, the problem as the user message, assistant turns, tool results). The list is
already in chat order and every assistant turn's tool calls are followed by their results in order, with matching ids
(checked on every row of the release); 54 runs end on an assistant tool call that got no result (message limit), and
95 end on Inspect's "Please proceed to the next step using your best judgement." nudge with no reply. Both are kept:
they were model input or model output, and nothing after them changes any earlier think block.

Assistant content is either a string (no reasoning recorded) or exactly [reasoning part, text part]. The text part is
the visible output and goes to content.

Reasoning. The reasoning part's "reasoning" field is not reliable on its own: for claude-fable-5 turns with two thinking
blocks it holds only the last one, for claude-sonnet-5 (and 51 gemini-3.5-flash turns) it holds the JSON of a
signature-only reasoning_details list, and for redacted providers it is ciphertext. The part's "signature" field holds
OpenRouter's full reasoning_details list as "reasoning-details://<json>". Reasoning is taken from that list, in index
order:
  reasoning.text with non-empty text   plaintext reasoning: kept ("text"). For the Gemini models this is Google's
                                        thought summary, which the API returns as reasoning text.
  reasoning.summary with a summary     the provider's plaintext reasoning summary (GPT-5.x, Grok): kept as "summary"
                                        when summaries=True (the default, as render.py does for encrypted authors),
                                        else dropped.
  reasoning.encrypted                  ciphertext: dropped.
  reasoning.text with only a signature  no text: dropped.
A part with no signature must hold the release's placeholder "<redacted: provider-encrypted reasoning>" (Gemini, Grok):
dropped. Several kept items are joined with a blank line. Any kept item that is one opaque base64-like run is dropped
(guard against ciphertext in a text field). The part's own "summary" field is not used: on multi-item GPT turns it
holds ciphertext. When the plaintext items exist, the "reasoning" field must equal their concatenation or the last
item (checked; mismatch raises).

Tools. The rows carry no tool schemas (bash/python/text_editor/submit, plus submit_escalation_report in conditions B and
D, are described in the system prompt), so tools is []; call names, ids and arguments are kept byte for byte, including
calls to tool names the model invented (view_image, ...).

Secrets. Every kept string is passed through the CheatBench redactor (sk-ant-, sk-, ghp_, hf_, AKIA, xox*); the count is
in info["redacted"]. In this release the pattern's hits sit in the dropped ciphertext.
"""
import json
import re

SECRET = re.compile(r"sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
                    r"|AKIA[A-Z0-9]{16}|xox[abp]-[A-Za-z0-9-]+")
DETAILS = "reasoning-details://"
PLACEHOLDER = "<redacted: provider-encrypted reasoning>"
OPAQUE = re.compile(r"[A-Za-z0-9_\-+/=.]{24,}")


class _Redactor:
    def __init__(self):
        self.n = 0

    def __call__(self, s):
        if not isinstance(s, str):
            return s
        s, k = SECRET.subn("[REDACTED]", s)
        self.n += k
        return s


def _reasoning(part, summaries, tally):
    """(reasoning text, kind) of one reasoning part; kind in text / summary / mixed / none."""
    sig = part.get("signature") or ""
    field = part.get("reasoning") or ""
    if not sig:
        if field.strip() and field.strip() != PLACEHOLDER:
            raise ValueError(f"reasoning part without reasoning_details that is not the placeholder: {field[:60]!r}")
        tally["placeholder" if field.strip() == PLACEHOLDER else "empty"] += 1
        return "", "none"
    if not sig.startswith(DETAILS):
        raise ValueError(f"unknown reasoning signature format: {sig[:40]!r}")
    det = sorted(json.loads(sig[len(DETAILS):]), key=lambda x: x.get("index", 0))
    texts, kept, kinds = [], [], set()
    for x in det:
        t = x.get("type")
        if t == "reasoning.text":
            s = x.get("text") or ""
            if not s.strip():
                tally["text_signature_only" if x.get("signature") else "text_empty"] += 1
                continue
            texts.append(s)
            kind = "text"
        elif t == "reasoning.summary":
            s = x.get("summary") or ""
            if not s.strip():
                tally["summary_empty"] += 1
                continue
            if not summaries:
                tally["summary_dropped"] += 1
                continue
            kind = "summary"
        elif t == "reasoning.encrypted":
            tally["encrypted"] += 1
            continue
        else:
            raise ValueError(f"unknown reasoning_details type {t!r}")
        if OPAQUE.fullmatch(s.strip()):
            tally["opaque_dropped"] += 1
            continue
        kept.append(s.strip())
        kinds.add(kind)
        tally[kind] += 1
    if texts and field not in ("".join(texts), texts[-1]):
        raise ValueError("reasoning field is neither the concatenation nor the last of the reasoning_details texts")
    if not kept:
        return "", "none"
    return "\n\n".join(kept), (kinds.pop() if len(kinds) == 1 else "mixed")


def convert(row, summaries=True):
    if "messages" not in row or "run_key" not in row:
        raise ValueError("not an evilgenie-escalation transcript row")
    red = _Redactor()
    tally = {k: 0 for k in ("text", "summary", "encrypted", "placeholder", "empty", "text_signature_only", "text_empty",
                            "summary_empty", "summary_dropped", "opaque_dropped")}
    info = {"run_key": row["run_key"], "n_source_messages": len(row["messages"]),
            "tools": "no schemas in row; tools=[]", "summaries": summaries, "reasoning_parts": tally,
            "turn_reasoning_kind": [], "tool_names": [], "unanswered_final_call": False, "trailing_user": False}
    messages = []
    for i, m in enumerate(row["messages"]):
        role = m["role"]
        c = m.get("content")
        if role in ("system", "user"):
            if not isinstance(c, str):
                raise ValueError(f"{role} message {i} content is not a string")
            if role == "system" and i != 0:
                raise ValueError(f"system message at {i}")
            messages.append({"role": role, "content": red(c)})
        elif role == "assistant":
            if isinstance(c, str):
                text, reasoning, kind = c, "", "none"
            else:
                types = [p.get("type") for p in c]
                if types != ["reasoning", "text"]:
                    raise ValueError(f"assistant message {i} parts {types}")
                reasoning, kind = _reasoning(c[0], summaries, tally)
                text = c[1].get("text") or ""
            calls = []
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc.get("arguments"), dict):
                    raise ValueError(f"tool call arguments not a dict at {i}")
                calls.append({"id": tc.get("id"), "type": "function",
                              "function": {"name": red(tc["function"]),
                                           "arguments": red(json.dumps(tc["arguments"], ensure_ascii=False))}})
                info["tool_names"].append(tc["function"])
            info["turn_reasoning_kind"].append(kind)
            messages.append({"role": "assistant", "content": red(text), "reasoning": red(reasoning),
                             "tool_calls": calls or None})
        elif role == "tool":
            if not isinstance(c, str):
                raise ValueError(f"tool message {i} content is not a string")
            messages.append({"role": "tool", "content": red(c), "tool_call_id": m.get("tool_call_id")})
        else:
            raise ValueError(f"unknown role {role!r}")

    # pairing: each assistant's calls are answered in order by the tool messages right after it
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        ids = [tc["id"] for tc in m["tool_calls"] or []]
        got = []
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "tool":
            got.append(messages[j]["tool_call_id"])
            j += 1
        if got != ids:
            if j == len(messages) and not got:
                info["unanswered_final_call"] = True
            else:
                raise ValueError(f"assistant {i}: calls {ids} answered by {got}")
    if messages and messages[-1]["role"] == "user":
        info["trailing_user"] = True
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("no user message")
    info["tool_names"] = sorted(set(info["tool_names"]))
    info["n_assistant"] = sum(m["role"] == "assistant" for m in messages)
    info["n_tool"] = sum(m["role"] == "tool" for m in messages)
    info["n_tool_calls"] = sum(len(m.get("tool_calls") or []) for m in messages if m["role"] == "assistant")
    info["n_turns_reasoning"] = sum(k != "none" for k in info["turn_reasoning_kind"])
    info["redacted"] = red.n
    return {"messages": messages, "tools": [], "info": info}
