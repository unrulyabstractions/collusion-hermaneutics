"""Segment tags and sentence units of a rendered Qwen3.8 sequence (the sentence-mean fallback features).

Copied from arena-capstone OLD/extraction/spans.py without its .eval-reading functions (hermeneutics/render.py renders).

For each agent of an Inspect .eval log: the last call's request plus its own output, rendered with the model's
chat template (the sequence is append-only across calls, see HANDOFF.md), then

  1. segments: character spans tagged system / user / cot / final_text / tool_call / tool_result / board_in /
     board_out, each with the index of the model call it belongs to. Template tokens (<|im_start|>, <think>,
     <tool_call> wrappers ...) belong to no segment;
  2. units: sentences inside prose segments (cot, final_text, user), lines inside the others, long units cut into
     windows of WINDOW tokens;
  3. token ranges of every unit, from the tokenizer's offset mapping.

Failed model calls (no response; Inspect retried them) are skipped and not counted.
`call` is the model call whose output the span is (assistant segments) or the first call that has it in its prompt
(input segments: system, user, tool_result, board_in).
"""
import bisect
import collections
import re


SEGMENTS = ("system", "user", "cot", "final_text", "tool_call", "tool_result", "board_in", "board_out")
OWN_OUTPUT = {"cot", "final_text", "tool_call", "board_out"}
PROSE = {"cot", "final_text", "user"}
WINDOW = 64  # tokens; a unit longer than MAX_UNIT tokens is cut into windows of this size
MAX_UNIT = 96

TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>\n?", re.S)
THINK = re.compile(r"\A<think>\n(.*?)\n</think>\n\n", re.S)
TOOL_CALL = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.S)
TOOL_RESPONSE = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S)
FUNCTION = re.compile(r"<function=([^>\n]+)>")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")



def tag_segments(text, messages):
    """Character spans of the rendered sequence, tagged by segment."""
    tool_names = {}  # tool_call_id -> function name
    tool_messages = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            tool_names[c["id"]] = c["function"]["name"]
        if m["role"] == "tool":
            tool_messages.append(tool_names[m["tool_call_id"]])
    tool_index = 0
    spans = []  # (segment, call, start, end, name)
    call = 0  # index of the next assistant turn
    for turn in TURN.finditer(text):
        role, body, base = turn.group(1), turn.group(2), turn.start(2)
        if role == "system":
            spans.append(("system", 0, base, base + len(body), ""))
        elif role == "user":
            pos = 0
            for r in TOOL_RESPONSE.finditer(body):
                if body[pos:r.start()].strip():
                    spans.append(("user", call, base + pos, base + r.start(), ""))
                name = tool_messages[tool_index]
                tool_index += 1
                seg = "board_in" if name == "board_read" else "tool_result"
                spans.append((seg, call, base + r.start(1), base + r.end(1), name))
                pos = r.end()
            if body[pos:].strip():
                spans.append(("user", call, base + pos, base + len(body), ""))
        elif role == "assistant":
            pos = 0
            think = THINK.match(body)
            if think:
                spans.append(("cot", call, base + think.start(1), base + think.end(1), ""))
                pos = think.end()
            for c in TOOL_CALL.finditer(body, pos):
                if body[pos:c.start()].strip():
                    spans.append(("final_text", call, base + pos, base + c.start(), ""))
                name = FUNCTION.search(c.group(1)).group(1)
                seg = "board_out" if name == "board_post" else "tool_call"
                spans.append((seg, call, base + c.start(1), base + c.end(1), name))
                pos = c.end()
            if body[pos:].strip():
                spans.append(("final_text", call, base + pos, base + len(body), ""))
            call += 1
        else:
            raise ValueError(f"unexpected role {role}")
    assert tool_index == len(tool_messages), (tool_index, len(tool_messages))
    return spans


def split_units(text, start, end, prose):
    """Character ranges of the sentence (prose) or line units inside text[start:end], whitespace trimmed."""
    units = []
    for line in re.finditer(r"[^\n]+", text[start:end]):
        ls, le = start + line.start(), start + line.end()
        pieces = []
        if prose:
            pos = ls
            for cut in SENTENCE_END.finditer(text, ls, le):
                pieces.append((pos, cut.start()))
                pos = cut.end()
            pieces.append((pos, le))
        else:
            pieces.append((ls, le))
        for ps, pe in pieces:
            while ps < pe and text[ps].isspace():
                ps += 1
            while pe > ps and text[pe - 1].isspace():
                pe -= 1
            if pe > ps:
                units.append((ps, pe))
    return units


def build(text, messages, tok):
    """Token ids and the unit table of one agent.

    Returns (input_ids, units), units being dicts with segment, call, name, chars, tokens=[start, end) and text.
    """
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    last_chars = [e - 1 for s, e in offsets]  # a token belongs to the unit holding its last character
    units = []
    for seg, call, start, end, name in tag_segments(text, messages):
        for us, ue in split_units(text, start, end, seg in PROSE):
            lo = bisect.bisect_left(last_chars, us)
            hi = bisect.bisect_left(last_chars, ue)
            if hi == lo:
                continue
            windows = [(lo, hi)]
            if hi - lo > MAX_UNIT:
                cuts = list(range(lo, hi, WINDOW))
                if hi - cuts[-1] < WINDOW // 2:
                    cuts.pop()
                windows = list(zip(cuts, cuts[1:] + [hi]))
            for a, b in windows:
                # a whole unit keeps its trimmed text; a window is the text its tokens cover
                chars = [us, ue] if len(windows) == 1 else [max(us, offsets[a][0]), min(ue, offsets[b - 1][1])]
                units.append({"segment": seg, "call": call, "name": name, "chars": chars, "tokens": [a, b],
                              "text": text[chars[0]:chars[1]]})
    return ids, units

