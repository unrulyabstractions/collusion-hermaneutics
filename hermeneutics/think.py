"""Positions of the think tokens that open and close each assistant turn of a rendered Qwen3.8 sequence.

The template renders every assistant turn as "<|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n\\n{content}", so a
turn's <think> (id 248068) follows the three tokens <|im_start|>, "assistant", "\\n" (248045, 74455, 198), and its
</think> (248069) is the first one after it. A literal "<think>" or "</think>" inside a tool output or a user message
tokenizes to the same ids; the prefix guard keeps those out, and a </think> is only taken while a guarded turn is open.
"""
OPEN, CLOSE = 248068, 248069
PREFIX = (248045, 74455, 198)  # <|im_start|> assistant \n


def think_positions(ids):
    """[(i_open, i_close)] per assistant turn, in order; a turn whose </think> is missing is dropped."""
    out = []
    n = len(ids)
    i = 0
    while i < n:
        if ids[i] == OPEN and i >= 3 and tuple(ids[i - 3:i]) == PREFIX:
            j = i + 1
            while j < n and ids[j] != CLOSE:
                j += 1
            if j < n:
                out.append((i, j))
                i = j
        i += 1
    return out
