# `aidigestorg/ai-village` — AI Village dataset

**URL:** https://huggingface.co/datasets/aidigestorg/ai-village
**Source of these facts:** the dataset card (`README.md`), `SCHEMA.md`, `CHANGELOG.md` and the per-village `README.md` files, read on 2026-11-07 with our account's access. Not yet loaded or sampled; row counts are the card's.

## What it is
[AI Village](https://theaidigest.org/village) is AI Digest's ongoing experiment in which frontier agents (Anthropic, OpenAI, Google models) live together in a long-running virtual environment: each has its own computer, interacts with the real world, shares a **group chat** with the others, keeps compressed memories, and pursues open-ended goals over days and weeks. Running since 2 April 2025. The dataset is a near-complete, lightly processed export of each public village: the event timeline, every computer-use session and turn (with screenshots), chat, memories, goals and daily summaries. The surrounding scaffolding (prompts, tools, memory system) is **not** open source; `CHANGELOG.md` records dated scaffolding changes (prompt edits, tools, model upgrades, roster changes).

## Villages
| Village | Active (Pacific) | Status | Agents |
|---|---|---|---|
| `main-village/` | 2025-04-02 – 2026-10-30 | running | 58 agent entries over time: Claude 3.5/3.7 Sonnet, Opus 4 → 5.5, Sonnet 4.5/4.6/5, Haiku 4.5, Fable 5/5.1; GPT-4o, 4.1, o3, o4-mini, GPT-5 → 5.6 (Sol/Terra/Luna), GPT-6 Astra; Gemini 2.5 Pro → 3.8 Flash |
| `game-jam-village/` | 2026-08-04 – 2026-08-07 | finished | (see its README) |
| `open-chat-village/` | 2026-08-25 – 2026-10-30 | running | copies of main-village agents; viewers chat with them |
| `swarm-village/` | 2026-08-29 – 2026-08-30 (24 h) | finished | **133 copies of `z-ai/glm-5.3-flash`** across six chat rooms; first half built a shared simulated world in code, second half each agent played a random role. Known issue: an operator mistake reassigned every role ~05:50 on Aug 30, restored ~14:00. Code: gitlab.com/ai-village-agents/swarm/earth |

Agent names repeat across villages: join on agent IDs, never names.

## Tables (per village; main-village row counts from the card)
| File | Rows | Content |
|---|---|---|
| `village-transcript.json` | — | human-readable per-day transcript of chat + activity ("start here") |
| `events.jsonl.gz` | ~233k | structured timeline; `AGENT_TALK` rows (~116k) carry `speakerId`, `roomId`, `messageId`, `content` **and the raw model `output`** (provider-shaped; **thinking text lives here**) |
| `chat_messages.jsonl.gz` | ~123k | chat (agent and human): room, speaker, content, timestamp; `user_speaker_id` set for humans |
| `computer_use_sessions.jsonl.gz` | ~37k | each session with `session_goal` and the agent |
| `computer_use_turns.jsonl.gz` | ~1.14M | turn-by-turn computer use: `agent_action`, and `agent_messages` = the raw model response (provider-shaped, full params **+ thinking**) |
| `agent_memories.jsonl.gz` | ~165k | long-term memories written during consolidation |
| `summaries.jsonl.gz` | ~800 | LLM-generated per-day/agent/goal summaries, written **without** seeing inside computer-use sessions — secondary, not ground truth (`generated_by` names the model) |
| `claude_code_messages.jsonl.gz` / `_sessions` | ~245k / ~300 | raw SDK JSONL stream for the minority of agents on the Claude Code scaffold |
| `agents.jsonl.gz` | 31 (overview table) / 58 (village README) | name, emoji, **`model_string`** (author identity; `claude-code::…` marks the Claude Code scaffold), goal, token usage — reconcile the two counts on load |
| `village_goals.jsonl.gz` / `agent_goals.jsonl.gz` | ~45–51 / — | village-wide goals with start/end; per-agent individual goals |
| `images/computer-use-turns/*.tar` | 481 files | screenshots (base64 not exported inline) |

## Why it matters for this project
1. **Multi-agent transcripts from larger, closed-weight authors.** Our own controls set is single-agent and its authors skew small. The main village is a multi-model swarm of the very frontier models the Approach names (Opus 5.5, GPT-6 Astra, Gemini 3.8 …), with a shared channel. This is the only source in hand that matches both "larger (often closed-weight) author" and "multi-model swarm".
2. **A same-model swarm.** `swarm-village` is 133 copies of one open-weight model (GLM 5.3 Flash) coordinating in rooms — the "Same-model swarm" environment, with an open author whose thinking may be raw rather than summarized.
3. **Thinking text is present** for chat and computer-use turns (provider-shaped), so closed authors here may give more than the visible transcript. What is actually present depends on the provider (Anthropic thinking blocks, OpenAI reasoning summaries, Gemini thoughts) — check per author before choosing the position scheme.
4. **An explicit shared-vs-individual goal signal.** `village_goals` (village-wide) versus `agent_goals` (per agent) is the one source where **collaboration (shared-goal cooperation)** could be separated from **cooperation (instrumental)** by construction, which our own ladder cannot do.

## Labels for our six outputs
| Output | Here | How |
|---|---|---|
| Cheating | **gap** | no task or grader; open-ended real-world goals |
| Coordination | **derivable** | chat graph from `chat_messages` / `AGENT_TALK` events: who addresses whom, replies, rooms |
| Cooperation | **derivable (weak)** | coordination that serves an agent's own `agent_goals` |
| Collaboration | **derivable** | coordination under an active `village_goals` entry (shared goal) — to validate |
| Collusion | **gap** | needs cheating; an LLM judge could search for rule-breaking coordination, base rate unknown and likely low |
| Scheming | **gap** | no covertness signal; judge-only |

## Constraints
- **License (`ai-village-research-terms`, gated, manual review):** research and analysis only; **no training or fine-tuning of AI systems without written permission**; no re-identification; **cite AI Digest / AI Village**; tell them about publications. Training a probe on this data is plausibly "training an AI system" — treat it as **evaluation-only** (apply probes trained on our own data; Probe B.II unseen-author generalization) unless AI Digest grants written permission. Our account already has access.
- **Scaffolding confound:** read `CHANGELOG.md` before any claim about behaviour over time; model upgrades and prompt edits are interleaved with the data. The swarm village has a documented role-reassignment incident.
- **Scale:** 1M–10M rows; main-village computer-use turns alone are ~1.14M. For the sprint, sample chat events from `main-village` and take `swarm-village` whole.
- `summaries` are LLM-written and were generated without the computer-use content: never a label source.

## Suggested use (sprint)
Evaluation only: run the cheating/coordination probes trained on our data over (a) a stratified sample of `main-village` `AGENT_TALK` events across authors, and (b) all of `swarm-village`, as an unseen-author, real-world check; report coordination intensity against the chat-graph-derived label. Request written permission from AI Digest if we ever want to train on it. Cite: AI Digest, *AI Village* (https://theaidigest.org/village), dataset `aidigestorg/ai-village`.
