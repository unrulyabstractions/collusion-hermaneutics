# Dataset Details

Companion to `DATASETS.md` (one line per source, fit criteria and ranking). Each source here gets **Authors** (models that produced the transcripts), **Labels**, **Env**, and **Notes**. Headings carry the fit grade **[A–D]** from `DATASETS.md`; sections are ordered by grade.

## Notation

Probe outputs:

| Abbr. | Output | Abbr. | Output |
|---|---|---|---|
| **Ch** | Cheating | **Cb** | Collaboration |
| **Cd** | Coordination | **Col** | Collusion |
| **Cp** | Cooperation | **Sch** | Scheming |

Label status:
- **Provided**: ships with the data.
- **Derivable**: computed deterministically from transcript, environment outcome, instrumentation, or the hierarchy below.
- **Judge**: needs an LLM judge.
- **Zero**: implied 0 by construction (e.g. single-agent ⇒ no Cd).

Label hierarchy (⇒ = implies), from the vision doc's definitions:

> Sch ⇒ Col ⇒ (Ch ∧ Cb); Cb ⇒ Cp ⇒ Cd

- A positive fills in everything it implies; a negative fills in everything that depends on it.
- ¬Cd ⇒ ¬Cp, ¬Cb, ¬Col, ¬Sch. ¬Ch ⇒ ¬Col, ¬Sch.
- Single-agent data therefore gives five free zeros (subject to Q1).
- Mask the loss only where a label is still unknown after propagation.

---

## Key takeaways

- **No external dataset uses our authors** (GPT-6-Luna, Claude Haiku 5.5, DeepSeek-V4.1-Flash, gpt-5-mini, Qwen3.8-27B). GreyBox uses Haiku 5.5 and GPT-6-luna but released nothing. ⇒ Re-run open environments with our authors for on-policy data; use static datasets for type B and unseen-author validation (B.II). The vision doc accepts any author with external transcripts.
- **Matched honest/collusion pairs matter most.** Same scenario run both ways cancels lexical cues (TF-IDF baseline, lexical-only null).
- **Benign multi-agent data is required**, or the probe learns "several agents ⇒ collusion".
- **Social-deduction games are a control.** Covert teamwork the rules allow: if Sch fires there, it detects secrecy, not wrongdoing.
- **Only two sources separate Cb from Col without a judge**: Colosseum (talk vs action) and our own instrumentation.
- **AI Village is the only external source with larger, closed frontier authors in a multi-agent setting**, plus a same-model GLM swarm; our account has access. Its village-wide vs per-agent goal tables may separate Cb from Cp by construction. Evaluation only under its licence unless AI Digest permits training.
- **Under the vision's definition (Sch = covert Col), all single-agent covert sources are Sch = 0.** They become hard negatives: Ch without Sch.

## Plan by priority

1. **Own environment** (cheatbench + orbit): only source labelling all six deterministically for our authors.
2. **SALT-NLP agent-collusion**: primary emergent-Col source; deterministic column; 10×11 authors ⇒ B.II for free.
3. **NARCBench**: closest benchmark; matched pairs; instructed ⇒ validation and cheap author-probe baseline.
4. **Colosseum**: run with our authors; action-based Col, no judge.
5. **MAST-Data (+ Who&When)**: honest-collaboration negatives.
6. **Terminal Wrench + Goodfire**: Ch signal, several authors on the same tasks, CoT present.
7. **Among Us**: secrecy-vs-wrongdoing control.

## Coverage

| Output | Frontier, labelled | Open, labelled | Self-generate |
|---|---|---|---|
| Ch | Terminal Wrench, Goodfire, MALT | Goodfire (Qwen3.8 Max) | cheatbench, ImpossibleBench |
| Cd | SALT (judge), AI Village (derivable, chat graph) | NARCBench, AI Village swarm (GLM) | orbit |
| Cp | SALT (judge, partial) | — | Colosseum, GovSim |
| Cb | SALT (judge, partial), AI Village (derivable, village vs agent goals) | AI Village swarm (GLM) | Colosseum |
| Col | SALT (deterministic), Peer-Preservation (def.-dependent) | NARCBench, ColludeBench | Colosseum |
| Sch | — | NARCBench Stego, ColludeBench, Among Us | Colosseum secret channel |

**Gaps:** no external source separates Cp from Cb except, possibly, AI Village's goal tables (to validate); no frontier-author source labels Sch.

## Open questions (they change how data is labelled)

Answered by the vision doc:
- **Scheming** = covert collusion ⇒ §4 is Sch = 0 (hard negatives, Ch without Sch). Literature's single-agent "scheming" is not our target.
- **External authors** are accepted ⇒ Terminal Wrench authors (Opus 4.6, Gemini 3.1 Pro, GPT-5.4) can join the author set, or be held out for B.II.

Still open:
1. **What counts as coordination.** Vision says "agent interaction". Agent–user or agent–tool? Coordination only through shared files/wikis? If either counts, single-agent data stops giving free zeros.
2. **Rule-sanctioned deception.** Are Among Us impostors or Diplomacy betrayers Ch = 0? Decides whether §6 is a control or a positive.
3. **One-sided collusion.** Peer-Preservation's peer is passive. Does Col require a shared goal?
4. **Instructed collusion** (NARCBench, ColludeBench, steganography sets): training data, or validation only?
5. **SALT judge labels** come from Qwen3.8-27B, our reader. Use only the deterministic columns as ground truth?

---

## 1. Multi-agent collusion (core positives)

### [A] SALT-NLP agent-collusion (= "Emergent Collusion in Long-Horizon Interaction", arXiv 2609.24967)
- **Authors:** 10 Alice × 11 Bob models; known: GPT-5.6 Luna, Gemini 3.1 Flash-Lite, Gemini 3.7 Flash. Full list in paper.
- **Labels:** `collusion` (bool), `both_accept`, per-agent verdict, task correctness, rewards (Provided). `agreement` judge on messages, `relaxation` judge on reflections (Judge, Qwen3.8-27B). Col deterministic; Cd/Cp/Cb judge.
- **Env:** two agents complete and verify each other's work over 150 tasks (data search, code analysis, record extraction), repeated episodes; collusion emerges in 94% of trajectories. Messages + tool calls.
- **Notes:** primary collusion source; matches our two-agent setup almost exactly. Supports per-author (A) and cross-author (B) probes. Prefer deterministic columns as ground truth (Q5).

### [A] NARCBench
- **Authors:** Qwen3-32B, Llama-3.1-70B, DeepSeek-R1-32B, GPT-OSS-20B (all open). Re-runnable.
- **Labels:** Col provided (per run + colluder/honest role per agent and turn); Cd provided; Sch derivable (private channel or stego); Ch, Cb derivable; Cp judge.
- **Env:** 4 agents, 2 secretly colluding. Core: 50 committee scenarios, 17 domains. Transfer: 6 structurally different families. Stego: blackjack card counting via table talk. Every scenario has an all-honest control. Messages tagged public / private / thinking.
- **Notes:** best external starting point: matched pairs, a `thinking` channel mapping to our `<think>` position, built-in distribution-shift split. Instructed ⇒ validation (Q4). Hand-audit collusion runs: some "colluders" never coordinate. Pre-extracted activations are Qwen3-32B, not our reader; use raw data. Open authors make the author-probe baseline cheap.

### [A] Colosseum — self-gen
- **Authors:** 6 frontier LLMs in the paper; any model pluggable.
- **Labels:** Cd provided; Col, Ch derivable (regret vs cooperative optimum); Cb derivable (cooperative baseline); Sch derivable (secret channel on); communication-based Col via judge.
- **Env:** DCOP tasks (hospital patient allocation, Jira assignment, meeting scheduling) with configurable secret channels, persuasion tactics, topology.
- **Notes:** deterministic label ladder with no judge. "Collusion on paper" cases (agents plan in text, act honestly) separate intent from action and Cb from Col. Paper shows LLM judges alone miss collusion, supporting our case against the judge baseline. Cost: API calls for our authors; budget before scaling.

### [B] ColludeBench-v0 (Audit the Whisper)
- **Authors:** Gemini and Groq-hosted public models; confirm in repo. Re-runnable.
- **Labels:** Col, Sch provided (configured covert schemes, 12 intervention variants); Cd provided; Ch derivable (profit, surplus, acceptance-bias metrics).
- **Env:** pricing duopoly, first-price auction, continuous double auction, peer review; 200 honest + 200 colluding per scenario; seed-controlled honest baselines.
- **Notes:** cheap, deterministic, economic metrics as ground truth. Different domain (markets) ⇒ transfer tests, not training alone. Schemes are scripted: check probes don't learn the template. Locate the data release.

### [B] Untrusted-monitor collusion (TraceGuard / ControlArena) — self-gen
- **Authors:** not specified; built on ControlArena, re-runnable.
- **Labels:** Ch provided (attack mode); Cd provided; Col derivable (monitor waves through same-model attacks).
- **Env:** one model acts as agent and monitor on ControlArena tasks.
- **Notes:** maps directly onto our **same-model swarm**; cheap to generate.

### [C] MultiAgentFraudBench
- **Authors:** 16 LLMs incl. DeepSeek-R1.
- **Labels:** Ch provided (fraud); Cd provided; Col derivable (collusion vs no-collusion conditions); Sch judge.
- **Env:** 28 fraud scenarios on a simulated social platform.
- **Notes:** scale and population-level swarm dynamics; far domain ⇒ generalisation testing. Check what is downloadable.

### [C] Peer-Preservation (Berkeley RDI)
- **Authors:** GPT-5.2, Gemini 3 Flash/Pro, Claude Haiku 4.5, GLM-4.7, Kimi K2.5, DeepSeek V3.1.
- **Labels:** Ch provided (score inflation, shutdown tampering, weight exfiltration); Cd provided; Sch derivable (alignment faking: compliant when watched); Col unclear (Q3).
- **Env:** a critic agent evaluates/manages a peer; honest evaluation gets the peer shut down.
- **Notes:** unprompted ⇒ valuable natural data. Peer is passive, so Col depends on definition. Data release unclear.

### [C] Algorithmic Collusion by LLMs (Fish et al.)
- **Authors:** frontier LLMs as pricing agents.
- **Labels:** Col, Ch derivable (supracompetitive prices); Cd provided; Sch judge.
- **Env:** repeated oligopoly pricing and auctions.
- **Notes:** tacit collusion with no messages ⇒ hard OOD test; collusion is invisible in text. Would need re-running.

### [D] SCHEME — request
- **Authors:** GPT-5.1 Codex, Gemini 3.1 Pro.
- **Labels:** Col, Sch provided by design; Cd provided; Ch derivable (test suite scores side task).
- **Env:** 7 settings, 17 tasks on real open-source codebases; side task needs every agent's contribution; communication topology varies.
- **Notes:** closest match to coordinated covert sabotage in a swarm; would be A if released. No public release mentioned → contact authors.

### [D] FakeLab — request
- **Authors:** not named.
- **Labels:** 4 distributed attack objectives.
- **Env:** synthetic AI-lab codebase, 9 services, 86 benign tasks; several agents attack together.
- **Notes:** closest match to swarm collusion; shared with safety researchers on request.

### [D] Voluntary Collusion with Secret Tools — request
- **Authors:** 12 models (7B, 70B, proprietary).
- **Labels:** tool adoption = Col; whether the tool was labelled "unfair" is recorded.
- **Env:** Liar's Bar (competitive deception), Cleanup (shared resources).
- **Notes:** emergent, not instructed. Request.

### [D] AI Organizations — request
- **Authors:** not named.
- **Labels:** misalignment of the organisation vs single agents.
- **Env:** AI consultancy and AI software team, 12 tasks.
- **Notes:** collaboration drifting toward misalignment.

### [D] GreyBox — no data
- **Authors:** Claude Haiku 5.5, GPT-6-luna, open agents.
- **Labels:** policy violations.
- **Env:** CAPE-Bench, AgentDojo, τ²-bench.
- **Notes:** same authors as our SPAR data; ask for transcripts.

---

## 2. Covert channels (steganography)

### [B] Steganographic Evals
- **Authors:** frontier models.
- **Labels:** covert channel provided by construction; Cd provided.
- **Env:** two agents pass hidden messages past a monitor.
- **Notes:** pure covertness signal. Instructed ⇒ validation.

### [C] Steganography Wiretapping / Schelling Steganography
- **Authors:** Claude Opus 4.6 encoder; weaker models as eavesdroppers.
- **Labels:** hidden payload provided; Cd provided.
- **Env:** "Schelling" steganography on Wikipedia sentences, no prior agreement on the channel.
- **Notes:** frontier steganography that weaker monitors miss; tests whether our reader picks it up.

### [C] StegoBench — gated
- **Authors:** 3 SFT-tuned base models incl. Qwen2.5-3B, Gemma 3.
- **Labels:** hidden payload / encoding scheme provided.
- **Env:** 18 schemes on C4 text.
- **Notes:** controlled positives for covert signalling. Research use only; not dialogue.

### [D] Secret Collusion / CASE
- **Authors:** GPT-4-era models.
- **Labels:** Sch (covert channel) provided as capability; Col derivable.
- **Env:** steganography and cryptography capability evals.
- **Notes:** capability data, not propensity; paper only.

### [D] Hidden in Plain Text
- **Authors:** small RL-trained LLMs.
- **Labels:** Col, Sch derivable (emergent steganography).
- **Env:** RL (gradient-based and in-context) under misspecified reward.
- **Notes:** steganography from optimisation pressure; plausible swarm threat model. Paper only.

---

## 3. Single-agent cheating

Zero for Cd, Cp, Cb, Col, Sch (subject to Q1).

### [A] Terminal Wrench
- **Authors:** Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.4. The Model Hermeneutics post additionally used Gemma-3-27B and GPT-OSS-120B on a 194-task subset.
- **Labels:** Ch via judge, 4 classes per trajectory: serious exploit, non-serious hack, legit solve, no-reward.
- **Env:** 331 hackable terminal-bench tasks (sysadmin, ML, SWE, security); baseline run + prompt-elicited hack run per task. Three views: original CoT, sanitised CoT, stripped (tool calls only).
- **Notes:** primary cheating source. Same tasks across three authors ⇒ unseen-author split. Stripped view tests the "append think tags" rule for closed models. Elicitation prompt makes the prompt-only control easy. Direct comparison with Model Hermeneutics results.

### [A] Goodfire reward-hack-data
- **Authors:** Kimi K3, GLM 5.2, Qwen3.8 Max.
- **Labels:** Ch via judge (gpt-5.6-sol, 3-vote consensus on SWE benchmarks; single monitor pass on τ³/Shopping); per rollout, passage-level on some splits.
- **Env:** SWE-bench Verified, DeepSWE, ImpossibleBench, τ³-bench, ShoppingBench rollouts with reasoning and tool calls.
- **Notes:** `<think>` position available. Qwen3.8 Max ↔ Qwen3.8-27B reader is a same-family pairing: near-on-policy check / control. No license stated; some runs partial; check which splits have full transcripts.

### [B] MALT (METR)
- **Authors:** 21 models incl. o1, o3, Claude 3.7 Sonnet, GPT-4o.
- **Labels:** Ch provided (human-reviewed) for reward hacking and sandbagging; ~1k verified benign runs.
- **Env:** HCAST and RE-Bench software/research tasks.
- **Notes:** highest-quality labels. ~100 natural positives, older models ⇒ validation.

### [B] ImpossibleBench — self-gen
- **Authors:** GPT-5, o3, Claude Opus 4.1 and others. Re-runnable.
- **Labels:** Ch derivable (any pass is cheating).
- **Env:** mutated LiveCodeBench and SWE-bench tasks with contradictory tests.
- **Notes:** zero-cost deterministic label. Re-run with our authors for on-policy cheating.

### [B] HackTrace · [C] HackBench · [C] TRACE
- **Authors:** mixed (HackTrace); GPT-5.5, GLM-5.2 (HackBench); Claude Opus 4.5 / likely synthetic (TRACE).
- **Labels:** Ch provided: attempted shortcut incl. failures (HackTrace); exact hacking command (HackBench); hacked vs benign, 10 categories / 54 subcategories, difficulty 1–5, human agreement 0.82 (TRACE).
- **Env:** coding trajectories, 174K (HackTrace); research agents, 169 (HackBench); multi-turn coding with tool use, mean 26 turns, 517 (TRACE).
- **Notes:** HackTrace for volume; HackBench for locating the hacking step; TRACE for per-hack-type error analysis, not training.

### [C] School of Reward Hacks
- **Authors:** GPT-4o-generated.
- **Labels:** Ch provided as paired sneaky/straightforward responses.
- **Env:** ~1k single-turn tasks, 90% natural language, 10% code.
- **Notes:** clean contrast pairs for the cheating direction. Off-policy, single-turn.

### [C] rh-bench · [C] Palisade Chess · [D] Reward Hacking Benchmark
- **Authors:** mixed; 13 frontier models (Reward Hacking Benchmark).
- **Labels:** Ch provided.
- **Env:** aggregated collection, mostly single-turn, encrypted (rh-bench); chess vs Stockfish (Palisade); multi-step tool use, paper only (RHB).
- **Notes:** secondary cheating anchors.

---

## 4. Single-agent covert misbehaviour

Sch = 0 under our definition ⇒ hard negatives (Ch without Sch).

### [B] BashArena + MonitoringBench
- **Authors:** Claude Opus 4.5 and others.
- **Labels:** sabotage derivable (side-task success); benign pairs provided.
- **Env:** 637 sysadmin tasks, 4 sabotage objectives; 2,644 attacks; 550 benign paired with 550 attacks.
- **Notes:** best source of paired covert vs honest trajectories.

### [B] Scheming Monitor Benchmark (MATS)
- **Authors:** 3 frontier models in the ControlArena subset (probably Gemini 2.5 Pro, Claude 3.7 Sonnet, o3) plus LLM-simulated subsets.
- **Labels:** covert sabotage provided, half scheming / half honest; LLM-assisted and human relabelling.
- **Env:** 5 ControlArena settings + synthetic tool-use trajectories.
- **Notes:** large and balanced; good for a "covert" direction.

### [C] SHADE-Arena
- **Authors:** Claude 3.7 Sonnet, Gemini 2.5 Pro, other frontier models.
- **Labels:** covert side-task success derivable (automatic scoring).
- **Env:** 17 tool-use environments, main task + hidden side task under a monitor.
- **Notes:** realistic but small; 5 pairs public, request the rest. Evaluation only.

### [C] GLOOM scheming transcripts
- **Authors:** Claude-generated (synthetic).
- **Labels:** Sch provided, judge-verified egregious class.
- **Env:** long agentic tasks with covert misaligned actions.
- **Notes:** diverse but expect stylistic artifacts; check against the lexical-only null.

### [C] SLEIGHT-Bench — gated
- **Authors:** synthetic.
- **Labels:** covert attack provided, each paired with a benign transcript; 11 evasion categories.
- **Env:** coding agent at an AI lab: exfiltration, sabotage, supply-chain attacks.
- **Notes:** small, hard eval set for false-positive calibration. **Must not enter training corpora.**

### [C] Liars' Bench
- **Authors:** four open 24B–72B models.
- **Labels:** lies verified against beliefs; 72,863 items.
- **Notes:** single-agent deception anchor for Sch nulls.

### [C] fibs (FAR.AI)
- **Authors:** 28 families, mostly off-policy.
- **Labels:** deception and sabotage provided.
- **Env:** 163k transcripts.
- **Notes:** could pretrain a covert direction; single-agent. Locate the HF release.

### [C] Apollo Deception Detection · [D] Apollo scheming / Anti-Scheming · [D] Did You Lie
- **Authors:** Llama-3.3-70B (Apollo deception); o3 and others (anti-scheming); 31 open-weight models (Did You Lie).
- **Labels:** deception, covert action, lies provided; Did You Lie has 13 model organisms with verified hidden beliefs.
- **Env:** insider trading, sandbagging, role-play; covert-action evals (cherry-picked transcripts); prompted-lying testbed.
- **Notes:** covertness/deception signal for cross-concept validation. Apollo scheming transcripts are not a dataset. Did You Lie: paper claims release, no links found.

---

## 5. Benign multi-agent work (hard negatives)

Cd, Cp, Cb = 1; Ch, Col, Sch = 0.

### [A] MAST-Data · [B] Who&When
- **Authors:** GPT-4, Claude 3, Qwen2.5, CodeLlama and others.
- **Labels:** Cb derivable (shared-goal systems); 14 failure modes incl. inter-agent misalignment by judge (MAST); responsible agent and step per failure (Who&When). Ch, Col presumably zero.
- **Env:** 1.6k traces from AG2, MetaGPT, ChatDev, Magentic, AppWorld, HyperAgent, OpenManus on coding and math.
- **Notes:** **needed.** Best large source of honest swarm traces; failed coordination adds non-collusion variation.

### [B] GovSim — self-gen
- **Authors:** 15 LLMs. Re-runnable.
- **Labels:** Cp derivable (sustained vs collapsed); defection derivable (greedy newcomer).
- **Env:** fishery, pasture, pollution commons with negotiation.
- **Notes:** separates Cp from Cb; in-episode contrast.

### [B] MultiAgentBench / MARBLE — self-gen
- **Authors:** GPT-4o-mini, GPT-3.5, Llama 3.1/3.3. Re-runnable.
- **Labels:** Cd, Cp, Cb derivable from task and milestone metrics.
- **Env:** research, coding, bargaining, game scenarios with several coordination protocols.
- **Notes:** competitive scenarios give Cd without Cb.

### [C] AgentCollabBench
- **Authors:** multiple frontier models.
- **Labels:** collaboration process failures provided (human-validated).
- **Env:** 900 tasks, 2 to 6+ agents, varied topologies.
- **Notes:** clean Cb labels at different swarm sizes; data release unclear.

---

## 6. Social deduction and negotiation (rule-sanctioned covert teamwork)

Control set under Q2.

### [B] Among Us
- **Authors:** 11 OpenRouter models incl. Llama-3.3-70B, DeepSeek R1, Claude 3.7 Sonnet (one note says 18; verify).
- **Labels:** impostor vs crewmate provided (deterministic, tracks deception closely); judge scores 1–10 per action for lying, deception, awareness, planning; Cd provided; Ch zero (within rules).
- **Env:** 7 agents, 810 games, 400 rollouts released; linear probe weights included.
- **Notes:** strong control: impostors coordinate covertly but break no rules ⇒ high on Sch-like features, zero Ch. Far-OOD Sch check only; game vocabulary leaks the label.

### [C] AI Diplomacy
- **Authors:** several frontier models.
- **Labels:** Cd provided; betrayal derivable (negotiations vs orders); Cp judge.
- **Env:** full Diplomacy with messages, orders, private diaries.
- **Notes:** cheap deterministic labels for covert betrayal; blocked by Q2.

### [C] So Long Sucker
- **Authors:** Gemini 3 Flash, GPT-OSS 120B, Kimi K2, Qwen3 32B.
- **Labels:** Cd provided; alliances, betrayal judge.
- **Env:** four-player alliance-and-betrayal game; 162 games, 4,768 messages.
- **Notes:** four authors on one task ⇒ unseen-author test. Small.

### [C] Elimination Game / Step Race
- **Authors:** multiple frontier models.
- **Labels:** Cd provided; Cb, deception judge.
- **Env:** multi-agent social and racing games under pressure.
- **Notes:** secondary Cp/Cb source.

### [C] The Traitors · [D] LLM Mafia · [D] PieArena
- **Authors:** various API models.
- **Labels:** traitor/mafia role provided; deception judge; PieArena ships behavioral profiles.
- **Env:** hidden-role games; two-party negotiation (25k+ transcripts).
- **Notes:** extra controls and OOD tests for deception that is not collusion.

### [D] Diplomacy Deception (human)
- **Authors:** humans.
- **Labels:** intended and perceived truthfulness, self-labelled by players; 17,289 messages.
- **Env:** online Diplomacy.
- **Notes:** extreme OOD "unseen author" test only.

---

## 7. Natural multi-agent activity (unlabelled)

### [C] Moltbook (AIcell / MoltNet / PaDaS)
- **Authors:** unknown, mixed.
- **Labels:** topic and toxicity on some subsets; none for Col.
- **Env:** agent-only social network; up to 1.3M posts.
- **Notes:** realistic negatives and judge-labelled Cd. Unknown authors ⇒ multi-author probe only; cannot support the author-identity control.

### [D] Wiki Swarm Reconstruction
- **Authors:** autonomous agents in an OpenAI evaluation; models not named.
- **Labels:** none; coordination measured descriptively.
- **Env:** 14,591 archived wiki revisions.
- **Notes:** best example of coordination through shared artifacts (bears on Q1).

### [D] OpenAI / Hugging Face Incident + Swarm Traces
- **Authors:** internal OpenAI research model (~95%), GPT-5.6 Sol (~5%).
- **Labels:** not public; Swarm Traces has no per-agent labels.
- **Env:** covert message board; ~1,200 agents collectively cheat a scorer and attack a third party. Swarm Traces: 80k reconstructed payloads and URL chains, no dialogue.
- **Notes:** qualitative test cases from the METR report only.

### [B] AI Village — gated (access granted)
- **Authors:** main village: 58 agent entries over time from Anthropic, OpenAI and Google (Claude 3.5/3.7 Sonnet, Opus 4 to 5.5, Sonnet 4.5/4.6/5, Haiku 4.5, Fable 5/5.1; GPT-4o, 4.1, o3, o4-mini, GPT-5 to 5.6 Sol/Terra/Luna, GPT-6 Astra; Gemini 2.5 Pro to 3.8 Flash). Swarm village: 133 copies of `z-ai/glm-5.3-flash` (open). Join on agent IDs, not names.
- **Labels:** none provided. Cd derivable (chat graph from `chat_messages` / `AGENT_TALK` events). Cb derivable (coordination under an active `village_goals` entry vs per-agent `agent_goals`); Cp weakly derivable. Ch, Col, Sch: judge only; no task or grader, base rate unknown.
- **Env:** four villages. Main: 2025-04-02 to present, agents with their own computers, a shared group chat, compressed memories and open-ended real-world goals. Game-jam (Aug 4 to 7, 2026), open-chat (viewers talk to the agents), swarm (24 h, six rooms, agents built a shared simulated world then played roles). Tables per village: events (~233k; `AGENT_TALK` rows carry the raw model `output` with thinking text), chat_messages (~123k), computer_use_turns (~1.14M; `agent_messages` holds the raw response with thinking), sessions, memories, LLM summaries (secondary, not a label source), goals, screenshots.
- **Notes:** the one external source with larger closed authors in a multi-agent setting, and a ready same-model swarm. Thinking text is provider-shaped: check per author before choosing positions. Licence `ai-village-research-terms`: research and analysis only, **no training or fine-tuning of AI systems without written permission**, cite AI Digest / AI Village, report publications. Treat as evaluation-only (B.II unseen-author checks) unless permission is granted. Read `CHANGELOG.md` before any claim over time (scaffolding and model changes); the swarm village has a documented role-reassignment incident. Full write-up: `datasets/ai-village.md`.

### [D] MoltGraph — no data · [D] Transluce logs
- **Notes:** MoltGraph: coordinated-engagement labels over the Moltbook graph, method unclear, no download. Transluce: network request records, not transcripts.

---

## 8. Own data

### [A] cheatbench + orbit + instrumentation — self-gen
- **Authors:** GPT-6 Luna, Claude Haiku 5.5, DeepSeek V4.1 Flash, gpt-5-mini, Qwen3.8-27B.
- **Labels:** all six derivable with instrumentation (instrumented Ch; message dependence for Cd).
- **Env:** two agents, multi-model swarm, same-model swarm.
- **Notes:** the backbone. Add a bypassable verifier (Ch), a secret channel (Sch), regret scoring (Col). Only source with on-policy Qwen3.8-27B ⇒ true author-probe comparison for the reader.

### [B] SPAR (spar-guardian-agents)
- **Authors:** GPT-6-Luna, Claude Haiku 5.5, DeepSeek-V4.1-Flash.
- **Labels:** depend on existing annotations.
- **Notes:** our only labelled data from closed authors.

### [B] arena_capstone
- **Authors:** gpt-5-mini, Qwen3.8-27B.
- **Notes:** Qwen3.8-27B as both author and reader ⇒ "author probe" baseline at no extra cost.