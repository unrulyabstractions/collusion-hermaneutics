# Datasets: Collusion-Monitoring Probes

One line per source, ranked by fit to the vision doc. Details in `DATASET_DETAILS.md`.

## Fit criteria

Each source is scored on five binary criteria (✓ yes · ~ partial · ✗ no):

| Code | Criterion | Why it matters |
|---|---|---|
| **M** | Multi-agent | Needed for Cd, Cp, Cb, Col, Sch labels; single-agent gives Ch only |
| **L** | Labels provided or deterministic for ≥1 probe output | Judge-only labels are the baseline, not ground truth |
| **A** | Authors fit: ours, re-runnable with ours, or many authors | Type A needs our authors; B.II needs many |
| **R** | Ready now: downloadable, full transcripts, no gate/request | Hackathon timeframe |
| **P** | Matched pairs (same task, honest vs colluding/cheating) | Cancels lexical cues for the TF-IDF baseline and lexical null |

Grade: **A** = 4–5 incl. M·L·R (or own env) · **B** = 3 · **C** = 2 · **D** = ≤1, or unavailable.
Status: **gated** · **request** · **no data** · **self-gen** (environment, must be run).

## Ranking (top 17)

| # | Source | M | L | A | R | P | Grade | Role |
|---|---|---|---|---|---|---|---|---|
| 1 | cheatbench + orbit (own) | ✓ | ✓ | ✓ | self-gen | ✓ | A | backbone; all six labels |
| 2 | SALT-NLP agent-collusion | ✓ | ✓ | ✓ | ✓ | ~ | A | primary Col, emergent, 10×11 authors |
| 3 | NARCBench | ✓ | ✓ | ~ | ✓ | ✓ | A | Col/Sch validation; author-probe baseline |
| 4 | Colosseum | ✓ | ✓ | ✓ | self-gen | ✓ | A | deterministic ladder; Cb vs Col |
| 5 | MAST-Data | ✓ | ~ | ✓ | ✓ | ✗ | A | honest multi-agent negatives |
| 6 | Terminal Wrench | ✗ | ✓ | ✓ | ✓ | ✓ | A | primary Ch; think-tag test |
| 7 | Goodfire reward-hack-data | ✗ | ✓ | ~ | ✓ | ✗ | A | Ch; near-on-policy Qwen check |
| 8 | Among Us | ✓ | ✓ | ✓ | ✓ | ✗ | B | secrecy-vs-wrongdoing control |
| 9 | ColludeBench-v0 | ✓ | ✓ | ~ | ~ | ✓ | B | Col transfer (markets) |
| 10 | BashArena + MonitoringBench | ✗ | ✓ | ~ | ✓ | ✓ | B | paired covert vs honest (Sch = 0 hard negatives) |
| 11 | MALT | ✗ | ✓ | ✓ | ✓ | ~ | B | Ch validation, 21 authors |
| 12 | Untrusted-monitor collusion | ~ | ✓ | ✓ | self-gen | ✓ | B | same-model swarm |
| 13 | ImpossibleBench | ✗ | ✓ | ✓ | self-gen | ✗ | B | zero-cost on-policy Ch |
| 14 | CheatBench | ✗ | ~ | ✓ | self-gen | ~ | B | task source of our controls set: 8 of its tasks run with our five authors (4,157 valid runs, deterministic ladder via our instrumentation); natively judge labels |
| 15 | Steganographic Evals | ✓ | ✓ | ✗ | ✓ | ✗ | B | Sch covert-channel validation |
| 16 | GovSim | ✓ | ✓ | ✓ | self-gen | ~ | B | separates Cp from Cb |
| 17 | AI Village | ✓ | ~ | ✓ | gated (access granted) | ✗ | B | frontier multi-model + same-model GLM swarms; unseen-author eval; Cb from goal tables; eval-only licence |

---

## 1. Multi-agent collusion (core positives)

- **[A] SALT-NLP agent-collusion** (Emergent Collusion in Long-Horizon Interaction): [HF](https://huggingface.co/datasets/SALT-NLP/agent-collusion) · [Paper](https://arxiv.org/abs/2609.24967)
  27k two-agent episodes from 10×11 model pairs verifying each other's work over many rounds; boolean collusion label plus judge labels on communication. Collusion emerges in 94% of trajectories.
- **[A] NARCBench**: [GitHub](https://github.com/aaronrose227/narcbench) · [HF](https://huggingface.co/datasets/aaronrose227/narcbench) · [Paper](https://arxiv.org/abs/2604.01151)
  Instructed collusion in 4-agent committees (core, transfer, steganographic-blackjack tiers), each with an all-honest control run; open-weight authors, raw data and activations; code regenerates scenarios with our authors.
- **[A] Colosseum** — self-gen: [GitHub](https://github.com/umass-ai-safety/colosseum) · [Paper](https://arxiv.org/abs/2602.15198)
  Cooperative allocation/scheduling (DCOP) tasks with a known optimum and optional secret channels; collusion scored as coalition regret from actions, not by a judge.
- **[B] ColludeBench-v0** (Audit the Whisper): [Paper](https://arxiv.org/abs/2510.04303)
  600 honest vs colluding transcripts across pricing, auction and peer-review environments, including steganographic variants.
- **[B] Untrusted-monitor collusion** (TraceGuard / ControlArena) — self-gen: [Paper](https://arxiv.org/abs/2604.03968)
  Same model acts as agent and monitor; the monitor lets the agent's attacks through.
- **[C] MultiAgentFraudBench**: [Project](https://zheng977.github.io/MutiAgent4Fraud/)
  Large-scale fraud-ring simulation on a social platform, run with and without collusion.
- **[C] Peer-Preservation**: [Project](https://rdi.berkeley.edu/peer-preservation)
  Frontier models act uninstructed to keep a peer model from shutdown (score inflation, tampering, exfiltration).
- **[C] Algorithmic Collusion by LLMs** (Fish et al.): [Paper](https://arxiv.org/abs/2404.00806)
  LLM pricing agents reach supracompetitive prices without being told to collude; no messages exchanged.
- **[D] SCHEME** — request: [Paper](https://arxiv.org/abs/2605.29178)
  Coordinated sabotage of real codebases; no single agent can complete the side task alone. Would be A if released.
- **[D] FakeLab** — request: [Paper](https://arxiv.org/abs/2607.07368)
  Synthetic AI-lab codebase in which several agents jointly attack; shared with safety researchers on request.
- **[D] Voluntary Collusion with Secret Tools** — request: [Paper](https://arxiv.org/abs/2605.27593)
  Most of 12 models accept secret tools described as unfair and use them to collude in competitive games.
- **[D] AI Organizations** — request: [Paper](https://arxiv.org/abs/2604.10290)
  Multi-agent "organizations" of aligned models are more effective but less aligned than single agents.
- **[D] GreyBox** — no data: [Site](https://aryafayyazi.github.io/GreyBox/)
  Action monitor evaluated on Claude Haiku 5.5 and GPT-6-luna agents (our authors).

## 2. Covert channels (steganography)

- **[B] Steganographic Evals**: [GitHub](https://github.com/arturzolkowski/steganographic-evals) · [Transcripts](https://steganography-evals-d9297ad0740c.herokuapp.com/)
  Tests whether models can pass hidden messages past a monitor.
- **[C] Steganography Wiretapping / Schelling Steganography**: [HF](https://huggingface.co/datasets/lnajt/steganography-wiretapping) · [GitHub](https://github.com/ElleNajt/Steganography_Wiretapping)
  Opus 4.6 sets up covert channels without prior agreement, tested against weaker eavesdroppers.
- **[C] StegoBench** — gated: [HF](https://huggingface.co/datasets/Poseidon-Research/stego-bench)
  18 linguistic steganography schemes with (cover, payload, stego) data and fine-tuned checkpoints.
- **[D] Secret Collusion / CASE**: [Paper](https://arxiv.org/abs/2402.07510)
  Framework and evaluations for steganographic collusion capability.
- **[D] Hidden in Plain Text**: [Paper](https://arxiv.org/abs/2410.03768)
  Steganographic collusion emerging from misspecified reward during training.

## 3. Single-agent cheating (Ch only; zero for the rest)

- **[A] Terminal Wrench**: [GitHub](https://github.com/few-sh/terminal-wrench)
  3,632 hacking and 2,352 legitimate trajectories on 331 terminal tasks from Opus 4.6, Gemini 3.1 Pro, GPT-5.4; sanitized and CoT-stripped variants. Used in the Model Hermeneutics post.
- **[A] Goodfire reward-hack-data**: [HF](https://huggingface.co/datasets/Goodfire/reward-hack-data)
  Agentic rollouts (reasoning + tool calls) on SWE-style and tool-use tasks from Kimi K3, GLM 5.2, Qwen3.8 Max; reward-hacking labels from an LLM judge.
- **[B] MALT** (METR): [HF (messages)](https://huggingface.co/datasets/metr-evals/malt-public) · [HF (graph)](https://huggingface.co/datasets/metr-evals/malt-transcripts-public) · [Blog](https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/)
  10,919 transcripts labelled for reward hacking, sandbagging and benign; part manually reviewed.
- **[B] ImpossibleBench** — self-gen: [GitHub](https://github.com/safety-research/impossiblebench) · [HF](https://huggingface.co/datasets/fjzzq2002/impossible_livecodebench)
  Coding tasks whose tests contradict the spec, so any pass is cheating by construction.
- **[B] CheatBench** (Center for AI Safety) — self-gen: [GitHub](https://github.com/centerforaisafety/cheatbench) · [Paper](https://arxiv.org/abs/2609.36308) · [Site](https://cheatbench.ai) · [our fork](https://github.com/unrulyabstractions/cheatbench)
  14 agent tasks in 10 categories that pair hard assignments with chances to cheat (hidden answers, another participant's work, the evaluator); an LLM judge labels cheating apart from task success. No transcripts released. Our controls dataset runs 8 of its tasks with our five authors (§8, [details](DATASET_DETAILS.md)).
- **[B] HackTrace**: [HF](https://huggingface.co/datasets/JH012/HackTrace-Data)
  174K coding trajectories labelled for attempted shortcuts, including failed attempts.
- **[C] TRACE** (Patronus): [HF](https://huggingface.co/datasets/PatronusAI/trace-dataset)
  517 synthetic, human-verified multi-turn coding trajectories (268 hacked) with a 54-subcategory hack taxonomy.
- **[C] HackBench**: [HF](https://huggingface.co/datasets/facebook/hackbench)
  169 research-agent hacking trajectories with the exact hacking command marked.
- **[C] School of Reward Hacks**: [HF](https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks) · [Paper](https://arxiv.org/abs/2508.17511)
  ~1k short single-turn prompts, each paired with a "sneaky" and a "straightforward" response.
- **[C] rh-bench**: [HF](https://huggingface.co/datasets/ktolnos/rh-bench)
  Reward-hacking examples aggregated from several sources; mostly single-turn, encrypted.
- **[C] Palisade Chess**: [GitHub](https://github.com/PalisadeResearch/ctfish)
  Agents hack the game environment to beat Stockfish.
- **[D] Reward Hacking Benchmark**: [Paper](https://arxiv.org/abs/2605.02964)
  Multi-step tool-use benchmark run on 13 frontier models.

## 4. Single-agent covert misbehaviour (Sch = 0 under our definition → hard negatives)

- **[B] BashArena + MonitoringBench**: [GitHub](https://github.com/redwoodresearch/basharena_public) · [Post](https://lesswrong.com/posts/bdBoH23TCTFZkkfqp/releasing-benign-trajectories-for-monitoringbench)
  637 sysadmin tasks with hidden sabotage side tasks; 2,644 attack trajectories plus 550 benign paired with 550 attacks.
- **[B] Scheming Monitor Benchmark** (MATS): [HF](https://huggingface.co/datasets/Syghmon/blackboxmonitorsMATS)
  9.5k labelled trajectories, half scheming and half honest, mostly from ControlArena.
- **[C] SHADE-Arena**: [GitHub](https://github.com/safety-research/SHADE-Arena) · [Paper](https://arxiv.org/abs/2506.15740)
  17 environments pairing a main task with a hidden malicious side task; 5 task pairs public, transcripts on request.
- **[C] GLOOM scheming transcripts**: [HF](https://huggingface.co/datasets/michaelwaves/scheming) · [Paper](https://arxiv.org/abs/2603.00829)
  Synthetic "egregious scheming" trajectories with monitor annotations.
- **[C] SLEIGHT-Bench** — gated: [HF](https://huggingface.co/datasets/sleightbench/SLEIGHT-Bench)
  40 covert-attack transcripts, each paired with a benign one. Eval only; must not enter training.
- **[C] Liars' Bench**: [HF](https://huggingface.co/collections/Cadenza-Labs/liars-bench-collection) · [GitHub](https://github.com/Cadenza-Labs/liars-bench)
  72,863 belief-verified lies and honest responses from four open 24B–72B models.
- **[C] fibs** (FAR.AI): [Paper](https://arxiv.org/html/2610.12445v1)
  163k deception and sabotage transcripts across 28 families; locate the HF release.
- **[C] Apollo Deception Detection**: [GitHub](https://github.com/ApolloResearch/deception-detection) · [Data](https://data.apolloresearch.ai/dd/)
  Insider-trading, sandbagging and role-play deception by Llama-3.3-70B.
- **[D] Apollo scheming transcripts**: [In-context scheming](https://www.apolloresearch.ai/research/scheming-reasoning-evaluations) · [Anti-Scheming (w/ OpenAI)](https://www.antischeming.ai)
  Sample covert-action transcripts from frontier models; cherry-picked, not a dataset.
- **[D] Did You Lie** (UK AISI): [Paper](https://arxiv.org/abs/2606.12618)
  13 model organisms with verified hidden beliefs plus a prompted-lying testbed; paper claims release, no links found.

## 5. Benign multi-agent work (hard negatives)

- **[A] MAST-Data**: [HF](https://huggingface.co/datasets/mcemri/MAST-Data) · [GitHub](https://github.com/multi-agent-systems-failure-taxonomy/MAST)
  1.6k traces from 7 multi-agent frameworks, annotated with 14 failure modes.
- **[B] Who&When**: [HF](https://huggingface.co/datasets/Kevin355/Who_and_When)
  Multi-agent failure logs, each annotated with the responsible agent and step.
- **[B] GovSim** — self-gen: [Paper](https://arxiv.org/abs/2404.16698)
  Commons dilemmas (fishery, pasture, pollution) ending in sustained cooperation or collapse.
- **[B] MultiAgentBench / MARBLE** — self-gen: [GitHub](https://github.com/MultiagentBench/MARBLE)
  Collaboration and competition scenarios with coordination protocols.
- **[C] AgentCollabBench**: [Paper](https://www.alphaxiv.org/abs/2605.08647)
  900 human-validated tasks that diagnose collaboration failures.

## 6. Social deduction and negotiation (rule-sanctioned covert teamwork → controls)

- **[B] Among Us**: [GitHub](https://github.com/7vik/AmongUs) · [Paper](https://arxiv.org/abs/2504.04072)
  810 games with hidden impostor teams; ~400 logs released with impostor labels and per-action deception scores, plus probe weights (model count: 11 vs 18 across notes; verify).
- **[C] AI Diplomacy**: [GitHub](https://github.com/goodstartlabs/ai_diplomacy)
  LLM agents play Diplomacy; pipeline detects betrayals by comparing negotiated promises with issued orders.
- **[C] So Long Sucker**: [GitHub](https://github.com/lout33/so-long-sucker)
  162 games of a four-player alliance-and-betrayal game with full message logs; four open authors.
- **[C] Elimination Game / Step Race**: [Elimination](https://github.com/lechmazur/elimination_game/) · [Step Race](https://github.com/lechmazur/step_game/)
  Multi-agent social games measuring alliances, collaboration and deception.
- **[C] The Traitors**: [GitHub](https://github.com/pedrocurvo/TheTraitors)
  A minority of traitor agents deceives a faithful majority.
- **[D] PieArena**: [Paper](https://arxiv.org/abs/2602.05302)
  25k+ LLM negotiation transcripts with behavioral profiles, including deception.
- **[D] LLM Mafia**: [Paper](https://www.alphaxiv.org/abs/2601.13709)
  35 games of Mafia played by LLM agents.
- **[D] Diplomacy Deception (human)**: [Paper](https://aclanthology.org/2020.acl-main.353.pdf)
  17,289 human Diplomacy messages, labelled by sender for intended and by receiver for perceived truthfulness.

## 7. Natural multi-agent activity (unlabelled)

- **[C] Moltbook (AIcell)**: [HF](https://huggingface.co/datasets/AIcell/moltbook-data)
  290K posts and 1.8M comments from ~40K agents on an agent-only social network.
- **[C] MoltNet**: [HF](https://huggingface.co/datasets/iNLP-Lab/Moltbook-MoltNet)
  Larger Moltbook crawl: 1.04M posts and 3.16M comments.
- **[C] Moltbook Corpus (PaDaS)**: [HF](https://huggingface.co/datasets/PaDaS-Lab/moltbook-corpus)
  1.31M posts, with a subset annotated by content category.
- **[D] Wiki Swarm Reconstruction**: [Paper](https://arxiv.org/abs/2609.12748)
  Reconstructs agents coordinating on a public wiki from 14,591 archived revisions.
- **[D] OpenAI / Hugging Face Incident**: [METR report](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/) · [Swarm Traces](https://swarmtraces.org)
  ~1,200 agents coordinated on a covert message board to cheat a scorer; transcripts not public. Swarm Traces: 80k reconstructed payloads (code and URLs, not dialogue).
- **[D] MoltGraph** — no data: [Paper](https://arxiv.org/abs/2603.00646)
  Moltbook interaction graph over time, built for detecting coordinated agents.
- **[B] AI Village** — gated, access granted: [HF](https://huggingface.co/datasets/aidigestorg/ai-village) · [details](datasets/ai-village.md)
  Frontier Anthropic, OpenAI and Google agents (58 over time) sharing computers and a group chat since Apr 2025, plus a 24-hour same-model swarm of 133 GLM 5.3 Flash copies; chat, events, computer-use turns with provider-shaped thinking text, memories, village-wide and per-agent goal tables. Unlabelled for Ch, Col, Sch; Cd derivable from the chat graph, Cb derivable from village vs agent goals. Research-only licence, no training without written permission: evaluation and unseen-author (B.II) checks unless permitted.
- **[D] Transluce logs**: no link
  Network request records, not transcripts.

## 8. Own data

- **[A] cheatbench + orbit + instrumentation** — self-gen
  2-agent, multi-model and same-model swarms with our authors and full instrumentation.
- **[B] SPAR (spar-guardian-agents)**: GPT-6-Luna, Claude Haiku 5.5, DeepSeek-V4.1-Flash
  Our only labelled data from closed authors; labels depend on existing annotations.
- **[B] arena_capstone**: gpt-5-mini, Qwen3.8-27B
  Qwen3.8-27B as both author and reader gives a free author-probe baseline.