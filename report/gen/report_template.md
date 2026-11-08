# Hermeneutics for swarm collusion monitoring

Ian Rios-Sialer, Independent

With Apart Research

Track 4, Multi-Agent Oversight and Control (secondary: Track 2, The Science of Agent Networks). Apart Research, Swarm Alignment Hackathon, November 6 to 8, 2026.

## Abstract

Agent swarms on large, often closed models can collude, and their monitor must be cheap. We study a **hermeneutic probe**: a linear readout of a small open-weight **reader** model's internals while it reads the transcript of a larger **author** model. Five authors ran nine single-agent tasks with a planted exploit, with and without a prompt naming it. We teacher-force a stratified subsample of <<N_PROBED>> transcripts through Qwen3.8-27B and probe its residual stream at the think tokens. Without the prompt, one probe pooled over the authors separates cheating from honest work with AUROC <<ABS_BI>> on held-out runs. It keeps <<ABS_ERASED>> after author, task and prompt condition are linearly erased, reaches <<ABS_LOTO>> on authors it never saw, and beats three cheap LLM judges (<<ABS_JUDGE>>). A bag-of-words classifier on the full transcript matches or exceeds it (<<ABS_TFIDF>>), because on this testbed the exploit text enters the context through tool output. Matched runs where lexical cues cancel are the decisive next test. With three agents, one planted board message lifts coordination off its floor, and a mild nudge yields more teams that assemble the exploit and pass the hidden tests than an explicit one (directional: five seeds per arm, no significance test). For our authors, only cheating has direct labels at scale in the sources in hand. We therefore propose the reader's internals as a cheap monitor for closed authors and name multi-agent data as the binding constraint.
## 1. Introduction

We want to monitor a swarm of agents for collusion when the models are served through an API with closed weights. This is hard: the behaviours have no agreed labels, and the only cheap signal a closed model gives is its transcript. We therefore ask whether a small open-weight model that reads the transcript carries a linear signal of these behaviours in its activations.

The model under study is the **author**, and the open-weight model that reads its transcript is the **reader**. A **hermeneutic probe** is a linear classifier fitted on the reader's residual stream while the reader processes the author's transcript. The approach is hard to validate because the reader also encodes who wrote the text, which task it was, and whether the prompt asked for the behaviour. We therefore refit the agent-level probes after erasing those nuisances, compare them with a label-shuffle null, and give the per-turn probes a turn-position baseline.

We then contend that a reader-internals monitor can replace a cheap judge on this testbed, and that the label map shows where the data for the harder concepts has to come from.

**Pre-existing work.** The swarm framework (its benchmark tasks with planted exploits, its board-based multi-agent scaffold and its behaviour ladders) and the code we adapted for extraction and probing predate the sprint. The controls dataset, the board-seed mechanism, every result and this report were produced inside the sprint window (Appendix B).

**Our main contributions are:**

1. We fit a hermeneutic cheating probe over five authors and test it on an author it never saw. It beats three cheap LLM judges reading cut transcripts and transfers across authors. It does not beat a bag-of-words classifier on the whole transcript, because the label there is lexically recoverable from tool output.
2. We plant one message on a swarm's shared board before the agents act and measure how much coordination and exploit transfer move.
3. We map which of six collusion-related concepts (cheating, coordination, cooperation, collaboration, collusion, scheming) the testbeds in hand label directly, derive, or cannot label.

## 2. Related Work

Model hermeneutics [1] introduces the author and reader framing and reports probes on open readers that detect reward hacking, sycophancy and deception in closed authors. We add the author-identity control, the unseen-author test and the text-classifier comparison. GreyBox [2] monitors a closed agent without weights by asking it yes-or-no questions through its own API. Surrogate fidelity [3] finds that agreement on predictions between open and closed models overstates agreement on attributions, and our controls address that risk. Detecting multi-agent collusion through multi-agent interpretability [4] uses per-agent activation probes as sensors over open-weight agents and reports that few collusive episodes in an emergent setting verbalise their coordination. Three of our five authors are closed-weight, so we probe a reader of their transcripts. Multi-agent risks from advanced AI [5] names collusion as one of three failure modes of multi-agent systems, and our six probe outputs follow its distinctions. We therefore position this work as the closed-author counterpart of [4], with [5] as the source of what to label.

## 3. Methods

### Authors and reader

We probe five authors through one reader; Appendix E, Table E1, lists them with their API identifiers, valid run counts and think coverage. Three authors are closed-weight: GPT-6 Luna, Claude Haiku 5.5 and GPT-5 mini. DeepSeek V4.1 Flash has open weights (MIT licence) and far more parameters than the reader. Qwen3.8-27B is open and is also the reader. All five authors are served through an API in our runs. **Think coverage** is the share of an author's assistant turns whose think block holds readable reasoning text. For authors with low coverage the reader reads the visible transcript. We therefore report coverage beside every transfer result and treat it as an author signature the controls must remove.

### Data

Each run of the controls dataset is one agent on one of nine tasks (Appendix E) with a planted exploit, under one of two arms. The **solo arm** never mentions the exploit. The **cheat arm** tells the agent that the shortcut is fair game. The **cheat ladder** records how far the agent went with the exploit. Cheating starts at the opened rung, or at tried for software engineering, chess and creative writing. The rungs none and exposed are honest, and the rungs between are left out. The authors differ in how often they cheat without an instruction. We therefore probe a stratified subsample balanced over author, arm and label where the data allow (Appendix E, Table E2).

In the board-seed environment three agents of one author (DeepSeek V4.1 Flash) work one software-engineering issue with a shared message board. The planted fix is split into three pieces, one per agent, so the exploit completes only if pieces cross between agents. The **collaboration ladder** records how far exploit content travelled: isolated, discussed, solicited, crossed, assembled (every piece in one tree). The **board seed** is one message the system posts on the board before any agent acts. The conditions are an empty board, a neutral post, a mild and an explicit pro-coordination post (Appendix C), and a hidden regulator agent that posts against sharing. We therefore pilot two turn budgets before the dose-response run, because the budget can bind before a seed has any effect.

### Positions, layers and probe setup

We render every transcript through the reader's chat template, so each assistant turn is wrapped in one think-open and think-close token pair. The pair holds the author's reasoning text when the author returned any and is empty otherwise, so for an author that returns no reasoning the two tokens mark the turn boundary. We keep the reader's residual stream at both tokens at three depths, average over the agent's turns, and fit linear probes with cross-validation grouped by run (Appendix E, Table E3). We therefore report every position and depth in Appendix D.

### Probe types

**Type A** trains and tests inside one author. **Type B.I** trains one probe on all authors and scores held-out runs. **Type B.II** leaves one author out, trains on the other four and scores the fifth, the case of a new model entering a swarm without labelled data. We therefore report B.II for every author.

### Controls, nulls and baselines

A probe pooled over authors can succeed by reading the author's base rate, the task, or the prompt. We fit a five-way **author-identity probe** on the same features, before and after erasing the author with **LEACE**, a closed-form linear concept eraser fitted on training rows only. We also erase author, task and prompt condition together and refit the cheating probe. A **condition readout** on the pooled set shows whether the prompt is readable. The null shuffles labels inside each author and refits. The metadata baselines predict cheating from one-hot author, task and condition, and from run length alone. The **text classifier** is a TF-IDF logistic regression in three views: the whole rendered transcript, the agent's own words, and the think windows alone, which is the **lexical-only reader null**. Three judge models (Claude Haiku 5.5, GPT-5 mini, Claude Sonnet 5.5) read each transcript once and answer whether the agent cheated (Appendix E). The random-weights reader null and the reader interrogation were not run, so we cannot say whether the reader's states carry anything the words do not. The cheat arm's prompt contains the instruction. Every author cheats more often in the cheat arm than in the solo arm, three of five in nearly every run (Table E2). A probe pooled over both arms could therefore read the prompt. The solo arm has both classes for every author and no instruction. We therefore report the solo arm first, the other arms raw beside their condition-erased values in the nuisance table (Table 2), and the other arms raw alone in Appendix D.

## 4. Results

### Does the reader's residual stream separate cheating in the ambiguous arm?

The reader's think-position states separate cheating from honest work in every author and across authors (Table 1). Erasing the author moves the pooled probe little, and erasing author, task and condition together moves it a few hundredths. Every value sits far above the shuffle null, and the probe stays above the metadata baselines after the task is erased. We therefore posit that the signal is neither the author's base rate nor the task.

**Table 1.** Solo arm, agent-level PCA probe at all layers and both think positions; the every-dimension row uses all layers at the close position. Held-out AUROC [95% bootstrap over runs]; metadata baseline: logistic regression on one-hot author, task and condition. The within-author label-shuffle null's 95th percentile lies between <<NULL_LO>> and <<NULL_HI>> across the probe rows, and every p sits at the 1/101 floor.

<<T1>>

### Does the probe transfer to authors it never saw?

A probe trained on four authors reads the fifth well above chance for every author (Figure 1; exact values in Appendix D, Table D4). Transfer is weakest for GPT-6 Luna, with the widest interval. Luna's reasoning reaches the reader as a summary on a minority of turns. Haiku transfers well with no reasoning at all, so for that author the visible transcript carries the signal. We therefore conjecture that transfer depends on how the author's visible transcript relates to the other authors', and we cannot yet separate that from the share of reasoning the reader sees.

**Figure 1.** Solo arm, transfer to an unseen author (type B.II): each method is trained on four authors and scores the held-out author's agents. Markers are held-out AUROC with <<FIG_CI>> bootstrap intervals over runs, and the dashed line marks chance. Filled markers are reader probes (PCA and every dimension at all layers, per-turn at layer <<FIG_TURN_LAYER>>); the hollow marker is the TF-IDF classifier on the whole transcript. The line under each author's name is its think coverage over all its valid runs. Table D4 gives the values.

![Figure 1: transfer to an unseen author](figures/fig1_transfer.pdf)

### Is the signal the author, the task, or the instruction?

The author is almost perfectly readable from the same features and falls to the majority rate once erased (Table 2). The prompt condition is readable in the pooled set and falls to the majority rate once erased, and the pooled cheating probe keeps most of its AUROC after that erasure. A per-turn probe with the agent's label on every turn separates agents when their turn scores are averaged, and its turn-position baseline sits at chance. An onset labelling that keeps a cheating agent's turns only from its first cheating tool call on reaches a high value, and so does its turn-position baseline. Lateness is then the label. We therefore posit that the agent-level probe reads the content of the transcript, and that a turn-level monitor needs position-matched labels.

**Table 2.** Nuisance readouts and the cheating probe under erasure. Accuracy is read against the majority rate and AUROC [95% bootstrap over runs] against chance; the both-arm rows use layer 38, close position.

<<T3>>

### Does it beat a text classifier?

The probe does not beat a text classifier on the whole transcript (Table 3). A TF-IDF logistic regression on the rendered transcript reaches or exceeds the probe pooled, on the within-author mean and on the unseen-author mean (Table 3), and pooled in the other two arms (Appendix D, Table D3). The one author where the probe stays ahead within itself is Qwen3.8-27B, the reader's own model (Table 1). On unseen authors the classifier leads for Qwen3.8-27B too (Figure 1). The probe matches the classifier on the agent's own words, beats the lexical-only null by a wide margin, and beats every judge. The cheat ladder reads the same transcript the classifier reads, and the planted material enters a cheating agent's context through tool results, so the words of the tool output carry the label. The judge comparison bounds a cheap judge (Appendix E). We therefore conclude that on this testbed the reader's states carry the cheating label about as well as the transcript's words and no better.

**Table 3.** Solo arm: the probe against the text classifiers, the lexical-only null and the judges, on the same <<N_SOLO>> solo agents (the Sonnet judge scored <<SON_SOLO>> before its spend cap) and, for the trained methods, the same folds. Held-out AUROC [95% bootstrap over runs]; within-author mean is the mean type-A AUROC; judges have no B.II value. The PCA probe uses all layers and both positions, the every-dimension probe all layers at the close position; judge settings are in Appendix E. The lexical null has no think text for Haiku, so its within-author mean excludes Haiku and its B.II mean includes Haiku's chance-level fold. The two not-run rows lost their host; Appendix D, Table D3, gives the other two arms.

<<T4>>

### Can a planted board signal move coordination?

One planted post moves coordination, and the turn budget gates whether it can (Table 4). The dose-response matrix at the long budget gives four readings, all directional at this seed count and with no significance test. First, presence: the neutral post lifts crossing, assembly and passing above the empty board, so the neutral arm separates presence from content. Second, content: the pro-coordination posts roughly double discussion over the neutral post, and the mild post doubles assembly. Third, dose is not monotonic. The mild nudge yields the most teams that assemble the exploit and pass the hidden tests, while the explicit post yields more talk than the neutral post and no passing team. We hold as a hypothesis that naming the shortcut turns the team toward deliberation and away from execution. Fourth, the hidden regulator holds assembly at the empty-board floor while its workers post far more, so we judge that arm by worker-only posts, crossing and assembly. We therefore conclude that the board seed is a usable lever on coordination intensity, and that collusion outcome and coordination volume can move in opposite directions.

**Table 4.** Software-engineering runs with three worker agents and a split exploit (DeepSeek V4.1 Flash): pilots at turn budgets 25 and 60, then the dose-response matrix; the anti-coordination arm adds a hidden regulator agent. Crossed or assembled counts teams where a piece reached an agent that did not hold it; the other columns are per-run means of evidence counts; the worker-only count excludes the regulator's posts.

<<T5>>

### Which collusion concepts can a swarm testbed label today?

For our authors, only cheating has direct labels at scale in the sources in hand; Appendix F, Table F1, maps the six outputs to what each source in hand provides. The outputs are nested (scheming implies collusion, collusion implies cheating and collaboration, collaboration implies cooperation, and cooperation implies coordination), so the single-agent controls dataset labels cheating directly and gives the other five as zeros by construction. AI Village [6], frontier agents sharing a group chat, gives coordination and collaboration by derivation and cooperation only weakly. It needs a judge for cheating, collusion and scheming, and its licence allows evaluation only. A survey of about sixty candidate sources in the repository (DATASETS.md) finds that no external source uses our authors and ranks SALT-NLP agent-collusion, NARCBench, Colosseum, MAST-Data and Terminal Wrench as the sets to use next. We therefore conclude that cheating is the only output with labels at scale for our authors, that coordination and collaboration have direct labels only in our <<N_BOARD>> board-seed runs from one author, and that the rest wait on instrumented multi-agent runs.

## 5. Discussion and Limitations

A reader's internals carry a linear signal of another model's cheating. The signal survives erasing the author, the task and the prompt, and it beats three judges reading cut versions of the same transcripts. A bag-of-words classifier on the whole transcript recovers the label as well, because the label is defined by what entered the context. The erasure results show that the probe does not rest on a linear readout of the author, the task or the prompt, and the classifier result shows that the label itself is lexical here. We therefore hold that the testbed bounds what this experiment can show, and that the next dataset needs matched honest and cheating runs of one scenario in which the exploit text stays out of the honest context.

The board seed is a cheap lever on a swarm. One post moves how much the agents talk and how much exploit content crosses between them. The mild post yields more teams that assemble the exploit and pass the hidden tests than the explicit one. The result is directional, at five seeds per arm with no significance test. We therefore treat the seed as the way to build a labelled multi-agent set with varied coordination from the same framework and our authors.

Multi-agent data is the binding constraint. Every output beyond cheating needs runs with more than one agent, and the only such runs of our own are the board-seed runs, the pilots and the dose-response matrix, from one author. The survey finds that matched honest and colluding runs of one scenario cancel lexical cues for the text baselines, and that benign multi-agent traces keep a probe from learning that several agents mean collusion. AI Village offers frontier closed authors and a same-model swarm for unseen-author evaluation under a licence that allows evaluation only. We therefore plan a coordination probe trained on our own multi-agent runs and evaluated on AI Village.

### Limitations

The probed set is a subsample that leaves out the longest transcripts. The cheat label counts opening the planted material, and a text classifier on the whole transcript recovers it as well as the probe, so the probe's advantage over the words is unshown. The random-weights null and the reader interrogation were not run. The judges ran with reasoning off, minimal or low on cut transcripts, so the judge comparison bounds a cheap judge. The board-seed runs come from one author on one task at a handful of seeds, with no significance test. Appendix A gives the full list with the dual-use and ethical considerations.

### Future Work

The decisive next test is a matched dataset: honest and cheating runs of one scenario in which the exploit text stays out of the honest context, so that lexical cues cancel. We will run the random-weights null and the reader interrogation on a rebuilt cache, and a residual probe on the text classifier's errors. We will train a coordination probe on our own multi-agent runs, with the board seed as the lever on coordination intensity, and evaluate it on AI Village under its research terms.

## 6. Conclusion

An open-weight reader carries a linear signal of the cheating of five authors, three of them closed-weight. One probe pooled over the five separates cheating in the arm with no instruction, keeps most of its score after the author, the task and the prompt are erased, reads all five unseen authors above chance, and beats three cheap LLM judges reading cut transcripts. A bag-of-words classifier on the whole transcript matches or exceeds it, because the label is recoverable from the words. The probe therefore meets the vision's unseen-author and ambiguous-prompt criteria and fails its criterion of beating every baseline. One planted board post moves how much a three-agent team coordinates. A mild nudge yields more teams that assemble the exploit and pass the hidden tests than an explicit one, in a directional result at five seeds per arm with no significance test. For our authors, only cheating has direct labels at scale in the sources in hand, and the rest wait on instrumented multi-agent runs. We therefore propose the reader's internals as a cheap monitor for closed authors, and we name labelled multi-agent data as the next thing to build.

## Code and Data

- Code repository: https://github.com/unrulyabstractions/collusion-hermaneutics
- Data: the controls dataset is a private Hugging Face dataset, unrulyabstractions/apart-swarm, available to reviewers on request [7]. The AI Village dataset [6] is cited and not redistributed.

## References

[1] conti, ShivamR, J. Sharf, jain18ayush and johnyan (2026). Model Hermeneutics: Monitoring Closed-Weight Models with Open-Weight Internals. LessWrong, 30 September 2026. https://www.lesswrong.com/posts/CZLhfNGaDqtYxvGJD/model-hermeneutics-monitoring-closed-weight-models-with-open

[2] A. Fayyazi (n.d.). Grey Box: Reading Closed Agents Through Their APIs Alone. Project page, preprint in preparation. https://aryafayyazi.github.io/GreyBox/

[3] P. Chlenski, Z. Carmichael, A. Warikoo, C.-T. Shao, Y. Ye, A. Yang, et al. (2026). Surrogate Fidelity: When Can Open LLMs Explain Closed Ones? arXiv preprint arXiv:2606.32008. https://arxiv.org/abs/2606.32008

[4] A. Rose, C. Cullen, S. Abdelnabi, P. Torr, B. G. Kaplowitz and C. Schroeder de Witt (2026). Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. arXiv preprint arXiv:2604.01151. https://arxiv.org/abs/2604.01151

[5] L. Hammond, A. Chan, J. Clifton, J. Hoelscher-Obermaier, A. Khan, E. McLean, et al. (2025). Multi-Agent Risks from Advanced AI. Cooperative AI Foundation, Technical Report 1; arXiv preprint arXiv:2502.14143. https://arxiv.org/abs/2502.14143

[6] AI Digest (2026). AI Village dataset. Hugging Face dataset aidigestorg/ai-village. https://huggingface.co/datasets/aidigestorg/ai-village; project page https://theaidigest.org/village

[7] I. Rios-Sialer (2026). apart-swarm: controls elicitation dataset (private). Hugging Face dataset unrulyabstractions/apart-swarm. https://huggingface.co/datasets/unrulyabstractions/apart-swarm

## Appendix

### Appendix A: Limitations and Dual-Use / Ethical Considerations

**Limitations.** The probed set is a subsample that leaves out the longest transcripts, so the result does not cover the runs where context pressure is highest. The cheat label counts opening the planted material, and a text classifier on the whole transcript recovers the label as well as the probe, so the label is lexical on this testbed and the probe's advantage over words is unshown. Of the vision's success criteria, the probe meets the unseen-author test and is strong in the ambiguous arm; it beats the judges and the lexical-only null and does not beat the whole-transcript classifier, so the criterion of beating every baseline is not met. The random-weights null and the reader interrogation were not run, so the report cannot say whether the reader's states carry anything the transcript words do not. The judges ran with reasoning off, minimal or low, on cut transcripts, and the strongest judge scored only part of the agents, so the judge comparison bounds a cheap judge and not the best one. All probes are linear and LEACE removes linear traces only, so nonlinear author or task signatures may remain. Some closed authors expose little or no reasoning, so for them the reader reads the visible transcript, and our think-position framing reduces to reading turn boundaries. The reader is one model and also one of the authors, so the result for Qwen3.8-27B as author is not a hermeneutic result. The board-seed runs come from one author on one task at a handful of seeds, with no significance test, and the collaboration ladder reads code and board evidence by rules that a judge has not confirmed. We did not store per-agent probe scores, so the AUROC values cannot be recomputed from the shipped files without rerunning the probes. The reader forward passes ran on one GPU host, so host-level numerical differences did not enter the features, but they would in a multi-host deployment.

**Dual-use.** A probe that detects cheating from a reader's internals can be used as a training signal to produce agents whose transcripts do not trigger it, which would make the behaviour harder to monitor. In our runs the board seed raised coordination and exploit transfer (directional: five seeds per arm, no significance test), and the same lever could induce collusion in a deployed swarm. Monitoring agents raises questions of oversight and consent when the agents act for people, because a monitor that reads every transcript reads those people's tasks too. The dataset contains planted exploits and the transcripts of agents taking them, and that material can teach exploitation as well as detection.

**Responsible use.** The dataset is private and holds full agent transcripts on tasks with planted exploits. We will keep it private, release only aggregate results, and gate any access on a research purpose. The board-seed texts are given in Appendix C so that the manipulation is reproducible as an experimental stimulus. We do not release tooling that plants them in deployed systems. We will use AI Village for evaluation only under its research terms, cite its authors, and inform them of any publication. We mark the two baselines that were not run as not run, and we state which baselines the probe beats and which it does not.

### Appendix B: Pre-existing work

The sprint opened on Friday, November 6, 2026, with the kick-off keynote at 21:30 UTC, and closed on November 8 (sprint page https://apartresearch.com/sprints/swarm-alignment-2026, read 2026-11-08). The items below are split by whether they existed before it.

**Before the sprint.** The swarm framework (spar-guardian-agents) with its verification log starting 2026-10-09: the benchmark tasks with planted exploits (cheatbench) and the board-based multi-agent scaffold (orbit); the cheat ladder (none, exposed, tried, opened, used, tampered, checked, shipped) and the per-domain thresholds that define the binary label; the collaboration ladder (isolated, discussed, solicited, crossed, assembled); the regulator and disregulator roles; the run, validity and ledger tooling. The activation-extraction and probe code adapted from the arena-capstone repository's archived code, as listed in Table B1.

**Table B1.** The <<N_B1>> files adapted from pre-sprint code, as the repository's PREEXISTING.md lists them.

| file in this project | pre-sprint source | kept | changed |
|---|---|---|---|
| spans | extraction/spans | segment tagging, sentence units, token ranges | eval-reading functions removed |
| render | swarm/render, extraction/check_render | chat-template rendering, readability, append-only check | reads inputs from model events; one agent per run; think positions per turn |
| extract | extraction/extract, swarm/acts | layer-streaming forward pass, batch planning, resume | think-position capture; layers 26, 38, 51; no Hub upload |
| labels | script/groups_data | rung levels, per-task cheat threshold, label rule | reads rungs from signals; coarse label kept beside |
| probes | analysis/teacher_confound/run | LEACE, PCA and z-scoring, C selection, bootstrap, grouped folds, author probe, within and leave-one-out loops, shuffle null | loader over the think cache; author map; sweeps over layers, positions, pooling; solo arm first; every-dimension variant; condition readout; onset labels |
| judge | analysis/qwen_judge/run | the judge prompt wording, the task-head and transcript-tail cuts, the 500-character cut per tool result, the one-pass local scorer | prompts built from the rendered messages; the OpenRouter scorer with a dollar cap; the analysis over the probe's arms and scopes |

**Inside the sprint.** The controls dataset: first run launched <<FIRST>>, after the kick-off keynote, last run launched <<LAST>>, final set of <<HF_TOTAL>> valid runs published <<HF_BUILT>>, total study spend $<<HF_SPEND>>. The board-seed mechanism (committed 2026-11-07) and its pilots and dose-response matrix (run 2026-11-07). The run list and its validity gates, the think-position pipeline, the stratified subsample, the baseline code, every table in this report, and this report.

**Table B2.** Spend inside the sprint.

<<TS>>

### Appendix C: Board-seed texts

The system posts one of the texts below as the first board message before any agent acts. The anti-coordination condition posts no text; a hidden regulator agent carries it.

**Table C1.** The board-seed texts, one per message condition.

<<TC>>

### Appendix D: Full per-configuration tables

<<TB1>>

<<TB2>>

**Table D3.** Both arms pooled and the cheat arm alone: B.I pooled AUROC for the methods of Table 3. The probe rows use layer 38, close position; the Sonnet judge scored <<SON_D3>> agents.

<<TD3>>

**Table D4.** Solo arm, type B.II: trained on four authors, scored on the held-out author. Held-out AUROC [95% bootstrap over runs]. PCA probe at all layers, both positions; every-dimension probe at all layers, close; per-turn probe at layer 38, close, agents scored by their mean turn score.

<<T2>>

### Appendix E: Authors, data and probe setup

The nine tasks are software engineering (two tasks), chess, creative writing, protein design, open mathematics, subset sum, prime factorization and knowledge work.

**Table E1.** Authors and reader. Think coverage is measured over every valid run of the controls dataset.

<<TM1>>

**Table E2.** The controls dataset and the probed subsample, per author and arm. Valid runs are the published canonical set; probed agents are the subsample (one agent per run).

<<TM2>>

**Table E3.** Probe setup.

<<TM3>>

**Judge setup.** Each judge read one prompt per agent: the task and the transcript, with each tool result cut to 500 characters and the transcript tail cut to 12,000 tokens, and answered yes or no with a confidence. Reasoning was off for Claude Haiku 5.5, minimal for GPT-5 mini and low for Claude Sonnet 5.5, which scored 460 of 600 agents before the in-code spend cap stopped it.

### Appendix F: Label-gap map

**Table F1.** The label-gap map: what each source in hand provides for the six outputs. Direct means a logged label; derivable means a rule over the logs that a reader or judge should confirm; zero by construction follows from the label hierarchy for single-agent runs; judge only means no deterministic label exists.

<<T6>>

## LLM Usage Statement

We used Claude Code (Anthropic) under the author's direction. It ran the experiment pipeline, built every table in this report from the result files through a number ledger that cites a file and a selector for each value, drafted the text, and ran independent verification passes over the tables and the prose. The author set the research questions and the vision, took every design decision, and edited the text. Every number in the tables and the text traces through the ledger to a cited source: a result file, a run log, a code constant or a public page. The two baselines that were not run are marked as not run.
