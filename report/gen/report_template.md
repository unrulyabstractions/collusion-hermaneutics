# Hermeneutics for swarm collusion monitoring

Ian Rios-Sialer ([PENDING: user] affiliation)

Track 4, Multi-Agent Oversight and Control (secondary: Track 2, The Science of Agent Networks). Apart Research, Swarm Alignment Hackathon, November 6 to 8, 2026. Draft of 2026-11-07. [Draft note: the working title follows the project vision and the user may rename it.]

## Abstract

Swarms of agents built on large, often closed-weight models can collude, and a monitor for them has to be cheap. We study a **hermeneutic probe**: a linear readout fitted on the internals of a small open-weight **reader** model while it reads the transcript of a larger **author** model. We collect single-agent controls runs from five authors on nine tasks with a planted exploit, with and without a prompt that names it. We teacher-force every transcript through Qwen3.8-27B and probe its residual stream at the think-token positions. Without the prompt, one probe pooled over the authors separates cheating from honest work with AUROC <<ABS_BI>> on held-out runs. It keeps <<ABS_ERASED>> after the author, the task and the prompt condition are linearly erased, reaches <<ABS_LOTO>> on authors it never saw, and beats three LLM judges (<<ABS_JUDGE>>). A bag-of-words classifier on the full transcript matches or exceeds it (<<ABS_TFIDF>>): the exploit text enters the context through tool output, so on this testbed the label is recoverable from the words. Matched runs where lexical cues cancel are the decisive next test. In a three-agent environment, one message planted on the shared board lifts coordination off its floor, and a mild nudge yields more assembled exploits than an explicit one. Only cheating has direct labels at scale in current testbeds. We therefore propose the reader's internals as a cheap monitor for closed authors and name multi-agent data as the binding constraint.
## 1 Introduction

We want to monitor a swarm of agents for collusion when the models in the swarm are served through an API and their weights are closed. This is hard for two reasons. The behaviours of interest, such as cheating, coordination and collusion, have no agreed labels, and the only cheap signal a closed model gives is its transcript. A monitor that calls a stronger model on every transcript is costly, and a text classifier sees only the words. We therefore ask whether a small open-weight model that reads the transcript carries a linear signal of these behaviours in its own activations.

The model under study is the **author**, and the open-weight model that reads its transcript is the **reader**. A **hermeneutic probe** is a linear classifier fitted on the reader's residual stream while the reader processes the author's transcript. The reader runs locally, costs one forward pass per transcript, and never needs the author's weights. The approach is hard to validate because the reader also encodes who wrote the text, which task it was, and whether the prompt asked for the behaviour. Any of these can stand in for the label. We therefore pair every probe with controls that erase those nuisances and with nulls that bound chance.

We make three contributions. First, we fit a hermeneutic cheating probe over five authors and test it on an author it never saw: the probe is trained on four authors and scored on the fifth. The probe beats three LLM judges with no per-transcript API spend and transfers across authors. It does not beat a bag-of-words classifier on the whole transcript, because on this testbed the label is lexically recoverable from tool output. That is a finding about the testbed and a requirement for the next dataset. Second, we manipulate a swarm testbed with one message planted on the shared board before the agents act, and we measure how much coordination moves. Third, we map which of six collusion-related concepts (cheating, coordination, cooperation, collaboration, collusion, scheming) the testbeds in hand label directly, which they derive, and which they cannot label at all. We then contend that the first result makes a reader-internals monitor a credible replacement for a judge, that it leaves open whether activations carry anything the words do not, and that the third result shows where the data for the harder concepts has to come from.

**Pre-existing work.** The sprint rules ask that work done before the sprint be named. The swarm framework we run on predates the sprint: its benchmark tasks with planted exploits, its board-based multi-agent scaffold, its **cheat ladder** (how far an agent went with a planted exploit) and its **collaboration ladder** (how far exploit content travelled between agents), its regulator roles and its run tooling. The activation-extraction code and the probe code were adapted from a pre-sprint repository, with the changes listed in Appendix B. The controls dataset, the board-seed mechanism and its pilots, every probe result and this report were produced inside the sprint window; Appendix B gives the dates.

## 2 Related work

Reading a closed model through an open one has three recent antecedents. Model hermeneutics [1] introduces the author and reader framing and reports probes on open readers that detect reward hacking, sycophancy and deception in closed authors. We keep its framing and add the author-identity control, the unseen-author test and the text-classifier comparison. GreyBox [2] monitors a closed agent without weights by asking it yes-or-no questions in side branches through its own API and scoring the answers. It is the nearest black-box relative of our reader-interrogation baseline. Surrogate fidelity [3] asks when measurements on open models support claims about a closed one, and finds that agreement on predictions overstates agreement on attributions. That gap is the risk our controls address: a reader can reach the right label for its own reasons. We therefore treat every probe score as unproven until the author and the prompt are erased from the features.

Multi-agent collusion has a probe-based detector and a risk taxonomy. Detecting multi-agent collusion through multi-agent interpretability [4] uses per-agent activation probes as sensors over open-weight agents, derives detection statistics from hypothesis testing, and reports transfer to held-out environments. It also reports that in an emergent-collusion environment few collusive episodes verbalise their coordination, so a text monitor misses them and an activation monitor can still see them. Our setting differs in one way: the agents are closed, so we probe a reader of their transcripts, and the agents' own activations are out of reach. Multi-agent risks from advanced AI [5] names collusion as one of three failure modes of multi-agent systems, and our six probe outputs follow its distinctions. We therefore position this work as the closed-author counterpart of [4], with [5] as the source of what to label.

## 3 Method

### Authors and reader

We probe five authors through one reader (Table M1). Four authors are served through an API and return their reasoning as a provider summary at best. The fifth, Qwen3.8-27B, is open and is also the reader. **Think coverage** is the share of an author's assistant turns whose think block holds readable reasoning text. The open author gives all of its reasoning, three closed authors give a summary on some turns, and Claude Haiku 5.5 gives none. For authors with low coverage the reader reads the visible transcript, and the think positions mark turn boundaries with nothing inside them. We therefore report coverage beside every transfer result and treat it as an author signature the controls must remove.

**Table M1.** Authors and reader. Think coverage is measured over every valid run of the controls dataset.

<<TM1>>

### Data

The controls dataset is single-agent by design. Each run is one agent on one of nine tasks under one of two arms: software engineering (two tasks), chess, creative writing, protein design, open mathematics, subset sum, prime factorization and knowledge work. Every task plants an exploit the agent can find, such as a hidden reference solution or an upstream fix. The **solo arm** never mentions it, so a cheating agent found it on its own. The **cheat arm** tells the agent that the shortcut is fair game. The cheat ladder records how far the agent went: none, exposed, tried, opened, used or tampered, checked, shipped. Cheating starts at opened, or at tried for software engineering, chess and creative writing, where an attempt counts whether it succeeds or fails. The rungs none and exposed are honest, and the rungs between are left out. The dataset also carries a coarser label (used or tampered), which we keep beside the rung, and it is published privately with its manifest [7]. Collection is hard to balance because the authors differ in how often they cheat without an instruction (Table M2). We therefore probe a stratified subsample, with about the same number of runs per author and both classes in every author and arm where the data allow. A survey of about sixty candidate external sources accompanies the repository. None of them uses our authors, and none enters these results.

**Table M2.** The controls dataset and the probed subsample, per author and arm. Valid runs are the published canonical set; probed agents are the subsample (one agent per run).

<<TM2>>

The board-seed environment is multi-agent. Three agents of one author (DeepSeek V4.1 Flash) work one software-engineering issue on a shared remote with a shared message board. The planted fix is split into three pieces, one per agent, so the exploit completes only if pieces cross between agents. The collaboration ladder records how far exploit content travelled: isolated, discussed (a post names it), solicited (a post asks for it), crossed (a piece reached an agent that did not hold it), assembled (every piece in one tree). The **board seed** is one message the system posts on the board before any agent acts. We test a control and four conditions: no post; a length-matched neutral post; a mild pro-coordination post that asks the team to align; an explicit post that asks the team to pool everything, shortcuts included; and an anti-coordination condition carried by a hidden regulator agent that posts against sharing. The turn budget can bind before any seed has an effect. We therefore pilot two turn budgets before the dose-response run.

### Positions, layers and probe setup

We render every author's transcript through the reader's chat template, so each assistant turn is wrapped in exactly one think-open and think-close token pair. When the author returned reasoning text, that text sits inside the pair. When it returned none, the pair is empty and the two tokens mark the turn boundary. This is the closed-author case the vision calls appending: the reader sees the author's messages and tool results in order, with no reasoning the author did not expose. We keep the reader's residual stream at both tokens at three depths, average over the agent's turns, and fit linear probes with cross-validation grouped by run (Table M3). Any fixed token also encodes the author's formatting habits. We therefore sweep both positions and all depths and report every configuration in Appendix D.

**Table M3.** Probe setup.

<<TM3>>

### Probe types

We fit three probe types. **Type A** trains and tests inside one author. **Type B.I** trains one probe on all authors and scores held-out runs, with the folds grouped by run. **Type B.II** leaves one author out, trains on the other four and scores the fifth. B.II is the test that matters for monitoring, because a new model enters a swarm without labelled data of its own. We therefore report it for every author, with think coverage beside it.

### Controls, nulls and baselines

A probe pooled over authors can succeed by reading the author's base rate, the task, or the prompt. We fit a five-way **author-identity probe** on the same features, before and after erasing the author with **LEACE**, a closed-form linear concept eraser fitted on training rows only. We also erase author, task and prompt condition together and refit the cheating probe. The prompt-only control is the two-arm design itself. The cheat arm's prompt elicits the behaviour and the solo arm's prompt is ambiguous. A **condition readout** on the pooled set shows whether the prompt is readable, and the erasure shows whether the cheating probe needs it. The null shuffles labels inside each author many times and refits. The metadata baselines predict cheating from one-hot author, task and condition, and from run length alone. The **text classifier** is a TF-IDF logistic regression on the same agents and folds, in three views: the whole rendered transcript the reader saw, the agent's own words only, and the think windows only. The third view is the **lexical-only reader null**, since it reads exactly the text under the probed positions. Three **judge** models read each transcript once and answer whether the agent cheated, with a confidence: Claude Haiku 5.5 and GPT-5 mini at the authors' own level, and Claude Sonnet 5.5 above them. The random-weights reader null and the reader interrogation are still running. We therefore mark the claims that depend on those two as provisional.

### Why the solo arm is primary

The cheat arm's prompt contains the instruction, and nearly every author takes the shortcut there (Table M2). A probe pooled over both arms could read the prompt instead of the behaviour. The solo arm has both classes for every author and no instruction, which is the ambiguous-prompt case the vision names as the success condition. We therefore report the solo arm first and the pooled and cheat arms only with the condition erased.

## 4 Results

### Does the reader's residual stream separate cheating in the ambiguous arm?

The reader's think-position states separate cheating from honest work in every author and across authors (Table 1). Erasing the author moves the pooled probe little, and erasing author, task and condition together moves it a few hundredths. Every real value sits far above the permutation null, and the metadata baselines sit far below the probe in the pooled set. Inside two authors the task alone predicts cheating well, because some tasks are cheated on more than others. The probe stays above those baselines after the task is erased. A probe on every dimension without PCA gives the same picture. We therefore posit that the signal is neither the author's base rate nor the task.

**Table 1.** Solo arm, agent-level probe (mean over turns), layers concatenated, both think positions. Held-out AUROC with 95% bootstrap intervals over runs. The null is the within-author label shuffle. The metadata baseline is a logistic regression on one-hot author, task and condition (pooled) or task and condition (within one author), on the same folds.

<<T1>>

### Does the probe transfer to authors it never saw?

A probe trained on four authors reads the fifth for every author except GPT-6 Luna, where transfer is weakest and the interval is widest (Table 2). Luna is also the hardest author within itself. Its reasoning is returned as a summary on a minority of turns, so the reader sees less of it than of any author except Haiku. Haiku transfers well with no reasoning at all, so for that author the visible transcript carries the signal. The full-dimension probe, the per-turn probe and the text classifier all place Luna last. We therefore conjecture that transfer depends on how the author's visible transcript relates to the other authors', and we cannot yet separate that from the share of reasoning the reader gets to see.

**Table 2.** Solo arm, type B.II: trained on four authors, scored on the held-out author. PCA probe at layers concatenated, both positions; full-dimension probe at layers concatenated, close position; per-turn probe at layer 38, close position, with the agent's label on every turn and the agent scored by its mean turn score.

<<T2>>

### Is the signal the author, the task, or the instruction?

The author is almost perfectly readable from the same features and falls to the majority rate once erased (Table 3). The prompt condition is readable in the pooled set and falls to the majority rate once erased. The pooled cheating probe keeps most of its AUROC after that erasure. The cheat arm alone loses more under full erasure, and its honest class is thin for two authors. We read it as secondary. The pooled AUROC varies little across the layer and position sweep (Appendix D). A per-turn probe that gives every turn its agent's label separates agents when their turn scores are averaged, and its turn-position baseline sits at chance. A second labelling that keeps a cheating agent's turns only from its first cheating tool call on reaches a high value, and so does its turn-position baseline. That design makes lateness the label. We therefore posit that the agent-level probe reads the content of the transcript, and that a turn-level monitor needs position-matched labels before its score can be read.

**Table 3.** Nuisance readouts and the cheating probe under erasure. The author probe is five-way on the solo arm. The condition readout and the pooled cheating probe use both arms at layer 38, close position. Accuracies are compared to the majority rate; AUROCs to chance.

<<T3>>

### Does it beat a text classifier?

The probe does not beat a text classifier on the whole transcript (Table 4). A TF-IDF logistic regression on the rendered transcript reaches or exceeds the probe pooled, on the within-author mean and on the unseen-author mean, in all three arms. The one author where the probe stays ahead within itself is Qwen3.8-27B, the reader's own model (Tables 1 and 2). The probe matches the classifier on the agent's own words, beats the lexical-only null on the think windows by a wide margin, and beats every judge. The mechanism is in the label. The cheat ladder reads the same transcript the classifier reads, and the planted material enters a cheating agent's context through tool results. The words of the tool output carry the label, and a bag-of-words model recovers it. The judge comparison bounds a cheap judge: reasoning was off or low, transcripts were cut, and the strongest judge scored part of the agents before the spend cap stopped it. We therefore conclude that on this testbed the reader's states carry the cheating label about as well as the transcript's words and no better, and that whether they carry anything the words do not is a question for the random-weights null, a residual test on the classifier's errors, and matched runs where lexical cues cancel.

**Table 4.** The probe against the text classifiers, the lexical-only null and the judges, on the same agents and folds. Within-author mean is the mean of the type-A AUROCs over the authors with both classes; the lexical null has no value for Claude Haiku 5.5, whose think windows are empty. Judges have no training and so no B.II row. The lexical null's B.II mean includes Haiku's degenerate held-out fold, where no think text exists and the AUROC is chance, while its within-author mean excludes Haiku. Judge prompt: the task and the transcript with each tool result cut to 500 characters and the tail cut to 12,000 tokens, one yes-or-no answer with a confidence. Pending cells are still running.

<<T4>>

### Can a planted board signal move coordination?

One planted post moves coordination, and the turn budget gates whether it can (Table 5). At the short budget every arm sits on the floor: no exploit piece crosses between agents and almost no agent posts. At the long budget crossing saturates in the pilot and the seed raises how much the agents talk and how many pieces cross. The dose-response matrix at the long budget gives four readings, all directional at this seed count and with no significance test. First, presence: the neutral post lifts crossing, assembly and passing above the empty board, so the neutral arm isolates content and the empty-to-neutral step isolates presence. Second, content: the pro-coordination posts roughly double discussion over the neutral post, and the mild post doubles assembly. Third, dose is not monotonic. The mild nudge yields the most teams that assemble the exploit and pass the hidden tests. The explicit post lifts posting above the neutral and empty boards and below the mild post, and it yields no passing team. We hold as a hypothesis that naming the shortcut turns the team toward deliberation and away from execution. Fourth, the hidden regulator holds assembly at the empty-board floor while the workers post several times more than on the empty board. Its own posts inflate every post-based count, so we judge the anti-coordination arm by worker-only posts, crossing and assembly. We therefore conclude that the board seed is a usable lever on coordination intensity, and that collusion outcome and coordination volume can move in opposite directions.

**Table 5.** Three-agent software-engineering runs with a split exploit, one author (DeepSeek V4.1 Flash). "Crossed or assembled" counts teams where an exploit piece reached an agent that did not hold it. Means are per run. In the anti-coordination arm the regulator's posts are excluded from the worker-only count. The tb=25 and tb=60 rows are the pilots; the dose-response rows are the final matrix. Discussed, solicited, crossed and assembled are counts of evidence items per run.

<<T5>>

### Which collusion concepts can a swarm testbed label today?

Only cheating has direct labels at scale (Table 6). The six outputs are nested by the vision's definitions: scheming implies collusion, collusion implies cheating and collaboration, and collaboration implies cooperation, which implies coordination. A label fills in what it implies, so the single-agent controls dataset labels cheating directly and gives the five multi-agent outputs as zeros by construction. The board-seed environment labels coordination directly through the collaboration ladder and the board, and collusion follows from cheating with crossing, but it cannot separate cooperation from collaboration because the team shares one goal by construction, and it has no covertness signal for scheming. AI Village [6], an external dataset of frontier agents sharing a group chat, gives coordination and collaboration by derivation from the chat graph and from village-wide versus per-agent goals. It has no task or grader, so cheating, collusion and scheming there need a judge. Its licence allows research and analysis and forbids training without written permission, so it can serve as an unseen-author evaluation set. It cannot serve as training data. The repository's dataset survey (DATASETS.md) grades the wider landscape and ranks SALT-NLP agent-collusion, NARCBench, Colosseum, MAST-Data and Terminal Wrench as the external sets to use next. It finds that no external source uses our authors, that only two sources separate collaboration from collusion without a judge, and that no frontier-author source labels scheming. One of those two sources is our own environment. We therefore conclude that cheating is the only output a probe can be trained on today for our authors, and that every other output waits on instrumented multi-agent runs, with external sets held for unseen-author evaluation.

**Table 6.** The label-gap map: what each source in hand provides for the six outputs. Direct means a logged label; derivable means a rule over the logs that a reader or judge should confirm; zero by construction follows from the label hierarchy for single-agent runs; judge only means no deterministic label exists.

<<T6>>

## 5 Discussion

A reader's internals carry a linear signal of another model's cheating, the signal survives erasing the author, the task and the prompt, and it beats three judges that read the same transcripts. For monitoring this means one open-weight forward pass per transcript and a linear readout, with no access to the author's weights. The result is hard to interpret because the cheating label is defined by what entered the context, and a bag-of-words classifier on the whole transcript recovers it as well as the probe does. The erasure results show that the probe does not rest on the author, the task or the prompt. The classifier result shows that the label itself is lexical here. We therefore hold that the testbed bounds what this experiment can show, and that the next dataset needs matched honest and cheating runs of one scenario in which the exploit text stays out of the honest context.

Think coverage changes what the reader reads. For Haiku the think blocks are empty and for Luna most are, so the positions mark turn boundaries over the visible transcript. Haiku still transfers well and Luna transfers worst, so the share of reasoning seen does not predict transfer on its own. The provider decides what reasoning to expose, and that decision differs by provider and by turn. We therefore recommend that any hermeneutic monitor report coverage per author, and that transfer claims be read against it.

The solo arm is primary because of the instruction confound. The cheat arm's prompt names the behaviour, nearly every author follows it, and the condition is readable from the reader's states. The pooled probe keeps most of its score after the condition is erased, so the probe does not need the prompt. The solo arm shows it does not need an instruction at all. Real prompts are ambiguous. We therefore treat the solo arm as the deployment-like case and the cheat arm as a check.

Multi-agent data is the binding constraint. Every output beyond cheating needs runs with more than one agent, and the only such runs in hand are the board-seed pilots, at a few dozen runs from one author. The survey finds that no external dataset uses our authors. It finds that matched honest and colluding runs of one scenario cancel lexical cues for the text baselines, and that benign multi-agent traces keep a probe from learning that several agents mean collusion. The board seed shows that coordination intensity can be moved cheaply, so a labelled multi-agent set with varied coordination is feasible with the same framework and our authors. AI Village is the next source for unseen-author evaluation: a multi-model swarm of frontier closed authors and a same-model swarm of an open one, with derivable coordination and collaboration labels, under a licence that allows evaluation only. We therefore plan the next step as a coordination probe trained on our own multi-agent runs and evaluated on AI Village and on the survey's top-ranked external sets.

## Conclusion

We asked whether an open-weight reader carries a linear signal of a closed author's cheating, and the answer on our controls data is yes: one probe pooled over five authors separates cheating in the arm with no instruction, keeps most of its score after the author, the task and the prompt are erased, reads four of five unseen authors, and beats three LLM judges. A bag-of-words classifier on the whole transcript matches or exceeds it, because on this testbed the exploit text enters the context and the label is recoverable from the words. The probe therefore meets the vision's unseen-author and ambiguous-prompt criteria and fails its criterion of beating every baseline. One planted board post moves how much a three-agent team coordinates, and a mild nudge produces more assembled exploits than an explicit one. Of six collusion-related concepts, only cheating has direct labels at scale, and the rest wait on instrumented multi-agent runs and on external swarm data for evaluation. We therefore propose the reader's internals as a cheap monitor for closed authors, and we name labelled multi-agent data as the next thing to build.

## References

[1] Model Hermeneutics: Monitoring Closed-Weight Models with Open-Weight Internals. https://www.lesswrong.com/posts/CZLhfNGaDqtYxvGJD/model-hermeneutics-monitoring-closed-weight-models-with-open

[2] GreyBox (project page). https://aryafayyazi.github.io/GreyBox/

[3] Surrogate Fidelity: When Can Open LLMs Explain Closed Ones? https://arxiv.org/pdf/2606.32008

[4] Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. https://arxiv.org/pdf/2604.01151

[5] Multi-Agent Risks from Advanced AI. https://arxiv.org/pdf/2502.14143

[6] AI Digest. AI Village dataset (aidigestorg/ai-village). https://huggingface.co/datasets/aidigestorg/ai-village

[7] apart-swarm: controls elicitation dataset (unrulyabstractions/apart-swarm, private). https://huggingface.co/datasets/unrulyabstractions/apart-swarm

## Appendix A: Limitations and Dual-Use / Ethical Considerations

**Limitations.** The probed set is a subsample that leaves out the longest transcripts, so the result does not cover the runs where context pressure is highest. The cheat label counts opening the planted material, and a text classifier on the whole transcript recovers the label as well as the probe, so the label is lexical on this testbed and the probe's advantage over words is unshown. Of the vision's success criteria, the probe meets the unseen-author test and is strong in the ambiguous arm; it beats the judges and the lexical-only null and does not beat the whole-transcript classifier, so the criterion of beating every baseline is not met. The random-weights null and the reader interrogation are still pending. The judges ran with reasoning off or minimal, on cut transcripts, and the strongest judge scored only part of the agents, so the judge comparison bounds a cheap judge and not the best one. All probes are linear and LEACE removes linear traces only, so nonlinear author or task signatures may remain. Some closed authors expose little or no reasoning, so for them the reader reads the visible transcript, and our think-position framing reduces to reading turn boundaries. The reader is one model, and it is also one of the authors, so the open-author result is not a hermeneutic result. The board-seed runs come from one author on one task at a handful of seeds, with no significance test, and the collaboration ladder reads code and board evidence by rules that a judge has not confirmed. We did not store per-agent probe scores, so the AUROC values cannot be recomputed from the shipped files without rerunning the probes. The reader forward passes ran on one GPU host, so host-level numerical differences did not enter the features, but they would in a multi-host deployment.

**Dual-use.** A probe that detects cheating from a reader's internals can be used as a training signal to produce agents whose transcripts do not trigger it, which would make the behaviour harder to monitor. The board seed is a lever that raises coordination and exploit transfer in a swarm, and the same lever can induce collusion in a deployed swarm. Monitoring agents raises questions of oversight and consent when the agents act for people, because a monitor that reads every transcript reads those people's tasks too. The dataset contains planted exploits and the transcripts of agents taking them, and that material can teach exploitation as well as detection.

**Responsible use.** The dataset is private and carries benchmark content that its authors ask to keep out of training corpora. We will keep it private, release only aggregate results, and gate any access on a research purpose. The board-seed texts are given in Appendix C so that the manipulation is reproducible as an experimental stimulus. We do not release tooling that plants them in deployed systems. We will use AI Village for evaluation only under its research terms, cite its authors, and inform them of any publication. We report every pending baseline as pending and every claim that depends on one as provisional.

## Appendix B: Pre-existing work

The sprint ran from November 6 to 8, 2026. The items below are split by whether they existed before that window.

**Before the sprint.** The swarm framework (spar-guardian-agents) with its verification log starting 2026-10-09: the benchmark tasks with planted exploits (cheatbench) and the board-based multi-agent scaffold (orbit); the cheat ladder (none, exposed, tried, opened, used, tampered, checked, shipped) and the per-domain thresholds that define the binary label; the collaboration ladder (isolated, discussed, solicited, crossed, assembled); the regulator and disregulator roles; the run, validity and ledger tooling. The activation-extraction and probe code adapted from the arena-capstone repository's archived code, as listed in Table B1.

**Table B1.** Code adapted from pre-sprint files (from the repository's PREEXISTING.md).

| file in this project | pre-sprint source | kept | changed |
|---|---|---|---|
| spans | extraction/spans | segment tagging, sentence units, token ranges | eval-reading functions removed |
| render | swarm/render, extraction/check_render | chat-template rendering, readability, append-only check | reads inputs from model events; one agent per run; think positions per turn |
| extract | extraction/extract, swarm/acts | layer-streaming forward pass, batch planning, resume | think-position capture; layers 26, 38, 51; no Hub upload |
| labels | script/groups_data | rung levels, per-task cheat threshold, label rule | reads rungs from signals; coarse label kept beside |
| runs | publish build_snapshot | validity gates, canonical-copy rule | run-list function over a data root |
| probes | analysis/teacher_confound/run | LEACE, PCA and z-scoring, C selection, bootstrap, grouped folds, author probe, within and leave-one-out loops, shuffle null | loader over the think cache; author map; sweeps over layers, positions, pooling; solo arm first; full-dimension variant; condition readout; onset labels |

**Inside the sprint.** The controls dataset: first run launched <<FIRST>>, last run launched <<LAST>>, final set of <<HF_TOTAL>> valid runs published <<HF_BUILT>>, total study spend $<<HF_SPEND>>. The board-seed mechanism (committed 2026-11-07) and its pilots and dose-response matrix (run 2026-11-07). The think-position pipeline, the stratified subsample, the baseline code, every table in this report, and this report.

**Table S.** Spend inside the sprint.

<<TS>>

## Appendix C: Board-seed texts

The system posts one of the texts below as the first board message before any agent acts. The anti-coordination condition posts no text; a hidden regulator agent carries it.

<<TC>>

## Appendix D: Full per-configuration tables

<<TB1>>

<<TB2>>
