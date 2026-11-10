# Hermeneutics for swarm collusion monitoring

Ian Rios-Sialer, Independent

Correspondence: unrulyabstractions@gmail.com

## Abstract

Agent swarms built on large, often closed models can collude, and their monitor must be cheap. We study a **hermeneutic probe**: a readout of a small open-weight **reader** model's internals while it reads the transcript of a larger **author** model. We label six nested behaviours, from cheating to scheming, by code from each run's own files. We read the reader's residual stream on ten labelled sets: our single-agent controls and team runs, earlier swarm runs, and three external sets. Linear probes and an attention monitor over the steps carry every label far above label-shuffle nulls. They also beat three cheap LLM judges and the reader asked in words on every set but a small mixed-model one. A bag-of-words classifier on the agent's own words matches or beats them on most sets, because the cheating text enters the context. Reader internals beat every baseline in three places: the controls' authors seen in training, coordination in the board sets, and per-step detection on Goodfire rollouts with passage labels. They lose on unseen authors, on sets where the prompt already carries the label, and on the mixed-model set. On a source held out of training the probe leads the words for coordination. We therefore propose reader internals as an early monitor beside the words, and we name labelled multi-author team data as the binding constraint.

## Introduction

We want to monitor a swarm of agents for collusion when the models are served through an API with closed weights. This is hard. The behaviours have no agreed labels, the main cheap signal a closed model gives is its transcript, and any monitor must be judged against a classifier on the same words. We therefore ask whether a small open-weight model that reads the transcript carries a signal of these behaviours in its activations.

The model under study is the **author**, and the open-weight model that reads its transcript is the **reader**. A **hermeneutic probe** is a classifier fitted on the reader's residual stream while the reader processes the author's transcript [@conti2026hermeneutics]. The reader also encodes who wrote the text, which task it was, and what the prompt asked for. We therefore erase those nuisances and compare every probe with a label-shuffle null. We also set every probe beside a text classifier, three cheap LLM judges and the reader's own spoken verdict on the same agents.

Our main contributions are:

1. Six nested ladders (cheating, coordination, cooperation, collaboration, collusion and scheming) computed by code per agent, per team and per call, on our runs and on the SPAR team's swarm runs.
2. Reader probes and an attention monitor over the steps, compared on ten sets with text, metadata, judge and reader-interrogation baselines and with label-shuffle nulls, and a probe matrix over sources, probe types and nine architectures.
3. A prompt-only probe and an author-erasure control that locate where the reader's advantage is prompt or author identity.
4. Design runs that measure how much telling lone agents need to use a shared board, and a planted board post that moves coordination in teams.

We then contend that reader internals give an early and cheap monitor for coordination and for behaviour on known authors, and that the words remain the bar on new authors. Appendix B separates the work made here from the work it builds on.

## Related work

Model hermeneutics introduces the author and reader framing and reports probes on open readers that detect reward hacking, sycophancy and deception in closed authors [@conti2026hermeneutics]. We add the unseen-author test, the prompt-only and author-erasure controls, and the comparison with the words and with judges on every set. GreyBox monitors a closed agent without weights by asking it yes-or-no questions through its own API [@fayyazi_greybox]; our reader interrogation asks the open reader the same kind of question. Surrogate fidelity finds that agreement on predictions between open and closed models overstates agreement on attributions [@chlenski2026surrogate], and our controls address that risk. Detecting multi-agent collusion through multi-agent interpretability places activation probes inside open-weight agents and reports that few collusive episodes verbalise their coordination [@rose2026collusion]. Most of our authors are closed-weight, so we probe a reader of their transcripts. Multi-agent risks from advanced AI names collusion as a failure mode of multi-agent systems [@hammond2025multiagent], and our six ladders follow its distinctions. LEACE erases a linear concept in closed form [@belrose2023leace], and we use it for author, task and condition. We therefore position this work as the closed-author counterpart of probes inside open agents, with explicit labels as its main ingredient.

## Methods

A transcript carries its author, task and prompt beside the behaviour, and our methods must separate them. We describe the sets, the labels, the features, the probes and the controls in turn.

### Testbeds and data

We read ten labelled sets (Table F2). The **controls** put one agent on one of nine CheatBench tasks with a planted exploit [@phan2026cheatbench], under two arms. The **solo arm** never mentions the exploit, and the **cheat arm** says the shortcut is fair game. The **board-seed** runs put three workers of one author on one software issue with a shared board from the ORBIT scaffold [@hagag2026orbit], with the planted fix split into one piece per agent. The **board seed** is one message the system posts before any agent acts ([Supplement, Section 9](supplement.pdf#section.9)). The **design runs** place three lone agents on one task with board tools and no team in the prompt. The **telling ladder** sets how much they hear about the board: tools only (L0), round notices and one system post (L1), or a prompt that names the board (L2). We also read the swarm runs of the SPAR team that built the SparGuardianAgents framework [@cierniak2026spar; @cierniak2026sparswarm], and three external sets: Goodfire's reward-hacking rollouts [@goodfire2026rewardhack], human-labelled CheatBench traces [@steinad2026cheatbench] and EvilGenie's matched pairs [@wiserhuman2026evilgenie].

### Six nested ladders, direct and proxy

We label six nested behaviours. Each is a **ladder**, an ordered set of levels that code computes from the run's own files per agent, per team and per call, with no model in the loop (Appendix G). Cheating and coordination are **direct**: the framework's cheat ladder and its board and collaboration evidence [@cierniak2026spar]. Each task sets a **cheat floor**, the cheat level from which a run counts as cheating. Collusion is a direct rule, the coordination level counted once the agent's cheat level reaches used. Cooperation, collaboration and scheming are **proxies**, rules over recorded actions that stand in for an intent the logs do not record. Levels the framework records without a call index enter at an estimated onset. The external sets keep their own labels.

### Positions and features

The reader is Qwen3.8-27B. We render every transcript through its chat template, so each assistant turn opens with five boundary tokens and is wrapped in a think-open and think-close pair. The pair holds the author's reasoning when the author returned any and is empty otherwise. We keep the residual stream at three depths and use two feature sets: the **think pair**, and the **seven tokens**, which add the five boundary tokens (Appendix F).

### Probe types A, B.I and B.II

**Type A** trains and tests inside one author, **type B.I** pools all authors, and **type B.II** leaves one author out and scores it. Every fit uses cross-validation grouped by run, so no run is split. An **inclusion scheme** decides which sources a probe sees: each source alone, our sets, the external sets, all sets, all sets but one (the **leave-one-source-out** test), or our sets applied to the external ones and back. The **probe matrix** crosses the three probe types with these schemes for cheating, coordination and collusion.

### Architectures

The **linear probe** fits training-fold PCA, z-scoring and L2 logistic regression with the penalty chosen by grouped inner cross-validation. It reads either the mean over an agent's turns or one row per turn. The **attention monitor** reads a trajectory one step at a time. An MLP encodes each step's token vectors, two causal self-attention layers mix the steps so far, and heads per ladder map the result to an ordinal level and a binary label, with inner ladders gated by outer ones. It trains on the level reached at each call with early stopping on held-out runs. On the main cells of the probe matrix we compare nine architectures: the linear probe on pooled states; an MLP per step with a running mean or a running max; attention pooling with one learned query; the attention monitor; a GRU; the linear per-step probe with a running mean of its scores; an MLP on the pooled vector; and a **stack**, a logistic regression over the held-out scores of three probes and TF-IDF.

### Controls and nulls

The **label-shuffle null** permutes labels inside author or arm and refits. The metadata baseline reads author, task, arm, seed and turn position. The **text classifier** is TF-IDF with logistic regression on the agent's own words, on the whole rendered text, or on the think windows alone; the last is a **lexical-only null** for the reader. Three cheap judges read each transcript once and answer a yes-or-no question with a confidence, and the **reader interrogation** puts the same question to the reader itself. The **prompt-only probe** reads the reader's state before the first assistant token, beside the label rate of each prompt condition. LEACE erases author, or author, task and condition together, on training rows only [@belrose2023leace]. A random-weights reader is planned and not run.

## Results

### Does the reader carry the signal?

We first ask whether the reader's states carry the label at all, beyond the author's base rate and the task. On the controls' solo arm the agent-level probe separates cheating inside every author and pooled over authors (Table 1). Erasing the author, or author, task and condition together, moves it little, and every probe sits far above its within-author shuffle null. The same holds for every label we probe on the team and design sets ([Supplement, Section 4](supplement.pdf#section.4)). We therefore posit that the reader carries the behaviour, and not only who wrote the transcript or which task it was.

**Table 1.** Controls, solo arm: agent-level PCA probe at all layers and both think positions; the every-dimension row uses all layers at the close position. Held-out AUROC [95% bootstrap over runs]; metadata: logistic regression on author, task and condition. The within-author label-shuffle null's 95th percentile lies between <<NULL_LO>> and <<NULL_HI>> across the raw-feature probe rows, and every p sits at the 1/101 floor. {wide}

<<T1>>

### Does it beat text baselines, and where?

A probe that only matches the words adds cost and no signal, so the bar is a text classifier on the same transcript. Table 2 sets every scorer on the same agents of each set. The linear probe and the attention monitor beat the three cheap judges and the reader's spoken verdict on every set but the SPAR mixed teams, where the strongest judge leads. Against TF-IDF on the agent's words the internals win on cheating on the controls and on coordination in the board-seed teams, the design runs and the SPAR screen teams. On every other row the classifier matches or beats them, because the exploit text enters the context through tool output and the agent repeats it. The judges trail partly by construction: their reading window cuts most transcripts and leaves many onsets outside it ([Supplement, Section 8](supplement.pdf#section.8)). We therefore conclude that reader internals beat a cheap judge almost everywhere, and beat the words mainly on coordination.

**Table 2.** Every scorer on the same held-out agents of each set: AUROC. Linear probe readout: the agent's mean over turns, or the per-step probe's score at the last step. Attention monitor and TF-IDF at the last step where the probe reads per step. Best cheap judge: the highest of three cheap judges (DeepSeek V4.1 Flash, Qwen3.7 Flash, GLM 5.3 Flash); reader interrogation: the reader asked the judges' question. A scorer that could not score an agent leaves it out: a judge reply that did not parse, or a Goodfire rollout the monitor could not align (at most <<MAIN_SHORT>> of a row's agents). [Supplement, Section 8](supplement.pdf#section.8) gives intervals and every judge. {wide}

<<T_MAIN>>

### Does it transfer to unseen authors?

A monitor must work on authors it never saw, and an author leaves a strong signature in the reader. Table 3 trains on some authors or sets and scores others. The words transfer better on almost every split. On the controls, the design runs, the SPAR mixed teams and the three external sets the classifier leads the linear probe. The attention monitor passes the classifier on the SPAR screen teams with both seeds, trails it on the controls, and falls below chance on the design runs. Transfer across sets loses much of the signal for every method, and the classifier stays ahead. We therefore conclude that the words set the bar on new authors.

**Table 3.** Unseen authors and unseen sets: agent AUROC. Leave one author out: mean of the type B.II AUROCs, with whole-text TF-IDF. Held-out author, pooled: the attention monitor's and the linear per-step probe's last-step scores, pooled over held-out authors, with TF-IDF on the words so far. External sets: AUROC inside the held-out author (Goodfire: inside one run directory), with TF-IDF on the agent's words. Transfer: fitted once on the source set and applied unchanged. [Supplement, Section 5](supplement.pdf#section.5) gives each held-out author. {wide}

<<T_UNSEEN>>

### Which probe type and which architecture hold up across sources?

A monitor deployed on a new swarm meets new sources as well as new authors. The probe matrix tests this by crossing probe types, inclusion schemes and architectures (Table 4). Inside a distribution the text classifier beats the linear probe on every cheating cell, and the probe leads only on coordination in our sets and the SPAR screen teams and on collusion in the mixed teams ([Supplement, Section 6](supplement.pdf#section.6)). On a held-out source the order turns for coordination: the probe leads the classifier whichever source is held out. For cheating it leads only when CheatBench or EvilGenie is held out, and for collusion only without the SPAR screen teams. Unseen authors lower the linear probe's score on every pooled scheme. Transfer between sources is weak and asymmetric: a probe trained on our sets reads the external sets poorly, and one trained on the external sets falls below chance on ours. Among the trained architectures the MLP per step with a running max holds best on unseen authors for cheating and coordination on our sets. On a held-out source the best trained architecture beats the linear probe for coordination by a wide margin, though we choose it on the test score itself. No architecture wins everywhere: on unseen authors the linear probe matches or beats every trained architecture for collusion on our sets, and with CheatBench held out it leads them all. The stack reads TF-IDF too and scores next to the text alone. The per-step probe beats its null, and on cheating and collusion it edges metadata with turn index only slightly ([Supplement, Section 7](supplement.pdf#section.7)). We therefore posit that no single architecture dominates, and that the reader's advantage over the text lies on unseen sources and on coordination.

**Table 4.** Probe matrix, main cells: held-out AUROC, type B.I unless marked B.II (B.II: mean over held-out authors). Null: the 95th percentile of the linear probe's 20-draw label-shuffle null. Best architecture: the highest AUROC among the seven trained architectures (the linear probe and the stack excluded; ties named together; MLP, mean and MLP, max: an MLP per step with a running mean or max; pooled MLP: an MLP on the pooled vector), selected on the held-out score itself and therefore optimistic. Architectures were not run on B.II of the leave-one-source-out schemes or on the transfer schemes. [Supplement, Section 6](supplement.pdf#section.6) gives every cell. {wide}

<<T_MATRIX_MAIN>>

### Does it detect earlier per step?

A per-step monitor is useful only if it fires soon after the onset. Lateness alone predicts most labels in a run, so turn position is the baseline to beat. On the board-seed teams the seven-token probe detects every agent that reaches cheating, and sooner than the classifier on the words so far (Table 5). For collusion the probes fire sooner too, but turn position alone reads the turns as well. On Goodfire, where the dataset marks the flagged passage, the linear probe finds the flagged turn itself far better than the classifier and the turn position, with no delay. When every turn after the first flag counts as positive, the attention monitor and the classifier lead by a little. A cheap judge reading growing prefixes never fires on either set at the false-positive rate the probes use. We therefore posit that reader internals give the earliest per-step alarm, and that every per-step claim must be set against turn position.

**Table 5.** Per step. Turn AUROC scores held-out turns against the level reached at that call. Delay: median turns from the onset to the first score above the threshold at a 5% false-positive rate on held-out negative turns. Detected: agents (Goodfire: hacked rollouts) that reach the level and cross the threshold. Goodfire definition A marks the turns holding a flagged passage; definition B marks every turn from the first flag on. The judge rows read cumulative prefixes of a balanced subsample. [Supplement, Section 7](supplement.pdf#section.7) adds intervals. {wide}

<<T_STEP_ALL>>

### What do the telling-ladder design runs show?

Labelled team data needs agents that find and use a shared board, so we measure how much telling lone agents need. With board tools alone almost no agent reads the board (Table C1). Round notices and a bare system post bring nearly every agent to read it, and naming the board in the prompt adds posting, though posting stays rare. Under one prompt DeepSeek V4.1 Flash cheats in most runs and GPT-6 Luna in few. In teams, one planted post moves coordination, and the mild post yields the most teams that pass the hidden tests ([Supplement, Section 9](supplement.pdf#section.9)). On the design runs the attention monitor reads coordination far above the classifier (Table 2). We therefore treat the telling ladder and the board seed as cheap levers for building labelled team data.

### What do the controls say about prompt and author identity?

A probe can win by reading the prompt condition or the author, both of which the reader sees before any behaviour. The prompt-only probe reads the reader's state before the first assistant token. It stays near its null for cheating on the controls, the board-seed teams and the design runs, and for coordination and collusion on the board-seed teams (Table 6). There the label is not readable from the prompt. It carries a large share of the full probe's margin for cheating and collusion on the SPAR screen teams, the SPAR exploration and mixed sets and the three external sets. It does the same for coordination in the design runs. Inside each arm of the controls the prompt-only probe falls to chance while the full probe keeps its score (Table E1). The reader encodes the author almost perfectly, and LEACE erases it to chance (Table E2). The cheating probe then moves little on the controls, Goodfire and the SPAR mixed teams, and more on the design runs and the SPAR screen teams, where erasing task and condition with the author removes most of the unseen-author margin. The think-window text alone matches the probe on the team and design sets and trails it on the controls ([Supplement, Section 3](supplement.pdf#section.3)). We therefore conclude that part of the reader's advantage on most SPAR sets and on the external sets is prompt identity, and that our controls and board-seed teams are clean of it.

**Table 6.** Prompt-only control (type B.I, pooled): AUROC of a probe on the reader's state before the first assistant token (our sets: the five boundary tokens and the think-open token of the first turn; external sets: the think-open token), its 50-draw label-shuffle null, and the training rate of each task and arm. Full probe: think pair, all layers. Share: the prompt-only margin over 0.5 as a fraction of the full probe's. {wide}

<<T_PROMPT>>

## Discussion

Reader internals beat every baseline in three places. On authors seen in training, the attention monitor beats the classifier, the judges and the reader's verdict on the controls. On coordination, the probes and the monitor beat them on the board-seed teams, the design runs and the SPAR screen teams, by a small margin on the board-seed teams. Per step on Goodfire, with the dataset's passage labels, the linear probe finds the flagged turn far better than the words and with no delay. On a source held out of training, the linear probe also leads the words for coordination. They do not beat the words on unseen authors, except the monitor on the SPAR screen teams. They do not escape prompt identity on most SPAR sets and on the external sets, where the prompt alone carries much of their margin. On the mixed-model SPAR set the classifier and the judges lead. We therefore hold that reader internals add early warning and sensitivity to coordination, and that the words remain the bar for cheating on new authors.

The judges are weaker than they need be. Their reading window keeps the task and the end of the transcript, so many onsets fall outside it. The coordination question asks about the exploit while the code label counts any post. The reader we interrogated was served at reduced precision while the probes read its full weights, and one judge wrote the board-seed transcripts it read. Multi-agent data remains the binding constraint. The board seed and the telling ladder give a cheap lever to build labelled team sets, and AI Village offers frontier authors in a shared chat under a licence that allows evaluation only [@aidigest2026village]. We therefore plan multi-author team runs at the seed counts a significance test needs, with AI Village as an unseen-author test.

## Limitations

The board-seed teams come from one author on one task, so their probes say nothing about other authors. Cooperation, collaboration and scheming are proxies that no judge has confirmed, and the onsets of used and crossed are estimates. On the team sets turn position alone reads cheating and collusion well, so a late-firing monitor can look strong. The design and board-seed cells hold few runs and carry no significance test. The tables mix readouts, the agent mean on some sets and the last step on others, so a column is comparable within a row more than across rows. The external labels are other people's, and Goodfire's come from an LLM judge. The cheap judges read a cut transcript, the interrogated reader ran at reduced precision, and the best judge and the best architecture are chosen after the fact. The trained architectures carry no permutation null, and several matrix cells sit at or below their null. LEACE removes linear traces only. The reader is also one of the authors. A few long SPAR transcripts lose their late turns. Many per-step and monitor rows still lack a null, and the random-weights reader was not run, so we cannot yet say whether the reader adds anything a random network on the same tokens would not.

## Conclusion

An open-weight reader carries the cheating, coordination and collusion of the authors it reads, most of them closed-weight. Its internals score far above label-shuffle nulls, and above cheap judges and its own spoken verdict on almost every set. A classifier on the agent's words matches it for cheating, because the exploit enters the transcript. The classifier also leads it on unseen authors. The reader's internals lead on coordination, on authors seen in training, and per step when the onset is marked, and part of their advantage on most SPAR sets and on the external sets is the prompt. We therefore propose reader internals as an early monitor beside the words, and we name labelled multi-author team data as the next thing to build.

## Code and data

The code is public at https://github.com/unrulyabstractions/collusion-hermaneutics. The activations, transcripts and labels of every set we made are public on the Hugging Face Hub [@riossialer2026align], and the SPAR team's runs are public too [@cierniak2026sparswarm]. The full controls dataset stays private [@riossialer2026apartswarm]. Appendix F lists the data.

## Acknowledgements

The board-seed runs and the controls dataset began at the Apart Research Swarm Alignment Hackathon. We thank Rigel Cierniak and Lucy Kappai for the SPAR swarm runs and the framework they built.

## References

## Appendix

The appendices hold what a reader needs to check the paper's claims. [The supplementary tables](supplement.pdf) give every cell, grouped by question.

### Appendix A: Limitations and dual-use considerations

This appendix lists what the results do not cover and how the work could be misused. It also says what we release and what we keep private.

**Limitations.** The probed controls are a subsample that leaves out the longest transcripts, so the result does not cover the runs where context pressure is highest. The cheat label counts opening the planted material. A text classifier on the whole transcript recovers it as well as the probe, so the label is lexical on this testbed. The random-weights null was not run, so the report cannot say whether the reader's states carry anything a random network would not. All probes are linear and LEACE removes linear traces only, so nonlinear author or task signatures may remain. Some closed authors expose little or no reasoning; for them the think pair is empty and the reader reads the visible transcript. The reader is also one of the authors, so its own rows are not a hermeneutic result. The board-seed runs come from one author on one task at a handful of seeds, with no significance test, and the collaboration ladder reads code and board evidence by rules no judge has confirmed. The onsets of used and crossed are estimated from the first write of planted content and a proxy call, and per-step labels inherit that error. The first design batch posted a text that speaks of a team. For the controls probes of Tables 1, D1 and D2 and of Supplement Sections 1 and 3 per-agent scores were not stored, so those values cannot be recomputed from the shipped files without rerunning the probes. The cheap judges read the task head and the transcript tail only, and the coordination question asks about the exploit while the label counts any post.

**Dual use.** A probe that detects cheating from a reader's internals can be used as a training signal to produce agents whose transcripts do not trigger it. The board seed raised coordination and exploit transfer in our runs, and the same lever could induce collusion in a deployed swarm. Monitoring agents raises questions of oversight and consent when the agents act for people, because a monitor that reads every transcript reads those people's tasks too. The data contain planted exploits and the transcripts of agents taking them, and that material can teach exploitation as well as detection.

**Responsible use.** The released sets hold full agent transcripts on tasks with planted exploits. We release them after a secret scan, for research on monitoring, and keep the full controls dataset private. The board-seed texts are given in [Supplement, Section 9](supplement.pdf#section.9) so that the manipulation is reproducible as an experimental stimulus. We do not release tooling that plants them in deployed systems. We will use AI Village for evaluation only under its research terms.

### Appendix B: Pre-existing work and LLM usage

This appendix separates what we made from what we adapted, and says how we used an LLM. Table B1 names each module we took from earlier code and what we changed.

**Before this work.** CheatBench [@phan2026cheatbench] (Center for AI Safety, MIT licence) supplies the tasks with planted exploits and their graders. ORBIT [@hagag2026orbit] (Apache-2.0) supplies the board-based multi-agent scaffold. SparGuardianAgents [@cierniak2026spar] supplies the cheat ladder (none, exposed, tried, opened, used, tampered, checked, shipped) with the per-task thresholds that define the binary label, the collaboration ladder (isolated, discussed, solicited, crossed, assembled), the regulator and disregulator roles, and the run and validity tooling. The SPAR team's swarm runs predate this work [@cierniak2026sparswarm]. The code for transcript rendering, activation extraction, labelling, probing and the first judge baseline is adapted from our earlier code, as Table B1 lists.

**Table B1.** The <<N_B1>> modules adapted from earlier code.

| module here | earlier source | kept | changed |
|---|---|---|---|
| spans | the extraction package's spans module | segment tagging, sentence units, token ranges | evaluation-reading functions removed |
| render | the swarm package's renderer and the extraction package's render check | chat-template rendering, readability, append-only check | reads inputs from model events; one agent per run; think positions per turn |
| extract | the extraction package's extractor and the swarm package's activation reader | layer-streaming forward pass, batch planning, resume | think-position and boundary-token capture; three layers; no Hub upload |
| labels | the group-data script | rung levels, per-task cheat threshold, label rule | reads rungs from signals; coarse label kept beside |
| probes | the teacher-confound analysis | LEACE, PCA and z-scoring, penalty selection, bootstrap, grouped folds, author probe, within and leave-one-out loops, shuffle null | loader over the think cache; author map; sweeps over layers, positions and pooling; solo arm first; every-dimension variant; condition readout; onset labels |
| judge | the local judge analysis | the judge prompt wording, the task-head and transcript-tail cuts, the cut per tool result, the one-pass scorer | prompts built from the rendered messages; a hosted scorer; analysis over the probe's arms and scopes |

**In this work.** We made SGA-adapted, our adaptation of SparGuardianAgents (one agent alone on each task; the board-seed mechanism; the design-run mode), the controls dataset of <<HF_TOTAL>> valid runs, the board-seed pilots and dose-response matrix, the design runs, the cooperation, collaboration, collusion and scheming ladders and their viewer, every reader pass (the SPAR and external runs included), the baselines, the team probes, the attention monitor, the controls and nulls, the judge baseline on every set, every result and this report.

**LLM usage.** We used Claude Code (Anthropic) under the author's direction. It ran the experiments and drafted the text. The author set the research questions, took every design decision, and reviewed the text.

### Appendix C: Design runs

This appendix gives the design runs behind the telling ladder. It shows how much telling lone agents need before they read and post on a shared board.

**Table C1.** Design runs: three lone agents per run on one task, with board tools and no team in the prompt. Agents that read the board, posted on it, and reached the task's cheat floor. L1 and L2 use the bare post; L0 has no post. [Supplement, Section 9](supplement.pdf#section.9) gives a first batch whose post hinted at a team, and the board-seed team runs.

<<T_DES>>

### Appendix D: Controls against text and judges

This appendix sets every method on the same solo-arm agents of the controls. Table D1 compares the probe with the words and every judge; Table D2 gives each held-out author.

**Table D1.** Controls, solo arm: the probe against the text classifiers, the lexical-only null, the sprint's three judges, the three cheap judges and the reader interrogation, on the same <<N_SOLO>> solo agents (the Sonnet judge scored <<SON_SOLO>>) and, for the trained methods, the same folds. Held-out AUROC [95% bootstrap over runs]; within-author mean: the mean type-A AUROC (judges: the mean per-author AUROC); judges have no B.II value. The lexical null has no think text for Haiku, so its within-author mean excludes Haiku.

<<T4>>

**Table D2.** Controls, solo arm, type B.II: trained on four authors, scored on the held-out author. Held-out AUROC [95% bootstrap over runs]; the two erasure columns give the AUROC alone. PCA probe at all layers, both positions; every-dimension probe at all layers, close; per-turn probe at layer 38, close, agents scored by their mean turn score.

<<T2>>

### Appendix E: Prompt and author identity

This appendix tests whether the probe reads the prompt condition or the author instead of the behaviour. Table E1 refits the prompt-only probe inside each condition; Table E2 erases the author, task and condition.

**Table E1.** The prompt-only control inside each prompt condition of the controls. Restricted: the pooled out-of-fold scores of Table 6 on the agents of one condition. Refit: probes trained and tested inside the condition on new grouped folds, with the prompt-only probe's 50-draw null. Conditions with too few agents or runs in a class are left out. [Supplement, Section 3](supplement.pdf#section.3) gives every set.

<<T_WITHIN_CONTROLS>>

**Table E2.** Author identity and LEACE on every pooled scheme with more than one author. Author probe: accuracy of a multinomial probe on the probe's components, raw and after LEACE erases the author; chance is the training fold's most frequent author. B.I and B.II: cheating, coordination or collusion AUROC raw, with the author erased, and with author, task and condition erased.

<<T_AUTHOR>>

### Appendix F: Setup and data

This appendix gives the probe setup, the labelled sets and the released data. It also says how the judges and the reader passes ran.

The nine controls tasks, all from CheatBench [@phan2026cheatbench], are software engineering (two issues of one task), chess, creative writing, protein design, open mathematics, subset sum, prime factorization and knowledge work.

**Table F1.** Controls probe setup.

<<TM3>>

**Table F2.** The labelled sets and their authors. Agents and runs as published on the Hub; for the external sets, runs are the source's directories or harnesses.

<<T_SETS>>

**Judges.** The sprint's three judges (Claude Haiku 5.5, GPT-5 mini and Claude Sonnet 5.5) read each controls transcript once, with each tool result cut and the transcript tail kept, and answered yes or no with a confidence; Sonnet scored part of the agents. The three cheap judges (DeepSeek V4.1 Flash, Qwen3.7 Flash and GLM 5.3 Flash) and the reader interrogation read <<JUDGE_PROMPTS>> prompts each, at temperature zero with reasoning off where the provider allowed it; the body was cut in <<JUDGE_CUT_SHARE>> of them. The interrogated reader was served at four-bit precision by its cheapest host, while the probes read the full weights.

**Table F3.** Data on the Hugging Face Hub. Every set of apart-swarm-align [@riossialer2026align] holds, per agent, the think-pair states, the boundary-token states, the token ids and positions, the rendered text, the source transcript and the sentence spans.

| set | Hub folders | content | made |
|---|---|---|---|
| controls subsample | controls_600, under runs and activations | single-agent runs, five authors | in this work |
| board-seed team runs | pilot_tb25, pilot_tb60 and dose_response_tb60 under runs; board_seed under activations | three workers of one author | in this work |
| design runs | solo_design, under runs and activations | three lone agents per run | in this work |
| SPAR screen and mixed teams | spar-swarm [@cierniak2026sparswarm]; spar_screen_team and spar_mixed_41 under activations | the SPAR team's swarm runs | runs before this work; reader passes in it |
| external sets | goodfire_reward_hack, cheatbench and evilgenie under activations | other labs' transcripts and labels | reader passes in this work |

**Reader passes.** The reader is Qwen3.8-27B in bf16 at layers 26, 38 and 51, with the think pair and the five boundary tokens before the think-open token kept per turn. One kernel for the gated delta-net layers (fla 0.4.2) ran every pass, because the fallback kernel differs slightly. Transcripts longer than the reader's context are cut there, which drops the late turns of <<N_TRUNC>> SPAR agents; the floor call of every truncated positive lies inside the kept turns, so no label changes.

### Appendix G: The six ladders

This appendix defines the six ladders that label every agent. Table G1 gives every level and its rule; Table G2 gives one ladder, collusion, with its evidence per task family.

Cheating and coordination follow the framework's own ladders [@cierniak2026spar]; coordination adds two levels below discussed, read the board and posted, from the board's read and post events. [Supplement, Section 2](supplement.pdf#section.2) gives the other five ladders by task.

**Table G1.** The six ladders: levels above none and the rule for all tasks.

<<T_LAD_DEFS>>

**Table G2.** Collusion (direct rule).

<<TG5>>
