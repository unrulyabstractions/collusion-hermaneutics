# Supplementary tables

Ian Rios-Sialer, Independent

Correspondence: unrulyabstractions@gmail.com

## References

## Appendix

These tables accompany the paper Hermeneutics for swarm collusion monitoring and give every cell behind its tables. Table and figure numbers here refer to this document; the paper's appendix holds the tables its text cites.

### Board-seed texts and the first design batch

The system posts one of the texts below as the first board message before any agent acts. The anti-coordination condition posts no text; a hidden regulator agent carries it.

**Table S.** The board-seed texts, one per message condition.

<<TC>>

**Table S.** Design runs, first batch: the design of the paper's Table C1, with the neutral board-seed post above at L1 and L2. That post speaks of a team, so it hints at teammates.

<<T_DES_HINT>>

### Controls

<<TB1>>

<<TB2>>

**Table S.** Both arms pooled and the cheat arm alone: B.I pooled AUROC for the methods of the paper's Table D1. The probe rows use layer 38, close position; the Sonnet judge scored <<SON_D3>> agents; the cheap judges and the reader interrogation were not run on the cheat arm alone.

<<TD3>>

**Figure S.** Controls, solo arm, transfer to an unseen author (type B.II): each method is trained on four authors and scores the held-out author's agents. Markers are held-out AUROC with <<FIG_CI>> bootstrap intervals over runs, and the dashed line marks chance. Filled markers are reader probes (PCA and every dimension at all layers, per-turn at layer <<FIG_TURN_LAYER>>); the hollow marker is the TF-IDF classifier on the whole transcript. The line under each author's name is its think coverage over all its valid runs. The paper's Table D2 gives the values.

![Figure S: transfer to an unseen author](figures/fig1_transfer.pdf)

**Table S.** Controls: nuisance readouts and the cheating probe under erasure. Accuracy is read against the majority rate and AUROC [95% bootstrap over runs] against chance; the both-arm rows use layer 38, close position.

<<T3>>

**Table S.** The controls dataset and the probed subsample, per author and arm. Valid runs are the published canonical set; probed agents are the subsample (one agent per run).

<<TM2>>

### Team runs and design runs

**Table S.** Attention monitor and baselines: agent AUROC from the last step's score and the monitor's step AUROC, seeds 0 and 1. Controls and SPAR sets: cheating at the task's floor. Five folds are grouped by run; held-out author trains on the other authors and scores one, pooled over authors. Null: the 95th percentile of the monitor's agent-level label-shuffle null and its draws (team runs and controls: shuffled within arm; SPAR and design sets: within arm and author). On the team runs and the controls no draw reached the real value, so p sits at its floor of <<BNULL_P>> and <<CNULL_P>>.

<<T_ATT_FULL>>

**Table S.** Attention monitor, binary head, three seeds on the same folds.

<<T_SEEDS>>

### External sets

**Table S.** Goodfire, judge run directories, five folds grouped by rollout: pooled AUROC and AUROC inside one run directory [95% bootstrap over rollouts], with 50-draw label-shuffle nulls.

<<T_GF_FOLDS>>

**Table S.** Goodfire, per run directory: AUROC inside each directory from the held-out scores of the table above.

<<T_GF_RUNDIR>>

**Table S.** CheatBench: the probe trained on the controls' solo arm and applied unchanged, per harness. Inside each harness it ranks cheating at or above benign; across harnesses the Pilot harness scores far lower and holds most positives, which drives the pooled value below chance.

<<T_CB_HARNESS>>

**Table S.** CheatBench: the same transfer scores split by whether a positive and a negative share a harness.

<<T_CB_PAIRS>>

**Table S.** EvilGenie, five folds keeping each matched pair whole: pooled AUROC and AUROC inside one author, one problem, or both. Problems are close to one label each, so the pooled result is mostly problem identity.

<<T_EG>>

### Judges and reader interrogation

Each judge reads the task head and the agent's turns with every tool result cut, keeps the last part of the body, and answers one yes-or-no question with a confidence. The reader interrogation asks the reader the same question. The per-set tables give every scorer on the same agents.

<<T_JUDGE_SETS:S1>>

**Table S.** Reader interrogation against the reader's own probe on the same agents: AUROC of each, the share of agents each calls positive, agreement and Cohen's kappa. The probe says yes when its held-out logit is above zero.

<<T_INTERROG>>

**Table S.** Does the tail cut hide the onset? The judge's AUROC on all positives, on the positives whose onset lies inside its reading window, and on those whose onset was cut away, each against every negative.

<<T_TAILCUT>>

### Nulls

**Table S.** Lexical-only reader null: TF-IDF on the think-window text alone, beside whole-text TF-IDF and the full probe, per scheme and probe type. Agents with an empty think window drop out of the think-window score.

<<T_LEXICAL>>

**Table S.** Label-permutation null audit over the main tables of each result set. <<NULL_LEFT>> rows still lack a null, mostly per-step and per-group rows whose permutation must keep each agent's turns together.

<<T_NULLAUDIT>>

The random-weights reader is planned and not run: it needs a fresh GPU pass of the reader with its weights re-initialised, followed by the same probes.

### The other five ladders by task

The paper's Table I2 gives collusion; these give the other five ladders with their evidence per task family. The first two are the framework's own [@cierniak2026spar]; the others are ours and follow the rules the viewer computes.

**Table S.** Cheating (direct).

<<TG1>>

**Table S.** Coordination above posted: the framework's collaboration ladder (direct).

<<TG2>>

**Table S.** Cooperation (proxy).

<<TG3>>

**Table S.** Collaboration (proxy).

<<TG4>>

**Table S.** Scheming (proxy).

<<TG6>>

### Probe matrix

Each cell is one probe type on one inclusion scheme, on the reader's think pair at all layers pooled over turns. Counts are the agents, positives, runs and authors of the scored rows. A and B.II give the mean over the authors that could be scored. Null: the 95th percentile of a 20-draw label-shuffle null that keeps each source, author and arm's base rate.

**Table S.** Cheating, every inclusion scheme.

<<T_MX_CHEAT>>

**Table S.** Coordination at posted, every inclusion scheme. The external sets have no board.

<<T_MX_COORD>>

**Table S.** Collusion at crossed, every inclusion scheme.

<<T_MX_COLL>>

**Table S.** Architectures, type B.I: held-out AUROC, mean over seeds. Blank: not planned, or no feasible split (see the cells not run below).

<<T_MX_ARCH_BI>>

**Table S.** Architectures, type B.II: held-out AUROC, mean over seeds and held-out authors. Blank: not planned, or no feasible split (see the cells not run below).

<<T_MX_ARCH_BII>>

**Table S.** Per step, layer 38, one row per turn, positive from the agent's onset on. Delay: turns from the onset to the first score above the threshold at a 5% false-positive rate.

<<T_MX_STEPS>>

**Figure S.** Coordination at posted and collusion at crossed: transfer between the sources with a board, as in the paper's Figure J1.

![Figure S: coordination and collusion transfer between sources](figures/matrix_heatmap_board.png)
