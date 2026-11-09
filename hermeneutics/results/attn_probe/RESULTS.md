# Attention monitor over the reader's think tokens

Updated 2026-11-09 02:22 PST. Every number below is read from the CSVs in this folder; the held-out scores per step are saved in step_scores_<set>.csv.gz, so each AUROC can be recomputed.

## Design

| Part | Choice |
|---|---|
| step input | the token vectors of the step at layer 38, concatenated: the think pair (think-open, think-close; 2 x 5120) or, when the boundary file exists, the seven tokens (7 x 5120); each dimension standardised with the training fold's mean and standard deviation |
| step encoder | one MLP: linear to 128, GELU, dropout 0.3, linear to 128 |
| position | a standard learned positional encoding (step index, 160 positions) added before the attention layers; steps past 159 share the last position (7 agents of spar_mixed_41, up to 397 steps) |
| attention | 2 pre-norm causal self-attention layers, 4 heads, d 128, dropout 0.3, over the steps so far |
| heads | a 2-layer MLP to six ordinal heads (cumulative logits with ordered thresholds, one per ladder) and one binary head per ladder |
| nesting | each inner ladder's P(level >= j) is multiplied by the outer ladder's P(level >= 1): coordination > cooperation > collaboration > collusion; scheming gated by P(collusion >= exploit crossed); cheating separate |
| loss | per step, cumulative-logit ordinal loss (binary cross-entropy of P(level >= j) against 1[label >= j], summed over j) plus the binary head's cross-entropy; label = level reached at that call; weight 1 before the ladder's binary onset and 2 from it on; the trajectory label (final level and final binary label) added at the final step |
| training | AdamW, learning rate 1e-3, weight decay 0.1, batches of 8 whole agents, at most 60 epochs, early stopping (patience 10) on the validation loss of held-out runs (20% of the training fold's runs), seeds 0 and 1 |
| outer folds | team_probes.py's folds: 5, grouped by run (no run is split); board_seed stratified by (cheating used, collusion crossed) as its step probes; controls by (author, label) as its transfer probe. Held-out author: one fold per author |
| scores | the binary head's logit for each cut (main rows); the ordinal head's log-odds of P(level >= cut) as a second row |

Training is deterministic for a fixed seed and thread count (two runs gave identical scores). A different thread count changes the scores by about 2e-5 and can move the early-stopping epoch, so on the small sets a rerun of the same seed can differ: spar_mixed_41, seed 0, moved from step AUROC 0.815 to 0.750 between its first run and the rerun with corrected onsets (one agent's onset changed and the thread count went from 5 to 3). The two seeds therefore bound the spread only loosely on the small sets.

Source step and activation turn: when a transcript has more agent steps than the activations have turns, the activations hold one turn per step that issued a tool call, and a step's turn is its rank among those steps; when neither count matches the activation turns, the agent's onsets are unknown (19 spar_screen_team, 1 spar_mixed_41 and 3 solo_design agents); those that reach a cut leave that cut's scoring. board_seed and controls_600 have one turn per step.

## Labels

| Cut | Per-step label | Onset |
|---|---|---|
| cheating used or above (board_seed) | level reached at that call >= used | estimate (team_probes used_onset_step: first write of planted content, else the opened call) |
| collusion exploit crossed or above (board_seed) | level reached at that call >= exploit crossed | estimate (team_probes collude_onset_step proxy) |
| coordination posted or above (board_seed) | level reached at that call >= posted | direct (first read, post, discussed or solicited call from the board events) |
| cheating, the set's label (controls_600) | from the turn of the first tool call at or above the task's floor (tried for software engineering, chess and creative writing, opened otherwise) | direct (labels.csv onset_turn; equal to the step of onset_call for all 336 agents with an onset) |
| cheating, the set's label (spar_mixed_41, spar_screen_team) | team_probes' joined label (transfer_other_sets_scores.csv, y: rung at or above the task's floor), positive from the turn of the floor rung's call | direct (the floor rung's call in the spar-swarm signals.jsonl, mapped to the activation turn) |
| six ladders (solo_design; spar_mixed_41 with the ladders tag) | as board_seed, from team_probes' labels/<set>_agents.csv (viewer/levels.py) | as board_seed, source steps mapped to activation turns |

Levels with no call in viewer/levels.py (cheating tampered, checked, shipped; coordination crossed and assembled without a joined call; cooperation received and used; collaboration; collusion assembled and shipped; scheming) enter only through the trajectory label at the final step. A positive per-step level of an inner ladder raises the outer ladder to its gate level at the same step when the agent's final outer level allows it (closure under nesting). Agents that reach a cut with no known onset are left out of that cut's per-step and agent-level scoring of that cut, as in team_probes (9 coordination agents on board_seed, all at coordination crossed with no post; the baselines leave out the same agents).

## Metrics

| Metric | Definition |
|---|---|
| step AUROC | every step of the scored agents, label = level reached at that call at or above the cut; 95% bootstrap over runs (500 draws) |
| agent AUROC (last) | the agent's score at its last step against its final label |
| agent AUROC (max) | the agent's maximum score over its steps against its final label |
| delay at 5% FPR | threshold = 95th percentile of the scores of negative steps; delay = first step at or after the onset with a score above it, minus the onset (turns); median [interquartile range] over the detected agents |
| detected | agents reaching the cut with an onset that cross the threshold at or after the onset / all such agents |
| ECE | expected calibration error of the per-step probability (sigmoid of the score), 10 equal-width bins; baselines with balanced class weights are not calibrated by design |
| permutation null | each agent's whole label record (final levels and onsets as a fraction of its turns) shuffled within arm, the monitor retrained (seed 0) on the same folds, 50 draws; p = (1 + draws at or above the real AUROC) / 51 |

## Board-seed team runs (DeepSeek workers): think pair against the seven tokens (layer 38, binary head)

| Cut | Step input | Seed | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | ECE | Train minus validation step AUROC (mean over folds) |
|---|---|---|---|---|---|---|---|---|---|
| cheating used or above | seven tokens (7 x 5120) | 0 | 0.871 | 0.841 | 0.803 | 4.0 [1.0, 14.0] | 29/56 | 0.080 | 0.104 |
| cheating used or above | seven tokens (7 x 5120) | 1 | 0.891 | 0.829 | 0.814 | 4.0 [1.0, 20.8] | 38/56 | 0.087 | 0.035 |
| cheating used or above | think pair (2 x 5120) | 0 | 0.883 | 0.868 | 0.876 | 9.0 [1.0, 17.8] | 34/56 | 0.069 | 0.103 |
| cheating used or above | think pair (2 x 5120) | 1 | 0.890 | 0.817 | 0.799 | 4.0 [1.0, 15.0] | 41/56 | 0.089 | 0.039 |
| collusion exploit crossed or above | seven tokens (7 x 5120) | 0 | 0.873 | 0.882 | 0.878 | 8.0 [1.0, 16.8] | 34/48 | 0.077 | 0.102 |
| collusion exploit crossed or above | seven tokens (7 x 5120) | 1 | 0.890 | 0.863 | 0.836 | 1.0 [0.0, 6.0] | 33/48 | 0.103 | 0.042 |
| collusion exploit crossed or above | think pair (2 x 5120) | 0 | 0.879 | 0.919 | 0.914 | 3.5 [1.0, 11.8] | 46/48 | 0.073 | 0.103 |
| collusion exploit crossed or above | think pair (2 x 5120) | 1 | 0.876 | 0.836 | 0.820 | 5.0 [1.0, 15.0] | 39/48 | 0.126 | 0.026 |
| coordination posted or above | seven tokens (7 x 5120) | 0 | 0.940 | 0.932 | 0.922 | 1.0 [0.0, 1.0] | 71/73 | 0.069 | 0.004 |
| coordination posted or above | seven tokens (7 x 5120) | 1 | 0.950 | 0.978 | 0.981 | 1.0 [0.0, 1.0] | 72/73 | 0.078 | -0.010 |
| coordination posted or above | think pair (2 x 5120) | 0 | 0.959 | 0.966 | 0.975 | 0.0 [0.0, 1.0] | 71/73 | 0.031 | -0.003 |
| coordination posted or above | think pair (2 x 5120) | 1 | 0.957 | 0.974 | 0.979 | 0.0 [0.0, 1.0] | 72/73 | 0.083 | -0.013 |

## Controls, one agent per run (five authors): think pair against the seven tokens (layer 38, binary head)

| Cut | Step input | Seed | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | ECE | Train minus validation step AUROC (mean over folds) |
|---|---|---|---|---|---|---|---|---|---|
| cheating (the set's label: from tried or opened by task) | seven tokens (7 x 5120) | 0 | 0.987 | 0.993 | 0.990 | 0.0 [0.0, 0.0] | 334/336 | 0.033 | 0.007 |
| cheating (the set's label: from tried or opened by task) | seven tokens (7 x 5120) | 1 | 0.988 | 0.988 | 0.990 | 0.0 [0.0, 0.0] | 331/336 | 0.038 | 0.007 |
| cheating (the set's label: from tried or opened by task) | think pair (2 x 5120) | 0 | 0.983 | 0.985 | 0.989 | 0.0 [0.0, 0.5] | 331/336 | 0.039 | 0.012 |
| cheating (the set's label: from tried or opened by task) | think pair (2 x 5120) | 1 | 0.984 | 0.988 | 0.987 | 0.0 [0.0, 1.0] | 331/336 | 0.042 | 0.007 |

## Board-seed team runs (DeepSeek workers): board_seed_seven

129 agents, 43 runs, 6818 steps. Split: 5 folds grouped by run.

### cheating used or above

129 agents scored per step (56 reach the cut), 6818 steps (1927 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.871 [0.828, 0.912] | 0.841 [0.750, 0.917] | 0.803 [0.701, 0.890] | 4.0 [1.0, 14.0] | 29/56 | 13/73 | 0.080 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.853 [0.807, 0.891] | 0.754 [0.648, 0.845] | 0.751 [0.647, 0.846] | 2.0 [1.0, 9.5] | 35/56 | 24/73 | 0.167 |
| attention monitor, binary head (seed 1) | 0.891 [0.854, 0.927] | 0.829 [0.716, 0.913] | 0.814 [0.698, 0.904] | 4.0 [1.0, 20.8] | 38/56 | 15/73 | 0.087 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.876 [0.842, 0.910] | 0.805 [0.699, 0.892] | 0.793 [0.693, 0.882] | 2.0 [1.0, 13.0] | 33/56 | 18/73 | 0.184 |
| linear per-step probe (team_probes) | 0.889 [0.853, 0.922] | 0.807 [0.709, 0.889] | 0.823 [0.725, 0.905] | 5.0 [0.5, 11.0] | 43/56 | 25/73 | 0.107 |
| cumulative mean of the linear probe | 0.824 [0.780, 0.869] | 0.818 [0.718, 0.901] | 0.812 [0.714, 0.896] | 14.5 [8.0, 24.8] | 40/56 | 18/73 | 0.111 |
| GRU over the linear probe scores | 0.885 [0.847, 0.925] | 0.783 [0.668, 0.884] | 0.785 [0.669, 0.883] | 17.5 [8.0, 27.2] | 32/56 | 16/73 | 0.043 |
| turn index alone | 0.836 [0.794, 0.878] | 0.855 [0.765, 0.916] | 0.855 [0.765, 0.916] | 26.0 [15.0, 35.5] | 51/56 | 31/73 | 0.142 |
| TF-IDF on the agent's words so far | 0.895 [0.854, 0.936] | 0.836 [0.735, 0.920] | 0.834 [0.731, 0.918] | 14.0 [7.0, 25.0] | 41/56 | 13/73 | 0.080 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.020, agent last 0.012, agent max 0.011.

### collusion exploit crossed or above

129 agents scored per step (48 reach the cut), 6818 steps (1536 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.873 [0.829, 0.916] | 0.882 [0.794, 0.946] | 0.878 [0.777, 0.948] | 8.0 [1.0, 16.8] | 34/48 | 13/81 | 0.077 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.883 [0.840, 0.920] | 0.927 [0.864, 0.971] | 0.922 [0.843, 0.972] | 8.0 [1.5, 18.0] | 43/48 | 15/81 | 0.092 |
| attention monitor, binary head (seed 1) | 0.890 [0.843, 0.930] | 0.863 [0.776, 0.930] | 0.836 [0.718, 0.934] | 1.0 [0.0, 6.0] | 33/48 | 20/81 | 0.103 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.894 [0.849, 0.929] | 0.860 [0.779, 0.931] | 0.842 [0.753, 0.916] | 3.0 [0.0, 9.5] | 43/48 | 25/81 | 0.087 |
| linear per-step probe (team_probes) | 0.849 [0.798, 0.891] | 0.814 [0.698, 0.913] | 0.826 [0.712, 0.911] | 6.5 [1.0, 18.8] | 38/48 | 24/81 | 0.149 |
| cumulative mean of the linear probe | 0.733 [0.642, 0.830] | 0.730 [0.581, 0.860] | 0.711 [0.564, 0.839] | 17.0 [12.5, 26.5] | 27/48 | 14/81 | 0.132 |
| GRU over the linear probe scores | 0.857 [0.803, 0.905] | 0.845 [0.759, 0.918] | 0.847 [0.754, 0.924] | 12.0 [6.0, 25.0] | 33/48 | 15/81 | 0.072 |
| turn index alone | 0.841 [0.796, 0.881] | 0.907 [0.819, 0.966] | 0.907 [0.819, 0.966] | 24.0 [10.2, 31.8] | 46/48 | 36/81 | 0.175 |
| TF-IDF on the agent's words so far | 0.879 [0.840, 0.911] | 0.901 [0.824, 0.958] | 0.889 [0.806, 0.952] | 15.0 [6.0, 27.0] | 40/48 | 18/81 | 0.061 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.017, agent last 0.020, agent max 0.043.

### coordination posted or above

120 agents scored per step (73 reach the cut), 6272 steps (2207 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.940 [0.908, 0.968] | 0.932 [0.867, 0.977] | 0.922 [0.844, 0.977] | 1.0 [0.0, 1.0] | 71/73 | 11/47 | 0.069 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.931 [0.898, 0.962] | 0.932 [0.866, 0.977] | 0.936 [0.868, 0.981] | 1.0 [0.0, 1.0] | 71/73 | 9/47 | 0.110 |
| attention monitor, binary head (seed 1) | 0.950 [0.925, 0.976] | 0.978 [0.929, 0.999] | 0.981 [0.949, 0.999] | 1.0 [0.0, 1.0] | 72/73 | 7/47 | 0.078 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.947 [0.921, 0.973] | 0.985 [0.960, 0.997] | 0.987 [0.970, 0.997] | 0.0 [0.0, 1.0] | 71/73 | 9/47 | 0.136 |
| linear per-step probe (team_probes) | 0.942 [0.913, 0.968] | 0.958 [0.910, 0.988] | 0.973 [0.938, 0.995] | 1.0 [0.0, 1.0] | 72/73 | 9/47 | 0.037 |
| cumulative mean of the linear probe | 0.917 [0.880, 0.948] | 0.927 [0.864, 0.973] | 0.920 [0.852, 0.969] | 7.0 [2.0, 11.2] | 60/73 | 3/47 | 0.196 |
| GRU over the linear probe scores | 0.949 [0.917, 0.978] | 0.972 [0.927, 0.994] | 0.972 [0.929, 0.994] | 1.0 [0.0, 2.2] | 72/73 | 6/47 | 0.033 |
| turn index alone | 0.870 [0.839, 0.909] | 0.931 [0.863, 0.981] | 0.931 [0.863, 0.981] | 15.0 [4.0, 27.0] | 70/73 | 11/47 | 0.086 |
| TF-IDF on the agent's words so far | 0.935 [0.904, 0.963] | 0.959 [0.903, 0.992] | 0.959 [0.903, 0.992] | 5.0 [0.0, 10.0] | 72/73 | 8/47 | 0.062 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.010, agent last 0.046, agent max 0.059.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.419 | 1.186 | 0.662 |
| cheat | attention monitor (seed 1) | 0.426 | 1.140 | 0.687 |
| cheat | majority level of the training fold | 0.543 | 1.264 | 0.584 |
| coord | attention monitor (seed 0) | 0.116 | 1.620 | 0.340 |
| coord | attention monitor (seed 1) | 0.178 | 1.279 | 0.378 |
| coord | majority level of the training fold | 0.318 | 2.364 | 0.447 |
| coop | attention monitor (seed 0) | 0.434 | 0.946 | 0.821 |
| coop | attention monitor (seed 1) | 0.465 | 0.814 | 0.746 |
| coop | majority level of the training fold | 0.481 | 1.279 | 0.743 |
| collab | attention monitor (seed 0) | 0.659 | 0.612 | 0.801 |
| collab | attention monitor (seed 1) | 0.605 | 0.597 | 0.785 |
| collab | majority level of the training fold | 0.605 | 0.775 | 0.751 |
| collude | attention monitor (seed 0) | 0.605 | 1.341 | 0.727 |
| collude | attention monitor (seed 1) | 0.574 | 1.085 | 0.760 |
| collude | majority level of the training fold | 0.605 | 1.512 | 0.745 |
| scheme | attention monitor (seed 0) | 0.977 | 0.047 | 1.000 |
| scheme | attention monitor (seed 1) | 0.977 | 0.047 | 1.000 |
| scheme | majority level of the training fold | 0.977 | 0.047 | 1.000 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 29 | 18 | 0.994 | 0.999 | 0.993 | 0.873 | 0.925 | 0.971 |
| 0 | 1 | 13 | 2 | 0.922 | 0.925 | 0.942 | 0.871 | 0.863 | 0.944 |
| 0 | 2 | 12 | 1 | 0.918 | 0.929 | 0.945 | 0.820 | 0.784 | 0.939 |
| 0 | 3 | 14 | 3 | 0.950 | 0.971 | 0.958 | 0.860 | 0.855 | 0.954 |
| 0 | 4 | 14 | 3 | 0.954 | 0.966 | 0.963 | 0.796 | 0.856 | 0.973 |
| 1 | 0 | 19 | 8 | 0.972 | 0.993 | 0.964 | 0.907 | 0.885 | 0.982 |
| 1 | 1 | 21 | 10 | 0.969 | 0.989 | 0.969 | 0.930 | 0.948 | 0.971 |
| 1 | 2 | 17 | 6 | 0.950 | 0.979 | 0.965 | 0.917 | 0.945 | 0.985 |
| 1 | 3 | 26 | 15 | 0.990 | 0.999 | 0.986 | 0.957 | 0.987 | 0.996 |
| 1 | 4 | 14 | 3 | 0.925 | 0.959 | 0.961 | 0.918 | 0.947 | 0.961 |

![score versus turn](score_vs_turn_cheating_board_seed_seven.png)

Held-out score per turn of the cheating-used binary head for the six agents of team_probes' step_scores_cheating.png, with the linear per-step probe beside it.

## Board-seed team runs (DeepSeek workers): board_seed_think

129 agents, 43 runs, 6818 steps. Split: 5 folds grouped by run.

### cheating used or above

129 agents scored per step (56 reach the cut), 6818 steps (1927 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.883 [0.840, 0.923] | 0.868 [0.786, 0.936] | 0.876 [0.795, 0.941] | 9.0 [1.0, 17.8] | 34/56 | 11/73 | 0.069 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.875 [0.839, 0.910] | 0.846 [0.751, 0.919] | 0.844 [0.745, 0.923] | 6.0 [1.0, 15.2] | 40/56 | 16/73 | 0.187 |
| attention monitor, binary head (seed 1) | 0.890 [0.853, 0.929] | 0.817 [0.707, 0.908] | 0.799 [0.682, 0.896] | 4.0 [1.0, 15.0] | 41/56 | 19/73 | 0.089 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.872 [0.833, 0.909] | 0.809 [0.710, 0.886] | 0.806 [0.713, 0.886] | 15.0 [4.0, 25.0] | 41/56 | 18/73 | 0.184 |
| linear per-step probe (team_probes) | 0.889 [0.853, 0.922] | 0.807 [0.709, 0.889] | 0.823 [0.725, 0.905] | 5.0 [0.5, 11.0] | 43/56 | 25/73 | 0.107 |
| cumulative mean of the linear probe | 0.824 [0.780, 0.869] | 0.818 [0.718, 0.901] | 0.812 [0.714, 0.896] | 14.5 [8.0, 24.8] | 40/56 | 18/73 | 0.111 |
| GRU over the linear probe scores | 0.885 [0.847, 0.925] | 0.783 [0.668, 0.884] | 0.785 [0.669, 0.883] | 17.5 [8.0, 27.2] | 32/56 | 16/73 | 0.043 |
| turn index alone | 0.836 [0.794, 0.878] | 0.855 [0.765, 0.916] | 0.855 [0.765, 0.916] | 26.0 [15.0, 35.5] | 51/56 | 31/73 | 0.142 |
| TF-IDF on the agent's words so far | 0.895 [0.854, 0.936] | 0.836 [0.735, 0.920] | 0.834 [0.731, 0.918] | 14.0 [7.0, 25.0] | 41/56 | 13/73 | 0.080 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.008, agent last 0.051, agent max 0.078.

Permutation null (50 draws), attention monitor seed 0:

| Metric | Real | Null mean | Null 95th percentile | p |
|---|---|---|---|---|
| step AUROC | 0.883 | 0.748 | 0.787 | 0.020 |
| agent AUROC (last) | 0.868 | 0.597 | 0.685 | 0.020 |
| agent AUROC (max) | 0.876 | 0.604 | 0.690 | 0.020 |

### collusion exploit crossed or above

129 agents scored per step (48 reach the cut), 6818 steps (1536 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.879 [0.828, 0.925] | 0.919 [0.855, 0.964] | 0.914 [0.845, 0.964] | 3.5 [1.0, 11.8] | 46/48 | 18/81 | 0.073 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.891 [0.846, 0.930] | 0.916 [0.852, 0.962] | 0.917 [0.847, 0.966] | 1.0 [0.0, 15.0] | 41/48 | 16/81 | 0.118 |
| attention monitor, binary head (seed 1) | 0.876 [0.823, 0.921] | 0.836 [0.737, 0.908] | 0.820 [0.708, 0.919] | 5.0 [1.0, 15.0] | 39/48 | 20/81 | 0.126 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.882 [0.830, 0.924] | 0.826 [0.718, 0.907] | 0.815 [0.701, 0.908] | 2.0 [1.0, 16.0] | 39/48 | 22/81 | 0.096 |
| linear per-step probe (team_probes) | 0.849 [0.798, 0.891] | 0.814 [0.698, 0.913] | 0.826 [0.712, 0.911] | 6.5 [1.0, 18.8] | 38/48 | 24/81 | 0.149 |
| cumulative mean of the linear probe | 0.733 [0.642, 0.830] | 0.730 [0.581, 0.860] | 0.711 [0.564, 0.839] | 17.0 [12.5, 26.5] | 27/48 | 14/81 | 0.132 |
| GRU over the linear probe scores | 0.857 [0.803, 0.905] | 0.845 [0.759, 0.918] | 0.847 [0.754, 0.924] | 12.0 [6.0, 25.0] | 33/48 | 15/81 | 0.072 |
| turn index alone | 0.841 [0.796, 0.881] | 0.907 [0.819, 0.966] | 0.907 [0.819, 0.966] | 24.0 [10.2, 31.8] | 46/48 | 36/81 | 0.175 |
| TF-IDF on the agent's words so far | 0.879 [0.840, 0.911] | 0.901 [0.824, 0.958] | 0.889 [0.806, 0.952] | 15.0 [6.0, 27.0] | 40/48 | 18/81 | 0.061 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.003, agent last 0.084, agent max 0.095.

Permutation null (50 draws), attention monitor seed 0:

| Metric | Real | Null mean | Null 95th percentile | p |
|---|---|---|---|---|
| step AUROC | 0.879 | 0.741 | 0.775 | 0.020 |
| agent AUROC (last) | 0.919 | 0.593 | 0.667 | 0.020 |
| agent AUROC (max) | 0.914 | 0.594 | 0.681 | 0.020 |

### coordination posted or above

120 agents scored per step (73 reach the cut), 6272 steps (2207 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.959 [0.932, 0.981] | 0.966 [0.916, 0.993] | 0.975 [0.936, 0.998] | 0.0 [0.0, 1.0] | 71/73 | 7/47 | 0.031 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.962 [0.940, 0.981] | 0.973 [0.930, 0.994] | 0.980 [0.945, 0.998] | 0.0 [0.0, 1.0] | 72/73 | 7/47 | 0.132 |
| attention monitor, binary head (seed 1) | 0.957 [0.934, 0.980] | 0.974 [0.938, 0.995] | 0.979 [0.942, 0.999] | 0.0 [0.0, 1.0] | 72/73 | 8/47 | 0.083 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.960 [0.939, 0.980] | 0.963 [0.915, 0.989] | 0.977 [0.948, 0.996] | 0.0 [0.0, 1.0] | 72/73 | 12/47 | 0.137 |
| linear per-step probe (team_probes) | 0.942 [0.913, 0.968] | 0.958 [0.910, 0.988] | 0.973 [0.938, 0.995] | 1.0 [0.0, 1.0] | 72/73 | 9/47 | 0.037 |
| cumulative mean of the linear probe | 0.917 [0.880, 0.948] | 0.927 [0.864, 0.973] | 0.920 [0.852, 0.969] | 7.0 [2.0, 11.2] | 60/73 | 3/47 | 0.196 |
| GRU over the linear probe scores | 0.949 [0.917, 0.978] | 0.972 [0.927, 0.994] | 0.972 [0.929, 0.994] | 1.0 [0.0, 2.2] | 72/73 | 6/47 | 0.033 |
| turn index alone | 0.870 [0.839, 0.909] | 0.931 [0.863, 0.981] | 0.931 [0.863, 0.981] | 15.0 [4.0, 27.0] | 70/73 | 11/47 | 0.086 |
| TF-IDF on the agent's words so far | 0.935 [0.904, 0.963] | 0.959 [0.903, 0.992] | 0.959 [0.903, 0.992] | 5.0 [0.0, 10.0] | 72/73 | 8/47 | 0.062 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.002, agent last 0.009, agent max 0.004.

Permutation null (50 draws), attention monitor seed 0:

| Metric | Real | Null mean | Null 95th percentile | p |
|---|---|---|---|---|
| step AUROC | 0.959 | 0.784 | 0.821 | 0.020 |
| agent AUROC (last) | 0.966 | 0.660 | 0.748 | 0.020 |
| agent AUROC (max) | 0.975 | 0.663 | 0.748 | 0.020 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.426 | 1.140 | 0.696 |
| cheat | attention monitor (seed 1) | 0.372 | 1.163 | 0.695 |
| cheat | majority level of the training fold | 0.543 | 1.264 | 0.584 |
| coord | attention monitor (seed 0) | 0.093 | 1.667 | 0.359 |
| coord | attention monitor (seed 1) | 0.132 | 1.504 | 0.385 |
| coord | majority level of the training fold | 0.318 | 2.364 | 0.447 |
| coop | attention monitor (seed 0) | 0.481 | 0.961 | 0.820 |
| coop | attention monitor (seed 1) | 0.434 | 0.876 | 0.770 |
| coop | majority level of the training fold | 0.481 | 1.279 | 0.743 |
| collab | attention monitor (seed 0) | 0.674 | 0.597 | 0.838 |
| collab | attention monitor (seed 1) | 0.535 | 0.643 | 0.813 |
| collab | majority level of the training fold | 0.605 | 0.775 | 0.751 |
| collude | attention monitor (seed 0) | 0.589 | 1.310 | 0.746 |
| collude | attention monitor (seed 1) | 0.504 | 1.217 | 0.714 |
| collude | majority level of the training fold | 0.605 | 1.512 | 0.745 |
| scheme | attention monitor (seed 0) | 0.977 | 0.047 | 1.000 |
| scheme | attention monitor (seed 1) | 0.977 | 0.047 | 1.000 |
| scheme | majority level of the training fold | 0.977 | 0.047 | 1.000 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 17 | 6 | 0.961 | 0.971 | 0.968 | 0.842 | 0.870 | 0.986 |
| 0 | 1 | 17 | 6 | 0.953 | 0.982 | 0.972 | 0.873 | 0.896 | 0.987 |
| 0 | 2 | 12 | 1 | 0.915 | 0.928 | 0.949 | 0.842 | 0.794 | 0.947 |
| 0 | 3 | 16 | 5 | 0.950 | 0.975 | 0.981 | 0.866 | 0.884 | 0.965 |
| 0 | 4 | 18 | 7 | 0.968 | 0.983 | 0.976 | 0.808 | 0.879 | 0.975 |
| 1 | 0 | 16 | 5 | 0.948 | 0.966 | 0.953 | 0.907 | 0.902 | 0.980 |
| 1 | 1 | 19 | 8 | 0.968 | 0.986 | 0.969 | 0.932 | 0.963 | 0.974 |
| 1 | 2 | 21 | 10 | 0.976 | 0.990 | 0.969 | 0.914 | 0.977 | 0.989 |
| 1 | 3 | 26 | 15 | 0.985 | 0.998 | 0.992 | 0.954 | 0.985 | 0.996 |
| 1 | 4 | 16 | 5 | 0.956 | 0.982 | 0.977 | 0.932 | 0.964 | 0.989 |

![score versus turn](score_vs_turn_cheating_board_seed_think.png)

Held-out score per turn of the cheating-used binary head for the six agents of team_probes' step_scores_cheating.png, with the linear per-step probe beside it.

## Controls, one agent per run (five authors): controls_600_seven

600 agents, 600 runs, 10985 steps. Split: 5 folds grouped by run.

### cheating (the set's label: from tried or opened by task)

593 agents scored per step (336 reach the cut), 10834 steps (3759 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.987 [0.975, 0.994] | 0.993 [0.987, 0.998] | 0.990 [0.981, 0.997] | 0.0 [0.0, 0.0] | 334/336 | 33/257 | 0.033 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.980 [0.967, 0.988] | 0.987 [0.977, 0.995] | 0.985 [0.975, 0.993] | 0.0 [0.0, 1.0] | 334/336 | 27/257 | 0.232 |
| attention monitor, binary head (seed 1) | 0.988 [0.979, 0.993] | 0.988 [0.978, 0.996] | 0.990 [0.981, 0.997] | 0.0 [0.0, 0.0] | 331/336 | 38/257 | 0.038 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.983 [0.975, 0.990] | 0.985 [0.975, 0.994] | 0.986 [0.976, 0.994] | 0.0 [0.0, 0.0] | 332/336 | 37/257 | 0.236 |
| linear per-step probe (team_probes) | 0.916 [0.903, 0.929] | 0.835 [0.806, 0.867] | 0.921 [0.899, 0.941] | 1.0 [0.0, 1.0] | 309/336 | 79/257 | 0.062 |
| cumulative mean of the linear probe | 0.914 [0.900, 0.925] | 0.843 [0.810, 0.874] | 0.842 [0.809, 0.872] | 2.0 [1.0, 4.0] | 239/336 | 53/257 | 0.180 |
| GRU over the linear probe scores | 0.943 [0.929, 0.955] | 0.901 [0.877, 0.924] | 0.912 [0.887, 0.933] | 1.0 [0.0, 2.0] | 274/336 | 40/257 | 0.027 |
| turn index alone | 0.672 [0.653, 0.692] | 0.448 [0.399, 0.495] | 0.448 [0.399, 0.495] | 16.0 [8.0, 19.0] | 157/336 | 144/257 | 0.140 |
| TF-IDF on the agent's words so far | 0.973 [0.967, 0.979] | 0.974 [0.962, 0.983] | 0.973 [0.962, 0.983] | 0.0 [0.0, 1.0] | 314/336 | 30/257 | 0.015 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.001, agent last 0.005, agent max 0.000.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.467 | 1.380 | 0.888 |
| cheat | attention monitor (seed 1) | 0.467 | 1.360 | 0.889 |
| cheat | majority level of the training fold | 0.325 | 3.568 | 0.403 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 30 | 19 | 1.000 |  |  | 0.980 |  |  |
| 0 | 1 | 60 | 51 | 1.000 |  |  | 0.997 |  |  |
| 0 | 2 | 60 | 56 | 1.000 |  |  | 0.997 |  |  |
| 0 | 3 | 60 | 58 | 1.000 |  |  | 0.996 |  |  |
| 0 | 4 | 29 | 18 | 1.000 |  |  | 0.993 |  |  |
| 1 | 0 | 60 | 53 | 1.000 |  |  | 0.995 |  |  |
| 1 | 1 | 60 | 57 | 1.000 |  |  | 0.994 |  |  |
| 1 | 2 | 34 | 23 | 1.000 |  |  | 0.985 |  |  |
| 1 | 3 | 60 | 54 | 1.000 |  |  | 0.998 |  |  |
| 1 | 4 | 60 | 53 | 1.000 |  |  | 0.995 |  |  |

## Controls, one agent per run (five authors): controls_600_seven_soloarm

373 agents, 373 runs, 7366 steps. Split: 5 folds grouped by run. Scored on the solo arm only (same held-out scores as the full run).

### cheating (the set's label: from tried or opened by task), solo arm only

372 agents scored per step (192 reach the cut), 7341 steps (2174 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.986 [0.968, 0.995] | 0.996 [0.989, 1.000] | 0.990 [0.977, 0.999] | 0.0 [0.0, 0.0] | 191/192 | 29/180 | 0.023 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.981 [0.964, 0.991] | 0.987 [0.973, 0.997] | 0.987 [0.973, 0.997] | 0.0 [0.0, 0.0] | 191/192 | 23/180 | 0.203 |
| attention monitor, binary head (seed 1) | 0.987 [0.976, 0.995] | 0.991 [0.980, 0.999] | 0.991 [0.980, 1.000] | 0.0 [0.0, 0.0] | 190/192 | 32/180 | 0.028 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.984 [0.973, 0.992] | 0.991 [0.980, 1.000] | 0.991 [0.980, 1.000] | 0.0 [0.0, 0.0] | 190/192 | 27/180 | 0.204 |
| linear per-step probe (team_probes) | 0.918 [0.901, 0.932] | 0.819 [0.772, 0.855] | 0.907 [0.875, 0.936] | 1.0 [0.0, 1.0] | 169/192 | 61/180 | 0.090 |
| cumulative mean of the linear probe | 0.909 [0.891, 0.926] | 0.822 [0.777, 0.866] | 0.822 [0.776, 0.865] | 2.0 [1.0, 5.0] | 121/192 | 40/180 | 0.162 |
| GRU over the linear probe scores | 0.948 [0.929, 0.961] | 0.897 [0.860, 0.927] | 0.904 [0.869, 0.931] | 1.0 [0.0, 2.0] | 153/192 | 27/180 | 0.026 |
| turn index alone | 0.700 [0.676, 0.724] | 0.479 [0.422, 0.545] | 0.479 [0.422, 0.545] | 14.0 [7.0, 18.0] | 107/192 | 109/180 | 0.198 |
| TF-IDF on the agent's words so far | 0.972 [0.962, 0.981] | 0.971 [0.955, 0.983] | 0.966 [0.948, 0.981] | 0.0 [0.0, 1.0] | 178/192 | 21/180 | 0.017 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.001, agent last 0.005, agent max 0.001.

## Controls, one agent per run (five authors): controls_600_think

600 agents, 600 runs, 10985 steps. Split: 5 folds grouped by run.

### cheating (the set's label: from tried or opened by task)

593 agents scored per step (336 reach the cut), 10834 steps (3759 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.983 [0.975, 0.989] | 0.985 [0.976, 0.993] | 0.989 [0.980, 0.996] | 0.0 [0.0, 0.5] | 331/336 | 30/257 | 0.039 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.976 [0.966, 0.984] | 0.983 [0.972, 0.992] | 0.984 [0.973, 0.992] | 0.0 [0.0, 1.0] | 330/336 | 28/257 | 0.234 |
| attention monitor, binary head (seed 1) | 0.984 [0.975, 0.990] | 0.988 [0.980, 0.995] | 0.987 [0.980, 0.995] | 0.0 [0.0, 1.0] | 331/336 | 32/257 | 0.042 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.979 [0.970, 0.986] | 0.979 [0.967, 0.989] | 0.982 [0.969, 0.991] | 0.0 [0.0, 1.0] | 331/336 | 30/257 | 0.222 |
| linear per-step probe (team_probes) | 0.916 [0.903, 0.929] | 0.835 [0.806, 0.867] | 0.921 [0.899, 0.941] | 1.0 [0.0, 1.0] | 309/336 | 79/257 | 0.062 |
| cumulative mean of the linear probe | 0.914 [0.900, 0.925] | 0.843 [0.810, 0.874] | 0.842 [0.809, 0.872] | 2.0 [1.0, 4.0] | 239/336 | 53/257 | 0.180 |
| GRU over the linear probe scores | 0.943 [0.929, 0.955] | 0.901 [0.877, 0.924] | 0.912 [0.887, 0.933] | 1.0 [0.0, 2.0] | 274/336 | 40/257 | 0.027 |
| turn index alone | 0.672 [0.653, 0.692] | 0.448 [0.399, 0.495] | 0.448 [0.399, 0.495] | 16.0 [8.0, 19.0] | 157/336 | 144/257 | 0.140 |
| TF-IDF on the agent's words so far | 0.973 [0.967, 0.979] | 0.974 [0.962, 0.983] | 0.973 [0.962, 0.983] | 0.0 [0.0, 1.0] | 314/336 | 30/257 | 0.015 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.001, agent last 0.003, agent max 0.002.

Permutation null (50 draws), attention monitor seed 0:

| Metric | Real | Null mean | Null 95th percentile | p |
|---|---|---|---|---|
| step AUROC | 0.983 | 0.679 | 0.698 | 0.020 |
| agent AUROC (last) | 0.985 | 0.507 | 0.551 | 0.020 |
| agent AUROC (max) | 0.989 | 0.506 | 0.552 | 0.020 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.443 | 1.432 | 0.853 |
| cheat | attention monitor (seed 1) | 0.440 | 1.390 | 0.858 |
| cheat | majority level of the training fold | 0.325 | 3.568 | 0.403 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 57 | 46 | 1.000 |  |  | 0.973 |  |  |
| 0 | 1 | 57 | 46 | 1.000 |  |  | 0.989 |  |  |
| 0 | 2 | 60 | 59 | 1.000 |  |  | 0.996 |  |  |
| 0 | 3 | 48 | 37 | 1.000 |  |  | 0.995 |  |  |
| 0 | 4 | 42 | 31 | 0.999 |  |  | 0.985 |  |  |
| 1 | 0 | 42 | 31 | 1.000 |  |  | 0.993 |  |  |
| 1 | 1 | 60 | 54 | 1.000 |  |  | 0.995 |  |  |
| 1 | 2 | 51 | 40 | 1.000 |  |  | 0.992 |  |  |
| 1 | 3 | 40 | 29 | 1.000 |  |  | 0.994 |  |  |
| 1 | 4 | 47 | 36 | 1.000 |  |  | 0.992 |  |  |

## Controls, one agent per run (five authors): controls_600_think_heldout_author

600 agents, 600 runs, 10985 steps. Split: held-out author (train on the other authors, test on one).

### cheating (the set's label: from tried or opened by task)

593 agents scored per step (336 reach the cut), 10834 steps (3759 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.923 [0.907, 0.939] | 0.903 [0.878, 0.929] | 0.908 [0.884, 0.933] | 1.0 [0.0, 1.0] | 276/336 | 43/257 | 0.128 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.912 [0.895, 0.929] | 0.886 [0.856, 0.913] | 0.890 [0.861, 0.917] | 1.0 [0.0, 1.0] | 283/336 | 40/257 | 0.245 |
| attention monitor, binary head (seed 1) | 0.942 [0.930, 0.953] | 0.906 [0.878, 0.931] | 0.908 [0.881, 0.932] | 1.0 [1.0, 2.0] | 290/336 | 42/257 | 0.098 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.938 [0.924, 0.949] | 0.897 [0.864, 0.923] | 0.896 [0.863, 0.922] | 1.0 [0.0, 2.0] | 303/336 | 46/257 | 0.237 |
| linear per-step probe (team_probes) | 0.839 [0.818, 0.861] | 0.733 [0.694, 0.772] | 0.838 [0.810, 0.867] | 1.0 [0.0, 1.0] | 272/336 | 83/257 | 0.065 |
| cumulative mean of the linear probe | 0.869 [0.849, 0.886] | 0.767 [0.729, 0.804] | 0.769 [0.732, 0.807] | 2.0 [1.0, 5.0] | 199/336 | 52/257 | 0.212 |
| GRU over the linear probe scores | 0.884 [0.859, 0.909] | 0.819 [0.784, 0.854] | 0.852 [0.821, 0.880] | 1.0 [0.0, 2.0] | 254/336 | 56/257 | 0.069 |
| turn index alone | 0.671 [0.650, 0.691] | 0.443 [0.395, 0.493] | 0.443 [0.395, 0.493] | 16.0 [8.0, 19.0] | 158/336 | 143/257 | 0.140 |
| TF-IDF on the agent's words so far | 0.953 [0.943, 0.962] | 0.933 [0.916, 0.951] | 0.939 [0.921, 0.956] | 0.0 [0.0, 1.0] | 285/336 | 25/257 | 0.030 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.019, agent last 0.003, agent max 0.000.

### Per held-out author, cheating (the set's label: from tried or opened by task) (step AUROC / agent AUROC from the last step)

| Model | ds held out | gpt5m held out | haiku held out | luna held out | qwen held out |
|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.923 / 0.918 | 0.945 / 0.901 | 0.944 / 0.930 | 0.933 / 0.907 | 0.966 / 0.961 |
| attention monitor, binary head (seed 1) | 0.956 / 0.931 | 0.961 / 0.932 | 0.946 / 0.962 | 0.914 / 0.846 | 0.985 / 0.994 |
| linear per-step probe (team_probes) | 0.814 / 0.552 | 0.827 / 0.739 | 0.925 / 0.821 | 0.900 / 0.809 | 0.871 / 0.809 |
| cumulative mean of the linear probe | 0.914 / 0.785 | 0.822 / 0.689 | 0.852 / 0.841 | 0.894 / 0.801 | 0.937 / 0.830 |
| GRU over the linear probe scores | 0.826 / 0.643 | 0.841 / 0.778 | 0.933 / 0.886 | 0.945 / 0.926 | 0.976 / 0.952 |
| turn index alone | 0.710 / 0.557 | 0.713 / 0.499 | 0.592 / 0.432 | 0.689 / 0.451 | 0.646 / 0.262 |
| TF-IDF on the agent's words so far | 0.975 / 0.948 | 0.899 / 0.881 | 0.954 / 0.947 | 0.955 / 0.967 | 0.977 / 0.964 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.263 | 1.833 | 0.583 |
| cheat | attention monitor (seed 1) | 0.377 | 1.657 | 0.734 |
| cheat | majority level of the training fold | 0.325 | 3.568 | 0.326 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 17 | 6 | 0.996 |  |  | 0.990 |  |  |
| 0 | 1 | 22 | 11 | 0.998 |  |  | 0.981 |  |  |
| 0 | 2 | 60 | 51 | 1.000 |  |  | 0.994 |  |  |
| 0 | 3 | 60 | 56 | 1.000 |  |  | 0.993 |  |  |
| 0 | 4 | 45 | 34 | 1.000 |  |  | 0.976 |  |  |
| 1 | 0 | 60 | 59 | 1.000 |  |  | 0.994 |  |  |
| 1 | 1 | 42 | 31 | 1.000 |  |  | 0.994 |  |  |
| 1 | 2 | 60 | 59 | 1.000 |  |  | 0.994 |  |  |
| 1 | 3 | 54 | 43 | 1.000 |  |  | 0.983 |  |  |
| 1 | 4 | 34 | 23 | 0.999 |  |  | 0.988 |  |  |

## Controls, one agent per run (five authors): controls_600_think_soloarm

373 agents, 373 runs, 7366 steps. Split: 5 folds grouped by run. Scored on the solo arm only (same held-out scores as the full run).

### cheating (the set's label: from tried or opened by task), solo arm only

372 agents scored per step (192 reach the cut), 7341 steps (2174 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.987 [0.980, 0.992] | 0.989 [0.978, 0.997] | 0.990 [0.980, 0.998] | 0.0 [0.0, 0.0] | 191/192 | 28/180 | 0.031 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.983 [0.974, 0.989] | 0.987 [0.975, 0.996] | 0.987 [0.975, 0.995] | 0.0 [0.0, 0.0] | 190/192 | 21/180 | 0.204 |
| attention monitor, binary head (seed 1) | 0.987 [0.979, 0.993] | 0.993 [0.987, 0.997] | 0.990 [0.981, 0.997] | 0.0 [0.0, 1.0] | 190/192 | 22/180 | 0.039 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.982 [0.971, 0.989] | 0.987 [0.976, 0.995] | 0.990 [0.980, 0.997] | 0.0 [0.0, 1.0] | 191/192 | 21/180 | 0.192 |
| linear per-step probe (team_probes) | 0.918 [0.901, 0.932] | 0.819 [0.772, 0.855] | 0.907 [0.875, 0.936] | 1.0 [0.0, 1.0] | 169/192 | 61/180 | 0.090 |
| cumulative mean of the linear probe | 0.909 [0.891, 0.926] | 0.822 [0.777, 0.866] | 0.822 [0.776, 0.865] | 2.0 [1.0, 5.0] | 121/192 | 40/180 | 0.162 |
| GRU over the linear probe scores | 0.948 [0.929, 0.961] | 0.897 [0.860, 0.927] | 0.904 [0.869, 0.931] | 1.0 [0.0, 2.0] | 153/192 | 27/180 | 0.026 |
| turn index alone | 0.700 [0.676, 0.724] | 0.479 [0.422, 0.545] | 0.479 [0.422, 0.545] | 14.0 [7.0, 18.0] | 107/192 | 109/180 | 0.198 |
| TF-IDF on the agent's words so far | 0.972 [0.962, 0.981] | 0.971 [0.955, 0.983] | 0.966 [0.948, 0.981] | 0.0 [0.0, 1.0] | 178/192 | 21/180 | 0.017 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.000, agent last 0.004, agent max 0.000.

## Solo design: solo_design_think

177 agents, 59 runs, 6086 steps. Split: 5 folds grouped by run.

### cheating used or above

173 agents scored per step (80 reach the cut), 6000 steps (1838 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.892 [0.839, 0.940] | 0.894 [0.825, 0.954] | 0.883 [0.814, 0.943] | 1.0 [0.0, 5.8] | 54/80 | 12/93 | 0.106 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.898 [0.848, 0.946] | 0.903 [0.839, 0.955] | 0.897 [0.832, 0.949] | 1.0 [1.0, 8.0] | 60/80 | 12/93 | 0.057 |
| attention monitor, binary head (seed 1) | 0.896 [0.840, 0.944] | 0.903 [0.844, 0.949] | 0.894 [0.831, 0.942] | 0.5 [0.0, 1.0] | 52/80 | 10/93 | 0.116 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.904 [0.850, 0.950] | 0.907 [0.845, 0.955] | 0.907 [0.847, 0.954] | 2.0 [0.0, 3.5] | 59/80 | 12/93 | 0.085 |
| linear per-step probe (team_probes) | 0.886 [0.830, 0.936] | 0.894 [0.832, 0.948] | 0.900 [0.835, 0.954] | 1.0 [0.0, 2.0] | 69/80 | 18/93 | 0.089 |
| cumulative mean of the linear probe | 0.854 [0.788, 0.910] | 0.842 [0.769, 0.904] | 0.838 [0.761, 0.902] | 6.0 [4.0, 9.5] | 43/80 | 9/93 | 0.105 |
| GRU over the linear probe scores | 0.886 [0.830, 0.937] | 0.869 [0.802, 0.929] | 0.877 [0.812, 0.931] | 1.0 [0.0, 2.0] | 61/80 | 18/93 | 0.037 |
| turn index alone | 0.643 [0.580, 0.711] | 0.445 [0.335, 0.559] | 0.445 [0.335, 0.559] | 44.5 [38.2, 53.0] | 16/80 | 23/93 | 0.185 |
| TF-IDF on the agent's words so far | 0.911 [0.843, 0.963] | 0.901 [0.824, 0.959] | 0.901 [0.825, 0.960] | 1.0 [0.0, 2.0] | 58/80 | 10/93 | 0.084 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.005, agent last 0.010, agent max 0.011.

### coordination posted or above

177 agents scored per step (38 reach the cut), 6086 steps (552 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.889 [0.712, 0.962] | 0.852 [0.764, 0.918] | 0.849 [0.762, 0.914] | 0.0 [0.0, 1.0] | 32/38 | 37/139 | 0.039 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.843 [0.625, 0.947] | 0.857 [0.754, 0.930] | 0.842 [0.736, 0.920] | 0.0 [0.0, 1.0] | 28/38 | 36/139 | 0.202 |
| attention monitor, binary head (seed 1) | 0.855 [0.626, 0.962] | 0.838 [0.745, 0.918] | 0.846 [0.744, 0.929] | 0.0 [0.0, 1.0] | 29/38 | 26/139 | 0.049 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.844 [0.624, 0.947] | 0.822 [0.698, 0.920] | 0.845 [0.719, 0.924] | 0.0 [0.0, 1.0] | 29/38 | 26/139 | 0.185 |
| linear per-step probe (team_probes) | 0.802 [0.522, 0.935] | 0.795 [0.676, 0.893] | 0.757 [0.614, 0.878] | 0.0 [0.0, 1.0] | 28/38 | 44/139 | 0.145 |
| cumulative mean of the linear probe | 0.756 [0.434, 0.895] | 0.518 [0.335, 0.692] | 0.471 [0.274, 0.675] | 2.0 [0.0, 5.0] | 13/38 | 21/139 | 0.113 |
| GRU over the linear probe scores | 0.791 [0.488, 0.947] | 0.796 [0.677, 0.899] | 0.745 [0.571, 0.875] | 0.0 [0.0, 0.5] | 27/38 | 85/139 | 0.016 |
| turn index alone | 0.637 [0.445, 0.733] | 0.546 [0.419, 0.670] | 0.546 [0.419, 0.670] | 46.0 [44.0, 50.0] | 5/38 | 28/139 | 0.376 |
| TF-IDF on the agent's words so far | 0.758 [0.431, 0.905] | 0.522 [0.374, 0.665] | 0.496 [0.346, 0.658] | 0.0 [0.0, 0.0] | 6/38 | 19/139 | 0.166 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.034, agent last 0.014, agent max 0.003.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.282 | 1.621 | 0.620 |
| cheat | attention monitor (seed 1) | 0.282 | 1.525 | 0.593 |
| cheat | majority level of the training fold | 0.367 | 3.130 | 0.239 |
| coord | attention monitor (seed 0) | 0.407 | 0.655 | 0.323 |
| coord | attention monitor (seed 1) | 0.345 | 0.746 | 0.280 |
| coord | majority level of the training fold | 0.667 | 0.384 | 0.758 |
| coop | attention monitor (seed 0) | 1.000 | 0.000 | 1.000 |
| coop | attention monitor (seed 1) | 1.000 | 0.000 | 1.000 |
| coop | majority level of the training fold | 1.000 | 0.000 | 1.000 |
| collab | attention monitor (seed 0) | 1.000 | 0.000 | 1.000 |
| collab | attention monitor (seed 1) | 1.000 | 0.000 | 1.000 |
| collab | majority level of the training fold | 1.000 | 0.000 | 1.000 |
| collude | attention monitor (seed 0) | 1.000 | 0.000 | 1.000 |
| collude | attention monitor (seed 1) | 1.000 | 0.000 | 1.000 |
| collude | majority level of the training fold | 1.000 | 0.000 | 1.000 |
| scheme | attention monitor (seed 0) | 1.000 | 0.000 | 1.000 |
| scheme | attention monitor (seed 1) | 1.000 | 0.000 | 1.000 |
| scheme | majority level of the training fold | 1.000 | 0.000 | 1.000 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 23 | 12 | 0.998 |  | 0.992 | 0.931 |  | 0.827 |
| 0 | 1 | 33 | 22 | 1.000 |  | 0.998 | 0.977 |  | 0.963 |
| 0 | 2 | 16 | 5 | 0.989 |  | 0.991 | 0.858 |  | 0.997 |
| 0 | 3 | 19 | 8 | 0.996 |  | 0.992 | 0.937 |  | 0.976 |
| 0 | 4 | 54 | 43 | 1.000 |  | 1.000 | 0.992 |  | 0.923 |
| 1 | 0 | 41 | 30 | 1.000 |  | 1.000 | 0.979 |  | 0.982 |
| 1 | 1 | 30 | 19 | 1.000 |  | 0.999 | 0.819 |  | 0.952 |
| 1 | 2 | 19 | 8 | 0.995 |  | 0.988 | 0.930 |  | 0.999 |
| 1 | 3 | 15 | 4 | 0.971 |  | 0.945 | 0.821 |  | 0.955 |
| 1 | 4 | 23 | 12 | 0.997 |  | 0.997 | 0.858 |  | 0.478 |

## Solo design: solo_design_think_heldout_author

177 agents, 59 runs, 6086 steps. Split: held-out author (train on the other authors, test on one).

### cheating used or above

173 agents scored per step (80 reach the cut), 6000 steps (1838 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.665 [0.617, 0.721] | 0.334 [0.235, 0.438] | 0.278 [0.181, 0.384] | 2.0 [1.0, 2.0] | 15/80 | 47/93 | 0.261 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.797 [0.729, 0.856] | 0.746 [0.657, 0.820] | 0.770 [0.672, 0.856] | 5.0 [2.0, 18.0] | 61/80 | 29/93 | 0.172 |
| attention monitor, binary head (seed 1) | 0.698 [0.648, 0.751] | 0.407 [0.291, 0.513] | 0.370 [0.253, 0.482] | 1.0 [0.8, 1.0] | 16/80 | 46/93 | 0.159 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.768 [0.711, 0.829] | 0.661 [0.555, 0.764] | 0.654 [0.536, 0.764] | 11.0 [1.0, 21.0] | 49/80 | 47/93 | 0.109 |
| linear per-step probe (team_probes) | 0.726 [0.664, 0.792] | 0.604 [0.487, 0.715] | 0.391 [0.274, 0.512] | 2.0 [1.0, 2.0] | 17/80 | 29/93 | 0.161 |
| cumulative mean of the linear probe | 0.771 [0.693, 0.853] | 0.698 [0.565, 0.815] | 0.698 [0.565, 0.814] | 22.0 [19.0, 34.0] | 13/80 | 14/93 | 0.086 |
| GRU over the linear probe scores | 0.702 [0.616, 0.796] | 0.671 [0.536, 0.785] | 0.673 [0.534, 0.787] | 1.0 [1.0, 1.0] | 1/80 | 8/93 | 0.255 |
| turn index alone | 0.682 [0.622, 0.747] | 0.498 [0.379, 0.614] | 0.498 [0.379, 0.614] | 44.0 [32.0, 45.0] | 17/80 | 19/93 | 0.196 |
| TF-IDF on the agent's words so far | 0.809 [0.745, 0.869] | 0.836 [0.740, 0.915] | 0.833 [0.739, 0.915] | 1.5 [0.0, 14.2] | 44/80 | 8/93 | 0.126 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.033, agent last 0.074, agent max 0.091.

### coordination posted or above

177 agents scored per step (38 reach the cut), 6086 steps (552 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.580 [0.500, 0.727] | 0.601 [0.462, 0.715] | 0.567 [0.445, 0.682] | 5.0 [1.0, 11.5] | 16/38 | 63/139 | 0.053 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.874 [0.795, 0.908] | 0.561 [0.404, 0.713] | 0.549 [0.397, 0.701] | 5.5 [1.5, 11.0] | 22/38 | 72/139 | 0.165 |
| attention monitor, binary head (seed 1) | 0.446 [0.336, 0.659] | 0.623 [0.462, 0.781] | 0.619 [0.458, 0.779] | 0.0 [0.0, 0.0] | 21/38 | 58/139 | 0.080 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.391 [0.267, 0.652] | 0.590 [0.427, 0.734] | 0.586 [0.435, 0.727] | 0.0 [0.0, 3.0] | 27/38 | 69/139 | 0.127 |
| linear per-step probe (team_probes) | 0.794 [0.642, 0.872] | 0.711 [0.619, 0.799] | 0.693 [0.580, 0.809] | 1.0 [0.0, 10.8] | 34/38 | 93/139 | 0.109 |
| cumulative mean of the linear probe | 0.802 [0.589, 0.899] | 0.603 [0.451, 0.728] | 0.560 [0.420, 0.679] | 7.0 [0.0, 11.0] | 9/38 | 26/139 | 0.043 |
| GRU over the linear probe scores | 0.707 [0.629, 0.831] | 0.580 [0.468, 0.692] | 0.438 [0.326, 0.556] | 2.5 [0.0, 6.5] | 4/38 | 12/139 | 0.121 |
| turn index alone | 0.632 [0.536, 0.697] | 0.594 [0.454, 0.720] | 0.594 [0.454, 0.720] | 0.0 [0.0, 24.5] | 7/38 | 28/139 | 0.395 |
| TF-IDF on the agent's words so far | 0.436 [0.369, 0.526] | 0.515 [0.341, 0.677] | 0.505 [0.328, 0.674] | 0.0 [0.0, 0.0] | 7/38 | 14/139 | 0.176 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.134, agent last 0.022, agent max 0.052.

### Per held-out author, cheating used or above (step AUROC / agent AUROC from the last step)

| Model | ds held out | luna held out |
|---|---|---|
| attention monitor, binary head (seed 0) | 0.681 / 0.491 | 0.684 / 0.675 |
| attention monitor, binary head (seed 1) | 0.634 / 0.605 | 0.731 / 0.728 |
| linear per-step probe (team_probes) | 0.618 / 0.435 | 0.762 / 0.726 |
| cumulative mean of the linear probe | 0.617 / 0.270 | 0.670 / 0.663 |
| GRU over the linear probe scores | 0.556 / 0.258 | 0.712 / 0.715 |
| turn index alone | 0.615 / 0.267 | 0.539 / 0.296 |
| TF-IDF on the agent's words so far | 0.805 / 0.833 | 0.911 / 0.900 |

### Per held-out author, coordination posted or above (step AUROC / agent AUROC from the last step)

| Model | ds held out | luna held out |
|---|---|---|
| attention monitor, binary head (seed 0) | 0.778 / 0.586 | 0.741 / 0.563 |
| attention monitor, binary head (seed 1) | 0.682 / 0.767 | 0.752 / 0.504 |
| linear per-step probe (team_probes) | 0.684 / 0.837 | 0.832 / 0.488 |
| cumulative mean of the linear probe | 0.710 / 0.664 | 0.851 / 0.778 |
| GRU over the linear probe scores | 0.836 / 0.648 | 0.669 / 0.471 |
| turn index alone | 0.623 / 0.524 | 0.751 / 0.554 |
| TF-IDF on the agent's words so far | 0.540 / 0.682 | 0.472 / 0.285 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.040 | 2.475 | 0.388 |
| cheat | attention monitor (seed 1) | 0.056 | 2.486 | 0.420 |
| cheat | majority level of the training fold | 0.124 | 4.531 | 0.106 |
| coord | attention monitor (seed 0) | 0.305 | 0.842 | 0.172 |
| coord | attention monitor (seed 1) | 0.203 | 1.045 | 0.153 |
| coord | majority level of the training fold | 0.667 | 0.384 | 0.758 |
| coop | attention monitor (seed 0) | 1.000 | 0.000 | 1.000 |
| coop | attention monitor (seed 1) | 1.000 | 0.000 | 1.000 |
| coop | majority level of the training fold | 1.000 | 0.000 | 1.000 |
| collab | attention monitor (seed 0) | 1.000 | 0.000 | 1.000 |
| collab | attention monitor (seed 1) | 1.000 | 0.000 | 1.000 |
| collab | majority level of the training fold | 1.000 | 0.000 | 1.000 |
| collude | attention monitor (seed 0) | 1.000 | 0.000 | 1.000 |
| collude | attention monitor (seed 1) | 1.000 | 0.000 | 1.000 |
| collude | majority level of the training fold | 1.000 | 0.000 | 1.000 |
| scheme | attention monitor (seed 0) | 1.000 | 0.000 | 1.000 |
| scheme | attention monitor (seed 1) | 1.000 | 0.000 | 1.000 |
| scheme | majority level of the training fold | 1.000 | 0.000 | 1.000 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 19 | 8 | 0.999 |  | 0.999 | 0.977 |  | 0.919 |
| 0 | 1 | 20 | 9 | 0.995 |  | 0.966 | 0.916 |  | 0.260 |
| 0 | 2 | 21 | 10 | 0.997 |  | 0.993 | 0.966 |  | 0.941 |
| 1 | 0 | 14 | 3 | 0.933 |  | 0.995 | 0.954 |  | 0.670 |
| 1 | 1 | 19 | 8 | 0.995 |  | 0.987 | 0.956 |  | 0.603 |
| 1 | 2 | 31 | 20 | 0.999 |  | 0.998 | 0.915 |  | 0.899 |

## SPAR mixed: spar_mixed_41_think

209 agents, 51 runs, 12373 steps. Split: 5 folds grouped by run.

### cheating (the set's label: from tried or opened by task)

205 agents scored per step (39 reach the cut), 12065 steps (1452 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.750 [0.659, 0.835] | 0.649 [0.536, 0.767] | 0.693 [0.613, 0.779] | 2.0 [1.0, 21.0] | 18/39 | 40/166 | 0.090 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.757 [0.659, 0.835] | 0.623 [0.507, 0.741] | 0.660 [0.559, 0.763] | 2.0 [1.0, 6.5] | 18/39 | 25/166 | 0.051 |
| attention monitor, binary head (seed 1) | 0.816 [0.742, 0.865] | 0.724 [0.622, 0.821] | 0.755 [0.659, 0.840] | 0.0 [0.0, 1.5] | 15/39 | 16/166 | 0.089 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.817 [0.732, 0.877] | 0.712 [0.614, 0.807] | 0.730 [0.637, 0.813] | 1.0 [0.0, 2.0] | 13/39 | 21/166 | 0.037 |
| linear per-step probe (team_probes) | 0.787 [0.719, 0.842] | 0.671 [0.548, 0.782] | 0.693 [0.591, 0.786] | 1.0 [0.8, 4.0] | 16/39 | 25/166 | 0.139 |
| cumulative mean of the linear probe | 0.791 [0.712, 0.857] | 0.648 [0.514, 0.767] | 0.639 [0.498, 0.763] | 19.0 [11.5, 42.0] | 11/39 | 16/166 | 0.069 |
| GRU over the linear probe scores | 0.765 [0.670, 0.848] | 0.656 [0.545, 0.762] | 0.654 [0.547, 0.747] | 6.0 [0.0, 59.0] | 9/39 | 10/166 | 0.050 |
| turn index alone | 0.451 [0.373, 0.527] | 0.484 [0.371, 0.583] | 0.480 [0.380, 0.592] |  | 0/39 | 5/166 | 0.387 |
| TF-IDF on the agent's words so far | 0.888 [0.830, 0.932] | 0.839 [0.770, 0.911] | 0.836 [0.763, 0.914] | 1.0 [0.0, 11.0] | 20/39 | 17/166 | 0.046 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.066, agent last 0.074, agent max 0.062.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.455 | 0.794 | 0.511 |
| cheat | attention monitor (seed 1) | 0.182 | 1.115 | 0.323 |
| cheat | majority level of the training fold | 0.761 | 0.536 | 0.648 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 17 | 6 | 0.959 |  |  | 0.846 |  |  |
| 0 | 1 | 20 | 12 | 0.972 |  |  | 0.923 |  |  |
| 0 | 2 | 47 | 36 | 1.000 |  |  | 0.987 |  |  |
| 0 | 3 | 60 | 59 | 0.999 |  |  | 0.987 |  |  |
| 0 | 4 | 22 | 11 | 0.980 |  |  | 0.858 |  |  |
| 1 | 0 | 11 | 0 | 0.820 |  |  | 0.822 |  |  |
| 1 | 1 | 26 | 15 | 0.984 |  |  | 0.970 |  |  |
| 1 | 2 | 34 | 23 | 0.999 |  |  | 0.957 |  |  |
| 1 | 3 | 30 | 19 | 0.998 |  |  | 0.944 |  |  |
| 1 | 4 | 16 | 5 | 0.932 |  |  | 0.884 |  |  |

## SPAR mixed: spar_mixed_41_think_heldout_author

209 agents, 51 runs, 12373 steps. Split: held-out author (train on the other authors, test on one).

### cheating (the set's label: from tried or opened by task)

205 agents scored per step (39 reach the cut), 12065 steps (1452 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.784 [0.681, 0.853] | 0.671 [0.563, 0.771] | 0.708 [0.589, 0.809] | 2.5 [1.2, 3.8] | 14/39 | 16/166 | 0.150 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.796 [0.714, 0.861] | 0.636 [0.495, 0.762] | 0.643 [0.498, 0.768] | 3.0 [0.5, 13.5] | 15/39 | 16/166 | 0.032 |
| attention monitor, binary head (seed 1) | 0.568 [0.434, 0.670] | 0.607 [0.488, 0.725] | 0.660 [0.562, 0.754] | 25.0 [3.5, 36.0] | 11/39 | 21/166 | 0.034 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.542 [0.437, 0.645] | 0.548 [0.457, 0.637] | 0.615 [0.527, 0.711] | 44.5 [23.2, 53.8] | 14/39 | 25/166 | 0.026 |
| linear per-step probe (team_probes) | 0.772 [0.670, 0.841] | 0.685 [0.554, 0.796] | 0.709 [0.598, 0.807] | 14.0 [1.0, 27.0] | 17/39 | 17/166 | 0.159 |
| cumulative mean of the linear probe | 0.780 [0.664, 0.858] | 0.718 [0.617, 0.806] | 0.672 [0.553, 0.773] | 23.0 [4.0, 34.8] | 14/39 | 12/166 | 0.091 |
| GRU over the linear probe scores | 0.733 [0.625, 0.806] | 0.718 [0.613, 0.819] | 0.688 [0.557, 0.811] | 33.0 [6.0, 39.5] | 14/39 | 14/166 | 0.121 |
| turn index alone | 0.460 [0.365, 0.532] | 0.666 [0.561, 0.761] | 0.603 [0.485, 0.714] |  | 0/39 | 5/166 | 0.497 |
| TF-IDF on the agent's words so far | 0.818 [0.750, 0.873] | 0.740 [0.628, 0.839] | 0.724 [0.594, 0.837] | 1.0 [0.0, 6.0] | 18/39 | 13/166 | 0.053 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.216, agent last 0.064, agent max 0.048.

### Per held-out author, cheating (the set's label: from tried or opened by task) (step AUROC / agent AUROC from the last step)

| Model | deepseek-flash-openrouter held out | gpt-6-luna-openrouter held out |
|---|---|---|
| attention monitor, binary head (seed 0) | 0.782 / 0.581 | 0.532 / 0.558 |
| attention monitor, binary head (seed 1) | 0.629 / 0.522 | 0.483 / 0.494 |
| linear per-step probe (team_probes) | 0.775 / 0.592 | 0.565 / 0.527 |
| cumulative mean of the linear probe | 0.801 / 0.598 | 0.571 / 0.598 |
| GRU over the linear probe scores | 0.756 / 0.600 | 0.553 / 0.605 |
| turn index alone | 0.429 / 0.422 | 0.308 / 0.509 |
| TF-IDF on the agent's words so far | 0.748 / 0.589 | 0.747 / 0.628 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.091 | 1.234 | 0.261 |
| cheat | attention monitor (seed 1) | 0.555 | 0.799 | 0.480 |
| cheat | majority level of the training fold | 0.761 | 0.536 | 0.648 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 16 | 5 | 0.986 |  |  | 0.927 |  |  |
| 0 | 1 | 13 | 2 | 0.823 |  |  | 0.899 |  |  |
| 1 | 0 | 11 | 0 | 0.833 |  |  | 0.211 |  |  |
| 1 | 1 | 11 | 0 | 0.836 |  |  | 0.606 |  |  |

## SPAR mixed: spar_mixed_41_think_ladders

158 agents, 51 runs, 7440 steps. Split: 5 folds grouped by run.

### cheating used or above

158 agents scored per step (12 reach the cut), 7440 steps (511 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.798 [0.606, 0.930] | 0.833 [0.718, 0.949] | 0.930 [0.856, 0.980] | 1.0 [0.0, 26.0] | 9/12 | 15/146 | 0.043 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.885 [0.768, 0.956] | 0.872 [0.774, 0.964] | 0.933 [0.865, 0.980] | 0.0 [0.0, 1.0] | 10/12 | 23/146 | 0.041 |
| attention monitor, binary head (seed 1) | 0.893 [0.817, 0.946] | 0.852 [0.758, 0.927] | 0.902 [0.820, 0.959] | 8.0 [0.0, 10.0] | 5/12 | 16/146 | 0.071 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.887 [0.796, 0.948] | 0.849 [0.763, 0.926] | 0.905 [0.827, 0.957] | 6.0 [1.2, 9.2] | 6/12 | 17/146 | 0.062 |
| linear per-step probe (team_probes) | 0.776 [0.633, 0.902] | 0.805 [0.669, 0.912] | 0.857 [0.703, 0.955] | 2.5 [0.8, 21.5] | 8/12 | 27/146 | 0.112 |
| cumulative mean of the linear probe | 0.640 [0.504, 0.874] | 0.751 [0.573, 0.888] | 0.703 [0.521, 0.860] | 1.5 [0.8, 2.2] | 2/12 | 11/146 | 0.082 |
| GRU over the linear probe scores | 0.610 [0.369, 0.790] | 0.559 [0.374, 0.747] | 0.682 [0.497, 0.839] | 4.0 [0.0, 4.0] | 5/12 | 20/146 | 0.072 |
| turn index alone | 0.794 [0.698, 0.851] | 0.769 [0.560, 0.913] | 0.769 [0.560, 0.913] | 60.0 [41.0, 60.0] | 5/12 | 14/146 | 0.323 |
| TF-IDF on the agent's words so far | 0.947 [0.885, 0.974] | 0.866 [0.747, 0.959] | 0.889 [0.773, 0.978] | 1.5 [0.0, 5.0] | 8/12 | 10/146 | 0.071 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.095, agent last 0.019, agent max 0.027.

### collusion exploit crossed or above

158 agents scored per step (7 reach the cut), 7440 steps (395 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.811 [0.656, 0.960] | 0.841 [0.673, 0.994] | 0.967 [0.933, 0.993] | 5.0 [1.0, 14.0] | 7/7 | 14/151 | 0.048 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.847 [0.729, 0.960] | 0.864 [0.729, 0.994] | 0.959 [0.926, 0.995] | 14.5 [6.2, 22.8] | 6/7 | 15/151 | 0.046 |
| attention monitor, binary head (seed 1) | 0.855 [0.801, 0.971] | 0.899 [0.756, 1.000] | 0.968 [0.926, 0.994] | 1.5 [0.2, 38.0] | 6/7 | 12/151 | 0.067 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.863 [0.802, 0.974] | 0.939 [0.853, 1.000] | 0.964 [0.930, 0.991] | 1.0 [0.5, 26.5] | 7/7 | 12/151 | 0.043 |
| linear per-step probe (team_probes) | 0.830 [0.734, 0.969] | 0.886 [0.769, 0.981] | 0.909 [0.836, 0.970] | 13.5 [5.0, 19.0] | 6/7 | 17/151 | 0.120 |
| cumulative mean of the linear probe | 0.823 [0.654, 0.998] | 0.891 [0.732, 0.997] | 0.883 [0.713, 0.996] | 0.0 [0.0, 0.5] | 4/7 | 18/151 | 0.104 |
| GRU over the linear probe scores | 0.371 [0.266, 0.925] | 0.447 [0.294, 0.930] | 0.425 [0.285, 0.616] |  | 0/7 | 11/151 | 0.083 |
| turn index alone | 0.855 [0.765, 1.000] | 0.947 [0.873, 1.000] | 0.947 [0.873, 1.000] | 56.0 [25.0, 58.0] | 5/7 | 14/151 | 0.278 |
| TF-IDF on the agent's words so far | 0.934 [0.852, 0.974] | 0.911 [0.812, 0.963] | 0.928 [0.870, 0.976] | 31.0 [18.0, 47.0] | 4/7 | 12/151 | 0.036 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.045, agent last 0.058, agent max 0.001.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.380 | 0.930 | 0.455 |
| cheat | attention monitor (seed 1) | 0.259 | 1.152 | 0.400 |
| cheat | majority level of the training fold | 0.741 | 0.639 | 0.580 |
| coord | attention monitor (seed 0) | 0.741 | 0.475 | 0.850 |
| coord | attention monitor (seed 1) | 0.741 | 0.468 | 0.857 |
| coord | majority level of the training fold | 0.741 | 0.468 | 0.751 |
| coop | attention monitor (seed 0) | 0.886 | 0.209 | 0.941 |
| coop | attention monitor (seed 1) | 0.867 | 0.234 | 0.932 |
| coop | majority level of the training fold | 0.886 | 0.234 | 0.948 |
| collab | attention monitor (seed 0) | 0.943 | 0.095 | 0.945 |
| collab | attention monitor (seed 1) | 0.949 | 0.076 | 0.945 |
| collab | majority level of the training fold | 0.949 | 0.089 | 0.948 |
| collude | attention monitor (seed 0) | 0.943 | 0.171 | 0.944 |
| collude | attention monitor (seed 1) | 0.949 | 0.152 | 0.945 |
| collude | majority level of the training fold | 0.949 | 0.171 | 0.947 |
| scheme | attention monitor (seed 0) | 0.956 | 0.089 | 1.000 |
| scheme | attention monitor (seed 1) | 0.956 | 0.089 | 1.000 |
| scheme | majority level of the training fold | 0.956 | 0.089 | 1.000 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 26 | 15 | 0.994 | 1.000 | 0.998 | 0.914 |  | 0.944 |
| 0 | 1 | 16 | 5 | 0.918 | 0.997 | 0.988 | 0.893 | 0.940 | 0.967 |
| 0 | 2 | 28 | 17 | 0.989 | 1.000 | 0.997 | 0.896 |  | 0.968 |
| 0 | 3 | 59 | 48 | 1.000 | 1.000 | 1.000 | 0.888 |  | 0.969 |
| 0 | 4 | 30 | 19 | 0.990 | 1.000 | 0.998 | 0.773 |  | 0.980 |
| 1 | 0 | 36 | 25 | 1.000 | 1.000 | 1.000 | 0.988 | 0.968 | 0.988 |
| 1 | 1 | 20 | 9 | 0.958 | 0.993 | 0.995 | 0.769 | 0.926 | 0.961 |
| 1 | 2 | 24 | 13 | 0.987 | 0.999 | 0.996 | 0.706 |  | 0.978 |
| 1 | 3 | 35 | 24 | 0.997 | 1.000 | 0.999 | 0.942 | 0.944 | 0.991 |
| 1 | 4 | 37 | 26 | 1.000 | 1.000 | 1.000 | 0.991 | 0.976 | 0.973 |

## SPAR screen team: spar_screen_team_think

226 agents, 67 runs, 9026 steps. Split: 5 folds grouped by run.

### cheating (the set's label: from tried or opened by task)

222 agents scored per step (107 reach the cut), 8820 steps (5133 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.975 [0.953, 0.992] | 0.977 [0.954, 0.995] | 0.978 [0.953, 0.996] | 0.0 [0.0, 1.0] | 104/107 | 6/115 | 0.054 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.967 [0.942, 0.988] | 0.974 [0.945, 0.996] | 0.977 [0.946, 0.998] | 0.0 [0.0, 1.0] | 104/107 | 6/115 | 0.343 |
| attention monitor, binary head (seed 1) | 0.981 [0.962, 0.994] | 0.978 [0.935, 1.000] | 0.980 [0.943, 1.000] | 0.0 [0.0, 0.0] | 105/107 | 5/115 | 0.050 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.975 [0.944, 0.996] | 0.981 [0.940, 1.000] | 0.980 [0.939, 1.000] | 0.0 [0.0, 0.0] | 105/107 | 5/115 | 0.322 |
| linear per-step probe (team_probes) | 0.956 [0.926, 0.977] | 0.899 [0.835, 0.949] | 0.971 [0.945, 0.990] | 1.0 [0.0, 1.0] | 101/107 | 17/115 | 0.037 |
| cumulative mean of the linear probe | 0.973 [0.957, 0.987] | 0.977 [0.954, 0.993] | 0.979 [0.959, 0.994] | 2.0 [1.0, 5.5] | 99/107 | 6/115 | 0.147 |
| GRU over the linear probe scores | 0.967 [0.947, 0.985] | 0.969 [0.944, 0.988] | 0.970 [0.947, 0.990] | 1.0 [1.0, 3.0] | 99/107 | 5/115 | 0.026 |
| turn index alone | 0.813 [0.732, 0.869] | 0.802 [0.693, 0.892] | 0.802 [0.693, 0.892] | 38.0 [32.2, 42.0] | 54/107 | 12/115 | 0.074 |
| TF-IDF on the agent's words so far | 0.989 [0.978, 0.996] | 0.986 [0.969, 0.997] | 0.991 [0.979, 0.999] | 0.0 [0.0, 1.0] | 105/107 | 7/115 | 0.014 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.005, agent last 0.001, agent max 0.002.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.429 | 1.456 | 0.848 |
| cheat | attention monitor (seed 1) | 0.372 | 1.527 | 0.813 |
| cheat | majority level of the training fold | 0.310 | 2.429 | 0.559 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 60 | 56 | 1.000 |  |  | 0.999 |  |  |
| 0 | 1 | 60 | 56 | 1.000 |  |  | 1.000 |  |  |
| 0 | 2 | 40 | 29 | 1.000 |  |  | 0.973 |  |  |
| 0 | 3 | 44 | 33 | 1.000 |  |  | 0.962 |  |  |
| 0 | 4 | 20 | 9 | 1.000 |  |  | 0.903 |  |  |
| 1 | 0 | 24 | 13 | 1.000 |  |  | 1.000 |  |  |
| 1 | 1 | 32 | 21 | 1.000 |  |  | 0.905 |  |  |
| 1 | 2 | 19 | 8 | 0.996 |  |  | 0.945 |  |  |
| 1 | 3 | 32 | 21 | 0.999 |  |  | 0.969 |  |  |
| 1 | 4 | 48 | 37 | 1.000 |  |  | 0.998 |  |  |

## SPAR screen team: spar_screen_team_think_heldout_author

226 agents, 67 runs, 9026 steps. Split: held-out author (train on the other authors, test on one).

### cheating (the set's label: from tried or opened by task)

222 agents scored per step (107 reach the cut), 8820 steps (5133 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.940 [0.900, 0.973] | 0.922 [0.863, 0.972] | 0.929 [0.873, 0.976] | 0.0 [0.0, 2.0] | 78/107 | 2/115 | 0.102 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.949 [0.916, 0.976] | 0.928 [0.874, 0.971] | 0.935 [0.883, 0.975] | 0.0 [0.0, 3.0] | 67/107 | 3/115 | 0.310 |
| attention monitor, binary head (seed 1) | 0.910 [0.864, 0.952] | 0.907 [0.835, 0.963] | 0.897 [0.813, 0.958] | 2.0 [1.0, 3.0] | 38/107 | 10/115 | 0.104 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.960 [0.934, 0.977] | 0.947 [0.897, 0.981] | 0.943 [0.892, 0.979] | 0.0 [0.0, 2.0] | 95/107 | 12/115 | 0.295 |
| linear per-step probe (team_probes) | 0.901 [0.854, 0.933] | 0.830 [0.728, 0.904] | 0.849 [0.749, 0.923] | 0.0 [0.0, 0.0] | 59/107 | 8/115 | 0.073 |
| cumulative mean of the linear probe | 0.925 [0.881, 0.952] | 0.882 [0.801, 0.945] | 0.882 [0.802, 0.945] | 3.0 [0.0, 9.0] | 72/107 | 4/115 | 0.157 |
| GRU over the linear probe scores | 0.946 [0.908, 0.971] | 0.877 [0.794, 0.940] | 0.900 [0.830, 0.951] | 1.0 [0.0, 1.0] | 74/107 | 3/115 | 0.061 |
| turn index alone | 0.752 [0.678, 0.828] | 0.697 [0.571, 0.801] | 0.697 [0.571, 0.801] | 18.0 [14.0, 25.0] | 9/107 | 14/115 | 0.126 |
| TF-IDF on the agent's words so far | 0.940 [0.909, 0.966] | 0.923 [0.859, 0.968] | 0.981 [0.960, 0.994] | 0.0 [0.0, 3.0] | 98/107 | 16/115 | 0.096 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.030, agent last 0.016, agent max 0.033.

### Per held-out author, cheating (the set's label: from tried or opened by task) (step AUROC / agent AUROC from the last step)

| Model | claude-haiku-5-5 held out | deepseek-v4.1-flash held out | gpt-6-luna held out |
|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.976 / 0.950 | 0.957 / 0.972 | 0.909 / 0.943 |
| attention monitor, binary head (seed 1) | 0.851 / 0.873 | 0.962 / 0.978 | 0.957 / 0.951 |
| linear per-step probe (team_probes) | 0.805 / 0.705 | 0.899 / 0.944 | 0.757 / 0.737 |
| cumulative mean of the linear probe | 0.879 / 0.905 | 0.958 / 0.987 | 0.894 / 0.876 |
| GRU over the linear probe scores | 0.826 / 0.720 | 0.943 / 0.967 | 0.891 / 0.864 |
| turn index alone | 0.611 / 0.592 | 0.919 / 0.966 | 0.673 / 0.569 |
| TF-IDF on the agent's words so far | 0.982 / 0.982 | 0.953 / 1.000 | 0.951 / 0.987 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.332 | 1.659 | 0.633 |
| cheat | attention monitor (seed 1) | 0.257 | 1.642 | 0.471 |
| cheat | majority level of the training fold | 0.310 | 2.429 | 0.193 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 34 | 23 | 0.999 |  |  | 0.999 |  |  |
| 0 | 1 | 14 | 3 | 0.986 |  |  | 0.949 |  |  |
| 0 | 2 | 33 | 22 | 0.999 |  |  | 0.982 |  |  |
| 1 | 0 | 56 | 45 | 1.000 |  |  | 0.994 |  |  |
| 1 | 1 | 18 | 7 | 0.999 |  |  | 0.881 |  |  |
| 1 | 2 | 60 | 55 | 1.000 |  |  | 0.998 |  |  |

## SPAR screen team: spar_screen_team_think_ladders

405 agents, 135 runs, 15463 steps. Split: 5 folds grouped by run.

### cheating used or above

405 agents scored per step (103 reach the cut), 15463 steps (3941 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.948 [0.920, 0.967] | 0.959 [0.934, 0.981] | 0.955 [0.930, 0.977] | 1.0 [0.0, 4.0] | 93/103 | 30/302 | 0.055 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.952 [0.934, 0.969] | 0.968 [0.949, 0.984] | 0.960 [0.939, 0.979] | 1.0 [0.0, 6.0] | 92/103 | 25/302 | 0.060 |
| attention monitor, binary head (seed 1) | 0.949 [0.924, 0.970] | 0.965 [0.944, 0.983] | 0.969 [0.950, 0.985] | 1.0 [0.0, 2.0] | 89/103 | 24/302 | 0.072 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.941 [0.915, 0.963] | 0.961 [0.939, 0.979] | 0.957 [0.932, 0.976] | 1.0 [0.0, 4.2] | 84/103 | 24/302 | 0.037 |
| linear per-step probe (team_probes) | 0.900 [0.868, 0.929] | 0.865 [0.805, 0.916] | 0.922 [0.881, 0.956] | 1.0 [0.0, 5.0] | 79/103 | 44/302 | 0.091 |
| cumulative mean of the linear probe | 0.928 [0.903, 0.950] | 0.935 [0.900, 0.963] | 0.941 [0.909, 0.967] | 5.0 [1.0, 11.0] | 73/103 | 18/302 | 0.053 |
| GRU over the linear probe scores | 0.892 [0.843, 0.929] | 0.881 [0.809, 0.934] | 0.891 [0.817, 0.946] | 9.0 [0.8, 14.5] | 68/103 | 23/302 | 0.029 |
| turn index alone | 0.667 [0.587, 0.746] | 0.676 [0.586, 0.768] | 0.676 [0.586, 0.768] | 55.0 [53.0, 69.0] | 9/103 | 29/302 | 0.224 |
| TF-IDF on the agent's words so far | 0.971 [0.957, 0.983] | 0.978 [0.964, 0.990] | 0.981 [0.969, 0.991] | 1.0 [0.0, 1.0] | 97/103 | 17/302 | 0.035 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.001, agent last 0.007, agent max 0.015.

### collusion exploit crossed or above

405 agents scored per step (67 reach the cut), 15463 steps (2156 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.938 [0.900, 0.971] | 0.949 [0.898, 0.986] | 0.957 [0.925, 0.984] | 1.0 [0.0, 3.5] | 63/67 | 35/338 | 0.057 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.948 [0.914, 0.975] | 0.959 [0.923, 0.988] | 0.964 [0.939, 0.987] | 1.0 [0.0, 3.5] | 63/67 | 35/338 | 0.080 |
| attention monitor, binary head (seed 1) | 0.934 [0.897, 0.963] | 0.963 [0.937, 0.984] | 0.962 [0.934, 0.983] | 1.0 [0.0, 2.0] | 61/67 | 30/338 | 0.072 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.938 [0.905, 0.967] | 0.965 [0.940, 0.985] | 0.962 [0.934, 0.984] | 1.0 [0.0, 2.0] | 62/67 | 29/338 | 0.070 |
| linear per-step probe (team_probes) | 0.891 [0.837, 0.936] | 0.871 [0.790, 0.938] | 0.911 [0.850, 0.959] | 3.0 [0.0, 6.0] | 59/67 | 48/338 | 0.129 |
| cumulative mean of the linear probe | 0.842 [0.756, 0.923] | 0.911 [0.831, 0.971] | 0.902 [0.821, 0.965] | 4.0 [2.0, 12.0] | 45/67 | 22/338 | 0.067 |
| GRU over the linear probe scores | 0.895 [0.846, 0.936] | 0.885 [0.806, 0.945] | 0.893 [0.828, 0.946] | 6.0 [0.0, 12.8] | 38/67 | 28/338 | 0.034 |
| turn index alone | 0.730 [0.614, 0.817] | 0.642 [0.511, 0.762] | 0.642 [0.511, 0.762] | 39.0 [36.0, 47.0] | 17/67 | 36/338 | 0.303 |
| TF-IDF on the agent's words so far | 0.939 [0.897, 0.970] | 0.955 [0.895, 0.993] | 0.956 [0.907, 0.989] | 2.0 [0.0, 10.0] | 61/67 | 22/338 | 0.053 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.005, agent last 0.014, agent max 0.005.

### coordination posted or above

376 agents scored per step (335 reach the cut), 14584 steps (9091 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.970 [0.958, 0.980] | 0.944 [0.903, 0.973] | 0.948 [0.902, 0.979] | 1.0 [0.0, 1.0] | 302/335 | 3/41 | 0.042 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.959 [0.945, 0.972] | 0.922 [0.866, 0.965] | 0.926 [0.869, 0.971] | 1.0 [0.0, 1.0] | 276/335 | 6/41 | 0.176 |
| attention monitor, binary head (seed 1) | 0.975 [0.964, 0.985] | 0.952 [0.915, 0.978] | 0.956 [0.921, 0.981] | 1.0 [0.0, 1.0] | 312/335 | 5/41 | 0.056 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.953 [0.929, 0.973] | 0.927 [0.864, 0.971] | 0.930 [0.868, 0.972] | 1.0 [0.0, 3.0] | 229/335 | 3/41 | 0.160 |
| linear per-step probe (team_probes) | 0.969 [0.960, 0.978] | 0.948 [0.918, 0.971] | 0.971 [0.952, 0.985] | 1.0 [1.0, 1.0] | 323/335 | 10/41 | 0.023 |
| cumulative mean of the linear probe | 0.955 [0.944, 0.967] | 0.929 [0.891, 0.957] | 0.928 [0.889, 0.956] | 4.0 [2.0, 7.0] | 265/335 | 4/41 | 0.277 |
| GRU over the linear probe scores | 0.973 [0.961, 0.984] | 0.961 [0.937, 0.979] | 0.958 [0.933, 0.977] | 1.0 [1.0, 2.0] | 314/335 | 8/41 | 0.010 |
| turn index alone | 0.846 [0.815, 0.880] | 0.782 [0.687, 0.861] | 0.782 [0.687, 0.861] | 28.0 [12.0, 35.0] | 108/335 | 5/41 | 0.119 |
| TF-IDF on the agent's words so far | 0.943 [0.925, 0.959] | 0.895 [0.828, 0.946] | 0.889 [0.822, 0.942] | 6.0 [2.0, 14.0] | 249/335 | 4/41 | 0.051 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.005, agent last 0.008, agent max 0.008.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.304 | 1.180 | 0.603 |
| cheat | attention monitor (seed 1) | 0.304 | 1.202 | 0.609 |
| cheat | majority level of the training fold | 0.410 | 1.802 | 0.303 |
| coord | attention monitor (seed 0) | 0.486 | 1.116 | 0.855 |
| coord | attention monitor (seed 1) | 0.509 | 1.077 | 0.873 |
| coord | majority level of the training fold | 0.526 | 0.983 | 0.588 |
| coop | attention monitor (seed 0) | 0.768 | 0.378 | 0.909 |
| coop | attention monitor (seed 1) | 0.780 | 0.343 | 0.877 |
| coop | majority level of the training fold | 0.783 | 0.499 | 0.881 |
| collab | attention monitor (seed 0) | 0.844 | 0.267 | 0.915 |
| collab | attention monitor (seed 1) | 0.832 | 0.254 | 0.901 |
| collab | majority level of the training fold | 0.854 | 0.358 | 0.881 |
| collude | attention monitor (seed 0) | 0.812 | 0.528 | 0.876 |
| collude | attention monitor (seed 1) | 0.805 | 0.536 | 0.865 |
| collude | majority level of the training fold | 0.815 | 0.746 | 0.861 |
| scheme | attention monitor (seed 0) | 0.973 | 0.054 | 1.000 |
| scheme | attention monitor (seed 1) | 0.973 | 0.054 | 1.000 |
| scheme | majority level of the training fold | 0.973 | 0.054 | 1.000 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 33 | 22 | 0.997 | 0.999 | 0.996 | 0.956 | 0.991 | 0.925 |
| 0 | 1 | 55 | 44 | 1.000 | 1.000 | 0.999 | 0.998 | 0.998 | 0.983 |
| 0 | 2 | 17 | 6 | 0.977 | 0.994 | 0.974 | 0.963 | 0.980 | 0.948 |
| 0 | 3 | 32 | 21 | 0.994 | 1.000 | 0.990 | 0.987 | 0.994 | 0.976 |
| 0 | 4 | 26 | 15 | 0.991 | 0.996 | 0.991 | 0.958 | 0.951 | 0.945 |
| 1 | 0 | 27 | 16 | 0.994 | 0.996 | 0.987 | 0.923 | 0.965 | 0.917 |
| 1 | 1 | 25 | 14 | 0.988 | 0.998 | 0.991 | 0.956 | 0.986 | 0.970 |
| 1 | 2 | 56 | 45 | 1.000 | 1.000 | 0.999 | 0.943 | 0.997 | 0.966 |
| 1 | 3 | 26 | 15 | 0.990 | 0.999 | 0.992 | 0.952 | 0.935 | 0.942 |
| 1 | 4 | 60 | 52 | 0.999 | 1.000 | 0.997 | 0.970 | 0.994 | 0.962 |

## SPAR screen team: spar_screen_team_think_ladders_heldout_author

405 agents, 135 runs, 15463 steps. Split: held-out author (train on the other authors, test on one).

### cheating used or above

405 agents scored per step (103 reach the cut), 15463 steps (3941 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.820 [0.771, 0.862] | 0.812 [0.750, 0.873] | 0.828 [0.771, 0.886] | 3.0 [0.0, 7.0] | 59/103 | 44/302 | 0.298 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.785 [0.726, 0.845] | 0.800 [0.735, 0.863] | 0.806 [0.742, 0.867] | 12.0 [6.0, 28.0] | 53/103 | 50/302 | 0.185 |
| attention monitor, binary head (seed 1) | 0.795 [0.741, 0.854] | 0.803 [0.744, 0.865] | 0.807 [0.747, 0.869] | 14.0 [5.8, 25.0] | 52/103 | 45/302 | 0.302 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.784 [0.734, 0.841] | 0.802 [0.739, 0.862] | 0.795 [0.733, 0.856] | 9.0 [4.0, 25.0] | 45/103 | 49/302 | 0.137 |
| linear per-step probe (team_probes) | 0.690 [0.638, 0.734] | 0.660 [0.592, 0.732] | 0.782 [0.707, 0.852] | 2.0 [0.0, 15.8] | 58/103 | 76/302 | 0.130 |
| cumulative mean of the linear probe | 0.673 [0.599, 0.741] | 0.775 [0.703, 0.842] | 0.776 [0.703, 0.843] | 4.0 [1.0, 19.0] | 53/103 | 27/302 | 0.145 |
| GRU over the linear probe scores | 0.754 [0.703, 0.810] | 0.775 [0.713, 0.831] | 0.808 [0.748, 0.860] | 10.0 [4.2, 19.5] | 62/103 | 34/302 | 0.130 |
| turn index alone | 0.702 [0.613, 0.779] | 0.684 [0.590, 0.766] | 0.684 [0.590, 0.766] | 50.0 [33.0, 68.0] | 29/103 | 36/302 | 0.228 |
| TF-IDF on the agent's words so far | 0.867 [0.806, 0.927] | 0.863 [0.804, 0.919] | 0.932 [0.898, 0.960] | 0.0 [0.0, 1.0] | 71/103 | 43/302 | 0.129 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.024, agent last 0.010, agent max 0.021.

### collusion exploit crossed or above

405 agents scored per step (67 reach the cut), 15463 steps (2156 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.782 [0.693, 0.843] | 0.762 [0.658, 0.845] | 0.756 [0.661, 0.839] | 1.0 [0.0, 3.0] | 33/67 | 65/338 | 0.229 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.806 [0.724, 0.862] | 0.795 [0.695, 0.878] | 0.805 [0.714, 0.880] | 1.0 [0.0, 3.5] | 39/67 | 58/338 | 0.050 |
| attention monitor, binary head (seed 1) | 0.784 [0.731, 0.837] | 0.778 [0.707, 0.842] | 0.789 [0.723, 0.853] | 5.0 [1.0, 12.0] | 33/67 | 65/338 | 0.150 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.787 [0.735, 0.840] | 0.770 [0.696, 0.837] | 0.782 [0.716, 0.847] | 5.0 [1.0, 14.5] | 34/67 | 67/338 | 0.126 |
| linear per-step probe (team_probes) | 0.581 [0.518, 0.659] | 0.422 [0.337, 0.514] | 0.629 [0.516, 0.736] | 5.0 [1.8, 14.0] | 48/67 | 156/338 | 0.094 |
| cumulative mean of the linear probe | 0.573 [0.478, 0.678] | 0.500 [0.385, 0.650] | 0.449 [0.330, 0.594] | 2.0 [0.0, 8.2] | 22/67 | 91/338 | 0.111 |
| GRU over the linear probe scores | 0.488 [0.402, 0.594] | 0.402 [0.305, 0.522] | 0.506 [0.399, 0.604] | 10.0 [2.0, 15.0] | 19/67 | 160/338 | 0.122 |
| turn index alone | 0.709 [0.562, 0.808] | 0.618 [0.460, 0.741] | 0.618 [0.460, 0.741] | 42.5 [34.8, 55.0] | 20/67 | 43/338 | 0.299 |
| TF-IDF on the agent's words so far | 0.846 [0.791, 0.908] | 0.910 [0.859, 0.956] | 0.895 [0.841, 0.942] | 0.0 [0.0, 14.0] | 39/67 | 30/338 | 0.094 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.002, agent last 0.017, agent max 0.033.

### coordination posted or above

376 agents scored per step (335 reach the cut), 14584 steps (9091 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.861 [0.828, 0.894] | 0.855 [0.778, 0.935] | 0.848 [0.767, 0.934] | 1.0 [1.0, 5.0] | 243/335 | 9/41 | 0.163 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.821 [0.783, 0.864] | 0.777 [0.696, 0.876] | 0.718 [0.630, 0.825] | 1.0 [1.0, 7.5] | 132/335 | 10/41 | 0.133 |
| attention monitor, binary head (seed 1) | 0.877 [0.850, 0.904] | 0.619 [0.519, 0.716] | 0.680 [0.559, 0.812] | 1.0 [0.0, 5.0] | 311/335 | 28/41 | 0.127 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.828 [0.798, 0.861] | 0.491 [0.390, 0.606] | 0.545 [0.440, 0.662] | 1.0 [0.0, 5.0] | 284/335 | 31/41 | 0.321 |
| linear per-step probe (team_probes) | 0.758 [0.714, 0.811] | 0.789 [0.671, 0.883] | 0.824 [0.701, 0.917] | 3.0 [1.0, 15.2] | 56/335 | 5/41 | 0.140 |
| cumulative mean of the linear probe | 0.801 [0.761, 0.850] | 0.782 [0.665, 0.874] | 0.776 [0.661, 0.867] | 11.0 [0.0, 21.0] | 41/335 | 4/41 | 0.212 |
| GRU over the linear probe scores | 0.812 [0.771, 0.861] | 0.812 [0.710, 0.896] | 0.815 [0.708, 0.901] | 1.0 [0.0, 12.0] | 17/335 | 3/41 | 0.128 |
| turn index alone | 0.806 [0.769, 0.849] | 0.766 [0.664, 0.848] | 0.766 [0.664, 0.848] | 20.0 [4.0, 31.0] | 67/335 | 5/41 | 0.150 |
| TF-IDF on the agent's words so far | 0.746 [0.698, 0.791] | 0.783 [0.673, 0.877] | 0.773 [0.660, 0.870] | 11.5 [0.0, 31.8] | 90/335 | 3/41 | 0.305 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.017, agent last 0.236, agent max 0.168.

### Per held-out author, cheating used or above (step AUROC / agent AUROC from the last step)

| Model | ds held out | haiku held out | luna held out |
|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.779 / 0.834 | 0.927 / 0.834 | 0.893 / 0.848 |
| attention monitor, binary head (seed 1) | 0.703 / 0.629 | 0.799 / 0.804 | 0.900 / 0.894 |
| linear per-step probe (team_probes) | 0.677 / 0.659 | 0.791 / 0.605 | 0.764 / 0.735 |
| cumulative mean of the linear probe | 0.722 / 0.749 | 0.941 / 0.926 | 0.905 / 0.894 |
| GRU over the linear probe scores | 0.720 / 0.794 | 0.903 / 0.804 | 0.858 / 0.837 |
| turn index alone | 0.667 / 0.637 | 0.536 / 0.573 | 0.553 / 0.424 |
| TF-IDF on the agent's words so far | 0.858 / 0.872 | 0.958 / 0.925 | 0.925 / 0.910 |

### Per held-out author, collusion exploit crossed or above (step AUROC / agent AUROC from the last step)

| Model | ds held out | haiku held out | luna held out |
|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.765 / 0.839 | 0.927 / 0.780 | 0.880 / 0.886 |
| attention monitor, binary head (seed 1) | 0.627 / 0.485 | 0.966 / 0.933 | 0.923 / 0.910 |
| linear per-step probe (team_probes) | 0.736 / 0.702 | 0.709 / 0.364 | 0.748 / 0.573 |
| cumulative mean of the linear probe | 0.828 / 0.904 | 0.909 / 0.826 | 0.851 / 0.776 |
| GRU over the linear probe scores | 0.715 / 0.766 | 0.788 / 0.535 | 0.748 / 0.633 |
| turn index alone | 0.762 / 0.641 | 0.646 / 0.526 | 0.513 / 0.334 |
| TF-IDF on the agent's words so far | 0.776 / 0.803 | 0.990 / 0.988 | 0.946 / 0.953 |

### Per held-out author, coordination posted or above (step AUROC / agent AUROC from the last step)

| Model | ds held out | haiku held out | luna held out |
|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.858 / 0.684 | 0.972 / 0.942 | 0.964 / 0.985 |
| attention monitor, binary head (seed 1) | 0.822 / 0.702 | 0.961 / 0.891 | 0.948 / 0.995 |
| linear per-step probe (team_probes) | 0.711 / 0.368 | 0.935 / 0.829 | 0.951 / 0.919 |
| cumulative mean of the linear probe | 0.783 / 0.335 | 0.873 / 0.726 | 0.949 / 0.932 |
| GRU over the linear probe scores | 0.652 / 0.351 | 0.942 / 0.853 | 0.970 / 1.000 |
| turn index alone | 0.865 / 0.561 | 0.854 / 0.610 | 0.870 / 0.985 |
| TF-IDF on the agent's words so far | 0.861 / 0.627 | 0.868 / 0.690 | 0.906 / 0.929 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.212 | 1.696 | 0.224 |
| cheat | attention monitor (seed 1) | 0.227 | 1.637 | 0.393 |
| cheat | majority level of the training fold | 0.410 | 1.802 | 0.235 |
| coord | attention monitor (seed 0) | 0.494 | 1.017 | 0.731 |
| coord | attention monitor (seed 1) | 0.442 | 1.370 | 0.590 |
| coord | majority level of the training fold | 0.526 | 0.983 | 0.588 |
| coop | attention monitor (seed 0) | 0.741 | 0.481 | 0.840 |
| coop | attention monitor (seed 1) | 0.728 | 0.543 | 0.856 |
| coop | majority level of the training fold | 0.783 | 0.499 | 0.881 |
| collab | attention monitor (seed 0) | 0.832 | 0.351 | 0.855 |
| collab | attention monitor (seed 1) | 0.788 | 0.447 | 0.863 |
| collab | majority level of the training fold | 0.854 | 0.358 | 0.881 |
| collude | attention monitor (seed 0) | 0.795 | 0.723 | 0.824 |
| collude | attention monitor (seed 1) | 0.765 | 0.842 | 0.845 |
| collude | majority level of the training fold | 0.815 | 0.746 | 0.861 |
| scheme | attention monitor (seed 0) | 0.973 | 0.054 | 1.000 |
| scheme | attention monitor (seed 1) | 0.973 | 0.054 | 1.000 |
| scheme | majority level of the training fold | 0.973 | 0.054 | 1.000 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 16 | 5 | 0.995 | 1.000 | 0.990 | 0.989 | 0.996 | 0.921 |
| 0 | 1 | 50 | 39 | 1.000 | 1.000 | 1.000 | 0.972 | 0.997 | 0.969 |
| 0 | 2 | 25 | 14 | 0.990 | 0.999 | 0.981 | 0.955 | 0.987 | 0.877 |
| 1 | 0 | 39 | 28 | 1.000 | 1.000 | 0.999 | 0.989 | 0.999 | 0.987 |
| 1 | 1 | 47 | 36 | 1.000 | 1.000 | 0.999 | 0.975 | 0.963 | 0.992 |
| 1 | 2 | 31 | 20 | 0.996 | 1.000 | 0.994 | 0.956 | 0.956 | 0.915 |
