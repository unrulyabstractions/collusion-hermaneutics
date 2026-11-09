# Attention monitor over the reader's think tokens

Updated 2026-11-09 01:54 PST. Every number below is read from the CSVs in this folder; the held-out scores per step are saved in step_scores_<set>.csv.gz, so each AUROC can be recomputed.

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

## Labels

| Cut | Per-step label | Onset |
|---|---|---|
| cheating used or above (board_seed) | level reached at that call >= used | estimate (team_probes used_onset_step: first write of planted content, else the opened call) |
| collusion exploit crossed or above (board_seed) | level reached at that call >= exploit crossed | estimate (team_probes collude_onset_step proxy) |
| coordination posted or above (board_seed) | level reached at that call >= posted | direct (first read, post, discussed or solicited call from the board events) |
| cheating, the set's label (controls_600) | from the turn of the first tool call at or above the task's floor (tried for software engineering, chess and creative writing, opened otherwise) | direct (labels.csv onset_turn; equal to the step of onset_call for all 336 agents with an onset) |

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

Permutation null (33 draws), attention monitor seed 0:

| Metric | Real | Null mean | Null 95th percentile | p |
|---|---|---|---|---|
| step AUROC | 0.983 | 0.679 | 0.698 | 0.029 |
| agent AUROC (last) | 0.985 | 0.508 | 0.550 | 0.029 |
| agent AUROC (max) | 0.989 | 0.508 | 0.552 | 0.029 |

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

176 agents scored per step (83 reach the cut), 6082 steps (1872 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.898 [0.840, 0.948] | 0.896 [0.825, 0.950] | 0.879 [0.802, 0.942] | 1.0 [0.0, 1.0] | 62/83 | 13/93 | 0.106 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.914 [0.863, 0.958] | 0.889 [0.810, 0.951] | 0.892 [0.809, 0.949] | 1.0 [0.0, 2.0] | 70/83 | 15/93 | 0.090 |
| attention monitor, binary head (seed 1) | 0.890 [0.824, 0.945] | 0.915 [0.863, 0.958] | 0.914 [0.859, 0.960] | 0.0 [0.0, 1.0] | 70/83 | 12/93 | 0.111 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.914 [0.862, 0.959] | 0.921 [0.863, 0.965] | 0.923 [0.861, 0.967] | 1.0 [0.0, 2.0] | 68/83 | 12/93 | 0.078 |
| linear per-step probe (team_probes) | 0.898 [0.849, 0.944] | 0.885 [0.821, 0.942] | 0.899 [0.840, 0.950] | 1.0 [0.0, 2.0] | 70/83 | 19/93 | 0.095 |
| cumulative mean of the linear probe | 0.876 [0.820, 0.924] | 0.861 [0.795, 0.920] | 0.856 [0.784, 0.919] | 8.0 [5.0, 11.5] | 47/83 | 11/93 | 0.133 |
| GRU over the linear probe scores | 0.896 [0.844, 0.941] | 0.899 [0.848, 0.948] | 0.897 [0.835, 0.949] | 1.0 [0.0, 2.0] | 64/83 | 18/93 | 0.047 |
| turn index alone | 0.638 [0.575, 0.705] | 0.440 [0.334, 0.556] | 0.440 [0.334, 0.556] | 44.5 [37.5, 53.2] | 16/83 | 25/93 | 0.183 |
| TF-IDF on the agent's words so far | 0.924 [0.864, 0.966] | 0.925 [0.859, 0.971] | 0.924 [0.860, 0.970] | 1.0 [0.0, 1.5] | 63/83 | 10/93 | 0.086 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.008, agent last 0.019, agent max 0.035.

### coordination posted or above

177 agents scored per step (38 reach the cut), 6086 steps (552 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.917 [0.778, 0.972] | 0.890 [0.812, 0.948] | 0.887 [0.802, 0.947] | 0.0 [0.0, 1.0] | 32/38 | 36/139 | 0.042 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.843 [0.606, 0.950] | 0.883 [0.792, 0.949] | 0.885 [0.787, 0.959] | 0.0 [0.0, 1.0] | 31/38 | 37/139 | 0.217 |
| attention monitor, binary head (seed 1) | 0.798 [0.464, 0.964] | 0.824 [0.703, 0.919] | 0.823 [0.681, 0.925] | 0.0 [0.0, 1.0] | 29/38 | 25/139 | 0.045 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.891 [0.727, 0.961] | 0.865 [0.783, 0.929] | 0.857 [0.762, 0.930] | 0.0 [0.0, 1.0] | 29/38 | 24/139 | 0.210 |
| linear per-step probe (team_probes) | 0.802 [0.522, 0.935] | 0.795 [0.676, 0.893] | 0.757 [0.614, 0.878] | 0.0 [0.0, 1.0] | 28/38 | 44/139 | 0.145 |
| cumulative mean of the linear probe | 0.756 [0.434, 0.895] | 0.518 [0.335, 0.692] | 0.471 [0.274, 0.675] | 2.0 [0.0, 5.0] | 13/38 | 21/139 | 0.113 |
| GRU over the linear probe scores | 0.791 [0.488, 0.947] | 0.796 [0.677, 0.899] | 0.745 [0.571, 0.875] | 0.0 [0.0, 0.5] | 27/38 | 85/139 | 0.016 |
| turn index alone | 0.637 [0.445, 0.733] | 0.546 [0.419, 0.670] | 0.546 [0.419, 0.670] | 46.0 [44.0, 50.0] | 5/38 | 28/139 | 0.376 |
| TF-IDF on the agent's words so far | 0.758 [0.431, 0.905] | 0.522 [0.374, 0.665] | 0.496 [0.346, 0.658] | 0.0 [0.0, 0.0] | 6/38 | 19/139 | 0.166 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.119, agent last 0.066, agent max 0.064.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.243 | 1.661 | 0.630 |
| cheat | attention monitor (seed 1) | 0.299 | 1.514 | 0.652 |
| cheat | majority level of the training fold | 0.367 | 3.130 | 0.245 |
| coord | attention monitor (seed 0) | 0.452 | 0.593 | 0.372 |
| coord | attention monitor (seed 1) | 0.497 | 0.565 | 0.489 |
| coord | majority level of the training fold | 0.667 | 0.384 | 0.765 |
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
| 0 | 0 | 30 | 19 | 1.000 |  | 0.998 | 0.944 |  | 0.930 |
| 0 | 1 | 43 | 32 | 1.000 |  | 1.000 | 0.989 |  | 0.947 |
| 0 | 2 | 16 | 5 | 0.991 |  | 0.992 | 0.908 |  | 0.997 |
| 0 | 3 | 24 | 13 | 1.000 |  | 0.997 | 0.975 |  | 0.977 |
| 0 | 4 | 35 | 24 | 1.000 |  | 0.999 | 0.991 |  | 0.877 |
| 1 | 0 | 60 | 49 | 1.000 |  | 0.999 | 0.987 |  | 0.983 |
| 1 | 1 | 28 | 17 | 0.999 |  | 0.998 | 0.820 |  | 0.911 |
| 1 | 2 | 60 | 56 | 1.000 |  | 1.000 | 0.972 |  | 1.000 |
| 1 | 3 | 15 | 4 | 0.971 |  | 0.947 | 0.898 |  | 0.954 |
| 1 | 4 | 19 | 8 | 0.993 |  | 0.993 | 0.827 |  | 0.818 |

## Solo design: solo_design_think_heldout_author

177 agents, 59 runs, 6086 steps. Split: held-out author (train on the other authors, test on one).

### cheating used or above

176 agents scored per step (83 reach the cut), 6082 steps (1872 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.680 [0.621, 0.738] | 0.558 [0.451, 0.660] | 0.536 [0.430, 0.638] | 10.0 [1.0, 21.8] | 38/83 | 38/93 | 0.275 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.774 [0.715, 0.837] | 0.772 [0.666, 0.855] | 0.804 [0.704, 0.884] | 14.0 [3.0, 22.0] | 64/83 | 27/93 | 0.094 |
| attention monitor, binary head (seed 1) | 0.710 [0.659, 0.764] | 0.397 [0.293, 0.505] | 0.361 [0.243, 0.471] | 1.0 [0.8, 1.0] | 16/83 | 42/93 | 0.113 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.799 [0.735, 0.864] | 0.765 [0.666, 0.848] | 0.763 [0.654, 0.851] | 16.0 [9.0, 26.0] | 33/83 | 22/93 | 0.132 |
| linear per-step probe (team_probes) | 0.667 [0.611, 0.726] | 0.388 [0.289, 0.490] | 0.266 [0.163, 0.377] | 2.0 [1.0, 2.0] | 17/83 | 37/93 | 0.164 |
| cumulative mean of the linear probe | 0.779 [0.705, 0.856] | 0.682 [0.548, 0.798] | 0.678 [0.544, 0.796] | 21.0 [16.8, 32.8] | 16/83 | 14/93 | 0.056 |
| GRU over the linear probe scores | 0.716 [0.644, 0.799] | 0.650 [0.529, 0.757] | 0.653 [0.532, 0.765] | 0.0 [0.0, 0.0] | 5/83 | 8/93 | 0.235 |
| turn index alone | 0.674 [0.616, 0.741] | 0.496 [0.377, 0.615] | 0.496 [0.377, 0.615] | 44.0 [32.0, 45.0] | 17/83 | 19/93 | 0.195 |
| TF-IDF on the agent's words so far | 0.792 [0.728, 0.856] | 0.768 [0.650, 0.878] | 0.766 [0.649, 0.875] | 3.0 [0.0, 4.0] | 17/83 | 6/93 | 0.121 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.030, agent last 0.161, agent max 0.175.

### coordination posted or above

177 agents scored per step (38 reach the cut), 6086 steps (552 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.757 [0.666, 0.831] | 0.694 [0.580, 0.802] | 0.696 [0.584, 0.811] | 1.0 [0.8, 11.0] | 32/38 | 77/139 | 0.079 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.626 [0.534, 0.760] | 0.630 [0.503, 0.739] | 0.617 [0.501, 0.721] | 5.0 [1.0, 14.0] | 19/38 | 59/139 | 0.189 |
| attention monitor, binary head (seed 1) | 0.392 [0.279, 0.619] | 0.622 [0.448, 0.796] | 0.621 [0.444, 0.798] | 0.0 [0.0, 0.0] | 21/38 | 54/139 | 0.152 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.539 [0.448, 0.707] | 0.576 [0.446, 0.697] | 0.543 [0.413, 0.667] | 4.0 [1.0, 11.0] | 13/38 | 76/139 | 0.106 |
| linear per-step probe (team_probes) | 0.794 [0.642, 0.872] | 0.711 [0.619, 0.799] | 0.693 [0.580, 0.809] | 1.0 [0.0, 10.8] | 34/38 | 93/139 | 0.109 |
| cumulative mean of the linear probe | 0.802 [0.589, 0.899] | 0.603 [0.451, 0.728] | 0.560 [0.420, 0.679] | 7.0 [0.0, 11.0] | 9/38 | 26/139 | 0.043 |
| GRU over the linear probe scores | 0.707 [0.629, 0.831] | 0.580 [0.468, 0.692] | 0.438 [0.326, 0.556] | 2.5 [0.0, 6.5] | 4/38 | 12/139 | 0.121 |
| turn index alone | 0.632 [0.536, 0.697] | 0.594 [0.454, 0.720] | 0.594 [0.454, 0.720] | 0.0 [0.0, 24.5] | 7/38 | 28/139 | 0.395 |
| TF-IDF on the agent's words so far | 0.436 [0.369, 0.526] | 0.515 [0.341, 0.677] | 0.505 [0.328, 0.674] | 0.0 [0.0, 0.0] | 7/38 | 14/139 | 0.176 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.366, agent last 0.072, agent max 0.075.

### Per held-out author, cheating used or above (step AUROC / agent AUROC from the last step)

| Model | ds held out | luna held out |
|---|---|---|
| attention monitor, binary head (seed 0) | 0.629 / 0.500 | 0.763 / 0.785 |
| attention monitor, binary head (seed 1) | 0.621 / 0.477 | 0.770 / 0.739 |
| linear per-step probe (team_probes) | 0.618 / 0.427 | 0.751 / 0.629 |
| cumulative mean of the linear probe | 0.616 / 0.268 | 0.657 / 0.575 |
| GRU over the linear probe scores | 0.614 / 0.269 | 0.711 / 0.639 |
| turn index alone | 0.614 / 0.262 | 0.539 / 0.296 |
| TF-IDF on the agent's words so far | 0.806 / 0.814 | 0.896 / 0.849 |

### Per held-out author, coordination posted or above (step AUROC / agent AUROC from the last step)

| Model | ds held out | luna held out |
|---|---|---|
| attention monitor, binary head (seed 0) | 0.741 / 0.826 | 0.692 / 0.545 |
| attention monitor, binary head (seed 1) | 0.687 / 0.838 | 0.660 / 0.487 |
| linear per-step probe (team_probes) | 0.684 / 0.837 | 0.832 / 0.488 |
| cumulative mean of the linear probe | 0.710 / 0.664 | 0.851 / 0.778 |
| GRU over the linear probe scores | 0.836 / 0.648 | 0.669 / 0.471 |
| turn index alone | 0.623 / 0.524 | 0.751 / 0.554 |
| TF-IDF on the agent's words so far | 0.540 / 0.682 | 0.472 / 0.285 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.028 | 2.379 | 0.425 |
| cheat | attention monitor (seed 1) | 0.023 | 2.395 | 0.275 |
| cheat | majority level of the training fold | 0.124 | 4.531 | 0.100 |
| coord | attention monitor (seed 0) | 0.525 | 0.554 | 0.446 |
| coord | attention monitor (seed 1) | 0.158 | 1.102 | 0.146 |
| coord | majority level of the training fold | 0.667 | 0.384 | 0.765 |
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
| 0 | 0 | 39 | 28 | 1.000 |  | 1.000 | 0.982 |  | 0.933 |
| 0 | 1 | 20 | 9 | 0.999 |  | 0.975 | 0.887 |  | 0.287 |
| 0 | 2 | 21 | 10 | 0.997 |  | 0.993 | 0.967 |  | 0.936 |
| 1 | 0 | 12 | 1 | 0.913 |  | 0.991 | 0.926 |  | 0.670 |
| 1 | 1 | 19 | 8 | 0.994 |  | 0.990 | 0.929 |  | 0.616 |
| 1 | 2 | 24 | 13 | 0.999 |  | 0.995 | 0.947 |  | 0.776 |

## SPAR mixed: spar_mixed_41_think

209 agents, 51 runs, 12373 steps. Split: 5 folds grouped by run.

### cheating (the set's label: from tried or opened by task)

205 agents scored per step (39 reach the cut), 12065 steps (1452 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.815 [0.741, 0.874] | 0.702 [0.588, 0.808] | 0.713 [0.612, 0.803] | 4.5 [0.8, 35.0] | 20/39 | 38/166 | 0.086 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.792 [0.710, 0.859] | 0.660 [0.534, 0.776] | 0.704 [0.593, 0.799] | 3.0 [1.0, 15.0] | 17/39 | 26/166 | 0.042 |
| attention monitor, binary head (seed 1) | 0.820 [0.748, 0.872] | 0.737 [0.631, 0.839] | 0.769 [0.678, 0.846] | 0.5 [0.0, 1.0] | 18/39 | 17/166 | 0.095 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.827 [0.745, 0.883] | 0.725 [0.619, 0.817] | 0.755 [0.662, 0.836] | 1.0 [0.0, 1.5] | 15/39 | 18/166 | 0.033 |
| linear per-step probe (team_probes) | 0.787 [0.719, 0.842] | 0.671 [0.548, 0.782] | 0.693 [0.591, 0.786] | 1.0 [0.8, 4.0] | 16/39 | 25/166 | 0.139 |
| cumulative mean of the linear probe | 0.791 [0.712, 0.857] | 0.648 [0.514, 0.767] | 0.639 [0.498, 0.763] | 19.0 [11.5, 42.0] | 11/39 | 16/166 | 0.069 |
| GRU over the linear probe scores | 0.765 [0.670, 0.848] | 0.656 [0.545, 0.762] | 0.654 [0.547, 0.747] | 6.0 [0.0, 59.0] | 9/39 | 10/166 | 0.050 |
| turn index alone | 0.451 [0.373, 0.527] | 0.484 [0.371, 0.583] | 0.480 [0.380, 0.592] |  | 0/39 | 5/166 | 0.387 |
| TF-IDF on the agent's words so far | 0.888 [0.830, 0.932] | 0.839 [0.770, 0.911] | 0.836 [0.763, 0.914] | 1.0 [0.0, 11.0] | 20/39 | 17/166 | 0.046 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.006, agent last 0.035, agent max 0.057.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.354 | 0.900 | 0.405 |
| cheat | attention monitor (seed 1) | 0.225 | 1.053 | 0.316 |
| cheat | majority level of the training fold | 0.761 | 0.536 | 0.663 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 17 | 6 | 0.958 |  |  | 0.847 |  |  |
| 0 | 1 | 20 | 12 | 0.972 |  |  | 0.923 |  |  |
| 0 | 2 | 35 | 24 | 0.999 |  |  | 0.977 |  |  |
| 0 | 3 | 40 | 29 | 0.999 |  |  | 0.977 |  |  |
| 0 | 4 | 22 | 11 | 0.981 |  |  | 0.853 |  |  |
| 1 | 0 | 11 | 0 | 0.808 |  |  | 0.803 |  |  |
| 1 | 1 | 26 | 15 | 0.984 |  |  | 0.970 |  |  |
| 1 | 2 | 34 | 23 | 0.999 |  |  | 0.960 |  |  |
| 1 | 3 | 30 | 19 | 0.997 |  |  | 0.947 |  |  |
| 1 | 4 | 16 | 5 | 0.932 |  |  | 0.884 |  |  |

## SPAR mixed: spar_mixed_41_think_heldout_author

209 agents, 51 runs, 12373 steps. Split: held-out author (train on the other authors, test on one).

### cheating (the set's label: from tried or opened by task)

205 agents scored per step (39 reach the cut), 12065 steps (1452 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.779 [0.671, 0.850] | 0.651 [0.536, 0.753] | 0.707 [0.588, 0.815] | 2.5 [1.2, 3.8] | 14/39 | 16/166 | 0.145 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.805 [0.730, 0.868] | 0.628 [0.489, 0.758] | 0.656 [0.524, 0.773] | 3.0 [0.5, 13.5] | 15/39 | 16/166 | 0.028 |
| attention monitor, binary head (seed 1) | 0.579 [0.447, 0.680] | 0.630 [0.505, 0.746] | 0.660 [0.554, 0.755] | 25.0 [3.5, 36.0] | 11/39 | 19/166 | 0.022 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.503 [0.408, 0.598] | 0.479 [0.399, 0.571] | 0.531 [0.424, 0.644] | 42.0 [17.0, 54.0] | 17/39 | 61/166 | 0.022 |
| linear per-step probe (team_probes) | 0.772 [0.670, 0.841] | 0.685 [0.554, 0.796] | 0.709 [0.598, 0.807] | 14.0 [1.0, 27.0] | 17/39 | 17/166 | 0.159 |
| cumulative mean of the linear probe | 0.780 [0.664, 0.858] | 0.718 [0.617, 0.806] | 0.672 [0.553, 0.773] | 23.0 [4.0, 34.8] | 14/39 | 12/166 | 0.091 |
| GRU over the linear probe scores | 0.733 [0.625, 0.806] | 0.718 [0.613, 0.819] | 0.688 [0.557, 0.811] | 33.0 [6.0, 39.5] | 14/39 | 14/166 | 0.121 |
| turn index alone | 0.460 [0.365, 0.532] | 0.666 [0.561, 0.761] | 0.603 [0.485, 0.714] |  | 0/39 | 5/166 | 0.497 |
| TF-IDF on the agent's words so far | 0.818 [0.750, 0.873] | 0.740 [0.628, 0.839] | 0.724 [0.594, 0.837] | 1.0 [0.0, 6.0] | 18/39 | 13/166 | 0.053 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.200, agent last 0.021, agent max 0.047.

### Per held-out author, cheating (the set's label: from tried or opened by task) (step AUROC / agent AUROC from the last step)

| Model | deepseek-flash-openrouter held out | gpt-6-luna-openrouter held out |
|---|---|---|
| attention monitor, binary head (seed 0) | 0.782 / 0.581 | 0.506 / 0.511 |
| attention monitor, binary head (seed 1) | 0.629 / 0.522 | 0.478 / 0.506 |
| linear per-step probe (team_probes) | 0.775 / 0.592 | 0.565 / 0.527 |
| cumulative mean of the linear probe | 0.801 / 0.598 | 0.571 / 0.598 |
| GRU over the linear probe scores | 0.756 / 0.600 | 0.553 / 0.605 |
| turn index alone | 0.429 / 0.422 | 0.308 / 0.509 |
| TF-IDF on the agent's words so far | 0.748 / 0.589 | 0.747 / 0.628 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.115 | 1.211 | 0.273 |
| cheat | attention monitor (seed 1) | 0.713 | 0.617 | 0.562 |
| cheat | majority level of the training fold | 0.761 | 0.536 | 0.663 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 16 | 5 | 0.986 |  |  | 0.927 |  |  |
| 0 | 1 | 13 | 2 | 0.824 |  |  | 0.901 |  |  |
| 1 | 0 | 11 | 0 | 0.833 |  |  | 0.211 |  |  |
| 1 | 1 | 11 | 0 | 0.833 |  |  | 0.605 |  |  |

## SPAR mixed: spar_mixed_41_think_ladders

158 agents, 51 runs, 7440 steps. Split: 5 folds grouped by run.

### cheating used or above

158 agents scored per step (12 reach the cut), 7440 steps (511 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.763 [0.542, 0.926] | 0.757 [0.613, 0.886] | 0.903 [0.819, 0.958] | 0.5 [0.0, 33.5] | 8/12 | 16/146 | 0.046 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.886 [0.780, 0.953] | 0.874 [0.782, 0.958] | 0.924 [0.862, 0.972] | 0.5 [0.0, 1.0] | 10/12 | 22/146 | 0.040 |
| attention monitor, binary head (seed 1) | 0.895 [0.818, 0.947] | 0.851 [0.757, 0.927] | 0.904 [0.822, 0.959] | 6.5 [1.2, 9.5] | 6/12 | 16/146 | 0.069 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.890 [0.799, 0.950] | 0.849 [0.761, 0.925] | 0.905 [0.827, 0.957] | 5.0 [0.5, 8.5] | 7/12 | 17/146 | 0.061 |
| linear per-step probe (team_probes) | 0.776 [0.633, 0.902] | 0.805 [0.669, 0.912] | 0.857 [0.703, 0.955] | 2.5 [0.8, 21.5] | 8/12 | 27/146 | 0.112 |
| cumulative mean of the linear probe | 0.640 [0.504, 0.874] | 0.751 [0.573, 0.888] | 0.703 [0.521, 0.860] | 1.5 [0.8, 2.2] | 2/12 | 11/146 | 0.082 |
| GRU over the linear probe scores | 0.610 [0.369, 0.790] | 0.559 [0.374, 0.747] | 0.682 [0.497, 0.839] | 4.0 [0.0, 4.0] | 5/12 | 20/146 | 0.072 |
| turn index alone | 0.794 [0.698, 0.851] | 0.769 [0.560, 0.913] | 0.769 [0.560, 0.913] | 60.0 [41.0, 60.0] | 5/12 | 14/146 | 0.323 |
| TF-IDF on the agent's words so far | 0.947 [0.885, 0.974] | 0.866 [0.747, 0.959] | 0.889 [0.773, 0.978] | 1.5 [0.0, 5.0] | 8/12 | 10/146 | 0.071 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.132, agent last 0.094, agent max 0.001.

### collusion exploit crossed or above

158 agents scored per step (7 reach the cut), 7440 steps (395 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.831 [0.705, 0.960] | 0.871 [0.734, 0.994] | 0.979 [0.946, 1.000] | 5.0 [1.0, 14.0] | 7/7 | 14/151 | 0.047 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.867 [0.776, 0.960] | 0.882 [0.758, 0.994] | 0.972 [0.936, 1.000] | 5.0 [1.0, 14.5] | 7/7 | 14/151 | 0.046 |
| attention monitor, binary head (seed 1) | 0.857 [0.803, 0.970] | 0.899 [0.756, 1.000] | 0.968 [0.926, 0.994] | 1.5 [0.2, 38.0] | 6/7 | 12/151 | 0.066 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.865 [0.805, 0.973] | 0.939 [0.853, 1.000] | 0.965 [0.932, 0.991] | 1.0 [0.5, 26.5] | 7/7 | 12/151 | 0.042 |
| linear per-step probe (team_probes) | 0.830 [0.734, 0.969] | 0.886 [0.769, 0.981] | 0.909 [0.836, 0.970] | 13.5 [5.0, 19.0] | 6/7 | 17/151 | 0.120 |
| cumulative mean of the linear probe | 0.823 [0.654, 0.998] | 0.891 [0.732, 0.997] | 0.883 [0.713, 0.996] | 0.0 [0.0, 0.5] | 4/7 | 18/151 | 0.104 |
| GRU over the linear probe scores | 0.371 [0.266, 0.925] | 0.447 [0.294, 0.930] | 0.425 [0.285, 0.616] |  | 0/7 | 11/151 | 0.083 |
| turn index alone | 0.855 [0.765, 1.000] | 0.947 [0.873, 1.000] | 0.947 [0.873, 1.000] | 56.0 [25.0, 58.0] | 5/7 | 14/151 | 0.278 |
| TF-IDF on the agent's words so far | 0.934 [0.852, 0.974] | 0.911 [0.812, 0.963] | 0.928 [0.870, 0.976] | 31.0 [18.0, 47.0] | 4/7 | 12/151 | 0.036 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.027, agent last 0.027, agent max 0.011.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.399 | 0.924 | 0.451 |
| cheat | attention monitor (seed 1) | 0.259 | 1.146 | 0.402 |
| cheat | majority level of the training fold | 0.741 | 0.639 | 0.580 |
| coord | attention monitor (seed 0) | 0.741 | 0.475 | 0.842 |
| coord | attention monitor (seed 1) | 0.741 | 0.468 | 0.854 |
| coord | majority level of the training fold | 0.741 | 0.468 | 0.751 |
| coop | attention monitor (seed 0) | 0.886 | 0.215 | 0.941 |
| coop | attention monitor (seed 1) | 0.867 | 0.234 | 0.932 |
| coop | majority level of the training fold | 0.886 | 0.234 | 0.948 |
| collab | attention monitor (seed 0) | 0.943 | 0.095 | 0.945 |
| collab | attention monitor (seed 1) | 0.943 | 0.082 | 0.946 |
| collab | majority level of the training fold | 0.949 | 0.089 | 0.948 |
| collude | attention monitor (seed 0) | 0.943 | 0.177 | 0.944 |
| collude | attention monitor (seed 1) | 0.949 | 0.152 | 0.947 |
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
| 0 | 3 | 56 | 45 | 1.000 | 1.000 | 1.000 | 0.905 |  | 0.962 |
| 0 | 4 | 30 | 19 | 0.990 | 1.000 | 0.998 | 0.775 |  | 0.980 |
| 1 | 0 | 36 | 25 | 0.999 | 1.000 | 1.000 | 0.988 | 0.969 | 0.988 |
| 1 | 1 | 20 | 9 | 0.958 | 0.993 | 0.995 | 0.769 | 0.926 | 0.961 |
| 1 | 2 | 24 | 13 | 0.987 | 0.999 | 0.996 | 0.706 |  | 0.978 |
| 1 | 3 | 35 | 24 | 0.997 | 1.000 | 0.999 | 0.942 | 0.944 | 0.991 |
| 1 | 4 | 37 | 26 | 1.000 | 1.000 | 1.000 | 0.990 | 0.976 | 0.971 |

## SPAR screen team: spar_screen_team_think

226 agents, 67 runs, 9026 steps. Split: 5 folds grouped by run.

### cheating (the set's label: from tried or opened by task)

226 agents scored per step (111 reach the cut), 9026 steps (5314 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.964 [0.940, 0.982] | 0.938 [0.890, 0.974] | 0.948 [0.908, 0.980] | 0.0 [0.0, 1.0] | 96/111 | 6/115 | 0.058 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.970 [0.949, 0.985] | 0.944 [0.888, 0.982] | 0.966 [0.925, 0.993] | 0.0 [0.0, 1.0] | 105/111 | 7/115 | 0.358 |
| attention monitor, binary head (seed 1) | 0.967 [0.942, 0.988] | 0.962 [0.924, 0.992] | 0.967 [0.928, 0.993] | 0.0 [0.0, 0.0] | 106/111 | 3/115 | 0.042 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.977 [0.958, 0.992] | 0.973 [0.933, 0.998] | 0.976 [0.939, 0.998] | 0.0 [0.0, 0.0] | 106/111 | 4/115 | 0.352 |
| linear per-step probe (team_probes) | 0.948 [0.913, 0.971] | 0.871 [0.797, 0.932] | 0.962 [0.928, 0.985] | 0.5 [0.0, 1.2] | 104/111 | 16/115 | 0.046 |
| cumulative mean of the linear probe | 0.965 [0.943, 0.982] | 0.965 [0.937, 0.986] | 0.967 [0.941, 0.987] | 3.0 [1.0, 7.5] | 99/111 | 7/115 | 0.159 |
| GRU over the linear probe scores | 0.958 [0.929, 0.978] | 0.946 [0.901, 0.976] | 0.958 [0.926, 0.983] | 3.0 [1.0, 5.0] | 97/111 | 7/115 | 0.024 |
| turn index alone | 0.813 [0.734, 0.870] | 0.803 [0.695, 0.888] | 0.803 [0.695, 0.888] | 37.5 [31.8, 42.0] | 56/111 | 12/115 | 0.079 |
| TF-IDF on the agent's words so far | 0.982 [0.965, 0.994] | 0.974 [0.946, 0.995] | 0.980 [0.955, 0.997] | 0.0 [0.0, 1.0] | 106/111 | 6/115 | 0.025 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.003, agent last 0.025, agent max 0.019.

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.376 | 1.633 | 0.787 |
| cheat | attention monitor (seed 1) | 0.438 | 1.438 | 0.827 |
| cheat | majority level of the training fold | 0.310 | 2.429 | 0.563 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 60 | 59 | 1.000 |  |  | 0.999 |  |  |
| 0 | 1 | 60 | 57 | 1.000 |  |  | 0.999 |  |  |
| 0 | 2 | 16 | 5 | 0.996 |  |  | 0.981 |  |  |
| 0 | 3 | 14 | 3 | 0.974 |  |  | 0.949 |  |  |
| 0 | 4 | 41 | 30 | 1.000 |  |  | 0.990 |  |  |
| 1 | 0 | 19 | 8 | 0.998 |  |  | 0.999 |  |  |
| 1 | 1 | 51 | 40 | 1.000 |  |  | 0.993 |  |  |
| 1 | 2 | 31 | 20 | 1.000 |  |  | 1.000 |  |  |
| 1 | 3 | 32 | 21 | 1.000 |  |  | 0.968 |  |  |
| 1 | 4 | 43 | 32 | 1.000 |  |  | 0.997 |  |  |

## SPAR screen team: spar_screen_team_think_heldout_author

226 agents, 67 runs, 9026 steps. Split: held-out author (train on the other authors, test on one).

### cheating (the set's label: from tried or opened by task)

226 agents scored per step (111 reach the cut), 9026 steps (5314 positive).

| Model | Step AUROC | Agent AUROC (last) | Agent AUROC (max) | Delay at 5% FPR, turns | Detected | Negative agents ever above threshold | ECE |
|---|---|---|---|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.949 [0.918, 0.970] | 0.929 [0.868, 0.971] | 0.957 [0.919, 0.984] | 0.0 [0.0, 1.0] | 71/111 | 2/115 | 0.080 |
| attention monitor, ordinal head P(level >= cut) (seed 0) | 0.947 [0.914, 0.970] | 0.924 [0.866, 0.971] | 0.945 [0.899, 0.982] | 0.0 [0.0, 2.0] | 60/111 | 3/115 | 0.307 |
| attention monitor, binary head (seed 1) | 0.962 [0.944, 0.976] | 0.921 [0.875, 0.962] | 0.930 [0.886, 0.967] | 0.0 [0.0, 1.0] | 106/111 | 17/115 | 0.076 |
| attention monitor, ordinal head P(level >= cut) (seed 1) | 0.975 [0.958, 0.986] | 0.948 [0.909, 0.980] | 0.966 [0.936, 0.988] | 0.0 [0.0, 1.0] | 104/111 | 14/115 | 0.329 |
| linear per-step probe (team_probes) | 0.892 [0.839, 0.927] | 0.817 [0.714, 0.899] | 0.843 [0.744, 0.921] | 0.0 [0.0, 1.0] | 65/111 | 8/115 | 0.070 |
| cumulative mean of the linear probe | 0.913 [0.867, 0.946] | 0.873 [0.792, 0.939] | 0.873 [0.793, 0.940] | 4.0 [1.0, 11.0] | 73/111 | 5/115 | 0.157 |
| GRU over the linear probe scores | 0.901 [0.853, 0.940] | 0.829 [0.732, 0.907] | 0.857 [0.763, 0.924] | 18.0 [4.0, 33.5] | 59/111 | 8/115 | 0.107 |
| turn index alone | 0.752 [0.679, 0.829] | 0.696 [0.569, 0.800] | 0.696 [0.569, 0.800] | 18.0 [14.0, 25.0] | 9/111 | 14/115 | 0.129 |
| TF-IDF on the agent's words so far | 0.901 [0.850, 0.946] | 0.865 [0.773, 0.933] | 0.940 [0.893, 0.977] | 1.0 [0.0, 4.0] | 93/111 | 16/115 | 0.123 |

Seed-to-seed spread of the binary head (max minus min over seeds): step 0.013, agent last 0.008, agent max 0.026.

### Per held-out author, cheating (the set's label: from tried or opened by task) (step AUROC / agent AUROC from the last step)

| Model | claude-haiku-5-5 held out | deepseek-v4.1-flash held out | gpt-6-luna held out |
|---|---|---|---|
| attention monitor, binary head (seed 0) | 0.963 / 0.935 | 0.950 / 0.957 | 0.886 / 0.889 |
| attention monitor, binary head (seed 1) | 0.975 / 0.936 | 0.958 / 0.972 | 0.974 / 0.944 |
| linear per-step probe (team_probes) | 0.798 / 0.713 | 0.894 / 0.917 | 0.753 / 0.685 |
| cumulative mean of the linear probe | 0.864 / 0.895 | 0.951 / 0.968 | 0.884 / 0.878 |
| GRU over the linear probe scores | 0.818 / 0.711 | 0.924 / 0.936 | 0.853 / 0.813 |
| turn index alone | 0.610 / 0.592 | 0.920 / 0.960 | 0.673 / 0.569 |
| TF-IDF on the agent's words so far | 0.974 / 0.961 | 0.945 / 0.986 | 0.935 / 0.979 |

### Level accuracy

| Ladder | Model | Final-step level accuracy | Final-step level mean absolute error | Per-step level accuracy |
|---|---|---|---|---|
| cheat | attention monitor (seed 0) | 0.385 | 1.584 | 0.578 |
| cheat | attention monitor (seed 1) | 0.261 | 1.642 | 0.464 |
| cheat | majority level of the training fold | 0.310 | 2.429 | 0.148 |

### Fit check (binary heads, step AUROC on the fold's training and validation agents)

| Seed | Fold | Epochs run | Best epoch | train|cheat | train|collude | train|coord | val|cheat | val|collude | val|coord |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 58 | 47 | 1.000 |  |  | 1.000 |  |  |
| 0 | 1 | 14 | 3 | 0.986 |  |  | 0.950 |  |  |
| 0 | 2 | 17 | 6 | 0.995 |  |  | 0.973 |  |  |
| 1 | 0 | 60 | 56 | 1.000 |  |  | 1.000 |  |  |
| 1 | 1 | 18 | 7 | 0.999 |  |  | 0.886 |  |  |
| 1 | 2 | 60 | 55 | 1.000 |  |  | 1.000 |  |  |
