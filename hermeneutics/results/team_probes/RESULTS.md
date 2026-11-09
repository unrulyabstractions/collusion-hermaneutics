# Hermeneutic probes on the team runs

Updated 2026-11-09 01:04 PST. Every number below comes from the CSVs in this folder; held-out scores per agent, team and turn are saved beside them (transfer scores in transfer_scores.csv and transfer_turn_scores.csv.gz), so each AUROC can be recomputed.

## Data

| Set | Agents on the Hub | Agents fetched | Agents in the probes | Runs on the Hub |
|---|---|---|---|---|
| board_seed | 134 | 134 | 129 | 43 |
| controls_600 | 600 | 600 | 600 | 600 |
| spar_mixed_41 | 209 | 209 |  | 51 |
| spar_screen_team | 450 | 226 |  | 135 |
| solo_design | 165 | 54 |  | 55 |
| cheatbench | 244 | 244 |  | 3 |
| evilgenie | 266 | 0 |  | 10 |
| goodfire_reward_hack | 2396 | 0 |  | 13 |
| spar_exploration_single | 313 | 0 |  | 96 |
| spar_screen_solo | 63 | 0 |  | 63 |

| Board-seed population | Count |
|---|---|
| runs | 43 |
| agents | 134 |
| workers | 129 |
| regulators (anticoord arm, excluded from every probe) | 5 |
| workers at cheating used or above | 56 |
| workers below used | 73 |
| workers at cheating opened or above | 126 |
| workers at collusion exploit crossed or above | 48 |
| workers at collusion none | 78 |
| workers at collusion discussed or solicited (left out of probe 2) | 3 |
| workers at coordination posted or above | 82 |
| workers at coordination discussed or above | 65 |
| teams at the collusion floor | 23 |
| used workers whose used onset is the first write of planted content | 54 |
| used workers whose used onset falls back to the opened call | 2 |
| agents written by DeepSeek V4.1 Flash (meta.json author) | 134 |
| workers below opened, all in the anticoord arm | 3 |
| agents whose step count and tool-call count match signals.jsonl and meta.json | 134 |

## Recipe

| Setting | Value |
|---|---|
| features | the reader's residual stream at the tokens of each assistant turn named by the setting; layers 26, 38, 51 concatenated unless a row names one layer |
| agent vector | mean over the agent's turns |
| probe | training-fold PCA to min(128, half the training rows), z-scoring, L2 logistic regression with balanced class weights |
| C | chosen by inner 3-fold CV grouped by run, from 1e-4, 1e-3, 1e-2, 1e-1, 1 |
| outer folds | 5, grouped by run (no run is split), stratified by the label |
| interval | 95% bootstrap over runs, 1000 draws |
| null (agent and team) | labels shuffled within arm, 100 draws, C fixed at the median chosen C; p = (1 + draws at or above the real AUROC) / (1 + draws with a finite AUROC) |
| null (per step) | each agent's pair (reaches the level, onset as a fraction of its turns) permuted within arm, turn labels rebuilt from it, 100 draws, C fixed; for the main settings only |
| metadata baseline | arm one-hot, turn budget one-hot, seed; logistic regression on the same folds |
| text baselines | TF-IDF (word 1-2 grams, fitted on the training fold) with L2 logistic regression and C by inner CV: the agent's own words (reasoning and tool calls, no tool results) and the whole rendered text |
| PCA solver | randomized SVD for the per-step setting A fits; for the per-step setting B fits, the exact top components of the training Gram matrix summed over 5120-column blocks (to bound memory); the think pair fitted both ways agrees |
| call index | signals.jsonl call k is the 0-based index over the agent's individual tool calls; it maps to the assistant step holding it, and acts turn t is step t |

## Feature settings

| Setting | Tokens per turn | Agents |
|---|---|---|
| A: think pair, all agents | think-open and think-close, concatenated; each alone as sub-rows, and each layer alone | every agent with activations |
| B: all seven tokens, agents with turns.safetensors | previous <|im_end|>, newline, <|im_start|>, assistant, newline, think-open, think-close, concatenated; each alone as sub-rows, the five boundary tokens concatenated, and the think pair on the same agents | 134 of 134 board_seed agents so far (complete: every board_seed agent has turns.safetensors) |

## Probes 1 to 3: agent level (workers, one vector per agent)

### 1a cheating: used or above, setting A: think pair, all agents

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 129 | 43 | 56 | 73 | 0.850 [0.742, 0.941] | 0.577 | 0.666 | 0.010 |
| probe Lall think-open | 129 | 43 | 56 | 73 | 0.834 [0.737, 0.917] | 0.579 | 0.681 | 0.010 |
| probe Lall think-close | 129 | 43 | 56 | 73 | 0.850 [0.759, 0.927] | 0.572 | 0.661 | 0.010 |
| probe L26 think-open | 129 | 43 | 56 | 73 | 0.870 [0.787, 0.932] | 0.570 | 0.677 | 0.010 |
| probe L26 think-close | 129 | 43 | 56 | 73 | 0.884 [0.806, 0.943] | 0.564 | 0.669 | 0.010 |
| probe L38 think-open | 129 | 43 | 56 | 73 | 0.827 [0.726, 0.906] | 0.571 | 0.677 | 0.010 |
| probe L38 think-close | 129 | 43 | 56 | 73 | 0.854 [0.763, 0.930] | 0.562 | 0.662 | 0.010 |
| probe L51 think-open | 129 | 43 | 56 | 73 | 0.813 [0.713, 0.900] | 0.578 | 0.668 | 0.010 |
| probe L51 think-close | 129 | 43 | 56 | 73 | 0.855 [0.769, 0.929] | 0.572 | 0.656 | 0.010 |
| TF-IDF agent's words | 129 | 43 | 56 | 73 | 0.890 [0.798, 0.954] |  |  |  |
| TF-IDF whole text | 129 | 43 | 56 | 73 | 0.871 [0.797, 0.925] |  |  |  |
| metadata (arm, seed, turn budget) | 129 | 43 | 56 | 73 | 0.802 [0.693, 0.887] |  |  |  |

### 1a cheating: used or above, setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 129 | 43 | 56 | 73 | 0.850 [0.742, 0.941] | 0.577 | 0.666 | 0.010 |
| probe Lall think-open | 129 | 43 | 56 | 73 | 0.834 [0.737, 0.917] | 0.579 | 0.681 | 0.010 |
| probe Lall think-close | 129 | 43 | 56 | 73 | 0.850 [0.759, 0.927] | 0.572 | 0.661 | 0.010 |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 129 | 43 | 56 | 73 | 0.893 [0.808, 0.956] | 0.585 | 0.665 | 0.010 |
| probe Lall boundary 1 (previous <|im_end|>) | 129 | 43 | 56 | 73 | 0.836 [0.741, 0.911] | 0.574 | 0.664 | 0.010 |
| probe Lall boundary 2 (newline after it) | 129 | 43 | 56 | 73 | 0.811 [0.697, 0.899] | 0.594 | 0.673 | 0.010 |
| probe Lall boundary 3 (<|im_start|>) | 129 | 43 | 56 | 73 | 0.905 [0.842, 0.952] | 0.572 | 0.666 | 0.010 |
| probe Lall boundary 4 (assistant) | 129 | 43 | 56 | 73 | 0.845 [0.766, 0.914] | 0.592 | 0.676 | 0.010 |
| probe Lall boundary 5 (newline after assistant) | 129 | 43 | 56 | 73 | 0.868 [0.774, 0.940] | 0.583 | 0.669 | 0.010 |
| probe Lall boundary 1-5 concatenated | 129 | 43 | 56 | 73 | 0.864 [0.789, 0.924] | 0.596 | 0.672 | 0.010 |
| TF-IDF agent's words | 129 | 43 | 56 | 73 | 0.890 [0.798, 0.954] |  |  |  |
| TF-IDF whole text | 129 | 43 | 56 | 73 | 0.871 [0.797, 0.925] |  |  |  |
| metadata (arm, seed, turn budget) | 129 | 43 | 56 | 73 | 0.802 [0.693, 0.887] |  |  |  |

### 1b cheating: opened or above, setting A: think pair, all agents

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 129 | 43 | 126 | 3 | 0.770 [0.382, 0.984] | 0.506 | 0.892 | 0.170 |
| probe Lall think-open | 129 | 43 | 126 | 3 | 0.593 [0.172, 1.000] | 0.509 | 0.875 | 0.400 |
| probe Lall think-close | 129 | 43 | 126 | 3 | 0.862 [0.666, 0.984] | 0.500 | 0.865 | 0.070 |
| probe L26 think-open | 129 | 43 | 126 | 3 | 0.981 [0.945, 1.000] | 0.557 | 0.937 | 0.010 |
| probe L26 think-close | 129 | 43 | 126 | 3 | 0.847 [0.591, 0.987] | 0.541 | 0.919 | 0.110 |
| probe L38 think-open | 129 | 43 | 126 | 3 | 0.751 [0.524, 0.984] | 0.597 | 0.925 | 0.330 |
| probe L38 think-close | 129 | 43 | 126 | 3 | 0.698 [0.401, 0.929] | 0.588 | 0.925 | 0.370 |
| probe L51 think-open | 129 | 43 | 126 | 3 | 0.720 [0.367, 1.000] | 0.518 | 0.885 | 0.190 |
| probe L51 think-close | 129 | 43 | 126 | 3 | 0.542 [0.180, 0.961] | 0.491 | 0.850 | 0.450 |
| TF-IDF agent's words | 129 | 43 | 126 | 3 | 0.574 [0.339, 0.805] |  |  |  |
| TF-IDF whole text | 129 | 43 | 126 | 3 | 0.852 [0.728, 0.968] |  |  |  |
| metadata (arm, seed, turn budget) | 129 | 43 | 126 | 3 | 0.968 [0.922, 0.992] |  |  |  |

### 1b cheating: opened or above, setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 129 | 43 | 126 | 3 | 0.770 [0.382, 0.984] | 0.506 | 0.892 | 0.170 |
| probe Lall think-open | 129 | 43 | 126 | 3 | 0.593 [0.172, 1.000] | 0.509 | 0.875 | 0.400 |
| probe Lall think-close | 129 | 43 | 126 | 3 | 0.862 [0.666, 0.984] | 0.500 | 0.865 | 0.070 |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 129 | 43 | 126 | 3 | 0.497 [0.180, 0.836] | 0.539 | 0.905 | 0.540 |
| probe Lall boundary 1 (previous <|im_end|>) | 129 | 43 | 126 | 3 | 0.503 [0.214, 0.859] | 0.543 | 0.876 | 0.520 |
| probe Lall boundary 2 (newline after it) | 129 | 43 | 126 | 3 | 0.513 [0.159, 1.000] | 0.559 | 0.926 | 0.560 |
| probe Lall boundary 3 (<|im_start|>) | 129 | 43 | 126 | 3 | 0.550 [0.344, 0.740] | 0.633 | 0.944 | 0.700 |
| probe Lall boundary 4 (assistant) | 129 | 43 | 126 | 3 | 0.508 [0.172, 0.930] | 0.543 | 0.839 | 0.560 |
| probe Lall boundary 5 (newline after assistant) | 129 | 43 | 126 | 3 | 0.582 [0.328, 0.953] | 0.550 | 0.898 | 0.520 |
| probe Lall boundary 1-5 concatenated | 129 | 43 | 126 | 3 | 0.439 [0.291, 0.586] | 0.555 | 0.855 | 0.690 |
| TF-IDF agent's words | 129 | 43 | 126 | 3 | 0.574 [0.339, 0.805] |  |  |  |
| TF-IDF whole text | 129 | 43 | 126 | 3 | 0.852 [0.728, 0.968] |  |  |  |
| metadata (arm, seed, turn budget) | 129 | 43 | 126 | 3 | 0.968 [0.922, 0.992] |  |  |  |

### 2 collusion: exploit crossed or above vs none, setting A: think pair, all agents

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 126 | 43 | 48 | 78 | 0.887 [0.802, 0.950] | 0.639 | 0.737 | 0.010 |
| probe Lall think-open | 126 | 43 | 48 | 78 | 0.875 [0.786, 0.944] | 0.633 | 0.713 | 0.010 |
| probe Lall think-close | 126 | 43 | 48 | 78 | 0.856 [0.757, 0.937] | 0.642 | 0.738 | 0.010 |
| probe L26 think-open | 126 | 43 | 48 | 78 | 0.870 [0.783, 0.936] | 0.645 | 0.739 | 0.010 |
| probe L26 think-close | 126 | 43 | 48 | 78 | 0.915 [0.847, 0.963] | 0.649 | 0.720 | 0.010 |
| probe L38 think-open | 126 | 43 | 48 | 78 | 0.900 [0.819, 0.958] | 0.645 | 0.729 | 0.010 |
| probe L38 think-close | 126 | 43 | 48 | 78 | 0.919 [0.853, 0.968] | 0.637 | 0.722 | 0.010 |
| probe L51 think-open | 126 | 43 | 48 | 78 | 0.899 [0.822, 0.954] | 0.631 | 0.716 | 0.010 |
| probe L51 think-close | 126 | 43 | 48 | 78 | 0.874 [0.785, 0.944] | 0.633 | 0.735 | 0.010 |
| TF-IDF agent's words | 126 | 43 | 48 | 78 | 0.933 [0.874, 0.975] |  |  |  |
| TF-IDF whole text | 126 | 43 | 48 | 78 | 0.923 [0.853, 0.974] |  |  |  |
| metadata (arm, seed, turn budget) | 126 | 43 | 48 | 78 | 0.833 [0.701, 0.942] |  |  |  |

### 2 collusion: exploit crossed or above vs none, setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 126 | 43 | 48 | 78 | 0.887 [0.802, 0.950] | 0.639 | 0.737 | 0.010 |
| probe Lall think-open | 126 | 43 | 48 | 78 | 0.875 [0.786, 0.944] | 0.633 | 0.713 | 0.010 |
| probe Lall think-close | 126 | 43 | 48 | 78 | 0.856 [0.757, 0.937] | 0.642 | 0.738 | 0.010 |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 126 | 43 | 48 | 78 | 0.962 [0.921, 0.989] | 0.660 | 0.730 | 0.010 |
| probe Lall boundary 1 (previous <|im_end|>) | 126 | 43 | 48 | 78 | 0.939 [0.869, 0.982] | 0.628 | 0.714 | 0.010 |
| probe Lall boundary 2 (newline after it) | 126 | 43 | 48 | 78 | 0.936 [0.883, 0.977] | 0.662 | 0.751 | 0.010 |
| probe Lall boundary 3 (<|im_start|>) | 126 | 43 | 48 | 78 | 0.942 [0.895, 0.978] | 0.631 | 0.728 | 0.010 |
| probe Lall boundary 4 (assistant) | 126 | 43 | 48 | 78 | 0.948 [0.900, 0.983] | 0.663 | 0.750 | 0.010 |
| probe Lall boundary 5 (newline after assistant) | 126 | 43 | 48 | 78 | 0.900 [0.816, 0.965] | 0.652 | 0.719 | 0.010 |
| probe Lall boundary 1-5 concatenated | 126 | 43 | 48 | 78 | 0.955 [0.906, 0.989] | 0.675 | 0.738 | 0.010 |
| TF-IDF agent's words | 126 | 43 | 48 | 78 | 0.933 [0.874, 0.975] |  |  |  |
| TF-IDF whole text | 126 | 43 | 48 | 78 | 0.923 [0.853, 0.974] |  |  |  |
| metadata (arm, seed, turn budget) | 126 | 43 | 48 | 78 | 0.833 [0.701, 0.942] |  |  |  |

### 3a coordination: posted or above, setting A: think pair, all agents

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 129 | 43 | 82 | 47 | 0.943 [0.882, 0.982] | 0.643 | 0.718 | 0.010 |
| probe Lall think-open | 129 | 43 | 82 | 47 | 0.890 [0.790, 0.965] | 0.656 | 0.727 | 0.010 |
| probe Lall think-close | 129 | 43 | 82 | 47 | 0.918 [0.840, 0.972] | 0.630 | 0.714 | 0.010 |
| probe L26 think-open | 129 | 43 | 82 | 47 | 0.911 [0.822, 0.970] | 0.633 | 0.735 | 0.010 |
| probe L26 think-close | 129 | 43 | 82 | 47 | 0.903 [0.824, 0.960] | 0.613 | 0.693 | 0.010 |
| probe L38 think-open | 129 | 43 | 82 | 47 | 0.892 [0.792, 0.965] | 0.648 | 0.735 | 0.010 |
| probe L38 think-close | 129 | 43 | 82 | 47 | 0.916 [0.831, 0.981] | 0.616 | 0.711 | 0.010 |
| probe L51 think-open | 129 | 43 | 82 | 47 | 0.879 [0.779, 0.957] | 0.655 | 0.723 | 0.010 |
| probe L51 think-close | 129 | 43 | 82 | 47 | 0.916 [0.836, 0.972] | 0.634 | 0.728 | 0.010 |
| TF-IDF agent's words | 129 | 43 | 82 | 47 | 0.911 [0.840, 0.968] |  |  |  |
| TF-IDF whole text | 129 | 43 | 82 | 47 | 0.873 [0.766, 0.953] |  |  |  |
| metadata (arm, seed, turn budget) | 129 | 43 | 82 | 47 | 0.903 [0.820, 0.964] |  |  |  |

### 3a coordination: posted or above, setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 129 | 43 | 82 | 47 | 0.943 [0.882, 0.982] | 0.643 | 0.718 | 0.010 |
| probe Lall think-open | 129 | 43 | 82 | 47 | 0.890 [0.790, 0.965] | 0.656 | 0.727 | 0.010 |
| probe Lall think-close | 129 | 43 | 82 | 47 | 0.918 [0.840, 0.972] | 0.630 | 0.714 | 0.010 |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 129 | 43 | 82 | 47 | 0.868 [0.751, 0.952] | 0.664 | 0.744 | 0.010 |
| probe Lall boundary 1 (previous <|im_end|>) | 129 | 43 | 82 | 47 | 0.915 [0.850, 0.967] | 0.653 | 0.739 | 0.010 |
| probe Lall boundary 2 (newline after it) | 129 | 43 | 82 | 47 | 0.835 [0.704, 0.941] | 0.662 | 0.756 | 0.010 |
| probe Lall boundary 3 (<|im_start|>) | 129 | 43 | 82 | 47 | 0.873 [0.746, 0.966] | 0.635 | 0.728 | 0.010 |
| probe Lall boundary 4 (assistant) | 129 | 43 | 82 | 47 | 0.886 [0.768, 0.973] | 0.654 | 0.746 | 0.010 |
| probe Lall boundary 5 (newline after assistant) | 129 | 43 | 82 | 47 | 0.879 [0.780, 0.955] | 0.661 | 0.742 | 0.010 |
| probe Lall boundary 1-5 concatenated | 129 | 43 | 82 | 47 | 0.893 [0.799, 0.968] | 0.661 | 0.742 | 0.010 |
| TF-IDF agent's words | 129 | 43 | 82 | 47 | 0.911 [0.840, 0.968] |  |  |  |
| TF-IDF whole text | 129 | 43 | 82 | 47 | 0.873 [0.766, 0.953] |  |  |  |
| metadata (arm, seed, turn budget) | 129 | 43 | 82 | 47 | 0.903 [0.820, 0.964] |  |  |  |

### 3b coordination: discussed or above, setting A: think pair, all agents

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 129 | 43 | 65 | 64 | 0.865 [0.776, 0.938] | 0.646 | 0.729 | 0.010 |
| probe Lall think-open | 129 | 43 | 65 | 64 | 0.901 [0.828, 0.960] | 0.644 | 0.727 | 0.010 |
| probe Lall think-close | 129 | 43 | 65 | 64 | 0.856 [0.764, 0.932] | 0.636 | 0.721 | 0.010 |
| probe L26 think-open | 129 | 43 | 65 | 64 | 0.864 [0.762, 0.940] | 0.646 | 0.739 | 0.010 |
| probe L26 think-close | 129 | 43 | 65 | 64 | 0.948 [0.903, 0.982] | 0.647 | 0.727 | 0.010 |
| probe L38 think-open | 129 | 43 | 65 | 64 | 0.863 [0.765, 0.939] | 0.645 | 0.721 | 0.010 |
| probe L38 think-close | 129 | 43 | 65 | 64 | 0.878 [0.805, 0.943] | 0.632 | 0.714 | 0.010 |
| probe L51 think-open | 129 | 43 | 65 | 64 | 0.889 [0.814, 0.948] | 0.640 | 0.720 | 0.010 |
| probe L51 think-close | 129 | 43 | 65 | 64 | 0.860 [0.769, 0.931] | 0.638 | 0.733 | 0.010 |
| TF-IDF agent's words | 129 | 43 | 65 | 64 | 0.900 [0.823, 0.963] |  |  |  |
| TF-IDF whole text | 129 | 43 | 65 | 64 | 0.908 [0.832, 0.965] |  |  |  |
| metadata (arm, seed, turn budget) | 129 | 43 | 65 | 64 | 0.909 [0.835, 0.967] |  |  |  |

### 3b coordination: discussed or above, setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Agents | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 129 | 43 | 65 | 64 | 0.865 [0.776, 0.938] | 0.646 | 0.729 | 0.010 |
| probe Lall think-open | 129 | 43 | 65 | 64 | 0.901 [0.828, 0.960] | 0.644 | 0.727 | 0.010 |
| probe Lall think-close | 129 | 43 | 65 | 64 | 0.856 [0.764, 0.932] | 0.636 | 0.721 | 0.010 |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 129 | 43 | 65 | 64 | 0.902 [0.824, 0.964] | 0.673 | 0.742 | 0.010 |
| probe Lall boundary 1 (previous <|im_end|>) | 129 | 43 | 65 | 64 | 0.871 [0.784, 0.941] | 0.656 | 0.723 | 0.010 |
| probe Lall boundary 2 (newline after it) | 129 | 43 | 65 | 64 | 0.851 [0.767, 0.926] | 0.669 | 0.745 | 0.010 |
| probe Lall boundary 3 (<|im_start|>) | 129 | 43 | 65 | 64 | 0.890 [0.812, 0.955] | 0.654 | 0.731 | 0.010 |
| probe Lall boundary 4 (assistant) | 129 | 43 | 65 | 64 | 0.857 [0.772, 0.933] | 0.681 | 0.757 | 0.010 |
| probe Lall boundary 5 (newline after assistant) | 129 | 43 | 65 | 64 | 0.884 [0.805, 0.948] | 0.652 | 0.739 | 0.010 |
| probe Lall boundary 1-5 concatenated | 129 | 43 | 65 | 64 | 0.934 [0.871, 0.979] | 0.692 | 0.755 | 0.010 |
| TF-IDF agent's words | 129 | 43 | 65 | 64 | 0.900 [0.823, 0.963] |  |  |  |
| TF-IDF whole text | 129 | 43 | 65 | 64 | 0.908 [0.832, 0.965] |  |  |  |
| metadata (arm, seed, turn budget) | 129 | 43 | 65 | 64 | 0.909 [0.835, 0.967] |  |  |  |

## Probe 4: team collusion floor (team cheat at least used and a piece crossed)

### 4 team collusion floor (one row per run), setting A: think pair, all agents

| Features | Runs scored | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close, max of workers' vectors | 43 | 43 | 23 | 20 | 0.815 [0.658, 0.945] | 0.550 | 0.718 | 0.010 |
| probe Lall think-open+close, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.863 [0.724, 0.968] | 0.545 | 0.726 | 0.010 |
| probe Lall think-open, max of workers' vectors | 43 | 43 | 23 | 20 | 0.770 [0.593, 0.920] | 0.547 | 0.716 | 0.020 |
| probe Lall think-open, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.750 [0.577, 0.894] | 0.552 | 0.720 | 0.030 |
| probe Lall think-close, max of workers' vectors | 43 | 43 | 23 | 20 | 0.839 [0.696, 0.956] | 0.553 | 0.744 | 0.010 |
| probe Lall think-close, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.872 [0.749, 0.972] | 0.553 | 0.720 | 0.010 |
| probe L26 think-open, max of workers' vectors | 43 | 43 | 23 | 20 | 0.657 [0.480, 0.818] | 0.563 | 0.746 | 0.228 |
| probe L26 think-open, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.780 [0.633, 0.909] | 0.561 | 0.737 | 0.030 |
| probe L26 think-close, max of workers' vectors | 43 | 43 | 23 | 20 | 0.798 [0.656, 0.929] | 0.534 | 0.724 | 0.010 |
| probe L26 think-close, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.830 [0.689, 0.937] | 0.530 | 0.707 | 0.010 |
| probe L38 think-open, max of workers' vectors | 43 | 43 | 23 | 20 | 0.761 [0.605, 0.900] | 0.539 | 0.711 | 0.020 |
| probe L38 think-open, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.824 [0.680, 0.949] | 0.550 | 0.739 | 0.010 |
| probe L38 think-close, max of workers' vectors | 43 | 43 | 23 | 20 | 0.789 [0.633, 0.922] | 0.555 | 0.718 | 0.030 |
| probe L38 think-close, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.904 [0.794, 0.978] | 0.539 | 0.746 | 0.010 |
| probe L51 think-open, max of workers' vectors | 43 | 43 | 23 | 20 | 0.709 [0.514, 0.873] | 0.545 | 0.722 | 0.079 |
| probe L51 think-open, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.726 [0.550, 0.885] | 0.554 | 0.724 | 0.059 |
| probe L51 think-close, max of workers' vectors | 43 | 43 | 23 | 20 | 0.826 [0.677, 0.953] | 0.558 | 0.733 | 0.010 |
| probe L51 think-close, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.878 [0.749, 0.980] | 0.561 | 0.724 | 0.010 |
| max over workers of the held-out agent collusion probe score, probe Lall think-close | 43 | 43 | 23 | 20 | 0.835 [0.691, 0.944] |  |  |  |
| max over workers of the held-out agent collusion probe score, probe Lall think-open | 43 | 43 | 23 | 20 | 0.857 [0.729, 0.950] |  |  |  |
| max over workers of the held-out agent collusion probe score, probe Lall think-open+close | 43 | 43 | 23 | 20 | 0.846 [0.700, 0.948] |  |  |  |
| mean over workers of the held-out agent collusion probe score, probe Lall think-close | 43 | 43 | 23 | 20 | 0.813 [0.671, 0.926] |  |  |  |
| mean over workers of the held-out agent collusion probe score, probe Lall think-open | 43 | 43 | 23 | 20 | 0.878 [0.765, 0.963] |  |  |  |
| mean over workers of the held-out agent collusion probe score, probe Lall think-open+close | 43 | 43 | 23 | 20 | 0.870 [0.759, 0.957] |  |  |  |
| TF-IDF workers' whole texts | 43 | 43 | 23 | 20 | 0.835 [0.693, 0.956] |  |  |  |
| TF-IDF workers' words | 43 | 43 | 23 | 20 | 0.846 [0.695, 0.961] |  |  |  |
| metadata (arm, seed, turn budget) | 43 | 43 | 23 | 20 | 0.798 [0.629, 0.929] |  |  |  |

### 4 team collusion floor (one row per run), setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Runs scored | Runs | Positive | Negative | AUROC [95% CI] | Null mean | Null 95th pct | Null p |
|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close, max of workers' vectors | 43 | 43 | 23 | 20 | 0.815 [0.658, 0.945] | 0.550 | 0.718 | 0.010 |
| probe Lall think-open+close, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.863 [0.724, 0.968] | 0.545 | 0.726 | 0.010 |
| probe Lall think-open, max of workers' vectors | 43 | 43 | 23 | 20 | 0.770 [0.593, 0.920] | 0.547 | 0.716 | 0.020 |
| probe Lall think-open, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.750 [0.577, 0.894] | 0.552 | 0.720 | 0.030 |
| probe Lall think-close, max of workers' vectors | 43 | 43 | 23 | 20 | 0.839 [0.696, 0.956] | 0.553 | 0.744 | 0.010 |
| probe Lall think-close, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.872 [0.749, 0.972] | 0.553 | 0.720 | 0.010 |
| probe Lall all seven tokens (boundary 1-5 + think pair), max of workers' vectors | 43 | 43 | 23 | 20 | 0.813 [0.662, 0.939] | 0.565 | 0.720 | 0.010 |
| probe Lall all seven tokens (boundary 1-5 + think pair), mean of workers' vectors | 43 | 43 | 23 | 20 | 0.848 [0.707, 0.955] | 0.560 | 0.728 | 0.010 |
| probe Lall boundary 1 (previous <|im_end|>), max of workers' vectors | 43 | 43 | 23 | 20 | 0.793 [0.632, 0.932] | 0.573 | 0.737 | 0.020 |
| probe Lall boundary 1 (previous <|im_end|>), mean of workers' vectors | 43 | 43 | 23 | 20 | 0.761 [0.593, 0.909] | 0.560 | 0.711 | 0.010 |
| probe Lall boundary 2 (newline after it), max of workers' vectors | 43 | 43 | 23 | 20 | 0.774 [0.623, 0.903] | 0.572 | 0.707 | 0.020 |
| probe Lall boundary 2 (newline after it), mean of workers' vectors | 43 | 43 | 23 | 20 | 0.789 [0.638, 0.923] | 0.569 | 0.713 | 0.010 |
| probe Lall boundary 3 (<|im_start|>), max of workers' vectors | 43 | 43 | 23 | 20 | 0.841 [0.700, 0.963] | 0.569 | 0.720 | 0.010 |
| probe Lall boundary 3 (<|im_start|>), mean of workers' vectors | 43 | 43 | 23 | 20 | 0.841 [0.708, 0.951] | 0.554 | 0.711 | 0.010 |
| probe Lall boundary 4 (assistant), max of workers' vectors | 43 | 43 | 23 | 20 | 0.800 [0.643, 0.946] | 0.594 | 0.728 | 0.020 |
| probe Lall boundary 4 (assistant), mean of workers' vectors | 43 | 43 | 23 | 20 | 0.780 [0.603, 0.922] | 0.552 | 0.688 | 0.010 |
| probe Lall boundary 5 (newline after assistant), max of workers' vectors | 43 | 43 | 23 | 20 | 0.809 [0.648, 0.932] | 0.583 | 0.729 | 0.010 |
| probe Lall boundary 5 (newline after assistant), mean of workers' vectors | 43 | 43 | 23 | 20 | 0.789 [0.628, 0.924] | 0.581 | 0.724 | 0.020 |
| probe Lall boundary 1-5 concatenated, max of workers' vectors | 43 | 43 | 23 | 20 | 0.811 [0.671, 0.926] | 0.579 | 0.720 | 0.010 |
| probe Lall boundary 1-5 concatenated, mean of workers' vectors | 43 | 43 | 23 | 20 | 0.807 [0.667, 0.920] | 0.564 | 0.713 | 0.010 |
| TF-IDF workers' whole texts | 43 | 43 | 23 | 20 | 0.835 [0.693, 0.956] |  |  |  |
| TF-IDF workers' words | 43 | 43 | 23 | 20 | 0.846 [0.695, 0.961] |  |  |  |
| metadata (arm, seed, turn budget) | 43 | 43 | 23 | 20 | 0.798 [0.629, 0.929] |  |  |  |

## Probe 5: per step (one row per turn, label = the level reached at that call)

Turn AUROC scores held-out turns against the turn label. Agent AUROC scores the mean turn score against whether the agent ever reaches the level. Delay is the first turn whose score passes the threshold at a 5% false positive rate on held-out negative turns, minus the onset step, over agents that reach the level; a negative delay fires before the onset. The null rows exist for the main settings only.

### 5a cheating used, onset = first write of planted content, setting A: think pair, all agents

| Features | Turns | Positive turns | Agents | Runs | Positive agents | Turn AUROC [95% CI] | Agent AUROC [95% CI] | Delay median [IQR] | Detected / reaching | Fired before onset | Turn null 95th / p | Agent null 95th / p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 6818 | 1927 | 129 | 43 | 56 | 0.889 [0.850, 0.922] | 0.818 [0.719, 0.901] | 5.0 [0.0, 11.0] | 43 / 56 | 7 | 0.789 / 0.010 | 0.704 / 0.010 |
| probe Lall think-open | 6818 | 1927 | 129 | 43 | 56 | 0.885 [0.846, 0.917] | 0.802 [0.698, 0.893] | 8.0 [2.0, 14.0] | 53 / 56 | 7 | 0.785 / 0.010 | 0.704 / 0.010 |
| probe Lall think-close | 6818 | 1927 | 129 | 43 | 56 | 0.881 [0.835, 0.917] | 0.834 [0.740, 0.911] | 9.0 [1.0, 14.5] | 43 / 56 | 3 | 0.790 / 0.010 | 0.699 / 0.010 |
| probe L26 think-open | 6818 | 1927 | 129 | 43 | 56 | 0.892 [0.855, 0.924] | 0.794 [0.689, 0.879] | 10.0 [3.0, 19.0] | 49 / 56 | 6 |  /  |  /  |
| probe L26 think-close | 6818 | 1927 | 129 | 43 | 56 | 0.890 [0.848, 0.924] | 0.833 [0.731, 0.917] | 5.0 [2.0, 11.8] | 46 / 56 | 4 |  /  |  /  |
| probe L38 think-open | 6818 | 1927 | 129 | 43 | 56 | 0.877 [0.833, 0.917] | 0.791 [0.675, 0.899] | 7.0 [2.0, 12.0] | 51 / 56 | 6 |  /  |  /  |
| probe L38 think-close | 6818 | 1927 | 129 | 43 | 56 | 0.899 [0.862, 0.930] | 0.863 [0.761, 0.939] | 5.0 [2.0, 9.8] | 54 / 56 | 6 |  /  |  /  |
| probe L51 think-open | 6818 | 1927 | 129 | 43 | 56 | 0.880 [0.840, 0.913] | 0.817 [0.713, 0.906] | 8.0 [3.0, 15.0] | 53 / 56 | 5 |  /  |  /  |
| probe L51 think-close | 6818 | 1927 | 129 | 43 | 56 | 0.880 [0.835, 0.915] | 0.827 [0.732, 0.909] | 5.0 [1.5, 18.0] | 47 / 56 | 4 |  /  |  /  |
| TF-IDF agent's words so far | 6818 | 1927 | 129 | 43 | 56 | 0.895 [0.846, 0.933] | 0.798 [0.698, 0.884] | 14.0 [7.0, 25.0] | 41 / 56 | 1 |  /  |  /  |
| TF-IDF whole text so far | 6818 | 1927 | 129 | 43 | 56 | 0.863 [0.812, 0.907] | 0.719 [0.599, 0.831] | 21.0 [13.0, 31.0] | 43 / 56 | 1 |  /  |  /  |
| metadata (arm, seed, turn budget) | 6818 | 1927 | 129 | 43 | 56 | 0.680 [0.604, 0.749] | 0.814 [0.702, 0.898] | -32.0 [-34.0, -26.0] | 5 / 56 | 5 |  /  |  /  |
| metadata + turn position | 6818 | 1927 | 129 | 43 | 56 | 0.849 [0.807, 0.890] | 0.821 [0.716, 0.903] | 22.0 [10.0, 32.0] | 45 / 56 | 2 |  /  |  /  |
| turn position (index, fraction) | 6818 | 1927 | 129 | 43 | 56 | 0.842 [0.797, 0.886] | 0.855 [0.769, 0.920] | 23.5 [13.8, 34.8] | 50 / 56 | 3 |  /  |  /  |

### 5a cheating used, onset = first write of planted content, setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Turns | Positive turns | Agents | Runs | Positive agents | Turn AUROC [95% CI] | Agent AUROC [95% CI] | Delay median [IQR] | Detected / reaching | Fired before onset | Turn null 95th / p | Agent null 95th / p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 6818 | 1927 | 129 | 43 | 56 | 0.890 [0.850, 0.923] | 0.816 [0.717, 0.898] | 6.0 [2.0, 11.5] | 43 / 56 | 5 | 0.788 / 0.010 | 0.703 / 0.010 |
| probe Lall think-open | 6818 | 1927 | 129 | 43 | 56 | 0.889 [0.851, 0.920] | 0.829 [0.737, 0.910] | 8.0 [3.0, 15.2] | 52 / 56 | 7 |  /  |  /  |
| probe Lall think-close | 6818 | 1927 | 129 | 43 | 56 | 0.886 [0.842, 0.921] | 0.833 [0.739, 0.915] | 9.0 [1.5, 14.0] | 51 / 56 | 3 |  /  |  /  |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 6818 | 1927 | 129 | 43 | 56 | 0.896 [0.860, 0.925] | 0.812 [0.710, 0.900] | 6.5 [2.0, 13.0] | 56 / 56 | 6 | 0.792 / 0.010 | 0.719 / 0.010 |
| probe Lall boundary 1 (previous <|im_end|>) | 6818 | 1927 | 129 | 43 | 56 | 0.861 [0.816, 0.900] | 0.773 [0.665, 0.874] | 10.0 [3.0, 26.0] | 49 / 56 | 2 |  /  |  /  |
| probe Lall boundary 2 (newline after it) | 6818 | 1927 | 129 | 43 | 56 | 0.852 [0.804, 0.893] | 0.829 [0.725, 0.911] | 11.0 [3.5, 26.5] | 43 / 56 | 5 |  /  |  /  |
| probe Lall boundary 3 (<|im_start|>) | 6818 | 1927 | 129 | 43 | 56 | 0.882 [0.848, 0.912] | 0.823 [0.715, 0.913] | 6.0 [1.0, 12.5] | 55 / 56 | 9 |  /  |  /  |
| probe Lall boundary 4 (assistant) | 6818 | 1927 | 129 | 43 | 56 | 0.894 [0.855, 0.926] | 0.772 [0.666, 0.874] | 5.5 [1.2, 16.8] | 50 / 56 | 5 |  /  |  /  |
| probe Lall boundary 5 (newline after assistant) | 6818 | 1927 | 129 | 43 | 56 | 0.866 [0.823, 0.903] | 0.821 [0.721, 0.905] | 10.5 [5.0, 19.2] | 52 / 56 | 3 |  /  |  /  |
| probe Lall boundary 1-5 concatenated | 6818 | 1927 | 129 | 43 | 56 | 0.894 [0.859, 0.924] | 0.817 [0.719, 0.903] | 5.5 [2.0, 14.2] | 56 / 56 | 5 |  /  |  /  |
| TF-IDF agent's words so far | 6818 | 1927 | 129 | 43 | 56 | 0.895 [0.846, 0.933] | 0.798 [0.698, 0.884] | 14.0 [7.0, 25.0] | 41 / 56 | 1 |  /  |  /  |
| TF-IDF whole text so far | 6818 | 1927 | 129 | 43 | 56 | 0.863 [0.812, 0.907] | 0.719 [0.599, 0.831] | 21.0 [13.0, 31.0] | 43 / 56 | 1 |  /  |  /  |
| metadata (arm, seed, turn budget) | 6818 | 1927 | 129 | 43 | 56 | 0.680 [0.604, 0.749] | 0.814 [0.702, 0.898] | -32.0 [-34.0, -26.0] | 5 / 56 | 5 |  /  |  /  |
| metadata + turn position | 6818 | 1927 | 129 | 43 | 56 | 0.849 [0.807, 0.890] | 0.821 [0.716, 0.903] | 22.0 [10.0, 32.0] | 45 / 56 | 2 |  /  |  /  |
| turn position (index, fraction) | 6818 | 1927 | 129 | 43 | 56 | 0.842 [0.797, 0.886] | 0.855 [0.769, 0.920] | 23.5 [13.8, 34.8] | 50 / 56 | 3 |  /  |  /  |

### 5b cheating used, onset = opened call (lower bound), setting A: think pair, all agents

| Features | Turns | Positive turns | Agents | Runs | Positive agents | Turn AUROC [95% CI] | Agent AUROC [95% CI] | Delay median [IQR] | Detected / reaching | Fired before onset | Turn null 95th / p | Agent null 95th / p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 6818 | 3305 | 129 | 43 | 56 | 0.739 [0.682, 0.793] | 0.848 [0.752, 0.923] | 31.0 [11.0, 41.0] | 41 / 56 | 1 | 0.652 / 0.010 | 0.690 / 0.010 |
| probe Lall think-open | 6818 | 3305 | 129 | 43 | 56 | 0.710 [0.654, 0.764] | 0.813 [0.715, 0.894] | 27.0 [17.5, 41.0] | 43 / 56 | 1 | 0.650 / 0.010 | 0.685 / 0.010 |
| probe Lall think-close | 6818 | 3305 | 129 | 43 | 56 | 0.734 [0.678, 0.788] | 0.847 [0.750, 0.931] | 33.0 [18.0, 41.0] | 42 / 56 | 0 | 0.666 / 0.010 | 0.706 / 0.010 |
| probe L26 think-open | 6818 | 3305 | 129 | 43 | 56 | 0.714 [0.648, 0.778] | 0.750 [0.654, 0.839] | 29.0 [8.0, 41.0] | 35 / 56 | 0 |  /  |  /  |
| probe L26 think-close | 6818 | 3305 | 129 | 43 | 56 | 0.734 [0.677, 0.793] | 0.827 [0.730, 0.913] | 32.0 [20.5, 43.5] | 51 / 56 | 0 |  /  |  /  |
| probe L38 think-open | 6818 | 3305 | 129 | 43 | 56 | 0.713 [0.647, 0.777] | 0.793 [0.692, 0.878] | 25.0 [12.0, 40.0] | 37 / 56 | 2 |  /  |  /  |
| probe L38 think-close | 6818 | 3305 | 129 | 43 | 56 | 0.757 [0.695, 0.813] | 0.843 [0.754, 0.923] | 29.0 [11.0, 41.0] | 46 / 56 | 1 |  /  |  /  |
| probe L51 think-open | 6818 | 3305 | 129 | 43 | 56 | 0.704 [0.651, 0.753] | 0.822 [0.726, 0.901] | 29.0 [15.5, 39.2] | 40 / 56 | 0 |  /  |  /  |
| probe L51 think-close | 6818 | 3305 | 129 | 43 | 56 | 0.728 [0.676, 0.778] | 0.848 [0.753, 0.929] | 37.0 [17.2, 46.0] | 44 / 56 | 0 |  /  |  /  |
| TF-IDF agent's words so far | 6818 | 3305 | 129 | 43 | 56 | 0.675 [0.597, 0.746] | 0.738 [0.633, 0.835] | 29.0 [22.0, 44.0] | 12 / 56 | 0 |  /  |  /  |
| TF-IDF whole text so far | 6818 | 3305 | 129 | 43 | 56 | 0.722 [0.657, 0.788] | 0.758 [0.664, 0.838] | 17.0 [8.0, 39.0] | 13 / 56 | 0 |  /  |  /  |
| metadata (arm, seed, turn budget) | 6818 | 3305 | 129 | 43 | 56 | 0.738 [0.643, 0.815] | 0.825 [0.723, 0.899] | -4.5 [-7.5, -3.5] | 8 / 56 | 8 |  /  |  /  |
| metadata + turn position | 6818 | 3305 | 129 | 43 | 56 | 0.766 [0.677, 0.839] | 0.830 [0.724, 0.903] | 39.5 [23.8, 51.0] | 30 / 56 | 0 |  /  |  /  |
| turn position (index, fraction) | 6818 | 3305 | 129 | 43 | 56 | 0.780 [0.707, 0.846] | 0.855 [0.767, 0.921] | 40.0 [33.5, 46.5] | 35 / 56 | 0 |  /  |  /  |

### 5b cheating used, onset = opened call (lower bound), setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Turns | Positive turns | Agents | Runs | Positive agents | Turn AUROC [95% CI] | Agent AUROC [95% CI] | Delay median [IQR] | Detected / reaching | Fired before onset | Turn null 95th / p | Agent null 95th / p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 6818 | 3305 | 129 | 43 | 56 | 0.743 [0.685, 0.797] | 0.852 [0.761, 0.928] | 30.5 [10.8, 41.2] | 40 / 56 | 1 | 0.651 / 0.010 | 0.697 / 0.010 |
| probe Lall think-open | 6818 | 3305 | 129 | 43 | 56 | 0.713 [0.657, 0.768] | 0.821 [0.727, 0.899] | 28.0 [12.5, 42.0] | 43 / 56 | 1 |  /  |  /  |
| probe Lall think-close | 6818 | 3305 | 129 | 43 | 56 | 0.735 [0.678, 0.790] | 0.844 [0.747, 0.929] | 34.5 [15.5, 41.0] | 42 / 56 | 0 |  /  |  /  |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 6818 | 3305 | 129 | 43 | 56 | 0.712 [0.655, 0.768] | 0.840 [0.739, 0.920] | 36.0 [26.5, 45.0] | 44 / 56 | 0 | 0.658 / 0.010 | 0.693 / 0.010 |
| probe Lall boundary 1 (previous <|im_end|>) | 6818 | 3305 | 129 | 43 | 56 | 0.688 [0.639, 0.739] | 0.833 [0.730, 0.919] | 36.0 [26.8, 44.2] | 52 / 56 | 0 |  /  |  /  |
| probe Lall boundary 2 (newline after it) | 6818 | 3305 | 129 | 43 | 56 | 0.672 [0.624, 0.719] | 0.821 [0.715, 0.907] | 30.0 [25.0, 42.0] | 39 / 56 | 0 |  /  |  /  |
| probe Lall boundary 3 (<|im_start|>) | 6818 | 3305 | 129 | 43 | 56 | 0.680 [0.635, 0.727] | 0.831 [0.725, 0.921] | 28.0 [21.8, 37.2] | 36 / 56 | 0 |  /  |  /  |
| probe Lall boundary 4 (assistant) | 6818 | 3305 | 129 | 43 | 56 | 0.731 [0.672, 0.785] | 0.858 [0.761, 0.934] | 35.0 [24.5, 45.0] | 47 / 56 | 2 |  /  |  /  |
| probe Lall boundary 5 (newline after assistant) | 6818 | 3305 | 129 | 43 | 56 | 0.689 [0.641, 0.741] | 0.825 [0.719, 0.917] | 35.0 [26.0, 39.0] | 45 / 56 | 0 |  /  |  /  |
| probe Lall boundary 1-5 concatenated | 6818 | 3305 | 129 | 43 | 56 | 0.707 [0.654, 0.760] | 0.852 [0.752, 0.930] | 36.0 [27.0, 44.0] | 52 / 56 | 0 |  /  |  /  |
| TF-IDF agent's words so far | 6818 | 3305 | 129 | 43 | 56 | 0.675 [0.597, 0.746] | 0.738 [0.633, 0.835] | 29.0 [22.0, 44.0] | 12 / 56 | 0 |  /  |  /  |
| TF-IDF whole text so far | 6818 | 3305 | 129 | 43 | 56 | 0.722 [0.657, 0.788] | 0.758 [0.664, 0.838] | 17.0 [8.0, 39.0] | 13 / 56 | 0 |  /  |  /  |
| metadata (arm, seed, turn budget) | 6818 | 3305 | 129 | 43 | 56 | 0.738 [0.643, 0.815] | 0.825 [0.723, 0.899] | -4.5 [-7.5, -3.5] | 8 / 56 | 8 |  /  |  /  |
| metadata + turn position | 6818 | 3305 | 129 | 43 | 56 | 0.766 [0.677, 0.839] | 0.830 [0.724, 0.903] | 39.5 [23.8, 51.0] | 30 / 56 | 0 |  /  |  /  |
| turn position (index, fraction) | 6818 | 3305 | 129 | 43 | 56 | 0.780 [0.707, 0.846] | 0.855 [0.767, 0.921] | 40.0 [33.5, 46.5] | 35 / 56 | 0 |  /  |  /  |

### 5c cheating opened or above (direct onset), setting A: think pair, all agents

| Features | Turns | Positive turns | Agents | Runs | Positive agents | Turn AUROC [95% CI] | Agent AUROC [95% CI] | Delay median [IQR] | Detected / reaching | Fired before onset | Turn null 95th / p | Agent null 95th / p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 6818 | 5910 | 129 | 43 | 126 | 0.948 [0.897, 0.990] | 0.823 [0.645, 0.947] | 1.0 [1.0, 3.0] | 103 / 126 | 1 | 0.913 / 0.010 | 0.805 / 0.040 |
| probe Lall think-open | 6818 | 5910 | 129 | 43 | 126 | 0.922 [0.839, 0.993] | 0.598 [0.070, 0.945] | 7.0 [3.0, 10.0] | 89 / 126 | 1 | 0.907 / 0.020 | 0.770 / 0.277 |
| probe Lall think-close | 6818 | 5910 | 129 | 43 | 126 | 0.926 [0.856, 0.982] | 0.648 [0.320, 0.891] | 2.0 [1.0, 5.0] | 82 / 126 | 5 | 0.910 / 0.010 | 0.765 / 0.248 |
| probe L26 think-open | 6818 | 5910 | 129 | 43 | 126 | 0.950 [0.894, 0.987] | 0.786 [0.531, 0.992] | 1.0 [1.0, 1.0] | 77 / 126 | 4 |  /  |  /  |
| probe L26 think-close | 6818 | 5910 | 129 | 43 | 126 | 0.954 [0.913, 0.988] | 0.868 [0.695, 0.984] | 1.0 [1.0, 3.0] | 103 / 126 | 3 |  /  |  /  |
| probe L38 think-open | 6818 | 5910 | 129 | 43 | 126 | 0.961 [0.914, 0.996] | 0.857 [0.685, 1.000] | 1.0 [1.0, 4.0] | 123 / 126 | 0 |  /  |  /  |
| probe L38 think-close | 6818 | 5910 | 129 | 43 | 126 | 0.935 [0.871, 0.989] | 0.693 [0.449, 0.907] | 3.0 [2.0, 6.0] | 102 / 126 | 1 |  /  |  /  |
| probe L51 think-open | 6818 | 5910 | 129 | 43 | 126 | 0.941 [0.876, 0.992] | 0.725 [0.461, 0.992] | 1.0 [1.0, 5.0] | 117 / 126 | 2 |  /  |  /  |
| probe L51 think-close | 6818 | 5910 | 129 | 43 | 126 | 0.927 [0.864, 0.979] | 0.693 [0.382, 0.913] | 2.0 [1.0, 5.0] | 91 / 126 | 5 |  /  |  /  |
| TF-IDF agent's words so far | 6818 | 5910 | 129 | 43 | 126 | 0.991 [0.978, 0.999] | 0.960 [0.927, 0.992] | 1.0 [1.0, 2.0] | 125 / 126 | 2 |  /  |  /  |
| TF-IDF whole text so far | 6818 | 5910 | 129 | 43 | 126 | 0.964 [0.921, 0.999] | 0.825 [0.685, 0.976] | 4.0 [2.0, 9.0] | 101 / 126 | 0 |  /  |  /  |
| metadata (arm, seed, turn budget) | 6818 | 5910 | 129 | 43 | 126 | 0.654 [0.593, 0.700] | 0.937 [0.854, 0.992] | -6.0 [-7.0, -4.2] | 6 / 126 | 6 |  /  |  /  |
| metadata + turn position | 6818 | 5910 | 129 | 43 | 126 | 0.916 [0.874, 0.960] | 0.943 [0.877, 0.984] | 9.5 [6.0, 13.0] | 126 / 126 | 2 |  /  |  /  |
| turn position (index, fraction) | 6818 | 5910 | 129 | 43 | 126 | 0.888 [0.809, 0.977] | 0.291 [0.071, 0.488] | 42.0 [37.0, 47.0] | 102 / 126 | 0 |  /  |  /  |

### 5c cheating opened or above (direct onset), setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Turns | Positive turns | Agents | Runs | Positive agents | Turn AUROC [95% CI] | Agent AUROC [95% CI] | Delay median [IQR] | Detected / reaching | Fired before onset | Turn null 95th / p | Agent null 95th / p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 6818 | 5910 | 129 | 43 | 126 | 0.942 [0.885, 0.988] | 0.802 [0.655, 0.922] | 2.0 [1.0, 5.0] | 103 / 126 | 2 | 0.914 / 0.010 | 0.807 / 0.089 |
| probe Lall think-open | 6818 | 5910 | 129 | 43 | 126 | 0.957 [0.914, 0.994] | 0.876 [0.743, 1.000] | 1.0 [1.0, 4.0] | 124 / 126 | 1 |  /  |  /  |
| probe Lall think-close | 6818 | 5910 | 129 | 43 | 126 | 0.924 [0.854, 0.984] | 0.635 [0.398, 0.937] | 4.0 [2.0, 7.0] | 97 / 126 | 1 |  /  |  /  |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 6818 | 5910 | 129 | 43 | 126 | 0.925 [0.847, 0.991] | 0.622 [0.211, 0.945] | 1.0 [1.0, 2.5] | 99 / 126 | 0 | 0.912 / 0.010 | 0.781 / 0.307 |
| probe Lall boundary 1 (previous <|im_end|>) | 6818 | 5910 | 129 | 43 | 126 | 0.931 [0.873, 0.989] | 0.706 [0.555, 0.860] | 1.0 [1.0, 5.0] | 125 / 126 | 2 |  /  |  /  |
| probe Lall boundary 2 (newline after it) | 6818 | 5910 | 129 | 43 | 126 | 0.916 [0.852, 0.985] | 0.630 [0.476, 0.797] | 4.0 [1.0, 9.0] | 123 / 126 | 5 |  /  |  /  |
| probe Lall boundary 3 (<|im_start|>) | 6818 | 5910 | 129 | 43 | 126 | 0.940 [0.891, 0.987] | 0.743 [0.602, 0.873] | 1.0 [1.0, 3.0] | 126 / 126 | 3 |  /  |  /  |
| probe Lall boundary 4 (assistant) | 6818 | 5910 | 129 | 43 | 126 | 0.951 [0.902, 0.992] | 0.823 [0.682, 0.953] | 1.0 [1.0, 2.0] | 126 / 126 | 0 |  /  |  /  |
| probe Lall boundary 5 (newline after assistant) | 6818 | 5910 | 129 | 43 | 126 | 0.910 [0.836, 0.981] | 0.521 [0.222, 0.758] | 1.0 [1.0, 14.8] | 78 / 126 | 5 |  /  |  /  |
| probe Lall boundary 1-5 concatenated | 6818 | 5910 | 129 | 43 | 126 | 0.947 [0.899, 0.991] | 0.778 [0.638, 0.914] | 1.0 [1.0, 2.0] | 126 / 126 | 2 |  /  |  /  |
| TF-IDF agent's words so far | 6818 | 5910 | 129 | 43 | 126 | 0.991 [0.978, 0.999] | 0.960 [0.927, 0.992] | 1.0 [1.0, 2.0] | 125 / 126 | 2 |  /  |  /  |
| TF-IDF whole text so far | 6818 | 5910 | 129 | 43 | 126 | 0.964 [0.921, 0.999] | 0.825 [0.685, 0.976] | 4.0 [2.0, 9.0] | 101 / 126 | 0 |  /  |  /  |
| metadata (arm, seed, turn budget) | 6818 | 5910 | 129 | 43 | 126 | 0.654 [0.593, 0.700] | 0.937 [0.854, 0.992] | -6.0 [-7.0, -4.2] | 6 / 126 | 6 |  /  |  /  |
| metadata + turn position | 6818 | 5910 | 129 | 43 | 126 | 0.916 [0.874, 0.960] | 0.943 [0.877, 0.984] | 9.5 [6.0, 13.0] | 126 / 126 | 2 |  /  |  /  |
| turn position (index, fraction) | 6818 | 5910 | 129 | 43 | 126 | 0.888 [0.809, 0.977] | 0.291 [0.071, 0.488] | 42.0 [37.0, 47.0] | 102 / 126 | 0 |  /  |  /  |

### 5d collusion crossed, onset proxy, setting A: think pair, all agents

| Features | Turns | Positive turns | Agents | Runs | Positive agents | Turn AUROC [95% CI] | Agent AUROC [95% CI] | Delay median [IQR] | Detected / reaching | Fired before onset | Turn null 95th / p | Agent null 95th / p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 6818 | 1536 | 129 | 43 | 48 | 0.849 [0.798, 0.894] | 0.730 [0.581, 0.861] | 6.5 [-0.8, 18.8] | 38 / 48 | 10 | 0.790 / 0.010 | 0.717 / 0.020 |
| probe Lall think-open | 6818 | 1536 | 129 | 43 | 48 | 0.843 [0.789, 0.891] | 0.720 [0.579, 0.851] | 8.5 [0.2, 18.8] | 42 / 48 | 10 | 0.783 / 0.010 | 0.711 / 0.059 |
| probe Lall think-close | 6818 | 1536 | 129 | 43 | 48 | 0.849 [0.797, 0.895] | 0.775 [0.636, 0.893] | 6.0 [1.0, 18.0] | 41 / 48 | 9 | 0.787 / 0.010 | 0.711 / 0.010 |
| probe L26 think-open | 6818 | 1536 | 129 | 43 | 48 | 0.875 [0.823, 0.922] | 0.792 [0.671, 0.902] | 6.0 [1.0, 17.0] | 45 / 48 | 6 |  /  |  /  |
| probe L26 think-close | 6818 | 1536 | 129 | 43 | 48 | 0.871 [0.818, 0.915] | 0.794 [0.658, 0.913] | 5.0 [0.0, 18.2] | 44 / 48 | 9 |  /  |  /  |
| probe L38 think-open | 6818 | 1536 | 129 | 43 | 48 | 0.861 [0.808, 0.908] | 0.762 [0.620, 0.883] | 3.5 [1.0, 7.0] | 44 / 48 | 7 |  /  |  /  |
| probe L38 think-close | 6818 | 1536 | 129 | 43 | 48 | 0.861 [0.806, 0.907] | 0.794 [0.663, 0.905] | 4.5 [0.0, 18.0] | 46 / 48 | 9 |  /  |  /  |
| probe L51 think-open | 6818 | 1536 | 129 | 43 | 48 | 0.836 [0.784, 0.880] | 0.714 [0.569, 0.846] | 6.0 [-6.0, 20.5] | 39 / 48 | 13 |  /  |  /  |
| probe L51 think-close | 6818 | 1536 | 129 | 43 | 48 | 0.843 [0.792, 0.889] | 0.763 [0.618, 0.883] | 6.5 [1.0, 18.0] | 42 / 48 | 9 |  /  |  /  |
| TF-IDF agent's words so far | 6818 | 1536 | 129 | 43 | 48 | 0.879 [0.837, 0.914] | 0.775 [0.661, 0.880] | 15.0 [6.0, 27.0] | 40 / 48 | 4 |  /  |  /  |
| TF-IDF whole text so far | 6818 | 1536 | 129 | 43 | 48 | 0.874 [0.821, 0.913] | 0.818 [0.703, 0.912] | 13.0 [7.0, 23.5] | 43 / 48 | 4 |  /  |  /  |
| metadata (arm, seed, turn budget) | 6818 | 1536 | 129 | 43 | 48 | 0.758 [0.667, 0.844] | 0.870 [0.749, 0.968] |  [, ] | 0 / 48 | 0 |  /  |  /  |
| metadata + turn position | 6818 | 1536 | 129 | 43 | 48 | 0.878 [0.811, 0.933] | 0.869 [0.735, 0.965] | 13.0 [5.0, 22.0] | 41 / 48 | 6 |  /  |  /  |
| turn position (index, fraction) | 6818 | 1536 | 129 | 43 | 48 | 0.866 [0.811, 0.914] | 0.904 [0.816, 0.967] | 14.5 [5.8, 24.0] | 44 / 48 | 9 |  /  |  /  |

### 5d collusion crossed, onset proxy, setting B: all seven tokens, agents with turns.safetensors (complete: every board_seed agent has turns.safetensors)

| Features | Turns | Positive turns | Agents | Runs | Positive agents | Turn AUROC [95% CI] | Agent AUROC [95% CI] | Delay median [IQR] | Detected / reaching | Fired before onset | Turn null 95th / p | Agent null 95th / p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe Lall think-open+close | 6818 | 1536 | 129 | 43 | 48 | 0.853 [0.800, 0.897] | 0.769 [0.622, 0.891] | 8.0 [-1.0, 18.0] | 39 / 48 | 11 | 0.788 / 0.010 | 0.714 / 0.010 |
| probe Lall think-open | 6818 | 1536 | 129 | 43 | 48 | 0.840 [0.786, 0.887] | 0.716 [0.570, 0.845] | 8.5 [-1.2, 18.8] | 42 / 48 | 11 |  /  |  /  |
| probe Lall think-close | 6818 | 1536 | 129 | 43 | 48 | 0.852 [0.799, 0.896] | 0.781 [0.635, 0.904] | 6.0 [1.0, 17.2] | 44 / 48 | 9 |  /  |  /  |
| probe Lall all seven tokens (boundary 1-5 + think pair) | 6818 | 1536 | 129 | 43 | 48 | 0.855 [0.804, 0.898] | 0.776 [0.642, 0.893] | 7.0 [0.0, 15.2] | 44 / 48 | 10 | 0.789 / 0.010 | 0.710 / 0.010 |
| probe Lall boundary 1 (previous <|im_end|>) | 6818 | 1536 | 129 | 43 | 48 | 0.824 [0.774, 0.873] | 0.746 [0.612, 0.858] | 7.5 [2.5, 17.8] | 46 / 48 | 8 |  /  |  /  |
| probe Lall boundary 2 (newline after it) | 6818 | 1536 | 129 | 43 | 48 | 0.821 [0.772, 0.867] | 0.769 [0.643, 0.880] | 10.0 [4.2, 22.2] | 48 / 48 | 8 |  /  |  /  |
| probe Lall boundary 3 (<|im_start|>) | 6818 | 1536 | 129 | 43 | 48 | 0.827 [0.777, 0.870] | 0.771 [0.649, 0.880] | 9.0 [0.0, 19.0] | 40 / 48 | 9 |  /  |  /  |
| probe Lall boundary 4 (assistant) | 6818 | 1536 | 129 | 43 | 48 | 0.861 [0.811, 0.905] | 0.801 [0.683, 0.904] | 6.0 [1.5, 16.0] | 47 / 48 | 6 |  /  |  /  |
| probe Lall boundary 5 (newline after assistant) | 6818 | 1536 | 129 | 43 | 48 | 0.818 [0.768, 0.864] | 0.740 [0.592, 0.854] | 7.5 [4.2, 18.5] | 44 / 48 | 7 |  /  |  /  |
| probe Lall boundary 1-5 concatenated | 6818 | 1536 | 129 | 43 | 48 | 0.837 [0.786, 0.884] | 0.757 [0.613, 0.873] | 7.0 [2.8, 19.0] | 48 / 48 | 9 |  /  |  /  |
| TF-IDF agent's words so far | 6818 | 1536 | 129 | 43 | 48 | 0.879 [0.837, 0.914] | 0.775 [0.661, 0.880] | 15.0 [6.0, 27.0] | 40 / 48 | 4 |  /  |  /  |
| TF-IDF whole text so far | 6818 | 1536 | 129 | 43 | 48 | 0.874 [0.821, 0.913] | 0.818 [0.703, 0.912] | 13.0 [7.0, 23.5] | 43 / 48 | 4 |  /  |  /  |
| metadata (arm, seed, turn budget) | 6818 | 1536 | 129 | 43 | 48 | 0.758 [0.667, 0.844] | 0.870 [0.749, 0.968] |  [, ] | 0 / 48 | 0 |  /  |  /  |
| metadata + turn position | 6818 | 1536 | 129 | 43 | 48 | 0.878 [0.811, 0.933] | 0.869 [0.735, 0.965] | 13.0 [5.0, 22.0] | 41 / 48 | 6 |  /  |  /  |
| turn position (index, fraction) | 6818 | 1536 | 129 | 43 | 48 | 0.866 [0.811, 0.914] | 0.904 [0.816, 0.967] | 14.5 [5.8, 24.0] | 44 / 48 | 9 |  /  |  /  |

![step_scores_cheating.png](step_scores_cheating.png)

![step_scores_collusion.png](step_scores_collusion.png)

## Probe 6: cheating probe trained on controls_600, applied unchanged to the team workers

| Training arm | Agents | Cheat | No cheat | Authors |
|---|---|---|---|---|
| solo | 373 | 193 | 180 | 5 |
| solo + cheat | 600 | 343 | 257 | 5 |

### Trained on controls solo

| Features | Scored on | Rows | Positive | Negative | AUROC [95% CI] |
|---|---|---|---|---|---|
| probe Lall think-open+close | controls, cross-validated (reference) | 373 | 193 | 180 | 0.957 [0.935, 0.976] |
| probe Lall think-open+close | team workers, cheating used or above | 129 | 56 | 73 | 0.621 [0.520, 0.714] |
| probe Lall think-open+close | team workers, cheating tried or above (the controls' rule for software engineering) | 129 | 127 | 2 | 0.646 [0.398, 0.881] |
| probe Lall think-open+close | team workers, cheating opened or above | 129 | 126 | 3 | 0.534 [0.270, 0.858] |
| probe Lall think-open+close | team turns, cheating used at that call (onset = first write of planted content) | 6818 | 1927 | 4891 | 0.502 [0.472, 0.528] |
| probe Lall think-open | controls, cross-validated (reference) | 373 | 193 | 180 | 0.951 [0.929, 0.970] |
| probe Lall think-open | team workers, cheating used or above | 129 | 56 | 73 | 0.626 [0.527, 0.713] |
| probe Lall think-open | team workers, cheating tried or above (the controls' rule for software engineering) | 129 | 127 | 2 | 0.535 [0.323, 0.748] |
| probe Lall think-open | team workers, cheating opened or above | 129 | 126 | 3 | 0.407 [0.118, 0.709] |
| probe Lall think-open | team turns, cheating used at that call (onset = first write of planted content) | 6818 | 1927 | 4891 | 0.518 [0.491, 0.544] |
| probe Lall think-close | controls, cross-validated (reference) | 373 | 193 | 180 | 0.951 [0.929, 0.970] |
| probe Lall think-close | team workers, cheating used or above | 129 | 56 | 73 | 0.622 [0.508, 0.732] |
| probe Lall think-close | team workers, cheating tried or above (the controls' rule for software engineering) | 129 | 127 | 2 | 0.524 [0.195, 0.852] |
| probe Lall think-close | team workers, cheating opened or above | 129 | 126 | 3 | 0.526 [0.228, 0.828] |
| probe Lall think-close | team turns, cheating used at that call (onset = first write of planted content) | 6818 | 1927 | 4891 | 0.501 [0.467, 0.532] |
| TF-IDF agent's words | team workers, cheating used or above | 129 | 56 | 73 | 0.736 [0.648, 0.820] |
| TF-IDF agent's words | team workers, cheating tried or above (the controls' rule for software engineering) | 129 | 127 | 2 | 0.894 [0.789, 0.969] |
| TF-IDF agent's words | team workers, cheating opened or above | 129 | 126 | 3 | 0.868 [0.761, 0.953] |
| probe Lall think-open+close | spar_screen_team, cheating (the set's own label) | 226 | 111 | 115 | 0.857 [0.794, 0.914] |
| probe Lall think-open+close | spar_mixed_41, cheating (the set's own label) | 209 | 43 | 166 | 0.791 [0.709, 0.865] |
| probe Lall think-open | spar_screen_team, cheating (the set's own label) | 226 | 111 | 115 | 0.821 [0.748, 0.886] |
| probe Lall think-open | spar_mixed_41, cheating (the set's own label) | 209 | 43 | 166 | 0.720 [0.627, 0.809] |
| probe Lall think-close | spar_screen_team, cheating (the set's own label) | 226 | 111 | 115 | 0.823 [0.740, 0.896] |
| probe Lall think-close | spar_mixed_41, cheating (the set's own label) | 209 | 43 | 166 | 0.782 [0.702, 0.861] |

### Trained on controls solo + cheat

| Features | Scored on | Rows | Positive | Negative | AUROC [95% CI] |
|---|---|---|---|---|---|
| probe Lall think-open+close | controls, cross-validated (reference) | 600 | 343 | 257 | 0.955 [0.939, 0.969] |
| probe Lall think-open+close | team workers, cheating used or above | 129 | 56 | 73 | 0.609 [0.513, 0.696] |
| probe Lall think-open+close | team workers, cheating tried or above (the controls' rule for software engineering) | 129 | 127 | 2 | 0.394 [0.062, 0.756] |
| probe Lall think-open+close | team workers, cheating opened or above | 129 | 126 | 3 | 0.333 [0.078, 0.727] |
| probe Lall think-open+close | team turns, cheating used at that call (onset = first write of planted content) | 6818 | 1927 | 4891 | 0.499 [0.473, 0.523] |
| probe Lall think-open | controls, cross-validated (reference) | 600 | 343 | 257 | 0.946 [0.926, 0.962] |
| probe Lall think-open | team workers, cheating used or above | 129 | 56 | 73 | 0.473 [0.389, 0.556] |
| probe Lall think-open | team workers, cheating tried or above (the controls' rule for software engineering) | 129 | 127 | 2 | 0.398 [0.079, 0.744] |
| probe Lall think-open | team workers, cheating opened or above | 129 | 126 | 3 | 0.376 [0.102, 0.704] |
| probe Lall think-open | team turns, cheating used at that call (onset = first write of planted content) | 6818 | 1927 | 4891 | 0.465 [0.444, 0.487] |
| probe Lall think-close | controls, cross-validated (reference) | 600 | 343 | 257 | 0.958 [0.943, 0.973] |
| probe Lall think-close | team workers, cheating used or above | 129 | 56 | 73 | 0.709 [0.593, 0.803] |
| probe Lall think-close | team workers, cheating tried or above (the controls' rule for software engineering) | 129 | 127 | 2 | 0.386 [0.179, 0.617] |
| probe Lall think-close | team workers, cheating opened or above | 129 | 126 | 3 | 0.381 [0.203, 0.600] |
| probe Lall think-close | team turns, cheating used at that call (onset = first write of planted content) | 6818 | 1927 | 4891 | 0.544 [0.509, 0.576] |
| TF-IDF agent's words | team workers, cheating used or above | 129 | 56 | 73 | 0.703 [0.614, 0.788] |
| TF-IDF agent's words | team workers, cheating tried or above (the controls' rule for software engineering) | 129 | 127 | 2 | 0.921 [0.865, 0.969] |
| TF-IDF agent's words | team workers, cheating opened or above | 129 | 126 | 3 | 0.921 [0.867, 0.967] |

## Data gaps

- One author. Every team worker is DeepSeek V4.1 Flash, so these probes cannot separate the behaviour from the author's style; the controls transfer is the only cross-author test.
- Small samples. The probes rest on a few dozen runs; intervals are wide and the folds hold few runs each.
- Opened or above has almost no negatives among workers, and all of them sit in the anticoord arm, so the agent-level rows of 1b and 5c largely read the arm (the metadata baseline shows it) and their within-arm null shuffles inside that arm alone.
- Probe 2 leaves out the workers at collusion discussed or solicited; probe 5d keeps them as agents that never reach exploit crossed.
- Proxies. The ladders time exposed, tried and opened per call; used and crossed carry no call. The used onset is estimated as the first tool call whose arguments contain an added line of the planted diff the agent opened (falling back to the opened call), and probe 5b gives the opened call as a lower bound. The collusion onset is a proxy: the later of the used onset (or, for a sharer below used, its first post) and its first board read or post. Cooperation, collaboration and scheming are proxies in levels.py itself.
- Labels that depend on the future. A turn after the opened call is negative for an agent that ends at opened and positive for one that goes on to use the content, so the per-step used label partly asks the probe to predict what comes next.
- Setting B covers only the agents whose turns.safetensors is on the Hub; until the backfill completes its rows rest on a subset, named with its agent and run counts. The list of agents with boundary tokens is boundary_agents.csv.
- The 5% false positive threshold is set on the pooled held-out negative turns of all folds, whose fold models differ in scale.
- The controls label counts cheating from tried for software engineering, chess and creative writing and from opened for other tasks; the team label counts from used. Probe 6 scores the team workers against both rules.
