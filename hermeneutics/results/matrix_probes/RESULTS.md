# Probe matrix: probe types by inclusion schemes

Every probe type crossed with every way of including or excluding the labelled sources, on the reader's (Qwen3.8-27B) think-token states. Built on this Mac from the local copy of the public Hub dataset, with no model calls and no spend. Code: `hermeneutics/matrix_probes.py`, which imports the PCA, z-scoring, logistic regression, inner-CV choice of C, grouped folds, bootstrap and permutation helpers of `hermeneutics/team_probes.py`.

## Design

**Probe types.** **A** is one probe per author, cross-validated within that author with folds grouped by run; the table gives each author and the mean over authors. **B.I** is one probe on every author of the scheme, cross-validated with 5 folds grouped by run (stratified by source, author and label). **B.II** leaves one author out: the probe trains on every other author and scores the held-out author; runs that hold any agent of the held-out author are removed from training. In the transfer schemes (leave one source out, ours to external, external to ours) B.I trains on the training sources and scores the test source, A trains per author on the training sources and scores the same author in the test source, and B.II trains on the training sources without the test author, again removing every training run that holds any agent of that author.

**Probe.** Each agent is the mean over its turns of the think-open and think-close states at layers 26, 38 and 51 (30,720 dimensions). Training-fold PCA to at most 128 components, z-scoring, L2 logistic regression with balanced class weights, C chosen by an inner 3-fold CV grouped by run. Feature rows: the think pair at all layers (main), the think pair at layer 38 alone, and the seven tokens (five turn-boundary tokens plus the think pair, all layers) where every agent of the scheme has turns.safetensors.

**Each cell.** Held-out AUROC with a 95% bootstrap interval over runs (1,000 draws; for a mean over authors, one run draw shared by every author). Null: labels permuted within source, author and arm (20 draws), probe refitted with C fixed at the median chosen C; cross-validated cells score against the permuted labels (team_probes.null_aucs), split cells permute the training labels and score the real test labels. The null keeps each stratum's base rate, so it measures what the probe adds beyond source, author and arm. Baselines on the same folds: **TF-IDF text** (word unigrams and bigrams of the rendered transcript, the reader's whole input; vocabulary of the 50,000 most frequent hashed terms with min_df 2, sublinear tf and idf fitted on the training rows; 2^22 hash buckets, so rare terms can collide; C by inner CV) and **metadata** (one-hot source, author, arm, task, role, log turns, log tokens; logistic regression, C = 1).

**Labels.** Cheating: controls_600 uses the set's own label (data/labels.csv `cheat`: from `tried` for software engineering, chess and creative writing, from `opened` otherwise); the team and SPAR sets use the code ladder of viewer/levels.py, used or above; Goodfire uses the judge verdict of labels/judge_rollouts.jsonl, yes against no, split dropped; CheatBench and EvilGenie use the external_probes agent's label tables (CheatBench meta.json label, cheating or attempt against benign; EvilGenie sel_label, hack against legitimate, 133 matched pairs of one author, each pair one group in every fold). Coordination: posted or above. Collusion: crossed or above. Coordination and collusion only on board_seed, solo_design, spar_screen_team and spar_mixed_41, as asked; spar_exploration_single also has a board but is left out of those two labels. A source enters a label when it has at least 3 agents in each class; a cell runs when each class has at least 3 agents and there are at least 5 runs.

## Sources, counts and provenance

| source | agents | with features | cheat labelled | cheat pos | coord pos | collude pos | runs | authors | data | activations extracted by |
|---|---|---|---|---|---|---|---|---|---|---|
| controls_600 | 600 | 600 | 600 | 343 |  |  | 600 | 5 | ours; SGA-adapted single-agent controls run 2026-11-06 22:20 to 2026-11-07 UTC, during the sprint (opened 2026-11-06 21:30 UTC) | another agent (Plan B extraction on rented boxes, 2026-11-09) |
| board_seed | 134 | 134 | 134 | 56 | 87 | 48 | 43 | 1 | ours; three-worker team runs made 2026-11-07, during the sprint | another agent (Plan B extraction on rented boxes, 2026-11-09) |
| solo_design | 177 | 177 | 177 | 84 | 38 | 0 | 59 | 3 | ours; runs made 2026-11-09, during the sprint | another agent (Plan B extraction on rented boxes, 2026-11-09) |
| spar_screen_team | 450 | 450 | 450 | 109 | 406 | 67 | 135 | 3 | SPAR team's runs (SparGuardianAgents, dataset unrulyabstractions/spar-swarm), 2026-10-30 to 10-31; predate the sprint | another agent (Plan B extraction on rented boxes, 2026-11-09) |
| spar_mixed_41 | 209 | 209 | 209 | 12 | 205 | 7 | 51 | 6 | SPAR team's runs, 2026-11-01 to 11-02; predate the sprint | another agent (Plan B extraction on rented boxes, 2026-11-09) |
| spar_exploration_single | 313 | 313 | 313 | 71 |  |  | 96 | 3 | SPAR team's runs, 2026-11-01 to 11-03; predate the sprint | another agent (Plan B extraction on rented boxes, 2026-11-09) |
| spar_screen_solo | 90 | 90 | 90 | 47 |  |  | 90 | 3 | SPAR team's runs, 2026-10-30; predate the sprint | another agent (Plan B extraction on rented boxes, 2026-11-09) |
| goodfire_reward_hack | 5553 | 5553 | 2833 | 2083 |  |  | 5553 | 3 | external (Goodfire/reward-hack-data, arXiv:2609.19101); predates the sprint | another agent (hermeneutics/extract_goodfire.py, 2026-11-09) |
| cheatbench | 244 | 224 | 224 | 134 |  |  | 224 | 4 | external (steinad/CheatBench); predates the sprint | another agent |
| evilgenie | 266 | 266 | 266 | 133 |  |  | 133 | 10 | external (WiserHumanExperimental/evilgenie-escalation); predates the sprint | another agent |

Labels of every source were built here; for board_seed, solo_design, spar_screen_team and spar_mixed_41 they equal the team_probes agent's label CSVs on every agent (cheat, coordination and collusion indices, onsets, roles). For spar_exploration_single and spar_screen_solo the run dirs were fetched from the public unrulyabstractions/spar-swarm dataset and run through the same levels.py code. Goodfire: 2083 yes, 750 no, 297 split (dropped), 2423 without a judge verdict (tau3bench and shoppingbench carry only the generic monitor, and 6 DeepSWE rollouts were not judged). A Goodfire rollout is its own run. CheatBench and EvilGenie: labels read from the external_probes agent's `labels/cheatbench.csv` and `labels/evilgenie.csv` (that agent checked the joins, the counts and the matched pairs in its own VERIFICATION.md); 20 CheatBench rollouts with zero assistant turns have no features and are left out. Neither set has enough per-call onsets for the per-step probe (CheatBench 21 of 134 positives, EvilGenie none), so the per-step 'all' scheme keeps ours and Goodfire.

**Authors.** Names are normalised from each agent's own model id (the episode's model for team runs, so regulators of another model are their own author), never from a short name: `qwen` in controls_600 is Qwen3.8 27B, `qwenmax` in Goodfire is Qwen3.8 Max. One judgement call: solo_design's `openrouter/qwen/qwen3.8-max-0902` and Goodfire's `Qwen/Qwen3.8-2.4T-A95B-FP8` are both named Qwen 3.8 Max by their sources and are one author here.

| model id in the metadata | author | sources | agents |
|---|---|---|---|
| Qwen/Qwen3.8-2.4T-A95B-FP8 | Qwen3.8 Max | goodfire | 963 |
| anthropic/claude-haiku-5-5 | Claude Haiku 5.5 | spar_solo, spar_team | 180 |
| claude-3-7-sonnet | Claude 3.7 Sonnet | cheatbench | 60 |
| claude-fable-5 | Claude Fable 5 | evilgenie | 10 |
| claude-opus-4-6 | Claude Opus 4.6 | cheatbench | 131 |
| claude-opus-4.8 | Claude Opus 4.8 | evilgenie | 6 |
| gemini-3.1-pro-preview | Gemini 3.1 Pro | evilgenie | 34 |
| gemini-3.5-flash | Gemini 3.5 Flash | evilgenie | 56 |
| gpt-5.3-codex | GPT-5.3 Codex | cheatbench, evilgenie | 77 |
| gpt-5.4 | GPT-5.4 | cheatbench | 18 |
| gpt-5.6-luna | GPT-5.6 Luna | evilgenie | 28 |
| gpt-5.6-sol | GPT-5.6 Sol | evilgenie | 24 |
| grok-4.5 | Grok 4.5 | evilgenie | 40 |
| kimi-k2.7-code | Kimi K2.7 Code | evilgenie | 18 |
| kimi-k3 | Kimi K3 | evilgenie | 8 |
| moonshotai/Kimi-K3 | Kimi K3 | goodfire | 1580 |
| moonshotai/kimi-k3 | Kimi K3 | goodfire | 586 |
| openai/gpt-6-luna | GPT-6 Luna | spar_solo, spar_team | 180 |
| openrouter/anthropic/claude-haiku-5.5 | Claude Haiku 5.5 | controls, spar_expl, spar_mixed | 184 |
| openrouter/anthropic/claude-opus-5.5 | Claude Opus 5.5 | spar_mixed | 2 |
| openrouter/anthropic/claude-sonnet-5.5 | Claude Sonnet 5.5 | spar_mixed | 5 |
| openrouter/deepseek/deepseek-v4.1-flash-1017 | DeepSeek V4.1 Flash | board_seed, controls, solo_design, spar_expl, spar_mixed, spar_solo, spar_team | 596 |
| openrouter/google/gemini-4-argon | Gemini 4 Argon | spar_mixed | 12 |
| openrouter/openai/gpt-5-mini | GPT-5 mini | controls | 120 |
| openrouter/openai/gpt-6-luna | GPT-6 Luna | controls, solo_design, spar_expl, spar_mixed | 568 |
| openrouter/qwen/qwen3.8-27b | Qwen3.8 27B | controls | 120 |
| openrouter/qwen/qwen3.8-max-0902 | Qwen3.8 Max | solo_design | 6 |
| z-ai/glm-5.2 | GLM 5.2 | goodfire | 634 |
| zai-org/GLM-5.2 | GLM 5.2 | goodfire | 500 |
| zai-org/GLM-5.2-FP8 | GLM 5.2 | goodfire | 1290 |

## The matrix (main feature: think pair, all layers)

AUROC [95% CI]; counts are agents / positives / runs / authors of the scored test rows. A and B.II give the mean over the authors that could be scored. `null` is the 95th percentile of the 20-draw permutation null of the same cell. TF-IDF and metadata are the B.I baselines on the same folds.

### cheating

| scheme | A mean | A n/pos/runs/auth | B.I | B.I n/pos/runs/auth | B.I null 95th | B.I TF-IDF | B.I metadata | B.II mean | B.II n/pos/runs/auth | note |
|---|---|---|---|---|---|---|---|---|---|---|
| alone: controls | 0.938 [0.918, 0.957] | 600/343/600/5 | 0.960 [0.944, 0.973] | 600/343/600/5 | 0.604 | 0.990 | 0.690 | 0.913 [0.891, 0.934] | 600/343/600/5 |  |
| alone: board_seed | 0.880 [0.809, 0.935] | 134/56/43/1 | 0.880 [0.809, 0.935] | 134/56/43/1 | 0.676 | 0.905 | 0.826 | n/a |  | one author only: nothing to hold out |
| alone: solo_design | 0.816 [0.694, 0.905] | 171/79/57/2 | 0.914 [0.860, 0.960] | 177/84/59/3 | 0.777 | 0.922 | 0.866 | 0.630 [0.532, 0.723] | 171/79/57/2 | too few agents or runs in a class |
| alone: spar_team | 0.926 [0.878, 0.963] | 450/109/135/3 | 0.927 [0.885, 0.962] | 450/109/135/3 | 0.700 | 0.982 | 0.934 | 0.751 [0.689, 0.811] | 450/109/135/3 |  |
| alone: spar_mixed | 0.715 [0.492, 0.882] | 162/12/51/2 | 0.734 [0.606, 0.869] | 209/12/51/6 | 0.858 | 0.937 | 0.837 | 0.708 [0.493, 0.859] | 162/12/51/2 | too few agents or runs in a class |
| alone: spar_expl | 0.910 [0.781, 0.968] | 313/71/96/3 | 0.955 [0.918, 0.984] | 313/71/96/3 | 0.788 | 0.970 | 0.936 | 0.658 [0.456, 0.753] | 313/71/96/3 |  |
| alone: spar_solo | 0.926 [0.847, 0.987] | 60/17/60/2 | 0.962 [0.915, 0.995] | 90/47/90/3 | 0.895 | 0.985 | 0.926 | 0.727 [0.520, 0.921] | 60/17/60/2 | too few agents or runs in a class |
| alone: goodfire | 0.908 [0.896, 0.919] | 2833/2083/2833/3 | 0.921 [0.910, 0.931] | 2833/2083/2833/3 | 0.695 | 0.927 | 0.809 | 0.822 [0.803, 0.840] | 2833/2083/2833/3 |  |
| alone: cheatbench | 0.956 [0.913, 0.987] | 224/134/224/4 | 0.951 [0.919, 0.979] | 224/134/224/4 | 0.872 | 0.966 | 0.944 | 0.775 [0.697, 0.846] | 224/134/224/4 |  |
| alone: evilgenie | 0.827 [0.785, 0.881] | 252/126/126/8 | 0.872 [0.831, 0.911] | 266/133/133/10 | 0.490 | 0.926 | 0.640 | 0.885 [0.845, 0.915] | 266/133/133/10 | too few agents or runs in a class |
| ours | 0.944 [0.928, 0.958] | 1948/717/1072/5 | 0.950 [0.940, 0.960] | 1973/722/1074/9 | 0.772 | 0.981 | 0.843 | 0.860 [0.838, 0.880] | 1948/717/1072/5 | too few agents or runs in a class |
| external | 0.873 [0.846, 0.905] | 3317/2347/3187/14 | 0.921 [0.911, 0.930] | 3323/2350/3190/15 | 0.721 | 0.935 | 0.814 | 0.784 [0.752, 0.816] | 3323/2350/3190/15 | too few agents or runs in a class |
| all | 0.891 [0.871, 0.914] | 5271/3069/4261/19 | 0.920 [0.912, 0.927] | 5296/3072/4264/23 | 0.787 | 0.964 | 0.850 | 0.795 [0.756, 0.822] | 5277/3072/4264/20 | too few agents or runs in a class |
| leave out: controls | 0.772 [0.720, 0.819] | 360/201/360/3 | 0.684 [0.637, 0.727] | 600/343/600/5 | 0.681 | 0.802 | 0.541 | 0.682 [0.639, 0.726] | 600/343/600/5 | author absent from the training sources |
| leave out: board_seed | 0.762 [0.666, 0.851] | 134/56/43/1 | 0.465 [0.361, 0.570] | 134/56/43/1 | 0.602 | 0.856 | 0.846 | 0.749 [0.635, 0.843] | 134/56/43/1 |  |
| leave out: solo_design | 0.867 [0.773, 0.929] | 171/79/57/2 | 0.821 [0.747, 0.885] | 177/84/59/3 | 0.820 | 0.899 | 0.662 | 0.693 [0.576, 0.801] | 171/79/57/2 | too few in a class on one side |
| leave out: spar_team | 0.717 [0.649, 0.820] | 450/109/135/3 | 0.792 [0.727, 0.855] | 450/109/135/3 | 0.717 | 0.873 | 0.831 | 0.735 [0.669, 0.795] | 450/109/135/3 |  |
| leave out: spar_mixed | 0.789 [0.625, 0.894] | 162/12/51/2 | 0.915 [0.855, 0.959] | 209/12/51/6 | 0.891 | 0.984 | 0.783 | 0.638 [0.492, 0.771] | 162/12/51/2 | too few in a class on one side |
| leave out: spar_expl | 0.949 [0.912, 0.981] | 313/71/96/3 | 0.906 [0.855, 0.951] | 313/71/96/3 | 0.844 | 0.987 | 0.855 | 0.646 [0.547, 0.747] | 313/71/96/3 |  |
| leave out: spar_solo | 0.923 [0.845, 0.983] | 60/17/60/2 | 0.852 [0.769, 0.918] | 90/47/90/3 | 0.865 | 0.943 | 0.793 | 0.930 [0.844, 0.992] | 60/17/60/2 | too few in a class on one side |
| leave out: goodfire | 0.348 [0.308, 0.395] | 935/733/935/1 | 0.648 [0.624, 0.670] | 2833/2083/2833/3 | 0.625 | 0.678 | 0.666 | 0.600 [0.574, 0.625] | 2833/2083/2833/3 | too few in a class on one side |
| leave out: cheatbench | 0.919 [0.816, 0.993] | 35/9/35/1 | 0.805 [0.751, 0.858] | 224/134/224/4 | 0.670 | 0.582 | 0.642 | 0.664 [0.548, 0.773] | 224/134/224/4 | author absent from the training sources |
| leave out: evilgenie | 0.491 [0.265, 0.768] | 50/25/25/2 | 0.652 [0.607, 0.700] | 266/133/133/10 | 0.587 | 0.554 | 0.531 | 0.704 [0.593, 0.774] | 266/133/133/10 | author absent from the training sources |
| ours to external | n/a |  | 0.630 [0.609, 0.649] | 3323/2350/3190/15 | 0.625 | 0.728 | 0.676 | 0.541 [0.461, 0.603] | 3323/2350/3190/15 | too few in a class on one side |
| external to ours | n/a |  | 0.397 [0.366, 0.428] | 1973/722/1074/9 | 0.623 | 0.600 | 0.538 | 0.443 [0.406, 0.480] | 1948/717/1072/5 | too few in a class on one side |

### coordination posted

| scheme | A mean | A n/pos/runs/auth | B.I | B.I n/pos/runs/auth | B.I null 95th | B.I TF-IDF | B.I metadata | B.II mean | B.II n/pos/runs/auth | note |
|---|---|---|---|---|---|---|---|---|---|---|
| alone: board_seed | 0.894 [0.784, 0.974] | 134/87/43/1 | 0.894 [0.784, 0.974] | 134/87/43/1 | 0.722 | 0.895 | 0.929 | n/a |  | one author only: nothing to hold out |
| alone: solo_design | 0.814 [0.686, 0.910] | 171/37/57/2 | 0.747 [0.579, 0.878] | 177/38/59/3 | 0.558 | 0.774 | 0.753 | 0.715 [0.597, 0.803] | 171/37/57/2 | too few agents or runs in a class |
| alone: spar_team | 0.752 [0.649, 0.841] | 450/406/135/3 | 0.909 [0.843, 0.963] | 450/406/135/3 | 0.756 | 0.889 | 0.794 | 0.690 [0.633, 0.755] | 450/406/135/3 |  |
| alone: spar_mixed | n/a |  | 0.600 [0.464, 0.726] | 209/205/51/6 | 0.805 | 0.726 | 0.627 | n/a |  | too few agents or runs in a class |
| ours | 0.944 [0.922, 0.962] | 945/719/286/3 | 0.969 [0.958, 0.979] | 970/736/288/7 | 0.883 | 0.957 | 0.924 | 0.904 [0.877, 0.930] | 945/719/286/3 | too few agents or runs in a class |
| external | n/a |  | n/a |  |  |  |  | n/a |  | no labelled source |
| all | 0.944 [0.922, 0.962] | 945/719/286/3 | 0.969 [0.958, 0.979] | 970/736/288/7 | 0.883 | 0.957 | 0.924 | 0.904 [0.877, 0.930] | 945/719/286/3 | identical source set to 'ours' |
| leave out: board_seed | 0.908 [0.846, 0.955] | 134/87/43/1 | 0.942 [0.885, 0.980] | 134/87/43/1 | 0.905 | 0.889 | 0.929 | 0.818 [0.736, 0.892] | 134/87/43/1 |  |
| leave out: solo_design | 0.705 [0.550, 0.825] | 171/37/57/2 | 0.758 [0.647, 0.853] | 177/38/59/3 | 0.736 | 0.496 | 0.444 | 0.763 [0.615, 0.870] | 171/37/57/2 | author absent from the training sources |
| leave out: spar_team | 0.873 [0.796, 0.953] | 300/291/90/2 | 0.863 [0.812, 0.907] | 450/406/135/3 | 0.808 | 0.654 | 0.726 | 0.805 [0.721, 0.882] | 450/406/135/3 | too few in a class on one side |
| leave out: spar_mixed | n/a |  | 0.854 [0.763, 0.924] | 209/205/51/6 | 0.833 | 0.470 | 0.423 | n/a |  | too few in a class on one side |
| ours to external | n/a |  | n/a |  |  |  |  | n/a |  | no labelled source |
| external to ours | n/a |  | n/a |  |  |  |  | n/a |  | no labelled source |

### collusion crossed

| scheme | A mean | A n/pos/runs/auth | B.I | B.I n/pos/runs/auth | B.I null 95th | B.I TF-IDF | B.I metadata | B.II mean | B.II n/pos/runs/auth | note |
|---|---|---|---|---|---|---|---|---|---|---|
| alone: board_seed | 0.901 [0.814, 0.964] | 134/48/43/1 | 0.901 [0.814, 0.964] | 134/48/43/1 | 0.725 | 0.950 | 0.852 | n/a |  | one author only: nothing to hold out |
| alone: spar_team | 0.842 [0.710, 0.960] | 450/67/135/3 | 0.910 [0.857, 0.959] | 450/67/135/3 | 0.716 | 0.980 | 0.920 | 0.827 [0.762, 0.886] | 450/67/135/3 |  |
| alone: spar_mixed | 0.818 [0.696, 0.961] | 36/7/12/1 | 0.967 [0.920, 1.000] | 209/7/51/6 | 0.941 | 0.950 | 0.960 | n/a |  | too few agents or runs in a class |
| ours | 0.809 [0.704, 0.895] | 774/122/229/3 | 0.933 [0.896, 0.964] | 793/122/229/6 | 0.758 | 0.984 | 0.908 | 0.843 [0.780, 0.904] | 774/122/229/3 | too few agents or runs in a class |
| external | n/a |  | n/a |  |  |  |  | n/a |  | no labelled source |
| all | 0.809 [0.704, 0.895] | 774/122/229/3 | 0.933 [0.896, 0.964] | 793/122/229/6 | 0.758 | 0.984 | 0.908 | 0.843 [0.780, 0.904] | 774/122/229/3 | identical source set to 'ours' |
| leave out: board_seed | 0.857 [0.769, 0.929] | 134/48/43/1 | 0.888 [0.810, 0.945] | 134/48/43/1 | 0.680 | 0.955 | 0.669 | 0.750 [0.653, 0.843] | 134/48/43/1 |  |
| leave out: spar_team | 0.845 [0.727, 0.932] | 150/37/45/1 | 0.814 [0.712, 0.899] | 450/67/135/3 | 0.767 | 0.657 | 0.651 | 0.778 [0.714, 0.844] | 300/30/90/2 | too few in a class on one side |
| leave out: spar_mixed | 0.458 [0.086, 0.712] | 36/7/12/1 | 0.907 [0.812, 0.980] | 209/7/51/6 | 0.956 | 0.932 | 0.647 | 0.695 [0.487, 0.879] | 36/7/12/1 | too few in a class on one side |
| ours to external | n/a |  | n/a |  |  |  |  | n/a |  | no labelled source |
| external to ours | n/a |  | n/a |  |  |  |  | n/a |  | no labelled source |

## Source-to-source transfer

Train on every labelled agent of the row source, score every labelled agent of the column source (think pair, all layers). The diagonal is B.I within the source. Figures: `heatmap_cheating.png`, `heatmap_board.png`; numbers with intervals in `heatmap.csv`.

![cheating transfer](heatmap_cheating.png)

![coordination and collusion transfer](heatmap_board.png)

**cheating** (rows train, columns test)

| train \ test | controls | board_seed | solo_design | spar_team | spar_mixed | spar_expl | spar_solo | goodfire | cheatbench | evilgenie |
|---|---|---|---|---|---|---|---|---|---|---|
| controls | 0.960 | 0.584 | 0.871 | 0.686 | 0.596 | 0.636 | 0.831 | 0.702 | 0.516 | 0.592 |
| board_seed | 0.574 | 0.880 | 0.717 | 0.746 | 0.887 | 0.828 | 0.812 | 0.744 | 0.878 | 0.470 |
| solo_design | 0.727 | 0.754 | 0.914 | 0.830 | 0.871 | 0.725 | 0.907 | 0.684 | 0.432 | 0.430 |
| spar_team | 0.683 | 0.872 | 0.883 | 0.927 | 0.883 | 0.926 | 0.880 | 0.553 | 0.579 | 0.471 |
| spar_mixed | 0.586 | 0.590 | 0.561 | 0.734 | 0.734 | 0.832 | 0.701 | 0.549 | 0.899 | 0.549 |
| spar_expl | 0.671 | 0.848 | 0.777 | 0.853 | 0.902 | 0.955 | 0.931 | 0.552 | 0.868 | 0.506 |
| spar_solo | 0.652 | 0.742 | 0.799 | 0.774 | 0.864 | 0.829 | 0.962 | 0.505 | 0.853 | 0.478 |
| goodfire | 0.477 | 0.691 | 0.465 | 0.638 | 0.535 | 0.530 | 0.599 | 0.921 | 0.529 | 0.652 |
| cheatbench | 0.653 | 0.462 | 0.608 | 0.556 | 0.804 | 0.672 | 0.548 | 0.588 | 0.951 | 0.531 |
| evilgenie | 0.413 | 0.584 | 0.273 | 0.537 | 0.462 | 0.483 | 0.392 | 0.705 | 0.575 | 0.872 |

**coordination posted** (rows train, columns test)

| train \ test | board_seed | solo_design | spar_team | spar_mixed |
|---|---|---|---|---|
| board_seed | 0.894 | 0.693 | 0.838 | 0.729 |
| solo_design | 0.833 | 0.747 | 0.750 | 0.468 |
| spar_team | 0.897 | 0.809 | 0.909 | 0.828 |
| spar_mixed | 0.872 | 0.626 | 0.569 | 0.600 |

**collusion crossed** (rows train, columns test)

| train \ test | board_seed | spar_team | spar_mixed |
|---|---|---|---|
| board_seed | 0.901 | 0.747 | 0.933 |
| spar_team | 0.901 | 0.910 | 0.972 |
| spar_mixed | 0.679 | 0.663 | 0.967 |

## Per step (layer 38, every turn)

One row per turn; a turn is positive from the agent's onset turn on (levels.py per-call onsets for our sets: cheating at the first write of planted content, else the opened call; coordination at the first post; collusion at team_probes' onset proxy. controls_600: labels.csv onset_turn. Goodfire: the first turn at or after the first positive passage of judge_passages.jsonl). Agents that reach the label with no usable onset are dropped. Probe: training-fold PCA (exact, from the streamed training covariance) to 128 components, z-scoring, logistic regression, 5 folds grouped by run. Agent AUROC uses the mean of the agent's turn scores. Detection delay: first turn above the 5% false-positive threshold of negative turns, minus the onset turn. Null: agents' (reaches, onset fraction) permuted within source, author and arm, 20 draws.

| label | scheme | features | turn AUROC | turn null 95th | agent AUROC | agent null 95th | detected / reaching | delay median (IQR) | fired before onset | turns / pos | agents / pos | dropped (no onset) | runs | authors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cheating | ours | think pair, layer 38 (per turn) | 0.900 [0.888, 0.912] | 0.821 | 0.871 [0.852, 0.889] | 0.770 | 585/704 | 1 (-1 to 4) | 149 | 71113/16060 | 1955/704 | 18 | 1061 | 9 |
| cheating | ours | turn index | 0.601 [0.563, 0.637] |  | 0.488 [0.453, 0.522] |  | 14/704 | 64 (53 to 71) | 1 | 71113/16060 | 1955/704 | 18 | 1061 | 9 |
| cheating | ours | turn index + fraction | 0.682 [0.666, 0.697] |  | 0.524 [0.491, 0.563] |  | 574/704 | 12 (4 to 20) | 8 | 71113/16060 | 1955/704 | 18 | 1061 | 9 |
| cheating | ours | metadata + turn index + fraction | 0.893 [0.878, 0.906] |  | 0.833 [0.812, 0.855] |  | 472/704 | 6 (1 to 14) | 102 | 71113/16060 | 1955/704 | 18 | 1061 | 9 |
| cheating | all (ours + goodfire: the sources with per-call labels) | think pair, layer 38 (per turn) | 0.910 [0.901, 0.918] | 0.847 | 0.878 [0.867, 0.888] | 0.784 | 2119/2767 | 2 (0 to 7) | 285 | 211113/90005 | 4768/2767 | 38 | 3874 | 11 |
| cheating | all (ours + goodfire: the sources with per-call labels) | turn index | 0.623 [0.605, 0.641] |  | 0.585 [0.565, 0.605] |  | 100/2767 | 97 (18 to 117) | 19 | 211113/90005 | 4768/2767 | 38 | 3874 | 11 |
| cheating | all (ours + goodfire: the sources with per-call labels) | turn index + fraction | 0.685 [0.677, 0.694] |  | 0.413 [0.392, 0.431] |  | 2369/2767 | 17 (7 to 30) | 56 | 211113/90005 | 4768/2767 | 38 | 3874 | 11 |
| cheating | all (ours + goodfire: the sources with per-call labels) | metadata + turn index + fraction | 0.881 [0.871, 0.889] |  | 0.846 [0.833, 0.859] |  | 2134/2767 | 9 (4 to 19) | 238 | 211113/90005 | 4768/2767 | 38 | 3874 | 11 |
| coordination posted | ours (= all: no external source has a board) | think pair, layer 38 (per turn) | 0.976 [0.971, 0.982] | 0.875 | 0.960 [0.945, 0.973] | 0.886 | 669/691 | 1 (0 to 1) | 135 | 41368/22927 | 925/691 | 45 | 283 | 7 |
| coordination posted | ours (= all: no external source has a board) | turn index | 0.799 [0.770, 0.826] |  | 0.678 [0.624, 0.731] |  | 269/691 | 34 (13 to 43) | 27 | 41368/22927 | 925/691 | 45 | 283 | 7 |
| coordination posted | ours (= all: no external source has a board) | turn index + fraction | 0.798 [0.773, 0.823] |  | 0.677 [0.622, 0.731] |  | 318/691 | 36 (16 to 44) | 27 | 41368/22927 | 925/691 | 45 | 283 | 7 |
| coordination posted | ours (= all: no external source has a board) | metadata + turn index + fraction | 0.927 [0.909, 0.941] |  | 0.924 [0.899, 0.946] |  | 614/691 | 10 (3 to 17) | 73 | 41368/22927 | 925/691 | 45 | 283 | 7 |
| collusion crossed | ours (= all: no external source has a board) | think pair, layer 38 (per turn) | 0.926 [0.906, 0.946] | 0.798 | 0.930 [0.891, 0.963] | 0.755 | 115/122 | 0 (-9 to 6) | 54 | 37165/4087 | 793/122 | 0 | 229 | 6 |
| collusion crossed | ours (= all: no external source has a board) | turn index | 0.719 [0.661, 0.773] |  | 0.683 [0.595, 0.764] |  | 9/122 | 64 (47 to 71) | 1 | 37165/4087 | 793/122 | 0 | 229 | 6 |
| collusion crossed | ours (= all: no external source has a board) | turn index + fraction | 0.705 [0.682, 0.728] |  | 0.404 [0.325, 0.492] |  | 120/122 | 26 (17 to 42) | 1 | 37165/4087 | 793/122 | 0 | 229 | 6 |
| collusion crossed | ours (= all: no external source has a board) | metadata + turn index + fraction | 0.915 [0.886, 0.941] |  | 0.912 [0.870, 0.947] |  | 91/122 | 15 (-0 to 22) | 23 | 37165/4087 | 793/122 | 0 | 229 | 6 |

Turn AUROC within each source of the scheme (same scores):

| label | scheme | source | turn AUROC | agent AUROC | turns / pos | agents / pos | turn index + fraction, turn AUROC |
|---|---|---|---|---|---|---|---|
| cheating | ours | board_seed | 0.868 | 0.897 | 7281/1927 | 134/56 | 0.756 |
| cheating | ours | controls | 0.858 | 0.765 | 10834/3759 | 593/336 | 0.711 |
| cheating | ours | solo_design | 0.899 | 0.878 | 6000/1838 | 173/80 | 0.667 |
| cheating | ours | spar_expl | 0.930 | 0.916 | 14580/2620 | 313/71 | 0.686 |
| cheating | ours | spar_mixed | 0.911 | 0.886 | 12373/511 | 209/12 | 0.649 |
| cheating | ours | spar_solo | 0.794 | 0.896 | 2600/1165 | 84/41 | 0.641 |
| cheating | ours | spar_team | 0.903 | 0.913 | 17445/4240 | 449/108 | 0.620 |
| cheating | all (ours + goodfire: the sources with per-call labels) | board_seed | 0.828 | 0.805 | 7281/1927 | 134/56 | 0.758 |
| cheating | all (ours + goodfire: the sources with per-call labels) | controls | 0.747 | 0.636 | 10834/3759 | 593/336 | 0.710 |
| cheating | all (ours + goodfire: the sources with per-call labels) | goodfire | 0.916 | 0.884 | 140000/73945 | 2813/2063 | 0.722 |
| cheating | all (ours + goodfire: the sources with per-call labels) | solo_design | 0.866 | 0.820 | 6000/1838 | 173/80 | 0.670 |
| cheating | all (ours + goodfire: the sources with per-call labels) | spar_expl | 0.877 | 0.825 | 14580/2620 | 313/71 | 0.691 |
| cheating | all (ours + goodfire: the sources with per-call labels) | spar_mixed | 0.911 | 0.885 | 12373/511 | 209/12 | 0.658 |
| cheating | all (ours + goodfire: the sources with per-call labels) | spar_solo | 0.757 | 0.779 | 2600/1165 | 84/41 | 0.654 |
| cheating | all (ours + goodfire: the sources with per-call labels) | spar_team | 0.854 | 0.834 | 17445/4240 | 449/108 | 0.625 |
| coordination posted | ours (= all: no external source has a board) | board_seed | 0.964 | 0.954 | 6735/2617 | 125/78 | 0.852 |
| coordination posted | ours (= all: no external source has a board) | solo_design | 0.899 | 0.767 | 6086/552 | 177/38 | 0.668 |
| coordination posted | ours (= all: no external source has a board) | spar_mixed | 0.966 | 0.897 | 12056/9347 | 206/202 | 0.809 |
| coordination posted | ours (= all: no external source has a board) | spar_team | 0.977 | 0.941 | 16491/10411 | 417/373 | 0.869 |
| collusion crossed | ours (= all: no external source has a board) | board_seed | 0.882 | 0.893 | 7281/1536 | 134/48 | 0.766 |
| collusion crossed | ours (= all: no external source has a board) | spar_mixed | 0.924 | 0.950 | 12373/395 | 209/7 | 0.626 |
| collusion crossed | ours (= all: no external source has a board) | spar_team | 0.934 | 0.930 | 17511/2156 | 450/67 | 0.694 |

## Architectures

A third axis: the probe architecture, on the main cells, with the same folds and held-out-author splits as the linear matrix (so the TF-IDF and metadata baselines of the linear tables apply unchanged). Per-step input: the layer-38 think pair of every turn, standardised with the training turns. Architectures 2 to 6 share one training objective, the binary part of the attention monitor's: per-turn cross-entropy (0 before the onset, 1 from it on, weight 1 and 2; weight 0 on the turns of an agent that reaches the label with no usable onset) plus the trajectory label at the last step; AdamW (lr 1e-3, weight decay 0.1), batch 32, at most 30 epochs, early stopping (patience 5) on 20% of the training runs held out. Their trajectory score is the last step's output, their per-step score the output at each step. Architecture 7 trains on the turns of training agents with a usable onset, so positives without one (most of CheatBench, all of EvilGenie) leave its training set; where no training positive has an onset it cannot run. Architecture 8 trains on agents (positive class weighted to balance). Trained models (2 to 6, 8) ran with two seeds; the table gives the mean over seeds and the seed range. Gap: AUROC on the training agents (fit set) minus held-out AUROC. Turn AUROC and the detection delay (median turns from onset to the first score above the 5% false-positive threshold of held-out negative turns) are for per-step models, on held-out agents with a usable onset. Architecture 9 stacks 1, 2, 5 and TF-IDF with a logistic regression fitted on inner 3-fold out-of-fold scores of the outer training agents only. No permutation null was run for architectures 2 to 9.

- **linear**: 1 linear: PCA + z + L2 logistic, mean-pooled think pair (all layers)
- **mlp_mean**: 2 MLP-pool: per-step MLP, causal running mean, 2-layer MLP
- **mlp_max**: 3 MLP-max: per-step MLP, causal running max, 2-layer MLP
- **attn_pool**: 4 attention pooling: per-step MLP, one learned query over the steps so far, MLP
- **attn_monitor**: 5 attention monitor (attn_probe.build_model: 2 causal self-attention layers, 4 heads, d 128)
- **gru**: 6 GRU: per-step MLP, one-layer GRU (hidden 128), MLP
- **linear_step**: 7 linear per step (PCA + z + L2 logistic on turns), running mean of its scores
- **mlp_traj**: 8 MLP on the mean-pooled all-layers vector (PCA 256 first)
- **stack**: 9 stacked: logistic regression over held-out scores of 1, 2, 5 and TF-IDF (inner folds)

### cheating: ours

B.I counts: 1973 agents, 722 positive, 1074 runs, 9 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| stack | 0.984 | [0.978, 0.989] |  |  | -0.004 | not run |  |  |  | 5 |
| attn_monitor | 0.969 (0.968 to 0.970) | [0.960, 0.976] | 0.964 | 0.0 | 0.025 | 0.875 (0.865 to 0.885) | 0.899 | 0.116 | 5 | 1762726 |
| mlp_max | 0.966 (0.966 to 0.967) | [0.958, 0.974] | 0.965 | 0.0 | 0.029 | 0.906 (0.902 to 0.911) | 0.919 | 0.088 | 5 | 1327489 |
| gru | 0.963 (0.959 to 0.966) | [0.957, 0.974] | 0.960 | 0.0 | 0.028 | 0.872 (0.852 to 0.892) | 0.897 | 0.115 | 5 | 1426561 |
| attn_pool | 0.962 (0.956 to 0.969) | [0.960, 0.976] | 0.957 | 0.0 | 0.028 | 0.881 (0.877 to 0.885) | 0.905 | 0.108 | 5 | 1360641 |
| mlp_traj | 0.957 (0.957 to 0.958) | [0.948, 0.966] |  |  | 0.042 | 0.850 (0.839 to 0.861) |  | 0.150 | 5 | 33025 |
| linear | 0.950 | [0.940, 0.960] |  |  |  | 0.860 |  |  | 5 |  |
| mlp_mean | 0.942 (0.936 to 0.948) | [0.936, 0.959] | 0.951 | 1.0 | 0.039 | 0.849 (0.841 to 0.856) | 0.886 | 0.137 | 5 | 1327489 |
| linear_step | 0.869 | [0.850, 0.888] | 0.902 | 6.0 | 0.012 | 0.787 | 0.816 | 0.097 | 5 | 129 |
| baseline: TF-IDF rendered text | 0.981 |  |  |  |  | 0.945 |  |  |  |  |
| baseline: metadata | 0.843 |  |  |  |  | 0.632 |  |  |  |  |

### cheating: all

B.I counts: 5296 agents, 3072 positive, 4264 runs, 23 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| stack | 0.967 | [0.963, 0.972] |  |  | -0.004 | not run |  |  |  | 5 |
| mlp_max | 0.943 (0.941 to 0.944) | [0.938, 0.950] | 0.951 | 0.0 | 0.032 | 0.766 (0.761 to 0.771) | 0.880 | 0.207 | 20 | 1327489 |
| attn_monitor | 0.941 (0.937 to 0.945) | [0.930, 0.943] | 0.949 | 0.5 | 0.025 | 0.782 (0.768 to 0.797) | 0.884 | 0.187 | 20 | 1762726 |
| gru | 0.938 (0.936 to 0.940) | [0.929, 0.942] | 0.950 | 0.0 | 0.021 | 0.772 (0.771 to 0.774) | 0.878 | 0.188 | 20 | 1426561 |
| mlp_traj | 0.937 (0.937 to 0.938) | [0.930, 0.943] |  |  | 0.047 | 0.806 (0.782 to 0.830) |  | 0.179 | 20 | 33025 |
| attn_pool | 0.937 (0.936 to 0.938) | [0.931, 0.944] | 0.947 | 0.0 | 0.022 | 0.814 (0.805 to 0.823) | 0.887 | 0.150 | 20 | 1360641 |
| mlp_mean | 0.920 (0.916 to 0.925) | [0.908, 0.923] | 0.942 | 1.0 | 0.022 | 0.801 (0.783 to 0.819) | 0.868 | 0.154 | 20 | 1327489 |
| linear | 0.920 | [0.912, 0.927] |  |  |  | 0.795 |  |  | 20 |  |
| linear_step | 0.829 | [0.818, 0.839] | 0.902 | 7.0 | 0.060 | 0.742 | 0.821 | 0.147 | 20 | 129 |
| baseline: TF-IDF rendered text | 0.964 |  |  |  |  | 0.895 |  |  |  |  |
| baseline: metadata | 0.850 |  |  |  |  | 0.684 |  |  |  |  |

### coordination posted: ours

B.I counts: 970 agents, 736 positive, 288 runs, 7 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.990 (0.990 to 0.990) | [0.984, 0.995] | 0.993 | 0.0 | 0.008 | 0.977 (0.977 to 0.977) | 0.985 | 0.020 | 3 | 1327489 |
| attn_pool | 0.989 (0.988 to 0.990) | [0.984, 0.995] | 0.994 | 0.0 | 0.008 | 0.975 (0.972 to 0.978) | 0.983 | 0.019 | 3 | 1360641 |
| stack | 0.988 | [0.982, 0.994] |  |  | 0.000 | not run |  |  |  | 5 |
| gru | 0.987 (0.987 to 0.988) | [0.979, 0.993] | 0.992 | 0.0 | 0.007 | 0.938 (0.935 to 0.941) | 0.963 | 0.055 | 3 | 1426561 |
| mlp_mean | 0.984 (0.984 to 0.984) | [0.976, 0.991] | 0.984 | 1.0 | 0.011 | 0.960 (0.960 to 0.960) | 0.968 | 0.030 | 3 | 1327489 |
| attn_monitor | 0.983 (0.979 to 0.988) | [0.959, 0.992] | 0.993 | 0.0 | 0.012 | 0.953 (0.948 to 0.959) | 0.976 | 0.043 | 3 | 1762726 |
| linear | 0.969 | [0.958, 0.979] |  |  |  | 0.904 |  |  | 3 |  |
| linear_step | 0.958 | [0.943, 0.971] | 0.963 | 3.0 | 0.008 | 0.885 | 0.933 | 0.071 | 3 | 129 |
| mlp_traj | 0.954 (0.954 to 0.954) | [0.937, 0.968] |  |  | 0.046 | 0.836 (0.823 to 0.848) |  | 0.164 | 3 | 33025 |
| baseline: TF-IDF rendered text | 0.957 |  |  |  |  | 0.863 |  |  |  |  |
| baseline: metadata | 0.924 |  |  |  |  | 0.815 |  |  |  |  |

### collusion crossed: ours

B.I counts: 793 agents, 122 positive, 229 runs, 6 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| stack | 0.984 | [0.971, 0.994] |  |  | -0.004 | not run |  |  |  | 5 |
| mlp_traj | 0.945 (0.939 to 0.950) | [0.923, 0.972] |  |  | 0.055 | 0.668 (0.634 to 0.703) |  | 0.332 | 3 | 32641 |
| mlp_mean | 0.933 (0.929 to 0.937) | [0.909, 0.961] | 0.930 | 3.8 | 0.051 | 0.711 (0.684 to 0.739) | 0.820 | 0.272 | 3 | 1327489 |
| linear | 0.933 | [0.896, 0.964] |  |  |  | 0.843 |  |  | 3 |  |
| mlp_max | 0.932 (0.928 to 0.937) | [0.888, 0.960] | 0.928 | 1.0 | 0.051 | 0.748 (0.705 to 0.791) | 0.833 | 0.247 | 3 | 1327489 |
| attn_monitor | 0.932 (0.924 to 0.940) | [0.900, 0.970] | 0.929 | 0.5 | 0.055 | 0.463 (0.435 to 0.491) | 0.684 | 0.516 | 3 | 1762726 |
| gru | 0.930 (0.922 to 0.938) | [0.903, 0.967] | 0.926 | 1.5 | 0.055 | 0.527 (0.526 to 0.528) | 0.702 | 0.447 | 3 | 1426561 |
| linear_step | 0.930 | [0.891, 0.963] | 0.905 | 10.0 | 0.038 | 0.748 | 0.823 | 0.220 | 3 | 129 |
| attn_pool | 0.917 (0.917 to 0.917) | [0.879, 0.948] | 0.912 | 4.0 | 0.041 | 0.626 (0.572 to 0.681) | 0.759 | 0.344 | 3 | 1360641 |
| baseline: TF-IDF rendered text | 0.984 |  |  |  |  | 0.856 |  |  |  |  |
| baseline: metadata | 0.908 |  |  |  |  | 0.760 |  |  |  |  |

### cheating: alone: controls

B.I counts: 600 agents, 343 positive, 600 runs, 5 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_pool | 0.980 (0.978 to 0.982) | [0.966, 0.987] | 0.975 | 0.0 | 0.019 | 0.949 (0.942 to 0.957) | 0.951 | 0.050 | 5 | 1360641 |
| mlp_max | 0.971 (0.964 to 0.979) | [0.948, 0.976] | 0.970 | 0.0 | 0.026 | 0.956 (0.950 to 0.961) | 0.959 | 0.043 | 5 | 1327489 |
| attn_monitor | 0.971 (0.971 to 0.971) | [0.957, 0.982] | 0.968 | 0.0 | 0.027 | 0.916 (0.915 to 0.918) | 0.935 | 0.078 | 5 | 1762726 |
| gru | 0.969 (0.967 to 0.971) | [0.957, 0.983] | 0.967 | 0.0 | 0.027 | 0.919 (0.902 to 0.936) | 0.933 | 0.075 | 5 | 1426561 |
| linear | 0.960 | [0.944, 0.973] |  |  |  | 0.913 |  |  | 5 |  |
| mlp_traj | 0.953 (0.944 to 0.962) | [0.926, 0.961] |  |  | 0.047 | 0.883 (0.880 to 0.886) |  | 0.117 | 5 | 24833 |
| mlp_mean | 0.950 (0.948 to 0.952) | [0.931, 0.963] | 0.962 | 0.0 | 0.039 | 0.913 (0.910 to 0.915) | 0.935 | 0.084 | 5 | 1327489 |
| linear_step | 0.874 | [0.843, 0.900] | 0.903 | 2.0 | 0.036 | 0.831 | 0.894 | 0.080 | 5 | 129 |
| baseline: TF-IDF rendered text | 0.990 |  |  |  |  | 0.971 |  |  |  |  |
| baseline: metadata | 0.690 |  |  |  |  | 0.535 |  |  |  |  |

### cheating: alone: board_seed

B.I counts: 134 agents, 56 positive, 43 runs, 1 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.880 | [0.809, 0.935] |  |  |  | not run |  |  |  |  |
| mlp_traj | 0.858 (0.855 to 0.862) | [0.789, 0.923] |  |  | 0.139 | not run |  |  |  | 5761 |
| linear_step | 0.844 | [0.761, 0.917] | 0.853 | 13.0 | 0.091 | not run |  |  |  | 129 |
| mlp_max | 0.834 (0.831 to 0.837) | [0.730, 0.923] | 0.898 | 11.8 | 0.140 | not run |  |  |  | 1327489 |
| attn_pool | 0.833 (0.824 to 0.843) | [0.755, 0.917] | 0.891 | 10.5 | 0.112 | not run |  |  |  | 1360641 |
| gru | 0.825 (0.804 to 0.845) | [0.760, 0.912] | 0.894 | 7.5 | 0.123 | not run |  |  |  | 1426561 |
| mlp_mean | 0.819 (0.808 to 0.831) | [0.739, 0.905] | 0.889 | 12.5 | 0.132 | not run |  |  |  | 1327489 |
| attn_monitor | 0.778 (0.761 to 0.794) | [0.691, 0.887] | 0.876 | 6.0 | 0.119 | not run |  |  |  | 1762726 |
| baseline: TF-IDF rendered text | 0.905 |  |  |  |  |  |  |  |  |  |
| baseline: metadata | 0.826 |  |  |  |  |  |  |  |  |  |

### cheating: alone: solo_design

B.I counts: 177 agents, 84 positive, 59 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.914 | [0.860, 0.960] |  |  |  | 0.630 |  |  | 2 |  |
| mlp_traj | 0.909 (0.902 to 0.916) | [0.874, 0.955] |  |  | 0.091 | 0.544 (0.494 to 0.594) |  | 0.455 | 2 | 7297 |
| mlp_max | 0.893 (0.875 to 0.910) | [0.850, 0.955] | 0.873 | 1.0 | 0.091 | 0.607 (0.582 to 0.631) | 0.699 | 0.392 | 2 | 1327489 |
| attn_monitor | 0.874 (0.867 to 0.882) | [0.788, 0.931] | 0.865 | 1.5 | 0.080 | 0.510 (0.506 to 0.514) | 0.647 | 0.462 | 2 | 1762726 |
| gru | 0.873 (0.860 to 0.886) | [0.801, 0.949] | 0.887 | 1.0 | 0.092 | 0.519 (0.501 to 0.538) | 0.671 | 0.455 | 2 | 1426561 |
| attn_pool | 0.871 (0.868 to 0.873) | [0.802, 0.932] | 0.880 | 1.0 | 0.092 | 0.490 (0.487 to 0.493) | 0.652 | 0.449 | 2 | 1360641 |
| mlp_mean | 0.853 (0.842 to 0.863) | [0.796, 0.920] | 0.870 | 3.0 | 0.088 | 0.541 (0.508 to 0.574) | 0.680 | 0.449 | 2 | 1327489 |
| linear_step | 0.808 | [0.706, 0.888] | 0.831 | 5.0 | 0.145 | 0.443 | 0.647 | 0.526 | 2 | 129 |
| baseline: TF-IDF rendered text | 0.922 |  |  |  |  | 0.710 |  |  |  |  |
| baseline: metadata | 0.866 |  |  |  |  | 0.680 |  |  |  |  |

### cheating: alone: spar_team

B.I counts: 450 agents, 109 positive, 135 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear_step | 0.935 | [0.901, 0.961] | 0.932 | 6.0 | 0.039 | 0.841 | 0.862 | 0.137 | 3 | 129 |
| attn_pool | 0.933 (0.929 to 0.938) | [0.906, 0.965] | 0.916 | 0.2 | 0.058 | 0.784 (0.744 to 0.823) | 0.825 | 0.198 | 3 | 1360641 |
| linear | 0.927 | [0.885, 0.962] |  |  |  | 0.751 |  |  | 3 |  |
| mlp_mean | 0.922 (0.922 to 0.923) | [0.879, 0.958] | 0.923 | 0.5 | 0.065 | 0.765 (0.755 to 0.775) | 0.819 | 0.212 | 3 | 1327489 |
| mlp_max | 0.921 (0.920 to 0.923) | [0.881, 0.951] | 0.901 | 0.5 | 0.054 | 0.841 (0.826 to 0.856) | 0.860 | 0.152 | 3 | 1327489 |
| attn_monitor | 0.919 (0.911 to 0.927) | [0.886, 0.959] | 0.919 | 0.8 | 0.071 | 0.728 (0.724 to 0.733) | 0.765 | 0.249 | 3 | 1762726 |
| mlp_traj | 0.906 (0.899 to 0.913) | [0.859, 0.956] |  |  | 0.094 | 0.760 (0.738 to 0.782) |  | 0.240 | 3 | 18817 |
| gru | 0.886 (0.872 to 0.900) | [0.853, 0.937] | 0.896 | 2.0 | 0.090 | 0.651 (0.647 to 0.654) | 0.756 | 0.324 | 3 | 1426561 |
| baseline: TF-IDF rendered text | 0.982 |  |  |  |  | 0.918 |  |  |  |  |
| baseline: metadata | 0.934 |  |  |  |  | 0.865 |  |  |  |  |

### cheating: alone: spar_mixed

B.I counts: 209 agents, 12 positive, 51 runs, 6 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.862 (0.862 to 0.863) | [0.757, 0.943] | 0.904 | 13.0 | 0.051 | 0.680 (0.637 to 0.723) | 0.634 | 0.320 | 2 | 1327489 |
| mlp_mean | 0.837 (0.822 to 0.852) | [0.726, 0.938] | 0.893 | 23.5 | 0.082 | 0.581 (0.577 to 0.585) | 0.696 | 0.384 | 2 | 1327489 |
| attn_pool | 0.831 (0.778 to 0.885) | [0.817, 0.943] | 0.866 | -20.5 | 0.091 | 0.559 (0.548 to 0.570) | 0.689 | 0.403 | 2 | 1360641 |
| mlp_traj | 0.804 (0.784 to 0.824) | [0.698, 0.926] |  |  | 0.127 | 0.721 (0.713 to 0.729) |  | 0.265 | 2 | 8961 |
| gru | 0.796 (0.773 to 0.819) | [0.654, 0.942] | 0.866 | 21.0 | 0.074 | 0.428 (0.415 to 0.442) | 0.528 | 0.555 | 2 | 1426561 |
| attn_monitor | 0.764 (0.732 to 0.797) | [0.627, 0.914] | 0.837 | -42.0 | 0.048 | 0.521 (0.503 to 0.539) | 0.565 | 0.441 | 2 | 1762726 |
| linear_step | 0.758 | [0.587, 0.912] | 0.682 | -17.0 | 0.227 | 0.414 | 0.529 | 0.576 | 2 | 129 |
| linear | 0.734 | [0.606, 0.869] |  |  |  | 0.708 |  |  | 2 |  |
| baseline: TF-IDF rendered text | 0.937 |  |  |  |  | 0.330 |  |  |  |  |
| baseline: metadata | 0.837 |  |  |  |  | 0.497 |  |  |  |  |

### cheating: alone: spar_expl

B.I counts: 313 agents, 71 positive, 96 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.955 | [0.918, 0.984] |  |  |  | 0.658 |  |  | 3 |  |
| mlp_traj | 0.952 (0.944 to 0.959) | [0.932, 0.984] |  |  | 0.048 | 0.661 (0.592 to 0.729) |  | 0.339 | 3 | 13185 |
| attn_pool | 0.950 (0.950 to 0.950) | [0.913, 0.977] | 0.961 | 0.5 | 0.043 | 0.698 (0.669 to 0.727) | 0.728 | 0.291 | 3 | 1360641 |
| mlp_max | 0.945 (0.927 to 0.963) | [0.931, 0.987] | 0.961 | 0.0 | 0.051 | 0.606 (0.569 to 0.642) | 0.685 | 0.394 | 3 | 1327489 |
| gru | 0.940 (0.936 to 0.944) | [0.904, 0.974] | 0.963 | 0.0 | 0.055 | 0.477 (0.426 to 0.527) | 0.563 | 0.506 | 3 | 1426561 |
| mlp_mean | 0.940 (0.933 to 0.946) | [0.889, 0.967] | 0.964 | 0.0 | 0.045 | 0.692 (0.667 to 0.717) | 0.714 | 0.292 | 3 | 1327489 |
| attn_monitor | 0.936 (0.929 to 0.943) | [0.883, 0.968] | 0.963 | 0.0 | 0.053 | 0.491 (0.463 to 0.520) | 0.579 | 0.493 | 3 | 1762726 |
| linear_step | 0.933 | [0.887, 0.964] | 0.916 | 4.5 | 0.040 | 0.726 | 0.695 | 0.254 | 3 | 129 |
| baseline: TF-IDF rendered text | 0.970 |  |  |  |  | 0.893 |  |  |  |  |
| baseline: metadata | 0.936 |  |  |  |  | 0.804 |  |  |  |  |

### cheating: alone: spar_solo

B.I counts: 90 agents, 47 positive, 90 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_traj | 0.966 (0.965 to 0.967) | [0.917, 0.998] |  |  | 0.034 | 0.755 (0.729 to 0.781) |  | 0.244 | 2 | 3969 |
| linear | 0.962 | [0.915, 0.995] |  |  |  | 0.727 |  |  | 2 |  |
| linear_step | 0.959 | [0.916, 0.990] | 0.929 | 2.0 | 0.037 | 0.835 | 0.881 | 0.155 | 2 | 129 |
| mlp_mean | 0.941 (0.937 to 0.944) | [0.896, 0.982] | 0.894 | 3.0 | 0.025 | 0.630 (0.618 to 0.641) | 0.572 | 0.369 | 2 | 1327489 |
| mlp_max | 0.933 (0.918 to 0.949) | [0.904, 0.985] | 0.894 | 3.5 | 0.051 | 0.642 (0.613 to 0.671) | 0.578 | 0.358 | 2 | 1327489 |
| attn_pool | 0.929 (0.919 to 0.939) | [0.887, 0.979] | 0.887 | 4.0 | 0.041 | 0.627 (0.614 to 0.640) | 0.618 | 0.364 | 2 | 1360641 |
| attn_monitor | 0.895 (0.881 to 0.909) | [0.804, 0.947] | 0.848 | 3.5 | 0.041 | 0.482 (0.433 to 0.532) | 0.472 | 0.471 | 2 | 1762726 |
| gru | 0.889 (0.881 to 0.898) | [0.805, 0.943] | 0.866 | 2.2 | 0.045 | 0.490 (0.452 to 0.527) | 0.444 | 0.461 | 2 | 1426561 |
| baseline: TF-IDF rendered text | 0.985 |  |  |  |  | 0.837 |  |  |  |  |
| baseline: metadata | 0.926 |  |  |  |  | 0.624 |  |  |  |  |

### cheating: alone: goodfire

B.I counts: 2833 agents, 2083 positive, 2833 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.937 (0.936 to 0.938) | [0.927, 0.947] | 0.937 | 0.5 | 0.041 | 0.909 (0.899 to 0.919) | 0.901 | 0.058 | 3 | 1327489 |
| attn_pool | 0.936 (0.936 to 0.937) | [0.926, 0.944] | 0.938 | 0.0 | 0.028 | 0.910 (0.907 to 0.912) | 0.900 | 0.058 | 3 | 1360641 |
| gru | 0.933 (0.929 to 0.937) | [0.918, 0.939] | 0.933 | 0.5 | 0.022 | 0.909 (0.908 to 0.911) | 0.896 | 0.052 | 3 | 1426561 |
| mlp_mean | 0.933 (0.930 to 0.935) | [0.925, 0.944] | 0.933 | 1.5 | 0.022 | 0.894 (0.889 to 0.899) | 0.882 | 0.068 | 3 | 1327489 |
| attn_monitor | 0.930 (0.926 to 0.933) | [0.923, 0.942] | 0.932 | 1.0 | 0.027 | 0.913 (0.913 to 0.914) | 0.897 | 0.056 | 3 | 1762726 |
| linear | 0.921 | [0.910, 0.931] |  |  |  | 0.822 |  |  | 3 |  |
| mlp_traj | 0.916 (0.914 to 0.917) | [0.902, 0.925] |  |  | 0.075 | 0.829 (0.822 to 0.837) |  | 0.162 | 3 | 33025 |
| linear_step | 0.903 | [0.889, 0.914] | 0.908 | 8.0 | 0.005 | 0.890 | 0.867 | 0.014 | 3 | 129 |
| baseline: TF-IDF rendered text | 0.927 |  |  |  |  | 0.916 |  |  |  |  |
| baseline: metadata | 0.809 |  |  |  |  | 0.761 |  |  |  |  |

### cheating: alone: cheatbench

B.I counts: 224 agents, 134 positive, 224 runs, 4 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_traj | 0.964 (0.961 to 0.966) | [0.937, 0.990] |  |  | 0.036 | 0.789 (0.781 to 0.797) |  | 0.211 | 4 | 9345 |
| mlp_max | 0.961 (0.955 to 0.968) | [0.949, 0.983] | 0.818 | -0.5 | 0.019 | 0.823 (0.793 to 0.852) | 0.668 | 0.148 | 4 | 1327489 |
| mlp_mean | 0.961 (0.954 to 0.968) | [0.945, 0.984] | 0.796 | -2.5 | 0.020 | 0.667 (0.662 to 0.673) | 0.828 | 0.300 | 4 | 1327489 |
| attn_pool | 0.953 (0.934 to 0.971) | [0.953, 0.986] | 0.778 | -8.5 | 0.025 | 0.684 (0.659 to 0.710) | 0.820 | 0.302 | 4 | 1360641 |
| linear | 0.951 | [0.919, 0.979] |  |  |  | 0.775 |  |  | 4 |  |
| gru | 0.948 (0.934 to 0.961) | [0.901, 0.961] | 0.766 | 3.8 | 0.017 | 0.664 (0.600 to 0.727) | 0.577 | 0.290 | 4 | 1426561 |
| attn_monitor | 0.941 (0.932 to 0.949) | [0.921, 0.972] | 0.787 | -1.5 | 0.027 | 0.745 (0.719 to 0.772) | 0.684 | 0.222 | 4 | 1762726 |
| linear_step | 0.792 | [0.733, 0.847] | 0.558 | -2.5 | 0.161 | 0.659 | 0.502 | 0.293 | 4 | 129 |
| baseline: TF-IDF rendered text | 0.966 |  |  |  |  | 0.786 |  |  |  |  |
| baseline: metadata | 0.944 |  |  |  |  | 0.745 |  |  |  |  |

### cheating: alone: evilgenie

B.I counts: 266 agents, 133 positive, 133 runs, 10 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.872 | [0.831, 0.911] |  |  |  | 0.885 |  |  | 10 |  |
| mlp_traj | 0.848 (0.839 to 0.858) | [0.792, 0.881] |  |  | 0.148 | 0.848 (0.834 to 0.862) |  | 0.147 | 10 | 11137 |
| mlp_max | 0.761 (0.724 to 0.798) | [0.679, 0.769] |  |  | 0.161 | 0.797 (0.795 to 0.799) |  | 0.143 | 10 | 1327489 |
| attn_pool | 0.732 (0.717 to 0.748) | [0.698, 0.794] |  |  | 0.139 | 0.746 (0.730 to 0.762) |  | 0.128 | 10 | 1360641 |
| mlp_mean | 0.720 (0.714 to 0.725) | [0.668, 0.761] |  |  | 0.152 | 0.751 (0.749 to 0.753) |  | 0.114 | 10 | 1327489 |
| gru | 0.715 (0.710 to 0.719) | [0.651, 0.766] |  |  | 0.154 | 0.727 (0.724 to 0.730) |  | 0.152 | 10 | 1426561 |
| attn_monitor | 0.699 (0.683 to 0.716) | [0.626, 0.738] |  |  | 0.148 | 0.722 (0.716 to 0.727) |  | 0.096 | 10 | 1762726 |
| baseline: TF-IDF rendered text | 0.926 |  |  |  |  | 0.892 |  |  |  |  |
| baseline: metadata | 0.640 |  |  |  |  | 0.604 |  |  |  |  |

### coordination posted: alone: board_seed

B.I counts: 134 agents, 87 positive, 43 runs, 1 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.975 (0.971 to 0.979) | [0.936, 0.992] | 0.965 | 0.0 | 0.023 | not run |  |  |  | 1327489 |
| attn_pool | 0.965 (0.953 to 0.977) | [0.903, 0.986] | 0.953 | 0.0 | 0.029 | not run |  |  |  | 1360641 |
| mlp_mean | 0.951 (0.938 to 0.963) | [0.867, 0.983] | 0.944 | 3.5 | 0.037 | not run |  |  |  | 1327489 |
| gru | 0.945 (0.944 to 0.946) | [0.871, 0.988] | 0.918 | 0.0 | 0.043 | not run |  |  |  | 1426561 |
| linear_step | 0.932 | [0.874, 0.973] | 0.866 | 6.0 | 0.022 | not run |  |  |  | 129 |
| attn_monitor | 0.922 (0.898 to 0.945) | [0.819, 0.960] | 0.925 | 0.0 | 0.059 | not run |  |  |  | 1762726 |
| mlp_traj | 0.900 (0.891 to 0.909) | [0.827, 0.963] |  |  | 0.100 | not run |  |  |  | 5633 |
| linear | 0.894 | [0.784, 0.974] |  |  |  | not run |  |  |  |  |
| baseline: TF-IDF rendered text | 0.895 |  |  |  |  |  |  |  |  |  |
| baseline: metadata | 0.929 |  |  |  |  |  |  |  |  |  |

### coordination posted: alone: solo_design

B.I counts: 177 agents, 38 positive, 59 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_monitor | 0.823 (0.798 to 0.849) | [0.763, 0.919] | 0.847 | 0.0 | 0.104 | 0.531 (0.497 to 0.566) | 0.569 | 0.388 | 2 | 1762726 |
| gru | 0.811 (0.808 to 0.814) | [0.713, 0.901] | 0.815 | -1.5 | 0.085 | 0.618 (0.599 to 0.637) | 0.602 | 0.302 | 2 | 1426561 |
| mlp_max | 0.777 (0.713 to 0.841) | [0.683, 0.960] | 0.814 | -0.5 | 0.111 | 0.709 (0.683 to 0.736) | 0.698 | 0.263 | 2 | 1327489 |
| linear | 0.747 | [0.579, 0.878] |  |  |  | 0.715 |  |  | 2 |  |
| mlp_traj | 0.741 (0.698 to 0.783) | [0.631, 0.892] |  |  | 0.171 | 0.568 (0.503 to 0.633) |  | 0.326 | 2 | 7553 |
| attn_pool | 0.707 (0.706 to 0.709) | [0.569, 0.834] | 0.761 | -0.5 | 0.094 | 0.647 (0.626 to 0.667) | 0.737 | 0.250 | 2 | 1360641 |
| mlp_mean | 0.682 (0.670 to 0.694) | [0.564, 0.807] | 0.801 | 2.2 | 0.141 | 0.599 (0.558 to 0.640) | 0.675 | 0.208 | 2 | 1327489 |
| linear_step | 0.665 | [0.500, 0.816] | 0.812 | 2.5 | 0.126 | 0.435 | 0.677 | 0.408 | 2 | 129 |
| baseline: TF-IDF rendered text | 0.774 |  |  |  |  | 0.516 |  |  |  |  |
| baseline: metadata | 0.753 |  |  |  |  | 0.540 |  |  |  |  |

### coordination posted: alone: spar_team

B.I counts: 450 agents, 406 positive, 135 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.991 (0.990 to 0.992) | [0.985, 0.997] | 0.993 | 1.0 | 0.006 | 0.968 (0.966 to 0.969) | 0.984 | 0.032 | 3 | 1327489 |
| gru | 0.989 (0.989 to 0.990) | [0.979, 0.996] | 0.990 | 0.5 | 0.007 | 0.958 (0.954 to 0.962) | 0.968 | 0.042 | 3 | 1426561 |
| attn_pool | 0.988 (0.988 to 0.989) | [0.979, 0.997] | 0.992 | 1.0 | 0.009 | 0.962 (0.951 to 0.973) | 0.983 | 0.038 | 3 | 1360641 |
| attn_monitor | 0.973 (0.960 to 0.986) | [0.977, 0.994] | 0.986 | 1.0 | 0.012 | 0.913 (0.908 to 0.917) | 0.960 | 0.084 | 3 | 1762726 |
| mlp_mean | 0.968 (0.958 to 0.977) | [0.957, 0.992] | 0.973 | 1.0 | 0.015 | 0.817 (0.817 to 0.818) | 0.935 | 0.161 | 3 | 1327489 |
| linear | 0.909 | [0.843, 0.963] |  |  |  | 0.690 |  |  | 3 |  |
| linear_step | 0.907 | [0.870, 0.939] | 0.953 | 2.0 | 0.031 | 0.742 | 0.927 | 0.205 | 3 | 129 |
| mlp_traj | 0.889 (0.880 to 0.898) | [0.837, 0.953] |  |  | 0.111 | 0.692 (0.683 to 0.702) |  | 0.308 | 3 | 18561 |
| baseline: TF-IDF rendered text | 0.889 |  |  |  |  | 0.680 |  |  |  |  |
| baseline: metadata | 0.794 |  |  |  |  | 0.625 |  |  |  |  |

### coordination posted: alone: spar_mixed

B.I counts: 209 agents, 205 positive, 51 runs, 6 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.993 (0.993 to 0.993) | [0.976, 1.000] | 0.983 | 1.0 | 0.007 | not run |  |  |  | 1327489 |
| attn_pool | 0.992 (0.985 to 0.999) | [0.994, 1.000] | 0.975 | 1.0 | 0.007 | not run |  |  |  | 1360641 |
| mlp_mean | 0.986 (0.980 to 0.991) | [0.947, 1.000] | 0.967 | 1.0 | 0.014 | not run |  |  |  | 1327489 |
| gru | 0.960 (0.948 to 0.973) | [0.918, 1.000] | 0.977 | 1.0 | 0.033 | not run |  |  |  | 1426561 |
| linear_step | 0.954 | [0.900, 0.995] | 0.958 | 4.0 | 0.038 | not run |  |  |  | 129 |
| attn_monitor | 0.940 (0.907 to 0.973) | [0.937, 1.000] | 0.963 | 1.0 | 0.056 | not run |  |  |  | 1762726 |
| mlp_traj | 0.909 (0.879 to 0.939) | [0.864, 0.995] |  |  | 0.089 | not run |  |  |  | 9089 |
| linear | 0.600 | [0.464, 0.726] |  |  |  | not run |  |  |  |  |
| baseline: TF-IDF rendered text | 0.726 |  |  |  |  |  |  |  |  |  |
| baseline: metadata | 0.627 |  |  |  |  |  |  |  |  |  |

### collusion crossed: alone: board_seed

B.I counts: 134 agents, 48 positive, 43 runs, 1 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.901 | [0.814, 0.964] |  |  |  | not run |  |  |  |  |
| mlp_max | 0.888 (0.888 to 0.889) | [0.821, 0.942] | 0.896 | 5.5 | 0.102 | not run |  |  |  | 1327489 |
| attn_pool | 0.880 (0.842 to 0.917) | [0.849, 0.967] | 0.882 | 6.2 | 0.075 | not run |  |  |  | 1360641 |
| mlp_traj | 0.853 (0.835 to 0.870) | [0.739, 0.914] |  |  | 0.145 | not run |  |  |  | 5505 |
| mlp_mean | 0.814 (0.813 to 0.814) | [0.721, 0.894] | 0.855 | 14.8 | 0.160 | not run |  |  |  | 1327489 |
| attn_monitor | 0.811 (0.806 to 0.816) | [0.709, 0.895] | 0.844 | 1.0 | 0.112 | not run |  |  |  | 1762726 |
| gru | 0.811 (0.777 to 0.845) | [0.757, 0.917] | 0.867 | 3.5 | 0.132 | not run |  |  |  | 1426561 |
| linear_step | 0.785 | [0.652, 0.906] | 0.807 | 16.0 | 0.151 | not run |  |  |  | 129 |
| baseline: TF-IDF rendered text | 0.950 |  |  |  |  |  |  |  |  |  |
| baseline: metadata | 0.852 |  |  |  |  |  |  |  |  |  |

### collusion crossed: alone: spar_team

B.I counts: 450 agents, 67 positive, 135 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_traj | 0.941 (0.939 to 0.943) | [0.900, 0.976] |  |  | 0.059 | 0.782 (0.727 to 0.837) |  | 0.218 | 3 | 18945 |
| mlp_mean | 0.931 (0.915 to 0.947) | [0.853, 0.963] | 0.923 | 1.8 | 0.058 | 0.852 (0.801 to 0.903) | 0.865 | 0.132 | 3 | 1327489 |
| attn_pool | 0.929 (0.923 to 0.936) | [0.894, 0.970] | 0.913 | 0.5 | 0.046 | 0.663 (0.662 to 0.663) | 0.737 | 0.326 | 3 | 1360641 |
| attn_monitor | 0.926 (0.921 to 0.931) | [0.862, 0.969] | 0.931 | -0.2 | 0.061 | 0.585 (0.583 to 0.586) | 0.763 | 0.406 | 3 | 1762726 |
| mlp_max | 0.922 (0.914 to 0.931) | [0.875, 0.976] | 0.919 | 2.8 | 0.069 | 0.839 (0.813 to 0.865) | 0.827 | 0.159 | 3 | 1327489 |
| gru | 0.918 (0.905 to 0.931) | [0.835, 0.962] | 0.928 | 0.0 | 0.069 | 0.657 (0.637 to 0.677) | 0.780 | 0.330 | 3 | 1426561 |
| linear | 0.910 | [0.857, 0.959] |  |  |  | 0.827 |  |  | 3 |  |
| linear_step | 0.909 | [0.836, 0.969] | 0.827 | 5.0 | 0.079 | 0.797 | 0.828 | 0.190 | 3 | 129 |
| baseline: TF-IDF rendered text | 0.980 |  |  |  |  | 0.847 |  |  |  |  |
| baseline: metadata | 0.920 |  |  |  |  | 0.845 |  |  |  |  |

### collusion crossed: alone: spar_mixed

B.I counts: 209 agents, 7 positive, 51 runs, 6 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.967 | [0.920, 1.000] |  |  |  | not run |  |  |  |  |
| linear_step | 0.915 | [0.794, 1.000] | 0.801 | -23.0 | 0.080 | not run |  |  |  | 129 |
| mlp_mean | 0.885 (0.864 to 0.907) | [0.830, 0.976] | 0.866 | 3.0 | 0.087 | not run |  |  |  | 1327489 |
| attn_monitor | 0.868 (0.861 to 0.875) | [0.744, 0.971] | 0.789 | -1.8 | 0.104 | not run |  |  |  | 1762726 |
| gru | 0.855 (0.795 to 0.916) | [0.670, 0.924] | 0.786 | -13.0 | 0.102 | not run |  |  |  | 1426561 |
| mlp_traj | 0.810 (0.793 to 0.826) | [0.545, 1.000] |  |  | 0.185 | not run |  |  |  | 8961 |
| attn_pool | 0.801 (0.648 to 0.955) | [0.343, 0.962] | 0.781 | -77.0 | 0.115 | not run |  |  |  | 1360641 |
| mlp_max | 0.711 (0.470 to 0.953) | [0.082, 0.952] | 0.692 | 6.0 | 0.221 | not run |  |  |  | 1327489 |
| baseline: TF-IDF rendered text | 0.950 |  |  |  |  |  |  |  |  |  |
| baseline: metadata | 0.960 |  |  |  |  |  |  |  |  |  |

### cheating: leave out: controls

B.I counts: 600 agents, 343 positive, 600 runs, 5 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.817 (0.808 to 0.825) | [0.774, 0.841] | 0.861 | 1.0 | 0.160 | not run |  |  |  | 1327489 |
| attn_monitor | 0.793 (0.762 to 0.824) | [0.789, 0.858] | 0.864 | 1.0 | 0.169 | not run |  |  |  | 1762726 |
| gru | 0.786 (0.783 to 0.789) | [0.746, 0.819] | 0.857 | 1.0 | 0.176 | not run |  |  |  | 1426561 |
| attn_pool | 0.768 (0.764 to 0.772) | [0.725, 0.800] | 0.845 | 1.0 | 0.186 | not run |  |  |  | 1360641 |
| mlp_mean | 0.757 (0.744 to 0.771) | [0.701, 0.782] | 0.836 | 1.0 | 0.186 | not run |  |  |  | 1327489 |
| mlp_traj | 0.747 (0.745 to 0.749) | [0.708, 0.780] |  |  | 0.234 | not run |  |  |  | 33025 |
| linear | 0.684 | [0.637, 0.727] |  |  |  | 0.682 |  |  | 5 |  |
| linear_step | 0.543 | [0.494, 0.590] | 0.705 | 10.0 | 0.367 | not run |  |  |  | 129 |
| baseline: TF-IDF rendered text | 0.802 |  |  |  |  | 0.748 |  |  |  |  |
| baseline: metadata | 0.541 |  |  |  |  | 0.512 |  |  |  |  |

### cheating: leave out: board_seed

B.I counts: 134 agents, 56 positive, 43 runs, 1 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_mean | 0.867 (0.854 to 0.879) | [0.808, 0.936] | 0.837 | 16.0 | 0.100 | not run |  |  |  | 1327489 |
| linear_step | 0.779 | [0.678, 0.860] | 0.820 | 20.0 | 0.114 | not run |  |  |  | 129 |
| mlp_max | 0.776 (0.731 to 0.821) | [0.735, 0.892] | 0.816 | 13.8 | 0.198 | not run |  |  |  | 1327489 |
| attn_monitor | 0.748 (0.714 to 0.783) | [0.686, 0.863] | 0.816 | 8.8 | 0.220 | not run |  |  |  | 1762726 |
| gru | 0.714 (0.706 to 0.722) | [0.606, 0.829] | 0.750 | 8.0 | 0.245 | not run |  |  |  | 1426561 |
| mlp_traj | 0.676 (0.671 to 0.681) | [0.575, 0.785] |  |  | 0.309 | not run |  |  |  | 33025 |
| attn_pool | 0.602 (0.557 to 0.647) | [0.457, 0.657] | 0.661 | 8.2 | 0.361 | not run |  |  |  | 1360641 |
| linear | 0.465 | [0.361, 0.570] |  |  |  | 0.749 |  |  | 1 |  |
| baseline: TF-IDF rendered text | 0.856 |  |  |  |  | 0.808 |  |  |  |  |
| baseline: metadata | 0.846 |  |  |  |  | 0.838 |  |  |  |  |

### cheating: leave out: solo_design

B.I counts: 177 agents, 84 positive, 59 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_pool | 0.957 (0.952 to 0.962) | [0.930, 0.984] | 0.944 | 1.0 | 0.002 | not run |  |  |  | 1360641 |
| mlp_max | 0.937 (0.922 to 0.952) | [0.918, 0.979] | 0.938 | 1.0 | 0.046 | not run |  |  |  | 1327489 |
| gru | 0.936 (0.935 to 0.936) | [0.885, 0.970] | 0.934 | 1.0 | 0.026 | not run |  |  |  | 1426561 |
| attn_monitor | 0.935 (0.931 to 0.940) | [0.882, 0.966] | 0.940 | 1.0 | 0.035 | not run |  |  |  | 1762726 |
| mlp_mean | 0.923 (0.919 to 0.928) | [0.879, 0.964] | 0.911 | 1.0 | 0.039 | not run |  |  |  | 1327489 |
| mlp_traj | 0.877 (0.855 to 0.898) | [0.796, 0.906] |  |  | 0.112 | not run |  |  |  | 33025 |
| linear | 0.821 | [0.747, 0.885] |  |  |  | 0.693 |  |  | 2 |  |
| linear_step | 0.785 | [0.704, 0.863] | 0.798 | 10.0 | 0.104 | not run |  |  |  | 129 |
| baseline: TF-IDF rendered text | 0.899 |  |  |  |  | 0.858 |  |  |  |  |
| baseline: metadata | 0.662 |  |  |  |  | 0.556 |  |  |  |  |

### cheating: leave out: spar_team

B.I counts: 450 agents, 109 positive, 135 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_monitor | 0.873 (0.873 to 0.873) | [0.823, 0.921] | 0.891 | 2.5 | 0.099 | not run |  |  |  | 1762726 |
| attn_pool | 0.858 (0.848 to 0.867) | [0.793, 0.903] | 0.878 | 1.0 | 0.103 | not run |  |  |  | 1360641 |
| gru | 0.857 (0.817 to 0.896) | [0.852, 0.937] | 0.862 | 4.5 | 0.085 | not run |  |  |  | 1426561 |
| mlp_max | 0.845 (0.820 to 0.871) | [0.817, 0.924] | 0.879 | 9.2 | 0.115 | not run |  |  |  | 1327489 |
| mlp_mean | 0.825 (0.814 to 0.836) | [0.776, 0.893] | 0.864 | 7.0 | 0.129 | not run |  |  |  | 1327489 |
| mlp_traj | 0.821 (0.819 to 0.823) | [0.759, 0.883] |  |  | 0.156 | not run |  |  |  | 33025 |
| linear | 0.792 | [0.727, 0.855] |  |  |  | 0.735 |  |  | 3 |  |
| linear_step | 0.754 | [0.679, 0.832] | 0.791 | 23.0 | 0.133 | not run |  |  |  | 129 |
| baseline: TF-IDF rendered text | 0.873 |  |  |  |  | 0.862 |  |  |  |  |
| baseline: metadata | 0.831 |  |  |  |  | 0.788 |  |  |  |  |

### cheating: leave out: spar_mixed

B.I counts: 209 agents, 12 positive, 51 runs, 6 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.944 (0.937 to 0.950) | [0.880, 0.978] | 0.946 | 18.0 | 0.032 | not run |  |  |  | 1327489 |
| mlp_traj | 0.942 (0.938 to 0.946) | [0.882, 0.973] |  |  | 0.046 | not run |  |  |  | 33025 |
| attn_pool | 0.926 (0.925 to 0.928) | [0.860, 0.971] | 0.931 | 9.0 | 0.043 | not run |  |  |  | 1360641 |
| attn_monitor | 0.926 (0.920 to 0.933) | [0.857, 0.969] | 0.936 | 13.0 | 0.048 | not run |  |  |  | 1762726 |
| gru | 0.918 (0.910 to 0.926) | [0.859, 0.975] | 0.935 | 16.2 | 0.038 | not run |  |  |  | 1426561 |
| linear | 0.915 | [0.855, 0.959] |  |  |  | 0.638 |  |  | 2 |  |
| mlp_mean | 0.914 (0.901 to 0.927) | [0.839, 0.950] | 0.928 | 29.2 | 0.033 | not run |  |  |  | 1327489 |
| linear_step | 0.887 | [0.815, 0.943] | 0.914 | 28.0 | -0.008 | not run |  |  |  | 129 |
| baseline: TF-IDF rendered text | 0.984 |  |  |  |  | 0.839 |  |  |  |  |
| baseline: metadata | 0.783 |  |  |  |  | 0.418 |  |  |  |  |

### cheating: leave out: spar_expl

B.I counts: 313 agents, 71 positive, 96 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.986 (0.985 to 0.988) | [0.976, 0.997] | 0.988 | 0.0 | -0.015 | not run |  |  |  | 1327489 |
| attn_monitor | 0.981 (0.976 to 0.985) | [0.973, 0.994] | 0.982 | 0.0 | -0.013 | not run |  |  |  | 1762726 |
| gru | 0.979 (0.977 to 0.982) | [0.966, 0.995] | 0.981 | 0.5 | -0.024 | not run |  |  |  | 1426561 |
| attn_pool | 0.973 (0.972 to 0.974) | [0.951, 0.988] | 0.971 | 1.0 | -0.002 | not run |  |  |  | 1360641 |
| mlp_traj | 0.936 (0.927 to 0.945) | [0.904, 0.976] |  |  | 0.050 | not run |  |  |  | 33025 |
| mlp_mean | 0.917 (0.898 to 0.935) | [0.894, 0.967] | 0.958 | 0.0 | 0.036 | not run |  |  |  | 1327489 |
| linear | 0.906 | [0.855, 0.951] |  |  |  | 0.646 |  |  | 3 |  |
| linear_step | 0.751 | [0.660, 0.831] | 0.865 | -2.0 | 0.136 | not run |  |  |  | 129 |
| baseline: TF-IDF rendered text | 0.987 |  |  |  |  | 0.956 |  |  |  |  |
| baseline: metadata | 0.855 |  |  |  |  | 0.588 |  |  |  |  |

### cheating: leave out: spar_solo

B.I counts: 90 agents, 47 positive, 90 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.940 (0.936 to 0.943) | [0.885, 0.976] | 0.866 | 1.0 | 0.034 | not run |  |  |  | 1327489 |
| gru | 0.937 (0.932 to 0.943) | [0.881, 0.972] | 0.854 | 1.5 | 0.026 | not run |  |  |  | 1426561 |
| attn_monitor | 0.926 (0.911 to 0.940) | [0.848, 0.963] | 0.858 | 1.5 | 0.044 | not run |  |  |  | 1762726 |
| attn_pool | 0.921 (0.916 to 0.925) | [0.863, 0.979] | 0.878 | 1.0 | 0.045 | not run |  |  |  | 1360641 |
| linear | 0.852 | [0.769, 0.918] |  |  |  | 0.930 |  |  | 2 |  |
| mlp_traj | 0.833 (0.832 to 0.833) | [0.742, 0.914] |  |  | 0.152 | not run |  |  |  | 33025 |
| mlp_mean | 0.778 (0.726 to 0.829) | [0.614, 0.823] | 0.738 | 0.5 | 0.172 | not run |  |  |  | 1327489 |
| linear_step | 0.749 | [0.643, 0.839] | 0.793 | 3.0 | 0.142 | not run |  |  |  | 129 |
| baseline: TF-IDF rendered text | 0.943 |  |  |  |  | 0.986 |  |  |  |  |
| baseline: metadata | 0.793 |  |  |  |  | 0.656 |  |  |  |  |

### cheating: leave out: goodfire

B.I counts: 2833 agents, 2083 positive, 2833 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear_step | 0.770 | [0.749, 0.790] | 0.771 | 27.0 | 0.120 | not run |  |  |  | 129 |
| attn_monitor | 0.672 (0.646 to 0.698) | [0.678, 0.719] | 0.649 | 12.0 | 0.300 | not run |  |  |  | 1762726 |
| attn_pool | 0.670 (0.661 to 0.680) | [0.658, 0.702] | 0.714 | 13.0 | 0.260 | not run |  |  |  | 1360641 |
| mlp_mean | 0.664 (0.664 to 0.664) | [0.641, 0.685] | 0.708 | 18.5 | 0.279 | not run |  |  |  | 1327489 |
| mlp_traj | 0.657 (0.652 to 0.662) | [0.638, 0.685] |  |  | 0.339 | not run |  |  |  | 33025 |
| gru | 0.656 (0.618 to 0.695) | [0.673, 0.717] | 0.694 | 16.8 | 0.298 | not run |  |  |  | 1426561 |
| linear | 0.648 | [0.624, 0.670] |  |  |  | 0.600 |  |  | 3 |  |
| mlp_max | 0.630 (0.593 to 0.667) | [0.569, 0.618] | 0.606 | 24.0 | 0.347 | not run |  |  |  | 1327489 |
| baseline: TF-IDF rendered text | 0.678 |  |  |  |  | 0.732 |  |  |  |  |
| baseline: metadata | 0.666 |  |  |  |  | 0.696 |  |  |  |  |

### cheating: leave out: cheatbench

B.I counts: 224 agents, 134 positive, 224 runs, 4 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.805 | [0.751, 0.858] |  |  |  | 0.664 |  |  | 4 |  |
| mlp_mean | 0.717 (0.681 to 0.753) | [0.605, 0.752] | 0.707 | -0.5 | 0.234 | not run |  |  |  | 1327489 |
| attn_pool | 0.640 (0.612 to 0.669) | [0.582, 0.754] | 0.769 | 1.0 | 0.325 | not run |  |  |  | 1360641 |
| linear_step | 0.587 | [0.506, 0.664] | 0.765 | 0.0 | 0.301 | not run |  |  |  | 129 |
| mlp_traj | 0.578 (0.569 to 0.587) | [0.487, 0.643] |  |  | 0.405 | not run |  |  |  | 33025 |
| mlp_max | 0.533 (0.515 to 0.551) | [0.463, 0.640] | 0.725 | 2.0 | 0.442 | not run |  |  |  | 1327489 |
| gru | 0.505 (0.451 to 0.559) | [0.472, 0.642] | 0.738 | 0.0 | 0.465 | not run |  |  |  | 1426561 |
| attn_monitor | 0.452 (0.424 to 0.481) | [0.392, 0.566] | 0.735 | 0.0 | 0.516 | not run |  |  |  | 1762726 |
| baseline: TF-IDF rendered text | 0.582 |  |  |  |  | 0.729 |  |  |  |  |
| baseline: metadata | 0.642 |  |  |  |  | 0.847 |  |  |  |  |

### cheating: leave out: evilgenie

B.I counts: 266 agents, 133 positive, 133 runs, 10 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| linear_step | 0.672 | [0.626, 0.717] |  |  | 0.211 | not run |  |  |  | 129 |
| linear | 0.652 | [0.607, 0.700] |  |  |  | 0.704 |  |  | 10 |  |
| mlp_traj | 0.633 (0.607 to 0.658) | [0.562, 0.654] |  |  | 0.352 | not run |  |  |  | 33025 |
| attn_pool | 0.618 (0.609 to 0.627) | [0.561, 0.658] |  |  | 0.358 | not run |  |  |  | 1360641 |
| mlp_mean | 0.613 (0.603 to 0.623) | [0.575, 0.671] |  |  | 0.355 | not run |  |  |  | 1327489 |
| attn_monitor | 0.596 (0.567 to 0.625) | [0.517, 0.618] |  |  | 0.381 | not run |  |  |  | 1762726 |
| gru | 0.583 (0.581 to 0.586) | [0.532, 0.633] |  |  | 0.385 | not run |  |  |  | 1426561 |
| mlp_max | 0.543 (0.536 to 0.550) | [0.504, 0.601] |  |  | 0.446 | not run |  |  |  | 1327489 |
| baseline: TF-IDF rendered text | 0.554 |  |  |  |  | 0.651 |  |  |  |  |
| baseline: metadata | 0.531 |  |  |  |  | 0.585 |  |  |  |  |

### coordination posted: leave out: board_seed

B.I counts: 134 agents, 87 positive, 43 runs, 1 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_monitor | 0.980 (0.976 to 0.984) | [0.947, 0.995] | 0.990 | 0.0 | 0.019 | not run |  |  |  | 1762726 |
| gru | 0.978 (0.978 to 0.978) | [0.950, 0.996] | 0.987 | 0.0 | 0.021 | not run |  |  |  | 1426561 |
| mlp_max | 0.978 (0.973 to 0.982) | [0.944, 0.994] | 0.992 | 0.0 | 0.021 | not run |  |  |  | 1327489 |
| attn_pool | 0.970 (0.966 to 0.974) | [0.943, 0.995] | 0.993 | 0.0 | 0.029 | not run |  |  |  | 1360641 |
| mlp_mean | 0.960 (0.958 to 0.963) | [0.933, 0.986] | 0.975 | 1.0 | 0.036 | not run |  |  |  | 1327489 |
| linear_step | 0.949 | [0.897, 0.983] | 0.951 | 6.0 | 0.017 | not run |  |  |  | 129 |
| linear | 0.942 | [0.885, 0.980] |  |  |  | 0.818 |  |  | 1 |  |
| mlp_traj | 0.907 (0.904 to 0.910) | [0.845, 0.961] |  |  | 0.093 | not run |  |  |  | 33025 |
| baseline: TF-IDF rendered text | 0.889 |  |  |  |  | 0.713 |  |  |  |  |
| baseline: metadata | 0.929 |  |  |  |  | 0.904 |  |  |  |  |

### coordination posted: leave out: solo_design

B.I counts: 177 agents, 38 positive, 59 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_pool | 0.974 (0.973 to 0.974) | [0.933, 0.999] | 0.942 | 0.5 | 0.022 | not run |  |  |  | 1360641 |
| mlp_max | 0.972 (0.969 to 0.975) | [0.906, 1.000] | 0.921 | -0.5 | 0.028 | not run |  |  |  | 1327489 |
| mlp_mean | 0.963 (0.960 to 0.965) | [0.919, 0.987] | 0.937 | 1.0 | 0.035 | not run |  |  |  | 1327489 |
| gru | 0.960 (0.958 to 0.961) | [0.901, 0.993] | 0.940 | 0.0 | 0.036 | not run |  |  |  | 1426561 |
| attn_monitor | 0.950 (0.932 to 0.968) | [0.921, 0.997] | 0.934 | -0.5 | 0.046 | not run |  |  |  | 1762726 |
| linear_step | 0.772 | [0.661, 0.861] | 0.895 | 2.0 | 0.195 | not run |  |  |  | 129 |
| linear | 0.758 | [0.647, 0.853] |  |  |  | 0.763 |  |  | 2 |  |
| mlp_traj | 0.703 (0.689 to 0.718) | [0.565, 0.798] |  |  | 0.297 | not run |  |  |  | 33025 |
| baseline: TF-IDF rendered text | 0.496 |  |  |  |  | 0.780 |  |  |  |  |
| baseline: metadata | 0.444 |  |  |  |  | 0.565 |  |  |  |  |

### coordination posted: leave out: spar_team

B.I counts: 450 agents, 406 positive, 135 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_mean | 0.949 (0.948 to 0.951) | [0.920, 0.978] | 0.982 | 1.0 | 0.049 | not run |  |  |  | 1327489 |
| mlp_max | 0.943 (0.931 to 0.955) | [0.928, 0.975] | 0.991 | 1.0 | 0.053 | not run |  |  |  | 1327489 |
| attn_pool | 0.926 (0.921 to 0.931) | [0.896, 0.963] | 0.983 | 1.0 | 0.068 | not run |  |  |  | 1360641 |
| gru | 0.911 (0.907 to 0.914) | [0.877, 0.947] | 0.980 | 0.5 | 0.084 | not run |  |  |  | 1426561 |
| attn_monitor | 0.866 (0.819 to 0.914) | [0.866, 0.952] | 0.985 | 1.0 | 0.128 | not run |  |  |  | 1762726 |
| linear | 0.863 | [0.812, 0.907] |  |  |  | 0.805 |  |  | 3 |  |
| linear_step | 0.824 | [0.767, 0.874] | 0.921 | 2.0 | 0.125 | not run |  |  |  | 129 |
| mlp_traj | 0.808 (0.762 to 0.854) | [0.798, 0.907] |  |  | 0.192 | not run |  |  |  | 26881 |
| baseline: TF-IDF rendered text | 0.654 |  |  |  |  | 0.630 |  |  |  |  |
| baseline: metadata | 0.726 |  |  |  |  | 0.652 |  |  |  |  |

### coordination posted: leave out: spar_mixed

B.I counts: 209 agents, 205 positive, 51 runs, 6 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_max | 0.957 (0.956 to 0.957) | [0.924, 0.984] | 0.982 | 1.0 | 0.040 | not run |  |  |  | 1327489 |
| attn_pool | 0.956 (0.951 to 0.961) | [0.915, 0.981] | 0.983 | 1.0 | 0.040 | not run |  |  |  | 1360641 |
| gru | 0.921 (0.915 to 0.927) | [0.839, 0.986] | 0.971 | 1.0 | 0.073 | not run |  |  |  | 1426561 |
| mlp_traj | 0.890 (0.876 to 0.904) | [0.722, 0.990] |  |  | 0.110 | not run |  |  |  | 33025 |
| linear | 0.854 | [0.763, 0.924] |  |  |  | not run |  |  |  |  |
| mlp_mean | 0.623 (0.593 to 0.652) | [0.344, 0.961] | 0.930 | 11.5 | 0.370 | not run |  |  |  | 1327489 |
| linear_step | 0.371 | [0.067, 0.891] | 0.874 | 28.0 | 0.581 | not run |  |  |  | 129 |
| attn_monitor | 0.299 (0.277 to 0.322) | [0.005, 0.981] | 0.906 | 21.0 | 0.696 | not run |  |  |  | 1762726 |
| baseline: TF-IDF rendered text | 0.470 |  |  |  |  |  |  |  |  |  |
| baseline: metadata | 0.423 |  |  |  |  |  |  |  |  |  |

### collusion crossed: leave out: board_seed

B.I counts: 134 agents, 48 positive, 43 runs, 1 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_monitor | 0.900 (0.887 to 0.913) | [0.852, 0.961] | 0.855 | 8.5 | 0.093 | not run |  |  |  | 1762726 |
| gru | 0.895 (0.868 to 0.923) | [0.874, 0.966] | 0.873 | 7.5 | 0.098 | not run |  |  |  | 1426561 |
| linear | 0.888 | [0.810, 0.945] |  |  |  | 0.750 |  |  | 1 |  |
| mlp_traj | 0.872 (0.866 to 0.878) | [0.799, 0.945] |  |  | 0.128 | not run |  |  |  | 33025 |
| linear_step | 0.866 | [0.783, 0.929] | 0.858 | 18.0 | 0.118 | not run |  |  |  | 129 |
| mlp_max | 0.860 (0.814 to 0.905) | [0.838, 0.956] | 0.839 | 6.0 | 0.139 | not run |  |  |  | 1327489 |
| attn_pool | 0.817 (0.784 to 0.849) | [0.761, 0.915] | 0.803 | 8.0 | 0.158 | not run |  |  |  | 1360641 |
| mlp_mean | 0.789 (0.765 to 0.814) | [0.652, 0.858] | 0.805 | 9.8 | 0.191 | not run |  |  |  | 1327489 |
| baseline: TF-IDF rendered text | 0.955 |  |  |  |  | 0.876 |  |  |  |  |
| baseline: metadata | 0.669 |  |  |  |  | 0.257 |  |  |  |  |

### collusion crossed: leave out: spar_team

B.I counts: 450 agents, 67 positive, 135 runs, 3 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_traj | 0.872 (0.869 to 0.874) | [0.814, 0.925] |  |  | 0.128 | not run |  |  |  | 17409 |
| linear | 0.814 | [0.712, 0.899] |  |  |  | 0.778 |  |  | 2 |  |
| attn_pool | 0.715 (0.705 to 0.726) | [0.573, 0.829] | 0.780 | -2.8 | 0.262 | not run |  |  |  | 1360641 |
| linear_step | 0.705 | [0.598, 0.801] | 0.743 | 15.5 | 0.261 | not run |  |  |  | 129 |
| mlp_mean | 0.688 (0.684 to 0.691) | [0.560, 0.812] | 0.762 | -3.8 | 0.255 | not run |  |  |  | 1327489 |
| attn_monitor | 0.656 (0.623 to 0.688) | [0.479, 0.756] | 0.736 | 3.0 | 0.282 | not run |  |  |  | 1762726 |
| gru | 0.651 (0.649 to 0.654) | [0.492, 0.800] | 0.746 | 0.0 | 0.343 | not run |  |  |  | 1426561 |
| mlp_max | 0.614 (0.589 to 0.639) | [0.418, 0.760] | 0.718 | 6.5 | 0.344 | not run |  |  |  | 1327489 |
| baseline: TF-IDF rendered text | 0.657 |  |  |  |  | 0.390 |  |  |  |  |
| baseline: metadata | 0.651 |  |  |  |  | 0.452 |  |  |  |  |

### collusion crossed: leave out: spar_mixed

B.I counts: 209 agents, 7 positive, 51 runs, 6 authors. Rows ranked by B.I trajectory AUROC.

| architecture | B.I trajectory AUROC (seed range) | B.I 95% CI (seed 0) | B.I turn AUROC | B.I delay (turns) | B.I gap | B.II mean trajectory AUROC (seed range) | B.II turn AUROC | B.II gap | B.II authors | parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_pool | 0.970 (0.961 to 0.978) | [0.961, 0.995] | 0.936 | 15.8 | -0.029 | not run |  |  |  | 1360641 |
| mlp_mean | 0.965 (0.946 to 0.985) | [0.962, 1.000] | 0.929 | 23.8 | -0.024 | not run |  |  |  | 1327489 |
| gru | 0.909 (0.884 to 0.934) | [0.868, 0.968] | 0.917 | 24.0 | 0.077 | not run |  |  |  | 1426561 |
| linear | 0.907 | [0.812, 0.980] |  |  |  | 0.695 |  |  | 1 |  |
| mlp_traj | 0.904 (0.898 to 0.910) | [0.827, 0.957] |  |  | 0.096 | not run |  |  |  | 29697 |
| attn_monitor | 0.858 (0.806 to 0.910) | [0.829, 0.952] | 0.896 | 31.0 | 0.121 | not run |  |  |  | 1762726 |
| linear_step | 0.857 | [0.763, 0.933] | 0.846 | 41.5 | 0.107 | not run |  |  |  | 129 |
| mlp_max | 0.776 (0.760 to 0.792) | [0.723, 0.863] | 0.847 | -47.0 | 0.219 | not run |  |  |  | 1327489 |
| baseline: TF-IDF rendered text | 0.932 |  |  |  |  | 0.631 |  |  |  |  |
| baseline: metadata | 0.647 |  |  |  |  | 0.236 |  |  |  |  |

### Summary across schemes: B.I trajectory AUROC (mean over seeds)

| architecture | cheat: ours | cheat: all | coord: ours | collu: ours | cheat: alone: controls | cheat: alone: board_seed | cheat: alone: solo_design | cheat: alone: spar_team | cheat: alone: spar_mixed | cheat: alone: spar_expl | cheat: alone: spar_solo | cheat: alone: goodfire | cheat: alone: cheatbench | cheat: alone: evilgenie | coord: alone: board_seed | coord: alone: solo_design | coord: alone: spar_team | coord: alone: spar_mixed | collu: alone: board_seed | collu: alone: spar_team | collu: alone: spar_mixed | cheat: leave out: controls | cheat: leave out: board_seed | cheat: leave out: solo_design | cheat: leave out: spar_team | cheat: leave out: spar_mixed | cheat: leave out: spar_expl | cheat: leave out: spar_solo | cheat: leave out: goodfire | cheat: leave out: cheatbench | cheat: leave out: evilgenie | coord: leave out: board_seed | coord: leave out: solo_design | coord: leave out: spar_team | coord: leave out: spar_mixed | collu: leave out: board_seed | collu: leave out: spar_team | collu: leave out: spar_mixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.950 | 0.920 | 0.969 | 0.933 | 0.960 | 0.880 | 0.914 | 0.927 | 0.734 | 0.955 | 0.962 | 0.921 | 0.951 | 0.872 | 0.894 | 0.747 | 0.909 | 0.600 | 0.901 | 0.910 | 0.967 | 0.684 | 0.465 | 0.821 | 0.792 | 0.915 | 0.906 | 0.852 | 0.648 | 0.805 | 0.652 | 0.942 | 0.758 | 0.863 | 0.854 | 0.888 | 0.814 | 0.907 |
| mlp_mean | 0.942 | 0.920 | 0.984 | 0.933 | 0.950 | 0.819 | 0.853 | 0.922 | 0.837 | 0.940 | 0.941 | 0.933 | 0.961 | 0.720 | 0.951 | 0.682 | 0.968 | 0.986 | 0.814 | 0.931 | 0.885 | 0.757 | 0.867 | 0.923 | 0.825 | 0.914 | 0.917 | 0.778 | 0.664 | 0.717 | 0.613 | 0.960 | 0.963 | 0.949 | 0.623 | 0.789 | 0.688 | 0.965 |
| mlp_max | 0.966 | 0.943 | 0.990 | 0.932 | 0.971 | 0.834 | 0.893 | 0.921 | 0.862 | 0.945 | 0.933 | 0.937 | 0.961 | 0.761 | 0.975 | 0.777 | 0.991 | 0.993 | 0.888 | 0.922 | 0.711 | 0.817 | 0.776 | 0.937 | 0.845 | 0.944 | 0.986 | 0.940 | 0.630 | 0.533 | 0.543 | 0.978 | 0.972 | 0.943 | 0.957 | 0.860 | 0.614 | 0.776 |
| attn_pool | 0.962 | 0.937 | 0.989 | 0.917 | 0.980 | 0.833 | 0.871 | 0.933 | 0.831 | 0.950 | 0.929 | 0.936 | 0.953 | 0.732 | 0.965 | 0.707 | 0.988 | 0.992 | 0.880 | 0.929 | 0.801 | 0.768 | 0.602 | 0.957 | 0.858 | 0.926 | 0.973 | 0.921 | 0.670 | 0.640 | 0.618 | 0.970 | 0.974 | 0.926 | 0.956 | 0.817 | 0.715 | 0.970 |
| attn_monitor | 0.969 | 0.941 | 0.983 | 0.932 | 0.971 | 0.778 | 0.874 | 0.919 | 0.764 | 0.936 | 0.895 | 0.930 | 0.941 | 0.699 | 0.922 | 0.823 | 0.973 | 0.940 | 0.811 | 0.926 | 0.868 | 0.793 | 0.748 | 0.935 | 0.873 | 0.926 | 0.981 | 0.926 | 0.672 | 0.452 | 0.596 | 0.980 | 0.950 | 0.866 | 0.299 | 0.900 | 0.656 | 0.858 |
| gru | 0.963 | 0.938 | 0.987 | 0.930 | 0.969 | 0.825 | 0.873 | 0.886 | 0.796 | 0.940 | 0.889 | 0.933 | 0.948 | 0.715 | 0.945 | 0.811 | 0.989 | 0.960 | 0.811 | 0.918 | 0.855 | 0.786 | 0.714 | 0.936 | 0.857 | 0.918 | 0.979 | 0.937 | 0.656 | 0.505 | 0.583 | 0.978 | 0.960 | 0.911 | 0.921 | 0.895 | 0.651 | 0.909 |
| linear_step | 0.869 | 0.829 | 0.958 | 0.930 | 0.874 | 0.844 | 0.808 | 0.935 | 0.758 | 0.933 | 0.959 | 0.903 | 0.792 |  | 0.932 | 0.665 | 0.907 | 0.954 | 0.785 | 0.909 | 0.915 | 0.543 | 0.779 | 0.785 | 0.754 | 0.887 | 0.751 | 0.749 | 0.770 | 0.587 | 0.672 | 0.949 | 0.772 | 0.824 | 0.371 | 0.866 | 0.705 | 0.857 |
| mlp_traj | 0.957 | 0.937 | 0.954 | 0.945 | 0.953 | 0.858 | 0.909 | 0.906 | 0.804 | 0.952 | 0.966 | 0.916 | 0.964 | 0.848 | 0.900 | 0.741 | 0.889 | 0.909 | 0.853 | 0.941 | 0.810 | 0.747 | 0.676 | 0.877 | 0.821 | 0.942 | 0.936 | 0.833 | 0.657 | 0.578 | 0.633 | 0.907 | 0.703 | 0.808 | 0.890 | 0.872 | 0.872 | 0.904 |
| stack | 0.984 | 0.967 | 0.988 | 0.984 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Summary across schemes: B.II trajectory AUROC (mean over seeds, mean over held-out authors)

| architecture | cheat: ours | cheat: all | coord: ours | collu: ours | cheat: alone: controls | cheat: alone: solo_design | cheat: alone: spar_team | cheat: alone: spar_mixed | cheat: alone: spar_expl | cheat: alone: spar_solo | cheat: alone: goodfire | cheat: alone: cheatbench | cheat: alone: evilgenie | coord: alone: solo_design | coord: alone: spar_team | collu: alone: spar_team | cheat: leave out: controls | cheat: leave out: board_seed | cheat: leave out: solo_design | cheat: leave out: spar_team | cheat: leave out: spar_mixed | cheat: leave out: spar_expl | cheat: leave out: spar_solo | cheat: leave out: goodfire | cheat: leave out: cheatbench | cheat: leave out: evilgenie | coord: leave out: board_seed | coord: leave out: solo_design | coord: leave out: spar_team | collu: leave out: board_seed | collu: leave out: spar_team | collu: leave out: spar_mixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.860 | 0.795 | 0.904 | 0.843 | 0.913 | 0.630 | 0.751 | 0.708 | 0.658 | 0.727 | 0.822 | 0.775 | 0.885 | 0.715 | 0.690 | 0.827 | 0.682 | 0.749 | 0.693 | 0.735 | 0.638 | 0.646 | 0.930 | 0.600 | 0.664 | 0.704 | 0.818 | 0.763 | 0.805 | 0.750 | 0.778 | 0.695 |
| mlp_mean | 0.849 | 0.801 | 0.960 | 0.711 | 0.913 | 0.541 | 0.765 | 0.581 | 0.692 | 0.630 | 0.894 | 0.667 | 0.751 | 0.599 | 0.817 | 0.852 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mlp_max | 0.906 | 0.766 | 0.977 | 0.748 | 0.956 | 0.607 | 0.841 | 0.680 | 0.606 | 0.642 | 0.909 | 0.823 | 0.797 | 0.709 | 0.968 | 0.839 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| attn_pool | 0.881 | 0.814 | 0.975 | 0.626 | 0.949 | 0.490 | 0.784 | 0.559 | 0.698 | 0.627 | 0.910 | 0.684 | 0.746 | 0.647 | 0.962 | 0.663 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| attn_monitor | 0.875 | 0.782 | 0.953 | 0.463 | 0.916 | 0.510 | 0.728 | 0.521 | 0.491 | 0.482 | 0.913 | 0.745 | 0.722 | 0.531 | 0.913 | 0.585 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| gru | 0.872 | 0.772 | 0.938 | 0.527 | 0.919 | 0.519 | 0.651 | 0.428 | 0.477 | 0.490 | 0.909 | 0.664 | 0.727 | 0.618 | 0.958 | 0.657 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| linear_step | 0.787 | 0.742 | 0.885 | 0.748 | 0.831 | 0.443 | 0.841 | 0.414 | 0.726 | 0.835 | 0.890 | 0.659 |  | 0.435 | 0.742 | 0.797 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| mlp_traj | 0.850 | 0.806 | 0.836 | 0.668 | 0.883 | 0.544 | 0.760 | 0.721 | 0.661 | 0.755 | 0.829 | 0.789 | 0.848 | 0.568 | 0.692 | 0.782 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Architecture cells with no result

Cells cut by the time budget: 0. Every other planned cell ran or is listed here with the reason it could not run.

- cheating, alone: board_seed, B.II, 2 MLP, running mean: no feasible split (one author, or too few in a class)
- coordination posted, alone: board_seed, B.II, 2 MLP, running mean: no feasible split (one author, or too few in a class)
- coordination posted, alone: spar_mixed, B.II, 2 MLP, running mean: no feasible split (one author, or too few in a class)
- collusion crossed, alone: board_seed, B.II, 2 MLP, running mean: no feasible split (one author, or too few in a class)
- collusion crossed, alone: spar_mixed, B.II, 2 MLP, running mean: no feasible split (one author, or too few in a class)
- cheating, alone: board_seed, B.II, 3 MLP, running max: no feasible split (one author, or too few in a class)
- coordination posted, alone: board_seed, B.II, 3 MLP, running max: no feasible split (one author, or too few in a class)
- coordination posted, alone: spar_mixed, B.II, 3 MLP, running max: no feasible split (one author, or too few in a class)
- collusion crossed, alone: board_seed, B.II, 3 MLP, running max: no feasible split (one author, or too few in a class)
- collusion crossed, alone: spar_mixed, B.II, 3 MLP, running max: no feasible split (one author, or too few in a class)
- cheating, alone: board_seed, B.II, 4 attention pooling: no feasible split (one author, or too few in a class)
- coordination posted, alone: board_seed, B.II, 4 attention pooling: no feasible split (one author, or too few in a class)
- coordination posted, alone: spar_mixed, B.II, 4 attention pooling: no feasible split (one author, or too few in a class)
- collusion crossed, alone: board_seed, B.II, 4 attention pooling: no feasible split (one author, or too few in a class)
- collusion crossed, alone: spar_mixed, B.II, 4 attention pooling: no feasible split (one author, or too few in a class)
- cheating, alone: board_seed, B.II, 5 attention monitor: no feasible split (one author, or too few in a class)
- coordination posted, alone: board_seed, B.II, 5 attention monitor: no feasible split (one author, or too few in a class)
- coordination posted, alone: spar_mixed, B.II, 5 attention monitor: no feasible split (one author, or too few in a class)
- collusion crossed, alone: board_seed, B.II, 5 attention monitor: no feasible split (one author, or too few in a class)
- collusion crossed, alone: spar_mixed, B.II, 5 attention monitor: no feasible split (one author, or too few in a class)
- cheating, alone: board_seed, B.II, 6 GRU: no feasible split (one author, or too few in a class)
- coordination posted, alone: board_seed, B.II, 6 GRU: no feasible split (one author, or too few in a class)
- coordination posted, alone: spar_mixed, B.II, 6 GRU: no feasible split (one author, or too few in a class)
- collusion crossed, alone: board_seed, B.II, 6 GRU: no feasible split (one author, or too few in a class)
- collusion crossed, alone: spar_mixed, B.II, 6 GRU: no feasible split (one author, or too few in a class)
- cheating, alone: board_seed, B.II, 7 linear per step: no feasible split (one author, or too few in a class)
- cheating, alone: evilgenie, B.II, held-out Claude Fable 5, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out Claude Opus 4.8, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out GPT-5.3 Codex, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out GPT-5.6 Luna, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out GPT-5.6 Sol, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out Gemini 3.1 Pro, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out Gemini 3.5 Flash, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out Grok 4.5, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out Kimi K2.7 Code, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- cheating, alone: evilgenie, B.II, held-out Kimi K3, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)
- coordination posted, alone: board_seed, B.II, 7 linear per step: no feasible split (one author, or too few in a class)
- coordination posted, alone: spar_mixed, B.II, 7 linear per step: no feasible split (one author, or too few in a class)
- collusion crossed, alone: board_seed, B.II, 7 linear per step: no feasible split (one author, or too few in a class)
- collusion crossed, alone: spar_mixed, B.II, 7 linear per step: no feasible split (one author, or too few in a class)
- cheating, alone: board_seed, B.II, 8 MLP (pooled): no feasible split (one author, or too few in a class)
- coordination posted, alone: board_seed, B.II, 8 MLP (pooled): no feasible split (one author, or too few in a class)
- coordination posted, alone: spar_mixed, B.II, 8 MLP (pooled): no feasible split (one author, or too few in a class)
- collusion crossed, alone: board_seed, B.II, 8 MLP (pooled): no feasible split (one author, or too few in a class)
- collusion crossed, alone: spar_mixed, B.II, 8 MLP (pooled): no feasible split (one author, or too few in a class)
- cheating, alone: evilgenie, B.I, 7 linear per step: no feasible split: the training turns carry one class (positives without a per-turn onset)

Not planned for the architectures: the A probe type, the external and transfer schemes (ours to external, external to ours), B.II of the leave-one-source-out schemes, and the stack outside the B.I cells of ours and all (cheating) and ours (coordination, collusion).

## One table per scheme

Each scheme has two tables. The first has architecture rows and probe-type columns: held-out AUROC of the agent-level (trajectory) score, the mean over held-out authors for A and B.II, the mean over two seeds with the seed range for trained architectures; 'not planned' marks a probe type outside the architecture plan, 'not reached' one the time budget did not reach. The second table gives the linear probe in full. Columns: the main probe with its interval and null; the same probe on layer 38 alone and on the seven tokens (blank where not every agent has turns.safetensors); the TF-IDF and metadata baselines on the same folds or split; the counts of the scored rows (B.II and transfer rows also give the training size).

### cheating

#### cheating: alone: controls

Train controls_600; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.938 [0.918, 0.957] | 0.960 [0.944, 0.973] | 0.913 [0.891, 0.934] |
| 2 MLP, running mean | not planned | 0.950 (0.948 to 0.952) | 0.913 (0.910 to 0.915) |
| 3 MLP, running max | not planned | 0.971 (0.964 to 0.979) | 0.956 (0.950 to 0.961) |
| 4 attention pooling | not planned | 0.980 (0.978 to 0.982) | 0.949 (0.942 to 0.957) |
| 5 attention monitor | not planned | 0.971 (0.971 to 0.971) | 0.916 (0.915 to 0.918) |
| 6 GRU | not planned | 0.969 (0.967 to 0.971) | 0.919 (0.902 to 0.936) |
| 7 linear per step | not planned | 0.874 | 0.831 |
| 8 MLP (pooled) | not planned | 0.953 (0.944 to 0.962) | 0.883 (0.880 to 0.886) |
| linear: permutation null 95th | 0.569 | 0.604 | 0.628 |
| baseline: TF-IDF rendered text | 0.953 | 0.990 | 0.971 |
| baseline: metadata | 0.768 | 0.690 | 0.535 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.960 [0.944, 0.973] | 0.604 | 0.961 | 0.984 | 0.990 | 0.690 | 600 | 343 | 600 | 5 |  |
| A | Claude Haiku 5.5 | 0.958 [0.918, 0.987] | 0.588 | 0.970 | 0.944 | 0.958 | 0.769 | 120 | 60 | 120 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.962 [0.928, 0.988] | 0.652 | 0.959 | 0.983 | 0.964 | 0.798 | 120 | 72 | 120 | 1 |  |
| A | GPT-5 mini | 0.901 [0.836, 0.952] | 0.635 | 0.912 | 0.936 | 0.939 | 0.755 | 120 | 60 | 120 | 1 |  |
| A | GPT-6 Luna | 0.901 [0.841, 0.954] | 0.589 | 0.813 | 0.939 | 0.940 | 0.608 | 120 | 69 | 120 | 1 |  |
| A | Qwen3.8 27B | 0.969 [0.937, 0.993] | 0.682 | 0.944 | 0.981 | 0.965 | 0.910 | 120 | 82 | 120 | 1 |  |
| A | **mean over authors** | 0.938 [0.918, 0.957] | 0.569 | 0.920 | 0.957 | 0.953 | 0.768 | 600 | 343 | 600 | 5 |  |
| B.II | Claude Haiku 5.5 | 0.895 [0.833, 0.944] | 0.741 | 0.901 | 0.932 | 0.985 | 0.589 | 120 | 60 | 120 | 1 | train 480 |
| B.II | DeepSeek V4.1 Flash | 0.873 [0.803, 0.930] | 0.718 | 0.852 | 0.987 | 0.988 | 0.557 | 120 | 72 | 120 | 1 | train 480 |
| B.II | GPT-5 mini | 0.937 [0.892, 0.972] | 0.624 | 0.894 | 0.963 | 0.983 | 0.429 | 120 | 60 | 120 | 1 | train 480 |
| B.II | GPT-6 Luna | 0.874 [0.809, 0.931] | 0.669 | 0.844 | 0.970 | 0.904 | 0.604 | 120 | 69 | 120 | 1 | train 480 |
| B.II | Qwen3.8 27B | 0.986 [0.969, 0.997] | 0.885 | 0.992 | 1.000 | 0.996 | 0.497 | 120 | 82 | 120 | 1 | train 480 |
| B.II | **mean over authors** | 0.913 [0.891, 0.934] | 0.628 | 0.897 | 0.970 | 0.971 | 0.535 | 600 | 343 | 600 | 5 |  |

#### cheating: alone: board_seed

Train board_seed; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.880 [0.809, 0.935] | 0.880 [0.809, 0.935] | n/a |
| 2 MLP, running mean | not planned | 0.819 (0.808 to 0.831) | no feasible split |
| 3 MLP, running max | not planned | 0.834 (0.831 to 0.837) | no feasible split |
| 4 attention pooling | not planned | 0.833 (0.824 to 0.843) | no feasible split |
| 5 attention monitor | not planned | 0.778 (0.761 to 0.794) | no feasible split |
| 6 GRU | not planned | 0.825 (0.804 to 0.845) | no feasible split |
| 7 linear per step | not planned | 0.844 | no feasible split |
| 8 MLP (pooled) | not planned | 0.858 (0.855 to 0.862) | no feasible split |
| linear: permutation null 95th | 0.672 | 0.676 |  |
| baseline: TF-IDF rendered text | 0.905 | 0.905 |  |
| baseline: metadata | 0.826 | 0.826 |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.880 [0.809, 0.935] | 0.676 | 0.867 | 0.908 | 0.905 | 0.826 | 134 | 56 | 43 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.880 [0.809, 0.935] | 0.672 | 0.867 | 0.908 | 0.905 | 0.826 | 134 | 56 | 43 | 1 |  |
| A | **mean over authors** | 0.880 [0.809, 0.935] | 0.672 | 0.867 | 0.908 | 0.905 | 0.826 | 134 | 56 | 43 | 1 |  |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 134 | 56 | 43 | 1 | one author only: nothing to hold out |

#### cheating: alone: solo_design

Train solo_design; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.816 [0.694, 0.905] | 0.914 [0.860, 0.960] | 0.630 [0.532, 0.723] |
| 2 MLP, running mean | not planned | 0.853 (0.842 to 0.863) | 0.541 (0.508 to 0.574) |
| 3 MLP, running max | not planned | 0.893 (0.875 to 0.910) | 0.607 (0.582 to 0.631) |
| 4 attention pooling | not planned | 0.871 (0.868 to 0.873) | 0.490 (0.487 to 0.493) |
| 5 attention monitor | not planned | 0.874 (0.867 to 0.882) | 0.510 (0.506 to 0.514) |
| 6 GRU | not planned | 0.873 (0.860 to 0.886) | 0.519 (0.501 to 0.538) |
| 7 linear per step | not planned | 0.808 | 0.443 |
| 8 MLP (pooled) | not planned | 0.909 (0.902 to 0.916) | 0.544 (0.494 to 0.594) |
| linear: permutation null 95th | 0.649 | 0.777 | 0.656 |
| baseline: TF-IDF rendered text | 0.888 | 0.922 | 0.710 |
| baseline: metadata | 0.832 | 0.866 | 0.680 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.914 [0.860, 0.960] | 0.777 | 0.908 | 0.935 | 0.922 | 0.866 | 177 | 84 | 59 | 3 |  |
| A | DeepSeek V4.1 Flash | 0.786 [0.658, 0.888] | 0.626 | 0.794 | 0.884 | 0.918 | 0.818 | 90 | 65 | 30 | 1 |  |
| A | GPT-6 Luna | 0.845 [0.612, 0.983] | 0.703 | 0.848 | 0.832 | 0.857 | 0.846 | 81 | 14 | 27 | 1 |  |
| A | Qwen3.8 Max |  |  |  |  |  |  | 6 | 5 | 2 | 1 | too few agents or runs in a class |
| A | **mean over authors** | 0.816 [0.694, 0.905] | 0.649 | 0.821 | 0.858 | 0.888 | 0.832 | 171 | 79 | 57 | 2 |  |
| B.II | DeepSeek V4.1 Flash | 0.559 [0.422, 0.696] | 0.755 | 0.666 | 0.724 | 0.837 | 0.806 | 90 | 65 | 30 | 1 | train 87 |
| B.II | GPT-6 Luna | 0.701 [0.559, 0.838] | 0.737 | 0.634 | 0.610 | 0.582 | 0.554 | 81 | 14 | 27 | 1 | train 96 |
| B.II | Qwen3.8 Max |  |  |  |  |  |  | 6 | 5 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | **mean over authors** | 0.630 [0.532, 0.723] | 0.656 | 0.650 | 0.667 | 0.710 | 0.680 | 171 | 79 | 57 | 2 |  |

#### cheating: alone: spar_team

Train spar_screen_team; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.926 [0.878, 0.963] | 0.927 [0.885, 0.962] | 0.751 [0.689, 0.811] |
| 2 MLP, running mean | not planned | 0.922 (0.922 to 0.923) | 0.765 (0.755 to 0.775) |
| 3 MLP, running max | not planned | 0.921 (0.920 to 0.923) | 0.841 (0.826 to 0.856) |
| 4 attention pooling | not planned | 0.933 (0.929 to 0.938) | 0.784 (0.744 to 0.823) |
| 5 attention monitor | not planned | 0.919 (0.911 to 0.927) | 0.728 (0.724 to 0.733) |
| 6 GRU | not planned | 0.886 (0.872 to 0.900) | 0.651 (0.647 to 0.654) |
| 7 linear per step | not planned | 0.935 | 0.841 |
| 8 MLP (pooled) | not planned | 0.906 (0.899 to 0.913) | 0.760 (0.738 to 0.782) |
| linear: permutation null 95th | 0.654 | 0.700 | 0.726 |
| baseline: TF-IDF rendered text | 0.938 | 0.982 | 0.918 |
| baseline: metadata | 0.941 | 0.934 | 0.865 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.927 [0.885, 0.962] | 0.700 | 0.937 | 0.960 | 0.982 | 0.934 | 450 | 109 | 135 | 3 |  |
| A | Claude Haiku 5.5 | 0.960 [0.872, 1.000] | 0.815 | 0.975 | 0.941 | 0.978 | 0.986 | 150 | 12 | 45 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.913 [0.848, 0.962] | 0.559 | 0.911 | 0.946 | 0.944 | 0.929 | 150 | 64 | 45 | 1 |  |
| A | GPT-6 Luna | 0.906 [0.813, 0.976] | 0.690 | 0.910 | 0.922 | 0.894 | 0.908 | 150 | 33 | 45 | 1 |  |
| A | **mean over authors** | 0.926 [0.878, 0.963] | 0.654 | 0.932 | 0.936 | 0.938 | 0.941 | 450 | 109 | 135 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.697 [0.599, 0.783] | 0.798 | 0.760 | 0.815 | 0.924 | 0.904 | 150 | 12 | 45 | 1 | train 300 |
| B.II | DeepSeek V4.1 Flash | 0.736 [0.639, 0.815] | 0.770 | 0.669 | 0.846 | 0.877 | 0.832 | 150 | 64 | 45 | 1 | train 300 |
| B.II | GPT-6 Luna | 0.821 [0.670, 0.927] | 0.854 | 0.817 | 0.857 | 0.953 | 0.858 | 150 | 33 | 45 | 1 | train 300 |
| B.II | **mean over authors** | 0.751 [0.689, 0.811] | 0.726 | 0.748 | 0.839 | 0.918 | 0.865 | 450 | 109 | 135 | 3 |  |

#### cheating: alone: spar_mixed

Train spar_mixed_41; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.715 [0.492, 0.882] | 0.734 [0.606, 0.869] | 0.708 [0.493, 0.859] |
| 2 MLP, running mean | not planned | 0.837 (0.822 to 0.852) | 0.581 (0.577 to 0.585) |
| 3 MLP, running max | not planned | 0.862 (0.862 to 0.863) | 0.680 (0.637 to 0.723) |
| 4 attention pooling | not planned | 0.831 (0.778 to 0.885) | 0.559 (0.548 to 0.570) |
| 5 attention monitor | not planned | 0.764 (0.732 to 0.797) | 0.521 (0.503 to 0.539) |
| 6 GRU | not planned | 0.796 (0.773 to 0.819) | 0.428 (0.415 to 0.442) |
| 7 linear per step | not planned | 0.758 | 0.414 |
| 8 MLP (pooled) | not planned | 0.804 (0.784 to 0.824) | 0.721 (0.713 to 0.729) |
| linear: permutation null 95th | 0.712 | 0.858 | 0.713 |
| baseline: TF-IDF rendered text | 0.832 | 0.937 | 0.330 |
| baseline: metadata | 0.591 | 0.837 | 0.497 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.734 [0.606, 0.869] | 0.858 | 0.608 | 0.751 | 0.937 | 0.837 | 209 | 12 | 51 | 6 |  |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 0 | 28 | 1 | too few agents or runs in a class |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | too few agents or runs in a class |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | too few agents or runs in a class |
| A | DeepSeek V4.1 Flash | 0.663 [0.443, 0.833] | 0.680 | 0.461 | 0.597 | 0.683 | 0.560 | 36 | 9 | 12 | 1 |  |
| A | GPT-6 Luna | 0.767 [0.355, 0.992] | 0.855 | 0.702 | 0.675 | 0.981 | 0.623 | 126 | 3 | 43 | 1 |  |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | too few agents or runs in a class |
| A | **mean over authors** | 0.715 [0.492, 0.882] | 0.712 | 0.581 | 0.636 | 0.832 | 0.591 | 162 | 12 | 51 | 2 |  |
| B.II | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 0 | 28 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | held-out author or training set has too few in a class |
| B.II | DeepSeek V4.1 Flash | 0.733 [0.616, 0.865] | 0.643 | 0.568 | 0.560 | 0.420 | 0.449 | 36 | 9 | 12 | 1 | train 161 |
| B.II | GPT-6 Luna | 0.683 [0.258, 0.937] | 0.843 | 0.631 | 0.276 | 0.241 | 0.545 | 126 | 3 | 43 | 1 | train 32 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | held-out author or training set has too few in a class |
| B.II | **mean over authors** | 0.708 [0.493, 0.859] | 0.713 | 0.600 | 0.418 | 0.330 | 0.497 | 162 | 12 | 51 | 2 |  |

#### cheating: alone: spar_expl

Train spar_exploration_single; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.910 [0.781, 0.968] | 0.955 [0.918, 0.984] | 0.658 [0.456, 0.753] |
| 2 MLP, running mean | not planned | 0.940 (0.933 to 0.946) | 0.692 (0.667 to 0.717) |
| 3 MLP, running max | not planned | 0.945 (0.927 to 0.963) | 0.606 (0.569 to 0.642) |
| 4 attention pooling | not planned | 0.950 (0.950 to 0.950) | 0.698 (0.669 to 0.727) |
| 5 attention monitor | not planned | 0.936 (0.929 to 0.943) | 0.491 (0.463 to 0.520) |
| 6 GRU | not planned | 0.940 (0.936 to 0.944) | 0.477 (0.426 to 0.527) |
| 7 linear per step | not planned | 0.933 | 0.726 |
| 8 MLP (pooled) | not planned | 0.952 (0.944 to 0.959) | 0.661 (0.592 to 0.729) |
| linear: permutation null 95th | 0.679 | 0.788 | 0.679 |
| baseline: TF-IDF rendered text | 0.877 | 0.970 | 0.893 |
| baseline: metadata | 0.915 | 0.936 | 0.804 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.955 [0.918, 0.984] | 0.788 | 0.952 | 0.973 | 0.970 | 0.936 | 313 | 71 | 96 | 3 |  |
| A | Claude Haiku 5.5 | 0.935 [0.828, 1.000] | 0.754 | 0.787 | 0.819 | 0.729 | 0.877 | 36 | 5 | 12 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.888 [0.569, 1.000] | 0.643 | 0.962 | 0.904 | 0.965 | 0.946 | 36 | 26 | 12 | 1 |  |
| A | GPT-6 Luna | 0.905 [0.827, 0.964] | 0.770 | 0.948 | 0.973 | 0.936 | 0.923 | 241 | 40 | 72 | 1 |  |
| A | **mean over authors** | 0.910 [0.781, 0.968] | 0.679 | 0.899 | 0.899 | 0.877 | 0.915 | 313 | 71 | 96 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.652 [0.433, 0.859] | 0.917 | 0.555 | 0.832 | 0.916 | 0.774 | 36 | 5 | 12 | 1 | train 277 |
| B.II | DeepSeek V4.1 Flash | 0.746 [0.175, 0.893] | 0.693 | 0.800 | 0.888 | 1.000 | 0.996 | 36 | 26 | 12 | 1 | train 277 |
| B.II | GPT-6 Luna | 0.577 [0.484, 0.660] | 0.739 | 0.678 | 0.848 | 0.763 | 0.640 | 241 | 40 | 72 | 1 | train 72 |
| B.II | **mean over authors** | 0.658 [0.456, 0.753] | 0.679 | 0.678 | 0.856 | 0.893 | 0.804 | 313 | 71 | 96 | 3 |  |

#### cheating: alone: spar_solo

Train spar_screen_solo; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.926 [0.847, 0.987] | 0.962 [0.915, 0.995] | 0.727 [0.520, 0.921] |
| 2 MLP, running mean | not planned | 0.941 (0.937 to 0.944) | 0.630 (0.618 to 0.641) |
| 3 MLP, running max | not planned | 0.933 (0.918 to 0.949) | 0.642 (0.613 to 0.671) |
| 4 attention pooling | not planned | 0.929 (0.919 to 0.939) | 0.627 (0.614 to 0.640) |
| 5 attention monitor | not planned | 0.895 (0.881 to 0.909) | 0.482 (0.433 to 0.532) |
| 6 GRU | not planned | 0.889 (0.881 to 0.898) | 0.490 (0.452 to 0.527) |
| 7 linear per step | not planned | 0.959 | 0.835 |
| 8 MLP (pooled) | not planned | 0.966 (0.965 to 0.967) | 0.755 (0.729 to 0.781) |
| linear: permutation null 95th | 0.675 | 0.895 | 0.770 |
| baseline: TF-IDF rendered text | 0.893 | 0.985 | 0.837 |
| baseline: metadata | 0.871 | 0.926 | 0.624 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.962 [0.915, 0.995] | 0.895 | 0.964 | 0.965 | 0.985 | 0.926 | 90 | 47 | 90 | 3 |  |
| A | Claude Haiku 5.5 | 1.000 [1.000, 1.000] | 0.690 | 1.000 | 1.000 | 0.944 | 1.000 | 30 | 6 | 30 | 1 |  |
| A | DeepSeek V4.1 Flash |  |  |  |  |  |  | 30 | 30 | 30 | 1 | too few agents or runs in a class |
| A | GPT-6 Luna | 0.852 [0.688, 0.975] | 0.742 | 0.842 | 0.890 | 0.842 | 0.742 | 30 | 11 | 30 | 1 |  |
| A | **mean over authors** | 0.926 [0.847, 0.987] | 0.675 | 0.921 | 0.945 | 0.893 | 0.871 | 60 | 17 | 60 | 2 |  |
| B.II | Claude Haiku 5.5 | 0.611 [0.224, 1.000] | 0.854 | 0.660 | 0.812 | 0.938 | 0.646 | 30 | 6 | 30 | 1 | train 60 |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 30 | 30 | 30 | 1 | held-out author or training set has too few in a class |
| B.II | GPT-6 Luna | 0.842 [0.660, 0.995] | 0.768 | 0.861 | 0.866 | 0.737 | 0.603 | 30 | 11 | 30 | 1 | train 60 |
| B.II | **mean over authors** | 0.727 [0.520, 0.921] | 0.770 | 0.760 | 0.839 | 0.837 | 0.624 | 60 | 17 | 60 | 2 |  |

#### cheating: alone: goodfire

Train goodfire_reward_hack; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.908 [0.896, 0.919] | 0.921 [0.910, 0.931] | 0.822 [0.803, 0.840] |
| 2 MLP, running mean | not planned | 0.933 (0.930 to 0.935) | 0.894 (0.889 to 0.899) |
| 3 MLP, running max | not planned | 0.937 (0.936 to 0.938) | 0.909 (0.899 to 0.919) |
| 4 attention pooling | not planned | 0.936 (0.936 to 0.937) | 0.910 (0.907 to 0.912) |
| 5 attention monitor | not planned | 0.930 (0.926 to 0.933) | 0.913 (0.913 to 0.914) |
| 6 GRU | not planned | 0.933 (0.929 to 0.937) | 0.909 (0.908 to 0.911) |
| 7 linear per step | not planned | 0.903 | 0.890 |
| 8 MLP (pooled) | not planned | 0.916 (0.914 to 0.917) | 0.829 (0.822 to 0.837) |
| linear: permutation null 95th | 0.623 | 0.695 | 0.656 |
| baseline: TF-IDF rendered text | 0.921 | 0.927 | 0.916 |
| baseline: metadata | 0.781 | 0.809 | 0.761 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.921 [0.910, 0.931] | 0.695 | 0.927 |  | 0.927 | 0.809 | 2833 | 2083 | 2833 | 3 |  |
| A | GLM 5.2 | 0.871 [0.849, 0.891] | 0.597 | 0.863 |  | 0.908 | 0.730 | 1006 | 609 | 1006 | 1 |  |
| A | Kimi K3 | 0.902 [0.878, 0.924] | 0.610 | 0.893 |  | 0.900 | 0.797 | 935 | 733 | 935 | 1 |  |
| A | Qwen3.8 Max | 0.951 [0.930, 0.969] | 0.706 | 0.945 |  | 0.956 | 0.815 | 892 | 741 | 892 | 1 |  |
| A | **mean over authors** | 0.908 [0.896, 0.919] | 0.623 | 0.900 |  | 0.921 | 0.781 | 2833 | 2083 | 2833 | 3 |  |
| B.II | GLM 5.2 | 0.801 [0.774, 0.830] | 0.631 | 0.809 |  | 0.871 | 0.700 | 1006 | 609 | 1006 | 1 | train 1827 |
| B.II | Kimi K3 | 0.765 [0.727, 0.802] | 0.683 | 0.820 |  | 0.925 | 0.785 | 935 | 733 | 935 | 1 | train 1898 |
| B.II | Qwen3.8 Max | 0.901 [0.872, 0.929] | 0.787 | 0.938 |  | 0.953 | 0.797 | 892 | 741 | 892 | 1 | train 1941 |
| B.II | **mean over authors** | 0.822 [0.803, 0.840] | 0.656 | 0.855 |  | 0.916 | 0.761 | 2833 | 2083 | 2833 | 3 |  |

#### cheating: alone: cheatbench

Train cheatbench; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.956 [0.913, 0.987] | 0.951 [0.919, 0.979] | 0.775 [0.697, 0.846] |
| 2 MLP, running mean | not planned | 0.961 (0.954 to 0.968) | 0.667 (0.662 to 0.673) |
| 3 MLP, running max | not planned | 0.961 (0.955 to 0.968) | 0.823 (0.793 to 0.852) |
| 4 attention pooling | not planned | 0.953 (0.934 to 0.971) | 0.684 (0.659 to 0.710) |
| 5 attention monitor | not planned | 0.941 (0.932 to 0.949) | 0.745 (0.719 to 0.772) |
| 6 GRU | not planned | 0.948 (0.934 to 0.961) | 0.664 (0.600 to 0.727) |
| 7 linear per step | not planned | 0.792 | 0.659 |
| 8 MLP (pooled) | not planned | 0.964 (0.961 to 0.966) | 0.789 (0.781 to 0.797) |
| linear: permutation null 95th | 0.566 | 0.872 | 0.641 |
| baseline: TF-IDF rendered text | 0.834 | 0.966 | 0.786 |
| baseline: metadata | 0.904 | 0.944 | 0.745 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.951 [0.919, 0.979] | 0.872 | 0.957 |  | 0.966 | 0.944 | 224 | 134 | 224 | 4 |  |
| A | Claude 3.7 Sonnet | 0.924 [0.845, 0.983] | 0.725 | 0.895 |  | 0.761 | 0.855 | 60 | 15 | 60 | 1 |  |
| A | Claude Opus 4.6 | 1.000 [1.000, 1.000] | 0.732 | 0.977 |  | 1.000 | 0.991 | 111 | 106 | 111 | 1 |  |
| A | GPT-5.3 Codex | 0.953 [0.872, 1.000] | 0.614 | 0.850 |  | 0.752 | 0.842 | 35 | 9 | 35 | 1 |  |
| A | GPT-5.4 | 0.946 [0.804, 1.000] | 0.876 | 0.911 |  | 0.821 | 0.929 | 18 | 4 | 18 | 1 |  |
| A | **mean over authors** | 0.956 [0.913, 0.987] | 0.566 | 0.908 |  | 0.834 | 0.904 | 224 | 134 | 224 | 4 |  |
| B.II | Claude 3.7 Sonnet | 0.453 [0.294, 0.618] | 0.750 | 0.481 |  | 0.766 | 0.864 | 60 | 15 | 60 | 1 | train 164 |
| B.II | Claude Opus 4.6 | 0.958 [0.900, 1.000] | 0.998 | 0.992 |  | 0.940 | 0.575 | 111 | 106 | 111 | 1 | train 113 |
| B.II | GPT-5.3 Codex | 0.795 [0.610, 0.940] | 0.744 | 0.786 |  | 0.564 | 0.611 | 35 | 9 | 35 | 1 | train 189 |
| B.II | GPT-5.4 | 0.893 [0.712, 1.000] | 0.717 | 0.893 |  | 0.875 | 0.929 | 18 | 4 | 18 | 1 | train 206 |
| B.II | **mean over authors** | 0.775 [0.697, 0.846] | 0.641 | 0.788 |  | 0.786 | 0.745 | 224 | 134 | 224 | 4 |  |

#### cheating: alone: evilgenie

Train evilgenie; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.827 [0.785, 0.881] | 0.872 [0.831, 0.911] | 0.885 [0.845, 0.915] |
| 2 MLP, running mean | not planned | 0.720 (0.714 to 0.725) | 0.751 (0.749 to 0.753) |
| 3 MLP, running max | not planned | 0.761 (0.724 to 0.798) | 0.797 (0.795 to 0.799) |
| 4 attention pooling | not planned | 0.732 (0.717 to 0.748) | 0.746 (0.730 to 0.762) |
| 5 attention monitor | not planned | 0.699 (0.683 to 0.716) | 0.722 (0.716 to 0.727) |
| 6 GRU | not planned | 0.715 (0.710 to 0.719) | 0.727 (0.724 to 0.730) |
| 7 linear per step | not planned | no feasible split | no feasible split |
| 8 MLP (pooled) | not planned | 0.848 (0.839 to 0.858) | 0.848 (0.834 to 0.862) |
| linear: permutation null 95th | 0.550 | 0.490 | 0.638 |
| baseline: TF-IDF rendered text | 0.900 | 0.926 | 0.892 |
| baseline: metadata | 0.634 | 0.640 | 0.604 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.872 [0.831, 0.911] | 0.490 | 0.812 |  | 0.926 | 0.640 | 266 | 133 | 133 | 10 |  |
| A | Claude Fable 5 | 0.920 [0.760, 1.000] | 0.920 | 0.800 |  | 1.000 | 0.360 | 10 | 5 | 5 | 1 |  |
| A | Claude Opus 4.8 |  |  |  |  |  |  | 6 | 3 | 3 | 1 | too few agents or runs in a class |
| A | GPT-5.3 Codex | 0.898 [0.794, 0.973] | 0.609 | 0.878 |  | 0.905 | 0.755 | 42 | 21 | 21 | 1 |  |
| A | GPT-5.6 Luna | 0.760 [0.628, 0.908] | 0.710 | 0.765 |  | 0.923 | 0.786 | 28 | 14 | 14 | 1 |  |
| A | GPT-5.6 Sol | 0.882 [0.743, 1.000] | 0.668 | 0.903 |  | 0.972 | 0.868 | 24 | 12 | 12 | 1 |  |
| A | Gemini 3.1 Pro | 0.727 [0.561, 0.893] | 0.620 | 0.761 |  | 0.806 | 0.578 | 34 | 17 | 17 | 1 |  |
| A | Gemini 3.5 Flash | 0.809 [0.687, 0.902] | 0.630 | 0.694 |  | 0.816 | 0.574 | 56 | 28 | 28 | 1 |  |
| A | Grok 4.5 | 0.752 [0.592, 0.890] | 0.564 | 0.757 |  | 0.787 | 0.672 | 40 | 20 | 20 | 1 |  |
| A | Kimi K2.7 Code | 0.864 [0.753, 1.000] | 0.780 | 0.877 |  | 0.988 | 0.481 | 18 | 9 | 9 | 1 |  |
| A | Kimi K3 |  |  |  |  |  |  | 8 | 4 | 4 | 1 | too few agents or runs in a class |
| A | **mean over authors** | 0.827 [0.785, 0.881] | 0.550 | 0.804 |  | 0.900 | 0.634 | 252 | 126 | 126 | 8 |  |
| B.II | Claude Fable 5 | 1.000 [1.000, 1.000] | 0.882 | 1.000 |  | 1.000 | 0.320 | 10 | 5 | 5 | 1 | train 256 |
| B.II | Claude Opus 4.8 | 1.000 [1.000, 1.000] | 1.000 | 1.000 |  | 1.000 | 0.556 | 6 | 3 | 3 | 1 | train 260 |
| B.II | GPT-5.3 Codex | 0.912 [0.805, 0.984] | 0.758 | 0.855 |  | 0.925 | 0.723 | 42 | 21 | 21 | 1 | train 224 |
| B.II | GPT-5.6 Luna | 0.974 [0.928, 1.000] | 0.633 | 0.893 |  | 0.969 | 0.862 | 28 | 14 | 14 | 1 | train 238 |
| B.II | GPT-5.6 Sol | 0.993 [0.938, 1.000] | 0.724 | 0.986 |  | 0.917 | 0.722 | 24 | 12 | 12 | 1 | train 242 |
| B.II | Gemini 3.1 Pro | 0.564 [0.381, 0.737] | 0.712 | 0.581 |  | 0.540 | 0.478 | 34 | 17 | 17 | 1 | train 232 |
| B.II | Gemini 3.5 Flash | 0.736 [0.555, 0.865] | 0.637 | 0.740 |  | 0.689 | 0.589 | 56 | 28 | 28 | 1 | train 210 |
| B.II | Grok 4.5 | 0.830 [0.705, 0.945] | 0.660 | 0.858 |  | 0.882 | 0.838 | 40 | 20 | 20 | 1 | train 226 |
| B.II | Kimi K2.7 Code | 0.840 [0.642, 0.988] | 0.775 | 0.864 |  | 1.000 | 0.457 | 18 | 9 | 9 | 1 | train 248 |
| B.II | Kimi K3 | 1.000 [1.000, 1.000] | 0.941 | 1.000 |  | 1.000 | 0.500 | 8 | 4 | 4 | 1 | train 258 |
| B.II | **mean over authors** | 0.885 [0.845, 0.915] | 0.638 | 0.878 |  | 0.892 | 0.604 | 266 | 133 | 133 | 10 |  |

#### cheating: ours

Train controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.944 [0.928, 0.958] | 0.950 [0.940, 0.960] | 0.860 [0.838, 0.880] |
| 2 MLP, running mean | not planned | 0.942 (0.936 to 0.948) | 0.849 (0.841 to 0.856) |
| 3 MLP, running max | not planned | 0.966 (0.966 to 0.967) | 0.906 (0.902 to 0.911) |
| 4 attention pooling | not planned | 0.962 (0.956 to 0.969) | 0.881 (0.877 to 0.885) |
| 5 attention monitor | not planned | 0.969 (0.968 to 0.970) | 0.875 (0.865 to 0.885) |
| 6 GRU | not planned | 0.963 (0.959 to 0.966) | 0.872 (0.852 to 0.892) |
| 7 linear per step | not planned | 0.869 | 0.787 |
| 8 MLP (pooled) | not planned | 0.957 (0.957 to 0.958) | 0.850 (0.839 to 0.861) |
| 9 stack | not planned | 0.984 | not planned |
| linear: permutation null 95th | 0.662 | 0.772 | 0.754 |
| baseline: TF-IDF rendered text | 0.961 | 0.981 | 0.945 |
| baseline: metadata | 0.866 | 0.843 | 0.632 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.950 [0.940, 0.960] | 0.772 | 0.941 | 0.966 | 0.981 | 0.843 | 1973 | 722 | 1074 | 9 |  |
| A | Claude Haiku 5.5 | 0.975 [0.957, 0.990] | 0.793 | 0.953 | 0.978 | 0.974 | 0.922 | 364 | 83 | 235 | 1 |  |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | too few agents or runs in a class |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | too few agents or runs in a class |
| A | DeepSeek V4.1 Flash | 0.920 [0.896, 0.942] | 0.644 | 0.915 | 0.935 | 0.949 | 0.850 | 596 | 322 | 292 | 1 |  |
| A | GPT-5 mini | 0.901 [0.836, 0.952] | 0.580 | 0.912 | 0.936 | 0.939 | 0.755 | 120 | 60 | 120 | 1 |  |
| A | GPT-6 Luna | 0.957 [0.937, 0.974] | 0.754 | 0.942 | 0.964 | 0.979 | 0.892 | 748 | 170 | 337 | 1 |  |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | too few agents or runs in a class |
| A | Qwen3.8 27B | 0.969 [0.937, 0.993] | 0.684 | 0.944 | 0.981 | 0.965 | 0.910 | 120 | 82 | 120 | 1 |  |
| A | Qwen3.8 Max |  |  |  |  |  |  | 6 | 5 | 2 | 1 | too few agents or runs in a class |
| A | **mean over authors** | 0.944 [0.928, 0.958] | 0.662 | 0.933 | 0.959 | 0.961 | 0.866 | 1948 | 717 | 1072 | 5 |  |
| B.II | Claude Haiku 5.5 | 0.895 [0.851, 0.933] | 0.882 | 0.885 | 0.942 | 0.981 | 0.830 | 364 | 83 | 235 | 1 | train 1520 |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | held-out author or training set has too few in a class |
| B.II | DeepSeek V4.1 Flash | 0.749 [0.698, 0.794] | 0.656 | 0.707 | 0.792 | 0.865 | 0.686 | 596 | 322 | 292 | 1 | train 1365 |
| B.II | GPT-5 mini | 0.857 [0.782, 0.919] | 0.736 | 0.822 | 0.929 | 0.958 | 0.440 | 120 | 60 | 120 | 1 | train 1853 |
| B.II | GPT-6 Luna | 0.822 [0.773, 0.869] | 0.793 | 0.831 | 0.892 | 0.966 | 0.712 | 748 | 170 | 337 | 1 | train 1174 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | held-out author or training set has too few in a class |
| B.II | Qwen3.8 27B | 0.976 [0.952, 0.993] | 0.908 | 0.974 | 0.983 | 0.957 | 0.491 | 120 | 82 | 120 | 1 | train 1853 |
| B.II | Qwen3.8 Max |  |  |  |  |  |  | 6 | 5 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | **mean over authors** | 0.860 [0.838, 0.880] | 0.754 | 0.844 | 0.908 | 0.945 | 0.632 | 1948 | 717 | 1072 | 5 |  |

#### cheating: external

Train goodfire_reward_hack, cheatbench, evilgenie; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.873 [0.846, 0.905] | 0.921 [0.911, 0.930] | 0.784 [0.752, 0.816] |
| linear: permutation null 95th | 0.563 | 0.721 | 0.598 |
| baseline: TF-IDF rendered text | 0.898 | 0.935 | 0.869 |
| baseline: metadata | 0.733 | 0.814 | 0.700 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.921 [0.911, 0.930] | 0.721 | 0.923 |  | 0.935 | 0.814 | 3323 | 2350 | 3190 | 15 |  |
| A | Claude 3.7 Sonnet | 0.924 [0.845, 0.983] | 0.580 | 0.895 |  | 0.761 | 0.855 | 60 | 15 | 60 | 1 |  |
| A | Claude Fable 5 | 0.920 [0.760, 1.000] | 0.886 | 0.800 |  | 1.000 | 0.360 | 10 | 5 | 5 | 1 |  |
| A | Claude Opus 4.6 | 1.000 [1.000, 1.000] | 0.713 | 0.977 |  | 1.000 | 0.991 | 111 | 106 | 111 | 1 |  |
| A | Claude Opus 4.8 |  |  |  |  |  |  | 6 | 3 | 3 | 1 | too few agents or runs in a class |
| A | GLM 5.2 | 0.871 [0.849, 0.891] | 0.612 | 0.863 |  | 0.908 | 0.730 | 1006 | 609 | 1006 | 1 |  |
| A | GPT-5.3 Codex | 0.904 [0.832, 0.959] | 0.631 | 0.858 |  | 0.921 | 0.829 | 77 | 30 | 56 | 1 |  |
| A | GPT-5.4 | 0.946 [0.804, 1.000] | 0.633 | 0.911 |  | 0.821 | 0.929 | 18 | 4 | 18 | 1 |  |
| A | GPT-5.6 Luna | 0.760 [0.628, 0.908] | 0.716 | 0.765 |  | 0.923 | 0.786 | 28 | 14 | 14 | 1 |  |
| A | GPT-5.6 Sol | 0.882 [0.743, 1.000] | 0.752 | 0.903 |  | 0.972 | 0.868 | 24 | 12 | 12 | 1 |  |
| A | Gemini 3.1 Pro | 0.727 [0.561, 0.893] | 0.747 | 0.761 |  | 0.806 | 0.578 | 34 | 17 | 17 | 1 |  |
| A | Gemini 3.5 Flash | 0.809 [0.687, 0.902] | 0.560 | 0.694 |  | 0.816 | 0.574 | 56 | 28 | 28 | 1 |  |
| A | Grok 4.5 | 0.752 [0.592, 0.890] | 0.650 | 0.757 |  | 0.787 | 0.672 | 40 | 20 | 20 | 1 |  |
| A | Kimi K2.7 Code | 0.864 [0.753, 1.000] | 0.827 | 0.877 |  | 0.988 | 0.481 | 18 | 9 | 9 | 1 |  |
| A | Kimi K3 | 0.906 [0.880, 0.929] | 0.620 | 0.911 |  | 0.914 | 0.799 | 943 | 737 | 939 | 1 |  |
| A | Qwen3.8 Max | 0.951 [0.930, 0.969] | 0.710 | 0.945 |  | 0.956 | 0.815 | 892 | 741 | 892 | 1 |  |
| A | **mean over authors** | 0.873 [0.846, 0.905] | 0.563 | 0.851 |  | 0.898 | 0.733 | 3317 | 2347 | 3187 | 14 |  |
| B.II | Claude 3.7 Sonnet | 0.578 [0.399, 0.740] | 0.661 | 0.766 |  | 0.756 | 0.858 | 60 | 15 | 60 | 1 | train 3263 |
| B.II | Claude Fable 5 | 1.000 [1.000, 1.000] | 0.886 | 0.840 |  | 1.000 | 0.600 | 10 | 5 | 5 | 1 | train 3313 |
| B.II | Claude Opus 4.6 | 0.219 [0.045, 0.376] | 0.984 | 0.928 |  | 0.892 | 0.855 | 111 | 106 | 111 | 1 | train 3212 |
| B.II | Claude Opus 4.8 | 1.000 [1.000, 1.000] | 1.000 | 1.000 |  | 1.000 | 1.000 | 6 | 3 | 3 | 1 | train 3317 |
| B.II | GLM 5.2 | 0.789 [0.760, 0.819] | 0.624 | 0.821 |  | 0.873 | 0.696 | 1006 | 609 | 1006 | 1 | train 2317 |
| B.II | GPT-5.3 Codex | 0.840 [0.744, 0.926] | 0.825 | 0.870 |  | 0.823 | 0.557 | 77 | 30 | 56 | 1 | train 3246 |
| B.II | GPT-5.4 | 0.875 [0.656, 1.000] | 0.894 | 0.893 |  | 0.893 | 0.929 | 18 | 4 | 18 | 1 | train 3305 |
| B.II | GPT-5.6 Luna | 0.837 [0.663, 0.959] | 0.692 | 0.801 |  | 0.985 | 0.617 | 28 | 14 | 14 | 1 | train 3295 |
| B.II | GPT-5.6 Sol | 0.903 [0.757, 1.000] | 0.758 | 0.958 |  | 0.938 | 0.514 | 24 | 12 | 12 | 1 | train 3299 |
| B.II | Gemini 3.1 Pro | 0.554 [0.377, 0.723] | 0.663 | 0.498 |  | 0.429 | 0.443 | 34 | 17 | 17 | 1 | train 3289 |
| B.II | Gemini 3.5 Flash | 0.624 [0.459, 0.760] | 0.605 | 0.607 |  | 0.719 | 0.546 | 56 | 28 | 28 | 1 | train 3267 |
| B.II | Grok 4.5 | 0.790 [0.637, 0.920] | 0.753 | 0.688 |  | 0.853 | 0.713 | 40 | 20 | 20 | 1 | train 3283 |
| B.II | Kimi K2.7 Code | 0.975 [0.926, 1.000] | 0.772 | 1.000 |  | 1.000 | 0.593 | 18 | 9 | 9 | 1 | train 3305 |
| B.II | Kimi K3 | 0.839 [0.808, 0.871] | 0.735 | 0.903 |  | 0.927 | 0.781 | 943 | 737 | 939 | 1 | train 2380 |
| B.II | Qwen3.8 Max | 0.940 [0.919, 0.960] | 0.771 | 0.934 |  | 0.952 | 0.801 | 892 | 741 | 892 | 1 | train 2431 |
| B.II | **mean over authors** | 0.784 [0.752, 0.816] | 0.598 | 0.834 |  | 0.869 | 0.700 | 3323 | 2350 | 3190 | 15 |  |

#### cheating: all

Train controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo, goodfire_reward_hack, cheatbench, evilgenie; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.891 [0.871, 0.914] | 0.920 [0.912, 0.927] | 0.795 [0.756, 0.822] |
| 2 MLP, running mean | not planned | 0.920 (0.916 to 0.925) | 0.801 (0.783 to 0.819) |
| 3 MLP, running max | not planned | 0.943 (0.941 to 0.944) | 0.766 (0.761 to 0.771) |
| 4 attention pooling | not planned | 0.937 (0.936 to 0.938) | 0.814 (0.805 to 0.823) |
| 5 attention monitor | not planned | 0.941 (0.937 to 0.945) | 0.782 (0.768 to 0.797) |
| 6 GRU | not planned | 0.938 (0.936 to 0.940) | 0.772 (0.771 to 0.774) |
| 7 linear per step | not planned | 0.829 | 0.742 |
| 8 MLP (pooled) | not planned | 0.937 (0.937 to 0.938) | 0.806 (0.782 to 0.830) |
| 9 stack | not planned | 0.967 | not planned |
| linear: permutation null 95th | 0.577 | 0.787 | 0.592 |
| baseline: TF-IDF rendered text | 0.914 | 0.964 | 0.895 |
| baseline: metadata | 0.768 | 0.850 | 0.684 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.920 [0.912, 0.927] | 0.787 | 0.918 |  | 0.964 | 0.850 | 5296 | 3072 | 4264 | 23 |  |
| A | Claude 3.7 Sonnet | 0.924 [0.845, 0.983] | 0.634 | 0.895 |  | 0.761 | 0.855 | 60 | 15 | 60 | 1 |  |
| A | Claude Fable 5 | 0.920 [0.760, 1.000] | 0.728 | 0.800 |  | 1.000 | 0.360 | 10 | 5 | 5 | 1 |  |
| A | Claude Haiku 5.5 | 0.975 [0.957, 0.990] | 0.773 | 0.953 |  | 0.974 | 0.922 | 364 | 83 | 235 | 1 |  |
| A | Claude Opus 4.6 | 1.000 [1.000, 1.000] | 0.648 | 0.977 |  | 1.000 | 0.991 | 111 | 106 | 111 | 1 |  |
| A | Claude Opus 4.8 |  |  |  |  |  |  | 6 | 3 | 3 | 1 | too few agents or runs in a class |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | too few agents or runs in a class |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | too few agents or runs in a class |
| A | DeepSeek V4.1 Flash | 0.920 [0.896, 0.942] | 0.642 | 0.915 |  | 0.949 | 0.850 | 596 | 322 | 292 | 1 |  |
| A | GLM 5.2 | 0.871 [0.849, 0.891] | 0.610 | 0.863 |  | 0.908 | 0.730 | 1006 | 609 | 1006 | 1 |  |
| A | GPT-5 mini | 0.901 [0.836, 0.952] | 0.628 | 0.912 |  | 0.939 | 0.755 | 120 | 60 | 120 | 1 |  |
| A | GPT-5.3 Codex | 0.904 [0.832, 0.959] | 0.645 | 0.858 |  | 0.921 | 0.829 | 77 | 30 | 56 | 1 |  |
| A | GPT-5.4 | 0.946 [0.804, 1.000] | 0.774 | 0.911 |  | 0.821 | 0.929 | 18 | 4 | 18 | 1 |  |
| A | GPT-5.6 Luna | 0.760 [0.628, 0.908] | 0.715 | 0.765 |  | 0.923 | 0.786 | 28 | 14 | 14 | 1 |  |
| A | GPT-5.6 Sol | 0.882 [0.743, 1.000] | 0.560 | 0.903 |  | 0.972 | 0.868 | 24 | 12 | 12 | 1 |  |
| A | GPT-6 Luna | 0.957 [0.937, 0.974] | 0.739 | 0.942 |  | 0.979 | 0.892 | 748 | 170 | 337 | 1 |  |
| A | Gemini 3.1 Pro | 0.727 [0.561, 0.893] | 0.680 | 0.761 |  | 0.806 | 0.578 | 34 | 17 | 17 | 1 |  |
| A | Gemini 3.5 Flash | 0.809 [0.687, 0.902] | 0.604 | 0.694 |  | 0.816 | 0.574 | 56 | 28 | 28 | 1 |  |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | too few agents or runs in a class |
| A | Grok 4.5 | 0.752 [0.592, 0.890] | 0.574 | 0.757 |  | 0.787 | 0.672 | 40 | 20 | 20 | 1 |  |
| A | Kimi K2.7 Code | 0.864 [0.753, 1.000] | 0.765 | 0.877 |  | 0.988 | 0.481 | 18 | 9 | 9 | 1 |  |
| A | Kimi K3 | 0.906 [0.880, 0.929] | 0.614 | 0.911 |  | 0.914 | 0.799 | 943 | 737 | 939 | 1 |  |
| A | Qwen3.8 27B | 0.969 [0.937, 0.993] | 0.701 | 0.944 |  | 0.965 | 0.910 | 120 | 82 | 120 | 1 |  |
| A | Qwen3.8 Max | 0.944 [0.918, 0.964] | 0.718 | 0.937 |  | 0.944 | 0.804 | 898 | 746 | 894 | 1 |  |
| A | **mean over authors** | 0.891 [0.871, 0.914] | 0.577 | 0.872 |  | 0.914 | 0.768 | 5271 | 3069 | 4261 | 19 |  |
| B.II | Claude 3.7 Sonnet | 0.686 [0.529, 0.829] | 0.463 | 0.841 |  | 0.807 | 0.862 | 60 | 15 | 60 | 1 | train 5236 |
| B.II | Claude Fable 5 | 0.720 [0.200, 1.000] | 0.566 | 1.000 |  | 1.000 | 0.600 | 10 | 5 | 5 | 1 | train 5286 |
| B.II | Claude Haiku 5.5 | 0.874 [0.832, 0.913] | 0.837 | 0.850 |  | 0.980 | 0.836 | 364 | 83 | 235 | 1 | train 4843 |
| B.II | Claude Opus 4.6 | 0.942 [0.853, 1.000] | 0.933 | 0.958 |  | 0.781 | 0.868 | 111 | 106 | 111 | 1 | train 5185 |
| B.II | Claude Opus 4.8 | 1.000 [1.000, 1.000] | 0.894 | 1.000 |  | 1.000 | 1.000 | 6 | 3 | 3 | 1 | train 5290 |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | held-out author or training set has too few in a class |
| B.II | DeepSeek V4.1 Flash | 0.671 [0.620, 0.724] | 0.715 | 0.685 |  | 0.868 | 0.721 | 596 | 322 | 292 | 1 | train 4688 |
| B.II | GLM 5.2 | 0.797 [0.771, 0.826] | 0.647 | 0.806 |  | 0.876 | 0.698 | 1006 | 609 | 1006 | 1 | train 4290 |
| B.II | GPT-5 mini | 0.695 [0.595, 0.791] | 0.692 | 0.694 |  | 0.963 | 0.482 | 120 | 60 | 120 | 1 | train 5176 |
| B.II | GPT-5.3 Codex | 0.732 [0.618, 0.824] | 0.712 | 0.821 |  | 0.840 | 0.547 | 77 | 30 | 56 | 1 | train 5219 |
| B.II | GPT-5.4 | 0.982 [0.893, 1.000] | 0.574 | 0.929 |  | 0.911 | 0.929 | 18 | 4 | 18 | 1 | train 5278 |
| B.II | GPT-5.6 Luna | 0.709 [0.530, 0.872] | 0.782 | 0.663 |  | 0.985 | 0.597 | 28 | 14 | 14 | 1 | train 5268 |
| B.II | GPT-5.6 Sol | 0.792 [0.625, 0.938] | 0.801 | 0.833 |  | 0.938 | 0.500 | 24 | 12 | 12 | 1 | train 5272 |
| B.II | GPT-6 Luna | 0.815 [0.771, 0.855] | 0.784 | 0.823 |  | 0.966 | 0.638 | 748 | 170 | 337 | 1 | train 4497 |
| B.II | Gemini 3.1 Pro | 0.478 [0.304, 0.640] | 0.691 | 0.280 |  | 0.443 | 0.439 | 34 | 17 | 17 | 1 | train 5262 |
| B.II | Gemini 3.5 Flash | 0.735 [0.629, 0.843] | 0.643 | 0.790 |  | 0.773 | 0.536 | 56 | 28 | 28 | 1 | train 5240 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | held-out author or training set has too few in a class |
| B.II | Grok 4.5 | 0.757 [0.570, 0.908] | 0.780 | 0.677 |  | 0.905 | 0.713 | 40 | 20 | 20 | 1 | train 5256 |
| B.II | Kimi K2.7 Code | 0.753 [0.580, 0.926] | 0.672 | 0.914 |  | 1.000 | 0.580 | 18 | 9 | 9 | 1 | train 5278 |
| B.II | Kimi K3 | 0.872 [0.843, 0.900] | 0.731 | 0.837 |  | 0.926 | 0.777 | 943 | 737 | 939 | 1 | train 4353 |
| B.II | Qwen3.8 27B | 0.974 [0.950, 0.990] | 0.831 | 0.959 |  | 0.984 | 0.552 | 120 | 82 | 120 | 1 | train 5176 |
| B.II | Qwen3.8 Max | 0.908 [0.883, 0.931] | 0.537 | 0.904 |  | 0.946 | 0.800 | 898 | 746 | 894 | 1 | train 4398 |
| B.II | **mean over authors** | 0.795 [0.756, 0.822] | 0.592 | 0.813 |  | 0.895 | 0.684 | 5277 | 3072 | 4264 | 20 |  |

#### cheating: leave out: controls

Train board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo, goodfire_reward_hack, cheatbench, evilgenie; test controls_600.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.772 [0.720, 0.819] | 0.684 [0.637, 0.727] | 0.682 [0.639, 0.726] |
| 2 MLP, running mean | not planned | 0.757 (0.744 to 0.771) | not planned |
| 3 MLP, running max | not planned | 0.817 (0.808 to 0.825) | not planned |
| 4 attention pooling | not planned | 0.768 (0.764 to 0.772) | not planned |
| 5 attention monitor | not planned | 0.793 (0.762 to 0.824) | not planned |
| 6 GRU | not planned | 0.786 (0.783 to 0.789) | not planned |
| 7 linear per step | not planned | 0.543 | not planned |
| 8 MLP (pooled) | not planned | 0.747 (0.745 to 0.749) | not planned |
| linear: permutation null 95th | 0.608 | 0.681 | 0.698 |
| baseline: TF-IDF rendered text | 0.735 | 0.802 | 0.748 |
| baseline: metadata | 0.452 | 0.541 | 0.512 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.684 [0.637, 0.727] | 0.681 | 0.688 |  | 0.802 | 0.541 | 600 | 343 | 600 | 5 | train 4696 |
| A | Claude Haiku 5.5 | 0.790 [0.696, 0.864] | 0.725 | 0.749 |  | 0.657 | 0.336 | 120 | 60 | 120 | 1 | train 244 |
| A | DeepSeek V4.1 Flash | 0.683 [0.583, 0.773] | 0.660 | 0.689 |  | 0.803 | 0.388 | 120 | 72 | 120 | 1 | train 476 |
| A | GPT-5 mini |  |  |  |  |  |  | 120 |  |  | 1 | author absent from the training sources |
| A | GPT-6 Luna | 0.842 [0.773, 0.903] | 0.729 | 0.798 |  | 0.746 | 0.633 | 120 | 69 | 120 | 1 | train 628 |
| A | Qwen3.8 27B |  |  |  |  |  |  | 120 |  |  | 1 | author absent from the training sources |
| A | **mean over authors** | 0.772 [0.720, 0.819] | 0.608 | 0.745 |  | 0.735 | 0.452 | 360 | 201 | 360 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.677 [0.577, 0.769] | 0.721 | 0.713 |  | 0.746 | 0.580 | 120 | 60 | 120 | 1 | train 4363 |
| B.II | DeepSeek V4.1 Flash | 0.604 [0.494, 0.705] | 0.809 | 0.550 |  | 0.641 | 0.571 | 120 | 72 | 120 | 1 | train 4208 |
| B.II | GPT-5 mini | 0.638 [0.543, 0.738] | 0.682 | 0.641 |  | 0.773 | 0.572 | 120 | 60 | 120 | 1 | train 4696 |
| B.II | GPT-6 Luna | 0.575 [0.465, 0.679] | 0.706 | 0.528 |  | 0.666 | 0.572 | 120 | 69 | 120 | 1 | train 4017 |
| B.II | Qwen3.8 27B | 0.916 [0.858, 0.961] | 0.876 | 0.926 |  | 0.913 | 0.267 | 120 | 82 | 120 | 1 | train 4696 |
| B.II | **mean over authors** | 0.682 [0.639, 0.726] | 0.698 | 0.672 |  | 0.748 | 0.512 | 600 | 343 | 600 | 5 |  |

#### cheating: leave out: board_seed

Train controls_600, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo, goodfire_reward_hack, cheatbench, evilgenie; test board_seed.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.762 [0.666, 0.851] | 0.465 [0.361, 0.570] | 0.749 [0.635, 0.843] |
| 2 MLP, running mean | not planned | 0.867 (0.854 to 0.879) | not planned |
| 3 MLP, running max | not planned | 0.776 (0.731 to 0.821) | not planned |
| 4 attention pooling | not planned | 0.602 (0.557 to 0.647) | not planned |
| 5 attention monitor | not planned | 0.748 (0.714 to 0.783) | not planned |
| 6 GRU | not planned | 0.714 (0.706 to 0.722) | not planned |
| 7 linear per step | not planned | 0.779 | not planned |
| 8 MLP (pooled) | not planned | 0.676 (0.671 to 0.681) | not planned |
| linear: permutation null 95th | 0.690 | 0.602 | 0.679 |
| baseline: TF-IDF rendered text | 0.826 | 0.856 | 0.808 |
| baseline: metadata | 0.851 | 0.846 | 0.838 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.465 [0.361, 0.570] | 0.602 | 0.398 |  | 0.856 | 0.846 | 134 | 56 | 43 | 1 | train 5162 |
| A | DeepSeek V4.1 Flash | 0.762 [0.666, 0.851] | 0.690 | 0.880 |  | 0.826 | 0.851 | 134 | 56 | 43 | 1 | train 462 |
| A | **mean over authors** | 0.762 [0.666, 0.851] | 0.690 | 0.880 |  | 0.826 | 0.851 | 134 | 56 | 43 | 1 |  |
| B.II | DeepSeek V4.1 Flash | 0.749 [0.635, 0.843] | 0.679 | 0.751 |  | 0.808 | 0.838 | 134 | 56 | 43 | 1 | train 4688 |
| B.II | **mean over authors** | 0.749 [0.635, 0.843] | 0.679 | 0.751 |  | 0.808 | 0.838 | 134 | 56 | 43 | 1 |  |

#### cheating: leave out: solo_design

Train controls_600, board_seed, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo, goodfire_reward_hack, cheatbench, evilgenie; test solo_design.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.867 [0.773, 0.929] | 0.821 [0.747, 0.885] | 0.693 [0.576, 0.801] |
| 2 MLP, running mean | not planned | 0.923 (0.919 to 0.928) | not planned |
| 3 MLP, running max | not planned | 0.937 (0.922 to 0.952) | not planned |
| 4 attention pooling | not planned | 0.957 (0.952 to 0.962) | not planned |
| 5 attention monitor | not planned | 0.935 (0.931 to 0.940) | not planned |
| 6 GRU | not planned | 0.936 (0.935 to 0.936) | not planned |
| 7 linear per step | not planned | 0.785 | not planned |
| 8 MLP (pooled) | not planned | 0.877 (0.855 to 0.898) | not planned |
| linear: permutation null 95th | 0.633 | 0.820 | 0.699 |
| baseline: TF-IDF rendered text | 0.824 | 0.899 | 0.858 |
| baseline: metadata | 0.605 | 0.662 | 0.556 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.821 [0.747, 0.885] | 0.820 | 0.824 |  | 0.899 | 0.662 | 177 | 84 | 59 | 3 | train 5119 |
| A | DeepSeek V4.1 Flash | 0.849 [0.765, 0.918] | 0.700 | 0.793 |  | 0.703 | 0.429 | 90 | 65 | 30 | 1 | train 506 |
| A | GPT-6 Luna | 0.885 [0.697, 0.981] | 0.662 | 0.874 |  | 0.945 | 0.781 | 81 | 14 | 27 | 1 | train 667 |
| A | Qwen3.8 Max |  |  |  |  |  |  | 6 | 5 | 2 | 1 | too few in a class on one side |
| A | **mean over authors** | 0.867 [0.773, 0.929] | 0.633 | 0.833 |  | 0.824 | 0.605 | 171 | 79 | 57 | 2 |  |
| B.II | DeepSeek V4.1 Flash | 0.603 [0.454, 0.737] | 0.678 | 0.612 |  | 0.806 | 0.414 | 90 | 65 | 30 | 1 | train 4601 |
| B.II | GPT-6 Luna | 0.784 [0.577, 0.928] | 0.738 | 0.681 |  | 0.910 | 0.698 | 81 | 14 | 27 | 1 | train 4401 |
| B.II | Qwen3.8 Max |  |  |  |  |  |  | 6 | 5 | 2 | 1 | too few in a class on one side |
| B.II | **mean over authors** | 0.693 [0.576, 0.801] | 0.699 | 0.647 |  | 0.858 | 0.556 | 171 | 79 | 57 | 2 |  |

#### cheating: leave out: spar_team

Train controls_600, board_seed, solo_design, spar_mixed_41, spar_exploration_single, spar_screen_solo, goodfire_reward_hack, cheatbench, evilgenie; test spar_screen_team.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.717 [0.649, 0.820] | 0.792 [0.727, 0.855] | 0.735 [0.669, 0.795] |
| 2 MLP, running mean | not planned | 0.825 (0.814 to 0.836) | not planned |
| 3 MLP, running max | not planned | 0.845 (0.820 to 0.871) | not planned |
| 4 attention pooling | not planned | 0.858 (0.848 to 0.867) | not planned |
| 5 attention monitor | not planned | 0.873 (0.873 to 0.873) | not planned |
| 6 GRU | not planned | 0.857 (0.817 to 0.896) | not planned |
| 7 linear per step | not planned | 0.754 | not planned |
| 8 MLP (pooled) | not planned | 0.821 (0.819 to 0.823) | not planned |
| linear: permutation null 95th | 0.635 | 0.717 | 0.698 |
| baseline: TF-IDF rendered text | 0.785 | 0.873 | 0.862 |
| baseline: metadata | 0.812 | 0.831 | 0.788 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.792 [0.727, 0.855] | 0.717 | 0.797 |  | 0.873 | 0.831 | 450 | 109 | 135 | 3 | train 4846 |
| A | Claude Haiku 5.5 | 0.516 [0.333, 0.794] | 0.763 | 0.559 |  | 0.914 | 0.964 | 150 | 12 | 45 | 1 | train 214 |
| A | DeepSeek V4.1 Flash | 0.703 [0.596, 0.797] | 0.493 | 0.782 |  | 0.501 | 0.584 | 150 | 64 | 45 | 1 | train 446 |
| A | GPT-6 Luna | 0.932 [0.832, 0.992] | 0.816 | 0.933 |  | 0.940 | 0.889 | 150 | 33 | 45 | 1 | train 598 |
| A | **mean over authors** | 0.717 [0.649, 0.820] | 0.635 | 0.758 |  | 0.785 | 0.812 | 450 | 109 | 135 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.699 [0.575, 0.808] | 0.737 | 0.777 |  | 0.926 | 0.926 | 150 | 12 | 45 | 1 | train 4543 |
| B.II | DeepSeek V4.1 Flash | 0.743 [0.633, 0.847] | 0.809 | 0.670 |  | 0.729 | 0.615 | 150 | 64 | 45 | 1 | train 4388 |
| B.II | GPT-6 Luna | 0.762 [0.642, 0.863] | 0.840 | 0.719 |  | 0.931 | 0.822 | 150 | 33 | 45 | 1 | train 4197 |
| B.II | **mean over authors** | 0.735 [0.669, 0.795] | 0.698 | 0.722 |  | 0.862 | 0.788 | 450 | 109 | 135 | 3 |  |

#### cheating: leave out: spar_mixed

Train controls_600, board_seed, solo_design, spar_screen_team, spar_exploration_single, spar_screen_solo, goodfire_reward_hack, cheatbench, evilgenie; test spar_mixed_41.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.789 [0.625, 0.894] | 0.915 [0.855, 0.959] | 0.638 [0.492, 0.771] |
| 2 MLP, running mean | not planned | 0.914 (0.901 to 0.927) | not planned |
| 3 MLP, running max | not planned | 0.944 (0.937 to 0.950) | not planned |
| 4 attention pooling | not planned | 0.926 (0.925 to 0.928) | not planned |
| 5 attention monitor | not planned | 0.926 (0.920 to 0.933) | not planned |
| 6 GRU | not planned | 0.918 (0.910 to 0.926) | not planned |
| 7 linear per step | not planned | 0.887 | not planned |
| 8 MLP (pooled) | not planned | 0.942 (0.938 to 0.946) | not planned |
| linear: permutation null 95th | 0.762 | 0.891 | 0.718 |
| baseline: TF-IDF rendered text | 0.950 | 0.984 | 0.839 |
| baseline: metadata | 0.652 | 0.783 | 0.418 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.915 [0.855, 0.959] | 0.891 | 0.920 |  | 0.984 | 0.783 | 209 | 12 | 51 | 6 | train 5087 |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 0 | 28 | 1 | too few in a class on one side |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 |  |  | 1 | author absent from the training sources |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 |  |  | 1 | author absent from the training sources |
| A | DeepSeek V4.1 Flash | 0.584 [0.296, 0.792] | 0.710 | 0.597 |  | 0.905 | 0.634 | 36 | 9 | 12 | 1 | train 560 |
| A | GPT-6 Luna | 0.995 [0.976, 1.000] | 0.865 | 0.997 |  | 0.995 | 0.669 | 126 | 3 | 43 | 1 | train 622 |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 |  |  | 1 | author absent from the training sources |
| A | **mean over authors** | 0.789 [0.625, 0.894] | 0.762 | 0.797 |  | 0.950 | 0.652 | 162 | 12 | 51 | 2 |  |
| B.II | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 0 | 28 | 1 | too few in a class on one side |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | too few in a class on one side |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | too few in a class on one side |
| B.II | DeepSeek V4.1 Flash | 0.560 [0.371, 0.786] | 0.659 | 0.613 |  | 0.716 | 0.605 | 36 | 9 | 12 | 1 | train 4527 |
| B.II | GPT-6 Luna | 0.715 [0.549, 0.862] | 0.840 | 0.442 |  | 0.962 | 0.230 | 126 | 3 | 43 | 1 | train 4465 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | too few in a class on one side |
| B.II | **mean over authors** | 0.638 [0.492, 0.771] | 0.718 | 0.527 |  | 0.839 | 0.418 | 162 | 12 | 51 | 2 |  |

#### cheating: leave out: spar_expl

Train controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_screen_solo, goodfire_reward_hack, cheatbench, evilgenie; test spar_exploration_single.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.949 [0.912, 0.981] | 0.906 [0.855, 0.951] | 0.646 [0.547, 0.747] |
| 2 MLP, running mean | not planned | 0.917 (0.898 to 0.935) | not planned |
| 3 MLP, running max | not planned | 0.986 (0.985 to 0.988) | not planned |
| 4 attention pooling | not planned | 0.973 (0.972 to 0.974) | not planned |
| 5 attention monitor | not planned | 0.981 (0.976 to 0.985) | not planned |
| 6 GRU | not planned | 0.979 (0.977 to 0.982) | not planned |
| 7 linear per step | not planned | 0.751 | not planned |
| 8 MLP (pooled) | not planned | 0.936 (0.927 to 0.945) | not planned |
| linear: permutation null 95th | 0.828 | 0.844 | 0.774 |
| baseline: TF-IDF rendered text | 0.973 | 0.987 | 0.956 |
| baseline: metadata | 0.698 | 0.855 | 0.588 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.906 [0.855, 0.951] | 0.844 | 0.901 |  | 0.987 | 0.855 | 313 | 71 | 96 | 3 | train 4983 |
| A | Claude Haiku 5.5 | 0.968 [0.933, 1.000] | 0.910 | 0.897 |  | 0.929 | 0.523 | 36 | 5 | 12 | 1 | train 328 |
| A | DeepSeek V4.1 Flash | 0.942 [0.841, 1.000] | 0.933 | 0.965 |  | 1.000 | 0.662 | 36 | 26 | 12 | 1 | train 560 |
| A | GPT-6 Luna | 0.938 [0.889, 0.973] | 0.792 | 0.857 |  | 0.989 | 0.909 | 241 | 40 | 72 | 1 | train 507 |
| A | **mean over authors** | 0.949 [0.912, 0.981] | 0.828 | 0.906 |  | 0.973 | 0.698 | 313 | 71 | 96 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.465 [0.256, 0.667] | 0.801 | 0.439 |  | 0.974 | 0.484 | 36 | 5 | 12 | 1 | train 4566 |
| B.II | DeepSeek V4.1 Flash | 0.646 [0.445, 0.818] | 0.804 | 0.727 |  | 0.923 | 0.873 | 36 | 26 | 12 | 1 | train 4411 |
| B.II | GPT-6 Luna | 0.828 [0.746, 0.899] | 0.852 | 0.707 |  | 0.970 | 0.406 | 241 | 40 | 72 | 1 | train 4425 |
| B.II | **mean over authors** | 0.646 [0.547, 0.747] | 0.774 | 0.624 |  | 0.956 | 0.588 | 313 | 71 | 96 | 3 |  |

#### cheating: leave out: spar_solo

Train controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, goodfire_reward_hack, cheatbench, evilgenie; test spar_screen_solo.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.923 [0.845, 0.983] | 0.852 [0.769, 0.918] | 0.930 [0.844, 0.992] |
| 2 MLP, running mean | not planned | 0.778 (0.726 to 0.829) | not planned |
| 3 MLP, running max | not planned | 0.940 (0.936 to 0.943) | not planned |
| 4 attention pooling | not planned | 0.921 (0.916 to 0.925) | not planned |
| 5 attention monitor | not planned | 0.926 (0.911 to 0.940) | not planned |
| 6 GRU | not planned | 0.937 (0.932 to 0.943) | not planned |
| 7 linear per step | not planned | 0.749 | not planned |
| 8 MLP (pooled) | not planned | 0.833 (0.832 to 0.833) | not planned |
| linear: permutation null 95th | 0.660 | 0.865 | 0.885 |
| baseline: TF-IDF rendered text | 0.971 | 0.943 | 0.986 |
| baseline: metadata | 0.735 | 0.793 | 0.656 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.852 [0.769, 0.918] | 0.865 | 0.808 |  | 0.943 | 0.793 | 90 | 47 | 90 | 3 | train 5206 |
| A | Claude Haiku 5.5 | 0.965 [0.884, 1.000] | 0.758 | 0.917 |  | 1.000 | 0.868 | 30 | 6 | 30 | 1 | train 334 |
| A | DeepSeek V4.1 Flash |  |  |  |  |  |  | 30 | 30 | 30 | 1 | too few in a class on one side |
| A | GPT-6 Luna | 0.880 [0.731, 0.989] | 0.650 | 0.919 |  | 0.943 | 0.603 | 30 | 11 | 30 | 1 | train 718 |
| A | **mean over authors** | 0.923 [0.845, 0.983] | 0.660 | 0.918 |  | 0.971 | 0.735 | 60 | 17 | 60 | 2 |  |
| B.II | Claude Haiku 5.5 | 0.993 [0.964, 1.000] | 0.924 | 0.972 |  | 1.000 | 0.785 | 30 | 6 | 30 | 1 | train 4783 |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 30 | 30 | 30 | 1 | too few in a class on one side |
| B.II | GPT-6 Luna | 0.866 [0.713, 0.991] | 0.890 | 0.632 |  | 0.971 | 0.526 | 30 | 11 | 30 | 1 | train 4437 |
| B.II | **mean over authors** | 0.930 [0.844, 0.992] | 0.885 | 0.802 |  | 0.986 | 0.656 | 60 | 17 | 60 | 2 |  |

#### cheating: leave out: goodfire

Train controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo, cheatbench, evilgenie; test goodfire_reward_hack.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.348 [0.308, 0.395] | 0.648 [0.624, 0.670] | 0.600 [0.574, 0.625] |
| 2 MLP, running mean | not planned | 0.664 (0.664 to 0.664) | not planned |
| 3 MLP, running max | not planned | 0.630 (0.593 to 0.667) | not planned |
| 4 attention pooling | not planned | 0.670 (0.661 to 0.680) | not planned |
| 5 attention monitor | not planned | 0.672 (0.646 to 0.698) | not planned |
| 6 GRU | not planned | 0.656 (0.618 to 0.695) | not planned |
| 7 linear per step | not planned | 0.770 | not planned |
| 8 MLP (pooled) | not planned | 0.657 (0.652 to 0.662) | not planned |
| linear: permutation null 95th | 0.664 | 0.625 | 0.522 |
| baseline: TF-IDF rendered text | 0.516 | 0.678 | 0.732 |
| baseline: metadata | 0.253 | 0.666 | 0.696 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.648 [0.624, 0.670] | 0.625 | 0.706 |  | 0.678 | 0.666 | 2833 | 2083 | 2833 | 3 | train 2463 |
| A | GLM 5.2 |  |  |  |  |  |  | 1006 |  |  | 1 | author absent from the training sources |
| A | Kimi K3 | 0.348 [0.308, 0.395] | 0.664 | 0.369 |  | 0.516 | 0.253 | 935 | 733 | 935 | 1 | train 8 |
| A | Qwen3.8 Max |  |  |  |  |  |  | 892 | 741 | 892 | 1 | too few in a class on one side |
| A | **mean over authors** | 0.348 [0.308, 0.395] | 0.664 | 0.369 |  | 0.516 | 0.253 | 935 | 733 | 935 | 1 |  |
| B.II | GLM 5.2 | 0.619 [0.584, 0.656] | 0.493 | 0.732 |  | 0.657 | 0.588 | 1006 | 609 | 1006 | 1 | train 2463 |
| B.II | Kimi K3 | 0.665 [0.614, 0.710] | 0.616 | 0.586 |  | 0.736 | 0.743 | 935 | 733 | 935 | 1 | train 2455 |
| B.II | Qwen3.8 Max | 0.516 [0.463, 0.568] | 0.526 | 0.519 |  | 0.804 | 0.758 | 892 | 741 | 892 | 1 | train 2457 |
| B.II | **mean over authors** | 0.600 [0.574, 0.625] | 0.522 | 0.613 |  | 0.732 | 0.696 | 2833 | 2083 | 2833 | 3 |  |

#### cheating: leave out: cheatbench

Train controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo, goodfire_reward_hack, evilgenie; test cheatbench.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.919 [0.816, 0.993] | 0.805 [0.751, 0.858] | 0.664 [0.548, 0.773] |
| 2 MLP, running mean | not planned | 0.717 (0.681 to 0.753) | not planned |
| 3 MLP, running max | not planned | 0.533 (0.515 to 0.551) | not planned |
| 4 attention pooling | not planned | 0.640 (0.612 to 0.669) | not planned |
| 5 attention monitor | not planned | 0.452 (0.424 to 0.481) | not planned |
| 6 GRU | not planned | 0.505 (0.451 to 0.559) | not planned |
| 7 linear per step | not planned | 0.587 | not planned |
| 8 MLP (pooled) | not planned | 0.578 (0.569 to 0.587) | not planned |
| linear: permutation null 95th | 0.911 | 0.670 | 0.541 |
| baseline: TF-IDF rendered text | 0.504 | 0.582 | 0.729 |
| baseline: metadata | 0.876 | 0.642 | 0.847 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.805 [0.751, 0.858] | 0.670 | 0.869 |  | 0.582 | 0.642 | 224 | 134 | 224 | 4 | train 5072 |
| A | Claude 3.7 Sonnet |  |  |  |  |  |  | 60 |  |  | 1 | author absent from the training sources |
| A | Claude Opus 4.6 |  |  |  |  |  |  | 111 |  |  | 1 | author absent from the training sources |
| A | GPT-5.3 Codex | 0.919 [0.816, 0.993] | 0.911 | 0.923 |  | 0.504 | 0.876 | 35 | 9 | 35 | 1 | train 42 |
| A | GPT-5.4 |  |  |  |  |  |  | 18 |  |  | 1 | author absent from the training sources |
| A | **mean over authors** | 0.919 [0.816, 0.993] | 0.911 | 0.923 |  | 0.504 | 0.876 | 35 | 9 | 35 | 1 |  |
| B.II | Claude 3.7 Sonnet | 0.603 [0.449, 0.750] | 0.553 | 0.865 |  | 0.719 | 0.864 | 60 | 15 | 60 | 1 | train 5072 |
| B.II | Claude Opus 4.6 | 0.877 [0.778, 0.956] | 0.893 | 0.953 |  | 0.647 | 0.875 | 111 | 106 | 111 | 1 | train 5072 |
| B.II | GPT-5.3 Codex | 0.551 [0.330, 0.754] | 0.707 | 0.778 |  | 0.568 | 0.722 | 35 | 9 | 35 | 1 | train 5030 |
| B.II | GPT-5.4 | 0.625 [0.228, 1.000] | 0.596 | 0.893 |  | 0.982 | 0.929 | 18 | 4 | 18 | 1 | train 5072 |
| B.II | **mean over authors** | 0.664 [0.548, 0.773] | 0.541 | 0.872 |  | 0.729 | 0.847 | 224 | 134 | 224 | 4 |  |

#### cheating: leave out: evilgenie

Train controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo, goodfire_reward_hack, cheatbench; test evilgenie.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.491 [0.265, 0.768] | 0.652 [0.607, 0.700] | 0.704 [0.593, 0.774] |
| 2 MLP, running mean | not planned | 0.613 (0.603 to 0.623) | not planned |
| 3 MLP, running max | not planned | 0.543 (0.536 to 0.550) | not planned |
| 4 attention pooling | not planned | 0.618 (0.609 to 0.627) | not planned |
| 5 attention monitor | not planned | 0.596 (0.567 to 0.625) | not planned |
| 6 GRU | not planned | 0.583 (0.581 to 0.586) | not planned |
| 7 linear per step | not planned | 0.672 | not planned |
| 8 MLP (pooled) | not planned | 0.633 (0.607 to 0.658) | not planned |
| linear: permutation null 95th | 0.668 | 0.587 | 0.638 |
| baseline: TF-IDF rendered text | 0.631 | 0.554 | 0.651 |
| baseline: metadata | 0.519 | 0.531 | 0.585 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.652 [0.607, 0.700] | 0.587 | 0.607 |  | 0.554 | 0.531 | 266 | 133 | 133 | 10 | train 5030 |
| A | Claude Fable 5 |  |  |  |  |  |  | 10 |  |  | 1 | author absent from the training sources |
| A | Claude Opus 4.8 |  |  |  |  |  |  | 6 |  |  | 1 | author absent from the training sources |
| A | GPT-5.3 Codex | 0.608 [0.490, 0.735] | 0.670 | 0.667 |  | 0.574 | 0.726 | 42 | 21 | 21 | 1 | train 35 |
| A | GPT-5.6 Luna |  |  |  |  |  |  | 28 |  |  | 1 | author absent from the training sources |
| A | GPT-5.6 Sol |  |  |  |  |  |  | 24 |  |  | 1 | author absent from the training sources |
| A | Gemini 3.1 Pro |  |  |  |  |  |  | 34 |  |  | 1 | author absent from the training sources |
| A | Gemini 3.5 Flash |  |  |  |  |  |  | 56 |  |  | 1 | author absent from the training sources |
| A | Grok 4.5 |  |  |  |  |  |  | 40 |  |  | 1 | author absent from the training sources |
| A | Kimi K2.7 Code |  |  |  |  |  |  | 18 |  |  | 1 | author absent from the training sources |
| A | Kimi K3 | 0.375 [0.000, 0.750] | 0.750 | 0.688 |  | 0.688 | 0.312 | 8 | 4 | 4 | 1 | train 935 |
| A | **mean over authors** | 0.491 [0.265, 0.768] | 0.668 | 0.677 |  | 0.631 | 0.519 | 50 | 25 | 25 | 2 |  |
| B.II | Claude Fable 5 | 0.640 [0.160, 0.960] | 0.340 | 0.920 |  | 0.880 | 0.600 | 10 | 5 | 5 | 1 | train 5030 |
| B.II | Claude Opus 4.8 | 0.889 [0.000, 1.000] | 0.672 | 1.000 |  | 0.667 | 1.000 | 6 | 3 | 3 | 1 | train 5030 |
| B.II | GPT-5.3 Codex | 0.628 [0.485, 0.780] | 0.712 | 0.732 |  | 0.442 | 0.585 | 42 | 21 | 21 | 1 | train 4995 |
| B.II | GPT-5.6 Luna | 0.714 [0.541, 0.872] | 0.755 | 0.587 |  | 0.418 | 0.551 | 28 | 14 | 14 | 1 | train 5030 |
| B.II | GPT-5.6 Sol | 0.715 [0.569, 0.875] | 0.862 | 0.792 |  | 0.674 | 0.528 | 24 | 12 | 12 | 1 | train 5030 |
| B.II | Gemini 3.1 Pro | 0.471 [0.311, 0.637] | 0.655 | 0.325 |  | 0.429 | 0.405 | 34 | 17 | 17 | 1 | train 5030 |
| B.II | Gemini 3.5 Flash | 0.736 [0.642, 0.838] | 0.653 | 0.700 |  | 0.686 | 0.514 | 56 | 28 | 28 | 1 | train 5030 |
| B.II | Grok 4.5 | 0.723 [0.542, 0.895] | 0.736 | 0.560 |  | 0.623 | 0.685 | 40 | 20 | 20 | 1 | train 5030 |
| B.II | Kimi K2.7 Code | 0.654 [0.519, 0.827] | 0.731 | 0.827 |  | 0.877 | 0.605 | 18 | 9 | 9 | 1 | train 5030 |
| B.II | Kimi K3 | 0.875 [0.750, 1.000] | 0.631 | 0.875 |  | 0.812 | 0.375 | 8 | 4 | 4 | 1 | train 4095 |
| B.II | **mean over authors** | 0.704 [0.593, 0.774] | 0.638 | 0.732 |  | 0.651 | 0.585 | 266 | 133 | 133 | 10 |  |

#### cheating: ours to external

Train controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo; test goodfire_reward_hack, cheatbench, evilgenie.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | 0.630 [0.609, 0.649] | 0.541 [0.461, 0.603] |
| linear: permutation null 95th |  | 0.625 | 0.599 |
| baseline: TF-IDF rendered text |  | 0.728 | 0.570 |
| baseline: metadata |  | 0.676 | 0.672 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.630 [0.609, 0.649] | 0.625 | 0.654 |  | 0.728 | 0.676 | 3323 | 2350 | 3190 | 15 | train 1973 |
| A | Claude 3.7 Sonnet |  |  |  |  |  |  | 60 |  |  | 1 | author absent from the training sources |
| A | Claude Fable 5 |  |  |  |  |  |  | 10 |  |  | 1 | author absent from the training sources |
| A | Claude Opus 4.6 |  |  |  |  |  |  | 111 |  |  | 1 | author absent from the training sources |
| A | Claude Opus 4.8 |  |  |  |  |  |  | 6 |  |  | 1 | author absent from the training sources |
| A | GLM 5.2 |  |  |  |  |  |  | 1006 |  |  | 1 | author absent from the training sources |
| A | GPT-5.3 Codex |  |  |  |  |  |  | 77 |  |  | 1 | author absent from the training sources |
| A | GPT-5.4 |  |  |  |  |  |  | 18 |  |  | 1 | author absent from the training sources |
| A | GPT-5.6 Luna |  |  |  |  |  |  | 28 |  |  | 1 | author absent from the training sources |
| A | GPT-5.6 Sol |  |  |  |  |  |  | 24 |  |  | 1 | author absent from the training sources |
| A | Gemini 3.1 Pro |  |  |  |  |  |  | 34 |  |  | 1 | author absent from the training sources |
| A | Gemini 3.5 Flash |  |  |  |  |  |  | 56 |  |  | 1 | author absent from the training sources |
| A | Grok 4.5 |  |  |  |  |  |  | 40 |  |  | 1 | author absent from the training sources |
| A | Kimi K2.7 Code |  |  |  |  |  |  | 18 |  |  | 1 | author absent from the training sources |
| A | Kimi K3 |  |  |  |  |  |  | 943 |  |  | 1 | author absent from the training sources |
| A | Qwen3.8 Max |  |  |  |  |  |  | 892 | 741 | 892 | 1 | too few in a class on one side |
| B.II | Claude 3.7 Sonnet | 0.385 [0.256, 0.535] | 0.518 | 0.486 |  | 0.659 | 0.862 | 60 | 15 | 60 | 1 | train 1973 |
| B.II | Claude Fable 5 | 0.560 [0.120, 0.961] | 0.804 | 0.720 |  | 0.440 | 0.600 | 10 | 5 | 5 | 1 | train 1973 |
| B.II | Claude Opus 4.6 | 0.919 [0.846, 0.978] | 0.945 | 0.709 |  | 0.851 | 0.892 | 111 | 106 | 111 | 1 | train 1973 |
| B.II | Claude Opus 4.8 | 0.556 [0.000, 1.000] | 1.000 | 0.889 |  | 0.111 | 1.000 | 6 | 3 | 3 | 1 | train 1973 |
| B.II | GLM 5.2 | 0.595 [0.559, 0.632] | 0.547 | 0.603 |  | 0.707 | 0.578 | 1006 | 609 | 1006 | 1 | train 1973 |
| B.II | GPT-5.3 Codex | 0.435 [0.313, 0.563] | 0.775 | 0.600 |  | 0.328 | 0.502 | 77 | 30 | 56 | 1 | train 1973 |
| B.II | GPT-5.4 | 0.679 [0.267, 1.000] | 0.878 | 0.214 |  | 0.750 | 0.929 | 18 | 4 | 18 | 1 | train 1973 |
| B.II | GPT-5.6 Luna | 0.597 [0.393, 0.781] | 0.720 | 0.485 |  | 0.352 | 0.551 | 28 | 14 | 14 | 1 | train 1973 |
| B.II | GPT-5.6 Sol | 0.479 [0.257, 0.688] | 0.843 | 0.444 |  | 0.618 | 0.514 | 24 | 12 | 12 | 1 | train 1973 |
| B.II | Gemini 3.1 Pro | 0.346 [0.201, 0.481] | 0.556 | 0.339 |  | 0.512 | 0.401 | 34 | 17 | 17 | 1 | train 1973 |
| B.II | Gemini 3.5 Flash | 0.538 [0.389, 0.682] | 0.541 | 0.597 |  | 0.580 | 0.515 | 56 | 28 | 28 | 1 | train 1973 |
| B.II | Grok 4.5 | 0.555 [0.393, 0.740] | 0.731 | 0.515 |  | 0.395 | 0.677 | 40 | 20 | 20 | 1 | train 1973 |
| B.II | Kimi K2.7 Code | 0.407 [0.148, 0.716] | 0.880 | 0.407 |  | 0.704 | 0.593 | 18 | 9 | 9 | 1 | train 1973 |
| B.II | Kimi K3 | 0.633 [0.578, 0.684] | 0.641 | 0.637 |  | 0.726 | 0.735 | 943 | 737 | 939 | 1 | train 1973 |
| B.II | Qwen3.8 Max | 0.432 [0.384, 0.483] | 0.539 | 0.543 |  | 0.824 | 0.736 | 892 | 741 | 892 | 1 | train 1967 |
| B.II | **mean over authors** | 0.541 [0.461, 0.603] | 0.599 | 0.546 |  | 0.570 | 0.672 | 3323 | 2350 | 3190 | 15 |  |

#### cheating: external to ours

Train goodfire_reward_hack, cheatbench, evilgenie; test controls_600, board_seed, solo_design, spar_screen_team, spar_mixed_41, spar_exploration_single, spar_screen_solo.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | 0.397 [0.366, 0.428] | 0.443 [0.406, 0.480] |
| linear: permutation null 95th |  | 0.623 | 0.556 |
| baseline: TF-IDF rendered text |  | 0.600 | 0.503 |
| baseline: metadata |  | 0.538 | 0.431 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.397 [0.366, 0.428] | 0.623 | 0.547 |  | 0.600 | 0.538 | 1973 | 722 | 1074 | 9 | train 3323 |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 364 |  |  | 1 | author absent from the training sources |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 |  |  | 1 | author absent from the training sources |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 |  |  | 1 | author absent from the training sources |
| A | DeepSeek V4.1 Flash |  |  |  |  |  |  | 596 |  |  | 1 | author absent from the training sources |
| A | GPT-5 mini |  |  |  |  |  |  | 120 |  |  | 1 | author absent from the training sources |
| A | GPT-6 Luna |  |  |  |  |  |  | 748 |  |  | 1 | author absent from the training sources |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 |  |  | 1 | author absent from the training sources |
| A | Qwen3.8 27B |  |  |  |  |  |  | 120 |  |  | 1 | author absent from the training sources |
| A | Qwen3.8 Max |  |  |  |  |  |  | 6 | 5 | 2 | 1 | too few in a class on one side |
| B.II | Claude Haiku 5.5 | 0.320 [0.241, 0.401] | 0.571 | 0.409 |  | 0.579 | 0.470 | 364 | 83 | 235 | 1 | train 3323 |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | too few in a class on one side |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | too few in a class on one side |
| B.II | DeepSeek V4.1 Flash | 0.459 [0.400, 0.522] | 0.565 | 0.499 |  | 0.552 | 0.480 | 596 | 322 | 292 | 1 | train 3323 |
| B.II | GPT-5 mini | 0.392 [0.299, 0.494] | 0.528 | 0.499 |  | 0.497 | 0.580 | 120 | 60 | 120 | 1 | train 3323 |
| B.II | GPT-6 Luna | 0.368 [0.292, 0.434] | 0.597 | 0.427 |  | 0.306 | 0.341 | 748 | 170 | 337 | 1 | train 3323 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | too few in a class on one side |
| B.II | Qwen3.8 27B | 0.676 [0.572, 0.782] | 0.807 | 0.708 |  | 0.583 | 0.283 | 120 | 82 | 120 | 1 | train 3323 |
| B.II | Qwen3.8 Max |  |  |  |  |  |  | 6 | 5 | 2 | 1 | too few in a class on one side |
| B.II | **mean over authors** | 0.443 [0.406, 0.480] | 0.556 | 0.508 |  | 0.503 | 0.431 | 1948 | 717 | 1072 | 5 |  |

### coordination posted

#### coordination posted: alone: board_seed

Train board_seed; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.894 [0.784, 0.974] | 0.894 [0.784, 0.974] | n/a |
| 2 MLP, running mean | not planned | 0.951 (0.938 to 0.963) | no feasible split |
| 3 MLP, running max | not planned | 0.975 (0.971 to 0.979) | no feasible split |
| 4 attention pooling | not planned | 0.965 (0.953 to 0.977) | no feasible split |
| 5 attention monitor | not planned | 0.922 (0.898 to 0.945) | no feasible split |
| 6 GRU | not planned | 0.945 (0.944 to 0.946) | no feasible split |
| 7 linear per step | not planned | 0.932 | no feasible split |
| 8 MLP (pooled) | not planned | 0.900 (0.891 to 0.909) | no feasible split |
| linear: permutation null 95th | 0.753 | 0.722 |  |
| baseline: TF-IDF rendered text | 0.895 | 0.895 |  |
| baseline: metadata | 0.929 | 0.929 |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.894 [0.784, 0.974] | 0.722 | 0.882 | 0.908 | 0.895 | 0.929 | 134 | 87 | 43 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.894 [0.784, 0.974] | 0.753 | 0.882 | 0.908 | 0.895 | 0.929 | 134 | 87 | 43 | 1 |  |
| A | **mean over authors** | 0.894 [0.784, 0.974] | 0.753 | 0.882 | 0.908 | 0.895 | 0.929 | 134 | 87 | 43 | 1 |  |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 134 | 87 | 43 | 1 | one author only: nothing to hold out |

#### coordination posted: alone: solo_design

Train solo_design; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.814 [0.686, 0.910] | 0.747 [0.579, 0.878] | 0.715 [0.597, 0.803] |
| 2 MLP, running mean | not planned | 0.682 (0.670 to 0.694) | 0.599 (0.558 to 0.640) |
| 3 MLP, running max | not planned | 0.777 (0.713 to 0.841) | 0.709 (0.683 to 0.736) |
| 4 attention pooling | not planned | 0.707 (0.706 to 0.709) | 0.647 (0.626 to 0.667) |
| 5 attention monitor | not planned | 0.823 (0.798 to 0.849) | 0.531 (0.497 to 0.566) |
| 6 GRU | not planned | 0.811 (0.808 to 0.814) | 0.618 (0.599 to 0.637) |
| 7 linear per step | not planned | 0.665 | 0.435 |
| 8 MLP (pooled) | not planned | 0.741 (0.698 to 0.783) | 0.568 (0.503 to 0.633) |
| linear: permutation null 95th | 0.569 | 0.558 | 0.753 |
| baseline: TF-IDF rendered text | 0.798 | 0.774 | 0.516 |
| baseline: metadata | 0.834 | 0.753 | 0.540 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.747 [0.579, 0.878] | 0.558 | 0.752 | 0.804 | 0.774 | 0.753 | 177 | 38 | 59 | 3 |  |
| A | DeepSeek V4.1 Flash | 0.699 [0.490, 0.883] | 0.609 | 0.716 | 0.769 | 0.760 | 0.708 | 90 | 20 | 30 | 1 |  |
| A | GPT-6 Luna | 0.929 [0.812, 0.998] | 0.638 | 0.915 | 0.856 | 0.835 | 0.960 | 81 | 17 | 27 | 1 |  |
| A | Qwen3.8 Max |  |  |  |  |  |  | 6 | 1 | 2 | 1 | too few agents or runs in a class |
| A | **mean over authors** | 0.814 [0.686, 0.910] | 0.569 | 0.815 | 0.812 | 0.798 | 0.834 | 171 | 37 | 57 | 2 |  |
| B.II | DeepSeek V4.1 Flash | 0.709 [0.551, 0.848] | 0.709 | 0.618 | 0.677 | 0.671 | 0.708 | 90 | 20 | 30 | 1 | train 87 |
| B.II | GPT-6 Luna | 0.721 [0.571, 0.850] | 0.828 | 0.435 | 0.532 | 0.361 | 0.371 | 81 | 17 | 27 | 1 | train 96 |
| B.II | Qwen3.8 Max |  |  |  |  |  |  | 6 | 1 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | **mean over authors** | 0.715 [0.597, 0.803] | 0.753 | 0.526 | 0.605 | 0.516 | 0.540 | 171 | 37 | 57 | 2 |  |

#### coordination posted: alone: spar_team

Train spar_screen_team; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.752 [0.649, 0.841] | 0.909 [0.843, 0.963] | 0.690 [0.633, 0.755] |
| 2 MLP, running mean | not planned | 0.968 (0.958 to 0.977) | 0.817 (0.817 to 0.818) |
| 3 MLP, running max | not planned | 0.991 (0.990 to 0.992) | 0.968 (0.966 to 0.969) |
| 4 attention pooling | not planned | 0.988 (0.988 to 0.989) | 0.962 (0.951 to 0.973) |
| 5 attention monitor | not planned | 0.973 (0.960 to 0.986) | 0.913 (0.908 to 0.917) |
| 6 GRU | not planned | 0.989 (0.989 to 0.990) | 0.958 (0.954 to 0.962) |
| 7 linear per step | not planned | 0.907 | 0.742 |
| 8 MLP (pooled) | not planned | 0.889 (0.880 to 0.898) | 0.692 (0.683 to 0.702) |
| linear: permutation null 95th | 0.573 | 0.756 | 0.620 |
| baseline: TF-IDF rendered text | 0.700 | 0.889 | 0.680 |
| baseline: metadata | 0.813 | 0.794 | 0.625 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.909 [0.843, 0.963] | 0.756 | 0.898 | 0.889 | 0.889 | 0.794 | 450 | 406 | 135 | 3 |  |
| A | Claude Haiku 5.5 | 0.854 [0.790, 0.914] | 0.548 | 0.876 | 0.901 | 0.718 | 0.721 | 150 | 115 | 45 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.664 [0.393, 0.875] | 0.755 | 0.838 | 0.834 | 0.667 | 0.740 | 150 | 144 | 45 | 1 |  |
| A | GPT-6 Luna | 0.737 [0.613, 0.862] | 0.739 | 0.583 | 0.580 | 0.714 | 0.980 | 150 | 147 | 45 | 1 |  |
| A | **mean over authors** | 0.752 [0.649, 0.841] | 0.573 | 0.766 | 0.772 | 0.700 | 0.813 | 450 | 406 | 135 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.734 [0.635, 0.823] | 0.688 | 0.707 | 0.810 | 0.596 | 0.493 | 150 | 115 | 45 | 1 | train 300 |
| B.II | DeepSeek V4.1 Flash | 0.338 [0.202, 0.522] | 0.761 | 0.714 | 0.677 | 0.588 | 0.436 | 150 | 144 | 45 | 1 | train 300 |
| B.II | GPT-6 Luna | 0.998 [0.986, 1.000] | 0.916 | 0.810 | 0.998 | 0.857 | 0.946 | 150 | 147 | 45 | 1 | train 300 |
| B.II | **mean over authors** | 0.690 [0.633, 0.755] | 0.620 | 0.743 | 0.828 | 0.680 | 0.625 | 450 | 406 | 135 | 3 |  |

#### coordination posted: alone: spar_mixed

Train spar_mixed_41; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | 0.600 [0.464, 0.726] | n/a |
| 2 MLP, running mean | not planned | 0.986 (0.980 to 0.991) | no feasible split |
| 3 MLP, running max | not planned | 0.993 (0.993 to 0.993) | no feasible split |
| 4 attention pooling | not planned | 0.992 (0.985 to 0.999) | no feasible split |
| 5 attention monitor | not planned | 0.940 (0.907 to 0.973) | no feasible split |
| 6 GRU | not planned | 0.960 (0.948 to 0.973) | no feasible split |
| 7 linear per step | not planned | 0.954 | no feasible split |
| 8 MLP (pooled) | not planned | 0.909 (0.879 to 0.939) | no feasible split |
| linear: permutation null 95th |  | 0.805 |  |
| baseline: TF-IDF rendered text |  | 0.726 |  |
| baseline: metadata |  | 0.627 |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.600 [0.464, 0.726] | 0.805 | 0.763 | 0.779 | 0.726 | 0.627 | 209 | 205 | 51 | 6 |  |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 28 | 28 | 1 | too few agents or runs in a class |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 1 | 2 | 1 | too few agents or runs in a class |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 3 | 5 | 1 | too few agents or runs in a class |
| A | DeepSeek V4.1 Flash |  |  |  |  |  |  | 36 | 36 | 12 | 1 | too few agents or runs in a class |
| A | GPT-6 Luna |  |  |  |  |  |  | 126 | 125 | 43 | 1 | too few agents or runs in a class |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 12 | 12 | 1 | too few agents or runs in a class |
| B.II | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 28 | 28 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 1 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 3 | 5 | 1 | held-out author or training set has too few in a class |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 36 | 36 | 12 | 1 | held-out author or training set has too few in a class |
| B.II | GPT-6 Luna |  |  |  |  |  |  | 126 | 125 | 43 | 1 | held-out author or training set has too few in a class |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 12 | 12 | 1 | held-out author or training set has too few in a class |

#### coordination posted: ours

Train board_seed, solo_design, spar_screen_team, spar_mixed_41; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.944 [0.922, 0.962] | 0.969 [0.958, 0.979] | 0.904 [0.877, 0.930] |
| 2 MLP, running mean | not planned | 0.984 (0.984 to 0.984) | 0.960 (0.960 to 0.960) |
| 3 MLP, running max | not planned | 0.990 (0.990 to 0.990) | 0.977 (0.977 to 0.977) |
| 4 attention pooling | not planned | 0.989 (0.988 to 0.990) | 0.975 (0.972 to 0.978) |
| 5 attention monitor | not planned | 0.983 (0.979 to 0.988) | 0.953 (0.948 to 0.959) |
| 6 GRU | not planned | 0.987 (0.987 to 0.988) | 0.938 (0.935 to 0.941) |
| 7 linear per step | not planned | 0.958 | 0.885 |
| 8 MLP (pooled) | not planned | 0.954 (0.954 to 0.954) | 0.836 (0.823 to 0.848) |
| 9 stack | not planned | 0.988 | not planned |
| linear: permutation null 95th | 0.805 | 0.883 | 0.783 |
| baseline: TF-IDF rendered text | 0.897 | 0.957 | 0.863 |
| baseline: metadata | 0.880 | 0.924 | 0.815 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.969 [0.958, 0.979] | 0.883 | 0.970 | 0.970 | 0.957 | 0.924 | 970 | 736 | 288 | 7 |  |
| A | Claude Haiku 5.5 | 0.908 [0.859, 0.951] | 0.695 | 0.897 | 0.917 | 0.767 | 0.748 | 178 | 143 | 73 | 1 |  |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 1 | 2 | 1 | too few agents or runs in a class |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 3 | 5 | 1 | too few agents or runs in a class |
| A | DeepSeek V4.1 Flash | 0.937 [0.894, 0.971] | 0.860 | 0.943 | 0.947 | 0.943 | 0.911 | 410 | 287 | 130 | 1 |  |
| A | GPT-6 Luna | 0.986 [0.970, 0.997] | 0.957 | 0.992 | 0.989 | 0.983 | 0.980 | 357 | 289 | 115 | 1 |  |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 12 | 12 | 1 | too few agents or runs in a class |
| A | Qwen3.8 Max |  |  |  |  |  |  | 6 | 1 | 2 | 1 | too few agents or runs in a class |
| A | **mean over authors** | 0.944 [0.922, 0.962] | 0.805 | 0.944 | 0.951 | 0.897 | 0.880 | 945 | 719 | 286 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.870 [0.810, 0.926] | 0.671 | 0.766 | 0.906 | 0.814 | 0.642 | 178 | 143 | 73 | 1 | train 703 |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 1 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 3 | 5 | 1 | held-out author or training set has too few in a class |
| B.II | DeepSeek V4.1 Flash | 0.853 [0.796, 0.904] | 0.801 | 0.885 | 0.922 | 0.832 | 0.850 | 410 | 287 | 130 | 1 | train 548 |
| B.II | GPT-6 Luna | 0.988 [0.977, 0.996] | 0.969 | 0.991 | 0.994 | 0.943 | 0.954 | 357 | 289 | 115 | 1 | train 562 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 12 | 12 | 1 | held-out author or training set has too few in a class |
| B.II | Qwen3.8 Max |  |  |  |  |  |  | 6 | 1 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | **mean over authors** | 0.904 [0.877, 0.930] | 0.783 | 0.881 | 0.941 | 0.863 | 0.815 | 945 | 719 | 286 | 3 |  |

#### coordination posted: external

Train none; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | n/a | n/a |
| linear: permutation null 95th |  |  |  |
| baseline: TF-IDF rendered text |  |  |  |
| baseline: metadata |  |  |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| A | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| B.II | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |

#### coordination posted: all

Train board_seed, solo_design, spar_screen_team, spar_mixed_41; cross-validated within the training sources.

identical source set to 'ours'.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.944 [0.922, 0.962] | 0.969 [0.958, 0.979] | 0.904 [0.877, 0.930] |
| linear: permutation null 95th | 0.805 | 0.883 | 0.783 |
| baseline: TF-IDF rendered text | 0.897 | 0.957 | 0.863 |
| baseline: metadata | 0.880 | 0.924 | 0.815 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.969 [0.958, 0.979] | 0.883 | 0.970 | 0.970 | 0.957 | 0.924 | 970 | 736 | 288 | 7 |  |
| A | Claude Haiku 5.5 | 0.908 [0.859, 0.951] | 0.695 | 0.897 | 0.917 | 0.767 | 0.748 | 178 | 143 | 73 | 1 |  |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 1 | 2 | 1 |  |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 3 | 5 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.937 [0.894, 0.971] | 0.860 | 0.943 | 0.947 | 0.943 | 0.911 | 410 | 287 | 130 | 1 |  |
| A | GPT-6 Luna | 0.986 [0.970, 0.997] | 0.957 | 0.992 | 0.989 | 0.983 | 0.980 | 357 | 289 | 115 | 1 |  |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 12 | 12 | 1 |  |
| A | Qwen3.8 Max |  |  |  |  |  |  | 6 | 1 | 2 | 1 |  |
| A | **mean over authors** | 0.944 [0.922, 0.962] | 0.805 | 0.944 | 0.951 | 0.897 | 0.880 | 945 | 719 | 286 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.870 [0.810, 0.926] | 0.671 | 0.766 | 0.906 | 0.814 | 0.642 | 178 | 143 | 73 | 1 | train 703 |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 1 | 2 | 1 |  |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 3 | 5 | 1 |  |
| B.II | DeepSeek V4.1 Flash | 0.853 [0.796, 0.904] | 0.801 | 0.885 | 0.922 | 0.832 | 0.850 | 410 | 287 | 130 | 1 | train 548 |
| B.II | GPT-6 Luna | 0.988 [0.977, 0.996] | 0.969 | 0.991 | 0.994 | 0.943 | 0.954 | 357 | 289 | 115 | 1 | train 562 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 12 | 12 | 1 |  |
| B.II | Qwen3.8 Max |  |  |  |  |  |  | 6 | 1 | 2 | 1 |  |
| B.II | **mean over authors** | 0.904 [0.877, 0.930] | 0.783 | 0.881 | 0.941 | 0.863 | 0.815 | 945 | 719 | 286 | 3 |  |

#### coordination posted: leave out: board_seed

Train solo_design, spar_screen_team, spar_mixed_41; test board_seed.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.908 [0.846, 0.955] | 0.942 [0.885, 0.980] | 0.818 [0.736, 0.892] |
| 2 MLP, running mean | not planned | 0.960 (0.958 to 0.963) | not planned |
| 3 MLP, running max | not planned | 0.978 (0.973 to 0.982) | not planned |
| 4 attention pooling | not planned | 0.970 (0.966 to 0.974) | not planned |
| 5 attention monitor | not planned | 0.980 (0.976 to 0.984) | not planned |
| 6 GRU | not planned | 0.978 (0.978 to 0.978) | not planned |
| 7 linear per step | not planned | 0.949 | not planned |
| 8 MLP (pooled) | not planned | 0.907 (0.904 to 0.910) | not planned |
| linear: permutation null 95th | 0.882 | 0.905 | 0.844 |
| baseline: TF-IDF rendered text | 0.897 | 0.889 | 0.713 |
| baseline: metadata | 0.917 | 0.929 | 0.904 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.942 [0.885, 0.980] | 0.905 | 0.939 | 0.945 | 0.889 | 0.929 | 134 | 87 | 43 | 1 | train 836 |
| A | DeepSeek V4.1 Flash | 0.908 [0.846, 0.955] | 0.882 | 0.912 | 0.927 | 0.897 | 0.917 | 134 | 87 | 43 | 1 | train 276 |
| A | **mean over authors** | 0.908 [0.846, 0.955] | 0.882 | 0.912 | 0.927 | 0.897 | 0.917 | 134 | 87 | 43 | 1 |  |
| B.II | DeepSeek V4.1 Flash | 0.818 [0.736, 0.892] | 0.844 | 0.881 | 0.911 | 0.713 | 0.904 | 134 | 87 | 43 | 1 | train 548 |
| B.II | **mean over authors** | 0.818 [0.736, 0.892] | 0.844 | 0.881 | 0.911 | 0.713 | 0.904 | 134 | 87 | 43 | 1 |  |

#### coordination posted: leave out: solo_design

Train board_seed, spar_screen_team, spar_mixed_41; test solo_design.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.705 [0.550, 0.825] | 0.758 [0.647, 0.853] | 0.763 [0.615, 0.870] |
| 2 MLP, running mean | not planned | 0.963 (0.960 to 0.965) | not planned |
| 3 MLP, running max | not planned | 0.972 (0.969 to 0.975) | not planned |
| 4 attention pooling | not planned | 0.974 (0.973 to 0.974) | not planned |
| 5 attention monitor | not planned | 0.950 (0.932 to 0.968) | not planned |
| 6 GRU | not planned | 0.960 (0.958 to 0.961) | not planned |
| 7 linear per step | not planned | 0.772 | not planned |
| 8 MLP (pooled) | not planned | 0.703 (0.689 to 0.718) | not planned |
| linear: permutation null 95th | 0.657 | 0.736 | 0.712 |
| baseline: TF-IDF rendered text | 0.355 | 0.496 | 0.780 |
| baseline: metadata | 0.470 | 0.444 | 0.565 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.758 [0.647, 0.853] | 0.736 | 0.801 | 0.775 | 0.496 | 0.444 | 177 | 38 | 59 | 3 | train 793 |
| A | DeepSeek V4.1 Flash | 0.761 [0.583, 0.908] | 0.721 | 0.801 | 0.691 | 0.413 | 0.422 | 90 | 20 | 30 | 1 | train 320 |
| A | GPT-6 Luna | 0.649 [0.420, 0.839] | 0.688 | 0.604 | 0.668 | 0.298 | 0.518 | 81 | 17 | 27 | 1 | train 276 |
| A | Qwen3.8 Max |  |  |  |  |  |  | 6 |  |  | 1 | author absent from the training sources |
| A | **mean over authors** | 0.705 [0.550, 0.825] | 0.657 | 0.702 | 0.679 | 0.355 | 0.470 | 171 | 37 | 57 | 2 |  |
| B.II | DeepSeek V4.1 Flash | 0.659 [0.426, 0.826] | 0.723 | 0.682 | 0.682 | 0.774 | 0.577 | 90 | 20 | 30 | 1 | train 461 |
| B.II | GPT-6 Luna | 0.868 [0.695, 0.981] | 0.803 | 0.903 | 0.948 | 0.786 | 0.553 | 81 | 17 | 27 | 1 | train 466 |
| B.II | Qwen3.8 Max |  |  |  |  |  |  | 6 | 1 | 2 | 1 | too few in a class on one side |
| B.II | **mean over authors** | 0.763 [0.615, 0.870] | 0.712 | 0.792 | 0.815 | 0.780 | 0.565 | 171 | 37 | 57 | 2 |  |

#### coordination posted: leave out: spar_team

Train board_seed, solo_design, spar_mixed_41; test spar_screen_team.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.873 [0.796, 0.953] | 0.863 [0.812, 0.907] | 0.805 [0.721, 0.882] |
| 2 MLP, running mean | not planned | 0.949 (0.948 to 0.951) | not planned |
| 3 MLP, running max | not planned | 0.943 (0.931 to 0.955) | not planned |
| 4 attention pooling | not planned | 0.926 (0.921 to 0.931) | not planned |
| 5 attention monitor | not planned | 0.866 (0.819 to 0.914) | not planned |
| 6 GRU | not planned | 0.911 (0.907 to 0.914) | not planned |
| 7 linear per step | not planned | 0.824 | not planned |
| 8 MLP (pooled) | not planned | 0.808 (0.762 to 0.854) | not planned |
| linear: permutation null 95th | 0.743 | 0.808 | 0.790 |
| baseline: TF-IDF rendered text | 0.702 | 0.654 | 0.630 |
| baseline: metadata | 0.710 | 0.726 | 0.652 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.863 [0.812, 0.907] | 0.808 | 0.868 | 0.898 | 0.654 | 0.726 | 450 | 406 | 135 | 3 | train 520 |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 150 | 115 | 45 | 1 | too few in a class on one side |
| A | DeepSeek V4.1 Flash | 0.760 [0.622, 0.938] | 0.709 | 0.638 | 0.822 | 0.436 | 0.426 | 150 | 144 | 45 | 1 | train 260 |
| A | GPT-6 Luna | 0.986 [0.966, 1.000] | 0.938 | 0.995 | 1.000 | 0.968 | 0.993 | 150 | 147 | 45 | 1 | train 207 |
| A | **mean over authors** | 0.873 [0.796, 0.953] | 0.743 | 0.817 | 0.911 | 0.702 | 0.710 | 300 | 291 | 90 | 2 |  |
| B.II | Claude Haiku 5.5 | 0.759 [0.666, 0.849] | 0.768 | 0.840 | 0.836 | 0.652 | 0.603 | 150 | 115 | 45 | 1 | train 403 |
| B.II | DeepSeek V4.1 Flash | 0.671 [0.385, 0.899] | 0.820 | 0.721 | 0.878 | 0.397 | 0.397 | 150 | 144 | 45 | 1 | train 248 |
| B.II | GPT-6 Luna | 0.986 [0.966, 1.000] | 0.984 | 0.850 | 0.952 | 0.841 | 0.957 | 150 | 147 | 45 | 1 | train 262 |
| B.II | **mean over authors** | 0.805 [0.721, 0.882] | 0.790 | 0.804 | 0.889 | 0.630 | 0.652 | 450 | 406 | 135 | 3 |  |

#### coordination posted: leave out: spar_mixed

Train board_seed, solo_design, spar_screen_team; test spar_mixed_41.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | 0.854 [0.763, 0.924] | n/a |
| 2 MLP, running mean | not planned | 0.623 (0.593 to 0.652) | not planned |
| 3 MLP, running max | not planned | 0.957 (0.956 to 0.957) | not planned |
| 4 attention pooling | not planned | 0.956 (0.951 to 0.961) | not planned |
| 5 attention monitor | not planned | 0.299 (0.277 to 0.322) | not planned |
| 6 GRU | not planned | 0.921 (0.915 to 0.927) | not planned |
| 7 linear per step | not planned | 0.371 | not planned |
| 8 MLP (pooled) | not planned | 0.890 (0.876 to 0.904) | not planned |
| linear: permutation null 95th |  | 0.833 |  |
| baseline: TF-IDF rendered text |  | 0.470 |  |
| baseline: metadata |  | 0.423 |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.854 [0.763, 0.924] | 0.833 | 0.912 | 0.879 | 0.470 | 0.423 | 209 | 205 | 51 | 6 | train 761 |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 28 | 28 | 1 | too few in a class on one side |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 |  |  | 1 | author absent from the training sources |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 |  |  | 1 | author absent from the training sources |
| A | DeepSeek V4.1 Flash |  |  |  |  |  |  | 36 | 36 | 12 | 1 | too few in a class on one side |
| A | GPT-6 Luna |  |  |  |  |  |  | 126 | 125 | 43 | 1 | too few in a class on one side |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 |  |  | 1 | author absent from the training sources |
| B.II | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 28 | 28 | 1 | too few in a class on one side |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 1 | 2 | 1 | too few in a class on one side |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 3 | 5 | 1 | too few in a class on one side |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 36 | 36 | 12 | 1 | too few in a class on one side |
| B.II | GPT-6 Luna |  |  |  |  |  |  | 126 | 125 | 43 | 1 | too few in a class on one side |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 12 | 12 | 1 | too few in a class on one side |

#### coordination posted: ours to external

Train board_seed, solo_design, spar_screen_team, spar_mixed_41; test none.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | n/a | n/a |
| linear: permutation null 95th |  |  |  |
| baseline: TF-IDF rendered text |  |  |  |
| baseline: metadata |  |  |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| A | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| B.II | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |

#### coordination posted: external to ours

Train none; test board_seed, solo_design, spar_screen_team, spar_mixed_41.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | n/a | n/a |
| linear: permutation null 95th |  |  |  |
| baseline: TF-IDF rendered text |  |  |  |
| baseline: metadata |  |  |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| A | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| B.II | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |

### collusion crossed

#### collusion crossed: alone: board_seed

Train board_seed; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.901 [0.814, 0.964] | 0.901 [0.814, 0.964] | n/a |
| 2 MLP, running mean | not planned | 0.814 (0.813 to 0.814) | no feasible split |
| 3 MLP, running max | not planned | 0.888 (0.888 to 0.889) | no feasible split |
| 4 attention pooling | not planned | 0.880 (0.842 to 0.917) | no feasible split |
| 5 attention monitor | not planned | 0.811 (0.806 to 0.816) | no feasible split |
| 6 GRU | not planned | 0.811 (0.777 to 0.845) | no feasible split |
| 7 linear per step | not planned | 0.785 | no feasible split |
| 8 MLP (pooled) | not planned | 0.853 (0.835 to 0.870) | no feasible split |
| linear: permutation null 95th | 0.727 | 0.725 |  |
| baseline: TF-IDF rendered text | 0.950 | 0.950 |  |
| baseline: metadata | 0.852 | 0.852 |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.901 [0.814, 0.964] | 0.725 | 0.894 | 0.931 | 0.950 | 0.852 | 134 | 48 | 43 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.901 [0.814, 0.964] | 0.727 | 0.894 | 0.931 | 0.950 | 0.852 | 134 | 48 | 43 | 1 |  |
| A | **mean over authors** | 0.901 [0.814, 0.964] | 0.727 | 0.894 | 0.931 | 0.950 | 0.852 | 134 | 48 | 43 | 1 |  |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 134 | 48 | 43 | 1 | one author only: nothing to hold out |

#### collusion crossed: alone: spar_team

Train spar_screen_team; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.842 [0.710, 0.960] | 0.910 [0.857, 0.959] | 0.827 [0.762, 0.886] |
| 2 MLP, running mean | not planned | 0.931 (0.915 to 0.947) | 0.852 (0.801 to 0.903) |
| 3 MLP, running max | not planned | 0.922 (0.914 to 0.931) | 0.839 (0.813 to 0.865) |
| 4 attention pooling | not planned | 0.929 (0.923 to 0.936) | 0.663 (0.662 to 0.663) |
| 5 attention monitor | not planned | 0.926 (0.921 to 0.931) | 0.585 (0.583 to 0.586) |
| 6 GRU | not planned | 0.918 (0.905 to 0.931) | 0.657 (0.637 to 0.677) |
| 7 linear per step | not planned | 0.909 | 0.797 |
| 8 MLP (pooled) | not planned | 0.941 (0.939 to 0.943) | 0.782 (0.727 to 0.837) |
| linear: permutation null 95th | 0.585 | 0.716 | 0.822 |
| baseline: TF-IDF rendered text | 0.942 | 0.980 | 0.847 |
| baseline: metadata | 0.933 | 0.920 | 0.845 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.910 [0.857, 0.959] | 0.716 | 0.917 | 0.895 | 0.980 | 0.920 | 450 | 67 | 135 | 3 |  |
| A | Claude Haiku 5.5 | 0.706 [0.338, 1.000] | 0.717 | 0.883 | 0.986 | 0.993 | 0.980 | 150 | 6 | 45 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.820 [0.686, 0.916] | 0.530 | 0.829 | 0.901 | 0.871 | 0.822 | 150 | 37 | 45 | 1 |  |
| A | GPT-6 Luna | 0.999 [0.996, 1.000] | 0.622 | 0.993 | 0.999 | 0.961 | 0.997 | 150 | 24 | 45 | 1 |  |
| A | **mean over authors** | 0.842 [0.710, 0.960] | 0.585 | 0.902 | 0.962 | 0.942 | 0.933 | 450 | 67 | 135 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.797 [0.657, 0.932] | 0.926 | 0.846 | 0.808 | 0.988 | 0.928 | 150 | 6 | 45 | 1 | train 300 |
| B.II | DeepSeek V4.1 Flash | 0.719 [0.593, 0.848] | 0.763 | 0.268 | 0.869 | 0.554 | 0.679 | 150 | 37 | 45 | 1 | train 300 |
| B.II | GPT-6 Luna | 0.965 [0.931, 0.990] | 0.842 | 0.972 | 0.989 | 0.999 | 0.927 | 150 | 24 | 45 | 1 | train 300 |
| B.II | **mean over authors** | 0.827 [0.762, 0.886] | 0.822 | 0.695 | 0.889 | 0.847 | 0.845 | 450 | 67 | 135 | 3 |  |

#### collusion crossed: alone: spar_mixed

Train spar_mixed_41; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.818 [0.696, 0.961] | 0.967 [0.920, 1.000] | n/a |
| 2 MLP, running mean | not planned | 0.885 (0.864 to 0.907) | no feasible split |
| 3 MLP, running max | not planned | 0.711 (0.470 to 0.953) | no feasible split |
| 4 attention pooling | not planned | 0.801 (0.648 to 0.955) | no feasible split |
| 5 attention monitor | not planned | 0.868 (0.861 to 0.875) | no feasible split |
| 6 GRU | not planned | 0.855 (0.795 to 0.916) | no feasible split |
| 7 linear per step | not planned | 0.915 | no feasible split |
| 8 MLP (pooled) | not planned | 0.810 (0.793 to 0.826) | no feasible split |
| linear: permutation null 95th | 0.657 | 0.941 |  |
| baseline: TF-IDF rendered text | 0.281 | 0.950 |  |
| baseline: metadata | 0.704 | 0.960 |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.967 [0.920, 1.000] | 0.941 | 0.907 | 0.748 | 0.950 | 0.960 | 209 | 7 | 51 | 6 |  |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 0 | 28 | 1 | too few agents or runs in a class |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | too few agents or runs in a class |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | too few agents or runs in a class |
| A | DeepSeek V4.1 Flash | 0.818 [0.696, 0.961] | 0.657 | 0.724 | 0.729 | 0.281 | 0.704 | 36 | 7 | 12 | 1 |  |
| A | GPT-6 Luna |  |  |  |  |  |  | 126 | 0 | 43 | 1 | too few agents or runs in a class |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | too few agents or runs in a class |
| A | **mean over authors** | 0.818 [0.696, 0.961] | 0.657 | 0.724 | 0.729 | 0.281 | 0.704 | 36 | 7 | 12 | 1 |  |
| B.II | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 0 | 28 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | held-out author or training set has too few in a class |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 36 | 7 | 12 | 1 | held-out author or training set has too few in a class |
| B.II | GPT-6 Luna |  |  |  |  |  |  | 126 | 0 | 43 | 1 | held-out author or training set has too few in a class |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | held-out author or training set has too few in a class |

#### collusion crossed: ours

Train board_seed, spar_screen_team, spar_mixed_41; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.809 [0.704, 0.895] | 0.933 [0.896, 0.964] | 0.843 [0.780, 0.904] |
| 2 MLP, running mean | not planned | 0.933 (0.929 to 0.937) | 0.711 (0.684 to 0.739) |
| 3 MLP, running max | not planned | 0.932 (0.928 to 0.937) | 0.748 (0.705 to 0.791) |
| 4 attention pooling | not planned | 0.917 (0.917 to 0.917) | 0.626 (0.572 to 0.681) |
| 5 attention monitor | not planned | 0.932 (0.924 to 0.940) | 0.463 (0.435 to 0.491) |
| 6 GRU | not planned | 0.930 (0.922 to 0.938) | 0.527 (0.526 to 0.528) |
| 7 linear per step | not planned | 0.930 | 0.748 |
| 8 MLP (pooled) | not planned | 0.945 (0.939 to 0.950) | 0.668 (0.634 to 0.703) |
| 9 stack | not planned | 0.984 | not planned |
| linear: permutation null 95th | 0.683 | 0.758 | 0.739 |
| baseline: TF-IDF rendered text | 0.966 | 0.984 | 0.856 |
| baseline: metadata | 0.942 | 0.908 | 0.760 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.933 [0.896, 0.964] | 0.758 | 0.933 | 0.948 | 0.984 | 0.908 | 793 | 122 | 229 | 6 |  |
| A | Claude Haiku 5.5 | 0.582 [0.299, 0.817] | 0.859 | 0.851 | 0.757 | 0.966 | 0.983 | 178 | 6 | 73 | 1 |  |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | too few agents or runs in a class |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | too few agents or runs in a class |
| A | DeepSeek V4.1 Flash | 0.875 [0.805, 0.929] | 0.651 | 0.895 | 0.896 | 0.936 | 0.843 | 320 | 92 | 100 | 1 |  |
| A | GPT-6 Luna | 0.970 [0.922, 1.000] | 0.748 | 0.980 | 0.988 | 0.996 | 0.999 | 276 | 24 | 88 | 1 |  |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | too few agents or runs in a class |
| A | **mean over authors** | 0.809 [0.704, 0.895] | 0.683 | 0.909 | 0.880 | 0.966 | 0.942 | 774 | 122 | 229 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.802 [0.617, 0.955] | 0.841 | 0.800 | 0.813 | 0.988 | 0.979 | 178 | 6 | 73 | 1 | train 526 |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | held-out author or training set has too few in a class |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | held-out author or training set has too few in a class |
| B.II | DeepSeek V4.1 Flash | 0.751 [0.676, 0.817] | 0.730 | 0.764 | 0.723 | 0.579 | 0.535 | 320 | 92 | 100 | 1 | train 461 |
| B.II | GPT-6 Luna | 0.976 [0.946, 0.997] | 0.932 | 0.971 | 0.998 | 1.000 | 0.765 | 276 | 24 | 88 | 1 | train 466 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | held-out author or training set has too few in a class |
| B.II | **mean over authors** | 0.843 [0.780, 0.904] | 0.739 | 0.845 | 0.845 | 0.856 | 0.760 | 774 | 122 | 229 | 3 |  |

#### collusion crossed: external

Train none; cross-validated within the training sources.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | n/a | n/a |
| linear: permutation null 95th |  |  |  |
| baseline: TF-IDF rendered text |  |  |  |
| baseline: metadata |  |  |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| A | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| B.II | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |

#### collusion crossed: all

Train board_seed, spar_screen_team, spar_mixed_41; cross-validated within the training sources.

identical source set to 'ours'.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.809 [0.704, 0.895] | 0.933 [0.896, 0.964] | 0.843 [0.780, 0.904] |
| linear: permutation null 95th | 0.683 | 0.758 | 0.739 |
| baseline: TF-IDF rendered text | 0.966 | 0.984 | 0.856 |
| baseline: metadata | 0.942 | 0.908 | 0.760 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.933 [0.896, 0.964] | 0.758 | 0.933 | 0.948 | 0.984 | 0.908 | 793 | 122 | 229 | 6 |  |
| A | Claude Haiku 5.5 | 0.582 [0.299, 0.817] | 0.859 | 0.851 | 0.757 | 0.966 | 0.983 | 178 | 6 | 73 | 1 |  |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 |  |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 |  |
| A | DeepSeek V4.1 Flash | 0.875 [0.805, 0.929] | 0.651 | 0.895 | 0.896 | 0.936 | 0.843 | 320 | 92 | 100 | 1 |  |
| A | GPT-6 Luna | 0.970 [0.922, 1.000] | 0.748 | 0.980 | 0.988 | 0.996 | 0.999 | 276 | 24 | 88 | 1 |  |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 |  |
| A | **mean over authors** | 0.809 [0.704, 0.895] | 0.683 | 0.909 | 0.880 | 0.966 | 0.942 | 774 | 122 | 229 | 3 |  |
| B.II | Claude Haiku 5.5 | 0.802 [0.617, 0.955] | 0.841 | 0.800 | 0.813 | 0.988 | 0.979 | 178 | 6 | 73 | 1 | train 526 |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 |  |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 |  |
| B.II | DeepSeek V4.1 Flash | 0.751 [0.676, 0.817] | 0.730 | 0.764 | 0.723 | 0.579 | 0.535 | 320 | 92 | 100 | 1 | train 461 |
| B.II | GPT-6 Luna | 0.976 [0.946, 0.997] | 0.932 | 0.971 | 0.998 | 1.000 | 0.765 | 276 | 24 | 88 | 1 | train 466 |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 |  |
| B.II | **mean over authors** | 0.843 [0.780, 0.904] | 0.739 | 0.845 | 0.845 | 0.856 | 0.760 | 774 | 122 | 229 | 3 |  |

#### collusion crossed: leave out: board_seed

Train spar_screen_team, spar_mixed_41; test board_seed.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.857 [0.769, 0.929] | 0.888 [0.810, 0.945] | 0.750 [0.653, 0.843] |
| 2 MLP, running mean | not planned | 0.789 (0.765 to 0.814) | not planned |
| 3 MLP, running max | not planned | 0.860 (0.814 to 0.905) | not planned |
| 4 attention pooling | not planned | 0.817 (0.784 to 0.849) | not planned |
| 5 attention monitor | not planned | 0.900 (0.887 to 0.913) | not planned |
| 6 GRU | not planned | 0.895 (0.868 to 0.923) | not planned |
| 7 linear per step | not planned | 0.866 | not planned |
| 8 MLP (pooled) | not planned | 0.872 (0.866 to 0.878) | not planned |
| linear: permutation null 95th | 0.703 | 0.680 | 0.711 |
| baseline: TF-IDF rendered text | 0.946 | 0.955 | 0.876 |
| baseline: metadata | 0.512 | 0.669 | 0.257 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.888 [0.810, 0.945] | 0.680 | 0.934 | 0.923 | 0.955 | 0.669 | 134 | 48 | 43 | 1 | train 659 |
| A | DeepSeek V4.1 Flash | 0.857 [0.769, 0.929] | 0.703 | 0.901 | 0.896 | 0.946 | 0.512 | 134 | 48 | 43 | 1 | train 186 |
| A | **mean over authors** | 0.857 [0.769, 0.929] | 0.703 | 0.901 | 0.896 | 0.946 | 0.512 | 134 | 48 | 43 | 1 |  |
| B.II | DeepSeek V4.1 Flash | 0.750 [0.653, 0.843] | 0.711 | 0.636 | 0.502 | 0.876 | 0.257 | 134 | 48 | 43 | 1 | train 461 |
| B.II | **mean over authors** | 0.750 [0.653, 0.843] | 0.711 | 0.636 | 0.502 | 0.876 | 0.257 | 134 | 48 | 43 | 1 |  |

#### collusion crossed: leave out: spar_team

Train board_seed, spar_mixed_41; test spar_screen_team.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.845 [0.727, 0.932] | 0.814 [0.712, 0.899] | 0.778 [0.714, 0.844] |
| 2 MLP, running mean | not planned | 0.688 (0.684 to 0.691) | not planned |
| 3 MLP, running max | not planned | 0.614 (0.589 to 0.639) | not planned |
| 4 attention pooling | not planned | 0.715 (0.705 to 0.726) | not planned |
| 5 attention monitor | not planned | 0.656 (0.623 to 0.688) | not planned |
| 6 GRU | not planned | 0.651 (0.649 to 0.654) | not planned |
| 7 linear per step | not planned | 0.705 | not planned |
| 8 MLP (pooled) | not planned | 0.872 (0.869 to 0.874) | not planned |
| linear: permutation null 95th | 0.822 | 0.767 | 0.817 |
| baseline: TF-IDF rendered text | 0.774 | 0.657 | 0.390 |
| baseline: metadata | 0.561 | 0.651 | 0.452 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.814 [0.712, 0.899] | 0.767 | 0.759 | 0.831 | 0.657 | 0.651 | 450 | 67 | 135 | 3 | train 343 |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 150 | 6 | 45 | 1 | too few in a class on one side |
| A | DeepSeek V4.1 Flash | 0.845 [0.727, 0.932] | 0.822 | 0.791 | 0.842 | 0.774 | 0.561 | 150 | 37 | 45 | 1 | train 170 |
| A | GPT-6 Luna |  |  |  |  |  |  | 150 | 24 | 45 | 1 | too few in a class on one side |
| A | **mean over authors** | 0.845 [0.727, 0.932] | 0.822 | 0.791 | 0.842 | 0.774 | 0.561 | 150 | 37 | 45 | 1 |  |
| B.II | Claude Haiku 5.5 | 0.681 [0.582, 0.779] | 0.909 | 0.191 | 0.972 | 0.213 | 0.583 | 150 | 6 | 45 | 1 | train 226 |
| B.II | DeepSeek V4.1 Flash |  |  |  |  |  |  | 150 | 37 | 45 | 1 | too few in a class on one side |
| B.II | GPT-6 Luna | 0.875 [0.768, 0.955] | 0.768 | 0.712 | 0.974 | 0.568 | 0.320 | 150 | 24 | 45 | 1 | train 166 |
| B.II | **mean over authors** | 0.778 [0.714, 0.844] | 0.817 | 0.452 | 0.973 | 0.390 | 0.452 | 300 | 30 | 90 | 2 |  |

#### collusion crossed: leave out: spar_mixed

Train board_seed, spar_screen_team; test spar_mixed_41.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | 0.458 [0.086, 0.712] | 0.907 [0.812, 0.980] | 0.695 [0.487, 0.879] |
| 2 MLP, running mean | not planned | 0.965 (0.946 to 0.985) | not planned |
| 3 MLP, running max | not planned | 0.776 (0.760 to 0.792) | not planned |
| 4 attention pooling | not planned | 0.970 (0.961 to 0.978) | not planned |
| 5 attention monitor | not planned | 0.858 (0.806 to 0.910) | not planned |
| 6 GRU | not planned | 0.909 (0.884 to 0.934) | not planned |
| 7 linear per step | not planned | 0.857 | not planned |
| 8 MLP (pooled) | not planned | 0.904 (0.898 to 0.910) | not planned |
| linear: permutation null 95th | 0.754 | 0.956 | 0.817 |
| baseline: TF-IDF rendered text | 0.818 | 0.932 | 0.631 |
| baseline: metadata | 0.803 | 0.647 | 0.236 |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled | 0.907 [0.812, 0.980] | 0.956 | 0.826 | 0.938 | 0.932 | 0.647 | 209 | 7 | 51 | 6 | train 584 |
| A | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 0 | 28 | 1 | too few in a class on one side |
| A | Claude Opus 5.5 |  |  |  |  |  |  | 2 |  |  | 1 | author absent from the training sources |
| A | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 |  |  | 1 | author absent from the training sources |
| A | DeepSeek V4.1 Flash | 0.458 [0.086, 0.712] | 0.754 | 0.552 | 0.552 | 0.818 | 0.803 | 36 | 7 | 12 | 1 | train 284 |
| A | GPT-6 Luna |  |  |  |  |  |  | 126 | 0 | 43 | 1 | too few in a class on one side |
| A | Gemini 4 Argon |  |  |  |  |  |  | 12 |  |  | 1 | author absent from the training sources |
| A | **mean over authors** | 0.458 [0.086, 0.712] | 0.754 | 0.552 | 0.552 | 0.818 | 0.803 | 36 | 7 | 12 | 1 |  |
| B.II | Claude Haiku 5.5 |  |  |  |  |  |  | 28 | 0 | 28 | 1 | too few in a class on one side |
| B.II | Claude Opus 5.5 |  |  |  |  |  |  | 2 | 0 | 2 | 1 | too few in a class on one side |
| B.II | Claude Sonnet 5.5 |  |  |  |  |  |  | 5 | 0 | 5 | 1 | too few in a class on one side |
| B.II | DeepSeek V4.1 Flash | 0.695 [0.487, 0.879] | 0.817 | 0.581 | 0.695 | 0.631 | 0.236 | 36 | 7 | 12 | 1 | train 300 |
| B.II | GPT-6 Luna |  |  |  |  |  |  | 126 | 0 | 43 | 1 | too few in a class on one side |
| B.II | Gemini 4 Argon |  |  |  |  |  |  | 12 | 0 | 12 | 1 | too few in a class on one side |
| B.II | **mean over authors** | 0.695 [0.487, 0.879] | 0.817 | 0.581 | 0.695 | 0.631 | 0.236 | 36 | 7 | 12 | 1 |  |

#### collusion crossed: ours to external

Train board_seed, spar_screen_team, spar_mixed_41; test none.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | n/a | n/a |
| linear: permutation null 95th |  |  |  |
| baseline: TF-IDF rendered text |  |  |  |
| baseline: metadata |  |  |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| A | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| B.II | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |

#### collusion crossed: external to ours

Train none; test board_seed, spar_screen_team, spar_mixed_41.

| architecture | A (mean over authors) | B.I | B.II (mean over held-out authors) |
|---|---|---|---|
| 1 linear (pooled) | n/a | n/a | n/a |
| linear: permutation null 95th |  |  |  |
| baseline: TF-IDF rendered text |  |  |  |
| baseline: metadata |  |  |  |

| type | unit | think pair, all layers [95% CI] | null 95th | layer 38 | seven tokens | TF-IDF text | metadata | agents | pos | runs | authors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B.I | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| A | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |
| B.II | pooled |  |  |  |  |  |  |  |  |  |  | no labelled source |

