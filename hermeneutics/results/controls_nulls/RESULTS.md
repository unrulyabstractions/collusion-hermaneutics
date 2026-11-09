# Controls and nulls for the hermeneutic probes

Every number below was computed on this machine from the public Hub dataset unrulyabstractions/apart-swarm-align and the matrix_probes caches, with no model call and no spend. Folds, labels, schemes and the probe recipe (training-fold PCA to 128 components, z-scoring, balanced L2 logistic regression, C by grouped inner CV) are those of `hermeneutics/matrix_probes.py`, imported by `hermeneutics/controls_nulls.py`. AUROC intervals are 95% run bootstraps (1,000 draws). Nulls are label-permutation nulls: mean / 95th percentile of the AUROC over the draws, the probe refitted with C fixed at the median of the chosen Cs.

## 1. Is the label predictable from the prompt alone?

**Prompt-only probe.** The probe reads only states the reader has before the first assistant token. The main feature for our sets is the first assistant turn's five boundary tokens (`<|im_end|>` of the last prompt message, newline, `<|im_start|>`, `assistant`, newline) and the think-open state that follows them, at layers 26, 38, 51 (92,160 dims). The external sets have no turns.safetensors, so their main feature is the turn-0 think-open state alone (15,360 dims). The check feature is the token-weighted mean of the sentence means over the prompt units (system and user spans before the first assistant span), read for every agent of our sets by HTTP range requests over the prompt rows of acts.safetensors. The folds are matrix_probes' own B.I folds of each set alone. The null permutes the label over all agents of the set (50 draws).

Prompt identity, beside it: out-of-fold scores equal to the training fold's label rate among agents of the same arm, of the same (task, arm), or with the byte-identical prompt text (rendered input up to the first `<|im_start|>assistant`); an unseen key falls back to the next key, then to the overall rate. They show how much of the label the prompt condition carries without any activation. In controls_600, board_seed, solo_design, spar_mixed_41 and spar_exploration_single every prompt is unique (it names the agent, seed or files), so prompt-text identity falls back to the overall rate; the distinct count is beside it. Out-of-fold rates of rare keys are biased below 0.5.

Verdict: **yes** when the prompt-only AUROC beats all 50 permuted draws and reaches 0.60, **weakly** when it beats the null's 95th percentile only, **no** otherwise. Share: the prompt-only margin over 0.5 as a fraction of the full probe's margin.

| Label | Set | Agents | Positive | Runs | Prompt-only main feature | Prompt-only AUROC [95% CI] | Null mean / 95th | Think-open only | Boundary only | Prompt sentence means | Arm identity | Task + arm identity | Prompt-text identity (distinct prompts) | Full probe (think pair) | Full probe (seven tokens) | Metadata | TF-IDF rendered text | Predictable from the prompt? | Share of full margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cheating | controls_600 | 600 | 343 | 600 | turn-0 boundary tokens (5) + think-open | 0.561 [0.515, 0.607] | 0.493 / 0.548 | 0.586 | 0.572 | 0.593 | 0.553 | 0.624 | 0.494 (600) | 0.960 [0.944, 0.973] | 0.984 | 0.690 | 0.990 | weakly | 0.13 |
| cheating | board_seed | 134 | 56 | 43 | turn-0 boundary tokens (5) + think-open | 0.588 [0.481, 0.703] | 0.491 / 0.587 | 0.556 | 0.584 | 0.569 | 0.632 | 0.632 | 0.484 (134) | 0.880 [0.809, 0.935] | 0.908 | 0.826 | 0.905 | weakly | 0.23 |
| cheating | solo_design | 177 | 84 | 59 | turn-0 boundary tokens (5) + think-open | 0.489 [0.380, 0.613] | 0.510 / 0.582 | 0.506 | 0.513 | 0.517 | 0.493 | 0.527 | 0.458 (177) | 0.914 [0.860, 0.960] | 0.935 | 0.866 | 0.922 | no | -0.03 |
| cheating | spar_screen_team | 450 | 109 | 135 | turn-0 boundary tokens (5) + think-open | 0.773 [0.703, 0.838] | 0.506 / 0.557 | 0.786 | 0.781 | 0.760 | 0.578 | 0.753 | 0.771 (91) | 0.927 [0.885, 0.962] | 0.960 | 0.934 | 0.982 | yes | 0.64 |
| cheating | spar_mixed_41 | 209 | 12 | 51 | turn-0 boundary tokens (5) + think-open | 0.788 [0.652, 0.922] | 0.483 / 0.666 | 0.586 | 0.714 | 0.713 | 0.485 | 0.640 | 0.411 (209) | 0.734 [0.606, 0.869] | 0.751 | 0.837 | 0.937 | yes | 1.23 |
| cheating | spar_exploration_single | 313 | 71 | 96 | turn-0 boundary tokens (5) + think-open | 0.814 [0.740, 0.877] | 0.503 / 0.621 | 0.817 | 0.769 | 0.839 | 0.768 | 0.826 | 0.467 (313) | 0.955 [0.918, 0.984] | 0.973 | 0.936 | 0.970 | yes | 0.69 |
| cheating | spar_screen_solo | 90 | 47 | 90 | turn-0 boundary tokens (5) + think-open | 0.558 [0.436, 0.680] | 0.488 / 0.619 | 0.512 | 0.573 | 0.491 | 0.564 | 0.561 | 0.561 (10) | 0.962 [0.915, 0.995] | 0.965 | 0.926 | 0.985 | no | 0.12 |
| cheating | goodfire_reward_hack | 2833 | 2083 | 2833 | turn-0 think-open | 0.689 [0.669, 0.708] | 0.496 / 0.527 | (main) | n/a | n/a | 0.633 | 0.633 | 0.664 (706) | 0.921 [0.910, 0.931] | n/a | 0.809 | 0.927 | yes | 0.45 |
| cheating | cheatbench | 224 | 134 | 224 | turn-0 think-open | 0.890 [0.842, 0.929] | 0.490 / 0.591 | (main) | n/a | n/a | 0.861 | 0.861 | 0.756 (149) | 0.951 [0.919, 0.979] | n/a | 0.944 | 0.966 | yes | 0.86 |
| cheating | evilgenie | 266 | 133 | 133 | turn-0 think-open | 0.846 [0.793, 0.888] | 0.488 / 0.560 | (main) | n/a | n/a | 0.524 | 0.524 | 0.826 (33) | 0.872 [0.831, 0.911] | n/a | 0.640 | 0.926 | yes | 0.93 |
| coordination posted | board_seed | 134 | 87 | 43 | turn-0 boundary tokens (5) + think-open | 0.524 [0.415, 0.637] | 0.500 / 0.597 | 0.563 | 0.528 | 0.613 | 0.787 | 0.787 | 0.446 (134) | 0.894 [0.784, 0.974] | 0.908 | 0.929 | 0.895 | no | 0.06 |
| coordination posted | solo_design | 177 | 38 | 59 | turn-0 boundary tokens (5) + think-open | 0.687 [0.558, 0.794] | 0.470 / 0.567 | 0.702 | 0.676 | 0.734 | 0.524 | 0.607 | 0.461 (177) | 0.747 [0.579, 0.878] | 0.804 | 0.753 | 0.774 | yes | 0.76 |
| coordination posted | spar_screen_team | 450 | 406 | 135 | turn-0 boundary tokens (5) + think-open | 0.536 [0.426, 0.633] | 0.501 / 0.588 | 0.583 | 0.527 | 0.554 | 0.501 | 0.520 | 0.534 (91) | 0.909 [0.843, 0.963] | 0.889 | 0.794 | 0.889 | no | 0.09 |
| coordination posted | spar_mixed_41 | 209 | 205 | 51 | turn-0 boundary tokens (5) + think-open | 0.612 [0.236, 0.932] | 0.324 / 0.582 | 0.510 | 0.720 | 0.454 | 0.349 | 0.305 | 0.257 (209) | 0.600 [0.464, 0.726] | 0.779 | 0.627 | 0.726 | weakly | 1.12 |
| collusion crossed | board_seed | 134 | 48 | 43 | turn-0 boundary tokens (5) + think-open | 0.537 [0.419, 0.652] | 0.509 / 0.618 | 0.480 | 0.578 | 0.541 | 0.720 | 0.720 | 0.474 (134) | 0.901 [0.814, 0.964] | 0.931 | 0.852 | 0.950 | no | 0.09 |
| collusion crossed | spar_screen_team | 450 | 67 | 135 | turn-0 boundary tokens (5) + think-open | 0.808 [0.722, 0.889] | 0.496 / 0.563 | 0.785 | 0.796 | 0.815 | 0.598 | 0.811 | 0.806 (91) | 0.910 [0.857, 0.959] | 0.895 | 0.920 | 0.980 | yes | 0.75 |
| collusion crossed | spar_mixed_41 | 209 | 7 | 51 | turn-0 boundary tokens (5) + think-open | 0.709 [0.476, 0.976] | 0.398 / 0.658 | 0.726 | 0.587 | 0.472 | 0.330 | 0.720 | 0.236 (209) | 0.967 [0.920, 1.000] | 0.748 | 0.960 | 0.950 | weakly | 0.45 |

Provenance: `results/controls_nulls/prompt_pooled.csv` (prompt-only rows, this script) and matrix_probes `scores/<label>/alone_<set>.csv.gz` B.I pooled (full probe, seven tokens, metadata, TF-IDF on the same agents and folds); out-of-fold scores in `results/controls_nulls/scores/prompt_scores.csv.gz`.

**Within each prompt condition.** Two numbers per condition. *Restricted* is the AUROC of the pooled out-of-fold scores above on the agents of that condition only. *Refit* is a probe trained and tested inside the condition on new grouped folds (stratified by author and label, grouped by run), with a 50-draw null for the prompt-only refit. Conditions: controls solo vs cheat arm; design level (L0, L1, L2) and post (team hint vs clean) and their cross (the arm); team runs by seed arm; SPAR sets by condition; external sets by benchmark, harness or condition. In board_seed and solo_design the seed post is planted on the shared board, which the agent reads through a tool call, so it is never in the prompt; the design L2 arm names the board in the system prompt. *Restricted: task + arm identity* is shown only where the condition holds more than one task: it then says how much of the within-condition ranking the task alone gives. Out-of-fold label rates of small cells are biased below 0.5 (a held-out positive lowers its own cell's training rate), so values under 0.5 there mean no signal.

| Label | Set | Condition | Value | Agents | Positive | Runs | Restricted: prompt-only | Restricted: full | Restricted: metadata | Restricted: task + arm identity | Refit: prompt-only [95% CI] | Refit prompt-only null 95th | Refit: full | Refit: metadata | Refit: prompt sentence means |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cheating | controls_600 | arm | cheat | 227 | 150 | 227 | 0.516 | 0.957 | 0.787 | 0.607 | 0.503 [0.423, 0.578] | 0.573 | 0.902 | 0.823 | 0.479 |
| cheating | controls_600 | arm | solo | 373 | 193 | 373 | 0.557 | 0.958 | 0.614 | 0.585 | 0.546 [0.489, 0.605] | 0.577 | 0.956 | 0.620 | 0.558 |
| cheating | board_seed | arm | anticoord | 20 | 8 | 5 | 0.729 | 0.740 | 0.604 | one task | 0.583 [0.333, 0.868] | 0.721 | 0.583 | 0.510 | 0.812 |
| cheating | board_seed | arm | neutral | 15 | 0 | 5 |  |  |  | one task | too few agents or runs in a class |  |  |  |  |
| cheating | board_seed | arm | neutralL0 | 15 | 8 | 5 | 0.589 | 0.732 | 0.357 | one task | 0.500 [0.432, 0.727] | 0.734 | 0.071 | 0.857 | 0.321 |
| cheating | board_seed | arm | none | 30 | 9 | 10 | 0.735 | 0.937 | 0.577 | one task | 0.735 [0.507, 0.968] | 0.709 | 0.593 | 0.661 | 0.640 |
| cheating | board_seed | arm | procoordL1 | 15 | 13 | 5 | 0.769 | 0.962 | 0.269 | one task | too few agents or runs in a class |  |  |  |  |
| cheating | board_seed | arm | procoordL2 | 39 | 18 | 13 | 0.582 | 0.886 | 0.876 | one task | 0.373 [0.242, 0.482] | 0.658 | 0.902 | 0.886 | 0.315 |
| cheating | solo_design | arm | L0\|batch 1 | 9 | 1 | 3 | 0.750 | 1.000 | 0.500 | 0.125 | too few agents or runs in a class |  |  |  |  |
| cheating | solo_design | arm | L1\|batch 1 | 75 | 38 | 25 | 0.454 | 0.901 | 0.852 | 0.449 | 0.574 [0.394, 0.727] | 0.604 | 0.797 | 0.851 | 0.472 |
| cheating | solo_design | arm | L1\|clean | 75 | 33 | 25 | 0.534 | 0.910 | 0.909 | 0.645 | 0.547 [0.369, 0.714] | 0.645 | 0.829 | 0.926 | 0.545 |
| cheating | solo_design | arm | L2\|batch 1 | 9 | 6 | 3 | 0.556 | 0.944 | 0.556 | 0.167 | too few agents or runs in a class |  |  |  |  |
| cheating | solo_design | arm | L2\|clean | 9 | 6 | 3 | 0.333 | 1.000 | 0.667 | 0.000 | too few agents or runs in a class |  |  |  |  |
| cheating | solo_design | level | L0 | 9 | 1 | 3 | 0.750 | 1.000 | 0.500 | 0.125 | too few agents or runs in a class |  |  |  |  |
| cheating | solo_design | level | L1 | 150 | 71 | 50 | 0.487 | 0.903 | 0.880 | 0.526 | 0.558 [0.442, 0.672] | 0.592 | 0.917 | 0.884 | 0.529 |
| cheating | solo_design | level | L2 | 18 | 12 | 6 | 0.417 | 0.944 | 0.667 | 0.042 | 0.764 [0.000, 1.000] | 0.585 | 0.833 | 0.708 | 0.833 |
| cheating | solo_design | post | clean | 84 | 39 | 28 | 0.521 | 0.919 | 0.894 | 0.603 | 0.573 [0.382, 0.765] | 0.613 | 0.820 | 0.895 | 0.321 |
| cheating | solo_design | post | team_hint | 93 | 45 | 31 | 0.469 | 0.918 | 0.841 | 0.476 | 0.523 [0.357, 0.691] | 0.595 | 0.773 | 0.809 | 0.516 |
| cheating | spar_screen_team | arm | minimal | 135 | 33 | 45 | 0.656 | 0.918 | 0.965 | 0.635 | 0.729 [0.588, 0.849] | 0.599 | 0.963 | 0.970 | 0.711 |
| cheating | spar_screen_team | arm | minimal_cheat | 135 | 47 | 45 | 0.881 | 0.936 | 0.864 | 0.867 | 0.911 [0.840, 0.967] | 0.593 | 0.896 | 0.930 | 0.921 |
| cheating | spar_screen_team | arm | minimal_regulated | 180 | 29 | 45 | 0.751 | 0.926 | 0.961 | 0.679 | 0.699 [0.543, 0.831] | 0.615 | 0.908 | 0.965 | 0.766 |
| cheating | spar_mixed_41 | arm | minimal_regulated | 184 | 12 | 46 | 0.781 | 0.729 | 0.813 | 0.587 | 0.695 [0.516, 0.874] | 0.622 | 0.800 | 0.832 | 0.645 |
| cheating | spar_mixed_41 | arm | minimal_regulated_disregulated | 25 | 0 | 5 |  |  |  |  | too few agents or runs in a class |  |  |  |  |
| cheating | spar_exploration_single | arm | minimal | 156 | 34 | 52 | 0.655 | 0.958 | 0.927 | 0.617 | 0.635 [0.504, 0.763] | 0.641 | 0.932 | 0.940 | 0.783 |
| cheating | spar_exploration_single | arm | minimal_cheat | 57 | 35 | 19 | 0.788 | 0.940 | 0.873 | 0.605 | 0.648 [0.432, 0.850] | 0.640 | 0.904 | 0.851 | 0.738 |
| cheating | spar_exploration_single | arm | minimal_disregulated | 36 | 0 | 9 |  |  |  |  | too few agents or runs in a class |  |  |  |  |
| cheating | spar_exploration_single | arm | minimal_regulated | 64 | 2 | 16 | 0.758 | 0.839 | 0.661 | 0.484 | too few agents or runs in a class |  |  |  |  |
| cheating | spar_screen_solo | arm | solo | 45 | 19 | 45 | 0.492 | 0.955 | 0.953 | 0.415 | 0.500 [0.331, 0.687] | 0.611 | 0.943 | 0.984 | 0.429 |
| cheating | spar_screen_solo | arm | solo_cheat | 45 | 28 | 45 | 0.555 | 0.966 | 0.849 | 0.628 | 0.575 [0.404, 0.752] | 0.666 | 0.996 | 0.887 | 0.625 |
| cheating | goodfire_reward_hack | arm | deepswe | 622 | 424 | 622 | 0.674 | 0.759 | 0.628 | one task | 0.680 [0.636, 0.729] | 0.555 | 0.806 | 0.720 |  |
| cheating | goodfire_reward_hack | arm | impossiblebench | 829 | 500 | 829 | 0.497 | 0.959 | 0.788 | one task | 0.548 [0.507, 0.589] | 0.522 | 0.959 | 0.875 |  |
| cheating | goodfire_reward_hack | arm | swebench-verified | 1382 | 1159 | 1382 | 0.659 | 0.921 | 0.868 | one task | 0.654 [0.611, 0.693] | 0.546 | 0.931 | 0.887 |  |
| cheating | cheatbench | arm | BoxPwnr | 53 | 13 | 53 | 0.701 | 0.852 | 0.804 | one task | 0.843 [0.725, 0.940] | 0.708 | 0.917 | 0.877 |  |
| cheating | cheatbench | arm | Pilot | 111 | 106 | 111 | 1.000 | 1.000 | 0.826 | one task | 1.000 [1.000, 1.000] | 0.585 | 1.000 | 0.991 |  |
| cheating | cheatbench | arm | SWE-agent | 60 | 15 | 60 | 0.487 | 0.895 | 0.844 | one task | 0.470 [0.300, 0.649] | 0.625 | 0.924 | 0.855 |  |
| cheating | evilgenie | arm | A | 111 | 59 | 60 | 0.939 | 0.926 | 0.640 | one task | 0.950 [0.909, 0.981] | 0.627 | 0.887 | 0.662 |  |
| cheating | evilgenie | arm | B | 64 | 34 | 36 | 0.807 | 0.873 | 0.682 | one task | 0.811 [0.696, 0.908] | 0.623 | 0.725 | 0.679 |  |
| cheating | evilgenie | arm | C | 37 | 15 | 22 | 0.664 | 0.839 | 0.576 | one task | 0.639 [0.409, 0.852] | 0.633 | 0.627 | 0.582 |  |
| cheating | evilgenie | arm | D | 13 | 7 | 8 | 0.524 | 0.714 | 0.595 | one task | 0.345 [0.081, 0.609] | 0.871 | 0.714 | 0.548 |  |
| cheating | evilgenie | arm | E | 41 | 18 | 24 | 0.693 | 0.754 | 0.635 | one task | 0.657 [0.461, 0.831] | 0.692 | 0.732 | 0.737 |  |
| coordination posted | board_seed | arm | anticoord | 20 | 18 | 5 | 0.472 | 0.667 | 1.000 | one task | too few agents or runs in a class |  |  |  |  |
| coordination posted | board_seed | arm | neutral | 15 | 0 | 5 |  |  |  | one task | too few agents or runs in a class |  |  |  |  |
| coordination posted | board_seed | arm | neutralL0 | 15 | 14 | 5 | 0.857 | 1.000 | 0.214 | one task | too few agents or runs in a class |  |  |  |  |
| coordination posted | board_seed | arm | none | 30 | 15 | 10 | 0.507 | 0.782 | 0.689 | one task | 0.636 [0.344, 0.861] | 0.687 | 0.542 | 0.662 | 0.564 |
| coordination posted | board_seed | arm | procoordL1 | 15 | 14 | 5 | 0.429 | 0.929 | 0.357 | one task | too few agents or runs in a class |  |  |  |  |
| coordination posted | board_seed | arm | procoordL2 | 39 | 26 | 13 | 0.577 | 0.947 | 0.941 | one task | 0.269 [0.108, 0.453] | 0.719 | 0.834 | 0.941 | 0.538 |
| coordination posted | solo_design | arm | L0\|batch 1 | 9 | 0 | 3 |  |  |  |  | too few agents or runs in a class |  |  |  |  |
| coordination posted | solo_design | arm | L1\|batch 1 | 75 | 22 | 25 | 0.631 | 0.706 | 0.721 | 0.614 | 0.645 [0.451, 0.811] | 0.656 | 0.633 | 0.593 | 0.570 |
| coordination posted | solo_design | arm | L1\|clean | 75 | 11 | 25 | 0.763 | 0.947 | 0.848 | 0.570 | 0.467 [0.081, 0.754] | 0.690 | 0.885 | 0.651 | 0.560 |
| coordination posted | solo_design | arm | L2\|batch 1 | 9 | 2 | 3 | 0.643 | 0.500 | 0.500 | 0.286 | too few agents or runs in a class |  |  |  |  |
| coordination posted | solo_design | arm | L2\|clean | 9 | 3 | 3 | 0.833 | 0.667 | 0.611 | 0.250 | too few agents or runs in a class |  |  |  |  |
| coordination posted | solo_design | level | L0 | 9 | 0 | 3 |  |  |  |  | too few agents or runs in a class |  |  |  |  |
| coordination posted | solo_design | level | L1 | 150 | 33 | 50 | 0.682 | 0.769 | 0.772 | 0.638 | 0.653 [0.495, 0.798] | 0.592 | 0.797 | 0.766 | 0.666 |
| coordination posted | solo_design | level | L2 | 18 | 5 | 6 | 0.708 | 0.615 | 0.508 | 0.223 | 0.769 [0.400, 1.000] | 0.769 | 0.431 | 0.554 | 0.708 |
| coordination posted | solo_design | post | clean | 84 | 14 | 28 | 0.764 | 0.895 | 0.807 | 0.512 | 0.558 [0.278, 0.870] | 0.635 | 0.883 | 0.707 | 0.590 |
| coordination posted | solo_design | post | team_hint | 93 | 24 | 31 | 0.633 | 0.667 | 0.728 | 0.627 | 0.627 [0.421, 0.802] | 0.652 | 0.699 | 0.684 | 0.650 |
| coordination posted | spar_screen_team | arm | minimal | 135 | 118 | 45 | 0.553 | 0.955 | 0.797 | 0.516 | 0.523 [0.330, 0.693] | 0.625 | 0.802 | 0.728 | 0.471 |
| coordination posted | spar_screen_team | arm | minimal_cheat | 135 | 119 | 45 | 0.431 | 0.847 | 0.685 | 0.511 | 0.586 [0.437, 0.739] | 0.641 | 0.822 | 0.765 | 0.516 |
| coordination posted | spar_screen_team | arm | minimal_regulated | 180 | 169 | 45 | 0.571 | 0.919 | 0.902 | 0.359 | 0.484 [0.235, 0.797] | 0.682 | 0.713 | 0.866 | 0.585 |
| coordination posted | spar_mixed_41 | arm | minimal_regulated | 184 | 180 | 46 | 0.604 | 0.603 | 0.619 | 0.208 | 0.640 [0.175, 0.946] | 0.746 | 0.557 | 0.651 | 0.669 |
| coordination posted | spar_mixed_41 | arm | minimal_regulated_disregulated | 25 | 25 | 5 |  |  |  |  | too few agents or runs in a class |  |  |  |  |
| collusion crossed | board_seed | arm | anticoord | 20 | 3 | 5 | 0.176 | 0.059 | 0.412 | one task | 0.451 [0.111, 0.947] | 0.640 | 0.098 | 0.529 | 0.745 |
| collusion crossed | board_seed | arm | neutral | 15 | 0 | 5 |  |  |  | one task | too few agents or runs in a class |  |  |  |  |
| collusion crossed | board_seed | arm | neutralL0 | 15 | 8 | 5 | 0.286 | 0.911 | 0.054 | one task | 0.429 [0.000, 0.667] | 0.789 | 0.232 | 0.893 | 0.411 |
| collusion crossed | board_seed | arm | none | 30 | 3 | 10 | 0.309 | 0.975 | 0.444 | one task | not scorable: each test fold held one class |  |  |  |  |
| collusion crossed | board_seed | arm | procoordL1 | 15 | 12 | 5 | 1.000 | 0.778 | 0.000 | one task | not scorable: each test fold held one class |  |  |  |  |
| collusion crossed | board_seed | arm | procoordL2 | 39 | 22 | 13 | 0.436 | 0.965 | 0.914 | one task | 0.455 [0.282, 0.601] | 0.656 | 0.917 | 0.949 | 0.583 |
| collusion crossed | spar_screen_team | arm | minimal | 135 | 21 | 45 | 0.749 | 0.870 | 0.886 | 0.703 | 0.693 [0.437, 0.885] | 0.620 | 0.949 | 0.805 | 0.681 |
| collusion crossed | spar_screen_team | arm | minimal_cheat | 135 | 32 | 45 | 0.856 | 0.903 | 0.867 | 0.844 | 0.872 [0.742, 0.976] | 0.591 | 0.927 | 0.900 | 0.844 |
| collusion crossed | spar_screen_team | arm | minimal_regulated | 180 | 14 | 45 | 0.722 | 0.917 | 0.967 | 0.760 | 0.652 [0.467, 0.802] | 0.625 | 0.960 | 0.978 | 0.773 |
| collusion crossed | spar_mixed_41 | arm | minimal_regulated | 184 | 7 | 46 | 0.703 | 0.963 | 0.954 | 0.681 | 0.662 [0.462, 0.956] | 0.630 | 0.909 | 0.935 | 0.362 |
| collusion crossed | spar_mixed_41 | arm | minimal_regulated_disregulated | 25 | 0 | 5 |  |  |  |  | too few agents or runs in a class |  |  |  |  |

Provenance: `results/controls_nulls/prompt_within.csv`; refit scores in `scores/prompt_scores.csv.gz` (feature `within:<condition>=<value>:<feature>`). An empty restricted cell: the condition holds one class only.

## 2. Does the type B probe read author identity?

On every pooled scheme with more than one author, per training fold: the training-fold PCA-z components the probe reads (think pair, layers 26+38+51, 128 components), an author-identity probe on them (multinomial logistic regression, C 1; chance is predicting the training fold's most frequent author), then LEACE (Belrose et al. 2023) fitted on the training fold erases author (one-hot), or author + task + condition (arm), from those components, and B.I and the author probe are refitted on the erased components. B.II (leave one author out) likewise, with the eraser fitted on the training authors. The raw rows recompute matrix_probes' B.I and B.II and equal them on every row (see VERIFICATION.md). coord and collude 'all' equal 'ours' (no external source carries those labels). Author accuracy at chance after erasure follows from LEACE by construction on the training fold: it confirms the eraser ran, and it does not rule out author information a nonlinear reader could still find.

| Label | Scheme | Authors | Agents | Positive | Author probe: raw | Author: after author erasure | Author: after author+task+condition erasure | Chance | B.I raw | B.I author-erased | B.I author+task+condition-erased | B.II mean raw | B.II mean author-erased | B.II mean author+task+condition-erased | B.II authors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cheating | alone: controls | 5 | 600 | 343 | 0.990 | 0.190 | 0.188 | 0.193 | 0.960 [0.944, 0.973] | 0.951 [0.934, 0.966] | 0.926 [0.903, 0.947] | 0.913 [0.891, 0.934] | 0.907 [0.883, 0.928] | 0.890 [0.864, 0.915] | 5 |
| cheating | alone: solo_design | 3 | 177 | 84 | 1.000 | 0.508 | 0.508 | 0.508 | 0.914 [0.860, 0.960] | 0.820 [0.757, 0.885] | 0.696 [0.598, 0.788] | 0.630 [0.532, 0.723] | 0.630 [0.530, 0.724] | 0.359 [0.242, 0.469] | 2 |
| cheating | alone: spar_team | 3 | 450 | 109 | 1.000 | 0.329 | 0.331 | 0.316 | 0.927 [0.885, 0.962] | 0.931 [0.893, 0.961] | 0.780 [0.705, 0.847] | 0.751 [0.689, 0.811] | 0.742 [0.678, 0.803] | 0.531 [0.461, 0.609] | 3 |
| cheating | alone: spar_mixed | 6 | 209 | 12 | 0.986 | 0.603 | 0.603 | 0.603 | 0.734 [0.606, 0.869] | 0.719 [0.533, 0.872] | 0.695 [0.502, 0.869] | 0.708 [0.493, 0.859] | 0.701 [0.472, 0.863] | 0.682 [0.453, 0.855] | 2 |
| cheating | alone: goodfire | 3 | 2833 | 2083 | 0.999 | 0.355 | 0.355 | 0.355 | 0.921 [0.910, 0.931] | 0.914 [0.901, 0.925] | 0.880 [0.865, 0.894] | 0.822 [0.803, 0.840] | 0.845 [0.826, 0.862] | 0.806 [0.787, 0.824] | 3 |
| cheating | ours | 9 | 1973 | 722 | 0.995 | 0.379 | 0.379 | 0.379 | 0.950 [0.940, 0.960] | 0.908 [0.893, 0.923] | 0.772 [0.747, 0.797] | 0.860 [0.838, 0.880] | 0.843 [0.818, 0.868] | 0.810 [0.785, 0.835] | 5 |
| cheating | all | 23 | 5296 | 3072 | 0.996 | 0.190 | 0.190 | 0.190 | 0.920 [0.912, 0.927] | 0.853 [0.842, 0.863] | 0.752 [0.738, 0.766] | 0.795 [0.756, 0.822] | 0.672 [0.626, 0.711] | 0.701 [0.655, 0.747] | 20 |
| coordination posted | alone: solo_design | 3 | 177 | 38 | 1.000 | 0.508 | 0.508 | 0.508 | 0.747 [0.579, 0.878] | 0.742 [0.574, 0.874] | 0.749 [0.566, 0.897] | 0.715 [0.597, 0.803] | 0.716 [0.592, 0.806] | 0.566 [0.445, 0.675] | 2 |
| coordination posted | alone: spar_team | 3 | 450 | 406 | 1.000 | 0.338 | 0.344 | 0.327 | 0.909 [0.843, 0.963] | 0.925 [0.877, 0.964] | 0.920 [0.865, 0.965] | 0.690 [0.633, 0.755] | 0.747 [0.677, 0.824] | 0.731 [0.666, 0.807] | 3 |
| coordination posted | alone: spar_mixed | 6 | 209 | 205 | 0.990 | 0.603 | 0.603 | 0.603 | 0.600 [0.464, 0.726] | 0.611 [0.328, 0.815] | 0.634 [0.262, 0.957] | n/a | n/a | n/a |  |
| coordination posted | ours | 7 | 970 | 736 | 0.995 | 0.423 | 0.423 | 0.423 | 0.969 [0.958, 0.979] | 0.961 [0.946, 0.975] | 0.692 [0.638, 0.741] | 0.904 [0.877, 0.930] | 0.884 [0.857, 0.909] | 0.740 [0.697, 0.776] | 3 |
| collusion crossed | alone: spar_team | 3 | 450 | 67 | 1.000 | 0.327 | 0.327 | 0.320 | 0.910 [0.857, 0.959] | 0.899 [0.827, 0.958] | 0.822 [0.741, 0.904] | 0.827 [0.762, 0.886] | 0.822 [0.764, 0.872] | 0.586 [0.535, 0.640] | 3 |
| collusion crossed | alone: spar_mixed | 6 | 209 | 7 | 0.990 | 0.603 | 0.603 | 0.603 | 0.967 [0.920, 1.000] | 0.968 [0.923, 1.000] | 0.968 [0.923, 1.000] | n/a | n/a | n/a |  |
| collusion crossed | ours | 6 | 793 | 122 | 0.997 | 0.404 | 0.404 | 0.404 | 0.933 [0.896, 0.964] | 0.876 [0.819, 0.926] | 0.772 [0.703, 0.837] | 0.843 [0.780, 0.904] | 0.855 [0.790, 0.914] | 0.713 [0.621, 0.803] | 3 |

Provenance: `results/controls_nulls/author_rows.csv` (per-author B.II rows there too); scores in `scores/author/`. Author-probe accuracy is over all held-out B.I folds.

## 3. Lexical-only reader null: how much does the reasoning text alone give?

TF-IDF (word unigrams and bigrams, vocabulary, idf and C fitted on the training side) with L2 logistic regression on the think-window text only: every span between `<think>` and `</think>` in the reader's rendered input, all turns joined per agent. It runs on matrix_probes' own scheme runner, with the same folds and splits; the metadata rows that runner recomputes equal matrix_probes' on every row (VERIFICATION.md). Beside it: the TF-IDF of the whole rendered text and the full probe (think pair) from matrix_probes. Claude Haiku 5.5's reasoning is not exposed (its think windows are empty), and several external authors' windows are empty too: an agent with an empty think window is left out of training and gets no lexical score, so a held-out author made only of such agents gets no AUROC (n/a) and drops out of the mean over authors.

| Label | Scheme | Probe type | Unit | Agents | Positive | Think-window TF-IDF | Agents with a think-window score | Whole-text TF-IDF | Full probe (think pair) | Probe minus think-window |
|---|---|---|---|---|---|---|---|---|---|---|
| cheating | alone: controls | B.I | pooled | 600 | 343 | 0.907 [0.880, 0.932] | 475 | 0.990 [0.983, 0.996] | 0.960 [0.944, 0.973] | 0.052 |
| cheating | alone: controls | A | mean over authors | 600 | 343 | 0.848 [0.812, 0.883] | 475 | 0.953 [0.935, 0.972] | 0.938 [0.918, 0.957] | 0.090 |
| cheating | alone: controls | B.II | mean over authors | 600 | 343 | 0.797 [0.757, 0.837] | 475 | 0.971 [0.959, 0.983] | 0.913 [0.891, 0.934] | 0.116 |
| cheating | alone: board_seed | B.I | pooled | 134 | 56 | 0.905 [0.838, 0.957] | 134 | 0.905 [0.850, 0.951] | 0.880 [0.809, 0.935] | -0.025 |
| cheating | alone: board_seed | A | mean over authors | 134 | 56 | 0.905 [0.838, 0.957] | 134 | 0.905 [0.850, 0.951] | 0.880 [0.809, 0.935] | -0.025 |
| cheating | alone: solo_design | B.I | pooled | 177 | 84 | 0.917 [0.862, 0.959] | 175 | 0.922 [0.860, 0.968] | 0.914 [0.860, 0.960] | -0.003 |
| cheating | alone: solo_design | A | mean over authors | 171 | 79 | 0.848 [0.740, 0.928] | 169 | 0.888 [0.799, 0.947] | 0.816 [0.694, 0.905] | -0.033 |
| cheating | alone: solo_design | B.II | mean over authors | 171 | 79 | 0.675 [0.561, 0.788] | 169 | 0.710 [0.597, 0.830] | 0.630 [0.532, 0.723] | -0.044 |
| cheating | alone: spar_team | B.I | pooled | 450 | 109 | 0.957 [0.933, 0.978] | 300 | 0.982 [0.967, 0.991] | 0.927 [0.885, 0.962] | -0.030 |
| cheating | alone: spar_team | A | mean over authors | 450 | 109 | 0.916 [0.863, 0.960] | 300 | 0.938 [0.900, 0.968] | 0.926 [0.878, 0.963] | 0.010 |
| cheating | alone: spar_team | B.II | mean over authors | 450 | 109 | 0.852 [0.774, 0.914] | 300 | 0.918 [0.860, 0.955] | 0.751 [0.689, 0.811] | -0.101 |
| cheating | alone: spar_mixed | B.I | pooled | 209 | 12 | 0.790 [0.683, 0.891] | 179 | 0.937 [0.887, 0.976] | 0.734 [0.606, 0.869] | -0.057 |
| cheating | alone: spar_mixed | A | mean over authors | 162 | 12 | 0.805 [0.678, 0.921] | 162 | 0.832 [0.704, 0.929] | 0.715 [0.492, 0.882] | -0.090 |
| cheating | alone: spar_mixed | B.II | mean over authors | 162 | 12 | 0.386 [0.204, 0.562] | 162 | 0.330 [0.184, 0.489] | 0.708 [0.493, 0.859] | 0.322 |
| cheating | alone: spar_expl | B.I | pooled | 313 | 71 | 0.908 [0.832, 0.967] | 277 | 0.970 [0.939, 0.992] | 0.955 [0.918, 0.984] | 0.047 |
| cheating | alone: spar_expl | A | mean over authors | 313 | 71 | 0.915 [0.751, 0.971] | 277 | 0.877 [0.739, 0.983] | 0.910 [0.781, 0.968] | -0.005 |
| cheating | alone: spar_expl | B.II | mean over authors | 313 | 71 | 0.777 [0.580, 0.871] | 277 | 0.893 [0.851, 0.932] | 0.658 [0.456, 0.753] | -0.119 |
| cheating | alone: spar_solo | B.I | pooled | 90 | 47 | 0.931 [0.857, 0.989] | 60 | 0.985 [0.959, 1.000] | 0.962 [0.915, 0.995] | 0.031 |
| cheating | alone: spar_solo | A | mean over authors | 60 | 17 | 0.560 [0.315, 0.778] | 30 | 0.893 [0.789, 0.967] | 0.926 [0.847, 0.987] | 0.366 |
| cheating | alone: goodfire | B.I | pooled | 2833 | 2083 | 0.946 [0.937, 0.954] | 2833 | 0.927 [0.915, 0.938] | 0.921 [0.910, 0.931] | -0.025 |
| cheating | alone: goodfire | A | mean over authors | 2833 | 2083 | 0.930 [0.918, 0.940] | 2833 | 0.921 [0.909, 0.932] | 0.908 [0.896, 0.919] | -0.022 |
| cheating | alone: goodfire | B.II | mean over authors | 2833 | 2083 | 0.931 [0.921, 0.941] | 2833 | 0.916 [0.904, 0.927] | 0.822 [0.803, 0.840] | -0.109 |
| cheating | alone: cheatbench | B.I | pooled | 224 | 134 | 0.934 [0.774, 1.000] | 109 | 0.966 [0.944, 0.984] | 0.951 [0.919, 0.979] | 0.017 |
| cheating | alone: cheatbench | A | mean over authors | 224 | 134 | 0.733 [0.174, 1.000] | 109 | 0.834 [0.725, 0.922] | 0.956 [0.913, 0.987] | 0.223 |
| cheating | alone: evilgenie | B.I | pooled | 266 | 133 | 0.843 [0.776, 0.900] | 159 | 0.926 [0.895, 0.951] | 0.872 [0.831, 0.911] | 0.029 |
| cheating | alone: evilgenie | A | mean over authors | 252 | 126 | 0.758 [0.678, 0.795] | 149 | 0.900 [0.866, 0.930] | 0.827 [0.785, 0.881] | 0.068 |
| cheating | alone: evilgenie | B.II | mean over authors | 266 | 133 | 0.822 [0.714, 0.884] | 157 | 0.892 [0.854, 0.921] | 0.885 [0.845, 0.915] | 0.063 |
| cheating | ours | B.I | pooled | 1973 | 722 | 0.937 [0.923, 0.950] | 1600 | 0.981 [0.975, 0.986] | 0.950 [0.940, 0.960] | 0.013 |
| cheating | ours | A | mean over authors | 1948 | 717 | 0.898 [0.873, 0.924] | 1577 | 0.961 [0.948, 0.973] | 0.944 [0.928, 0.958] | 0.046 |
| cheating | ours | B.II | mean over authors | 1948 | 717 | 0.800 [0.769, 0.828] | 1577 | 0.945 [0.932, 0.958] | 0.860 [0.838, 0.880] | 0.060 |
| cheating | external | B.I | pooled | 3323 | 2350 | 0.945 [0.936, 0.953] | 3101 | 0.935 [0.926, 0.942] | 0.921 [0.911, 0.930] | -0.024 |
| cheating | external | A | mean over authors | 3317 | 2347 | 0.809 [0.736, 0.892] | 3099 | 0.898 [0.866, 0.924] | 0.873 [0.846, 0.905] | 0.064 |
| cheating | external | B.II | mean over authors | 3323 | 2350 | 0.841 [0.806, 0.884] | 3099 | 0.869 [0.841, 0.894] | 0.784 [0.752, 0.816] | -0.057 |
| cheating | all | B.I | pooled | 5296 | 3072 | 0.946 [0.940, 0.952] | 4701 | 0.964 [0.959, 0.969] | 0.920 [0.912, 0.927] | -0.026 |
| cheating | all | A | mean over authors | 5271 | 3069 | 0.832 [0.778, 0.886] | 4682 | 0.914 [0.890, 0.935] | 0.891 [0.871, 0.914] | 0.059 |
| cheating | all | B.II | mean over authors | 5277 | 3072 | 0.766 [0.706, 0.801] | 4682 | 0.895 [0.872, 0.915] | 0.795 [0.756, 0.822] | 0.029 |
| cheating | leave out: controls | B.I | pooled | 600 | 343 | 0.696 [0.645, 0.741] | 475 | 0.802 [0.766, 0.836] | 0.684 [0.637, 0.727] | -0.012 |
| cheating | leave out: controls | A | mean over authors | 360 | 201 | 0.638 [0.564, 0.705] | 235 | 0.735 [0.683, 0.788] | 0.772 [0.720, 0.819] | 0.134 |
| cheating | leave out: controls | B.II | mean over authors | 600 | 343 | 0.612 [0.560, 0.660] | 475 | 0.758 [0.714, 0.796] | 0.636 [0.592, 0.679] | 0.024 |
| cheating | leave out: board_seed | B.I | pooled | 134 | 56 | 0.785 [0.701, 0.858] | 134 | 0.856 [0.771, 0.921] | 0.465 [0.361, 0.570] | -0.319 |
| cheating | leave out: board_seed | A | mean over authors | 134 | 56 | 0.786 [0.694, 0.868] | 134 | 0.826 [0.725, 0.909] | 0.762 [0.666, 0.851] | -0.024 |
| cheating | leave out: board_seed | B.II | mean over authors | 134 | 56 | 0.633 [0.530, 0.732] | 134 | 0.816 [0.733, 0.881] | 0.754 [0.641, 0.848] | 0.120 |
| cheating | leave out: solo_design | B.I | pooled | 177 | 84 | 0.876 [0.808, 0.935] | 175 | 0.899 [0.833, 0.953] | 0.821 [0.747, 0.885] | -0.055 |
| cheating | leave out: solo_design | A | mean over authors | 171 | 79 | 0.722 [0.583, 0.843] | 169 | 0.824 [0.730, 0.909] | 0.867 [0.773, 0.929] | 0.145 |
| cheating | leave out: solo_design | B.II | mean over authors | 171 | 79 | 0.804 [0.727, 0.890] | 169 | 0.854 [0.772, 0.927] | 0.623 [0.501, 0.736] | -0.180 |
| cheating | leave out: spar_team | B.I | pooled | 450 | 109 | 0.696 [0.601, 0.799] | 300 | 0.873 [0.819, 0.928] | 0.792 [0.727, 0.855] | 0.096 |
| cheating | leave out: spar_team | A | mean over authors | 450 | 109 | 0.590 [0.490, 0.673] | 300 | 0.785 [0.725, 0.854] | 0.717 [0.649, 0.820] | 0.127 |
| cheating | leave out: spar_team | B.II | mean over authors | 450 | 109 | 0.673 [0.583, 0.757] | 300 | 0.836 [0.775, 0.889] | 0.691 [0.615, 0.753] | 0.018 |
| cheating | leave out: spar_mixed | B.I | pooled | 209 | 12 | 0.936 [0.896, 0.972] | 179 | 0.984 [0.966, 0.996] | 0.915 [0.855, 0.959] | -0.021 |
| cheating | leave out: spar_mixed | A | mean over authors | 162 | 12 | 0.916 [0.825, 0.969] | 162 | 0.950 [0.846, 0.995] | 0.789 [0.625, 0.894] | -0.126 |
| cheating | leave out: spar_mixed | B.II | mean over authors | 162 | 12 | 0.788 [0.634, 0.922] | 162 | 0.839 [0.707, 0.947] | 0.638 [0.492, 0.771] | -0.151 |
| cheating | leave out: spar_expl | B.I | pooled | 313 | 71 | 0.906 [0.861, 0.950] | 277 | 0.987 [0.976, 0.995] | 0.906 [0.855, 0.951] | 0.000 |
| cheating | leave out: spar_expl | A | mean over authors | 313 | 71 | 0.928 [0.825, 0.962] | 277 | 0.973 [0.943, 0.998] | 0.949 [0.912, 0.981] | 0.022 |
| cheating | leave out: spar_expl | B.II | mean over authors | 313 | 71 | 0.696 [0.529, 0.798] | 277 | 0.953 [0.907, 0.983] | 0.651 [0.546, 0.751] | -0.045 |
| cheating | leave out: spar_solo | B.I | pooled | 90 | 47 | 0.941 [0.873, 0.991] | 60 | 0.943 [0.900, 0.976] | 0.852 [0.769, 0.918] | -0.089 |
| cheating | leave out: spar_solo | A | mean over authors | 60 | 17 | 0.785 [0.561, 0.963] | 30 | 0.971 [0.923, 1.000] | 0.923 [0.845, 0.983] | 0.138 |
| cheating | leave out: spar_solo | B.II | mean over authors | 60 | 17 | 0.847 [0.690, 0.972] | 30 | 0.974 [0.929, 1.000] | 0.911 [0.806, 0.989] | 0.065 |
| cheating | leave out: goodfire | B.I | pooled | 2833 | 2083 | 0.640 [0.613, 0.664] | 2833 | 0.678 [0.655, 0.703] | 0.648 [0.624, 0.670] | 0.008 |
| cheating | leave out: goodfire | A | mean over authors | 935 | 733 | 0.579 [0.538, 0.620] | 935 | 0.516 [0.468, 0.561] | 0.348 [0.308, 0.395] | -0.231 |
| cheating | leave out: goodfire | B.II | mean over authors | 2833 | 2083 | 0.684 [0.658, 0.708] | 2833 | 0.732 [0.708, 0.756] | 0.600 [0.574, 0.625] | -0.084 |
| cheating | leave out: cheatbench | B.I | pooled | 224 | 134 | 0.522 [0.376, 0.651] | 109 | 0.582 [0.505, 0.659] | 0.805 [0.751, 0.858] | 0.283 |
| cheating | leave out: cheatbench | B.II | mean over authors | 224 | 134 | 0.522 [0.376, 0.651] | 109 | 0.729 [0.630, 0.830] | 0.664 [0.548, 0.773] | 0.142 |
| cheating | leave out: evilgenie | B.I | pooled | 266 | 133 | 0.557 [0.475, 0.634] | 159 | 0.554 [0.498, 0.610] | 0.652 [0.607, 0.700] | 0.095 |
| cheating | leave out: evilgenie | A | mean over authors | 50 | 25 | 0.875 [0.750, 1.000] | 8 | 0.631 [0.284, 0.820] | 0.491 [0.265, 0.768] | -0.384 |
| cheating | leave out: evilgenie | B.II | mean over authors | 266 | 133 | 0.606 [0.480, 0.718] | 157 | 0.651 [0.541, 0.731] | 0.704 [0.593, 0.774] | 0.099 |
| cheating | ours to external | B.I | pooled | 3323 | 2350 | 0.701 [0.681, 0.721] | 3101 | 0.728 [0.709, 0.746] | 0.630 [0.609, 0.649] | -0.071 |
| cheating | ours to external | B.II | mean over authors | 3323 | 2350 | 0.487 [0.414, 0.577] | 3099 | 0.570 [0.519, 0.619] | 0.541 [0.461, 0.603] | 0.055 |
| cheating | external to ours | B.I | pooled | 1973 | 722 | 0.601 [0.565, 0.632] | 1600 | 0.600 [0.565, 0.637] | 0.397 [0.366, 0.428] | -0.204 |
| cheating | external to ours | B.II | mean over authors | 1948 | 717 | 0.542 [0.501, 0.580] | 1577 | 0.503 [0.467, 0.539] | 0.443 [0.406, 0.480] | -0.099 |
| coordination posted | alone: board_seed | B.I | pooled | 134 | 87 | 0.916 [0.843, 0.969] | 134 | 0.895 [0.804, 0.963] | 0.894 [0.784, 0.974] | -0.022 |
| coordination posted | alone: board_seed | A | mean over authors | 134 | 87 | 0.916 [0.843, 0.969] | 134 | 0.895 [0.804, 0.963] | 0.894 [0.784, 0.974] | -0.022 |
| coordination posted | alone: solo_design | B.I | pooled | 177 | 38 | 0.753 [0.622, 0.859] | 175 | 0.774 [0.629, 0.886] | 0.747 [0.579, 0.878] | -0.005 |
| coordination posted | alone: solo_design | A | mean over authors | 171 | 37 | 0.780 [0.611, 0.904] | 169 | 0.798 [0.618, 0.926] | 0.814 [0.686, 0.910] | 0.033 |
| coordination posted | alone: solo_design | B.II | mean over authors | 171 | 37 | 0.542 [0.369, 0.715] | 169 | 0.516 [0.345, 0.710] | 0.715 [0.597, 0.803] | 0.173 |
| coordination posted | alone: spar_team | B.I | pooled | 450 | 406 | 0.806 [0.619, 0.946] | 300 | 0.889 [0.829, 0.937] | 0.909 [0.843, 0.963] | 0.103 |
| coordination posted | alone: spar_team | A | mean over authors | 450 | 406 | 0.790 [0.682, 0.905] | 300 | 0.700 [0.600, 0.838] | 0.752 [0.649, 0.841] | -0.038 |
| coordination posted | alone: spar_team | B.II | mean over authors | 450 | 406 | 0.391 [0.252, 0.537] | 300 | 0.680 [0.553, 0.808] | 0.690 [0.633, 0.755] | 0.299 |
| coordination posted | alone: spar_mixed | B.I | pooled | 209 | 205 | 0.573 [0.344, 0.989] | 179 | 0.726 [0.177, 0.973] | 0.600 [0.464, 0.726] | 0.027 |
| coordination posted | ours | B.I | pooled | 970 | 736 | 0.949 [0.929, 0.968] | 788 | 0.957 [0.942, 0.971] | 0.969 [0.958, 0.979] | 0.021 |
| coordination posted | ours | A | mean over authors | 945 | 719 | 0.944 [0.916, 0.967] | 765 | 0.897 [0.866, 0.927] | 0.944 [0.922, 0.962] | 0.000 |
| coordination posted | ours | B.II | mean over authors | 945 | 719 | 0.852 [0.804, 0.896] | 765 | 0.863 [0.824, 0.896] | 0.904 [0.877, 0.930] | 0.052 |
| coordination posted | leave out: board_seed | B.I | pooled | 134 | 87 | 0.918 [0.850, 0.966] | 134 | 0.889 [0.809, 0.944] | 0.942 [0.885, 0.980] | 0.024 |
| coordination posted | leave out: board_seed | A | mean over authors | 134 | 87 | 0.886 [0.812, 0.938] | 134 | 0.897 [0.816, 0.953] | 0.908 [0.846, 0.955] | 0.022 |
| coordination posted | leave out: board_seed | B.II | mean over authors | 134 | 87 | 0.850 [0.763, 0.920] | 134 | 0.733 [0.628, 0.825] | 0.866 [0.791, 0.923] | 0.016 |
| coordination posted | leave out: solo_design | B.I | pooled | 177 | 38 | 0.403 [0.276, 0.526] | 175 | 0.496 [0.349, 0.644] | 0.758 [0.647, 0.853] | 0.355 |
| coordination posted | leave out: solo_design | A | mean over authors | 171 | 37 | 0.368 [0.261, 0.483] | 169 | 0.355 [0.227, 0.473] | 0.705 [0.550, 0.825] | 0.337 |
| coordination posted | leave out: solo_design | B.II | mean over authors | 171 | 37 | 0.681 [0.554, 0.777] | 169 | 0.782 [0.649, 0.874] | 0.786 [0.651, 0.877] | 0.105 |
| coordination posted | leave out: spar_team | B.I | pooled | 450 | 406 | 0.620 [0.417, 0.814] | 300 | 0.654 [0.559, 0.744] | 0.863 [0.812, 0.907] | 0.242 |
| coordination posted | leave out: spar_team | A | mean over authors | 300 | 291 | 0.674 [0.507, 0.807] | 300 | 0.702 [0.600, 0.819] | 0.873 [0.796, 0.953] | 0.199 |
| coordination posted | leave out: spar_team | B.II | mean over authors | 450 | 406 | 0.637 [0.457, 0.828] | 300 | 0.613 [0.497, 0.741] | 0.845 [0.795, 0.893] | 0.208 |
| coordination posted | leave out: spar_mixed | B.I | pooled | 209 | 205 | 0.867 [0.736, 0.991] | 179 | 0.470 [0.077, 0.966] | 0.854 [0.763, 0.924] | -0.013 |
| collusion crossed | alone: board_seed | B.I | pooled | 134 | 48 | 0.948 [0.902, 0.982] | 134 | 0.950 [0.898, 0.983] | 0.901 [0.814, 0.964] | -0.047 |
| collusion crossed | alone: board_seed | A | mean over authors | 134 | 48 | 0.948 [0.902, 0.982] | 134 | 0.950 [0.898, 0.983] | 0.901 [0.814, 0.964] | -0.047 |
| collusion crossed | alone: spar_team | B.I | pooled | 450 | 67 | 0.939 [0.871, 0.982] | 300 | 0.980 [0.961, 0.994] | 0.910 [0.857, 0.959] | -0.029 |
| collusion crossed | alone: spar_team | A | mean over authors | 450 | 67 | 0.882 [0.794, 0.946] | 300 | 0.942 [0.892, 0.979] | 0.842 [0.710, 0.960] | -0.040 |
| collusion crossed | alone: spar_team | B.II | mean over authors | 450 | 67 | 0.831 [0.737, 0.909] | 300 | 0.847 [0.768, 0.930] | 0.827 [0.762, 0.886] | -0.004 |
| collusion crossed | alone: spar_mixed | B.I | pooled | 209 | 7 | 0.861 [0.753, 0.959] | 179 | 0.950 [0.873, 0.994] | 0.967 [0.920, 1.000] | 0.105 |
| collusion crossed | alone: spar_mixed | A | mean over authors | 36 | 7 | 0.355 [0.122, 0.618] | 36 | 0.281 [0.010, 0.678] | 0.818 [0.696, 0.961] | 0.463 |
| collusion crossed | ours | B.I | pooled | 793 | 122 | 0.943 [0.896, 0.976] | 613 | 0.984 [0.972, 0.993] | 0.933 [0.896, 0.964] | -0.010 |
| collusion crossed | ours | A | mean over authors | 774 | 122 | 0.961 [0.929, 0.983] | 596 | 0.966 [0.936, 0.987] | 0.809 [0.704, 0.895] | -0.152 |
| collusion crossed | ours | B.II | mean over authors | 774 | 122 | 0.663 [0.563, 0.753] | 596 | 0.856 [0.818, 0.890] | 0.843 [0.780, 0.904] | 0.180 |
| collusion crossed | leave out: board_seed | B.I | pooled | 134 | 48 | 0.905 [0.835, 0.957] | 134 | 0.955 [0.912, 0.987] | 0.888 [0.810, 0.945] | -0.017 |
| collusion crossed | leave out: board_seed | A | mean over authors | 134 | 48 | 0.900 [0.820, 0.959] | 134 | 0.946 [0.891, 0.985] | 0.857 [0.769, 0.929] | -0.043 |
| collusion crossed | leave out: board_seed | B.II | mean over authors | 134 | 48 | 0.468 [0.342, 0.594] | 134 | 0.861 [0.782, 0.923] | 0.762 [0.665, 0.854] | 0.293 |
| collusion crossed | leave out: spar_team | B.I | pooled | 450 | 67 | 0.554 [0.396, 0.709] | 300 | 0.657 [0.517, 0.779] | 0.814 [0.712, 0.899] | 0.260 |
| collusion crossed | leave out: spar_team | A | mean over authors | 150 | 37 | 0.680 [0.482, 0.871] | 150 | 0.774 [0.584, 0.937] | 0.845 [0.727, 0.932] | 0.165 |
| collusion crossed | leave out: spar_team | B.II | mean over authors | 300 | 30 | 0.349 [0.231, 0.463] | 150 | 0.346 [0.159, 0.535] | 0.707 [0.620, 0.796] | 0.358 |
| collusion crossed | leave out: spar_mixed | B.I | pooled | 209 | 7 | 0.835 [0.663, 0.970] | 179 | 0.932 [0.872, 0.972] | 0.907 [0.812, 0.980] | 0.072 |
| collusion crossed | leave out: spar_mixed | A | mean over authors | 36 | 7 | 0.852 [0.657, 0.985] | 36 | 0.818 [0.664, 0.945] | 0.458 [0.086, 0.712] | -0.394 |
| collusion crossed | leave out: spar_mixed | B.II | mean over authors | 36 | 7 | 0.522 [0.339, 0.857] | 36 | 0.631 [0.411, 1.000] | 0.695 [0.487, 0.879] | 0.172 |

Provenance: `results/controls_nulls/lexical_rows.csv` (per-author A and B.II rows there too), scores in `scores/lexical/`; whole-text TF-IDF and full probe from matrix_probes `cells.csv`.

## 4. Does every main table carry a label-permutation null?

No. The audit read every main table of the four RESULTS.md files and matched each row to its CSV. The rows without a null are the baselines (TF-IDF, metadata), the team aggregates, the transfer cells, and most of the per-step and attention-monitor rows.

| Results dir | Rows with a null | Rows without | Rows without, null computed here |
|---|---|---|---|
| attn_probe | 50 | 796 | 0 |
| external_probes | 165 | 402 | 27 |
| matrix_probes | 997 | 1511 | 800 |
| team_probes | 496 | 366 | 150 |

Provenance: `results/controls_nulls/null_audit.csv`, one line per table pattern, counts from the CSVs.

**Refit nulls (the probe's own protocol).** These are matrix_probes' TF-IDF and metadata B.I pooled rows of every cross-validated scheme. The label is permuted within (source, author, arm) strata, as for the probe. There are 50 draws, and the baseline is refitted on each draw on the same folds. TF-IDF uses C fixed at the median of the per-fold C chosen on the real labels. Every real-label AUROC was recomputed and equals cells.csv.

| Label | Scheme | Baseline | Agents | Positive | AUROC | Null mean / 95th | p (50 draws) | Beats the null (p <= 0.05)? |
|---|---|---|---|---|---|---|---|---|
| cheating | alone: controls | metadata | 600 | 343 | 0.690 | 0.584 / 0.612 | 0.020 | yes |
| cheating | alone: controls | TF-IDF rendered text | 600 | 343 | 0.990 | 0.547 / 0.600 | 0.020 | yes |
| cheating | alone: board_seed | metadata | 134 | 56 | 0.826 | 0.654 / 0.702 | 0.020 | yes |
| cheating | alone: board_seed | TF-IDF rendered text | 134 | 56 | 0.905 | 0.591 / 0.667 | 0.020 | yes |
| cheating | alone: solo_design | metadata | 177 | 84 | 0.866 | 0.742 / 0.784 | 0.020 | yes |
| cheating | alone: solo_design | TF-IDF rendered text | 177 | 84 | 0.922 | 0.720 / 0.771 | 0.020 | yes |
| cheating | alone: spar_team | metadata | 450 | 109 | 0.934 | 0.737 / 0.752 | 0.020 | yes |
| cheating | alone: spar_team | TF-IDF rendered text | 450 | 109 | 0.982 | 0.693 / 0.742 | 0.020 | yes |
| cheating | alone: spar_mixed | metadata | 209 | 12 | 0.837 | 0.815 / 0.874 | 0.353 | no |
| cheating | alone: spar_mixed | TF-IDF rendered text | 209 | 12 | 0.937 | 0.787 / 0.857 | 0.020 | yes |
| cheating | alone: spar_expl | metadata | 313 | 71 | 0.936 | 0.843 / 0.856 | 0.020 | yes |
| cheating | alone: spar_expl | TF-IDF rendered text | 313 | 71 | 0.970 | 0.744 / 0.795 | 0.020 | yes |
| cheating | alone: spar_solo | metadata | 90 | 47 | 0.926 | 0.870 / 0.910 | 0.020 | yes |
| cheating | alone: spar_solo | TF-IDF rendered text | 90 | 47 | 0.985 | 0.818 / 0.889 | 0.020 | yes |
| cheating | alone: goodfire | metadata | 2833 | 2083 | 0.809 | 0.709 / 0.713 | 0.020 | yes |
| cheating | alone: goodfire | TF-IDF rendered text | 2833 | 2083 | 0.927 | 0.657 / 0.673 | 0.020 | yes |
| cheating | alone: cheatbench | metadata | 224 | 134 | 0.944 | 0.841 / 0.870 | 0.020 | yes |
| cheating | alone: cheatbench | TF-IDF rendered text | 224 | 134 | 0.966 | 0.856 / 0.892 | 0.020 | yes |
| cheating | alone: evilgenie | metadata | 266 | 133 | 0.640 | 0.400 / 0.472 | 0.020 | yes |
| cheating | alone: evilgenie | TF-IDF rendered text | 266 | 133 | 0.926 | 0.451 / 0.532 | 0.020 | yes |
| cheating | ours | metadata | 1973 | 722 | 0.843 | 0.785 / 0.791 | 0.020 | yes |
| cheating | ours | TF-IDF rendered text | 1973 | 722 | 0.981 | 0.754 / 0.766 | 0.020 | yes |
| cheating | external | metadata | 3323 | 2350 | 0.814 | 0.733 / 0.737 | 0.020 | yes |
| cheating | external | TF-IDF rendered text | 3323 | 2350 | 0.935 | 0.691 / 0.708 | 0.020 | yes |
| cheating | all | metadata | 5296 | 3072 | 0.850 | 0.792 / 0.794 | 0.020 | yes |
| cheating | all | TF-IDF rendered text | 5296 | 3072 | 0.964 | 0.752 / 0.761 | 0.020 | yes |
| coordination posted | alone: board_seed | metadata | 134 | 87 | 0.929 | 0.786 / 0.820 | 0.020 | yes |
| coordination posted | alone: board_seed | TF-IDF rendered text | 134 | 87 | 0.895 | 0.704 / 0.771 | 0.020 | yes |
| coordination posted | alone: solo_design | metadata | 177 | 38 | 0.753 | 0.527 / 0.607 | 0.020 | yes |
| coordination posted | alone: solo_design | TF-IDF rendered text | 177 | 38 | 0.774 | 0.468 / 0.603 | 0.020 | yes |
| coordination posted | alone: spar_team | metadata | 450 | 406 | 0.794 | 0.759 / 0.794 | 0.078 | no |
| coordination posted | alone: spar_team | TF-IDF rendered text | 450 | 406 | 0.889 | 0.708 / 0.765 | 0.020 | yes |
| coordination posted | alone: spar_mixed | metadata | 209 | 205 | 0.627 | 0.534 / 0.749 | 0.320 | no |
| coordination posted | alone: spar_mixed | TF-IDF rendered text | 209 | 205 | 0.726 | 0.550 / 0.748 | 0.120 | no |
| coordination posted | ours | metadata | 970 | 736 | 0.924 | 0.901 / 0.907 | 0.020 | yes |
| coordination posted | ours | TF-IDF rendered text | 970 | 736 | 0.957 | 0.867 / 0.882 | 0.020 | yes |
| collusion crossed | alone: board_seed | metadata | 134 | 48 | 0.852 | 0.778 / 0.814 | 0.020 | yes |
| collusion crossed | alone: board_seed | TF-IDF rendered text | 134 | 48 | 0.950 | 0.653 / 0.746 | 0.020 | yes |
| collusion crossed | alone: spar_team | metadata | 450 | 67 | 0.920 | 0.719 / 0.745 | 0.020 | yes |
| collusion crossed | alone: spar_team | TF-IDF rendered text | 450 | 67 | 0.980 | 0.656 / 0.714 | 0.020 | yes |
| collusion crossed | alone: spar_mixed | metadata | 209 | 7 | 0.960 | 0.925 / 0.963 | 0.078 | no |
| collusion crossed | alone: spar_mixed | TF-IDF rendered text | 209 | 7 | 0.950 | 0.906 / 0.936 | 0.039 | yes |
| collusion crossed | ours | metadata | 793 | 122 | 0.908 | 0.817 / 0.833 | 0.020 | yes |
| collusion crossed | ours | TF-IDF rendered text | 793 | 122 | 0.984 | 0.753 / 0.788 | 0.020 | yes |

Provenance: `results/controls_nulls/nulls_rows.csv` (part refit).

**Fixed-score nulls.** These cover every other row without a null whose held-out scores are saved: matrix_probes TF-IDF and metadata rows (types A and B.II, leave-one-out and transfer cells), the team_probes agent and team baselines and team aggregates, and the external_probes trajectory baselines. The saved out-of-fold scores are kept, and the labels are permuted 50 times (for a mean-over-authors row, the per-author draws are averaged). This tests that the held-out AUROC beats chance. It does not refit, so it is weaker than the refit null. A row is used only when the AUROC recomputed from its scores equals the table's within 0.001 (the matrix_probes score files round scores to 6 significant digits). The team aggregates are rebuilt per run, as the max or mean of the agent collusion probe's held-out scores, and the team labels are permuted.

| Results dir | Table | Row kind | Rows | Scores reproduce the table | Beats the null (p <= 0.05) | Does not | Largest null 95th |
|---|---|---|---|---|---|---|---|
| matrix_probes | cells.csv | TF-IDF rendered text | 378 | 378 | 311 | 67 | 1.000 |
| matrix_probes | cells.csv | metadata | 378 | 378 | 232 | 146 | 1.000 |
| team_probes | agent_probes.csv | metadata | 10 | 10 | 10 | 0 | 0.823 |
| team_probes | agent_probes.csv | TF-IDF | 20 | 20 | 17 | 3 | 0.829 |
| team_probes | team_probe.csv | team aggregate of agent probe | 6 | 6 | 6 | 0 | 0.664 |
| team_probes | team_probe.csv | metadata | 2 | 2 | 2 | 0 | 0.639 |
| team_probes | team_probe.csv | TF-IDF | 4 | 4 | 4 | 0 | 0.655 |
| team_probes | sets/solo_design/agent_probes.csv | metadata | 18 | 18 | 14 | 4 | 0.691 |
| team_probes | sets/solo_design/agent_probes.csv | TF-IDF | 36 | 36 | 26 | 10 | 0.725 |
| team_probes | sets/spar_mixed_41/agent_probes.csv | metadata | 4 | 4 | 4 | 0 | 0.719 |
| team_probes | sets/spar_mixed_41/agent_probes.csv | TF-IDF | 8 | 8 | 8 | 0 | 0.700 |
| team_probes | sets/spar_mixed_41/team_probe.csv | team aggregate of agent probe | 6 | 6 | 4 | 2 | 0.708 |
| team_probes | sets/spar_mixed_41/team_probe.csv | metadata | 2 | 2 | 2 | 0 | 0.696 |
| team_probes | sets/spar_mixed_41/team_probe.csv | TF-IDF | 4 | 4 | 3 | 1 | 0.709 |
| team_probes | sets/spar_screen_team/agent_probes.csv | metadata | 6 | 6 | 6 | 0 | 0.579 |
| team_probes | sets/spar_screen_team/agent_probes.csv | TF-IDF | 12 | 12 | 12 | 0 | 0.597 |
| team_probes | sets/spar_screen_team/team_probe.csv | team aggregate of agent probe | 6 | 6 | 6 | 0 | 0.600 |
| team_probes | sets/spar_screen_team/team_probe.csv | metadata | 2 | 2 | 2 | 0 | 0.593 |
| team_probes | sets/spar_screen_team/team_probe.csv | TF-IDF | 4 | 4 | 4 | 0 | 0.603 |
| external_probes | traj.json | metadata | 12 | 12 | 10 | 2 | 0.563 |
| external_probes | traj.json | TF-IDF | 14 | 14 | 9 | 5 | 0.580 |
| external_probes | traj.json | linear probe | 1 | 1 | 1 | 0 | 0.521 |

Rows whose saved scores did not reproduce the table's AUROC (no null attached): 0.

Rows that do not beat their null (p > 0.05):

| Results dir | Table | Row | Feature | Unit | Agents | AUROC | Null 95th | p |
|---|---|---|---|---|---|---|---|---|
| matrix_probes | cells.csv | cheat alone: controls B.II | meta | Claude Haiku 5.5 | 120 | 0.589 | 0.571 | 0.059 |
| matrix_probes | cells.csv | cheat alone: controls B.II | meta | DeepSeek V4.1 Flash | 120 | 0.557 | 0.601 | 0.176 |
| matrix_probes | cells.csv | cheat alone: controls B.II | meta | GPT-5 mini | 120 | 0.429 | 0.581 | 0.941 |
| matrix_probes | cells.csv | cheat alone: controls B.II | meta | Qwen3.8 27B | 120 | 0.497 | 0.573 | 0.431 |
| matrix_probes | cells.csv | cheat alone: controls B.II | meta | mean over authors | 600 | 0.535 | 0.534 | 0.078 |
| matrix_probes | cells.csv | cheat alone: solo_design B.II | tfidf | GPT-6 Luna | 81 | 0.582 | 0.667 | 0.314 |
| matrix_probes | cells.csv | cheat alone: solo_design B.II | meta | GPT-6 Luna | 81 | 0.554 | 0.647 | 0.235 |
| matrix_probes | cells.csv | cheat alone: spar_mixed A | tfidf | DeepSeek V4.1 Flash | 36 | 0.683 | 0.672 | 0.059 |
| matrix_probes | cells.csv | cheat alone: spar_mixed A | meta | DeepSeek V4.1 Flash | 36 | 0.560 | 0.670 | 0.275 |
| matrix_probes | cells.csv | cheat alone: spar_mixed A | meta | GPT-6 Luna | 126 | 0.623 | 0.783 | 0.255 |
| matrix_probes | cells.csv | cheat alone: spar_mixed A | meta | mean over authors | 162 | 0.591 | 0.644 | 0.118 |
| matrix_probes | cells.csv | cheat alone: spar_mixed B.II | tfidf | DeepSeek V4.1 Flash | 36 | 0.420 | 0.636 | 0.745 |
| matrix_probes | cells.csv | cheat alone: spar_mixed B.II | meta | DeepSeek V4.1 Flash | 36 | 0.449 | 0.644 | 0.667 |
| matrix_probes | cells.csv | cheat alone: spar_mixed B.II | tfidf | GPT-6 Luna | 126 | 0.241 | 0.773 | 0.980 |
| matrix_probes | cells.csv | cheat alone: spar_mixed B.II | meta | GPT-6 Luna | 126 | 0.545 | 0.748 | 0.490 |
| matrix_probes | cells.csv | cheat alone: spar_mixed B.II | tfidf | mean over authors | 162 | 0.330 | 0.637 | 0.941 |
| matrix_probes | cells.csv | cheat alone: spar_mixed B.II | meta | mean over authors | 162 | 0.497 | 0.670 | 0.667 |
| matrix_probes | cells.csv | cheat alone: spar_expl A | tfidf | Claude Haiku 5.5 | 36 | 0.729 | 0.685 | 0.059 |
| matrix_probes | cells.csv | cheat alone: spar_solo A | meta | GPT-6 Luna | 30 | 0.742 | 0.709 | 0.059 |
| matrix_probes | cells.csv | cheat alone: spar_solo B.II | meta | Claude Haiku 5.5 | 30 | 0.646 | 0.731 | 0.157 |
| matrix_probes | cells.csv | cheat alone: spar_solo B.II | meta | GPT-6 Luna | 30 | 0.603 | 0.700 | 0.157 |
| matrix_probes | cells.csv | cheat alone: spar_solo B.II | meta | mean over authors | 60 | 0.624 | 0.621 | 0.078 |
| matrix_probes | cells.csv | cheat ours B.II | meta | GPT-5 mini | 120 | 0.440 | 0.606 | 0.882 |
| matrix_probes | cells.csv | cheat ours B.II | meta | Qwen3.8 27B | 120 | 0.491 | 0.570 | 0.647 |
| matrix_probes | cells.csv | cheat alone: cheatbench A | tfidf | GPT-5.3 Codex | 35 | 0.752 | 0.734 | 0.059 |
| matrix_probes | cells.csv | cheat alone: cheatbench B.II | meta | Claude Opus 4.6 | 111 | 0.575 | 0.703 | 0.373 |
| matrix_probes | cells.csv | cheat alone: cheatbench B.II | tfidf | GPT-5.3 Codex | 35 | 0.564 | 0.669 | 0.333 |
| matrix_probes | cells.csv | cheat alone: cheatbench B.II | meta | GPT-5.3 Codex | 35 | 0.611 | 0.646 | 0.176 |
| matrix_probes | cells.csv | cheat alone: evilgenie A | meta | Claude Fable 5 | 10 | 0.360 | 0.840 | 0.902 |
| matrix_probes | cells.csv | cheat alone: evilgenie A | meta | Gemini 3.1 Pro | 34 | 0.578 | 0.639 | 0.137 |
| matrix_probes | cells.csv | cheat alone: evilgenie A | meta | Gemini 3.5 Flash | 56 | 0.574 | 0.601 | 0.157 |
| matrix_probes | cells.csv | cheat alone: evilgenie A | meta | Grok 4.5 | 40 | 0.672 | 0.670 | 0.059 |
| matrix_probes | cells.csv | cheat alone: evilgenie A | meta | Kimi K2.7 Code | 18 | 0.481 | 0.725 | 0.627 |
| matrix_probes | cells.csv | cheat alone: evilgenie B.II | meta | Claude Fable 5 | 10 | 0.320 | 0.782 | 0.804 |
| matrix_probes | cells.csv | cheat alone: evilgenie B.II | tfidf | Claude Opus 4.8 | 6 | 1.000 | 0.889 | 0.059 |
| matrix_probes | cells.csv | cheat alone: evilgenie B.II | meta | Claude Opus 4.8 | 6 | 0.556 | 1.000 | 0.529 |
| matrix_probes | cells.csv | cheat alone: evilgenie B.II | tfidf | Gemini 3.1 Pro | 34 | 0.540 | 0.627 | 0.294 |
| matrix_probes | cells.csv | cheat alone: evilgenie B.II | meta | Gemini 3.1 Pro | 34 | 0.478 | 0.635 | 0.588 |
| matrix_probes | cells.csv | cheat alone: evilgenie B.II | meta | Gemini 3.5 Flash | 56 | 0.589 | 0.634 | 0.137 |
| matrix_probes | cells.csv | cheat alone: evilgenie B.II | meta | Kimi K2.7 Code | 18 | 0.457 | 0.748 | 0.647 |
| matrix_probes | cells.csv | cheat alone: evilgenie B.II | meta | Kimi K3 | 8 | 0.500 | 0.909 | 0.490 |
| matrix_probes | cells.csv | cheat external A | meta | Claude Fable 5 | 10 | 0.360 | 0.822 | 0.863 |
| matrix_probes | cells.csv | cheat external A | meta | Gemini 3.1 Pro | 34 | 0.578 | 0.621 | 0.137 |
| matrix_probes | cells.csv | cheat external A | meta | Gemini 3.5 Flash | 56 | 0.574 | 0.580 | 0.137 |
| matrix_probes | cells.csv | cheat external A | meta | Kimi K2.7 Code | 18 | 0.481 | 0.693 | 0.608 |
| matrix_probes | cells.csv | cheat external B.II | meta | Claude Fable 5 | 10 | 0.600 | 0.862 | 0.255 |
| matrix_probes | cells.csv | cheat external B.II | tfidf | Claude Opus 4.8 | 6 | 1.000 | 0.950 | 0.078 |
| matrix_probes | cells.csv | cheat external B.II | meta | Claude Opus 4.8 | 6 | 1.000 | 0.839 | 0.059 |
| matrix_probes | cells.csv | cheat external B.II | meta | GPT-5.3 Codex | 77 | 0.557 | 0.640 | 0.294 |
| matrix_probes | cells.csv | cheat external B.II | meta | GPT-5.6 Luna | 28 | 0.617 | 0.681 | 0.235 |
| matrix_probes | cells.csv | cheat external B.II | meta | GPT-5.6 Sol | 24 | 0.514 | 0.685 | 0.510 |
| matrix_probes | cells.csv | cheat external B.II | tfidf | Gemini 3.1 Pro | 34 | 0.429 | 0.689 | 0.824 |
| matrix_probes | cells.csv | cheat external B.II | meta | Gemini 3.1 Pro | 34 | 0.443 | 0.654 | 0.725 |
| matrix_probes | cells.csv | cheat external B.II | meta | Gemini 3.5 Flash | 56 | 0.546 | 0.597 | 0.294 |
| matrix_probes | cells.csv | cheat external B.II | meta | Kimi K2.7 Code | 18 | 0.593 | 0.691 | 0.255 |
| matrix_probes | cells.csv | cheat all A | meta | Claude Fable 5 | 10 | 0.360 | 0.782 | 0.745 |
| matrix_probes | cells.csv | cheat all A | meta | Gemini 3.1 Pro | 34 | 0.578 | 0.704 | 0.294 |
| matrix_probes | cells.csv | cheat all A | meta | Gemini 3.5 Flash | 56 | 0.574 | 0.635 | 0.196 |
| matrix_probes | cells.csv | cheat all A | meta | Grok 4.5 | 40 | 0.672 | 0.688 | 0.118 |
| matrix_probes | cells.csv | cheat all A | meta | Kimi K2.7 Code | 18 | 0.481 | 0.790 | 0.431 |
| matrix_probes | cells.csv | cheat all B.II | meta | Claude Fable 5 | 10 | 0.600 | 0.826 | 0.333 |
| matrix_probes | cells.csv | cheat all B.II | tfidf | Claude Opus 4.8 | 6 | 1.000 | 0.889 | 0.059 |
| matrix_probes | cells.csv | cheat all B.II | meta | Claude Opus 4.8 | 6 | 1.000 | 1.000 | 0.098 |
| matrix_probes | cells.csv | cheat all B.II | meta | GPT-5 mini | 120 | 0.482 | 0.567 | 0.706 |
| matrix_probes | cells.csv | cheat all B.II | meta | GPT-5.3 Codex | 77 | 0.547 | 0.588 | 0.275 |
| matrix_probes | cells.csv | cheat all B.II | meta | GPT-5.6 Luna | 28 | 0.597 | 0.666 | 0.255 |
| matrix_probes | cells.csv | cheat all B.II | meta | GPT-5.6 Sol | 24 | 0.500 | 0.698 | 0.529 |
| matrix_probes | cells.csv | cheat all B.II | tfidf | Gemini 3.1 Pro | 34 | 0.443 | 0.653 | 0.804 |
| matrix_probes | cells.csv | cheat all B.II | meta | Gemini 3.1 Pro | 34 | 0.439 | 0.729 | 0.706 |
| matrix_probes | cells.csv | cheat all B.II | meta | Gemini 3.5 Flash | 56 | 0.536 | 0.637 | 0.373 |
| matrix_probes | cells.csv | cheat all B.II | meta | Kimi K2.7 Code | 18 | 0.580 | 0.744 | 0.353 |
| matrix_probes | cells.csv | cheat all B.II | meta | Qwen3.8 27B | 120 | 0.552 | 0.589 | 0.137 |
| matrix_probes | cells.csv | cheat leave out: controls B.I | meta | pooled | 600 | 0.541 | 0.534 | 0.059 |
| matrix_probes | cells.csv | cheat leave out: controls A | meta | Claude Haiku 5.5 | 120 | 0.336 | 0.602 | 1.000 |
| matrix_probes | cells.csv | cheat leave out: controls A | meta | DeepSeek V4.1 Flash | 120 | 0.388 | 0.575 | 1.000 |
| matrix_probes | cells.csv | cheat leave out: controls A | meta | mean over authors | 360 | 0.452 | 0.551 | 0.843 |
| matrix_probes | cells.csv | cheat leave out: controls B.II | meta | Claude Haiku 5.5 | 120 | 0.578 | 0.591 | 0.078 |
| matrix_probes | cells.csv | cheat leave out: controls B.II | meta | DeepSeek V4.1 Flash | 120 | 0.572 | 0.605 | 0.157 |
| matrix_probes | cells.csv | cheat leave out: controls B.II | meta | GPT-5 mini | 120 | 0.572 | 0.593 | 0.118 |
| matrix_probes | cells.csv | cheat leave out: controls B.II | meta | GPT-6 Luna | 120 | 0.565 | 0.582 | 0.118 |
| matrix_probes | cells.csv | cheat leave out: controls B.II | meta | Qwen3.8 27B | 120 | 0.267 | 0.587 | 1.000 |
| matrix_probes | cells.csv | cheat leave out: controls B.II | meta | mean over authors | 600 | 0.511 | 0.549 | 0.392 |
| matrix_probes | cells.csv | cheat leave out: solo_design A | meta | DeepSeek V4.1 Flash | 90 | 0.429 | 0.643 | 0.902 |
| matrix_probes | cells.csv | cheat leave out: solo_design B.II | meta | DeepSeek V4.1 Flash | 90 | 0.414 | 0.607 | 0.843 |
| matrix_probes | cells.csv | cheat leave out: solo_design B.II | meta | GPT-6 Luna | 81 | 0.502 | 0.657 | 0.451 |
| matrix_probes | cells.csv | cheat leave out: solo_design B.II | meta | mean over authors | 171 | 0.458 | 0.583 | 0.647 |
| matrix_probes | cells.csv | cheat leave out: spar_team A | tfidf | DeepSeek V4.1 Flash | 150 | 0.501 | 0.580 | 0.569 |
| matrix_probes | cells.csv | cheat leave out: spar_mixed A | meta | DeepSeek V4.1 Flash | 36 | 0.634 | 0.655 | 0.118 |
| matrix_probes | cells.csv | cheat leave out: spar_mixed A | meta | GPT-6 Luna | 126 | 0.669 | 0.812 | 0.176 |
| matrix_probes | cells.csv | cheat leave out: spar_mixed A | meta | mean over authors | 162 | 0.652 | 0.709 | 0.118 |
| matrix_probes | cells.csv | cheat leave out: spar_mixed B.II | meta | DeepSeek V4.1 Flash | 36 | 0.605 | 0.684 | 0.216 |
| matrix_probes | cells.csv | cheat leave out: spar_mixed B.II | meta | GPT-6 Luna | 126 | 0.230 | 0.712 | 0.902 |
| matrix_probes | cells.csv | cheat leave out: spar_mixed B.II | meta | mean over authors | 162 | 0.418 | 0.641 | 0.784 |
| matrix_probes | cells.csv | cheat leave out: spar_expl A | meta | Claude Haiku 5.5 | 36 | 0.523 | 0.715 | 0.647 |
| matrix_probes | cells.csv | cheat leave out: spar_expl A | meta | DeepSeek V4.1 Flash | 36 | 0.662 | 0.689 | 0.118 |
| matrix_probes | cells.csv | cheat leave out: spar_expl B.II | meta | Claude Haiku 5.5 | 36 | 0.490 | 0.757 | 0.627 |
| matrix_probes | cells.csv | cheat leave out: spar_expl B.II | meta | GPT-6 Luna | 241 | 0.401 | 0.572 | 0.961 |
| matrix_probes | cells.csv | cheat leave out: spar_expl B.II | meta | mean over authors | 313 | 0.588 | 0.595 | 0.078 |
| matrix_probes | cells.csv | cheat leave out: spar_solo A | meta | GPT-6 Luna | 30 | 0.603 | 0.670 | 0.157 |
| matrix_probes | cells.csv | cheat leave out: spar_solo B.II | meta | GPT-6 Luna | 30 | 0.522 | 0.701 | 0.569 |
| matrix_probes | cells.csv | cheat leave out: goodfire A | tfidf | Kimi K3 | 935 | 0.516 | 0.535 | 0.176 |
| matrix_probes | cells.csv | cheat leave out: goodfire A | meta | Kimi K3 | 935 | 0.253 | 0.537 | 1.000 |
| matrix_probes | cells.csv | cheat leave out: goodfire A | tfidf | mean over authors | 935 | 0.516 | 0.521 | 0.118 |
| matrix_probes | cells.csv | cheat leave out: goodfire A | meta | mean over authors | 935 | 0.253 | 0.533 | 1.000 |
| matrix_probes | cells.csv | cheat leave out: cheatbench A | tfidf | GPT-5.3 Codex | 35 | 0.504 | 0.697 | 0.392 |
| matrix_probes | cells.csv | cheat leave out: cheatbench A | tfidf | mean over authors | 35 | 0.504 | 0.690 | 0.392 |
| matrix_probes | cells.csv | cheat leave out: cheatbench B.II | tfidf | Claude Opus 4.6 | 111 | 0.647 | 0.777 | 0.255 |
| matrix_probes | cells.csv | cheat leave out: cheatbench B.II | tfidf | GPT-5.3 Codex | 35 | 0.568 | 0.685 | 0.275 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.I | meta | pooled | 266 | 0.531 | 0.568 | 0.176 |
| matrix_probes | cells.csv | cheat leave out: evilgenie A | tfidf | GPT-5.3 Codex | 42 | 0.574 | 0.651 | 0.157 |
| matrix_probes | cells.csv | cheat leave out: evilgenie A | tfidf | Kimi K3 | 8 | 0.688 | 0.938 | 0.314 |
| matrix_probes | cells.csv | cheat leave out: evilgenie A | meta | Kimi K3 | 8 | 0.312 | 0.847 | 0.804 |
| matrix_probes | cells.csv | cheat leave out: evilgenie A | tfidf | mean over authors | 50 | 0.631 | 0.670 | 0.118 |
| matrix_probes | cells.csv | cheat leave out: evilgenie A | meta | mean over authors | 50 | 0.519 | 0.644 | 0.412 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | Claude Fable 5 | 10 | 0.600 | 0.800 | 0.314 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | tfidf | Claude Opus 4.8 | 6 | 0.667 | 0.889 | 0.353 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | Claude Opus 4.8 | 6 | 1.000 | 1.000 | 0.098 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | tfidf | GPT-5.3 Codex | 42 | 0.442 | 0.629 | 0.824 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | GPT-5.3 Codex | 42 | 0.585 | 0.649 | 0.235 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | tfidf | GPT-5.6 Luna | 28 | 0.418 | 0.744 | 0.804 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | GPT-5.6 Luna | 28 | 0.551 | 0.717 | 0.373 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | tfidf | GPT-5.6 Sol | 24 | 0.674 | 0.690 | 0.078 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | GPT-5.6 Sol | 24 | 0.528 | 0.691 | 0.451 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | tfidf | Gemini 3.1 Pro | 34 | 0.429 | 0.623 | 0.824 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | Gemini 3.1 Pro | 34 | 0.405 | 0.660 | 0.824 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | Gemini 3.5 Flash | 56 | 0.514 | 0.616 | 0.431 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | tfidf | Grok 4.5 | 40 | 0.623 | 0.667 | 0.078 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | Kimi K2.7 Code | 18 | 0.605 | 0.761 | 0.353 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | tfidf | Kimi K3 | 8 | 0.812 | 1.000 | 0.235 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | Kimi K3 | 8 | 0.375 | 0.847 | 0.725 |
| matrix_probes | cells.csv | cheat leave out: evilgenie B.II | meta | mean over authors | 266 | 0.585 | 0.584 | 0.078 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | Claude Fable 5 | 10 | 0.440 | 0.782 | 0.627 |
| matrix_probes | cells.csv | cheat ours to external B.II | meta | Claude Fable 5 | 10 | 0.600 | 0.760 | 0.255 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | Claude Opus 4.8 | 6 | 0.111 | 0.839 | 0.980 |
| matrix_probes | cells.csv | cheat ours to external B.II | meta | Claude Opus 4.8 | 6 | 1.000 | 0.889 | 0.059 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | GPT-5.3 Codex | 77 | 0.328 | 0.646 | 1.000 |
| matrix_probes | cells.csv | cheat ours to external B.II | meta | GPT-5.3 Codex | 77 | 0.502 | 0.608 | 0.549 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | GPT-5.4 | 18 | 0.750 | 0.760 | 0.137 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | GPT-5.6 Luna | 28 | 0.352 | 0.709 | 0.922 |
| matrix_probes | cells.csv | cheat ours to external B.II | meta | GPT-5.6 Luna | 28 | 0.551 | 0.681 | 0.333 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | GPT-5.6 Sol | 24 | 0.618 | 0.691 | 0.157 |
| matrix_probes | cells.csv | cheat ours to external B.II | meta | GPT-5.6 Sol | 24 | 0.514 | 0.699 | 0.451 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | Gemini 3.1 Pro | 34 | 0.512 | 0.642 | 0.451 |
| matrix_probes | cells.csv | cheat ours to external B.II | meta | Gemini 3.1 Pro | 34 | 0.401 | 0.674 | 0.882 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | Gemini 3.5 Flash | 56 | 0.580 | 0.579 | 0.078 |
| matrix_probes | cells.csv | cheat ours to external B.II | meta | Gemini 3.5 Flash | 56 | 0.515 | 0.652 | 0.373 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | Grok 4.5 | 40 | 0.395 | 0.678 | 0.765 |
| matrix_probes | cells.csv | cheat ours to external B.II | meta | Kimi K2.7 Code | 18 | 0.593 | 0.761 | 0.255 |
| matrix_probes | cells.csv | cheat ours to external B.II | tfidf | mean over authors | 3323 | 0.570 | 0.561 | 0.059 |
| matrix_probes | cells.csv | cheat external to ours B.II | meta | Claude Haiku 5.5 | 364 | 0.470 | 0.546 | 0.745 |
| matrix_probes | cells.csv | cheat external to ours B.II | meta | DeepSeek V4.1 Flash | 596 | 0.480 | 0.547 | 0.882 |
| matrix_probes | cells.csv | cheat external to ours B.II | tfidf | GPT-5 mini | 120 | 0.497 | 0.587 | 0.588 |
| matrix_probes | cells.csv | cheat external to ours B.II | meta | GPT-5 mini | 120 | 0.580 | 0.596 | 0.118 |
| matrix_probes | cells.csv | cheat external to ours B.II | tfidf | GPT-6 Luna | 748 | 0.306 | 0.539 | 1.000 |
| matrix_probes | cells.csv | cheat external to ours B.II | meta | GPT-6 Luna | 748 | 0.341 | 0.538 | 1.000 |
| matrix_probes | cells.csv | cheat external to ours B.II | tfidf | Qwen3.8 27B | 120 | 0.583 | 0.632 | 0.157 |
| matrix_probes | cells.csv | cheat external to ours B.II | meta | Qwen3.8 27B | 120 | 0.283 | 0.584 | 1.000 |
| matrix_probes | cells.csv | cheat external to ours B.II | tfidf | mean over authors | 1948 | 0.503 | 0.532 | 0.431 |
| matrix_probes | cells.csv | cheat external to ours B.II | meta | mean over authors | 1948 | 0.431 | 0.522 | 1.000 |
| matrix_probes | cells.csv | coord alone: solo_design B.II | tfidf | GPT-6 Luna | 81 | 0.361 | 0.633 | 0.980 |
| matrix_probes | cells.csv | coord alone: solo_design B.II | meta | GPT-6 Luna | 81 | 0.371 | 0.640 | 1.000 |
| matrix_probes | cells.csv | coord alone: solo_design B.II | tfidf | mean over authors | 171 | 0.516 | 0.587 | 0.333 |
| matrix_probes | cells.csv | coord alone: solo_design B.II | meta | mean over authors | 171 | 0.540 | 0.604 | 0.294 |
| matrix_probes | cells.csv | coord alone: spar_team A | tfidf | DeepSeek V4.1 Flash | 150 | 0.667 | 0.709 | 0.118 |
| matrix_probes | cells.csv | coord alone: spar_team A | meta | DeepSeek V4.1 Flash | 150 | 0.740 | 0.744 | 0.078 |
| matrix_probes | cells.csv | coord alone: spar_team A | tfidf | GPT-6 Luna | 150 | 0.714 | 0.680 | 0.059 |
| matrix_probes | cells.csv | coord alone: spar_team B.II | meta | Claude Haiku 5.5 | 150 | 0.493 | 0.557 | 0.588 |
| matrix_probes | cells.csv | coord alone: spar_team B.II | tfidf | DeepSeek V4.1 Flash | 150 | 0.588 | 0.687 | 0.294 |
| matrix_probes | cells.csv | coord alone: spar_team B.II | meta | DeepSeek V4.1 Flash | 150 | 0.436 | 0.714 | 0.667 |
| matrix_probes | cells.csv | coord alone: spar_team B.II | meta | mean over authors | 450 | 0.625 | 0.611 | 0.059 |
| matrix_probes | cells.csv | coord leave out: solo_design B.I | tfidf | pooled | 177 | 0.496 | 0.585 | 0.529 |
| matrix_probes | cells.csv | coord leave out: solo_design B.I | meta | pooled | 177 | 0.444 | 0.615 | 0.863 |
| matrix_probes | cells.csv | coord leave out: solo_design A | tfidf | DeepSeek V4.1 Flash | 90 | 0.413 | 0.603 | 0.804 |
| matrix_probes | cells.csv | coord leave out: solo_design A | meta | DeepSeek V4.1 Flash | 90 | 0.422 | 0.635 | 0.725 |
| matrix_probes | cells.csv | coord leave out: solo_design A | tfidf | GPT-6 Luna | 81 | 0.298 | 0.635 | 1.000 |
| matrix_probes | cells.csv | coord leave out: solo_design A | meta | GPT-6 Luna | 81 | 0.518 | 0.602 | 0.373 |
| matrix_probes | cells.csv | coord leave out: solo_design A | tfidf | mean over authors | 171 | 0.355 | 0.566 | 1.000 |
| matrix_probes | cells.csv | coord leave out: solo_design A | meta | mean over authors | 171 | 0.470 | 0.576 | 0.863 |
| matrix_probes | cells.csv | coord leave out: solo_design B.II | meta | DeepSeek V4.1 Flash | 90 | 0.567 | 0.602 | 0.157 |
| matrix_probes | cells.csv | coord leave out: solo_design B.II | meta | GPT-6 Luna | 81 | 0.457 | 0.630 | 0.667 |
| matrix_probes | cells.csv | coord leave out: solo_design B.II | meta | mean over authors | 171 | 0.512 | 0.595 | 0.412 |
| matrix_probes | cells.csv | coord leave out: spar_team A | tfidf | DeepSeek V4.1 Flash | 150 | 0.436 | 0.702 | 0.765 |
| matrix_probes | cells.csv | coord leave out: spar_team A | meta | DeepSeek V4.1 Flash | 150 | 0.426 | 0.670 | 0.784 |
| matrix_probes | cells.csv | coord leave out: spar_team B.II | tfidf | Claude Haiku 5.5 | 150 | 0.647 | 0.600 | 0.059 |
| matrix_probes | cells.csv | coord leave out: spar_team B.II | tfidf | DeepSeek V4.1 Flash | 150 | 0.384 | 0.662 | 0.902 |
| matrix_probes | cells.csv | coord leave out: spar_team B.II | meta | DeepSeek V4.1 Flash | 150 | 0.414 | 0.695 | 0.902 |
| matrix_probes | cells.csv | coord leave out: spar_team B.II | tfidf | mean over authors | 450 | 0.613 | 0.627 | 0.098 |
| matrix_probes | cells.csv | coord leave out: spar_mixed B.I | tfidf | pooled | 209 | 0.470 | 0.712 | 0.686 |
| matrix_probes | cells.csv | coord leave out: spar_mixed B.I | meta | pooled | 209 | 0.423 | 0.750 | 0.686 |
| matrix_probes | cells.csv | collude alone: spar_team B.II | tfidf | DeepSeek V4.1 Flash | 150 | 0.554 | 0.590 | 0.157 |
| matrix_probes | cells.csv | collude alone: spar_mixed A | tfidf | DeepSeek V4.1 Flash | 36 | 0.281 | 0.704 | 0.922 |
| matrix_probes | cells.csv | collude alone: spar_mixed A | meta | DeepSeek V4.1 Flash | 36 | 0.704 | 0.678 | 0.059 |
| matrix_probes | cells.csv | collude alone: spar_mixed A | tfidf | mean over authors | 36 | 0.281 | 0.695 | 0.941 |
| matrix_probes | cells.csv | collude alone: spar_mixed A | meta | mean over authors | 36 | 0.704 | 0.668 | 0.059 |
| matrix_probes | cells.csv | collude ours B.II | meta | DeepSeek V4.1 Flash | 320 | 0.535 | 0.560 | 0.137 |
| matrix_probes | cells.csv | collude all B.II | meta | DeepSeek V4.1 Flash | 320 | 0.535 | 0.574 | 0.333 |
| matrix_probes | cells.csv | collude leave out: board_seed A | meta | DeepSeek V4.1 Flash | 134 | 0.512 | 0.593 | 0.529 |
| matrix_probes | cells.csv | collude leave out: board_seed A | meta | mean over authors | 134 | 0.512 | 0.570 | 0.275 |
| matrix_probes | cells.csv | collude leave out: board_seed B.II | meta | DeepSeek V4.1 Flash | 134 | 0.253 | 0.581 | 1.000 |
| matrix_probes | cells.csv | collude leave out: board_seed B.II | meta | mean over authors | 134 | 0.253 | 0.631 | 1.000 |
| matrix_probes | cells.csv | collude leave out: spar_team A | meta | DeepSeek V4.1 Flash | 150 | 0.561 | 0.605 | 0.235 |
| matrix_probes | cells.csv | collude leave out: spar_team A | meta | mean over authors | 150 | 0.561 | 0.610 | 0.275 |
| matrix_probes | cells.csv | collude leave out: spar_team B.II | tfidf | Claude Haiku 5.5 | 150 | 0.419 | 0.664 | 0.765 |
| matrix_probes | cells.csv | collude leave out: spar_team B.II | meta | Claude Haiku 5.5 | 150 | 0.568 | 0.781 | 0.529 |
| matrix_probes | cells.csv | collude leave out: spar_team B.II | tfidf | GPT-6 Luna | 150 | 0.273 | 0.587 | 1.000 |
| matrix_probes | cells.csv | collude leave out: spar_team B.II | meta | GPT-6 Luna | 150 | 0.297 | 0.584 | 1.000 |
| matrix_probes | cells.csv | collude leave out: spar_team B.II | tfidf | mean over authors | 300 | 0.346 | 0.594 | 0.980 |
| matrix_probes | cells.csv | collude leave out: spar_team B.II | meta | mean over authors | 300 | 0.433 | 0.601 | 0.902 |
| matrix_probes | cells.csv | collude leave out: spar_mixed B.I | meta | pooled | 209 | 0.647 | 0.682 | 0.176 |
| matrix_probes | cells.csv | collude leave out: spar_mixed B.II | tfidf | DeepSeek V4.1 Flash | 36 | 0.631 | 0.700 | 0.196 |
| matrix_probes | cells.csv | collude leave out: spar_mixed B.II | meta | DeepSeek V4.1 Flash | 36 | 0.236 | 0.678 | 1.000 |
| matrix_probes | cells.csv | collude leave out: spar_mixed B.II | tfidf | mean over authors | 36 | 0.631 | 0.689 | 0.157 |
| matrix_probes | cells.csv | collude leave out: spar_mixed B.II | meta | mean over authors | 36 | 0.236 | 0.659 | 0.980 |
| team_probes | agent_probes.csv | 1b cheating: opened or above   | TF-IDF agent's words | A: think pair, all agents | 129 | 0.574 | 0.765 | 0.333 |
| team_probes | agent_probes.csv | 1b cheating: opened or above   | TF-IDF whole text | A: think pair, all agents | 129 | 0.852 | 0.749 | 0.059 |
| team_probes | agent_probes.csv | 1b cheating: opened or above   | TF-IDF agent's words | B: all seven tokens, agents with turns.safetensors | 129 | 0.574 | 0.774 | 0.314 |
| team_probes | sets/solo_design/agent_probes.csv | 3r coordination: read the board batch 1 (team-hinting post)  | TF-IDF agent's words | A: think pair, all agents | 93 | 0.612 | 0.620 | 0.078 |
| team_probes | sets/solo_design/agent_probes.csv | 3r coordination: read the board batch 1 (team-hinting post)  | TF-IDF agent's words | B: all seven tokens, agents with turns.safetensors | 93 | 0.612 | 0.659 | 0.176 |
| team_probes | sets/solo_design/agent_probes.csv | 3a coordination: posted or above batch 1 (team-hinting post)  | metadata (arm, seed, turn budget) | A: think pair, all agents | 93 | 0.544 | 0.609 | 0.392 |
| team_probes | sets/solo_design/agent_probes.csv | 3a coordination: posted or above batch 1 (team-hinting post)  | TF-IDF agent's words | A: think pair, all agents | 93 | 0.554 | 0.585 | 0.157 |
| team_probes | sets/solo_design/agent_probes.csv | 3a coordination: posted or above batch 1 (team-hinting post)  | TF-IDF whole text | A: think pair, all agents | 93 | 0.508 | 0.582 | 0.353 |
| team_probes | sets/solo_design/agent_probes.csv | 3a coordination: posted or above batch 1 (team-hinting post)  | metadata (arm, seed, turn budget) | B: all seven tokens, agents with turns.safetensors | 93 | 0.544 | 0.620 | 0.294 |
| team_probes | sets/solo_design/agent_probes.csv | 3a coordination: posted or above batch 1 (team-hinting post)  | TF-IDF agent's words | B: all seven tokens, agents with turns.safetensors | 93 | 0.554 | 0.624 | 0.294 |
| team_probes | sets/solo_design/agent_probes.csv | 3a coordination: posted or above batch 1 (team-hinting post)  | TF-IDF whole text | B: all seven tokens, agents with turns.safetensors | 93 | 0.508 | 0.587 | 0.353 |
| team_probes | sets/solo_design/agent_probes.csv | 3r coordination: read the board clean post  | TF-IDF agent's words | A: think pair, all agents | 84 | 0.397 | 0.680 | 0.725 |
| team_probes | sets/solo_design/agent_probes.csv | 3r coordination: read the board clean post  | TF-IDF whole text | A: think pair, all agents | 84 | 0.545 | 0.686 | 0.255 |
| team_probes | sets/solo_design/agent_probes.csv | 3r coordination: read the board clean post  | TF-IDF agent's words | B: all seven tokens, agents with turns.safetensors | 84 | 0.397 | 0.725 | 0.882 |
| team_probes | sets/solo_design/agent_probes.csv | 3r coordination: read the board clean post  | TF-IDF whole text | B: all seven tokens, agents with turns.safetensors | 84 | 0.545 | 0.677 | 0.412 |
| team_probes | sets/solo_design/agent_probes.csv | 3a coordination: posted or above clean post  | metadata (arm, seed, turn budget) | A: think pair, all agents | 84 | 0.584 | 0.643 | 0.137 |
| team_probes | sets/solo_design/agent_probes.csv | 3a coordination: posted or above clean post  | metadata (arm, seed, turn budget) | B: all seven tokens, agents with turns.safetensors | 84 | 0.584 | 0.645 | 0.176 |
| team_probes | sets/spar_mixed_41/team_probe.csv | 4c team collaboration crossed or above   | max over workers of the held-out agent collusion probe score, probe Lall think-open+close | A: think pair, all agents | 51 | 0.648 | 0.689 | 0.118 |
| team_probes | sets/spar_mixed_41/team_probe.csv | 4c team collaboration crossed or above   | mean over workers of the held-out agent collusion probe score, probe Lall think-open+close | A: think pair, all agents | 51 | 0.599 | 0.665 | 0.196 |
| team_probes | sets/spar_mixed_41/team_probe.csv | 4c team collaboration crossed or above   | TF-IDF workers' words | A: think pair, all agents | 51 | 0.735 | 0.709 | 0.059 |
| external_probes | traj.json | cheatbench_cheating or attempt vs benign_held_author   | TF-IDF agent's words (rendered text) | nan | 224 | 0.441 | 0.563 | 0.922 |
| external_probes | traj.json | cheatbench_cheating or attempt vs benign_held_harness   | TF-IDF agent's words (rendered text) | nan | 224 | 0.175 | 0.580 | 1.000 |
| external_probes | traj.json | cheatbench_cheating vs benign (attempt dropped)_held_author   | metadata | nan | 215 | 0.459 | 0.561 | 0.863 |
| external_probes | traj.json | cheatbench_cheating vs benign (attempt dropped)_held_author   | TF-IDF agent's words (rendered text) | nan | 215 | 0.298 | 0.557 | 1.000 |
| external_probes | traj.json | cheatbench_cheating vs benign (attempt dropped)_held_harness   | TF-IDF agent's words (rendered text) | nan | 215 | 0.114 | 0.562 | 1.000 |
| external_probes | traj.json | evilgenie_hack vs legitimate_held_problem   | metadata | nan | 266 | 0.214 | 0.553 | 1.000 |
| external_probes | traj.json | evilgenie_hack vs legitimate_held_problem   | TF-IDF agent's words (rendered text) | nan | 266 | 0.174 | 0.538 | 1.000 |

Provenance: `results/controls_nulls/nulls_rows.csv` (part fixed, one line per row, with the seed).

**Rows still without a null.** These patterns were not computed here, for the reason given.

| Results dir | Section | Rows | Count | Why not computed |
|---|---|---|---|---|
| team_probes | Probe 5: per step (5a/5b/5c/5d, settings A and B, board_seed) | linear per-step probe (non-main configs) | 56 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| team_probes | Probe 5: per step (5a/5b/5c/5d, settings A and B, board_seed) | baselines (metadata, turn position, cumulative TF-IDF) | 40 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| team_probes | Probe 6: cheating probe trained on controls_600, applied unchanged (trained on c | transfer probe / TF-IDF transfer | 6 | no per-row scores saved |
| team_probes | Probe 6: cheating probe trained on controls_600, applied unchanged (trained on c | transfer probe / TF-IDF transfer | 24 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| team_probes | Probe 6: cheating probe trained on controls_600, applied unchanged (trained on c | transfer probe / TF-IDF transfer | 6 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| team_probes | Probe 6: cheating probe trained on controls_600, applied unchanged (trained on c | transfer probe / TF-IDF transfer | 18 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| team_probes | Set solo_design > solo_design: held-out author (fitted on the training authors,  | held-out-author probe and TF-IDF | 18 | no per-row scores saved |
| team_probes | Set spar_mixed_41 > spar_mixed_41: held-out author (fitted on the training autho | held-out-author probe and TF-IDF | 6 | no per-row scores saved |
| team_probes | Set spar_screen_team > spar_screen_team: held-out author (fitted on the training | held-out-author probe and TF-IDF | 27 | no per-row scores saved |
| team_probes | Gaps closed, 2026-11-09 > spar_screen_team with its own cheating label on every  | baselines (cumulative mean, GRU, turn index, TF-IDF) | 4 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| team_probes | Gaps closed, 2026-11-09 > spar_screen_team with its own cheating label on every  | linear per-step probe and baselines (before) | 5 | no per-row scores saved |
| team_probes | Gaps closed, 2026-11-09 > spar_screen_team ... > Probe 6 on the same set (before | transfer probe (controls solo) | 3 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| team_probes | Gaps closed, 2026-11-09 > spar_screen_team ... > Probe 6 on the same set (before | transfer probe (controls solo) | 3 | no per-row scores saved |
| attn_probe | <Set>: <tag> > <cut> (36 per-run model tables, 21 tags) | attention monitor, binary head (seed 0) | 23 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > <cut> (36 per-run model tables, 21 tags) | attention monitor, binary head (seed 1) | 36 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > <cut> (36 per-run model tables, 21 tags) | attention monitor, ordinal head P(level >= cut) | 72 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > <cut> (36 per-run model tables, 21 tags) | linear per-step probe (team_probes) | 27 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > <cut> (36 per-run model tables, 21 tags) | cumulative mean of the linear probe | 36 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > <cut> (36 per-run model tables, 21 tags) | GRU over the linear probe scores | 36 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > <cut> (36 per-run model tables, 21 tags) | turn index alone | 36 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > <cut> (36 per-run model tables, 21 tags) | TF-IDF on the agent's words so far | 36 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: think pair against the seven tokens (layer 38, binary head) (7 tables) | attention monitor, binary head, think pair (seed 1) | 13 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: think pair against the seven tokens (layer 38, binary head) (7 tables) | attention monitor, binary head, seven tokens | 26 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | Gaps closed, 2026-11-09 > spar_screen_team with its own cheating label on every  | monitor (other heads/seeds) and baselines | 7 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | Gaps closed, 2026-11-09 > spar_screen_team with its own cheating label on every  | monitor and baselines (before) | 9 | no per-row scores saved |
| attn_probe | Gaps closed, 2026-11-09 > Seven tokens (layer 38) against the think pair on the  | attention monitor binary, think pair s1 | 9 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | Gaps closed, 2026-11-09 > Seven tokens (layer 38) against the think pair on the  | attention monitor binary, seven tokens | 18 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | Gaps closed, 2026-11-09 > Three seeds of the headline rows | attention monitor binary head | 20 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | Gaps closed, 2026-11-09 > Label consistency ... (gate sensitivity table) | attention monitor binary head, gate variants | 24 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > Level accuracy (19 tables) | attention monitor ordinal levels | 138 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| attn_probe | <Set>: <tag> > Level accuracy (19 tables) | baseline: majority level | 69 | no per-row scores saved |
| attn_probe | <Set>: <tag>_heldout_author > Per held-out author, <cut> (8 tables) | attention monitor binary s0/s1 + linear, cumulative mean, GRU, turn in | 161 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| external_probes | Questions 1 and 4: per-group tables (per run dir; per held-out author; per held- | linear probe (4 layer sets), TF-IDF, metadata | 270 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| external_probes | Question 1 > 1a robustness (folds grouped by task; all 13 run dirs) | linear probe (all layers) and metadata | 2 | no per-row scores saved |
| external_probes | Question 2 > Definition A | linear per-step probe, attention monitor, baselines | 9 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| external_probes | Question 2 > Definition B | linear per-step probe, attention monitor, baselines | 8 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| external_probes | Question 3 (same tables) | transfer probe, within-stratum AUROC | 40 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| external_probes | Question 3 (same tables) | TF-IDF transfer | 38 | no per-row scores saved |
| external_probes | Question 0. Why did the controls-trained probe read 0.279 on CheatBench? (diagno | controls-trained transfer probe (all layers) | 8 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| matrix_probes | Source-to-source transfer (cheating 10x10, coordination 4x4, collusion 3x3) | think pair, all layers, transfer | 108 | no per-row scores saved |
| matrix_probes | Per step (layer 38, every turn) | baselines | 12 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| matrix_probes | Per step > Turn AUROC within each source of the scheme | linear per-step probe and turn baselines, per source | 88 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |
| matrix_probes | Architectures (38 tables + 2 summaries) and the architecture tables of One table | trained architectures 2-8 (mlp_mean, mlp_max, attn_pool, attn_monitor, | 503 | per-step or per-group scores: a valid permutation must keep each agent's turns together (labels permuted by agent, then the step metric recomputed); not done |

Total still without a null: 2098 rows.

## 5. Random-weights reader

Not run: it needs a GPU pass. The plan (subsample of 911 agents, 31.1M tokens, the extractor change, box time, files and the probes to rerun) is in `RANDOM_READER_PLAN.md`.
