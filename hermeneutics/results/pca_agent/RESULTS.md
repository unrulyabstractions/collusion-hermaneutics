# Results: a hermeneutic cheating probe on the controls subsample

Reader: Qwen3.8-27B, teacher-forced over each author's transcript. Features: the reader's residual stream at the `<think>` (open) and `</think>` (close) token of every assistant turn, at layers 26, 38 and 51 (HF `hidden_states` numbering, 64 blocks), averaged over the agent's turns (pooling `agent`) or one row per turn (pooling `turn`, scored per turn and as the mean turn score per agent). Probe: training-fold PCA to 256 components, z-scoring, L2 logistic regression; 5 folds grouped by run and stratified by author x label.

The **solo arm** is the primary test: no cheating instruction in the prompt, and both classes occur for every author. In the **cheat arm** the prompt asks for the shortcut and nearly every author takes it, so a probe pooled over both arms can read the condition from the prompt instead of the cheating; the pooled arm is reported with the condition erased for that reason. The author-identity probe and the LEACE erasure expose the author signature that think coverage alone creates (qwen 1.00, ds 0.88, gpt5m 0.48, luna 0.34, haiku 0.00 of turns carry readable reasoning; Haiku's think blocks are all empty).

## Agents in the cache

| author | condition | cheat | no cheat | turns | readable turns |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | cheat | 30 | 5 | 619 | 547 |
| DeepSeek V4.1 Flash | solo | 42 | 43 | 1681 | 1439 |
| GPT-5 mini | cheat | 30 | 30 | 1094 | 522 |
| GPT-5 mini | solo | 30 | 30 | 1244 | 623 |
| Claude Haiku 5.5 | cheat | 30 | 30 | 882 | 0 |
| Claude Haiku 5.5 | solo | 30 | 30 | 957 | 0 |
| GPT-6 Luna | cheat | 30 | 12 | 569 | 191 |
| GPT-6 Luna | solo | 39 | 39 | 1476 | 445 |
| Qwen3.8-27B | cheat | 30 | 0 | 455 | 455 |
| Qwen3.8-27B | solo | 52 | 38 | 2008 | 2008 |

## Solo arm (primary)

### Cheating AUROC, pooling `agent`

Layer all, position `both` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| All authors, pooled probe (B.I) | 193 / 180 | 0.967 [0.950, 0.981] | 0.964 [0.946, 0.978] | 0.944 [0.921, 0.965] |
| Within: DeepSeek V4.1 Flash (A) | 42 / 43 | 0.860 [0.775, 0.937] | 0.858 [0.771, 0.937] | 0.884 [0.807, 0.950] |
| Within: GPT-5 mini (A) | 30 / 30 | 0.862 [0.762, 0.953] | 0.868 [0.768, 0.956] | 0.898 [0.807, 0.978] |
| Within: Claude Haiku 5.5 (A) | 30 / 30 | 0.929 [0.852, 0.987] | 0.913 [0.831, 0.975] | 0.943 [0.884, 0.986] |
| Within: GPT-6 Luna (A) | 39 / 39 | 0.853 [0.757, 0.928] | 0.845 [0.749, 0.921] | 0.811 [0.694, 0.896] |
| Within: Qwen3.8-27B (A) | 52 / 38 | 0.955 [0.910, 0.987] | 0.943 [0.893, 0.984] | 0.945 [0.899, 0.981] |
| Leave out: DeepSeek V4.1 Flash (B.II) | 42 / 43 | 0.892 [0.813, 0.952] | 0.899 [0.827, 0.954] | 0.918 [0.848, 0.969] |
| Leave out: GPT-5 mini (B.II) | 30 / 30 | 0.946 [0.884, 0.992] | 0.953 [0.899, 0.993] | 0.969 [0.922, 0.997] |
| Leave out: Claude Haiku 5.5 (B.II) | 30 / 30 | 0.973 [0.934, 0.997] | 0.964 [0.910, 0.997] | 0.941 [0.875, 0.984] |
| Leave out: GPT-6 Luna (B.II) | 39 / 39 | 0.800 [0.698, 0.886] | 0.780 [0.670, 0.871] | 0.780 [0.675, 0.875] |
| Leave out: Qwen3.8-27B (B.II) | 52 / 38 | 0.966 [0.925, 0.993] | 0.972 [0.941, 0.995] | 0.981 [0.955, 0.998] |

Sweep over layers and positions, pooling `agent`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | both | 0.938 [0.913, 0.959] | 0.929 | 0.567 | 0.010 |
| 26 | close | 0.940 [0.915, 0.962] | 0.930 | 0.556 | 0.010 |
| 26 | open | 0.946 [0.924, 0.966] | 0.939 | 0.569 | 0.010 |
| 38 | both | 0.954 [0.932, 0.972] | 0.943 | 0.555 | 0.010 |
| 38 | close | 0.956 [0.932, 0.974] | 0.950 | 0.557 | 0.010 |
| 38 | open | 0.944 [0.920, 0.964] | 0.936 | 0.554 | 0.010 |
| 51 | both | 0.959 [0.940, 0.975] | 0.957 | 0.549 | 0.010 |
| 51 | close | 0.961 [0.943, 0.977] | 0.959 | 0.558 | 0.010 |
| 51 | open | 0.955 [0.935, 0.973] | 0.951 | 0.561 | 0.010 |
| all | both | 0.967 [0.950, 0.981] | 0.963 | 0.550 | 0.010 |
| all | close | 0.959 [0.941, 0.975] | 0.957 | 0.561 | 0.010 |
| all | open | 0.952 [0.932, 0.971] | 0.948 | 0.555 | 0.010 |

### Author-identity probe

Majority-class rate 0.241.

| features | accuracy | balanced accuracy |
|---|---|---|
| raw | 0.989 | 0.989 |
| author erased | 0.241 | 0.200 |
| author, task, condition erased | 0.241 | 0.200 |

Confusion matrix, raw features (rows true, columns predicted):

| true \ predicted | DeepSeek V4.1 Flash | GPT-5 mini | Claude Haiku 5.5 | GPT-6 Luna | Qwen3.8-27B |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 85 | 0 | 0 | 0 | 0 |
| GPT-5 mini | 0 | 59 | 0 | 1 | 0 |
| Claude Haiku 5.5 | 0 | 0 | 59 | 1 | 0 |
| GPT-6 Luna | 0 | 0 | 1 | 77 | 0 |
| Qwen3.8-27B | 1 | 0 | 0 | 0 | 89 |

### Label-permutation null (labels shuffled within author)

| features | scope | real AUROC | null mean | null 95th pct | p |
|---|---|---|---|---|---|
| raw | pooled | 0.967 | 0.496 | 0.550 | 0.010 |
| raw | ds | 0.860 | 0.489 | 0.626 | 0.010 |
| raw | gpt5m | 0.862 | 0.495 | 0.646 | 0.010 |
| raw | haiku | 0.929 | 0.489 | 0.643 | 0.010 |
| raw | luna | 0.853 | 0.488 | 0.636 | 0.010 |
| raw | qwen | 0.955 | 0.499 | 0.626 | 0.010 |
| author erased | pooled | 0.964 | 0.502 | 0.557 | 0.010 |
| author erased | ds | 0.858 | 0.488 | 0.626 | 0.010 |
| author erased | gpt5m | 0.868 | 0.493 | 0.651 | 0.010 |
| author erased | haiku | 0.913 | 0.489 | 0.642 | 0.010 |
| author erased | luna | 0.845 | 0.487 | 0.629 | 0.010 |
| author erased | qwen | 0.943 | 0.498 | 0.621 | 0.010 |


## Both arms pooled (secondary)

### Cheating AUROC, pooling `agent`

Layer 38, position `close` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| All authors, pooled probe (B.I) | 343 / 257 | 0.963 [0.949, 0.976] | 0.956 [0.941, 0.970] | 0.945 [0.926, 0.963] |
| Within: DeepSeek V4.1 Flash (A) | 72 / 48 | 0.945 [0.902, 0.979] | 0.947 [0.904, 0.981] | 0.926 [0.874, 0.966] |
| Within: GPT-5 mini (A) | 60 / 60 | 0.855 [0.777, 0.918] | 0.864 [0.786, 0.927] | 0.892 [0.828, 0.947] |
| Within: Claude Haiku 5.5 (A) | 60 / 60 | 0.959 [0.922, 0.984] | 0.960 [0.924, 0.983] | 0.954 [0.913, 0.983] |
| Within: GPT-6 Luna (A) | 69 / 51 | 0.845 [0.766, 0.911] | 0.845 [0.765, 0.909] | 0.899 [0.833, 0.955] |
| Within: Qwen3.8-27B (A) | 82 / 38 | 0.966 [0.927, 0.993] | 0.955 [0.907, 0.989] | 0.973 [0.944, 0.993] |
| Leave out: DeepSeek V4.1 Flash (B.II) | 72 / 48 | 0.917 [0.865, 0.960] | 0.928 [0.875, 0.967] | 0.893 [0.837, 0.942] |
| Leave out: GPT-5 mini (B.II) | 60 / 60 | 0.937 [0.886, 0.975] | 0.927 [0.872, 0.968] | 0.942 [0.896, 0.975] |
| Leave out: Claude Haiku 5.5 (B.II) | 60 / 60 | 0.923 [0.868, 0.968] | 0.924 [0.869, 0.966] | 0.864 [0.799, 0.922] |
| Leave out: GPT-6 Luna (B.II) | 69 / 51 | 0.896 [0.837, 0.944] | 0.893 [0.832, 0.941] | 0.873 [0.809, 0.928] |
| Leave out: Qwen3.8-27B (B.II) | 82 / 38 | 0.977 [0.952, 0.995] | 0.974 [0.948, 0.992] | 0.976 [0.950, 0.995] |

Sweep over layers and positions, pooling `agent`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | both | 0.945 [0.926, 0.964] | 0.944 | 0.558 | 0.010 |
| 26 | close | 0.942 [0.924, 0.959] | 0.938 | 0.557 | 0.010 |
| 26 | open | 0.958 [0.942, 0.973] | 0.956 | 0.561 | 0.010 |
| 38 | both | 0.957 [0.941, 0.972] | 0.956 | 0.557 | 0.010 |
| 38 | close | 0.963 [0.949, 0.976] | 0.961 | 0.555 | 0.010 |
| 38 | open | 0.940 [0.921, 0.958] | 0.939 | 0.568 | 0.010 |
| 51 | both | 0.960 [0.943, 0.975] | 0.959 | 0.566 | 0.010 |
| 51 | close | 0.955 [0.936, 0.972] | 0.956 | 0.561 | 0.010 |
| 51 | open | 0.943 [0.922, 0.961] | 0.940 | 0.565 | 0.010 |
| all | both | 0.962 [0.946, 0.976] | 0.961 | 0.564 | 0.010 |
| all | close | 0.958 [0.939, 0.974] | 0.958 | 0.560 | 0.010 |
| all | open | 0.951 [0.933, 0.968] | 0.951 | 0.563 | 0.010 |

### Author-identity probe

Majority-class rate 0.200.

| features | accuracy | balanced accuracy |
|---|---|---|
| raw | 0.990 | 0.990 |
| author erased | 0.198 | 0.198 |
| author, task, condition erased | 0.192 | 0.192 |

Confusion matrix, raw features (rows true, columns predicted):

| true \ predicted | DeepSeek V4.1 Flash | GPT-5 mini | Claude Haiku 5.5 | GPT-6 Luna | Qwen3.8-27B |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 120 | 0 | 0 | 0 | 0 |
| GPT-5 mini | 0 | 119 | 0 | 1 | 0 |
| Claude Haiku 5.5 | 0 | 0 | 118 | 2 | 0 |
| GPT-6 Luna | 0 | 0 | 2 | 118 | 0 |
| Qwen3.8-27B | 1 | 0 | 0 | 0 | 119 |

### Condition readout (cheat-arm prompt vs solo prompt)

Majority-class rate 0.622.

| features | accuracy |
|---|---|
| raw | 0.988 |
| author erased | 0.987 |
| author, task, condition erased | 0.622 |

### Label-permutation null (labels shuffled within author)

| features | scope | real AUROC | null mean | null 95th pct | p |
|---|---|---|---|---|---|
| raw | pooled | 0.963 | 0.512 | 0.555 | 0.010 |
| raw | ds | 0.945 | 0.506 | 0.632 | 0.010 |
| raw | gpt5m | 0.855 | 0.504 | 0.623 | 0.010 |
| raw | haiku | 0.959 | 0.491 | 0.611 | 0.010 |
| raw | luna | 0.845 | 0.490 | 0.610 | 0.010 |
| raw | qwen | 0.966 | 0.488 | 0.624 | 0.010 |
| author erased | pooled | 0.956 | 0.500 | 0.545 | 0.010 |
| author erased | ds | 0.947 | 0.506 | 0.627 | 0.010 |
| author erased | gpt5m | 0.864 | 0.504 | 0.624 | 0.010 |
| author erased | haiku | 0.960 | 0.492 | 0.610 | 0.010 |
| author erased | luna | 0.845 | 0.491 | 0.614 | 0.010 |
| author erased | qwen | 0.955 | 0.487 | 0.622 | 0.010 |


## Cheat arm (secondary)

### Cheating AUROC, pooling `agent`

Layer 38, position `close` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| All authors, pooled probe (B.I) | 150 / 77 | 0.940 [0.902, 0.969] | 0.896 [0.843, 0.939] | 0.862 [0.808, 0.911] |
| Within: DeepSeek V4.1 Flash (A) | 30 / 5 | 0.967 [0.870, 1.000] | 0.973 [0.894, 1.000] | 0.973 [0.898, 1.000] |
| Within: GPT-5 mini (A) | 30 / 30 | 0.829 [0.715, 0.929] | 0.833 [0.709, 0.932] | 0.910 [0.826, 0.974] |
| Within: Claude Haiku 5.5 (A) | 30 / 30 | 0.867 [0.760, 0.956] | 0.872 [0.766, 0.958] | 0.884 [0.791, 0.966] |
| Within: GPT-6 Luna (A) | 30 / 12 | 0.811 [0.647, 0.956] | 0.814 [0.647, 0.956] | 0.844 [0.691, 0.970] |
| Leave out: DeepSeek V4.1 Flash (B.II) | 30 / 5 | 0.840 [0.597, 1.000] | 0.793 [0.531, 1.000] | 0.840 [0.592, 1.000] |
| Leave out: GPT-5 mini (B.II) | 30 / 30 | 0.643 [0.488, 0.780] | 0.629 [0.481, 0.768] | 0.614 [0.470, 0.753] |
| Leave out: Claude Haiku 5.5 (B.II) | 30 / 30 | 0.853 [0.744, 0.944] | 0.856 [0.750, 0.944] | 0.844 [0.733, 0.939] |
| Leave out: GPT-6 Luna (B.II) | 30 / 12 | 0.936 [0.821, 1.000] | 0.978 [0.925, 1.000] | 0.939 [0.819, 1.000] |

Sweep over layers and positions, pooling `agent`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | both | 0.930 [0.895, 0.961] | 0.909 | 0.701 | 0.010 |
| 26 | close | 0.929 [0.893, 0.962] | 0.901 | 0.702 | 0.010 |
| 26 | open | 0.916 [0.871, 0.957] | 0.898 | 0.702 | 0.010 |
| 38 | both | 0.906 [0.862, 0.944] | 0.887 | 0.696 | 0.010 |
| 38 | close | 0.940 [0.902, 0.969] | 0.930 | 0.694 | 0.010 |
| 38 | open | 0.937 [0.901, 0.968] | 0.922 | 0.695 | 0.010 |
| 51 | both | 0.920 [0.878, 0.957] | 0.913 | 0.701 | 0.010 |
| 51 | close | 0.935 [0.895, 0.969] | 0.915 | 0.694 | 0.010 |
| 51 | open | 0.904 [0.861, 0.942] | 0.898 | 0.700 | 0.010 |
| all | both | 0.910 [0.859, 0.950] | 0.894 | 0.700 | 0.010 |
| all | close | 0.917 [0.870, 0.956] | 0.898 | 0.689 | 0.010 |
| all | open | 0.920 [0.883, 0.952] | 0.913 | 0.697 | 0.010 |

### Author-identity probe

Majority-class rate 0.264.

| features | accuracy | balanced accuracy |
|---|---|---|
| raw | 0.987 | 0.987 |
| author erased | 0.242 | 0.183 |
| author, task, condition erased | 0.233 | 0.177 |

Confusion matrix, raw features (rows true, columns predicted):

| true \ predicted | DeepSeek V4.1 Flash | GPT-5 mini | Claude Haiku 5.5 | GPT-6 Luna | Qwen3.8-27B |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 35 | 0 | 0 | 0 | 0 |
| GPT-5 mini | 0 | 60 | 0 | 0 | 0 |
| Claude Haiku 5.5 | 0 | 0 | 59 | 1 | 0 |
| GPT-6 Luna | 0 | 0 | 2 | 40 | 0 |
| Qwen3.8-27B | 0 | 0 | 0 | 0 | 30 |

### Label-permutation null (labels shuffled within author)

| features | scope | real AUROC | null mean | null 95th pct | p |
|---|---|---|---|---|---|
| raw | pooled | 0.940 | 0.630 | 0.694 | 0.010 |
| raw | ds | 0.967 | 0.426 | 0.734 | 0.010 |
| raw | gpt5m | 0.829 | 0.503 | 0.647 | 0.010 |
| raw | haiku | 0.867 | 0.476 | 0.635 | 0.010 |
| raw | luna | 0.811 | 0.493 | 0.713 | 0.020 |
| author erased | pooled | 0.896 | 0.509 | 0.588 | 0.010 |
| author erased | ds | 0.973 | 0.422 | 0.734 | 0.010 |
| author erased | gpt5m | 0.833 | 0.506 | 0.662 | 0.010 |
| author erased | haiku | 0.872 | 0.477 | 0.625 | 0.010 |
| author erased | luna | 0.814 | 0.493 | 0.720 | 0.020 |

