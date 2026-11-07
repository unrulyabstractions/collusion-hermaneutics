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

Layer all, position `close` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| All authors, pooled probe (B.I) | 193 / 180 | 0.971 [0.954, 0.984] | 0.963 [0.944, 0.979] | 0.941 [0.914, 0.963] |
| Within: DeepSeek V4.1 Flash (A) | 42 / 43 | 0.930 [0.867, 0.975] | 0.927 [0.863, 0.974] | 0.941 [0.887, 0.984] |
| Within: GPT-5 mini (A) | 30 / 30 | 0.942 [0.872, 0.990] | 0.949 [0.880, 0.993] | 0.942 [0.875, 0.991] |
| Within: Claude Haiku 5.5 (A) | 30 / 30 | 0.974 [0.934, 1.000] | 0.976 [0.936, 1.000] | 0.958 [0.909, 0.994] |
| Within: GPT-6 Luna (A) | 39 / 39 | 0.880 [0.792, 0.951] | 0.893 [0.811, 0.961] | 0.896 [0.811, 0.961] |
| Within: Qwen3.8-27B (A) | 52 / 38 | 0.952 [0.906, 0.986] | 0.948 [0.901, 0.985] | 0.947 [0.902, 0.984] |
| Leave out: DeepSeek V4.1 Flash (B.II) | 42 / 43 | 0.924 [0.863, 0.975] | 0.908 [0.837, 0.968] | 0.939 [0.880, 0.985] |
| Leave out: GPT-5 mini (B.II) | 30 / 30 | 0.978 [0.940, 0.999] | 0.981 [0.942, 1.000] | 0.969 [0.921, 0.996] |
| Leave out: Claude Haiku 5.5 (B.II) | 30 / 30 | 0.928 [0.853, 0.984] | 0.899 [0.814, 0.969] | 0.907 [0.824, 0.971] |
| Leave out: GPT-6 Luna (B.II) | 39 / 39 | 0.784 [0.676, 0.874] | 0.776 [0.663, 0.868] | 0.727 [0.609, 0.831] |
| Leave out: Qwen3.8-27B (B.II) | 52 / 38 | 0.939 [0.886, 0.980] | 0.950 [0.903, 0.986] | 0.928 [0.871, 0.973] |

Sweep over layers and positions, pooling `agent`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | both | 0.958 [0.936, 0.974] | 0.945 | 0.562 | 0.010 |
| 26 | close | 0.952 [0.929, 0.970] | 0.938 | 0.554 | 0.010 |
| 26 | open | 0.953 [0.931, 0.971] | 0.938 | 0.566 | 0.010 |
| 38 | both | 0.966 [0.949, 0.980] | 0.958 | 0.562 | 0.010 |
| 38 | close | 0.968 [0.952, 0.982] | 0.963 | 0.563 | 0.010 |
| 38 | open | 0.953 [0.931, 0.972] | 0.944 | 0.550 | 0.010 |
| 51 | both | 0.967 [0.950, 0.980] | 0.964 | 0.552 | 0.010 |
| 51 | close | 0.965 [0.948, 0.980] | 0.964 | 0.558 | 0.010 |
| 51 | open | 0.961 [0.942, 0.977] | 0.953 | 0.564 | 0.010 |
| all | both | 0.969 [0.951, 0.984] | 0.966 | 0.564 | 0.010 |
| all | close | 0.971 [0.954, 0.984] | 0.970 | 0.559 | 0.010 |
| all | open | 0.962 [0.942, 0.979] | 0.954 | 0.560 | 0.010 |

### Author-identity probe

Majority-class rate 0.241.

| features | accuracy | balanced accuracy |
|---|---|---|
| raw | 0.992 | 0.992 |
| author erased | 0.241 | 0.200 |
| author, task, condition erased | 0.241 | 0.200 |

Confusion matrix, raw features (rows true, columns predicted):

| true \ predicted | DeepSeek V4.1 Flash | GPT-5 mini | Claude Haiku 5.5 | GPT-6 Luna | Qwen3.8-27B |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 85 | 0 | 0 | 0 | 0 |
| GPT-5 mini | 0 | 59 | 0 | 1 | 0 |
| Claude Haiku 5.5 | 0 | 0 | 60 | 0 | 0 |
| GPT-6 Luna | 0 | 0 | 1 | 77 | 0 |
| Qwen3.8-27B | 1 | 0 | 0 | 0 | 89 |

### Label-permutation null (labels shuffled within author)

| features | scope | real AUROC | null mean | null 95th pct | p |
|---|---|---|---|---|---|
| raw | pooled | 0.971 | 0.499 | 0.559 | 0.010 |
| raw | ds | 0.930 | 0.491 | 0.621 | 0.010 |
| raw | gpt5m | 0.942 | 0.492 | 0.658 | 0.010 |
| raw | haiku | 0.974 | 0.490 | 0.648 | 0.010 |
| raw | luna | 0.880 | 0.490 | 0.641 | 0.010 |
| raw | qwen | 0.952 | 0.503 | 0.633 | 0.010 |
| author erased | pooled | 0.963 | 0.503 | 0.570 | 0.010 |
| author erased | ds | 0.927 | 0.491 | 0.621 | 0.010 |
| author erased | gpt5m | 0.949 | 0.493 | 0.652 | 0.010 |
| author erased | haiku | 0.976 | 0.492 | 0.640 | 0.010 |
| author erased | luna | 0.893 | 0.490 | 0.635 | 0.010 |
| author erased | qwen | 0.948 | 0.502 | 0.630 | 0.010 |


## Both arms pooled (secondary)

### Cheating AUROC, pooling `agent`

Layer 26, position `open` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| All authors, pooled probe (B.I) | 343 / 257 | 0.955 [0.940, 0.969] | 0.945 [0.928, 0.962] | 0.925 [0.903, 0.946] |
| Within: DeepSeek V4.1 Flash (A) | 72 / 48 | 0.945 [0.901, 0.978] | 0.942 [0.896, 0.977] | 0.941 [0.895, 0.976] |
| Within: GPT-5 mini (A) | 60 / 60 | 0.889 [0.824, 0.943] | 0.880 [0.813, 0.937] | 0.891 [0.825, 0.945] |
| Within: Claude Haiku 5.5 (A) | 60 / 60 | 0.944 [0.898, 0.980] | 0.939 [0.891, 0.978] | 0.967 [0.931, 0.992] |
| Within: GPT-6 Luna (A) | 69 / 51 | 0.858 [0.789, 0.922] | 0.864 [0.794, 0.926] | 0.877 [0.810, 0.932] |
| Within: Qwen3.8-27B (A) | 82 / 38 | 0.957 [0.908, 0.992] | 0.951 [0.900, 0.990] | 0.932 [0.872, 0.980] |
| Leave out: DeepSeek V4.1 Flash (B.II) | 72 / 48 | 0.925 [0.876, 0.965] | 0.917 [0.861, 0.962] | 0.926 [0.880, 0.969] |
| Leave out: GPT-5 mini (B.II) | 60 / 60 | 0.934 [0.885, 0.975] | 0.931 [0.879, 0.972] | 0.943 [0.893, 0.981] |
| Leave out: Claude Haiku 5.5 (B.II) | 60 / 60 | 0.900 [0.834, 0.957] | 0.911 [0.849, 0.965] | 0.863 [0.785, 0.929] |
| Leave out: GPT-6 Luna (B.II) | 69 / 51 | 0.830 [0.750, 0.900] | 0.829 [0.750, 0.901] | 0.809 [0.729, 0.884] |
| Leave out: Qwen3.8-27B (B.II) | 82 / 38 | 0.963 [0.922, 0.992] | 0.959 [0.917, 0.988] | 0.927 [0.874, 0.971] |

Sweep over layers and positions, pooling `agent`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | close | 0.949 [0.934, 0.965] | 0.946 | 0.568 | 0.010 |
| 26 | open | 0.955 [0.940, 0.969] | 0.951 | 0.570 | 0.010 |

### Author-identity probe

Majority-class rate 0.200.

| features | accuracy | balanced accuracy |
|---|---|---|
| raw | 0.983 | 0.983 |
| author erased | 0.188 | 0.188 |
| author, task, condition erased | 0.193 | 0.193 |

Confusion matrix, raw features (rows true, columns predicted):

| true \ predicted | DeepSeek V4.1 Flash | GPT-5 mini | Claude Haiku 5.5 | GPT-6 Luna | Qwen3.8-27B |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 120 | 0 | 0 | 0 | 0 |
| GPT-5 mini | 0 | 118 | 1 | 1 | 0 |
| Claude Haiku 5.5 | 0 | 0 | 116 | 4 | 0 |
| GPT-6 Luna | 0 | 0 | 3 | 117 | 0 |
| Qwen3.8-27B | 1 | 0 | 0 | 0 | 119 |

### Condition readout (cheat-arm prompt vs solo prompt)

Majority-class rate 0.622.

| features | accuracy |
|---|---|
| raw | 0.993 |
| author erased | 0.995 |
| author, task, condition erased | 0.622 |

### Label-permutation null (labels shuffled within author)

| features | scope | real AUROC | null mean | null 95th pct | p |
|---|---|---|---|---|---|
| raw | pooled | 0.955 | 0.512 | 0.570 | 0.010 |
| raw | ds | 0.945 | 0.511 | 0.637 | 0.010 |
| raw | gpt5m | 0.889 | 0.493 | 0.607 | 0.010 |
| raw | haiku | 0.944 | 0.493 | 0.593 | 0.010 |
| raw | luna | 0.858 | 0.494 | 0.633 | 0.010 |
| raw | qwen | 0.957 | 0.486 | 0.603 | 0.010 |
| author erased | pooled | 0.945 | 0.503 | 0.559 | 0.010 |
| author erased | ds | 0.942 | 0.512 | 0.637 | 0.010 |
| author erased | gpt5m | 0.880 | 0.492 | 0.589 | 0.010 |
| author erased | haiku | 0.939 | 0.494 | 0.592 | 0.010 |
| author erased | luna | 0.864 | 0.494 | 0.623 | 0.010 |
| author erased | qwen | 0.951 | 0.488 | 0.593 | 0.010 |

