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

### Cheating AUROC, pooling `turn`

Layer 26, position `open` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw |
|---|---|---|
| All authors, pooled probe (B.I) turns 0.949 | 192 / 180 | 0.986 [0.975, 0.993] |
| Within: DeepSeek V4.1 Flash (A) turns 0.962 | 42 / 43 | 0.967 [0.929, 0.994] |
| Within: GPT-5 mini (A) turns 0.961 | 29 / 30 | 0.972 [0.931, 1.000] |
| Within: Claude Haiku 5.5 (A) turns 0.898 | 30 / 30 | 0.966 [0.907, 1.000] |
| Within: GPT-6 Luna (A) turns 0.901 | 39 / 39 | 0.964 [0.919, 0.995] |
| Within: Qwen3.8-27B (A) turns 0.978 | 52 / 38 | 0.991 [0.974, 1.000] |
| Leave out: DeepSeek V4.1 Flash (B.II) turns 0.948 | 42 / 43 | 0.983 [0.952, 1.000] |
| Leave out: GPT-5 mini (B.II) turns 0.907 | 29 / 30 | 0.968 [0.925, 0.998] |
| Leave out: Claude Haiku 5.5 (B.II) turns 0.923 | 30 / 30 | 0.982 [0.950, 0.999] |
| Leave out: GPT-6 Luna (B.II) turns 0.740 | 39 / 39 | 0.832 [0.735, 0.911] |
| Leave out: Qwen3.8-27B (B.II) turns 0.953 | 52 / 38 | 0.995 [0.983, 1.000] |

Sweep over layers and positions, pooling `turn`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | close | 0.969 [0.951, 0.985] | 0.964 | n/a | n/a |
| 26 | open | 0.986 [0.975, 0.993] | 0.981 | n/a | n/a |
| 38 | close | 0.978 [0.964, 0.988] | 0.974 | n/a | n/a |
| 38 | open | 0.980 [0.968, 0.989] | 0.978 | n/a | n/a |
| 51 | close | 0.984 [0.975, 0.992] | 0.983 | n/a | n/a |
| 51 | open | 0.979 [0.967, 0.988] | 0.981 | n/a | n/a |


## Both arms pooled (secondary)

### Cheating AUROC, pooling `turn`

Layer 38, position `close` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw |
|---|---|---|
| All authors, pooled probe (B.I) turns 0.938 | 336 / 257 | 0.985 [0.977, 0.993] |
| Within: DeepSeek V4.1 Flash (A) turns 0.952 | 72 / 48 | 0.983 [0.961, 0.997] |
| Within: GPT-5 mini (A) turns 0.893 | 53 / 60 | 0.985 [0.967, 0.997] |
| Within: Claude Haiku 5.5 (A) turns 0.946 | 60 / 60 | 0.972 [0.931, 0.999] |
| Within: GPT-6 Luna (A) turns 0.918 | 69 / 51 | 0.987 [0.968, 0.999] |
| Within: Qwen3.8-27B (A) turns 0.957 | 82 / 38 | 0.979 [0.952, 1.000] |
| Leave out: DeepSeek V4.1 Flash (B.II) turns 0.874 | 72 / 48 | 0.955 [0.915, 0.986] |
| Leave out: GPT-5 mini (B.II) turns 0.795 | 53 / 60 | 0.880 [0.806, 0.934] |
| Leave out: Claude Haiku 5.5 (B.II) turns 0.928 | 60 / 60 | 0.975 [0.948, 0.991] |
| Leave out: GPT-6 Luna (B.II) turns 0.835 | 69 / 51 | 0.938 [0.893, 0.977] |
| Leave out: Qwen3.8-27B (B.II) turns 0.841 | 82 / 38 | 0.993 [0.980, 1.000] |

Sweep over layers and positions, pooling `turn`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | close | 0.983 [0.973, 0.992] | 0.983 | n/a | n/a |
| 26 | open | 0.984 [0.975, 0.992] | 0.985 | n/a | n/a |
| 38 | close | 0.985 [0.977, 0.993] | 0.984 | n/a | n/a |
| 38 | open | 0.980 [0.969, 0.989] | 0.978 | n/a | n/a |
| 51 | close | 0.985 [0.977, 0.992] | 0.985 | n/a | n/a |
| 51 | open | 0.979 [0.967, 0.988] | 0.980 | n/a | n/a |


## Cheat arm (secondary)

### Cheating AUROC, pooling `turn`

Layer 51, position `open` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw |
|---|---|---|
| All authors, pooled probe (B.I) turns 0.956 | 144 / 77 | 0.981 [0.965, 0.993] |
| Within: DeepSeek V4.1 Flash (A) turns 0.835 | 30 / 5 | 0.907 [0.707, 1.000] |
| Within: GPT-5 mini (A) turns 0.876 | 24 / 30 | 0.944 [0.879, 0.992] |
| Within: Claude Haiku 5.5 (A) turns 0.960 | 30 / 30 | 0.977 [0.939, 1.000] |
| Within: GPT-6 Luna (A) turns 0.722 | 30 / 12 | 0.861 [0.677, 1.000] |
| Leave out: DeepSeek V4.1 Flash (B.II) turns 0.621 | 30 / 5 | 0.887 [0.771, 0.979] |
| Leave out: GPT-5 mini (B.II) turns 0.653 | 24 / 30 | 0.678 [0.529, 0.801] |
| Leave out: Claude Haiku 5.5 (B.II) turns 0.928 | 30 / 30 | 0.986 [0.949, 1.000] |
| Leave out: GPT-6 Luna (B.II) turns 0.912 | 30 / 12 | 0.997 [0.985, 1.000] |

Sweep over layers and positions, pooling `turn`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | close | 0.968 [0.942, 0.988] | 0.948 | n/a | n/a |
| 26 | open | 0.980 [0.959, 0.996] | 0.962 | n/a | n/a |
| 38 | close | 0.973 [0.952, 0.991] | 0.964 | n/a | n/a |
| 38 | open | 0.975 [0.954, 0.992] | 0.966 | n/a | n/a |
| 51 | close | 0.967 [0.937, 0.992] | 0.945 | n/a | n/a |
| 51 | open | 0.981 [0.965, 0.993] | 0.979 | n/a | n/a |

