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

Layer 38, position `close` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw |
|---|---|---|
| All authors, pooled probe (B.I) turns 0.822 | 193 / 180 | 0.882 [0.848, 0.914] |
| Within: DeepSeek V4.1 Flash (A) turns 0.841 | 42 / 43 | 0.884 [0.796, 0.954] |
| Within: GPT-5 mini (A) turns 0.806 | 30 / 30 | 0.832 [0.716, 0.923] |
| Within: Claude Haiku 5.5 (A) turns 0.886 | 30 / 30 | 0.942 [0.860, 0.998] |
| Within: GPT-6 Luna (A) turns 0.691 | 39 / 39 | 0.794 [0.682, 0.893] |
| Within: Qwen3.8-27B (A) turns 0.894 | 52 / 38 | 0.943 [0.892, 0.983] |
| Leave out: DeepSeek V4.1 Flash (B.II) turns 0.795 | 42 / 43 | 0.812 [0.707, 0.907] |
| Leave out: GPT-5 mini (B.II) turns 0.674 | 30 / 30 | 0.723 [0.578, 0.849] |
| Leave out: Claude Haiku 5.5 (B.II) turns 0.800 | 30 / 30 | 0.928 [0.855, 0.982] |
| Leave out: GPT-6 Luna (B.II) turns 0.654 | 39 / 39 | 0.714 [0.595, 0.822] |
| Leave out: Qwen3.8-27B (B.II) turns 0.704 | 52 / 38 | 0.878 [0.805, 0.942] |

Sweep over layers and positions, pooling `turn`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | close | 0.871 [0.835, 0.905] | 0.849 | n/a | n/a |
| 26 | open | 0.875 [0.839, 0.908] | 0.852 | n/a | n/a |
| 38 | close | 0.882 [0.848, 0.914] | 0.865 | n/a | n/a |
| 38 | open | 0.878 [0.841, 0.910] | 0.860 | n/a | n/a |
| 51 | close | 0.870 [0.832, 0.903] | 0.865 | n/a | n/a |
| 51 | open | 0.866 [0.829, 0.899] | 0.860 | n/a | n/a |


## Both arms pooled (secondary)

### Cheating AUROC, pooling `turn`

Layer 38, position `open` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw |
|---|---|---|
| All authors, pooled probe (B.I) turns 0.838 | 343 / 257 | 0.910 [0.887, 0.932] |
| Within: DeepSeek V4.1 Flash (A) turns 0.893 | 72 / 48 | 0.944 [0.895, 0.982] |
| Within: GPT-5 mini (A) turns 0.791 | 60 / 60 | 0.856 [0.781, 0.919] |
| Within: Claude Haiku 5.5 (A) turns 0.877 | 60 / 60 | 0.943 [0.897, 0.979] |
| Within: GPT-6 Luna (A) turns 0.721 | 69 / 51 | 0.837 [0.758, 0.901] |
| Within: Qwen3.8-27B (A) turns 0.899 | 82 / 38 | 0.941 [0.882, 0.984] |
| Leave out: DeepSeek V4.1 Flash (B.II) turns 0.731 | 72 / 48 | 0.792 [0.705, 0.871] |
| Leave out: GPT-5 mini (B.II) turns 0.611 | 60 / 60 | 0.666 [0.567, 0.761] |
| Leave out: Claude Haiku 5.5 (B.II) turns 0.685 | 60 / 60 | 0.715 [0.620, 0.806] |
| Leave out: GPT-6 Luna (B.II) turns 0.700 | 69 / 51 | 0.794 [0.702, 0.868] |
| Leave out: Qwen3.8-27B (B.II) turns 0.806 | 82 / 38 | 0.960 [0.921, 0.987] |

Sweep over layers and positions, pooling `turn`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | close | 0.894 [0.869, 0.919] | 0.881 | n/a | n/a |
| 26 | open | 0.905 [0.880, 0.929] | 0.897 | n/a | n/a |
| 38 | close | 0.907 [0.883, 0.929] | 0.898 | n/a | n/a |
| 38 | open | 0.910 [0.887, 0.932] | 0.904 | n/a | n/a |
| 51 | close | 0.892 [0.867, 0.916] | 0.886 | n/a | n/a |
| 51 | open | 0.886 [0.858, 0.910] | 0.881 | n/a | n/a |


## Cheat arm (secondary)

### Cheating AUROC, pooling `turn`

Layer 26, position `open` (the configuration with the highest pooled raw AUROC in this arm; every configuration is in `auroc.csv`). Cells: held-out AUROC [95% bootstrap over runs].

| scope | cheat / no cheat | raw |
|---|---|---|
| All authors, pooled probe (B.I) turns 0.879 | 150 / 77 | 0.938 [0.906, 0.967] |
| Within: DeepSeek V4.1 Flash (A) turns 0.835 | 30 / 5 | 0.867 [0.565, 1.000] |
| Within: GPT-5 mini (A) turns 0.746 | 30 / 30 | 0.826 [0.711, 0.917] |
| Within: Claude Haiku 5.5 (A) turns 0.842 | 30 / 30 | 0.857 [0.751, 0.950] |
| Within: GPT-6 Luna (A) turns 0.768 | 30 / 12 | 0.883 [0.750, 0.989] |
| Leave out: DeepSeek V4.1 Flash (B.II) turns 0.787 | 30 / 5 | 0.967 [0.903, 1.000] |
| Leave out: GPT-5 mini (B.II) turns 0.534 | 30 / 30 | 0.582 [0.424, 0.733] |
| Leave out: Claude Haiku 5.5 (B.II) turns 0.796 | 30 / 30 | 0.826 [0.708, 0.921] |
| Leave out: GPT-6 Luna (B.II) turns 0.819 | 30 / 12 | 0.942 [0.857, 0.991] |

Sweep over layers and positions, pooling `turn`, raw features:

| layer | position | pooled AUROC | within-author mean | null 95th pct | null p |
|---|---|---|---|---|---|
| 26 | close | 0.916 [0.878, 0.951] | 0.888 | n/a | n/a |
| 26 | open | 0.938 [0.906, 0.967] | 0.916 | n/a | n/a |
| 38 | close | 0.915 [0.873, 0.954] | 0.880 | n/a | n/a |
| 38 | open | 0.928 [0.890, 0.959] | 0.910 | n/a | n/a |
| 51 | close | 0.927 [0.889, 0.963] | 0.896 | n/a | n/a |
| 51 | open | 0.919 [0.880, 0.953] | 0.902 | n/a | n/a |

