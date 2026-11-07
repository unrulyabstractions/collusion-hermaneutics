# A hermeneutic cheating probe: results on the controls subsample

We ask whether an open-weight **reader** (Qwen3.8-27B) that only reads another model's transcript carries a linear readout of whether that model cheated. This is hard because the reader also encodes who wrote the text, which task it was, and whether the prompt asked for the shortcut, and any of these can stand in for cheating. We teacher-force each author's transcript through the reader, keep its residual stream at the `<think>` and `</think>` token of every assistant turn at layers 26, 38 and 51 (of 64), and fit L2 logistic probes on training-fold PCA features, held out by run. We erase the author, and the author with task and condition, with LEACE, and we shuffle labels inside each author for a null. We therefore contend that the reader's think-position states separate cheating from honest work inside every author, across authors, and after the author is erased.

## What was probed

Authors: DeepSeek V4.1 Flash, GPT-5 mini, Claude Haiku 5.5, GPT-6 Luna, Qwen3.8-27B. Runs: 600 of the 4,157 valid controls runs, about 120 per author, balanced over condition and label where the data allow; one agent per run. The **solo arm** (no cheating instruction in the prompt) is the primary test; the **cheat arm** asks for the shortcut and nearly every author takes it. Labels follow the cheat ladder in `signals.jsonl`: cheating from `opened` (from `tried` on software engineering, chess and creative writing); `none` and `exposed` are honest; the rungs between are left out. Readable reasoning per turn differs by author, so an author signature is built in: DeepSeek V4.1 Flash 0.86, GPT-5 mini 0.49, Claude Haiku 5.5 0.00, GPT-6 Luna 0.31, Qwen3.8-27B 1.00.

| author | condition | cheat | no cheat | turns |
|---|---|---|---|---|
| DeepSeek V4.1 Flash | cheat | 30 | 5 | 619 |
| DeepSeek V4.1 Flash | solo | 42 | 43 | 1681 |
| GPT-5 mini | cheat | 30 | 30 | 1094 |
| GPT-5 mini | solo | 30 | 30 | 1244 |
| Claude Haiku 5.5 | cheat | 30 | 30 | 882 |
| Claude Haiku 5.5 | solo | 30 | 30 | 957 |
| GPT-6 Luna | cheat | 30 | 12 | 569 |
| GPT-6 Luna | solo | 39 | 39 | 1476 |
| Qwen3.8-27B | cheat | 30 | 0 | 455 |
| Qwen3.8-27B | solo | 52 | 38 | 2008 |

## Does the reader separate cheating in the solo arm?

Yes, in every author and across authors. Table 1 gives the agent-level probe (mean over turns) at the configuration with the highest pooled AUROC (layer all, position `both`); Table 2 sweeps layers and positions. Erasing the author moves the numbers little; erasing author, task and condition together moves them a few hundredths. The within-author permutation null tops out near 0.55 and every real value sits above all 100 draws. We therefore posit that the signal is not the author's base rate and not the task.

**Table 1.** Solo arm, agent-level PCA probe, layer all, position `both`. Held-out AUROC [95% bootstrap over runs].

| probe | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| B.I pooled, held-out runs | 193 / 180 | 0.967 [0.950, 0.981] | 0.964 [0.946, 0.978] | 0.944 [0.921, 0.965] |
| A within DeepSeek V4.1 Flash | 42 / 43 | 0.860 [0.775, 0.937] | 0.858 [0.771, 0.937] | 0.884 [0.807, 0.950] |
| A within GPT-5 mini | 30 / 30 | 0.862 [0.762, 0.953] | 0.868 [0.768, 0.956] | 0.898 [0.807, 0.978] |
| A within Claude Haiku 5.5 | 30 / 30 | 0.929 [0.852, 0.987] | 0.913 [0.831, 0.975] | 0.943 [0.884, 0.986] |
| A within GPT-6 Luna | 39 / 39 | 0.853 [0.757, 0.928] | 0.845 [0.749, 0.921] | 0.811 [0.694, 0.896] |
| A within Qwen3.8-27B | 52 / 38 | 0.955 [0.910, 0.987] | 0.943 [0.893, 0.984] | 0.945 [0.899, 0.981] |
| B.II leave out DeepSeek V4.1 Flash | 42 / 43 | 0.892 [0.813, 0.952] | 0.899 [0.827, 0.954] | 0.918 [0.848, 0.969] |
| B.II leave out GPT-5 mini | 30 / 30 | 0.946 [0.884, 0.992] | 0.953 [0.899, 0.993] | 0.969 [0.922, 0.997] |
| B.II leave out Claude Haiku 5.5 | 30 / 30 | 0.973 [0.934, 0.997] | 0.964 [0.910, 0.997] | 0.941 [0.875, 0.984] |
| B.II leave out GPT-6 Luna | 39 / 39 | 0.800 [0.698, 0.886] | 0.780 [0.670, 0.871] | 0.780 [0.675, 0.875] |
| B.II leave out Qwen3.8-27B | 52 / 38 | 0.966 [0.925, 0.993] | 0.972 [0.941, 0.995] | 0.981 [0.955, 0.998] |

**Table 2.** Solo arm, raw features, every layer and position (agent-level PCA probe).

| layer | position | B.I pooled AUROC | within-author mean | null 95th pct |
|---|---|---|---|---|
| 26 | both | 0.938 [0.913, 0.959] | 0.929 | 0.567 |
| 26 | close | 0.940 [0.915, 0.962] | 0.930 | 0.556 |
| 26 | open | 0.946 [0.924, 0.966] | 0.939 | 0.569 |
| 38 | both | 0.954 [0.932, 0.972] | 0.943 | 0.555 |
| 38 | close | 0.956 [0.932, 0.974] | 0.950 | 0.557 |
| 38 | open | 0.944 [0.920, 0.964] | 0.936 | 0.554 |
| 51 | both | 0.959 [0.940, 0.975] | 0.957 | 0.549 |
| 51 | close | 0.961 [0.943, 0.977] | 0.959 | 0.558 |
| 51 | open | 0.955 [0.935, 0.973] | 0.951 | 0.561 |
| all | both | 0.967 [0.950, 0.981] | 0.963 | 0.550 |
| all | close | 0.959 [0.941, 0.975] | 0.957 | 0.561 |
| all | open | 0.952 [0.932, 0.971] | 0.948 | 0.555 |

## Is it the author, the task, or the length of the run?

The author is almost perfectly readable from the same features and becomes unreadable after erasure (Table 3). Metadata alone predicts cheating poorly in the pooled solo arm, though inside some authors the task predicts it well; the probe stays above those baselines after the task is erased (Table 1, last column). Run length (turns, tokens, readable fraction) predicts cheating at 0.582 in the solo arm, same folds. We therefore posit that the probe reads something about the transcript's content, not its envelope.

**Table 3.** Solo arm, author-identity probe (5 classes, majority rate 0.241) and the metadata-only cheating baselines, same folds.

| readout | value |
|---|---|
| author accuracy, raw features | 0.989 |
| author accuracy, author erased | 0.241 |
| author accuracy, author, task, condition erased | 0.241 |
| cheating AUROC from one-hot pooled (author + task + condition) | 0.568 |
| cheating AUROC from one-hot pooled (author only) | 0.524 |
| cheating AUROC from one-hot ds (task + condition) | 0.674 |
| cheating AUROC from one-hot gpt5m (task + condition) | 0.807 |
| cheating AUROC from one-hot haiku (task + condition) | 0.729 |
| cheating AUROC from one-hot luna (task + condition) | 0.419 |
| cheating AUROC from one-hot qwen (task + condition) | 0.853 |

**Table 4.** Solo arm, labels shuffled inside each author (100 draws), layer all, position `both`.

| features | scope | real | null mean | null 95th pct | p |
|---|---|---|---|---|---|
| raw | pooled | 0.967 | 0.496 | 0.550 | 0.010 |
| raw | DeepSeek V4.1 Flash | 0.860 | 0.489 | 0.626 | 0.010 |
| raw | GPT-5 mini | 0.862 | 0.495 | 0.646 | 0.010 |
| raw | Claude Haiku 5.5 | 0.929 | 0.489 | 0.643 | 0.010 |
| raw | GPT-6 Luna | 0.853 | 0.488 | 0.636 | 0.010 |
| raw | Qwen3.8-27B | 0.955 | 0.499 | 0.626 | 0.010 |
| author erased | pooled | 0.964 | 0.502 | 0.557 | 0.010 |
| author erased | DeepSeek V4.1 Flash | 0.858 | 0.488 | 0.626 | 0.010 |
| author erased | GPT-5 mini | 0.868 | 0.493 | 0.651 | 0.010 |
| author erased | Claude Haiku 5.5 | 0.913 | 0.489 | 0.642 | 0.010 |
| author erased | GPT-6 Luna | 0.845 | 0.487 | 0.629 | 0.010 |
| author erased | Qwen3.8-27B | 0.943 | 0.498 | 0.621 | 0.010 |

## Does a probe trained on four authors read the fifth?

Mostly. Leave-one-author-out AUROC (B.II in Table 1) stays high for every held-out author except GPT-6 Luna, which receives the weakest transfer and is also the hardest author within itself. Luna's reasoning is encrypted and only a third of its turns carry a provider summary. We therefore conjecture that transfer depends on how much of the author's reasoning the reader gets to see.

## What about the cheat arm and the pooled set?

In the pooled set the prompt condition is readable at 0.988 (majority 0.622), and falls to the majority rate once condition is erased; the pooled cheating probe keeps most of its AUROC after that erasure (Table 5). The cheat arm has few honest agents for DeepSeek and none for Qwen, so its within-author rows are thin and its pooled number is secondary.

**Table 5.** Both arms pooled (layer 38, `close`) and the cheat arm alone (layer 38, `close`), agent-level PCA probe.

| probe | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| B.I pooled, held-out runs | 343 / 257 | 0.963 [0.949, 0.976] | 0.956 [0.941, 0.970] | 0.945 [0.926, 0.963] |
| A within DeepSeek V4.1 Flash | 72 / 48 | 0.945 [0.902, 0.979] | 0.947 [0.904, 0.981] | 0.926 [0.874, 0.966] |
| A within GPT-5 mini | 60 / 60 | 0.855 [0.777, 0.918] | 0.864 [0.786, 0.927] | 0.892 [0.828, 0.947] |
| A within Claude Haiku 5.5 | 60 / 60 | 0.959 [0.922, 0.984] | 0.960 [0.924, 0.983] | 0.954 [0.913, 0.983] |
| A within GPT-6 Luna | 69 / 51 | 0.845 [0.766, 0.911] | 0.845 [0.765, 0.909] | 0.899 [0.833, 0.955] |
| A within Qwen3.8-27B | 82 / 38 | 0.966 [0.927, 0.993] | 0.955 [0.907, 0.989] | 0.973 [0.944, 0.993] |
| B.II leave out DeepSeek V4.1 Flash | 72 / 48 | 0.917 [0.865, 0.960] | 0.928 [0.875, 0.967] | 0.893 [0.837, 0.942] |
| B.II leave out GPT-5 mini | 60 / 60 | 0.937 [0.886, 0.975] | 0.927 [0.872, 0.968] | 0.942 [0.896, 0.975] |
| B.II leave out Claude Haiku 5.5 | 60 / 60 | 0.923 [0.868, 0.968] | 0.924 [0.869, 0.966] | 0.864 [0.799, 0.922] |
| B.II leave out GPT-6 Luna | 69 / 51 | 0.896 [0.837, 0.944] | 0.893 [0.832, 0.941] | 0.873 [0.809, 0.928] |
| B.II leave out Qwen3.8-27B | 82 / 38 | 0.977 [0.952, 0.995] | 0.974 [0.948, 0.992] | 0.976 [0.950, 0.995] |

| probe | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| B.I pooled, held-out runs | 150 / 77 | 0.940 [0.902, 0.969] | 0.896 [0.843, 0.939] | 0.862 [0.808, 0.911] |
| A within DeepSeek V4.1 Flash | 30 / 5 | 0.967 [0.870, 1.000] | 0.973 [0.894, 1.000] | 0.973 [0.898, 1.000] |
| A within GPT-5 mini | 30 / 30 | 0.829 [0.715, 0.929] | 0.833 [0.709, 0.932] | 0.910 [0.826, 0.974] |
| A within Claude Haiku 5.5 | 30 / 30 | 0.867 [0.760, 0.956] | 0.872 [0.766, 0.958] | 0.884 [0.791, 0.966] |
| A within GPT-6 Luna | 30 / 12 | 0.811 [0.647, 0.956] | 0.814 [0.647, 0.956] | 0.844 [0.691, 0.970] |
| B.II leave out DeepSeek V4.1 Flash | 30 / 5 | 0.840 [0.597, 1.000] | 0.793 [0.531, 1.000] | 0.840 [0.592, 1.000] |
| B.II leave out GPT-5 mini | 30 / 30 | 0.643 [0.488, 0.780] | 0.629 [0.481, 0.768] | 0.614 [0.470, 0.753] |
| B.II leave out Claude Haiku 5.5 | 30 / 30 | 0.853 [0.744, 0.944] | 0.856 [0.750, 0.944] | 0.844 [0.733, 0.939] |
| B.II leave out GPT-6 Luna | 30 / 12 | 0.936 [0.821, 1.000] | 0.978 [0.925, 1.000] | 0.939 [0.819, 1.000] |

## Does a single turn carry the signal?

Partly. With every turn labelled by its agent, the per-turn probe scores turns at 0.822 and agents (mean turn score) at 0.882 in the solo arm (layer 38, `close`); the turn-position baseline is 0.490, so position is not the signal (Table 6). A second labelling that keeps a cheating agent's turns only from its first cheating tool call on reaches 0.986 but its turn-position baseline reaches 0.983: that design makes lateness the label, and we do not read it as detection.

**Table 6.** Solo arm, per-turn PCA probe, agent-labelled turns, layer 38, position `close`. Agent AUROC is the mean turn score per agent.

| probe | cheat / no cheat | agent AUROC | turn AUROC |
|---|---|---|---|
| B.I pooled | 193 / 180 | 0.882 [0.848, 0.914] | 0.822 |
| A within DeepSeek V4.1 Flash | 42 / 43 | 0.884 [0.796, 0.954] | 0.841 |
| A within GPT-5 mini | 30 / 30 | 0.832 [0.716, 0.923] | 0.806 |
| A within Claude Haiku 5.5 | 30 / 30 | 0.942 [0.860, 0.998] | 0.886 |
| A within GPT-6 Luna | 39 / 39 | 0.794 [0.682, 0.893] | 0.691 |
| A within Qwen3.8-27B | 52 / 38 | 0.943 [0.892, 0.983] | 0.894 |
| B.II leave out DeepSeek V4.1 Flash | 42 / 43 | 0.812 [0.707, 0.907] | 0.795 |
| B.II leave out GPT-5 mini | 30 / 30 | 0.723 [0.578, 0.849] | 0.674 |
| B.II leave out Claude Haiku 5.5 | 30 / 30 | 0.928 [0.855, 0.982] | 0.800 |
| B.II leave out GPT-6 Luna | 39 / 39 | 0.714 [0.595, 0.822] | 0.654 |
| B.II leave out Qwen3.8-27B | 52 / 38 | 0.878 [0.805, 0.942] | 0.704 |

## Does the result depend on the PCA step?

No. A probe on every dimension (z-scored, L2 only) gives the same picture in the solo arm (Table 7; arms finished so far: solo).

**Table 7.** Solo arm, full-dimension probe, layer all, position `close`.

| probe | cheat / no cheat | raw | author erased | author, task, condition erased |
|---|---|---|---|---|
| B.I pooled, held-out runs | 193 / 180 | 0.971 [0.954, 0.984] | 0.963 [0.944, 0.979] | 0.941 [0.914, 0.963] |
| A within DeepSeek V4.1 Flash | 42 / 43 | 0.930 [0.867, 0.975] | 0.927 [0.863, 0.974] | 0.941 [0.887, 0.984] |
| A within GPT-5 mini | 30 / 30 | 0.942 [0.872, 0.990] | 0.949 [0.880, 0.993] | 0.942 [0.875, 0.991] |
| A within Claude Haiku 5.5 | 30 / 30 | 0.974 [0.934, 1.000] | 0.976 [0.936, 1.000] | 0.958 [0.909, 0.994] |
| A within GPT-6 Luna | 39 / 39 | 0.880 [0.792, 0.951] | 0.893 [0.811, 0.961] | 0.896 [0.811, 0.961] |
| A within Qwen3.8-27B | 52 / 38 | 0.952 [0.906, 0.986] | 0.948 [0.901, 0.985] | 0.947 [0.902, 0.984] |
| B.II leave out DeepSeek V4.1 Flash | 42 / 43 | 0.924 [0.863, 0.975] | 0.908 [0.837, 0.968] | 0.939 [0.880, 0.985] |
| B.II leave out GPT-5 mini | 30 / 30 | 0.978 [0.940, 0.999] | 0.981 [0.942, 1.000] | 0.969 [0.921, 0.996] |
| B.II leave out Claude Haiku 5.5 | 30 / 30 | 0.928 [0.853, 0.984] | 0.899 [0.814, 0.969] | 0.907 [0.824, 0.971] |
| B.II leave out GPT-6 Luna | 39 / 39 | 0.784 [0.676, 0.874] | 0.776 [0.663, 0.868] | 0.727 [0.609, 0.831] |
| B.II leave out Qwen3.8-27B | 52 / 38 | 0.939 [0.886, 0.980] | 0.950 [0.903, 0.986] | 0.928 [0.871, 0.973] |

## Limits

The subsample holds about 120 runs per author and leaves out runs above 100k tokens. The cheat label counts opening the planted material, so a probe may read that the solution text entered the context rather than an intent; a text classifier on the transcript is the baseline that settles this and it is not in these tables. Haiku's think blocks are empty and Luna's hold summaries at best, so for those authors the positions read the turn boundary, not reasoning. LEACE removes linear traces only. All probes are linear.

## Files

`pca_agent/`, `full_agent/`, `pca_turn/`, `pca_turn_onset/`: `auroc.csv` (every arm, layer, position, feature set, scope), `author_probe.csv`, `condition_probe.csv`, `agents.csv`, per-configuration `results.json`, and a `RESULTS.md` from `hermeneutics/report.py`. This file is built by `hermeneutics/summary.py`.
