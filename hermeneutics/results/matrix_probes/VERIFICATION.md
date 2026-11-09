# Verification of the probe matrix

Run 2026-11-09 14:26 PST by `python -m hermeneutics.matrix_probes verify`.

## 1. One AUROC per scheme recomputed from the saved scores

For each scheme the B.I main-probe row (or, when B.I did not run, the first A or B.II row with an AUROC) is recomputed twice from `scores/<label>/<scheme>.csv.gz` and compared with cells.csv: with sklearn.metrics.roc_auc_score, and by hand as the share of positive-negative pairs the probe orders correctly (ties count half), counted pair by pair with no library AUROC code. The saved scores keep 6 significant digits, so scores can tie on disk that did not tie in the run; a row matches when the difference is within the largest change those ties can cause (0.5 per tied positive-negative pair over positives x negatives; 0 when no score ties) and the counts are equal.

| label | scheme | row | cells.csv | sklearn | pair count | difference | tie bound | n/pos | result |
|---|---|---|---|---|---|---|---|---|---|
| cheat | alone: controls | B.I pooled think | 0.959569 | 0.959569 | 0.959569 | 0.0e+00 | 0.0e+00 | 600/343 | match |
| cheat | alone: board_seed | B.I pooled think | 0.880037 | 0.880037 | 0.880037 | 0.0e+00 | 0.0e+00 | 134/56 | match |
| cheat | alone: solo_design | B.I pooled think | 0.914107 | 0.914107 | 0.914107 | 1.1e-16 | 0.0e+00 | 177/84 | match |
| cheat | alone: spar_team | B.I pooled think | 0.927117 | 0.927117 | 0.927117 | 1.1e-16 | 0.0e+00 | 450/109 | match |
| cheat | alone: spar_mixed | B.I pooled think | 0.733503 | 0.733503 | 0.733503 | 0.0e+00 | 0.0e+00 | 209/12 | match |
| cheat | alone: spar_expl | B.I pooled think | 0.955186 | 0.955186 | 0.955186 | 0.0e+00 | 0.0e+00 | 313/71 | match |
| cheat | alone: spar_solo | B.I pooled think | 0.961900 | 0.961900 | 0.961900 | 1.1e-16 | 0.0e+00 | 90/47 | match |
| cheat | alone: goodfire | B.I pooled think | 0.921044 | 0.921044 | 0.921044 | 0.0e+00 | 0.0e+00 | 2833/2083 | match |
| cheat | alone: cheatbench | B.I pooled think | 0.950580 | 0.950580 | 0.950580 | 0.0e+00 | 0.0e+00 | 224/134 | match |
| cheat | alone: evilgenie | B.I pooled think | 0.871672 | 0.871672 | 0.871672 | 0.0e+00 | 0.0e+00 | 266/133 | match |
| cheat | ours | B.I pooled think | 0.950285 | 0.950285 | 0.950285 | 1.1e-16 | 0.0e+00 | 1973/722 | match |
| cheat | external | B.I pooled think | 0.921150 | 0.921150 | 0.921150 | 0.0e+00 | 0.0e+00 | 3323/2350 | match |
| cheat | all | B.I pooled think | 0.919874 | 0.919874 | 0.919874 | 2.2e-07 | 2.2e-07 | 5296/3072 | match |
| cheat | leave out: controls | B.I pooled think | 0.684076 | 0.684076 | 0.684076 | 1.1e-16 | 0.0e+00 | 600/343 | match |
| cheat | leave out: board_seed | B.I pooled think | 0.465201 | 0.465201 | 0.465201 | 0.0e+00 | 0.0e+00 | 134/56 | match |
| cheat | leave out: solo_design | B.I pooled think | 0.821301 | 0.821301 | 0.821301 | 1.1e-16 | 0.0e+00 | 177/84 | match |
| cheat | leave out: spar_team | B.I pooled think | 0.791923 | 0.791923 | 0.791923 | 0.0e+00 | 0.0e+00 | 450/109 | match |
| cheat | leave out: spar_mixed | B.I pooled think | 0.914975 | 0.914975 | 0.914975 | 0.0e+00 | 0.0e+00 | 209/12 | match |
| cheat | leave out: spar_expl | B.I pooled think | 0.906414 | 0.906414 | 0.906414 | 0.0e+00 | 0.0e+00 | 313/71 | match |
| cheat | leave out: spar_solo | B.I pooled think | 0.852053 | 0.852053 | 0.852053 | 0.0e+00 | 0.0e+00 | 90/47 | match |
| cheat | leave out: goodfire | B.I pooled think | 0.647860 | 0.647860 | 0.647860 | 3.2e-07 | 9.6e-07 | 2833/2083 | match |
| cheat | leave out: cheatbench | B.I pooled think | 0.805141 | 0.805141 | 0.805141 | 0.0e+00 | 0.0e+00 | 224/134 | match |
| cheat | leave out: evilgenie | B.I pooled think | 0.652100 | 0.652100 | 0.652100 | 0.0e+00 | 0.0e+00 | 266/133 | match |
| cheat | ours to external | B.I pooled think | 0.630265 | 0.630265 | 0.630265 | 0.0e+00 | 0.0e+00 | 3323/2350 | match |
| cheat | external to ours | B.I pooled think | 0.396628 | 0.396628 | 0.396628 | 5.5e-07 | 1.7e-06 | 1973/722 | match |
| coord | alone: board_seed | B.I pooled think | 0.893862 | 0.893862 | 0.893862 | 0.0e+00 | 0.0e+00 | 134/87 | match |
| coord | alone: solo_design | B.I pooled think | 0.747255 | 0.747255 | 0.747255 | 1.1e-16 | 0.0e+00 | 177/38 | match |
| coord | alone: spar_team | B.I pooled think | 0.908867 | 0.908867 | 0.908867 | 0.0e+00 | 0.0e+00 | 450/406 | match |
| coord | alone: spar_mixed | B.I pooled think | 0.600000 | 0.600000 | 0.600000 | 0.0e+00 | 0.0e+00 | 209/205 | match |
| coord | ours | B.I pooled think | 0.969400 | 0.969400 | 0.969400 | 0.0e+00 | 0.0e+00 | 970/736 | match |
| coord | external |  |  |  |  | no own scores (identical to another scheme or no cell) |
| coord | all |  |  |  |  | no own scores (identical to another scheme or no cell) |
| coord | leave out: board_seed | B.I pooled think | 0.941795 | 0.941795 | 0.941795 | 0.0e+00 | 0.0e+00 | 134/87 | match |
| coord | leave out: solo_design | B.I pooled think | 0.758046 | 0.758046 | 0.758046 | 1.1e-16 | 0.0e+00 | 177/38 | match |
| coord | leave out: spar_team | B.I pooled think | 0.862965 | 0.862965 | 0.862965 | 0.0e+00 | 0.0e+00 | 450/406 | match |
| coord | leave out: spar_mixed | B.I pooled think | 0.853659 | 0.853659 | 0.853659 | 0.0e+00 | 0.0e+00 | 209/205 | match |
| coord | ours to external |  |  |  |  | no own scores (identical to another scheme or no cell) |
| coord | external to ours |  |  |  |  | no own scores (identical to another scheme or no cell) |
| collude | alone: board_seed | B.I pooled think | 0.901163 | 0.901163 | 0.901163 | 0.0e+00 | 0.0e+00 | 134/48 | match |
| collude | alone: spar_team | B.I pooled think | 0.909551 | 0.909551 | 0.909551 | 0.0e+00 | 0.0e+00 | 450/67 | match |
| collude | alone: spar_mixed | B.I pooled think | 0.966761 | 0.966761 | 0.966761 | 0.0e+00 | 0.0e+00 | 209/7 | match |
| collude | ours | B.I pooled think | 0.932924 | 0.932924 | 0.932924 | 0.0e+00 | 0.0e+00 | 793/122 | match |
| collude | external |  |  |  |  | no own scores (identical to another scheme or no cell) |
| collude | all |  |  |  |  | no own scores (identical to another scheme or no cell) |
| collude | leave out: board_seed | B.I pooled think | 0.887597 | 0.887597 | 0.887597 | 0.0e+00 | 0.0e+00 | 134/48 | match |
| collude | leave out: spar_team | B.I pooled think | 0.814232 | 0.814232 | 0.814232 | 0.0e+00 | 0.0e+00 | 450/67 | match |
| collude | leave out: spar_mixed | B.I pooled think | 0.906648 | 0.906648 | 0.906648 | 0.0e+00 | 0.0e+00 | 209/7 | match |
| collude | ours to external |  |  |  |  | no own scores (identical to another scheme or no cell) |
| collude | external to ours |  |  |  |  | no own scores (identical to another scheme or no cell) |

## 2. Runs and folds

Cross-validated score sets (B.I and A, main probe): 110; with a run in more than one fold: 0. B.II score sets: 153; holding other than exactly the one held-out author: 0.

Every B.II fold (from `folds/<label>/<scheme>_BII.json.gz`): test agents, test runs, training runs, runs in both, training runs that hold any agent of the held-out author (must be 0, so no run-mate of that author trains), authors in the test set.

| label/scheme | held-out author | test agents | test runs | train runs | runs in both | train runs holding the author | test authors | result |
|---|---|---|---|---|---|---|---|---|
| cheat/all | Claude 3.7 Sonnet | 60 | 60 | 4204 | 0 | 0 | Claude 3.7 Sonnet | ok |
| cheat/all | Claude Fable 5 | 10 | 5 | 4259 | 0 | 0 | Claude Fable 5 | ok |
| cheat/all | Claude Haiku 5.5 | 364 | 235 | 4029 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/all | Claude Opus 4.6 | 111 | 111 | 4153 | 0 | 0 | Claude Opus 4.6 | ok |
| cheat/all | Claude Opus 4.8 | 6 | 3 | 4261 | 0 | 0 | Claude Opus 4.8 | ok |
| cheat/all | DeepSeek V4.1 Flash | 596 | 292 | 3972 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/all | GLM 5.2 | 1006 | 1006 | 3258 | 0 | 0 | GLM 5.2 | ok |
| cheat/all | GPT-5 mini | 120 | 120 | 4144 | 0 | 0 | GPT-5 mini | ok |
| cheat/all | GPT-5.3 Codex | 77 | 56 | 4208 | 0 | 0 | GPT-5.3 Codex | ok |
| cheat/all | GPT-5.4 | 18 | 18 | 4246 | 0 | 0 | GPT-5.4 | ok |
| cheat/all | GPT-5.6 Luna | 28 | 14 | 4250 | 0 | 0 | GPT-5.6 Luna | ok |
| cheat/all | GPT-5.6 Sol | 24 | 12 | 4252 | 0 | 0 | GPT-5.6 Sol | ok |
| cheat/all | GPT-6 Luna | 748 | 337 | 3927 | 0 | 0 | GPT-6 Luna | ok |
| cheat/all | Gemini 3.1 Pro | 34 | 17 | 4247 | 0 | 0 | Gemini 3.1 Pro | ok |
| cheat/all | Gemini 3.5 Flash | 56 | 28 | 4236 | 0 | 0 | Gemini 3.5 Flash | ok |
| cheat/all | Grok 4.5 | 40 | 20 | 4244 | 0 | 0 | Grok 4.5 | ok |
| cheat/all | Kimi K2.7 Code | 18 | 9 | 4255 | 0 | 0 | Kimi K2.7 Code | ok |
| cheat/all | Kimi K3 | 943 | 939 | 3325 | 0 | 0 | Kimi K3 | ok |
| cheat/all | Qwen3.8 27B | 120 | 120 | 4144 | 0 | 0 | Qwen3.8 27B | ok |
| cheat/all | Qwen3.8 Max | 898 | 894 | 3370 | 0 | 0 | Qwen3.8 Max | ok |
| cheat/alone_cheatbench | Claude 3.7 Sonnet | 60 | 60 | 164 | 0 | 0 | Claude 3.7 Sonnet | ok |
| cheat/alone_cheatbench | Claude Opus 4.6 | 111 | 111 | 113 | 0 | 0 | Claude Opus 4.6 | ok |
| cheat/alone_cheatbench | GPT-5.3 Codex | 35 | 35 | 189 | 0 | 0 | GPT-5.3 Codex | ok |
| cheat/alone_cheatbench | GPT-5.4 | 18 | 18 | 206 | 0 | 0 | GPT-5.4 | ok |
| cheat/alone_controls | Claude Haiku 5.5 | 120 | 120 | 480 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/alone_controls | DeepSeek V4.1 Flash | 120 | 120 | 480 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/alone_controls | GPT-5 mini | 120 | 120 | 480 | 0 | 0 | GPT-5 mini | ok |
| cheat/alone_controls | GPT-6 Luna | 120 | 120 | 480 | 0 | 0 | GPT-6 Luna | ok |
| cheat/alone_controls | Qwen3.8 27B | 120 | 120 | 480 | 0 | 0 | Qwen3.8 27B | ok |
| cheat/alone_evilgenie | Claude Fable 5 | 10 | 5 | 128 | 0 | 0 | Claude Fable 5 | ok |
| cheat/alone_evilgenie | Claude Opus 4.8 | 6 | 3 | 130 | 0 | 0 | Claude Opus 4.8 | ok |
| cheat/alone_evilgenie | GPT-5.3 Codex | 42 | 21 | 112 | 0 | 0 | GPT-5.3 Codex | ok |
| cheat/alone_evilgenie | GPT-5.6 Luna | 28 | 14 | 119 | 0 | 0 | GPT-5.6 Luna | ok |
| cheat/alone_evilgenie | GPT-5.6 Sol | 24 | 12 | 121 | 0 | 0 | GPT-5.6 Sol | ok |
| cheat/alone_evilgenie | Gemini 3.1 Pro | 34 | 17 | 116 | 0 | 0 | Gemini 3.1 Pro | ok |
| cheat/alone_evilgenie | Gemini 3.5 Flash | 56 | 28 | 105 | 0 | 0 | Gemini 3.5 Flash | ok |
| cheat/alone_evilgenie | Grok 4.5 | 40 | 20 | 113 | 0 | 0 | Grok 4.5 | ok |
| cheat/alone_evilgenie | Kimi K2.7 Code | 18 | 9 | 124 | 0 | 0 | Kimi K2.7 Code | ok |
| cheat/alone_evilgenie | Kimi K3 | 8 | 4 | 129 | 0 | 0 | Kimi K3 | ok |
| cheat/alone_goodfire | GLM 5.2 | 1006 | 1006 | 1827 | 0 | 0 | GLM 5.2 | ok |
| cheat/alone_goodfire | Kimi K3 | 935 | 935 | 1898 | 0 | 0 | Kimi K3 | ok |
| cheat/alone_goodfire | Qwen3.8 Max | 892 | 892 | 1941 | 0 | 0 | Qwen3.8 Max | ok |
| cheat/alone_solo_design | DeepSeek V4.1 Flash | 90 | 30 | 29 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/alone_solo_design | GPT-6 Luna | 81 | 27 | 32 | 0 | 0 | GPT-6 Luna | ok |
| cheat/alone_spar_expl | Claude Haiku 5.5 | 36 | 12 | 84 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/alone_spar_expl | DeepSeek V4.1 Flash | 36 | 12 | 84 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/alone_spar_expl | GPT-6 Luna | 241 | 72 | 24 | 0 | 0 | GPT-6 Luna | ok |
| cheat/alone_spar_mixed | DeepSeek V4.1 Flash | 36 | 12 | 39 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/alone_spar_mixed | GPT-6 Luna | 126 | 43 | 8 | 0 | 0 | GPT-6 Luna | ok |
| cheat/alone_spar_solo | Claude Haiku 5.5 | 30 | 30 | 60 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/alone_spar_solo | GPT-6 Luna | 30 | 30 | 60 | 0 | 0 | GPT-6 Luna | ok |
| cheat/alone_spar_team | Claude Haiku 5.5 | 150 | 45 | 90 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/alone_spar_team | DeepSeek V4.1 Flash | 150 | 45 | 90 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/alone_spar_team | GPT-6 Luna | 150 | 45 | 90 | 0 | 0 | GPT-6 Luna | ok |
| cheat/external | Claude 3.7 Sonnet | 60 | 60 | 3130 | 0 | 0 | Claude 3.7 Sonnet | ok |
| cheat/external | Claude Fable 5 | 10 | 5 | 3185 | 0 | 0 | Claude Fable 5 | ok |
| cheat/external | Claude Opus 4.6 | 111 | 111 | 3079 | 0 | 0 | Claude Opus 4.6 | ok |
| cheat/external | Claude Opus 4.8 | 6 | 3 | 3187 | 0 | 0 | Claude Opus 4.8 | ok |
| cheat/external | GLM 5.2 | 1006 | 1006 | 2184 | 0 | 0 | GLM 5.2 | ok |
| cheat/external | GPT-5.3 Codex | 77 | 56 | 3134 | 0 | 0 | GPT-5.3 Codex | ok |
| cheat/external | GPT-5.4 | 18 | 18 | 3172 | 0 | 0 | GPT-5.4 | ok |
| cheat/external | GPT-5.6 Luna | 28 | 14 | 3176 | 0 | 0 | GPT-5.6 Luna | ok |
| cheat/external | GPT-5.6 Sol | 24 | 12 | 3178 | 0 | 0 | GPT-5.6 Sol | ok |
| cheat/external | Gemini 3.1 Pro | 34 | 17 | 3173 | 0 | 0 | Gemini 3.1 Pro | ok |
| cheat/external | Gemini 3.5 Flash | 56 | 28 | 3162 | 0 | 0 | Gemini 3.5 Flash | ok |
| cheat/external | Grok 4.5 | 40 | 20 | 3170 | 0 | 0 | Grok 4.5 | ok |
| cheat/external | Kimi K2.7 Code | 18 | 9 | 3181 | 0 | 0 | Kimi K2.7 Code | ok |
| cheat/external | Kimi K3 | 943 | 939 | 2251 | 0 | 0 | Kimi K3 | ok |
| cheat/external | Qwen3.8 Max | 892 | 892 | 2298 | 0 | 0 | Qwen3.8 Max | ok |
| cheat/external_to_ours | Claude Haiku 5.5 | 364 | 235 | 3190 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/external_to_ours | DeepSeek V4.1 Flash | 596 | 292 | 3190 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/external_to_ours | GPT-5 mini | 120 | 120 | 3190 | 0 | 0 | GPT-5 mini | ok |
| cheat/external_to_ours | GPT-6 Luna | 748 | 337 | 3190 | 0 | 0 | GPT-6 Luna | ok |
| cheat/external_to_ours | Qwen3.8 27B | 120 | 120 | 3190 | 0 | 0 | Qwen3.8 27B | ok |
| cheat/leave_out_board_seed | DeepSeek V4.1 Flash | 134 | 43 | 3972 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/leave_out_cheatbench | Claude 3.7 Sonnet | 60 | 60 | 4040 | 0 | 0 | Claude 3.7 Sonnet | ok |
| cheat/leave_out_cheatbench | Claude Opus 4.6 | 111 | 111 | 4040 | 0 | 0 | Claude Opus 4.6 | ok |
| cheat/leave_out_cheatbench | GPT-5.3 Codex | 35 | 35 | 4019 | 0 | 0 | GPT-5.3 Codex | ok |
| cheat/leave_out_cheatbench | GPT-5.4 | 18 | 18 | 4040 | 0 | 0 | GPT-5.4 | ok |
| cheat/leave_out_controls | Claude Haiku 5.5 | 120 | 120 | 3549 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/leave_out_controls | DeepSeek V4.1 Flash | 120 | 120 | 3492 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/leave_out_controls | GPT-5 mini | 120 | 120 | 3664 | 0 | 0 | GPT-5 mini | ok |
| cheat/leave_out_controls | GPT-6 Luna | 120 | 120 | 3447 | 0 | 0 | GPT-6 Luna | ok |
| cheat/leave_out_controls | Qwen3.8 27B | 120 | 120 | 3664 | 0 | 0 | Qwen3.8 27B | ok |
| cheat/leave_out_evilgenie | Claude Fable 5 | 10 | 5 | 4131 | 0 | 0 | Claude Fable 5 | ok |
| cheat/leave_out_evilgenie | Claude Opus 4.8 | 6 | 3 | 4131 | 0 | 0 | Claude Opus 4.8 | ok |
| cheat/leave_out_evilgenie | GPT-5.3 Codex | 42 | 21 | 4096 | 0 | 0 | GPT-5.3 Codex | ok |
| cheat/leave_out_evilgenie | GPT-5.6 Luna | 28 | 14 | 4131 | 0 | 0 | GPT-5.6 Luna | ok |
| cheat/leave_out_evilgenie | GPT-5.6 Sol | 24 | 12 | 4131 | 0 | 0 | GPT-5.6 Sol | ok |
| cheat/leave_out_evilgenie | Gemini 3.1 Pro | 34 | 17 | 4131 | 0 | 0 | Gemini 3.1 Pro | ok |
| cheat/leave_out_evilgenie | Gemini 3.5 Flash | 56 | 28 | 4131 | 0 | 0 | Gemini 3.5 Flash | ok |
| cheat/leave_out_evilgenie | Grok 4.5 | 40 | 20 | 4131 | 0 | 0 | Grok 4.5 | ok |
| cheat/leave_out_evilgenie | Kimi K2.7 Code | 18 | 9 | 4131 | 0 | 0 | Kimi K2.7 Code | ok |
| cheat/leave_out_evilgenie | Kimi K3 | 8 | 4 | 3196 | 0 | 0 | Kimi K3 | ok |
| cheat/leave_out_goodfire | GLM 5.2 | 1006 | 1006 | 1431 | 0 | 0 | GLM 5.2 | ok |
| cheat/leave_out_goodfire | Kimi K3 | 935 | 935 | 1427 | 0 | 0 | Kimi K3 | ok |
| cheat/leave_out_goodfire | Qwen3.8 Max | 892 | 892 | 1429 | 0 | 0 | Qwen3.8 Max | ok |
| cheat/leave_out_solo_design | DeepSeek V4.1 Flash | 90 | 30 | 3943 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/leave_out_solo_design | GPT-6 Luna | 81 | 27 | 3895 | 0 | 0 | GPT-6 Luna | ok |
| cheat/leave_out_spar_expl | Claude Haiku 5.5 | 36 | 12 | 3945 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/leave_out_spar_expl | DeepSeek V4.1 Flash | 36 | 12 | 3888 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/leave_out_spar_expl | GPT-6 Luna | 241 | 72 | 3903 | 0 | 0 | GPT-6 Luna | ok |
| cheat/leave_out_spar_mixed | DeepSeek V4.1 Flash | 36 | 12 | 3933 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/leave_out_spar_mixed | GPT-6 Luna | 126 | 43 | 3919 | 0 | 0 | GPT-6 Luna | ok |
| cheat/leave_out_spar_solo | Claude Haiku 5.5 | 30 | 30 | 3969 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/leave_out_spar_solo | GPT-6 Luna | 30 | 30 | 3867 | 0 | 0 | GPT-6 Luna | ok |
| cheat/leave_out_spar_team | Claude Haiku 5.5 | 150 | 45 | 3939 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/leave_out_spar_team | DeepSeek V4.1 Flash | 150 | 45 | 3882 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/leave_out_spar_team | GPT-6 Luna | 150 | 45 | 3837 | 0 | 0 | GPT-6 Luna | ok |
| cheat/ours | Claude Haiku 5.5 | 364 | 235 | 839 | 0 | 0 | Claude Haiku 5.5 | ok |
| cheat/ours | DeepSeek V4.1 Flash | 596 | 292 | 782 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| cheat/ours | GPT-5 mini | 120 | 120 | 954 | 0 | 0 | GPT-5 mini | ok |
| cheat/ours | GPT-6 Luna | 748 | 337 | 737 | 0 | 0 | GPT-6 Luna | ok |
| cheat/ours | Qwen3.8 27B | 120 | 120 | 954 | 0 | 0 | Qwen3.8 27B | ok |
| cheat/ours_to_external | Claude 3.7 Sonnet | 60 | 60 | 1074 | 0 | 0 | Claude 3.7 Sonnet | ok |
| cheat/ours_to_external | Claude Fable 5 | 10 | 5 | 1074 | 0 | 0 | Claude Fable 5 | ok |
| cheat/ours_to_external | Claude Opus 4.6 | 111 | 111 | 1074 | 0 | 0 | Claude Opus 4.6 | ok |
| cheat/ours_to_external | Claude Opus 4.8 | 6 | 3 | 1074 | 0 | 0 | Claude Opus 4.8 | ok |
| cheat/ours_to_external | GLM 5.2 | 1006 | 1006 | 1074 | 0 | 0 | GLM 5.2 | ok |
| cheat/ours_to_external | GPT-5.3 Codex | 77 | 56 | 1074 | 0 | 0 | GPT-5.3 Codex | ok |
| cheat/ours_to_external | GPT-5.4 | 18 | 18 | 1074 | 0 | 0 | GPT-5.4 | ok |
| cheat/ours_to_external | GPT-5.6 Luna | 28 | 14 | 1074 | 0 | 0 | GPT-5.6 Luna | ok |
| cheat/ours_to_external | GPT-5.6 Sol | 24 | 12 | 1074 | 0 | 0 | GPT-5.6 Sol | ok |
| cheat/ours_to_external | Gemini 3.1 Pro | 34 | 17 | 1074 | 0 | 0 | Gemini 3.1 Pro | ok |
| cheat/ours_to_external | Gemini 3.5 Flash | 56 | 28 | 1074 | 0 | 0 | Gemini 3.5 Flash | ok |
| cheat/ours_to_external | Grok 4.5 | 40 | 20 | 1074 | 0 | 0 | Grok 4.5 | ok |
| cheat/ours_to_external | Kimi K2.7 Code | 18 | 9 | 1074 | 0 | 0 | Kimi K2.7 Code | ok |
| cheat/ours_to_external | Kimi K3 | 943 | 939 | 1074 | 0 | 0 | Kimi K3 | ok |
| cheat/ours_to_external | Qwen3.8 Max | 892 | 892 | 1072 | 0 | 0 | Qwen3.8 Max | ok |
| collude/alone_spar_team | Claude Haiku 5.5 | 150 | 45 | 90 | 0 | 0 | Claude Haiku 5.5 | ok |
| collude/alone_spar_team | DeepSeek V4.1 Flash | 150 | 45 | 90 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| collude/alone_spar_team | GPT-6 Luna | 150 | 45 | 90 | 0 | 0 | GPT-6 Luna | ok |
| collude/leave_out_board_seed | DeepSeek V4.1 Flash | 134 | 43 | 129 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| collude/leave_out_spar_mixed | DeepSeek V4.1 Flash | 36 | 12 | 90 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| collude/leave_out_spar_team | Claude Haiku 5.5 | 150 | 45 | 66 | 0 | 0 | Claude Haiku 5.5 | ok |
| collude/leave_out_spar_team | GPT-6 Luna | 150 | 45 | 51 | 0 | 0 | GPT-6 Luna | ok |
| collude/ours | Claude Haiku 5.5 | 178 | 73 | 156 | 0 | 0 | Claude Haiku 5.5 | ok |
| collude/ours | DeepSeek V4.1 Flash | 320 | 100 | 129 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| collude/ours | GPT-6 Luna | 276 | 88 | 141 | 0 | 0 | GPT-6 Luna | ok |
| coord/alone_solo_design | DeepSeek V4.1 Flash | 90 | 30 | 29 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| coord/alone_solo_design | GPT-6 Luna | 81 | 27 | 32 | 0 | 0 | GPT-6 Luna | ok |
| coord/alone_spar_team | Claude Haiku 5.5 | 150 | 45 | 90 | 0 | 0 | Claude Haiku 5.5 | ok |
| coord/alone_spar_team | DeepSeek V4.1 Flash | 150 | 45 | 90 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| coord/alone_spar_team | GPT-6 Luna | 150 | 45 | 90 | 0 | 0 | GPT-6 Luna | ok |
| coord/leave_out_board_seed | DeepSeek V4.1 Flash | 134 | 43 | 158 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| coord/leave_out_solo_design | DeepSeek V4.1 Flash | 90 | 30 | 129 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| coord/leave_out_solo_design | GPT-6 Luna | 81 | 27 | 141 | 0 | 0 | GPT-6 Luna | ok |
| coord/leave_out_spar_team | Claude Haiku 5.5 | 150 | 45 | 125 | 0 | 0 | Claude Haiku 5.5 | ok |
| coord/leave_out_spar_team | DeepSeek V4.1 Flash | 150 | 45 | 68 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| coord/leave_out_spar_team | GPT-6 Luna | 150 | 45 | 83 | 0 | 0 | GPT-6 Luna | ok |
| coord/ours | Claude Haiku 5.5 | 178 | 73 | 215 | 0 | 0 | Claude Haiku 5.5 | ok |
| coord/ours | DeepSeek V4.1 Flash | 410 | 130 | 158 | 0 | 0 | DeepSeek V4.1 Flash | ok |
| coord/ours | GPT-6 Luna | 357 | 115 | 173 | 0 | 0 | GPT-6 Luna | ok |

## 3. Five author normalisations, checked against the raw files

| agent | raw field read | raw model id | normalised author | result |
|---|---|---|---|---|
| controls_600/binder_cheat_qwen_s20/k4s3x | data/labels.csv model_id; meta author 'qwen' | openrouter/qwen/qwen3.8-27b | Qwen3.8 27B | ok |
| goodfire_reward_hack/deepswe-qwen3.8-2.4t-a95b-fp8/00dc6efb575c435683bf569d8f31a813 | meta author_model; meta author 'qwenmax', author_name 'Qwen 3.8 Max' | Qwen/Qwen3.8-2.4T-A95B-FP8 | Qwen3.8 Max | ok |
| spar_mixed_41/binder_minimal_regulated_luna_argon_ambiguous1/n9k3q | episodes.jsonl model of agent n9k3q; meta author 'gpt-6-luna-openrouter' | openrouter/google/gemini-4-argon | Gemini 4 Argon | ok |
| spar_screen_team/binder_minimal_cheat_haiku1/a01 | episodes.jsonl model of agent a01; meta author 'claude-haiku-5-5' | anthropic/claude-haiku-5-5 | Claude Haiku 5.5 | ok |
| solo_design/solo_L1_flipt-02e2_qmax_s0/k5t8q | episodes.jsonl model of agent k5t8q; meta author 'openrouter/qwen/qwen3.8-max-0902' | openrouter/qwen/qwen3.8-max-0902 | Qwen3.8 Max | ok |

In spar_mixed_41 the meta.json `author` field names the run's worker model even for regulators; the episode's model is used instead, so regulators count as their own author.

## 4. Pooled features against team_probes

| agent | max abs difference / max abs value vs team_probes.agent_matrix | result |
|---|---|---|
| controls_600/poem_cheat_gpt5m_s37/z2m6n | 1.5e-04 | ok (fp16 rounding) |
| board_seed/cm_anticoord_s4/p3x6n | 1.1e-04 | ok (fp16 rounding) |
| solo_design/solo_L1c_subset_ds_s0/p8m9j | 7.5e-05 | ok (fp16 rounding) |
| spar_screen_team/game_minimal_haiku3/a03 | 2.8e-04 | ok (fp16 rounding) |
| spar_mixed_41/flipt_minimal_regulated_dsflash_haiku_enforce_fixed3/n6z9n | 3.2e-04 | ok (fp16 rounding) |
| spar_exploration_single/flipt_minimal_haiku_identified_fixed1/j6q9k | 2.2e-04 | ok (fp16 rounding) |
| spar_screen_solo/poem_solo_dsflash2/a01 | 2.6e-04 | ok (fp16 rounding) |
| goodfire_reward_hack/deepswe-glm-5.2-fp8/85cea65d76594e7f999c97b719b966df | 1.8e-04 | ok (fp16 rounding) |

## 5. Labels against the team_probes agent's CSVs

| source | agents here | joined | cheat index equal | coord index equal | collude index equal | cheat onset equal | result |
|---|---|---|---|---|---|---|---|
| board_seed | 134 | 134 | 134 | 134 | 134 | 134 | ok |
| solo_design | 177 | 177 | 177 | 177 | 177 | 177 | ok |
| spar_mixed_41 | 209 | 209 | 209 | 209 | 209 | 209 | ok |
| spar_screen_team | 450 | 450 | 450 | 450 | 450 | 450 | ok |

controls_600: label equal to data/labels.csv on 600 of 600 runs; onset_turn equal to the step of onset_call in source.json on 336 positives (0 mismatches). Goodfire: judge_rollouts verdict equal to meta.json verdict on 3136 rollouts (0 mismatches).

## 6. Streamed per-step PCA against team_probes.prep_folds

| fold | training turns | min abs corr, top 20 components | median abs corr, 128 components |
|---|---|---|---|
| 0 | 5602 | 1.00000 | 0.99997 |
| 1 | 5988 | 1.00000 | 0.99995 |
| 2 | 5793 | 1.00000 | 0.99994 |
| 3 | 5886 | 1.00000 | 0.99997 |
| 4 | 5855 | 1.00000 | 0.99997 |

Turn AUROC (board_seed cheating, C = 0.01): streamed 0.7060, prep_folds 0.7068; score correlation 0.9863. prep_folds uses a 4-iteration randomized SVD above 1,500 rows, so the two agree to that solver's precision.

## 7. Architectures

Every architecture row with an AUROC (1436) was recomputed with sklearn from `arch/scores/`; 1436 match within the tie bound of section 1 with the same count. One row per cell is shown:

| label | scheme | type | row | arch_cells.csv | recomputed | result |
|---|---|---|---|---|---|---|
| cheat | alone: spar_solo | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.437500 | 0.437500 | 26/26 rows of the cell match |
| cheat | alone: spar_solo | B.I | mlp_mean pooled seed 0 | 0.944087 | 0.944087 | 13/13 rows of the cell match |
| cheat | ours | B.I | mlp_mean pooled seed 0 | 0.947955 | 0.947955 | 14/14 rows of the cell match |
| cheat | ours | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.889980 | 0.889980 | 65/65 rows of the cell match |
| cheat | all | B.I | mlp_mean pooled seed 0 | 0.915868 | 0.915868 | 14/14 rows of the cell match |
| cheat | all | B.II | mlp_mean Claude 3.7 Sonnet seed 0 | 0.814815 | 0.814815 | 260/260 rows of the cell match |
| coord | ours | B.I | mlp_mean pooled seed 0 | 0.983858 | 0.983858 | 14/14 rows of the cell match |
| coord | ours | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.935664 | 0.935664 | 39/39 rows of the cell match |
| collude | ours | B.I | mlp_mean pooled seed 0 | 0.937224 | 0.937224 | 14/14 rows of the cell match |
| collude | ours | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.603682 | 0.603682 | 39/39 rows of the cell match |
| cheat | alone: controls | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.915000 | 0.915000 | 65/65 rows of the cell match |
| cheat | alone: solo_design | B.II | mlp_mean DeepSeek V4.1 Flash seed 0 | 0.345846 | 0.345846 | 26/26 rows of the cell match |
| cheat | alone: spar_team | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.731884 | 0.731884 | 39/39 rows of the cell match |
| cheat | alone: spar_mixed | B.II | mlp_mean DeepSeek V4.1 Flash seed 0 | 0.588477 | 0.588477 | 26/26 rows of the cell match |
| cheat | alone: spar_expl | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.612903 | 0.612903 | 39/39 rows of the cell match |
| cheat | alone: goodfire | B.II | mlp_mean GLM 5.2 seed 0 | 0.872314 | 0.872314 | 39/39 rows of the cell match |
| cheat | alone: cheatbench | B.II | mlp_mean Claude 3.7 Sonnet seed 0 | 0.868148 | 0.868148 | 52/52 rows of the cell match |
| cheat | alone: evilgenie | B.II | mlp_mean Claude Fable 5 seed 0 | 1.000000 | 1.000000 | 120/120 rows of the cell match |
| coord | alone: solo_design | B.II | mlp_mean DeepSeek V4.1 Flash seed 0 | 0.661429 | 0.661429 | 26/26 rows of the cell match |
| coord | alone: spar_team | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.927950 | 0.927950 | 39/39 rows of the cell match |
| collude | alone: spar_team | B.II | mlp_mean Claude Haiku 5.5 seed 0 | 0.902778 | 0.902778 | 39/39 rows of the cell match |
| cheat | alone: controls | B.I | mlp_mean pooled seed 0 | 0.947964 | 0.947964 | 13/13 rows of the cell match |
| cheat | alone: board_seed | B.I | mlp_mean pooled seed 0 | 0.830586 | 0.830586 | 13/13 rows of the cell match |
| cheat | alone: solo_design | B.I | mlp_mean pooled seed 0 | 0.863415 | 0.863415 | 13/13 rows of the cell match |
| cheat | alone: spar_team | B.I | mlp_mean pooled seed 0 | 0.921709 | 0.921709 | 13/13 rows of the cell match |
| cheat | alone: spar_mixed | B.I | mlp_mean pooled seed 0 | 0.851523 | 0.851523 | 13/13 rows of the cell match |
| cheat | alone: spar_expl | B.I | mlp_mean pooled seed 0 | 0.932546 | 0.932546 | 13/13 rows of the cell match |
| cheat | alone: goodfire | B.I | mlp_mean pooled seed 0 | 0.934895 | 0.934894 | 13/13 rows of the cell match |
| cheat | alone: cheatbench | B.I | mlp_mean pooled seed 0 | 0.968159 | 0.968159 | 13/13 rows of the cell match |
| cheat | alone: evilgenie | B.I | mlp_mean pooled seed 0 | 0.714399 | 0.714399 | 12/12 rows of the cell match |
| coord | alone: board_seed | B.I | mlp_mean pooled seed 0 | 0.938127 | 0.938127 | 13/13 rows of the cell match |
| coord | alone: solo_design | B.I | mlp_mean pooled seed 0 | 0.693866 | 0.693866 | 13/13 rows of the cell match |
| coord | alone: spar_team | B.I | mlp_mean pooled seed 0 | 0.977217 | 0.977217 | 13/13 rows of the cell match |
| coord | alone: spar_mixed | B.I | mlp_mean pooled seed 0 | 0.980488 | 0.980488 | 13/13 rows of the cell match |
| collude | alone: board_seed | B.I | mlp_mean pooled seed 0 | 0.813469 | 0.813469 | 13/13 rows of the cell match |
| collude | alone: spar_team | B.I | mlp_mean pooled seed 0 | 0.915163 | 0.915163 | 13/13 rows of the cell match |
| collude | alone: spar_mixed | B.I | mlp_mean pooled seed 0 | 0.907355 | 0.907355 | 13/13 rows of the cell match |
| cheat | leave out: controls | B.I | mlp_mean pooled seed 0 | 0.743701 | 0.743701 | 13/13 rows of the cell match |
| cheat | leave out: board_seed | B.I | mlp_mean pooled seed 0 | 0.879350 | 0.879350 | 13/13 rows of the cell match |
| cheat | leave out: solo_design | B.I | mlp_mean pooled seed 0 | 0.927547 | 0.927547 | 13/13 rows of the cell match |
| cheat | leave out: spar_team | B.I | mlp_mean pooled seed 0 | 0.835831 | 0.835831 | 13/13 rows of the cell match |
| cheat | leave out: spar_mixed | B.I | mlp_mean pooled seed 0 | 0.901438 | 0.901438 | 13/13 rows of the cell match |
| cheat | leave out: spar_expl | B.I | mlp_mean pooled seed 0 | 0.935398 | 0.935398 | 13/13 rows of the cell match |
| cheat | leave out: spar_solo | B.I | mlp_mean pooled seed 0 | 0.725878 | 0.725878 | 13/13 rows of the cell match |
| cheat | leave out: goodfire | B.I | mlp_mean pooled seed 0 | 0.664095 | 0.664095 | 13/13 rows of the cell match |
| cheat | leave out: cheatbench | B.I | mlp_mean pooled seed 0 | 0.681343 | 0.681343 | 13/13 rows of the cell match |
| cheat | leave out: evilgenie | B.I | mlp_mean pooled seed 0 | 0.622873 | 0.622873 | 13/13 rows of the cell match |
| coord | leave out: board_seed | B.I | mlp_mean pooled seed 0 | 0.962583 | 0.962583 | 13/13 rows of the cell match |
| coord | leave out: solo_design | B.I | mlp_mean pooled seed 0 | 0.959864 | 0.959864 | 13/13 rows of the cell match |
| coord | leave out: spar_team | B.I | mlp_mean pooled seed 0 | 0.951019 | 0.951019 | 13/13 rows of the cell match |
| coord | leave out: spar_mixed | B.I | mlp_mean pooled seed 0 | 0.592683 | 0.592683 | 13/13 rows of the cell match |
| collude | leave out: board_seed | B.I | mlp_mean pooled seed 0 | 0.765019 | 0.765019 | 13/13 rows of the cell match |
| collude | leave out: spar_team | B.I | mlp_mean pooled seed 0 | 0.691166 | 0.691166 | 13/13 rows of the cell match |
| collude | leave out: spar_mixed | B.I | mlp_mean pooled seed 0 | 0.985149 | 0.985149 | 13/13 rows of the cell match |

| label | scheme | type | architecture | check | result |
|---|---|---|---|---|---|
| cheat | alone: spar_solo | B.II | mlp_mean | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_solo | B.II | mlp_max | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_solo | B.II | attn_pool | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_solo | B.II | attn_monitor | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_solo | B.II | gru | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_solo | B.II | linear_step | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_solo | B.II | mlp_traj | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_solo | B.I | mlp_mean seed 0 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | mlp_mean seed 1 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | mlp_max seed 0 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | mlp_max seed 1 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | attn_pool seed 0 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | attn_pool seed 1 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | attn_monitor seed 0 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | attn_monitor seed 1 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | gru seed 0 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | gru seed 1 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | linear_step seed 0 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | mlp_traj seed 0 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_solo | B.I | mlp_traj seed 1 | 90/90 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | mlp_mean seed 0 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | mlp_mean seed 1 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | mlp_max seed 0 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | mlp_max seed 1 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | attn_pool seed 0 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | attn_pool seed 1 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | attn_monitor seed 0 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | attn_monitor seed 1 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | gru seed 0 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | gru seed 1 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | linear_step seed 0 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | mlp_traj seed 0 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | mlp_traj seed 1 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.I | stack seed 0 | 1973/1973 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | ours | B.II | mlp_mean | 5/5 held-out author sets of exactly that author | ok |
| cheat | ours | B.II | mlp_max | 5/5 held-out author sets of exactly that author | ok |
| cheat | ours | B.II | attn_pool | 5/5 held-out author sets of exactly that author | ok |
| cheat | ours | B.II | attn_monitor | 5/5 held-out author sets of exactly that author | ok |
| cheat | ours | B.II | gru | 5/5 held-out author sets of exactly that author | ok |
| cheat | ours | B.II | linear_step | 5/5 held-out author sets of exactly that author | ok |
| cheat | ours | B.II | mlp_traj | 5/5 held-out author sets of exactly that author | ok |
| cheat | all | B.I | mlp_mean seed 0 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | mlp_mean seed 1 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | mlp_max seed 0 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | mlp_max seed 1 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | attn_pool seed 0 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | attn_pool seed 1 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | attn_monitor seed 0 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | attn_monitor seed 1 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | gru seed 0 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | gru seed 1 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | linear_step seed 0 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | mlp_traj seed 0 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | mlp_traj seed 1 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.I | stack seed 0 | 5296/5296 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | all | B.II | mlp_mean | 20/20 held-out author sets of exactly that author | ok |
| cheat | all | B.II | mlp_max | 20/20 held-out author sets of exactly that author | ok |
| cheat | all | B.II | attn_pool | 20/20 held-out author sets of exactly that author | ok |
| cheat | all | B.II | attn_monitor | 20/20 held-out author sets of exactly that author | ok |
| cheat | all | B.II | gru | 20/20 held-out author sets of exactly that author | ok |
| cheat | all | B.II | linear_step | 20/20 held-out author sets of exactly that author | ok |
| cheat | all | B.II | mlp_traj | 20/20 held-out author sets of exactly that author | ok |
| coord | ours | B.I | mlp_mean seed 0 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | mlp_mean seed 1 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | mlp_max seed 0 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | mlp_max seed 1 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | attn_pool seed 0 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | attn_pool seed 1 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | attn_monitor seed 0 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | attn_monitor seed 1 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | gru seed 0 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | gru seed 1 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | linear_step seed 0 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | mlp_traj seed 0 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | mlp_traj seed 1 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.I | stack seed 0 | 970/970 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | ours | B.II | mlp_mean | 3/3 held-out author sets of exactly that author | ok |
| coord | ours | B.II | mlp_max | 3/3 held-out author sets of exactly that author | ok |
| coord | ours | B.II | attn_pool | 3/3 held-out author sets of exactly that author | ok |
| coord | ours | B.II | attn_monitor | 3/3 held-out author sets of exactly that author | ok |
| coord | ours | B.II | gru | 3/3 held-out author sets of exactly that author | ok |
| coord | ours | B.II | linear_step | 3/3 held-out author sets of exactly that author | ok |
| coord | ours | B.II | mlp_traj | 3/3 held-out author sets of exactly that author | ok |
| collude | ours | B.I | mlp_mean seed 0 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | mlp_mean seed 1 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | mlp_max seed 0 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | mlp_max seed 1 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | attn_pool seed 0 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | attn_pool seed 1 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | attn_monitor seed 0 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | attn_monitor seed 1 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | gru seed 0 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | gru seed 1 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | linear_step seed 0 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | mlp_traj seed 0 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | mlp_traj seed 1 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.I | stack seed 0 | 793/793 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | ours | B.II | mlp_mean | 3/3 held-out author sets of exactly that author | ok |
| collude | ours | B.II | mlp_max | 3/3 held-out author sets of exactly that author | ok |
| collude | ours | B.II | attn_pool | 3/3 held-out author sets of exactly that author | ok |
| collude | ours | B.II | attn_monitor | 3/3 held-out author sets of exactly that author | ok |
| collude | ours | B.II | gru | 3/3 held-out author sets of exactly that author | ok |
| collude | ours | B.II | linear_step | 3/3 held-out author sets of exactly that author | ok |
| collude | ours | B.II | mlp_traj | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: controls | B.II | mlp_mean | 5/5 held-out author sets of exactly that author | ok |
| cheat | alone: controls | B.II | mlp_max | 5/5 held-out author sets of exactly that author | ok |
| cheat | alone: controls | B.II | attn_pool | 5/5 held-out author sets of exactly that author | ok |
| cheat | alone: controls | B.II | attn_monitor | 5/5 held-out author sets of exactly that author | ok |
| cheat | alone: controls | B.II | gru | 5/5 held-out author sets of exactly that author | ok |
| cheat | alone: controls | B.II | linear_step | 5/5 held-out author sets of exactly that author | ok |
| cheat | alone: controls | B.II | mlp_traj | 5/5 held-out author sets of exactly that author | ok |
| cheat | alone: solo_design | B.II | mlp_mean | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: solo_design | B.II | mlp_max | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: solo_design | B.II | attn_pool | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: solo_design | B.II | attn_monitor | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: solo_design | B.II | gru | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: solo_design | B.II | linear_step | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: solo_design | B.II | mlp_traj | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_team | B.II | mlp_mean | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_team | B.II | mlp_max | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_team | B.II | attn_pool | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_team | B.II | attn_monitor | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_team | B.II | gru | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_team | B.II | linear_step | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_team | B.II | mlp_traj | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_mixed | B.II | mlp_mean | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_mixed | B.II | mlp_max | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_mixed | B.II | attn_pool | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_mixed | B.II | attn_monitor | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_mixed | B.II | gru | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_mixed | B.II | linear_step | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_mixed | B.II | mlp_traj | 2/2 held-out author sets of exactly that author | ok |
| cheat | alone: spar_expl | B.II | mlp_mean | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_expl | B.II | mlp_max | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_expl | B.II | attn_pool | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_expl | B.II | attn_monitor | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_expl | B.II | gru | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_expl | B.II | linear_step | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: spar_expl | B.II | mlp_traj | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: goodfire | B.II | mlp_mean | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: goodfire | B.II | mlp_max | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: goodfire | B.II | attn_pool | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: goodfire | B.II | attn_monitor | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: goodfire | B.II | gru | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: goodfire | B.II | linear_step | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: goodfire | B.II | mlp_traj | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: cheatbench | B.II | mlp_mean | 4/4 held-out author sets of exactly that author | ok |
| cheat | alone: cheatbench | B.II | mlp_max | 4/4 held-out author sets of exactly that author | ok |
| cheat | alone: cheatbench | B.II | attn_pool | 4/4 held-out author sets of exactly that author | ok |
| cheat | alone: cheatbench | B.II | attn_monitor | 4/4 held-out author sets of exactly that author | ok |
| cheat | alone: cheatbench | B.II | gru | 4/4 held-out author sets of exactly that author | ok |
| cheat | alone: cheatbench | B.II | linear_step | 4/4 held-out author sets of exactly that author | ok |
| cheat | alone: cheatbench | B.II | mlp_traj | 4/4 held-out author sets of exactly that author | ok |
| cheat | alone: evilgenie | B.II | mlp_mean | 10/10 held-out author sets of exactly that author | ok |
| cheat | alone: evilgenie | B.II | mlp_max | 10/10 held-out author sets of exactly that author | ok |
| cheat | alone: evilgenie | B.II | attn_pool | 10/10 held-out author sets of exactly that author | ok |
| cheat | alone: evilgenie | B.II | attn_monitor | 10/10 held-out author sets of exactly that author | ok |
| cheat | alone: evilgenie | B.II | gru | 10/10 held-out author sets of exactly that author | ok |
| cheat | alone: evilgenie | B.II | mlp_traj | 10/10 held-out author sets of exactly that author | ok |
| coord | alone: solo_design | B.II | mlp_mean | 2/2 held-out author sets of exactly that author | ok |
| coord | alone: solo_design | B.II | mlp_max | 2/2 held-out author sets of exactly that author | ok |
| coord | alone: solo_design | B.II | attn_pool | 2/2 held-out author sets of exactly that author | ok |
| coord | alone: solo_design | B.II | attn_monitor | 2/2 held-out author sets of exactly that author | ok |
| coord | alone: solo_design | B.II | gru | 2/2 held-out author sets of exactly that author | ok |
| coord | alone: solo_design | B.II | linear_step | 2/2 held-out author sets of exactly that author | ok |
| coord | alone: solo_design | B.II | mlp_traj | 2/2 held-out author sets of exactly that author | ok |
| coord | alone: spar_team | B.II | mlp_mean | 3/3 held-out author sets of exactly that author | ok |
| coord | alone: spar_team | B.II | mlp_max | 3/3 held-out author sets of exactly that author | ok |
| coord | alone: spar_team | B.II | attn_pool | 3/3 held-out author sets of exactly that author | ok |
| coord | alone: spar_team | B.II | attn_monitor | 3/3 held-out author sets of exactly that author | ok |
| coord | alone: spar_team | B.II | gru | 3/3 held-out author sets of exactly that author | ok |
| coord | alone: spar_team | B.II | linear_step | 3/3 held-out author sets of exactly that author | ok |
| coord | alone: spar_team | B.II | mlp_traj | 3/3 held-out author sets of exactly that author | ok |
| collude | alone: spar_team | B.II | mlp_mean | 3/3 held-out author sets of exactly that author | ok |
| collude | alone: spar_team | B.II | mlp_max | 3/3 held-out author sets of exactly that author | ok |
| collude | alone: spar_team | B.II | attn_pool | 3/3 held-out author sets of exactly that author | ok |
| collude | alone: spar_team | B.II | attn_monitor | 3/3 held-out author sets of exactly that author | ok |
| collude | alone: spar_team | B.II | gru | 3/3 held-out author sets of exactly that author | ok |
| collude | alone: spar_team | B.II | linear_step | 3/3 held-out author sets of exactly that author | ok |
| collude | alone: spar_team | B.II | mlp_traj | 3/3 held-out author sets of exactly that author | ok |
| cheat | alone: controls | B.I | mlp_mean seed 0 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | mlp_mean seed 1 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | mlp_max seed 0 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | mlp_max seed 1 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | attn_pool seed 0 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | attn_pool seed 1 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | attn_monitor seed 0 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | attn_monitor seed 1 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | gru seed 0 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | gru seed 1 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | linear_step seed 0 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | mlp_traj seed 0 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: controls | B.I | mlp_traj seed 1 | 600/600 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | mlp_mean seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | mlp_mean seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | mlp_max seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | mlp_max seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | attn_pool seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | attn_pool seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | attn_monitor seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | attn_monitor seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | gru seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | gru seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | linear_step seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | mlp_traj seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: board_seed | B.I | mlp_traj seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | mlp_mean seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | mlp_mean seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | mlp_max seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | mlp_max seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | attn_pool seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | attn_pool seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | attn_monitor seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | attn_monitor seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | gru seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | gru seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | linear_step seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | mlp_traj seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: solo_design | B.I | mlp_traj seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | mlp_mean seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | mlp_mean seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | mlp_max seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | mlp_max seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | attn_pool seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | attn_pool seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | attn_monitor seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | attn_monitor seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | gru seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | gru seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | linear_step seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | mlp_traj seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_team | B.I | mlp_traj seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | mlp_mean seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | mlp_mean seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | mlp_max seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | mlp_max seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | attn_pool seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | attn_pool seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | attn_monitor seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | attn_monitor seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | gru seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | gru seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | linear_step seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | mlp_traj seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_mixed | B.I | mlp_traj seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | mlp_mean seed 0 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | mlp_mean seed 1 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | mlp_max seed 0 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | mlp_max seed 1 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | attn_pool seed 0 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | attn_pool seed 1 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | attn_monitor seed 0 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | attn_monitor seed 1 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | gru seed 0 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | gru seed 1 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | linear_step seed 0 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | mlp_traj seed 0 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: spar_expl | B.I | mlp_traj seed 1 | 313/313 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | mlp_mean seed 0 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | mlp_mean seed 1 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | mlp_max seed 0 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | mlp_max seed 1 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | attn_pool seed 0 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | attn_pool seed 1 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | attn_monitor seed 0 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | attn_monitor seed 1 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | gru seed 0 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | gru seed 1 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | linear_step seed 0 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | mlp_traj seed 0 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: goodfire | B.I | mlp_traj seed 1 | 2833/2833 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | mlp_mean seed 0 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | mlp_mean seed 1 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | mlp_max seed 0 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | mlp_max seed 1 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | attn_pool seed 0 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | attn_pool seed 1 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | attn_monitor seed 0 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | attn_monitor seed 1 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | gru seed 0 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | gru seed 1 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | linear_step seed 0 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | mlp_traj seed 0 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: cheatbench | B.I | mlp_traj seed 1 | 224/224 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | mlp_mean seed 0 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | mlp_mean seed 1 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | mlp_max seed 0 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | mlp_max seed 1 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | attn_pool seed 0 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | attn_pool seed 1 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | attn_monitor seed 0 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | attn_monitor seed 1 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | gru seed 0 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | gru seed 1 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | mlp_traj seed 0 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| cheat | alone: evilgenie | B.I | mlp_traj seed 1 | 266/266 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | mlp_mean seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | mlp_mean seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | mlp_max seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | mlp_max seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | attn_pool seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | attn_pool seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | attn_monitor seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | attn_monitor seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | gru seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | gru seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | linear_step seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | mlp_traj seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: board_seed | B.I | mlp_traj seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | mlp_mean seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | mlp_mean seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | mlp_max seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | mlp_max seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | attn_pool seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | attn_pool seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | attn_monitor seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | attn_monitor seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | gru seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | gru seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | linear_step seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | mlp_traj seed 0 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: solo_design | B.I | mlp_traj seed 1 | 177/177 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | mlp_mean seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | mlp_mean seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | mlp_max seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | mlp_max seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | attn_pool seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | attn_pool seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | attn_monitor seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | attn_monitor seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | gru seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | gru seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | linear_step seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | mlp_traj seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_team | B.I | mlp_traj seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | mlp_mean seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | mlp_mean seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | mlp_max seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | mlp_max seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | attn_pool seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | attn_pool seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | attn_monitor seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | attn_monitor seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | gru seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | gru seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | linear_step seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | mlp_traj seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| coord | alone: spar_mixed | B.I | mlp_traj seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | mlp_mean seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | mlp_mean seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | mlp_max seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | mlp_max seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | attn_pool seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | attn_pool seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | attn_monitor seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | attn_monitor seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | gru seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | gru seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | linear_step seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | mlp_traj seed 0 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: board_seed | B.I | mlp_traj seed 1 | 134/134 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | mlp_mean seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | mlp_mean seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | mlp_max seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | mlp_max seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | attn_pool seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | attn_pool seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | attn_monitor seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | attn_monitor seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | gru seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | gru seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | linear_step seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | mlp_traj seed 0 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_team | B.I | mlp_traj seed 1 | 450/450 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | mlp_mean seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | mlp_mean seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | mlp_max seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | mlp_max seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | attn_pool seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | attn_pool seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | attn_monitor seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | attn_monitor seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | gru seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | gru seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | linear_step seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | mlp_traj seed 0 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |
| collude | alone: spar_mixed | B.I | mlp_traj seed 1 | 209/209 agents in the same fold as the linear matrix; runs in two folds: 0 | ok |

The stack's components on the outer training agents come from an inner 3-fold split of those agents only (StratifiedGroupKFold over their runs); the outer test agents enter only through the four component scores of models fitted on the whole outer training set.

## Result

All checks passed.
