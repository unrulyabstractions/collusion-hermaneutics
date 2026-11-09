# Verification of the controls and nulls

Every check below was run by `python -m hermeneutics.controls_nulls verify` on the saved files, 2026-11-09 19:52 UTC.

## 1. Prompt-only probe

- PASS: every pooled AUROC of table 1 recomputed from scores/prompt_scores.csv.gz. 175 rows, max |difference| 0.00e+00
- PASS: one AUROC by sklearn: cheating, controls_600, prompt-only main feature. table 0.561378, sklearn roc_auc_score 0.561378, 600 agents
- PASS: pooled prompt-only folds equal matrix_probes' B.I folds (by agent) and keep every run in one fold. 175 (label, set, feature) score sets, 0 with a different fold, 0 with a run split across folds
- PASS: every within-condition refit AUROC recomputed from its saved scores; refit folds grouped by run. 189 rows, max |difference| 0.00e+00, 0 with a run split across folds
- PASS: every restricted AUROC recomputed by filtering the pooled scores to the condition. 649 rows, max |difference| 0.00e+00
- PASS: turn-0 boundary tokens contiguous and ending one token before the turn-0 think-open, every agent with turns.safetensors. {"boundary_contiguous_and_adjacent": 1973}
- PASS: prompt sentence means: header unit count equals spans.jsonl, prompt units end at the turn-0 boundary token. 1973 of 1973 agents read; last prompt token end equals the first boundary position on 1973; first assistant span: {'cot': 1089, 'final_text': 245, 'tool_call': 639}
- PASS: external sets: only system and user messages precede the first <think> (turn-0 think-open is prompt-only). board_seed: {'system/user/assistant': 134}; cheatbench: {'user/assistant': 164, 'system/user/assistant': 60}; controls_600: {'system/user/assistant': 600}; evilgenie: {'system/user/assistant': 266}; goodfire_reward_hack: {'system/user/assistant': 4356, 'system/assistant': 1197}; solo_design: {'system/user/assistant': 177}; spar_exploration_single: {'system/user/assistant': 313}; spar_mixed_41: {'system/user/assistant': 209}; spar_screen_solo: {'system/user/assistant': 90}; spar_screen_team: {'system/user/assistant': 450} (20 agents with zero turns left out, as in matrix_probes)

## 2. Author-identity control

- PASS: raw B.I and B.II rows (pooled, every held-out author, mean over authors) equal matrix_probes cells.csv think. 80 rows, max |difference| 0.00e+00
- PASS: every B.I and per-author B.II AUROC of table 2 recomputed from scores/author (scores saved to 6 significant digits). 204 rows, max |difference| 2.07e-06
- PASS: LEACE: author-probe accuracy on held-out B.I folds after erasure within 0.02 of chance (training fold's most frequent author), and training cross-covariance of the erased components with the concept below 1e-6. 28 erased rows; accuracy minus chance from -0.005 to +0.018; raw accuracy from 0.986 to 1.000; max cross-covariance 3.4e-14
    - cheat, alone: controls, author: author accuracy 0.190 vs chance 0.193 over 600.0 held-out agents
    - cheat, alone: controls, author+task+condition: author accuracy 0.188 vs chance 0.193 over 600.0 held-out agents
    - cheat, alone: solo_design, author: author accuracy 0.508 vs chance 0.508 over 177.0 held-out agents
    - cheat, alone: solo_design, author+task+condition: author accuracy 0.508 vs chance 0.508 over 177.0 held-out agents
    - cheat, alone: spar_team, author: author accuracy 0.329 vs chance 0.316 over 450.0 held-out agents
    - cheat, alone: spar_team, author+task+condition: author accuracy 0.331 vs chance 0.316 over 450.0 held-out agents
    - cheat, alone: spar_mixed, author: author accuracy 0.603 vs chance 0.603 over 209.0 held-out agents
    - cheat, alone: spar_mixed, author+task+condition: author accuracy 0.603 vs chance 0.603 over 209.0 held-out agents
    - cheat, alone: goodfire, author: author accuracy 0.355 vs chance 0.355 over 2833.0 held-out agents
    - cheat, alone: goodfire, author+task+condition: author accuracy 0.355 vs chance 0.355 over 2833.0 held-out agents
    - cheat, ours, author: author accuracy 0.379 vs chance 0.379 over 1973.0 held-out agents
    - cheat, ours, author+task+condition: author accuracy 0.379 vs chance 0.379 over 1973.0 held-out agents
    - cheat, all, author: author accuracy 0.190 vs chance 0.190 over 5296.0 held-out agents
    - cheat, all, author+task+condition: author accuracy 0.190 vs chance 0.190 over 5296.0 held-out agents
    - coord, alone: solo_design, author: author accuracy 0.508 vs chance 0.508 over 177.0 held-out agents
    - coord, alone: solo_design, author+task+condition: author accuracy 0.508 vs chance 0.508 over 177.0 held-out agents
    - coord, alone: spar_team, author: author accuracy 0.338 vs chance 0.327 over 450.0 held-out agents
    - coord, alone: spar_team, author+task+condition: author accuracy 0.344 vs chance 0.327 over 450.0 held-out agents
    - coord, alone: spar_mixed, author: author accuracy 0.603 vs chance 0.603 over 209.0 held-out agents
    - coord, alone: spar_mixed, author+task+condition: author accuracy 0.603 vs chance 0.603 over 209.0 held-out agents
    - coord, ours, author: author accuracy 0.423 vs chance 0.423 over 970.0 held-out agents
    - coord, ours, author+task+condition: author accuracy 0.423 vs chance 0.423 over 970.0 held-out agents
    - collude, alone: spar_team, author: author accuracy 0.327 vs chance 0.320 over 450.0 held-out agents
    - collude, alone: spar_team, author+task+condition: author accuracy 0.327 vs chance 0.320 over 450.0 held-out agents
    - collude, alone: spar_mixed, author: author accuracy 0.603 vs chance 0.603 over 209.0 held-out agents
    - collude, alone: spar_mixed, author+task+condition: author accuracy 0.603 vs chance 0.603 over 209.0 held-out agents
    - collude, ours, author: author accuracy 0.404 vs chance 0.404 over 793.0 held-out agents
    - collude, ours, author+task+condition: author accuracy 0.404 vs chance 0.404 over 793.0 held-out agents

## 3. Lexical-only reader null

- PASS: metadata rows recomputed by the same runner equal matrix_probes cells.csv on every (label, scheme, type, unit): same folds and splits. 382 rows matched, max |difference| 0.00e+00
- PASS: every think-window TF-IDF AUROC (pooled and per author) recomputed from scores/lexical (matrix_probes' save_frames writes 6 significant digits). 243 rows, max |difference| 1.39e-04
- PASS: no think-window unit scores all its agents alike, and no agent with an empty think window gets a score. 310 (file, type, unit) score sets; 0 constant []; 0 scored agents with an empty window
- NOTE: think windows found equal meta n_turns on 8010 of 8016 agents; the 6 others (spar_mixed_41) hold more `<think>` tags than turns (tags inside tool output or board posts); their text between those tags is kept.

## 4. Label-permutation nulls

- PASS: refit nulls: the real-label AUROC recomputed by the null code equals cells.csv for every TF-IDF and metadata B.I pooled row; at least 45 of the 50 draws finite. 44 rows, max |difference| 0.00e+00, finite draws per row [49, 50] (a draw is undefined when a permutation leaves a training fold with one class: coord alone: spar_mixed meta: 49; coord alone: spar_mixed tfidf: 49)
- PASS: fixed-score nulls: saved scores reproduce the table AUROC. 933 of 933 rows reproduce within 0.001 (matrix_probes saves scores to 6 significant digits); max |difference| among them 1.6e-04; rows that do not reproduce carry no null and are listed in RESULTS.md
- PASS: one fixed-score null recomputed from its seed. cheat alone: controls A Claude Haiku 5.5 tfidf: null 95th 0.5897 saved, 0.5897 recomputed

**Summary: all checks PASS.**
