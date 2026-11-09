# Verification of the attention monitor outputs

## 2026-11-09, board_seed, think pair

- Folds: every one of the 43 runs sits in exactly one fold (step_scores_board_seed_think.csv.gz, fold per run); the fold of each of the 129 agents equals its fold in team_probes' step_turn_scores.csv.gz (129 of 129). VERIFIED.
- AUROC by hand: the per-step AUROC of the cheating-used binary head, seed 0, recomputed from the saved step scores with a rank (Mann-Whitney) formula and with sklearn: 0.88256, equal to metrics_board_seed_think.csv. The agent AUROC (last step) of the collusion binary head, seed 1, recomputed with sklearn from the last-step scores and the final labels in labels_board_seed.jsonl: 0.83591, equal to the CSV. VERIFIED.
- Linear baseline: the per-step linear probe on the same folds gives 0.889 (cheating used) and 0.849 (collusion crossed), equal to team_probes' step_probes.csv rows 5a and 5d for probe Lall think-open+close. VERIFIED.
- Plot score_vs_turn_cheating_board_seed_think.png opened and viewed: six panels, the six agents of team_probes' step_scores_cheating.png, both seeds and the linear probe drawn, shaded positive span from the estimated used onset. VERIFIED.
- Labels: the final cheating, collusion and coordination levels of every record equal team_probes' labels/agents.csv (asserted while building labels_board_seed.jsonl). VERIFIED.
- Independent verifier agent (read-only) on the board_seed think outputs: every AUROC of the 27 metric rows recomputed with sklearn to 1e-16; folds equal team_probes' on 6818 of 6818 rows; y|cheat and y|collude equal team_probes 5a and 5d per-turn labels on 6818 of 6818 rows; final levels equal agents.csv on all six ladders; RESULTS.md tables match the CSVs; PNG viewed; no test-fold leakage found in standardisation, early stopping, GRU inner scores or TF-IDF. One text error found and fixed: agents reaching a cut with no onset are dropped from that cut's agent-level scoring too (9 coordination agents), not kept. VERIFIED.

## 2026-11-09, board_seed, seven tokens

- No run crosses folds (43 runs). Per-step AUROC of the collusion binary head, seed 1, recomputed with sklearn from step_scores_board_seed_seven.csv.gz equals the CSV (all 129 agents reach collusion with an onset or not at all, so no agent is dropped). Baseline rows equal the think-pair run's to 0.0. Plot score_vs_turn_cheating_board_seed_seven.png viewed: six panels, same agents. VERIFIED.

## 2026-11-09, board_seed null complete; controls_600 think pair, 5 folds

- Board null: 50 draws on disk (draws 0-24 and 25-49, 3 cuts each); no draw reaches the real AUROC on any cut or metric, so p = 1/51 = 0.020, the floor. VERIFIED from the CSVs.
- Controls: no run crosses folds (600 runs). Agent AUROC (last step) of the binary head, seed 0, recomputed with sklearn from the last-step scores and the set's labels: 0.98539, equal to the CSV. Solo-arm rows are recomputed from the same saved held-out scores restricted to the solo arm (subset_metrics asserts the per-step labels equal the saved ones). VERIFIED.

## 2026-11-09, controls_600 held-out author

- Each fold holds exactly one author and each author one fold (step_scores_controls_600_think_heldout_author.csv.gz). Per-author step AUROC of the binary head, seed 1, Luna held out, recomputed with sklearn: 0.91446, equal to authors_controls_600_think_heldout_author.csv. VERIFIED.

## 2026-11-09, SPAR sets (spar_mixed_41, spar_screen_team)

- Labels: team_probes' joined cheating label (transfer_other_sets_scores.csv, y); onsets from the floor rung's call in the cached spar-swarm signals.jsonl, used only where the signals' call count equals the transcript's tool-call count (209 of 209 and 226 of 226 agents). 4 spar_mixed_41 positives have no onset and are left out of that cut's scoring.
- No run crosses folds in either set. Per-step AUROC of the binary head, seed 0, recomputed with sklearn from the saved scores: 0.963906 (spar_screen_team) and 0.814647 (spar_mixed_41), equal to the CSVs. VERIFIED.
- spar_screen_team covers the 226 agents fetched locally of the 450 on the Hub (partial).
