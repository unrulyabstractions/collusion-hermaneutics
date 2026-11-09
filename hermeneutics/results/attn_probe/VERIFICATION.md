# Verification of the attention monitor outputs

## 2026-11-09, board_seed, think pair

- Folds: every one of the 43 runs sits in exactly one fold (step_scores_board_seed_think.csv.gz, fold per run); the fold of each of the 129 agents equals its fold in team_probes' step_turn_scores.csv.gz (129 of 129). VERIFIED.
- AUROC by hand: the per-step AUROC of the cheating-used binary head, seed 0, recomputed from the saved step scores with a rank (Mann-Whitney) formula and with sklearn: 0.88256, equal to metrics_board_seed_think.csv. The agent AUROC (last step) of the collusion binary head, seed 1, recomputed with sklearn from the last-step scores and the final labels in labels_board_seed.jsonl: 0.83591, equal to the CSV. VERIFIED.
- Linear baseline: the per-step linear probe on the same folds gives 0.889 (cheating used) and 0.849 (collusion crossed), equal to team_probes' step_probes.csv rows 5a and 5d for probe Lall think-open+close. VERIFIED.
- Plot score_vs_turn_cheating_board_seed_think.png opened and viewed: six panels, the six agents of team_probes' step_scores_cheating.png, both seeds and the linear probe drawn, shaded positive span from the estimated used onset. VERIFIED.
- Labels: the final cheating, collusion and coordination levels of every record equal team_probes' labels/agents.csv (asserted while building labels_board_seed.jsonl). VERIFIED.
