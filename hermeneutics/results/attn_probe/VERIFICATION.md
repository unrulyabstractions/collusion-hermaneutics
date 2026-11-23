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

## 2026-11-09, controls_600 seven tokens

- All 600 agents have the boundary file. No run crosses folds. Per-step AUROC of the binary head, seed 1, recomputed with sklearn: 0.987832, equal to metrics_controls_600_seven.csv. Baseline rows equal the think-pair run's (0.916, 0.914, 0.943, 0.672, 0.973). VERIFIED.

## 2026-11-09, solo_design and spar_mixed_41 six ladders

- Labels from team_probes' labels/solo_design_agents.csv and labels/spar_mixed_41_agents.csv (viewer/levels.py ladders, same columns as board_seed's agents.csv); the mapper reproduces board_seed's own records exactly (129 of 129 final levels, 0 onset mismatches). Regulators left out.
- solo_design: no run crosses folds; agent AUROC (last step) of the coordination binary head, seed 0, recomputed with sklearn from the last-step scores and the CSV's final coordination level: equal to metrics_solo_design_think.csv. Collusion has no agent at exploit crossed, so that cut is not scored. VERIFIED.
- spar_mixed_41 six ladders: coordination posted is reached by 157 of 158 workers, so that cut is not scored (fewer than 3 agents per class); cheating used has 12 and collusion crossed 7 positive agents, so its intervals are wide.

## 2026-11-09, second verifier (controls and SPAR) and the step-turn fix

- Independent verifier (read-only): 63 metric rows over the 7 controls and SPAR tags reproduced with sklearn to 2.2e-16; folds and held-out-author folds correct; controls and SPAR agent labels equal their sources; solo-arm rows, per-author tables and all 1463 RESULTS.md cells match the CSVs; no leakage. VERIFIED.
- It found one label defect, now fixed: for SPAR (and solo_design) transcripts with agent steps that issued no tool call, the source step index is not the activation turn (e.g. spar_screen_team binder_minimal_cheat_haiku1/a02: 29 steps, 22 turns). attn_probe.py now maps a step to its rank among tool-call steps when the counts call for it, and marks onsets unknown when neither count matches. board_seed onsets are unchanged (0 mismatches). spar_screen_team, spar_mixed_41 (both label modes) and solo_design were rerun. After the fix, flipt_minimal_haiku2/a01's first positive turn is 12 (the tried call's turn), and the per-step AUROC of the spar_screen_team binary head, seed 0, recomputed with sklearn is 0.975351, equal to the CSV. VERIFIED.
- Determinism: two trainings with the same seed and thread count gave identical scores; 3 against 5 threads differ by at most 2e-5 per score.

## 2026-11-09, controls_600 null complete

- 50 draws (null_controls_600_think_0_50.csv, run on MPS; the real scores were trained on CPU). No draw reaches the real AUROC on any metric, so p = 1/51 = 0.020. VERIFIED from the CSV.

## 2026-11-09, spar_screen_team six ladders

- Labels from team_probes' labels/spar_screen_team_agents.csv (405 workers, 135 runs, ds/haiku/luna 135 each), source steps mapped to activation turns. No run crosses folds. Agent AUROC (last step) of the collusion binary head, seed 1, recomputed with sklearn: 0.962952, equal to the CSV. VERIFIED.

## 2026-11-09 10:40 PST, gaps closed

Hand recomputes with sklearn from the saved held-out step scores, with the agent exclusion rebuilt from the label sources (transfer_other_sets_scores.csv y for the set's label, labels/<set>_agents.csv for the ladders, controls_600 subsample.csv) and checked against the saved step, positive-step, agent and positive-agent counts. Every value below equals its metrics CSV to 1e-9 with all four counts equal.

| Table | Row recomputed | Result |
|---|---|---|
| spar_screen_team, the set's label, all agents | binary head seed 0: step 0.958467, agent (last) 0.954854; seed 1: step 0.949176; linear per-step probe: step 0.904964, agent (last) 0.851684; 15986 steps, 425 agents (163 positive) | VERIFIED |
| seven tokens against the think pair | spar_screen_team_seven seed 0 set label 0.974473; spar_screen_team_seven_ladders collusion seed 1 0.954728; spar_mixed_41_seven seed 0 0.881019; spar_mixed_41_seven_ladders cheating seed 1 0.855010; solo_design_seven coordination seed 0 0.901967 | VERIFIED |
| three seeds | board_seed_think_seed2 cheating 0.885220; board_seed_seven_seed2 collusion 0.893922; controls_600_think_seed2 0.976986; controls_600_seven_seed2 0.986708 (593 agents: 7 positives with no onset dropped) | VERIFIED |
| gate sensitivity | board_seed_think_gatecoop collusion seed 1 0.897483; spar_screen_team_think_ladders_gatecoop coordination seed 0 0.973800 | VERIFIED |
| permutation nulls, monitor | all 18 monitor rows recomputed from the null CSVs (mean, 95th percentile, p = (1 + draws at or above real) / (1 + draws)); 50 draws for every tag, no draw index repeated; the three runs in flight finished all draws (spar_screen_team 0-24 and 25-49, spar_mixed_41 0-49) before their 3 h limit | VERIFIED |
| permutation nulls, linear | all 18 linear rows recomputed; 100 draws each; the stored real row equals the metrics CSV (difference 0) | VERIFIED |
| label consistency | nesting rules counted directly from the four label CSVs, independent of levels.py: 0 violations in every set; collusion above none with collaboration none 3 / 0 / 1 / 17 (board_seed / solo_design / spar_mixed_41 / spar_screen_team), all with cooperation above none | VERIFIED |
| truncation | truncated_at counted in every local meta.json: 1 / 24 / 3 / 6 (spar_mixed_41 / spar_screen_team / spar_exploration_single / spar_screen_solo), 34 in all; equal to truncation_counts.csv. The Hub blob comparison was not rerun | VERIFIED (local) |
| unknown onsets | 30 of the 433 labelled spar_screen_team agents and 27 of its 405 workers are unmappable; 24 of the 30 and 21 of the 27 are truncated | VERIFIED |

Fold checks: every new step-score file (5 seven-token tags, 4 seed-2 tags, 4 gate tags: gatecollab and gatecoop on board_seed_think and spar_screen_team_think_ladders) has each run in exactly one of 5 folds, equal to its meta fold_of_run and to the fold of the same run in the think-pair or seed-0/1 run it is compared with (43, 600, 135, 51 and 59 runs). The set's label and the six ladders on spar_screen_team are stratified on different labels, so their folds differ by design; each comparison stays within one labelling.

Report: RESULTS.md regenerated at 10:37 PST; against the 09:47 copy only the null rows and draw counts, the timestamp and three design lines changed.

Independent verifier agent (read-only, 10:42 PST): every cell of the gaps-closed tables recomputed from the CSVs (36 + 18 null rows, 18 seven-token rows, 8 seed rows, 12 gate rows, the before/after and probe 6 tables, the 9 per-set null blocks) matches; no run spans two folds in 15 step-score files; nesting and truncation counts equal. It found no BROKEN item and two wording slips in this entry and one ambiguous sentence in RESULTS.md (433 labelled agents are 388 workers and 45 regulators), all fixed. UNVERIFIED: the regenerated-label columns of the label table (need a levels.py rerun), the Hub-equality columns of the truncation table (need the Hub listing), and the floor-call source steps of truncated_agents.csv.
