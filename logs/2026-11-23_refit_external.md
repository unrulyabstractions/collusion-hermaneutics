# 2026-11-23 refit, area external (hermeneutics/results/external_probes)

Labels: the controls (controls_600) and the board-seed team workers now take `cheat` from
hermeneutics/results/new_ladders/agents.csv (Cheating attempted or hacked). Goodfire, CheatBench and EvilGenie keep their
own labels. Stages rerun: `transfer`, `transfer_text`, `verify` (to keep VERIFICATION.md's Q3 rows current), `report`.
`traj`, `step`, `labels`, `cache`, `diag` were not rerun; their files are untouched (git shows no change to them).
Run logs: logs/external_refit/{transfer,transfer_text,verify,report}.log (all rc=0).

Code change (hermeneutics/external_probes.py): `new_cheat()` joins agents.csv on (source, run, agent) and asserts every
row is found; `controls_frame()` sets `y` from it (old label kept as `y_old`); `team_frame()` adds `cheat_new`; the team
targets "cheating used or above" and "cheating opened or above" are replaced by one target "cheating attempted or
above". Report: a label-definitions paragraph, a rewritten question-3 reading, and two sentences that claimed the
transfer table reproduces 0.279 now say that held on the old labels.

| Output | How verified | Result |
|---|---|---|
| transfer.csv | Opened it; 148 rows (old 152: the four opened-or-above rows are gone). Positives: controls solo 96 of 373, controls all 196 of 600, team workers 51 of 129; agents.csv gives controls_600 196 (solo arm 96, counted on the subsample's condition column) and board_seed 51 (all 51 on workers, the 5 regulators are 0). Goodfire-trained rows on CheatBench and EvilGenie: 16 rows, every column equal to the old file. | VERIFIED |
| transfer_text.csv | Opened it; 37 rows (old 38). Same positives as above. The 4 Goodfire-trained rows on CheatBench and EvilGenie equal the old file in every column. | VERIFIED |
| transfer_scores.csv.gz | Reloaded: 53,460 rows as before. Goodfire-trained scores (4,956 rows) equal the old file exactly, same keys; y on CheatBench and EvilGenie unchanged. y on controls_600 (2,400 rows) and board_seed (516 rows) equals agents.csv `cheat` row by row. Recomputed with sklearn roc_auc_score: Goodfire to team workers layer 51 0.772750, Goodfire to controls all layer 26 0.556527, controls solo to CheatBench concatenated 0.189967, each equal to transfer.csv. | VERIFIED |
| verification.json / VERIFICATION.md | Rerun of `verify`: all 16 AUROC checks match to 1e-9 (two new Q3 rows: controls solo to Goodfire 0.595574, Goodfire to team workers attempted-or-above 0.731523 on 129 rows, 51 positive); three Hub tensors equal. Diff against the old VERIFICATION.md: only the date and the two Q3 rows changed. | VERIFIED |
| RESULTS.md | Diffed against the old file: changed lines are the new label paragraph, the question-0 sentence on 0.279, the three question-3 headings and tables, and the question-3 reading. Questions 0, 1, 2 and 4 tables are byte-identical. Read the new reading against the tables: each claim checked (pooled inside null, within-run-dir below 0.5 except layers 26 and 38 of solo + cheat, Goodfire to team just above the null 95th). | VERIFIED |
| Within-Goodfire, CheatBench and EvilGenie results (traj.json, step.json, traj_scores_*.csv.gz, step_scores.csv.gz, folds, cheatbench_diagnosis.json) | Not rerun; git diff shows them unchanged; VERIFICATION.md's recomputations of those tables still match. | VERIFIED (unchanged) |
| Bootstrap intervals and permutation nulls of the new rows | Copied from the stage, not recomputed a second way. | UNVERIFIED |

Note for the paper phase: paper/src/tables_1109.py looks up the heading "Trained on controls solo (373 agents, 193
cheat)". The new heading is "Trained on controls solo (373 agents, 96 cheat)"; the solo + cheat heading is now
"Trained on controls solo + cheat (600 agents, 196 cheat)"; the Goodfire heading "Trained on Goodfire judge sets (2833
rollouts, 2083 yes)" is unchanged. The team-worker target is now "team workers (board seed), cheating attempted or
above". The paper reads RESULTS.md at its pinned commit (paper/src/pinned.py EXT), so it does not break until the pin
moves.
