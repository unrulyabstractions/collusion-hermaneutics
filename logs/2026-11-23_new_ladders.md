# 2026-11-23: the three new ladders (Cheating, Coordination, Collusion) on our seven sets

## Submodule merge (spar-guardian-agents 86b4e96, branch board-seed-injection, pushed)
- WHAT: origin/main 18b8830 merged into board-seed-injection; board_seed wired beside main's seed packs; test_board_seed follows main.
- HOW: tests/unit on main's tip 4220 passed; on the merge 4224 passed and 2 failed (both test_board_seed.py), then test_board_seed.py 6 of 6 after the fix (the full suite was not re-run after the fix). Independent review of the conflict resolution: nothing of main's side lost; ours lost only the turn_budget block, as intended; test_board_seed.py and test_seed_packs.py 52 passed.
- RESULT: VERIFIED. Known latent issue: with a seed_pack that holds posts, the board seed is not post 1 (no study sets both). run.json keeps the board seed only in config.board_seed and board_level (the exporter writes no board_seed block; this predates the merge).

## Relabelled runs (data/new_ladders/, untracked)
- WHAT: 102 board-seed and design runs re-exported from their .eval logs (`python -m src regenerate`); 972 controls and SPAR runs from HF spar-swarm with signals recomputed (`python -m src signals`).
- HOW: an independent verifier audited all 1074 folders: new ladder keys in every agent row, signals version 2 equal to the registry, agent sets equal to the activation agents (311 = 311 on board and design), no failed regeneration, Hub v1 rungs identical to the recomputed rungs. Seven runs (one per source) re-run from code gave byte-equal parsed rows.
- RESULT: VERIFIED. Notes: every versions.json stamps code d804d04 because the signals ran on the merge before it was committed (clean tree, mtimes and the seven re-runs show the content is 86b4e96). scratchpad sig_failed.json lists 54 poem runs that later succeeded once their eval logs were fetched. For 79 of 1662 SPAR agents the transcript has more steps than the activations (34 cut at 262,144 tokens): a label can rest on turns the reader never saw; this predates the relabel.

## Labels (hermeneutics/new_ladders.py, results/new_ladders/)
- WHAT: agents.csv (1973 agents), compare.csv, signals_rows.csv; results/matrix_probes_new_ladders/labels/agents.csv (8036 rows, same order as the old table, onsets cleared on our sets, external sets unchanged).
- HOW: an independent verifier recomputed cheat, coord and collude for all 1973 agents with its own code: 0 mismatches; agent identity checked through eval log, run id and transcript name for every row; compare.csv recomputed exactly; the matrix table's external rows byte-identical.
- RESULT: VERIFIED.

## Probe matrix refit (results/matrix_probes_new_ladders/, results/new_ladders/matrix_compare.csv)
- WHAT: the agent-level `matrix` stage, 47 schemes, on the new labels; the comparison with the old cells.
- HOW: an independent verifier recomputed matrix_compare.csv from both cells.csv (all 119 rows equal), recomputed 1634 of 1636 per-unit AUROCs from the saved scores (2 differ by 4e-5 from score rounding), confirmed that the scores' y equals the new labels, that the external-only cells equal the old ones exactly, that n and pos match the label table, and that all 47 schemes are present with no errors in the log.
- RESULT: VERIFIED. Not refit: matrix steps, archs, heatmap; team_probes, attn_probe, the controls Table 1 probes, TF-IDF baselines, judges, controls_nulls, external_probes; the paper (pinned in paper/src/pinned.py).
