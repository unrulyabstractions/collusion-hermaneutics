# Verification log: team probes

## 2026-11-08 23:20 PST: first board_seed run (134 agents) and controls_600 transfer (600 agents)

| Output | How it was checked | Result |
|---|---|---|
| call index mapping | signals.jsonl tried_call of j8z4k, m9j2m, x4s5k (pilot60_none_s0) matched by hand to the command in source.json: k is the 0-based index over individual tool calls, not over assistant steps | VERIFIED |
| turn alignment | for all 134 agents: meta n_turns = assistant steps in source.json = think tensor rows; tool calls in source.json = signals.jsonl calls | VERIFIED |
| think positions | for all 134 board_seed agents: the token at think_pos is the think-open / think-close id of meta.json (fetch_log.jsonl) | VERIFIED |
| labels | levels.compute on the 43 run dirs: 129 workers + 5 regulators, team floor 23 of 43, no nesting violations | VERIFIED |
| AUROC recomputation | agent_scores.csv, team_scores.csv, step_turn_scores.csv.gz: every probe's think-pair AUROC (and metadata, TF-IDF, think-close rows) recomputed by pair counting and by sklearn roc_auc_score, equal to the tables to 1e-9 | VERIFIED |
| folds | no run in two folds (agent and step probes) | VERIFIED |
| agent counts | 129 workers in the probes + 5 regulators = the 134 board_seed agents of the Hub listing | VERIFIED |
| fast AUROC | the weighted Mann-Whitney AUROC equals sklearn on random scores with ties and weights | VERIFIED |
| step_scores_cheating.png, step_scores_collusion.png | opened and looked at: six agents each, four reaching the level and two not, onset shading, threshold and opened call drawn | VERIFIED |
| setting B code path | sandbox run with synthetic boundary tensors for 60 agents: B tables, counts and report sections produced; numbers synthetic and discarded | VERIFIED (code path only) |
| setting B results | no board_seed agent had turns.safetensors on the Hub at this run | NOT RUN |

## 2026-11-08 23:35 PST: rerun with the committed code, independent verifier, spar_mixed_41

| Output | How it was checked | Result |
|---|---|---|
| independent verifier on commit 317a0e0 | a separate agent recomputed all 60 agent rows, all team rows and all 56 step rows from the saved scores, the bootstrap intervals, the onset switches of all 129 agents, the thresholds and delays of all 56 step rows, read the code for leakage and viewed both PNGs | VERIFIED (no broken item) |
| provenance | the verifier found probes 1 to 5 came from an earlier revision of the script; all stages were rerun with this revision and every probe row reproduced | VERIFIED |
| transfer AUROCs | transfer_scores.csv and transfer_turn_scores.csv.gz now saved; all 30 team-scored transfer rows recomputed with sklearn, equal to 1e-9 | VERIFIED |
| tie noise | mean turn scores rounded to 1e-10 so agents with equal scores stay tied (moved the 5b metadata agent AUROC in the fourth decimal) | FIXED |
| team authors | meta.json and episodes.jsonl: all 134 agents (129 workers, 5 regulators) are DeepSeek V4.1 Flash | VERIFIED |
| spar_mixed_41 labels | one cheat and one no-cheat agent traced by hand to the spar-swarm signals.jsonl rung and the task floor | VERIFIED |
| RESULTS.md wording | null denominators, the per-step null, the anticoord-only negatives of 1b and 5c, and 5d keeping the discussed or solicited workers now stated | FIXED |

## 2026-11-09 01:10 PST: setting B on the full board_seed set (134 of 134 agents with turns.safetensors)

| Output | How it was checked | Result |
|---|---|---|
| boundary files | fetch_log.jsonl for all 134 agents: shape [n_turns, 5, 5120], turn ids 248046 198 248045 74455 198, positions increasing and the last one token before think-open | VERIFIED |
| streamed exact PCA | on 12 agents, the Gram-matrix projections equal a full SVD of the training rows up to sign (max difference 0.002 in z units, float32 accumulation) | VERIFIED |
| setting A rows after the rerun | every A row reproduced the previous run | VERIFIED |
| setting B agent and step rows | all 125 recomputed from agent_scores.csv and step_turn_scores.csv.gz with sklearn, 0 mismatches; row counts equal the kept rows | VERIFIED |
| setting B team rows | all 20 recomputed from team_scores.csv, 0 mismatches | VERIFIED |
| B think pair against A think pair | same agents, two PCA solvers: per-step turn AUROCs agree within 0.01 | VERIFIED |
| PNGs | reopened after the rerun; same six agents and curves | VERIFIED |
| zero-turn agents | one cheatbench agent with zero turns is now skipped at load (it crashed the watch loop's transfer scoring) | FIXED |

## 2026-11-09 02:05 PST: labelled sets solo_design (177 agents) and spar_mixed_41 (209 agents)

| Output | How it was checked | Result |
|---|---|---|
| means-based loading | agent vectors from per-agent turn means equal those from the full tensors (max difference 2e-6, float32) | VERIFIED |
| solo_design labels | levels.compute on the 59 run dirs; rung equals runs.csv rung for 177 of 177; read the board and posted equal runs.csv reads > 0 and posts > 0 for all 177; three workers traced to signals.jsonl | VERIFIED |
| solo_design probes | 225 rows recomputed from agent_scores.csv, 0 mismatches; labels equal the ladder rules in pooled, batch 1 and clean; no run in two folds; 177 labelled = 177 on the Hub | VERIFIED |
| solo_design collusion | every agent at collusion none: no collusion cut exists, the probe is skipped | NOT APPLICABLE |
| solo_design step counts | 3 agents where the reader saw one turn more than source.json has steps; agent-level probes only, not affected | NOTED |
| spar_mixed_41 labels | levels.compute on the 51 spar-swarm run dirs; three workers traced to signals.jsonl | VERIFIED |
| spar_mixed_41 probes | 50 agent rows and 44 team rows recomputed, 0 mismatches; no run in two folds; 209 labelled = 209 on the Hub | VERIFIED |
| spar_mixed_41 thin classes | 12 workers used or above, 7 collusion crossed; read and posted have no contrast (skipped) | NOTED |
| spar_mixed_41 regulator transcript | one regulator rendered 232 of 358 steps; regulators are not probed | NOTED |

## 2026-11-09 02:20 PST: spar_screen_team (450 agents) and the step-to-turn correction of the labelled sets

| Output | How it was checked | Result |
|---|---|---|
| spar_screen_team probes | 75 agent rows and 44 team rows recomputed, 0 mismatches; labels equal the ladder rules; no run in two folds; three workers traced to signals.jsonl; 450 labelled = 450 on the Hub | VERIFIED |
| step-to-turn mapping | label CSVs rewritten: solo_design 174 by index, 3 unmappable; spar_mixed_41 208 by index, 1 unmappable (a regulator); spar_screen_team 381 by index, 37 by tool-call rank, 32 unmappable; unmappable agents carry no per-call step | VERIFIED |
| example binder_minimal_cheat_haiku1/a02 | traced by hand: 29 agent steps, 22 tool-call steps = 22 turns; opened call 3 sits at source step 2, rank 2 among tool-call steps; label opened_step 2 | VERIFIED |
| ladder columns after the rewrite | cheat, coord, coop, collab, collude, scheme identical before and after in all three sets; the only other change is used_onset_source "turns unmappable" for 3 solo_design agents | VERIFIED |
| probe rows after the rewrite | the labelled sets have agent-level and team probes only; they read no per-call column, so no row moves | VERIFIED (no row to move) |

## 2026-11-09 10:40 PST: gaps closed

| Output | How it was checked | Result |
|---|---|---|
| spar_screen_team linear rows, the set's label | linear per-step probe recomputed with sklearn from the base column of step_scores_spar_screen_team_think.csv.gz: step 0.904964, agent (last) 0.851684, 15986 steps, 425 agents; equal to the CSV | VERIFIED |
| probe 6 on spar_screen_team | AUROC recomputed from transfer_other_sets_scores.csv: 0.782063 / 0.756506 / 0.781929 (open+close / open / close), 433 agents, 171 positive; equal to transfer.csv | VERIFIED |
| linear permutation nulls | 5 files, 100 draws each, no repeated draw; real row equals the metrics CSV; mean, 95th percentile and p of all 18 rows equal RESULTS.md | VERIFIED |
| label CSVs against strict nesting | violations counted directly from labels/agents.csv and labels/<set>_agents.csv: 0 in every set; the 3 board_seed agents with collusion above none and collaboration none (cm_anticoord_s4/p3x6n, cm_none_s0/n9s3p, cm_procoordL1_s3/k9p8s) have cooperation 3, 1 and 1, which levels.py allows; no label regenerated, no row rerun | VERIFIED |
| truncation counts | truncated_at in the local meta.json: 1 / 24 / 3 / 6, 34 in all, equal to gaps/truncation_counts.csv | VERIFIED (local) |
