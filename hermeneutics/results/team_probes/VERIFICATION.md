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
