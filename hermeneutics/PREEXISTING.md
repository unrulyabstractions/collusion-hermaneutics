# Pre-existing code

The hackathon rules ask that code written before the sprint be marked. Every file below was adapted from the
author's earlier arena-capstone repository. The sprint opened on Friday, November 6, 2026 (kick-off keynote 21:30 UTC);
every source named below was last changed on or before 2026-11-06 14:24 UTC, and the commit and date of its last
change are given beside it. The adaptation is described per file; the rest of the file is pre-sprint code. Files not
listed were written during the sprint.

| file here | pre-sprint source | what was kept | what changed |
|---|---|---|---|
| `hermeneutics/spans.py` | `extraction/spans.py` (5480448, 2026-11-03) | segment tagging, sentence units, token ranges (`tag_segments`, `split_units`, `build`) | the two `.eval`-reading functions removed (`render.py` renders); docstring |
| `hermeneutics/render.py` | `swarm/render.py` (8143add, 2026-11-05), `extraction/check_render.py` (beb8e23, 2026-11-03) | `to_template`, `render`, `TEMPLATE_KWARGS`, `_text`, `_readable`, `_reasoning`, `to_openai`, `to_tool`, `_key`, the append-only check of `agent_sequences` | reads `ModelEvent.input` (the new logs hold inputs as references, which `read_eval_log` resolves) instead of `call.request["messages"]`; the consecutive-assistant merge path dropped (never seen here); one agent per run; `sequence()` adds think positions and readability per turn; the summary CLI |
| `hermeneutics/extract.py` | `extraction/extract.py` (e278df9, 2026-11-03), `swarm/acts.py` (8143add, 2026-11-05) | layer-streaming forward pass (`Weights`, `load_layer`, `apply_layer`, chunked RMSNorm and MLP, `pool`), batch planning by token budget, resume by `meta.json`, non-finite check | think-position capture (`think_L{L}` tensors), run list from `runs.py`, layers 26/38/51, no Hub upload |
| `hermeneutics/labels.py` | `script/groups_data.py` (cc8edc8, 2026-11-06; `RUNG_LEVEL`, `CHEATED_FROM`, `cheat_floor`, `simple_rung`) | the per-task cheating threshold and the cheat / no cheat / other rule | reads the rung from `signals.jsonl` rows; the coarse heuristic label follows arena-capstone `publish_apart_swarm/build_snapshot.py` (2026-11-07 02:00 UTC) |
| `hermeneutics/probes.py` | `analysis/teacher_confound/run.py` (ca1cc47, 2026-11-06) | `Leace`, `zs` (PCA + z-score), `choose_C`, `fit_score`, `boot_ci`, `auc`, `onehot`, the fold design (StratifiedGroupKFold by run), the teacher probe, within-teacher and leave-one-teacher-out loops, the label-shuffle null, the tables | new `load()` over the think-position cache; author map ds/gpt5m/haiku/luna/qwen; loops over layers x positions x pooling; the solo arm reported first; a full-dimension probe variant, a condition readout, "onset" per-turn labels with a turn-position baseline; the within-agent step analysis dropped |
| `hermeneutics/judge.py` | `analysis/qwen_judge/run.py` (9157891, 2026-11-06) | the judge prompt wording (`INTRO`, the question), the task-head and transcript-tail cuts, the 500-character cut per tool result, the one-pass local scorer (streamed blocks, final norm, lm_head on the last position, score = log P(Yes) - log P(No) with thinking off) | prompts built from `render.py` messages instead of spans.jsonl; the OpenRouter scorer with a dollar cap; the analysis over the probe's arms and scopes |

New in the sprint: `hermeneutics/think.py`, `hermeneutics/subsample.py`, `hermeneutics/baselines.py` (TF-IDF text
baselines), `hermeneutics/runs.py` (adapted from `publish_apart_swarm/build_snapshot.py`, itself written during the
sprint on 2026-11-07), the random-weights reader path of `extract.py` (`init_random`), `hermeneutics/summary.py`,
`hermeneutics/recompute.py`, `hermeneutics/verify.py`, `hermeneutics/report.py`, `hermeneutics/__init__.py`, the
results and this file.
