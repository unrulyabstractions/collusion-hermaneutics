# Pre-existing code

The hackathon rules ask that code written before the sprint be marked. Every file below was copied from the
arena-capstone repository (`/Users/unrulyabstractions/work/arena-capstone/OLD/`, archived before the sprint) and adapted
here. The adaptation is described per file; the rest of the file is pre-sprint code. Files not listed were written
during the sprint.

| file here | pre-sprint source | what was kept | what changed |
|---|---|---|---|
| `hermeneutics/spans.py` | `OLD/extraction/spans.py` | segment tagging, sentence units, token ranges (`tag_segments`, `split_units`, `build`) | the two `.eval`-reading functions removed (`render.py` renders); docstring |
| `hermeneutics/render.py` | `OLD/swarm/render.py`, `OLD/extraction/check_render.py` | `to_template`, `render`, `TEMPLATE_KWARGS`, `_text`, `_readable`, `_reasoning`, `to_openai`, `to_tool`, `_key`, the append-only check of `agent_sequences` | reads `ModelEvent.input` (the new logs hold inputs as references, which `read_eval_log` resolves) instead of `call.request["messages"]`; the consecutive-assistant merge path dropped (never seen here); one agent per run; `sequence()` adds think positions and readability per turn; the summary CLI |
| `hermeneutics/extract.py` | `OLD/extraction/extract.py`, `OLD/swarm/acts.py` | layer-streaming forward pass (`Weights`, `load_layer`, `apply_layer`, chunked RMSNorm and MLP, `pool`), batch planning by token budget, resume by `meta.json`, non-finite check | think-position capture (`think_L{L}` tensors), run list from `runs.py`, layers 26/38/51, no Hub upload |
| `hermeneutics/labels.py` | `OLD/script/groups_data.py` (`RUNG_LEVEL`, `CHEATED_FROM`, `cheat_floor`, `simple_rung`) | the per-task cheating threshold and the cheat / no cheat / other rule | reads the rung from `signals.jsonl` rows; the coarse heuristic label follows arena-capstone `publish_apart_swarm/build_snapshot.py` (2026-11-07 02:00 UTC) |
| `hermeneutics/runs.py` | arena-capstone `publish_apart_swarm/build_snapshot.py` (`inspect_copy` gates, `canon_key`) | validity gates (ok, trajectories, team row) and the canonical-copy rule | rewritten as a run-list function over a parametrised data root |
| `hermeneutics/probes.py` | `OLD/analysis/teacher_confound/run.py` | `Leace`, `zs` (PCA + z-score), `choose_C`, `fit_score`, `boot_ci`, `auc`, `onehot`, the fold design (StratifiedGroupKFold by run), the teacher probe, within-teacher and leave-one-teacher-out loops, the label-shuffle null, the tables | new `load()` over the think-position cache; author map ds/gpt5m/haiku/luna/qwen; loops over layers x positions x pooling; the solo arm reported first; a full-dimension probe variant, a condition readout, "onset" per-turn labels with a turn-position baseline; the within-agent step analysis dropped |

New in the sprint: `hermeneutics/think.py`, `hermeneutics/subsample.py`, `hermeneutics/__init__.py`, the results and
this file.
