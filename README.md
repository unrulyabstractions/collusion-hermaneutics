# Hermeneutics for swarm collusion monitoring

Submission to the Apart Research Swarm Alignment 2026 sprint (November 6 to 8, 2026), Track 4, Multi-Agent Oversight and Control. Author: Ian Rios-Sialer, Independent.

We ask whether a small open-weight **reader** model, Qwen3.8-27B, can monitor larger and often closed-weight **author** models by reading their agent transcripts: a linear probe on the reader's hidden states at the reasoning positions is trained to recover a behaviour label (cheating) that was assigned from the authors' actions. We compare the probe with text classifiers, a lexical-only reader null and API judges, test transfer to authors the probe never saw, and run a planted board signal in a three-agent swarm. The report is `report/REPORT_DRAFT.md`; the PDF is built from it.

## Layout

| path | what it holds |
|---|---|
| `report/` | the report (`REPORT_DRAFT.md`), its number ledger (`NUMBERS.md`, one line per number with its source and selector), the abstract, the generator (`gen/gen_report.py`, `gen/report_template.md`) and the LaTeX build (`tex/`, `build_pdf.sh`) |
| `hermeneutics/` | the probe pipeline: transcript rendering, think-position extraction, labels, probes, text baselines, judges, result summaries; `results/RESULTS.md` is script-generated from the CSV and JSON result files beside it |
| `submodule/spar-guardian-agents` | the agent framework that produced the transcripts (branch `board-seed-injection`), including the board-seed mechanism `src/swarm/board_seed.py` |
| `tools/controls_run/` | the scripts that drove the five-author controls run across the hosts (archival; hostnames replaced by placeholders) |
| `publish_apart_swarm/` | the scripts that built, published and verified the controls dataset on Hugging Face |
| `DATASETS.md`, `DATASET_DETAILS.md`, `datasets/` | the survey of transcript datasets and what each can label |
| `VERIFICATION_LOG.md`, `logs/` | the verification log: every result, upload and change with how it was checked |
| `PROJECT_DESCRIPTION.md` | the project vision as written by the author |

`data/` is not tracked. It holds the 4,157-run controls dataset (private Hugging Face dataset `unrulyabstractions/apart-swarm`, available to reviewers on request), the board-seed pilot transcripts, the labels and the rendered-transcript summaries. The AI Village dataset is cited, not redistributed.

## Pre-existing code

The sprint rules ask that code written before the sprint be marked. The sprint opened on Friday, November 6, 2026 (kick-off keynote 21:30 UTC). Two parts of this repository contain code written before then.

1. **`submodule/spar-guardian-agents`.** The agent framework, its tasks, model registry and Inspect integration were written before the sprint (log begins 2026-10-09). Written during the sprint on branch `board-seed-injection`: the board-seed mechanism (`src/swarm/board_seed.py` and its setup solver, commit 2c72d93), the `gpt-5-mini` registry entry and the pilot and dose-response verification entries.
2. **`hermeneutics/`.** Six files were adapted from the author's earlier capstone code, each source last changed on or before 2026-11-06 14:24 UTC (segment and sentence spans, transcript rendering, layer-streaming activation extraction, the cheat-rung label rule, the probe and LEACE machinery, the judge prompt and local scorer). `hermeneutics/PREEXISTING.md` lists each file, its source, what was kept and what changed. Written during the sprint: `think.py`, `subsample.py`, `baselines.py`, `runs.py` (adapted from sprint code), the random-weights path of `extract.py`, `summary.py`, `recompute.py`, `verify.py`, `report.py`, and every result.
3. **Nothing else.** `report/`, `tools/controls_run/`, `publish_apart_swarm/`, the dataset survey and the logs were written during the sprint.

## Reproducing the report

```
.venv/bin/python report/gen/gen_report.py     # rebuilds REPORT_DRAFT.md, NUMBERS.md, ABSTRACT.txt from the result files
bash report/build_pdf.sh                      # Markdown to LaTeX to PDF (xelatex), prints the page count
```

Every number in the tables and every number written in digits in the text has a line in `report/NUMBERS.md` naming its source and selector: a result file and its cell, a run log, a code constant, a repository document, or, for a few facts (such as DeepSeek V4.1 Flash's licence and size, the sprint's dates and tracks, and reference years), the public page and the field read. The tables were checked cell by cell against those sources by independent verification passes; the entries are in `VERIFICATION_LOG.md`.

## LLM use

Claude Code (Anthropic) ran the experiment pipeline, built every table from its cited sources through a number ledger, drafted the text and ran the verification passes under the author's direction. The author set the research questions and the vision, took every design decision and edited the text.
