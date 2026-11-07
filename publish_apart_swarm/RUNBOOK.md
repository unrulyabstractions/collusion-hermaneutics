# Publish apart_controls → HF dataset `unrulyabstractions/apart-swarm` (PRIVATE)

Publishes the local `data/apart_controls/` elicitation transcripts + labels to the private HF
dataset as a snapshot. Reuses the run driver's validity+dedup logic. Safe to re-run; it rebuilds
staging from scratch and re-uploads as a new commit (only changed/new files transfer).

## Prerequisites
- `HF_TOKEN` set in env with write access to `unrulyabstractions/*` (never print it).
- `hf` CLI on PATH (`~/.local/bin/hf`). Install: `curl -LsSf https://hf.co/cli/install.sh | bash`.
- The run driver's incremental rsync has landed the runs you want (for the FINAL publish, have the
  driver force a final full sync first so nothing is only-on-host, then run this).

## Steps
```sh
cd /Users/unrulyabstractions/work/arena-capstone/publish_apart_swarm
STAGE=/Users/unrulyabstractions/work/arena-capstone/data/.hf_staging_apart_swarm

# 1. Build a stable, valid-only staging tree + manifest.jsonl + README.md.
#    Keeps ALL valid copies (canonical in apart_controls/, extras in extra_duplicate_samples/).
#    Add --complete ONLY at ALL_COMPLETE so the README/stats mark it the final set (default: in-progress).
python3 -I build_snapshot.py "$STAGE"              # in-progress snapshot
# python3 -I build_snapshot.py "$STAGE" --complete  # final set (at ALL_COMPLETE)

# 2. Upload (resumable, parallel; single logical commit). ~740MB / ~16k files for the full set.
hf upload-large-folder unrulyabstractions/apart-swarm "$STAGE" --type dataset --num-workers 8 --no-bars

# 3. Verify the upload independently (count + per-path size diff + sha256 vs source + round-trip)
python3 -I verify_upload.py "$STAGE"   # exits 0 == VERIFIED, 1 == FAILED
```

## Notes
- The dataset is PRIVATE and must stay private (research transcript data). Do not flip it public.
- Validity gate per copy (runs BEFORE any canonical choice, so no broken copy is ever published):
  `episodes.jsonl` parses with `ok==true`, `trajectories/` non-empty, a `signals` team-row label exists.
- Duplicates (a 4-orchestrator restart bug produced duplicate valid copies of some
  `run_name = {task}_{cell}_{model}_s{seed}` across hosts): NO valid transcript is dropped. One
  canonical copy per run_name goes to `apart_controls/`; surplus valid copies go to
  `extra_duplicate_samples/<model>/<task>/<cell>/s<seed>__<host>/`. Canonical rule (driver-confirmed):
  among valid copies prefer `ended_by` submit > end > other, then earliest `.eval`-log timestamp, then
  host-alpha. The manifest has one row per copy; `key` is unique (`run_name`, or `run_name__<host>` for
  extras), `canonical` marks the primary. Filter `canonical==true` for the clean uniquely-keyed set.
- Seed semantics: `seed` does NOT change the task instance — each task is one fixed benchmark instance;
  seeds vary only agent-id salt + stochastic sampling. Duplicate copies are extra replicate samples.
- Layout in the repo: `apart_controls/<model>/<task>/<cell>/s<seed>/...` + `extra_duplicate_samples/...`,
  top-level `manifest.jsonl` and `README.md`.
- `STAGE` defaults to `<repo>/data/.hf_staging_apart_swarm` (gitignored) if you omit the argument.

## FINAL (`--complete`) publish — gated checklist (coordinator-required, 2026-11-07)
Do NOT run the final publish until BOTH hold, then follow every step; this is a one-time, correctness-critical publish.

0. GATE — the run driver's ALL_COMPLETE ping MUST include a byte-level per-host filesystem sweep result:
   every run dir AND any only-on-host files enumerated and compared by SIZE against local, with
   **0 missing / 0 mismatch**, so host-completeness is proven before rebuild. If that proof is not in
   the ping, request it and wait — do not rebuild on an unproven sync.
1. Rebuild the FULL set with the complete flag:
   `python3 -I build_snapshot.py "$STAGE" --complete`   (this flips the README/stats off "in-progress").
2. Re-upload: `hf upload-large-folder unrulyabstractions/apart-swarm "$STAGE" --type dataset --num-workers 8 --no-bars`.
3. Re-verify the WHOLE uploaded set (NOT just the delta): `python3 -I verify_upload.py "$STAGE"`.
   Its per-path size diff covers every file in the repo (0 mismatch / 0 missing / only-repo=.gitattributes),
   plus a sha256 sample vs original source + manifest/README round-trip. Must exit 0.
4. Confirm the live README no longer says "in-progress" (status == "complete (final set)").
5. Append a dated entry to `/Users/unrulyabstractions/work/arena-capstone/VERIFICATION_LOG.md`:
   canonical/extra/total counts, the sweep proof, the verify result, and the HF commit.
- Keep the dataset PRIVATE throughout. Never make it public.
