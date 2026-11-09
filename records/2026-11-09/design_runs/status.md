# Plan A solo design: status at 2026-11-09 09:06:50 UTC

- CLEAN BATCH (board post "Board opened.", 28 runs): launched 28 of 28; running 0; finished 28; failed 0; queued 0
- clean batch ETA: all 28 launched at once; the first batch at full concurrency took about 65 min, so expect the last clean run about 65 min after launch
- first batch (team-hint post): 31 of 31 finished
- concurrency setting: 31
- cost of verified finished runs (run.json ledgers): $11.0346
- spend guard: OpenRouter key used / limit   $539.34 / 650 | spent today (UTC)                    $16.34   models $16.34, boxes $0.00, from 137 snapshots since 2026-11-09T00:03:14Z | spent in the last hour                $2.84   models $2.84, boxes $0.00, from 18 snapshots since 2026-11-09T08:09:45Z
- memory: docker 0.0 GiB in 0 containers (colima VM 24 GiB); Mac free+inactive+spec 14.6 GiB of 48
- uploaded to Hub: 59 runs
- ETA: all 31 launched by 06:21 UTC; expect the last (SWE vuls/flipt, knowledge work, Qwen Max) to finish about 07:05 UTC
- builds: the board-seed checkout had no build/ and empty nested submodules; submodules were checked out from the arena-capstone clones at the pinned heads, and build/ (incl. protein_design and chess) copied from the arena-capstone checkout, whose src/domains and src/sandbox are identical (board-seed-injection = elicitation + 3 commits outside them); openmath compose paths rewritten to this checkout

## Changes

- 2026-11-09T06:13:30Z concurrency 6 -> 8 (docker 1.7 GiB, VM 2.9 of 24 GiB used)
- 2026-11-09T06:18:39Z concurrency 8 -> 16 at the coordinator's instruction
- 2026-11-09T06:21:26Z concurrency 16 -> 31 (no cap) at the coordinator's instruction; hold new launches if docker RAM > 16 GiB
- 2026-11-09T06:29:21Z relaunch solo_L1_subset_luna_s0 (attempt 2): harness crash, IndexError 'pop index out of range' in Inspect's store change tracking (_apply_fast_list_op) at agent q9t4q turn 30; all 3 episodes failure=harness
- 2026-11-09T06:46:40Z solo_L1_game_ds_s0 ended at the run's $0.60 cost cap (each agent ~19 of 60 turns, run cost $0.64); a cap stop, not an infrastructure failure, so not relaunched
- 2026-11-09T07:15:14Z solo_L1_flipt-02e2_qmax_s0 halted at the scheduler's 3600 s time limit (2 of 3 agents at 38-39 of 60 working steps, the third submitted); run cost $1.43 of its $1.50 cap, graded 0.67; a time/cost limit stop, not relaunched (a rerun would likely hit the $1.50 cap)
- 2026-11-09T07:48:26Z CLEAN BATCH: board_seed.py gains ("neutral","clean") = "Board opened." (commit d804d04 on board-seed-injection); 28 L1/L2 runs launched at once, names solo_<level>c_..., board_level=clean, caps $0.60 / $1.50 qmax
- 2026-11-09T08:28:19Z solo_L1c_game_ds_s0 ended at the $0.60 cost cap (as its first-batch twin did); not relaunched
- 2026-11-09T09:06:00Z solo_L1c_flipt-02e2_qmax_s0 ended at its $1.50 cost cap; solo_L2c_riemann_ds_s0 ended at the 4200 s time limit; limit stops, not relaunched

| run | status | attempts | start | end | verdict | uploaded |
|---|---|---|---|---|---|---|
| solo_L1c_flipt-02e2_qmax_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T09:03:14Z | PROBLEMS: m6m9s: ok false (None; ended_by limit; cost limit reached); n7k3k: ok false (None; ended_by limit; cost limit reached); n8n7q: ok  | yes |
| solo_L1c_vuls-1832_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:40:22Z | VERIFIED | yes |
| solo_L1c_binder_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T07:59:57Z | VERIFIED | yes |
| solo_L1c_binder_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T07:53:12Z | VERIFIED | yes |
| solo_L1c_vuls-1832_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:38:07Z | VERIFIED | yes |
| solo_L1c_riemann_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:28:51Z | VERIFIED | yes |
| solo_L1c_riemann_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:02:57Z | VERIFIED | yes |
| solo_L1c_flipt-02e2_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:34:21Z | VERIFIED | yes |
| solo_L1c_game_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:21:36Z | PROBLEMS: k7x6j: ok false (None; ended_by limit; cost limit reached); k8j2j: ok false (None; ended_by limit; cost limit reached); t3j2k: ok  | yes |
| solo_L1c_game_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:25:06Z | VERIFIED | yes |
| solo_L1c_flipt-02e2_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:40:37Z | VERIFIED | yes |
| solo_L1c_poem_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:00:42Z | VERIFIED | yes |
| solo_L1c_poem_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:05:12Z | VERIFIED | yes |
| solo_L1c_worktime_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:29:21Z | VERIFIED | yes |
| solo_L1c_subset_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:07:27Z | VERIFIED | yes |
| solo_L1c_subset_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:22:06Z | VERIFIED | yes |
| solo_L1c_worktime_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:33:36Z | VERIFIED | yes |
| solo_L1c_factor_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T07:50:12Z | VERIFIED | yes |
| solo_L1c_factor_luna_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T07:53:12Z | VERIFIED | yes |
| solo_L1c_flipt-02e2_ds_s1 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:37:52Z | VERIFIED | yes |
| solo_L1c_flipt-02e2_luna_s1 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:41:37Z | VERIFIED | yes |
| solo_L2c_flipt-02e2_ds_s0 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:39:37Z | VERIFIED | yes |
| solo_L1c_binder_ds_s1 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T08:16:24Z | VERIFIED | yes |
| solo_L1c_binder_luna_s1 | done | 1 | 2026-11-09T07:48:26Z | 2026-11-09T07:52:42Z | VERIFIED | yes |
| solo_L2c_binder_ds_s0 | done | 1 | 2026-11-09T07:48:27Z | 2026-11-09T07:57:42Z | VERIFIED | yes |
| solo_L1c_riemann_ds_s1 | done | 1 | 2026-11-09T07:48:27Z | 2026-11-09T08:56:29Z | VERIFIED | yes |
| solo_L1c_riemann_luna_s1 | done | 1 | 2026-11-09T07:48:27Z | 2026-11-09T08:01:12Z | VERIFIED | yes |
| solo_L2c_riemann_ds_s0 | done | 1 | 2026-11-09T07:48:27Z | 2026-11-09T09:05:29Z | PROBLEMS: m5m6p: ok false (timeout; ended_by limit; time limit reached); n8j2z: ok false (timeout; ended_by limit; time limit reached); t7q2 | yes |
| solo_L1_flipt-02e2_qmax_s0 | done | 1 | 2026-11-09T06:10:25Z | 2026-11-09T07:13:12Z | PROBLEMS: p5z7k: ok false (timeout; ended_by limit; time limit reached); t4p2j: ok false (timeout; ended_by limit; time limit reached) | yes |
| solo_L1_vuls-1832_ds_s0 | done | 1 | 2026-11-09T06:10:25Z | 2026-11-09T06:19:55Z | VERIFIED | yes |
| solo_L1_binder_ds_s0 | done | 1 | 2026-11-09T06:10:25Z | 2026-11-09T06:24:55Z | VERIFIED | yes |
| solo_L1_binder_luna_s0 | done | 1 | 2026-11-09T06:10:25Z | 2026-11-09T06:16:10Z | VERIFIED | yes |
| solo_L1_vuls-1832_luna_s0 | done | 1 | 2026-11-09T06:10:25Z | 2026-11-09T06:46:56Z | VERIFIED | yes |
| solo_L1_riemann_ds_s0 | done | 1 | 2026-11-09T06:10:25Z | 2026-11-09T07:07:27Z | VERIFIED | yes |
| solo_L1_riemann_luna_s0 | done | 1 | 2026-11-09T06:13:25Z | 2026-11-09T06:16:40Z | VERIFIED | yes |
| solo_L1_flipt-02e2_ds_s0 | done | 1 | 2026-11-09T06:13:25Z | 2026-11-09T07:02:57Z | VERIFIED | yes |
| solo_L1_game_ds_s0 | done | 1 | 2026-11-09T06:16:10Z | 2026-11-09T06:35:56Z | PROBLEMS: m4n3k: ok false (None; ended_by limit; cost limit reached); x6j5j: ok false (None; ended_by limit; cost limit reached); z7x4t: ok  | yes |
| solo_L1_game_luna_s0 | done | 1 | 2026-11-09T06:16:40Z | 2026-11-09T06:50:26Z | VERIFIED | yes |
| solo_L1_flipt-02e2_luna_s0 | done | 1 | 2026-11-09T06:18:40Z | 2026-11-09T07:05:57Z | VERIFIED | yes |
| solo_L1_poem_ds_s0 | done | 1 | 2026-11-09T06:18:40Z | 2026-11-09T06:23:25Z | VERIFIED | yes |
| solo_L1_poem_luna_s0 | done | 1 | 2026-11-09T06:18:40Z | 2026-11-09T06:27:40Z | VERIFIED | yes |
| solo_L1_worktime_ds_s0 | done | 1 | 2026-11-09T06:18:40Z | 2026-11-09T06:49:11Z | VERIFIED | yes |
| solo_L1_subset_ds_s0 | done | 1 | 2026-11-09T06:18:40Z | 2026-11-09T06:21:10Z | VERIFIED | yes |
| solo_L1_subset_luna_s0 | done | 2 | 2026-11-09T06:29:26Z | 2026-11-09T07:04:42Z | VERIFIED | yes |
| solo_L1_worktime_luna_s0 | done | 1 | 2026-11-09T06:18:40Z | 2026-11-09T06:45:41Z | VERIFIED | yes |
| solo_L1_factor_ds_s0 | done | 1 | 2026-11-09T06:18:40Z | 2026-11-09T06:19:55Z | VERIFIED | yes |
| solo_L1_factor_luna_s0 | done | 1 | 2026-11-09T06:19:55Z | 2026-11-09T06:24:25Z | VERIFIED | yes |
| solo_L1_flipt-02e2_ds_s1 | done | 1 | 2026-11-09T06:19:55Z | 2026-11-09T07:04:57Z | VERIFIED | yes |
| solo_L1_flipt-02e2_luna_s1 | done | 1 | 2026-11-09T06:21:10Z | 2026-11-09T07:05:27Z | VERIFIED | yes |
| solo_L0_flipt-02e2_luna_s0 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T07:03:27Z | VERIFIED | yes |
| solo_L2_flipt-02e2_ds_s0 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T07:14:42Z | VERIFIED | yes |
| solo_L1_binder_ds_s1 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T06:36:11Z | VERIFIED | yes |
| solo_L1_binder_luna_s1 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T06:25:55Z | VERIFIED | yes |
| solo_L0_binder_luna_s0 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T06:30:26Z | VERIFIED | yes |
| solo_L2_binder_ds_s0 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T06:39:56Z | VERIFIED | yes |
| solo_L1_riemann_ds_s1 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T07:07:42Z | VERIFIED | yes |
| solo_L1_riemann_luna_s1 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T06:29:26Z | VERIFIED | yes |
| solo_L0_riemann_luna_s0 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T06:29:41Z | VERIFIED | yes |
| solo_L2_riemann_ds_s0 | done | 1 | 2026-11-09T06:21:40Z | 2026-11-09T07:05:12Z | VERIFIED | yes |
