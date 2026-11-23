# Random-weights reader: plan (not run)

The random-weights reader is a null. A Qwen3.8-27B with the trained architecture and randomly initialised weights reads the same transcripts at the same positions and layers. Each probe is then refit on its states. If a probe on the random reader matches the trained reader, the probe reads the text's surface statistics, not anything the trained model computes. This needs a GPU pass and has not been run. It runs on a rented box once the user approves.

## Subsample

| Set | Agents | Runs | Tokens (sum of meta.json n_tokens) | Longest agent | Turns |
|---|---|---|---|---|---|
| controls_600 | 600 | 600 | 14,182,746 | 99,743 | 10,985 |
| board_seed (team runs) | 134 | 43 | 8,977,815 | 167,827 | 7,281 |
| solo_design (design runs) | 177 | 59 | 7,973,293 | 171,076 | 6,086 |
| total | 911 | 702 | 31,133,854 | | 24,352 |

Every agent of the three sets is in the subsample. No sequence was truncated: full_tokens equals n_tokens for all 911, and all are under the 262,144-token cap.

## Extractor change

The reader is the same `hermeneutics/extract.py` with `--random-init SEED`, which already exists. `load_layer` builds each decoder block from the config, on the meta device, and draws every parameter with `init_random`, the rule of `Qwen3_5PreTrainedModel._init_weights`: Linear and Conv1d weights N(0, 0.02) with zero bias, RMSNorm weight 0 (the norm scales by 1 + weight), gated RMSNorm weight 1, `dt_bias` 1, `A_log = log U(0.01, 16)`. The embeddings are N(0, 0.02), also inside `run_batch`. Any parameter without a rule raises. The seed is `SEED * 1000 + block`, so a rerun gives the same weights. Only the config is needed (`AutoConfig.from_pretrained(...).text_config`). The 52 GB checkpoint is not downloaded.

The positions and layers are unchanged: layers 26, 38, 51 (hidden_states numbering, the pass stops after block 51 of 64), the think pair `think_pos` of every turn, and the five turn-boundary tokens `turn_pos` of every turn.

To guarantee the same tokens and positions as the trained reader, the input does not come from re-rendering. It comes from the Hub copy of the trained reader's own output. One new driver, `hermeneutics/extract_random_hub.py`, about 80 lines, does this per agent:

1. Read `ids` and `think_pos` from `activations/<set>/<run>/<agent>/acts.safetensors` by HTTP range requests over those two tensors only (4 bytes per token, about 125 MB in all; the 300 MB of sentence means are not fetched), and `turn_pos` from `turns.safetensors`.
2. Build the job tuple `extract.run_batch` takes, `(row, agent, ids, units, think, readable, info)`, with `units=None` (no sentence means) and `think` from `think_pos`.
3. Call `extract.run_batch(batch, [26, 38, 51], cfg, None, dev, LINEAR_CHUNK, random_seed=0)`, extended by one optional argument `extra_pos`: per job, the flattened `turn_pos` rows, gathered at each requested layer exactly as `think_pos` is (`hk[pos]`). This is a three-line change in `run_batch`. Without the argument the function behaves as now.
4. Check `ids[think_pos[:, 0]] == 248068` and `ids[think_pos[:, 1]] == 248069` (the think tokens) and `turn_pos[t, -1] + 1 == think_pos[t, 0]` on every turn. Fail the agent otherwise.

## Box time

31,133,854 tokens at 5,000 tokens per second is 6,227 s, about 1.73 h of forward pass. Staging takes about 15 minutes more: the Python environment, the config, and the 125 MB of token ids. That makes about 2.0 h of box time. Batches follow `extract.plan` with `--batch-tokens 2000000`, so the longest agents (171k tokens) fit one batch each. One 48 GB card is enough, because the random blocks are created on the GPU one at a time as in the trained run. Hidden states wait in host RAM at 10 KiB per token, so a 2M-token batch needs about 20 GB, and the box needs at least 32 GB of RAM. The 5,000 tokens per second figure is the coordinator's planning rate. The trained reader measured 2,050 tok/s on an L40S, and 1,175 tok/s under contention. At 2,050 tok/s the pass is 4.2 h.

## Files it writes

Under `acts_random/<set>/<run>/<agent>/` on the box, then uploaded to the Hub dataset `unrulyabstractions/apart-swarm-align` under `activations_random_init0/<set>/<run>/<agent>/`:

| File | Content |
|---|---|
| `acts.safetensors` | `think_L26`, `think_L38`, `think_L51` [n_turns, 2, 5120] fp16; `think_pos` [n_turns, 2] int32; `think_readable` [n_turns] int8; `ids` [n_tokens] int32 |
| `turns.safetensors` | `turn_L26`, `turn_L38`, `turn_L51` [n_turns, 5, 5120] fp16; `turn_pos` [n_turns, 5] int32 |
| `meta.json` | the trained reader's meta.json with `model: random-init-seed0`, `random_init_seed: 0`, `source_acts: activations/<set>/<run>/<agent>` |

Per turn at three layers, fp16: 60 KB for the think pair (2 x 5120 x 2 bytes x 3) and 150 KB for the boundary tokens (5 x 5120 x 2 bytes x 3), so 215,040 bytes per turn and about 5.2 GB for the 24,352 turns. Locally, `team_probes.fetch` with the random prefix writes the same npz caches under `data/acts_hub_random/think` and `data/acts_hub_random/boundary`.

## Probes to rerun on the random reader

The folds, labels and code are unchanged, with the feature caches pointed at the random reader:

1. `matrix_probes` cells for the three sets alone and pooled (B.I, A, B.II): cheating on controls_600, board_seed and solo_design; coordination on board_seed and solo_design; collusion on board_seed. Features: think pair (layers 26+38+51), think pair layer 38, seven tokens. These are the rows the paper compares against the trained reader.
2. `team_probes` agent probes 1a to 3 on board_seed (settings A and B).
3. `controls_nulls prompt` (turn-0 boundary tokens and think-open) on the three sets. It shows how much of the prompt-only AUROC a random reader recovers.
4. `controls_nulls author`: author-probe accuracy on the random reader's states. A random reader that still separates authors would mean author identity sits in token statistics.

The table to report: trained reader, random reader, TF-IDF of the rendered text, and lexical-only (think-window TF-IDF), on the same agents and folds, with the 50-draw label-permutation null on the random-reader rows.
