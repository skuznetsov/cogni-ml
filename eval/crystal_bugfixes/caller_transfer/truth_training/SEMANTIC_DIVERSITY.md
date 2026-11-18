# Full SFT on more varied Crystal caller transformations

Replacing the 32 original training prompts with 32 examples spanning argument swaps, Boolean comparisons and equivalent expressions reduced complete holdout pairs from 4/16 to 2/16. The shared tool-four starting adapter solved 7/16.

## Matched training and preflight

Both full-SFT checkpoints start from tool-four on Qwen3.5-2B Q4_K_M. The pinned native trainer uses Adam at 0.0001, training seed 42 and rank 8 on Q, K, V, O, gate, up and down in blocks 11, 15, 19 and 23. Both runs visit the same 50 row indices and have the same answer at every visited position. Each answer contributes four targets including EOS, for 200 supervised targets. The integer-zero instrument prompt is shared. The original checkpoint was reused after the training log, populated audit and adapter hash were revalidated by the same training harness.

The new train and holdout splits each contain four families, crossed with public contract polarity and requested input order. Each family has two implementations. Two families per split change the required truth table between implementations and two preserve it. Holdout uses fresh control-flow templates. Before training, Crystal executed the public caller on both inputs under each of the four possible helper truth tables. Exactly one table satisfied each of the 64 prompts. The run sealed the 256 oracle checks, disjoint prompts and train/holdout inputs before the first update.

All 28 projections and 56 LoRA factors updated. The new run took 299.1 seconds. The archived control took 292.2 seconds. Mean training prompt lengths were 207.9 and 195.5 tokens respectively across the 32 rows. These are two runs with different input lengths, so their timings give the recorded cost of each arm.

## Fresh holdout

All three checkpoints received the same 32 prompts with 4,096 context, seed 101, temperature 0.8, top-k 40 and up to 128 new tokens. A complete pair requires both sampled answers to match the executed truth tables. A first-token pair requires the correct literal to have the higher logit on both sides when comparing `true` with `false`.

| Checkpoint | Correct responses | Complete pairs | Flip pairs | Invariant pairs | First-token pairs |
|---|---:|---:|---:|---:|---:|
| Tool-four | 21/32 | 7/16 | 1/8 | 6/8 | 8/16 |
| Original full SFT | 16/32 | 4/16 | 0/8 | 4/8 | 5/16 |
| Diverse full SFT | 14/32 | 2/16 | 0/8 | 2/8 | 4/16 |

| Held-out transformation | Tool-four responses / pairs | Original SFT responses / pairs | Diverse SFT responses / pairs |
|---|---:|---:|---:|
| Polarity in `if`, flip | 4/8 · 0/4 | 4/8 · 0/4 | 4/8 · 0/4 |
| Argument permutation in `case`, flip | 5/8 · 1/4 | 3/8 · 0/4 | 3/8 · 0/4 |
| Argument permutation with negation, invariant | 8/8 · 4/4 | 8/8 · 4/4 | 6/8 · 2/4 |
| Equivalent guard returns, invariant | 4/8 · 2/4 | 1/8 · 0/4 | 1/8 · 0/4 |

The new SFT lost two previously solved sampled pairs, both in permutation with negation, and gained zero. In both lost cases the correct first literal was `true`, and the sampler drew `false` after the correct log-odds margin fell from 0.436 to 0.293 and 0.413 to 0.309. Across all 32 prompts, `true` had the higher of the two logits in 12 cases for tool-four, 29 for original SFT and 32 for diverse SFT. The mean directed log-odds shift toward the required first literal on the eight flip pairs was +0.342, +0.120 and +0.041 respectively. The mean log-odds of `true` were similar across the two full-SFT checkpoints (+0.344 and +0.335). The prompt-level range narrowed from [−0.253, +0.917] to [+0.044, +0.585], leaving all 16 false-first cases with the wrong first-literal leader.

A second run revalidated both adapters, executed the oracle again and reproduced all recorded generations and readouts. An independent scorer reproduced the pair totals from raw answer strings and first-literal probabilities. Although the inherited parser accepts capitalization, the strict lowercase pass produced the same correctness totals. The original SFT produced one `no, true` and one `True, false`, both incorrect. All responses from the parent and diverse checkpoints matched the lowercase two-literal grammar.

The holdout uses four synthetic families under one generation seed. The comparison estimates the effect of this particular selection of more varied training prompts, which also changes prompt lengths. The score measures recovery of an executable Boolean contract.

## Reproduction

Run from the populated native QLoRA worktree. Use an unused output directory to start a fresh run.

```sh
trial=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-replay
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/semantic_diversity.py --output "$trial"
```

The expected signals are 256 executed Crystal checks, 50 audited updates with the archived visit schedule, 200 supervised targets, and scores for all three checkpoints on the 16 held-out pairs. The recorded output directory is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1`.

| Recorded artifact | SHA-256 |
|---|---|
| Frozen protocol | `12e0901a6846b3c009f584db3161ea0b756cfe89965b78b57c4fb970b4ec71bc` |
| Training audit | `4bfef31ef2a3c753fb1b9dfe6788db9da7ef68a2dab383900e01ee55b9219fe7` |
| Evaluation protocol | `595c87f6de78e0abb03590080b8b46aa3f3366825d8af42f26f9804087cfec12` |
| Results | `0526f81359f7c472ed38d4fa394c2de528fe52764550892e0a8442d2370d5e8f` |
| Diverse adapter | `eef59bb818266c1394a8ae6be59b4b7cc1adbe98e9b6b99f0c1a4bcb9ff82f1c` |
