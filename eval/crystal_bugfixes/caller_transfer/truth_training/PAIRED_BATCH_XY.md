# X/Y labels in both candidate orders

Y was the most frequent answer for every adapter with both label assignments included. The shuffled-pair adapter selected the same code in all 32 matched comparisons with A/B. The other two adapters changed some choices and each gained three full-suite passes.

Letter and position counts use 32 presentations per adapter. The last two columns pair the two label assignments at each of the 16 fixed code orders.

| Adapter | Prior B | X | Y | First line | Second line | Kept Y in both assignments | Kept code line in both assignments |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tool-four` | 24/32 | 9/32 | 23/32 | 19/32 | 13/32 | 7/16 | 9/16 |
| Real pairs | 26/32 | 11/32 | 21/32 | 19/32 | 13/32 | 5/16 | 11/16 |
| Shuffled pairs | 26/32 | 6/32 | 26/32 | 14/32 | 18/32 | 10/16 | 6/16 |

All pairs that retained a letter retained Y. For the corresponding A/B comparisons all retained letters were B, in 8/16, 10/16, and 10/16 pairs respectively. The selected line matched its A/B counterpart in 29/32 presentations for `tool-four`, 27/32 for real pairs, and 32/32 for shuffled pairs.

| Adapter | Compiled | Targets | Regressions | Full suite with X/Y | Full suite with Y/X | Full suite before → after |
|---|---:|---:|---:|---:|---:|---:|
| `tool-four` | 32/32 | 29/32 | 28/32 | 14/16 | 11/16 | 22/32 → 25/32 |
| Real pairs | 32/32 | 28/32 | 27/32 | 13/16 | 10/16 | 20/32 → 23/32 |
| Shuffled pairs | 32/32 | 28/32 | 24/32 | 10/16 | 10/16 | 20/32 → 20/32 |

`tool-four` gained one pass each on cursor a, slot a, and slot b. Real pairs gained one each on cursor a, slot a, and both suffix variants, and lost one on cursor b.

Each variant has four presentations from the two row orders and the two label assignments.

| Variant | `tool-four` | Real pairs | Shuffled pairs | Check that fails when the foil is selected |
|---|---:|---:|---:|---|
| Cursor a, candidate first | 3/4 | 3/4 | 2/4 | Target direction checks |
| Cursor b, checkpoint first | 2/4 | 1/4 | 2/4 | Target direction checks |
| Slot a, slot count | 4/4 | 4/4 | 3/4 | Count boundary and empty table regressions |
| Slot b, last index | 2/4 | 1/4 | 1/4 | Last allocated slot regression |
| Outbox a, inline count | 4/4 | 4/4 | 4/4 | Single-record and consecutive-batch regressions |
| Outbox b, count alias | 4/4 | 4/4 | 4/4 | Single-record and consecutive-batch regressions |
| Backup suffix a, loop | 3/4 | 3/4 | 2/4 | Rejection of `.bak.old` |
| Backup suffix b, `any?` | 3/4 | 3/4 | 2/4 | Rejection of `.bak.old` |

All three adapters passed the four presentations of each outbox variant. The slot-count variant also passed all four for `tool-four` and real pairs. Both of those adapters chose the exact suffix match throughout X/Y. With Y/X they chose Y for both suffix variants in both row orders, so the longer-suffix regression failed when Y labeled the substring match.

## Protocol and verification

The [A/B choice run](PAIRED_BATCH_CHOICE.md) and [label swap](PAIRED_BATCH_RELABEL.md) supply the two matched sets of prompts and replies. The new prompts map A to X and B to Y in the answer instruction and option labels. A character-level audit of each of the 32 corresponding prompts found four substituted characters. All source, candidate positions, and individual prefix lengths are identical. The prefixes span 165–207 tokens.

All three adapters and the Qwen3.5-2B Q4_K_M GGUF are pinned to their prior digests. The native sampler uses a 4,096-token context limit, seed 101, temperature 0.8, top-k 40, and up to 128 new tokens. All 96 new responses consisted of a single option letter followed by EOS. The results measure four reused synthetic families at one seed.

Crystal rebuilt and tested the two candidate lines for each of the eight variants. The 16 candidate scores matched the archived controls byte for byte. The raw audit reconstructed all 96 matched choices from the 192 current and prior generation records and checked their outcomes against the Crystal logs. A fresh run reproduced all 96 generation files and the four summary files byte for byte.

## Reproduction

Use the populated native QLoRA worktree containing both prior A/B runs.

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/paired_batch_xy.py
```

The default output is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/paired-batch-choice-xy-v1`. Use an unused directory with `--out` for fresh generations.

| Artifact | SHA-256 |
|---|---|
| Protocol | `b6ea8eefb7522c56ddc57a83badba0aaede04fab35042e2a9674dcd38d004b07` |
| Results | `816372b6c1fb4a2ec3603f4416ffb5e034a14b44acbcb8d3550bf5719c55a432` |
| Prompt plan | `629abd6681bb998f37d30828b418e30b9ac56d963b00b8197b5d699a65fa657b` |
| Candidate checks | `e25b19d19b301e75a83205c854aa89f13cd56a0b4135cc25277bf1b1957c8f57` |
