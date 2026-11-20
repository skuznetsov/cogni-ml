# A/B labels with candidate lines held fixed

Of the 40 responses that previously chose B, 28 kept B after it moved to the first line and 12 switched to A and kept the second line. Across the 32 presentations from variants where an adapter had answered B in both orders, 26 kept B and six kept the line.

| Adapter | Previous B | Kept B and picked line 1 | Switched to A and kept line 2 | Kept label across all pairs | Kept line across all pairs |
|---|---:|---:|---:|---:|---:|
| `tool-four` | 12 | 8 | 4 | 8/16 | 8/16 |
| Real pairs | 14 | 10 | 4 | 10/16 | 6/16 |
| Shuffled pairs | 14 | 10 | 4 | 10/16 | 6/16 |

Each pair of letters gives the responses when the passing line is first and then second. The arrow compares the original prompts with the relabeled prompts. All three adapters made identical choices after relabeling.

| Variant | `tool-four` | Real pairs | Shuffled pairs | Full suite after relabeling |
|---|---|---|---|---:|
| Cursor a, candidate first | BB → BB | BB → BB | BB → BB | 1/2 |
| Cursor b, checkpoint first | BB → BB | BB → BB | BB → BB | 1/2 |
| Slot a, slot count | BB → BA | BB → BA | BB → BA | 2/2 |
| Slot b, last index | BB → AB | BB → AB | BB → AB | 0/2 |
| Outbox a, inline count | AB → BA | AB → BA | AB → BA | 2/2 |
| Outbox b, count alias | AB → BA | AB → BA | AB → BA | 2/2 |
| Backup suffix a, loop | AB → BB | BB → BB | BB → BB | 1/2 |
| Backup suffix b, `any?` | AB → BB | BB → BB | BB → BB | 1/2 |

For each adapter, all 16 selections compiled, 14 passed targets, 12 passed regressions, and 10 passed the full suite. The cursor errors failed direction targets. Slot b failed boundary regressions in both orders. The two suffix variants failed the `.bak.old` regression when the foil was labeled B. The three variants passing both relabeled orders were slot a and both outbox variants. Both outbox variants also passed both original orders for all three adapters.

## Protocol and checks

The [original A/B run](PAIRED_BATCH_CHOICE.md) supplies both code orders for each of eight variants and the matching baseline responses. The first option label changes from A to B and the second from B to A. The instruction still requests A or B. A character-wise audit checked all 16 prompt pairs. The source, candidate code, line order, system message, sampling settings, and individual prefix lengths were identical. Prefixes range from 165 to 207 tokens.

The frozen Qwen3.5-2B Q4_K_M model and all three adapters retain their original digests. Each presentation receives one sample at 4K, seed 101, temperature 0.8, top-k 40, and a maximum of 128 generated tokens. Every new response contained one A or B token followed by EOS. The analysis pairs the requests by case and by the passing line's physical position.

Crystal rebuilt and ran all 16 candidate lines on the pinned fixtures. The candidate score file matched the original byte for byte. An audit recovered the selected lines directly from all 96 old and new native generation records and checked the scoring against the raw Crystal logs. A fresh run reproduced the 48 new generation files and four summary files byte for byte. These results describe the sampled decisions on four synthetic families at seed 101.

## Reproduction

Use the populated native QLoRA worktree and the original run artifacts.

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/paired_batch_relabel.py
```

The default output is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/paired-batch-choice-relabel-v1`. Use an unused directory with `--out` for fresh generations.

| Artifact | SHA-256 |
|---|---|
| Protocol | `8b5ddd63b353b5a5a52502f462f92e62d9eddaca0d807850261a061ee18edaf3` |
| Results | `b80d8ca3a7af51b0c804cb95f3679e469212981806ef03be8aec8a403fe8e56d` |
| Prompt plan | `1d87ea8cdfb362b152f7d3fa479c0fe23c4e6b7be20714daebafdbf1a93ebf09` |
| Candidate checks | `e25b19d19b301e75a83205c854aa89f13cd56a0b4135cc25277bf1b1957c8f57` |
