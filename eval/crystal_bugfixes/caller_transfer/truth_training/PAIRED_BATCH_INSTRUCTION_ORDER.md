# Answer instruction order with fixed X/Y labels

Changing `Reply with only X or Y.` to `Reply with only Y or X.` increased Y selections from 70/96 to 90/96. All 20 changed answers switched from X to Y. The three adapters made identical choices on all 32 revised prompts.

Each adapter saw eight variants in both row orders and both label assignments. The last two columns pair the two label assignments for each of the 16 fixed row orders.

| Adapter | Y selections before → after | X → Y | Y → X | Kept Y across assignments before → after | Kept code line across assignments before → after |
|---|---:|---:|---:|---:|---:|
| `tool-four` | 23/32 → 30/32 | 7 | 0 | 7/16 → 14/16 | 9/16 → 2/16 |
| Real pairs | 21/32 → 30/32 | 9 | 0 | 5/16 → 14/16 | 11/16 → 2/16 |
| Shuffled pairs | 26/32 → 30/32 | 4 | 0 | 10/16 → 14/16 | 6/16 → 2/16 |

After the swap each adapter selected Y in all 16 prompts with Y on the second row and in 14 of 16 with Y on the first row. The two exceptions picked the second line in both label assignments of the two outbox variants with the correct line second.

## Crystal outcomes

| Adapter | Compiled | Target tests before → after | Regressions before → after | Full suite before → after |
|---|---:|---:|---:|---:|
| `tool-four` | 32/32 → 32/32 | 29/32 → 28/32 | 28/32 → 22/32 | 25/32 → 18/32 |
| Real pairs | 32/32 → 32/32 | 28/32 → 28/32 | 27/32 → 22/32 | 23/32 → 18/32 |
| Shuffled pairs | 32/32 → 32/32 | 28/32 → 28/32 | 24/32 → 22/32 | 20/32 → 18/32 |

Seventeen previously passing presentations failed after the swap, and three previously failing presentations passed. With the revised instruction each adapter passed 8/16 in the X/Y label layout and 10/16 in the Y/X layout.

| Variant | `tool-four` before → after | Real pairs before → after | Shuffled pairs before → after | Checks that fail when the foil is selected |
|---|---:|---:|---:|---|
| Cursor a, candidate first | 3/4 → 2/4 | 3/4 → 2/4 | 2/4 → 2/4 | Target direction checks |
| Cursor b, checkpoint first | 2/4 → 2/4 | 1/4 → 2/4 | 2/4 → 2/4 | Target direction checks |
| Slot a, slot count | 4/4 → 2/4 | 4/4 → 2/4 | 3/4 → 2/4 | Count boundary and empty table regressions |
| Slot b, last index | 2/4 → 2/4 | 1/4 → 2/4 | 1/4 → 2/4 | Last allocated slot regression |
| Outbox a, inline count | 4/4 → 3/4 | 4/4 → 3/4 | 4/4 → 3/4 | Single-record and consecutive-batch regressions |
| Outbox b, count alias | 4/4 → 3/4 | 4/4 → 3/4 | 4/4 → 3/4 | Single-record and consecutive-batch regressions |
| Backup suffix a, loop | 3/4 → 2/4 | 3/4 → 2/4 | 2/4 → 2/4 | Rejection of `.bak.old` |
| Backup suffix b, `any?` | 3/4 → 2/4 | 3/4 → 2/4 | 2/4 → 2/4 | Rejection of `.bak.old` |

## Protocol and verification

The [previous X/Y run](PAIRED_BATCH_XY.md) supplies the matched 32 prompts and 96 replies. Each prompt changed by exactly two characters inside the instruction, as checked against the archived plan. All option labels, code, row orders, and individual prefix lengths stayed fixed. Prefixes span 165–207 tokens.

Inference used the same pinned Qwen3.5-2B Q4_K_M GGUF and three existing adapters with a 4,096-token context limit, seed 101, temperature 0.8, top-k 40, and up to 128 new tokens. All 96 responses consisted of one option letter followed by EOS. The comparison measures the effect of reversing the instruction on these four synthetic families at one sampling seed.

Crystal rebuilt and tested both candidate lines for all eight variants. The 16 scores matched the prior controls byte for byte. The raw audit independently recovered the selected lines and test outcomes from the 192 current and prior generation records and the Crystal status files and spec logs. A second fresh run reproduced all 96 current generation files and the protocol, result, prompt, and control files byte for byte.

## Reproduction

Use the populated native QLoRA worktree containing the previous X/Y run and its dependencies.

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/paired_batch_instruction_order.py
```

The default output is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/paired-batch-choice-xy-instruction-yx-v1`. Use an unused directory with `--out` to generate fresh replies.

| Artifact | SHA-256 |
|---|---|
| Protocol | `c058ce13a07dc2aa455cc98ef4117d40967d923b566adf9d453647cd48489ad8` |
| Results | `92ded3258ecbea2a78b3dffceab8f4081c6e43c92f82bf0061554eaa19a919d6` |
| Prompt plan | `a123a81bead2b79640dc0287683aa2401ef9654ebd5e6faa1a7b2999acc1bfdd` |
| Candidate checks | `e25b19d19b301e75a83205c854aa89f13cd56a0b4135cc25277bf1b1957c8f57` |
