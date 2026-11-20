# A/B repair selection after paired SFT

With a passing candidate supplied, `tool-four` selected a full-suite pass in 12 of 16 presentations. The real-pair and shuffled-pair adapters made identical selections, with 10 passes each. Requiring a pass with the candidate in either position gave `tool-four` 4 of 8 variants and each trained adapter 2 of 8. In the earlier free-generation run, each adapter completed 0 of 8 on these variants.

| Adapter | Valid choices | Compiled | All targets | All regressions | Full suite | Variants passed in both orders | Families passed in both orders |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tool-four` | 16/16 | 16/16 | 14/16 | 14/16 | 12/16 | 4/8 | 2/4 |
| Real pairs | 16/16 | 16/16 | 14/16 | 12/16 | 10/16 | 2/8 | 1/4 |
| Shuffled pairs | 16/16 | 16/16 | 14/16 | 12/16 | 10/16 | 2/8 | 1/4 |

Each cell below shows the two choices with the passing line displayed as A and then B. AB passes both presentations. BB passes only when B is correct.

| Variant | `tool-four` | Real pairs | Shuffled pairs | Failed check when the foil is selected |
|---|---|---|---|---|
| Cursor a, candidate first | BB | BB | BB | Target direction checks |
| Cursor b, checkpoint first | BB | BB | BB | Target direction checks |
| Slot a, slot count | BB | BB | BB | Index equal to the count and empty table |
| Slot b, last index | BB | BB | BB | Last allocated slot |
| Outbox a, inline count | AB | AB | AB | Single-record and consecutive batches |
| Outbox b, count alias | AB | AB | AB | Single-record and consecutive batches |
| Backup suffix a, loop | AB | BB | BB | Rejection of `.bak.old` |
| Backup suffix b, `any?` | AB | BB | BB | Rejection of `.bak.old` |

All adapters passed 8/8 with the working line as B. With the working line as A, `tool-four` passed 4/8 and each trained adapter passed 2/8. Every adapter chose B in both orders on all four variants whose correct expression changes with the caller. Both trained adapters also chose B in both orders on the two suffix variants. That gave the two trained arms two extra regression failures when the failing substring match was B.

The supplied candidates make successful repair reachable and remove the need to emit Crystal code. These results measure selection among supplied lines on four reused synthetic families. The fixed A/B labels also coincide with list positions, and the generation uses one seed. The data support a difference in choices on the suffix variants for these checkpoints and settings.

## Protocol

The study reuses the three adapters, GGUF, caller/helper code, issue descriptions, and executable fixtures pinned in [the free-generation run](PAIRED_BATCH_TRANSFER.md). It replaces only the output instruction and adds two candidate lines. For a family whose required expression changes with the caller, the foil is the sibling variant's passing expression. For the two invariant families it is the existing regression trap. Both display orders are frozen before sampling. Each request receives one sample at a 4,096-token context limit, generation seed 101, temperature 0.8, top-k 40, and 128 maximum new tokens. Prefixes span 165–207 tokens.

Crystal built and ran all 16 candidate lines on the original isolated fixtures. Each correct line passed the full suite. Both cursor foils failed targets and passed regressions. All six other foils passed targets and failed at least one regression. The scorer maps an EOS-terminated, whitespace-trimmed A or B to the corresponding tested line and uses that line's measured result. All 48 responses contained one token followed by EOS, with no recovery.

## Reproduction

Use the populated native QLoRA worktree with the pinned outputs from [the free-generation run](PAIRED_BATCH_TRANSFER.md).

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/paired_batch_choice.py
```

The command defaults to `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/paired-batch-choice-v1`. Rerunning in the same directory rebuilds the candidate fixtures and verifies the cached choices. Pass an unused directory to `--out` for fresh generations. The `--preflight-only` flag freezes the prompts and executes the candidate fixtures.

The raw audit reproduced all test groups and selected scores from the Crystal run logs, the printed option text, and the recorded native replies. A fresh run reproduced all 48 generation records and the four summary artifacts byte for byte.

| Artifact | SHA-256 |
|---|---|
| Protocol | `c99015e67db0894bdb9be9ecf7dd9bbf2ad5a44146fdff6f7bf0ae176525a93a` |
| Results | `669b8ea8159241b5c5a38310c1e43cb2df865e109c0dcae404be2b5afb52a1a6` |
| Candidate checks | `e25b19d19b301e75a83205c854aa89f13cd56a0b4135cc25277bf1b1957c8f57` |
| Prompt plan | `d9ff0acc7148ee136b985266ae0c2f153e9c82ab0e32a608004b089584a3717c` |
