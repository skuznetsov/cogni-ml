# Real and shuffled SFT pairs on Crystal repairs

Shuffled pairs produced four compilable repairs out of eight. Real pairs produced two, and the starting `tool-four` adapter produced one. All seven compilable repairs failed at least one target test. Each arm completed 0/8 repairs and 0/4 families.

| Adapter | Insertable | Compiled | All targets | All regressions | Full suite | Both variants in family |
|---|---:|---:|---:|---:|---:|---:|
| `tool-four` | 8/8 | 1/8 | 0/8 | 0/8 | 0/8 | 0/4 |
| Real pairs | 6/8 | 2/8 | 0/8 | 0/8 | 0/8 | 0/4 |
| Shuffled pairs | 6/8 | 4/8 | 0/8 | 1/8 | 0/8 | 0/4 |

The cells show the first blocking stage or the two test-group results after a successful build.

| Family and variant | `tool-four` | Real pairs | Shuffled pairs |
|---|---|---|---|
| Cursor order a, candidate first | Compile | Compile | Compile |
| Cursor order b, checkpoint first | Compile | Compile | Target fail, regressions pass |
| Slot bound a, slot count | Compile | Format | Format |
| Slot bound b, last index | Compile | Format | Target fail, regressions fail |
| Outbox a, inline count | Compile | Target fail, regressions fail | Target fail, regressions fail |
| Outbox b, count alias | Target fail, regressions fail | Target fail, regressions fail | Target fail, regressions fail |
| Backup suffix a, loop | Compile | Compile | Format |
| Backup suffix b, `any?` | Compile | Compile | Compile |

The shuffled cursor b repair returned `left < right` when the caller passed the checkpoint first. It passed the equality and invalid-position checks and failed both direction checks. Both trained adapters returned `return true;` on the two outbox variants, so the empty-batch target failed. Three answers had a trailing `end` on a second line, and the remaining format failure returned a fenced file and replacement wrapper. Compiler errors also included references to variables available only in the caller.

Four of 24 samples stopped at format, 13 at compilation, and seven reached the tests. All generation responses ended with EOS and used the same sampler settings without infrastructure recovery. This is one training schedule per arm, one false-partner permutation, one generation seed, and four synthetic families. Changing the false partner also changes when that example reaches Adam. With zero full repairs, the compile-count difference gives limited evidence about semantic transfer.

## Training protocol

Both arms started from `tool-four` on the same Qwen3.5-2B Q4_K_M GGUF. Native Crystal QLoRA applied rank 8 to Q, K, V, O, gate, up, and down in full-attention blocks 11, 15, 19, and 23. The arms each used all 32 Original examples once. Each batch contained one example whose assistant completion began with `true` and one whose completion began with `false`.

The real pairs share the family, public contract, and requested argument order. In each pair, the caller's `!` changes the two required answers. The shuffled schedule keeps the same `true` rows at every update and permutes their `false` partners with seed 42. It retains zero original partners. One shuffled batch still draws both examples from the same family.

The driver computes two full SFT gradients at the same adapter weights and averages them in FP32. A fresh native Adam optimizer applies one step per batch with learning rate 0.0001, betas 0.9 and 0.999, and epsilon 1e-8. Adam carries both moments and the counter across batches. Each completion has four supervised tokens including EOS, for eight per batch, 128 per arm, and 16 updates.

## Holdout protocol

Four new synthetic Crystal families provide eight executable repairs. Cursor argument order and slot count versus last index change the required helper expression. The pending-count alias and the suffix-scan loop versus `any?` preserve it. The issue states the same public contract for both variants of each family.

The fixture specifications and prompts were frozen before training. The prompt contains the issue and caller/helper source with one `<HOLE>`. Prefixes range from 146 to 192 tokens. Each checkpoint gets one sample per variant at a 4,096-token context limit, generation seed 101, temperature 0.8, top-k 40, and up to 128 new tokens. The original `tool-four` checkpoint serves as the before-training reference.

A normalized single-line replacement is inserted into the helper and built in an isolated Crystal spec fixture. The target and regression groups are evaluated separately. A complete family requires both variants to compile and pass both groups.

The 32 preflight checks compiled all variants of the broken code, teacher edits, sibling teacher edits, and regression traps. All eight broken variants failed a target, all eight teachers passed the full suite, and all eight traps passed both targets and failed a regression. The swapped cursor teachers failed targets and the swapped slot teachers failed boundary regressions. Both invariant families passed when the sibling teacher was applied.

## Verification

The logs contain 32 prefills and 16 Adam updates per arm. Both losses at each update were finite. The first true example gave the same loss at the common starting weights. The native driver checked weights and Adam state around both gradients and all 56 parameter counters after every update. The factors A and B changed in every trained projection of both arms. The two output digests differ.

All 28 native QLoRA specs passed. An independent read of the raw Crystal compile statuses and test summaries reproduced the scores for all 24 generations and 32 preflight checks.

Parallel adapter audits initially interfered with console output during cached verification. The coordinator now audits adapters after joining both training processes. A fresh run reproduced both trained adapter digests, all 24 generation artifacts byte for byte, and every score under the same training and sampling settings.

## Reproduction

Use the populated native QLoRA worktree with the pinned Original archives and `build/bridge.o`.

```sh
paired_driver=build/qlora/scale-2b/paired-batch-sft-v1/train_paired_batch_sft
mkdir -p "$(dirname "$paired_driver")"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_paired_batch_sft.cr --release \
  -o "$paired_driver" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/paired_batch_transfer.py "$paired_driver"
```

Rerun the Python command with the existing binary to verify cached generations and rerun the Crystal checks. Use `--out` with an unused directory for a fresh training and evaluation run. `--preflight-only` freezes and checks the protocol before training.

The default output is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/paired-batch-transfer-v2`.

| Artifact | SHA-256 |
|---|---|
| Protocol | `1948146f79fd0ef030076af17afef4b3e2b9e4fa2748be4495ca902dda4bc610` |
| Results | `4af529809a9d739981ad48efb83bac57f3847b2ac357ad6ec1a17f13a252c060` |
| Real schedule | `616a925929a166da6cd7b46937896ec5e58af23a9ac25db63a1a9d0d2b86a943` |
| Shuffled schedule | `cc504508e81e2aec7497ed690ef8a53c6ea01083b4b452f5a2bd39b9ec0ad7a3` |
| Real adapter | `f4caf2f2ebb0c589c1d431a1982141ce76e94e943c5a9415ac0a28765648d452` |
| Shuffled adapter | `a8d27c1791e6218614ff765410483293d8087265aa4c9688293265875f859a7f` |
