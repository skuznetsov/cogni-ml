# Two-choice first-token training

Both children branch from tool-four on the same Qwen3.5-2B Q4_K_M GGUF. Both use the original 32 examples in the same 50-update order, seed 42, Adam with a learning rate of 0.0001, LoRA rank 8 and seven projections in blocks 11, 15, 19 and 23. The [full-SFT child](README.md) supervises four completion tokens per update. The choice child consumes the same prompt and supervises only the first completion token, which is either `true` (1802) or `false` (3721).

The choice loss for target t is `logsumexp(z_true, z_false) - z_t`. It extracts two rows from the frozen Q6_K output head and reuses the native completion loss and RMSNorm backward path. The training audit checks the original prompt lengths, one target per update, the exact update order, finite loss and changes to both factors in all 28 projections.

The same 16 held-out sign pairs each have a direct caller and a caller with one added exclamation mark. For each pair, t is the first literal required by the inverted caller. We measure the change in `z_t - z_other` and in `P(t | first token is true or false)`, holding t fixed across both callers. A two-choice argmax pair is correct when the higher-logit literal is correct on both callers.

The primary score requires both complete two-literal answers in a pair to be correct under the original sampler and parser. The sampler uses the full vocabulary, temperature 0.8, top-k 40, seed 101, a 4,096-token context and a 128-token response cap. The tool-four and full-SFT scores and logits come from the sealed [previous runs](FIRST_TOKEN.md). The choice adapter was sampled with the same executable, and its first-token readouts were repeated in reverse prompt order.

## Results

| Adapter | Mean conditional P(t), direct → inverted | P(t) rose | Mean log-odds shift | Two-choice argmax pairs | Correct sampled answers | Complete sampled pairs |
|---|---:|---:|---:|---:|---:|---:|
| Tool-four | 27.95% → 36.26% | 15/16 | +0.553 | 2/16 | 17/32 | 2/16 |
| Full SFT | 38.87% → 46.49% | 16/16 | +0.347 | 4/16 | 22/32 | 6/16 |
| First-token choice | 39.59% → 46.71% | 16/16 | +0.349 | 1/16 | 14/32 | 1/16 |

The choice loss preserved the mean log-odds response to negation at roughly a 1.42-fold change in odds. It strengthened the mean conditional shift when the inverted answer starts with `false` from +6.54 to +9.23 percentage points. The shift for `true` fell from +8.69 to +5.02 points. For all eight inverted cases requiring `false`, `true` still had the higher logit. The largest conditional probability for `false` among those cases was 42.49%.

Adding negation raised the conditional probability of the required literal on all ten prior full-SFT errors for both children. The choice adapter fixed zero of these ten sampled answers and introduced eight individual errors. Its one complete sampled pair was already correct under full SFT. Only 30/32 responses parsed as two Boolean literals, and the average full-vocabulary probability mass on `true` and `false` at the first answer slot fell from 98.70% to 75.43%. With argmax restricted to the two literals, the complete-pair count still fell from 4/16 to 1/16.

On the 32 original training prompts, the mean two-class first-token negative log-likelihood decreased from 0.727 for tool-four to 0.646 for the choice adapter. Full SFT reached 0.637 on the same measure. These checkpoint measurements use the native inference readout with the original train messages through the user turn and the first literal of each assistant message as the label.

The 16 pairs span four closely related Crystal templates and one generation seed. The measured effect is limited to these prompts, this adapter initialization, this objective and this update budget.

## Individual pairs

The three numeric columns show conditional P(t) for the direct and inverted caller, followed by the log-odds shift in parentheses. The last column lists which adapters returned both complete answers correctly in the original sampler.

| Template | Public at 0 | Order | Inverted t | Tool-four | Full SFT | First-token choice | Complete sampled pair |
|---|---|---|---|---:|---:|---:|---|
| Mosaic | false | 0, 1 | true | 63.69 → 61.30 (−0.102) | 57.99 → 65.06 (+0.300) | 61.14 → 64.28 (+0.134) | Full SFT |
| Mosaic | false | 1, 0 | false | 3.11 → 12.16 (+1.463) | 16.39 → 28.75 (+0.722) | 13.65 → 28.29 (+0.914) | — |
| Mosaic | true | 0, 1 | false | 6.98 → 17.05 (+1.008) | 23.10 → 30.01 (+0.356) | 20.94 → 30.56 (+0.508) | — |
| Mosaic | true | 1, 0 | true | 40.30 → 51.84 (+0.467) | 56.13 → 62.62 (+0.269) | 57.38 → 61.44 (+0.169) | Full SFT, First-token choice |
| Pine | false | 0, 1 | true | 17.71 → 23.68 (+0.366) | 40.69 → 52.04 (+0.458) | 41.06 → 52.09 (+0.445) | — |
| Pine | false | 1, 0 | false | 51.73 → 55.01 (+0.132) | 39.80 → 41.27 (+0.061) | 38.09 → 42.49 (+0.183) | Full SFT |
| Pine | true | 0, 1 | false | 24.37 → 40.16 (+0.733) | 34.16 → 39.85 (+0.244) | 32.56 → 41.06 (+0.367) | — |
| Pine | true | 1, 0 | true | 18.29 → 22.70 (+0.271) | 42.33 → 50.55 (+0.331) | 43.58 → 49.75 (+0.248) | — |
| Silver | false | 0, 1 | true | 27.81 → 42.48 (+0.651) | 48.30 → 60.88 (+0.510) | 55.54 → 59.73 (+0.172) | Full SFT |
| Silver | false | 1, 0 | false | 32.22 → 42.52 (+0.442) | 31.55 → 33.64 (+0.095) | 33.68 → 37.90 (+0.183) | Tool-four |
| Silver | true | 0, 1 | false | 25.08 → 35.85 (+0.512) | 31.93 → 35.48 (+0.159) | 30.44 → 37.69 (+0.324) | — |
| Silver | true | 1, 0 | true | 25.42 → 33.97 (+0.412) | 48.03 → 57.89 (+0.397) | 52.43 → 56.40 (+0.160) | — |
| Velvet | false | 0, 1 | true | 56.21 → 61.95 (+0.238) | 53.40 → 61.71 (+0.341) | 59.23 → 63.73 (+0.190) | Full SFT |
| Velvet | false | 1, 0 | false | 4.02 → 10.40 (+1.020) | 18.27 → 30.90 (+0.693) | 14.43 → 29.14 (+0.891) | — |
| Velvet | true | 0, 1 | false | 8.04 → 16.39 (+0.808) | 24.30 → 31.95 (+0.380) | 20.08 → 30.56 (+0.561) | — |
| Velvet | true | 1, 0 | true | 42.23 → 52.69 (+0.421) | 55.55 → 61.23 (+0.234) | 59.26 → 62.31 (+0.128) | Tool-four, Full SFT |

## Reproduction

Use the populated native QLoRA worktree. Choose a fresh trial directory for the experiment artifacts.

```sh
trial=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/first-token-loss-replay
trainer="$PWD/build/qlora/scale-2b/train_first_token-replay"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_first_token.cr --release -o "$trainer" --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/fit_first_token.py "$trainer" --output "$trial"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/compare_first_token_loss.py --output "$trial"
```

The output contains the training protocol, step log, checked adapter, training audit, comparison protocol, choice readouts, native generations and scored pairs.

The recorded run completed in 290.2 seconds and changed all 56 LoRA factors. The 24 native QLoRA specs passed, including the two-class loss and gradient, Q6_K row offsets and scales, and the full-vocabulary loss.

| Evidence from the recorded run | SHA-256 |
|---|---|
| Training protocol | `8afecf8756a665eec5d366229711023958cde165a91dc4a739e5a68641ac6544` |
| Training audit | `38b814bf757f1d3b32d006ace83386d4da8349d191700669aab303bb175fbe00` |
| Comparison protocol | `56023bd4c68f1356e2b370b1b514bd4f580009be28118fceb20f62ba34184ab1` |
| Results | `bf47f72830d4c70e1388c20cacc9fea013cbaa4617f2e17c6418a8137e003750` |
| Choice adapter | `16114be5e7d0bc9a1a12900ce1620424cdfd724ed2ef17ac3ad07292724dc92f` |
| Training prompt readout check | `a61f0f4cc183d1dcf193fbd17d3164cdde689d3131d93ea9444913f4085a7605` |
