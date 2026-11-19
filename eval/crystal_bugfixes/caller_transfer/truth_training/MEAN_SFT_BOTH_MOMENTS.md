# Both Adam moments reset before one averaged SFT update

Clearing both moments reduced the matched step norm by a factor of 209–904 compared with clearing only `v`. All six starts returned 32 valid completions. Relative to carrying Adam state, four starts solved the same number of pairs and two solved one fewer. Relative to clearing only `m`, four tied and two solved one fewer.

The L2 norm measures the change in all 1,277,952 serialized FP32 LoRA parameters from the common checkpoint.

| Training set | Start | Carry Adam state | Zero `m` | Zero `v` | Zero `m` and `v` |
|---|---:|---:|---:|---:|---:|
| Original | 8 | 0.03100 | 0.01241 | 45.59613 | 0.05519 |
| Original | 23 | 0.01265 | 0.00278 | 38.70615 | 0.05962 |
| Original | 48 | 0.00759 | 0.00289 | 16.32213 | 0.07820 |
| Diverse | 8 | 0.03550 | 0.00568 | 49.88460 | 0.05517 |
| Diverse | 23 | 0.01798 | 0.00384 | 37.16587 | 0.05963 |
| Diverse | 48 | 0.00923 | 0.00246 | 22.29062 | 0.07805 |

Each cell gives `b / s / complete pairs out of 16` after one update. On the eight flip pairs, `b` is the mean midpoint of the first-token log-odds of `true`. The signal `s` is the mean half-difference between the log-odds on the prompts that require `true` and `false`, in that order.

| Training set | Start | Carry Adam state | Zero `m` | Zero `v` | Zero `m` and `v` |
|---|---:|---:|---:|---:|---:|
| Original | 8 | -0.4940 / +0.2032 / 6 | -0.2218 / +0.1945 / 6 | -0.5377 / +0.0227 / 0 | +0.1233 / +0.1654 / 5 |
| Original | 23 | +0.1277 / +0.0506 / 5 | +0.0638 / +0.0544 / 4 | +9.1650 / -0.0183 / 0 | +0.4111 / +0.0601 / 4 |
| Original | 48 | -0.2239 / +0.0775 / 4 | -0.1865 / +0.0745 / 4 | -0.2520 / +0.5850 / 4 | -0.1205 / +0.0536 / 4 |
| Diverse | 8 | -0.4616 / +0.0713 / 4 | -0.3684 / +0.0809 / 4 | +3.4779 / +0.0019 / 0 | -0.5992 / +0.0744 / 4 |
| Diverse | 23 | +0.2610 / +0.0239 / 2 | +0.1539 / +0.0243 / 3 | -1.0250 / -0.0005 / 0 | +0.1007 / +0.0315 / 2 |
| Diverse | 48 | +0.0533 / +0.0345 / 4 | +0.0322 / +0.0363 / 4 | +0.2888 / -0.0044 / 0 | +0.1757 / +0.0435 / 4 |

Relative to carry, clearing both moments reduced `|b|` at three starts and increased `s` at four. Original 8 gained two solved flip pairs and lost three invariant pairs. Diverse 48 gained one flip pair and lost one invariant pair. Original 48 tied all three controls on complete pairs. Clearing only `v` at that start solved two flip pairs and two invariant pairs, while clearing both moments solved four invariant pairs.

With both moments cleared, the numerator and denominator of the next update use the same mean gradient. At a nonzero coordinate, `m` is proportional to the gradient and `sqrt(v)` is proportional to its magnitude before bias correction. The ideal-arithmetic L2 upper bounds using the retained counters are 0.05525, 0.05984 and 0.07864 at starts 8, 23 and 48. The measured norms were close to these bounds and were 1.55–10.31 times as large as carry. Clearing `v` alone keeps the historical first moment even where the current gradient is small.

## Method and verification

The native Crystal QLoRA driver replays the archived visits from `tool-four` on Qwen3.5-2B Q4_K_M for both 32-example training sets. LoRA uses rank 8 across seven projections in each of full-attention blocks 11, 15, 19 and 23. At starts 8, 23 and 48, two full-SFT gradients are computed at identical weights, each over four target tokens. Their FP32 average is passed to one native Adam step.

The driver first reproduces the archived carry digest at each start. It then restores the common weights and all 56 moment pairs with zeros in both moments. The exact check also compares the retained per-parameter counters. The counters advance from 8 to 9, 23 to 24 and 48 to 49. Reversing the two gradients in the average reproduced each treatment adapter byte for byte. All 96 prefix updates matched the archived rows, first target IDs and losses.

Evaluation uses the original 32 holdout prompts in 16 Crystal pairs, a 4,096-token context limit, seed 101, temperature 0.8, top-k 40 and up to 128 new tokens. A solved pair requires both sampled two-literal answers to match the truth tables and terminate with EOS. The three control runs were reused after verifying protocol pins and all checkpoint, audit and observation digests. A separate calculation reproduced the norms and scored all 768 observations, including 192 new observations. The results cover one update per start and one generation seed on this holdout.

All 28 native QLoRA specs passed. The trace validator rejected a missing reset event, a reset counter, a changed carry digest and an order-dependent treatment. A cached rerun reproduced the result.

## Reproduction

Use the populated native QLoRA worktree with its frozen archives.

```sh
mean_mv_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/mean-sft-both-moments-v1
mkdir -p "$mean_mv_root"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_mean_sft_both_moments.cr --release \
  -o "$mean_mv_root/native-mean" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/mean_sft_both_moments.py --output "$mean_mv_root"
```

Reuse the existing binary and Python command to verify cached outputs. Use an unused output directory for a fresh run.

| Recorded artifact | SHA-256 |
|---|---|
| Both-moments protocol | `95700992bbc3268146ddceda9d7c1f4279fe3496349c223ae9fce66dba78de7d` |
| Results | `2e98563442604edd5cec002346a9bb3e681e1d56b4c54e205f8f531ee0d18bbc` |
