# Second-moment reset before one averaged SFT update

Zeroing Adam's second moment increased absolute first-token bias `|b|` at all six starts relative to both controls. Separation `s` fell at five starts and rose at Original 48. Five starts solved zero pairs after the reset. Original 48 solved four pairs in each condition, with two flip pairs replacing two invariant pairs after the reset.

Each cell gives `b / s / complete pairs out of 16` after one update. For each of the eight flip pairs, `b` is the midpoint of the raw first-token log-odds of `true` on the two prompts. The half-difference `s` is oriented toward the required label. The table reports their means.

| Training set | Start | Carry Adam state | Zero `m` | Zero `v` |
|---|---:|---:|---:|---:|
| Original | 8 | -0.4940 / +0.2032 / 6 | -0.2218 / +0.1945 / 6 | -0.5377 / +0.0227 / 0 |
| Original | 23 | +0.1277 / +0.0506 / 5 | +0.0638 / +0.0544 / 4 | +9.1650 / -0.0183 / 0 |
| Original | 48 | -0.2239 / +0.0775 / 4 | -0.1865 / +0.0745 / 4 | -0.2520 / +0.5850 / 4 |
| Diverse | 8 | -0.4616 / +0.0713 / 4 | -0.3684 / +0.0809 / 4 | +3.4779 / +0.0019 / 0 |
| Diverse | 23 | +0.2610 / +0.0239 / 2 | +0.1539 / +0.0243 / 3 | -1.0250 / -0.0005 / 0 |
| Diverse | 48 | +0.0533 / +0.0345 / 4 | +0.0322 / +0.0363 / 4 | +0.2888 / -0.0044 / 0 |

At Original 48 the gained pairs were `semantic-holdout-branch-polarity-z0-q1` and `semantic-holdout-branch-polarity-z1-q0`. The lost pairs were `semantic-holdout-return-layout-z0-q1` and `semantic-holdout-return-layout-z1-q0`. Both controls returned 32 valid completions at every start. With `v=0`, Original 48 returned 28, Diverse 48 returned one, and the other four starts returned zero.

The L2 norm of the change in the saved FP32 LoRA factors was 16.32–49.88 with `v=0`, compared with 0.00759–0.03550 with carry state and 0.00246–0.01241 with `m=0`. Within each matched start the `v=0` change was about 1,400–3,060 times as large as carry. The update retains the accumulated `m` in the numerator while rebuilding `v` from the square of the current mean gradient. The bias and separation measurements capture the effect of this much larger parameter step.

## Method and verification

The Crystal QLoRA driver replays the archived visits from `tool-four` for each training set on Qwen3.5-2B Q4_K_M. LoRA uses rank 8 across seven projections in each of full-attention blocks 11, 15, 19 and 23. At starts 8, 23 and 48, two full-SFT gradients are computed at identical factor weights, each over four target tokens. Their FP32 average is passed to one native Adam step.

The driver verifies the carry update by reproducing its archived adapter digest. It then restores the common snapshot and zeros only the 56 second-moment tensors. An exact check covers the weights, retained first moments and per-parameter counters before treatment. The counters advance to 9, 24 and 49, respectively. Reversing the two gradients in the average reproduces each treatment adapter byte for byte. All 96 prefix updates matched the archived rows, first target IDs and losses.

The sampler used the 32 held-out prompts in 16 Crystal pairs, a 4,096-token context limit, seed 101, temperature 0.8, top-k 40 and up to 128 new tokens. A solved pair requires both sampled two-literal answers to match the truth tables and terminate with EOS. The two historical control sets are reused after validating their protocol pins and checkpoint, audit and prediction digests. An independent check recomputed the scores from 576 saved observations, including 192 new observations. The result covers this holdout and one generation seed.

All 28 native QLoRA specs passed. The trace validator rejected a missing reset event, a reset counter, a mismatched carry digest and an order-dependent treatment. A cached rerun reproduced the result.

## Reproduction

Use the populated native QLoRA worktree with its frozen archives.

```sh
mean_v_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/mean-sft-second-moment-v1
mkdir -p "$mean_v_root"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_mean_sft_second_moment.cr --release \
  -o "$mean_v_root/native-mean" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/mean_sft_second_moment.py --output "$mean_v_root"
```

Reuse the existing binary and Python command to verify cached outputs. Use an unused output directory for a fresh run.

| Recorded artifact | SHA-256 |
|---|---|
| Second-moment protocol | `08c01c5b606797e71a812191253f48c2a0f69fcbc6f392710c572f95d1817a8f` |
| Results | `4ae96141d24c99b5eaf354e0ce54f400f603cb66caecf3bc992feb3f00974824` |
