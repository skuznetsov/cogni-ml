# First-moment reset before one averaged SFT update

Zeroing Adam's first moment reduced the absolute first-token bias `|b|` at all six starts. The set of solved pairs stayed identical in four comparisons. At start 23, Original lost one flip pair and Diverse gained one. The separation `s` rose in four comparisons and fell in two.

Each cell gives `b / s / complete pairs out of 16` after one update.

| Training set | Start | Carry Adam state | Zero `m` |
|---|---:|---:|---:|
| Original | 8 | -0.4940 / +0.2032 / 6 | -0.2218 / +0.1945 / 6 |
| Original | 23 | +0.1277 / +0.0506 / 5 | +0.0638 / +0.0544 / 4 |
| Original | 48 | -0.2239 / +0.0775 / 4 | -0.1865 / +0.0745 / 4 |
| Diverse | 8 | -0.4616 / +0.0713 / 4 | -0.3684 / +0.0809 / 4 |
| Diverse | 23 | +0.2610 / +0.0239 / 2 | +0.1539 / +0.0243 / 3 |
| Diverse | 48 | +0.0533 / +0.0345 / 4 | +0.0322 / +0.0363 / 4 |

At Original step 23 the lost pair was `semantic-holdout-branch-polarity-z0-q0`. At Diverse step 23 the gained pair was `semantic-holdout-branch-permutation-z1-q0`. Both pairs change the required first token between prompts.

## Method and verification

The native QLoRA driver replays 48 updates from `tool-four` per training set using the archived visits, Qwen3.5-2B Q4_K_M and the same rank-8 adapter configuration. At steps 8, 23 and 48 it computes the two full-completion SFT gradients at the same weights. Both examples contribute four target tokens. Both branches pass the same FP32 mean gradient to one native Adam update.

At each start, the driver applies the carry update and records its adapter digest. It restores the pre-update weights and optimizer state, zeros all 56 first-moment tensors, and checks the restored weights, second moments and per-parameter counters before the treatment update. The counter advances by one. The driver rechecks the result with the operands of the mean reversed and restores the common state before continuing. The runner matched all six carry adapter digests to the archived mean-gradient run. The 96 replayed updates matched the archived visits, first target tokens and losses.

Evaluation reuses the 32 frozen prompts in 16 Crystal pairs, the 4,096-token context limit and generation seed 101 with temperature 0.8, top-k 40 and up to 128 new tokens. `b` and `s` respectively average the midpoint and half-difference of raw first-token log-odds of `true` on the eight flip pairs, with the half-difference oriented toward the correct label. A complete pair requires both sampled answers to match the truth tables and end with EOS. The carry predictions are reused by adapter hash. An independent check recomputed both metrics and the pair counts from 384 saved observations across the two conditions. The comparison covers one seed and this holdout.

## Reproduction

Use the populated native QLoRA worktree and its frozen archives.

```sh
mean_reset_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/mean-sft-first-moment-v1
mkdir -p "$mean_reset_root"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_mean_sft_first_moment.cr --release \
  -o "$mean_reset_root/native-mean" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/mean_sft_first_moment.py --output "$mean_reset_root"
```

Reuse the existing binary and rerun the Python command to verify the cached outputs. Use an unused output directory for a fresh run. All 28 native QLoRA specs passed.

| Recorded artifact | SHA-256 |
|---|---|
| First-moment protocol | `0b135f27a614467bc95950382a7f02adf0715ec1e1c35d68cce5af176cf22866` |
| Results | `90a832677fa4cacb5c761f88c49795c502d49e2abddbcbce6684d3cbdb4fc7b4` |
