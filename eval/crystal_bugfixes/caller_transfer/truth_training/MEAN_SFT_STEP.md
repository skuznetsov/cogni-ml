# Averaged full-SFT gradient with one Adam update

One averaged update matched the better sequential order on complete pairs in five of six comparisons. At Diverse step 23 it solved 2/16 pairs, versus 3/16 for `true → false`. The absolute first-token bias `|b|` grew from the common starting checkpoint in all six comparisons. Original step 23 was the only case with lower `|b|` than both sequential endpoints.

Each cell gives `b / s / complete pairs out of 16`. The start column refers to completed optimizer updates before branching. The two sequential columns show the results after two updates.

| Training set | Start | Shared checkpoint | `true → false` | `false → true` | Averaged gradient |
|---|---:|---:|---:|---:|---:|
| Original | 8 | -0.3502 / +0.2107 / 6 | -0.2714 / +0.1778 / 6 | -0.8158 / +0.1963 / 5 | -0.4940 / +0.2032 / 6 |
| Original | 23 | +0.0496 / +0.0540 / 4 | +0.2562 / +0.0472 / 4 | +0.1535 / +0.0487 / 5 | +0.1277 / +0.0506 / 5 |
| Original | 48 | -0.1867 / +0.0751 / 4 | -0.1871 / +0.0779 / 4 | -0.3231 / +0.0797 / 4 | -0.2239 / +0.0775 / 4 |
| Diverse | 8 | -0.3538 / +0.0819 / 5 | -0.4515 / +0.0643 / 4 | -0.6395 / +0.0609 / 4 | -0.4616 / +0.0713 / 4 |
| Diverse | 23 | +0.1556 / +0.0238 / 3 | +0.4018 / +0.0240 / 3 | +0.2610 / +0.0249 / 2 | +0.2610 / +0.0239 / 2 |
| Diverse | 48 | +0.0301 / +0.0366 / 4 | +0.1398 / +0.0330 / 3 | +0.0097 / +0.0340 / 4 | +0.0533 / +0.0345 / 4 |

The averaged update had higher `s` than both sequential endpoints in four cases, with gains of 0.0004–0.0070 over the larger sequential value. It had lower `s` than the starting checkpoint in four cases. One flip pair was solved at Original step 23 after averaging. All other averaged runs solved their pairs on the invariant cases.

## Method

Starting from `tool-four`, the native QLoRA driver replayed each frozen 32-row dataset's first 48 updates in the archived alternating order. The base was Qwen3.5-2B Q4_K_M, with rank 8 on seven projections in full-attention blocks 11, 15, 19 and 23. Training used seed 42 and the native Adam settings from the ordinary runs. At steps 8, 23 and 48, each branch computed two full-completion SFT gradients at the same weights. Each example supervised four target tokens. The driver passed their FP32 arithmetic mean to Adam once, carrying forward both moments and every parameter's update count.

All 96 replayed updates matched the archived row, first target token and loss. At all six starts the common adapter and the two single-example control adapters matched the archive byte for byte. Each example's loss matched its corresponding archived first update. The optimizer advanced from the starting counter to the next counter for all 56 trainable tensors. Reversing the two operands of the arithmetic mean produced the same adapter bytes at every start. These checks tie both archived sequential endpoints to the same starting conditions.

Evaluation used the same 16 held-out Crystal pairs and 32 prompts. The context limit was 4,096 tokens with seed 101, temperature 0.8, top-k 40 and up to 128 new tokens. For each flip pair, `ℓT` and `ℓF` are the raw first-token log-odds of `true` against `false` on the prompts requiring those respective labels. `b` is the mean of `(ℓT + ℓF) / 2` over eight flip pairs. `s` is the mean of `(ℓT − ℓF) / 2`. A complete pair requires both sampled answers to match the truth tables and end with EOS. A separate check recomputed these metrics from the raw probabilities and response strings.

Each comparison uses the same two examples and eight target tokens on all branches. The comparison covers one training and generation seed. The two regimes differ in optimizer update count as well as how the gradients enter the optimizer.

## Reproduction

Use the native QLoRA worktree with its frozen archives.

```sh
mean_run_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/mean-sft-step-v1
mkdir -p "$mean_run_root"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_mean_sft_step.cr --release \
  -o "$mean_run_root/native-mean" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/mean_sft_step.py --output "$mean_run_root"
```

Run the Python command again to revalidate the cached checkpoints and observations. For a fresh run, use an unused output directory for both the driver and the output. The native QLoRA specs also passed with 28 examples and zero failures.

| Recorded artifact | SHA-256 |
|---|---|
| Protocol | `9540efcb14c9f183d26d0274ff1d42b53484aebdfb774f6ea0d8026a4e209d14` |
| Results | `1bab01d4258800212242fe753b0af08ea9eaaca1f0d064b895d52e35c833fdde` |
