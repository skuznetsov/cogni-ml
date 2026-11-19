# Per-update timing of the last-two-visit order effect

The between-order difference in mean first-token log-odds appears after the first of the two updates in all six comparisons. At steps 9 and 49 the swap shifts all 32 prompts toward `false` relative to the forward order. At step 24 it shifts all 32 toward `true`. The second update reduces the absolute mean difference by 12.0–15.1% and the direction holds on every prompt.

Each cell gives mean log-odds of `true` and the number of fully solved pairs out of 16. Each arrow in the order columns goes from the first to the second update. The gap is swapped minus forward.

| Training set | Steps | Shared prefix | Forward order | Swapped order | Mean gap after first → second |
|---|---|---:|---|---|---:|
| Original | 8 → 9 → 10 | −0.421 / 6 | `true → false`  −0.285 / 6 → −0.330 / 6 | `false → true`  −0.915 / 6 → −0.866 / 5 | −0.630 → −0.536 |
| Original | 23 → 24 → 25 | +0.028 / 4 | `false → true`  +0.045 / 4 → +0.132 / 5 | `true → false`  +0.163 / 6 → +0.233 / 4 | +0.118 → +0.101 |
| Original | 48 → 49 → 50 | −0.229 / 4 | `true → false`  −0.188 / 4 → −0.229 / 4 | `false → true`  −0.342 / 4 → −0.361 / 4 | −0.154 → −0.132 |
| Diverse | 8 → 9 → 10 | −0.400 / 5 | `true → false`  −0.392 / 4 → −0.495 / 4 | `false → true`  −0.607 / 4 → −0.678 / 4 | −0.215 → −0.183 |
| Diverse | 23 → 24 → 25 | +0.096 / 3 | `false → true`  +0.123 / 3 → +0.203 / 2 | `true → false`  +0.276 / 3 → +0.338 / 3 | +0.154 → +0.135 |
| Diverse | 48 → 49 → 50 | +0.003 / 4 | `true → false`  +0.098 / 3 → +0.105 / 3 | `false → true`  −0.051 / 4 → −0.021 / 4 | −0.149 → −0.126 |

Pair counts first differ between the orders at step 24 on Original and step 49 on Diverse. The second update creates a difference at step 10 on Original and step 25 on Diverse. At step 24 on Original the swap leads by two pairs, and at step 25 the forward order leads by one. The starting `tool-four` adapter solves 7/16 pairs. The highest count among the 30 measured points is 6/16.

## Method

For each dataset, the forward trajectory starts from `tool-four` on the frozen Qwen3.5-2B Q4_K_M GGUF. It uses native full-completion SFT with Adam at 0.0001, training seed 42 and rank 8 on seven projections in full-attention blocks 11, 15, 19 and 23. At steps 8, 23 and 48, the driver saves the common adapter and copies the factors and Adam moments and step counters. It executes the swapped two-update branch, restores the common training state once, and resumes the forward trajectory. Each visit supervises four target tokens.

All 100 forward updates and 12 swapped updates match the archived visit, first target token and loss. All 12 final adapters match the archived files byte for byte. Their archived predictions are reused. The other 18 adapters are measured with the same sampler and first-token readout. A separate check reproduces the odds from raw probabilities and the solved-pair counts from the sampled answer strings.

Evaluation uses the same 16 held-out pairs from four Crystal families, with 32 prompts balanced on the expected first literal. A complete pair requires both sampled two-literal answers to match their truth tables. Generation uses a 4,096-token context limit, seed 101, temperature 0.8, top-k 40 and up to 128 new tokens. Log-odds use the raw first-token probabilities before sampling. Each comparison covers one fixed pair of training examples and one training and generation seed.

## Reproduction

Use the populated native QLoRA worktree and its frozen archives.

```sh
run_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/per-update-order-v1
mkdir -p "$run_root"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_order_branches.cr --release \
  -o "$run_root/native-branches" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/per_update_order.py --output "$run_root"
```

Run the Python command again to revalidate the cached checkpoints and observations. For a fresh run, compile the driver under an unused output directory and pass that directory to `--output`.

| Recorded artifact | SHA-256 |
|---|---|
| Protocol | `f9ffa5ab57ec680b8ca11ab20edf3dfcde962cc182e548d6749d7fca28085d42` |
| Results | `0f280efa78d3cb8571956746783dce843659f8063a99094527e76def60d78321` |
