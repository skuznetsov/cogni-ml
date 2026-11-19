# Adam second-moment reset on the final two updates

Clearing the second moment once increased the L2 norm of the first LoRA factor update by 435–1,234× across the 12 paired branches. The second update was 6.1–140.1× larger. The updated factors and optimizer states were finite at each checkpoint.

Ordinary Adam gave higher mean log-odds of `true` to the `true → false` order at each of the 12 dataset, target and update comparisons. Clearing `v` reversed the mean gap in seven of them. This included both updates on `original` at 25 and `diverse` at 10 and 25. On `original` at 50, the gap after the reset was positive after the first update and negative after the second.

After the first update, three reset branches solved more pairs than ordinary Adam, eight solved fewer and one tied. After the second update, four solved more, six solved fewer and two tied. On `diverse` at 10, none of the 32 responses in any of the four reset checkpoints fit the required format. Both `original` branches at 50 rose from four to seven solved pairs after the second update, matching the starting `tool-four` score.

## Matched checkpoint results

Each cell gives ordinary Adam → Adam after the second-moment reset. Each value gives mean natural log-odds of `true` over 32 prompts and the number of fully solved pairs out of 16. A solved pair requires two correct sampled two-literal answers that stop at EOS. For target steps 10, 25 and 50, the first update here is step 9, 24 or 49, respectively.

| Dataset | Target step | Final two labels | After first update | After second update |
| --- | ---: | --- | --- | --- |
| original | 10 | true → false | -0.285 / 6 → -0.504 / 0 | -0.330 / 6 → -0.577 / 4 |
| original | 10 | false → true | -0.915 / 6 → -1.889 / 0 | -0.866 / 5 → -1.921 / 0 |
| original | 25 | true → false | +0.163 / 6 → -0.206 / 3 | +0.233 / 4 → -0.473 / 4 |
| original | 25 | false → true | +0.045 / 4 → +0.624 / 0 | +0.132 / 5 → +0.833 / 4 |
| original | 50 | true → false | -0.188 / 4 → -0.021 / 3 | -0.229 / 4 → -0.707 / 7 |
| original | 50 | false → true | -0.342 / 4 → -0.716 / 6 | -0.361 / 4 → -0.084 / 7 |
| diverse | 10 | true → false | -0.392 / 4 → -1.142 / 0 | -0.495 / 4 → -1.283 / 0 |
| diverse | 10 | false → true | -0.607 / 4 → +0.364 / 0 | -0.678 / 4 → +0.446 / 0 |
| diverse | 25 | true → false | +0.276 / 3 → +0.370 / 3 | +0.338 / 3 → +0.282 / 4 |
| diverse | 25 | false → true | +0.123 / 3 → +2.236 / 4 | +0.203 / 2 → +2.337 / 4 |
| diverse | 50 | true → false | +0.098 / 3 → +1.200 / 4 | +0.105 / 3 → +0.201 / 2 |
| diverse | 50 | false → true | -0.051 / 4 → -0.755 / 3 | -0.021 / 4 → -0.231 / 4 |

The order gaps subtract the mean log-odds of the `false → true` order from those of the `true → false` order.

| Dataset | Target step | First-update gap, ordinary → reset | Second-update gap, ordinary → reset |
| --- | ---: | --- | --- |
| original | 10 | +0.630 → +1.385 | +0.536 → +1.344 |
| original | 25 | +0.118 → -0.830 | +0.101 → -1.306 |
| original | 50 | +0.154 → +0.695 | +0.132 → -0.623 |
| diverse | 10 | +0.215 → -1.506 | +0.183 → -1.729 |
| diverse | 25 | +0.154 → -1.866 | +0.135 → -2.055 |
| diverse | 50 | +0.149 → +1.955 | +0.126 → +0.432 |

## Method

The native trainer replays full-completion SFT on Qwen3.5-2B Q4_K_M from `tool-four`, with seed 42, Adam learning rate 0.0001 and rank 8 on seven projections in full-attention blocks 11, 15, 19 and 23. For each target, it takes one snapshot at update 8, 23 or 48. Both orders load the same factors, first moments and update counters. The trainer clears all second-moment tensors once and then lets Adam accumulate them normally over the two updates.

The pre-update loss matched the ordinary branch on the first visit of each of the 12 treated branches. The replays matched all 100 ordinary update records. The six shared and six ordinary forward endpoint adapters matched byte for byte. The driver checks the finiteness of updated factors and moments and verifies the Adam counters before each treated checkpoint is saved. The independent reader measures actual displacement of the serialized FP32 LoRA factors, from the common state to the first checkpoint and from the first checkpoint to the second. At the first update, every coordinate that moved under ordinary Adam also moved in the same direction by at least as much after the reset.

Evaluation reuses the 16 held-out pairs and the frozen ordinary observations. Each treated checkpoint uses the same 4,096-token limit, seed 101, temperature 0.8, top-k 40 and up to 128 new tokens. Log-odds come from the two first-token logits. The independent check obtains them from the raw probability logs and rechecks the sampled answers against the expected pairs.

Immediately after clearing `v`, the first update forms its denominator from the current squared gradient and the retained bias-correction counter. A coordinate with a small current gradient can therefore have a small denominator while historical `m` still drives the numerator. The measurements describe this particular intervention with retained `m` and counters. The second update already incorporates the changed weights, gradient and newly accumulated `v`. The comparison uses fixed datasets and one training and generation seed.

## Reproduction

Use the populated native QLoRA worktree and its frozen ordinary per-update archive.

```sh
run_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/second-moment-reset-v1
mkdir -p "$run_root"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_second_moment_reset.cr --release \
  -o "$run_root/native-reset" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/second_moment_reset.py --output "$run_root"
```

Run the Python command again to revalidate the completed checkpoints and observations. For a fresh run, compile to an unused output directory and pass that directory to `--output`.

The run directory contains the raw observations, both training logs and the independent regrade with per-checkpoint factor displacements. These SHA-256 digests identify the artifacts used here.

| File | SHA-256 |
| --- | --- |
| `protocol.json` | `02f2ad06cf322930ddf6a7f49830c1368055131b0ff379193643f94edc4afaa3` |
| `results.json` | `dfa2b50b2123081ca7de31865bbef083deab13bae35bf92c7de897bc38477555` |
| `verified-metrics.json` | `716d5d3043e6ec848212daa5cf7b3836b66888ac590acc19a36e42411902303c` |
