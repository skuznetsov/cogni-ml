# Adam first-moment reset on the final two updates

The order effect appeared at the first update of each branch, with or without the first moment inherited from the shared prefix. The `true → false` order gave higher first-token log-odds of `true` on all 32 holdout prompts at both updates for each dataset and target step. The second update reduced the absolute mean gap by 12.3–18.5% after the reset.

The reset shifted the two orders' mean log-odds in the same direction at each dataset, target and update. On `diverse` at step 25, the means after the second update fell by 0.182 and 0.183, while the order gap changed in magnitude from 0.135 to 0.134. The largest attenuation was on `original` at step 10, where the gap fell from 0.630 to 0.560 at the first update and from 0.536 to 0.456 at the second. At the other five targets the absolute change in the gap was at most 0.008 at either update.

Across the 24 matched observations, the reset increased the solved-pair count in seven, decreased it in five and left it equal in twelve. The best reset checkpoint solved six pairs. The starting `tool-four` adapter solved seven.

## Matched checkpoint results

Each cell gives ordinary Adam → Adam with the first-moment reset. Each value gives mean natural log-odds of `true` over the 32 prompts and the number of fully solved pairs out of 16. A solved pair requires both sampled two-literal responses to match their expected answers and stop at EOS.

| Dataset | Target step | Final two labels | After first update | After second update |
| --- | ---: | --- | --- | --- |
| original | 10 | true → false | -0.285 / 6 → -0.072 / 5 | -0.330 / 6 → +0.038 / 6 |
| original | 10 | false → true | -0.915 / 6 → -0.632 / 6 | -0.866 / 5 → -0.418 / 6 |
| original | 25 | false → true | +0.045 / 4 → -0.018 / 5 | +0.132 / 5 → +0.016 / 4 |
| original | 25 | true → false | +0.163 / 6 → +0.101 / 5 | +0.233 / 4 → +0.120 / 6 |
| original | 50 | true → false | -0.188 / 4 → -0.152 / 4 | -0.229 / 4 → -0.160 / 4 |
| original | 50 | false → true | -0.342 / 4 → -0.306 / 4 | -0.361 / 4 → -0.293 / 4 |
| diverse | 10 | true → false | -0.392 / 4 → -0.302 / 5 | -0.495 / 4 → -0.343 / 5 |
| diverse | 10 | false → true | -0.607 / 4 → -0.525 / 4 | -0.678 / 4 → -0.533 / 4 |
| diverse | 25 | false → true | +0.123 / 3 → +0.018 / 4 | +0.203 / 2 → +0.021 / 4 |
| diverse | 25 | true → false | +0.276 / 3 → +0.171 / 2 | +0.338 / 3 → +0.154 / 2 |
| diverse | 50 | true → false | +0.098 / 3 → +0.079 / 3 | +0.105 / 3 → +0.072 / 3 |
| diverse | 50 | false → true | -0.051 / 4 → -0.071 / 4 | -0.021 / 4 → -0.057 / 4 |

The order gaps below subtract the scheduled order's mean log-odds from the swapped order's mean. The scheduled order at step 25 starts with `false`. The scheduled orders at steps 10 and 50 start with `true`.

| Dataset | Target step | First-update gap, ordinary → reset | Second-update gap, ordinary → reset |
| --- | ---: | --- | --- |
| original | 10 | -0.630 → -0.560 | -0.536 → -0.456 |
| original | 25 | +0.118 → +0.120 | +0.101 → +0.104 |
| original | 50 | -0.154 → -0.154 | -0.132 → -0.133 |
| diverse | 10 | -0.215 → -0.222 | -0.183 → -0.190 |
| diverse | 25 | +0.154 → +0.153 | +0.135 → +0.134 |
| diverse | 50 | -0.149 → -0.150 | -0.126 → -0.128 |

## Method

Both datasets replay the native full-SFT trajectory from `tool-four` on the frozen Qwen3.5-2B Q4_K_M GGUF. Training uses Adam at 0.0001, seed 42 and rank 8 on seven projections in full-attention blocks 11, 15, 19 and 23. At updates 8, 23 and 48 the driver saves the common adapter and snapshots its parameters and Adam state. Each order starts from that state with `m` set to zero once. The pre-update checks compare the restored factors, parameter values, second moments and step counters against the snapshot. Adam accumulates `m` normally during both updates.

Both control trajectories reproduce all 100 archived updates, six shared adapters and six ordinary forward endpoint adapters. The loss evaluated before the first update of every reset branch matches its corresponding ordinary branch exactly. The sampler evaluates 24 fresh checkpoints against the 24 frozen ordinary checkpoint observations. The same 16 held-out pairs use the 4,096-token context limit, seed 101, temperature 0.8, top-k 40 and at most 128 new tokens. An independent check regrades the raw outputs and recomputes the log-odds as the log ratio of the raw probabilities.

The comparison measures the carryover of `m` from each fixed prefix and its effect over the next two updates. At the second update the state already includes the first update's effects on the weights and the newly accumulated moment. The comparisons use one training seed, one generation seed and fixed pairs of training examples.

## Reproduction

Use the populated native QLoRA worktree with the ordinary per-update archive.

```sh
run_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/first-moment-reset-v1
mkdir -p "$run_root"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_first_moment_reset.cr --release \
  -o "$run_root/native-reset" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/first_moment_reset.py --output "$run_root"
```

Repeat the Python command to check the completed runs and observations. For a fresh run, point both commands to an unused output directory.

The run folder contains the per-checkpoint observations and both training logs. The SHA-256 digests identify the artifacts scored for this report.

| File | SHA-256 |
| --- | --- |
| `protocol.json` | `6e175594d348637641eadbfae9010886101b9c123ff27d32d0a7fc7e9b4131cf` |
| `results.json` | `88d6f705e50a9f691bd7ab6776cec40d1f8b4720d06756ac7251fe57e5a594f0` |
| `verified-metrics.json` | `044b9b764436aaecd0afb2aee75e92f4ce07eec1d717d4b4d1aaa47911048e0c` |
