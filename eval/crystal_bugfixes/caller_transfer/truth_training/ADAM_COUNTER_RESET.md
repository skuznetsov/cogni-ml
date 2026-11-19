# Adam counter reset on the final two QLoRA updates

Clearing the Adam counter along with both moments increased the first update norm from 0.0552–0.0785 to 0.112980–0.113026, a factor of 1.44–2.05. The second update norm increased from 0.0385–0.0574 to 0.0603–0.0698, a factor of 1.12–1.65.

The order gap in mean first-token log-odds grew in all six dataset and target combinations after each update. It grew from 1.04–1.51 to 1.93–2.78 after the first update and from 0.82–1.11 to 1.39–2.07 after the second. With the counter reset, the second update narrowed the gap by 26–31% relative to the first.

The first update solved fewer pairs in three of 12 matched branches and the same number in nine. The second update improved two, worsened five and tied five. On `original` with the `true → false` order, the second update went from four solved pairs to zero at target 10 and from four to eight at target 50. The starting `tool-four` solved seven pairs.

## Checkpoints

Each cell gives the L2 norm of the latest update to the concatenated FP32 LoRA factors, mean natural log-odds of `true` against `false` across 32 prompts, and fully solved pairs out of 16. Both expected two-literal completions must be correct and stop at EOS to solve a pair. Both conditions clear `m` and `v` once before the two updates.

| Dataset | Target | Last two labels | Update | Retained counter | Reset counter |
| --- | ---: | --- | ---: | --- | --- |
| original | 10 | true → false | 9 | 0.0552 / +0.232 / 5 | 0.1130 / +0.805 / 3 |
| original | 10 | true → false | 10 | 0.0404 / +0.298 / 4 | 0.0665 / +0.732 / 0 |
| original | 10 | false → true | 9 | 0.0552 / -1.166 / 6 | 0.1130 / -1.977 / 4 |
| original | 10 | false → true | 10 | 0.0431 / -0.816 / 5 | 0.0695 / -1.334 / 4 |
| original | 25 | true → false | 24 | 0.0597 / +0.592 / 1 | 0.1130 / +1.084 / 1 |
| original | 25 | true → false | 25 | 0.0438 / +0.606 / 1 | 0.0645 / +0.955 / 3 |
| original | 25 | false → true | 24 | 0.0597 / -0.496 / 4 | 0.1130 / -0.986 / 4 |
| original | 25 | false → true | 25 | 0.0469 / -0.223 / 5 | 0.0696 / -0.520 / 4 |
| original | 50 | true → false | 49 | 0.0785 / +0.427 / 5 | 0.1130 / +0.704 / 5 |
| original | 50 | true → false | 50 | 0.0531 / +0.278 / 4 | 0.0603 / +0.486 / 8 |
| original | 50 | false → true | 49 | 0.0785 / -0.921 / 4 | 0.1130 / -1.229 / 4 |
| original | 50 | false → true | 50 | 0.0544 / -0.688 / 4 | 0.0608 / -0.907 / 4 |
| diverse | 10 | true → false | 9 | 0.0552 / +0.133 / 5 | 0.1130 / +0.691 / 5 |
| diverse | 10 | true → false | 10 | 0.0416 / -0.021 / 5 | 0.0676 / +0.383 / 4 |
| diverse | 10 | false → true | 9 | 0.0552 / -0.910 / 4 | 0.1130 / -1.451 / 4 |
| diverse | 10 | false → true | 10 | 0.0385 / -0.844 / 4 | 0.0620 / -1.187 / 4 |
| diverse | 25 | true → false | 24 | 0.0597 / +0.657 / 5 | 0.1130 / +1.150 / 4 |
| diverse | 25 | true → false | 25 | 0.0480 / +0.479 / 6 | 0.0698 / +0.754 / 5 |
| diverse | 25 | false → true | 24 | 0.0597 / -0.443 / 4 | 0.1130 / -0.971 / 4 |
| diverse | 25 | false → true | 25 | 0.0451 / -0.370 / 4 | 0.0652 / -0.729 / 4 |
| diverse | 50 | true → false | 49 | 0.0784 / +0.744 / 4 | 0.1130 / +1.059 / 4 |
| diverse | 50 | true → false | 50 | 0.0564 / +0.556 / 4 | 0.0630 / +0.786 / 4 |
| diverse | 50 | false → true | 49 | 0.0784 / -0.766 / 4 | 0.1130 / -1.120 / 4 |
| diverse | 50 | false → true | 50 | 0.0574 / -0.476 / 4 | 0.0641 / -0.719 / 4 |

The order gap is the mean log-odds for `true → false` minus the mean for `false → true`.

| Dataset | Target | Update | Retained counter | Reset counter |
| --- | ---: | ---: | ---: | ---: |
| original | 10 | 9 | +1.398 | +2.782 |
| original | 10 | 10 | +1.113 | +2.066 |
| original | 25 | 24 | +1.087 | +2.070 |
| original | 25 | 25 | +0.829 | +1.475 |
| original | 50 | 49 | +1.348 | +1.933 |
| original | 50 | 50 | +0.966 | +1.393 |
| diverse | 10 | 9 | +1.044 | +2.142 |
| diverse | 10 | 10 | +0.823 | +1.570 |
| diverse | 25 | 24 | +1.100 | +2.121 |
| diverse | 25 | 25 | +0.849 | +1.482 |
| diverse | 50 | 49 | +1.510 | +2.179 |
| diverse | 50 | 50 | +1.032 | +1.505 |

## Intervention and verification

Each new branch starts at the exact archived `tool-four` continuation weights at step 8, 23 or 48. The Crystal driver loads those factors before creating a fresh native QLoRA trainer. Adam initializes each parameter's moments at zero. The driver checks the empty state at the start and checks all 56 parameter counters and finite state after each update. Each continued branch uses the corresponding frozen order. All 12 first-update losses matched the retained-counter branches exactly.

Training uses Qwen3.5-2B Q4_K_M with training seed 42, rank 8 on seven projections in full-attention blocks 11, 15, 19 and 23, full SFT and Adam with learning rate 0.0001 and zero weight decay. Evaluation uses the same 16 held-out Boolean pairs, 4,096-token context, sampling seed 101, temperature 0.8, top-k 40 and at most 128 new tokens. The first update norm uses the shared checkpoint as its origin. The second uses the first saved state in its own branch. The second update includes the change in gradient induced by the first update.

With both moments zero, restarting the counter makes the corrected first and second moments equal to the current gradient and its square on the first update. In exact arithmetic the coordinate displacement is bounded by the learning rate, and the L2 ceiling is about 0.113047 over 1,277,952 coordinates. For archived Adam update index `t`, set `u = (1 − beta1) / (1 − beta1**t)` and `w = sqrt((1 − beta2) / (1 − beta2**t))`. For archived signed displacement `d` the shared-gradient prediction with the counter reset is `d / (u + (1 − w) * abs(d) / lr)`. The maximum coordinate error against this prediction was below 1e-8 over all first updates. The results cover the two fixed training schedules and one evaluation seed.

The runner's cache recheck reproduced the result file. A separate reader regraded 1,536 raw responses from the two conditions and recomputed all 48 step norms directly from the saved adapter factors.

## Reproduction

Run in the populated native QLoRA worktree with the frozen `both-moments-reset-v1` archive. Use an unused directory for a fresh run.

```sh
run_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/adam-counter-reset-v1
mkdir -p "$run_root"
if [ ! -x "$run_root/native-reset" ]; then
  crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_adam_counter_reset.cr --release \
    -o "$run_root/native-reset" \
    --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
fi
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/adam_counter_reset.py --output "$run_root"
```

Use `python3 -B "$run_root/verify_raw.py" --output "$run_root"` to regrade the recorded run and verify the first displacements against the Adam formula.

| Artifact | SHA-256 |
| --- | --- |
| `protocol.json` | `978b7577429a1db308c3467e85bb02fe8d0b7a6bb6cbb55f563e720b53dfb77f` |
| `results.json` | `9ea01934de315708ffe8ec3e8ba54b944cef2c694187fa6110f79c93d7bd6a61` |
