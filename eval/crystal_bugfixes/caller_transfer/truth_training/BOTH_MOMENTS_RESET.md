# Both Adam moments reset on the final two updates

The first LoRA factor update had an L2 norm of 0.0552–0.0785 after the joint reset, 1.42–10.54 times the matched ordinary step. The second update measured 0.0385–0.0574, 1.18–7.71 times ordinary. The joint reset reduced the norm relative to the second-moment-only reset by factors of 75–738 on the first update and 2.11–111.47 on the second.

The `true → false` order gave higher mean first-token log-odds at both updates for every dataset and target. The joint reset increased this gap over ordinary Adam at all 12 comparisons. The gap arose at the first update and then shrank by 20–32% at the second. Clearing only `v` had reversed the gap in seven comparisons.

After the first update, the joint reset solved more pairs than ordinary Adam in five of 12 comparisons, fewer in two and the same number in five. After the second update the counts were four, two and six. On `diverse` at target 10, all four joint-reset checkpoints produced 32 formatted answers and solved four or five pairs. The four second-moment-only checkpoints produced zero formatted answers. At target 25 in the `true → false` order, the joint reset took the second checkpoint from four to one pair on `original` and from three to six on `diverse`. The largest joint-reset count was six. The starting `tool-four` score was seven.

## Paired checkpoint results

Each value gives the L2 norm of the update to the concatenated LoRA factors, mean natural log-odds of `true` over 32 prompts and fully solved pairs out of 16. A solved pair requires both expected two-literal answers with an EOS stop. The ordinary and single-moment scores come from the frozen matched archives.

| Dataset | Target | Last two labels | Update | Ordinary | Reset m | Reset v | Reset both |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| original | 10 | true → false | 9 | 0.0319 / -0.285 / 6 | 0.0238 / -0.072 / 5 | 19.3831 / -0.504 / 0 | 0.0552 / +0.232 / 5 |
| original | 10 | true → false | 10 | 0.0275 / -0.330 / 6 | 0.0168 / +0.038 / 6 | 0.3946 / -0.577 / 4 | 0.0404 / +0.298 / 4 |
| original | 10 | false → true | 9 | 0.0377 / -0.915 / 6 | 0.0133 / -0.632 / 6 | 30.3644 / -1.889 / 0 | 0.0552 / -1.166 / 6 |
| original | 10 | false → true | 10 | 0.0291 / -0.866 / 5 | 0.0194 / -0.418 / 6 | 0.9302 / -1.921 / 0 | 0.0431 / -0.816 / 5 |
| original | 25 | true → false | 24 | 0.0160 / +0.163 / 6 | 0.0068 / +0.101 / 5 | 11.7578 / -0.206 / 3 | 0.0597 / +0.592 / 1 |
| original | 25 | true → false | 25 | 0.0122 / +0.233 / 4 | 0.0051 / +0.120 / 6 | 0.2680 / -0.473 / 4 | 0.0438 / +0.606 / 1 |
| original | 25 | false → true | 24 | 0.0107 / +0.045 / 4 | 0.0050 / -0.018 / 5 | 12.9481 / +0.624 / 0 | 0.0597 / -0.496 / 4 |
| original | 25 | false → true | 25 | 0.0128 / +0.132 / 5 | 0.0056 / +0.016 / 4 | 0.1546 / +0.833 / 4 | 0.0469 / -0.223 / 5 |
| original | 50 | true → false | 49 | 0.0074 / -0.188 / 4 | 0.0083 / -0.152 / 4 | 6.4811 / -0.021 / 3 | 0.0785 / +0.427 / 5 |
| original | 50 | true → false | 50 | 0.0081 / -0.229 / 4 | 0.0056 / -0.160 / 4 | 0.1352 / -0.707 / 7 | 0.0531 / +0.278 / 4 |
| original | 50 | false → true | 49 | 0.0136 / -0.342 / 4 | 0.0086 / -0.306 / 4 | 5.9213 / -0.716 / 6 | 0.0785 / -0.921 / 4 |
| original | 50 | false → true | 50 | 0.0070 / -0.361 / 4 | 0.0056 / -0.293 / 4 | 0.1145 / -0.084 / 7 | 0.0544 / -0.688 / 4 |
| diverse | 10 | true → false | 9 | 0.0330 / -0.392 / 4 | 0.0093 / -0.302 / 5 | 40.7625 / -1.142 / 0 | 0.0552 / +0.133 / 5 |
| diverse | 10 | true → false | 10 | 0.0331 / -0.495 / 4 | 0.0104 / -0.343 / 5 | 4.6350 / -1.283 / 0 | 0.0416 / -0.021 / 5 |
| diverse | 10 | false → true | 9 | 0.0390 / -0.607 / 4 | 0.0130 / -0.525 / 4 | 35.6617 / +0.364 / 0 | 0.0552 / -0.910 / 4 |
| diverse | 10 | false → true | 10 | 0.0326 / -0.678 / 4 | 0.0095 / -0.533 / 4 | 0.2084 / +0.446 / 0 | 0.0385 / -0.844 / 4 |
| diverse | 25 | true → false | 24 | 0.0209 / +0.276 / 3 | 0.0067 / +0.171 / 2 | 12.2608 / +0.370 / 3 | 0.0597 / +0.657 / 5 |
| diverse | 25 | true → false | 25 | 0.0185 / +0.338 / 3 | 0.0076 / +0.154 / 2 | 0.2642 / +0.282 / 4 | 0.0480 / +0.479 / 6 |
| diverse | 25 | false → true | 24 | 0.0167 / +0.123 / 3 | 0.0080 / +0.018 / 4 | 19.4611 / +2.236 / 4 | 0.0597 / -0.443 / 4 |
| diverse | 25 | false → true | 25 | 0.0181 / +0.203 / 2 | 0.0070 / +0.021 / 4 | 0.1113 / +2.337 / 4 | 0.0451 / -0.370 / 4 |
| diverse | 50 | true → false | 49 | 0.0116 / +0.098 / 3 | 0.0068 / +0.079 / 3 | 11.9994 / +1.200 / 4 | 0.0784 / +0.744 / 4 |
| diverse | 50 | true → false | 50 | 0.0081 / +0.105 / 3 | 0.0047 / +0.072 / 3 | 0.1435 / +0.201 / 2 | 0.0564 / +0.556 / 4 |
| diverse | 50 | false → true | 49 | 0.0106 / -0.051 / 4 | 0.0065 / -0.071 / 4 | 10.9659 / -0.755 / 3 | 0.0784 / -0.766 / 4 |
| diverse | 50 | false → true | 50 | 0.0083 / -0.021 / 4 | 0.0049 / -0.057 / 4 | 0.1779 / -0.231 / 4 | 0.0574 / -0.476 / 4 |

The gap subtracts the `false → true` order's mean log-odds from those of the `true → false` order.

| Dataset | Target | Update | Ordinary | Reset m | Reset v | Reset both |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 10 | 9 | +0.630 | +0.560 | +1.385 | +1.398 |
| original | 10 | 10 | +0.536 | +0.456 | +1.344 | +1.113 |
| original | 25 | 24 | +0.118 | +0.120 | -0.830 | +1.087 |
| original | 25 | 25 | +0.101 | +0.104 | -1.306 | +0.829 |
| original | 50 | 49 | +0.154 | +0.154 | +0.695 | +1.348 |
| original | 50 | 50 | +0.132 | +0.133 | -0.623 | +0.966 |
| diverse | 10 | 9 | +0.215 | +0.222 | -1.506 | +1.044 |
| diverse | 10 | 10 | +0.183 | +0.190 | -1.729 | +0.823 |
| diverse | 25 | 24 | +0.154 | +0.153 | -1.866 | +1.100 |
| diverse | 25 | 25 | +0.135 | +0.134 | -2.055 | +0.849 |
| diverse | 50 | 49 | +0.149 | +0.150 | +1.955 | +1.510 |
| diverse | 50 | 50 | +0.126 | +0.128 | +0.432 | +1.032 |

## Method

The Crystal trainer uses Qwen3.5-2B Q4_K_M and the frozen `tool-four` starting adapter. It replays the same two 50-update full-SFT schedules, with training seed 42, Adam learning rate 0.0001 and rank 8 on seven projections in full-attention blocks 11, 15, 19 and 23. The branches at updates 8, 23 and 48 restore the same LoRA factors and Adam counter for each order. Each branch clears `m` and `v` once, then executes two ordinary Adam updates. The driver checks that both cleared tensors are zero and validates optimizer counters and finite factors after both updates.

The replays matched 100 ordinary update records. The six common adapters and six ordinary forward endpoints matched byte for byte. The first pre-update loss matched in every treated order. The archived ordinary and single-moment controls use the same model, requests and schedules. Each new checkpoint was evaluated with 32 fixed prompts forming 16 pairs, a 4,096-token context limit, seed 101, temperature 0.8, top-k 40 and up to 128 generated tokens. Two local evaluation workers used the same native sampler in separate processes and scored separate checkpoint directories.

The independent reader regraded 24 checkpoints for each of the four Adam conditions from raw probabilities and answers and recomputed the step norms from the serialized FP32 factors. The second step is measured from the first saved update in its own branch. With both moments freshly zero, Adam forms the first numerator from the current gradient and the denominator from that gradient's square. With no weight decay, the first coordinate displacement is bounded by `lr * (1 - beta1) * sqrt((1 - beta2**t) / (1 - beta2)) / (1 - beta1**t)`, where `t` is the retained counter incremented for this update. All 12 first updates obeyed the bound with FP32 rounding allowance. The experiment uses fixed schedules, synthetic Boolean repair pairs and one seed for each of training and sampling.

## Reproduction

Use the populated native QLoRA worktree and the frozen archives. The existing run directory supports a cache recheck. For a fresh run, choose an unused directory for `run_root`.

```sh
run_root=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/both-moments-reset-v1
mkdir -p "$run_root"
if [ ! -x "$run_root/native-reset" ]; then
  crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_both_moments_reset.cr --release \
    -o "$run_root/native-reset" \
    --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
fi
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/both_moments_reset.py --output "$run_root"
```

The recorded run directory also holds the independent reader `verify_raw.py`. Invoke it with `python3 -B "$run_root/verify_raw.py" --output "$run_root"` from the worktree to regrade the raw runs and recompute the four sets of norms. For a fresh run keep the recorded reader's path and pass the new directory to `--output`.

The SHA-256 digests identify the artifacts used for this report.

| File | SHA-256 |
| --- | --- |
| `protocol.json` | `d18102dba0032d4390c335cdbd6e36500b7eeb64f210e9f103007325598e7f00` |
| `results.json` | `2241478b47edc2f97bd706a9ce097d79203c5c06a1e6789ded84206c07e18e1a` |
| `verify_raw.py` | `ea64de69a29fd7a229f08cb59d8042ab6b094a6edff49632ed766f086edf9ad8` |
| `verified-metrics.json` | `11447b520fb3d5dc01e34c9cd6dcaeb8ed85d9e3e79bd82c7f895cbab5aba39d` |
