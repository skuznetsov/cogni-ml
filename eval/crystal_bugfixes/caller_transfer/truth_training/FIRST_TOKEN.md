# First-token response to caller negation

We measured the probability of the first answer token for tool-four and its frozen truth-table adapter on the 16 sign pairs in the [truth-training experiment](README.md). In every pair the two prompts differ by one exclamation mark. We fixed the subset of ten previous child errors from the archived responses before reading any logits.

For each pair let t be the first literal required by the inverted caller. The reported shift is P(t | inverted caller) minus P(t | direct caller). The target t stays fixed on both sides. Raw probabilities use softmax over all 248,320 vocabulary logits at temperature 1. Sampling probabilities reproduce the previous temperature of 0.8 and top-40 filtering. The tokenizer encodes each lowercase literal as a single token, ID 1802 for true and 3721 for false.

## Results

| Adapter | Cases | Raw P(t), direct → inverted | Increased | Sampling P(t), direct → inverted | Increased |
|---|---|---:|---:|---:|---:|
| tool-four | All pairs | 20.71% → 25.92% | 15/16 | 21.67% → 29.05% | 15/16 |
| tool-four | Prior child errors | 13.95% → 20.38% | 10/10 | 11.97% → 20.00% | 10/10 |
| Trained | All pairs | 38.36% → 45.85% | 16/16 | 36.74% → 45.69% | 16/16 |
| Trained | Prior child errors | 30.68% → 38.57% | 10/10 | 27.44% → 36.66% | 10/10 |

The trained adapter moved toward the correct first token in all ten previous errors. The raw increase across these errors ranged from 2.02 to 12.46 percentage points. After negation the correct token ranked first in all three errors that required true and second in all seven that required false. The historical one-token sampler still selected the wrong literal in all ten with seed 101. Tool-four moved in the same direction on all ten but never ranked the correct token first on them.

## Individual pairs

Each cell shows the full-vocabulary probability in percent for the direct caller followed by the inverted caller. The last column identifies the previously wrong sampled child responses. The two literals can sum to less than 100% because the vocabulary includes other tokens.

| Template | Public at 0 | Query order | Inverted target | Tool-four P(true) | Tool-four P(false) | Trained P(true) | Trained P(false) | Prior child error |
|---|---|---|---|---:|---:|---:|---:|---|
| Mosaic | false | 0, 1 | true | 45.92 → 38.45 | 26.18 → 24.28 | 57.28 → 64.23 | 41.50 → 34.49 |  |
| Mosaic | false | 1, 0 | false | 82.45 → 63.49 | 2.64 → 8.79 | 82.78 → 70.36 | 16.23 → 28.40 | yes |
| Mosaic | true | 0, 1 | false | 81.08 → 62.53 | 6.08 → 12.85 | 76.19 → 69.19 | 22.89 → 29.66 | yes |
| Mosaic | true | 1, 0 | true | 23.09 → 28.82 | 34.21 → 26.78 | 55.13 → 61.57 | 43.08 → 36.75 |  |
| Pine | false | 0, 1 | true | 15.90 → 20.96 | 73.90 → 67.57 | 40.11 → 51.28 | 58.46 → 47.25 | yes |
| Pine | false | 1, 0 | false | 37.55 → 33.28 | 40.24 → 40.68 | 59.24 → 57.82 | 39.16 → 40.63 |  |
| Pine | true | 0, 1 | false | 64.39 → 48.21 | 20.75 → 32.35 | 65.01 → 59.30 | 33.74 → 39.30 | yes |
| Pine | true | 1, 0 | true | 15.83 → 19.36 | 70.70 → 65.94 | 41.58 → 49.70 | 56.64 → 48.62 | yes |
| Silver | false | 0, 1 | true | 24.64 → 35.98 | 63.96 → 48.72 | 47.80 → 60.12 | 51.16 → 38.63 |  |
| Silver | false | 1, 0 | false | 49.68 → 38.81 | 23.61 → 28.71 | 67.45 → 65.33 | 31.10 → 33.12 | yes |
| Silver | true | 0, 1 | false | 66.05 → 53.90 | 22.11 → 30.12 | 67.36 → 63.69 | 31.59 → 35.02 | yes |
| Silver | true | 1, 0 | true | 22.22 → 28.31 | 65.21 → 55.02 | 47.36 → 57.00 | 51.25 → 41.47 | yes |
| Velvet | false | 0, 1 | true | 35.78 → 39.08 | 27.87 → 24.00 | 52.90 → 61.06 | 46.16 → 37.88 |  |
| Velvet | false | 1, 0 | false | 73.83 → 69.81 | 3.09 → 8.11 | 81.05 → 68.38 | 18.12 → 30.58 | yes |
| Velvet | true | 0, 1 | false | 83.30 → 72.47 | 7.28 → 14.21 | 75.13 → 67.39 | 24.12 → 31.64 | yes |
| Velvet | true | 1, 0 | true | 22.20 → 27.95 | 30.36 → 25.09 | 54.71 → 60.32 | 43.77 → 38.19 |  |

## Interpreting the shift

As an exploratory check, the mean probability mass on the two literal tokens across all 32 prompts was 76.77% for tool-four and 98.70% for the child. Conditioning on those two possible tokens gives a different view of the effect of the exclamation mark. The mean shift toward the target was +8.31 percentage points for tool-four and +7.62 for the child. The trained adapter more consistently concentrates its probability on the requested answer format. The larger change in full-vocabulary probability alone cannot establish improved reasoning about negation. The 16 pairs cross four closely related templates; each original generation used one seed.

## Reproduction

Use the populated native experiment worktree. The Crystal readout reuses the sampled evaluator’s chat encoder, QLoRA prefill, adapted tail, output projection and seeded sampler. The prefill reserves the original 128-token response budget in the 4K context. Each primary readout matched the first literal of its archived answer and a separate single-token replay using the pinned historical binary. A second readout in reversed request order reproduced all probabilities exactly.

```sh
out=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/first-token-v1
mkdir -p "$out"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/first_token.cr --release -o "$out/readout" --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/measure_first_token.py "$out/readout" --output "$out"
```

The output folder includes the sealed error IDs, per-token logits, vocabulary ranks, raw and sampling probabilities, one-token replays and both paired comparisons. An alternative output directory keeps a fresh build or run separate from the immutable evidence.

Run protocol SHA-256 `3e5a653ab9dedcb0c876aa66a29fd4e7bc67ddcb0fc2618cbf4a730a608f366c`. Results SHA-256 `998f968f3c0a01cc6bda8f75d2b1cb26a806b0839815cd5abbb0894d9ca26492`.
