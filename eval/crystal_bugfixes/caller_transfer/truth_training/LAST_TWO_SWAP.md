# Swapping the last two visits of each alternating prefix

Swapping the final two training visits shifted first-token log-odds toward the former last label on all 32 prompts in each of the six comparisons. The [whole-schedule reversal](REVERSED_ORDER.md) had shifted the mean toward its new last label in five of six comparisons.

Each arrow goes from the original `true` → `false` schedule to the prefix with its final two visits swapped.

| Training set | Step | Last target | `true` leads of 32 | Mean log-odds of `true` | Complete pairs of 16 |
|---|---:|---|---:|---:|---:|
| Original | 10 | `false` → `true` | 12 → 8 | −0.330 → −0.866 | 6 → 5 |
| Original | 25 | `true` → `false` | 22 → 24 | +0.132 → +0.233 | 5 → 4 |
| Original | 50 | `false` → `true` | 10 → 3 | −0.229 → −0.361 | 4 → 4 |
| Diverse | 10 | `false` → `true` | 2 → 1 | −0.495 → −0.678 | 4 → 4 |
| Diverse | 25 | `true` → `false` | 25 → 28 | +0.203 → +0.338 | 2 → 3 |
| Diverse | 50 | `false` → `true` | 20 → 17 | +0.105 → −0.021 | 3 → 4 |

The majority label held at all six checkpoints. At step 50 on the diverse dataset, the mean crossed zero and favored `false` while `true` still led on 17 of 32 prompts. The pair counts fell by one at steps 10 and 25 on the original dataset and rose by one at steps 25 and 50 on the diverse dataset. The best swapped checkpoint solved 5/16 pairs. The starting `tool-four` adapter solved 7/16.

## Method

The baseline is the [true-first alternating run](ALTERNATING_ORDER.md). For each dataset and target step, a fresh native QLoRA run starts from `tool-four` on the same Qwen3.5-2B Q4_K_M GGUF. It uses full-completion SFT, Adam at 0.0001, training seed 42 and rank 8 on seven projections in full-attention blocks 11, 15, 19 and 23. Each run stops at its target step.

Only the final two visits in each prefix exchange positions. Row multiplicities and first-label counts match exactly at all three steps. The counts are 5:5, 13:12 and 25:25 for `true:false`. The logged row, first target token and loss match exactly for the first 8, 23 or 48 updates, respectively. The six audited runs supervise 40, 100 or 200 tokens and update all 28 projections and 56 factors. The pinned native executable is the one used for the baseline.

Evaluation uses the same 16 held-out pairs in four families, with 32 prompts balanced on the expected first literal. A lead means that the first-token logit for `true` exceeds the logit for `false`. Mean log-odds average that logit difference. A complete pair requires both sampled two-literal answers to match their Crystal truth tables. Generation uses 4,096 context tokens, seed 101, temperature 0.8, top-k 40 and a limit of 128 new tokens. Independent scoring from the raw probabilities and answer strings reproduces the counts. These measurements use one training seed and one generation seed.

## Reproduction

In the populated native QLoRA worktree, run the coordinator with the frozen baseline artifacts and executable.

```sh
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/swap_last_two_checkpoints.py
```

A cached replay revalidates the six results against the logs and observations. Use `--output` with an unused directory for six fresh training runs. The protocol, adapters, logs and observations are stored under `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/semantic-diversity-v1/last-two-swap-v1`.

| Recorded artifact | SHA-256 |
|---|---|
| Protocol | `c577405b42bf3db04e8f39c77168af1bc17b363e8ab13143a030873db6da8b81` |
| Results | `406ebc4f52c403114f8168584759321a673492be07bc49d8008b1dd560427901` |
