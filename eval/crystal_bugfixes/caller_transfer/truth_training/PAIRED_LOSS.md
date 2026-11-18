# Full SFT with a paired first-token margin

The paired child starts from tool-four on the same Qwen3.5-2B Q4_K_M GGUF as the [full-SFT](README.md) and [first-token choice](FIRST_TOKEN_LOSS.md) children. It uses rank 8 on seven projections in blocks 11, 15, 19 and 23, Adam at 0.0001, and the same 50 anchor visits with training seed 42.

The 32 training rows form 16 exact pairs. The two user messages in each pair differ by the `!` in the caller's return statement and their first answer labels are opposite. Pairing is derived from the training JSONL. Each step still supervises all four target positions of its original anchor, including EOS. This preserves the full-SFT total of 200 target positions.

Let `m = logit(correct literal) - logit(other literal)` at the first answer position. Each Adam update uses this loss.

```text
CE_full(anchor) + 0.5 * max(0, 0.5 - min(m_anchor, m_mate))
```

The penalty reaches zero when both sides put the right literal ahead by at least 0.5 log-odds. The hinge gradient goes to the weaker side, with half to each on a tie. The mate shares the anchor's weights and adds only the first-position hinge gradient. The trainer reads the mate margin before the anchor pass and recomputes the mate with a tape when its gradient is active. It accumulates both contributions before the optimizer update. The auxiliary margin uses two rows of the frozen Q6_K head and the frozen output RMSNorm.

Evaluation reuses the same 16 held-out sign pairs, 4,096-token context, 128-token generation cap, seed 101, temperature 0.8 and top-k 40. For each pair, `t` is the literal required first by the inverted caller. The conditional shift holds `t` fixed and measures the effect of adding `!` on `log(P(t) / P(other))`. A first-token pair requires the correct higher-logit literal on both sides. A complete sampled pair requires both two-literal answers to parse and match the expected truth table. The prior arms reuse the pinned original outputs. The paired arm uses the original sampler and a separate native readout repeated in reverse prompt order.

## Results

| Adapter | Conditional P(t), direct → inverted | P(t) rose | Mean log-odds shift | First-token pairs | Correct sampled answers | Complete sampled pairs |
|---|---:|---:|---:|---:|---:|---:|
| Tool-four | 27.95% → 36.26% | 15/16 | +0.553 | 2/16 | 17/32 | 2/16 |
| Full SFT | 38.87% → 46.49% | 16/16 | +0.347 | 4/16 | 22/32 | 6/16 |
| First-token choice | 39.59% → 46.71% | 16/16 | +0.349 | 1/16 | 14/32 | 1/16 |
| Full SFT + paired margin | 40.29% → 47.43% | 16/16 | +0.302 | 2/16 | 21/32 | 5/16 |

Relative to full SFT, conditional P(false) rose in 31 of 32 prompts. When the inverted caller required `false`, its mean probability rose from 33.98% to 44.29% and its maximum reached 49.65%. The leader was `true` in all eight of those cases. On the eight inverted prompts requiring `true`, mean P(true) fell from 59.00% to 50.57%, with four of eight still favoring the correct literal.

The paired adapter corrected six of ten previously wrong sampled answers and introduced seven new errors. It solved five new complete sampled pairs and lost all six complete pairs from full SFT. All 32 responses parsed as two Boolean literals. The first-slot probability mass on the two literals was 97.59% across the full vocabulary. The 16 pairs come from four closely related templates and the sampled scores use one seed.

At the final checkpoint on the 16 training pairs, the mean hinge was 1.212 for tool-four, 0.715 for full SFT and 0.580 for paired SFT. Full SFT and paired SFT each had four training pairs with both first-token leaders correct. The 0.5 margin on both sides was reached by zero pairs in either child. The lower training hinge coincided with a smaller mean log-odds shift and fewer correct first-token pairs on holdout.

## Individual holdout pairs

The probability columns show conditional P(t) before and after adding `!`, followed by the log-odds shift in parentheses.

| Template | Public at 0 | Order | Inverted t | Full SFT | Paired SFT | Correct first-token pair | Complete sampled pair |
|---|---|---|---|---:|---:|---|---|
| Mosaic | false | 0, 1 | `true` | 57.99 → 65.06 (+0.300) | 54.79 → 55.42 (+0.026) | — | SFT |
| Mosaic | false | 1, 0 | `false` | 16.39 → 28.75 (+0.722) | 25.42 → 43.90 (+0.831) | — | Paired |
| Mosaic | true | 0, 1 | `false` | 23.10 → 30.01 (+0.356) | 32.65 → 42.06 (+0.404) | — | Paired |
| Mosaic | true | 1, 0 | `true` | 56.13 → 62.62 (+0.269) | 42.77 → 49.34 (+0.265) | — | SFT |
| Pine | false | 0, 1 | `true` | 40.69 → 52.04 (+0.458) | 40.27 → 46.57 (+0.257) | SFT | — |
| Pine | false | 1, 0 | `false` | 39.80 → 41.27 (+0.061) | 46.00 → 49.65 (+0.146) | — | SFT |
| Pine | true | 0, 1 | `false` | 34.16 → 39.85 (+0.244) | 40.97 → 46.59 (+0.228) | — | — |
| Pine | true | 1, 0 | `true` | 42.33 → 50.55 (+0.331) | 39.27 → 44.35 (+0.209) | SFT | — |
| Silver | false | 0, 1 | `true` | 48.30 → 60.88 (+0.510) | 48.35 → 56.81 (+0.340) | SFT, Paired | SFT |
| Silver | false | 1, 0 | `false` | 31.55 → 33.64 (+0.095) | 36.21 → 39.30 (+0.132) | — | — |
| Silver | true | 0, 1 | `false` | 31.93 → 35.48 (+0.159) | 38.73 → 41.05 (+0.097) | — | Paired |
| Silver | true | 1, 0 | `true` | 48.03 → 57.89 (+0.397) | 43.56 → 51.46 (+0.318) | SFT, Paired | — |
| Velvet | false | 0, 1 | `true` | 53.40 → 61.71 (+0.341) | 51.28 → 52.52 (+0.050) | — | SFT |
| Velvet | false | 1, 0 | `false` | 18.27 → 30.90 (+0.693) | 26.09 → 46.24 (+0.891) | — | Paired |
| Velvet | true | 0, 1 | `false` | 24.30 → 31.95 (+0.380) | 34.49 → 45.51 (+0.462) | — | Paired |
| Velvet | true | 1, 0 | `true` | 55.55 → 61.23 (+0.234) | 43.77 → 48.08 (+0.174) | — | SFT |

## Reproduction

Use the populated native QLoRA worktree and a fresh output directory.

```sh
trial=build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/paired-loss-replay
trainer="$PWD/build/qlora/scale-2b/train_paired_truth-replay"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/train_paired_truth.cr --release -o "$trainer" --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/fit_paired_truth.py "$trainer" --output "$trial"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/compare_paired_truth.py --output "$trial"
crystal spec spec/qwen35_qlora_*_spec.cr --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
```

The output contains input hashes, the training pair map, the update log, the checked adapter, both protocol files, native generations and first-token readouts, and the per-pair comparison.

The run recorded 50 matched optimizer steps in 535.7 seconds, with 200 full-SFT target positions and changes to all 56 factors. The hinge was active on all 50 steps, including 32 steps with a mate backward pass. The anchor SFT loss at step one matched the original full-SFT log exactly. All 28 native QLoRA specs passed, and separate scoring from the original response strings and literal logits reproduced the reported pair counts.

| Evidence from the recorded run | SHA-256 |
|---|---|
| Fit protocol | `a532f48ce2251c7741542b378217a7f276632da9bb8c0ec10264958a3a5b89cf` |
| Training audit | `c46ef5329c8191cd9df8c038a198c790d86dfded2605002cfa63a203c707a04b` |
| Comparison protocol | `8057a300a220d64a9fab06fc21f922c0c32bf3b06623ce24db8af8d515454562` |
| Results | `30ce4a0842e1cba937a1fad766bdebc25c1a60cb446b63e330445b3f896d6157` |
| Training readouts | `dab6d825795e0789c9153977655c0457b0f68dddc9528afbf2b29819025cbc5a` |
| Paired adapter | `6c03cbb6546638d830799718687b3f3db2e064c9f073a369053e33b49cd8424c` |
