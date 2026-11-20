# First-token logits for candidate selection

Reversing the answer instruction from `X or Y` to `Y or X` shifted all 72 matched first-token log-odds toward Y. Scoring the two lines across the four balanced presentations selected the same line under both instructions in all six variants and for all three adapters. Each adapter passed the full suite on four variants with either wording.

The probe uses Cursor, Slot, and Outbox from the [archived instruction-order experiment](PAIRED_BATCH_INSTRUCTION_ORDER.md). Each family supplies two variants. Each variant has four presentations covering both row orders and both X/Y label assignments. The same Qwen3.5-2B Q4_K_M model and three frozen adapters received all 48 prompts. The context limit was 4,096 tokens with 128 reserved for the response. The logit readout precedes temperature, top-k filtering, and sampling.

## Letter bias and line preference

For each presentation, let `d = logit(X) - logit(Y)`. The mean of `d` across the four presentations is the letter bias `b`. For line preference `s`, retain `d` when X identifies the canonical line and negate it when Y identifies that line, then average. The balanced design cancels additive letter and row-position biases. Interactions between the code and the presentation can still contribute to the score.

The canonical line is variant a's passing line and stays fixed for variant b. The predeclared criterion requires a positive then negative score for Cursor and Slot, and two positive scores for Outbox. A positive score favors the canonical line. An exact tie abstains.

| Adapter | Mean `b`, X or Y → Y or X | Shifts toward Y | Sampled full passes out of 24 | Two-letter argmax full passes out of 24 |
|---|---:|---:|---:|---:|
| `tool-four` | +0.1065 → −0.3824 | 24/24 | 19 → 14 | 16 → 16 |
| Real pairs | +0.1765 → −0.2986 | 24/24 | 17 → 14 | 16 → 14 |
| Shuffled pairs | +0.0546 → −0.3029 | 24/24 | 16 → 14 | 15 → 15 |

Each cell below reports `s(a), s(b)` before and after reversing the instruction.

| Adapter | Cursor, `left < right` | Slot, `index < limit` | Outbox, `pending_count > 0` |
|---|---|---|---|
| `tool-four` | +0.0839, +0.0538 → +0.0449, +0.0194 | +0.1341, +0.1257 → +0.1518, +0.1188 | +0.7628, +0.8383 → +0.7667, +0.8280 |
| Real pairs | +0.0904, +0.0703 → +0.0395, +0.0278 | +0.1600, +0.1503 → +0.1722, +0.1393 | +0.6802, +0.7660 → +0.6914, +0.7577 |
| Shuffled pairs | +0.0905, +0.0716 → +0.0426, +0.0295 | +0.1483, +0.1362 → +0.1587, +0.1233 | +0.6621, +0.7456 → +0.6837, +0.7457 |

All twelve Cursor and Slot contrasts lowered the canonical line's score in variant b, where the other line is required. All twelve stayed above zero. The decreases ranged from 0.0084 to 0.0355 logit units. Both Outbox variants kept the correct line in all six adapter and wording combinations.

## Crystal outcomes

Each adapter and instruction combination produced one decision per variant from four readouts. All six selections compiled, five passed targets, five passed regressions, and four passed both test groups. Cursor a, Slot a, and both Outbox variants passed. Cursor b failed the direction targets. Slot b failed the last allocated slot regression.

A fixed choice of variant a's passing line yields the same outcomes. The control shows that the 4/6 score can coexist with zero correct switches. This experiment measures six reused synthetic variants. A ranking improvement would need to survive both halves of a flip pair on fresh families. This probe uses four prefills per decision, and exhaustive verification can already cover the two candidates in these fixtures with two checks.

## Reproduction

Use the populated native QLoRA worktree containing the pinned X/Y runs and bridge object.

```sh
readout_dir=build/qlora/scale-2b/choice-first-token
mkdir -p "$readout_dir"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/choice_first_token.cr --release \
  -o "$readout_dir/readout" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/truth_training/paired_batch_choice_logits.py "$readout_dir/readout"
```

The default output is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/paired-batch-choice-logits-v1`. Use `--out` with an unused directory for another binary or run.

The native readout used exact one-token labels and reproduced all 144 archived seed-101 draws. Fresh processes with the request order reversed reproduced all 144 logit records exactly. Crystal rebuilt both lines for each variant, and the 12 candidate scores matched the prior controls. An independent calculation recovered the results directly from the raw logits, displayed code, compilation statuses, and test logs. Synthetic controls with known letter and position preferences checked the estimator and its abstention rule.

| Artifact | SHA-256 |
|---|---|
| Protocol | `720322df68beec24d013772a22605189abf5747d6884290a5c63eceb2f45a11f` |
| Prompt plan | `b4eda99bbd67184cc249048c443a66ddf5d5d5b3369baa27a3acc5a713f057a2` |
| Candidate checks | `8a7e67d92bf0ae6e6f631fbee65e8497ff4efb3b35cf910d6ed2361b5e5864a7` |
| Results | `b0fdbdb56900a6127de67f6543e6303d925fae7c59e23fc8de60f23989c8f040` |
