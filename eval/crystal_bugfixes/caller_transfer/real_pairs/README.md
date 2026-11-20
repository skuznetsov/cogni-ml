# Native QLoRA on paired Crystal edits from Git history

Each pair has two Crystal contexts and the same two candidate lines. The passing candidate reverses across the contexts. The fixture records a unique excerpt at a fixed Git revision for each side, how the local test isolates the changed behavior, and the expected contract. The extracted fixtures pin only the code needed for this decision. Some pairs use an edited branch and an unchanged sibling. Their provenance is recorded per pair.

The nine pairs cover medium and long n-gram guards, contiguous and transposed indexing, LayerNorm and RMSNorm, Q and K widths, GEMM and attention policy, terminal and continuing cache routes, metadata and tensor readers, native and BOS-capable tokenizers, and control and normal token types. The index and normalization contracts come from a feature commit. Source excerpts are adapted into small programs. Two historical witnesses also ran directly on the original and fixed code. The 14-token n-gram case fails at the parent of `82e91f66` and passes at that commit. The M5 Max matrix GEMM admission check changes from fail to pass at `b846b5ba`.

The driver builds both lines in both contexts and runs target and regression checks for every build. Both lines must compile. Each side must have exactly one line that passes both groups, with opposite winners. Only the requirement, code excerpt, options and correct one-token label enter training. Patch revisions, test setup, tests and holdout prompts stay outside the trainer inputs.

The experiment starts at the frozen Qwen3.5-2B Q4_K_M checkpoint and its four-block tool-four adapter. It uses rank 8, seven projections on full-attention layers 11, 15, 19 and 23, and 50 native Adam updates at a learning rate of 0.0001. The adapter factors and training gradients use Float32. Each update shows both contexts with identical option order and X/Y labels. If side a's correct label is X, write `d(v) = logit(X) - logit(Y)`. The loss is `(softplus(-d(a)) + softplus(d(b))) / 2`. If a's correct label is Y, reverse the signs. Both backward passes use the same weights before Adam updates them. Each step supervises one X and one Y. The training seed is 42. The data scheduler rotates through both option orders and both label assignments for each pair.

Cursor, Slot and Outbox retain the 48 exact requests from the [prior logit readout](../truth_training/PAIRED_BATCH_CHOICE_LOGITS.md), at a 4,096-token cap with 128 tokens reserved. Seed 101, temperature 0.8 and top-k 40 are used for the unrestricted first-token draw. Logits are scored before sampling.

Within a variant, the score **s** averages the X-minus-Y logit difference aligned to the canonical line across the four presentations. For Cursor and Slot, a correct switch requires positive **s** for variant a and negative **s** for variant b. Outbox requires two positive scores. The transfer criterion is to gain at least one complete flip under both wordings while preserving both Outbox cases. The driver also records choices for each individual presentation and whether their Crystal targets and regressions pass. A sampled token outside X and Y is an invalid choice. This probe measures candidate selection at the first token.

## Recorded run

Crystal 1.22.3 compiled all 36 training candidate substitutions and verified a unique full-suite winner in each of the 18 contexts. The two candidate lines for all six held-out variants also compiled, with one full-suite winner per variant. The native trainer visited 72 distinct presentations in 50 Adam updates. It supervised 100 first tokens and changed both Float32 factors in all 28 projections. The trainer took 601.7 seconds on Apple M2 Max. At the first presentation of each training pair in the schedule, complete correct pairs went from 2/9 for tool-four to 6/9 for the new adapter, and correct individual signs went from 11/18 to 15/18.

The holdout transfer criterion was unmet. The new adapter selected the canonical line in every variant, as did tool-four. A positive **s** favors the canonical line. The table shows tool-four → new adapter and the test outcome for the chosen line under either wording.

| Variant | Required sign | `X or Y` | `Y or X` | Crystal outcome |
|---|---:|---:|---:|---|
| Cursor a | + | +0.0839 → +0.1845 | +0.0449 → +0.0925 | 4/4 pass |
| Cursor b | − | +0.0538 → +0.0989 | +0.0194 → +0.0273 | 2/4 pass. Both direction targets fail |
| Slot a | + | +0.1341 → +0.2361 | +0.1518 → +0.1861 | 6/6 pass |
| Slot b | − | +0.1257 → +0.2009 | +0.1188 → +0.1193 | 5/6 pass. Last allocated slot regression fails |
| Outbox a | + | +0.7628 → +2.1276 | +0.7667 → +2.1700 | 4/4 pass |
| Outbox b | + | +0.8383 → +2.4030 | +0.8280 → +2.5027 | 4/4 pass |

For each adapter and wording, all six selected lines compiled. Five variants passed their target group, five passed their regression group, and four passed both. The Cursor b selection used `left < right` where the caller needs `left > right`. The Slot b selection used `index < limit` and made the final slot unreadable. Both Outbox variants passed with `pending_count > 0`. Cursor and Slot each gained separation between their two contexts. In both contexts of both families, the canonical line nevertheless gained preference under each wording, including where the opposite line was required.

With one first-token draw per presentation at seed 101, full passes went from 19/24 to 16/24 under `X or Y` and stayed 14/24 under `Y or X`. The restricted two-letter argmax scored 16/24 → 16/24 and 16/24 → 15/24. Under neither wording did the child pick both correct lines in the same presentation of either flip pair. All draws were valid letter tokens. Reversing the instruction moved all 24 raw log-odds per adapter toward Y. The mean raw X-minus-Y bias went from +0.1065 to −0.3824 for tool-four and from +0.2352 to −0.6975 for the child as the wording changed.

The parent readout reproduced all 48 archived holdout records exactly. A summary-print error interrupted the first evaluation after that readout. The evaluation was rerun from the saved adapter with the print loop corrected, and the parent replay matched the initial readout including its training probes. The frozen driver and its recovery record are in the run directory. The per-variant margins and outcomes were independently recomputed from the displayed options, native logits, Crystal exit statuses and test logs. A fresh process with the request order reversed reproduced all 48 holdout records for the child. The focused Crystal spec run passed 42 examples across n-gram, Nomic policy and native QLoRA loss checks.

The evidence covers these three synthetic holdout families with one sampler seed. The comparison changes both the training dataset and objective relative to the starting adapter. It measures how this continuation ranks provided lines and leaves model size, quantization sensitivity, and free-form repair performance open.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `f2f30b03bab310b965f6d0d4051bb9e13ba9604556202cc039aad4f0274caa93` |
| Native training audit | `7572a642e28d420404883d88643ac09ee3121c74453eca03f5acc86882acee3f` |
| Child adapter | `08abfa16481f41b939a46f9cd6cde5631915f45d89d1f10d7a68d417e62cfea2` |
| Evaluation recovery | `0e6add189df70ddfe6b5bb01927dbd55edec81a583c3c6d28fdf2a1924974256` |
| Results | `b2531921256f6b8caf4d90c2b1fe3404be69051b79d05bdf9a4a9b7cf85f5fc0` |

## Reproduction

From the populated native QLoRA worktree, with the historical Git revisions and bridge object available, build the trainer and reuse the pinned native choice readout.

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
mkdir -p "$bins"
crystal build eval/crystal_bugfixes/caller_transfer/real_pairs/train_choice_pairs.cr --release \
  -o "$bins/train" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
crystal build eval/crystal_bugfixes/caller_transfer/truth_training/choice_first_token.cr --release \
  -o "$bins/readout" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run.py "$bins/train" "$bins/readout"
~~~

The default output directory is `build/qlora/scale-2b/own-vs-peer-v1/diagnostics/helper-truth-training-v1/real-choice-pairs-v1`. The recorded run uses the sibling directory `real-choice-pairs-v1-attempt2`. Use `--out` with a fresh directory for another run. The run keeps the frozen train rows, schedule, protocol, candidate test logs, adapter, native training audit and per-presentation holdout readouts.
