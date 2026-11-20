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

## Pair center ablation

The follow-up restarts from tool-four and replays the exact 50-update schedule from `real-choice-pairs-v1-attempt2`. The only objective change is adding `0.25 * ((d(a) + d(b)) / 2)^2` to the mean choice CE. Both differences `d` use `logit(X) - logit(Y)` regardless of which letter is correct. The derivative with respect to either difference is `0.25 * (d(a) + d(b)) / 2`. One extra forward obtains both differences before either backward pass. The trainer accumulates both gradients and updates Adam once per pair.

The holdout reuses all 48 pinned requests at 4K and sampling seed 101. The comparison uses a fresh readout of the archived paired CE adapter and a fresh Crystal replay of both candidate lines in every holdout variant. For a pair of canonical-line scores `s_a` and `s_b`, the center is `(s_a + s_b) / 2` and the separation is `(s_a - s_b) / 2`. A correct flip requires separation greater than the absolute center. Outbox must retain a positive score on both sides. The driver also measures both adapters on all 36 training presentations.

### Ablation results

The native run completed 50 Adam updates and supervised 100 first tokens in 672.3 seconds. Both Float32 factors changed in all 28 projections. The first attempt stopped after a Float32 rounding discrepancy in the reported penalty. The recorded run restarted from tool-four with the penalty value computed in Float64 and Float32 gradients. Recomputing the two CE terms from the fixed-orientation margins verified the objective on all 50 batches. The first CE terms matched the control exactly.

Across the 36 training presentations, RMS pair center fell from 0.685 to 0.120 and mean separation fell from 1.000 to 0.114. Both correct signs occurred in 20/36 presentations with paired CE and 17/36 with the added penalty. The corresponding first-presentation counts were 6/9 and 5/9.

All holdout scores below are paired CE → centered CE. Each test count applies to both adapters and both wordings.

| Variant | Required sign | `X or Y` | `Y or X` | Crystal examples passed |
|---|---:|---:|---:|---|
| Cursor a | + | +0.1845 → +0.0402 | +0.0925 → +0.0198 | 4/4 |
| Cursor b | − | +0.0989 → +0.0275 | +0.0273 → +0.0110 | 2/4. Both direction targets fail |
| Slot a | + | +0.2361 → +0.0743 | +0.1861 → +0.0709 | 6/6 |
| Slot b | − | +0.2009 → +0.0687 | +0.1193 → +0.0568 | 5/6. Last allocated slot regression fails |
| Outbox a | + | +2.1276 → +0.4260 | +2.1700 → +0.4276 | 4/4 |
| Outbox b | + | +2.4030 → +0.4621 | +2.5027 → +0.4583 | 4/4 |

The transfer criterion was unmet for both wordings. Both adapters compiled 6/6 selected lines and passed 5/6 target groups, 5/6 regression groups, and 4/6 full suites. Cursor and Slot each had a smaller center and separation, with both scores still positive. Outbox kept both full-suite passes. Single-presentation first-token draws produced 16/24 → 14/24 full passes under `X or Y` and 14/24 → 13/24 under `Y or X`. Restricted argmax full passes were 16/24 → 16/24 and 15/24 → 14/24. Neither flip family had a complete correct pair on any matched single presentation under either scoring rule.

The current paired CE adapter exactly reproduced its archived 48 holdout and 18 training-probe readouts. A fresh process reproduced the centered adapter's 48 holdout readouts exactly in reversed request order. An independent replay derived choices from the displayed candidate text, raw logits, Crystal exit statuses, and test logs. The focused QLoRA spec run passed 30 examples. These measurements cover candidate selection on three pinned synthetic holdout families at one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `2a183b65c62f8063738fa74ee11f0483ed666f18a95dea3434615aacffc6c2ce` |
| Native training audit | `657ab1c16ad8eb38987953a8015227edcddf7e9f6ab63c53a288a93015a18a78` |
| Centered adapter | `39dddfe8db8baf5950d37dc833bd7a6495b60493b6721d7934ed5a53c207c678` |
| Results | `8b2be0f9419c3c8111713e8363354787a58ae439919e5fb8f4da7539af718cae` |

### Reproduce the ablation

Build an updated trainer and use the exact archived `choice-first-token/readout` binary from the original run. The driver checks the original protocol, adapter, training rows, schedule, scoring fixtures and readout hashes before training.

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
crystal build eval/crystal_bugfixes/caller_transfer/real_pairs/train_choice_pairs.cr --release \
  -o "$bins/train_centered_replay" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_centered_replay" build/qlora/scale-2b/choice-first-token/readout \
  --out "$bins/centered-replay"
~~~

Use a fresh `--out` directory for each repetition. The recorded run uses `real-choice-pairs-centered-v1-attempt2` beside the previous run.
