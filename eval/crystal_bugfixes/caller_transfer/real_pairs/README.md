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

## Minimum correct margin ablation

Both arms start from tool-four and use the same nine Crystal pairs, 50 Adam updates and 100 target tokens. The control uses the centered objective above. The new arm adds `0.5 * max(0, 0.5 - min(m_a, m_b))`, where each `m` is the logit of the correct letter minus the logit of the other letter in its context. Below the floor, the hinge gradient goes to the weaker side. Tied sides split it equally. Both correct margins must reach 0.5 logits for the hinge to vanish.

The comparison replays the control on the 48 pinned holdout requests and 72 training views. The holdout settings remain 4,096 tokens and seed 101. Crystal recompiles and tests both lines in all six holdout variants. The transfer criterion still requires a correct flip for Cursor or Slot under both wordings and full-suite passes for both Outbox variants.

### Ablation results

The native run took 655.7 seconds and updated both Float32 factors in all 28 projections. Recomputing the CE, center and hinge terms from the logged margins verified the objective on all 50 batches. The first batch matched the centered control's logit differences and CE exactly. The hinge was active on 49 batches.

On the 36 training presentations, correct choices on both sides rose from 17/36 to 19/36. Both margins reached 0.5 logits in 1/36 → 2/36. The first scheduled presentation of each distinct pair stayed at 5/9 correct. Mean separation increased from 0.114 to 0.152, and RMS pair center increased from 0.120 to 0.159.

| Variant | Required sign | `X or Y` | `Y or X` | Crystal examples passed |
|---|---:|---:|---:|---|
| Cursor a | + | +0.04020 → +0.04371 | +0.01976 → +0.02443 | 4/4 |
| Cursor b | − | +0.02749 → +0.03001 | +0.01098 → +0.01199 | 2/4. Both direction targets fail |
| Slot a | + | +0.07428 → +0.08818 | +0.07089 → +0.08754 | 6/6 |
| Slot b | − | +0.06870 → +0.07846 | +0.05675 → +0.06728 | 5/6. Last allocated slot regression fails |
| Outbox a | + | +0.42604 → +0.50264 | +0.42761 → +0.52937 | 4/4 |
| Outbox b | + | +0.46215 → +0.55528 | +0.45834 → +0.57604 | 4/4 |

The table shows centered CE → centered CE with hinge. Each test count applies to both arms and both wordings. Cursor and Slot gained separation between contexts, while the preference for the canonical line increased in both contexts. Neither family met the flip criterion. Both Outbox cases passed all checks. Each arm and wording compiled 6/6 selected lines and passed 5/6 target groups, 5/6 regression groups and 4/6 full suites.

First-token draws passed 14/24 full suites for both arms under `X or Y`. Under `Y or X`, the hinge arm gained one correct Outbox draw, moving from 13/24 to 14/24. Restricted argmax stayed at 16/24 and 14/24, respectively. Neither flip family had a complete correct pair in any matched single presentation by either scoring rule.

The centered control reproduced all 120 archived records exactly. A fresh process reproduced all 48 holdout records for the hinge arm with the requests in reverse order. An independent scorer reconstructed the decisions from candidate text, raw logits, and Crystal exit statuses and test logs. The focused QLoRA specs passed 32 examples. This result covers the specified loss weights and margin floor on three synthetic holdout families at one seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `2e34ac6455a12f79f1a9423294fd38cc9eba68b0b6d6af6f3fe8a99ac59ce6f9` |
| Native training audit | `e217f5e07687ef3f9a6d96b9f862050f36fe81112d862f43c6371297e192f42e` |
| Center and hinge adapter | `9a5c166dffdd5b93b6bbc604e211835d8435cd892aef8fa69b4833244f59fc6d` |
| Results | `5b1c2649517d8a7234ce1d365fec047799a429ddb6c7149ab779d4ac43e14b05` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
crystal build eval/crystal_bugfixes/caller_transfer/real_pairs/train_choice_pairs.cr --release \
  -o "$bins/train_centered_hinge" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_centered_hinge" build/qlora/scale-2b/choice-first-token/readout --hinge
~~~

The driver pins the centered control from `real-choice-pairs-centered-v1-attempt2`. The default output directory is its sibling `real-choice-pairs-centered-hinge-v1`. Use `--out` with a fresh directory for another run.

## Directed log-odds difference ablation

Both arms start from tool-four and use the same nine pairs, 50 Adam updates and 100 target tokens. For each pair let `d_a` and `d_b` be the X-minus-Y logit differences in the two contexts, and let `t` be +1 when X is correct in the first context and −1 when Y is correct. Write `s = t * (d_a - d_b) / 2` and `b = (d_a + d_b) / 2`. Both arms use the same choice CE and `0.25 * b^2` center penalty. The minimum-margin control adds `0.5 * max(0, 0.5 - (s - abs(b)))`. The new arm adds `0.5 * max(0, 0.5 - s)`.

The normalized difference keeps both hinge values equal at a zero center. Its 0.5-logit floor is equivalent to a 1.0-logit floor on the full directed difference. An active hinge adds equal and opposite slopes to the two raw logit differences. Both sides choose correctly only when `s > abs(b)`, so reaching the directed floor still allows a wrong leader when the center is large.

The driver replays the minimum-margin control on the 48 pinned holdout requests and 72 training views. It rebuilds and tests both candidates for each of the six holdout variants. Evaluation stays at 4,096 tokens and seed 101. The transfer criterion requires a new complete flip for Cursor or Slot under both wordings and full-suite passes for both Outbox variants.

### Ablation results

The native training run took 675.4 seconds and updated both Float32 factors in all 28 projections. The first-batch CE and logit differences matched the control exactly. An independent calculation from the recorded logit differences reproduced the loss and the hinge slopes for all 50 updates. The directed hinge was active on 41/50 batches. In two of the nine inactive batches one side still ranked the wrong line first before the update.

On the 36 training presentations, mean separation rose from 0.152 to 0.243 and the number meeting the directed floor increased from 3/36 to 6/36. RMS pair center increased from 0.159 to 0.254. Both correct signs occurred in 19/36 → 18/36, and both correct margins reached 0.5 logits in 2/36 → 3/36. The first scheduled presentations scored 5/9 → 4/9.

| Variant | Required sign | `X or Y` | `Y or X` | Crystal examples passed |
|---|---:|---:|---:|---|
| Cursor a | + | +0.04371 → +0.07909 | +0.02443 → +0.04013 | 4/4 |
| Cursor b | − | +0.03001 → +0.05180 | +0.01199 → +0.01943 | 2/4. Both direction targets fail |
| Slot a | + | +0.08818 → +0.11918 | +0.08754 → +0.11140 | 6/6 |
| Slot b | − | +0.07846 → +0.10893 | +0.06728 → +0.08502 | 5/6. Last allocated slot regression fails |
| Outbox a | + | +0.50264 → +0.77815 | +0.52937 → +0.74070 | 4/4 |
| Outbox b | + | +0.55528 → +0.84252 | +0.57604 → +0.80116 | 4/4 |

The table shows minimum-margin hinge → directed-difference hinge. The test counts apply to both arms and wordings. Cursor and Slot gained separation, and their centers increased by more than their separation did. Their variant b scores moved further above zero. Outbox kept both full-suite passes. Each arm and wording compiled 6/6 selected lines and passed 5/6 target groups, 5/6 regression groups and 4/6 full suites. The transfer criterion was unmet.

First-token draws passed 14/24 → 16/24 under `X or Y` and 14/24 → 14/24 under `Y or X`. The `X or Y` draws gained one pass on each Outbox variant. One Slot a draw gained a pass and one Slot b draw lost a pass. Restricted argmax passed 16/24 → 15/24 and 14/24 → 14/24, respectively. Neither flip family achieved both correct choices in any matched single presentation by either scoring rule.

The minimum-margin control reproduced all 120 archived records exactly. A fresh process reproduced all 48 directed hinge holdout records with the requests in reverse order. An independent scorer reconstructed the holdout from the displayed code, raw logits, Crystal exit statuses and test logs. The focused QLoRA specs passed 34 examples. The measurements cover ranking the supplied lines on three synthetic holdout families at one seed with the specified loss weights and floor.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `79d688b3e4ff52f989e6256394a5804e879f4dad6cadeddeae8c393eff8ce89e` |
| Native training audit | `51a6daf21393e3371b4a47107c664466db96eb4bc38c0f0bc35fb3f737989b66` |
| Directed hinge adapter | `93726831eca4c0b05157d5f08a5b4f0182e4a52c43b44a5db6ab8898dcfc7df9` |
| Results | `0aba166d10b79deb6d56f72255706e39c74d37641dc5222c2b53ef506d326764` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
crystal build eval/crystal_bugfixes/caller_transfer/real_pairs/train_choice_pairs.cr --release \
  -o "$bins/train_directed_hinge" \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout --directed-hinge
~~~

The control comes from `real-choice-pairs-centered-hinge-v1`. The default output directory is its sibling `real-choice-pairs-directed-hinge-v1`. Use `--out` with a fresh directory for another run.

## Higher center weight with directed hinge

The follow-up raises the coefficient of `b^2` from 0.25 to 1.0 and keeps the directed hinge `0.5 * max(0, 0.5 - s)`. Both arms start from tool-four on the same Qwen3.5-2B Q4_K_M model and reuse the same native trainer, nine pairs, 50 Adam updates and 100 target tokens. The driver pins the control artifacts, rebuilds and tests both candidate lines for each holdout variant, and replays the same 48 requests at 4,096 tokens and seed 101.

For each wording, let `q_a` and `q_b` be the canonical-line logit margins averaged over all four balanced presentations of their respective variants. The table reports `s = (q_a - q_b) / 2` and `|b| = abs((q_a + q_b) / 2)`. Cursor and Slot require `q_a > 0 > q_b`, equivalently `s > |b|`. Outbox requires both margins to stay positive.

### Ablation results

The native run took 694.4 seconds and updated both Float32 factors in all 28 projections. The first batch reproduced the control's margins, CE and hinge exactly, with four times the center penalty and scale. Recomputing from the logged margins verified all 50 losses and hinge slopes. The hinge was active on 41/50 → 49/50 batches.

On the 36 training presentations, RMS center fell from 0.254 to 0.072 and mean separation fell from 0.243 to 0.045. Both signs were correct in 18/36 → 16/36 presentations. The number reaching the directed floor fell from 6/36 to 0/36. The first presentations of the nine pairs scored 4/9 in both arms.

| Family | Wording | Separation, 0.25 → 1.0 | Absolute center, 0.25 → 1.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.01364 → +0.00280 | 0.06545 → 0.01398 |
| Cursor | `Y or X` | +0.01035 → +0.00156 | 0.02978 → 0.00496 |
| Slot | `X or Y` | +0.00513 → +0.00106 | 0.11406 → 0.04209 |
| Slot | `Y or X` | +0.01319 → +0.00415 | 0.09821 → 0.03857 |
| Outbox | `X or Y` | −0.03219 → −0.00668 | 0.81034 → 0.21053 |
| Outbox | `Y or X` | −0.03023 → −0.00508 | 0.77093 → 0.21735 |

The flip slack `s - |b|` improved for both flip families under both wordings and stayed negative. Cursor came closest at −0.00340 under `Y or X`. Both Outbox variants retained positive margins. Each arm and wording compiled 6/6 selected lines and passed 5/6 target groups, 5/6 regression groups and 4/6 full suites.

Cursor a passed 4/4. Cursor b scored 2/4 with `left < right` and failed both direction targets. Slot a passed 6/6. Slot b scored 5/6 with `index < limit` and failed the last allocated slot regression. Each Outbox variant passed 4/4 with `pending_count > 0`.

Single-prompt first-token draws passed 16/24 → 12/24 under `X or Y` and 14/24 → 12/24 under `Y or X`. The change in passes came entirely from selecting the wrong Outbox line more often. All draws were valid letters. Restricted argmax passed 15/24 → 16/24 and 14/24 → 14/24, respectively. Both flip families scored 0/4 paired passes on matched single presentations with either scoring rule and either wording.

The control reproduced all 120 archived records. A fresh native process reproduced all 48 new holdout records with the requests in reverse order. An independent scorer reconstructed the selected lines and outcomes from the prompts, native logits, Crystal exit statuses and test logs. The focused QLoRA specs passed 34 examples. The result covers the three pinned holdout families and one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `c6064a99a5792cc3377c5f54ac2e31c96810865726876bb4efc3d6284de14135` |
| Native training audit | `120af91e1164d1ddd93fda9641e5476b1a14ba3cdfbafb60e6851646bf61bb4c` |
| Higher center weight adapter | `0145b1c52a657ebb4a15f3e96b14292530c79b664c902521247106ef9ca8d125` |
| Results | `aef557d5dbbba1a67188a49d1106fa1e61f20db88fe9b82dc08efe0d2ce795ae` |

### Reproduce the ablation

Reuse the exact archived directed hinge trainer and readout binaries.

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-center-1
~~~

The default output directory is `real-choice-pairs-directed-hinge-center-1-v1` beside the `real-choice-pairs-directed-hinge-v1` control. Use `--out` with a fresh directory for another run.

## Higher directed hinge weight at center weight one

This ablation raises the multiplier of `max(0, 0.5 - s)` from 0.5 to 2.0. Both arms use `1.0 * b^2` and restart from tool-four with the same Qwen3.5-2B Q4_K_M model, nine pairs, 50 Adam updates and 100 target tokens. The driver pins the control and reuses the same native binaries. It compiles and tests both candidate lines on all six holdout variants and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101.

### Ablation results

The native run took 753.1 seconds and updated both Float32 factors in all 28 projections. The first batch reproduced the control's margins, CE and center term exactly. The hinge contribution and its slopes were four times larger. Recomputing from the logged margins verified all 50 losses and hinge slopes. The hinge was active on 49/50 → 44/50 batches.

On the 36 training presentations, mean separation rose from 0.045 to 0.216 and RMS center rose from 0.072 to 0.233. The number reaching the directed floor rose from 0/36 to 5/36. Both correct signs stayed at 16/36 overall and 4/9 on the first presentations of the nine pairs.

| Family | Wording | Separation, 0.5 → 2.0 | Absolute center, 0.5 → 2.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00280 → +0.01308 | 0.01398 → 0.06079 |
| Cursor | `Y or X` | +0.00156 → +0.00898 | 0.00496 → 0.02881 |
| Slot | `X or Y` | +0.00106 → +0.00463 | 0.04209 → 0.10800 |
| Slot | `Y or X` | +0.00415 → +0.01212 | 0.03857 → 0.09408 |
| Outbox | `X or Y` | −0.00668 → −0.02605 | 0.21053 → 0.73591 |
| Outbox | `Y or X` | −0.00508 → −0.02484 | 0.21735 → 0.69936 |

The flip slack `s - |b|` fell under both wordings for both flip families. The closest case was still Cursor under `Y or X`, where slack moved from −0.00340 to −0.01983. Both Outbox margins stayed positive. In each arm and wording, 6/6 selected lines compiled, 5/6 target groups passed, 5/6 regression groups passed and 4/6 full suites passed.

Cursor a passed 4/4. Cursor b scored 2/4 with `left < right` and failed both direction targets. Slot a passed 6/6. Slot b scored 5/6 with `index < limit` and failed the last allocated slot regression. Both Outbox variants passed 4/4 with `pending_count > 0`.

Single-prompt first-token draws passed 12/24 → 16/24 under `X or Y` and 12/24 → 14/24 under `Y or X`. All net gains came from Outbox, which rose from 4/8 to 8/8 and from 4/8 to 6/8, respectively. Restricted argmax held at 16/24 and 14/24. Cursor and Slot both scored 0/4 paired passes on matched single presentations under either scoring rule and either wording.

The control reproduced all 120 archived records. A fresh native process reproduced all 48 new holdout records with the requests in reverse order. An independent scorer reconstructed the choices and outcomes from the displayed candidates, raw logits, Crystal exit statuses and test logs. The focused QLoRA specs passed 34 examples. The result covers the three pinned holdout families and one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `ac9bcca88cd0419225191a72f71f8259b04dfcdde6fb527af1b4049f208f7682` |
| Native training audit | `a19dd3c07fc313b694478130e35df3e362288b54b49013526f42801c2da8e012` |
| Higher hinge weight adapter | `89ec9cde82a502823379a41bcb20aba6e8391e33a6dffee36d5596b80108cead` |
| Results | `d4de912b7e7129d7a4b0770b016111e0d9bdd8e75bb405312dbd6cad64fd7d28` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-2-center-1
~~~

The default output directory is `real-choice-pairs-directed-hinge-2-center-1-v1` beside the `real-choice-pairs-directed-hinge-center-1-v1` control. Use `--out` with a fresh directory for another run.

## Center weight four with directed hinge weight two

This ablation raises the coefficient of `b^2` from 1.0 to 4.0. Both arms keep `2.0 * max(0, 0.5 - s)` and restart from tool-four on the same Qwen3.5-2B Q4_K_M model. The native binaries, nine pairs, 50 Adam updates, and 100 target tokens are pinned. The driver compiles and tests both candidate lines in each holdout variant and evaluates the same 48 requests at 4,096 tokens and sampler seed 101.

### Ablation results

The native run took 671.7 seconds and updated both Float32 factors in all 28 projections. On the first batch, CE, logit differences, and the hinge matched the control. The center contribution rose from 0.10215 to 0.40859. Recomputing from the logged margins reproduced the loss and hinge slopes for all 50 updates. The hinge was active on 44/50 → 49/50 batches.

On the 36 training presentations, RMS center fell from 0.233 to 0.051 and mean separation fell from 0.216 to 0.021. Both signs were correct on 16/36 → 12/36 presentations, including 4/9 → 3/9 first presentations. The number meeting the directed floor fell from 5/36 to 0/36.

| Family | Wording | Separation, 1.0 → 4.0 | Absolute center, 1.0 → 4.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.01308 → +0.00072 | 0.06079 → 0.00516 |
| Cursor | `Y or X` | +0.00898 → +0.00028 | 0.02881 → 0.00067 |
| Slot | `X or Y` | +0.00463 → +0.00095 | 0.10800 → 0.02791 |
| Slot | `Y or X` | +0.01212 → +0.00107 | 0.09408 → 0.02547 |
| Outbox | `X or Y` | −0.02605 → −0.00346 | 0.73591 → 0.11315 |
| Outbox | `Y or X` | −0.02484 → −0.00170 | 0.69936 → 0.12274 |

Cursor and Slot moved closer to the flip condition `s > |b|` under both wordings. The closest case was Cursor under `Y or X`, where `s - |b|` rose from −0.01983 to −0.00038. Both still led with the same line across contexts. Outbox retained both positive margins. For each arm and wording, 6/6 selected lines compiled, 5/6 target groups passed, 5/6 regression groups passed, and 4/6 full suites passed.

| Family | Variant a | Variant b |
|---|---|---|
| Cursor | 4/4 | 2/4. `left < right` fails both cursor direction targets. |
| Slot | 6/6 | 5/6. `index < limit` fails the last allocated slot regression. |
| Outbox | 4/4 | 4/4. Both variants selected `pending_count > 0`. |

Single-prompt first-token draws passed 16/24 → 12/24 under `X or Y` and 14/24 → 12/24 under `Y or X`. All net losses came from Outbox, which went from 8/8 to 4/8 and from 6/8 to 4/8, respectively. Restricted argmax passed 16/24 → 15/24 and 14/24 → 14/24. Cursor and Slot each scored 0/4 paired passes on matched single presentations with either scoring rule and either wording.

The control reproduced all 120 archived records. A fresh native process reproduced all 48 new holdout records in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed candidates, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed 34 examples. The comparison covers ranking the supplied lines on three holdout families with one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `cdd0692f0585875944cf2dbc7c3d043a3f43afa6390c2cb3c01c14f7c07cc9d8` |
| Native training audit | `0a84f56c3ce443d454073f6320d92a72c453f02175c4d3ea0575a34a49183cbd` |
| Center weight four adapter | `e4bc33c9472272b6ccd887c45782455b90cde8740d4dbd9d0ab3c9b7a93e6ca7` |
| Results | `c224df3a630b0cbb80edcc182a40b110e7d11507987edb7f97b6b0c93140105d` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-2-center-4
~~~

The default output directory is `real-choice-pairs-directed-hinge-2-center-4-v1` beside the `real-choice-pairs-directed-hinge-2-center-1-v1` control. Use `--out` with a fresh directory for another run.

## Directed hinge weight eight at center weight four

This ablation raises the multiplier of `max(0, 0.5 - s)` from 2.0 to 8.0. Both arms use `4.0 * b^2` and restart from tool-four with the same Qwen3.5-2B Q4_K_M model, nine pairs, 50 Adam updates, and 100 target tokens. The driver pins the control and reuses the same native binaries. It rebuilds and tests both candidate lines in each holdout variant and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101.

### Ablation results

The native run took 755.2 seconds and updated both Float32 factors in all 28 projections. On the first batch, CE, logit differences, and the center penalty matched the control. The hinge contribution rose from 1.20592 to 4.82367 and its slopes increased fourfold. Recomputing from the logged margins reproduced all 50 losses and hinge slopes. The hinge was active on 49/50 → 44/50 batches.

On the 36 training presentations, mean separation rose from 0.021 to 0.181 and RMS center rose from 0.051 to 0.201. Both signs were correct on 12/36 → 17/36 presentations, including 3/9 → 5/9 first presentations. The number meeting the directed floor rose from 0/36 to 5/36.

| Family | Wording | Separation, 2.0 → 8.0 | Absolute center, 2.0 → 8.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00072 → +0.00996 | 0.00516 → 0.05232 |
| Cursor | `Y or X` | +0.00028 → +0.00692 | 0.00067 → 0.02353 |
| Slot | `X or Y` | +0.00095 → +0.00376 | 0.02791 → 0.09716 |
| Slot | `Y or X` | +0.00107 → +0.01119 | 0.02547 → 0.08441 |
| Outbox | `X or Y` | −0.00346 → −0.02225 | 0.11315 → 0.63888 |
| Outbox | `Y or X` | −0.00170 → −0.02062 | 0.12274 → 0.60610 |

The flip slack `s - |b|` fell for Cursor and Slot under both wordings. The closest case was Cursor under `Y or X`, where slack moved from −0.00038 to −0.01661. Both flip families kept the same leading line across contexts. Outbox ranked the common correct line first in both variants with larger margins. For each arm and wording, 6/6 selected lines compiled, 5/6 target groups passed, 5/6 regression groups passed, and 4/6 full suites passed.

| Family | Variant a | Variant b |
|---|---|---|
| Cursor | 4/4 | 2/4. `left < right` fails both cursor direction targets. |
| Slot | 6/6 | 5/6. `index < limit` fails the last allocated slot regression. |
| Outbox | 4/4 | 4/4. Both variants selected `pending_count > 0`. |

Single-prompt first-token draws passed 12/24 → 16/24 under `X or Y` and 12/24 → 14/24 under `Y or X`. All net gains came from Outbox, which rose from 4/8 to 8/8 and from 4/8 to 6/8, respectively. Restricted argmax passed 15/24 → 16/24 and 14/24 → 14/24. Cursor and Slot each scored 0/4 paired passes on matched single presentations with either scoring rule and either wording.

The control reproduced all 120 archived records. A fresh native process reproduced all 48 new holdout records in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed candidates, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed 34 examples. The comparison covers ranking the supplied lines on three holdout families with one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `4c145e841f6715224c75d9cd69feec0136f7281b6d403b9840cb080368efa98a` |
| Native training audit | `a0c34d5fdc6f176de92e99761ed2523b5bb97e0d4d96a0fe20bf37610cd9ed32` |
| Hinge weight eight adapter | `49aff7288b5ceb4899a6b9512753b1c04323560729b0057fcc3fd38b5240ccfa` |
| Results | `3ade8941cb8ca0902e0d5a7eb1c67439e3f5cd8b4f83f529d4e4a8967316fbd2` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-8-center-4
~~~

The default output directory is `real-choice-pairs-directed-hinge-8-center-4-v1` beside the `real-choice-pairs-directed-hinge-2-center-4-v1` control. Use `--out` with a fresh directory for another run.

## Center weight sixteen with directed hinge weight eight

This ablation raises the coefficient of `b^2` from 4.0 to 16.0. Both arms keep `8.0 * max(0, 0.5 - s)` and restart from tool-four on the same Qwen3.5-2B Q4_K_M model. The native binaries, nine pairs, 50 Adam updates, and 100 target tokens are pinned. The driver compiles and tests both candidate lines in each holdout variant and evaluates the same 48 requests at 4,096 tokens and sampler seed 101.

### Ablation results

The native run took 746.5 seconds and updated both Float32 factors in all 28 projections. On the first batch, CE, logit differences, and the hinge matched the control. The center contribution rose from 0.40859 to 1.63436. Recomputing from the logged margins reproduced the loss and hinge slopes for all 50 updates. The hinge was active on 44/50 → 49/50 batches.

On the 36 training presentations, RMS center fell from 0.201 to 0.049 and mean separation fell from 0.181 to 0.017. Both signs were correct on 17/36 → 10/36 presentations, including 5/9 → 2/9 first presentations. The number meeting the directed floor fell from 5/36 to 0/36.

| Family | Wording | Separation, 4.0 → 16.0 | Absolute center, 4.0 → 16.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00996 → +0.00089 | 0.05232 → 0.00402 |
| Cursor | `Y or X` | +0.00692 → +0.00023 | 0.02353 → 0.00073 |
| Slot | `X or Y` | +0.00376 → +0.00065 | 0.09716 → 0.02457 |
| Slot | `Y or X` | +0.01119 → +0.00259 | 0.08441 → 0.02214 |
| Outbox | `X or Y` | −0.02225 → −0.00287 | 0.63888 → 0.09428 |
| Outbox | `Y or X` | −0.02062 → −0.00063 | 0.60610 → 0.10257 |

The flip slack `s - |b|` improved for Cursor and Slot under both wordings and stayed negative. The closest case was Cursor under `Y or X`, where slack moved from −0.01661 to −0.00050. Both of its margins changed sign. The leading line changed from `left < right` in both contexts to `left > right` in both, moving the target failure from variant b to variant a. Outbox kept the correct common line first in both contexts with smaller margins.

| Family and wording | Variant a, 4.0 → 16.0 | Variant b, 4.0 → 16.0 |
|---|---|---|
| Cursor, `X or Y` | 4/4 → 4/4 | 2/4 → 2/4 |
| Cursor, `Y or X` | 4/4 → 2/4 | 2/4 → 4/4 |
| Slot, both wordings | 6/6 → 6/6 | 5/6 → 5/6 |
| Outbox, both wordings | 4/4 → 4/4 | 4/4 → 4/4 |

Both Cursor direction targets failed in the 2/4 outcomes, and its regression examples passed. Slot b selected `index < limit` and failed the last allocated slot regression. Both Outbox variants selected `pending_count > 0` and passed all examples. Each arm and wording compiled 6/6 selected lines and passed 5/6 target groups, 5/6 regression groups, and 4/6 full suites.

Single-prompt first-token draws passed 16/24 → 12/24 under `X or Y` and 14/24 → 12/24 under `Y or X`. All net losses came from Outbox, which went from 8/8 to 4/8 and from 6/8 to 4/8, respectively. Restricted argmax passed 16/24 → 14/24 and 14/24 → 14/24. Cursor and Slot each scored 0/4 paired passes on matched single presentations with either scoring rule and either wording.

The control reproduced all 120 archived records. A fresh native process reproduced all 48 new holdout records in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed candidates, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed 34 examples. The comparison covers ranking the supplied lines on three holdout families with one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `0ccb2c27d73e90e33c250679f20099274146d251daeb254e79cfb0e545b5b00f` |
| Native training audit | `936e6c9c9238e3a40d577d28d89a2ff475e034d40f321a492e76abd57e0540dd` |
| Center weight sixteen adapter | `4e5ec7767b9499d712dea0893885025573258b341b3d50cfca9789ac1315d95b` |
| Results | `8f30fe426ca9c1149d4a06dad824f45f1391023ba91d9b2d43bc0826d3ef8b5c` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-8-center-16
~~~

The default output directory is `real-choice-pairs-directed-hinge-8-center-16-v1` beside the `real-choice-pairs-directed-hinge-8-center-4-v1` control. Use `--out` with a fresh directory for another run.

## Directed hinge weight thirty-two at center weight sixteen

This ablation raises the multiplier of `max(0, 0.5 - s)` from 8.0 to 32.0. Both arms use `16.0 * b^2` and restart from tool-four with the same Qwen3.5-2B Q4_K_M model, nine pairs, 50 Adam updates, and 100 target tokens. The driver pins the control and reuses the same native binaries. It rebuilds and tests both candidate lines in each holdout variant and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101.

### Ablation results

The native run took 795.6 seconds and updated both Float32 factors in all 28 projections. On the first batch, CE, logit differences, and the center penalty matched the control. The hinge contribution rose from 4.82367 to 19.29468 and its slopes increased fourfold. Recomputing from the logged margins reproduced all 50 losses and hinge slopes. The hinge was active on 49/50 → 44/50 batches.

On the 36 training presentations, mean separation rose from 0.017 to 0.170 and RMS center rose from 0.049 to 0.191. Both signs were correct on 10/36 → 17/36 presentations, including 2/9 → 5/9 first presentations. The number meeting the directed floor rose from 0/36 to 4/36.

| Family | Wording | Separation, 8.0 → 32.0 | Absolute center, 8.0 → 32.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00089 → +0.00985 | 0.00402 → 0.04978 |
| Cursor | `Y or X` | +0.00023 → +0.00689 | 0.00073 → 0.02365 |
| Slot | `X or Y` | +0.00065 → +0.00356 | 0.02457 → 0.09412 |
| Slot | `Y or X` | +0.00259 → +0.01057 | 0.02214 → 0.08181 |
| Outbox | `X or Y` | −0.00287 → −0.02136 | 0.09428 → 0.61025 |
| Outbox | `Y or X` | −0.00063 → −0.01922 | 0.10257 → 0.57832 |

The flip slack `s - |b|` fell for Cursor and Slot under both wordings. The closest case was Cursor under `Y or X`, where slack moved from −0.00050 to −0.01676. Its leading line changed from `left > right` in both contexts to `left < right` in both, moving the target failure from variant a to variant b. Outbox ranked the common correct line first in both variants with larger margins.

| Family and wording | Variant a, 8.0 → 32.0 | Variant b, 8.0 → 32.0 |
|---|---|---|
| Cursor, `X or Y` | 4/4 → 4/4 | 2/4 → 2/4 |
| Cursor, `Y or X` | 2/4 → 4/4 | 4/4 → 2/4 |
| Slot, both wordings | 6/6 → 6/6 | 5/6 → 5/6 |
| Outbox, both wordings | 4/4 → 4/4 | 4/4 → 4/4 |

Both Cursor direction targets failed in the 2/4 outcomes, and its regression examples passed. Slot b selected `index < limit` and failed the last allocated slot regression. Both Outbox variants selected `pending_count > 0` and passed all examples. Each arm and wording compiled 6/6 selected lines and passed 5/6 target groups, 5/6 regression groups, and 4/6 full suites.

Single-prompt first-token draws passed 12/24 → 16/24 under `X or Y` and 12/24 → 14/24 under `Y or X`. All net gains came from Outbox, which rose from 4/8 to 8/8 and from 4/8 to 6/8, respectively. Restricted argmax passed 14/24 → 16/24 and 14/24 → 14/24. Cursor and Slot each scored 0/4 paired passes on matched single presentations with either scoring rule and either wording.

The control reproduced all 120 archived records. A fresh native process reproduced all 48 new holdout records in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed candidates, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed 34 examples. The comparison covers ranking the supplied lines on three holdout families with one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `833caf584cf6eef2cb816942ad089a13eb7efec95cea4c8b96ddfd2a38f728c8` |
| Native training audit | `5d3b708f1c30eebfdd48fc527168092935f3f4a89af7c4be1e8f0791cbd45f53` |
| Hinge weight thirty-two adapter | `1ecc230a08910a4a105493146ebd394aaae4594bd30ea71d2c67e6fcfed67cb2` |
| Results | `1f16fa9c182b444fefd2d0c63c7e068f772565d881b0dc0eaff58f274b88f343` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-32-center-16
~~~

The default output directory is `real-choice-pairs-directed-hinge-32-center-16-v1` beside the `real-choice-pairs-directed-hinge-8-center-16-v1` control. Use `--out` with a fresh directory for another run.

## Center weight sixty-four with directed hinge weight thirty-two

This ablation raises the coefficient of `b^2` from 16.0 to 64.0. Both arms keep `32.0 * max(0, 0.5 - s)` and restart from tool-four on the same Qwen3.5-2B Q4_K_M model. The native binaries, nine pairs, 50 Adam updates, and 100 target tokens are pinned. The driver compiles and tests both candidate lines in each holdout variant and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101.

### Ablation results

The native run took 752.8 seconds and updated both Float32 factors in all 28 projections. With matching first-batch logit differences, CE, and hinge, the center contribution rose from 1.63436 to 6.53743. Recomputing the 50 losses and slopes from the logged margins reproduced the native audit. The hinge was active on 44/50 → 49/50 batches.

On the 36 training presentations of the saved adapters, RMS center fell from 0.191 to 0.049 and mean separation fell from 0.170 to 0.016. Both signs were correct on 17/36 → 9/36 presentations, including 5/9 → 2/9 first presentations. The number meeting the directed floor fell from 4/36 to 0/36.

| Family | Wording | Separation, 16.0 → 64.0 | Absolute center, 16.0 → 64.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00985 → +0.00070 | 0.04978 → 0.00276 |
| Cursor | `Y or X` | +0.00689 → −0.00113 | 0.02365 → 0.00118 |
| Slot | `X or Y` | +0.00356 → +0.00085 | 0.09412 → 0.02427 |
| Slot | `Y or X` | +0.01057 → +0.00163 | 0.08181 → 0.02192 |
| Outbox | `X or Y` | −0.02136 → −0.00234 | 0.61025 → 0.08968 |
| Outbox | `Y or X` | −0.01922 → −0.00059 | 0.57832 → 0.09849 |

Both `|b|` and `|s|` fell in all six comparisons. The flip slack `s - |b|` improved in all four Cursor and Slot comparisons and ended below zero in each. At center weight 64.0, it was −0.00206 and −0.00231 for Cursor, and −0.02341 and −0.02029 for Slot. Under `Y or X`, Cursor switched from `left < right` in both variants to `left > right` in both. Outbox ranked `pending_count > 0` first in both variants. Its smallest correct margin fell from +0.58889 to +0.08734 under `X or Y`, and from +0.55910 to +0.09790 under `Y or X`.

| Family and wording | Variant a, 16.0 → 64.0 | Variant b, 16.0 → 64.0 |
|---|---|---|
| Cursor, `X or Y` | 4/4 → 4/4 | 2/4 → 2/4 |
| Cursor, `Y or X` | 4/4 → 2/4 | 2/4 → 4/4 |
| Slot, both wordings | 6/6 → 6/6 | 5/6 → 5/6 |
| Outbox, both wordings | 4/4 → 4/4 | 4/4 → 4/4 |

The Cursor 2/4 outcomes fail both retry-direction targets and pass both regressions. Slot b selects `index < limit` and fails the last allocated slot regression. Each arm and wording compiles 6/6 selected lines and passes 5/6 target groups, 5/6 regression groups, and 4/6 full suites.

Single-prompt first-token draws pass 16/24 → 12/24 under `X or Y` and 14/24 → 12/24 under `Y or X`. Outbox accounts for the net losses, with 8/8 → 4/8 and 6/8 → 4/8, respectively. Restricted argmax passes 16/24 → 14/24 and 14/24 → 14/24. Cursor and Slot each have 0/4 paired passes on matched single presentations under either rule and either wording.

The control reproduced all 120 archived records. A new native readout process reproduced all 48 holdout records for the 64.0 adapter in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed code, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed all 34 examples. The comparison covers supplied-line ranking on three holdout families at one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `90c6eb36a38231fc421173af9e4266657cc35ef3a7677ec80670bee4aa146aaa` |
| Native training audit | `addf6daf4201439138aac2d4f059953c2e9101615fcea36fc774033e79c44239` |
| Center weight sixty-four adapter | `ecd88f0abc56a798b6d4773e5b94aae569c195c7e55dddc5223c887cbd557e66` |
| Results | `b2fad49732eb5852676592d68ee8a5be7a640d911a1971bb822d193abe9408c7` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-32-center-64
~~~

The default output directory is `real-choice-pairs-directed-hinge-32-center-64-v1` beside the `real-choice-pairs-directed-hinge-32-center-16-v1` control. Use `--out` with a fresh directory for another run.

## Directed hinge weight one hundred twenty-eight at center weight sixty-four

This ablation raises the multiplier of `max(0, 0.5 - s)` from 32.0 to 128.0. Both arms keep `64.0 * b^2` and restart from tool-four on the same Qwen3.5-2B Q4_K_M model. The native binaries, nine pairs, 50 Adam updates, and 100 target tokens are pinned. The driver rebuilds and tests both candidate lines in each holdout variant and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101.

### Ablation results

The native run took 703.8 seconds and updated both Float32 factors in all 28 projections. The first batch matched the control on CE, logit differences, and the center term. The hinge contribution rose from 19.29468 to 77.17871. Recomputing from the logged margins reproduced all 50 losses and slopes. The hinge was active on 49/50 → 44/50 batches.

On the 36 training presentations of the saved adapters, mean separation rose from 0.016 to 0.168 and RMS center rose from 0.049 to 0.189. Both signs were correct on 9/36 → 17/36 presentations, including 2/9 → 5/9 first presentations. The number meeting the directed floor rose from 0/36 to 4/36.

| Family | Wording | Separation, 32.0 → 128.0 | Absolute center, 32.0 → 128.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00070 → +0.00921 | 0.00276 → 0.04945 |
| Cursor | `Y or X` | −0.00113 → +0.00613 | 0.00118 → 0.02353 |
| Slot | `X or Y` | +0.00085 → +0.00378 | 0.02427 → 0.09270 |
| Slot | `Y or X` | +0.00163 → +0.00975 | 0.02192 → 0.08178 |
| Outbox | `X or Y` | −0.00234 → −0.02007 | 0.08968 → 0.60295 |
| Outbox | `Y or X` | −0.00059 → −0.01865 | 0.09849 → 0.57142 |

The flip slack `s - |b|` worsened in all four Cursor and Slot comparisons. It fell to −0.04024 and −0.01740 for Cursor and to −0.08892 and −0.07203 for Slot. Under `Y or X`, Cursor switched from `left > right` in both variants to `left < right` in both. Outbox ranked `pending_count > 0` first in both variants. Its smallest correct margin grew from +0.08734 to +0.58288 under `X or Y` and from +0.09790 to +0.55277 under `Y or X`.

| Family and wording | Variant a, 32.0 → 128.0 | Variant b, 32.0 → 128.0 |
|---|---|---|
| Cursor, `X or Y` | 4/4 → 4/4 | 2/4 → 2/4 |
| Cursor, `Y or X` | 2/4 → 4/4 | 4/4 → 2/4 |
| Slot, both wordings | 6/6 → 6/6 | 5/6 → 5/6 |
| Outbox, both wordings | 4/4 → 4/4 | 4/4 → 4/4 |

The Cursor 2/4 outcomes fail both retry-direction targets and pass both regressions. Slot b selects `index < limit` and fails the last allocated slot regression. Each arm and wording compiles 6/6 selected lines and passes 5/6 target groups, 5/6 regression groups, and 4/6 full suites.

Single-prompt first-token draws pass 12/24 → 16/24 under `X or Y` and 12/24 → 14/24 under `Y or X`. All net gains come from Outbox, which improves from 4/8 to 8/8 and from 4/8 to 6/8, respectively. Restricted argmax passes 14/24 → 16/24 and 14/24 → 14/24. Cursor and Slot each have 0/4 paired passes on matched single presentations under either rule and either wording.

The control reproduced all 120 archived records. A new native readout process reproduced all 48 holdout records for the hinge 128.0 adapter in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed code, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed all 34 examples. The comparison covers supplied-line ranking on three holdout families at one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `2885a5241a79cc2a64032015698bca6e8bf7c09baf081bd9663f55fdda19cb18` |
| Native training audit | `ad1177f60a20572be17fbd7ab7fc08feb5003a88d7de516e83ad80458b2a75cd` |
| Directed hinge 128.0 adapter | `2b63f7eae3be97df7101603e1d820675d0100e87e1f19666739abe7f7afb0b71` |
| Results | `38d1019cc50a0c3751bc475494183a43c24eb14c55a1c3b9c951bd402364968b` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-128-center-64
~~~

The default output directory is `real-choice-pairs-directed-hinge-128-center-64-v1` beside the `real-choice-pairs-directed-hinge-32-center-64-v1` control. Use `--out` with a fresh directory for another run.

## Center weight 256 at directed hinge weight 128

This ablation raises the coefficient of `b^2` from 64.0 to 256.0. Both arms keep `128.0 * max(0, 0.5 - s)` and restart from tool-four on the same Qwen3.5-2B Q4_K_M model. The native binaries, nine pairs, 50 Adam updates, and 100 target tokens are pinned. The driver compiles and tests both candidate lines in each holdout variant and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101.

### Ablation results

The native run took 698.5 seconds and updated both Float32 factors in all 28 projections. The first batch matched the control on CE, logit differences, and the hinge term. The center contribution rose from 6.53743 to 26.14970. Recomputing from the logged margins reproduced all 50 losses and slopes. The hinge was active on 44/50 → 49/50 batches.

On the 36 training presentations of the saved adapters, RMS center fell from 0.189 to 0.049 and mean separation fell from 0.168 to 0.015. Both signs were correct on 17/36 → 9/36 presentations, including 5/9 → 2/9 first presentations. The number meeting the directed floor fell from 4/36 to 0/36.

| Family | Wording | Separation, 64.0 → 256.0 | Absolute center, 64.0 → 256.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00921 → +0.00091 | 0.04945 → 0.00256 |
| Cursor | `Y or X` | +0.00613 → +0.00081 | 0.02353 → 0.00134 |
| Slot | `X or Y` | +0.00378 → +0.00109 | 0.09270 → 0.02400 |
| Slot | `Y or X` | +0.00975 → +0.00095 | 0.08178 → 0.02092 |
| Outbox | `X or Y` | −0.02007 → −0.00258 | 0.60295 → 0.08882 |
| Outbox | `Y or X` | −0.01865 → −0.00099 | 0.57142 → 0.09774 |

Both `|b|` and `|s|` fell in all six comparisons. The flip slack `s - |b|` improved in all four Cursor and Slot comparisons and ended below zero in each. At center weight 256.0 it was −0.00165 and −0.00053 for Cursor, and −0.02291 and −0.01997 for Slot. Under `Y or X`, Cursor switched from `left < right` in both variants to `left > right` in both. Outbox ranked `pending_count > 0` first in both variants. Its smallest correct margin fell from +0.58288 to +0.08623 under `X or Y` and from +0.55277 to +0.09675 under `Y or X`.

| Family and wording | Variant a, 64.0 → 256.0 | Variant b, 64.0 → 256.0 |
|---|---|---|
| Cursor, `X or Y` | 4/4 → 4/4 | 2/4 → 2/4 |
| Cursor, `Y or X` | 4/4 → 2/4 | 2/4 → 4/4 |
| Slot, both wordings | 6/6 → 6/6 | 5/6 → 5/6 |
| Outbox, both wordings | 4/4 → 4/4 | 4/4 → 4/4 |

The Cursor 2/4 outcomes fail both retry-direction targets and pass both regressions. Slot b selects `index < limit` and fails the last allocated slot regression. Each arm and wording compiles 6/6 selected lines and passes 5/6 target groups, 5/6 regression groups, and 4/6 full suites.

Single-prompt first-token draws pass 16/24 → 12/24 under `X or Y` and 14/24 → 12/24 under `Y or X`. Outbox accounts for the net losses, with 8/8 → 4/8 and 6/8 → 4/8, respectively. Restricted argmax passes 16/24 → 13/24 and 14/24 → 14/24. Cursor and Slot each have 0/4 paired passes on matched single presentations under either rule and either wording.

The control reproduced all 120 archived records. A new native readout process reproduced all 48 holdout records for the 256.0 adapter in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed code, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed all 34 examples. The comparison covers supplied-line ranking on three holdout families at one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `7227c8d029b12fc604ee9d27313d98ee907ef827165da2620bf4fc4c475d0a66` |
| Native training audit | `b8584e417975559132ecd4c6fea17fd6ac54252798b5d946072ff336e443c6fb` |
| Center weight 256.0 adapter | `984fc907d2f88cbb0ad7afc9f3eebfd0d65ec7d3d1f540bad810275da76f7245` |
| Results | `792b8cff26648bfa6847bad3b5a38f66237d69d9a39c9aeeefcf393260f4b58b` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-128-center-256
~~~

The default output directory is `real-choice-pairs-directed-hinge-128-center-256-v1` beside the `real-choice-pairs-directed-hinge-128-center-64-v1` control. Use `--out` with a fresh directory for another run.

## Directed hinge weight 512 at center weight 256

This ablation raises the multiplier of `max(0, 0.5 - s)` from 128.0 to 512.0. Both arms keep `256.0 * b^2` and restart from tool-four on the same Qwen3.5-2B Q4_K_M model. The native binaries, nine pairs, 50 Adam updates, and 100 target tokens are pinned. The driver compiles and tests both candidate lines in each holdout variant and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101.

### Ablation results

The native run took 710.3 seconds and updated both Float32 factors in all 28 projections. The first batch matched the control on cross-entropy, logit differences, and the center term. The hinge contribution rose from 77.17871 to 308.71484. Recomputing from the logged margins reproduced all 50 losses and the center and hinge derivatives. The hinge was active on 49/50 → 44/50 batches. Inactive batches with an incorrect sign increased from one to two.

On the 36 training presentations of the saved adapters, mean separation rose from 0.015 to 0.168 and RMS center rose from 0.049 to 0.188. Both signs were correct on 9/36 → 17/36 presentations, including 2/9 → 5/9 first presentations. The number meeting the directed floor rose from 0/36 to 3/36.

| Family | Wording | Separation, 128.0 → 512.0 | Absolute center, 128.0 → 512.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00091 → +0.00961 | 0.00256 → 0.04948 |
| Cursor | `Y or X` | +0.00081 → +0.00699 | 0.00134 → 0.02263 |
| Slot | `X or Y` | +0.00109 → +0.00359 | 0.02400 → 0.09270 |
| Slot | `Y or X` | +0.00095 → +0.00951 | 0.02092 → 0.08091 |
| Outbox | `X or Y` | −0.00258 → −0.02037 | 0.08882 → 0.60139 |
| Outbox | `Y or X` | −0.00099 → −0.01899 | 0.09774 → 0.56878 |

The flip slack `s - |b|` fell in all four comparisons where a correct choice depends on the context. It fell from −0.00165 to −0.03986 and from −0.00053 to −0.01564 on Cursor, and from −0.02291 to −0.08911 and from −0.01997 to −0.07140 on Slot. Both adapters had zero correct flips out of four. Under `Y or X`, Cursor switched from `left > right` in both variants to `left < right` in both. Outbox ranked `pending_count > 0` first in both variants. Its smallest correct margin rose from +0.08623 to +0.58103 under `X or Y` and from +0.09675 to +0.54980 under `Y or X`.

| Family and wording | Variant a, 128.0 → 512.0 | Variant b, 128.0 → 512.0 |
|---|---|---|
| Cursor, `X or Y` | 4/4 → 4/4 | 2/4 → 2/4 |
| Cursor, `Y or X` | 2/4 → 4/4 | 4/4 → 2/4 |
| Slot, both wordings | 6/6 → 6/6 | 5/6 → 5/6 |
| Outbox, both wordings | 4/4 → 4/4 | 4/4 → 4/4 |

The Cursor 2/4 outcomes fail both retry-direction targets and pass both regressions. Slot b selects `index < limit` and fails the last allocated slot regression. Each arm and wording compiles 6/6 selected lines and passes 5/6 target groups, 5/6 regression groups, and 4/6 full suites.

Single-prompt first-token draws pass 12/24 → 16/24 under `X or Y` and 12/24 → 14/24 under `Y or X`. All net gains come from Outbox, which improves from 4/8 to 8/8 and from 4/8 to 6/8, respectively. Restricted argmax passes 13/24 → 16/24 and 14/24 → 14/24. Cursor and Slot each have 0/4 paired passes on matched single presentations under either rule and either wording.

The control reproduced all 120 archived records. A new native readout process reproduced all 48 holdout records for the hinge 512.0 adapter in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed code, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed all 34 examples. The comparison covers supplied-line ranking on three holdout families at one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `0c82dc180307d534d34fa61cb67a731c8bde6ece49734d87abb624c2585ed9bf` |
| Native training audit | `5df5a712f64a509630eb0c1c492e9352afc3d514ba65754304f8ac2eb3695367` |
| Directed hinge 512.0 adapter | `311d97cf62f92dee72ade2060d5ddda0aff08ee3035bbceb9f7620638ea40e9d` |
| Results | `d4a0df07f5305b03c4c1b73c56f256ce27b2a789f8a78e245b330cefb2e780c4` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-512-center-256
~~~

The default output directory is `real-choice-pairs-directed-hinge-512-center-256-v1` beside the `real-choice-pairs-directed-hinge-128-center-256-v1` control. Use `--out` with a fresh directory for another run.

## Center weight 1024 at directed hinge weight 512

This ablation raises the coefficient of `b^2` from 256.0 to 1024.0. Both arms keep `512.0 * max(0, 0.5 - s)` and restart from tool-four on the same Qwen3.5-2B Q4_K_M model. The native binaries, nine pairs, 50 Adam updates, and 100 target tokens are pinned. The driver compiles and tests both candidate lines in each holdout variant and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101. Training seed 42 stays fixed.

### Ablation results

The native run took 702.3 seconds and updated both Float32 factors in all 28 projections. The first batch matched the control on cross-entropy, logit differences, and the hinge term. The center contribution rose from 26.14970 to 104.59881. Recomputing from the logged margins reproduced all 50 losses and the center and hinge derivatives. The hinge was active on 44/50 → 49/50 batches. Inactive batches with an incorrect sign fell from two to one.

On the 36 training presentations of the saved adapters, mean separation fell from 0.168 to 0.015 and RMS center fell from 0.188 to 0.049. Both signs were correct on 17/36 → 9/36 presentations, including 5/9 → 2/9 first presentations. The number meeting the directed floor fell from 3/36 to 0/36.

| Family | Wording | Separation, 256.0 → 1024.0 | Absolute center, 256.0 → 1024.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00961 → +0.00031 | 0.04948 → 0.00338 |
| Cursor | `Y or X` | +0.00699 → −0.00009 | 0.02263 → 0.00088 |
| Slot | `X or Y` | +0.00359 → +0.00101 | 0.09270 → 0.02397 |
| Slot | `Y or X` | +0.00951 → +0.00150 | 0.08091 → 0.02104 |
| Outbox | `X or Y` | −0.02037 → −0.00265 | 0.60139 → 0.08880 |
| Outbox | `Y or X` | −0.01899 → −0.00146 | 0.56878 → 0.09718 |

Both `|b|` and `|s|` fell in all six comparisons. The flip slack `s - |b|` improved from −0.03986 to −0.00307 and from −0.01564 to −0.00097 on Cursor, and from −0.08911 to −0.02296 and from −0.07140 to −0.01954 on Slot. Both adapters had zero correct flips out of four. Under `Y or X`, Cursor switched from `left < right` in both variants to `left > right` in both. Outbox ranked `pending_count > 0` first in both variants. Its smallest correct margin fell from +0.58103 to +0.08615 under `X or Y` and from +0.54980 to +0.09573 under `Y or X`.

| Family and wording | Variant a, 256.0 → 1024.0 | Variant b, 256.0 → 1024.0 |
|---|---|---|
| Cursor, `X or Y` | 4/4 → 4/4 | 2/4 → 2/4 |
| Cursor, `Y or X` | 4/4 → 2/4 | 2/4 → 4/4 |
| Slot, both wordings | 6/6 → 6/6 | 5/6 → 5/6 |
| Outbox, both wordings | 4/4 → 4/4 | 4/4 → 4/4 |

The Cursor 2/4 outcomes fail both retry-direction targets and pass both regressions. Slot b selects `index < limit` and fails the last allocated slot regression. Each arm and wording compiles 6/6 selected lines and passes 5/6 target groups, 5/6 regression groups, and 4/6 full suites.

Single-prompt first-token draws pass 16/24 → 12/24 under `X or Y` and 14/24 → 12/24 under `Y or X`. Outbox accounts for all six losses, with 8/8 → 4/8 and 6/8 → 4/8, respectively. Restricted argmax passes 16/24 → 14/24 and 14/24 → 14/24. Cursor and Slot each have 0/4 paired passes on matched single presentations under either rule and either wording.

The control reproduced all 120 archived records. A new native readout process reproduced all 48 holdout records for the center 1024.0 adapter in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed code, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed all 34 examples. The comparison covers supplied-line ranking on three holdout families at one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `45c45a3aeacc55d3686a36d51a28d213660cf3e7732b14aad9d60d7f6789d45b` |
| Native training audit | `3a5cda681c33ca8c3e2ad2dacb4a04df29fcb78944f89295ff54b664748712f6` |
| Center weight 1024.0 adapter | `cd49560f0cf3b4c78ecbd22ac786dcb37a1c02cd4644785c294e2d42ef7a925f` |
| Results | `e1f2358b017dfabf8e38b9379f6caab1748461a092e2b343b57595778647ffe4` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-512-center-1024
~~~

The default output directory is `real-choice-pairs-directed-hinge-512-center-1024-v1` beside the `real-choice-pairs-directed-hinge-512-center-256-v1` control. Use `--out` with a fresh directory for another run.

## Directed hinge weight 2048 at center weight 1024

This ablation raises the multiplier of `max(0, 0.5 - s)` from 512.0 to 2048.0. Both arms keep `1024.0 * b^2` and restart from tool-four on the same Qwen3.5-2B Q4_K_M model. The native binaries, nine pairs, 50 Adam updates, and 100 target tokens are pinned. The driver compiles and tests both candidate lines in each holdout variant and evaluates the same 48 requests at a 4,096-token cap and sampler seed 101. Training seed 42 stays fixed.

### Ablation results

The native run took 823.2 seconds and updated both Float32 factors in all 28 projections. The first batch matched the control on cross-entropy, logit differences, and the center term. The hinge contribution rose from 308.71484 to 1234.85938. Recomputing from the logged margins reproduced all 50 losses and the center and hinge derivatives. The hinge was active on 49/50 → 44/50 batches. Inactive batches with an incorrect sign increased from one to two.

On the 36 training presentations of the saved adapters, mean separation rose from 0.015 to 0.168 and RMS center rose from 0.049 to 0.189. Both signs were correct on 9/36 → 17/36 presentations, including 2/9 → 5/9 first presentations. The number meeting the directed floor rose from 0/36 to 3/36.

| Family | Wording | Separation, 512.0 → 2048.0 | Absolute center, 512.0 → 2048.0 |
|---|---|---:|---:|
| Cursor | `X or Y` | +0.00031 → +0.00901 | 0.00338 → 0.04985 |
| Cursor | `Y or X` | −0.00009 → +0.00654 | 0.00088 → 0.02305 |
| Slot | `X or Y` | +0.00101 → +0.00446 | 0.02397 → 0.09267 |
| Slot | `Y or X` | +0.00150 → +0.00949 | 0.02104 → 0.08064 |
| Outbox | `X or Y` | −0.00265 → −0.02069 | 0.08880 → 0.60195 |
| Outbox | `Y or X` | −0.00146 → −0.01952 | 0.09718 → 0.57001 |

The flip slack `s - |b|` fell from −0.00307 to −0.04085 and from −0.00097 to −0.01651 on Cursor, and from −0.02296 to −0.08821 and from −0.01954 to −0.07115 on Slot. Both adapters had zero correct flips out of four. Under `Y or X`, Cursor switched from `left > right` in both variants to `left < right` in both. Outbox ranked `pending_count > 0` first in both variants. Its smallest correct margin rose from +0.08615 to +0.58126 under `X or Y` and from +0.09573 to +0.55049 under `Y or X`.

| Family and wording | Variant a, 512.0 → 2048.0 | Variant b, 512.0 → 2048.0 |
|---|---|---|
| Cursor, `X or Y` | 4/4 → 4/4 | 2/4 → 2/4 |
| Cursor, `Y or X` | 2/4 → 4/4 | 4/4 → 2/4 |
| Slot, both wordings | 6/6 → 6/6 | 5/6 → 5/6 |
| Outbox, both wordings | 4/4 → 4/4 | 4/4 → 4/4 |

The Cursor 2/4 outcomes fail both retry-direction targets and pass both regressions. Slot b selects `index < limit` and fails the last allocated slot regression. Each arm and wording compiles 6/6 selected lines and passes 5/6 target groups, 5/6 regression groups, and 4/6 full suites.

Single-prompt first-token draws pass 12/24 → 16/24 under `X or Y` and 12/24 → 14/24 under `Y or X`. All six additional passes come from Outbox, which improves from 4/8 to 8/8 and from 4/8 to 6/8, respectively. Restricted argmax passes 14/24 → 16/24 and 14/24 → 14/24. Cursor and Slot each have 0/4 paired passes on matched single presentations under either rule and either wording.

The control reproduced all 120 archived records. A new native readout process reproduced all 48 holdout records for the hinge 2048.0 adapter in reverse request order. An independent scorer reconstructed choices and outcomes from the displayed code, raw logits, Crystal exit statuses, and test logs. The focused QLoRA specs passed all 34 examples. The comparison covers supplied-line ranking on three holdout families at one sampler seed.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `e84919b5725cedf5a7f54b877917c1ceee758d9432054130fa7ae5ebe79a1fb1` |
| Native training audit | `52b753305a95892012f3b571e7e78cbeb16754b0a56ae9e67bac870c4ff20079` |
| Directed hinge 2048.0 adapter | `33239bf6324a7c226b97e1a0ab96ad2e393902f8ed2c0e6b630beeabccde5f57` |
| Results | `0e22d4f8e36f019efdf5c386a5a82abc83b43177d83490da373fafd63c2e2aa3` |

### Reproduce the ablation

~~~sh
bins=build/qlora/scale-2b/real-choice-pairs
python3 -B eval/crystal_bugfixes/caller_transfer/real_pairs/run_centered.py \
  "$bins/train_directed_hinge" build/qlora/scale-2b/choice-first-token/readout \
  --directed-hinge-2048-center-1024
~~~

The default output directory is `real-choice-pairs-directed-hinge-2048-center-1024-v1` beside the `real-choice-pairs-directed-hinge-512-center-1024-v1` control. Use `--out` with a fresh directory for another run.
