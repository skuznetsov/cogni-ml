# Hy-Nomas first native video frontier

## Segment prompt edit frontier (2026-11-01)

Design-sealed scope: replace one noninitial five-second segment's authenticated
Qwen conditioning, preserve the parent's exact latent/KV boundary, and regenerate
all descendants. The following segment returns to its original prompt as a new
upstream text turn. Initial noise, target RoPE, cache epsilon, BF16 model weights,
F32 arithmetic and `atol=rtol=2e-5` are unchanged. Noise and prompt edits are
separate interventions. Camera controls and action-tail masks remain rejected.

The pinned upstream selects conditioning by prompt index and caches each new
turn's text once. All B video/text KV is invalid after editing B; restore after
A instead of retaining original B. Different visible text lengths must be
admitted before allocation, with correct per-turn KV extents and one shared
serial workspace. GPU source/output copies precede consumers on the existing
chain. No encode-time buffer allocation, readback or additional wait is allowed.

CAUTION: checkpoint/conditioning contracts change. A future edit must preserve
the post-A contract but bind every post-B descendant to the actual conditioning
bytes and text-turn schedule. Old executable-bound checkpoints remain rejected.
Rollback is the untouched original branch/default path at `2f2091c4`.

DoD: focused Crystal contract/history specs and Python hostile receipt tests;
tiny same-input upstream attention/cache checks at the standing tolerances;
guarded native original/B-edited controls and split restart B/C, with exact A
prefix and exact continuous-versus-restarted steps/timeline. Use a fresh common
post-A state from the new binary. Preserve the 10% cache-aware memory reserve,
process cap, swap/deadline stops and one terminal drain per process. Reject stale
post-B checkpoints, edited prompt metadata without its Qwen crop, wrong masks,
noise/RoPE changes, nonfinite tensors and vacuous B/C comparisons. Full upstream
video parity and visual quality require their own evidence.

The real three-segment qualification passes with original/edited visible text
lengths 557/585. The edited schedule is `[557,585,557]` with a new text cache
turn for each segment; the original schedule remains `[557,557,557]` with only
the initial text cached. Restore after A, replace only B's conditioning, then
regenerate B and C with the original initial noise. All eight B/C Euler updates,
both segment slices and the full 829,440-value timeline match bitwise between
continuous and split/restarted edited runs. Every A latent byte (552,960 bytes)
and the unused future suffix remain unchanged. Both B and C differ from their
original controls. The fresh original full bank also matches the previously
qualified original bank bitwise.

The prompt identity authenticates the conditioned manifest/payload, prompt
digest, actual valid text length and following-turn policy. Post-A remains
reusable; post-B binds that identity and its base boundary. Five real negative
checks reject stale original/edited branches, another authenticated prompt,
old executable state, combined noise/prompt edits and prefix regeneration
before Metal/model setup. The CPU comparator reconstructs the base after-B
contract from authenticated after-A state; a coordinated forged after-B digest
and descendant link is rejected by a hostile test. Eleven new Python tests,
59 total Python regression tests and 36 focused Crystal examples pass.

The new prompt's native Qwen prefix matches all 27 upstream BF16 states and
the 1,280-token conditioning crop bitwise (4,587,520 crop values). A tiny
native GPU fixture with changing text lengths and poisoned padding also matches
the pinned upstream causal attention/cache implementation at the unchanged
`2e-5` tolerances; maximum error is `7.45058e-9`, and cached K/V match exactly.
The real refiner additionally matches pinned upstream MPS bitwise for both
585-token B and 557-token original C, at timesteps 1000 and 206.897. All valid
`projected_text` rows and `vec` match with zero differing F32 bits. Native
storage retains 1,280 slots; padding-query outputs are not consumed or claimed
equal. Upstream full/compact valid-prefix controls also match bitwise. Each of
the four native refiner processes records zero encode allocations/readbacks
and one terminal wait. These establish scoped Qwen/refiner/history checks,
not full upstream DiT/decoder neural or video parity.

All five DiT intervals record zero instrumented encode allocations, readbacks
and preterminal waits, with one terminal drain each. The maximum sampled
footprint is 29.9163 GiB, minimum cache-aware headroom 17.0209 GiB and maximum
sampled swap growth about 0.82 MiB. Guards retain a 32 GiB process cap, 10%
reserve, 64 MiB swap-growth stop and 900 s deadline. Setup authentication and
terminal checkpoint/media I/O remain on the CPU; opaque MPS allocations and
between-sample spikes remain outside the counter/sample guarantees. The
ordered functional runs support no speedup conclusion.

Matched 15-second decoding uses the edited restart C bank and a previous
original raw decode only after proving its entire consumed latent bank equals
the fresh original bitwise. The first 100 pixel frames match bitwise. The
overlapping final 20 A pixel frames can change despite exact A latents; B and
C differ nonvacuously. Raw outputs contain 637,009,920 finite F32 bytes each.
Both 512x288/24 fps originals and the synchronized 1024x288 comparison contain
360 frames and last 15 seconds. Left is original; right is changed B plus
regenerated C. Lossless previews are sampled from the authenticated raw files.
This does not yet establish perceptual quality or instruction-following.

Artifact root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-prompt-edit-20261101.svygsr2x`.
`qualification-summary.json` freshly revalidates the native matrix, consumed
decoder banks, raw prefix and media hashes instead of trusting the export
adapter's pair-report boolean. The canonical scoped evidence index is
[the prompt-edit packet](../.landmark/evidence/2026-11-01-hy-nomas-prompt-edit.json).
The original branch/default remains intact. This slice admits one prompt edit,
exact checkpoints, existing geometry and no action-tail/camera controls. UI
integration, arbitrary prompt timelines, combined edits, new edited QBit
execution and full upstream neural/video parity remain open. Refresh on source,
binary, prompt/input/noise, weights, backend/device or geometry change, or
modification/loss of the local artifacts.

Native usage: prepare a separate authenticated generation-v2 conditioning
bundle for B, then restore the exact post-A state. Keep the original sequence
manifest/noise and pass all three prompt flags:

```sh
"$BINARY" --inputs "$SEQUENCE_INPUTS" --inputs-sha "$SEQUENCE_SHA" \
  --output-dir "$FRESH_EMPTY_SSD_DIR" --segments 3 \
  --start-segment 1 --stop-after-segments 2 \
  --state "$POST_A_STATE" --state-sha "$POST_A_SHA" \
  --prompt-edit-segment 1 --prompt-inputs "$B_INPUTS" \
  --prompt-inputs-sha "$B_INPUTS_SHA" \
  --state-format exact --cache-roundtrip exact --budget-mib 32768
```

Run this command under `examples/support/wan_memory_guard.py` with file-cache
accounting, 10% runtime floor, 32768 MiB footprint cap, 64 MiB swap-growth
stop and 900 s deadline. This is not an unguarded launch recipe. For C, keep
the same prompt flags, use interval `[2,3)` and restore the new edited post-B
state. C's original conditioning is selected automatically as a new text turn.
For a single continuous B/C process use `[1,3)`. All output directories must
already exist and be empty on the external SSD. `real/matrix-index.json` and
the linked command receipts record the exact executed qualification commands.

Focused refresh commands (use a fresh report path):

```sh
crystal spec spec/hy_nomas_segment_prompt_edit_spec.cr \
  spec/hy_nomas_segment_edit_spec.cr spec/hy_nomas_prompt_history_spec.cr \
  spec/hy_nomas_continuation_history_spec.cr \
  spec/hy_nomas_resume_bundle_spec.cr \
  spec/hy_nomas_sequence_admission_spec.cr -Dcpu_only
python3 -B -m unittest spec.hy_nomas_prompt_edit_resume_reference_test -v
python3 -B spec/support/hy_nomas_prompt_edit_resume_reference.py \
  --index /opt/mldata_ssd/cogni-ml/hy-nomas-prompt-edit-20261101.svygsr2x/real/matrix-index.json \
  --output "$RUN_ROOT/fresh-prompt-edit-report.json"
```

## Matched segment new-take videos (2026-11-01)

Status: decoder-pair integrity and bounded execution pass. The original B/C
continuation and edited restart C terminal banks produce matched 15-second
512x288/24 fps videos. Visual quality awaits user motion review. This step
decodes saved banks only; it runs no new DiT generation. Both branches use
exact checkpoints from the native matrix below.

The new comparator freshly authenticates that matrix and selects original
`[1,3]` and edited restart `[2,3]` terminals. It binds each full 180-frame bank
to the decoder receipt, actual executable bytes, original conditioning/noise,
upstream revision, weights and settings. Both native runs gather the first
90 latent frames and 360 decoder-noise frames per channel on the GPU, then
decode one global schedule of 17 windows at latent starts `[0,5,...,80]`.
Executing the pinned upstream pure `_window_starts` helper reproduces that
schedule. This establishes geometry only, not full upstream neural/video
parity. Native arithmetic and `atol=rtol=2e-5` remain unchanged.

The streamed raw planar RGB comparison proves all 44,236,800 F32 values in
pixel frames `[0,100)` bitwise equal. The overlap tail `[100,120)` changes in
5,106,470 values; B and C change in 51,622,966 and 51,852,014 of 53,084,160
values each. Their raw RMSEs are 0.105157857 and 0.135158929. These are
branch-difference diagnostics for an intended new take, not decoder parity,
quantization-error or quality measures. The numerical gate is unchanged. A's latent
prefix is still immutable, but the first changed decoder window begins at
4.167 seconds and includes B. Freezing all 120 A pixel frames would require
a separate boundary policy. The fresh original raw output additionally
matches the previous exact 15-second native control bitwise.

Both raw outputs contain exactly 637,009,920 finite F32 bytes. Terminal MP4
exports and the synchronized original-left/edited-right comparison pass
hash, frame-count, 15-second duration, geometry and codec checks. Lossless PNG
previews sample both seams and late frames directly from the validated raw
pixels. H264 is re-encoded; the raw files remain the fine-detail reference.
Selected stills retain the scene and characters while expression/background
details differ; this observation does not assess motion quality.

Both native processes pass the 26 GiB footprint cap, 10% cache-aware reserve,
64 MiB swap-growth stop and 1200 s deadline. Sampled peak is 18.8106 GiB,
minimum headroom 29.0074 GiB and swap growth zero. Instrumented encode
allocations/readbacks/preterminal waits are zero, with one terminal drain
per run. Opaque MPS allocations and between-sample spikes remain outside
those counter/sample guarantees. Request times are 230.2643 and 239.7489 s;
this single ordered pair supports no speedup conclusion.

A fresh release decoder build and 48 Python tests pass, including ten new
video-comparator tests with malformed source/noise/settings/counter/pixel
mutations and per-channel prefix checks. Same-lineage correlated adversary
review is ROBUST for pair integrity and bounded execution. Quality, full
upstream video parity, prompt editing and edited QBit execution stay open.

Artifact root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-segment-edit-video-20261101.0oq19ejm`.
`qualification-summary.json` links the fresh matrix comparison, source pins,
decoder/guard receipts, finite raw comparisons, prior default regression,
exports and previews. Native sources/objects/executable/runner were pinned
before GPU runs; checker and export sources were pinned before their runs.
Local-only hashes detect change and cannot recover lost files. Refresh on
code/binary/input/noise/weights/codec/geometry/backend/device changes or
artifact loss. Rollback reuses the untouched original branch.

DoD refresh commands (fresh report output; no model execution):

```sh
python3 -B -m unittest spec/hy_nomas_segment_edit_video_compare_test.py \
  spec/hy_nomas_segment_edit_resume_reference_test.py \
  spec/hy_nomas_segment_edit_reference_test.py \
  spec/hy_nomas_cache_qbit_video_compare_test.py
python3 -B spec/support/hy_nomas_segment_edit_video_compare.py \
  --edit-index /opt/mldata_ssd/cogni-ml/hy-nomas-segment-edit-20261101.qehg8u73/real/matrix-index.json \
  --original-decoder-dir /opt/mldata_ssd/cogni-ml/hy-nomas-segment-edit-video-20261101.0oq19ejm/decode/original \
  --edited-decoder-dir /opt/mldata_ssd/cogni-ml/hy-nomas-segment-edit-video-20261101.0oq19ejm/decode/edited \
  --output "$RUN_ROOT/fresh-edit-video-report.json"
```

Expected signals: 48 tests, `pair_integrity_passed=true`, exact first 100
pixel frames and finite full raw outputs. The saved external
`run_decode_pair.py` and `postprocess.py` record the guarded producer/export
commands. The next gate is user motion review of the two takes.

## Segment new-take frontier (2026-11-01)

Status: bounded native new-take implementation and exact-checkpoint restart
qualification pass. Preserve A's latent prefix and post-writer state, replace
B's initial noise with unused F noise from the authenticated input bank, and
regenerate B/C using the unchanged shared scene conditioning. All eight B/C
Euler updates and the full 829,440-value timeline are bitwise equal between
continuous and split/restarted edited runs. B and C both differ from the
unedited control; A and the unused future suffix remain bitwise unchanged.
This slice changes the noise take; per-segment prompt editing remains open.

Design laws and falsifiers:

- Only an active noninitial segment may change. Its donor must lie outside
  the entire active run, so generated latents can never become initial noise.
  Noise is gathered on the GPU; RoPE, cache epsilon and decoder noise retain
  the target's global offsets. Existing arithmetic and tolerances stay fixed.
- Post-A remains reusable in the same executable. Every checkpoint after an
  applied edit binds that edit's identity as well as its original full
  contract. A stale post-B state or another take must fail before model/GPU
  allocation. Setup authentication and terminal file I/O remain on the CPU.
- Exact checkpoints remain the default; compact restore/pack must obey the
  same branch identity. Original files are immutable and outputs are fresh.
- The checkpoint's trusted digest comes from the native producer/launcher.
  It authenticates bytes and contract; it does not certify the semantics of
  arbitrary forged states. Qualification also compares the unused donor and
  future suffix against the authenticated source bank before/after the fork.
- The decoder overlaps segment boundaries: changing B may change the last
  20 pixel frames of A even when every A latent is bitwise unchanged. Export
  must decode the continuous bank; whole-A pixel immutability is not claimed.

Rejected surface: prompt edits, arbitrary new seeds, multiple edits per
branch, cross-build checkpoint migration and Studio integration. The initial
input bank has six segments, so this donor-based slice admits runs with at
most five active segments. Rollback is to omit the opt-in edit and use the
original branch; no original checkpoint is rewritten.

Qualification uses five fresh native processes with BF16 weights, three active
five-second segments and four Euler updates per segment: produce/save A;
continue original B/C from A; continue edited B/C from the same A; produce/save
edited B; restart C from that new B. The actual edited post-B bundle is rejected
by both the unedited branch and a different-donor branch before model setup.
Exact and QBit stale-contract gates also pass focused specs; the real edited
matrix tests exact checkpoints only. QBit branch execution is not qualified
by this matrix. The unedited full timeline additionally matches the previously
qualified native exact control bitwise; this is a default-path regression
check, not new upstream neural parity.

The tiny GPU probe checks donor gather, target-only scatter, retained prefix,
target RoPE selection and C's unchanged noise/position. Executed temporal
gather/scatter expressions from pinned upstream revision
`911cea6458a89c2663513ff7099561ec190dd491` match the terminal native GPU fixture
with maximum error zero. The upstream RoPE check establishes its source-level
global-offset contract only, not full rotary or neural numerical parity.
No full BF16/CUDA/Magi or decoded-video quality qualification is claimed.

All five native intervals record zero instrumented encode allocations,
readbacks and preterminal waits, with one terminal drain each. The original
and edited B/C runs both encode 19,457 command segments: the substitution only
changes the existing noise-gather offset. Sampled peak process footprint is
29.2121 GiB, minimum cache-aware headroom 19.4075 GiB and swap growth zero.
All pass the 30 GiB process cap, 10% reserve, 64 MiB swap-growth stop and 900 s
deadline. Opaque MPS allocations are outside the allocation counters; polling
cannot bound between-sample spikes. Cache-aware headroom estimates reclaimable
file-backed memory. Ordered run timings are not a speedup experiment.

Release builds, 48 Crystal examples, 47 Python tests and formatting pass.
Rejection tests challenge donor/target selection, edited checkpoint identity,
lineage, retained A/future suffix, vacuous C regeneration, drift and counters.
Tolerances remain `atol=rtol=2e-5`. Exact post-A and edited post-B bundles are
4,319,223,820 and 8,141,283,340 bytes; this slice does not reduce live K/V
residency. The matched decoder follow-up is recorded above; visual quality
remains open.

Artifact root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-segment-edit-20261101.qehg8u73`.
`qualification-summary.json` authenticates native/checker pins, command and
guard receipts, the real matrix, default regression and branch rejection
results. Native sources/executable/runner were pinned before full-size runs;
auxiliary finalization hashes are not original-execution attestations. The
same-lineage correlated adversary verdict is ROBUST for this bounded native
continuation. Local-only digests detect changes and cannot
recover lost artifacts. Refresh on source/binary/input/noise/weights/codec/
geometry/backend/device change or artifact loss.

DoD refresh commands (use a fresh external build/output directory):

```sh
make BUILD_DIR="$RUN_ROOT/build" spec_hy_nomas_segment_edit spec_hy_nomas_compact_resume
make BUILD_DIR="$RUN_ROOT/build" \
  "$RUN_ROOT/build/hy_nomas_segment_edit_probe" \
  "$RUN_ROOT/build/hy_nomas_compact_sequence_probe"
python3 -B -m unittest discover -s spec -p 'hy_nomas*resume_reference_test.py' -v
python3 -B spec/hy_nomas_segment_edit_reference_test.py -v
python3 -B spec/support/hy_nomas_segment_edit_resume_reference.py \
  --index /opt/mldata_ssd/cogni-ml/hy-nomas-segment-edit-20261101.qehg8u73/real/matrix-index.json \
  --output "$RUN_ROOT/fresh-matrix-report.json"
```

Expected signals: 48 Crystal examples, 47 Python tests, `passed=true`,
`edited_restart.all_steps_bitwise_equal=true` and full-timeline bitwise equality.
The saved external `run_edit_matrix.py` records the bounded five-process
commands and enforces pins. New-take options are paired, zero-based
`--noise-edit-segment 1 --noise-source-segment 5`; the runtime reuses post-A
state and binds every descendant to this identity. To resume C, use the newly
written post-B state and the same options, never the old branch's post-B state.
Decoder invalidation is global latent frames `[30,90)` and pixel frames
`[100,360)` for this case; the overlap reaches A's final 20 pixel frames.
Matched continuous-bank decoding is now qualified above; the next gate is
user motion review. Prompt edits and CineStudio integration need a separate
conditioning/cache invalidation design.

## Matched 15-second compact-checkpoint video review (2026-11-01)

Status: the saved exact and compact three-segment restart chains have been
decoded and exported successfully. This is a decoder-only follow-up to the
matrix below, with no new DiT generation. On 2026-11-01 the user reported that
QBit works with practically no visual differences in this 15-second pair after
two compact save/restart boundaries. This is a visual observation for one
scene and configuration, not general perceptual equivalence. Exact checkpoints
remain the default.

The native imported-latent example now admits an opt-in 90-frame prefix from
the authenticated 180-frame NCTHW bank. Both arms use the same frozen native
decoder executable, BF16 block/head weights, F32 stem, original noise record
and settings. GPU prefix gathers select the first 90 latent frames and 360
noise frames **per channel**. One global schedule uses 17 overlapping windows
at latent starts `[0,5,...,80]`; it does not concatenate independently decoded
segments. The pinned upstream `_window_starts` source helper returns the same
schedule, and the native config test checks full coverage (40 single-covered
and 320 double-covered output frames). This is an upstream geometry check,
not a new full upstream decoder numerical qualification.

The video comparator freshly authenticates the compact matrix, selects the
terminal `[2,3]` runs from `exact_restart_chain` and `compact_restart_chain`,
binds each decoder's full-bank path and digest, and checks identical decoder,
weights, noise and settings. Both raw outputs have 637,009,920 finite F32 bytes.
Each original MP4 has 360 frames at 512x288/24 fps (15 seconds). A synchronized
1024x288 comparison places exact on the left and compact QBit on the right.
That comparison is H264 re-encoded; separate MP4s, raw F32 files and lossless
PNG previews remain available for detail review.

The unchanged-prefix falsifiers pass bitwise:

- Exact versus QBit matches for output frames `[0,100)`.
- Each new arm matches its prior 10-second raw-pixel control for `[0,220)`.

An overlapping decoder window at latent start 25 sees A and B, so B can affect
the last 20 pixel frames of A. Start 55 likewise crosses B/C and affects pixels
from frame 220. Entire A/B pixel segments are therefore not immutable when
future latents change, even though their DiT latent prefixes are immutable.
Display-range pixel differences versus exact are diagnostics, not quality
acceptance thresholds:

| Pixel segment | MAE | RMSE | Maximum absolute error |
| --- | ---: | ---: | ---: |
| A, frames 0-119 | 0.000144225 | 0.000922836 | 0.1328125 |
| B, frames 120-239 | 0.006521547 | 0.016688471 | 0.5703125 |
| C, frames 240-359 | 0.018942099 | 0.047378404 | 0.984375 |

Sampled peak process footprint is 18.8096 GiB, minimum cache-aware headroom is
25.1403 GiB and swap growth is zero. Both serial runs pass the 26 GiB process
cap, 10% reserve, 64 MiB swap-growth stop and 1200 s deadline. Instrumented
decoder encode allocations/readbacks/preterminal waits remain zero, with one
terminal wait per process. Opaque MPS allocations and between-sample spikes
remain outside those guarantees; terminal file analysis/export uses the CPU
after the existing drain. Decoder wall times are 220.95 s and 237.18 s;
this single ordered pair is not a speedup experiment.

Release build, 39 Crystal config/assembly examples, 19 Python comparator tests
and Crystal formatting pass. New rejection cases cover unadmitted duration,
wrong bank/noise, missing window, wrong output geometry, wrong terminal matrix
arm and ambiguous CLI modes. Tolerances remain `atol=rtol=2e-5`; lossy QBit
versus exact is not promoted to exact equivalence. Review is correlated
same-lineage review. Full upstream BF16/CUDA video parity remains unproven.

Artifact root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-compact-video-20261101.8cfa0q7j`.
`qualification-summary.json` indexes source/executable pins, command receipts,
guard samples, raw checks and exports. `videos/f32-vs-adaptive-qbit.mp4` is the
comparison; originals are `videos/f32-control.mp4` and
`videos/adaptive-qbit.mp4`. All large artifacts stay outside the repository.
Digests detect changes and do not guarantee recovery of local-only artifacts.
Refresh on code/binary/input/noise/weights/codec/geometry/backend/device change
or artifact loss. The user visual-review gate for this pair is satisfied;
existing raw qualification/export receipts retain their original pending
status as execution-time records. No requantization-policy change follows
from this observation. The bounded noise-take fork described above now
qualifies segment invalidation and restart lineage. Its decoded visual review
and CineStudio integration remain later work.

The integrity command is below; choose a fresh output path:

```sh
python3 -B spec/support/hy_nomas_cache_qbit_video_compare.py \
  --compact-index /opt/mldata_ssd/cogni-ml/hy-nomas-compact-checkpoint-20261101.3efczbmo/real/matrix-index.json \
  --exact-decoder-dir /opt/mldata_ssd/cogni-ml/hy-nomas-compact-video-20261101.8cfa0q7j/decode/exact \
  --qbit-decoder-dir /opt/mldata_ssd/cogni-ml/hy-nomas-compact-video-20261101.8cfa0q7j/decode/qbit \
  --output /a/fresh/external/path/pair-validation.json
```

Expected integrity signal: `pair_integrity_passed=true`, authenticated 90-frame
prefix / 360-frame pixels, execution-time `quality_verdict` pending and
`full_upstream_video_parity=false`. The external `postprocess.py` and
`run_decode_pair.py` record exact guarded commands and enforce source pins;
their output paths are intentionally fresh-only.

## Compact QBit checkpoints and repeated continuation (2026-11-01)

Status: bounded native compact save/restart qualification passed. Eight fresh
native processes cover three five-second segments (A/B/C), four Euler updates
each, with BF16 DiT weights unchanged. Exact restart is bitwise equal to its
continuous control; compact restart is bitwise equal to the old in-memory
QBit roundtrip control across all 12 updates and the entire latent/noise bank.
The user reported no visible difference in the previous matched 10-second
F32/QBit video pair. That observation does not qualify the new three-segment
continuation's perceptual quality; this run did not decode video.

Admit an opt-in compact boundary format: preserve the complete latent/noise bank
exactly and store every active K/V record as authenticated row headers plus its
used Gaussian p4/p5/p8/F32 payload. Bind codec policy/codebook, exact checkpoint
contract, next-segment cursor and payload inventory. Keep the existing exact
bundle and exact F32 default unchanged. File I/O and integrity checks occur only
in setup or after the producer's existing terminal drain; quantization and
reconstruction run on GPU. Pack each row into its own existing 512-byte F32
slot, after retaining all source values/codes needed for that row; restore must
stage compressed codes before overwriting the same slot. Only row headers,
codebook and stats are allocated before encode. Terminal serialization streams
the occupied prefixes and rebases disk offsets; it does no CPU quantization.
Packing is destructive and admitted only at the producer's final extendable
boundary. This reduces checkpoint storage; live attention
still consumes F32 buffers and no residency or speedup is assumed.

Qualification compares a continuous exact control, a continuous control
with the same QBit roundtrips at both boundaries, and separate exact and compact
fresh-process chains through A/B/C. Compact restart must be bitwise equal to
the matching roundtrip control, while lossy-versus-exact errors are reported
separately per segment. Preserve
`atol=rtol=2e-5`, the 10% cache-aware memory reserve, sampled process caps and
swap-growth/deadline stops. Only one model-owning process runs at a time.

Reject malformed/truncated/modified files, codec or base-contract mismatch,
wrong inventory/cursor, invalid tier/row moments/payload offsets, nonfinite
rows, partial/repeated restore, and save before the noisy writer drains.
Do not add an encode-time readback, allocation or wait. The relevant guard is
zero instrumented encode allocations/readbacks/preterminal waits and one
terminal wait per process; opaque MPS resources remain outside those counters.
Full stock upstream BF16/CUDA parity and quality of long continuations remain
unproven; refresh the pinned cache-contract upstream oracle and codec checks.

Actual files include the exact 3,317,760-byte latent/noise bank, row headers,
codec metadata, record digests and the bundle envelope:

| Boundary | Exact checkpoint bytes | Compact checkpoint bytes | Exact / compact |
| --- | ---: | ---: | ---: |
| After A | 4,319,223,820 | 2,005,497,939 | 2.154x |
| After B | 8,141,283,340 | 3,755,273,865 | 2.168x |

Active logical F32 K/V bytes are 4,314,857,472 after A and 8,136,916,992 after B;
their packed representations, including row headers, are 2,001,104,400 and
3,750,880,016 bytes. The resident history capacity is 8,136,916,992 bytes for
this three-segment run even in the A producer; do not confuse capacity with
the active prefix. No second packed-payload GPU bank is allocated. Codec-only
named budget is 134,846,336 bytes in the A producer and 389,132,032 bytes
for the B restore/save process. Restore-plan capacity is checked against that
conservative budget before encode.

Loss versus the exact F32-cache control is real and grows across boundaries. Final-segment
latent RMSE / relative L2 / maximum absolute error are:

| Segment | RMSE | Relative L2 | Maximum absolute error |
| --- | ---: | ---: | ---: |
| A | 0 | 0 | 0 |
| B | 0.08850569793 | 0.07588765 | 2.251953125 |
| C | 0.23010752454 | 0.18802071 | 3.2216796875 |

B and C fail the unchanged exact tolerance. Compact disk serialization adds
no error to the matched QBit treatment, but repeated lossy continuation is not
promoted to an exact or generally perceptually equivalent mode. Full timeline
prefixes and untouched future noise remain bitwise unchanged at each interval.

All eight native model processes recorded zero instrumented encode allocations,
readbacks and preterminal waits, with one terminal wait each. Sampled peak
footprint is 29.5670 GiB, minimum cache-aware headroom 14.5482 GiB and swap
growth zero, under a 30 GiB process cap, 10% reserve, 64 MiB swap-growth stop
and 900 s per-process deadline. Samples do not bound between-sample spikes;
file-backed memory is a reclaimability estimate. No timing speedup is claimed.

The small fresh-process GPU codec probe matches all 768 F32 cache values and
the timeline bitwise against the existing in-memory codec, including p4/p5/p8,
exact escapes, constants and signed zero. Focused checks pass 37 Crystal
examples and 17 Python comparator tests. Both legacy sequence callers build
in release mode. Corruption, bounded-copy, timestamp-preserving mutation,
inventory/contract, lifecycle and stale-treatment falsifiers are covered.
The refreshed pinned upstream source-body F32/MPS cache-attention adapter
matches all 9,072 output, 1,080 K and 1,080 V values with maximum error zero.
Its synthetic 54-layer/14-forward fixture does not qualify full real BF16 DiT,
CUDA/Magi or PixelUMM equivalence.

Reproduction uses the external artifact root below. `run_job.py` exports its
`TMPDIR` and `CRYSTAL_CACHE_DIR`, and applies the cache-aware guard. The matrix
runner fixes one executable and runs each of its eight intervals serially.
Exact command arrays, native/checker source pins, binary pins and per-run guard summaries are
indexed by the evidence packet and external `qualification-summary.json`.
The external orchestration scripts are hashed at finalization; their original
execution bytes have no separate attestation. Authenticated outputs, linked
state hashes and per-arm sidecars are the numerical evidence.

```sh
ROOT=/opt/mldata_ssd/cogni-ml/hy-nomas-compact-checkpoint-20261101.3efczbmo
python3 -B "$ROOT/run_job.py" --cap 4096 --timeout 600 specs-compact-resume -- \
  make BUILD_DIR="$ROOT/build" spec_hy_nomas_compact_resume
python3 -B "$ROOT/run_job.py" --cap 4096 --timeout 300 specs-compact-comparator-final -- \
  python3 -B spec/hy_nomas_compact_resume_reference_test.py
python3 -B "$ROOT/run_real_matrix.py"
python3 -B "$ROOT/run_job.py" --cap 1024 --timeout 300 real-matrix-compare -- \
  python3 -B spec/support/hy_nomas_compact_resume_reference.py \
  --index "$ROOT/real/matrix-index.json" --output "$ROOT/real/matrix-report.json"
```

These commands require fresh log/output paths; use a new artifact root when
refreshing. DoD for compact native restart passed at unchanged tolerances;
visual quality of C remains open. CAUTION rollback is the unchanged default
`--state-format exact --cache-roundtrip exact`, or reverting the opt-in helper
and probe. Cross-build migration, crash/power-loss recovery and Studio
integration are not qualified. Refresh on source/binary/codec/input/weights/
geometry/backend/device changes. Evidence:
`.landmark/evidence/2026-11-01-hy-nomas-compact-resume.json`.

## Adaptive continuation-cache quality experiment (2026-10-31)

Status: bounded exact-versus-lossy comparison integrity passed. On 2026-11-01,
the user reviewed the matched pair and reported no visible difference. This
single-pair visual observation does not establish numerical or general quality
parity. The four-step BF16 DiT exact
control/save/restart gate passed: all four B updates and the full timeline are
bitwise equal, with A and the untouched future noise unchanged. The QBit arm
restores the same authenticated post-writer boundary, uses the same B
conditioning/noise and four Euler updates, then both timelines use one frozen
decoder executable, identical BF16 decoder weights/settings and decoder noise.
The comparison covers two five-second segments (10 seconds total), 240 frames
at 512x288 and 24 fps. Decoder settings are one step, a 10-latent window, five
latent overlap, and starts 0, 5, ..., 50; the same first-240-frame noise prefix
is used for both. The existing exact F32 path remains the default.

For B, QBit K/V roundtrip has RMSE 0.08850569793, relative L2 0.07588765 and
maximum absolute error 2.251953125; it is not bitwise equal and fails the
unchanged `atol=rtol=2e-5` latent tolerance. This is the expected falsifier for
exactness of a lossy cache representation, not a video-quality verdict. The
decoder pair's provenance and output integrity checks passed, and two finite
240-frame videos were exported with matching decoder/noise/settings receipts.
Full-upstream video parity is not established. The exact arm's refreshed
timeline matches the stored pinned source-body compact-MPS final-latent oracle
for all 276,480 selected values, and its 106,168,320 output pixel-channel
values match the previous native decode by full SHA-256. These are
stored-oracle/native-decode comparisons, not a new full-upstream CUDA DiT or
upstream PixelUMM decoder run. The user's later visual observation is scoped
to this pair; broader quality promotion remains unproven.

The first experiment retains the final F32 attention buffers. A reusable GPU
scratch packs and restores one cache record at a time, so it isolates lossy
representation error without changing attention arithmetic. It does not yet
reduce live cache residency or write a compact restart bundle. Report logical
compact bytes including all row headers separately from scratch capacity and
the exact restart file size. Do not infer GPU-memory savings from this probe.

Each 128-value head row has F32 mean/sigma, payload offset and tier (16 header
bytes). Gaussian conditional centroids reconstruct progressive p4/p5 prefixes;
p8 uses the existing Gaussian codec's full-code reconstruction. Select the
lowest tier satisfying both normalized maximum and RMS residual bounds:
p4 (0.50, 0.12), p5 (0.25, 0.06), p8 (0.10, 0.02); otherwise retain exact F32.
These are candidate selection heuristics, not changes to numerical tolerances
or a quality certificate. Nonfinite data invalidates the run. Preserve exact
F32 escapes and constant rows, and keep selection entirely on the GPU.

Falsifiers exercised in this run: small GPU pack/restore matches the existing
Gaussian decoder on identical stored moments/codes at unchanged
`atol=rtol=2e-5`; corrupted payload and nonfinite controls are detected. The
exact restart gate remained separate from lossy continuation. In the paired
run, A and untouched future noise remain bitwise equal, while the measured B
output differs as reported above. Both finite pixel outputs used identical
settings and were exported for user review. Exact-path upstream qualification
retains its preceding scope; lossy continuation is not called upstream-
equivalent.

The logical compact K/V representation is 2,001,104,400 bytes versus
4,314,857,472 F32 bytes (2.156x including row headers). The codec adds
41,207,936 resident bytes for this comparison. F32 attention caches remain
resident and the exact restart bundle remains 4,319,223,820 bytes: this run
shows neither live GPU-memory savings nor a compact on-disk restart format.
No speedup is claimed; the compared runs were fresh processes without a matched
timing design.

The small GPU codec probe passed its tier, Gaussian CPU-parity, mutation and
nonfinite checks. Four DiT and two decoder runs each recorded zero instrumented
encode allocations/readbacks/preterminal waits and one terminal wait. Sampled
peaks were 24.7916 GiB for DiT and 18.4178 GiB for decode; minimum measured
headroom across phases was 15.4279 GiB and swap growth was zero. Guards used
30 GiB DiT and 26 GiB decoder caps, a 10% system-memory reserve and a 64 MiB
swap-growth stop. Sampling does not bound between-sample peaks. The focused
suite passed 38 Crystal examples and 36 Python tests.

Reproduction used external scratch/build state and the guarded runners below;
their exact argv and receipts are under `$ROOT/logs/` in the evidence root.
`run_job.py` sets `TMPDIR` and `CRYSTAL_CACHE_DIR` to that root and invokes the
memory guard with file-cache accounting, the 10% reserve and 64 MiB swap stop.

```sh
ROOT=/opt/mldata_ssd/cogni-ml/hy-nomas-cache-qbit-20261031.vnlq7utw
export TMPDIR="$ROOT/tmp" CRYSTAL_CACHE_DIR="$ROOT/crystal-cache"
BUILD_DIR="$ROOT/build"

python3 -B "$ROOT/run_job.py" --cap 4096 --timeout 600 integrated-build -- \
  make BUILD_DIR="$BUILD_DIR" hy_nomas_sequence_resume_probe hy_nomas_decode_latent
python3 -B "$ROOT/run_job.py" --cap 4096 --timeout 600 final-crystal-spec -- \
  make BUILD_DIR="$BUILD_DIR" spec_hy_nomas_resume ARGS=spec/pixelumm_generation_spec.cr
python3 -B "$ROOT/run_job.py" --cap 4096 --timeout 300 final-python-spec-v2 -- \
  python3 -B "$ROOT/run_cpu_checks.py"
python3 -B "$ROOT/run_real_matrix.py"
python3 -B "$ROOT/run_decode_pair.py"
python3 -B "$ROOT/run_export_pair.py"
```

The real-matrix runner invokes
`spec/support/hy_nomas_sequence_resume_reference.py` on control/save/exact and
`spec/support/hy_nomas_cache_qbit_compare.py` on exact/QBit outputs. The decoder
runner invokes `hy_nomas_decode_latent` twice with `--latent-frames 60
--latent-bank-frames 180`, then
`spec/support/hy_nomas_cache_qbit_video_compare.py` binds both decoder receipts
to their full timeline-bank SHA-256 and validates the raw pixels before export.
The export runner uses `examples/hy_nomas_export.py` for 240 frames at
512x288/24 fps and validates the MP4 files with `ffprobe`. The reports and exact
guarded command records are indexed by
`.landmark/evidence/2026-10-31-hy-nomas-cache-qbit.json`.

DoD for comparison integrity passed; the user's visual acceptance is limited
to this pair. The main failure mode remains amplification of local K/V errors
by continuation.
Removing opt-in codec selection and diagnostic files is rollback; defaults stay
exact F32. Refresh on codec/selector, binary, input, weights, backend or device
changes.

## Segment-boundary save/restart frontier (2026-10-31)

Status: qualified for the bounded native save/restart path. A real BF16
54-layer DiT stopped after segment A's separate noisy cache-writer forward,
saved its state and exited. A fresh process restored that state and reproduced
segment B with the same authenticated noise and conditioning. All four B Euler
updates (138,240 F32 values each) and the entire latent timeline (829,440 values)
are bitwise equal to uninterrupted A-to-B; A's prefix is unchanged. Pixel
decoding and complete video generation were not run.

The saved boundary must contain the clean/generated latent timeline with the
untouched future noise, the valid F32 K/V prefix of every layer, the next
segment cursor, and an exact identity for weights, conditioning, noise banks,
geometry, solver, text-cache policy and implementation. A last-frame latent
alone is rejected. Mid-denoise, pre-writer and final/non-extendable boundaries
are rejected. Restore is admitted only into fresh owners after the entire
bundle identity, inventory and payload have authenticated.

File I/O occurs after the producer's existing terminal drain and before the
consumer encodes GPU work. Shared Metal storage is streamed directly to/from
disk without intermediate CPU tensor arrays or duplicate GPU cache banks.
Each native process retains one terminal wait; instrumented encode allocations,
readbacks and preterminal waits are zero. These counters do not include opaque
MPS resource allocations. Restart changes the process boundary, not the
numerical path. `atol=rtol=2e-5` and the 10% system-memory reserve are unchanged.
Checkpoint storage does not bound growing continuation history.

Falsifiers: corrupted/truncated payload, wrong identity/geometry/policy,
partial or repeated restore, changed future noise, wrong temporal offset,
and a continuation without A's K/V must fail or change the measured output.
The synthetic 54-layer probe also reproduces B and C bitwise after restart.
Its continuous control matches the pinned upstream history-attention oracle
at revision `911cea6458a89c2663513ff7099561ec190dd491`: all 9,072 attention-output
and 2,160 final-cache values are bitwise equal. This is the F32 MPS adapter of
the extracted upstream cache-attention function; it does not establish full
real-DiT or stock BF16/CUDA continuation parity. Zeroed/changed K/V and changed
future Q/K/V inputs are detected. CPU checks reject corrupted/truncated files,
identity or inventory mismatch, repeated restore and unsafe boundaries.

The real checkpoint is 4,319,223,820 bytes (about 4.02 GiB), including the
bounded header. It holds 109 records: the timeline and 54 K/V pairs. The real
matrix kept a 30 GiB sampled process cap, 10% reserve, 64 MiB swap-growth stop
and 900-second deadline per process. Sampled maximum footprint was 24.792 GiB,
minimum cache-aware headroom 18.571 GiB, and swap growth zero. Sampling cannot
exclude between-sample peaks. Saving/export took 11.197 seconds; fresh-process
state authentication is part of the 8.860-second preflight, and direct restore
took 2.434 seconds. These are one-run observations, not a speedup claim.

Qualification requires the same unchanged executable on the same host for
producer, control and consumer. The contract hashes the executable pathname;
hot replacement while a process is alive is unsupported because that pathname
need not identify its loaded image. Cross-build/device migration, crash or
power-loss recovery, and a Studio resume API remain unqualified. Full-file and
per-record digests authenticate exact bytes, not untrusted-code provenance.

DoD passed: the small GPU/upstream matrix, the three-process real BF16 comparison,
28 Crystal examples and 15 Python tests. Build with `make hy_nomas_resume_probe
hy_nomas_sequence_resume_probe`; CPU checks use `make spec_hy_nomas_resume`,
`python3 -B spec/hy_nomas_sequence_resume_reference_test.py` and the existing
`spec/hy_nomas_continuation_reference_test.py` in the reference environment.
Use an external `BUILD_DIR`, `TMPDIR` and `CRYSTAL_CACHE_DIR`. The real comparator
is `spec/support/hy_nomas_sequence_resume_reference.py`; it hashes the checkpoint
in bounded chunks without retaining a CPU copy. Exact guarded commands, binary
identity and reports are indexed by
`.landmark/evidence/2026-10-31-hy-nomas-segment-restart.json`.

Rollback: remove the opt-in resume entry points; keep the uninterrupted control.
Refresh on implementation, executable, weights, inputs, cache policy, geometry,
backend or device change. Next: qualify edited A and invalidation of its dependent
continuation; restart parity alone does not qualify selective regeneration.

The [performance paradigm audit](hy-nomas-performance-paradigm-audit.md) separates
current execution evidence, workload counts and proposed architecture changes.
Its CPU-only work-budget helper does not qualify a new kernel or decoder.

## FFN/attention frequency qualification (2026-10-31)

The diagnostic ran the existing full J21120, 36-block-plus-head fixture
with `--profile-aa` and an external power observer. It is not the J12480
saved-latent recovery workload or full generation. Six requests reuse the
same weights, input, scratch and selectors; two requests enable existing
phase/API/continuation metadata. The sole terminal wait remains unchanged.
The observed mean active GPU frequency must differ by no more than 2% for a
profile/control comparison, with at least five valid interior power samples
and mean sampled active residency of at least 99%. Frequency mismatch rejects
timing attribution; multiplying wall time by MHz is not a substitute.

GPU root intervals are partitioned into disjoint family-membership buckets.
Mixed boundary roots and cross-family overlap stay unassigned rather than
being divided or counted twice. These are command-buffer interval bounds,
including stalls, not exclusive shader service or occupancy measurements.
Existing Graph API wall time and nested continuation time remain separate.

The leading measured candidate is dense FFN cost. A diagnostic-only
falsifier is a single connected FFN MPSGraph, preserving every BF16 boundary
between gate/up, F32 SiLU, F32 product and down. This differs from the refuted
shared gate/up graph, which left activation and down in separate encodes.
The first gate compares all five FFN intermediates on the same saved input
against the existing upstream controls and baseline, at unchanged
`atol=rtol=2e-5`, with a nonfinite positive control. An altered SiLU or folded
rounding boundary rejects the candidate before timing. Only a passing
candidate may proceed to reused-buffer, paired full-block timing at matched
frequency. A lower command count alone is insufficient for promotion.

This is a CAUTION performance experiment: production selectors remain
unchanged and removing the diagnostic files is the rollback. The process
guard retains cache-aware accounting, a 10% physical reserve, a 64 MiB
swap-growth stop, a bounded deadline and a declared footprint cap. DoD is a
guarded numerical report plus frequency-qualified paired timing and focused
CPU checks; an unsuccessful candidate is retained only as a scoped refutation.

The six-request J21120 run completed with all saved intermediate and head
comparisons passing. Request encode-plus-wait times were 43.662, 43.592,
42.322, 42.273, 42.273 and 42.280 seconds. The two profiled requests averaged
1393.11 and 1393.16 MHz, with 100% sampled active residency. Their preceding
off control averaged 1346.11 MHz: the 3.49% difference fails the declared 2%
two-sided frequency gate, although the following off control matches. These
results do not establish frequency-matched observer overhead or a speedup.

Both profiled requests have stable descriptive accounting. In request 2,
disjoint root-interval buckets belonging only to gate/up occupy 12.890 seconds,
only down 6.734 seconds and only attention 11.334 seconds, out of a 42.311-second
root union. Mixed boundaries and overlapping families remain separate. The
same-family unions overlap and must not be summed. Host continuation time is
38.309 seconds in attention and 0.963 seconds in FFN; this attributes where the
CPU waits, not which queued GPU operation causes the wait. FFN is the leading
candidate from the unambiguous GPU buckets, rather than from API wait location.

Peak sampled process footprint was 26.386 GiB; minimum cache-aware physical
headroom was 19.854 GiB and swap growth was zero. The guard kept its 28.5 GiB
process cap, 10% reserve and 64 MiB swap-growth stop. The observer stopped its
owned sampler successfully and finalized all 291 valid samples. The analyzer
checks clock anchors, sample and raw-file digests, stop receipts, root-union
conservation and both bracketing controls; telemetry does not silently qualify
an out-of-frequency comparison.

The first profile attempt exposed a stale diagnostic count, rather than a GPU
failure: Q/K statistics now chunk F32 work at 16 MiB. Their self-prepare Graph
count is `7 + ceil(J/1024) + ceil(J/4096)`, which is 34 at J21120. The complete
36-block-plus-head count is 7925, not the old 7025. Native profile builders and
strict Python analyzers now derive this count from token geometry; legacy
reports without the new token-count field keep their legacy contract.

The connected FFN correctness falsifier has already rejected two forms at
J5760 on saved window 0, block 0. In the full graph, captured gate/up, SiLU and
product are bitwise equal to the baseline and upstream, while both down outputs
fail 6,585,038 elements at the unchanged tolerance. A separate down graph reading
a materialized BF16 product makes the captured path entirely bitwise exact,
but its independently compiled output-only front still fails 7,398,981 down
elements. Thus captured-stage parity does not qualify an output-only executable.
Neither rejected form was timed or promoted. A repeat with an output-only
product comparator localizes the second form's first observed mismatch before
down: 8,219,268 product elements fail. Compiler fusion or cast elimination is
an explanation to investigate, not an established root cause.

Making all four BF16 front outputs mandatory in the actual executable, followed
by the separate production down graph, passes every active intermediate
comparison bitwise against both saved upstream and the native baseline.
The independent captured executable also passes; neither executable may
substitute for the other's gate. This exact candidate was frozen for an
isolated J5760 timing falsifier. No production mode has changed.

The timing probe allocates one baseline and one candidate owner before encoding,
reuses the same input and weights, and encodes one chain with one terminal wait.
It runs 128 identical FFNs in each of ten samples ordered
`A_warm,B_warm,A,B,B,A,B,A,A,B`. Product/down checks follow each sample on GPU;
all 57 comparator slots are read once after the terminal wait. Four measured
AB/BA pairs require the same 2%/five-interior-samples/99%-active frequency gate.
Advancing to a full-block trial additionally requires at least 3% median
root-union improvement and a positive contrast in every pair. Warmups are
excluded. This is an admission rule for another experiment, not a production
speedup claim. The isolated guard is 8 GiB, 10%, 64 MiB swap growth and 600 s.
The candidate's named GPU buffers total 2,047,869,840 bytes, below 2 GiB.

The isolated batch completed with every active product/down comparison bitwise
equal to the saved upstream values. Gate/up controls and the injected-NaN
positive control also pass. Each sample has 128 FFNs; these are command-root
unions divided by 128, not exclusive kernel timings:

| Pair/order | Baseline ms/FFN | Candidate ms/FFN | Observed gain | Mean frequency difference |
|---|---:|---:|---:|---:|
| 1 / AB | 172.031 | 171.280 | +0.437% | 0.271% |
| 2 / BA | 174.455 | 170.573 | +2.225% | 1.337% |
| 3 / BA | 183.607 | 178.085 | +3.008% | 2.537% (rejected) |
| 4 / AB | 187.078 | 189.155 | -1.110% | 1.848% |

All samples have 7--9 valid interior observations and 100% mean sampled active
residency. The third pair fails the unchanged 2% frequency gate; the fourth
regresses. The raw four-pair median is 1.331%, below the 3% trial threshold,
and there is no qualified four-pair median. The front graph reduces explicit
Graph calls from three to two per FFN, but measured roots increase from 897 to
1153 per sample (+28.54%). Of 10,266 continuations, 10,246 are nested inside
Graph encoding and 20 are the asynchronous sample boundaries. This establishes
internal continuation activity at aggregate level; it does not identify the
particular executable responsible for each extra continuation. Fewer API calls
did not establish less GPU work. Do not advance this candidate to a full-block
trial or change production defaults.

There is one terminal wait, one packed post-wait comparator readback, zero
tracked encoding allocations and zero preterminal readbacks. Sampled peak
footprint is 2.6375 GiB, minimum cache-aware headroom 43.8500 GiB and swap
growth zero. The guard and owned observer both finish successfully. Sampled
limits do not rule out between-poll spikes or count opaque MPS allocations.

The CLI binds native report, launch, guard and binary digests, the actual
passing parity report and its source/fixture pins. It normalizes the correctness
probe's mixed relative/absolute source keys without accepting duplicate aliases
or paths outside the checkout. API accounting follows `mps_api_profile.h`:
Graph exclusive time subtracts child continuations; a continuation's nested
time is a subset of its own calls, not another child span to subtract.
The historical completion receipt does not bind the three observer files.
Their internal hashes/clock coverage are checked, but frequency labels remain
descriptive: a coherently replaced observer bundle is not excluded. The new
CLI therefore also refuses full-block admission without completion-time hashes
for `power/summary.json`, `power/samples.jsonl` and `power/powermetrics.txt`.
Partial or mismatched observer hash inventories are rejected. No receipt was
retroactively rewritten and no affirmative timing claim uses this gap.

The scoped outcome is exact isolated parity and rejection of this speed
candidate, not an acceleration or proof that all FFN optimization is exhausted.
Previously tested shared/packed gate/up, row tiling, graph orientation, classic
MPS and SiLU/product coalescing remain unpromoted; repeating their unchanged
forms adds no discriminating evidence. Sharing eight GQA K/V heads instead of
materializing 32 remains an untested attention hypothesis, but its byte-volume
estimate alone does not support a large full-request gain. The next useful
signal must identify avoidable GEMM, materialization or kernel work at unchanged
BF16 boundaries, before another candidate is encoded and timed. Full generation
remains outside this experiment. Evidence and exact replay commands are indexed
in `.landmark/evidence/2026-10-31-pixelumm-frequency-ffn.json`; refresh parity
after source/weights/fixture/geometry/runtime changes and timing after device,
queue policy or host-state changes.

## PixelUMM recovery A/A experiment (2026-10-31)

The matched after-DiT matrix did not isolate a causal prelude penalty because
time/order drift exceeded its mean contrast. The next bounded diagnostic uses
two fresh standalone PixelUMM processes, each with one identical warmup and
two complete decodes on the same GPU owners. The measured idle intervals are
1 and 120 seconds, in short-long and long-short order. Each process receives
120 seconds of recovery before launch. Every pass restores the same saved
latent, poisons the output and snapshots the full display on GPU; all exports
and hashes are deferred until its three requests have completed. No DiT,
video export or full generation is admitted.

Prediction: if a recoverable hardware state dominates the slowdown, longer
idle intervals should restore GPU frequency/power behavior and reduce request
latency in both orders. Continued drift after frequency recovery falsifies
that simple explanation; advisory nominal thermal pressure alone is not a
falsifier. The external observer records real powermetrics GPU telemetry,
without adding Metal synchronization. Background GPU use, opaque MPS state,
and observer overhead remain possible confounds.

The guarded `matrix-v2` completed both orders at source parent `fec31155` with
the same release binary and immutable source manifest. Observed wall times and
mean active GPU frequencies from the interior telemetry windows are:

| Order | Pass | Wall | Mean active GPU frequency |
|---|---|---:|---:|
| short-long | warmup | 145.102 s | 1356.3 MHz |
| short-long | short | 155.390 s | 1253.8 MHz |
| short-long | long | 145.518 s | 1355.7 MHz |
| long-short | warmup | 145.466 s | 1352.0 MHz |
| long-short | long | 145.703 s | 1352.0 MHz |
| long-short | short | 161.184 s | 1200.8 MHz |

Long-idle requests take 6.35% and 9.60% less request wall time than short-idle
requests in their respective cells. The short/long latency ratios are 1.06784
and 1.10625. All eligible request samples report 100% GPU active residency;
the short requests have lower frequency and power and more `Heavy` thermal
pressure. The exploratory product of request wall time and interior mean MHz
varies by less than 2% across all six passes. This supports frequency state as
an explanation for this matrix's modest drift; that product is not calibrated
GPU work, and the experiment does not establish a causal thermal root cause or
explain the earlier roughly twofold slowdown. There is only one pair per order,
with background load, opaque MPS state and observer overhead still possible
confounds. `powermetrics` thermal pressure and `NSProcessInfo.thermalState` are
different advisory APIs, not interchangeable categories.

Long-idle times agree within 0.13%; the two warmups agree within 0.26%. Summed
per-root command-buffer GPU durations are within 0.029--0.057 s of request wall time;
the existing terminal waits are 0.765--0.844 s. These spans include in-buffer
stalls, so they are not active compute time. Together with observed residency,
the results do not support CPU starvation as the dominant delay in this run.
The long pause is diagnostic: adding roughly 119 s of idle saves only 10--15 s
of request time and would worsen request throughput. The next falsifier is a GPU
FFN/attention cost reduction measured at matched observed frequency and warmup,
with the same output parity and queue policy.

A fresh analyzer invocation rehashed all six 460,062,720-byte display files;
each equals the authenticated source-MPS oracle SHA `5ed0974d...`. Every pass
has 11,731 segments, one terminal wait, zero tracked encode allocations,
readbacks or preterminal waits, and 42 stable decoder owners within its
process. This is stricter than unchanged `atol=rtol=2e-5`; it qualifies final
display values for this saved latent/noise only, not new intermediate or full
DiT trajectory parity. The sampled maximum process footprint was 21.154 GiB,
minimum cache-aware headroom 25.800 GiB, and maximum swap growth zero. Limits
remained 26 GiB process cap, 10% physical reserve with file-cache accounting,
64 MiB swap-growth stop and 1,200 seconds per process. Named buffers including
all three GPU snapshots total 18.654 GiB, below the 24 GiB named budget.

Actual short gaps are 1.004212 and 1.010257 s; long gaps are 120.010220 and
120.010411 s. The deadline loop and receipts use `CLOCK_MONOTONIC_RAW(4)`, the
installed Crystal `Time.instant` clock; clock 6 and UTC are advisory gap
diagnostics. Minimum deadlines and original 5/125-second upper bounds remain
strict. Earlier `matrix-v1` was excluded after an interval-bound rejection
before its long pass; its exact failed interval was not recorded, so its cause
remains unresolved. Fake-clock tests cover early wakeups, and a host-only
qualification measured 1.003024 and 120.003146 s with the final helper.

The external observer captured 714/714 valid two-second samples, then stopped
early deliberately after the matrix; its full 3,600-second cap was not needed.
Request and long-idle summaries exclude samples within one interval of either
boundary because headers have one-second UTC precision. The one-second idle
conditions have zero eligible interior samples and are not assigned frequency
or power estimates. Cleanup verifies the entire owned process group absent
before finalizing raw data; child-exited/group-present and permission-error
falsifiers pass. No observer/model processes remained after completion.

DoD executed: release build and plan-only admission; 4 replay lifecycle, 4
deadline, 20 matrix and 14 observer CPU checks; both guarded recovery orders;
and a fresh `pixelumm_recovery_matrix.py --analyze-only --output-root <matrix>`
run that reproduced the saved analysis exactly. All added code is diagnostic;
production generation is unchanged. Rollback is an atomic revert. Evidence:
`.landmark/evidence/2026-10-31-pixelumm-recovery-aa.json`; raw runs and alignment
script are under `/opt/mldata_ssd/cogni-ml/pixelumm-recovery-20261031`, telemetry
under `/opt/mldata_ssd/pixelumm-power-observer-20261031-matrix-02`. Changed
inputs, weights/source/oracle invalidate parity; changed hardware/OS/MPS,
source/binary, queue/recovery policy or host conditions require fresh timing.

## Matched PixelUMM lifecycle probe (2026-10-31)

The old 154.806-second decoder-only observation and 295.519-second post-DiT
observation used different terminal latents. They cannot establish a DiT
lifecycle penalty. `pixelumm_after_dit_probe` admits only the saved Cinematic
latent `ea79dc68...` and its authenticated 768x416 input/noise bundle. The
standalone arm and four-step BF16 Cinematic DiT-then-release arm restore those
same latent values with the same GPU copy before decoding.

Each fresh process constructs PixelUMM once and performs a cold request followed
by a warm request on the same plans, weights and scratch. A diagnostic-only
restart resets completed stem/assembly cursors after exactly one terminal wait;
it is not loaded by the production generator. Partial, failed, released,
unfinalized and undrained requests cannot restart. Both requests poison the
display buffer on GPU before rewriting it. Cold display values are copied to
a preallocated GPU snapshot; both display exports/hashes occur after the warm
request, with no CPU tensor diagnostics between requests. The cold timing
includes its snapshot copy; the warm timing does not. No raw capture is enabled.

DoD: release-build the probe; pass `spec_hy_nomas_decoder_replay` and the matrix
launcher's CPU falsifiers; complete guarded ABBA cells with zero tracked
encode allocations/readbacks/preterminal waits, one terminal wait per request,
stable decoder owner identities, and both full display outputs byte-identical
to the authenticated same-latent source-MPS oracle. This is stricter than the
unchanged `atol=rtol=2e-5`. Run under a 26 GiB sampled process cap, 10% physical
reserve including file-cache accounting, 64 MiB swap-growth stop and finite
deadline. Rollback is removal of the opt-in probe/helper and Makefile targets;
the production execution path is unchanged.

ABBA balances execution order but does not control thermal/frequency state.
Any effect is scoped to the operational DiT-prepare/compute/release prelude;
an allocation-only control would be needed to attribute it specifically to
DiT computation. Co-residency of both models is not admitted by this budget.
Full generation and new video export are excluded.

The guarded `matrix-v2` ABBA run completed with this release binary at source
parent `b57e8d79`; the binary/source manifest was checked around every cell.
Every cell used one cold pass as the same-owner warmup for its repeat, one
second between passes and 30 seconds between fresh processes. Observed times:

| Cell | Mode | Decoder setup | Cold wall / GPU span | Repeat wall / GPU span |
|---|---|---:|---:|---:|
| A1 | standalone | 22.555 s | 162.645 / 162.573 s | 209.900 / 209.879 s |
| B2 | DiT then release | 23.202 s | 184.269 / 184.217 s | 224.447 / 224.427 s |
| B3 | DiT then release | 23.035 s | 214.301 / 214.250 s | 287.647 / 287.625 s |
| A4 | standalone | 21.758 s | 224.316 / 224.258 s | 281.089 / 281.068 s |

The operational after-DiT/standalone mean ratios are 1.0300 cold and 1.0430
repeat. Cold adjacent pairs change sign (1.1330 and 0.9554); repeat pairs are
1.0693 and 1.0233. Standalone end/start ratios are 1.3792 cold and 1.3392 repeat.
Thus the earlier roughly twofold contrast was not reproduced on these matched
inputs. The 3--4% mean difference is descriptive, not a causal DiT penalty:
request/time/order drift is larger, thermal readings were unavailable, and
runtime-cache/frequency/other-load explanations remain open. Cold-versus-repeat
ratios also include the asymmetric snapshot copy and weight-conversion cache.

All eight full display outputs (115,015,680 F32 values each) have the pinned
source-MPS SHA `5ed0974d...`. A fresh CPU analyzer run rehashed the files and
reproduced the report exactly. All passes have 11,731 segments, zero tracked
encode allocations/readbacks/preterminal waits, exactly one terminal wait and
42 stable decoder owners within each cell. Terminal waits span 0.911--1.467 s;
the remaining wall interval is encode/submit, which includes Metal/MPS
backpressure. The GPU metric sums command-buffer start/end elapsed spans;
stalls inside a buffer are included, so this is not active compute time.
No new per-operation synchronization
or CPU tensor diagnostics were inserted into the measured path. B has its
separate existing terminal wait for the DiT prelude.

The sampled maximum process footprint was 24.0716 GiB, minimum cache-aware
headroom 8.8459 GiB and maximum swap growth zero; all four guards passed the
10% floor and 26 GiB cap. An earlier `matrix-v1` attempt was rejected before
model/GPU launch when startup headroom dipped below the same 32.4 GiB admission
requirement. No limits were weakened for the successful retry. Release build,
four replay-guard specs, five CPU analyzer falsifiers, format and diff checks
passed. Mini-Quad verdict: ROBUST for identity, display parity, diagnostic
replay lifecycle and sampled guard; causal timing attribution remains open.

Evidence index: `.landmark/evidence/2026-10-31-pixelumm-after-dit.json`.
Raw run: `/opt/mldata_ssd/cogni-ml/pixelumm-after-dit-20261031/matrix-v2`.
Next discriminator: fixed-input PixelUMM A/A with controlled recovery intervals
and observable frequency/thermal state, followed by an allocation-only DiT
prelude control if a residual prelude effect survives. This experiment neither
changes production defaults nor qualifies a speedup or full DiT trajectory.

## Cinematic LoRA quality probe (2026-10-31)

Two five-second 768x416 visual pairs separate decoder arithmetic from adapter
appearance. The first replays old and Q/K-corrected PixelUMM on identical saved
latent/noise. The second compares original BF16 DiT with the official Cinematic
adapter using the same input payload, seed, four DiT steps, one decoder step
and corrected decoder weights. The original BF16 default remains unchanged;
Cinematic is an explicit opt-in. Both clips and their source identities are
indexed in `.landmark/evidence/2026-10-31-hy-nomas-cinematic-quality.json`.

The offline cache binds the original index/source, official adapter revision,
config/payload, and actual derived BF16 index by hash. The original Q4 index's
source field denotes base ancestry, not a merged checkpoint. Only the 756
active double-block targets are merged; 162 action-module targets are excluded
because camera/action conditioning is disabled for this admitted request.
Packing uses original F32 base + F32 `(B @ A) * 2`, followed by one BF16 cast.
The 54 derived BF16 files contain 16,314,338,848 bytes; unchanged Q4/global
siblings are hardlinks. Merge completion fences and cache serialization occur
offline, never during inference encode.

All 54 blocks have been compared with the pinned upstream `merge()` on MPS:
8,157,072,384 BF16 words match exactly, including unchanged block tensors;
109 shared Q4/global assets are byte-identical. This qualifies cached weight
words, not a complete candidate DiT trajectory. The current verifier also
checks authenticated file identities around consumption and report publication;
four CPU mutation/symlink falsifiers cover that guard. The merge runtime is
Torch 2.8.0/MPS with CPU fallback disabled.

The candidate native request completed in 406.881 seconds, with a 63.859-second
DiT and 295.519-second decoder. This is an unpaired observation with saved Qwen
conditioning, excluding live text encoding and MP4 export; it proves no
speedup. The guarded subprocess took 408.615 seconds. Sampled process peak was
24.069 GiB, minimum cache-aware headroom 12.468 GiB, and swap growth was zero
under a 26 GiB process cap, 10% physical reserve, 64 MiB swap-growth stop and
1,200-second deadline. Both encode phases report zero tracked allocations,
activation readbacks and preterminal waits, with the existing one terminal
wait per phase. Counters cover instrumented MetalBuffer operations; opaque MPS
resources remain covered only by sampled process telemetry. Diagnostic exports
occur after the existing waits and do not feed the GPU latent handoff.

The candidate latent's complete decoder replay also matches exactly: raw and
display outputs each contain 115,015,680 finite F32 values, with zero error;
all 468 source/cache weight checks pass. The unchanged tolerance is
`atol=rtol=2e-5`. This oracle is the pinned Torch 2.6 gathered spatial-tile MPS
adapter, not stock CUDA/FlexAttention. It qualifies the decoder on this
candidate latent/noise, not the candidate DiT trajectory or video quality.

Sparse paired frames show small local changes from the decoder correction,
without a convincing eye fix. Cinematic changes composition and lighting much
more strongly; apparent fabric, vehicle and facial detail increase in the
inspected frames. The faces remain stylized. Full-motion eye naturalness,
spatial fidelity across prompts/seeds and multi-segment continuity are open;
no new user acceptance has been recorded. More diffusion steps, exact cache
weights or decoder parity do not establish perceptual improvement.

Opt in by adding both `--cinematic-cache /absolute/path/cinematic-lineage.json`
and `--cinematic-cache-sha <trusted-lineage-sha256>` to the existing BF16 preview
command. Omitting both preserves the original checkpoint. Admission rejects
incomplete identity pairs, wrong lineage, Q4/adapter combinations, enabled
camera/action conditioning, and cache inventory/precision mismatches before
Metal allocation. The focused Crystal admission suite has 22 passing examples;
the offline packer has 10 passing CPU tests. The release binary passed original
plan-only admission and the guarded candidate request above.

Risk remains CAUTION: an early BF16 cast, wrong target mapping or an unlabelled
partial adapter invalidates the comparison. Rollback is the unchanged default
index and an atomic revert of the opt-in integration. Refresh qualification on
source, weights, adapter, runtime/backend/device, geometry or input changes,
and on artifact loss. Before any later run, re-admit memory with the same guard.

## Opt-in spatial quality candidate (2026-10-30)

`hy_nomas_generate` now admits an explicit `--resolution 768x416` candidate
with conditioned v3 inputs. The default remains 512x288. Geometry is derived
consistently for the 30x13x24 DiT grid, 10x26x48 PixelUMM grid, local-attention
tiles, memory admission, patch/condition indexing and output assembly. The
patch/final token limit is shared with CPU preflight, so an unsupported extent
is rejected before constructing Metal owners. Weight-cache architecture and
payload identities remain separate from the runtime token extent.

The high-resolution stem on the initial input latent matches pinned upstream
BF16 output exactly on all 51,118,080 values. The separate five-window
synthetic merge/display control
matches all 115,015,680 F32 values exactly. Its injected-NaN negative control is
detected. This qualifies these boundaries, not all high-resolution DiT/decoder
intermediates. The native probe uses one terminal wait and no tracked encode
allocations, readbacks or preterminal waits.

The bounded four-step, five-second preview completed in 369.414 seconds per
request (371.021 seconds including the guarded launcher). Its sampled process
peak was 24.069 GiB, minimum cache-aware headroom 11.088 GiB, with no swap
growth under a 26 GiB cap and 10% physical reserve. DiT and decoder retain their
existing terminal waits and zero tracked encode allocations/readbacks. Opaque
MPS allocations are covered by sampled process telemetry, not those counters.

Selected frames look more detailed, but this is not a resolution-caused eye
fix: equal seeds with different noise tensor shapes produce different sampled
realizations and compositions. Temporal quality and full high-resolution
DiT parity remain open. The v3 bundle is explicit and cannot be accepted as
a default-size v1/v2 bundle. Evidence, frozen commands, media and refresh rules
are indexed in
`.landmark/evidence/2026-10-30-hy-nomas-spatial-geometry.json`.

The first full decoder comparison on the saved terminal latent failed the
unchanged `atol=rtol=2e-5`: raw maximum/mean error was 0.182129/0.00180326,
and display error was 0.0898438/0.000901566. The first 20 unblended frames also
failed, while all 468 source/cache weight checks passed. The historical
failure is indexed in
`.landmark/evidence/2026-10-30-hy-nomas-decoder-high-resolution-mismatch.json`.

Terminal-latent stem and first-window prefix replays localized that failure.
The stem is bitwise equal on all 51,118,080 BF16 values. The first two blocks
also match; block 2 attention originally had 17 tolerance failures, followed
by 530 block-output failures and 531,082 block 3 output failures. The prepared
Q/K diagnostic initially substituted a whole-token recomputation for the
pinned source's chunked feed. It now retains the actual source Q/K returns;
all 27 CPU reference tests pass, including a shape-sensitive regression.
The corrected reference preserves the original sequential block endpoints.

On identical raw projections, the first native divergence is the K statistics
reduction in the final 192-token chunk. There are 1,085 differing F32 words,
each one ULP apart (maximum 3.72529e-9). These become three differing BF16
prepared-K words, including two tolerance failures; the first is
token 12,352, KV head 6, channel 61. Replacing only that tail's statistics
with the source-shaped reduction makes all seven retained K/output stages
bitwise equal without changing the inverse, gain or RoPE implementation.

The production fix therefore matches the pinned `_prepare_qk` 16 MiB F32
input cap: Q statistics use 1,024-token chunks and K statistics use 4,096-token
chunks. Full chunks share compiled executables; immutable packed offset views
bind the existing GPU buffers before encode. The views remain owned through
chain drain. No staging activation buffer, per-encode buffer/view allocation,
CPU activation transfer or new explicit synchronization is introduced.
Multi-chunk plans require macOS 15 buffer aliases and reject older runtimes
before graph construction; single-chunk plans retain macOS 14 support.

On the final source, block 2's seven K/output captures and the first four
blocks' eight attention/output captures are bitwise equal in both the gate
and repeat, with production F32 FFN graph tensors. Each request uses one
terminal wait and zero tracked encode allocations or preterminal activation
reads. Their sampled peaks are 5,293,426,608 and 6,970,592,328 bytes under
separate 8 GiB process caps, with no swap growth and the 10% reserve.

At the default 512x288 geometry (J=5,760), a fresh current-helper synthetic
block 0 reference also matches all 26 retained stages bitwise in gate and
repeat. The regenerated inputs and all 13 weight records are authenticated
against the earlier fixture; its old captures are never used as truth.
The native sampled peak is 4,870,704,152 bytes under the same 8 GiB/10%/64 MiB
guards, with no swap growth, one terminal wait per request, and no tracked
encode allocations or preterminal reads. This is a synthetic single-block
geometry regression, not a real-conditioned default-size full decoder replay.

The complete decoder-only replay now also passes: all 36 blocks plus head,
five overlapping windows, shared scratch and the actual two output banks.
Both raw and finalized display outputs match the saved source-MPS control
bitwise over 115,015,680 F32 values each. The replay uses the same terminal
latent, BF16 weight cache and decoder noise as the failed baseline. It retains
one terminal wait, zero tracked encode allocations/readbacks/preterminal
waits, and the unchanged tolerance. Sampled process peak is 20.214 GiB,
minimum cache-aware headroom 17.255 GiB, with no swap growth under a 26 GiB
cap, 10% reserve, 64 MiB swap-growth stop and 1,200-second deadline.
Its 154.806-second decoder time is an unpaired observation, not a speedup.

This qualifies the saved 768x416 decoder replay against the pinned Torch 2.6
gathered local-tile MPS adapter, not stock CUDA/FlexAttention or video quality.
It does not qualify every intermediate of all 36 blocks, other real inputs or
devices. Refresh on source, weights, latent/noise, geometry, precision or
backend changes, or artifact loss. Build commands, before/after gates, full
output identities and guard records are indexed in
`.landmark/evidence/2026-10-30-hy-nomas-qk-chunk-statistics.json`.

The separate ten-second gaze control keeps 512x288 geometry and all seven
non-Qwen tensor identities unchanged. Its baseline repeat reproduces the
saved latent and pixel banks exactly. The treatment jointly requests fixed
forward gaze, minimal head motion, and no mutual glance or smile; it does not
isolate eye rendering. Both arms use four diffusion steps per five-second
segment, with one PixelUMM decode step. The user reports that the right-hand
treatment's eyes look noticeably more natural in motion. Sampled treatment
frames have steadier forward-facing posture, but inconsistent pupils and
smiles remain. The unchanged subject descriptions still mention smiles, so
retained smiles do not establish model
noncompliance with a consistent no-smile instruction. Camera/vehicle text is
unchanged, and numerical camera conditioning is inactive. This is not a
sideways-motion correction or a demonstrated visual fix. Both guarded arms
drain with zero swap growth and unchanged encode/readback counters. Media
identities are in
`.landmark/evidence/2026-10-30-hy-nomas-terminal-boundaries-and-gaze.json`;
the step counts and user assessment are also pinned in the Q/K evidence above.
Both gaze arms predate the Q/K statistics correction. The earlier
native/source-BF16 visual pair changed DiT only
and used the native decoder on both sides; a separate same-latent comparison
now varies the decoder itself.

## Echo quality: real-input arithmetic diagnostics (2026-10-30)

The same-scene text-cache correction below did not remove the reported drifting
eyes and uneven motion. The quality defect remains open. No production
precision default changes as a result of the following diagnostics.

The complete pinned source cache body now reproduces both native five-second
segments exactly: all 276,480 final latent values match, and segment 0 also
matches the prior single-segment control. This uses nine forwards through all
54 blocks, including the cache writer, with identical BF16 weights and native
F32/BF16 storage boundaries. The source adapter removes only the 19 false
text-padding K/V rows before chunked MPS SDPA. Keeping those masked rows
physically present changed the MPS reduction and already diverged in segment 0;
it was not evidence of a native continuation error. The diagnostic defaults to
the padded layout and exposes compact mode explicitly.

The guarded compact run used 23.715 GiB at peak, retained at least 19.816 GiB
cache-aware headroom, and caused no swap growth. It performs no activation
readback before its single terminal export. Separately, the new opt-in
60-latent standalone decoder replay reproduces all 106,168,320 saved native
display values exactly using the authenticated continuation noise prefix.
The default standalone decode remains 30 latents.

These checks qualify native/source-MPS final-latent agreement and native
decoder replay, not CUDA/Magi parity or a visual fix. The user sees only a
small improvement from source BF16; the principal eye defect persists.
Background-flow measurements correlate the slowdown with five-second
boundaries, without establishing its cause. Evidence, controls, commands and
refresh conditions are in
`.landmark/evidence/2026-10-30-hy-nomas-two-segment-control.json`.

An authenticated Echo first-step capture now records patch, the complete
1,280-row text refiner, and dual-stream block 0 on one native command chain.
It uses original resident BF16 weights with the existing F32 activation path,
557 valid text rows, and one terminal wait, with no tracked encode allocations
or activation readbacks. The pinned original Python module bodies run as an
external Torch-MPS reference; their chunked SDPA adapter is not stock
CUDA/Magi attention parity.

The F32 reference reproduces all 54 patch/refiner checkpoints bit for bit.
The first block's text attention projection has 17 values outside the unchanged
`atol=rtol=2e-5`, but its final image and text outputs pass that tolerance.
Attention replay from identical captured Q/K/V has zero failures over
10,027,008 elements. These results qualify this first-step arithmetic control,
not later-layer, diffusion, continuation, or perceptual parity.

The original BF16 activation reference differs from the native F32 path:
the patch output has maximum absolute difference 0.001952708, and the final
text-refiner output has RMS difference 0.05182612. This is a precision-mode
difference, not evidence that it causes the visible defect. The separate block
BF16 comparison uses rounded native prelude inputs; it is not a complete
BF16-prelude-to-block replay.

A one-row, 2,048-to-12,288 modulation-linear falsifier exposed an issue in the
initial diagnostic: casts to BF16 and back within one graph did not establish
physical rounding. A caller-owned BF16 graph output followed by a GPU
expansion produces 12,288 outputs bit-identical to Torch-MPS, with all expanded
F32 low 16 bits zero. Both the BF16 graph and explicit F32-accumulation variants
pass on this case. Its input is already BF16-aligned, so arbitrary F32 input
casts remain unqualified. This helper is diagnostic-only.

The refreshed reference report is
`/opt/mldata_ssd/cogni-ml/hy-nomas-real-attention-20261030/final-controls-v2/full-oracle/report.json`
(SHA256 `a8f1f5ba58b73ff998fbe1c9a65aecac24bd1c2572792e8514ee87166e9eb9f7`).
Physical-store linear controls are the adjacent `bf16-graph-physical-store` and
`f32-accum-physical-store` directories. Their bounded runs retain the 10%
cache-aware reserve, finite process limits, and 64 MiB swap-growth stop.
Source, dtype, input, shape, or backend changes require refreshing these
claims.

The native physical-BF16 patch output now matches the original Torch-MPS
PatchEmbed on all 8,847,360 values, bit for bit, for both linear variants.
Its input is the authenticated, already BF16-aligned first Echo latent;
this is not a qualification of arbitrary F32 input conversion or full DiT.
The patch comparison evidence is under
`/opt/mldata_ssd/cogni-ml/hy-nomas-real-attention-20261030/bf16-patch-remap-v3`.

For the authenticated writer0/reader1 block0 inputs, native F32/MPS Q/K/V
match the pinned upstream block exactly. Saved writer history K/V and direct
reader history-attention output also match the chunked Torch-MPS adapter
exactly. This tests the separate history-attention plan on real QKV/history;
the oracle stops at attention dispatch. It does not compare a history-attached
block output, later layers, BF16 operation boundaries, or video quality.
One native chain retains zero tracked encode allocations/readbacks and one
terminal wait. The boundary report is
`/opt/mldata_ssd/cogni-ml/hy-nomas-real-boundary-20261030/oracle/report.json`
(SHA256 `0bb84bfeb8c40c91c124d9cc15697a0b20effd5e7c3dcdbab5eda37fcc4ea52c`).
The same-lineage adversary verdict is ROBUST for this narrow arithmetic claim,
VULNERABLE for complete block/boundary parity or a quality fix.

A complete first-segment F32 model-body replay now matches the saved native
final latent bit for bit: all 138,240 values after 54 blocks and four Euler
steps. The reference uses exact BF16 checkpoint values expanded to F32 and
reproduces the native initial-latent, velocity and per-step BF16 rounding
boundaries. One reusable F32 block receives device-to-device casts from the
54 resident BF16 source blocks. This qualifies the final latent for this
input, not every intermediate value or a continuation boundary. The sampled
peak is 18.566 GiB, with 22.583 GiB minimum cache-aware headroom and no swap
growth. The external 26 GiB cap, 10% reserve and 64 MiB swap-growth stop remain
in force.

The matched original-BF16 model-body control also completed. Repeating it
after adding the F32 mode reproduces its terminal payload exactly. Its latent
differs from the native F32 path by RMS 0.500732 and maximum 4.385742; this is
not a tolerance pass or a quality score. Both latents were decoded with the
same native PixelUMM, noise and five overlapping windows. The standalone
native control reproduces the saved first 100 display frames exactly;
later frames of this five-second decode lack the ten-second control's future
lookahead and are not used for that equality claim. Both decode runs retain
zero tracked encode allocations, readbacks and preterminal waits, with one
terminal wait. Selected matched frames show unstable pupils in both precision
arms. No eye-quality fix is established.

The synchronized comparison is
`/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-quality-20261030/comparison-v1/native-f32-vs-source-bf16-5s.mp4`.
The full F32 arithmetic comparison is the adjacent
`native-f32-control-v1/native-latent-comparison.json`; the BF16 repeat is
`source-bf16-repeat-v1/completion.json`. The frozen control script SHA256 is
`e711f50dcde9e037b04334938eb9d0a1ad899381f2dd74069bc88d858379356d`.
These controls use original pinned source bodies with the disclosed MPS SDPA
adapter, not unmodified CUDA/Magi execution. No production precision default
is changed.

The 512x288 output has a 9x16 latent grid, compared with the published
1280x704 recipe's 22x40 grid; this is a recipe difference, not proof that the
current shape is outside the training distribution. The two-segment arithmetic
control above now closes the tested native/source-MPS final-latent boundary.
Eye-quality work continues with a matched conditioning intervention that
reduces gaze, head and smile movement, while the higher-resolution decoder
mismatch is localized separately.

## Same-scene continuation text cache (2026-10-30)

The experimental shared-scene sequence entry point now uses upstream's
`cache_txt` contract for chunks of one semantic scene. Its first writer stores
video and text; later denoise forwards use the cached scene text and only the
current video K/V. Later writers attend current text transiently but append
only current video rows. All current text queries remain active. For the Echo
fixture (4,320 video and 557 valid text tokens), four-chunk denoise key counts
are 4,877 / 9,197 / 13,517 / 17,837 rather than repeating another 557 text keys
per chunk. Padded text rows remain excluded.

The lower-level provider keeps its legacy `PerChunk` default. `OncePerScene`
requires the caller to preserve one scene's conditioning; the sequence entry
point enforces one JSON scene with no action tails and reuses one encoded
prompt. `--legacy-text-per-chunk` restores the old experimental sequence rule
for a controlled comparison. No weights, precision, solver or tolerances change.
Both denoise and writer plans are prepared before encode; admission includes
the distinct writer workspaces, which are released with the provider.

A three-chunk GPU probe exercises four denoise forwards per chunk and two
writers across all 54 layer caches, with distinct forward inputs and poisoned
padding. All 9,072 attention output values and 2,160 final cache values match
the pinned upstream function body and actual dense-cache helper exactly.
The reference uses F32 Torch SDPA on MPS; it does not establish stock
BF16/Flash/Magi or full DiT parity. The native probe has zero tracked encode
allocations, readbacks or preterminal waits, and one terminal wait. The
separate post-terminal F64 formula differs by at most 7.451e-9. Eighteen
Crystal geometry/admission examples and four CPU oracle tests pass at unchanged
`atol=rtol=2e-5`.

This corrects a source-confirmed cache semantic difference. It does not by
itself prove that the reported uneven motion or drifting eyes are fixed.
The matched 10-second BF16 preview preserves all first 30 latent frames exactly,
changes the second chunk, and leaves the unused latent-bank suffix untouched.
The first 100 display frames also match the saved control exactly; decoder
lookahead then includes changed later latents. All 240 frames are finite and
nonconstant. The request takes 376.07 seconds, including setup and raw export;
this is one bounded run, not a matched performance comparison. The sampled
peak is 24.792 GiB with 17.875 GiB minimum cache-aware headroom and no swap
growth under a 28.5 GiB process cap and 10% reserve.

The motion symptom remains open: top-background median pixel flow after/before
the first seam is 0.472 in the saved control and 0.486 after the correction;
the farther-right ROI also retains slowing. These are single-clip screen-space
diagnostics, not a quality score, physical camera speed or eye tracking.
The side-by-side preview is saved at
`/opt/mldata_ssd/cogni-ml/hy-nomas-cache-fix-20261030/preview-10s/echo-before-after-10s.mp4`.
The next discriminator is real chunk-two neural parity on the same prefix,
including the separate native-F32/upstream-BF16 history precision boundary.
The prior decoder seam result also remains limited to its selected display
frames. Evidence and refresh boundaries are recorded in
`.landmark/evidence/2026-10-30-hy-nomas-scene-text-cache.json`.

## Active quality and duration experiment (2026-10-30)

Historical experiment snapshot, before the same-scene cache correction above.
Structural export: VERIFIED within the bounded native execution and file
integrity scope. Semantic motion quality: VULNERABLE. Full upstream video and
continuation parity are not established; the default remains unchanged.

Native PixelUMM and the pinned upstream BF16 equations produce bit-identical
merged raw pixels on the same generated latent and authenticated decoder
noise against upstream revision
`911cea6458a89c2663513ff7099561ec190dd491`: maximum pixel error zero, finite
outputs, and 468 source-weight checks.
The comparison uses the gathered spatial-tile F32 QK/softmax/PV adapter on
MPS; it does not establish unmodified CUDA/FlexAttention parity. Tolerances
remain `atol=rtol=2e-5`.

The matched four-step BF16 prompt A/B uses identical initial noise, seed,
geometry, decoder and solver, changing only simple versus structured text
conditioning. The simple arm reproduces its saved control exactly. Selected
structured-prompt stills show clearer rails and street/building clearance;
this is not a full-motion quality verdict. Both 120-frame MP4s and a
synchronized comparison are saved. Conditioning authenticates the pinned
source/template/mask and saved upstream Qwen state26; this does not measure
native live text encoding.

A preceding guarded two-segment DiT qualification showed the first thirty
latent frames bit-identical to the frozen five-second control, a changed next
segment, and an unchanged unused bank suffix. It qualified prefix preservation
for that fixture, not upstream continuation or the new scene's visual quality.

The 20-second run contains four linked five-second AR segments and 480 frames
at 24 fps. It completed in 742.0248 seconds, or 37.1012 seconds per video
second. The global decoder emitted 23 windows, starting at latent frames
0,5,...,110. All frames were finite and nonconstant; each segment changed
from its initial noise and the unused latent-bank suffix stayed unchanged.
The guard sampled a 33.676 GiB peak, 11.782 GiB minimum cache-aware headroom,
and 9.31 MiB swap growth. Limits were a 10% reserve, 35 GiB footprint,
64 MiB swap growth, and 1200 seconds. The run passed. These are sampled
resource observations; they do not bound between-sample spikes or prove
perceptual quality. Instrumented DiT and decoder encode paths recorded zero
allocations, readbacks, or preterminal waits, and one terminal wait per phase.
Timing includes preflight, setup, inference and raw export; it excludes saved
conditioning preparation and MP4 encoding.

The shared prompt is one authenticated scene description with
`contained_actions: []`, 342 valid text tokens, and continuous motion in
`visual_description_long`. Pinned upstream's no-action branch has no action
offsets or tail masks; this supplies shared scene conditioning without
per-segment directorial control or a guarantee that described events occur at
specified times. Selected frames showed the tram near the camera around
7.5–12.5 seconds and far away around 17.5–19.958 seconds. The user's full-motion
review identifies the same tram exiting beyond the frame, then returning
backward along its previous track/path to the scene middle, not as another tram
on the nearer opposite-direction tracks. The requested trajectory is
VULNERABLE; the reversal's cause is unresolved among model, prompt, and native
continuation. This is a human full-motion observation, not framewise tracking
or a perceptual metric.

The continuation keeps layer-specific F32 K/V, valid prior video/text rows,
global temporal RoPE, and the pinned separate noisy cache-writer forward after
four read-only denoise forwards (`sigma=0.206897`, `t=206.897`). Upstream
history is BF16, while native history is F32. The initial, cache-noise, and
pixel-noise banks are authenticated MPS-generated inputs, not stock upstream
CPU-seed RNG values. Therefore the run is not a full upstream continuation
reproduction. The next discriminator is a source-faithful upstream segment-2
run with the same shared prefix, comparing cache state, intermediates, and raw
output before attributing visual differences to the model.

The frozen run identity is parent HEAD `0757d5682be0c521927b035de3fb6cbcc07ea17e`,
binary SHA256 `72d429dbafbe21cf7055b880d4e2f75a6eceeba987fe53ba3608b01e88e96a37`,
and 45 authenticated modified-source files. The post-run mechanical format
pass changed five files; `formatting-lineage.json` verifies that each equals
`crystal tool format` applied to its frozen source, with the other forty files
unchanged. A separate release rebuild, 71 Crystal examples, 32 Python tests
(one skipped), and the format check pass. This does not imply a model rerun.
Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-quality-20261030/results/sequence-twenty`
(`validation.json`, `identity.json`, and `correlated-adversary.json`). The
evidence packet is saved at
`.landmark/evidence/2026-10-30-hy-nomas-quality-continuation.json`.

## Matched BF16 64/96/128-step previews (2026-10-30)

All three requested five-second previews completed at 512x288, 24 fps with
the same authenticated prompt, seed42, initial latent, saved Qwen state26,
decoder noise, BF16 path, shift5, CFG1 and one-step PixelUMM. The saved32
control is authenticated and reused, not rerun. The user previously reported
more detail at24/32, an apparent approach to the right-hand building below16
and weak or absent rails. The prompt was deliberately unchanged and does not
explicitly request visible rails. Multiples of ten receive no special
treatment in the generic Euler solver. These larger counts are experimental
beyond the distilled checkpoint's documented4/6/8 settings.

| DiT steps | DiT wall, s | Decoder wall, s | Guarded generator, s | Seconds per video second |
| ---: | ---: | ---: | ---: | ---: |
| 32 (saved control) | 186.651 | 88.432 | 323.692 | 64.738 |
| 64 | 309.529 | 69.615 | 425.846 | 85.169 |
| 96 | 621.893 | 75.483 | 746.990 | 149.398 |
| 128 | 762.612 | 80.512 | 889.429 | 177.886 |

New counts ran serially in ascending order, one fresh process per count,
without concurrent models or Studio. File/Metal caches, thermal state and
host load were uncontrolled; guard/metadata inspection continued during the
runs. Timings include admission, weight setup, inference and raw export,
excluding MP4 encoding and live text encoding. Saved32 belongs to another
cohort. Unchanged decoder time also drifts; these are descriptive observations,
not causal speed ratios, isolated per-step costs or longer-clip estimates.
An early status-based inference that128 started faster than96 was corrected:
exact step1-to19 encode intervals are89.979s versus84.752s. Submission markers
are not per-step GPU completion fences. See timing-analysis.json.

Both contact sheets were inspected at frames0/24/48/72/96/119. Composition,
tram proportions, signage and window/door shapes change with the schedule.
Some96/128 edges look sharper than64, but neither consistently preserves all
details better than saved32. Warped windows, people and invented lettering
persist. No clear continuous rail pair is established in the sampled visible
street portions, with much foreground occluded by the tram. This does not
prove rail absence across all frames, motion quality or collision behavior.
The synchronized comparison is supplied for full-motion viewing. No default
promotion or assumption that200 steps will fix geometry follows; keep8.

The BF16 scheduler and shared full-pass cursor now admit1..128. Default8 and
Q4's separate cap8 remain. Zero/129 are rejected before Metal setup; the
cursor rejects pass129, ordering violations and reuse without exactly one
terminal wait on the preceding chain. Schedule storage grows768B from32 to128.
Full-K math and GPU scratch/handoffs remain. Rollback omits --steps; the
eight-step default is retained.

Fresh96-step upstream qualification exposed a real F32 schedule discrepancy:
separate CPU linspace multiplication/addition rounds sigma positions26/29
one ULP below the pinned source, and timestep26 differs by6.1035156e-5,
exceeding the existing2e-5 absolute tolerance. Explicit Math.fma in the first
half of the two-ended linspace reproduces the pinned CPU contraction. The
new exact regression fails before that change and passes after it. This is
schedule setup only, with no GPU kernel, arithmetic precision or tolerance
change. Default8 and saved32 schedules remain exact. A CPU fmaf candidate
matches the pinned solver at all counts1..128; actual native GPU qualification
covers8/32/64/96/128, not every count in that interval.

All five fresh MPS Euler controls match sigma, dt, timestep, initial BF16
rounding and every intermediate updated latent exactly at unchanged2e-5
absolute tolerance. Generated step events also match upstream timestep/dt
F32 bits. This qualifies the solver and tested boundaries; full upstream
neural-video parity remains false in every receipt.

All three guards finish normally and drain their process groups:3,950 samples,
20.331006GiB sampled peak,16.498596GiB minimum cache-aware headroom and zero
sampled swap growth. Keep10% reserve,23GiB footprint cap and64MiB swap-growth
stop. The timeout is1800s per request for this bounded batch, versus900s in
the preceding experiment. Each request records one terminal wait per weight
phase and zero instrumented encode allocations/readbacks/early waits. Opaque
MPS resources and terminal pixel export are outside these counters; sampled
memory does not bound between-poll peaks. DiT observed roots are65792/98688/
131584 and decoder roots5601 each, metadata rather than an exact per-step-work
invariant. All120 frames per arm are finite and nonconstant, integrity checks
rather than perceptual quality scores.

Evidence root: /opt/mldata_ssd/cogni-ml/hy-nomas-steps-64-96-128-20261030.
It contains three videos, comparison-32-64-96-128.mp4, two contact sheets,
viewing-notes.md, summary/final-check.json, build identity, fixtures, receipts,
guards and retained fail-before evidence. Parentcccc3c87; frozen source/binary
hashes match after generation. The107-file evidence-index.json plus four
authenticated saved32 references has SHA256
c844c6430c7d4ae6f01f5c2df04ccaf55742a457806c0e6b22214e5b02cf64f4.
DoD passes: release builds,32 focused Crystal examples, nine compiled negative
cases, five exact GPU Euler controls, three guarded120-frame exports,
validate_matrix.py and Crystal formatting checks. The one-shot DoD command
`python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-steps-64-96-128-20261030/validate_matrix.py`
exited0 and created final-check.json; it refuses to overwrite that evidence.
Same-lineage Mini-Quad is
ROBUST for this bounded evidence, not independent replication or a monotonic
quality claim. Refresh on scoped source/input/artifact, precision, geometry,
backend, device, toolchain or guard changes; timing also decays with load,
cache, order and thermal state.

Next artifact falsifier: hold generated DiT latent and decoder noise fixed and
compare native/upstream one-step PixelUMM merged raw pixels before display
conversion and MP4. A separate numerical hypothesis is that per-step BF16
rounding can remove very small updates at large step counts: both native
Euler and the upstream pipeline round each updated latent. No lost-update
rate or artifact causality was measured. A test-only compact GPU counter on
fixed sampled coordinates, read after the existing terminal wait, could
measure that mechanism without new per-step synchronization.

## Matched BF16 8/16/24/32-step previews (2026-10-30)

All four requested five-second previews completed at 512x288, 24 fps. They
share the preceding experiment's authenticated prompt, seed42, initial latent,
saved Qwen state26 conditioning, decoder noise, shift5 and CFG1. PixelUMM
keeps one step and five overlapping J5760 windows. The new eight-step raw
pixels and MP4 reproduce the previous eight-step arm byte-for-byte. Only the
DiT schedule length changes. The generic upstream Euler solver supports these
lengths, but the distilled checkpoint documents4/6/8, not16/24/32; the50-step
SFT configuration uses another checkpoint. Larger counts remain experimental.

| DiT steps | DiT wall, s | Decoder wall, s | Guarded generator, s | Seconds per video second |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 35.078 | 65.468 | 144.123 | 28.825 |
| 16 | 72.899 | 75.074 | 198.684 | 39.737 |
| 24 | 128.538 | 78.841 | 255.249 | 51.050 |
| 32 | 186.651 | 88.432 | 323.692 | 64.738 |

These are one fresh process per count in ascending order, with uncontrolled
file/Metal caches and load. Decoder time also drifts despite its unchanged
settings. Guarded elapsed includes admission, weight setup, inference and raw
export; MP4 encoding and live text encoding are excluded. The timings are
descriptive, not an isolated per-step cost, causal speed ratio or bound for
longer clips.

Both contact sheets were visually inspected at frames0/24/48/72/96/119.
Twenty-four and thirty-two look broadly similar, sometimes with clearer tram
edges or window detail than eight. Neither consistently wins; distorted
people, irregular windows and invented lettering remain. Framing and content
also change with the schedule. The sampled stills do not establish motion
quality, monotonic improvement or the cause of the artifacts. Keep default8;
there is no supported quality promotion from this one prompt/seed.

The BF16 scheduler and loader now admit1..32 before Metal setup; the default
stays8 independently of the cap. Q4 retains its own eight-step limit. The
full-pass cursor rejects pass33, ordering violations and a new chain without
exactly one terminal wait on the preceding chain. The schedule grows192B
from8 to32; opaque MPS resources are outside this named-byte calculation.
Optional API/root tracing stays off, with its8192-event cap insufficient here.
Rollback omits --steps; default precision, arithmetic and step policy remain.

Fresh pinned-source MPS Euler fixtures for all four counts match sigma, dt,
timestep, initial BF16 rounding and every intermediate updated latent exactly
at unchanged2e-5 tolerances. Every generated step event also matches its
upstream F32 timestep/dt. This qualifies the solver and tested boundaries,
not full upstream neural-video parity, which remains false in each receipt.

All four guards finish normally and drain their process groups:1,772 samples,
19.966656GiB sampled peak,14.811356GiB minimum cache-aware headroom and zero
sampled swap growth. The10% reserve,23GiB process cap,64MiB swap-growth stop
and900s timeout remain. Each request has one terminal wait per weight phase,
with zero instrumented encode allocations, readbacks or early waits. Opaque
MPS resources and terminal pixel export are outside these counters; sampled
memory does not bound between-poll peaks. All120 frames/count are finite and
nonconstant, an integrity check rather than a perceptual quality score.

Observed DiT root counts are8224/16448/24672/32895. The last is one below the
empirical1028*steps extrapolation. Source inspection shows this tracker counts
observed rootCommandBuffer identities around opaque MPS calls, with NSUInteger
storage and an Int32 boundary, not a fixed number of model operations. All32
ordered step events and both completed phases are present. The extrapolation
is rejected as an acceptance invariant; no runtime or tolerance was changed.
The exact segmentation difference or a missed tracker observation remains
unresolved. See root-segmentation-audit.json; this is not proof of full neural
parity or of a particular MPS cause.

Evidence: /opt/mldata_ssd/cogni-ml/hy-nomas-steps-8-16-24-32-20261030 contains
the four videos, synchronized comparison-8-16-24-32.mp4, contact sheets,
summary.json, fixed-input receipts, guards, build-identity.json, fresh Euler
fixtures, final-check.json and the retained failed root-count extrapolation.
Parent617c81c9; measured source/binary hashes remain unchanged. The70-file
evidence-index.json SHA256 is
6932a7bcbc61843c5478da6ee23c289342771cdf800908b6ca4fe78139101ed2.
DoD passes: release builds,31 focused Crystal examples, eight compiled
negative cases, four fresh GPU Euler controls, four guarded120-frame exports
and offline source/input/media/hash/guard checks. Same-lineage Mini-Quad is
ROBUST for those bounded observations and VULNERABLE for monotonic quality
or32-over24 superiority claims. Refresh on scoped source/input/artifact,
precision, geometry, backend, device or toolchain changes; timing also decays
with load, cache and order. Next artifact falsifier: hold a generated DiT
latent and decoder noise fixed and compare native/upstream one-step PixelUMM
merged raw pixels before display conversion and MP4 export.

## Matched BF16 2/4/6/8-step previews (2026-10-30)

The four authorized five-second previews completed at 512x288, 24 fps with
identical authenticated input bytes, seed 42, saved Qwen state-26 conditioning,
decoder noise, shift 5 and CFG 1. Each starts from the original latent with
its own complete Euler schedule. Only the DiT step count changes; PixelUMM
keeps one step and the same five overlapping windows. The eight-step raw
pixels and MP4 are byte-identical to the previous BF16 preview.

| DiT steps | DiT wall, s | Decoder wall, s | Guarded generator, s | Seconds per video second |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 12.755 | 89.889 | 150.543 | 30.109 |
| 4 | 25.037 | 81.802 | 153.912 | 30.782 |
| 6 | 32.121 | 73.980 | 155.626 | 31.125 |
| 8 | 35.797 | 74.166 | 156.632 | 31.326 |

These are single samples in launch order 8/2/4/6, with uncontrolled file and
Metal caches and substantial decoder drift. Guarded elapsed includes input
authentication, weight setup, inference and raw export; MP4 encoding and live
Qwen text encoding are excluded. This is a quality experiment with descriptive
timings, not a causal whole-request speed comparison or a longer-video bound.

Both contact sheets were visually inspected at frames 0/24/48/72/96/119. Two
steps produce strong smearing and repeated edges. Four gives much clearer
tram/building detail. Six and eight are sharper but retain distorted people,
invented lettering and irregular geometry. Changing the schedule also changes
framing and content, so sharpness alone cannot rank fidelity. These sampled
stills do not establish motion smoothness or monotonic quality improvement.
Upstream documents distilled 4/6/8; two is an experimental quality setting.

The new `--steps N` defaults to eight and, in this measured build, admits 1..8
before Metal setup. A failing six-step regression exposed a one-ULP difference
from upstream CPU `torch.linspace`: direct rational division changed a model
timestep beyond the unchanged 2e-5 tolerance. Matching its F32 step and
two-ended indexing fixes this. Fresh pinned-source MPS Euler fixtures for
2/4/6/8 match CPU sigma/dt/timestep, GPU timestep, initial BF16 boundary and
every updated latent exactly. This qualifies the solver and tested BF16
boundaries, not full native-versus-upstream neural-video parity, which remains
false in each receipt. Legacy eight-step fixtures still pass.

All four guards finish normally: 1,187 samples, 19.874432 GiB sampled peak,
17.087662 GiB minimum cache-aware headroom and zero sampled swap growth. The
23 GiB process cap, 10% reserve, 64 MiB swap-growth stop and 900-second timeout
remain. Every request retains one terminal wait per weight phase and zero
instrumented encode allocations, readbacks or preterminal waits. These
counters exclude opaque MPS resources and terminal pixel-file export; sampled
memory is not a bound on between-poll peaks. All 120 frames per arm are finite
and nonconstant; that is an integrity check, not a quality score.

Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-step-comparison-20261030` contains
`summary.json`, per-arm commands/receipts/guards/MP4s, the synchronized
`comparison-2-4-6-8.mp4`, two contact sheets, `build-identity.json`, upstream
Euler fixtures/logs and CLI/oracle negative controls. Parent source is
`9c6d4271`; the measured source and binary hashes identify this experiment.
DoD: release builds, 13 focused Crystal examples, four fresh GPU Euler
fixtures plus the legacy control, early malformed-input rejections, four
guarded generations and 120-frame MP4 probes pass. Same-lineage Mini-Quad is
ROBUST for these bounded observations; the cause of remaining artifacts is
unresolved. Refresh on scoped source, input, artifact, precision, geometry,
backend, device or toolchain changes; timing also decays with load/cache/order.

The next artifact discriminator is to hold one generated DiT latent and its
decoder noise fixed and compare native/upstream one-step PixelUMM merged raw
pixels before display conversion and MP4 export. It does not require another
DiT run. The user subsequently requested 8/16/24/32; larger schedules require
fresh admission and upstream checks and do not promote a quality default.

## Aligned release/GC preserves parity but leaves variable diagnostic-placement effects (2026-10-29)

The opt-in lifecycle follow-up completed both 36-block-plus-head schedules at
J=21,120 with the same source/binary/resources and 60-second pacing. Native
chain metadata and condition snapshots are saved before release; explicit GC
then precedes eager/standard CPU comparison or deferred return. This leaves
ordinary inference ordering and all numerical graphs unchanged.

Following-probe totals after eager/deferred diagnostics are 44.535279/43.250263 s
in normal order and 58.362926/43.399179 s in reverse. Eager-context requests are
2.9711%/34.4793% longer in these two descriptive contrasts. The reverse tail
is also slower at 47.391884 s. Common cleanup did not remove the observed
variation; do not claim a stable speedup, isolated release/GC benefit, pure
CPU-comparator cause or specific Metal driver wait. Recomputed root-interval
unions differ too, but they are not active shader cycles. The earlier
50-second results cannot be pooled with this changed schedule.

All 516 retained comparisons remain finite and bitwise equal to authenticated
saved upstream controls at unchanged atol=rtol=2e-5. Every request retains one
wait, 43 post-wait reads, zero early reads/tracked encode allocations and
4,579 error-free completed roots. Both guards finish normally: 2,667 samples,
26.201907 GiB sampled peak, 13.673111 GiB minimum cache-aware headroom and no
sampled swap increase over admission baselines, with the 10% reserve/27.5 GiB
cap unchanged. No generation ran.
The first 50-second aligned attempt safely rejected a 51.775842 s diagnostic
overrun before the following chain was created; it is excluded from inference.

The release build, 77 CPU examples, seven matched early CLI rejections and
offline mutation/lineage/lifecycle checks pass. Source parent is `af34cece`,
with all 202 measured source hashes unchanged across both processes. Evidence:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cpu-lifecycle60-20261029`;
paired-analysis SHA256
`4589940af59d8eb7b179e2de776e6b22488bce79b5c1486d9d468b8d4e4eb0b0`.
See [the conditional timing table and safeguards](pixelumm-cpu-diagnostic-placement.md).

Next proposed performance falsifier: batch independent attention windows at
B2 while preserving per-window Q/K geometry and reduction dimensions. First
check raw QK, PV with baseline P and connected upstream parity, then the full
block/decoder. Additional named scratch is approximately 444 MiB, with fresh
memory admission still required. Do not revive rejected query-row concatenation
or infer a latency saving merely from fewer API calls. This candidate has not
been implemented or timed.

## CPU diagnostic placement changes the following request in both sampled orders (2026-10-29)

The bounded 36-block-plus-head policy experiment is complete at J=21,120.
After eager/deferred diagnostic placement the following requests take
63.953500/48.236443 s in normal order and 48.303293/43.263825 s in reverse.
These are two descriptive contrasts, not a stable production speed estimate.
GPU graphs, persistent buffers, input A and 50-second pacing targets match.
Completed-chain release and post-request GC move with the diagnostics, so pure
CPU-comparison causality remains open. No inference default or math changed.

All 516 retained saved-upstream comparisons are finite and bitwise equal at
unchanged 2e-5 tolerances. All 12 requests preserve one terminal wait, 43
post-wait reads and zero early reads/tracked encode allocations. The 10%
cache-aware guard passes both complete processes with zero swap growth;
sampled peak footprint is 26.195773 GiB, minimum headroom 16.343918 GiB.
No diffusion or generation ran. See
[the diagnostic design, timing table and scoped evidence](pixelumm-cpu-diagnostic-placement.md).
The aligned follow-up above completes the proposed common release/GC
discriminator. Do not infer a shader acceleration or driver-wait cause from
either placement contrast.

## Gate/up row tiling does not qualify as acceleration (2026-10-29)

Status: VERIFIED for bounded saved-output parity, lifecycle/guard checks and
failure to meet the declared speed-admission criterion. No stable speedup,
slowdown, equivalence or production throughput claim is admitted. The ordinary
MPS path remains the default. No diffusion or full generation ran.

The preceding matched-pause profile identifies gate/up as a candidate family
(13.3822 s of overlapping command intervals), not an exclusive shader cost.
The new diagnostic partitions only M: separate gate/up graphs retain the full
K=4096 F32 reduction, cast each result to BF16, concatenate rows in order and
write the existing buffers. At J=21,120, M4096 covers five full tiles plus a
640-row tail. Each projection has six matmul graph nodes instead of one; this
does not establish the optimized GPU kernel count or its bandwidth cost.

`pixelumm_ffn_isolated_probe --rows 21120 --row-tile 4096` uses the authenticated
full-J FFN fixture, not repeated short rows. Its 24 normal GPU comparisons are
finite and bitwise equal, including unchanged down controls; the NaN positive
control rejects. Median gate/up device intervals are 371.139729 ms baseline
and 372.344271 ms tiled, a descriptive 0.3246% regression rather than a win.
No new named Metal buffer is introduced; opaque MPS temporaries remain owned
by the backend.

`pixelumm_decoder_integrated_probe --tiled-gate-up-rows 4096` prepares both graph
pairs on all 36 plans before the first encode and retains both in all arms.
The serialized owner selects only while the shared workspace is idle; a real
in-flight selection attempt is rejected. Release restores the baseline pair.
The original two gate/up graph calls per block, GPU weight banks, block scratch and
direct block35-to-head buffer edge remain. Internal continuations increase:
4,939 roots/request tiled versus 4,579 baseline. Rollback omits the new flag;
the production inference selector is unchanged.

The prospective CAUTION discriminator was at least 3% lower complete-request
time in both serial orders, with strict correctness as a prerequisite. Each
process has two warmups followed by four timed requests. Normal timed order is
baseline/tiled/tiled/baseline; `--reverse-tiled-gate-up` reverses both warm and
timed modes. All requests use identical input A and the same resident resources.
API/continuation observers are off; memory telemetry remains on. The primary
timer spans encoding through the existing terminal wait, including intervening
memory telemetry and lifecycle checks, and excludes preparation and post-wait
comparisons. It is a diagnostic latency measurement, not production throughput.

| Order | Timed baseline requests, seconds | Timed tiled requests, seconds | Median reduction |
| --- | --- | --- | ---: |
| Normal | 84.561390, 50.566983 | 64.564609, 50.436531 | +14.8949% |
| Reverse | 43.932217, 43.750964 | 44.655761, 42.603742 | +0.4832% |

The reverse order fails 3%; the candidate does not qualify. Normal's apparent
gain is heavily confounded by drift: the same baseline was 44.938757 s during
warmup, then 84.561390 s and 50.566983 s. Do not pool controls across processes
or discard the slow baseline to manufacture a stable estimate. These two
samples per mode/order cannot establish a smaller gain or its absence.

All 12 full-decoder requests pass 516 finite bitwise-equal retained upstream
comparisons at unchanged atol=rtol=2e-5. Every request retains one terminal
wait, zero tracked named encode allocations/preterminal activation reads and
43 existing post-wait diagnostic reads. The oracle is authenticated saved
Torch 2.6 MPS source-method output, not a fresh connected upstream run or an
independent backend. Coverage is the 36 attention captures, block34/35 outputs
and five head stages, not all overwritten intermediates.

The next pre-recorder milestone follows the preceding terminal-wait return by
50.001163..50.009773 s; the analyzer recomputes these intervals and lateness
from raw timestamps, including cross-request links. This precedes recorder
setup, rather than the exact first encode call. These are matched gaps, not
50 seconds of idle sleep. In this mode `prepare_combined_attention=false`, so
the reusable CPU comparison pair is absent (`diagnostic_cpu_scratch_bytes=0`).
Both arms use the legacy per-capture host buffers and post-wait GC. The ready
milestones occur 37.866824..45.590176 s after the previous wait. This policy is
outside request timing but can affect the following request's host/device
state. The old comment claiming pooled comparison arrays also applied here
was inaccurate; it describes only prepared combined-attention diagnostics.
The GPU workspace reuse and absence of encode-time activation readbacks are
unaffected. This experiment does not isolate the cause of the drift.

Normal's baseline warm-to-first-timed GPU-root envelope grows by 39.622502 s,
and its interval union by 39.621598 s. Whole-request envelope-minus-union gaps
remain only 2.60..3.54 ms across both orders. Thus the large change appears in
GPU command intervals as well as host encoding, rather than only a CPU timer
or terminal-wait transfer. These intervals do not prove active shader cycles,
kernel identity, thermal throttling, bandwidth use or queue-admission causality.

All 2,445 full-run guard samples pass the unchanged cache-aware 10% reserve,
27.5 GiB process-group cap, 64 MiB swap-growth stop and 900 s/process deadline.
Sampled peak footprint is 27.346103 GiB, minimum headroom 14.280060 GiB, positive
swap growth zero. The unique named inventory remains 24,108,145,128 bytes.
File-cache reclaimability and sampled bounds do not guarantee opaque transients
or future fit. Both native processes exited normally; no model remains running.

Executed DoD: release-build both probes, `make spec_pixelumm_decoder_integrated`
(67 examples), the focused FFN/decoder/tiling specs (24 examples), full-J GPU
isolation, then normal and reverse guarded full-decoder runs and strict audits.
Six Python tests cover mode grouping, encode-only false wins, raw-timestamp
inconsistency and canonical artifact-path binding. Four CLI negatives reject
before device initialization. A separate CPU adversary check rejects 14
in-memory report corruptions. Format/diff checks pass. The correlated Mini-Quad
verdict is ROBUST within these numerical, guard and nonqualification claims;
stable performance attribution remains VULNERABLE.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-gate-up-rows-20261029`.
`results/gate-up-rows-analysis.json` SHA256
`22c9ea1aa1abb21498dad4a6aa94a176317f808ad061bc77fe39712f1eeba181`
pins the per-order reports, audits, receipts, 199 measured sources and binary.
`results/root-adversary-cpu.json` SHA256
`0e5e0edb59e62225f1e6fef37e14acd9b675ed5d370a10b32557e2c5fbbcd934`
records the targeted CPU challenges. `results/final-cpu-checks.json` SHA256
`8a90e5913bb23e3c177f8f2f0e45e78e843c71b68b1e833a41b32a513bdcaa44`
retains commands, exit codes and output for the final 67/24 Crystal examples,
six Python tests, format/diff checks and four pre-device CLI negatives.
Measured source snapshots and build remain under the same root. After
measurement, two full-line comments were corrected
to describe the per-capture host-buffer policy accurately.
`results/comment-only-annotation.json` SHA256
`2e3ced87676ac0d82ad15300707733a735c3be55bc6a49c6bc361449302fdb5d`
records the exact diff, equality of every noncomment line against both measured
snapshots, the other 198 unchanged source pins and all unchanged build artifacts.
The measured release binary remains SHA256
`1d06a8e8f9fc192e3ccffaad69170e92280803b4064ffc3f8ed9e8cca431fdd0`.
Runtime evidence refers to those archived sources and binary; this annotation
is not a new live-source GPU audit. Refresh on scoped executable-source,
artifact, fixture, weights, geometry/precision, lifetime, schedule,
backend/device/toolchain or guard
changes; admit memory afresh before replay. Do not extend a tile-size grid from
this result. If reopening timing, first falsify diagnostic carryover in a short
replicated FFN A/A with GPU checks retained and no per-capture CPU allocation/GC
between measured samples; this is a proposed discriminator, not an acceleration.

## Matched-pause PixelUMM A/A and delayed roots (2026-10-29)

Status: VERIFIED for bounded matched-spacing A/A, retained saved-output parity,
source/artifact integrity and sampled memory fit. Queue-admission causality,
the earlier severe drift and a production speedup remain separate questions.
Production selectors, queue capacity, GPU math and precision are unchanged;
there was no diffusion or full generation.

`pixelumm_decoder_integrated_probe --profile-aa` holds baseline inference,
input A, full J=21,120 geometry, weights and scratch lifetime fixed for six
36-block-plus-head requests. Only API/continuation observers cross over:
OFF warmup, OFF/ON/ON/OFF, OFF tail; `--reverse-aa-profile` reverses the middle
observer order in a second serial process. Only the first request poisons
scratch. Trace-only labels identify Metal roots without adding GPU commands.

The prospective discriminator was whether matching the long diagnostic gaps
would narrow same-mode drift. Each next pre-recorder, pre-encode milestone is
targeted at the preceding terminal-wait return plus 50 seconds, including
profile extraction, comparisons, chain release and post-diagnostic GC. Actual
gaps are 50.003350 to 50.010330 s. The memory recorder and CPU timer setup follow
this milestone; it is not the exact first encode call. Diagnostic overrun
rejects before chain creation; setup lateness above 250 ms quarantines the
unsubmitted chain until exit. No pause follows the final request. This is a
matched idle interval, not production throughput timing.

The following complete-request seconds cover encode and the existing terminal
wait; the 43 diagnostic comparisons follow outside the timer:

| Request | Normal observer | Normal seconds | Reverse observer | Reverse seconds |
| --- | --- | ---: | --- | ---: |
| 0, warm | OFF | 42.121775 | OFF | 42.205562 |
| 1 | OFF | 42.590244 | ON, Instruments | 43.559919 |
| 2 | ON | 42.819418 | OFF | 42.282234 |
| 3 | ON | 42.954380 | OFF | 42.289385 |
| 4 | OFF | 43.250136 | ON | 42.293956 |
| 5, tail | OFF | 42.754445 | OFF | 42.282982 |

Reverse request1 is explicitly excluded from timing distributions, continuation
aggregates and adjacent deltas touching it by `--external-capture reverse:1`.
The analyzer retains its profile separately and requires a separately audited
capture receipt; it does not authenticate caller-supplied capture declarations.
Later reverse arms may retain capture effects. Neither off/on placement nor
these two runs provide causal profiler calibration. The remaining 11 requests
range 42.121775 to 43.250136 s; the earlier 60.195153 s outlier did not recur.
This observation does not prove that fixed spacing caused the narrower range
or exclude thermal, scheduling and other run-state interactions.

Each of the four observer-on requests records 4,578 continuations. The largest
calls start with 63 earlier roots whose GPU end lies after call entry; the
current `before_root_sequence` root is excluded and is about to be committed.
All four top-ten sets have this boundary, with the earliest unfinished root
ending about 0.17 to 0.33 ms before the call returns. The default Metal queue
allows 64 uncompleted command buffers, as documented by the local SDK's
`MTLDevice.h` and Apple's `makeCommandQueue()` documentation. Thus 63 earlier
roots plus the current root are consistent with queue admission pressure.
GPU-unfinished does not distinguish queued from running roots, and GPU end
does not equal completion-handler observation. This alignment alone does not
locate the blocked CPU stack or prove that a deeper queue would help.

The reverse-request1 Metal System Trace contains 4,579 native roots joined
one-to-one through command-buffer labels, application IDs, submission,
completion and accelerator-ID records for PID11191. Grouped GPU intervals are
4,543 single-interval roots and 36 three-interval roots; their extrema agree
with every native GPUStart/GPUEnd within 42 ns, below the existing 2 us bound.
MPS preserves root0's label and inherits `.root.(n-1)+` for later submitted
roots; the join handles that observed transform rather than equating a label
suffix to the native sequence. The ten longest captured continuations are
206.18 to 218.08 ms; earlier-root GPU intervals cover 99.9996 to 99.9997% of
each host span. These are command/encoder intervals, not active shader cycles
or proof of ALU/bandwidth utilization. Host phase markers can wait on prior
FFN work and cannot rank exclusive GPU attention versus FFN cost.

All ten host locations are `local_attention`, but the earliest GPU-unfinished
prior root has the single native phase candidate `ffn_down` in nine cases and
`ffn_gate_up` in one. These identify candidate predecessors, not the global
critical path or proof that a particular command caused the wait. The time
profile has `record-waiting-threads=0`: no blocked main-thread samples cover
these calls, so this capture cannot identify a semaphore/admission stack.
The current root is still uncommitted at entry; 63 prior roots therefore do
not prove that 64 buffers are already in flight. Commit could fill slot64 and
make creation of the continued buffer wait, but that remains a hypothesis.

Across the selected captured request, the union of all reported GPU intervals
is 43.539713 s within a 43.542493 s root envelope, leaving 2.780 ms uncovered.
Of 4,579 roots, 4,291 have exactly one native phase candidate; the other 288
remain separate mixed-candidate buckets and are never divided among phases.
Single-candidate family interval unions are:

| Native candidate family | Roots | GPU interval union, seconds |
| --- | ---: | ---: |
| FFN gate/up | 180 | 13.3822 |
| FFN down | 72 | 7.0684 |
| Local attention | 3,924 | 11.3844 |
| QKV projection | 36 | 2.0639 |
| Head | 6 | 0.0684 |
| Input norm | 37 | 0.0430 |
| Residual/post norm | 36 | 0.0332 |

These intervals are not exclusive shader utilization or a critical-path proof;
different families can overlap, and mixed roots and internal device stalls
remain unassigned. Their scale makes
FFN the leading operation-level optimization candidate, while the almost
continuous raw interval coverage weakens producer starvation as the current
lever. A deeper queue can shorten CPU continuation wall without shortening
GPU service or the whole request. Retain whole-request A/A as the acceptance
metric for any materially different FFN implementation.

All 12 requests pass 516 finite bitwise-equal retained saved-upstream
comparisons at unchanged atol=rtol=2e-5. They retain 4,579 roots/request, the
direct block35/head GPU-buffer handoff, one terminal wait, zero tracked named
encode allocations/preterminal activation reads and 43 existing post-wait
diagnostic reads using the reusable 330 MiB host comparison pair. Coverage is
36 attention outputs, block34/35 outputs and five head stages, not every
overwritten early output or internal F32 probability. The controls are saved
upstream PyTorch2.6 MPS captures, not fresh independent-backend inference.
No new synchronization was introduced on the successful measured request
path; each request retains exactly its one existing terminal wait.

Both processes return zero with identical 197 measured source pins, three
object pins, binary, inputs and weights. The full integrity audit hashes five
unique fixture payloads totaling 57,168,398,336 bytes. All 2,239 native guard
samples pass the unchanged 27.5 GiB group cap, cache-aware 10% reserve, 64 MiB
positive swap-growth stop and 900 s/process deadline: peak 26.453175 GiB,
minimum headroom 19.696732 GiB, zero swap growth. The old formula label remains
wrong; checked arithmetic is `free_spec - speculative + file_backed`, without
double-counting speculative pages. These are sampled bounds, not guarantees
about opaque MPS transients or future fit. Both native processes have ended.

DoD executed: release build; 73 focused Crystal examples; 12 Python analysis
tests including damaged-report/capture-exclusion controls; 12 integrity
corruptions; four incompatible-CLI rejection controls using complete valid
fixture arguments; format, diff and build-freshness checks. Correlated
Mini-Quad review is ROBUST for the declared integrity/lifecycle scope, not
independent backend or causal acceleration evidence. The capture receipt's
`capture_exit_mach_ns` is a wrapper/reap lifetime bound, not exact tracer exit.
Trace-tree hashing is an audit-time snapshot, not capture-time sealing.

The next causal falsifier is a narrowly scoped capture with waiting-thread
stacks enabled, joining the blocked site to the current-buffer entry status
and prior GPU-end/callback. The present near-continuous interval coverage
provides no material producer-starvation signal during the longest calls.
Do not increase queue depth on this evidence. Only after admission blocking
and a material GPU idle gap are demonstrated should a 64 versus 80/96 trial
keep identical operations, geometry, buffers, one terminal wait and guards,
requiring at least 3% whole-request improvement in both orders and a confirming
repeat. Reject a trial that only moves waiting to the terminal boundary. Earlier
softmax/PV graph grouping gave no material gain, while direct MPS FFN reduced
root count but regressed request time by 7.7081%; fewer calls/roots alone are
not the objective. If GPU service stays critical, choose the next operation
from actual command evidence and preserve prepared graphs/bound resources
through the terminal wait. Rollback omits the A/A flag; defaults stay unchanged.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-aa-spacing-20261029/`.
`python3 -B run_integrated.py normal-1` and `reverse-1` execute the guarded
serial requests.
`python3 -B audit_integrity.py --label normal-1 --label reverse-1 --output
results/integrity-aa-final.json` authenticates receipts, sources, binary,
fixture payloads, reports and sampled guards. The checked-in analyzer runs as
`python3 -B examples/support/analyze_pixelumm_aa_profile.py --report results/normal-1.json
--report results/reverse-1.json --external-capture reverse:1 --output
results/analysis-final.json`; it describes reports without authenticating
execution or asserting bitwise equality on its own.

Integrity JSON SHA256:
`4dd9eeefe4ccc2a33823e6964cf52f06b1695dee95ccd244750e05ef6543791b`.
A/A analysis SHA256:
`2f142641d1f4fcaff06e7b2520be1b0149d70f9e59b6d5a252d36ce52c149707`.
Analyzer source SHA256:
`7b75717b3e22aef7fa86238043e7e90f039899d6e51476e7c713955754dd52f8`.
External auditor source SHA256:
`0e684e44adb71ec09f316292d0cc168b74402bce71960dcac07eb4b009d4021e`.
Measured release binary SHA256:
`9afec75cfa81b6ec73293fcf4bdf0705f830e1b82760b1aa36abcd729bde544a`.
The external `analyze_gpu_capture.py` joins the already captured
`results/reverse-req1-metal.trace` exports to native request1; it performs no
model work. GPU join JSON `results/gpu-reverse-req1-analysis.json` SHA256:
`4aaa05201afe595e4e494c998516307b882064da49f42522c465459e19c69a8f`.
GPU analyzer source SHA256:
`9eba5ea2eb68f97a2714bf79fa01c9fb2c773ceb134882700ffda60c2394f5c3`.
Its export manifest `results/gpu-reverse-req1-export.json` SHA256:
`1a1027280c239bd2f069524b69ae9fd517b0650ae6be0aa26d67d30c774b787d`.
The selected XML SHA256 is
`af2693dec945bac314f549f704505c86fbe4d9dc7e7cbf90e11dea458caa3a7a`.
The all-four-arm native queue discriminator is
`results/gpu-native-queue-depth.json` SHA256:
`bf41c12ee933046e2493c3e2ab6377dbf3d5407d6d383c4f93a0c5aff866d9d1`.
The separate CPU-only `analyze_gpu_phases.py` SHA256 is
`31df199ce1e57d39471d18150572e0395f09ee8f9785ae15b3f8e69ff7a72a3b`;
its `results/gpu-reverse-req1-phase-unions.json` SHA256 is
`56ed147be6bd39eae5b58f51c58af91fb972912c59302701e1b6d9e066dae92e`.
It also reconciles two trace-tree hashes from the same 578 audited file
entries: the exporter hashes length-prefixed paths plus raw file digests,
while the integrity auditor hashes canonical JSON of path/size/digest records.
Those different digest formats are not a content discrepancy.
Refresh after scoped source/artifact, fixture, geometry/precision, schedule or
lifetime, clock/backend/device/toolchain or guard changes. Every new process
requires live memory admission; this decoder-only probe authorizes no full
generation or full-pipeline memory claim.

## Combined-attention order localization (2026-10-29)

Status: VERIFIED for bounded host-phase/API localization, retained saved-output
parity and sampled memory fit. The physical cause of the timing drift and a
stable production speedup remain unresolved. Production selectors stay off;
there was no diffusion or full generation.

`pixelumm_decoder_integrated_probe --profile-combined-attention` adds a separate
eight-request diagnostic on authenticated input A. Both modes warm with poisoned
scratch and profiling off; the next four requests use profiled ABBA, followed
by two unprofiled tails. A second serial process reverses every mode assignment.
Shared gate/up stays disabled. Completed root intervals are collected after the
existing terminal wait on all eight requests; phase wall/thread CPU, API and
continuation records are collected only on the middle four. Exact graph counts
and all 36 latched per-Plan selectors guard against fusion misattribution.
The prospective discriminator was whether the difference followed the mode,
ordinal position, or broad continuation waiting with little caller CPU.

The following seconds are observer-on host measurements, not kernel timings:

| Process/request | Mode | Whole request | Attention marker | Attention continuation API wall | Continuation thread CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| normal-1/2 | baseline | 42.792485 | 40.050632 | 39.841887 | 0.035886 |
| normal-1/3 | combined | 39.870393 | 37.603717 | 37.400612 | 0.035003 |
| normal-1/4 | combined | 39.832005 | 37.566830 | 37.352387 | 0.037410 |
| normal-1/5 | baseline | 60.195153 | 56.773947 | 56.536751 | 0.040737 |
| reverse-1/2 | combined | 40.381760 | 38.068375 | 37.888100 | 0.032238 |
| reverse-1/3 | baseline | 42.584312 | 39.855497 | 39.661136 | 0.033747 |
| reverse-1/4 | baseline | 42.345675 | 39.654472 | 39.471536 | 0.032657 |
| reverse-1/5 | combined | 39.862891 | 37.602929 | 37.427409 | 0.030485 |

The largest same-mode drift is normal baseline request2 to request5:
whole-request wall grows 17.402668 s, attention marker wall 16.723315 s
(96.096% of that change), and nested continuation wall 16.694864 s.
Attention marker CPU grows only 0.033059 s. All 36 block host spans increase;
this is not a single early/late block. FFN gate/up plus down marker wall changes
0.973432 to 1.204150 s, and terminal wait 0.921548 to 1.233585 s. The dominant
change appears while encoding attention, inside the MPS continuation call.
Attention graph-encode wall after excluding nested continuations is 0.157588 to
0.208488 s across these requests. Inclusive graph and nested continuation
spans must not be added together.

The raw Mach trace brackets `[super commitAndContinue]`; our root bookkeeping
lies outside that span. Joining it to GPU root intervals excludes the root
being committed and unions intersections with earlier roots. Earlier-root
intervals cover 99.98794%/99.98795% of attention continuation wall in normal/
reverse processes under the existing 2 microsecond clock bound. Raw attention
continuation wall sums are 171.118798/154.436725 s, while caller thread CPU sums
are 0.140415/0.121149 s. This supports waiting during previously submitted GPU
work, not CPU instruction cost of encoding or our bookkeeping locks. It does
not identify the GPU kernel responsible, distinguish bandwidth/compute/thermal
or other scheduling effects, or prove exclusive GPU busy time. A host attention
span can wait for preceding FFN work; the small FFN host markers do not bound
FFN GPU cost. Parent/depth checks are syntactic because not every graph API
scope is exported; callback observation time is not exact GPU completion time.

All profiled combined arms remain 39.832005 to 40.381760 s at both ends/orders.
Three baseline controls are 42.345675 to 42.792485 s; the fourth is 60.195153 s.
Reverse request5 is fast, contradicting a simple monotonically later-is-slower
explanation. This is a repeatable mode association in these two profile runs,
not causal separation of the selector from run-state interactions. The normal
unprofiled tails remain shifted at 49.432697 s baseline and 42.480520 s combined.
Off warm/tail controls do not calibrate observer overhead. Every request has
4,579 roots; every profile has 6,480 attention graph encodes and 3,960 attention
continuations. Whole-chain root unions track the broad wall shift; shared
phase candidate sums are non-additive and do not attribute individual kernels.

Existing memory-event clocks also expose substantial off-timer observer work:
after the first request, the 43-capture comparison span is 36.023 to 36.183 s.
Profile extraction is about 48 to 50 ms when enabled versus 6 to 8 ms off;
post-diagnostic GC spans are 1.090 to 4.088 ms. These spans include recorder
overhead and are not instrumented GC stop-time or isolated comparator timings.
They cannot directly explain 17.4 s inside request timing, but the long gaps
can change device state and limit extrapolation to uninterrupted production.

Both processes return zero with identical 197 measured source files, release
binary, objects and fixtures. All 16 requests pass 688 finite bitwise-equal
retained saved-upstream comparisons at unchanged atol=rtol=2e-5. They preserve
the direct block35/head GPU handoff, one terminal wait, zero tracked named
encode allocations/preterminal activation reads and the existing 43 post-wait
diagnostic reads, using the same 330 MiB host comparison pair. Retained coverage
is 36 attention outputs, block34/35 outputs and five head stages, not all
overwritten early outputs or internal F32 probabilities. Saved upstream MPS
controls are not fresh connected independent-backend inference.

All 2,612 guard samples pass the unchanged 27.5 GiB cap, cache-aware 10% reserve,
64 MiB positive-swap-growth stop and 900 s per-process deadline. Maximum sampled
footprint is 26.347140 GiB, minimum headroom 19.517944 GiB, observed swap growth
zero. The historical headroom formula label remains wrong; pinned code and
samples use `free_spec - speculative + file_backed`, without double counting
speculative pages. Sampling does not bound opaque MPS transients or future fit.
Native processes have ended.

DoD executed: release build; 62 focused Crystal examples; four Python analysis
tests including 15 corrupted-report controls; eight integrity corruptions;
three incompatible-CLI rejection controls on the measured binary; format,
diff and build-freshness checks. `run_integrated.py normal-1` and `reverse-1`
execute the serial native requests. `audit_integrity.py --label normal-1
--label reverse-1 --output results/integrity-final.json` authenticates source,
binary, fixture, receipt, report, lifecycle and guard identities. The checked-in
`examples/support/analyze_pixelumm_order_profile.py --report
results/normal-1.json --report results/reverse-1.json --output
results/analysis-final.json` recomputes phase, API, root-overlap and observer
gaps; it does not itself authenticate execution receipts or assert bitwise
parity. Correlated Mini-Quad reviews are not independent backend evidence.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-order-profile-20261029/`.
Integrity index `results/integrity-final.json` SHA256:
`37b3b84cad62fe08540b22e8e0bb138b6e976c2e83fcefb42fdc8cbba8d2bd3c`.
Analysis `results/analysis-final.json` SHA256:
`13f8bb895f0033e11877b1489540016739bf164091f86bb50cd88c5f59394cce`.
Analyzer source SHA256:
`2933ce822cffa30accaa3e5d03a50a60b904f1d19607ffe9f36d57a822f948c4`.
External integrity auditor source SHA256:
`84046dfc6aa7d22681bac5af12cb0501f402b69a2ff0c9c1eba0d8009e783a54`.
Measured release binary SHA256:
`b5e501416d6b69685a199df8d2414455a8b4c955a33f6710e5533ce690aeffa5`.
Next discriminator: fixed-mode A/A with matched between-request spacing and
kernel-level GPU attribution of the delayed roots. Do not optimize host locks
or promote a stable speed claim from phase wall alone. Rollback omits the new
profile flag. Refresh on scoped source/artifact, fixture, geometry/precision,
schedule/lifetime, clock, device/backend/toolchain or guard changes; each new
process requires live admission. Earlier unprofiled epochs below are not pooled.

## Combined attention on 36 PixelUMM blocks and head (2026-10-29)

Status: VERIFIED for the bounded opt-in diagnostic, saved-output parity and
sampled memory fit. The prospective timing screen passes descriptively;
a causal or stable generation-speed claim remains VULNERABLE. Production
selectors stay off. No diffusion or full generation was run.

`pixelumm_decoder_integrated_probe --compare-combined-attention` prepares the
existing scale/softmax alternative once on every Plan, then selects it together
with fused K/V gather on all 36 idle Plans. Shared gate/up remains disabled.
Four poisoned gates cover both modes on both authenticated inputs. Four timed
same-input requests use ABBA; another process reverses warm and timed orders.
A third process repeats ABBA. All three runs use the same 197 source files,
runner, objects, fixtures and release binary. Its SHA256 is
`ed8fca7b1ad8d65d79c74232c939de7edac3c3bcec4bcf65a4f38502df0de290`.

The connected T6/H44/W80/J21,120 decoder and head completed 24 requests and
1,032 finite bitwise-equal comparisons at unchanged atol=rtol=2e-5. Every
request retains the direct same-buffer block35/head handoff, one terminal
wait, zero tracked named encode allocations and no preterminal activation
readbacks. Its 43 existing diagnostic CPU reads happen after that wait and
outside request timing. Retained captures are 36 attention outputs, block34/35
outputs and five head stages. Outputs 0-33 are overwritten GPU handoffs, not
separate saved goldens. The oracle composes authenticated saved upstream MPS
controls; it is not fresh connected independent-backend inference or internal
softmax-probability parity. In-flight selector rejection and poisoned replay
checks pass; exceptional failed-wait quarantine is source-reviewed, not injected.

| Run | Timed order | Baseline median | Combined median | Median adjacent paired time reduction |
| --- | --- | ---: | ---: | ---: |
| normal-1 | ABBA | 48.999847 s | 45.762902 s | 6.5132% |
| reverse-1 | BAAB | 49.977961 s | 47.533238 s | 4.9058% |
| normal-2 | ABBA | 49.965031 s | 48.135272 s | 3.2761% |

All three per-run medians exceed the prospective 3% screen, with both orders
and a confirming repeat. The six paired reductions range from -1.4346% to
+9.8273%; four pairs improve. Baseline-then-candidate pairs have a +9.1331%
median, while candidate-then-baseline pairs have a -0.0158% median. The pooled
paired median is +5.9400%, but six pairs and this order sensitivity do not
establish a causal or stable speedup. Do not promote defaults from the screen
alone or interpret request wall time as kernel-only work or video throughput.

The diagnostic preallocates one distinct expected/actual host pair after
aggregate admission and before GPU setup: 173,015,040 bytes each, 346,030,080
bytes total (330 MiB). All 43 captures and requests reuse exact prefix slices.
Expected records are reauthenticated on every read; actual reads, element
scans, tolerance checks and hashes remain unchanged. Post-return collection
runs equally outside both arms' timed intervals. Comparison allocation churn
is bounded by the audited 64 MiB limit, and the reported post-GC host heap does
not grow across these eight-request processes. This bounds those two arrays,
not every host allocation or opaque MPS resource.

All 4,559 guard samples pass the unchanged 27.5 GiB process cap, cache-aware
10% reserve, 64 MiB positive-swap-growth stop and 900 s per-process deadline.
The largest sampled peak is 26.286456 GiB, minimum headroom 19.468338 GiB,
and observed swap growth is zero. The guard's historical formula label is
inaccurate: pinned code and all samples use `free_spec - speculative +
file_backed`, avoiding speculative-page double counting. The final audit
explicitly records this label mismatch; it does not reinterpret the label as
arithmetic or weaken admission. Sampled bounds do not cover between-poll spikes
or guarantee future admission. Native processes have ended.

Release build, formatting, 60 focused CPU examples and 30 validator corruption
controls pass. The native DoD was executed serially by `run_integrated.py`
with labels `normal-1`, `reverse-1`, `normal-2`, each returning zero with
unchanged sources; `validate_integrated.py` over all three reports returns zero.
The runner refuses consumed labels. Preserve evidence rather than overwrite it.
Correlated Mini-Quad reviews support the bounded parity/lifecycle/guard claim;
they are not independent backend agreement.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-combined-full-reuse-20261029/`.
Final index: `results/audit-final.json`, SHA256
`6c09bd6bbf51ce3372dbc94ab1dcda806f967346c904c666e66daabf984d1e40`.
Validator SHA256:
`a58d12cd75811eef8ca8f0e10649a299361a5850f36e4ab2fcc301e8d59db322`.
Receipts pin report, fixture and measured-source identities; source snapshots
and raw guard/log/report files remain external. Rollback omits combined mode
or reverts the scoped diagnostic change. Refresh on scoped source/artifact,
fixture, geometry, precision, schedule/lifetime, backend/device/toolchain or
guard changes; each new process still requires live memory admission.

Earlier epochs remain excluded from timing: the original 28.5 GiB launch was
refused before native execution; the stricter-cap attempt stopped at request5
with a 27.597 GiB sampled peak; post-return GC alone stopped before request4
completed at 27.595798 GiB. Tracked Metal inventory stayed constant while the
old host heap grew; this did not prove a GPU leak or its retention mechanism.
The reusable comparison pair is the subsequent successful source epoch. Raw
attempts remain in the sibling `combined-full`, `combined-full-cap27p5` and
`combined-full-postgc` evidence directories, all dated 20261029.

## Combined K/V gather and score-scale/softmax experiment (2026-10-29)

Prospective bounded test: `pixelumm_fused_kv_gather_probe --scale-softmax`
selects both existing alternatives together, after the owner resets each
request and before creating its command chain. The same four plans retain
the prepared scale graphs in both arms. Baseline disables both flags;
candidate enables both. Production defaults remain off. Rollback leaves both
native selectors disabled; omitting the flag restores the K/V-only diagnostic.

Hypothesis: the isolated gains may compose, but MPS scheduling and queue
behavior can make them non-additive. Correctness must first pass all saved
upstream stage/output comparisons at unchanged atol=rtol=2e-5. Only then admit
unprofiled balanced AB/BA timing on the same release binary and source bytes.
The prospective promotion screen remains at least 3% paired whole-request
improvement and a confirming repeat; fewer dispatches alone are insufficient.

Scope is four J5760 whole-block passes: two windows, each with native block0
output feeding block1. Full captures qualify both arms, followed by output-only
requests. No raw K/V panel or combined internal-F32 comparator is asserted.
Geometry, precision, scratch bindings and GPU-resident handoffs stay fixed.
Each request must retain one terminal wait, one 464-byte packed post-wait
metrics read, zero tracked encode-time named allocations and no preterminal
readbacks. Setup owns the extra softmax graphs; their opaque MPS storage is
outside the named-buffer inventory. Run under the cache-aware 10% reserve,
16 GiB process cap, 64 MiB positive-swap-growth stop and 600 s deadline. No full
generation is admitted. Refresh evidence on scoped source/artifact, fixture,
geometry, precision, schedule/lifetime, backend/device/toolchain or guard changes.

Two serial unprofiled 40-request runs passed that screen on identical measured
source bytes and release binary:

| Run | Baseline request median | Combined request median | Median paired time reduction | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.636905 s | 1.585109 s | 4.1034% | 16/16 |
| 2 | 1.500376 s | 1.387504 s | 7.2021% | 16/16 |

Percentages are medians of adjacent pair reductions, not ratios of the two
arm medians. The pooled paired median is 5.6375%, with 32/32 wins. AB/BA
medians are 3.9808/4.3595% and 7.2021/7.2864%. The absolute regime and effect
size vary between runs; these are descriptive bounded results, not a promised
7% production gain or proof of super-additivity over isolated experiments.
Whole-request time includes owner setup/selection, comparator encoding, the
sole terminal wait and metrics read. Body encode excludes comparator encoding;
root durations are not exclusive GPU busy time. Both arms retain 125 roots.

All 400 compared saved-upstream stage/output slots across timing runs are
finite and bitwise equal. Every request retained one terminal wait, one packed
metrics read, zero tracked encode-time named allocation and no preterminal
readback. Sampled peak footprint was 7.625 GiB, minimum cache-aware headroom
22.402 GiB and positive swap growth zero under the declared limits. Samples
do not bound opaque MPS allocation transients. The artifact validator rejects
14 damaged controls covering schedule, coverage, NaN control, receipt, combined
flags, tolerance and memory floor. Twenty-one focused CPU examples pass.
Correlated Mini-Quad source and artifact reviews return ROBUST within this
successful unprofiled, saved-stage/output scope. Independent pair recomputation
also distinguishes the 3.1643/7.5230% ratios of arm medians from the stated
paired estimates; neither estimator establishes a stable production effect.

The combination passes the prospective local request screen. Keep production
selectors off pending connected 36-block/head validation; no full generation
was executed. DoD executed: the release `pixelumm_fused_kv_gather_probe` target,
four-request correctness gate, then two balanced timing runs under
`examples/support/wan_memory_guard.py`. Numerical failure would block timing.
The pinned fixture, weight cache, common driver and comparison stages are
unchanged. Evidence, raw reports, exact measured-source archives, runner and
validator are under:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-fused-kv-scale-softmax-20261029/`.
Summary SHA256 `b8ebc68b5c949196791a15b03d04f8f58c569b10609a817e7942fc0a2bf96bd9`;
timing binary `7282e0b3897be2dcefa2586aad67ef94552248e95c0368fd318375405e35b073`;
raw reports `660ca83379046c0a5ee69dd2bb2be1eff4d3b6159a173ac181696d2ab141995c`
and `a0149bb970ece3c07b38ae586d22d7a48957b9a738a963b8a85624ee10e4fd75`.

Source review found that a partial preparation failure could label the graph
configuration as already prepared. A post-timing setup/report-only fix now
sets that flag only after all four plans finish preparation; request clocks,
selection, encode, wait and comparison code are unchanged. The original
timing binary and source archives remain intact. A separate final release
build passed the guarded four-request correctness gate; its report SHA256 is
`0ea67e27250711ea9f030ada36c8d92115885d7dbcbfe939b9bf3d5bec89131a` and
binary is `280831e14493976b71ea42a367ef0963605582907588ca721294cb54046c62e6`.
`results/final-qualification.json` authenticates the source delta. Timing
claims belong to the archived measured binary, not a new final-binary timing.
The optional Instruments failure-cleanup route was not exercised; the
successful unprofiled request lifecycle is the admitted scope.

## Exact FFN/attention rewrites rejected by the request screen (2026-10-29)

The remaining dense FFN projections remain the strongest hotspot candidate by
arithmetic volume and overlapping matrix-shader intervals; those intervals do
not establish exclusive FFN GPU cost. A refined timestamp-containment audit
uniquely assigns all 1,138 matrix-shader intervals to a GPU root. In captured
output-only requests, the definite FFN/attention matrix timeline unions were
378.341/119.211 ms (baseline) and 288.138/90.792 ms (packed). These subsets do
not cover unresolved intervals, active cycles or unprofiled request speed.
The reproducible strict-containment calculation and input identities are in
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-hotspot-20261028/results/matrix-attribution-20261029.json`
(SHA256 `56d563dcd90858748daee36725b1cccec0de79e7ef72f2045d40cc00ddd43942`).
Shader rows do not carry command-buffer IDs; the assignments use timestamps.
Two rows partially intersect a second root while remaining fully contained in
exactly one. The companion Python script recomputes these checks from the raw
capture exports; this does not establish causal or exclusive shader costs.
Reorienting all three F32 GEMMs as
`transpose(W @ transpose(X))` passed all six saved upstream BF16 stages on
both J5760 windows and blocks 0/1, but failed the whole-request speed screen:
-0.6382% median paired improvement, 4/16 wins, negative AB and BA medians.
Roots increased from 125 to 137. Reject this unchanged route; shorter terminal
waits did not compensate for longer encoding/MPS call intervals.

Evidence (diagnostic source archived with receipts):
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-ffn-orientation-20261028/results/analysis-timing-1.json`.
No full-J timing, connected decoder, exclusive GPU-cost or generation claim.

The output-only softmax/PV rewrite also passed both J2048 and J21120 raw
F32 attended-output comparisons and all saved BF16 captures, but failed the
whole-request speed screen: -0.7061% median paired improvement, 6/16 wins,
negative AB and BA medians, unchanged 125 roots. Removing an external output
binding did not establish removal of internal probability storage. Keep it
explicitly diagnostic and off by default; do not repeat the unchanged route.
Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-output-only-pv-20261029/results/analysis-timing-1.json`.

The explicit macOS 15 SDPA candidate failed the first BF16 rejection screen:
plan 0 post-attention norm had 4,899 tolerance failures / 7,776 differing words,
followed by downstream FFN/output failures. Baseline passed. No SDPA timing was
admitted. The diagnostic implementation was archived externally and removed
from the live core rather than leaving a known-inaccurate alternative there.
Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-sdpa-20261029/results/correctness-1.json`.

Pairing adjacent query groups with identical ordered K lists reduced 12 groups
to 6 but failed the unchanged BF16 gate: plan 0 post-attention norm had 206
tolerance failures / 329 differing words, followed by larger FFN and successor
failures. Baseline passed, with no nonfinite values. This is the first captured
failure, not a localization to QK, softmax or PV. No timing was admitted.
Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-query-pair-20261029/results/correctness-1.json`.

Classic `MPSMatrixMultiplication` for gate/up/down also passed the unchanged
saved upstream stage gate with bitwise equality, including the final-source
correctness-only repetition. Explicit GPU conversion retains BF16-to-F32
operands, full-K F32 multiplication and the original BF16 output boundaries.
All descriptors, kernels and the serialized 192 MiB F32 weight stage are
prepared before requests. Combined named storage is 7,582,827,060 bytes;
a 7 GiB negative admission rejects before fixture/Metal initialization.

The final-source unprofiled run completed 40 requests, including 16 balanced
AB/BA timed pairs. Baseline/candidate request medians were 1.582481/1.708753 s;
the paired median time reduction was -7.7081%, with zero candidate wins and
negative medians in both orders. Encoding was 26.99% shorter but the terminal
wait was 37.71% longer. GPU root-duration sums also worsened by 7.39%; neither
root durations nor API waits establish exclusive kernel costs. Root count
fell 125 to 97. Reject the unchanged candidate: fewer roots and shorter host
encoding are not a request speedup.

All 200 compared slots were finite and bitwise equal. Each request kept one
terminal wait, one 464-byte post-wait metrics read, no preterminal read and
zero tracked encode-time named allocations. Guard peak was 7.807 GiB, minimum
cache-aware headroom 26.041 GiB and positive swap growth zero. The 10% reserve,
16 GiB process cap, 64 MiB swap-growth stop and 600 s deadline held; sampled
guard evidence does not bound unsampled MPS allocations. The final-source
build and artifact validator passed; four damaged CPU artifacts were rejected.
Static correlated adversary review found the transactional release and
reciprocal diagnostic-mode guards robust within their serialized lifecycle.

Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-ffn-direct-mps-20261029/results/analysis-timing-1.json`;
raw report SHA256 `d8f497d76d65502621cce994ef2c52a023d8b5d4ae5d67660c610e6e07c3bdd7`,
binary `4d54eb37e6c0c95edfa19a3e4496d896a5d90d58a95650fcedd175f971fe082c`.
Archived measured-source copies and execution receipts preserve the precise
source state. Rollback leaves the MPSGraph projections selected. No full
generation or improvement was established by this candidate.

Scope correction: the frozen common driver and runtime binding verifier used
in these probes connect each native block-0 output to its block-1 input. Earlier
text describing independent saved block-1 inputs was incorrect. The driver
SHA256 is `e83f70829a67652baa0b37226a066c0acaac8bf23277cc8e621133f478ccdf39`;
its archived copies and execution receipts agree across scale-softmax,
orientation, output-only, query-pair and packed gate/up runs. This establishes
two connected blocks per window, not a complete decoder/head or generation.

## Fused K/V gather qualification (2026-10-29)

The next opt-in candidate gathers K and V in a single native dispatch per
attention tile, retaining the original head/token mapping and two disjoint F32
destinations. At J5760 this replaces 24 K/V dispatches with 12 per block. It
does not reduce tensor storage or claim lower physical DRAM traffic. The
baseline selector remains the rollback; native CogniGraph islands are mutually
excluded. Plans and the added pipeline are prepared before requests.

The falsifier is the unchanged upstream stage gate on two connected blocks in
each of two windows, including full captures, output-only checks and the NaN
control. Require atol=rtol=2e-5, one terminal wait and packed post-wait read,
zero preterminal reads or tracked encode-time buffer allocations, a 10%
cache-aware reserve and the existing 16 GiB/64 MiB swap-growth/600 s limits.
Only after correctness may 16 balanced AB/BA pairs test the prospective 3%
whole-request improvement screen. No raw K/V panel bitwise comparator or full
generation is part of this candidate's claim.

The release build, correctness-only qualification and subsequent final-source
40-request run passed. All 200 compared slots were finite and bitwise equal;
the NaN and malformed-selector controls rejected as expected. Sixteen balanced
pairs gave baseline/candidate medians 1.448511/1.435620 s, paired median time
reduction 1.2331%, 16/16 candidate wins, and positive AB/BA medians
1.1986/1.6199%. Root count stayed 125. This is a small descriptive gain below
the prospective 3% default-promotion screen; keep the selector off by default.
Do not interpret differences from previous runs' control medians as a gain.

Each request retained one terminal wait and one 464-byte post-wait read, zero
preterminal reads and zero tracked encode-time named buffer allocations.
Sampled peak footprint was 7.618 GiB, minimum cache-aware headroom 24.321 GiB
and positive swap growth zero. The unchanged limits held. The artifact
validator passed and rejected four damaged CPU controls; the focused CPU
schedule/lifecycle suite passed 29 examples. Correlated source review found
the default selector, buffer mapping and ABI/link isolation robust in the
serialized, unprofiled request scope. The added pipeline is created at plan
setup even with the selector off: preservation applies to default request
execution, not zero setup overhead or setup failure independence.

Evidence and archived measured source:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-fused-kv-20261029/results/analysis-timing-1.json`;
raw report SHA256 `94cac65039df4e67cf9c033d44ed4f130e0f140fc0475a38eccade9c26e27f7c`,
binary `6e22a64d909cd4613ab4601060fb2b79f0beaf88d07365c65193ecb207457e68`.
Refresh after scoped source, geometry, backend/device, schedule or guard
changes. No full generation or exclusive GPU-utilization claim is established.

## Operation-count roofline for the fixed BF16 preview (2026-10-29)

This is a CPU-only arithmetic model, not a new model run or measured hardware
utilization. The local device has 38 M2 Max GPU cores and 64 GiB unified memory.
[Apple specifies 400 GB/s bandwidth](https://support.apple.com/en-us/111838).
The nominal 13.6 TFLOP/s FP32 figure comes from the methodology of
[Benazir and Lin's primary study](https://arxiv.org/html/2508.08531v1), not an
Apple-published FP32 guarantee or a calibrated sustained MPS rate on this host.

Count two FLOPs per MAC. The fixed first-segment preset has 120 frames at
512x288/24 fps, 54 DiT blocks over eight steps and 36 PixelUMM blocks over five
overlapping windows. Saved Qwen state is an input; live Qwen encoding is
excluded. DiT linears process 4,320 video plus 64 stored text rows; attention
uses 4,320 plus 23 valid text rows. For D=2,048 and F=8,192, each DiT block
costs `24D^2 + 8D^2*S + 4D*F*S + 4D*A^2`, with S=4,384 and A=4,343.

PixelUMM uses J=5,760, D=4,096, F=12,288, 32 query heads, eight KV heads and
head width 128. The 3x4 local-tile layout has eight 640-query and four
160-query groups; clamped neighborhoods give every group 4,320 keys. One block
therefore has 1.739462 TFLOP FFN, 0.483184 TFLOP Q/K/V/O and 0.407686 TFLOP
QK/PV: 2.630332 TFLOP of counted matrix work.

| Counted matrix work | TFLOP |
| --- | ---: |
| DiT blocks, 54 x 8 | 257.439 |
| Text refiner and DiT input/output adapters | 2.222 |
| PixelUMM blocks, 36 x 5 | 473.460 |
| PixelUMM stem and head, five windows | 1.457 |
| Total | 734.577 |

The fresh unchanged-baseline diagnostic is 10.521328 TFLOP across four full
J5760 block passes, divided by 1.448511 s: 7.264 TFLOP/s of counted work per
request wall second, or 53.4% of the nominal FP32 rate. The earlier 1.582481 s
control gives 48.9%; between-run drift is not a candidate speedup. These ratios
are not ALU utilization. The conditional all-work-at-FP32-peak time is 0.774 s;
only the explicit-F32 subset gives 0.632 s. BF16-fed Q/K/V/O graph lowering is
opaque.

The historical BF16 preview receipt was reread, not rerun: 103.315 s neural
wall, 45.098 s setup and 149.110 s application total. This is 20.663 s neural
time per video second, or 29.822 s including cold setup. All counted work at
13.6 TFLOP/s would take 54.013 s; the explicit-F32 subset alone is 647.604 TFLOP
and takes 47.618 s at that assumed peak. A hypothetical double rate for opaque
BF16-fed projections gives 50.816 s. These conditional compute-only models
omit elementwise work, conversions, reductions, resource preparation and
scheduling; they are not achievable latency predictions or physical floors.
The conditional gap is about twofold, not evidence for a 10-100x kernel gain.
It does not extrapolate to higher resolutions, longer videos or quality.

The bandwidth ridge at the cited nominal rates is 34 FLOP/byte. An ideal
one-pass F32 gate GEMM has about 1,002 FLOP per interface byte. This explains
why smaller weight storage alone need not accelerate the large-row GEMMs;
actual tile reloads, conversions and attention traffic still require direct
measurement. Allocation sizes and logical interface traffic are not physical
DRAM bytes, and these estimates do not prove the whole pipeline compute-bound.

Reproduction, exact integer counts, source/artifact identities and limitations:
`python3 /opt/mldata_ssd/cogni-ml/hy-nomas-roofline-20261029/calculate_roofline.py`.
The report is `results/roofline.json` in that directory. Refresh after changes
to shapes, steps/windows, precision, kernels/backend, hardware or scoped source
and measurement identities. No new wait, readback, model or generation ran for
this calculation.

## F32 score scale inside softmax: bounded paired gain (2026-10-28)

The candidate replaces the standalone in-place F32 score multiply with an F32
multiply inside the rank-4, axis-3 softmax graph. QK and PV retain rank-3 batch
32 geometry. The scale uses the same Float32 value as the baseline. Selection
remains opt-in; rollback is the unchanged separate-scale graph selection.

The principal failure is a changed rounding/reduction boundary hidden by the
later BF16 cast. Before timing, compare probabilities and attended F32 values
tile-by-tile from identical raw QK/V, candidate first so the baseline's in-place
scale cannot corrupt its input. Require finite values and unchanged
`atol=rtol=2e-5`, followed by the saved upstream BF16 block captures. The existing
F32 softmax/PV checker consumes already-scaled scores and must not be reused
unchanged. No persisted upstream fixture currently contains all internal F32
attention tensors; native same-input checks supplement, rather than replace,
the authenticated upstream block oracle.

DoD: the correctness-only plan-reuse probe passes the direct F32 checks and
NaN control; the scale-softmax probe passes the two-window/block-0/1 upstream
captures and completes balanced ABBA/BAAB unprofiled whole-request pairs.
Every request keeps one terminal wait, one packed post-wait metrics read, no
preterminal reads or tracked encode-time named buffer allocations. Use the
cache-aware 10% reserve, 16 GiB process cap, 64 MiB swap-growth stop and 600 s
timeout. No full generation. A graph-level multiply alone does not establish
that lowering removed a panel sweep; promotion requires a measured request win.

The correctness-only release probe passed both J2048 and full J21120 at the
unchanged tolerances. At full J it compared 2,229,534,720 probability values and
86,507,520 attended F32 values tile-by-tile: zero failures, differing words or
nonfinite values. All seven saved upstream block captures also matched bitwise.
These internal F32 checks are against the native same-input baseline; they are
not a newly captured independent upstream F32 attention oracle. The NaN control
failed exactly once as intended. An external mutant applying scale twice was
rejected: 75,379,622 probability and 8,379,834 attended values exceeded tolerance
on J2048, and the saved upstream attention gate failed. No tolerance changed.

Two serial unprofiled processes each completed 40 requests: four correctness,
four ABBA warmups, and 32 timed requests in balanced ABBA/BAAB cycles. Each
request contains four complete J5760 block passes, two initial noise windows
times blocks 0/1. Each window starts from its saved upstream input; the native
block-0 output feeds block 1. This is connected two-block diagnostic timing,
not full-J timing, a complete decoder/head or generation qualification.

| Whole-request wall metric | Run 1 | Run 2 |
| --- | ---: | ---: |
| Baseline median | 1.590239 s | 2.904161 s |
| Scale/softmax median | 1.559931 s | 2.860022 s |
| Paired median time reduction | 2.17145% | 2.00291% |
| Candidate wins | 14/16 | 14/16 |
| Baseline-first paired reduction | 1.07183% | 2.03262% |
| Candidate-first paired reduction | 2.77269% | 2.00291% |

Across all 32 pairs the descriptive median reduction is 2.04714%, with 28
candidate wins. This is below the pre-existing prospective 3% promotion
screen; the candidate remains off by default. The much larger absolute drift
between processes remains
unexplained. Do not infer generation throughput, a stable host baseline or a
causal bandwidth limit from these measurements. Both arms retain 125 MPS
roots/request. The candidate omits one explicit Metal scale dispatch per tile;
the physical multiply/softmax lowering and DRAM traffic were not profiled.

All 80 requests passed their numerical gates. Correctness requests compare
six captures per plan, and every request compares all four final outputs;
all compared values match bitwise. The separate direct-F32 checker is excluded
from timing. Every request retains one terminal wait and one 464-byte packed
metrics read, zero tracked encode-time named allocations and preterminal reads.
The direct checker instead uses its existing single 176-byte metrics read.
Sampled timing-run peak footprint was 7.625 GiB, minimum cache-aware headroom
26.411 GiB and positive swap growth zero. Full-J correctness peaked at 9.606
GiB. These are sampled overlapping memory counters, not bounds on transients
or evidence of candidate-specific memory savings; both alternatives coexist.

Release builds, 37 focused CPU examples, format/diff checks and the artifact
validator pass. The validator verifies schedule, stage coverage, positive
controls, execution/report identity and source stability; eight deliberately
damaged CPU artifact controls were rejected. A correlated Mini-Quad review
independently recomputed the raw pairs and returned ROBUST for this bounded
descriptive claim. It found an incorrect timer-description string: plan body
encoding excludes comparator encoding, which is reported separately and is
included in full-request wall. Only this metadata string changed after timing;
the evidence packet pins both versions and leaves the raw reports intact.

Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-scale-softmax-20261028/results/summary.json`
SHA256 `48207f1eff554fd0f7b3daadd7da72d870f05fe8485fc487b1fefc29b5213aab`.
Reproduce direct qualification with `pixelumm_plan_reuse_probe --scale-softmax
--check-scale-softmax-f32` and the pinned small/full manifests; use
`pixelumm_scale_softmax_probe` with the pinned two-window reference and BF16
weight cache for timing. Exact guarded argv, source copies, fixture hashes,
binary/object identities and guard logs are indexed by the packet. The 10%
admission expires at each launch. Refresh on computational source, fixture,
weights, geometry, precision, scheduling, lifetime, backend/device/OS/compiler,
clock or guard changes, or artifact loss. Next falsifier: a larger exact
FFN/attention change must first meet the prospective whole-request screen,
then pass connected decoder/head validation before default selection. No full
generation ran; the larger FFN/attention bottlenecks remain open.

## PixelUMM hotspot profile and packed gate/up refutation (2026-10-28)

The optimization space is not established as exhausted. The admitted profile
and candidate experiment completed; the candidate demonstrated no acceleration
and remains diagnostic-only. Production graph selection and GPU arithmetic
are unchanged. The useful pivot is from graph/encoder counts to measured work
and intermediate traffic, with whole-request time as the promotion metric.

Scope: J5760, two saved initial-noise windows and blocks 0/1, four Plan passes
per request, BF16 handoffs and F32 FFN. Within each window, native block 0
feeds block 1. This is a connected two-block diagnostic, not a complete decoder,
decoder head or full-generation qualification. The oracle is the existing
authenticated Torch 2.6.0 MPS upstream-source-method capture, not a newly run
independent backend. Keep `atol=rtol=2e-5`, attention QK/PV rank-3 batch 32 and
rank-4 softmax axis 3. No tolerance, synchronization or generation expansion.

The first profile exposed mismatched CPU clock domains: a nested Mach trace
could exceed its enclosing CLOCK_MONOTONIC API interval. API wall timers and
trace resolution admission now use CLOCK_UPTIME_RAW, matching Mach raw uptime
after timebase conversion. The existing 2 us accounting bound is unchanged.
A CPU-only nested-clock probe qualifies the relation, while preserving the
old two-output CPU-probe ABI. No Metal object or GPU wait is created by it.

The repeated baseline profile completed 102 requests: two correctness, four
warmup and 96 balanced ABBA/BAAB timed requests, 48 each with API profiling on
and off. In the off arm, median request wall time was 1.555039 s, current-thread
CPU 8.842 ms, body encode wall 723.879 ms and terminal wait 824.499 ms. Encode
wall includes MPS backpressure; it is not active CPU encoding cost. These CPU
figures exclude other driver threads. Three collections totalled 1.863 ms over
the 156.176 s observed loop. GC/host-lock removal is therefore a weak primary
lever for this workload; this does not establish a general runtime verdict.
First/last eight off-arm requests drifted from 1.427 to 1.614 s; the older
4.6 s severe drift was not reproduced or causally explained.

The tested candidate concatenates BF16 gate/up weights inside one MPS graph,
performs one logical F32 matmul, rounds its combined output to BF16, then
slices into the original output buffers. Runtime concatenation is included.
The shared input conversion does not remove gate/up arithmetic. Logical
matmul count is not a statement about MPS's physical kernel decomposition.
The bridge is linked only into `pixelumm_packed_gate_up_probe`; it retains and
restores Plan selector state under a serialized idle-caller precondition.

The unprofiled run completed 40 requests: four correctness, four warmup and
32 balanced timed requests, 16 per arm. Adjacent-pair comparisons:

| Metric | Baseline median | Packed median | Paired candidate change |
| --- | ---: | ---: | ---: |
| Whole request | 1.425029 s | 1.427755 s | +0.1228% |
| Body encode wall | 675.196 ms | 570.635 ms | -15.5643% |
| Terminal wait | 749.863 ms | 857.328 ms | +14.2570% |
| GPU root-duration sum | 1.402999 s | 1.404167 s | +0.0251% |

Packed won only 5/16 whole-request pairs; the paired range was -0.6415% to
+0.5934%. The apparent encode win moved into terminal waiting. Neither an
acceleration nor equivalence is established. Roots fell from 125 to 117 per
request, providing another counterexample to root-count-as-performance.
All compared finite capture/output words matched upstream bitwise, and the
NaN positive control fired. Each request retained one terminal wait, one
464-byte packed metrics read, zero preterminal tensor reads and zero tracked
encode-time named MetalBuffer allocations. These counters do not measure
opaque MPS/native allocations. Failed sole waits are never retried; an
undrained owner is retained rather than explicitly freeing in-flight resources.

A fresh Instruments capture covered four correctness requests, two per arm;
none qualifies for speed comparisons. Exact-PID mapping accounts for all 484
roots and 862 target Compute rows, with zero unmapped target rows. Root-duration
error and single-clock-offset spread were both about 42 ns, within the unchanged
2 us gate. Shader rows are associated by time overlap, not command-buffer
identity. Their phase/root coverage is non-exclusive, and capture/output-only
modes expose different labels. Do not add phase totals or infer exclusive
matrix cost from them. The GPU-counter export exceeded its 90 s limit; partial
counter XML is excluded. Bandwidth, occupancy, frequency and ALU saturation
remain unqualified. The trace narrows candidate areas without proving a roofline.

### What still offers a plausible lever

The following are source-derived work counts or hypotheses, not measured gains:

- At J5760/D4096/FFN12288, one FFN projection requires 579.821 GFLOPs; gate,
  up and down together require about 1.739 TFLOPs per block. Combining calls
  leaves that arithmetic intact. A faster dense path would matter, but the
  prior native F32 GEMM and lower-precision reductions failed speed or accuracy
  gates. Do not reopen them without addressing the recorded failure.
- Local attention processes 796,262,400 F32 score values per block pass (one
  window), a cumulative
  3,185,049,600 bytes of panels, plus the same probability volume. This is not
  resident memory or measured DRAM traffic: peak scores+probabilities are
  675 MiB, and caches/lowering affect physical traffic. QK and PV together
  require about 407.686 GFLOPs. Removing materialization is a potentially large
  structural lever, but its actual latency contribution remains unmeasured.
- The smallest exact attention probe is folding the standalone F32 score scale
  into softmax while preserving the F32 rounding boundary and batch geometry.
  Verify that lowering actually removes a panel sweep; a shared graph alone
  is insufficient. Compare scaled scores/probabilities/attended outputs on
  identical inputs before the existing BF16 block gates. The existing helper
  expects already-scaled scores and cannot be reused blindly for this test.
- A streamed QK/softmax/PV path could avoid score/probability storage. It is a
  higher-risk route because online softmax and tiled accumulation change
  reductions. It remains a hypothesis until unchanged intermediate/output
  gates pass; numerical algebra alone is not an equivalence certificate.
- One-time weight layout preparation or batching FFN rows from two independent
  windows might improve matrix reuse. MPS may already absorb transposes, and
  a larger M dimension may select different reductions. Require evidence of
  repeated conversion/layout work, account for extra live storage and setup,
  and do not infer a gain from fewer calls. Prior F32 staging alone was neutral
  or worse.
- Cold pipeline preparation and decoder overlap deserve separate accounting.
  The earlier five-second preview spent about 45 s preparing DiT/decoder;
  those are historical measurements, not this run. Reusing owners may reduce
  startup latency but retaining every phase's weights needs a fresh memory
  budget. Five overlapping decoder windows process 200 frame positions for
  120 displayed frames; context changes prevent naive cross-window activation
  reuse. Neither timestamp equality nor identical weights proves cache safety.

The scale/softmax experiment proposed here is reported above. The prospective
promotion policy requires repeated unprofiled balanced whole-request improvement in
both orderings, outside the observed A/A variation, with setup/deferred work
and resource growth included. Use a prospective 3% median improvement screen,
not a claim that this threshold proves significance. Hardware-limit claims
also need qualified counters and stable device state; none is made here.

All three native runs passed the 10% cache-aware reserve, 16 GiB group cap,
64 MiB swap-growth stop and 600 s timeout. Sampled group peaks were 8,180,976,952 B
(baseline), 8,989,412,664 B (packed) and 10,329,643,728 B (Instruments), with zero
positive swap growth. Lowest sampled headroom across them was 28,249,341,952 B.
Cache headroom is a reclaimability estimate and polling cannot prevent transient
spikes. The diagnostic's named inventory remains 7,381,500,468 B, including
unused 576 MiB staging and saved comparison captures; it is not a pipeline budget.

DoD commands: `make pixelumm_packed_gate_up_probe`,
`make spec_mps_api_profile_clock`, and `crystal spec -Dcpu_only` over the six
profile/schedule/weight-reuse spec files listed in the execution logs. Release
build, 44 CPU examples, two native CPU clock examples, eight Python accounting
tests and format/diff checks passed. The clock spec is excluded from CPU-only
and non-Darwin runs. Correlated adversary review is ROBUST for these bounded
numerical, lifecycle and descriptive measurement claims; severe-drift causality,
full-pipeline throughput and GPU saturation remain unverified. Rollback leaves
the packed candidate unselected; production defaults are unchanged.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-hotspot-20261028`.
`results/hotspot-summary-1.json` SHA256
`4a9417ef0948d3605394d751e576dd5ac87480274cb6f2efd8c4d59985a523ee`
pins reports, guard logs, execution receipts, selected analysis and final CPU
checks. Measured sources were unchanged during each run and archived per run.
The unprofiled 40-request run predates attribution-only report/attach-gate
additions and restoration of the old CPU-probe ABI; the four-request trace
qualifies the current ABI/numerical path, not a new timed comparison. Final
formatting, the test guard and Makefile test target followed the trace; the final
release binary built successfully but has no separate GPU performance claim.
Refresh on source, fixture, weights, runtime, device, geometry, precision,
schedule, backend or guard changes, or evidence loss. Admit memory afresh
before every GPU replay. No full generation ran in this experiment.

## Active experiment: prepared CogniGraph native islands (2026-10-28)

Implemented opt-in, CAUTION: share the existing CogniGraph dependency compiler with
prepared native PixelUMM islands. MPSGraph remains the projection, normalization
and attention backend. Compile access/range dependencies and copy fixed dispatch
parameters once during plan preparation. Each island borrows the current root;
it must close its encoder before any MPSGraph call or capture blit. Retain all
bound resources under the existing serialized Plan/Workspace lifetime.

The first bounded candidates are independent Q/K preparation, Q/K/V panel
gathers together with the preceding tile's independent output scatter, and
SiLU followed by product with one dependency barrier. The final tile retains
its trailing scatter. Capture blits remain ordered and may split the FFN island.
No operation may cross an opaque MPSGraph boundary. The control keeps the
original encoders; selection is explicit and defaults off. This is reuse of CogniGraph scheduling,
not a new LTP/WBA certificate or a claim about the historical Nomic MoE speedup.

Prediction: fewer native encoders can reduce whole-block request time without
changing BF16 rounding or MPS reduction geometry. Strongest failure: concurrent
dispatches alias mutable scratch or escape the current MPS root. Guard it with
shared scheduler dependency tests, prepared-plan validation, poisoned transients,
the six upstream capture points per block (`post_attn_norm`, `gate_proj`,
`up_proj`, `product`, `down_proj`, `output`), output-only candidate comparisons, and
one terminal wait/readback. Keep `atol=rtol=2e-5`, a cache-aware 10% reserve,
the existing 16 GiB process cap and 64 MiB swap-growth bound. No full generation.

Implementation/rollback: factor the scheduler in `src/ml/metal/compute_graph.cr`,
add its CPU-testable native schedule adapter, and integrate opt-in prepared
islands in `src/ml/video/pixelumm_decoder_block_mps_bridge.mm`. Add a balanced
same-process probe over the pinned J5760 two-window/two-block fixture. Rollback
is leaving the candidate disabled; do not change production selection without
numerical and paired whole-block evidence. DoD: focused scheduler/probe specs,
release `pixelumm_cognigraph_probe`, unchanged upstream gates including a seeded
nonfinite control, then guarded ABBA/BAAB whole-request timing with both arms
sharing weights, inputs, scratch and MPS graphs. Record root/encoder counts,
host encode wall/CPU, terminal wait, GPU intervals and peak memory separately.

Observed result: all 40 requests pass the unchanged upstream gates. The baseline
and candidate correctness requests each compare six stages for each of four
plans; every request compares all four block outputs. All compared words are
bitwise equal to the saved authenticated upstream-source-method MPS captures;
the seeded NaN control detects exactly once. This is J5760, blocks 0/1 on two
windows, not a fresh connected upstream run, full-J21,120 qualification, the
36-block decoder, head or full generation. Optional Q/K captures and the
SiLU-capture fallback have source review but no direct runtime qualification
from this fixture.

Four ABBA+BAAB cycles give 16 adjacent timed pairs. Raw full-request medians are
1.599111 s baseline and 1.602950 s candidate; paired median change is -0.1124%
(mean +0.0897%, range -2.145% to +2.716%, eight wins in sixteen pairs). This run
does not demonstrate a meaningful speedup; it does not prove zero effect.
Calling-thread CPU medians are 12.057/13.718 ms, with a noisy paired median
change of +20.165%. Each arm retains 125 MPS roots/request. Native encoder
counts of 66/29 per block are prepared-plan model counters, not measured API
calls. Recipes, bindings and dependency waves are reused; Metal commands are
still encoded into each current root, not replayed from a preencoded command
buffer. Candidate selection remains off by default.

Every request has one terminal wait, zero tracked named encode allocations or
preterminal readbacks, and one 464-byte metrics read after completion. Sampled
process-group peak is 7.619 GiB against the 16 GiB cap; minimum cache-aware
headroom is 29.868 GiB and swap growth is zero. The 10% reserve remains intact.
The footprint includes both retained alternatives; it is not a candidate-only
memory comparison. Sampled limits do not bound inter-poll transients.

Validation: release build and fresh `make -q`, 37 focused CPU examples, four
Python analyzer tests, three existing GPU ComputeGraph examples, format/diff
checks, and the guarded 40-request run pass. The guard launches the binary
directly so its PID matches the report. Its analyzer rejects missing process
liveness, wrong guard policy, changed tolerance/pins, extra reads and wrong
mode transitions. The execution receipt pins unchanged measured source and
binary hashes. Correlated Mini-Quad verdict: ROBUST for bounded correctness and
the absence of a demonstrated gain in this run; no general speedup claim.

Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-cognigraph-20261028/results/summary.json`,
SHA256 `26b34c70cb12f4cbcdb9e0e452086f62a5a4c7d37f1acae9220e53f97d830dac`.
It pins the raw report, guard, execution receipt, analysis sources and curator.
Refresh after scoped source/artifact/backend/device/OS/toolchain, geometry,
precision, scheduling, lifetime, guard or clock changes, or evidence loss;
memory admission is fresh per launch. Next falsifier: use the existing phase
and root profile on identical operands to target dominant GPU FFN/attention
work; fewer host encoders alone did not improve this request. Ordinary GC
remains selected. The historical preview boundary below is not a new generation
launch request.

## Preview scope and admitted composition

### Matched denoising-step comparison (2026-10-30)

The user authorized four five-second BF16-rounded previews at 2/4/6/8 DiT
Euler steps, replacing the previous no-generation boundary for this bounded
experiment. Reuse the authenticated first-video input manifest (seed 42,
yellow-tram prompt), initial latent bytes, saved Qwen state-26 conditioning,
decoder noise, shift 5, CFG 1 and output geometry. Each arm starts from the
original latent with its own complete N-step schedule; an eight-step trajectory
prefix is not an N-step sample. PixelUMM remains the same one-step decoder in
all arms. Upstream documents the distilled DiT for 4/6/8 steps; two steps are
an experimental extrapolation, not a supported quality setting.

The existing eight-step BF16 preview already has visible artifacts. This
comparison tests sensitivity to DiT step count, not a presumption that fewer
steps caused them. Decoder limitations, native/upstream composition differences
and model quality remain competing explanations. Full upstream neural-video
parity is still unqualified.

The implementation keeps eight as the default and admits only 1..8 steps
before model setup. Its strongest failure mode is using a stale hardcoded
schedule/count or a different starting latent while labeling the result as a
matched comparison. Guard count-dependent admission, receipt and loop length;
verify every requested schedule and BF16 Euler boundary against the pinned
upstream solver before loading weights. In particular, six steps exercise
non-binary fractions not covered by the prior eight-step oracle.

DoD: bounded CPU admission/schedule regressions; `hy_nomas_euler_probe` against
fresh pinned-source MPS fixtures for all four counts; four successful serial
`hy_nomas_generate --dit-weights bf16 --steps N` requests with the same input
SHA; identical export settings, 120 finite nonconstant frames per arm and a
synchronized comparison artifact. Keep the 10% cache-aware reserve, 23 GiB
process cap, 64 MiB swap-growth stop and 900-second per-request timeout. Keep
the existing DiT and decoder terminal waits without additional encode-time
readbacks, waits or named-buffer allocations. Rollback is the default
eight-step invocation; no precision or production tuning policy changes.

The user authorized a real five-second preview before further kernel tuning.
The first bounded target is one first segment, 120 frames at 24 fps,
512x288, seed 42, CFG 1, eight shifted Euler steps (shift 5), camera/action
conditioning off. Its DiT grid is `[30,9,16]` (4320 tokens). PixelUMM uses the
pinned preset's ten-latent windows, five-latent overlap, no spatial cropping,
and the existing eight-token local attention layout on grid `[10,18,32]`.

BF16 remains the default DiT control. The user also authorized an experimental
Q4_K DiT preview with the identical pinned inputs and schedule for visual
comparison. PixelUMM and the global DiT adapters remain BF16. Q4 is decoded on
GPU into the existing single-block BF16 stage; it does not introduce another
wait or a CPU activation handoff. This experiment does not promote Q4 quality
or change the numerical tolerance. Same-decoded-Q4 numerical parity does not
establish acceptable image quality: the recorded Q4 versus
BF16 full-forward discrepancy is large. The first prompt may reuse the
authenticated saved Qwen state-26 crop; that boundary must be stated in the
run receipt rather than presented as live native text encoding.

## Existing anchors and missing composition

- The 54-block DiT, input/final adapters and text/time refiner are qualified
  separately on the saved first-segment fixture.
- The 36-block PixelUMM decoder plus head are qualified separately on real
  stem activations. That fixture is not a generated video.
- The missing production edges are GPU Euler integration, ordered repeated
  forwards with a changing GPU timestep, PixelUMM input stem, window decode
  and cosine merge, display-range conversion, and terminal MP4 export.
- Pinned upstream: `911cea6458a89c2663513ff7099561ec190dd491`.
  Existing source and weight authentication remain mandatory.

## Risk, resource boundary and rollback

CAUTION: this composes previously separate model phases and introduces repeated
use of fixed scratch. The strongest failure is stale/overwritten data from an
incorrect pass or window order, followed by plausible but invalid pixels.
Reject wrong ordering before appending work, retain all GPU owners until their
chain drains, and check finite output and nonconstant frames. Keep the old
single-forward loader contract as the default; repeated forwards are bounded
and explicit. New generation entry points stay opt-in.

Keep a cache-aware 10% physical-memory reserve, a fixed process cap, and a
64 MiB maximum system swap increase. Run heavy GPU tasks serially. Admit each
weight phase before allocation. DiT weights and decoder weights cannot simply
be added to the same residency budget. If a phase handoff requires draining a
chain to release weights, report that synchronization explicitly; retain the
small latent GPU buffer across the handoff and do not round-trip it through
the CPU. There are no waits or activation readbacks inside diffusion steps,
decoder blocks or window assembly.

Rollback is removal of the opt-in entry point and new glue, with the default
single-forward probes and current MPS baseline unchanged. Preserve unrelated
working-tree files. All temporary outputs and build products live on
`/opt/mldata_ssd/`.

## Definition of done and falsifiers

1. Run the scheduler/order specs and the new stem/merge oracle comparison at
   the unchanged `atol=rtol=2e-5`; nonfinite negative controls must fail.
2. Run a guarded native first-segment request through eight DiT forwards,
   all 36 decoder blocks and head for each of five windows, followed by GPU
   merge/postprocess. Record named allocations, encode readbacks, terminal
   waits, elapsed times, peak footprint, and guard verdict.
3. Export only completed display-range F32 pixels using
   `examples/hy_nomas_export.py`; verify 120 frames, 512x288, 24 fps, five
   seconds, finite and nonconstant decoded frames. Inspect a contact sheet and
   the actual clip. A valid MP4 alone does not certify generation quality.
4. Compare the newly composed numerical boundaries with pinned upstream on
   identical inputs. Keep execution parity, image quality and performance as
   separate claims. A failing boundary blocks a verified parity claim; never
   expand the tolerance to pass it.

The bounded native execution and terminal export below are VERIFIED within
their stated scope. Full-video upstream parity and general prompt support
remain open. No native generated clip was available at the start of this work.

## Boundary checks, 2026-10-28

The fixed input manifest is
`5b162b8bf4b28c6cfacd8a3d8c14a7758b699034b2a46e84c224522eb0a75a7b`.
It carries the saved natural-tram conditioning, MPS-generated seed-42 noise,
and pinned upstream RoPE tables for the preview geometry. This is a fixed
pilot, not a general prompt CLI. Initial F32 noise is rounded through BF16 on
GPU before the first DiT forward. The native DiT retains its qualified F32
arithmetic; the Euler boundary rounds model velocity and each updated latent
through BF16 as the source pipeline does.

The pinned `EulerSolver` module produces a 256-element MPS oracle under
Diffusers 0.36.0. Native initial rounding, all eight timestep values and the
final latent are bit-exact. The probe has one packed terminal diagnostic read
and one wait, with no preterminal activation read or wait. Its fixture SHA is
`3fce6402ec76036437dc0bdfee9f07008610bdca23c040c2bab55d339777d6de`.

The real-weight PixelUMM stem and synthetic five-window temporal assembly
also match the pinned MPS source bit-exactly, with a NaN rejection control.
The probe records zero encode allocations/readbacks/preterminal waits and one
terminal wait. The corrected oracle manifest SHA is
`3f6976cbd8d1efeeb3aa663b817980cba78b8d793e7ed56e90384278a535409f`.
An earlier oracle was invalid: large MPS int64 remainder operations corrupted
the synthetic inputs. Constructing the exact 8192-value period before repeating
it removes that oracle error. The source display conversion also requires
separate BF16 rounding after multiply and add, before F32 export. Neither fix
widens the tolerance. The earlier failing artifacts remain outside the repo.

The immutable DiT weight cache records its V5280 qualification geometry.
The preview authenticates that original cache geometry separately from its
V4320 activations. A CPU check permits only this video-token extent to differ
and requires identical parameter bindings; cache validation itself is unchanged.
Runtime buffers, plans, RoPE and memory accounting use V4320 throughout.

Exact named residency is 21,011,384,964 bytes for DiT and 16,563,852,432 bytes
for the decoder phase. The external guard uses a 23 GiB process-tree cap,
10% cache-aware reserve, 64 MiB swap-growth stop and 900 s timeout. These are
admission and sampled runtime guards, not guarantees against between-poll spikes.

## First completed preview, 2026-10-28

The accepted `generation-v3` process completed on Apple M2 Max, 64 GiB unified
memory. It generated the fixed prompt about a yellow tram in a wet downtown
street. `preview.mp4` contains 120 frames, 512x288, 24 fps and exactly five
seconds, independently checked with ffprobe and decoded to a contact sheet.
The sampled frames contain a recognizable tram, buildings and umbrellas;
the tram position changes over time. This inspection is not an upstream
quality comparison or a certification of temporal consistency.

| Phase | Measured wall time |
| --- | ---: |
| DiT preparation | 22.877 s |
| Eight DiT steps | 35.768 s |
| PixelUMM preparation | 22.221 s |
| Five decoder windows and display conversion | 67.547 s |
| Guarded process, including CPU admission and terminal file output | 155.672 s |
| MP4 export | 0.417 s |

This is about 31.2 seconds per second of video including startup and export,
or 20.7 seconds per second for the two neural execution phases. These are one
preview's measurements at 512x288, not a warmed benchmark, a result for 480p
or 720p, or a comparison with Wan or another backend.

The guard recorded 300 samples: peak process-group footprint 21,340,353,160
bytes (19.874 GiB), minimum cache-aware headroom 23,171,694,592 bytes
(21.580 GiB), swap growth zero, exit zero. Both neural phases report one
terminal wait and zero preterminal waits. Allocation/readback counters report
zero only for instrumented MetalBuffer methods; they do not measure opaque
MPS resources or the direct terminal pixel-file export. The receipt now states
that scope explicitly. The one weight-phase handoff is an intentional wait;
there is no CPU round trip for the latent.

The completed pixels have SHA256
`f13481b1d5790761df94943b6eac9e768177a240998c859d3ea9f21a633c2f85`;
the exported MP4 has SHA256
`7a323eb0b9e02443eedf45a5576f36a3c738ea41b687d1a476cbbdcfd0ea1cd3`.
All 120 display-range frames are finite and nonconstant. Frame standard
deviation ranges from 0.2316 to 0.2433; consecutive-frame mean absolute
differences range from 0.02968 to 0.06836. These statistics reject empty or
frozen output; they do not establish acceptable motion quality.

Artifacts live outside Git under
`/opt/mldata_ssd/cogni-ml/hy-nomas-first-video-20261028/`: `inputs.json`,
`generation-v3.log`, `generation-v3.guard.jsonl`, `results/receipt.json`,
`export.json`, `ffprobe.json`, `video-analysis.json`, `contact-sheet.jpg`, and
`preview.mp4`. The original receipt precedes the metric-scope wording addition;
its counter interpretation is the scoped one above. Earlier failed attempts
are retained separately. One stopped before loading weights on the recorded
cache-geometry mismatch; another completed GPU work but rejected Crystal's
unsupported `wx` file mode. The writer now uses O_EXCL and a CPU no-clobber
regression check. Neither failure was a successful video result.

Final checks: release builds, 39 Crystal examples, 13 Python tests, formatting
and diff checks; bit-exact upstream checks for the newly composed Euler and
stem/merge boundaries as described above. Correlated adversary review found
no enqueue-order/lifetime blocker and required the metric-scope correction.
This does not add a full generated-video upstream comparison.

## Experimental Q4 preview, 2026-10-28

The opt-in `--dit-weights q4-k` selector uses the authenticated existing
`HyNomasQ4Stack`, including its eight-forward order/chain limit. The default
remains `bf16`. Only DiT block weights differ: all pinned input bytes, global
adapters, Euler schedule, decoder noise, BF16 PixelUMM weights, and export
settings match the accepted BF16 preview. This is the staged Q4_K path, not
the separate direct-Q4 GEMM experiment. Compact block payloads occupy
4,593,162,240 bytes. Their GPU decode reuses one BF16 block stage; the complete
Q4 DiT phase admits 9,290,402,436 bytes instead of 21,011,384,964 for BF16.
The decoder admission remains 16,563,852,432 bytes and dominates the Q4 peak.

DoD: release build, focused selection/admission/order specs, CPU admission in
both modes, invalid-mode rejection before Metal, and one guarded complete
preview with the identical input manifest. Memory estimates must match actual
owner admission. A mode mismatch, input/schedule mismatch, added encode-time
readback/wait, guard stop or nonfinite output rejects the execution result.
Q4-versus-BF16 quality remains a visual experiment; it is not a passing 2e-5
parity test. Existing same-decoded-weight kernel gates retain that tolerance.

The Q4 run completed all 120 frames at 512x288/24 fps. Contact frames show a
recognizable tram but substantially more blurred and distorted buildings and
tram contours than the BF16 control. Q4 quality is not promoted. The raw output
is finite and nonconstant in all 120 frames, and ffprobe independently confirms
the frame count, geometry, frame rate and five-second duration. These checks
establish execution and export, not acceptable perceptual quality.

| Phase | Prior BF16 launch | Q4 launch |
| --- | ---: | ---: |
| DiT preparation | 22.877 s | 7.296 s |
| Eight DiT steps | 35.768 s | 48.976 s |
| PixelUMM preparation | 22.221 s | 22.313 s |
| Five decoder windows/display | 67.547 s | 85.906 s |
| Guarded process | 155.672 s | 168.534 s |
| MP4 export | 0.417 s | 0.311 s |
| Sampled peak group footprint | 19.874 GiB | 16.784 GiB |

These are separate single launches, not an interleaved speed experiment.
The unchanged decoder also took longer in the Q4 launch. Do not attribute the
whole timing difference to quantization or claim an acceleration. Faster DiT
preparation and lower resident bytes do not remove PixelUMM compute cost.
All weight files currently resolve through `/opt/mldata_ssd` to the external
SSD. Disk reads occur during setup; the block payloads are GPU resident during
encode. A local mirror could improve startup but has not been benchmarked.

The Q4 guard recorded 324 samples, peak footprint 18,021,842,016 bytes, minimum
cache-aware headroom 23,878,877,184 bytes, zero swap growth and exit zero under
the unchanged 10% reserve, 23 GiB process cap, 64 MiB swap stop and 900 s deadline.
There is one terminal wait per phase, zero preterminal waits/readbacks and zero
instrumented MetalBuffer encode allocations. Opaque MPS resources and terminal
pixel-file export remain outside the counter scope. The Q4 receipt explicitly
records the mode, experimental quantization, BF16 global/decoder weights and
Euler boundary rounding; full generated-video upstream parity remains untested.

Artifacts: `/opt/mldata_ssd/cogni-ml/hy-nomas-q4-preview-20261028/` contains
`launch.json` (source/binary/baseline digests), `plan-bf16.jsonl`,
`plan-q4.jsonl`, `specs.log`, `invalid-mode.log`, `generation.log`,
`generation.guard.jsonl`, `results/receipt.json`, `comparison.json`,
`export.json`, `ffprobe.json`, `contact-sheet-comparison.jpg`,
`preview-q4.mp4`, and `comparison-bf16-left-q4-right.mp4`.
Q4 pixels SHA256:
`dc7a80be9c802b134e91ae784f7be0c05a5bae398cdd1fdd7e8cfe2d6cff94ae`.
Q4 MP4 SHA256:
`fb4cfacc8f7a8abd2be264e9a2f083dad2bbe6c6837afa572b62a505f3434074`.
The focused suite passed 21 examples. Correlated adversary review found no
composition/lifetime blocker; it required explicit Euler-rounding wording.
The final artifact review returned ROBUST for matching input/schedule/decoder
identities, guarded execution and finite output; it does not qualify visual
quality, full upstream parity or a speed improvement.

## Duration boundary

The native configuration explicitly admits one first segment and rejects
history/multiple segments. The current input, latent, RoPE and assembly are
fixed to 120 output frames. Five overlapping decoder windows are not five
autoregressive segments. Upstream implements segment continuation with K/V
history and temporal RoPE offsets; that state contract is not ported here.

Pure division of the first BF16 launch gives about 1 minute 55 seconds of
independent five-second output per wall-clock hour at 512x288. Dividing only
the neural phases gives about 2 minutes 54 seconds per hour, excluding setup
and requiring a future reuse design. Neither number is sustained throughput
or a maximum continuous duration. In particular, two continuous minutes
cannot yet be requested from this entry point. Long-duration timing and memory
must be measured after continuation is implemented; history adds work and
memory absent from the present preview.

## Next optimization frontier

The Q4 preview was visually rejected by the user. BF16 remains the quality
control and the active performance target; do not promote Q4 on memory savings
or projection-level numerical tests alone.

PixelUMM accounts for 65.4% of the two neural phases. Weight authentication,
loading and plan construction account for another 45.1 seconds of the native
request and should be separated before proposing a persistent owner. Do not
remove integrity checks to make startup appear faster.

The next exact-path probe should group independent same-sized local-attention
tiles into batched MPS graphs while retaining each tile's context, BF16
boundaries and reduction order. First measure the current 5760-token window
by FFN, projections, attention and command transitions; a reduction in API
calls alone does not establish a full-window speedup. Compare identical inputs
against the current implementation and pinned source at the unchanged 2e-5,
then measure the complete decoder window. This is PROPOSED, not implemented.
The overlapping windows decode 200 frame positions to produce 120 output
frames, but later-layer overlap activations depend on different attention
contexts and cannot be reused solely because their frame indices match.

The bounded block-zero probe now admits T10/H18/W32 (J5760), the preview's
window geometry, including all 26 intermediate captures. It uses actual
pretrained weights and seeded synthetic hidden states, not the generated
clip's activations. The unchanged native implementation matches all 26 tensors
bitwise against the pinned upstream source under PyTorch 2.6.0 MPS at the
unchanged 2e-5 gate. The existing decoder-head control and poisoned alternate
replay also match all five tensors bitwise. Each request retains one terminal
wait, zero instrumented encode-time named allocations and zero preterminal
activation readbacks. All launches passed the cache-aware 10% floor, process
footprint cap and 64 MiB swap-growth guard. Evidence is under
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-bf16-perf-20261028/results/`:
`native-torch26-all.json`, `native-torch26-admission-all.json`,
`head-torch26-original.json` and their guard logs.

An initially mixed PyTorch 2.8 oracle changed inverse-root lowering and falsely
suggested a native RMS regression. A proposed RMS replacement matched that
oracle but failed the existing 2.6 head control; it was reverted. Preview MPS
fixture production now rejects a runtime other than 2.6.0 before GPU work and
records interpreter/package provenance. The native J5760 consumer also rejects
other runtimes or CPU fixtures before payload access and reports the reference
producer's preflight separately from its own memory guard.
A runtime migration requires a separate
consistent block/head lineage, not a relaxed tolerance.

The J5760 layout has 12 query tiles and two distinct ordered key lists. A
candidate reduced K/V gathers from 24 to four while preserving intermediate
results in the isolated 2.8 experiment. Four ABBA launches gave profile-off
full-block medians of 0.515/0.531/0.571/0.594 seconds; GPU drift dominated the
roughly 0.6% balanced difference. No reliable speedup was established, so that
candidate and the experimental RMS code remain outside production. Preserve
the rank-three batch-32 matmuls and rank-four softmax axis 3. The retained
external patch and reports document the rejected route, not a production gain.

### Measured BF16 startup improvement

The BF16 entry point previously authenticated all Q4 payloads three times
through the shared weight index before computing with separately authenticated
BF16 controls. It now creates one request-scoped `HyNomasQ4StackIndex::Admission`
and refreshes its captured file identities, sizes and modification times at the
two later admission boundaries. Construction still performs the original full
index/manifest/payload hash and inventory checks. Every consumed BF16 control
and the globals still receive their full SHA-256, inventory and direct-copy
checks. Q4 inference retains its full loader path and rejects the reusable BF16
admission. No GPU arithmetic, dispatch, synchronization or precision changed.

The freshness guarantee is stat-based: replacement, removal or an mtime change
rejects across all five file classes. A same-inode, same-size Q4 byte mutation
with restored mtime can evade refresh; this is not fresh content authentication.
The BF16 runtime does not consume those Q4 bytes. The corresponding same-stat
mutation of a consumed BF16 control still fails its full hash before parsing
or Metal allocation. Reuse is local to this request, not a persistent trust
cache or authorization for skipping BF16 hashes.

A native Crystal CPU benchmark compared three complete index loads with one
complete admission plus two refreshes, including the first full scan in each
case. It used the same 54 ordered blocks and 4,593,162,240 Q4 payload bytes,
one untimed full-scan warmup, no cache purge, and ABBA followed by BAAB. Across
four timed cases per arm, medians were **6.136654 s versus 2.055800 s**, a
2.985x ratio and 4.080854 s difference for this authentication portion. Both
order groups retained approximately 3x ratios. The last two cases slowed
together (10.050/3.427 s); these are warm-cache startup observations, not a
cold-start guarantee or a GPU/complete-video speedup. All identity fingerprints
matched and tracked Metal allocations were zero. The cache-aware guard passed
the 10% floor, 2 GiB process cap, 64 MiB swap-growth stop and 180 s timeout:
85 samples, 8,470,936 B sampled peak, no swap growth.

The real supplied-admission BF16Stack constructor also authenticated all 54
BF16 controls (16,314,144,768 payload bytes), admitted exact raw-copy plans and
released with no Metal allocation. The release preview binary passed
`--plan-only` on the pinned input, reporting BF16 selection and the unchanged
21,011,384,964/16,563,852,432 B DiT/decoder named budgets. No generation ran.
The receipt now records `preflight_seconds` and `request_seconds`; the latter
includes preflight and input upload through pixel-file export. Historical
`total_seconds` continues to start after preflight/input upload, so compare
like scopes. External MP4 encoding and process startup remain excluded.

DoD: release `make hy_nomas_generate`; CPU-only specs
`hy_nomas_q4_stack_spec.cr`, `hy_nomas_generation_admission_spec.cr` and
`hy_nomas_bf16_serial_admission_spec.cr` (23 examples passed); native
`make spec_hy_nomas_bf16_cache` (18 examples passed); real guarded authentication
benchmark and BF16 control admission; guarded preview `--plan-only`;
format/diff checks. An initial CPU-only run of the broader BF16 cache suite hit
two existing nested-`pending` errors in its Metal-required cases; those cases
passed in the proper native suite. No unrelated skip/stub code was changed.
Evidence in the same external namespace: `tmp/authentication_benchmark.cr`,
`results/authentication-paired.jsonl`, `authentication-summary.json`,
`authentication-bf16-controls.jsonl`, `generation-plan-admission.jsonl` and
their guard logs. The summary pins benchmark source/binary and preview binary
hashes. Correlated mini-Quad verdict is ROBUST for this startup change and its
stated stat boundary; unchanged block/head math retains the numerical control
above. This does not requalify complete-video quality or throughput.

### Bounded FFN weight staging: exact, no established speedup

Each independent decoder window currently feeds BF16 FFN weights into graphs
containing F32 casts. Amortizing those casts across five windows was a possible
lever; the bounded two-window experiment below did not establish a gain.
Simple loop reordering is
invalid: prepared graphs bind fixed inputs/outputs and the shared two-bank
workspace cursor requires complete slot-ordered passes. A block-major runner
needs separate window activation banks and explicit per-window graph bindings.
One block's three F32 FFN matrices require 576 MiB of staging; retained current
and next states for five windows require 450 MiB, about 315--360 MiB more than
the current stem/output banks. Opaque MPS resources need separate measurement.

The diagnostic compares two J5760 windows through blocks 0 and 1: ordinary
BF16 graph feeds against block-major reuse of one 576 MiB F32 staging set.
It supports both window-major and block-major baseline ordering so a same-order
control can isolate staging from loop reordering.
Each window/block has a distinct output binding; all four plans share transient
scratch and both graph variants remain resident. GPU conversion of all three
FFN matrices is included once per block in the staged timing. Staging must not
alias weights, live window states, captures or metrics, and the next block may
overwrite it only after both windows have consumed the current block.

Inputs are overlapping slices of the saved preview initial-noise conditioning,
passed through the actual pretrained upstream stem, at latent starts 0 and 5
(noise-frame starts 0 and 20). They are not final generated DiT latents. The
fixture must retain this provenance limitation and exact source, payload and
weight identities. Local temporal RoPE resets in each upstream decoder window.

DoD: build `pixelumm_ffn_weight_reuse_probe`; pass its CPU schedule/admission
specs and strict GPU comparison of post-attention normalization, gate, up,
product, down and output for both windows/blocks against pinned Torch 2.6.0
upstream at unchanged `atol=rtol=2e-5`; reject poison/known mismatches and verify
staged weight words exactly. Only then run four ABBA warmups and eight timed
ABBA/BAAB requests with intermediate capture copies disabled, staging included,
one terminal wait and one packed metrics read. Track named and opaque resources,
fresh cache-aware admission, the 10% reserve and the 64 MiB swap-growth stop.
No full generation or production schedule change is admitted by this probe.

Risk: stale fixed graph bindings or staging overwrite could silently mix
windows/blocks; distinct retained outputs, schedule guards, poison and per-boundary
upstream checks are the falsifiers. Rollback is removal of the new diagnostic
target and sources; production math and defaults remain unchanged.

Both arms matched all 24 upstream captures bitwise. The staged correctness
request additionally checked 603,979,776 expanded F32 weight words across its
four consumers, with no differences, failures or nonfinite values. Poisoned
buffers were refreshed before every request and the NaN positive control fired.
Each launch used two correctness requests, four ABBA warmups and eight timed
ABBA/BAAB requests, four per arm. Timed requests disabled intermediate capture
copies and exact-weight checker kernels, included both block conversions,
and retained four terminal output comparisons. Every request had one terminal
wait, one 464-byte packed metrics read, zero tracked encode-time MetalBuffer
allocations and zero preterminal readbacks. These counters do not prove zero
Crystal or Objective-C allocations.

Profile-off request medians on Apple M2 Max, in seconds:

| Baseline order / run | Ordinary BF16 graph | Shared F32 staging | Wall reduction |
| --- | ---: | ---: | ---: |
| Window-major / initial | 1.459559 | 1.465904 | -0.435% |
| Block-major / first control | 2.319832 | 2.312451 | +0.318% |
| Block-major / corrected report | 1.700959 | 1.710128 | -0.539% |

The initial comparison also changed scheduling. The two same-order controls
have mixed signs and substantial process-to-process drift; the corrected run
was slower in three of four adjacent pairs. Its body-encoding median fell
from 0.793150 to 0.628362 seconds while terminal waiting rose from 0.907483 to
1.086075 seconds. Encoding overlaps GPU work and may block inside Metal/MPS;
this is not evidence of a corresponding CPU saving or faster inference.
The measured GPU interval is the command-chain span, not isolated kernel time.
Request wall time also includes common poison/comparators, terminal readback
and a memory sample. No reliable staging speedup or default promotion follows.

The named live inventory was 7,381,500,468 bytes, including the 603,979,776-byte
staging set and 2,264,924,160 bytes of expected captures. Native launches passed
the cache-aware 10% reserve, 16 GiB footprint cap, 64 MiB swap-growth stop and
600-second timeout. Across them, sampled peak footprint was 8,182,353,184 bytes
and swap growth was zero. The corrected launch retained at least
37,270,994,944 bytes of cache-aware headroom. Upstream production passed its
separate 12 GiB cap with a 5,910,612,224-byte sampled peak and zero swap growth.
Named bytes, GC heap, RSS, footprint and Metal device allocation overlap and
must not be added; opaque MPS bytes were not independently estimated. These
diagnostic capture inventories do not forecast full-pipeline memory.

The first two reports accidentally retained an alias to the native handle
array, which release then cleared. Preparation guards and nested resource
metadata still verified four distinct plans. The runner now snapshots the
array before release; the corrected run also verifies all four top-level
handles. Earlier reports and their measured source/binary remain preserved
under `measured-source-v1/` and `results/summary-v1.json`, without rewriting
the original evidence.

Evidence root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-ffn-staging-20261028/`.
`results/summary.json` SHA-256 is
`8905d2e45b8ac37f14ff5c2c9b84c9de7ede986dc5b4cb4415d6dfbe80123bec`;
it pins execution sources, binaries, the upstream fixture, raw reports and
guards, including the preserved earlier lineage. Fixture metadata SHA-256 is
`e9b1ddce57fac94efb0262bd3e5f0be2963ef9df4255dc74cf254ff4b0217ddc`.
Release build, eight Crystal CPU examples, nine Python CPU tests, format/diff
checks and the guarded numerical/timing replays passed. Correlated mini-Quad
adversary verdict is ROBUST for this fixed two-window/two-block numerical reuse
boundary; a speedup claim remains unsupported. No full generation ran.

The next proposed exact-path falsifier is batching independent windows' FFN
rows, following Qwen's prefill GEMM routing, while keeping attention contexts
and local RoPE separate. Another small candidate is fusing SiLU/product while
retaining the intermediate BF16 rounding in registers. Current shared scratch,
direct block-to-block GPU handoff and prepared graphs already reuse the Qwen
lifetime pattern. Qwen decode GEMV bandwidth gains and GDN affine recurrence
parallelization do not directly apply to J5760 PixelUMM FFN/softmax attention.
None of these proposals has a speedup certificate here.

Refresh all execution claims after changes to source, weights, input identity,
geometry, precision, schedule, command ordering, backend/device or guard.
Each launch requires fresh memory admission; missing external artifacts reopen
the corresponding numerical or performance claim.

### Bounded PixelUMM drift localization (2026-10-28)

The GC experiment below observed a common GPU-chain increase from about 2 to
4.6 seconds, while its single natural collection lasted about one millisecond.
The diagnostic keeps the same J5760, two-window/two-block BF16 inputs,
four resident Plans, shared scratch and window-major schedule. Compare ordinary
unprofiled requests with interleaved requests using the existing native phase,
Metal/MPS API and continuation collectors. Profiling must not split roots or
add commands, waits, tensor transfers or conversions.

This is CAUTION diagnostic work. The strongest false positive is assigning
Metal/MPS submission backpressure to CPU computation, or summing overlapping
root intervals as independent GPU phase costs. Preserve shared root candidates
at phase and plan boundaries. Keep graph-inclusive and nested continuation
times separate, and report whole-chain interval union and envelope. A
profile-on/off comparison is required because CPU instrumentation can perturb
submission timing. No production optimization is admitted by this probe alone.

DoD: release-build `make pixelumm_drift_probe`, pass its focused CPU accounting
tests, and run two correctness requests, four warmups and 96 balanced timed
requests under the existing memory guard: 16 GiB process-group footprint,
10% cache-aware reserve, 64 MiB swap-growth stop and 600-second timeout. Both
correctness requests must pass all 24 pinned upstream captures and the existing
NaN positive control at unchanged atol=rtol=2e-5. Each request retains one
terminal wait and one packed metrics read; tracked encode-time buffer
allocations and preterminal readbacks remain zero. CPU-only profile extraction
follows that wait. Full generation is excluded. If drift does not recur, report
its non-reproduction instead of inventing a causal attribution.

Rollback removes this diagnostic and its optional probe-only profiling hook;
production math, synchronization and GC defaults remain unchanged. Refresh the
evidence after changing sources, inputs, geometry, precision, scheduling,
backend/device or guard settings.

Observed result: all 102 requests completed (two correctness, four warmup,
96 timed; 48 timed/profile arm) in a 167.121-second measured loop. The original
4.6-second severe drift was not reproduced; its preceding GC experiment ran
454.663 seconds. This shorter run reproduced a smaller chronological increase
in both arms. The last GPU root-duration sum was 2.076724 seconds. First/last
16 timed-request medians, including both profile arms:

| Observation | First 16 | Last 16 |
| --- | ---: | ---: |
| Sum of GPU command-root intervals | 1.405981 s | 1.862077 s |
| Host encode wall | 0.676150 s | 0.877546 s |
| Host encode thread CPU | 11.095 ms | 9.218 ms |
| Terminal wait envelope | 0.751987 s | 1.004449 s |

The normal-path terminal wait includes encoder closure. The native GPU elapsed
value is the sum of command-root intervals, not exclusive kernel-active time
or the whole-chain envelope. In the 48 profiled timed requests, the median
interval-union gap was 74.4 us (maximum 88.2 us), with 1.63 ms median overlap
between roots. This rules out large gaps *between those root intervals* for
this run; it does not measure stalls inside them or establish GPU utilization.

The same root topology, 125 roots/request, was retained in both arms. Comparing
the first/last eight profiled timed requests, nested `commitAndContinue` wall
time increased as follows. These call spans identify the CPU-side location of
waiting and must not be added to inclusive MPSGraph time or treated as the
exclusive GPU cost of the named phase.

| Metal/MPS API phase | First eight | Last eight |
| --- | ---: | ---: |
| FFN | 273.532 ms | 381.803 ms |
| Self-attention preparation | 158.545 ms | 210.163 ms |
| Self-attention | 226.477 ms | 270.613 ms |
| Self projection/residual | 7.663 ms | 9.128 ms |

FFN continuation wall overlaps earlier submitted roots for a median 99.983%
of call time early and 99.988% late (per-request wall-weighted fractions).
The root being submitted by the current call is excluded from this calculation.
Root intervals whose sole marker-candidate phase kind is gate/up also increase
from 0.400821 to 0.566966 seconds; shared boundary roots retain every candidate.
These are root durations rather than exclusive kernel timings. FFN is the
largest measured continuation-wait contributor, while several phases grow.
The evidence supports descriptive localization to widening command-root
intervals and Metal/MPS call waiting; the underlying cause remains open.

Instrumentation control: the 48 adjacent on/off pairs have median GPU
root-duration-sum change +0.0808% (range -3.292% to +5.868%) and total host wall
change +0.1036%. Encode thread CPU changes +8.425%, so the observer is not
neutral. Both arms drift, but these results do not establish equivalence or a
production performance change. Three natural collections occur after the wait,
during profile extraction: 2.070416 ms total collection envelope, 1.829666 ms
rendezvous-inclusive pause, 1.809873 ms fully stopped, overlapping categories.
They cannot account for the observed seconds-scale change. Advisory thermal
state changes from nominal to fair at request 50, after drift has started;
this is not a GPU clock measurement or evidence of thermal causation.

Numerical and guard DoD passed: both correctness requests match all 24 saved
upstream Torch2.6.0 captures bitwise, all timed output comparisons pass, and
the NaN positive control fires at unchanged atol=rtol=2e-5. Every request has
one terminal wait, one 464-byte packed read afterward, zero preterminal reads,
zero tracked encode MetalBuffer allocations and zero stage conversions.
No GPU math, command geometry, production defaults or GC policy changed.
Sampled peak footprint was 8,180,731,168 bytes (7.62 GiB), minimum cache-aware
headroom 32,139,329,536 bytes, minimum pressure-free 80%, with zero positive
swap growth across 360 guard samples. Categories overlap; the diagnostic's
7,381,500,468 named bytes include an unused 576 MiB stage and are not a full
pipeline memory forecast. Correlated adversary verdict: ROBUST for this bounded
descriptive localization; causal attribution and extrapolation to 4.6 seconds
remain unverified.

Reproduction: `make pixelumm_drift_probe` builds the native diagnostic;
keep `BUILD_DIR` and `TMPDIR` under the external evidence root for builds.
`make spec_pixelumm_drift_probe spec_pixelumm_block_profile` passes 36 CPU
examples, and `python3 -B -m unittest spec/pixelumm_drift_analysis_test.py`
passes eight accounting/guard falsifier tests. The existing GC observer and
reuse suite passes its real-collection qualification and 12 CPU examples.
Use the guarded argv in the external execution receipt, with fresh report and
guard paths and fresh memory admission, rather than running the binary bare.
Run the CPU analyzer with its report positional argument, `--guard LOG` and
`--output FRESH_JSON`. Source/binary snapshots, raw report, guard, validated
analysis and a compact independently recomputed index are under
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-drift-20261028`; the index is
`results/summary.json`. The raw report SHA256 is
`e9758ab1c917493afad4e7960147f7f86fb6a2adc46a42b6c8e75ff13605ecc4`.
The compact index SHA256 is
`b7f254bf9b29879963309a68be1241f519088f8a88683e11f363db5d59a46d88`;
it pins the final analyzer, execution receipt, binary/source lineage and raw
guard/report hashes. Earlier exploratory analyzer summaries are not the
selected evidence.

Next falsifier for the severe drift: a longer matched serial exposure that
actually reproduces the hump, with direct device frequency/power/utilization
telemetry and a check of profiler intrusion. Separate system competition and
device state from per-Plan resource behavior before changing GC or command
scheduling. This diagnostic alone does not select a production optimization.

### Bounded host-allocation and GC experiment (2026-10-28)

Compare normal Boehm GC with GC disabled only from request admission through
the existing terminal GPU wait. Use the authenticated two-window/two-block
J5760 BF16 diagnostic above, with both arms executing the ordinary baseline
graphs in window-major order. A standalone driver is required: GC disabling
is process-wide and is not admitted in Studio or concurrent inference here.
Restore GC in an exception-safe scope, before metrics, teardown and reporting.
Keep those costs and following-request collection debt visible in end-to-end
time; a shorter encode interval alone cannot qualify a speedup.

This is CAUTION diagnostic work. The strongest false positive is shifting a
collection outside the disabled interval or mistaking Metal/MPS backpressure
for GC. Record allocation bytes, collection events and rendezvous-inclusive
and fully stopped intervals at existing CPU phase boundaries, without new GPU
waits or tensor readbacks. Qualify the allocation-free native observer with
real collections in a separate CPU-only positive control. Refuse an occupied
collector callback, unsupported GC or an already disabled collector. No forced
collections or artificial allocation pressure belong in measured GPU requests.
Zero observed collections after warmup cannot establish a general GC verdict.

DoD: release-build `make pixelumm_gc_probe`, run its CPU lifecycle/observer
qualification, then run two correctness trials, four warmups and 64 balanced
timed trials, each followed by one matched normal-GC tail (140 requests total),
under `examples/support/wan_memory_guard.py` with a 16 GiB
footprint cap, 10% cache-aware reserve, 64 MiB swap-growth stop and 600-second
timeout. Expected signal: both arms pass all 24 pinned Torch 2.6.0 captures at
unchanged atol=rtol=2e-5; the NaN positive control fires; each request retains
one terminal wait, one packed metrics read and zero tracked encode-time named
MetalBuffer allocations/preterminal reads. Numerical checking stays on GPU.
Report setup and diagnostics separately from the bounded inference body.

Rollback removes the new diagnostic and restores the previous diagnostic's
default request-count gate. Production inference, precision, synchronization
and GC policy are unchanged.

The guarded native run completed all 140 requests in 474.119 seconds, including
admission/setup/reporting. The observed request loop was 454.663 seconds.
Both correctness primaries matched all 24 saved Torch 2.6.0 captures bitwise;
the unchanged tolerance gate and NaN positive control passed. All requests
retained their single terminal wait and 464-byte packed metrics read, with zero
tracked encode-time named-buffer allocations or preterminal reads. No full
generation or production policy change ran.

The 64 timed primaries used ABBA/BAAB repeated eight times, 32 per arm. Results
below are descriptive because both arms experienced large common drift:

| Metric | Ordinary GC | Scoped `GC.disable` |
| --- | ---: | ---: |
| Primary wall median | 3.516415 s | 3.560862 s |
| Primary plus ordinary-GC tail median | 7.019055 s | 7.102461 s |
| Boehm allocation bytes per primary, median | 22,056 B | 24,624 B |
| Collections in timed primaries | 0 | 0 |
| Collections including matched tails | 0 | 1 |

The median paired `(normal - disabled) / normal` difference was -0.9701% for
primaries and -0.4741% including tails; disabled won 13/32 pairs. This is no
established speedup, slowdown or equivalence. Keep ordinary GC enabled.

The entire loop allocated 3,301,280 Boehm-managed bytes. Its heap remained
7,159,808 bytes at both endpoints. Exactly one natural collection occurred in
trial 56's ordinary-GC tail after a disabled primary, during encoding. Its
collector envelope was 1.104083 ms, rendezvous-inclusive pause 0.888000 ms and
fully stopped interval 0.840875 ms. These windows overlap; do not add them.
This one event cannot characterize a pause distribution or establish a causal
transfer of GC debt. The ordinary timed arm had no collections to contrast.
Allocation volume covers this diagnostic's process-wide Boehm activity, not
allocation counts, production inference or Objective-C/Metal allocation bytes.

Primary GPU command-chain spans also drifted: ordinary-GC early/late medians
1.950924 -> 4.650484 s, disabled 2.051612 -> 4.618436 s. Body encoding and
terminal waits grew together. The reported GPU span includes queue/gap effects;
it is not pure kernel active time, and encoding includes Metal/MPS backpressure.
The observed GC event cannot explain this seconds-scale drift. Its cause is
still unlocalized; further bounded GPU/MPS/queue-state measurement is a better
next falsifier than removing Crystal's collector or rewriting the runtime.

Sampled peak process-group footprint was 8,180,682,040 bytes (7.62 GiB), minimum
cache-aware headroom 25,697,992,704 bytes (23.93 GiB), with zero positive swap
growth and no guard trigger across 879 samples. The named inventory stayed
7,381,500,468 bytes, including an unused 576 MiB F32 staging set inherited from
the diagnostic; it is not an optimized production memory forecast. Limits are
sampled, and file-cache headroom remains a reclaimability estimate.

CPU qualification passed real collection/suppression/callback-ownership tests,
12 Crystal examples and 7 Python analyzer tests. An uncontended 10,001-point
CPU sampling qualification allocated zero bytes, observed no collection and
averaged 0.111 microseconds/point. It does not measure contended GC locks;
instrumentation costs were not subtracted. Release build and format/diff checks
passed. The correlated adversary verdict is ROBUST for these bounded numerical
and measurement claims, not a general GC-policy or production performance claim.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-gc-20261028`.
`results/summary.json` SHA256
`3dff8dbcc9f413ba20206bcb111b08a013e1ceb639839e485c0e0834823790f5`
pins the raw report, guard, execution receipt, analyzer v2 and qualified sources.
Measured executable sources/binary are unchanged; the subsequent Makefile edit
only adds transitive build prerequisites. Request phase deltas are recomputed
from raw points; trial-inclusive values have lower/outer bounds rather than
exact endpoint reconstruction because those two points were not serialized.
The single ordinary tail exposes immediate carryover only, not all future GC
debt. Refresh on measured source, fixture, weights, runtime, geometry, precision,
schedule, backend, device or guard changes, or artifact loss; admit memory afresh
before any GPU replay.
