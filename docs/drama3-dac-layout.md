# DRAMA-3 DAC weight layout

Source base `fc717747`, Apple M2 Max, 2026-10-16. The user target is sustained
RTF 0.8–0.9, with first-audio latency tracked separately. One 2048-sample frame
at 44.1 kHz represents 46.440 ms of audio; that target permits 37.152–41.796 ms
of generation work per frame. Earlier measured backbone/tower decode alone
takes about 196 ms/frame. A DAC improvement cannot by itself close that gap.

## Change and rollback

Adjacent Conv1d GPU lanes produce adjacent output channels, but the original
`[out, in, tap]` effective-weight layout separates their loads by `in * taps`.
The candidate stores the same F32 effective values as `[in, tap, out]` inside
the existing weight-normalization apply pass. The new Conv1d reads those
contiguous values while keeping bias initialization, input-channel/tap loop
order, multiply/add order, causal padding, dilation and output shape unchanged.
There is no new allocation, normalization pass, command, completion wait or
inference activation read. The scale reduction is unchanged.

`DRAMA3_DAC_CONV_LAYOUT=native|coalesced` is validated and frozen when a
`CodecResident` is constructed. `coalesced` is the qualified default. Only
weight-normalized Conv1d with one group
is eligible. Grouped/depthwise Conv1d, ConvTranspose1d and linear kernels retain
their original layout. `native` is the rollback. Precision and upstream
tolerances remain unchanged; this is ordinary memory-layout optimization.

## Controls

Heavy builds and GPU runs are serial under `scripts/run_safe.sh`, a 15% free
memory floor and no quiet-host requirement. Other projects remain running.
Weights remain local; builds, raw inputs and output artifacts live in
`/opt/mldata_ssd/cogni-ml/drama3-realtime-20261016`.

The fixed full-request control is “Hello, this is a voice test.”, seed 1,
temperature 0.8, top-k 30, top-p 0.9, no voice reference, maximum 64 frames.
It stops on EOS at 60 frames: 122880 samples, 2.786394558 s. Complete code
frames, sampler inputs, WAV bytes and graph/read ledgers must match the saved
native baseline. Request wall includes report serialization and the existing
between-job GC; model loading and excluded cold requests are separate.

The added `measure_host_pairs.py --dac-layout-schedule` changes only that
selector, keeping scratch, profiling and timeline off, both dense selectors
at `tree`, affine expert width 64, native expert-down, MAP_PRIVATE and queue
depth 2. Warm results in one worker are correlated. Standalone DAC timings
also exclude observer/comparator dispatches.

## Numerical evidence

Seven focused Metal examples pass, including odd channel/output tails,
96-channel residual geometry, dilation, layout freeze and invalid settings.
The direct geometry guards compare the remapped effective-weight bits and
output bits on GPU, including a one-bit negative control, and read one 20-byte
receipt per geometry. The existing terminal whole-codec comparison also covers
the depthwise fallback. An unset selector chooses `coalesced`; an explicit
`native` selector remains frozen even if the environment later changes.

On the pinned 60-frame input, all 58 DAC stage arrays (157,782,720 values)
compare bitwise, with zero nonfinite pairs. The comparator negative control
detects two bit mismatches and one nonfinite pair. Timed decodes contain no
comparison dispatch or readback; the later diagnostic decodes read only
464 bytes of stage counters plus 8 bytes for the negative control. Each decode
retains its 32 existing submissions/completions/waits. The complete standalone
probe has 385, including its one isolated negative-control command.

The first standalone report is excluded from numerical evidence: a lazy
`each_with_index.map` iterator was consumed by the first total, leaving the
nonfinite total and JSON stage list empty. The corrected probe materializes
the results, requires exactly 58 entries, and exits nonzero on either mismatch.
Only the rebuilt `dac-layout-bench-r2` report qualifies the stage comparison.

On the pinned MLX 197-frame positive fixture, both layouts reproduce all 58
upstream comparison metrics exactly, the same WAV SHA256 and the same graph
and read ledgers. Each has 32 submissions/completions/waits and zero pending
commands. These observer-enabled times are correctness controls, not the
primary performance measurement.

The full upstream DAC gate remains red in both arms, first at
`quantizer.post_module.layers.0`, with maximum PCM error 16. This reproduces
the existing numerical frontier. Identical metrics and native parity establish
no new regression, not full native/MLX compatibility. No tolerance is widened.

## Measured speed

The observer-free standalone ABBA/BAAB probe, after one excluded prime per
layout, gives median DAC decode 6.298823 s native versus 2.841204 s coalesced:
2.217x for the same 2.786395 s of audio. This is DAC alone (RTF 2.261 to 1.020),
not complete generation.

The full-request ABBA matrix uses four fresh workers, each with an excluded
cold request and two warm requests. Per-worker warm medians are the units:

| Worker | Layout | Request seconds | Generation seconds | DAC seconds |
| --- | --- | ---: | ---: | ---: |
| 0 | native | 31.060417 | 22.974521 | 8.015513 |
| 1 | coalesced | 22.872645 | 20.035846 | 2.791956 |
| 2 | coalesced | 23.390653 | 20.608236 | 2.744284 |
| 3 | native | 26.928595 | 20.608228 | 6.260869 |

The two mirrored request ratios are 1.35797x and 1.15125x; DAC ratios are
2.87093x and 2.28142x. The median of worker medians is 28.994506 to 23.131649 s
for the whole request (1.25346x; RTF 10.406 to 8.302), and 7.138191 to
2.768120 s for DAC (2.57871x). All 12 requests retain identical complete codes,
WAV bytes, sampler inputs and read/wait ledgers. Generation itself is unchanged
by this optimization; its observed timing shift is a host/order confound.
ABBA mitigates linear drift, not abrupt contention or paging changes.

The consistent DAC gain, both full-request pair wins and exactness controls
justify the default promotion. This remains far from sustained real time.
The rebuilt Studio with no layout environment override passes a fresh cold
and warm request against the original saved native codes/WAV/tape/read/wait
baseline (33.493505/26.045502 s; the warm DAC is 2.662282 s). This validates
the actual constructor default, not another speed pair.
The next generation-kernel discriminator is packing several MoE output rows
into a 64-thread group, preserving the native per-group and final sum order.

## Rejected packed-Q4 load candidate

The separate expert probe uses actual captured layer-39 gate/up inputs and
per-rank down inputs, with the same eight expert IDs. It replaces only the
packed-word byte assembly with an aligned word load and an odd-offset fallback.
All 7,077,888 native output values compare bitwise on GPU. The negative control
detects two bit mismatches and one nonfinite pair; all eight alignment offsets
pass. Only 336 bytes of final counters are read, with 81 diagnostic commands
and completion waits and no intermediate activation read.

Seven measured pairs per projection, after two excluded warmup pairs, yield
median paired native/candidate GPU command-duration ratios 0.996851 gate,
0.999277 up and 0.994630 down. Candidate wins are 2/7, 3/7 and 3/7. There is
no sustained gain. Command intervals overlap and are not additive kernel
costs. Production retains the original loader; compiler load folding remains
an unproven explanation for the neutral outcome.

## Evidence lifetime

Raw commands, source and binary hashes, environment controls, exit codes and
memory observations are in the external command receipt. Join build and run
records with the binary hashes; runtime source hashes alone do not identify
compiled production code. Same-lineage mini-Quad review is correlated, not
independent execution. Refresh on changes
to kernels, decoder selection, compiler, model, input codes, reference, queue,
hardware, OS or host workload. Timing results describe this active host and
fixed trajectory; they do not promise real-time or long-window performance.

The curated index is `.landmark/evidence/drama3-dac-layout.json`. Its immutable
external `dac-layout-commands.json` joins all 13 DAC build/run records and raw
log hashes; it excludes the broken first standalone numerical report.
