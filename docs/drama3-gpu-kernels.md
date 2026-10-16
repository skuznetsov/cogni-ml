# DRAMA-3 GPU execution and command-wait attribution

Source base: `22c99628`, Apple M2 Max, 2026-10-16. This slice separates GPU
execution from existing command waits before evaluating a dense-kernel change.
Model arithmetic, precision, sampler inputs and upstream tolerances are fixed.
Streaming remains a design branch in `drama3-native-frontier.md`.

## Measurement boundaries

`DRAMA3_METAL_TIMELINE=1` now records host-clock bounds around the existing
commit and wait calls, then reads `GPUStartTime` / `GPUEndTime` after that same
completion wait. It adds no command, encoder, completion wait, activation read,
or sampler draw. Logging happens outside the measured completion-wait scope.
Completion status remains authoritative: missing timestamps or a failed
diagnostic callback cannot turn a successful command into an inference error
or hide a failed command. Disabled execution uses the ordinary commit/wait API.
This is not a claim of zero instrumentation overhead.

The offline `scripts/drama3/analyze_command_timeline.py` validates host and GPU
clock ordering, rejects duplicate indices, and partitions an existing host wait
into its intersections before, during and after GPU execution. Missing GPU
timestamps retain unpartitioned wait instead of assuming zero execution time.
The 1 ms clock-bound allowance is a diagnostic validity policy, not a numerical
inference tolerance or an Apple timing guarantee.

Commit-to-GPU-start is not pure queue time. A post-GPU wait tail does not by
itself identify CPU scheduling. Command durations can overlap; their sum is
not request wall time. The driver request clock includes report serialization
and the worker's existing post-job collection, but excludes model loading.

Command indices continue across requests in one warm worker. Select requests
using their recorded submitted-command counts and preserve original indices.
For this fixed 60-frame trajectory, each request has 122 commands: relative
0–28 prefill, 29 first selector, 30–89 mixed frame/next-selector decode, and
90–121 DAC. Do not reuse this phase map for another prompt or frame count.

## Wait-variation experiment

Two sequential fresh-allocation workers each ran one cold and two warm requests
with host profiling and native timeline enabled, without Instruments attached.
All six requests preserve the full 60-frame trajectory, sampler input digest,
WAV bytes and read/wait ledgers. All 732 command records have valid timestamps.
The text is “Hello, this is a voice test.”, no reference, seed 1, cap 64 frames,
temperature 0.8, top-k 30, top-p 0.9. The WAV has 122880 samples at 44100 Hz
(2.786394558 s). Free-memory floor is 15%; unrelated host work remains running.

Each value is seconds. Warm repeats in the same worker are correlated.

| Worker / warm repeat | Request wall | GPU duration sum | Existing waits | Before GPU | During GPU | After GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / 1 | 24.9434 | 23.1230 | 23.3085 | 0.6557 | 22.5711 | 0.0817 |
| 0 / 2 | 25.0001 | 23.4619 | 23.3774 | 0.6043 | 22.7415 | 0.0316 |
| 1 / 1 | 30.5440 | 23.1629 | 28.8627 | 6.2372 | 22.5956 | 0.0299 |
| 1 / 2 | 31.5176 | 23.1676 | 29.7272 | 7.1441 | 22.5373 | 0.0459 |

Worker 1 repeat 2 is 6.5175 s slower than worker 0 repeat 2 despite a 0.2943 s
lower GPU duration sum. Its before-GPU wait is 6.5398 s higher. The first
prefill command (original index 244) alone has 6.7246 s of before-GPU wait,
0.1784 s GPU execution, and a 0.0008 s post-GPU tail. Its counterpart has
0.4902 s of before-GPU wait: the first-command difference of 6.2344 s is the
dominant part of the total 6.5398 s increase, with other commands contributing
about 0.3054 s. Repeat 1 independently
shows a 5.6007 s request delta and a 5.5814 s before-GPU wait delta, with only
0.0399 s difference in GPU duration sums.

Cold requests take 40.8294 / 38.1817 s. Their first prefill commands spend
14.6025 / 11.8456 s in before-GPU wait and 0.1727 / 0.1719 s executing.
These observations reject an interpretation of this wall-time spread as slower
kernel execution. They do not distinguish resource preparation, page residency,
driver work, competing GPU work, or another cause of the pre-start interval.

## Instruments qualification

Metal System Trace with Shader Timeline captures named `dr3_dense`,
`dr3_experts`, tower, sampler and DAC shaders in the unchanged binary. The
completed bounded trace covers 13.1554 s; its target process has 54508 shader
rows, 44 labels, and 41 Compute/Active GPU rows across 34 command-buffer IDs.
Every shader sample start falls within a target GPU interval; their union
covers 95.6149% of the shader sample-start window.

The exported shader rows expose sampled durations and per-row `% GPU Work`,
but `% Total GPU Work` is unavailable throughout. Sampled intervals overlap
across shader labels and expose no common submission/kick key with the GPU
interval table. Summing those durations or percentages would double-count.
Thus this trace qualifies coverage and shader identity; it does not establish
an additive family cost or a dominant kernel. The installed profiler schema
also supplies no additive weighting formula for these derived rows.

On this M2 Max, the timestamp counter set supports stage-boundary sampling,
but `supportsCounterSampling(AtDispatchBoundary)` is false. An attempted
per-dispatch counter route is rejected instead of adding unsupported sampling
or per-kernel waits. Direct homogeneous GPU probes and full-request paired
measurements are the next discriminators.

Raw Instruments traces can capture process environments. Launch captures with
an explicit environment allowlist, export only selected tables, remove any
environment node before inspecting a TOC, and discard raw captures after safe
exports. Raw captures from this experiment have been removed; the retained
measurement tables contain no process-environment export.

## Dense treatment and correctness boundary

The paired driver accepts `--dense-schedule tree,simd,simd,tree`. It disables
host/allocation/timeline profiling and generation/DAC scratch, varies only the
generic dense reduction selector, and pins the separate backbone qkv/z/b/a
selector to `tree`. Generic selection affects tower and other default-context
backbone projections; it is not a pure tower treatment. Cold jobs remain
separate from warm performance summaries. Every job must reproduce the saved
codes, sampler inputs, WAV digest and submit/wait/read ledgers.

Focused GPU parity cases cover F32/BF16/F16 mapped weights, aligned and odd
BF16 byte offsets, input views, 31/32/33 and 63/64/65 boundaries, wide tails,
the environment-selected context default, and a nonzero rank-3 expert offset.
The cancellation witness distinguishes the original tree from a left fold.
All four examples pass on the M2 Max, with bitwise equality required.

The homogeneous tower probe uses the actual mapped BF16 layer-0 FFN weights,
with deterministic generated F32 inputs uploaded once per width. It queues
tree/candidate command buffers in alternating order, with eight repeated
dispatches per arm, two excluded warmup pairs and seven measured pairs per
projection. Both arms drain together, then every output is compared bitwise.
It adds no per-dispatch wait or intermediate readback. These generated inputs
are not captured tower hidden states, and overlapping command intervals are
not additive family costs.

For the existing 64-lane `simd` candidate, median paired tree/candidate GPU
duration ratios are 1.113 / 1.119 / 1.268 for w1 / w3 / w2. All 27 pairs,
including warmups, match bitwise (1032192 compared elements). The profiler-off
full-request ABBA check also preserves all outputs in 12 jobs, but gives these
warm times in seconds:

| Worker | Dense mode | Warm repeat 1 | Warm repeat 2 |
| --- | --- | ---: | ---: |
| 0 | tree | 33.3008 | 26.6567 |
| 1 | simd | 35.5726 | 28.6539 |
| 2 | simd | 37.0697 | 39.1267 |
| 3 | tree | 38.5583 | 27.8337 |

The pooled warm medians are 30.5673 / 36.3212 s for tree/simd. Shared-host and
order variation are large; this does not admit a request-to-WAV speedup or a
default change. The external saved baseline is a correctness control, not a
performance baseline. Its runtime binary and request are bound by a separate
certificate. Its recorded resident-source hash differs from the named base
commit and the exact differing source was not retained, so this experiment
does not certify it as an unmodified build of that commit.

The new opt-in `simd32` candidate gives each lane the two original partial
accumulators, adds them in the original order, then applies the same shuffle
tree. This removes the cross-SIMD-group barrier without changing precision or
the arithmetic tree. It is selected only for execution width 32; other widths
retain the original kernel. The focused parity suite passes on M2 Max. Its
separate tower replay gives paired ratios 1.071 / 1.056 / 1.062, again with all
27 pairs bitwise equal. Separate runs do not compare `simd` against `simd32`
head to head. Full-request qualification determines promotion; default remains
`tree`.

The profiler-off `tree,simd32,simd32,tree` matrix also preserves the full output
and read/wait contract in 12 jobs. Its warm times are:

| Worker | Dense mode | Warm repeat 1 | Warm repeat 2 |
| --- | --- | ---: | ---: |
| 0 | tree | 33.2352 | 26.2011 |
| 1 | simd32 | 37.7053 | 27.9355 |
| 2 | simd32 | 42.0919 | 32.1256 |
| 3 | tree | 36.3939 | 35.2308 |

Pooled medians are 34.2330 / 34.9154 s for tree/simd32, approximately
12.286 / 12.531 request seconds per audio second. There is no established
end-to-end benefit; the smaller homogeneous-probe win remains local.

## Affine expert launch probe

The unchanged `dr3_experts` shader assigns quantization groups to lanes in
strides of 64, then lane 0 sums those groups in scalar order. With at most
32 groups, a 32-thread launch performs the same operations; at group 33 it
would leave a group unwritten. The opt-in `DRAMA3_EXPERT_THREADS=32` therefore
uses 32 only up to that boundary and retains 64 above it, for both native
scalar and batched affine paths. The setting is frozen when Context is
created. Default remains 64; the alternate MLX-Q4 kernel is unchanged.

The standalone launch probe checks actual layer-39 Q4/group-64 gate/up/down
weights against the pinned full MoE trace before GPU execution. It uploads
the same upstream normalized input, selected expert order and per-rank down
activation once, then submits eight gate/up/down triplets per arm at graph
depth 2. Two warmup pairs precede seven measured pairs with alternating order.
Only the final paired drain precedes bitwise terminal comparison. All nine
pairs match (1769472 elements per arm); there are 18 existing waits, zero
control/diagnostic readbacks, and zero interim waits/reads inside each pair.

Median paired GPU duration64/duration32 is 1.043834, with 32 faster in all
seven measured pairs. Arm medians are 20.3913 / 19.5765 ms per 24-dispatch
command. A second capture on the final benchmark source gives a paired ratio
of 1.061200, again faster in all seven measured pairs, with arm medians
17.7805 / 16.8267 ms and all nine pairs bitwise equal. Both captures are
reported; their absolute intervals vary. The commands overlap and mix gate/up/down; these intervals do not
rank the three projections or give an additive MoE family cost. Native32/
native64 identity on saved upstream inputs does not establish native-output
identity with upstream. The known MoE39 numerical failure remains open.

The direct-width probe bypasses the production selector. Separate independent
Q4/Q8 witnesses at group counts 32 and 33 check scalar and shared/per-rank
batched production calls against exact constants, including the final group.
Profiler-off full requests vary only the expert-thread setting with both
dense selectors at `tree`; their result controls any default promotion.

The `64,32,32,64` full-request matrix preserves all codes, sampler inputs, WAV
bytes and graph/readback ledgers in all 12 jobs. Warm times are seconds:

| Worker | Expert threads | Warm repeat 1 | Warm repeat 2 |
| --- | ---: | ---: | ---: |
| 0 | 64 | 31.3688 | 24.5201 |
| 1 | 32 | 34.0770 | 29.5189 |
| 2 | 32 | 31.6580 | 27.8118 |
| 3 | 64 | 36.5888 | 28.2036 |

Pooled warm medians are 29.7862 / 30.5885 s for 64/32 threads, or
10.690 / 10.978 request seconds per audio second. Cold jobs are retained
separately. There is no demonstrated request-to-WAV improvement; 64 remains
the default. The local GPU launch benefit is insufficient for promotion.

## Storage and evidence lineage

With user authorization, reference/run caches moved to
`/opt/mldata_ssd/cogni-ml`, with compatibility
symlinks at the recorded original paths. The DRAMA checkpoint remains on the
local SSD, following the user's subsequent instruction. All 25067 stable
regular files (52095286973 bytes) matched by SHA-256 before originals were
removed; symlink targets were preserved. Any later cold/cache-sensitive timing
needs its own baseline and storage qualification; it cannot be paired across
this move. The expert-thread full-request matrix ran after the move, with
identical storage placement in all four arms.
The active browser-log directory remains local: its console log grew during
the first verification, which correctly rejected deletion of the source.
A same-volume rename preserves that directory's inode; both compatibility
paths point to it while the stable run artifacts move to the external SSD.

The saved pinned MLX F32-core shared-sampler control and the 158-row prefill
fixture are bounded upstream controls. They do not certify a fresh stock MLX
WAV or close the existing layer-39 MoE numerical failure. No tolerance increase
is allowed to admit an optimization.

## Bounded validation

At the current source state, 62 Metal examples and 10 offline timeline tests
pass. The native fake-command test also covers successful, failed and missing
GPU timestamps, idempotence, and a throwing diagnostic callback. Numerical
kernel cases require bitwise identity; the callback/status tests establish
completion semantics rather than a performance claim.

Selected `simd32` and final expert-32 full requests each match all 60 saved
MLX-control frames and all eleven IDs per frame on GPU with the same 704
effective F32 sampler draws.
The 158-row prefill compares fresh and reused GPU storage: all 2048 hidden
values and 4097 main logits are bitwise equal between native arms, with zero
upstream pointwise failures at `atol=3e-5`, `rtol=3e-4`. Worst error/allowance
ratios are 0.05126616 and 0.00797102. Each native arm retains 159 existing
submissions/waits; terminal comparison metrics add no intermediate activation
read. This gate uses saved upstream activations, not a newly executed stock
MLX audio generation, and does not close the known MoE39 discrepancy.

The final prefill combines `simd32` dense and 32-thread expert selection. Its
production source matches the final full-request, build and regression gates;
older dense/timeline measurements retain their capture-time lineage. The source
seal indexes selected command records and file hashes, not a reproducible-build
attestation. The older saved baseline's differing source bytes remain unavailable.
Evidence paths and hashes are curated in
`../.landmark/evidence/drama3-gpu-kernels.json`.
The final scoped mini-Quadrumvirate review returns ROBUST; same-lineage review
is correlated and does not broaden these measurement or numerical boundaries.

## Next bounded discriminator

The pre-start wait still needs a probe that separates residency/resource
preparation from competing GPU/driver work. Ordinary command durations cannot
resolve that cause. A separate kernel hypothesis is the Q4 packed-word loader:
compare a guarded aligned `uint` load with the existing four byte loads on
captured weights/activations, including unaligned fallback and unchanged float
order. Inspect generated behavior before assuming the compiler has not already
combined those loads. No dominant-kernel or speedup claim follows from source
counts. The numerical return frontier remains the known layer-39 MoE failure.
