# DRAMA-3 tower FFN paired projections

The tower computes `w2(silu(w1(x)) * w3(x))`. On the pinned checkpoint,
`w1` and `w3` are separate BF16 matrices of shape `[6144, 2048]` in each of
four shared and nine residual layers. Both consume the same resident F32
normalized input.

## Candidate and rollback

`DRAMA3_TOWER_FFN_PAIR_FUSION=1` selects one two-output dispatch for the
two projections. The selector is strict (`0` or `1`) and frozen at tower
construction. Unset/`0` retains the original two inline matvec calls.

For equal-row BF16 matrices with even, in-range mapped offsets, one
threadgroup computes the same row in both projections. It reads each input
value once into two independent F32 accumulators and expands each BF16 word
exactly. Each accumulator keeps the original lane-strided products,
sequential subtotal and 64-lane reduction tree with contraction disabled.
The launch uses the common row count, not the sum of both row counts.

Other supported pairs use a combined row domain selecting one matrix and
output per threadgroup. No shared subtotal, new precision, repacking or
activation transfer is introduced. Output buffers are allocated in the same
order and sizes as before. Input and output offsets are passed to Metal;
mapped weight offsets remain in the parameters.

Only rank-two, unscaled/unbiased F32/BF16/F16 matrices with expert zero and
bounded positive row counts qualify. Width and ownership checks run before
allocation or encoding. BF16 specialization additionally checks both mapped
ranges with overflow-safe arithmetic. Unsupported layouts and non-tree
reductions use the two established matvecs. A tower row saves thirteen
dispatches, without changing
production allocation, command submission, wait or read boundaries.

The new `buffer_trace` observer receives borrowed resident buffers. It is nil
in normal requests and lets diagnostics compare intermediate values on GPU.
The comparer copies Q/K norm outputs on GPU at observation time because
RoPE subsequently mutates those buffers in place. These diagnostic copies
introduce no production copy or additional synchronization.

## Qualification

Release Metal checks cover widths 31/32/33/63/64/65/2048, unequal output rows,
F32/BF16/F16, aligned and odd mapped offsets, input/output views and sentinels,
cancellation, unsupported-format/reduction fallback, ownership/width rejection
before effects, and selector freezing. Twenty examples pass, including exact
BF16 zero/subnormal/Inf/NaN comparison, explicit specialized/generic branch
checks and a payload ending exactly at its mapped buffer's end. An invalid
mapped range cannot be constructed through the checked public constructors;
that defensive fallback is source-reviewed rather than runtime-injected.
The observer test has no read or wait before terminal collection; `Context#read` then
consumes its existing four-byte finite-status word and 24-byte receipt.

`export_mlx_tower_pair.py` uses the pinned MLX-audio checkout and a retained
87-row native generation report. It captures MLX backbone hidden rows once,
teacher-forces the tower prompt history, and records final prompt tower row85
and first decode row0. Both reference/native decode arms use the actual saved
main token and greedy residual sampling. This avoids comparing different
stochastic codebook histories.

An uninstrumented tower replay uses the same already captured MLX hidden
rows, fresh caches and greedy policy. GPU bit/finite predicates check the
selected prefill output, final layer outputs and decode frame before accepting
the fixture. Immutable BF16 payloads of the six measured projection matrices
are hashed; config/index, source and revision pins are also recorded. Whole
checkpoint shards are not hashed.

The local probe alternates separate/fused production API calls over three
captured normalized inputs: shared0/prefill85, residual5/decode0 and
residual8/decode0. Two warmup pairs precede seven measured pairs per case;
each arm encodes eight repetitions. GPU bit/nonfinite receipts and injected
one-bit/NaN errors guard the comparison. Timing uses existing overlapping
command intervals; it does not establish isolated or additive kernel cost.

The first implementation only combined row domains. It preserved all
2,654,208 local values and passed six MLX controls, but measured just 1.04135x
at the pooled paired median (17/21 wins), including a 0.99875x case.

The BF16 row-pair implementation preserves the same 2,654,208 values and
passes all six MLX controls. All 21 measured pairs favor it. Median paired
separate/fused ratios are 1.20037x, 1.15581x and 1.14107x; pooled median is
1.16337x. Per-arm command medians per repetition are respectively
0.684974/0.570635ms, 0.675682/0.575109ms and 0.677578/0.596865ms.
These are local command intervals over captured inputs, not full-request gains.

The complete tower comparison replays the same 87 MLX hidden rows and prompt
frames into off/on caches and compares all available non-cache trace sites on
GPU. Tolerances remain `atol=6e-4, rtol=3e-4` internally and
`atol=3e-5, rtol=3e-4` for final hidden/head logits. Native parity and the
strict upstream verdict are separate results. Existing backbone/DAC strict
upstream failures are outside this optimization and remain open.

Across 674 captured tower boundaries plus the decode frame, all 675 native
off/on comparisons are bit-exact and finite. These are 675 comparison
triplets (2,025 raw metric records). At each site, baseline and candidate MLX
metrics match, including three failures already present in
the unfused baseline of this capture. They are decode residual8 `w2`
(11 failing values, maximum ratio 5.30739), decode residual8 FFN residual
(17, ratio 4.91615), and prefill row85 residual8 FFN residual (1, ratio
1.11942). This is no numerical regression; the strict upstream gate remains
red. V1 and V2 produce identical complete comparison records and the same
MLX decode frame. The diagnostic uses one command/wait and a 40,500-byte
terminal metric receipt; it does not read intermediate activations to CPU.

## Complete WAV control

The profiler-off request is `Hello, this is a voice test.`, seed1,
temperature0.8/top-k30/top-p0.9, cap64, no reference. It reaches EOS at60 frames
and produces 122880 samples at44.1kHz (2.786394558s).

The first twelve-request ABBA series has four fresh workers, each with one
excluded cold request and two warm requests. Full codes, sampler tape, WAV,
allocation counts/bytes, reads and waits match. Warm worker median request
times are 25.357579s separate /22.402665s fused; mirrored ratios are
1.210776/1.051621. However, unchanged DAC times also vary substantially.
Reverse BAAB gives 22.715963s separate /23.195108s fused, with mirrored ratios
1.037785/0.925733. Thus the first variant has no reproducible full-request
gain and is not a default-promotion candidate. All 24 requests retain exact
full codes, sampler tape, WAV and execution/allocation ledgers.

The BF16 specialization uses a separate ABBA and reverse BAAB series against
the same unfused path, with the same release binary, request and sampler
inputs. ABBA preserves all twelve trajectories and ledgers but gives
23.175658s separate /24.907944s fused, with mirrored request ratios
0.904047/0.960349. Generation is 20.354636/20.950108s and unchanged DAC is
2.770035/3.838288s. This series does not confirm full-request benefit.
The completed reverse BAAB repeat gives 26.576364s separate /28.817057s fused,
with mirrored request ratios 0.787350/1.078039. Generation is
21.890659/25.128233s and unchanged DAC is 4.546508/3.611178s. Neither order
establishes a reproducible complete-request gain. Default remains off;
these active-host observations do not isolate a causal kernel slowdown.

All 24 qualifying V2 requests retain the complete codes, sampler tape and
WAV exactly, with unchanged cold/warm allocation counts and requested bytes,
122 submissions/completions/waits, 496 control bytes and 494160 terminal
bytes per request. Warm requests allocate 135590 buffers and request
4,079,562,724 bytes cumulatively, not simultaneously. The WAV SHA256 is
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`.
Warm full-request RTF is 8.317/8.939 in ABBA and 9.538/10.342 in reverse
BAAB (separate/fused). The real-time target remains open.

The first V2 reverse attempt stopped after nine completed exact requests:
the unfused arm2 worker aborted at shutdown with an uncaught
`std::system_error: mutex lock failed: Invalid argument`. It is excluded
from qualifying timing results. The whole reverse order was repeated with
unchanged source and binary; the strict worker-exit check is retained.
The crash stack identifies the repeating GCD command watchdog locking its
process-global mutex in unchanged `bridge.mm`; the last worker's graphs
were drained. A callback racing static-mutex destruction is a hypothesis,
not a proven root cause. The repeat exits cleanly, but this optimization
does not explain or fix the shutdown failure.

## Evidence and replay scope

The curated index is `.landmark/evidence/drama3-tower-pair-fusion.json`.
Raw logs, binaries, fixture, analyses and immutable selected command records
are under `/opt/mldata_ssd/cogni-ml/drama3-realtime-20261016/`; they are not
repository payloads. The selected command records retain full argv, source
maps, environment controls, bridge/binary/log hashes and baseline-producer
joins. The current command set is export-v3, spec-v5, local build/run-v2,
compare build/run-v4, Studio build-v2, and request ABBA-v2/BAAB-v2-r2, all prefixed
`tower-pair-`. Failed or source-stale attempts do not qualify current code.

Source qualification covers the recorded paths and all changed production
paths, not a complete dependency closure. The runner omitted unchanged
`src/ml/metal/dispatch.cr`; it is not retroactively pinned. Historical V1
receipts are kept separately and cannot qualify V2 source. A frozen selector
observation is not itself a Metal dispatch receipt; branch specs and source
review support the selected production path. The three captured inputs and
selected tower rows do not establish numerical parity for every possible
cache history, voice, language or sampling trajectory.

Runs are serial under `run_safe`, with the authorized15% memory floor,
quiet-host disabled, external scratch and local weights. Unrelated processes
are untouched. Timing observations decay with host load, input, compiler,
source, device or OS changes; numerical evidence additionally decays with
model/reference/tolerance changes or loss of raw artifacts.

The next performance discriminator is full decode attribution from a GPU
trace, checked against profiler-off complete requests. These local command
intervals do not identify an additive dominant FFN cost. The separate
reliability return pointer is a minimal watchdog/process-exit reproducer;
a clean benchmark rerun is not closure of that failure.
