# Repeated Q4_K reconstruction inside native GEMM

Status: VERIFIED for bounded same-format parity and measured improvement over
the original direct kernel (2026-10-24), starting at `bad01824` on
`codex/wan-cross-segments`. Superiority to MPS and model-wide benefits remain
unestablished. No default or full-model change. The preceding
[register-reconstruction experiment](wan-q4k-register-gemm-benchmark.md)
passed bounded parity but lost to staged MPS by 9–16 times. This slice tests
specific redundant work with matched whole-projection ablations.

## Prediction and protected behavior

For M4480/N13824/K5120, M64/N32 tiles give 70 row tiles and 432 channel tiles.
The original core loads coordinate fragments 640 K8 steps times two column
fragments times four SIMD groups: 5,120 matrix loads per threadgroup, or
154,828,800 across the dispatch. Capturing one fragment per SIMD group reduces
this source-level count to 120,960. These are logical source counts, not
measured hardware instructions or a predicted speedup.

Each distinct weight is also reconstructed twice per M64 tile, once for each
row-half SIMD group. The complete projection therefore executes a logical
9,909,043,200 coefficient reconstructions for 70,778,880 distinct weights.
Neither ablation in this slice reduces that reconstruction count. At eight
corrections, the old lookup can perform up to 79,272,345,600 index comparisons;
early matches reduce the actual count.

The first opt-in ablation, `candidate-b64n32-sg4-device-q4k-map`, captures
Metal API-derived coordinates before K, keeping the old sparse scan. The
second, `candidate-b64n32-sg4-device-q4k-index`, also replaces the scan with a 4-bit
rank map: zero means unmatched, codes 1–8 select the already-half-rounded
delta. A channel owner initializes 32 words and inserts its records once per
256-K block, before the existing shared-memory barrier. Layout is
`map[K-word * 32 + channel]`. The extra map is 4 KiB, not a global F32 weight
buffer. Its initialization and threadgroup-memory cost can outweigh the saved
lookups; zero-correction runs help test that counterexample. At the full
projection shape, initialization alone performs 619,315,200 logical shared
word stores (2,477,260,800 bytes), excluding rank insertion and reads. This
source count is not a measurement of physical memory traffic.

Keep separate F32 base reconstruction and delta addition, unmatched corrected
`base + +0.0f`, ascending K8 matrix accumulation, F32 bias, malformed-block
isolation and sticky GPU status. Invalid duplicate codes must never index the
delta table: test malformed state before any map lookup. Preserve all original
selectors as rollback. No new host waits, activation readbacks, per-K barriers,
encode allocations or precision/tolerance changes are admitted.

## Measured whole-projection ablations

The additive `--compare-q4k-kernels` mode compares the original direct kernel,
coordinate capture, indexed corrections and complete block-validated
decode -> reused F32 stage -> MPS -> bias. Sixteen paired rounds rotate four
formats and four execution arms, reversing their orders across halves and
fresh processes. One sample per arm/format/round keeps 256 measured samples
within the existing 300-second process deadline. Every sample retains the
existing single terminal completion wait. Pre/post parity is outside timing.

M4480/N13824/K5120 uses one direct dispatch, 432 by 70 threadgroups, without
fringes. All direct arms retain M64/N32/K32, 128 threads and four SIMD groups.
Compiled static threadgroup memory is 10,368 bytes for baseline/Map and 14,464
for Index; the shared staged-tail pipeline uses 14,464 bytes. GPU medians in
milliseconds, with 16 measured samples per cell:

| Format | A baseline | A Map | A Index | A MPS | B baseline | B Map | B Index | B MPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 567.379 | 542.448 | 668.314 | 58.823 | 796.103 | 775.644 | 870.646 | 73.118 |
| +1 delta | 604.174 | 545.652 | 629.248 | 59.207 | 898.253 | 823.900 | 843.101 | 75.948 |
| +4 deltas | 784.827 | 706.198 | 636.630 | 58.740 | 1253.965 | 1081.479 | 861.382 | 77.989 |
| +8 deltas | 956.110 | 874.384 | 657.136 | 60.956 | 1460.629 | 1316.095 | 866.116 | 76.319 |

Absolute times drift substantially between fresh A/B processes. The cause
was not measured; do not attribute it to GPU clocks, occupancy, temperature
or background work. Compare same-format/same-round ratios within each
process. The following medians pool those 32 paired ratios, excluding M4;
they are not ratios of pooled absolute-time medians:

| Format | Map / baseline | Index / baseline | Index / Map | Selected latency reduction vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 0.95884 | 1.14233 | 1.18649 | Map: 4.12% |
| +1 delta | 0.90252 | 1.00396 | 1.09991 | Map: 9.75% |
| +4 deltas | 0.89698 | 0.77266 | 0.87078 | Index: 22.73% |
| +8 deltas | 0.91529 | 0.63984 | 0.70408 | Index: 36.02% |

Map improves each format's median in both processes and each half, although
some individual Q4_K pairs regress. Index improves all +4/+8 pairs; its
baseline ratios are A/B 0.81074/0.71330 and 0.68813/0.59518 respectively.
Index regresses Q4_K and gives no stable +1 benefit. Its additional lookup
storage/setup does not pay universally. Selection in the last column is an
experimental result, not an implemented automatic dispatch policy.

The complete control includes block-validated decode -> reused F32 stage ->
MPS -> F32 bias. Even the best direct arm remains approximately 9–11 times
slower: paired selected/MPS medians are 9.60593, 9.85730, 10.90198 and
10.97097. This slice reduces real redundant work but does not close the GEMM
throughput gap or establish a useful video-generation latency.

## Accuracy, synchronization and memory

Guarded DoD passes: 25 native examples, 38 analyzer tests, release build,
three source-bound probes and strict combined receipt analysis. The 264
same-format pre/post metric groups plus six BF16-control groups have zero
nonfinite values and zero violations of the unchanged finite
`2e-5 + 2e-5*abs(reference)` gate. Both M4480 runs are bit-exact in all 176
decoder/projection/pair metric groups. M4's largest absolute difference is
`2.336502e-5`, maximum combined-tolerance ratio `0.33293873`. M4 dispatches
only the existing staged fringe and does not measure either optimized core.

Native tests materialize all coefficients through the API-derived mapping
across all four SIMD groups, both 256-K blocks and all 32 channels, then
compare with scalar GPU decoding. They attack unsorted corrections, signed
zero, NaN/Inf, K-block transitions, malformed-neighbor isolation, sticky
status, destination extent, fringes and aliases. A duplicate at ranks 6/7
would form nibble code 15; malformed state is checked before indexing the
eight-slot delta array. Corruption controls in every probe reject exactly
two injected tolerance violations, including one NaN; final GPU status is
zero. Normal M4 rounding differences are not misclassified as corruption.

All 816 timing samples (768 measured and 48 warmups) retain one existing
terminal wait, zero inter-operator host waits, zero activation readbacks and
zero tracked explicit encode-time MetalBuffer allocations. Source review
finds no added threadgroup barriers. Status inspection remains after the
existing completion wait, outside the GPU interval. The allocation counter
does not cover all driver allocations.

| Probe | Sampled peak group footprint (GiB) | Minimum estimated cache-aware headroom (GiB) | Minimum pressure-free | System swap growth |
| --- | ---: | ---: | ---: | ---: |
| M4 | 1.00781 | 14.39070 | 73% | 0 |
| M4480 A | 2.74253 | 12.97713 | 70% | 0 |
| M4480 B | 2.74225 | 10.32604 | 69% | 0 |

Keep cache-aware 15% physical reserve and pressure floor, 4 GiB process-group
cap, 64 MiB system swap-growth stop and 240/300-second build/run deadlines.
File-cache reclaimability is an estimate and sampled guards miss between-poll
spikes. Builds, caches and evidence remain under `/opt/mldata_ssd`. The
combined harness retains the control's 270 MiB F32 stage, so it cannot measure
deployed process-footprint savings. No model, full block or generation runs.

All arms use the same packed payloads, sidecars and pinned upstream oracle;
PyTorch is absent from the measured native path. M4480 repeats four seeded
non-video rows 1,120 times. This tests execution scale, not representative
video quality. Existing lossy output L2 against original BF16 remains about
4.63/4.58/4.45/4.30% for 0/1/4/8 deltas on those four rows; exact same-format
parity is not equivalence to original BF16. No format or packing change.

## Executed DoD and qualifications

From the task worktree, the launcher was
`python3 -B examples/support/wan_q4k_delta_experiment.py` with these arguments:

```text
spec decode-reuse-native-v2
analysis-spec decode-reuse-analysis-v2
build decode-reuse-v1
probe decode-reuse-r4-v1 --build-tag decode-reuse-v1 --fixture-tag fused-r4-v1 --rows 4 --rounds 16 --compare-q4k-kernels
probe decode-reuse-a4480-v1 --build-tag decode-reuse-v1 --fixture-tag fused-r4480-v1 --rows 4480 --rounds 16 --compare-q4k-kernels
probe decode-reuse-b4480-v1 --build-tag decode-reuse-v1 --fixture-tag fused-r4480-v1 --rows 4480 --rounds 16 --compare-q4k-kernels --reverse-first
```

Strict combined analysis ran the source-pinned external
`build/wan-q4k-delta-20261023/analyze-decode-reuse-v1.py` wrapper under
`wan_memory_guard.py --memory-accounting=file-cache --max-footprint-mib=4096
--max-swap-growth-mib=64 --timeout-sec=300`, invoking
`examples/support/wan_q4k_delta_results.py` on those three probe tags. The
wrapper refuses evidence overwrites. Reproduction needs fresh tags and a
fresh analysis output prefix.

The first native spec attempt failed because Metal disallows helper-local
threadgroup arrays. Moving qualification-probe scratch declarations to
kernel entrypoints fixed compilation without changing the production
arithmetic. The first one-tag strict analysis rejected M4's corruption
control because it required exactly two bit differences, although ordinary
within-tolerance rounding already differed in 53,899 elements after
injection. The new mode now accepts at least two bit differences while
still requiring exactly two tolerance violations, one nonfinite value and
explicit rejection. A regression test covers these values and missing
injections; legacy non-comparison behavior remains strict. No numerical
tolerance changed. Failed receipts remain external and are not green DoD.

## Evidence lineage and decision

External root: `/opt/mldata_ssd/cogni-ml/video-scene-20261019/`.
Receipts/logs are under `evidence/wan-q4k-delta-20261023/`; build products,
caches and temporary files are under `build/wan-q4k-delta-20261023/`.
All successful completions have return code zero and `changed_files: []`.
The 248 runtime source pins match the build's captured source archive. Final
analyzer/spec-only changes have their own analysis receipts and do not alter
the measured binary.

| Receipt or artifact | SHA256 |
| --- | --- |
| `spec-decode-reuse-native-v2.completion.json` | `7f2f67e562db574f9a1bee3e4db63bdf0362677bd6e358f86ab8890c31c443e1` |
| `analysis-spec-decode-reuse-analysis-v2.completion.json` | `415fbe917a5b8e2bbf9069e9d335921683964f9b92e72dd12dc4741ccc02a34b` |
| `build-decode-reuse-v1.completion.json` | `63f569f5d9ab2f424e99ee32d9a4dbcd670a1bd1240228e5d8ca930222b474f2` |
| `probe-decode-reuse-r4-v1.completion.json` | `a76ea2463918c8e89b1b54573f1735871500ac8a7e2a1febab95b7ed1c32484d` |
| `probe-decode-reuse-a4480-v1.completion.json` | `526fcd9a70903f5a71ee6a63ab26177fb4f2d7c4f8df4396d6b42ed32ad826e7` |
| `probe-decode-reuse-b4480-v1.completion.json` | `80f00094e76622cbf3ba97a717000367c29ee24d468370a2098f4175c6363ba0` |
| `analysis-decode-reuse-v1.completion.json` | `1e2788bfd116179f2e8ab57523a90acb5ec1fd337c4f3e2fd2550f0e18080aba` |
| `summary-decode-reuse-v1.json` | `48f7e13cb901884783df2452d5e91eba3b5519b75d31bc096938b41dad09f48f` |
| `probe-decode-reuse-v1` binary (build folder) | `f43c159cf1e1c5032289ec0a3ddab4cb6e5d64a84b504a4d522edeed9bb96ed7` |

Inherited M4/M4480 fixture manifest hashes are respectively
`36b796df8ff172609cc78e75078d11dbf60446e82f3e4861fada4a2e76542891`
and `204432b92773cf54b9db9482d1216e533e249a2e3d26a9ae244c2902860a5af3`.
The six reused native objects retain the pinned
`evidence/wan-qbit-mps-stage-20261022-v2/build-v2-intent.json` receipt,
SHA256 `4cf673fd89d04112bc72de3e26e98d85111876f8f9b698042df2341aa2f6bca4`.
Only the T5 linear bridge has complete historical source/object ancestry;
the other five objects have current exact pins without a reconstructed
historical compilation chain. The quantizer is pinned, not rebuilt here.

Keep original selectors and staged MPS as rollback controls; both new
selectors remain opt-in. Correlated Mini-Quad review is ROBUST for bounded
correctness, observed guards/contracts and measured paired improvements on
this fixture. The MPS competitiveness/model-wide acceleration objective
remains VULNERABLE. Portability, hardware-cause attribution, representative
video/BF16 quality, deployed footprint savings and model-wide acceleration
remain VULNERABLE to missing evidence.

The remaining high-value hypothesis is reuse of a reconstructed B tile
between row-half SIMD groups or across a wider M tile, while retaining
ascending K8 accumulation and bounded scratch. The current variants still
perform 9.909 billion logical coefficient reconstructions; removing
coordinate loads and sparse scans leaves that work. Another separate route
is a compact rolling decoded panel feeding MPS. Neither is implemented or
measured here. Refresh after source/codec/fixture/geometry, compiler/native
object, backend/OS/SDK/device, numerical-policy or schedule changes.
