# Q4_K sparse-delta GEMM without a global F32 weight stage

Status: VERIFIED for the bounded native implementation and comparison
(2026-10-23). The opt-in fused path removes the 270 MiB global F32 weight
stage, preserves the same-format upstream numerical gate, and is slower than
staged MPS in every measured pair. It is not promoted for speed. Model paths,
checkpoint bytes and defaults are unchanged. The parent is
[block-shared delta validation](wan-q4k-block-validation-benchmark.md).

## Implementation and falsifier

`ML::Video::Q4KDeltaLinear.linear` consumes canonical Q4_K payloads and the
existing 0/1/4/8 sparse F16 corrections per 256 weights directly. Its
64-by-32-by-32 shared-panel GEMM uses four SIMD groups, F32 panels and F32
accumulators. K-contiguous weight loads are transposed into the shared B
panel. Correction records are validated/staged once per output-channel and
256-K block in each row tile, then reused across eight K32 panels. The
compiled fused pipeline reports 14,464 bytes of static threadgroup memory.
There is no global decoded-weight binding in the candidate. Local F32
reconstruction remains necessary; this is not arithmetic on raw INT4 codes.

Reconstruction retains separate F32 products/subtraction followed by addition
of the half-rounded correction, including default +0 in corrected formats.
Malformed correction lists zero only their output-channel's 256-K slice;
invalid base/result values zero only the affected coefficient. Sticky status
does not suppress otherwise valid work. Setup validates pipeline geometry;
encode validates dimensions, sizes, dispatch limits and storage overlaps.
All buffers are caller-owned. `dense_linear` uses the same tile geometry and
schedule with an ordinary F32 weight matrix. `decode_probe` shares the fused
coefficient helper, but is diagnostic and outside the measured candidate.

This CAUTION experiment predicted that simpler fixed Q4_K records might
make a fused consumer competitive after the earlier QBit route failed. The
cheapest discriminators were exact coefficient reconstruction, an unchanged
upstream output gate and the same-geometry dense control. The strongest
counterexamples were unsafe panel reuse, changed reconstruction/reduction,
and repeated decode work overwhelming saved traffic. Focused guards and
full-projection timings test these separately. Rollback is the unchanged
staged MPS route; do not pass `--compare-fused` to the probe.

## Paired timings

The real projection is `blocks.0.ffn.0.weight`, N=13,824 and K=5,120.
M4 has four unique seeded non-video activation rows. M4480 repeats those rows
1,120 times; it tests execution scale, not representative activation or video
quality. All values below are GPU milliseconds. A and B are fresh processes
with opposite first order, each with 16 measured rounds and two warmups.

The control includes block-validated decode -> reusable F32 stage -> MPS ->
bias. The candidate includes fused decode/GEMM/bias in one dispatch. Each
same-format control/candidate pair is adjacent; four format pairs rotate
through all eight slots, swap internal order every four rounds and reverse
the whole order between halves. Both pair orders occur in each half. Package
bytes, input, output geometry and tolerance are identical within each pair.

| Corrections per 256 | A, staged -> fused | B, staged -> fused | Paired fused/staged median A / B |
| --- | ---: | ---: | ---: |
| 0 | 57.97396 -> 419.51281 | 61.40583 -> 424.58619 | 7.23519x / 6.87922x |
| 1 | 58.02875 -> 448.70294 | 62.31202 -> 454.60448 | 7.73497x / 7.34545x |
| 4 | 58.12340 -> 511.96446 | 62.20110 -> 518.24006 | 8.81084x / 8.51406x |
| 8 | 59.72000 -> 594.81648 | 63.75567 -> 598.73873 | 9.95983x / 9.51305x |

The ratios are medians of same-round ratios, not ratios of arm medians.
Every pair, slot median, half and pair-order group is slower in both full-size
processes. The least unfavorable individual pair across all formats is
6.57492x slower. The staged Q4_K median drifts about 5.9% between A and B;
report the processes separately. That drift cannot explain the paired loss.
No causal hardware bottleneck is identified by this timing experiment alone.

The separate dense mode compares the same native tile against MPS on the
Q4_K-decoded F32 matrix, without decode in either measured arm. Sixteen
ABBA/BAAB rounds each contain two native and two MPS samples, with the order
phase reversed between halves and between processes.

| Geometry | MPS median | Native dense median | Paired native/MPS median |
| --- | ---: | ---: | ---: |
| M4 | 0.79058 | 3.09365 | 3.92607x |
| M4480 A | 55.57665 | 281.03219 | 5.05698x |
| M4480 B | 67.51260 | 308.18260 | 4.57342x |

The dense gap survives both orders and halves. It establishes that the
current native GEMM schedule is already uncompetitive before decoding.
Dense MPS medians drift about 21.5% between processes; do not pool them or
interpret the paired ratios as a stable hardware constant.
The fused/dense difference is not an isolated decode-time measurement:
different instructions can also alter scheduling and compiler resource use.
Bandwidth, register pressure, spills and occupancy remain unmeasured.

For K8, complete wall medians are 60.06290 -> 595.10623 ms in A and
64.08754 -> 598.96354 ms in B. Full-size fused encode medians range from
0.00765 to 0.01169 ms versus 0.04702-0.06002 ms for staged controls. Saving
CPU encode work does not offset the GPU regression. At M4, staged/fused
medians are 3.24973/4.43529, 3.34825/4.87081, 3.39831/5.82531 and
4.93771/7.11215 ms for quotas 0/1/4/8; no measured geometry wins.

## Numerical, failure and memory checks

The 9 independent CPU reference tests, 18 analyzer tests and 15 native Metal
examples pass. Native tests cover heterogeneous non-dyadic rows, M/N tails,
K256/K512, all quotas, correction endpoints, exact F32 reconstruction and
signed zero, duplicate/NaN/Inf records, neighboring valid slices/channels,
independent sticky status, aliases, invalid dimensions/counts/sizes,
untouched poisoned outputs on wrapper rejection, pipeline geometry and zero
tracked encode allocations. No tolerance was widened.

All GPU-packed sidecars match independent expected bytes. Before and after
timing, all 70,778,880 reconstructed coefficients per format are bit-exact
with upstream. All 61,931,520 projection outputs in each M4480 arm are also
bit-exact with the same-format upstream and staged control in A and B. M4
fused/dense outputs differ in low bits: the maximum absolute discrepancy is
2.336502e-5 and maximum tolerance ratio is 0.33293873, with zero nonfinite
values or violations of `abs(actual-ref) <= 2e-5 + 2e-5*abs(ref)`.
Coefficient bit-exactness and output tolerance are distinct checks.

The finite-plus-NaN corruption controls are rejected against both upstream
and the staged result: exactly two tolerance violations and one nonfinite.
The BF16 control passes and final candidate status is zero, observed by
scanning all status words after an existing untimed terminal wait. These are
implementation-parity checks against the same quantized format. Lossy output
relative L2 versus original BF16 remains 4.63165%/4.57702%/4.44768%/4.30013%
for quotas 0/1/4/8 on the four rows. Fusion does not establish BF16 quality
parity or perceptual acceptability.

Across the three probes, 636 timing samples including warmups retain one
existing terminal wait, zero inter-operator host waits, zero activation
readbacks and zero tracked explicit `ML::MetalBuffer` encode allocations.
Native execution alone is timed. Offline independent PyTorch MPS references
have CPU fallback disabled; diagnostic comparisons/status exports are outside
the timed command chain. Local threadgroup barriers add no host synchronization.

The candidate eliminates a 283,115,520-byte (270 MiB) global F32 weight stage
from its binding inventory. Complete quantized packages are unchanged:
37.96877/38.75978/41.13283/44.29689 MiB, including positions, deltas and the
16-byte descriptor. The zero-correction path has a separate 4-byte ignored
sidecar placeholder, outside the serialized package. Bias remains 55,296
bytes. The combined harness retains the control stage, original BF16,
all formats and diagnostic buffers. At M4480 it owns 2,155,814,480 explicit
buffer bytes, of which 1,872,698,960 are outside the stage; they are not all
diagnostic overhead. Thus no 270 MiB process-footprint drop was measured,
and no model-wide fit or throughput benefit is established.

Native probes peak at a sampled 2.02292 GiB process-group footprint; minimum
estimated cache-aware headroom is 18.87155 GiB, pressure-free at least 74%,
and swap growth zero. Oracle construction peaks at 1.76979 GiB with minimum
headroom 22.33055 GiB and pressure-free at least 75%. Preserve the 15%
physical reserve and pressure floor, 4 GiB process-group cap, 64 MiB system
swap-growth stop and deadlines. File-backed bytes are a reclaimability
estimate; sampled limits cannot exclude between-poll spikes. Opaque MPS and
compiler-private resources are outside the explicit buffer counter and
inside the sampled process guard. No full model, block, Studio, video or
generation was launched.

## Reproduction and lineage

Starting revision is `51b18ba0` on `codex/wan-cross-segments`, Apple M2 Max,
macOS 26.6.2. All temporary/build/cache/evidence storage remains under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019`. Let
`L=examples/support/wan_q4k_delta_experiment.py`. These final commands after
`python3 -B $L` returned zero with `changed_files: []`:

| Arguments | Signal |
| --- | --- |
| `cpu-spec fused-green-v1` | 9 tests, OK |
| `analysis-spec fused-green-v2` | 18 tests, OK |
| `spec fused-native-v3` | 15 examples, zero failures/errors/pending |
| `build fused-v3` | Source-bound release binary, observed final GPU status |
| `oracle fused-r4-v1 --rows 4` | Independent four-row fixture |
| `oracle fused-r4480-v1 --rows 4480` | Independent repeated-row fixture |
| `probe fused-r4-v2 --build-tag fused-v3 --fixture-tag fused-r4-v1 --compare-fused` | Full short parity/control/timing checks |
| `probe fused-a4480-v2 --build-tag fused-v3 --fixture-tag fused-r4480-v1 --compare-fused` | Full-size checks, A |
| `probe fused-b4480-v2 --build-tag fused-v3 --fixture-tag fused-r4480-v1 --compare-fused --reverse-first` | Full-size checks, opposite first order |

Probe rounds default to 16. The strict analyzer also returned zero under the
same memory guard, bound by `analysis-fused-v2.intent.json` and completion:

```sh
python3 -B examples/support/wan_memory_guard.py \
  --memory-accounting=file-cache --max-footprint-mib=4096 \
  --max-swap-growth-mib=64 --timeout-sec=300 \
  --log="$EVIDENCE/analysis-fused-v2.guard.jsonl" -- \
  python3 -B examples/support/wan_q4k_delta_results.py "$EVIDENCE" \
  fused-r4-v2 fused-a4480-v2 fused-b4480-v2 \
  > "$EVIDENCE/summary-fused-v2.json"
```

The native/analyzer test receipts predate the final benchmark-only status
telemetry change. Their executed sources are unchanged; the benchmark is
included in their broader source-pin inventory but is not imported by those
test entrypoints. Build v3 and all three final v2 probes exercise the current
benchmark bytes, including the observed final status-word scan.

`EVIDENCE` is the external base plus `evidence/wan-q4k-delta-20261023`.
Use fresh exclusive tags and filenames; never overwrite prior evidence.
The fresh manifests are byte-identical to the parent fixtures. Producer,
source, binary, fixture, guard and schedule pins are checked, including
negative analyzer cases for missing parity, changed geometry/order/storage,
nonfinite metrics and constraints. The six native objects are reused with
the prior byte pins. Only the T5 linear bridge has complete historical
source/object ancestry; the other five have current source and object pins,
not reconstructed historical build lineage. The ggml quantizer is pinned
at `7e4c0a96880dae4fc4268ad441f8a6446bd5460a`, not rebuilt here.

| Artifact | SHA256 |
| --- | --- |
| `spec-fused-native-v3.completion.json` | `72c386ef2b6d2b747d97ac382709b5b5362c3ff4d29ffe0b942c58c930d2ced5` |
| `analysis-spec-fused-green-v2.completion.json` | `04087150e53892560ae64fdff5cd6ccbcb7fe157284e593f254e733fa22cc673` |
| `build-fused-v3.completion.json` | `063fcec846c0b5c621847cf750b43f033bf0de7c885d9660c6132ff44e518eb0` |
| `probe-fused-r4-v2.completion.json` | `52c892381966b24cf0e34a22461b2ea7e36c8a3207b9e3604ad1e07de97b1e29` |
| `probe-fused-a4480-v2.completion.json` | `683bce661bd4c12982d0d732d93b8f9781a3471bddee2368bbabddbe4f37bf27` |
| `probe-fused-b4480-v2.completion.json` | `3f1040e02349187a386610931bafd8afcfa52c24ba30d27ee911e73d20957218` |
| `analysis-fused-v2.completion.json` | `7f15c602ced1ca40668cb4d5890d6f8a893dcc272a00b7b1bf7c16ca9b24eff6` |
| `summary-fused-v2.json` | `9d44bba01a8f9467adefbdd998aed6b0cf99cd8b4476368ebba1086c4f87caa4` |
| Release binary `probe-fused-v3` | `d0ed18d3e09553369c5e9b669752daf9fce3c8e4d9659c1941c58c9aef5b25d4` |
| Four-row manifest | `36b796df8ff172609cc78e75078d11dbf60446e82f3e4861fada4a2e76542891` |
| Repeated-row manifest | `204432b92773cf54b9db9482d1216e533e249a2e3d26a9ae244c2902860a5af3` |

## Decision and refresh boundary

Keep the fused wrapper and `--compare-fused` as opt-in research surfaces.
The bounded arithmetic/storage contract is established, but the speed
hypothesis is rejected for this tile, device and tested inputs. Lower memory
is a real execution-path benefit; its effect on whole-model fit, paging and
latency remains unmeasured. Continue using staged MPS for this throughput
comparison. The next execution discriminator is a competitive native dense
GEMM before another fused loader optimization, followed by identical-geometry
packed tests. No bandwidth/register/spill explanation is promoted without
a GPU profile.

The correlated Mini-Quad review is ROBUST for pinned bounded parity, observed
guards and speed non-promotion; VULNERABLE for representative quality,
model-wide memory/speed or universal fusion claims. Same-lineage review is
not independent replication. Refresh focused guards, full parity and paired
timings after source/dispatch, payload/sidecar/input/geometry, compiler/native
object, OS/SDK/device, tolerance or schedule changes before reusing this
evidence in a new context.
