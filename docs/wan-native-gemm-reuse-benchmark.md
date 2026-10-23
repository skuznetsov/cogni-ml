# Native F32 GEMM panel and fragment reuse

Status: bounded comparison completed (2026-10-23). The best measured
candidate saves about 2% versus the legacy native GEMM, while MPS is still
about 4.9 times faster.
Wider M and K panels regress. Q4_K with sparse deltas remains slower than
staged MPS. No production default changes. This follows
[the Q4_K fused GEMM comparison](wan-q4k-fused-gemm-benchmark.md), starting
at `4fab8597bc913050cb64669aa816d9df2f5e0a55` on `codex/wan-cross-segments`.
All 15 GPU probes and their strict source-bound analysis pass. Competitive
native F32 performance against MPS remains open.

## Change and falsifiers

The legacy M64/N32/K32 F32 tile already loses to MPS without quantized
loading. Three opt-in candidates test the following changes, with four
SIMD groups and the same ascending K8 accumulation per output:

| Selector suffix | M/N/K | Change | Dense / fused compiled static threadgroup bytes |
| --- | --- | --- | ---: |
| `legacy` | 64/32/32 | Original schedule | 12,288 / 14,464 |
| `b128n32-sg4-tk32` | 128/32/32 | Reuse B fragments across row blocks; one live A fragment reused across columns | 20,480 / not measured |
| `b64n32-sg4-tk32-bt` | 64/32/32 | Fragment reuse plus contiguous shared B[N,K], transposed SIMD loads | 12,288 / 14,464 |
| `b64n32-sg4-tk64-bt` | 64/32/64 | Same B64 schedule/layout, twice the K panel depth | 24,576 / 26,752 |

Candidate CLI selectors start with `candidate-`. Pipeline preparation checks
compiled geometry and resource limits before encoding. All operands,
shared panels and accumulation remain F32. The larger M tile halves repeated
weight-panel loads at M4480. The transposed panel changes shared access
layout; the TK64 variant halves panel barrier pairs. These are cost
predictions, not measured bank-conflict, occupancy, register-spill or
barrier-bottleneck diagnoses. This comparison does not isolate fragment
reuse from the B-layout change in the B64 candidate.

This CAUTION experiment keeps legacy selection and staged MPS as rollback.
Candidates are opt-in. The numerical gate remains finite
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`. Quantized reconstruction,
half-rounded delta addition, signed zero, malformed-slice isolation and sticky
status remain unchanged. Correction lists are staged once per channel/256-K
block in each row tile; TK32/TK64 reuse them across eight/four panels.
The A-panel prefix holds the output epilogue, with a compile-time capacity
check. No global F32 decoded-weight stage is bound by the fused candidate.

The strongest failure modes are increased resource pressure, unsafe panel
reuse, changed reconstruction/reduction and order-dependent timing. Focused
failure tests, full upstream comparisons and both initial orders test them.
A reduction in loads or barriers alone is not sufficient evidence of speed.

## Dense timings

The real projection is `blocks.0.ffn.0.weight`, N=13,824 and K=5,120.
M4 has four unique seeded non-video rows; M4480 repeats those rows 1,120
times. It tests execution scale, not representative video quality.
The exact same GPU-decoded Q4_K F32 coefficients, activations and bias feed
legacy native, candidate native and MPS. Decode is outside dense timing.

Each process has two warmup rounds and 16 measured rounds, with two samples
per arm per round. A rotated ABC+CBA palindrome exercises all six
rotation/direction orders. The second half changes the base direction;
`--reverse-first` exchanges that direction in a fresh process. Reversing an
already constructed palindrome would not change the order. Sixteen rounds
are not exactly balanced over three rotations; actual slot counts and all
order-group summaries are retained. The paired statistic is the median of
same-round candidate/control ratios, each computed from the two arm samples'
means. Processes are reported separately, never pooled.

GPU milliseconds and direct paired candidate/legacy ratios:

| Candidate / geometry | Legacy | Candidate | MPS | Paired candidate / legacy |
| --- | ---: | ---: | ---: | ---: |
| M128, M4 | 3.08590 | 5.83377 | 0.77700 | 1.89117x |
| M128, M4480 A | 281.29525 | 373.50304 | 55.45475 | 1.32791x |
| M128, M4480 B | 281.19706 | 373.53554 | 55.48417 | 1.32825x |
| B64/TK32 BT, M4 | 3.11746 | 3.05665 | 0.79152 | 0.98404x |
| B64/TK32 BT, M4480 A | 281.21279 | 275.82327 | 56.35571 | 0.98001x |
| B64/TK32 BT, M4480 B | 281.81277 | 276.02783 | 56.39656 | 0.97998x |
| B64/TK64 BT, M4 | 3.10537 | 5.44694 | 0.79058 | 1.74901x |
| B64/TK64 BT, M4480 A | 281.62471 | 627.35144 | 57.34473 | 2.22332x |
| B64/TK64 BT, M4480 B | 281.78692 | 627.10485 | 57.83356 | 2.22596x |

Both wider-panel speed hypotheses fail within the measured scope. The
B64/TK32 candidate improves native latency about 2%, consistently in the two
full-size processes. Its worst individual round ratios remain below one
(A range 0.97336-0.98882, B 0.97535-0.98710). This is not a competitive
alternative to MPS, and it does not identify the hardware cause of the gap.
Its paired candidate/MPS medians are 4.85540x/4.85572x. M4 round ratios
span 0.86180-1.05032, so there is no stable small-M improvement claim.
M128 timings belong to archived build v1, BT32 to v2, and TK64 to v3.
Final source retains their schedules but adds a compile-time K selector;
final native tests cover all selectors, while old timing receipts remain
bound to their archived source bytes rather than being relabeled as v3 runs.

## Return to Q4_K with sparse deltas

The complete control is block-validated decode -> reusable 270 MiB F32
stage -> MPS -> bias. The candidate fuses decode/GEMM/bias in one dispatch.
Adjacent same-format pairs rotate through all eight slots, swap internal
order and change base direction between halves; both pair orders occur in
each half. Quotas are 0/1/4/8 corrections per 256 coefficients. Both full-size
initial orders are tested for BT32 and TK64 after their dense comparisons.
No source, format, tolerance or resource-policy change occurs between a
build's dense and packed probes.

BT32 GPU milliseconds (staged -> fused):

| Corrections | A | B |
| --- | ---: | ---: |
| 0 | 58.62344 -> 416.53792 | 73.19933 -> 495.20842 |
| 1 | 58.97087 -> 445.82919 | 73.04875 -> 535.64375 |
| 4 | 59.81329 -> 508.94196 | 74.09431 -> 612.99933 |
| 8 | 61.85537 -> 589.92454 | 75.70200 -> 720.81640 |

Both paths drift materially between A/B. Only within-process paired ratios
support a comparison; differences from the parent experiment or between
separate builds do not isolate a fused-kernel gain. Packed execution still
loses at every quota.

TK64 GPU milliseconds (staged -> fused), with paired fused/staged ratios:

| Corrections | A | B | Paired A / B |
| --- | ---: | ---: | ---: |
| 0 | 58.98600 -> 780.29600 | 58.39256 -> 790.43448 | 13.22305x / 13.51645x |
| 1 | 60.22794 -> 843.52979 | 58.76656 -> 849.62454 | 14.05541x / 14.44161x |
| 4 | 59.18977 -> 969.80502 | 58.83100 -> 976.80850 | 16.44724x / 16.57646x |
| 8 | 61.22521 -> 1134.96894 | 59.96471 -> 1143.96444 | 18.57666x / 19.00054x |

The same-direction packed regressions confirm that halving panel barrier
pairs did not improve this implementation. They do not establish occupancy,
register pressure or another particular hardware mechanism as the cause.

## Numerical, failure and resource checks

Final source passes 18 native Metal examples and 28 analyzer tests. Earlier
M128 and BT32 native suites pass 16 and 17 examples. Tests cover
heterogeneous non-dyadic rows, M/N tails, K256/K512, quotas, signed zero,
malformed duplicate/NaN/Inf records, neighboring valid slices, sticky
status, alias/size/dimension rejection, poisoned output preservation,
pipeline geometry and tracked encode allocations. Analyzer negatives cover
missing parity/status, schedule/geometry/pipeline/resource tampering,
nonfinite metrics and inconsistent source/fixture/build receipts. An
initial analyzer test found a result-shape bug (`timings_ms.dense_gemm`);
it was corrected and the red receipt is retained. A first combined analysis
also rejected historical source pins after the checkout changed. Source
archives already contained every pinned file; the analyzer now checks probe
and build pins against the exact build-tag archive and requires one complete
matching snapshot for each oracle. Native objects, binaries and fixture files
remain checked against their live bytes. The archive regression initially
exposed a `/opt` versus resolved external-volume path mismatch. Parent-path
canonicalization preserves the source filename, so a retargeted live source
symlink cannot redirect historical lookup. Partial/missing/mutated archives
and mutated external fixture/object pins are rejected. All red receipts are
retained; the final guarded analyzer suite passes 28 tests.

All completed probes check every coefficient/output against the pinned
independent same-format upstream before and after timing. They also check
cross-implementation parity, seeded finite-plus-NaN corruption rejection
and actual final GPU status words. Coefficient exactness and the output
finite tolerance are separate gates. The oracle producer, fixtures and
manifest hashes are verified at every probe. Independent PyTorch MPS is
used only for offline reference construction, with CPU fallback disabled;
it is outside all measured execution. The unchanged nine CPU reference
tests are inherited evidence, not rerun in this slice.

Across 396 parity metric groups there are zero nonfinite values and zero
tolerance violations. Every M4480 output is bit-exact with its same-format
reference. The largest M4 absolute difference is 2.336502e-5 and the largest
error/tolerance ratio is 0.33293873. A maximum absolute difference larger
than 2e-5 can pass the unchanged relative-plus-absolute gate.

All 2,244 timing samples, including warmups, have one terminal wait and zero
inter-operator host waits, activation readbacks or tracked explicit encode
`ML::MetalBuffer` allocations. Full diagnostic readbacks and status
validation occur outside the measured chain. Threadgroup barriers are local GPU
synchronization. Opaque MPS/compiler resources are excluded from the explicit
buffer counter but included in the sampled process guard.

Cache-aware headroom includes reclaimable file-backed pages. Preserve the
15% physical reserve and pressure floor, 4 GiB process-group cap, 64 MiB
system swap-growth stop and deadlines. Sampling cannot exclude between-poll
spikes, and reclaimability is an estimate. All temporary/build/cache/evidence
files stay under `/opt/mldata_ssd`. Across native probes the sampled process
group peak is 2.46875 GiB, minimum estimated cache-aware headroom is
15.34151 GiB, pressure-free is at least 66%, and system swap growth is zero.
No full model, block, Studio or generation is launched.

The fused candidate binds zero global decoded-weight stage bytes versus
283,115,520 control bytes. The combined harness retains that stage, all
formats and diagnostic resources: this does not measure a process-footprint
saving or model-wide fit benefit. Package bytes, correction allocation and
lossy distortion are unchanged. Same-format parity does not establish BF16
quality equivalence; the parent four-row output relative L2 remains
4.63165%/4.57702%/4.44768%/4.30013% for quotas 0/1/4/8.

## Reproduction and lineage

Working tree: `codex/wan-cross-segments`, starting at `4fab8597bc913050cb64669aa816d9df2f5e0a55`.
Build receipts record macOS 26.6.2. Final native test initialization reports
Apple M2 Max with unified memory. Let
`L=examples/support/wan_q4k_delta_experiment.py`, and execute with
`python3 -B $L` in this worktree. Fresh exclusive tags are required.

| Arguments | Observed signal |
| --- | --- |
| `spec dense-reuse-native-v1` | 16 examples, zero failures/errors/pending |
| `build dense-reuse-v1` | Source-bound release build |
| `spec dense-reuse-native-v2` | 17 examples, zero failures/errors/pending |
| `build dense-reuse-v2` | Source-bound release build |
| `analysis-spec dense-reuse-analysis-v2` | 27 tests, OK |
| `analysis-spec dense-reuse-analysis-v4` | 28 tests, OK, including historical archive negatives |
| `spec dense-reuse-native-v3` | 18 examples, zero failures/errors/pending |
| `build dense-reuse-v3` | Source-bound release build |

The probe command is `probe TAG --build-tag BUILD --fixture-tag FIXTURE
--compare-gemm --gemm-variant=VARIANT` for dense, or `--compare-fused` for
packed. B processes add `--reverse-first`. M4 uses `fused-r4-v1`; M4480
uses `fused-r4480-v1`. Exact tag/build/variant mapping:

| Build | Variant | Dense tags | Packed tags |
| --- | --- | --- | --- |
| `dense-reuse-v1` | `candidate-b128n32-sg4-tk32` | `dense-reuse-r4-v1`, `dense-reuse-a4480-v1`, `dense-reuse-b4480-v1` | None: wider-M speed hypothesis rejected |
| `dense-reuse-v2` | `candidate-b64n32-sg4-tk32-bt` | `dense-bt-r4-v1`, `dense-bt-a4480-v1`, `dense-bt-b4480-v1` | `dense-bt-fused-r4-v1`, `dense-bt-fused-a4480-v1`, `dense-bt-fused-b4480-v1` |
| `dense-reuse-v3` | `candidate-b64n32-sg4-tk64-bt` | `dense-k64-r4-v1`, `dense-k64-a4480-v1`, `dense-k64-b4480-v1` | `dense-k64-fused-r4-v1`, `dense-k64-fused-a4480-v1`, `dense-k64-fused-b4480-v1` |

Evidence base: `/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-q4k-delta-20261023`.
Fixtures are reused unchanged from the parent. M4/M4480 manifest SHA256:
`36b796df8ff172609cc78e75078d11dbf60446e82f3e4861fada4a2e76542891` /
`204432b92773cf54b9db9482d1216e533e249a2e3d26a9ae244c2902860a5af3`.
Build v1/v2/v3 intent SHA256:
`727b9ff9a48a2a0988d8da4284ec5d283aed1943f3bee53ca58218d489ea9bcc` /
`779a7ddd916f032c71d65478d3be4135f241e5810313b3374233f1b342e3fffc` /
`bf24730765b83e0ca96f3c4d6c0c78fba65b6194915a0dcfe07ebca052f6f715`.

The guarded strict analyzer consumes all 15 probe receipts and their logs,
checksums and guard samples. `analysis-native-reuse-v2` returns zero with
`changed_files: []`, as does the final guarded 28-test analyzer suite
`analysis-spec-dense-reuse-analysis-v4`. Summary SHA256:
`948b1fd8a76319b1603fd6268803b1f7c753c66676d7c14ebba72e238dec6ec0`.
Analysis intent / completion SHA256:
`ed5c28f338d4177557ae521546b5dc8fdba4d2b5449583ee3cdf86ffd4d955cb` /
`9199d3032e1d3f77c71e14c948c1a3186f482280dffda281c592f72638470135`.
Final analyzer-suite completion SHA256:
`9a921e12f14446c06aaf364b08e9d5d4b7415f648ad186a8dfb13c21a54e5ecd`.
Final native-suite completion SHA256:
`8215deaffac1b7defa4402b80d084c97b17d1dd17eeb6d6965726eb6d2b0619e`.
The exact guarded analysis command is recorded in the intent; its pinned
wrapper is `build/wan-q4k-delta-20261023/analyze-native-reuse-v2.py` under
the external experiment base. It refuses to overwrite existing receipts;
use a fresh prefix for a repeat. Launcher
receipts archive the source and bind binary/native objects, fixture producer,
manifest, schedule, policy and output files. All successful invocations
record `changed_files: []`. Six native objects and the ggml quantizer are
pinned and reused. Only the T5 linear bridge has complete historical
source/object ancestry; five other bridges have current pins without
reconstructed historical build lineage. No bridge/object implementation
changes in this experiment.

Build archives match all 248 repository source pins per build. The oracle
source subsets contain 13 pins each and each has a complete matching archive.
This is local byte consistency against captured receipts, not signed
provenance against an actor able to rewrite every artifact. Replacing a
checkout parent-directory symlink can cause a historical path to be treated
as external and rejected; that relocation case is not admitted here.

## Decision and next falsifier

Retain staged MPS and legacy production selection. BT32 is an optional
bounded native improvement; neither larger panels nor packed fusion is
promoted. A competitive exact F32 dense kernel remains an open obligation.
Do not treat the smaller global binding inventory or the 2% native gain as
model-wide memory, quality or throughput evidence.

The next experiment must change the cost frame rather than sweep more tile
sizes. Candidate: direct device F32 SIMD matrix loads, with a safe tail path,
to test the cost of panel staging/barriers. API support and correctness must
be established first; a dense-only win would still require a separate packed
return. No speed, hardware-cause or packed-transfer claim is made here.
Refresh numerical, failure, timing and memory evidence after source/dispatch,
input/geometry, payload/sidecar, compiler/native objects, device/OS/SDK,
tolerance or schedule changes. Correlated Mini-Quad: ROBUST for bounded
parity, failure controls, observed guards and the two-process BT32 native
gain; VULNERABLE for representative quality, hardware-cause inference,
model-wide memory/speed or superiority to MPS. Those broader claims are
not admitted by this comparison.
