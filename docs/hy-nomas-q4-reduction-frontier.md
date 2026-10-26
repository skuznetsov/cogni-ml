# Hy-Nomas direct Q4 reduction diagnostic

Current frontier, 2026-10-26: same-geometry dense F16 loading takes
4.277-4.279 ms at B64 versus packed Q4's 4.683-4.693 ms, with bitwise equal
outputs in both initial orders. This is about 8.7% less complete-projection
time, not a pure decoder-cost measurement. Dense B128 remains slower than
dense B64; even dense B64 remains slower than resident F32 MPS. These tested tile and
loading variants do not close the native GEMM gap.
See "Same-geometry Q4/F16 tile frontier" for exact scope, separate profile
coverage, numerical gates and the next falsifier. Production dispatch is
unchanged; the original-BF16 quality gate still fails. No generation was run.

Original diagnostic status: VERIFIED within its scope, 2026-10-25; the direct Q4 numerical
gate remains failed. Localize the 17 finite-tolerance failures on
the authenticated block-0 text FF down projection (M=23, K=8192, N=2048).
This diagnostic does not change production kernels or run generation.

The competing explanations are different F16 coefficients after Q4 decoding,
different accumulation with identical operands, or a bias/edge-layout error.
The existing gate remains `atol=rtol=2e-5`; Q4 versus original BF16 quality is
a separate frontier and is not qualified by an identical-operand comparison.

Risk: CAUTION (diagnostic GPU resource lifetimes and derived kernel source).
Rollback: remove the opt-in example; production dispatch and defaults do not
change. The strongest failure mode is a dense control that changes more than
weight loading and therefore cannot isolate the Q4 decoder. Derive it from the
actual Qwen F16 kernel with checked source anchors, and require full-output raw
identity with the actual direct kernel before interpreting reduction results.

Definition of done: build and run `hy_nomas_q4_reduction_probe` under the existing
cache-aware guard with a 10% reserve, 64 MiB swap-growth stop, a bounded process
footprint and deadline. Compare every F16 coefficient on GPU, including an
injected single-bit comparator control. Compare dense SIMD and MPS on identical
rounded operands, before/after the same bias, and at selected K prefixes. Keep
all resources rooted, use one command chain and one terminal wait/readback,
and require zero tracked encode allocations or early reads. Record the original
failures without widening tolerances. Interpret a compensated F32 dot product
only as a diagnostic numerical control, not a proven correctly rounded oracle.

The pinned upstream input and source/cache identities are inherited from the
previous authenticated projection fixture. Refresh after source, cache,
geometry, compiler, device, OS, precision or resource-lifetime changes. A failed
control invalidates the attribution, even if the aggregate tolerance improves.

## Observed result

`probe-v2` completed on Apple M2 Max, macOS 26.6.2 (25G83), SDK 27.0,
Crystal 1.22.1 / LLVM 23.1.4. The independent canonical decoder and the actual
Qwen `dequantize_q4_K_fn` agree at every F16 bit across 16,777,216 coefficients
in 65,536 Q4 blocks. There are no nonfinite decoded values. The same GPU
comparator reports zero differences for an identical-buffer control and exactly
one difference at index 17 when that coefficient's low bit is virtually flipped.

The dense F16 clone preserves the Qwen SIMD geometry, shared-memory panel
layout, MMA order and edge stores. It reproduces all 47,104 direct-Q4 outputs
bitwise, both before and after the common BF16 bias. The canonical/helper MPS
branches also reproduce each other bitwise. This refutes a decoder mismatch
for these weights and ties the standalone helper capture to the actual GEMM
result for this input. It is not an independent arithmetic oracle or a claim
about every Q4 tensor, geometry or activation.

| Identical-operand comparison | Raw bit differences | Failures at finite 2e-5 |
| --- | ---: | ---: |
| Dense SIMD versus MPS, K=32 | 0 | 0 |
| Dense SIMD versus MPS, K=256 | 0 | 0 |
| Dense SIMD versus MPS, K=2048 | 0 | 0 |
| Dense SIMD versus MPS, K=4096 | 0 | 0 |
| Dense SIMD versus MPS, K=6144 | 45,713 | 5 |
| Direct Q4 versus MPS, K=8192, before bias | 45,856 | 17 |
| Direct Q4 versus MPS, K=8192, after bias | 45,853 | 17 |

The same 17 indices fail before and after bias. At full K, maximum absolute
difference is 0.000247955322265625, post-bias RMSE is 0.000018544793163295328,
and relative L2 is 0.0000014113482393796142. All output banks are finite and
nonconstant. This localizes the discrepancy to the long-K numerical
accumulation path. The mechanism inside MPS, including its reduction schedule,
internal precision, fusion or kernel selection, is not established. Prefix
probes change the MPS K geometry; they are not snapshots inside its K=8192
dispatch, and untested K values are not qualified.

Against a GPU Neumaier compensated F32 sum of the same F16 products, MPS is
closer on 12 of the 17 failing entries and Qwen on five. The compensated control
is not a correctly rounded oracle, so this does not establish that either
backend is uniformly more accurate. No production kernel, rounding policy or
tolerance was changed, and no speedup is claimed from combined diagnostic
timing. A bounded next discriminator is alternative partial-sum grouping at
full K; do not promote it without a fresh upstream execution gate.

The authenticated saved upstream fixture supplies the identical activation.
Its original-BF16-weight output remains a separate lossy comparison: direct Q4
has 47,099 failures and RMSE 0.617303 against it. This diagnostic neither fixes
nor qualifies Q4 quality relative to original checkpoint weights. A matching
same-operand MPS comparison cannot substitute for that quality gate.

## Execution and evidence

Each native run uses one command chain/segment, one terminal wait and one packed
terminal readback, with zero tracked encode-time MetalBuffer allocations or
preterminal reads. The v2 named GPU buffer budget is 229,455,088 bytes. MPS and
driver-private memory are outside that count. The external guard retained its
10% file-cache-aware reserve, 2 GiB process cap, 64 MiB system swap-growth stop
and 120-second deadline, exited zero and observed no swap growth. Its short-run
samples miss active Metal residency and do not establish a reliable peak.
Minimum sampled cache-aware headroom was 38,989,561,856 bytes. File-backed
headroom remains a reclaimability estimate, not a guarantee of immediate frees.

The v1/v2 runs reproduce all 16 common output-bank hashes. v2 preserves the
3,768,440-byte packed terminal sidecar, all 20 bank hashes, 30 comparator words,
all 17 failing entries and before/after-bias indices. A separate postterminal
Python check recomputed all 15 comparison summaries, hashes, finite/nonconstant
checks and comparator/guard conditions. Correlated Mini-Quad review independently
recomputed those bytes and returned ROBUST for this scoped finding. It did not
identify MPS internals or qualify full-block or generation behavior.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-q4-reduction-20261025`.
Primary report: `reports/probe-v2.json`, SHA-256
`7d09512e91c3c84966c2c46141fb4236361b49e6986b22affe1669c9d3f3f26e`.
Packed output SHA-256:
`bbe709ef2cb509ad2005cf29cf65b632791e9f68d9aa694282893bec77c8004c`.
Selected sources, binary and link-object identities are retained under
`source/native-v2*` and `build/`; this is post-run identity capture, not a
hermetic build attestation. The repository evidence packet indexes the files.

Executed commands (the preserved runner supplies the declared guard, TMPDIR and
Crystal cache settings; exact argv is retained in `logs/*.argv.json`):

```sh
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-reduction-20261025/tmp/run_guarded.py build-v3 4096 600 make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1 hy_nomas_q4_reduction_probe
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-reduction-20261025/tmp/run_guarded.py probe-v2 2048 120 /opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1/hy_nomas_q4_reduction_probe --report /opt/mldata_ssd/cogni-ml/hy-nomas-q4-reduction-20261025/reports/probe-v2.json
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-reduction-20261025/source/verify_terminal_v2.py
crystal tool format --check examples/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction_probe.cr
git diff --check
```

Low-budget admission and an existing report target both reject before Metal
initialization, with the existing report unchanged. Expected nonzero exits are
recorded in `reports/admission-checks-v1.json`. The positive probe's exit zero
means the diagnostic completed, not that direct Q4 passed its MPS parity gate.

## Full-K grouping experiment, 2026-10-26

Status: VERIFIED for this one-projection diagnostic; correlated Mini-Quad
Adversary verdict ROBUST. At M=23/K=8192/N=2048, two contiguous F32 partial sums
over K=0..4095 and K=4096..8191, followed by one F32 addition, reproduce the
unchanged full-K MPS output bitwise across all 47,104 values. This holds before
and after adding the common BF16 bias once. Both merge implementations agree;
for two partials, their addition trees are equivalent.

The experiment keeps the authenticated decoded F16 operands and full-K MPS
geometry. It partitions native dense SIMD accumulation into 1/2/4/8/16/32 groups,
either contiguous or strided by 32-element tiles. Left-fold and adjacent balanced
tree merges consume the same partial buffers. Failures at the unchanged finite
`atol=rtol=2e-5` gate, before bias:

| Groups | Contiguous fold | Contiguous tree | Strided fold | Strided tree |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17 | 17 | 17 | 17 |
| 2 | 0 | 0 | 59 | 59 |
| 4 | 20 | 20 | 35 | 34 |
| 8 | 17 | 19 | 27 | 28 |
| 16 | 19 | 18 | 23 | 22 |
| 32 | 19 | 18 | 18 | 19 |

Only contiguous G2 passes, with zero bit differences, maximum error and RMSE,
both raw and biased. The one-group controls match the existing dense SIMD
baseline bitwise; its original 17 failures remain unchanged. GPU tile counters
record one visit for each of all 256 K32 tiles in every case. They count
aggregate visits from a representative output threadgroup, not ownership by
partition. Partition ownership is supported by the reviewed start/stride
arithmetic, not independently established by those counters. Reused partial
storage is poisoned with NaNs before each case; all output banks are finite
and nonconstant.

Guarded release build and two native runs passed. All 68 output-bank hashes are
identical between runs, including all 20 banks pinned by the previous report.
Each run retains one command segment, one terminal wait/readback, zero tracked
encode-time buffer allocations and zero preterminal reads. Named GPU storage is
253,596,912 bytes, including 6,029,312 bytes of reusable partials; MPS/driver
private storage is excluded. Both guards preserve the 10% file-cache-aware
reserve, 2 GiB process budget, 64 MiB swap-growth stop and 120-second deadline,
exit zero and observe no swap growth. Minimum sampled headroom is
37,356,142,592 and 38,283,542,528 bytes. These samples do not establish peak
memory or guarantee immediate file-cache reclamation.

The separate postterminal checker recomputes all 112 comparison summaries per
run, raw buffer hashes, decoder controls, K coverage and guard conditions.
Its first two attempts failed in the checker: JSON shortest serialization of
Float32 tolerances was compared directly with binary64, then Python rejected a
non-byte-to-non-byte memoryview cast. Preserved v3 authenticates the serialized
literal and uses the original Float32 value for thresholds; no native tolerance
changed. Budget/report/sidecar admission negatives pass before Metal starts.
The checker also rejects a widened tolerance, a fabricated grouping pass and
an altered output-bank hash without modifying the original artifacts.

The saved original-BF16 upstream comparison remains a failed, separate lossy
gate: contiguous G2 has 47,098 failures and RMSE 0.6173029922, versus 47,099 and
0.6173029243 for the original direct Q4 result. This is not a quantization-quality
fix. The candidate uses the dense decoded-F16 diagnostic kernel; the direct
packed-Q4 kernel is unchanged and remains unqualified. Matching this grouping
does not identify MPS internals or qualify other inputs/shapes. Combined probe
timing is not a kernel speed benchmark. No full block or generation was run.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026`.
Reports: `reports/group-v1.json` and `reports/group-v2.json`; both packed sidecars
have SHA-256 `ee12b3189b70c8fe7b291888cd500f51344a0654be90b924f235ce496e98ec93`.
Selected sources, binary and link-object identities are retained under
`source/native-v1*` and `build/`; this is post-run identity capture, not a
hermetic build attestation. The repository evidence packet indexes the files.

Executed DoD commands (runner supplies guards, TMPDIR and Crystal cache):

```sh
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/tmp/run_guarded.py build-v1 4096 600 make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1 hy_nomas_q4_reduction_probe
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/tmp/run_guarded.py group-v1 2048 120 /opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1/hy_nomas_q4_reduction_probe --report /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/reports/group-v1.json
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/tmp/run_guarded.py group-v2 2048 120 /opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1/hy_nomas_q4_reduction_probe --report /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/reports/group-v2.json
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/tmp/run_guarded.py verify-v3 1024 120 python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/source/verify_grouping_v3.py /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/reports/group-v1.json /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/reports/group-v2.json
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-grouping-20261026/source/admission_checks_v1.py
crystal tool format --check examples/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction_probe.cr
git diff --check
```

Artifacts use fresh paths and reject overwrites; use a new run identity for a
repeat. Next falsifier: transfer the two contiguous partials to direct packed-Q4
GEMM without expanded global weight storage, then repeat same-operand and
upstream gates before timing or promotion. Rollback: revert the opt-in diagnostic
commit. Refresh on changed source/cache/fixture, geometry, precision, ownership,
compiler/SDK/OS/device or MPS runtime.


## Direct packed-Q4 split frontier, 2026-10-26

Status: VERIFIED for the bounded same-operand projection diagnostic. The old
direct kernel retains 17 same-F16-operand MPS failures on authenticated
M23/K8192/N2048 operands. The new opt-in probe transfers the measured contiguous
G2 dense grouping to packed Q4 while
retaining the full packed row stride, the existing Qwen decoder, SIMD geometry,
and F32 MMA order inside each K4096 partition. Bias is applied once after merge.
This is a CAUTION numerical change confined to the opt-in diagnostic; production
kernels and defaults remain unchanged. Rollback is a revert of this diagnostic
commit.

Falsifier-first prediction: packed G1 must reproduce the old direct kernel
bitwise, and packed G2 must reproduce dense G2 and full-K MPS bitwise, raw and
biased. Any mismatch rejects qualification. Preserve all 68 old output hashes
and unchanged finite Float32 atol=rtol=2e-5. Compare against the saved original
BF16 upstream output separately; the existing Q4 lossy gate is not waived.

The strongest likely implementation failure is treating a partition as a short
packed row: z=1 must start 16 Q4 blocks into each original 32-block row, and both
output store paths must use the z offset. A GPU counter and partition-owner mask
for every K32 tile, a poisoned reused partial buffer, G1 controls, and all-output
packed-versus-dense comparisons guard these failures. Ownership counters cover a
representative output threadgroup, not every threadgroup. The packed branch
reads the resident Q4 payload directly and introduces no global expanded-weight
buffer; the diagnostic still retains decoded-weight buffers for its MPS and
dense controls.

DoD: guarded release build; two fresh native packed-v1/v2 reports; a separate
postterminal checker recomputing output summaries, all previous bank hashes,
coverage and ownership, source identities and guard evidence. Require one
command segment, one terminal wait/readback, zero tracked encode allocations or
preterminal reads, 10% file-cache-aware reserve, 2 GiB process cap, 64 MiB swap
stop and 120-second run deadline. Add admission/validator negative controls and
run Crystal formatting and git diff --check. Combined diagnostic timing is not
a kernel speed benchmark. No full block or generation. Evidence root:
/opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026. Refresh on changed
source/cache/fixture, geometry, precision, ownership, compiler/SDK/OS/device or
MPS runtime.


The guarded release build and two native runs passed. Direct packed G2 matches
both dense G2 and full-K MPS bitwise over all 47,104 raw outputs and all 47,104
biased outputs: zero tolerance failures, bit differences, maximum error and
RMSE. Packed G1 reproduces the old direct kernel bitwise, including its 17 MPS
failures. All 72 output banks repeat across runs; all 68 previous bank hashes
remain fixed. GPU decoder self/single-bit controls remain unchanged.

The packed kernels retain the Q4 decoder in shared-memory tile loading. They
bind the original resident payload and reuse the existing partial buffer; they
have no dependency on a globally expanded weight buffer. Each packed G1/G2
coverage slice records all 256 K32 tiles exactly once with the expected owner
mask. Counters track computed loop offsets in one representative output
threadgroup; actual load correctness is supported by source review and the
all-output comparisons, not address-level instrumentation. The exercised output
store is the cooperative edge path for M=23; fast-store geometry is unqualified.

Both runs retain one command segment, one terminal wait/readback, zero tracked
encode-time buffer allocations and zero preterminal reads. Named GPU storage is
255,112,432 bytes, including all dense/MPS controls and the unchanged 6,029,312
bytes of reusable partial scratch. Guards retain the 10% file-cache-aware floor,
2 GiB process cap, 64 MiB swap-growth stop and 120-second deadline. Sampled minimum
headroom is 36,171,628,544 and 36,261,904,384 bytes with zero observed swap growth;
short-run samples do not bound the Metal/MPS peak or guarantee cache reclamation.

The separate checker recomputes all 122 comparison summaries per run from the
captured terminal bytes and saved upstream fixture, verifies the 68 previous
bank hashes, raw decoder/coverage/owner words and guard evidence. Five CPU-only
checker tests pass. Four additional captured-data falsifiers reject widened
tolerance, a fabricated passing packed-G1 comparison, a flipped G2 owner mask
and an altered packed-bank hash. Budget, report-collision and sidecar-collision
admission tests reject before Metal initialization without changing existing
reports. Native exit zero means diagnostic completion; G2 qualification comes
from the separately recomputed byte comparisons, not its summary booleans.

The original-BF16 upstream lossy gate remains failed: packed G2 has 47,098
failures, RMSE 0.6173029922206691 and relative L2 0.0470375792911417. No quality,
kernel speedup, production-default, full-block or generation claim follows from
this result. Mini-Quad source and terminal-evidence review is correlated; its
ROBUST verdict is limited to this projection, input and representation.

Evidence reports are reports/packed-v1.json and reports/packed-v2.json under the
external root above; their sidecars share SHA-256
6f13dfede1e5fab0c893cfd1809b050f16e2923db11c8116fdee93d10357000f.
The selected 18-file source snapshot predates native execution and was rechecked
after both runs; source/native-v1-receipt.json records the frozen binary and link
objects. This is source/build identity evidence, not a hermetic attestation.
The generic Landmark packet template/graph validator is absent in this checkout;
the curated index is instead JSON-parsed and every referenced external file is
digest-checked. That checks index integrity, not the numerical claim itself.

Executed DoD commands (the runner supplies guards and external TMPDIR/cache):

```sh
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/tmp/run_guarded.py build-v1 4096 600 make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1 hy_nomas_q4_reduction_probe
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/tmp/run_guarded.py packed-v1 2048 120 /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/build/hy_nomas_q4_direct_split_probe-v1 --report /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/reports/packed-v1.json
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/tmp/run_guarded.py packed-v2 2048 120 /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/build/hy_nomas_q4_direct_split_probe-v1 --report /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/reports/packed-v2.json
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/tmp/run_guarded.py verify-v1 1024 120 python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/source/verify_direct_split_v1.py /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/reports/packed-v1.json /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/reports/packed-v2.json
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/source/test_verify_direct_split_v1.py
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/source/admission_checks_v1.py
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/tmp/run_guarded.py checker-pressure-v1 1024 120 python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-direct-split-20261026/source/checker_pressure_v1.py
crystal tool format --check examples/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction_probe.cr
git diff --check
```

Next falsifier: isolated paired packed-G1/G2 versus MPS timings on identical
operands, then numerical qualification at larger real token counts before any
full-block integration. The original-BF16 lossy gate remains separately open.

## Paired projection benchmark frontier, 2026-10-26

Status: bounded benchmark verified; general G2 promotion rejected at M1280.
The implementation is opt-in; production kernels and defaults remain unchanged. This CAUTION
slice removes diagnostic coverage atomics from a source-derived packed kernel,
then compares contiguous G1/G2 including tile decoding, F32 merge and one bias
against resident MPS and Q4-to-F32 expansion plus MPS separately. Resident MPS
retains a 64 MiB F32 matrix; that storage and setup are not free. Rollback is a
revert of this benchmark slice.

The falsifier is an unchanged-tolerance mismatch, changed M23 baseline hash,
nonfinite or stale output, invalid/shared GPU timestamp range, order-dependent
performance reversal, intermediate readback/wait, or named allocation during
encoding. A GPU positive control must make the output comparator fail. Poisoned
correctness controls and output checks stay outside measured roots. Each timed
root contains only repeated complete projection calls, with balanced forward
and reverse case order. Asynchronous root continuations are diagnostic command
boundaries, not CPU completion waits. Exactly one terminal wait/readback per
run remains mandatory. Report CPU encode time separately from GPU root time.

DoD: release build under the existing guard, two order-reversed M23 runs,
postterminal byte-based comparison/timing checks, then M32 and M1280 geometry
checks. Larger M initially repeats the authenticated 23 real activation rows on
GPU, so it checks store geometry and shape-dependent MPS behavior, not a new real
full-prefix distribution. Stop promotion if the larger-M gate fails. Preserve
Float32 atol=rtol=2e-5, a 10% file-cache-aware floor, 2 GiB run cap, 64 MiB swap
growth stop and 120-second run deadline. No full block or generation, and no
full-video speed or original-BF16 quality claim. Artifacts belong under
/opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026. Refresh after source, input,
cache, geometry, order/repetition schedule, compiler/SDK/OS/device or MPS drift.

Five guarded runs on Apple M2 Max exercise K8192/N2048, with 16 complete
projections per timed root, two warmup and six measured rounds per arm. Each
round alternates forward/reverse arm order; M23 and M1280 additionally have
separate reverse-first runs. The table contains median GPU milliseconds per
projection; ranges span the two runs, not confidence intervals.

| M | Packed G1 | Packed G2 | Resident F32 MPS | Q4 expand + MPS |
|---|---:|---:|---:|---:|
| 23 | 0.340-0.343 | 0.239-0.240 | 0.320-0.339 | 0.532-0.547 |
| 32 | 0.348 | 0.238 | 0.336 | 0.550 |
| 1280 | 5.040-5.312 | 4.999-5.302 | 3.999-4.151 | 4.185-4.402 |

All 12 measured M23 pairs favor G2 over resident MPS (paired speed ratio
1.199-1.562, median 1.377); all 12 M1280 pairs favor MPS (resident/G2 ratio
0.768-0.834, median 0.798). Thus the small-M speedup does not generalize to
large M. M1280 G1 is also slower than MPS including expansion. These are warm,
isolated repeated-projection timings; they do not establish a full-block,
mixed-layer cache, cold-load or generation speedup. No profiler is enabled.

Numerical gates remain finite F32 atol=rtol=2e-5. Packed G2 equals dense G2,
packed G1 equals the original direct kernel, and expanded MPS equals resident
MPS bitwise at every tested M, before and after bias. G2 matches MPS bitwise at
M23/M32; G1 has 17 failures at M23 and 27 at M32. At M1280, G1 instead matches MPS bitwise;
G2 has 949 failures per 2,621,440-element raw/biased output, max absolute error
0.000247955322265625 and raw RMSE 1.8545895110040667e-5. Both M1280 runs reproduce
every bank hash and failure. The all-M G2 promotion falsifier therefore fires.

The postterminal cross-geometry byte check localizes the changed result to MPS:
all eight direct/dense/packed raw/biased banks remain exact cyclic repeats of
their M23 bank at M32/M1280. Only the four MPS banks change at M1280; all four
then equal cyclic M23 G1 bytes. This supports a shape-dependent accumulation
interaction, but does not identify MPS's private reduction algorithm. The
separate original-BF16 saved-output gate still fails (M23 G2 RMSE 0.617303).
Neither same-quantized-operand parity nor this repeated-row geometry check
qualifies original-BF16 quality or a real full-prefix activation distribution.

Each run has 65 asynchronous command roots, zero timed-root overlap, one
terminal wait/readback, zero tracked encode-time MetalBuffer allocations and
zero preterminal reads. Root timestamp intervals are checked against every
other root and rejected on overlap; they are not kernel hardware counters.
Named storage including all correctness controls is 192,699,488 / 195,132,512 /
532,511,840 bytes for M23/M32/M1280. The projection's packed weight is 9 MiB
versus the resident MPS matrix's 64 MiB. The benchmark retains both plus a
32 MiB F16 dense control, so it does not measure a minimal production footprint.
MetalBuffer counters do not include private MPS allocations. Guards pass with
minimum sampled headroom 31,811,829,760 bytes and no observed swap growth;
sampling does not establish peak memory or guarantee file-cache reclamation.

The separate checker recomputes ten same-operand and three saved-BF16 summaries
per run from terminal bytes, verifies all 140 raw comparator words, M23 baseline
hashes, schedules, 65 GPU-root timestamps and guard logs. Decoder and output
comparators both include injected-bit positive controls. Thirteen CPU checker/
analyzer tests pass; seven captured-report falsifiers reject widened tolerance,
forged G1 success, wrong segment/root counts, overlap, bank hash and metric words.
Five admission negatives reject before Metal initialization and preserve
existing artifacts. Correlated Mini-Quad source/evidence review is ROBUST only
for this bounded diagnostic. Native exit zero alone does not establish parity.

Evidence is indexed by .landmark/evidence/2026-10-26-hy-nomas-q4-paired.json.
Reports and sidecars are paired-m23-forward-v1, paired-m23-reverse-v1,
paired-m32-forward-v1, paired-m1280-forward-v1 and paired-m1280-reverse-v1 under
the external reports directory. verify-small-v1.json, verify-large-v1.json and
cross-geometry-v1.json contain postterminal results. The selected 21-file source
snapshot predates native execution; its frozen binary and ten linked objects
were rechecked after all runs. This is not a hermetic build/driver attestation.

Executed DoD entry points (the runner supplies the unchanged guards):

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1 hy_nomas_q4_paired_benchmark
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/tmp/run_guarded.py paired-m23-forward-v1 2048 120 /opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/build/hy_nomas_q4_paired_benchmark-v1 --report /opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/reports/paired-m23-forward-v1.json --rows 23 --repeats 16
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/source/verify_paired_v1.py /opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/reports/paired-m23-forward-v1.json /opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/reports/paired-m23-reverse-v1.json /opt/mldata_ssd/cogni-ml/hy-nomas-q4-paired-20261026/reports/paired-m32-forward-v1.json
crystal tool format --check examples/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_paired_benchmark.cr
git diff --check
```

Next falsifier: retain full-K G1 accumulation at M1280 while reducing repeated
Q4 decoding across M tiles, first against the same MPS controls. Keep G2's
small-M result separate; no dispatch threshold is qualified between M32 and
M1280. Original-BF16 quality, real full-prefix inputs, full-block performance
and generation remain open. No full generation was launched.

## Full-K weight-tile reuse frontier, 2026-10-26

Status: VERIFIED within the bounded M1280 projection diagnostic. The existing
Qwen B64 and exact-tile B128/SG8 kernels share each decoded 64x32 weight tile
across more input rows. Both are tested on the same M1280/K8192/N2048 operands
against M32 G1, resident F32 MPS, and Q4 expansion plus MPS. The existing
one-group F32 merge and bias remain in all packed timed arms so this probe
changes M reuse rather than fusing work.
Production kernels and dispatch policy remain unchanged.

Risk: CAUTION (GPU scratch ownership and changed launch geometry). Rollback:
revert the opt-in benchmark change. The strongest numerical failure mode is
misrouting B128's two input bands; require all raw and biased outputs to match
G1 bitwise, check the original 23 rows separately from cyclic repeats, and
retain the unchanged finite `atol=rtol=2e-5` MPS gate. The performance falsifier
is that register/shared-memory pressure erases the nominal 2x/4x reduction in
weight-tile decodes. This is not a hardware-bandwidth claim.

DoD: guarded release build of `hy_nomas_q4_tile_reuse_benchmark`; native M1280
runs with both initial warmup orders, two warmup and six measured rounds per
arm; postterminal replay of all output bytes, injected comparator controls,
timing/root isolation and guard logs. Require one final wait/readback, zero
tracked encode-time MetalBuffer allocations/early reads, the file-cache-aware
10% floor, 2 GiB process cap, 64 MiB swap-growth stop and 120 s deadline.
Admission negatives must reject before Metal initialization. No full model,
block or generation is admitted. Original-BF16 quantization quality remains a
separate gate; M1280 still repeats the same 23 real rows on GPU.

Two guarded native runs on Apple M2 Max use opposite initial warmup orders;
each alternates forward/reverse rounds, with 16 complete calls per timed root.
Ranges below are the two run medians, not confidence intervals. Qualification,
poisoning, comparators and packing remain outside the isolated timed roots.

| Complete projection | Median GPU ms/call | M tile | Threads | Shared bytes |
|---|---:|---:|---:|---:|
| Packed G1 control | 5.039-5.042 | 32 | 128 | 12,288 |
| Packed B64 G1 | 4.706-4.709 | 64 | 256 | 16,384 |
| Packed B128/SG8 G1 | 4.697-4.704 | 128 | 256 | 24,576 |
| Resident F32 MPS | 4.001-4.003 | internal | internal | internal |
| Q4 expansion + MPS | 4.183-4.206 | internal | internal | internal |

All 12 measured paired rounds favor B64/B128 over M32 G1. The median paired
speed ratios are 1.070912 and 1.073027 respectively, corresponding to roughly
6.6%/6.8% less time. All 12 rounds still favor both MPS controls. Resident/B128
ratios are 0.846368-0.853107, median 0.851470. B128 has no demonstrated material
advantage over B64 despite nominally fourfold rather than twofold fewer
weight-tile decodes. These are repeated warm projection timings, not a
full-block, mixed-weight-cache, cold-load or video throughput result. Do not
infer an occupancy, bandwidth or decoder-only bottleneck from this comparison:
the launch geometry changes register and shared-memory pressure as well.

Both candidates equal M32 G1 and resident MPS bitwise for all 2,621,440 raw
outputs and again after bias, under unchanged finite F32 atol=rtol=2e-5.
Postterminal row-band checks separately verify the 23 authentic rows and every
cyclic row, including B128's two-band boundary. All 16 bank hashes repeat
between orders; the 12 prior M1280 control hashes remain unchanged. The
candidate summary booleans require bitwise identity in addition to the
per-comparison tolerance gate; a false strict summary need not imply a
tolerance failure. G2 still fails 949 elements against MPS. The separate
original-BF16 saved-output gate still fails (G1 RMSE 0.617325, relative L2
0.0470342); matching identical Q4 operands does not qualify quantization
quality or a new real full-prefix distribution.

Each run has 81 command roots/80 asynchronous continuations, zero timed-root overlap,
one final wait/readback, zero preterminal reads and zero tracked encode-time
MetalBuffer allocations. Named storage is 616,398,176 bytes including every
control/output bank. The packed projection consumes its 9 MiB weights directly;
the diagnostic also retains a 64 MiB resident F32 MPS matrix and a 32 MiB F16
control. It does not measure minimal production storage. The largest sampled
process footprint is 955.18 MiB, minimum sampled cache-aware headroom is
28.5765 GiB, and observed swap growth is zero. The 10% reserve, 2 GiB native
cap, 64 MiB swap stop and 120-second deadline remain enforced. Sampling is
not a peak-memory proof; MetalBuffer counts exclude private MPS allocations.

Both postterminal replays recompute 18 same-operand and three saved-BF16
comparisons from captured bytes, validate 172 raw comparator words including
injected-bit controls, and check row bands, baseline hashes, 81 root timestamps,
paired schedules and native guard logs. Thirteen CPU checker tests pass,
including a row-64 mutation with recomputed sidecar/bank hashes that still
cannot forge candidate success. Nine captured-report/timing falsifiers reject
widened tolerance, forged G2 success, wrong geometry/segments/root/launch,
overlap, candidate hash and metric words. Six admission negatives reject before
Metal initialization and preserve existing artifacts. Correlated Mini-Quad
source and evidence review is ROBUST only within this bounded diagnostic.

Evidence: .landmark/evidence/2026-10-26-hy-nomas-q4-tile-reuse.json. External
artifacts live under /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026:
reuse-m1280-forward-v1/reuse-m1280-reverse-v1 reports and sidecars,
verify-forward-v1/verify-reverse-v1, paired-analysis-v1, checker-pressure-v1,
admission-v1 and terminal-identities-v1. The selected 23-file source snapshot,
frozen binary and ten linked objects were rechecked after execution. The final
checker predates both replays. This is not a hermetic build/driver attestation.

Executed DoD entry points (the runner supplies the unchanged guards):

```sh
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/tmp/run_guarded.py build-v1 4096 600 make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1 hy_nomas_q4_tile_reuse_benchmark
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/tmp/run_guarded.py reuse-m1280-forward-v1 2048 120 /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/build/hy_nomas_q4_tile_reuse_benchmark-v1 --report /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/reports/reuse-m1280-forward-v1.json --rows 1280 --repeats 16
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/tmp/run_guarded.py verify-forward-v1 1024 120 python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/source/verify_tile_reuse_v1.py /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/reports/reuse-m1280-forward-v1.json
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-reuse-20261026/source/test_verify_tile_reuse_v1.py -v
crystal tool format --check examples/hy_nomas_q4_tile_reuse_benchmark.cr examples/support/hy_nomas_q4_tile_reuse_benchmark.cr
git diff --check
```

Next falsifier: compare Q4 loading with a dense F16 control at the same B64/B128
geometry, preserving the per-output K order, and profile MMA/staging/register
pressure. Account for changed weight traffic before attributing a time
difference to decoding. Increasing M reuse alone does not close the MPS gap.
Real full-prefix inputs, original-BF16 quality, other projections and full-block
performance remain open before production promotion. Refresh on changed
source/cache/fixture, geometry, repetition/order schedule, precision, ownership,
compiler/SDK/OS/device/MPS or missing external artifacts. No full generation.


## Same-geometry Q4/F16 tile frontier, 2026-10-26

Status: VERIFIED within the bounded M1280 projection comparison. An opt-in
benchmark derives dense F16 B64/B128 kernels from the existing Qwen Q4 kernels
by changing the kernel name and replacing only the two Q4 decoder calls with
canonical F16 loads. Checked source anchors preserve the shared panels,
barriers, full-K MMA order, stores, one-group F32 merge and bias. No production
kernel or dispatch is changed. The canonical 32 MiB F16 matrix was already
retained by the preceding probe's correctness controls.

Risk: CAUTION (derived shader source and diagnostic scratch lifetime). Rollback:
revert only this opt-in target and example. The strongest failure is a dense
control that silently changes accumulation or staging, followed by a decoder
attribution based on incompatible work. Require matching kernel geometry,
all-output bit identity, baseline hashes and unchanged finite F32
`atol=rtol=2e-5`; report the traffic/register confound explicitly.

DoD: guarded release build, opposite-initial-order M1280 runs, postterminal
byte/row-band/comparator/root/guard replay and seeded checker failures, then a
separate bounded GPU capture with target PID/pipeline coverage. The native
budget remains a 10% file-cache-aware available-memory floor, 2 GiB process-group
cap, 64 MiB swap-growth stop and 120 s deadline. No full block/model/generation.
All neural math and repetition stay on GPU; there is one final wait/readback,
no preterminal reads and no tracked encode-time MetalBuffer allocations.

Both profiler-off runs use M1280/K8192/N2048, two warmup and six measured rounds
per arm, alternating direction and opposite first order. Each isolated timed
root contains 16 complete projections. M1280 cyclically repeats the same 23
saved real input rows on GPU, not a new full-prefix activation distribution.
Ranges are the two run medians, not confidence intervals.

| Complete projection | Median GPU ms/call | M tile | Threads | Shared bytes |
|---|---:|---:|---:|---:|
| Packed G1 control | 5.043-5.048 | 32 | 128 | 12,288 |
| Packed B64 G1 | 4.683-4.693 | 64 | 256 | 16,384 |
| Dense F16 B64 G1 | 4.277-4.279 | 64 | 256 | 16,384 |
| Packed B128/SG8 G1 | 4.699-4.701 | 128 | 256 | 24,576 |
| Dense F16 B128/SG8 G1 | 4.469-4.469 | 128 | 256 | 24,576 |
| Resident F32 MPS | 3.999-4.005 | internal | internal | internal |
| Q4 expansion + MPS | 4.190-4.193 | internal | internal | internal |

All 12 measured paired rounds favor dense over packed at both geometries.
Median paired packed/dense ratios are 1.095997 at B64 and 1.051755 at B128.
Dense B64 takes 8.65-8.81% less time than packed B64; dense B128 remains
4.44-4.47% slower than dense B64. Dense B64 still takes 6.84-6.95% longer than
resident MPS. Removing decode in this control does not remove the native GEMM
gap. The representation substitution also changes weight traffic (9 MiB Q4
versus 32 MiB F16), cache behavior, instructions and compiler register allocation;
it does not measure isolated decoder cost or establish a bandwidth bottleneck.
One-time expansion/setup is excluded, and no resident-F16 production policy is
qualified by this comparison.

All four B64/B128 packed/dense candidate gates require both finite 2e-5
agreement and zero raw bit differences; all pass before and after bias for
2,621,440 outputs per bank. All 20 bank hashes repeat between orders and all
16 prior M1280 hashes remain fixed. Separate row-band checks cover authentic
and cyclic rows, including B128's two-band boundary. G2 still fails 949 entries
against identical-operand MPS. The separate original-BF16 quality gate still
fails: G1 RMSE 0.6173250074, relative L2 0.047034245, max abs 3.2914271355,
2,621,164 failures. Neither failure is waived or hidden by bitwise Q4 controls.

Each full-schedule run has 113 command roots and 112 asynchronous continuations,
with no timed-root overlap. Named storage is 700,284,768 bytes, including all
control representations, twenty output banks and terminal packing. The largest
sampled native process-group footprint is 1,315.77 MiB; minimum sampled
cache-aware headroom is 26.5709 GiB; observed swap growth is zero. Samples do
not bound true peaks, and MetalBuffer counters exclude private MPS allocations.
The diagnostic does not measure minimal production storage.

The separate CPU replay recomputes 26 same-operand and three saved-BF16
comparisons from bytes, validates 236 raw comparator words with injected-bit
controls, bank hashes, all row bands, root isolation, launch geometry, schedules
and guard logs. Twenty checker tests pass, including a row-64 mutation with
recomputed bank/sidecar hashes, widened tolerance, false candidate/G2 summaries,
wrong source/geometry/root/launch, overlap, metrics and profiler intent. Six
admission negatives pass against the current build before Metal initialization;
three additional cases reject an invalid attach-gate scope or pre-existing
ready/release artifacts without modifying the artifacts.

The current companion adds an opt-in CPU ready/release gate after resource
setup and before command-chain creation, for attaching Instruments to a live
process. It requires `--profile-run`, checks a 30 s deadline before accepting
release, rejects stale gate files and adds no GPU work, barrier or wait. A fresh
profiler-off smoke of this current build passes the same 56-sample/113-root
checks at one call per sample. Its timings do not enter the primary aggregate.
Primary timings bind to frozen binary v1 / native-v2 sources; the CPU-gate
companion binds to binary v3 / native-v4. Their shader identities are identical.

A separate Instruments attach capture is qualified for target PID 91018 and
named pipeline-family presence. It contains 29 target command-buffer submissions,
58 target application intervals, 35 target GPU intervals and 667 target shader
sample rows. Submission/application/GPU tables resolve the same 29 command-buffer
IDs. Packed B64/B128, dense B64/B128, MPS projection/bias and expansion pipeline
labels are present. Shader sample rows are not dispatch counts or additive times;
their schema has no command-buffer/submission ID. The selected exports expose
no launch dimensions or counters. Source-derived geometry is checked by the
native report/replay, not observed directly in this trace. The capture cannot
split resident/expand MPS projection labels or distinguish Q4 decoding, loads,
staging and MMA inside a fused GEMM. No occupancy, instruction-stall percentage
or decoder-only time is claimed.

The successful profile uses an external checked clone with zero warmup and two
opposite-order rounds at one call per sample: 14 samples, 29 roots and 68 stats
words, with all 20 qualification banks and numerical checks retained. Its bank
and shader hashes equal the primary runs. It records the same kernel geometries
and workload shape, but its timings never enter the profiler-off aggregate.
The sampled target-plus-recorder group peak is 2,036.34 MiB, headroom at least
23.4880 GiB and swap flat, within the unchanged guard. Earlier launch captures
failed target coverage; the first attach was stopped too early during saving,
and the corrected full-schedule attach hit the 2 GiB cap at 2,068 MiB. Reducing
only capture repetitions enabled the successful trace. All owned processes
were reaped and stopped raw traces removed; only sanitized selected exports
remain. Correlated Mini-Quad review is ROBUST within the measured comparison
and target/pipeline coverage, not an independent-model assessment.

Evidence: `.landmark/evidence/2026-10-26-hy-nomas-q4-tile-dense.json`. Reports,
sidecars, exact guarded commands and selected source/binary/object receipts live
under `/opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026`. This is a curated
index to executed evidence, not a hermetic build or driver attestation.

Executed DoD entry points (the recorded runner supplies the unchanged guard):

```sh
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/tmp/run_guarded.py build-v4 4096 600 make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025/build/native-v1 hy_nomas_q4_tile_dense_benchmark
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/tmp/run_guarded.py dense-m1280-forward-v1 2048 120 /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/build/hy_nomas_q4_tile_dense_benchmark-v1 --report /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/reports/dense-m1280-forward-v1.json --rows 1280 --repeats 16
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/tmp/run_guarded.py verify-forward-v1 1024 120 python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/source/verify_tile_dense_v1.py /opt/mldata_ssd/cogni-ml/hy-nomas-q4-tile-dense-20261026/reports/dense-m1280-forward-v1.json
crystal tool format --check examples/hy_nomas_q4_tile_dense_benchmark.cr examples/support/hy_nomas_q4_tile_dense_benchmark.cr
git diff --check
```

Next falsifier: optimize B64's shared-panel staging/MMA schedule while retaining
full-K per-output order, starting with the dense control before applying the
same change to Q4. Require identical bits and a balanced complete-projection
speedup against the unchanged MPS controls. Do not widen precision or make the
larger B128 tile the default from nominal reuse counts. Original-BF16 quality,
new real full-prefix inputs, other projections and full-block performance remain
open. Refresh on changed source/cache/fixture, geometry/schedule, precision,
resource ownership, compiler/SDK/OS/device/MPS, or unavailable external evidence.
