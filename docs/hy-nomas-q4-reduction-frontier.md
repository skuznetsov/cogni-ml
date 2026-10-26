# Hy-Nomas direct Q4 reduction diagnostic

Status: VERIFIED for this diagnostic scope, 2026-10-25; the direct Q4 numerical
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
