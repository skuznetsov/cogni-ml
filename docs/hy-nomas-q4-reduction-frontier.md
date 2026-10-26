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
