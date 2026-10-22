# Wan QBit GEMM without a global decoded weight stage

Research snapshot: 2026-10-22, source baseline `7c178df1`, Apple M2 Max,
macOS 26.6.2. No Metal compilation, GPU probe, model, block, Studio or generation
was launched for this audit. The new execution routes below are proposals.

## Decision and falsifier

Prioritize a bounded Metal Performance Primitives (MPP) TensorOps experiment:
decode QBit values and positional corrections into a SIMD-group cooperative
F32 input tensor, then consume it directly with `matmul2d`. Keep a conventional
Steel-style threadgroup-panel loader as a second candidate. Both remove the
global dense-weight stage while retaining local decoding. Neither has measured
speed or numerical parity on this host.

The decisive risk is losing optimized matrix throughput, not the existence of
a decoding buffer. First establish a fast dense F32 control for the candidate
geometry; then replace only its weight loader. Reject a no-stage latency claim
unless the complete packed projection beats the existing staged MPS route in
paired timings after the unchanged numerical gate passes.

Rollback is the existing opt-in packed -> reusable F32 stage -> MPS path.
Transformer defaults and deployed weight loading are unchanged.

## Existing evidence and the memory objective

The selected projection is M=4480, K=5120, N=13824: 634.18 GFLOP, 70,778,880
weights and 61,931,520 outputs. The packed positional matrix occupies
39,994,012 bytes including offsets and LUTs (38.1413 MiB, 4.52045 bits/weight),
versus 135 MiB BF16. Its current MPS consumer also needs one reusable 270 MiB
F32 stage. These are buffer sizes, not measured device traffic.

Historical paired medians are 60.629 ms for packed decode/MPS/bias and
59.172 ms for F32 expansion/MPS/bias. Pair direction flips with order; these
measurements establish neither a speedup nor latency equivalence. They use
four unique seeded activation rows repeated on GPU, not a video trajectory.
See [the existing frontier](wan-performance-frontier.md) for receipts.

Packed full-set residency can avoid paging or repeated weight transfers. That
capacity benefit is independent of a pressure-free projection's latency and
has not been measured here. Removing the stage saves 270 MiB of workspace;
it does not make the matrix MACs, activation traffic or output traffic disappear.

The previous 16x8 fused kernel took about 3.519 s; its predecoded dense F32
control took 0.905788 s versus 0.057411 s direct MPS. This rejects the claim
that decoder-only work suffices for that unchanged schedule. It does not reject
all fused geometries.

## Candidate A: F32 cooperative input to TensorOps

Apple documents custom dequantization directly into a cooperative tensor,
whose elements are distributed across participating threads' private memory,
and its consumption by `matmul2d` without a threadgroup store/reload.
This is the relevant seam for our Gaussian codebook, rather than the native
integer-int4 input path. [Apple custom TensorOps operations](https://developer.apple.com/videos/play/wwdc2026/330/)

Apple places cooperative-tensor matmul inputs among its macOS 26 updates.
The API is portable across M1-M5; older GPUs use optimized shader
implementations. The local 26.6.2 host is a candidate target; the exact
SDK/runtime/type combination still needs the bounded compile and execution
check below. This does not give M2 the M5's Neural Accelerators.
[Apple TensorOps availability](https://developer.apple.com/videos/play/tech-talks/111432/),
[Metal feature tables](https://developer.apple.com/metal/capabilities/)

Apple also discusses panel accumulation, cache-local traversal and the
register-pressure cost of larger tiles in its [MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
Its M5-specific tile recommendations are not measured M2 settings.

The installed MacOSX SDK's `MPPTensorOpsMatMul2d.h` advertises
`float x float -> float`, cooperative right-input construction and compatibility
checks. Its descriptor defaults to `relaxed_precision=false`; keep that mode.
The implementation limits this cooperative-input path to `execution_simdgroup`
and compile-time K. A loop over fixed K panels is a candidate, not a compiled
implementation. A threadgroup may schedule independent SIMD-group tiles, but
one cooperative input cannot simply be shared as an arbitrary multi-SIMDgroup
matmul input. Check `is_compatible_as_right_input` and the generated layout.

F32 types and non-relaxed API mode are not proof of bitwise agreement with MPS
or of a particular instruction lowering. Register pressure, spills, a smaller
reuse domain and panel-accumulation order can erase the benefit. Source-level
private storage also does not prove that the compiler never spills to memory.
Compilation, strict output comparison and profiling remain necessary.

Proposed dataflow:

```text
packed codes + exact local F32 LUT + positional corrections
  -> cooperative F32 weight panel -> F32 matmul accumulator -> output/bias
```

There is no persistent dense-weight buffer in that design. No host round trip
is required; append work on the caller's existing encoder and retain resources
through its existing completion boundary.

## Candidate B: a cooperative threadgroup panel loader

MLX's classic quantized Steel GEMM cooperatively loads/dequantizes B into a
threadgroup panel, then feeds its matrix multiply. Its default 32x32x32 tile
and F32 accumulation are a useful reference, not a drop-in Gaussian decoder.
[MLX classic quantized GEMM, pinned source](https://github.com/ml-explore/mlx/blob/c6abf0dbb4fc141fad739b32ae486c06728e49e5/mlx/backend/metal/kernels/quantized.h#L674-L778)

That geometry divides the selected shape: 60,480 threadgroups rather than
483,840 for the existing 16x8 kernel. A B value serves 32 rows instead of 16;
an A value serves 32 outputs instead of 8. K32 panels reduce the number of
panel synchronization phases compared with the current K8 loop. These are
source-level work counts, not predicted speedups. Preserve ascending K8
fragments initially; still compare the changed reduction with upstream.

The important transferable FLUTE ideas are an offline compressed execution
layout and vectorized/replicated LUT access. Its CUDA, small-batch results are
not a speed prediction for M=4480 on M2. Do not import reduced-precision global
partials into this F32 route. [FLUTE paper](https://arxiv.org/html/2407.10960v2)

For either candidate:

- Separate the archival record layout from a compact execution-panel layout
  only if measurements justify repacking; count all directories and sidecars.
- Locate a panel's sorted corrections once and apply them to decoded values,
  rather than binary-searching every weight. Benchmark merge versus scatter;
  neither is assumed faster for every correction density.
- Amortize validation with an immutable prepared-weight handle, exact buffer
  identities, generation/lifetime ownership and sticky fail-closed GPU status.
  Raw mutable buffers cannot inherit an earlier validation certificate.
- Preserve BF16 replacement semantics and F16-delta rounding before the MMA.
  A separate sparse correction dot product changes floating-point association.

## Can the Gaussian lookup be replaced by arithmetic on bits?

For a group, preserve the decoder's scalar operation order:

```text
D[c] = round_f32(mean + round_f32(sigma * L[c]))
weight = BF16 replacement, or round_f32(D[c] + F16 delta), where present
```

Moving mean/scale outside the dot product or decomposing D into binary basis
functions is an identity over real arithmetic, not necessarily the same F32
projection. Exact reconstructed coefficients and unchanged GEMM output accuracy
are separate gates.

A tiny CPU-only exact-rational falsifier read both 1024-byte LUTs and eight
8-byte group headers per precision; no full weight tensor was read. The raw
p4 table needs 8 nonzero Walsh terms in its original code labels (4 linear,
4 cubic); p5 needs 16 (5 linear, 10 cubic, 1 degree-five). An exact constant
plus the original 4/5 linear bit terms is therefore insufficient for these LUTs.
The best least-squares original-bit affine fit has RMS centroid errors
0.170698/0.234459 for p4/p5; this nonlinearity is not solely rounding noise.

After the decoder's separate F32 multiplication and addition, the eight
sampled group tables have respectively 9/16 and 17/32 nonzero terms. Five p4
samples and six p5 samples have full support. These deliberately selected
samples do not estimate population prevalence. Some added terms may be tiny;
their exact nonzero count is not a lower bound on a tolerance-passing method's
cost. Different code labels, bases or approximate factorizations remain open.

An alternative without per-weight reconstruction is to sum activations by
code, then multiply each bucket by D[c]. It requires output-dependent grouping,
changes reduction order and can lose efficient matrix hardware execution.
LUT-GEMM's binary-coding factorization and T-MAC's bit-plane lookup show useful
approaches, but neither makes this fixed Gaussian table an exact p-term affine
map. Keep this as a secondary experiment, not a presumed faster drop-in.
[LUT-GEMM](https://arxiv.org/html/2206.09557v4),
[T-MAC](https://arxiv.org/html/2407.00088v2)

The coefficient falsifier is ROBUST only for the scalar representations and
sampled headers above; it is VULNERABLE as a claim about every factorization,
projection accuracy, throughput or model quality. Signed-zero identity and
overflow-edge binary32 inputs are outside this bounded rational probe.

## Next bounded experiment and admission contract

1. Compile one static-K SIMD-group TensorOps tile with F32 inputs/accumulator,
   `relaxed_precision=false` and an asserted compatible input layout. Check
   decoded values, corrections and full output against the selected decoder
   oracle, including cancellation, tails and invalid-input controls.
2. Establish a fast dense F32 control using the same candidate geometry before
   attributing any packed-route performance to decoding. A dense control far
   behind MPS invalidates the simple fast-loader hypothesis for that schedule.
3. Replace only B loading with the real packed records. Require the finite
   `2e-5 + 2e-5*abs(reference)` full-output gate, including +1/NaN comparator
   sensitivity, before paired full-projection timings. Keep both warmup orders.
4. Report complete encoded/GPU/wall time, workspace and complete resident bytes,
   together with representative activation coverage. A toy tile timing is not
   a full-projection or video claim. Profile spills and reuse only after a
   competitive candidate exists.

Keep one terminal completion per route, no new in-chain host waits,
submissions, activation readbacks or tracked encode-buffer allocations. Retain
the cache-aware 15% reserve/floor, 4 GiB owned-group cap, 64 MiB swap-growth
stop and bounded build/run deadlines. No full model, block or generation is
authorized by this research result.

Same-format parity does not close the existing approximately 6.95% relative-L2
projection error against original BF16, or establish video quality. Preserve
that separate quality frontier; do not widen the numerical gate.

## Evidence and refresh boundary

External evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-no-stage-research-20261022/`.

- `exact_codebook_basis.py`: SHA256
  `779626b255eb9566a3dcffe7e5f1c965fc382a5a912849c3ac471903dd2acd43`.
- `exact_codebook_basis-root.json`: root CPU rerun, SHA256
  `6e8c34617c9a2ff31137ae01629a4482d63d4aecf30e9e153cd3ba4ee87f1ebb`.
  Five rounding controls, positive affine and seeded nonlinear controls, and
  exact inverse reconstruction of every inspected table entry pass.
- `best-affine-metrics-root.json`: orthogonal degree-0/1 fit of the complete
  raw centroid tables; no projection or timing claim.
- Input manifest SHA256:
  `0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703`.
  LUT hashes are reverified, packed file lengths checked, headers recorded.
  Full packed-file hashes were not reverified for this tiny read.

CPU reproduction (external scratch only):

```sh
cd /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-no-stage-research-20261022
python3 exact_codebook_basis.py > exact_codebook_basis-rerun.json
shasum -a 256 exact_codebook_basis-rerun.json
```

Expected JSON hash is the root-rerun hash above at the stated fixture paths.
This evidence decays on LUT, group header, scalar decoding semantics or script
changes. Re-run the scalar probe then. Refresh API assumptions on SDK/OS/device
changes; compile and measure on the target host before any execution claim.
The three read-only Quad reviews are correlated analysis, not independent GPU
measurements. The next permitted signal is the bounded tile experiment above.
