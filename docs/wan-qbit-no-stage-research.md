# Wan QBit GEMM without a global decoded weight stage

Research baseline: `7c178df1`; bounded implementation at `31f4d6ce` plus the
source-pinned probe below, 2026-10-22, Apple M2 Max, macOS 26.6.2, SDK 27.0.
TensorOps F32 and direct QBit now compile and pass the declared numerical
checks. They are slower than MPS on this projection. No full model, block,
Studio or generation was launched. The threadgroup-panel alternative remains
PROPOSED.

## Decision and falsifier

Prioritize a bounded Metal Performance Primitives (MPP) TensorOps experiment:
decode QBit values and positional corrections into a SIMD-group cooperative
F32 input tensor, then consume it directly with `matmul2d`. Keep a conventional
Steel-style threadgroup-panel loader as a second candidate. Both aim to remove
the global dense-weight stage while retaining local decoding. The TensorOps
experiment below establishes same-format parity and reduced candidate weight
storage, but rejects a latency improvement for its measured schedules.

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
implementations. The local 26.6.2 host passes the bounded runtime compile and
execution checks below. This does not give M2 the M5's Neural Accelerators.
[Apple TensorOps availability](https://developer.apple.com/videos/play/tech-talks/111432/),
[Metal feature tables](https://developer.apple.com/metal/capabilities/)

Apple also discusses panel accumulation, cache-local traversal and the
register-pressure cost of larger tiles in its [MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
Its M5-specific tile recommendations are not measured M2 settings.

The installed MacOSX SDK's `MPPTensorOpsMatMul2d.h` advertises
`float x float -> float`, cooperative right-input construction and compatibility
checks. Its descriptor defaults to `relaxed_precision=false`; keep that mode.
The implementation limits this cooperative-input path to `execution_simdgroup`
and compile-time K. A threadgroup may schedule independent SIMD-group tiles, but
one cooperative input cannot simply be shared as an arbitrary multi-SIMDgroup
matmul input. Use the right-input factory and `run` type checks for a fresh
cooperative input. `is_compatible_as_right_input` only tests a destination
cooperative tensor for conversion and cannot test the fresh input itself.

F32 types and non-relaxed API mode are not proof of bitwise agreement with MPS
or of a particular instruction lowering. Register pressure, spills, a smaller
reuse domain and panel-accumulation order can erase the benefit. Source-level
private storage also does not prove that the compiler never spills to memory.
Compilation, strict output comparison and profiling remain necessary.

Tested TensorOps dataflow:

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
   `relaxed_precision=false` and the factory's compatible input layout. Check
   decoded values, corrections and full output against the selected decoder
   oracle, including cancellation, tails and invalid-input controls.
2. Establish a fast dense F32 control using the same candidate geometry before
   attributing any packed-route performance to decoding. A dense control far
   behind MPS rejects competitiveness for the dense schedule. Compressed loads
   still require their own bounded test because they change the byte traffic;
   neither result alone proves a DRAM bandwidth bottleneck.
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
measurements. The bounded GPU follow-up is recorded below.

## Active TensorOps implementation slice (2026-10-22)

CAUTION, probe-only: compile an explicit non-relaxed F32 cooperative-right-input
32x32x32 tile, then compare its complete dense projection with the existing MPS
consumer. The experiment is admitted; deployment/default changes and full
generation remain rejected. Removal of the new probe restores the prior state.
DoD: a guarded Metal/Crystal release build, finite full-output comparisons with
the unchanged tolerance against MPS and the pinned upstream fixture, comparator
+1/NaN sensitivity, tails/cancellation controls, and counterbalanced paired dense
timings. Keep one existing terminal completion per route and zero tracked
encode-time buffer allocations. The strongest failure is inadequate matrix
throughput on M2. Require dense numerical qualification before the QBit loader;
measure compressed loading separately even if the dense schedule is slow.
Report an unsupported SDK/runtime combination or a measured-red route as such;
neither licenses relaxed arithmetic or a wider tolerance. No candidate is
promoted from a storage ratio or a successful compile alone.

## Bounded TensorOps results and decision

Implementation: `examples/wan_tensorops_probe.cr` and
`examples/support/wan_tensorops_{kernel.cr,probe.mm,probe.metal,qbit.metal}`.
These are opt-in diagnostics; model defaults and existing consumers are
unchanged. The runtime bridge explicitly requests Metal 4.0 and safe math.
The offline `metal` toolchain was unavailable, so the installed runtime compiler
was used without installing dependencies. Unsupported runtimes fail closed.
Cooperative right input and F32 accumulation use non-relaxed mode, one 32-lane
SIMD group and ascending K32 panels; bias is added separately.

The small F32 checks cover 32x64x32, 35x73x37 and 17x33x9 M/K/N shapes,
near-cancellation, odd K, M/N tails and an asymmetric one-hot orientation case.
All four comparisons with a separate GPU sequential-FMA oracle and direct MPS
are bitexact. NaN-poisoned outputs and the +1/NaN comparator control prevent
unwritten or insensitive comparisons from passing.

Full projection: M4480/K5120/N13824. Each route has four measured samples in
counterbalanced ABBA/BAAB orders after both warmup orders. Values below are
descriptive GPU medians, not confidence intervals or video timings.

| Candidate M/N/K tile | Candidate GPU ms | Same-run preexpanded MPS ms | Receipt |
|---|---:|---:|---|
| Dense F32 32/32/32 | 211.114 | 58.043 | dense-v3 |
| Dense F32 64/32/32 | 191.765 | 55.260 | dense-v4-t64x32 |
| Dense F32 32/64/32 | 267.633 | 55.747 | dense-v4-t32x64 |
| Dense F32 64/64/32 | 286.650 | 57.860 | dense-v4-t64x64 |
| Dense F32 64/32/32, final source | 195.330 | 56.985 | dense-v7-t64x32 |
| Direct QBit 64/32/32, barrier baseline | 773.847 | 56.109 | qbit-v5-t64x32 |
| Direct QBit 64/32/32, SIMD error vote | 757.682 | 57.818 | qbit-v7-t64x32 |

Final dense/QBit encode medians are 0.029/0.016 ms; wall medians are
196.533/757.911 ms. Preparation, uploads, validation comparisons, logging and
the separate failure-control batch are outside these intervals. MPS timing
consumes the already uploaded F32 control matrix directly, without another
weight-stage copy. Every measured route keeps one terminal completion and no
inter-operator host wait, activation readback or tracked MetalBuffer allocation.
That allocation counter does not measure opaque MPS/driver allocations.

Pre/post checks compare every one of the 61,931,520 outputs against the pinned
upstream **same-format** projection and between routes: zero bit differences,
zero nonfinite values and zero violations of `2e-5 + 2e-5*abs(reference)`.
QBit status is eight zeros before and after timing and is checked after every
timed completion. These inputs are four seeded upstream-captured rows repeated
4480 times on GPU, not representative video activations. This does not close
the approximately 6.95% relative-L2 error against the original BF16 projection.

Five separate tiny GPU controls run in one untimed command: valid raw BF16,
out-of-blob offset, unsupported mode, nonfinite raw coefficient and incoming
capacity failure. The valid case is bitexact with the scalar oracle. All four
failure cases turn 105 initially finite output elements into NaNs. The first
three local decode/format failures set sticky word 7; the incoming failure
preserves word 6 without inventing a local error. The pinned blob is untouched.

Replacing only the per-K32 error reduction's two threadgroup barriers with
`simd_any` preserves those controls and full parity. Exact one-SIMDgroup launch
is essential; record-publication barriers remain. The roughly 2% raw timing
difference is from separate process runs and does not establish an optimization
benefit. These separate runs provide no evidence that the barriers explain the
large gap; they do not isolate the barriers' contribution. No spill or
DRAM-bandwidth claim follows.

The direct QBit route binds **39,994,012 bytes** of packed weights, offsets and
LUTs (38.1413 MiB), plus a 32-byte status buffer. It never binds or produces the
270 MiB F32 weight stage. This is 3.54x smaller than the original 135 MiB BF16
matrix. The full diagnostic deliberately retains a dense matrix for MPS and
two output matrices; its explicit live buffers total 910,670,620 bytes. It does
not demonstrate whole-model peak-memory savings or a complete quantized model.
The archival plan is verified for lineage but is not uploaded by the candidate.

Final dense/QBit guards observe at least 23.30 GiB cache-aware headroom, at most
885.01 MiB owned process-group footprint, memory-pressure free percentage at
least 78%, and zero swap growth. The 15% floor, 4 GiB cap, 64 MiB swap-growth
stop and deadlines are unchanged. These are sampled limits, not a guarantee
against between-sample spikes; file-backed reclaimability remains an estimate.

Decision: ROBUST for the declared fixture parity, negative controls and
noncompetitive latency within this host/geometry. Original-weight quality,
representative activations, whole-model capacity and end-to-end speed remain
OPEN. Keep staged MPS as the existing control. The next useful experiment is
a conventional multi-SIMDgroup tiled GEMM with cooperatively decoded local B
panels, first qualifying its dense F32 throughput. Per-weight bitplane lookup,
correction searches and reuse across M remain hypotheses to discriminate;
do not repeat the current schedule solely with wider tiles. This is a future
bounded probe, not a default change or full-generation authorization.

## TensorOps reproduction and receipt boundary

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-tensorops-20261022/`.
`run-experiment.py` freezes commands, source/object hashes, binary/bridge hashes,
environment versions, guard limits, and stdout/stderr/guard digests. Final
`source-v7/index.json` pins the five owned implementation files and launcher;
older `source-v3`, `source-v4`, `source-v5` and failed `source-v6` snapshots
preserve their historical bytes. This is a partial source archive: other
tracked dependencies are hash-checked against the live worktree separately.
Source snapshots and receipts are evidence indexes, not compiler attestations.

Six existing native objects are pinned to prior build receipt SHA256
`4cf673fd89d04112bc72de3e26e98d85111876f8f9b698042df2341aa2f6bca4`.
The new Metal 4 bridge has a fresh source/object/link receipt. The prior receipt
does not provide source-to-object ancestry for every reused native object;
current native-source hashes alone cannot manufacture that ancestry. The
same pinned objects serve both candidate and MPS controls. An initial harness
compile failure (`build-v6`) is retained explicitly and superseded by successful
v7 build/experiments; it is not an accepted GPU result.

Commands below require the pinned local fixture and object paths. Choose a
fresh version token for a rerun; the launcher rejects overwriting evidence.
The final executed token is `v7`.

```sh
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-tensorops-20261022/run-experiment.py bridge v7
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-tensorops-20261022/run-experiment.py build v7
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-tensorops-20261022/run-experiment.py small v7 --tile 64x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-tensorops-20261022/run-experiment.py dense v7 --tile 64x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-tensorops-20261022/run-experiment.py qbit v7 --tile 64x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-tensorops-20261022/run-experiment.py spec v7
```

Expected: build/run return code zero and unchanged source hashes; four small
F32 cases and five QBit controls pass; full pre/post comparisons have no
violations; focused specs report 5 examples, 0 failures/errors, 1 intentional
pending cost probe. Only small projections are executed by the specs.
Refresh all execution evidence after source, packed layout/LUT, fixture,
geometry, compiler flags, native objects, SDK, OS or device changes. Changes
to activation coverage reopen quality claims even when same-format parity
continues to pass. Preserve the numerical gate and memory guards on refresh.

## Shared-panel GEMM experiment (2026-10-23)

CAUTION, opt-in diagnostics only, based on `4792a15f`. Test a conventional
four-SIMDgroup F32 GEMM with shared local A/B panels against the same uploaded
MPS control, then replace only B loading with the pinned positional QBit arm.
Admit 32/32/32 and 64/32/32 M/N/K tiles initially. Preserve ascending K fragments,
F32 coefficients/accumulation, separate bias, and the unchanged finite
`2e-5 + 2e-5*abs(reference)` gate. No execution defaults, full model, block or
generation changes. Omitting `--backend shared-panel` keeps the prior probe
backend; removing the new adapters/shader and reverting the harness additions
restores the prior state. The staged MPS consumer remains control and rollback.

DoD: guarded release build, small GPU scalar/MPS comparisons covering odd K,
tails, cancellation, asymmetric one-hot inputs and cross-SIMDgroup panel reuse;
five sticky-status QBit failure controls; full pre/post-output gates against
pinned same-format upstream; counterbalanced dense then QBit timings including
all local decode work. Keep one terminal completion per route, zero new
inter-operator host waits/submissions, activation readbacks or tracked encode
MetalBuffer allocations. GPU threadgroup barriers protect shared panel lifetime
and are included in timing. Preserve cache-aware 15% reserve/floor, 4 GiB owned
process-group cap, 64 MiB swap-growth stop, 240-second builds/300-second runs.
The strongest failure is unsafe panel reuse or changed floating-point order;
poisoned/tail and independently varying input-row controls must pass before
timing. Reduced weight bytes alone cannot admit a throughput or quality claim.

### Qualification and measured result

The final v3 adapter launches exactly four 32-lane SIMD groups. All groups
reuse one local A panel and one local B panel per ascending K32 iteration;
F32 `simdgroup_load` / `simdgroup_multiply_accumulate` / `simdgroup_store`
perform the matrix operations. Each group owns a disjoint output quadrant.
The dead A panel becomes C scratch after the final panel-read barrier, then
the threads copy only in-bounds results and add bias separately. The QBit
variant changes B preparation to local positional decoding; it binds no dense
weight matrix and creates no global decoded-weight stage. Both paths remain
explicit probe backends, not production alternatives selected by default.

Two rejected versions are retained. v1 compiled the host binary but failed
runtime shader compilation because the grid/threadgroup Metal attributes had
inconsistent dimensions; no GPU candidate result was produced. v2 compiled
and dispatched but all 1,024 outputs of its first small F32 case violated the
gate while MPS matched the scalar oracle. Replacing manual fragment/output
mapping with documented matrix loads/stores resolved the observed failure in
v3. This change does not isolate which removed manual mapping was wrong.
Neither rejected version supplies an accepted performance result.

Both final tiles pass six small F32 cases against a separate sequential-FMA
GPU oracle and MPS: cancellation, odd K, asymmetric one-hot selection, M/N
tails, and heterogeneous non-dyadic data varying independently across rows
and K. Each QBit run additionally passes heterogeneous raw-BF16 records at
65/2048/35 and a pinned corrected-p4 prefix at 65/5120/35, followed by the five
existing sticky-status controls. The prefix has 175 corrected p4 records and
1,135 BF16 replacements. A CPU header rescan of the complete selected arm
finds 69,120 p4 records, 459,430 replacements, and no p5/raw records. The p5
decoder branch therefore remains unqualified by this experiment.

Full-projection dimensions are M=4480, K=5120, N=13824. Dense here means the
F32 expansion of the selected `positional_tile1024_t050_mode0` arm, not the
original BF16 weights. Both routes use the same coefficients, bias and four
pinned upstream input rows repeated on GPU. All 61,931,520 outputs are bitexact
against same-format upstream and MPS before timing, after timing, and in an
additional poisoned re-execution inside the existing post-check command.
Each small output is also poisoned before execution. The finite
`2e-5 + 2e-5*abs(reference)` gate is unchanged, with a positive comparator
control that must detect both a finite defect and a nonfinite value.
Individual timed invocations are not poisoned or independently checked;
the post-check reads the last timed outputs and then checks the poisoned
re-execution. Original-weight error and trajectory quality remain open.

Measured GPU time on M2 Max / macOS 26.6.2 / SDK 27.0:

| Candidate | M/N/K tile | Candidate mean, ms | Same-run MPS mean, ms | Candidate / MPS |
| --- | --- | ---: | ---: | ---: |
| Dense F32 shared panel | 32/32/32 | 272.113 | 65.390 | 4.16x |
| Dense F32 shared panel | 64/32/32 | 238.040 | 58.058 | 4.10x |
| Direct positional QBit | 32/32/32 | 1598.175 | 58.446 | 27.34x |
| Direct positional QBit | 64/32/32 | 905.045 | 58.841 | 15.38x |

Each row is a separate process with eight warmups and eight measured records
in ABBA/BAAB orders, four measured samples per route. Candidate GPU ranges
are 270.771-273.974, 233.015-247.370, 1535.827-1660.972, and
900.637-909.413 ms respectively. Compare each candidate to its own MPS
control; these runs do not isolate the causal effect of changing tile size
or of the earlier one-SIMDgroup TensorOps schedule. Full-projection timing
excludes input fill, upload, fixture expansion and output comparison, but
includes all candidate record validation and local decode work. Candidate
host encode averages are 0.020-0.026 ms; this gap is in GPU execution, not
explained by host command encoding. No specific barrier, bandwidth, spill,
cache or instruction-lowering bottleneck has been isolated.

The packed route uses 39,994,012 bytes (38.1413 MiB, 4.52045 bits/weight),
including offsets and LUTs, plus a 32-byte status buffer. It avoids the
270 MiB global F32 stage; the original BF16 matrix is 135 MiB. The diagnostic
still retains the dense MPS control and both output matrices: explicit buffers
are 870,676,636 bytes for dense and 910,670,680 bytes for QBit. These numbers
do not establish whole-model peak memory or quality. The earlier approximately
6.95% projection relative-L2 difference against original BF16 remains open.

Final bridge/build, both small runs, both dense runs, both QBit runs and the
focused spec command return zero with unchanged source pins. Specs report
5 examples, 0 failures, 0 errors, 1 intentional pending cost probe. Sampled
cache-aware headroom stays at least 16.879 GiB, owned group footprint at most
928.580 MiB, pressure-free at least 77%, and swap growth zero. The 15% floor,
4 GiB cap, 64 MiB swap-growth stop and deadlines are unchanged. Sampling
cannot exclude between-poll spikes, and reclaimable file cache is an estimate.
No model, complete block, Studio or generation is loaded. There are zero new
inter-operator host waits/submissions, activation readbacks or tracked encode
MetalBuffer allocations. One terminal completion per timed route and the
required local threadgroup barriers remain; opaque MPS allocations are outside
the tracked allocation counter.

### Reproduction and continuation

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/`.
The frozen launcher SHA256 is
`ab6f5bc34dc8a76f0a58592d7db22be6f492102d45a004e0ffa32d09fe7f1914`.
`source-v3/source-hashes.json` and per-command intent/completion receipts pin
the executable, bridge, relevant source and dependencies. `source-v1` and
`source-v2` retain the failed implementations. This is a partial source
archive; other dependency files are hash-checked live. The six reused native
objects have the same prior receipt and ancestry limitation documented above;
the runtime Metal 4 bridge is rebuilt from source. Both candidates and MPS
use the same pinned native objects.

The focused spec command and stdout are receipt-backed, but the two individual
spec files were omitted from the launcher's source-hash manifest. A later
`git diff --quiet HEAD -- spec/qbit_positional_mps_linear_spec.cr spec/video_linear_mps_projection_spec.cr`
returns zero; it does not retroactively pin their launch-time bytes. Keep that
ancestry limit separate from the fully pinned GPU harness and shader evidence.
The external CPU verifier `validate-shared-panel.py` rejects seeded mutations
of numerical gates, route/status checks, failure-control order, resource
boundaries and finite timing. Root reruns accept both final tile receipts:
`summary-v3-root-t32.json` (SHA256
`21a4b323ff80ef0851094a56bd630bcd5bb18c394b08c21728ed99b3389fe069`)
and `summary-v3-root-t64.json` (SHA256
`a5a91d6505841264ebad0e9ebb310420e63080da965b664bb688744ee04edb19`).
The correlated Mini-Quad verdict is ROBUST for the bounded fixture parity,
exercised failures and measured noncompetitive latency. That verdict does not
admit whole-model speed, quality, p5 coverage or a particular GPU bottleneck.

From the owned worktree, with the pinned fixtures/objects available, execute
the following using a fresh version token on a rerun (the accepted token is
`v3`; existing evidence cannot be overwritten):

```sh
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py bridge v3
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py build v3
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py small v3 --tile 32x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py small v3 --tile 64x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py dense v3 --tile 32x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py dense v3 --tile 64x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py qbit v3 --tile 32x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py qbit v3 --tile 64x32
python3 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-shared-panel-20261023/run-experiment.py spec v3
```

Expected: all commands exit zero, no source drift, all small/failure and full
pre/post/poisoned gates pass, four finite measured samples per route in both
orders, guard completion without termination, and the focused spec counts
above. Format and whitespace checks are
`crystal tool format --check examples/wan_tensorops_probe.cr examples/support/wan_shared_panel_kernel.cr`
and `git diff --check`. Refresh execution evidence after source, fixture,
layout/LUT, geometry, compiler/native object, SDK, OS or device changes.

Decision: keep the shared-panel path opt-in; MPS remains the execution control.
Four SIMD groups and reduced global weight bytes alone do not produce a
competitive GEMM in this schedule. Qualification is limited to tested formats,
inputs and geometry; there is no full-model speed or quality claim. Next
separate B-panel preparation from matrix execution on the same schedule before
attributing the deficit to decode, memory access, validation or barriers.

A concrete static discriminator is the dense B loader: at TN=32 its adjacent
lanes vary N at fixed K in the source `[N,K]` matrix. At K=5120 their addresses
are 20,480 bytes apart. This establishes an address stride, not a measured
DRAM bottleneck. Compare a contiguous-K load followed by a local transpose
under the same numerical and timing gates; transpose/bank-conflict overhead
could negate the gain. QBit reconstruction repeats for each M tile (140 at
TM=32, 70 at TM=64); those are reconstruction counts, not DRAM traffic counts.
