# Original DCAE-V native decoder frontier

The native opt-in decoder has verified full-context BF16 parity.
Conv3D profiling is implemented and performance gains remain unproven.
The tiled G8 diagnostic matches the full stage-2 pre-bias Conv3D output
bitwise for both real cache cases. Paired BF16 timing shows no consistent
advantage over MPS. Earlier reduction schedules failed the full-weight
upstream gate.

## Goal and boundary

Port the pinned Hy-Nomas f32t4c32 original DCAE-V decoder to Crystal and
Metal/MPS, then compare native decoder time and sampled process memory with
PixelUMM on the same authenticated final T0/T1 latent bank. The existing
upstream DCAE control establishes a reference, not native implementation.
This is CAUTION work: causal feature-cache state, BF16 operation boundaries,
and resource lifetimes affect visible output.

Admitted control: upstream module at revision
`911cea6458a89c2663513ff7099561ec190dd491`, module history
`a31c0fe0bcc493be188fd37ddba4ce2f674becb4`, original BF16 checkpoint SHA-256
`ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b`.
The prior final decoder comparison is indexed by
`.landmark/evidence/2026-11-08-hy-nomas-final-dcae-pixelumm.json`.

Admitted path: native decoder with original 171 BF16 tensors, prepared
reusable buffers and MPS plans, and one terminal drain per request. Preserve
the global ten-latent tile grid and original feature caches across tiles.
Reject unsupported geometries, incomplete weights, wrong dtypes, aliasing,
and premature release. A workspace is serial: callers must not bind it to
simultaneously executing plans. No default decoder switch, model training, DiT rerun,
quantization, CPU arithmetic fallback, or wider numerical tolerances.

## Invariants and falsifiers

- Match raw BF16 outputs and intermediate stage captures to pinned upstream
  at `atol=rtol=2e-5`; report unequal values independently. Compare original
  DCAE with itself for parity; DCAE versus PixelUMM remains descriptive.
- First discriminate a single tile and two consecutive tiles at small spatial
  geometry. A second-tile mismatch can expose missing cache transport even
  when the first tile passes. Add a short-final-tile control if admitted.
- Qualify comparisons with seeded wrong outputs and reject mismatched input,
  weights, temporal context, capture shapes, and output extents.
- All model arithmetic stays on GPU. Weight/input uploads precede execution;
  intermediates remain resident until the sole terminal export. Instrument
  named buffer allocations, readbacks, and preterminal waits; require zero.
- Use the existing cache-aware external guard with reserve 10% and a 64 MiB
  swap-growth stop. Admit named residency before model allocation, separately
  report opaque MPS workspace, and run only one model-owning process at a time.
- Timing comparisons use the same final `[1,32,60,9,16]` bank and complete
  temporal context, with setup/upload, GPU execution, terminal/export, and
  sampled memory separated. Do not infer a full-generation rate from decode.

## Implementation plan and observed signals

1. Extend decoder-only checkpoint conversion and negative CPU tests, retaining
   existing encoder conversion behavior. Files: a dedicated converter under
   `examples/support/`, corresponding Python tests under `spec/`.
2. Add decoder geometry, exact binding inventory, liveness plan, native shuffle
   operations, and prepared convolution/normalization plans under
   `src/ml/video/`; reuse existing primitives when their semantics match.
   CPU specs must establish exact tensor counts, shapes, budgets, and rejects.
3. Add a bounded native probe and upstream stage oracle. Run small single/two
   tile numerical controls before the full paired bank. On failure, isolate
   the first divergence without weakening tolerances or adding production
   waits/readbacks.
4. After parity, measure both decoders serially on the same fixture. Record
   one first decode request after prepared setup and state its upload boundary;
   report setup, request time, and sampled process memory independently.
   This is descriptive, not a matched warm or process-cold benchmark.
   Curate evidence and
   update this frontier before an atomic implementation commit.

DoD: focused Python and Crystal specs pass; a release native probe builds;
native/upstream original-DCAE comparisons pass at the declared tolerance for
small temporal-cache controls and the final full-context fixture; execution
counters show no new intermediate boundary; both native decoder measurements
finish inside the 10% reserve. A blocked guard is safety evidence, not parity
or speed evidence. Strongest failure mode: a layout or cache mismatch hidden
by static input; use varying per-frame input in the small control.

Rollback: omit the new opt-in decoder/probe; preserve PixelUMM and encoder
defaults. Refresh results after source, binary, runtime, weight, geometry,
input, rounding, cache schedule, or measurement-policy changes.

## Observed result, 2026-11-08

The export after two native requests matches original upstream BF16 values
bitwise on all 18 tensors in the varying two-tile fixture. The second request
reuses the same buffers and plans; the first request's output is not exported
separately. On the final `[1,32,60,9,16]` bank, all 106,168,320
output values also match bitwise, with no nonfinite values; unchanged
`atol=rtol=2e-5` therefore passes without consuming tolerance. Full-size stage
captures were deliberately omitted to bound residency.

Both native decoder measurements use that exact bank, numerically unchanged
by BF16 conversion. It contains five seconds of previous context and five
seconds of final generated context: 240 decoded frames at 24 fps, 512x288.

| Native decoder | Decode request after setup | GPU interval sum | Sampled process peak | Named buffers |
| --- | ---: | ---: | ---: | ---: |
| Original DCAE-V | 684.661 s | 684.615 s | 10.607 GiB | 5.002 GiB |
| PixelUMM | 141.552 s | 141.489 s | 18.413 GiB | 17.009 GiB |

These are serial first requests excluding setup and terminal file
serialization. Original BF16 DCAE weights are copied directly into resident
buffers during setup. Its first-request `PreparedWeights#encode` call only
validates lifetimes: zero staging bytes means no GPU upload or conversion.
The earlier `weights_upload=true` log label was inaccurate. PixelUMM also
prepares its block cache outside the request timer.
DCAE prepared setup was 3.513 s; PixelUMM's decoder-specific setup was
7.167 s. Those setup fields also have different admission boundaries.
There was no matched warmup, frequency control, repeated full-size pair,
or process-cold cache control. DCAE used 42.4% less sampled peak memory.
The request-time ratio is 4.84, but remains descriptive because preparation,
warmup and runtime conditions were not matched; it is not a matched speedup result. Neither metric
estimates end-to-end video generation. The original
upstream oracle's 671.056 s includes its terminal pack/readback and is not a
matched native/upstream speedup result.

Each native request records zero instrumented named-buffer allocations,
readbacks and explicit waits before its one terminal drain. All named
workspace and bindings are prepared once. MPS may allocate internal resources
and wait inside its API; these counters do not measure or eliminate that
behavior. Sampled whole-process memory includes opaque MPS resources and CPU
state. The cache-aware guard remained at 10%, all measured model processes
were serial, and observed swap growth was zero.

The fresh PixelUMM output hash is identical to the previously verified whole
bank. Its T1 export also matches all 212,336,640 bytes of the saved upstream
T1 output in a direct comparison. This joins to the prior frozen fixture,
not another upstream PixelUMM execution. Original DCAE has a fresh full-size
upstream execution here. That execution used the archived oracle source before
a post-export `traces.clear()` cleanup; the current oracle produced the same
18-tensor small artifact after the cleanup. Both source hashes are retained.

The exact 171-tensor decoder-only BF16 sidecar is kept separately at
`/opt/mldata_ssd/models/hy-nomas/dcae-v-native-decoder-ebbee483/`.
Its 2,443,960,856 payload bytes are extracted without quantization or numerical
conversion. Model files and run artifacts are excluded from the repository.

Checks: release probe build, 11 focused Crystal specs, 26 Python tests, Ruff,
authenticated chunked output comparisons, and guarded artifact audit passed.
The canonical index is
`.landmark/evidence/2026-11-08-native-dcae-v-decoder.json`; raw artifacts live
under `/opt/mldata_ssd/cogni-ml/dcae-native-decoder-20261108.t9_k76ut/`.

## Entry points and remaining boundary

- `ML::Video::DCAEVDecoder` exposes weight bindings, residency planning and a
  prepared reusable plan. The serial caller owns the arena and terminal drain.
- `make dcae_v_decoder_probe` builds the opt-in native probe.
- `make spec_dcae_v_decoder` checks the CPU
  geometry/binding/budget contracts without loading a model.
- `python3 -B -m unittest spec.dcae_v_decoder_checkpoint_test
  spec.dcae_v_decoder_reference_test spec.dcae_v_decoder_compare_test
  spec.dcae_v_checkpoint_test` runs the focused Python controls.
- `examples/support/dcae_v_decoder_checkpoint.py` authenticates and extracts
  the original decoder weights; `spec/support/dcae_v_decoder_reference.py`
  supplies the original-MPS oracle. The native probe requires authenticated
  weights/reference, a fresh external-SSD output path and an external memory
  guard; the executed command manifests record all pins and guard arguments.

The verified GPU fixtures use batch one and complete ten-latent tiles. Short
final tiles are rejected. A workspace must not be shared across simultaneous
plans, and this port is not yet wired into the end-to-end generation pipeline.
No default decoder has changed. Output parity establishes reproduction of
the original decoder, not identity preservation or a motion-quality repair.

## Conv3D profiling frontier, 2026-11-08

The bounded discriminator is one complete ten-latent tile at real spatial
geometry, `[1,32,10,9,16]` to `[1,3,40,288,512]`. It uses original BF16
weights and an authenticated causal prefix of the frozen full-size upstream
capture. The two-tile small fixture separately exercises changing temporal
input and cache reuse, with all stage captures enabled.

Profiling observes operation host encode scopes and the existing MPS command
root intervals after the existing terminal wait. Shared-root timing is an
attribution bound, not exclusive kernel time. Optional graph API scopes report
wall time and calling-thread CPU separately; nested scopes are not additive.
No profiling scope creates a command root, submit, wait, or host tensor read.
The strict root validator rejects invalid GPU timestamps rather than dropping
or repairing them.

The initial native real-tile baseline took 105.123 s after setup. The existing GPU
roots cover 105.078 s; definite stage-2 and stage-1 residual Conv3D-associated intervals
are 53.340 s and 26.614 s, respectively, at least 76.1% of the total root
interval sum. Their whole stages cover about 84.6%. The result matches all
17,740,800 input/output BF16 values bitwise. The graph API's large host wall
scopes are consistent with blocking/backpressure: 296 `commitAndContinue` calls
total 80.410 s wall but only 5.697 ms of calling-thread CPU. This aggregate does
not resolve temporal overlap with GPU roots and does not establish
which GPU hardware units execute Conv3D or which kernel-level limit dominates.

A later default-path profile took 114.383 s,
with 114.336 s of GPU root intervals and 297 roots. The two dominant residual
Conv3D groups have at least 57.303 + 30.685 = 87.988 s of associated intervals,
or 77.0% of that interval sum. These intervals can include internal GPU waits
or idle time and do not isolate convolution arithmetic. All 17,740,800
input/output BF16 values match the same
frozen upstream prefix bitwise. The corresponding small two-tile default control also
matches all 18 tensors after two serial requests. The 105.123/114.383 s samples
are not an A/A drift estimate: their source states and measurement order differ.

After removing all experimental mode flags, a fresh release build of the
final source took 111.490 s for the same real tile: 85.801 s encode wall and
25.689 s terminal wait, with 111.443 s in 297 GPU root intervals. Definite
stage-2/stage-1 residual Conv3D-associated intervals are 57.079/27.780 s,
76.146% combined. The 296 commit/continue calls consume 85.711 s wall but
6.188 ms of calling-thread CPU. These scopes overlap and must not be summed.
The result matches all 17,740,800 input/output BF16 values bitwise, and the
fresh two-tile default fixture matches all 18 tensors after two requests.
Sampled process peak is 9,988,017,032 bytes (9.302 GiB), minimum cache-aware
headroom is 27,320,614,912 bytes, and observed swap growth is zero. Named
hot allocations/readbacks/preterminal waits remain zero, with one terminal
wait per request. This is a fresh final-source profile, not a matched
performance comparison with the earlier samples.

| Candidate | Strict upstream result | Timing evidence | Decision |
| --- | --- | --- | --- |
| Automatic layout conversion | Both paired exports bitwise | Unprofiled ABBA after equal warmup: A 118.93/137.93 s; B 149.72/144.05 s | No admitted gain; substantial order drift |
| Explicit NDHWC/DHWIO reorders | Small stages and real output bitwise | 111.93 s real tile, unmatched against 105.12 s baseline | No demonstrated gain |
| Temporal windows lowered to Conv2D, stage 2 only | First captured difference at stage 2; 65,870 output tolerance failures | Real-size timing not admitted | Rejected; removed from working API |
| BF16 expanded to F32 Conv3D, stage 2 only | First captured difference at stage 2; 78,896 output tolerance failures | Real-size timing not admitted | Rejected; removed from working API |
| Four BF16 temporal subtiles, stage 2 only | All 18 small traces and real output bitwise | 110.18 s real tile without profiling, unmatched | No demonstrated gain |
| MPSGraph core optimization Level 0, stage 2 only | All 18 small traces and real output bitwise | 105.93 s real tile without profiling, unmatched | No demonstrated gain |
| Conv1 + BF16 SiLU graph, stage 2 only | All 18 small traces after two requests and real T10 input/output bitwise | 109.32 s unpaired; ABBA and BAAB both show large position/reuse drift | Numerical gate passed; no speedup admitted |

The failed arithmetic lowerings pass index/shape checks but fail the unchanged
`atol=rtol=2e-5` oracle gate. Their differing backend operation/reduction has
not been localized. Their sources, binaries, and failed comparisons are
archived outside the repository; do not reintroduce them under a new name
without a discriminator addressing that failure.

The temporal-tiling profiled run completed its request in 107.609 s but then
failed strict profile interpretation because one root had invalid timestamps.
It exported no tensor artifact. Its phase attribution is inadmissible; the
110.18 s unprofiled retry supplies the separate numerical result. No memory
or timestamp guard was weakened to accept it.

The core optimization control keeps the original BF16 graph and NCDHW/OIDHW
operands, setting only `MPSGraphOptimizationLevel0` for stage-2 residual Conv3D
instead of the nil/default Level 1 descriptor. It tests the net optimization
level, not placement alone. Its 105.93 s sample and the later 114.38 s default
profile are unpaired and have different instrumentation; they establish no
speedup. No candidate is promoted. A promising future latency requires equal
warmups and both matched crossover orders before promotion.

The causal-prefix derivation was independently rechecked: each BCTHW channel
plane's first T frames from the authenticated T60 source matches the complete
derived input/output payload byte for byte for T10 and T20. The geometry
validator alone does not establish this lineage; the separate hash audit does.

An initial combined Metal System Trace / GPU Counters recording exceeded the
unchanged 18 GiB process-group cap. Its partial trace was unreadable, and no
timing or numerical claim uses it. Guard termination reported a permission
error for one group member; a subsequent external process check found neither
the native probe nor the recording process still running. A bounded 20-second
Metal System Trace completed inside the guard, and its model output again
matched upstream bitwise. This instrumented execution is diagnostic only.
Shader inventory rows are available, but shader-duration and hardware-counter
tables contain no samples. Inventory frequency cannot establish GPU dominance,
occupancy, bandwidth, or a compute-versus-stall cause. Raw trace bundles include
process metadata and are retired after extracting an allowlisted summary.

The first stage-2 convolution's existing ordered-FMA Metal kernel was compared
with MPS on the same real BF16 weights and synthetic shared input/cache buffers,
with C512/T40 and smaller H3/W5. It fails the unchanged tolerance on 67/69 of
307,200 values for invalid/valid history, respectively (maximum absolute error
0.015625). The formulation is rejected before any performance claim; this is
not a captured model activation or an upstream single-layer execution.

A separate single-chain diagnostic exposed Conv3D output before bias and
post-bias BF16 output for the same tiny fixture, with invalid and valid cache.
Its multi-target post-bias control matches the untouched original graph
bitwise. A fresh separate PyTorch MPS operator oracle, with CPU fallback
disabled and no full checkpoint/model, matches both MPS boundaries bitwise.
The native ordered-FMA formulation already differs before bias: 159/156 bit
mismatches and 82/83 tolerance failures for invalid/valid cache. Its explicit
BF16 bias addition is internally exact, leaving 67/69 post-bias tolerance
failures. The first demonstrated discrepancy is the convolution
pre-bias BF16 output. The exact internal reduction remains unresolved.

An independent upcast-F32 MPS Conv3D lane rounds to the same BF16 pre/post
values on this fixture; its F32 accumulator differs from native by at most
5.579e-5. It is not the hidden accumulator of BF16 MPSGraph. Small reduction
differences cross BF16 midpoint thresholds, so a small F32 difference cannot
justify widening the output tolerance. This diagnostic is synthetic and
small-spatial, not a real-size activation capture, full-model execution or
performance measurement. Its comparator exits zero only for the unchanged
post-bias control; candidate parity is reported separately.

The archived fusion probe combines only stage-2 conv1 and its existing BF16 SiLU
operations in one MPSGraph executable. The original convolution, bias, logical
SiLU operations and cache schedule remain; conv2, RMSNorm and residual stay
separate. The strongest failure mode is changed eager BF16 rounding after
graph fusion. All 18 traces of the changing-input two-tile small fixture after
two requests, and the real T10 input/output, match upstream bitwise at the
unchanged tolerance. The unprofiled 109.32 s T10 observation is not a matched
speedup. In ABBA after equal warmup, the measured control times are
140.383/121.575 s and fusion times 128.791/126.325 s. In the reverse BAAB crossover, B is 127.541/117.792 s and A is
143.951/123.644 s. A slows on its second measured use and recovers on its
third in both orders. This position/reuse effect confounds the variant;
neither crossover admits a repeatable speedup. All four exported results
match upstream bitwise. Do not pool their means as evidence of a gain.
The observed 297 roots count
command-chain roots, not Metal dispatches or internal MPSGraph nodes. An
unchanged count therefore does not establish that fusion had no effect.
The exact experimental implementation and retained binary are archived on the
external SSD for replay. The active API uses the original convolution path.
More layout sweeps without an arithmetic or kernel-level discriminator are not
admitted.

All seven experimental controls have been removed from the active API; their
pinned sources, binaries, inputs and results remain in external evidence.
Only generic profiling remains opt-in; original Conv3D arithmetic is retained.
Keep the 10% cache-aware reserve, 18 GiB process-footprint cap and 64 MiB
swap-growth stop in the external guard. Named hot-path allocation, readback
and preterminal-wait counters must remain zero, with the sole existing terminal
wait per request. Opaque MPS-private resources are covered by process sampling,
not by named-buffer counters. No full video generation, default decoder
change, lower precision, or tolerance widening is admitted.

DoD for the profiling implementation: focused CPU contract tests, a release
probe build, fresh default-path two-tile upstream parity, guarded real-tile
profile, an unprofiled matched crossover with its numerical exports, and static
review of synchronization/resource boundaries. These do not close the separate
performance objective until a repeatable matched speedup is demonstrated.
Refresh after source, binary, runtime, weights, geometry, input, cache schedule,
rounding, or measurement-policy changes. Raw artifacts are stored under
`/opt/mldata_ssd/cogni-ml/dcae-conv-profile-20261108.dwteqxm5/`; the curated
index is `.landmark/evidence/2026-11-08-dcae-conv3d-profile.json`.

## Pre-BF16 Conv3D reduction frontier, 2026-11-08

The next bounded probe keeps the original decoder unchanged and uses the first
stage-2 residual conv1 weights at the real operator geometry
`[1,512,40,72,128]`, with `[512,512,3,3,3]` weights. It compares full-shape
original/multi-target MPS controls before interpreting sampled native reductions.
The initial shared input is deterministic synthetic BF16, explicitly not a
captured model activation. Both invalid and valid causal-history cases are
required. Boundary, stratified and GPU-selected near-midpoint output coordinates
must be identified separately; sampled agreement cannot establish full-output
or full-decoder parity.

The falsifier is any nonfinite value or unchanged-tolerance prebias failure on
an authenticated coordinate. The old ordered-FMA mismatch is the positive
control: an instrument that cannot detect it on the retained small fixture is
not admitted. Native variants distinguish accumulation order, grouping and
rounding from input/index errors. A separately upcast F32 MPS convolution is a
reference lane, never the hidden BF16 accumulator. A separate PyTorch MPS
operator oracle authenticates exposed BF16 boundaries and shares the Apple
backend; it is not backend-independent evidence or part of a native timing.

This is a CAUTION diagnostic change, reversible by removing the new probe
files. No production arithmetic/default changes are admitted. The external
guard retains a 10% cache-aware reserve, at most 18 GiB process-group footprint,
64 MiB swap-growth stop and a bounded timeout. All outputs and graph bindings
are allocated before the chain, with zero named hot allocations/readbacks and
zero preterminal waits, followed by the existing sole terminal drain. Opaque
MPS allocations remain covered by sampled process accounting.

DoD is a release probe build, focused comparator/rounding checks including a
known mismatch, guarded small-fixture control, guarded real-geometry native
probe and separate upstream operator comparison, and a synchronization/index
audit. A passing diagnostic does not admit a speedup, real-activation parity or
full generation. Refresh on source/binary/runtime/weights/input/shape/cache or
sampling-policy changes. New raw artifacts belong under
`/opt/mldata_ssd/cogni-ml/dcae-conv-reduction-20261108.jeXGNXnF/`.

The guarded probe now executes both this full geometry and the retained
`H3/W5` positive control. On the full shape, the original and multi-target
post-bias MPS graphs agree bitwise on all 188,743,680 output values per cache
case. Rounding the separate upcast-F32 MPS convolution to BF16 also agrees
bitwise with the exposed pre-bias BF16 output on every value in both cases.
This establishes an observable boundary equivalence for this synthetic
fixture, not access to the hidden BF16 accumulator. The pinned upstream
operator matches all 8,192 sampled pre/post-bias values per case, including
its original biased Conv3D call versus a separate BF16 bias addition.

The native exact-integer reference passes all 256 independently constructed
Fraction/RN-even vectors for both F32 and BF16, including cancellation,
subnormal, overflow and double-rounding discriminators. Its direct BF16
result nevertheless fails unchanged upstream tolerances on 1,872/8,192
selected coordinates per full-shape case: 6/3,584 stratified coordinates and
1,866/4,096 deliberately midpoint-enriched coordinates. The boundary samples
have zero tolerance failures. All 18 ordered/grouped native F32 reductions
also fail; ordinary ordered FMA has 1,852/1,856 failures. These sampled counts
are not full-tensor prevalence, and exact mathematical summation is not an
upstream reduction schedule. Neither higher accuracy nor these grouping
variants are admitted as decoder replacements.

The compact post-drain double-rounding check changes 49/8,192 exact BF16
values per case when exact accumulation is first rounded to F32. This repairs
only 42/44 of the 1,872 direct-exact tolerance failures; 1,830/1,828 remain.
Thus staged F32 rounding is a real contributor on this sample, but cannot
explain most mismatches. The current data do not identify the MPS schedule.

Both native runs retain one terminal drain and zero instrumented named hot
allocations/readbacks/preterminal waits. The full native process group has
a sampled peak of 3.686 GiB; the separate upstream process group peaks at
3.563 GiB. All four completed native/upstream runs preserve the 10% reserve
and have zero observed swap growth. Process sampling cannot exclude spikes
between samples, driver waits or allocations in external compiler services.
The complete diagnostic GPU interval includes multiple full convolutions,
sampling and 19 native candidates; it is not a Conv3D speed measurement.

The comparator requires an explicitly named candidate for a passing candidate
gate. Observation-only control success is separate, and both ordered-FMA and
exact-integer candidate gates return failure on both fixtures. Nine focused
rounding/gate tests and 44 memory-guard tests pass. The diagnostic is verified
within this fixture; native upstream-equivalent reduction and performance
improvement remain open. The curated evidence index is
`.landmark/evidence/2026-11-08-dcae-conv3d-reduction.json`.

Next discriminator: capture the true stage-2 BF16 input and causal cache by
running only the decoder prefix, then repeat the same declared comparisons.
This tests distribution dependence without a full generation. Do not promote
the weak score advantage of one grouping or repeat layout sweeps as evidence
of an identified reduction schedule.

## Real-activation stage-2 prefix probe, 2026-11-08

Design boundary: execute only the first two complete ten-latent tiles of the
authenticated decoder input through the entry to
`decoder.stages.2.op_list.1.main.conv1.conv`. Preserve the existing prefix
arithmetic and causal cache schedule. The cold tile and the following cached
tile must have their own activation buffers; do not substitute synthetic
history into the first activation and call it a real second tile. The target
input is BF16 `[1,512,40,72,128]`; its old cache is `[1,512,1,72,128]`.

This is a CAUTION, opt-in diagnostic extension. Strongest failure mode:
capturing the cache after it was overwritten, or interpreting prefix error as
Conv3D reduction error. The guard checks are full BF16 input/cache comparison
against an independently executed pinned upstream prefix, followed by a
separate upstream operator call on the identical exported native input/cache.
The latter is a post-drain diagnostic process, not an intermediate transfer in
the connected native path or independent evidence about Apple's backend.

The native prefix and reduction consumers share GPU buffers on one existing
chain. Allocate owners, graph bindings and outputs before execution; retain
them until its sole terminal drain. Require zero named hot allocations,
readbacks and preterminal waits. Retain the synthetic small fixture as the
known-mismatch instrument control, rather than assuming real activations
must reproduce the same mismatch. Report boundary, stratified and midpoint
samples separately at unchanged `atol=rtol=2e-5`.

Before launch, compute named residency and admit a process cap plus the 10%
cache-aware reserve in the existing external guard; retain the 64 MiB
swap-growth stop and bounded timeout. Run native and upstream jobs serially.
No full decoder, DiT, generation, default arithmetic change or speedup claim
is admitted. Rollback removes the new opt-in prefix/diagnostic entry points;
the existing complete-decoder path and synthetic control remain available.

DoD: focused prefix-boundary and comparator falsifiers, release probe build,
guarded real-activation native and upstream prefix runs, full input/cache
authentication, unchanged-tolerance sampled reduction comparison on identical
inputs, and a resource/synchronization adversary audit. Refresh on changes to
source, binary, runtime, weights, input, tile/cache schedule, capture boundary
or sample selection.

The final release probe and fresh upstream runs now complete inside the guard.
The prefix materializes only 90 original BF16 tensors (2,096,936,960 bytes),
bitwise joined to the native decoder cache. Both full stage-2 activations and
both old-cache buffers match upstream bitwise: 773,849,088 BF16 payload bytes
in total. The cold cache is all zero; the cached tile's old cache equals the
channelwise final temporal frame of the preceding activation. Target weight
and bias exports separately match their exact checkpoint payloads. The latent
fixture is the authenticated prefix of a retained full reference, not a new
DiT-generated distribution or a complete decoder invocation.

All full native MPS controls pass on 188,743,680 output values per case.
The fresh pinned upstream operator matches all 8,192 sampled pre/post-bias
values per case, including its biased Conv3D call versus separate BF16 bias
addition. All 18 native F32 schedules and the exact-integer reference still
fail at unchanged `atol=rtol=2e-5`:

| Pre-bias tolerance failures | Cold tile | Following cached tile |
| --- | ---: | ---: |
| Ordered FMA: boundary / stratified / midpoint | 0 / 5 / 1,850 | 0 / 3 / 1,944 |
| Exact integer: boundary / stratified / midpoint | 0 / 5 / 1,905 | 0 / 2 / 1,909 |

The sample groups contain 512 / 3,584 / 4,096 coordinates respectively.
The midpoint group is deliberately enriched; these are not error-prevalence
estimates. Exact accumulation staged through F32 changes 37/46 BF16 values,
repairs 27/40 direct-exact failures, and introduces 10/6 new failures.
It still fails on 1,893/1,877 coordinates. Staged rounding changes observable
values without identifying a general cause; equal BF16 results alone do not
establish equal F32 sums. Mathematical exactness does not identify the upstream
reduction schedule. A grouping's score on these selected points
is not a candidate for promotion without a held-out check.

The connected native prefix and reductions retain one terminal drain and zero
instrumented named hot allocations/readbacks/preterminal waits. Named GPU
residency is 5,945,318,352 bytes. Sampled process-group peaks are 7.878 GiB for
native prefix plus diagnostics, 7.106 GiB for the separate upstream prefix,
and 4.561 GiB for the separate operator oracle. All preserve the cache-aware
10% reserve with zero observed swap growth. Sampling cannot exclude opaque
driver/compiler waits or between-poll spikes. The diagnostic interval includes
the prefix, multiple full convolutions, sampling and 19 native candidates;
it is neither a single-convolution benchmark nor a generation-rate estimate.

The final comparator checks report schema, nested pinned source/checkpoint,
cache joins, reference lineage and artifact records as well as full payloads.
Report consistency is not execution proof: separate source-pinned launcher
receipts record the observed runs. Nine CLI checks reject ordered/exact
candidate promotion, absent prefix evidence, altered flat/nested provenance,
and a matching corrupted cache whose previous-frame invariant fails.
Fourteen comparator, four upstream-prefix and 44 memory-guard tests pass,
as do two native CPU prefix-contract examples. The correlated source/evidence
adversary is scoped to this diagnostic; no production arithmetic is replaced.
Every recorded real comparison supplies `--require-upstream --prefix-oracle`.
The CLI still permits synthetic comparisons when both prefix metadata and
the oracle argument are absent; it is not a general real-input admission gate.
Whole-fixture authentication comes from the separately pinned launch and oracle
execution, rather than metadata consistency alone.
The curated result is
`.landmark/evidence/2026-11-08-dcae-real-stage2-reduction.json`.

Next discriminator: isolate inner-tile versus outer-partial F32 accumulation
at the real failing coordinates, using sparse K masks while holding the full
operator geometry fixed. Any kernel-selection change remains a confound.
Test a surviving formulation on a separate tile before promotion. Further
layout/grouping sweeps without a discriminating arithmetic probe are parked.

## Conv3D K-tile accumulation and partial-sum merge

Reuse the authenticated stage-2 input/cache and output coordinates. Use
tap-major K with 64 adjacent channels per tile, giving 216 tiles. Materialize
two FP32 partial vectors per point (forward and reverse local FMA), then
consume each immutable vector in both merge orders. The paired comparisons
hold one factor fixed. Report the prior failing coordinates separately from
the boundary, stratified and midpoint-enriched populations.

For upstream, run the original Conv3D operator with the same full tensor
geometry and runtime weight masks for four individual tiles and their union.
Every eighth output channel keeps only the middle coefficient of each tile,
giving a one-product indexing and arithmetic control at the same geometry.
Test all 15 binary addition trees over the four stored upstream FP32 sums
against the upstream union. Compare raw FP32 and the unchanged BF16 gate
(`atol=rtol=2e-5`), and calibrate FP32-to-BF16 against the actual upstream
BF16 call for each mask. Freeze the mask and policies before looking at the
following cached activation.

This opt-in diagnostic runs Metal and the pinned upstream operator on the
same MPS stream with one terminal readout, guarded at 10% cache-aware reserve,
18 GiB process cap and 64 MiB swap growth. Admission requires the prior
authenticated payload digests. DoD requires the fresh full-shape BF16
baseline, the singleton control, a sensitive paired experiment, and a guarded
run on both activations. An exact F32 mismatch with all 15 trees rejects
addition-only composition of these separately computed sums in this mask
geometry. Actual internal boundaries and the full-weight failure mechanism
remain open. Masking and dtype can affect backend arithmetic. Rollback
removes the opt-in probe.

The fresh unmasked BF16 and F32 upstream outputs reproduced their respective
retained controls on all 8,192 points in each case. The Metal cancellation
witness gave `[0, 2^-24, 0, 2^-24]` for FMA and merge. With the other
factor held fixed, changing the selected order moved these numbers of values
among the points that failed the previous ordered-FMA gate:

| Factor | Cold F32 / BF16 (1,855 points) | Cached F32 / BF16 (1,947 points) |
| --- | ---: | ---: |
| Reverse FMA within K64 | 1,030–1,083 / 55–62 | 1,072–1,079 / 53–54 |
| Reverse merge of frozen F32 partials | 1,685–1,689 / 154–163 | 1,773–1,777 / 176–181 |

The ranges are the two fixed settings of the other factor. All four native
variants still fail the unchanged full-weight BF16 gate, with 1,867–1,887
failures cold and 1,846–1,880 cached. Boundary failures are zero; stratified
failures are 5–6 cold and 1–2 cached. The remainder is in the
midpoint-enriched group, so these counts do not estimate prevalence.
Comparing the full-weight native F32 values directly to upstream F32 gives
one or two tolerance failures cold and two or three cached.

Forward FMA reproduced each of the four isolated upstream F32 masks exactly
on both activations. Reverse FMA differed on 1,482–2,194 points per mask and
case. Both orders matched the single-product controls on all 1,024 such
points per mask and case. In the union, none of the 15 F32 trees was exact on
2,765/2,719 points (cold/cached), including 638/660 of the previous failing
points. Every tree was exact in the single-product control, including
939/977 points where all four products were nonzero. The largest pointwise
best-tree F32 error was `7.153e-7` in each case. Every tree passed the F32
tolerance check. After one BF16 rounding, 13 fixed trees passed the union
gate on both activations; 12 matched numerically exactly in both. For each
mask and case, rounded upstream F32 matched the actual upstream BF16 call.
Exact F32 non-composability of these isolated sums and sparse BF16 gate
passage are therefore separate observations.

The guarded GPU run used the pinned Tencent source at `911cea6458a8` with
Torch 2.8.0 MPS on M2 Max, macOS 26.6.2. Sampled process-group peak was
4.372 GiB, minimum estimated headroom 21.350 GiB, and observed swap growth
zero. There was one terminal transfer for numerical reporting and no
explicit intermediate device wait. Sampling and source inspection leave
driver-internal waits and between-poll spikes unobserved. The 72.494 s
interval includes all masked controls and is diagnostic elapsed time.

The source-pinned command, execution receipt, guard samples and 4.5 MiB
observation payload are in
`/opt/mldata_ssd/cogni-ml/dcae-k-partials-20261108.ieU66mM0`.
The payload SHA-256 is
`0b9b08839b49c34f13ff527313bce7bce8b43ab256552017878d529f913a09dc`.
The pinned prior artifact manifest provides both per-case index lists.
They share the first 4,096 coordinates and have case-specific midpoint
halves.

The offline comparator replays the 15 GPU trees and checks the BF16 rounding
against the drained F32 values. It reproduced the independent counts on the
prior failing indices. It rejected corrupted bytes, a changed fixture pin,
an invalid cancellation witness, aliased merge directions and swapped tree
outputs. The last two controls had updated payload digests. The admitted
`comparison.json` in the run directory has SHA-256
`ea3a6039458dcaa06ba0c1606715d75b707ddd1d9c5446b1fa2cfac2aaa67a5d`.

The next discriminator uses the adjacent K64 pair alone and compares
carrying the first tile's accumulator into the second against merging their
isolated sums. A different mask/order must reproduce its prediction on the
cached case before it can inform a native kernel. Refresh on source,
backend, geometry, mask, sample or fixture changes.

## Conv3D accumulator carry across adjacent K64 tiles

Test pairs `106/107` and `181/182` in distinct temporal taps on both retained
activations. Freeze forward FMA within each tile. Compare the full-geometry
upstream union in F32 with `F32(A + B)`, `fold(B, A)` and `fold(A, B)`, where
`A` and `B` are the isolated F32 sums. Score exact matches where the
alternatives disagree, including the prior failing coordinates. Each tile
must first reproduce the separately masked upstream output. A cancellation
control must distinguish both carry directions from the sum and channels
with one product per tile must reproduce the union. Fresh unmasked outputs
must reproduce the pinned F32 and BF16 baselines. The actual BF16 union is
checked separately at `atol=rtol=2e-5`.

Retain the previous 10% cache-aware reserve, 18 GiB process-group cap,
64 MiB swap-growth cap and single terminal transfer. The new probe is
opt-in and rolls back by removal. A second pair and a cached activation
falsify a carry rule inferred from the first cold pair. Kernel selection,
full-weight equivalence and performance are outside this comparison.

Each isolated tile matched the pinned upstream F32 bytes on all 8,192
samples per case. Carrying the accumulator A → B matched the upstream
union bitwise in F32 and after rounding to BF16 on all four comparisons
(32,768 selected values). These are exact F32 mismatch counts:

| Adjacent tiles | Cache | Add the isolated sums | Carry A → B | Carry B → A |
| --- | --- | ---: | ---: | ---: |
| 106/107 | Cold | 2,340 | 0 | 3,465 |
| 106/107 | Previous frame | 2,353 | 0 | 3,381 |
| 181/182 | Cold | 2,119 | 0 | 3,105 |
| 181/182 | Previous frame | 1,996 | 0 | 3,017 |

The GPU cancellation witness distinguished both carry directions and the
single-product channels matched all three policies. On the previous
full-weight failing coordinates, carry A → B was the only exact F32
candidate on 459/469 samples for 106/107 and 413/406 for 181/182
(cold/cached). All three policies passed the F32 tolerance gate. After BF16
rounding, adding the isolated sums also matched exactly in all four cases.
Reverse carry had one BF16 gate failure on cold 106/107 (`abs=2.4414e-4`).
The fresh unmasked outputs matched the retained F32 and BF16 baselines.

The corrected command exited zero with unchanged pinned sources. Its
2,424,832 observation bytes matched those saved by the first run before a
report-key error. The post-drain audit independently replayed the counts
and rejected an aliased carry, bad BF16 rounding, changed isolated-tile
samples, and changed unmasked reference samples. The source-pinned command,
guard telemetry, audit and report are in
`/opt/mldata_ssd/cogni-ml/dcae-adjacent-k64-20261108._hlrsdg4`.
The observation SHA-256 is
`009eec11878bc8138d2f5aea9970d088cdedd04882af55bdf2e40e30f458f194`.
The comparison SHA-256 is
`9059657f4581349f4857506a6cd842cfcb5fb611b866e2f87f84c7d548947150`.

The corrected run used one process, one terminal reporting transfer and no
explicit intermediate wait. Sampled peak was 4.409 GiB, minimum cache-aware
headroom 19.372 GiB, and observed swap growth zero under the 10% policy.
The 82.129 s interval includes all controls and is diagnostic elapsed time.

The result distinguishes continuous FMA from adding frozen F32 sums inside
these two sparse, same-tap masks. It does not identify the dense reduction
schedule. Keep the production MPS path. Next compare whole-tap carry with
channel-interleaved FMA across temporal taps on the same channel range;
`34/106` also reaches the prior-frame cache and `106/178` provides a second
pair. Require a prediction on both activations before expanding the mask.
Refresh on source, backend, geometry, masks, sample or fixture changes.

## Conv3D carry across temporal taps

Keep the authenticated C512/T40/H72/W128 fixture and the original operator.
Mask the same 64 input channels (128–191) at the center spatial position
of temporal taps 0/1 (`34/106`) and 1/2 (`106/178`). Test isolated sums,
F32 addition of those sums, and continuous FMA in both tap directions.
For continuous FMA, alternate chunks of 1, 2, 4, 8, 16, 32 or 64 channels
from each tap; 64 is whole-tap carry. The primary discriminator is 1
versus 64. The other sizes test whether an intermediate chunk width
reproduces the observations without changing the upstream calls.

Before ranking a candidate, require identical F32 isolated-tile bytes and
fresh unmasked controls, GPU cancellation witnesses with known answers,
and singleton agreement. On the first frame of `34/106`, the first tap
must be zero with the cold cache and nonzero on active cached samples.
On the last frame of `106/178`, the second tap reads zero padding in both
cases. Report exact F32 and actual BF16 comparisons separately, including
active cache points and prior full-weight failing coordinates. Only a
fixed policy that survives both pairs and both activations merits a
wider-mask test. An exact match is scoped to these masks and geometry.

The opt-in GPU probe is guarded by the existing cache-aware 10% floor,
18 GiB process-group cap and 64 MiB swap-growth cap. Its DoD is a successful
guarded launch and offline replay of the drained observations, including
rejection of a forged boundary or aliased policy. One terminal transfer;
no model-wide run, native kernel promotion or performance claim. Removal
of the diagnostic is the rollback.

The guarded capture and offline replay passed. `g8_ab` was the only fixed
policy with exact F32 on all 32,768 sampled outputs: visit eight channels
of the earlier tap, eight of the later tap, then advance the channel
range, carrying one accumulator throughout. These are F32 bit mismatches:

| Input | Mask | Merge | `g1_ab` | `g4_ab` | `g8_ab` | `g64_ab` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cold | 34/106 | 2,981 | 1,195 | 777 | 0 | 2,520 |
| Cold | 106/178 | 2,970 | 1,263 | 820 | 0 | 2,504 |
| Cached | 34/106 | 3,044 | 1,254 | 825 | 0 | 2,628 |
| Cached | 106/178 | 2,931 | 1,239 | 802 | 0 | 2,415 |

On the cached first frame, both contributions were nonzero at 327 points;
`g1_ab` and `g64_ab` differed at 114 of them. `g8_ab` matched all 327.
The post-drain audit located the nonzero cache and weight words for 45
active singleton points and confirmed the cold-cache and right-padding
controls. The 14 GPU cancellation inputs distinguished all 15 policies.
Fresh unmasked controls and isolated F32 samples matched their respective
references. All 15 policies passed the actual BF16 gate at `atol=rtol=2e-5`
on these sparse masks; `g8_ab` had zero BF16 value differences. This gate
does not distinguish the candidate orders on the masked samples.

The source-pinned command, sampled guard telemetry, observations, replay
and adversary audit are in
`/opt/mldata_ssd/cogni-ml/dcae-cross-tap-k64-20261108.8x8migwz`.
The observation SHA-256 is
`c4a7c34c63414d08fab98054878afdbb1989c74b936c26540e70eec496c8304a`.
The comparison SHA-256 is
`1d88a280fe180991772472b0b4c7d99199157288ef40bdede05485d928935c1c`.
The audit independently recounted the F32 and BF16 comparisons and
rejected aliased policies, invalid BF16 rounding and a forged cold-cache
contribution with consistent isolated reference values. It imports no
Torch and dispatches no Metal. To reproduce the report without a GPU:

```sh
python3 -B examples/support/dcae_v_cross_tap_k64.py \
  /opt/mldata_ssd/cogni-ml/dcae-real-stage2-20261108.9Enu7v29/real \
  /opt/mldata_ssd/cogni-ml/dcae-cross-tap-k64-20261108.8x8migwz --replay
```

The admitted run took 72.058 s including all controls, with one terminal
reporting transfer and no explicit intermediate wait. Sampled group peak
was 5.112 GiB, minimum cache-aware headroom 16.419 GiB, and swap growth
zero. This elapsed time is for the diagnostic, not a kernel speedup.

The evidence establishes G8 for the tested masked operator, including
the cache edge. The next falsifier is an exact prediction for the three
aligned taps `34/106/178` plus a crossed mask with an earlier spatial tap
or disjoint channel blocks. This separates pairwise agreement from a
general channel-group order. Keep the production MPS path until the
dense F32 and BF16 checks pass. Refresh this result on changes to the
upstream operator, MPS runtime, precision, geometry, masks or fixture.

## Conv3D G8 prediction on three-tile masks

Predict one F32 FMA accumulator, ordered by global eight-channel group,
then ascending `kt,kh,kw`, then lane. Test `34/106/178` (three aligned
temporal taps) and `99/106/107` (the left spatial tap in channels 192–255
crosses the center tap in channels 128–191). Use G4, G16, reverse tap
within G8, G64 channel-first, and tap-first controls. G64 and tap-first
share an order on the aligned mask and differ on the crossed mask. GPU
cancellation inputs must distinguish the unequal schedules across masks.

Keep the same stage-2 geometry, weights, sampled coordinates and cold/live
cache. Recheck full upstream outputs, isolated tiles, the cold first frame,
last-frame temporal padding and left spatial padding against the pinned
controls. Require interior samples that exercise all three tiles and a
cached first-frame witness. An F32 bit mismatch with these controls intact
falsifies the candidate for that mask. Test each rounded policy separately
against actual upstream BF16 at `atol=rtol=2e-5`.

The diagnostic is opt-in. Its completion check is a cache-aware guarded
`dcae_v_crossed_g8.py --admitted` capture with the existing 10% floor,
18 GiB process-group cap and 64 MiB swap-growth cap, followed by `--replay`
and an offline challenge to the controls. Use one terminal transfer and
zero explicit intermediate waits. Scope is the sampled masks. The probe
can be rolled back by removing its two support files.

G8 was the only fixed policy with bitwise F32 agreement on all 32,768
outputs.

| Cache | Mask | G4 | G8 | G16 | G8 reverse | G64 | Tap-first |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold | 34/106/178 | 1,742 | 0 | 2,334 | 3,277 | 3,697 | 3,697 |
| Cold | 99/106/107 | 928 | 0 | 1,297 | 2,283 | 2,947 | 3,610 |
| Live | 34/106/178 | 1,819 | 0 | 2,352 | 3,278 | 3,787 | 3,787 |
| Live | 99/106/107 | 865 | 0 | 1,231 | 2,176 | 2,911 | 3,543 |

On the aligned cached first frame, all three tiles contributed at 327
sampled points, and reverse G8 differed from G8 at 127 of them. The audit
verified nonzero prior-frame activation and weight words at all 45
active singleton points in that set. The two new masks projected to the
pinned pair outputs in F32 and actual BF16 at all tested cache and
padding edges. Fresh unmasked outputs matched the pinned controls, and
the eight GPU cancellation inputs distinguished all six policies across
the two masks.

G8 had zero BF16 value differences against the actual upstream BF16
outputs. G16 exceeded the unchanged gate once (cold aligned mask,
absolute error `3.0517578125e-5`). The other four policies passed.

The source-pinned command, guard telemetry, observations, comparison
and post-drain audit are in
`/opt/mldata_ssd/cogni-ml/dcae-crossed-g8-20261108.Cz0kzvYV`.
The observation SHA-256 is
`3b662bdd3d44ac30666acf2b3e76ab52442f7ec1b603a507dc85b1ea61d232fa`.
The comparison SHA-256 is
`ab940abbe273c9348704ff91acc3c98a6a533336ce02acc29c81fd18cda8da4f`.
The audit independently enumerated G8 and recounted all F32/BF16
failures, then rejected aliased policies, bad rounding, a forged cold
frame and a changed full-weight control.

```sh
python3 -B examples/support/dcae_v_crossed_g8.py \
  /opt/mldata_ssd/cogni-ml/dcae-real-stage2-20261108.9Enu7v29/real \
  /opt/mldata_ssd/cogni-ml/dcae-crossed-g8-20261108.Cz0kzvYV --replay
```

The guarded diagnostic on Apple M2 Max (macOS build `25G83`, Torch 2.8.0)
took 42.771 s including its controls. Sampled process-group peak was
5.107 GiB, minimum cache-aware headroom 18.835 GiB and swap growth zero.
It used one terminal reporting transfer and zero explicit intermediate
waits.

The next falsifier is the same channel-group/tap/lane order on dense K,
on the retained sampled coordinates with both caches. Score the prior
full-weight failure points separately. Dense parity and any performance
gain are separate gates. Refresh on upstream operator, MPS runtime,
precision, geometry, mask or fixture changes.

## Dense Conv3D G8 on the retained stage-2 coordinates

Predict bitwise F32 parity for a continuous FMA over all 13,824 terms,
ordered by eight-channel group, then ascending `kt,kh,kw`, then lane.
Run the same native kernel with full weights and with the two preceding
three-tile masks. The masked arms must reproduce their pinned upstream
F32 values, including boundary points. Fresh dense upstream F32 and
actual BF16 must reproduce their pinned controls. An exact F32 mismatch
with these controls intact falsifies this dense schedule at that point.

Compare the native and upstream F32 results rounded to BF16 separately
against actual upstream BF16 at `atol=rtol=2e-5`. Record boundary,
stratified, midpoint-selected and prior-failure counts. Count both fixed
and new BF16 failures to expose churn behind a lower aggregate. These
counts describe the selected coordinates.

The completion check is a guarded `dcae_v_dense_g8.py --admitted` capture,
then `--replay` and offline challenges to the controls. The process-group
cap is 6 GiB, based on the previous 5.107 GiB peak with more upstream arms
and the same tensor shapes. The cache-aware reserve is 10% and the
swap-growth limit is 64 MiB. One terminal transfer and zero explicit
intermediate waits. Removing the isolated diagnostic rolls it back.

The full-weight G8 kernel matched upstream bitwise in F32 and after
rounding to BF16 at all 16,384 tested coordinates before bias. The BF16
reference comes from a separate upstream call with BF16 inputs and weights.
Both masked controls matched their pinned upstream F32 bytes at every point.
Fresh dense upstream F32 and actual BF16 matched the authenticated references.

| Cache | F32 bit mismatches | BF16 bit mismatches | Prior BF16 failures resolved | New BF16 failures |
| --- | ---: | ---: | ---: | ---: |
| Cold | 0 / 8,192 | 0 / 8,192 | 1,855 / 1,855 | 0 |
| Live | 0 / 8,192 | 0 / 8,192 | 1,947 / 1,947 | 0 |

Each cache case includes 512 boundary, 3,584 stratified and 4,096
midpoint-enriched coordinates. The previous BF16 failures comprised 5/3
stratified and 1,850/1,944 midpoint-enriched coordinates (cold/live).
The offline audit independently verified the 13,824-weight traversal,
raw F32 and BF16 bits, and earlier failure counts. It rejected swapped
mask and cache references and a corrupted BF16 rounding value. Changing
two candidate results with consistent rounding correctly produced one
surviving prior failure and one new failure for each cache case.

The source-pinned command, guard telemetry, observations, replay and
audit are in `/opt/mldata_ssd/cogni-ml/dcae-dense-g8-20261108.3JZuRnrX`.
The observation SHA-256 is
`8c9b561030bd1facf8c941f532089aa945d2727e2c3f9496c89d15eaebde5510`.
The comparison SHA-256 is
`f97086af6de6e36913b2f165e2b0ee55f6dab3dc48aaa164085350c86dc95603`.
The audit SHA-256 is
`2be30f050b98a488b0fcb6072137278c53a5eaf47c8259ce617b1bcaefca3056`.

```sh
python3 -B examples/support/dcae_v_dense_g8.py \
  /opt/mldata_ssd/cogni-ml/dcae-real-stage2-20261108.9Enu7v29/real \
  /opt/mldata_ssd/cogni-ml/dcae-dense-g8-20261108.3JZuRnrX --replay
```

The guarded diagnostic on Apple M2 Max (macOS build `25G83`, Torch 2.8.0)
took 15.133 s including all controls. Sampled process-group peak was
5.061 GiB, minimum cache-aware headroom 10.746 GiB, and observed swap
growth zero. It used one terminal readout and zero explicit intermediate
waits. The 62 focused comparator, oracle and memory-guard tests passed.

The next performance falsifier is an output-tiled native kernel that
preserves one G8 accumulator per output. Verify full-tensor F32 and BF16
parity on both cache cases before timing paired full-shape Conv3D against
MPS. The sampled result covers stage-2 conv1 before bias for the pinned
fixture and runtime. Refresh it on upstream operator, MPS runtime,
precision, geometry, weight, fixture or sampling changes.

## Full-output stage-2 Conv3D tiled over G8 accumulators

Keep the authenticated cold and live stage-2 inputs and weights. Tile only
the output coordinates and retain one continuous F32 accumulator per
coordinate in the dense G8 order. Pack BF16 weights on GPU for adjacent
output channels, with identical bits. Cover all 188,743,680 outputs per
cache before bias. Test native F32 and native BF16 stores separately against
the corresponding pinned upstream calls. Require exact F32/BF16 equality
and report failures at the existing `atol=rtol=2e-5` gate.

The falsifiers are a missing or duplicate output coordinate, a failed
comparison witness, a mismatch against the retained upstream sample, or any
full-output difference. Initialize candidate buffers with non-finite
sentinels and reduce coverage, raw-bit differences and finite errors on GPU.
Time hot BF16 only after parity. Warm both arms and measure both orders on
each cache with the same operands and output geometry. Keep preparation and
comparison outside the intervals. Require one terminal readout per probe
chain and zero explicit intermediate GPU waits.

Admit one guarded process at a time with an 8 GiB process-group cap, 10%
cache-aware reserve and 64 MiB swap-growth stop. Completion requires a
guarded `dcae_v_tiled_g8.py` parity capture, then paired full-shape BF16
intervals against the pinned MPS operator. The risk is limited to the
diagnostic; removing its files rolls it back. Production promotion requires
a separate decoder-level performance result.

The native kernel tiles four adjacent output channels and two x positions
per thread, with adjacent SIMD lanes visiting adjacent x. Sixteen
dispatches cover 32 output channels each. BF16 inputs and reordered weights
remain resident, with one F32 accumulator per output and an optional direct
round-to-nearest-even BF16 store. No partial reductions or intermediate F32
output buffer are used in the BF16 arm.

| Cache | F32 bit mismatches | BF16 bit mismatches | Max absolute error |
| --- | ---: | ---: | ---: |
| Cold | 0 / 188,743,680 | 0 / 188,743,680 | 0 |
| Live | 0 / 188,743,680 | 0 / 188,743,680 | 0 |

Each fresh upstream column also reproduced its 8,192 retained sample
values bitwise. The comparator witness covered a strided tail and
distinguished signed zero, a finite error inside tolerance, a finite error
outside tolerance and an equal-bit NaN. An offline audit checked coverage
and spatial bounds, and rejected shortened coverage and a forged tail
mismatch.

The source-pinned parity command, guard telemetry, compact result and audit
are in `/opt/mldata_ssd/cogni-ml/dcae-tiled-g8-20261108.8gG8Qbk5`. The parity
SHA-256 is `53da951ea304722546f6cdac622646ec73e871ab416bc444540e50d7af63d3b1`.
The guarded run on Apple M2 Max (macOS build `25G83`, Torch 2.8.0) took
23.392 s including setup, both precisions, both caches and controls.
The sampled process-group peak was 5.084 GiB, minimum cache-aware headroom
25.610 GiB and observed swap growth zero.

Reject the stage-boundary timestamp attempt. The first clock witness
reported 0.048 ms for two F32 1536-square MPS matmuls. A warmed witness
returned an invalid timestamp. Those markers do not establish elapsed
Conv3D time across the MPS graph path on this runtime.

The paired timing uses completed BF16 batch wall time in one process.
The setup chain prepares weights and activations, warms both arms in
ABBA order on each cache, and checks both full BF16 outputs. Each
measured batch runs two convolutions and reads one scalar after the
second call. The scalar is checked against the warmup result. ABBA
and BAAB are measured separately on each cache. Wall time per call
includes host encoding and the amortized terminal drain. The tiled
arm uses a prepared output buffer, and the pinned MPS operator uses
its eager output allocation. No intermediate tensor reads, timestamp
encoders, explicit device waits or CPU arithmetic fallback are used.

| Causal cache | Tiled G8 median per call | MPS median per call | Median paired G8 / MPS | Pair range |
| --- | ---: | ---: | ---: | ---: |
| Cold | 2.735 s | 2.696 s | 1.006× | 0.977–1.038× |
| Live | 2.891 s | 2.763 s | 1.034× | 0.989–1.067× |

The cold-cache pairs change sign with order, and the live-cache pairs
also span parity. This single guarded matrix establishes no repeatable
speedup. Keep the tiled G8 route as a diagnostic. MPS is the production
decoder route. The evidence covers the pinned stage-2 pre-bias Conv3D
shape and fixtures on M2 Max. These are warm operator-batch times
including host encoding and terminal completion. Refresh on changes to
the upstream operator, runtime, GPU, geometry, inputs, weights, cache
semantics, arithmetic or timing method.

The source-pinned timing command, execution receipt, guard telemetry and
independent recount are in the same scratch evidence directory. The
timing SHA-256 is `9c90c854e2987e5b8008f7769444cde97927c27ff96b53bd06209fb3273c6cf1`.
The guarded timing took 113.221 s including setup, warmups and all
measured batches. The sampled process-group peak was 5.933 GiB, minimum
cache-aware headroom 23.997 GiB and observed swap growth zero. Both runs
kept the 10% reserve. The focused 62 tests passed.
