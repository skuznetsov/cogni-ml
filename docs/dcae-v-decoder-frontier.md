# Original DCAE-V native decoder frontier

Status: native opt-in decoder and full-context BF16 parity verified;
Conv3D profiling implemented, with no admitted performance gain yet.

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
