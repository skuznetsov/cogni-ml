# Original DCAE-V native decoder frontier

Status: native opt-in decoder and full-context BF16 parity verified;
one descriptive decoder time/memory comparison completed.

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
serialization. DCAE's request includes its weight encode/upload; PixelUMM
uploads its block cache during decoder setup, outside the request timer.
DCAE prepared setup was 3.513 s; PixelUMM's decoder-specific setup was
7.167 s. Those setup fields also have different admission boundaries.
There was no matched warmup, frequency control, repeated full-size pair,
or process-cold cache control. DCAE used 42.4% less sampled peak memory.
The request-time ratio is 4.84, but is descriptive because the upload and
setup boundaries differ; it is not a matched speedup result. Neither metric
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

Next performance discriminator: profile the two high-resolution Conv3D
stages in this native decoder. The architecture cost model assigns about 78%
of logical dense Conv3D FLOPs to their residual convolutions; this is a
candidate-selection calculation, not an observed GPU bottleneck or speedup.
