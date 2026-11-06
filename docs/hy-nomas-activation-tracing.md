# Hy-Nomas diagnostic activation tracing

This opt-in recorder observes the pinned upstream BF16 DiT equations through
the existing PyTorch 2.8 MPS attention adapter. It preserves the selected
four-step source trajectory. It does not change native generation defaults.

The diagnostic exports tensors from GPU to CPU and external SSD at every
block boundary. These exports add synchronization and disk work; elapsed
recording time is not an inference benchmark. Normal inference does not use
the observer.

## Scope and storage

The first admitted fixture is one five-second, 512x288 continuation with 30
latent frames, 4 Euler steps and 54 DiT blocks. Each layer runs the reference
cache writer T0 followed by one selected T1 arm. The default arm is `A1_high`,
which includes the existing soft attention prior; it is not vanilla upstream
CUDA/Magi attention.

Every one of the 432 block calls records 53 named boundaries:

- Image/text inputs, modulation, LayerNorm and projection inputs/outputs.
- Normalized Q/K, image Q/K before and after RoPE, actual attention Q/K/V.
- Attention outputs, projection, both residual boundaries, FFN hidden and
  activated outputs, and post-dispatch cache K/V.
- The time vector and exact RoPE phases used by the call.

Each step also records floating leaf-module outputs of the prelude/refiner and
head, patches, predicted velocity and the updated Euler latent. Functional
temporaries without an observed boundary, opaque MPS scratch, full attention
score/probability matrices, Qwen internals and PixelUMM internals are excluded.
This is a named-boundary trace, not a claim to capture every internal value.

For scale, materializing both F32 QK scores and probabilities for the compact
valid rows would require about 1.57 TiB across this run:
`2 * 216 * 16 * 4 * (4583**2 + 4544 * 9127)` bytes. The fused MPS attention
path does not expose those full matrices. Saved Q/K and authenticated masks
support a later selected-query reconstruction; it must preserve the source
scaling, key order, valid lengths and attention prior before being interpreted.

Storage retains the source dtype without quantization. In particular, MPS
autocast leaves LayerNorm and modulated projection inputs in F32; these are not
rounded to BF16 for the trace. Block payload alone is 295,245,250,560 bytes
(about 275 GiB). Eight auxiliary batches each have a fail-closed 1 GiB ceiling.
The writer caps total payload at 300 GiB, each batch at 1 GiB, and preserves
32 GiB free on the external SSD. A partial or interrupted trace retains its
`.incomplete` marker.

The layout is:

```text
RUN/
  activations/
    index.json
    batch-000000.bin
    ...
  replay.json
  final-latent.f32le
```

The aggregate index maps names to shard byte ranges, shapes, original dtypes
and SHA-256 hashes. BF16 uses raw little-endian 16-bit words. To interpret BF16
in NumPy, expand uint16 words to uint32, shift left by 16 bits, then reinterpret
as float32. Do not interpret the words as IEEE float16. Shards remain outside
Git.

## Admission and execution

Inspect the CPU-only plan first:

```sh
python3 -B spec/support/hy_nomas_upstream_activation_recording.py --plan-only
```

Use the authenticated external PyTorch 2.8 oracle environment and
`PYTORCH_ENABLE_MPS_FALLBACK=0`. The output must be a fresh absolute directory
under the canonical external data SSD path; resolve `/opt/mldata_ssd` before
forming the output path. Do not overwrite a previous run. Keep byte quotas,
the file-cache-aware 10% host reserve, the 16 GiB process cap and the 64 MiB
swap-growth stop. Only one model-owning process may run at a time.

Run `--pilot --output RUN` through `examples/support/wan_memory_guard.py` first.
It executes T0 and T1 of block 0, then removes hooks and replays both exact
calls. It requires bitwise output equality and equality of replayed T0 K/V and
mask with original T1 incoming state. The pilot validates every serialized
hash and independently checks BF16 output/cache words and F32 LayerNorm words
against live source values. It does not exercise the final head.

After a successful pilot, use `--execute --output NEW_RUN` with the same guard
and fresh output. The full trace requires all 432 exact block contexts and
every one of the four Euler latents to match the authenticated accepted source
payload bitwise. Original `atol=rtol=2e-5` is not widened. All raw shard and
record hashes are checked before the full run is finalized. A failed equality
or byte check must not publish a successful replay receipt.

The source/input manifests, weights, interpreter, runtime helpers, masks,
precision and geometry are part of the experiment. Preserve the exact source
snapshots and command alongside the external run. Missing or changed evidence
invalidates its use as a replay control.

## Video lineage and interpretation

Decode `final-latent.f32le` directly with upstream PixelUMM in a separate guarded
process. Require its SHA-256 from `replay.json` and bitwise equality with the
F32 expansion of `step4.ARM.head.after_euler` in the trace. Bind decoder noise,
weights, source revision and output pixels/video in the decoder receipt.
An identically named old MP4 or latent bank is insufficient provenance.

The decoder driver is `spec/support/hy_nomas_upstream_trace_decode.py`.
Inspect `--plan-only`, then use `--execute --recording-dir RUN`,
`--index-sha256`, `--replay-sha256`, `--latent-sha256` and a fresh
`--output-dir`. Compute those three hashes from the completed recording.
It authenticates the full trace, exact final latent, decoder weights, pixel
noise and producer sources before MPS execution. It stages one BF16 decoder
block at a time across the five overlapping windows; the full cache is never
mapped. The decode output contains raw pixels, display RGB, MP4 and a receipt.
The internal allocation estimate is supplementary: enforce actual current
host availability with the same external file-cache-aware memory guard,
10% floor, 10 GiB process cap and 64 MiB swap-growth stop. A polled guard does
not certify between-sample peaks. Keep producer sources frozen until decode
finishes because admission verifies their hashes.

The static plan is not an actual-artifact admission test. Before the expensive
full trace scan, the observed run also exercised metadata, replay, exact
BF16-to-F32 latent binding, noise, decoder source, cache layout and CPU model
construction against the real files in the pinned PyTorch 2.8.0 interpreter.
This exposed lexical symlink comparisons, a decoder-only source incorrectly
required in the DiT inventory, and a cache-order assumption. The reader now
uses the canonical pinned upstream root, verifies decoder-only sources
separately, and follows the authenticated cache's lexical tensor-name order.
Revision, content hashes and contiguous byte-range checks remain required.
This bounded preflight does not hash the full activation corpus or execute
MPS; the subsequent guarded decode still performs its full integrity checks.

After completion, a bounded CPU analysis can read selected records without
loading the corpus:

```sh
python3 -B spec/support/hy_nomas_activation_trace_analysis.py \
  --trace RUN/activations --output NEW_ANALYSIS.json
```

It compares attention and FFN residual RMS in fixed coordinate ROIs at layers
0, 13, 26, 39 and 53 across all four steps. Its hash claim is explicitly
`selected_only`; it does not substitute for full recording/decode validation.

The current fixture requests an Alex head turn followed by Resume's glove
gesture. These are requested actions, not labeled ground-truth motion. Inspect
the decoded video and keep observations separate from activation statistics.
Large residual updates, changing temporal latent values or attention affinity
alone do not prove object identity, physical motion or successful control.

## Observed recording (2026-11-06)

The guarded pilot and full recording completed at the frozen source snapshot
of this change. The pilot's traced/untraced T0 and T1 outputs and replayed
cache were bit-identical. The full run contains 23,040 records in 440 shards,
296,265,981,952 payload bytes, and all 432 expected block calls. Full shard and
record hash validation passed. All four Euler latents equal the archived
source control bitwise, a stronger gate than the unchanged mixed tolerance.

The full recorder's sampled process-group peak was 3,715,975,760 bytes;
minimum cache-aware headroom was 15,005,532,160 bytes, and observed swap growth
was zero. The run included diagnostic exports and integrity scans, so its
elapsed time is not an inference-speed measurement.

The exact saved final latent was decoded by the pinned upstream PixelUMM MPS
adaptation into 120 frames at 24 fps, 512x288, five seconds. Its receipt binds
the trace index, replay and final BF16-to-F32 expansion to raw pixels, display
RGB and MP4. All 468 BF16 weight tensor ranges were hashed while streamed;
the declared aggregate cache SHA was not separately recomputed. The decoder's
sampled process-group peak was 5,060,760,976 bytes, minimum cache-aware headroom
24,935,120,896 bytes, and observed swap growth zero. Both successful processes
exited with no live children. Decoder internals were not activation-recorded.

Six selected frames and enlarged actor crops show Resume's glove rising toward
its chest after about three seconds while its body and head remain mostly
fixed. A substantial requested Alex head turn is not visible in these frames.
The first frame is softer than later selected frames; its cause is not
localized. These observations belong to this exact-linked upstream video,
not the rejected association with the previous identically named MP4.
They do not certify full-playback quality or successful actor control.

The first selected-record analysis hashes 64 records at five layers and four
steps. Both fixed coarse ROIs have nonzero attention and FFN residual RMS at
those sampled calls. This excludes only zero updates in the selected tensor
ranges. It does not identify an action-text branch, prove that its signal
survives, or explain weak articulation. The next discriminating analysis is
selected-query attention replay with the exact compact K order and masks,
followed by a same-input intervention; ROI RMS alone cannot justify a motion
controller.

External recording root:
`/opt/mldata_ssd/cogni-ml/hy-nomas-upstream-trace-20261106.zd8_wvuz`.
The `full-v1` recording, guard logs, exact commands and frozen source snapshots
remain outside Git. Missing or mutated sources, weights, inputs, schedules,
geometry, precision or bound artifacts invalidate dependent replay claims.
`decode-v2/display.mp4` is the exact-linked video; `analysis-v1.json` contains
the selected-record statistics and `inspection-v3` the selected-frame crops.

The focused suite below passes 72 CPU tests; eight new Python files pass Ruff.
A read-only adversary review returned ROBUST within the capture-to-video
scope after checking the final BF16 extent/F32 expansion, receipt, decoded
output hashes, 23 saved/current decoder source bindings and guard records.
It did not repeat the full activation or 468-range weight scan. Same-lineage
review is correlated, not independent replication. The generic cache helper's
numeric offset coercion remains outside the robustness claim: this CLI admits
only the exact authenticated integer-valued manifest.

Evidence: `.landmark/evidence/2026-11-06-hy-nomas-upstream-activation-recording.json`;
external artifact index: `qualified-artifacts-v2.json` under the recording root.

Focused CPU checks:

```sh
python3 -B -m unittest \
  spec.hy_nomas_activation_trace_test \
  spec.hy_nomas_activation_trace_analysis_test \
  spec.hy_nomas_upstream_activation_recording_test \
  spec.hy_nomas_upstream_trace_decode_test \
  spec.hy_nomas_soft_route_render_reference_test \
  spec.hy_nomas_soft_route_render_qualify_test \
  spec.hy_nomas_soft_route_test
```

To roll back this diagnostic, omit the observer or the recorder invocation.
The source helper's default three-arm contract remains unchanged.
