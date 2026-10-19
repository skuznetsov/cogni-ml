# Opt-in native F32 video attention

`ML::Video::MetalOps.attention_tiled` is an unmasked SDPA primitive for separate
query/key lengths, shared batch geometry and head dimensions 64 or 128. It is
opt-in; the existing `attention` implementation remains the default. Q, K, V,
products, online softmax and accumulators stay F32 on GPU. A SIMD group shares
eight query rows over ascending blocks of 32 keys, without a quadratic score
buffer. This is ordinary tiling, not an LTP/WBA claim.

Setup compiles both pipelines. Encoding borrows the caller's serial encoder
and resident buffers, allocates no Metal buffers, submits no commands, waits
for no completion and reads no activations. Keep the buffers alive until the
caller completes the command. Unsupported dimensions, invalid extents, short
or released buffers and output/input overlap are rejected before dispatch.
Concurrent encoders and concurrent reuse of one output are outside this API.

## Executed gate

On Apple M2 Max, the focused and common Metal primitive specs reported
**19 examples, zero failures, zero errors, zero pending**. Six independent
PyTorch 2.8.0 MPS SDPA cases used F32 inputs with CPU fallback disabled:

| B | Q | K | H | D | Input profile |
| --- | --- | --- | --- | --- | --- |
| 2 | 13 | 97 | 3 | 64 | Normal; query and key tails |
| 1 | 64 | 1,024 | 40 | 128 | Normal; real Wan width |
| 1 | 64 | 36,608 | 40 | 128 | Normal; first-request Wan key length |
| 1 | 9 | 67 | 2 | 128 | Q/K multiplied by eight; sharp softmax |
| 2 | 1 | 1 | 1 | 64 | Single key |
| 1 | 9 | 65 | 2 | 64 | K multiplied by 1e-5; near ties |

Every finite output passes the unchanged per-value gate
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`. Each case encodes two
applications in one command; repeated outputs are bit-identical and the cumulative
Metal-buffer allocation counter is unchanged during encoding. Exact case
coverage, byte lengths and input/output file hashes are checked. These probes
challenge tile tails, batch isolation, softmax range and the long reduction.
The correlated mini-Quadrumvirate verdict is ROBUST for this tested primitive
scope. It is not an independent attestation or a full-transformer certificate.

A preceding bounded probe at Q=64, K=36,608, H=40, D=128 measured 52.16 ms GPU
for the tiled kernel and 209.46 ms for the existing kernel. It checked both
outputs against MPS, but used one baseline-first pair. This is a provisional
component timing, not a controlled cold/repeat pair or video-generation time.
Full-query cost, distinct-layer propagation and full-video speed remain open.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh "$ORACLE_PYTHON" 120 4096 \
  spec/support/video_attention_tiled_reference.py --output "$EVIDENCE/attention" --long

COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh crystal 180 6144 build \
  spec/video_attention_tiled_spec.cr spec/video_metal_ops_spec.cr \
  -o "$BUILD/video_attention_tiled_spec" \
  --link-flags="$BUILD/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld"

VIDEO_TILED_ATTENTION_FIXTURE="$EVIDENCE/attention/manifest.json" \
  COGNI_RUN_SAFE_MIN_FREE_PCT=15 scripts/run_safe.sh \
  "$BUILD/video_attention_tiled_spec" 120 4096
```

Local evidence is under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full`:
`attention-tiled-qualified-oracle.log`, `attention-tiled-final-{build,native}.log` and
`attention-tiled-qualified/manifest.json`. The provisional timing is in
`attention-tiled-native.log`. These local logs are evidence pointers, not
committed fixtures or build attestations. Refresh after changes to either
implementation, geometry, PyTorch/runtime, hardware or tolerance. A missing or
failed fixture blocks admission into the full native Wan pipeline.
