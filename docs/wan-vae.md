# Native Wan 2.1 VAE component

`ML::Video::WanVAE` implements the Wan S2V image/motion encoder and causal video
decoder with resident F32 weights, activations and accumulation. All neural
operations and frame copies execute on Metal. The opt-in API requires an
explicit workspace budget and retained caller-owned buffers. Setup prepares
pipelines, five scratch buffers and 24 encoder plus 32 decoder history slots.
Encoding adds no GPU buffers, submissions, waits or activation downloads.

The workspace is exclusively owned by one serialized command stream. It must
not be shared between overlapping command buffers. This contract is documented,
not enforced with an in-flight lease. Reset changes host cache metadata; the
queued writes and reads remain ordered in the caller's encoder.

## Executed scope

The real 194-tensor checkpoint passed the unchanged finite per-value gate
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)` on Apple M2 Max against
PyTorch 2.8.0 MPS at image sizes 8x8 and 16x24. Both cases exercise image
encoding, 73 motion frames (19 latents), and decoding 22 latents (85 raw frames).
The explicit temporal crop starts at raw frame four and delivers 80 frames.
The oracle also records the actual first, second and later upstream temporal
cache states, including the decoder's initial `Rep` sentinels.

The 16x24 final spec additionally decodes twice in one command with the same
workspace. The repeated output is bit-identical and the tracked GPU-buffer
allocation count does not change during encode/decode. The final run reports
three examples, zero failures, errors or pending cases. These are correctness
specs, not generation benchmarks. Full 480p memory, performance, batch sizes
above one, and end-to-end video quality are unverified.

The motion test rejected a frame-major native history writer: the causal
convolution consumes NCDHW history. The writer now stores `[B,C,2,H,W]` with
channel-local temporal slices. Residual scratch selection also excludes the
live skip buffer. Correlated mini-Quadrumvirate review found this serialized
cache/scratch contract ROBUST within the executed fixtures.

## Reproduction and provenance

Upstream: `Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff`,
`wan/modules/vae2_1.py`, source SHA256
`d0dfe9eccd0cdbcb37ffde9887579be58f4c1bb6fc2dbb1b72e8916e170df180`.
The locally converted F32 safetensors checkpoint has SHA256
`2ae212ca06c67f00681d5e155d1a954f5ce81eeedb9f5ffd344829817718991f`.
The spec verifies this digest before loading it. No checkpoint is distributed.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$ORACLE_PYTHON" 180 6144 \
  spec/support/wan_vae_reference.py --upstream "$WAN_SOURCE" \
  --checkpoint "$VAE_CHECKPOINT" --height 16 --width 24 \
  --output "$EVIDENCE/vae-spatial.json"

crystal build spec/wan_vae_spec.cr -o "$BUILD/wan_vae_spec" \
  --link-flags="$BUILD/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld"

WAN_VAE_FIXTURE="$EVIDENCE/vae-spatial.json" COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$BUILD/wan_vae_spec" 120 4096
```

Use height/width 8 for the smaller fixture. Local evidence is under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full`:
`vae-cache-fixed-native.log` (8x8), `vae-spatial-repeat-native.log` (16x24).
Fixture SHA256 values are
`80d8506a9b0ae293e78e1e4fd7f3b4f5e9c4b1b24ddbd080dc73f393df13d09b`
and `94c483926c2b03a316926dbb86b48aa4f2b95154454e19fff45c7fa9034400d3`.
GPU runs were serial with finite deadlines, process-tree RSS caps and a 15%
free-memory stop floor. Sampled free memory does not establish physical peak.

Refresh after native/oracle/upstream/checkpoint/shape/arithmetic/runtime/device
changes, or missing/changed evidence. Hashes are not build attestations. The
next gate is full spatial geometry and pipeline lifetime accounting before a
bounded complete request; this component result does not close that gate.
