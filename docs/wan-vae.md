# Native Wan 2.1 VAE component

`ML::Video::WanVAE` implements the Wan S2V image/motion encoder and causal video
decoder with resident F32 weights, activations and accumulation. All neural
operations and frame copies execute on Metal. The opt-in API requires an
explicit workspace budget and retained caller-owned buffers. Setup prepares
pipelines, five scratch buffers and 24 encoder plus 32 decoder history slots.
Encoding adds no GPU buffers, host waits or activation downloads. Passing an
optional `MPSCommandChain` splits long convolutions and attention queries into
asynchronous continuations on the same serial queue; the caller performs the
single terminal wait. Omitting it keeps the caller's existing command encoder.

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

The 16x24 final spec additionally decodes twice in one command chain with the same
workspace. The repeated output is bit-identical and the tracked GPU-buffer
allocation count does not change during encode/decode. After the scratch fix
below, both default and segmented paths passed six examples with zero failures,
errors or pending cases, including two direct GPU tiling checks. These are
correctness specs, not generation benchmarks. Full 480p memory, performance,
batch sizes above one, and end-to-end video quality remain separate gates.

## Scratch-capacity correction

The earlier scratch plan counted the final decoder channels after projection.
The last nearest upsample writes twice that many channels before projection:
`B * 4 * 2dim * H * W`. The old 16x24 plan allocated 147,456 F32 values per
slot, while the second and later latent chunks require 294,912. Frame zero fits;
later chunks could write out of bounds. Historical small-fixture passes do not
qualify that old allocation, even where their output happened to pass.

`activation_scratch_values` now supplies one checked plan to preparation,
workspace validation and the complete runner's phase-pool planner. It includes
raw RGB staging, encoder activations, the pre-projection decoder upsample,
posterior channels, QKV and temporal-convolution peaks. Dispatch capacity guards
reject undersized input and output views before encoding. At 832x512 each of
the five scratch slots requires 327,155,712 F32 values (1,308,622,848 bytes).

The refreshed 16x24 default and segmented paths both pass the unchanged upstream
gate. A paired diagnostic compared 147 GPU snapshots, including all decoded
frames: 6,189,984 values are bit-identical, finite and within the same gate. It
performs one terminal host wait, zero activation downloads and zero tracked
MetalBuffer allocations while encoding. `WanVAETrace` is optional: callers must
allocate disjoint snapshot storage before encoding and retain it through
completion. This diagnostic does not establish full-resolution quality.

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
  --link-flags="$BUILD/bridge.o $BUILD/metal_mps_command_chain.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++ -fuse-ld=/usr/bin/ld"

WAN_VAE_FIXTURE="$EVIDENCE/vae-spatial.json" WAN_VAE_EXECUTION_CHAIN=1 \
  COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$BUILD/wan_vae_spec" 120 4096
```

Use height/width 8 for the smaller fixture. Local evidence is under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full`:
`vae-cache-fixed-native.log` (historical 8x8),
`vae-scratch-fixed-default-native.log` and `vae-scratch-fixed-chain-native.log`
(current 16x24), and `vae-paired-trace-fixed.json` / its native log. The two
six-example runs also include `spec/wan_vae_dispatch_tiling_spec.cr`: full versus
tiled attention and F32/BF16 convolution outputs are bit-identical, with one
terminal wait and no encoding allocations. Set `WAN_VAE_EXECUTION_CHAIN=0` to
repeat the default route.
Fixture SHA256 values are
`80d8506a9b0ae293e78e1e4fd7f3b4f5e9c4b1b24ddbd080dc73f393df13d09b`
and `94c483926c2b03a316926dbb86b48aa4f2b95154454e19fff45c7fa9034400d3`.
GPU runs were serial with finite deadlines, process-tree RSS caps and a 15%
free-memory stop floor. Sampled free memory does not establish physical peak.

Refresh after native/oracle/upstream/checkpoint/shape/arithmetic/runtime/device
changes, or missing/changed evidence. Hashes are not build attestations. The
next gate is full spatial geometry and pipeline lifetime accounting before a
bounded complete request; this component result does not close that gate.
