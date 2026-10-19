# Native Wan S2V causal audio component

`ML::Video::WanS2VCausalAudioEncoder` implements the pinned Wan MotionEncoder
with F32 weights and arithmetic on Metal. It consumes resident XLSR features,
weights their 25 hidden states, prepends the first feature frame, applies the
local/global causal convolution paths, and emits local and global conditioning.
For 84 input buckets and 17 prefix frames, the convolution lengths are
101/51/26. Dropping five motion-prefix latents leaves 21 conditioning frames.
The local output is `[21,5,5120]`; the global output is `[21,5120]`.

Setup prepares the pipelines and reusable scratch. Encoding uses the caller's
single serialized encoder without allocating GPU buffers, submitting commands,
waiting, or downloading activations. Overlapping use of the workspace is not
supported. There is no implicit lease across command buffers.

## Executed gate

The tiny synthetic fixture and the real 12-tensor checkpoint both pass every
one of the 22 trace checkpoints against pinned PyTorch 2.8.0 MPS. The unchanged
finite per-value gate is
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`.
The final run reports four examples, zero failures, errors, or pending cases.
Both paths execute twice in one encoder; repeated local/global outputs are
bit-identical, and encoding does not change the tracked GPU-buffer allocation
count. An aliased input/output is rejected before encoding.

The tiny fixture exposes a consequential MPS LayerNorm detail: PyTorch's
version-pinned kernel computes raw moments and clamps variance below `1e-6`
to zero. A centered-variance implementation differs by more than 0.1 on the
fixture's nearly constant row. The Wan-specific kernel follows the pinned MPS
reduction and clamp; the general video normalization implementation is unchanged.
This behavior is compatibility with that oracle, not a claim of mathematical
equivalence to other LayerNorm implementations. Correlated mini-Quadrumvirate
review found the serialized component contract ROBUST within these fixtures.

The real fixture loads 380,206,180 bytes of F32 weights and uses 4,673,536 bytes
of scratch. Its admitted plan peak, including staging, I/O and all traces, is
805,952,712 bytes under a 2 GiB budget. These are accounted bytes, not a physical
memory peak. The feature input is a deterministic oracle fixture: qualification
of the complete XLSR-to-video pipeline and generation speed remains open.

## Reproduction and provenance

Wan source: `Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff`.
The oracle checks these source hashes before constructing the component:

- `wan/modules/s2v/audio_utils.py`:
  `76931d7291f69a6b6ed3a7a1a9a106ad8b5f5d494a451db67718b4fbcc4dbcfd`
- `wan/modules/s2v/auxi_blocks.py`:
  `6ff7b5560036e5f0eaaa34460a328afcdbd7bd62baf93b699d5f6d2d556f95d6`

The compatibility discriminator uses PyTorch `v2.8.0`
`aten/src/ATen/native/mps/kernels/LayerNorm.metal`, SHA256
`1a289179208970a07e5da56d72c8533adb2549cea49427541c0178ded525d597`,
and its `Normalization.mm` dispatcher, SHA256
`4acdf9b945e74359254383077c10a0e0142cd824a6410d6ad0f4c8c9f32dffce`.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$ORACLE_PYTHON" 150 4096 \
  spec/support/wan_s2v_audio_reference.py \
  --checkpoint "$WAN_CHECKPOINT" --output "$EVIDENCE/causal-audio-real.json"

crystal build spec/wan_s2v_audio_spec.cr -o "$BUILD/wan_s2v_audio_spec" \
  --link-flags="$BUILD/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld"

WAN_S2V_AUDIO_FIXTURE="$EVIDENCE/causal-audio-tiny.json" \
WAN_S2V_AUDIO_PRODUCTION_FIXTURE="$EVIDENCE/causal-audio-real.json" \
COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh "$BUILD/wan_s2v_audio_spec" 150 8192
```

Omit `--checkpoint` to generate the tiny fixture. Local evidence is under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full`:
`causal-audio-final-build.log`, `causal-audio-final-native.log`, and the two
fixtures. Their JSON SHA256 values are
`43f88828b127d5b39834cdab3665ae6db1cc17d5c911623378e95f09d8414d90`
(tiny) and
`925be67555b97b2f8afb4799f1a95342b89d16da8a94c6e3597fcbca3ad38425`
(real). The fixture weight hashes checked by the spec are
`03cae47fa62f9da012ea936d9390220e533d703e130b09aed44d353b230ef420`
and `713354ba7be02b62d1f09ae968f5a68fbdcc50c813179017fb39b050506c94ba`.

All heavy runs were serial, guarded by finite deadlines, process-tree RSS caps,
and a 15% free-memory stop floor. Sampled memory does not prove a continuous
minimum. Refresh this gate after native/oracle/upstream/checkpoint/shape/
arithmetic/runtime/device changes, or missing/changed evidence. Hashes are not
transitive build attestations.
