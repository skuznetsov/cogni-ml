# Native speech-to-video frontier

Status: IN_PROGRESS; first resident block slices, not a video pipeline.

## Scope and source anchors

Implement opt-in native Metal backends for Wan 2.2 S2V and LTX 2.5.
The eventual comparison uses one Qwen Image portrait and one DRAMA3 WAV for a
5–10 second speaking, laughing, head-turning scene. No speed or quality winner
has been measured.

- Wan upstream: `Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff`.
- LTX upstream: `Lightricks/LTX-2@e98f6d927d83c92ef2be6aec76c8440b19306e39`.
- Base engine: `6263c34d82bc733cf4d861531dde2e47a7a99e13`.
- Qwen Image buffer-level projections, Metal dispatch, and CogniGraph lifetime
  machinery exist; their transformer topology is not either video topology.

## Current boundary

Design-sealed: F32 resident block probes with preallocated scratch, caller-owned
serial command encoding, GPU output chaining, and terminal diagnostic reads.
Wan's audio encoder output and both models' conditioning may be fixed block
inputs. This does not admit native audio/text encoders or full checkpoint use.

The initial Wan fixture uses two distinct batch members, C=32, two attention
heads, two video frames with four spatial tokens each, and a four-token motion
suffix. Audio injection sees five local tokens per frame. Q/K RMSNorm is over
full C before head reshaping.
The native slice admits fixed, unpadded sequence/context lengths shared across
the batch; upstream per-example lengths and attention masks are not implemented.
The composite block/audio call requires the same video prefix in both configs.
The MPS oracle executes the pinned upstream modules with a declared SDPA adapter
for their CUDA-only attention entry points and F32 application of upstream FP64
RoPE coefficients prepared at setup. Its positions include video coordinates up
to 256/512/1023 and a negative-time motion suffix. It does not certify CUDA flash
numerics or production head dimensions.

The initial LTX fixture is a batch-one dual-stream block with video C=32 and
audio C=16. It exercises self/text attention, both AV cross-attention directions,
and both feed-forward paths. `cross_attention_adaln=false` and
`apply_gated_attention=false` are explicit fixture choices, not a claim that the
production LTX 2.5 checkpoint uses those settings. AdaLN values and split-RoPE
tables are supplied as resident inputs; their native embedding producers remain
open.

Rejected until separately evidenced: a complete video-generation API, whole
checkpoint residency, BF16/quantized parity, CPU neural fallback, hidden weight
offload, production throughput claims, and a claim of successful lip sync.

## Design laws and risk

CAUTION: new GPU math and buffer-lifetime contracts. Rollback is dropping the
opt-in modules/new worktree; existing model routes and defaults remain intact.
All neural math in an admitted block runs in Metal. Host work is limited to
setup, command encoding, I/O, and terminal verification. No per-op waits,
activation downloads, or scratch allocation. Encoders are serial. Input and
output aliasing must be rejected where kernels do not support it; buffers must
outlive the caller's final command completion.

LTX-specific implementation and adapted fixtures are segregated and carry the
LTX-2.x Community License. Shared independent primitives keep the repository's
existing license. Wan adaptations retain Apache-2.0 notices. Source and weight
license conditions remain distinct from the ability to execute a block.

## Implementation order and falsifiers

1. Shared GPU linear, normalization, modulation, activation, residual, and
   noncausal attention; guard shapes and unsafe aliases before dispatch.
2. Wan block 0, reference/noise modulation and first audio injector; compare
   intermediate values with pinned upstream using identical inputs/weights.
3. LTX dual-stream block and cross-modal attention; same intermediate gate.
4. Attack changed inputs, scratch reuse, device-output chaining, invalid shapes,
   audio perturbations, and a deliberately corrupted reference value.
5. Only after those gates: checkpoint loading and missing encoders/VAEs,
   end-to-end denoising, then controlled full-scene speed/memory comparisons.

The initial F32 oracle gate is fixed before numerical execution:
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)` for every finite value.
This does not change any existing Qwen/DRAMA tolerances. RoPE must be compared
against upstream FP64 rotations at realistic positions before checkpoint
admission; a tiny-position pass is insufficient.

DoD commands (run serially with `COGNI_RUN_SAFE_MIN_FREE_PCT=15`): focused
`spec/video_metal_ops_spec.cr`, `spec/wan_s2v_block_spec.cr`, and
`spec/ltx2_block_spec.cr`, then pinned upstream fixture generation and native
intermediate comparison. Expected signals: all checkpoint values within the
declared gate, perturbation detected, no GPU-buffer allocation during repeated
encoding, and output chaining without an intermediate host wait/read.

Stop rules: 15% free-system-memory floor, bounded RSS and finite timeout; no
full model or Studio startup until resource planning passes. Large scratch,
compiler caches, source snapshots, and generated fixtures live under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/`.

## Reproducing the block gates

The oracle scripts execute external pinned upstream source, not a retyped CPU
formula. A macOS MPS build of PyTorch and its upstream Python dependencies are
required. The measured environment used PyTorch 2.8.0; no package installation
or modification of the existing oracle environment was performed for this work.
The Wan adapter expects the following files under `wan-audit/`, flattened with
`/` replaced by `_`: `wan/modules/model.py`, `wan/modules/s2v/model_s2v.py`,
`wan/modules/s2v/audio_utils.py`, `wan/modules/s2v/s2v_utils.py`, and
`wan/modules/s2v/auxi_blocks.py`. It also requires the exact Diffusers v0.31.0
`src/diffusers/models/normalization.py`, saved as
`diffusers-normalization-v0.31.0.py`. The LTX source root is its pinned
`packages/ltx-core/src/` directory. Both generators reject changed source hashes.

With those source snapshots prepared and `oracle_python` set to an existing
MPS-capable Python executable, generate the fixtures serially:

```sh
video_tmp=/opt/mldata_ssd/cogni-ml/video-scene-20261019
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
export COGNI_RUN_SAFE_MIN_FREE_PCT=15
scripts/run_safe.sh "$oracle_python" 120 4096 \
  spec/support/wan_s2v_reference.py --upstream "$video_tmp/wan-audit" \
  --output "$video_tmp/wan-audit/wan-s2v-mps.json"
scripts/run_safe.sh "$oracle_python" 120 4096 \
  spec/support/ltx2_reference.py \
  --source-root "$video_tmp/ltx-audit/source/packages/ltx-core/src" \
  --output "$video_tmp/ltx-audit/ltx2-dual-stream-mps.json" --device mps
```

Build the shared bridge into the external directory, then compile each focused
spec before running it under the memory/deadline guard:

```sh
make BUILD_DIR="$video_tmp/build" "$video_tmp/build/bridge.o"
export CRYSTAL_CACHE_DIR=/opt/mldata_ssd/cogni-ml/crystal-cache
for unit in video_metal_ops wan_s2v_block ltx2_block; do
  crystal build "spec/${unit}_spec.cr" -o "$video_tmp/build/${unit}_spec" \
    --link-flags="$video_tmp/build/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld"
done
export WAN_S2V_BLOCK_FIXTURE="$video_tmp/wan-audit/wan-s2v-mps.json"
export LTX2_BLOCK_FIXTURE="$video_tmp/ltx-audit/ltx2-dual-stream-mps.json"
for unit in video_metal_ops wan_s2v_block ltx2_block; do
  scripts/run_safe.sh "$video_tmp/build/${unit}_spec" 60 2048 --verbose
done
```

Missing fixtures produce pending oracle specs during ordinary repository tests;
pending results are not a parity pass. The declared gate requires the fixture
environment variables and zero pending results. The test allocation counter is
cumulative, so temporary GPU buffers cannot hide behind an unchanged live count.
Diagnostic checkpoint copies and terminal reads are probe overhead, not a
production performance measurement.

## Observed block admission, October 19, 2026

On Apple M2 Max, the combined binary requiring all three focused specs finished
with **16 examples, 0 failures, 0 errors, 0 pending**. Both pinned MPS fixture
generators completed with CPU neural fallback disabled. The unchanged F32 gate
passed all 35 Wan block/audio checkpoints at B=2 and 64 LTX checkpoint mappings
across two input cases and two chained blocks. This admits only the reduced
configurations above. It does not admit production checkpoint flags or sizes.

Repeated encoding kept the cumulative GPU-buffer allocation count unchanged.
Wan block-to-audio and LTX block-to-block outputs stayed resident in one serial
encoder, followed by one terminal completion/read phase. Guard probes rejected
unsafe backing-range aliases and mismatched Wan video-prefix contracts; seeded
input changes affected the expected streams, and corrupted reference values
were rejected. AdaIN also passed a separate C=320 coverage probe.

The large-position gate exposed an omitted head divisor in Wan RoPE indexing,
then ordinary Metal trigonometric error despite safe math mode. At angle 801,
an isolated GPU probe measured sine error 5.325852e-5 versus 1.773396e-9 for
`precise::sin`. The measured trig errors predict the failing rotated component
within 3e-8. The admitted kernel corrects the indexing and uses precise sine
and cosine; neither the positions nor the tolerance were reduced.

Private source/binary/fixture/log pins and the executed result are indexed in
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/final-block-gates.json`
(SHA256 `40be04204a58d9b00e3b1f7f2f6a838a56000ff807c7145877fe5dd675d3279c`).
The final native log is `evidence/video-all-spec.log` under that same external
root. Source and binary pins are not a transitive reproducible-build attestation.
The run used a configured 15% memory floor, finite timeouts, and tree-RSS limits;
the logs do not establish an observed minimum free-memory percentage. No full
model or Studio was started and no throughput measurement was made.
The correlated mini-Quadrumvirate review returned ROBUST within these tested
resident-block fixtures; it does not extend that verdict to the open obligations.

## Open obligations

No complete model execution has been performed. Wan FP64 RoPE, audio feature
extraction, per-example attention lengths/masks, temporal packing, both video
VAEs, LTX conditioning/upsampling, and whole-request resource limits remain open.
Revalidate source-specific evidence
when its pinned upstream/configuration or native kernels change.
