# Hy-Nomas native Metal frontier

Document status: implemented F32-math core; bounded verification below (2026-10-24).
Current frontier: attention precision and real checkpoint validation, not a video pipeline.
Upstream: Tencent-Hunyuan/Hy-Nomas at
`911cea6458a89c2663513ff7099561ec190dd491` (Apache-2.0 source).

## Scope and design laws

The user has temporarily redirected video porting to Hy-Nomas. The immediate
slice is one native dual-stream DiT block on a single segment. Text/video
modulation, Q/K normalization, video RoPE, joint attention and feed-forward must
follow the pinned upstream execution order. Camera conditioning is a separate
post-block module and remains outside this first slice. Unsupported
history/mask/conditioning modes must fail at admission rather than silently
execute an unmasked substitute.

GPU command encoding must consume resident inputs and reusable scratch. No
activation readbacks, inter-operation host waits or GPU-buffer allocations in
the block's encode method are admitted. A diagnostic runner may perform its
existing terminal command wait and read all saved intermediate buffers after
completion. Setup may upload inputs, weights and positional constants. Neural
operations, including any source-dtype conversion, execute on GPU.

Use existing native Metal/MPS operations and strict checkpoint admission.
Checkpoint bindings must validate names, shapes, dtype and memory budget before
allocating. F32 controls and original BF16 residency are separate representations;
INT8/FP8 support is not implied by BF16/F32 admission. Source-file I/O and GPU
uploads at setup are not a CPU inference/offload path.
Use `HyNomasDualStreamBlock.weight_plan` for model-specific admission before
generic preparation. Checkpoint bytes and headers must be trusted and immutable
for the lifetime of the plan and prepared weights. The loader's metadata checks
detect ordinary file replacement; they are not an adversarial content-integrity
guarantee against in-place rewrites that restore file metadata.

The finite numerical gate remains `abs(error) <= 2e-5 + 2e-5*abs(reference)`.
Do not relax it after observing a mismatch. The oracle must invoke pinned
upstream code; any CUDA-only adapter must be enumerated. Oracle time is excluded
from native measurements. Synthetic fixtures establish parity with the declared
F32 math/order control, not video quality or real checkpoint fidelity. The MPS
adapter keeps Q/K/V, attention and its output in F32. Pinned Magi attention
instead casts F32 Q/K/V to BF16 and its result back to the original dtype. This
first slice does not claim numeric parity with that CUDA BF16 runtime, even when
the source weights are stored as BF16. Native BF16 activation/attention parity is
a separate open gate. Normalization epsilon is fixed to upstream's `1e-6`.

## Rejected and guard-only surface

Full model/Studio/video generation remains disabled for this slice. Audio-driven
video, lip-sync, trained reference-image conditioning, diffusion scheduling,
PixelUMM decoding, text tokenization/encoding, camera/action modules,
cross-segment KV-cache behavior,
INT8/FP8 inference and model-wide performance are unimplemented boundaries.
Public RTX 4090 timings are author-reported CUDA results, not Metal predictions.
No GPL community-port code is imported into this Apache-source implementation.

## Verification plan and rollback

Risk: CAUTION (new inference implementation); isolated branch
`codex/hy-nomas-native`, based on Wan commit `e22fc2ba`.
Existing work and inference defaults remain untouched. Rollback is omission of
the new opt-in files; do not reset unrelated work.

1. Audit exact upstream block equations, conditioning and state-dict names.
   Consumers: new `src/ml/video/hy_nomas_block.cr`, its kernels and tests only.
   Inventory: `rg --files src/ml/video spec examples`; no Hy-Nomas implementation
   exists at the starting revision.
2. Add a pinned-source upstream fixture producer and native specs that first
   fail without the new block. Include nonzero modulation, both streams,
   invalid shapes/configurations and resident-weight types. Admit only one
   fully valid unmasked segment with no history, camera or action windows.
3. Implement only the new block and checkpoint bindings, reusing shared engine
   APIs. Preallocate diagnostic scratch. Keep an explicit unsupported-mode gate.
4. Run bounded upstream/native parity and repeated native encode checks under
   the existing cache-aware memory guard. Record every intermediate mismatch,
   terminal wait boundary, memory peak and scope of any timing claim.

DoD: generate bounded upstream fixtures on MPS; compile and run the focused
Crystal spec with fresh Metal/MPS bridge objects; check format and the exact diff.
Expected signal: no nonfinite values or numerical violations, strict negative
admission tests pass, and no model-path synchronization/readback/allocation.

The strongest failure modes are reordered text/video tokens or RoPE channels,
incorrect modulation/gates, BF16 dtype confusion and silently missing temporal
masks. Intermediate upstream
checkpoints and deliberate invalid-mode tests are the discriminators.

All runtime probes use a 15% physical-memory reserve/floor, cache-aware admission,
a bounded process-group footprint, a swap-growth stop and a deadline. Preserve
guard failure as an incomplete measurement, never evidence of acceleration.
Temporary fixtures, objects and logs belong under `/opt/mldata_ssd/cogni-ml/`.

The external oracle JSON is trusted test input. Capture its SHA-256 immediately
after trusted fresh generation and pass it separately in
`HY_NOMAS_FIXTURE_SHA256`. Do not take this digest from mutable metadata inside
the fixture. The native harness hashes one read of the JSON and parses those
same bytes; the supplied digest also binds expected tensors and sidecar hashes.
Checkpoint bytes remain subject to the immutable-file precondition above.

## Reproduction commands

Run from the feature checkout on macOS Metal. Use an existing PyTorch MPS
environment with safetensors; the environment used here is the external
DRAMA-reference venv. It is an oracle dependency, not part of native execution.
Choose a fresh `probe_root`: the oracle refuses to replace existing outputs.

```sh
probe_root=/opt/mldata_ssd/cogni-ml/hy-nomas-native-repro
probe_upstream=/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024
probe_python=/opt/mldata_ssd/cogni-ml/drama3-fish-reference-20261015/venv/bin/python
mkdir -p "$probe_root"/build "$probe_root"/tmp "$probe_root"/crystal-cache "$probe_root"/fixtures "$probe_root"/logs

guarded() {
  probe_log=$1
  shift
  python3 examples/support/wan_memory_guard.py \
    --max-footprint-mib 4096 --max-swap-growth-mib 64 --timeout-sec 600 \
    --memory-accounting file-cache --log "$probe_root/logs/$probe_log.jsonl" -- "$@"
}

guarded oracle-tiny env TMPDIR="$probe_root/tmp" PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$probe_python" spec/support/hy_nomas_reference.py \
  --upstream "$probe_upstream" --output "$probe_root/fixtures/tiny.json"
guarded oracle-width env TMPDIR="$probe_root/tmp" PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$probe_python" spec/support/hy_nomas_reference.py \
  --upstream "$probe_upstream" --output "$probe_root/fixtures/width.json" --real-width
shasum -a 256 "$probe_root/fixtures/tiny.json" "$probe_root/fixtures/width.json"

guarded bridges make BUILD_DIR="$probe_root/build" \
  "$probe_root/build/bridge.o" "$probe_root/build/metal_mps_command_chain.o" "$probe_root/build/metal_stage_io.o"
probe_link="$probe_root/build/bridge.o $probe_root/build/metal_mps_command_chain.o $probe_root/build/metal_stage_io.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -lc++ -fuse-ld=/usr/bin/ld"
guarded build env TMPDIR="$probe_root/tmp" CRYSTAL_CACHE_DIR="$probe_root/crystal-cache" \
  crystal build spec/hy_nomas_block_spec.cr -o "$probe_root/build/hy_nomas_block_spec" \
  --error-trace --link-flags="$probe_link"
```

For each of `tiny` and `width`, set `probe_fixture_digest` to its corresponding
digest printed immediately after generation, then run:

```sh
probe_case=tiny
probe_fixture_digest=REPLACE_WITH_TRUSTED_GENERATION_DIGEST
guarded "native-$probe_case" env TMPDIR="$probe_root/tmp" \
  HY_NOMAS_BLOCK_FIXTURE="$probe_root/fixtures/$probe_case.json" \
  HY_NOMAS_FIXTURE_SHA256="$probe_fixture_digest" \
  HY_NOMAS_REPORT="$probe_root/logs/$probe_case-comparisons.txt" \
  "$probe_root/build/hy_nomas_block_spec"
crystal tool format --check src/ml/video/hy_nomas_block.cr spec/hy_nomas_block_spec.cr
git diff --check
```

Without an external fixture, the parity example is pending; a green local
unit-only run cannot establish upstream-control parity. Do not combine the two
GPU runs or run a model alongside them merely because each has its own guard.

## Bounded result, 2026-10-24

The focused spec passes **6 examples, 0 failures/errors/pending** for each
geometry. Both have 6 video and 3 text tokens. The broadcast case has B=2;
the per-token modulation case has B=1. They use synthetic nonzero weights,
not a downloaded checkpoint.

| Geometry | C / heads / head dim / FFN | Checkpoint comparisons | Compared values | Maximum absolute error | Maximum gate ratio |
|---|---|---:|---:|---:|---:|
| Tiny | 64 / 2 / 32 / 256 | 444 | 282,240 | 9.536743e-7 | 0.028442943 |
| Nomas-G widths | 2048 / 16 / 128 / 8192 | 444 | 9,031,680 | 5.9604645e-6 | 0.1444688 |

Each geometry compares 37 named intermediates in the first block, a second
application consuming the first block's resident outputs, and a seeded
video-perturbed application, for both F32 and BF16 weight storage in both
modulation cases. A repeated first encode verifies scratch overwrite/reuse.
Video perturbation must change both stream outputs; the block cannot pass as
an identity. Shape/mode/dtype/budget/alias gates, nonfinite/corruption detection,
and the trusted-fixture-digest negative control pass. The BF16 control rounds
weights to BF16, then uses F32 neural math in both implementations.

All four native encodes plus preparation share one command/encoder per variant,
with one terminal wait and subsequent diagnostic reads. The tracked encode
MetalBuffer allocation delta is zero. There is no inter-operation activation
readback or explicit completion wait in the admitted block. The native measured
path has no PyTorch dependency.

At Nomas-G widths the resident block weights occupy 604,227,584 bytes as F32 or
302,113,792 bytes as BF16. Three diagnostic scratch sets occupy 15,435,264 bytes
for broadcast or 8,922,624 bytes for per-token modulation. These numbers do not
include the rest of the model, encoder, decoder or video-length activations.

Diagnostic GPU command spans on Apple M2 Max were 58.516/40.279 ms for F32 and
154.233/116.229 ms for BF16 (broadcast/per-token). These include **four encodes
and weight preparation** and are single observations from a correctness build,
with only nine tokens per item. They are neither per-block steady-state
benchmarks nor video-speed estimates. The shared baseline linear kernel uses
a serial F32 K reduction; no MPS GEMM speedup has been established. BF16's
smaller residency does not imply a faster typed-load kernel in this probe.

Final guarded oracle runs sampled at most 3,527.395 MiB process-group footprint;
final native runs sampled at most 2,096.705 MiB. Minimum cache-aware headroom
was 25.385% for the oracle and 27.872% for native. All final runs exited zero
with no swap growth. The 4 GiB sampled cap, 15% floor, 64 MiB swap-growth stop
and 600 s deadline were retained. Sampled peaks can miss between-sample spikes.
No full model, Studio or video generation was run.

Correlated MiniQuad/adversary review is **ROBUST within the declared F32-control
scope**, assuming trusted fixture digests and immutable checkpoint files. It
does not certify Magi BF16 runtime parity, quality, full-model fit or speed.
The review caught the CUDA BF16 attention distinction and a configurable-epsilon
admission expansion; the latter now rejects non-default epsilon before
allocation. The fixture mutation control closes the inspected false-pass path.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-native-20261024/`.
Final receipts are `logs/oracle-{tiny,real-width}-guard-v2.jsonl`,
`logs/native-tiny-guard-v4.jsonl`, `logs/native-real-width-guard-v2.jsonl`,
`logs/tiny-comparisons-v4.txt` and `logs/real-width-comparisons-v2.txt`.
Build receipt: `logs/build-parity-guard-v5.jsonl`; source is this feature commit.
Toolchain: Crystal 1.22.0, Apple clang 21.0.0, macOS 26.6.2, SDK 27.0;
oracle PyTorch 2.8.0 MPS with fallback disabled. Format and exact-diff checks pass.

SHA-256 pins:

| Artifact relative to evidence root | SHA-256 |
|---|---|
| `fixtures/tiny-v2.json` | `51849dfabe68280ab186e4a13594261fbae45ae39d38a165cee7a133cc68b978` |
| `fixtures/real-width-v2.json` | `13b50759bb75784b40eaec75ca2b8952020807f64fc53a60b7b05ac5789acd71` |
| `build/hy_nomas_block_spec` | `92b6f085ed9bdc1dd8d7d4b4ea383320eba3e68e751d28f5f66716ba0361a7a5` |
| `build/bridge.o` | `16613762decfabc1d44d6615ef0af95986fdccafc07ad93d9f007c5440a055f3` |
| `build/metal_mps_command_chain.o` | `c2ba6e89f532848235e6710eb36e245a117665d6f6b5d0fbbba5656a2f5e0a86` |
| `build/metal_stage_io.o` | `7ee1def8e387edea8213c848441d40ac18ee9ec69f398423dce197d78244370a` |

The oracle's per-source and sidecar-weight pins are retained in each fixture's
manifest. Regenerate rather than edit reference arrays after a source change.

## Continuation

Active goal: native Hy-Nomas port, initially one proven DiT block. Return pointer:
Wan performance work remains at `docs/wan-performance-frontier.md`, commit
`e22fc2ba`; no Wan default has been changed. Next discriminator: model-native
BF16 attention boundaries against pinned Magi semantics, followed by actual
checkpoint weights. Then measure MPS GEMM/attention at representative token
counts before constructing full-model residency, camera/history and PixelUMM.
Refresh evidence after source, fixture, geometry, numerical policy, compiler,
native bridge, SDK/OS/device or command-scheduling changes.
