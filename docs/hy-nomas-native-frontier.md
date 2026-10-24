# Hy-Nomas native Metal frontier

Document status: actual-checkpoint F32-math and BF16-boundary MPS block controls
verified within the bounded scope below (2026-10-24). The opt-in prepared MPS
backend matches the pinned MPS oracle, including chained output reuse. Historical
failures of the original Metal path remain recorded. CUDA/Magi parity, target
geometry performance and the video pipeline remain open.
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
instead casts F32 Q/K/V to BF16 and converts the returned result to the original
dtype when necessary. The wrapper alone does not establish the kernel's output
dtype. This
first slice does not claim numeric parity with that CUDA BF16 runtime, even when
the source weights are stored as BF16. The new `encode_mps` control certifies the
declared MPS BF16 boundary policy, not full BF16 activation execution or CUDA
FlashAttention reductions. Normalization epsilon is fixed to upstream's `1e-6`.

## Rejected and guard-only surface

Full model/Studio/video generation remains disabled for this slice. Audio-driven
video, lip-sync, trained reference-image conditioning, diffusion scheduling,
PixelUMM decoding, text tokenization/encoding, camera/action modules,
cross-segment KV-cache behavior,
INT8/FP8 inference and model-wide performance are unimplemented boundaries.
Public RTX 4090 timings are author-reported CUDA results, not Metal predictions.
No GPL community-port code is imported into this Apache-source implementation.

## Active slice: BF16 boundaries and checkpoint weights

Status: actual-weight F32 and BF16-boundary controls pass through the opt-in MPS
backend, 2026-10-24. The original native backend's BF16 failures remain open;
preserve its existing F32-control default.
An explicit opt-in BF16-boundary control rounds concatenated F32 Q/K/V to
BF16 with round-to-nearest-even, compute attention in F32, then round its output
to BF16 before the F32 projection. The output round-trip is an explicit control
assumption until the exact Magi kernel is executed or its contract is evidenced.
Fuse these conversions into existing packing
and splitting dispatches where possible. This emulates pinned Magi storage
boundaries; it does not emulate or certify CUDA FlashAttention's internal
reduction or BF16-activation execution throughout the block.

Falsifier: exact same-input boundary tests must expose a missing conversion or
incorrect tie rounding. Full-block intermediate comparisons retain the finite
`2e-5 + 2e-5*abs(reference)` gate. If upstream/native F32 differences cross a
BF16 rounding midpoint, retain that mismatch and isolate attention on identical
Q/K/V; do not increase tolerances or silently label the boundary path exact.

For real weights, enumerate the official distilled checkpoint at an immutable
Hugging Face revision and materialize only the 32 tensors of `double_blocks.0`.
Validate exact keys, shapes, source dtypes, byte ranges and source identity before
publishing a local subset. Keep F32 source weights and a GPU-converted BF16
residency control distinct. No full-checkpoint, full-model or decoder allocation
is admitted by this slice. Setup file extraction is not neural CPU compute.

The generic checkpoint planner retains its default rejection of F32 sources in
BF16-resident mode. An explicit `allow_f32_to_bf16` setup option may admit only
F32-to-BF16 conversion, with raw F32 staging included in the pre-allocation
budget. Conversion uses a GPU rounding kernel in the caller's existing encoder;
original BF16 tensors continue to copy directly to their final buffers. F16
sources remain rejected in BF16-resident mode even when the option is enabled.

DoD: guarded focused specs establish deterministic rounding on crafted values,
unchanged F32-control results, upstream comparisons for the boundary variant,
and actual one-block weights against the pinned class. Report any failed gate
as an open frontier. One terminal completion/readback per diagnostic variant,
zero encode-time device allocations, the 15% floor and existing process/swap
budgets remain mandatory. Rollback is removal of the new opt-in mode and probe;
the existing F32-control API remains the baseline.

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
  "$probe_root/build/bridge.o" "$probe_root/build/metal_mps_command_chain.o" "$probe_root/build/metal_stage_io.o" \
  "$probe_root/build/wan_t5_mps_bridge.o" "$probe_root/build/wan_t5_rms_mps_bridge.o" \
  "$probe_root/build/hy_nomas_mps_reductions.o"
probe_link="$probe_root/build/bridge.o $probe_root/build/metal_mps_command_chain.o $probe_root/build/metal_stage_io.o $probe_root/build/wan_t5_mps_bridge.o $probe_root/build/wan_t5_rms_mps_bridge.o $probe_root/build/hy_nomas_mps_reductions.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -lc++ -fuse-ld=/usr/bin/ld"
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

## Historical synthetic result, commit 840c3a58

At the previous feature commit, the focused spec passed
**6 examples, 0 failures/errors/pending** for each
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

## Actual-checkpoint and boundary-control result, 2026-10-24

The official distilled safetensors file at Hugging Face revision
`bac719a80c8d02f91c485fca804bb8221e149b70`, path
`DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors`, stores F32 weights.
The extractor validates the pinned metadata, bounded complete header, exact
32-key block inventory, shapes/dtypes and contiguous nonoverlapping ranges.
It streams only the block's 604,227,584 payload bytes via an exact HTTP 206 range
`968270320-1572497903`, then publishes a fresh subset and provenance manifest.
There is no tensor decoding or CPU weight conversion during extraction.
The resulting block has 151,056,896 parameters. Its F32 residency is
604,227,584 bytes and GPU-rounded BF16 residency is 302,113,792 bytes.

Source authentication relies on HTTPS at the immutable repository revision.
The declared LFS source hash is
`73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70`;
the complete 34,213,991,280-byte source file was not downloaded or independently
hashed. The observed block payload hash is
`4711e53c3b9ab29abe552f25b95d3cea6f9d75df74e098d7a6acc880befae28b`.
The independently checked local subset-file digest is pinned below. These are
different scopes of evidence; a matching local digest does not verify the
unfetched parts of the upstream checkpoint.

The oracle loads these exact 32 tensors into the pinned upstream class at
C2048/H16/D128/F8192. Both cases still use 6 video and 3 text tokens; this is
real width and actual weights, not representative video sequence length.
F32 storage and GPU-rounded BF16 storage each use F32 neural arithmetic.

| Fixture / mode | Spec result | Comparison records | Failed records | Max absolute error | Max gate ratio |
|---|---|---:|---:|---:|---:|
| Tiny synthetic, F32 math | 8 examples, 0 failures/errors/pending | 488 | 0 | 9.536743e-7 | 0.028442943 |
| Actual block, F32 math | 8 examples, 0 failures/errors/pending | 488 | 0 | 2.4795532e-5 | 0.8565121 |
| Tiny synthetic, BF16 boundaries | 7 examples, 1 numerical failure | 488 | 7 | 0.00048828125 | 22.381378 |
| Actual block, BF16 boundaries | 8 examples, 1 numerical failure | 488 | 289 | 0.125 | 529.8862 |

Each fixture covers four variants: broadcast B=2 and per-token modulation B=1,
each with F32 and BF16 resident weights. Each variant compares 40 named
intermediates for the first, chained and video-perturbed applications, plus
raw and post-boundary attention replays on identical upstream Q/K/V:
`4 * (40 * 3 + 2) = 488`. A repeated first encode checks scratch overwrite.
The tiny BF16 result preceded the additional source-mode unit test; its
488 comparison records use the same gates. A failed comparison record may
contain many violating values; counts here are not counts of bad elements.

Both final F32-control fixtures pass and retain zero tracked encode-time
MetalBuffer allocations. Original BF16 sources retain direct-copy preparation;
actual F32 sources require 604,227,584 bytes of staging for either residency
control. The conversion, four block encodes and attention replay share one
diagnostic command with one terminal completion/readback per variant. No new
production inter-operation wait or readback is added. The diagnostic command
times include all this work and are not per-block benchmarks or video estimates.
The separate checkpoint suite passes 20 examples, including exact rounding ties,
mixed source residency, conversion chaining and pre-allocation budget failures.
The extractor's 11 offline tests and oracle structure/device self-tests pass.

### BF16 failure is retained

Actual-weight BF16-boundary failures per variant are 81/76/75/57 records
(broadcast BF16/F32 residency, then per-token BF16/F32 residency). The first
broadcast/BF16-resident application's upstream/native F32 projections,
normalization and RoPE pass the original gate, while the rounded joint Q/K/V
do not. One captured query value illustrates the discontinuity:

```text
native before round:    0.7675781   (F32 bits 3f448000)
upstream before round:  0.7675783   (F32 bits 3f448003)
native after round:     0.765625
upstream after round:   0.76953125
```

This is an F32 difference crossing a BF16 midpoint. The literal tie tests pass;
changing rounding or removing the boundary would conceal the tested behavior.
Same-input raw F32 attention replay passes all four actual-weight variants
(maximum absolute error 1.4305115e-6). After the output boundary, the
broadcast/F32-resident replay has one violating element, index 10120:
`-0.140625` versus `-0.13964844`, delta `0.0009765625`. The other three rounded
replays pass. Thus implementation numerical differences exist inside attention
as well as before Q/K/V packing; the replay alone does not identify their local
cause. The isolated diagnostics below narrow that attribution. Chained block
and final output gates also fail; these cannot be dismissed as harmless tiny
intermediate differences. No generated
video or quality impact has been tested. BF16-boundary mode stays experimental
and default-off; no Magi/CUDA runtime parity has been established.

### Memory, provenance and evidence

Two early actual-weight oracle attempts hit the unchanged sampled 4 GiB cap
(4106.488 and 4153.456 MiB, guard exit 76); neither is a successful measurement.
Meta initialization and per-parameter GPU BF16 rounding remove duplicate retained
weight copies. A third attempt exposed the `mps` versus `mps:0` device guard
distinction and exited 1; the corrected guard accepts an unspecified MPS index
but rejects CPU or a mismatched explicit index. Its device self-test uses no GPU.

Final oracle F32/BF16-boundary sampled peaks are 3074.222/3508.597 MiB, with
minimum cache-aware headroom 27.328%/25.205%. Final native actual-weight
F32/BF16-boundary peaks are 2789.034/2772.065 MiB, with minimum headroom
23.157%/24.391%. All have zero observed swap growth. The native BF16 run exits 1
for its numerical gate, not for a memory guard stop. The 15% reserve/floor,
4 GiB sampled cap, 64 MiB swap-growth stop and 600 s deadline remain unchanged.
Sampling can miss between-sample peaks. No full model, Studio or video runs.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024/`.
Final receipts: `logs/oracle-real-f32-v4.jsonl`,
`logs/oracle-real-bf16-v1.jsonl`, `logs/native-{tiny,real}-f32-v6.jsonl`,
`logs/native-real-bf16-v6.jsonl`, `logs/native-tiny-bf16-v5.jsonl`,
their matching `*-comparisons.txt` reports and `logs/checkpoint-v3.jsonl`.
Extraction: `logs/extract-block0-v2.jsonl`. Final build receipts:
`logs/build-hy-v6.jsonl`, `logs/build-checkpoint-v3.jsonl`.
The previous evidence root's three bridge objects are reused unchanged with
the SHA-256 pins in the historical table above. Toolchain and device are unchanged.

| Artifact relative to current evidence root | SHA-256 |
|---|---|
| `weights/distilled-block0-f32.safetensors` | `6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1` |
| `fixtures/tiny-f32-v2.json` | `6c2a31852f2e39a7f20017040dcd54ccbbb6043d80b417bf90099bbe7fda4afa` |
| `fixtures/tiny-bf16-v1.json` | `3e1b9d014d02d71a1b38d4a9ea4eccd1a8f28f3824500d4f6f7ac691f67de0d6` |
| `fixtures/real-f32-v4.json` | `f889e7b7726285e8baf0054a7f23652d7004351ef12fd9ab23a72e79a2596f25` |
| `fixtures/real-bf16-v1.json` | `1fb14da66820ca061a87ec28024881de3165301d3993adb944cb45f275bc72e1` |
| `build/hy_nomas_block_spec` | `512f92dca5f75f9049851dfa33e3aef6576eeba63a871549c072e2d310214a6e` |
| `build/video_checkpoint_spec` | `ae8c2b1518814e763fa391e2b20626a7066b34af2d4200d6b1e21591327bf6bf` |

Correlated MiniQuad/adversary review is ROBUST for actual-checkpoint loading
and F32-math control under the trusted producer/digest and immutable-file
preconditions. It is not an independent CUDA comparison. The BF16-boundary
parity gate remains failed. The review added an explicit source-mode gate:
actual-weight runs must supply `HY_NOMAS_CHECKPOINT_SHA256` separately from
`HY_NOMAS_FIXTURE_SHA256`; a generic synthetic fixture cannot satisfy it.
Metadata presence alone does not establish provenance.

### Reproduce the actual-weight slice

Reuse the `guarded` function, bridge build and link flags above, but use a fresh
directory **under** `/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024/` for
`probe_root` (the extractor deliberately restricts its output to that root).
Create its `weights`, `fixtures`, `logs`, `tmp`, `build` and `crystal-cache`
directories. Do not overwrite existing fixtures or receipts.

```sh
guarded extract python3 spec/support/hy_nomas_checkpoint_subset.py \
  "$probe_root/weights/block0.safetensors" --max-payload-bytes 604227584
shasum -a 256 "$probe_root/weights/block0.safetensors"
probe_checkpoint_digest=REPLACE_WITH_INDEPENDENTLY_CHECKED_SUBSET_DIGEST

guarded oracle-actual-f32 env TMPDIR="$probe_root/tmp" PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$probe_python" spec/support/hy_nomas_reference.py \
  --upstream "$probe_upstream" --output "$probe_root/fixtures/actual-f32.json" \
  --real-width --weights "$probe_root/weights/block0.safetensors" \
  --weights-sha256 "$probe_checkpoint_digest" --stage-telemetry
shasum -a 256 "$probe_root/fixtures/actual-f32.json"
probe_fixture_digest=REPLACE_WITH_TRUSTED_GENERATION_DIGEST
guarded native-actual-f32 env TMPDIR="$probe_root/tmp" \
  HY_NOMAS_BLOCK_FIXTURE="$probe_root/fixtures/actual-f32.json" \
  HY_NOMAS_FIXTURE_SHA256="$probe_fixture_digest" \
  HY_NOMAS_CHECKPOINT_SHA256="$probe_checkpoint_digest" \
  HY_NOMAS_REPORT="$probe_root/logs/actual-f32-comparisons.txt" \
  "$probe_root/build/hy_nomas_block_spec"

guarded build-checkpoint env TMPDIR="$probe_root/tmp" CRYSTAL_CACHE_DIR="$probe_root/crystal-cache" \
  crystal build spec/video_checkpoint_spec.cr -o "$probe_root/build/video_checkpoint_spec" \
  --error-trace --link-flags="$probe_link"
guarded checkpoint env TMPDIR="$probe_root/tmp" "$probe_root/build/video_checkpoint_spec"
PYTHONDONTWRITEBYTECODE=1 python3 spec/hy_nomas_checkpoint_subset_test.py
PYTHONDONTWRITEBYTECODE=1 "$probe_python" spec/support/hy_nomas_reference.py --structure-self-test
PYTHONDONTWRITEBYTECODE=1 "$probe_python" spec/support/hy_nomas_reference.py --device-match-self-test
```

Generate a separate fresh oracle fixture with
`--attention-precision bf16-boundaries` and repeat the native invocation with
that fixture's independently captured digest to reproduce the failing gate.
An exit 1 must remain a failure; do not ignore it in automation or treat the
successful F32 control as certification of this precision mode.

## Pilot-frame performance gate, 2026-10-24

No end-to-end video speed has been measured. The native implementation is one
dual-stream block, not the full sampler/text encoder/PixelUMM pipeline. Its
nine-token correctness fixture must not be extrapolated to video throughput.

At pinned upstream revision `911cea6458a89c2663513ff7099561ec190dd491`,
`configs/experiment/inference.yaml` selects 8 Euler steps, 54 DiT blocks and
guidance scales of 1.0. The default final segment has one DiT branch per step
and no cache-write forward (`hymm/pipeline/causal_ar.py`, guidance selection,
denoise loop and `should_cache_chunk`). A non-final segment or preserved cache
adds one whole-stack cache-write pass. A 24-frame, 24-fps prefix uses 6 latent
frames and, with the preset PixelUMM grid, 22 by 40 spatial tokens: 5,280 video
plus 1,280 text tokens. This grid decodes to 704 by 1280; verify actual output
dimensions rather than labeling the result 720p.

For the user's ceiling of 120 wall seconds per video second:

```text
T_other + sum(block-call times) <= 120 seconds
mean block-call time <= (120 - T_other) / (8 * 54)
```

The zero-overhead ceiling is 277.8 ms per block, or 246.9 ms with the ninth
cache-write pass. `T_other` includes conditioning, scheduler, stack overhead,
PixelUMM decoding and output handling. Exceeding this ceiling at representative
geometry falsifies the target; passing one block is only a necessary condition.
Cold request-to-first-pixel and warmed prefix time need separate measurements.

With J=6,560, D=2,048 and FF=8,192, dense projection arithmetic is
`2*J*(4*D^2 + 2*D*FF)`, plus two batch-row modulation projections
`4*B*D*(6*D)` and QK/PV `4*J^2*D`: 1.012983 TFLOP per block. The 432 calls
therefore need at least 3.647 TFLOP/s sustained if the whole 120-second budget
were available to this arithmetic; conditioning and decoding make the actual
requirement higher. This is a necessary throughput bound, not a speed result.

The current named-checkpoint scratch formula requires 1.682 GiB at this
geometry with the default broadcast modulation, or about 2.03 GiB in the
separate per-token mode, before inputs, weights and opaque runtime workspace.
Default conditioning already projects one row per stream and broadcasts it;
there is no default per-token modulation GEMM redundancy to remove. The 54
BF16 block weight sets alone require about 15.19 GiB, excluding embeddings and
other models. Independent MPS workspaces additionally retain about 404.3 MiB of
conversion/attention/RMS stages each, or about 21.32 GiB over 54 blocks. Independent
named scratch sets would add 90.84 GiB. Both scratch lifetime reuse and shared
serial stages are prerequisites for a whole-stack implementation; shared
staging alone is insufficient. Reusing the expansion stage saves allocation
space but still performs the conversion for each projection call.

A target-shape probe needs a
cache-aware residency preflight and compact attention/scratch before launch;
the existing 4 GiB probe cap and 15% host reserve remain in force. A full pilot
also requires the missing pipeline components. The current request excludes
full generation; a future pilot must be a separately scoped run.

The opt-in MPS backend now replaces scalar-output projection reductions and the
serial per-key attention loop. Its nine-token diagnostic command is still not a
video benchmark. Next candidates are compact grouped/chunked MPS attention,
shared serial weight/RMS stages and liveness-based scratch reuse. At J=6,560,
one F32 `[B,H,J,J]` score-sized tensor is 2.565 GiB; the graph's opaque peak can
exceed this, so a complete dense attention graph plus full block scratch is not
admitted under the 4 GiB probe cap. Reducing this workspace is needed before a
representative full-block measurement. Any attention grouping/chunking must
repeat the same-input and chained BF16 gates: a different graph shape may select
different reductions even though heads are mathematically independent.
Remove repeated conditioning work only where the upstream input invariant proves
the rows are identical. A shared conversion stage saves capacity but still
expands each weight matrix: fourteen linears write about 576 MiB of F32 staging
per block call. This is another candidate after target-shape phase timings.
Quantization or fewer diffusion steps are separate quality-changing experiments;
they are not implied by the present unchanged-tolerance arithmetic work.

## MPS reduction intervention frontier, 2026-10-24 (initial falsifier)

The initial falsifier was a native MPS projection/QK/softmax/PV/SDPA control on the
same trusted frozen operands. The existing failing report remains the anchor:
raw arithmetic passes, but small reduction differences can straddle BF16
midpoints. The intervention must retain the original native report and failed
gates, use unchanged finite and bitwise rounded gates, preserve input identities,
allocate plans and scratch before encoding, and use one terminal wait per case.
The completed intervention is recorded below; no default, weight policy or
full-generation scope changes.

Risk is CAUTION: changing graph rank, layout or materialized targets may select
different MPS arithmetic. Standalone QK and frozen-score softmax must therefore
remain distinct from internal SDPA intermediates. The custom online recurrence
and prescaled paths remain explicitly identified controls. Rollback is omission
of the opt-in test controls. DoD is a fresh guarded tiny and actual-weight
comparison report with per-operation outcomes, unchanged native controls,
zero tracked encode buffer allocations, one terminal wait, and cache-aware
memory remaining above 15%. A parity claim is limited to the operation/case
inventory that actually passes; the diagnostic's original failing exit is
retained until all its comparison records pass.

## Same-input attention diagnostics, 2026-10-24

This test-only slice freezes operands from the preceding trusted upstream
fixture. It separates local arithmetic differences from propagation through
earlier operations. It changes no production kernel, rounding rule, tolerance,
default or command schedule. Risk is CAUTION because an incorrect intervention
could misattribute the failed BF16 gate. The guard is exact operand lineage,
bitwise reference reproduction and a factored-native control. Rollback is
omission of `spec/hy_nomas_attention_probe_spec.cr` and its two support files.

The producer invokes pinned upstream RMSNorm/RoPE and PyTorch 2.8.0 MPS math;
PyTorch is only the oracle. Both oracle and native neural operations run on GPU.
The actual-weight geometry remains C2048/H16/D128/F8192 with **6 video and 3
text tokens**, not a representative video length. Each fixture has the same
four broadcast/per-token and F32/BF16-resident weight variants. BF16 residency
means source F32 to BF16 to F32 compute, with conversions on GPU.

Each variant contains 47 isolated operations: biased, unbiased and explicit-bias
versions of eight projections; Q/K RMSNorm and video RoPE; QK, softmax and PV
under three scaling placements; canonical/explicit MPS attention controls;
and frozen-score/probability interventions. Frozen Q/K/V must be exactly the
BF16-RNE round-trip of the original concatenated upstream operands. Layouts,
scales, weight policy, source mode and independent fixture/checkpoint digests
are checked before GPU work. Corruption controls reject malformed tensors,
changed digests, incorrect layouts/scales, missing variants, source-mode
substitution and broken boundary lineage.

All operations and GPU rounding for a variant share one command with **one
terminal wait**. Readbacks occur after that wait. Inputs remain bitwise unchanged
and encode performs zero tracked MetalBuffer allocations. A factored native
QK plus online reduction is bitwise identical to unchanged production attention
in all four actual and all four tiny variants, validating this intervention
within the tested geometry. These are diagnostic command times, not performance
benchmarks.

### Numerical result and causal limits

The diagnostic DoD is a complete, trusted same-input report with unchanged
finite gates, faithful native controls and retained failing BF16 records. It
does **not** require or establish BF16 parity. Five unit controls pass; the
fixture example completes and intentionally exits 1 for numerical failures.

| Actual-weight check | Observed result |
|---|---|
| Raw F32 comparisons | 188/188 pass; max absolute error `2.861023e-6`, max gate ratio `0.0698829` |
| Reproduction of preceding upstream captures | 68/68 records bitwise identical |
| Factored native versus production attention | 4/4 records bitwise identical |
| Rounded diagnostic records | 101/188 fail bitwise equality; 83 also fail the original finite gate |
| Canonical attention output, after BF16 rounding | Two differing elements across four variants; one fails the finite gate |

The bitwise BF16 diagnostic is stricter than the unchanged finite gate. The
101 records include alternative arithmetic paths and counterfactual rounding
sites; they are **not 101 unique live model-boundary failures**. Tiny controls
have 13/188 failing rounded records. No failed gate is waived.
The 188-record counts cover the 47 named operations per variant. Eight additional
raw/rounded report rows compare the factored and frozen-QK interventions with
canonical attention; those overlapping controls are not included in that count.

On identical projection inputs, upstream `linear(x,w,b)` and
`linear(x,w,None)+b` are bitwise equal across all 32 base projection slots
(442,368 values). Native no-bias projections still differ in low F32 bits,
consistent with the different native FMA-lane reduction and MPS GEMM arithmetic;
bias placement does not explain the tested drift. Among normal projections,
28 V values cross a BF16 bin at the actual Q/K/V boundary, 14 exceeding the
finite post-rounding gate. Other projection rounding probes occur before
Q/K RMSNorm/RoPE or after attention and are counterfactual boundaries.

Same-input final Q/K RMSNorm/RoPE probes have zero BF16 bin differences across
16 records. This does not clear chained Q/K: incoming projection differences
are deliberately removed by these frozen-input tests and can propagate in a
complete block.

Two interventions distinguish the canonical attention failures:

- **Broadcast/F32, index 10120:** native raw output is exactly
  `-0.14013671875`, a BF16 midpoint, and rounds to `-0.140625`. Canonical MPS
  gives approximately `-0.14013669`, rounding to `-0.13964844`. Replacing only
  QK scores with frozen explicit-MPS postscale scores in the same native online
  reducer restores the canonical BF16 value. The entire frozen-QK variant has
  zero BF16 differences from canonical MPS here. QK arithmetic is causally
  sufficient to explain this point under these inputs; RNE itself is correct.
- **Broadcast/BF16 weights, index 9949:** freezing MPS QK does not restore the
  canonical BF16 value. Native PV using probabilities returned by the actual
  MPS math-SDPA operator does restore it. This implicates arithmetic between
  scores and output in addition to QK. It does not isolate softmax alone:
  the private MPS probability control includes its internal QK and softmax.
  Frozen-QK online reduction also introduces other BF16 mismatches in this
  variant, so it is not a general replacement or fix.

Canonical SDPA, explicit postscale attention and private MPS math-SDPA expected
outputs are bitwise equal for all four variants (110,592 values per path).
The standalone frozen-score online reference is a custom recurrence control,
not the canonical upstream attention algorithm. No CUDA/Magi execution,
quality comparison, production fix or speedup is claimed.

Correlated MiniQuad source/fixture/report review is ROBUST for this bounded
diagnostic under the trusted-producer/digest and immutable-weight assumptions,
and VULNERABLE as evidence for a general BF16 fix. Cross-path expected-output
equality was independently checked from the pinned fixture during that review;
the compact report verifier below checks report completion and control counts,
not that additional fixture-array comparison.

The next discriminator is to compare the native projection/QK reductions with
the corresponding MPS GEMM/math path, then isolate normalization and PV with
frozen actual MPS probabilities. Preserve the original F32 control and failed
BF16 gates when evaluating a candidate change.

### Safety and pinned evidence

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-attention-20261024/`.
The same 15% physical reserve/floor, cache-aware file-cache accounting, sampled
4 GiB process-group cap, 64 MiB swap-growth stop and 600 s deadline are retained.
Jobs ran serially, with no full model, Studio or video generation.

| Guard receipt under `logs/` | Sampled peak MiB | Minimum cache-aware headroom | Swap growth MiB | Exit |
|---|---:|---:|---:|---:|
| `oracle-tiny-v1.jsonl` | 229.033 | 36.184% | 0 | 0 |
| `oracle-real-v1.jsonl` | 1752.237 | 30.992% | 0 | 0 |
| `native-tiny-v4.jsonl` | 19.954 | 35.091% | 0 | 1, numerical gate |
| `native-real-v2.jsonl` | 2738.659 | 27.507% | 0 | 1, numerical gate |

Sampling can miss between-sample peaks. Source state is parent `bd0bf23a` plus
the three test files in this slice. Toolchain/device and the three bridge
objects remain unchanged from the historical controls above. The final actual
report uses the final native binary; the tiny v4 report precedes only cosmetic
report-field renaming. `logs/native-controls-v4.jsonl` records five passing unit
controls and one pending GPU example with no external fixture.

| Artifact relative to evidence root | SHA-256 |
|---|---|
| `fixtures/tiny-v1.json` | `736864935ca2abc91d69609a03a823a32b7beb55ff2c09c9a95ea7d34982f6ea` |
| `fixtures/real-v1.json` | `481450208e45da4e648538c30855e5919538485e2eacf80e92773dddcc27d937` |
| `build/hy_nomas_attention_probe_spec` | `69c5be3d8597fd61c3f577b1a83662f9f7490c20148098f9a8b6754e6046cec2` |
| `logs/real-v2-comparisons.jsonl` | `46cee80de912bd891c3a645945bfe8f58cf7ff6017fce576b223a4209ece1086` |
| `logs/verification-v1.json` | `0547b5b5eb9eb9db000ab4e5b4d8b5f809a54c3d4987e36dff30e2b98a86be92` |

Actual source fixture/checkpoint pins are the preceding `real-bf16-v1.json`
and actual F32 subset digests, respectively. The tiny source fixture digest is
`3e1b9d014d02d71a1b38d4a9ea4eccd1a8f28f3824500d4f6f7ac691f67de0d6`;
its F32 weight-sidecar digest is
`c16f797818810ae066ed62733813a2adc36f1868b1edf6cc4f24f16f5ee37102`.

### Reproduce the diagnostic slice

Reuse the `guarded` function and pinned upstream/Python paths above, with a
fresh `probe_root` under `/opt/mldata_ssd/cogni-ml/` and its usual subdirectories.
Use freshly built bridges and `probe_link` from the build recipe above, or the
unchanged, digest-checked historical objects. Never run GPU jobs concurrently.

```sh
guarded build-attention env TMPDIR="$probe_root/tmp" CRYSTAL_CACHE_DIR="$probe_root/crystal-cache" \
  crystal build spec/hy_nomas_attention_probe_spec.cr \
  -o "$probe_root/build/hy_nomas_attention_probe_spec" --error-trace --link-flags="$probe_link"
PYTHONDONTWRITEBYTECODE=1 "$probe_python" spec/support/hy_nomas_attention_reference.py --structure-self-test
guarded attention-unit-controls env TMPDIR="$probe_root/tmp" "$probe_root/build/hy_nomas_attention_probe_spec"

probe_source_root=/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024
probe_source_digest=1fb14da66820ca061a87ec28024881de3165301d3993adb944cb45f275bc72e1
probe_checkpoint_digest=6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1
guarded attention-oracle-actual env TMPDIR="$probe_root/tmp" PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$probe_python" spec/support/hy_nomas_attention_reference.py \
  --fixture "$probe_source_root/fixtures/real-bf16-v1.json" --fixture-sha256 "$probe_source_digest" \
  --weights "$probe_source_root/weights/distilled-block0-f32.safetensors" \
  --weights-sha256 "$probe_checkpoint_digest" --upstream "$probe_upstream" \
  --output "$probe_root/fixtures/attention-actual.json"
shasum -a 256 "$probe_root/fixtures/attention-actual.json"
probe_fixture_digest=REPLACE_WITH_TRUSTED_GENERATION_DIGEST
guarded attention-native-actual env TMPDIR="$probe_root/tmp" \
  HY_NOMAS_ATTENTION_SOURCE=actual \
  HY_NOMAS_ATTENTION_FIXTURE="$probe_root/fixtures/attention-actual.json" \
  HY_NOMAS_ATTENTION_FIXTURE_SHA256="$probe_fixture_digest" \
  HY_NOMAS_ATTENTION_WEIGHTS="$probe_source_root/weights/distilled-block0-f32.safetensors" \
  HY_NOMAS_ATTENTION_WEIGHTS_SHA256="$probe_checkpoint_digest" \
  HY_NOMAS_ATTENTION_REPORT="$probe_root/logs/attention-actual-comparisons.jsonl" \
  "$probe_root/build/hy_nomas_attention_probe_spec"
```

Expected native exit is **1**, with four summaries and a final
`status=numerical_gate_failed` record. Missing completion, runtime/guard failure
or changed provenance is not an acceptable diagnostic result. Do not mask the
exit in parity automation. This separate diagnostic check verifies completion
and the scoped result; it does not turn failed BF16 parity into a pass:

```sh
python3 - "$probe_root/logs/attention-actual-comparisons.jsonl" \
  "$probe_fixture_digest" "$probe_checkpoint_digest" "$probe_source_digest" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
manifest, complete = rows[0], rows[-1]
assert manifest['phase'] == 'manifest' and manifest['source_mode'] == 'actual'
assert manifest['fixture_sha256'] == sys.argv[2]
assert manifest['checkpoint_sha256'] == sys.argv[3]
assert manifest['source_fixture_sha256'] == sys.argv[4]
assert (manifest['gate_atol'], manifest['gate_rtol']) == (2e-5, 2e-5)
assert manifest['rounded_gate'] == 'bitwise' and not manifest['production_changes']
raw = [r for r in rows if r['phase'] == 'raw' and 'kind' in r]
rounded = [r for r in rows if r['phase'] == 'rounded' and 'kind' in r]
assert len(raw) == len(rounded) == 188
assert all(r['stats']['violations'] == 0 for r in raw)
assert sum(r['stats']['bitwise_mismatches'] > 0 for r in rounded) == 101
assert sum(r['stats']['violations'] > 0 for r in rounded) == 83
for phase, count in [('reference_reproduction', 68), ('bitwise_control', 4)]:
    controls = [r for r in rows if r['phase'] == phase]
    assert len(controls) == count
    assert all(r['stats']['bitwise_mismatches'] == 0 for r in controls)
summaries = [r for r in rows if r['phase'] == 'summary']
assert len(summaries) == 4
assert all(r['operations'] == 47 and r['terminal_waits'] == 1 and
           r['encode_metal_buffer_allocations'] == 0 for r in summaries)
assert complete == dict(phase='complete', cases=4,
                        failed_comparison_records=101, status='numerical_gate_failed')
print('Diagnostic complete; raw F32 controls pass; BF16 parity remains failed')
PY
```

For the tiny replay, substitute its source fixture/sidecar and trusted digests
above, use fresh output paths and set `HY_NOMAS_ATTENTION_SOURCE=synthetic`.

Refresh this diagnosis after changes to operands/checkpoint, upstream or private
MPS operator semantics, geometry/dtype, native arithmetic, producer/validator,
compiler, bridge, runtime/device or command scheduling. Artifact loss requires
regeneration rather than trusting the summary counts.

## Prepared MPS block result, 2026-10-24

The new `HyNomasDualStreamBlock#encode_mps` is an explicit opt-in backend. It
keeps activations and BF16/F32 weights resident, reuses setup-time plans and
scratch, and preserves the existing native encoder and its recorded failures.
The measured path is Crystal/Metal/MPSGraph; PyTorch is only the pinned fixture
oracle. Risk remains CAUTION because graph rank, layout, constants and operation
order affect reduction arithmetic. Rollback is to omit `hy_nomas_block_mps.cr`
and its explicit backend; no default or tolerance was changed.

### Causal controls and implementation

Same-input MPS projection and QK controls remove the native reduction drift:
96 raw projection records and 12 raw QK records, plus their corresponding
rounded records, are bitwise equal to the declared MPS references on actual
weights. Sixteen raw softmax and sixteen raw PV controls also match bitwise,
as do their rounded records. Canonical MPS
attention matches its oracle. Alternative scaling placements are retained as
separate experiments; they are not promoted merely because their raw F32 error
fits the finite gate. The original diagnostic still exits 1 on known native and
alternative-path BF16 failures. This expected diagnostic failure is not a failed
test of the new opt-in full block.

Matching projections/attention alone did not close chained BF16 parity. Tiny
upstream/native F32 differences before a BF16 midpoint altered later bins. Each
intervention first ran on identical frozen operands; only the matching variant
was then inserted into the block:

- Non-affine LayerNorm follows the pinned PyTorch 2.8 single-row reduction and
  variance arithmetic; eight frozen raw comparisons are bitwise equal.
- Modulation and gated residual preserve separate products/additions instead of
  FMA contraction; sixteen frozen comparisons are bitwise equal.
- SiLU and tanh-GELU use setup-prepared MPSGraph plans with the pinned PyTorch
  operation order, constant rank and graph shape. SiLU has four same-input
  bitwise controls, repeated on one chain; eight frozen GELU controls match.
- Q/K RMS uses the existing native square / MPSGraph mean / native normalization
  path with fixed scratch; sixteen raw/rounded controls match. Video RoPE keeps
  adjacent-pair products/additions separate; eight raw/rounded controls match.

Prepared F32 projection stages accept BF16 resident weights via GPU conversion.
There are fourteen fixed linear bindings per block. The MPS workspace owns its
weight, head-major attention and RMS scratch; it is strongly rooted until an
explicit release after the registered chain drains. Projection hooks require
block preflight and obtain the encoder from that exact chain. Rejected early
release, direct projection before preflight and RoPE after release are negative
checks. A separate command-chain fix prevents a rejected pre-wait call from
being mistaken for completed GPU work (`345378c0`).

MPSGraph encodes use nonblocking execution descriptors. Encoder values are
refreshed after graph pauses/resumes. The runtime gate asserts zero tracked
encode-time `MetalBuffer` allocations and exactly one terminal wait per variant;
all diagnostic reads occur afterward. This does not claim zero opaque MPSGraph
allocation or zero internal driver/queue backpressure.

### Final verification and evidence

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-mps-reductions-20261024/`.
The block binary is `build/hy_nomas_block_spec-v11`, compiled from the committed
Crystal source with the fresh command-chain, Hy-Nomas reductions/activation and
Wan T5 linear/RMS bridge objects. Core bridge/stage objects are from the earlier
`hy-nomas-native-20261024/build/` root. The current source/bridge revisions, trusted
checkpoint and fixture digests must be retained together.

Both the actual C2048/H16/D128/FF8192 checkpoint and the tiny configuration pass
the BF16-boundary fixture: eight examples each, four cases (broadcast/per-token
modulation crossed with BF16/F32 weight residency), 536 finite comparisons per
fixture, no violations. Each case encodes a first block, a second block consuming
its resident output, a repeat of the first bindings and perturbed video input.
For actual weights all 480 saved first/chained/perturbed checkpoint arrays are
raw-F32 bitwise equal; all 72 selected BF16 boundary arrays have zero bit
mismatches. This is evidence for these fixtures, not a universal bitwise promise.
Receipts: `logs/block-{real,tiny}-v11-guard.jsonl` and corresponding
`-comparisons.txt` reports. The unchanged finite gate is still
`2e-5 + 2e-5*abs(reference)` and the BF16 boundary gate is still bitwise.

Frozen RMS/RoPE evidence is in `logs/attention-real-v5-comparisons.jsonl`;
same-input projection/QK evidence is in `logs/{tiny,real}-v1-comparisons.jsonl`.
The SiLU reference producer validates source fixture identity and extracts only
the four frozen modulation inputs; it does not load a checkpoint or instantiate
a model. Its references are `pointwise-{tiny,real}-v1.json` (SHA-256 below).
Native same-input SiLU controls and serial plan reuse are in
`logs/pointwise-real-v6-comparisons.jsonl` and its guard receipt.

```text
pointwise-tiny-v1.json d8a9070b87ca4f99ca47ef7695514fc3d9ea6fe567c374b3e05211dab70be2a1
pointwise-real-v1.json 8184c4c0e6e3293d0ed0b947f092c88d9c4fb08ef2834ffe87fe13ed774b90af
```

To replay the completed MPS block slice, use the guarded build/link commands
above including `wan_t5_rms_mps_bridge.o`. Run each trusted F32 or BF16-boundary
fixture serially with `HY_NOMAS_BLOCK_MPS=1`,
`HY_NOMAS_NORM_CONTROLS=1`, `HY_NOMAS_MODULATION_CONTROLS=1`, and
`HY_NOMAS_ACTIVATION_CONTROLS=1`, plus the independently pinned fixture/checkpoint
digests and a fresh `HY_NOMAS_REPORT`. Native-default regression uses
`HY_NOMAS_BLOCK_MPS=0 HY_NOMAS_ACTIVATION_CONTROLS=0` on an F32 fixture. Setting
activation graph controls without the MPS chain is rejected. The same-input
attention diagnostic retains its expected failing exit and original records.

Verdict: correlated MiniQuad ROBUST for the supported `encode_mps` fixture
contract, binding/lifetime guards and successful-path synchronization. This
does not certify CUDA/Magi output dtype or FlashAttention reduction, all-BF16
activations, camera/history/streaming, video quality, target-shape performance or
the full pipeline. Refresh after scoped source, operand/checkpoint, graph rank or
layout, precision policy, bridge/compiler, PyTorch/MPS/OS/device or scheduling
changes. Missing evidence requires reproduction; a new failure reopens this gate.

The final F32 fixtures also pass through `encode_mps`: eight examples and 536
finite comparisons each for actual and tiny weights. All 480 saved checkpoint
arrays per fixture are raw-F32 bitwise equal. The original native-default F32
actual-weight regression still passes eight examples and 488 comparisons under
the same finite gate. Reports/receipts are
`logs/block-{real,tiny}-f32-v11-*` and `logs/block-native-f32-v11-*`.
The producer's offline structure self-test, Crystal format checks and exact-diff
whitespace check pass. The one final formatting-only change to
`spec/support/hy_nomas_modulation_controls.cr` does not alter its operations.

The final five block runs completed with zero swap growth, sampled process-group
footprints below the 4 GiB cap (maximum 3541.144 MiB) and minimum cache-aware host
headroom of 24.532%, above the unchanged 15% floor. Sampling cannot exclude
between-poll peaks; file-backed cache is an estimate of reclaimable memory,
not guaranteed immediately free pages. Build/probe receipts retain these guard
limitations. macOS 26.6.2, Apple M2 Max, Crystal 1.22.1, PyTorch 2.8.0 oracle.
No full checkpoint/model/pipeline was allocated. The nine-token command timings
include repeated blocks and diagnostics with different native/MPS controls;
they are not paired block throughput measurements or evidence for video speed.

## Compact attention slice, 2026-10-24 (explicit opt-in)

`HyNomasCompactAttentionMPSPlan` and its native bridge pack K/V once per encode
into fixed BHTD buffers. Fixed smaller Q/output stages are reused serially. Each
chunk runs the same rank-4 QK -> epsilon -> scale -> softmax -> PV graph. All
bindings, offset views, graphs and execution descriptors are prepared before
encoding. The canonical dense plan remains the default/control; rollback is to
omit the new workspace keywords (`attention_query_chunk` and
`attention_heads_per_group`, both zero by default).

Every query graph has exactly C rows. The last window begins at `T-C` when a
short tail would otherwise remain: independent attention rows are recomputed
and overwritten on the GPU. This preserves the prepared graph shape, at the
cost of at most C-1 redundant rows. No mask, causal, camera or history semantics
are introduced. MPS command roots are refreshed after graph encoding before
subsequent native gather/scatter work. Explicit release requires the latest
chain to be drained; abandoned compact plans deliberately stay strongly
retained until release/process exit.

The workspace rejects known reduction counterexamples before device allocation:
G must equal H; partial chunks require B=1 and C>=32. These are guards against
observed failures, not a proof for every admitted dimension or activation.
The general standalone constructor retains grouped/multi-batch/short-chunk
variants for diagnostics only. No compact default is promoted.

### Numerical falsifiers and measured scope

Changing graph shape can change MPS reduction order. G4 at T129/T257 and C4 at
T9 have BF16-RNE counterexamples under the unchanged gate. Adding a B2/T65/H16/
D128/C32/G16 test found another: max F32 error `5.9604645e-8`, but 17 BF16 bins
differ. That partial multi-batch case is now rejected by workspace admission.
Its failed receipt is retained as `logs/compact-overlap-v6-f32-*`.
`HY_NOMAS_COMPACT_GROUP_DIAGNOSTIC=1` explicitly reruns these counterexamples;
this mode must fail the original gates, not become a passing qualification.

Final standalone controls use full-index avalanche inputs rather than a short
periodic pattern, changed GPU-resident Q/K/V, repeated encodes in one chain,
a second completed-chain reuse, end canaries, binding/alias/size preflight and
premature-release rejection. F32 and BF16-rounded inputs each pass 19 examples
and 48 dense-versus-compact comparisons under the unchanged finite and exact
BF16-bin gates (`logs/compact-overlap-v7-{f32,bf16}-*`). In each input policy,
45/48 comparisons are also raw-F32 exact. The three T65/C32 first/changed/reuse
records have raw-F32 drift up to `5.2154064e-8` with zero finite violations or
BF16-bin differences. The B2/H3 partial-tail positive case exercises offsets
without qualifying partial multi-batch use in the block workspace. The guard
and retained plan-count checks also pass. Full block fixture runs also
exercise the workspace constructor directly: grouped and partial short/multi-batch
requests must reject before any tracked MetalBuffer or lifetime root is added.

The independent synthetic oracle uses PyTorch 2.8.0 MPS, git
`a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`, with CPU fallback disabled. Producer
`spec/support/hy_nomas_compact_attention_reference.py` SHA-256 is
`d5f5a4d8ac0a75f719470e532d57ca0a45b6a5cabe03bc399cf6c787647a670f`.
The verified [upstream Attention.mm](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/mps/operations/Attention.mm)
has SHA-256
`167ee89bddb489c154b5d5ca72cf042745443f557d624fc48175f2d29f45a351`.
The producer/consumer pin manifests, binary outputs, runtime/source provenance,
input formula, shapes and per-record byte ranges before device allocation.

For both F32 and BF16-rounded Q/K/V, C128/G16 matches the full upstream T257
output (526,336 values) and three selected 128-row outputs with full K/V length
6,560 (786,432 values, starts 0/3,200/6,432). These are raw-F32 bitwise equal.
The latter is a selected-row reference, not a full-Q6560 upstream graph or an
end-to-end model oracle. PyTorch is used only to create fixtures; the measured
native path does not contain PyTorch.

The block fixture matrix covers actual C2048/H16/D128/FF8192 checkpoint weights
and the tiny configuration, both F32 and BF16-boundary policies, both weight
residencies, broadcast/per-token modulation, chained outputs and perturbed
inputs. Integration uses C=T=9 (G16 actual/G2 tiny); it proves the workspace
binding/lifetime integration, not a representative long-query block speed.
Every finite comparison retains `2e-5 + 2e-5*abs(reference)` and selected BF16
boundaries require exact bits. Each of the four final compact block runs passes
eight examples, all 536 finite comparisons and all 480 raw-F32 checkpoint arrays;
the two BF16 fixtures also match all 72 selected boundary arrays exactly. A dense
MPS tiny-F32 default-path regression passes the same eight examples/536 comparisons.
Encode paths assert zero tracked MetalBuffer
allocations and one terminal wait per independent case. This does not bound
opaque MPSGraph/driver allocations or hidden API stalls.

### Native attention timing and memory geometry

Release-mode, GPU-filled BF16-rounded inputs, H16/D128, warm plans, four serial
paired repetitions in AB/BA/AB/BA order. Timing includes `plan.encode` through
one terminal wait, including K/V packing and query gather/output scatter;
setup/compilation, input fill and a separate GPU diagnostic are outside it.
Only the diagnostic's five counters are copied to the CPU after completion.
No synchronization was added inside production attention encode.

| Build/run | Tokens | Arm A | Arm B | Median wall A/B (ms) | Median GPU A/B (ms) |
| --- | ---: | --- | --- | ---: | ---: |
| Initial | 1,024 | C128 | canonical dense | 3.260 / 2.557 | 2.654 / 2.150 |
| Initial | 6,560 | C128 | C256 | 66.784 / 61.589 | 66.403 / 61.227 |
| Initial | 6,560 | C256 | C512 | 61.479 / 62.646 | 61.121 / 62.318 |
| Initial | 6,560 | C256 A/A | C256 A/A | 61.475 / 61.586 | 61.018 / 61.133 |
| Makefile | 1,024 | C128 | canonical dense | 2.808 / 2.271 | 2.489 / 1.930 |
| Makefile | 6,560 | C128 | C256 | 68.467 / 62.277 | 68.013 / 61.846 |
| Makefile | 6,560 | C256 | C512 | 62.920 / 63.581 | 62.286 / 62.961 |
| Makefile | 6,560 | C256 A/A | C256 A/A | 65.778 / 65.371 | 62.332 / 62.355 |

C256 reduces observed long-input median wall time by 7.8% versus C128 in the
initial run and 9.0% after a fresh Makefile build, with the same sign in all
eight pairs. The initial C256 A/A within-pair wall ratios range from 0.9922 to
1.0058. The Makefile A/A has wall outliers (ratios 0.8273 to 1.0445), while GPU
ratios remain 0.9955 to 1.0055. One arm takes 83.5 ms wall with 62.7 ms reported
GPU time. This is evidence of latency outside the reported GPU span; it does
not isolate host scheduling from queue/API delays. Both runs are retained.
C512 encodes faster on the CPU but takes longer on the GPU; its small advantage
or disadvantage in wall time is vulnerable to these outliers. No default is
selected from that margin. These are local attention measurements, not
full-block/video throughput.
The short dense control is faster; compact attention buys a bounded score
shape for long sequences, not a universal speedup over dense MPS.
All arm comparisons have zero nonfinite values, finite violations and BF16
bin mismatches, and maximum absolute error zero over every output value.
The long-input full-output parity chain is transitive: C128 versus C256 in one
run, then C256 versus C512 in a separate run on the same generated inputs.
Independent upstream evidence for C128 is limited to the selected rows above.

| J=6,560, B1/H16/D128 | Named stages (MiB) | Largest score (MiB) | Graph calls |
| --- | ---: | ---: | ---: |
| Dense | 205.0 | 2,626.5625 | 1 |
| C128/G16 | 104.5 | 51.25 | 52 |
| C256/G16 | 106.5 | 102.5 | 26 |
| C512/G16 | 110.5 | 205.0 | 13 |

Named stages are `2*B*H*(T+C)*D*4`; score shape is `G*C*T*4` bytes. They are not
opaque peak-memory bounds. The benchmark preflight estimates named buffers,
three score-sized reserves per arm and a 256 MiB opaque reserve; it rejects the
J6560 dense comparison before Arena creation under the 4 GiB estimate. A fit
is only a geometry estimate. The external guard is still mandatory for every
run and is not enforced by the benchmark executable itself.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-compact-attention-20261024/`.
`logs/*-argv.json` stores exact final commands; fixture pins and source/object/
binary hashes are indexed by `logs/final-receipt-v1.json`. Positive runs and
expected-failing counterexamples have separate outputs and guard logs. Every
reported run uses the unchanged sampled 4 GiB maximum (smaller caps for small
probes/builds), cache-aware 15% host floor, 64 MiB system swap-growth stop and
600 s deadline. The five final block runs (`logs/constructor-v4-*`) have a
maximum sampled group footprint of 3412.112 MiB, minimum cache-aware headroom
19.379% and zero swap growth. Sampling can miss short peaks; reclaimable
file-backed cache is an estimate, not guaranteed immediately free pages.
The fresh Makefile release target also builds successfully under the guard;
its four benchmark runs use the exact final benchmark source and bridge.

Rebuild the benchmark with the Makefile target below inside the same external
guard, with TMPDIR and CRYSTAL_CACHE_DIR on the SSD:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-compact-rebuild \
  /opt/mldata_ssd/cogni-ml/hy-compact-rebuild/hy_nomas_attention_benchmark
```

Run through `examples/support/wan_memory_guard.py` with
`--memory-accounting file-cache --max-footprint-mib 4096`,
`--max-swap-growth-mib 64 --timeout-sec 600 --log FRESH`. Pass the benchmark
`--tokens=6560 --query-chunk=128 --compare-query-chunk=256`,
`--input-bf16-boundary --repeats=4`. Use `--compare-dense` instead of the second
chunk for the T1024 control.
`make spec_hy_nomas` includes the standalone and pinned-fixture consumer specs;
the upstream consumer is pending unless all trusted fixture pins are supplied.

## Continuation

Active goal: native Hy-Nomas port, initially one proven DiT block. Return pointer:
Wan performance work remains at `docs/wan-performance-frontier.md`, commit
`e22fc2ba`; no Wan default has been changed. Confirmed frontier: the actual
checkpoint block passes the declared F32 and BF16-boundary MPS controls with
both weight residencies and chained GPU outputs. Compact attention is explicit
opt-in; C256/G16 is the fastest tested long-query candidate, with selected-row
upstream parity and full-output native cross-chunk parity on the exercised
inputs. Known grouping/short-chunk/multi-batch counterexamples remain rejected
by workspace admission and retained as failed diagnostics. No general shape or
activation qualification follows from the measured matrix.

Next discriminator: representative native phase/whole-block timing at J=6,560
with C256/G16, against the 277.8 ms necessary block ceiling for two minutes per
video second at 8 steps/54 blocks. Share serial weight stages and liveness
scratch before full-model residency. The roughly 62 ms reported GPU time for
standalone attention alone does not
establish video throughput: projections, conditioning, scheduler,
text encoder, camera/history and PixelUMM remain outside it. Exact CUDA/Magi
output dtype and BF16 activation semantics are also unresolved. Standing
authority remains bounded GPU probes under the 15% floor and sampled 4 GiB cap;
full generation is outside this slice. Refresh evidence after source, fixture,
geometry, numerical policy, compiler, bridge, SDK/OS/device or scheduling changes.
