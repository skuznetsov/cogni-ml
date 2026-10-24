# Hy-Nomas native Metal frontier

Document status: actual-checkpoint F32-math block control verified within the
bounded scope below (2026-10-24). Same-input diagnostics localize projection and
attention differences, but BF16 attention-boundary parity still fails. A video
pipeline remains unimplemented.
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

## Active slice: BF16 boundaries and checkpoint weights

Status: actual-weight F32 control passes; BF16-boundary gate remains open,
2026-10-24. Preserve the existing F32-control default.
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

## Continuation

Active goal: native Hy-Nomas port, initially one proven DiT block. Return pointer:
Wan performance work remains at `docs/wan-performance-frontier.md`, commit
`e22fc2ba`; no Wan default has been changed. Confirmed frontier: one actual
checkpoint block passes the F32-math control with both weight residencies.
Frozen-input diagnostics retain BF16 projection/attention failures: QK score
substitution resolves index 10120, while index 9949 requires a broader
score-to-output arithmetic investigation. Next discriminator: compare native
projection/QK reductions with MPS and isolate normalization/PV with frozen MPS
probabilities. Preserve all failed boundary records and the original gate.
Exact CUDA/Magi output dtype
and BF16 activation semantics remain unresolved. Then measure MPS GEMM/attention
at representative token counts before full-model residency, camera/history
and PixelUMM. Standing authority remains bounded GPU probes under the 15% floor
and sampled 4 GiB cap; full generation is outside this slice.
Refresh evidence after source, fixture, geometry, numerical policy, compiler,
native bridge, SDK/OS/device or command-scheduling changes.
