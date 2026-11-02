# Hy-Nomas local draft suite

## Reference-image frontier (2026-11-02)

Status: the native DCAE-V encoder is numerically qualified at the fixed
512x288x120 reference geometry. The saved clean latent also decodes exactly
against the adapted source MPS control described below. Image-conditioned
continuation is implemented behind an explicit experimental opt-in; BF16 DiT
cache parity and character/style quality are not yet qualified. It is not
enabled by default.
The opt-in standalone block-0 pre-attention path now reproduces all 22
captured BF16 boundaries exactly on identical exported inputs. Its separate
first-T0 history probe also matches physical BF16 K/V storage and F32 reads
exactly against the pinned upstream cache function. Its attention consumer
now matches an isolated invocation of the pinned Torch branch on the same
4,849 active rows exactly. Padded Torch/Flash3, remaining block operations
and all-layer history remain separate gates.
The rejected text-only RoleWright take does not qualify character or style
preservation. Do not generate another text-only take as a substitute for this
capability.

Current upstream added reference-image initialization in
[`a31c0fe0`](https://github.com/Tencent-Hunyuan/Hy-Nomas/commit/a31c0fe0bcc493be188fd37ddba4ce2f674becb4).
The existing DiT/PixelUMM numerical reference uses the older October 24
snapshot. The DCAE-V encoder oracle is pinned to `a31c0fe0`; the older demo's
lack of image input must not be attributed to current upstream. The new
[demo contract](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/a31c0fe0bcc493be188fd37ddba4ce2f674becb4/apps/README.md#start-with-a-reference-image)
and [encoder entry point](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/a31c0fe0bcc493be188fd37ddba4ce2f674becb4/hymm/pipeline/reference_image.py)
were inspected at that immutable revision. This is source evidence, not a
locally executed image-conditioned generation or a quality qualification.

### Admitted upstream representation

The image initializes a new persistent session as a fixed five-second segment.
Fit the image to the target frame with preserved aspect ratio and centered
black padding, normalize pixels to [-1,1], repeat it in pixel space for 120
frames at 24 fps, and encode that complete segment through DCAE-V. Preserve
the encoder's temporal feature-cache semantics; encoding one frame and
duplicating its latent is not equivalent. Use `latent_dist.mode()` with no
extra latent scaling. The result has shape `[1,32,30,H/32,W/32]`; at the native
512x288 geometry it is `[1,32,30,9,16]`.

Skip denoising of the fixed reference segment. A single DiT cache-writer
forward uses `(1-sigma)*z + sigma*noise`, with sigma 0.206897 and timestep
206.897, to fill image/text K/V history. Decode the clean reference latent
through PixelUMM and preserve the streaming timeline for the next generated
segment. Upstream retains the full reference segment in history while its UI
shows only the last 12 decoded frames as T0. The subsequent instruction drives
T1. These are separate clean-latent, noisy-cache and display contracts.

This route uses the released DiT without additional training. It admits one
complete scene image at session initialization, not arbitrary mid-session
replacement, independent character/style reference lists, or guaranteed
identity preservation. The demo caption contract selects one primary subject;
the quality of a two-character RoleWright scene remains an open probe.

### Native gap and first implementation slice

The experimental native generator now accepts an authenticated fitted RGB
bundle and a separately authenticated DCAE-V encoder checkpoint. The encoder
implements the checkpoint topology, chunk-causal convolutions, temporal cache,
normalization, downsampling and pixel-unshuffle semantics with prepared
MPSGraph operations and native Metal copies/arithmetic. Its tested boundaries
are listed below. Wan VAE latents are incompatible; PixelUMM is a decoder,
and Qwen-VL image recognition is not this DiT conditioning path.

The implementation reuses cache-noise arithmetic and compatible timeline/history
buffers. A distinct reference-initialization transition skips denoising and
admits exactly one writer to seed T0 history. Generated-segment guards still
apply to T1 onward. The input manifest and runtime receipt bind the image digest,
fit geometry, encoder checkpoint, caption, noise and cache policy. An existing
noise bank or restored text-only checkpoint is not an image reference.

Plan encoder weights and scratch as a separate residency phase before DiT;
retain the encoded latent and release encoder resources at the safe phase
handoff. Use reusable GPU buffers and GPU tensor computation, without new
readbacks or waits inside command chains. Establish a cache-aware memory plan
before any model run, retaining the 10% floor, process/swap guards and one
model-owning process. Source inspection alone does not establish the encoder
peak, supported memory limit, or runtime.

The acceptance order is: (1) compare native DCAE outputs on identical pixels
against upstream encoder revision `a31c0fe0`; (2) compare clean-latent
PixelUMM reconstruction; (3) compare noisy-cache inputs and per-layer K/V;
(4) compare a short image-conditioned continuation on the same caption and
noise. Keep existing 2e-5 absolute/relative tolerances; localize a mismatch
rather than widening them. Diagnostic captures belong outside the measured
path. Numeric parity and visual character/style preservation are separate
gates; neither substitutes for the other. The current implementation slice
admits bounded encoder and initialization probes, not a full video generation.

For RoleWright, first prepare an approved single scene containing Alex and the
living resume in the supplied style. Character turnarounds and expression
sheets are preparation references, not frames of the generated world. Test
one simple action before the complete ad. Reference conditioning does not
itself implement timed actions, eye-line control or exact interface text.

Rollback is the unchanged text-only runtime. Refresh this design after an
upstream/checkpoint, geometry, temporal-cache or timeline-contract change;
require new runtime evidence before admitting native image-input support.

### Implementation admission and rollback

The missing original checkpoint is pinned to `tencent/Hy-Nomas` revision
`bb6a8ca451bfe84947f0ea1c6d51957f983ffffe`, path
`VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt`, 4,277,389,995 bytes,
SHA-256 `ebbee483c80890c7795a2babc06e1cb4d42564174cf3bae2b698c0589d95cc3b`.
The checkpoint has been downloaded and authenticated. Its encoder-only
safetensors derivative preserves all 169 encoder tensors and 1,833,310,016
BF16 payload bytes (1,833,331,160 file bytes), SHA-256
`c77e705b3d1534f1d3fcca68d146c9eecd479be5d01110aadee7b81ddc05ed50`.
Conversion uses the restricted weights-only
loader and verifies the derivative against the source; it is setup work,
not part of native inference.

CAUTION: a reference must seed clean T0 pixels/latents and exactly one noisy
cache writer, without denoising T0, replacing the fixed noise bank, or admitting
text-only checkpoint restore as a reference session. Keep the original path as
the default rollback. The first opt-in rejects unsupported combinations rather
than silently changing checkpoint, prompt-edit, Cinematic or cache semantics.

DoD: ingestion and schedule negative tests; a fresh guarded native build;
encoder memory planning before allocation; identical BF16 input pixels and
intermediate/final encoder parity against the original checkpoint on MPS;
and zero instrumented encode allocations, readbacks or preterminal waits.
Probe one small clip first, then a two-tile temporal-cache boundary before
the full reference. Use the existing terminal drain at the encoder-to-DiT
weight-phase handoff and keep its output GPU allocation resident. Stage
captures and artifact exports are diagnostic work after the terminal drain.
Numerical qualification does not establish RoleWright visual quality; that
requires a subsequent short controlled image-conditioned continuation.

### Observed encoder qualification (2026-11-01)

The guarded native encoder and the pinned original PyTorch 2.8 MPS encoder
consumed the same fitted RoleWright RGB pixels and original BF16 weights.
The native measurement path contains no PyTorch. All captured BF16 payloads
match byte-for-byte, with unchanged `atol=rtol=2e-5`:

| Probe | Captures | Result |
| --- | --- | --- |
| 32x32x4 | normalized input, post-stage-5, latent | zero value error |
| 32x32x80 | input, post-stage-5 tiles 0/1, latent | zero value error |
| 512x288x120 | input, post-stage-5 tiles 0/1/2, latent | byte-identical |

The full native encode took 306.621 s wall / 306.584 s GPU, with zero
instrumented MetalBuffer allocations, readbacks or preterminal waits and one
terminal drain. This is reference initialization cost, not generated-video
throughput or a paired performance claim. Its named diagnostic residency was
4,663,709,504 bytes; the production Session plan is 4,654,862,144 bytes without
the three stage captures. The sampled process-tree peak was 8.828 GiB under
a 12 GiB cap; minimum file-cache-aware headroom was 32.772 GiB, swap growth
zero, and the 10% floor remained active. Opaque MPSGraph resources are not
included in named-buffer counters; sampling cannot rule out between-poll spikes.

The full-J reference timeline arithmetic probe also passed, including clean T0
scatter/gather, cache noise and RoPE/time preparation, with no tracked
allocations/readbacks/preterminal waits. It is not an actual per-layer DiT
cache comparison. Focused checks passed: 52 CPU-only Crystal examples,
29 Metal-linked checkpoint/cache examples, 14 Python tests and fresh native
builds. Mini-Quadrumvirate review found and fixed missing conv1/downsample
bias and MPS executable feed reordering; the current integrated schedule and
ownership paths have a scoped correlated Adversary verdict of ROBUST.

Evidence and exact artifact/source identities:
`.landmark/evidence/2026-11-01-hy-nomas-reference-encoder.json`.
Local weights, fixtures, build products and logs remain under
`/opt/mldata_ssd/cogni-ml/hy-nomas-reference-20261101.SsKF88Bo`.
The annotated supplied image is a numerical fixture, not an approved final
RoleWright scene. No full video was generated in this slice.

The clean-reference PixelUMM gate is now qualified within the adapted source
MPS scope below. Remaining gates are noisy-cache inputs plus layer-specific
K/V against upstream, then a short controlled continuation from a single
scene. Partial final tiles after a full
tile and other production geometries are not qualified. Refresh encoder
evidence on source, binary, weights, input, geometry, cache policy, backend,
compiler/runtime or device change; local hashes cannot recover lost artifacts.

### Reference handoff qualification slice

This diagnostic slice reuses the authenticated, byte-identical saved encoder
latent for diagnostics. It must not synthesize a denoising receipt or represent
that saved fixture as an independently executed live encoder-to-DiT handoff.
First decode the clean `[1,32,30,9,16]` BF16 tensor through the native PixelUMM
and the source MPS control on identical decoder noise. Then inspect the noisy
T0 writer at the actual DiT projection/RoPE/cache boundaries before admitting
a two-segment run (fixed T0 plus five generated seconds in T1).

CAUTION: the existing native history stores F32 K/V, whereas the upstream
cache boundary is BF16. A native F32 continuation cannot establish upstream
cache parity merely because the encoder is exact. Any cache precision change
must be opt-in for reference initialization and must preserve the default
text-only/checkpoint path. The strongest failure is a numerically plausible
T0 reconstruction followed by a different T1 because of cache rounding or
incorrect writer scheduling. The guard is a same-input per-layer comparison,
plus negative schedule tests and unchanged 2e-5 absolute/relative tolerances.

Diagnostic DoD: authenticated saved-latent admission and rejection tests; a clean-latent
native/source-MPS decoder comparison including raw and display outputs;
same-input T0 cache-writer input and block-0 boundary captures that report
numerical failure without treating the F32 control as BF16 parity; zero instrumented
allocations/readbacks/preterminal waits inside the admitted native chains;
guard logs meeting the file-cache-aware 10% floor, process cap and swap-growth
limit. This diagnostic DoD is distinct from the continuation acceptance gate.
Only after the BF16 numerical gates pass may a short controlled scene probe
test motion and style. Numeric parity does not establish character identity,
script adherence or the quality of the complete RoleWright advertisement.
Rollback remains omission of the experimental reference flags. All diagnostic
artifacts and builds stay on the external scratch volume.

Observed clean-latent result: native and pinned source-MPS raw pixels and
display pixels are byte-identical, with zero error over 53,084,160 values in
each `[1,3,120,288,512]` output. The first 20 frames before overlap blending
are exact as well. This control uses the existing gathered-spatial-tile
QK/softmax/PV adapter with F32 accumulation; it does not qualify unmodified
eager flex-attention. No DiT, solver, or fresh DCAE encode ran in this decoder
comparison. Native time was 63.515 s wall / 63.466 s GPU, with zero tracked
encode allocations/readbacks/preterminal waits and one terminal drain.
Sampled process-tree peaks were 17.029 GiB native and 5.678 GiB source; minimum
file-cache-aware headroom was 24.260 / 23.024 GiB and swap growth was zero.

The native T0 input exporter preserves the actual 529-token valid prefix in
the 1280-row text-storage geometry. It captured timeline/noise/patch/refiner
outputs after one terminal drain with zero tracked encode allocations,
readbacks or preterminal waits. On the exact exported BF16 latent and epsilon,
the upstream cache-noise formula also produced byte-identical widened F32
values for all 138,240 latent elements. This isolates noise arithmetic; it
does not establish stock RNG-stream parity or patch/refiner parity. A separate
source block-0 probe retains BF16 parameters and autocast boundaries and stops
at K/V append, before attention. A widened F32 source run is a localization
control, not a substitute for the BF16 gate.

On identical exported preblock inputs, all 14 native projection/QK-normalization/
RoPE/prospective-cache captures and all eight pre-QKV captures match the widened
F32 source control exactly. For the F32 cache diagnostic, comparison is against
the unrounded source constituents before its BF16 store. The native probe
captures prospective joint K/V; it does not execute a history append. This
does not qualify the complete upstream T0 patch/refiner path or actual cache
storage.

The original BF16 source comparison fails at unchanged `atol=rtol=2e-5`.
All 14 main boundaries and all eight pre-QKV boundaries fail. The first captured
difference is image modulation SiLU: 777 of 2,048 values exceed tolerance,
maximum absolute error 0.0003126636. Source modulation SiLU/linear outputs are
BF16, but its LayerNorm and modulated QKV inputs are F32 under MPS autocast.
The native `BF16BoundaryControl` only rounds selected later boundaries and
does not reproduce this mixed precision path. The largest main-capture error
is 0.5 at prospective V; that is a failed gate, not an accepted tolerance.

A separate small MPS probe isolates the earliest difference on the same
2,048-element vector. F32 SiLU matches native exactly. Rounding only its output
to BF16 still leaves 410 values outside tolerance. Rounding both the input
and output to BF16 matches the source BF16 SiLU exactly. Thus the first
discrepancy is explained by the missing dtype boundaries on this fixture,
not by a different SiLU formula. This is localization, not a production fix
or proof of the remaining modulation/projection/reduction boundaries.

The next falsifier from that diagnostic was an opt-in native reference control reproducing those
input/output SiLU boundaries, followed by modulation linear, LayerNorm input,
QKV, QK RMS and RoPE on identical inputs. Its admitted scope ends at prospective
joint K/V, before attention, residuals and FFN. It must reject the ordinary
full-block entry point, serial reuse and history-provider attachment rather
than accidentally execute those unqualified stages. The BF16 cache store is
a subsequent gate, not part of a prospective-K/V pass. Do not blanket cast
the block to BF16: the observed source path is mixed. Keep the text-only
default unchanged and retain the experimental reference gate until the
BF16 comparison passes. No generated T1 or full video ran in this slice.

This CAUTION change reuses fixed-buffer BF16 linear plans and adds setup-owned
rounding buffers. Its rollback is omission of the reference-preattention flag.
The principal failure is an incorrect BF16 boundary silently reaching a full
forward; explicit entry-point rejection and unchanged same-input F32/BF16
comparisons guard it. DoD is the reference handoff CPU suite plus all 22
pre-attention captures against the pinned source at `atol=rtol=2e-5`, with zero
instrumented encode allocations/readbacks/preterminal waits and one terminal
drain under the 10% file-cache-aware guard. If a later boundary fails, retain
the first mismatch and narrow the claim; do not admit continuation. BF16 graph
and explicit F32 accumulation variants are candidate linear formulations, not
assumed source equivalence.

Focused CPU checks passed: `make spec_hy_nomas_reference_handoff` (53 Crystal
examples) and `python3 -B -m unittest spec.hy_nomas_reference_decode_test
spec.hy_nomas_reference_cache_test spec.hy_nomas_reference_block0_compare_test`
(43 Python tests). The comparison tools reject wrong parent artifacts, source
or dtype dependency maps, mismatched shape/dtype/accounting, symlink leaves
and a companion artifact differing from the explicit CLI argument. Correlated
mini-Quadrumvirate review is ROBUST for that bounded diagnostic scope; it is
not independent execution evidence or BF16 continuation qualification.

Canonical evidence: `.landmark/evidence/2026-11-01-hy-nomas-reference-handoff.json`.
Commands, guarded logs, binaries, captures and the SiLU discriminator remain
under `/opt/mldata_ssd/cogni-ml/hy-nomas-reference-handoff-20261101.pDnd7TV1`.
The native receipt's reference/implementation revision fields describe the
upstream semantic contract; the evidence packet separately binds the actual
local parent revision, source fingerprints and executed binaries. Refresh on
source, input, checkpoint, geometry, precision/backend/runtime/device changes
or artifact loss. Digests do not restore lost files.

### BF16 pre-attention qualification

The opt-in `reference-bf16-preattention` probe now implements the observed
mixed-precision boundaries with fixed GPU buffers: BF16-rounded SiLU inputs
and outputs, BF16 projection operands/results, F32 LayerNorm/modulation,
BF16 rounding of `1 + scale`, source-ordered QK RMS casts and BF16 RoPE
outputs. It uses the prepared `bf16-graph` linear variant. The separate
`f32-accum` candidate is not qualified by this result.

On the same authenticated preblock inputs, original BF16 weights and pinned
source captures above, all eight pre-QKV and 14 main tensors match exactly:
zero maximum/RMS error and zero mismatches, with unchanged `atol=rtol=2e-5`.
The image stream has 4,320 tokens and the valid text prefix has 529 tokens
in 1,280 rows of storage. The ordinary F32 control also matches all 22
captured tensors exactly. The previous native BF16 control still fails all
22 comparisons under the current comparator; the gate was not relaxed.
These are same-input block-0 results, not complete upstream T0 preprocessing
or RNG parity.

Both final native probes report zero tracked GPU-buffer allocations,
readbacks or preterminal waits, and one terminal drain. The BF16 path adds
114,753,536 named bytes for rounding/linear outputs, making its diagnostic
named-buffer total 2,282,853,888 bytes; unused ordinary block plans remain
prepared. Opaque MPSGraph resources are excluded. Serialized guarded runs
retain the file-cache-aware 10% floor, 6 GiB process-tree cap and 64 MiB
swap-growth limit. This is numerical qualification, not a speed benchmark.

The workspace rejects the ordinary full-block entry point, serial blocks,
history attachment, attention and residual execution. Source-BF16 RMS
rejects padded reduction rows before allocating its scratch, because those
rows would read unwritten squared inputs. An observed subclass lifetime-root
test failure was fixed by explicitly sharing the base-class registry. Roots
hold the complete dynamic workspace until terminal-drained release. A
failure injected after RMS pauses the encoder confirms that `ensure`
restores encoding without an extra allocation or wait; premature release
remains rejected.

Focused checks pass: 61 CPU Crystal examples, 44 Python tests, and nine
Metal-linked workspace/default serial/prefix/modulation examples. The last
group exercises constructor cleanup, exact extra-buffer accounting, shared
root counts, scope rejection and terminal-drained release. Source review
plus the executed comparisons and failure-path tests give a correlated
mini-Quadrumvirate verdict of ROBUST within this fixture-only scope.

Evidence: `.landmark/evidence/2026-11-01-hy-nomas-reference-bf16-preattention.json`.
Builds, commands, captures and guard logs remain under
`/opt/mldata_ssd/cogni-ml/hy-nomas-reference-bf16-20261101.3450pdjh`.
Rollback is omission of the reference-preattention flag; the text-only
arithmetic is unchanged. Refresh on source, binary, input, weights,
geometry, precision, cache policy, backend, compiler/runtime/device changes
or local artifact loss.

The next falsifier is an actual reference-history K/V store/read comparison
on the same tensors, followed by the remaining attention/residual/FFN and
layer-specific cache boundaries. Only then admit a controlled clean-scene
T0/T1 continuation. Keep `--allow-unqualified-reference`: no actual history
append, generated T1, full video, or RoleWright character/style quality was
qualified by the standalone pre-attention result.

### BF16 history store/read — qualified block-0 fixture

The separate opt-in `--precision reference-bf16-history` probe now stores
the active joint K/V prefix in physical BF16 buffers and loads that pair
into prepared F32 reader buffers on the same command chain. Its 4,849 rows
contain 4,320 image tokens followed by 529 valid text tokens; the remaining
751 text-storage rows are excluded. Only `bf16-graph` is admitted here.

Both physical BF16 tensors are byte-identical to upstream. Both F32 reads
are also exact (9,930,752 values each, maximum/RMS error and mismatch count
zero) at unchanged `atol=rtol=2e-5`. Fresh regressions retain all 22 exact
BF16 pre-attention captures and all 22 exact F32 controls. A negative control
with repaired receipt hashes rejects changed BF16 bytes; a separate F32
mutation produces exactly one mismatch and a failed comparison.

The source probe executes the pinned `causal_attention_with_cache_mp`
function twice, intercepting only its attention consumer. After T0 store,
it zeros the same current Q/K/V in place and invokes the read-only branch:
the saved prefix remains unchanged and the current suffix is zero. Its
input Q/K/V comes from the preceding source capture, treated explicitly as
a trusted fixture. Hashes bind those bytes but do not independently attest
how that earlier capture was produced. No attention math is qualified by
this interception.

The strongest false positive was reading prospective K/V instead of the
cache. Metal tests overwrite the original source after store, then read;
another changes the physical stored values and observes them in the reader.
Geometry, aliasing, single-write, read-before-write, cross-chain and drained
release guards pass. A failure during pre-attention consumes the history
attempt before any retry; history-prepared workspaces reject the standalone
pre-attention entry point. Cache-owned buffers remain rooted until their
latest read chain drains, including a read on a later chain.

The cache pair costs 39,723,008 bytes and its two diagnostic F32 read stages
79,446,016 bytes: 119,169,024 extra named bytes, bringing this standalone
diagnostic to 2,402,022,912 bytes. Opaque MPSGraph resources are excluded.
The native history probe records zero tracked encode allocations/readbacks/
preterminal waits and one terminal drain. Its sampled peak is 1.559 GiB,
minimum cache-aware headroom 38.876 GiB, with zero swap growth under a 6 GiB
process cap, 10% floor and 64 MiB swap-growth stop. Sampling does not bound
between-poll peaks; these are diagnostic measurements, not a speed claim.

DoD passed: 71 CPU Crystal examples, 43 Python tests, eight Metal-linked
cache/workspace examples, fresh native/source captures and strict numerical
comparisons. Correlated source reviews plus root execution returned ROBUST
within this fixture-only scope. Evidence:
`.landmark/evidence/2026-11-01-hy-nomas-reference-history.json`; local run:
`/opt/mldata_ssd/cogni-ml/hy-nomas-reference-history-20261101.291siufs`.

Rollback is omission of the explicit diagnostic mode. The existing F32
continuation provider remains separate. Next falsifier: compare the
attention consumer on identical Q and stored K/V, then residual/FFN and
layer-specific histories before T0/T1 continuation. Keep
`--allow-unqualified-reference`: no full block, multi-segment append,
all-layer cache, generated T1 or RoleWright identity/style quality is yet
qualified. Refresh on source, binary, inputs, weights, geometry, precision,
cache policy, backend, compiler/runtime/device change or artifact loss.

### BF16 stored-history attention — qualified active-prefix fixture

The opt-in `reference-bf16-attention` block-0 diagnostic consumes physical
first-T0 BF16 K/V through a prepared native MPSGraph plan. All 9,930,752
BF16 output values are byte-identical to an isolated invocation of the
pinned upstream `attn_mode == 'torch'` branch body on MPS, with byte-identical
Q and stored K/V. Maximum/RMS error and mismatch count are zero at unchanged
`atol=rtol=2e-5`. The fixture has 4,320 image + 529 valid text rows, 16 heads
and head dimension 128. It does not establish parity with the padded
5,600-row Torch path, Flash3, or the full upstream T0 handoff. The earlier
source Q/K/V capture remains an explicitly trusted fixture, not an
independently attested full-model execution.

The initial fused MPSGraph SDPA node failed 997,904 output values, with
maximum error 0.5 despite exact Q/K/V. The installed source control is Torch
2.6.0 (`2236df1770800ffea5697b11b0bb0d910b2e59e1`); its
[MPS implementation](https://github.com/pytorch/pytorch/blob/2236df1770800ffea5697b11b0bb0d910b2e59e1/aten/src/ATen/native/mps/operations/Attention.mm)
uses BF16 QK, F32 scaling, a BF16 scaled-logit boundary, BF16 softmax and
BF16 PV. Reproducing that explicit graph eliminated the mismatch. This
rejects the fused node for this fixture; it does not characterize its
unobserved internal arithmetic or establish parity with another Torch version.

The plan binds physical cache storage, compiles and allocates before
encoding, and widens its BF16 context into the existing F32 destination on
GPU. A Metal discriminator poisons prospective sources after store, assigns
distinct per-token/head/channel values to physical V, and checks exact
means with zero Q/K. Tail canaries, alias/extent checks and latest-chain
release leases pass. Hash-repaired output corruption produces one numerical
mismatch; hash-repaired Q corruption is rejected before comparison.

The new Q/output buffers add 39,723,008 named bytes, for 2,441,745,920
diagnostic bytes including the previous history/read stages; opaque MPSGraph
resources are excluded. Native counters remain zero tracked encode
allocations/readbacks/preterminal waits and one terminal drain. Guarded
source/native runs retain the file-cache-aware 10% floor, 6 GiB process cap
and 64 MiB swap-growth stop, with zero sampled swap growth. Sampling does
not bound between-poll peaks; this is not a speed or production-memory claim.

DoD passed: 73 CPU Crystal, 49 Python and nine Metal examples; fresh final
attention/history captures; strict comparisons; all 22 BF16 and 22 F32
pre-attention regression tensors remain exact. Evidence:
`.landmark/evidence/2026-11-02-hy-nomas-reference-attention.json`; local run:
`/opt/mldata_ssd/cogni-ml/hy-nomas-reference-attention-20261102.i888t5gm`.
Rollback is omission of this diagnostic mode. Next falsifier: output
projection/residual and FFN on identical context and modulation inputs,
then a complete block and layer-specific history. Keep
`--allow-unqualified-reference`: no cache append, all-layer parity,
generated T1, full video or RoleWright identity/style quality is qualified.
Refresh on source/binary/input/weights/geometry/precision/cache/backend/
compiler/runtime/device changes or local artifact loss.

### Preparing and admitting an image

Prepare one complete scene image with
`examples/support/hy_nomas_reference_image.py --image /absolute/scene.png
--output-dir /opt/mldata_ssd/cogni-ml/reference-scene --encoder-sha
c77e705b3d1534f1d3fcca68d146c9eecd479be5d01110aadee7b81ddc05ed50`.
The output directory must be fresh. The helper decodes and fits the image once
and prints the manifest path and SHA-256. It writes RGB8 pixels, not latents;
normalization, temporal repetition and all encoder tensor operations run on GPU.

The sequence generator accepts the following four flags together, in addition
to its existing authenticated timeline inputs, segment count and named budget:

```text
--reference-image-inputs /absolute/reference-scene/manifest.json
--reference-image-inputs-sha <manifest SHA-256 printed by preparation>
--reference-encoder /absolute/encoder.safetensors
--reference-encoder-sha c77e705b3d1534f1d3fcca68d146c9eecd479be5d01110aadee7b81ddc05ed50
```

Run `--plan-only` first; it authenticates image/weight payloads without creating
a Metal model. Execution additionally requires `--allow-unqualified-reference`
and the external file-cache-aware memory guard. The first slice rejects
Cinematic, legacy per-chunk text caching and checkpoint restore/edit/capture
combinations. Omitting the quartet retains the text-only path.

`--segments 2` means a technical five-second T0 reference plus five seconds of
generated T1, not ten generated seconds. The raw pixel output currently includes
all of T0. Upstream's display-only trim to T0's last 12 frames is not applied;
the receipt labels reference and generated durations separately.

## Design-sealed Cinematic comparison (2026-11-01)

The Cinematic comparison adds an explicit BF16 opt-in to the two-segment
drafter and compares the retained six scenes with their original first takes.
Keep the exact frozen suite, authenticated conditioning/timeline manifests,
noise banks, four steps, geometry, decoder and history policy. Reuse validated
baseline input bundles instead of re-encoding prompts or generating new noise.
Base remains the default. The existing offline merged cache must pass its
lineage/index admission; no online LoRA merge or new GPU operation is admitted.

This is CAUTION for model selection and artifact provenance. The strongest
failure is a visually plausible comparison made with a different input or a
mislabelled weight index. Bind the requested variant and cache identity through
preflight, loader and final receipt, and reject a mismatched selected pair
before rendering that scene. Keep the existing 10% file-cache-aware floor,
30 GiB process cap, swap-growth stop and one model-owning process at a time.
The sequence checkpoint/resume path remains base-only unless its contracts
are separately extended; a Cinematic instance must not emit base-labelled
checkpoints. Roll back by omitting the new flags; retained base media are
read-only and must never be overwritten.

DoD: failing-before-fix variant/input identity checks, focused Python and
Crystal admission tests, a fresh guarded native build, CPU-only memory plans,
and six completed Cinematic clips paired with authenticated base clips.
Require unchanged input/noise digests, variant-specific index/lineage digests,
240 frames at 24 fps, zero instrumented encode allocations/readbacks or
preterminal waits, and one existing terminal drain per GPU phase. Refresh the
selected upstream merge check and authenticate the prior intermediate evidence
without widening 2e-5 absolute or relative tolerances. Inspect paired media and
report quality observations separately from numerical parity; full upstream
video parity is not admitted.

VACE is an explanatory branch only: no new VACE weights, renderer, training or
model launch is included in this implementation.

### Opt-in usage and controls

`examples/hy_nomas_draft.py` accepts `--cinematic-cache` (the existing offline
BF16 cache's `cinematic-lineage.json`) together with `--cinematic-cache-sha`.
The native sequence CLI accepts the same pair. Omit both to use original BF16
weights. The selected variant, index digest and lineage are bound to the CPU
memory plan and final native receipt; the gallery displays the effective
variant instead of relabelling Cinematic as original.

For controlled comparisons, pass `--reuse-inputs-from /absolute/results.json`
with the exact original suite bytes. Admission requires a completed run with
the same ordered scenes, prompts, Qwen conditioning and fixed noise. It checks
the original raw pixels, MP4s and payloads, then reuses those input paths.
Scene metadata and selected payloads are checked again before native preflight;
native preflight and execution independently authenticate the payloads.
Neither Qwen encoding nor RNG/input production is rerun. Reuse does not turn
archived runtime measurements into a fresh paired speed benchmark.
Admission initially authenticates all scene payloads; subsequent selected-scene
checks reject a changed payload before that scene's preflight or execution.
This is an incremental suite, not an atomic transaction over concurrently
mutable future inputs. A later failure preserves earlier completed scenes and
leaves the overall report incomplete.

For example, use a new output directory and a separately trusted lineage
digest rather than deriving that trust from an unverified cache at launch:

```sh
python3 -B examples/hy_nomas_draft.py \
  --suite /absolute/frozen-suite.json \
  --generator /absolute/build/hy_nomas_sequence_generate \
  --run-dir /absolute/new-cinematic-run \
  --reuse-inputs-from /absolute/base-run/results.json \
  --cinematic-cache /absolute/cinematic-cache/cinematic-lineage.json \
  --cinematic-cache-sha "$CINEMATIC_LINEAGE_SHA"
```

### Observed six-scene comparison (2026-11-01)

All six Cinematic first takes completed with the exact retained native prompts,
conditioning manifests/tensors and three noise banks. Each clip contains two
five-second segments, four steps per segment, 240 frames at 512x288/24 fps,
original-precision BF16 merged weights and the existing F32 live history. No
conditioning producer or RNG was rerun. Base remains the default.

| Scene | Total scene seconds | Sampled process-group peak GiB |
| --- | ---: | ---: |
| portrait | 283.053 | 24.221 |
| pouring-water | 308.829 | 24.305 |
| running-dog | 307.578 | 24.221 |
| balloon-car | 278.368 | 24.321 |
| coast-flight | 266.863 | 24.217 |
| desert-donut | 266.600 | 24.320 |

The median observed total was 28.071 seconds per output video second, excluding
the already-retained conditioning preparation. This is descriptive timing,
not a fresh paired speed benchmark: five of the six base clips are historical.
A fresh original portrait on the new binary reproduced the retained raw pixels
and latent bank byte-for-byte. The historical receipts lack explicit variant
identity; the fresh control anchors that unchanged base path for one scene.

All 18 guarded stages finished successfully and left no live child process in
their sampled groups. Maximum sampled peak was 24.3214 GiB, minimum estimated
cache-aware headroom was 22.7736 GiB, and observed stage swap growth was zero.
The 10% floor, 30 GiB native cap and 64 MiB swap-growth stop were unchanged.
Both inference phases in every receipt report zero tracked encode allocations,
readbacks and preterminal waits, with one existing terminal drain per phase.
These counters exclude opaque MPS resources; sampling cannot bound short spikes.

Focused verification passed: 60 Python tests, 11 CPU-only Crystal examples,
Crystal formatting and a fresh guarded Metal build. The fresh upstream merge
check compared 302,113,792 BF16 values in blocks 0 and 1 with zero mismatches.
The prior authenticated all-block merge and same-latent PixelUMM evidence is
indexed by `.landmark/evidence/2026-10-31-hy-nomas-cinematic-quality.json`.
No new complete Cinematic DiT trajectory or ten-second intermediate-output
gate was executed. The source-MPS oracle is not unmodified CUDA/FlexAttention.
The existing 2e-5 absolute and relative tolerances were not widened.

Retained artifacts are under
`/opt/mldata_ssd/cogni-ml/cinematic-drafter-20261101.CSoblFK7/`:
`experiment-summary.json` binds the source, executable, input, model and guard
identities; `pairs/comparison.json` binds the six authenticated comparisons;
`pairs/reel-60s.mp4` and `pairs/index.html` provide playback. Each pair has the
base on the left and Cinematic on the right. The pair builder rejected an
incomplete suite, passed eight hostile geometry/publication checks, and then
published the completed directory atomically. The final reel decodes to 1,440
frames at 1024x320/24 fps; each pair decodes to 240 frames.

Inspection of paired stills at 0, 3, 6 and 9 seconds found stronger surface
texture and contrast with Cinematic, changed composition, and one visible
walker rather than multiple walkers in the doughnut scene. Balloon count still
changes, and both dog variants face left despite the requested left-to-right
motion. These are sampled visual observations, not full-motion, physics or
general quality qualifications. Preserve all first takes for user review.

The curated evidence packet is
`.landmark/evidence/2026-11-01-hy-nomas-cinematic-drafter.json`. Requalify after
producer/validator source, binary, model/cache, inputs/noise, geometry/history
or runtime changes; repeat timing after host/device-state changes. Missing or
changed local artifacts invalidate their use as evidence. Cinematic sequence
checkpoint/capture/restore remains explicitly unsupported.

### VACE integration boundary

The official [VACE README](https://github.com/ali-vilab/VACE/blob/main/README.md)
supports text, source video, masks and reference images; its Wan renderer
loads a separate VACE model/control path. A practical CineStudio branch is
Hy-Nomas draft pixels → prepared depth/pose/layout or editing masks → Wan VACE
render. An editable 3D draft could provide corrected motion/depth directly.
This is a proposed separate renderer, not an implemented Hy-Nomas feature.

VACE cannot be attached as this Cinematic LoRA: direct Hy-Nomas conditioning
would require a compatible trained control branch and representation. Pixel
input is the existing interchange boundary; native continuation latent bundles
are not interchangeable. VACE is not a physics solver, and transferring a bad
draft trajectory can preserve its error. The official
[user guide](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md) recommends
roughly five-second clips and sequential extension for long work; seams and
temporal preservation would need separate testing on our ten-second scenes.

## Design-sealed first-take frontier (2026-11-01)

The requested output is six independent photorealistic ten-second clips, with
their prompts, actual generation times and an offline gallery. Reuse the
existing native sequence generator at two five-second segments, 512x288,
24 fps and four Euler steps. Original BF16 weights, F32 live history and
once-per-scene text caching remain the control. No model kernels, rounding
boundaries, arithmetic tolerances or command-chain synchronization change.

New prompts require authenticated Qwen state-26 conditioning. The current
sequence CLI consumes saved conditioning; preparation and provenance must be
reported separately from native DiT/PixelUMM execution. Each scene receives
its own input bundle and fresh process. The input format fixes seed-42 noise
banks, so this is a controlled prompt variation, not six independent random
samples. Preserve the first take of every scene, including visible defects.

This is CAUTION for process orchestration and provenance. The strongest
failure mode is a successful-looking gallery that relabels stale conditioning,
truncated video, a failed guard or output from another scene. Bind the intended
prompt and exact input manifest digest to the native receipt, raw-pixel digest
and exported MP4. Do not launch the next scene after a failed stage. Store
artifacts under a fresh directory on `/opt/mldata_ssd`; reject occupied outputs.

Only one model-owning process may run at a time. Preserve a 10% file-cache-aware
memory floor, a sampled process-group footprint cap, 64 MiB maximum swap growth
and bounded deadlines. Preflight each input's named-memory plan before model
setup. Sampling does not reserve memory or bound opaque MPS spikes. A guard
rejection is an incomplete run, not permission to weaken these limits.

DoD: hostile unit checks reject prompt/input identity mismatches, incorrect
geometry, missing drains, extra encode allocations/readbacks/waits, stale or
truncated outputs and incorrect MP4 frame/timing metadata. Each completed clip
must contain 60 clean latent frames and 240 decoded pixel frames at 24 fps,
512x288, exactly ten seconds, with eleven PixelUMM windows. Require zero
instrumented encode allocations/readbacks/preterminal waits, one terminal
drain per existing GPU phase and successful guard cleanup. Export and media
inspection occur after the GPU phases drain. Run the focused Python tests and
the guarded six-scene suite, then inspect representative frames and deliver
the actual videos. Numerical upstream parity and visual quality remain
separate from artifact integrity; full upstream video parity is not claimed.

No audio generation, weight quantization, scene selection, Studio server,
cross-scene history or full editor are admitted in this slice. Roll back by
removing the new drafter files; the native sequence generator and existing
artifacts remain unchanged. Evidence expires when the scene manifest,
authenticated inputs, native executable, producer sources, weights or exported
media change.

## Authorized prompt-planning extension (2026-11-01)

The user separately authorized GPT-6 Luna through the API key already in the
environment, and asked for an empirical evaluation of its prompts. A separate
planner may call the fixed OpenAI Responses endpoint with `gpt-6-luna`,
`reasoning.effort=medium`, a bounded output and a strict JSON schema. Medium is
a starting choice, not a measured quality/cost optimum. Official model and
Structured Outputs documentation establish the request surface:
https://developers.openai.com/api/docs/models/gpt-6-luna and
https://developers.openai.com/api/docs/guides/structured-outputs.

Planning produces an editable frozen suite with provenance; it does not
produce Qwen embeddings or launch a renderer. The default is a network-free
request preview; `--execute` performs one bounded call, with no silent retry,
model fallback or automatic render. Preserve the six scene IDs/order, native
geometry and guard settings. Reject refusals, incomplete/malformed results,
duplicate IDs, extra scenes and unsupported action-tail scheduling. Never
print or persist the credential, authorization headers or a raw environment.

DoD: mocked transport/negative parser tests, one authorized real six-scene
planning call with its usage/model/effort receipt, a manual intent and
feasibility review before rendering, then the same first-take media gates.
The rendering process does not receive the planner credential. Assess intent
retention and motion/camera clarity separately from syntax; a model-generated
plan does not establish better video quality or isolate planner causality.

An initial real call was rejected by an overly literal local keyword check
before its body was saved. That discarded response cannot support a semantic
quality judgment. The planner now saves a bounded response-body artifact,
including on post-response validation failures, without request headers. Style
is validated as a typed field; shot continuity and intent are reviewed from
the actual content rather than inferred from two English keywords.

## Local use

Planning and rendering are separate steps. Set `OPENAI_API_KEY` in the shell
environment; do not put it in a command argument, suite file or browser code.
First preview the bounded request without a network call:

```sh
python3 -B examples/hy_nomas_plan.py \
  --suite examples/data/hy_nomas_photorealistic_drafts.json \
  --output /opt/mldata_ssd/cogni-ml/my-luna-suite.json
```

Add `--execute` to send one authorized API request. Medium effort is the
default. Inspect the resulting suite and adjacent receipt/response artifact
before rendering. They use fresh paths and are never silently overwritten.
The response artifact is preserved on local post-response rejection so the
same paid result can be inspected. There is no automatic paid retry.

Build the native producer with a scratch build directory, then give the
drafter a fresh run directory:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/draft-build hy_nomas_sequence_generate
python3 -B examples/hy_nomas_draft.py \
  --suite /opt/mldata_ssd/cogni-ml/my-luna-suite.json \
  --generator /opt/mldata_ssd/cogni-ml/draft-build/hy_nomas_sequence_generate \
  --run-dir /opt/mldata_ssd/cogni-ml/my-draft-run
```

`--reference-python`, `--upstream` and `--checkpoint` can override the local
prepared asset paths. The runner uses existing local weights and does not
download them. Its `index.html` and `results.json` expose completed clips and
failures as the suite progresses. Rendering never calls the cloud planner.
Each MP4 is a silent ten-second diagnostic draft; sound and UI integration
are separate work.

Focused checks:

```sh
python3 -B -m unittest spec/hy_nomas_plan_test.py \
  spec/hy_nomas_draft_test.py spec/hy_nomas_draft_results_test.py
```

## Retained experiment (2026-11-01)

Artifacts are under
`/opt/mldata_ssd/cogni-ml/hy-nomas-drafts-20261101.hj3x23ur`.
The retained plan is `luna2-suite.json`, with SHA-256
`43a1e368dbe9c8046ea8da24f99cf0f7a57d176a2fce203288c913587172c838`.
Its response/usage receipt records `gpt-6-luna`, medium effort, 1,334 input
tokens and 1,250 output tokens. This usage covers the retained response only;
the earlier discarded response is not included.

Manual text review found the core actions retained in five scenes. Luna
removed the water scene's stop-pouring and set-down actions. That scene was
retained as a reduced pouring-only diagnostic, rather than counted as full
intent preservation. The editable planning stage is useful, but these six
examples do not establish that medium effort is optimal or that Luna improves
video quality over the original prompts: no planner A/B experiment was run.

The completed run is `suite-run3/results.json`; `suite-run3/index.html` is its
offline gallery. `suite-run3/preview-reel.mp4` concatenates the six ten-second
exports without re-encoding. The successful native executable was freshly
built, with its command and hash retained. Selected producer source snapshots
are in `selected-source-snapshot/`; automatic correspondence between all build
inputs and the executable is not proven. Post-run gallery/report corrections
do not relabel those producer snapshots or change the videos.

| Scene | Total seconds | Sampled process peak, GiB | Sampled visual finding |
| --- | ---: | ---: | --- |
| Portrait | 294.46 | 24.23 | Consistent face/framing; blink and full head turn are not established by the samples. |
| Pouring water | 315.55 | 24.31 | Stream and filling glass are visible; the reduced prompt keeps pouring through the end. |
| Running dog | 301.74 | 24.22 | Dog faces left despite the left-to-right request; trajectory and continuous gait remain unverified. |
| Balloon car | 295.59 | 24.32 | No yellow balloon is visible; the finish looks overly smooth and the requested camera angle is uncertain. |
| Coast flight | 295.66 | 24.22 | Photographic-looking coastal samples; continuous flight smoothness remains unverified. |
| Desert donut | 300.90 | 24.32 | Clear failure: one requested walker becomes several people with inconsistent clothing. |

Total scene time was 1,803.91 seconds for sixty seconds of video; the median
was 29.83 seconds per video second. These times include each scene's
conditioning, loading, native execution and export. They exclude the build,
API planning, post-run inspection and earlier memory-limited attempts. No
output from those interrupted attempts was promoted. The largest sampled
process-group footprint was 24.32 GiB, and minimum sampled file-cache-aware
headroom was 20.34 GiB. All 36 guarded stages exited zero with empty process
groups at their final samples; the 10% floor and 30 GiB native cap were retained.
These sampled observations do not bound between-sample MPS spikes.

Each exported clip contains 240 frames, 512x288 at 24 fps for ten seconds.
Raw-pixel and MP4 digests bind the exports to their receipts. All six native
receipts report zero instrumented encode allocations, readbacks and
preterminal waits, with one terminal drain per GPU phase. The metric excludes
opaque MPS resources and terminal exports. Conditioning uses authenticated
saved upstream Qwen state-26; live native text encoding and full upstream
video parity remain outside this result. No arithmetic or tolerance changed.

`visual-review.json` records inspection of twelve frames per scene, including
frames 119/120/121 around the five-second boundary. This supports sampled
composition and visible prompt-fidelity findings, not full-motion quality or
temporal continuity. The same-lineage adversary review is correlated evidence.
`experiment-summary.json` indexes the measured times and guard cleanup checks.
The focused suite passes 36 tests. The actual final gallery CLI also exits
zero for all six videos, validating raw-pixel hashes and MP4 hashes/metadata.
That CLI check exposed a non-serializable `Path` in the public report; the
report now explicitly emits string paths, with a regression that reproduced
the failure before the fix. `postrun-review.json` binds those post-run report
changes and checks to the retained producer snapshots without changing their
source identity.

Recheck all media and regenerate the gallery with the current report code:

```sh
python3 -B examples/support/hy_nomas_draft_results.py \
  --suite /opt/mldata_ssd/cogni-ml/hy-nomas-drafts-20261101.hj3x23ur/luna2-suite.json \
  --results /opt/mldata_ssd/cogni-ml/hy-nomas-drafts-20261101.hj3x23ur/suite-run3/results.json \
  --output /opt/mldata_ssd/cogni-ml/hy-nomas-drafts-20261101.hj3x23ur/suite-run3/index.html \
  --validate-final
```
