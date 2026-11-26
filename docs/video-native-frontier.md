# Native speech-to-video frontier

Status: IN_PROGRESS; resident block gates and complete Wan BF16 transformer-weight
residency pass, while complete video generation remains open.

## Active transition: native Wan S2V request

The current request is a complete five-second 480p Wan S2V generation, with a
cold process and an immediate repeated request measured separately. This is a
CAUTION implementation boundary, not an extrapolation from the two-token block
or weight-residency measurements. Native conditioning, the full transformer,
the flow scheduler and VAE must all be exercised before reporting generation
time. A precomputed Python conditioning tensor is a component fixture, not a
complete native pipeline. Tokenization, asset byte I/O and terminal video-file
encoding may run on the host; neural activations remain on GPU.

The initial inventory found no complete S2V forward, native UMT5/Wav2Vec2,
native Wan VAE or denoising runner. Native Wan VAE (`9395eb95`), GPU UniPC
(`a4622abd`) and the causal audio conditioning projection (`4409c0f0`) now pass
their scoped component gates; see their component documents for dimensions and
remaining limits. The native transformer is committed in `5b3f60f8`, with
serial block/audio scratch sharing in `4000d9b2` and opt-in tiled attention in
`a079a7ce`. The complete text encoder passes its captured gate; the audio
encoder retains recorded numerical differences. Existing block math and the per-value finite
2e-5 absolute/relative gate remain the baseline. Component fixtures compare the
same inputs against the pinned upstream modules, including first-call temporal
caches, real-width projections, sequence layouts and scheduler coefficients.

The real-width UMT5 block now matches all 16 captured checkpoints bitwise when
using the source GELU operation order and prepared rank-three Graph attention.
The full 24-layer, 512-token native gate passes 39 intermediate tensors and
the masked context in 23.19 seconds with one terminal wait. This is a component
test duration, not video generation speed. Audio waveform normalization preserves eager operation
boundaries, making the waveform and all seven feature-extractor stages bitwise
equal to upstream. All first-layer captured points pass the unchanged gate,
but the complete interpolated hidden-state output still fails. Source-ordered
bucket interpolation removes the first three hidden-state failures; the first
remaining rejected state is layer 3, and the maximum captured absolute drift is
1.274109e-3. All captured values are finite. These are
separate qualification boundaries: a passing first block cannot admit the
complete encoder.

The 40-layer real-weight stack now has a complete exact-prefix comparison at
reduced spatial dimensions. With prepared MPS projections and typed GPU bias,
all 104 captured points are finite; 21 of
31,948,800 values fail the unchanged gate, with a maximum absolute difference
of 2.899170e-4 on the prepared query-chunked MPS route. These differences are recorded separately from the audio
encoder differences. Neither comparison bounds a full-spatial denoising
trajectory or establishes visible quality. See `docs/video-numerical-drift.md`.

Upstream CFG zeros the input audio features before the causal audio encoder,
not the encoded local/global outputs. A targeted production-width zero-input
oracle and native comparison passed all four examples under the unchanged
gate (`causal-audio-null-native.log` in the external evidence directory).
Both encoded outputs are nonzero because the encoder includes biases and
nonlinearities. The pooled pipeline must retain these separate null-branch
outputs; sharing the positive audio outputs is a semantic error, outside the
recorded numerical-drift exception.

The first full-spatial DiT block probe failed Metal completion with `Impacting
Interactivity`. A controlled full-attention comparison isolates a command-size
boundary: 4,096 query-tile/head groups per root fail; 320 groups complete the
same Q=K=36,608, H=40, D=128 geometry with all 187,432,960 synthetic values exact.
The chain overload now defaults to 320 without changing query reductions.
With that boundary, one real BF16 block 0 plus audio injector 0 completes:
36.952 seconds summed GPU interval, 1,148 segments, one terminal host wait,
zero activation readbacks, and a GPU scan confirms finite outputs. This is a
cost diagnostic with procedural inputs, not upstream full-shape parity or
a completed generation. See `docs/video-mps-projection.md` for receipts.

The opt-in prepared MPS query-chunk route subsequently completes all 72 chunks
at the same full attention dimensions with 13 command segments and one terminal
wait. Its analytical GPU control checks every one of 187,432,960 values exactly.
A separate repeat measured 10.76 seconds wall time and 3,078,882,120 bytes peak
process memory footprint with no swaps. This is a standalone synthetic attention
probe, not a full request or a proof of the private-workspace peak when every
pipeline plan is resident. See `docs/video-attention-mps.md`.

A subsequent real BF16 block-0 plus audio-injector-0 probe uses that same
512-row chunk route at sequence length 36,608. Both block and injector outputs
pass the GPU finite scan, with one terminal wait, zero activation readbacks,
and 589 command segments. The first run reports 13.676 seconds summed GPU
interval; an independent process repeat reports 14.708 seconds, 15.06 seconds
wall time, and 2,418,591,400 bytes peak process memory footprint with no swaps.
The explicit weight/input/scratch plan including a 1 GiB reserve is
16,330,007,044 bytes; process footprint is a separate OS counter and does not
replace that resource accounting. These procedural-input cost diagnostics
(`dit-full-shape-mps-chunk512-{probe,peak}.log`) establish neither full-shape
upstream parity nor complete-video performance.

An opt-in tiled F32 projection primitive (`37913d80`) preserves BF16 parameter
storage and passes all 24 pinned MPS dtype/tail cases at the unchanged gate.
Four alternating pairs at 64 rows, input width 5,120 and output width 13,824
give median GPU intervals of 18.274 ms for the tiled kernel and 135.890 ms for
the scalar baseline, a 7.44 ratio. These synthetic resident-weight intervals
do not measure a full block or video, and this arithmetic route is not yet
enabled by default in the transformer. Optional prepared MPS projections share
a 270 MiB F32 weight expansion stage across fifteen plans. The actual typed-bias
path passes its projection fixture and official block gate; its refreshed
full-depth finite drift is recorded above. Synthetic local MPS/tiled ratios
range from 6.77 to 20.29 at 64/512/2,048 rows. These component intervals cannot
establish request speed. No five-second cold/repeat measurement is available
yet. See `docs/video-mps-projection.md`.

The official first request uses `drop_first_motion=True`: it has no FramePack
history tokens and zeroes the encoded no-pose condition on GPU. Its 480p working
canvas is 832x512, with 36,608 transformer tokens, including the reference frame.
Retained-history and empty-history paths require separate oracle gates. The
temporal recipe decodes 22 latents to 85 raw frames, selects raw frames [4,84)
and delivers exactly 80 frames at 16 fps after a centered 480-row spatial crop.

The runner must resolve spatial alignment and the upstream initial three-frame
drop explicitly. It must report internal generated frames, delivered frames,
resolution, frame rate, steps, guidance, seed and exact source/weight revisions;
rounding a shorter clip to five seconds is rejected. Cold timing includes
resource/weight setup through the terminal output; repeated timing reuses only
resources actually retained under the memory budget. OS file-cache state is
reported rather than implicitly equated with a cold process.

Before a full request, record total live weights and planned activation/cache
storage across all stages. Weight residency alone is insufficient. GPU runs
remain serial, guarded by run_safe with a 15% free-system-memory floor, finite
deadlines and stage-appropriate RSS caps. No Studio or service is started.
Caller-owned chains can asynchronously continue between bounded GPU phases;
neural encode methods allocate no buffers, wait on no commands and read no activations.
Diagnostic reads are terminal and excluded from generation timing.

`8450cf45` admits the bounded serial stage-pool and asynchronous Metal I/O
infrastructure: twelve prepared stage transitions pass with one terminal wait
and no encode-time `MetalBuffer` creation. The full model runner must still
qualify these transitions with actual model resources, numerical gates, and
retained cold/repeat requests. Allocation counters cover explicit wrappers;
private execution workspace also remains subject to the external memory guard.

The unified header-only plan now estimates 44,664,405,284 bytes including a
1 GiB caller reserve, with one 42,842,423,296-byte pool shared by sequential
weight-plus-scratch phases. A compiled plan snapshot accepts a 44,032 MiB
budget and rejects 40,960 MiB before Metal initialization. These calculations
do not measure actual allocation cursors or private execution peaks. The
snapshot and `pooled-memory-plan.json` / `pooled-memory-plan-rejected.json` receipts are in the
external `wan-full` evidence directory.

The first full invocation stopped during resource preparation, before payload
loads or GPU work: the causal-audio checkpoint has 25 BF16 mixture weights
(50 bytes), while Arena's raw word-storage view requires a multiple of four
bytes. The runner now pads that view to 52 bytes, retains the exact 50-byte
StageIO extent, and decodes 25 values using the checkpoint dtype. Page-rounded
pool capacity is unchanged. A standalone extraction of the actual stage-loader
class passes six BF16/F16 scalar, odd-count and even-count GPU cases, with exact
outputs, one terminal wait and no encode-time MetalBuffer allocations
(`raw-staging-native.log`). This corrects setup; the complete request remains
unmeasured.

The next full invocation reached execution but timed out at its first T5
readiness wait. A staging-only reproducer failed on the same 11.36 GB load
without T5 math. Commit `2210c5cc` splits logical plans into at most 512 MiB
I/O batches with asynchronous compute-root retirements between them. The real
T5 staging-only probe then passed with 22 batches/roots, one terminal wait,
484 exact boundary samples, 12.39 seconds external wall time and no swaps.
The common staging gate passes 27 examples, including rejection of a partially
submitted stage after its submitted batches drain. This establishes the bounded
staging path, not a complete request or strict neural parity. The runner also
preserves the first execution exception if cleanup reports pending I/O. See
`docs/video-stage-io.md` for the evidence and refresh boundary.

The first batched full invocation then stopped before denoising because the
runner placed caller sequence/time/output buffers in the same Arena passed as
transformer scratch. The scratch/input overlap guard correctly rejected this
ownership violation. Caller activations now have a separate retained Arena on
the same phase cursor, with unchanged allocation order/capacity and explicit
release after terminal drain. The legacy caller also separates its inputs and
constants. A verbatim extraction of the pooled runner's allocation block first
reproduced the failure, then passed the pinned tiny patch/text/time/modulation
oracle with the unchanged tolerance, one terminal wait and zero encode-time
MetalBuffer allocations. It checks the exact 27,066,368-byte reduced-case pool
cursor, successful view release, and continued rejection of a deliberate alias
(`pooled-transformer-bindings-before.log`,
`pooled-transformer-bindings-native.log`). This is a setup regression check;
the complete denoising loop and full request remain unmeasured.

The VAE scratch-capacity correction in `bfaad7c3` is required before a full
request: the decoder's nearest upsample writes the larger channel count before
projection. Both current 16x24 execution modes pass the unchanged upstream
gate, and 147 paired GPU snapshots are bit-identical. A separate 832x512 probe
with the full 22-latent workspace decodes two duplicated reference latents into
five frames in 200.13 seconds process wall time (198.31 seconds summed GPU
intervals), with one terminal host wait, no non-finite values, no encode-time
MetalBuffer allocations and a 16.72 GB peak process memory footprint. This is
a spatial/cache cost control, not generated video or a full 85-frame decode.
See `docs/wan-vae.md`.

Run receipts report each request through final raw output writing and receipt
hashes, excluding receipt serialization and resource release. The first
request also reports initial setup plus that request; the second reuses the
pool and plans but reloads checkpoint phases. Neither is a whole-process wall
timer. An external `/usr/bin/time -l` measurement covers the complete process;
MP4 packaging is measured separately. No file-cache eviction is implied.

The ordered implementation/DoD is: (1) native conditioning and VAE component
fixtures; (2) distinct-layer full-forward fixture and GPU scheduler fixture;
(3) bounded full-shape allocation and command-cost probe; (4) short complete
request compared with upstream; (5) the guarded five-second cold/repeat pair,
with output and sampled memory/timing receipts. Strict component qualification
requires its focused Crystal spec and pinned upstream fixture to pass with no
pending oracle cases. The strongest fallback checks are same-input intermediate
comparisons, exact output geometry, cumulative allocation counters during
encode, and two retained-resource requests. A failed numerical gate or memory
floor stops the dependent full request by default. The user's 2026-10-19
authorization permits an explicitly labelled experiment using the exact finite
audio-conditioning and full-depth stack drift ledgers; strict parity remains
open and tolerances stay unchanged. An experimental execution is the means of
obtaining full-request evidence, not a claim that an unexecuted full request
already passed. See `docs/video-numerical-drift.md`. Runtime, shape, source,
checkpoint and memory failures cannot use this exception. It does not admit
CPU neural fallback or an invented timing. Rollback removes the new opt-in
pipeline modules/runner while preserving the admitted block and loader routes.

## Scope and source anchors

Implement opt-in native Metal backends for Wan 2.2 S2V and LTX 2.5.
The eventual comparison uses one Qwen Image portrait and one DRAMA3 WAV for a
5–10 second speaking, laughing, head-turning scene. No speed or quality winner
has been measured.

- Wan source: `Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff`.
- LTX source: `Lightricks/LTX-2@e98f6d927d83c92ef2be6aec76c8440b19306e39`.
- Initial block milestone: `79e8bf85`; underlying engine base:
  `6263c34d82bc733cf4d861531dde2e47a7a99e13`.
- Wan checkpoint: `Wan-AI/Wan2.2-S2V-14B` at
  `dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b`.
- LTX checkpoint access was tested at `Lightricks/LTX-2.5` revision
  `49b093a39c19169a8a81721dcc541e07e945232f` and
  `Lightricks/LTX-2.5-Diffusers` revision
  `da0b690960026eab794b8ceaf841233cf7064478`. After acceptance of the terms,
  authenticated access to the primary repository's
  `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` succeeds
  (42,018,190,584 bytes). The Diffusers config still returned HTTP 403.
  This is an access/size check, not a download or admission of actual LTX
  checkpoint configuration, tensor metadata, or real-weight parity.

## Admitted block boundary

All admitted block neural math uses F32 Metal arithmetic and resident buffers.
Wan parameters may remain in original BF16 storage; consumers expand operands
to F32 on GPU. Weight byte I/O and preparation of fixed constants occur at setup.
A caller-owned serial encoder can perform weight conversion, a block, a chained
block, and scratch reuse before one terminal completion/read phase. No
block/loader encode method submits, waits, downloads activations, or allocates
a Metal buffer.
Cumulative allocation guards cover temporary as well as retained GPU buffers;
they make no zero-host-allocation or GC claim.

The fixed gate remains
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)` at every compared value,
with both operands finite and exact checkpoint-name sets required. No positions
were excluded or tolerances widened. Existing Qwen/DRAMA gates are unchanged.

| Slice | Dimensions and inputs | Weight evidence | Upstream gate |
| --- | --- | --- | --- |
| Wan block 0 + first audio injector/AdaIN | C=5120, H=40, D=128, FF=13824; B=2, sequence=2, one video token and one motion token per sample | 39 exact official BF16 tensors, plus independent synthetic F32 stress weights | All 35 checkpoints, repetition, audio effect, and resident chaining pass |
| LTX source-builder width | Video C=4096, H=32, D=128, FF=16384; audio C=2048, H=32, D=64, FF=8192; video/audio tokens=2/3 | 78 synthetic F32 bindings from pinned upstream block state_dict | 22 checkpoints × 2 input cases × 2 chained block applications = 88 array comparisons pass |
| LTX reduced variants | Video/audio C=32/16; independent video/audio attention gates; cross-attention AdaLN and bias-free video FFN | 68/78/72/72 synthetic F32 bindings for neither/both/video-only/audio-only gates | All four variants pass the same 88 comparisons each |

Wan widths, Q/K normalization, cross-attention normalization and AdaIN settings
match the pinned official config and selected tensor headers. The reduced Wan
fixture remains a guard for longer token lists, two distinct batch members,
large video coordinates, negative-time motion suffixes, and C=320 AdaIN coverage.
The slice requires fixed unpadded lengths shared across the batch and equal
video-prefix contracts; masks and per-example lengths remain open.

LTX builder dimensions are established from pinned source defaults. The large
probe deliberately enables cross-attention AdaLN and both modality gates, and
sets video FFN bias=false/audio bias=true. These are synthetic overrides, not
verified official checkpoint settings. Reduced variants also exercise the
asymmetric modality gates. Prompt scale/shift tables use `prompt_timestep=None`;
native dynamic timestep/prompt embedding producers remain open. Chaining reuses
the same selected block weights, not two distinct checkpoint layers.

The oracles execute pinned upstream modules on PyTorch 2.8.0 MPS with CPU neural
fallback disabled. Declared adapters replace CUDA-only attention entry points,
materialize contiguous Q/K/V on GPU for SDPA, and apply Wan FP64 RoPE coefficients
in F32. This certifies the stated F32 path, not CUDA flash or native BF16 math.

## Scoped checkpoint loading

### Original BF16 storage with F32 accumulation

`WeightPlan` accepts `resident_dtype: WeightDType::BF16`; its default remains F32.
Native BF16 mode admits only original BF16 sources and copies their raw bytes
directly into final Metal buffers at setup, with zero staging bytes and no F32
parameter mirror. F32/F16 inputs are rejected rather than silently quantized.
Activations, normalization arithmetic, the 16-FMA-lane projection reduction,
and outputs remain F32. Every learned Wan parameter is typed, including biases,
norm coefficients and modulation tables. All pipelines are prepared at setup.

The official 39-tensor Wan fixture passes all 35 upstream checkpoints in both
storage modes. A separate comparison of 38 captured native arrays, including
scratch reuse and resident chaining, finds all 817,152 F32 values bit-identical
between BF16 storage and F32-expanded weights. Chained outputs in this comparison
are native-versus-native controls, not an additional upstream layer oracle.
The original F32 Wan and LTX fixture matrix also passes unchanged tolerances.
Mixed linear/norm regressions cover all weight/bias dtype pairs, reduction tails
and cancellation at widths 15/16/17/31/32/33, aliases and invalid lifetimes.

`src/ml/video/checkpoint.cr` reads bounded safetensors headers without mapping a
whole shard or using the engine's CPU dequantizer. It rejects duplicate keys,
unknown tensor fields, unsupported dtypes, invalid shapes, overlapping/missing
payload ranges, changed files, missing/duplicate bindings and over-budget plans
before GPU allocation. Exposed descriptors/plans are defensive snapshots.

In default F32 mode, supported source dtypes are BF16, F16 and F32. Setup copies
selected raw bytes into GPU-visible staging; `weight_decode.metal` converts them
into resident F32 destinations in the caller's encoder. Raw staging and
destinations must outlive command completion. The caller releases staging
afterward; Arena owns destinations in either mode. BF16 encode validates
lifetimes/identities and emits no conversion dispatch. File identity/size/mtime
checks assume immutable local files and are not an adversarial cryptographic
file-locking guarantee.

The explicit plan includes staging, typed destinations and caller-reserved
other residency. `require_all: true` additionally rejects missing or multiply
bound source tensors. PreparedWeights checks existing buffers in its supplied
Arena against the reserve, including typed allocations whose public-map entries
were replaced or deleted. It rejects reused names before allocating staging.
Typed identity checks prevent consumers from interpreting a replacement with
stale dtype metadata; removal/release frees both current and original buffers.
Rollback restores allocation counts/bytes while preserving caller buffers.
It does not independently enumerate all other arenas or system processes;
total residency remains a caller contract with run-time guards.

- Official Wan subset: 1,017,443,328 raw tensor bytes. Native BF16 uses exactly
  that many destination bytes with zero staging; default F32 uses 2,034,886,656
  destination bytes plus raw staging. Both storage modes pass the same oracle.
- LTX builder-width synthetic subset: 1,546,306,304 bytes each for raw staging
  and F32 destinations. Including 1 GiB reserve gives 4,166,354,432 planned bytes
  under the 4 GiB budget. This is accounting, not an observed physical peak.
- All four pinned Wan shard headers match the complete index: 1260 BF16 tensors,
  exactly 32,591,511,218 bytes (30.35 GiB). F32 expansion would require twice
  those destination bytes, before staging/activations/encoders.

The current loader passed 18 examples, including all three GPU conversions
followed by projection in one encoder, repeated encoding without new GPU
allocations, and malformed/mutable/over-budget/lifetime/rollback negative cases.
Common primitives passed 17 examples. The Wan fixture additionally pins the exact 39
source/target/shape/dtype mappings and the selected BF16 payload SHA.
LTX bindings currently describe an upstream block state_dict; actual checkpoint
shard prefixes, dtypes and flags require authorized metadata/weight access.

### Complete Wan transformer-weight residency probe

`examples/wan_s2v_weight_residency.cr` independently checks all four pinned
headers/index coverage, loads all 1260 tensors simultaneously into live Metal
destinations, and scans every BF16 word on GPU. The bounded M2 Max run passed:

| Signal | Observed value |
| --- | --- |
| Requested live weight buffers/bytes | 1260 / 32,591,511,218 |
| Raw staging / F32 parameter mirror | 0 / 0 bytes |
| Device allocation delta after loading | 32,592,494,592 bytes |
| Admission budget / other-residency reserve | 35 GiB / 2 GiB |
| Admission / minimum sampled free-memory percentage | 80% / 30% |
| Configured stop floor | 15% |
| Setup / whole probe elapsed | 34.88 s / 50.05 s |
| GPU scan / non-finite weights | 62,930 chunks / 0 |

Setup copies checkpoint bytes; the scan uses one caller-owned serial encoder
and one terminal completion/read of 503,440 diagnostic bytes. Those operations
are not generation timings. System free-memory and tree RSS were sampled about
every 0.5 seconds; the observed minimum is not a continuous minimum or physical
GPU-memory peak. UMA RSS and Metal bytes must not be added as independent usage.
The scan-output SHA hashes modular chunk checksums, not the raw shard payloads;
header pins and the HF download receipt do not attest full-payload identity.
This establishes live GPU storage/readability of the diffusion-transformer
weights, not whole S2V inference or a complete T5/VAE/audio memory budget.

## Numerical falsifiers and repairs

The reduced milestone passed 16 combined examples before this extension. Its
large-position gate rejected a missing head divisor in Wan RoPE, then standard
Metal trigonometric error: sin(801) error 5.325852e-5 versus 1.773396e-9 using
precise sine. The production D=128 extension additionally exposed phase rounding.
Setup now splits the pinned FP64 inverse frequencies into high/low F32 parts;
GPU FMA residual/TwoSum reconstruction and precise sine/cosine angle addition
preserve the phase. The 64 RoPE pairs partition 22/21/21; the scratch frequency
key includes theta, head width and axis partition. No activation returns to CPU.

An apparent full-width attention mismatch was isolated to the PyTorch 2.8.0 MPS
oracle on these inputs. Noncontiguous transposed SDPA disagreed with independent
GPU matmul/softmax/matmul at 19,966 values. Contiguous SDPA and native attention
both passed that formula at every value, with maximum normalized errors 0.01576
and 0.01610. The oracle now materializes all three transposes on GPU. Wan also
checks SDPA against the independent GPU formula on each captured attention call
and exports audit scalars only at the terminal boundary. This is a scoped backend
discriminator, not a general claim about every MPS attention configuration.

Real-weight activations exposed a separate Metal tanh-GELU NaN at finite x=12.
For |x|>=10 the F32 tanh factor already saturates, so the approximate GELU kernel
returns x or zero before evaluating the overflowing tanh/cubic expression.
Regression values include ±10, ±12, ±100, ±10000 and ±1e20.

Synthetic full-width Wan and the second LTX block exposed long serial F32 dot
reduction error. Same-input upstream projections separated operator error from
propagated input error. The native linear kernel now uses 16 interleaved F32 FMA
accumulators with an explicit balanced final tree, retaining safe-math settings,
complete tails and the same dispatch/buffer/synchronization structure. Six
independent scalar-reference regressions cover widths 15/16/17/31/32/33.

The replacement passes every block checkpoint, plus same-input and propagated
projection checks. The closest Wan synthetic FFN result uses 0.935 of the allowed
error; this margin does not establish arbitrary-input or full-model stability.
LTX builder-width maximum absolute error is 1.073e-5; every element passes its
own combined absolute/relative limit. No speedup of this reduction is claimed.

## Reproducing the gates

CAUTION: new GPU math, parsing and lifetime contracts. Rollback is removal of the
opt-in loader/block modes; existing model routes and defaults are unchanged.
The strongest falsifiers are unchanged upstream comparisons at full head widths,
chain accumulation, tail/cancellation inputs, malformed checkpoint descriptors,
and invalid lifetimes/aliases before dispatch.

Use the external root below for source snapshots, fixtures, caches and binaries.
Oracle scripts pin source hashes and reject changed snapshots. Wan's six files
are flattened under `wan-audit/`, including the exact Diffusers v0.31.0
normalization source. LTX requires its pinned `packages/ltx-core/src/` directory.
The measured oracle environment was reused without dependency installation.

```sh
video_tmp=/opt/mldata_ssd/cogni-ml/video-scene-20261019
export TMPDIR="$video_tmp/tmp"
export CRYSTAL_CACHE_DIR=/opt/mldata_ssd/cogni-ml/crystal-cache
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0
export COGNI_RUN_SAFE_MIN_FREE_PCT=15
make BUILD_DIR="$video_tmp/build" "$video_tmp/build/bridge.o"
for unit in video_metal_ops video_checkpoint wan_s2v_block ltx2_block; do
  scripts/run_safe.sh crystal 180 4096 build "spec/${unit}_spec.cr" -o "$video_tmp/build/${unit}_spec" \
    --link-flags="$video_tmp/build/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld"
done
```

Generate production-width fixtures serially with an existing MPS-capable Python:

```sh
scripts/run_safe.sh "$oracle_python" 240 12288 \
  spec/support/wan_s2v_reference.py --upstream "$video_tmp/wan-audit" \
  --production-width --output "$video_tmp/evidence/final/wan-synthetic-width.json"
scripts/run_safe.sh "$oracle_python" 240 12288 \
  spec/support/ltx2_reference.py \
  --source-root "$video_tmp/ltx-audit/source/packages/ltx-core/src" \
  --production-width --cross-attention-adaln --gated-attention \
  --output "$video_tmp/evidence/final/ltx-builder-width-final.json"
```

The Wan generator additionally accepts `--checkpoint-file` for the official
39-tensor BF16 selected-block sidecar. The fixture records its checkpoint source
and payload SHA; retain the separately pinned extraction receipt. A synthetic
width pass alone does not establish the official-weight gate. See the executed
command receipts for the exact prepared-source/fixture paths and flags.

```sh
export WAN_S2V_BLOCK_FIXTURE="$video_tmp/evidence/wan-official-block0/wan-official.json"
export LTX2_BLOCK_FIXTURE="$video_tmp/evidence/final/ltx-builder-width-final.json"
scripts/run_safe.sh "$video_tmp/build/video_metal_ops_spec" 90 2048 --verbose
scripts/run_safe.sh "$video_tmp/build/video_checkpoint_spec" 90 2048 --verbose
scripts/run_safe.sh "$video_tmp/build/wan_s2v_block_spec" 180 8192 --verbose
scripts/run_safe.sh "$video_tmp/build/ltx2_block_spec" 180 8192 --verbose
```

Expected: respectively 17/18/6/2 examples, zero failures, errors or pending cases,
all named values finite and within the fixed gate, unchanged cumulative GPU
allocations during encode, and chaining without an intermediate host wait/read.
Reduced gates are regenerated without `--production-width`; LTX additionally
uses neither gate, `--gated-attention`, `--video-gated-attention`, and
`--audio-gated-attention`, with cross-AdaLN/video-FF-no-bias enabled in the latter
three. Missing fixture environment variables produce pending oracle tests;
pending results are not parity passes.

Repeat the official Wan fixture with `WAN_S2V_WEIGHT_DTYPE=bf16` and `f32`;
`WAN_S2V_SNAPSHOT_DUMP` writes the 38 native arrays used for the F32-bit comparison.
BF16 mode requires the actual BF16 sidecar and rejects a synthetic F32 fixture.
To reproduce the separate complete-weight probe after downloading all four
pinned official shards and their index into `$video_tmp/checkpoints/wan`:

```sh
scripts/run_safe.sh crystal 180 4096 build examples/wan_s2v_weight_residency.cr \
  -o "$video_tmp/build/wan_s2v_weight_residency" \
  --link-flags="$video_tmp/build/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld"
scripts/run_safe.sh "$video_tmp/build/wan_s2v_weight_residency" 240 40960 \
  "$video_tmp/checkpoints/wan" 35840 2048
```

All GPU runs are serial under run_safe: configured 15% free-system-memory floor,
finite deadlines and tree-RSS caps. Only the complete-weight probe additionally
records sampled system headroom/RSS; block logs establish configured guards and
successful runs, not observed minima or physical peaks. No complete inference
model, Studio or service was started. Diagnostic snapshots and terminal reads
are probe overhead, not generation timings.

## Evidence, review and remaining frontier

The full S2V request is still under construction. Its original staged draft
used blocking stage completion and is not an admitted request path. The bounded
replacement infrastructure now passes the repeated alias-overwrite gate:
page-aligned weight pools, phase-specific no-copy views, MetalIO range loads
and shared-event ordering preserve twelve distinct GPU-consumer payloads with
one terminal host wait. Its integration into the complete runner is still
open. Aggregate model/scratch/state capacity, opaque workspace, full-shape
command completion and retained repeated requests must be qualified before
the five-second cold/repeat result. Existing resident paths remain the
rollback boundary; no timing may be reported for a request rejected before
execution.

The current private receipts are under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-bf16/`:
the final-source fixture matrix, post-review lifetime tests, BF16/F32 bit
comparison, and the separate full-residency report. The curated index is
`evidence/wan-bf16/qualification.json` (SHA256
`67e48510835e6fc9425d99d53c692f8cc2f110c1037d390b5a61e36419bf2e17`);
it pins native/oracle source, binaries, fixtures, receipts and logs.
It is not a transitive reproducible-build or
full-shard cryptographic attestation. The preceding F32 production milestone
is retained in `evidence/production-block-gates.json` (SHA256
`a4582e1834743252f27056fdd989508e24ade38f29c745d0109fbbdc8902501c`),
and the reduced milestone in `evidence/final-block-gates.json` (SHA256
`40be04204a58d9b00e3b1f7f2f6a838a56000ff807c7145877fe5dd675d3279c`).
Correlated mini-Quadrumvirate review returned ROBUST for the exercised parser,
typed storage/conversion, block/gate configurations, reduction tails, mutable-map
ownership/budget guards and caller-owned encoder contracts. It does not admit
the following open capabilities:

- Actual LTX checkpoint config, prefixes/dtypes and real-weight parity; primary
  download access is now confirmed, while the Diffusers config remains blocked.
- Remaining distinct transformer layers, long-sequence/model accumulation and
  whole-request resource accounting. BF16 weight residency alone does not
  establish full-pipeline capacity or arithmetic in BF16.
- Qualification of native UMT5/XLSR conditioning, the complete distinct-layer
  transformer, and their composition with the admitted Wan VAE/UniPC/causal
  audio components; full-shape stage lifetimes and delivered-frame geometry.
  LTX conditioning, VAE and scheduling, and general per-example masks/lengths,
  remain outside the admitted block boundary.
- Full 5–10 second output, lip sync, accumulated quality and controlled end-to-end
  latency, throughput and memory comparisons.

Wan adaptations retain Apache-2.0 notices; segregated LTX adaptations/fixtures
carry the LTX-2.x Community License. See `third_party/VIDEO_NOTICES.md`.
Refresh this evidence on native/oracle/upstream/checkpoint/configuration,
precision/tolerance, runtime/device/driver changes, or missing/changed artifacts.
