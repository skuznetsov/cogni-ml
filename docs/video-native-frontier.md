# Native speech-to-video frontier

Status: IN_PROGRESS; resident block gates and complete Wan BF16 transformer-weight
residency pass, while complete video generation remains open.

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
- Native text/audio/image conditioning and dynamic embeddings, motion packing,
  per-example masks/lengths, both video VAEs and complete denoising/scheduling.
- Full 5–10 second output, lip sync, accumulated quality and controlled end-to-end
  latency, throughput and memory comparisons.

Wan adaptations retain Apache-2.0 notices; segregated LTX adaptations/fixtures
carry the LTX-2.x Community License. See `third_party/VIDEO_NOTICES.md`.
Refresh this evidence on native/oracle/upstream/checkpoint/configuration,
precision/tolerance, runtime/device/driver changes, or missing/changed artifacts.
