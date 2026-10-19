# Native speech-to-video frontier

Status: IN_PROGRESS; resident block and scoped weight-loading gates pass, while
complete video generation remains open.

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
  `da0b690960026eab794b8ceaf841233cf7064478`. Official config/weight access
  returned HTTP 403 for the current account. Actual LTX checkpoint configuration
  and tensor metadata are not admitted by the synthetic probes below.

## Admitted block boundary

All admitted block neural math uses F32 Metal kernels and resident buffers.
Weight byte I/O and preparation of fixed constants occur at setup. A caller-owned
serial encoder can perform weight conversion, a block, a chained block, and
scratch reuse before one terminal completion/read phase. No block/loader encode
method submits, waits, downloads activations, or allocates a Metal buffer.
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

`src/ml/video/checkpoint.cr` reads bounded safetensors headers without mapping a
whole shard or using the engine's CPU dequantizer. It rejects duplicate keys,
unknown tensor fields, unsupported dtypes, invalid shapes, overlapping/missing
payload ranges, changed files, missing/duplicate bindings and over-budget plans
before GPU allocation. Exposed descriptors/plans are defensive snapshots.

Supported source dtypes are BF16, F16 and F32. Setup copies selected raw bytes
into GPU-visible staging; `weight_decode.metal` converts them into resident F32
destinations in the caller's encoder. Raw staging and destination buffers must
outlive command completion. The caller releases staging afterward; Arena owns
the F32 destinations. File identity/size/mtime checks assume immutable local
files and are not an adversarial cryptographic file-locking guarantee.

The explicit plan includes raw staging, F32 destinations and caller-reserved
other residency. PreparedWeights checks existing buffers in its supplied Arena
against that reserve. It does not independently enumerate all other arenas or
system processes; total residency remains a caller contract with run-time guards.

- Official Wan subset: 1,017,443,328 raw tensor bytes and 2,034,886,656 F32
  destination bytes. With 1 GiB reserve the plan fits its 8 GiB budget. Only
  selected ranges from three official shards were fetched, not the whole model.
- LTX builder-width synthetic subset: 1,546,306,304 bytes each for raw staging
  and F32 destinations. Including 1 GiB reserve gives 4,166,354,432 planned bytes
  under the 4 GiB budget. This is accounting, not an observed physical peak.
- Wan's full checkpoint index declares 32,591,511,218 weight bytes. Assuming
  all indexed weights are BF16 (only three selected shard headers were inspected,
  all BF16), expansion implies about 65.18 GB of F32 destinations, above the
  memory left after the required 15% floor on this 64 GiB host. This is an
  estimate, not a full-shard residency measurement. Whole-model startup remains
  unadmitted; native lower-precision kernels or a separately qualified residency
  strategy are needed first.

The loader passed 10 examples, including all three GPU conversions followed by
projection in one encoder, repeated encoding without new GPU allocations, and
malformed/mutable/over-budget negative cases. The Wan fixture additionally pins
the exact 39 source/target/shape/dtype mappings and the selected BF16 payload SHA.
LTX bindings currently describe an upstream block state_dict; actual checkpoint
shard prefixes, dtypes and flags require authorized metadata/weight access.

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
  crystal build "spec/${unit}_spec.cr" -o "$video_tmp/build/${unit}_spec" \
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

Expected: respectively 10/10/6/2 examples, zero failures, errors or pending cases,
all named values finite and within the fixed gate, unchanged cumulative GPU
allocations during encode, and chaining without an intermediate host wait/read.
Reduced gates are regenerated without `--production-width`; LTX additionally
uses neither gate, `--gated-attention`, `--video-gated-attention`, and
`--audio-gated-attention`, with cross-AdaLN/video-FF-no-bias enabled in the latter
three. Missing fixture environment variables produce pending oracle tests;
pending results are not parity passes.

All GPU runs are serial under run_safe: configured 15% free-system-memory floor,
finite deadlines and tree-RSS caps. Logs establish configured guards and successful
runs, not an observed minimum free-memory percentage or physical peak. No full
model, Studio or service was started. Diagnostic snapshots and terminal reads
are probe overhead, not generation timings.

## Evidence, review and remaining frontier

The current private index is
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/production-block-gates.json`.
SHA256: `a4582e1834743252f27056fdd989508e24ade38f29c745d0109fbbdc8902501c`.
It pins native/oracle source, binaries, fixture/weight payloads, command receipts,
red discriminators and final logs. It is a curated evidence index, not a
transitive reproducible-build attestation. The preceding reduced milestone is
retained separately in `evidence/final-block-gates.json` (SHA256
`40be04204a58d9b00e3b1f7f2f6a838a56000ff807c7145877fe5dd675d3279c`).
Correlated mini-Quadrumvirate review returned ROBUST for the exercised parser,
conversion, block/gate configurations, reduction tails, and caller-owned encoder
contracts. It does not admit the following open capabilities:

- Authorized actual LTX config/weights, prefixes/dtypes and real-weight parity.
- Remaining distinct transformer layers and whole-request resource accounting;
  native lower-precision kernels require a separate accuracy gate.
- Native text/audio/image conditioning and dynamic embeddings, motion packing,
  per-example masks/lengths, both video VAEs and complete denoising/scheduling.
- Full 5–10 second output, lip sync, accumulated quality and controlled end-to-end
  latency, throughput and memory comparisons.

Wan adaptations retain Apache-2.0 notices; segregated LTX adaptations/fixtures
carry the LTX-2.x Community License. See `third_party/VIDEO_NOTICES.md`.
Refresh this evidence on native/oracle/upstream/checkpoint/configuration,
precision/tolerance, runtime/device/driver changes, or missing/changed artifacts.
