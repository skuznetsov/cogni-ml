# Qwen-Image 2.1 GGUF Frontier

Status: active implementation frontier; prompt-to-PNG path admitted, with
optimization candidates gated by real-prompt parity and paired latency checks

## Local package frontier (2026-09-23)

Current frontier: one local directory and one text-to-image entry point may
compose the admitted CPU Qwen3-VL conditioning, native Metal GGUF DiT, and CPU
VAE decode. This is a packaging/launch boundary, not a new model backend.

- **Admitted target:** a versioned manifest names only package-local component
  paths, identifies the official model revision and the mixed-quant DiT GGUF,
  and declares the runtime as `hybrid`. A single command validates that package
  before starting any stage, then runs the existing three stage contracts in
  order and writes one PNG. Inputs include an explicit prompt, dimensions,
  seed, step count, and output path. The command does not download weights.
- **Rejected:** describing this route as a native Qwen3-VL or native VAE
  implementation; silently accepting an old Qwen-Image model, external path
  traversal in a static manifest, an absent component, an unknown model
  revision, or a partial output as a successful image. Image editing and
  guidance are still rejected.
- **Guard-only next track:** a text-only Qwen3-VL port may replace the reference
  encoder only after a fixture pins exact input token IDs, masks, drop index,
  and pre-final-norm hidden states from the official pipeline, followed by
  layer/output parity against that fixture. Native VAE follows separately.
- **Falsifiers:** a malformed or escaping manifest, mismatched model identity,
  missing files, wrong GGUF, failed stage, interrupted run, or an output file
  that is not a decoded PNG. A successful launcher test does not imply a speed
  or image-quality improvement. Real-weight smoke tests must disclose the
  source revision, GGUF policy, prompt, shape, and sampling settings.

The package evidence decays if the manifest schema, stage CLIs, official model
revision, GGUF contents, or upstream processor/encoder semantics change.
Rollback is to run the three already admitted stage commands directly. A
package manifest and launcher should remain outside the repository's weight
tree; no large weights are committed.
This is a local-use bundle, not permission to redistribute the model; the
[upstream Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)
sets separate non-commercial and redistribution conditions.

The package launcher is `scripts/qwen_image21_package.py`. Initialize it with
an already-downloaded official component snapshot, the pinned mixed-quant DiT
GGUF, and a separately compiled macOS arm64 Metal denoiser. It hard-links the
large files, checks available source-revision metadata and the GGUF SHA-256,
records file hashes, and atomically publishes a new directory. The package
still needs an external
Python environment and the denoiser's Homebrew dynamic libraries; it is not a
standalone application or a single-weights-file format. The tested Python
environment used Torch 2.6.0, Transformers 5.17.0, and Diffusers 0.41.0.dev0
from commit `8b3c707ebd3ec4881f4190cf42931da07eaf3b65`.

```sh
python scripts/qwen_image21_package.py init \
  --package-dir /path/to/new-qwen-image21-package \
  --model-dir /path/to/Qwen-Image-2.1-components \
  --gguf /path/to/Qwen-Image-2.1-Q4.gguf \
  --denoiser /path/to/compiled-qwen-image21-denoiser \
  --revision 790c92633540aa0cb11d9abf19eb46d861714758
python /path/to/new-qwen-image21-package/bin/qwen_image21_package.py generate \
  --package-dir /path/to/new-qwen-image21-package \
  --prompt 'red cube' --width 256 --height 256 --seed 7 --steps 40 \
  --output /path/to/new-red-cube.png
```

Generation rehashes all packaged model files and the GGUF before starting a
stage, so this integrity guard adds cold-start I/O. The manifest is editable:
these hashes detect drift relative to that manifest, not coordinated tampering
with both artifacts and manifest. The package directory must be trusted and
must not be concurrently mutated by an adversary. The launcher retains the
validated canonical root and checks its device/inode before each subprocess;
this rejects the tested root-symlink swap, but does not eliminate races during
file opening. The `generate` command sets Hugging Face, Transformers,
Datasets, and Diffusers offline flags for all
three stages and refuses to overwrite an existing PNG. Package smoke tests
are limited to the exact prompt and settings reported below; neither this
wrapper nor the hard-link layout improves model quality or runtime latency.

The real-package `red cube` run on 2026-09-23 used the official component
revision above and the pinned 5,959,127,264-byte GGUF with tensor policy
`BF16:8,F32:65,Q8_0:96,Q6_K:64,Q5_K:32`. CPU BF16 conditioning, 40 native
Metal DiT steps on M2 Max, and CPU FP32 VAE decode produced a 256x256 RGBA PNG
with SHA-256 `396ee177a3689ca7d9a035ab4d26ecd58a065215017df56ee64fb1d7084fd841`,
byte-identical to the earlier direct three-command path for prompt `red cube`
and seed 7. The package CLI requires Metal access: on this host the process
sandbox hid the device, while the same local command outside that sandbox
completed. This is one smoke case, not general prompt or quality validation.

For the next native text-encoder transition,
`scripts/qwen_image21_text_reference.py` captures the official pipeline's raw
processor input IDs and masks, the 14-token prefix drop, all 37 hidden states,
and the pre-final-RMSNorm embeddings. The pinned `red cube` CPU BF16 fixture
has 24 raw tokens, 10 retained tokens, and embeddings of shape `[1, 10, 4096]`.
Its payload SHA-256 is
`3edcd7bf7964237d649a43c35cd82f6d6bd7b15835fddec2b1fe42f3a89b1e07`.
After BF16-to-F32 conversion, all 40,960 embedding scalars exactly matched
the earlier conditioning bundle for the same prompt and model revision. The
fixture remains outside the repository; no weights or generated tensors are
committed. This establishes a parity target, not a native Qwen3-VL inference
implementation.

## In-process multi-seed conditioning frontier (2026-09-26)

Status: **implemented and parity-checked for two seeds; not a general
speed or image-quality promotion**. The current text-to-image
conditioner loads the official CPU Qwen3-VL pipeline and encodes the same
prompt on every separate invocation, even when only the seed changes. The
pinned pipeline encodes `prompt, image=None, device` before it prepares latents
from image dimensions and a seeded CPU generator. Existing 768px and 1024px
bundles for the same prompt have byte-identical text embeddings and masks;
their latent shapes differ. This supports a narrow in-process reuse probe,
not a persistent cache or an end-to-end speed claim.

- **Admitted default:** the existing single-request CLI, payload schema v1,
  seeded initial-noise semantics, and package `generate` path remain unchanged.
- **Guard-only opt-in:** one text-to-image prompt and one image size may request
  several distinct seeds in one conditioner process. Load the official pipeline
  and encode the prompt once; create a fresh CPU generator and standard v1
  bundle for each seed in a new `seed-<n>` subdirectory. Validate all seeds and
  output collisions before loading weights. Batch bundle files use exclusive
  leaf creation and best-effort rollback of owned files on failure; this is
  not a security boundary against a hostile same-UID process with access to
  output paths.
- **Rejected in this slice:** disk-persistent embedding cache, image-edit
  conditioning reuse, mixed prompts or image sizes, multi-output package
  publication, and any claim that AB2 or batch conditioning reduces DiT work.
- **Falsifiers:** duplicate/invalid seeds and pre-existing outputs must fail
  before model load without clobbering data; a counting test must see one model
  load, one encode, and one independent `prepare_latents` call per seed; each
  bundle must retain the v1 shape, seed, and checksums. Refresh real pinned
  batch-versus-single byte parity when the model or runtime changes. Measure
  full conditioning time per image before claiming a wall-time win.

Rollback is the existing single-request conditioner. This evidence decays if
the official prompt encoder, latent-preparation semantics, model revision, or
processor/runtime changes. Prompt-embedding equality is not image-quality
evidence, and text reuse cannot stabilize portraits across different sigma
grids.

The opt-in `--seeds 7 11` path passed 11 batch-specific CPU tests plus the
unchanged conditioner, conditioning A/B, and package tests (49 Python tests
total) in the pinned environment. With model revision
`790c92633540aa0cb11d9abf19eb46d861714758`, Diffusers commit
`8b3c707ebd3ec4881f4190cf42931da07eaf3b65`, and CPU/BF16 Qwen3-VL,
the 512px portrait batch wrote both bundles in 21.23 s
on its first run and 13.43 s on a subsequent warm run. A single seed-7 run
between them took 14.19 s; the separately prepared seed-11 input took
22.69 s. Both batch seeds' manifests and payloads were byte-identical to
their single-request counterparts (payload SHA-256s
`9fca4e6ebc57d573b28612def1482a4931e4108e05ccb345479ed6f900c4bc58`
and `8f78b243a902b5565175ef0b5313aadf9e6b2a9681188c2fa92badb61c259a2a`).
After adding exclusive leaf-file creation, a repeat batch run took 23.43 s
and both seeds' manifest and payload bytes still matched their earlier
single-request outputs. This is a small sequential observation
with cache/host-order effects, not a throughput distribution. It shows the
conditioning process can amortize model load and prompt encoding across
seeds without changing those two outputs; it does not accelerate a DiT call,
VAE decode, or the current single-output package `generate` command.

## Native text-only Qwen3-VL frontier (2026-09-23)

Status: **native encoder not admitted**. The pinned CPU BF16 bundle above is
the oracle for the next backend, not evidence that the backend exists. The
local official `text_encoder/config.json` describes 36 decoder layers with
4096 hidden units, 32 query heads, 8 KV heads, 128 head dimensions, 12288
SwiGLU intermediate units, and interleaved multimodal RoPE. Text-only prompt
conditioning must retain the official processor's token IDs and left-padding
semantics, then return the attended hidden state **before** the model's final
RMSNorm and after the pipeline's 14-token prefix drop. The vision tower and
LM head are outside this slice.

- **Admitted already:** the pinned reference capture and hybrid Python encoder
  remain the package's production path. No native encoder result may enter the
  package yet.
- **Guard-only implementation sequence:** validate the fixture and sharded
  safetensors metadata; establish exact embedding lookup parity; establish
  per-layer parity for a real prompt with explicit precision tolerances; then
  validate all 36 layers and the final pre-norm, post-drop embedding handoff.
  Only after that may the package select a native text path. Reuse generic
  matmul/norm/Metal infrastructure where its numerical contract matches; the
  Qwen3.5 layer topology itself is not an assumed drop-in implementation.
  The existing `qi21_bf16_batch_matmul` Metal kernel is a candidate projection
  primitive, but it emits F32 and uses a different reduction order. Qwen3-VL
  still needs validated BF16 module boundaries, text RoPE, GQA, masks, and
  residual arithmetic around that primitive.
- **Rejected:** claiming a full native Qwen3-VL encoder from a loader-only or
  single-layer smoke; using a quantized encoder as the sole parity reference;
  enabling QBit or fusion on this path before the BF16 reference discrepancy
  has been measured; invoking the vision branch for text-only prompts.
- **Falsifiers:** invalid fixture hashes/offsets, incorrect safetensors shard
  mapping or tensor shapes, any embedding lookup mismatch, divergent first
  layer, or final embeddings that fail the declared accuracy gate. A small
  clean prompt alone does not cover padding, multimodal IDs, or longer
  sequences; those become separate gates before broad native admission.
- **Rollback:** leave `scripts/qwen_image21_prepare_conditioning.py` as the
  default conditioning route. Any native experiment must be opt-in until the
  package's exact prompt-to-PNG regression and parity gates pass.

The scaffold DoD is synthetic corruption and shape guards for the reader and
weight loader, plus an independently calculated tiny BF16 decoder-block case.
The first **model-backed admission gate** is an actual pinned-fixture read and
an exact BF16-bit embedding lookup check against the official safetensors
shard. That gate is now observed at the current loader source: all 24 raw token
rows (98,304 BF16 scalars) matched `hidden_state_000` byte-for-byte. It does
not promote the full text encoder. Layer parity must report both maximum
absolute error and a
scale-aware error over the whole `[1, raw_tokens, 4096]` state, not just a
matching sample or a visually plausible PNG. Evidence decays when model
revision, Transformers implementation, processor template, fixture format, or
weight conversion changes.

Implementation plan for this CAUTION slice (rollback: keep the hybrid encoder
default and remove only the new opt-in native modules if falsified):

1. `src/ml/gguf/qwen3vl_text_reference.cr` plus its focused spec: reject
   corrupted or inconsistent fixture metadata/payload before exposing any
   token or hidden-state values. Real-fixture read is gated by
   `QWEN3VL_TEXT_REFERENCE_DIR`.
2. `src/ml/gguf/qwen3vl_text_weights.cr` plus its focused spec: validate the
   pinned text tensor inventory and shard bounds before reading selected rows;
   compare all 24 input embedding rows bit-for-bit with `hidden_state_000`.
   The real-weight run is gated by `QWEN3VL_TEXT_ENCODER_DIR`.
3. `src/ml/gguf/qwen3vl_text_block.cr` plus its focused spec: establish a
   synthetic first-block arithmetic check, then, after the model-backed gate,
   compare `hidden_state_001` against the pinned reference. Do not report
   text-encoder completion from a single block or synthetic-only tests.

The original focused scaffold DoD command passed 19 examples, including a
positive embedding read from a complete sparse synthetic shard:

```sh
crystal spec spec/qwen3vl_text_reference_spec.cr spec/qwen3vl_text_weights_spec.cr \
  spec/qwen3vl_text_block_spec.cr --link-flags '-fuse-ld=/usr/bin/ld'
```

With both real-artifact environment paths set at the current source state,
that focused command now passes 27 examples, including exact 24-row embedding
parity and a real layer-0 norm-vector read. The separate strict first-layer
parity spec remains red by design while backend arithmetic is investigated.

The model-backed embedding run sets both environment paths above and reports
the fixture-recorded revision, compared scalar count, and mismatch count. The
strongest pre-mortem is a wrong shard/layout mapping that passes shape checks
but silently changes embeddings; the all-token BF16-bit comparison is its
guard. The checkpoint and fixture are currently restored under
`/private/tmp/qwen3vl-artifact-UO3bIW`; this scratch location is not durable.
The local safetensors do not prove their own Hub revision: the restored shard
hashes were checked separately against the pinned Hub objects, while the
runtime loader validates tensor inventory and bounds rather than hashing all
17.5 GB on every read. A changed local shard invalidates the numerical
measurements until its identity is checked again.

The current implementation contains a guarded schema reader, an exact-shape
398-tensor BF16 inventory and bounded embedding-row loader, and a synthetic
CPU evaluator for one text decoder block. These are scaffolding, not a native
prompt-conditioning path. The synthetic block fixture checks Qwen3-VL-style
GQA, RoPE, masking, RMSNorm, and SwiGLU against a tiny independent PyTorch BF16
case; it does not establish parity for the official checkpoint. The current
model-backed loader reads the 11 decoder-block tensors as validated BF16 and
decodes them to F32 for a CPU probe. In a two-token causal layer-0 comparison
against the pinned full-prompt `red cube` reference, the experimental block
currently differs at 2,479 of 8,192 BF16 scalars: 93 in token 0 and 2,386
in token 1. The maximum absolute error is 0.0625 and relative RMS error is
0.00304. This is a **failed exact-bit gate**, not evidence of accepted layer
parity. An official CPU rerun resolved the checkpoint's attention backend to
`sdpa`: both the full 24-token run and the two-token prefix reproduced the
fixture's first two layer-0 output rows exactly. Forcing `eager` changed 2,388
of 8,192 values, all in token 1. The native eager-style block differs from
official eager at only 100 values (93 in token 0, seven in token 1; maximum
absolute error 0.001953125 and relative RMS error 0.000170). Thus backend
arithmetic, not sequence truncation or tensor loading, explains most of the
observed discrepancy. After all 36 layers, the official SDPA rerun also
reproduced all 40,960 post-drop, pre-final-RMSNorm prompt embedding values
exactly. Forcing eager on the same official model changed 38,113 values, with
relative RMS error 0.0812 against the SDPA fixture. This is a consequential
numeric backend distinction, not just bit-level noise at layer 0. The
experimental `SdpaF32` attention mode keeps score, softmax, and value-reduction
intermediates in F32 before materializing its BF16 output. It passes a tiny
PyTorch 2.6 CPU SDPA oracle while leaving the eager default unchanged. With
the real checkpoint, it reduces the first-two-token layer-0 discrepancy to
121/8,192 BF16 values (93 in token 0 and 28 in token 1), maximum absolute
error 0.001953125, relative RMS error 0.000211. **Exact-bit parity remains
red.** The next gate is to locate those residual boundary differences before
full-layer and final-conditioning parity. No native conditioning path is
enabled.

The exact-bit spec is an arithmetic discriminator, not by itself the future
release threshold for a Metal implementation. A justified tolerance must be
set against full retained-token embeddings, masks, same-seed generated images,
and quality regressions; lowering this spec to an arbitrary layer-0 threshold
would not establish that result.

A two-token BF16 boundary trace further narrows that gap. The block input and
input RMSNorm are exact; `q_proj` differs at 2 values, while `k_proj` and
`v_proj` remain exact. The attention output projection differs at 4 values,
but the post-attention RMSNorm is exact again. MLP `gate_proj` and `up_proj`
each differ at 7 values, `down_proj` at 232, and the final residual at 121.
This localizes the remaining discrepancy to low-level arithmetic around dense
projections and their composition; it does not prove a particular BLAS reduction
order or establish whole-encoder parity.

The one-ULP projection differences are consistent with F32 accumulation-order
sensitivity, but the exact PyTorch/BLAS reduction tree is not established. A
candidate 32-lane reduction is not yet a native correctness contract.

The backend discriminator is reproducible with the local, pinned BF16 artifacts
and CPU PyTorch 2.6.0 / Transformers 4.57.3:

```sh
python scripts/qwen3vl_text_layer_trace.py \
  --model-dir /path/to/text_encoder \
  --fixture-dir /path/to/reference \
  --output /private/tmp/qwen3vl-text-layer-trace.json \
  --dump-layer0-bf16
```

This diagnostic runs the full prompt and a two-token prefix through both the
checkpoint default attention backend and eager. Its JSON records the selected
backend, fixture hashes, layer-0 metrics, and the post-drop pre-final-norm
conditioning comparison; the optional BF16 sidecars allow byte-level checks
against the native block. It is an expensive CPU reference probe, not a native
inference path or a quality benchmark.

The bounded native layer sweep separates each block's own arithmetic error
from accumulated input drift. With the pinned fixture and `SdpaF32`, the first
two tokens give the following results. Layers 0–3 were independently repeated;
layer 35 comes from one bounded 36-layer run:

| Decoder layer | Isolated BF16 mismatches | Composed BF16 mismatches | Composed relative RMS error |
| --- | ---: | ---: | ---: |
| 0 | 121/8,192 | 121/8,192 | 0.000211 |
| 1 | 3/8,192 | 1,325/8,192 | 0.000560 |
| 2 | 549/8,192 | 2,921/8,192 | 0.001004 |
| 3 | 207/8,192 | 3,890/8,192 | 0.001284 |
| 35 | 1,278/8,192 | 7,879/8,192 | 0.035759 |

The composed result is the relevant warning for a future full encoder;
isolated near-parity does not certify the stack. These are two-token,
single-prompt diagnostics, not a prompt-conditioning or image-quality gate.
Both probed tokens belong to the 14-token template prefix that the pipeline
later drops. Their states can affect later tokens through causal attention,
but the measured rows are not themselves the returned conditioning. A full
raw-prompt run must compare the ten retained rows before any native handoff.
The 36-layer probe took 11 minutes 13 seconds on this CPU host while streaming
one F32-expanded block at a time; that is a diagnostic cost, not an inference
performance measurement.

```sh
crystal run scripts/qwen3vl_text_layer_sweep.cr \
  --link-flags '-fuse-ld=/usr/bin/ld' -- \
  --layers=2 --text-encoder-dir=/path/to/text_encoder \
  --reference-dir=/path/to/reference
```

The default sweep still uses only the first two tokens. Use `--layers=36` to
reproduce that prefix diagnostic; add `--full-prompt --composed-only` for all
24 raw tokens without redundant isolated-block passes.
For a single full-prompt layer-0 boundary trace, add
`--layers=1 --full-prompt --composed-only --layer0-trace-dir=/path/to/new-dir`.
The new directory contains 16 BF16 stage files and a shape- and SHA-indexed
`trace.json`; an existing directory is never overwritten.
Compare that directory with one official full-prompt CPU pass:

```sh
python3 -B scripts/qwen3vl_text_layer_trace.py \
  --model-dir /path/to/text_encoder \
  --fixture-dir /path/to/reference \
  --full24-only --native-trace-dir /path/to/new-dir \
  --output /private/tmp/qwen3vl-full24-native-comparison.json
```

The comparison requires the pinned prompt, fixture SHA, model revision, stage
shapes, and stage hashes; it reports the first divergent BF16 boundary and
separates raw prefix rows from retained rows.

## Full-prompt native conditioning discriminator (2026-09-24)

With the pinned `red cube` CPU/BF16 reference, the 24-token native `SdpaF32`
layer-0 run differs at 19,098/98,304 BF16 values across all raw rows. The
ten retained rows (raw rows 14–23) differ at 10,827/40,960 values, with
relative RMS error 0.001554. The first two raw rows still have the same
93 and 28 mismatches as the separate two-token probe, guarding the causal
prefix comparison.

The official full-prompt layer-0 pass matches the pinned fixture at
0/98,304 BF16 values, including its input. The native input is also exact.
The first native divergence is `layers.0.input_layernorm`: 39/98,304
BF16 values, all in raw row 22 (one of the retained rows), with maximum
absolute error 0.000244. `q_proj` then differs at 912/98,304 values
(23 in dropped prefix rows and 889 in retained rows). Of the 912 `q_proj`
differences, 866 are in row 22; 46 remain on rows whose RMSNorm input to
that projection is BF16-exact, so RMSNorm drift alone cannot explain the
projection mismatch. The post-attention `o_proj` differs at 6,352/98,304.
These stage counts separate the earliest RMSNorm difference from later
projection and attention differences; by themselves, they do not establish
a specific arithmetic root cause or an acceptable image-level tolerance.

An independent replay of `q_proj` on the 23 rows with BF16-exact normalized
inputs attributes those 46 differences to reduction order: PyTorch 2.6 CPU
`nn.Linear` reproduces the mismatch count and its per-row distribution, while
serial F32 dot products reproduce the native `q_proj` sidecar bit-for-bit.
The BF16 checkpoint payload, `[out, in]` shape, and loader indexing agree, so
changing the weight orientation would attack the wrong cause. This conclusion
is limited to those 23 input-exact rows; row 22 has an upstream norm difference.

The opt-in equal-input operator replay makes the operator-local comparison
repeatable with the pinned 24-row native trace. It feeds each official
PyTorch operator its **native** BF16 input sidecar, then compares the result
with the native output sidecar. The stage comparison instead compares the two
already composed paths, so it includes upstream differences:

| Layer-0 boundary | Composed official-vs-native BF16 differences | Equal-native-input operator differences |
| --- | ---: | ---: |
| `q_proj` | 912/98,304 | 47/98,304 |
| Causal attention (`attended`) | 2,287/98,304 | 61/98,304 |
| `o_proj` | 6,352/98,304 | 47/98,304 |

The equal-input `q_proj` count includes one difference on raw row 22; the
other 46 are on the 23 rows whose normalized inputs were already BF16-exact
in the composed comparison. Attention uses 32 Q heads, 8 KV heads repeated
four times, the PyTorch 2.6 CPU SDPA MATH backend, and a causal/no-explicit-mask
call. The probe rejects a non-all-visible input mask, because this fixture's
24 raw tokens are all attended. Every native stage is shape-checked and its
SHA-256 is checked against the native trace manifest, which names the pinned
fixture payload SHA. This detects accidental sidecar drift, not coordinated
edits to a sidecar and its manifest. The probe also does not authenticate the
local checkpoint's weight files; it assumes the trusted official snapshot.

```sh
python3 -B scripts/qwen3vl_text_layer_trace.py \
  --model-dir /path/to/text_encoder \
  --fixture-dir /path/to/reference \
  --full24-only --native-trace-dir /path/to/native-layer0-full24 \
  --equal-input-ops --output /private/tmp/qwen3vl-equal-input-ops.json
```

This isolates three local arithmetic differences on one CPU/BF16 fixture; it
does not prove exactness of the other operators, bound 36-layer amplification,
or establish image-quality tolerance. A smaller operator-local mismatch count
is not itself a reason to change the native reduction tree.

A bounded row-22 replay with the pinned layer-0 norm weight narrows that
first difference further: PyTorch's F32 `pow(2).mean()` produces variance
`0.0005493450444`, while the current scalar, sequential F32 accumulation
produces `0.0005493425415`. Holding the BF16 casts and reciprocal-square-root
path fixed, the latter reproduces all 39 native mismatches; replacing only
the variance with the Torch value removes all 39. An F64-accurate mean also
matches the official BF16 RMSNorm output on all 24 rows in this one fixture.
This is a concrete reduction-order falsifier, not a general guarantee for
other prompts or a fix for the 46 `q_proj` differences on RMSNorm-exact rows.

The composed 36-layer, 24-token run differs from the official
pre-final-RMSNorm retained embeddings at 37,364/40,960 BF16 values across all ten
retained rows (maximum absolute error 28, relative RMS error 0.03114). Its
fixture guard confirms that the official `hidden_state_036` retained rows
equal the official pre-final-RMSNorm embeddings at 0/40,960 BF16 mismatches.
The composed CPU diagnostic took 247.76 s wall time on this host; it is not
a Metal inference benchmark. The difference is too large to promote native
conditioning or infer image-quality parity.

Two bounded RMSNorm reduction experiments tested whether removing the first
39 BF16 differences helps the composed output. Both made the full-prompt
layer-0 input RMSNorm BF16-exact (0/98,304) and reduced layer-0 output
differences to 17,456/98,304, but neither improved the ten final retained
rows. All values below compare against the same pinned CPU/BF16 fixture:

| Variance reduction | Layer-0 output BF16 differences | Final retained BF16 differences | Final retained relative RMS |
| --- | ---: | ---: | ---: |
| Serial F32 (current source) | 19,098/98,304 | 37,364/40,960 | 0.031138 |
| F64 sum, rounded to F32 | 17,456/98,304 | 37,454/40,960 | 0.031968 |
| Adjacent-pair F32 tree | 17,456/98,304 | 37,474/40,960 | 0.032575 |

The F64 method also disagrees with a 16-value PyTorch CPU BF16 RMSNorm
counterexample; a separate 16-value counterexample distinguishes serial F32
from the pairwise method. The pairwise tree is a useful local oracle match,
not proof of PyTorch's general reduction order. Both alternatives were
rejected and the source kept its serial F32 reduction: local layer-0 parity
does not compose into a better final embedding proxy here. The final metrics
are not image-quality measurements, and one prompt cannot settle broader
tolerances. The next promotion gate is a same-seed image A/B when full DiT/VAE
artifacts are available. Until then, PyTorch GEMM reduction order in `q_proj`
remains a bounded diagnostic question, not a promoted native change.

The optional `--retained-bf16-out=/path/to/native.bf16le` exports only this
full-prompt, 36-layer result, plus a checksummed JSON sidecar. It is an
experimental discriminator, not a production model artifact. A separate
tool clones a pinned CPU/BF16 conditioning bundle and replaces only its
retained embeddings with those native BF16 values expanded to F32:

```sh
python3 -B scripts/qwen_image21_conditioning_ab.py \
  --baseline-bundle /path/to/red-cube-cpu-bf16-bundle \
  --native-manifest /path/to/native.bf16le.json \
  --output-dir /path/to/new-native-ab-bundle
```

The A/B preparation requires the baseline's ten retained embedding rows to
match the pinned official text fixture's BF16 SHA-256; the 2026-09-24 local
baseline had 0/40,960 BF16 differences from that fixture. The new bundle's
embeddings are exactly the native sidecar expanded
to F32; text masks, image masks, and seed-7 initial latents are byte-identical
to the baseline. Payload hashes and the unchanged manifest fields were
checked separately; both real bundles passed the native conditioning loader's
optional A/B spec. These checks establish an isolated conditioning input
comparison, **not** a generated-image comparison by themselves. At that
stage, the full DiT GGUF and VAE were not available locally; the same-seed
image A/B below resolves that particular gap. An explicit multi-prompt
quality/tolerance gate still remains before replacing the hybrid CPU
text-encoder route.

The fixture's whole-payload SHA, its retained-embedding tensor SHA, and each
conditioning bundle's payload SHA identify different byte streams; the A/B
manifest records them separately.

### Same-seed native-conditioning image A/B (2026-09-24)

The previously blocked image-level discriminator ran at source revision
`c883965d24e908e9e31189d95a6a0d9830883df6` on an Apple M2 Max. The
unchanged pinned `red cube` 256x256/seed-7 conditioning bundles passed the
optional Crystal A/B spec (4 examples, 0 failures). Their prompt, model
revision, masks, image shape, and initial latents match; only the retained
text-embedding tensor differs. The baseline/native conditioning payload SHA-256
values are `56bfb2e98e22d193242a66a1bf2ea905482aecbc6e1ee6ad877775d66238bade`
and `602b6fb2daa2c01cf0feb582632346b0c1071682b6402532f4b7b4fd73c49202`.

Both bundles ran for 40 steps through the same freshly built native Metal
denoiser, with the pinned community Q4 DiT GGUF (5,959,127,264 bytes, SHA-256
`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`).
Both final latents were decoded by the same CPU/FP32 official Qwen-Image 2.1
VAE (`790c92633540aa0cb11d9abf19eb46d861714758`; safetensors SHA-256
`a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417`).
The baseline PNG reproduced the earlier pinned PNG **byte-for-byte** (SHA-256
`396ee177a3689ca7d9a035ab4d26ecd58a065215017df56ee64fb1d7084fd841`).
The native-conditioning PNG SHA-256 is
`3916250a9a1d72129812e0a97e70fe48a2ab15972430431e24be4af819b60ed7`.

The native conditioning changes all 16,384 final F32 latent values relative
to baseline, with relative RMS difference `0.003412` and maximum absolute
difference `0.0909724`. The decoded RGBA images differ in 43,097/262,144
channel values across 32,888/65,536 pixels, but their mean absolute channel
difference is only `0.1793` levels on a 0–255 scale (RMSE `0.5272` levels,
maximum `28` levels); alpha differs at 131 pixels, by at most one level.
Visual inspection found the same
red-cube composition and no obvious defect. The 62.75 s versus 51.90 s
denoising times were sequential cold/warm observations, **not** a throughput
A/B or evidence that native text encoding is faster.

The source and temporary artifacts for this check are
`/private/tmp/qwen21-ab-{baseline,native}-red-cube-20260924*`,
`/private/tmp/qwen21-ab-{baseline,native}-latents-20260924`, and
`/private/tmp/qwen21-ab-{baseline,native}-20260924.png`. The native bundle
uses a `-v3` suffix. Re-run the optional A/B spec with both manifest paths,
then run `scripts/qwen_image21_generate_latents.cr` on each bundle with the
same GGUF and 40 steps, and decode each output with
`scripts/qwen_image21_vae_decode.py --device cpu --dtype float32`. The native
runner needs Metal access; the ordinary sandbox hid the device, while the
permitted Metal run saw the M2 Max. These `/private/tmp` paths are ephemeral.

The first-run latent manifests did **not** contain the consumed conditioning
payload SHA-256, so those saved output artifacts alone could not independently
prove their input binding. An additive provenance slice now retains the digest
of the exact payload bytes validated by the native conditioning reader and
writes `conditioning_payload_sha256` to each latent manifest. Its focused spec
passed 5/5 examples, including the optional real A/B bundles; the existing
corrupted-payload case still rejects a checksum mismatch. Fresh 40-step
Metal runs wrote the expected baseline/native digests above into distinct
manifests under `/private/tmp/qwen21-ab-{baseline,native}-latents-bound-20260924`.
Their latent payload SHA-256 values, respectively
`d835709261d2d36870ebd564cfb55d3d4db8a1173f1f8e8d511684c8eb7fa844`
and `9fd301aa3d74d17fb996b9a639420653d0abd24f23fd34f63e4781dfbdcc8d08`,
match the first-run payloads byte-for-byte. The CPU/FP32 VAE accepted both
extended manifests and reproduced both PNG hashes above. This closes the
conditioning-payload provenance gap for these outputs; a source/executable
attestation is outside this slice, so a manifest is not proof of the runner
binary's identity or honesty. The package validator accepts the additive
field but does not yet enforce it against its input bundle; this A/B checked
the two hashes directly. These `/private/tmp` artifacts are ephemeral.

**Decision:** this one simple prompt passes a narrow same-seed image smoke
comparison, but it does not establish perceptual parity or acceptable native
conditioning across prompts, typography, composition, resolutions, or seeds.
Keep the hybrid CPU text-encoder route as the default. The next discriminator
is a small, fixed multi-prompt/multi-seed image set with explicit visual
failure criteria, followed by a performance comparison only if that quality
gate passes. Refresh this evidence after checkpoint, GGUF, VAE, Diffusers,
Metal source/driver, or conditioning-schema changes, or if the temporary
artifacts disappear.

### Multi-prompt native-conditioning diagnostic frontier (2026-09-24)

Current frontier: extend the opt-in red-cube discriminator to two fixed prompts
(an elven forest castle and a futuristic station with a `NOVA` sign), without
changing the packaged hybrid CPU-encoder default. At each prompt, the official
CPU/BF16 bundle is the baseline; the diagnostic native path may consume its
official processor/embedding input and replace only the retained 36-layer text
output. This is **not** an independent native tokenizer or full encoder.

- **Admitted diagnostic behavior:** derive raw and retained token counts from a
  checksummed official text reference rather than a red-cube constant; require
  exact prompt, pinned model revision, prefix-drop, mask, retained shape, and
  official BF16 embedding agreement with the baseline. Clone the baseline
  bundle while preserving masks and initial latents byte-for-byte. Keep native
  use explicit and outside the package's default generation command.
- **Rejected:** silently treating a new reference as the pinned red-cube
  fixture, relaxing source checks to accept arbitrary mismatched tokenization,
  promoting native conditioning for production, or claiming image-quality or
  speed parity from two prompts and one seed.
- **Falsifiers:** a prompt/reference/sidecar mismatch; truncation or an
  unexpected processor template; a retained-row or BF16 baseline mismatch;
  changed masks/initial noise; non-finite native values; or a failed native
  Metal/VAE run. Compare composition and `NOVA` legibility visually, not by
  pixel MAE alone. A missing or unreadable sign is a task-level failure even
  if the PNGs are numerically close.
- **Rollback and decay:** the pinned red-cube diagnostic and official hybrid
  path remain available. Evidence expires on checkpoint, processor/Diffusers,
  GGUF, VAE, runner, or conditioning-schema changes. Two fixed prompts are a
  useful discriminator, not a complete prompt or seed distribution.

For this discriminator the prompts are fixed verbatim:

1. `A majestic elven castle built among ancient trees in a dense forest at dawn, white stone towers with graceful arches, narrow bridges between trees, glowing windows, a winding river in the foreground, mist and shafts of golden sunlight, intricate fantasy concept art, wide view.`
2. `A futuristic coastal city at blue hour, a silver maglev train crossing an elevated bridge toward a glass station, turquoise neon lights, flying vehicles above, wet reflective pavement, and a large clearly readable sign saying NOVA above the station entrance, cinematic science fiction concept art, wide view.`

The hybrid baseline for each is 512x512, seed 7, 40 native-Metal DiT steps,
official CPU/BF16 Qwen3-VL conditioning, and official CPU/FP32 VAE decode.
The pinned model revision is `790c92633540aa0cb11d9abf19eb46d861714758`,
the Q4 DiT GGUF SHA-256 is
`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`,
and the VAE safetensors SHA-256 is
`a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417`.
The castle and `NOVA` baseline conditioning payloads have SHA-256
`be4c1701aba493f00d65d1b1a0c53ae31d765bdc3380e22165bce10c72fa3716`
and `b2000c6c53a835c78deb9080694913e293edbc15cb20961a02d9f232a99bd00b`;
their decoded PNGs have SHA-256
`2111d0e555553648dfc14ff3b444fd9af0afa7cd3887443a049be421757bdd46`
and `501ae2554be36ee4c8ae33abbc293f224ddf51db83e544228c233b1fef9ef645`.
The initial latent tensor bytes are identical across these two baseline
bundles (SHA-256
`d03158064c86fd691927cf93258b00fac2fc8d09e14a578d0f28b4fe8358932c`).
An independent offline call to the pinned local `Qwen3VLProcessor` reproduced
each reference's complete `input_ids` and `attention_mask` byte-for-byte
(77 raw/63 retained castle tokens; 79 raw/65 retained `NOVA` tokens); its
`tokenizer.json` SHA-256 was
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
These artifacts live under `/private/tmp/qwen21-ab-baseline-{elven-castle,nova-city}-20260924*`
and are ephemeral; the baseline images are not official end-to-end Diffusers
outputs. For arbitrary future prompt references, internally consistent hashes
alone do not prove that token IDs encode the declared prompt; the capture
process or an independent tokenizer check remains a trust boundary.

The official text-reference payload SHA-256 values for castle and `NOVA` are
`695dfc0644b0caf0626dc51c9c50f959739a157267a84e44c2232457b83eadd6`
and `4daeaa36088ff65ff8df964b489d04606cb78ab295278c0735ef20dbf246fc56`;
their manifest SHA-256 values are
`c66830e454b970673d556b05a2c02f7eadbf43cc0bce1b1b7b6eebf71b523fde`
and `4664b370f2aed97cb2591f88ee8f39dc5228577c1ef8b4991744299b48dd51a5`.
The opt-in native 36-layer `Accelerate` text runs produced BF16 retained
sidecars with SHA-256
`a40d894b1332611bf456540d14d684fef034958fdbc5efc866a13829821b2b3c`
and `8c2ef509e7864975139ecfc9aa488220e1754011509db3171e1269dea7565395`.
Relative RMS differences from the corresponding official retained BF16
embeddings are `0.0231506763` and `0.0218952444`; these are fidelity probes,
not image-quality scores. Each sidecar binds the exact reference payload and
manifest hashes. The A/B bridge accepted both, producing conditioning payload
SHA-256 values
`e075da7b38e511cb5ed35f3d0cfa3571169092800cc79bf932000591fda1698c`
and `00fd51037985b8b358749335a2a2baf0a0662c506f8c251d9e00c390853b1d0b`.
Independent byte-range checks found only `encoder_hidden_states` changed;
the masks and initial latents are identical to each prompt's baseline.

The earlier scalar castle run was interrupted after layer 0 with no output
sidecar; that partial log is not a full-stack or backend-speed result. One
real 77-token Accelerate layer had the same aggregate official-reference
error counts as a prior scalar layer, but exact backend output parity was not
established. The two completed Accelerate sweeps overlapped other work, so
their wall times must not be used as an uncontrolled performance comparison.

Both native-conditioning bundles completed the same 40-step Metal DiT and
offline CPU/FP32 VAE path as their respective hybrid baselines. The native
castle latent manifest and payload SHA-256 values are
`7a8a1cc229425c781df284dd4652a6ea4eae0216a67de573aa780a2e9d47c917`
and `1947d25fabc05b3e2b36e1e7a6a74f79ce13c87be56fd6972e80b8f08d451b30`;
the native `NOVA` values are
`e6e9bed0510aeba535339abe7a64d8decb4271619c65afab380f16eac04fe3f0`
and `2ec384d8740aea6fb4eae6ff66a9c874238b06adfad92f7d19a8eea2e342fbff`.
Each latent manifest binds the exact prompt, revision, seed 7, 512x512 image,
40 steps, and its corresponding conditioning payload SHA-256 above. The
decoded native PNG SHA-256 values are
`c3841d143241bb177ebc4aa2549a734668c36c56907068ce2d01bf0f189f274d`
and `1837d818ae2ee94d15b0c0354b161e426771988dc197301ae716eaf6ca62ad34`;
both are 512x512 RGBA under
`/private/tmp/qwen21-ab-native-{elven-castle,nova-city}-20260924.png`.
Independent byte checks and PNG inspection found that the castle retained
its towers, bridges, river, forest light, and camera composition, while the
futuristic scene retained its maglev train, glass station, flying vehicles,
wet pavement, and legible `NOVA` sign. RGB mean absolute error versus each
hybrid baseline PNG was 0.191/255 for the castle and 0.606/255 for `NOVA`;
latent relative RMS was 0.00356 and 0.01210, respectively. These image and
latent differences are descriptive probes, not quality metrics. Two prompts
at one seed support a narrow native-text compatibility finding, not general
prompt fidelity, typography reliability, performance parity, or promotion of
the native path as the default. A next falsifier would vary seeds, prompt
length and typography while preserving this same one-tensor A/B control.
The temporary images and manifests can disappear independently of this source
revision, in addition to the model/processor/runner decay triggers above.

### Photorealistic portrait A/B diagnostic (2026-09-24)

The same opt-in, one-tensor native-conditioning comparison was repeated on two
fictional adult portraits. This did not change the packaged hybrid default.
The prompts were fixed verbatim:

1. `Photorealistic editorial head-and-shoulders portrait of a fictional adult woman with medium-brown skin, short natural curls and faint freckles, sitting by a window in a quiet apartment, looking directly at the camera with a relaxed expression. Soft overcast daylight, natural skin pores and fine lines, realistic eyes and individual hair strands, shallow depth of field, 85mm portrait lens, neutral color, no heavy retouching, no illustration, no text or watermark.`
2. `Photorealistic environmental head-and-shoulders portrait of a fictional elderly man with silver hair and a weathered face, wearing a dark wool coat on a city street after rain at blue hour, looking slightly away from the camera. Warm storefront light on one side of his face, cool evening light and softly blurred wet reflections behind him, natural wrinkles and skin texture, realistic eyes and hair, 50mm portrait lens, no heavy retouching, no illustration, no text or watermark.`

Both used the pinned model revision, Q4 GGUF DiT, and VAE identified above,
512x512 output, seed 7, 40 Metal denoising steps, and offline CPU/FP32 VAE
decode. The official CPU/BF16 Qwen3-VL text output was the baseline. The
native arm64 36-layer Accelerate sweep replaced only
`encoder_hidden_states`; independent byte checks confirmed that both masks and
the initial latents remained identical within each pair. The initial noise was
also identical across prompts (SHA-256
`d03158064c86fd691927cf93258b00fac2fc8d09e14a578d0f28b4fe8358932c`).
An independent offline call to the pinned local processor reproduced each
official reference's complete token IDs and attention mask byte-for-byte:
119 raw/105 retained tokens for daylight, 121 raw/107 retained for rainy.
The native sweep was built from the current source with Apple's linker after
Crystal's default `ld64.lld` rejected this SDK's `arm64e.x1` `.tbd` entries;
the sweep binary SHA-256 was
`74cdb9db89657db64fb4757f344b6622fbaefcb2d0c5113c893dcdfb6f39ab73`.
The immutable Metal latent runner SHA-256 was
`39aa3ca65937b101033bd7bbce8fc2efe17649abd523b06e66824779110438f7`.

| Portrait | Official reference payload | Native BF16 sidecar | Baseline / native conditioning payloads |
| --- | --- | --- | --- |
| Daylight | `1f73114e45ccce4acc742794f1dcfa09d6e7e8508339e4951f3ff100461d68fd` | `7cb7ed8ce1e9ee541a98c8c14d6551f4a14ec5f505f1b0a2006b3d03fec65d80` | `9fca4e6ebc57d573b28612def1482a4931e4108e05ccb345479ed6f900c4bc58` / `83afba7717b44de8c6cbd1556a94e9431a7e331cd0d2bfbd002ae232564cf252` |
| Rainy | `aaa1f52035cf5990affab9216a7edf1b35b17a21c5a466202d570e779c6f6e49` | `2217ee0c89b1b2570b94608c211e9ce747d52a2e8972d9e7a564aad5e80afc1a` | `36afb59f8f2f087da2681bdb32f9dc493d24c7c2ecfd35ef5944db6fc827007f` / `febb6ecce97cb9739f60ec2fc9b468ef9a93c076f35b99e98f122e87da3af49e` |

All four Metal runs and CPU/FP32 VAE decodes exited successfully. Their
latent manifests bound the exact prompt, revision, seed, dimensions, 40 steps,
and corresponding conditioning payload SHA-256. The baseline/native decoded
512x512 PNG SHA-256 pairs were:

- Daylight: `8fc566c291bf575587928dfad9b7b8a605d3c77b031da23c008dfb5cb75534e0` / `8388263efbefdb0e5a0a5787fa7f42e89784d9e480dcc64cac65b11fedc3ec3d`.
- Rainy: `1912246788a8e866cb3c1578c6d99baf7b2ce2f738a2349395d5386e0e91b4c4` / `42a7c5b1e4d5817d27666391c27b7c9f9cf3f0ff35dd09bd185e1a2449db43ba`.

The baseline portraits are visibly photographic and on prompt: the daylight
frame has a direct gaze, curls, and soft window light; the rainy frame has a
silver-haired subject, warm storefront light, cool street light, and wet
reflections. Both native frames preserve face structure, eyes, hair,
composition, and lighting at normal 512px display scale. The baseline images
already smooth away some requested freckles, pores, and fine skin lines; this
cannot be attributed to the native path. The native/baseline RGB mean absolute
differences were 0.398/255 (daylight) and 0.238/255 (rainy); latent relative
RMS differences were 0.00963 and 0.00648. These are descriptive A/B distances,
not perceptual-quality or speed scores. Native retained BF16 text outputs were
not exact matches to the official reference (relative RMS 0.02071 and 0.02049;
most BF16 elements differ), so the result does not establish encoder parity.
The two concurrent GPU runs in each stage also preclude a controlled speed
comparison.

This verifies only two prompt/seed-7 image comparisons, not broad portrait
fidelity, skin-texture quality, exact embedding equivalence, or readiness to
promote native conditioning to the default. A next falsifier should vary
seeds, resolution, face angle, and fine-texture demands while keeping the
one-tensor A/B control. The PNGs and manifests currently live under
`/private/tmp/qwen21-portrait-daylight-20260924-p1` and
`/private/tmp/qwen21-portrait-rainy-20260924` and are ephemeral. Checkpoint,
processor/Diffusers, GGUF, VAE, or runner changes also invalidate this
evidence until the experiment is repeated.

### Long Russian scene and Cyrillic sign A/B (2026-09-26)

A longer compositional text-only probe used this exact Russian prompt as test
data:

> Фотореалистичная сцена ранним утром в маленьком книжном магазине Санкт-Петербурга. Пожилая продавщица в синем фартуке передаёт пакет с апельсинами молодому курьеру в жёлтой куртке, который придерживает стеклянную дверь. Рыжий кот спит на стопке газет перед прилавком; за окном видны трамвайные рельсы и два велосипедиста. Над прилавком деревянная вывеска с единственной чёткой надписью кириллицей: «ЧИТАЛЬНЯ № 7», без других букв. Тёплый янтарный свет внутри контрастирует с холодным синим снегом снаружи; средний план, естественные лица, широкая композиция.

The pinned official processor produced 244 raw tokens and 230 retained text
rows after the 14-token prefix drop, within the native probe's 256-raw-token
guard.

The official CPU/BF16 text reference payload SHA-256 was
`c5487a29449bd7335fd781e02c8372496873ba264749fa1f32a1b8735d5258d6`.
The native 36-layer Accelerate retained-BF16 sidecar SHA-256 was
`827a5bcf65abc67589448f883899d429cf0d816acb79ab203f41e9209c95b0ba`;
its relative RMS error against the official retained embeddings was
`0.01884813`, not exact parity. The guarded A/B bridge replaced only the
float32 `encoder_hidden_states`; an independent byte comparison confirmed
that both masks and all seed-7 initial latents were identical. Official and
native conditioning payload SHA-256s were respectively
`007ad14a01440a9f786ae874e78cb4aef3a3729d2768fec1a503363bf66114ab`
and `5dc709f954210e2283268ce6c7ceace4efd6e6cef58cc74e20f9369891fc9705`.

Both arms completed 512x512, seed-7, Euler-10 Metal DiT runs with model
revision `790c92633540aa0cb11d9abf19eb46d861714758`, Q4 GGUF SHA-256
`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`,
and the same CPU/FP32 VAE decode. Their DiT times were 87.615 s
(official) and 85.924 s (native), sequential single runs that do not
establish a speed difference. Decoded official/native PNG SHA-256s were
`9c113cf0da630c7f576eeef47032afaa4873b879404ec6f42ed87594f14c015f`
and `f96a7fa5cf3627e00e3f6753a5789331040cbbbbe2ebc846b4804a407c5212d0`.
Visual inspection found nearly the same composition, subjects, doorway,
books, cat, and warm/cold lighting. The final latent cosine similarity was
`0.998954`; RGB mean absolute channel difference was `1.60/255`. These are
descriptive deltas, not quality scores. The sign is broadly recognizable, but
individual glyphs and `№ 7` are not dependable as exact typography in either
arm. This supports narrow scene-level native-text compatibility on one long
prompt, not exact encoder parity,
typography fidelity, or general prompt/seed quality. The scratch images and
manifests may disappear; processor, checkpoint, GGUF, VAE, and runner changes
invalidate the comparison.

Project-owner inspection of enlarged 10-step crops found a narrower exception:
the native rendering's adjacent `ЬН` avoids an extra glyph-like stroke visible
in the official rendering. Other letters remain imperfect, and the supplied
bookseller face crop shows small defects. This local observation does not make
the whole sign or face more reliable across seeds.

The same two conditioning bundles were then reused for independent 512x512,
seed-7, Euler-20 and Euler-40 Metal DiT runs, all decoded with the same local
CPU/FP32 VAE. The 20- and 40-step sigma schedules differ; these are not
checkpoints along one trajectory. The runner SHA-256 was
`7a39e4c304ac0697c44dbbb7f361c57c1b638aa14144ca5ad430c598cb27c7b7`;
the VAE weights SHA-256 was
`a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417`.
The Q4-labeled GGUF and conditioning hashes are pinned above. Both masks and
the initial target latents remained byte-identical across the arms; only
`encoder_hidden_states` differed. The final four runner executions and four
CPU VAE decodes exited successfully after an initial sandboxed Metal attempt
and a system-Python decode attempt failed before producing outputs:

| Steps | Text conditioning | Metal DiT denoise | CPU VAE decode | PNG SHA-256 |
| --- | --- | ---: | ---: | --- |
| 20 | Official Qwen3VL (precomputed) | 328.289 s | 21.31 s | `bfb47bcf29deb82418c9508a4e7a28e195336d101176fe223f2763ddd4fa5b2a` |
| 20 | Native Qwen3VL (precomputed) | 368.979 s | 17.97 s | `08404cf8df78672895433beb8abe3de7527fc546b5fb09f8a309481aa9ad5d64` |
| 40 | Official Qwen3VL (precomputed) | 637.755 s | 17.62 s | `d87d5933dd0e32584ea5e3ec2d704277438fb0ea6cea6c2acab80496fab85d66` |
| 40 | Native Qwen3VL (precomputed) | 788.051 s | 18.42 s | `24c8b67228544368c5a4e16575832fee54ccc6da868a6fe29f5052ebace4f5cd` |

Within each step-count pair, the composition is very close and both signs
look approximately like `ЧИТАЛЬНЯ № 7`; inspection at the full 512px scale
found no defensible `ЬН` or bookseller-face winner. In this seed the 40-step
scene has a larger sign that appears more legible, but also a substantially
different arrangement of the people than the 20-step scene. It cannot be
described as a simple cleanup of the 20-step image. The owner's 10-step
face-crop concern is not disproven by the small faces in these full-frame
comparisons. These are single-seed visual observations, not a general
typography or portrait-quality result. The timing rows are sequential single
runs with precomputed conditioning, not a speed
comparison of the two encoders. There were no same-arm repeat runs or GPU
clock/thermal measurements, so the unequal DiT times cannot be attributed to
the embedding source either. Exact per-arm commands, environment, and
ephemeral PNGs live under
`/private/tmp/qwen21-russian-steps-ab-20260926-r1`; refresh the comparison
after processor, checkpoint, GGUF, VAE, runner, or Metal-route changes.

### Where the Russian A/B latents diverge (2026-09-26)

An opt-in `QWEN_IMAGE21_STEP_LATENT_DIR` now records a copy of the
post-update target latents after each solver step as `step-000.bin` through
`step-039.bin` (`32x32x64`, tokens-HWC, normalized float32 little-endian).
The directory must be new, and the observer receives a copied array so it
cannot modify the live trajectory. The focused flow-matching spec passed
16/16 examples, including an observer-mutation check. A matched two-step
official-conditioning smoke produced exactly the same final latent SHA-256
with and without snapshots:
`807537b77bb0731e9e71f8d73dc6e5035fbe12c187f0ed55bf616c7b2afd24b4`.

Both independent 40-step runs reused the conditioning, GGUF, seed, mask,
initial target latents, solver, and Metal path from the preceding A/B. Only
the Qwen3VL `encoder_hidden_states` differed. The instrumented runner SHA-256
was `98bc33d563a8b5ca26a57be1437ee43345357f6ea8658f11336ab3092a0e58d5`.
Each arm emitted 40 finite 262,144-byte snapshots; its final snapshot matched
its saved bundle byte-for-byte and reproduced the pre-instrumentation final
SHA-256 (`d3c20f...` official, `a94192...` native; full hashes in the run
logs). The runs were sequential, not concurrent. A sandboxed Metal attempt
failed before running; the bounded device-enabled runs succeeded.

At the first DiT evaluation, the starting image latent is identical in both
arms, so the step-0 difference is a direct response to the different text
conditioning. Later differences include both ongoing conditioning and
feedback from the already diverged latent. The table uses
`||native - official||_2 / ||official||_2` at the same post-update step;
the two approximate spatial ROIs are a `5x4`-token face crop (rows 9:14,
columns 20:24) and a `5x14`-token sign crop (rows 0:5, columns 18:32).

| Euler step, zero-based | Whole latent | Selected face ROI | Selected sign ROI |
| ---: | ---: | ---: | ---: |
| 0 | 0.0207% | 0.0197% | 0.0123% |
| 5 | 0.1475% | 0.1391% | 0.0719% |
| 10 | 0.5769% | 0.4791% | 0.1961% |
| 15 | 2.2918% | 1.9256% | 0.3899% |
| 20 | 4.5160% | 3.9610% | 0.7134% |
| 25 | 6.9656% | 6.5157% | 1.1037% |
| 30 | 9.0935% | 9.3082% | 1.5282% |
| 35 | 10.5552% | 11.6381% | 2.0472% |
| 39 | 11.1655% | 13.0016% | 2.5439% |

The whole-latent relative difference increased at every measured step; there
was no single late discontinuity. The selected tight face crop stayed above
the whole-latent difference from step 29 onward, but this is ROI-sensitive:
shifting that crop by one latent token yielded final differences from 9.67%
to 13.30%, and a broader `7x6`-token crop ended at 11.05%. These ROIs are
not a precise inverse mapping of VAE pixels to latent tokens, and neither
relative L2 nor a face crop is a perceptual-quality score. Official/native
final latent norms were 280.286/279.935; their ratio stayed within about 0.13%
of one across the trajectory. That argues against a gross amplitude
explosion, not against an off-manifold displacement.

The absolute cross-arm distance grows from `0.05153` after step 0 to `31.295`
after step 39 (607-fold), but the final difference is nearly orthogonal to
the initial difference (cosine `0.0169`). It is therefore not a scalar
amplification of the first-step error. Each of the 39 subsequent cross-arm
Euler-update differences has a positive dot product with the preceding
cross-arm difference (median cosine `0.888`), but these saved trajectories
cannot separate propagation of an existing state difference from a new
conditioning effect on every step. At the final step, the top 10% of latent
tokens account for 70.7% of squared cross-arm error (effective support
about 101 of 1,024 tokens). A post-hoc selected `4x4`-token hotspot at
rows 20:24, columns 4:8 accounts for 27.2% of final squared error (39.1%
at step 15); this is latent-space concentration, not a pixel-accurate face
or sign map. A crossed-pair DiT intervention below tests three saved
transitions while holding latent state or conditioning fixed.

A per-step spatial audit of all 80 finite snapshots verified both final
snapshot/bundle pairs byte-for-byte. The top 10% of tokens carried 29.12%
of squared cross-arm error at step 0, 66.00% at step 10, 80.48% at step 15,
and 70.74% at step 39. The selected rows 20:24, columns 4:8 carried 4.22%,
30.45%, 39.10%, and 27.16% at those same steps. Its absolute squared error
first reached 10% of its own final value at step 20 and 50% at step 32;
relative share and absolute magnitude answer different questions. The
approximate `5x4` face and `5x14` sign token ROIs from the table above
accounted for only 2.55% and 0.32% of final global squared error,
respectively. Their visual salience therefore cannot be inferred from the
whole-latent error ranking. The top-16 error-token sets at steps 0 and 39
overlap in only one token
(Jaccard 0.032), whereas adjacent top-decile sets have median Jaccard
0.962. Together with the first/final signed-difference cosine 0.0169, this
rejects a fixed error vector or fixed earliest hotspot across the entire
trajectory, while allowing stable localization later. These are token-space
statistics, not evidence that the selected region maps to the observed face
defect or that it has a direct perceptual-quality score. The read-only
analyzer and report are under `/private/tmp/qwen21_spatial_drift_audit.py`
and `/private/tmp/qwen21_spatial_drift_audit.md`; rerun the analyzer on the
same 80 saved tensors after any input-identity or layout change.

Equal-input four-corner native DiT probes at transition indices 10, 20,
and 39 evaluated each arm's saved pre-step state against both encoder
payloads, holding the other argument fixed in each contrast. Fresh layer
stacks built a separate prefix cache for every forward. At all three
indices, both diagonal predictions reconstructed their saved post-step
latents exactly (`65,536/65,536` Float32 values each); the predicted
paired update difference agreed with observed snapshots to max absolute
error `2.38e-7` (Float64 comparison after Float32 Euler rounding). Without
these controls, the crossed contrasts would be invalid.

For `v_ij = DiT(x_i, c_j, t)` with `i,j` official or native, define the
symmetric conditioning contrast
`C = ((v_ON-v_OO) + (v_NN-v_NO))/2` and state contrast
`S = ((v_NO-v_OO) + (v_NN-v_ON))/2`. The identity
`C+S = v_NN-v_OO` held exactly at each sample. The nonadditive interaction
is `I = v_NN-v_NO-v_ON+v_OO`; it is not a third additive part of the paired
diagonal difference. All norms below include each transition's own Euler
`|dt|`, and projections are signed along its pre-step native-minus-official
drift direction.

| Transition | Pre- to post-step drift L2 | `||dt*C||_2` | `||dt*S||_2` | `||dt*I||_2` | State/conditioning | `dt*C` projection |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.83737 to 1.23178 | 0.01711 | 0.47478 | 0.02061 | 27.75x | -0.001484 |
| 20 | 7.71887 to 8.58743 | 0.01585 | 0.92897 | 0.02653 | 58.62x | +0.000133 |
| 39 | 30.56281 to 31.29533 | 0.00400 | 0.94170 | 0.00505 | 235.22x | +0.000020 |

At these local transitions, feedback from the already different image
state dominates the *contemporaneous* conditioning contrast by norm. The
paired update difference is still only about 56%, 12%, and 3% of the
pre-existing drift at indices 10, 20, and 39. Conditioning's signed
projection changes sign and is not a consistently positive drift driver;
its interaction with state exceeds its symmetric main effect at each
sample. The original state difference was itself initiated by different
conditioning. These observations do not exonerate Qwen3VL, establish a
DiT implementation bug, show a VAE-manifold departure, or score visual
quality.
The opt-in scratch harness and full run log are under
`/private/tmp/qwen21-crossed-dit-20260927/`; their SHA-256 values are
`397b9658042308533c285d58bfd1d0e013d027f09e40791781c7ad62838562a9`
and `613ea6ca1d071172616bdbf6d0c7c9b4ba764996dc1014ba71ff1c35b85e43d4`.
Re-evaluate after source, GGUF, conditioning, scheduler, Metal route, or
saved-state changes; other timesteps, seeds, and prompts remain open.

A separate scheduler-parity audit found a DiT-time input mismatch relative
to the pinned official BF16 pipeline. All 40 raw scheduler timesteps,
sigmas, and Euler `dt` values are bit-identical between the reference and
native schedules. The official pipeline first casts raw `t` to the BF16
latent dtype, then divides by 1000 in BF16; the native schedule divides
raw `t` by 1000 in Float32. Consequently the DiT time value differs at
39/40 indices. At index 20, both start from raw `623.0131226`, but the
official DiT receives `0.625` and native receives `0.62301314`; the
largest absolute discrepancy is `0.002673745` at index 21. This is a
real implementation-parity gap, but it was not the *differing input*
between the two saved conditioning arms: both used the same native time
path. It may still affect how their conditioning difference propagates.
A bounded same-state, same-conditioning native DiT
intervention at index 20 then changed only `t=0.62301314` to the official
BF16-effective `t=0.625`. Its Euler-scaled output difference had L2
`0.02615` on the official-conditioning arm's saved state and `0.03846`
on the native-conditioning arm's saved state, versus the four-corner
conditioning contrast `0.01585` and state contrast `0.92897` at that step.
Both original-time diagonal Euler reconstructions were bit-exact. This
demonstrates a nonzero local DiT sensitivity, not the direction of image
quality or the accumulated effect of changing the time path for all 40
steps. The scratch harness and log are under
`/private/tmp/qwen21-timestep-counterfactual-20260927/`. Refresh after
Diffusers pipeline, latent dtype, native scheduler, GGUF, or Metal changes.

An independent same-value index-20 DiT oracle then used the pinned official
BF16/MPS checkpoint and native mixed-quant GGUF/Metal path. The saved
post-step-19 official-conditioning state was rounded once to BF16 and
supplied numerically identically to both models; all 230 conditioning rows
were already BF16-exact. Both received the same masks, shapes, and
`t=0.625`. With output shapes `1x1024x64`, the corrected native-minus-BF16
velocity difference had relative L2 `0.014126`, cosine `0.999902`, RMS
`0.019704`, and max absolute value `0.168639`; the index-20 Euler-scaled
L2 difference was `0.121686`. This is a *single-forward combined*
implementation/mixed-quant/precision/provenance contrast, not an isolated
quantization error, 40-step error bound, or perceptual score. It supports
close local vector-field agreement at this tested input while leaving
trajectory-level amplification and localized face quality open.

The first oracle readout was invalid: a direct MPS BF16 sliced-view to CPU
Float32 conversion mishandled the 230-row storage offset, shifting the
corrected output by exactly 115 Float32 rows and corrupting the tail.
The corrected readout transferred the full BF16 tensor to CPU before
slicing and widening; an independent zero-offset BF16 clone agreed for
all `65,536` output values, each BF16-exact. The original direct-view
transfer differed at `65,472/65,536` entries, so its apparent relative
L2 `1.3613` must not be used as DiT evidence. The pinned scratch oracle
scripts, guarded input/output reports, and comparison are under
`/private/tmp/qwen21-matched-oracle-20260927/`; re-run after checkpoint,
GGUF, Diffusers, Torch/MPS, masks, conditioning, or native kernel changes.

A separate CPU/float32 VAE probe re-decoded both saved endpoints to exactly
their existing PNG pixels. Decoding latent interpolants at alpha 0, 0.25,
0.5, 0.75, and 1 changed the selected face smoothly, without an observed
decoder cliff. This probes local VAE sensitivity only: the interpolants are
not valid diffusion trajectories and need not lie on the training manifold.
The reference and native decoder endpoints still differ in the face pixels
(12.36/255 mean absolute channel difference in the selected 60x69 pixel
crop, versus 4.37/255 over the full frame).

### Selectable BF16-effective DiT time (2026-09-27)

The pinned Diffusers BF16 pipeline casts the raw scheduler timestep to the
latent dtype *before* dividing by 1000. The native FlowMatch path now exposes
`QWEN_IMAGE21_TIMESTEP_PRECISION=bfloat16` to reproduce that scalar input:
round raw `t` to BF16 (round-to-nearest-even), divide by 1000, round the
quotient to BF16, then widen to Float32 for the native DiT. `float32` remains
the default because the native latent state is Float32; this option does not
make the state, weights, or full DiT computation BF16. The latent manifest
records `model_timestep_precision`. The existing per-step log's `timestep`
field continues to report the *raw scheduler timestep*, not the selected
model-time scalar.

The focused flow-match suite passed 20/20 cases, including all 40 pinned
BF16 model-time words for 1024 target tokens, raw Float32 timestep bits at
indices 20 and 21, ties-to-even, mode parsing, and callback forwarding:
`crystal spec spec/qwen_image21_flow_match_spec.cr --link-flags
'-fuse-ld=ld'`. An invalid CLI mode failed before model loading. A fresh
runner (SHA-256
`3163b2875fa31851201a16af803ad1e332247c63bf61a8e260523fcf47b26a26`)
was built from this source with the absolute `build/bridge.o` path and the
system linker. Its no-flag, two-step Float32 control reproduced the prior
final latent SHA-256 exactly:
`807537b77bb0731e9e71f8d73dc6e5035fbe12c187f0ed55bf616c7b2afd24b4`.
The BF16 two-step smoke also completed and wrote a distinct final hash
`947093960823750ee5bf0e73fc02ca3f38e38659ab655cc9eee866e7ee36e905`.

For the full comparison, both saved Float32-time 40-step trajectories were
replayed with the new BF16-time option, using the same pinned Q4 GGUF
(`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`),
precomputed official/native Qwen3VL conditioning payloads, seed 7, 512x512
geometry, and Euler solver. Here “official” names **only the Qwen3VL text
conditioning**; all four trajectories use the native GGUF/Metal DiT. Each
new arm produced 40 finite 262,144-byte post-Euler snapshots and a final
snapshot byte-identical to its bundle. The first BF16-vs-Float32 difference
occurs at step 1 in both arms; step 0 is bit-identical within each arm.
Measured denoise times were 627.613 s (official text) and 613.462 s (native
text), sequential observations, not a performance comparison.

The table reports relative L2 percentages of *matched post-update latent*
differences. “Time effect” compares BF16 vs Float32 time with text fixed;
“text effect” compares native vs official text with time mode fixed.

| Step | Time effect, official text | Time effect, native text | Text effect, Float32 time | Text effect, BF16 time |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000% | 0.000% | 0.0207% | 0.0207% |
| 10 | 0.313% | 0.285% | 0.577% | 0.306% |
| 20 | 1.803% | 1.031% | 4.516% | 3.624% |
| 30 | 4.471% | 2.178% | 9.093% | 7.580% |
| 39 | 6.077% | 3.094% | 11.165% | 9.329% |

The BF16-time final latent SHA-256 values were
`979fb31de351594140a21bbbd0c2c8b25fbf599c873bf6c0f0033f150e8a357b`
(official text) and
`1489585ba0897346e04b0729a604a1218435fb9b21f7e5b17a8a47aaf8bbf2ea`
(native text). In the same previously selected 60x69-pixel face ROI, the
cross-text RGB MAE decreased from 12.36/255 with Float32 time to 3.64/255
with BF16 time; full-frame cross-text MAE decreased from 4.37/255 to
3.31/255. The BF16-vs-Float32 pixel MAE was 11.74/255 in that face ROI for
official text but only 1.06/255 for native text. These are descriptive
same-seed differences, not ground-truth image-quality scores. Visual
inspection found the composition and broad sign preserved but some clothing
and face shading changed; no defensible improvement in eye or glyph fidelity
was established. The VAE was held fixed and decoded both outputs successfully,
so these changes originate before VAE; neither an off-manifold latent nor a
VAE defect is demonstrated.

The replay inputs, compiled runner, four endpoint bundles, two new PNGs,
per-step snapshots, and comparison script are under
`/private/tmp/qwen21-bf16-time-russian-20260927.nKrSo3/` and may expire.
The comparison script checks manifest identity, finite snapshots, final
snapshot/bundle equality, and first divergence; its spatial ROIs are only
approximate latent regions, not VAE-exact pixel maps. A full official BF16
DiT trajectory on the same conditioning and initial latents was subsequently
captured below. Native Float32 latent states, the Float64 intermediate trig
in native timestep embedding, mixed GGUF quantization, and the Qwen3VL
projection mismatches remain separate candidate sources. Recheck after
source, compiler, GGUF, conditioning, VAE, Torch/Diffusers, device/OS, or
Metal-route changes.

### Matched official BF16 DiT latent trajectory (2026-09-27)

The pinned Hugging Face BF16/MPS DiT was run for all 40 steps with the
pipeline-default KV cache, the exact official-Qwen3VL conditioning payload
SHA-256 `007ad14a01440a9f786ae874e78cb4aef3a3729d2768fec1a503363bf66114ab`,
and the same BF16-exact initial target latent SHA-256
`d03158064c86fd691927cf93258b00fac2fc8d09e14a578d0f28b4fe8358932c`
consumed by the native runs. The source
checkpoint revision was `790c92633540aa0cb11d9abf19eb46d861714758`;
the compared native GGUF SHA-256 was
`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`.
Each of the 40 official BF16 post-step states was widened to Float32 for
comparison.
The initial latent was byte-identical; independently, the same-input native
step-0 velocity and Float32 Euler update reconstructed the saved native
step-0 state bit-for-bit (65,536 values). An initial apparent noise mismatch
was an audit error: a *final* native latent bundle had been mistaken for x0.

| Completed steps | 1 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: |
| Native BF16-time post-state relative L2 to official BF16 state | 0.1805% | 2.3936% | 9.5513% | 22.7345% |

The denominator is the official state norm. This is a progressively growing
trajectory difference, not a measured VAE failure or a perceptual quality
score. At step 40, the older matched-input Float32-time native trace was
23.6411% from the official state, versus 22.7345% for native BF16 time;
the improvement in this numerical proxy does not establish better faces or
text. Both traces used the *official* text payload. The separate native-text
trace has a different conditioning SHA and was not used for this comparison.

Independent teacher-forced DiT forwards held the BF16-exact latent, text,
mask, shapes, and effective timestep fixed. Native mixed-Q4 GGUF/Metal versus
official BF16/MPS target-velocity relative L2 was 2.7252% at index 0,
1.3638% at index 20, and 4.6690% at index 39 (official velocity norm as
denominator). These include quantization, kernel, arithmetic, and provenance
differences; they do not isolate any one operation. The saved official MPS
post-states at indices 0, 1, 19, 20, and 39 were reconstructed bit-for-bit
(65,536 values each) with Float32 `dt` times BF16 velocity, a BF16-rounded
product, Float32 addition, and a BF16-rounded state. Pinned Torch CPU scalar
promotion instead pre-rounds `dt` to BF16 and misses 270 post-step values
already at index 0. The MPS trace checked its manual formula against the
scheduler at every step; it did not save the intermediate product, so exact
product behavior is inferred from the post-state rather than observed
separately. With the same teacher velocity but the native Float32 solver,
the first post-state was still 0.1671% away from official; the actual native
post-state was 0.1805% away. These counterfactual distances are not additive
causal shares.

At index 20, the native and official input states had already separated by
9.5513%. Holding the *native* DiT fixed while switching only its input from
the official state to the native state changed its velocity by 20.5430%
(denominator: native velocity at the official state). This is much larger
than the 1.3638% same-state native-versus-official velocity contrast, so
feedback through the evolving state dominates the realized local discrepancy
at this point. The state difference itself arose from earlier DiT and solver
differences; this is not an exoneration of DiT or proof that the state dtype
alone is the cause. A fixed CPU/Float32 VAE decoded the endpoints; its prior
local interpolation probe showed no decoder cliff, but did not prove that
the trajectories occupy the same learned manifold.

The official scratch source is under
`/private/tmp/qwen21-official-trajectory-20260927/`; its 40 snapshots,
manifest, and decoded image are under `official40-cache-default/`;
the native equal-input forwards and split report are under
`/private/tmp/qwen21-native-matched-forward-20260927/`. These are ephemeral
local artifacts. Refresh the comparison after source model, GGUF,
conditioning, scheduler, cache semantics, Torch/MPS, Metal kernel, or VAE
changes. The next controlled intervention is an opt-in BF16 latent-state
update, retaining the current Float32 path as rollback. Its matched 40-step
and decoded-image comparison follows below.

### Opt-in BF16 latent-state update and 40-step A/B (2026-09-27)

`QWEN_IMAGE21_LATENT_STATE_PRECISION=bfloat16` now selects BF16-exact Euler
states independently of `QWEN_IMAGE21_TIMESTEP_PRECISION`. It rounds the
initial state, each DiT target velocity, each Float32-`dt` product, and each
updated state to BF16 with round-to-nearest-even; stored snapshots remain
Float32 containers of BF16-exact values. This follows the observed official
MPS post-state arithmetic, not the pinned Torch CPU scalar-promotion path.
The default remains `float32` with its previous operation ordering; BF16
state with Adams-Bashforth 2 is rejected rather than implying unverified
official semantics. The output manifest records `latent_state_precision`.
The mode does not change GGUF weights or Metal transformer arithmetic.

The focused flow-match suite passed 29/29, including an official MPS
coordinate that distinguishes Float32 from BF16-pre-rounded `dt`, a 16-word
official first-step slice, finite/overflow guards, callback states, and
default-path parity. An opt-in two-step Metal smoke emitted finite BF16-exact
states and a final snapshot equal to its bundle. A no-flag two-step Metal
smoke on the final runner reproduced the prior Float32 bundle SHA-256
`807537b77bb0731e9e71f8d73dc6e5035fbe12c187f0ed55bf616c7b2afd24b4`.
The matched 40-step run
used the same source revision, GGUF, official-Qwen3VL payload, seed 7,
initial state, Euler schedule, BF16-effective model times, and fixed
CPU/Float32 VAE as the BF16-time-only control above. All 40 BF16-state
snapshots were finite and BF16-exact, and the final snapshot matched the
bundle. Its final bundle SHA-256 was
`4fb1a867d0772f93190f4e7e7aa4a6925d9465f4ea7cdbf086a388953e17fdb1`.

| Completed steps | 1 | 10 | 20 | 40 |
| ---: | ---: | ---: | ---: | ---: |
| BF16-time, Float32-state relative L2 to official | 0.1805% | 2.3936% | 9.5513% | 22.7345% |
| BF16-time, BF16-state relative L2 to official | 0.1569% | 2.3570% | 9.1774% | 21.2774% |

An independent read-only recomputation of all 40 paired post-Euler snapshots
found strictly increasing relative L2 **and** absolute RMSE at every completed
step. The BF16-state trace first exceeded 1% at step 7, 5% at step 15, 10%
at step 21, and 20% at step 35. The largest single relative-L2 increase was
0.8608 percentage points into step 23; the largest absolute-RMSE increase was
0.01142 into step 39. There is no isolated failed denoising step in this
trace: a nonzero first-step discrepancy grows under repeated DiT/Euler
feedback. This monotonicity does not assign a causal share to quantization,
Metal arithmetic, or the solver. All 40 official step files matched their
per-step hashes, and all 40 native BF16-state files matched the run log and
were finite and BF16-exact. The existing scratch comparison script cited
below checks source/payload/solver manifests, all steps, and endpoint hashes;
the all-step recomputation uses the same files and Float64 metric arithmetic.

The BF16-state trace was closer to the official *latent* state at all 40
steps, but the final 21.28% residual is large. A separate native forward at
index 20, holding the DiT implementation and text fixed but supplying the
new native step-19 state, reduced native velocity distance to the official
teacher velocity from 20.5982% to 19.0101%. For comparison, the same-input
native-vs-official DiT difference was 1.3638%. Thus matching the scheduler
state arithmetic reduces a feedback-amplified discrepancy; it does not
identify the first divergent internal DiT operation or isolate quantization.

The same decoder produced a 512x512 PNG for each endpoint. Whole-frame RGB
RMSE to the official image moved from 21.63 to 20.57/255, while a fixed
80x105 face rectangle moved from 26.65 to 26.79/255. The latter is slightly
*worse*, and these pixel distances are alignment-sensitive proxies, not
perceptual quality or evidence that the eye defects are fixed. Visually the
new and old native images remain similar and both differ from the official
vendor's clothing and face. The safe conclusion is better numerical
trajectory agreement, with face quality unresolved. Next discriminate the
first internal DiT boundary on identical x0/text/time: compare official and
native image/text projections before block 0, then block-0 output; only if
the pre-block tensor agrees should mixed-Q4 block weights and Metal kernels
be isolated with equal activation inputs.

The opt-in run, 40 snapshots, numeric comparison script, new-state index-20
forward, and decoded image are under
`/private/tmp/qwen21-bf16-state-20260927.b6UNJF/` and may expire. Refresh
after model/GGUF, conditioning, schedule, Torch/MPS, Metal, or VAE changes.

### Equal-input DiT internal drift at index 0 (2026-09-27)

The pinned official BF16/MPS and native mixed-GGUF/Metal DiTs were captured
on the same Qwen3VL conditioning payload (`007ad14a...66114ab`), initial
latent (`d0315806...8358932c`), and effective model time 1.0. The official
one-forward velocity reproduced its earlier BF16 SHA-256
`f35c27adee76539f7dbe2771213c3122402618d553fadf915cf31e99a7300401`;
the native 32-block resident route reproduced its earlier Float32 target
velocity SHA-256
`3d3d1112987eef9e1ebdbd36aeb6c2da955295d349f8096089f7792a3e707791`.
Full official BF16 tensors were made contiguous and transferred to CPU
*before* widening or slicing; native Float32 captures came from the actual
resident path. Both capture manifests checked tensor shapes, finite values,
file hashes, and the input identity.

Before block 0, the text prefix was 0.1831% and the image target 0.1657%
from official (relative L2). Rounding the native image projection to BF16
made 4,194,139/4,194,304 target values exact, with 0.000788% relative L2;
the text prefix still differed by 0.1208%, with only 429,348/942,080
BF16-rounded values exact. After block 0, relative L2 was 0.3622% for text
and 0.4398% for target. These are *equal-input internal* observations, not
evidence of a VAE failure or proof of a single bad block kernel.

The text projection was split further. Native text normalization, rounded
to BF16, matched the official output in all 942,080 values. The first
linear projection was the first measured text divergence: baseline
BF16-rounded output differed by 0.159681% and matched 630,210 values.
An isolated scratch intervention rounding *only* the normalized input
before that matmul reduced the projection difference to 0.004931% and
matched 941,807 values. Its full target-velocity difference to official
decreased only from 2.725202% to 2.702657%. This establishes a real
text-projection precision boundary but not the dominant full-DiT error.

An additional scratch control rounded all text projection stage boundaries
to BF16. Its final text projection difference after BF16 rounding fell
from baseline 0.120819% to 0.055212%, but its full target velocity was
2.725973% from official, slightly worse than baseline. At the GELU boundary
it still differed by 0.115575% (677,910/942,080 BF16 values exact), despite
the preceding linear output differing by only 0.004931%. A standalone Torch
2.6.0/MPS `GELU(approximate="tanh")` on the captured official BF16 input
reproduced the hooked official GELU output exactly; the same Torch CPU BF16
operator matched only 678,640/942,080 values (0.115504% relative L2).
Thus this runtime's GELU arithmetic is another specific text-stage parity
boundary. Neither text-only intervention established better image quality.

The checked GGUF payloads for `img_in`, `txt_in.text_norm`, both text linears,
both time linears, `modulation`, and `norm_out` equal the pinned official
checkpoint values bit-for-bit (the text norm is an exact BF16-to-Float32
widening). This rules out static conversion error for those weights, **not**
mixed quantization elsewhere: block-0 Q/K are Q8_0 and V is Q6_K. A
CPU-only dequantization audit against pinned official BF16 tensors measured
weight relative L2 of 0.57575% (Q), 0.57448% (K), and 1.83678% (V), with
direct `[out, in]` orientation. This is static weight error, not a measured
contribution to the generated image; it cannot explain a pre-QKV gap. The
first-block attention input already differs by 0.3777% for target tokens,
and the tanh gate by 0.2562%. A pinned Torch 2.6.0/MPS capture of the
official Fourier time features, cast to BF16 at the source module's consumer
boundary, matched an independent CPU reproduction in all 512 values (raw
BF16 SHA-256 `661d6707...0cb85`). These features differ from native
Float64-trig/F32 features rounded to BF16 in 5/512 values (0.02446%
relative L2). A scratch-only, full 32-block Metal forward replacing *only*
the native time features with those official MPS BF16 values (widened to the
resident Float32 input) passed a SHA guard immediately before
`timestep_linear_1`; its text-path tensor hashes stayed identical to
baseline. The official-relative target gate difference barely changed from
0.2561683% to 0.2561196%; the block-0 modulated attention input changed
from 0.3777360% to 0.3779579% (worse). Target velocity changed from
2.7252019% to 2.7201716% relative L2, with the intervention output SHA-256
`3c2276b92707d101a501e10eb2371800450739f771a1cb4a30d915323be1e066`.
Thus the five Fourier-value mismatches contribute a small amount to the
full output distance but do not explain the gate or first-block drift.

A separate paired Torch 2.6/MPS BF16 versus native Metal replay of the
timechain, using the exact official Fourier BF16 input and byte-identical
BF16 weights for both time linears, modulation, and `norm_out`, isolated the
next boundary. The native first time linear, rounded to BF16, matched the
official output in all 8,192 values. Without an intervening BF16 rounding,
the native SiLU output matched only 6,124/8,192 official BF16 values. Feeding
the *rounded* first-linear output to the same native Metal SiLU instead
matched all 8,192. The next native linear from that rounded SiLU output
matched 8,183/8,192 official BF16 values, leaving a small separate matmul
arithmetic difference. With BF16 rounding at each remaining timechain stage,
the second SiLU still differed at 9/8,192 values, modulation at 6/32,768,
and `norm_out` scale at 1/8,192. The selected block-0 gate differed at
1,254/5,136,384 BF16 values: only two modulation coefficients differed,
repeated over 1,024 target and 230 prefix tokens. The official standalone
modulation gate reproduced the full-forward, pre-`tanh` block-0 gate exactly,
avoiding a mistaken
pre-/post-`tanh` comparison. This establishes the first timechain divergence
under equal BF16 features as a missing intermediate BF16 boundary, not an
incorrect Fourier formula, wrong weight payload, or intrinsically different
SiLU formula. It does **not** establish that this boundary dominates
full-DiT velocity or visual quality. A guarded full-forward A/B used exact
official time features and only an RNE cast between the first time linear
and SiLU, comparing the consumed gate and velocity against the existing
official-feature native baseline. This guarded
32-block A/B used identical x0, conditioning, model, and official time-feature
hashes; image and text projections remained byte-identical. Although the
single cast fixes the isolated SiLU1 BF16 output, the official-relative
target velocity **worsened** from 2.720172% to 2.736765%. The target
post-`tanh` gate went from 0.256120% to 0.256490%, and the modulated
attention input from 0.377958% to 0.378126%. Its target velocity SHA-256 was
`cd43ab5ec0f978c205266a3704de1639be4eac0334b3a3170494964f7988d82e`.
Thus this first BF16 boundary is real but not an individually promotable
full-output correction. A separate scratch treatment rounded all six BF16
timechain boundaries with the same pinned inputs. It improved the target
post-`tanh` gate from 0.256120% to 0.169585% and the modulated attention
input from 0.377958% to 0.348730%, yet **worsened** official-relative target
velocity from 2.720172% to 2.868087% (its velocity SHA-256 was
`9b591779cbc737040e9b1a0ecebdf5c741e22c6b3e35f6df6de35dbf04d2964a`).
The two native velocities differed by 0.886629% relative L2. A root-run
independent repeat of the all-six-boundaries 32-block forward, with a
separate output directory, reproduced its target-velocity SHA-256 exactly.
These A/B results were independently recomputed from raw tensors; both
treatments kept the image/text projections and pre-block-0 tensor byte-identical to
the official-feature native baseline. This is a concrete local-parity versus
global-output mismatch. It does not identify the downstream cause or prove
a face-quality change. The next discriminator was an observational
per-block error map, followed by an equal-input block replay to separate
new local error from propagation.

Official internal captures and their manifest are under
`/private/tmp/qwen21-dit-internal0-official-20260927/`; native boundary,
text-stage, and intervention captures are under
`/private/tmp/qwen21-dit-boundary0-native-20260927/`; the hash-gated
comparator is under `/private/tmp/qwen21-dit-boundary0-compare-20260927/`.
The selected block-0 Q/K/V payload hashes and weight-error audit are in
`/private/tmp/qwen21-block0-qkv-weight-audit-20260927.json`.
The paired timechain captures, isolated Metal SiLU replay, and BF16 canary
are under `/private/tmp/qwen21-timechain-boundary-20260927/`; the independently
recounted staged mismatches prevent claiming exact parity.
The successful time-feature intervention is under
`/private/tmp/qwen21-dit-boundary0-native-20260927/time_feature_official_mps_intervention_retry1/`;
the first-time-linear BF16-cast full-forward output is under
`/private/tmp/qwen21-timechain-boundary-20260927/timechain_first_linear1_bf16_rne_full_forward/`;
the all-timechain-boundaries output is under
`/private/tmp/qwen21-timechain-boundary-20260927/timechain_all_boundaries_bf16_rne_full_forward/`;
the independent repeat is under
`/private/tmp/qwen21-timechain-boundary-20260927/all_boundaries_root_repeat/`.
The original all-boundaries scratch report's embedded
`build_command`/`launch_command` strings retain
the earlier runner basename, while the actual all-boundaries binary was
`native_timechain_full_forward_all` (SHA-256
`a9cada6be0afcb4d798c85a3bfab2a5724f342b326e227ab3c4151bc0afed894`).
Treat those two report strings as stale provenance, not executable evidence;
an earlier scratch attempt imported the repository transformer instead of
the instrumented copy and was excluded after its missing capture failed the
guard. The retry added a model-free import canary and resident-input SHA gate.
These artifacts are ephemeral. Refresh after model/GGUF, conditioning,
Diffusers/Torch/MPS, Metal kernel, time semantics, or VAE changes.

### Selected post-block DiT state map at index 0 (2026-09-27)

An observational six-checkpoint capture compared the pinned official
BF16/MPS DiT with the native mixed-GGUF/Metal resident path at the same
index-0 x0, official Qwen3VL conditioning, and effective model time 1.0.
The native run injected the *official* MPS BF16 Fourier time features,
widened exactly to Float32, so these measurements apply to that controlled
time-feature route rather than the default native Fourier implementation.
Official forward hooks saved complete BF16 module-return tensors before
widening; native same-command-buffer copies saved `next_hidden_buf` after
the final FFN residual, before each buffer swap. Both are joint
`[230 text, 1024 image, 4096 hidden]` states. All capture hashes, shapes,
finite checks, and input gates passed. The official target-velocity SHA-256
was the canonical `f35c27ad...0401`; native was the prior official-feature
control's `3c2276b9...e066`. The block-0 snapshot bytes also matched each
path's earlier independent capture, ruling out observable capture-induced
changes in those controls.

| Block output | Image-token hidden-state rel-L2 | Image-token hidden-state RMSE | Text rel-L2 |
| ---: | ---: | ---: | ---: |
| 0 | 0.4403% | 0.02869 | 0.3622% |
| 4 | 0.6849% | 0.05402 | 0.8139% |
| 8 | 1.6458% | 0.09784 | 1.2041% |
| 16 | 5.5086% | 0.27023 | 1.4255% |
| 24 | 6.4548% | 0.38007 | 1.5646% |
| 31 | 3.7403% | 0.84547 | 3.5111% |

Relative L2 uses the official state norm at the same checkpoint. Its image
drop from block 24 to 31 is **not** recovery: the official image-state norm
grows from 12,059 to 46,293, while image absolute RMSE more than doubles.
The largest sampled image relative-L2 increase is between blocks 8 and 16;
the largest sampled L2 change in the native-minus-official error vector is
later, between 24 and 31.
These are cumulative state differences, not isolated per-block causal
contributions. BF16-rounding native snapshots for comparison did not erase
the gap. The finer map below narrows the growth interval; neither map alone
can name a bad block or blame Q/K/V quantization, arithmetic, or the VAE.

The scratch-only official manifest, native report, raw states, and SHA-gated
comparison are under `/private/tmp/qwen21-block-drift-map-20260927/`.
An initial official capture attempted to hook the Float32 temporal projection
before its BF16 consumer cast; it was incomplete and is excluded. The
corrected capture hooks the BF16 input to the timestep embedder. These
local artifacts may expire; refresh after model/GGUF,
conditioning, Diffusers/Torch/MPS, Metal code, or time semantics change.

### Every-block DiT state map at index 0 (2026-09-27)

A second scratch capture saved all 32 post-block states on both routes with
the same inputs and official-MPS time-feature intervention as the six-point
map. The official and native velocity SHA-256 gates again matched
`f35c27ad...0401` and `3c2276b9...e066`, respectively. All six overlapping
block snapshots matched the previous run byte-for-byte on *both* routes;
all 32 full states were finite. This is a single equal-input DiT forward,
not a 40-step denoising trajectory. The numbers below are for the 1,024
image-token rows of the post-block hidden state, not decoded pixels:

| Post-block | Relative L2 to official | Absolute RMSE | Adjacent error-vector change RMSE |
| ---: | ---: | ---: | ---: |
| 0 | 0.4403% | 0.02869 | — |
| 8 | 1.6458% | 0.09784 | 0.07762 |
| 13 | 4.5896% | 0.24940 | 0.18692 |
| 14 | 5.1100% | 0.27161 | 0.20605 |
| 27 | 7.7751% | 0.39985 | 0.15834 |
| 28 | 9.0978% | 0.44294 | 0.21058 |
| 29 | 11.0634% | 0.52874 | 0.32815 |
| 30 | 13.1201% | 0.74408 | 0.52692 |
| 31 | 3.7403% | 0.84547 | 0.45536 |

The adjacent column compares `(native - official)` after this block with
that vector after the immediately preceding block, element by element.
It is not a teacher-forced measurement of the block's intrinsic error.
There is no single early discontinuity: relative and absolute differences
grow through blocks 9–14, and the largest adjacent image-token shift is
29→30. The apparent relative-L2 recovery at 31 is a denominator effect:
the official image-state norm jumps from about 11,615 after block 30 to
46,293 after block 31, while absolute RMSE increases from 0.744 to 0.845.
Post-hoc BF16 rounding of native snapshots does not remove the gap. This
map narrows where the *observed* divergence accelerates; a fixed-input
single-block replay is required before blaming block 30, its quantized
weights, or a specific Metal kernel. The replay below supplies that control.

The all-32 official manifest, native report, raw states, and CPU-only
comparison are under `/private/tmp/qwen21-block-drift-all32-20260927/`.
The reported percentages and RMSE values were independently recalculated
from raw Float32 files for blocks 0, 8, 13–16, and 27–31. These artifacts
are ephemeral; refresh after input, model/GGUF, runtime, kernel, or time
feature changes.

### Fixed-input block replay at index 0 (2026-09-27)

To separate accumulated input drift from a block's own discrepancy, a
scratch-only runner replayed native Metal blocks 13 and 30 with the exact
official post-previous-block BF16 states widened to Float32. The official
MPS BF16 modulation projection was widened exactly and selected per token
with the same prefix/target layout. A separate block-30 resident-route
canary used the native captured post-block-29 state and native modulation:
its standalone output matched the full native post-block-30 Float32 tensor
byte-for-byte (SHA-256 `c7cb3679...a900d`, zero RMSE). Thus the standalone
API reproduced this measured native block before teacher-forced comparison.
All input, source, tensor, shape, finiteness, and model hashes passed their
guards. The first sandboxed GPU attempt stopped before any block calculation
because `MTLCreateSystemDefaultDevice()` returned nil; one authorized retry
with GPU access initialized Apple M2 Max and completed. The failed attempt
is not counted as a model result.

| Native block on official input and official modulation | Target hidden-state rel-L2 to official output | Target RMSE | Prefix rel-L2 |
| --- | ---: | ---: | ---: |
| 13 (official post-block 12 input) | 0.8695% | 0.04725 | 0.2721% |
| 30 (official post-block 29 input) | 2.0084% | 0.11390 | 0.3088% |

For block 30, a further replay held **native modulation fixed** while
switching only the hidden input between official and native post-block 29.
The target error vector has the exact identity
`N(native_pre) - O_post = [N(official_pre) - O_post] + [N(native_pre) - N(official_pre)]`,
where `N` is the same native block with the same modulation. The observed
post-block-30 target discrepancy was 13.1201% relative L2 / 0.74408 RMSE.
The same-input block term was 2.0145% / 0.11425; the input-drift transport
term was 12.9649% / 0.73528, each normalized by the official post-block-30
norm. Their vectors had cosine -0.000192 and reconstructed the observed
error with zero Float64 residual. These percentages are vector norms, not
additive causal shares. The incoming target RMSE was 0.52874 after block 29;
the native block mapped that input difference to 0.73528 RMSE (1.39x in this
one direction). The transported input difference dominates the *cumulative*
post-block-30 discrepancy; this decomposition does not partition the
adjacent change in error vectors from block 29 to 30. A real ~2% local
equal-input gap remains. Changing native to official modulation
on the official hidden input moved the block-30 target output by only
0.01726 RMSE; it did not remove the local gap.

This locates numerical divergence inside the DiT, before the VAE, and
rejects block 30 as the sole origin of the 13.12% cumulative difference.
It does **not** establish which earlier block is the dominant source or
predict facial quality from hidden-state L2 alone. The guarded report and
raw outputs are under `/private/tmp/qwen21-block-fixedinput-20260927/` and
may expire. Refresh after model/GGUF, inputs, official runtime, Metal code,
or time/modulation semantics change. The controlled weight/route split below
further narrows the local gap.

### Equal-input DiT weight-versus-route split (2026-09-27)

A scratch-only three-arm replay held the official input and modulation fixed
at blocks 13 and 30, with the same prefix/target layout and Metal block
implementation. Arm A used the existing mixed-GGUF projection weights and
quantized kernels. Arm B dequantized the **same** six GGUF projections to
Float32 and used a common Float32 Metal GEMV route. Arm C used the official
BF16 projection weights widened exactly to Float32 on that **same** route.
The six projections were Q, K, V, attention output, fused gate/up, and MLP
output; Q/K norm weights remained native and matched the official widened
weights exactly. The official gate/proj order was checked against the pinned
Diffusers source and the native SwiGLU interpretation. A's two full-output
hashes matched the preceding fixed-input replay byte-for-byte. A second
run saved all six full Float32 outputs, whose hashes matched the first run;
their target metrics below were independently recomputed against the
SHA-gated official BF16 outputs widened to Float32.

| Block, target hidden state | A: mixed GGUF | B: same GGUF weights, F32 route | C: official BF16 weights, F32 route |
| --- | ---: | ---: | ---: |
| 13 relative L2 to official | 0.8695% | 0.8693% | 0.4154% |
| 30 relative L2 to official | 2.0084% | 2.0089% | 0.5785% |
| 13 absolute RMSE | 0.04725 | 0.04724 | 0.02257 |
| 30 absolute RMSE | 0.11390 | 0.11393 | 0.03281 |

At block 30, A and B differ by only 0.0289% of the official target-state
L2 norm, while B and C differ by 1.9214% of that same norm; at block 13
the corresponding distances are 0.0132% and 0.7715%. These distances are
not additive shares of the teacher error. They were recomputed from raw
outputs with the **official teacher** norm; the report's pairwise fields
instead use the right-hand arm's norm. Under the controlled F32 route,
the GGUF-versus-official weight payload accounts for much more of these
two *local* output differences than quantized-versus-F32 projection
dispatch does. The largest static weight error is the Q5_K fused gate/up
projection (about 3.74% relative L2 in block 13 and 3.79% in block 30);
V and MLP output are Q6_K (about 1.8–2.1%), while Q/K/attention output
are Q8_0 (about 0.5–0.6%). Static matrix error alone does **not** establish
which projection dominates the block-output error; a selective swap is
needed. An initial selective-swap scratch harness stopped in CPU preflight
before any GPU forward because its shard-size guard serialized a 4.26 GB
file size as Int32; the corrected measurement is recorded below.

Arm C's nonzero residual is not a pure Metal-arithmetic measurement:
the official BF16 weights are widened, but the block still executes via
native F32 Metal operations rather than official BF16 MPS operations.
This single-input, two-block experiment does not prove that a different
quantization policy improves the 40-step trajectory, facial detail, text,
or final images. The reports do not self-bind the scratch runner source or
native Git revision, so their method lineage must be refreshed for a later
audit; independent raw-output recomputation and the prior A canaries are
the available controls here. The first-run report and raw-output replay
manifests are
`/private/tmp/qwen21-quant-vs-metal-20260927/gpu_run.json` (SHA-256
`12a2e094...a5c7816`) and `gpu_replay.json` (SHA-256
`1b7d7e4b...839c30`); these scratch artifacts may expire. Refresh after
GGUF/model, source checkpoint, input/modulation, or Metal route changes.

### Fixed-input block-30 projection-family swaps (2026-09-27)

The follow-up scratch runner retained the same official pre-block-30 hidden
state, modulation, teacher output, and Float32 Metal projection route as arm B
above. It independently re-created B and C byte-for-byte before replacing one
projection family at a time with official BF16 weights widened to Float32;
native Q/K norm weights stayed fixed. The two control output SHA-256 values were
`55780d42...306e31a` (B) and `63610d86...2bed38` (C), identical to the prior
raw-output replay. CPU preflight pinned the official checkpoint shard, GGUF,
inputs, source SHA-256 `2c135852...d421833d2`, and compiled binary SHA-256
`cdc39bbf...138b1d29`. The sandboxed launch had no Metal device; one
device-enabled launch on Apple M2 Max completed all arms without changing those
inputs or the runner.

| Block-30 target output on identical input | Relative L2 to official | Absolute RMSE |
| --- | ---: | ---: |
| B: all six GGUF projections dequantized to Float32 | 2.008916% | 0.11393194 |
| Only fused gate/up from official | 1.209111% | 0.06857250 |
| Only MLP output from official | 1.818562% | 0.10313637 |
| Only V from official | 1.917261% | 0.10873390 |
| Q, K, and attention output from official | 1.992175% | 0.11298249 |
| C: all six projections from official | 0.578489% | 0.03280793 |

Root independently checked the SHA-256 and 20,545,536-byte size of all six
raw Float32 outputs and the teacher, verified finite values, and recomputed
every target relative L2 and RMSE in Float64. The fused Q5_K gate/up payload
is therefore the largest *single tested family* in this fixed-input local
block discrepancy: replacing it alone closes 55.9% of the B-to-C
relative-L2 **metric gap**. This is not an additive causal share or a proof
that gate/up dominates other blocks or the 40-step image. The four individual
output deltas nearly sum to C minus B on this input (interaction residual
0.0346% of teacher L2 norm), but this does not license global linearization.

The report has a metadata defect: its C and swap-arm
`weight_sources_and_f32_sha256` objects are empty because the scratch runner
cleared their mutable hashes before JSON serialization. This does **not**
change the saved raw outputs or their independently checked metrics; the
compiled source specifies each swap and the B/C exact-output controls anchor
the route. It does weaken standalone provenance of the per-arm weight hashes,
so a repack decision must recheck tensor inventories and raw donor bytes.
`block30_group_swap_report.json` has SHA-256
`757d2a8ea37d179add54a01ff7e423ea2b091418fd5f533cd8195c2ccaf550cf`
under `/private/tmp/qwen21-quant-vs-metal-20260927/`, alongside the preflight,
runner, and raw outputs; all are ephemeral. Refresh after model/GGUF,
checkpoint, input/modulation, Metal route, compiler, or hardware changes.
Next test a higher-precision gate/up donor with strict per-tensor inventory,
then repeat at earlier blocks and across the full denoising trajectory before
claiming improvement in facial detail or text.

### Spatial check of the final denoising latent (2026-09-27)

To test whether the visible face defects coincide with unusually large
*spatial* latent disagreement, the matched official and native 40-step
BF16-state final snapshots were compared separately from the index-0
DiT map above. Both files are finite Float32 interchange views of
`[32, 32, 64]` `tokens_hwc` states from the same prompt, seed 7, initial
latent, official Qwen3VL conditioning payload, and Euler schedule. Their
SHA-256 values are `73aeef25...1f368` (official) and `4fb1a867...7fdb1`
(native). The previously documented face pixel crop, x=320–379 and
y=150–218 inclusive in the 512x512 image, maps approximately to latent
rows 9:14 and columns 20:24 (20 of 1,024 tokens). This is only a coarse
spatial correspondence; VAE receptive fields cross crop boundaries.

| Final-state area | Relative L2 to official | Absolute RMSE | Share of squared error |
| --- | ---: | ---: | ---: |
| Whole latent | 21.2774% | 0.23186 | 100% |
| Approximate face ROI | 16.7089% | 0.17639 | 1.1304% |
| Remaining tokens | 21.3535% | 0.23283 | 98.8696% |

The same SHA-gated endpoint pair shows no gross per-channel scale drift:
across the 64 latent channels, native/official spatial standard-deviation
ratios range from 0.9523 to 1.0741 (median 1.0045), and the largest
channel-mean shift is 0.0973 official-channel standard deviations. The
median paired residual RMS is still 0.2504 official-channel standard
deviations. These are descriptive statistics from one image, **not** a
learned-manifold test: correlations, spatial structure, and the VAE's
nonlinear response could differ despite similar channel moments.

The face crop occupies 1.9531% of tokens but carries only 1.1304% of
the total squared latent discrepancy. Thus this one seed does **not** show
a face-localized excess of native-versus-official latent error. It does not
exonerate the DiT, show that facial details are perceptually correct, or
rule out nonlinear VAE sensitivity: a small error in a semantically delicate
region can still change an eye. Quantized weights and distinct execution
paths remain confounders. The paired snapshots are under
`/private/tmp/qwen21-official-trajectory-20260927/official40-cache-default/`
and `/private/tmp/qwen21-bf16-state-20260927.b6UNJF/official40/`;
the ROI coordinates and layout justification are in the earlier face/VAE
probe and this document. Refresh after trajectory inputs, scheduler,
model/GGUF, or VAE changes.

Upstream, the native token embedding lookup exactly reproduced official
Qwen3VL `hidden_state_000` for all 244 raw tokens (0/999,424 BF16
mismatches). A one-block native Accelerate sweep starting from that official
input differed from official `hidden_state_001` in 315,527/999,424 BF16
values, with relative RMS 0.2000%; all 244 rows had mismatches. Thus the
text-path difference is present before the first image-latent step.

### Matched Russian layer-0 operation trace (2026-09-27)

A subsequent 244-token, CPU/BF16 per-operation comparison used the same
Russian reference payload SHA-256
`c5487a29449bd7335fd781e02c8372496873ba264749fa1f32a1b8735d5258d6`
and local text checkpoint on both sides. The native side used the Accelerate
projection backend; the official side used PyTorch 2.6.0, Transformers
5.17.0, and SDPA attention. The official trace was gated against the pinned
`hidden_state_000` and `hidden_state_001` at 0/999,424 BF16 mismatches at
each endpoint. The native trace started with the same input exactly and
reproduced the one-block endpoint above. All 16 stage sidecars were checked
for shape, byte count, and SHA-256 before comparison.

| Layer-0 boundary | Official/native BF16 mismatches | Relative RMS difference |
| --- | ---: | ---: |
| Input token embeddings | 0/999,424 | 0 |
| `input_layernorm` | 321/999,424 | 0.0119% |
| `q_proj` | 7,586/999,424 | 0.0201% |
| Attention output | 57,848/999,424 | 0.0618% |
| Block output | 315,527/999,424 | 0.2000% |

The first divergent operation is `layers.0.input_layernorm`. Its 321 small
BF16 differences occur in 12 of the 244 raw rows, all among the 230 rows
retained for image conditioning; maximum absolute error
is `0.000244140625`. An equal-input replay loaded the same exact BF16 input,
the checkpoint's BF16 norm weight, and `eps=1e-6`: the native serial-F32
reduction and BF16 rounding reproduced the native sidecar exactly, while the
official PyTorch module reproduced the official sidecar exactly. PyTorch's
mean-square and the serial-F32 mean-square had different F32 bits on all 244
rows (largest absolute difference `4.13e-9`). Substituting PyTorch's
mean-square into the native-style normalization and rounding removed all
321 output mismatches. This identifies reduction order as the first local
arithmetic cause on this fixture; the native per-row variance was reproduced
from source, not captured directly from the running Crystal process.

It is **not** the only arithmetic difference: on the 232 rows where the
normalized inputs still match exactly, `q_proj`, `k_proj`, and `v_proj`
already differ by 284, 85, and 73 BF16 values respectively. Those cannot
be downstream of this layer's RMSNorm mismatch. An equal-input PyTorch
`nn.Linear` replay with the exact checkpoint BF16 Q/K/V weights reproduced
all three official projection sidecars exactly on all 244 rows; the
284/85/73 native differences remained on the 232 input-exact rows.
Spot-checking three rows and four channels per projection with row-major
scalar F32 dots reproduced the native output in all 12 selected elements
for each projection. This supports, but does not prove for the full matrices,
an arithmetic reduction-order explanation rather than a weight-layout error.
The old red-cube fixture likewise showed that two RMSNorm reduction
changes improved layer-0 parity but *worsened* the final 36-layer embedding;
therefore no native arithmetic change is promoted from this local result.
Any candidate needs a full-text-encoder and same-seed image A/B gate.

The matched official trace and norm replay live under
`/private/tmp/qwen21-russian-official-block0-iYwyqW/`; the native trace is
under `/private/tmp/qwen21-russian-native-block0-0VaL0X/trace/`. These
scratch artifacts may expire. Re-run after changes to the checkpoint,
Transformers/PyTorch version, projection backend, or norm implementation.

### Same-input Qwen3VL layer-0 BF16 SwiGLU falsifier (2026-09-28)

The 244-row Russian fixture has BF16 gate/up sidecars on both the native
Accelerate and official CPU/PyTorch routes. For each route separately, a
scratch Crystal harness reopened the production `Qwen3VLTextBlock` module
and called its actual private `silu_bf16` and `bf16` methods on the saved
gate/up pairs. A PyTorch 2.6.0 CPU replay computed `F.silu(gate) * up` from
the **same** BF16 inputs. The output BF16 bytes matched exactly for all
2,998,272 elements per route, including the 14 dropped and 230 retained
rows. The native-input output SHA-256 was
`5e6383217c334b6a25e53f511901dfed9093357e927a17878fcfc7e755c0318e`;
the official-input output SHA-256 was
`11e7644cafa39b7711db42b1c51c96b048dbc27ace4d5a791149b1969dd33b9c`
on both Crystal and PyTorch. A separate source-matched NumPy BF16-RNE replay
also had zero per-element mismatches against PyTorch on both inputs.
Native gate/up input SHA-256s were
`8252377427b6e7070a3c235c2ceeaa1c68f5d382903d028b49ca378545fcd497`/
`f467f8b0e1c1afb344348c93b4ed53829857112a63f5e58fd4cebd73c68801ae`;
official gate/up input SHA-256s were
`ca093d46b747bad6d61fdf2364fbfdc7ca79fcea400dff50f74f23c52e96ef54`/
`63252d9500ca29f4630aa789eeb90d2434b9889e70580fc92fb6710ca234e75d`.

This rejects a standalone BF16 SiLU/product-staging mismatch on these saved
inputs; it does **not** erase upstream gate/up projection differences,
measure `down_proj`, establish a full-layer/full-encoder parity fix, or
predict image quality. The harness is
`/private/tmp/qwen3vl_bf16_activation_audit_20260928.cr` (SHA-256
`74cccfc79882084138fdc13e9728a0f05c0bbeab4935e3248a206cec3297e2c4`)
against production module SHA-256
`412daaf2fcde2bb13f35a91035909eb00995aa162147ea8bc15cd7fd4d97944a`.
No model checkpoint or full inference was run. Scratch may expire; refresh
after input sidecars, Crystal activation source, or PyTorch CPU arithmetic
change. The next Qwen3VL accuracy discriminator should target a same-input
projection/reduction or `down_proj` boundary, not this activation alone.

The downstream causal claim remains deliberately narrow: different Qwen3VL
outputs produce a small step-0 latent difference that accumulates under the
same native Q4-labeled mixed-quant GGUF/Metal DiT. This experiment does
**not** compare that DiT to the official BF16 DiT, prove a DiT implementation
bug, or prove the native latent is outside the VAE's training distribution.
The local text-encoder shards also lack independent content attestation in
this probe.

Scratch logs and all per-step tensors live under
`/private/tmp/qwen21-latent-drift-20260926.2s838A/`; the VAE sensitivity
report is under `/private/tmp/qwen21-face-vae-probe-20260926-r1/`, and the
one-layer text probe under
`/private/tmp/qwen21-russian-text-layer-probe-20260926-r1/`. Those files
may expire. Re-run after changes to the processor, text checkpoint,
GGUF/Metal path, scheduler, or VAE.

### Full-matrix equal-input Qwen3-VL layer-0 projection replay (2026-09-28)

A CPU-only follow-up removed the weight and layout confound left by the
earlier scalar spot checks. It loaded only the three exact BF16 layer-0
Q/K/V tensors from the local text checkpoint and replayed the pinned
244-row stage sidecars with PyTorch 2.6.0 `F.linear` and the source-matched
Accelerate `cblas_sgemm` row-major/transpose call, each rounded to BF16.
The official replay reproduced all three official captures byte-for-byte;
the Accelerate replay separately reproduced all three native captures
byte-for-byte. Source/sidecar/tensor hashes, shape, output-channel-order
negative controls, and a second deterministic run gated this result.

| Projection on the 232 BF16-identical norm-output rows | Native/official BF16 mismatches | Relative RMS | Maximum absolute delta |
| --- | ---: | ---: | ---: |
| Q | 284/950,272 | 0.010825% | 0.001953125 |
| K | 85/237,568 | 0.002591% | 0.00048828125 |
| V | 73/237,568 | 0.004931% | 0.00048828125 |

Root independently checked the six replay-output hashes against the
captured sidecars, the current source hashes, and the equal-input row mask
and mismatch counts directly from raw BF16 files. The same input and weight
bytes, exact per-backend capture replay, and strong channel-order negative
controls localize these small discrepancies to the **CPU projection
arithmetic path** on this fixture. They do not determine the micro-level
FMA/reduction explanation, quantify the downstream RoPE/attention effect,
or predict final 36-layer text-conditioning or image quality. The text path
remains distinct from the full-Q8 DiT parity trajectory above, which used
official Qwen3-VL conditioning; this result cannot explain that DiT drift.

The two-run report is
`/private/tmp/qwen21-vl-block0-cause-20260928/report.json` (SHA-256
`6eeda4b1107ca1bbcf7a832a44a927170b99b1de91fe09805c40acd960b30bbc`);
runner SHA-256:
`4195148c674e002feb2f3e13f7a4d2f5d7ee4ef12b6fa232dadc270423255407`.
It was executed at source HEAD `6b749860f784f68092df6de5115c02ec19fbffbe`;
later documentation-only commits leave the two pinned source-file hashes
unchanged. The replay hashes the selected local weight tensors, but does
not independently attest the upstream checkpoint shard provenance. Scratch
may expire; refresh after source, local checkpoint, captured inputs,
PyTorch/Accelerate, BF16 conversion, or shape changes. An arithmetic
change would require a full-encoder and same-seed image A/B gate.

### Same-input Qwen3-VL layer-0 down-projection cut (2026-09-29)

On the pinned 244-row Russian fixture, a CPU-only replay reconstructed each
route's BF16 SwiGLU activation from its saved gate/up sidecars. Both
activation hashes matched the independent Crystal/PyTorch falsifier above.
The replay used the same checkpoint BF16 `down_proj` weight (SHA-256
`b02a7008533f273efc0aac03c494a4109ee0c907fe69e404bbc1d93fb838af06`)
with PyTorch 2.6.0 BF16 `F.linear` and the native route's Accelerate F32
SGEMM followed by BF16 rounding. Official-input/PyTorch reproduced the
captured official `down_proj` byte-for-byte, and native-input/Accelerate
reproduced the captured native `down_proj` byte-for-byte. All 16 sidecars
per route, identical block input/model/fixture pins, 244/14/230 row
accounting, and a zero-input/no-bias control passed. Root re-ran the
scratch replay and obtained the same report SHA-256.

Holding the native Accelerate backend and exact weight fixed while replacing
only its activation with the official-route BF16 activation gave:

| `down_proj` versus official capture | Native observed | Native backend, official activation |
| --- | ---: | ---: |
| BF16 mismatches, all 999,424 outputs | 393,091 | 732 |
| Relative RMS, all 244 rows | 0.212298% | 0.006760% |
| Squared output error, all 244 rows | 0.103897214 | 0.000105342 |
| Squared output error, retained 230 rows | 0.102498025 | 0.000103872 |

The intervention removes 99.898609% of this **operator-output squared
error** on all rows and 99.898660% on retained rows. The small remainder
matches the measured same-input PyTorch-versus-Accelerate projection
arithmetic gap (732 BF16 values on official activation); native activation
gives 781 backend mismatches. Native and official gate/up sidecars already
differ, while same-input SwiGLU is bit-exact, so the activation discrepancy
is inherited from upstream gate/up values rather than intrinsic BF16
activation staging. This does not yet separate post-attention norm input
drift from gate/up projection arithmetic, nor establish a full-layer,
36-layer conditioning, DiT, VAE, or image-quality gain. The equal-input
gate/up cut immediately below tests that remaining projection boundary.

The scratch runner is
`/private/tmp/qwen21-qwen3vl-downproj-cut-20260929/downproj_same_input.py`
(SHA-256 `1ad6e53e02be5d13e9f14373a6b154ae607603367dfd5a13c1c8f59b01ee04ff`);
its report is `/private/tmp/qwen21-qwen3vl-downproj-cut-20260929/report.json`
(SHA-256 `7ae74a775f7671196886a2b4dd8fd9c191d437622206c390eaa4d17488238380`).
Refresh after checkpoint, source/backend, PyTorch/Accelerate, fixture, or
sidecar-boundary changes; scratch files may expire.

### Russian layer-0 gate/up equal-input projection cut (2026-09-29)

A CPU-only replay used the same pinned 244-row Russian block-0 fixture,
including 14 dropped prefix rows and 230 retained rows, and the exact BF16
checkpoint `gate_proj` and `up_proj` weights. The official PyTorch 2.6.0
BF16 `F.linear` on the official captured `post_attention_layernorm` output
reproduced both official projection sidecars byte-for-byte. The native
Accelerate F32 SGEMM followed by BF16 rounding on the native captured norm
output likewise reproduced both native sidecars byte-for-byte. All 16
sidecars per route, model revision, payload, source fingerprints, weight
hashes, zero-input/no-bias control, and a deliberately reversed-channel
negative control passed. Root reran the scratch replay with the same report
SHA-256 and independently recomputed the observed errors from raw BF16
sidecars.

Holding the exact native projection backend and weight fixed while replacing
only its input with the official norm output gave:

| Projection versus official capture, all 244 rows | Native observed mismatches | Official-input/native-backend mismatches | Native observed squared error | Official-input/native-backend squared error | Squared error removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gate_proj`, 2,998,272 outputs | 781,510 | 874 | 0.398803834 | 0.000272334 | 99.931712% |
| `up_proj`, 2,998,272 outputs | 806,323 | 946 | 0.311308664 | 0.000314286 | 99.899044% |

On the retained 230 rows, the same counterfactual removes 99.935253% of
`gate_proj` and 99.906664% of `up_proj` squared output error. The captured
norm outputs themselves differ in 85,872 of 999,424 BF16 values across all
rows (85,300 on retained rows), with squared error 0.011901554. Using the
same PyTorch backend on native versus official norm input preserves nearly
the full observed projection error. Thus the large gate/up output gap on
this fixed fixture is inherited from their differing norm-output inputs;
the residual same-input backend arithmetic is small. This does **not**
identify why the norm outputs differ, prove a full block or encoder repair,
or establish an image-quality gain. The next text discriminator must split
the `post_attention_layernorm` input/residual path from the norm operator
under a matched block input, with a route-exact no-op guard.

The scratch runner is
`/private/tmp/qwen21-qwen3vl-gateup-cut-20260929/gateup_cut.py`
(SHA-256 `3636b22aa3acd1ef72adb3afcbb1e2fb589539b8f1f71dc12c9ab3975305bbfd`);
its report is `/private/tmp/qwen21-qwen3vl-gateup-cut-20260929/report.json`
(SHA-256 `3912f19dbe6af5f6073a8ebd8a93d45430c938ff15085c4ed5500527f7d79a4f`).
Refresh after checkpoint, source/backend, PyTorch/Accelerate, fixture, or
sidecar-boundary changes; scratch files may expire.

### Russian layer-0 post-attention residual/RMSNorm cut (2026-09-29)

A CPU-only replay used the same pinned 244-row Russian Qwen3-VL fixture and
BF16 checkpoint `post_attention_layernorm` weight. It reconstructed each
route's post-attention residual from the identical saved block input and
its own saved `self_attn.o_proj` output, then replayed that route's
RMSNorm arithmetic. Both the official and native replay matched their
captured norm output byte-for-byte: **0/999,424 BF16 mismatches** each.
The two BF16-add formulations also agreed for each route's operands.
Weight/model/payload/source hashes, all six consumed sidecars, a zero-input
no-bias control, and a reversed-channel negative control passed. Root
independently reran the final runner to the same report SHA-256 and
recomputed the observed norm-output squared error from the raw BF16 files.

| Comparison, all 244 rows | BF16 mismatches / 999,424 | Squared output error | Relative RMS |
| --- | ---: | ---: | ---: |
| Captured native norm vs captured official norm | 85,872 | 0.011901554 | 0.129916% |
| Native vs official norm arithmetic on the **same official residual** | 53 | 0.000013663 | 0.004402% |
| Official norm arithmetic on the **native residual** vs official capture | 85,945 | 0.011937540 | 0.130112% |

The reconstructed native and official residual inputs themselves differ at
83,784 BF16 values (squared error 0.003124782, relative RMS 0.085686%).
On the retained 230 rows, observed norm-output error is 85,300 mismatches
and squared error 0.011854107; same-official-residual arithmetic differs
at 53 values with squared error 0.000013663. Thus the large norm-output
gap on this fixture is inherited overwhelmingly from the post-attention
residual input, not the RMSNorm arithmetic. The same-input arithmetic
remainder is nonzero; these nonlinear counterfactuals are not additive
causal shares. The input drift originates upstream of this norm boundary,
but this cut does not distinguish the attention projection from its Q/K/V,
RoPE, or attention inputs, nor establish 36-layer conditioning, DiT,
decoded-image, or eye-quality benefit. The next text discriminator is a
matched-input cut within that attention branch, followed by full-encoder
and same-seed image A/B before any production arithmetic change.

The route formulas match the visible Transformers v5.17.0
`Qwen3VLTextRMSNorm.forward` and decoder residual expression. That norm
class is decorated with `use_kernel_forward_from_hub("RMSNorm")`, so source
text alone does not identify the active runtime kernel; the byte-exact
route replays certify this captured fixture, not universal backend
equivalence. The final scratch runner is
`/private/tmp/qwen21-qwen3vl-postattn-ln-cut-20260929/postattn_ln_cut.py`
(SHA-256 `c1b1dbea798d15df9f1f096f2c00b3797998b12b73814600e28649eeb0a622a8`);
the final report is
`/private/tmp/qwen21-qwen3vl-postattn-ln-cut-20260929/report_corrected.json`
(SHA-256 `b1b759d45f7cd15a1cf11c300320da084754ec3e647e92bc1bc8508f0b4ea6dc`).
The earlier `report.json` is superseded. These results were obtained at
HEAD `4af5669bc72b5f7d02539bf803b344c15b1775a2`; refresh after
checkpoint, trace, source/backend, BF16, fixture, or row-selection changes.
Scratch files may expire.

### Russian layer-0 attention output-projection crossover (2026-09-29)

A CPU-only, hash-gated 2×2 cut replayed the pinned Russian layer-0
`self_attn.o_proj` with both saved BF16 `attended` inputs and both
source-matched operator routes: PyTorch BF16 `F.linear` and Accelerate CBLAS
F32-to-BF16. Each route reproduced its own captured `o_proj` output
byte-for-byte (0/999,424 BF16 mismatches). The checkpoint BF16
`o_proj.weight`, common model/payload, both manifests and input/output
sidecars, and production-source hashes passed their gates. Zero/no-bias,
repeat, and reversed-channel negative controls passed. Root independently
reran the final runner to an identical report SHA-256.

| Layer-0 comparison, all 244 rows | BF16 mismatches / 999,424 | Squared error | Relative RMS |
| --- | ---: | ---: | ---: |
| Native versus official saved `attended` input | 57,848 | 0.000125138 | 0.061796% |
| Captured native versus official `o_proj` output | 161,871 | 0.002361587 | 0.082659% |
| Native projection route on **official** `attended` versus official capture | 480 | 0.000021552 | 0.007896% |
| Official projection route on **native** `attended` versus official capture | 161,871 | 0.002220906 | 0.080160% |

On retained rows 14–243, the observed `o_proj` output difference is 160,470
mismatches with squared error 0.002356449; native projection on the official
input leaves only 443 mismatches and squared error 0.000021447. Thus
substituting the official context into the unchanged native projection
removes 99.087413% of the *operator-output squared error* on this fixture.
The same-input backend gap is not zero (435 mismatches on native input,
480 on official input), but the dominant observed output difference is
inherited from the attention-context input. Both vector decompositions
close exactly and have negative cross-terms, so the component squared norms
are not additive causal shares. The saved traces do not independently log
the active runtime kernel identity; byte-exact route replay qualifies this
captured fixture only. The earlier saved-Q/K/V attention replay also found
the native attention-operator gap small relative to its input drift, but
Q/K/V projection, normalization, and RoPE contributions are not resolved
by this cut. Nothing here establishes full 36-layer conditioning parity,
DiT trajectory, decoded-image, eyes, or VAE benefit. The next text cut
should isolate Q/K/V normalization and RoPE under fixed inputs, then test
full-encoder and same-seed images before a production correction.

The scratch runner is
`/private/tmp/qwen21-qwen3vl-oproj-cut-20260928/oproj_cut.py`
(SHA-256 `5f03c3e9b3e8011122ff07e0c0c99a416ae8c51f52801bb732267fffef41ef33`);
its report is `/private/tmp/qwen21-qwen3vl-oproj-cut-20260928/report.json`
(SHA-256 `4d75f4634b99b40418086f4f3367b21701cfa54a000f759ea5b897524693b74d`).
The replay ran at docs-only HEAD `90051a795fca967632ee305bdc5d5b53dfe531ed`;
production-source hashes were unchanged. Refresh after checkpoint/model,
text trace/fixture, PyTorch/Accelerate/CBLAS backend, BF16 staging, or
operator-boundary changes. Scratch may expire.

### Russian layer-0 attention replay on saved inputs (2026-09-28)

A CPU-only, SHA-gated PyTorch 2.6.0 SDPA MATH replay separated the native
attention operator from the already-different post-RoPE Q/K and V inputs in
the pinned 244-token Russian Qwen3-VL trace. It used the official all-visible
mask with causal attention, 32 query and eight KV heads of width 128, and
four-way `repeat_interleave` for GQA. No model/checkpoint, Transformers, or
GPU was loaded. Replaying the *official* saved Q/K/V reproduced its attended
BF16 output exactly (0/999,424 mismatches), establishing a layout and
backend positive control. Noncausal attention and group-major KV repetition
were large negative controls (47.203% and 118.181% relative RMS to the
correct native-input replay).

| Saved-input comparison | BF16 mismatches / 999,424 | Relative RMS |
| --- | ---: | ---: |
| Official-input SDPA vs official attended | 0 | 0% |
| Native-input SDPA vs native attended | 727 | 0.004291% |
| Native-input SDPA vs official attended | 57,683 | 0.061854% |
| Captured native attended vs official attended | 57,848 | 0.061796% |

Thus the native attention arithmetic has a small but nonzero same-input gap
on this fixture. Feeding the already-diverged native Q/K/V through official
SDPA yields nearly the entire observed attended-output gap, so attention
arithmetic alone is not a plausible dominant explanation for this layer's
composed error. These norms are not additive causal shares: upstream
normalization, projections, and RoPE remain mixed in the native inputs.
This does not establish 36-layer text-encoder parity or image-quality impact.
In the current full-Q8 DiT parity trajectory the conditioning is the
*official* Qwen3-VL output, so this native-encoder gap cannot explain that
trajectory's same-input DiT latent drift.

The scratch replay script and report are
`/private/tmp/qwen21-russian-attention-replay-20260928/replay.py` (SHA-256
`3d4a01d81be3fc0577d2d574e3f3267da8752efa1cb36cd0babc902c58e7b453`)
and `report.json` (SHA-256
`f0f86a827adb03675ab04e49dc0a312348a7d213579d713dae8a39afd36bbf6b`).
The original replay was pinned to source HEAD
`7aa36d9c202ec10eb007944b7ba6dd798c5fefb0`. Root re-executed its
`build_report` under the later docs-only HEAD
`cec39845ef213a6aa9191c86c2752fc8bc026a27`, changing only the expected
HEAD guard in memory; all pinned code and sidecar hashes passed, and every
comparison and replay-output hash matched the report. Scratch sidecars may
expire. Refresh after changing model/prompt, native attention source,
PyTorch SDPA backend, head/mask layout, or saved traces.

### Portrait resolution and latency probe (2026-09-24)

The daylight prompt above was also run through the unchanged hybrid path on
an Apple M2 Max, with pinned official CPU/BF16 text conditioning, the
Q4-labeled GGUF DiT on Metal, seed 7, and offline CPU/FP32 VAE decode.
The 512-, 768-, and 1024-pixel conditioning bundles retained identical text
embeddings and masks;
their initial-noise shapes necessarily differed. The immutable Metal runner
SHA-256 was
`39aa3ca65937b101033bd7bbce8fc2efe17649abd523b06e66824779110438f7`.
The source revision was `3782f406a0db5f0f4ee461a354ec7989b735ff3c`;
the GGUF and VAE SHA-256 values were
`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`
and `a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417`.
The 768-pixel conditioning payload SHA-256 was
`14f1c790d0edeb86e007ee61a43e8495d3eec02b83a08fd43e9faef34edcb86b`.

| Image size | Target latent tokens | Steps / path | DiT denoise time | Result |
| --- | ---: | --- | ---: | --- |
| 512x512 | 1,024 | 2, ordinary | 18.077 s | Latent payload emitted; timing probe only |
| 768x768 | 2,304 | 2, ordinary | 58.364 s | Latent payload emitted; timing probe only |
| 768x768 | 2,304 | 2, phase-split diagnostic | 113.856 s | Latents byte-identical to ordinary path |
| 768x768 | 2,304 | 40, ordinary | 2,006.751 s | Complete latent payload and decoded PNG |
| 1024x1024 | 4,096 | 2, ordinary, first attempt | Not available | A native command-buffer wait exceeded the 120 s watchdog; no latent output |
| 1024x1024 | 4,096 | 2, phase-split diagnostic | 131.021 s | Finite latent payload; no decoded PNG |
| 1024x1024 | 4,096 | 2, ordinary, one retry | 126.964 s | Finite latent payload, byte-identical to phase-split |

The 768x768 ordinary 40-step runner exited successfully after 2,006.94 s
wall time (33 min 27 s). Conditioning preparation took 23.75 s, and the
separate VAE decode took 53.58 s, for about 34 min 44 s from existing weights
to PNG, excluding downloads and environment setup. The PNG is 768x768 RGBA
with SHA-256
`306348ec73e07264d3edc263ecb6b86cf354f35feac98b493e860ae3c03b1926`;
the 589,824-byte float32 latent payload has SHA-256
`3f0005070d50de6240f59dd6bd93f9ac8bf8f6c17490a48250a150901ac400f6`.
All 147,456 latent values are finite. The portrait is visually photographic,
but its composition differs from the 512x512 output; this is not a controlled
causal proof of quality improvement from resolution alone.

The 768x768 two-step phase-split run tested the diagnostic path with the
default watchdog and produced exactly the same 147,456 float32 values as the
ordinary two-step run (`max_abs=0`), at 1.95x its denoise time. This phase
mode changes command-buffer boundaries and the projection route, so its
latency must not be reported as the ordinary path's speed. The ordinary
1024x1024 first-attempt failure hit the repository's per-command-buffer safety
watchdog, not an official Qwen-Image resolution cap or a proven out-of-memory
condition. A subsequent two-step phase-split diagnostic completed at 1024x1024
without changing that watchdog (131.157 s process wall time), emitting a
1,048,576-byte payload with all 262,144 float32 values finite, SHA-256
`8bee97ed578ab3ea628d7fda6d836188f6f6a76892ca24cdd18305cf8904c602`.
One ordinary-path retry then succeeded with the same unchanged watchdog:
126.964 s denoise, 127.103 s process wall, and a byte-identical finite latent
payload (`max_abs=0`). The first failure therefore did not reproduce under
this retry; its underlying cause remains unknown. Both successful runs are
execution-capability probes, not completed 1024-pixel images or sustained-run
latency measurements. Their total denoise times exceed 120 s because the guard
applies to each native command-buffer wait, not the entire run.

The current input validators accept dimensions divisible by 32 from 32 to
4096 pixels per side, but that is a syntactic limit, not a measured runnable
maximum. The [official Qwen-Image-2.1 model card](https://huggingface.co/Qwen/Qwen-Image-2.1)
shows a 2048x2048, 40-step example and presets near 4.2 megapixels; it does
not declare those dimensions an absolute maximum. Our highest completed
prompt-to-PNG result here is 768x768; the highest successful two-step latent
probe is 1024x1024. The current attention kernel scans keys for each query,
so its image-token work grows quadratically. Accordingly,
the observed 512-to-768 two-step denoise increase is 3.23x for 2.25x as many
pixels; the successful 768-to-1024 two-step comparison is 2.18x for 1.78x as
many pixels. The 768 two-step time extrapolates poorly to 40 steps:
58.364 s x 20 would predict 19 min 27 s, while the measured run took
33 min 27 s. Do not project full-image latency from a two-step probe without
sustained-run timing. These artifacts live under
`/private/tmp/qwen21-resolution-probe-20260924-r1` and are ephemeral; model,
conditioning, runner, driver, or host-pressure changes invalidate the timings.

## Goal

Admit Qwen-Image 2.1 weights into the native Metal engine without treating a
file name such as `Q4` as evidence of its actual tensor policy.

## Text-to-image prototype (2026-09-22)

The narrow text-to-image path uses the existing GGUF Metal DiT and FlowMatch
driver, with the official Qwen3-VL text encoder/processor and 64-channel VAE
running through local PyTorch/Diffusers components. Diffusers does not execute
the transformer. This is a hybrid integration, not a fully native image engine.
The initial route uses a deterministic seed, explicit pixel dimensions, and
40-step guidance-free sampling.

- **Admitted after evidence:** a versioned, shape-checked handoff of Qwen3-VL
  prompt embeddings, valid-token mask, target noise, and target shape to the
  native denoiser; a versioned, shape-checked handoff of its final latents to
  the reference VAE decoder; and a PNG produced from an actual prompt with
  the pinned real GGUF. On 2026-09-22, `red cube`, seed 7, 256x256, 40 native
  DiT steps produced an RGBA PNG (SHA-256
  `396ee177a3689ca7d9a035ab4d26ecd58a065215017df56ee64fb1d7084fd841`).
  The official component revision was
  `790c92633540aa0cb11d9abf19eb46d861714758`; the Diffusers source was
  `8b3c707ebd3ec4881f4190cf42931da07eaf3b65`. The finite real-prompt
  two-step Metal regression and all 52 Qwen-Image 2.1 Crystal examples passed;
  the Python bridge tests passed 14 cases with one expected skip. A second
  40-step run with the default code path reproduced both latent and PNG hashes.
- **Guard:** Qwen-Image 2.1 consumes *unpatched* 64-channel VAE latents. Its
  spatial sequence is `height/16 * width/16`, and each DiT token is one VAE
  spatial location. The VLM image-placeholder mask still expands one slot to
  four image tokens; that slot expansion is not 2x2 latent packing. Reject a
  bridge that uses the older Qwen-Image 2x2 pack/unpack or misaligns either
  mask with the target sequence.
- **Deferred in the initial prompt-to-PNG transition:** image-conditioned
  editing, classifier-free guidance, prompt rewriting, image-quality ranking,
  custom text-encoder/VAE kernels, and DiT speedups. Optimization is now a
  separate, measured transition; the other features remain deferred.
- **Falsifiers:** a mismatched binary length, non-finite float, wrong channel
  order, incorrect mask/placeholder count, wrong VAE normalization, or a PNG
  produced without executing the native GGUF denoiser. A smoke PNG is not an
  image-quality claim; visual inspection and a reference comparison remain
  separate checks.
- **Known guard:** the fused BF16 text projection originally yielded all-NaN
  output for real Qwen3-VL embeddings on M2 Max. A saturated-GELU guard now
  keeps its real projection finite, but `QWEN_IMAGE21_FUSED_TEXT=1` remains
  opt-in because full-run latency has not improved. The default route still
  uses separate Metal projections with host GELU. Fused timestep and resident
  block paths remain enabled.

The observed PNG supports basic prompt adherence only; quality, reference
parity, varied prompts and larger resolutions remain unverified. The evidence
decays if model revisions, GGUF packing, Diffusers behavior, Metal kernels or
the bridge schema change. Rollback is to remove the bridge and reference-side
scripts while retaining the earlier transformer/scheduler implementation.

With an isolated Python environment exposing Torch, Transformers 5.x,
Diffusers `QwenImage21Pipeline` and Pillow, and local official model components
under `$MODEL_DIR`, reproduce the same three boundaries as follows. `$GGUF`
must be the pinned file below; compile the native runner with the project's
Metal bridge before use.

```sh
MODEL_DIR=/path/to/Qwen-Image-2.1-components
GGUF=/path/to/Qwen-Image-2.1-Q4.gguf
crystal build --link-flags='-fuse-ld=/usr/bin/ld build/bridge.o -framework Metal -framework Foundation -lc++' \
  scripts/qwen_image21_generate_latents.cr -o /private/tmp/qwen-image21-generate-latents
python scripts/qwen_image21_prepare_conditioning.py \
  --model-dir "$MODEL_DIR" --output-dir /private/tmp/qwen21-condition \
  --prompt 'red cube' --width 256 --height 256 --seed 7
/private/tmp/qwen-image21-generate-latents "$GGUF" \
  /private/tmp/qwen21-condition/qwen_image21_conditioning.json \
  /private/tmp/qwen21-latents 40
python scripts/qwen_image21_vae_decode.py \
  --manifest /private/tmp/qwen21-latents/qwen_image21_latents.json \
  --model-dir "$MODEL_DIR" --output /private/tmp/qwen21-red-cube.png \
  --device cpu --dtype float32
```

These output directories must be new or empty. The prepared bundle records
the official model revision and payload SHA-256; the native reader checks the
schema, payload checksum and presence of a revision before running the DiT.
Reference components are loaded locally, without a network fallback during
generation.

## Optimization evidence (2026-09-23)

The pinned real `red cube`, 256x256, seed-7 path took 19.84 s for CPU prompt
conditioning, 57.99 s for 40 native Metal DiT steps, and 9.65 s for CPU FP32
VAE decode in one sequential cold run. These are stage observations, not a
repeatable end-to-end speed claim. A later `--device auto` conditioning run on
the same Mac selected CPU and reproduced the baseline binary payload SHA-256
exactly. Automatic MPS selection was removed because the default Qwen3-VL MPS
grouped-query attention path aborted; explicit MPS with eager attention is
experimental. One such run produced finite embeddings
and a PNG, but its final latents differed from CPU conditioning by relative
L2 `0.0207`, and quality across prompts is untested. CPU remains the safe Mac
default; CUDA selection is unchanged.

The fused Metal text chain's first BF16 projection was finite; its GELU
produced NaNs on large finite activations. Clamping only the saturated GELU
tails (`x > 10` to `x`, `x < -10` to zero) preserved finite real-model output:
the fused/separate text projections differed by at most `0.0002442`. The
40-step fused run had finite latents, relative latent L2 `0.000348` against
the default route, and a decoded PNG differing in 3216 of 65536 pixels by at
most two channel levels. Its 58.72 s denoise observation did not establish a
speedup against the approximately 58 s default. Keep the fused route opt-in.
The complete post-fix Qwen-Image Metal suite passed 54 examples against the
pinned real GGUF and conditioning bundle; the Python bridge suite passed 16
tests with one expected skip.

A run-local prompt-projection cache was prototyped and rejected for now. It
avoided the two text matrix multiplications on later steps, and all six
40-step runs preserved the baseline latent SHA-256. However, two on-first
pairs favored the cache slightly while a reversed off-first pair favored
uncached execution by 2.39 s. That falsified a robust speedup claim under
current host noise; the added mutable state and concurrency restriction did
not earn a place in the runtime. The evidence can be revisited if text
projection becomes a measured bottleneck at a different prompt scale.

A real 256x256 step profile attributes approximately 63% of diagnostic GPU
phase time to the Q8_0 Q, K, and attention-output projections and 10% to
attention itself. The diagnostic path splits a normal one-command-buffer step
into 418 buffers, so those shares rank optimization targets but do not predict
whole-step speedup. A proposed Q8_0 input-reuse tile passed exact output
parity but failed the normal-path latency gate below. Register-local reuse
across two output channels subsequently passed the bounded whole-forward
gate below on M2 Max, without staging input in threadgroup memory.

## Admitted now

- Parse GGUF `qwen_image` metadata without mapping the tensor payload.
- Support GGML BF16 tensor type 30 in byte sizing and CPU dequantization.
- Inventory actual per-tensor quantization types and storage sizes.
- Recover and validate ComfyUI `comfy.gguf.orig_shape.<tensor>` metadata.
- Validate the exact 32-block Qwen-Image 2.1 DiT tensor inventory.
- Report whether every tensor type is readable by the current Cogni-ML GGUF
  dequantization layer.
- Execute a parameterized one-block CPU reference with the upstream 2.1
  equations: affine-less LayerNorm, per-head Q/K RMSNorm, three-axis RoPE,
  block-causal attention, shared tanh modulation gates, and fused SwiGLU.
- Multiply BF16 GGUF matrices directly in the CPU reference path.
- Check the complete tiny block against an independent PyTorch oracle.
- Reject truncated GGUFs before exposing any mmap-backed tensor slice.
- Recover mathematical projection dimensions from Comfy `orig_shape` metadata
  while preserving ordinary GGUF `[in, out]` dimension semantics.
- Load all top-level and 32 transformer-block weights as zero-copy mmap slices,
  with exact per-projection shape validation and explicit mmap lifetime.
- Reuse the Qwen 3.5 Q8/Q5/Q6 Metal projection kernels inside the exact block
  reference. On two real tokens through block 0, the CPU/Metal boundary produced
  `max_abs=1.5258789e-5` and cosine `0.999999999999972` across 8192 outputs.
- Execute the exact batch-one outer transformer contract: zero-centered text
  RMSNorm and projection, VLM image-slot expansion, condition/target latent
  substitution, shape-derived image block ids, centered three-axis positions,
  padding validity, sinusoidal timestep embedding, causal `t=0` prefix
  modulation, all transformer blocks, adaptive output norm, and output head.
- Check that complete outer forward against an independent tiny PyTorch oracle,
  including the appended target placeholder slots used by the official
  pipeline. Adjacent image slots are explicitly kept as separate attention
  blocks, and prefix output is timestep-independent under causal conditioning.
- Execute the complete outer transformer with all 32 real mixed-quant blocks
  for a minimum valid `2x2` target. The measured hybrid run used 192 Metal block
  projections and produced finite output. Before the top-level BF16 Metal route,
  this minimum-size check completed in approximately `3.61` seconds.
- Keep a complete DiT block sequence Metal-resident behind one outer-transformer
  backend call. Affine-less LayerNorm and modulation, per-head Q/K RMSNorm,
  three-axis RoPE, segmented block-causal attention, SwiGLU, residual updates,
  and all six mixed-quant projections per block are encoded into one command
  buffer with shared scratch and ping-pong hidden buffers.
- Execute all 32 real mixed-quant blocks with one command buffer, zero
  intermediate readbacks, 192 projection dispatches, and one final hidden-state
  readback. Against the prior hybrid reference on the minimum valid `2x2`
  target, the outer output produced `max_abs=2.771616e-6` and cosine
  `0.9999999999998054`. A real single-block check produced
  `max_abs=0.00012588501` and cosine `0.999999999999791`.
- Cache the causal prefix K/V tensors for every DiT layer in persistent Metal
  buffers after the first transformer evaluation. Later FlowMatch evaluations
  recompute only the contiguous target suffix while assembling full attention
  K/V on-device and retaining one command buffer, zero intermediate readbacks,
  and one final readback. A real 32-layer check with an unchanged text prefix,
  changed target latents, and changed timestep recomputed four of five tokens
  and matched a complete hybrid reference with `max_abs=5.401671e-6` and cosine
  `0.9999999999988506`.
- Invalidate the prefix cache fail-closed when its token boundary, prefix hidden
  state, prefix modulation, prefix positions, prefix image ids, prefix key
  validity, layer objects, or block configuration changes. A tiny GPU check
  verifies both the cache-hit path and rebuild after a changed prefix value.
- Execute all eight top-level BF16 matrix shapes with a batch-capable Metal
  kernel; the empty-text minimum target evaluates six of them. The kernel reuses
  the registered whole-model mmap buffer plus tensor offset rather than copying
  weights into a second allocation, while synthetic heap-backed weights retain
  the existing per-weight upload fallback. A full real outer-forward check
  with one text token exercises all eight shapes; against a selective reference
  that leaves only BF16 on CPU it produced `max_abs=3.8146973e-6` and cosine
  `0.9999999999996982`.
- The prototype includes a fused text-projection/GELU command buffer and a
  separate fused timestep/modulation/output-scale command buffer. The real
  Qwen3-VL fused text chain is finite after the saturated-GELU fix, but remains
  opt-in without a measured whole-run gain; the separate BF16 Metal projections
  and host GELU remain the default text path. The fused timestep path remains
  in use. This is not a throughput claim.
- Keep the resident block result on-device through affine-less LayerNorm,
  per-token output scaling, and the BF16 output projection. The final head is
  encoded as the 193rd projection dispatch in the existing stack command
  buffer, so only the projected transformer output is read back. On a prefix
  cache hit, the persistent prefix output and resident target output are joined
  on-device before the final head; the target hidden state is not read back or
  re-uploaded. The real no-text 32-layer check produced
  `max_abs=2.1457672e-6` and cosine `0.9999999999997731`; the real changed-target
  prefix-cache check produced `max_abs=1.013279e-5` and cosine
  `0.9999999999977098`.
- For full passes without a reusable causal prefix, project `img_in` and the
  available timestep rows directly into the resident stack command buffer. GPU kernels
  assemble the mixed text/image joint sequence and select per-token modulation
  and output-scale rows; no image or timestep projection is read back to the
  host. Text normalization and its fused projection still return a host array.
  The real mixed condition-image/text/target-image check produced
  `max_abs=0.0039245486` against the hybrid reference and zero difference
  against the former stack input route. A separate causal two-row check
  produced `max_abs=2.3841858e-6`. The Qwen-Image hybrid reference now uses
  the same buffer-level K-quant GEMV route as the resident stack at all batch
  sizes; the general Qwen 3.5 Q5/Q6 batch GEMM route diverged on the mixed
  nine-token probe and is not used as this reference.
- Build and reuse causal prefix K/V with Metal-resident image projection,
  timestep projections, joint-sequence assembly, and row selection. A raw-input
  certificate compares encoder inputs and their projected text, condition-image
  latents, prefix source/layout metadata, the `t=0` timestep row, top-level
  projection tensor identities, text normalization, layer identities, and block
  configuration, assuming loaded tensor payloads remain immutable. Changed
  target latents and the active timestep row may hit;
  changed encoder or condition-image inputs rebuild. Prefix queries must not
  attend suffix keys. The real 32-layer text-prefix hit recomputed four of five
  tokens with `max_abs=1.013279e-5` against a full reference; a mixed
  text/condition-image prefix also hit and rebuilt on changed condition data.
- On a resident prefix hit with an ordered image-only target suffix, `img_in`
  uploads and projects only target rows directly into the active hidden buffer.
  Target modulation is selected for active rows; final output scales still
  cover the complete prefix-plus-target sequence. A nonconforming suffix takes
  the uncached full resident route. The real condition-image check projected
  eight image rows on build/rebuild and four on hit, while retaining output
  parity with the full reference at `max_abs < 0.05`. An eight-pair warm profile
  on M2 Max measured median full-forward wall time of `128.4 ms` on rebuild and
  `85.1 ms` on hit; complete GPU command time was `123.0 ms` and `80.7 ms`.
  The input-command encoding medians were `0.077 ms` and `0.052 ms`. This is a
  small paired hit-versus-rebuild observation, not an old-versus-new A/B or a
  component-level GPU attribution; DiT also processes fewer active tokens on
  a hit. A full resident pass over nine tokens and a cached pass over four
  cross the default eight-row GGUF GEMM/GEMV threshold: their outputs differed
  by `0.0021353364` maximum. Two independent full passes were identical, all
  32 prefix K/V tensors matched, and the difference survived temporary
  restoration of the former full-input preparation. With
  `QWEN35_GEMM_BATCH_THRESHOLD=16`, both paths used GEMV and this difference
  disappeared. Reproduce timing with `scripts/qwen_image21_prefix_profile.cr`.
- On the same host and minimum `2x2` target, the unfused outer-forward observation
  changed from approximately `3.61` to `0.92` seconds, and two FlowMatch steps
  changed from approximately `6.76` to `1.31` seconds. These are implementation
  smoke timings, not representative-token throughput claims.
- Profile the real 32-layer GGUF with 256 condition-image and 256 target-image
  tokens plus one text token (513 joint tokens), using synthetic embeddings and
  latents. On M2 Max, three warm paired samples measured median complete
  forward wall time of `5398 ms` for a cache rebuild and `2797 ms` for a hit;
  their GPU command times were `5356 ms` and `2757 ms`. A diagnostic mode
  splits the normally single DiT command buffer at phase boundaries, and
  matches the normal outputs exactly in this probe. Across two diagnostic
  samples, the combined Q/K/V plus output projections accounted for roughly
  73% of summed phase GPU time, attention roughly 12%, and cache-copy below
  0.2%. A further single diagnostic sample split Q, K, and V: on rebuild,
  Q=`1101 ms`, K=`1113 ms`, V=`78 ms`, output=`1107 ms`, and attention=`541 ms`;
  on hit, Q=`543 ms`, K=`543 ms`, V=`38 ms`, output=`541 ms`, and
  attention=`286 ms`. This is phase attribution under extra command-buffer
  submissions, not an end-to-end speedup or a production-resolution result.
  Q/K/output use Q8_0 GGUF tensors; at the time of this baseline, the buffer
  route selected Q8_0 GEMV for every batch size. The F32 prefix K/V allocation at this shape
  is theoretically `269484032` bytes; that is not a measured total Metal
  working set. The profile reports host buffer preparation and final readback
  separately, but does not isolate GPU transfer or launch cost from command
  waiting. Reproduce the old-route baseline with
  `QWEN_IMAGE21_Q8_BATCH=0 crystal run scripts/qwen_image21_prefix_profile.cr -- MODEL.gguf 3 16 16 2`.
- Use a Qwen-Image-local Q8_0 kernel that reuses each quantized weight block
  across eight adjacent batch rows for Q, K, and attention output projections
  at batch size `>=16`. Other weights and shorter batches retain the existing
  Qwen 3.5 route. `QWEN_IMAGE21_Q8_BATCH=0` restores the prior route without a
  weight-format change. A synthetic Q8_0 matrix test covers batch sizes
  `1,7,8,9,16,17,32,64,256`; a real mixed-quant block and three real-weight
  32-layer A/B shapes preserve output parity, with `max_abs=0` in the A/B
  profiles. After one warm pair, three alternating-order pairs on M2 Max gave
  median candidate/baseline wall ratios for rebuild/hit of `0.738/0.789` at
  16 target + 16 condition-image tokens, `0.631/0.632` at 64+64, and
  `0.693/0.671` at 256+256. These are scoped synthetic-latent, real-weight
  measurements, not production-resolution or image-quality evidence. The
  host's absolute timing drifted substantially between runs, so the paired
  ratios are more informative than cross-run millisecond comparisons. Set
  `QWEN_IMAGE21_Q8_BATCH_AB=1 crystal run scripts/qwen_image21_prefix_profile.cr -- MODEL.gguf 3 16 16 0`
  for the alternating route/parity probe.
- Reuse each Q8_0 input value across two output channels within a SIMDgroup,
  preserving the existing per-channel accumulation order and avoiding
  threadgroup storage and barriers. The automatic route is limited to exact
  `Apple M2 Max` and batch `>=256`; `QWEN_IMAGE21_Q8_REGISTER_REUSE=0` restores
  the established batched Q8_0 kernel, while `=1` forces the reuse kernel for
  eligible batched Q8_0 calls as an experiment. The outer
  `QWEN_IMAGE21_Q8_BATCH=0` rollback to the older Qwen 3.5 route is unchanged.
  Synthetic GPU checks compare the two Q8 kernels exactly across batch,
  output-channel, and odd-Q8-block tails with non-unit scales and a NaN mask.
  The bounded real-weight 513-token evidence and its scope are recorded below.
- With that Q8_0 batch route enabled, profile the same real 32-layer GGUF at
  larger synthetic condition/target image grids, each with one text token.
  After a warm pair, the `24x24 + 24x24` (1153-token) run used two normal
  rebuild/hit pairs and one diagnostic split pair; the `32x32 + 32x32`
  (2049-token) run used one normal pair and one diagnostic split pair. The
  diagnostic path matched normal outputs exactly on both rebuild and hit
  (`max_abs=0`). Its 418 command buffers make its phase sums unsuitable as
  direct estimates of normal one-command-buffer latency. Within each
  diagnostic pass, attention and the three Q8_0 Q/K/output projections took:

  | Joint tokens | Attention, rebuild/hit | Q/K/output, rebuild/hit |
  | ---: | ---: | ---: |
  | 1153 | 29.0% / 31.3% | 54.9% / 51.1% |
  | 2049 | 40.4% / 42.3% | 47.0% / 43.9% |

  Attention is the largest individual phase at 2049 tokens and approaches
  the three Q8_0 projections combined, but this single diagnostic sample per
  shape does not establish a stable crossover or a new kernel's speedup. The
  2049-token normal wall observations were `55.43 s` rebuild and `28.09 s`
  hit (`n=1`), with noisy host load; do not compare their absolute values to
  prior runs as an A/B. The theoretical F32 prefix K/V allocation at this
  shape is `1074790400` bytes, not a measured total GPU working set. Reproduce
  with `scripts/qwen_image21_prefix_profile.cr` using `1 32 32 1` for the
  2049-token probe.
- Reproduce the model's configured deterministic FlowMatch Euler schedule:
  linear input sigmas, exponential resolution shift over the exact
  `256..8192` sequence-length range, terminal stretching to `0.02`, and Euler
  updates. A two-step model-backed loop executed 396 Metal projections across
  two complete 32-block evaluations and produced finite changed latents.

## Pinned bootstrap artifact

- Repository: `realrebelai/Qwen-Image-2.1_GGUFs`
- Revision: `8d393b750593a72ee040fe7d40611f479ccee679`
- File: `Qwen-Image-2.1-Q4.gguf`
- Bytes: `5959127264`
- SHA-256: `51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`
- Actual tensor policy: `BF16:8,F32:65,Q8_0:96,Q6_K:64,Q5_K:32`

The repository README describes this policy as its Qwen-specific mixed Q4
release even though its recommended long `Q4_K_M-HQv3` label does not match the
current short filename. The loader trusts the inspected tensor directory, not
either label.

## Rejected attention probe

The one-SIMD-group-per-query/head prototype removed per-key threadgroup
barriers while retaining online softmax, the existing block-causal/image mask,
and O(1) per-query scratch. A real-weight, synthetic-latent full-forward A/B
warmed both routes and alternated their order over three measured pairs at each
shape. Median candidate/legacy wall ratios were:

| Joint tokens | Rebuild | Prefix hit |
| ---: | ---: | ---: |
| 513 | 0.984 | 1.031 |
| 1153 | 1.025 | 1.073 |
| 2049 | 0.983 | 1.096 |

Ratios below one favor the candidate. The 2049-token rebuild gain was only
1.7%, while the repeated cache-hit path regressed by 9.6%; both paths
regressed at 1153 tokens. Candidate-versus-legacy outputs had maximum
absolute differences up to 0.133 across these probes, exceeding the existing
0.05 resident-stack/cache parity limit despite cosine similarity above
0.99999. That numerical drift and the hit regression falsified promotion. The
candidate kernel, dispatch switch, and route-specific A/B harness were
discarded; the legacy kernel remains active. Host load varied, so these are
bounded paired observations, not a universal kernel ranking or an image-quality
result. A mixed text/image, invalid-key, prefix-build/hit/rebuild GPU
regression test remains as a guard for the next attention candidate.

## Rejected Q8 input-reuse tile

An opt-in 8-row by 8-Q8-block Metal threadgroup tile staged each input slice
once for four output channels, keeping the original quantized arithmetic and
output ownership. Synthetic parity covered 16- and 17-block K dimensions,
non-unit half scales, inactive output-channel simdgroups, and batch tails.
Two real-prompt two-step off/on pairs in opposite orders produced identical
latent payload hashes; the tiled route took 4.40/4.44 s versus 3.37/3.30 s
for the existing Q8 batch route, although those process-level totals included
pipeline setup. A warmed, alternating-order real-weight 513-token A/B then
confirmed exact full-forward parity but median tiled/default wall ratios of
1.375 on prefix rebuild and 1.371 on hit (`n=2` measured pairs, one warm pair).
The corresponding median one-command-buffer GPU times were 4036/2923 ms on
rebuild and 1999/1448 ms on hit. The candidate therefore regressed both
paths by roughly 37%; shared-memory loads and barriers are the leading
explanation, but their individual costs were not isolated. The prototype and
its A/B switch were discarded.
This rejection is scoped to the tested tile and host, not to every Q8 tiling
strategy or GPU. Reopen only with a distinct mechanism and a paired normal
full-forward falsifier, not an isolated kernel timing.

## Admitted Q8 register-local reuse on M2 Max

The register-reuse kernel computes two output channels per SIMDgroup over
eight adjacent batch rows. It reads each input value once for the two
channels, retaining the existing Q8_0 weight layout, block and lane reduction
order, and independent output ownership. It uses no threadgroup memory or
barriers. Synthetic Metal checks matched the established Q8 batch kernel
exactly across batch, output, and K-block tails, non-unit half scales, and
NaN propagation. The full model-backed Qwen-Image spec suite passed 56
examples with zero failures or pending cases after the automatic policy change.

The precommitted gate required exact output parity plus one warm pair and at
least three measured alternating-order, normal one-command-buffer A/B pairs
at 513 joint tokens; median paired wall ratios below 0.95 on both rebuild
and prefix hit, non-regressing GPU-command ratios, and no order reversal.
A forced-kernel screen against the established Q8 batch route gave median
reuse/baseline wall ratios of 0.896 on rebuild and 0.889 on hit, with GPU
command ratios of 0.895 and 0.886. Rebuild and prefix-hit full-forward
outputs matched exactly in every pair; the gain survived both execution
orders. These are full 32-layer forward passes using the pinned real GGUF
with synthetic input embeddings and latents, not an isolated projection or
phase-only speedup. The automatic policy is narrower than the forced-kernel
screen, so its dispatch was tested directly as well.

The final harness asserted the device name `Apple M2 Max`, compared the
automatic override-unset path against explicit rollback `=0`, and warmed both
routes before three alternating-order measured pairs. The normal 513-token
forward used one command buffer per rebuild and prefix hit. Median paired
auto/baseline wall ratios were 0.897 on rebuild and 0.890 on hit; GPU-command
ratios were 0.895 and 0.887. Every full-forward rebuild and hit output matched
exactly (`max_abs=0`), and neither execution order reversed the gain. The Q8
calls at this shape use 513 rows on rebuild and 256 on the hit target suffix,
both meeting the automatic batch threshold. Reproduce with
`QWEN_IMAGE21_Q8_REGISTER_AB=1 crystal run scripts/qwen_image21_prefix_profile.cr -- MODEL.gguf 3 16 16 0`
with the Metal bridge linked and the pinned GGUF; the harness
also retains `QWEN_IMAGE21_Q8_REGISTER_FORCE_AB=1` for the wider forced-on
experiment.

On the pinned real `red cube`, seed-7, 256x256, 40-step path, an automatic
policy run with the override unset took 48.113 s for denoising. Its latent
payload SHA-256 was identical to the rollback route's
`d835709261d2d36870ebd564cfb55d3d4db8a1173f1f8e8d511684c8eb7fa844`.
Two prior forced-on runs took 48.670 and 49.344 s versus one rollback run at
57.782 s. These sequential process observations support this one prompt and
host, but are not a paired end-to-end latency distribution; they do not
establish image quality or gains on another device, quantization policy, or
resolution. The unchanged latent bytes preserve the previous reference-VAE
decode input, but no new PNG was decoded in this optimization run.

## 768px portrait step-count and DiT timing probe (2026-09-24)

The same 768x768 photorealistic portrait conditioning (seed 7, Qwen model
revision `790c92633540aa0cb11d9abf19eb46d861714758`, conditioning payload
SHA-256 `14f1c790d0edeb86e007ee61a43e8495d3eec02b83a08fd43e9faef34edcb86b`)
was run through the native Metal DiT with 20 and 24 FlowMatch steps, then
decoded by the local CPU/FP32 Qwen-Image 2.1 VAE. The earlier 40-step run is
the reference. All three use the same GGUF (SHA-256
`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`),
whose actual tensor types are `BF16:8,F32:65,Q8_0:96,Q6_K:64,Q5_K:32`, not
literal Q4 despite its filename. The VAE safetensors SHA-256 is
`a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417`.

| Steps | Native DiT denoise | CPU/FP32 VAE decode | Output |
| ---: | ---: | ---: | --- |
| 20 | 1,024.582 s | 32.08 s | `/private/tmp/qwen21-step-quality-20260924-r1/20-metal/portrait.png` |
| 24 | 1,361.133 s | 24.22 s | `/private/tmp/qwen21-step-quality-20260924-r1/24-metal/portrait.png` |
| 40 | 2,006.751 s | 53.58 s | `/private/tmp/qwen21-resolution-probe-20260924-r1/768/full40/portrait.png` |

The 20- and 24-step latent bundles passed shape and finite-value checks
(48x48x64 Float32). Their decoded PNGs are 768x768 RGBA. Although the 20- and
24-step results share a rough composition, the eyes and ear lobes visibly
change; the 40-step result changes pose and crop enough to read as a different
portrait. Neither scene similarity nor pixel distance establishes facial
identity or image quality, and the 40-step output is a comparator, not ground
truth. The fixed seed and conditioning preserve the starting input, but each
step count constructs a different sigma grid. This observation alone cannot
separate normal trajectory sensitivity from a scheduler or port defect. The
pinned 20/24/40-step schedule comparison below now excludes a material
schedule-grid mismatch in the corrected source, but it does not validate the
earlier portraits against that source or establish their quality. The 20-step
run overlaps a brief unrelated Python import probe; none of these timings is a
quiet-host, repeated, paired A/B. The 24-step run's 22:41 DiT time is
disproportionately longer than the 20-step run's
17:05; do not infer a constant seconds-per-step rate or a universal quality
ranking. Host snapshots recorded no memory throttling or new swap-outs.

Opt-in per-step logging is now available with `QWEN_IMAGE21_STEP_TIMING=1`;
`QWEN_IMAGE21_TIMING=1` retains aggregate-only behavior. The per-step flag
also enables the existing one-command GPU elapsed-time profiler unless a
profile mode was already selected. A real 256px, two-step red-cube run produced
byte-identical latent payloads with and without per-step logging (both SHA-256
`02beea7d8d6688047f245742625a45179b41b540bb3e1c3d69d418417d9822c7`).
Its first pass built the causal prefix and the second hit it; each used one
command buffer and no intermediate readback. This establishes parity for that
model/input/step count, not every possible input. The focused FlowMatch spec
passed `6 examples, 0 failures`; the real-GGUF, real-conditioning Metal suite
passed `43 examples, 0 failures, 0 pending` with the new timer source. A run
without Metal device access only reported pending Metal cases and is not
counted as a passed model-backed suite.

An eight-step trace on the same 768px portrait reported GPU-command times
of `31.08, 30.61, 33.67, 38.71, 51.41, 64.16, 77.04, 63.42` seconds.
Every step used one command buffer and 198 projection dispatches. The first
step built the prefix; the other seven hit it, with 2304 active target tokens
on each hit. Wall step times tracked GPU-command time closely while rising
from roughly 31 to 77 seconds and then falling. This locates the variance
inside GPU execution, but does not identify thermal throttling, frequency
changes, or competing GPU activity as the cause. The eight-step latent bundle
is under `/private/tmp/qwen21-step-instrumentation-ab-20260924/trace768x8`.

The first DiT optimization candidate is now an opt-in tiled attention kernel
that reuses K/V loads across multiple queries while preserving block-causal
masks and online-softmax behavior. The earlier one-SIMD-group candidate failed
both numerical and prefix-hit latency gates; it is not a fallback. The full
32-layer output-parity and alternating-order, one-command-buffer paired A/B
gate at a real 768px token shape passed in the bounded run below, but noisy
host conditions and the missing multi-prompt decoded-image check still block
default promotion.
Reducing the number of steps is an explicit quality/latency trade, not an
exact-preserving kernel speedup. Do not promote 20 steps as a portrait mode or
default from this single sample. The schedule parity gate below is checked;
facial details across multiple prompts and seeds are not. A PyTorch MPS VAE
decode-only trial is a separate bounded follow-up: compare the same latent
against CPU/FP32 with fallback disabled, explicit MPS synchronization, and
raw/pixel parity.
VAE is called once, while DiT dominates the measured 768px path.

These `/private/tmp` artifacts are ephemeral. The input hashes, output paths,
runner logs, and exact source revision must be refreshed if the files are
removed or the model/runtime changes. The initial sandboxed 20/24 preflights
could not enumerate a Metal device; the successful runs used the same runner
with Metal device access outside that sandbox. This was an environment
permission boundary, not evidence of a model or numeric failure.

## Pinned 768px scheduler and causal-prefix parity gate (2026-09-24)

The portrait's model revision `790c92633540aa0cb11d9abf19eb46d861714758`
resolves to a `scheduler_config.json` with SHA-256
`5895f3a167c14a967fe9ac70c64924ae5acc79799e0679fd12907e594a713cd1`.
The reference runtime is Diffusers commit
`8b3c707ebd3ec4881f4190cf42931da07eaf3b65`, the commit recorded in the
conditioning bundle. `spec/support/qwen_image21_flow_match_reference.py`
requires those pins, checks the installed scheduler and pipeline source hashes,
and emits the checked-in `spec/fixtures/qwen_image21_flow_match_diffusers.json`.
The fixture uses the official Qwen-Image 2.1 pipeline's linear input sigmas and
resolution shift, then the real `FlowMatchEulerDiscreteScheduler` at 2304
target tokens. The Crystal spec compares every Float32 sigma and timestep at
20, 24, and 40 steps, not just the endpoints. It also checks the actual
transformer input against the pipeline's Float32 `timestep / 1000` operation.

The pre-fix exact-array spec failed. An isolated calculation of the old
Float64 formula found 12/20, 13/24, and 21/40 preterminal sigmas different
from the pinned reference, with a maximum absolute difference of `1.19e-7`.
The local schedule used Float64 shift/stretch intermediates where the
reference operates on Float32 arrays. Keeping those elementwise operations
in Float32 makes all three complete sigma and timestep arrays exactly equal
to the fixture. A further falsifier found one Float32-ULP model-input gap at
40 steps, index 26: reading raw sigma gave `0.48297691345214844`, while the
pipeline's `timestep / 1000` gives `0.4829769432544708`. `model_timestep`
now follows that division; the focused spec passed (7 examples, no failures or
pending cases). The model-backed Metal regression suite below passed 44
examples without failures, errors, or pending cases on the same M2 Max.
This is schedule parity, not an explanation for the visible 20/24/40-step
portrait differences. The earlier PNGs were generated before this numeric
correction and have not been regenerated or quality-ranked against it.

The real 768px, seed-7 conditioning bundle has 105 text tokens and 2304 target
tokens. `scripts/qwen_image21_cache_parity.cr` loads the full 32-layer GGUF,
builds a prefix on denoising step 0, advances the target latents by one Euler
step, then compares the step-1 cached hit against the same Metal stack's
uncached full resident-input route at identical latents and timestep. On an
Apple M2 Max, both the full 2409-token output and the target suffix had
`max_abs=0`, `RMS=0`, and cosine `1.0`. The cache hit processed 2304 active
tokens versus 2409 for the uncached forward; each evaluation used one command
buffer, zero intermediate readbacks, and one final readback. The probe was
rebuilt and rerun after both numerical corrections with the same parity result.
This rules out a cache-vs-full-output discrepancy for that one real transition;
it does not prove parity at later steps or other prompts, image quality, or a
latency gain. The subsequent tiled-attention A/B is reported below.

Reproduce with the pinned model config and Diffusers environment described
above, then run the focused spec. The GPU probe needs the local GGUF and
conditioning bundle plus Metal device access:

```bash
python spec/support/qwen_image21_flow_match_reference.py \
  --config /path/to/pinned/scheduler_config.json | \
  diff -u spec/fixtures/qwen_image21_flow_match_diffusers.json -
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk make build/bridge.o
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  crystal spec spec/qwen_image21_flow_match_spec.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  crystal build scripts/qwen_image21_cache_parity.cr \
  -o /tmp/qwen_image21_cache_parity \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"
/tmp/qwen_image21_cache_parity MODEL.gguf CONDITIONING.json 40
```

## Opt-in tiled-attention candidate (2026-09-25)

`QWEN_IMAGE21_ATTENTION_TILE=1` selects an experimental head-dimension-128
kernel when there are at least eight keys and four queries. The default and
explicit `=0` retain the legacy kernel. Four query SIMDgroups share an
eight-key K/V tile (8 KiB of threadgroup K/V storage), keep the ordered four
partial dot products and online-softmax recurrence, and preserve the
absolute-offset block-causal/image/invalid-key mask. This is cross-query K/V
reuse, not LTP/WBA. The first tile implementation retained two threadgroup
barriers per key; the later SIMD-local recurrence below removes them while
retaining a barrier at each K/V tile boundary. The speed result must come
from measured full forwards, not the barrier count alone.

The direct Metal falsifier used head dimension 128, five local queries at
absolute offset six, mixed image IDs and invalid keys, an 8+3 key tail, and
three inactive SIMDgroups in the final query tile. It covered both one head
with ordinary scores and three heads with much larger dot products.
Candidate and legacy outputs were finite and exactly equal (`max_abs=0`) in
both cases. With the candidate enabled, the full model-backed Metal regression
suite passed **46 examples, 0 failures, errors, or pending cases**. A
separately rebuilt real 768px prefix probe found
`max_abs=0`, RMS `0`, and cosine `1.0` between candidate cached hit and
candidate uncached output on the same step-1 input.

`scripts/qwen_image21_attention_ab.cr` pins the inspected GGUF SHA-256
`51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`
and seed-7 conditioning payload SHA-256
`14f1c790d0edeb86e007ee61a43e8495d3eec02b83a08fd43e9faef34edcb86b`.
It checks that both full and cache-hit query shapes select the intended
kernel, derives step-1 latents once from the baseline step-0 Euler output,
then holds those latents and timestep fixed across both attention modes. Each
mode executes a 32-layer prefix build, cached hit, and uncached resident-input
forward at the real 105-text + 2304-image = 2409-token layout. Every route
used one Metal command buffer, no intermediate readback, and one final
readback; the hit processed 2304 active tokens and build/uncached 2409.
In one warmup and all six alternating AB/BA measured pairs, candidate versus
legacy full and target outputs were exactly equal (`max_abs=0`, RMS `0`,
cosine `1.0`) on all three routes. Within each mode, hit and uncached outputs
were also exactly equal.

| 32-layer route | Legacy median wall | Tiled median wall | Tiled/legacy wall | Tiled/legacy GPU-command |
| --- | ---: | ---: | ---: | ---: |
| Prefix build | 30.784 s | 24.735 s | 0.804 | 0.803 |
| Prefix hit | 29.174 s | 23.030 s | 0.789 | 0.788 |
| Uncached | 31.706 s | 24.657 s | 0.778 | 0.778 |

All six paired ratios were below one for every route, including both
measurement orders. These are bounded observations on Apple M2 Max for this
one GGUF, conditioning bundle, timestep pair, and token shape; they are not a
40-step-image runtime or image-quality result. The host-load observer
reported `noise_observed=true` throughout much of the series, so do not
attribute the exact percentage to the kernel in isolation or promote it as a
general device default. No unrelated processes were stopped. The route stays
opt-in pending a quiet-host replication and a multi-prompt/seed 40-step
decoded-image quality check; `QWEN_IMAGE21_ATTENTION_TILE=0` is the immediate
rollback. The A/B runner reports performance without a speed hard gate, but
fails closed on pin, route, finite-output, and numerical-parity violations.

### SIMD-local recurrence: implemented, opt-in, not speed-promoted

The experiment removes the two *per-key* threadgroup barriers used
to publish and consume each query's softmax probability/correction. Each query
is already owned by one SIMDgroup, so the candidate broadcasts those two
lane-zero scalars within that SIMDgroup. K/V remain shared across query
SIMDgroups: an unconditional barrier is required after every key tile, before
any SIMDgroup overwrites the staged K/V for the next tile. Inactive query
SIMDgroups and partial final tiles must take the same barrier path.

The mask/tail/multi-head direct Metal spec passed (two cases, `max_abs=0`).
The pinned real 768px 32-layer build/hit/uncached warmup and one measured pair
each produced exact candidate-versus-legacy full and target outputs, and exact
hit-versus-uncached outputs within each mode. One paired 256px, seed-7,
40-step denoise produced identical post-Euler SHA-256 hashes at all 40 steps,
identical final latent bytes, and an identical CPU-decoded RGBA PNG. The
model-backed resident Metal suite passed 23/23 cases, the flow-match suite
passed 8/8, and the combined Qwen-Image 2.1 suite passed 62 examples with
zero failures, errors, or pending cases. The one 256px pair measured 83.10 s
legacy versus 77.77 s tiled for denoising, but the host was noisy; this is
not a speedup claim. The 768px one-pair forward ratios were 0.706 build,
0.838 hit, and 0.823 uncached,
also with `noise_observed=true`. These are correctness checks and a pilot,
not a statistically reliable latency result.

The subsequent full-resolution pilot used source revision `05430b52`, the
same pinned 768x768 portrait conditioning (seed 7, payload SHA-256
`14f1c790d0edeb86e007ee61a43e8495d3eec02b83a08fd43e9faef34edcb86b`),
GGUF, 40-step schedule, and guarded generator for sequential legacy (`=0`)
and SIMD-local tiled (`=1`) runs. Both exited successfully with 40 post-Euler
step hashes, one causal-prefix build and 39 hits, one command buffer per
step, 198 projection dispatches per step, and no intermediate readbacks.
The operator reported launch settings `QWEN_IMAGE21_ATTENTION_TILE=0` and `=1`;
the generator does not log selected pipeline names, so route attribution
follows the source policy at head dimension 128 and these logged token shapes,
not a per-step selected-kernel counter.
All 40 `(index, timestep, SHA-256)` records matched exactly. The final F32
latent binaries and JSON manifests were byte-identical (latent SHA-256
`37bc64dc8933dc97c07fdb4c8dd7de3aece7be16fbcd9ee830bc3abe1a523a84`).
Separate decodes through the same local CPU/FP32 Qwen-Image 2.1 VAE yielded
byte-identical 768x768 RGBA PNGs and pixels (PNG SHA-256
`c5682ea31fb5e544ff585d780d25a03cd2afa7cf9306e86cea2c7c49ffbd6a5a`).
The outputs and logs are under `/private/tmp/qwen21-simd-full768.CvZJti`;
these scratch artifacts are ephemeral.

Generator-reported DiT denoising took 1,294.247 s legacy and 997.297 s tiled.
This is one sequential correctness pilot, **not** a 22.9% kernel or
end-to-end speedup certificate: host load varied during the runs and the
post-run quiet-host gate was still false. The safety wrapper's approximate
elapsed display is not the denoising timer. The result establishes exact
conditioning-to-PNG parity for this model, pinned conditioning, seed,
schedule, resolution, and runtime; the Qwen3-VL encoder was not rerun in
either arm. It does not establish parity or quality across prompts/seeds or
justify default selection. Refresh the certificate after kernel/compiler,
model, conditioning, schedule, VAE, device/OS, or runner changes.

For future runs, `QWEN_IMAGE21_ATTENTION_ROUTE_TRACE=1` now records the
selected kernel after the actual Metal attention dispatch call and prints
aggregated `(kernel, head_dim, total_tokens, query_tokens, dispatches)` rows
after generation. It is opt-in and does not change the kernel or command
boundaries. A separately rebuilt, guarded two-step run on the same pinned
768px conditioning emitted 32 legacy dispatches on the 2409-query prefix
build and 32 on the 2304-query cache hit with `ATTENTION_TILE=0`; with
`ATTENTION_TILE=1`, it emitted 32 tiled dispatches on each route. Both runs
used one command buffer per step, matched both post-Euler hashes and final
latent bytes exactly (SHA-256
`d1384af943ceb45547cdd8a5a76e9c347d2ca6e3df3b5b79bedd19a880901988`).
The CPU-only Metal-spec invocation reported 11 examples, zero failures/errors,
and eight expected Metal-dependent pending cases; the direct
model-backed two-step runs both exited successfully with Metal access.
Scratch logs are under `/private/tmp/qwen21-route-trace.72JwfT`. This new
instrumentation cannot retroactively supply selected-kernel records for the
earlier 40-step pair. Enabled tracing adds host-side locking/hash updates per
attention dispatch, so its denoising times are not uninstrumented speed
measurements; the noisy two-step timing is not a speed certificate. For a
future latency gate, bracket trace-off measurements with trace-on route
controls at the same model, shapes, source, and settings.

Admitted behavior remains the legacy default and the experimental opt-in
`QWEN_IMAGE21_ATTENTION_TILE=1` path. A quiet-host alternating AB/BA replication
and more paired 40-step conditioning-to-PNG checks across prompts/seeds are needed
before a stronger performance or image-quality claim. Isolated kernel or
forward latency alone cannot promote the path. Rejected claims include an
exact cross-step activation cache, an LTP/WBA certificate, and a general
device-wide speedup.
`QWEN_IMAGE21_ATTENTION_TILE=0` remains the immediate rollback.

```bash
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  crystal build scripts/qwen_image21_attention_ab.cr \
  -o /tmp/qwen_image21_attention_ab \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"
/tmp/qwen_image21_attention_ab MODEL.gguf CONDITIONING.json --pairs=6
```

## Same-prompt low-step preview probe (2026-09-25)

The local Metal generator with `QWEN_IMAGE21_ATTENTION_TILE=1` and the same
Q4 GGUF produced additional seed-7 portrait latents. The 512px baseline
conditioning payload is SHA-256
`9fca4e6ebc57d573b28612def1482a4931e4108e05ccb345479ed6f900c4bc58`;
its 105 text embedding rows and masks are byte-identical to the 768px
conditioning payload above. The initial noise shape changes with resolution,
and FlowMatch also changes its resolution-dependent shift. Each step count
builds a new sigma grid; these are distinct generated images, not checkpoints
on one 40-step trajectory. All decoded outputs below used the same local
CPU/FP32 Qwen-Image 2.1 VAE.

| Resolution | Steps | One-run DiT time | Decoded PNG |
| ---: | ---: | ---: | --- |
| 768px | 5 | 139.224 s | `/private/tmp/qwen21-preview-20260925-r1/5/portrait.png` |
| 768px | 10 | 408.813 s | `/private/tmp/qwen21-preview-20260925-r1/10/portrait.png` |
| 768px | 16 | 390.859 s | `/private/tmp/qwen21-preview-20260925-r1/16/portrait.png` |
| 768px | 40 | 997.297 s | `/private/tmp/qwen21-simd-full768.CvZJti/tile.png` |
| 512px | 5 | 36.120 s | `/private/tmp/qwen21-preview-512-20260925-r1/5/portrait.png` |
| 512px | 10 | 90.909 s | `/private/tmp/qwen21-preview-512-20260925-r1/10/portrait.png` |
| 512px | 16 | 213.635 s | `/private/tmp/qwen21-preview-512-20260925-r1/16/portrait.png` |

On visual inspection, the 512px 5-step image is recognizable but visibly
soft and changes clothing/face relative to the 10-step output. The 10-step
image already has a coherent face, window scene, and clothing; 16 steps
change details but retain much of its
composition. Both differ materially in person, crop, and clothing from the
768px 40-step output. Thus 512px/10 is a promising *scene preview* for this
one prompt and seed, not a faithful preview of the final portrait identity.
The 768px 10-step run took longer than 16 steps, while the 512px 16-step
per-step average exceeded that of 10 steps. These unpaired, single-run timings
do not establish a stable latency curve or isolate a kernel speedup. They omit
text conditioning and VAE time; scratch PNGs may disappear.
The instrumented 512px/5 run reported one 9.066 s prefix build and four
6.747–6.791 s cache-hit steps, each with one command buffer, 198 projection
dispatches, and no intermediate readbacks. That narrow trace does not explain
the slower per-step averages of the separate 10- and 16-step runs.
An instrumented repeat of 512px/10 took 85.044 s versus the first 90.909 s;
both produced the same F32 latent SHA-256
`96487a1fadad8ee3ab4c2e4b8ca29bddecf50263973ff06d53747992c018fb2e`.
Its cache-hit GPU-command time rose from 7.624 s at step 1 to 9.139 s at
step 9 despite the same 1024 active tokens, 198 projection dispatches, and
one command buffer. Source inspection found no step-count- or timestep-driven
shape/dispatch change on this route. Device throughput drift is a hypothesis,
not an attribution; replaying a fixed latent/timestep through the same
cache-hit forward is the cheapest next discriminator.

The next algorithmic falsifier was an opt-in variable-step Adams-Bashforth-2
solver on the actual FlowMatch sigma grid, with Euler on the first interval.
It still costs one DiT evaluation per step; only maintaining image quality
with fewer steps would accelerate generation. Neither this solver nor
ordinary attention tiling is an LTP/WBA certificate.

### Experimental solver frontier (design-sealed; not quality-promoted)

- **Admitted default:** the official-compatible shifted FlowMatch sigma grid,
  model timestep, and first-order Euler update remain unchanged.
- **Guard-only opt-in:** variable-step AB2 may reuse the previous DiT velocity
  on that same grid, with Euler on the first interval and one model evaluation
  per interval. It must not silently select itself or change the default
  trajectory. Its first intended use is a preview-quality experiment.
- **Rejected claims:** matching Euler latents, preserving portrait identity,
  reducing latency at a fixed evaluation count, or promoting a universal
  preview mode from one prompt/seed.
- **Falsifiers:** nonuniform-step analytic ODE and constant-field unit cases;
  default Euler byte parity; finite latents and valid schedule bounds; pinned
  Metal generation and VAE decode at fewer evaluations across distinct
  scenes/seeds; and paired prompt-to-PNG latency with visual quality review.
  A numerical error improvement alone cannot promote the mode.

The first pinned 512x512, seed-7 Q4 pilot decoded the following images with
the same conditioning bundle for each scene. DiT times are isolated single
runs, not paired end-to-end measurements; the separate Euler-10 portrait run
from the new binary took 166.099 s versus 85.044-90.909 s in earlier runs,
so throughput drift is unresolved.

| Scene | Solver and DiT evaluations | DiT time | Observation against Euler-10 |
| --- | --- | ---: | --- |
| Portrait | Euler-8 | 101.611 s | Coherent face, but different pose/details. |
| Portrait | AB2-8 | 100.907 s | Coherent face, visually closer to Euler-10 than Euler-8 in this sample; latent RMSE 0.1685 versus Euler-8's 0.2005. |
| Portrait | AB2-5 | 58.730 s | Rough scene preview; pose, clothing, and identity still differ materially. Latent RMSE 0.3942 versus Euler-5's 0.4054. |
| Castle | AB2-8 | 85.216 s | Castle/forest/bridge composition visually close to Euler-10. |
| City | AB2-8 | 82.118 s | Train/station composition close; sign lettering differs and is not reliable. |

The default Euler-10 latent SHA-256 from the new binary exactly matched the
pre-change binary (`96487a1fadad8ee3ab4c2e4b8ca29bddecf50263973ff06d53747992c018fb2e`).
AB2-8 and Euler-8 had similar measured DiT time on the portrait: AB2 does
not accelerate a fixed evaluation count. Reducing 10 to 8 evaluations saves
two DiT calls by construction, but no general quality or wall-time claim is
admitted from these single-seed samples.

A subsequent paired two-seed portrait probe (2026-09-26) used the same pinned
Q4 GGUF, 512x512 official CPU/BF16 conditioning, Metal DiT, and CPU/FP32 VAE
for Euler-10, AB2-8, and Euler-8. Each seed's three arms consumed the same
conditioning bundle; seed 7 payload SHA-256 was
`9fca4e6ebc57d573b28612def1482a4931e4108e05ccb345479ed6f900c4bc58`
and seed 11 was
`8f78b243a902b5565175ef0b5313aadf9e6b2a9681188c2fa92badb61c259a2a`.
These are single-run times, not a latency distribution:

| Seed | Euler-10 DiT / bundle-to-PNG | AB2-8 DiT / bundle-to-PNG | Euler-8 DiT / bundle-to-PNG |
| ---: | ---: | ---: | ---: |
| 7 | 68.890 / 84.84 s | 55.020 / 67.85 s | 55.111 / 67.29 s |
| 11 | 70.949 / 84.99 s | 55.525 / 68.37 s | 55.292 / 67.59 s |

The seed-11 conditioner was timed separately at 22.69 s and shared by the
three arms; seed 7 reused an existing bundle. Thus the table does not claim
independently measured package-level prompt-to-PNG throughput. AB2-8 saved
two DiT evaluations and about 20-22% of denoising time against Euler-10 in
these two runs, but was within 0.23 s of Euler-8 at the same call count.
All six decoded PNGs are coherent portraits; expressions and facial details
differ, and neither seed gives AB2 a clear visual win over Euler-8. The two
sigma grids are different trajectories, so this is scene preview evidence,
not portrait-identity preservation. Artifacts are under
`/private/tmp/qwen21-ab-paired-20260926-r1/` and
`/private/tmp/qwen21-ab-seed11-20260926-r1/`; they may disappear. The
3-step lower bound, blind quality review, broader prompts/seeds, and paired
full-package end-to-end timing remain open before any preview-mode promotion.

The shelved CogniFusion experiment's adjacent-layer coherence loss and
adjacent-step consistency loss are **training** regularizers, not inference
updates for frozen GGUF weights. Its historical logs show backward NaN and
overflow-skipped updates, but do not establish that either regularizer caused
the instability. A 3-5-evaluation student would require a separate trainable
distillation path with explicit finite-gradient guards and a distillation-only
control; the present AB2 pilot does not test that hypothesis.

### Coarse-step student frontier (gradient feasibility only)

- **Admitted experiment:** load the official Qwen-Image 2.1 DiT from the pinned
  `790c92633540aa0cb11d9abf19eb46d861714758` transformer snapshot outside
  the repository. Freeze the base transformer and attach a small, explicitly
  enumerated LoRA adapter. Reuse a pinned, checksummed text-conditioning bundle
  and the model's own latent geometry; do not reload or train Qwen3-VL or VAE.
  The first real-weight probe uses two frozen teacher forwards and one LoRA
  student forward/backward at batch one and 256x256 resolution, with **zero
  optimizer steps** and no weight artifact.
  Source preflight on 2026-09-25 matched both shard SHA-256 values in the
  snapshot metadata (`9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b`,
  `3aaf234dcbe128530479735854a346b5e3e66283b7c11db56f836bbd1c13ebaa`);
  all 297 indexed tensors were present and BF16. Recheck after replacing the
  local snapshot.
- **Gradient gate:** fail closed unless the loss, outputs, and every trainable
  gradient are finite; at least one LoRA gradient must be nonzero; no base
  parameter may receive a gradient. Record model revision, source and bundle
  hashes, device/dtype, shapes, preflight memory budget, observed post-backward
  device allocation, and the exact loss tested. A
  tiny randomly initialized model test checks wiring, not real-weight
  feasibility or image quality.
- **Pinned numerical semantics:** the 4-step FlowMatch fixture supplies nested
  nodes for the two-substep teacher and one coarse student step. The official
  BF16 pipeline casts the raw scheduler timestep to the latent dtype *before*
  dividing by 1000; reversing that order changes the second teacher time
  embedding (`0.7421875` versus `0.74609375` at raw `744.611389...`). The
  probe and its regression test preserve the pipeline order. Endpoint loss is
  computed in FP32 for the gradient check, not presented as byte-for-byte
  pipeline output. LoRA initialization uses a fixed, reported seed `20260925`
  while restoring the caller's RNG state; the seed does not make MPS execution
  globally deterministic.
- **Current observation (2026-09-25):** eleven focused wiring/guard tests
  passed, including an actual Diffusers adapter-injection path with a tiny
  model. The first MPS attempt stopped at the host-memory preflight (65%
  pressure-free memory). After host memory was freed, a full attempt completed
  the frozen teacher forwards but stopped before LoRA injection: the shared
  Python environment exposed PEFT 0.15.1, below the pinned Diffusers adapter
  API's 0.17.0 minimum. An isolated, SHA-256-verified PEFT 0.21.0 wheel overlay
  then allowed the same pinned real-weight probe to complete on MPS with BF16
  base weights and FP32 LoRA weights. With fixed adapter seed `20260925`, two
  consecutive full MPS probes returned the same endpoint MSE
  (`0.02703850343823433`) and finite, nonzero LoRA gradient norm
  (`0.0012787174136338632`). Both selected LoRA parameters had finite
  gradient tensors, and no frozen base parameter received one. Post-backward
  MPS allocation was 14.25 GB (current) and 15.11 GB (driver); these are
  observations after the probe, **not measured peak memory**.
  No optimizer step or weight write occurred. This establishes one real-weight
  gradient-plumbing/stability point, not multi-step training stability or image
  quality. Reproduction requires compatible PEFT on `PYTHONPATH` (or in the
  environment) and the guarded command `python
  scripts/qwen_image21_lora_grad_probe.py --model-dir
  <pinned-transformer-dir> --conditioning <pinned-conditioning-manifest>
  --device mps --dtype bfloat16 --mps-memory-fraction 0.70`; the input paths
  may move, but their pinned contents must not change.
- **Native cogni-ml training boundary:** the generic F32 autograd/Adam stack is
  not yet wired to the Qwen-Image quantized/Metal forward path. Its generic
  matmul backward copies to CPU, the Qwen-Image kernels are forward-only, and
  the current transformer API does not expose an intermediate activation and
  modulation/layout bundle for a trainable suffix. Because the adapter is in
  block 31, blocks 0–30 can mathematically remain detached; the smallest
  hybrid candidate is a native frozen prefix plus a PyTorch-autograd final
  block/output head. This requires an explicit activation boundary, matching
  weight/numerical semantics, and gradient/update parity before replacing the
  working full-PyTorch probe. Porting the last-block VJP to Metal is a later,
  separately measured step, not a prerequisite for the first distillation
  control.
- **Next admitted comparison, only after that gate:** train a distillation-only
  control against a frozen teacher's integrated endpoint over a *coarse sigma
  interval*. Compare the student update with the teacher endpoint on the exact
  shifted FlowMatch grid; an instantaneous velocity at a different sigma is
  not an endpoint target. Add horizontal adjacent-step or vertical
  adjacent-layer losses one at a time only after a stable baseline, each with
  independent gradient and held-out image checks.
- **Rejected promotion:** an untested 3-5-step quality claim, conversion of the
  training loss into a frozen-GGUF inference heuristic, or calling a lower
  training loss an image-quality improvement. Any learned adapter remains
  experimental and opt-in; the current Euler/AB2 inference paths are unchanged.
- **Falsifiers and rollback:** reject the route on nonfinite/zero LoRA gradients,
  base-gradient leakage, unsupported training operators, memory pressure beyond
  the host guard, or a worse held-out quality/latency tradeoff than equal-cost
  controls. Stop before an optimizer step if the gradient gate fails. Reverting
  the opt-in training script leaves inference and model files untouched.

### Distillation-only optimizer control (bounded real-weight slice)

- **Admitted surface:** on the same pinned 256x256 bundle and official BF16 DiT,
  calculate the frozen two-substep teacher endpoint once. Optimize only the
  rank-4 FP32 LoRA on the last block's attention Q projection so one coarse
  student update approaches that fixed endpoint. Bound the first control to
  1-8 optimizer steps with explicit learning rate and gradient clipping. Log
  pre-clipping gradient norm, post-step loss, adapter finiteness, and memory.
- **Stop rule:** no optimizer step when loss or a selected gradient is missing,
  zero in aggregate, or nonfinite; when a base parameter has a gradient; or
  when the host/GPU memory preflight fails. After a step, roll back the current
  adapter update if its weights or freshly evaluated endpoint loss are
  nonfinite. The adapter, not the base transformer, is the optimizer's
  parameter set; an independent trainer check requires the exact final-block
  LoRA A/B names and every other parameter frozen before constructing AdamW.
  Keep the existing inference path untouched.
- **Falsifiers:** tiny-model tests must show a fixed teacher target, parameter
  isolation, a loss-reducing control case, and a deliberately broken gradient
  that prevents the step. The real-weight run must show finite per-step state;
  improvement on this one pinned sample is a training-plumbing signal only.
- **Observed control (2026-09-25):** the new
  `scripts/qwen_image21_lora_distill_control.py` passed 10 focused tests (21
  together with the existing gradient-probe tests). On the pinned `red cube`
  256x256 bundle, two independent MPS BF16 runs with the same LoRA seed,
  AdamW learning rate `0.001`, zero weight decay, and two optimizer steps
  returned identical FP32 `no_grad` evaluation losses:
  `0.027024995535612106` before training and `0.026850158348679543` after
  step two (final minus initial `-0.00017483718693256378`, about `-0.65%`).
  Both updates had finite adapter weights and nonzero finite gradients; no
  frozen base parameter received a gradient. Post-step MPS allocation was
  about 14.25 GB current and 15.11 GB driver, not a measured peak. The
  gradient-enabled loss immediately before a step is **not** interchangeable
  with the `no_grad` evaluation loss at the same weights on this MPS run:
  after step one they differed by about `1.3e-5`. The shared `no_grad`
  evaluator supplies the comparable initial/final observation. The control
  wrote no adapter checkpoint, changed no native/GGUF inference code, and
  demonstrates only local endpoint-loss reduction on one prompt and one
  coarse sigma interval. The evidence decays if the pinned model, bundle,
  schedule, runtime kernels, or evaluation context changes. Reproduce the
  bounded run with a compatible PEFT overlay using
  `python scripts/qwen_image21_lora_distill_control.py --model-dir
  <pinned-transformer-dir> --conditioning <pinned-conditioning-manifest>
  --device mps --dtype bfloat16 --mps-memory-fraction 0.70 --steps 2
  --learning-rate 0.001 --grad-clip-norm 1.0`.
- **Rejected/guard-only:** no claim about 3-5-step image quality, other prompts,
  stability over a long run, native Metal backward, or trained-adapter parity
  with quantized GGUF inference. No adapter checkpoint or runtime integration
  is admitted in this first optimizer slice; those require a separate format,
  source-revision, and parity gate after training is stable. The local artifact
  inventory on 2026-09-25 had no distinct-prompt 256x256 conditioning bundle;
  the available 256x256 copies are all the red-cube prompt. Re-inventory or
  create a separately pinned bundle before any held-out-prompt claim. Existing
  512x512 prompts cannot be substituted directly: this pilot's pinned 4-step
  FlowMatch fixture is for image sequence length 256, not 1024.

### Held-out prompt and adapter-export frontier (bounded control)

- **Admitted held-out evaluation:** prepare a separate official CPU/BF16
  256x256 conditioning bundle for the previously used elven-castle prompt,
  keeping source revision `790c92633540aa0cb11d9abf19eb46d861714758`
  and CPU noise seed 7. Check its manifest and payload hash, prompt identity,
  256-token image geometry, and source revision before the run. The frozen
  two-substep teacher target is computed independently for that bundle. Report
  both training-prompt and held-out endpoint MSE before/after the same two
  optimizer steps, with each pair evaluated under the same `no_grad` context.
  The held-out target and activations must never enter backward or AdamW.
- **Stop/falsify:** reject the held-out claim if the bundle duplicates the
  training prompt or payload, source/geometry/schedule differ, the held-out
  pass changes adapter weights or gradients, or either evaluation is nonfinite.
  A rising held-out loss is a negative generalization signal for this one
  sample, not a reason to relabel it as an image-quality gain. One prompt and
  one coarse interval cannot establish a useful 3-5-step student.
- **Adapter-export gate:** a small opt-in checkpoint may be emitted only with
  exact LoRA tensor names, rank, scale, source/bundle/schedule hashes and
  integrity metadata. Reloaded reference-Diffusers inference must reproduce
  the trained endpoint on both prompts before claiming reference portability.
  Native quantized-GGUF use needs a separate tensor-orientation/scaling contract
  and paired numerical checks; export alone does not establish native parity.
  Rollback is to omit the opt-in checkpoint and continue using unchanged
  Euler/AB2 inference. No source model, GGUF, or existing conditioning bundle
  is modified by this slice.
- **Observed on 2026-09-26:** the separate 256x256 elven-castle bundle has
  manifest SHA256 `17a57ee3b07a64c0bf0f9a559ea1e721d75e4ebea95f3930abe21314fea1c01f`
  and payload SHA256 `921cdc9a14685877887733ab0f032cc8cfb2b258d421edfdb38e00e8497f4ea6`.
  With the same frozen BF16 teacher, two AdamW steps, and shared no-grad
  evaluator, the red-cube training endpoint MSE was `0.0270249955` before and
  `0.0268501583` after; the castle held-out endpoint MSE was `0.0439844504`
  before and `0.0439081527` after. The opt-in rank-4 FP32 checkpoint was
  emitted with manifest SHA256 `266572b3d0f21b70b83c57e1ee80a8b9d5cb3d30fa62ce722e8ed9f2c0bce7f3`
  and payload SHA256 `e124404bce8842d056242f216ff89a83db8fd4b63193affea19a6d12c6101f41`.
  These small endpoint-loss reductions are a one-sample control observation,
  not a decoded-image quality result or a 3-5-step inference claim.
- **Reference reload check:** `scripts/qwen_image21_lora_adapter_replay.py` checks
  the official snapshot and runtime, captures both frozen teacher endpoints
  before LoRA injection, and measures the seeded initial adapter and reloaded
  saved adapter on two sequential fresh BF16/MPS models. The 2026-09-26 run
  with an explicit checkpoint-manifest SHA256 pin reproduced all four endpoint
  MSE values above with observed numeric delta `0.0` for each. The declared
  comparison envelope is `1e-6` absolute or `1e-4` relative, whichever is
  larger; the combined held-out allowance was `8.7893e-6`, below the recorded
  held-out improvement `7.6298e-5`. This is same-snapshot reference replay,
  not a general bitwise guarantee, decoded-image validation, or parity with the
  quantized native GGUF path. The low-level adapter injection requires a base
  created by the verified snapshot loader; callers must not modify its weights
  between verification and injection.

### Decoded-image control for the two-step LoRA checkpoint

- **Method and scope (2026-09-26):**
  `scripts/qwen_image21_lora_visual_compare.py` ran the pinned BF16/MPS
  reference transformer on the saved red-cube and held-out elven-castle
  256x256 conditioning bundles (seed 7). For each prompt it compared (1) four
  frozen-base Euler evaluations, (2) three evaluations with the seeded zero-B
  rank-4 adapter on only the first coarse `sigma[0] -> sigma[2]` interval and
  the base model on the last two intervals, and (3) the same hybrid route with
  the saved two-optimizer-step adapter. The adapter was disabled for both tail
  evaluations. All arms shared the same bundle, initial noise, schedule,
  explicit FP32 Euler recurrence, and one pinned CPU/FP32 VAE decode path.
  This is a matched custom evaluator, not bitwise parity with the standard
  Diffusers pipeline or native GGUF inference. The seeded first-interval
  enabled/disabled velocity and endpoint differences were exactly zero for
  both prompts.
- **Observed distances from the four-step base:** final-latent RMSE changed
  from `0.320405` (seeded hybrid) to `0.319890` (trained hybrid) for red cube,
  and from `0.325135` to `0.324810` for castle. RGBA MAE on the raw 0-255
  channel scale changed from `18.1143` to `18.1667` for red cube (worse), and
  from `9.0860` to `9.0849` for castle (effectively unchanged). Trained versus
  seeded hybrid RGBA MAE was `0.2503` and `0.3080`, respectively. These are
  descriptive distances, not perceptual quality scores.
- **Visual verdict:** the four-step red-cube image has distinct cube forms,
  while both three-step variants are mostly a diffuse red blob. The four-step
  castle has a discernible castle-like central structure; both three-step
  variants are darker, repetitive vertical forms with little recognizable
  castle structure. The trained and seeded variants look nearly identical at
  this resolution. Thus the small teacher-endpoint MSE reduction did **not**
  yield a useful three-step preview in these examples. The relevant next
  research move is a materially stronger distillation objective, broader
  adapter capacity/placement, or a separate short-step model—not presenting
  this checkpoint as a quality-preserving speedup.
- **Evidence and decay:** six decoded PNGs plus a content-hashed manifest were
  written outside the repository to
  `/private/tmp/qwen21-lora-visual-compare-20260926`; manifest SHA256
  `b012d2ac3d34277fdf70908862526072f63a390b90e4d5f5a41ff18aee2e0bd5`.
  The manifest records every input/source/image hash and the exact route.
  The focused LoRA Python suite passed 64 tests. This one-seed, two-prompt
  observation does not establish a general quality ranking or measured
  latency gain; refresh it after model, conditioning, checkpoint, schedule,
  decoder, or evaluator changes.

### Where the 40-step latents diverge: selective Q8 gate/up probe (2026-09-27)

- **Question and intervention:** the earlier fixed-input block-30 replay identified
  fused image-MLP gate/up weights as the largest tested local projection-family
  difference. A scratch repacker replaced exactly the 32
  `transformer_blocks.<0..31>.img_mlp.gate_up.weight` payloads in the native Q4
  GGUF (Q5_K for this family) with Q8_0 payloads from a separate community
  donor. Root independently checked all 32 target raw payloads against the
  donor and all 233 non-target raw payloads against the base; tensor order and
  shapes stayed fixed. A later re-read of the pinned GGUF files corrected the
  artifact attribution: base and hybrid metadata are byte-identical, both with
  `general.file_type=15`. The `15 -> 7` exception belongs to the donor
  compatibility check; the donor itself has type 7. The hybrid is therefore a
  tensor-payload-only intervention relative to the base (its tensor directory
  and offsets necessarily change). The earlier scratch policy/report text that
  describes `15 -> 7` as a hybrid metadata change is incorrect.
  Base Q4 SHA256: `51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`;
  Q8 donor SHA256: `c3ef62b2b7b53bf92418cbd77fbc24b43a26c8305a1f001c4a9a9a99b1373c03`;
  hybrid SHA256: `d9f6449ac9d75fa8cd1fdabfe89ec660fa5290b82640388a32cd05cd8a4914ff`.
  The correction used the pinned `repack.py` GGUF parser on all three files:
  the raw base/hybrid metadata blocks compared equal and their file types
  were 15/15, versus donor type 7. A streaming hybrid/donor comparison found
  byte equality for all 201 same-type tensor payloads (5,405,458,432 bytes);
  only 64 tensor types differ, exactly the 32 `attn.to_v.weight` and 32
  `img_mlp.out.weight` matrices. The current Qwen-Image loader records
  `general.file_type` but the DiT forward source does not consume that getter.
  This is an artifact comparison, not proof that the two changed projection
  families are individually responsible for any image defect.
  A subsequent full streaming check encoded all 64 official BF16
  `gate_layer`/`proj` source tensors into Q8_0 and matched every corresponding
  donor gate/up half byte-for-byte: 3,422,552,064 Q8 bytes, zero mismatches.
  Both source safetensors shard SHA256 values matched their cached 64-hex ETags
  at revision `790c92633540aa0cb11d9abf19eb46d861714758`; the donor's
  full-file SHA256 matched the pin above. A one-bit in-memory mutation was
  detected. This establishes the targeted donor payloads as an exact local
  Q8_0 encoding of the pinned official BF16 tensors, not the provenance of
  the donor's non-target payloads or of the Q4 base. No GGUF or generated
  image is committed to the repository.
  A separate bounded Q4-base falsifier compared 32 selected rows from each
  of the 64 official BF16 gate/proj tensors after the installed ggml
  no-imatrix `quantize_row_q5_K_ref` recipe against the fused Q5_K payloads
  in the Q4 base. Of 5,767,168 sampled bytes, 400 differed and only 26/64
  slices were byte-exact. The first mismatch was a one-byte block-scale
  difference in layer-0 `proj`. This rejects *that exact quantizer recipe*
  for the base; it does not establish a different checkpoint because the
  original converter version, importance matrix, and policy are unpinned.
  The full Q5 scan was deliberately not run after the sample falsifier.
- **Matched 40-step controls:** official BF16/MPS, native Q4, and native
  Q4-plus-Q8-gate/up used the same official Qwen3-VL
  conditioning payload SHA256
  `007ad14a01440a9f786ae874e78cb4aef3a3729d2768fec1a503363bf66114ab`,
  prompt, seed 7, 512x512 image, 1024x64 latent layout, 40-step Euler
  schedule, and BF16-effective timestep/state semantics. Independent read of
  all 120 finite snapshots confirmed 65,536 BF16-exact stored values per
  snapshot and the final bundle matching `step-039.bin` in each run. Global
  post-Euler latent relative L2 to the official state (Float64 recomputation,
  denominator = official L2):

  | Completed step | Native Q4 | Hybrid Q8 gate/up |
  | ---: | ---: | ---: |
  | 1 | 0.156901% | 0.129089% |
  | 10 | 2.357001% | 0.666186% |
  | 20 | 9.177365% | 3.744071% |
  | 30 | 17.335191% | 8.525851% |
  | 40 | 21.277440% | 11.341314% |

  Hybrid Q8 was closer in all 40/40 matched steps, but both error curves
  increased at every step. This is accumulated **whole-latent** distance, not
  a perceptual score or a unique bad-step detector. The hybrid still has an
  11.34% final residual. The earlier approximate face-token ROI did not show
  excess Q4 error relative to the rest of the image; it cannot localize eye
  defects within a decoded face.
- **Same-state DiT discriminator:** separate guarded native forwards fed both
  GGUF artifacts the *same official BF16 latent state*, conditioning, and
  effective time at model-call indices 0, 20, and 39. Native velocity
  relative L2 to the official teacher output was respectively 2.725202%,
  1.363841%, and 4.669017% for Q4; with Q8 gate/up it was 1.760615%,
  0.982294%, and 2.266090%. Direct Q8-minus-Q4 relative L2 at those
  identical inputs, normalized to Q4 output, was 2.001418%, 0.955071%, and
  4.251594%. The output differences are therefore present **inside DiT
  before Euler accumulation or VAE**. This isolates a local effect of the
  pinned GGUF variant at those states; it does not quantify the separate
  feedback/transport contribution along the other 37 steps. The targeted
  Q8 donor payloads are now source-matched, but this alone does not prove
  that the Q4 base gate/up payloads came from the same official revision or
  isolate quantization from every native-versus-official arithmetic difference.
- **Decoded-image check:** all three final latents were independently decoded
  through the same pinned CPU/FP32 VAE. On raw RGB 0-255 pixels, official-vs-Q4
  MAE was 9.78498 globally and 15.98519 in the fixed face crop
  `x=320..379, y=150..218`; official-vs-hybrid-Q8 MAE was 5.68384 globally
  and 8.87697 in that crop. The hybrid is visually closer for this one prompt
  and seed, including the face, but pixel MAE does not prove perceptual
  superiority or text/eye fidelity in general. The first image-comparator
  manifest accidentally pointed at an F32-state Q4 run; re-decoding the
  matched BF16-state Q4 manifest reproduced the prior PNG byte-for-byte, and
  the corrected comparator now fails closed on all three control identities.
- **Performance and decision:** the hybrid file is 7,167,086,816 bytes versus
  5,959,127,264 bytes for Q4. The single native runs logged approximately
  1,235 s hybrid versus 647 s Q4 denoising; this is a warning, not a
  quiet-host throughput estimate. Do not promote Q8 gate/up as the default
  yet. These controls reject a **VAE-only** explanation for the observed
  drift but do not establish a VAE off-manifold failure: the VAE may simply
  decode different, imperfect latents. Stronger next discriminators are a
  Q4-base target-payload provenance check and an exact-revision,
  multi-prompt/seed precision sweep with matched same-state forwards and
  decoded-face/text evaluation, followed by a throughput gate.
- **Evidence and decay:** trajectory files are under
  `/private/tmp/qwen21-q8-gateup-40-20260927`, official and Q4 BF16-state
  controls under `/private/tmp/qwen21-official-trajectory-20260927` and
  `/private/tmp/qwen21-bf16-state-20260927.b6UNJF`, and teacher-forced raw
  outputs/report under `/private/tmp/qwen21-q8-teacher-forced-20260927`.
  Teacher-forced report SHA256:
  `7cf6b34b359bb5d61a6358caf45e7796c7d643c2b987dbd4469c91bea1d08d81`.
  The complete donor gate/up byte comparison is under
  `/private/tmp/qwen21-exact-gateup-provenance-20260927` in
  `full_compare_report.json`; its `full_compare.py` SHA256 is
  `11480d0b751259331e22f37a0ffb4b3f5680c06e3d29c11cb2fffd07deabaee5`.
  The bounded Q5 falsifier and its sample report are in that same scratch
  directory as `q5_base_compare.py` (SHA256
  `b6ece86e4245e687cb40bfcd7663b33b98a5ec480cdbe426e320c59cb346c764`)
  and `q5_base_compare_report.json` (SHA256
  `21341623178880d12c31f83df9c23dd4e4d1db6c30a38067208cf5f9b98ffb6d`).
  Its exit 2 deliberately stops at the non-exact sample; it is not an
  infrastructure or model-inference failure.
  The scratch comparators and tests are under
  `/private/tmp/qwen21-q8-trajectory-compare-20260928` and
  `/private/tmp/qwen21-q8-image-compare-20260927`. The controls expire on a
  changed model/donor, conditioning, schedule, state arithmetic, Metal route,
  VAE, decoder, or comparator; ephemeral `/private/tmp` evidence must be
  regenerated if removed.

### Crossed-state DiT probe: local output error versus trajectory feedback (2026-09-27)

The Q8-gate/up trajectory already differs from the official trajectory at
the first DiT call. A new probe separated the *velocity* gap at calls 0, 20,
and 39 without reconstructing unavailable official velocities at other calls.
For Q8 DiT velocity `N`, official BF16 velocity `T`, official pre-Euler state
`x_ref`, and the hybrid's own pre-Euler state `x_q8`, the exact vector identity
is `N(x_q8)-T(x_ref) = [N(x_q8)-N(x_ref)] + [N(x_ref)-T(x_ref)]`. The first
bracket is the same-Q8-model state-transport term; the second is its
same-official-state local discrepancy. At call 0 both states are the same.
All values below are raw-velocity L2 norms across 65,536 elements, **not**
additive causal shares or image-quality scores:

| DiT call | Input-state gap | Q8 local velocity gap | Q8 state transport | Total velocity gap |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 7.334246 | 0 | 7.334246 |
| 20 | 7.151546 | 3.494177 | 38.585152 | 38.755341 |
| 39 | 30.866094 | 6.482357 | 51.478731 | 51.728717 |

At calls 20 and 39, the two velocity components are nearly orthogonal
(cosines 0.0035 and -0.0243), so the total norm is close to the larger
transport norm in this sample; no scalar percentage attribution is implied.
Relative to official teacher-velocity norm, the Q8 local discrepancy is
0.9823% and 2.2661%, versus 10.8472% and 17.9958% for the state-transport
term. The local discrepancy is not a pure quantization term: even its exact
subdivision into Q8-minus-Q4 at `x_ref` and Q4-minus-official at `x_ref`
does not separate Q4 weights from native/official arithmetic. The larger
late transport term shows that by those calls the already-shifted latent is
a major driver of the instantaneous DiT output difference. It does not
identify which earlier call originated that state shift.

Controls: the Qwen3-VL conditioning payload, seed, Euler schedule, effective
BF16 timestep, source revision, and hybrid GGUF are pinned to the preceding
section. Exact official raw BF16 velocities exist at calls 0, 20, and 39.
Six existing same-official-state native outputs were reused; only two new
Q8 forwards were run at its own pre-Euler states (`step-019` and `step-038`).
The first sandbox attempt could not create a Metal device and produced no
velocity files. One retry under `scripts/run_safe.sh` (600 s, 16 GiB RSS,
35% free-memory floor) completed. All six BF16 Euler poststate canaries
(official and Q8 at each selected call) reconstructed the saved states
byte-for-byte, and both forms of the vector decomposition had zero residual
at all selected calls. Root independently rechecked output SHA256 values,
raw-vector norms, prestate gaps, and the vector identity with Float64 NumPy.
Six CPU unit tests and the full SHA-gated preflight passed. The new Q8
velocity SHA256 values are
`8e780decf780404fc75625f2a431b88230297b66a792a6b40eda15db321dab6f`
(call 20) and
`2f2994055c4e2229e8a127892f7d011889e54361f38b7306754db8a312cafaf5`
(call 39); the analysis-report SHA256 is
`4975403f20b9283f3b3f249d458e5a5817db2f571926e38dc9ac732b3b5be88d`.
Scratch inputs, guards, outputs, and report are under
`/private/tmp/qwen21-crossed-state-20260927`.

The 40 saved post-Euler snapshots still show monotonic whole-latent error
growth rather than one bad step; the final Q8 gap is 11.341314% relative
L2. The approximate face-token ROI is less divergent (7.6959%) than the
whole latent, which neither rules out perceptually sensitive eye errors nor
establishes that the VAE is outside its training distribution. The strongest
next discriminator is a controlled earlier-call/layer precision sweep with
matched trajectories and eye/text-specific evaluation; VAE out-of-support
needs a separately calibrated reference distribution. These measurements
expire if the weights, conditioning, schedule, BF16 arithmetic, native
runtime, decoder, or scratch evidence change.

### Call-0 DiT hidden-state drift with Q8 gate/up (2026-09-27)

An observational 32-block capture tested where the targeted Q8 gate/up
replacement changes the *first* DiT forward. It reused the pinned official
BF16/MPS block outputs and the prior Q4 Metal capture at the same official
Qwen3-VL conditioning (`007ad14a...66114ab`), initial latent
(`d0315806...8358932c`), and official-MPS BF16 Fourier time features
(`661d6707...0cb85`, widened to Float32). The Q8 model was the previously
verified hybrid (`d9f6449a...d8a4914ff`); only its 32 image-MLP gate/up
tensor payloads differ from Q4. Its metadata, including `general.file_type`,
is byte-identical to Q4 as corrected above. This is an **injected-time call-0
diagnostic**,
not the ordinary native-time-feature Q8 path or a warmed prefix-cache hit.
The scratch Metal change only copied the joint hidden buffer immediately
before block 0 into a separate buffer on the same command buffer; all 32
post-block copies were read only after the complete forward. Its pre-block-0
joint state was reconstructed independently from the prior Q4 text and image
projection captures and matched the new Q8 capture byte-for-byte (SHA-256
`459974c11a29688749f9a2e3c6624c9042ef6e40cf53d6d912ff5aa31cfcf007`).
Against official pre-block projections, that common input already differs by
0.183084% relative L2 on text rows and 0.165677% on image rows. Therefore
gate/up precision cannot be the source of the *earliest* pre-block difference.

The table reports image-token hidden-state relative L2 to the official
post-block state, with each row normalized by that block's official norm.
Root independently recalculated all 32 rows from the raw Float32 captures,
including finiteness and the Q4 control values. Q8 was lower at all 32
post-block checkpoints, but the residual still grows through the network:

| Post-block | Q4 with official time features | Q8 gate/up with official time features |
| ---: | ---: | ---: |
| 0 | 0.440317% | 0.424135% |
| 8 | 1.645803% | 1.325617% |
| 13 | 4.589565% | 3.868095% |
| 30 | 13.120123% | 9.838294% |
| 31 | 3.740302% | 2.785633% |

This is deliberately an **image-token** metric: over the combined text-plus-
image stream, Q8 is slightly farther from official at blocks 0 and 5-9.
Therefore the table does not establish a universal hidden-state improvement.
The apparent percentage recovery at block 31 is again a denominator effect:
Q8's image-token absolute RMSE rises from 0.557961 after block 30 to
0.629670 after block 31. The largest Q8 adjacent error-vector change is
29->30 (RMSE 0.385908), but this cumulative map does not isolate that block's
own error from incoming state drift. The Q8 versus Q4 post-block-0 outputs
already differ by 0.064884% of the official block-0 norm on exactly the
same native pre-block input. At the final call-0 velocity, relative L2 to
the official raw BF16 teacher is 2.720172% for the Q4 injected-time control
and 1.764384% for the Q8 injected-time capture. The Q8 output SHA-256 is
`42fc2aad86fac13eab7fc6c3b4b53f7de67095294f8c61832b3564c603bab9c7`.
These local hidden-state and velocity improvements agree in direction with
the separate default-time Q8 40-step trajectory, but are not an eye/text
quality verdict or a causal share of its final latent distance. The capture
is observational in source, but this Q8 injected-time output did not have a
separate uncaptured, same-time-feature byte-parity forward; its scope is the
captured route, not an asserted exact parity certificate for an unobserved
route.

The saved default-time, BF16-state trajectory confirms that image-latent
distance starts after the first Euler update, rather than appearing suddenly
late: Q8 relative L2 exceeds 1%, 5%, and 10% after completed steps 13, 23,
and 34, versus steps 7, 15, and 21 for Q4. Late velocity discrepancies are
already dominated by state transport in the sampled crossed-state probes
above. These observations locate an early, distributed DiT/model-path
discrepancy and its accumulation, **not** a unique bad layer or VAE
out-of-support event. Qwen3-VL conditioning has a separate measured effect;
its Float32-state trajectory percentages cannot be added to the BF16-state
Q8-versus-official percentages. The next low-cost independent discriminator
is a shared-x0 conditioner-by-DiT first-call contrast, then fixed-input Q8
block replays or a warmed-cache call-1 trace if the residual must be localized
further.

The Q8 scratch runner, capture, and report are under
`/private/tmp/qwen21-earliest-dit-probe-20260927/`. The runner source and
binary SHA-256 values are `b276d6718bdb9ff32a20734eefd43351120f95674aa4c54df644a855b4d0c9c9`
and `5c414d5a6f2db0d5ee640331ed1e146cf977d3fbc77bc824a30c839c77e25744`;
the report SHA-256 is
`ac5b08f3ff00eeaccbeeaa0c734e03bd8f3d01b509f2b277eb11a5d5fd5fad6c`.
The independent all-32 comparator under the same scratch root has source
SHA-256 `c196bb74bc1a1251c0d366c4aae6bc9e26b155b63ea94432568cebb4e2876604`
and reproducible report SHA-256
`bda3dd813124e9e7a7c10a044204aa9f9e874e01c56e65ad7cbb598d99ed3a43`;
root reran it successfully and separately recomputed its selected metrics.
Model, conditioning, x0, time-feature, tensor-shape, finiteness, and source
hash gates passed before/after the single Metal forward. The sandboxed
first attempt created no Metal device and no model output; one device-enabled
retry completed. The report's embedded `launch_command` and `build_command`
strings were inherited from the earlier Q4 harness and incorrectly name its
binary/source and Q4 expected output; they are **stale provenance text**, not
the executed command or a valid Q8 output gate. The actual device-enabled
invocation, with input validation performed by the runner, was:

```sh
QWEN_IMAGE21_INJECT_OFFICIAL_TIME_FEATURES_MPS=1 \
  /private/tmp/qwen21-earliest-dit-probe-20260927/native_q8_early_call0 \
  --index 0 \
  --latent /private/tmp/qwen21-official-trajectory-20260927/official40-cache-default/initial_latents_f32le.bin \
  --output /private/tmp/qwen21-earliest-dit-probe-20260927/q8-official-mps-timefeatures-all-blocks/native_velocity_step-000_f32.bin \
  --report /private/tmp/qwen21-earliest-dit-probe-20260927/q8-official-mps-timefeatures-all-blocks/native_report_step-000.json \
  --expect-latent-sha256 d03158064c86fd691927cf93258b00fac2fc8d09e14a578d0f28b4fe8358932c
```

Refresh all claims after
checkpoint/GGUF, conditioning, schedule, BF16 arithmetic, official runtime,
Metal source/compiler, or scratch-artifact changes.

### Call-0 conditioning x DiT precision and pre-block intervention (2026-09-28)

Two matched first-call probes tested whether the observed image-latent drift
is mainly entering through Qwen3-VL conditioning, the DiT input projection,
or later DiT computation. They used the same Russian-sign/portrait prompt,
seed 7, BF16-exact initial latent (`d0315806...8932c`), 40-step Euler
schedule, pinned official BF16/MPS teacher, and *injected official MPS BF16
Fourier time features* (`661d6707...94dcb8fa90cb85`, widened to Float32
`310f9984...72e047`). The Q4-labeled mixed-quant GGUF is
`51998ad7...8ceb15b8a`; the selective-Q8-gate/up GGUF is
`d9f6449a...8a4914ff`. The two Qwen3-VL
conditioning payloads differ only in their 230x4096 embedding prefix;
the mask and latent suffix are byte-identical (SHA-256
`5ef9e503...1029997`). This is a captured, matched **call-0** comparison,
not a full 40-step cross-product or a default-time native route.

| Native DiT variant | Qwen3-VL embeddings | Call-0 velocity relative L2 to official BF16 teacher | First BF16 Euler latent relative L2 |
| --- | --- | ---: | ---: |
| Q4 | Official | 2.720172% | 0.156638% |
| Q4 | Native | 2.624811% | 0.153856% |
| Selective Q8 gate/up | Official | 1.764384% | 0.129104% |
| Selective Q8 gate/up | Native | 1.727515% | 0.124881% |

Independent reanalysis re-read all four 65,536-element Float32 velocity files,
checked their SHA-256 and finiteness, and recomputed the table in Float64.
The official teacher's first Euler state was reproduced exactly at
65,536/65,536 elements using `BF16_RNE(x0 + BF16_RNE(dt * velocity))`, with
`dt=-0.015080928802490234`; a plain Float32 product then BF16 cast was
**not** byte-equivalent. Within Q4 and Q8, switching only the saved
conditioning embeddings moves the velocity by 0.812495% and 0.900159% of
the teacher norm, respectively. The difference of those two *vectors* is
0.573518% of that norm (effect cosine 0.780388), so conditioning and the
GGUF variant interact at this point. The slightly lower teacher error with
native embeddings is compatible with error cancellation; it does not show
that native Qwen3-VL is semantically better or will improve a 40-step image.
The larger controlled first-call reduction here is from the targeted Q8
gate/up variant, without assigning an additive causal percentage to it.

The next probe copied a saved tensor directly over the **joint DiT hidden
state immediately before block 0**, after native image/text input assembly,
on the same Metal command encoder. It held Q8 weights, official Qwen3-VL
conditioning, x0, and the official MPS time-feature input fixed; the time
modulation chain was not replaced. A no-op injection of the native pre-block
tensor reproduced the baseline velocity byte-for-byte (SHA-256
`42fc2aad...bab9c7`) and captured the expected native input SHA-256
`459974c1...31cfcf007`. The official BF16 pre-block tensor
(`96582cb4...e751207e`) was independently widened byte-for-byte to the
injected Float32 tensor (`45ca4101...dccfefff`), and that exact digest was
captured before block 0. Against official, the original native pre-block
state differed by 0.182926% relative L2 overall: 0.183084% on its first
230 text rows and 0.165677% on the remaining 1,024 image rows.

Replacing this entire pre-block state moved the final call-0 target velocity
by 0.147016% of the teacher norm. Its teacher-relative error fell only from
1.764384% to 1.739396% (1.42% of the prior *error magnitude*), and the
first post-Euler latent error fell from 0.129104% to 0.128126%. The first
block's combined-stream error remained nonzero after injection
(0.395589% -> 0.364730%). Thus the pre-block mismatch has a causal effect,
but **most of this Q8 call-0 output discrepancy survives exact official
pre-block input**. The residual is downstream of that boundary *in this
hybrid intervention*; it may involve native modulation, block arithmetic,
other GGUF quantized weights, and the output head, not one proven defective
block. The earlier full 40-step Q8 latent gap and face defects are not
explained by these single-call percentages alone. No new VAE
out-of-support claim follows: the decoder receives different latents, but
these probes do not calibrate the VAE's training support or perceptual quality.

The 2x2 report and analyzer are under
`/private/tmp/qwen21-conditioning-dit-2x2-20260927/`. Both independent
reruns of `analyze_2x2.py` passed all raw tensor/hash gates and agreed on
the displayed metrics. Their full JSON SHA-256 values differ between
Python 3.12.2/NumPy 1.26.4 (`a24440d244675b9a6494f15a41902c35b8ebe22bae48dd497a29c10ec0dbc232`)
and Python 3.14.6/NumPy 2.5.2 (`da4b15e15520d2c068e80563136ec1e21cd7ff928e2adc0198b3db94b4eeef08`)
because their Float64 reduction results differ in the last decimal places;
neither JSON digest is a cross-runtime canonical reproduction gate. Use
the separately checked raw input/output SHA-256 values instead.
The native-Q4 cell is a **salvaged post-forward output**: the full model
forward and captures completed, but runner JSON assembly failed while
hashing an absent scratch source path. Its raw velocity SHA-256 is
`22985700...eb423e`; the native-Q8 cell completed with report and raw
velocity SHA-256 `73644a92...093e4778`. The reused official-conditioning
reports carry stale build/launch metadata as noted above; this analysis
uses their checked raw vectors, input hashes, and actual new-run controls,
not the stale command strings. Neither newly captured arm has an uncaptured
byte-parity forward. The pre-block probe's scratch runner, reports, and raw
captures are under `/private/tmp/qwen21-preblock-attribution-20260927/`;
no-op and official-injection report SHA-256 values are
`bfc0d430...6b516413` and `413cf072...77a724e8`. Their inherited
`run_history_note` incorrectly describes a time-only intervention; the
per-arm override mode, input/capture hashes, and output digests establish
the actual scope. All scratch evidence is ephemeral. Re-run after a change
to checkpoint/GGUF, embeddings, schedule/BF16 arithmetic, time-feature
source, Metal kernels/compiler, or reference runtime. The next discriminator
is an equal-input, block-0 Q8-versus-official boundary replay (attention,
MLP, modulation, output head separated), followed by multiple prompts/seeds
and visual eye/glyph checks before promoting a precision policy.

### Call-0 block-0 modulation boundary (2026-09-28)

A scratch-only selective-Q8 runner tested the next available boundary with
the same official conditioning, BF16-exact x0, and injected official MPS
time features as above. Its new binary first ran a full 32-block **no-op**
and reproduced the prior native velocity SHA-256 byte-for-byte
(`42fc2aad...bab9c7`). The intervention then injected the exact official
BF16-widened joint pre-block-0 state (`45ca4101...dccfefff`), post-modulation
attention input (`24264a20...05cddec2`), and `tanh(gate1)`
(`b4db133d...44cb3d1e`) before the native Q/K/V projections; all inputs
passed their raw-BF16/widened-F32 representation and SHA gates. The native
post-modulation values were captured **before** replacement, and differed
from official by 0.353862% (attention input) and 0.269771% (gate), relative
L2. Only block 0 received those two modulation overrides; subsequent
blocks and the head ran on the native Q8/Metal path.

| Q8 call-0 intervention | Post-block-0 joint state relative L2 | Final velocity relative L2 |
| --- | ---: | ---: |
| Same-binary native-preblock no-op | 0.395586% | 1.764384% |
| Official pre-block only, earlier scratch binary | 0.364727% | 1.739396% |
| Official pre-block plus block-0 modulated input/gate | 0.331287% | 1.793929% |

Independent reanalysis recomputed these errors from the pinned raw tensors
and reran the 18-hash-gate analyzer. The combined intervention improved the
immediate block-0 boundary by 16.25% of the same-binary no-op error, yet
**increased** final teacher-relative velocity error from 1.764384% to
1.793929%. Versus the prior official-preblock-only run, the block-0 error
fell by 9.17% while final error rose by 3.14%; this latter contrast is
cross-scratch (same repository source, different instrumented binaries),
not a same-binary isolated modulation effect. The new final velocity moved
by 0.395532% of teacher norm relative to the same-binary no-op, or
0.346121% relative to the earlier pre-block-only run. Local boundary
accuracy and final output accuracy are non-monotone here; error cancellation
or later amplification is plausible. This rejects promoting a block-0
modulation-only correction from this sample, but does not assign the
remaining error uniquely to attention, MLP, quantization, later blocks, or
the output head. Official internal attention/MLP/head outputs were not
captured, and the original Diffusers source path recorded in the official
capture manifest has since disappeared.

The scratch runner, both reports, captured tensors, and CPU analyzer are
under `/private/tmp/qwen21-block0-boundary-replay-20260928/`. Its runner,
binary, and analyzer SHA-256 values are respectively
`adb363d4...75041cc3`, `41f44c2f...2d46156e`, and
`95c6e37e...fd6fdee0f`; the intervention velocity and post-block-0 state
SHA-256 values are `97f2f2a8...8c80917c` and
`d0a00a18...d6b288f36`. The wrapper's first sandboxed attempt failed
closed because process inspection was denied; a scoped retry passed both
input-only preflights and the two bounded forwards. No timing claim follows.
Unlisted old post-block capture files in the scratch directory are not part
of this run's five-tensor report inventory and must not be used. Refresh
after a change to any pinned source/model/input, Metal compiler/kernel,
official capture, or scratch artifact. The next DiT discriminator needs
official attention/MLP intermediate captures or equal-input sublayer
replays, then multiple prompts/seeds and decoded eye/glyph evaluation.

At the preceding checkpoint, a planned CPU-only VAE interpolation along the
*actual* official-to-Q4/Q8 final-latent paths could not run because the old
Python environment had lost importable `diffusers`. The exact Diffusers
commit `8b3c707ebd3ec4881f4190cf42931da07eaf3b65` was recovered into a
scratch source checkout; its Qwen-Image 2.1 VAE source SHA-256 is
`afb341db5e9d081e568ae4703119d1141d124aaf2fa9c061b3d4eb80ce73c15b`.
A separate scratch checkout of Hugging Face Hub 1.32.0 supplied the API absent
from the host's Hub 0.36.0. Neither checkout changed repository dependencies
or existing Python environments, and no model weights were downloaded.

The CPU/FP32 decoder then reproduced all three independently saved official,
Q4, and Q8 endpoint PNGs byte-for-byte (RGBA pixels and PNG SHA-256). Only
after that gate did it decode nine points at `alpha = 0, 0.125, ..., 1` along
each normalized-F32 latent segment. The fixed face ROI is
`x=[320,380), y=[150,219)` in the 512x512 image. RGB RMSE versus the official
decode at `alpha = 0.25/0.5/0.75/1.0` was, respectively, `6.19/12.00/17.59/23.05`
for Q4 and `3.77/7.33/10.85/14.23` for Q8 in that ROI; full-frame RMSE was
`5.57/10.79/15.78/20.57` and `3.25/6.38/9.65/13.12`. Adjacent 0.125-step
face-ROI RMSE remained within `3.15-3.52` (Q4) and `1.92-2.07` (Q8), with no
isolated spike on this sampled path. Visual face crops changed progressively
rather than showing an abrupt broad collapse. This weakens a *sharp decoder
cliff on these two straight-line paths* as the explanation for this one
scene's eye/face differences; it does not establish VAE training-support
membership, rule out a narrower excursion between samples, or settle other
prompts, seeds, and text regions. The oracle-gated report, probe, endpoint
hashes, and full/face montages are under
`/private/tmp/qwen21-vae-interpolation-20260928/`; `report.json` is the
machine-readable source, and `interpolation_probe.py` SHA-256 is
`5952ef5e787742906e4822693ebb6a9038a77ced9abf363d08db197e6c105179`.
Refresh after a change to latent payloads, decoder source/weights,
normalization, ROI, or scratch artifact availability.

An equal-input **call-0 block-0 sublayer** comparison now narrows the first
measured DiT difference. The official BF16/MPS hooked forward used the same
model revision and pinned Diffusers commit above; its final velocity SHA-256
`f35c27adee76539f7dbe2771213c3122402618d553fadf915cf31e99a7300401`
was byte-identical to both the no-hooks canary and the earlier official
capture. All six earlier official boundary hashes re-matched; 37 saved
captures passed file-integrity and exact BF16-to-F32-widening checks, with
35 explicitly shape-checked. The official manifest is
`/private/tmp/qwen21-official-sublayers-20260928/hooked/block0_sublayer_capture_manifest.json`
(instrumentation script SHA-256
`6e2f6e380341434ae7ec5728fe9e19db47e21519fdb2a2ab37e3a4f8cfbad75d`).
The native Metal control and nine-sublayer-tap runs used the *same* official
pre-block-0 hidden state, official block-0 modulated attention input, and
official `tanh(gate1)` as interventions. They produced byte-identical final
velocity outputs with SHA-256
`97f2f2a8bc4a27809490d16c2e6503029e6dcafb465f4a19c3cf5c2f8c80917c`.
Their corrected reports are respectively under
`/private/tmp/qwen21-native-sublayers-20260928/control-accounted-block0-only-preblock_and_block0_modgate_override_official/`
and
`/private/tmp/qwen21-native-sublayers-20260928/sublayers-accounted-block0-only-preblock_and_block0_modgate_override_official/`.
Actual Metal snapshot allocation is 4 buffers/82,182,144 bytes in control
and 13 buffers/410,910,720 bytes with taps; the earlier scratch reports'
2/11-buffer counts were stale. These are correctness runs, not timing evidence.

Comparing the official tensors *as exact BF16 widenings* with native F32
tensors gives relative L2 error across all 1,254 rows of 0% at the injected
pre-block input, 0.280020% at attention `to_out` before gate1, 0.225712% after
the attention residual, 2.736642% at the norm2/modulated MLP input, and
0.331287% at the block-0 output. The official `to_out` pre/post-dropout
hashes are identical (dropout 0). The 230 non-target/text rows and 1,024
target rows were derived from the saved target/key-valid masks. On the
*native shadow computation before override*, modulated attention input differs
by 0.353862% overall, with one text row at 6.28%; it cannot cause the
measured attention `to_out` difference in this injected run. The first
matched exercised difference is therefore at or before attention `to_out`,
not a demonstrated single faulty kernel. Q/K/V, BF16-vs-F32 arithmetic,
weight quantization, and downstream cancellation remain confounded. A large
relative difference in normalized MLP input is not by itself amplification
of final error; block-0 output error is smaller.

A third, scratch-only native pass added raw Q/K/V taps immediately after the
block-0 projections and before the in-place fused QK RMSNorm+RoPE kernel, plus
post-fused Q/K taps. Its final output was byte-identical to the prior native
control and sublayer-tap outputs; the SHA-256 remains
`97f2f2a8bc4a27809490d16c2e6503029e6dcafb465f4a19c3cf5c2f8c80917c`.
The report is
`/private/tmp/qwen21-native-sublayers-20260928/sublayers-accounted-block0-only-preblock_and_block0_modgate_override_official_qkv_stage_taps/native_report_step-000.json`
(report SHA-256
`9eb1136b0d9efe4973efc833b4c2a3f8c50ed4c0e3cee6cc0041c2c2e1850f88`;
instrumented Metal source SHA-256
`e89bdd402961abc75888f7c710f53dd4100196cb665a96bc1ae04e57c84a1e63`).
The five new snapshots raise the active Metal capture allocation to 18
buffers/513,638,400 bytes (1,027,276,800 bytes of copy plus readback).
Against the official raw BF16 projections widened to F32, the native raw
Q/K/V F32 projections differ by **0.243992% / 0.237986% / 0.985289%**
relative L2, respectively. Their resident weight types are Q8_0, Q8_0, and
Q6_K. Thus drift is already measurable at the projection boundary, before
RMSNorm, RoPE, attention weighting, later DiT blocks, or VAE decode. V is
the largest of these three discrepancies; a weight-precision contribution is
plausible but not isolated from BF16-vs-F32 arithmetic or unproven Q4-base
provenance. This does not yet show that improving V will improve the final
image.

The official capture has Q/K after RMSNorm but not after RoPE. Applying the
pinned Diffusers RoPE function on CPU to those hash-gated BF16 Q/K tensors
and saved complex64 frequencies yielded a *CPU-derived*, not MPS-observed,
teacher comparison: native post-fused Q/K relative L2 is 0.39689%/0.40185%.
It combines already different projections with norm/RoPE and arithmetic
differences, so it cannot identify a RoPE defect. The source- and input-gated
CPU report is
`/private/tmp/qwen21-official-sublayers-20260928/cpu-postrope/cpu_postrope_report.json`
(SHA-256
`db5b88c2650bc4ef8a922589efd16445510b0313cfc9280e9fe9e523e957f73d`).
Static GGUF inspection found `transformer_blocks.0.attn.to_v.weight` in the
existing Q8 donor at type Q8_0, 17,825,792 payload bytes, SHA-256
`4ed437ce77c64ff1c1b4692c6195e7e57b7070ce6d356f8df4b0460287c98dd0`.
The current hybrid retains the Q4-base Q6_K V payload, 13,762,560 bytes,
SHA-256
`a5c9192eb6c5348bd8d369b5f256c94c116ab508fe60168b2b0baf2b83dc05fb`.
The Q8 payload is 4,063,232 bytes larger and would overlap the next tensor
if overwritten in place. A read-only source check instead reopened the pinned
official BF16 safetensors shard (index SHA-256
`17987f6623b1c814d0ef55a137d99142b7b3b040eb1bf241b5575dd35af803a2`,
shard-1 SHA-256
`9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b`)
and Q8_0-quantized its `transformer_blocks.0.attn.to_v.weight` F32
widening. The resulting 17,825,792 bytes matched the donor V payload
byte-for-byte. This establishes that *donor V* source identity for the pinned
quantizer; it does not establish the Q6_K base V provenance or recipe.

A scratch-only, same-binary call-0 intervention then replaced only block-0
`to_v` in the loaded hybrid weight object, keeping all other block-0 weight
references and the 31 later layer objects unchanged. The Q8 donor was read
without mapping its full 7.69-GB model; no GGUF or repository code was
modified. OFF and ON used the same official pre-block state, official
block-0 modulated attention input and `tanh(gate1)`, official MPS temporal
features, BF16-exact initial latent, and 40-step scheduler's first model
call. The executable SHA-256 was
`bd073f8625a932985cf18d4ec5544e1cd6106d1ea142456ebbf246b7c55b3ce6`.
OFF reproduced the prior native final velocity byte-for-byte
(`97f2f2a8bc4a27809490d16c2e6503029e6dcafb465f4a19c3cf5c2f8c80917c`);
ON produced
`72eb70d3afd93f1f23924330660cab9b3b04a61972f749e6c77e437e3d603dda`.
The two hash-gated reports are under
`/private/tmp/qwen21-native-sublayers-20260928/` in the
`sublayers-accounted-block0-only-preblock_and_block0_modgate_override_official_qkv_stage_taps_block0_v_q8_off/`
and corresponding `_on/` directories (report SHA-256 values
`f5868ad7220c9c842b6102d9f14140e9023c6243104bda2fc3ee0e99ea8fb5d3`
and `1cc2b39fb7cdb8b70aa8f0c83890ec8e2f53f1e7d5fc39b7b56498c673b44c6f`).
The scoped comparison report is
`/private/tmp/qwen21-native-sublayers-20260928/v_q8_override_comparison_report.md`
(SHA-256
`6aa837d761ca3a77f4aebed42d63c03431275f7f4733d8adbfbca0b32ed97efe`).
Its official-source-match annotation is based on a separate read-only
reconstruction; the Crystal runner did not re-quantize official BF16 weights
during inference. Raw and post-RoPE Q/K, pre-block state, and captured
modulation inputs were byte-identical across arms.

Against the official BF16 tensors widened exactly to F32, the raw V
projection error fell from **0.985289% to 0.337856%** relative L2, and
attention `to_out` error fell from **0.280020% to 0.234882%**. Yet the
block-0 output error rose from **0.331287% to 0.337489%** overall: its
230 text rows improved 0.320078% to 0.316898%, while the 1,024 target
rows worsened 0.351888% to 0.373995%. The final target-velocity error
fell only from **1.793929% to 1.759205%**. The native
`step_bfloat16` scheduler first rounds model output to BF16, then rounds
the product and sum. With
`BF16_RNE(x0 + BF16_RNE(dt * BF16_RNE(velocity)))` and
`dt = -0.015080928802490234`, the independently recomputed official
first Euler state matched its saved 65,536/65,536 BF16 elements, while
the native first post-Euler latent error was **0.128879% OFF versus
0.128073% ON**. Omitting the initial native-velocity rounding yields the
distinct 0.128842%/0.127765% diagnostic, not the source-exact scheduler
result. This is a modest first-step
improvement with non-monotonic intermediate effects, not evidence of better
eyes, lettering, or a full 40-step image. ON also selects the specialized
Q8_0 batch kernel where OFF's Q6_K V uses generic quantized dispatch, so
the intervention is *V payload/type plus its dispatch*, not a pure
quantization-precision effect. The downstream norm2/MLP boundary test follows;
validate any promising policy on matched full trajectories,
multiple seeds, and decoded eye/glyph regions. Refresh after official or
native payload, input, kernel, capture, source, or scratch-artifact changes.

A further scratch-only **block-0 MLP boundary 2x2** used the base Q6_K V arm
and the same official pre-block state, modulated attention input,
`tanh(gate1)`, and MPS time features. At the exact native boundary after the
post-attention residual and norm2/modulation kernels, it copied the official
BF16-widened modulated MLP input, the official BF16-widened `tanh(gate2)`,
neither, or both into the buffers subsequently consumed by gate/up and the
MLP residual. The four arms ran sequentially with the same executable
SHA-256 `bddcac9ba1fccfe49aadc49f97f8949cb91f437c60a2222e2a4185c3be29c075`;
the OFF arm reproduced the prior velocity SHA
`97f2f2a8bc4a27809490d16c2e6503029e6dcafb465f4a19c3cf5c2f8c80917c`.
All four captured post-attention residuals are byte-identical (SHA-256
`53b1fe9c7dd786250717a6151e8dc15d165fed0c8c4a0d30e07249a22b174a27`),
as are the captured attention projections. The injected MLP input and gate2
captures exactly match their independently captured official F32-widening
hashes when selected. This is a controlled *native clamp effect*, not a
comparison of independent native and official MLP kernels at identical
upstream states: the residual state still contains native attention error.

Against official BF16 tensors widened to F32, the control's modulated MLP
input, `tanh(gate2)`, and down-projected MLP output differ by 2.736642%,
0.140559%, and 0.848225% relative L2, respectively. Injecting the official
MLP input reduced the down-projection mismatch to 0.311511%, identically
with or without the gate2 clamp; gate2 acts after this projection. The 2x2
effects are:

| Block-0 clamp | Post-block all rows | Post-block text rows | Post-block target rows | Final target velocity | First post-Euler target latent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neither | 0.331287% | 0.320078% | 0.351888% | 1.793929% | 0.128879% |
| Official MLP input | 0.284376% | 0.257019% | 0.330784% | 1.790202% | 0.128611% |
| Official gate2 | 0.346051% | 0.343868% | 0.350220% | 1.780770% | 0.128305% |
| Both | 0.280046% | 0.250599% | 0.329463% | 1.788628% | 0.128699% |

The first post-Euler column uses the source-exact BF16 roundings
`BF16_RNE(x0 + BF16_RNE(dt * BF16_RNE(velocity)))`; recomputing the official
state matched all 65,536 BF16 values and SHA-256
`fd01f30fd1891b0d9bc44f297ebb8996a1126df350b3237f36c152a0b0fb82e7`.
The best local post-block result is the both-clamps arm, but gate2-only is
closest at the final velocity and first latent step; local error reductions
are not monotone through later blocks. The large modulated-input discrepancy
does not by itself diagnose the norm2 kernel, since its native residual input
already differs from the official state. No arm was carried through a
40-step image trajectory, and no face/text quality improvement is claimed.
The per-arm captures and hash-gated JSON reports are under
`/private/tmp/qwen21-native-sublayers-20260928/` in the four directories
ending `_block0_v_q8_off_block0_mlp_{none,mlp_input,gate2,both}`. Their
CPU-only comparison is
`/private/tmp/qwen21-native-sublayers-20260928/block0_mlp_intervention_comparison.json`
(SHA-256 `1af28f750471ddedab7b6ba8ff7e6302222329dca16a5dbbec03bcd3fd897663`);
its fail-closed comparator script SHA-256 is
`9ece2b237921dcc760be3980b61fa54dcf25a62e7a59807a6265269b0ee136b6`.
Refresh
after a change to source tensors, model/Metal code, scheduler arithmetic,
input layout, or scratch-artifact availability; text-prefix/target-suffix
row accounting applies to this pinned fixture, not arbitrary interleaved
conditioning layouts.

### Spatial localization of the matched Q8 latent drift (2026-09-28)

A read-only audit independently hashed all 80 official/Q8 post-Euler state
snapshots from the pinned Russian-sign/portrait prompt, seed 7, at 512x512 and
40 BF16-state steps. Every file matched its recorded SHA-256, had 65,536
finite BF16-exact Float32 values, and the final snapshots matched the saved
bundles. The 40 sigma/timestep records matched within log precision. The
whole-latent relative L2 grows from 0.129% after step 1 to 11.341% after
step 40; there is no late, isolated onset in these snapshots.

The documented face crop `x=[320,380), y=[150,219)` maps coarsely to 20 of
1,024 latent tokens at the 16-pixel stride. An approximate eye-line window
estimated from the official PNG maps to only two tokens; VAE receptive fields
cross these cell boundaries, so this is not an exact eye attribution. At the
final step, the face's absolute RMSE is 0.08124 versus 0.12428 outside it,
and its *locally normalized* relative L2 is 7.696% versus 11.398% outside.
For the two-token eye estimate, RMSE is 0.05816 versus 0.12368 outside, and
local relative L2 is 4.787% versus 11.353% outside. A 12-token eye
neighborhood is also below its complement. The strongest contrary signal is
small and early: the estimated two-token eye RMSE is 1.045x its complement
at step 2, then falls below it by step 4. Thus this run does **not** show a
face- or eye-localized *excess* latent discrepancy, even after avoiding a
misleading whole-image denominator. It does not establish perceptual eye
quality or membership in the VAE's training support; fine facial features may
be sensitive to a distributed latent shift.

The SHA-gated report is
`/private/tmp/qwen21-latent-spatial-20260928/report.json` (SHA-256
`57d3f29dc4bff22edeeb71711696a2a879accd3c41ccb8f9636c6217f56df2d9`),
with a per-step CSV and analyzer in the same scratch directory (analyzer
SHA-256 `fc2212b12ac891c4ee073d93a47b065492843226bce84d332f0faf44c361cf2c`).
An independent final-step calculation reproduced the official and Q8 latent
file hashes, whole/face/eye RMSE, and local relative L2. Refresh the result
after any model, GGUF, conditioning, schedule, latent layout, decoder, ROI, or
ephemeral scratch-artifact change; it is one prompt/seed, not a general facial
quality verdict.

### Fixed-input block-0 Q/K/V discrepancy decomposition (2026-09-28)

The next CPU-only discriminator used eight deterministic rows (four text and
four target, source indices `0,76,152,229,230,571,912,1253`) from the same
call-0 official modulated attention input. The native report's *post-copy*
norm1 override hashes equal the official BF16 and exact-F32-widening input
hashes; the earlier `block0_native_attention_input_modulated` side-tap is
pre-override and must not be treated as the matmul input. Official Q/K/V are
bias-free BF16/MPS linear outputs; native Q/K/V are Q8_0/Q8_0/Q6_K raw
projections copied before Q/K RMSNorm and RoPE. The GGUF Q/K/V payloads,
official shards, captures, row mask, and input were hash-gated. Empirical
direct-versus-transposed comparisons reject a weight-layout transpose despite
the square 4096x4096 shapes.

For each projection, the exact sampled-vector identity is
`native - official = (native - CPU_GGUF_F32) +
(CPU_GGUF_F32 - CPU_BF16_weights_F32) +
(CPU_BF16_weights_F32 - official)`.
The entries below are relative L2 **percentages** normalized to the sampled
official output. They are magnitudes of different vectors, not additive
causal shares:

| Block-0 projection | Native vs official | GGUF payload vs official BF16 weights, same CPU F32 matmul | Native vs CPU GGUF F32 matmul |
| --- | ---: | ---: | ---: |
| Q (Q8_0) | 0.236714% | 0.171539% | 0.000099% |
| K (Q8_0) | 0.234419% | 0.168556% | 0.000095% |
| V (Q6_K) | 0.997847% | 0.984928% | 0.023769% |

The CPU F32 matmul with official BF16 weights, **BF16-rounded at output**,
matched the official MPS BF16 output exactly on all 32,768 selected values
for each head. This reproduces the observed output contract on these rows;
it does not prove MPS's internal reduction order in general. Without output
rounding, the CPU-full-to-official gap is about 0.163-0.166%, making it
unsafe to label that term a Metal error. The native V raw capture is exactly
F16-representable at all 5,136,384 values, unlike Q/K (~0.04%). Rounding the
matched input and dequantized Q6_K V weights to F16, doing CPU F32 matmul,
then F16-rounding and widening the result reproduced native V **exactly on
all 32,768 selected values**. This strongly supports the source's Q6 batch
GEMM F16-boundary route; the runner did not log or guard its route-affecting
environment variables or effective kernel, so the kernel identity is not an
independent runtime certificate. The V weight-payload term dominates this
one sampled raw projection, but its Q6_K base provenance is still unproved:
this is **not** a proof that quantization alone caused the difference, nor
that V dominates the final image error. Earlier Q8 gate/up trajectory tests
show broader distributed accumulation.

The independently rerun comparator, including a one-value corruption
positive control, is
`/private/tmp/qwen21-projection-decompose-20260928/projection_decompose.py`
(SHA-256 `ef20fafa05a2b2eaf49b10c818be58d5a064470dd483eefe6b3ff08e0f5323a4`);
its result JSON SHA-256 is
`15079bcb137ca14de91ee5ee499e35a9835fab0b0c3131169cf1b691e7d7ed69`.
The captured binary SHA-256 is
`bddcac9ba1fccfe49aadc49f97f8949cb91f437c60a2222e2a4185c3be29c075`.
It reports source commit `4821bc5d`; the intervening diff to the current
`c0540ca5` contains only this frontier document and `LANDMARKS.md`, not
math code. This certificate is limited to eight rows, call 0, block 0, and
the pinned GGUF/model/captures. Refresh it if the source/weights, capture
intervention, selected rows, compiler/device/dispatch, or ephemeral scratch
artifacts change. Before a production quality policy, verify Q6_K base
provenance and compare matched full trajectories and decoded eyes/glyphs
across more prompts and seeds, with paired latency and an explicit rollback.

### Fixed-input gate/up and Q6_K V source-candidate checks (2026-09-28)

Two CPU-only probes further separate the block-0/call-0 numerical discrepancy.
They use the same pinned official BF16 source and local GGUF artifacts as the
preceding section; neither establishes the historical converter recipe or
attributes a decoded facial defect.

For `attn.to_v`, exact widening of the official BF16 weight followed by the
tested local no-imatrix Q6_K quantizer did **not** reproduce the base GGUF
payload: 4,920 of 13,762,560 bytes differ, spanning 657 of 65,536 Q6 blocks.
The base and locally generated Q6 weights differ by only 0.065186% relative
L2 after dequantization, whereas their respective gaps to the official BF16
weight are 1.836780% and 1.836776%. A BF16-to-F16 pre-cast produced the same
locally generated Q6 payload, so that tested variant does not explain the
byte mismatch. As a source/layout positive control, quantizing the same
official V weight to Q8_0 reproduced the donor payload byte-for-byte across
17,825,792 bytes. The small Q6-versus-Q6 discrepancy is consistent with a
converter or recipe difference, but a different historical source checkpoint
cannot be excluded. The GGUF header does not record that provenance.

For gate/up, the probe selected four text-prefix and four image-suffix rows
`[0,57,115,229,230,571,912,1253]` and retained all 24,576 output
channels. The official BF16-widened MLP input and the native clamped input
are byte-identical (SHA-256
`1c6e92b512e74ba86e67837b514e2befd4e1a4498aca29c224c86224edc7713a`).
On this **same** official input, the base GGUF Q5_K gate/up projection has
2.229709% relative L2 error against the official BF16/MPS output; the Q8_0
donor has 0.371032%. An official-weight CPU matmul rounded to BF16 at output
is within 0.007648% of that capture (61 of 196,608 values differ). A local
no-imatrix Q5_K re-quantization changes the base Q5_K projection by only
0.042195% relative L2 on these rows. This supports weight precision as a
substantial source of the sampled base projection error, without proving
the original Q5_K conversion lineage.

There is also a distinct upstream-state effect: the native hybrid baseline MLP
input differs from the official input by 3.255375% relative L2 on these
rows. Holding Q8 weights fixed while changing only that input shifts the
projection by 1.593681%. These vector magnitudes are not additive shares
of trajectory error. The native packed Q8 Metal projection on the clamped
official input agrees with the CPU Q8 calculation to 0.0000232% relative
L2 (maximum absolute difference `7.15e-7`); a `+0.01` corruption was rejected
by the numerical matcher. Thus this probe finds no packed-Metal arithmetic
defect at this boundary. It does not determine where the earlier MLP-input
drift arose, nor whether either error controls eye or glyph quality.

The corrected, hash-gated Q6 provenance and dequantized-error reports are
`/private/tmp/qwen21-q6-provenance-20260928/q6_provenance_report_r3.json`
(SHA-256 `16f5f2ea72ac2a7cb4a3522535ce0dd5ce769a6b34fe1cf07b93ef3309375faa`)
and `q6_dequant_error_stats_r3.json` (SHA-256
`b57d3ba3e93504de0368f4b0081a6e7a60945c291b2a55d086245af103cec092`).
The gate/up script and report are
`/private/tmp/qwen21-gateup-precision-20260928/gateup_fixed_input_probe.py`
(SHA-256 `6072ccdcd7c54744e8f6ae63473b275c4fd0a9908d583d5b06410f4ba29e96d8`)
and `gateup_fixed_input_probe.json` (SHA-256
`b02e249523782ad6f32dd8ff34e7e665570f26a2dd81b99adfa0890d144ad931`).
Refresh these scoped results if the source or GGUF tensors, quantizer build,
row selection, model/Metal implementation, captures, or ephemeral scratch
artifacts change. Later-block matched MLP inputs and decoded multi-seed
comparisons remain necessary before selecting a precision policy.

### Same-state full-Q8 donor DiT discriminator (2026-09-28)

The earlier "Q8" trajectory and teacher-forced comparisons used the selective
gate/up hybrid, **not** the unmodified full-Q8 donor. A separate guarded
one-forward native Metal run loaded the full donor GGUF (SHA-256
`c3ef62b2b7b53bf92418cbd77fbc24b43a26c8305a1f001c4a9a9a99b1373c03`)
at official call indices 0, 20, and 39. All three variants used the same
official BF16-exact 65,536-value state, official Qwen3-VL conditioning payload
`007ad14a...66114ab`, and BF16-effective model time at each index. Root
independently re-read the raw official BF16 teacher and all nine native F32
outputs and recomputed the finite-value metrics in Float64:

| Official call | Q4 velocity error | Selective-Q8 velocity error | Full-Q8 velocity error |
| ---: | ---: | ---: | ---: |
| 0 | 2.725202% | 1.760615% | 1.350547% |
| 20 | 1.363841% | 0.982294% | 0.830599% |
| 39 | 4.669017% | 2.266090% | 1.974060% |

Each value is `||native velocity - official BF16 velocity||₂` divided by the
official velocity norm at that **same incoming state**, not a final-latent or
visual-quality score. At call 0 the official route extracts its prefix and the
native route builds it, making this the primary matched-state teacher contrast.
At calls 20/39 the official route uses a warmed cache while each native
one-forward run builds a fresh prefix; those teacher contrasts retain a
cache-route confound. Within the native runtime, the full donor differs from
the hybrid only in the 32 V and 32 MLP-output tensor qtypes/payloads plus
`general.file_type` metadata (unused by the forward source); the other 201
same-type tensor payloads were byte-identical in the streaming audit above.
The V/MLP-output contrast changes weight payloads and dispatch together, and
their complete source provenance is not established, so this is not a pure
quantization-error attribution.

The full donor lowers aggregate velocity relative L2 at all three sampled
states, yet its call-0 maximum absolute element error is **0.276533**, above
the hybrid's **0.266531**. Thus a better global norm does not certify every
feature, eye, or glyph. The remaining 1.350547% call-0 velocity discrepancy
occurs before Euler integration and VAE decoding, but this whole-forward
measurement does not localize the residual to a particular DiT block or
operation. The native cache-route and full-Q8 trajectory discriminators
follow below, followed by a 32-block equal-official-input replay. The earlier
selective-Q8 trajectory and cumulative block trace do not substitute for that
full-Q8 local replay.

The scratch comparator checks exact model/output hashes, official state and
teacher bytes, conditioning, time, shape, finiteness, and nonzero native
artifact contrasts. A deliberately altered byte string was rejected by its
output-hash guard. The comparator SHA-256 is
`009747a6c5b4dff659fec5fc89a58cff9f19586743dba25dbb2c8c0cfca94235`;
the resulting report SHA-256 is
`11d6f978732428af3d00973cb3c3231570616a6682f5f243c6f7ed4c0ca4a46c`
under `/private/tmp/qwen21-full-q8-teacher-20260928/`. Its source runner
inherits a misleading `model_sha256_verified=false` flag from the
inputs-only preflight even though the recorded full model SHA matches the
pre-pinned donor and root independently streamed the donor hash. The
comparator checks that actual SHA equality rather than trusting the flag.
Refresh this certificate after source, compiler/device/dispatch, model,
conditioning, schedule, cache route, or ephemeral scratch-artifact changes.

### Native full-Q8 prefix-cache route discriminator (2026-09-28)

A separate guarded Metal probe replayed the pinned official BF16 incoming
states at DiT calls 20 and 39 with the full-Q8 donor. For each call it compared
three **native, same-weight** routes: a prefix cached by the call-0 forward and
reused at the target call, a fresh stack that builds the prefix at the target
call, and a forced full resident-input path that bypasses prefix caching. The
call-0 seed was also checked against the pinned standalone full-Q8 output.
Each target route returned the full 1,254-token joint output, not merely an
unverified target slice. Its counters confirmed different execution routes:
cached hit `builds=1, hits=1, active_tokens=1024`; fresh build `1, 0, 1254`;
forced uncached `0, 0, 1254`. Each call had one Metal command buffer, zero
intermediate readbacks, one final readback, and 1,024 image-projection rows.

At **both** calls, all three full-output SHA-256 values were byte-identical,
as were all target-suffix values: call 20 full
`ab49da337b2603188b45247a1f6dbaffadf5ad0f1e0ffcccd8e0e0975dc0f0bb`
and target `e83a6a6d1941f3c733b518d145f22771a99cdf5e6aa799acacd5b5510a5bc71d`;
call 39 full
`4c791e2a5ea5a386b025588c35686a91531c2e97d7001eab80fe6e3c327059e3`
and target `9a2b652402204d25c0dd8588e53be8b6198ec0aa4794197b2ec5bed5e6c481f1`.
Pairwise maximum absolute difference and relative L2 were exactly zero. The
target hashes independently match the earlier fresh one-forward outputs, so
the test did not silently compare a different target state or timestep. This
removes **our native cache-route choice** as an explanation of the residual at
these two fixed inputs. It does not reproduce Hugging Face/MPS cached-prefix
values or prove that its cache arithmetic matches ours; the cross-runtime
official residual remains broader than this native route test.

The first isolated-agent launch could not create a Metal device and produced
no parity result; root reran both guarded indices with Apple M2 Max access.
The report has one non-computational bookkeeping error: its
`warm_seed.stats.prefix_cache_hits_after_call=1` is read after the target hit,
even though the runner asserted `hits=0` immediately after the call-0 seed.
The per-target route counters are snapshotted at the correct boundaries; this
delayed seed counter does not affect the output comparison. Source/binary
SHA-256: `ea575da6377e265a510d56d1f499e72bf584aff6be4dc0170a1178deb7160b0e` /
`1e17e778205c1e55bd86c994a5e6edf3825e1e02123ab999378b55dd75b48006`.
The reports under `/private/tmp/qwen21-cache-route-parity-20260928/results/`
have SHA-256 `0beb160bba1a0f0961a292e749f10d9a427d6d159ff0a81270d972f68ed80799`
(call 20) and
`efacde8bf7e0b891ab88a60b4c613b40d84eac176d9ef549e1564d0a35759009`
(call 39). Refresh after source, binary, GGUF, conditioning, schedule,
device/driver, or scratch evidence changes; an upstream cache comparison
would need separately pinned Hugging Face internal tensors.

### Full-Q8 donor 40-step latent trajectory (2026-09-28)

A guarded native Metal run completed all 40 Euler steps with the **full**
Q8 donor, the same official Qwen3-VL conditioning, seed 7, 512x512 latent
shape, pinned schedule, and BF16-effective timestep/post-step state semantics
used by the official, Q4, and selective-Q8 controls. The runner exited zero;
its independent comparison reported `COMPARISON_COMPLETE`, validated 40
candidate and reference snapshots, and checked the initial latent,
conditioning, model revision, scheduler config/sequence, and final bundle.
Root independently re-read all four sets of 40 F32-widened BF16 state files
and recomputed relative L2 in Float64:

| Completed step | Native Q4 | Selective Q8 gate/up | Full Q8 donor |
| ---: | ---: | ---: | ---: |
| 1 | 0.156901% | 0.129089% | 0.107892% |
| 10 | 2.357001% | 0.666186% | 0.476661% |
| 20 | 9.177365% | 3.744071% | 3.268054% |
| 30 | 17.335191% | 8.525851% | 7.437417% |
| 40 | 21.277440% | 11.341314% | 9.881818% |

Each entry divides the candidate-minus-official norm by the **same-step
official** latent norm. All three global error series increased at every
completed step; full Q8 was closer than the selective hybrid on all 40.
There is no isolated late-step cliff in this metric. At step 40, the
approximately mapped 20-token face ROI has 16.708862%/7.695944%/3.810557%
relative L2 for Q4/hybrid/full Q8, respectively; the approximate 70-token
sign ROI has 9.346851%/3.056520%/1.800065%. These are token-space distance
measurements, not pixel-aligned eye or glyph fidelity scores, and their local
denominators differ from the global denominator. The full-Q8 residual remains
large enough that better aggregate precision cannot be equated with a correct
rendered face. The same-state call-0 DiT residual above establishes an error
before Euler and VAE; the steadily growing trajectory is consistent with
feedback amplification but does not identify its exact source. Matched VAE
decoding and an independent-input replay of all 32 full-Q8 blocks follow below.

The run log, final bundle, and trajectory report live under
`/private/tmp/qwen21-full-q8-40-20260928/`. SHA-256 values: `run.log`
`cee9b3ce166c07ec832afb8d65899cfe32bb0fa4ba9b449f2a623160df0f3f92`,
final manifest `5db5f4695742fb6eb15cade80bb030d3eab7f7f749accaeb5e599f8043b22668`,
final payload (also `step-039.bin`)
`25745f532985948aa2837de7798624b4e17e460594f1e8e836196f4ec88521e9`,
and comparison report
`569f1bf660af378744a665c01ce963ba2367310c8cd5ec4f41fe87bfe7b1aa8f`.
The guarded driver SHA-256 is
`995bb94b6469438b484de8e600b0e5c7bd46d6b68936b13c85a43f18fcda3f28`.
Refresh after any model/conditioning/source/driver/scheduler/BF16 semantics,
Metal device or route, or ephemeral scratch evidence changes.

### Matched full-Q8 CPU/FP32 VAE decode (2026-09-28)

The full-Q8 final latent was decoded through the same pinned CPU/FP32 VAE
as the official, Q4, and selective-Q8 endpoints. Before candidate decoding,
the runner independently re-decoded those three baselines and required exact
RGBA pixel equality with their pinned oracle PNGs; all passed. A synthetic
one-pixel RGB defect in the fixed face crop and a one-byte hash mutation were
detected by the guards. Root independently recomputed RGB pixel differences
from all four PNGs:

| Candidate versus official | Full-frame RGB MAE (0–255) | Fixed face RGB MAE (0–255) |
| --- | ---: | ---: |
| Native Q4 | 9.784980 | 15.985185 |
| Selective Q8 gate/up | 5.683842 | 8.876973 |
| Full Q8 donor | 4.073781 | 4.409179 |

The fixed face crop is `x=[320,380), y=[150,219)` in the 512x512 image.
The full-Q8 face crop has RGB RMSE 6.391382 and maximum absolute RGB channel
delta 50 versus official. The VAE emits RGBA, and alpha arrays differ
slightly between *different* latents; RGB metrics explicitly exclude alpha.
The full-Q8 decode is visibly closer to the official scene and face on this
one prompt/seed, but small face details still differ. Pixel MAE does not
measure eye anatomy, lettering correctness, perceptual quality, or VAE
training-support membership. Exact baseline re-decode plus the earlier
same-state DiT error rejects a **VAE-only origin** for the drift, not the
possibility that VAE decoding magnifies some local latent differences.

The reviewed output under
`/private/tmp/qwen21-full-q8-visual-20260928/decoded-final-reviewed/`
is `full-q8-final-cpu-fp32.png` (SHA-256
`dd7b0187749d0cac4253038a285d65c023d0855f8a02f321219d7608c0d2bee7`)
and `report.json` (SHA-256
`d906df9512be133d0600891d1a3f168b1b7a2c1c71881a3030956c7c3cbf82da`).
The scratch driver SHA-256 is
`58ee64d9a90fc360513cf049f89abaa9b420f56407023402c9c9b7faf892bde5`.
Refresh after source/decoder/VAE/asset/baseline/candidate changes, or if
ephemeral scratch evidence is removed. Multi-prompt/seed perceptual and
eye/glyph evaluations remain open before any precision policy is promoted.

### Full-Q8 equal-official-input DiT block replay (2026-09-28)

A guarded Apple M2 Max Metal run independently invoked each of the 32 full-Q8
DiT blocks at official call 0. Block `i` received the **official BF16 output
of block `i-1` widened exactly to F32** (block 0 received the official
pre-block state), plus the captured official modulation. Each native output
was compared with that block's official BF16 output; no native block output
was passed to the next block. The runner validated all 33 official BF16
state hashes and every widening, the conditioning/layout and model pins, and
all six projection qtypes in each block as Q8_0. It completed 32/32 blocks,
and an exact second invocation of block 0 produced a bit-identical output.

| Block | Text relative L2 | Image relative L2 | Image absolute RMSE |
| ---: | ---: | ---: | ---: |
| 0 | 0.255690% | 0.346766% | 0.022596 |
| 5 | 0.358750% | 0.239864% | 0.018540 |
| 14 | 0.185531% | 0.480790% | 0.025555 |
| 30 | 0.203435% | **0.740903%** | 0.042019 |
| 31 | 0.325699% | 0.532138% | **0.120285** |

Every block has a nonzero local discrepancy: image relative L2 ranges from
0.225779% to 0.740903%, text from 0.165547% to 0.358750%. Block 30 has the
largest **relative** image discrepancy on its own official input; block 31
has the largest **absolute** image RMSE, but its official image-output norm
rises from 11,614.85 at block 30 to 46,293.41 at block 31. Thus its larger
raw RMSE does not by itself identify a new kernel failure or its contribution
to the final face. Rounding only each native output once to BF16
round-to-nearest-even increased image RMSE on all 32 blocks, so a final
output-format mismatch alone does not explain these block-level residuals.
The test cannot separate Q8 weight quantization/provenance from internal
Metal-vs-MPS arithmetic or implementation differences. It excludes the final
normalization/output head and does not measure how each local error propagates
through later blocks or Euler steps. Earlier selective-Q8 cumulative block
captures and this full-Q8 equal-input replay answer different questions; no
particular block is yet proven to cause the visible eye or glyph differences.

The guarded wrapper exited 0 after about 122 seconds with a 3,600-second,
24,576-MB process-tree and 35% system-free-memory stop guard; the runner
separately required at least 50% free memory before loading and observed 79%.
The report is
`/private/tmp/qwen21-full-q8-block-replay-20260928/gpu_replay_report.json`
(SHA-256
`f0550ba5fb25a14063118eeb65e361cbb061b5fdf985a06193ea8a45cfc5267e`);
the scratch runner and binary SHA-256 values are
`6ee848c25e65bbd1632e9b8b997b1b299a421283ddf8c8ae0d268307f6f2cdca`
and `0cad665e0c4e92f134e79a622bd6803608c42e8a45b88f7fd2b2ffc18cf164ce`.
The CPU preflight report SHA-256 is
`33d2bcba8562027699c8ec31acc9fd92e870e845443f23a814e59608faf718d5`;
its wrong-hash negative control, block-0 cross-capture identity, BF16
tie-to-even control, and metric known-answer control passed. Refresh this
certificate after source/compiler/device, model/conditioning, official
capture, modulation/layout, or scratch-artifact changes. The next causal
discriminator is an operator-level matched-input split of the strongest
image-error blocks and the output head, ideally with matched BF16 weights to
separate quantization from implementation arithmetic.

### Full-Q8 latent-drift onset and state transport (2026-09-28)

A separate SHA-gated spatial audit compared every one of the 40 saved
official and full-Q8 post-Euler latent states on the same prompt, seed,
schedule, and **saved official Qwen3-VL conditioning**. The full 32x32x64
latent relative L2 gap rises from 0.107892% after step 1 to 0.476661% after
step 10, 3.268054% after step 20, and 9.881818% after step 40. A fixed
20-token approximate face region (rows `[9,14)`, columns `[20,24)`) reaches
3.810557% locally normalized L2 at step 40. This is not an eye-anatomy
metric. A separate **post-hoc selected**, non-face 20-token hotspot (rows
`[18,23)`, columns `[5,9)`) contains 4.594% of cumulative squared error at
step 10, 10.520% at step 11, 21.706% at step 12, and 46.672% at step 20;
its area is 1.953% of image tokens. The post-hoc choice makes this a
description of where the error concentrates, not an unbiased detector or
proof of its cause. The squared *new error increment* within that same
region is 10.786% at completed step 10, 24.217% at step 11, and 35.867%
at step 12; the sharp local growth is around calls 10–12, not the first
appearance of a latent mismatch. Same-official-state full-Q8-versus-official
DiT velocity error is *not* concentrated there at the earlier sampled calls 0/20/39
(3.826%/1.744%/1.540% of squared velocity error), so the large latent
hotspot must not be read as an equally localized one-call operator error.

To discriminate local DiT output mismatch from propagation of an already
different input, a new same-binary full-Q8 Metal forward was run at both
official and native saved incoming BF16 states for calls 10 and 11. The
binary/input preflight checked the exact 65,536 BF16 values, and the reports
recorded the full-Q8 donor identity, official conditioning, and scheduler.
An independent CPU hash of the currently referenced 7.2-GB donor matches
the expected SHA; the capture itself did not assert model-byte rehash at
forward time. With the official FP32 sigma delta and BF16 round-to-nearest-even
at velocity, delta, and state-update boundaries, the counterfactual native
Euler update replayed the saved native post-step state **65,536/65,536
bitwise** at each call. The resulting
identity splits `native_next - official_next` into
`native_update(official_input) - official_next` and
`native_update(native_input) - native_update(official_input)`.
These are one-step state-vector terms, not a linearized derivative or a
unique causal allocation.

| Incoming call / completed step | Direct same-input one-step gap | Input-state transport term | Full next-state gap | Transport gain in post-hoc hotspot |
| --- | ---: | ---: | ---: | ---: |
| 10 / 11 | 0.116878% | 0.537981% | 0.549960% | 1.615x |
| 11 / 12 | 0.107540% | 0.699382% | 0.702207% | 1.806x |

The three gaps above are L2 norms relative to the official *next latent*,
so their scalar magnitudes do not add. Direct and transport vectors have
cosines -0.0050 and -0.0506. At call 11 the transport term contains 21.874%
of its squared energy in the hotspot, while the direct term contains 1.385%;
outside the hotspot, transport gain is still 1.170x, so this is local
amplification within broader drift, not an exclusively local failure. At
call 20, a separate exact native-state replay decomposes the **velocity**
gap into 0.830599% direct same-official-input and 9.568147% state-transport
terms relative to the official velocity norm (total 9.550403%); at call
39 these are 1.974060% and 16.237064% (total 16.317905%). Thus a
same-input DiT discrepancy exists early, and the propagated state difference
dominates by the measured calls. The strongest supported diagnosis is
positive feedback in the denoising trajectory, not a VAE-only failure.
This initial decomposition had no official raw velocities for calls 10/11,
so its table alone cannot label the direct one-step term a DiT-velocity
error. The subsequent official-velocity capture below closes that scheduler
confound for these two calls under the pinned native BF16 Euler formula.

This experiment does **not** establish the initial source of the mismatch:
Q8 weight representation, a Metal/MPS arithmetic boundary, block operator
ordering, and official cached-versus-native fresh-prefix behavior remain
separable hypotheses. The fixed official conditioner deliberately excludes
native Qwen3-VL accuracy from this A/B; it does not clear that encoder for
end-to-end use. A byte-exact CPU/FP32 VAE re-decode rejects a VAE-only
origin, but neither these norms nor spatial energy prove that the VAE was
fed an out-of-support latent or that this hotspot caused the visible eye or
glyph defects. The most useful next discriminator is an observationally
controlled operator-level split on official block-0 inputs, then a paired
official-versus-native conditioning trajectory and multi-seed perceptual
evaluation before any quality-policy change.

The spatial auditor and report SHA-256 values are
`238dd4fede0e29ea6fe40795c6d9a263432269e30c1145a6efb40ad7a7d0d718`
and `2836f90a843e0c047b8d5045fc373bc12f4e4bc3650f55d6a5f9626e5e4efe47`;
the guarded onset auditor/report values are
`4efa0acb8218783b237a3dcf44ae26bc3e14eb37be0c1e65b31046333b486dac`
and `6b2309f524155de5599f1a69142ec384a5c988487bdd4b99683062ea8211fe44`.
The onset full-Q8 binary SHA-256 is
`9446461e14fe2c5ae2a8a9175a3415b0c2f145789fdb7d999100e740192fe0e5`;
call-20/call-39 decomposition reports have SHA-256
`1d3b6109909bcd4c5157f0a445e92ffc7fac328a733a0c8a2857ddcb52086e9c`
and `d06c468fff311bdeba6f70792ac603d4ea8d9ed1a2c399bb0e1d01df74a6a70d`.
Scratch reports and source are under `/private/tmp/qwen21-spatial-drift-20260928/`,
`/private/tmp/qwen21-onset-20260928/`, and
`/private/tmp/qwen21-state-decomp-20260928/`. The reused onset binary's
free-text interpretation incorrectly calls native-input runs official, and
an input-subreport says `model_sha256_verified=false`; the auditor checks
input/output hashes and the reported model/conditioning identities, while
the independent donor rehash checks the current file, not bytes at capture
time. Refresh after source/compiler/device, quantized model, conditioner,
scheduler, trajectory captures, or scratch evidence changes, and before
generalizing to other prompts, seeds, resolutions, or precision policies.

### Full-Q8 first-block operator frontier (2026-09-28)

A guarded call-0 full-Q8 Metal comparison now narrows the earliest *matched*
DiT operator difference. The official BF16/MPS hooked capture retained exact
terminal-velocity parity with its no-hooks control. The native no-taps and
18-tap runs used one compiled binary and produced exactly the same terminal
velocity SHA-256
`f5fa49331995205540a67caa718a97d14c7399df82a33deaeeb2f726863ce2fe`.
All 32 layers' six projection families were validated as Q8_0. Both arms
used the same initial latent, saved official Qwen3-VL conditioning, official
time features, and an exact official BF16-widened pre-block-0 state. For
this diagnostic only, block 0 also *consumed* the exact official modulated
attention input and `tanh(gate1)` through explicit overrides. The saved
native norm1/gate1 values were copied **before** these overrides: they are
shadow observations, not the inputs to the measured Q/K/V projections or
attention residual. Their respective all-row relative L2 gaps were
0.353862% and 0.269771%, so the unclamped native path has an additional
upstream discrepancy that this intervention does not assign causally.

Under the matched consumed attention input, the first directly comparable
nonzero outputs are the parallel raw Q/K/V projections, before Q/K RMSNorm,
RoPE, attention weighting, later blocks, or VAE. Exact official BF16 captures
were widened to F32 for comparison with native F32 taps:

| Block-0 boundary | All-row relative L2 | Image-row relative L2 |
| --- | ---: | ---: |
| Injected pre-block state | 0% | 0% |
| Raw Q / K / V projections | 0.243992% / 0.237986% / 0.337856% | 0.244129% / 0.235977% / 0.335852% |
| Attention `to_out`, before gate1 | 0.234882% | 0.203929% |
| Post-attention residual | 0.238281% | 0.329206% |
| Norm2-modulated MLP input | 2.710809% | 2.833675% |
| MLP gate / up projections | 1.387420% / 1.496385% | 1.622878% / 1.931347% |
| SwiGLU / down projection | 1.454428% / 0.825618% | 2.348028% / 0.747446% |
| Post-block state | 0.331731% | 0.372024% |

The norm2 and later MLP gaps inherit an already-different attention
residual; none isolates a faulty norm2 or MLP kernel. A comparator initially
swapped the gate/up slices and falsely reported a 129.8% gate gap. The
corrected mapping follows the native shader's gate-first/up-second layout;
the packed buffer independently reconstructs its saved SwiGLU output at
5.81e-8 relative L2. Gate/up cosine checks also reject the swapped mapping.
The official post-RoPE Q/K reference remains CPU-derived, not directly
observed on MPS, and there is no official attention-context tap immediately
before `to_out`. This locates an operator *frontier*, not a unique kernel
or weight cause: Q8 representation, BF16-versus-F32 arithmetic, and backend
matmul behavior remain confounded. Nor does a call-0 block-0 discrepancy
establish which component caused the call-10-to-12 spatial hotspot or face
defects. The subsequent official raw-velocity capture and exact scheduler
replay resolve the call-10/11 direct term; matched-weight projection tests
remain the next discriminator between representation and execution paths.

The hash-gated comparator is
`/private/tmp/qwen21-block30-split-20260928/full-q8-block0/compare_full_q8_block0_official.py`
(SHA-256 `30f15ea66f5dedfe519011a8d1e2c0882d306e263057e1f0a2447b3862ba0dd0`);
its report is `operator_split_report.json` in the same directory (SHA-256
`2aa42c861628e70dd8dfb1b207e19c976e19489f7562618ff99337370eaaa7f7`).
The guarded runs exited zero with the 35% free-RAM floor, 24,576-MiB
process-tree RSS cap, timeout, and process-group containment intact.
Reproduce the CPU comparison with `python3` on the comparator path above;
refresh this evidence after official/native captures, model payload, Metal
source, BF16 runtime, conditioning, or scratch availability change.

### Official onset velocities: separating DiT from Euler (2026-09-28)

The pinned official BF16/MPS DiT was reloaded once, with the original
seed-7, 512px Qwen3-VL conditioning payload, 40-step schedule, and saved
incoming BF16 states. A call-0 `extract` prefill built the condition KV cache;
its raw velocity matched the saved official call-0 SHA-256 exactly. Two
`cached` forwards then measured **zero-based** calls 10 and 11 (saved
step-009 to step-010 and step-010 to step-011, respectively). The prior
section's notation means call 10 completes step 11 and call 11 completes
step 12: the saved files use zero-based filenames. Each official MPS
scheduler replay, with the appropriate isolated scheduler begin index,
matched its saved next latent **65,536/65,536 BF16 values**. The selected
velocity readback transfers the full BF16 transformer output to CPU before
slicing; an offset-view F32 readback disagreed at 65,454 elements in the
call-0 prefill and is not used as an oracle.

An independent hash-gated CPU comparison applied the native BF16
round-to-nearest-even Euler formula to the *captured official velocity* at
the same official input. At both calls it reproduced the saved official next
latent bit-for-bit. Swapping only the model's emitted velocity for the
captured native full-Q8/Metal F32 velocity, rounded to BF16 at the native
Euler boundary, then reproduced the full direct one-step term from the prior
decomposition. Only 7/65,536 and 5/65,536 native F32 velocity values at
calls 10/11 were already BF16-exact; retaining native F32 through the
product instead would yield 0.116888% / 0.108709% one-step gaps, not the
saved-native-path 0.116878% / 0.107540%. Thus the measured direct state
gap at these calls is due to a different emitted DiT velocity, **not** a
scheduler-only gap for these exact inputs. This does not establish why the
DiT emits a different velocity.

| Zero-based DiT call / completed step | Native-vs-official raw velocity relative L2, official velocity norm | Scheduler-only state gap | Velocity-swap state gap, official next-state norm | Post-hoc hotspot share of raw velocity squared error |
| --- | ---: | ---: | ---: | ---: |
| 10 / 11 | 1.255609% | 0% (bitwise) | 0.116878% | 13.076% |
| 11 / 12 | 0.951932% | 0% (bitwise) | 0.107540% | 1.494% |

The hotspot is the *post-hoc*, non-face 20-token region covering 1.953% of
image tokens. Its direct one-step squared-error shares are only 5.175% and
1.385% at these two calls. By contrast, at call 11 the earlier
state-transport term places 21.874% of its squared energy there and is
0.699382% of the next-state norm versus the 0.107540% direct DiT term.
This sharpens the finding: local DiT error is real, but the large
call-10-to-12 latent drift is predominantly propagation/amplification of
an already-shifted latent in this one trajectory. It is **not** an
eye-anatomy measurement or proof that the VAE received an out-of-support
input. Saved official conditioning means native Qwen3-VL quality is outside
this comparison. The full-Q8 block-0 Q/K/V split identifies an earlier
operator frontier, but connecting that specific boundary to the later
hotspot, and separating Q8 representation from BF16/F32 arithmetic and
backend implementation, still requires matched-weight interventions.

The guarded official capture script and report are under
`/private/tmp/qwen21-official-onset-velocities-20260928/` (SHA-256
`800dc5c2273ce218b6fd04b86fd52bddbf8856cb9a45b5d698073def8df51a00`
and `a7c5620622babb1fcf5c543d4800fe1134c258c26857dd29cbfe6f23034d4027`).
The independent comparator and report are under
`/private/tmp/qwen21-onset-20260928/` (SHA-256
`8239649d3d01f7c231b2b072dce457ec4d0d162e715fde3fcc549969ebe776a8`
and `e39d06f234b48e7d3ac0c51dcd67e05881faab4d01adcd340cde7db87b5ffb43`).
Controls expire if source weights, conditioning, schedule, BF16 runtime,
native Metal route, or saved scratch artifacts change. Neither a full
trajectory nor a decoded-image A/B was rerun in this slice.

### Sparse block-0 Q/K/V weight-versus-execution split (2026-09-28)

A CPU-only probe used the exact official BF16-widened modulated attention
input consumed by the native full-Q8 block-0 Q/K/V route. It selected seven
fixed joint-token rows: text 0/114/229 and image 230/758/811/1253. The
image positions include a center *proxy* and one post-hoc hotspot token;
image indices are offset by the 230 leading text tokens. Source BF16 shard,
full Q8 GGUF, official/native taps, mask, override-consumption proof, and
matrix layout are hash- or negative-control-gated. Both official and Q8
weights were multiplied by the *same* F32 input on the *same* CPU backend.
The wrong matrix orientation differs from the relevant captured tap by more
than 100% relative L2, whereas the selected dequantized-Q8 CPU output and
native Metal Q8 tap agree within about 0.0001% relative L2.

| Projection | CPU Q8-weight substitution vs CPU BF16-weight F32 output | CPU BF16-weight F32 vs official BF16/MPS tap | CPU dequant-Q8 F32 vs native Metal Q8 tap | Observed native Q8 vs official BF16/MPS tap |
| --- | ---: | ---: | ---: | ---: |
| Q | 0.175611% | 0.159989% | 0.000099% | 0.239672% |
| K | 0.176577% | 0.163396% | 0.000095% | 0.241529% |
| V | 0.309233% | 0.165307% | 0.000104% | 0.350058% |

Each cell is a vector relative-L2 on **these seven rows**, normalized by
the official MPS tap for a common scale. The official-weight CPU F32 result
becomes **bit-identical** to all sampled official MPS outputs after BF16 output
rounding; this is an observed sample property, not a general BF16-MPS GEMM
equivalence. Rounding the CPU Q8 output to BF16 does not close the gap:
Q/K/V remain 0.249424% / 0.252814% / 0.369436% from the official taps.
The three delta vectors (Q8-weight substitution, official output rounding,
CPU-to-Metal Q8 residual) exactly reconstruct the observed native-minus-
official delta on the sample; their norms are **not additive causal shares**.
Thus the sampled raw projection discrepancy is consistent with a substantial
Q8-weight substitution plus BF16 output-format difference, while the native
Q8 Metal projection arithmetic adds very little *on these rows*. This does
not prove the donor Q/K/V were quantized from the pinned BF16 matrices, a
full-tensor or later-step attribution, or a causal link to eyes, hotspot, or
VAE support. The call-10/11 matched-input intervention below tests whether
replacing only block-0 Q/K/V has a material terminal effect at the measured
onset; it does not run a complete trajectory or decoded-image A/B.

The CPU comparator, report, and sampled vectors are under
`/private/tmp/qwen21-block0-weight-vs-arithmetic-20260928/`; script SHA-256
`521d256dd0bb4ce6f09be348cd1327774be3a92de4f185eab461850eb841a6de`,
report SHA-256
`b066ef7328589812e54b8ef55267205d847cfcc0761c10310e11e5b98e937f78`,
and NPZ SHA-256
`1be326bf1a767c2147469c2969c2305721f7a36a23d55ff661421617f652de66`.
The guarded CPU-only, single-thread run exited zero below its 1.5-GB
post-run RSS threshold (observed ~1.00 GB). Wrong-hash and
wrong-orientation negative controls passed. Refresh if either model payload,
saved taps or input, mask/order,
modulation override, runtime arithmetic, or scratch artifact changes.

### Matched-input block-0 Q/K/V intervention at calls 10 and 11 (2026-09-28)

A scratch-only, one-forward Metal probe held the saved official BF16 latent,
official Qwen3-VL conditioning, schedule, and full-Q8 donor fixed at each
zero-based call. The Q8 no-op controls reproduced the earlier native velocity
files **bit-for-bit**: SHA-256 `cf040964...0bdc3d` at call 10 and
`5176ff28...cc959` at call 11. The probe captured the same block-0 modulated
attention input in each arm (identical 20,545,536-byte Float32 file and hash)
and rebuilt the prefix cache in a fresh process per arm. The treatment
replaced only block-0 Q/K/V with official BF16 weight values widened exactly
to Float32; all other block-0 weights were identity-checked. The earlier
call-0 sampled projection test supports the weight orientation, but there is
no official raw-Q/K/V teacher tap at calls 10/11.

| Call | Raw Q / K / V treatment-minus-Q8 rel-L2, normalized by Q8 tap | Q8 velocity rel-L2 to official | Treatment velocity rel-L2 to official | Q8 direct next-latent gap | Treatment direct next-latent gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.143420% / 0.155463% / 0.293004% | 1.255609% | 1.254056% | 0.116878% | 0.116337% |
| 11 | 0.139149% / 0.153263% / 0.292894% | 0.951932% | 0.961180% | 0.107540% | 0.107928% |

The direct next-latent gaps apply the same BF16-rounding native Euler formula
to each arm's velocity at the *official* incoming state, then compare with
the saved official next state. Root independently recomputed those gaps and
the velocity/vector metrics from the raw files. At call 10, the treatment
reduces velocity squared error by only 0.2473%; at call 11 it increases it
by 1.9524%, with a treatment-delta cosine of -0.20954 toward the official
velocity. Therefore this particular block-0 Q/K/V intervention does not
explain the main same-input DiT velocity discrepancy at the measured onset.
It cannot rule out other block-0 operations, later blocks, Qwen3-VL drift,
or a spatially sensitive decoded-image effect. The 20-token face proxy and
post-hoc hotspot are not eye-quality or causal metrics.

This is deliberately a **mixed weight-and-execution-route** intervention:
the treatment's Float32 weights use Qwen35Metal F32 GEMV while the baseline
uses native Q8_0 projections. The small, non-monotone terminal response is
not a clean weight-only effect. The first default-sandbox launch could not
create a Metal device and produced no model output; the unchanged binary's
device-enabled retry passed the exact Q8 control before any treatment.
Each device-enabled forward ran under the 300-second, 32-GiB process-tree
guard. The probe binary/source/instrumented-Metal SHA-256 values are
`c479e58b...7e297` / `c47d12df...05459a` /
`cad06b88...8941`; the paired call-10/11 metric-report SHA-256 values are
`4e25caf8...15abf5d` / `eadcdb58...4b4fd0`. Scratch artifacts are under
`/private/tmp/qwen21-call10-matched-weight-20260928/`. Refresh after model,
conditioning, scheduler, source, compiler, Metal device/runtime, or saved
input/tap changes. The next useful discriminator is matched-state blockwise
capture/replay at this onset, followed by a complete trajectory and decoded
images only if a correction materially improves the terminal velocity.

### Call-10 full-Q8 latent-drift localization across all DiT blocks (2026-09-28)

At the measured onset (zero-based call 10), a pinned official BF16/MPS
`cached` forward and a full-Q8 native Metal forward consumed the same saved
official latent and Qwen3-VL conditioning. The official capture retained the
target-image hidden state immediately before block 0 and after each of the 32
blocks; the native capture copied the corresponding 1,024 target-token rows
within the same command buffer, without feeding observations back into the
forward. The official velocity matched the earlier capture and its Euler
replay matched the saved next latent 65,536/65,536 BF16 values. The native
baseline velocity matched its earlier no-tap control byte-for-byte. All
official BF16 dumps were checked as exact F32 widenings; all native dumps
were SHA-checked, finite, and full-Q8_0 across all six projection families
in every block.

The native pre-block target state differs from the official BF16-widened
state by 0.167342% relative L2, but this is almost entirely the output
representation boundary: after rounding the native F32 values once to BF16
round-to-nearest-even and widening, 4,194,190/4,194,304 values match the
official tensor exactly, leaving only 0.001177% relative L2. The first
material observed hidden-state discrepancy is **after block 0**. The following
values compare native F32 with the corresponding official BF16-widened state;
relative L2 uses the official state norm at each boundary, while RMSE keeps
the same elementwise scale across boundaries.

| Target-image boundary | Relative L2 gap | Absolute RMSE |
| --- | ---: | ---: |
| Before block 0 | 0.167342% | 0.001111 |
| After block 0 | 0.355676% | 0.025023 |
| After block 12 | 2.250187% | 0.139864 |
| After block 29 | 5.553507% | 0.315484 |
| After block 30 | 4.712782% | 0.389066 |
| After block 31 | 2.004610% | 0.442185 |

The relative percentage falls after block 29 because the official state RMS
grows from 5.681 at block 29 to 22.058 at block 31; the absolute error still
increases. The adjacent *error-vector* change is largest at blocks 30, 31,
and 29 (L2 480.12, 434.39, and 374.82 respectively), but these blocks
consume already-diverged inputs, so this does **not** rank their intrinsic
operator error. Rounding only each native block output to BF16 does not close
the gap: block-0 and block-29 relative L2 become 0.387136% and 5.555455%.
The terminal same-input DiT velocity gap is 1.255609% relative L2.

A separate single-variable intervention supplied the exact official
BF16-widened temporal features to the native forward. Its pre-block target
state was byte-identical to baseline, and its terminal velocity gap was
1.260615%, slightly *worse* than baseline; the squared velocity error rose
0.799%. Thus mismatched input Fourier timestep features are not the main
explanation for this call-10 velocity error. This is a bounded intervention,
not a general statement about timestep handling at other steps or prompts.

Together with the exact scheduler replay, these captures locate an observed
same-input divergence inside the DiT, beginning at or before its first block
and growing through the block stack; they reject a VAE-only origin for this
trajectory. They do **not** prove that the face defects are caused by a
particular block, that the final latent is outside the VAE's training support,
or that a Metal kernel is wrong. Q8 weight substitution and BF16-versus-F32
arithmetic remain confounded. The cached-prefix versus full-joint text route
was unresolved in this capture; the follow-on experiment below tests it at
call 10 and replays selected blocks from identical *consumed* inputs. The
target-only official capture in this paragraph cannot by itself provide the
full-joint predecessor states or modulation needed for that replay.

The scratch analyzer and report are
`/private/tmp/qwen21-call10-all32-20260928/analysis/analyze_call10_all32.py`
(SHA-256
`57f22877fa9a6ba1a284af04a5026cc1369f09fcca5ac28d77bbb675c7163246`)
and `call10_paired_analysis_rne_counts.json` (SHA-256
`3d07ec72a35f170a9fb355a4e80a78c532d931875d11402c7021c15ce706fc66`).
Its synthetic positive/BF16 tie-to-even and tampered-SHA negative tests
passed; the real run checked all 34 official and 33/34 native capture records.
The official, baseline, and time-intervention manifest SHA-256 values are
`957330fa0658bfe208b5919f6ac9356eeb7fbb92535a96e7d0ea79a5692abecb`,
`3ea0e25c4148bf293dda32e349c5058e0e52812b20798c271dc1917f05fa50ac`,
and `c02db824aaf432d09029ad32a8addedcdf1f7d6def79e3585d4aabff8c7ff033`.
Refresh after source/weights, conditioner, BF16 or Metal runtime, schedule,
saved artifacts, or diagnostic source changes; no end-to-end trajectory or
image-quality claim is promoted by this block trace.

### Call-10 full-joint route control and equal-input block replay (2026-09-28)

The official BF16/MPS DiT was rerun at zero-based call 10 without its prefix
cache, holding the saved official latent, conditioning, timestep, weights, and
output readback fixed. Its target velocity matched both the cached forward
and an unhooked noncached control **byte-for-byte** (65,536/65,536 BF16
values). A second, observational full-joint capture retained the 230 text
rows and 1,024 target-image rows before block 0 and after every block, plus
the exact BF16-widened time and modulation tensors. The target suffix matched
the earlier cached capture at **all 33 boundaries**: 138,412,032/138,412,032
BF16 values. Thus the official cached/full route is closed as an explanation
for the measured *target-state* gap at this call. This is not a claim that the
text-prefix states or other calls are identical. A direct offset-view-to-F32
MPS readback gave incorrect values; transferring the complete BF16 tensor to
CPU before slicing passed an offset-zero control and was used for every
admitted capture. The first hook-helper attempt produced no block output;
only the corrected, validated `retry1` capture is used here.

A scratch native full-Q8 Metal replay then fed each selected block its exact
official BF16 predecessor **full-joint** state, widened to F32, and the
official modulation. It did not feed a previous Q8 block's output to the
next selected block. The full-Q8 donor was SHA-checked and all 32 layers'
six projection families were asserted Q8_0. The block-0 repeat was bit-exact.
Root independently rehashed all six saved 20,545,536-byte native outputs and
recomputed the text, image, and joint metrics from the raw tensors. For image
rows, the matched-input local gap is much smaller than the gap in the ordinary
native full-stack forward:

| After block | Cumulative image rel-L2 | Equal-input local image rel-L2 | Prior-state/setup term rel-L2 | Equal-input local image RMSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.355676% | 0.343305% | 0.114539% | 0.024153 |
| 1 | 0.388854% | 0.226088% | 0.309607% | 0.017307 |
| 12 | 2.250187% | 0.431016% | 2.209737% | 0.026790 |
| 29 | 5.553507% | 0.491089% | 5.526691% | 0.027898 |
| 30 | 4.712782% | 0.488420% | 4.681694% | 0.040322 |
| 31 | 2.004610% | 0.373535% | 1.966848% | 0.082396 |

The third numeric column is the norm of the exact vector difference between
the ordinary native block output and the equal-input native block output,
divided by the same official output norm. The terms add as vectors, **not**
as the displayed L2 magnitudes. It includes preceding hidden-state drift
*and* the difference
between the native and injected-official modulation/setup routes; it is not
a pure input-Jacobian or causal attribution. At block 29, however, the
5.553507% cumulative gap versus 0.491089% local gap rules out treating that
block's equal-input discrepancy as the whole observed error. The direct
local image error remains below 0.5% relative L2 at all six selected blocks,
while its absolute RMSE rises to 0.082396 at block 31. Rounding only native
block outputs to BF16 does not close any of the six local gaps; their
relative-L2 values become 0.374467%, 0.268110%, 0.461605%, 0.516358%,
0.513701%, and 0.407863%, respectively.

This locates substantial within-stack accumulation by call 10 in addition to
the already measured between-diffusion-step feedback. It does **not** isolate
Q8 weight error from native F32-versus-official BF16 arithmetic, identify a
bad kernel, or prove which latent feature causes an eye defect. The same
pinned CPU/FP32 VAE still decodes different final latents differently;
neither this replay nor pixel differences measure whether a final latent is
outside the VAE training distribution. A useful next discriminator is a
matched-input *operator* tap at an early and a late block (Q/K/V, attention,
MLP, and residual/modulation boundaries), separating Q8 weight substitution
from arithmetic before any quality-policy or fusion change.

The official full-joint manifest SHA-256 is
`49370980cd2826b713b2f08fe5bcaeb2ca06ad1c76831e5dbab776addee6a58b`;
the native replay report SHA-256 is
`6ebcf6b90a8916fe9c467dcdc642a732ea59f7008fd4af9e0f553e9a9e6b3b65`.
The scratch runner/source and executed binary SHA-256 values are
`398eec0bea53504b49c6a4439613f85a2ac9a6cc3d50f7ed5773d3bea0627247`
and `6f94a271cef9b6ae17266a6dc6891f177926ea4fe8ac482e0bad6a97e4af10ab`.
Artifacts are under `/private/tmp/qwen21-call10-all32-20260928/official-full-joint/`
and `/private/tmp/qwen21-call10-matched-block-replay-20260928/`. A 60-second
quiet-host prelaunch guard exited 75 without launching the GPU; the one
successful correctness run exited zero under a 300-second, 32-GiB
process-tree guard and a 50% system-free-memory floor, with only the
benchmark-noise quiet requirement disabled. Refresh after changes to model
or conditioning payloads, Diffusers/native source, scheduler input, BF16 or
Metal semantics, or any saved scratch capture.

### Call-10 block-0 operator frontier on equal full-joint input (2026-09-28)

The first-block operator split used the same saved official BF16 input state
(widened exactly to F32 for native Metal), official modulation, full 230-row
text prefix and 1,024-row target image suffix, and the pinned full-Q8 donor.
The official noncached BF16/MPS forward retained its previous block-0 output
and terminal velocity **byte-for-byte** with passive hooks. The native Q8/Metal
block-0 output was bit-exact with and without its copy-only taps and on a
third repeat. The official and native reports have SHA-256 values
`cb2dce2d30fc140bdc66da036435d1e385e643c3d2768fea552ddea0274b883f`
and `4c2bc63ed8b73c9be08fc7faa1b08366d7077d79494becf0e0e161d2ebba4989`.
They are under `/private/tmp/qwen21-call10-op-tap-official-20260928/tap_run1/`
and `/private/tmp/qwen21-call10-op-tap-native-20260928/`, respectively.
These controls establish that the following comparisons observe the existing
routes rather than a changed attention processor or tap-dependent output.

| Matched block-0 boundary | Text relative L2 | Target-image relative L2 | Target-image RMSE |
| --- | ---: | ---: | ---: |
| LayerNorm-1 + modulation output / attention input | 0.312677% | 0.307104% | 0.002097 |
| Raw Q projection | 0.275226% | 0.245395% | 0.020098 |
| Raw K projection | 0.309767% | 0.257086% | 0.020021 |
| Raw V projection | 0.414948% | 0.377128% | 0.012031 |
| Attention output before projection | 0.924588% | 0.994056% | 0.001962 |
| Attention output after projection | 0.341333% | 0.199159% | 0.012997 |
| Modulated MLP input | 0.615365% | 0.690595% | 0.000193 |
| MLP output | 0.251810% | 0.381326% | 0.002104 |
| Block output | 0.255690% | 0.343305% | 0.024153 |

All percentages divide by the official tensor norm at their **own** boundary;
they cannot be added or compared as causal contributions. The earliest matched
nonzero output in this equal-input block is the fused native LayerNorm-1 plus
modulation result, before Q/K/V projections. A CPU F32 recomputation of
LayerNorm-1 plus scale from the exact teacher block input and modulation
matches the native Metal output within **0.00000664%** relative L2 on target
rows. The official modulated output is reconstructed **exactly at all
4,194,304 target elements** from its observed BF16 LayerNorm output by BF16
round-to-nearest-even of `1 + scale`, then BF16 rounding of the product.
Starting from the CPU/native F32 formula, successive hybrids using the
observed official LayerNorm output, then BF16-rounded scale, then BF16-rounded
product have target relative-L2 gaps to the official output of 0.307104%,
0.244607%, 0.165229%, and 0%. These are *non-additive vector magnitudes*;
the official MPS LayerNorm output also differs from an independent CPU BF16
LayerNorm at about 0.197395% relative L2. This localizes the first discrepancy
to precision/backend staging at normalization and modulation, not to a
demonstrated error in the native fused kernel formula.

The native post-RoPE Q/K taps are **not** compared with the official pre-RoPE
Q/K taps; those are different boundaries. The attention and MLP differences
include inherited upstream error, Q8 weight substitution, and BF16/F32
arithmetic. In particular, the 0.994056% attention-output ratio has a small
0.001962 absolute RMSE and does not by itself establish attention as the main
cause of the final latent or face defect. This is one zero-based diffusion
call (10), one block, one pinned prompt/seed and 512px trajectory. The
equal-input, no-op-controlled block intervention is reported below; the
terminal-velocity and decoded-image consequences remain open. Recheck after
changing the donor, conditioning, Diffusers/native source, BF16/Metal runtime,
or any saved tap artifact.

### Call-10 block-0 modulated-input intervention (2026-09-28)

A scratch-only full-Q8 Metal replay now performs that first causal test at the
same official call-10 full-joint input. Immediately after the native fused
LayerNorm-1/modulation kernel and before Q/K/V, it copies the **exact official
BF16-to-F32-widened** 1,254-by-4,096 modulated attention input into the native
attention-input buffer. It does not replace the native gate-1 buffer, Q8
weights, attention/MLP operators, or any later block. The source was limited
to a one-block replay and did not change repository production code.

The donor, 33 teacher states, official injection file, and pre-block state
passed shape, finite-value, byte-count, and SHA-256 preflight. The current
native baseline reproduced the previously pinned block output SHA-256
`cb6279ab6576458c7d67727bafc809d784f50dc1b966c1587b6c799b2c8683b2`;
injecting its own saved input was bit-exact to that
baseline; the official injected attention-input tap matched the teacher F32
file SHA-256
`3c5e57474287de70a1732c7a9567db01bde28049621a56cd5762cfec0f881222`
exactly;
and the injected block output repeated bit-for-bit. All 14 injected operator
taps were finite. The initial sandboxed attempt could not create a Metal
device and produced no block result; the separately pinned device-enabled
retry completed on the M2 Max.

| Matched target-image boundary | Native baseline relative L2 | Official-input injection relative L2 |
| --- | ---: | ---: |
| Raw Q, before RoPE | 0.245395% | 0.217448% |
| Raw K, before RoPE | 0.257086% | 0.224119% |
| Raw V | 0.377128% | 0.333655% |
| Attention output before projection | 0.994056% | 0.967274% |
| Attention output after projection | 0.199159% | 0.187117% |
| Modulated MLP input | 0.690595% | 0.671047% |
| MLP output | 0.381326% | 0.380913% |
| Block-0 output | 0.343305% | 0.312606% |

Root independently recomputed the table from SHA-checked raw arrays. The
target-image block-output RMSE fell from 0.024153 to 0.021993, or **17.085%
less squared error** at this boundary; the text-prefix output changed much
less (0.255690% to 0.255090% relative L2). Therefore the modulated attention
input contributes causally to the local image-row block-0 discrepancy, but
does not account for the remaining 0.312606% output gap. The smaller
downstream MLP-output response does not uniquely assign that residual to
weights, gate-1, MLP, or backend arithmetic. Percentages at different
boundaries have different denominators and are not additive causal shares.
No full-DiT velocity, accumulated latent, VAE-support, eye-quality, or final
image improvement follows from this one-block replay. The no-op-controlled
full-DiT call-10 velocity replay with this *same* block-0 intervention follows
below; a production precision change still requires a paired trajectory and
decoded-image A/B.

The completed retry report SHA-256 is
`3c435b7fe71b39dc9ac133f100312bfa111632fdb8debfd6393f6916c258f7e0`;
its CPU preflight and runner SHA-256 values are
`38585b076f544975cad62fb26fe77125b0eefbb5ac494ba1fd100a5e945c482b`
and `01e4e8ea66acb6b7350ca7c9e68db49f0dbb04aae783e3260610fedf17000fc4`.
All files are under
`/private/tmp/qwen21-call10-modulation-injection-20260928/retry1/`.
Refresh after changes to the donor, conditioning, official/native captures,
Diffusers/native source, BF16/Metal runtime, or scratch runner.

### Call-10 block-0 intervention through the DiT suffix (2026-09-28)

A scratch-only full-Q8 Metal A/B continued the two saved block-0 outputs through
the same 31 later blocks and final head. Both arms used the same official
call-10 modulation projection, token layout, and final scales reconstructed
once from the exact-widened official `temb` and native BF16 `norm_out.linear`.
The official final-scale tensor was not captured, so error versus the official
velocity is **not** an exact end-to-end parity measurement. The block-0 seeds
were generated by the earlier retry1 instrumentation overlay; the suffix
itself compiled from the SHA-pinned repository-only source at
`49fac7c92fb45dc5ec4ea9e9a0569b2e9029c601`. The seed lineage is not
retroactively a stock-repository run. A separately loaded baseline repeat
produced bit-exact full-joint and target-velocity files before the single
injected arm ran. The injected arm was not repeated or order-counterbalanced.

| Target-image result, 65,536 values | Baseline A | Injected B |
| --- | ---: | ---: |
| Call-10 velocity relative L2 to official BF16 | 1.004713% | 1.001539% |
| Call-10 velocity absolute RMSE | 0.01410030 | 0.01405576 |
| Next BF16 latent relative L2 to saved official step | 0.108470% | 0.108480% |
| Next BF16 latent absolute RMSE | 0.000904603 | 0.000904686 |

Root and a separate read-only agent independently recalculated these values
from raw output files. The velocity squared error decreased by **0.630735%**
(SSE `13.0297676` to `12.9475843`), much less than the local block-0
improvement of 17.085%. The velocity correction was only 4.570% of the
baseline velocity-error norm and had cosine 0.0919 with the direction toward
the official velocity. For the next-state comparison, the pinned official
BF16 input and velocity with `dt = -0.018813252449035645`, BF16-RNE velocity,
BF16-RNE product, F32 addition, and BF16-RNE state reproduced the saved
official next latent **65,536/65,536** values exactly. Applying this same
scheduler arithmetic to A and B reversed the tiny ranking: injected next-state
squared error increased by **0.018343%** (SSE `0.053628542` to
`0.053638379`). A and B differed at 512 BF16 next-state values. B had 22
fewer unequal values than A against the official state, but the magnitude-
weighted error was slightly worse. Thus a small velocity proxy gain does not
survive as a next-latent L2 gain on this fixed call-10 fixture.

This rejects promoting the isolated modulated-input precision change from this
experiment. It does not show that the normalization difference is irrelevant
at other calls, or determine eye quality, full-trajectory behavior, or VAE
training-support membership. The report's inherited `scope` string still says
"CPU preflight" despite its `status=complete`; the GPU result is anchored by
the A/A bit-exact output hashes, raw A/B arrays, and independent recalculation,
not that stale string. The retry4 run report, CPU preflight, and runner SHA-256
values are respectively
`dc61f028c69720b11ba6472deda8f4dfa2e02d907946d67932e202156225b39e`,
`2946e6f099cdf45f298f793ac026242c142abae07a940a6807300ce76015b09d`,
and `1ecd01c4f5e319aa61091237f27a7c0b152610e31f5bc7d85539948b54598662`.
Raw files are under
`/private/tmp/qwen21-call10-modulation-suffix-20260928/retry4/` and may expire.
Next discriminate larger downstream error sources with equal-input projection-
family swaps, then require a matched full trajectory and decoded-image A/B
before changing production precision. Refresh this evidence after donor,
conditioning, official captures, source, BF16 scheduler/Metal runtime, or
scratch-runner changes.

### Call-10 upstream-state rescue and exact-input output-head probe (2026-09-28)

The prior block-0 intervention did not improve the next BF16 latent. A new
scratch-only, SHA-gated Metal experiment instead asks whether the terminal
call-10 discrepancy is already carried in the 31-block input to the final
block. It uses the same official full-joint pre-block-0 BF16 state, saved
official conditioning, Q8 donor, token layout, modulation, and *captured exact
official MPS BF16* final norm scale rows for every arm. The reference Euler
calculation matches all 65,536 saved official next-latent BF16 values. Arm A
runs native Q8 blocks 0–31; A/A repeats that route bit-for-bit; the rescue arm
replaces only the input to native block 31 with the official post-block-30
BF16 state, then uses the same native block 31 and output head. This is a
controlled state-boundary substitution, not a candidate runtime fix.

| Target-image call-10 boundary | A: native blocks 0–31 | Official post-30 state + native block 31 |
| --- | ---: | ---: |
| Velocity relative L2 to official | 1.004713% | 0.308901% |
| Velocity RMSE | 0.01410030 | 0.00433517 |
| Next BF16 latent relative L2 to official | 0.108470% | 0.056747% |
| Next BF16 latent RMSE | 0.000904603 | 0.000473251 |

An independent recount checked the report's raw output hashes and error ratios.
The rescue removes 90.547% of A's velocity squared error and 72.630% of its
next-state squared error on this fixed input. The A/A velocity and next-state
files are byte-identical. This strongly suggests that a substantial sampled
terminal error is already transported in the DiT hidden state *before* block
31; the observed latent gap itself exists before VAE decoding and is not
created solely by the final output projection. This does not
allocate error to particular preceding blocks or prove the rescue improves a
40-step trajectory or decoded face. The upstream 0–30 run and isolated
block-31 run have different call topology; a native-state split/no-op control
is still desirable before a stronger causal share claim.

An independent scratch Metal probe feeds the **exact official MPS BF16
pre-`proj_out` tensor**, widened without numerical change to the native F32
API, through the native BF16 `proj_out` weights twice. The repeats match
bit-for-bit. Against the official BF16 terminal velocity, 65,513 of 65,536
target values match after BF16-RNE; the target relative L2 gap is 0.006001%
(23 unequal values). Thus the output projection has a small nonzero
exact-input backend gap on this fixture, not a demonstrated 0.309% residual
by itself. The final norm is isolated below; block 31 still needs a native
split/no-op control for stronger attribution.
The head-only comparison does not bound the head's behavior under a drifted
input.

A CPU-only final-norm audit used exact official post-block-31 BF16 hidden
state and the exact selected BF16 per-token scales. On target tokens, the
native-style F32 fused LayerNorm/scale formula rounded only at output differs
from the official BF16 `pre_proj_out` tensor by 0.309656% relative L2;
rounding at the official BF16 LayerNorm and `1 + scale` stage boundaries
reduces that comparison to 0.182975%. These percentages are for the 4,194,304
pre-projection hidden values, **not** the 65,536 terminal velocity values.
CPU LayerNorm reductions need not match MPS or Metal bit-for-bit. This makes
final-norm staging a plausible remaining source. On the exact official
pre-head input, the native projection changes only one of the 65,536 BF16
next-latent values after the pinned Euler step. As a positive control, a
separate MPS BF16 replay of the pinned
`LayerNorm(hidden, eps=1e-6) * (1 + selected_scale)` from the exact official
post-block-31 state reproduces the official `pre_proj_out` capture **all
5,136,384 BF16 values byte-for-byte** (including all 4,194,304 target-row
values). Thus the selected scale/layout and captured official boundary are
consistent; the CPU BF16 residual is a backend reduction difference, not a
teacher-capture discrepancy. The standalone MPS replay manifest and script
SHA-256 values are
`1546e28c94b8755f9b501c5206a4d0818194b3b047ce019fba2e31e835896007`
and `cbe28130d033f7fd81784d3aa6aa3895e544e9fe1586c0c8f6aec094c3b44bba`.

The two CPU-derived final-norm candidates were then fed, without rerunning
any transformer block, to the same native Metal BF16 `proj_out` twice per
arm; each A/A pair was bit-exact. Against official call-10 BF16 velocity,
the F32-unfused candidate has 0.223499% relative L2 (RMSE 0.00313662;
21,845/65,536 unequal BF16 values), while BF16-staged has 0.191439%
(RMSE 0.00268669; 16,006 unequal). After the *exact* saved BF16 Euler step,
their next-latent gaps are 0.038683% (1,409 unequal values) and 0.035144%
(938 unequal), respectively; the official velocity reconstructs the saved
next latent byte-for-byte. These candidates are closer than the rescued
native-block-31 result (0.308901% velocity, 0.056747% next latent), but do
not isolate a production fix: CPU LayerNorm reduction differs from the
byte-exact MPS replay, and neither candidate runs the actual native Metal
final-norm kernel. The exact official pre-head input remains the relevant
projection-only floor on this fixture. The candidate-probe manifest and
runner SHA-256 values are
`20f5a6c2bce117daf35f532ab7c11179b8adc6458d9978c0c7a6f3d05bb9cece`
and `157ccc5cacfd0e46c1df90d28a76e31691d6be83031922e52dbc12d740eb1b84`.

At this stage the saved native full-joint post-block-30 state was absent, so
the upstream rescue lacked an exact native split/no-op topology control;
the follow-up below closes that confound on the pinned call. A separate
exact-input native tap was run with the **production
`qi21_final_layernorm_scale` kernel** and launch geometry on exact official
post-block-31 BF16 hidden state and exact selected BF16 scale, both widened
to F32. It reruns neither DiT blocks nor VAE. Two kernel repeats and two
native BF16 `proj_out` repeats are byte-identical. The native pre-projection
F32 result is within 0.00000655% relative L2 of the independent CPU F32
fused calculation, but differs from the official pre-projection BF16-widened
target tensor by 0.270371% relative L2 (0.309656% after BF16-RNE). On that
exact official post-block-31 state, the native norm plus native projection
produces a BF16 terminal velocity gap of **0.223499%** (21,846 unequal
values), and its exact BF16 Euler next latent differs by **0.038683%**
(1,409 unequal values). By contrast, changing only the pre-projection input
to exact official BF16 values while keeping the same native projection gives
0.006001% velocity and 0.003659% next-latent gaps (23 and one unequal
values). The BF16 velocity difference *between those two native-head inputs*
is 0.223467% relative L2, so final-norm arithmetic/staging is a measured
same-state head-parity source, not merely a CPU hypothesis. It does not prove
that official BF16 staging improves decoded quality: no native precision
change or full denoising trajectory was tested. The scratch tap separates
norm and projection with a readback between command buffers, rather than
reproducing the fused production command buffer. Its runner and manifest
SHA-256 values are
`8a7de366da3faee2e4b5f9c03988e0351634ba47743f3567384d088fa7ce1585`
and `e973085a2879c8407a7c929804c3a04b863c85c1a5d98f7ba220c0b331811677`.

An independent SHA-checked spatial recount mapped the 32x32x64 row-major
latent to the 512x512 official decode. The two approximate face boxes cover
45/1,024 tokens (4.395%). They contain 4.554% of A's next-latent squared
error and 4.697% of the rescue's net squared-error reduction; 41 of the 45
tokens improve. The per-token reduction is only 1.072 times the non-face
rate, and the face-box result ranks at the 61.1 percentile among 392,721
same-shape disjoint non-face control-box pairs. The ten largest A next-state
token errors all lie outside those face boxes. This one-step rescue therefore
does not show persuasive face-specific latent concentration and cannot prove
the cause of visible eye defects. The selected boxes are approximate and
do not segment actual eyes; a multi-call segmentation-controlled audit would
be needed to revisit that question.

The rescue report SHA-256 is
`c486b913d53239fc374873b8cfcd92c0ba12453416e0c978562ed8ba0cab3d50`;
the head-only manifest and runner SHA-256 values are
`47a89ae2ce3811515f8ce46c2db41bc8065d3f04b218ba9a5cd59ef789748e92`
and `8b0530ddf12b87d1da275f3397dabc7b8bb9b2f751f7a3a8e661ca71c646693b`.
Scratch inputs and outputs are under
`/private/tmp/qwen21-call10-boundary-rescue-20260928/` and
`/private/tmp/qwen21-call10-head-tap-20260928/`. They may expire. Recheck
after model, source, official capture, BF16/Metal, scheduler, or scratch-runner
changes.

### Call-10 split-topology control and post-block-30 text/image state split (2026-09-28)

A scratch-only, SHA-gated full-Q8 Metal replay closed the previous rescue's
call-topology confound on the same fixed official call-10 latent, full-joint
pre-block state, conditioning, modulation, and captured BF16 final scales.
It first ran native blocks 0–31 plus the head twice, then ran blocks 0–30
without the head, serialized their full 1,254-by-4,096 Float32 hidden state,
reloaded it bit-exactly, and ran block 31 plus the same head. The two
unsplit outputs, the split output, and the preceding rescue baseline all
have the identical target-velocity SHA-256
`0344ba596ca42d40c22183e2ac8d7ed645ba5ab05c56e2bcb8ee11843b68a51e`.
The official-post-30 substitution also reproduced the preceding rescue
velocity SHA-256
`6603936b9371485e2529188db6f6eadd48737234833e0b15b83c9710747e64d4`.
The native post-30 checkpoint SHA-256 is
`6a6e8aab42e8884885c1dc2d8375a99bf16084fdcdd16d291e2e5f26c6b0fc10`.
Thus splitting and host save/reload do not account for the observed rescue
on this pinned fixture.

With that no-op established, a 2-by-2 intervention supplied native or
official post-block-30 rows separately for the 230 text tokens and 1,024
target-image tokens. Every arm ran the same native Q8 block 31 and head;
the shared official BF16 Euler update generated the next-state comparisons.

| Post-30 text rows | Post-30 image rows | Velocity rel-L2 to official | Next BF16 latent rel-L2 to official |
| --- | --- | ---: | ---: |
| Native | Native | 1.004713% | 0.108470% |
| Official | Native | 1.002718% | 0.108322% |
| Native | Official | 0.310462% | 0.056765% |
| Official | Official | 0.308901% | 0.056747% |

Root independently rehashed all six velocity and six next-latent files and
recomputed each relative L2 from the raw arrays against the captured official
BF16 velocity and saved next latent. Replacing only the image rows removes
90.452% of baseline velocity squared error on this one call, nearly the
90.547% removed by replacing both row groups; replacing only text rows
removes 0.397%. These are **conditional intervention results**, not additive
causal shares. The image-row state gap after block 30 is 4.442% relative to
the official image rows, versus 1.573% for text rows; the denominators differ.
The experiment establishes that most of this measured terminal discrepancy
is transported through the target-image hidden state at the post-30 boundary,
not an artifact of splitting the native call or a dominant text-row effect
at that boundary. It does not identify which earlier block, weight family,
or arithmetic boundary created the image-state drift, nor establish an
eye-specific, VAE-support, full-trajectory, or decoded-quality benefit.

The completed Metal report and scratch runner are
`/private/tmp/qwen21-call10-split-control-20260928/main_results_retry2.json`
(SHA-256
`63b87ad72ae0257cf27d5fc3d8e95f03672f4d413da6f75c166756a6b37be4de`)
and `split_control.cr` (SHA-256
`c67d75a32d0bf908e198adf24be33718d7f4ed84ca74b14f8efb4978b7178d82`).
CPU preflight checked the pinned source/kernel/bridge, donor, official
states and BF16 widenings, conditioning, scales, and prior rescue outputs.
The default-sandbox attempt stopped before model output because it could
not create a Metal device; the unchanged bounded retry on Apple M2 Max
completed. Scratch artifacts may expire. Refresh this result after source,
compiler/device/runtime, donor, conditioning, scheduler, saved input/tap,
or diagnostic-runner changes. A next discriminating probe should inject
official image rows at selected earlier block boundaries under a matched
split/no-op control, then require a full 40-step decoded A/B before a
production precision change.

### Call-10 image-state drift across selected DiT boundaries (2026-09-28)

A follow-up scratch-only full-Q8 Metal replay used the same official
full-joint pre-block-0 input, conditioning, modulation, final scales, and
call-10 teacher output as the split control above. The official MPS capture
contains full 230-text/1,024-image post-block states at all 32 boundaries;
this bounded probe selected blocks 0, 12, and 29. For each, the native prefix
produced a Float32 checkpoint that survived a bit-exact host round trip.
The native suffix from the unchanged checkpoint reproduced the unsplit
baseline velocity and next latent *byte-for-byte* at all three boundaries;
the unsplit A/A was also exact. Only then did the treatment replace the
target-image rows with the exact BF16-widened official state, leave native
text rows in place, and run the same native suffix and head.

| Image-row source at boundary | Velocity rel-L2 to official | Next BF16 latent rel-L2 | Baseline velocity squared error removed |
| --- | ---: | ---: | ---: |
| Native throughout | 1.004713% | 0.108470% | 0% |
| Official after block 0 | 0.942125% | 0.104182% | 12.071% |
| Official after block 12 | 0.666389% | 0.084176% | 56.008% |
| Official after block 29 | 0.324285% | 0.058132% | 89.582% |
| Official after block 30 (preceding control) | 0.310462% | 0.056765% | 90.452% |

The native-versus-official image-state relative L2 at the three newly
sampled boundaries was 0.343305%, 2.227963%, and 5.325171%, respectively;
these compare states produced by this matched full-joint replay. Root
independently rehashed and recomputed every saved velocity/next-latent arm,
the three native checkpoints, and the text/image state gaps from raw arrays;
all matched the report, and every native split no-op was byte-exact.
The first-block image-state discrepancy is measurable, but replacing it
alone recovers little of the terminal velocity error. A later replacement
has much more leverage because it removes the *accumulated* difference by
that boundary. It cannot distinguish new error created by blocks 1–29 from
amplification or transformation of earlier errors, nor assign a faulty
block, Q8 weight family, Metal kernel, or BF16/F32 boundary. The three
selected points do not imply monotonic behavior at unsampled boundaries.
No 40-step or decoded-eye improvement was tested.

The bounded run took 246.83 seconds on Apple M2 Max after the default
sandbox stopped before model output at Metal device creation. The completed
report, preflight, and runner are under
`/private/tmp/qwen21-call10-drift-source-20260928/`, with SHA-256 values
`e444672b16962f9088786fc9d10d335eaf1a8669737105a73a310e9dced7f886`,
`cfdb9366a46d4d0771a826ba79236f4f19523d9e60c606adb62a282f7e5604ce`,
and `1a49d430250200d9497bf9b1d13e2800e95ba087712850b707f935f932cbf328`.
The report records exact source/kernel/bridge/model/input/tap pins and the
launch command. Scratch may expire; refresh after source, compiler/device,
Q8 donor, official capture, conditioning, BF16 scheduler, or runner changes.
A useful next discriminator is an equal-input operator-level replay in an
early and a middle block that separately changes Q8 weights and precision
staging, followed by a complete trajectory and decoded A/B before
production promotion.

### Fixed-VAE face-region latent counterfactual (2026-09-28)

To test whether the matched final-latent difference actually drives the
visible face difference, a CPU/Float32 replay held the pinned Qwen-Image 2.1
VAE, its normalization, and decoding code fixed. The two endpoints were the
same official and full-Q8 40-step latents and independently saved decoder
images used above. The 512x512 face pixel ROI was `[320,150,380,219)`;
its coarse 32x32 latent-grid cover was `x=[20,24), y=[9,14)` (20 tokens).
A one-token halo was `x=[19,25), y=[8,15)` (42 tokens). Each treatment
started with official latents and copied full-Q8 values either inside or
outside one mask. The official/full-Q8 decoder endpoints were pixel-exact
against saved oracles, and all-off/all-on latent-mask controls were exact.

| Full-Q8 final-latent residual supplied to the fixed VAE | Face RGB RMSE versus official (0–255) |
| --- | ---: |
| All tokens | 6.391382 |
| Face core only (20 tokens) | 6.262167 |
| Outside face core only | 1.837282 |
| Face halo only (42 tokens) | 6.349274 |
| Outside face halo only | 0.624513 |

The core and halo contain just 0.272560% and 0.782783%, respectively, of
the *global squared final-latent residual*, yet their separate transplants
produce nearly the full face-ROI pixel RMSE. This establishes that spatially
local final-latent drift is sufficient to produce most of this measured face
difference through an unchanged VAE on this single fixture. It does **not**
prove that the VAE is off its training manifold, that it is unusually
sensitive, that the eyes specifically account for the ROI metric, or that
these artificial hybrid latents are valid diffusion trajectories. RMSE arms
are not additive through the nonlinear decoder. The mask is a coarse
pixel-to-latent mapping, not a receptive-field proof; one prompt/seed and
one ROI cannot establish general image-quality behavior.

The already-pinned full-Q8 trajectory report shows that the same core's
latent-space residual begins after the **first** DiT/Euler step, rather than
appearing abruptly at decode: its local relative L2 to the same-step
official latent is 0.108206% at step 1, 0.394316% at step 10, 1.052098% at
step 20, and 3.810557% at step 40. Both this local relative L2 and the
local absolute RMSE increase at every saved step (checked across all 40
entries). That locates onset and accumulation in the generated latent
trajectory; it still does not assign the drift to a particular DiT operator,
nor turn intermediate latent distance into an eye-quality score.

The completed report and six decoded PNGs are under
`/private/tmp/qwen21-vae-face-attribution-20260928/`. Report SHA-256:
`8ae6fe2f355b658618f1ca3290c6502c4cefb624576bb999c97f5851b514d5af`;
runner SHA-256:
`96d29321fc76bfa8c6fb93d531f69f48d10c7c6ef84b35aba0e7c9da1b66b48c`.
The report pins latent manifests/payloads, VAE config/weights, Diffusers
source commit `8b3c707ebd3ec4881f4190cf42931da07eaf3b65`, exact oracle
PNGs, and CPU/Float32 runtime. Root independently recomputed the face RGB
metrics from raw PNGs, checked output hashes, mask coordinates/complements,
and latent residual-energy fractions. Scratch may expire; refresh after
latent trajectory, decoder/model/source/runtime, ROI/mask, or metric changes.
The next causal question is which DiT operator/precision boundary introduces
the local final-latent error, followed by a matched full-trajectory decoded
intervention before any production change.

### Call-10 face-token DiT velocity versus BF16 Euler (2026-09-28)

A CPU-only saved-artifact replay narrowed one generated-latent boundary
without another GPU run. At zero-based call 10, both paths start from the
*same official* BF16 input latent; the native full-Q8 DiT velocity is
compared with the saved official BF16 velocity before either is consumed by
the pinned Euler/BF16 scheduler. The 20-token face core is
`x=[20,24), y=[9,14)` (1,280 channel values). Official 40-step snapshots
`step-009` and `step-010` exactly match the separately saved call-10 input
and teacher next latent. Native A/A velocity and next-state controls, and
the native post-block-30 split/no-op, are byte-exact.

| Same official call-10 input | Face-core relative L2 to official | Global relative L2 to official |
| --- | ---: | ---: |
| DiT velocity before Euler, native F32 vs official BF16-widened | 1.447204% | 1.004713% |
| Next latent after pinned Euler and BF16 staging | 0.111110% | 0.108470% |

The face core holds 3.982457% of the global squared *velocity* residual at
this call, compared with 1.953125% of the latent-grid tokens. The saved
next-latent face core has 236/1,280 BF16 mismatches. Root independently
recomputed both rows of metrics from the raw files and separately replayed
BF16 rounding of velocity, timestep product, and next state: the predicted
official and native next-latent bytes matched their saved outputs exactly
at all 65,536 values, including hashes
`7287f0254b1fd3e95ddcd947c332e4d931aae936c775015e370a7049dacbc058`
and `2d9c23fc05e9960d4d937efda72b622b207cfe657ed69e74adce1724f2865594`.
Using an F32 timestep product and only final BF16 rounding misses 2,650
official and 2,899 native values; BF16-product staging is needed to
reproduce these saved outputs. The generic pinned Diffusers Euler expression
does not independently certify the MPS intermediate dtype, and this exact
replay does not establish a unique internal implementation.
Thus the local next-latent drift is already present in the DiT velocity on
this exact same-input call; an additional scheduler or serialization error
is not required to explain it. This is not the actual full-Q8 step-11
trajectory state, whose call-10 input already contains preceding drift, and
it does not identify an operator or quantify decoded eye quality.

The pinned CPU analysis and report are under
`/private/tmp/qwen21-face-step-boundary-20260928/`, runner SHA-256
`b9bee62a2e146e0540b6e3b7a157adf7b1ff2a51d230312ad18d90c51a3632a6`,
report SHA-256
`4cc9137d1be3a91ed16dae03e47c46d0f36d1ece44d620ab80623151c0c7640c`.
The report records file and source hashes, BF16/Euler semantics, the exact
face-token mask, and the saved call-10/trajectory alignment. Scratch may
expire; refresh after model/conditioning/scheduler/code/BF16 semantics,
saved artifact, or ROI changes. The next discriminator is the pinned
equal-input DiT operator route, followed by a full-trajectory intervention
before any quality claim.

### Call-10 block-29 gate/up fixed-input discriminator (2026-09-28)

An instrumented, no-op-checked native replay began block 29 at the exact
official post-block-28 call-10 state widened from BF16 to F32. It captured
the native block-29 MLP input after native attention, residual,
normalization, and modulation, and the production Metal Q8_0 gate/up
projection output. This MLP input is **not** an official MPS tap. The
instrumented and uninstrumented block outputs were byte-identical.

The exact official BF16 gate and up weight payloads re-encoded byte-for-byte
to the fused Q8_0 donor tensor, including each half separately. On the same
frozen native MLP input, full 1,254-row single-thread F32 GEMMs compared
(a) official BF16 weights widened to F32, (b) the donor Q8_0 weights
dequantized to F32, and (c) the captured production Metal Q8 output.
Only after each full GEMM were image and face rows selected for metrics.

| Gate/up comparison | Joint relative L2 | Face-20 relative L2 |
| --- | ---: | ---: |
| Q8-dequant F32 versus original BF16-weight F32 | 0.363279% | 0.337199% |
| Production Metal Q8 versus Q8-dequant F32 | 0.000026527% | 0.000024900% |

Root independently recomputed both face-row comparisons from the saved
native tap and original/Q8 weights. BF16-rounding the captured input before
the original-weight GEMM changes the face output by 0.097090% relative L2;
BF16-rounding the production output changes it by 0.165611%. These are
separate controls, not additive error terms. The official MPS BF16 gate/up
accumulation was **not** captured, so the original-weight F32 arm is not an
official operator-output oracle. The separate full block-29 image-output
gap versus official MPS was 0.491089% relative L2; this operator-only test
does not assign that gap, the final velocity, the 40-step trajectory, or
decoded face quality to gate/up quantization. A no-op-guarded gate/up
intervention tested its effect at the block output.

The no-op-guarded scratch-only splice then replaced **only** this gate/up
projection output with the full 1,254-row original-BF16-weight F32 GEMM
result on the same native MLP input; the rest of block 29 stayed native.
The unspliced baseline reproduced the preceding replay, the live input
and Q8 gate/up taps were identical in every arm, and writing back the live
native gate/up result was byte-exact at the block output. Relative L2 to
the official MPS block-29 output changed from 0.491089% to 0.474352%
across image rows and from 0.467986% to 0.451475% in the face core.
This removed 6.700072% and 6.931958% of the respective *squared block-output
errors* (3.119477% over all joint rows). Root independently recomputed
all three fractions and the no-op from raw saved states. Thus a selective
gate/up weight/precision correction contributes to the local block-output gap
in this fixture, but most of that gap remains. The substitute was a CPU
F32 GEMM using original BF16 payloads, **not** an official MPS BF16 gate/up
capture. No terminal DiT velocity, scheduler step, complete 40-step
trajectory, decoded eye, or VAE-manifold effect was measured.

The fixed-input report, splice report, analysis/emitter script, and pinned
CPU reference are under `/private/tmp/qwen21-dit-operator-cause-20260928/`.
Fixed-input report SHA-256:
`e17521e5c51ddab47f2a533339453d0c60759ac331a9c4af937904c3d82f8e19`;
splice report SHA-256:
`d68fc63eaba7df8cf54b424b85c64e3eecd6bf69787ded058d8e1fff92019ab2`;
current analysis/emitter script SHA-256:
`ab0956a4bb0346a3f2ff141a8e7f9d5b4e50979922b4192808e39c9b2123378a`;
CPU reference SHA-256:
`85cc7b4af41e99755c063a891b18c76624865276e77f2510b401ed634d7eed47`.
The reports separately hash native taps, runner/source, model payloads,
official teacher captures, and saved block outputs. Scratch may expire;
refresh after model payload, quantizer, source/kernel, device,
call-10 official capture, or MLP input changes. The next discriminator
must test whether a selective higher-precision path also lowers final
velocity, next latent, and decoded face error on matched full trajectories
before a production precision/performance change.

### Call-10 block-29 official operator taps and MLP-input boundary (2026-09-28)

An isolated official MPS/BF16 block-29 replay loaded only its nine weight
tensors (436,208,128 bytes), the saved exact post-block-28 joint state and
modulation, and the pinned Diffusers RoPE/mask/segment route. The explicit
all-true joint key-valid mask was retained; replacing it with `None` would
select a different attention route. With no hooks, the entire
`[1,1254,4096]` BF16 block output matched the saved official call-10
post-block-29 tensor byte-for-byte (SHA-256
`7df76603d10ecab8cc1b5ef99ab8bf945d40c62056365c0dd1a5be2306109582`).
A second replay with 23 passive hooks produced the same exact output. Root
independently compared both raw outputs with the saved teacher. The hooked
report verified tap shapes, BF16/MPS provenance, finiteness, and raw hashes.
The run did not load the whole transformer or execute a diffusion trajectory.

The official hook on `img_mlp` captures the modulated norm-2 output just
before gate/up. The native operator tap captures the resident Metal `norm2_buf`
at the same boundary, with both blocks seeded from the same exact official
post-block-28 state. Comparing the native F32 values to the official BF16
values widened to F32 yields:

| Matched block-29 MLP input | Joint | Image rows | Face-20 rows |
| --- | ---: | ---: | ---: |
| Relative L2, native F32 versus official BF16-widened | 0.841881% | 0.914917% | 0.810215% |
| Relative L2 after rounding native F32 to BF16 RNE | 0.858045% | 0.929769% | 0.826535% |

Root and a separate CPU comparator independently reproduced the face values
from raw taps; the comparator reproduced the joint and image values too.
The face-only native rounding term is 0.165261% relative L2 against the
unrounded native input, so a final BF16 cast at this boundary does not erase
the discrepancy. This establishes a same-input drift **before** gate/up in
this one block. It does not distinguish Q8 attention projections, Q/K norm,
RoPE/attention, residual, LayerNorm/modulation, or BF16 staging inside that
prefix, and it does not assign a fraction of the final block-output error to
the prefix. The earlier gate/up splice remains a separate, bounded causal
result. The official gate/up taps and native production gate/up tap are on
*different* MLP inputs and must not be interpreted as a fixed-input
quantization comparison.

The no-hook report SHA-256 is
`d4c7384b40f9e5aa0385e64205758feda03ed1c99833cccea07cc05570073212`;
the passive-tap report SHA-256 is
`8187b131543bd80d111153d92e9774e3609e2d6c825e4370e904145db0414ce1`;
the pinned native MLP-input tap SHA-256 is
`83f55b3704fccd33f3d0cf632a41ad7668732381f90a1c564b542da004822f2a`;
the official MLP-input tap SHA-256 is
`6e9e9611febf73ee822e26b3f2b0cdbd56d959cb406a4c36df31d91e272c59e4`.
The current dual-mode isolated-block harness and CPU comparator are under
`/private/tmp/qwen21-dit-operator-cause-20260928/`; the passive runner
SHA-256 is `dc5bdaaae2a5a177658e3018dc964f96caf8051e35cd9324b4afad2db7ad56fd`
and the CPU comparator SHA-256 is
`b559068e98be8b8e843b22393e1d92e5b8915fb531219a8d26585109a6bb29e8`.
The original no-hook runner version was subsequently changed to add the
passive mode, so its recorded hash is retained in its report but that exact
source file is not currently preserved; the no-hook raw output and
byte-exact teacher comparison remain independently verifiable. Scratch may
expire; refresh after model/weights, Diffusers/Hub/PyTorch/MPS route,
masks/RoPE, native source, input fixture, or tap boundaries change. The
no-op-guarded MLP-input transplant is reported below; the existing
gate/up-only splice could not perform that cut without a new instrumentation
stage.

### Call-10 block-29 MLP-input causal splice (2026-09-28)

A scratch-only resident Metal replay seeded block 29 with the same exact
official call-10 post-block-28 BF16 state widened to F32 and the same
modulation in every arm. It froze the native `norm2_buf` after the native
attention, first residual, norm-2, and modulation prefix, then substituted
only the gate/up input before the native Q8_0 MLP suffix. `state_buf` and
`gate2_buf` remained native; therefore this intervention does **not** repair
the whole upstream prefix. The official donor is the BF16-widened
`img_mlp` prehook, not the unmodulated `norm2_output` hook. All arms retained
the full 1,254-row joint state, not an image-only cached state.

The native no-op splice and the earlier unspliced replay had the identical
full-output SHA-256
`82ff90f724520104cde4c970c1abbd97421f6399042643557679b4dc768bd6cf`.
The installed official donor matched its expected F32 SHA-256
`00a9a952d9c0d29cd2feaf8fcf61b7dece03c81c6159c5c255aeae20ec7d4ff0`
bitwise and changed the native Q8 gate/up tap. Root independently recomputed
the following squared L2 errors from the saved F32 arm outputs and the
official MPS BF16-widened block output (SHA-256
`edcc6e85a57b3b6e2ba68089b7c04ab1e3538ac998af1e176c2c6c45bc252c52`):

| Block-29 output versus official | Baseline squared error | Official-input splice squared error | Error removed |
| --- | ---: | ---: | ---: |
| Full joint state | 10,190.543619 | 9,677.000684 | 5.039407% |
| Image rows | 3,264.385298 | 2,885.289470 | 11.613085% |
| Predeclared face-20 rows | 60.821958 | 54.077347 | 11.089104% |

A control that merely BF16-rounds the *native* MLP input before native Q8
gate/up instead increased squared error by 0.338512% joint, 0.637715%
image-wide, and 0.952501% in the face-20 rows. This separates the
official-input intervention from a generic BF16 boundary cast. It is a
bounded causal effect on one block's hidden-state output, not the intrinsic
fraction of prefix error: the residual and second gate bypass the splice,
and the earlier gate/up-weight treatment may overlap it. The pre-MLP input
error was not face-concentrated relative to the image rows. Neither a
velocity change, a 40-step trajectory improvement, a decoded-image gain,
nor an eye-specific mechanism is established. The next causal split is
within the pre-MLP prefix, with no-op guards at a boundary between the
attention output/first residual and norm-2/modulation.

As a secondary same-artifact check, root independently widened the saved
official MPS BF16 gate and up preactivations and compared them with the native
Q8 F32 `[gate | up]` tap. With the donor MLP input, squared gate/up error
versus MPS dropped by 69.684881%/75.925718% on image rows and
65.554271%/72.537825% in the face-20 rows relative to the native-input
baseline. These composed comparisons include both input drift and Q8-versus-
MPS weight/arithmetic differences; they do not measure the intrinsic Q8
error or add to the block-output improvement.

The guarded replay status was `complete` on source commit `aff2a86c` with
all 32 runtime projection sets confirmed Q8_0. The scratch replay report is
`/private/tmp/qwen21-dit-operator-cause-20260928/mlp_input_splice_20260928/operator_splice/splice_report.json`
(SHA-256 `1eaa54c8ace6f08fd1c84c059e6702bd746e2d32d2c66197fd74768a4b13ddfd`);
runner SHA-256 is
`104474ac9ee0ee456952a1f42bd9c3c8c58c96fbbeceb8d4bf0fb5ddd8da343c`;
scratch-instrumented Metal-dispatch source SHA-256 is
`5082fe024e2f61768dfde5eb9e98d719ece33af24a4c574e8abc213fe724b90f`.
No production source was modified. Scratch can expire. Refresh this result
after model/weight, official framework/MPS, native source/kernel, fixture,
or tap/splice-boundary changes.

### Call-10 block-29 first-residual causal splice (2026-09-29)

A scratch-only resident Metal replay used the same exact official full-joint
post-block-28 state and modulation in four arms: unspliced native baseline,
native helper-input no-op, official BF16-widened first-residual donor, and
native first-residual BF16-rounding control. The intervention entered the
fused helper at `state = hidden + gate1 * attention_projection`, immediately
before LayerNorm and modulation. It passed the donor as helper `hidden` and
all-positive-zero `gate1`, leaving QKV, attention, and `to_out` computed from
the original native hidden state. The attention projection and `gate2` taps
were byte-identical across all arms; the official donor appeared in
`state_buf` bitwise. The native no-op matched both the prior unspliced block
output and every new operator tap bitwise. All 32 runtime projection sets
remained Q8_0, and no production source was modified.

Root independently widened the official BF16 block output and recomputed
the following squared F32 L2 errors from raw full-joint arm files. The
predeclared face-20 rows are the same rows used in the earlier block probes.

| Block-29 output versus official MPS | Native baseline | Official first-residual donor | Error removed | Native BF16-round control |
| --- | ---: | ---: | ---: | ---: |
| Full joint, 1,254 rows | 10,190.543619 | 4,446.770508 | 56.363756% | 7,645.529518 |
| Image target, 1,024 rows | 3,264.385298 | 1,094.100850 | 66.483710% | 3,453.418739 |
| Face-20 rows | 60.821958 | 22.804227 | 62.506589% | 63.853555 |

The same treatment reduced the native modulated MLP-input relative L2 gap
to the official tap from 0.841881% to 0.303978% joint, from 0.914917%
to 0.302385% image-wide, and from 0.810215% to 0.303807% in face-20.
Merely rounding the *native residual state* to BF16 slightly worsened the
image and face-20 block-output errors (5.790782% and 4.984380%);
its joint error improved 24.974272%, so it is not a universal null.

This cut establishes that an error already present at the first-residual
state has a substantial causal effect on this one block's output under the
fixed call-10 input. It does **not** distinguish native attention/to-out
error from gate-1 or residual arithmetic, and the donor changes both the
direct residual path and the downstream norm/MLP input. Therefore neither
the difference from the earlier MLP-input splice nor these percentages are
an additive attribution. No full velocity, 40-step trajectory, decoded
face, or VAE-manifold effect was measured. The next DiT discriminator
should split the attention projection from gate-1/residual formation using
the same full-joint input and no-op guard.

The successful guarded retry report is
`/private/tmp/qwen21-dit-operator-cause-20260928/residual_state_splice_20260928/operator_splice_elevated_retry_01/splice_report.json`
(SHA-256 `6d003b50ca5a182583d9ed6af0d65e60fba23e56583bdb2fcdf5fdaf019e27dc`);
runner SHA-256 is
`91a8e3330b884e3e61666a3f99cd68d7f63de253e19ad613a76ba1416ee47ce8`.
The first default-sandbox attempt failed before GPU execution because Metal
device initialization was unavailable there; its partial report was
preserved separately, not used as model evidence. Scratch may expire;
refresh after official framework/MPS or model, GGUF/native kernel, input
fixture, or splice/tap-boundary changes.

### Call-10 block-29 attention-projection input splice (2026-09-29)

A scratch-only resident Metal replay held the exact official BF16-widened
call-10 post-block-28 full-joint state, modulation, and native block-29
execution fixed. After native QKV, attention, and Q8_0 `to_out` ran, it
substituted only the projected-buffer argument to the fused
residual/LayerNorm/modulation helper. Four arms used the native projection,
an identical native no-op donor, the official MPS BF16-widened post-`to_out`
donor, or the native projection rounded through BF16. All 1,254 rows were
retained. The native projection, gate-1, gate-2, and hidden-input taps were
bitwise identical across arms; the installed donor buffers were checked
bitwise. Baseline and no-op outputs and all their taps matched bitwise,
including the pinned prior baseline output SHA-256
`82ff90f724520104cde4c970c1abbd97421f6399042643557679b4dc768bd6cf`.
There were no nonfinite projection values. The single guarded GPU run
completed with 78% free host memory at launch; production sources were
unchanged.

Root independently recomputed squared F32 L2 errors from all four saved
full-joint outputs against the exact official post-block-29 reference. The
face-20 rows are the same predeclared region used in earlier block probes.

| Block-29 output versus official MPS | Native/no-op baseline | Official projection donor | Error removed | Native BF16-round control |
| --- | ---: | ---: | ---: | ---: |
| Full joint, 1,254 rows | 10,190.543619 | 8,211.758448 | 19.417857% | 10,218.755032 |
| Image target, 1,024 rows | 3,264.385298 | 1,595.054262 | 51.137684% | 3,288.989055 |
| Face-20 rows | 60.821958 | 33.990453 | 44.114833% | 61.887517 |

On image rows, the first-residual relative L2 gap to the official tap fell
from 0.340660% to 0.176191%, and the modulated MLP-input gap fell from
0.914917% to 0.368888%; the face-20 MLP-input gap fell from 0.810215% to
0.36965%. Generic BF16 rounding of the native projection slightly worsened
all three block-output squared errors. Thus a discrepancy at or before the
post-attention/`to_out` projection boundary materially affects this one
block's image and face-region hidden-state outputs. It does **not**
distinguish attention-context drift from `to_out` weight/quantization or
arithmetic error. The projection-donor effect and the earlier
first-residual-donor effect are nonlinear, overlapping interventions, not
additive error shares. Nothing here proves a full velocity or 40-step
trajectory gain, decoded face/eye improvement, or a VAE-manifold mechanism.
The next DiT discriminator is a matched-input split at the pre-`to_out`
attention-context boundary, with passive official taps and exact no-op
guards if that boundary has not already been saved.

The completed report is
`/private/tmp/qwen21-dit-operator-cause-20260928/attention_projection_splice_20260929/attention_projection_splice/attention_projection_splice_report.json`
(SHA-256 `4a438e49dbc1878f3d8c0ba8e594911895e551b8925fc13f0d43284d1610b226`);
its CPU preflight report SHA-256 is
`dbd715bc344d5ed532ef84a23b8038b3c0cb869bccd658e3523d2dec5dc34928`.
The scratch runner source SHA-256 is
`fc04b0ccd17e90f8820aead6b057816b9f52d31dd5d3bcad3869fc32d93744b3`;
the official BF16 donor and its exact F32 widening have SHA-256 values
`fe57479cc658caa6d4974d578728173c0853e5c2a43b85014777f0b005cfca85`
and `87d3de388160dce3d26dac93de3f5282717e4a42eac927845369be23aaf6fc42`.
The probe ran at HEAD `4af5669bc72b5f7d02539bf803b344c15b1775a2`;
refresh after official framework/model/MPS, GGUF/native source/kernel,
input fixture, or splice/tap-boundary changes. Scratch may expire.

## Not admitted by this slice

- A production-scale, end-to-end resident Metal pipeline or native text encoder
  and VAE. A decoded image exists through the hybrid reference components, but
  is not a fully native decoded-image path. On the no-prefix route, layout
  metadata,
  sinusoidal timestep input, and text normalization/projection are still built
  on the CPU, and the final output is read back. The causal-prefix hit route
  skips condition-image projection for an ordered image-only target suffix,
  but still performs the timestep projections and a full-token output head.
- Production-resolution throughput, image-quality, or end-to-end latency from
  the synthetic-input profiles. The exact attention kernel remains
  correctness-first and quadratic in token count; its share did grow across
  the measured 1153- and 2049-token shapes, but behavior on other input
  distributions and a faster replacement remain unproven. The tested
  one-SIMD-group replacement is explicitly rejected at the measured shapes.
- A speedup from either Q8_0 batch projection or register reuse on other GPU
  families, quantization variants, or untested sequence lengths. Automatic
  register reuse is limited to the measured M2 Max batch corridor, with an
  explicit rollback switch.
- A compressed or bounded-memory prefix cache. The admitted implementation
  stores per-layer prefix K/V as F32 Metal buffers and therefore trades memory
  for repeated-step projection savings.
- A claim that a readable GGUF has acceptable image quality.
- A custom weight format derived from the resident-KV adaptive QBit codec.
- Trusting repository or file labels (`Q4`, `dynamic`, `HQ`) over tensor data.

## Guard and next transition

The raw-input certificate and GPU cache build/hit are admitted only for the
supported BF16 top-level route and a closed causal prefix. The older
host-projected route remains for unsupported weights. The local batched Q8_0
route keeps the prior GEMV path as an environment-controlled rollback; its
additional GPU allocation is zero because it reuses existing inputs, outputs,
and mmap-backed weights. Larger-token profiling still identifies attention as
a candidate, but removing per-key barriers by assigning one SIMD group to each
query/head did not pass the full-forward latency gate. At the tested real
256x256 prompt, Q8_0 Q/K/output projections dominate the diagnostic phase
profile; the first input-reuse tile failed the full-forward latency gate, while
register-local reuse passed the bounded 513-token gate. Keep the original
batched kernel as the immediate rollback. Before widening beyond exact M2
Max/batch `>=256`, repeat a full-forward paired A/B with exact output parity
on each new device, shape, and model packing. At larger token counts,
attention's measured share grows, so reprofile before assuming projections
remain the dominant target. Text encoding, sampling, and VAE decode remain
separate frontiers.

The model-backed checks are:

```bash
QWEN_IMAGE21_GGUF=/path/to/Qwen-Image-2.1-Q4.gguf \
  QWEN_IMAGE21_CONDITIONING=/path/to/qwen_image21_conditioning.json \
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  crystal spec spec/qwen_image21_flow_match_spec.cr \
    spec/qwen_image21_transformer_spec.cr \
    spec/qwen_image21_weights_spec.cr spec/qwen_image21_metal_spec.cr \
    spec/qwen_image21_resident_metal_spec.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"
```

## Falsifiers

- A real candidate contains an unimplemented tensor type.
- A `comfy.gguf.orig_shape.*` entry changes the tensor element count.
- Any required top-level or per-block tensor is absent, duplicated, or appears
  under an incompatible architecture.
- A full candidate is shorter than the last tensor extent declared in its
  directory.
- Any real block projection lacks a strict Metal route or exceeds the declared
  CPU/Metal tolerance.
- A resident stack uses more than one command buffer or performs an
  intermediate host readback for one outer-transformer evaluation.
- The resident final head reads back a hidden state, uploads a normalized
  intermediate, or issues a second command buffer before the output projection.
- Prefix caching changes a prefix hidden state, key, or value when only the
  target latents and FlowMatch timestep change.
- A changed prefix input or block configuration reuses the prior prefix cache
  instead of rebuilding it.
- A raw-input certificate reuses K/V after an encoder, condition-image,
  `t=0` row, prefix layout, or weight-identity change, or a prefix query can
  attend a target key while the target is omitted from cached prefix work.
- A closed resident cache hit reprojects condition-image rows, or the target
  suffix is not an ordered image-only sequence and still enters the cache path.
- Diagnostic phase splitting changes the forward output, or the Q8_0 batch
  route fails parity on Q/K/output shapes or loses its paired full-forward
  latency advantage at the tested token counts.
- A proposed attention replacement changes same-image bidirectional,
  cross-block causal, or invalid-key semantics; fails full-forward parity; or
  only improves its isolated phase while regressing paired one-command-buffer
  rebuild/hit latency or adding quadratic-size scratch.
- A later image-quality corpus shows that the selected mixed quantization
  policy is worse than its declared baseline.
