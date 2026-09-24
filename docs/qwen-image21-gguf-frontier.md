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
