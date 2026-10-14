# DRAMA-3 native text-to-WAV frontier

Status: native CPU text-to-WAV smoke observed on the pinned Q4 checkpoint.
Goal: one short, single-speaker text prompt to a mono 44.1 kHz WAV, with all
inference computations in Crystal/native libraries. Python/MLX/PyTorch are
reference and fixture tools only, never the native inference implementation.

## Source identity and boundary

- MLX reference: cguldogan/mlx-audio, branch drama3-mlx, commit
  e7488935ec2ebfafd38ef2b58e290d79f3d05dc7.
- Official reference: fishaudio/fish-speech, commit
  541566932abbaebcb0a40e562a1f846738383f38.
- Official model: fishaudio/drama-3, revision
  e0a9218aa48888270f06ef1ed9ea67622c0e17d2.
- Converted weights: cguldogan/drama-3-mlx-q4, revision
  5331ab509359c0e4a7e353a19e255ddf3e8152a8. All five model shards, codec,
  tokenizer, configuration, and index have SHA-256 identities in the evidence
  report linked below.
- Existing Qwen runtime and dirty primary checkout remain untouched.
- First runtime is an offline CPU correctness path. Metal optimization,
  streaming, cloning, speaker switching, and performance promises are outside
  this initial slice. Unsupported configuration must fail explicitly.

## Design laws

- Keep residuals and zero-centered RMSNorm arithmetic in FP32. Do not pre-fold
  1 + w into BF16 weights.
- Backbone uses half-split partial RoPE; audio tower uses interleaved RoPE.
- Tower caches advance over time; prompt row p uses target frame p + 1.
- MoE routing is top-8 with normalized selected softmax weights plus a sigmoid
  shared expert. Nomic's MoE semantics must not be reused unchanged.
- Load quantized experts lazily; never expand the entire model to FP32/BF16.
- Reject non-finite intermediate results, invalid tensor shapes, and incomplete
  assets. WAV existence alone does not establish speech quality or parity.

## Execution plan and falsifiers

Risk: CAUTION (new model/weight/audio boundaries). Rollback: isolated worktree
and exact-path atomic commit; no changes to existing Qwen admission paths.

1. Native tensor math and mapped converted weights: compare dense/quantized
   projections, routing, norm, convolution, and both RoPE layouts to Python
   reference arrays; malformed shapes and corrupt inputs must reject.
2. Native backbone and tower: compare fixed-input intermediate tensors to
   pinned MLX; compare the numerically sensitive tower against official
   PyTorch as well. Test temporal cache continuation and prompt/target shift.
3. Native DAC decoder: compare codebook reconstruction, convolution stages,
   and waveform against reference. Encoder is deferred for text-only scope.
4. Prompt/tokenizer, generation loop, and PCM WAV: short bounded end-to-end
   run, retain token/code traces and numerical reports. Run under run_safe
   memory/time guards. No real-time or broad quality claim from this smoke.

DoD: focused Crystal specs pass; fixture comparison rejects a seeded defect;
real-weight intermediate reports identify revision/input/tolerance; native CLI
produces finite, non-silent 44.1 kHz mono audio from text. Until every part is
observed, report IN_PROGRESS and name the unresolved path.

Evidence decays on model/reference/native-source/precision changes. Refresh by
regenerating fixed-input intermediate traces and comparing before generation.

## Native use

The model directory must contain the pinned MLX-converted snapshot, including
all indexed safetensors shards, `config.json`, `tokenizer.json`, and
`codec.safetensors`. Inference uses Crystal and native CPU libraries only;
reference Python environments are optional and separate.

```sh
crystal build --release -Dcpu_only examples/drama3_generate.cr \
  -o /tmp/drama3_generate
COGNI_RUN_SAFE_MIN_FREE_PCT=25 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /tmp/drama3_generate 1200 40960 \
  --model "$MODEL_DIR" --text 'Hello.' \
  --output /tmp/hello-native.wav --max-frames 128 --seed 0
```

On the tested macOS/Xcode SDK, the build also needs
`--link-flags='-fuse-ld=/usr/bin/ld'` because the default lld rejects SDK stub
files. The output path and its `.json` sidecar must not already exist. The
sidecar retains prompt IDs, every generated code frame, sampling parameters,
EOS/truncation status, waveform statistics, and elapsed time. A bounded prefix
is explicitly reported as truncated. The runtime API lives under
`ML::Audio::Drama3`; the example directly composes `Tokenizer`, `Prompt`,
`Generator`, `Codec`, and `Wav` in one process.

## Observed verification

The measured source/asset identities and compact numerical results are in
[drama3-native-evidence.json](drama3-native-evidence.json). Reference replay
commands and dependencies are in [the reference tooling guide](../scripts/drama3/README.md).

- Focused native suite: 16 examples, zero failures/errors. The six Python
  contract checks include rejection of a seeded numerical defect, malformed
  traces, missing traces, non-finite values, and a Float32 JSON roundtrip case.
- Synthetic upstream fixtures exercise three cached backbone rows, temporal
  tower continuation, both RoPE layouts, zero-centered norms, and MLX affine
  Q4/Q8 lane order. Synthetic codec parity additionally uses unequal upsample
  rates, multiple residual books, clipping, and raw weight-normalization input.
- Tower configuration rejects unsupported bias, gating, normalization, RoPE,
  SWA/MoE, and feature-placement settings, including non-finite numeric values.
- Actual Q4 backbone: 243 boundary traces / 497,664 values across three cached
  rows compared with pinned MLX at `atol=6e-5, rtol=3e-4`. Maximum absolute
  error is 0.001220703125 at a residual magnitude of about 1712; the relative
  error there is 7.13e-7. The original `atol=3e-5` envelope failed four small
  residual values; the larger floor is limited to this recorded probe.
- Actual tower: 726 traces / 2,042,880 values compared with official PyTorch
  at identical hidden inputs and cache positions 0/1. The default
  `atol=3e-5, rtol=3e-4` envelope fails 55 intermediate values. A scoped internal
  floor of `6e-4` passes; aggregate relative L2 error is 7.41e-7. Final hidden
  and all head logits independently pass the original strict envelope. Logit
  max error is 1.14441e-5, all 18 argmaxes agree, and the smallest top-1 margin
  is 0.02979. This is measured drift, not bitwise parity or a universal bound.
- Actual DAC two-frame probe: all intermediate gates pass (max absolute
  error <=1e-3, RMS <=2e-4, normalized RMS <=2e-4). Waveform retains the stricter
  per-sample `max(5e-5, abs(reference)*1e-3)` bound; observed max/RMS errors
  are 2.8044e-5 / 6.2792e-6.
- Native `Hello.` run: 23 prompt rows, 17 generated frames, EOS observed,
  34,816 PCM16 samples at 44,100 Hz (0.78948 seconds). Peak 0.04967, RMS
  0.01073, no normalization. Model generation plus DAC took 241.83 seconds
  on an M2 Max. This is a correctness baseline with a slow scalar DAC.
- Final-source repeat: identical prompt IDs, all 17 code frames, and WAV bytes;
  elapsed 244.14 seconds. Final native backbone and tower traces were also
  regenerated against the same fixed inputs before committing.
- Full utterance DAC replay uses exactly those 17 native-generated frames in
  pinned MLX: 34,374 of 34,816 PCM16 samples match exactly; maximum difference
  is one PCM16 count, RMS difference 0.11267 counts. No peak or RMS
  renormalization is applied. This checks decoding of the complete smoke,
  independently of the two-frame intermediate probe.

Actual-checkpoint comparisons promote stored BF16 floating values to F32 on
both sides while preserving packed quantized tensors. They establish the
declared F32 execution path, not stock MLX BF16 equality. JSON trace values are
rounded to their declared Float32 type before comparison. Different RNGs and
floating-point reductions mean stochastic sequences need not match upstream.

## Remaining boundaries

Only short, single-speaker text generation is admitted by this smoke. Voice
cloning, reference-audio encoding, streaming, speaker switching, Metal kernels,
long-form stability, broad intelligibility/quality evaluation, and real-time
performance remain untested or unimplemented. No subjective listening claim
is made. Source and weight licensing are recorded separately in
[THIRD_PARTY_DRAMA3.md](../THIRD_PARTY_DRAMA3.md).

Refresh the evidence after a weight/tokenizer/config change, a source or
precision change, or a backend change. A failure reopens the affected parity
claim; retaining a WAV from an earlier run does not close it.
