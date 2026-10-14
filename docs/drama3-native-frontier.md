# DRAMA-3 native text-to-WAV frontier

Status: native CPU and opt-in resident Metal text-to-WAV smoke observed on
the pinned Q4 checkpoint. Resident inference keeps model computations,
activations, logits, routing, code selection, temporal state, and DAC on GPU.
The exact control/terminal read boundaries and measured scope are below.
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
- The initial runtime is an offline CPU correctness path. Metal now has a
  resident model graph; the earlier packed-expert/DAC hybrid is historical.
  Streaming, cloning, and speaker switching remain outside this slice.
  Unsupported configuration must fail explicitly.

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
`codec.safetensors`. Inference uses Crystal and native libraries; reference
Python environments are optional and separate.

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

## Native CPU verification baseline

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
cloning, reference-audio encoding, streaming, speaker switching,
long-form stability, broad intelligibility/quality evaluation, and real-time
performance remain untested or unimplemented. No subjective listening claim
is made. Source and weight licensing are recorded separately in
[THIRD_PARTY_DRAMA3.md](../THIRD_PARTY_DRAMA3.md).

Refresh the evidence after a weight/tokenizer/config change, a source or
precision change, or a backend change. A failure reopens the affected parity
claim; retaining a WAV from an earlier run does not close it.

## Resident Metal execution

`DRAMA3_BACKEND=metal` in the current Metal-capable CLI selects
`ResidentGenerator` and `CodecResident`. The preceding hybrid implementation
remains recorded in the historical section below. CPU remains the default
backend and explicit rollback. No existing Qwen admission path is changed.

Risk: CAUTION (GPU reductions, dynamic routing, cache ownership, and sampling
precision). Rollback: `DRAMA3_BACKEND=cpu`, or the preceding `9622936a` commit.
The highest-risk failure is altered routing/cache arithmetic hidden by a
plausible waveform. DoD: focused primitive/cache/failure tests, actual
backbone/tower/DAC gates at their existing tolerances, complete text-to-WAV,
call-site audit plus readback accounting, and matched component/total timing.
The executed evidence is indexed in
[drama3-resident-evidence.json](drama3-resident-evidence.json).

### Device/host contract

- Backbone: dense and packed projections, embeddings, norms, MoE top-k and
  shared-expert gating, both RoPE layouts, full attention, convolution ring,
  and Delta state updates execute in Metal. Packed experts stay packed.
- Tower: teacher forcing, temporal KV caches, enrichment/residual layers,
  logits, FP32 top-k/top-p selection, repeat fallback, and generated frames
  remain resident. The next backbone step consumes the original device frame.
- DAC: codebook reconstruction, raw-weight conversion and normalization,
  transformer, ConvNeXt, convolutions, Snake, GELU, and waveform run on GPU.
  Frame layout is `[main,c0,c1,...,c9]`; the decoder ignores main, uses c0 as
  semantic, and c1..c9 as residual codes. It consumes the valid prefix of the
  generator's original device allocation without a CPU code re-upload.
- Host work is tokenizer/config/shape validation, input and seeded-uniform
  preparation/upload, scheduling, EOS branching, terminal statistics and
  WAV/JSON I/O. A chosen main token (4 bytes) and a sticky finite-status word
  (4 bytes) are read per attempted frame. Completed codes are read once for
  the sidecar and PCM once for WAV; each terminal read checks 4 status bytes.
  Literal zero host reads is not claimed.
- Trace callbacks are diagnostic opt-ins. The normal CLI leaves them unset.
  `Context#read` rejects an `activation` purpose, and records explicit control,
  terminal, or diagnostic reads. Caller-supplied labels alone do not prove
  residency: the actual default call sites were also audited.

A serial command encoder retains temporary allocations and bound views until
checked GPU completion. Views retain their allocations; model/global mapped
weight caches retain mmap owners. Weight loads use 64-bit byte offsets and
unaligned-safe scalar decoding. Row/stage completion bounds live scratch.
GPU finite guards cover layer/component boundaries and sampling inputs; a
sticky fault rejects the next read even if the terminal tensor is finite.
This does not assert a finite check after every primitive.

Runtime objects are for one serialized inference. Do not reset global mapped
weight caches while a resident command/context is active. Finish all work
before `CodecMetal.clear!` and `MetalBackend.reset!` when unloading a model.
Concurrent inference/reset is outside the tested contract. Backbone cache
capacity is 1..1024; sampling top-k is 1..1024. CLI frame bounds do not guarantee
an aggregate RAM budget: long DAC tensors can be large. The executed admission
is the short 17-frame smoke, protected by run_safe, not long-form stability.

### Build and profile

```sh
make build/bridge.o
crystal build --release -Ddrama3_metal examples/drama3_generate.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld" \
  -o /tmp/drama3_generate_resident
DRAMA3_BACKEND=metal DRAMA3_PROFILE=/tmp/drama3-resident-profile.json \
  COGNI_RUN_SAFE_MIN_FREE_PCT=25 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /tmp/drama3_generate_resident 300 40960 \
  --model "$MODEL_DIR" --text 'Hello.' \
  --output /tmp/hello-resident.wav --max-frames 128 --seed 0
```

Omit `DRAMA3_PROFILE` for ordinary inference. Profiling synchronizes component
boundaries, so enclosing backbone, tower, and DAC spans include actual GPU
completion; inner MoE/dispatch spans can be enqueue time and are not kernel
measurements. Sum disjoint component spans only. Normal execution composes
adjacent component commands and synchronizes at row/control/stage boundaries.
First-use mapping, conversion, and pipeline compilation occurring during
inference are included. The current sidecar timer starts before generator
initialization and ends after WAV writing; it excludes tokenizer/config and
`WeightStore` construction, process startup, and compilation of the executable.
It is not a process cold-start metric. The JSON sidecar also records backend,
readback bytes by purpose, and submission count.

### Observed upstream and adversary gates

- 39 focused Metal examples pass; 18 CPU examples and six Python contracts
  also pass. They include eight recurrent/cache rows
  across convolution-ring wrap, capacity overflow, immediate EOS/empty prefix,
  invalid GPU expert ID/rank, non-finite intermediates hidden behind a finite
  terminal, tie order, top-p crossing, both codec layouts, poisoned main token,
  checkpoint singleton shapes `[1,C,1]` / `[C,1,1]`, and the highest semantic
  candidate in both primary and repeat-fallback selections. The last case
  failed before correcting an EOS-plus-vocabulary off-by-one bound.
- Final-source actual backbone: 243 traces / 497,664 values pass unchanged
  `atol=6e-5, rtol=3e-4`; max absolute error 0.0010986328125 at a large residual
  (relative error there 6.42e-7).
- Final-source actual tower: 726 traces / 2,042,880 values pass unchanged
  internal `atol=6e-4, rtol=3e-4`. Its final hidden states and 18 head logits
  separately pass the strict `3e-5, 3e-4` envelope: 20 traces / 77,824 values,
  max absolute error 1.04904e-5.
- Final-source DAC: all 29 internal stages pass `max_abs<=1e-3`, `RMS<=2e-4`,
  `normalized_RMS<=2e-4`. Worst observed values are 0.0007963181,
  0.0001249242, and 0.0000448816, respectively. All 4096 waveform values pass
  the per-sample `max(5e-5, abs(reference)*1e-3)` gate; max error 2.84016e-5.
- Full 17-frame DAC replay in pinned MLX: 34,359/34,816 PCM16 samples agree
  exactly, maximum error one PCM16 count, RMS error 0.1145694 counts. No
  normalization is used. Native CPU/GPU codes and prompt IDs match for this
  smoke; GPU/CPU PCM differs by at most one count.
- The normal smoke reads exactly 152 control bytes (18 attempts * 8 plus
  two terminal status checks * 4), and 140,012 terminal bytes (748 code bytes
  plus 139,264 F32 PCM bytes). Diagnostic and intermediate activation reads
  are zero; the DAC consumes device codes even though the sidecar also reads
  a terminal copy.

The sampler follows upstream ranking/tie/top-p rules in FP32. A precision
falsifier deliberately sets top_p=0.99999999 on equal logits: it rounds to 1
on GPU and can retain a candidate that CPU Float64 excludes. Reject universal
seeded CPU equivalence; do not hide the difference by widening trace gates.
Seed uniforms are prepared in the CPU RNG sequence and uploaded once, with
Float32 rounding. Real checkpoint comparisons continue to promote floating
weights to FP32 and preserve quantized weights; stock upstream BF16 equality
is outside this scope.

Correlated source adversary review is ROBUST for the default single-inference
residency/ownership path; executed upstream replays supply the numerical
support. The stronger exact-sampler, concurrent/reset, long-form, subjective
speech-quality, and general performance claims remain outside the evidence.

### Measured M2 Max latency

The final resident profile for `Hello.`, seed 0, 23 prompt rows and 17 frames
through EOS takes 19.9842 s: backbone 15.0293 s (40 rows), tower 2.0170 s
(22 teacher-forced plus 17 sampled frames), DAC 2.7622 s. It uses 129
submissions with profiling enabled. Earlier profiled runs took 22.1805 and
23.9230 s; an unprofiled run took 22.7126 s with 90 submissions. Those runs
preceded the final shape-metadata cleanup and/or highest-candidate guard fix;
their selected codes and WAV bytes are identical to the corrected build.
These repeats show local variability, not a statistical performance bound.

The same corrected binary takes 244.1005 s with `DRAMA3_BACKEND=cpu`, versus
19.9842 s resident Metal: 12.21x for this measured workload. Disjoint completed
component spans are:

| Component | CPU seconds | Resident Metal seconds |
| --- | ---: | ---: |
| Backbone | 96.4977 | 15.0293 |
| Tower | 8.2569 | 2.0170 |
| DAC | 139.2602 | 2.7622 |

Prompt IDs, all 17 generated code frames and EOS agree; 34,741/34,816 PCM16
samples agree exactly, with maximum difference one count. The earlier hybrid
run was 51.6059 s (2.58x slower than this resident run), but used the preceding
source and excluded generator initialization from its timer. Treat that as a
historical comparison, not the matched same-binary benchmark.

The first final-run prefill row takes 9.34 s, later rows about 0.19 s, and
generated rows about 0.21 s. Initial preparation versus execution is the next
discriminating profiling question; no attribution to a specific driver,
mapping, or compilation mechanism is established here. Host load was not
controlled and other processes were preserved. The result is a local M2 Max
observation, not a real-time, cross-device, or cold-start guarantee.

Refresh fixed-input gates, matched generation, readback audit/accounting,
and timings after changes to source, precision, assets, compiler, device/OS,
workload, or evidence availability. A failed gate reopens the affected claim;
an earlier WAV or a structurally valid evidence JSON does not close it.

## Historical hybrid Metal slice (`9622936a`)

The following records the preceding hybrid source state. Its implementation
and timing-boundary statements apply to that commit, not the current resident
CLI. Its preserved evidence is [drama3-metal-evidence.json](drama3-metal-evidence.json).

This slice profiles the existing CPU execution before choosing kernels.
Risk: CAUTION (parallel reductions and CPU/GPU ownership). Rollback is the
explicit CPU backend and the preceding native implementation commit.

- Measure backbone prefill/decode, tower teacher forcing/sampling, and DAC
  separately, including per-operation inclusive/exclusive host time. Record
  loading/setup separately from inference. Synchronous Metal timings must
  include transfer and command completion; kernel-only time is supplemental.
- First candidate: packed MLX affine projections and/or DAC convolutions,
  selected by measured cost. Preserve Float32 weights/activations and the
  existing cache and sampling semantics. No full-model dense expansion.
- The strongest failure mode is accumulated numerical drift that changes MoE
  routing or audio codes despite a plausible final waveform. Retain the
  upstream intermediate traces as gates at their existing scoped tolerances;
  do not widen them to make an optimization pass.
- DoD: profile all three components on the pinned checkpoint; accelerate a
  measured bottleneck with a selectable Metal backend; pass the synthetic and
  real-weight intermediate gates and the bounded native text-to-WAV smoke;
  report measured component and total latency including synchronization.
- Mini-Quadrumvirate at each key transition: predict the bottleneck/failure,
  test the discriminating assumption, implement the bounded move, and attack
  both parity and performance. A failed parity gate rejects that kernel path.

Performance evidence is local to this M2 Max, input, source, backend, and host
load. It expires after changes to kernels, dispatch, precision, weights, or
workload shape. CPU and Metal runs use the same bounded inputs; speed does not
substitute for correctness or establish subjective speech quality.

### Build and profiling

```sh
make build/bridge.o
crystal build --release -Ddrama3_metal examples/drama3_generate.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++ -fuse-ld=/usr/bin/ld" \
  -o /tmp/drama3_generate_metal
DRAMA3_BACKEND=metal DRAMA3_PROFILE=/tmp/drama3-metal-profile.json \
  COGNI_RUN_SAFE_MIN_FREE_PCT=25 RUN_SAFE_PASSTHROUGH_STDIO=1 \
  scripts/run_safe.sh /tmp/drama3_generate_metal 300 40960 \
  --model "$MODEL_DIR" --text 'Hello.' \
  --output /tmp/hello-metal.wav --max-frames 128 --seed 0
```

`DRAMA3_BACKEND=cpu` is the default, including in a Metal-capable binary. An
explicit Metal request fails if the binary/device cannot support it; it does
not silently fall back. Metal requires Darwin, unified memory, and a device
whose maximum buffer length can cover each mapped shard. All arithmetic in
this slice remains FP32, with packed Q4/Q8 weights and raw F32/BF16/F16 affine
scales/biases. Dense projections, attention, routing, gating, and sampling
continue on the CPU/Accelerate path.

`DRAMA3_PROFILE` enables nested synchronous host spans. JSON records each
scope path, category, call count, inclusive nanoseconds, and exclusive
nanoseconds. Sum disjoint component scopes; summing parent and child inclusive
times double-counts work. Metal spans include binding, submission, completion,
and readback; first-use mapping and pipeline compilation are included in
their enclosing spans. `weights.load` and `generator.initialize` are separate
setup spans. The WAV sidecar's `elapsed_seconds` starts after setup and ends
after WAV writing; it is inference latency, not process cold-start latency.
Profiling is disabled when the variable is absent.

`examples/drama3_trace.cr` and `spec/support/drama3/native_tower_trace.cr` accept
the same backend/profile variables. `examples/drama3_codec_trace.cr` accepts
`MODEL_DIR CODES_JSON TRACE_JSON [WAV_OUTPUT]` to decode fixed generated frames;
use `-` for `TRACE_JSON` to omit the potentially large trace bundle.

### Implementation and guards

The packed affine kernel computes each group's subtotal in CPU order, then
adds groups in ascending order. Contraction/fast math is disabled. Independent
gate/up projections share one command buffer; the down projections share the
next. CPU routing and final expert accumulation retain their original order.
Mapped weights remain packed and retain their mmap owners. Byte loads and
64-bit offsets support the actual checkpoint's unaligned U32 tensor starts
and scale offsets beyond 4 GiB. A real-weight replay rejected the initial
alignment assumption before this correction; focused tests cover odd byte
offsets, mmap tails, 65 groups, expert selection, reset, and F16 subnormals.

DAC Conv1d uses one output owner and the CPU accumulation order. Transposed
convolution gathers contributions in the original input-time/channel order,
with the same causal crop and bias placement. No atomics or reordered
reductions are used. Weight normalization remains on the CPU. Codec weight
buffers have a 128 MiB / 128-entry cache with value-change detection. Shared
scratch is serialized from host write through GPU completion and readback.
When unloading a model in a library host, call `CodecMetal.clear!` and
`MetalBackend.reset!` between inferences to release retained device views.

### Observed M2 Max result

The final matched pair uses the same release binary, pinned Q4 checkpoint,
`Hello.`, seed 0, 23 prompt rows, and all 17 generated frames through EOS.
Times below are disjoint component scopes and inference wall time in seconds:

| Scope | CPU | Metal enabled |
| --- | ---: | ---: |
| Backbone, 40 steps | 101.54 | 35.36 |
| Tower, 22 teacher-forced + 17 sampled frames | 7.84 | 11.27 |
| DAC | 141.07 | 4.91 |
| Generation + DAC + WAV write | 250.55 | 51.61 |

This pair observes 4.86x total acceleration and 28.72x DAC acceleration. Tower
arithmetic remains on CPU; these measurements do not establish a tower gain.
The host had unrelated CPU work and Spotlight activity, which were preserved.
There is no quiet-host or statistical performance claim. The earlier batched
Metal run took 48.54 s; the individual-submission Metal stage took 62.82 s.
Expert batching reduces 38,400 waits to 3,200, plus 44 DAC commands. It bypasses
`Matrix#matvec`, so compare `metal.affine_batch` and enclosing MoE counts,
rather than expecting Metal's `matvec.quant` call count to match the CPU's.

Both final backends reproduce the same prompt IDs, every code frame, EOS,
and WAV bytes. All 243 backbone traces and 726 tower traces equal the prior
CPU results exactly, then pass their existing upstream gates. Real two-frame
MLX DAC gates and the synthetic unequal-upsampling probe pass. Full 17-frame
MLX DAC PCM comparison retains the previous maximum one-count PCM16 error.
Final checks: 25 Metal examples including real codec parity, 18 CPU examples,
four focused synthetic codec examples, and six Python contract checks; no
failures, errors, or pending Crystal examples.

First-use preparation is substantial: the final Metal run's first prefill row
takes 23.48 s; later rows take about 0.38 s. Dense CPU projections and tower,
and bounded per-expert GPU weight views, are the next candidates. Separate
mapping/page preparation from lazy floating conversion before attributing
startup cost. This slice does not establish real-time generation.

The measured CPU baseline and both Metal stages, exact-output checks, source
hashes, commands, and remaining bottlenecks are in
[drama3-metal-evidence.json](drama3-metal-evidence.json). Existing upstream
tolerances above remain unchanged. A correlated read-only adversary review
found no in-scope indexing, ownership, or expert-ordering blocker; numerical
and timing claims rely on the executed replays, not that review alone.
