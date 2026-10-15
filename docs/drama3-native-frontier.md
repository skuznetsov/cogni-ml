# DRAMA-3 native text-to-WAV frontier

Status: native CPU and opt-in resident Metal text-to-WAV smoke observed on
the pinned Q4 checkpoint. Resident inference keeps model computations,
activations, logits, routing, code selection, temporal state, and DAC on GPU.
The exact control/terminal read boundaries and measured scope are below.
Goal: one short, single-speaker text prompt to a mono 44.1 kHz WAV, with all
inference computations in Crystal/native libraries. Python/MLX/PyTorch are
reference and fixture tools only, never the native inference implementation.
The pinned MLX upstream is a community port; its source-path controls below
do not establish agreement with the original Fish Audio implementation.

## Listening follow-up and local ASR limits — 2026-10-15

At base `3156210b`, the user confirms that neither
`native-original-first-chunk.wav` nor
`mlx-official-chunks-styled.chunk-001.wav` delivers the sentence about the cat.
This closes the listening question for those two named clips as a negative
content result. A separate delayed reply describes a first clip ending at
“шорох” and a second whispered clip ending at “происходит”. Its question was
labelled “without tags”, which conflicts with that description. Preserve the
reply without assigning it to both no-cue WAVs; their per-file listening
identity remains unresolved.

Direct audio forwarding was attempted, but this session returned
`audio content omitted because you do not support audio input`. No agent
listening claim follows from a local path or an audio player rendering.
Instead, a blind local ASR control uses the pinned
`mlx-community/whisper-large-v3-turbo-asr-fp16` checkpoint at
`624c19c9af5603fa73b83bce14d4aeea96156d18`, through the existing clean MLX-audio
revision. All twelve model files are SHA-bound, with available Hub LFS digests
checked. ASR model inference selects MLX GPU with FP16 weights; audio
resampling and ASR decoding control include host work, separate from DRAMA
inference. No DRAMA precision, tolerances or generation boundary changes.

Ten full-file/negative controls and seven isolated controls run with automatic
language detection, temperature zero, no initial prompt or hotwords, and no
conditioning on earlier transcript text. The expected transcript is never
supplied to the recognizer. The observed limits are material:

- Digital zero and a quiet ten-second no-cue suffix both produce the invented
  text “Thank you.” The recognizer therefore fails the empty-silence content
  guard. It cannot certify that a word was actually spoken or absent.
- Both full no-cue WAVs yield the first sentence through “шорох”, followed by
  “Продолжение следует...”, which is absent from the requested text. Native
  styled chunk 0 also yields only that first sentence; the MLX styled first
  chunk yields the whispered continuation through “происходит”. Neither ASR
  output contains the cat sentence. These are candidate transcripts, not an
  additional listening confirmation or proof of missing speech.
- Both MLX final chunks yield “Ну и напугал же он нас сегодня, маленький
  проказник.” This supports a candidate delivered final sentence, not a
  full-text pass for the preceding chunk.
- Seed 1 and seed 3 both yield the Russian and English sentences. Independent
  crops from four seconds onward also yield the English sentence, with auto
  language `en`; their first four seconds yield Russian. Seed-3 crop RMS is
  0.109728 on normalized F32 PCM. This conflicts with the earlier human
  Russian-only report. A byte-exact check binds every seed source to its
  complete segment in the listening sequence, including seed 3 and both
  half-second zero gaps. The content disagreement is unresolved; neither
  source is silently promoted over the other.

Consequently, the earlier interpretation of seed 3 as an established English
omission is reopened. Its 179 equal code frames and EOS remain measured facts;
content is contested. The original long quiet-onset result remains unchanged:
its common native/MLX code prefix extends well past the measured quiet onset.
No unique silence cause, speech fix, strict sampler/DAC pass, or ASR transcript
accuracy is claimed. A full-stream matched-code cross-decode is the next
causal discriminator for the new diverging first-chunk pair; shared-prefix
PCM agreement alone does not establish decoder parity for their continuation.

Artifacts and source/run bindings are indexed in
`.landmark/evidence/drama3-listening-asr.json`; raw controls are under
`sampler-voices-20261015/asr/`. The earlier sampler packet remains a historical
snapshot at its source state. This follow-up supersedes its seed-3 content
interpretation, not its code/EOS measurements. Refresh on source, checkpoint,
file identity, preprocessing, ASR options, runtime or listening-evidence drift.

## Shared-uniform sampling and bilingual listening — 2026-10-15

At base `3fc3de68`, the new diagnostic compares production Metal sampling
with the pinned MLX filters on the same F32 logits. It captures both input
arms, top-k order, full-vocabulary softmax/CDF, retained probabilities, exact
sample IDs, and rank-indexed common Gumbel noise. Generation controls upload
a common F32 uniform tape once, using the native Crystal RNG order: main,
unconditional repetition retry, then nine residual heads. MLX uses an explicit
inverse-CDF shadow for these controls. This isolates common-variate behavior;
it does not establish stock seeded MLX RNG parity. The reference keeps the
existing F32-promoted core and packed Q4 tensors, with checkpoint-dtype DAC.
These matched controls call the one-prompt `generate_codes` path. They do not
reproduce the default long-text orchestration of `Model.generate`.

The original thirty-word native clip and the matched reference have identical
prompt IDs and **448 complete 11-code frames**, about 20.8 seconds. Both have
the near-silent suffix from roughly 9.5 seconds. Their first emitted-code
difference is frame 448, after the quiet transition. A denser teacher-forced
onset replay captures steps 0/1/2/4/8/16/32/64 and every step 96–223:
6,127,752 actual-generation values pass unchanged `atol=6e-5, rtol=3e-4`.
The terminal prediction at step 224 is reported separately as counterfactual.
This excludes native/reference code divergence as a necessary cause of the
original quiet onset under the common-uniform control. It does not identify
why the checkpoint/reference trajectory enters the quiet region.

Actual random draws and main-token history explain persistence of the
repeated token. At frame 198, token 250598 is new to the 16-frame window and
the retry is unused. At 199, the first and retry samples both select 250598.
At 208, both ordinary and retry filters retain only that candidate. Upstream
repetition-aware sampling makes one retry and accepts its result even when
it repeats; it is not an exclusion mask. All nine captured RAS decisions
match the native generation report. The repeated run starts after the signal
activity transition, so this explains persistence, not the onset of silence.

Three native randomized variants use the exact requested text:

> Я живу в Голливуде, Флорида, США. I live in Holliwood, Florida, USA.

The user hears both languages in seeds 1 and 2, and only Russian in seed 3.
This is direct human content feedback, not an energy-based transcript or a
controlled speaker-identity comparison. Seed 3 and the matched MLX shadow
have identical prompt IDs, all 179 emitted frames, and EOS at the same point.
The local ASR follow-up above conflicts with the Russian-only interpretation;
English omission is not established for this file. Code equality and EOS do
not resolve content. Random seeds do not guarantee distinct, reproducible
speaker identities.

| Prompt / seed | Native / MLX frames | Identical complete prefix | Termination |
| --- | --- | --- | --- |
| Original whisper + laughter / 1 | 489 / 513 | 448 | EOS / EOS |
| Plain bilingual / 1 | 197 / 197 | 197, entire output | EOS / EOS |
| Plain bilingual / 2 | 148 / 321 | 12 | EOS / EOS |
| Plain bilingual / 3 | 179 / 179 | 179, entire output | EOS / EOS |
| Bilingual whisper wrapper / 1 | 384 / 213 | 26 | cap / EOS |
| Bilingual laughter cue / 1 | 384 / 384 | 198 | cap / cap |
| Bilingual whisper + laughter / 1 | 287 / 287 | 287, entire output | EOS / EOS |
| Original thirty words, both cues removed / 1 | 768 / 768 | 768, entire output | cap / cap |

Removing both cues from the original thirty-word text also produces a quiet
suffix: every half-second window from 5.5 seconds onward peaks at one PCM16
unit in both outputs. All 768 frames agree, and both hit the configured cap.
This is a prompt ablation, not a speech-content or termination pass. Native
wrapper time is 279.029 s and MLX 101.881 s on the shared host; this is not a
matched performance experiment. Full-text listening for this pair is pending.

Source inspection identifies an API-path difference for the long prompts:
the styled original is 360 UTF-8 bytes and the no-cue variant is 332, while
the pinned high-level `Model.generate` defaults to 300-byte text chunks with
generated-code history. Its first chunk contains the first three sentences
(265/237 bytes respectively), followed by the 94-byte final sentence. The
matched one-prompt controls bypass this split. The plain bilingual input is
only 93 bytes, so this difference does not explain its seed-3 omission.

Two additional controls now call official `Model.generate` with its default
splitter/history policy, preserving the same F32 core, Q4 weights, row-prefill,
sampling settings and common uniform prefix. The no-cue prompt emits 515 + 89
frames (23.917 + 4.133 seconds); the styled prompt emits 741 + 78 frames
(34.412 + 3.622 seconds). All four chunks terminate by EOS. In the no-cue
first chunk, every half-second from 7 seconds onward peaks at at most five
PCM16 units; the styled first chunk has peaks of 37–51 from 12 seconds onward.
Thus this high-level orchestration does not eliminate the long quiet suffix
inside the first chunk in these runs. The second chunks have active signal,
but signal activity does not establish word delivery. The user confirms that
styled chunk 0 omits the cat sentence; listening for the remaining chunks is
not independently resolved. The within-text comparison changes prompt splitting, history and DAC segment
boundaries together; the styled and no-cue arms do not isolate chunking from
markup. The 768-frame cap is per chunk versus per prompt, so total termination
budgets are not matched. Wrapper times of 121.732 / 162.294 seconds are shared-host observations,
not a performance comparison or promotion.

The opt-in `--chunk-text` reference adapter uses the official host prompt
builder with zero-code history placeholders, then replaces only guarded audio
spans using GPU MLX operations. Actual history-code values remain on GPU during
generation. After terminal evaluation, all four prompt tensors match the
restored official `build_chunk_prompt` byte-for-byte. A single uploaded tape
is consumed across the chunks; `--max-frames` is per chunk and capture-step
indices are global. The native first styled chunk has the same text, 71 prompt
IDs, model path, seed, settings and initial F32 variates as reference chunk 0.
It emits 427 frames versus 741, sharing 111 complete frames before columns 9/10
diverge at frame 111. That new fork is not numerically localized or a parity
pass. Native stops before residual sampling on terminal EOS, while MLX samples
all residual heads before its EOS check. The tape has an 11-slot stride;
native uses 11 slots on non-EOS frames and two on EOS, versus 11 on MLX EOS.
No native second-chunk or cross-chunk draw-consumption parity is claimed.
This control tests pinned high-level text orchestration, with a sampler shadow
and custom GPU history adapter; it is not stock full-program execution.

The short whisper control isolates another failure mode. All 1,216,539
captured actual-generation intermediate values through step 26 pass the
existing tolerance, yet residual codebook 4 diverges at step 26. Candidates
135 and 764 reverse rank by only 1.9–3.8e-6 in temperature-scaled logits.
The shared draw 0.33058440685272217 selects rank 6, away from a CDF boundary.
Both GPU samplers choose 135 on the native vector and 764 on the MLX vector.
The trajectory fork is explained by a near-tie input ranking change; this
does not localize the arithmetic operation that produced it. No tie band,
precision, tolerance, or production filter was changed.

The fixed-logit onset probe executes 199 cases, including one synthetic top-k
boundary tie and both native/reference vectors at nine steps. Exact selected
IDs, top-k order, kept masks/counts and common-noise IDs agree in those cases.
The strict gate nevertheless fails: two upstream full CDFs decrease by one
F32 ULP in their discarded tails. The invariant remains `1e-7`; violations
are recorded rather than silently relaxed or allowed to abort the remaining
diagnostics. Malformed/nonfinite data and gross arithmetic corruption still
reject. Agreement of selected IDs is narrower than full sampler parity.
The separate whisper fixed-logit probe passes all 67 cases, including 3
actual RAS decisions, with no CDF invariant failure.

The verifier independently enforces canonical normal/retry settings and
reopens both source traces by path and SHA-256. It re-derives the captured
vectors and input comparison statistics, rejecting unbound trace cases or
edited vectors even when their probe outputs agree. Twenty sampler CPU tests
and nine generation-control tests pass, including these negative mutations,
history-span guards and per-chunk/global capture-budget checks;
Python ASTs, Crystal formatting and `git diff --check` pass. A correlated
source adversary verdict is ROBUST for consistency with the retained local
artifacts. Strict sampler parity and a unique causal explanation remain
unsupported. Trace metadata alone does not authenticate all weight shards
or attest execution; the external receipts and checkpoint manifest retain
their stated local provenance limits.

Same-code full WAV comparisons retain the failed one-LSB DAC gate:
plain seed 1 maximum/RMS delta 16/0.618570 PCM16 units, seed 3 10/0.464103,
and whisper + laughter 10/0.228216. Lengths and declared 44.1 kHz clocks agree.
The full thirty-word no-cue pair has maximum/RMS delta 3/0.182296, also failing
the one-LSB gate.
These bounds do not establish general codec, pitch, or voice-identity parity.
Energy analysis is never used to certify delivery of either language.

The new native sampler probe queues all cases and production sample calls on
GPU, with zero pre-drain reads or completion waits, one submission and one
final drain. Terminal serialization adds no submission or completion wait.
The existing scalar EOS/status boundary in generation remains; the new tape
and captures add no per-step host reads. MLX retains its official row-prefill
evaluation, frame-EOS evaluation/read and per-chunk decoder-length scalar read.
The history adapter adds no in-loop code-value readback by source inspection.
Prompt/provenance preparation
and terminal reports/WAV serialization run on CPU.

Risk is CAUTION, limited to diagnostic examples, reference helpers and tests.
Production arithmetic, defaults, graph scheduling, MAP_SHARED and precision
are unchanged. Rollback is to revert this diagnostic commit. Local artifacts
and command receipts are in `sampler-voices-20261015/`; the curated index is
`.landmark/evidence/drama3-sampler-voices.json`. All nine local checkpoint
files are SHA-bound; remote model revision identity is a label, not an
authenticated-byte claim. Refresh on source, input, checkpoint, reference,
precision, runtime, compiler, device/OS or artifact drift. Full text delivery,
the unique silence cause, strict sampler/DAC parity and the prior MoE39
hidden[33] gate remain open. No performance or LTP/WBA promotion is claimed.

To reproduce the sampler diagnostic, use `compare_sampler_common.py prepare`
with native and MLX replay traces, a native-order random tape and selected
`--step` values. Run `examples/drama3_sampler_compare.cr` on the resulting
fixture, merge the pinned GPU reference with `sampler_common_reference.py`,
then run `compare_sampler_common.py compare`. The CLI help documents the
exact input/output flags. The tape is generated by
`Random.new(seed)` and `Array(Float32).new(frame_cap * 11) { random.rand.to_f32 }`,
with no skipped retry draw. The `uniforms` JSON array is uploaded once by
`generate_mlx_control.py --random-tape`; omit that option for stock MLX
sampling. Add `--chunk-text` for the bounded official text orchestration control
and budget the tape for `max_frames * chunk_count * 11` values. Source/build
receipts are part of the evidence index: shader
hashes inside a probe report alone do not attest the Crystal harness build.

## Listening feedback and decoder clock — 2026-10-15

After listening to `generate-native-seed1.wav` and `generate-f32-control.wav`,
the user reports that **both stop delivering speech after the whispered
phrase**. The user also hears the native whisper as faster and higher. This
closes the earlier listening question for these two files as a negative
full-text-delivery result; the other reference controls have not received
the same listening check. Pitch remains a perceptual observation, not a
measured fundamental-frequency difference.

Read-only analysis at base `f908862d` finds that all three saved WAVs, including
`upstream-codec.wav` decoded from the native codes, are mono PCM16 at 44,100 Hz
with exactly 2,048 samples per code frame. The same-code pair has identical
1,001,472-sample lengths and maximum/RMS differences of 7/0.168956 PCM16 units.
This excludes a substantial decoder or declared WAV clock change for that
pair, while retaining the failed one-LSB codec gate. It does not establish
general codec parity or rule out a separate playback-device issue.

With 20 ms windows and RMS above 32 PCM16 units, the last active window ends
at 9.02 s in native and 10.36 s in the independently sampled F32 MLX clip.
The roughly 1.34 s difference measures signal activity, not word timestamps,
uniform time scaling, or speech rate. Native and MLX choose different main
tokens in frame 0: 250521 versus 248764. There is no identical generated
frame prefix. Text, all 87 prompt IDs and explicit seed/temperature/top-k/top-p
match, but RNG implementations differ and per-step draws were neither
aligned nor captured. These files cannot isolate numerical error as the
cause of the perceived voice or timing difference.

The analysis and hash-bound input/WAV identities are local in
`~/.cache/cogni-ml/drama3-runs/listening-feedback-20261015/analysis.json`
(SHA-256 `44f78b93d5cb1884595e5a48ce85040c5a5f8bafc8333e462794b9b224470751`).
The NumPy probe ran through `scripts/run_safe.sh` with a 15% reserve; a
separate standard-library PCM calculation reproduced the three RMS-window
counts and endpoints, and checked hashes, prompt IDs and settings. A direct
check also matches all ten codebook columns in the saved MLX codec input to
the 489 native frames, and checks the two WAV headers for equality. This
inspection reuses the earlier generated artifacts; it is not a new model run.
The missing continuation and its root cause remain open. Refresh these
file-specific findings on artifact drift. The next generation discriminator
still needs a common prefix, equal reference precision and explicitly aligned
sampling inputs, rather than equal integer seeds alone.

## Thirty-word whisper/laughter measurement — 2026-10-15

At base `a21ee640`, the native resident path was timed on Apple M2 Max / 64 GiB,
macOS 26.6.2, Crystal 1.21.0 and the same pinned Q4 checkpoint. The input has
30 whitespace-delimited spoken words, excluding the `<whisper>` tags and
`[laughs]` cue:

> <whisper>Тише, кажется, за дверью снова шорох. Давай подождём немного и послушаем, что происходит.</whisper> [laughs] Ах, это всего лишь наш кот уронил коробку! Ну и напугал же он нас сегодня, маленький проказник.

Settings: temperature 0.8, top-k 30, top-p 0.9, dense `tree`, expert down
`native`, MAP_SHARED off and graph depth 2. The official prompt builder and
native report have identical 87 prompt IDs. This is a single-prompt native
smoke; upstream's high-level API normally splits at 300 UTF-8 bytes, whereas
this input has 360 bytes. Tags are ordinary model input, not an implemented
native style controller. The user reports that the whisper sounds excellent
but the rest of the requested speech is absent.

| Seed / frame cap | Result | WAV length | Whole wrapper wall time | Native generation / DAC phase |
| --- | --- | --- | --- | --- |
| 0 / 512 | 512 frames, capped | 23.777 s | 192.047 s | 135.092 / 56.013 s |
| 0 / 768 | 768 frames, capped | 35.666 s | 269.553 s | 188.797 / 79.596 s |
| 1 / 768 | 489 frames, EOS | 22.709 s | 186.776 s | 133.147 / 52.538 s |

The selected seed1 run takes 185.727 s inside the native timer. The wrapper
includes process/preflight overhead; the native timer starts after WeightStore
construction and prompt preparation. Phase timers use CPU monotonic clocks and
add no GPU operation, read or wait. The reported initialization phase is
resident object construction, not complete eager model loading; first-use work
can occur in generation. These are individual shared-host runs with uncontrolled
cache state, not a matched speedup or steady-state latency result. The two
seed0 runs have an identical 512-frame prefix.

**Full-text delivery failed.** EOS is not evidence that all 30 words were
spoken. From 9.5 s to the end, every half-second window peaks at only 1–2
PCM16 units. Official MLX DAC decoding of the same 489 code frames is equally
quiet there. Across the entire 1,001,472-sample clip the DAC difference is
maximum 7 PCM16 units, RMS 0.168956; the existing one-LSB gate fails. The
second half has maximum DAC delta 1 unit and RMS delta 0.046976. This localizes
the gross silence to the code stream, without claiming full codec parity or
identifying why the generator chose those codes. A 65-frame repeated main
token run begins at frame 198 (nominal 9.195 s); it is an observation, not a
proved cause. PCM activity is not speech recognition or word alignment.

The new common-prefix probes consume all 489 native frame rows on GPU and
compare backbone hidden/main logits and tower hidden/nine codebook heads.
They capture steps 0/1/2/4/8/16/32/64/128/192/256 plus every step 361–489:
140 steps, 1,670 stages and 6,269,068 F32 values per arm. The pinned official
MLX architecture uses rowwise prefill and F32-promoted floating checkpoint
tensors, with packed integers preserved. This is the existing numerical
reference precision, not stock BF16 upstream generation. The native frame
prefix is teacher-forced in both implementations; sampling is disabled in
these replays.

At unchanged `atol=6e-5, rtol=3e-4`, both native down and opt-in `mlx_q4` first
fail at step 447; all captured steps through 446 pass. Native has 1,051,036
failing values and maximum absolute error 0.861520767 at step 455 main logits;
the opt-in has 1,051,066 and 0.861519337. Main-logit top1 differs at steps 448
and 463 in both arms; minimum top30 set overlap is 28/30. These ranks describe
forced-prefix predictions, not newly sampled output. Dense-tail capture
reproduces all 146 overlapping sparse stages / 546,829 values bitwise in both
the native and upstream runs, controlling for capture perturbation on those
overlaps. The observed late failure around 20.8 s cannot explain the earlier
onset of silence; unsampled earlier boundaries and sampler behavior remain
open. It does not isolate a router, reduction or unique rounding operation.

A separate low-level official `generate_codes` control uses the same prompt
IDs and settings without forced codes. Its core parameters retain canonical
BF16/Q4 dtypes, with F32 DAC, and prefill is one row per chunk instead of the
official default 128. This bypasses high-level text chunking. The initial
official-loader attempt was stopped by run_safe at sampled free memory 13%,
below the authorized 15% reserve, without a behavioral result. Disabling the
MLX allocator's free-buffer cache let the retry complete in 46.820 s wrapper
time: 768 frames / 35.666 s, at the frame cap. After the initial second,
half-second PCM peaks range from 42 to 90 units. This does not establish
successful requested speech. Different RNGs and reference precision prevent
an exact native trajectory or silence-cause inference. The official loop
retains its per-frame EOS evaluation/read and DAC-length scalar read; this
control is not synchronization-free. Decode-call and terminal-evaluation
timings split lazy execution and are not comparable native phase latencies.

A second independently sampled control promotes core floating tensors to F32
using the replay's shardwise loader, preserving packed integers and the official
sampler/DAC. It returns 384 frames / 17.833 s before the cap in 72.111 s wrapper
time (40.215 s loading, 27.677 s generation). Its waveform has activity through
10.5 s, followed by half-second peaks of only 5–10 PCM16 units to the end.
There is also a quiet interval at 4–6 s. This is another incomplete behavioral
control, not a speech-content pass or a matched-RNG native comparison. The
official API does not expose its termination reason in this report. F32 alone
does not establish recovery of the requested text.

A third control uses the pinned public `Model.generate` path with the same
custom F32 core loader, default 300-byte text chunking and 128-row prefill.
It returns both expected chunks in 108.865 s wrapper time, including 45.113 s
loading and 61.867 s in the official iterator. The first chunk includes the
whisper, laughter cue and the sentence ending with "коробку!": 71 prompt rows,
534 frames / 24.799 s. Every half-second from 8 s to that chunk's end peaks
at only 20–28 PCM16 units. The history-conditioned second chunk has 633 prompt
rows and 114 frames / 5.294 s; it has active PCM through 4.5 s, followed by
a quiet tail. Neither chunk reaches its frame cap; the API does not report
the precise termination reason. Public chunking therefore does not remove
the long quiet tail inside the first chunk. This is a public-path control,
not a single-cause experiment: chunking, prefill and independent RNG change
the trajectory. PCM activity does not establish which words or styles were
delivered. Full speech delivery and the unique silence cause remain open.

The pinned port documents `[laughs]` as an inline cue. Its generic splitter
protects paired tags such as `<whisper>...</whisper>`, while the prompt builder
passes their text to the tokenizer unchanged; this is not an explicit style
controller or a source-backed guarantee of the model's learned tag semantics.
A fourth public-path F32 control removes only that wrapper, retaining every
spoken word, `[laughs]`, seed/settings and the two chunk boundaries. It returns
451 + 104 frames in 129.996 s wrapper time. The first chunk is 20.944 s, with
half-second peaks of only 1–4 PCM16 units from 7 s to its end; the second is
4.830 s, with activity through 4 s and a quiet tail. Both expected chunks
return before the cap. Removing the wrapper does not remove the quiet suffix
in this sampled control. The wrapper is therefore not necessary to reproduce
this symptom under these conditions; its possible contribution to the original
trajectory remains open. This single prompt ablation neither establishes
word delivery nor identifies the generator's silence or stop cause.

Native inference computations and state remain on GPU. The selected generation
records 3,928 control-read bytes and 4,027,404 terminal-read bytes; graph depth 2
drains with no pending work or retained bytes. Native diagnostic replays assert
zero pre-drain readbacks and no additional submissions/completion waits during
terminal serialization. MLX uses official per-row cache evaluation; its
internal backend submissions/waits are not instrumented. CPU performs prompt,
checkpoint, terminal WAV and report work.

Risk: CAUTION, bounded diagnostic and example CLI change. The CLI frame limit
now permits 1–1024 and rejects a resident prompt/frame bound exceeding the
existing 1024-row cache. Defaults and kernel arithmetic are unchanged. Rollback:
revert this measurement/probe commit. The important falsifier is a silent
suffix despite a non-silent global waveform and EOS, so neither global RMS nor
EOS is treated as a successful text-delivery gate. The strict comparator rejects
incomplete/malformed capture schedules, mismatched inputs/config/index, invalid
F32 data and widened tolerances. A native schedule bug that included checkpoints
above the frame count was caught before its GPU run and corrected.
The comparator checks capture contents and identity fields; it does not itself
attest source files or checkpoint shards. Exact trace hashes, primary-source
execution receipts and local checkpoint hashes are bound externally in the
evidence index. The older external control/activity helpers self-report hashes
matched to their final files; these are not independent launch-time receipts.
Observed guards: 18 focused Crystal generation/graph examples and 15 Python
capture/comparator tests pass; real CLI bounds and all 87 prompt IDs pass.
Mini-Quadrumvirate verdict: ROBUST for these bounded measurements and negative
gates; VULNERABLE for full-text delivery, which remains open. Correlated review
does not replace the executed controls.

Raw WAVs/traces and command logs remain local in
`~/.cache/cogni-ml/drama3-runs/whisper30-20261015/`; the curated index is
`.landmark/evidence/drama3-whisper30.json`. All actual cached checkpoint files
were hashed, with unchanged file metadata checked after the measurements;
this binds local bytes, not authenticated remote revision identity. Refresh
on source/checkpoint/input/reference/precision/runtime/compiler/device/OS or
artifact drift. Independent upstream sampling and speech-content verification
are separate from the common-prefix numerical gate. The next generation
discriminator needs equal reference precision, aligned random draws and
captures around the first trajectory divergence / silence onset. The original MoE39
hidden[33] gate and its next substitution probe below remain the return frontier.

## Native MLX Q4 arithmetic and full gate — 2026-10-15

At base `31920e41`, the opt-in native Metal down projection reproduces the
pinned MLX0.31.1 result bitwise on the retained row1/layer39 SwiGLU activation
and identical mapped packed weights. The original full-backbone gate still
fails one value at unchanged `atol=6e-5, rtol=3e-4`. This section supersedes
the older gate measurements below; production parity remains IN_PROGRESS.

`kernels/mlx_q4.metal` reproduces the fast Q4/group64 path: 16 activation
values per lane, packed-nibble input rescaling, `scale*dot + sum*bias` per
lane/group, 512-column block accumulation and `simd_sum`. It uses F32
arithmetic and GPU expert IDs; no dequantized weight expansion is introduced.
The separate source preserves the existing kernels' contraction settings.
The native two-dimensional dispatch permutes independent row/rank axes;
scalar and batched results are checked on the actual checkpoint. Source
expressions and `simd_sum` are not a disassembly proof of compiler contraction
or the hardware reduction tree. Upstream attribution is in
`kernels/mlx_q4.LICENSE`.

`DRAMA3_EXPERT_DOWN_REDUCTION=mlx_q4` selects only routed-expert down
projections; the value is captured when constructing the backbone. The default
remains `native`, with dense `tree` and MAP_SHARED off. Unsupported bit width,
group size, K tails or N tails reject before encoding. The admitted shape
requires affine Q4/group64, K divisible by 512 and N divisible by 8.

| Executed check | Observed result |
| --- | --- |
| Exact upstream activation, native MLX-style down | All 16,384 values across eight experts bitwise upstream; expert108 scalar and batched 2,048 values bitwise upstream |
| Repeat and retained-native controls | Candidate repeat bitwise; native down reproduces the previous three tolerance failures at expert108 469/766/1555 |
| Full upstream gate, candidate | 243 boundaries / 497,664 values; exactly one failure, row1/layer39.hidden[33] |
| Full upstream gate, default native | Same one failure as before; all 497,664 captured native values bitwise equal to the retained native trace |
| Synthetic native guards | 24 Crystal examples pass, including distinct K=1024 blocks, odd scale/bias offsets, F32/BF16/F16 metadata, GPU IDs, scalar/batch and frozen selector |
| Adjacent reference/capture guards | 32 Python tests pass at unchanged tolerance |

At hidden[33], upstream is `0.9219169616699219`; the candidate gives
`0.9215669631958008`, absolute error `0.00034999847412109375` against allowance
`0.00033657508850097655`. The earlier native error was
`0.00081634521484375`. The candidate is still outside the gate; neither the
smaller error nor synthetic exactly representable fixtures establish general
MLX bitwise equivalence or a complete correction.

All inference stays on GPU. `examples/drama3_mlx_q4_probe.cr` enqueues native,
candidate, repeat and scalar arms before one final drain. There are zero
pre-drain readbacks/submissions/waits, one final submission/wait and no added
wait during terminal serialization (212,992 terminal bytes). The passive
buffer callback in `examples/drama3_backbone_gpu_trace.cr` captures all three
rows in one GPU command buffer, then reads 1,990,656 terminal bytes and 972
finite-flag control bytes after that drain. It replaces the per-boundary host
callback for this diagnostic. CPU checkpoint/provenance preparation and
post-drain report analysis are outside inference.

A bounded follow-up uses the existing MoE trace on the exact upstream
normalized input. Gate/up, SwiGLU, candidate down, routed sum and shared stages
all pass fixed tolerance, but are not generally bitwise upstream. The first
failing official boundary is final `moe_output_official`, coordinates 1012 and
1111. The full run has a different normalized input and is explicitly
unqualified as an exact-input comparison. An initial depth0 MoE comparison
exited 2 because that comparator requires depth2; the depth2 rerun is valid
and exits 1 for those two numeric failures. Capture/no-capture control matches
8,192 values bitwise, with one final wait and no intermediate readback.

Risk: CAUTION. Rollback is disable the opt-in or revert this atomic change.
DoD was the real exact-input probe, original full comparator and adjacent
guards through `scripts/run_safe.sh` at sampled reserve 15%, with profiling
unset. The expected failure mode was wrong affine grouping, stride, mapping,
contraction or a misleading local pass; actual checkpoint controls, independent
synthetic indexing fixtures and the unchanged full gate discriminate these.
Mini-Quadrumvirate verdict: ROBUST for measured local down reproduction and
the bounded negative full-gate result; a production parity claim is VULNERABLE.
No latency, default or LTP/WBA promotion is claimed.

Exact commands, source/artifact hashes, runtime and next signal are indexed by
`.landmark/evidence/drama3-native-mlx-q4.json`; raw artifacts remain local in
`~/.cache/cogni-ml/drama3-runs/mlx-q4-20261015/`. Actual 14 target-MoE tensor
bytes match the pinned upstream digest before the probe. The historical full
reference manifest binds reference/input/source/config/index, but does not
hash every backbone weight shard; it is not an authenticated all-weight
certificate. Refresh on source, checkpoint, input, runtime/reference,
precision, compiler, device/OS or artifact drift.

Next falsifier: on common normalized input, substitute upstream gate/up-derived
activations and shared output independently on GPU, then compare the official
final MoE sum. Passing separate relative-tolerance stages does not guarantee
passing their sum near cancellation. Keep the original hidden[33] full gate
as the admission check before promoting any correction.

## Expert-108 quantized versus decoded F32 down — 2026-10-15

The common-input GPU experiment reproduces the upstream quantized down return
bitwise, while `mx.dequantize` followed by F32 `mx.matmul` differs at the
unchanged tolerance `atol=6e-5, rtol=3e-4`. This comparison resolves the next
falsifier from the MoE investigation below; it does not fix production parity.
No native arithmetic, precision, default or scheduling changes are included.

`scripts/drama3/probe_mlx_expert_down.py` loads the retained row1/layer39
SwiGLU return, preserves all eight explicit expert IDs and selects expert 108
at rank 7. It verifies the actual 14 target-MoE raw tensors and config/index
against the prior trace before importing MLX. The three down tensors retain
packed uint32 Q4 weights and BF16 scales/biases promoted to F32 exactly as in
the pinned reference. Actual lazy-array shape/dtype assertions pin direct
matmul input to `[1,512]`, decoded weights to `[2048,512]`, and outputs to F32.
The normalized-input fixture and actual expert activation have separate hashes.

| GPU arm or control | Observed comparison |
| --- | --- |
| Official-shape eight-expert `gather_qmm` | All 16,384 down values bitwise equal to retained upstream |
| Single-expert `gather_qmm` and `quantized_matmul` | Each bitwise equal to upstream expert 108, all 2,048 values |
| `dequantize → F32 matmul` versus upstream | Two tolerance failures: 469 and 766; maximum absolute difference 0.00390625 |
| Retained native down on exact upstream activation versus decoded F32 matmul | All 2,048 values pass unchanged tolerance; not bitwise equal |
| Retained native down versus quantized arm | Three failures: 469, 766 and 1555, reproducing the prior result |
| Explicit GPU uint4 `scale*q+bias` decode versus `mx.dequantize` | All 1,048,576 F32 weight bit patterns equal; zero maximum error |

At coordinate 33, upstream/quantized gives -9.041109085083008, decoded F32
matmul gives -9.04194164276123, and retained native gives
-9.041943550109863. Native and decoded F32 differ by only 1.9073486e-6 here.
The maximum absolute error in the table occurs on a large output and is not
the tolerance-failure criterion; every output is checked separately.

This rejects a mismatch between the two tested decode formulas and demonstrates
a difference between fused quantized dot arithmetic and decoded-weight F32
matmul on these exact bytes. It does not identify a unique rounding operation,
establish which result is more accurate, or prove a complete full-backbone
counterfactual. The explicit decode control implements the native-style
formula in MLX; it is not a capture of the native Metal decoder itself.

The pinned MLX v0.31.1 source (commit
`ce45c52505c8158ea48d2a54e8caae05efd86bfe`) factors group scale/bias outside
partial packed dots and uses SIMD reduction in its fast Q4 QMV path, whereas
the native kernel multiplies individually decoded weights and sums group
partials. This makes affine regrouping and reduction order the next target.
FMA remains unisolated. Source dispatch predicts ordinary QMV/GEMV for the
actual M=1 shapes, with no NAX/TF32 route; the run also sets
`MLX_ENABLE_TF32=0` before importing MLX. Kernel names and internal backend
wait counts were not profiled. Primary source/hash references are in the
local `source-audit.json` indexed by the evidence packet.

All projection, decode and weight-comparison arithmetic runs on GPU. Six
terminal result/metric reads follow one explicit `mx.eval`; no intermediate
GPU-to-host activation read or explicit synchronize call occurs in the probe.
This source/API audit does not measure MLX's internal submissions or waits.
CPU work is checkpoint loading/provenance, fixture setup and terminal analysis.
The original 497,664-value full gate remains open at hidden[33]; no full-model
run or latency claim is added by this diagnostic.

Risk: CAUTION, bounded reference probe only. Rollback is revert this diagnostic
commit. DoD: execute the real probe through `scripts/run_safe.sh` with the 15%
sampled reserve, reproduce the official full-eight capture bitwise, record all
2,048 target values for each arm, then run the negative provenance/tolerance/
readback guards and adjacent comparator/capture tests. Observed: probe exit 0,
32 Python tests pass, and all decoded-weight bits match. A second GPU run after
closing a replay-source hash guard reproduces all 22,528 projection values
bitwise across the four arms. Numeric differences
between diagnostic arms are report data, not an exit-0 assertion of parity.
Mini-Quadrumvirate verdict: ROBUST for this bounded discriminator, with the
full-backbone parity gate still open. Review and exact commands/source/artifact hashes are indexed
in `.landmark/evidence/drama3-down108-qmm.json`; raw artifacts remain local in
`~/.cache/cogni-ml/drama3-runs/down108-20261015/`. Refresh on source, checkpoint,
activation, reference/runtime, precision, device/OS or artifact drift.

Next falsifier: reproduce the pinned low-M packed Q4 dot's affine grouping and
reduction in an opt-in GPU diagnostic, preserving Q4 storage, F32 arithmetic
and one terminal drain. Qualify exact-input down first, then rerun the original
full upstream gate and adjacent guards before considering production adoption.

## Layer-39 MoE discrepancy investigation — 2026-10-15

Status: diagnostic localization observed; the production parity correction is
IN_PROGRESS. The retained full-backbone gate at base `7bf40df8` still fails at
exactly one of 497,664 values: row1/layer39 hidden[33], native
0.9211006164550781 versus upstream 0.9219169616699219. Absolute error is
0.00081634521484375; the unchanged allowance is 0.00033657508850097655.
Dense tree/SIMD arms are identical. Neither dense nor MAP_SHARED defaults are
promoted. No production arithmetic or Metal kernel changes in this diagnosis.

The probes retain actual GPU inputs, router logits, selected weights, expert
gate/up/activation/down returns, official routed/MoE returns and residual
boundaries. Expert rows are compared by explicit ID while retaining raw rank
order. Weighted terms and prefix sums computed by separate diagnostic kernels
are labelled auxiliary; they do not substitute for official return values.
The pinned MLX model runs with floating checkpoint tensors promoted to F32,
as in the previous reference; this is not stock BF16 execution parity.

| Boundary on exact upstream normalized input | Observation |
| --- | --- |
| Router logits | Maximum absolute difference 7.153e-7; passes fixed tolerance |
| Selected experts | Same eight IDs; upstream raw order 159,194,245,154,230,191,171,108; native order 108,171,191,230,154,245,194,159 |
| Selected weights by ID | Maximum absolute difference 5.960e-8; passes |
| Expert gate/up and activation | All pass; equality is tolerance based, not bitwise |
| Expert down returns | First failing stage; expert 108 fails at coordinates 469 and 766 |
| Official routed sum | Fails at coordinates 469,766,791 |
| Official MoE return | Fails at coordinates 1012,1111,1191,1640 |

The strongest intervention bypasses gate/up/SwiGLU: replay the **exact stored
upstream activation** through the existing native affine-Q4 down kernel. For
expert 108 at coordinate 33, native is -9.041943550109863 versus upstream
-9.041109085083008, delta -0.0008344650268554688. That delta survives identical
activation bytes. The same native down kernel replaying its own captured
activation reproduces all 16,384 captured down values bitwise.

Five GPU reduction arms separately hold terms, weights and order fixed.
Native reduction of upstream down rows and weights in upstream order matches
the actual upstream routed sum **bitwise across all 2,048 values**. Reordering
fixed terms with their paired weights changes bits at other coordinates but
causes no tolerance failures and no change at coordinate 33. Native terms
with upstream weights/order retain a -0.000834465 coordinate-33 sum delta;
native weights change that by only +0.000000954. Native terms/weights in
native order reproduce the captured native sum bitwise. These are bounded
interventions on this workload, not a universal claim about MLX reduction.

Full-native normalized input differs in 1,936/2,048 F32 bit patterns (maximum
1.90735e-6), so its stages are not qualified as an exact-input comparison.
At original coordinate 33, native-minus-upstream deltas are +0.000015259 for
the attention residual, -0.000831604 for MoE, and -0.000816345 for hidden.
Residual cancellation around 13.28 + (-12.36) exposes the final violation.
Replacing full-native input with upstream input changes native MoE[33] by
only about 1.907e-6. This supports the down projection as the dominant
contribution; it does not prove a complete counterfactual explanation of
every full-pass difference.

Capture controls: all 243 upstream full-backbone arrays match the retained
no-capture reference bitwise; detailed upstream full and isolated MoE stages
match bitwise. Native detailed/no-detailed captures match four output vectors
(8,192 values) bitwise, and three final rows plus the selected residual/hidden
match the retained native baseline bitwise. In the real trace and ablation,
graph submissions/waits are zero before the final drain and one afterward,
unchanged by serialization. Guard/status and diagnostic readbacks occur only
after that drain. No ordinary-inference diagnostic allocations, dispatches,
readbacks or completion boundaries are added when hooks are unset. Diagnostic
calculations and reference inference stay on GPU; CPU code handles checkpoint
loading/provenance, fixture indexing and terminal report analysis.

Both native examples hash the actual 14 target-MoE mapped tensor payloads from
their loaded `WeightStore`, plus config/index bytes, before GPU context
creation. Their target digest matches upstream:
`648d4c736dd60b6c077c032565aa647b6e90f6fd96d2ee957f22cbcd69a889dc`.
Checkpoint files must remain immutable during a run. This binds the selected
layer's weights, not every checkpoint shard or authenticated remote contents.
Final release checks pass 21 Crystal examples and 21 Python tests (20 report
guards plus one upstream capture regression); both real GPU examples exit 0.

Risk: CAUTION. Rollback is unset diagnostic callbacks or revert this diagnostic
commit. Tolerances remain `atol=6e-5, rtol=3e-4`; Q4 weights, floating precision
and ordinary inference scheduling are unchanged. Runs use the 15% sampled
reserve and do not modify unrelated processes. `DRAMA3_PROFILE` must be unset
(a value of `0` still enables profiling).

Diagnostic DoD: reproduce the original failure, qualify capture fidelity and
single-drain counters, compare exact-input stages by ID, then intervene on
down input and summation order. The observed first failing boundary is the
expert down projection. The mini-Quadrumvirate adversary verdict is ROBUST for
that bounded localization. Exact Q4 decode/product/reduction cause and the
production parity fix remain open. The down-path experiment above tests the
next falsifier on the same activation and raw weights. The subsequent pinned
MLX core source audit strengthens the affine-grouping/reduction hypothesis;
the exact rounding suboperation still needs a controlled GPU intervention.

Reproduction tools: `scripts/drama3/replay_mlx_moe.py` (full/isolated upstream),
`examples/drama3_moe_trace.cr`, `examples/drama3_moe_ablation.cr`, and
`scripts/drama3/moe_compare.py`. The comparator's exit 1 is the expected
preserved numerical mismatch; malformed pins, shapes, capture provenance,
expert maps or graph/readback controls are rejected separately. Run native
examples with `--release -Ddrama3_metal`, `DRAMA3_BACKEND=metal`, graph depth 2
and profiling unset through `scripts/run_safe.sh`. The exact commands,
test signals, source/artifact hashes and remaining frontier are indexed in
`.landmark/evidence/drama3-moe39-diagnostic.json`. Raw artifacts remain local
under `~/.cache/cogni-ml/drama3-runs/moe39-20261015/`; refresh on source, model,
reference/runtime, input, precision, compiler, device/OS or artifact drift.

## Dense backbone projection experiment — 2026-10-14

Status: opt-in candidate implemented; default promotion withheld. Measure the
actual checkpoint shapes of the shared-input `qkv/z/b/a` projections before
promoting an optimization. The baseline dense
kernel uses 64 lanes, lane-strided FP32 accumulation and a fixed binary tree
with seven threadgroup barriers. Candidate reductions must preserve every
lane's product/add order and that tree, raw BF16/F16/F32 decoding, GPU input
and output allocations, mapped-weight ownership, and command scheduling.

Risk: CAUTION. The strongest failure is a SIMD reduction that changes the
addition tree or assumes a SIMD width without a guard. First require bitwise
baseline/candidate equality on cancellation, tails, unaligned offsets and real
checkpoint shapes. Measure paired AB/BA GPU command intervals with preuploaded
inputs, depth two and only the final pair drain; reject additional submissions,
waits or reads. Retain the baseline selector for rollback. Promotion requires
repeatable projection improvement, unchanged full-generation WAV/codes/EOS and
readback/graph ledger, and refreshed pinned upstream intermediate comparisons.
The missing temporary reference was reconstructed from pinned MLX-audio
revision `e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, with MLX 0.31.1.
No new quantization or precision path is admitted. MAP_SHARED remains
default-off and outside this experiment.

DoD: release-build and run `examples/drama3_dense_benchmark.cr`, qualify its
paired timeline/counter/output records, execute the focused dense and resident
graph specs, replay backbone intermediates against pinned MLX-audio and compare
matched baseline/candidate full generation. Expected signal: bitwise native
outputs, unchanged upstream tolerances, zero intermediate waits/readbacks, and
a measured improvement without a material full-pipeline regression. If the
candidate fails, keep the baseline and pivot using the measured discriminator.

The shuffle-only candidate did not improve whole-generation GPU time. The
bounded follow-up uses an aligned BF16 `ushort` load in the candidate only,
guarded by dtype and even byte offset; odd offsets and other dtypes retain the
original decoder. Before the final full-generation matrix, the admission rule
is: all three real input rows must show a paired median kernel speedup above
1.03 for each projection, native bitwise/upstream gates must pass, and matched
full-generation GPU median candidate/baseline ratios must be at most 1.01 at
depths zero and two. Host wall time remains descriptive on this shared host.

The checkpoint header audit finds all 120 projections across 30 linear blocks
are dense BF16, have even mapped byte offsets, and occupy 1,517,813,760 bytes.
The layer-0 shapes are `qkv=[8192,2048]`, `z=[4096,2048]`, and
`b/a=[32,2048]`. The candidate keeps FP32 products/accumulation and the exact
64-lane summation tree. With 32-lane SIMD groups, one threadgroup barrier
publishes partials and shuffle instructions implement the remaining tree.
Other SIMD widths retain the shared-memory tree. Only aligned BF16 uses the
16-bit load; other dtypes and odd byte offsets retain the original decoder.
The non-32-lane branch is source-reviewed, not exercised on this M2 Max.

Three successive real prompt embeddings (token IDs `248045,8678,198`) were
replayed through upstream layer-0 normalization to obtain the timed inputs.
Each input has two warmups and seven alternating AB/BA measured pairs per
projection. Each arm repeats `qkv/z` eight times and `b/a` 1024 times in one
command buffer. Both buffers submit at graph depth two before one final pair
drain. GPU times are command-buffer timestamps, amortized over calls, not
isolated dispatch latency. These are hot, repeated-input microbenchmarks.

| Projection | Row 0 paired median speedup | Row 1 | Row 2 |
| --- | ---: | ---: | ---: |
| qkv | 1.083x | 1.082x | 1.068x |
| z | 1.101x | 1.095x | 1.109x |
| b | 1.591x | 1.587x | 1.635x |
| a | 1.581x | 1.611x | 1.578x |

Every measured pair improves in this bounded sample. A separate tree/tree
control gives paired medians 0.994, 0.997, 0.998 and 1.002 respectively. All
1,474,560 output elements per invocation match bitwise after drain. Each
invocation has 72 command buffers, maximum pending two, final pending/retained
zero, zero intermediate waits/readbacks, and 11,796,480 terminal verification
bytes. There are no new production submissions, completion boundaries,
readbacks, copies, scratch allocations or cache ownership rules.

The standalone upstream comparison passes all 37,056 values across three rows
and four projections at the unchanged `atol=6e-5, rtol=3e-4`. Maximum absolute
errors are `qkv=3.815e-6`, `z=2.027e-6`, `b/a=9.537e-7`; tree/SIMD match
bitwise. The reference promotes checkpoint floating tensors exactly to F32,
preserving packed integers, and executes the pinned MLX model. This is not a
claim of bitwise equality with stock BF16 execution. Fixtures bind input bytes,
source/checkpoint revisions, selected tensor hash and shape/dtype metadata;
they do not authenticate edited logs or hash the entire checkpoint.

Full native backbone traces also match bitwise: 243 tensors, 497,664 F32
values. However, the refreshed full upstream comparison fails for **both**
tree and SIMD at `backbone.row1.layers.39.hidden[33]`: MLX `0.921916962`,
native `0.921100616`, absolute error `8.164e-4`, allowed `3.366e-4`.
This is the sole out-of-tolerance coordinate. A fresh upstream run without
dense capture reproduces all 243 reference tensors exactly, excluding the
capture instrumentation as the cause. The discrepancy is downstream of the
last attention residual, in the existing MoE/residual path. Different expert
ordering/reduction is a source-supported hypothesis, not an established root
cause. Next discriminator: exact row1/layer39 router logits, expert IDs/order,
weighted expert outputs and partial sums. Do not widen the tolerance.

`DRAMA3_BACKBONE_DENSE_REDUCTION=simd` selects the candidate only for these
four projections in all linear blocks; rollback/default is `tree`. Generic
`DRAMA3_DENSE_REDUCTION` independently defaults to `tree`. The generation JSON
records the frozen, actual selectors, preventing a vacuous two-baseline timing
comparison. Default promotion remains blocked by the full upstream gate even
if the isolated performance and native bitwise gates pass. The existing
mapping-promotion frontier remains separate and default-off.

Matched release full generation (`Hello.`, seed zero, max 128 frames,
MAP_PRIVATE, profiling off, GPU timeline on) completed two warmups, four
AB/BA pairs at depth two and two at depth zero. Median paired
candidate/baseline ratios are:

| Graph depth | Pairs | GPU command sum | Inference elapsed | Whole process |
| --- | ---: | ---: | ---: | ---: |
| 2 | 4 | 0.99239 | 1.01218 | 1.02404 |
| 0 | 2 | 0.97089 | 0.88189 | 0.90373 |

The GPU non-regression gate passes, but these whole-process samples do not
establish a general latency improvement: depth two is 2.4% slower in process
time on the shared host. Host interference and pipeline setup have not been
separated. Do not equate an isolated 1.6x tiny-projection speedup with a
comparable whole-model gain. Earlier shuffle-only depth-two GPU ratios had
median 1.00402 and did not justify promotion.

All 14 wide-candidate runs reproduce the same 23 prompt IDs, 17 frame/code
rows, EOS stop, 34,816 waveform samples and WAV SHA-256
`3bcc331bd234ba6a3950d2122215bdc663aa4f7427a112e8f8df108085908a49`.
Both arms retain 152 control and 140,012 terminal readback bytes. Depth two
uses 73 submissions/completions/waits and maximum pending two; depth zero uses
90 and maximum pending zero. Every run drains pending/retained resources.
The initial harness incorrectly required equal submission counts across
depths; that rejected depth-zero run was excluded and the corrected matrix
checks the established per-depth counts and the actual selectors.

The final release GPU suite passes 15 examples covering dense BF16/F16/F32,
32 mapped dtype/width/offset fixtures, sliced input, strong cancellation,
selector/context ownership, resident operations and graph resource lifetimes.
Fourteen Python tests include negative controls for timing/counter/stream mutations,
stale fixtures, wrong model/revision/weight metadata, wrong candidate bits,
both native arms exceeding upstream tolerance, and attempted tolerance
widening. Adversary: ROBUST for bounded opt-in arithmetic, device ownership
and measured results; VULNERABLE for default promotion and general process
speedup. Correlated agent source reviews support, but do not replace, the
executed GPU and pinned-reference checks.

A separate warm-driver pipeline diagnostic records 0.263 ms total library,
function and pipeline creation for `dr3_dense_simd`; this small warm setup cost
does not explain the observed process-time difference. It is not a cold-driver
startup measurement. The diagnostic uses the existing host pipeline profiler
and adds no GPU synchronization.

Canonical evidence and executable commands are indexed in
`.landmark/evidence/drama3-dense-backbone.json`; raw local artifacts are in
`~/.cache/cogni-ml/drama3-runs/dense-20261014/`. Refresh on source, assets,
reference/runtime, precision, compiler, input, device/OS or artifact drift;
performance also decays with host/cache state. The sampled free-memory floor
was 15%, quiet-host gating was off, and unrelated processes were untouched.

## Sampler GPU cost and next target — 2026-10-14

Status: VERIFIED for the bounded diagnostic below; full-inference sampler
attribution remains open. `examples/drama3_sampler_benchmark.cr` builds with a
relative `require` and the normal Metal link flags, resolving the earlier
unsupported `-I` / `--path` attempts. No inference kernel or default changed.
Risk is CAUTION for measurement interpretation and queue lifetimes; rollback
is removal of the diagnostic and its analysis tools.

The release probe calls unchanged resident `Context#sample`, including its
finite-input guard. Each pair encodes 36 main-head calls (18 normal at
temperature 0.8 / top-p 0.9, 18 retry at 1.0 / 0.95) and 153 tower-head calls
(0.8 / 0.9), all at top-k 30, in two command buffers. Logits and uniform draws
are uploaded before timing. Graph depth two admits both submissions before
one final drain. Both operation groups have zero intermediate completion
waits and readbacks. Per-output checks run only after that drain and must add
no GPU waits/submissions. Warmups exercise both submission orders; seven
measured pairs per profile alternate main/tower order and reverse profile
order. GPU durations are `GPUEndTime - GPUStartTime` for whole command buffers,
not CPU encode wall time or per-dispatch timestamps.

Final run `root-measured-r3`, invocation
`80ab95a36d308772874d86d58da91914`, on M2 Max/macOS 26.6.2:

| Fixed-input profile | Main/retry 36 calls, median ms | Tower 153 calls, median ms | Pair GPU median ms | Pair min–max ms |
| --- | ---: | ---: | ---: | ---: |
| Retained decode head 0 / main proxy | 82.069 | 349.449 | 431.568 | 430.113–432.153 |
| Retained prefill head 0 / main proxy | 83.665 | 357.682 | 441.401 | 440.347–442.127 |
| Retained decode head 8 / main proxy | 78.760 | 334.678 | 413.687 | 412.930–414.760 |
| Synthetic ascending logits | 839.504 | 3569.096 | 4407.456 | 4404.510–4419.828 |
| Synthetic descending logits | 67.851 | 286.451 | 354.414 | 353.952–355.536 |
| Synthetic flat ties | 71.604 | 304.844 | 376.440 | 375.298–376.956 |
| Synthetic top-k cutoff ties | 67.468 | 288.039 | 355.561 | 353.644–356.542 |

Each column has its own median; component medians need not sum to the median
of paired sums. The three trace-based profiles cost about 2.19–2.34 ms/call
amortized over each command group, not measured isolated-call latency.
Only the tower rows are retained checkpoint logits; main rows prepend a
synthetic EOS logit to a tower row because no actual main-head logits were
retained. Reusing one row per group is a hot-input microbenchmark, not replay
of all 189 production distributions. Ascending input increases insertion
comparisons about 29.9 times and measured pair time about 12.4 times versus
descending. This confirms data-dependent cost, not an observed production
worst case. Host quiet gating was off; the runner's 15% sampled reserve passed.

All 11,907 outputs (2 warmup + 7 measured pairs, 7 profiles) match a scalar
FP32 oracle, including main retries. There are 126 submitted/completed command
buffers and final completion waits, maximum pending two, final pending and
retained bytes zero, and exactly 47,628 control plus 47,628 terminal read bytes
for post-drain verification. Deliberately wrong expected IDs and depth one
both fail. Nine Python analysis controls pass, including cross-run stream
mixing, shifted/missing/duplicate timestamps, invalid durations, extra waits,
readbacks, partial output verification and retained resources. Matching random
invocation stamps and zero-based command indices reject accidental mixed-run
logs; they are not authentication against edited logs. Global profile order
is also checked against the executable's alternating schedule.

`scripts/drama3_sampler_reference.py` extracts and executes the exact pinned
Fish-Speech sampler on CPU at revision
`541566932abbaebcb0a40e562a1f846738383f38`; no model import or new dependency is
needed beyond the existing Torch reference environment. All ten non-tie normal
role-cases have matching retained support, with conditional probability
max absolute error at most 5.96e-8. Main retry support matches in all five
non-tie cases, with max probability error 7.45e-9; its other two cases have
the same deliberate tie discrepancy. The four deliberate flat/cutoff tie cases
disagree: native ties prefer lower token IDs, while upstream CPU
`torch.sort(stable=False)` reorders equal logits. Uniform inverse-CDF draws
and upstream exponential-race draws also have different RNG contracts.
Injected exponential noise verifies in-support target selection only; it
does not prove RNG or general sampler parity. Expected GPU IDs come from the
scalar native oracle, whose host `exp` rounded to FP32 can differ from Metal
near a decision boundary. Prior backbone/tower/DAC intermediate and WAV gates
remain unchanged-source evidence, not freshly rerun inference in this probe.

Next attribution experiment: actual-shape backbone shared-input projections,
starting with `qkv/z/b/a` in `BackboneResident#linear_attention`. No projection
bottleneck has been established. Earlier command sums
place about 6.67 s in combined prefill/generation versus 1.75 s in DAC, while
this bounded sampler costs 0.414–0.441 s on trace-based profiles. These are
different measurement boundaries; do not subtract them or infer a production
speedup. First measure the projection groups with the same asynchronous
method. Reject fusion if actual-shape timing or intermediate replay regresses;
preserve each output's accumulation order. Sampler parallelization remains a
candidate, especially for the ascending tail, but changing denominator
reduction order risks top-p boundaries and is not justified by insertion
comparison counts alone. Any kernel change must rerun its relevant pinned
upstream intermediate replay and full-WAV check before promotion.

DoD commands, source/binary identities, the qualified report and reference
audit are indexed in `.landmark/evidence/drama3-sampler-cost.json`. Raw local
artifacts are under
`~/.cache/cogni-ml/drama3-runs/sampler-cost-20261014-r2/`; early runs remain
historical diagnostics, and only `root-measured-r3` supplies the table above.
Refresh on source/reference/assets/precision/compiler/input/device/OS or
artifact drift; timing additionally decays with host state. The mapping gate
below remains incomplete and default-off.

## Active mapping-promotion experiment

Status: IN_PROGRESS / default promotion withheld. The final controlled attempt
completed nine of sixteen required pairs, then failed warm-cache preparation
after both allowed repair passes. All eighteen completed arms passed exact
baseline WAV/prompt/codes/EOS/readbacks and graph-drain checks; the lowest
sampled inference headroom was 26%, above the authorized 15% reserve. The
blocker is repeatable cache conditioning, not an observed correctness failure
or the memory floor. Default and rollback remain `DRAMA3_MMAP_SHARED=0`.

The private default remains admitted until this experiment passes. Compare the
same release binary with mapping modes 0/1 at graph depths 0/2, separating
verified cold and warm OS file-cache states. Residency is measured with
`mincore`; read-only, model-scoped `msync(MS_SYNC | MS_INVALIDATE)` is eligible
only after a known-content fixture proves eviction without byte/stat changes.
This does not control SSD/controller caches. No global purge or interference
with unrelated processes is admitted.

Risk: CAUTION. Rollback is mapping mode 0. Each completed run must reproduce
the baseline prompt/codes/EOS/WAV/readback ledger and drain all graph leases.
Cold preparation requires zero resident model pages; warm preparation requires
all model pages resident immediately before launch. Record preparation time
separately from complete process time and retain both. The user explicitly
authorized a 15% system-free-memory reserve for this experiment after the
initial 30% reserve blocked its first arm. Require 60% before cache preparation
and cold launch, using the observed at-least-45-point drop plus the 15% reserve
as an admission estimate. Warm launch requires 20%, because cache preparation
has already made the model pages resident; it leaves a five-point margin above
the sampled reserve. Monitor preparation and inference; both the parent and
the owned `run_safe` wrapper stop at or below 15%, recording an invalid sample.
This is sampled pressure control, not a proof of a continuous physical-memory
bound, and resident file pages are not equivalent to permanently pinned RAM.
A post-warm high-headroom requirement was rejected because it could exclude the intended
cache state itself. Interrupted or cache/headroom-invalid pairs cannot promote
the default and remain visible in the evidence.

Predeclared promotion gate: at least four complete alternating-order pairs in
each cache/depth cell; shared/private process-time median at most 0.95 in every
cell, no pair above 1.05, no correctness/ownership/readback regression, and no
memory-guard or reserve violation. This is a local workload/device gate, not a
general model-performance guarantee. A failed cell keeps private as default
and redirects work to attribution. The strongest counterexample is that the
previous improvement disappears when initial file-cache residency is equal.

The first controlled attempt (`shared-controlled-20261014-a`) normalized all
1,578,824 model pages to zero residency and launched private/depth-zero at 75%
free. It reached the then-active 30% reserve at 14.912 seconds; the owned
workload was stopped and reaped. Zero pairs completed, so the interruption is
not a speed or parity result. Source/binary identities and model file stats
remained unchanged. The new 15% policy is a separate experiment; it does not
retroactively validate this arm. Fresh private and shared fixture mappings
must also see warm pages before invalidation, preventing an always-zero
fresh-map residency observation from qualifying the cache control.

The first 15%-reserve attempt (`shared-controlled-20261014-b`) completed three
exact-parity pairs before rejecting a partially resident preparation for
resident/depth-two/private. Its private/shared process times were 26.53/19.70 s
(nonresident/depth zero), 22.95/18.46 s (nonresident/depth two), and 15.62/9.77 s
(resident/depth zero). Its lowest sampled inference headroom was 21%; cache
conditioning, rather than the reserve, blocked this attempt. These three pairs
do not satisfy the matrix gate. That harness discarded the failed preparation's
cache snapshot before recording its exception; a later query is only a later
observation and cannot reconstruct that failed snapshot.

Unmapping each shard immediately after reading it allowed earlier shards to
be reclaimed during later reads. A cache-only discriminator kept all six
read-only mappings alive through the complete touch, then released them all:
fresh queries observed all 1,578,824 pages resident both while helpers existed
and after their release. The revised preparation creates all helper mappings
before touching any, releases all on normal/error/reserve exits, and requires
fresh all-resident mincore after cleanup and again immediately before launch.
No helper mapping or open model descriptor survives into the timed child.
The same conditioning applies to both mapping modes; preparation time remains
outside process time. A partial state fails closed with its per-file snapshot
retained. This is cache conditioning, not page pinning or an inference change.

Keeping mappings alive alone was insufficient: attempt
`shared-controlled-20261014-c` completed eight exact-parity pairs, then rejected
resident/depth-zero/private preparation at 1,501,733/1,578,824 resident pages.
The held and post-cleanup snapshots agreed: 77,091 pages had already been
reclaimed from the first model shard while all helpers were mapped. Preparation
headroom was 64–74%; the lowest completed-inference sample was 24%. This is a
cache-conditioning failure, not a reserve violation. Its eight pairs remain
separate and cannot promote the default.

A cache-only probe from a retained partial state restored all pages with one
missing-page pass and observed full residency after helper cleanup. The next
matrix starts from scratch. Warm conditioning now performs the same full read
for both modes, followed by at most two mincore-directed missing-range reads
while all six helpers remain mapped. It records the ranges, bytes, duration,
and before/after snapshots for each repair. Two passes are a fixed bound; a
partial post-cleanup or prelaunch state still rejects the arm and stops the
matrix. Repaired pages are newer than retained pages: exact mincore occupancy
and alternating order do not establish identical cache ages.

Final attempt `shared-controlled-20261014-d` stopped before launching
resident/depth-two/private pair two. Its full read left 1,447,197 pages resident;
repair one read 2,420,670,464 bytes and reached 1,487,301 pages; repair two read
3,728,264,286 bytes but fell to 1,462,154 pages. After helper cleanup only
1,456,273/1,578,824 pages (92.24%) remained resident. Preparation took 63.105 s
with sampled free-memory readings of 71–78%. More repair passes, weaker cache
criteria, and pooling earlier partial matrices are not admitted by this gate.
This refutes reliable full-residency conditioning by the bounded read/repair
procedure on the current host; it does not refute shared mapping's speedup.

The completed pairs are encouraging but insufficient for promotion:

| Initial file cache | Graph depth | Complete pairs | Median paired shared/private process time |
| --- | ---: | ---: | ---: |
| Zero resident pages | 0 | 2 | 0.730317 |
| Zero resident pages | 2 | 2 | 0.786786 |
| All resident pages | 0 | 3 | 0.638089 |
| All resident pages | 2 | 2 | 0.611097 |

Every completed pair favored shared mapping. A separate artifact recheck read
all eighteen WAVs and reports, recomputed the four cell summaries, checked
cache snapshots, readback/drain ledgers, and sampled reserves, and agreed that
the matrix cannot promote. This is same-lineage verification, not independent
replication. The harness/guard claim is ROBUST within the executed controls;
the speed/default-promotion claim remains VULNERABLE because no cell has the
required four pairs. The next mapping discriminator requires a changed cache
conditioning premise or a controlled host, rather than another identical retry.

Canonical identities, failed attempts, prior unchanged-source upstream lineage,
and refresh boundaries are in
[the controlled mapping packet](../.landmark/evidence/drama3-mapping-controlled.json).

Reproduce the guard checks and the full matrix from this worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh python3 120 1024 \
  -m unittest spec/drama3_mapping_benchmark_spec.py
PYTHONDONTWRITEBYTECODE=1 COGNI_RUN_SAFE_MIN_FREE_PCT=15 \
  scripts/run_safe.sh python3 60 512 scripts/drama3_mapping_benchmark.py \
  --model /Users/sergey/.cache/cogni-ml/drama3-mlx-q4-5331ab50 --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/drama3_mapping_benchmark.py \
  --model /Users/sergey/.cache/cogni-ml/drama3-mlx-q4-5331ab50 \
  --binary /tmp/drama3_shared_final_generate \
  --baseline-wav /Users/sergey/.cache/cogni-ml/drama3-runs/hello-speed-baseline.wav \
  --output-dir /Users/sergey/.cache/cogni-ml/drama3-runs/shared-controlled-new
```

The output directory must be new. Existing output is never overwritten; each
arm and any interrupted attempt remain in `progress.json`. `--self-test` uses
only a 1 MiB fixture on the model volume and pure gate/report controls; the
five Darwin guard specs use real owned CPU children with injected pressure or
cache readings, including rejection before child launch. Five additional
controls cover post-cleanup rejection, all-helper cleanup after preparation
failure, missing-range clipping, repair convergence, and the two-pass bound.
Neither check establishes model speed or upstream numerical parity.

The next optimization discriminator is GPU stage attribution inside the
backbone, without adding completion waits. Existing command timelines assign
about 6.67 seconds of GPU execution to the combined prefill/generation commands
and 1.75 seconds to DAC. These are command-group sums, not individual operator
times. Depth two reduces command completions from 90 to 73 while leaving those
GPU sums essentially unchanged; it is not evidence that deeper queuing will
reduce compute time. The current component profiler deliberately calls
`Context#finish` in `BackboneResident#step`, so its wall spans cannot identify
the next asynchronous kernel bottleneck. Attribute a measured operator before
changing kernels; require the existing intermediate upstream gates and matched
complete-process timings for any resulting optimization. The main-token EOS
control read remains an explicit boundary, with only 152 control bytes in the
current smoke; generated frames already pass directly between GPU operations.
The existing bridge exposes GPU start/end timestamps for whole command buffers,
not individual dispatches. A bounded fixed-input sampler/projection probe can
submit isolated groups at depth two and drain once, using the existing timeline
without adding per-operation completion waits. Such synthetic group timing
does not attribute the complete generation path by itself.

A bounded external sampler probe was initially drafted but produced no measurements:
both allowed build attempts exited on unsupported Crystal include-path flags
(`-I`, then `--path`) before compilation. No GPU workload ran. Its source,
latest build log, and a curated two-attempt diagnostic are retained outside
the repository and indexed by the mapping packet. The operator hypotheses
below were unmeasured at that point. The sampler-cost section above supersedes
that build blocker and records the subsequent bounded measurements.

Source-level follow-up candidates, pending operator attribution and matched
measurements:

- `dr3_sample` runs top-k insertion and the full-logit exponential denominator
  in one GPU thread. The actual main head has 4,097 EOS/semantic candidates,
  and the nine residual tower heads have 4,096 each; this is not the 252,288
  text vocabulary. The 17-frame/EOS smoke executes 36 main/retry and 153 tower
  samples at top-k 30. Time batches matching these 189 calls with fixed logits
  and draw slots. Parallel top-k may help, but a reassociated Float32
  denominator can change top-p cutoffs; retain tie order, threshold rules,
  ordered draw subtraction, and both main/retry draws. Existing Qwen top-2
  reductions are a structural reference, not a resident sampler replacement.
- `BackboneResident#linear_attention` dispatches `qkv`, `z`, `b`, and `a`
  projections from the same input. Test a combined projection dispatch on the
  actual shapes, preserving each output's accumulation order; fewer launches
  alone do not establish a full-process improvement.
- `TowerResident#codebook_sum` allocates a scalar row, embedding, and sum for
  every codebook. Test one gather-and-ordered-sum kernel, preserving code
  clipping, masking, and accumulation order, against the existing tower trace.
- DAC's convolution kernels and separate Snake materialization warrant
  attribution within the measured 1.75-second phase. A naive fusion that
  recomputes Snake for every convolution multiply may make it slower; require
  actual-shape GPU timings before choosing fusion or convolution tiling.

These are hypotheses, not measured speedups. Each candidate needs a fixed-input
baseline/candidate replay with one final drain, followed by the unchanged
intermediate upstream and complete-WAV gates. The command-phase reanalysis and
original timeline identities are retained with the mapping experiment evidence.

## Prior first-submit slice

Status: read-only shared checkpoint mapping implemented as opt-in;
performance/default promotion remains provisional. Set `DRAMA3_MMAP_SHARED=1`
to use `MAP_SHARED`; default and rollback are `0` (`MAP_PRIVATE`). Both modes
keep `PROT_READ`, existing mmap owners, whole-shard no-copy Metal views, Shared
storage, tracked hazards, kernels, and serial queue ordering. This adds no CPU
activation handoff. Checkpoint files must remain stable and immutable until all
referencing contexts are terminal; shader `const` inputs provide the current
in-tree GPU read-only constraint. Raw public pointers/native bindings are unsafe
plumbing, not an enforcement boundary. Reset/unload before every referencing
context drains remains unsupported.

The predecessor binary reproduced 14.310 seconds from commit to GPU start,
followed by 0.173 seconds of GPU execution. The final diagnostic's four fresh
processes used one identical tiny kernel and five identical dispatches per
command: first waits were 0.881/0.817 ms for a 16 KiB view versus 518.645/
519.265 ms for the same file's 5.50 GB shard view. All terminal words matched;
GPU intervals were about 24–25 microseconds. File-cache state was uncontrolled.
This reproduces a large extent-associated delay without the backbone, tower,
or DAC; it does not assign the entire model delay to a particular driver
pinning, page-in, or deferred-compilation mechanism.

Two complete same-binary shared/private AB/BA pairs at graph depth two measured
11.00/20.97 seconds and 18.60/25.20 seconds with `/usr/bin/time` (process launch,
loading, preparation, generation, WAV/report output, and cleanup included).
These pairs used the earlier timed binary; the final build differs only in
formatter whitespace and was separately replayed below. Full-run cache and host
state were uncontrolled. Both pairs favor shared mapping, with ratios 0.524559
and 0.738095. The additional unpaired shared run took 9.89 seconds. All five
completed runs reproduced the same 17 frames, EOS, and exact WAV bytes, plus
152 control / 140,012 terminal
readback bytes; graph leases drained. Per-process peak footprint was
1.709 GB shared versus 1.709–1.778 GB private. This counter excludes or differently
charges some file-backed/Metal residency; it is not a complete physical-memory
budget. No RAM saving or stable speedup is established from these samples.

Three later full attempts were terminated by the unchanged 25% system-free
memory guard: one private, one shared, and the final shared graph-depth-two
smoke. They remain recorded as incomplete attempts and contribute no successful
timing/parity result. The host had unrelated active workloads; their processes
were not modified. The final owned-binary synchronous shared route at graph
depth zero completed with exact WAV/readback parity (11.08 seconds application
time). A separate profiled owned-binary run measured 2.079 seconds
commit-to-GPU-start and 0.159
seconds GPU execution, with 12.16 seconds full-process time. This is a separate
cache/host state from the predecessor 14.310-second observation; the pair is
not an isolated causal estimate. Fresh upstream replays passed 243 backbone /
497,664 values, 726 tower / 2,042,880 values, 20 strict-head / 77,824 values,
29 DAC internal stages, and 4,096 waveform values at the unchanged gates below.
All component traces are bitwise equal to the preceding native route. Metal
passed 63 examples at each of depths zero and two; CPU passed 20, Python
contracts six.
The new mapping guard covers both modes, invalid configuration/ranges, queued
weight consumption through GC, terminal drain, and unchanged fixture-file bytes.

Rejected experiments: CPU page touching took 24.02 seconds of setup plus
5.74 seconds of first wait, worsening the measured full probe. Per-tensor views
had full-process view/default ratios 1.029940, 1.045088, 0.887771, and 1.025412
(median 1.027676); three of four pairs regressed despite exact WAV/ledger parity.
The tensor-view code was removed. Its lower retained-byte counter was backing
extent bookkeeping, not measured physical RAM. A single MADV_RANDOM scratch
probe did not improve the preceding uncontrolled-cache probe; no inference
madvise change was retained.

Risk: CAUTION (file sharing and resource preparation). Capability DoD is the
executed mapped-input guards, pinned upstream replays, and exact native full
WAV/readback checks. Performance promotion additionally requires repeated
same-binary full-process AB/BA runs with sufficient host headroom and no guard
terminations, at graph depths zero and two. The strongest failure is an
unintended mapped weight write or owner release before completion. Current
read-only shaders, GC/drain guards, and exact file/output checks cover the
in-tree serial route; external mutation/concurrency remain outside admission.
Ordinary mmap selection is not an LTP/WBA transformation. No Collapse or default
speed promotion follows from the small timing sample; global memory admission
and repeatable complete-wall descent remain open.

Evidence, source identities, exact commands, aborted attempts, and refresh
rules: [startup evidence](../.landmark/evidence/drama3-startup.json).
Diagnostic source: [startup extent probe](../examples/drama3_startup_probe.cr).
Next falsifier: repeat matched full-process pairs after host headroom recovers,
separating file-cache state from mapping mode and recording the first-submit
interval alongside the complete timer. Refresh on source/reference, checkpoint,
precision, compiler, input, device/OS, or artifact-availability drift.

## Admitted CogniGraph handoff slice

Status: bounded CogniGraph route admitted as opt-in; speed promotion withheld.
`DRAMA3_COGNIGRAPH_MAX_INFLIGHT` defaults to `0` (legacy synchronous route);
`1` or `2` enables the bounded graph queue. DRAMA's `Resident::Context` wraps
CogniGraph's `GraphSubmissionQueue` over one explicit Metal queue. Buffer owner
identity is scoped to the context and inherited by slices; cross-context reads,
dispatches, and native bindings reject ordinary foreign views. This is not a
security boundary: code that reconstructs a `Buffer` from its exposed backing
handle/owner or performs raw external writes can bypass the tag. Treat the
constructor/backing handle as internal unsafe plumbing. The scoped safe route is
the serial single-context native CLI without external mutation, rebinding, or
concurrent context use. Leases retain temporary bindings through checked
completion. Directly bound persistent attention caches and mapped weights stay
rooted in the resident model and MetalBackend until completion; callers must
keep these owners and the Context alive until drained. Scratch allocations are
fresh; there is no recycling pool.

Graph mode submits prefill rows without immediate waits and carries each
request-owned generated-frame view directly from tower output into the next
backbone step and DAC. The next main-token read remains a CPU control boundary;
terminal codes and PCM are read once at the output boundary. EOS-first exits
before a generated row. The output view is range-checked against main, hidden,
and previous-frame inputs before dispatch; sampled codes must not alias the
frame destination. Finite checks, explicit diagnostic reads, synchronous
component profiling, and single-use model semantics remain. An encoding/GPU
failure drains or cancels known leases and poisons the context; this does not
undo model state already mutated by completed commands. Reset/unload is legal
only after every resident context is terminal. Depth bounds live leases, not
bytes per lease, and is not long-form memory admission.

Measured scope: same kernels and eight complete runs across four same-binary
depth-0/graph pairs produced exactly 17 frames, EOS, and WAV bytes in every run.
Ordinary command submissions and completion waits each fell from 90 to 73.
Readback ledgers stayed at 152 control bytes and 140,012 terminal bytes, with no
activation or diagnostic readbacks. Graph/legacy complete-wall-time ratios
were 0.870419, 1.048293, 0.980546, and 0.974302 (median 0.977424); one pair
was 4.8% slower. `/usr/bin/time` per-process peak footprint rose from
1.576–1.639 GB at depth zero to 1.709–1.765 GB with graph mode. These
observations do not establish a reliable whole-run speedup, and depth zero
remains the default.

Numerical replays pass at the pinned inputs: 243 backbone traces / 497,664
values (gates 6e-5 / 3e-4); 726 tower traces / 2,042,880 values (6e-4 / 3e-4);
and 20 strict-head traces / 77,824 values (3e-5 / 3e-4). Codec gates cover 29
internal stages (max absolute error 1e-3, RMS 2e-4, normalized RMS 2e-4);
4,096 waveform values use the per-sample absolute-error gate
`abs(error) <= max(5e-5, abs(reference) * 1e-3)`. The final owned-binary
depth-0/2 replays passed with exact WAV bytes; all three upstream trace sets
match and intermediate values are bitwise equal to the previous native path.
The full Metal suite passed at depths zero and two (60 examples each, no
failures, errors, or pending examples); the CPU suite passed 18 examples and
the Python contract suite passed six tests. These results are recorded in the
evidence packet below.

The retained-byte high-water of 31,903,032,324 is a conservative sum of whole
backing allocations per live lease: mapped dense shards and shared allocations
can be counted more than once, packed weights bound directly by native methods
are omitted, and the number is not RAM usage. The per-process peak ranges above
are `/usr/bin/time` observations; the 40,960 MB process-tree RSS limit and 25%
minimum-free-memory floor are separate run guards.

Risk: CAUTION (submission ordering, ownership, and memory). Rollback is
`DRAMA3_COGNIGRAPH_MAX_INFLIGHT=0`; context failure is terminal. The local
mechanism evidence is bounded: a Spike found and removed redundant row waits;
the Ladder queue passed lifecycle/parity replays. LTP/WBA Collapse and default
speed promotion remain withheld because whole-run ratios are inconsistent and
measured peak footprint increased. The audit coordinates are ordered as
`(semantic/ownership error, undrained lease, memory-guard breach, full-corridor
wall time, avoidable waits)`; these measurements establish a lower final
coordinate, not global potential descent. The transport corridor is one FIFO
Metal queue; the Ladder move submits fresh/persistent owners, passes resident
outputs to dependent work, waits at control/terminal boundaries, then releases
only completed leases. Failed work drains/cancels and poisons the context; it is
not retried against mutated state. Depth zero is the dual frame. The first-submit
extent probe and opt-in shared-mapping results are recorded
in the active slice above; stable performance and global memory admission
remain separate from the admitted correctness and wait-count result.

Evidence and exact run inputs: [CogniGraph evidence](drama3-cognigraph-evidence.json).
Source anchors: [resident ownership/queue](../src/ml/audio/drama3/resident.cr),
[generator handoff](../src/ml/audio/drama3/generator_resident.cr),
[tower output view](../src/ml/audio/drama3/tower_resident.cr), and
[CogniGraph queue](../src/ml/metal/compute_graph.cr). Refresh parity, ownership,
readback, and matched timing claims on source/reference, precision, asset,
compiler, input, device/OS, or evidence-availability drift.

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
precision). Rollback: `DRAMA3_BACKEND=cpu`, or the preceding resident `b91ad1e7`.
The highest-risk failure is altered routing/cache arithmetic hidden by a
plausible waveform. DoD: focused primitive/cache/failure tests, actual
backbone/tower/DAC gates at their existing tolerances, complete text-to-WAV,
call-site audit plus readback accounting, and matched component/total timing.
The executed evidence is indexed in
[drama3-resident-evidence.json](drama3-resident-evidence.json), with the current
latency changes in
[drama3-resident-speed-evidence.json](drama3-resident-speed-evidence.json).

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

- 42 focused Metal examples pass; 18 CPU examples and six Python contracts
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

The sampler implements descending ranking, lower-ID ties and the upstream
top-p crossing rule in FP32. The later sampler audit above finds that upstream
CPU unstable-sort tie membership differs; universal upstream tie equivalence
is not supported. A precision
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

### Measured M2 Max latency at resident baseline `b91ad1e7`

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

### Packed rank batching and coalesced Delta state

The current resident backbone batches independent selected expert ranks into
one dispatch per packed projection. Gate/up share the input vector; down reads
a separate intermediate per rank. Quantized weights remain packed, router IDs
remain on GPU, and affine-group arithmetic retains its original order. A final
kernel adds ranks in their original order, then the existing shared expert is
added. This reduces the rank-dependent dispatches from 49 to 6 per MoE layer.
Dense fixture matrices retain the scalar expert route.

The private Delta state now stores `[value_head,key,value]`, making neighboring
value lanes contiguous. Each lane still processes keys in exactly the same
order. The state starts at zero and has no external layout contract. Both
choices are frozen at backbone initialization; setting `DRAMA3_MOE_BATCH=0`
and `DRAMA3_DELTA_TRANSPOSE=0` restores the previous kernels in the same binary.
Both default to 1 and reject values other than 0 or 1.

Risk: CAUTION (performance and recurrent layout). The baseline is `b91ad1e7`;
the rollback flags above or CPU restore the earlier routes. The strongest
falsifier is a kernel-only gain that disappears in complete text-to-WAV or
changes routing, cache continuation, codes, or the readback boundary. Promotion
requires the existing upstream gates, exact previous-native replay, focused
failure/cache checks and matched completed inference timing. Heavy probes run
serially under `scripts/run_safe.sh`; other processes are preserved.

Four same-binary fresh-process AB/BA pairs on M2 Max, `Hello.`, seed 0,
23 prompt rows and 17 generated frames through EOS, give:

| Completed interval | Median optimized / previous ratio | Time reduction |
| --- | ---: | ---: |
| Autoregressive backbone, 17 rows | 0.8812 | 11.9% |
| All GPU command intervals | 0.9438 | 5.6% |
| Complete inference through WAV writing | 0.9611 | 3.9% |

These are medians of four paired ratios, not ratios of aggregate medians.
Previous complete runs took 22.3971–23.6972 s; optimized runs took
21.5529–21.7429 s, with every pair improving. Both arms use identical synchronous
profiling and GPU-timeline flags (129 commands). Preparation and completion
remain inside the original sidecar timer. One extra ordinary unprofiled pair
(90 commands) took 23.8942 s previous / 19.3659 s optimized, with identical
output; this single pair does not establish a general 19% gain. Host load was
not controlled, and these are bounded local observations.

The isolated actual-shape Delta benchmark uses two warmups and seven AB/BA
pairs of 30 dispatches. Its median GPU ratio is 0.6969 (30.3% less kernel time),
with every final state/output bit equal. This is a small part of backbone
runtime, not a 30% whole-model speedup. Reproduce it with
`examples/drama3_delta_benchmark.cr` and the normal Metal build flags.

All 243 backbone traces, 726 tower traces (including 20 strict terminal gates)
and 30 DAC stage/waveform traces pass the unchanged upstream tolerances and
match the previous native values bitwise. Every full run has the same prompt,
codes, EOS and WAV bytes (SHA-256
`3bcc331bd234ba6a3950d2122215bdc663aa4f7427a112e8f8df108085908a49`).
The normal readback ledger remains 152 control / 140,012 terminal bytes, with
no activation/diagnostic reads. Full MLX PCM replay remains within one count.
New regressions compare Q4/Q8 shared/per-rank projections at unaligned offsets,
duplicate ranks, group tails, invalid IDs/widths, rank-sum cancellation, and
every Delta state/output bit across varied recurrent rows. The focused suite
passes 42 Metal / 18 CPU examples and six Python contracts.

The first submission remains the dominant open latency target. One instrumented
baseline command spends 10.9795 s before GPU start and 0.1348 s on GPU; all
library/function/pipeline API calls together account for about 13 ms. These
timestamps do not distinguish deferred driver work, resource preparation,
pinning, or page faults. No weight-view, hazard-tracking, scheduling or timer
hoisting change is promoted. Optional `DRAMA3_METAL_TIMELINE=1` emits GPU command
intervals; `COGNI_METAL_SUBMIT_PROFILE=1` adds the existing bridge timestamps.
First attribute this pre-GPU interval before choosing another startup change.

Correlated source adversary review is ROBUST within the validated matrix,
layout and arithmetic bounds; executed replay supplies numerical evidence.
The source/binary/input identities, per-pair observations and refresh policy
are in [drama3-resident-speed-evidence.json](drama3-resident-speed-evidence.json).
Refresh on the source, precision, assets, compiler, input, device/OS or artifact
availability changes listed above. Long-form, concurrent/reset, stock BF16,
subjective quality, realtime and general cross-device speed remain unclaimed.

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
