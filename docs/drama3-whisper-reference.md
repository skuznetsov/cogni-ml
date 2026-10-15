# DRAMA-3 whisper continuation with a reference

On 2026-10-15, four native Metal runs and four pinned MLX controls repeated
the whisper-then-laughter text in Russian and English, with one fixed public
reference pair or without a reference. The Russian reference pair has no
qualifying long quiet tail and its unprompted ASR contains the requested
post-whisper continuation. The other three conditions retain long quiet tails
in both implementations. This is a measured counterexample to a universal
reference-based remedy; the cause of the missing continuation remains open.

The scope is one seed, one reference audio/transcript pair, and these two
texts. Reference conditioning changes the complete prompt, including its
transcript and length. The experiment does not isolate an audio-only effect,
measure language prevalence, or certify emotion delivery or voice resemblance.

## Inputs and controls

The Russian target is the exact historical whisper prompt:

```text
<whisper>Тише, кажется, за дверью снова шорох. Давай подождём немного и послушаем, что происходит.</whisper> [laughs] Ах, это всего лишь наш кот уронил коробку! Ну и напугал же он нас сегодня, маленький проказник.
```

Its UTF-8 SHA-256 is
`2fc15bc52126b289d355d7cf9876a753229096233cd41acabee16ea4ebb8fe77`.
This is a new controlled run, not a claim of bytewise replay of historical
artifacts whose reports did not contain the current sampler-input digest.

The English target is a new translation with the same cue family and order:

```text
<whisper>Quiet, I think there's a rustle behind the door again. Let's wait a little and listen to see what's happening.</whisper> [laughs] Oh, it was just our cat knocking over a box! He really startled us today, the little rascal.
```

Its UTF-8 SHA-256 is
`cf81a0e664c1d68f069edb92de6c25b54021dfe7ecfab6020bd126ef8f0dbdfa`.
The cue strings are learned text inputs, including the closing XML-like tag;
the pinned implementations have no explicit whisper-state reset parser.

The reference is the public 3.904-second English excerpt used by
[the original model card](https://huggingface.co/fishaudio/drama-3), with transcript:

> We asked over twenty different people, and they all said it was his.

The canonical mono PCM16/44.1 kHz WAV has 172,167 samples and SHA-256
`a24ee1bf75c1a2bbe414e4ee9d41151b1e13bdff663c35434e8f86baa9b0a0c3`.
It is not a recording of the user. Input conversion was performed once in the
earlier reference experiment; its preparation receipt is retained there.

All eight generations use seed 1, temperature 0.8, top-k 30, top-p 0.9,
and a 512-frame cap. Native runs use private mmap and CogniGraph depth two;
MLX controls use the pinned MLX runtime. Native runs hash the actual
5,632-element little-endian F32 sampler array before its single upload.
That digest matches the native producer probe and MLX tape prefix:
`63e955c00512b965c17303e4c2517ed4eeb093193af6c881542045c0c7ece7a1`.
The full tape digest is
`ddc31bee6c0ffcd12c2ba781207747cad5eab9cbcb02faa074658caac8e16341`.
The MLX control uses the existing shared-uniform inverse-CDF shadow sampler,
not stock MLX random draws. Equal uniforms do not imply equal outputs after
reference conditioning changes the logits.

The upstream control is clean MLX-audio
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, MLX 0.31.1, with
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
The F32-core control retains packed quantized weights; codec weights retain
their checkpoint dtype. Original Fish full text generation is not covered.
Fresh hashes bind all nine checkpoint files, production source, reused native
binaries and their embedded shaders. The binaries were verified against the
current production files, not rebuilt for this diagnostic-only change.

## Generation and quiet windows

Frames contain one main token and ten codebook IDs. A cap is an incomplete
utterance, even when the entire returned waveform was successfully decoded.

| Condition | Frames native / MLX | Termination native / MLX | WAV duration native / MLX | Qualifying terminal quiet windows native / MLX |
| --- | ---: | --- | ---: | --- |
| RU, reference | 347 / 347 | EOS / EOS | 16.115 / 16.115 s | none / none |
| RU, no reference | 489 / 512 | EOS / cap | 22.709 / 23.777 s | 9.5–22.5 / 9.5–23.5 s |
| EN, reference | 512 / 512 | cap / cap | 23.777 / 23.777 s | 7.5–23.5 / 7.5–23.5 s |
| EN, no reference | 390 / 390 | EOS / EOS | 18.112 / 18.112 s | 7.0–18.0 / 7.0–18.0 s |

The unchanged rule uses complete 0.5-second windows, peak at most 64 PCM16,
adjacent windows merged and a minimum run of one second. Partial final windows
are excluded from these ranges; all three quiet conditions also have a
qualifying partial final window. These are low-amplitude intervals, not exact
digital zeros or verified speech boundaries.

RU without reference additionally has internal quiet runs at 0.5–1.5 and
3.5–5.0 seconds in both implementations. The other conditions have no
qualifying internal runs. An isolated half-second quiet window in RU with
reference is below the one-second minimum. WAV duration is not inference time;
the phase timings in the receipts are exploratory on a shared host.

| Condition | Exact leading frames, all 11 IDs | First codebook / main mismatch, zero-based | Complete returned stream equal? |
| --- | ---: | --- | --- |
| RU, reference | 347 | none / none | yes |
| RU, no reference | 448 | 448 / 448 | no; also different lengths |
| EN, reference | 28 | 28 / 30 | no |
| EN, no reference | 390 | none / none | yes |

All integer comparisons run on GPU. Exact sampled-token parity does not imply
bitwise waveform parity or close the existing DAC intermediate numeric gate.
RU no-reference divergence occurs after its measured quiet onset. EN without
reference reproduces the long tail with every returned token equal. A native
token divergence is therefore unnecessary for these measured quiet onsets.
The EN-reference fork remains a separate numeric/sampling frontier.

## Intermediate upstream gates

Each language has fresh reference encoder/prompt and first-prefill fixtures,
then native probes at graph depths zero and two. All four probes pass:

- All 102 encoder/VQ boundaries pass unchanged max absolute error `1e-3`,
  RMS `2e-4` and normalized RMS `2e-4`. Largest stage absolute error:
  `0.00062561035`.
- All 850 reference code IDs match exactly. The complete reference-conditioned
  prompt matches exactly: RU 207 by 11, EN 201 by 11.
- First-prefill hidden and main logits have zero pointwise failures under
  `3e-5 + 3e-4 * abs(reference)`. Hidden/logit maximum absolute errors are
  `4.7683716e-6` / `2.861023e-5` for RU and
  `2.682209e-6` / `2.4795532e-5` for EN.
- Stage, prompt, prefill and gate metrics are exactly equal between depths
  zero and two for each language. Terminal diagnostic reads add no graph
  submissions or waits.

This is an encoder/prompt/first-prefill gate, not a claim that every generation
step or every DAC intermediate passes. Generation preserves the resident
encoder → VQ → prompt → prefill → generated codes → DAC path. Host WAV parsing,
text tokenization, the initial sampler upload, existing EOS/control reads,
bounded CogniGraph waits and terminal outputs remain. There is no intermediate
reference-code/activation readback or new handoff drain. Diagnostic comparisons
read reduced results after completion. No precision or tolerance was widened.

## Unprompted ASR observations

Pinned `mlx-community/whisper-large-v3-turbo-asr-fp16` revision
`624c19c9af5603fa73b83bce14d4aeea96156d18` transcribes all eight complete WAVs
on GPU, with auto language detection, temperature zero, no expected transcript,
initial prompt or hotwords, and previous-text conditioning disabled. The human
operator knows the cases; this is not an investigator-blinded study.

Both RU-reference transcripts include the continuation through “маленький
проказник”. Both RU-no-reference transcripts recognize the whisper body through
“что происходит” and add an unrequested “Продолжение следует...” segment whose
reported end, 38.88 seconds, exceeds either source clip. All four EN transcripts
contain only the requested text before `</whisper>`. ASR does not certify that
laughter occurred or precisely locate the whisper transition.

The source-reference positive control recognizes its expected words without
being given them. The ten-second digital-zero negative control nevertheless
produces “Thank you.” and a 29.98-second segment. Consequently neither ASR text,
its confidence fields nor its segment times are a reliable speech-content
gate. The eight WAVs and their hashes are bound to the amplitude reports;
transcripts are fallible corroboration for listening, not a substitute for it.

PCM16 normalization, 44.1→16 kHz polyphase FIR, model features and inference use
MLX GPU arrays. The host parses the WAV container and constructs static filter
taps; no source-waveform CPU DSP is used. Word alignment is disabled. Diagnostic
waveform hashes/statistics are read only after inference. Five synthetic edge,
impulse and ramp cases compare the actual Metal resampler with a pinned SciPy
oracle at predeclared `rtol=2e-5`, `atol=2e-6`; all pass, maximum absolute error
`4.172325134277344e-7`. This diagnostic-only gate does not replace any model gate.
The initial shader compilation failed because `half` is a reserved Metal type.
Only that local name changed to `half_len`; the failed log, original helper
hash, exact-source reconstruction check and successful replacement are retained.

## Reproduction and continuation

The curated index is `.landmark/evidence/drama3-whisper-reference.json`.
Local raw artifacts are under
`/Users/sergey/.cache/cogni-ml/drama3-runs/whisper-reference-20261015`:
`plan.json`, `pipeline.py`, `postprocess.py`, `run.py`, `commands.json`,
`source-binding.json`, `measurements.json`, `asr.json`, `asr_probe.py`, and the
`ru/` and `en/` waveform/report/fixture directories. The collector validates
actual text, sampler, WAV, log and ASR-helper hashes. Preserve this completed
directory; use fresh outputs for reruns. Its plan uses the word “blinded” for
ASR; the executed scope is the recognizer-input withholding described above.

Serialize model jobs through `scripts/run_safe.sh`, with the admitted 15%
free-memory floor. The ledger preserves all executed arguments and environment,
including 22 successful commands and the resolved shader-compile failure.
`pipeline.py` generates the matrix and four upstream probes. The completed
postprocess stages compare both language directories and run the GPU ASR
controls. The retry command names and receipts remain explicit in the ledger.

For a new comparison, `scripts/drama3/compare_reference_pauses.py --language
ru|en --run-dir ... --random-tape ... --tape-proof ... --encoder-report ...
--output ...` selects the report filename namespace. The language switch does
not classify speech; the plan and UTF-8 digests bind actual text. The analyzer
now accepts the producer's `max_frames` status and legacy `frame_cap` spelling,
while retaining strict flags, counts, uniform use and waveform geometry.
`scripts/drama3/prefill_reference.py` accepts a captured step-zero prefill from
a strictly validated capped source, preserves raw termination, and labels the
source utterance `incomplete_at_frame_cap`. It does not promote a cap to EOS.

The executed CPU DoD is:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m unittest \
  spec.drama3_voice_control_spec spec.drama3_reference_voice_spec \
  spec.drama3_prefill_reference_spec spec.drama3_reference_pauses_spec
```

All 51 tests pass, including malformed caps, EOS/cap conflicts, mixed filename
namespaces, and mismatched text/conditioning bindings. No production inference
arithmetic changed. Rollback is the single tooling/report commit; cached WAVs
are not committed. [Personal voice input](drama3-reference-voice.md#using-your-own-voice)
already uses the native reference API, without training model weights.

Mini-Quadrumvirate review challenged seed-as-input equivalence, cap-as-EOS,
amplitude-as-transcript, reference-as-audio-only causality and first-prefill-as-
full-generation parity. These distinctions remain explicit. Review is correlated
same-lineage evidence, not an independent replication.

Next numeric discriminator: replay the common EN-reference prefix and inspect
the first differing tower/codebook decision at frame 28 with the same uniforms;
main-token divergence follows at frame 30. Separately vary only whisper cue
placement/syntax on fixed conditioning to test continuation behavior. Existing
MoE-39 and DAC floating-point frontiers remain open. Refresh affected claims on
source, checkpoint, sampler, text/cues, reference pair, artifact, runtime,
device/OS or gate changes; a failed refresh reopens the claim.
