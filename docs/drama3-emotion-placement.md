# DRAMA-3 emotion placement and control paths

Status: the eight exploratory native/MLX generations and terminal diagnostics
completed at `51789b8310812a0ebe4b681556bbc468fb15420a`. The unique cause of
silent continuation and excessive pauses remains IN_PROGRESS. Production
arithmetic, precision, tolerances, defaults and read boundaries are unchanged.

## Measured result

The user reports that the final phrase is audible in the new native Russian
end-whisper WAV, but not in its English counterpart. The exact reply is
“Слышна только в русском”. This is a report about those two native final
phrases, not all words, emotion quality, the MLX files, or language prevalence.
`human-listening.json` binds the observation to the two WAV hashes.

In the English end-whisper pair, all 512 reported frames match native/MLX
exactly, including the main token and all ten DAC books. Both decoders produce
a low-level tail from 6.5 to 23.777 s: every half-second window, including the
last partial window, peaks at 9–22 PCM16 units. Both runs reach the 512-frame
cap without EOS. A native-versus-MLX sampled-code divergence is therefore not
required for this particular English quiet output. This does not establish
intermediate floating-point parity or original Fish text-generation behavior.

| Prompt | Frames native / MLX | Duration native / MLX, s | EOS native / MLX | Code comparison |
| --- | ---: | ---: | --- | --- |
| Russian, emotions, whisper last | 512 / 494 | 23.777 / 22.941 | no / yes | first difference at zero-based frame 262; first 262 frames exact |
| English, emotions, whisper last | 512 / 512 | 23.777 / 23.777 | no / no | all 512 frames exact |
| Russian, same spoken body, no cues | 463 / 463 | 21.502 / 21.502 | yes / yes | all 463 frames exact |
| English, same spoken body, no cues | 239 / 239 | 11.099 / 11.099 | yes / yes | all 239 frames exact |

Prompt IDs match exactly in all four pairs. The Russian end-whisper pair also
differs in length from frame 494 onward; equal common-prefix length is not
whole-stream agreement. Neither end-whisper Russian WAV has a terminal run
of half-second windows at peak <=64. The no-cue Russian WAV has only a
1.678 ms zero-valued final partial window, not a prolonged quiet tail. The
no-cue English WAV has no such terminal tail. These signal statements do not
certify any phrase or emotion in the controls.

Internal pauses and terminal failure are distinct. Even the Russian no-cue
pair contains complete low-level half-second windows at 3–5, 7–9, 11.5–13.5,
and 15.5–18 s; each window peaks at two PCM16 units in both decoders. These
contained intervals are measured from the existing GPU window reductions,
not sentence alignments or precise speech boundaries. Removing the emotion
cues did not remove all long pauses in this one sample. No causal attribution
to whisper or emotion tags alone follows. In the Russian emotion sample,
the windows at 6.5–10.5 s instead peak at 128–142 units; low-level residual
noise can keep a pause above the explicit <=64 diagnostic threshold.

## What the code controls

### Emotions are learned text cues

Original Fish does the same: its prompt builder validates structural and
speaker tokens, then tokenizes caller content without an emotion grammar
(`fish_speech/models/text2semantic/fish_omni/generate.py:68–90,168–185`).
The pinned MLX prompt builder tokenizes the caller's text without interpreting
an emotion grammar (`mlx_audio/tts/models/fish_omni/prompt.py:149–165`). Its
paired-XML/bracket regex (`:53–56`) protects spans from text chunking; it does
not execute whisper transitions or assign a duration. Neither the native
prompt builder nor the generation loop implements a whisper mode/reset.

The actual checkpoint tokenizer encodes `[whisper]` as
`[58, 1266, 26483, 60]`, four ordinary BPE tokens. `[angry]`, `[sad]`,
`[laughs]`, `[pause]`, `<whisper>` and `</whisper>` are also ordinary text,
not added control tokens. They can still influence the model through learned
conditioning; ordinary text is not inert. `<|speaker:0|>` is instead one
special token, ID 248083. The tokenizer probe and inspected-source hashes
are retained in `code-control-audit.json`.

Fish documents bracket cues including `[whisper]`, `[angry]` and `[laughing]`
in its [pinned README](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/README.md).
Its [DRAMA-3 model card](https://huggingface.co/fishaudio/drama-3) demonstrates
point cues `[sighs]` and `[laughs]`. This experiment uses brackets. Prior XML
samples remain a different prompt family; no parser-defined closing/reset
semantics were found. A successful XML sample would not prove such semantics.

### Pauses are generated acoustic time

The current native entry point builds one prompt, generates its full code
stream, and calls DAC once (`examples/drama3_generate.cr:45,71–80`). The MLX
controls likewise call `generate_codes` and `decode_codes` once without
`--chunk-text` (`scripts/drama3/generate_mlx_control.py:1239–1297`). All eight
WAVs have exactly 2048 samples per emitted frame at 44.1 kHz, about 46.44 ms
per frame. The WAV writer serializes that waveform without inserting gaps
(`src/ml/audio/drama3/wav.cr:7–28`). Computation wait time cannot create
silence inside these offline WAV files.

Neither selected generation loop schedules a fixed gap after punctuation or
uses a sentence duration controller. They sample acoustic frames until EOS
or the frame cap. Punctuation and cues can condition learned prosody; the
code does not promise a pause length in seconds. A `[pause]` string is not an
implemented timed pause API. The exact no-cue Russian code trajectory and
matching quiet windows in both decoders show that a native-only scheduling
or WAV-stitching explanation is unnecessary for those pauses. A shared
model, conditioning, numerical or codec defect is not ruled out.

Original Fish likewise stops on selected main-token EOS or its frame cap,
and decodes all returned code frames without an energy-based trim
(`fish_speech/models/text2semantic/fish_omni/generate.py:252–286`,
`fish_speech/inference_engine/vq_manager.py:16–24`). Repetition-aware sampling
is not a silence detector: native and MLX select one retry when the first
semantic main token appears in the previous 16 main IDs. The retry is accepted
without another uniqueness check (`src/ml/audio/drama3/kernels/generator_resident.metal:36–52`,
`fish_omni.py:190–210`). This permits repeated IDs but is not evidence that
repetition caused quiet output. The saved selected IDs do not identify which
steps selected the retry; a matched-logit/candidate replay would be required.

Native checks EOS before residual-code sampling, while MLX checks after
sampling/evaluating the residual heads. Both omit the EOS frame. This work
difference can matter for cross-chunk random-draw accounting, but does not
explain pauses inside these standalone code streams. English end-whisper
has no EOS at all before its 512-frame cap; the quiet tail consists of emitted
frames, not waiting for model computation. A larger frame bound alone would
not establish that an omitted phrase is restored.

### Speaker identity needs audio conditioning

The MLX API accepts reference audio plus its transcript, or pre-encoded
reference codes plus text. Reference text and codes enter the system prompt;
history adds earlier user/assistant text and acoustic-code turns. The speaker
tag links turns to a participant; its number is not a preset voice name
(`prompt.py:123–153`, `fish_omni.py:343–385`).

The pinned MLX `Model.generate` discards its `voice` argument and arbitrary
`kwargs` (`fish_omni.py:348`). Passing a generic voice name or an unimplemented
speed keyword to that method does not select a voice or change speech rate.
The operative path is `ref_audio/ref_text` or `ref_codes/ref_texts`. The source
has no explicit speed/pause parameter in this method.

Our current native prompt builder/CLI supports text-only speaker 0 and has
no reference-audio, reference-code or history input (`src/ml/audio/drama3/prompt.cr:53–65`,
`examples/drama3_generate.cr:17–26`). All eight current controls omit voice
references. Changing a seed can change a generated voice and timing, but
neither a seed nor a speaker tag guarantees a stable timbre across text or
languages. Reference conditioning is a useful untested discriminator; its
absence is not yet the measured cause of the pauses or missing phrase.

Original Fish exposes the same audio-conditioning path: its VQ manager
encodes the reference waveform, and the prompt builder inserts the transcript
and ten-book codes into the system message
(`fish_speech/inference_engine/vq_manager.py:26–56`,
`fish_speech/models/text2semantic/fish_omni/generate.py:139–154`). This is
in-context voice cloning, with no optimizer or model-weight update on this
inference path. Each reference transcript can carry its own canonical
`<|speaker:N|>` tag. For these DRAMA-3 prompt builders, the tag carries the
role association; an untagged reference transcript defaults to speaker 0.
The serving API can assign that tag through its reference `speaker` field
(`fish_speech/inference_engine/__init__.py:72–96`). The new text selects
participants with the matching tags. Up to 16 tags are accepted
upstream; the native slice currently accepts only speaker 0.

The [official inference guide](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/docs/en/inference.md)
shows `--prompt-audio` with `--prompt-text`, and the API can retain an audio/
transcript pair under a `reference_id`. A saved ID names that reference data,
not a different model checkpoint. The model card demonstrates cloning from
one 3.9-second reference; this is an example, not a minimum/maximum duration
contract. Exact voice similarity, language transfer and reliable pauses
remain empirical properties. Natural-language descriptions can condition
style, but the inspected code provides no deterministic mapping from a
description to a fixed identity. Seed reproducibility is not an identity
guarantee, as the inference guide explicitly notes.

For a native implementation, the smallest first boundary is a verified
pre-encoded reference plus transcript and the corresponding prompt rows;
porting the reference-audio encoder is a separate boundary. The resident
generator already consumes multimodal prompt rows, but reference-prompt
construction, capacity checks and intermediate parity still need explicit
implementation/validation. Encoded references can be reused; a GPU-resident
integration must avoid copying their intermediate activations through CPU.
This is a proposed path, not a claim that native cloning is implemented.

## Exact inputs and control scope

Russian end-whisper prompt:

```text
[excited] Ура, ты вернулся! [angry] Почему молчал? [sad] Я боялась тебя потерять. [laughs] Теперь всё хорошо! [whisper] Только никому не говори наш секрет.
```

English end-whisper prompt:

```text
[excited] Hooray, you're back! [angry] Why didn't you call? [sad] I was afraid of losing you. [laughs] Now everything is fine! [whisper] Just don't tell anyone our secret.
```

The plain variants remove all bracket cues, preserving the complete spoken
body and punctuation. All four inputs are under 300 UTF-8 bytes. Settings:
seed 1, temperature 0.8, top-k 30, top-p 0.9, maximum 512 frames. No reference
voice, previous take, text chunking or gain/time/sample-rate adjustment.
This is one seed per input, not a language or seed-distribution benchmark.

Native uses resident Metal, depth-2 CogniGraph, MAP_PRIVATE, tree dense
reductions and the existing native expert-down path. MLX source revision is
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`; converted checkpoint revision is
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`. Its established shardwise F32-core
loader preserves packed integers; its DAC retains checkpoint dtype. Sampling
uses the shared-uniform inverse-CDF shadow, not stock MLX RNG. Original Fish
full text generation was not run. No reference precision was changed here.

The first 5632 F32 uniforms (512 frames x 11 draws) freshly match the prior
seed-1 tape prefix, SHA-256
`ddc31bee6c0ffcd12c2ba781207747cad5eab9cbcb02faa074658caac8e16341`.
The draw order is main, unconditional main retry, nine residual books.
Matching draws does not imply matching logits. Full hashes of all nine cached
checkpoint files were recomputed before generation. The manifest identifies
local bytes, not a new remote revision attestation.

Model arithmetic remains on Metal/MLX GPU. Existing native EOS/status control
reads and graph drains remain; the MLX official per-frame EOS and decoded
length scalar reads remain. There are no new per-step activation reads.
Completed PCM and code reports upload once for GPU comparisons/reductions,
followed by terminal reduced-metric serialization. Tokenizer/source inspection
is CPU preparation, not model inference. Sequential `run_safe` jobs use the
authorized 15% reserve; unrelated processes are untouched. Timings include
loading/diagnostics on an uncontrolled shared host and are not speed claims.

## Evidence, limitations and return frontier

Raw files are outside Git in
`/Users/sergey/.cache/cogni-ml/drama3-runs/emotion-whisper-end-20261015/`.
`summary.json` contains all GPU window metrics and code comparisons;
`commands.json` records 12 completed build/hash/generation/analysis/validation
commands, their exact argv/environment, source/log hashes and exit status.
`validation.json` passes provenance and structural binding checks for eight
WAVs and four exact prompt pairs. It checked the 11 receipts available before
its own completion; its completed receipt is the twelfth entry. A binding
check is not proof of runtime semantics, words, emotions or numerical parity.

Executed diagnostic DoD, with the receipt environment and `OUT` set to that
raw directory and `PY` to the reference venv Python:

```sh
scripts/run_safe.sh "$PY" 240 2048 "$OUT/analyze.py" ru-end en-end ru-plain en-plain
scripts/run_safe.sh "$PY" 120 2048 "$OUT/validate.py"
```

Both exited 0 with the results above. The analyzer refuses to overwrite its
summary; preserve the existing receipts when reproducing. Exact generation
commands are retained rather than reconstructed from a prose recipe.

The original plan also defines `ru-early` and `en-early`; neither ran. Those
variants swap both `[sad]` and `[whisper]`, and their post-whisper words also
carry other emotions. The plan's `whisper_exit` frequency prediction is
untested and cannot be inferred from the executed end/plain arms. Ending
with whisper supplies no post-whisper continuation, and removing all cues
does not isolate whisper from the other emotions.

The strongest next discriminators are: (1) one spoken body with no other
styles, comparing no whisper / whisper before an early clause followed by
untagged speech / whisper only before the last clause; (2) the same plain
text and matched sampler inputs with and without one clean, pre-encoded
voice reference, then additional seeds. These are separate interventions,
not an invitation to change multiple controls in one causal comparison.
They remain unexecuted. Do not add silence trimming as a causal fix: it could
hide omissions or remove quiet speech while leaving generation unchanged.

Keep the strict DAC RoPE-coefficient replay and code-fork/MoE39 investigations
open separately; exact sampled codes in three pairs do not close them.
The curated index is `.landmark/evidence/drama3-emotion-placement.json`.
Refresh on source/checkpoint/tokenizer/sampler/conditioning/input/WAV,
runtime/device/OS, precision/layout/gate or bound listening-evidence drift.
Raw cache retention is local, not remote durability. Reverting this report
commit changes no production behavior.
