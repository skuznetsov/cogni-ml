# Native DRAMA-3 reference voice

The resident Metal CLI accepts one reference waveform and its matching
speaker-0 transcript. The reference conditions the generated voice through
the same prompt representation as the pinned MLX port. Python, MLX and an
external encoder are not part of native inference.

```sh
DRAMA3_BACKEND=metal DRAMA3_COGNIGRAPH_MAX_INFLIGHT=2 \
  /path/to/drama3_generate --model "$MODEL_DIR" \
  --reference-audio /path/to/reference.wav \
  --reference-text 'Exact words spoken in the reference.' \
  --text 'The new words to say.' --seed 1 --max-frames 512 \
  --output /path/to/new-utterance.wav
```

Build `examples/drama3_generate.cr` with `--release -Ddrama3_metal` and the
existing native Metal bridge. Both reference arguments are required together.
The first supported input boundary is mono, uncompressed PCM16 at 44.1 kHz,
non-empty and at most ten seconds, with at most 1 MiB including metadata.
Unsupported layouts/rates, malformed RIFF chunks, a partial reference pair,
and prompt/cache overflow fail before weight loading. The CLI does not resample.
The existing no-reference invocation remains available.

## Using your own voice

Record a short, clean excerpt of your normal speaking voice, without music,
overlapping speakers or effects. Export it within the input boundary above:
mono PCM16 WAV at 44.1 kHz, up to ten seconds. Supply the exact words spoken in
that excerpt as `--reference-text`; `--text` contains the new utterance to
generate. The example command above applies directly to a personal recording.

This path conditions the model in context using reference audio codes and their
transcript. It does not train weights or save a standalone speaker embedding.
Reuse the same reference pair for subsequent requests. Voice resemblance and
style consistency require listening evaluation; exact encoder/prompt parity
alone does not establish them. The experiment below uses a public reference,
not a recording of the user.

The subsequent [bilingual whisper-continuation experiment](drama3-whisper-reference.md)
reuses this public reference with matched sampler inputs. The Russian reference
condition has no quiet run of at least one second and ASR includes the requested
continuation; English quiet tails remain. These are bounded measurements,
without a general silence-fix claim.

## Device boundary

The host reads one bounded WAV snapshot, hashes that snapshot, widens exact
signed PCM16 integers to F32, and tokenizes text. One upload starts the device
path: normalization by 32768, causal DAC encoder, semantic/residual vector
quantizers, prompt audio rows, backbone prefill, autoregressive generation,
and DAC decode. Encoder codes feed the prompt kernel directly; prefill consumes
rows of that resident prompt in the same context. There is no encoder-code or
activation readback, or extra drain at either handoff.

CogniGraph still performs bounded command-buffer completion waits, including
at depth zero. Existing EOS/finite-control reads and terminal output reads
remain. Diagnostic probes enqueue comparisons on GPU and read reduced metrics
after completion; production generation does not enable those observers.
This is resident buffer handoff, without an LTP/WBA promotion claim.

The encoder uses F32 checkpoint weights and loaded rotary coefficients. Its
dense, convolution, RMS, SiLU and causal attention kernels preserve the pinned
MLX arithmetic geometry where the stricter probes required it. Attention has a
specialized 64-channel path for this checkpoint; the general fallback is retained.
No precision, tolerance, sampling setting or existing decoder arithmetic was
relaxed to admit reference support.

## Upstream verification

The source base is `04b254dd0d6344e58cec9a285f72c0cbfb7dba36` plus this feature.
The control uses clean MLX-audio
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, MLX 0.31.1, and converted model
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
The codec SHA-256 is
`543445a2ebc5224d2af5956d422f88c2cd21f4018c1a26d81b51ec231bb630f2`.

On Apple M2 Max, the full reference probe passes at graph depths zero and two:

- All 102 encoder/VQ boundaries pass max absolute error `1e-3`, RMS `2e-4`,
  and normalized RMS `2e-4`. The largest stage absolute error is `0.00062561035`.
- All 850 code IDs and all 168 by 11 prompt elements match exactly.
- First prefill hidden and main logits pass the existing pointwise
  `3e-5 + 3e-4 * abs(reference)` gate, with zero failures. Maximum absolute
  errors are `5.722046e-6` and `2.193451e-5`, respectively.
- Stage, prompt and prefill metrics are identical between the two graph depths.
  Terminal diagnostic reads add no graph submissions or completion waits.

The MLX exporter compares observed and unobserved official `DAC.encode`
results exactly before exporting fixtures. The native probe compares actual
device buffers. The exact-code/prompt evidence comes from that separate probe,
using the same production source, checkpoint and waveform as generation. Native
generation deliberately does not read its reference codes back to hash them.
The command ledger binds build and execution source hashes and binary hashes;
it is local reproducibility evidence, not a remote attestation.

The captured MLX reports predate a wording-only helper revision and retain
"speaker 0 identity is conditioned" in their interpretation field. That field
describes reference audio/transcript inclusion in the speaker-0 prompt; voice
similarity was not evaluated. The command ledger preserves the executed helper
hash, and the original reports remain unchanged.

## Controlled pause comparison

The experiment uses one English reference from the model card, converted once
to the canonical input format. Its transcript is:

> We asked over twenty different people, and they all said it was his.

Canonical WAV SHA-256:
`a24ee1bf75c1a2bbe414e4ee9d41151b1e13bdff663c35434e8f86baa9b0a0c3`;
172,167 samples, 3.904 seconds, 85 codec frames. The public source is
[the model-card reference](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav).
The input preparation record includes its original hash and the GPU resampling
recipe; resampling is offline fixture preparation, outside native inference.

The target is the same plain Russian text in all four controls, without emotion
tags or chunking:

> Ура, ты вернулся! Почему молчал? Я боялась тебя потерять. Теперь всё хорошо! Только никому не говори наш секрет.

Settings are seed 1, temperature 0.8, top-k 30, top-p 0.9, a 512-frame cap,
CogniGraph depth two and private mmap. Both native runs hash the actual
5,632-element little-endian F32 uniform array before its single GPU upload.
That hash is checked against the native producer probe and the same prefix of
the MLX tape:
`63e955c00512b965c17303e4c2517ed4eeb093193af6c881542045c0c7ece7a1`.
The MLX control uses the established shared-uniform inverse-CDF shadow sampler,
not stock MLX random draws. Its F32-core backbone control preserves packed
quantized weights; DAC weights keep their checkpoint dtype.

The fresh source-bound native pair and pinned MLX controls agree on every
generated main/codebook token, all 11 IDs per frame, and both terminate at EOS:

| Conditioning | Frames, native / MLX | Waveform duration | Internal quiet runs of at least 1 s | Time in those runs |
| --- | ---: | ---: | ---: | ---: |
| No reference | 463 / 463 | 21.502 s | 4 | 8.5 s |
| One reference and transcript | 218 / 218 | 10.124 s | 0 | 0 s |

Without reference the measured runs are 3–5, 7–9, 11.5–13.5 and 15.5–18 seconds.
One additional isolated 0.5-second quiet window is excluded by the one-second
minimum. Native and MLX agree on all qualifying quiet windows in each case;
their PCM samples are not bit-identical, consistent with the separate decoder
numeric frontier. The native no-reference WAV also reproduces the earlier
plain-text control byte for byte.

Durations describe waveform length, not inference latency. Quiet intervals use
complete 0.5-second PCM16 windows with
peak at most 64, adjacent windows merged, and a minimum run of one second.
Partial final windows are excluded.

This is one waveform/transcript, target and seed. Signal windows do not certify
spoken words, sentence boundaries, voice identity or a universal pause remedy.
Original Fish full text generation is not covered. Existing decoder floating
parity, MoE-39 and other code-fork frontiers remain open.

## Reproduction and refresh

Use new output paths and serialize model jobs through `scripts/run_safe.sh`
with the admitted 15% memory floor. The completed command ledger in the evidence
index preserves exact local arguments and environment. The main steps are:

1. Export the pinned official encoder/prompt fixtures with
   `scripts/drama3/reference_voice_reference.py` and the first prefill capture
   with `scripts/drama3/prefill_reference.py`.
2. Build `examples/drama3_reference_compare.cr`; run its `--self-test`, then
   compare with `--model`, `--manifest`, `--wav`, `--reference-text`, `--text`,
   `--prefill-reference` and a new `--output`, separately at graph depths 0/2.
3. Generate native and MLX reference/no-reference pairs with the same settings
   and exact sampler inputs. `scripts/drama3/generate_mlx_control.py` supplies
   the pinned upstream control.
4. Run `scripts/drama3/compare_reference_pauses.py --run-dir ...
   --random-tape ... --tape-proof ... --encoder-report ... --output ...`.
   This rejects wrong sampler counts/hashes, conditioning and frame termination,
   and requires exact encoder code IDs and prompt rows before GPU token/PCM
   measurements. It cross-checks upstream/model identity consistency between
   reports and records the full numeric encoder gate. Separately require that
   numeric gate to pass and verify pinned source/checkpoint hashes using the
   probe and source-binding evidence; the analyzer does not enforce those pins.

CPU contract tests are `spec/drama3_voice_control_spec.py`,
`spec/drama3_reference_voice_spec.py`, `spec/drama3_prefill_reference_spec.py`
and `spec/drama3_reference_pauses_spec.py`. The resident GPU suite covers
reference audio, prompt handoff, generation, dense/conv/RMS encoder tails and
causal attention masks at 31/32/33 and 65 positions, including a fully masked
initial key tile.

The strongest failure mode is a valid-looking waveform conditioned by different
reference codes or a mismatched sampler tape. Exact code/prompt gates, run-side
sampler digests, source binding, and negative validator tests guard it. Revert
the single feature commit to roll back. Refresh these results on source,
checkpoint, waveform/transcript, target, sampler, gate, runtime/device or OS
changes. A failed refresh reopens the affected claim; do not widen the gate.
