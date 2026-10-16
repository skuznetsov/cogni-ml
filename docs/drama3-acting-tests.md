# DRAMA-3 acting prompts and frame-window experiment

These are diagnostic runs on the existing native Metal path and the pinned
MLX port. They test a newsreader-to-storyteller prompt and an actor alternating
two roles, without a voice reference. They do not establish separate speaker
identity, actor quality, or a remedy for missing speech.

The dialogue below is newly written test material, not an excerpt from a book.
The native prompt still uses one participant token, speaker 0. Age and role
directions are learned text cues rather than native speaker assignments.

## Controls and inputs

Source state: `8e1f326a83925f8af7f00109cf6e6989441d3c20`. The reused native
binary is bound to the current production source, embedded shaders and all
nine freshly hashed checkpoint files. No production inference code changed.
Upstream is clean MLX-audio `e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`,
MLX 0.31.1, with `cguldogan/drama-3-mlx-q4` checkpoint
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`. The MLX control uses its F32
core with packed Q4 weights and checkpoint codec dtype, plus the existing
shared-uniform inverse-CDF shadow sampler. This is not original Fish full
text-generation coverage or stock MLX random sampling.

Both cases use seed 1, temperature 0.8, top-k 30 and top-p 0.9. The native
sampler's actual uploaded F32 array matches the producer probe and the MLX
tape prefix; seed equality alone is insufficient. Native uses private mmap,
CogniGraph depth two and the existing reduction settings. Heavy commands run
serially through `scripts/run_safe.sh`, with a 15% free-memory floor, on the
shared host. Phase times are exploratory, not a performance promotion.

The initial prefix has 5,632 uniforms (512 times 11) and SHA-256
`63e955c00512b965c17303e4c2517ed4eeb093193af6c881542045c0c7ece7a1`.
The 768-frame prefix has 8,448 uniforms and SHA-256
`708654400de323b1e6698b230d01f4329065ae9582fad67bc71213e77d3a3dd6`.
Both come from the same full tape with SHA-256
`ddc31bee6c0ffcd12c2ba781207747cad5eab9cbcb02faa074658caac8e16341`.

Exact news prompt, UTF-8 SHA-256
`029217880c23eb56629b018a3618eef74d23cd38ad9881e10c323bb663aa122b`:

```text
Direction: One Russian male speaker. Begin as a composed, professional newsreader. At "А теперь представьте" switch to an animated, expressive storyteller, with growing wonder and excitement. Keep natural brief sentence pauses. Speak only the Russian line.
Line: [formal, composed newsreader] Добрый вечер. В городском парке открылась новая обсерватория. Первые посетители увидели кольца Сатурна.
[animated, expressive storytelling with wonder] А теперь представьте: я подхожу к телескопу, заглядываю в окуляр — и замираю. Передо мной целый мир! Вот это приключение!
```

Exact dialogue prompt, UTF-8 SHA-256
`801f156854c389511820de037faec8becafe155a2eaed8791de6340721172756`:

```text
Direction: Perform this short Russian dialogue as an actor playing two roles. Max has a youthful, curious, impulsive male voice. Juffin has an older, low, warm male voice and dry amusement. Alternate roles at each bracketed cue. Keep pauses brief and natural. Speak only the Russian dialogue.
Line: [young man Max, curious and surprised] Сэр Джуффин, почему эта дверь открывается только ночью?
[older man Juffin, low warm voice, dryly amused] Потому что днём она занята, Макс. Снится кому-нибудь.
[young man Max, eager and challenging] А если я всё-таки войду?
[older man Juffin, low warm voice, gently teasing] Тогда будь вежлив. Даже у чужого сна есть хозяин.
```

## Initial 512-frame results

All four runs return 512 frames at the cap, without EOS. Each complete returned
code array decodes to 1,048,576 samples at 44.1 kHz, or 23.777 seconds. This is
a decoded bounded output, not a completed utterance.

| Case | Exact leading native/MLX frames, all 11 IDs | First codebook / main mismatch, zero-based | Internal quiet intervals, native / MLX | Terminal quiet interval, both |
| --- | ---: | --- | --- | --- |
| Newsreader to storyteller | 512; entire returned stream | none / none | 6.0–7.0 / 6.0–7.0 s | 11.5–23.5 s |
| Max and Juffin | 126 | 126 / 132 | 11.0–12.0 and 14.0–18.5 / 13.5–18.0 and 20.0–22.5 s | none |

Integer token comparisons run on GPU. Quiet intervals merge complete 0.5-second
windows whose peak is at most 64 PCM16, with a minimum run of one second. The
partial final window is excluded from interval ranges; it is quiet in both
news clips and active in both dialogue clips. These are amplitude observations,
not word alignment. Exact IDs do not imply bitwise DAC waveform equality.

Unprompted GPU Whisper recognizes only the news body through "кольца Сатурна"
in both news clips. It recognizes the dialogue into the final reply, ending
near "есть хозя..." in native and near "Тогда будет..." in MLX. The
source-positive control recognizes its English words, while exact digital
silence hallucinates "Thank you." ASR cannot certify actor delivery or complete
spoken text. No listener confirmation is bound to these four clips yet.

## Wider 768-frame results

The wider plan changes only `max_frames`, from 512 to 768. The actual 8,448
F32 sampler inputs match the fresh production-producer probe. GPU comparisons
show that each backend's complete first 512 frames, including all 11 IDs,
equal its own initial run exactly, in all four comparisons. This is token-prefix
stability, not equality between the two backends or a WAV-prefix/DAC gate.

| Case | Native frames / stop / duration | MLX frames / stop / duration | Exact leading native/MLX frames, all 11 IDs |
| --- | --- | --- | ---: |
| Newsreader to storyteller | 768 / cap / 35.666 s | 741 / EOS / 34.412 s | 704 |
| Max and Juffin | 525 / EOS / 24.381 s | 645 / EOS / 29.954 s | 126 |

The news stream's first codebook and main-token mismatch is frame 704. Both
waveforms remain quiet from 11.5 seconds through their final complete windows,
35.5 seconds native and 34.0 seconds MLX. Both partial final windows peak at
2 PCM16. The added duration therefore does not restore a voiced continuation
under the measured quiet criterion. This rejects the 512-frame cap as the
sole explanation for the missing continuation in this prompt; it does not
identify its cause. The code fork is later than the quiet onset.

The dialogue retains its first codebook/main fork at 126/132 and the same
internal quiet intervals as the initial run. Unprompted ASR now recognizes the
full final reply through "есть хозяин" in both files. EOS is a model stop,
not a guarantee of complete spoken words or natural acting. The news ASR
still omits the storyteller body and adds the unrequested phrase "Продолжение
следует..." in both files. The exact-zero control still says "Thank you."
These are fallible ASR outputs, not confirmed additional speech. Listener
confirmation of delivery and text remains open for the wider files too.

All four wider generations, both wider-capacity numeric probes, GPU analysis
and artifact-composition validation exit successfully. The initial sampler
producer attempt failed at compilation because its command omitted the Metal
build flag/link flags; no generation ran in that attempt. Its failed receipt
is retained, and the corrected `-Ddrama3_metal` producer command passes before
any wider generation. No production code or default frame limit changes.

## Prompt-specific intermediate checks

Fresh first-prefill device-buffer comparisons pass the unchanged pointwise
gate `abs(actual - reference) <= 3e-5 + 3e-4 * abs(reference)`, with no
nonfinite values or failing elements:

| Case | Prompt rows | Wider resident capacity | Hidden max absolute error | Main-logit max absolute error |
| --- | ---: | ---: | ---: | ---: |
| Newsreader to storyteller | 158 | 925 | 4.7683716e-6 | 1.4305115e-5 |
| Max and Juffin | 190 | 957 | 2.861023e-6 | 1.04904175e-5 |

The initial probes allocate prompt-only capacity; the new probes allocate
`prompt_rows + 768 - 1` and request one generated frame to observe the first
prefill. Their stage metrics equal the initial probe metrics. They are not
768-frame intermediate traces.

The tokenizer's main prompt IDs match exactly. This is not a claim of comparing
all 11 columns of every prompt row. The probe dispatches GPU comparisons on
actual generator buffers, then reads reduced metrics after completion. Graph
statistics before and after terminal diagnostic reads are equal at depth two.
The exact-ID and pointwise positive/poison self-tests pass.

The fixture, upstream report, captured NPZ, executed probe and build-source
hashes compose in `prefill-lineage.json`. The wider validator also matches each
comparison report to the hashed successful command stdout and checks the exact
embedded comparator source bytes in both old and new executed binaries. The
wider build/run receipts explicitly hash both support comparator shaders;
the original receipts omitted those shader-file keys, so the original binary's
embedded bytes provide the additional retrospective binding. These checks cover the first prefill,
not every generation step, full DAC intermediates or the existing MoE frontier.
Depth-zero results inherited from the earlier reference experiment are not
promoted to these new prompts. Model arithmetic remains on GPU; initial
uploads, tokenization, existing EOS/control reads, bounded CogniGraph waits,
terminal outputs and diagnostics remain. No precision or tolerance changed.

## What limits the window

512 is the experiment's output cap. The CLI accepts up to 1024 frames, but
Metal additionally requires `prompt_rows + max_frames - 1 <= 1024`
(`examples/drama3_generate.cr`). The current prompt lengths therefore allow
867 and 835 output frames, approximately 40.3 and 38.8 seconds respectively.
Both admit 768 frames, approximately 35.7 seconds, with capacities 925 and 957.

The actual native ceiling is the resident backbone guard and the full-attention
kernel's `threadgroup float scores[1024]`, in
`src/ml/audio/drama3/backbone_resident.cr` and
`src/ml/audio/drama3/kernels/backbone_resident.metal`. Raising only the CLI
guard would not make larger contexts safe. Beyond 1024 requires an attention
scratch/algorithm change and matching resident guards, followed by numeric,
capacity and upstream checks. Simply dropping old KV rows would change the
model context. DAC allocations also grow with output length; their checked
Int32 products are not a 512-frame limit.

The checkpoint's larger text position capacity is not evidence that this
native attention implementation supports it. The audio tower's within-frame
codebook sequence length is also not the audio timeline limit.

## Evidence and refresh

Raw initial artifacts are outside Git at
`/Users/sergey/.cache/cogni-ml/drama3-runs/directing-roles-20261015`:
the immutable plan, command receipts, source/checkpoint binding, four WAVs,
ASR controls, prefill fixtures and reports, GPU measurements, analyzer CPU
guards and lineage validator. The wider experiment is isolated in
`directing-roles-wide-20261015`; no initial output is overwritten.

The declared diagnostic DoD is: complete both native/MLX matrices, validate the
actual uniform prefixes, compare tokens and quiet windows on GPU, compare each
backend's first 512 wide-run code frames to its initial output, retain the
unchanged first-prefill gates, and calibrate ASR against its controls.
Actor quality and the cause of missing continuation remain separate questions.

Observed DoD commands include cache-local `analyze.py`, first-prefill
positive/poison self-tests and `validate_prefill_lineage.py` for each matrix.
The wider command ledger retains the one
resolved compile-only failure and all successful follow-on commands. Analyzer
CPU guard suites pass (eight initial, three wider and three receipt guards);
the receipt binder rejects an edited report and a bad stdout digest.
The evidence index is `.landmark/evidence/drama3-acting-tests.json`.
The shared Landmark packet template and graph-validator script are absent in
this checkout; no result for them is claimed. Direct command, artifact,
source/binary, numeric and hash checks provide the scoped validation instead.

Correlated mini-Quadrumvirate verdict: ROBUST for measured window capacity,
same-backend prefix stability, first-prefill gates and quiet intervals;
VULNERABLE for a successful storyteller switch, natural actor delivery,
full generation/DAC numeric parity, or a unique silence-cause claim. A next
causal discriminator would change only the direction/cue format while keeping
spoken words and sampler inputs fixed. Line breaks alone are not an established
cause.

Refresh on source, checkpoint/tokenizer, prompt, sampler tape or implementation,
conditioning, runtime/device/OS, gate, artifact hash or listening-evidence change.
Rollback is the single report commit; no production inference behavior changes.
