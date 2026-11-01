# Hy-Nomas local draft suite

## Design-sealed Cinematic comparison (2026-11-01)

The active request adds an explicit Cinematic BF16 opt-in to the two-segment
drafter and compares the retained six scenes with their original first takes.
Keep the exact frozen suite, authenticated conditioning/timeline manifests,
noise banks, four steps, geometry, decoder and history policy. Reuse validated
baseline input bundles instead of re-encoding prompts or generating new noise.
Base remains the default. The existing offline merged cache must pass its
lineage/index admission; no online LoRA merge or new GPU operation is admitted.

This is CAUTION for model selection and artifact provenance. The strongest
failure is a visually plausible comparison made with a different input or a
mislabelled weight index. Bind the requested variant and cache identity through
preflight, loader and final receipt, and reject a mismatched selected pair
before rendering that scene. Keep the existing 10% file-cache-aware floor,
30 GiB process cap, swap-growth stop and one model-owning process at a time.
The sequence checkpoint/resume path remains base-only unless its contracts
are separately extended; a Cinematic instance must not emit base-labelled
checkpoints. Roll back by omitting the new flags; retained base media are
read-only and must never be overwritten.

DoD: failing-before-fix variant/input identity checks, focused Python and
Crystal admission tests, a fresh guarded native build, CPU-only memory plans,
and six completed Cinematic clips paired with authenticated base clips.
Require unchanged input/noise digests, variant-specific index/lineage digests,
240 frames at 24 fps, zero instrumented encode allocations/readbacks or
preterminal waits, and one existing terminal drain per GPU phase. Refresh the
selected upstream merge check and authenticate the prior intermediate evidence
without widening 2e-5 absolute or relative tolerances. Inspect paired media and
report quality observations separately from numerical parity; full upstream
video parity is not admitted.

VACE is an explanatory branch only: no new VACE weights, renderer, training or
model launch is included in this implementation.

### Opt-in usage and controls

`examples/hy_nomas_draft.py` accepts `--cinematic-cache` (the existing offline
BF16 cache's `cinematic-lineage.json`) together with `--cinematic-cache-sha`.
The native sequence CLI accepts the same pair. Omit both to use original BF16
weights. The selected variant, index digest and lineage are bound to the CPU
memory plan and final native receipt; the gallery displays the effective
variant instead of relabelling Cinematic as original.

For controlled comparisons, pass `--reuse-inputs-from /absolute/results.json`
with the exact original suite bytes. Admission requires a completed run with
the same ordered scenes, prompts, Qwen conditioning and fixed noise. It checks
the original raw pixels, MP4s and payloads, then reuses those input paths.
Scene metadata and selected payloads are checked again before native preflight;
native preflight and execution independently authenticate the payloads.
Neither Qwen encoding nor RNG/input production is rerun. Reuse does not turn
archived runtime measurements into a fresh paired speed benchmark.
Admission initially authenticates all scene payloads; subsequent selected-scene
checks reject a changed payload before that scene's preflight or execution.
This is an incremental suite, not an atomic transaction over concurrently
mutable future inputs. A later failure preserves earlier completed scenes and
leaves the overall report incomplete.

For example, use a new output directory and a separately trusted lineage
digest rather than deriving that trust from an unverified cache at launch:

```sh
python3 -B examples/hy_nomas_draft.py \
  --suite /absolute/frozen-suite.json \
  --generator /absolute/build/hy_nomas_sequence_generate \
  --run-dir /absolute/new-cinematic-run \
  --reuse-inputs-from /absolute/base-run/results.json \
  --cinematic-cache /absolute/cinematic-cache/cinematic-lineage.json \
  --cinematic-cache-sha "$CINEMATIC_LINEAGE_SHA"
```

### Observed six-scene comparison (2026-11-01)

All six Cinematic first takes completed with the exact retained native prompts,
conditioning manifests/tensors and three noise banks. Each clip contains two
five-second segments, four steps per segment, 240 frames at 512x288/24 fps,
original-precision BF16 merged weights and the existing F32 live history. No
conditioning producer or RNG was rerun. Base remains the default.

| Scene | Total scene seconds | Sampled process-group peak GiB |
| --- | ---: | ---: |
| portrait | 283.053 | 24.221 |
| pouring-water | 308.829 | 24.305 |
| running-dog | 307.578 | 24.221 |
| balloon-car | 278.368 | 24.321 |
| coast-flight | 266.863 | 24.217 |
| desert-donut | 266.600 | 24.320 |

The median observed total was 28.071 seconds per output video second, excluding
the already-retained conditioning preparation. This is descriptive timing,
not a fresh paired speed benchmark: five of the six base clips are historical.
A fresh original portrait on the new binary reproduced the retained raw pixels
and latent bank byte-for-byte. The historical receipts lack explicit variant
identity; the fresh control anchors that unchanged base path for one scene.

All 18 guarded stages finished successfully and left no live child process in
their sampled groups. Maximum sampled peak was 24.3214 GiB, minimum estimated
cache-aware headroom was 22.7736 GiB, and observed stage swap growth was zero.
The 10% floor, 30 GiB native cap and 64 MiB swap-growth stop were unchanged.
Both inference phases in every receipt report zero tracked encode allocations,
readbacks and preterminal waits, with one existing terminal drain per phase.
These counters exclude opaque MPS resources; sampling cannot bound short spikes.

Focused verification passed: 60 Python tests, 11 CPU-only Crystal examples,
Crystal formatting and a fresh guarded Metal build. The fresh upstream merge
check compared 302,113,792 BF16 values in blocks 0 and 1 with zero mismatches.
The prior authenticated all-block merge and same-latent PixelUMM evidence is
indexed by `.landmark/evidence/2026-10-31-hy-nomas-cinematic-quality.json`.
No new complete Cinematic DiT trajectory or ten-second intermediate-output
gate was executed. The source-MPS oracle is not unmodified CUDA/FlexAttention.
The existing 2e-5 absolute and relative tolerances were not widened.

Retained artifacts are under
`/opt/mldata_ssd/cogni-ml/cinematic-drafter-20261101.CSoblFK7/`:
`experiment-summary.json` binds the source, executable, input, model and guard
identities; `pairs/comparison.json` binds the six authenticated comparisons;
`pairs/reel-60s.mp4` and `pairs/index.html` provide playback. Each pair has the
base on the left and Cinematic on the right. The pair builder rejected an
incomplete suite, passed eight hostile geometry/publication checks, and then
published the completed directory atomically. The final reel decodes to 1,440
frames at 1024x320/24 fps; each pair decodes to 240 frames.

Inspection of paired stills at 0, 3, 6 and 9 seconds found stronger surface
texture and contrast with Cinematic, changed composition, and one visible
walker rather than multiple walkers in the doughnut scene. Balloon count still
changes, and both dog variants face left despite the requested left-to-right
motion. These are sampled visual observations, not full-motion, physics or
general quality qualifications. Preserve all first takes for user review.

The curated evidence packet is
`.landmark/evidence/2026-11-01-hy-nomas-cinematic-drafter.json`. Requalify after
producer/validator source, binary, model/cache, inputs/noise, geometry/history
or runtime changes; repeat timing after host/device-state changes. Missing or
changed local artifacts invalidate their use as evidence. Cinematic sequence
checkpoint/capture/restore remains explicitly unsupported.

### VACE integration boundary

The official [VACE README](https://github.com/ali-vilab/VACE/blob/main/README.md)
supports text, source video, masks and reference images; its Wan renderer
loads a separate VACE model/control path. A practical CineStudio branch is
Hy-Nomas draft pixels → prepared depth/pose/layout or editing masks → Wan VACE
render. An editable 3D draft could provide corrected motion/depth directly.
This is a proposed separate renderer, not an implemented Hy-Nomas feature.

VACE cannot be attached as this Cinematic LoRA: direct Hy-Nomas conditioning
would require a compatible trained control branch and representation. Pixel
input is the existing interchange boundary; native continuation latent bundles
are not interchangeable. VACE is not a physics solver, and transferring a bad
draft trajectory can preserve its error. The official
[user guide](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md) recommends
roughly five-second clips and sequential extension for long work; seams and
temporal preservation would need separate testing on our ten-second scenes.

## Design-sealed first-take frontier (2026-11-01)

The requested output is six independent photorealistic ten-second clips, with
their prompts, actual generation times and an offline gallery. Reuse the
existing native sequence generator at two five-second segments, 512x288,
24 fps and four Euler steps. Original BF16 weights, F32 live history and
once-per-scene text caching remain the control. No model kernels, rounding
boundaries, arithmetic tolerances or command-chain synchronization change.

New prompts require authenticated Qwen state-26 conditioning. The current
sequence CLI consumes saved conditioning; preparation and provenance must be
reported separately from native DiT/PixelUMM execution. Each scene receives
its own input bundle and fresh process. The input format fixes seed-42 noise
banks, so this is a controlled prompt variation, not six independent random
samples. Preserve the first take of every scene, including visible defects.

This is CAUTION for process orchestration and provenance. The strongest
failure mode is a successful-looking gallery that relabels stale conditioning,
truncated video, a failed guard or output from another scene. Bind the intended
prompt and exact input manifest digest to the native receipt, raw-pixel digest
and exported MP4. Do not launch the next scene after a failed stage. Store
artifacts under a fresh directory on `/opt/mldata_ssd`; reject occupied outputs.

Only one model-owning process may run at a time. Preserve a 10% file-cache-aware
memory floor, a sampled process-group footprint cap, 64 MiB maximum swap growth
and bounded deadlines. Preflight each input's named-memory plan before model
setup. Sampling does not reserve memory or bound opaque MPS spikes. A guard
rejection is an incomplete run, not permission to weaken these limits.

DoD: hostile unit checks reject prompt/input identity mismatches, incorrect
geometry, missing drains, extra encode allocations/readbacks/waits, stale or
truncated outputs and incorrect MP4 frame/timing metadata. Each completed clip
must contain 60 clean latent frames and 240 decoded pixel frames at 24 fps,
512x288, exactly ten seconds, with eleven PixelUMM windows. Require zero
instrumented encode allocations/readbacks/preterminal waits, one terminal
drain per existing GPU phase and successful guard cleanup. Export and media
inspection occur after the GPU phases drain. Run the focused Python tests and
the guarded six-scene suite, then inspect representative frames and deliver
the actual videos. Numerical upstream parity and visual quality remain
separate from artifact integrity; full upstream video parity is not claimed.

No audio generation, weight quantization, scene selection, Studio server,
cross-scene history or full editor are admitted in this slice. Roll back by
removing the new drafter files; the native sequence generator and existing
artifacts remain unchanged. Evidence expires when the scene manifest,
authenticated inputs, native executable, producer sources, weights or exported
media change.

## Authorized prompt-planning extension (2026-11-01)

The user separately authorized GPT-6 Luna through the API key already in the
environment, and asked for an empirical evaluation of its prompts. A separate
planner may call the fixed OpenAI Responses endpoint with `gpt-6-luna`,
`reasoning.effort=medium`, a bounded output and a strict JSON schema. Medium is
a starting choice, not a measured quality/cost optimum. Official model and
Structured Outputs documentation establish the request surface:
https://developers.openai.com/api/docs/models/gpt-6-luna and
https://developers.openai.com/api/docs/guides/structured-outputs.

Planning produces an editable frozen suite with provenance; it does not
produce Qwen embeddings or launch a renderer. The default is a network-free
request preview; `--execute` performs one bounded call, with no silent retry,
model fallback or automatic render. Preserve the six scene IDs/order, native
geometry and guard settings. Reject refusals, incomplete/malformed results,
duplicate IDs, extra scenes and unsupported action-tail scheduling. Never
print or persist the credential, authorization headers or a raw environment.

DoD: mocked transport/negative parser tests, one authorized real six-scene
planning call with its usage/model/effort receipt, a manual intent and
feasibility review before rendering, then the same first-take media gates.
The rendering process does not receive the planner credential. Assess intent
retention and motion/camera clarity separately from syntax; a model-generated
plan does not establish better video quality or isolate planner causality.

An initial real call was rejected by an overly literal local keyword check
before its body was saved. That discarded response cannot support a semantic
quality judgment. The planner now saves a bounded response-body artifact,
including on post-response validation failures, without request headers. Style
is validated as a typed field; shot continuity and intent are reviewed from
the actual content rather than inferred from two English keywords.

## Local use

Planning and rendering are separate steps. Set `OPENAI_API_KEY` in the shell
environment; do not put it in a command argument, suite file or browser code.
First preview the bounded request without a network call:

```sh
python3 -B examples/hy_nomas_plan.py \
  --suite examples/data/hy_nomas_photorealistic_drafts.json \
  --output /opt/mldata_ssd/cogni-ml/my-luna-suite.json
```

Add `--execute` to send one authorized API request. Medium effort is the
default. Inspect the resulting suite and adjacent receipt/response artifact
before rendering. They use fresh paths and are never silently overwritten.
The response artifact is preserved on local post-response rejection so the
same paid result can be inspected. There is no automatic paid retry.

Build the native producer with a scratch build directory, then give the
drafter a fresh run directory:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/draft-build hy_nomas_sequence_generate
python3 -B examples/hy_nomas_draft.py \
  --suite /opt/mldata_ssd/cogni-ml/my-luna-suite.json \
  --generator /opt/mldata_ssd/cogni-ml/draft-build/hy_nomas_sequence_generate \
  --run-dir /opt/mldata_ssd/cogni-ml/my-draft-run
```

`--reference-python`, `--upstream` and `--checkpoint` can override the local
prepared asset paths. The runner uses existing local weights and does not
download them. Its `index.html` and `results.json` expose completed clips and
failures as the suite progresses. Rendering never calls the cloud planner.
Each MP4 is a silent ten-second diagnostic draft; sound and UI integration
are separate work.

Focused checks:

```sh
python3 -B -m unittest spec/hy_nomas_plan_test.py \
  spec/hy_nomas_draft_test.py spec/hy_nomas_draft_results_test.py
```

## Retained experiment (2026-11-01)

Artifacts are under
`/opt/mldata_ssd/cogni-ml/hy-nomas-drafts-20261101.hj3x23ur`.
The retained plan is `luna2-suite.json`, with SHA-256
`43a1e368dbe9c8046ea8da24f99cf0f7a57d176a2fce203288c913587172c838`.
Its response/usage receipt records `gpt-6-luna`, medium effort, 1,334 input
tokens and 1,250 output tokens. This usage covers the retained response only;
the earlier discarded response is not included.

Manual text review found the core actions retained in five scenes. Luna
removed the water scene's stop-pouring and set-down actions. That scene was
retained as a reduced pouring-only diagnostic, rather than counted as full
intent preservation. The editable planning stage is useful, but these six
examples do not establish that medium effort is optimal or that Luna improves
video quality over the original prompts: no planner A/B experiment was run.

The completed run is `suite-run3/results.json`; `suite-run3/index.html` is its
offline gallery. `suite-run3/preview-reel.mp4` concatenates the six ten-second
exports without re-encoding. The successful native executable was freshly
built, with its command and hash retained. Selected producer source snapshots
are in `selected-source-snapshot/`; automatic correspondence between all build
inputs and the executable is not proven. Post-run gallery/report corrections
do not relabel those producer snapshots or change the videos.

| Scene | Total seconds | Sampled process peak, GiB | Sampled visual finding |
| --- | ---: | ---: | --- |
| Portrait | 294.46 | 24.23 | Consistent face/framing; blink and full head turn are not established by the samples. |
| Pouring water | 315.55 | 24.31 | Stream and filling glass are visible; the reduced prompt keeps pouring through the end. |
| Running dog | 301.74 | 24.22 | Dog faces left despite the left-to-right request; trajectory and continuous gait remain unverified. |
| Balloon car | 295.59 | 24.32 | No yellow balloon is visible; the finish looks overly smooth and the requested camera angle is uncertain. |
| Coast flight | 295.66 | 24.22 | Photographic-looking coastal samples; continuous flight smoothness remains unverified. |
| Desert donut | 300.90 | 24.32 | Clear failure: one requested walker becomes several people with inconsistent clothing. |

Total scene time was 1,803.91 seconds for sixty seconds of video; the median
was 29.83 seconds per video second. These times include each scene's
conditioning, loading, native execution and export. They exclude the build,
API planning, post-run inspection and earlier memory-limited attempts. No
output from those interrupted attempts was promoted. The largest sampled
process-group footprint was 24.32 GiB, and minimum sampled file-cache-aware
headroom was 20.34 GiB. All 36 guarded stages exited zero with empty process
groups at their final samples; the 10% floor and 30 GiB native cap were retained.
These sampled observations do not bound between-sample MPS spikes.

Each exported clip contains 240 frames, 512x288 at 24 fps for ten seconds.
Raw-pixel and MP4 digests bind the exports to their receipts. All six native
receipts report zero instrumented encode allocations, readbacks and
preterminal waits, with one terminal drain per GPU phase. The metric excludes
opaque MPS resources and terminal exports. Conditioning uses authenticated
saved upstream Qwen state-26; live native text encoding and full upstream
video parity remain outside this result. No arithmetic or tolerance changed.

`visual-review.json` records inspection of twelve frames per scene, including
frames 119/120/121 around the five-second boundary. This supports sampled
composition and visible prompt-fidelity findings, not full-motion quality or
temporal continuity. The same-lineage adversary review is correlated evidence.
`experiment-summary.json` indexes the measured times and guard cleanup checks.
The focused suite passes 36 tests. The actual final gallery CLI also exits
zero for all six videos, validating raw-pixel hashes and MP4 hashes/metadata.
That CLI check exposed a non-serializable `Path` in the public report; the
report now explicitly emits string paths, with a regression that reproduced
the failure before the fix. `postrun-review.json` binds those post-run report
changes and checks to the retained producer snapshots without changing their
source identity.

Recheck all media and regenerate the gallery with the current report code:

```sh
python3 -B examples/support/hy_nomas_draft_results.py \
  --suite /opt/mldata_ssd/cogni-ml/hy-nomas-drafts-20261101.hj3x23ur/luna2-suite.json \
  --results /opt/mldata_ssd/cogni-ml/hy-nomas-drafts-20261101.hj3x23ur/suite-run3/results.json \
  --output /opt/mldata_ssd/cogni-ml/hy-nomas-drafts-20261101.hj3x23ur/suite-run3/index.html \
  --validate-final
```
