# Hy-Nomas object control frontier

## Native continuation decode slice (2026-11-06)

Status: verified for the bounded native/Torch 2.6 MPS continuation fixture.
The standalone T1 decode omitted the preceding context, whereas the existing
cumulative native sequence already includes window starts 25 and 30 and remains
unchanged. For an explicitly supplied continuation, retain the previous five clean latent
frames and twenty decoder-noise frames, prepend them to the current segment
on GPU, use the existing ordered ten-latent windows and BF16 cosine blender,
then crop the first twenty pixel frames on GPU. A 30-latent continuation thus
decodes 35 latent frames with six windows and exports 120 new pixel frames.

The previous and current noise are explicit authenticated inputs; never
regenerate them or silently substitute current noise for history. The earlier
onset counterfactual used a declared noise surrogate and is not stock RNG
parity. Reference images remain conditioning inputs, not actor controllers.

The standalone CLI accepts the all-or-none quartet `--previous-latent`,
`--previous-latent-sha`, `--previous-noise`, `--previous-noise-sha`. It authenticates
both complete source histories before GPU allocation and preserves their full
channel strides. Its receipt distinguishes borrowed source bytes, joined decoder
inputs, raw 140-frame output and the cropped 120-frame display output.

Observed verification:

- CPU geometry/admission suite: 26 examples, zero failures. Synthetic GPU
  join/crop and existing overlap
  comparison are exact; omitting context changes the expected onset.
- All 36 blocks and six windows at 512x288: raw `[1,3,140,288,512]` and display
  `[1,3,120,288,512]` match the authenticated upstream gathered adapter **bitwise**:
  115,015,680 F32 values, maxabs 0, zero nonfinite values, unchanged tolerances.
- Instrumented encode allocations/readbacks/preterminal waits are zero; the
  decoder retains its one terminal drain. Named buffers total 17,520,565,392
  bytes; the 22 GiB guard exited 0 at the 10% cache-aware floor without swap
  growth. One native run took 116.95 seconds in decoder execution; this is not
  a paired performance claim.
- A separate Torch 2.8 comparison fails and remains recorded. On identical
  block-0 input, the square-reduction output is exact in both runtimes, but
  inverse-root/pre-gain BF16 values differ at 361 positions between 2.6 and 2.8.
  Native pre-gain and gain outputs match 2.6 bitwise. Runtime lineage must be
  explicit; this slice does not qualify the 2.8 arithmetic or stock CUDA path.

Evidence: [.landmark/evidence/2026-11-06-hy-nomas-native-continuation.json](../.landmark/evidence/2026-11-06-hy-nomas-native-continuation.json),
external root `/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue`.
The explicit previous-noise fixture is a frozen T1-noise-tail surrogate for
missing T0 noise. This qualifies assembly/decoding of the declared inputs,
not stock full-clip RNG replay, overall image-to-video parity, or actor control.

Risk: CAUTION (NCTHW channel strides, wrong temporal/noise origin, premature
resource release). Rollback: omit the opt-in previous-input arguments and use
the unchanged standalone or cumulative decoder. DoD: CPU geometry/admission
specs, a synthetic GPU join/crop/overlap sensitivity probe, and a bounded
same-latent native/upstream decode at unchanged `atol=rtol=2e-5`. Instrumented
encode allocations, readbacks and preterminal waits must remain zero, with
one existing terminal drain. Count every joined/cropped buffer before model
allocation; retain file-cache-aware 10% memory reserve and the 64 MiB
swap-growth stop. Missing-context and wrong-channel controls must fail.
Do not promote this slice to stock CUDA/RNG parity or independent actor
control. Requalify after changing inputs, weights, geometry, arithmetic,
overlap, runtime or any bound artifact.

### Local portrait smoke test (2026-11-06)

The explicitly supplied local portrait completed native reference initialization
and one generated T1 segment using original BF16 weights, admitted V263
conditioning, source-BF16 boundaries, four denoising steps and the recovered
encoder/cache identities. Saved upstream Qwen state26 is used; native live
text encoding is excluded. The exported five seconds are exactly source pixel
frames `[120,240)`, excluding technical T0, with 120 frames at 24 fps.

The request took 669.74 s: DCAE encoding 356.70 s, DiT 58.56 s and PixelUMM
193.76 s, with setup/preflight accounting for the remainder. Each phase retains
zero instrumented encode allocations/readbacks/preterminal waits and one
existing terminal drain. The 30 GiB process guard exited 0; sampled peak was
24.60 GiB, minimum cache-aware headroom 16.90 GiB and swap growth zero at the
10% reserve. This single reference-initialization run is not a speedup benchmark.

Selected frames preserve the recognizable face without obvious onset ghosting,
but the head is nearly static. The user reported noticeable changes only in
the final two frames, with unnatural eye displacement rather than blinking.
Targeted inspection of generated frames 114-119 and enlarged eye crops shows
asymmetric eye deformation and no complete close-and-reopen cycle. The earlier
blink interpretation from sparse sampling is withdrawn; neither a natural
blink nor a convincing head turn is demonstrated. This remains a reference-path
execution smoke test, not successful actor control or a fresh full-pipeline
upstream parity result. The tail inspection used stored F32 pixels on CPU only;
it did not rerun the model or change the GPU path.
The same-input source control below now covers the photo encoder, DiT result
and generated display pixels. Its bounded numerical parity does not establish
that the experimental photo-conditioning scheme produces useful motion.

Evidence: [.landmark/evidence/2026-11-06-hy-nomas-native-portrait-smoke.json](../.landmark/evidence/2026-11-06-hy-nomas-native-portrait-smoke.json).

### Same-portrait upstream control (2026-11-06)

The supplied portrait now passes a staged comparison against pinned upstream
`911cea6458a89c2663513ff7099561ec190dd491` on the same cached RGB8 pixels,
saved V263 Qwen state26, initial latent noise, cache epsilon and decoder noise.
No random tensor was redrawn. Original BF16 weights are used throughout;
the historical DiT cache directory name does not select Q4 arithmetic.

- Upstream DCAE-V encodes the repeated 120-frame photo into the same 30-frame
  clean T0: all 138,240 F32-expanded BF16 values are bitwise equal.
- Upstream DiT runs one T0 cache writer and four T1 denoising forwards through
  all 54 blocks. The 138,240 generated T1 values are bitwise equal. The complete
  first-60-latent decoder input has SHA-256
  `19250cf779b9218f8c2ae9055c6272fcd01444de1c30deaa37bf477eef97ff42`,
  independently equal to the channel-strided native bank prefix.
- Upstream PixelUMM consumes that upstream DiT output, saved noise frames
  `[0,240)` and the already tile-major saved RoPE. All 36 blocks and 11 windows
  (starts 0,5,...,50, ten latent frames with five-frame overlap) reproduce the
  native generated display crop `[120,240)` exactly: 53,084,160 finite F32
  values, maxabs/RMSE/unequal/outside-tolerance all zero. The exported five-second
  MP4 is also byte-identical. The side-by-side video is native left, source right.

The unchanged gate is `atol=rtol=2e-5`; equality is stronger for this fixture.
Encoder and DiT use Torch 2.8.0 MPS; PixelUMM uses the established Torch 2.6.0
gathered local-attention adapter. The rejected 2.8 decoder result above remains
rejected. This is not an unmodified CUDA/FlexAttention or live native Qwen test,
and no claim covers every DiT intermediate: native midpoint captures are absent.
Each isolated stage keeps arithmetic/comparison on MPS with zero intermediate
readbacks, zero explicit MPS synchronization calls and one packed terminal
transfer. The production native path is untouched. All three sampled guards
exit 0 with an 18 GiB process cap, 10% cache-aware reserve and 64 MiB swap-growth
stop. Peak footprints are 9.78/4.73/5.79 GiB for encoder/DiT/decoder; minimum
headroom is 19.84/24.27/17.85 GiB, with zero swap growth. These isolated diagnostic
timings and resource layouts do not establish a performance speedup.

The weak motion and late asymmetric eye deformation occur in both outputs.
There is still no demonstrated natural blink or convincing head turn. The
ordinary pinned `seg_ar` path initializes generated noise and encodes text;
it does not expose the repeated-photo T0 construction as a stock image-reference
entrypoint (`hymm/pipeline/base.py:268-295`, `causal_ar.py:1985-1998,2034-2039`).
The portrait T0 is an experimental conditioning extension. For this one
00:00-00:10 scene, first-chunk text caching followed by reuse agrees with source
(`causal_ar.py:2193-2201,2367,2617`); there is no demonstrated text-cache mismatch.
The preset cache sigma matches, but source draws epsilon from its CPU generator;
the present comparison explicitly replays saved tensors instead of claiming
stock seed/draw-order parity. Four denoising steps are a chosen control, whereas
the pinned experiment preset uses eight (`configs/experiment/inference.yaml:84`).
None of these observations identifies the cause of weak motion by itself.

Decision: stop treating this portrait's visible defect as evidence of an
unlocated native arithmetic error. The next bounded discriminator is the
photo-conditioning/motion contract, retaining the exact path as control.
Adversary verdict: ROBUST for the declared endpoints and their linkage only;
same-lineage review remains correlated. The decoder producer hash is recorded
from the frozen worktree, not attested inside its immutable run receipt.
Refresh after changes to sources, weights, runtime, conditioning/noise,
cache/RoPE/window/blender geometry or bound artifacts. Rollback: omit the new
diagnostic helpers; inference defaults are unchanged.

Evidence: [.landmark/evidence/2026-11-06-hy-nomas-portrait-upstream-control.json](../.landmark/evidence/2026-11-06-hy-nomas-portrait-upstream-control.json).
External media: `/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue/photo-upstream-comparison-v1/native-left-upstream-right-5s.mp4`.

Status (2026-11-06): bounded refiner/DiT probes and four-step replay pass
their frozen source/native gates. Frozen Q/K heads 12/15 retain strong
diagonal preference on three held-out frames of one authenticated T0 pass,
with a large pre/post-RoPE contrast. A fixed-destination-phase content roll
reduces relocated-content matches sharply; these heads are not qualified
as phase-invariant object correspondences. This qualifies the affinity
measurement, not semantic correspondence. Query transport and actor controllers remain proposed.
The accepted five-second Alex/Resume ROI take remains a visual-only control.
Its saved decoder bank does not authenticate the identically named DiT latent.
A new upstream activation recording now has an exact-linked five-second video;
selected frames still show limited acting (see the recording section below).
The original audit was read-only; subsequent bounded GPU experiments are recorded below.
Numerical parity of the generation path remains scoped to the frozen fixtures in
[the drafter frontier](hy-nomas-drafter-frontier.md); visible actor control
is still partial.

## What the current controls actually do

- At Tencent pin `911cea6458a89c2663513ff7099561ec190dd491`,
  `SubjectN` is a textual identity anchor, not a spatial actuator.
  Action-list order and timestamps select action-text ranges for video
  time windows across the whole spatial scene. The published SFT recipe
  enables this temporal-mask path; it is not a training receipt for the
  released checkpoint. See the
  [prompt serializer](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/911cea6458a89c2663513ff7099561ec190dd491/hymm/inference.py#L560-L694)
  and [temporal attention mask](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/911cea6458a89c2663513ff7099561ec190dd491/hymm/models/dit/modules/attention/magi_attn.py#L96-L124).
- Current upstream main was checked at
  `440f01e11b51ad3662cb572d21742a12a14638b1`. Its bundled
  [PE first-turn prompt](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/440f01e11b51ad3662cb572d21742a12a14638b1/apps/prompts/online_user_video_rewriter_system_prompt.md#L17-L20)
  permits exactly one controlled Subject1 slot/ID. The
  [continuation prompt](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/440f01e11b51ad3662cb572d21742a12a14638b1/apps/prompts/online_user_video_rewriter_intermediate_turn.md#L10-L72)
  preserves that roster and forbids assigning actions/IDs to new entities.
  The single subject can itself be a collective group; this still supplies
  no separately addressable per-member control. It does not mean the model
  can animate only one visible entity.
  Multiple subjects in offline prompt JSON do not establish independent
  object control. The checked core action-mask path has not changed since
  the numerical pin.
- Native v2 admits two ordered, nonoverlapping actions in final T1.
  `hy_nomas_action_conditioning.cr:195-237` labels an entire latent frame
  by action and leaves all video/history keys global. The scratch ROI
  experiment narrows video-query exposure to authored rectangles; it is
  neither segmentation nor tracking.
- The direct mask is correctly scoped:
  `hy_nomas_history_attention_bridge.mm:337-345` permits global keys or
  a positive key label matching the query. A global query cannot directly
  read Action1/Action2 text keys. This is not the missing isolation route.
- There are two other routes. First, the text refiner averages every valid
  Qwen row into shared conditioning `c`, and its text self-attention has
  no action mask. Action2 can therefore influence global/Action1 refined
  rows before DiT masking. See
  `kernels/hy_nomas_text_refiner.metal:34-57` and
  `hy_nomas_text_refiner.cr:625-650,720-728`.
  Second, an action-conditioned video state becomes a globally visible
  video key in the next DiT block. This permits relay between regions
  and times inside the current segment.
- The exported `vec` is produced by `time_in` from the timestep, not
  from pooled text (`hy_nomas_text_refiner.cr:653-656`). The shared
  refiner `c` and the exported time vector must not be conflated.

These are source-level reachability findings. They do not show that either
route caused the weak head turn. Cross-object context is also needed for
gaze, occlusion and contact; eliminating all cross-object attention is not
the objective.

## Competing explanations and the next experiment

Keep three hypotheses separate:

1. **Ambiguous requested motion.** Alex already faces approximately toward
   Resume in T0. A small pixel difference does not establish a distinct
   head orientation. The rejected look-down rewrite changes the trajectory
   and can regress without any cross-actor interference.
2. **Conditioning transfer.** Action2 changes the representations available
   to Action1/global queries through the refiner or later video relay.
   Reachability alone does not measure its strength or semantic effect.
3. **Weak learned binding/articulation.** The instruction reaches the
   intended region, but the model cannot reliably express the desired
   pose/contact while preserving the paper/human identities.

The first implementation slice, executed below, is a **refiner-only
counterfactual**:

- Freeze the accepted V224 input, weights, timestep, mask and source/native
  precision controls. Run A/A. Change only Action2 wording, preserving the
  exact token count and offsets. Prove global and Action1 Qwen input rows
  are identical before measuring refiner transfer. If they differ, reject
  the attribution experiment rather than crediting the refiner.
  The pinned encoder runs the full prompt before cropping
  (`hymm/models/text_encoder/encoder.py:421-447`), using the causal
  Qwen2.5-VL language model from
  [Transformers v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py).
  Suffix independence is an architectural expectation; compare actual
  prefix rows, positions and masks rather than assuming runtime equality.
- Verify source/native agreement at unchanged `atol=rtol=2e-5` separately
  for both inputs. Compare shared `c`, global rows, Action1 rows and
  Action2 rows against the A/A floor. A difference between *different*
  inputs is an intervention effect, not a numerical parity failure.
- Reuse buffers, keep computation on GPU, and inspect only at the existing
  terminal comparison boundary. No decoder or full DiT is needed.
  The dedicated runner is `examples/hy_nomas_refiner_transfer_probe.cr`,
  with the source oracle in
  `spec/support/hy_nomas_refiner_transfer_reference.py`.
- Synthetic Action2-row perturbations can precede the semantic pair as a
  cheap sensitivity probe, but do not establish transfer of a real
  instruction. A positive semantic result still needs a later matched
  visual counterfactual to connect it to the acting failure.

If unchanged global/Action1 outputs remain within A/A variation while
Action2 outputs change, deprioritize a refiner split for that pair.
If transfer is substantial, separate the shared-`c` route from text
self-attention in a follow-up ablation before treating all sharing as bad.
Then measure the direct DiT route before choosing a controller:
on saved refined states, compare M00/M10/M01/M11, where the two bits enable
the direct ROI-video-query to Action1/Action2 text-key routes. Keep text
query routes, refined states, time geometry, noise and T0 fixed. M00 means
**no direct ROI-video to action-text route**, not
action-free conditioning. Start with a same-latent DiT forward, not four
decoded videos. Measure velocity deltas inside each actor ROI, outside both
and in any contact region, including the interaction
`v11 - v10 - v01 + v00`; non-additivity is not itself an error.
Swapping actor ROIs is only a stress test: disagreement
could reflect a strong named-subject prior rather than a broken mask.

## First measured refiner transfer

The frozen V224 pair changes only Resume's Action2 wording: a glove raise
and chest tap becomes keeping both gloves at its sides without a tap.
The raw prompt length, crop, token positions, attention mask and action
spans remain equal. Global/Action1 Qwen rows `[0,132)` are bit-identical
before padding normalization (473,088 BF16 values). Changed valid input
rows are confined to `[193,224)` inside Action2. Padding `[224,1280)` is
explicitly normalized to zero in both inputs; this is not a claim that the
uncropped, unnormalized full encoder outputs match.

Fresh upstream Qwen states for B were captured with Transformers 4.57.3
and Torch 2.8.0 on MPS. The original project pins Transformers 4.57.1;
the actual capture does not qualify that runtime. The refiner oracle uses
Tencent source `911cea6458a89c2663513ff7099561ec190dd491`, its authenticated
42-tensor subset converted from F32 storage to BF16, and the previously
qualified MPS attention adapter. The whole original DiT checkpoint was
not downloaded or authenticated by this experiment.

All five checkpoints in each of A0/A1/B agree bit-for-bit between native
and source: masked mean at the BF16 Linear input, shared `c`, time-only
`vec`, and the valid-prefix outputs of both refiner blocks. Source and
native A/A are exact at all five checkpoints. The gate remains
`atol=rtol=2e-5`; no tolerance was widened.

| A1 versus B boundary | RMS(delta) / RMS(A1) | Maximum absolute delta |
| --- | ---: | ---: |
| Masked mean at BF16 Linear input | 2.9905% | 1.1875 |
| Shared refiner `c` | 3.9341% | 3.0 |
| Exported time `vec` | 0% (exact) | 0 |
| Block 1 global rows | 0.3821% | 0.5 |
| Block 1 Action1 rows | 0.3786% | 0.75 |
| Block 1 Action2 rows | 32.7017% | 109.625 |

This demonstrates transfer from the changed instruction to representations
of unchanged global/Action1 text in this pair. It does not identify the
relative contribution of shared `c` and text attention, establish a
visible acting effect, or show that cross-object sharing is undesirable.
The shared-`c` discriminator below separates conditional effects before
the same-latent M00/M10/M01/M11 DiT experiment.

The native runner prepares weights once and reuses its input, refiner and
snapshot buffers in one command chain. Instrumentation observes zero named
GPU-buffer allocations, activation readbacks or waits during encoding, one
terminal drain, then 15 checkpoint readbacks. This excludes opaque MPS
allocations and CPU bookkeeping. The named-buffer plan is 1,464,800,260
bytes under a 2 GiB budget; it is not a measured total GPU peak. Source
captures keep snapshots on GPU and export once at their terminal boundary.
The source mean-input hook verifies that MPS autocast produces an F32 mean
before the BF16 Linear cast.

All guarded processes completed with the cache-aware 10% floor and 64 MiB
swap-growth stop intact. Observed swap growth was zero; the smallest
sampled available headroom was 18.26 GiB during Qwen extraction. The short
native run's process samples cannot certify an instantaneous GPU peak.
No full DiT, decoder or new render was run in this slice. Timings from this
single qualification are not a performance comparison.

Evidence is retained outside git in
`/opt/mldata_ssd/cogni-ml/hy-nomas-refiner-transfer-20261105/`:

- `source-qwen-pair-audit.json` binds the semantic/token checks to the two
  source Qwen manifests. The native admission also authenticates their
  payloads, crop ranges and the accepted A generation bundle.
- `source-v2.json` SHA-256:
  `8974bfce4bb557f84a1c75be168e4e7ddc623b6fd6552c796360771bd201a877`.
- `native.json` SHA-256:
  `9ef4a28d6aa1595b452907cf99421670d938c7341237f7ff254313ed159d1d12`.
- `qualification.json` indexes source/binary pins, guarded command receipts,
  numeric results and seeded negatives. Source/native `*.argv.json` and
  build/admission receipts record the executed commands; repeat with fresh
  output paths because the probes reject replacement of existing reports.

CPU admission accepts the actual pair and rejects altered geometry and a
failed source A/A floor before Metal setup. Comparator controls reject an
out-of-tolerance value. These checks guard the oracle and admission; they
do not validate a controller or a visual-quality improvement.
The final read-only adversary review returned ROBUST for this frozen pair's
transfer and numerical-parity claim. It is a same-lineage review, not an
independent reproduction. Native admission relies on the trusted source
manifest digest; its A/A metadata checks are less strict than the source
producer's complete-record validator.

## Measured shared-conditioning discriminator

The factorial probe reuses the authenticated V224 semantic pair above and
intervenes only at the refiner's shared `c`, after the timestep/context
addition and before either refiner block. The production attention algorithm,
weights, masks, precision and input geometry remain unchanged.

| Case | Text rows | Shared `c` | Purpose |
| --- | --- | --- | --- |
| A0 | A | Natural A | Baseline, immutable GPU snapshot of `cA` |
| A1 | A | Frozen `cA` | No-op override and snapshot-alias control |
| B | B | Natural B | Combined intervention, immutable GPU snapshot of `cB` |
| X10 | B | Frozen `cA` | Text intervention at fixed shared conditioning |
| X01 | A | Frozen `cB` | Shared-conditioning intervention at fixed text |

The source producer authenticates and replays the parent A0/A1/B outputs.
The native probe uses the production encode for A0/B and a diagnostic-only
override for A1/X10/X01. Both implementations keep frozen conditioning and
captures on GPU until the existing terminal comparison boundary. A1 must
reproduce A0, overridden `c` must equal its frozen source, and time-only `vec`
must remain unchanged. The no-op and all 14 native factor invariants are exact;
the source payload also reproduces all 15 parent checkpoints bit-for-bit.
All 25 source/native checkpoints are bit-identical at the existing
`atol=rtol=2e-5` gate. Seven new CPU tests cover malformed factors, missing
cases/checkpoints, parent binding, source A/A and replay defects, and a seeded
numerical mismatch. Native CPU admission accepts the actual child and parent
before Metal allocation; its self-tests also reject a changed used-`c` value.

The second refiner block gives these `RMS(effect) / RMS(A0)` measurements:

| Output span | Text at fixed `cA`: X10-A0 | `c` at fixed text A: X01-A0 | Combined: B-A0 | Interaction: B-X10-X01+A0 |
| --- | ---: | ---: | ---: | ---: |
| Global | 0.382001% | 0.030372% | 0.382068% | 0.036175% |
| Action1 | 0.378594% | 0.028422% | 0.378625% | 0.037549% |
| Action2 | 32.701707% | 0.027275% | 32.701741% | 0.037050% |

The fixed-`c` intervention leaves transfer to unchanged global/Action1 rows
well above the exact A/A floor. After conditioning is fixed, attention is
the refiner's only remaining cross-row operation. Text self-attention is
therefore the larger measured conditional route for this pair; freezing
shared `c` alone leaves nearly the same Action1 effect as the combined
intervention. This is not a percentage attribution to independent channels:
`c` can alter later attention, and the interaction is nonzero. No conclusion
about visible acting or the value of removing useful context follows yet.

Interaction is computed elementwise in Float64 before any RMS reduction.
A separate CPU calculation authenticates the source payload, its parent
replay/no-op and used-`c`/mean/time invariants, then recomputes all 24
block/span effect summaries. Native and independently computed statistics
agree within 3.56e-15 absolute difference. This post-terminal check does
not insert a readback or wait into the GPU chain.

The native runner is `examples/hy_nomas_refiner_c_factorial_probe.cr`; its
source oracle is `spec/support/hy_nomas_refiner_c_factorial_reference.py`.
It records one prepared-weight encode and one command chain, zero named
GPU-buffer allocations/readbacks/waits during encoding, one terminal drain,
then 25 checkpoint reads. The aggregate named plan is 1,472,201,732 bytes
under 2 GiB; opaque MPS resources are excluded. Both model runs retain the
cache-aware 10% floor, 6 GiB process cap and 64 MiB swap-growth stop. Observed
swap growth is zero; minimum sampled headroom across source/native runtime
and admission stages is 28.51 GiB. The short native run's sampled process
peak is 1.64 GiB, not a certificate of instantaneous total GPU memory.
No full DiT, decoder or new render was run, and these single qualifications
are not speed benchmarks.

Evidence is outside git in
`/opt/mldata_ssd/cogni-ml/hy-nomas-refiner-c-factorial-20261105.w8oj5rem/`:

- `source.json` SHA-256:
  `8f9cb5cc6b7df387be446a93bdc3d550818fc930746ea4ecbf8b16f3807bf95b`.
- `native.json` SHA-256:
  `c5b5676b8b1ab962256675bd78c889903b8a9ef0c63737192e612759d7c95d09`.
- `qualification.json` binds current probe/source/binary/object hashes,
  the inherited source inputs and weights, actual command receipts, numerical
  results and guard limits. Replays need fresh output paths. The child
  manifest inherits the authenticated parent source fields; the current
  factorial producer is pinned separately in this index.

Final same-lineage adversary verdict is ROBUST for the conditional numerical
factorization and parity claim, not an independent reproduction or a visual
control verdict. Rollback is omission of the diagnostic probes; production
control and generation APIs are unchanged. Refresh on source/runtime, weights,
input/layout, precision, binary or evidence changes. The subsequent same-latent
direct DiT M00/M10/M01/M11 qualification is completed below; the current next
candidate is the bounded soft spatial routing dose response.

## Direct DiT route experiment contract

This bounded CAUTION slice is diagnostic only. It uses the original accepted
sequential-actions ROI take, not the later look-down rewrite or semantic B.
Its eight authenticated captures freeze T0/T1 patch, refined valid text and
time vector, T1 initial latent and the accepted first-step velocity. There
is no scheduler update, decoder or render. Rollback is omission of the new
probe; the production arithmetic and accepted take remain the control.

For each of all 54 blocks, enable the same two route bits by clearing only
the selected **video**-query to matching action-text-key edges. This is mask
equivalent to setting the selected query labels to global zero; the source
probe edits edge ranges directly. Never relabel action keys or text queries.
Preserve the authored spatial ROI clearing,
all history/current-video/global-text keys, text-query submatrix and all
non-target pairs. Relative to M11, M10 removes exactly 44,160 Action2 edges,
M01 removes 47,061 Action1 edges, and M00 removes their 91,221-edge union;
every case adds zero edges. A changed text-query route rejects the probe.

Run M11, M00, M10, M01 and M11_repeat. M11 must first replay the accepted
parent velocity; the repeated arm establishes the A/A floor. Full-forward
source/native comparison keeps `atol=rtol=2e-5`. The compact execution
schedule computes T0 then all five T1 arms at each layer, reusing one
layer's K/V and block scratch rather than retaining all 54 T0 caches or
layer output banks. Case state banks stay distinct. Equivalence of this
schedule is a hypothesis until the parent replay passes. Source means the
pinned Tencent equations through the BF16 MPS adapter, not stock CUDA.

Predeclare velocity regions at `[channels=32,time=30,height=9,width=16]`:
Alex `h[0,9),w[1,8)`; Resume `h[2,8),w[10,15)`; their outside union;
Action1/Alex `t[1,10)`; and Action2/Resume `t[12,28)`. A coarse authored
Resume chest proxy is `t[12,28),h[4,6),w[11,13)`; it is not an observed
contact or segmentation result. It overlaps Resume and must not be counted
as an independent partition. Compute `v10-v00`, `v01-v00`, `v11-v00` and
`v11-v10-v01+v00` elementwise in Float64 before regional summaries.
Report conditional route sensitivity, not successful acting, independent
route percentages, semantic synergy or a visual-quality improvement.

DoD is the new `spec.hy_nomas_dit_route_factorial_test` CPU suite, release
build and actual-fixture admission of
`examples/hy_nomas_dit_route_factorial_probe.cr`, then guarded source/native
full forwards with exact mask changes, accepted M11 replay, A/A and the
unchanged numeric gate. Observe named allocations/readbacks/waits during
encoding, preserving zero new in-chain synchronization. Recheck the
cache-aware 10% memory floor and 64 MiB swap-growth stop before execution.
Failure of replay, mask attribution, numeric comparison or either memory
guard stops this slice before a new render.
At block 0, all text outputs and video rows outside the disabled routes must
remain exact. The two disjoint query regions must have zero elementwise
image interaction there; a later interaction may arise through global video
relay. These local checks distinguish a targeted intervention from an
unintended change to every row.

## Measured direct DiT route sensitivity (source/native qualified)

The five arms ran through all 54 pinned source blocks on the same saved
refined inputs, T0, timestep and noise. M11 exactly replays all five accepted
parent checkpoints, including the final velocity. M11_repeat is exact at all
five checkpoints. At block 0, every text output and every video row outside
the disabled routes remains exact, and the elementwise image interaction is
zero. This supports the targeted-mask attribution for the frozen fixture.

The final velocity has these conditional effects. Each percentage is
`RMS(delta) / RMS(M00 in the same region)`, with elementwise Float64 subtraction
before the regional reduction. The regions were declared before execution.

| Region | Action1, Action2 off (`v10-v00`) | Action2, Action1 off (`v01-v00`) | Interaction (`v11-v10-v01+v00`) |
| --- | ---: | ---: | ---: |
| Alex during Action1 | 18.19582% | 1.02300% | 9.54723% |
| Resume during Action2 | 1.06056% | 9.28511% | 2.06126% |
| Resume chest proxy | 1.12107% | 19.33723% | 3.52204% |
| Outside both actor regions | 1.42065% | 1.18584% | 1.38269% |

The own-actor contrast is larger than the other-action contrast in each
authored actor window. This is evidence of localized numerical sensitivity,
not correct pose, gesture, contact or action completion. The chest proxy is
coarse authored support, not a segmentation or contact observation. Effects
outside the two regions and the later nonzero interaction reject a hard
spatial fence or additive independent-route interpretation. M00 retains the
same refined text, shared conditioning, global keys and T0 history; it does
not mean an unconditioned model.

The qualified source run stages one BF16 block at a time: 324 block-attention
calls and 5,832 query chunks, one explicit terminal synchronization and one
packed activation export. All F32 snapshot promotions are encoded before
that terminal synchronization. No scheduler, refiner, decoder or render is
called. The source guard retains the cache-aware 10% floor, an 8 GiB process
cap and a 64 MiB swap-growth stop; observed swap growth is zero. This single
qualification is not a speed benchmark or stock CUDA reproduction.

Evidence is outside git in
`/opt/mldata_ssd/cogni-ml/hy-nomas-direct-routes-20261105.klzuhg0r/`:

- `source-qualified.json` SHA-256:
  `efcf746fd334218d25b0a922647f10962c6d5102144f7d80cff67bb327a40c89`.
- `source-qualified.tensors.f32le` SHA-256:
  `2f2c634a97787df22be094efafb778924a545827192c6c50597a63df0578a24a`.
- `source2-independent.json` recomputes exact replay, A/A, block-0 locality
  and the Float64 effects directly from authenticated output bytes; its
  seeded numerical negative is rejected. Its SHA-256 is
  `d226c3b835a777a24b5482098bf1fa05407444d52d4ddb80262a7f6a73002644`.
- `source2.executed-runtime.json` SHA-256 is
  `88f95c92c060fdd25019ba5f85359ce9a431fdc8597f2ecda20f5639019ee71e`;
  it pins the executed source runtime at
  `b8e45aaeb207365f3e2927c461e5720d0b673f8b2f19034837e3956fec8294fb`.
- `qualification-source.json` preserves the earlier source-only index with
  native explicitly pending. It is historical, not the final native receipt.
  Its unchanged SHA-256 is
  `2a48f4140f52fa8f8497e1d3f26f1abfd1655f7756cbc8f0e34b7144e8a27625`.
- `native-fixed.json` SHA-256:
  `b879232edbeb83d78ef108e80d8c5396628b94d8826e2171ab6b672b6377de87`.
  Its 375,009,280-byte payload has the same SHA as the qualified source
  payload: all 25 checkpoint arrays are bit-identical.
- `native-fixed-independent.json` SHA-256:
  `4c49ea9e5dfaebf01d323e11a8101eb48ba6b21828da882bd4075e42b9073554`.
  The separately authored checker authenticates manifest, payload and tensor
  ranges, and recomputes all 25 comparisons, five native parent checkpoints,
  five native A/A checkpoints, exact block-0 locality and seven Float64
  regional effect summaries. Its seeded numerical negative is rejected.
  A metadata-only replacement of the parent manifest is also rejected.
- `qualification.json` binds final code, executed source runtime, native
  binary and actual linked objects, frozen inputs, command/guard receipts,
  independent-checker code and persistent-model recovery receipts. The
  checker uses Python assertions; the executed unoptimized invocation and
  negative controls keep those gates active. An index is not proof itself.

Final same-lineage adversary review is ROBUST for this frozen source/native
numerical sensitivity claim. It is not an independent reproduction, stock
CUDA comparison or visual-control qualification.

The native release build, CPU self-test, actual-fixture CPU admission, three
malformed-manifest rejections and 54 focused CPU tests pass. One native
command chain reuses the planned buffers across all 54 blocks and five
arms: zero named allocations or activation readbacks during encoding, one
terminal drain, then 25 checkpoint reads. There are no scheduler or decoder
calls. The named plan peaks at 19,625,061,552 bytes (18.2773 GiB), excluding
opaque MPS resources. Sampled process peak is 20,325,119,608 bytes; observed
swap growth is zero. The cache-aware 10% floor, 22 GiB process cap and
64 MiB swap-growth stop remain unchanged. Neither sampled process memory
nor the named-buffer plan certifies instantaneous total GPU peak. The
54.90-second qualification forward is not a generation-speed benchmark.

The first native run with a saved payload failed before qualification.
Its authenticated block-0 differences were confined to all 144 video rows
in frames 0, 10, 11, 28 and 29; the other 25 frames were bit-exact. Source
label `-1` means an active global-only video query outside action windows.
The native attention bridge instead reserves negative query labels for
padding and zero-scatters their context. The diagnostic binding now maps
only those 720 active video-query labels to native global `0`; both values
permit exactly global keys, so no Boolean mask edge changes. Self-tests
check the translation count and exact preservation of all remaining labels
for every arm. Source labels and mask hashes remain frozen; production
padding semantics are unchanged. Failed `native.json` and the range-hashed
`native3-block0-diagnostic.json` remain available to falsify this diagnosis.
Diagnostic output creation also uses exclusive `O_EXCL` writes; tests reject
replacement of an existing file or symlink without touching its contents.

Deleted DiT weights were recovered into the separate persistent folder
`/opt/mldata_ssd/models/hy-nomas/`, outside disposable experiments. The
34,213,991,280-byte pinned checkpoint and all 55 BF16 cache files
(16,623,052,016 bytes) match their prior hashes. `README.md` and
`recovery-verified.json` identify the location and exact checks; legacy
weight paths are directory aliases into that store. Q4 derivatives only
reproduce the existing cache index; this probe uses BF16. Recovery of other
pipeline models and Q4 visual quality are outside this claim.

PixelUMM was subsequently recovered into
`/opt/mldata_ssd/models/hy-nomas/pixelumm-ft-bac719a8/`. The complete pinned
31,553,868,864-byte source and the stem/head subset files match their archived
SHA-256 values. CPU integer BF16 RNE reconstruction reproduces the archived
13,892,143,104-byte 36-block payload exactly, SHA-256
`152ba021b52d9546e4c8821b17ab11f67f46486d521b500de3766f14dd2217ca`.
The new manifest honestly records CPU conversion provenance; it does not
recreate the deleted MPS-provenance manifest.

The latent decoder accepts an explicit paired `--decoder-cache` and
`--decoder-cache-sha` override. The historical default remains unchanged.
The override retains regular-file, full-digest, source-lineage, dtype, shape,
inventory and payload checks, and records its actual canonical identity in
both plan and output receipts. Twenty-six Crystal examples, a release build,
nine compiled CLI rejection cases and the positive CPU plan pass. A separate
check binds all 468 names/shapes/offsets to the canonical archived payload
order. The plan authenticates the complete payload before GPU construction;
its 30-frame fixture is initial noise, not generated video. No inference or
visual-equivalence claim follows from these admission checks. Evidence:
`.landmark/evidence/2026-11-05-pixelumm-recovered-cache.json`.

After the native gate, the next bounded candidate is a soft spatial routing
dose response with saved refined states/T0/noise unchanged. Retain global
video/history context and test whether own-versus-cross-actor selectivity
improves without comparable outside-region effects. A numerical gain still
needs a short rendered pose/gesture check. If the instructions reach the
right region but articulation remains weak, pivot to explicit pose/trajectory
support rather than repeating prompt amplification.

## Proposed authoring and runtime boundary

Keep global appearance/style/camera separate from actions. Each authored
object needs a stable ID, a user-confirmable reference region and, where
needed, a part track. Each action needs an owner, interval, visibly distinct
start/end pose, target and permitted interaction. These are authoring
constraints, not model-enforced facts.

For the first shot, test a distinct Alex head pose with desk hands still,
then a clearly visible Resume glove raise/lower. Qualify a precise
chest-tap only after those simpler endpoints work. A region path describes
where a part moves; it does not encode head yaw or articulated contact.
Do not silently turn the sequential v2 API into simultaneous actor control.

Choose the next controller from evidence, in this order:

| Candidate | Entry condition and discriminator | Cost / strongest failure |
| --- | --- | --- |
| Separate text conditioning | Measured refiner transfer matters to the visual failure. Encode each action with shared G but without the other action, refine separately, then stitch typed rows. | Additional Qwen/refiner work; distribution change. Slicing Action2 from combined Qwen states does not remove earlier causal context. |
| Soft part tracks in attention | Direct-route ablation shows useful spatial sensitivity. Bias action-text access along soft head/glove tubes, preserving video/history context and explicit contact regions. Reverse a track as a negative control. | One DiT forward per step; authoring/tracking cost. Can localize an instruction without creating the right pose. |
| Regional velocity mixing | Cheaper routing gives useful but insufficient localized action changes. Compare same-z/t branch residuals before a short render. | Roughly three DiT forwards per step, one decode; serial scratch reuse. Seams, conflicting contact and no learned-joint-distribution guarantee. |
| Layer composite | A fixed-camera fallback needs stable background pixels and has a clean plate plus actor mattes. | Edge ghosts, lost shadows/contact; not a substitute for generated acting. |

Separate conditioning would use R0(G), R1(G+A1), R2(G+A2). The current
source-BF16 adapter admits V263/V150/experimental V224 only
(`hy_nomas_text_refiner.cr:229-251`); nominal V49/V132/V141 branches
are **not qualified**. Fresh exact-shape source gates or a proved padded
design are required. Shared video keys would still permit interaction
and transfer after this change. Removing an earlier action also shifts
later text positions: record actual token/position IDs and source-row gather
ranges. Retained post-crop row indices are not Qwen position IDs.

For the expensive fallback, at the same latent z and time t:

```text
v = v0 + MA * (vA - v0) + MR * (vR - v0)
```

Use soft masks in [0,1] with MA+MR <= 1, with a declared overlap/contact
policy. This fences the immediate conditioned output increment spatially,
not actor identity. Later steps can propagate changes outside the region
through global attention. It is not a guarantee of an unchanged background
or of physical consistency. Keep branch outputs on GPU and reuse scratch
serially; do not triple activation memory by batching without a budget.

## Bounded part-prior enrichment experiment (2026-11-05)

The next CAUTION slice preserves the accepted M11 hard actor/time support.
It adds a finite BF16 pre-softmax logit prior only on already-visible
video-query -> matching-action-text-key pairs. This is enrichment inside a
hard ROI, not softening its excluded edges. History, video and global keys
keep zero additive bias; all text-query rows keep their original mask.
Their probabilities can still change through softmax normalization.

For a query in the selected action window, use `bias = lambda * w(h,w)`.
The authored static prior is a separable tent, normalized to peak one, on the
9x16 video-query grid. Alex's center is `(h=1.5,w=5.5)` over frames `[1,10)`;
Resume's glove/chest center is `(4.5,11.5)` over `[12,28)`. Alex's radii are
two cells; Resume's are 1.5 vertical and two horizontal cells. Cell `(h,w)`
is centered at pixel `(32*(w+0.5),32*(h+0.5))` in the 512x288 reference.
These locations are authored from the pinned reference
overlay, not measured pose tracks or segmentation. Pin the image, full
per-query BF16 values and row ordering before GPU execution. Never infer
these priors from the resulting video.

Use eight fixed-input arms: zero, Alex at `ln(1.5)` and `ln(2)`, Resume at
the same doses, a high-dose spatial sham for each, and zero repeated.
The shams shift Alex's prior down three cells and Resume's up two cells,
preserving the per-frame value histogram and actor/time support. A rejected
radius-two Resume prior with a one-cell sham overlapped too strongly for a
useful negative control; it was replaced before GPU execution. The doses
multiply matching-key prior odds by approximately 1.5 and 2 before BF16
rounding. They do not multiply motion magnitude.

Falsifiers and DoD:

- CPU admission rejects nonfinite, negative, over-budget, out-of-support or
  text-query bias, mismatched hashes/geometry and altered sham histograms.
- A small MPS additive-mask preflight must pass before a full source probe.
- Zero and repeated zero use the old Boolean path and must replay the accepted
  M11 parent bit-for-bit at all five checkpoints. Every new native capture
  must match the pinned-source equation adapter at the existing 2e-5 bounds.
- First-block text and image rows outside each prior's support stay exact.
  Later global propagation is measured, not prohibited. Report raw and
  baseline-normalized velocity RMS in the intended part, actor complement,
  other actor and outside both actors; compare the matched spatial shams.
  Fix independent evaluation boxes before execution: Alex's head uses
  `t[1,10),h[0,4),w[4,8)`; Resume's glove/chest uses
  `t[12,28),h[4,6),w[11,13)`. For each action, require the intended high-dose
  target RMS and target/complement RMS ratio to exceed its own sham, and
  require target RMS not to decrease from low to high dose. An undefined
  ratio cannot pass. Apply these gates separately to source and native
  increments. In both, the high-dose target increment RMS must exceed
  `RMS(2e-5 + 2e-5 * abs(own_zero))` over the same fixed target region.
  Freeze this floor before inspecting effect measurements: exact A/A alone
  does not resolve a change below the source/native parity allowance.
  These gates measure resolved representation sensitivity only; they do not
  establish a perceptual effect or intended acting.
- Encode reuses buffers and the existing mask dispatch. Require zero named
  encode allocations/readbacks, no intermediate waits and one terminal drain.
  Recompute the named-byte plan for eight arms; retain the external process
  footprint guard, cache-aware 10% floor and 64 MiB swap-growth stop.

The strongest failure is localized representation change without the intended
head turn or glove action. Numerical localization only earns a separate short
render; it cannot establish acting. Stop before rendering on parity, zero/A-A,
resource or locality failures, or if the intended prior does not beat the sham
in final localization. Do not select a dose from an image-difference score
alone. If localization passes but articulation fails, pivot to explicit pose
or trajectory support instead of increasing the dose.

Rollback is the unchanged default/nil-bias path and the accepted take. The
optional history-attention API remains experimental until its own gates pass;
this slice does not admit a production spatial controller. Evidence decays on
changes to priors, input/parent, source, precision, weights, geometry or binary.

### Frozen-input result (2026-11-05)

All 40 source/native tensors from the eight arms are byte-identical, including
the first and last block image/text checkpoints and final velocity. Zero and
repeated zero replay all five accepted parent checkpoints exactly on both
implementations. First-block text and zero-prior image rows also remain exact.
The fixed tolerances are still `atol=rtol=2e-5`.

Both implementations independently pass the predeclared dose, sham and
localization checks; their numbers are identical:

| Target | Low-dose target RMS | High-dose target RMS | Sham target RMS | High target/complement RMS | Sham target/complement RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alex head | 0.158176 | 0.204026 | 0.064890 | 5.815283 | 2.664713 |
| Resume glove/chest | 0.158965 | 0.257119 | 0.201574 | 3.899654 | 2.397252 |

The respective fixed parity-allowance RMS floors are `5.927495e-5` and
`5.845713e-5`. These finite-case results establish resolved representation
sensitivity and locality, not a head turn, glove gesture or visible quality.
Do not combine the two nonzero priors and call that combination qualified;
the tested arms change one actor prior at a time.

Native encoding uses one chain, one terminal wait, zero named allocations and
zero activation readbacks before the drain; its 40 captures are read afterward.
The named peak is 20,208,934,020 bytes (18.82 GiB). External sampling observed a
21,121,988,648-byte native process-group peak and minimum cache-aware headroom
of 21,506,048,000 bytes, with zero swap growth. The source peak was
4,435,905,200 bytes. Both stages ended with no live child processes under the
10% floor and 64 MiB swap-growth stop. Sampling excludes between-poll spikes;
named accounting excludes opaque MPS resources. These qualification forwards
are not a speed benchmark.

The accepted oracle used the isolated PyTorch 2.8.0 MPS environment. Earlier
2.6.0 parent replay and 2.8.0 manifest-publication failures remain rejected
stages; they are not successful qualifications or a controlled attribution of
the difference to the PyTorch version. The flat accepted-ROI replay contract
has a regression through comparison, manifest rendering and validation.

Evidence: `/opt/mldata_ssd/cogni-ml/hy-nomas-soft-routes-20261105.j0wgy7yl/`.
`qualification.json` SHA-256 is
`a5763db3a63ac91ed02069f0a72b890b226735296bd59a7c52c345c4da03c96e`;
source and native payload SHA-256 is
`3631809566303dc23ac81ed8789de0bde7d4b622e7a7798295bc41bdd03979f2`.
The strengthened effect floor was frozen before observing these effects.
The curated repository index is
`.landmark/evidence/2026-11-05-hy-nomas-soft-routes.json`; its external
`qualified-artifacts.json` binds the exact source/build, failed stages and
guard receipts. The final focused suites passed 87 Python tests and eight
Crystal examples. Same-lineage read-only review is correlated, not an
independent replication.

The bounded four-step visual comparison is now recorded below. Its numerical
gate passes, but the requested articulation is not established. Retain the
numerical result and pivot to explicit pose/trajectory support without
escalating dose.

### Bounded four-step render contract (2026-11-05)

The admitted render compares `zero`, `A1_high` and `A2_high` separately, with
the already qualified doses and hard M11 support. All three start from the
saved parent `t1.initial_latent` (SHA-256 `503615dc8920afe80d441fa025380c4b764b1e81f07d3a44525127010ddb2e5a`).
The conditioning anchor supplies the authenticated Qwen states only: its
different `input.latent` is explicitly unused. The pinned noise-selection
record binds that choice before model execution.

Four Euler steps use the accepted shift-5 schedule. Every step restarts T0
from its fixed patch/text states; each layer prepares one T0 K/V pair shared
by the three T1 arms. This recomputes T0 across steps rather than retaining
54 layers of history. The native path reuses one staged block, one-layer K/V
and preallocated arm states. Final-head output must populate the same GPU
velocity buffer consumed by that arm's Euler update.

Before decoding, independently compare all twelve post-Euler latent states,
three first-step velocities and zero-arm patch/text/vector against source at
the unchanged tolerances. Both implementations must exactly reproduce all
four accepted zero Euler states and its first-step prelude/velocity, plus the
three previously qualified first-step velocities: eighteen parity checks and
twenty-two exact replay gates. A shared wrong trajectory must fail despite
source/native agreement. Check execution receipts separately from numerical
values: zero named encode allocations/readbacks, one native terminal wait,
and observed per-arm attention counts. External guards retain the 10% floor,
64 MiB swap-growth stop and a 22 GiB process cap; the named ceiling is 21 GiB.

If those gates pass, decode the same clean T0 followed by each new T1 using
the explicitly pinned recovered BF16 PixelUMM cache and identical pixel
noise. Export only the new 120 frames (five seconds at 24 fps) for comparison.
Inspect head/gaze, Resume glove/body, hands/contact and appearance separately;
pooled image differences only locate change. No combined actor bias or
production controller is admitted. Rollback remains omission of these
diagnostic CLIs. The render result expires on source, binary, weight, input,
noise, bias, geometry, schedule or decoder changes.

### Four-step replay passes; requested articulation remains open (2026-11-06)

The three separate arms complete all four Euler steps through 54 BF16 blocks.
All eighteen source/native tensor pairs are byte-identical, and all twenty-two
exact replay gates pass independently against the accepted frozen captures.
Both payloads have SHA-256
`7a6aaee1339832106c52578fa6b2aff2c6f925bb50346153c4ce6bffc8d26d09`;
the current `render-v1/qualification.json` SHA-256 is
`2bf0aea53b2f4f3aa324dac1c9145e42f174d4b0df8c78e41d5e477fbb0ad792`.
Do not substitute the older eight-arm qualification at the evidence root.
The source oracle is the pinned Tencent equations through PyTorch 2.8.0 MPS
BF16 adaptation, not stock CUDA/Magi. Tolerances remain `atol=rtol=2e-5`.

The new diagnostic reuses GPU arm states, velocity, one-layer history and
staged weights. Native encoding records one chain and one terminal wait,
zero named encode allocations/readbacks, and eighteen post-drain captures.
Four outer steps require four ordered weight-stage passes; sixteen logical
T0/T1 forwards do not require sixteen passes. The stage-count receipt field
comes from the outer-step constant, not a private cursor read. A focused
cursor test admits four ordered passes and rejects the fifth.

Admission uses the resident BF16 weight size, not compressed Q4 payload size.
Named peak is 20,279,905,400 bytes, below the 21 GiB named ceiling. Sampling
observes a 20,588,213,328-byte native process-group peak. All source/native
and three decoder guards preserve the cache-aware 10% floor, zero observed
swap growth and no final live children. Each decoder uses the same recovered
BF16 cache and pixel-noise selection, one terminal wait and zero named encode
allocations/readbacks. Sampling excludes between-poll peaks and named
accounting excludes opaque Metal/MPS resources. These are qualification runs,
not a controlled performance benchmark.

Each decoded take exports the new 120 frames at 512 x 288 / 24 fps. The
comparison labels are `Baseline`, `Alex bias`, `Resume bias`. Inspection of
frames 0, 12, 24, 48, 72, 96 and 119 finds gaze/expression changes in the Alex
arm, but no substantial requested head turn. The Resume arm retains essentially
the same late glove-to-chest gesture visible in the other two arms. The paper
body and legs are mostly fixed in these selected frames. This is not a
full-playback review, pose/keypoint measurement or anatomy/contact certificate.
The small whole-clip pooled image-change differences do not establish acting.
No rendered sham, simultaneous bias or production object controller is qualified.

The predeclared articulation gate is therefore not satisfied by the available
visual evidence. Stop attention-dose escalation. A read-only audit of pinned
upstream `911cea6458a89c2663513ff7099561ec190dd491` finds camera trajectories
in `hymm/pipeline/bidirectional.py:265` and cached camera injection in
`hymm/pipeline/causal_ar.py:1177`. `joint_control_spec` is benchmark metadata,
stripped before encoding and restored only to the sidecar
(`hymm/inference.py:91`, `:169`). The accepted local route has no actor-local
pose/part-trajectory input. This does not prove full DiT retraining is required.
The next admissible investigation is a compatible pose/part conditioner;
its first positive test needs one head-yaw trajectory, neutral and spatial-sham
controls, and protected Resume/background measurements. Camera movement must
be separated from head movement relative to the background.

Release build, CPU admission, one Crystal cursor example and 37 focused Python
tests pass. A seeded wrong shared zero trajectory, late-step perturbation,
missing capture, extra wait/readback, incorrect memory basis and stale or
noncanonical output paths fail closed. The focused Python DoD is:

```sh
python3 -B -m unittest spec.hy_nomas_soft_route_render_qualify_test \
  spec.hy_nomas_soft_route_render_reference_test \
  spec.hy_nomas_soft_route_qualify_test spec.hy_nomas_export_test -q
```

The curated index is
`.landmark/evidence/2026-11-06-hy-nomas-soft-route-render.json`.
Its external `render-v1/qualified-artifacts.json` binds the exact commands,
guards, binary, 106 source / 102 native snapshot records, captures, three
decoder receipts, videos, visual review and rejected stages. Source bytes and
saved copies were hash-rechecked before final Crystal formatter alignment.
`final-formatting-equivalence.json` separately binds the archived originals
to the formatted CLI/support files: identical parsed AST renderings, line
counts and per-line non-whitespace contents; 37 Python tests, the cursor test
and format check pass afterward. GPU results remain tied to the frozen v5
binary and pre-format snapshots; no post-format GPU rerun is claimed.
`final-python-eof-equivalence.json` separately binds removal of one trailing
blank line from the oracle: its Python AST including positions is unchanged,
37 tests pass again, and the staged whitespace check passes.
Inherited object compile commands are not reconstructed and runtime metadata
is not a complete runtime byte snapshot. Same-lineage adversary review is
ROBUST within that frozen numerical scope and is correlated. The diagnostic
CLIs remain opt-in; omit them for rollback. Any changed bound input, source,
runtime, weight, binary/object, precision, geometry, schedule, bias or decoder,
or missing/mutated evidence, reopens the affected claim.

### Query self-correspondence screening (2026-11-06, qualified diagnostic)

The next bounded probe tests an attention prerequisite, not a controller.
QWERTY (arXiv:2607.01869, section 3) motivates transporting source queries
when they prefer the corresponding source keys. Its full-sequence,
50-step experiments do not establish compatibility with four-step causal
Hy-Nomas. The local probe implements neither STCD nor query warping.

At 512 x 288 the latent grid is 16 x 9: one cell spans 32 x 32 pixels.
The Resume glove occupies approximately one such cell. Direct latent
translation cannot isolate the glove from the wrist/page, and image-plane
rotation cannot specify head yaw. Neither is admitted as an acting fix.

The opt-in native probe computes F32 first-frame QK scores and six GPU
metrics on saved, losslessly BF16-expanded Q/K of shape [144, 16, 128].
The source-equation oracle uses the same inputs on MPS. Preserve
`atol=rtol=2e-5`, exact discrete metrics, reused buffers, zero in-chain
readbacks/waits and one terminal wait. A post-drain independent audit checks
the exported metrics against the scores. Positive identical-Q/K, shuffled
keys and all-tied controls must distinguish strict preference from a stable
argmax tie. Existing noisy T1 Q/K do not establish a reference-cache premise.
Capture operational T0 Q/K at the same upstream block-0 hook before attention;
retain the preprocessing scope and authenticated fixture lineage. The saved
native T0 bundle declares sigma 0.206897 / timestep 206.897, not zero noise.
Do not promote a same-state T0 measurement to a clean-input QWERTY result.
The 144-key self mass is restricted-frame affinity, not a full model attention
probability; the authored rectangles are not validated object masks.

This is CAUTION diagnostic work. Omit the new CLIs for rollback; production
generation remains unchanged. Runs use the existing cache-aware 10% floor
and 64 MiB swap-growth stop, with bounded process caps recorded per stage.
The executed DoD passes: 15 CPU tests, native release build and seven
CPU-only admission cases, three paired GPU controls and one authenticated
T0 capture. The identical-Q/K control has strict self preference 1.0;
shuffled and tied controls have 0.0. All four native/source score arrays
are byte-identical. T0 metric error is at most 1.7881393432617188e-7;
the largest metric difference across the controls is 1.1920928955078125e-6.
Exact discrete comparisons and both independent post-drain metric audits
pass. The native diagnostic uses 3,741,696 named bytes, two dispatches,
zero named encode allocations/readbacks/waits, one terminal wait and two
post-wait diagnostic readbacks. This does not qualify the full BF16
attention path or the legacy source capture's synchronization behavior.

The captured source block is pinned to `a31c0fe0bcc493be188fd37ddba4ce2f674becb4`;
`911cea6458a89c2663513ff7099561ec190dd491` is the separate timeline
preprocessing provenance. The block consumes authenticated native
image/text/vector/RoPE tensors. Latent, noisy-latent and timestep records
are authenticated but not consumed, so this is not an independent replay
of noise-to-preblock preparation. Rewrapping the embedded BF16 latent
reproduces the missing original safetensors SHA `9e21e8b9...` exactly;
the original manifest/payload remain unchanged. JSON receipt key order is
not semantic: admission checks exact inventory/count, header records and
contiguous tensor extents instead of comparing dictionary insertion order.

Real block-0 strict self preference is 572/2304 head-position pairs
(24.83%). The authored Alex/Resume rectangles give 23.05%/23.54%.
Heads 12 and 15 differ markedly from the pooled result: 120/144 and
136/144 positions prefer their own key. Within the authored Alex rectangle
they reach 12/16 and 11/16; within Resume, 23/30 and 29/30. This is an
exploratory head-selection signal from one state, not validated actor
segmentation or a transported-query result. The pooled restricted-frame
self mass is 0.05869 versus uniform 1/144; it is not full attention mass.

All ten qualified guarded stages finish with zero live children and zero
observed swap growth. Source-capture sampled peak is 1,732,561,032 bytes;
the 10% floor is preserved. Sampling misses between-poll spikes, and the
short native runs do not establish a measured native peak or speedup.
Evidence: `.landmark/evidence/2026-11-06-hy-nomas-query-affinity.json`;
external `query-correspondence-26ede1fe/qualified-artifacts-v1.json`.

Refresh with `python3 -B -m unittest spec.hy_nomas_query_correspondence_test
spec.hy_nomas_query_capture_test -q`, the saved build/guard argv and the
qualifier's `--verify-model-capture` mode. A numerical pass only admits the
affinity measurement. Next, test the head/layer selection on held-out
saved states before attempting transport; do not apply an all-head warp
based on the pooled result. A transport probe still needs neutral and
equal-area spatial-sham controls, protected actors/background, source
residue checks and a causal rollout before any visible-control claim.

### Frozen-head robustness and RoPE discriminator (2026-11-06, qualified diagnostic)

The following design was frozen before scoring; its saved declaration and
frontier snapshot are indexed with the results.

Freeze heads 12 and 15 from the discovery frame; do not select replacements
after viewing new scores. The authenticated source artifact also contains
all 30 frames of pre-RoPE QK-normalized and post-RoPE BF16 Q/K. Select frames
1, 14 and 29 before scoring. These are held-out row ranges in the same T0
forward, not independent scenes, noise seeds or model runs. Record exact
range identities and duplicate content; do not treat patches as independent
samples. The saved alternative runs have preblock inputs or K/V caches but
no usable authenticated Q/K capture for an independent-state comparison.

The exploratory high-diagonal-affinity screen requires each frozen head
to have at least 108/144 strict diagonal maxima on every selected post-RoPE
frame. Report each head/frame separately even if the screen fails. This
75% screening threshold is not an accuracy tolerance or a transport gate.
Compare pre/post-RoPE on each selected frame and the discovery frame's
pre-RoPE companion. A fixed key roll by 17 positions on frame 14 at both
boundaries must permute the score columns exactly; it is an equivariance
control, not a content-destruction test with position phases preserved.
It moves RoPE phases with the key. A pre/post contrast alone cannot prove
semantic matching or establish that RoPE causes any observed preference.

This CAUTION slice reuses the frozen native binary and equation oracle,
adds only opt-in CPU fixture preparation and post-drain evaluation, and
changes no production path. DoD: targeted CPU lineage/permutation falsifiers,
all nine paired native/MPS cases at unchanged `atol=rtol=2e-5` with exact
discrete columns and post-drain audits, exact shuffled-score equivariance,
and the existing 10% cache-aware floor / 64 MiB swap-growth stop. Inference
math stays on GPU with the existing one terminal wait; input byte selection
and diagnostic evaluation happen outside the chain. Rollback: omit the new
evaluation CLI. The strongest failure is a position/content proxy being
mistaken for actor correspondence. Transport remains blocked on independent
states with nonzero-displacement labeled correspondences beating same-location
and equal-area shams, followed by causal protected-actor/background controls.

The bounded DoD passes: 23 CPU tests, four real-fixture admission falsifiers,
all nine GPU native/source pairs and four exact source/native roll checks.
Every score array is byte-identical; the largest metric difference is
2.682209014892578e-7. Independent post-drain audits of both metric outputs
pass with maximum error 4.76837158203125e-7. Discrete metric columns and
source/native per-head strict counts, fractions and screen outcomes agree
exactly. Floating summary means are not byte-identical; the frozen numeric
tolerances were not changed. An exploratory whole-summary dictionary-equality
assertion was rejected in favor of exact discrete comparisons plus the
existing numeric checks; this was an audit correction, not a relaxed gate.

Strict diagonal maxima, out of 144 positions per head:

| Held-out frame | Head 12 before RoPE | Head 12 after RoPE | Head 15 before RoPE | Head 15 after RoPE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8 | 109 | 13 | 138 |
| 14 | 9 | 118 | 12 | 139 |
| 29 | 10 | 121 | 15 | 136 |

Both heads meet the frozen 108-position post-RoPE screen on all three
frames; head 12 at frame 1 clears it by only one position. Frame 0's
pre-RoPE companion gives 10/144 and 19/144. No heads were reselected.
The fixed key roll gives zero post-RoPE diagonal winners for both heads
on frame 14, while permuting score columns and stable argmax correctly.
This confirms indexing, not position-independent content matching.

For each held-out head/frame, mean absolute change of the same-position
logit across RoPE is only .0031-.0038, while mean diagonal margin improves
by .48-.68. In exact arithmetic a shared orthogonal rotation preserves
the same-position dot product; relative rotations change off-diagonal
competitors. BF16 rounding remains present here. The measured contrast is
consistent with positional structure, but does not isolate its causal
contribution or establish semantic tracking. The screen is not a transport gate.

All 21 guarded stages exit zero with no final children or observed swap
growth. Minimum cache-aware headroom is 25,206,915,072 bytes and minimum
reported pressure-free percentage is 91; the 10% floor, 2 GiB stage cap
and 64 MiB swap-growth stop remain enabled. Maximum sampled process-group
footprint is 191,186,288 bytes; polling misses short peaks. This is neither
a qualified native peak nor a speed benchmark. The unchanged native binary
keeps its existing one terminal wait and post-drain exports. No production
path or full generation was run. Same-lineage read-only adversary review is
ROBUST within this measurement scope and is correlated, not independent replication.

Evidence: `.landmark/evidence/2026-11-06-hy-nomas-query-head-robustness.json`;
external `query-heads-ht33zt1c/qualified-artifacts-v1.json`, SHA
`0efd537bed89d16c96932a63156583483f44f9d6621aa434c7d9cda2fab06fda`.
The index binds 188 experiment artifacts, ten source snapshots and six
inherited references. It does not reconstruct inherited linked-object
compilation or complete runtime bytes, or revalidate every parent artifact.
Missing/mutated bound evidence, source/runtime, binary/objects, weights,
inputs, geometry/precision or selection policy reopen affected claims.

Refresh CPU checks with `python3 -B -m unittest -v
spec.hy_nomas_query_head_robustness_test spec.hy_nomas_query_correspondence_test
spec.hy_nomas_query_capture_test`; exact guarded GPU/qualification argv are
indexed externally. Next falsifier: permute K content while holding the
destination RoPE phases fixed, using a GPU gather and the existing BF16
RoPE boundary. After that, capture independent states with labeled nonzero
displacement and compare against same-location/equal-area shams before
attempting transport. The alternative saved runs lack authenticated Q/K;
their preblock/KV bundles do not supply this missing evidence.

## Fixed-phase content null (2026-11-06, qualified diagnostic)

The following design was frozen before inspecting new scores. Freeze frames 1/14/29 and heads 12/15 from
the preceding same-pass screen. For each frame compare identity,
`R_j K_(j-17)` (content roll, destination phases fixed), and
`R_(j-17) K_(j-17)` (content and phase roll). Indices wrap over 144 cells;
the relocated content target for query i is `(i+17) mod 144`. Q is unchanged.
Apply the existing native RoPE and BF16 roundtrip kernels, with GPU gathers,
then the existing F32 score/metric kernels in one chain. The source control
calls the pinned upstream `apply_rotary_emb` on the same BF16 tensors and
authenticated F32 phases. No full generation or runtime-default change.

CAUTION: a wrong roll sign, moved phase table, lost BF16 boundary, or
post-hoc head selection could manufacture a correspondence result. DoD:
focused CPU rejection tests; identity Q/K matching the saved post-RoPE
capture exactly; all nine native/source pairs passing unchanged
`atol=rtol=2e-5` with exact discrete metrics; exact moving-arm roll
equivariance; zero in-chain allocation/readback/wait counters. Keep the
cache-aware 10% floor and 64 MiB swap-growth stop. Rollback is omission of
the new opt-in probe. Compare strict same-position and relocated-content
winners separately; neither is independent-state or semantic ground truth.
Even a favorable null result does not admit actor transport. Next evidence
must use independent states and labeled nonzero displacement.

All nine native/source pairs pass at unchanged `atol=rtol=2e-5`, with exact
discrete metrics. Query, key and score arrays are byte-identical in every
pair; the largest metric difference is 2.682209014892578e-7. Both identity
Q/K outputs also match the saved post-RoPE BF16 capture exactly. Moving-arm
scores are exact column rolls; independent post-drain metric audits pass
with maximum error 4.76837158203125e-7. The diagnostic computes F32 QK
after the upstream BF16 RoPE boundary; it is not full BF16 attention or PV.

Strict winners out of 144 query positions, measured separately for each head:

| Frame | Head | Identity: original content | Moving phases: relocated content | Fixed phases: relocated content | Fixed phases: same position |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 109 | 109 | 14 | 69 |
| 14 | 12 | 118 | 118 | 12 | 76 |
| 29 | 12 | 121 | 121 | 14 | 70 |
| 1 | 15 | 138 | 138 | 2 | 101 |
| 14 | 15 | 139 | 139 | 2 | 106 |
| 29 | 15 | 136 | 136 | 2 | 104 |

The moving-phase control preserves the original content-match counts.
With destination phases fixed, many winners instead remain
at the same position despite receiving different K content. This rejects
phase-invariant index retrieval for these two heads on the tested slices.
It does not establish that the heads lack semantic information, explain
the weak Resume acting, or test cross-frame tracking. The synthetic roll
is a restricted same-frame intervention, not a labeled physical motion.
Do not promote raw post-RoPE affinity into an actor transport controller.

The new opt-in native probe uses the production uncontracted RoPE kernel
and BF16 RNE helper, ten named buffers (8,607,744 bytes) and eight dispatches
in all three arms. Encode counters observe zero named buffer allocations,
readbacks or waits; one terminal wait precedes four diagnostic readbacks.
CPU fixture preparation and post-drain audits remain outside the chain.
The native reader verifies submitted byte integrity and format; the Python
qualifier additionally authenticates the capture ancestry and exact phase
rows. A standalone CLI report is not an ancestry certificate.

Twenty-eight CPU tests, five malformed native admissions and three
format-valid lineage substitutions pass their expected rejection checks.
The real native fixture is admitted without Metal initialization. All 23
positive build/preparation/measurement/qualification stages finish with
the cache-aware 10% floor, 2 GiB process cap and 64 MiB swap-growth stop.
Their minimum sampled headroom is 23,540,858,880 bytes, maximum sampled
process-group footprint is 481,494,048 bytes and observed swap growth is
zero. These samples do not establish a native GPU peak or speedup. No
production generation path, model defaults or full generation changed.
Same-lineage read-only adversary review is ROBUST within the frozen
measurement scope; it is correlated, not independent replication.

Evidence: `.landmark/evidence/2026-11-06-hy-nomas-query-fixed-phase.json`.
Exact guarded GPU commands, fixtures, source snapshots and outputs are
indexed under external `query-fixed-phases-tuyhkhbg/`. Re-run CPU checks with
`python3 -B -m unittest -v spec.hy_nomas_query_fixed_phase_test
spec.hy_nomas_query_head_robustness_test spec.hy_nomas_query_correspondence_test
spec.hy_nomas_query_capture_test`; GPU replay requires fresh output paths.
Refresh on source/runtime, binary/objects, weights, input/phase bytes,
geometry/precision/selection policy or missing/mutated evidence. Linked-object
compilation and complete runtime bytes are not reconstructed.

Next: obtain authenticated independent-state Q/K with labeled nonzero
displacements, then compare pre-RoPE features and position-controlled
candidates against same-location/equal-area shams before any causal rollout.
The current saved alternatives lack that Q/K evidence. Preserve the accepted
Alex/Resume take and original generation path as the visual/runtime controls.

## Bounded upstream activation recording (2026-11-06)

The user requested an upstream video replay with observable actions and
activation recording. This diagnostic is CAUTION: it may consume substantial
SSD space and export GPU tensors at block boundaries. It does not change the
native generation runtime, attention equations, precision, or tolerances.
Rollback is to omit the new recorder/runner and retain the accepted take.

First scope: replay one accepted four-step, 512x288, 120-frame action arm with
all 54 pinned upstream BF16 DiT blocks, the existing MPS attention adapter, and
layerwise weight staging. Record named module outputs plus projection inputs,
pre/post-RoPE Q/K, attention Q/K/V and outputs, both residual boundaries,
prelude/refiner/head outputs, and Euler latents at every step. Internal fused
MPS scratch and implicit QK/softmax matrices are excluded, rather than claimed
as captured. PixelUMM and Qwen internals are outside this first trace scope.
Decode the resulting latent separately and link the decoder provenance.

A provenance audit found that the previous `decode-A1_high/latent-bank.f32le`
does not contain the saved `A1_high.step4.latent` in its last 30 latent
frames. That prior video remains a visual-only comparison; it cannot
establish motion produced by this accepted DiT trajectory. The new decode
must consume the trace final latent directly and bind its exact hash.

The reference is the accepted saved source trajectory, not the text prompt:
each recorded Euler latent must replay the same arm exactly. Actual motion
must be checked in decoded frames; a requested movement is not a known
trajectory. The strongest failure is that capture mutates aliases or changes
execution results. Clone on GPU when observed; stream one bounded block batch
losslessly in its original BF16/F16/F32 representation; compare traced and
untraced block outputs and all four accepted Euler boundaries. A partial trace
must remain explicitly incomplete. Quota, disk reserve, duplicate labels,
truncation, and hash corruption must fail closed.

DoD: focused recorder/runner CPU tests, a guarded block-level snapshot parity
probe, a guarded full single-arm trace with exact Euler replay, trace-index
integrity validation, and an inspectable upstream video with source-linked
motion observations. Preserve atol=rtol=2e-5, the cache-aware 10% host reserve,
64 MiB maximum swap growth, one model-owning process, fresh external-SSD output
paths, and an explicit process/storage cap. Recording adds diagnostic exports;
its elapsed time is not an inference performance benchmark. No native runtime
speedup, full-pipeline parity, or semantic tracking claim follows from capture.

Observed result: the guarded pilot, full trace and separate upstream decode
passed. The trace contains 23,040 tensors in 440 shards (296,265,981,952 bytes),
all 432 block calls, and four Euler latents bit-identical to the accepted
source control. The exact trace final latent now binds to a 120-frame,
24-fps MP4. All 468 streamed decoder tensor hashes passed. Sampled peaks were
3,715,975,760 bytes for capture and 5,060,760,976 bytes for decode; both retain
the cache-aware 10% floor, zero observed swap growth and no final live children.

Six selected frames show Resume's glove rising toward its chest after about
three seconds, with mostly fixed body/head; a substantial Alex head turn is
not visible. The selected-record analysis finds nonzero attention and FFN
residual updates in both coarse actor ROIs, which only excludes zero updates
in those tensor ranges. Requested acting and its causal failure remain open.
Next falsifier: selected-query attention reconstruction with exact compact K
order, valid lengths, masks and prior, then a same-input intervention separating
history/current-video/global-text/action-text influence. Attention mass alone
will not qualify control.

The 72-test CPU suite and eight-file Ruff check pass. Actual-artifact adversary
review is ROBUST within named-boundary capture and exact-latent video lineage;
it does not certify native/full-CUDA parity or semantic motion. See
[the recorder guide](hy-nomas-activation-tracing.md) and
`.landmark/evidence/2026-11-06-hy-nomas-upstream-activation-recording.json`.

## Research precedents and compatibility limits

- [DiTraj](https://arxiv.org/abs/2509.21839) separates foreground/background
  conditioning and modifies foreground 3D-RoPE for trajectory consistency.
  It is a useful DiT precedent, but full-sequence attention/RoPE changes
  are not automatically compatible with Hy-Nomas causal cached K/V.
  Audit cache positions before any adaptation. Its
  [official code](https://github.com/xduzhangjiayu/DiTraj) is marked CC BY-NC;
  this is an idea reference, not an admitted dependency.
- [FreeTraj](https://arxiv.org/abs/2406.16863) combines noise and attention
  guidance. Its U-Net/longer-schedule results do not establish a drop-in
  four-step Hy-Nomas controller.
- [Text2Video-Zero](https://arxiv.org/abs/2303.13439) reuses first-frame K/V
  for appearance. Hy-Nomas already has reference history; more reference
  weight might stabilize identity or merely freeze the starting pose.
  Test that tradeoff instead of copying another full K/V bank.

## Gates, stop rules and continuation

Runtime changes are CAUTION. Keep the original BF16 path and accepted take
as rollback. All GPU probes retain the cache-aware 10% memory floor,
64 MiB swap-growth stop, one model-owning process, reused buffers and no
new in-chain CPU synchronization. New algorithms need their own
source-equation oracle; old receipts do not qualify changed conditioning.

After each key step, run a compact Quadrumvirate: name the predicted failure,
choose one discriminating counterfactual, take the bounded step, then attack
the result with A/A, a negative control and the scoped source comparison.
For renders, also inspect each actor's action, identity, protected parts and
contact at full playback. Pooled block differences locate image change;
they do not prove pose, ownership, natural motion or quality.

After two non-improving interventions at one level, pivot to the next
mechanism; do not repeat prompt amplification. Stop a slice on attribution
failure, numerical regression, memory/swap guard, identity loss or stronger
motion with worse hands/contact. Do not relax tolerances to admit it.

Initial audit DoD was CPU-only (29 tests). The transfer and factorial slices
add twenty-five admission/comparator tests; all 54 focused tests pass on
2026-11-05:

```sh
python3 -B -m unittest spec.hy_nomas_dit_route_factorial_test \
  spec.hy_nomas_refiner_c_factorial_test \
  spec.hy_nomas_refiner_transfer_test \
  spec.hy_nomas_action_conditioning_spec \
  spec.hy_nomas_reference_attention_test spec.video_block_diff_test
crystal tool format --check examples/hy_nomas_refiner_c_factorial_probe.cr \
  examples/support/hy_nomas_refiner_c_factorial_probe.cr \
  examples/hy_nomas_dit_route_factorial_probe.cr \
  examples/support/hy_nomas_dit_route_factorial_probe.cr \
  examples/support/hy_nomas_dit_route_factorial_gpu.cr
git diff --check
git diff --cached --check
```

The release builds, guarded source/native runs and actual-fixture admission
are separately recorded in their evidence directories;
passing CPU tests alone does not establish GPU parity. Rollback is to omit
the new probe; the production runtime and accepted take were not changed.
Refresh the numerical result on source, runtime, weights, input, geometry,
precision or binary changes. Refresh any later visual conclusion on prompt,
reference, noise, schedule, geometry, model or decoder changes.
Next: investigate an actual actor-local pose/part-trajectory conditioning
interface after the bounded render failed to establish the requested acting.
No production spatial-control API or simultaneous-object guarantee is admitted
by this document.
