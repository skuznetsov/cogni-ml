# Hy-Nomas object control frontier

Status (2026-11-05): source audit completed; controller changes are proposed.
The accepted five-second Alex/Resume ROI take remains the control. No model,
GPU probe, paid API call or new render was run for this audit. Numerical
parity is scoped to the frozen fixtures in
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

The first implementation slice is a **refiner-only counterfactual**:

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
  Reuse `examples/hy_nomas_text_conditioning_probe.cr`,
  `examples/support/hy_nomas_text_probe.cr` and
  `spec/support/hy_nomas_text_conditioning_reference.py` where practical;
  add a refiner-only runner before executing this slice.
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
the direct Action1/Action2 text routes. Keep text states, time geometry,
noise and T0 fixed. M00 means **no direct action-text route**, not
action-free conditioning. Start with a same-latent DiT forward, not four
decoded videos. Measure velocity deltas inside each actor ROI, outside both
and in any contact region, including the interaction
`v11 - v10 - v01 + v00`; non-additivity is not itself an error.
Swapping actor ROIs is only a stress test: disagreement
could reflect a strong named-subject prior rather than a broken mask.

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

Audit DoD (CPU only, executed 2026-11-05; 29 tests pass):

```sh
python3 -B -m unittest spec.hy_nomas_action_conditioning_spec \
  spec.hy_nomas_reference_attention_test spec.video_block_diff_test
git diff --check
git diff --cached --check
```

The tests guard existing metadata/mask/diagnostic behavior; they do not
validate a proposed controller. Refresh this audit on upstream/native source,
conditioning layout or precision changes; refresh any later visual conclusion
on prompt, reference, noise, schedule, geometry, model or decoder changes.
Next: implement and qualify the refiner-only paired runner, then use its
result to select the first controller. No production spatial-control API or
simultaneous-object guarantee is admitted by this document.
