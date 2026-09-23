# Qwen-Image 2.1 GGUF Frontier

Status: active implementation frontier; prompt-to-PNG path admitted, with
optimization candidates gated by real-prompt parity and paired latency checks

## Goal

Admit Qwen-Image 2.1 weights into the native Metal engine without treating a
file name such as `Q4` as evidence of its actual tensor policy.

## Text-to-image prototype (2026-09-22)

The narrow text-to-image path uses the existing GGUF Metal DiT and FlowMatch
driver, with the official Qwen3-VL text encoder/processor and 64-channel VAE
running through local PyTorch/Diffusers components. Diffusers does not execute
the transformer. This is a hybrid integration, not a fully native image engine.
The initial route uses a deterministic seed, explicit pixel dimensions, and
40-step guidance-free sampling.

- **Admitted after evidence:** a versioned, shape-checked handoff of Qwen3-VL
  prompt embeddings, valid-token mask, target noise, and target shape to the
  native denoiser; a versioned, shape-checked handoff of its final latents to
  the reference VAE decoder; and a PNG produced from an actual prompt with
  the pinned real GGUF. On 2026-09-22, `red cube`, seed 7, 256x256, 40 native
  DiT steps produced an RGBA PNG (SHA-256
  `396ee177a3689ca7d9a035ab4d26ecd58a065215017df56ee64fb1d7084fd841`).
  The official component revision was
  `790c92633540aa0cb11d9abf19eb46d861714758`; the Diffusers source was
  `8b3c707ebd3ec4881f4190cf42931da07eaf3b65`. The finite real-prompt
  two-step Metal regression and all 52 Qwen-Image 2.1 Crystal examples passed;
  the Python bridge tests passed 14 cases with one expected skip. A second
  40-step run with the default code path reproduced both latent and PNG hashes.
- **Guard:** Qwen-Image 2.1 consumes *unpatched* 64-channel VAE latents. Its
  spatial sequence is `height/16 * width/16`, and each DiT token is one VAE
  spatial location. The VLM image-placeholder mask still expands one slot to
  four image tokens; that slot expansion is not 2x2 latent packing. Reject a
  bridge that uses the older Qwen-Image 2x2 pack/unpack or misaligns either
  mask with the target sequence.
- **Deferred in the initial prompt-to-PNG transition:** image-conditioned
  editing, classifier-free guidance, prompt rewriting, image-quality ranking,
  custom text-encoder/VAE kernels, and DiT speedups. Optimization is now a
  separate, measured transition; the other features remain deferred.
- **Falsifiers:** a mismatched binary length, non-finite float, wrong channel
  order, incorrect mask/placeholder count, wrong VAE normalization, or a PNG
  produced without executing the native GGUF denoiser. A smoke PNG is not an
  image-quality claim; visual inspection and a reference comparison remain
  separate checks.
- **Known guard:** the fused BF16 text projection originally yielded all-NaN
  output for real Qwen3-VL embeddings on M2 Max. A saturated-GELU guard now
  keeps its real projection finite, but `QWEN_IMAGE21_FUSED_TEXT=1` remains
  opt-in because full-run latency has not improved. The default route still
  uses separate Metal projections with host GELU. Fused timestep and resident
  block paths remain enabled.

The observed PNG supports basic prompt adherence only; quality, reference
parity, varied prompts and larger resolutions remain unverified. The evidence
decays if model revisions, GGUF packing, Diffusers behavior, Metal kernels or
the bridge schema change. Rollback is to remove the bridge and reference-side
scripts while retaining the earlier transformer/scheduler implementation.

With an isolated Python environment exposing Torch, Transformers 5.x,
Diffusers `QwenImage21Pipeline` and Pillow, and local official model components
under `$MODEL_DIR`, reproduce the same three boundaries as follows. `$GGUF`
must be the pinned file below; compile the native runner with the project's
Metal bridge before use.

```sh
MODEL_DIR=/path/to/Qwen-Image-2.1-components
GGUF=/path/to/Qwen-Image-2.1-Q4.gguf
crystal build --link-flags='-fuse-ld=/usr/bin/ld build/bridge.o -framework Metal -framework Foundation -lc++' \
  scripts/qwen_image21_generate_latents.cr -o /private/tmp/qwen-image21-generate-latents
python scripts/qwen_image21_prepare_conditioning.py \
  --model-dir "$MODEL_DIR" --output-dir /private/tmp/qwen21-condition \
  --prompt 'red cube' --width 256 --height 256 --seed 7
/private/tmp/qwen-image21-generate-latents "$GGUF" \
  /private/tmp/qwen21-condition/qwen_image21_conditioning.json \
  /private/tmp/qwen21-latents 40
python scripts/qwen_image21_vae_decode.py \
  --manifest /private/tmp/qwen21-latents/qwen_image21_latents.json \
  --model-dir "$MODEL_DIR" --output /private/tmp/qwen21-red-cube.png \
  --device cpu --dtype float32
```

These output directories must be new or empty. The prepared bundle records
the official model revision and payload SHA-256; the native reader checks the
schema, payload checksum and presence of a revision before running the DiT.
Reference components are loaded locally, without a network fallback during
generation.

## Optimization evidence (2026-09-23)

The pinned real `red cube`, 256x256, seed-7 path took 19.84 s for CPU prompt
conditioning, 57.99 s for 40 native Metal DiT steps, and 9.65 s for CPU FP32
VAE decode in one sequential cold run. These are stage observations, not a
repeatable end-to-end speed claim. A later `--device auto` conditioning run on
the same Mac selected CPU and reproduced the baseline binary payload SHA-256
exactly. Automatic MPS selection was removed because the default Qwen3-VL MPS
grouped-query attention path aborted; explicit MPS with eager attention is
experimental. One such run produced finite embeddings
and a PNG, but its final latents differed from CPU conditioning by relative
L2 `0.0207`, and quality across prompts is untested. CPU remains the safe Mac
default; CUDA selection is unchanged.

The fused Metal text chain's first BF16 projection was finite; its GELU
produced NaNs on large finite activations. Clamping only the saturated GELU
tails (`x > 10` to `x`, `x < -10` to zero) preserved finite real-model output:
the fused/separate text projections differed by at most `0.0002442`. The
40-step fused run had finite latents, relative latent L2 `0.000348` against
the default route, and a decoded PNG differing in 3216 of 65536 pixels by at
most two channel levels. Its 58.72 s denoise observation did not establish a
speedup against the approximately 58 s default. Keep the fused route opt-in.
The complete post-fix Qwen-Image Metal suite passed 54 examples against the
pinned real GGUF and conditioning bundle; the Python bridge suite passed 16
tests with one expected skip.

A run-local prompt-projection cache was prototyped and rejected for now. It
avoided the two text matrix multiplications on later steps, and all six
40-step runs preserved the baseline latent SHA-256. However, two on-first
pairs favored the cache slightly while a reversed off-first pair favored
uncached execution by 2.39 s. That falsified a robust speedup claim under
current host noise; the added mutable state and concurrency restriction did
not earn a place in the runtime. The evidence can be revisited if text
projection becomes a measured bottleneck at a different prompt scale.

A real 256x256 step profile attributes approximately 63% of diagnostic GPU
phase time to the Q8_0 Q, K, and attention-output projections and 10% to
attention itself. The diagnostic path splits a normal one-command-buffer step
into 418 buffers, so those shares rank optimization targets but do not predict
whole-step speedup. A proposed Q8_0 input-reuse tile passed exact output
parity but failed the normal-path latency gate below. Register-local reuse
across two output channels subsequently passed the bounded whole-forward
gate below on M2 Max, without staging input in threadgroup memory.

## Admitted now

- Parse GGUF `qwen_image` metadata without mapping the tensor payload.
- Support GGML BF16 tensor type 30 in byte sizing and CPU dequantization.
- Inventory actual per-tensor quantization types and storage sizes.
- Recover and validate ComfyUI `comfy.gguf.orig_shape.<tensor>` metadata.
- Validate the exact 32-block Qwen-Image 2.1 DiT tensor inventory.
- Report whether every tensor type is readable by the current Cogni-ML GGUF
  dequantization layer.
- Execute a parameterized one-block CPU reference with the upstream 2.1
  equations: affine-less LayerNorm, per-head Q/K RMSNorm, three-axis RoPE,
  block-causal attention, shared tanh modulation gates, and fused SwiGLU.
- Multiply BF16 GGUF matrices directly in the CPU reference path.
- Check the complete tiny block against an independent PyTorch oracle.
- Reject truncated GGUFs before exposing any mmap-backed tensor slice.
- Recover mathematical projection dimensions from Comfy `orig_shape` metadata
  while preserving ordinary GGUF `[in, out]` dimension semantics.
- Load all top-level and 32 transformer-block weights as zero-copy mmap slices,
  with exact per-projection shape validation and explicit mmap lifetime.
- Reuse the Qwen 3.5 Q8/Q5/Q6 Metal projection kernels inside the exact block
  reference. On two real tokens through block 0, the CPU/Metal boundary produced
  `max_abs=1.5258789e-5` and cosine `0.999999999999972` across 8192 outputs.
- Execute the exact batch-one outer transformer contract: zero-centered text
  RMSNorm and projection, VLM image-slot expansion, condition/target latent
  substitution, shape-derived image block ids, centered three-axis positions,
  padding validity, sinusoidal timestep embedding, causal `t=0` prefix
  modulation, all transformer blocks, adaptive output norm, and output head.
- Check that complete outer forward against an independent tiny PyTorch oracle,
  including the appended target placeholder slots used by the official
  pipeline. Adjacent image slots are explicitly kept as separate attention
  blocks, and prefix output is timestep-independent under causal conditioning.
- Execute the complete outer transformer with all 32 real mixed-quant blocks
  for a minimum valid `2x2` target. The measured hybrid run used 192 Metal block
  projections and produced finite output. Before the top-level BF16 Metal route,
  this minimum-size check completed in approximately `3.61` seconds.
- Keep a complete DiT block sequence Metal-resident behind one outer-transformer
  backend call. Affine-less LayerNorm and modulation, per-head Q/K RMSNorm,
  three-axis RoPE, segmented block-causal attention, SwiGLU, residual updates,
  and all six mixed-quant projections per block are encoded into one command
  buffer with shared scratch and ping-pong hidden buffers.
- Execute all 32 real mixed-quant blocks with one command buffer, zero
  intermediate readbacks, 192 projection dispatches, and one final hidden-state
  readback. Against the prior hybrid reference on the minimum valid `2x2`
  target, the outer output produced `max_abs=2.771616e-6` and cosine
  `0.9999999999998054`. A real single-block check produced
  `max_abs=0.00012588501` and cosine `0.999999999999791`.
- Cache the causal prefix K/V tensors for every DiT layer in persistent Metal
  buffers after the first transformer evaluation. Later FlowMatch evaluations
  recompute only the contiguous target suffix while assembling full attention
  K/V on-device and retaining one command buffer, zero intermediate readbacks,
  and one final readback. A real 32-layer check with an unchanged text prefix,
  changed target latents, and changed timestep recomputed four of five tokens
  and matched a complete hybrid reference with `max_abs=5.401671e-6` and cosine
  `0.9999999999988506`.
- Invalidate the prefix cache fail-closed when its token boundary, prefix hidden
  state, prefix modulation, prefix positions, prefix image ids, prefix key
  validity, layer objects, or block configuration changes. A tiny GPU check
  verifies both the cache-hit path and rebuild after a changed prefix value.
- Execute all eight top-level BF16 matrix shapes with a batch-capable Metal
  kernel; the empty-text minimum target evaluates six of them. The kernel reuses
  the registered whole-model mmap buffer plus tensor offset rather than copying
  weights into a second allocation, while synthetic heap-backed weights retain
  the existing per-weight upload fallback. A full real outer-forward check
  with one text token exercises all eight shapes; against a selective reference
  that leaves only BF16 on CPU it produced `max_abs=3.8146973e-6` and cosine
  `0.9999999999996982`.
- The prototype includes a fused text-projection/GELU command buffer and a
  separate fused timestep/modulation/output-scale command buffer. The real
  Qwen3-VL fused text chain is finite after the saturated-GELU fix, but remains
  opt-in without a measured whole-run gain; the separate BF16 Metal projections
  and host GELU remain the default text path. The fused timestep path remains
  in use. This is not a throughput claim.
- Keep the resident block result on-device through affine-less LayerNorm,
  per-token output scaling, and the BF16 output projection. The final head is
  encoded as the 193rd projection dispatch in the existing stack command
  buffer, so only the projected transformer output is read back. On a prefix
  cache hit, the persistent prefix output and resident target output are joined
  on-device before the final head; the target hidden state is not read back or
  re-uploaded. The real no-text 32-layer check produced
  `max_abs=2.1457672e-6` and cosine `0.9999999999997731`; the real changed-target
  prefix-cache check produced `max_abs=1.013279e-5` and cosine
  `0.9999999999977098`.
- For full passes without a reusable causal prefix, project `img_in` and the
  available timestep rows directly into the resident stack command buffer. GPU kernels
  assemble the mixed text/image joint sequence and select per-token modulation
  and output-scale rows; no image or timestep projection is read back to the
  host. Text normalization and its fused projection still return a host array.
  The real mixed condition-image/text/target-image check produced
  `max_abs=0.0039245486` against the hybrid reference and zero difference
  against the former stack input route. A separate causal two-row check
  produced `max_abs=2.3841858e-6`. The Qwen-Image hybrid reference now uses
  the same buffer-level K-quant GEMV route as the resident stack at all batch
  sizes; the general Qwen 3.5 Q5/Q6 batch GEMM route diverged on the mixed
  nine-token probe and is not used as this reference.
- Build and reuse causal prefix K/V with Metal-resident image projection,
  timestep projections, joint-sequence assembly, and row selection. A raw-input
  certificate compares encoder inputs and their projected text, condition-image
  latents, prefix source/layout metadata, the `t=0` timestep row, top-level
  projection tensor identities, text normalization, layer identities, and block
  configuration, assuming loaded tensor payloads remain immutable. Changed
  target latents and the active timestep row may hit;
  changed encoder or condition-image inputs rebuild. Prefix queries must not
  attend suffix keys. The real 32-layer text-prefix hit recomputed four of five
  tokens with `max_abs=1.013279e-5` against a full reference; a mixed
  text/condition-image prefix also hit and rebuilt on changed condition data.
- On a resident prefix hit with an ordered image-only target suffix, `img_in`
  uploads and projects only target rows directly into the active hidden buffer.
  Target modulation is selected for active rows; final output scales still
  cover the complete prefix-plus-target sequence. A nonconforming suffix takes
  the uncached full resident route. The real condition-image check projected
  eight image rows on build/rebuild and four on hit, while retaining output
  parity with the full reference at `max_abs < 0.05`. An eight-pair warm profile
  on M2 Max measured median full-forward wall time of `128.4 ms` on rebuild and
  `85.1 ms` on hit; complete GPU command time was `123.0 ms` and `80.7 ms`.
  The input-command encoding medians were `0.077 ms` and `0.052 ms`. This is a
  small paired hit-versus-rebuild observation, not an old-versus-new A/B or a
  component-level GPU attribution; DiT also processes fewer active tokens on
  a hit. A full resident pass over nine tokens and a cached pass over four
  cross the default eight-row GGUF GEMM/GEMV threshold: their outputs differed
  by `0.0021353364` maximum. Two independent full passes were identical, all
  32 prefix K/V tensors matched, and the difference survived temporary
  restoration of the former full-input preparation. With
  `QWEN35_GEMM_BATCH_THRESHOLD=16`, both paths used GEMV and this difference
  disappeared. Reproduce timing with `scripts/qwen_image21_prefix_profile.cr`.
- On the same host and minimum `2x2` target, the unfused outer-forward observation
  changed from approximately `3.61` to `0.92` seconds, and two FlowMatch steps
  changed from approximately `6.76` to `1.31` seconds. These are implementation
  smoke timings, not representative-token throughput claims.
- Profile the real 32-layer GGUF with 256 condition-image and 256 target-image
  tokens plus one text token (513 joint tokens), using synthetic embeddings and
  latents. On M2 Max, three warm paired samples measured median complete
  forward wall time of `5398 ms` for a cache rebuild and `2797 ms` for a hit;
  their GPU command times were `5356 ms` and `2757 ms`. A diagnostic mode
  splits the normally single DiT command buffer at phase boundaries, and
  matches the normal outputs exactly in this probe. Across two diagnostic
  samples, the combined Q/K/V plus output projections accounted for roughly
  73% of summed phase GPU time, attention roughly 12%, and cache-copy below
  0.2%. A further single diagnostic sample split Q, K, and V: on rebuild,
  Q=`1101 ms`, K=`1113 ms`, V=`78 ms`, output=`1107 ms`, and attention=`541 ms`;
  on hit, Q=`543 ms`, K=`543 ms`, V=`38 ms`, output=`541 ms`, and
  attention=`286 ms`. This is phase attribution under extra command-buffer
  submissions, not an end-to-end speedup or a production-resolution result.
  Q/K/output use Q8_0 GGUF tensors; at the time of this baseline, the buffer
  route selected Q8_0 GEMV for every batch size. The F32 prefix K/V allocation at this shape
  is theoretically `269484032` bytes; that is not a measured total Metal
  working set. The profile reports host buffer preparation and final readback
  separately, but does not isolate GPU transfer or launch cost from command
  waiting. Reproduce the old-route baseline with
  `QWEN_IMAGE21_Q8_BATCH=0 crystal run scripts/qwen_image21_prefix_profile.cr -- MODEL.gguf 3 16 16 2`.
- Use a Qwen-Image-local Q8_0 kernel that reuses each quantized weight block
  across eight adjacent batch rows for Q, K, and attention output projections
  at batch size `>=16`. Other weights and shorter batches retain the existing
  Qwen 3.5 route. `QWEN_IMAGE21_Q8_BATCH=0` restores the prior route without a
  weight-format change. A synthetic Q8_0 matrix test covers batch sizes
  `1,7,8,9,16,17,32,64,256`; a real mixed-quant block and three real-weight
  32-layer A/B shapes preserve output parity, with `max_abs=0` in the A/B
  profiles. After one warm pair, three alternating-order pairs on M2 Max gave
  median candidate/baseline wall ratios for rebuild/hit of `0.738/0.789` at
  16 target + 16 condition-image tokens, `0.631/0.632` at 64+64, and
  `0.693/0.671` at 256+256. These are scoped synthetic-latent, real-weight
  measurements, not production-resolution or image-quality evidence. The
  host's absolute timing drifted substantially between runs, so the paired
  ratios are more informative than cross-run millisecond comparisons. Set
  `QWEN_IMAGE21_Q8_BATCH_AB=1 crystal run scripts/qwen_image21_prefix_profile.cr -- MODEL.gguf 3 16 16 0`
  for the alternating route/parity probe.
- Reuse each Q8_0 input value across two output channels within a SIMDgroup,
  preserving the existing per-channel accumulation order and avoiding
  threadgroup storage and barriers. The automatic route is limited to exact
  `Apple M2 Max` and batch `>=256`; `QWEN_IMAGE21_Q8_REGISTER_REUSE=0` restores
  the established batched Q8_0 kernel, while `=1` forces the reuse kernel for
  eligible batched Q8_0 calls as an experiment. The outer
  `QWEN_IMAGE21_Q8_BATCH=0` rollback to the older Qwen 3.5 route is unchanged.
  Synthetic GPU checks compare the two Q8 kernels exactly across batch,
  output-channel, and odd-Q8-block tails with non-unit scales and a NaN mask.
  The bounded real-weight 513-token evidence and its scope are recorded below.
- With that Q8_0 batch route enabled, profile the same real 32-layer GGUF at
  larger synthetic condition/target image grids, each with one text token.
  After a warm pair, the `24x24 + 24x24` (1153-token) run used two normal
  rebuild/hit pairs and one diagnostic split pair; the `32x32 + 32x32`
  (2049-token) run used one normal pair and one diagnostic split pair. The
  diagnostic path matched normal outputs exactly on both rebuild and hit
  (`max_abs=0`). Its 418 command buffers make its phase sums unsuitable as
  direct estimates of normal one-command-buffer latency. Within each
  diagnostic pass, attention and the three Q8_0 Q/K/output projections took:

  | Joint tokens | Attention, rebuild/hit | Q/K/output, rebuild/hit |
  | ---: | ---: | ---: |
  | 1153 | 29.0% / 31.3% | 54.9% / 51.1% |
  | 2049 | 40.4% / 42.3% | 47.0% / 43.9% |

  Attention is the largest individual phase at 2049 tokens and approaches
  the three Q8_0 projections combined, but this single diagnostic sample per
  shape does not establish a stable crossover or a new kernel's speedup. The
  2049-token normal wall observations were `55.43 s` rebuild and `28.09 s`
  hit (`n=1`), with noisy host load; do not compare their absolute values to
  prior runs as an A/B. The theoretical F32 prefix K/V allocation at this
  shape is `1074790400` bytes, not a measured total GPU working set. Reproduce
  with `scripts/qwen_image21_prefix_profile.cr` using `1 32 32 1` for the
  2049-token probe.
- Reproduce the model's configured deterministic FlowMatch Euler schedule:
  linear input sigmas, exponential resolution shift over the exact
  `256..8192` sequence-length range, terminal stretching to `0.02`, and Euler
  updates. A two-step model-backed loop executed 396 Metal projections across
  two complete 32-block evaluations and produced finite changed latents.

## Pinned bootstrap artifact

- Repository: `realrebelai/Qwen-Image-2.1_GGUFs`
- Revision: `8d393b750593a72ee040fe7d40611f479ccee679`
- File: `Qwen-Image-2.1-Q4.gguf`
- Bytes: `5959127264`
- SHA-256: `51998ad7c068ce7d68e233237537900ffe874ab4d5c72e20758f5f18ceb15b8a`
- Actual tensor policy: `BF16:8,F32:65,Q8_0:96,Q6_K:64,Q5_K:32`

The repository README describes this policy as its Qwen-specific mixed Q4
release even though its recommended long `Q4_K_M-HQv3` label does not match the
current short filename. The loader trusts the inspected tensor directory, not
either label.

## Rejected attention probe

The one-SIMD-group-per-query/head prototype removed per-key threadgroup
barriers while retaining online softmax, the existing block-causal/image mask,
and O(1) per-query scratch. A real-weight, synthetic-latent full-forward A/B
warmed both routes and alternated their order over three measured pairs at each
shape. Median candidate/legacy wall ratios were:

| Joint tokens | Rebuild | Prefix hit |
| ---: | ---: | ---: |
| 513 | 0.984 | 1.031 |
| 1153 | 1.025 | 1.073 |
| 2049 | 0.983 | 1.096 |

Ratios below one favor the candidate. The 2049-token rebuild gain was only
1.7%, while the repeated cache-hit path regressed by 9.6%; both paths
regressed at 1153 tokens. Candidate-versus-legacy outputs had maximum
absolute differences up to 0.133 across these probes, exceeding the existing
0.05 resident-stack/cache parity limit despite cosine similarity above
0.99999. That numerical drift and the hit regression falsified promotion. The
candidate kernel, dispatch switch, and route-specific A/B harness were
discarded; the legacy kernel remains active. Host load varied, so these are
bounded paired observations, not a universal kernel ranking or an image-quality
result. A mixed text/image, invalid-key, prefix-build/hit/rebuild GPU
regression test remains as a guard for the next attention candidate.

## Rejected Q8 input-reuse tile

An opt-in 8-row by 8-Q8-block Metal threadgroup tile staged each input slice
once for four output channels, keeping the original quantized arithmetic and
output ownership. Synthetic parity covered 16- and 17-block K dimensions,
non-unit half scales, inactive output-channel simdgroups, and batch tails.
Two real-prompt two-step off/on pairs in opposite orders produced identical
latent payload hashes; the tiled route took 4.40/4.44 s versus 3.37/3.30 s
for the existing Q8 batch route, although those process-level totals included
pipeline setup. A warmed, alternating-order real-weight 513-token A/B then
confirmed exact full-forward parity but median tiled/default wall ratios of
1.375 on prefix rebuild and 1.371 on hit (`n=2` measured pairs, one warm pair).
The corresponding median one-command-buffer GPU times were 4036/2923 ms on
rebuild and 1999/1448 ms on hit. The candidate therefore regressed both
paths by roughly 37%; shared-memory loads and barriers are the leading
explanation, but their individual costs were not isolated. The prototype and
its A/B switch were discarded.
This rejection is scoped to the tested tile and host, not to every Q8 tiling
strategy or GPU. Reopen only with a distinct mechanism and a paired normal
full-forward falsifier, not an isolated kernel timing.

## Admitted Q8 register-local reuse on M2 Max

The register-reuse kernel computes two output channels per SIMDgroup over
eight adjacent batch rows. It reads each input value once for the two
channels, retaining the existing Q8_0 weight layout, block and lane reduction
order, and independent output ownership. It uses no threadgroup memory or
barriers. Synthetic Metal checks matched the established Q8 batch kernel
exactly across batch, output, and K-block tails, non-unit half scales, and
NaN propagation. The full model-backed Qwen-Image spec suite passed 56
examples with zero failures or pending cases after the automatic policy change.

The precommitted gate required exact output parity plus one warm pair and at
least three measured alternating-order, normal one-command-buffer A/B pairs
at 513 joint tokens; median paired wall ratios below 0.95 on both rebuild
and prefix hit, non-regressing GPU-command ratios, and no order reversal.
A forced-kernel screen against the established Q8 batch route gave median
reuse/baseline wall ratios of 0.896 on rebuild and 0.889 on hit, with GPU
command ratios of 0.895 and 0.886. Rebuild and prefix-hit full-forward
outputs matched exactly in every pair; the gain survived both execution
orders. These are full 32-layer forward passes using the pinned real GGUF
with synthetic input embeddings and latents, not an isolated projection or
phase-only speedup. The automatic policy is narrower than the forced-kernel
screen, so its dispatch was tested directly as well.

The final harness asserted the device name `Apple M2 Max`, compared the
automatic override-unset path against explicit rollback `=0`, and warmed both
routes before three alternating-order measured pairs. The normal 513-token
forward used one command buffer per rebuild and prefix hit. Median paired
auto/baseline wall ratios were 0.897 on rebuild and 0.890 on hit; GPU-command
ratios were 0.895 and 0.887. Every full-forward rebuild and hit output matched
exactly (`max_abs=0`), and neither execution order reversed the gain. The Q8
calls at this shape use 513 rows on rebuild and 256 on the hit target suffix,
both meeting the automatic batch threshold. Reproduce with
`QWEN_IMAGE21_Q8_REGISTER_AB=1 crystal run scripts/qwen_image21_prefix_profile.cr -- MODEL.gguf 3 16 16 0`
with the Metal bridge linked and the pinned GGUF; the harness
also retains `QWEN_IMAGE21_Q8_REGISTER_FORCE_AB=1` for the wider forced-on
experiment.

On the pinned real `red cube`, seed-7, 256x256, 40-step path, an automatic
policy run with the override unset took 48.113 s for denoising. Its latent
payload SHA-256 was identical to the rollback route's
`d835709261d2d36870ebd564cfb55d3d4db8a1173f1f8e8d511684c8eb7fa844`.
Two prior forced-on runs took 48.670 and 49.344 s versus one rollback run at
57.782 s. These sequential process observations support this one prompt and
host, but are not a paired end-to-end latency distribution; they do not
establish image quality or gains on another device, quantization policy, or
resolution. The unchanged latent bytes preserve the previous reference-VAE
decode input, but no new PNG was decoded in this optimization run.

## Not admitted by this slice

- A production-scale, end-to-end resident Metal pipeline or native text encoder
  and VAE. A decoded image exists through the hybrid reference components, but
  is not a fully native decoded-image path. On the no-prefix route, layout
  metadata,
  sinusoidal timestep input, and text normalization/projection are still built
  on the CPU, and the final output is read back. The causal-prefix hit route
  skips condition-image projection for an ordered image-only target suffix,
  but still performs the timestep projections and a full-token output head.
- Production-resolution throughput, image-quality, or end-to-end latency from
  the synthetic-input profiles. The exact attention kernel remains
  correctness-first and quadratic in token count; its share did grow across
  the measured 1153- and 2049-token shapes, but behavior on other input
  distributions and a faster replacement remain unproven. The tested
  one-SIMD-group replacement is explicitly rejected at the measured shapes.
- A speedup from either Q8_0 batch projection or register reuse on other GPU
  families, quantization variants, or untested sequence lengths. Automatic
  register reuse is limited to the measured M2 Max batch corridor, with an
  explicit rollback switch.
- A compressed or bounded-memory prefix cache. The admitted implementation
  stores per-layer prefix K/V as F32 Metal buffers and therefore trades memory
  for repeated-step projection savings.
- A claim that a readable GGUF has acceptable image quality.
- A custom weight format derived from the resident-KV adaptive QBit codec.
- Trusting repository or file labels (`Q4`, `dynamic`, `HQ`) over tensor data.

## Guard and next transition

The raw-input certificate and GPU cache build/hit are admitted only for the
supported BF16 top-level route and a closed causal prefix. The older
host-projected route remains for unsupported weights. The local batched Q8_0
route keeps the prior GEMV path as an environment-controlled rollback; its
additional GPU allocation is zero because it reuses existing inputs, outputs,
and mmap-backed weights. Larger-token profiling still identifies attention as
a candidate, but removing per-key barriers by assigning one SIMD group to each
query/head did not pass the full-forward latency gate. At the tested real
256x256 prompt, Q8_0 Q/K/output projections dominate the diagnostic phase
profile; the first input-reuse tile failed the full-forward latency gate, while
register-local reuse passed the bounded 513-token gate. Keep the original
batched kernel as the immediate rollback. Before widening beyond exact M2
Max/batch `>=256`, repeat a full-forward paired A/B with exact output parity
on each new device, shape, and model packing. At larger token counts,
attention's measured share grows, so reprofile before assuming projections
remain the dominant target. Text encoding, sampling, and VAE decode remain
separate frontiers.

The model-backed checks are:

```bash
QWEN_IMAGE21_GGUF=/path/to/Qwen-Image-2.1-Q4.gguf \
  SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  crystal spec spec/qwen_image21_flow_match_spec.cr \
    spec/qwen_image21_transformer_spec.cr \
    spec/qwen_image21_weights_spec.cr spec/qwen_image21_metal_spec.cr \
    spec/qwen_image21_resident_metal_spec.cr \
  --link-flags="$(pwd)/build/bridge.o -framework Metal -framework Foundation -lc++"
```

## Falsifiers

- A real candidate contains an unimplemented tensor type.
- A `comfy.gguf.orig_shape.*` entry changes the tensor element count.
- Any required top-level or per-block tensor is absent, duplicated, or appears
  under an incompatible architecture.
- A full candidate is shorter than the last tensor extent declared in its
  directory.
- Any real block projection lacks a strict Metal route or exceeds the declared
  CPU/Metal tolerance.
- A resident stack uses more than one command buffer or performs an
  intermediate host readback for one outer-transformer evaluation.
- The resident final head reads back a hidden state, uploads a normalized
  intermediate, or issues a second command buffer before the output projection.
- Prefix caching changes a prefix hidden state, key, or value when only the
  target latents and FlowMatch timestep change.
- A changed prefix input or block configuration reuses the prior prefix cache
  instead of rebuilding it.
- A raw-input certificate reuses K/V after an encoder, condition-image,
  `t=0` row, prefix layout, or weight-identity change, or a prefix query can
  attend a target key while the target is omitted from cached prefix work.
- A closed resident cache hit reprojects condition-image rows, or the target
  suffix is not an ordered image-only sequence and still enters the cache path.
- Diagnostic phase splitting changes the forward output, or the Q8_0 batch
  route fails parity on Q/K/output shapes or loses its paired full-forward
  latency advantage at the tested token counts.
- A proposed attention replacement changes same-image bidirectional,
  cross-block causal, or invalid-key semantics; fails full-forward parity; or
  only improves its isolated phase while regressing paired one-command-buffer
  rebuild/hit latency or adding quadratic-size scratch.
- A later image-quality corpus shows that the selected mixed quantization
  policy is worse than its declared baseline.
