# Hy-Nomas performance introspection, 2026-10-26

This is a source and evidence audit at native parent `5b7803f8`, not a new GPU
benchmark or a generation-quality result. The three fresh-context Quad audits
are correlated review. No model, Studio, build, profiler or generation was
launched for this audit; production code and numerical gates are unchanged.

## Decision

Pause K16 loader tuning as the primary speed frontier. Keep K32 and the current
MPS control. The first decision probe is an inventory of the actual PixelUMM
checkpoint, target segment geometry and aggregate residency, before starting a
decoder port or timing it. The first low-cost implementation probe is
elimination of unused DiT text rows. Use actual image-stream activations before
another large-M Q4 kernel experiment. Qualify representation quality separately
from kernel parity; current uniform Q4 has not qualified as a quality-preserving
replacement for BF16.

The intended value remains usable video at roughly 60-120 seconds of wall time
per second of output, with acceptable appearance and motion. Neither a fast
projection nor a small weight file establishes that value. Required counter
metrics are decoder time, complete pipeline latency, peak resident memory,
swap growth, BF16-relative model error, and decoded visual/temporal quality.

## What the existing work establishes

- All 54 DiT blocks execute on GPU from a saved Qwen producer boundary, with
  zero tracked encode allocations, zero preterminal activation reads and one
  terminal wait. Fourteen captured outputs match the corresponding upstream
  BF16 or identically decoded Q4 control exactly. Scratch reuse and serial
  weight staging are already implemented; they are not new opportunities.
- BF16-weight/F32-math MPS saved-input forwards took 6.415-6.420 seconds;
  staged Q4 took 6.545-6.563 seconds. The runs were not interleaved, so no
  relative speed or staging-cost conclusion follows. Named storage falls from 22,616,779,332 to
  10,895,796,804 bytes. Compression has a demonstrated residency benefit even
  without a demonstrated arithmetic speedup.
- Q4 versus original BF16 reaches final-latent relative L2 1.184475 and cosine
  0.638643. This does not prove visible artifacts. It rejects a claim that this
  quantization error is negligible. The unchanged finite `atol=rtol=2e-5`
  implementation gate compares identical decoded operands, not lossy Q4 to
  original BF16.
- The K16/K32 probe uses block-0 text FFN-down weights and 23 saved input rows
  repeated to M1280. K16 loses all 12 paired rounds per representation: median
  duration ratios 1.155866 dense and 1.229616 Q4. B128/SG8 weight-panel reuse
  was already tested and had no material gain over B64. Direct Q4 still has
  no demonstrated advantage over the relevant MPS controls.
- A prior synthetic one-block profile reports 137.626 ms wall, 1.853 ms CPU
  encode and 136.704 ms GPU-root union. Attention-only interior roots sum to
  57.623 ms, but shared boundary roots and stalls prevent exact attribution.
  This is neither a phase profile of the saved-input full54 forward nor a
  video latency measurement.

Evidence: [full54 forward and quality](hy-nomas-q4-dit-frontier.md),
[K16/K32 and earlier tile controls](hy-nomas-q4-reduction-frontier.md),
[whole-block profile](hy-nomas-native-frontier.md#previous-slice-representative-one-block-profiling).
Frozen artifacts remain under `/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025`,
`hy-nomas-q4-panel-profile-20261026` and `hy-nomas-full-block-profile-20261024`.

## The overlooked decoder can be comparable to the denoiser

External sources were inspected at upstream main
`37372925fd4bbb7977a89ce3248584926fc9e953`. The native port remains pinned to
`911cea6458a89c2663513ff7099561ec190dd491`; this audit does not update that pin.
The current-main PixelUMM defaults are a Qwen3-like generation expert with
36 layers, hidden width 4096, FFN width 12288, 32 query heads and 8 KV heads
of width 128. A latent `[1,32,6,22,40]` produces a `6*44*80 = 21,120` token
grid. Its intended pixel shape is 24 frames at 704x1280; no frames were decoded
in the native full54 qualification.

Counting multiply-add as two FLOPs, the decoder's layer projections alone cost

```text
Ndec = 6 * (2*22) * (2*40) = 21,120
Pdec/layer = 2*4096^2 + 2*4096*1024 + 3*4096*12288 = 192,937,984
Fdec/stack = 2 * Ndec * 36 * Pdec/layer = 293.389 TFLOP
```

For comparison, eight passes through the current DiT geometry require about
285.272 TFLOP in their analogous token-wise projections, including Q/K/V and
attention-output projections as well as FFN linears on both sides. These
counts exclude pairwise QK/PV attention, norms, activations, decoder input/head
projections, blending and I/O. They do not predict elapsed time: operand
precision, GQA, locality, kernel efficiency and window overlap differ. They
show why treating the decoder as cheap postprocessing is unsafe. Decoder layer matrices alone
contain 13,891,534,848 BF16 bytes, excluding the input/head and other tensors.

The current upstream preset uses one decoder denoising step, temporal windows
of 10 latents with overlap 5, no spatial crop tiling and a local-attention tile
of 128 pixels. Local attention limits attention work; it does not remove the
dense FFN/projection cost. More decoder steps or overlapping windows increase
work. Do not silently substitute a smaller architecture or another checkpoint.
For this short input there is one temporal window and one spatial task, so the
upstream preset's eight-way window distribution does not split that decoder
task over eight GPUs.

This comparison is restricted to the six-latent, one-window fixture. The README
example uses 30-latent segments; overlapping windows can process latent positions
more than once, while DiT attention scales differently from the linears above.
Do not extrapolate the decoder/denoiser ratio to a five-second segment or a
complete video without recomputing the target geometry and window schedule.

There is also a reference-backend trap. The current `_flex_attention` compiles
its fused kernel only on CUDA; its docstring explicitly says eager execution
materializes the full score matrix. The default non-CUDA, non-compiled path
uses that eager implementation; the preset sets `compile_decoder: false`.
A hypothetical F32 score tensor at this fixture is
`21120^2 * 32 * 4 = 57,094,963,200 bytes`, or 53.174 GiB, before other tensors.
This is a size calculation, not an observed allocation or a verified MPS
execution. Do not launch the full eager decoder as an MPS oracle or implement
local attention as a dense score matrix plus a mask. Physically bounded
neighborhood/tiled computation needs its own unchanged-tolerance check against
the upstream semantics, initially at small shapes; an adapted oracle must be
identified as adapted rather than claimed as a CUDA execution.

Next discriminator: inventory the actual decoder checkpoint and its required
architecture, then decide whether a bounded decoder-block probe is feasible at
the target token/window shape. Admit aggregate Qwen + DiT + decoder weights and
overlapping workspace lifetimes before allocation. Reuse suitable CogniQwen
RMS/GQA/MLP building blocks, but explicitly implement the decoder's T/H/W RoPE,
bidirectional local attention and precision boundaries; a causal Qwen3.5 graph
is not a drop-in decoder. A block timing is only an estimate until the head,
windowing and complete decode are measured.

Sources: [decoder architecture and token mapping](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/hymm/models/vae/pixelumm_decoder/model.py#L34-L68),
[decoder forward](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/hymm/models/vae/pixelumm_decoder/model.py#L289-L350),
[local attention and eager fallback](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/hymm/models/vae/pixelumm_decoder/model.py#L125-L173),
[decode windows and sampler](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/hymm/models/vae/pixelumm_decoder/autoencoder.py#L350-L418),
[inference preset](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/configs/experiment/inference.yaml).

## Exact candidate: remove unused text rows after the refiner

The saved-input fixture has 5,280 image tokens, 1,280 text rows, and 23 valid
text tokens. Attention already uses only the 5,303-token valid prefix.
`HyNomasConfig.text_rows` and all text linear/FFN operations still use 1,280
rows. Padded rows have per-row norms, modulation and residuals, and never enter
valid attention queries or keys. Under the admitted batch-one, single-segment,
no-history contract, eliminating them after the full text refiner should not
change image outputs or the 23 valid text outputs. This is a dataflow argument,
not a full-size numerical result: changing M can change MPS reductions.

For D2048/F8192, token-wise Q/K/V, output and FFN linears cost

```text
Flin/row = 2*(4*D^2 + 2*D*F) = 100,663,296
Flin/block = Flin/row * (5280+1280) = 660.351 GFLOP
Fattn_QK_PV/block = 4*(5280+23)^2*D = 230.374 GFLOP
Fremoved/block = 1257 * Flin/row = 126.534 GFLOP
Fremoved / (Flin + Fattn_QK_PV) = 14.206%
```

This is an arithmetic opportunity, not a 14.2% latency claim. The fraction
depends on prompt length and excludes other work. It does not remove image
tokens or reduce already-compact attention. A short valid text prefix may
also favor a different small-M backend; existing M23 split-K evidence is not
permission to change the full model's operand/reduction contract.

Falsifier: keep Qwen/refiner inputs and outputs unchanged, bind a compact valid
DiT text prefix on GPU, and compare image outputs at blocks 0/7/53, valid text
outputs and final latent against the current same-operand upstream/control at
finite `atol=rtol=2e-5`. Poison the excluded suffix; valid outputs must be
invariant. These selected checkpoints and the final result establish only that
scoped contract; claiming all 54 intermediate boundaries match requires
streaming comparisons at every boundary. Rebuild fixed plans and verify
lifetimes, zero encode allocations, zero preterminal reads and one terminal
wait. Run opposite-order A/B pairs on the complete saved-input forward only
after correctness passes. Reject on any
valid-output failure; do not widen tolerances. Unused padded text outputs are
explicitly outside this proposed consumer contract, so the old all-1280-row
capture hash is not an appropriate equivalence claim. Multi-segment/history
support must reopen the proof before reusing this optimization.

Anchors: [block dataflow and linear shapes](../src/ml/video/hy_nomas_block.cr),
[valid-prefix attention plan](../src/ml/video/hy_nomas_block_mps.cr),
[existing tiny prefix test](../spec/hy_nomas_mps_padded_prefix_spec.cr),
[saved-input geometry](../examples/support/hy_nomas_q4_dit_probe.cr).

## Algorithmic leverage and the latency budget

The inspected current upstream README explicitly supports distilled 4/6/8-step
inference and shows a four-step Diffusers example; the native inference preset
still defaults to eight. The local source checkpoint is already the distilled
DiT. Check immutable checkpoint identity, scheduler and precision behavior
before qualifying a four-step native path; fewer steps are an approximate
generation policy, not an exact kernel rewrite.

Using the observed 6.42-second saved-input forward only as a conditional unit:

| DiT forward calls | Conditional DiT seconds |
| --- | ---: |
| 4 | 25.68 |
| 6 | 38.52 |
| 8 | 51.36 |
| 9, if a non-final cache-write forward is required | 57.78 |

These are not measured seconds per second of video. They exclude live Qwen
composition, scheduler, PixelUMM and output handling; input/segment geometry
and step-dependent runtime can differ. Do not multiply by CFG=2 blindly: the
inspected default guidance is 1.0. Do not remove a required cache write as if
it were redundant. A four-step path can halve denoiser calls while saving much
less than half the complete pipeline time if decoding is expensive.

Source: [distilled model and four-step example](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/37372925fd4bbb7977a89ce3248584926fc9e953/README.md#model-assets).
Qualify step-count quality with identical initial noise, prompts and output
settings against the eight-step control when decoding is available. Report
motion, consistency and semantic adherence as well as latency; implementation
parity alone does not establish acceptable four-step generation.

## Quantization: optimize the error seen by the model

For a fixed input X and weight error E, the projection error is `X E^T`:
`||X E^T||_F^2 = tr(E (X^T X) E^T)`. Weight-space MSE, a Gaussian weight
histogram, and largest scalar residuals omit activation covariance and
downstream amplification. Repeating 23 rows 56 times provides no new coverage.
The earlier Wan activation-fit experiment already improved its two training
rows and regressed on the other two in both folds. Do not repeat that route
without genuinely new calibration and independent held-out prompts/timesteps.

Prefer a bounded Q8 or mixed BF16/Q8/Q4 quality screen before adding codec or
GEMM complexity. Allocate precision to sensitive matrices using identical real
inputs and downstream outputs, with the complete resident byte budget counted.
Sparse deltas can repair isolated errors; they are not automatically a compact
model of structured error. A more ambitious hypothesis is a high-precision
low-rank branch extracted before quantizing the residual, evaluated against
higher-bit and sparse-patch alternatives at equal complete bytes. SVDQuant
provides a primary example of this design and of fusing the branches to avoid
activation traffic. Its NVIDIA W4A4 results do not establish M2 Max W4A16 speed
or Hy-Nomas quality, and a post-hoc low-rank fit of already-quantized error is
not the same algorithm.

Source: [SVDQuant, sections 4.2-4.3](https://arxiv.org/html/2411.05007v4#S4.SS2).
Prior falsifier: [Wan held-out calibration regression](wan-joint-affine-benchmark.md#distortion-and-allocation).
Every representation keeps a separate unchanged same-operand implementation
gate. Candidate representations must reduce distortion relative to original
BF16 model outputs, not only pass against their own decoded oracle. No
visible-quality claim is admitted before decode.

## Kernel work that earns another probe

Large-M GEMM reuses each weight over many input rows. Smaller stored weights
can improve feasibility and reduce pressure without reducing MMA count or
being the limiting bandwidth term. Register decoding, panel reloads, shared
memory and accumulator pressure can consume the benefit. The tested B128
weight reuse and K16 storage reduction are concrete counterexamples to a
universal smaller-payload/faster-kernel claim; they do not identify a causal
hardware stall.

Measure actual image weights/activations and the saved-input stack's phase
share before another large-M kernel branch. Amdahl's bound for a phase of
fraction p accelerated by s is `1 / (1-p+p/s)`; without p there is no defensible
video speed estimate. Existing timestamps can orient this without new waits.
If that share justifies kernel work, candidates are FC1 + unchanged F32 GELU
+ next-input rounding, and FC2 + bias + gated residual epilogues. Compare
intermediate values first, preserve contraction/rounding order, then time the
whole FFN/block. These fusions currently apply to the experimental direct-Q4
route; they are not already-qualified changes to the staged MPS production path.
Do not reopen split-K at M1280 without new evidence addressing its 949 failures
and measured slowdown.

## Next frontier and failure policy

1. Inventory the actual PixelUMM checkpoint, target segment/window geometry and
   aggregate resident memory. This is a decision probe, not a decoder-port
   commitment or a phase-timing result.
2. Implement the opt-in DiT valid-text compaction falsifier, then measure the
   same saved-input stack with interleaved orders if it passes.
3. Qualify and time a bounded decoder block only after the inventory admits it,
   with physically local attention and an explicitly scoped upstream control.
   Treat block timing as provisional until a complete decode is measured.
4. Screen higher-bit/mixed weights on real held-out inputs; retain original
   BF16 as the quality control and compact staging as the memory fallback.
5. Qualify the distilled four-step schedule against eight-step output quality
   after the decoder and live producer handoffs are available.

All GPU probes remain serial, fully resident in the model path, with no added
preterminal synchronization, unchanged finite tolerances and the cache-aware
10% physical-memory reserve plus existing footprint/swap/deadline guards.
Temporary files stay under `/opt/mldata_ssd/`. No full generation is authorized
by this audit. Rollback is to leave proposed paths unused and retain existing
BF16/MPS controls. Stop a branch when its scoped falsifier fails or its maximum
pipeline headroom cannot affect the value target; another geometry tweak alone
does not reopen a refuted hypothesis.

The correlated Adversary review is ROBUST within this source/evidence audit's
scope. It does not establish future runtime, end-to-end quality or independent
replication. Audit verification: inspected current source and frozen report
summaries; recomputed the FLOP, storage and conditional-time arithmetic directly; checked
document links and `git diff --check`. These checks validate this decision's
anchors and arithmetic, not future speed or quality. Refresh on source,
checkpoint, token/mask contract, segment/history, precision, toolchain, device,
or missing artifact changes. The decoder architecture is explicitly a
current-main observation and must be matched to its chosen checkpoint before
implementation.
