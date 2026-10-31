# Hy-Nomas performance: execution cost and architecture choices

## Scope and decision (2026-10-31)

This is a source and evidence audit of local revision `987cae68`, not a new
performance result. Three correlated reviewers examined benchmark provenance,
matrix/precision costs, and primary upstream implementations. The work-budget
helper below runs on the CPU without model tensors. No GPU/model run, full
generation, production selector, precision policy, or synchronization changed.

Preserve the current qualified BF16 weights/stage boundaries and production
F32-FFN mode. Stop treating fewer Graph calls or a
nominal twofold compute gap as sufficient evidence for another fusion sweep.
The next strict candidate is **compact K/V sharing with unchanged per-query
math**; full online attention is a separate, higher-risk step. An alternative
decoder is a model change, not a free inference optimization. The first cheap
architecture screen rejects the assumption that original DCAE-V necessarily
has less computation: its dense-convolution budget is about 602.487 TFLOP
versus PixelUMM's 474.917 TFLOP matrix budget on the same output shape.
Original BF16 quality remains the active target; this audit neither runs nor
enables a new decoder policy. No candidate has measured local speed or
quality qualification.

The current boundary remains unchanged: GPU-only model computation, reused
buffers, no new explicit waits/readbacks in the measured path,
`atol=rtol=2e-5`, cache-aware 10% physical reserve, declared process cap,
64 MiB swap-growth stop, bounded deadline, and no full generation.

## What the evidence says, and what it does not

The current implementation already reuses weights, plans and scratch across
decoder windows. Generation opts into the faster measured F32 FFN path while
preserving BF16 stage outputs. The earlier 25–29% prepared-block improvement
is already in this baseline; it is not an unclaimed future optimization.
See [the FFN experiment](hy-nomas-pixelumm-block-frontier.md#earlier-single-block-f32-ffn-graph-qualification-2026-10-27)
and `examples/support/hy_nomas_generation.cr:Decoder` (`ffn_f32: true`).

The recent full profile is one T6/H44/W80, J21120 diagnostic chain. A real
768x416 five-second decode executes five T10/H26/W48, J12480 windows. Comparing
their request times does not isolate engine overhead:

| Decoder geometry | Windows | Aggregate FFN rows | Local attention tiles/window | 36 blocks + head, TFLOP |
| --- | ---: | ---: | ---: | ---: |
| Diagnostic T6/H44/W80 | 1 | 21,120 | 60 | 335.016 |
| Saved 768x416 T10/H26/W48 | 5 | 62,400 | 24 | 1,060.015 |

The second workload contains 3.16 times the counted matrix work, excluding the
stem in both rows. Its aggregate `sum(query_count * key_count)` is 4.66 times
larger. Neither token count alone nor tile count alone normalizes attention.
These are operation counts, not measured GPU service or latency predictions.

The separate historical "twofold slowdown after DiT" was not reproduced in
matched-input ABBA controls (mean ratios 1.03 and 1.043). Those means are
descriptive: order/time drift exceeds the difference, and they do not identify
a causal DiT penalty. A nominal FP32-peak denominator also
does not establish that software wastes half the available throughput.
[The generation frontier](hy-nomas-generation-frontier.md) retains the scope,
frequency, fixture and observer-provenance limits of those experiments.

The latest profile assigns 12.890 s solely to gate/up, 6.734 s solely to down,
and 11.334 s solely to attention in a 42.311 s root-interval union. Mixed
roots remain separate. The preceding off control differs by 3.49% in sampled
active frequency and fails the 2% bracketing gate: these are descriptive
buckets, not qualified observer-overhead or speedup evidence. These intervals
include stalls. The 38.309 s of host
continuation time in attention locates CPU waiting, not the queued GPU work
causing it. Removing host waiting or reducing API calls cannot be assumed to
remove the GPU critical path. Sampled 100% GPU active residency is likewise
not ALU utilization.

The exact FFN front candidate reduced Graph calls but increased root commands
from 897 to 1153 per 128 FFNs. It failed its speed screen. Earlier native
tiles, orientation changes, packed gate/up and B2 schedules rejected specific
implementations, not every native kernel or all batching. Captured-stage
parity also did not qualify separately compiled output-only FFN graphs.

## Count the model before caching its denoiser

Reproduce the geometry and counts without loading weights:

```sh
python3 examples/hy_nomas_work_budget.py
```

The helper checks the published eight-step count and ragged tile geometry.
For the fixed 512x288, 120-frame first segment with saved Qwen conditioning:

| Matrix work, TFLOP (two FLOPs/MAC) | 4 DiT steps | 8 DiT steps |
| --- | ---: | ---: |
| DiT, including per-step refiner/adapters | 129.830 | 259.661 |
| PixelUMM, five windows including stem/head | 474.917 | 474.917 |
| Total | 604.747 | 734.577 |
| PixelUMM fraction of counted work | 78.53% | 64.65% |

Refiner and adapters also scale with denoising steps. These shares omit live
Qwen, elementwise operations, reductions, casts, resource preparation, export,
and later-segment history. They are not time shares or speedup bounds.
Denoiser-only changes have a smaller counted-work ceiling in this preset;
their wall impact remains unmeasured. PixelUMM itself is already distilled to
one decoder step. Its 36 spatial/temporal transformer blocks run for every
overlapping decode window; "one step" does not mean a small conventional VAE.

Five length-10 windows at stride 5 cover 30 unique latent frames through 50
window-frame appearances. The repeated inputs do not imply repeated block
states: attention is bidirectional inside each window and the extra context
differs. Reusing overlap K/V through the stack would change the model. Exact
stem reuse is plausible, but all stem/head matrix work combined is only
1.457 TFLOP in the low-resolution model; overlap-only stem savings are smaller.

## What faster upstream systems actually use

Primary sources were inspected at Hy-Nomas
[`effb8a49b83e6f1d35e26710db35194f2fd952a7`](https://github.com/Tencent-Hunyuan/Hy-Nomas/tree/effb8a49b83e6f1d35e26710db35194f2fd952a7)
and MLX
[`339279525ea3168b6fa55911529e3766a145d0b2`](https://github.com/ml-explore/mlx/tree/339279525ea3168b6fa55911529e3766a145d0b2).
This is newer than the pinned local Hy-Nomas oracle; it does not replace that
oracle or transfer its qualification to new source.

| Mechanism | Transfer to the current M2 path |
| --- | --- |
| SGLang FA3, Ulysses/TP, decoder windows on multiple GPUs | CUDA/hardware/topology-specific; not a single-M2 speed prediction. |
| Fused modulation, QK norm/RoPE, SwiGLU | Portable kernel patterns. Upstream's bitwise claim covers its CUDA path; local BF16 boundaries still need qualification. |
| INT8 DiT and historical KV storage | Memory optimization with BF16 computation; not evidence that packed GEMM accelerates our large-row FFN. |
| CUTLASS FP8 PixelUMM | Uses a different numerical path and NVIDIA kernels. No equivalent M2 throughput claim follows. |
| MLX fused Metal GQA SDPA | Relevant local kernel reference. D=128 and grouped KV are supported; reduction-order parity and speed here remain untested. |
| Routed sparse DiT attention | Omits keys. Upstream keeps dense early steps, then selects 25% of candidate blocks. Approximate quality branch. A dense masked SDPA alone does not save those FLOPs. |
| TeaCache / FirstBlockCache | Approximate denoiser reuse. Only 4–8 steps here, and it does not remove the one-step decoder. Some implementations also synchronize a GPU-derived host branch. |

Sources: [Hy-Nomas README](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/effb8a49b83e6f1d35e26710db35194f2fd952a7/README.md#inference-with-sglang),
[report section 6.2](https://arxiv.org/html/2610.36729v1#S6.SS2),
[sparse configuration](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/effb8a49b83e6f1d35e26710db35194f2fd952a7/configs/experiment/inference.yaml),
[MLX Metal dispatch](https://github.com/ml-explore/mlx/blob/339279525ea3168b6fa55911529e3766a145d0b2/mlx/backend/metal/scaled_dot_product_attention.cpp#L1172),
[TeaCache](https://arxiv.org/abs/2411.19108), and
[FirstBlockCache source](https://github.com/huggingface/diffusers/blob/86ef420b17363c28e36ebad6b58555e9aa8feda2/src/diffusers/hooks/first_block_cache.py).
This is a selected mechanism audit, not a comprehensive survey or a reproduced
comparison with those engines.

## Exact execution: change materialization first

Local `pixelumm_gather_kv` and `pixelumm_gather_kv_pair` expand eight BF16 KV
heads into 32 F32 panel heads using `head / 4`. QK, softmax and PV are separate
graphs with explicit F32 score/probability panels. Source anchors:
`src/ml/video/pixelumm_decoder_block_mps_bridge.mm:pixelumm_gather_kv_pair`
and `pixel_attention_qk_step` / `pixel_attention_pv_step`.

There are two different proposals:

1. **Keep only eight KV heads and share reads across four query heads.**
   Preserve every query's ordered keys, QK/PV reduction dimensions, F32
   scores/probabilities and BF16 output boundaries. A broadcast or stride
   change can still change MPS lowering; this is mathematically equivalent,
   not yet numerically qualified. Failed adjacent-query concatenation does
   not refute compact storage with independent query math.
2. **Eliminate scores/probabilities with tiled online softmax.** This changes
   the reduction schedule and has a harder precision gate. The old SDPA
   failure is a warning; repeating that unchanged implementation is not a
   new experiment. MLX is an implementation reference, not evidence of local
   parity or a reason to add an MLX runtime dependency.

Across five J12480 windows and 36 blocks, the count model describes 2.994 TB
of logical score/probability writes and 0.535 TB of duplicate KV writes that
compact storage could remove. These are not physical DRAM bytes. Dividing
them by advertised bandwidth would be a transfer scenario, not a bound or a
forecast of saved wall time. GQA preserves all 32 query-head QK/PV products;
neither compact storage nor FlashAttention gives a fourfold FLOP reduction.

The first falsifier is one saved real J12480 block: compare QK, probabilities,
attended F32, post-attention BF16 and block output to the current source-MPS
controls at unchanged tolerances. Use existing GPU comparators and the
terminal wait. Time only a passing executable, with reused resources and
qualified balanced controls. Require an isolated full-block speed screen
before testing the complete five-window decoder for promotion. Stop if
compact input lowering silently expands KV again or
does not improve the measured request.

BF16 rounding explains why this is harder than real-arithmetic equivalence.
Near magnitude one, adjacent BF16 numbers differ by about 0.0078, far above
the approximately 0.00004 comparison allowance there. A small F32 reduction
change can move a value across a BF16 midpoint. A more accurate real-valued
GEMM need not match the finite-precision MPS oracle. The contract constrains
observed results, however; it does not mandate a particular implementation.

F16 operands with F32 accumulation remain a separate research route. Many
BF16 operands are exactly representable in F16, but its exponent/subnormal
range is much narrower. Exact range splitting adds products and different
reductions. It must first survive cancellation, full-range operands and
BF16-midpoint falsifiers. New TensorOps APIs do not imply M2 neural-accelerator
hardware or a speedup over current MPS. This is lower priority than the
materialization question; the failed old tile family is not a global proof.
In the prior J2048 projection falsifier, BF16-to-F16-to-F32 MPS operands
produce bitwise-equal BF16 outputs despite recorded tiny operand losses, while
the native F16 MMA fails. Those losses alone therefore do not explain the
failure. Reduction scheduling remains an open discriminator; a different
precision name or a more accurate dot product is not sufficient evidence.

## Architecture: reject the cheap-decoder assumption before porting

The official pipeline still accepts original **DCAE-V f32t4c32** as an
alternative decoder for the same 32-channel latent space, with shared VAE
scaling/dtype preparation. Its chunk-causal convolutional decoder is a
different computation from the 36-block PixelUMM model. This is not a
configuration that removes PixelUMM work while preserving its outputs.

Primary source: [pipeline selection](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/effb8a49b83e6f1d35e26710db35194f2fd952a7/hymm/pipeline/base.py#L159),
[latent preparation](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/effb8a49b83e6f1d35e26710db35194f2fd952a7/hymm/pipeline/base.py#L297),
[DCAE-V implementation/config](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/effb8a49b83e6f1d35e26710db35194f2fd952a7/hymm/models/vae/autoencoder_dcae_v.py#L1509).

The authors explicitly introduced PixelUMM's generative prior because the
original DCAE-V decoder produced repetitive grid-like complex textures.
[Their report](https://arxiv.org/html/2610.36729v1#S6.SS2) identifies a quality
risk; it does not establish how this decoder performs on our generated
latents. The separate Q4 rejection does not decide this tradeoff.

The new CPU-only counter follows the pinned configuration: widths
`(128,256,512,512,1024,1024,1024)`, depths `(0,5,10,4,4,4,4)`, two full
3x3x3 convolutions per residual block, and pre-shuffle convolutions with
expanded output channels. Three causal length-10 latent tiles decode
32x30x9x16 latents to 3x120x288x512 without temporal tile overlap. For a
convolution, the conventional budget is `2*T*H*W*Cin*Cout*27` FLOPs:

| Decoder, same 120x288x512 output | Counted TFLOP |
| --- | ---: |
| PixelUMM, five windows, matrices including stem/head | 474.917 |
| DCAE-V, residual convolutions only | 542.060 |
| DCAE-V, all dense convolutions including upsampling/projectors | 602.487 |

The DCAE total is 1.269 times the PixelUMM count. These are conventional
dense-product budgets, not physical-instruction lower bounds or latency
predictions: padding treatment, convolution algorithms, arithmetic types and
hardware utilization differ. Normalization, activations, caching, residual
adds and shuffle traffic are excluded. This falsifies the **less nominal
arithmetic** rationale for prioritizing this port, not the possibility that
an efficient convolution implementation could run faster. No local DCAE-V
native timing or quality acceptance exists. It is demoted as a speed lead.

Reopen it only with evidence addressing both questions: a bounded same-latent
decode and full-motion quality comparison, plus a credible native convolution
time/memory budget. Check faces/eyes, fine textures, camera motion and chunk
seams across more than one clip. An upstream decode is a quality oracle;
timing a different backend is not a native-engine result. Plan memory before
running it. A draft-mode policy needs explicit full-motion quality acceptance;
static frames or a speed result cannot substitute. Each decoder's native
implementation must still match its own pinned upstream at the existing
tolerance; there is no cross-model 2e-5 equivalence claim.

A smaller decoder **trained or distilled from PixelUMM** is a more direct
longer-term architecture hypothesis: reduce width/depth/high-resolution work
while learning its texture prior. It requires training, data, quality criteria
and a separate resource decision. It is not an inference-only kernel trick
or a currently available speed result. Reducing temporal overlap or output
resolution is also a quality-changing policy, with seams/detail loss as
explicit falsifiers. None is enabled by this audit.

## Next decisions and evidence lifetime

The proposed next exact experiment is compact-KV numerical qualification.
Do not prioritize a DCAE-V port on the assumption of fewer operations. A
separate learned lightweight decoder is an architecture research direction,
not an approved replacement of the active quality target.
Defer further gate/up graph rearrangements and
denoiser-cache implementations unless a new probe addresses their recorded
failure or their actual fraction of request time.

Adversarial conclusion: **the claim that exact optimization is exhausted is
unsupported**. The defensible result is narrower: several local routes failed,
the model's decoder work is large, and current comparative numbers do not
prove a twofold software defect. Compact GQA/online attention and a smaller
learned decoder remain hypotheses; the presumed DCAE-V arithmetic advantage
did not survive the static screen.

The CPU helper executed successfully. A separate native CPU-only
`AttentionLayout` query/key enumeration reproduced the QK pair sums for all
three shapes: 69,672,960 (diagnostic), 64,972,800 (saved high resolution),
and 24,883,200 (low resolution). A second source-based convolution count
reproduced DCAE-V's 602,486,530,375,680 FLOPs exactly. Correlated adversarial
review is ROBUST for these count and claim scopes; it does not qualify a
kernel, native decoder, speedup, or video quality.

Refresh counts after shapes/windows/token rules/model changes; numerical
qualification after source/weights/backend/runtime changes; timing after host,
device/frequency, resource or queue-policy changes. Missing artifacts or
unbound telemetry cannot support affirmative performance claims.
