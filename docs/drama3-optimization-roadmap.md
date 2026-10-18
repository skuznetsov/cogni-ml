# DRAMA-3 optimization continuation

Audit anchor: `a8156b8c7a048faebd4f0eb10d4a323839169498`, Apple M2 Max,
38 GPU cores, 64 GiB. This is a source-linked continuation record, not new
benchmark evidence. User authorized implementation after the 2026-10-18 audit.

## Confirmed frontier

- Activations, routing, sampling arithmetic, reference encoding and DAC stay on
  GPU. Per-frame CPU EOS reads remain in `generator_resident.cr:179`.
- CogniGraph controls bounded submission and lifetimes, rather than replaying
  a previously encoded decode graph on this path.
- [Ordered down/sum fusion](drama3-expert-down-sum.md) is exact versus the native
  baseline in its saved scope and remains default-off. Local command gains do
  not establish a reliable full-WAV win.
- [Gate/up/SwiGLU fusion](drama3-expert-swiglu.md) now has bounded exact GPU,
  upstream no-regression and 24-request checks. It removes 21,360 warm-request
  allocations, but one timing block is unchanged and one mirrored contrast
  regresses. It remains default-off; this is not a repeatable full-WAV speedup.
- [GDN postprojection fusion](drama3-gdn-post.md) retains exact native output
  bits and saved MLX metrics in the tested scope. Local paired command intervals
  favor fusion at 2.5806x and warm allocations decrease by 5,340. Two of four
  full-WAV worker contrasts regress; it also remains default-off.
- [Late upstream differences](drama3-late-matched.md) remain unresolved. Preserve
  existing tolerances; unchanged MLX metrics are a no-regression claim.
- [Projection replay](drama3-decode-replay.md) provides bounded family costs;
  [shader attribution](drama3-complete-decode-profile.md) was rejected. Do not
  sum overlapping command intervals into an additive request budget.
- Header-only inspection confirms 30 BF16 qkv[8192,2048] and 30 BF16 z[4096,2048],
  totaling 1,509,949,440 logical source bytes/pass. These are neither persistent
  F32 matrices nor measured DRAM traffic.
- A saved 60-frame clip lasts 2.786395 s; before this slice, complete warm requests took
  roughly 22.6-25.1 s excluding weight loading. An older decode window implies
  about 185 ms/frame versus 46.44 ms audio/frame. RTF 0.8-0.9 needs 37-42 ms/frame;
  this motivates a 4.4-5x bounded decode target, not a hardware impossibility claim.

## Accepted implementation order

1. Implemented, default-off: exact gate/up/SwiGLU fusion for routed and shared
   experts. Preserve original
   F32 reductions and stable sigmoid; qualify actual specialized dispatches,
   intermediate GPU bit checks, old upstream metrics and full WAV pairs.
2. Implemented, default-off: GDN postprojection `rms(output, norm.weight) ->
   silu(z) -> multiply`. The sequential per-head RMS fold and expression order
   are preserved; recurrence state writes and `out_proj` are unchanged. The
   25-example GPU suite, 90 actual-input captures, 243 native boundaries and
   24 complete requests pass their declared no-regression gates. Strict MLX
   parity and repeatable causal full-request benefit remain unestablished.
   This is dataflow fusion, not temporal GDN parallelism.
3. q1 wavefront experiment: next backbone needs q0/q1 only, while tower finishes
   q2..q9. Actual overlap needs GPU queue dependencies, isolated scratch and
   retained frames. Single-queue reordering alone is insufficient; theoretical
   saving is bounded by min(backbone, tower-tail), neither measured here.
4. Bounded decode blocks with GPU EOS/active state. Inactive rows must suppress
   all cache, ring, recurrence and finite-guard effects. Test every stop position.
5. Separate approximate experiment: calibrated BF16 qkv/z to Q8, initially
   preserving router computation and F32 recurrence. Use activation-weighted
   output error and full audio quality, never scalar weight MSE alone.
6. Support the trained line-separated protocol and causal DAC chunks. Current
   EOS+4096-semantic head cannot emit audio_separator. This needs head and state
   machine support; it is not a demonstrated cause of whisper silence.
   Cache exact reference codes before attempting complete prefix-state reuse.

## Quality and promotion boundaries

Keep exact gates and approximate quality experiments separate. First-take
completeness, omissions, unrequested silence and latency including retries matter.
ASR hallucinated text on digital zero; ASR and RMS alone cannot certify speech.
Use RU/EN, reference/no-reference, whisper-middle/end, laughter/dialogue,
short/long inputs and multiple seeds. Qualify the intended combined flag stack,
not only each switch in isolation. Streaming improves first-audio latency but
does not sustain playback while generation is slower than playback.

Heavy runs stay serial under run_safe with external scratch on `/opt/mldata_ssd`,
local weights and the user-authorized 15% memory floor. Do not terminate unrelated
processes or restart the live Studio. No default promotion without repeatable
complete-request benefit. Ordinary fusion, batching and overlap are not LTP/WBA.

## Research pointers and open questions

- [DRAMA report](https://arxiv.org/html/2610.18010v2): partial codebook fusion,
  exact GDN fusion, failed router fusion, line-separated and lookahead protocols.
- [TreeWY](https://arxiv.org/html/2608.20961v1) and
  [ReplaySSM](https://arxiv.org/html/2610.13778v1): GDN state/verification ideas;
  not evidence of local bitwise parity or automatic target-evaluation savings.
- [RVQ speculation](https://arxiv.org/html/2609.37007v1): shared inner decoder
  differs from DRAMA's distinct residual layers. Distribution preservation is
  distinct from fixed-random-tape path identity.

## Earlier Qwen3.5 mathematical simplification

The intended reference is Qwen3.5's surrogate and temporal-GDN parallelism
work, not separate rank-one update/output fusion (`LM-codex-DN-FUSED-EXACT-1`).
Evidence is from primary Qwen HEAD `d07e3089`, not DRAMA measurements.

- **Projected-K surrogate (approximate state).** Calibration K vectors define a
  fixed per-head basis `U` (`bin/qwen35_deltanet_fixed_basis_probe.cr:3142-3158`).
  For `S≈MUᵀ`, `c=Uᵀk`, `q̄=Uᵀq`, update
  `M′=gM+β(v−gMc)cᵀ`, emit `y=M′q̄`
  (`bin/qwen35_deltanet_fixed_basis_probe.cr:1275-1309`; exact recurrence:
  `src/ml/gguf/qwen35_cpu.cr:1052-1099`). This is exact only for the projected-K
  model. Chunk K/Q projection is token-parallel, but the low-rank state scan
  still loops serially over tokens and parallelizes rank work
  (`src/ml/gguf/kernels/delta_net.metal:81-112, 397-470`): it reduces state
  work, not exact temporal dependence.
- Local known-span gains: 24-token core `9.63→4.55 ms` (9B),
  `13.99→4.04 ms` (27B); full layer `48.9→16.1 ms`, `77.1→29.1 ms`
  (`LANDMARKS.md:767-783`). Autoregressive draft was slower than exact
  (`188.9/630.5 ms` vs `180.1/493.9 ms`); ideal-overlap estimates were only
  `1.0579x/1.0839x` and remained draft-bound (`LANDMARKS.md:788-793`).
- **Block-residual surrogate (approximate layer skip).** The probe collects
  exact block input/output pairs, fits PCA plus a ridge residual map, then uses
  `x+Δ̂(x)` in place of the selected block
  (`bin/qwen35_deltanet_fixed_basis_probe.cr:3248-3287, 3839-3903, 3946-4025`).
  It predicts a block; it is not a GDN scan. On 9B band `26:27`, rank 16,
  gamma 8 reached `92.89%` mean / `83.33%` minimum acceptance with exact greedy
  parity, but self-spec was `0.4553x` sequential exact speed (`10727.5 ms`
  baseline vs `23565.8 ms`; ideal overlap `0.8764x`;
  `LANDMARKS.md:6556-6580`). A wider band fell from near break-even at gen 8 to
  `57.75%` mean acceptance at gen 16; correction preserved parity but charged
  the rejects (`LANDMARKS.md:6604-6624`).
- **Exact affine scan (temporal parallelism).** Per value row,
  `rₜ=rₜ₋₁Aₜ+Bₜ`, `Aₜ=gₜ(I−βₜkₜkₜᵀ)`, `Bₜ=βₜvₜkₜᵀ`; compose
  `(A₁,B₁)∘(A₂,B₂)=(A₁A₂,B₁A₂+B₂)`, scan block prefixes, then replay for every
  output (`src/ml/gguf/qwen35_deltanet_block_scan.cr:142-170`,
  `src/ml/gguf/qwen35_deltanet_scan_model.cr:26-63`). Algebra tests passed.
  A compact-B subkernel improved `2.179→0.302 ms`, but the full single-block,
  state-only route lost to rowwise (`0.422-0.714 ms` vs `0.277-0.516 ms`); prefix
  scan/replay/output costs remain, and no product integration followed
  (`LANDMARKS.md:4961-5071`). Changed F32 order means algebraic exactness is not
  bitwise equivalence.

**Verifier/deployment.** The scanner accepts candidate IDs through the first
exact target top-1 mismatch, emits the target ID, and leaves state replay to the
caller (`src/ml/gguf/qwen35_spec_acceptance.cr:1-47`). The wall runner snapshots
verifier state, verifies known candidates, then restores and replays the emitted
prefix through exact target prefill after rejection (`bin/qwen35_deltanet_fixed_basis_probe.cr:7280-7319`). This protects the fixed temperature-zero greedy sequence; it does
not preserve stochastic target distribution or a fixed random-tape sample path
(no probability-ratio or residual-draw step). Product generation still says
`product_self_spec=unsupported decode_path=unchanged`
(`bin/qwen35_generate.cr:721-740`); its plan validates routes without executing
approximate proposal bodies (`src/ml/gguf/qwen35_self_spec_plan.cr:7-11`).

**DRAMA transfer.** Exact affine scan is a prefill candidate: `BackboneResident#prefill`
is layer-major and batches projections, while GDN currently processes rows in
order (`src/ml/audio/drama3/backbone_prefill.cr:11-41, 87-101`;
`src/ml/audio/drama3/backbone_resident.cr:346-360`). Prompt-frame chunks are known inputs
(`src/ml/audio/drama3/generator_resident.cr:396-420`); preserve convolution
history order and existing F32 output/state gates. During generation, the next
main token and residual codes are sampled before the next scalar backbone step
(`src/ml/audio/drama3/generator_resident.cr:164-205`), so future GDN inputs are
unknown. Qwen therefore motivates a bounded prefill falsifier, not a DRAMA
decode claim. Projected-K or block residual changes the exact model/sampler
contract and requires separate quality and verification evidence.

Immediate mathematical falsifier: a bounded known-input prefill block at the
production GDN dimensions, including summary construction, prefix composition,
replay, every output and final state commit. Test the existing output/state gates
before forecasting speed or touching production routing. State-only subkernel
gains are insufficient; changed F32 reduction order may reject the route. q1
overlap and the other accepted branches remain available, unimplemented paths.

Refresh after source, compiler/driver, device, weights, prompt, sampler or
reference changes. Active return pointer: `docs/drama3-native-frontier.md`.
