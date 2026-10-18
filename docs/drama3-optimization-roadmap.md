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
2. Exact GDN postprojection fusion, preserving all reduction trees. Profile the
   unranked recurrence/normalization remainder before forecasting a gain.
   The smallest source-backed corridor is `rms(output, norm.weight) -> silu(z)
   -> multiply` at `backbone_resident.cr:358-362`. Preserve the sequential
   per-head RMS fold and expression order; leave recurrence state writes and
   `out_proj` unchanged. First falsifier: old versus fused F32 expressions at
   edge head widths, followed by GPU output/state bits on actual operands.
   This is dataflow fusion, not a new rank-one algebra result.
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

The strongest source-backed match for the deployed decode simplification is the
fused DeltaNet state update and output in
`src/ml/gguf/kernels/delta_net.metal:645` in the primary Qwen checkout:

```
delta = beta * (V - g * dot(S, K))
S_next = g * S + K * delta
y = scale * dot(S_next, Q)
```

The rank-one update allows output accumulation while writing the new state,
removing a separate materialized decay and a later state read. The historical
`LM-codex-DN-FUSED-EXACT-1` records direct numerical gates and paired 20/64-token
decode gains; `qwen35_metal.cr` still selects this path by default unless
`QWEN35_DN_FUSED=0`. Those are historical Qwen measurements, not DRAMA evidence.
Its prior numerical checks permit small differences; they do not certify the
unchanged bitwise reduction contract of the current DRAMA slice.

The alternate identity `dot(g*S + K*delta, Q) = g*dot(S,Q) + delta*dot(K,Q)`
was tested separately and failed to improve longer decode. Associative affine
composition/WY block scan has a small-shape algebra proof but its attempted
compact Metal route lost to the rowwise kernel. Do not present either as the
deployed decode win. See `LANDMARKS.md` entries `LM-codex-DN-FUSED-EXACT-1`,
the output-dot refutation near line 2888, and block scan near line 4961.

Two related techniques have different contracts. Q4 prefill shares an H16 input
conversion that both original GEMMs already perform; its guarded FFN route is
`qwen35_metal.cr:3597` and `:6820` (`LM-codex-Q4K-PAIR-H16-1`). This is exact
dataflow reuse only when that same rounding already exists. Introducing H16
inputs to a current F32 path would change precision. Projected-K rank-64
recurrence (`LM-QWEN35-DELTANET-LOWRANK-1`) is approximate/probe-only: a prompt
falsifier found a greedy-token mismatch. Exact verification can protect a
committed result but did not establish a speed win over ordinary exact decode.
Neither route qualifies DRAMA under its current precision contract. The
associative scan is a future long-prefill hypothesis, not a deployed win;
algebraic equivalence alone does not preserve F32 operation order.

A second relevant negative prior: Qwen's Q8 paired SwiGLU passed focused
correctness but regressed wall time and was reverted. DRAMA uses different
affine weights and shapes; real-input and complete-request pairs must decide
whether this attempt survives. Historical compressed-state and shape-specific
wins remain hypotheses here, not fresh DRAMA performance evidence.

Refresh after source, compiler/driver, device, weights, prompt, sampler or
reference changes. Active return pointer: `docs/drama3-native-frontier.md`.
