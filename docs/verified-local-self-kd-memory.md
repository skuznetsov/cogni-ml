# Verified Local Self-KD With External Memory

Date: 2026-05-28
Status: research synthesis, not a production claim

This note preserves the current working hypothesis that connects CogniQwen
surrogates, PCA/updown experiments, self-drafting, LTP/WBA, and the sibling
`../cogniformerus` external-memory work.

## Core Hypothesis

The useful compression target is not the whole Qwen model. The target is a
prompt-local and evidence-local transition operator:

```text
retrieved evidence + current hidden/state
  -> compact latent task state
  -> next reasoning/action/token proposal
  -> exact verifier accepts or falls back
```

This is closer to verified local self-distillation than to ordinary global
student distillation. The surrogate does not need to preserve every latent
feature or factual association. It only needs to propose candidates that the
exact model will accept inside a bounded corridor.

## Why PCA And Surrogates Look Plausible

Current PCA/updown and block-surrogate experiments suggest that parts of the
runtime trajectory live on much smaller local charts than the full dense FFN
space. That does not prove that the model is globally compressible. A more
conservative interpretation is:

- FFN blocks act partly like large associative memories.
- A prompt activates only a sparse/local subset of those associations.
- PCA can capture common/high-energy directions for the current regime.
- Rare tail facts and unusual mode switches are where low-rank surrogates drift.
- Exact verification makes this drift safe, but rejection economics still decide
  whether the surrogate is useful.

In this frame, the surrogate is not a replacement model. It is a proposal
operator whose cost, acceptance rate, and rollback tax must beat exact decode.

## External Memory Changes The Problem

`../cogniformerus` provides the missing complement: factual payload can be moved
out of the model's FFN tail and into typed retrieval. If external memory supplies
relevant facts as structured evidence, the surrogate can focus on local reasoning
and transformation over that evidence instead of preserving all facts internally.

Relevant sibling anchors at the time of this note:

- `/Users/sergey/Projects/Crystal/cogniformerus/LANDMARKS.md`: `Qwen3.5-9B` is
  the strongest measured offline coding default in that stack, with code eval
  `15/17` and small eval reasoning/tool path `100%`.
- `/Users/sergey/Projects/Crystal/cogniformerus/LANDMARKS.md`: the strategic
  multiplier is better evidence quality per token, not only bigger raw reasoning.
- `/Users/sergey/Projects/Crystal/cogniformerus/TODO.md`: the Small-Model Butler
  track targets a compact model with strong tool use and external-memory recall.
- `/Users/sergey/Projects/Crystal/cogniformerus/TODO.md`: the verified Qwen3.5-9B
  memory run reached `reasoning+memory 6/6`, `tool checks 3/3`, and overall
  `100%`; best cost/quality profile was `balanced + quad=on + max_tokens=192`.

This suggests a combined research direction: memory-conditioned local
surrogates, not standalone lossy Qwen compression.

## Proposed Object

```text
Memory-conditioned local surrogate
  key:
    evidence type
    topic/domain fingerprint
    layer band
    prompt-state features
    risk/entropy features
  value:
    PCA basis or local chart
    low-rank block residual map
    PCA up/down adapter
    risk gate and exact fallback policy
```

The object can be learned from exact Qwen activations and reused only when the
same trigger/corridor is detected. It should start as a probe artifact, then move
toward resident Metal/CUDA sidecar buffers only after acceptance and cost gates
pass.

## LTP/WBA Framing

This direction should only be called LTP/WBA when the full structure is present:

- Window / trigger: retrieved evidence is typed and matches a known local regime;
  hidden/state risk features are inside the safe corridor.
- Transport / corridor: a bounded token chunk, recurrent layer band, or evidence
  span is carried through a surrogate chart.
- Legal move: replace only the proposal path with a cheap surrogate; exact state,
  KV/session boundaries, and verifier semantics remain intact.
- Boundary safety: accepted chunks commit exact verifier state; rejected chunks
  discard surrogate state and resume exact decode.
- Lexicographic potential: reduce `(proposal bytes/kernels, verifier rows,
  rollback tax, prompt area)` without increasing earlier components after the
  active window is recomputed.
- Dual frame: exact decode, exact verifier, or full model path takes over when
  the local chart is sticky or risk gates fail.

This keeps LTP/WBA from degrading into generic speculation or batching.

## Falsifier Suite

The strongest next experiment is a closed-book versus memory-grounded comparison
on the same questions:

```text
A. closed-book prompt
B. prompt with retrieved typed evidence

Measure:
  PCA rank needed for top1 parity
  surrogate acceptance
  free-run drift
  verifier reject rate
  plain_speedup / wall speedup
```

Prediction: if the idea is real, memory-grounded prompts should need lower PCA
rank, show higher surrogate acceptance, and reject less often because factual
payload is supplied by retrieval rather than reconstructed from FFN tails.

If memory-grounded prompts do not improve rank/acceptance/reject economics, the
external-memory link is only useful for product quality, not for surrogate speed.

## Implementation Order

1. Add a probe that tags calibration/eval prompts as `closed_book` or
   `memory_grounded` and records the evidence type/domain fingerprint.
2. Reuse existing PCA-down/updown and low-rank self-spec harnesses on paired
   prompts.
3. Scan layer bands rather than broad suffixes; previous evidence shows 27B
   stable islands are not simply the final third.
4. Promote only configurations that improve both acceptance and wall economics.
5. Only then build fused/resident kernels for the winning surrogate family.

## Trust And Caveats

Trust: `{F:0.58,G:0.48,R:0.62}`.

Why not higher:

- The synthesis is grounded in local benchmark/probe history and sibling
  `cogniformerus` anchors, but the proposed paired falsifier has not run yet.
- PCA/updown has shown promising islands, but broad acceptance is fragile.
- External memory improves factual grounding in `cogniformerus`, but the speed
  coupling to CogniQwen surrogates remains a hypothesis.

Operational rule: do not promote this as a product or paper claim until the
paired falsifier shows lower rank or higher acceptance on memory-grounded prompts
with exact verifier parity.

## First Local Smoke: 2026-05-28

A first narrow 9B smoke was run with Qwen3.5-9B Q4_K_M, `tokens=32`,
`calib-tokens=16`, `rank=16`, `basis=pca`, `pca-iters=2`, and one recurrent
layer island (`24`). This is not a promotion gate; it only checks whether the
paired harness can produce falsifiable rows without a long run.

Prompt fixture: `examples/qwen_memory_grounded_pairs.txt`.
Summary helper: `scripts/summarize_memory_grounded_pairs.py`.

Feature-only residual run:

```text
/tmp/qwen_memory_grounded_pairs_features_20260528193306.log
```

Closed-book versus memory-grounded residual deltas:

```text
code_crystal:       residual_mean 0.4173 -> 0.3955, delta -0.0218
science_turing:     residual_mean 0.4040 -> 0.4188, delta +0.0148
literature_hamlet:  residual_mean 0.4222 -> 0.4333, delta +0.0111
```

Low-rank logit drift run:

```text
/tmp/qwen_memory_grounded_pairs_logit_20260528193338.log
```

Top1/top5 result:

```text
code_crystal:       top1 100.00% -> 100.00%, top5 100.00% -> 100.00%
science_turing:     top1 100.00% ->  93.75%, top5 100.00% -> 100.00%
literature_hamlet:  top1 100.00% -> 100.00%, top5 100.00% -> 100.00%
```

Interpretation:

- The harness works and can already falsify over-broad claims.
- Memory-grounding improved the cheap residual proxy for the code prompt, but
  worsened it for the science and literature prompts in this tiny setup.
- Top5 stayed intact for all three pairs, but science lost one top1 position.
- The current evidence says typed memory may help selected regimes; it is not a
  universal automatic surrogate-speed win.

Next gate: run more layers/ranks and a true self-spec acceptance row, but only in
smaller slices or with prompt subsets. A prior six-layer `gen=4` all-pair run was
stopped after several minutes because it exceeded the intended interactive probe
budget before printing rows.
