# Qwen 3.5/3.8 Engine API Frontier SDD

Document status: first native engine slice admitted with bounded attribution;
consumer migration pending
Current frontier: stable backend-neutral request/result and lifecycle contract,
a single-resident native CPU/Metal runtime, a Metal-first product generator,
typed Qwen 3.8 reasoning-effort routing for embedded text generation, and an
advanced experimental CUDA full-model mixed-stack/semantic-loop probe without
an admitted engine adapter
Bounded context: reusable Qwen 3.5/3.8 inference consumed by `cogni-ml` CLIs and
resident services such as Cogniformerus `cfmodeld`

## Problem Card

- Signal: downstream consumers duplicate tokenizer, state, greedy-generation,
  and constrained-label logic and have already drifted from the current
  low-level API.
- Why now: Cogniformerus calls the removed `forward_top2_allowed` helper and its
  `-Duse_qwen35_controller` build is red against current `cogni-ml`.
- Scope: define and admit the smallest stable high-level generation and
  constrained-classification contract, then migrate existing consumers.
- Not merely: renaming `Qwen35CPU`, moving the product CLI wholesale, or
  presenting experimental CUDA probes as feature-equivalent.
- Improvement probe: the high-level contract compiles in CPU-only and native
  Metal builds, reports the actual execution backend, and restores the
  Cogniformerus Qwen build without direct `Qwen35CPU` calls in migrated paths.
- Unknowns: the smallest extraction seam from `qwen35_generate`, runtime-safe
  CPU forcing in a Metal build, and the remaining CUDA product-parity gaps.
- Freshness: current `cogni-ml` and Cogniformerus checkouts inspected before
  implementation.
- Safe next move: migrate Cogniformerus consumers while preserving their
  existing semantic guards and fail-open policies.
- Validation boundary: local CPU/Metal evidence cannot promote CUDA behavior;
  CUDA admission requires an NVIDIA-host falsifier.

## Context Bridge

- `Qwen35CPU` in the current Metal build is both a CPU correctness oracle and a
  façade that may route work to `Qwen35Metal`.
- `Qwen35Metal` is the primary complete native generation backend.
- `ML::CUDA::QwenMixedStackRunner` can be assembled for all layers with
  resident recurrent/KV state, an output head, and an embedding-driven semantic
  greedy loop. That loop is still probe-local, raw-token oriented, and partly
  `--perf-only`; it is not yet the same product-facing contract as
  `qwen35_generate`.
- A future `Qwen35Engine` means the stable request/result and lifecycle
  boundary. It must not imply backend feature parity that has not been
  falsifier-tested.
- OpenAI `reasoning.effort` and Qwen 3.8 template `reasoning_effort` are
  different bounded contexts. The embedded engine admits only the intersection
  it can render and verify exactly: `none`, `low`, `medium`, and `xhigh`.
  `none` is an engine-level compatibility mode that maps to
  `enable_thinking=false`; it is not a Qwen 3.8 template effort string.
- Qwen 3.5 templates do not consume `reasoning_effort`. A shared public enum
  therefore does not imply that every loaded model supports non-`none` values.

## Admitted Surface

- Existing low-level CPU reference, Metal routing, weights, tokenizer, chat
  rendering, constraints, cache, and serving-route APIs remain admitted within
  their existing documented boundaries.
- Existing CUDA runners and probes remain admitted as experimental,
  explicitly selected execution paths.
- The first engine slice may provide:
  - model/tokenizer ownership;
  - bounded greedy text generation;
  - single-step constrained next-token label scoring over caller-supplied
    labels, without claiming semantic policy ownership;
  - per-operation preflight plus explicit requested backend, planned or
    observed attribution, component envelope, and model identity in every
    result;
  - explicit idempotent lifecycle closure;
  - fail-closed validation for empty labels, duplicate token IDs, invalid
    sequence bounds, and unsupported backend requirements.
- The embedded text runtime may provide typed `none`, `low`, `medium`, and
  `xhigh` effort routing when the loaded tokenizer template contains the exact
  Qwen 3.8 low/xhigh instruction contract. The default remains `none` so
  existing callers retain no-thinking generation semantics.
- Generation results report the effective effort so a runtime cannot silently
  ignore or rewrite the request.

## Rejected Surface

- Silent fallback after per-operation preflight, or after the caller explicitly
  requires Metal or CUDA.
- Claiming that `Qwen35CPU` calls prove CPU execution in a Metal-capable build.
- Claiming CUDA product parity from macOS builds or from layer-only probes.
- Exposing `Qwen35CPU::State`, Metal buffers, CUDA buffers, or private decode
  helpers in the stable engine contract.
- Treating unconstrained free text as an authoritative policy decision.
- Automatically promoting speculative, cache/replay, MTP, or tool-call routes
  through the new façade without their existing exactness guards.
- Passing arbitrary effort strings, silently mapping `high`, `max`, or
  `minimal`, or enabling non-`none` effort on a template without the verified
  Qwen 3.8 contract.
- Treating reasoning effort as a hard reasoning-token budget or as a guaranteed
  quality/latency level.

## Guard-Only Future

- Full `qwen35_generate` decode-policy parity behind the engine contract.
- Prompt/session cache ownership and exact continuation through the engine.
- Structured tool-call generation and host-normalized tool responses.
- A CUDA engine adapter with tokenizer, repeated full-model decode, bounded
  generation, constrained classification, cancellation, and parity evidence.
- Sampling and streaming.

## Design Laws

- Consumers depend on a stable engine contract, never a private backend method.
- Backend selection and actual execution identity are different claims.
- A planned hybrid route is not evidence that Metal executed; strict backend
  requirements are admitted only with observed attribution.
- Required backend mismatch is an error, not an implicit fallback.
- Backend-specific state remains owned behind the engine boundary.
- CPU is the correctness oracle; Metal and CUDA promotion require matched
  token/logit or output parity within their declared scope.
- Label-scoring inputs must map to unique single tokens inside the native
  runtime before inference; the pure request contract can validate only names
  and label text.
- Cache/replay and speculative routes retain their original exactness
  certificates; the façade cannot weaken them.
- Effort changes must be represented in the rendered token sequence before
  cache lookup. Exact token-prefix identity, not the enum value alone, remains
  the cache certificate.
- Low and xhigh instructions are copied byte-for-byte from the loaded Qwen 3.8
  template contract; medium adds no system instruction but enables the open
  thinking suffix. Unsupported templates fail before state creation.
- Model load, request execution, and result reporting expose enough identity to
  detect source/build/runtime drift without upgrading route availability into
  execution telemetry.

## Execution Order

1. Add a red public-contract spec independent of model files.
2. Implement request/result validation, per-operation preflight, backend route
   identity, and lifecycle closure.
3. Add the CPU/Metal-routed resident implementation and model-backed focused
   probes.
4. Refactor `qwen35_generate` only through behavior-preserving slices.
5. Qualify CUDA independently; keep unsupported behavior rejected.
6. Migrate Cogniformerus generation and constrained classifiers.
7. Add bounded hook classification only after the daemon path has deadlines and
   an explicit failure policy.

## Falsifier Roster

- Public contract rejects empty prompts, non-positive generation limits, empty
  labels, duplicate labels, route drift, use-after-close, and required-backend
  mismatch before runtime mutation.
- A fake backend proves preflight/request/result forwarding, planned-to-observed
  route refinement, lifecycle closure, and attributed backend reporting
  without requiring a GGUF model.
- Pure chat-render tests prove exact no-thinking, low, medium, and xhigh system
  prefixes/suffixes, including ordering before tool instructions.
- A fake backend rejects result effort drift, and the native runtime rejects
  non-`none` effort when the tokenizer template lacks the exact Qwen 3.8
  capability markers.
- A real Qwen 3.8 model smoke compares the embedded effort prompt/token path to
  the tokenizer template-derived contract and exercises effort-specific cold
  prefix reuse without weakening exact-prefix validation.
- The native runtime rejects multi-token or duplicate-token label mappings
  before state creation.
- CPU-only build compiles and runs the contract and existing Qwen unit specs.
- Native macOS build compiles `qwen35_generate` and the engine consumer.
- Metal model smoke compares engine greedy output with the current low-level
  route for the same model, prompt, and token limit.
- Cogniformerus `-Duse_qwen35_controller` specs compile without any migrated
  consumer calling `forward_top2_allowed`.
- NVIDIA-host CUDA build and model smoke are required before CUDA moves from
  guard-only to admitted product backend.
- Source inventory rejects direct private-backend calls in migrated
  Cogniformerus components.

## LTP/WBA Card

- Window or trigger: exact prompt/session reuse or proposal-assisted decode is
  requested through the future engine.
- Transport corridor: request identity -> tokenizer/model identity -> validated
  cached or proposal span -> backend state transition -> emitted result.
- Legal move: reuse or propose only through the existing exact cache,
  serving-route, and target-verification certificates.
- Boundary safety: model/tokenizer/config/source identity and continuation state
  requirements remain unchanged.
- Lexicographic potential: `(semantic mismatch, state corruption risk,
  unverified work, wall time, memory pressure)`.
- Recompute safety: after the route commits, emitted tokens and continuation
  state must match the target engine at the same boundary; a local latency win
  cannot worsen an earlier component.
- Dual frame: on certificate miss, use ordinary target generation instead of
  trusting the accelerated path.
- Local certificate: existing exact-span hashes, state replay validation,
  proposal acceptance checks, and route attribution.

## Stop Rules

- Stop widening the façade if the current CLI and engine disagree on output for
  the same deterministic request.
- Stop backend promotion on silent fallback, unknown actual-backend identity,
  state aliasing, non-finite logits, token mismatch, or cache certificate
  bypass.
- Stop CUDA product claims until an NVIDIA-host gate passes.
- Do not modify or commit unrelated Cogniformerus work from its current dirty
  tree.

## Implementation Seals

- Slice: engine contract and validation
  - Status: verified
  - Source/spec: `src/ml/gguf/qwen35_engine_contract.cr`,
    `spec/qwen35_engine_contract_spec.cr`
  - Boundary: no model load and no backend implementation required
- Slice: CPU/Metal resident greedy and constrained classification
  - Status: admitted for CPU and planned native hybrid routing, with asymmetric
    evidence; observed Metal remains guard-only
  - Source/spec: `src/ml/gguf/qwen35_native_runtime.cr`,
    `spec/qwen35_native_runtime_spec.cr`,
    `spec/qwen35_weights_lifecycle_spec.cr`
  - Boundary: deterministic greedy and unique single-token labels, one live
    native runtime, explicit close, and no arbitrary concurrent low-level mmap
    registration/replacement
  - Evidence: native Apple Metal model smoke matches the previous low-level
    greedy token and constrained logits and reports a planned Metal+CPU
    envelope rather than unobserved Metal execution; CPU-only contract and
    matched unit suite pass, while the full model-backed Engine parity smoke
    remains weaker than Metal because the long quiet-host run was not completed
  - Nonclaim: strict required-Metal execution is unavailable until backend
    telemetry can return an observed result
- Slice: CUDA engine adapter
  - Status: guard-only
  - Boundary: requires independent NVIDIA-host build, parity, lifecycle, and
    cancellation evidence
- Slice: Cogniformerus migration
  - Status: ready, with no-commit preservation required for the dirty and
    partly untracked consumer tree
