# Qwen 3.5/3.8 Engine API Frontier SDD

Document status: first native engine slice admitted with bounded attribution;
consumer migration pending
Current frontier: stable backend-neutral request/result and lifecycle contract,
a single-resident native CPU/Metal runtime, a Metal-first product generator,
typed Qwen 3.8 reasoning-effort routing for embedded text generation, and an
advanced experimental CUDA full-model mixed-stack/semantic-loop probe without
an admitted engine adapter. Cross-process Metal single-flight and killable
command-buffer waits are an admitted safety slice; in-process recovery after a
GPU timeout remains rejected because Metal exposes no command-buffer cancel.
Nonadaptive prefix/tail Flash-prefill is an explicit operator-tested experiment;
automatic continuation admission and full-model quality remain guard-only.
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
- Product Metal generation and label scoring acquire one cross-process lease
  before creating inference state. Direct-output cache hits remain outside the
  lease because they do not initialize Metal.
- Every synchronous Metal command-buffer wait has a 120-second watchdog by
  default. Timeout terminates the process with status 124; callers may override
  the deadline or explicitly disable it through the documented environment
  control. The setting is captured on the process's first Metal wait.

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
- Returning normally after a Metal command timeout: the submitted command may
  still access unretained buffers, so the inference process must terminate.

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
- The Metal lease is released by process exit and is reentrant only inside the
  owning process. It does not serialize CPU compilation or non-GPU cache hits.
- The lease serializes guarded active inference and guarded model loading; it is
  not a host-wide quota for model mappings kept resident by unguarded processes.
  Direct low-level Metal probes still require process/RSS supervision.
- A submitted Metal command may outlive a host-side deadline. Timeout handling
  therefore exits without Crystal or Objective-C unwinding; it never raises a
  recoverable exception through live command-buffer resources.

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
- A cross-process lock holder rejects a second Metal request before model load,
  process exit releases the lease, and nested same-process acquisition remains
  valid until its final holder closes.
- An empty native Metal command buffer completes through the bounded native
  wait. An uncommitted-buffer child probe with a short deadline must exit 124,
  proving the watchdog cannot unwind as a normal error.
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
- Stop GPU safety promotion if a second process can enter inference while the
  lease is held, if process exit strands the lease, or if timeout handling
  unwinds through live Metal buffers instead of terminating the process.
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
- Slice: cross-process Metal inference safety
  - Status: verified for guarded product routes on Apple M2 Max
  - Source/spec: `src/ml/metal/process_lease.cr`,
    `src/ml/metal/bridge.mm`, `spec/metal_process_lease_spec.cr`, and
    `spec/metal_device_resource_spec.cr`
  - Evidence: lease contention/reentrancy 2/2, native runtime 15/15 with one
    optional model-backed case pending, live Metal resource/wait 3/3, and an
    uncommitted-buffer probe exits 124 under a 25 ms watchdog. A guarded
    Qwen 3.8 27B Q4_K_M one-token smoke completed under RSS/free-memory guards.
  - Boundary: guarded active inference and model loading only; command timeout
    is a process-termination guard, not recoverable cancellation
  - Nonclaim: no host-wide quota over resident mappings or unguarded low-level
    Metal probes
- Slice: CUDA engine adapter
  - Status: guard-only
  - Boundary: requires independent NVIDIA-host build, parity, lifecycle, and
    cancellation evidence
- Slice: Cogniformerus migration
  - Status: ready, with no-commit preservation required for the dirty and
    partly untracked consumer tree

## Experimental prefix Flash-prefill

- Status: operator-tested, explicit-opt-in only; automatic admission unchanged.
- Scope: Apple M2 Max, nonadaptive F16 KV, D256, four KV heads and 16/24
  query heads; `QWEN35_PREFILL_ATTN_FLASH_D256=1`, 1..2048 appended tokens,
  nonnegative prefix, at most 8192 total visible tokens. `0` is rollback.
- Change: reuse the existing 64-key MMA path over an absolute causal prefix;
  handle incomplete query tiles without external reads/writes and the final
  key tail without reading padding or expanding the 16 KiB shared workspace.
- Risk/guard: an offset mask or tail error can silently corrupt continuation.
  Require fail-closed synthetic comparisons against row attention and an
  independent CPU oracle, including boundary shapes and output canaries;
  retain the old automatic policy and row-attention route for policy-rejected
  shapes. This is dispatch selection, not recovery: pipeline/command failures
  still propagate through the existing failure path without retrying attention.
- DoD: focused Flash admission spec passes; the guarded no-model
  `qwen35_attn_flash_d256_prefix_probe` exits zero on correctness cases.
  Operator ABBA timing is a separate measurement, not an engine-speed claim.
- Nonclaims: no adaptive QBit support, automatic prefix promotion, full-model
  continuation certificate, or certified LTP/WBA transformation in this slice.
- Evidence (2026-09-05, Apple M2 Max): admission red test reproduced the old
  nonzero-prefix rejection; the updated admission/cooldown specs passed 3/3.
  The no-model probe passed 32 shape/GQA cases, including 8191/8192-token
  boundaries, and 12 CPU-oracle comparisons per GPU path. Maximum Flash-vs-SG4
  error was `1.2e-7`; poisoned input guards and output sentinels passed. The
  same probe with the parent shader failed as expected on the first one-token
  case (4096 unwritten output values), so a no-op kernel cannot pass.
- Aligned-path regression control: parent/new shaders in one process gave
  bit-identical outputs for GQA4/GQA6 at T1024/2048, 16 timed samples per path.
  Old/new p50 ratios were `0.9721..1.0199`; a GQA4 timing outlier prevents a
  strong statistical no-regression claim. No automatic policy was expanded.
- Prefix operator timing: two bounded ABBA runs (four blocks, eight samples per
  path) used GQA6, P1024/4096, T256/512 and identical synthetic inputs. The
  second run's SG4/Flash p50 milliseconds were `11.4242/1.7599`,
  `22.8012/3.8222`, `60.4690/5.4812`, `102.1703/10.9995` respectively. Across
  both runs the operator ratios ranged `5.97..11.08x`; host drift was visible.
  These are completed-dispatch wall times, not GPU intervals, model pp/tg,
  llama.cpp comparisons, or token/ECS quality evidence.
- Comparator boundary: partial SG4 queries are padded to four rows to avoid
  relying on its partial-threadgroup barrier behavior; only the requested
  rows are compared. Flash receives the actual unpadded token count. The CPU
  oracle uses exactly representable H16 fixture Q/K/V, not arbitrary model
  activations. Guard poisoning is not a general GPU memory sanitizer.
- Source review: correlated Luna review found no causal-mask, synchronization
  or OOB blocker within the host allocation contract. The kernel cannot inspect
  actual buffer lengths; the outer `start_pos + n_tokens <= max_seq` check
  remains mandatory. Its documentation objection was resolved by explicitly
  distinguishing policy fallback from unavailable GPU-error recovery.
- Safety: every GPU run used `scripts/run_safe.sh`, 180 seconds, a 4096 MiB
  process-tree cap and the 35% free-memory floor. Quiet-host waiting was
  disabled under standing user authorization; no foreign process was stopped.
- Refresh after shader, host dispatch, cache representation, compiler/device,
  or fixture changes. Before automatic promotion, require a fenced full-model
  prefix append with state/continuation, top-2 and token-ECS quality checks,
  plus stable same-process whole-prefill timing.

### Full-model prefix falsifier

The full-model state-value gate is **red**, while the bounded eight-token
continuation gate passes (2026-09-05, engine `f4a29e1b`, Qwen3.8-27B Q4_K_M,
Apple M2 Max). This does not establish corrupt state or a semantic regression:
matching every downstream cache value is stronger than matching model output.
Do not relax the state tolerance from these observations or promote automatic
prefix admission. No production engine/kernel changed in this test slice.

`bin/qwen35_flash_prefix_model_probe.cr` builds the same raw code-token prefix
with Flash disabled in two independent states, then compares append policy
`0` versus `1`. It uses full-width `prefill_tokens_last_hidden` plus the full
GPU logit head, with device fences. The nonzero-prefix `prefill_tokens_logits`
API instead processes T-1 prefill rows plus a terminal decode, so it would not
measure the declared T-row Flash shape. Actual route markers must be 0 versus
16. The probe checks live F16 KV owners and all recurrent/conv state buffers;
caller-maintained position fields are not treated as live-length certificates.

State/logit budgets were fixed before model execution: each state value must satisfy
`abs(delta) <= 0.02 + 0.01*abs(reference)`, with no nonfinite values; each full
logit vector must have max absolute difference <=0.1 and cosine >=0.9999.
Baseline top1 must be covered by candidate top2; token ECS must be >=0.99
(made explicit before the final control/candidate rerun; no threshold relaxed);
fresh free continuation must match baseline IDs. Ranked top2 is diagnostic.
Token ECS uses `token_embd.weight`, not output logits. Identical token IDs have
ECS=1 by construction; this is not an independent semantic-quality score.
State and continuation verdicts are emitted separately, then combined.

- P256/T65, repeated: all 8 top1 IDs and 16 ranked top2 entries matched,
  minimum full-logit cosine `0.9999999029`, maximum logit error `0.0039978`,
  ECS=1. Fresh greedy text matched: ` seen = set()\n     result =`.
  After append, K/V had 5,639/7,147 values outside the state budget;
  max errors `0.9921875/1.4296875`, RMSE `0.00289566/0.00374892`.
  Conv/SSM stayed within budget. First above-budget K appeared at layer31;
  later outliers were not restricted to the final query row.
- P256/T64 with shape warmup: continuation/top2/ECS still matched, but K/V
  and two SSM values exceeded the budget after teacher continuation.
  Max K/V error `2.640625/3.5625`; max SSM error `0.0415637`.
  Thus a partial query/key tail is not necessary for the observed discrepancy.
- P256/T65 Flash-off A/A control: all compared state values/logits had zero
  numerical difference. The same-path un-warmed AB timing nevertheless showed
  an apparent `1.154x` ratio. Candidate un-warmed ratios varied `1.184..1.577x`;
  the warmed aligned sample was `1.200x`. These are diagnostic wall times,
  not a defensible speedup, pp/tg measurement, or llama.cpp comparison.
- Final probe qualification: no-model self-test and warmed P256/T65 A/A
  control passed; the warmed control still showed an apparent `1.131x` ratio.
  Its full-logit max error and every state max error were zero.
  The warmed Flash candidate reproduced the same state outliers and passed
  teacher/free continuation, exiting1 as intended; its diagnostic ratio was
  `1.162x`. Final logs: `/private/tmp/qwen_flash_model_final_control.log` and
  `/private/tmp/qwen_flash_model_final_candidate.log`.

Partial SG4 groups explicitly use row attention in both arms; aligned T64
uses the default SG4 comparator. The raw fixed-length fixture is not a held-out
coding test, ignores EOS stopping and only generates eight tokens. No long
session or semantic task certificate follows. Flash rounds Q to half at
`qwen35_attn_flash_d256.metal:77` and changes reduction order; amplification
through later layers is a hypothesis, not an established cause. Next useful
falsifier: replay a real layer's identical Q/K/V through both operators and a
row control with Q explicitly rounded to half, before changing kernels.

Model runs are sequential through `scripts/run_safe.sh`, 600 seconds,
24,576 MiB process-tree cap and 35% free-memory floor; quiet waiting disabled
under standing authority. No foreign process is stopped. Temporary evidence:
`/private/tmp/qwen_flash_model_p256_t65_trace.log`,
`/private/tmp/qwen_flash_model_p256_t65_control.log`,
`/private/tmp/qwen_flash_model_p256_t64.log`. Refresh after source, compiler,
model, device, fixture or comparator changes. A green comparator qualification
does not turn the red model state gate green.

Build and reproduce (control expects exit0; the candidate state falsifier
currently expects exit1, with continuation passing):

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen_flash_model_build crystal build \
  bin/qwen35_flash_prefix_model_probe.cr --release \
  -o /private/tmp/qwen35_flash_prefix_model_probe \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
/private/tmp/qwen35_flash_prefix_model_probe --self-test
COGNI_RUN_SAFE_REQUIRE_QUIET=0 COGNI_RUN_SAFE_WAIT_QUIET_SEC=0 \
  COGNI_RUN_SAFE_MIN_FREE_PCT=35 scripts/run_safe.sh \
  /private/tmp/qwen35_flash_prefix_model_probe 600 24576 \
  --prefix 256 --append 65 --gen 8 --warmup --control
# Repeat the guarded command without --control; use --append 64 for alignment.
```

Reproduce the bounded operator gate from the repository root:

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen_flash_prefix_build crystal build \
  bin/qwen35_attn_flash_d256_prefix_probe.cr --release \
  -o /private/tmp/qwen35_attn_flash_d256_prefix_probe \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
COGNI_RUN_SAFE_REQUIRE_QUIET=0 COGNI_RUN_SAFE_WAIT_QUIET_SEC=0 \
  COGNI_RUN_SAFE_MIN_FREE_PCT=35 scripts/run_safe.sh \
  /private/tmp/qwen35_attn_flash_d256_prefix_probe 180 4096 --perf
CRYSTAL_CACHE_DIR=/private/tmp/qwen_flash_prefix_spec crystal spec \
  spec/qwen35_forward_spec.cr:201 spec/qwen35_forward_spec.cr:282 \
  spec/qwen35_forward_spec.cr:364 \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
```
