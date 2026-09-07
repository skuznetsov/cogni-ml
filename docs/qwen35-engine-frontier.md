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
Nonadaptive prefix/tail Flash-prefill is an explicit operator-tested experiment
with bounded coding-smoke evidence; automatic continuation admission and
general full-model quality remain guard-only. Tile-local causal block skipping
is operator-verified with unchanged coding-smoke outputs; its whole-model
speed gate remains open: completed guarded repeats do not establish a gain.
Existing Q6 FFN down/add fusion remains opt-in: the bounded P256/T512 append
comparison preserves values but does not establish a speedup.
Q4 SG8-B128 single-buffer staging is measured-red in the isolated FFN pair
falsifier; production staging and all dispatch defaults remain unchanged.
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

### Real-Q rounding discriminator

`bin/qwen35_flash_real_q_replay_probe.cr` isolates the last attention layer63
on the same fixed-token P256/T64 or T65 fixture. It does not modify the engine:
after synchronized Flash-off prefill it reads existing scratch slots (missing
slots fail, never allocate) and layer63 F16 K/V. Ordinary row replay must
exactly reproduce all stored attention output values before attribution is
allowed. SG4 is explicitly off for **both** shapes, unlike the earlier aligned
full-model SG4 comparator. The probe is single-process/single-flight only.

Four replays hold gate/K/V fixed: row(Q), row(float(half(Q))), Flash(Q), and
Flash(float(half(Q))). The last pair must match exactly. A separately written
Float64 CPU oracle checks 12 complete query/head rows (3,072 values) for each
of ordinary row, rounded row and Flash-with-rounded-Q semantics. Its budget
is `0.001 + 0.0001*abs(reference)`; this is an operator check, not the earlier
model-state tolerance. Output guards, seeded comparator perturbation and
nonfinite rejection are also checked.

Observed on engine `5d33af99`, Qwen3.8-27B Q4_K_M, M2 Max (2026-09-05):

| Appended rows | Row / Flash RMSE | Rounded-row / Flash RMSE | Residual fraction |
| --- | --- | --- | --- |
| 65 | 7.85448e-5 | 6.40833e-7 | 0.008159 |
| 64 | 7.64759e-5 | 6.38485e-7 | 0.008349 |

Both captures reproduced the executed output with zero numerical difference
(399,360 and 393,216 values); both Flash round-idempotence controls also had
zero difference. CPU oracle maximum errors were at most `7.87e-6` across
the sampled comparisons. After matching Q precision, residual RMSE is about
120–123 times smaller. This supports **local Q-rounding dominance**, not a
claim that 99% of the whole model's state error is explained or fixed. Layer63
here receives baseline inputs; it does not replay the divergent Flash history.
No performance, coding-quality or automatic-admission claim follows from this
local rounding discriminator alone.
Correlated Luna source review returned ROBUST for capture in this corridor:
the final full-attention layer completes without an arena, and later FFN work
uses different scratch tags. Scratch itself has no layer/epoch identity, so
the static last-layer route plus exact output replay are both required.

Reproduction: build the replay probe using the full-model build command above,
substituting its source/output name; run `--self-test`, then the same guarded
600s/24576MiB command with `--append 65` and separately `--append 64` (no other
shape/generation flags). Keep the 35% free-memory floor and no quiet waiting.
Temporary logs: `/private/tmp/qwen_real_q_replay_t65.log` and
`/private/tmp/qwen_real_q_replay_t64.log`. Current next discriminator (user
correction): measure the existing Flash path on longer coding continuations
with external executable tests, top2/ECS diagnostics, and warmed order-balanced
whole-append timing with A/A controls. Internal state tolerance is not an
established semantic-quality boundary. Defer a Q-precision correction until
task-quality loss justifies its cost; retain the state diagnostic unchanged.
Automatic admission remains off. Three small author-created fixtures are
smoke tests, not held-out benchmarks or evidence of general coding ability.
Refresh after scratch routing/lifetimes, shader, compiler, model or fixture
changes. A scratch-layout change must invalidate capture rather than silently
switch to synthetic inputs. This is ordinary numerical analysis, not LTP/WBA.

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

### Flash coding-value check (2026-09-05)

This slice supersedes the immediate Q-precision-correction plan in LM1049.
The user objective is useful coding output at useful latency, not minimum
internal state error. No engine/shader/policy or numerical tolerance changed.
`bin/qwen35_flash_prefix_model_probe.cr --prompt-file` renders an untruncated
no-thinking chat prompt, derives its actual append length, and stops baseline
and fresh-candidate greedy continuations independently at EOS. The baseline
always consumes its own argmax; only the comparison candidate is teacher-forced.
Top2 denominators and final state coverage use the actual continuation length.
Prompt bytes, actual token counts and generation remain bounded before model
weight allocation. Raw fixed-token timing fixtures retain their old behavior.

`scripts/qwen_flash_coding_score.py` reuses the QBit scorer's Crystal extraction
and copied-project test runner. It rejects missing/duplicate config/summary,
non-chat fixtures, missing EOS and malformed records. A/A controls do not
promote Flash. External baseline failure is `invalid_flash_baseline`, never a
Flash regression or pass. Internal state and teacher metrics remain separate
diagnostics. The runner is **not a sandbox**: inspect generated source and the
fixture before execution. The three actual outputs were inspected as pure
array/set/search code before running them.

These are three small author-created smoke fixtures, not held-out tasks or a
general coding benchmark. Conditions: engine `651cb25e`, unchanged Flash
kernel, Qwen3.8-27B Q4_K_M, Apple M2 Max, common Flash-off prefix64,
`--gen 256 --warmup`, group limit1/cooldown50ms. Actual append lengths below
come from chat tokenization, not prompt-file bytes. Lower-bound used reverse
timing order. Fresh candidate generation and external tests ran for each task.

| Fixture | Append tokens | EOS tokens, each path | Top1 match | Ranked top2 match | External specs, each path |
| --- | ---: | ---: | ---: | ---: | --- |
| stable_unique | 73 | 88 | 88/88 | 175/176 | 2 examples, 0 failures |
| lower_bound | 94 | 102 | 102/102 | 204/204 | 2 examples, 0 failures |
| merge_ranges | 104 | 195 | 195/195 | 390/390 | 2 examples, 2 failures |

All three complete greedy outputs, including EOS, are identical across paths:
385/385 top1 tokens and 769/770 ranked top2 positions match. Token-embedding
ECS is 1 throughout, which is tautological for identical IDs. Both merge-ranges
outputs incorrectly merge adjacent intervals (`start <= current_end + 1`),
contrary to the prompt. This is a concrete counterexample to treating ECS or
baseline agreement as semantic correctness. No Flash-specific quality loss
was observed here; only two tasks have valid successful baselines.

All three teacher diagnostics pass, while the unchanged state tolerance still
fails. Minimum logit cosine is 0.9999950 across these runs; maximum absolute
logit difference is 0.070064. Their probe exit1 is the retained state gate,
not an external-test result. Do not turn it green by loosening the tolerance
or describe it as a proven semantic-quality loss. No automatic admission.

Temporary evidence: `/private/tmp/qwen_flash_value_{unique,lower,merge}.log`
and `/private/tmp/qwen_flash_scored_{unique,lower,merge}/report.json`.
First two model runs used the initial probe build; the final build additionally
avoids constructing discarded raw fixtures for chat and checks file size before
read. Inference/metric logic was unchanged. The final binary SHA256 is
`9e59ab8723b300e72e7ee843e86e7f29d0143133dc7d2abdc174e0f6cacbb892`;
the later baseline-greedy source comment is behavior-neutral.

Reproduction after the release build above (substitute each fixture name):

```sh
/private/tmp/qwen35_flash_prefix_model_probe --self-test
COGNI_RUN_SAFE_REQUIRE_QUIET=0 COGNI_RUN_SAFE_WAIT_QUIET_SEC=0 \
  COGNI_RUN_SAFE_MIN_FREE_PCT=35 scripts/run_safe.sh \
  /private/tmp/qwen35_flash_prefix_model_probe 600 24576 \
  --prefix 64 --gen 256 --warmup \
  --prompt-file spec/fixtures/qwen_flash_coding/stable_unique/prompt.txt
# Save the log, inspect its generated source, then use a fresh output directory:
python3 scripts/qwen_flash_coding_score.py \
  --probe-log /private/tmp/qwen_flash_value_unique.log \
  --project spec/fixtures/qwen_flash_coding/stable_unique \
  --hidden-spec spec/fixtures/qwen_flash_coding/stable_unique/check.cr \
  --source src/answer.cr --output /private/tmp/qwen_flash_scored_unique_new \
  --timeout 45
python3 -m unittest spec/qwen_flash_coding_score_spec.py
```

The scorer's seven tests cover parser rejection, control/baseline/regression
classification and actual execution of all three checked-in fixtures against
both known-correct and deliberately incorrect implementations. Incorrect seeds
must fail assertions rather than merely fail compilation. Fixtures include
deterministic randomized cases and no-input-mutation checks.

Warmed timing follow-up used four fresh sequential processes with raw P256/T512,
`--gen 2 --warmup`, the same release binary and pinned group1/cooldown50ms.
A and B denote row/SG4 and Flash respectively; A/A keeps Flash off in both
states. State allocation and common prefix are outside append timing; the
measured interval includes full-width appended prefill, terminal hidden
readback, full GPU logits, synchronization, and any policy cooldown inside
that work. It is not cold start, product pp/tg, adaptive-QBit timing, or a
llama.cpp comparison. Warmup covers both paths. No own compilation was run
during this four-process batch; the shared host was not required to be idle.

| Process / execution order | Baseline append ms | Candidate append ms | Baseline/candidate |
| --- | ---: | ---: | ---: |
| Same-path A/A | 4117.257 | 3901.633 | 1.0553 |
| Flash A/B | 4534.913 | 3781.700 | 1.1992 |
| Flash B/A | 3955.110 | 3960.293 | 0.9987 |
| Same-path A/A, reversed labels | 3933.984 | 4130.723 | 0.9524 |

The two Flash comparisons together form one order-balanced ABBA sample:
mean baseline4245.011ms versus Flash3870.996ms (8.81% shorter time, 1.0966x
ratio). Balanced A/A means differ by 0.24%, but individual A/A rows vary by
about 5%. Flash's reverse comparison is a tie. Thus the balanced mean is a
promising observation, **not a stable speed certificate**; repeated balanced
batches are still needed. The earlier lower-bound coding run also contained
an 8653ms candidate outlier versus1596ms baseline, despite about1510ms warmup
and1610ms fresh-candidate prefill. Its cause is unestablished; do not discard
it to promote the faster rows. Coding timings are not used for a speed claim.

Both A/A processes exit0 with zero state/logit difference. Both Flash
processes exit1 solely on the retained state diagnostic, with matching two
greedy tokens, all four ranked top2 positions and ECS1. All processes finish
without runner kill/timeout; retain `scripts/run_safe.sh` 600s/24576MiB and
35% free-memory floor, with quiet waiting off and no foreign process stopped.
Logs: `/private/tmp/qwen_flash_value_pp512_{aa,ab,ba,aa_reverse}.log`.
Reproduce with `--prefix 256 --append 512 --gen 2 --warmup`, adding
`--control`, neither flag, `--reverse`, and `--control --reverse` respectively
to the same guarded invocation. Do not disable cooldown to improve the row.

Next smallest optimization candidate is a per-query-tile causal key bound in
`src/ml/gguf/kernels/qwen35_attn_flash_d256.metal`: currently each tile visits
all full key blocks and masks future scores only after QK. Skipping fully
future blocks could reduce work without introducing extra Q rounding. Keep
the existing global scalar tail, ensure uniform barrier control, and first
test output guards and a separate oracle at 63/64/65 boundaries and long
prefixes. This is a proposal, not an implemented or measured speedup. A second
candidate is avoiding full last-chunk hidden readback when only the terminal
row is needed (`prefill_tokens_last_hidden`); preserve resident ownership and
CPU fallback semantics. Q_hi/Q_lo correction stays deferred until evidence
of useful quality loss warrants its cost. Refresh after shader, routing,
compiler/device, model, prompt or scorer changes. No LTP/WBA claim.

### Causal full-key block bound: operator verified, whole-model speed open

Prospective gate: change only the Flash MMA loop bound, keeping the global
scalar-tail origin, ABI, Q precision, allocations and host admission unchanged.
For bounded finite fixtures, old/new Flash must match bit-for-bit over GQA4/GQA6
query/key boundary and long-prefix cases; the existing Float64 causal oracle
and output canaries remain required. A deliberately rounded-down bound must
fail the comparator. Measure old/new Flash in warmed ABBA microbenchmarks;
only then run guarded full-model coding/EOS and append timing checks. Reject
numerical regression or a slower/no-benefit kernel rather than relax gates.
Risk: dropping the last visible key block, changing scalar-tail ownership, or
divergent barriers. Rollback is the prior shader; no wider Flash admission.

Operator evidence on M2 Max (2026-09-05): the shader now uses the minimum of
global full-key end and the 64-key ceiling of the active query tile's exclusive
causal end. The unchanged scalar tail still begins at the global end. Forty
cases (20 shapes x GQA4/GQA6) match the prior Flash bit-for-bit, including
base0 T1024/2048 and context8192; eight shapes x both GQA ratios also pass the
separately computed Float64 causal oracle. Output padding remains sentinel.
Old/old control passes; a floor-instead-of-ceiling mutant fails at P0/T64,
first output `-0.030496921` versus0. The exact comparator also rejects seeded
perturbation/nonfinite values. No tolerance was changed.

The probe's optional `--flash-reference=PATH` compiles an independent pipeline
from an old shader, rather than comparing the new kernel against itself.
Normal mode retains the SG4 comparator. Timings use one dispatch per completed
command buffer, reporting both fenced host wall and Metal GPU execution
intervals; shader compilation and buffer initialization are outside timing.
Below: deterministic synthetic inputs, warmup1, 12 ABBA blocks, 24 samples per
path, old shader from `2451a4fa`.
These are **attention-kernel** measurements, not full-model pp/tg.

| Prefix / appended rows | Old GPU p50 ms | New GPU p50 ms | Ratio |
| --- | ---: | ---: | ---: |
| 0 / 256 | 0.5802 | 0.3399 | 1.7071x |
| 0 / 512 | 1.2650 | 0.7736 | 1.6352x |
| 0 / 1024 | 5.5835 | 2.9886 | 1.8683x |
| 0 / 2048 | 22.7317 | 11.2613 | 2.0186x |
| 256 / 512 | 2.0039 | 1.3608 | 1.4726x |
| 1024 / 256 | 1.8007 | 1.4815 | 1.2155x |
| 1024 / 512 | 4.2055 | 3.5299 | 1.1914x |
| 4096 / 256 | 6.4513 | 5.8895 | 1.0954x |
| 4096 / 512 | 13.2349 | 12.5915 | 1.0511x |

Base0 same-shader A/A GPU ratios are 0.9743/1.0929/1.0150/1.0173 for
256/512/1024/2048 rows. Earlier host-only prefix samples varied, including a
0.9561x row at P4096/T256; gains at long prefixes are small relative to noise.
Do not extrapolate the approximately 2x T2048 operator result to the trunk or
adaptive-QBit path. The main structural benefit is eliminating nearly half
the full-key-block iterations at base0; a long prefix remains visible and
cannot be skipped. Inputs with nonfinite future V are outside exact old/new
equivalence: removing masked `0*NaN` work may change those outputs, and a
partially future block is still read. No nonfinite-safety claim is made.

Full-model regression on the same three coding prompts preserves all previous
outputs: stable_unique 88, lower_bound 102, merge_ranges 195 tokens including EOS.
All 385 emitted teacher records and all 9 emitted state-diagnostic records are
identical to the prior Flash runs, as are both paths' token IDs and full text.
This compares published diagnostics, not a new full-state byte dump. Ranked
top2 remains 769/770 and ECS 1; identical IDs do not establish task correctness.
The retained state gate still fails, unchanged; teacher diagnostics pass.
Processes exit1 on that diagnostic only, without kill/timeout. Coding timings
contain outliers and are not used as speed evidence.

After inspecting the unchanged generated source, external specs were rerun:
stable_unique and lower_bound each pass 2 specs on both paths; merge_ranges
fails both specs on both paths by merging adjacent intervals. Its scorer
verdict remains `invalid_flash_baseline`, not a Flash success or regression.
Reports: `/private/tmp/qwen_flash_causal_scored_{unique,lower,merge}/report.json`.
Reproduce with the existing scorer command above, substituting these logs and
fresh output directories. These are the same author-created smoke fixtures,
not an independent held-out benchmark.

The subsequent old/new full-model timing attempt at P256/T512, gen2, warmed,
was stopped by the unchanged system free-memory floor: old process 33% after
about 4s; the already-started candidate process 35% after about 4s. Neither
produced an append summary, so there is no complete ABBA or new whole-model
speed result. Stop the heavy batch here, do not retry with weaker guards.
This is separate from the three completed coding processes above. Runner
limits remain 600s/24576MiB for the model and 180s/4096MiB for the operator,
minimum free 35%, quiet waiting off; no foreign process was stopped.

Reproduction / local evidence index (temporary paths can expire):

- Build the operator and model probes using the commands above. The measured
  operator binary SHA256 is
  `188ec568943baecce3b1e7be9731b5fdf1c6fc3835397d518ff41ab53884972e`;
  new model binary
  `4a96010546bb2fb19b4d4c1c0ede4b21f6e89e2dc5df4cfde717c8ebbc653d1a`.
- Save the shader from `git show 2451a4fa:src/ml/gguf/kernels/qwen35_attn_flash_d256.metal`
  as `/private/tmp/qwen35_flash_before_causal_bound.metal`; expected SHA256
  `45394daf276f42a7bb2007c5e5a20fe5060a816745f817213e9ede534895034f`.
  Candidate shader SHA256 is
  `59b7f2723eef7ee35fc6e4b87ec38698b1127c5bd7d2abbd1f94babfc2ba4534`.
- Run the guarded operator with
  `--flash-reference=/private/tmp/qwen35_flash_before_causal_bound.metal --perf --reps=12`,
  then replace `--perf` with `--perf-only` for base0. A/A adds
  `--flash-source=/private/tmp/qwen35_flash_before_causal_bound.metal`.
  Logs `/private/tmp/qwen_flash_causal_final_{prefix,base0,aa}.log` all exit0.
- Negative control: change the old shader's MMA bound to
  `min(full_key_end, (base_pos + iq1 + min(uint(QWEN35_FLASH_Q), n_tokens - iq1)) / QWEN35_FLASH_C * QWEN35_FLASH_C)`
  (floor, not ceiling), and pass it as `--flash-source` with the old reference.
  `/private/tmp/qwen_flash_causal_final_red.log` exits1 at P0/T64 as expected.
  The final default SG4/oracle mode also exits0:
  `/private/tmp/qwen_flash_causal_default.log`. GPU timestamps are required
  even in correctness mode; portability to devices without them is untested.
- Model coding commands retain `--prefix 64 --gen 256 --warmup --prompt-file`
  with the three checked-in fixtures; lower_bound also uses `--reverse`.
  Logs `/private/tmp/qwen_flash_causal_{unique,lower,merge}.log` compare against
  `/private/tmp/qwen_flash_value_{unique,lower,merge}.log` from the prior slice.
- The incomplete full-model timing logs are
  `/private/tmp/qwen_flash_causal_full_{a1,b1}.log`; arguments
  `--prefix 256 --append 512 --gen 2 --warmup`. Old model binary SHA256:
  `9e59ab8723b300e72e7ee843e86e7f29d0143133dc7d2abdc174e0f6cacbb892`.

DoD: both release builds, model-probe no-model self-test, the final operator
gates and `crystal spec spec/qwen35_forward_spec.cr:201` with bridge link flags
pass; the latter is 1 example/0 failures. Crystal format and diff checks pass.
Correlated Luna adversary review finds no causal-end, barrier, global-tail,
pipeline-alias or timer defect in this bounded slice. Verdict: ROBUST for
the tested operator transformation; VULNERABLE for a whole-model speed,
general coding-quality or broader-admission claim. No automatic policy,
adaptive-QBit route or Q precision changed. Refresh after shader, route,
compiler, device, model, fixture or comparator changes. Next signal: repeat
the guarded full-model speed comparison only with adequate memory headroom;
terminal-row-only output remains a separate candidate, not part of this edit.

### Causal-bound full-model timing resumed (2026-09-05 local / September 6 UTC)

Memory preflight now reports 86% free. Rebuilt the expired temporary binaries:
old source from `git archive 2451a4fa src bin shard.yml shard.lock`, candidate
from `d0a9cd08`, Crystal 1.21.0 / LLVM 22.1.8, release mode and the same bridge.
The inference source differs only in the causal-bound shader; the full-model
probe source is byte-identical. No engine or policy changes in this follow-up.
Both no-model self-tests pass. No compilation overlaps model timing.

Same Qwen3.8-27B Q4_K_M / M2 Max, raw P256/T512, gen2, `--warmup`;
timed interval includes full-width append and the fenced full-logit GPU head,
not prefix setup, model loading or compilation. Each process also measures
the unchanged Flash-off row path. Four fresh processes ABBA, then one final
BAAB batch declared before execution to check the first batch's variability:

| Order / version | Flash append ms | Unchanged row append ms |
| --- | ---: | ---: |
| A1 old | 4044.772 | 4458.457 |
| B1 new | 4017.566 | 4476.637 |
| B2 new | 5170.500 | 5296.160 |
| A2 old | 3773.282 | 4119.684 |
| B3 new | 3716.692 | 4095.907 |
| A3 old | 3696.149 | 4093.323 |
| A4 old | 3740.824 | 4086.441 |
| B4 new | 3776.187 | 4114.421 |

ABBA means: old 3909.027ms, new 4594.033ms (+17.52% time); corresponding
unchanged-row means rise 13.93%. BAAB means: old 3718.486ms, new 3746.440ms
(+0.75% time), row means rise 0.37%. All samples retained: old 3813.757ms,
new 4170.236ms (+9.35% time), row means rise 7.31%. **No measured full-model
speedup.** The row variation is a countercheck, not a correction to subtract
or proof that host load caused the slowdown. The second batch is near parity;
neither it nor the noisy first batch identifies a causal regression magnitude.
Do not promote the earlier operator ratio into pp/tg or adaptive-QBit speed.

All eight runs have identical config, emitted teacher/state diagnostic
records, token IDs and text (` seen =`). Each has top1 2/2, ranked top2 4/4,
ECS 1 and a matching free continuation; Flash dispatch counts are exactly 16
when on and 0 when off. This two-token timing fixture is not a coding-quality
test or a fresh EOS certificate. Existing strict state diagnostics remain
red; each process exits1 only on that expected diagnostic, with no runner
kill/timeout. Keep 600s/24576MiB, minimum free 35%, quiet waiting off, sequential
processes and no foreign-process interference. Peak memory was not measured.

Temporary reproduction directory: `/private/tmp/qwen-flash-resume.jHg6ks`.
Logs are `{a1,b1,b2,a2,b3,a3,a4,b4}.log`, binaries `old` and `new`; run the
guarded model command above with `--prefix 256 --append 512 --gen 2 --warmup`
in that order. Old binary SHA256:
`cdb1903faafa245a2c9b29c55f10869d37ef93f9b672c1e4f3e448d8a5efd871`;
new: `8ce8e993758eff3f6a1c60ee7ae30298960401461aa4a91a2d389f9b88374707`.
Bridge SHA256:
`48bb1469e2a473d30a94ab102df91268d549a4dd3710b076a0e59d137691005a`.
Temporary artifacts may expire; source revisions and commands are the rebuild
path. DoD: both builds/self-tests, all eight bounded runs, parser assertions
for unique summaries/no kills/exact route counts/cross-run diagnostics, and
diff checks pass. Adversary verdict ROBUST for this observation; VULNERABLE
for stable whole-model speed or causal attribution. No further timing batch
in this slice. Next useful move is phase-level attribution of the full append
before another optimization; terminal-row-only output remains a candidate,
not an established bottleneck. Refresh on source/toolchain/device/model or
workload changes. No LTP/WBA or automatic-admission claim.

### Full append attribution and profiling command lifetime (2026-09-07)

The next bounded slice adds `--profile off|boundary|detail` to
`bin/qwen35_flash_prefix_model_probe.cr`; default `off` leaves the existing
route controls unchanged. Profiles cover append only, not prefix or decode.
Boundary mode retains shared commands and labels GPU intervals on stderr with
same-stream begin/end markers (the safe runner captures stderr separately).
Detail mode deliberately splits commands, waits per phase, uses CPU embedding
and removes shared-command rotation cooldowns. Its faster wall time is **not
an inference speedup**. Trace times are nested host wall time; phase waits are
host commit/wait, not kernel GPU timestamps. Never add GPU time to host waits.

Before accepting the profile, the first detail process stalled on its fourth
append. A stack sample localized the stall to native command creation, not a
GPU completion wait. Explicitly discarding unused terminal successors did
not resolve it in a bounded retry. A no-model test with GC disabled stalled
at the 65th create/discard command; 80 create/commit/wait commands completed.
This is evidence for command-slot retention, not a model-memory shortage;
native autorelease timing is an explanation, not a measured internal cause.

The fix avoids creating a successor after the final recurrent FFN checkpoint,
both in recurrent-only and full+recurrent detail paths, and after the last
coarse full+recurrent phase. Intermediate checkpoints still allocate their
required successor. The final command has completed before output copy/read.
Non-profiled command scheduling, kernels and numerical policies are unchanged.
`spec/qwen35_prefill_phase_lifecycle_spec.cr` exercises 80 terminal checkpoints
and a continuing-to-terminal pair without loading a model: 2 examples pass.

Scope: Qwen3.8-27B Q4_K_M, M2 Max, Crystal 1.21.0/LLVM 22.1.8,
raw P256/T512, gen2, warmed five-append process. A boundary-mode candidate
measured 4087.050ms hidden-call wall plus 3.898ms head/final fence. Its 16
shared commands summed to 7.388ms host encode, 3211.885ms submit/wait and
3195.532ms GPU intervals. Source policy adds 16 configured 50ms pauses
(800ms inferred from route/count, not independently measured sleep time).
These intervals cover 63 grouped layers; the last standalone full layer and
head are outside the grouped GPU sum.

The completed post-fix detail candidate gives these phase host-wait sums:

| Phase | Calls | Wait ms |
| --- | ---: | ---: |
| FFN up/gate + activation | 63 | 1194.99 |
| FFN down | 63 | 649.62 |
| Recurrent input projections | 48 | 474.46 |
| Recurrent post/O projection | 48 | 189.50 |
| Full-attention QKV | 15 | 123.88 |
| DeltaNet | 48 | 81.33 |
| Full-attention O projection | 15 | 58.36 |
| Recurrent preparation | 48 | 47.12 |
| Flash attention | 15 | 22.69 |

This is a bottleneck hypothesis under diagnostic scheduling, not hardware
counter attribution or a global speed certificate. The existing fused
`q4_h16_b128_sg8_swiglu_h16` route already serves the 63 up/gate calls;
do not propose that fusion as new. FFN GEMM is the next optimization target,
before further attention-only work. Head pruning has little support here.

Off, boundary and completed detail runs have identical emitted state and
teacher records and identical non-timing summaries: top1 2/2, ranked top2
4/4, ECS 1, matching IDs `[3753, 283]` and text ` seen =`. The existing strict
state diagnostic remains red; completed model runs exit1 on that diagnostic,
not a timeout. No general coding, EOS, state-byte or adaptive-QBit claim.

Reproduction: release-build the probe with `build/bridge.o` and
`-framework Metal -framework Foundation -lc++`; run `--self-test`, then
`scripts/run_safe.sh <probe> 180 24576 --model <Qwen3.8-27B-Q4_K_M.gguf>
--prefix 256 --append 512 --gen 2 --warmup --profile detail`. Set
`COGNI_RUN_SAFE_REQUIRE_QUIET=0`, `COGNI_RUN_SAFE_WAIT_QUIET_SEC=0`,
`COGNI_RUN_SAFE_MIN_FREE_PCT=35`, `RUN_SAFE_PASSTHROUGH_STDIO=1`.
Boundary/off controls used 600s originally; final checks use 180s. Runs are
sequential, no concurrent compilation or interference with foreign processes.
Temporary evidence: `/private/tmp/qwen-append-profile.bNoKlm/`, logs `off.log`,
`boundary-final.log`, `detail-no-tail.log`; rejected attempts `detail.log`,
`detail-fixed.log`, stack `detail-sample.txt`, no-model `tail-{discard,commit}.log`.
Final-binary shared-command regression `boundary-no-tail.log` also completes
all five appends with identical non-timing diagnostics (candidate 3726.286ms,
including 3.834ms head/fence). This extra row is not a balanced speed trial.
Final binary `probe-no-tail` SHA256:
`c4682d0d34380406a6c20c8de2662ff999ab351dcf6eab37052088cfd63a9a97`;
bridge SHA256 unchanged from the preceding section. Temporary artifacts may
expire; tracked source and arguments are the rebuild path. Rollback: keep
profiling off; do not restore unused terminal command allocations. Refresh
on source, model, device, toolchain, workload or scheduling-policy changes.
DoD: release build and no-model self-test, unknown-mode rejection before model
loading, two lifecycle specs, completed detail/shared-command runs, exact
non-timing diagnostic comparison and format/diff checks pass. Source audit
and the terminal/continuing-command falsifier support a ROBUST verdict for
this profiling-lifetime fix; global speed and general numerical equivalence
remain VULNERABLE claims. Coarse-phase terminal routing is source-reviewed,
not separately exercised by a full-model coarse-only run.

### Existing Q6 FFN down/add: exact bounded control, no speed promotion (2026-09-07)

Predeclared hypothesis: opt-in `QWEN35_PREFILL_FFN_DOWN_ADD_FUSED=1` may reduce
current 27B/P256/T512 append wall time. LM-410 refuted this as a primary lever
on an older 9B/pp1024 baseline; do not forget or generalize away that result.
Use Flash on in both arms, change down/add only in candidate append, preserve
prefix/decode controls, measure executed Q6-add routes and retain strict state,
teacher top2/ECS and free-generation gates. Start with one quality falsifier;
only if it passes run bounded fresh-process ABBA. Keep 35% free memory,
24GiB cap, sequential runs and current cooldowns. No kernel/default/cache or
queue changes. A neutral/regressing result rejects further down/add promotion
in this slice; a positive row alone cannot change production admission.

The quality preflight passed, followed by four fresh paired processes in the
predeclared AB/BA/BA/AB order. A is down/add off, B is on; Flash executes in
both. Each process warms both modes, reconstructs independent prefix states,
measures full-width append plus fenced full logits, then checks 16 teacher
tokens and a fresh free-generation state. Profiling stays off. This is an
F16 nonadaptive raw completion fixture, not terminal-row pp or adaptive KV.

| Pair | Order | A append ms | B append ms | Throughput change A/B - 1 |
| --- | --- | ---: | ---: | ---: |
| 1 | AB | 3915.488000 | 3757.946708 | +4.192% |
| 2 | BA | 4083.904333 | 4244.918125 | -3.793% |
| 3 | BA | 4098.755959 | 4321.906291 | -5.163% |
| 4 | AB | 3886.172625 | 3703.938209 | +4.920% |

Pair means are A 3996.080229ms and B 4007.177333ms: throughput -0.277%,
time +0.278%, paired median throughput +0.200%, two wins out of four.
The preliminary quality run (3856.386416/3673.496084ms, apparent +4.979%)
is excluded from those means. The sign tracks measured order; fixed warmup
and prefix-creation order plus uncontrolled host noise remain confounders.
Neither a speedup nor a statistically resolved slowdown is established.

All five processes exit0 without guard termination. Every append reports
16 Flash dispatches and 31 Q6-add routes when enabled versus zero when off.
This proves the selected route executed, not coverage of all 63 FFN-down
projections. All checked live K/V, convolution and SSM values match exactly
at common prefix, after append (768 tokens), and after teacher (783 tokens).
All compared logits have max_abs=0; top1 is 16/16, ordered top2 32/32, token
embedding cosine minimum 1.0. Independent free IDs and text match across
all processes. The short text starts ` seen = set()` and ends
`for value in values:`; it is unfinished, EOS was not reached, and no external
coding scorer ran. This does not resolve the separate Flash-versus-SG4 strict
state mismatch or certify general coding quality.

Reproduce with the preceding release-build/link recipe and:
`scripts/run_safe.sh <probe> 180 24576 --model <Qwen3.8-27B-Q4_K_M.gguf>
--prefix 256 --append 512 --gen 16 --warmup --compare-ffn-down-add`, adding
`--reverse` for BA. Preserve the preceding runner environment: no quiet wait,
35% minimum free memory, sequential workloads, no concurrent compilation or
foreign-process interference. The probe pins chunk2048/group1/cooldown50ms.
Device: Apple M2 Max; Crystal 1.21.0/LLVM22.1.8; token SHA256
`5b3bcbe45fbd2fdaa78a5454ccbf671d9e31607189e50fb3aaea55e7e91b9c06`.
Temporary evidence: `/private/tmp/qwen-ffn-downadd.hNj4mb/`, `gate.log`,
`ab1.log`, `ba1.log`, `ba2.log`, `ab2.log`, and `analyze.py` (checks routes,
order, guard completion, cross-run quality, and excludes gate from means).
Timing binary SHA256:
`dd9e6404715f9b66dbefea0f542dd3382565234a0b59603d42361109cc4bfdba`;
bridge SHA256 remains
`48bb1469e2a473d30a94ab102df91268d549a4dd3710b076a0e59d137691005a`.
The timing binary retained the old `comparator=default_sg4` label; its explicit
`experiment=q6_ffn_down_add`, `baseline_flash=true`, and every append's Flash
route fields establish the actual comparator. After the batch, only that
label and the self-test success text changed; the corrected comparator is
`flash_d256_down_add_off`. The final release binary SHA256 is
`93b1f212ada3e00eae0ed17cf093e3e5574ccd3c540c514011cf1f0269fa80ac`;
its no-model self-test passes and `--compare-ffn-down-add --append 65` rejects
the unsupported shape before model loading. Final format/diff checks pass.
The first final rebuild lost its prior temporary Crystal cache directory;
a fresh isolated cache rebuild passed. No model workload ran during builds.
Temporary files can expire; tracked probe/fixture and arguments are the rebuild
path. DoD is the release build/self-test, bounded-shape rejection, five completed
route/quality checks, predeclared four-pair comparison, and format/diff checks.

Decision: keep the existing fusion disabled by default; retain only the
bounded comparator and two executed-route markers. No kernel, precision,
scheduler, cache layout or default changes. Source audit and executed gates
support ROBUST scoped route attribution and sampled value equivalence;
a speedup claim is VULNERABLE. Refresh on source/model/device/toolchain,
workload or scheduling changes. Next falsifier starts from FFN up/gate GEMM
dataflow versus llama.cpp at the same shape, not another broad tile switch:
LM-1042 already rejected Q6 SG8-B128 with a raw-F32 rounding epilogue.

### Q4 SG8-B128 staging falsifier (2026-09-07, predeclared)

Hypothesis, not an admitted speed claim: keeping the current 64x128 tile and
MMA order while reducing double-buffered staging from 24 KiB to 12 KiB for
gate and 16 KiB for up+SwiGLU may improve occupancy enough to pay for an extra
threadgroup barrier per K iteration. The fused epilogue still needs 16 KiB.
This is ordinary kernel staging, not LTP/WBA. Compiler maximum launch threads
are a resource warning signal, not an occupancy measurement.

The local llama.cpp non-tensor Metal path uses a 64x32 tile with 6 KiB staging
for complete tiles (`ggml-metal-device.cpp`, `get_pipeline_mul_mm`;
`ggml-metal.metal`, `kernel_mul_mm`). Our path already reuses a weight tile over
128 input rows and fuses up with SwiGLU, after a separate gate GEMM. Neither
fact establishes a speed advantage. Earlier single-buffer Q4 F32 and H16
experiments were negative (LM-prefill-Q4-SINGLE-BUFFER-FALSIFIER and
decision_update_203); the changed 24-KiB B128 staging footprint is the only
reason to reopen this bounded operator experiment. Full F16 weight expansion,
per-thread quant caching and epilogue micro-barriers remain rejected by their
existing falsifiers, not silently retried here.

Risk: CAUTION, experimental synchronization. Keep the full-threadgroup barrier
before overwriting the shared K tile, the existing barrier after publishing
that tile, exact accumulator order, F32 gate boundary and H16 SwiGLU rounding.
No production source, cache/state lifetime, queue depth or precision changes.
Rollback is to discard the probe variant; the default engine is untouched.

DoD: build a standalone probe; qualify its source-transform and value checker
with negative controls; compile baseline/candidate regular and fused pipelines
without dispatch; reject reduced maximum-launch-thread headroom. Only then run
fresh, sequential guarded processes for batches 256/512/1024/2048, with the
same actual Qwen3.8-27B Q4_K gate/up tensors (5120x17408), deterministic H16
inputs, five warmups and ten balanced ABBA cycles per batch. Check every finite
F32 gate and H16 activation bit before and after timing. The primary timing is
the GPU interval of both dispatches in one command, not host phase wait time.
Record source/tensor digests and all cycle times. Reject a batch for nonpositive
timings, any mismatch, launch-resource regression or guard termination.

Predeclared continuation gate: at least 7.5% reduction in pair median GPU time
and at least 8/10 balanced-cycle wins at both 512 and 1024, with no >3% median
regression at 256 or 2048. Passing admits only a later whole-append A/B candidate,
never a default or whole-model speed claim. Otherwise stop this staging route.
Use `scripts/run_safe.sh <probe> 120 4096`, the 35% free-memory floor, no quiet
wait under standing user authority, and no concurrent compilation or model
workloads. The runner cannot cancel an already executing Metal kernel; static
uniform-barrier review and small bounded buffers remain necessary guards.

#### Result: reject single-buffer staging, retain the bounded falsifier

Final row-varying fixture, Apple M2 Max, Qwen3.8-27B Q4_K_M, one fresh process
per batch, five ABBA warmup cycles and ten measured ABBA cycles:

| Batch | Baseline gate+up median ms | Single-buffer median ms | Time increase | Candidate cycle wins |
| --- | ---: | ---: | ---: | ---: |
| 256 | 10.659229 | 11.986563 | 12.452% | 2/10 |
| 512 | 20.468833 | 23.576896 | 15.184% | 0/10 |
| 1024 | 41.338833 | 44.822292 | 8.427% | 0/10 |
| 2048 | 81.883375 | 90.477583 | 10.496% | 0/10 |

Here "pair" means the two FFN dispatches in one GPU command. Each median uses
20 such command intervals per arm, not the median of ten ABBA cycle means.
The latter alternative aggregation also rejects: time increases are
7.217/13.912/8.429/10.504%, respectively. Independent Python recomputation
checks all raw cycles, summary arithmetic, tensor/source identities, headroom,
quality fields and successful runner exits. This is operator time, not pp/tg,
full-append speed, an occupancy counter, or a current llama.cpp runtime score.

Every checked finite F32 gate and H16 activation matches bitwise before and
after timing, including fresh NaN-poisoned reruns in reverse order. Per output
type, checked element counts are 4,456,448 / 8,912,896 / 17,825,792 / 35,651,584.
Host negative controls detect source mutation, corruption, NaN and unwritten
all-zero results. Maximum launch threads remain 704/704 for gate and 832/832
for fused up. All four final guarded processes exit0; no guard is weakened.
No full-model generation, top1/top2/ECS or coding scorer ran in this slice;
operator equivalence on one layer and deterministic inputs is not model-wide
quality evidence. The extra barriers are a plausible explanation for the loss,
not measured causal attribution. No new production route is admitted.

The first compile exposed character-index versus byte-slice extraction of
UTF-8 shader text; byte-index extraction and a boundary assertion fix the
probe. The first completed sweep used repeating input rows and is superseded
by the final high-LCG-bit fixture, whose first two rows are checked distinct.
Both sweeps lost; only the final sweep appears in the table. Both tensor
metadata entries now validate before either payload is read. A missing
advisory environment marker rejected an initial invocation before Metal/model
work; it is not an actual containment credential, and the runner does not set
it automatically.

Rebuild `bin/qwen35_q4_b128_single_buffer_probe.cr` with Crystal 1.21.0,
`--release`, a fresh `CRYSTAL_CACHE_DIR`, and the bridge link recipe above.
Run `--self-test`; `--self-test --batch=257` must reject before model/Metal.
Run `--compile-only` under the guarded runner, then `--run --batch=N` in four
sequential fresh processes. Set `COGNI_RUN_SAFE_ACTIVE=1` (advisory),
`COGNI_RUN_SAFE_REQUIRE_QUIET=0`, `COGNI_RUN_SAFE_WAIT_QUIET_SEC=0`,
`COGNI_RUN_SAFE_MIN_FREE_PCT=35`, `RUN_SAFE_PASSTHROUGH_STDIO=1`; keep the
120-second / 4096-MiB runner caps. Source hashes emitted by every process:
baseline `1167773b32064b8983d93b6850d7feabe919017b98a350a5aa7ca8dd7d8839e5`,
candidate `97b5b0605b7081ac485dc8c70340880f527af886bba44987167b75f5577102db`.
Final binary SHA256:
`38b39fabec60861d34f62c67f0207190f9259b0d3bf82c7e31ba5ba27ae25ed4`.
Bridge SHA256 remains `48bb1469e2a473d30a94ab102df91268d549a4dd3710b076a0e59d137691005a`.
Each of the two metadata-only tensor payloads is 50,135,040 bytes; their
digests and deterministic input digests are emitted by the tracked probe.
Temporary evidence: `/private/tmp/qwen-b128-staging.0e44Tv/`, final `b256.log`,
`b512.log`, `b1024.log`, `b2048.log`, `compile.log`, `analyze.py`; `initial-*`
logs are superseded. Temporary files may expire; the probe is the rerun path.

Adversary: ROBUST for this bounded negative gate and sampled value comparison;
VULNERABLE as any global performance/quality claim. Source review confirms
uniform barriers before alias overwrite and unchanged ordinary-encoder
gate-to-up dependencies. Final self-test, shape rejection, format and diff
checks pass. Keep double buffering; do not retry staging-memory reduction
without a changed hardware/compiler or measured bottleneck premise. Remaining
FFN up/gate opportunities are unproven; this does not establish that all
optimization routes are exhausted. Refresh on kernel/model/input/device,
compiler/driver or scheduling changes.
