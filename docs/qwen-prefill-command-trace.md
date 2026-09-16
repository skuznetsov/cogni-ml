# Ordinary prefill command diagnostics

## Current frontier: offline native compilation available; spill counts still unknown (2026-09-15)

After the user accepted the Xcode license and installed Metal Toolchain,
`xcrun metal --version` reports 32023.921 (metalfe-32023.921.6). The earlier
license/toolchain blocker below is historical. The installed asset contains
`air-nt`, `air-objdump`, and their Metal-named manuals even though
`xcrun --find metal-objdump` / `metal-tt` still fail. Direct `air-arch -name`
identifies `applegpu_g14s (Apple M2 Max)`; this is device enumeration, not compute.

Bounded compile-only probe in `/private/tmp/qwen-sg4-offline.TGRulx`:

```sh
xcrun metal -fmodules-cache-path="$PROBE/modules" -fmetal-math-mode=safe \
  "$SOURCE" -o "$PROBE/baseline.metallib"
xcrun metal -fmodules-cache-path="$PROBE/modules" -fmetal-math-mode=safe \
  -DQWEN35_SG4_REGISTER_GATE=1 "$SOURCE" -o "$PROBE/register.metallib"
# TOOLBIN is the installed Metal.xctoolchain/usr/bin, not Xcode's wrapper bin.
"$TOOLBIN/air-nt" -arch applegpu_g14s -platform_version macos 26.0 27.0 \
  -j 1 "$PROBE/baseline.metallib" -N "$PROBE/sg4.mtlp-json" -o "$PROBE/baseline.gpu"
# The identical native command with register input/output also passes.
```

`SOURCE` is `src/ml/gguf/kernels/fullattn_qwen35.metal`, SHA256
`824f224369ce719cb05ad766717006915a7b25536926e9b1f3eaca913ffdf682`.
The script contains only `pipelines.compute_pipelines`, with two objects whose
`compute_function` values are `qwen35_attn_decode_rows_sg4` and
`qwen35_attn_decode_rows_sg4_pregate`; other descriptor fields use defaults.
Both native translations exit0. Initial compilation needed a private module
cache; native translation needed an explicit platform version and pipeline
script (defaults rejected AIR2.8 versus2.5, then a bare function script).

`air-objdump --macho --descriptor --reflection` reads both native outputs.
Static local allocations sum to direct4608/pregate8704/candidate4608 bytes,
consistent with earlier runtime getters. These are shared-memory allocations,
not private spills. The inspected dumps provide no register/spill count.
Native disassembly fails for both outputs: `g14s-b0` is not recognized and
the disassembler cannot initialize `applegpu_g14s-apple-ios`. A separate
`air-nt -S` attempt on a metallib parses binary input as assembly and fails;
that invalid invocation is not evidence that assembly emission is impossible.
`air-binary-perf --help` advertises section-loading timing, not shader statistics.

Native output SHA256 baseline:
`2b0eba3197e773e482cef70cc3352b4bef960be16a05e326b5bebcf1e9f0d4de`;
candidate: `0f510ec4cdfb32b3c1fe3b08c4aa4539d666276a8ccd6bec1ef35b69089d8409`.
No GPU dispatch, model load, production edit, or speed claim. Offline compiler
output is not certified identical to runtime driver compilation. ROBUST only
for this compile/tooling boundary; register pressure, spills and their relation
to prior timing remain unknown. Next discriminating route is an M2-compatible
compiler-statistics capture, not identical timing reruns. Refresh after toolchain,
driver, source, descriptor or device changes; temporary artifacts are not durable.

## Historical tooling gate before installation (2026-09-15)

Read-only follow-up to the tail/paired gate; no compilation, GPU submission,
model load or toolchain change. `DEVELOPER_DIR=/Library/Developer/CommandLineTools
xcrun --find metal` and `--find metal-objdump` both fail with utility-not-found.
An inventory of Xcode's toolchain/usr bin directories and CLT/usr/bin finds
Xcode `metal` and `xctrace`, but no `metal-objdump`. Default `xcrun --find metal`
exits69 requiring Xcode license acceptance. Do not accept or bypass that gate
automatically; no claim is made about optional/uninstalled toolchain packages.

The installed SDK's `MTLComputePipeline.h` exposes resource/thread limits and
reflection bindings/arguments, not register/spill counts. Inspected companion
headers: `MTL4ComputePipeline.h`, `MTL4CompilerTask.h`,
`MTL4PipelineDataSetSerializer.h`, `MTLCounters.h`. Compiler tasks expose status;
dataset serialization is opaque; common counter structs do not add register
statistics. This bounded API inspection is not proof that no tool/private API
can report them. No runtime counter enumeration or capture was performed.

Apple's [Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/)
describes compiler spilled-byte statistics through GPU profiling, and explains
why private arrays with dynamic indices can spill. Neither statement measures
our candidate. Its threadgroup-size and constant-index suggestions remain
hypotheses here. The newer [Shader Cost Graph workflow](https://developer.apple.com/videos/play/tech-talks/111374/)
is described for M3/A17 Pro (Apple family9), not an M2 Max capability certificate.

Decision: park the register candidate's optimization claim; do not add guessed
spill telemetry or repeat the same noisy timing series. Reopen after an actually
available M2-compatible statistics/capture path, or a new discriminating method.
The existing bridge's safe math and production routing remain unchanged.
`python3 /private/tmp/qwen-two-stage.vlxfep/compare_stages.py` was rerun: pinned
capture hashes and195 matched layer keys pass; suffix attention still accounts
for2385.079/2715.734ms across15 instrumented full-layer calls (host time, not
whole-request or new performance data). Source state: `5f2624ba` plus unrelated
pre-existing WIP. Refresh tooling findings after SDK/Xcode/license/device change.
Adversary verdict: ROBUST for the inspected-route limitation; a universal
"Metal cannot expose spills" claim would be unsupported. The 195-key/hash check
passes and `git diff --check` is clean; neither proves a new optimization.

## Tail/offset and balanced timing gate (2026-09-15, predeclared)

No shader or production routing change. First, one register-candidate process
executes 1+2+3+58 query rows at base7839/fixture193, F32/D256/heads24/KV4.
Each command advances only Q/gate/output bindings; K/V stays global. Validate
CPU oracle, exact prior prefix, untouched future rows and trailing canaries
after each command. Stop on first error; only a complete four-command pass
admits timing. Then eight fresh single-command processes, order ABBA BAAB
(A=direct, B=register), same binary/three-pipeline compilation order/fixture,
selected64 rows, no warmup/cooldown. Keep lease,35%/24GiB/300s and no quiet wait.
Any failure stops the entire series; no replacement samples. Compare within
blocks and report raw GPU/host times, never production/general speed from
this one model-free shape. DoD: source/selector/offset self-tests, CPU/Metal
builds, guarded logs and a complete-series check. Rollback: omit new modes.

Result: all four tail commands pass (completed rows 1/3/6/64), including
nonzero Q/gate/output offsets 24,576/73,728/147,456 bytes. Prior output remains
exactly unchanged; future rows and trailing canaries remain poisoned. Maximum
CPU-oracle absolute error is 1.064646237225464e-6. Eight fresh paired processes
also pass with that same maximum error, exit0, no replacements or GPU errors.
This is per-run oracle agreement, not saved-array bitwise direct/candidate
equivalence or real-model quality evidence.

Raw times in execution order (milliseconds):

| Sample | Kernel | GPU | Host |
| --- | --- | ---: | ---: |
| 1 | direct | 58.161917 | 62.585417 |
| 2 | register | 88.589667 | 91.765292 |
| 3 | register | 60.205875 | 63.070416 |
| 4 | direct | 59.962625 | 62.750000 |
| 5 | register | 60.275167 | 63.822833 |
| 6 | direct | 61.130250 | 65.100792 |
| 7 | direct | 79.047250 | 82.414875 |
| 8 | register | 65.665750 | 68.724125 |

Block1 GPU means: direct59.062271 vs register74.397771ms (candidate25.97%
slower); block2: direct70.088750 vs register62.970458ms (candidate10.16%
faster). Host signs agree. No samples excluded. **No repeatable speed win**;
the opposing signs and visible variability preclude promotion. Equal compile
sets remove one known asymmetry but not driver caching, clock or host noise.
Do not repeat this series merely to seek a favorable result. Next: inspect
available compiler/occupancy/spill diagnostics without compute; if unavailable,
park this candidate and select a different measured bottleneck. Production
routing is unchanged; prior intermittent interactivity failures remain open.

Verification: source-contract test observed red before implementation, then
15 SG4/stage-split specs pass; CPU-only and Metal release builds plus both
self-tests pass. Six new malformed paired selectors are rejected, alongside
existing six selector negatives and four prefix/future/canary negatives.
Malformed Metal CLI and valid GPU modes on CPU-only builds reject. Temporary
series checker verifies all eight terminal results, distinct PIDs, mode/order,
source/shape, bounds, finite times and oracle guards; six mutated-log negatives
reject. Parent also inspected the tail log and complete raw series. Build uses
private bridge/cache with CommandLineTools (no shared build artifacts changed).
Parent adversary verdict: ROBUST for these bounded diagnostic runs and exact
CLI selection; VULNERABLE as a speed/stability generalization. The source-text
spec does not itself prove execution, and equal maximum errors do not prove
pairwise equality; executed oracle/canary checks supply the scoped evidence.
Correlated read-only Luna review also returned ROBUST for CLI, tail offsets,
prefix guards and equal compilation order, with no P1 blocker. The temporary
checker qualifies log rejection, not injected live GPU fail-fast behavior.
Guards: lease,35% free/24GiB/300s; quiet wait disabled; preflight76% for tail,
78% for every paired run. No model weights loaded. Refresh evidence after
source/compiler/device/driver/shape drift or loss of temporary artifacts.

Reproduction: build `bin/qwen35_sg4_tail_probe.cr --release` with a private
`bridge.o` and Metal/Foundation/c++ link flags, using
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`. Under `scripts/run_safe.sh
<probe> 300 24576`, run `--register-tail-check`, then only on complete success
`--paired-command=direct/register` in the table order. Environment:
`COGNI_METAL_LEASE_WAIT_MS=0`, unset `COGNI_METAL_LEASE_PATH`,
`COGNI_RUN_SAFE_REQUIRE_QUIET=0`, `COGNI_RUN_SAFE_WAIT_QUIET_SEC=0`,
`COGNI_RUN_SAFE_MIN_FREE_PCT=35`, `COGNI_METAL_COMMAND_TIMEOUT_MS=180000`.

Evidence directory: `/private/tmp/qwen-sg4-tail-pair.SWABsJ/` (temporary).
SHA256 provenance:

```text
probe source b158c93b69a976738d827360b3929985f6fc9f8fbf4b610a2dbbf6cd7aad7083
shader 824f224369ce719cb05ad766717006915a7b25536926e9b1f3eaca913ffdf682
flagged input f1b45404dd4069efd96f2579304dcae4106b990257a33ad93f094b196165bd18
probe binary d5ef15571d8bd565f43d4e9631e426c4e94301dca6814e7b082903090b465e63
bridge a681a678ba98e1c0c6c62e8eed84777437a213ae4c8891f975cb87d693f97663
tail.log 26e20ef1000f410256fe1796ff4746edaec235a88998474f3f91c9fa43a80328
check.py a297f11d71f03fd4df97979b45bfd8168b03e348575ade64293f359b3422ce23
1-direct.log c46e9b9681514613e505189073282140fecf2ed982c96ff15f9e59e0feead8d0
2-register.log 3cdc7e02d7e88098b4357282eeb10a3afcd262c31ec0175e94d530d237bbea72
3-register.log c22d6bb828463118ed18ed2e76f76e5345e62734909fd61a4fa13261586e9959
4-direct.log 2237dd46f8195e26a1729b70694c0658df28c2d9d7d4e3134bf29771b430fea9
5-register.log b21e14757e28db9defdb865d4587a3d48cce32a3c97b5dd8c9ae23b3cbfc9f85
6-direct.log cff681b7d50ee6d5e2ead22e68777604d68165cc55cff4b46946397c0d6fd22e
7-direct.log 0801564102dc062db587d678e57eb1916f694b549889f979ed51732c349a7744
8-register.log 2de88e6022ca16cca671c29a77a4c5ee7532fbeb7bbb4ef7730ac4b83d8a826d
```

## Earlier diagnostic: thread-local gate passes one bounded case

Register-gate experiment (2026-09-15, predeclared): keep pregate's early loads
but store each lane's eight values in thread-local storage. Define
`QWEN35_SG4_REGISTER_GATE=1` only in the diagnostic compiler input; no runtime
environment switch or production admission. Compare default preprocessed
kernel with HEAD, test lane-to-slot ownership, build both probe targets, then
compile three metadata pipelines. If static bytes fall to4,608 and SIMD32/max
threads admit128, allow exactly one candidate64-row command, base7839/F32,
fixture193, CPU oracle and untouched future/canaries. Preserve35%/24GiB/300s,
lease/no quiet wait, no warmup/retry; stop on first error. Rollback: omit the
macro. One pass/time is not stability or speed promotion. Metal's available
getters do not report registers/spills: this limitation remains explicit.

Result: compiled candidate4,608B, direct4,608B, pregate8,704B; all SIMD32 and
max1,024 threads. One candidate dispatch on M2 Max passes: oracle max absolute
error1.064646237225464e-6, all129 future query rows and trailing canaries intact,
GPU55.505208ms / host58.141083ms, exit0, preflight free78%. No model, warmup,
retry or further dispatch. Timing is diagnostic only: historical direct and
pregate results are not a contemporaneous balanced comparison. The prior
interactivity failure remains open; partial-row GPU coverage, register/spill
behavior, repeatability and speed promotion are not admitted.

Verification: the new lane-ownership test failed before implementation;
14 SG4/stage-split specs and both CPU/Metal builds/self-tests pass. CPU-only
candidate metadata and malformed selectors reject before Metal initialization.
Clang-preprocessed pregate with macro unset/0/2 matches HEAD `0ba87a39`
byte-for-byte (SHA31303beb4873596f51af36ad3b958b7ce98dffe2cdba1ab4a0cd23f60aed72e8).
The enabled preprocessed diff changes only gate declaration/pointer/load/read;
attention arithmetic, output expression and SIMD-group barriers are unchanged.
This proves the default source boundary, not binary identity across compilers.
Correlated Luna review: ROBUST for lane ownership/default-off isolation in
the fixed D256 probe. There is no kernel-level D<=256 guard; larger dimensions
exceed both the candidate's eight slots and the existing Q scratch. Do not
route arbitrary shapes into this candidate. Partial-tail GPU behavior has
not been measured here despite the unchanged, statically reviewed barriers.

Evidence: `/private/tmp/qwen-sg4-register.Md08Ji/{metadata,single}.log`;
runner commands use the existing `300 24576` limits and respectively
`--pipeline-info-register` / `--single-command=register`. Lease wait0, quiet
requirement/wait0, memory floor35%, command timeout180000ms; private bridge
and Command Line Tools builds. Captured hashes unchanged before/after:

```text
probe-source 800b6262ca67b107140dfaadc401272b2ca839cd6cf3a9dc1b2f7a5ef220549d
shader 824f224369ce719cb05ad766717006915a7b25536926e9b1f3eaca913ffdf682
candidate-input f1b45404dd4069efd96f2579304dcae4106b990257a33ad93f094b196165bd18
probe-binary e9ac3652bb0fd3fb01daf3ea8304d37669b6dca895eac6b5d1e76a97b23e0773
bridge-object a681a678ba98e1c0c6c62e8eed84777437a213ae4c8891f975cb87d693f97663
metadata.log 0dde25ef8e148b505ade2abf41ea3efdde961a10249c867df6c92ff2cae47267
single.log f878fa2a66f8b8550c1f4030ab395fbf8d8fca91511b2f14e150b548568ccb20
```

Next: bounded tail/offset checks before a same-binary balanced direct/candidate
comparison with equal compilation sets. Do not promote from one55.5ms sample.
Refresh on source/compiler/device/driver/shape drift or temporary evidence loss.

### Previous compile-only resource measurement

Bounded diagnostic (2026-09-15): `qwen35_sg4_tail_probe --pipeline-info` compiles
the same F32 source in direct/pregate order and reads compiled static
threadgroup bytes, execution width and maximum threads. No tensor fixture,
compute command, encoder or dispatch is admitted. Additive read-only bridge
getters; production routing and shader bytes stay unchanged. Rollback: omit
the diagnostic mode. DoD: CPU/Metal builds, source branch guard, existing
self-tests/specs, malformed-mode rejection, then one leased compile-only run
under the existing 35% free/24GiB/300s guard (quiet wait disabled). Stop on
error; no compute retry. Source predicts 4,608 versus 8,704 bytes, but report
the compiled values even if different. Neither equal nor different metadata
establishes runtime occupancy, register pressure or watchdog causality.

Measured on Apple M2 Max: direct **4,608 bytes**, pregate **8,704 bytes**;
both execution width **32**, maximum threads **1,024**. Compiled static
threadgroup storage therefore preserves the source's extra 4,096-byte gate
array (1.889x), not a measured occupancy or speed ratio. `compute_commands=0`
is the inspected branch contract, not a driver-wide activity counter: device
initialization still creates a command queue and compilation uses the driver.
The branch and its callees create no compute command or tensor fixture.
This closes the static-allocation question only; the earlier callback remains
unresolved. Next candidate is removing shared gate staging, with explicit
register-pressure and numerical checks before any production promotion.

Verification: 13 SG4/stage-split specs pass (new branch guard failed before
implementation), CPU-only and Metal release builds/self-tests pass; CPU-only
metadata mode and malformed/extra arguments reject before initialization.
A temporary native-FFI check confirms null handles return invalid sentinels
(width0, bytes-1) without initializing the device.
One leased compile-only run exits 0, preflight free78%, guards unchanged.
Build used `DEVELOPER_DIR=/Library/Developer/CommandLineTools` and a private
bridge object; default Xcode required license acceptance, which was not changed.
Command: `scripts/run_safe.sh /private/tmp/qwen-sg4-metadata.AJb8qV/probe 300
24576 --pipeline-info`, with lease wait0, quiet requirement/wait0 and free35%.
Evidence directory is temporary; refresh on source/build/device/driver drift
or evidence loss.

Luna's correlated review found no P1 and returned ROBUST for API wiring and
the application-level no-dispatch route. Afterwards only a probe comment was
clarified to distinguish command queues from command buffers; capture hashes
below refer to the measured build, not that comment-only revision. SHA256
before/after measurement unchanged:

```text
probe-source e5a34f130d203bf6202644d3d05d4450998db74e365e826e6c8823b33883d7c3
bridge-source 8100559599f0857e2e4d9bfedb616f52de124761dce2476ca592fd0a621d949c
device-source ecc5d96332833a8a94d9a55914d515b5e34efe001f762d65aaa7c919b907c934
shader a53054dd97bdfdbfa2e4a8cdc160898f9c7e6c7a5907fbb6b1884bfa1dd2eff1
bridge-object a681a678ba98e1c0c6c62e8eed84777437a213ae4c8891f975cb87d693f97663
probe-binary d89152e6d92669287223a37d4f2b62e1d408fdc83031dcf127b5f45382e591b9
pipeline-info.log b2a27f5c43b3f928b69704e9269dea7ea2c72f57ad597512d48cfeaca00698e5
```

The 2026-09-14 model-free F32 single-command BAAB series is **measured-red**:
pregate/direct/direct pass, then pregate fails on the first and only dispatch
in its fresh process. Previous dispatches within that process are not required
for the callback. The same pregate fixture both passes and fails across this
series; driver/host history and kernel resource sensitivity remain unresolved.
No retry after GPU failure, production change or speed promotion. Stop varying
row size/process reset as assumed fixes; inspect the kernel/resource boundary
before another workload. See "Fresh-process single-command discriminator".

### Existing standalone stage-split diagnostic

Implemented diagnostic: `QWEN35_FULL_PREFILL_STAGE_SPLIT=after_attention`
keeps only the boundary after attention (prepare/KV + attention, then output/FFN).
`1` retains three commands; unset/invalid values disable the diagnostic. Parse
the selected mode once for each helper invocation, keep existing standalone/F32
admission, and label the combined command `prepare_kv_attention` in traces.
No successor may be created at the skipped KV-write cut or after a failed wait.
DoD for implementation: mode/routing, fake-command identity/order/failure tests,
source call-site guards, the existing model-free suite, CPU-only and Metal
compile checks. GPU parity/stability remains a separate guarded experiment.

Default-off diagnostic, not a production fix: `QWEN35_FULL_PREFILL_STAGE_SPLIT=1`
splits only standalone ordinary F32 full-attention into synchronous
`prepare_kv`, `attention`, and `output_ffn` commands. Shared, adaptive and F16
routes remain unchanged; values other than `1` or `after_attention` preserve
the original command.
Each successor is created only after the previous encoder has ended and its
command wait succeeded; the final stage allocates no successor. Tensor buffers,
shapes, kernel selection, precision and operation order are unchanged. There
are no new sleeps, retries or concurrent commands. A failed stage must propagate
the original exception; partial state is not reusable or a checkpoint.

The three stages are norm/QKV/QK normalization/RoPE/KV write; attention alone;
then O projection/add-normalization/FFN/final add/output copy. Stage records
carry command identity, start/rows and stage, and nest inside the existing
exact-layer call records on the serial provider path when
`QWEN35_PREFILL_COMMAND_TRACE=1` is also set. Without that outer trace, stage
records alone do not identify the layer. They time host commit/wait, not GPU
execution. No exact failing kernel is inferred within a multi-op stage. The
existing aggregate encode profile includes the new intermediate waits; do not
interpret that profile as encode-only or use this mode for speed comparisons.

DoD: fake-command success and failure at each stage, unchanged exception with
broken logging, stage routing/default-off checks, source placement/ended-encoder
guards, CPU-only and Metal builds. Then at most one guarded same-input replay,
retaining fusion OFF, FFN capacity reuse OFF and existing memory/timeout limits.
Stop after its first GPU failure. A pass is scheduling-sensitive diagnostic
evidence, not root cause, state parity, speed or production-stability closure.
Rollback is unset `QWEN35_FULL_PREFILL_STAGE_SPLIT`; do not promote the splitter.

Current evidence: the three-stage replay completed both calls; a later OFF
control using that same binary failed on call 2 and that series stopped.
The new two-stage implementation also completed one separately authorized
guarded replay with the saved second-output digest. This is not a same-binary
two-stage/OFF comparison. Production stability, causal attribution and speed
remain open; both diagnostics stay default OFF. Details and provenance below.

`QWEN35_PREFILL_COMMAND_TRACE=1` adds flushed stderr records around the existing
ordinary shared-command commit/wait and ordinary routed full-layer call in
`qwen35_cpu.cr`. Unset, zero and other
values disable it. It does not enable GPU timing, change command grouping,
insert sleeps or retries, alter cache publication, or modify CogniGraph.

The motivating failure is the second-call `Impacting Interactivity` observed
with FFN capacity reuse both on and off (see `qwen-memory-sampling.md`). The
existing boundary profiler writes after successful wait/publication and cannot
identify a failed wait through its completed timing records alone.

Each command emits `qwen35_prefill_command` with:

- `phase=submit_wait_begin` before submission, followed by `submit_wait_end`
  or `submit_wait_failed` if control returns through the corresponding path;
- trace-local sequence and command object identity (not a Metal command ID);
- `start_pos`, logical `rows`, and the existing fused-group counter;
- loop cursor at the previous ordinary flush (initially zero) and at this flush;
- host elapsed milliseconds on the terminal record, **not GPU execution time**.

Cursor positions localize control flow; they are not an exact encoded layer
interval. For example, standalone adaptive work flushes before incrementing
the current layer. Group count likewise is the existing rotation counter, not
an exhaustive dispatch or layer count. Trace identity is local to one process.
These command records do not cover CogniGraph, private/internal command buffers,
finalization before submission, publication after wait, or hard process/device
termination. The additional full-layer call records below have a different scope.
A successful wait record does not certify successful cache publication.

### Routed full-layer call records

`qwen35_prefill_layer` is a separate record type around the ordinary
`full_attn_layer_chunk_project_routed` call, after its existing pre-call flush.
It carries `route=full_attn_chunk_routed`, exact zero-based `layer`, `start_pos`,
logical `rows`, and trace-local identity/sequence. Phases are `call_begin`, then
`call_end` for a returned array (including an empty array), `call_declined` for
nil, or `call_failed` before rethrowing the original exception. A declined call
can fall back to CPU; do not count it as successful GPU work.

This boundary includes setup, encoding, private command commit/wait and readback.
Its host elapsed time is **not** a wait-only or GPU duration. A failure record
identifies the routed call's layer, not which operation failed inside it or
whether submission occurred. These records deliberately omit command identity.
Fused full/recurrent, adaptive/shared, final-layer-specialized and recurrent-only
calls are not covered by this added wrapper. Hard termination or a broken stderr
sink can still leave incomplete evidence. The default-off branch calls the same
helper directly; command grouping, math and cache handling are unchanged.

Logging is best-effort and flushed before submission. Logging exceptions are
suppressed; command exceptions are rethrown unchanged without logging their
payload. A blocked stderr sink can still perturb latency. The safe runner may
buffer stderr until process exit, so a flushed record is not necessarily live
in the caller's log. Do not use diagnostic timing as a performance promotion.

## Verification and rollback

Eleven no-GPU specs exercise explicit opt-in, pre-submit visibility/flush, result
preservation, failed-wait emission/original exception identity, broken logging
and repeated command identity. A source guard checks ordinary-branch placement
before publication and the adaptive flush-before-cursor-increment distinction;
it is not runtime route coverage. Added full-layer tests check nil versus empty
array, original exception identity with no payload, failed logging, exact layer
metadata and exclusive enabled/default-off source wiring after the existing
flush. Commands:

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen-command-trace-spec crystal spec spec/qwen_prefill_command_trace_spec.cr
crystal tool format --check src/ml/gguf/qwen_prefill_command_trace.cr src/ml/gguf/qwen35_cpu.cr spec/qwen_prefill_command_trace_spec.cr
git diff --check
```

CPU-only compilation and the release CrystalBall two-call provider build also
passed on 2026-09-11. These checks establish the diagnostic helper and compile
integration, not GPU stability or complete route coverage. Rollback is unset
flag; reverting this diagnostic slice removes it. The separate capacity-reuse
experiment remains default-off and uncommitted.

Before GPU use, pin the new binary and source/input identities, retain the
existing 35% free-memory floor, 24,576-MiB cap, 300-second timeout, 2,048-row
chunks, group limit 1 and cooldown 50 ms. Run one feature-OFF diagnostic replay
and stop at its first failure; no uninstrumented retry loop. A source, build,
route or device change invalidates causal comparisons until rechecked.

## First guarded diagnostic replay (2026-09-11)

One feature-OFF replay with trace enabled failed with the same Metal
`Impacting Interactivity` / completion status `-6`. No retry followed. There
were 131 begin records and 131 matching terminal records, including one failed
wait and no unmatched command identities. The helper's failure path therefore
also has real-route evidence, not only a simulated throwing block.

Call one completed the captured 27-token output in 91.333 seconds. Call two
hit the resident prefix at 7,839 tokens with 195 helper rows. Its ordinary
shared-command trace was:

| Sequence | Cursor before / at flush | Groups | Host submit/wait ms | Result |
| --- | --- | --- | --- | --- |
| 1 | 0 / 7 | 1 | 656.780 | returned |
| 2 | 7 / 11 | 1 | 273.420 | returned |
| 3 | 11 / 15 | 1 | 210.074 | failed |

These observations localize the failure to the third ordinary shared-command
wait in the suffix. They do not identify a kernel, prove an individual-command
duration threshold, or exonerate cumulative occupancy, driver state or resource
history. The first command took longer to return successfully than the failed
command took to return an error; host elapsed time alone is not a GPU watchdog
threshold measurement. The two-call quality gate remains red, with no second
output. Do not classify the trace analyzer's successful pairing check as a
successful inference replay.

Launcher wall time was 99.971 seconds, exit 1; the observer exited 0 with 50
samples and no collection errors. Sampled free memory started at 76% and reached
43%, above the 35% floor; no floor kill was logged. All preceding launch guards
were retained. Logging/rebuilding can perturb timing, so this is not an exact
performance comparison with the uninstrumented binary.

Evidence root: `/private/tmp/qwen-command-trace.KC3KjM/`, with `run_inventory.py`,
`check_trace.py`, `command-trace-off-{dry,gpu}` logs/result/samples and source
hash manifests for cogni-ml `src` plus `build/bridge.o`, and the tracked
CrystalBall provider sources/probe/lockfile. The manifest files were captured
during the build and rechecked afterward; they are not a complete hermetic
toolchain/dependency attestation. Binary SHA256:
`07bb89c7fb055d28bd0cc3e217b0831a62ce73b3cc4d27a0416d1533d79f7f6a`.
The experimental FFN capacity-reuse source remained present but disabled.

ROBUST for bounded ordinary-command failure localization and exception-preserving
telemetry; the underlying interactivity failure is unresolved. Next inspect the
encoded operations at cursor 11 / 15 before designing a smaller-command
falsifier. No scheduling change, default enablement or further GPU run is
justified merely by this location. Temporary artifact loss or changes to the
source/model/device/route require refreshed evidence.

## SG4 partial-group barrier correction (2026-09-11)

Source inspection found a synchronization defect in both SG4 attention kernels:
each SIMD group owns a query row and private threadgroup-memory slices, but an
out-of-range group returns before the initial whole-threadgroup barrier. A
195-row dispatch leaves only three participating SIMD groups in its final
threadgroup; the first request's 2,048/1,668-row chunks are both divisible by
four. The existing Flash comparison probe already pads its SG4 reference to
avoid this condition, so that comparison did not certify unpadded production.

CAUTION scope: replace only the two initial barriers with SIMD-group barriers,
retaining the threadgroup-memory fence. No padding, math, allocation, queue,
cache representation or admission changes. Each active SIMD group must have
32 lanes, and scratch ownership must remain disjoint by `sgitg`; introducing
cross-group scratch consumers invalidates this correction. Rollback is a
revert of this slice, not a recommendation to execute the unsafe tail kernel.

Falsifier-first: `crystal spec spec/qwen35_sg4_tail_safety_spec.cr` failed two
kernel checks before the change (three examples total). The guard rejects
whole-threadgroup barriers after per-SIMD-group retirement and checks scratch
partitioning and initialization-before-consumption. This is a structural
regression test, not a Metal execution proof.

DoD: the static test must pass; the model-free SG4 tail probe must match its
independent CPU oracle with exact row counts, untouched output guards, both
gate variants and F32/F16 KV. Then perform one source-pinned original two-call
replay under the existing guards. Stop on the first GPU failure. Until that
replay completes, the synchronization defect is a root-cause candidate, not
proof that it caused the observed interactivity error. No speed claim follows.

The synchronization choice follows Apple's distinction between SIMD-local
memory ordering and cross-SIMD-group communication in
[Threadgroup memory synchronization (WWDC20)](https://developer.apple.com/videos/play/wwdc2020/10631/).
The GPU probe is bounded to the existing Apple-GPU/32-lane, d256, GQA6 contract,
not a new cross-device capability certificate.

Executed kernel checks on Apple M2 Max:

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen-sg4-tail-spec crystal spec spec/qwen35_sg4_tail_safety_spec.cr spec/qwen_prefill_command_trace_spec.cr
CRYSTAL_CACHE_DIR=/private/tmp/qwen-sg4-tail-build crystal build bin/qwen35_sg4_tail_probe.cr --release -o /private/tmp/qwen-sg4-fix.qcyAuV/tail-probe --link-flags="/Users/sergey/Projects/Crystal/cogni-ml/build/bridge.o -framework Metal -framework Foundation -lc++"
COGNI_RUN_SAFE_REQUIRE_QUIET=0 COGNI_RUN_SAFE_WAIT_QUIET_SEC=0 COGNI_RUN_SAFE_MIN_FREE_PCT=35 COGNI_METAL_COMMAND_TIMEOUT_MS=180000 scripts/run_safe.sh /private/tmp/qwen-sg4-fix.qcyAuV/tail-probe 300 24576 --run
```

Nine spec examples passed. All 48 model-free GPU cases passed, including exact
193/194/195/196-row shapes and 195 rows at base 7,839. F32/F16 KV and direct/
pregate kernels matched the non-uniform Float64 CPU oracle within the declared
1e-5 absolute tolerance; worst observed error was 1.0808095e-6. All output guard
rows remained unchanged. The runner reported exit 0 after approximately one
second, preflight free memory 75%; no model weights were loaded. Probe binary
SHA256: `ba76db164abd94cbd398bc76ce9659a771b3d289f63cbd30599314d54d81cfe7`.
The synthetic KV values are exactly representable in binary16: this validates
both storage variants, not arbitrary F32-to-F16 quantization quality.

### Original two-call replay: still failing

The corrected release provider, with FFN capacity reuse still OFF, passed the
metadata-only dry check but the single guarded GPU replay exited 1. The first
call matched the captured tool call and 27-token count, taking 88,445.4 ms.
The second call again reached the exact 7,839-token cached prefix / 196-token
suffix (195 helper rows), then failed with `Impacting Interactivity`. This time
it failed in the **first** suffix command, cursor 0 / 7, at 517.172 ms host
submit/wait time; the prior trace failed at cursor 11 / 15. There is no second
output and no completed two-call parity certificate. No retry followed.

Trace pairing: 129 begins / 129 terminals, one failed terminal, no unmatched
identities. Observer: 48 samples, zero collection errors, sampled free memory
75% initially / minimum 42%; no memory-floor kill. Monotonic launcher wall time
was 96,705.197 ms. The runner's approximate tick counter printed `~69s`; use
the launcher clock for elapsed time, not that label.

Evidence: `/private/tmp/qwen-sg4-fix.qcyAuV/`, `run_inventory.py`,
`sg4-fix-off-{dry,gpu}` logs/results/samples, `check_replay.py`, and the two
source manifests. The checker correctly exits nonzero for the failed replay
even though trace pairing passes. Provider binary SHA256:
`f8539c3493eac0d60f1c549e6792eeca0f3d3df5bf130b63430aad63a94b666c`.
Shader SHA256:
`a53054dd97bdfdbfa2e4a8cdc160898f9c7e6c7a5907fbb6b1884bfa1dd2eff1`.
Source manifests were captured before the build and rechecked after it; the
launcher revalidates them and the pinned input/binary identities before use.
The separate uncommitted FFN capacity helper remained present but disabled.

**Decision:** retain the two-barrier correctness correction and its passing
kernel regression gate. Reject the hypothesis that this correction alone
resolves the provider's interactivity failure. The full two-call DoD is red;
do not promote stability, memory reuse, or speed. Moving failure location also
weakens any diagnosis specific to layer 11. Next discriminator is a bounded
separation of operations inside the first suffix shared command (including
the full-attention/recurrent fused handoff), with identical math and input,
not another repetition or arbitrary padding. This is proposed only; no such
scheduling edit or additional GPU experiment was made in this slice.

The existing `QWEN35_PREFILL_FUSE_FULL_REC_OFF=1` switch returns from the fused
route before encoding (`qwen35_cpu.cr`, `full_attn_then_recurrent_chunk_project_many_routed`)
and is the first candidate for a no-new-code ablation. It changes routing and
may alter scratch/timing, so a pass would localize the issue, not prove a
particular kernel defect. This applies to this ordinary F32-KV replay only:
adaptive QBit admission explicitly rejects disabling the fused corridor.

Post-run adversary review added two probe-only guards: validate the embedded
corrected shader SHA256 before Metal initialization, and acquire the existing
cross-process Metal lease before GPU use. Rebuilt as `tail-probe-guarded`;
dry mode and `--self-test` pass, including rejection of one-byte source drift
and whole-threadgroup-barrier substitution. The combined static/trace/lease
suite passes 11 examples. These final launcher guards were checked without
another GPU run; the 48-case GPU evidence above uses the earlier probe binary
with identical shader bytes. The lease coordinates only cooperating clients,
not arbitrary third-party GPU workloads or WindowServer.

### Fusion-off ablation: standalone full-attention failure remains

One further authorized replay reused the identical provider binary and source
manifests above. The only execution-environment change was
`QWEN35_PREFILL_FUSE_FULL_REC_OFF=1`; FFN capacity reuse remained OFF. Dry mode
passed. The single GPU run exited 1 with `Impacting Interactivity`, so disabling
the fused corridor is not a sufficient workaround. No retry followed.

The first request passed the captured tool-call and 27-output-token checks
(84,223.7 ms); this is not full-logit parity or a speed comparison. The second
request retained the 7,839-token prefix / 196-token suffix / 195 helper rows.
Its initial traced shared command, cursor 0 to 3, completed in 291.386 ms.
Subsequent memory events reached layer cursors 8, 16 and 24 before the failure.
The exception stack names the standalone `full_attn_layer_chunk_project` wait,
called through the ordinary F32 route. It does **not** identify the first full
layer, an exact failing layer, or the attention kernel within that helper.
The helper also encodes projections, normalization, cache writes and FFN work.

Only nine ordinary shared-command begin/terminal pairs were traced, with no
unmatched or failed trace records. Standalone helper commands are outside that
trace: successful pairing does not certify the GPU replay. `check_replay.py`
correctly rejects the run using its nonzero exit and missing second completion.
The observer collected 47 samples without errors, minimum free memory 42%
(initial 75%), under the unchanged 35% floor / 24-GiB cap / 300s timeout.
No memory-floor kill occurred; this does not exclude GPU resource pressure.
Monotonic launcher wall time was 94,330.570 ms.

Evidence: `/private/tmp/qwen-fusion-off.ikJywu/`, `run_inventory.py`,
`fusion-off-{dry,gpu}` logs/results/samples and `check_replay.py`. The launcher
rechecks the previous source/input/binary identities. Executed commands were
`python3 /private/tmp/qwen-fusion-off.ikJywu/run_inventory.py --dry` and `--run`.
The launcher refuses to overwrite existing evidence; any future authorized run
needs a fresh output directory and refreshed identity checks. Temporary artifacts
are not a permanent fixture; refresh if those inputs become unavailable.

**Decision:** ROBUST for the bounded negative result, not root-cause closure.
The flag changes both requests, scratch usage, host readbacks and fused-group
rotation/cooldowns; it is not a pure one-command scheduling intervention.
Keep production defaults unchanged. Next add narrowly scoped standalone-command
layer/failure attribution before attempting a stage-level discriminator; do not
infer an SG4 defect from the enclosing helper stack or claim a speedup from
this single failed run. Refresh after source/model/device/input/scheduling
changes. The full two-call stability gate remains red.

### Full-layer attribution extension: local checks

The missing `observe_full_layer` regression initially failed to compile. After
adding the wrapper, all 11 command/layer trace examples passed; the combined
trace, SG4-tail and Metal process-lease suite passed 16 examples without GPU.
Formatting and `git diff --check` passed. A CPU-only executable and the actual
release two-call provider both built successfully:

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen-standalone-trace-spec crystal spec spec/qwen_prefill_command_trace_spec.cr spec/qwen35_sg4_tail_safety_spec.cr spec/metal_process_lease_spec.cr --no-color
CRYSTAL_CACHE_DIR=/private/tmp/qwen-layer-trace.s9ILiM/cpu-cache crystal build -Dcpu_only bin/qwen35_generate.cr -o /private/tmp/qwen-layer-trace.s9ILiM/cpu-probe
# From ../crystal_ball; the provider annotation already links build/bridge.o.
CRYSTAL_CACHE_DIR=/private/tmp/qwen-layer-trace.s9ILiM/metal-cache crystal build scripts/cogni_qwen_two_call_probe.cr --release -o /private/tmp/qwen-layer-trace.s9ILiM/probe
```

The initial provider link attempt supplied bridge.o a second time and failed
with duplicate symbols; removing that redundant build flag succeeded, without
source changes. Source manifests were captured before compilation and rechecked
after it; the metadata-only replay also passed. New provider binary SHA256:
`45ad10677acedf5174a577f45f7fae385b481fd29e8d4ec926742a8771fdd1e2`.
The Metal implementation and separate dirty FFN capacity experiment are
unchanged. Review verdict is ROBUST for the bounded instrumentation contract,
not for GPU stability. The runtime gate is reported separately below.

### Instrumented fusion-off replay: layer 7 call attributed

The single guarded replay with this binary still exited 1. First-call captured
tool/count checks passed (27 output tokens, 80,205.5 ms); second input identities
remained unchanged: prefix 7,839, suffix 196, helper rows 195, capacity 8,845.
No second output or full two-call parity certificate exists. The new records
localize this run as follows, with host wall times rather than GPU times:

| Boundary | Host ms | Result |
| --- | ---: | --- |
| Ordinary shared command, cursor 0 to 3 | 467.763 | wait returned |
| Routed full-attention call, layer 3 | 240.800 | call returned |
| Routed full-attention call, layer 7 | 113.584 | call failed |

The native exception again originates from the private full-attention helper's
wait and reports `Impacting Interactivity`. There are 122 full-layer begin/terminal
pairs, exactly one `call_failed`, and no unmatched identities. All nine ordinary
shared-command pairs end successfully: the added layer wrapper catches the
previously invisible failure. The checker verifies both pairings but rejects
the failed replay; trace completeness is not success.

The preceding uninstrumented fusion-off replay reached a layer-24 memory event,
whereas this replay fails at layer 7. Do not label layer 7 a deterministic bad
layer, blame a particular attention kernel, or infer a watchdog duration from
the failed call's elapsed time. Logging, rebuild and host scheduling are possible
confounders; this is attribution for one run, not causal isolation.

Observer: 45 samples, zero collection errors, free memory initially 76%, minimum
41%; no memory-floor kill. The existing 35% floor / 24-GiB process-tree cap /
300s timeout / 2,048-row chunks / group limit 1 / cooldown 50ms were unchanged.
The cooperating-client Metal lease spans both calls. Launcher wall time was
89,175.155 ms (the runner's approximate tick label was `~64s`). No retries or
further GPU workloads followed the failure.

Evidence root: `/private/tmp/qwen-layer-trace.s9ILiM/`; source manifests,
`run_inventory.py`, `layer-trace-{dry,gpu}` logs/results/samples and
`check_replay.py`. The launcher differs from the prior fusion-off launcher only
in binary/source-manifest identities and output paths; execution flags and
pinned session/model are retained. GPU command was
`python3 /private/tmp/qwen-layer-trace.s9ILiM/run_inventory.py --run`.

**Decision:** the diagnostic extension is verified for this scoped failure
attribution plus the model-free contracts. GPU stability remains unresolved;
no speed promotion. Next inspect a bounded stage-level discriminator inside
the standalone full-attention helper. Any stage splitting changes scheduling
and must be treated as a diagnostic intervention, not a drop-in fix. Refresh
after source/device/model/input/toolchain/scheduling changes; do not replay
merely to obtain a different failing layer.

### Standalone stage split: one successful two-call replay (2026-09-12)

The missing stage-split helper first produced a compile failure. After adding
the helper and its three guarded call sites, the combined model-free suite
passed 21 examples, with no failures/errors/pending:

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen-stage-split.NCS57S/spec-cache crystal spec spec/qwen_prefill_stage_split_spec.cr spec/qwen_prefill_command_trace_spec.cr spec/qwen35_sg4_tail_safety_spec.cr spec/metal_process_lease_spec.cr --no-color
CRYSTAL_CACHE_DIR=/private/tmp/qwen-stage-split.NCS57S/cpu-cache crystal build -Dcpu_only bin/qwen35_generate.cr -o /private/tmp/qwen-stage-split.NCS57S/cpu-probe
# From ../crystal_ball; its link annotation supplies the bridge.
CRYSTAL_CACHE_DIR=/private/tmp/qwen-stage-split.NCS57S/metal-cache crystal build scripts/cogni_qwen_two_call_probe.cr --release -o /private/tmp/qwen-stage-split.NCS57S/probe
python3 /private/tmp/qwen-stage-split.NCS57S/run_stage_split.py --dry
python3 /private/tmp/qwen-stage-split.NCS57S/run_stage_split.py --run
python3 /private/tmp/qwen-stage-split.NCS57S/check_stage_split.py
```

Both builds, formatting and diff checks passed. Fake commands cover each failed
stage, exact exception identity, commit failure without wait, and closed logging;
source guards cover bypass conditions, ended encoders and successor placement.
These are not GPU numerical tests. The source review found no hidden command
submission inside reachable `encode_matmul` helpers. Scratch survives all three
stages under the existing serialized route. The replay holds the Metal lease
across both calls and closes the provider on failure; low-level callers must
likewise discard failed state, because partial F32 KV writes are not rolled back.

The one GPU replay exited 0, with all 585 stage begin/terminal pairs nested
within 195 successful exact-layer call pairs. No stage failed or lacked a
terminal record. The first call matched the captured tool output and 27-token
count. The second input retained 8,035 prompt tokens, 7,839 cached tokens,
196 suffix tokens (195 full-helper rows), and capacity 8,845. It used
`resident_prefix_hit=1`, emitted no tool calls, and its content SHA256 matched
the saved successful baseline:
`ed251864987c367e9641fbdc89c1d83e9bf0fa2e3eecef8f301c79f619bfac81`.

Observed first/second provider wall times were 75,804.9 / 7,048.7 ms; second
prefill/top1 was 4,071.3 ms and decode body 2,904.3 ms. Launcher wall time was
90,650.485 ms, including the fixed 7.1-second captured-tool delay. The runner's
approximate `~66s` tick label is not the wall-clock duration. Do not compare
these diagnostic timings as a speedup against an earlier failed run.

Observer: 45 samples, zero collection errors, initial free memory 79%, minimum
48%, no guard kill or native GPU error. Controls stayed at 35% free-memory
floor, 24,576-MiB process-tree cap, 300-second timeout, 2,048-row chunks,
group limit 1 and 50ms cooldown; fusion and FFN capacity reuse remained OFF.
Quiet-host waiting remained disabled under the existing operator authorization;
this is not quiet-host benchmark evidence. No second GPU run was made.

Evidence root: `/private/tmp/qwen-stage-split.NCS57S/`, containing source
manifests, launcher/checker, `stage-split-{dry,gpu}` logs/results and samples.
Manifests were checked before execution and again afterward with no drift.
The measured checkout includes the separate dirty FFN experiment, explicitly
disabled at runtime; that WIP is excluded from the diagnostic commit.

- Provider binary SHA256: `cc179fcbb521309c25afb44f73d6a619529e403c2eab8f347636b41c6c0d22fd`.
- Measured Metal source SHA256: `430bfa229045e6a2cc0733bf3cd2de88323877b8af0cc5b4a75440ce74960633`.
- Stage helper SHA256: `2bb8a85dae2858530db53622c3d8722e75fda5620be4bd983f56cdfe963885b6`.
- GPU stderr SHA256: `0f0842fab0a91540198b8be7e14379db89b9d09f3f3618888205f94faea5e7a4`.
- GPU stdout SHA256: `50857cca738b084545c02593c04f29c8e0e75692bfed7f3db109e4c1804abca3`.

**Decision:** ROBUST for the default-off diagnostic contract and this one
output-replay pass, supported by local checks and correlated Luna source review.
Not a state-tensor parity certificate, production fix, deterministic kernel
localization or watchdog-causality proof. Splitting changes submission,
completion and host-encoding timing, including lazy pipeline setup placement;
host conditions and different prior binaries remain confounders. Keep defaults
unchanged. The next discriminator is a same-binary, same-input unsplit/split
comparison under the existing first-failure stop policy, not another kernel
rewrite. Only after reproducing a schedule-dependent difference should a
smaller two-stage cut be investigated. Refresh after source/device/model/input/
toolchain/scheduling changes or loss of the temporary evidence.

### Same-binary unsplit control fails; series stopped (2026-09-12)

The next diagnostic reused the exact successful stage-split binary above
(`cc179fcb...c0d22fd`), with the same pinned session, prompt/tool identities,
model path and literal runtime controls, except stage split OFF. Both metadata-only
arm checks passed. No rebuild or production-code edit was made. Source/provider/
binary manifests matched before execution and again afterward; model identity
was checked by device/inode/size/mtime, not a full weight-file digest. The binary
still includes the separate dirty FFN experiment, disabled at runtime.

The one GPU OFF arm exited 1. Call 1 matched the captured tool/count (27 output
tokens). Call 2 retained 8035 prompt / 7839 cached / 196 suffix / 8845 capacity,
then failed in `full_attn_chunk_routed`, layer 7, start position 7839, 195 rows:
`Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)`.
Layer 3 had completed; all 122 layer calls have terminal records, with zero
stage records as expected for OFF. There is no second-call output. The checker
correctly returns exit 1: complete failure attribution is not a replay pass.
The series stopped immediately; no new ON GPU arm or retry was run.

At second `prefill_enter`, both this OFF and the earlier successful ON report
identical inventories (apart from timestamp): 53 pipeline entries, 915 scratch
entries, 4,342,707,224 retained scratch bytes, 1076 live buffers,
22,950,743,664 live/peak buffer bytes and 22,955,573,248 Metal-allocated bytes.
Thus a difference in these recorded entrance inventories does not explain the
contrast. This does not measure hidden driver state, within-command pressure,
or exclude a resource mechanism elsewhere.

Observer: 49 samples, zero errors, initial free 76%, minimum 43%; no guard kill.
The 35% floor / 24,576-MiB tree cap / 300s timeout / 2048-row chunks / group 1 /
50ms cooldown, disabled fusion/FFN reuse, and process-spanning Metal lease were
retained. Quiet-host waiting stayed disabled under standing authorization.
Launcher wall time was 98,009.429 ms; the runner's `~71s` tick is not wall time.
First-call provider time was 89,537.8 ms. Neither it nor the 151.708-ms failed
layer elapsed time establishes a speed comparison or watchdog threshold.

Evidence root: `/private/tmp/qwen-stage-ab.kIMKDt/`; `off-gpu` logs/results/
samples, input-control snapshots, `off-check.json`, and launcher/checker controls.
Commands:

```sh
python3 /private/tmp/qwen-stage-ab.kIMKDt/qualify_check.py
python3 /private/tmp/qwen-stage-ab.kIMKDt/run_ab.py off --dry
python3 /private/tmp/qwen-stage-ab.kIMKDt/run_ab.py on --dry
python3 /private/tmp/qwen-stage-ab.kIMKDt/run_ab.py off --run
python3 /private/tmp/qwen-stage-ab.kIMKDt/check_ab.py off
```

The final two commands return 1 as expected for this negative diagnostic; do not
rerun them merely to reproduce a pass. Checker qualification accepts the prior
ON positive, rejects the prior failed OFF and wrong-arm trace, and rejects an
ON trace with a whole layer's stage triplet removed. Successful ON must contain
exactly three paired stages per paired layer call. These are trace/result checks,
not state-tensor parity checks.

- Launcher SHA256: `4e6a11df82f63e9a8ba0a484343e43250f0f053abd03273fc06176d5ce24dfe2`.
- Checker SHA256: `108cfdfdd15ac7465804bf4f6e8e7717578f0f26898c3810a18d815bd9d83664`.
- Qualifier SHA256: `1460a1383e484e6bc239949d3a9cbb9bffaa9204cc964d71190dc61b23a9c4ef`.
- OFF stderr SHA256: `7490e59e3dd0998c80494d90542cac7b1016791e5e46a9c58acd52745d3a9e7c`.
- OFF stdout SHA256: `347dcc34e0259ef1a4aa4fe2d59b9fad91f60f935fec405452542738b7d42502`.

**Decision:** ROBUST for the observed same-binary ON-pass/OFF-failure contrast
and stopping policy. The prior ON and current OFF occurred hours apart, not as
a completed contemporaneous OFF/ON pair. Rebuild differences no longer explain
this contrast, but host/time/order and driver state remain confounders. No
deterministic layer fault, root cause, stability fix, state parity or speedup
is established. Keep stage split default OFF. Next inspect the existing three
stage boundaries and pipeline-setup placement for a smaller diagnostic cut;
any new GPU experiment remains separate and must retain the first-failure stop.
Refresh on source/model/device/input/toolchain/scheduling drift or evidence loss.

### Stage-boundary inspection: narrow the cut, not the precision (2026-09-12)

No new GPU replay or production edit. Reanalysis of the pinned successful ON
stderr above pairs the enclosing layer trace with its three stage terminals.
At start 7839 / rows 195, coverage is 15 standalone calls, layers 3,7,...,59;
the final specialized full layer is not covered by this timing wrapper.

| Host commit/wait stage | Median ms | Sum across 15 calls, ms |
| --- | ---: | ---: |
| Prepare/KV | 4.860 | 71.513 |
| Attention | 159.103 | 2385.079 |
| Output/FFN | 16.322 | 244.986 |

Whole-layer time totals 2715.734 ms; subtracting the three waits per call leaves
0.721..1.412 ms (median 0.940) for setup, encoding, readback and trace overhead.
This is host-time accounting, not GPU utilization, and it does not measure OFF
internals. Earlier successful attention-only commands at start 6144 / rows1668
had median 606.643 ms and maximum 613.486 ms. A universal 150-ms host-wait cutoff
does not fit these observations; no native GPU watchdog threshold is inferred.
All nine recorded second-prefill memory samples retain 53 pipeline entries,
while scratch entries grow from 915 to 960. Do not confuse scratch shape keys
with compiled pipeline variants or generalize these samples to hidden driver
allocations.

The current cuts follow ended KV-write and attention encoders
(`qwen35_metal.cr`, `full_attn_layer_chunk_project`); no new tensor readback is
inserted. Each wait succeeds before successor construction. The native wait
path consumes the retained command handle (`device.cr`, `CommandBuffer#wait`;
`bridge.mm`, `gs_wait_command_buffer_status`), so splitting also changes when
the engine relinquishes its native command reference. This is not a measurement
of immediate driver resource reclamation. Scratch stays retained across the
cuts; equal buffer inventories do not establish equal transient driver resource
lifetimes.

**Next candidate, PROPOSED only:** retain the cut after attention and combine
prepare/KV with attention, leaving output/FFN separate. This removes one of two
extra boundaries without changing kernels, shapes, cache precision or operation
order. It tests whether the early KV-write boundary is necessary; it does not
by itself separate scheduler effects from command resource lifetime. Admission
would require default-off policy and fake-command tests before a separate
guarded replay with the existing failure-stop rule. A pass is not stability or
speed promotion; a failure does not prove attention itself is defective.

### Two-stage implementation: model-free gate (2026-09-12)

`after_attention` now retains the same unsubmitted command through PrepareKV,
then waits once for `prepare_kv_attention` before constructing `output_ffn`.
The helper factory snapshots the mode once. Existing `1` semantics and the
standalone ordinary-F32 gate remain unchanged; default is still OFF.

The new tests first failed because the factory did not exist. After implementation:

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen-two-stage-spec crystal spec \
  spec/qwen_prefill_stage_split_spec.cr \
  spec/qwen_prefill_command_trace_spec.cr \
  spec/qwen35_sg4_tail_safety_spec.cr \
  spec/metal_process_lease_spec.cr --no-color
# 25 examples, 0 failures, 0 errors, 0 pending
CRYSTAL_CACHE_DIR=/private/tmp/qwen-stage-split.NCS57S/cpu-cache \
  crystal build -Dcpu_only bin/qwen35_generate.cr \
  -o /private/tmp/qwen-two-stage.vlxfep/cpu-probe
# exit 0
# From ../crystal_ball:
CRYSTAL_CACHE_DIR=/private/tmp/qwen-stage-split.NCS57S/metal-cache \
  crystal build scripts/cogni_qwen_two_call_probe.cr --release \
  -o /private/tmp/qwen-two-stage.vlxfep/probe
# exit 0
```

Format checks for the helper, spec and Metal caller, plus `git diff --check`,
pass. Tests cover admission, retained command identity, paired stage records,
mode snapshot and original exception propagation for commit/wait failures in
both commands; source guards cover the conditional successor allocation.
Bounded Luna source review: ROBUST for this control flow, not independent
runtime evidence. Builds used the working checkout, including separate FFN
WIP that is excluded from this change's commit.

No GPU workload was run for this implementation. Stability, tensor parity and
speed remain open. Before a separate guarded two-call replay, refresh source/
binary/input identity and adapt the trace checker: require exactly two stages,
each with paired begin/terminal records per admitted standalone call, named
`prepare_kv_attention` and
`output_ffn`, with no separate `prepare_kv`. The old three-stage checker is not
a valid oracle for this mode. Retain fusion/FFN reuse OFF, the 35% free-memory
floor, 24GiB cap, 300s timeout and first-GPU-failure stop. Unset the option for
rollback. Refresh evidence after source, toolchain, device or input drift.

### Two-stage replay: one successful two-call diagnostic (2026-09-12)

One GPU attempt on implementation `23b6acf4`, using the previously compiled
release provider with the separate FFN WIP present but disabled. No production
source changed in this experiment. Source, bridge, runner, sampler, binary and
pinned input manifests were verified before and after execution; the model
identity is stat-based, not a full weights digest. Metadata-only output matches
the earlier three-stage dry run byte-for-byte (first prompt: 7,813 tokens).

```sh
python3 /private/tmp/qwen-two-stage.vlxfep/run_two_stage.py --dry
python3 /private/tmp/qwen-two-stage.vlxfep/run_two_stage.py --run
python3 /private/tmp/qwen-two-stage.vlxfep/run_two_stage.py --verify
python3 /private/tmp/qwen-two-stage.vlxfep/qualify_check.py
python3 /private/tmp/qwen-two-stage.vlxfep/check_two_stage.py
```

The GPU runner and sampler exited 0. All 195 instrumented standalone layer
calls returned, with 390 stage begin/terminal pairs: one
`prepare_kv_attention` and one `output_ffn` per call. A separate awk tally
agrees with these counts. The first call matched the captured tool output and
27-token count (asserted by the pinned probe before its call-end event).
The second used `resident_prefix_hit=1`, with 8,035 prompt / 7,839 cached /
196 suffix tokens and capacity 8,845; no tool calls were emitted. Its output
SHA256 matches the saved successful baseline:
`ed251864987c367e9641fbdc89c1d83e9bf0fa2e3eecef8f301c79f619bfac81`.

Observed first/second provider wall times: 76,713.4 / 7,013.2 ms; second
prefill/top1: 4,061.5 ms, decode body: 2,878.1 ms. Launcher wall was
91,208.225 ms, including the fixed 7.1-second tool-result delay. These are
diagnostic timings, not speed evidence: the earlier three-stage second call
was 7,048.7 ms, but the runs differ in time/order/build and are not balanced.
Fewer commands did not establish a material wall-time gain.

All guards were retained: 35% free-memory floor, 24,576-MiB process-tree cap,
300-second timeout, chunk 2,048 / group 1 / cooldown 50ms, fusion and FFN
reuse OFF. Quiet waiting remains disabled under standing operator authority;
no quiet-host claim. Observer: 46 samples, zero collection errors, initial
free 79%, minimum 45%; no guard kill or GPU failure. No second GPU attempt.

The two-stage checker returns `replay_pass=true`. Its qualification accepts
an explicitly synthetic, in-memory transformed copy of the older successful
trace, rejects the actual earlier failed replay, and rejects a missing stage
terminal, all stages missing from one successful layer, and the old three-stage
format even when relabeled. The synthetic positive tests the checker, not GPU
behavior. Parent reread and reran the Luna-authored checker and controls;
this is correlated review, not independent replication. The 25 model-free
stage/trace/tail/lease specs and `git diff --check` also pass.

Evidence root: `/private/tmp/qwen-two-stage.vlxfep/`, with launcher, manifests,
input controls, `two-stage-{dry,gpu}` results/logs and sampler records.

- Binary SHA256: `5daf2e7ee7535bac7156eb6b07705dc747cdec96ad1518206b1b95775de86bcd`.
- Metal source SHA256: `9ce459301e9a6a867ffbb64b648cca6c1ca2f629dcbdb6381acfdcc6aa22dd13`.
- Helper SHA256: `ba0002d15cf3c8972b66dd1d8628418e77e64a65e7a3117d6c9151a4a0218768`.
- GPU stderr SHA256: `db1a3c814abf0942ed6302fc120f71254f0d843d780b721ca65e5760765bc95d`.
- GPU stdout SHA256: `eb875dc83f3ad023ab69d19a6176dd7cede6e42f76443728f3eca5759b2dceae`.

**Scoped verdict: ROBUST** for the observed two-call output/trace pass and guard
record, not a production fix. The earlier PrepareKV boundary is not necessary for this
one successful observed replay. That does not establish that the remaining
boundary prevents the intermittent error, or identify attention as its cause.
No full hidden/KV/recurrent-state comparison was performed. Keep the mode OFF
by default; do not combine this scheduling result with the separate FFN reuse
experiment. Next use existing traces to choose any further discriminating
experiment, rather than promote or immediately repeat this run. Refresh on
source/model/device/input/toolchain/scheduling drift or evidence loss.

### Matched-shape trace inspection: select an operator probe (2026-09-12)

Read-only follow-up to the two-stage replay; no new GPU run, kernel or policy
change. `python3 /private/tmp/qwen-two-stage.vlxfep/compare_stages.py` verifies
the three pinned stderr digests and matches the 195 successful layer keys
`(start_pos, rows, layer)` between the two split captures. Coverage below is
15 ordinary full-attention layers per shape, not every model operation.

| Host-time accounting, sum over 15 layers | Three-stage | Two-stage |
| --- | ---: | ---: |
| Prefix7839 / rows195: enclosing calls | 2715.734 ms | 2714.097 ms |
| Same shape: prepare + attention waits | 2456.592 ms | 2454.882 ms |
| Same shape: output/FFN waits | 244.986 ms | 245.233 ms |
| Same shape: enclosing time outside stage waits | 14.156 ms | 13.982 ms |
| Prefix6144 / rows1668: enclosing calls | 11420.761 ms | 11430.525 ms |

The three-stage attention-only sum for rows195 is 2385.079ms, about 87.8%
of those enclosing calls, not 87.8% of full request time. Removing the early
boundary produces no material observed time difference. These host waits do
not isolate GPU execution or establish a kernel-speed ranking. OFF supplies
no internal stage timing: it completes only one suffix layer before layer7
fails, so its truncated suffix cannot be compared as a completed workload.

All nine suffix inventory snapshots in the two successful captures match
after removing timestamps: stage/position/layer, pipeline entries, Scratch
entries/bytes, live/peak Metal buffers and device-allocated bytes. All three
captures also match at suffix entrance. Pipelines remain53, Scratch entries
915/959/960 in both successful suffixes. This narrows the measured cache-growth
hypothesis, not hidden driver allocation, command lifetime or watchdog cause.

Source-resolved route under captured controls: `qwen35_metal.cr` SG4 policy
(default direct-gate minimum1024) and the standalone attention dispatch select
F32 `qwen35_attn_decode_rows_sg4_pregate` for rows195, but direct-gate
`qwen35_attn_decode_rows_sg4` for rows1668. Prefix length is absent from this
threshold. This is inferred from pinned source/config, not a kernel-name event
in the capture. LM-414 retains the historical short-pp64 direct-gate regression/
noise caveat; a short continuation after 7839 cached tokens is a different,
unmeasured regime, not grounds to remove that guard.

**Next discriminator, PROPOSED:** a no-model F32 operator comparison using the
existing two SG4 kernels, identical Q/gate/K/V inputs, heads24/KV4/D256 and
prefix7839/rows193..196, plus prefix0/rows64 as the historical short control
and prefix0/rows195 to isolate prefix length at the observed continuation size.
Reuse the corrected partial-SIMD-group synchronization and CPU oracle/canaries
from `bin/qwen35_sg4_tail_probe.cr`. Require finite output, oracle agreement and
direct/pregate equality before warmed, balanced ABBA timing. No global routing
change, precision conversion, extra padding or command-overlap experiment.
If the candidate is slower, inconclusive or fails correctness, keep the current
route and stop this candidate; do not widen the benchmark until it looks good.
The direct-gate minimum override affects all eligible chunks, not just rows195;
do not describe a whole-session `MIN=1` replay as a single-shape intervention.
Only a repeatable operator win would justify a separate two-call output/state
gate. Retain the existing first-failure and memory/time guards.

Flash MMA is not a drop-in control here: its present admission requires F16
KV, and the prefix model-state discrepancy remains recorded in
`qwen35-engine-frontier.md`. Do not change representation or relax that gate
to accelerate this F32 diagnostic. The current three SG4 source-safety specs
pass; source/binary manifests remain unchanged. Scoped verdict: ROBUST for
trace accounting and route selection, not a causal fix or speed claim.
Comparison script SHA256:
`56c4b184671febc56e1aa13e188bcb178d0331c24de68cb83d514e6ab8ea2d7e`.
Refresh after source/config/input/device/toolchain drift or evidence loss.

### Bounded F32 SG4 operator experiment (2026-09-14)

Admission is diagnostic only: extend `bin/qwen35_sg4_tail_probe.cr` with an
explicit benchmark mode, preserving the old dry/self-test and 48-case regression
mode. No inference policy, shader, precision, allocation layout or defaults
change. Rollback is removal of the benchmark mode, not a production switch.

Predeclared comparison: heads24/KV4/D256; prefix7839/rows193..196 and
prefix0/rows64,195. Direct and pregate share input/output allocations within
each shape. Before timing, re-poison output and require both kernels to satisfy
the Float64 CPU oracle (max absolute error <=1e-5), trailing write canary,
finite output and mutual max absolute difference <=1e-6. GPU-free injected
output/canary defects must be rejected by the same validation code.

Warm both kernels, then record three fixed ABBA blocks per shape (72 timed
commands total). Each sample is one completed command, not a batch with hidden
overlap. Collect completed-command Metal GPU intervals and host encode/commit/
wait time separately; exclude compilation, CPU oracle and buffer initialization
from operator timing. Use the same synthetic data for both routes, and report
the data/model generality limit rather than equating this with inference speed.

One guarded benchmark attempt: `scripts/run_safe.sh`, 300s/24576MiB/35% free,
shared Metal process lease, terminal first error, no retry or guard relaxation.
Quiet waiting stays disabled under standing operator authority; do not claim a
quiet host. A failed correctness gate stops timing for that shape; inconclusive
or negative speed evidence does not admit a lower production threshold.
Even a consistent operator win requires a separate model output/state gate.

Implementation checks: Metal release build and CPU-only build pass. Both
`--self-test` binaries accept clean output and reject two source mutations plus
five validation defects (NaN, canary overwrite, unwritten sentinel, wrong finite
output, pair mismatch), without Metal initialization. The CPU-only build
rejects GPU modes. The 12 SG4-tail/stage-split specs and format/diff checks pass.
The original 48-case GPU regression mode remains available but was not rerun.

Executed once:

```sh
env -u COGNI_METAL_LEASE_PATH COGNI_METAL_LEASE_WAIT_MS=0 COGNI_RUN_SAFE_REQUIRE_QUIET=0 COGNI_RUN_SAFE_WAIT_QUIET_SEC=0 COGNI_RUN_SAFE_MIN_FREE_PCT=35 COGNI_METAL_COMMAND_TIMEOUT_MS=180000 scripts/run_safe.sh /private/tmp/qwen-sg4-operator.wYIRes/probe 300 24576 --benchmark
```

**Result: measured-red, exit1 after approximately2s.** Apple M2 Max, preflight
free77%; no memory-floor kill or retry. At prefix7839/rows193 both initial
outputs have max oracle error `1.080809547859829e-6`, mutual difference0 and
intact canaries. Six timed samples also pass. Their observed GPU intervals
span115.443–216.849ms; these are truncated observations, not a kernel ranking.
The next command raises completion_status=-6, native status5/error_code1,
`Impacting Interactivity`. From fixed loop order and the dispatch stack, the
next command is inferred to be pregate/block1/order2; the log does not contain
a pre-submit kernel event. No shape summary or final benchmark PASS is emitted.

The pinned shader is unchanged, both pipelines are compiled before the loop,
and only one shape was reached. This reproduces the callback without model
weights, a provider session, or increasing pipeline variants within the loop.
The fixture's five explicit Metal buffers sum80,125,952B (~76.414MiB), computed
from allocation sizes, not total driver/GPU memory telemetry. CPU arrays,
pipeline/driver resources and other applications remain outside that number.
This narrows necessary conditions; it does not prove gate staging, memory
pressure or command duration is the cause, nor rule out host/driver history.

Source/bridge/binary/runner manifests match before and after. The independent
capture checker rejects the actual incomplete run; its synthetic72-row parser
fixture and six negative controls pass (parser qualification, not GPU evidence).
Bounded correlated Luna review finds the runner ROBUST as a diagnostic, not a
successful benchmark. Per-sample `pair_max_abs` means difference from the
initial validated pregate output, not a fresh paired sample; shape-summary
oracle fields are preflight values, not aggregate timed statistics.

Next, PROPOSED only: prove query-row slicing with identical F32 Q/gate/K/V and
global causal bounds. Existing buffer-offset binding can advance Q/gate/output
while keeping K/V unchanged and advancing base_pos; a <=64-row command is a
candidate, not an admitted policy or known fix. Require oracle/canary equality
and explicit progress traces before a separately guarded experiment. Keep the
production threshold, precision, stage-split defaults and all guards unchanged.

Evidence root `/private/tmp/qwen-sg4-operator.wYIRes/` (temporary):

- Probe source SHA256: `50fede1e4c3a2ccb93360038a909ca9482019d86a8a02b6f007c63d12e32c47f`.
- Probe binary SHA256: `ec9d228e33e5b4091f3214cfc22e27e49c55a6af606ec67d56b1616922953083`.
- Combined log SHA256: `3edce19b1d082545b8184be13b6985398292d802a489118fe68b41e8d683af7f`.
- Checker SHA256: `e1fc182eb93f20579fd0bd22186a6b93ddf77e4f32db1179f1e0b5eb793eb07e`.

Refresh on source/build/device/driver/input/scheduling drift or evidence loss.

### Query-row slicing diagnostic (2026-09-14, predeclared)

Hypothesis: reducing each SG4 command to at most64 query rows can preserve
F32 attention results while shortening individual GPU intervals. It is not a
root-cause claim: aggregate GPU load, driver history and host scheduling remain
alternatives. Add diagnostic-only `--slice-check`; no production changes.

For each existing six-shape benchmark fixture, run direct then pregate once.
Bind Q/gate/output at `row_start * 24 * 256 * 4` bytes, keep K/V at zero,
and pass `base_pos + row_start` with the slice row count. The local causal end
then equals the original global end. All slices are synchronous; poison once
per kernel sequence. Check CPU oracle, unchanged completed prefix, and unwritten
future rows plus trailing canaries after every slice. Emit flushed pre-submit
identity and post-validation events. No added cooldown or retry.

DoD: CPU-only/Metal builds and self-tests pass, including slice coverage/offset
checks and seeded premature/overwritten output rejection; existing12 source
specs remain green. Then one guarded GPU attempt using the same35% free,
24GiB/300s and zero-wait process lease controls. First error is terminal. A
complete capture must contain all42 slice submit/completion pairs,12 kernel
checks,6 shape pairs and finalPASS. These are correctness diagnostics, not
balanced speed samples; per-slice readback/validation changes host pacing.
Rollback: remove this opt-in probe mode. Even all-pass does not admit a
production scheduling policy or prove the watchdog cause.

#### Observed result and decision

Instrument checks pass: the new self-test initially failed at missing
`query_slices`; CPU-only and Metal release builds now pass. Both self-tests
cover16 slice plans, four injected slice-output defects, and the existing
seven source/output controls without Metal initialization. The12 SG4/stage-split
specs, format and diff checks pass; CPU-only rejects `--slice-check`.

The sandbox launch aborted with exit75 because `ps` was prohibited and the
runner could not verify process-group isolation. Its log contains no Metal
initialization or submission records. An authorized outside-sandbox launch
used the identical binary and guards, without bypassing containment:

```sh
env -u COGNI_METAL_LEASE_PATH COGNI_METAL_LEASE_WAIT_MS=0 COGNI_RUN_SAFE_REQUIRE_QUIET=0 COGNI_RUN_SAFE_WAIT_QUIET_SEC=0 COGNI_RUN_SAFE_MIN_FREE_PCT=35 COGNI_METAL_COMMAND_TIMEOUT_MS=180000 scripts/run_safe.sh /private/tmp/qwen-sg4-slices.XIoCjM/probe 300 24576 --slice-check
```

**GPU result: measured-red, exit1.** Apple M2 Max, preflight and postrun free78%,
no guard kill or added cooldown. Direct prefix7839/rows193 completes slices
64+64+64+1, with GPU intervals78.256/50.278/54.247/25.990ms. Every slice passes
the oracle, exact earlier-prefix comparison, future sentinel and trailing
canary. The complete direct output max error is `1.080809547859829e-6`.
These intervals are diagnostic observations, not balanced performance data.

The next pre-submit event identifies
`qwen35_attn_decode_rows_sg4_pregate`, start0, rows64, base7839, offset0.
It fails with native status5/error_code1, completion_status=-6,
`Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)`.
There are5 submit events,4 completed/validated slices and1 completed kernel;
no shape pair or finalPASS. No further GPU attempt was made. The complete-run
checker correctly rejects this capture; its synthetic42-command fixture and
seven mutation controls qualify only the parser, not GPU behavior.

Decision: reject64-row slicing as a sufficient prevention policy on this
observed route. Nonzero binding offset and partial SG4 tail are not necessary
properties of the failing command; prior direct work/driver history can still
matter. Do not conclude that pregate itself is defective: it runs second, and
both kernels succeeded before the earlier unsliced failure. The live GPU result
certifies only the tested direct output, not all six shapes, pregate slicing,
production state, adaptive cache behavior, or overall stability/speed.

Next proposed discriminator: single-kernel, single64-row-command fresh-process
fixtures, with order explicitly counterbalanced and the same first-failure
stop rule. Design and source review come before any further GPU workload.
Do not keep reducing slice size or introduce sleeps as an unlabelled fix.

Evidence root `/private/tmp/qwen-sg4-slices.XIoCjM/` is temporary; the unchanged
pre/post manifest covers probe, shader, buffer/Metal wrappers, bridge, runner
and binary. Refresh after source/build/device/driver/input/scheduling drift or
evidence loss. No production source was changed by this slice.

- Probe source SHA256: `ee895551d0e1d32516a473bf777763dd8a9b1e715228e90a83b998145802d683`.
- Probe binary SHA256: `845972bcb9f2bc139de90f61cf2107f8681a02b6b508dd03b42af858c3dd7d5b`.
- Authorized GPU log SHA256: `25e1383f9ea53409973ae1a5f69639236374ec4df323ac4e08301306bd8dc6c6`.
- Sandbox abort log SHA256: `8706aaef203ca35d880451dfb1d25e5d2f4a8f686d711ed9ceb0c76a04a10494`.
- Capture checker SHA256: `46036b46ccbb821912e5ea85240c12045c49353815199afb5cdc8a24cd87e58e`.

### Fresh-process single-command discriminator (predeclared 2026-09-14)

Hypothesis: the pregate failure depends on prior dispatches in its process.
Prediction: its first64-row command succeeds in a fresh process. A failure
before any direct dispatch refutes the need for that process-local predecessor,
not driver/host history or all accumulation effects.

Add diagnostic-only `--single-command=direct|pregate`. Keep the exact previous
prefix7839/rows193 F32 fixture and compile both pipelines in direct/pregate
order, then submit only selected kernel/start0/rows64 once, without warmup.
Validate the64-row output prefix against the CPU oracle and require all129
future rows plus trailing canaries to remain poisoned. Record selected kernel,
PID, pre-submit identity, completed-command timing, oracle and guard outcome.
No production source, shader, shape, precision, or allocation-layout changes.

Predeclare at most four fresh guarded processes in order pregate/direct/direct/
pregate (BAAB). First nonzero exit, missing completion or failed validation
stops the entire series; no alternate-kernel fallback or retry after failure.
Retain35% free memory,24GiB/300s, command timeout180s, zero-wait process lease
and disabled quiet wait under standing authority. No added sleeps. Launch
outside the sandbox so the unchanged runner can inspect process groups.

Instrument DoD: CPU/Metal builds, strict selector negative controls and existing
self-tests/12 source specs pass; capture requires exactly one submit and one
validated completion per admitted process. GPU DoD is conditional: all four
completed commands for a bounded success, otherwise record measured-red and
stop. Even four passes cannot establish production stability or speed; fresh
processes retain OS/driver history and setup changes host pacing. Rollback is
removal of the opt-in mode. A pregate-first failure requires a frame change
away from repeatedly shrinking row batches.

#### Result: process-local dispatch history is not necessary

CPU-only and Metal release builds pass. Self-tests initially failed at the
missing selector and now pass both valid selectors, six malformed selectors,
16 slice plans and the11 existing source/output mutation controls. The12
SG4/stage-split specs, format/diff checks and invalid-mode CLI checks pass;
CPU-only rejects the GPU mode. Correlated Luna source review found no P1
instrumentation blocker (ROBUST within this diagnostic contract).

The external launcher used `set -e` around each guarded process and its capture
checker, in the predeclared BAAB order. Exact per-trial invocation:

```sh
env -u COGNI_METAL_LEASE_PATH COGNI_METAL_LEASE_WAIT_MS=0 COGNI_RUN_SAFE_REQUIRE_QUIET=0 COGNI_RUN_SAFE_WAIT_QUIET_SEC=0 COGNI_RUN_SAFE_MIN_FREE_PCT=35 COGNI_METAL_COMMAND_TIMEOUT_MS=180000 scripts/run_safe.sh /private/tmp/qwen-sg4-single.uIaps6/probe 300 24576 --single-command=pregate
```

Replace only selector with `direct` for trials2/3. All trials used Apple M2 Max,
prefix7839, full fixture193 rows, selected first64 query rows and distinct
process IDs. Preflight free76% on all four; postrun free76%. No guard kill,
warmup, sleep, model weights or production inference session.

| Trial | Kernel | PID | Result | Completed GPU interval |
|---|---|---|---|---|
| 1 | pregate | 22637 | oracle/guard PASS, exit0 | 110.453ms |
| 2 | direct | 22691 | oracle/guard PASS, exit0 | 57.823ms |
| 3 | direct | 22767 | oracle/guard PASS, exit0 | 59.716ms |
| 4 | pregate | 22821 | first dispatch fails, exit1 | unavailable |

Each successful64-row prefix has max oracle error `1.064646237225464e-6`;
all129 unwritten rows and trailing canaries remain intact. Trial4 emits one
pre-submit record for pregate/start0/rows64/base7839/offset0, then native
status5/error_code1, completion_status=-6, `Impacting Interactivity` with the
same callback code `0000000e`. There is no result/PASS record for that trial.
The launcher stops with exit1; no further GPU run follows.

The checker accepts the three complete captures and rejects trial4; its eight
mutation controls are parser qualification only. Source/bridge/binary/runner/
checker manifests match before and after. Timings are cold first-command
diagnostics from an incomplete balanced series, not a throughput comparison or
evidence to change the production gate threshold.

Decision: a direct predecessor or multiple dispatches in the *failing process*
are not necessary conditions. Fresh process isolation is insufficient on this
host. One successful and one failed pregate run also prevents an unconditional
"this shape always fails" claim. OS/driver history, compile-order effects and
kernel-resource sensitivity remain open; no specific root cause is established.
Next is a read-only kernel/resource-boundary audit, not more identical runs,
smaller query slices or unlabelled cooldown changes.

Read-only next-step anchor: `fullattn_qwen35.metal` declares4608B of static
threadgroup arrays in direct (Q+scores) and8704B in pregate (Q+gate+scores).
These are source sums, not compiled allocation or measured occupancy. The
bridge currently exposes `maxTotalThreadsPerThreadgroup` but not
`staticThreadgroupMemoryLength`/`threadExecutionWidth`. A separately reviewed
metadata-only pipeline probe, with no compute dispatch, can establish the
compiled properties before considering a resource-reduction change. Equal
metadata would not rule out register pressure or driver/history effects;
different metadata alone would not prove watchdog causality.

Evidence `/private/tmp/qwen-sg4-single.uIaps6/` is temporary. Refresh on source,
build/device/driver/input/scheduling drift or evidence loss. SHA256:

- Probe source: `704beb574b2f3eee309bc675b5ae949956d6718faf4994fb3d36c992d4c3d557`.
- Probe binary: `4e2d7c00c4bca18c9fdf28ab29b1e878928ecb41d5dd5bf867e658ea20914190`.
- Checker: `d5408da220045c961afb2c3cff9cbca91cdd5185e8f3e42099982fac6a593dc2`.
- Trial1 log: `07d9249a1147f2f3af611ae8b8d17c8eef5d26560bf0927c707afc9de9d6879b`.
- Trial2 log: `b31c3e376b02d855416f185ccb63e0d5d93555a9d98ad9aee5adaa4b9e08cf10`.
- Trial3 log: `55a8df85e5bdd8cf25b4ec6f22ebbaca66f081c9739042232aef9667e424958b`.
- Trial4 log: `6caaa461f1d1e6fbaf4063b0609ece97dadbe93bc187c2e67c6a19992141fece`.
