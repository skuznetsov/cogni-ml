# Ordinary prefill command diagnostics

## Active diagnostic frontier: standalone stage split

Default-off diagnostic, not a production fix: `QWEN35_FULL_PREFILL_STAGE_SPLIT=1`
splits only standalone ordinary F32 full-attention into synchronous
`prepare_kv`, `attention`, and `output_ffn` commands. Shared, adaptive and F16
routes remain unchanged; unset or other values preserve the original command.
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

Current evidence: the single 2026-09-12 guarded split replay completed both
calls and matched the saved second-output digest. A later OFF control using
the same binary failed on call 2; the series stopped. This removes the rebuild
confound, not host/time/order differences. The read-only stage inspection below
narrows the next discriminator; production stability, causal attribution and
any speed claim remain open.

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
