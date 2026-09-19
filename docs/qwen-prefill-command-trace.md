# Ordinary prefill command diagnostics

## CPU scheduling interval discriminator (2026-09-19, predeclared)

Read kernelStartTime/kernelEndTime only after the existing completion, under
the same split-submit opt-in. Apple defines their difference as the interval
the CPU/kernel spends scheduling the command buffer, not GPU execution:
https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelstarttime
https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelendtime
No new callbacks/waits, allocations, ABI, inference route or default changes.
Emit raw pair and validity; finite positive endpoints with end>=start admit
resolution-limited zero duration, while zero endpoints are unavailable.

Prediction: if scheduling duration is comparable to the first ~7.5s pre-GPU
gap, CPU scheduling is implicated as an elapsed interval, not proof of active
CPU work or compilation. If it is tiny, that reported interval does not
account for the pause. Retain possible overlap: do not assert kernelEndTime
precedes GPUStartTime, subtract cross-family endpoints, or label their
duration difference a distinct phase. No common epoch is assumed for these
new endpoints. Missing/invalid data leaves this discriminator inconclusive.

DoD: native positive/zero-duration/invalid pair tests and fake completion
success/failure pass; fresh bridge/release build, CLI/dry and trace specs pass.
One guarded7839-token run at most, all64 ordered trace/native/timeline/
schedule/boundary groups with valid finite intervals and preserved prior
Mach accounting. Reject missing/failed/nonfinite/reversed pair mutations.
Sources except declared bridge/spec changes remain pinned; same model/tokens/
F32 capacity8036/chunk2048/group1/cooldown50/graph0. Keep startup70/runtime30%,
24GiB,300s/180s,lease0,quiet bypass; no retry or extra GPU workload. Artifacts:
`/private/tmp/qwen-prefix-schedule.GmVlc7/`. Rollback removes this opt-in pair
log/helper only. No stability fix, quality or throughput claim is admitted.

### Result: scheduling itself has a long first-command interval

One guarded 7839-token attempt completes all 64 ordered trace/native/timeline/
schedule/boundary groups. First-command scheduling is 7836.884ms, pre-GPU
7672.387ms, GPU execution 1417.449ms, and post-GPU return 0.109ms. These are
overlapping intervals, not additive phases. Scheduling is calculated only
from its own endpoint pair; no cross-family clock subtraction is admitted.
Native commit is 0.032ms and wait 9089.956ms; enclosing host is 9090.866ms,
Mach host bounds 9089.945ms, and separate encode 652.537ms.

| Chunk start | First scheduling ms | First pre-GPU ms | First GPU ms |
| --- | ---: | ---: | ---: |
| 0 | 7836.884 | 7672.387 | 1417.449 |
| 2048 | 21.398 | 2.117 | 1587.277 |
| 4096 | 20.744 | 1.520 | 1872.102 |
| 6144 | 95.025 | 1.696 | 1827.058 |

Other 63 scheduling intervals total 271.132ms (maximum 95.025ms). The first
reported CPU/driver scheduling interval is itself anomalously long and
comparable to the pre-GPU pause. This is elapsed scheduling, not active CPU
time, proof of compilation, a complete root cause, or a speedup/stability fix.

Native pair tests red then green; separate fake-command stderr check verifies
success valid=1 and failed-command valid=0 with raw endpoints retained. Fresh
bridge/release build, metadata dry-run, 22 CLI rejection controls and 11 trace
specs pass. Offline checker passes 64 groups and rejects 17 evidence mutations;
separate raw arithmetic and source/model/binary identity checks agree.
Correlated Luna source review: ROBUST for the bounded opt-in diagnostic.
Exit0/observer0; startup78%, minimum44%, 41 clean memory samples, final tree
absent. No Metal failure, guard kill, timeout or retry. Launcher wall83.218s
is not pp/tg or matched speed evidence. This attempt is consumed.

Next: inspect weight residency as a discriminating hypothesis. Our
`Qwen35Metal.register_mmap` wraps model mmap as one shared no-copy buffer;
`create_buffer_no_copy_impl` does not explicitly request residency. Current
local llama.cpp `ggml-metal-device.m:ggml_metal_buffer_rset_init` conditionally
creates a residency set, adds allocations, commits and requests residency,
with corresponding release lifecycle. Llama can also use one large no-copy
buffer; buffer splitting alone is not the distinguishing fact. Audit that
lifecycle and design a bounded test before changing it. Measure total cold
load-to-output time, not only a shortened first submit: moving preparation
earlier is not removing it. No broad warmup or weight pinning is admitted yet.

Reproduction: artifact directory above, `python3 check.py` for offline evidence;
native spec is `spec/metal_submit_profile_test.mm`, trace spec is
`spec/qwen_prefill_command_trace_spec.cr`. Refresh on source/model/input/device/
toolchain drift or temporary evidence loss. SHA256:

- manifest: `d879b1beb2de86e41a23c821a9df34369aa9d0dfc21415794365142ae9f6d73a`
- bridge: `c6bc885cdca873ad43836fa6a27ad49f6a54be526f1f614c8d3847585fcb7b80`
- binary: `712f196f4c869dbc4a6bf617962fe7668589a3b0dd553e39a4477bc2c0568e45`
- checker: `b65b790bc1088840d185771db4d67b34c4ae6c6db109df645029df43aff9fe6d`
- stderr: `32337d9d3ed831892b1f3ec4923909a8b0af9cae52646b649f0650a9cf1b4487`

## Mach timeline discriminator (2026-09-19, predeclared)

Extend the existing opt-in split-submit diagnostic only: record host Mach
seconds immediately before commit and immediately after wait returns, then
read GPUStartTime/GPUEndTime after completion. Apple documents GPU timestamps
as seconds relative to system Mach time:
https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime?language=objc
Convert mach_absolute_time with mach_timebase_info; do not subtract watchdog
CLOCK_MONOTONIC timestamps from Metal values. No callbacks or extra waits.

Prediction: start-minus-before dominates if the gap precedes GPU execution;
after-minus-end dominates if it follows execution. The latter includes host
thread wakeup/scheduling and is not notification latency alone. The former
includes the small commit interval and does not isolate driver compilation,
queue contention, memory residency, or other causes. GPU end-minus-start is
execution interval, not kernel occupancy. Preserve raw signed differences.

DoD: native synthetic before/after cases and invalid timestamp controls pass;
fresh private bridge/release probe, unchanged-source identity, metadata dry
and CLI checks pass; at most one guarded 7839-token attempt. Require all64
trace/native/timeline/boundary records, finite positive Mach endpoints, strict
GPU duration and enclosing bounds (1ms uncertainty, no clamping). Missing,
invalid or mismatched evidence blocks attribution, not inference execution.
Preserve startup70/runtime30%,24GiB,300s/180s,lease0,quiet bypass and the prior
model/input/F32 capacity8036/chunk2048/group1/cooldown50/graph0. No retry.
Artifacts: `/private/tmp/qwen-prefix-mach.HZ3gZn/`; rollback removes only
the opt-in Mach fields/helper/log. Uninstrumented stability remains open.

### Result: the first-command gap precedes GPU execution

One attempt completes all7839 tokens and64 ordered trace/native/timeline/
boundary records, all timeline-valid. Same model/input/source identities
(except declared bridge/spec instrumentation) and controls as the prior run.

| First command interval | ms |
| --- | ---: |
| Before commit to GPU start | 7566.682 |
| GPU start to GPU end | 1324.569 |
| GPU end to return from wait | 0.138 |

Native synchronous commit is0.031ms. Enclosing host submit/wait8892.274ms
includes logging/FFI/bookkeeping outside the Mach interval8891.389ms; encode
99.331ms is separate. Do not add overlapping native and GPU intervals.
For the next chunks, first pre-GPU intervals are1.403/3.938/1.227ms and
post-GPU0.158/0.157/0.139ms. Across all64 commands, post-GPU sums10.115ms
(max0.221ms); pre-GPU excluding the first totals53.557ms.

This falsifies a dominant post-GPU completion-return tail for the observed
first-command delay. It localizes the pause before GPU execution, not to a
particular driver subsystem. Queue waiting, CPU scheduling/preparation,
resource residency and compilation remain competing explanations. The
uninstrumented failure and any general stability/speed claim remain open.

Native test red (missing helper/fields) then green; release build, metadata
dry-run,22 CLI rejects and11 trace specs pass. Synthetic timestamps exercise
both gap positions and reject missing/nonfinite/misordered endpoints. Fake
FFI success still uses timestamps1.0/1.001 and consequently emits valid=0;
it checks completion behavior, not a valid same-epoch stderr schema. The live
checker requires valid=1 for every quartet, agrees with GPU duration and
native elapsed intervals, and rejects12 missing/failed/nonfinite/misaligned
record mutations. Separate raw-log arithmetic and pinned-hash checks agree.
Correlated Luna source review: ROBUST within this opt-in single-submitter scope.

Exit0/observer0, startup80%, minimum46%,38 clean samples, final tree absent.
No Metal failure, memory kill, timeout or retry. Launcher wall77.023s is not
pp/tg or comparable speed evidence. Preparation initially hit sandbox process
group isolation (before workload) and an Xcode SDK TAPI incompatibility in the
standalone trace test. The guarded build ran outside that sandbox restriction;
trace specs passed with the same Command Line Tools SDK used for the build.
No safety thresholds were changed. Aborted build-only artifacts are retained
at `/private/tmp/qwen-prefix-timeline.pf2KbP/`; no GPU attempt occurred there.

Next narrow discriminator: inspect Metal kernelStartTime/kernelEndTime CPU
scheduling semantics and, if usable, read them after the existing completion
without new waits. A long scheduling interval still would not uniquely prove
compilation. Avoid blind replay or a broad memory warmup until its causal
question, memory budget and comparison are explicit. Current attempt consumed.
Refresh on source/model/input/device/toolchain drift or temporary evidence loss.
Artifact directory above; SHA256:

- manifest: `5f5f315fa91a1737b6e83a52e7e3fb2e2d56c31a99682938f3da9a9d7128cc42`
- bridge: `1c7ddcdf2a82fffa5564d1bc8effbd4f1a34516053797b0824e3bbb5a9b9be09`
- binary: `f6a4ebd2d24baa180ca893edce2f4a31309f6fe86f2fcdb7accba1cd3fed0d41`
- launcher: `0ef427a94c362f82f340f34b9d1e4a458a90e1fd89b921d5f0fe6cf785c810da`
- checker: `c4bffed18906d20e25fa8f9ea904570f7f7e23748182747ae18bb076fe209128`
- stderr: `304599c64bf36da380ed2948f883444d530cbc055adce81b7909f60ff0cc9f69`

## Native commit/wait discriminator (2026-09-19, predeclared)

Extend prefix profile with optional `--split-submit`, allowed only after
`--prefix-profile`. Set exact opt-in `COGNI_METAL_SUBMIT_PROFILE=1`; explicitly
clear it in all other probe modes. Only the bridge's GPU-timed commit/wait
entry point admits the native profile. No ABI, shader, routing or extra wait.
Record monotonic host setup (timeout/watchdog registration), commit, completion
wait and retire (watchdog unregister) durations; print after the wait. Preserve
watchdog coverage of commit and wait, completion status and native ownership.
GPU execution can overlap both host phases: do not subtract the GPU interval
from wait alone or call a host phase pure compilation/CPU execution.

Competing predictions: a dominant commit interval localizes the extra delay
inside synchronous submission; a short commit plus long wait localizes it
after submission returns; a long setup/retire interval points to host-side
watchdog bookkeeping instead. None uniquely establishes driver compilation or
the cause of the earlier intermittent Metal failure.

DoD: qualify split intervals using a fake native command with separate seeded
commit/wait delays, status failure, disabled profiling, already-committed wait
and active watchdog-registration assertions (no GPU). Build a fresh private
bridge and probe, run CLI/dry and trace specs, then at most one guarded actual
7839-token prefix. Require 64 ordered trace/native/profile triples, matching
completion, positive GPU intervals and bounded accounting residuals; missing
or failed records are not zero-cost results. Preserve model/tokens/F32 capacity,
chunk2048/group1/cooldown50/graph0, startup70/runtime30%,24GiB,300s process,
180s watchdog,lease0,quiet bypass. Stop on first failure; no retry. Artifacts:
`/private/tmp/qwen-prefix-submit.646NF9/`. Pin all other production sources
against the preceding profile; explicitly repin the changed bridge and probe.
Rollback is removal of this opt-in instrumentation. Long warm timing and
uninstrumented stability remain open regardless of a profiled completion.

### Result: submission returns quickly; the delay remains in completion wait

One admitted attempt completes all 7839 prefix tokens and 64 paired
trace/native/boundary records. First command, chunk0/rows2048/cursors0->7:

| Native host phase | Duration ms |
| --- | ---: |
| Timeout/watchdog setup | 0.016 |
| Synchronous commit | 0.029 |
| waitUntilCompleted | 9423.567 |
| Watchdog retirement | 0.001 |

The enclosing boundary submit/wait is 9423.877ms; GPU execution interval is
1389.812ms, leaving 8034.065ms outside that interval. Encode is separately
22.627ms. GPU execution overlaps host waiting; these are not additive phases.
The next chunks' first commit durations are 0.007/0.005/0.007ms and enclosing
host-minus-GPU differences 1.608/1.461/1.766ms. All 64 commits total 0.526ms
(maximum 0.029ms); watchdog setup/retire total 0.075/0.137ms. Boundary host
durations total 76608.646ms versus GPU intervals 68495.186ms. The residual
between boundary host and summed native phases totals 26.146ms (maximum
20.072ms); logging, FFI and scheduling outside the native timestamps remain
included in the enclosing measurement. Do not interpret this as zero cost.

This rules out synchronous commit and watchdog bookkeeping as the dominant
delay in this run. It does not distinguish waiting before GPU execution from
completion-notification delivery afterwards, nor establish compilation,
residency, or the earlier intermittent failure's cause. No kernel/default
promotion, throughput comparison or stability fix follows from this result.

Release probe/private bridge build, metadata dry-run, 22 CLI negative cases,
11 trace specs and the native fake-command test pass. The fake tests require
one commit/wait under watchdog registration, separately delayed commit/wait,
the unprofiled and already-committed paths, and success/failure propagation;
they request no Metal device. Reproduce the native test standalone (do not
link another bridge object):

```sh
env DEVELOPER_DIR=/Library/Developer/CommandLineTools xcrun clang++ \
  -isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 \
  -std=c++17 -fobjc-arc spec/metal_submit_profile_test.mm \
  -framework Metal -framework Foundation -o /private/tmp/metal-submit-profile-test
/private/tmp/metal-submit-profile-test
```

The offline checker pairs all 64 records and rejects seven mutations (missing
native/boundary, failed native/trace status, nonfinite native/GPU duration,
untimed flag). Independent raw-log totals and pinned identity checks agree.
Native status is the existing mapped completion result, not raw Metal enum;
GPU interval remains in the boundary record. Pairing is positional in this
single-submitter graph0 probe, not a concurrent telemetry contract. Fake tests
do not assert stderr schema; this run's checker checks the observed records.

Exit0/observer0, startup78%, minimum44%, 40 clean memory samples, final process
tree absent. All guards unchanged; no failure, kill, timeout or retry. Launcher
wall81.010s includes setup/cleanup and is not pp/tg. Correlated Luna review and
direct checks: ROBUST for this bounded separation and preserved wait/watchdog
path; long warm timing and uninstrumented stability remain IN_PROGRESS.

Next discriminator: locate the pre-execution versus post-execution part of the
gap using verified timestamp semantics. Do not subtract unrelated clock epochs
or add waitUntilScheduled as if it were passive instrumentation. No additional
GPU attempt is authorized by this result alone; retain the bounded run policy.
Evidence: `/private/tmp/qwen-prefix-submit.646NF9/` (temporary); refresh on
source/model/input/device/toolchain drift or evidence loss. SHA256:

- manifest: `46791356fdb1bf55ea1200d298e80999abfb75611cfb29d215fadead0d7e5c6e`
- private bridge: `757be2d91a8388357f27e3b6a0d8f1dca39935d35a3911a490822baecebfda00`
- binary: `6573d564fc9f2bdbce190ce3a5705b18d35c53a6bf3f2a813d2d27ae7c7f0103`
- launcher: `10a10e1628fd4ce0b1bbdc47b436e71b858d9b23d4a6b49cc7e118ff30e1bf3d`
- checker: `c9182666d3cd2e4139f009ee6f14a0702a91c130dd3c5392482f2304a6f09d45`
- stderr: `ced86eb0c54a88a99f2ca10aa1e47059fb57ae8ae076d3ca482d30f599b888a1`

## Initial-prefix host/GPU interval discriminator (2026-09-19, predeclared)

Add `--shape=7839:193 --prefix-profile [--dry-run]`: the same prefix-only
diagnostic as trace mode, plus the existing boundary profiler. Preserve trace
mode unchanged, reject mixed modes, and report both controls explicitly.
The profiler switches to a timed synchronous wait; it measures completed Metal
GPUStartTime to GPUEndTime, not per-kernel activity, occupancy or throughput.
No production source, kernel, routing, precision or allocation changes.

Prediction: if the first long host wait is mostly outside the GPU execution
interval, `submit_wait_ms - gpu_ms` will dominate; otherwise GPU execution is
the stronger lead. Neither outcome uniquely identifies compilation or the
earlier intermittent failing command. Trace begin/end must remain enabled
because boundary profiles are emitted only after successful completion.

DoD: release build and strict CLI self-test pass, metadata dry/live controls
and token hashes agree, and one guarded attempt yields paired command records
with finite positive GPU intervals or a preserved failure. Independently pair
all64 expected waits with profiles in emission order and chunk identity;
missing/untimed/nonfinite records are inconclusive, never zero GPU cost.
Keep startup70/runtime30%,24GiB,300s/180s watchdog,lease0,quiet bypass,
chunk2048/group1/cooldown50 and graph disabled. Stop on first failure, no retry.
Pin production/bridge/model against the failed timing manifest and repin probe,
binary and launcher. Artifacts: `/private/tmp/qwen-prefix-profile.KBHkT1/`.
Rollback removes only the opt-in diagnostic. A successful run is not a fix,
stability certificate, correctness comparison or warm append speed result.

### Result: first-command delay is mostly outside its GPU interval

One admitted attempt completed all7839 prefix tokens. Every one of64 traced
shared waits pairs with one finite positive completed-command GPU interval.
All controls and input/source/bridge/model/binary identities match the pinned
manifest; explicit cooldown remains50ms and CogniGraph remains disabled.

| Chunk start | First host submit/wait ms | First GPU interval ms | Difference ms |
| ---: | ---: | ---: | ---: |
| 0 | 9793.106 | 1417.325 | 8375.781 |
| 2048 | 1570.931 | 1569.371 | 1.560 |
| 4096 | 1772.827 | 1771.278 | 1.549 |
| 6144 | 1752.675 | 1751.212 | 1.463 |

The first command has cursors0->7, groups1 and separately measured encode time
106.789ms. About85.5% of its host submit/wait lies outside its GPU execution
interval. Across all64 waits, host time sums74232.581ms and GPU intervals
65802.856ms; after excluding the first command, their residual totals53.944ms.
These are instrumented intervals, not end-to-end pp/tg or a speed comparison.
The residual may include commit/driver/queue/scheduling/completion overhead;
it is not measured CPU computation and does not uniquely identify compilation.
This observation cannot be retroactively assigned to the earlier failed run.

Process/observer exit0, startup80%, minimum48%,39 error-free memory samples,
final process tree absent. Launcher wall78.861s includes setup and cleanup.
No Metal failure, memory kill or timeout; no retry or second GPU attempt.
Release build, metadata dry-run,17 CLI negatives,11 trace specs, format and
diff checks pass. Independent raw-log totals and pinned-hash checks pass.
The offline checker pairs trace/profile order and chunk identity and rejects
four mutations: missing profile, untimed flag, failed terminal and NaN interval.
Its initial overly strict groups1 assertion rejected the final command of each
chunk; source and the previous trace establish terminal cursors63->64/groups0.
The checker was corrected offline, without rerunning or changing GPU evidence.

Verdict ROBUST for this profiled completion and measured host/GPU separation.
The profiler changes the wait API and logging can perturb scheduling; neither
this pass nor the previous traced pass proves uninstrumented stability or a
watchdog fix. Root cause and long warm append timing remain IN_PROGRESS.
Next narrow discriminator: inspect/instrument the host duration of native
`[cmd commit]` separately from `[cmd waitUntilCompleted]`, preserving wait and
watchdog behavior. Only a newly bounded probe may run it. If commit is short,
the remaining scheduling/completion gap needs a different observation; neither
branch alone certifies compiler causality. Do not modify kernels on this result.

Artifacts: `/private/tmp/qwen-prefix-profile.KBHkT1/`; temporary availability,
refresh on source/model/input/device/toolchain drift or evidence loss. SHA256:

- manifest: `cdb1dc4f78c4fe4dfa8d8357c7530045585c33a58908a1b1794f776bebf075be`
- binary: `3bbe7700c932ffdba876f16427a52c529c4859471d093f48e1dc1ac12897fa16`
- probe: `00363347c2a994918eb1fc448c9b2722f00b1675ad5f9bbddd33e562f8c778d5`
- launcher: `2f2324238e1ff1c40d0be2c6e5b603263aa7e95f044da752cb7b569e527adb90`
- corrected checker: `45e59d5973d8fdf2d87eff57ad3239fc2fb96038c4b79a865d067417e8104984`
- stderr: `da14a27252615b8ca0d9ca6e7bd9941226868809e49fa86a4ea9a1d7d193284d`

## Initial-prefix command localization (2026-09-19, predeclared)

The warm timing series failed before constructing its shared7839-token prefix.
Use a separate `--shape=7839:193 --prefix-trace` mode with the same public
tokens, actual model, F32 capacity8036, direct SG4, Flash off, chunk2048,
group1 and cooldown50ms. Enable only the existing host command trace, recording
its control settings and prefix token hash. Stop after the initial prefix:
no state copy, append, warmup or timing comparison. No production code change.

Competing explanations remain: a particular shared-command/shape boundary,
cumulative pressure later in the prefix, or scheduling/interactivity sensitivity.
A failed begin/terminal pair can identify start_pos, rows and loop cursors;
those cursors are not an exact encoded layer interval or an offending kernel.
A pass with tracing narrows repeatability only, not the failure's cause. Existing
trace covers ordinary shared-command waits and routed full-layer calls, not
every GPU setup, standalone recurrent command or CogniGraph flight.

DoD for the diagnostic: self-test rejects incompatible modes, metadata dry-run
matches input/config identity, existing trace exception/flush/placement specs
pass, and at most one fresh guarded process produces either completed prefix
records or a localizable failure. Inspect raw stderr pairing and failure stack
separately; an untraced failure remains unresolved. Never use traced host wait
times for GPU-kernel speed or repeat a failure to obtain a pass.

Preserve startup70%/runtime30%,24GiB,300s/180s watchdog, lease0 and authorized
quiet bypass. Stop on first GPU failure, no automatic retry. Pin production
source/bridge against the failed timing manifest and pin the new probe/binary,
model stat and launcher. Artifacts: `/private/tmp/qwen-prefix-trace.8nRDsz/`.
The earlier dry-only scope `qwen-prefix-trace.D06S7O` caught a controls JSON
array/object mismatch; it launched no GPU workload. Controls now serialize as
an explicit object, rebuilt and repinned before admission.
The earlier series is consumed and untouched. Rollback removes this diagnostic
mode; long-prefix stability and two-shape timing remain open until new evidence.

### Result: traced prefix completes; failing command not reproduced

The one admitted GPU attempt completed prefix7839, with no state copy or
append. All64 ordinary shared-command waits have matching begin/end records:
16 per recursive chunk, each with its own trace identity and sequence1..16.
There are no failed/unmatched records, standalone layer records, Metal errors,
guard kills or timeouts. Separate raw-log pairing checks reproduce the counts
and verify all begin/terminal fields agree. This validates trace accounting
for this run, not coverage of every internal GPU command.

| start_pos | rows | Completed traced waits | Longest host wait ms |
| ---: | ---: | ---: | ---: |
| 0 | 2048 | 16 | 9052.165 |
| 2048 | 2048 | 16 | 1482.338 |
| 4096 | 2048 | 16 | 1730.654 |
| 6144 | 1695 | 16 | 1689.600 |

The9052.165ms observation is sequence1, cursors0->7, groups1 in the first
chunk; the median over that chunk's16 waits is759.087ms. It is host
commit/wait time, not measured GPU
execution or proof of driver compilation. It does not identify the command
that failed in the earlier untraced run. A deterministic inevitable failure
at this input is not supported; intermittent failure and scheduling/host
effects remain possible. Tracing did not prove that it caused the pass.

Process/observer exit0, source/binary/model identity unchanged, startup81%,
minimum48%,38 error-free memory samples, final process tree absent. Launcher
wall76.772s includes model/setup/cleanup; not a prefill throughput measurement.
Both prefix markers and the narrowly scoped completion summary are present.
Input prefix SHA256 is
`c846e1cf9311d1e56905afa56c3c775bcd04957d0b945c60bbbc71202dafea9d`.

Build, corrected dry-run and self-test pass (12 rejected CLI combinations);
11 focused trace specs cover failure logging, exception preservation, broken
sink behavior and source placement. Format/diff checks pass. The initial
dry-only JSON mismatch was corrected before any GPU attempt, not hidden by
repeating a GPU failure. Correlated Luna source review found no scoped blocker.
Verdict ROBUST for one traced prefix completion; failure localization remains
IN_PROGRESS. No engine/kernel/routing fix or speed improvement is claimed.

Artifacts in the directory above include the consumed `attempt.json`, manifest,
config dry/live records, paired trace, exit result and memory observations.
SHA256:

- manifest: `8b1cd0c87c7d1dc93fe87c6ac53af553d5ba003cf68f19468e2bca8265c186c6`
- binary: `262318c419b2a291dfd11d5d100c666b5ecb3f40fc540e6d016f9ccd38a2d54c`
- probe: `42f126df4e48c1e2ae721cd94255db25c69ddcd98bc5a899492383b012c58719`
- launcher: `72dbef2af78740bbf6e38a3c1757de28aaef40d0ce588294e4808159dd961df5`

Next discriminator is a separately bounded prefix profile comparing host wait
with Metal's completed-command GPU interval, especially the first command.
Existing `QWEN35_PREFILL_BOUNDARY_PROFILE` provides this without new kernels;
the probe currently scrubs external overrides, so a future diagnostic must
admit that control explicitly and repin identity. Explicit cooldown50 and
disabled CogniGraph must remain fixed. Missing/failed GPU timestamps are not
zero-cost execution; host-minus-GPU time does not uniquely identify compilation.
No second GPU attempt was made. Refresh after source/model/input/device/toolchain
drift; long timing and uninstrumented stability remain open.

## Warm balanced append timing (2026-09-19, predeclared)

Extend the real-model probe with `--timing` for the same two admitted shapes,
256/195 then 7839/193, without changing production kernels or routing. The
previous fixed-order unwarmed times are not the performance baseline.

Use two states only: an immutable synchronized direct-prefilled prefix and one
working state. Restore the latter from the former before every append, outside
the timer. Warm up in ABBA order, then measure ABBA/BAAB/ABBA/BAAB (A=ordinary
rows, B=direct SG4): two warmups and eight measured samples per arm. No adaptive
stopping, trimming, retries or best-of selection. Keep every sample.

Time the complete full-width append plus full output head and final GPU fence,
in host milliseconds. Exclude prefix construction, state reset, state/logit
validation and diagnostic output. Disable pipeline-binding logging before
process startup; underlying production source must match the previously traced
parity run. Reject measured samples that grow the application pipeline cache
(this observes application entries, not hidden driver compiler variants).

After every warmup and measurement, require exact logits and SHA256 agreement
over all live F32 KV and complete conv/SSM values against the first row warmup;
that reference is also checked for nonfinite values. Confirm the immutable
prefix remains unchanged. This is append parity, not a new four-token decoding
quality claim. Pipeline route tracing and prior greedy parity remain separate
evidence; no per-dispatch logging is included in the timed run.

Report median, quartiles, min/max, four balanced-block ratios and paired-order
ratios. A consistent local latency gain requires >3% reduction in every block
and in both AB/BA pair-order medians; otherwise report mixed/inconclusive or a
regression. Eight samples in one process are correlated, not independent trials
or a basis for broad confidence intervals. Do not infer full pp/tg, isolated
attention GPU time, llama.cpp competitiveness, or general production speed.
Reset/copy and validation can perturb caches despite being outside the timer.

One fresh process per shape, stop the entire series on first failure. Retain
startup free>=70%, runtime floor30%, 24GiB tree cap, 300s process timeout, 180s
command watchdog and zero-wait Metal lease. User-authorized quiet bypass stays
in force; read-only memory and decaying `ps` CPU snapshots annotate host noise
but cannot exclude external GPU load. Source/model-stat/input/binary identity
is pinned. Artifacts: `/private/tmp/qwen-sg4-timing.oUzWgP/`; consumed earlier
series remain untouched. Rollback removes timing mode; defaults stay unchanged.

### Result: small short-prefix gain; long-prefix timing unavailable

The first series attempt is consumed. Short shape256/195 completed all four
warmups and16 measurements. Separate raw-log recomputation agrees with the
launcher; no samples were omitted. Times include append, output head and fence:

| Arm | Median ms | Q1–Q3 ms | Min–max ms | Measured n |
| --- | ---: | ---: | ---: | ---: |
| Ordinary rows | 2380.081 | 2369.126–2383.051 | 2365.223–2396.498 | 8 |
| Direct SG4 | 2328.562 | 2324.808–2342.564 | 2316.927–2377.420 | 8 |

Median latency reduction is2.1646% (ratio1.02212), not the large apparent gain
in the prior fixed-order/unwarmed run. All four balanced-block ratios favor SG4
(1.01652,1.01558,1.02288,1.01652), but **none meets the predeclared >3% gate**.
Report a small observed local difference, not a promoted material speedup.
Warmup row timings were2597.346/2368.151ms; SG4 was2323.090/2311.940ms.
This supports using warmed/order-balanced evidence, but does not isolate the
cause of every difference from the earlier process.

All20 passes have identical live KV/conv/SSM byte hashes and numerically exact
full logits, with top2 `[3753,653]`; measured application pipeline counts stay33.
The immutable prefix hash remains unchanged. The hash covers tensor values,
not `LayerState#position`: position equality is checked at the initial deep
copy, subsequent `copy_from!` assigns it, and append receives explicit prefix256.
Do not interpret the tensor hash as a complete serialization/state certificate.
Route controls are set by pinned probe source after clearing QWEN overrides;
production source matches the prior traced run. There is no new per-dispatch
route trace or general runtime-control attestation in this timing process.

Long shape7839/193 failed while constructing the initial shared prefix, before
the common-prefix comparison, warmups or either timed append arm. Metal reports
`Impacting Interactivity`, completion_status=-6; process exit1, observer exit0.
There are zero timing samples, not a slow candidate result. Source/binary
identity did not change. This reopens long-prefix stability despite the prior
successful parity run; neither SG4-versus-row causality nor the offending
command/layer is established by this uninstrumented stack. No retry was made.

Short startup82%, minimum54%,33 memory samples; long startup81%, minimum50%,34
samples. All observer records are error-free and both final process trees are
absent. No run_safe memory kill or process timeout is reported. These snapshots
do not establish absence of GPU resource pressure. Decaying CPU totals ranged
125.4–572.6% (short) and82.6–243.6% (long); they are noise annotations, not GPU
load or independent interval utilization. Guards remain70/30,24GiB,300s/180s.

Verification: launcher `check`, `build` and `dry` pass; comparator/CLI/timing
self-test, six report-checker negatives and synthetic statistics controls pass.
Nineteen focused Crystal metrics/trace/shape examples, five Python admission
tests, format and diff checks also pass. The first sandboxed spec launch was
refused at process-group isolation; the contained approved run passes without
loading a model or rerunning GPU timing.
`run` returns failure at the long case as predeclared; no `complete.json` exists.
The short timing/quality gate passes, but its >3% gain gate does not. Separate
post-run checks authenticate source/binary identity and recompute raw medians.
Correlated Luna pre/post review is ROBUST for the completed short diagnostic,
not for a complete two-shape result. The complete two-shape timing objective
remains IN_PROGRESS, not VERIFIED performance improvement.

Artifacts: `/private/tmp/qwen-sg4-timing.oUzWgP/`, including manifest, consumed
series marker, dry inputs, per-case logs/exit records, memory/CPU observations
and short-case report. SHA256:

- manifest: `896ce573b53e68370c765e907dab4ea0bd5b11c95f9dd3be41697b87b4f610ab`
- binary: `b17854cfa95c8f8a7bed3e35b416df676bb6c7ff42a04a5f2ca9f262f252df7e`
- probe: `826b9d9b8a4b9c67e2e66c490b572c3d3bc9b9ed1223116713003e0bf924b404`
- launcher: `5c3cfcd85c1bbe84aac1343a027175fadec2ab5fa4b3a7154fbc0b2c010448a2`

Next discriminator: separately instrument initial-prefix command boundaries
before attempting another long-context timing comparison. Do not lower guards,
replay the consumed series or promote production routing. New source/model,
device, input or toolchain requires a new evidence scope; no global pp/tg,
llama.cpp comparison, adaptive-QBit or broad stability claim is admitted.

## Real-model direct versus row attention (2026-09-19, predeclared)

Next test two F32 full-width append shapes, in order: prefix256/rows195, then
prefix7839/rows193. Use public code-completion tokens and actual Qwen3.8-27B
Q4_K_M weights, not the separable synthetic attention fixture. This is not the
saved tool-session replay. The reference is ordinary one-row-per-group attention
(`SG4_OFF=1`), not the intermittently failing pregate kernel. Candidate uses
direct SG4 (`SG4_OFF=0`, direct minimum1); Flash remains off. Production defaults
are unchanged. A shared direct-prefilled prefix is synchronized and deep-copied
into a second nonaliasing F32 state; compare the live prefix exactly before
either append. No extra prefix replay, warmup or retry.

Gate all live K/V and recurrent state after append, full logits, top2 coverage,
token ECS, and four independent greedy steps with equal token histories.
Reuse existing diagnostic tolerances: state0.02+0.01*abs(reference), logits0.1,
logit cosine>=0.9999, reference top1 covered, token ECS>=0.99; also require
identical greedy token IDs. Do not relax a tolerance after seeing results.
Bindings within completed append intervals must show16 row kernels versus16
direct SG4 kernels. Binding telemetry alone is not execution proof.

One fresh process per shape; first failure stops the entire series. Initial
free memory>=70%, runtime floor30%, tree cap24GiB,300s process limit,180s command
watchdog, lease0, quiet gate disabled. Preserve existing consumed attempts.
Pin all compiled sources, private bridge, input token hash, model stat and
binary; record process-tree memory observations. The current unrelated FFN
capacity-reuse WIP is left untouched and its opt-in environment flag cleared.
This gate checks bounded numerical agreement, not broad coding quality, model
accuracy, speed, stability or the cause of earlier Metal failures. Four tokens
do not establish sentence-level semantic quality. Artifacts:
`/private/tmp/qwen-sg4-model.h5PWz3/`. Next action depends on observed parity;
do not widen production routing on this two-case result alone.

### Result: both real-model append comparisons pass

`bin/qwen35_sg4_model_probe.cr` completed both first attempts on Apple M2 Max.
For each case the synchronized common prefix was numerically exact after deep
copy, with 256 distinct state-buffer addresses. All live F32 K/V values and all
conv/SSM values remained numerically exact after the append and three consumed
continuation tokens (max absolute difference0, no nonfinite/out-of-tolerance
values). All248,320 logits at each of four greedy positions also had difference0.
Each case matched top1 4/4, ranked top2 8/8, reference-top1 coverage4/4 and ECS1;
both branches emitted ` seen = set()`. ECS1 follows identical token IDs; this
does not test the semantic tolerance of different tokens or complete the task.

| Prefix / append | Live tokens after continuation | Initial / minimum free | Row / direct append wall |
| --- | ---: | ---: | ---: |
| 256 / 195 | 454 | 80% / 52% | 4.489 / 2.363s |
| 7839 / 193 | 8035 | 78% / 42% | 6.458 / 4.044s |

Both processes and observers exited0; memory observations13/59, no collection
errors, Metal failure, guard kill or timeout, and final workload trees absent.
Each completed append interval bound exactly16 ordinary-row or16 direct-SG4
pipelines as declared. Dry/live token hashes matched; all131 pinned source,
bridge and launcher artifacts, binary and model stat identity remained stable.
The series is consumed; neither case may be repeated through this launcher.

DoD: `python3 /private/tmp/qwen-sg4-model.h5PWz3/run.py build`, `dry`, then `run`
all passed. Standalone self-test rejects perturbed/nonfinite/empty comparisons
and eight invalid CLI combinations; trace checker passes one positive and five
negative controls. Additional13 Crystal metrics/trace/shape examples and five
Python pressure-admission tests pass; type-only build, format and diff checks
pass. Separate post-run log checks confirmed all component/step counts and
zero differences. Correlated Luna source review found no scoped blocker.

Artifacts are the directory above: `manifest.json`, `dry-*-input.json`,
`run-*.{stdout.log,stderr.log,memory.jsonl}`, per-case exit/pass certificates,
`series-attempt.json`, and `complete.json`. SHA256:

- manifest: `ab6be1725671bb307e2ef639ddc40e397c0bbe5a68660959c8a35489974b3a94`
- binary: `9c9a5f68ad536374d64c7496b94482758490c08949165fc7bb0023b7266aded2`
- probe source: `8e11064c426b425aa29b007e569468ff2553aae7673bec158bc683fdad275b07`
- launcher: `75773c7268a2d38e411b4e5bfa4eeee85149b58392efa0fa77c8a1ed6c34ff9f`

Verdict: ROBUST for append-path numerical agreement on these two public-token
fixtures and four continuation positions. The shared direct prefix is not a
direct-versus-row prefix comparison. Timings are single, fixed-order, unwarmed
host-wall measurements including the output head/fence and possible pipeline
compilation; they do not establish a speedup. Root cause of prior pregate
failures, broader stability, independent model correctness, adaptive-QBit
behavior and default routing remain open. Next discriminator is a separately
declared balanced timing test, not promotion of a195-row threshold. Refresh
requires a new evidence scope after source/model/device/input/toolchain drift;
do not silently reuse consumed attempts. Rollback removes the standalone probe;
production kernels, policies and unrelated FFN-capacity WIP are unchanged.

## Direct full-shape neighbors (2026-09-19, predeclared)

The next discriminator is model-free, not another 27B replay. Extend the existing
SG4 tail probe with exact `--direct-shape=base:rows` admission for
`7839:193`, `7839:194`, `7839:195`, `7839:196`, `0:64`, `0:195` only.
Unlike legacy `--single-command`, this dispatches and validates every row,
not the first 64 of a 193-row fixture. Each fresh process compiles only the
direct F32 pipeline and submits one full-shape command, without warmups.
The existing Float64 synthetic oracle, finite/unwritten checks and four-row
trailing canary must pass. This checks synthetic shape correctness/execution;
it cannot prove model parity, root cause, speed or production stability.

One six-case series, stop on the first nonzero status or validation failure;
no retries or consumed replay reuse. Start >=70% free, runtime floor 30%,
model-free process-tree cap 2GiB, 120s per case, watchdog 180s, lease 0, quiet gate
disabled. Clear inherited Qwen/Metal/runner controls, pin source/binary/bridge
hashes before and after, and require six distinct PIDs. A private bridge is
built with installed Xcode because the root object lacks pipeline metadata
symbols and CLT native compilation cannot find C++ headers. Neither the shared
bridge nor production routing/kernel/defaults changes. Launcher/evidence:
`/private/tmp/qwen-direct-neighbors.pTeZdX/run.py` and
`/private/tmp/qwen-direct-neighbors-build.LYTGib/`.

### Result: six full-shape direct checks pass

All six first attempts passed on Apple M2 Max at 79% initial free memory, in
six distinct PIDs, one command each; no Metal error, guard violation, memory
kill or timeout. The 131 pinned source/bridge/launcher artifacts and binary
identity matched before/after each run. Series is consumed. The admission
checker's 27B profile label supplies only the conservative 70/30 memory check;
it is not geometry or demand certification for this model-free experiment.

| Prefix | Rows validated | Maximum absolute error | GPU ms (single diagnostic) |
| ---: | ---: | ---: | ---: |
| 7839 | 193 | 1.081e-6 | 113.346 |
| 7839 | 194 | 1.081e-6 | 109.145 |
| 7839 | 195 | 1.081e-6 | 114.394 |
| 7839 | 196 | 1.081e-6 | 116.839 |
| 0 | 64 | 1.073e-7 | 0.480 |
| 0 | 195 | 1.073e-7 | 2.795 |

All output values satisfy the 1e-5 Float64 oracle tolerance; trailing canaries
remain untouched. CPU self-test rejects seeded source/output/guard corruption;
the new malformed CLI rejects before Metal. Nineteen focused shape/safety/stage
specs, formatting and diff checks pass. Luna's correlated review found no blocker
for this diagnostic contract; a separate six-log check confirms unique PIDs,
one PASS each and exit 0. The launcher seals the series before first submission
and checks exit, full-row trace, oracle/canary result and source identity before
allowing a successor.

Reproducible CPU-only DoD (no GPU submission):

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools CRYSTAL_CACHE_DIR=/private/tmp/qwen-neighbor-spec-cache crystal spec spec/qwen_sg4_probe_shape_spec.cr spec/qwen35_sg4_tail_safety_spec.cr spec/qwen_prefill_stage_split_spec.cr --no-color
DEVELOPER_DIR=/Library/Developer/CommandLineTools CRYSTAL_CACHE_DIR=/private/tmp/qwen-neighbor-spec-cache crystal run -Dcpu_only bin/qwen35_sg4_tail_probe.cr -- --self-test
```

ROBUST only for these synthetic F32 shapes. The separable, repeated synthetic
K/V values do not cover arbitrary model distributions, layer composition,
FP16/QBit, repeated-process stability or performance comparison. The full-model
intermittent failure remains unresolved. Next discriminating step is real-input
neighbor-shape validation of an opt-in routing candidate, including short-prefix
regression controls; first inspect the route/cost boundary, do not make 195 a
universal default or repeat the consumed experiments. Refresh on shader, bridge,
device/toolchain/input drift or artifact loss. Rollback removes this diagnostic
mode; production behavior is unchanged.

Evidence SHA256 (logs and manifest under the build directory above):

- Binary: `b9670e2017f831ba2f5f4072aa031f495e55bd4cbcba976267a59853be0d2b31`.
- Private bridge: `7329853fc218a89cff768b2fbdb217449958625e5dd3d567ce0de0f31c305a6f`.
- Manifest: `bd14543ddb8de47932e24379b4811510cc7295db59074dd2e539935e8ad1ff5e`.
- Launcher: `fa293ca03abf55ed82b76b6c39cd2921cd386f84b0873a639743e996fb70d05b`.
- `1-7839-193.log`: `7ed988b38c28af8aa62bab99baf2c9df0fbc2e5875eafed2f89dde95bb0f72d0`.
- `2-7839-194.log`: `5b96f8ba1d697d499c08f26f77d6471da3c361cbd2212de55d8fa842b9e65cda`.
- `3-7839-195.log`: `728d8df3611128436c9522e95dc227ce3650beccc74273e388ffd24aff8b4618`.
- `4-7839-196.log`: `73d68c520388ad71fee0cabf56357525c6bf3c60866576ec910e7f7fd33278d4`.
- `5-0-64.log`: `9566f3f04042065ca250db7d27ed7c17b6a87b00b4ff1d1c7b394fc98cf48996`.
- `6-0-195.log`: `6c80263bf0f384152525609f606b40375a88cbf34f4801220d0bbcbeb8374ebe`.

## Suffix-only route discriminator (2026-09-18, predeclared)

Re-reading the completed direct trace identifies standalone rows2048,1668,195,
8,4,3; the default trace stops during195. In both, rows8/4 already occur before
the second call. Thus gate1 versus default1024 changes earlier short commands,
not only the failing195-row interval. Prior three/two-stage successes and the
fresh-process attention-only pregate failure make another identical stage split
weakly discriminating; do not repeat that experiment.

Use the existing gate195, no engine/kernel edit: rows195 select direct, rows8/4
retain default pregate, rows3 retain scalar-row attention, and initial2048/1668
remain direct. This is suffix-only for the observed trace, not a context-aware
production policy: any newly observed row count195..1023 would also change.
Prediction: if changing195 alone suffices for this observed replay, both calls
complete with the saved second-output digest despite earlier/later pregate8/4.
A failure rejects that bounded recovery attempt, not all future executions.
A pass does not establish causality, hidden-state parity, speed or stability.

One fresh attempt, preserving consumed earlier pairs; gate70/30, cap24GiB,
timeout300s/watchdog180s, lease0 and first-failure stop. Reuse the instrumented
binary only after all compiled-source/bridge/input hashes and model stat match;
new manifest pins the wrapper and updated admission checker separately. Actual
pipeline names must corroborate195 direct and8/4 pregate within enclosing calls.
No saved-session tools execute. Metadata-only validation and31 focused
stage/trace/tail specs pass; a sandbox process-group admission refusal is kept
separately, not counted as a GPU attempt. Artifacts:
`/private/tmp/qwen-suffix195.3gnnjT/` (launcher) and `attempt/` (validated dry run).
Rollback removes the environment override; no production default is changed.

### Result: suffix195 direct passes while tiny batches remain pregate

One attempt was admitted at78%, with the same binary SHA256
`c9f10d244f5f6382c0c04c913e0c80067b686f3b02071fce123ddbafe4714661`.
All260 source/input/bridge hashes, model stat identity and binary identity
matched before and after. Both calls and final completion passed the launcher's
assertions; call2 output SHA256 remains
`ed251864987c367e9641fbdc89c1d83e9bf0fa2e3eecef8f301c79f619bfac81`.
No tool calls. Exit0, observer0, wall89,235.810ms; call1 total75,107.1ms,
call2 total6,233.9ms (prefill+top1 3,311.2ms, decode-body9.43 tokens/s).
Unbalanced diagnostic timings are not speedup evidence.

The route checker verifies all195 standalone intervals:15 suffix195 direct
bindings,60 rows4/8 pregate bindings, remaining scalar/large direct routes
unchanged. It checks exact shape/layer sets and completed enclosing calls,
not GPU-internal attribution or the specialized final layer. Qualification
accepts an explicitly synthetic relabelled positive and rejects the actual
gate1 trace, failed default trace, wrong tiny binding and missing terminal.
Those controls qualify parsing only; the new complete run is the live positive.
Forty-four samples:78->min45->76%, no collection errors, Metal error, memory
kill or timeout; final observer sees no workload tree. Attempt is consumed.

ROBUST for this bounded successful replay with tiny4/8 still pregate. Changing
tiny batches was not required for this success; this is not a proof that the
195-row pregate kernel causes the intermittent error. Whole-command budget,
resource lifetime and host/driver history remain confounders. Do not promote
195 as a production threshold: it was chosen to separate observed shapes.
Next candidate policy must include prefix/context and validate other suffix
lengths plus short-prefix regressions; no such policy is implemented here.
Refresh after source/model/input/device/toolchain drift or artifact loss.
SHA256 (paths relative to the experiment root above):

- `replay.py`: `a046d1295305990ebda6665c57432078a94e847788f75f29032b3e2e350c51dc`.
- `check_routes.py`: `8444c0c71a9a04a6a7d2f094c3cf6eb1d847c9b9f63cb875a8f05cbd991bff63`.
- `attempt/manifest.json`: `29b56f68ffad113fad0b9a1250866b17ec2533c5aeef658937f1cd220633fc47`.
- `attempt/run.stdout.log`: `9ff40b7b764083fd3fd55bc2d4458b5c2d0a318f0212d590090975e4d93cd3f5`.
- `attempt/run.stderr.log`: `c524dcdf7a90af262ef161477d6d4b17f78b335e7c684570163c077abf92e35a`.
- `attempt/memory.jsonl`: `5aaf70c53dccff361aa366c2c531da40a4f1b3bb9d5f543d51b98659e3d87289`.

## Instrumented default fails; scoped admission is now70/30 (2026-09-18)

The operator authorized initial free memory >=70%, retaining the30% runtime
floor, 24GiB cap, 300s workload timeout, 180s command watchdog, zero-wait lease
and first-failure stop. This changes only the documented64GiB replay profile,
not global runner defaults. Admission actually observed78%; this attempt
would also have passed the prior75% gate. It supplies no evidence about
starting below75%, and the earlier43% minimum is not a guaranteed demand bound.

One default attempt used the pinned instrumented binary/input/source of the
direct attempt below. Effective numerical controls differ only by the direct
gate override; admission policy and host/order differ. Call1 completed:
total81,179.5ms, prefill+top1 77,708.9ms, pp_top1 100.54 tokens/s, decode-body
8.69 tokens/s, output27 tokens. The second input matched8,035 prompt tokens,
7,839 cached,196 suffix, capacity8,845. No saved-session tools were executed.

Call2 failed at start7839/rows195/layer23, sequence7. Its actual binding was
`qwen35_attn_decode_rows_sg4_pregate`, followed by `Impacting Interactivity`
and completion_status=-6. Direct completed the matching layer23 interval
(host155.173ms), as well as layer35. The earlier uninstrumented default failed
at layer35 instead: do not attach the failure to one fixed layer. Binding
telemetry is not kernel execution attribution; the failed command also owns
projections, normalization, FFN and final add. No second-call completion or
output-parity result exists for default, and no speedup is established.

Default exit1, observer0, monotonic wall91,627.484ms. Its45 memory samples were
78% -> minimum45% ->75%, with no collection errors, memory kill or runner
timeout. Final observer reports the workload tree absent. This does not rule
out memory-related GPU effects. Both attempt markers are consumed and
`pair-stopped.json` forbids retry. ROBUST bounded failure reproduction and
stop behavior; causal attribution and default promotion remain open.
Next discriminate command-stage/budget effects before another GPU experiment.

The70% admission sidecar preserved the pinned launcher and post-run identity
checks. Only after that run ended was the repository checker changed75->70;
its old manifest hash is now intentionally stale, not a historical mismatch.
Boundary/CLI tests failed against75 and then passed all5 methods against70;
refusal exit code75 remains unchanged. Policy rollback is restoring75, never
resetting consumed attempts. Historical pending instructions below are superseded.

Artifacts: `/private/tmp/qwen-pipeline-pair.JKEUrc/default/` (ephemeral); refresh
on artifact loss or source/model/input/device/toolchain drift. SHA256:

- Admission sidecar: `00b15e99bce0a8a6ec75957b28388dc26fe6d54d63d8855a216906b5c48ec10e`.
- Default stdout: `0e530fbb6807bad4a041e390d54f9becb97533922dab34775df38803261f0071`.
- Default stderr: `bcc3908630e9556b2947f95eb4b411fdabd7f59b13026c4261ab2611db5c7b6f`.
- Default memory: `de1e346aa5c5522accd0a9225f7fd2dd29d3bbbcaf3a6ea10a62fe077da2839b`.

## Instrumented direct passes; default admission pending (2026-09-18)

At HEAD `5c13d27f` plus the unchanged pre-existing FFN WIP, a fresh release
build passed under run_safe300s/8GiB with CLT. Both metadata-only routes and
the22 pipeline/command/SG4 specs passed. The two manifests have identical
source hashes, model identity, binary and input; their controls differ only
in `QWEN35_PREFILL_ATTN_ROWS_SG4_DIRECT_GATE_MIN` (direct1 versus absent/default).
Both use `COGNI_METAL_PIPELINE_TRACE_PREFIX=qwen35_attn_`, initial75%, runtime30%,
24GiB cap, 300s workload timeout, 180s command watchdog and zero-wait lease.
No saved-session tool is executed. The launcher separately waits for its
observer; workload timeout and launcher wall time are not interchangeable.

Direct was admitted at75% and completed both calls. The actual second-call
start7839/rows195/layer35 interval contains
`pipeline="qwen35_attn_decode_rows_sg4"`, followed by `call_end`, host130.216ms.
This is now an observed host pipeline binding, not routing inference or a
per-kernel GPU timer. Traces also show direct SG4 for the later4-row batches,
so the override is not isolated to the195-row suffix.

Call1: total77,936.5ms, prefill+top1 74,691.6ms, reported104.60 pp_top1 tokens/s
and9.39 decode-body tokens/s. Call2: total6,409.0ms, prefill+top1 3,431.9ms,
decode-body9.30 tokens/s. Both output27 tokens. Input matched prompt8,035,
cached7,839, suffix196, capacity8,845; second content SHA256 remained
`ed251864987c367e9641fbdc89c1d83e9bf0fa2e3eecef8f301c79f619bfac81`, no tool calls.
Exit0, observer0, `run_pass=true`; monotonic launcher wall93,767.110ms.
Forty-six samples: first75%, minimum43%, final72%, no collection errors.
No Metal failure, memory kill or timeout signature; final observer reports
root absent and no workload-tree PIDs. Sampled headroom is not a safety bound.

The subsequent default launch returned75 at admission72%, before any attempt
marker, model load or GPU work. Default's dry result remains valid; direct
is consumed and must not be repeated. Admission refusal is not a launched
attempt: later admission may be retried only under the unchanged75% gate and
fresh identity check. A launched workload/validation failure seals the pair.
This distinction was explicitly checked during Luna's bounded wrapper review.

Verdict: ROBUST bounded direct replay and observed pipeline selection only.
The same-guard comparison is incomplete; root cause, pregate culpability,
speedup and default promotion remain open. Host/order/logging confounds persist.
Next command when admitted: `python3 /private/tmp/qwen-pipeline-pair.JKEUrc/replay.py run default`.
Do not lower the gate, modify pinned sources or reuse older consumed launchers.

Artifacts are ephemeral in `/private/tmp/qwen-pipeline-pair.JKEUrc/`; refresh
on source/model/device/input/toolchain drift or artifact loss. SHA256:

- Binary: `c9f10d244f5f6382c0c04c913e0c80067b686f3b02071fce123ddbafe4714661`.
- Build manifest: `6bdac840fd5c1fdb6313481aa8dd8ea2938750f96e6833ca48363d2367669a33`.
- Wrapper: `b43c6a3ce6d7621906dac7a819bd0aa5d8ec7f3a60e418135852019dc03bfedb`.
- Direct stdout: `6fd7ebbaae82332a3cd1b27c072631e304aebbe25b13c01be8aaece44ccfec62`.
- Direct stderr: `3b978db5e3c09d4ac1e9067fec6a70645208ff9624fc2c1d835f9def479b682e`.
- Direct memory: `e4f097512c088db89c68aae741bfecc24109b478dba78c15504bb16577dbd341`.

## Actual pipeline selection telemetry (2026-09-18)

The next discriminator must observe the actual bound pipeline, not infer it
from `full_attn_chunk_routed`. `ComputeEncoder#set_pipeline` now records
`pipeline.name` immediately after the host FFI binding when the process-start
environment contains a nonempty matching prefix:

```sh
COGNI_METAL_PIPELINE_TRACE_PREFIX=qwen35_attn_
```

Each flushed stderr record contains `phase=selected`, native command/encoder
handles, and the escaped pipeline name. Configuration is cached once; unset
or empty disables output. No command boundary, barrier, wait, routing decision
or GPU dispatch is added. Sink exceptions are swallowed, but synchronous I/O
can still block or perturb host timing: use a regular local log file, not a
slow pipe, and do not treat traced timings as an uninstrumented benchmark.
Handles can be reused and are not durable unique IDs. Correlate only within
the current synchronous layer-call interval. A binding is not dispatch,
execution or completion; even the last selected pipeline is not necessarily
the cause of a command-buffer failure.
Names alone do not identify compilation variants sharing one function name;
pin the source/build controls as well. The attention prefix is not a full
inventory of all kernels encoded in the command.

Read-only failure reconstruction: the earlier log's `command_id=23205873696`
belongs to a shared layers0-2 append, not layer35. The `read_output=true` path
flushes before the standalone routed call (`qwen35_cpu.cr:4500-4513`). With
stage splitting disabled, its one command includes normalization, Q/K/V
projections, split/QK norms/RoPE, F32 KV write, attention, output projection,
residual/RMSNorm, FFN gate/up, SwiGLU, down and final add
(`qwen35_metal.cr:8158-8341`). Commit/wait at8349-8350 precedes readback at8353.
Thus the observed completion failure covers this whole sequence, not only
attention. Source predicts SG4 pregate for195 rows; the historical log does
not capture that binding. Luna's bounded read-only audit agrees; exact
non-attention kernel variants and the causal culprit remain unknown.

Verification (no GPU or model load): the new spec first failed because the
helper did not exist; after implementation, the pipeline, prefill-command and
SG4 safety suites passed together: 22 examples, zero failures/errors. Tests
cover prefix filtering, distinct direct/pregate/H16/flash names, escaped names,
sink failure, flush and the source binding hook. The two-call provider probe
passed `crystal build scripts/cogni_qwen_two_call_probe.cr --no-codegen` from
`../crystal_ball`, with `DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
Running that command from cogni-ml first failed shard lookup (`db`); using the
owning project resolved it. This establishes typechecking, not a linked build
or live Metal trace. New helper/spec formatting and `git diff --check` pass.
An additional `-Dcpu_only --no-codegen` provider check is blocked by undefined
`ML::Metal::Device` at `qwen35_cpu.cr:556`; no CPU-provider success is claimed.
The new binding hook is confined to the non-CPU encoder branch.

Next: rebuild and pin one instrumented binary, then qualify the same-input
direct/default discriminator with identical tracing and 75%/30% guards.
Prior consumed manifests/binaries do not cover this dispatch-source change.
Retain cap24GiB, timeout300s, command watchdog180s, zero-wait lease and
first-failure stop; do not promote the route default or claim a speedup.
Rollback: unset the prefix; remove the helper and single binding hook if this
telemetry stops providing discriminating evidence. Refresh on encoder/routing,
model, input, device or toolchain changes. The Metal root cause remains open.

## Default-gate control reproduces second-call Metal failure (2026-09-18)

At source HEAD `f0611c69` plus unchanged FFN WIP, the prepared default-gate
control was authorized to proceed despite the previous74% snapshot. Actual
launch admission was75%, so no exception or launcher modification was needed.
One attempt ran with runtime floor30%, cap24GiB, timeout300s and zero-wait
Metal lease. Source/model/binary identities matched before and after execution.
The metadata dry result remained valid. No rebuild or production change.

Call1 completed: total80,767.4ms, prefill+top1 77,493.6ms, reported100.82
pp_top1 tokens/s and9.24 decode-body tokens/s, output27 tokens. The second
input matched the candidate's hash and dimensions: prompt8,035, cached7,839,
suffix196, capacity8,845. Call2 failed at start7839/rows195, full-attention
layer35, trace sequence10. The previous layer31 returned; the failed layer
recorded142.547ms host time. Metal reported `status=5 error_code=1`,
`Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)`,
then `completion_status=-6`. There was no second call-end, completion event
or second-output digest. The layer timer covers its routed command, not a
per-kernel GPU interval; this does not identify an individual faulty kernel.

Runner exit1, observer0, `passed=false`, monotonic launcher wall93,715.884ms.
The approximate runner counter `~66s` is not elapsed-time authority. Forty-six
observer samples had no collection errors: first75%, minimum39%, final71%.
No runtime memory kill or runner timeout occurred; sampled pressure does not
exclude a transient driver/resource issue. Known runner/probe/watchdog PIDs
were absent after exit. The attempt is consumed; no retry was performed.

Together with the direct-gate success above, this strengthens the short-row
route hypothesis but does not prove pregate is the root cause. The same binary
and input are controlled; host state, initial headroom, order and runtime
floor differ. The gate override also affects other short constrained batches.
Source selection with the default1024 threshold points to pregate for195 rows;
this is source/control inference, not a captured kernel-level failure. Verdict:
ROBUST bounded reproduction; VULNERABLE as causal proof, speed comparison or
production-default justification. Next narrow step: inspect the failed command's
contents and qualify a same-guard route discriminator before another GPU run.

Artifacts: `/private/tmp/qwen-default-gate.KW6gCL/` (ephemeral). Refresh on
source/model/device/input drift; do not reuse this consumed attempt. SHA256:
- Manifest: `4021ff71428f44e738bdfe4723098ce4430d8519576b524c3d95a3b1cdad3ed0`.
- Stdout: `b6c5a06f07bbc087a698cc37beee540b6fbc0abf4acd21c08dead19b313d9a0b`.
- Stderr: `4203bcfe0bf256b23a174afeab7f7ed7db0dae59122f7ef9c6b37a3a22c09626`.
- Memory: `592745a7e9b81002163738c4d112f01aafcb2b3a6b70d08e3989e57658faba2b`.

## Current admission policy: operator-selected 75% / 30% (2026-09-18)

The operator superseded the earlier 80% initial / 35% runtime policy for this
bounded replay: require initial memory pressure free percentage >=75; stop at
<=30 during execution. These are policy thresholds, not a measured safety
bound. Keep the 24-GiB tree cap, 300-second timeout, 180-second command watchdog,
zero-wait Metal lease, and one-attempt/first-failure rule. No global runner
default changes or unrelated process control are authorized by this slice.
The read-only checker reports the runtime floor; the launcher's environment
must actually set `COGNI_RUN_SAFE_MIN_FREE_PCT=30`.

The control uses the same pinned binary/input with the direct-gate minimum
override removed (source default1024), not the successful override1. The
runtime-floor change is an additional safety-policy difference, so the pair
is not strictly single-variable; if neither guard fires, it does not itself
change numerical routing. Require source/binary/input identities, both call
ends, completion, resident7,839/suffix196 and the recorded output digest.
Below75%, failed pressure query or identity drift rejects before model load.
Tests must reject74 and admit75, retaining malformed-report rejection.
Rollback restores the prior profile; changing host/workload requires review.

Verification: changed-boundary tests failed against the old constants, then
all five test methods passed after the policy update. The new control wrapper
`/private/tmp/qwen-default-gate.KW6gCL/replay.py` checks the original pinned
source/model/binary identity, admits only the two declared control differences,
and passed metadata-only replay. At the actual launch gate free memory was74%,
so it exited75 before creating any GPU-attempt marker or loading the model.
That preparation did not consume an attempt; the subsequent measured attempt
is recorded above and is now consumed. Global runner defaults are unchanged.
The admission tests establish only the tested boundaries, not runtime30%
shutdown or safety under GPU load.

## Operator-admitted 79% replay passed (2026-09-18)

The operator explicitly authorized one attempt at 79% initial free-memory
pressure instead of the temporary 80% target. This is a one-shot exception,
not a change to `qwen_two_call_preflight.py`. Runtime floor35%, tree cap24GiB,
300-second timeout, 180-second command watchdog and zero-wait Metal lease
remained unchanged; no unrelated process was stopped and no retry followed.

At HEAD `d4b4874e` with the pre-existing FFN WIP preserved, the original pinned
binary still matched its source/model/control manifest before and after the
run. No rebuild was needed. Metadata-only input verification passed; 17
SG4/command-trace specs passed with `DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
The default Xcode SDK failed the test link on `arm64e.x1`; this was not a test
assertion failure. An initial sandboxed dry run could not isolate its process
group; it was retained, and dry/run used authorized unsandboxed isolation.

Fresh one-shot wrapper: `/private/tmp/qwen-direct-replay-79.tqxjyp/replay.py`.
It validates the original launcher and binary, writes only to the new directory,
requires at least79% immediately before launch, and retains exclusive attempt
markers. Both call-end events and completion were observed: exit0, observer0,
`passed=true`, launcher monotonic wall101,907.060ms. The runner's approximate
`~72s` counter is not the elapsed-time authority. Fifty observer samples had
no collection errors: first79%, minimum41%, final73%. No memory kill, timeout
or Metal completion error was recorded. Sampled headroom is not a guarantee
against transient pressure or reboot.

Call1: 7,813 prompt tokens, total85,152.8ms, prefill+top1 81,730.5ms,
reported95.59 pp_top1 tokens/s, decode-body8.88 tokens/s. Call2: resident hit
7,839 tokens, prompt8,035, suffix196, capacity8,845; total8,416.5ms,
prefill+top1 5,260.7ms, decode-body8.73 tokens/s, output27 tokens. Its output
SHA256 matched `ed251864987c367e9641fbdc89c1d83e9bf0fa2e3eecef8f301c79f619bfac81`.
The layer trace includes rows195 at start7839 and later constrained rows4;
the direct-gate override is process-global, not a suffix-only intervention.

Verdict: ROBUST for one bounded output replay with the direct-gate override;
not causal proof of pregate failure, tensor parity, general stability or a
speedup. No matched control/ABBA or quiet-host comparison was run. Next useful
discriminator is a separately bounded same-input default-gate control, subject
to fresh memory/identity admission, not promotion of a production default.
Identity drift invalidates this certificate; temporary artifacts may expire.
Manifest/stdout/stderr SHA256 respectively:
`13555464d29f24d95fc53256b7065d321a08ed3ea68e7060d782dca11e70f5d7`,
`100618207a5964ed6c962f3abc632c418fad36e916d0d566ada14dbcba0c4862`,
`0193fa582c40f99e93266d3893dc40ac005bf00612ef99cd271b274024447eb4`.

## Initial-memory admission for this replay (2026-09-17)

Before a new bounded attempt, run `python3 scripts/qwen_two_call_preflight.py`.
Exit 0 admits only this memory snapshot; 75 rejects low headroom; 2 rejects a
missing/malformed report, query failure or non-64-GiB host. This read-only
check launches no workload. It is an explicit prerequisite for the next
one-shot launcher, not a new global rule in `run_safe.sh`; the old consumed
launcher must not be reset or treated as a fresh attempt.

The initial target is **80%** on the `memory_pressure -Q` scale: runtime floor
35% + observed 37-percentage-point decline + chosen 8-point reserve. The
37-point observation uses one observer's 67->30 samples from the stopped run,
not the runner's differently timed 66% preflight. Historical complete runs
had 80->47 and 79->45 samples. These sparse, system-wide observations include
other activity and stop latency; they do not bound future model demand.
The 8-point reserve is policy, not a measured guarantee. Do not turn these
percentages into physical free GiB or add overlapping RSS/Metal/Scratch counts.

Scope: this M2 Max / 64-GiB / Qwen3.8-27B Q4_K_M / ordinary F32 / 7,813-token
two-call replay, with the existing chunk/group/precision controls. The checker
verifies only capacity and the pressure report, not model, device identity or
runtime configuration; the launcher's identity checks are still mandatory.
Keep the runtime 35% guard, 24-GiB cap, timeout, lease and first-failure stop.
A snapshot reserves no memory and can become stale immediately. Recalibrate
or remove this temporary threshold after changed workload/source/device or
better peak-demand evidence; never equate admission with stable inference.

Verification: the new tests first failed because the checker was absent;
after implementation, all five test methods pass, including 79/80 boundaries,
low-memory cases, malformed/duplicate/out-of-range reports, unsupported host
capacity and command failures. The mocked command check confirms only
`memory_pressure -Q` is invoked. Live check returned 75 at 61%, before model
load; no build or GPU attempt followed. The direct-gate question remains open.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s spec -p qwen_two_call_preflight_spec.py -v
python3 scripts/qwen_two_call_preflight.py
```

## Direct-gate two-call discriminator (predeclared 2026-09-17)

Return to the provider's two-call failure, not the parked register experiment.
Hypothesis: the ordinary F32 short-suffix pregate route is necessary for the
observed failure on this replay. Test the existing
`QWEN35_PREFILL_ATTN_ROWS_SG4_DIRECT_GATE_MIN=1` override, without production
code changes. For SG4-eligible rows this selects the existing direct kernel;
the first prompt's expected 2,048/1,668-row batches already meet the default
threshold. The override is process-global: any other eligible short batch
would also change. Inspect the emitted row counts rather than assume a
suffix-only intervention. This is not a new attention layout.
A failure falsifies this route as sufficient recovery under the tested state;
a single success does not establish causality or production stability.

Keep ordinary F32 KV, fusion OFF, FFN capacity reuse OFF, stage splitting OFF,
chunk 2,048 / append groups 1 / cooldown 50 ms, 180-second command watchdog,
35% free-memory floor, 24,576-MiB process-tree cap and 300-second runner limit.
No quiet wait under standing operator authority; no quiet-host/speed claim.
Zero-wait Metal lease avoids overlapping another cooperating GPU workload.
No Xcode replay, profiling, repeated model runs, or unrelated process control.

Before GPU: rebuild with source/binary identity, pass the metadata-only probe
against the saved session/prompt/tools hashes, and run the existing no-GPU
SG4/command-trace specs. Then admit at most one guarded two-call attempt and
stop at the first failure. The probe replays the saved tool result; it executes
no shell tool or code edit. Require both call-end events and completion, the
7,839-token resident hit / 196-token suffix, and the previously recorded second
output digest. This tests output replay, not hidden/KV/recurrent tensor parity
or successful completion of a coding task. Preserve failure outcomes.

Some historical temporary launchers/binaries have expired. The original
session remains available; its identity must match the retained dry-run log.
New artifacts are isolated under `/private/tmp/qwen-direct-replay.zsmdyT/`.
Source/model/device/toolchain drift or artifact loss invalidates reuse of this
experiment's certificate. Rollback is removal of the environment override;
no production default is admitted by this test.

### Result: memory guard stopped call one; route hypothesis remains untested

Source base `66b32027`, CrystalBall `b79052cc63805c9d5502190b2a5067f95afbd646`,
with the pre-existing default-off FFN WIP retained. A fresh release build
passed; a second build pinned a source manifest before/after compilation.
The manifest covers both repositories' source trees, bridge, probe, runner,
sampler and session; model identity is stat-based, not a complete weight hash.
The 17 SG4-safety/command-trace specs passed. Metadata-only replay passed and
matched the retained prompt/tools/session hashes (7,813 prompt tokens).

One guarded model attempt exited 1 after 12,822.552 ms launcher wall.
Runner preflight reported 66% free memory, then issued `[KILL]` at 34%, below
the retained 35% threshold. The observer exited 0, collected seven error-free
samples and observed a minimum of 30%; the guard is sampled/reactive, not a
guarantee that memory never crosses its threshold. Its final sample was 57%.
The probe process was confirmed absent after termination. No repeat followed.

Only `input` and `call_begin(1)` appeared; there was no first-call result,
second-call input, output digest or completion event. The last trace is the
first shared submit/wait at start0/rows2048/cursor0->3. The first chunk already
selects direct SG4 under the default threshold, so this attempt never reached
the intended short-suffix intervention. No Metal `completion_status=-6` was
recorded. Neither improvement nor regression of the direct-gate override was
tested; memory pressure is not evidence for a kernel-specific cause.

The launcher explicitly rejected the nonzero runner exit and records
`passed=false`. Source/model/binary identities still matched after execution.
This is ROBUST evidence of this bounded guard-triggered stop, not proof of
reboot prevention, full GPU cancellation, tensor parity, stability or speed.
No production code/default change or FFN-WIP commit is admitted. Next: recover
adequate initial headroom and budget model plus workspace before a separately
bounded repeat of this still-open discriminator; never lower the 35% guard.
The observed free-memory swing exceeds 30 percentage points, so merely being
above the floor at startup is insufficient for this 27B replay.

Commands (launcher clears inherited experimental switches; no tools run):

```sh
python3 /private/tmp/qwen-direct-replay.zsmdyT/run.py build
python3 /private/tmp/qwen-direct-replay.zsmdyT/run.py dry
python3 /private/tmp/qwen-direct-replay.zsmdyT/run.py run
CRYSTAL_CACHE_DIR=/private/tmp/qwen-direct-replay.zsmdyT/spec-cache crystal spec spec/qwen35_sg4_tail_safety_spec.cr spec/qwen_prefill_command_trace_spec.cr --no-color
```

The first command pins a build; the last qualifies the existing instrument.
The `run` command permits one attempt only and is consumed. Artifacts remain
ephemeral, not portable fixtures. SHA256:

- Binary: `1a118d5ebf4e54b92cd0474801c2807ff38bab873864ba21822770c8dc38a277`.
- Manifest: `26de1d8650a43fb499240923886afc9f0ddbdbe3964c0e25ce1ded849c544a5c`.
- Run stdout: `c782f38d09900fddd99a091c5033b8f0de8dcce4acb7f147b7c2aa7df55b98d2`.
- Run stderr: `134144393e2bba2d26f59e90594b56d60ad21347197159a7754790236fca7a5e`.

## Earlier frontier: one GUI replay exposes compiler statistics, not spill evidence (2026-09-15)

At source state `e1e4562f` plus unrelated WIP, the user explicitly authorized
one replay of the saved model-free trace outside the runner's automatic
limits, with Profile OFF, no repeat execution and no unrelated process control.
Preflight memory free 73%; after inspection 70%. These snapshots are not peak
memory or continuous pressure monitoring. No production source was changed.

Xcode opened `/private/tmp/qwen-sg4-capture.Bwt7d7/register.gputrace` on the
same Apple M2 Max/macOS26.6.2, with Profile after replay unchecked. An initial
UI action was refused because the app state changed; the refreshed UI still
showed the replay landing page. One subsequent Replay entered Preparing frame,
then Debugging GPU Workload without a visible replay error. No second execution,
Profile, Debug Shader, new capture or model load was requested.

The replay Summary reports **1 command buffer, 1 compute encoder, 1 dispatch**;
the API list shows dispatchThreadgroups `{24,16,1}` with threadsPerThreadgroup
`{128,1,1}`, followed by endEncoding, commit and waitUntilCompleted. This now
corroborates command count independently of the capture plist's frame count.
Summary buffer allocation is 76.44 MiB, not total host/GPU process memory.

Navigation: dispatch 15 -> Bound Resources -> MTLComputePipelineState 1 ->
Compute Function -> Statistics. Xcode exposes the following static compiler
fields without Profile:

| Displayed field | Value |
| --- | ---: |
| Instructions | 554 |
| ALU | 387 |
| FP16 / FP32 instructions | 46 / 102 |
| Int16 / Int32 instructions | 107 / 103 |
| Branch / Wait | 39 / 10 |
| Threadgroup Load / Store Instructions | 9 / 2 |
| Device Load / Store Instructions | 11 / 8 |
| Temporary Registers | 40 |

The function is displayed as `qwen35_attn_decode_rows_sg4_pregate`: the probe
uses that entry point for the register-macro variant too, so the label alone
does not distinguish baseline from candidate. Candidate identity depends on
the prior capture log, pinned binary and macro source lineage below. Current
shader, probe and capture-metadata SHA256 values were rechecked and match the
recorded values; the metadata digest still does not cover all trace payloads.

Adversary scope: ROBUST for successful UI replay and these displayed fields
only. There is no explicit spill counter in this Statistics list; 40 temporary
registers does not prove register-only gate storage, occupancy, absence of
spill, or improvement over the uncaptured baseline. Device load/store counts
are not spill counts. The unprofiled Performance value 0.00 ns is not a timing
measurement; displayed descriptor width/thread-limit zeros are not runtime
hardware limits. Replay did not rerun the probe's CPU oracle assertion.

This consumes the single-replay exception. Compiler register count is now
observed, but spills, the cause of prior interactivity failures and a repeatable
speed win remain open. Keep the candidate diagnostic-only. A useful future
comparison would need matched baseline/candidate compiler evidence, not another
unbalanced timing sample; it is not authorized by this one-shot replay grant.
Refresh on source, compilation options, driver/toolchain/device changes or loss
of temporary artifacts. Earlier sections below describe historical boundaries,
not the current replay capability.

## Follow-up: offline diagnostic flags yield no statistics (2026-09-15)

At source state `ee027ccf` plus unrelated WIP, the installed `air-nt -mllvm
--help-hidden` lists `--print-detailed-perf-diags`, `--print-regusage`, and
`--scalar-opt-harvest-stats`. One compile-only experiment passed all three via
`-mllvm`, with the previous register.metallib/pipeline script, applegpu_g14s,
macos26.0/SDK27.0 and one compiler thread. It ran under scripts/run_safe.sh
with120s/8GiB/free35%, quiet wait OFF; preflight free72%, exit0. No compute,
capture, Replay or Profile was launched. Captured stdout and stderr were empty.
This invocation did not deliver the requested register/spill measurements;
empty output is not a zero-spill result or evidence that every compiler route
is unavailable. A descriptor read confirms both expected pipeline names only.

Temporary outputs: `/private/tmp/qwen-sg4-compiler-stats.19GGMy/register.log`
SHA256 `caab6b22a917b098a7d415e5fe26606ddc9771204e44d59b7d2bc94a187c3b9d`;
`register.gpu` SHA256
`8b504c82570fe8d142f2d1fd942777c77d807397a92e38b0374fa418adbf312b`.
Native output differs from the earlier archive; no code-equivalence claim or
runtime promotion follows. No identical diagnostic reruns are warranted.

Containment inspection: scripts/run_safe.sh tracks the workload process group,
not arbitrary launchd/XPC services. Launching another Xcode under that runner
alone is therefore not a containment certificate for its replay services.
Apple's [Replay documentation](https://developer.apple.com/documentation/xcode/replaying-a-gpu-trace-file)
separates replay from optional profiling; a successful replay by itself does
not promise compiler statistics. Preserve the existing trace while resolving
the replay execution boundary; do not terminate or attach guards to the user's
existing Xcode session. Register allocation and spills remain unknown.

Correlated Luna read-only inventory found no supported replay CLI in the
inspected Xcode installation. Parent rechecks: xcrun cannot resolve gpudebug
or gpucapture; GPUDebugger.xcplugindata registers ReplayCapture as the Xcode
GUI action `GPUDebugger_replayCapture:`. This bounded discovery is not proof
that no private or future route exists. Do not execute private agent/XPC
binaries with guessed arguments. The observed GUI route remains outside the
runner's current automatic timeout/RSS/pressure protection. Next decision:
explicitly authorize a single GUI replay with this limitation (Profile OFF,
no retries or unrelated process control), or park this diagnostic candidate.
No GPU execution was admitted in this follow-up. Scoped verdict: ROBUST for
the recorded compile outcome and runner boundary, not for safe replay or spills.

## Earlier frontier: one-command capture opens in Xcode; replay/statistics pending (2026-09-15)

One model-free register-candidate command was captured on Apple M2 Max using
`MTLCaptureManager`, restricted to the probe's default command queue. The
unchanged `--single-command=register` probe compiled the same three pipelines,
then dispatched64 rows at base7839/fixture193, F32/D256/heads24/KV4. CPU oracle
max error1.064646237225464e-6 and future/trailing guards pass; command status0,
runner exit0. Capture timing is instrumented and is **not benchmark evidence**.
No model load, production shader/routing edit, or capture replay was performed.

Temporary evidence: `/private/tmp/qwen-sg4-capture.Bwt7d7/` contains
`capture_bridge.mm`, private `bridge.o`, `probe`, `capture.log` and
`register.gputrace` (about77MiB allocated). Its binary plist metadata records
`captured_frames_count=1`; this is not an independently decoded command count
or a register/spill report. The private wrapper macro-renames the original
`gs_create_command_buffer` and `gs_commit_and_wait_status_gpu_elapsed`, starts
queue capture before creating the command, and stops after the original
watchdog-protected terminal wait. It refuses a second wrapped command, an
existing output path, active capture or unsupported destination. These are
guards for this single-threaded diagnostic, not a general capture API.

Build: `xcrun clang++ -c "$PROBE/capture_bridge.mm" -o "$PROBE/bridge.o"
-std=c++17 -fobjc-arc -fPIC -O2`; then build the existing Crystal probe with
`DEVELOPER_DIR=/Library/Developer/CommandLineTools`, private Crystal cache and
`--link-flags="$PROBE/bridge.o -framework Metal -framework Foundation -lc++"`.
The initial default-Xcode link failed because Crystal's lld cannot parse SDK27
`arm64e.x1` TAPI entries; switching only this build to installed CLT passed.
The unchanged `--self-test` passes without initializing GPU. Shared build
outputs and unrelated worktree changes were not touched.

Executed once, with `PROBE=/private/tmp/qwen-sg4-capture.Bwt7d7`:

```sh
env -u COGNI_METAL_LEASE_PATH MTL_CAPTURE_ENABLED=1 \
  SG4_CAPTURE_PATH="$PROBE/register.gputrace" COGNI_METAL_LEASE_WAIT_MS=0 \
  COGNI_RUN_SAFE_REQUIRE_QUIET=0 COGNI_RUN_SAFE_WAIT_QUIET_SEC=0 \
  COGNI_RUN_SAFE_MIN_FREE_PCT=35 COGNI_METAL_COMMAND_TIMEOUT_MS=180000 \
  scripts/run_safe.sh "$PROBE/probe" 300 24576 --single-command=register
```

Preflight free memory72%; no retries or lowered guards. Log SHA256
`9d937c0488aec4b8eda3e2ec40dfbb04c999666a835f1c1ac092ba9ee433d6a2`;
probe `e30692960b767fdb664ab2e7fb9a46d46ce5724efcf0b5c54f1f9f7a24f6cb99`;
wrapper `d1f05b5dca14f8cf3d7e3a072b8f4b4c885b2c9bbd3016a2df0dd9a691ed0bba`;
capture metadata `487e28f97b9dc9c0a936b1d8f8d9ef91e33a29709e9806e2589b41b2a6542d51`.
The metadata hash does not cover all capture payloads.

On current macOS26.6.2, `xcrun --find gpucapture` / `gpudebug` fail; Apple
[documents gpucapture as macOS27+](https://github.com/apple/game-porting-toolkit/blob/main/game-porting-skills/skills/using-gpucapture/SKILL.md).
Xcode initially opened an **Install Required** dialog. No installation was
initiated by this probe. A later read-only `xcodebuild -checkFirstLaunchStatus`
returned0 and GUI advanced to onboarding; its **External Agent Access** choice
was left to the user rather than confirming the preselected Always permission.
The user then completed setup. Xcode opened this trace and displayed its Apple
M2 Max/macOS26.6.2 origin, a Replay button, and Profile after replay unchecked.
No Replay/Profile action was taken: that would execute in Xcode outside the
existing runner and requires a separately contained workload boundary. Next:
establish that replay boundary, then inspect compiler statistics from this trace.
Do not repeat capture or timing just to obtain favorable numbers.
Register counts, spills, causality of prior interactivity failures and speed
benefit remain unknown. Refresh on source/compiler/driver/device drift or loss
of temporary artifacts. Capture creation is evidenced; useful compiler
statistics remain an open capability, not a completion claim. Parent source/log
checks and correlated Luna read-only audit are ROBUST for this one-command
capture success path only; Xcode recognition is not replay certification.

## Offline native compilation available; spill counts still unknown (2026-09-15)

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
