# DRAMA-3 first-prefill startup attribution

The bounded experiment separates short host preparation from a long Metal
driver scheduling duration and GPU execution. Cold process page activity is
large, but page-in alone cannot explain the warm spread: several warm requests
spend about four seconds of process system CPU after commit with zero recorded
page-ins or disk reads. This slice adds default-off diagnostics; it does not
promote an optimization or change inference precision.

## Observation contract

`DRAMA3_METAL_RESOURCES=1` enables process resource snapshots and implies the
existing `DRAMA3_METAL_TIMELINE`. CogniGraph depth must be 1 or 2. The ordinary
disabled path retains its existing commit/wait behavior. Snapshots bracket
command encoding, commit and completion; the studio also records request,
generation and WAV boundaries. They collect `getrusage` user/system CPU,
`TASK_EVENTS_INFO` page-ins/faults/COW faults and `proc_pid_rusage` disk-read,
resident and footprint accounting. API groups can be independently unavailable.

The native timeline reads Metal's `kernelStartTime`/`kernelEndTime` after the
same existing completion wait. Their difference is the CPU-side driver
scheduling duration. GPU execution uses the existing GPU start/end pair. No
callback, `waitUntilScheduled`, warm-up command, page purge, activation readback
or inference synchronization is added. Authoritative completion status remains
independent of optional telemetry validity and serialization.

Apple documents [kernelStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelstarttime)
and [kernelEndTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/kernelendtime)
as host scheduling times, and [GPUStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)
as mach-relative. This report uses pair-local scheduling duration; it does not
assume the kernel timestamp epoch matches the mach clock or subtract it from
GPU start to manufacture a pure queue delay. Similar durations do not by
themselves prove interval alignment.

Process counters include all threads and do not certify GPU/driver page
residency. Zero page-ins or disk reads do not exclude driver-managed memory
work. Page-ins are not converted into seconds. Disk-read accounting does not
prove physical-media traffic or a controlled filesystem-cache state. Resident
and footprint values are gauges, so signed changes are allowed. Decreasing
counters are unavailable for the affected window.

With depth two, an earlier command's commit-to-completion resource window can
include the next command's preparation. These windows are nonadditive. The
offline analyzer emits min/mean/max distributions rather than sums; request
totals use the studio's own boundary snapshots. The mini-Quad review caught
and rejected the initial additive-window interpretation before closure.

## Counterbalanced experiment

Base source revision: `ccb0d178`. Four sequential workers use resource observer
`0,1,1,0`, with host profiling and timeline on in both arms. Each worker has one
new-process request and two warm requests. Other controls remain fixed: native
dense tree, expert width 64, MAP_PRIVATE, CogniGraph depth two, generation/DAC
scratch off. The free-memory threshold is configured at 15%, quiet-host gating
is off, and unrelated processes remain untouched. Weights stay on the local
SSD; builds, outputs and reference data use `/opt/mldata_ssd/cogni-ml`.

“Cold” below means a new worker and mappings after model load, not a purged
filesystem cache. The model-load interval is excluded. The fixed text is
“Hello, this is a voice test.”, no reference, seed 1, frame cap 64, temperature
0.8, top-k 30 and top-p 0.9. All requests emit 60 frames and 122880 samples at
44100 Hz, or 2.786394558 seconds of audio.

First-command intervals in seconds; process CPU and counters cover that
command's `after_commit` to `after_wait` snapshot window. Counter windows may
include preparation of the next in-flight command.

| Observer-on worker / request | Entry to first commit | Driver scheduling | Before-GPU wait | GPU execution | System CPU | Page-ins | Disk-read bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / cold | 0.0181 | 11.9583 | 11.9511 | 0.1696 | 3.5112 | 1464429 | 23994392576 |
| 1 / warm 1 | 0.0107 | 4.8326 | 4.8255 | 0.1632 | 4.8165 | 0 | 0 |
| 1 / warm 2 | 0.0113 | 0.3496 | 0.3412 | 0.1954 | 0.3483 | 0 | 0 |
| 2 / cold | 0.0247 | 11.3804 | 11.3719 | 0.1662 | 3.5109 | 1464429 | 23994269696 |
| 2 / warm 1 | 0.0112 | 3.9096 | 3.9023 | 0.1686 | 3.9090 | 0 | 0 |
| 2 / warm 2 | 0.0139 | 4.6349 | 4.6256 | 0.1684 | 4.6183 | 0 | 0 |

The request-entry and first-before-commit sampling uncertainty is 22–31
microseconds in these rows. Commit itself takes tens of microseconds. Host
request preparation before generation takes 2.24–7.80 ms; first-command
encoding is also measured separately. Those short intervals cannot account
for the observed seconds before GPU execution.

The warm observation refutes page-in as the sole explanation for this spread.
It also refutes an interpretation of the whole delay as idle waiting: the
process accrues approximately one core-second of system CPU per second during
the long window. The closely matching Metal scheduling duration supports
driver-side preparation/validation as the next investigation target. It does
not identify the work inside the driver kernel, prove the CPU and GPU clocks align, or
exclude concurrent GPU contention.

Whole-request times include WAV/report work and the worker's existing post-job
GC. They are context for observer perturbation, not a speedup result:

| Worker / observer | Cold | Warm 1 | Warm 2 |
| --- | ---: | ---: | ---: |
| 0 / off | 39.5087 | 28.4916 | 23.1791 |
| 1 / on | 36.6479 | 27.9442 | 24.3500 |
| 2 / on | 35.9465 | 28.1317 | 29.8210 |
| 3 / off | 37.4721 | 26.5682 | 24.1310 |

Observer-off first-command scheduling durations also vary: 14.6938/5.6077/
0.3867 s in worker 0 and 11.1871/3.6771/0.3628 s in worker 3. The delay exists
without resource sampling. Each observed request has 492 snapshots; their
nonoverlapping collection calls total 4.34–4.95 ms of wall time. This excludes
JSON/logging costs and indirect pacing effects. The small, noisy matrix does
not bound total instrumentation overhead or establish a performance gain.

## Separate CPU stack discriminator

An additional worker runs the same cold request and two warm requests. Only
its own PID is sampled with `/usr/bin/sample <pid> 8 10 -file <path>` during
each warm request. These perturbed runs are excluded from the paired matrix.
All three complete outputs and sampler/read/wait ledgers remain exact.

The first warm request records 5.2024 s of first-command driver scheduling,
5.1730 s process system CPU, zero page-ins and zero disk-read bytes. The next
warm request records 0.3423 s, 0.3411 s, zero and zero respectively. In both
captures, the main thread is observed in the existing completion wait, and
`com.Metal.CommandQueueDispatch` has this stack:

```text
_MTLCommandQueue _submitAvailableCommandBuffers
IOGPUMetalCommandQueue submitCommandBuffers:count:
IOGPUMetalCommandQueue _submitCommandBuffers:count:
IOGPUCommandQueueSubmitCommandBuffers
iokit_user_client_trap
```

This identifies the submission corridor to investigate. It does not identify
validation, page-management or scheduling work inside the kernel trap. Stack
sample counts are not CPU time, exact interval durations or a partition of the
first command; each eight-second sample includes subsequent work. The runtime
reported by the sampler is macOS 26.6.2 (25G83), on Apple M2 Max.

## Verification and numerical scope

All 12 matrix requests preserve the complete 60-by-11 code stream, WAV SHA-256
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`, and
704 F32 sampler inputs with digest
`e13a73aa1f41d156e7c2f811bb941c1b63993aa97989b267a14b65a5665a545b`.
Every request retains 122 submissions/completions/waits, depth two, no pending
work, 496 control bytes and 494160 terminal readback bytes. Resource telemetry
does not read model buffers.

The native fake-command test preserves one commit/wait for valid, unavailable
and failed timings. The Metal suite passes 63 examples, including observer
off/on read/output/queue transparency at depths one and two and the unsupported
depth-zero guard. Six CPU examples check independent availability and explicit
JSON nulls. Twenty-three offline tests include malformed and reset counters,
invalid clocks, legacy records and overlapping-command summaries. Early tests
caught omitted-null serialization and a syntax error in the new environment
cleanup; corrected runs pass, and failed receipts are retained.

The saved pinned MLX F32-core control uses the same F32 sampler tape. All
60-by-11 codes compare exactly on the MLX GPU. This is a saved shared-input
control, not fresh stock-MLX audio or RNG parity. Checkpoint revision is
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`; MLX-audio revision is
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`.

The unchanged 158-row fixture checks 2048 final hidden values and 4097 main
logits on GPU. Fresh/reused native outputs are bitwise equal; upstream
pointwise failures and nonfinite counts are zero. Worst error/allowance ratios
are 0.05126616 and 0.0079710195 at unchanged `atol=3e-5`, `rtol=3e-4`.
Both native arms retain 159 waits, and the terminal GPU comparison makes 319
in total; only 16 control and 40 reduced diagnostic bytes are read. The known
full intermediate MoE39 failure remains open. No general voice, prosody,
long-speech, stock-RNG, DAC or full-intermediate numerical closure follows.

The final mini-Quad adversary verdict is ROBUST for this bounded diagnostic
and attribution scope. The review is same-lineage and correlated; it checked
the source, artifact hashes and test receipts rather than rerunning the tests.

## Evidence and continuation

Raw local artifacts are under
`/opt/mldata_ssd/cogni-ml/drama3-prefill-start-20261016/`:
`resources-worker.json`, four selected worker stderr streams,
`startup-analysis.json`, `cpu-stack-analysis.json`, two own-process CPU samples,
`commands.json`, upstream results and build/test logs.
The curated index is `.landmark/evidence/drama3-prefill-startup.json`.
The paired run's wrapper was interrupted; its child completed independently.
Its PASS is established by the driver log and validated 12-job manifest, not
an invented wrapper exit code or a claim of continuous wrapper memory guarding
after the interruption. Production source hashes match the recorded
release build, and every worker's binary hash matches that build artifact.

Refresh these observations on source, shader, checkpoint, input/tape, fixture,
tolerance, graph, storage, runtime, device or OS changes, contradictory
results, or eviction of the local evidence. Keep the known numerical return
pointer at MoE39/common-prefix and DAC post-module checks. Rollback is disabling
resource/timeline flags or reverting this isolated diagnostic commit.

Next falsifier: compare stable resource identity/registration with fresh Metal
resource wrappers over the same weight mappings and fixed sampler trajectory.
Retain the existing queue/read boundaries and unchanged upstream gates; first
establish which resources the submission path actually revalidates. No broad
buffer-lifetime or residency rewrite is justified by these observations alone.
