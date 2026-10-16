# DRAMA-3 host allocation and command attribution

Status: verified within the declared measurement and unchanged-output scope;
final correlated mini-Quadrumvirate review returned ROBUST.
The 24-job paired measurement, 12-job legacy binary guard and eight timestamp
controls are complete.
Source base: `b3144c85`.
Investigate the inconsistent request-to-WAV timing after generation backing
reuse before changing defaults. The old final comparison had one warm job per
mode and did not identify a cause.

## Measurement boundaries

`DRAMA3_HOST_PROFILE=1` enables a per-context accumulator with fixed event and
counter arrays. Hot measurements use monotonic ticks and a bounded nesting
stack; they create no timer frames, strings, hashes or mutex acquisitions.
Disabled scopes skip the fine clocks/counters. Snapshot-to-JSON allocation
happens after the through-WAV snapshot. This instrumentation changes no shader,
precision, sampler input, queue order, submit, wait or activation readback.

- `Backing`: `MetalBuffer.new`, including any host/driver work in that call.
- `BufferView`: resident wrapper construction, including subviews.
- `Workspace`: slot lookup/growth/reuse; exclusive time excludes nested backing
  and view construction.
- `Pipeline`: native pipeline-cache lookup/compile, including its existing lock.
- `EncoderCreate` and `EncoderAPI`: native encoder construction and the actual
  API calls for pipeline/buffer/bytes binding, dispatch and encoder ending.
- `Retention`: per-command deduplication hash, retained resources and callbacks.
- `CommandCreate` and `CommandSubmit`: command creation and graph enqueue/commit.
- `CompletionWait`: existing normal-path command waits, including graph window
  pressure and terminal draining. Error-only depth-0 abort cleanup is unmeasured.
- `EncoderAcquire`: graph admission and encoder setup. Its inclusive time can
  contain a window-pressure wait. Exclusive time subtracts measured children;
  reporting it all as encoding would incorrectly attribute waiting to encoding.

These are host wall intervals, not CPU utilization or GPU execution time.
They overlap GPU work. Never subtract their sum or a whole-request GPU total
from generation wall time to infer CPU work. API calls can themselves stall in
the driver. Generic dispatch binding/parameter counters cover only that helper;
API binding/dispatch counters also cover direct dense/MoE/attention paths.
Retention counters count leases/callbacks, not distinct physical storage.

The engine samples coarse process GC values at initialization, generation,
codec and postprocessing boundaries even with fine profiling off. Allocated
bytes are cumulative managed-heap allocation deltas; collection count is
Boehm's cumulative cycle counter. They exclude Objective-C/Metal allocations
and are not a typed object census. Fine profiling starts the Boehm cumulative
full-GC timer once: `full_gc_ms` is nullable and measures only full collections
at millisecond resolution. It is neither all GC CPU time nor an extra scope
to add to host timers (collections already occur within those intervals).

The worker already called `GC.collect` between completed jobs before this
change. Its existing call is now timed separately as `worker_gc`; no extra
collection is introduced. Engine through-WAV snapshots exclude report/sidecar
serialization and this post-job collection. Driver request wall time includes
both, and excludes worker startup/model loading. The worker remains sequential.

## Paired experiment

The checked-in driver removes inherited `DRAMA3_*` controls before applying
and recording an explicit allowlist. It fixes the former reduction, precision,
graph and sampler choices. The default mirrored scratch:profile order is
`0:0,1:0,1:1,0:1,0:1,1:1,1:0,0:0`. Each separate worker has one cold request
followed by two warm requests: 24 jobs, four warm observations per condition,
from two workers. Repeats in one worker are correlated. Adjacent worker pairs
reverse scratch order; profile conditions have the same mean temporal position.
This reduces order bias but cannot eliminate shared-host or nonlinear drift.

All requests use “Hello, this is a voice test.” without a reference, seed 1,
maximum 64 frames, temperature 0.8, top-k 30 and top-p 0.9. DAC scratch, mmap
sharing and tower embedding fusion stay off. Detailed allocation clocks and
GPU timeline output stay off in both profile conditions so the on/off guard
does not confound the fine profiler with timeline logging.

Run serially under `scripts/run_safe.sh`, preserving 15% free memory, with
quiet-host enforcement off and unrelated processes untouched. Hash codes,
sampler inputs and WAVs only after request timing. Each job must preserve the
saved full 60-frame/11-ID trajectory, byte-identical WAV and existing queue/read
ledgers. Cold and warm observations must remain separate.

## Measured host costs

The complete primary matrix preserved every frame/ID, WAV byte, sampler input
and graph/read ledger in all 24 jobs. Each table cell below is a separate median
of four warm observations from two workers; those repeats are correlated. All
times in the next table are **milliseconds**, from fine-profile-on jobs.

| Generation host interval | Fresh | Reuse |
| --- | ---: | ---: |
| Backing construction, exclusive | 770.283 | 8.201 |
| Buffer/view construction, exclusive | 86.524 | 0.887 |
| Workspace lookup, exclusive | absent | 2.591 |
| Encoder API, exclusive | 54.621 | 46.284 |
| Resource retention, exclusive | 16.881 | 19.733 |
| Pipeline lookup, exclusive | 4.784 | 3.501 |
| Encoder creation, exclusive | 0.971 | 0.930 |
| Command creation, exclusive | 0.887 | 0.856 |
| Command submission, exclusive | 1.106 | 0.881 |
| Encoder acquisition, exclusive | 1.126 | 1.084 |
| Existing completion waits | 17751.749 | 18214.742 |
| Full-GC cumulative wall delta, overlapping other scopes | 1222.5 | 20.5 |

Full-GC generation deltas range from 746 to 1706 ms fresh and 13 to 27 ms
reused. Generation managed allocations are approximately 95.966 -> 74.731 MB
(decimal), with 57 -> 1 collection cycles. Thus backing reuse leaves substantial
managed allocation. Measured workspace lookup and retention are too small to
account for four seconds as additive host work; this measurement does not rule
out indirect timing effects. These clocks overlap
device work and GC; they are not an additive latency budget or CPU-time profile.

Both modes encode 156163 generation dispatches, 482849 buffer bindings and
156163 byte bindings (962108 measured encoder API calls, 90 encoder endings).
Buffer/view constructions fall 163873 -> 31772, backing constructions
135139 -> 1894. Pending-list entries are 538971 -> 540832; per-command unique
retained leases/callbacks are 172938 -> 172818. Reuse therefore still creates
command metadata and callbacks; these are counters, not a heap-object census.

The existing post-job full collection has warm fine-profile-off wall medians
29.859 ms fresh and 106.313 ms reused. It is somewhat more expensive with reuse
in this run, but the roughly 76 ms difference cannot account for several seconds.
No collection was moved into inference or added by this experiment.

## End-to-end timing and audio rate

| Fine profiling | Scratch | Warm request median, seconds | Warm range, seconds | Median request seconds / audio second |
| --- | --- | ---: | ---: | ---: |
| Off | Fresh | 26.853 | 22.891–29.065 | 9.637 |
| Off | Reuse | 28.466 | 25.804–31.111 | 10.216 |
| On | Fresh | 25.252 | 22.772–29.969 | 9.063 |
| On | Reuse | 25.695 | 21.990–29.390 | 9.221 |

The waveform is exactly 122880 samples at 44100 Hz, or 2.786394558 seconds.
The operational rate with profiling off is therefore about **9.6–10.2 seconds
of request wall time per second of audio** at the two medians; observed warm jobs
span 8.2–11.2. This includes WAV writing, report/sidecar serialization and the
worker's existing post-job GC, and excludes model loading. It measures this
short no-reference prompt on the shared M2 Max host, not a long-speech rate.

Warm off-profile generation medians are 19.843 s fresh / 21.454 s reused; codec
medians are 6.111 / 6.892 s. The four matched warm reuse-minus-fresh request
deltas are +0.073, +6.384, -3.175 and +4.903 s with profiling off, and -0.579,
-0.782, -0.645 and +1.529 s with profiling on. Off-profile cold request medians,
excluded from warm rates, are 38.036 s fresh / 34.958 s reused, also excluding
model loading.

The off-profile reuse median is still slower in this bounded series. Order
reversal and repeated jobs do not establish a stable effect size or its cause:
same-mode repeats vary by seconds, including waits and codec work whose
dispatch/queue counts are unchanged. Profiling-on results cannot be used as a
speed promotion or a negative-overhead claim. The primary matrix measures
fine-clock perturbation inside the instrumented binary; the disabled facade
still runs in profile-off jobs. A separate legacy/current ABBA guard measures
that fixed disabled-instrumentation cost. Keep generation scratch opt-in.

The separate 12-job legacy/current/current/legacy control also preserved every
frame, WAV byte, sampler input and queue/read ledger. All arms use fresh
backings and fine profiling off. Its four correlated warm observations per
binary have request medians of 28.755 s legacy and 25.123 s current, with ranges
27.344–31.225 s and 23.596–31.896 s. Matched current-minus-legacy differences
are +4.229, -6.864, -1.459 and -6.247 s. This does not establish a repeatable
multi-second disabled-instrumentation penalty, zero overhead, or a speedup of
the instrumented binary. It is a control for the measurement implementation,
not another scratch comparison.

## Existing GPU timestamp control

A separate fresh/reuse/reuse/fresh series runs one cold and one warm request
per worker with fine profiling and the existing GPU timeline enabled. All eight
jobs preserve the same complete outputs and queue/read ledgers. The four warm
rows below remain separate from the primary timeline-off matrix; timestamps
and their stderr serialization are themselves instrumentation.

| Arm | Scratch | Generation wall, s | Host completion waits, s | Generation command duration sum, s | DAC wall, s | DAC command duration sum, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Fresh | 25.099 | 23.945 | 15.532 | 6.093 | 6.055 |
| 1 | Reuse | 20.146 | 20.007 | 15.528 | 6.452 | 6.034 |
| 2 | Reuse | 19.900 | 19.756 | 15.509 | 6.195 | 6.017 |
| 3 | Fresh | 20.302 | 19.151 | 15.529 | 6.063 | 6.022 |

The fixed trajectory has 90 generation commands (including pending initialization)
followed by 32 DAC commands. The GPU duration sums stay close in both orders,
while generation waits vary by seconds. The matched request reuse-minus-fresh
differences are -4.566 s and -0.226 s here. In this eight-job series, aggregate
command-duration sums were similar and dispatch counts were unchanged; this
does not rule out smaller or intermittent effects or identify a unique cause
of the earlier slower request. Queue admission,
resource residency, driver completion and host/device scheduling remain possible
sources of the wait variation. Their individual contributions were not measured.
These sums are not occupancy or pure arithmetic time and must not be subtracted
from host wall intervals to infer CPU time or queue idle time.

The next high-value performance probe is GPU execution inside generation
(backbone/tower kernel attribution), with DAC as a separate approximately
six-second GPU-command workload for this short output. The measured tens of
milliseconds in encoder/retention bookkeeping do not support prioritizing a
broad host-object rewrite. Preserve the older upstream numerical gates for
any kernel or fusion experiment; no such optimization is admitted by this report.

## Correctness and source scope

The final release binary passes 33 CPU examples and 51 resident Metal examples,
including profile-off/on queue/read transparency at graph depths 0 and 2.
All 44 full worker jobs use graph depth 2: 122 submissions/completions/waits,
zero pending commands at the terminal boundary, maximum two pending commands,
496 control bytes and 494160 terminal bytes. Full codes, effective sampler tape
and WAV SHA256 are exact in every job, including the six legacy-binary jobs.

The full 60-frame/11-ID native trajectory matches the saved pinned MLX F32-core
control on GPU with the same effective F32 sampler inputs. This is a comparison
against an existing control, not newly regenerated MLX audio or stock-RNG parity.
The upstream checkout is clean at `e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`;
all 19 control source hashes and checkpoint config/index pins were refreshed.

The current-source 158-row prefill at capacity 2205 also passes. Fresh/reused
hidden and logits are bitwise equal on GPU, with zero nonfinite values or
pointwise MLX failures under unchanged `3e-5 + 3e-4 * abs(reference)`. Worst
error/allowance ratios are 0.05126616 for hidden and 0.0079710195 for logits.
Each arm retains its existing 159 submits/waits; reading terminal reduced
metrics adds no graph work. Reused backbone and tower.prefill scopes allocate
zero new backings after their first use. These checks do not close the older
MoE39/common-prefix or DAC numerical frontiers.

Raw local receipts are under
`~/.cache/cogni-ml/drama3-runs/host-attribution-20261016/`: `commands.json`
contains exact argv, controls, source and log hashes; `paired-worker.json`,
`legacy-worker.json` and `timeline-worker.json` contain the observations;
the three `*-analysis.json` files retain individual values and paired deltas.
The current binary SHA256 begins `d4bdc5972790`; `source-seal.json` checks 42
production files across the final build, all three series, final suites and
prefill commands. Hash-matched snapshots preserve the primary and legacy driver
versions; the checked-in driver adds the separate timeline option.
Receipts, binaries and audio remain local cache artifacts and can be evicted.
The curated index is `.landmark/evidence/drama3-host-attribution.json`.

## Reproduce and refresh

Build the release studio as in [drama3-studio.md](drama3-studio.md). Run
`scripts/drama3/measure_host_pairs.py` as shown in the reference-tools README.
Add `--legacy-binary "$OLD_STUDIO_BINARY" --prefix legacy` for the fresh,
profile-off legacy/current/current/legacy guard (12 jobs, eight warm). It uses
the same controls and full output/ledger validation; the old report need not
contain the new profiling fields.
For a separate existing-timestamp control, use `--timeline --prefix timeline
--schedule 0:1,1:1,1:1,0:1 --warm-repeats 1`. This changes telemetry at the
existing completion boundary only; keep these observations separate from the
timeline-off matrix. Completed-command timestamps are not occupancy, kernel
instruction counts or an additive complement to host wall intervals.
Use `DRAMA3_HOST_PROFILE=1` for diagnostic studio runs; leave it unset/off for
ordinary use. `DRAMA3_GENERATION_SCRATCH` remains opt-in. Do not run another
benchmark on the same device concurrently.

Refresh timing and correctness evidence after source/shader/checkpoint/input/
sampler/precision/gate/graph/runtime/device/OS changes or contradictory results.
Disabling the profile flag removes fine clocks/counters; the encoder facade and
coarse GC snapshots remain. Revert this isolated commit to remove all new
instrumentation. Older MoE39/common-prefix and DAC numerical frontiers, long speech,
prosody, voice identity, cross-job pooling and whole-pipeline GC removal remain
separate obligations.
