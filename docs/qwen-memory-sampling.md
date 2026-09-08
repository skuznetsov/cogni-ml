# Read-only Qwen memory sampling

`scripts/qwen_memory_sample.py` observes an **already guarded** process on macOS.
It does not start, stop, signal, or change a workload. Keep the existing
`scripts/run_safe.sh` memory, time, and process-tree limits; the sampler is not
a safety controller. Rollback is simply omitting the observer.

```sh
python3 scripts/qwen_memory_sample.py --pid GUARDED_PROCESS_PID \
  --phase-log /private/tmp/probe.stdout.log \
  --output /private/tmp/new-memory.samples.jsonl --seconds 310 --interval 2
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts -p test_qwen_memory_sample.py
```

The output path must not exist. The observer exits when its root disappears or
its deadline expires. Exit 0 means at least one error-free, root-present sample,
**not** complete coverage: inspect every row's `errors` and timestamp gaps.
Unavailable values are null, never a zero-memory measurement.

Each JSONL row contains system free percentage from `memory_pressure -Q`, a
PPID-linked tree RSS sum from `ps`, the five largest external RSS observations,
monotonic/wall timestamps, collection duration, and the last flushed probe
event. Only process basenames/PIDs/RSS are stored, not arguments or full paths.
Event payloads are not copied. The exact phase vocabulary is:

- `before_call_1`, `call_1`, `after_call_1`;
- `tool_gap`, `second_input`;
- `call_2`, `after_call_2`, `complete`.

These are coarse provider events, not load/prefill/decode/kernel attribution.
Reads are sequential, not atomic. The next collection starts after the selected
interval **plus** collection time; brief peaks can be missed. A malformed or
partially written JSON line is ignored, retaining the last complete event.

RSS is not physical footprint or Metal allocation accounting. Shared pages can
be counted multiple times; reparented children can leave the PPID snapshot.
External top-five membership can change and is not total external memory.
Short bounded runs assume the root PID is not reused; this is not a persistent
process-identity or ownership certificate. Correlation between system pressure
and a provider phase does not establish which allocator caused it.

## Qualification

The CPU positive control used a live parent/child tree and a touched 64-MiB
child allocation. Eight samples observed 65,696 KiB RSS growth, no sample
errors, and null tree RSS after the root exited. It exercised real descendant
capture, separately from parser tests. Temporary evidence:
`/private/tmp/qwen-two-call.HVIZvS/memory_control.py` and
`memory-control.samples.jsonl` in that directory. These are local ephemeral
evidence, not checked-in reproducible fixtures.

Refresh qualification after changes to macOS command output, process ancestry,
sampler semantics, or probe event schema. No inference-engine behavior or
production default changes are part of this diagnostic.

## One guarded unprofiled reuse observation (2026-09-08)

The existing two-call provider probe was run once with profiling off, ordinary
F32 KV, Qwen3.8-27B Q4_K_M on Apple M2 Max / 64 GiB. The unchanged guard was
300 seconds, 24,576 MiB tree RSS, and 35% system free memory. Prefill retained
chunk 2048, append groups 1, cooldown 50 ms. No fresh arm or retry followed.

- Guard exit 1: free memory reached 35%; monotonic launcher wall 65.913 s.
  The runner's `after ~47s` is its loop counter, not this wall measurement.
- Only `input` and `call_begin(1)` were emitted. Neither call completed;
  no second-call, latency comparison, or Metal-error reproduction is certified.
- 33 samples, no collection errors; phases `before_call_1` and `call_1` only.
  Sampled free minimum was 36%, while the independently timed guard saw 35%.
  Maximum observed tree RSS was 1,517,776 KiB (1.448 GiB).
- Median collection time 67.088 ms, maximum 134.603 ms; maximum sample-start
  spacing 2.135 s. This measures observer collection wall time, not its causal
  overhead on inference.
- Free percentage fell from 71% to 39% by the sample at 10.4 s. The large
  system change is not explained by the observed RSS sum. A later post-exit
  query returned 67%; this correlation does not identify an allocator.

Evidence directory: `/private/tmp/qwen-two-call.HVIZvS`, files
`reuse-memory-gpu.{stdout.log,stderr.log,samples.jsonl,json}` and `run_memory.py`.
The launcher verifies the existing probe binary SHA256
`d8de510c3aa9232a8938d1da639561123f57ab33c2edf3441da0ddb6ecea2458`
and source/config/session digests before launch. The emitted 7,813-token first
prompt hash matches the prior probe:
`bb41547af95aee4df74b5532ba8c84254abb289e530a0027e4de563bb5faf552`. Private
session data and raw logs are not committed.

Verdict: the observer is qualified for bounded RSS/free/event snapshots, not
Metal footprint accounting or causal attribution. The run remains
memory-censored during the first call. Next inspect physical-footprint and
Metal allocated-byte measurement before another GPU replay. The existing
`ML::Metal::Device#current_allocated_size` wrapper in `src/ml/metal/device.cr`
already exposes `MTLDevice.currentAllocatedSize` from `src/ml/metal/bridge.mm`;
its diagnostic insertion and coverage remain untested here. Do not weaken the
guard, infer a profiling regression, blame external workloads, or change
production defaults from this observation.
