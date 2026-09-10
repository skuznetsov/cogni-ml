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

## Opt-in in-process inventories

`QWEN35_MEMORY_TRACE=1` emits JSONL `qwen_memory` records to stderr on entry to
the chunked prefill helper, around recursive chunks, and before each eighth
layer on its optimized layer loop. It does not add device initialization,
command submission/waits, pool eviction, padding, or changes to GPU lifetimes.
Omit the flag for rollback. JSON construction and output have observer cost;
do not use this diagnostic to promote a speed comparison.

Fields distinguish three overlapping views; **do not sum them**:

- `pipeline_entries`: Crystal pipeline-cache key count, not driver-internal
  compiler variants, number of unique native objects, or compiled-code bytes.
- `scratch_entries`, `scratch_retained_bytes`, `scratch_largest_tags`: exact-size
  pool entry count and nominal buffer lengths, with the largest eight tag
  groups. Symbol and string tags remain distinct. Multiple sizes under one tag
  increase its entry count. Fresh asynchronous arenas are excluded; aliases
  are not deduplicated, and this is not a physical-residency measurement.
- `metal_buffers`: existing live-buffer count, nominal live bytes and lifetime
  peak, including no-copy wrappers. `metal_allocated_bytes`: existing device
  allocation counter, null before device initialization or if unavailable.

Snapshots run on the existing inference owner thread, not a new sampling
thread. Scratch is read under its existing mutex; the other counters and wall
timestamp are sequential observations, not an atomic global snapshot. These
hooks do not instrument model loading, state construction, every allocation,
single-token decode, or every fallback route. An absent later record can mean
the process was stopped before reaching that boundary. Short peaks and native
compiler memory can remain invisible.

The optimized loop may advance across a fused group of layers in one step.
Therefore the every-eight-index hook samples only visited loop indices; it is
not a per-eight-executed-layer guarantee and does not attribute fused internals.

No-GPU qualification (invalid placeholders never create native handles):

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen-memory-inventory-cache \
  crystal spec spec/qwen_memory_inventory_spec.cr \
  --link-flags="$PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
```

The focused spec checks same-key reuse, new-size retention, symbol/string
separation, detached inventory results, pipeline-key counts, opt-in output,
and unavailable-as-null without initializing Metal. It is a standalone test
process; its fake-cache fixtures must not be used in a live inference process.

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

## In-process inventory observation (2026-09-09)

One newly built release probe used the same captured input and guard settings,
with only the diagnostic flag added. Metadata-only dry validation passed:
7,813 first-prompt tokens and the same prompt/tools/session hashes. The prefill
helper received 7,812 rows (the provider handles the final prompt token
separately), split into 2,048-row chunks. This is ordinary F32 KV, not adaptive
QBit. No model, kernel, allocation policy, or guard was changed.

| Observed boundary | Pipeline keys | Scratch entries | Scratch GiB | Tracked buffer GiB | Device allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill entry | 0 | 0 | 0 | 16.8820 | 16.8831 |
| First 2,048-row chunk end | 31 | 425 | 3.3167 | 20.3159 | 20.3186 |
| Second chunk, layer-loop index 0 | 31 | 425 | 3.3167 | 20.3550 | 20.3577 |

The guard stopped the process at 35% free memory, exit 1, launcher monotonic
wall 24.031 s (runner loop-counter message `after ~16s`). It started at 72%.
There were eight in-process records and 13 external samples with no collection
errors; the final external sample saw the root gone and free memory at 65%.
No call completed and no second provider request ran. There was no retry.

The eight largest recorded scratch tag groups each contain **one** size entry
of 142,606,336 bytes (136 MiB), including full/recurrent FFN gate, up and combined
activation buffers. The visible multi-GiB increase already exists after the
first canonical-size chunk. The next equal-size chunk begins without additional
pipeline keys or retained scratch entries. This weakens the varying-prompt-size
explanation for this observed stop; it does not rule out unobserved native
compiler memory, transient peaks, or later accumulation across other sizes.

Next candidate: audit eager alternative-path FFN allocation before padding.
For example, `frec_full_ffn_comb`, `frec_rec_ffn_comb`, and
`rec_chunk_many_ffn_comb` are allocated before their consumers select in-place
SwiGLU. The default in-place route selects the up buffer instead of the F32
combined buffer. Eliminating these three allocations could save 408 MiB at
this shape, **if** all consumers/fallbacks and lifetime constraints permit it.
That is a static candidate, not an implemented or measured saving. Keep the
kernel math, logical token counts, and fallback semantics unchanged in any
future falsifier. Do not widen buffer aliases across pending commands.

Evidence: `/private/tmp/qwen-two-call.HVIZvS/run_inventory.py`,
`reuse-inventory-{dry,gpu}.{stdout.log,stderr.log}`, GPU `.samples.jsonl` and
`.json`; new binary `probe-inventory`, SHA256
`3df7ba47b0ad8efde650666776059c2ec3151a9a9aec668da9afba5a4fbe39c1`.
Build uses the existing CrystalBall two-call probe and unchanged bridge object,
with the inventory source changes in this commit. Four focused no-GPU specs,
CPU-only trace-no-op evaluation, nine external sampler tests, release build,
and format/diff checks passed. ROBUST for these bounded inventory observations;
root-cause closure, a memory fix, quality parity and speed promotion remain open.
Refresh after code/model/device/toolchain, route, context, or host-load changes.

## Lazy F32 FFN fallback allocation: bounded contract

This slice is limited to `rec_chunk_many_ffn_comb`, `frec_full_ffn_comb`,
and `frec_rec_ffn_comb`. In-place SwiGLU must select the existing up buffer
without requesting its unused F32 combined alternative. With
`QWEN35_SWIGLU_INPLACE_OFF=1`, preserve the original combined tag, size, and
consumer destination. H16 buffers, kernels, token counts, command ordering,
and cross-command reuse are outside this change. No pool eviction is admitted.

Selection happens during the original scratch-setup phase, before this route
encodes work. This avoids moving allocation failure after a profiling
checkpoint that may already have mutated KV/recurrent state. The setting is
sampled during setup, not re-read for every recurrent layer; changing process
environment during an in-flight route is not supported. Existing retained
combined buffers from earlier out-of-place calls are not evicted.

The falsifier is a no-GPU lazy-selection test: default/explicit enabled mode
must not invoke a fallback allocator; disabled mode must return its buffer and
propagate allocation failure. Check all three call sites and compile the real
provider probe. Rollback is reverting this slice; the existing environment
switch restores out-of-place selection and allocation during scratch setup.
Do not claim physical-memory savings, completed-call parity, or speed from
these checks alone. A future fresh-process guarded replay must establish them.

Implementation evidence (2026-09-09): all three call sites now use
`prefill_ffn_activation_buffer` during setup. Keeping selection outside the
recurrent loop also preserves one fallback allocation per route when
`QWEN35_SCRATCH_OFF=1` or fresh scratch is active. The seven focused specs pass:
three lazy-routing/source-order checks in one example, selection identity and
allocator-call counts, failure propagation, and four inventory examples.
The test initially failed on the missing selector. CPU-only compilation/no-op
and format/diff checks pass. The real CrystalBall release two-call probe builds;
its metadata-only run reports `run_gpu=false`, 7,813 tokens and unchanged
prompt/tools/session hashes. No GPU workload ran for this slice.

Build command, run from the neighboring `crystal_ball` repository:

```sh
CRYSTAL_CACHE_DIR=/private/tmp/qwen-two-call-memory-build crystal build \
  scripts/cogni_qwen_two_call_probe.cr --release \
  -o /private/tmp/qwen-ffn-lazy.IzgaS4/probe \
  --link-flags="-framework Metal -framework Foundation -lc++"
```

That local binary has SHA256
`f4558cdabda523f05cb6b3007819df2bad2b2e425203e5ebf67a8eb122418fcc`.
The 408-MiB reduction remains a nominal allocation prediction at the previously
observed 2,048-row shape, not a measured physical-memory or speed improvement.
Next gate: a fresh-process replay under the unchanged 35% free-memory floor,
24,576-MiB tree cap and 300-second timeout, checking inventory and completed-call
output before interpreting performance. Stop on the first guard/Metal failure;
do not lower the floor to obtain a complete call.

Scoped source review: F32 fused/unfused SwiGLU and fused/unfused down-projection
consumers keep their previous selected destination; H16 consumers retain their
separate combined buffer. ROBUST for lazy setup selection under stable route
configuration. End-to-end numerical parity and physical-memory improvement are
not certified by the helper/source-order tests or metadata-only dry run.

## Guarded lazy-allocation replay (2026-09-09)

One fresh-process replay at source `0c314c19`, using the binary above, completed
both calls with exit 0. The 35% free-memory floor, 24,576-MiB tree cap,
300-second timeout, 2,048-row chunks, append group limit 1 and 50-ms cooldown
were unchanged. Quiet preflight remained explicitly disabled as previously
authorized. This is the ordinary F32 KV provider route, not adaptive QBit.
No retry or additional GPU experiment was performed.

At both the first chunk end and the second chunk's layer-loop index 0, the
matched baseline/candidate comparison gives:

| Counter | Baseline | Lazy allocation | Difference |
| --- | ---: | ---: | ---: |
| Scratch entries | 425 | 422 | -3 |
| Scratch bytes | 3,561,280,512 | 3,133,461,504 | -427,819,008 |
| Pipeline keys | 31 | 31 | 0 |

Tracked live-buffer bytes and `MTLDevice.currentAllocatedSize` each decrease
by the same **427,819,008 bytes (408 MiB)** at these boundaries. This matches
the three-buffer prediction and reduces retained Scratch by 12.01%. These
overlapping counters must not be added together or equated with an identical
change in system free/resident memory.

The first call processed 7,813 prompt tokens and produced 27 output tokens;
the probe's captured first-output assertion passed. The second call reused
7,839 cached tokens, processed a 196-token suffix, and produced 27 output tokens
with `resident_prefix_hit=1`. Its content SHA256 was
`ed251864987c367e9641fbdc89c1d83e9bf0fa2e3eecef8f301c79f619bfac81`, matching
both previous successful `fresh-gpu` and `reuse-gpu` outputs in the baseline
artifact directory; all three report an empty second-call tool list.
The recorded tool result was replayed, not executed as an external tool.
This probe does not measure top-2 or embedding cosine similarity.

Launcher monotonic wall time was 100.667 seconds, including a fixed 7.1-second
replay gap. Provider totals were 81.266 seconds and 11.633 seconds. There were
50 external samples, zero collection errors, and a sampled minimum of 47%
free memory. The process started at 80%, versus 72% in the guard-stopped
baseline, so successful completion cannot be attributed solely to this change.
This is not a balanced speed comparison or a general stability certificate.

### Next discriminator: tail-size buffer retention

| Chunk start | Logical rows | Pipeline keys | Scratch entries | Scratch bytes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2,048 | 31 | 422 | 3,133,461,504 |
| 2,048 | 2,048 | 31 | 422 | 3,133,461,504 |
| 4,096 | 2,048 | 31 | 422 | 3,133,461,504 |
| 6,144 | 1,668 | 32 | 494 | 5,676,934,656 |

Equal-size chunks do not grow retained Scratch. The 1,668-row tail adds
2,543,473,152 bytes (2.369 GiB) and 72 entries, versus one pipeline key.
The eight largest scratch tag groups each acquire a second size entry;
each of the six visible FFN gate/up groups grows from 142,606,336 to
258,752,512 bytes. This supports exact-size scratch duplication as the next
target, rather than widespread per-length pipeline compilation. It does not
exclude native compiler memory outside these counters.

Next falsifier: audit a narrow capacity-reuse path that can serve 1,668 logical
rows from existing 2,048-row buffers. Keep dispatch dimensions, strides,
logical token counts, KV publication and DeltaNet steps correct for the actual
rows; do not introduce padded tokens or unsafe aliases across pending commands.
Require per-buffer bounds/lifetime checks and output/state parity before
promotion. The whole 2.369 GiB is not yet certified recoverable. Later small
constraint spans and the second call also create size variants; the last
instrumented boundary has 6,022,161,112 Scratch bytes, not a post-decode snapshot.

Evidence: `/private/tmp/qwen-ffn-lazy.IzgaS4/run_inventory.py`,
`summarize.py`, `reuse-lazy-inventory-{dry,gpu}.{stdout.log,stderr.log}`,
GPU `.samples.jsonl` and `.json`. Execute the pinned launcher with
`python3 /private/tmp/qwen-ffn-lazy.IzgaS4/run_inventory.py --run`; it verifies
the binary, input and source identities before invoking the guarded runner.
Baseline inventory and successful output references remain under
`/private/tmp/qwen-two-call.HVIZvS/`. These are local temporary artifacts, not
a portable fixture; refresh if they disappear or source, model, device,
toolchain, route or context changes.

Adversary verdict: ROBUST for the matched allocation reduction and this
two-call output/cache-hit replay. General speed, reboot prevention, adaptive
KV behavior and capacity-reuse gains remain unproven. No allocation policy
beyond the already committed lazy-selection change was modified in this run.
