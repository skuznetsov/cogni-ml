# Wan asynchronous command segmentation

An optional caller-owned `MPSCommandChain` now continues between Wan block
phases and attention query-group ranges. The encoder-only API remains the
default. This changes command submission, not tensor precision or the
per-query ascending key reduction. It does not establish a command-duration
bound or complete-video throughput.

`MetalOps.attention_tiled_range` dispatches contiguous global query-tile/head
groups. The chain overload divides these groups into ranges of 4,096 by
default. Every query retains its complete key range and reduction order;
output ranges are disjoint. There is no quadratic attention-score allocation.

Block and transformer callers refresh the chain's current encoder after each
continuation. A borrowed encoder becomes invalid after a continuation. The
caller must retain all arenas, weights, trace destinations and plans until the
single terminal wait. Encode creates no `MetalBuffer`, reads no activation
and performs no host wait. Separate command buffers still allocate Metal
service objects; the buffer counter does not measure those objects.

## Executed checks

On Apple M2 Max, the external fixture
`evidence/wan-full/attention-tiled-qualified/manifest.json` exercises query/key
tails, multiple batches and heads, sharp logits, near ties and 36,608 keys.
Two attention examples pass. Nonuniform ranges and continued roots are
bitwise equal to the corresponding unsplit native kernel. They retain the
existing finite upstream per-value gate:
`abs(actual - expected) <= 2e-5 + 2e-5 * abs(expected)`.

The combined block, command-chain, stage-pool, stage-I/O and primitive
regression passes **29 examples, zero failures, errors or pending examples**.
The block uses the original BF16 checkpoint fixture at
`evidence/wan-official-block0/wan-official.json`, production channel widths,
tiled attention and `WAN_S2V_SEGMENTED_COMMANDS=1`. It checks upstream block
and injector intermediates, repeated and chained work, no encode-time buffer
creation, zero encode-time waits and one terminal wait. An F32 synthetic
fixture cannot be substituted while requesting resident BF16 weights.

Logs live under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-full/`:
`attention-segmented-native.log` and
`segmented-regression-official-native.log`. These are component correctness
checks, not full spatial performance measurements.

## Full spatial limit remains open

The earlier unsplit full spatial block/injector command failed with Metal
`Impacting Interactivity` after an 89.41-second unsuccessful GPU span.
Terminal-only telemetry now records the failed segment's status, bounded
error text and existing GPU timestamps without an extra wait.

The segmented probe, with 36,608 tokens and 512 text context tokens, did not
complete under a finite 240-second runner budget and exited with status 1.
The wrapper watchdog did not preserve a terminal command receipt. This is
neither a completed block timing nor evidence that a particular projection
or attention dispatch caused the delay. Dense projections still issue one
large scalar dispatch; their cost must be isolated before selecting the next
backend. No full five-second video or cold/repeat timing is established.

The scoped adversary verdict is ROBUST for internal encoder refresh, range
coverage, caller-owned lifetimes and the tested single-wait contract, and
VULNERABLE for any full spatial watchdog or throughput claim. These checks
decay with changes to dispatch geometry, reduction order, encoder ownership
or scratch aliases. Refresh the attention and combined regressions before
reusing them. Rollback omits `execution_chain` and uses the encoder-only path.
