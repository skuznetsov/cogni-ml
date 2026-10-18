# DRAMA-3 captured projection replay

The diagnostic executable captures the real decode projection operands on GPU
and replays production dense/expert kernels in homogeneous commands. On the
pinned short request, expert down projections are the most expensive individual
captured family, followed by expert gate/up and linear-attention qkv. This is a
local command-cost ranking, not a full decode profile or a request speedup.
No production source, selector, arithmetic, tolerance or default changed.

## Capture and ownership

Source base is `0f9d6d3a`, Apple M2 Max. The fixed request is
`Hello, this is a voice test.`, seed1, max64, temperature0.8, top-k30, top-p0.9.
Checkpoint: `cguldogan/drama-3-mlx-q4` at
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
The saved native trajectory is a correctness control, not a timing baseline
or a fresh stock-MLX generation.

`examples/drama3_decode_replay.cr` clears inherited DRAMA controls, pins graph
depth2, dense tree reductions, native 64-thread experts, transpose delta, private
mapping, coalesced DAC, and disables scratch and the experimental fusions.
Inference/allocation/host/timeline profiling is disabled during capture.
`examples/support/drama3_decode_replay.cr` intercepts matvec/expert/experts only
in this executable. Inputs and selected expert IDs are copied to fresh GPU
storage before the original call; outputs are copied before the caller can
mutate them, including in-place RoPE. No activation snapshot is read to CPU or
serialized. Snapshot dispatches are excluded from omitted-operation counters.

Capture rows0/29/59 follow the generator's zero-based decode iterations: tower
generates that row, and backbone updates hidden state for the following main
selection. Each row includes all 40 backbone layers, all four shared tower
blocks and all nine residual tower blocks. The exact name inventory is checked
before writing the manifest and again before replay:

| Work per selected row | Calls |
| --- | ---: |
| 30 linear-attention layers, qkv/z/b/a/out | 150 |
| 10 full-attention layers, q/k/v/o projections | 40 |
| 40 MoE blocks, routed/shared projections and gates | 320 |
| Tower projections and nine codebook heads | 87 |
| Main head | 1 at rows29/59; omitted at row0 |

The first main head was already encoded when `prompt.main_logits` enabled the
observer. It is explicitly omitted, not replaced by a synthetic call. The
capture contains 597/598/598 records, 1,793 total, using 41,924,840 snapshot
bytes. Missing, duplicate, unknown-layer and wrong-row names fail closed;
kernel/reduction/dtype/quantization/threadgroup mixtures within one timed
family also fail closed. These guards apply to this fixed architecture and
request, not arbitrary checkpoints.

## Measurement contract

For each frame/family cell, two identical production arms are queued, followed
by a GPU comparison command. Both arms are checked against the immutable output
from the original call, for every repetition. A seeded bit difference and NaN
must produce counters `[2,1]`. Neither arm is an optimization candidate.

Two warmup rounds precede seven measured rounds, alternating arm order and
forward/reverse frame/family order. Each arm contains one pass over its captured
records. Existing completed-command timestamps measure only the projection
commands; snapshot and comparison commands are excluded. The bounded diagnostic
pair drains three commands before reading its 8-byte receipt, with 8 extra bytes
for the single negative control. These are diagnostic waits and receipts;
capture itself preserves the normal inference wait/read ledgers.

The offline analyzer checks the complete schedule, timing validity, command
indices, exact/finite counters, negative control, consecutive graph/readback
ledgers and final totals. It reports identical-arm variation rather than a
speedup ratio. Rank1 means highest local paired median cost. The producer's
exact name inventory is CPU-tested and executed before timing; the analyzer
alone does not authenticate the manifest or checkpoint payloads.
Kernel labels are derived from the current Context dispatch branches, rather
than a pipeline-ID trace. Refresh that mapping after any dispatch change.

## Observed costs

The final role-split run, `attempt3-roles`, has 47 cells, 423 pairs and 1,269
diagnostic commands. Each table entry is the median of seven paired arm means,
in milliseconds for the complete family at that selected decode row.

| Captured family | Row0 | Row29 | Row59 |
| --- | ---: | ---: | ---: |
| Expert down | 16.292 | 16.142 | 16.242 |
| Expert gate/up | 12.753 | 13.047 | 12.967 |
| Linear-attention qkv | 8.605 | 8.719 | 8.649 |
| Tower attention projections | 7.496 | 6.994 | 7.189 |
| Tower FFN w2 | 5.878 | 5.913 | 5.874 |
| Tower FFN w1/w3 | 5.698 | 5.743 | 5.707 |
| Linear-attention out | 5.551 | 5.520 | 5.785 |
| Full-attention projections | 5.264 | 5.355 | 5.316 |
| Linear-attention z | 4.674 | 4.923 | 4.622 |
| Tower eh_proj | 3.618 | 3.897 | 3.800 |
| Shared expert | 3.527 | 3.536 | 3.549 |
| Router and shared gate | 1.984 | 1.901 | 1.944 |
| Tower heads | 1.724 | 2.211 | 1.418 |
| Linear-attention a | 0.704 | 0.735 | 0.704 |
| Linear-attention b | 0.687 | 0.699 | 0.735 |
| Main head | omitted | 0.290 | 0.350 |

Median identical-arm relative differences are 2.59–2.89% for down,
0.97–1.99% for gate/up and 3.19–3.36% for qkv. The small a/b and router/gate
commands have roughly 12–15% variation; their tiny differences do not support
a precise ordering. Keep every measured pair; there is no best-run selection.

The earlier `attempt2` grouped all five linear roles together: its local
paired medians were 19.486–19.866ms, versus 15.992–16.385ms down and
12.868–12.971ms gate/up. Splitting roles changes command/cache domains; do not
add the role timings or compare their sum as a speedup over that earlier group.
The initial failed attempt rejected a misclassified `tower.*` projection;
no timing from that failed run is admitted.

Attention execution, recurrent delta state, routing/reduction, sampler,
normalization, embeddings and other direct encoder paths remain unranked.
Observed omitted generic dispatch counts are saved in the manifest, with an
explicit unobserved-direct-path list. Do not subtract these projection timings
from the previous 11.1s decode command-window union, add them into a request
cost, or claim they establish the largest bottleneck of the entire request.

## Numerical and synchronization controls

Both successful captures reproduce all 60 frame-code rows, sampler tape and
the complete 122,880-sample WAV exactly. SHA256:
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`.
Capture retains 122 submissions/completions/waits, depth2/max-pending2, no final
pending work or retained graph resources, control496B and terminal494,160B.
The diagnostic snapshots add allocations and copies; capture wall time is
not normal-request throughput.

The final replay checks 55,616,202 values per arm with zero bit mismatches or
nonfinite pairs. Its seeded control returns `[2,1]`. Final command counts are
1,391 including the ordinary request; control stays496B and terminal becomes
497,552B, exactly 3,392 extra receipt bytes. No diagnostic activation reads
are admitted. Peak graph-retained bytes include conservatively retained mapped
resources and are not the live snapshot-memory footprint.

The release tower upstream harness was rebuilt and rerun on the pinned
87-row fixture, with MLX revision `e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`.
All 2,025 comparison records and graph/readback metrics equal the saved report;
all 675 native off/on sites are exact. Strict upstream compatibility remains
false: the same three late residual8 sites exceed the unchanged tolerances
(six mirrored baseline/candidate records). This is bounded no regression;
the known backbone/DAC numerical frontiers remain open.

## Reproduce and continue

Build with `--release -Ddrama3_metal` and the existing bridge/framework flags,
then run serially under `scripts/run_safe.sh`, using external scratch, local
weights and the authorized 15% memory floor:

```text
drama3-decode-replay MODEL_DIR BASELINE_JSON NEW_OUTPUT_DIR
python3 -B scripts/drama3/analyze_decode_replay.py EVENTS_JSONL TIMELINE_JSONL NEW_ANALYSIS_JSON
crystal spec spec/drama3_decode_replay_coverage_spec.cr
python3 -B -m unittest spec/drama3_decode_replay_spec.py -v
```

Seven Crystal guard examples and thirteen Python analyzer examples pass. The
release capture and analyzer pass on the final runtime source. Raw receipts,
manifests, complete commands, binaries and reports are under
`/opt/mldata_ssd/cogni-ml/drama3-decode-replay-20261018/`; hashes and scoped claims
are indexed by `../.landmark/evidence/drama3-decode-replay.json`.

The next target must use a new discriminator. Down deserves a varying-input,
all-40-layer paired test of a fused down-projection and expert weighted sum,
preserving the original projection subtotals and expert accumulation order.
This can test the intervening output materialization and dispatch cost, unlike
another cached single-layer repetition. Its previously tested packed-row
candidate remains off until complete-WAV benefit is reproducible.
Qkv is the largest dense role: a new shape-specific
loading or reuse hypothesis must preserve every original F32 lane subtotal and
tree. Blanket SIMD selectors, qkv/z/b/a fusion and tower FFN fusion already
lacked repeatable complete-request benefit and must not be promoted from these
local timings. Unmeasured recurrent/sampler work may still change priorities.

Any candidate needs unchanged upstream metrics and profiler-off complete-WAV
pairs with identical codes, sampler tape and read/wait ledgers. No default
promotion follows from this diagnostic result. Rollback is omitting the
standalone probe. Refresh after source, compiler, runtime, checkpoint, input,
queue, fixture, device, OS or host-load changes, or artifact loss.
