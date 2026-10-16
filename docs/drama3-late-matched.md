# DRAMA-3 late-layer matched-input comparison

Source base `d05bc45d`, Apple M2 Max, 2026-10-16. The first cumulative
tolerance violations are sequence-MLX layer 35 hidden and scalar-MLX layer 37
hidden. On exact upstream operands, the tested shared expert operations stay
within tolerance, and residual addition is bitwise exact. A separate local
expert-down discrepancy remains at layer 39. These results do not establish
one cause for both observations. Production source and defaults are unchanged.

## Experiment

The pinned community MLX port is `cguldogan/mlx-audio` at
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, MLX 0.31.1. The checkpoint is
`cguldogan/drama-3-mlx-q4` at `5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
All five local checkpoint shards, config and index were hashed before the
capture; their device/inode/size/mtime identities remained unchanged through
the final verification. This assumes immutable files during the operation,
rather than continuously authenticating every weight read.

`scripts/drama3/late_mlx_capture.py` retains original operator return values
and captures both official scalar and sequence executions of the same
158-row F32 embedding input. It captures all 81 decoder residual/hidden
boundaries, and detailed operands at layers 34 through 39. Sequence routing
records the official sort and inverse. The analyzer checks the stable route
order and canonical expert arrays against the original captured arrays;
swapping equal expert IDs alone is insufficient to pass this guard.

The new capture has 313 scalar and 319 sequence tensors. All 219 arrays that
overlap the previous full-prompt capture are bitwise unchanged, including
dtype, shape and payload hash. Newly captured shared internals have no
earlier no-hook witness; the fidelity statement is limited to that overlap.

The native probe executes one chunk-16 backbone trajectory and compares it
with both MLX arms. The arms denote reference execution, not two native
implementations. It then performs 156 matched-input replays: 13 per selected
layer and reference arm. Quantized down replay uses the exact canonical
per-rank activation and GPU expert IDs. Shared projections, activation, down,
sigmoid weighting, MoE sum and residual have separate operand controls.
Whole-MoE replay starts from exact upstream post-attention-normalized input;
it is distinct from replaying the final addition alone.

Intermediate tolerance remains `6e-5 + 3e-4 * abs(reference)`; only final
backbone hidden uses the existing `3e-5` absolute tolerance. A finite error
must be strictly greater than its allowance to fail. Bit mismatch, tolerance
failure and nonfinite are separate predicates. No tolerance was widened.

## First cumulative boundaries

The first bit difference is already layer 0 attention residual, row 0,
channel 0: scalar error `2.2351742e-8`, sequence error `5.736947e-7`.
Both are well within tolerance. The first strict failures are later:

| Reference arm | First failing boundary | Row / channel | Native | MLX | Error | Allowance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Sequence | Layer 35 hidden | 154 / 1277 | -0.004533291 | -0.0046081543 | 7.4863434e-5 | 6.1382445e-5 |
| Scalar | Layer 37 hidden | 154 / 1557 | 0.01712513 | 0.017191052 | 6.592274e-5 | 6.5157314e-5 |

Across the 81 full boundaries, scalar MLX has 31 failing elements in four
groups; sequence has 61 in seven. These are the residual/hidden subset of the
earlier, broader intermediate gate, not new totals for that gate. Every
boundary count and F32 maximum reproduces the corresponding saved strict
full-prompt gate at both graph depths. Final hidden remains within its
endpoint tolerance; this does not close intermediate fidelity.

## Matched-input replays

All values are finite. The table aggregates six selected layers per arm;
failure counts belong only to these exact replay inputs.

| Replay | Scalar failures | Sequence failures | Observation |
| --- | ---: | ---: | --- |
| Post-attention RMS | 0 | 0 | Maximum error/allowance below 0.0041 |
| Native expert down | 7 | 1 | All failures at layer 39 |
| Existing `mlx_q4` expert down | 0 | 7 | Scalar bitwise exact; sequence layer 39 still fails |
| Shared gate/up projections | 0 | 0 | Maximum ratio below 0.0642 |
| Shared SwiGLU from exact gate/up | 0 | 0 | Maximum ratio below 0.00113 |
| Shared down from exact activation | 0 | 0 | Maximum ratio 0.183878 / 0.275990 |
| Shared sigmoid weighting from exact operands | 0 | 0 | Maximum ratio below 0.00111 |
| Exact routed sum + exact weighted shared | 0 | 0 | Bitwise exact |
| Native fused shared weighting/add | 0 | 0 | Maximum ratio below 0.03261 |
| Whole native MoE from exact normalized input | 4 | 2 | All failures at layer 39 |
| Exact residual + exact upstream MoE | 0 | 0 | Bitwise exact |
| Exact residual + whole native MoE replay | 1 | 5 | All failures at layer 39 |

The scalar `mlx_q4` down path matches all 15,532,032 compared values bitwise
over layers 34 through 39. This does not generalize to sequence MLX. Its
sequence layer-39 first failure is row 1, rank 0, expert 108, channel 469:
native `1.9601784`, MLX `1.9610152`, error `0.0008368492` against
allowance `0.0006483046`. The two reference arms use different activations;
each native replay consumes its own reference arm's exact activation.
Differences in upstream scalar/sequence quantized reduction paths remain a
candidate explanation, not a demonstrated backend instruction trace.

Default native down has these first local layer-39 failures:

| Reference | Row / rank / expert / channel | Error | Allowance |
| --- | --- | ---: | ---: |
| Scalar | 1 / 0 / 108 / 469 | 0.0007828474 | 0.0006480693 |
| Sequence | 3 / 3 / 108 / 40 | 0.00010498613 | 9.594759e-5 |

Matching the final addition operands removes every bit difference in the
residual and two-term MoE sum controls. Matching all tested local operands
at layers 35 and 37 removes their tolerance failures. This separates
accumulated input drift from an independently reproducible layer-39 local
MoE/down discrepancy. It does not attribute the earlier crossing to layer 39,
prove that down is the sole source of cumulative drift, or identify every
failed coordinate as expert 108.

## GPU and graph contract

Route restoration, replay arithmetic, comparison, maxima and first-coordinate
selection execute on GPU. The comparator preserves 64-bit flat indices and
GPU-selected expert IDs. It merges chunk receipts on GPU before one terminal
compact read: 318 groups, 1776 partial receipts, 66,144 terminal bytes plus
the existing 4-byte finite-status control. There are no pre-terminal
activation reads or diagnostic activation dumps. CPU preparation validates
files and metadata; MLX raw export is an offline terminal fixture operation.

Both graph depths submit and complete 11 diagnostic commands with 11 existing
completion waits and zero pending work at exit. No per-stage finish/read was
added. Depth 2 reaches two pending commands and 22,928,410,032 retained bytes;
depth 0 reports zero pending/retained high-water bytes. Retained accounting is
not measured physical residency. All 318 metrics objects, including every
first witness and its bits, are exactly equal across graph depths 0 and 2.

The full native CLI writes the complete report and exits 1 for the known red
strict boundary gate. Exit 1 is retained as a numerical failure, not relabeled
as successful compatibility. Analysis exits 0 when a complete, internally
consistent failing report is correctly validated; that is a different claim.

The executed GPU oracle covers exact allowance versus next-F32 failure,
separate first-bit/first-tolerance positions, NaN/Inf, tile boundary and tail,
indices beyond 2^32, expert-ID witnesses and an out-of-order GPU receipt merge.
One Metal example and sixteen offline capture/analyzer tests pass. Final
cross-artifact verification checks both executed source/binary hashes, all
162 prior boundary metrics per depth, graph equality and checkpoint identity.

## Reproduce and continue

From this worktree, build with `-Ddrama3_metal --release` and the project's
Metal bridge/framework flags. Invoke:

```text
drama3-late-compare MODEL_DIR MANIFEST OUTPUT.json --chunk=16 --depth=0
drama3-late-compare MODEL_DIR MANIFEST OUTPUT.json --chunk=16 --depth=2
python -B scripts/drama3/analyze_late_compare.py MANIFEST --native OUTPUT.json
python -B -m unittest discover -s spec -p 'drama3_late_*_spec.py'
crystal spec -Ddrama3_metal spec/drama3_late_compare_spec.cr [bridge/framework link flags]
```

Exact commands, environment allowlist, source/binary/input hashes and logs are
in `/opt/mldata_ssd/cogni-ml/drama3-late-matched-20261016/commands.json`.
The repository index is `.landmark/evidence/drama3-late-matched.json`.
Build/capture setup failures are retained separately from numerical results.

Correlated mini-Quad review is ROBUST for the bounded first-boundary and
matched-replay observations. A single-cause or complete upstream-fidelity
claim remains VULNERABLE. The next numerical discriminator is the upstream
sequence quantized down path on these same operands, followed by exact-input
decomposition at the earlier cumulative failing coordinates. Keep defaults
at chunk 1, native down, dense tree and MAP_SHARED off.

The user's speed target is sustained RTF 0.8–0.9 with separately measured
first-audio latency. This diagnostic establishes no speed gain. Precision-
preserving decode kernels and DAC require their own paired measurements and
full fixed-trajectory guards. A local kernel win cannot establish real time.
Refresh these findings after source, checkpoint, input, reference, tolerance,
queue, device, OS or artifact changes. Rollback is omitting the diagnostics
or reverting this isolated slice.
