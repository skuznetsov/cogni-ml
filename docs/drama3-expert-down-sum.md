# DRAMA-3 ordered expert down and weighted-sum fusion

The opt-in Metal path combines each routed expert's down projection and the
weighted rank sum in one dispatch. It preserves the existing native arithmetic:
sequential F32 affine-group subtotals, ascending group fold, then ascending
weighted rank fold. It removes the production per-rank down output and one
dispatch per eligible MoE call. Model precision and compiler math options are
unchanged. The default remains `DRAMA3_EXPERT_DOWN_SUM_FUSION=0`.

## Admission and ownership

The selector is strictly validated and frozen when the resident Context is
constructed. Eligible matrices use Q4/Q8, group size 64, native 64-lane expert
reduction, 1..32 groups dividing 64, 1..8 selected ranks and at most 256 rank/group
lanes. The compiled pipeline's threadgroup limit is also checked. Other valid
geometries, explicit MLX reduction and requested 32-thread launches use the
ordered two-dispatch fallback. Foreign buffers, bad shapes, unsupported affine
encodings and invalid reduction requests are rejected before encoding work.

One threadgroup owns each output row. A lane computes one rank/group subtotal;
the leader folds groups and ranks in the original order. Invalid expert IDs
still reach the shared-memory barrier and reproduce the original NaN result.
Weights stay mapped, activations and routing stay on GPU, and CogniGraph retains
the bound buffers through completion. No production read, submit or wait is
added. This is ordinary kernel fusion, with no LTP/WBA claim.

MoE tracing uses the candidate kernel and optionally materializes its actual
per-rank outputs. Tracing never silently substitutes the fallback. The change
applies to `BackboneResident#step`, including sequential prompt prefill using
that method. The separate batched-prefill implementation is unchanged.

To select the path for a newly started worker, set
`DRAMA3_EXPERT_DOWN_SUM_FUSION=1`. Restore 0 and create a new Context to roll back.
The running Studio was not restarted or replaced during qualification.

## Correctness evidence

Qualification used Apple M2 Max and checkpoint
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
The native reference request is `Hello, this is a voice test.`, seed 1, max 64,
temperature 0.8, top-k 30, top-p 0.9. All arithmetic comparisons run on GPU;
only compact diagnostic receipts and the existing control/audio outputs return
to the host.

- The fail-first build rejects the absent API on base `27c1d00c`.
- Twelve GPU examples pass, including Q4/Q8, mixed F32/F16/BF16 affines, mapped
  offsets and sliced buffers, duplicate/shuffled/invalid IDs, rounded and maximum
  threadgroups, cancellation and signed zero, fallback geometries, ownership and
  pre-dispatch rejection. Bit/NaN negative controls must fail the same comparator.
  Four offline measurement-driver checks also pass.
- The varying-input probe captures all 40 layers at decode rows 0/29/59: 120 actual
  fused calls, with per-rank down and final weighted outputs. All are bit-exact
  against the explicit old expert and weighted-sum kernels. All 240 variation
  receipts cover both frame transitions, every layer and all three operand kinds
  (activation, IDs, weights); each changes and remains finite.
- Ten local pairs, including two warmups, check 4,915,200 values per arm with
  zero bit or nonfinite mismatches and reject the seeded `[2,1]` control. Capture
  preserves the normal generation ledger: 122 commands/completions/waits,
  496 control bytes and 494,160 terminal bytes. The subsequent comparisons have
  their own explicitly diagnostic commands and compact receipts.
- The backbone boundary gate compares 243 captures per arm, 729 comparisons total,
  with actual dispatch counters proving 120 fused calls versus 120 old reference
  calls. Native boundaries are bit-exact; both arms produce identical metrics
  against the saved MLX reference. The existing strict failure remains at
  row 1 `backbone.layers.39.hidden`; this is a no-regression result, not strict
  upstream parity. Tolerances remain `6e-5 + 3e-4*abs(reference)`, with final hidden
  `3e-5 + 3e-4*abs(reference)`.

The saved reference comes from community port `cguldogan/mlx-audio` revision
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7` with MLX 0.31.1. This run hashes and
reuses that reference; it does not regenerate a fresh MLX oracle or certify the
original Fish text-generation pipeline. Config and safetensors index are hashed;
full checkpoint shard payloads are not newly authenticated.

## Performance scope

The local replay contains 120 captured calls repeated twice per timed arm.
Across eight measured counterbalanced pairs, median command intervals are
78.792ms unfused and 46.948ms fused; the median paired ratio is 1.679x, and all
eight favor fusion. Host encoding medians are 2.621ms and 1.038ms. Commands are
queued at depth 2 and their GPU intervals may overlap, so these are paired-load
command windows, not isolated shader latency or additive request costs.

Complete-WAV timing uses the same release worker binary with profiler and
scratch disabled, coalesced DAC, native 64-thread experts, graph depth 2 and other
experimental fusions disabled. Each fresh worker serves one excluded cold and
two warm requests. Warm repeats are summarized within worker before comparing
modes. Both balanced orders are retained, without selecting best runs. Weight
loading is recorded separately from complete request latency. The shared host
is not quiet; no host-load exclusion rule is applied after seeing results.

Both blocks pass all 12 requests (24 total, eight fresh workers). Each worker's
two warm requests are reduced to one median; the table shows the median of the
two worker summaries per mode, in seconds. Cold requests are correctness controls
only. The unchanged audio is 2.786395 seconds.

| Worker order | Complete request, off / on | Generation, off / on | DAC, off / on |
| --- | ---: | ---: | ---: |
| 0,1,1,0 | 25.075 / 22.633 | 20.875 / 19.586 | 4.136 / 2.973 |
| 1,0,0,1 | 23.979 / 23.371 | 20.792 / 19.849 | 3.141 / 3.426 |

Generation time decreases by 6.18% and 4.53% at these block summaries. Across
all four mirrored worker contrasts, off/on generation ratios are 1.0944,
1.0357, 0.9852 and 1.1097. The pooled four-worker-per-mode generation medians
are 20.875 / 19.849 seconds. These are descriptive observations on this request,
not a causal estimate or independent replication. In particular, the unchanged
DAC phase moves substantially, warm requests within workers vary, and one
mirrored contrast regresses. The larger complete-request reductions cannot be
attributed entirely to this kernel. Default promotion is therefore withheld;
the arithmetic-preserving candidate remains available through the opt-in flag.

Each eligible request uses 3,560 fused calls when enabled, versus zero when
disabled. Warm Metal allocations decrease from 135,590 to 132,030; cumulative
requested GPU bytes decrease by 233,308,160 bytes (222.5MiB). This is allocation
traffic, not a claim about peak physical residency. Full codes, sampler tape,
EOS/truncation, audio length, WAV bytes and normal read/wait ledgers match in all
24 requests. The common WAV SHA-256 is
`38f2530ed1c90f0caba45e222432909ac5d92cc3633e4c34a21fbe39d693e07d`.

## Reproduction and refresh

The small evidence index is
`.landmark/evidence/drama3-expert-down-sum.json`. It binds source and binary
hashes, build/check commands, external manifests, the analyzer and its output.
Raw logs, binaries, WAVs and diagnostic artifacts remain under
`/opt/mldata_ssd/cogni-ml/drama3-expert-down-sum-20261018/`; weights stay on the
local SSD. Heavy commands run serially through `scripts/run_safe.sh` with the
authorized 15% free-memory floor and external TMPDIR/Crystal cache.

The executable probes are `examples/drama3_expert_down_sum_benchmark.cr` and
`examples/drama3_dense_decode_compare.cr` with the optional fifth argument
`--expert-down-sum`. The normal four-argument dense comparison mode is retained.
`scripts/drama3/measure_host_pairs.py --expert-down-sum-schedule 0,1,1,0`
and reverse `1,0,0,1` provide the full worker matrix.

Refresh correctness after kernel/reduction/options, Context ownership/dispatch,
callback, model or fixture changes. Refresh timing after compiler, driver,
device, queue depth, workload or host-load changes. Missing or changed artifacts
invalidate the associated measurements; keep the default off until the relevant
qualification is repeated. Review is root execution plus correlated same-lineage
mini-Quadrumvirate review, not independent replication.
