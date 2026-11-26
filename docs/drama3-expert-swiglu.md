# DRAMA-3 exact expert gate/up/SwiGLU fusion

The opt-in Metal path combines gate projection, up projection, stable SiLU and
the final multiply for routed and shared experts. Each eligible call replaces
four dispatches and four result allocations with one. Weight precision,
intermediate F32 arithmetic and compiler math options are unchanged. The frozen
selector is `DRAMA3_EXPERT_SWIGLU_FUSION=0|1`, default 0; rollback is 0 followed by
construction of a new resident Context.

## Admitted implementation

Routed pairs use affine Q4 or Q8, group size 64, equal logical matrix geometry,
1..32 groups, 1..32 selected ranks and the existing 64-thread native expert
reduction. Gate and up may have different bit widths or affine encodings. Their
packed column counts therefore need not match. Each half of a 64-thread group
computes one projection's original sequential group subtotals. The leader folds
those groups in ascending order, then performs the same stable sigmoid and two
F32 multiplications as the old `silu` and `mul` kernels. Invalid IDs reach the
barrier and produce the original nonfinite output class.

Shared pairs use aligned rank-two BF16 weights, equal logical rows/columns and
the existing dense tree reduction. The two independent 64-thread accumulation
and reduction trees retain their original order. This removes intermediate
materialization while sharing input loads. Other valid geometry, encodings or
reductions retain the established gate/up/SiLU/multiply path. Owners and pair
shapes are checked before work is encoded; pipeline thread limits are guarded.

The Backbone step path calls these APIs for batched routed experts and shared
experts. Sequential prompt prefill uses the same step path; the separate
batched-prefill implementation is unchanged. A Context freezes the selector at
construction and counts actual routed and dense candidate dispatches. The
ordinary Studio worker reports per-request counter deltas.

Production activations, routing IDs, weights and results stay on GPU. No
production read, submit or wait is added. Existing CogniGraph ownership retains
bound buffers until completion. Diagnostic MoE tracing materializes gate/up
from the candidate kernel itself, so a callback does not silently qualify the
fallback. This is ordinary kernel fusion; no LTP/WBA certificate is claimed.

## Executed correctness checks

Qualification uses Apple M2 Max and checkpoint
`cguldogan/drama-3-mlx-q4@5331ab509359c0e4a7e353a19e255ddf3e8152a8`.
The pinned request is `Hello, this is a voice test.`, seed 1, max 64 frames,
temperature 0.8, top-k 30, top-p 0.9, without a voice reference. It produces
60 frames and 2.786395 seconds of mono audio at 44.1 kHz.

- The fail-first compile rejected the absent API at base
  `a8156b8c7a048faebd4f0eb10d4a323839169498`. The first GPU run then exposed an
  erroneous packed-shape guard for mixed Q4/Q8 pairs; replacing it with logical
  rank/expert-count checks made the focused and adjacent suite pass.
- The release Metal suite passes 19 examples: new expert/dense SwiGLU cases and
  existing down/sum, dense and tower pair fusions. The new cases exercise Q4/Q8
  mixed pairs, F32/F16/BF16 affines, mapped offsets and sliced buffers,
  cancellation, signed zero, duplicate/shuffled/invalid IDs, frozen/default-off
  selection, unsupported geometry/reduction fallbacks and owner/shape guards.
  GPU comparisons reject seeded bit/nonfinite defects. Eight offline checks
  cover the new and preceding paired-request drivers.
  Special-value fixtures compare NaN classes rather than payload/sign bits;
  they do not certify bit identity between different NaN encodings.
- Actual-input capture covers all 40 routed and 40 shared sites at decode rows
  0, 29 and 59: 240 candidate calls. Gate, up and activation are bit-exact against
  explicit old kernels. Each captured call must increment its candidate counter.
  Inputs vary across both frame transitions and remain finite. Selected IDs are
  also checked for finite variation, without requiring every layer to reroute.
  This benchmark admits homogeneous routed pairs; the pinned model uses Q4 for
  both gate and up at every site. Mixed Q4/Q8 is covered by synthetic GPU specs,
  not by this model capture or its grouped timing signature.
- Local paired replay verifies 11,059,200 output values per arm with zero bit or
  nonfinite mismatches, including warmups. Each family rejects the seeded
  `[2 bit mismatches, 1 nonfinite pair]` control. Comparison commands and compact
  receipt reads are diagnostic and excluded from timed projection commands.
- The capture's ordinary generation and decode preserve all codes, sampler
  tape, EOS state and WAV bytes. They retain 122 submissions/completions/waits,
  496 control bytes and 494,160 terminal bytes, with zero activation readback.
- The backbone gate compares 243 boundaries per arm, 729 comparisons total.
  Actual dispatch counters prove 120 routed and 120 dense candidate calls versus
  120+120 explicit old reference calls. Native boundaries are bit-exact and both
  arms have identical MLX metrics at the previous tolerances. The existing
  strict failure remains at row 1 `backbone.layers.39.hidden`; this is a
  no-regression check, not strict upstream parity.

The saved upstream comes from community port `cguldogan/mlx-audio` revision
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`, MLX 0.31.1. This run hashes and
reuses that reference rather than regenerating an MLX oracle. Config and
safetensors index are hashed; checkpoint shard payloads are not newly
authenticated. The original Fish text-generation pipeline is outside this gate.
Tolerances stay `6e-5 + 3e-4*abs(reference)`, with final hidden
`3e-5 + 3e-4*abs(reference)`.

## Performance qualification

Local timing uses homogeneous captured operands, two excluded warmups and eight
measured counterbalanced pairs per family, with 120 records repeated twice per
arm. Existing command timestamps are read after completion; queue depth is two.
These GPU command intervals may overlap and are neither isolated shader latency
nor additive request costs.

The median paired unfused/fused GPU command interval is 1.2165x for routed
experts and 1.6828x for shared experts, with eight of eight measured intervals
lower in each family. Median command windows are 71.167 -> 58.457 ms and
12.201 -> 7.227 ms respectively. These windows each replay 120 records twice;
they are not per-layer latencies. Median host encoding intervals are
16.135 -> 1.773 ms and 7.100 -> 0.789 ms, also local replay measurements.

Full-WAV qualification completed 24 requests in opposite worker orders,
`0,1,1,0` and `1,0,0,1`, with one excluded cold request and two warm requests per
fresh worker. Each worker's warm repeats are reduced before comparing modes.
All 24 preserve codes, sampler inputs, stop state, full WAV and ordinary
read/wait ledgers. Each on request executes 3,560 routed plus 3,560 dense fused
dispatches; each off request executes zero. Profiling, scratch and other
experimental fusions are disabled; prefill chunk is one, expert reduction
native64, DAC coalesced and graph depth two.

| Warm worker medians | Generation off -> on | DAC off -> on | Full request off -> on |
| --- | ---: | ---: | ---: |
| Forward `0,1,1,0` | 20.706 -> 20.708 s | 3.301 -> 3.409 s | 24.043 -> 24.213 s |
| Reverse `1,0,0,1` | 20.560 -> 18.949 s | 2.933 -> 3.167 s | 23.528 -> 22.200 s |
| Both blocks | 20.560 -> 19.302 s | 2.933 -> 3.167 s | 23.528 -> 22.266 s |

The four mirrored worker generation off/on ratios are 1.097, 0.913, 1.053 and
1.117; full-request ratios are 1.151, 0.859, 1.024 and 1.096. One contrast
regresses, the forward generation median is unchanged, and the unmodified DAC
phase moves materially. The host is shared and workers/repeats are correlated.
The combined medians are descriptive, not a causal or repeatable full-request
speedup. Keep default 0. Do not select runs after inspecting host load or
attribute unchanged-phase timing drift to fusion.

Warm-request allocations fall from 135,590 to 114,230, a reduction of 21,360.
Cumulative requested bytes fall from 4,079,562,724 to 3,882,708,964. These are
allocation counters, not peak memory or measured DRAM traffic. Allocation
timing is unavailable with profiling disabled; a reported zero is not zero cost.
For the 2.786395-second clip, combined request RTF is 8.444 off and 7.991 on;
this does not establish real-time generation.

Root executed the analyzer and 18 in-memory malformed-receipt checks. Missing
or duplicate captures/requests, silent fallback, changed source hashes, tape,
WAV, ledgers, controls and upstream failures are rejected for the intended
reason. Hashes of all 35 original inputs checked by this attack remain unchanged.
Correlated mini-Quadrumvirate source review found no production counterexample
and returned ROBUST for default-off implementation and bounded no-regression
claims. It does not certify default promotion or independent replication.

## Reproduction and refresh

The diagnostic executable is `examples/drama3_expert_swiglu_benchmark.cr` with
`MODEL_DIR BASELINE_JSON OUTPUT_DIR`. Use a qualified baseline report that records
`resident_expert_down_sum_fusion=false`; older reports without that field are
rejected before loading weights. The upstream probe is
`examples/drama3_dense_decode_compare.cr` with optional `--expert-swiglu`.
Existing dense and `--expert-down-sum` modes remain available.

The full worker driver is `scripts/drama3/measure_host_pairs.py` with
`--expert-swiglu-schedule` and `--warm-repeats 2`. All heavy builds/runs are serial
under `scripts/run_safe.sh`, with the authorized 15% free-memory floor and
external TMPDIR/cache/artifacts under
`/opt/mldata_ssd/cogni-ml/drama3-expert-swiglu-20261018/`. Weights stay local.
The live Studio is not restarted during qualification.

Evidence index: `.landmark/evidence/drama3-expert-swiglu.json`. External receipts:
`capture/capture.json`, `decode-comparison.json`, `analysis.json`,
`analysis-negative-mutations.json`, and the two `expert-swiglu-worker.json`
manifests under `wav-pairs/` and `wav-pairs-reverse/` in the artifact directory.
The index records source/build hashes, commands and limits; it does not attest
build provenance or turn a receipt into an independent execution certificate.
The capture manifest pins source hashes but does not directly record the
benchmark executable digest; the index hashes the retained binary. Full-request
worker manifests directly pin their common executable digest.

Next discriminator: the source-backed GDN postprojection corridor described in
`docs/drama3-optimization-roadmap.md`. Its RMS/SiLU/multiply candidate still needs
a falsifier and actual-operand measurement before any speed forecast.

Refresh correctness after kernel, reduction, math options, Context ownership,
callback, model or fixture changes. Refresh timing after compiler, driver,
device, queue depth, workload or host-load changes. Missing or changed evidence
invalidates the associated observations. Review combines root execution and
correlated same-lineage mini-Quadrumvirate review, not independent replication.
