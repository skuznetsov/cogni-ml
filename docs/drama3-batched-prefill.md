# DRAMA-3 native batched prefill

## Scope and decision

From `a15d425366f8cff4ad1b287a347da766aa8b46f2`, the resident backbone has a
genuine layer-major sequence producer. Dense projections, routing and affine
expert projections consume multiple token rows in one dispatch family. This
extends the existing within-token top-8 expert batching; the baseline already
has `DRAMA3_MOE_BATCH=1`.

The generator exposes `DRAMA3_PREFILL_CHUNK=1..32`, default **1**. A value of 16
selects the measured experimental path for host and resident/reference prompts.
The strict upstream intermediate gate remains open on both native paths, so
there is no default promotion. Roll back by selecting 1 or reverting this slice.
There is no arithmetic, dtype, quantization or tolerance relaxation.

## Execution and ownership

`BackboneResident#prefill` validates owner, row count, width and remaining cache
capacity before encoding. Each layer computes projections for the chunk, then
updates full-attention KV/causal prefixes or convolution/delta state sequentially
at `base + row`. It batches output projections and MoE across token/rank pairs.
Dense reduction, expert group reduction, routing ties, rank summation and stable
shared-gate sigmoid retain the scalar order. Dense fixtures and the explicitly
disabled expert-batch option retain the original per-row MoE fallback.

The generator forms the embedding/VQ input table on GPU, computes one backbone
chunk, then teacher-forces exactly N-1 prompt pairs in temporal order. Backbone
and tower states are disjoint; teacher forcing consumes the corresponding
hidden row and does not produce the next backbone input. Submissions remain at
the existing per-prompt-row boundaries. Normal prefill adds no read, finish,
completion wait or submit. Existing EOS/control and terminal reads remain.

With generation scratch enabled, `backbone.prefill` owns the chunk table and
temporaries separately from decode and tower workspaces. The final borrowed
hidden row is consumed before reuse; workspaces retire after the existing
terminal boundary. Device diagnostic callbacks disable workspace reuse. The
older explicitly enabled CPU trace callback still reads diagnostics and is
excluded from the normal no-readback claim.

## Numerical evidence

The pinned checkpoint has **40** backbone layers, hidden width 2048. The real
prompt has 158 rows; chunk 16 includes a 14-row tail. GPU comparisons cover 81
layer/final boundaries and 12 MoE39 stages, route-ID canonicalization, final
head output and a following ordinary decode step. At graph depths 0 and 2,
all **932 native scalar/batch checks are bitwise exact**, with no intermediate
activation readback. The two saved MLX terminal hidden/logit checks pass the
unchanged `3e-5 + 3e-4*abs(reference)` tolerance.

Fresh pinned MLX captures run both scalar and full-sequence arms from identical
GPU embedding inputs and fresh temporal caches. Raw upstream route order is
preserved; diagnostic canonical views restore sorted SwitchGLU routes and sort
by expert ID. In both captured arms the independently reconstructed route-ID
table equals the canonical IDs byte-for-byte. Its inverse-order transform also
has an offline negative test. The native comparator consumes canonical IDs,
weights and expert gate/up/activation/down, but does not require the duplicate
reconstructed-ID field as a universal runtime guard.

Every native scalar and batch comparison against a given MLX tensor has the
same metrics. Both graph depths also have identical metrics. Router logits,
expert IDs/weights and expert gate/up/activation pass. The strict intermediate
tolerance remains `6e-5 + 3e-4*abs(reference)`, and **fails** as follows, per
native arm:

| Upstream execution | Failed elements, summed over checked stages | Failing stages |
| --- | ---: | ---: |
| Scalar | 100 | 8 |
| Sequence | 222 | 11 |

These are counts over several tensors, not unique semantic errors. Failures
include late residuals (scalar layers 37-39, sequence layers 35-39), MoE39
routed/shared/output sums and expert down. Canonical expert down contributes
37/102 failed elements, with maximum error/allowance 5.9914/6.9374. The final
normalized hidden table passes the stricter terminal tolerance in both MLX
arms. Passing it does not repair failing intermediate values. Existing and
newly exposed fidelity gaps remain open; the evidence does not establish that
every failed coordinate descends from the historical expert-108 discrepancy.

The GPU comparator is attacked with finite error and NaN poison. It must report
two bit differences, one tolerance failure and one nonfinite value. Only compact
metrics are read at the diagnostic terminal boundary. Fixture export and MLX
raw reference serialization are diagnostic terminal operations, outside native
production inference. The final parity CLI preserves JSON then exits nonzero
on any failed saved-terminal or upstream gate; capture completion is not green.

## Measurement method

All heavyweight runs/builds are serial under `scripts/run_safe.sh`, with a 15%
free-memory floor, quiet-host check disabled and unrelated processes untouched.
Device: Apple M2 Max. Temporary, build, reference and raw outputs live at
`/opt/mldata_ssd/cogni-ml/drama3-batched-prefill-20261016/`; model weights stay on
the local SSD. Memory availability and active-host scheduling limit inference
beyond this fixed workload.

The isolated benchmark retains one Context and converted weight caches across
both arms/components. Fresh temporal state is prepared and drained outside the
clock. Each component excludes one scalar and one batch prime, then measures
ABBA and BAAB (four observations per arm). MoE means **layer 39 on the same 158
normalized input rows**, not an average over layers. Backbone means all layers
from the same embedding table, excluding embedding and tower.

Wall time ends after the required diagnostic terminal drain. Passive existing
completion timestamps supply completed GPU command-interval sums; these are
not isolated shader time. Depth 0 exposes durations only; depth 2 also exposes
absolute intervals/span. Scoped command counts, status, timing validity, zero
readbacks and fully completed queue ledgers are qualified. Logging overhead
can scale with command count (158 scalar versus 10 chunked commands); GPU
interval evidence must accompany wall time. Unmeasured preparation events are
excluded by markers. A prior fresh-Context pilot was aborted and is rejected
as evidence of warm performance.

The complete-request worker retains static weights, creates fresh generation
states, and changes only chunk size. One prime per arm precedes ABBA. The same
seed-1 sampler tape, fixed text, sampling parameters and 60-frame bound are
required. All codes, PCM16 WAV bytes, samples, tape, prompt/reference identity,
and read boundaries must match the scalar prime. Queue counts must be equal
except for one source-derived empty final-row submit/wait elision when the
last chunk has more than one row. For 158 rows and chunk 16 this is exactly
one fewer submission and completion wait, not a new synchronization. Terminal GC is
charged to its own request before result publication, preventing carryover
into the next request clock. Request wall includes its generation, codec,
WAV/report writing, terminal collection and passive telemetry.
The 60-frame bound yields 122880 samples (2.786395 seconds) and truncates this
fixed prompt. These are complete bounded requests, not complete utterances or
a steady-state seconds-of-generation-per-second-of-audio estimate.

## Observed isolated timings

Warm medians in seconds, four scored observations per arm, chunk 16 versus 1:

| Graph depth | Component | Scalar wall | Batch wall | Wall ratio | Scalar GPU interval sum | Batch GPU interval sum | GPU sum ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | MoE39 | 0.291786 | 0.133111 | 2.192 | 0.240668 | 0.127430 | 1.889 |
| 0 | Backbone | 20.296115 | 8.747524 | 2.320 | 18.668015 | 8.030887 | 2.325 |
| 2 | MoE39 | 0.156851 | 0.115619 | 1.357 | 0.271385 | 0.226026 | 1.201 |
| 2 | Backbone | 18.912527 | 8.031174 | 2.355 | 18.789585 | 7.984778 | 2.353 |

At depth 2, median completed GPU spans are 0.156527 / 0.115105 seconds for
MoE39 and 18.904967 / 7.984783 seconds for backbone (scalar / batch). Command
intervals can overlap, so their sum can exceed wall/span; do not interpret it
as isolated shader service time. Each isolated scalar scope has 158 commands,
each batch scope 10, with no readbacks and all completions accounted for.
Depth-0 backbone wall ranges are 20.270915-20.490129 / 8.714104-8.762635;
depth-2 ranges are 18.659579-19.341649 / 8.027691-8.233569. MoE gain is
queue-depth sensitive and cannot be extrapolated from the earlier Nomic result.

## Observed complete bounded requests

At depth 0, the scored ABBA block (two observations per arm) has scalar/batch
request wall medians **47.263830 / 38.826715 seconds**, a **1.217x** ratio.
Ranges are 44.648347-49.879313 / 36.569874-41.083556 seconds. Completed GPU
interval-sum medians are **42.976829 / 31.559585 seconds**, a **1.362x** ratio.
All six requests, including the two excluded primes, preserve all 60x11 codes,
WAV bytes, sample count, sampler tape and existing control/terminal readbacks.
Queues drain completely: scalar 310, batch 309 submits/completion waits.
The one-command difference is exactly the multirow-tail elision above.

At depth 2, scalar/batch request wall medians are **45.247341 / 33.115445
seconds**, a **1.366x** ratio; ranges are 42.696080-47.798603 /
32.142516-34.088374 seconds. Completed GPU interval-sum medians are
**42.295568 / 31.566566 seconds** (1.340x); completed GPU span medians are
42.547000 / 31.759563 seconds. All six requests pass the same exact guards.
Depth 2 has 251 / 250 submits/completion waits, both with pending zero. Queue
depth already changes the baseline command boundaries; compare counts within
each depth. All twelve requests across depths have the same WAV digest
`bc4c5b258c08f25828c727c69c1033691da651dc936ab30583ade88a0994ff31`.

The request result is smaller than the backbone gain because tower, decode,
DAC and host/observer work remain in the request. This matrix does not isolate
their individual speed limits. The strict upstream gate, active-host schedule,
small sample count and logging overhead prevent default promotion or a general
latency claim.

## Checks and reproduction

`metal-suite-final` executes 86 relevant Metal examples, including the batch
primitives and backbone/generator suites. They cover F32/BF16 dense reductions,
affine packed projections, route ties and sums, nonzero cache base, N=2/5,
convolution ring wrap, a following decode step, host/resident prompts, mixed
semantic rows, scratch reuse and borrowed final-row lifetime at graph 0/2.
Invalid shapes, capacity and foreign owners must fail before advancing state.
The genuine missing `Context#matvec_rows` API witness precedes implementation.

`examples/drama3_prefill_benchmark.cr` provides `export`, `parity` and `bench`
modes. Parity uses `examples/support/drama3_prefill_compare.metal` for device
comparisons. `scripts/drama3/prefill_intermediates.py` captures the pinned
scalar and sequence upstream arms without substituting a native algorithm.
`scripts/drama3/analyze_prefill_benchmark.py` qualifies raw scoped completion
logs. `scripts/drama3/measure_prefill_requests.py` drives the persistent
`examples/drama3_prefill_worker.cr` and checks fixed request outputs and queues.

Exact guarded builds, CLI arguments, environment controls and input hashes are
retained in the external `commands.json`. The final `parity-final-depth0` and
`parity-final-depth2` receipts deliberately expect exit **1** because upstream
fails; each still writes its complete 1307-check JSON. The earlier parity CLI
returned zero after capture despite a red upstream field, and those receipts
are not successful compatibility gates. The fixed CLI fails closed.

`offline-final` passes 29 tests across `drama3_prefill_intermediates_spec`,
`drama3_prefill_benchmark_spec` and `drama3_prefill_requests_spec`. Negative
cases reject malformed/tampered references, bad inverse-route restoration,
nonpersistent or malformed schedules, invalid/incomplete GPU timings, changed
request outputs and unexpected queue differences. `verify-final` replays raw
log qualification, checks all 219 reference hashes and both actual reconstructed
route tables, identical native/upstream metrics across arms/depths, final
production-source/build hashes, twelve code/WAV/tape/read ledgers and required
command/log receipts. It returns `PASS_WITH_UPSTREAM_GATE_OPEN`, not a green
upstream compatibility result.

## Lineage, limits and refresh

Checkpoint `cguldogan/drama-3-mlx-q4` at
`5331ab509359c0e4a7e353a19e255ddf3e8152a8`; MLX audio at
`e7488935ec2ebfafd38ef2b58e290d79f3d05dc7`. Command receipts retain source,
binary, input, log and manifest hashes. Config/index/tokenizer metadata are
pinned; full weight-shard hashes and reproducible-build attestation are not
claimed. The bridge object is reused and hashed, not freshly rebuilt.

Mini-Quad review is correlated same-lineage review, with root-executed GPU
evidence. Execution mechanics and the bounded scalar/batch no-regression claim
are distinct from strict upstream compatibility. Upstream default-promotion
remains VULNERABLE to the observed failing intermediate gate. No general voice,
prosody, reference cloning, whisper, long-speech or DAC fidelity claim follows.

Refresh numerical and performance evidence after source/model/input/tape,
fixture/tolerance, queue/workspace, observer, device/OS/storage changes, a
contradiction or raw-artifact eviction. Raw evidence and exact commands are
indexed in `.landmark/evidence/drama3-batched-prefill.json`.

Next numerical falsifier: identify the earliest failed row/channel in the late
residuals and separate expert-down reduction from shared/residual accumulation
using identical inputs. Do not assume every failed coordinate is the old
expert-108 case. A later performance candidate is direct token-row addressing
inside expert kernels, avoiding the current GPU input expansion. It remains
unmeasured.
