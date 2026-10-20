# Wan S2V performance frontier

The active target is exact GPU inference at B=1, N=36,608, C=5,120,
H=40, D=128, text context M=512 and FFN width F=13,824. This analysis
separates executable operation counts, measured latency and unmeasured ideas.
The numerical gate stays `2e-5 + 2e-5*abs(reference)`, with finite operands.
No new model-path host waits, activation readbacks or encode-time Metal-buffer
allocations are admitted. Experimental defaults remain off until their stated
whole-block gate passes.

## What the existing evidence permits

[The attention comparison](wan-self-attention-benchmark.md) is committed at
`0f94f5b3`. Explicit QK/softmax/PV reduced standalone median latency by 31.55%
in three pairs. Its real BF16 whole-block warmup had 5,085 strict violations
out of 187,432,960 values, before any measured block pair. It is rejected for
promotion. A fast primitive is insufficient evidence for the complete block.
The separate [40-layer numerical drift](video-numerical-drift.md) also remains
open; a block-0 regression cannot close it.

The old 19/40 video process was terminated with user authorization and must not
be restarted automatically. Earlier pressure-query/RSS guards did not bound
physical footprint or swap. New probes require admission based on conservative
physical headroom, process-group footprint, swap growth and a monotonic deadline,
in addition to the unchanged 15% pressure threshold. These are sampled guards,
not a bound on allocations between samples. Do not reuse the old external
pause/resume controller containing the terminated PID.

[`wan_memory_guard.py`](../examples/support/wan_memory_guard.py) launches only
the supplied command in a new session. Default admission requires `memory_pressure -Q`
at least 15% and free-plus-speculative physical bytes at least 15% of RAM plus
the requested footprint budget. During execution it samples the process group's
summed `ri_phys_footprint`, physical headroom and system swap growth. Missing
telemetry stops the owned group; a monotonic timeout and wrapper SIGINT/SIGTERM
also clean it up. The unreaped session leader pins the group's identity until
cleanup. This covers the group, not descendants that deliberately escape with
`setsid` or `setpgid`, and cannot prevent a between-sample allocation spike.
Cleanup keeps the leader unreaped through bounded group-quiescence checks.
Darwin cannot query a zombie's PGID, so WNOWAIT pins its original identity;
a process-list outage does not prevent signaling that pinned group. A live
descendant after SIGKILL or unverifiable cleanup returns 77 with an incomplete
record, without reaping away the pin or claiming success. There is no unbounded
cleanup wait. The CPU cleanup suite passes 25/0 on Python 3.11 and 3.15;
live-telemetry timeout and injected-floor probes leave their owned groups absent.
Source/log bindings are in `wan-self-attention/guard-cleanup-cpu-receipt-v5.json`
under the external evidence directory used below.

The CPU-only guard check is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 spec/wan_memory_guard_spec.py
```

A bounded invocation for an already built RoPE probe is:

```sh
python3 examples/support/wan_memory_guard.py \
  --max-footprint-mib=5120 --max-swap-growth-mib=128 --timeout-sec=180 \
  --log=/opt/mldata_ssd/cogni-ml/rope-guard-UNIQUE.jsonl -- \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-block-benchmark/wan_s2v_rope_benchmark-v1
```

Logs use exclusive creation; choose a new path for each attempt. The footprint
budget is an explicit stop limit rather than a reservation or an estimate of
model size. Builds also run serially under the guard.
Free-plus-speculative pages are a narrow reserve, not a measurement of all
reclaimable RAM. A rejected launch does not establish OOM or a Wan leak.
Inactive and file-backed counters overlap; they are not additive free capacity.

### Opt-in cache-aware admission

The Activity Monitor correction exposes a false interpretation of the strict
reserve: file cache can be reused even when free-plus-speculative pages are
small. The explicit `--memory-accounting=file-cache` policy uses
`vm_stat` printed free bytes plus file-backed bytes. Speculative pages already
belong to the file-backed counter and must not be added again. Inactive,
purgeable, anonymous and compressed counters are not additional free capacity.
The existing `free-speculative` policy remains the default and old receipts
retain that meaning.

The new quantity is a cache-inclusive headroom estimate, not a reservation or
proof that every file-backed page is clean, cold or immediately reclaimable.
Admission requires a 15%-of-RAM policy floor on the selected estimate plus the
entire requested footprint budget. Runtime checks retain that policy floor,
pressure query threshold, group footprint, swap growth, timeout and fail-closed
telemetry.
The falsifiers are speculative double counting, anonymous inactive memory being
admitted as cache, missing/inconsistent counters, and a low-cache runtime sample
failing to terminate the owned child. The CPU guard suite passes 33/0 on Python
3.11 and 3.15. A live-telemetry child exits 0, with a peak sampled process-group
footprint of 7,111,088 bytes, minimum cache-inclusive headroom 33,956,118,528
bytes and no sampled swap growth. This checks collection, policy selection,
logging and child lifecycle; it does not qualify GPU-load memory behavior.
Source hashes, commands and v4 CPU/live logs are bound by
`wan-self-attention/cache-accounting-cpu-receipt-v1.json` (SHA-256
`a39bf0c404402ffdb94e87ed1feb206e8baa0dedb8075c2b1dcb3ecd1cc3063d`).
Rollback is omission of the explicit policy flag. Full video generation remains
deferred by user instruction while performance work continues on bounded probes.

Accounting sources: [Apple's Activity Monitor guide](https://support.apple.com/guide/activity-monitor/view-memory-usage-actmntr1004/mac),
[vm_stat's printed counters](https://github.com/apple-oss-distributions/system_cmds/blob/main/vm_stat/vm_stat.c),
[XNU host statistics](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/host.c)
and [speculative queue accounting](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_resident.c).
These public source links explain the counters; their main branches are not a
pinned source audit of the running Darwin build.

## Arithmetic cost, not a timing profile

For the ten main-block dense projections and self/text attention, counting each
multiply-add as two FLOPs:

| Operation | FLOPs | TFLOPs per block | Fraction of counted FLOPs |
| --- | ---: | ---: | ---: |
| Self QKV and output projections | `8*N*C*C` | 7.6773 | 15.43% |
| Self QK and PV | `4*N*N*C` | 27.4462 | 55.15% |
| Cross query and output projections | `4*N*C*C` | 3.8386 | 7.71% |
| Cross key and value projections | `4*M*C*C` | 0.0537 | 0.11% |
| Cross QK and PV | `4*N*M*C` | 0.3839 | 0.77% |
| Two FFN projections | `4*N*C*F` | 10.3643 | 20.83% |
| Total | | **49.7639** | 100% |

This excludes the audio injector, norms, bias, activation, trigonometry, copies
and framework overhead. The fractions are not latency fractions. Multiplying
by layer/step/CFG counts is conditional on the actual pipeline execution;
neither theoretical FLOPs nor summed overlapping GPU intervals predict video
latency. Sources: [block topology](../src/ml/video/wan_s2v_block.cr),
[projection plans](../src/ml/video/metal_linear_mps.cr), and
[full-block geometry](../examples/wan_s2v_full_block_benchmark.cr).

Both cross K/V weights are C-by-C; M changes their FLOPs, not weight size.
The ten matrices contain `8*C*C + 2*C*F = 351,272,960` values. Reading BF16
weights and writing their F32 expansions moves 2,107,637,760 logical bytes
(1.963 GiB) per expansion of all ten. This is not measured DRAM traffic and does
not establish that conversion dominates. Existing projection timings vary by
shape. The shared staging allocation is 270 MiB, rather than a whole F32 model.

## First experiment: precompute final RoPE coefficients

The [current kernel](../src/ml/video/kernels/wan_s2v.metal) computes the same
phase for every head and separately for Q and K. The coefficient depends on
positions, frequency configuration and pair index, not head, Q/K activation,
weights or diffusion timestep. For one block:

| Quantity | Inline Q plus K | One coefficient table |
| --- | ---: | ---: |
| Phase evaluations | 187,432,960 | 2,342,912 |
| Source-level precise trig calls | 749,731,840 | 9,371,648 |
| Additional persistent scratch | 0 | 18,743,296 bytes (17.875 MiB) |

This removes an 80-fold duplication of phase work, not an 80-fold latency
claim. The compiler may combine trig operations; source-level calls are not
instruction counts. Both full-size Q/K rotations and their memory traffic remain.

The first implementation deliberately rebuilds the final F32 cosine/sine table
inside every block encode and uses it for that block's Q and K. This avoids
cross-call cache invalidation and charges preparation to every candidate arm.
The original hi/lo phase construction, precision and rotation expressions must
remain intact. Storing a coefficient between kernels can still change rounding
through compiler contraction/materialization, so algebraic equality alone does
not admit this path.

Qualification order: bitwise inline/table comparison on B=2, all axis splits,
negative/high/fractional coordinates, in-place Q/K, changed positions in reused
storage and repeated encodes; all intermediate pinned upstream block fixtures;
full-shape RoPE pairs; then exact full-block pairs. Every timed encode must have
zero explicit buffer allocations and its existing single terminal wait. Table
extent/configuration/alias checks must reject before the first dispatch.

Opt-in callers must use `prepare!(precompute_rope: true)` and
`prepare_scratch!(..., precompute_rope: true)` during setup before passing the
encode flag. The standalone probe instead prepares only RoPE scratch. The
Metal-buffer allocation check does not cover small host-side config-key strings
or private framework allocations; the prepared-pipeline precondition keeps
kernel compilation outside the measured encode.

### Measured RoPE result: 2026-10-20

The candidate is numerically eligible but **rejected as a speedup** on this
Apple M2 Max. It remains opt-in; the transformer/pipeline continues using inline
RoPE. The predeclared primitive speed gate failed, so no full-block timing or
video run was launched for this candidate.

| Full-shape Q/K pair | Inline | Shared table, including build | Candidate saving |
| --- | ---: | ---: | ---: |
| AB | 8.196042 ms | 8.378291 ms | -0.182249 ms |
| BA | 8.206375 ms | 8.378292 ms | -0.171917 ms |
| AB | 8.236709 ms | 8.388750 ms | -0.152041 ms |
| Median | 8.206375 ms | 8.378292 ms | **2.095% slower** |

These are encode-through-terminal-completion wall times, excluding input fill
and comparison. GPU intervals agree in direction in all three pairs; the table
was rebuilt inside every candidate encode. All four warmup/measured comparisons
are bit-identical across 374,865,920 Q/K values each, with zero nonfinites or
strict violations. The one-value corruption control rejects exactly that value.
The measured encode counter reports zero Metal-buffer allocations and each arm
uses one terminal wait; no inter-operator host wait or activation readback was
added. The explicit allocation plan is 3,018,110,484 bytes, matched to Arena
accounting, and loads no model weights.

The focused two-batch test passes 3/0 examples. Official pinned block/injector
fixtures pass 6/0 each for BF16 and F32 weight storage, including all intermediate
checks, repeat/chaining and allocation guards. Those fixtures have the production
width but a short sequence; the full-shape pair is synthetic. Neither establishes
a full-spatial upstream block oracle or closes the known 40-layer drift.

All admitted runs finish successfully. The sampled minimum free-plus-speculative
reserve is 21,270,003,712 bytes (30.95% of 64 GiB); pressure-query minimum is 70%,
and no swap growth is observed during those runs. Sampling can miss brief peaks.
An earlier build was rejected before spawning when the conservative physical
reserve was low; its later recovery did not require stopping another project.

Raw logs and source/binary/object bindings are under the evidence directory
below. `rope-bindings-v1.json` has SHA-256
`74e55e08953349becb533a00db132f74b60c159eedcae262b568904595b03891`;
`rope-qualified-summary-v1.json` has SHA-256
`2f17407130ec98a23b25d6dc4afa8aee44b08481c1f605135ff3dc279c39ebd7`.
The summary independently recomputes pair membership, statistics, all comparison
controls, spec outcomes and sampled guard limits from the raw records. The
correlated mini-Quadrum source review is ROBUST for arithmetic order, indexing,
freshness and alias/extent guards; a performance-promotion claim is rejected.
Receipt review also recomputes the raw outcomes and all paired statistics.
`rope-binding-supplement-v1.json` (SHA-256
`c3f6d6d89049a95fb302f059810d773dca1c74873f07c6d5c8838df26273068a`)
binds the separately built full-block benchmark and its mismatched-chunk CLI
rejection. This is build/argument validation, not a full-block GPU timing.

This falsifies the simple assumption that duplicated precise trig dominates
latency at this geometry. It does not identify the hardware bottleneck by itself.
The next discriminator is stage attribution on existing asynchronous command
roots, with interval union/overlap reported after the existing terminal wait.

Positions are shared through the current transformer layer loop, so extending
reuse across layers is a later possible saving. Reuse across diffusion steps or
CFG branches requires a separate caller-lifetime proof. Buffer identity alone
does not establish unchanged positions, including FramePack offsets.

## Diagnostic: existing command-root intervals

The opt-in diagnostic labels four coarse phases at existing command-chain
boundaries: self preparation, self attention plus output projection/residual,
cross attention plus projections/residual, and FFN. The last phase includes
the existing post-GELU and post-output-projection continuations. It must not add
dispatches, submissions, waits, activation readback or Metal-buffer allocations.
Root metadata is consumed only after the existing terminal wait, outside the
arm's wall measurement. The default remains off.

The experiment requires a fresh benchmark-owned command root immediately before
the block. A root shared with preceding or following caller work cannot be
attributed exclusively to that block. Phase ranges use the exclusive root count
before each continuation; the next phase starts at that same sequence index.
Every root must be assigned once, have a completed status and valid positive
GPU timestamps, and the root-duration sum must reconcile with the existing
native aggregate. Missing metadata rejects attribution.

Report the interval envelope, interval union, `sum(duration)-union` overlap
excess and `envelope-union` gaps. Per-phase unions are separate diagnostics;
they cannot be added to infer block latency when roots overlap. Root spans can
also contain idle gaps, so they are not per-kernel execution times.

The falsifiers are CPU interval/partition counterexamples, then a guarded
`--comparison=profile-replay --phase-roots --pairs=1` full-block A/A replay.
Both arms must use query chunk 128, cross groups 320, the original K/V layout,
SDPA and inline RoPE; only B records phases. Require bit-identical outputs,
matching root/segment counts, one terminal wait and zero explicit encode-time
Metal allocations. Keep the pinned upstream intermediate gate and the 15%
memory reserve. This diagnostic cohort is not a speedup/overhead benchmark.

### Bounded fallback while full-block headroom is unavailable

The profiler's CPU interval/partition tests pass 9/0, including a large-epoch
counterexample that first rejected the unparenthesized union calculation.
The native four-root A/A test passes with bit-identical outputs, unchanged root
counts, one terminal wait and zero explicit Metal-buffer allocations. Existing
root identity/retention tests pass 2/0. The pinned intermediate fixture passes
6/0 each with BF16 and F32 weights, both with profiling disabled and in a
separate profiling-enabled replay. The latter uses a fresh chain around one
direct block, terminal-waits immediately after it, verifies the four named phase
ranges, and checks bit-identical output against the unprofiled block plus the
unchanged upstream gate. It does not attribute a shared block/audio root. These
are short-sequence fixtures at the model's real width. A full-spatial block
profile-replay launch was rejected before spawning by the physical-headroom
guard; no phase timings have been measured on that geometry. The latest attempt
had 15,469,756,416 free-plus-speculative bytes, below the 25,340,307,047 bytes
required for its 14 GiB process budget plus the unchanged 15% physical reserve.
This is a conservative admission result, not an OOM measurement.

The current-source qualification is indexed by external
`wan-self-attention/phase-profile-receipt-v2.json` (SHA-256
`354c9ece2f8f6f4d063956d3593b508045f35549cee1e9b186f7986ebadeaadf`).
It binds the source, native objects, binaries, raw logs, guard outcomes and
`phase-profile-invocations-v2.json`, which preserves the actual profile-enabled
BF16/F32 command environments. Quiet spec stdout alone does not establish those
flags. The combined interval/root-retention CPU suite passes 11/0 and the GPU
chain suite passes 2/0. An initial CPU link invocation omitted native link flags;
the corrected invocation passed without a source change. Correlated read-only
adversary review is ROBUST for the fresh short-fixture replay, with no claim
about full-spatial timings, profiler overhead or exclusive per-kernel costs.

### Smaller query chunks: 64 rejected, 96 incomplete

The opt-in `small-chunks` mode compares 64 or 96 query rows against 128 with
the complete K/V set, the same GPU-filled inputs and full-output upstream plus
native baseline checks. It rejects other candidates before fixture/GPU work.
The default chunk-size and other comparison modes retain their existing paths.
The full geometry is B=1, Q=K=36,608, H=40, D=128 (187,432,960 output values).

| AB/BA/AB pair | Chunk 128 | Chunk 64 | Baseline minus candidate |
| --- | ---: | ---: | ---: |
| AB | 8.544911 s | 9.874381 s | -1.329470 s |
| BA | 9.141665 s | 9.955952 s | -0.814287 s |
| AB | 10.090853 s | 12.138794 s | -2.047941 s |
| Median | 9.141665 s | 9.955952 s | **8.91% slower** |

Chunk 64 loses all three pairs and is rejected as an acceleration. All eight
warmup/measured full-output checks have zero strict/nonfinite violations and
bit-identical native baseline outputs. The maximum upstream absolute error is
1.8328428e-6 (maximum tolerance ratio 0.08037424); upstream bit identity is not
claimed. The final one-value corruption control rejects exactly that value.
Each timed arm has one terminal wait and zero explicit encode-time Metal-buffer
allocations; validation is separate from timing. The guard finishes with exit 0,
minimum free-plus-speculative reserve 15,801,204,736 bytes (22.99% of RAM), peak
sampled group footprint 4,886,873,696 bytes and no swap growth.

Chunk 96 is **incomplete**, with seven timed arms and six completed validations.
Those completed checks pass, but there is no final pair, decision or corruption
control. Free-plus-speculative bytes reached 10,251,468,800, below the unchanged
15% floor of 10,307,921,511. The old guard then hit the Darwin zombie-cleanup
bug documented above; its original group was subsequently verified absent.
Commit `86cc46fc` fixes that separate guard defect. Partial 96 timings are not
eligible for a speed verdict, and the headroom decline does not establish a
Wan leak. No production default or video-throughput claim follows either run.

Chunk 64 doubles query-chunk calls and command segments (48 to 96), and its CPU
encode counter rises to seconds. Submission/copy/framework overhead is a
hypothesis; overlapping host/GPU counters cannot be subtracted to localize it.
The next discriminator is existing-root attribution with less scratch memory,
or the native tiled-attention path under the smaller memory budget.

External `wan-self-attention/small-chunks-qualified-summary-v2.json` (SHA-256
`1cf93cfc30db578d794d169b5f615307f7f99013136f26a3d0696f097b0f5b2c`)
binds the raw logs, executable, attention backend, build/CLI guards and fixture.
Its v1 recomputation verifies pair membership, statistics, every validation and
negative control. Correlated receipt/source review is ROBUST for the bounded
64 rejection and the incomplete 96 classification. Refresh after source,
shape, inputs/precision, device/runtime/queue or memory-policy changes.

## Opt-in block scratch lifetime reuse

`WanS2VBlock.prepare_scratch!(..., reuse_scratch: true)` allocates seven
activation owners instead of twelve. Pass the same flag to direct `encode`;
the default layout remains unchanged. The four reuse groups are:

| Shared owner | Names with disjoint lifetimes |
| --- | --- |
| `norm1` | `norm1`, `self_attn` |
| `self_k` | `self_k`, `self_proj` |
| `self_v` | `self_v`, `cross_norm`, `ffn_out` |
| `self_q` | `self_q`, `ffn_in`; capacity is `max(N*C, N*F)` |

`self_res` must remain separate: putting it in the first group lets cross
attention overwrite it before the cross residual add. That more aggressive
proposal is rejected. Each permitted alias shares the exact `MetalBuffer`
object on the existing tracked serial queue. Partial groups, cross-group
overlap, distinct wrappers over the same address, and input/output/weight/trace
overlap reject before dispatch. Stage-pool Arenas reject before allocating.
The Arena retains a shared allocation until its last name is removed.

At the full target geometry, tested CPU accounting gives 8,792,855,040 bytes
for default scratch and 5,044,195,840 bytes for reuse: a prospective saving of
3,748,659,200 bytes (3.49 GiB). This is scratch storage, not total model memory
or a measured full-shape saving. No extra dispatch, wait or activation read is
introduced by the layout.

The pinned real-width **short** upstream fixture passes 15/0 with BF16 and
15/0 with F32 weights, native tiled self attention and fixed MPS projections.
All 24 block intermediates satisfy the unchanged finite upstream gate and match
default-layout bits exactly. Same-chain repeats, chained inputs and replay on
a second completed chain match default bits. Encode makes zero explicit
MetalBuffer allocations; each test chain uses one terminal wait. The 38-checkpoint
block/audio checkpoint set also retains its existing gate. Actual short-fixture
scratch is 651,776 versus 1,061,376 bytes; its dimensions differ from the full
target. Guards complete without swap growth; minimum sampled physical reserve
is 21.69% for BF16 and 18.02% for F32, above the unchanged 15% floor.

External `wan-self-attention/scratch-reuse-receipt-v1.json` (SHA-256
`3556473680210b58274f8315bb78a260c9309b46558e699e6f866f2d9626aa2a`)
binds source, executable, native objects, actual invocations, fixture and logs.
Correlated separate-agent source and log review is ROBUST within that scope.
That receipt does not qualify full-spatial execution, MPSGraph self attention,
colored audio integration, latency or video quality. Do not use this layout with combined
audio/reused-scratch maps: their additional aliases need a separate lifetime
proof. The short MPSGraph qualification below advances the next probe.
Refresh after source/lifetime, geometry/input/weight or
precision, queue/device/runtime or memory-policy changes. The known 40-layer
drift remains open.

## MPSGraph scratch qualification and compact profiling plan

The same-object scratch layout also passes the pinned real-width short fixture
with MPSGraph SDPA, fixed MPS projections and segmented commands: BF16 query
chunks of one and three rows, and F32 chunks of one row. Each case passes 15/0,
including 24 block intermediates, exact default-layout bits, same-chain repeats,
chained inputs and replay on a second completed chain. One-row chunks exercise
multiple graph calls; three-row chunks exercise padding at the two-token tail.
Each layout owns its chunk workspace. Reusing the original workspace with the
aliased scratch rejects changed bindings before dispatch.

The MPS workspace accepts B=1. These tests validate the original pinned B=2
fixture first, then select its first batch from six input arrays and all 35
checkpoint arrays. Weights remain unchanged. The marker hashes the same bytes
that were parsed and explicitly records that this is a derived B=1 oracle,
not a fresh B=1 upstream execution. The native B=2 BF16 regression also passes
15/0. All four runs retain the unchanged finite upstream tolerance, zero
explicit MetalBuffer allocations during encode and one terminal wait per chain.
Minimum sampled free-plus-speculative reserve is 18.59%; swap growth is zero.
The allocation counter excludes framework/GC allocations and the guard is
sampled, not a hard limit between polls.

`wan_s2v_full_block_benchmark` now accepts opt-in `--reuse-block-scratch`
for both arms, warmups and timed encodes. It checks prepared scratch bytes
against the selected plan. `--plan-only` verifies the pinned sidecar and prints
the full-shape plan before any GPU setup or dispatch. Successful plan-only
executions for profile replay (36,608 tokens, chunk 128) report:

| Bytes | Default layout | Reused layout |
| --- | ---: | ---: |
| Block scratch | 8,792,855,040 | 5,044,195,840 |
| Estimated total, including 1 GiB opaque MPS reserve | 13,443,721,748 | 9,695,062,548 |

The plan-only logs contain one JSON record each and no Metal initialization.
These are CPU memory plans, not full-spatial GPU allocations or speed
measurements. The attempted compact phase-root run was blocked before starting
a child: free-plus-speculative memory was 17,030,709,248 bytes, below the
21,045,339,751 bytes required for the unchanged 15% floor plus a 10 GiB process
budget. No full-block timing follows from that attempt.

External `wan-self-attention/scratch-mps-receipt-v1.json` (SHA-256
`bcc0508689419bf782b630503dccb05f09200d4e788ab4906c7ee195ce7dfee6`)
binds current source, both executables, native objects, actual invocations,
derived-oracle provenance and raw logs, including failed/blocked attempts.
Correlated separate-agent source, log and bound-hash review is ROBUST within
the short-fixture and CPU-plan scope.
DoD signals are four guarded 15/0 runs and two successful plan-only runs with
the exact scratch difference of 3,748,659,200 bytes. Full-spatial GPU execution,
latency, audio coloring and video quality remain unqualified; the known
40-layer drift remains open. Next: one guarded compact
`--comparison=profile-replay --phase-roots --candidate-self-chunk=128 --pairs=1`
run to attribute the four block phases, then choose the dominant measured
cost. Both A/A arms use the reused layout, so that run alone cannot establish
a speedup from coloring. Refresh on the source, lifetime, shape/input/weight,
precision, queue/device/runtime or memory-policy changes listed above.

## Bounded FFN bias/GELU epilogue qualification

Full video generation remains deferred. The standalone
[`wan_s2v_ffn_epilogue_benchmark.cr`](../examples/wan_s2v_ffn_epilogue_benchmark.cr)
compares the existing MPS bias dispatch followed by approximate GELU against
one fused, in-place dispatch. It keeps the F32 bias sum, GELU expression order
and `abs(v) >= 10` saturation branch. Production code and defaults are unchanged;
rollback is to stop using the standalone candidate.

On Apple M2 Max, the final v3 executable processes 4,096 x 13,824 F32 values
(216 MiB per arm). One warmup AB pair precedes AB/BA/AB measured pairs for
each source bias dtype. GPU fill, comparisons, trace checks and corruption
control are outside timed arms. Most inputs are deterministic values in
`[-4,4)`; the first 16 rows cover cancellation, signed zero, adjacent F32 values
around +/-10 and large finite magnitudes.

| Source bias | Existing GPU median | Fused GPU median | Fused / existing | Pair wins |
| --- | ---: | ---: | ---: | ---: |
| F32 | 3.819375 ms | 1.892625 ms | 0.49553 | 3/3 |
| BF16 | 3.570542 ms | 1.777042 ms | 0.49770 | 3/3 |

Both dtypes have zero bit differences, nonfinite values and strict-tolerance
violations for every warmup/measured output and the optional pre-GELU trace
and trace-path final output. The one-value `+1` control triggers exactly one
bit and tolerance failure. A separate 17-row run exercises the same edge rows
and checks. Each timed arm has one terminal wait, zero inter-operator host
waits, zero activation readbacks and zero encode-time tracked MetalBuffer
allocations. These are isolated kernel timings; the trace variant is checked
for correctness but not timed. They do not establish a full-block speedup.

The optional `--upstream-fixture=PATH` mode uses the immutable pinned
`wan-official-block0/wan-official.json` manifest and BF16 sidecar. It loads only
`blocks.0.ffn.0.weight` and `.bias`, supplies the upstream `ffn_mod` input, and
runs one 5,120-to-13,824 MPS projection at B=2, seq=2. Six separate GPU metric
buffers check both pre-GELU paths and both final outputs against each other
and the corresponding upstream arrays. All six checks pass at the unchanged
`2e-5 + 2e-5*abs(reference)` finite gate. Pre-GELU output is bitwise equal to
upstream; fused final output is bitwise equal to the current native path.
Both final paths differ from upstream at 4,578 of 55,296 values, with maximum
absolute error `4.7683716e-7`, normalized error `0.00873967` and zero strict
violations. The existing small GELU difference is preserved.

This qualifies the projection/bias/GELU slice given fixed upstream `ffn_mod`,
not its production earlier in the block, a full model, or an official F32-source
checkpoint. Manifest and sidecar hashes are verified, but inputs must remain
immutable throughout the run: hashing and consumption use separate file reads,
not a transaction against concurrent metadata-preserving modification.

DoD commands after the guarded release build are:

```sh
wan_s2v_ffn_epilogue_benchmark-v3 --rows=17 --trace-pre-gelu
wan_s2v_ffn_epilogue_benchmark-v3 --trace-pre-gelu
wan_s2v_ffn_epilogue_benchmark-v3 --upstream-fixture=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-official-block0/wan-official.json
```

All three were run serially through the cache-aware guard with a 1,536 MiB
owned-group footprint stop cap, 64 MiB swap-growth cap, 90 s deadline and
unchanged 15% policy floor, and exited 0. The synthetic explicit allocation
cap is 1 GiB; optional traces use 679,560,212 bytes at the default shape. The
upstream slice uses 426,110,072 explicit bytes including its F32 MPS weight
stage. Opaque MPS/driver storage is not in those explicit counts, and samples
can miss the peak of short runs; the sampled footprint is not an actual-peak
claim. No sampled swap growth occurred. Builds used the same memory/swap
limits with a 180 s deadline. The v2 constructor-type failure was corrected
before the successful v3 build; no v2 GPU run was made.

External `wan-self-attention/ffn-epilogue-receipt-v1.json` (SHA-256
`e110da994d742c467c432da8fcbec64167b27d6a282b2dd94d466a20524c5884`)
binds source, executables, native objects, actual commands, raw logs, fixture
identities and the recomputation script. Correlated separate-agent review is
ROBUST for synthetic epilogue timing and the immutable fixed-input upstream
slice. Requalify after source/native-object, input/shape/weight/precision,
device/runtime/queue or memory-policy changes. Integration needs the existing
whole-block numerical and latency gates; short kernel evidence cannot choose
the dominant complete-block bottleneck. Long-key self-attention and large
GEMM remain the higher-impact profiling candidates.

## Bounded full-key attention qualification

Full video, model and complete-block execution remain deferred. The standalone
probe keeps K=36,608, H=40, D=128 and F32 operands, but selects at most 1,024 independent
query rows. Its expected output is the corresponding per-head slice of the
pinned full-query upstream oracle; keys and values are never truncated. This
is a reduced-query primitive experiment, not a whole-block latency estimate.

The first comparison uses existing MPSGraph chunk-128 SDPA and existing native
tiled attention. The second compares native tiled attention with a probe-local variant
that loads immutable Q fragments before the ascending key loop. The latter
may save repeated loads, or lose occupancy through register pressure. It must
retain all score, online-softmax and value-reduction arithmetic in source order.
No production kernel or default is changed; stopping the standalone probe is
the rollback. The earlier rejected explicit-graph, small-chunk and K/V-layout
routes are not being reopened.

Inputs, plans, pipelines and buffers are prepared before timing. Each arm uses
one terminal wait and zero explicit encode-time MetalBuffer allocations; input
fill, GPU comparisons and compact metric readback are separate diagnostics.
Require one warmup and three AB/BA/AB measured pairs, finite strict upstream
and native-output comparisons for every pair, and a seeded one-value error
control. The Q-fragment candidate additionally requires native bit identity.
The primary timing uses Q=256: two MPS calls and four 320-group native
segments. Q=257 needs a third padded MPS call and a fifth native segment, so
its tail/last-query slice is a separate correctness gate, not pooled timing.
Numerical failure or
failure to win all three pairs rejects promotion; no tolerance widening is
admitted. Reduced-query occupancy, MPSGraph private allocations and whole-block
error propagation remain unqualified until their own later gates.

Use the cache-aware 15% admission policy, a separately declared owned-group
footprint cap, a 64 MiB swap-growth limit and a 90 s deadline. Keep explicit
buffers below 2 GiB and report opaque MPS storage separately as unbounded by
that count. Record source/native-object/binary/fixture identities with results.

### Observed results on M2 Max, 2026-10-20

The release build of
[`wan_s2v_bounded_attention_benchmark.cr`](../examples/wan_s2v_bounded_attention_benchmark.cr)
completed, followed by four guarded standalone runs. Each used one AB warmup,
three AB/BA/AB pairs, a strict GPU comparison after each pair and one seeded
error control. Input fill, pipeline/plan construction and validation were
outside the timed arms. The primary measure is host arm wall time through its
one terminal wait; the reported sum of command-root GPU intervals is a separate
diagnostic, not additive kernel cost.

| Q=256, start=0 paired cohort | Baseline median | Candidate median | Candidate/baseline | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| MPS chunk-128 vs existing native tiled | 61.482625 ms | 208.984750 ms | 3.39909 | 0/3 |
| Existing native tiled vs Q-fragment hoist | 207.366334 ms | 171.417000 ms | 0.82664 | 3/3 |

The Q-fragment hoist reduced native arm wall time by 17.336% in this cohort,
with bitwise-identical native output. This admits further primitive review;
it does not replace MPS or establish a generation speedup. The two rows are
separate cohorts, so the cached candidate was not paired directly against MPS.
Retain MPS as the evaluated block baseline and leave application/backend
defaults unchanged; the probe does not qualify a replacement.

Both comparisons also passed on the final 257 queries, starting at 36,351 and
ending at 36,608. These cover the partial native tile and padded third MPS
chunk. Their diagnostic timings are not pooled with Q=256, and the executable
forces timing admission false for all shapes except Q=256/start=0.

All 48 finite numerical checks passed the unchanged
`2e-5 + 2e-5*abs(reference)` tolerance. The largest upstream absolute error was
`8.493662e-7`, with maximum tolerance ratio `0.039907727`. All eight cached-vs-native
checks were bitwise equal. Outputs started as NaN, so the first finite checks
also reject omitted writes. Each run's final `candidate[0] += 1` control
produced exactly one bit difference and one strict violation. No activation
values were read back; only compact diagnostic metrics were read after their
separate terminal waits. Every timed arm used one terminal wait and zero tracked
encode-time MetalBuffer allocations; source inspection found no intermediate
host wait. This does not count opaque MPS allocations as tracked buffers.

The explicit plans matched Arena residency: 1,528,299,580 bytes for MPS-vs-tiled
and 1,520,435,260 bytes for the native pair at Q=256. The tail plans add 81,920
bytes. Runtime guards used a 5 GiB owned-process budget, 90 s deadline, 64 MiB
swap-growth limit and the cache-aware 15% floor. Across the four short runs,
the largest sampled footprint was 1,551,680,352 bytes (1.445 GiB), the smallest
sampled cache headroom estimate was 24,592,924,672 bytes (22.903 GiB), minimum
reported free pressure was 81%, and sampled swap growth was zero. All groups
ended with zero live processes. Samples do not establish a between-poll peak,
and file-backed cache is not guaranteed immediately reclaimable memory.

The receipt is
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/bounded-attention-receipt-v1.json`,
SHA-256 `9749560bf4b4db6b982a51ac4edc4128bddcefcf0a4dd2e5473e91a83e394601`.
It binds 46 files: the final probe and native objects/binary, pinned oracle and
sources, exact guarded invocations, raw logs and the CPU recomputation script.
The successful qualification consumed binary `wan_s2v_bounded_attention_benchmark-v3`;
the initial NamedTuple-access compile failure and superseded v2 build are
retained as history, not runtime evidence. Full-file oracle hashes are checked
before parsing/slicing, under the explicit immutable-fixture assumption.
Refresh after code, binary/native objects, fixture/input, precision, device or
runtime, queue, shape or lifetime changes. Full-query occupancy, real BF16 block
numerics, the previously open stack drift and full generation remain open.

## Other candidates, in order of investigation

1. **Static audio K/V.** In
   [the audio injector](../src/ml/video/wan_s2v_block.cr), K/V projections and K
   normalization depend on `local_audio`, not the changing video activation.
   Twelve injection sites at 21 frames, five audio tokens and C=5,120 need
   51,609,600 bytes (49.22 MiB) for normalized K and V. Reuse is conditional on
   proving audio/weight/configuration identity across steps and CFG branches.
   Require byte comparisons to recomputation and mutation/invalidation controls.
   Text K/V caching has the same conditional argument but costs 1.5625 GiB for
   40 layers and two branches while eliminating only 0.11% of counted FLOPs;
   prioritize the smaller cache under the current memory constraints.
2. **Share K/V tiles across more query groups.** The existing
   [F32 tiled kernel](../src/ml/video/kernels/attention_tiled.metal) shares each
   key/value matrix operand within an eight-query SIMD group, with zero
   threadgroup storage. Staging one K/V tile across two or four such groups
   could reduce repeated loads. K=32, D=128 and two F32 tiles would add 32 KiB
   of threadgroup storage; double buffering would need 64 KiB. Check device
   limits and occupancy before implementation. All SIMD groups, including
   fully masked tail groups, must reach both tile barriers. Preserve
   each row's ascending key reduction, then check bitwise parity to the old
   native kernel, strict full-shape upstream parity and the real full block.
   The existing native tiled reduction is not automatically a qualified
   replacement for the MPSGraph long-key path. Its segmentation budget counts
   threadgroups, so changing queries per group also changes work per segment.
   Q loads are textually inside the key loop, but compiler hoisting is unknown;
   explicitly keeping Q matrices alive can increase register pressure.
3. **FFN bias/GELU epilogue.** A post-MPS bias-plus-GELU kernel could remove one
   full FFN-intermediate read/write, about 3.77 GiB of logical traffic per block,
   and one dispatch, while retaining the MPS dot-product reduction. GELU already
   aliases the FFN-input backing, so this is not an additional buffer saving.
   Preserve the rounded F32 bias result and the current GELU operation order;
   coefficient/FMA changes still require bitwise intermediate and block checks.
   Traces must continue to expose the pre-GELU value when requested.
   The bounded qualification above confirms the isolated candidate; full-block
   integration and latency remain open.
4. **BF16/F32 GEMM and paired projections.** Q/K/V are currently separate
   linear calls. Packing projections or using native mixed-storage GEMM may
   improve input reuse and avoid expansion, but it changes matrix shape or the
   reduction tree. BF16-to-F32 conversion is exact; a new accumulation tree is
   not necessarily bitwise equal. Qualify each shape before replacing MPS.
   Park another direct-tiled comparison until there is a new backend or premise:
   the existing 5,120-to-13,824 probe at 64/512/2,048 rows already found MPS
   6.77--20.29 times faster (see [projection evidence](video-mps-projection.md)).
   That comparison checks native tolerance, not wide-shape upstream parity.

These are hypotheses about latency. Their logical byte counts are not measured
memory-controller traffic. A larger dispatch or fewer command segments has
already failed to improve complete-block latency in a prior comparison.

## Which mathematical rewrites are legal

Softmax attention can be tiled with an online normalizer in real arithmetic.
For a tile S, maintain maximum m, denominator l and unnormalized output o:

```text
m' = max(m, max(S))
l' = exp(m-m')*l + sum(exp(S-m'))
o' = exp(m-m')*o + sum(exp(S-m')*V)
output = o/l
```

This is the IO-saving principle in
[FlashAttention](https://arxiv.org/abs/2205.14135). It retains every key.
Different F32 reduction/normalization order can change results, so exact real
algebra is not a bitwise or unchanged-tolerance certificate.

There is no universal exact `Q*(K^T*V)` shortcut for softmax attention like a
linear recurrent state. A rank-one score matrix can become full rank after
softmax: take q=(0,1,2,3), k=(ln(1),ln(2),ln(3),ln(4)). The elementwise
exponential of `q*k^T` is the Vandermonde matrix with rows
`[1,1,1,1]`, `[1,2,3,4]`, `[1,4,9,16]`, `[1,8,27,64]`.
Its determinant is 12; row sums are 4,10,30,100, so the row-softmax determinant
is exactly `1/10000`, with rank four. Scaling can be absorbed in q or k.
This rejects the universal low-rank rewrite; it says nothing about the empirical
rank or approximation quality of a particular learned Wan attention matrix.
Surrogates, block skipping and step reuse need a separately authorized quality
budget and cannot inherit the current exactness claim.

## Analysis provenance and refresh

The CPU-only cost receipt is
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/wan-static-cost-model-v1.json`,
SHA-256 `7410fa83e5ebc8719f1bade0ab85faa14c213a47f5f29f17cc6f194d0506edc1`.
It binds the pre-experiment source at `0f94f5b3`; the arithmetic counts are not
GPU measurements. The independent CPU formula check confirms 49,763,907,010,560
counted FLOPs, 351,272,960 dense weights and 18,743,296 table bytes.
Recompute after topology or geometry changes. Requalify performance/numerics
after kernel, lifetime, input, weight, precision, device/runtime or queue changes.
