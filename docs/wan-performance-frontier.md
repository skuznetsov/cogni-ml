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

The bounded F32/F16/BF16 attention comparison below retains F32. Both lower
precision modes fail the unchanged dtype-matched upstream and F32 gates. F16
has a selected-query latency advantage; BF16 is slower on this M2 Max. Neither
result establishes video quality or a full-block speedup.

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

## Shared K/V tile probe

The bounded-query/full-key F32 experiment compares the Q-fragment-hoist baseline
against two or four SIMD groups sharing one 32-by-128 K tile and one V tile.
Both arms cache Q.
The two staged tiles require 32 KiB of static threadgroup memory, with no
additional MetalBuffer. Query rows in a threadgroup share the same head;
cooperative loads retain exact F32 values. Each row retains the original
ascending key, score, softmax and PV reduction order. Uniform barriers protect
load visibility and tile reuse, including fully masked query groups at a tail.
This is a probe-only performance rewrite; removing its comparison mode is the
rollback. Model kernels and application defaults remain unchanged.

The prediction is lower native arm wall time through reduced K/V loading. It
is falsified by strict upstream or native-bitwise failure, a nonfinite value,
an omitted write, or any of the three primary pairs failing to improve. Cache
reuse may already remove most device traffic, while staging, register pressure
and barriers may cost more than they save. A primitive win cannot establish a
whole-block, MPS-relative or generation improvement.

Before dispatch, query the actual device's threadgroup-memory limit and each
pipeline's static memory, execution width and maximum threads. Require 32-lane
SIMD groups, enough static storage and 64/128 supported threads. Divide the
existing 320-SIMD-group segment budget by groups per threadgroup; grouping must
not multiply work per submission. Keep one terminal wait and no tracked encode
MetalBuffer allocations. The existing 15% cache-aware floor, 5 GiB owned-group
cap, 64 MiB swap-growth limit and 90 s runtime deadline remain in force.

The DoD is a release build followed by guarded
`--comparison=q-cache-vs-shared-kv-sg2` and `--comparison=q-cache-vs-shared-kv-sg4`
runs of the bounded attention probe using the pinned fixture. Use Q=256/start=0
for one AB warmup and three AB/BA/AB measured pairs. Also check Q=257/start=36,351
for each grouping, with timing admission disabled. All pairs must pass the
unchanged finite upstream tolerance and bitwise equality to the Q baseline;
the seeded one-value control must fail exactly once. Do not run the full block
or model for this qualification. Record source/object/binary identities and
recompute timing and guard results from raw logs before any admission claim.

### Observed shared-tile result on M2 Max, 2026-10-20

Both shared-tile variants pass the numerical gate but are **rejected as
speedups** in the primary Q=256/start=0 cohort:

| Shared SIMD groups | Q-cache baseline median | Shared K/V median | Candidate/baseline | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| 2 | 172.302000 ms | 476.170458 ms | 2.76358 | 0/3 |
| 4 | 171.949417 ms | 207.044208 ms | 1.20410 | 0/3 |

Each row is its own paired cohort, with one AB warmup and AB/BA/AB measured
pairs. GPU command-root intervals agree in direction in every primary pair.
The four-group variant remains 20.410% slower; do not promote it or sweep more
group counts on the assumption that staging must help. Both variants also pass
the separate final-257-query correctness check, including fully masked SIMD
groups reaching both barriers. Tail timings are diagnostic only.

All 48 finite numerical checks pass at the unchanged tolerance, and all 16
candidate-vs-baseline checks are bitwise identical. The largest upstream
absolute error is `8.493662e-7`, with tolerance ratio `0.039907727`. NaN output
initialization guards omitted writes; all four seeded one-value controls
produce exactly one bit difference and one strict violation. Every timed arm
reports one terminal wait and zero tracked encode-time MetalBuffer allocations.
Only compact diagnostic metrics are read back after separate validation waits.

The actual device limit is 32,768 bytes of threadgroup memory, and both compiled
pipelines report exactly that amount of static storage. Pipeline execution width is 32,
with maximum 448 threads; launches use 64 or 128. Four segments at Q=256 and
five at Q=257 retain at most 320 SIMD groups per submission in both arms.
There are 1,144 key tiles and two threadgroup barriers per tile. Staging,
occupancy and barrier costs are plausible explanations for the slowdown, but
no hardware cache, occupancy or stall-counter profile was collected. The
measurements reject this staging design at this geometry; they do not isolate
the physical cause or reject all K/V reuse designs.

The explicit allocation plan remains 1,520,435,260 bytes at Q=256 and
1,520,517,180 bytes at Q=257. Runtime guards all exit 0, with a largest sampled
owned-group footprint of 1,538,540,192 bytes (1.433 GiB), minimum sampled cache
headroom estimate of 25,794,625,536 bytes (24.023 GiB), minimum pressure query
82%, no sampled swap growth and zero live owned processes at completion.
The 15% cache-aware floor, 5 GiB runtime cap, 64 MiB swap-growth cap and 90 s
deadline stay unchanged. These short sampled runs do not establish an actual
memory peak or immediately reclaimable cache. Both guarded build steps also
exit 0 under their 1,536 MiB cap and 180 s deadline.

Building this standalone probe additionally requires the probe-only
[`wan_s2v_bounded_attention_limits.mm`](../examples/support/wan_s2v_bounded_attention_limits.mm)
helper. Compile it with `xcrun clang++ -std=c++17 -O3 -fobjc-arc -c`, and append
its object to the existing native bridge link flags when building the Crystal
executable. It reads the existing device's capability without creating a
device, queue, buffer or command. A standard Makefile target was not qualified;
the exact successful compile and link commands are in the bound invocation
records. The tested executable is
`wan_s2v_bounded_attention_benchmark-shared-kv-v1`.

The external receipt is
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/shared-kv-attention-receipt-v1.json`,
SHA-256 `3a5e75e639b0dfb0c585cffcb65b23b52f5d52533d87788994548d73adacc40f`.
It binds 45 files: source/native objects/executable, pinned oracle and sources,
both build and four runtime invocations, raw logs and a CPU recomputation
script. Source and evidence reviews return ROBUST within this bounded probe's
scope. They are correlated mini-Quadrum checks, not independent hardware
evidence. Refresh after the source, binary/native objects,
fixture/input, shape, precision, device/runtime, queue or lifetime changes.
No model kernels or defaults changed, and no full block, model or video ran.
MPS remains the evaluated block baseline; its latency was not measured in this
new paired cohort. Full-query occupancy, real BF16 block numerics and the open
stack drift retain their separate gates.

## Register-level K/V reuse probe

The tested design lets one SIMD group own two independent
eight-query tiles of the same head. Load each K fragment once for both score
accumulators and each V fragment once for both output accumulators. Both query
tiles cache Q. Preserve each row's ascending key, score, online-softmax and PV
reduction order, F32 storage and F32 arithmetic. This probe uses no threadgroup
staging, barriers, host waits between operators or additional MetalBuffer.
Only the standalone benchmark and its new kernel may change; model kernels
and defaults remain unchanged. Removing the new comparison mode is rollback.

The prediction is lower Q=256/start=0 native arm wall time than the Q-cache
baseline. The main falsifier is register pressure/spilling or lower occupancy
erasing K/V-load savings; source-level reuse alone does not prove physical
traffic savings. Reject on any numerical/nonfinite/omitted-write failure,
native bit mismatch, or any of three primary measured pairs failing to win.
Check pipeline width 32, support for 32 threads and zero static threadgroup
storage before allocating large buffers. Retain at most 320 eight-query tile
equivalents per submission: the candidate gets 160 sixteen-query groups.

DoD: a guarded release build, then `--comparison=q-cache-vs-register-kv-q2`
with the pinned full-K=36,608/H=40/D=128 upstream oracle at Q=256/start=0 and
Q=257/start=36,351. Require one AB warmup and three AB/BA/AB pairs, strict
finite upstream checks, native bit identity, NaN output sentinels and a seeded
one-value error control. Q=257 is correctness-only. Each timed arm retains one
terminal wait and zero tracked encode-time MetalBuffer allocations; diagnostic
validation waits and metric readbacks remain outside timing. If the primary
native speed gate passes, also pair the candidate directly against MPS
chunk-128 at Q=256 before any MPS-relative claim. Do not run a complete block,
model or video. Retain the cache-aware 15% floor, 5 GiB owned-group runtime cap,
64 MiB swap-growth cap and 90 s deadline, serial execution and source/binary/
oracle/raw-log bindings. A primitive win leaves real BF16 block numerics,
whole-query occupancy, stack drift and generation throughput unqualified.

### Observed register-reuse result on M2 Max, 2026-10-20

The primary Q=256/start=0 cohort passes the numerical gate but is **rejected
as a speedup**:

| Q-cache baseline median | Register K/V median | Candidate/baseline | Candidate wins |
| ---: | ---: | ---: | ---: |
| 171.445833 ms | 244.636000 ms | 1.42690 | 0/3 |

One AB warmup and three AB/BA/AB measured pairs show a 42.690% regression.
The separate Q=257/start=36,351 check also passes correctness; its timing is
diagnostic only. All 24 finite checks pass at the unchanged tolerance, including
eight candidate-vs-baseline checks with bitwise identity. The largest upstream
absolute error is `8.493662e-7`, with tolerance ratio `0.039907727`. Both seeded
one-value controls produce exactly one bit difference and one strict violation.
NaN output initialization covers omitted writes; there is no separate logged
NaN-injection control.

The pipeline reports execution width 32, maximum 384 threads and zero static
threadgroup storage; launches use one 32-thread SIMD group. Four segments at
Q=256 and five at Q=257 preserve the 320 eight-query tile-equivalent cap per
submission. Every timed arm reports one terminal wait and zero tracked
encode-time MetalBuffer allocations. Diagnostic validation waits and compact
metric readbacks are excluded from timing.

This mapping has no threadgroup barriers, so barriers alone cannot explain
the regressions across the two reuse designs. Doubling live query, score and
output state may hurt register pressure, scheduling or latency hiding, but no
hardware-counter profile identifies the cause. Pipeline maximum threads are
not an occupancy measurement. Source-level K/V reuse does not prove lower
physical memory traffic. Park further query-group sweeps without a new premise.

The explicit allocation plan remains 1,520,435,260 bytes at Q=256 and
1,520,517,180 bytes at Q=257. The guarded release build and both runtime runs
exit 0. Runtime samples show a largest owned-group footprint of 1,536,737,976
bytes (1.431 GiB), minimum cache-headroom estimate of 24,702,353,408 bytes
(23.006 GiB), minimum pressure query 83%, no swap growth and empty final owned
groups. The 15% cache-aware floor, 5 GiB runtime cap, 64 MiB swap-growth cap and
90 s deadline remain unchanged. The build uses the existing 1,536 MiB cap and
180 s deadline. Sampling does not establish an actual memory peak or immediately
reclaimable cache.

The tested executable is `wan_s2v_bounded_attention_benchmark-register-kv-v1`.
It reuses the unchanged probe-only capability-helper object described above;
the exact release-build and runtime commands are in the bound invocation
records. The conditional `mps-vs-register-kv-q2` path compiles but was not run
because the native speed gate failed. It remains runtime-unqualified, and this
cohort supports no MPS-relative claim.

The external receipt is
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/register-kv-attention-receipt-v1.json`,
SHA-256 `c1995f20e726d2f85c034269047f0c419169c4733eda8d02f1ed954bf6a7d012`.
It binds 37 files, including source/native objects/executable, the pinned
oracle and its sources, build and runtime invocations, raw logs and a CPU
recomputation script. Source review and a separate raw-evidence recomputation
return ROBUST for this bounded negative finding; the performance hypothesis
is BROKEN at Q=256/start=0. These same-lineage mini-Quadrum reviews are
correlated. Refresh after source, binary/native objects, fixture/input, shape,
precision, device/runtime, queue or lifetime changes. No model kernels or
defaults changed, and no complete block, model or video ran. Real BF16 block
numerics, full-query occupancy and stack drift retain their separate gates.

## Static audio K/V cache: bounded qualification contract

This contract was sealed before measurement; bounded results follow below.
This is ordinary exact reuse, not an LTP/WBA promotion. The trigger is repeated
serial use of unchanged local audio and one injector's weights/configuration.
The corridor is that injector's K projection,
K RMS normalization and V projection. Cache normalized K and V in caller-owned
F32 device storage. Preserve the existing BF16-resident weights, F32 MPS
projection/accumulation, bias and norm order. The changing video queries,
attention and output projection remain outside this cache.

The cache key must bind request/audio/weight revisions, injector identity,
geometry, dtype, buffer bindings and numerical configuration. Pointer identity
alone is insufficient. In-place mutation requires advancing the corresponding
revision before the next encode; this caller obligation cannot be detected by
a pointer-only host check without additional GPU validation. No automatic
cross-request, concurrent-chain or CFG sharing is admitted. A cache hit may
reuse the existing immutable result only while all producing buffers, plans,
weights and cache storage remain alive and ordered on the same serial chain.
An aborted/discarded producer invalidates its cache state; a host-side key is
not proof that GPU work completed. Release follows the caller's terminal wait.

The first probe uses one injector, B=1/F=21/T=5/C=5,120/H=40/D=128, with
selected original BF16 checkpoint tensors. K plus V require 4,300,800 bytes;
twelve independent injectors would require 51,609,600 bytes (49.22 MiB) for one
conditioning identity. That is a capacity estimate, not a production cache.
Use the pinned original upstream K/V/norm modules on MPS at the same inputs
and weights, with CPU fallback disabled. Bind all oracle bytes and sources.

DoD: guarded release build and a bounded GPU probe comparing eight complete
K/V recomputations against one cache fill plus seven hits. Charge the initial
fill and identical downstream GPU copies/consumption to their respective arms;
also report fill/hit counts and CPU encoding versus terminal-wait intervals.
Use one AB warmup and three AB/BA/AB measured pairs. Require the unchanged
finite upstream tolerance `2e-5 + 2e-5*abs(reference)` for raw K, normalized K
and V, plus native bit identity. Exercise same-address audio and weight/bias
mutation with revised keys, request/injector/config revision changes, explicit
invalidation and a fresh command chain. Seed one-value and nonfinite validation
controls and initialize outputs with NaNs to expose omitted writes.

Each timed arm retains one terminal wait, zero activation readbacks and zero
tracked encode-time MetalBuffer allocations. Diagnostic metric readbacks occur
after separate validation commands. Reuse buffers allocated during setup; no
model/Studio/full-block/video run is permitted. Retain serial execution, the
15% cache-aware memory floor, a 3 GiB owned-group runtime/oracle cap, 64 MiB
swap-growth cap and 120 s deadline; builds use 1,536 MiB and 180 s. Stop on any
guard/numerical/lifecycle failure. Rollback removes the standalone probe;
production kernels, transformer routing and defaults remain unchanged.

The local potential is repeated projection/norm work avoided after charging
fill, consumption, key checks and storage. A primitive win cannot establish
global descent: audio K/V contributes a small fraction of counted model FLOPs,
and lifetime, memory pressure, queue/encode overhead and the actual dominant
full-request phase still require separate evidence. Reject a global speedup
claim until that complete corridor is measured. The strongest failure modes
are stale same-address contents and uncharged fill/consumer costs, rather than
the algebra of reuse itself.

### Bounded audio K/V result, 2026-10-20

**VERIFIED within the fixed-input injector-0 primitive; adversary: ROBUST in
that scope.** A guarded release build and original-upstream oracle export
passed, followed by one AB warmup and three AB/BA/AB measured pairs. Each arm
performs eight uses at B=1/F=21/T=5/C=5,120/H=40, with original BF16-resident
weights and F32 MPS projection. Recompute performs eight fills; cached performs
one fill and seven hits. Identical K/V GPU-copy consumers are charged after
every use. The input repeats the pinned five-row fixture with GPU-generated
row offsets, exercising the full 105-row geometry; it is not a captured
21-frame encoder output or a representative-input quality study.

| Independently computed median, ms | Recompute | Cached |
| --- | ---: | ---: |
| Completed arm wall time | 21.136250 | 3.263417 |
| Sum of completed command-root GPU intervals | 20.267417 | 2.666376 |
| Host encode interval | 0.181875 | 0.211458 |
| Terminal-wait interval | 20.847666 | 3.128542 |

Completed wall time is **6.48x faster / 84.56% lower**, with 3/3 paired wins;
individual wall ratios are 5.18x, 6.48x and 6.38x. CPU encoding's median did
not improve. Each row is a separate median and the rows need not add up.
GPU-root intervals are not per-kernel active time. Three pairs on one device
do not establish a latency distribution or whole-video speedup.

Raw K and V are bit-identical to the pinned original upstream MPS modules.
Normalized K has 232,526 values with differing bit patterns, maximum absolute
error `9.536743e-7`, maximum tolerance ratio `0.008825604`, zero strict
violations and zero nonfinite values. The finite tolerance remains
`2e-5 + 2e-5*abs(reference)`. All eight native K/V comparisons across warmup
and measured pairs are bit-identical. Same-address mutations of audio and all
five selected weight tensors expose stale results when revisions are omitted;
advancing the relevant revision forces refill, with twelve bit-identical K/V
checks. Independent `+1` and NaN controls each produce exactly one strict
violation. Host checks cover changed request/audio/weight/injector/config keys,
fresh chains and bindings, explicit clear, discarded producers, caller-array
snapshotting and rejection of replacing an incomplete producer.

Each timed arm has one command root, one terminal wait, zero inter-operator
waits, zero activation readbacks and zero tracked encode-time MetalBuffer
allocations. GPU comparison/metric readbacks and oracle fixture export are
explicit diagnostic boundaries outside these arms. The MetalBuffer counter
does not cover Objective-C/MPS allocations. The release helper relies on the
caller to drain work first; the audited CLI honors that obligation, including
its exceptional timed-arm path. This does not qualify a general production
cache API or cross-step/CFG sharing.

Runtime peak sampled owned-group footprint is 272,352,096 bytes (259.74 MiB),
minimum cache-inclusive headroom 27,838,971,904 bytes (25.93 GiB), minimum
pressure percentage 82%, and sampled swap growth zero. Oracle and successful
build sampled peaks are 1,343,718,936 and 518,112,240 bytes. All guards retain
the 15% floor and their declared caps, and every process group drains to zero.
These are sampled bounds, not continuous peak measurements. Initial build
attempts exposed a CLI nil-narrowing and a NamedTuple-access compile error;
both were repaired before the successful release build and sole GPU run.

Production cache remains disabled. No full block, model, Studio or video ran.
Rollback removes the three standalone probe/oracle files and this contract;
production routing and kernels are unchanged. Requalify after changes to
source/native objects, oracle/weights/input, geometry, precision, identity or
lifetime rules, device/runtime or queue behavior. The next falsifier is the
bounded submission-cadence experiment below, not a full generation run.

The external receipt and raw-log recomputation scripts are under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/`.
`audio-kv-receipt-v1.json` binds the final sources, native objects and binary,
oracle bytes, exact guarded invocations, raw logs, arithmetic and historical
runner anchors (86 file bindings; SHA-256
`ec66010ae21109ebc33196ae674763418ebf7099ea62e5e1fa3e89aac9b6c56e`).
Recheck it without a GPU/model run using:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/check-audio-kv-receipt-v1.py --verify
```

The executed runtime DoD is the command vector in
`audio-kv-pairs-v1.invocation.json`: `wan_memory_guard.py` with file-cache
accounting, 3,072 MiB group cap, 64 MiB swap-growth cap and 120 s deadline,
running `wan_s2v_audio_kv_benchmark-v3 --reference=.../reference.json`.
Expected signal: exit zero, all gates passed, 3/3 paired wall wins and complete
group cleanup. The receipt is an evidence index; its hash checks do not replace
the numerical, lifecycle, timing and guard observations above.

### Whole-request introspection, 2026-10-20

The strongest surviving omission is submission cost, rather than another
unqualified attention tile sweep. Source inspection of the historical runner
and its receipt confirms MPS projections, MPS chunked self-attention at 512
queries, and native text cross-attention with 320 groups per segment. The
slow run already selected those routes; missing MPS routing is refuted.
Its working canvas is 512x832, latent grid 64x104 and delivered crop 480p.
With motion tokens dropped, `(21+1)*32*52 = 36,608` sequence tokens, matching
the receipt. The alternative calculation using the delivered crop was wrong
and is rejected. Motion-drop does not imply a smaller sequence than this target.

The runner reports `queued denoise_step` before its sole final
`commit_and_wait`. Those timestamps include host encoding, scheduling,
submission backpressure and any GPU progress already made; they are neither
completed-step timestamps nor measurements of pure CPU encoding. The killed
19/40 run produced no completed raw video output. A recorded main-thread
sample in `continue_async! -> commitAndContinue -> AGX queue semaphore` supports
submission backpressure as a hypothesis, not a GPU-time percentage or a claim
that the GPU was idle. Do not extrapolate a qualified generation rate from it.

For 40 steps, two CFG forwards per step and 40 blocks, source/arithmetic counts
are 3,200 block calls, 36,800 MPS projection calls and 230,400 self-attention
MPSGraph encodes at chunk 512. These plans are prepared and reused, not
recompiled per chunk. Native text cross-attention has
`ceil(36,608/8)*40 = 183,040` query/head groups per block. At 320 groups per
segment, there are 572 dispatches and 571 explicit continuations per block,
or **1,827,200 continuations** for that operation alone per request. These are
source counts, not measured driver/kernel times; audio, other layer boundaries
and implementation-private MPS continuations add work.

Investigate these levers in this order:

1. **Separate dispatch geometry from submission cadence.** Retain the same
   320-group kernel dispatches, inputs, reduction order and serial dependencies;
   compare one, two and four dispatches per command root in a bounded K=512
   cross-attention probe. Count roots and measure CPU encode, terminal-wait and
   completed GPU-root intervals separately. An equal-work speedup with unchanged
   outputs supports the scheduling hypothesis. No gain or Metal completion
   failure rejects it. The old 4,096-group long-key completion failure is a
   relevant guard: grouping dispatches may still make a root too long. Do not
   change the production cadence before qualification or conflate this probe
   with increasing groups per dispatch. Cross-attention's 0.77% FLOP share is
   not evidence that its million submissions are cheap.
2. **Account for host tracking objects.** The command-chain tracker releases
   completed Metal roots, but retains one CPU summary per root until final
   completion; it also installs callbacks and uses a lock. The tracked
   MetalBuffer-allocation counter does not cover these Objective-C objects or
   opaque MPS resources. A bounded many-root CPU/allocation profile can test
   scaling. Aggregate normal-run summaries while retaining detailed records
   only for profiling is a possible later design, requiring preservation of
   error reporting, pending-root ownership and timing semantics. No latency or
   memory saving is claimed from source inspection alone.
3. **Profile the dominant math against the actual backend.** Self QK/PV accounts
   for 55.15% of counted block FLOPs; dense projections account for 44.08%.
   Use a bounded completed-work profile to distinguish kernel execution,
   submission stalls, copies, BF16 expansion and opaque MPS work before a new
   backend rewrite. Native direct BF16 GEMM and shared/register K/V grouping
   already regressed; repeating them without a changed premise is not progress.
   A library GEMM/layout or attention backend is a new hypothesis, but different
   reduction order must pass the same strict upstream gate. The faster explicit
   QK/softmax/PV route remains rejected by its real-block numerical failures.
4. **Keep exact static reuse proportionate.** Audio K/V caching can remove at
   most 1,872 of 1,920 such projection calls across two separately conditioned
   branches (39/40 saved per branch). It avoids about 294.44 GB of logical
   BF16-read/F32-stage-write traffic; this is not measured DRAM traffic. Its
   maximum saved 10.3054 TFLOP is only **0.00647%** of the counted 159,244.5
   TFLOP request arithmetic, excluding audio's other operators and the rest of
   the pipeline. Capacity is 49.22 MiB per branch, 98.44 MiB for two branches.
   A projection-only speedup cannot be multiplied into a whole-video speedup.

Mathematically, the existing tiled attention already exploits the low-rank
Q/K head factors; softmax does not preserve that factorization. A surrogate,
sparse/window attention, skipped diffusion step or reused changing activation
would change the algorithm and needs a separate quality contract. It is not
an exact counterpart of parallelizing a GDN recurrence and is not admitted by
the current unchanged-tolerance task. Prefer eliminating repeated fixed work
and scheduling overhead first; even eliminating all counted dense projections
would leave the 55.15% self-attention arithmetic.

The arithmetic magnitude also limits the promise of local fusions. For this
unchanged dense algorithm, 159,244.5 TFLOP would take 15,924.45 seconds at a
hypothetical sustained 10 TFLOP/s, or 1,592.45 seconds at 100 TFLOP/s. These
illustrations are not device peaks, measured throughput or a runtime forecast;
they exclude other work. They become lower bounds only if the stated rate
bounds effective throughput for this workload. A proved exact reduction in
work invalidates the count and requires recomputation. Submission improvements
can recover throughput, but a small epilogue or static cache alone cannot
remove the dense attention obligation. Do not infer achieved throughput from
the old queued milestones.

The historical runner/source/receipt/log and the arithmetic inputs must remain
bound in the bounded cache evidence packet. Refresh these counts if geometry,
CFG branches, steps, injection sites, chunk/segment cadence or backend routing
changes. No full block, model, Studio or video run is authorized by this audit.

### Sealed text cross-attention cadence experiment, 2026-10-20

The admitted next experiment changes only submission cadence in a standalone
probe. It does not instantiate a block, model, Studio or generation pipeline.
The fixed target is B=1, Q=36,608, K=512, H=40, D=128, native BQHD F32,
eight queries per SIMD group and 320 groups per kernel dispatch. All 572
dispatch ranges, kernel arguments, inputs and reduction order remain identical.
One, two or four dispatches share a command root, giving 572, 286 or 143 roots.
No production route or default changes in this slice. This is command batching,
not an LTP/WBA transformation.

This is a CAUTION performance experiment. The factual anchor is the unchanged
`attention_tiled_range` implementation and the source-counted 571 continuations
per text-attention call above. The hypothesis is that fewer command roots reduce
completed latency. A speedup in encoding alone does not establish that claim.
The strongest failure mode is a longer root failing completion or causing a
numerical, unwritten-output or memory regression. A 257-query boundary case
exercises the partial final dispatch/root at all three cadences. A completion
error, strict SDPA-adapter violation, non-finite value, native bit difference,
extra wait, tracked encode allocation or external guard trip rejects that arm.

The DoD is a guarded build of `examples/wan_s2v_cross_cadence_benchmark.cr`,
a pinned-source PyTorch MPS F32 SDPA-adapter oracle generated by
`examples/support/wan_s2v_cross_cadence_reference.py`, then serial guarded runs
with `--reference=PATH --query-rows=257` and `--query-rows=36608`.
The oracle matches the pinned real-stack SDPA adapter's layout and math; it
does not execute Wan's CUDA-only flash-attention routine. Official-source
hashes establish provenance, not an official CUDA or real-weight execution.
Expected signals are a strict finite SDPA-adapter comparison at the unchanged
`2e-5 + 2e-5*abs(reference)` bound, rejected +1/NaN controls, bitwise native
agreement for every warm and measured arm, exactly one terminal wait per arm,
zero inter-operator host waits/activation readbacks/MetalBuffer allocations,
and the expected command-root counts. Three measured balanced orders are
1/2/4, 2/4/1 and 4/1/2 after one warmup per cadence (the cadence-1 adapter
qualification supplies its warmup). Report per-arm completed wall, chain
creation, encode (including scheduling/backpressure),
terminal wait and summed completed GPU-root intervals separately. GPU-root
sums exclude gaps and are not whole-request latency.

Q, two output/reference backings and K/V require 2,270,167,040 bytes at the
full target, plus small diagnostic resources. Stream the oracle fixture into
its backing instead of retaining a second 750 MB host copy. Once the first
native result passes the full SDPA-adapter comparison, preserve it as a native bitwise
anchor and reuse the working output across all arms. Initialize that output
to NaN outside timing before each arm. All attention and comparisons execute
on GPU; only compact diagnostic metrics return after completed diagnostic
commands. Fixture serialization/input construction belongs to oracle setup.

Build admission is 1,536 MiB/180 seconds; each oracle or native run is capped
at 3,072 MiB/120 seconds. The existing sampled guard requires both
`memory_pressure -Q >= 15%` and a cache-aware free-plus-file-backed headroom
estimate above 15% physical RAM, with startup reserving the process budget;
swap growth is capped at 64 MiB. These are sampled guards, not reservations.
Evidence, temporary files and binaries stay under `/opt/mldata_ssd/`.
Rollback is deletion of the probe files/reversion of their isolated commit;
the production cadence remains unchanged. Refresh the evidence after changes
to the kernel, chain bridge, geometry, source oracle, driver or fixture.

### Text cross-attention cadence result, 2026-10-20

The guarded release build and both native shapes completed on Apple M2 Max.
Three balanced measured rounds at Q=36,608 produced the following medians.
The primary wall interval includes chain creation, encoding and the single
terminal wait. Encoding includes `commitAndContinue` backpressure; the columns
are separate medians and need not add exactly.

| Dispatches per command root | Command roots | Completed wall, ms | Encode, ms | Terminal wait, ms | Sum of GPU-root intervals, ms |
| --- | --- | --- | --- | --- | --- |
| 1 | 572 | 393.390 | 349.652 | 43.787 | 392.304 |
| 2 | 286 | 388.900 | 302.123 | 86.906 | 388.502 |
| 4 | 143 | 384.330 | 212.979 | 172.544 | 384.033 |

Four dispatches per root reduced completed wall by **2.303%** relative to one
(1.02357x), winning all three paired rounds against both one and two. Two
reduced wall by 1.141%. The respective wall ranges were 392.493–394.081,
388.174–389.342 and 383.854–386.286 ms. A 39.09% encoding reduction for four
does not imply a matching completion speedup: more outstanding GPU work is
accounted for by the larger terminal wait. These measurements reject submission
cadence as a sufficient explanation for the orders-of-magnitude performance
gap. They do not separate arithmetic, memory traffic or in-root scheduling.

The full comparison covered all 187,432,960 output values. Against the generated
MPS F32 SDPA output, max absolute error was 2.6226044e-6 and max tolerance ratio
0.06367766, with zero strict violations or non-finite values at the unchanged
bound. Every cadence's measured and warm output was bitwise identical to the
native cadence-1 anchor. The +1 and NaN controls were rejected. The 257-query
case covered the partial final range/root (5/3/2 roots), passed the same checks,
and was too noisy to justify a small-shape performance claim.

Each timed arm reported one terminal wait and zero tracked MetalBuffer
allocations. NaN output fills and GPU comparisons were outside timing; only
five diagnostic words were read on the host. No attention activations were
copied to the CPU during the native run. The full native process peaked at a
sampled 2,333,197,584 bytes (2.173 GiB); minimum cache-inclusive headroom was
14,751,088,640 bytes (21.466% of RAM), minimum `memory_pressure -Q` was 79%,
and sampled swap growth was zero. All four guard processes exited zero and
reported no owned processes after cleanup. Sampled peaks are not exact maxima.

Four dispatches per command root is the observed candidate for the next
bounded integration comparison. The production default remains one: no full
block, model, Studio or generation was run, and this isolated gain does not
establish whole-block speed or quality. Kernel geometry and arithmetic are
unchanged. A separate completed-latency block gate remains necessary before
promotion; the existing late-stack numerical discrepancy remains open.

Evidence is stored externally in
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/`:
`cross-cadence-{upstream,build,tail257,full}-v1.invocation.json` records exact
commands, environment overrides and log paths. Their `.log` and `.guard.jsonl`
files preserve raw measurements and guard samples. The reference manifest and
output are under `cross-cadence-reference-v1/`; the native process verifies their
SHA-256 and source pins before use. `cross-cadence-receipt-v1.json` binds probe,
bridge and oracle sources, reused native objects, binary and raw evidence.
The receipt is an evidence index, not an execution or accuracy proof. Recompute
the raw invariants and statistics with this CPU-only DoD command:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/check-cross-cadence-results-v1.py
/opt/homebrew/opt/python@3.11/bin/python3.11 /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/check-cross-cadence-receipt-v1.py --verify
```

The source/oracle/driver/geometry decay conditions above also apply to these
results. Same-lineage source review is correlated; the raw measurements and
negative controls support only the stated isolated scope.

## Additional candidates and their current gates

1. **Static audio K/V.** In
   [the audio injector](../src/ml/video/wan_s2v_block.cr), K/V projections and K
   normalization depend on `local_audio`, not the changing video activation.
   Twelve injection sites at 21 frames, five audio tokens and C=5,120 need
   51,609,600 bytes (49.22 MiB) for normalized K and V. Reuse is conditional on
   proving audio/weight/configuration identity across steps and CFG branches.
   The bounded qualification above passes byte comparisons and mutation controls;
   persistent request integration and cross-branch identity remain unqualified.
   Text K/V caching has the same conditional argument but costs 1.5625 GiB for
   40 layers and two branches while eliminating only 0.11% of counted FLOPs;
   prioritize the smaller cache under the current memory constraints.
2. **Native attention operand reuse.** The existing
   [F32 tiled kernel](../src/ml/video/kernels/attention_tiled.metal) shares each
   key/value matrix operand within an eight-query SIMD group, with zero
   threadgroup storage. Q-fragment hoisting passes the bounded primitive speed
   gate above; 32 KiB staging across two or four groups fails it. Park this
   staging design unless a new premise addresses its measured regression.
   Register reuse across two query tiles also fails the bounded speed gate
   above, despite bitwise identity and no barriers. Park further grouping
   sweeps; a matched hardware-counter profile could discriminate occupancy,
   spills, scheduling and cache effects before another attention rewrite.
   Prioritize the bounded submission-cadence probe next. Even a native primitive
   win needs separate MPS-relative and whole-block gates.
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
That comparison changed dispatch geometry and segmentation together; it does
not settle the separate fixed-geometry submission-cadence hypothesis.

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

## Long-key MPS attribution experiment, 2026-10-20

The current task profiles the already-qualified opt-in MPSGraph SDPA path with
query chunk 128. It does not run a block, model, Studio or video generation.
The production default is unchanged. The source-level default generic kernel
is not the baseline of the existing MPS block measurements; re-testing that
generic route or the already-slower native tiled alternatives would not explain
the remaining MPS cost.

This CAUTION probe keeps B=1, K=36,608, H=40, D=128 and F32 math. Selected query
prefixes of 128/256/512/1,024/4,096/8,192 rows use the same avalanche inputs and
the pinned full-query F32 SDPA reference. A 257-row tail slice exercises padding.
The oracle directly runs F32 PyTorch MPS SDPA with the same core semantics as
the real-stack harness adapter; it hashes but does not execute the official
attention module or adapter. It retains BHQD for fixture export, whereas the
harness adapter returns BQHD. The official BF16 fallback is outside this gate.
The fixed 128-row graph shape, resident K/V, Q-copy/SDPA/output-copy sequence and existing
command-root transitions are preserved. No synchronization or extra root is
inserted to manufacture timing boundaries. Host call intervals include backend
scheduling/backpressure. Completed GPU-root intervals are not kernel timings
and cannot assign operations sharing a root to separate phases.

The competing explanations are device computation/memory work, repeated
layout/materialization work, and host submission/backpressure. A query-count
sweep alone cannot distinguish them: all can scale with the number of chunks.
The discriminating probe is a bounded Instruments Metal System Trace of an
otherwise identical arm, if the installed runtime exposes sufficiently detailed
events. Untraced samples supply timing claims. If the trace cannot separate
internal MPS operations, attribution remains open rather than being inferred
from FLOPs or encode-versus-wait subtraction. Head-major K/V, explicit
QK/softmax/PV and rejected chunk shapes remain parked under their prior gates.

DoD: guarded release build of `examples/wan_s2v_self_attention_profile.cr`,
CPU plan checks, serial guarded selected-query profiles, and an available
bounded Metal trace. Each warm/measured arm must pass the unchanged finite
`2e-5 + 2e-5*abs(reference)` comparison against that pinned F32 oracle,
with rejected +1/NaN controls, exactly one terminal arm wait, zero tracked
MetalBuffer allocations and no inter-operator readback. A profile-off control
checks the effect of host clocks; a padded tail checks unwritten outputs.
Setup fills, fixture loading and diagnostic commands are outside arm timing.
Only compact comparison metrics return after the diagnostic terminal wait.

The strongest failure mode is treating opaque overlapping root spans as an
exclusive GPU-kernel breakdown or extrapolating selected queries to a full
block/video. Both claims are excluded. Build budget is 1,536 MiB/180 seconds;
native probe budget is 5,120 MiB/120 seconds. The first trace collector used
2,048 MiB/90 seconds; its bounded retry used 3,072 MiB/90 seconds and a shorter
four-second recording. Every launcher retains the sampled cache-aware
15%-of-RAM reserve, memory-pressure floor and 64 MiB swap-growth stop. These
guards are not reservations. All temporary and trace files stay under
`/opt/mldata_ssd/`. Rollback is reversion of the isolated probe commit; no
production path changes. Refresh the profile after graph, bridge, geometry,
fixture, device, driver, runtime or queue changes.

### Selected-query results

The release probe ran on Apple M2 Max. Each untraced process used one warmup
and three measured arms. The initial v2 sweep held the graph shape and full
K/V fixed:

| Selected query rows | Median completed wall time, ms |
| ---: | ---: |
| 128 | 29.854 |
| 256 | 66.719 |
| 512 | 128.522 |
| 1,024 | 258.983 |
| 4,096 | 1,048.458 |
| 8,192 | 2,057.606 |

The final v3 source corrects overlap accounting for nested root intervals and
adds timing-boundary metadata. Its GPU operators are unchanged. Fresh v3
qualification measured **1,984.888 ms** for 8,192 query rows, with median host
encoding **2.938 ms** and completed-root envelope **1,984.276 ms**. The eleven
roots had a median total of about 6.54 microseconds of uncovered gaps.
The v2/v3 timing difference is not an optimization result. A final 257-row
tail at query start 36,351 passed at 98.750 ms. Two profile-off controls at
1,024 rows measured 256.505/257.518 ms; a profile-on repeat measured 253.311 ms.
Those observations do not establish an overhead difference beyond run variation.

Across twelve untraced processes, 48 warm/measured validation arms compared
532,193,280 values. Including the two trace-target processes gives 54 arms and
741,908,480 values. Every arm passed the finite unchanged tolerance, rejected
both comparator controls, recorded one terminal arm wait and zero tracked
MetalBuffer allocations. Maximum absolute error was `1.1473894e-6`, maximum
error/tolerance ratio `0.052908488`. Only fifteen metric words returned after
each separate diagnostic chain; output activations stayed on GPU. Opaque MPS
allocations are outside the tracked-allocation claim.

The explicit 8,192-row buffer plan is 1,917.5 MiB, including full K/V and
comparison storage. Sampled process footprints, pressure, reserve and swap
are recorded separately in the evidence below. Such sampling cannot certify
the peak between polls, nor reserve reclaimable cache in advance.

### Bounded Instruments evidence and decision

The initial twelve-second recording exceeded its collector's 2,048 MiB
process-tree cap; the guard stopped it with exit 76. Its native target passed,
but no trace was accepted. The owned process group was checked empty and its
raw trace removed. The four-second retry at 4,096 query rows completed under
the separate 3,072 MiB collector budget; its sampled collector peak was
1,172.15 MiB. Raw trace data was removed after four bounded table exports.
Neither collector nor native runs triggered the 15% reserve/pressure or
64 MiB swap-growth stops. No full generation or full block was launched.

The typed parser checks the four export hashes, exact table schemas, target
PID/process ownership and nanosecond units. Among 183 target Shader Timeline
rows, 157 have the `sdpa_tile_fwd_8x8x8_noEdgeCheck` label. Their interval union
is 834,337,582 ns, versus 834,388,624 ns for all observed shader labels. These
are sampled interval coverages, not exclusive kernel runtimes or an exact
percentage of attention latency. Rows lack the ownership needed to join a
shader to a particular root/graph phase. Fourteen `ndArrayIdentity` rows do
not establish repeated physical K/V transpose cost. Ten `video_linear_f32`
and two `video_norm_mixed` rows remain unexplained: Arena setup compiles those
pipelines, but no dispatch is present in this profiler's source path. They
must not be assigned to an attention phase from their names alone.

The evidence prioritizes work inside the GPU SDPA path over additional host
submission batching. It does not distinguish arithmetic from bandwidth,
prove a layout bottleneck, or extrapolate to full-block/video latency. The
same-lineage source/raw-evidence adversary review is ROBUST for these bounded
claims, and VULNERABLE for those stronger attributions. No production default
or arithmetic changed.

### Precision experiment rationale

The [official Wan attention fallback](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/attention.py)
defaults to BF16 and casts Q/K/V before PyTorch SDPA. Our pinned F32 harness
gate therefore does not establish parity with that official BF16 execution.
BF16/F16 attention is the first simplification experiment: lower storage and
operand precision may reduce traffic or select faster kernels. The bounded
measurements below test latency and storage; quality and the availability of
controlled F32 accumulation on this Metal path remain unmeasured. A factor-of-two
storage change is not a factor-of-two latency claim.

Keep the F32 mode and its existing strict gate. Qualify any separate dtype
mode against a pinned dtype-matched upstream oracle with the same finite
tolerance; do not widen the F32 tolerance or disguise intentional rounding as
an implementation error. Only a later paired, same-input video comparison
can assess the quality effect on detail, motion and lip synchronization.
Fewer denoising steps, attention sparsity and cross-step reuse remain separate
quality-changing candidates; this profiler provides no acceptance evidence
for them. None was implemented or used for generation in this experiment.

### Precision comparison contract, 2026-10-20

This CAUTION experiment compares F32/F16/BF16 MPSGraph SDPA at the same
B=1, K=36,608, H=40, D=128 and query chunk 128. Only selected query rows run;
no full block, model, Studio or video generation is admitted. The current
F32 entrypoints and all production defaults remain unchanged. An opt-in typed
plan uses matching Q/K/V/context storage. The native implementation is checked
against execution of the pinned official Wan fallback with an explicit dtype.
Its result is also compared separately with the unrounded F32 reference.
Both use the existing finite `2e-5 + 2e-5*abs(reference)` threshold. A mode
that fails either comparison is reported with that failure; no threshold is
widened, and dtype-matched parity alone cannot establish F32 accuracy or video
quality. Accumulator precision is not inferred from the storage dtype.

Resident typed-buffer arms measure the attention path after input preparation.
Separate F32-boundary arms include GPU Q/K/V conversion and context widening in
the same command chain, with their additional live buffers counted. Inputs,
geometry and graph chunking stay fixed. F16 represents this avalanche fixture
exactly; BF16 rounds some input values. Explicit typed bytes, reference and
diagnostic storage, and sampled total process footprint are reported separately.
Opaque MPSGraph workspace is not covered by explicit buffer accounting.

DoD: a guarded release precision probe, CPU plan/rejection checks, small typed
capability/strict checks, then serial full-key paired runs and a padded tail.
Each arm must have zero tracked MetalBuffer allocations, one terminal wait,
no activation readback, finite GPU comparison metrics, and rejected +1/NaN
controls. Failed numeric gates may still be timed as rejected candidates; they
cannot change production defaults. The strongest failure mode is mistaking
lower storage bytes for total memory savings or reporting speed while omitting
boundary conversion or an accuracy failure. The comparison distinguishes all
three. Cache-aware reserve and memory-pressure floors remain 15%, swap-growth
limit 64 MiB; all launches have process-footprint and time budgets, all temporary
files stay under `/opt/mldata_ssd/`. Rollback is reversion of this isolated
experiment commit. Evidence expires on dtype implementation, graph, geometry,
fixture, device, driver or runtime changes.

### Reproduction and evidence boundary

The retained command manifests, raw JSON logs, guards, typed trace exports,
source snapshots and receipt are under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/long-key-profile-20261020/`.
`build-v3.invocation.json` records the release command and reused object paths;
the receipt pins the object bytes.
`native-q8192-v3.invocation.json` binds the final untraced run. The v2 snapshot
was reconstructed by reversing only the final metadata correction; it is not
a build-time source capture. The final v3 snapshot matches the current probe source.

The CPU-only `check-profile-receipt-v1.py` rechecks artifact/source hashes,
strict raw validation records, root-interval unions, guard exits and trace
ownership/export pins, and writes `profile-receipt-v1.json`. It is a curated
index to the observed runs, not another GPU run or an independent proof of
upstream execution. Re-run it with Python 3.11 from the isolated checkout;
the guarded GPU commands are retained for deliberate later refresh.

### Precision results, 2026-10-20

The release probe in `examples/wan_s2v_attention_precision_profile.cr` ran on
Apple M2 Max with B=1, Q=8,192, K=36,608, H=40, D=128 and fixed 128-row query
chunks. Each process ran one warmup and three measured arms. Five modes ran
serially in forward order, then reverse order. The oracle executes the pinned
official `wan_modules_attention.py::attention` fallback on PyTorch 2.8.0 MPS
with an explicit input dtype and CPU operator fallback disabled. This is
stronger provenance than the preceding direct-PyTorch F32 fixture, which only
hashed the official source.

| Mode | Forward median, s | Reverse median, s | Attention buffers, MiB | All explicit buffers, MiB | Sampled process peak range, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| F32 resident | 2.072137 | 2.931888 | 1,757.50 | 2,077.50 | 2,145.03–2,145.42 |
| F16 resident | 1.273433 | 1.803110 | 878.75 | 1,198.75 | 1,264.66–1,267.25 |
| BF16 resident | 5.719131 | 5.276199 | 878.75 | 1,198.75 | 1,264.52–1,265.89 |
| F16 with F32 boundary | 1.356436 | 1.420566 | 878.75 | 2,948.75 | 3,014.52–3,016.95 |
| BF16 with F32 boundary | 5.663267 | 5.673443 | 878.75 | 2,948.75 | 3,012.89–3,013.03 |

Attention buffers include Q/K/V, context and three chunk slots. All-explicit
bytes also include two F32 references, diagnostics and, in boundary mode,
resident F32 Q/K/V plus widened context. Thus typed attention storage halves,
but the resident probe's total explicit storage falls by about 42.3%; the
boundary probe's total rises by about 41.9%. These allocations do not count
opaque MPS workspace. Sampled process footprints cover a different accounting
surface and cannot certify between-sample peaks or model-wide savings.

The within-direction F32/F16 resident median ratios are 1.6272 and 1.6260.
F32 itself shifts by 41.5% between directions, so retain both observed medians
instead of treating a single time as stable throughput. The boundary runs
include all Q/K/V narrowing and context widening on GPU, in the arm's command
chain. Cross-process drift prevents subtraction of these medians from isolating
conversion cost. BF16 is slower in both directions despite its smaller buffers;
no cause such as accumulator width or bandwidth saturation is established.

Every full-size validation arm compares 41,943,040 output values against each
reference with the unchanged finite `2e-5 + 2e-5*abs(reference)` test. Native
metrics are identical across the two directions and resident/boundary modes:

| Dtype | Dtype-matched upstream violations | Matched max absolute error | Unrounded F32 violations | F32 max absolute error | Admission |
| --- | ---: | ---: | ---: | ---: | --- |
| F32 | 0 | 1.1473894e-6 | 0 | 1.1473894e-6 | Pass within this primitive |
| F16 | 5,004,258 | 3.0517578e-4 | 90,285 | 1.0111928e-4 | Rejected |
| BF16 | 30,103,661 | 1.953125e-3 | 30,157,253 | 6.5576285e-4 | Rejected |

The discrepancy is not just intentional BF16 input rounding: F16 represents
this fixture exactly, yet also fails against its dtype-matched upstream.
Equivalent high-level SDPA semantics do not guarantee identical low-precision
intermediates, reduction or output rounding across the two backends. The
first divergent internal operation is still unknown. BF16 input conversion
has source-level RNE alignment but no independent buffer-level parity check.
These are numerical findings, not evidence of visible artifacts or harmless
quality loss. Neither dtype is admitted at the existing contract.

The 128-row capability controls, 257-row tail ending at K=36,608 and same-plan
changed-Q controls all executed. F32 passes both gates, including the final
one-row tail. Lower precision remains finite but fails both gates in these
controls, so these runs do not qualify candidate tail/reuse correctness.
Every typed arm has one terminal wait and zero tracked MetalBuffer allocations;
the Q=8,192 arms have eleven existing command roots. Comparison uses a separate
diagnostic chain with its existing one-wait boundary, returns twenty metric
words and rejects +1/NaN controls. Outputs are NaN-poisoned on GPU before
reuse. No native output activation is read back; oracle fixture serialization
is separate setup work. No full block, model or video ran.

CPU-only checks accept fifteen memory plans and reject nine invalid option
combinations before Metal access. Native typed binding checks reject invalid
dtype, short Q storage, aliased K/V, zero extent and overflowing dimensions
before command submission. The unchanged legacy F32 entrypoint is separately
rebuilt with the typed bridge for a bounded Q=128/full-K upstream regression:
zero violations/nonfinite values, maximum error `5.5506825e-7`, one terminal
arm wait and zero tracked MetalBuffer allocations. This is not a full-block
regression.

All main matrix guards completed without a guard abort. Their minimum sampled
cache-aware headroom is 28.73% of RAM, minimum pressure free percentage is 75%,
and observed swap growth is zero. These exceed the unchanged 15% floors;
process caps remain 5,120 MiB and 120 seconds, with a 64 MiB swap-growth stop.
The oracle uses a separate 4,096 MiB cap; release builds use 1,536 MiB and
180 seconds. These are sampled guards, not advance memory reservations.

Decision: keep production F32 and the old tolerance. The typed bridge is an
opt-in experimental entrypoint only. A useful next discriminator is to locate
the first QK/scale/softmax/PV difference on identical operands before designing
a mixed-storage path with explicit F32 arithmetic and context. No such path or
quality-changing approximation is admitted by these results.

Precision command manifests, source/binary/object pins, oracle outputs, raw
native logs, guards and the CPU receipt are retained under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/precision-profile-20261020/`.
`native-main-v1.tasks.json` records the mirrored matrix;
`native-boundaries-v1.tasks.json` records tail/reuse checks;
`legacy-regression-v1.tasks.json` records the legacy F32 rebuild/control.
The CPU-only `check-precision-receipt-v1.py` indexes and rechecks those artifacts;
it is not an independent upstream execution or attestation. Refresh after
graph/bridge/probe, dtype/geometry/input, compiler/device/runtime/queue or guard
policy changes, or artifact loss. Temporary files remain on `/opt/mldata_ssd/`.

### F16 stage-localization contract, 2026-10-21

This bounded CAUTION diagnostic keeps B=1, Q=128, K=36,608, H=40 and D=128.
It runs no complete block, model or generation and changes no production
entrypoint or default. The hypothesis is that an observable QK, scaling,
softmax or PV boundary explains the rejected F16 endpoint. The falsifier is
failure to anchor the diagnostic decomposition to the stock endpoint: in that
case, its first differing stage is scoped to that decomposition, not to hidden
MPSGraph SDPA internals.

Both implementations borrow the same resident, contiguous, zero-offset Torch
MPS storage, with native encoding on the Torch serial queue and command stream.
Storage bindings remain alive through a single terminal metric read. Neither
activation exports nor intermediate host waits are admitted. GPU comparisons
retain the finite `2e-5 + 2e-5*abs(reference)` gate and +1/NaN rejection controls.
The stock pinned Wan fallback is anchored to Torch's actual MPS math endpoint;
its returned probabilities also permit an isolated PV check. Exposing targets
can change compilation, so tapped and untapped endpoint controls are required.
Accumulator precision is not inferred from tensor storage or a matching output.

DoD is a guarded diagnostic build and full-key Q=128 run with byte-identical
shared input bindings, compact GPU metrics, explicit stage/endpoint provenance,
and no intermediate host synchronization. Guards retain the 15% cache-aware
reserve and pressure floors, 64 MiB swap-growth stop, process and time budgets.
Scratch, build products and evidence stay under `/opt/mldata_ssd/`. Rollback is
reverting only this diagnostic commit. Evidence expires on graph/target set,
dtype/input/geometry, compiler, Torch/OS/device or command-stream changes.

#### Observed F16 localization result

The candidate is native Metal/MPSGraph code in
`examples/support/wan_attention_f16_stages_bridge.mm`. Python supplies the
pinned upstream oracle and borrows the same GPU allocations for the candidate;
it is not an alternative inference implementation or a performance benchmark.
The endpoint and split-stage phases use the geometry above, native GPU input
fill and GPU comparisons. Each process read only its final compact metric table. No
production GPU synchronization or arithmetic changed.
Successful process and guard completion establishes a usable diagnostic;
`candidate_comparisons_passed` remains false because the opaque candidate and
materialized-prefix comparisons intentionally retain their measured failures.

The endpoint phase anchors the actual pinned Wan fallback to the probabilities
and output returned by PyTorch 2.8.0's MPS math SDPA operator. Wan's output,
the native connected F16 decomposition's output, its tapped probabilities and
output, and isolated native PV using upstream probabilities all have exact
finite value equality. The comparator checks values after dtype conversion,
not raw bits; signed-zero identity is not established.
Adding the probability target did not change this endpoint on this fixture.
The production opaque F16 SDPA graph instead has 78,311 strict violations among
655,360 output values, with maximum absolute error `9.1552734375e-5` and maximum
tolerance ratio `4.492565155`. Its previous rejection reproduces. Identity,
+1 and NaN comparator controls pass; the latter two reject all output values.

The split-stage phases compare connected native prefixes with separately
materialized Torch operations on the same inputs. Each score/probability tensor contains
187,432,960 values:

| Native boundary versus materialized reference | Strict violations | Maximum absolute error |
| --- | ---: | ---: |
| QK | 0, exactly equal values | 0 |
| Connected QK through F32 scaling | 149,043,779 | 0.002762317657470703 |
| Isolated F32 scaling of the same F16 QK | 0, exactly equal values | 0 |
| Connected prefix through F16 scaled logits | 48,782,605 | 0.00390625 |
| Isolated scaling and F16 rounding | 0, exactly equal values | 0 |
| Connected prefix through softmax | 113 | 6.103515625e-5 |
| Isolated softmax of the same F16 logits | 0, exactly equal values | 0 |

The actual upstream probabilities have the same 113 violations against that
separately materialized reference, while matching the connected native prefix
exactly by value in the endpoint phase. Thus the first differing *observable
prefix* is F32 scaling after QK, and isolated scaling is exact. This supports a graph
context/materialization boundary; retaining wider intermediates or fusion is
a hypothesis, not an identified compiler transformation. It does not expose
the opaque SDPA's internal QK, scale or softmax, and does not identify its first
causal divergence. The isolated explicit native PV applied to actual upstream P
reproduces upstream Y; this does not validate opaque SDPA's internal PV.

A historical `wide-v1` native graph widening QK, softmax and PV to F32 before
casting the output to F16 fails against both upstream (72,236 violations) and opaque SDPA
(5,004). Merely widening the exposed graph does not reproduce either endpoint;
tensor storage and endpoint similarity do not establish accumulator precision.
Production remains F32. No lower-precision quality or speed claim is admitted.

#### Reproduction and safety boundary

The external evidence directory is
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/f16-stages-20261021/`.
It retains `build-v6.invocation.json`, the four final invocation/JSON/log/guard
sets (`endpoints-v3`, `qk-scale-v3`, `round-v3`, `softmax-v3`), the rebuilt
legacy F32 regression, and `stage-receipt-v3.json` with source/binary pins and
sampled guard summaries.
The receipt indexes executed evidence; it is not independent attestation.
The historical `wide-v1` numeric result is pinned to `source-v1/` and its
executed driver, not the final driver. Its generic stage-order label incorrectly
includes intermediate F16 rounding; that label is quarantined. Native kind 12
source defines its arithmetic. The optional wide phase was not replayed.

The monolithic `stages-v2` process was guard-stopped at a sampled footprint of
6,347,671,088 bytes, exceeding its 5 GiB cap. Its emitted JSON is disqualified
as a successful run. The final driver splits QK/scaling, rounding and softmax
into separate processes; the first two skip the upstream SDPA endpoint, and the
softmax phase reuses the consumed QK buffer for rounded logits. This bounds the
diagnostic's simultaneous tensors without changing the candidate graphs or
adding intermediate host waits.

CPU-only plans run without importing Torch:

```sh
python3 -S examples/support/wan_s2v_attention_f16_stages.py --plan-only --phase=endpoints
python3 -S examples/support/wan_s2v_attention_f16_stages.py --plan-only --phase=qk-scale
python3 -S examples/support/wan_s2v_attention_f16_stages.py --plan-only --phase=round
python3 -S examples/support/wan_s2v_attention_f16_stages.py --plan-only --phase=softmax
python3 -S examples/support/wan_s2v_attention_f16_stages.py --plan-only --phase=wide
```

For a GPU phase, use the built library and pinned Torch 2.8.0 environment from
the invocation records. The driver itself does not implement memory monitoring;
the external guard is required. Use a fresh output name for each run:

```sh
python3 examples/support/wan_memory_guard.py \
  --memory-accounting=file-cache --max-footprint-mib=5120 \
  --max-swap-growth-mib=64 --timeout-sec=120 --log="$stage_guard_log" -- \
  "$stage_torch_python" examples/support/wan_s2v_attention_f16_stages.py \
  --phase=endpoints --library="$stage_library" --output="$stage_output"
```

The same guard applies to every phase, retaining the 15% reserve
and pressure floors. Builds use 1,536 MiB / 180 seconds. File-cache headroom is
an estimate of reclaimable memory and polling can miss intervening spikes;
sampled footprints are not proven instantaneous maxima. No full block, model,
Studio or generation ran. The diagnostic alone opts into private GPU buffers
on the Torch stream; every existing public constructor still requires shared
buffers, including a measured private-buffer rejection control.

The production bridge object and the existing Crystal F32 profiler were rebuilt.
The legacy Q=128/full-key control passes with zero strict violations and maximum
absolute error `5.5506825e-7` against its cached upstream fixture; this is a
native regression check, not a fresh full-model upstream execution. The failed
monolithic attempt is retained separately from accepted guarded runs. The
four final processes finish with guard return code zero, minimum sampled
cache-aware headroom 25.01%, minimum pressure-free percentage 78 and unchanged
swap. The short split processes finish between footprint samples: their small
recorded process footprints do not measure peak GPU workspace or establish
that the process stayed below its cap between polls. The
historical wide diagnostic's sampled footprint reached about
4,739 MiB, leaving only about 381 MiB below its cap; its explicit tensor
estimate excludes substantial framework workspace. It is not a low-memory
inference candidate.

The next optimization belongs in the native probe: time a connected F16 graph
that preserves the anchored endpoint, then test tails and changed inputs before
considering wider query coverage. It must beat the current F32 path and pass
the unchanged upstream gate with bounded workspace. The exposed single-stage
cuts are diagnostic observations, not a recipe for inserting materialization
or synchronization into inference. The opaque SDPA's internal cause remains
open. Source and raw-evidence adversary review is ROBUST within the fixed-fixture
observation scope, excluding speed, quality, tails, changed inputs and opaque
internal-cause claims. The review is correlated, not independent attestation.

### Native connected F16 execution contract (2026-10-21)

Time the connected QK -> F32 scale -> F16 rounding -> softmax -> PV graph in
the existing native precision profiler, against the unchanged F32 production
SDPA. Torch is an offline oracle only: cached, source-pinned fixtures are loaded
before timing, and neither Torch nor Python participates in native execution.
Keep B=1, K=36,608, H=40, D=128 and 128-row chunks. Admit only opt-in graph
constructors; preserve all existing defaults and rollback by reverting this
slice. This is a CAUTION performance change: graph compilation may change
rounding, hidden workspace, or reuse behavior even with the same operations.

The first discriminator is native Q=128 against dtype-matched upstream and F32
fixtures with the unchanged finite `2e-5 + 2e-5*abs(reference)` gate. Failed
candidates may be timed descriptively but cannot replace production F32. Then
use paired order-reversed Q=8192 runs, a 257-row tail ending at K and changed-Q
reuse. Prepare inputs, compile plans and validate outside measured arms. Require
zero tracked MetalBuffer allocations, one existing terminal wait per arm, no
activation readback, and no intermediate host waits. A possible second lever is
preparing head-major K/V once on the same GPU chain; any measurement must include
that preparation when the caller starts with BTHD inputs.

DoD: guarded release build, native profiler plan and malformed-binding controls,
the strict fixture comparisons above, paired wall/GPU timing, and the legacy
F32 regression. Inspect source and raw results for benchmark boundary mismatch
before accepting a speed claim. Keep 15% cache-aware reserve and pressure floors,
5 GiB process cap, 64 MiB swap-growth cap and 120-second runs; builds use 1.5 GiB
and 180 seconds. Stop on guard failure and disqualify emitted results from that
process. Polling does not bound instantaneous peaks. No full block, model,
Studio or generation is authorized in this slice. Refresh evidence after source,
OS/GPU/compiler, geometry, fixture, or graph-boundary changes.

### Native connected F16 result: faster locally, rejected at the F32 gate

The opt-in native graph now reproduces the anchored upstream F16 endpoint in
the existing Crystal profiler. No Torch process was launched for these runs;
the binary links Metal/MPSGraph and does not link Torch or Python. Python only
orchestrates the process and samples its memory outside the native timed arm.
Inputs, conversions, layout preparation and attention remain on the GPU.
The graph retains QK -> F32 scale -> F16 rounding -> softmax -> PV and the final
BHTD-to-BTHD context transpose as its target. Tensor dtypes do not establish
matmul accumulator width.

Completed native wall-time medians on Apple M2 Max, Q=8192, K=36608, H=40,
D=128, 128-row chunks, one warmup and three measured arms per process:

| Native arm | Forward median (s) | Reverse median (s) | Device allocation after wait (MiB) |
| --- | ---: | ---: | ---: |
| Existing opaque F32, resident inputs | 1.837295 | 1.850854 | 2081.875 |
| Existing opaque F16, resident inputs | 1.190076 | 1.186548 | 1201.125 |
| Connected F16, BTHD inputs | 0.843248 | 0.835841 | 1915.125 |
| Connected F16, prepare head-major K/V in arm | 0.845239 | 0.844205 | 2630.125 |
| Connected F16, F32 input/output boundary | 0.850226 | 0.850866 | 3665.125 |

These are order-reversed, separate-process observations, not randomized
same-process ABBA or full-block measurements. The connected resident graph is
2.18x/2.21x faster than opaque F32 and 1.41x/1.42x faster than opaque F16 in
these two directions. The F32-boundary arm charges GPU narrowing of all Q/K/V
and widening of the context; its advantage is 2.16x/2.18x versus F32 here.
Neither comparison establishes video throughput or equivalent F32 accuracy.

All connected F16 checks have zero widened-F32 bit differences and zero
nonfinite values against the dtype-matched, source-pinned upstream fixtures:
Q=128, the changed-Q salt-4 reuse check, Q=8192 in both timing orders, and
Q=257 starting at 36351 with a padded final chunk. BTHD and head-major paths
pass those coverage points; the F32-boundary arm passes Q=8192 and the tail.
The fixed K/V inputs are immutable across query chunks; prepared K/V is rebuilt
on every arm, with its preparation charged rather than reused across requests.
The unchanged F32 baseline passes; the new connected F32 constructor also
matches the Q=128 F32 fixture exactly. The separately rebuilt legacy F32 probe
passes with max absolute error 5.5506825e-7 and zero strict violations.

**Promotion is rejected.** Connected F16 differs from upstream F32 in
4,607,259/41,943,040 values outside the unchanged finite
`2e-5 + 2e-5*abs(reference)` gate at Q=8192, with max absolute error
3.1819195e-4 and max tolerance ratio 14.857483. Matching upstream F16 is a
different predicate. Existing opaque F16 also fails its dtype-matched gate
(max error 3.0517578e-4); its smaller F32 drift is not grounds to admit it.
Native exit 3 denotes dtype-matched success plus F32 rejection; exit 2 denotes
dtype-matched rejection. Neither is a memory-guard abort. No tolerance or
production F32 default was changed.

Explicit resident buffers fall from 2077.500 to 1198.750 MiB, but the connected
F16 graph's additional backend allocations leave only about 8% less current
device allocation than F32, rather than half the total memory. This is an
after-wait device counter, not peak workspace or process footprint. Preparing
head-major K/V costs an extra 715 MiB without an observed timing advantage;
park that lever for this geometry. The F32-boundary arm retains both storage
types and is not a memory-saving path.

All 17 precision processes complete without guard abort, covering 57 validation
arms. Each native arm retains one existing terminal wait and zero tracked
MetalBuffer allocations. Validation uses a separate existing diagnostic chain
and reads only 20 metric words; it exports no activations. Six CPU-only plan
preflights and five existing plus six connected malformed-binding controls
reject unsupported combinations, short buffers, aliasing and invalid geometry.
Minimum sampled cache-aware headroom is 25.32%, minimum pressure-free reading
79%, and sampled swap growth zero. All sampled footprints remain under 5 GiB;
short processes can finish between polls, so this does not certify instantaneous
peaks. No owned native profile or guard process remains after completion.

The connected graph uses 129 command roots for 64 chunks versus 11 for opaque
SDPA. Host encoding takes 0.423-0.427 s versus 0.0022-0.0028 s for F32, while
summed root GPU spans are 0.835-0.843 s versus 1.837-1.850 s. Encoding and GPU
work overlap; these figures are not additive and do not isolate kernel time.
Reducing repeated graph submission and its resource preparation is a concrete
next profiling target, but no production gain follows until the unchanged F32
gate and eventual complete-block timing pass. Do not repeat the previously
rejected all-F32 widening route without a new discriminator.

External evidence lives under `/opt/mldata_ssd/cogni-ml/video-scene-20261019/`
`evidence/wan-self-attention/native-connected-20261021/`:
`native-connected-summary-v1.json` aggregates checked raw metrics;
`native-source-receipt-v1.json` and `source-v1/` pin the executed source,
objects and binaries; `native-plan-preflights-v1.json` records CPU controls;
`native-linked-libraries-v1.txt` records the link surface. Per-case
`*.invocation.json`, `*.log` and `*.guard.jsonl` retain exact argv, native
results and sampled safety telemetry. Cached oracle manifests retain the
previous pinned official Wan source and generator hashes. This index supplies
lineage, not an independent attestation. No full block, model or generation ran.

Native reproduction command from this worktree, using the pinned build and
cached fixtures (no oracle process):

```sh
WAN_NATIVE=/opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-block-benchmark/wan_s2v_attention_precision_profile-connected-v1
WAN_FIXTURES=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/precision-profile-20261020
WAN_EVIDENCE=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/native-connected-20261021
/opt/homebrew/opt/python@3.11/bin/python3.11 examples/support/wan_memory_guard.py \
  --memory-accounting=file-cache --max-footprint-mib=5120 \
  --max-swap-growth-mib=64 --timeout-sec=120 --log="$WAN_EVIDENCE/rerun.guard.jsonl" \
  -- "$WAN_NATIVE" --fixture="$WAN_FIXTURES/oracle-f16-q8192" \
  --f32-fixture="$WAN_FIXTURES/oracle-f32-q8192" --dtype=f16 \
  --graph=connected --kv-layout=bthd --mode=resident --query-rows=8192 --samples=3
```

Expected signal: exit 3, all dtype-matched checks true and F32 checks false,
one terminal wait and zero tracked allocations per arm, with no guard abort.
Timing is descriptive and must be refreshed on the contract's decay triggers.

Correlated source/raw-evidence adversary review is ROBUST within this native
fixture and measurement scope. It cross-checked all 17 aggregate rows and 57
arm/validation pairs against raw logs, source snapshot hashes, reverse ordering,
changed-Q reuse and tail masking. It does not qualify production F16 accuracy,
perceptual quality, instantaneous memory peaks or full-block speed.

### Native command encoding investigation contract (2026-10-21)

The connected F16 graph spends about 0.42 s in the host encoding boundary for
64 chunks; those wall seconds may include GPU backpressure rather than CPU
computation. Inputs, graph target/cast order, chunk128 and K36608/H40/D128 stay
fixed. First distinguish executable calls from copy/chain bookkeeping and
on-thread CPU time from elapsed time. Existing tensor-data bindings and compiled
executables are already retained, so their repeated construction is not an
established cause. Do not remove lifetime/overlap guards to chase a proxy.

CAUTION: any scheduling or resource-lifetime change remains opt-in until it
preserves native upstream comparisons and improves the complete measured arm.
Keep production defaults, the unchanged finite 2e-5+2e-5abs F32 gate, and the
existing F16 promotion rejection. Rollback is the preceding atomic commit or
the baseline option. No full block/model/Studio/generation or Torch runtime.

DoD: guarded native release build; Q128 with changed-Q reuse and tail257 ending
at K; order-reversed Q8192 resident runs with one warmup plus three samples;
the legacy F32 regression; source/raw adversary review. Use opt-in timing and
an uninstrumented control to bound observer cost. Require one existing terminal
wait, no intermediate host wait/activation readback, zero tracked buffer
allocations, unchanged precision metrics and comparator controls. Use the
existing file-cache memory guard with 15% reserve/pressure floors, 3 GiB process
cap for resident probes, 64 MiB swap-growth cap and 120 s timeout; builds retain
1.5 GiB/180 s limits. Disqualify guard-aborted runs. Sampled limits cannot bound
instantaneous peaks. Refresh on source, runtime/device/toolchain, graph target,
geometry/inputs, queue/resource lifetime or guard changes.

The initial opt-in CPU-clock probe finds only 8-10 ms of calling-thread CPU
inside about 427 ms encode wall. The default Cogni queue uses `newCommandQueue`;
the installed Metal SDK documents a limit of 64 non-completed buffers, while
this arm records 129 cumulative roots. Test that backpressure hypothesis with
probe-only queues capped at 32/64/128, holding the graph and all GPU arithmetic
fixed. A change in encode wall without a completed-arm gain is a scheduling
diagnostic, not an inference speedup. Queue-cap experiments remain opt-in.

### Native command encoding result (2026-10-21, bounded)

The principal cause of the 0.42 s encode interval is command-queue admission
backpressure, not 0.42 s of CPU computation. The installed Metal SDK documents
64 non-completed command buffers for `newCommandQueue`; a probe-only factory
varies this limit without changing the graph, arithmetic or root sequence.
For resident connected F16, Q8192/K36608/H40/D128, chunk128, the two directions
of a reversed process sequence give these medians (one warmup, three samples):

| Queue capacity | Encode wall A / B (ms) | Completed arm A / B (ms) | Calling-thread CPU A / B (ms) |
| --- | --- | --- | --- |
| 32 | 638.29 / 637.85 | 845.05 / 846.12 | 10.95 / 8.56 |
| 64 | 426.00 / 423.32 | 842.83 / 838.10 | 8.36 / 6.28 |
| 128 | 4.98 / 5.69 | 839.39 / 838.32 | 4.36 / 5.69 |

All three use 129 cumulative command roots. Completed GPU root gaps sum to
about 0.07 ms, while GPU spans sum to about 0.84 s. The timestamps exclude the
delay before the first root and do not isolate individual kernels or the exact
blocking API. The capacity ablation establishes the queue-admission component;
increasing capacity mostly moves waiting to the existing terminal boundary.
There is no established complete-arm speedup to promote. Do not add encode and
GPU time or rank candidates by encode wall alone.

An explicit `--encoder-policy=paused` probe keeps the compute encoder closed
across blits and graph calls. `WanAttentionMPSPlan.copy(..., encoder_paused:
true)` exposes the same checked precondition; the production plan's encode
method and every existing default stay unchanged. At Q8192 this avoids 192
otherwise empty compute-encoder creations (193 to 1, derived from call sites,
not a native creation counter). The command-root count remains 129.
With queue128, median query-copy plus output-copy calling-thread CPU falls from
0.872/0.918 ms to 0.461/0.434 ms in the two orders. Total CPU improvement is
variable, and complete-arm times are 840.01/840.79 ms versus eager
839.39/838.32 ms. Keep this small scheduling experiment opt-in. It does not
qualify a production or video-throughput improvement.

`--profile-host=true` adds preallocated stage accumulators, Darwin calling-thread
and process CPU clocks, and a root timestamp summary read after the existing
wait. It creates no per-chunk records and submits no GPU work. CPU values
include observer cost; process CPU covers all process threads. The observer is
off by default. The on/off reversed controls show later GPU spans drifting to
0.90-0.96 s, so these wall measurements cannot establish observer overhead or
attribute the drift to a particular cause. The explicit capacity controls above
remain close to 0.84 s in both directions.

Twenty final-source precision processes complete normally, covering 73
arm/validation pairs. All F16 arms match the dtype-matched pinned upstream
fixtures bit-for-bit after widening, including changed-Q reuse and a tail257
ending at K. Queue and encoder policy preserve the existing Q8192 F32 rejection
exactly: 4,607,259 violations, max error 3.1819195e-4, ratio 14.857483. The
unchanged finite `2e-5 + 2e-5*abs(reference)` gate still rejects F16 promotion.
The paused opaque F32 control passes both gates; the separately rebuilt legacy
F32 probe passes with max error 5.5506825e-7 and zero strict violations.
Positive/+1 and NaN comparator controls pass. Five CPU-only CLI controls and
six paused-state/alignment/alias rejection controls pass. Existing buffer,
lifetime and native binding guards remain in place.

Every measured arm retains one existing terminal wait and zero tracked
MetalBuffer allocations; validation retains its separate existing wait and
reads only 20 metric words. No activation readback or new synchronization was
added. All final runs use the 15% reserve/pressure policy, 3 GiB process cap,
64 MiB swap-growth cap and 120 s timeout. Minimum sampled cache-aware headroom
is 19.60%, pressure free is at least 79%, and sampled swap growth is zero.
No guard abort occurs. Polling does not certify instantaneous peaks. No full
block, model, Studio, generation or Torch runtime ran.

Evidence is under the same external SSD base, in
`evidence/wan-self-attention/native-encode-20261021/`:
`native-encode-summary-v1.json`, `native-encode-source-receipt-v1.json`,
`source-v1/`, `native-encode-preflights-v1.json`, and per-case invocations, logs
and guards. `native-encode-raw-receipt-v1.json` pins 80 raw files and indices;
its SHA256 is `ba310dd90779dc88d9aff808edfc22394d7c625183bf2dee94e79199078505f2`.
`paired-matrix-v1.json` and `run-encode-matrix-v1.py` preserve exact
serial order and checks; cached oracle manifests retain upstream provenance.
Build invocations include the probe-only queue object, compiled with
`clang++ -std=c++17 -fobjc-arc -fPIC -c
examples/support/wan_attention_profile_queue.mm`. It is linked only into the
precision profiler, not the model runtime. The executed profiler is
`build/wan-block-benchmark/wan_s2v_attention_precision_profile-encode-v3`
under the external base. Use the preceding guarded native reproduction with
this binary, 3072 MiB instead of 5120 MiB, and explicit
`--profile-host=true --encoder-policy=eager --queue-capacity=64` or `128`.
Expected exit remains 3 for connected F16, with every dtype-matched comparison
true and every F32 comparison false.

The next useful discriminator is GPU execution inside the native graph, not
another attempt to eliminate hundreds of milliseconds of presumed CPU work.
Refresh these observations on the contract's source, target, input, runtime,
queue, geometry, fixture and guard changes. This receipt is an evidence index,
not independent attestation or production qualification.

Correlated source/raw-evidence adversary review is ROBUST for the bounded
queue-backpressure attribution, unchanged explicit synchronization and fixture
checks, and the rejection of a stable complete-arm speedup. It independently
recounts the 20 cases, 73 arm/validation pairs and 162 guard samples. The exact
internal blocking API, observer overhead, instantaneous peaks and production
speed remain unqualified. The raw digest manifest addresses the review's
artifact-mutation gap; verify it before reusing these receipts.

### Native query-geometry investigation contract (2026-10-21)

The next CAUTION probe holds full K36608/H40/D128, GPU-filled dyadic inputs,
output layout and the existing terminal boundaries fixed. Test graph query
chunks 128 and 256 in the native precision profiler, including opaque and
connected F32 where the guard permits. This is a graph-shape experiment, not a
claim to reproduce a CUDA kernel's internal tile. First measure the already
available connected F32 graph at chunk128; its Q128 fixture passed previously,
but Q8192 performance and larger graph shapes remain unqualified. An apparent
F16 gain still cannot pass the existing F32 promotion rejection by assertion.

The portable motivation is K/V operand reuse in DeepInfer's pinned FlashMLA
dense prefill path. Larger outer graph chunks may instead leave its internal
kernel unchanged, increase workspace, select a different rounding path or
regress completion. The falsifier is unchanged strict upstream comparisons
plus reversed-order completed-arm timing. Reject a candidate on any numerical,
completion, NaN-poison coverage, lifetime or memory-guard failure; retain the
production chunk/default and use the baseline option as rollback.

DoD: guarded native release build; CPU-only argument/plan controls; Q128
changed-query reuse, tail257 ending at K, and Q8192 one-warmup/three-sample
forward/reverse runs for viable shapes. The cached pinned upstream fixtures
remain chunk128 reference computations: query rows are independent with full
K/V, so candidate graph chunking is deliberately separate from oracle chunking.
No PyTorch process, block/model/Studio/full generation or new host wait is
admitted. Require the unchanged finite 2e-5+2e-5abs gate, comparator controls,
one existing terminal arm wait, zero tracked MetalBuffer allocations and no
activation readback. Keep 15% reserve/pressure floors, 3 GiB process cap,
64 MiB swap-growth cap and 120 s timeout (builds 1.5 GiB/180 s). Guard-aborted
cases supply no performance evidence. Refresh on graph/source, geometry,
inputs/fixture, device/runtime/toolchain, queue/lifetime or guard changes.

### Query-geometry results and DeepInfer transfer limits (2026-10-21)

The native release profiler now accepts opt-in `--chunk-rows=128|256`.
The default remains 128, and its metadata distinguishes the candidate graph
chunk from the pinned upstream fixture chunk of 128. No bridge arithmetic,
model runtime or production default changed. Eight CPU-only controls check
the unchanged default, both memory plans and five unsupported chunk values.

Q8192/K36608/H40/D128 uses the same resident dyadic inputs, BTHD layout,
queue128, eager encoder policy and host observer in each process. Each timing
is the median of three completed arms after one warmup. The reversed process
sequence repeats each comparison; later GPU spans drift, so the two sequences
are reported separately rather than pooled.

| Graph/dtype | Chunk128 forward / reverse (s) | Chunk256 forward / reverse (s) | Chunk256 time change |
| --- | --- | --- | --- |
| Opaque F32 SDPA | 2.01344 / 2.41175 | 2.36242 / 2.78515 | +17.33% / +15.48% |
| Connected F16, diagnostic only | 0.86381 / 0.97708 | 0.85038 / 0.92008 | -1.56% / -5.83% |

Opaque F32 roots fall from 11 to 6, yet completion regresses in both orders.
Its upstream maximum error is 1.1473894e-6, ratio 0.052908488, zero strict
violations for both chunks. Fewer command roots are not a GPU speedup proxy.
Connected F16 roots fall from 129 to 65 and post-wait device-current allocation
rises by 754,712,576 bytes (719.75 MiB). It is still bit-exact against the
dtype-matched upstream output and still fails the unchanged F32 gate:
4,607,259 violations, max error 3.1819195e-4, ratio 14.857483. The small F16
timing differences are descriptive: reverse sample ranges overlap, per-chunk
host instrumentation scales with chunk count, and precision already rejects
promotion. Keep chunk128 and all production defaults; no admissible complete-arm
speedup is established.

Connected F32 chunk128 passes Q128 changed-input reuse and the final tail257
bit-for-bit against upstream. At Q8192, both process orders exceed the 3 GiB
process-tree footprint cap before producing an arm. Guards stop them at
sampled footprints 3,761,439,872 and 3,764,225,152 bytes, about 3.50 GiB.
Neither supplies numerical or timing evidence at Q8192. Connected F32 chunk256
was not attempted after chunk128 exceeded the budget. Larger intermediate
workspace is not qualified by the explicit Arena count, and these stops do
not by themselves establish a leak.

The final-source matrix contains 18 normally completed processes, two guard
stops and 47 arm/validation pairs. It covers Q128 changed-query reuse, tail257
ending at K, and Q8192 for the viable shapes. All admitted arms retain one
terminal wait and zero tracked MetalBuffer allocations. Validation retains its
existing separate wait, reads 20 metric words without activation readback,
and rejects both +1 and NaN controls. Across 156 samples, minimum cache-aware
headroom is 26.7176%, pressure free is at least 75%, and swap growth is zero.
Polling can overshoot the process cap and miss instantaneous peaks. File-cache
headroom is an estimate of reclaimable capacity. No Torch runtime, complete
block, model, Studio or generation ran.

The read-only DeepInfer audit pins commit
[`9bd9cf2`](https://github.com/deepseek-ai/DeepInfer/commit/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)
and its FlashMLA gitlink
[`2e5429f`](https://github.com/deepseek-ai/FlashMLA/tree/2e5429fc5653bab6e081f09477126f731882a6a9).
Its dense CUDA kernel uses an internal 256Q/128K/128D tile and FP32 QK/PV
accumulators ([types and tile](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cuh#L43-L56)).
It streams online softmax state and interleaves QK/PV stages
([mainloop](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/dense/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp#L397-L711)).
Streaming K/V and online softmax already exist in the local native tiled path;
prior shared-K/V staging and register grouping regressed. The remaining
transfer hypothesis is K/V load/compute overlap, conditional on a bounded GPU
profile showing corresponding stalls. CUDA TMA, tensor-memory and warp
pipelines need a different Metal mapping. An outer MPSGraph chunk is not this
kernel's internal tile, and the chunk sweep does not test that CUDA schedule.
DeepSeek-specific MLA/sparse attention/cache compression changes Wan's dense
noncausal operation and is excluded from this strict-parity optimization.

The next discriminator is a bounded native GPU profile separating attention
kernel execution from copies and exposing memory/compute stalls before another
staging rewrite. Preserve the F32 gate. The previously bit-exact Q-fragment
hoist remains a local native-kernel candidate, but its 17.336% Q256 improvement
was against a native baseline that was 3.4x slower than MPS in a separate cohort;
it does not establish a replacement or whole-block gain.

Evidence is under the external SSD base in
`evidence/wan-self-attention/query-geometry-20261021/`:
`query-geometry-summary-v1.json`, `query-geometry-source-receipt-v1.json`,
`geometry-preflights-v1.json`, `source-v1/`, and per-case invocations/logs/guards.
`run-geometry-matrix-v1.py` and the two matrix configs preserve exact serial
order; `summarize-geometry-v1.py` recomputes aggregate counts and comparisons.
The source receipt checks eight cached upstream fixture manifests and output
hashes, unchanged native object lineage, and the executed release binary
`build/wan-block-benchmark/wan_s2v_attention_precision_profile-geometry-v1`.
`deepinfer/` retains six official source snapshots, hashes and the audit.
`query-geometry-raw-receipt-v1.json` pins 92 raw/index/source files; SHA256
`69d25a529f0cbf04cccf921d07a33c357defdb63cdc3a1e9745887f955e61d40`.
This is a curated evidence index, not independent attestation. Refresh after
the contract's source/target, input/fixture, runtime, queue, geometry, guard
or resource-lifetime changes, or artifact loss.

Correlated source/raw-evidence adversary review is ROBUST for these bounded
fixture observations, source/control invariants, memory exclusions and the
rejection of an accuracy-admissible speedup. It independently recomputes the
18/2/47 process/guard/validation counts and verifies all 92 receipt file hashes.
It does not qualify causality for the small F16 timing differences, production
performance, quality or the internal CUDA schedule on Metal.

## Long-key GPU counter profile, 2026-10-21

A bounded PID-attached Instruments capture now exposes two SDPA timeline
intervals in the unchanged opaque F32 MPSGraph path on M2 Max. The accepted
case is one Q128 chunk, full K=36,608, H=40, D=128. Shader rows are joined by
PID and unique temporal containment to one compute encoder, not by a direct
shader-to-command key; they may be portions of one dispatch. The exported
command-buffer envelope (61.154542 ms) matches the native root duration within
43 ns. Absolute native and trace-relative clock epochs are not subtracted.
No GPU operator, command root, synchronization, tensor readback or production
default changed. No complete block, model, Studio or generation ran.

Counters are GPU-wide. Excluding overlap with exported foreign-process Active
intervals leaves 7.912 ms of the first SDPA interval; actual counter intervals
cover only 3.600 ms (180 samples, 45.50% of selected time). Over that sampled
coverage, Buffer Read Limiter averages 97.6404%, Buffer Load Utilization
37.6004%, ALU Limiter 38.2028%, and Compute Occupancy 12.4548%. Means are weighted
by sample overlap, without extrapolation or zero filling. The second SDPA
interval has much lower read limiter/utilization, 33.1436%/11.9973%, over
2.683 ms of coverage (55.95%). This is contrary evidence to generalizing the
first signature across both observed intervals.

The admitted observation is sampled buffer-read subsystem pressure; K/V-specific
causality, physical stall fractions/load latency, DRAM saturation, repeatability
and load/compute-overlap benefit remain open. The exact exported utilization
label is Buffer Load Utilization; the installed Instruments mapping aliases it
to the UI's Buffer Read Utilization. Zero observed bandwidth/LLC/MMU counters
cannot establish absent traffic. The 29.252 ms target gap before SDPA is also
unattributed; foreign Active work overlaps 5.160 ms of target uncovered ranges.
Neither the remaining gap nor the counter-filtered slices establish GPU
idleness or K/V residency waits.

The existing binary/six selected source hashes and cached upstream fixture pins
match. All 20 arm/validation pairs across seven native runs pass the unchanged
finite `2e-5 + 2e-5*abs(reference)` gate and +1/NaN controls: zero violations or
nonfinites, maximum absolute error 1.1473894e-6. Every arm retains one existing
terminal wait and zero tracked MetalBuffer allocations; diagnostics retain their
existing separate wait and metric-only readback. Inputs are fixed-salt,
GPU-generated avalanche data; there is no new Torch runtime. This qualifies
synthetic primitive parity, not real model activations or video quality.

Native and collector groups have separate 3,072 MiB caps, 120/90 second
deadlines, 15% cache-aware headroom and pressure floors, and 64 MiB swap-growth
stops. Admission charges both budgets together. Minimum sampled headroom is
27.3668%/27.2594%, pressure free at least 75%, swap growth zero; all owned groups
end with zero live processes. File cache is only a reclaimability estimate,
and sampled guards cannot certify instantaneous peaks.

Three earlier collector attempts timed out; their native parity checks passed,
but failed/partial counter exports are excluded. The final single-chunk capture
exports seven complete tables and 276,320 counter rows. Only its complete
counter-interval table supplies counter values. The accepted raw trace was
removed after export/owned-writer termination; older helpers also removed
failed raw traces. Two malformed exports remain quarantined as `.partial`.
Untraced Q8192 before/after medians drift from 3.0511 to 2.4066 s; Q128 has only
a post-capture control. No trace-overhead, throughput or speedup claim follows.

Evidence base: `/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/gpu-stalls-20261021/`.
`GPU_STALL_PROFILE.md` records the scope and next falsifier;
`gpu-stalls-summary-v1.json`, `stalls-typed-analysis-v3.json` and
`independent-native-guard-check-v1.json` retain typed results. Exact guarded
native/collector invocations, templates, sanitized exports and the Apple
source/SDK audit are retained. `gpu-stalls-raw-receipt-v1.json` indexes hashes;
it is a curated index, not independent attestation. Receipt SHA256:
`36f34a4ed6a2b71546de0ffdcab8749c95783bf2a76840e1c7bdb971c4a65ea4`.

DoD commands, with `E` set to the evidence base above:
`python3 -B "$E/summarize-stalls-v1.py"`, then
`python3 -B "$E/summarize-stalls-v1.py" --verify`.
Expected: the original strict native/control runs pass; CPU recomputation
finds 20 pairs, checks scoped source/binary/fixture pins and sampled guards,
validates receipt hashes and seven complete XML exports. Correlated adversary
review is ROBUST for these bounded observations and VULNERABLE for stronger
K/V causality or throughput claims.

Next falsifier: a bounded native K/V load/compute-overlap A/B, preserving strict
F32 parity and existing waits, must win untraced complete-attention time with
staging charged. Prior staging regressions remain contrary evidence; counter
improvement alone is insufficient. No production route is promoted here.
Refresh on source/binary, input/fixture, dtype/geometry, device/runtime,
queue/resource, collector/guard changes or artifact loss. Rollback is this
documentary commit; the runtime path is unchanged.

## Bounded K/V fragment lookahead, 2026-10-21

The CAUTION experiment is probe-only F32 software pipelining: one SIMD
group still owns one eight-query tile, caches Q, and uses the same ascending
QK, online-softmax and PV arithmetic as the Q-cache baseline. Load one next
K/V fragment before the current matrix multiply, without shared staging,
another query accumulator, a barrier, a buffer or a host wait. The strongest
failure modes are changed arithmetic, invalid tail indexing, compiler sinking
of loads, and extra live state offsetting any latency benefit. Source ordering
alone cannot establish physical overlap. The opaque MPS profile only motivates
this test; it does not attribute native-kernel stalls to K/V.

DoD: build a uniquely named release benchmark from pinned current sources,
then run `--comparison=q-cache-vs-prefetch-kv --query-rows=256 --query-start=0`
with the pinned full-K upstream fixture. One AB warmup and three AB/BA/AB pairs
must preserve finite `2e-5+2e-5*abs(reference)` upstream tolerance, native-pair
bit identity, NaN output sentinels, +1/NaN negative controls, zero tracked
arm-time MetalBuffer allocations and one existing terminal wait per arm.
Keep the 320-group segment limit; validation retains its existing separate
wait and reads metrics only. Test Q257/start36351 for the partial query tile;
its timings are diagnostic. No Torch runtime, profiler, block, model, Studio
or full generation is admitted.

All three primary pairs and their median must improve complete arm wall time
before a direct `mps-vs-prefetch-kv` cohort is useful. The latter is required
before any MPS-relative claim. A failed speed gate rejects this formulation;
do not sweep more staging variants without a new premise. Build/runtime
guards retain 15% cache-aware headroom and pressure floors, 64 MiB swap-growth
stops, 1.5/3 GiB owned-process caps and 180/120 second deadlines. Run heavy
jobs serially; account for setup and private workspace separately. Rollback
is removal of the new probe modes/kernel; production routes stay unchanged.

### Unprofiled result and promotion boundary

At parent `9205352e`, the new release benchmark ran on Apple M2 Max with full
K=36,608, H=40, D=128. Each cohort has one AB warmup and three measured
AB/BA/AB pairs in one process. The measured interval includes command encoding,
submission and the existing terminal wait; setup fill and diagnostics are
outside it. These are selected-query attention timings, not block or video
throughput. Inputs are fixed-salt GPU avalanche data, not model activations.

| Cohort | Baseline median | Candidate median | Interpretation |
| --- | ---: | ---: | --- |
| Q256/start0, Q-cache vs fragment lookahead | 174.420042 ms | 131.391333 ms | 24.6696% lower wall time, 3/3 paired wins |
| Q257/start36351, same native pair | 214.309125 ms | 164.880125 ms | Partial query-tile parity diagnostic only |
| Q256/start0, MPS chunk128 vs lookahead | 65.552417 ms | 127.231875 ms | Candidate 1.9409x slower, 0/3 paired wins |

All 36 checks across 12 validation batches pass the unchanged finite upstream
gate. The native pairs are bit-identical in all eight Q256/Q257 batches.
Maximum upstream absolute error is 8.493662e-7 and maximum tolerance ratio
0.039907727 (limit 1). Each of three negative controls rejects exactly one +1
and one NaN: two bit differences, two violations, one nonfinite output.
All 24 arms, including warmups, retain one terminal wait and zero tracked
arm-time MetalBuffer allocations. Each validation batch and negative control
retains its existing separate wait and reads metrics only. No production
source/default, dtype, reduction order or synchronization boundary changed.

The mapping remains one SIMD group/eight-query tile, 32 threads, and at most
320 groups per segment: four segments at Q256, five at Q257. The candidate
reports SIMD width 32, maximum supported threads 448 and zero static
threadgroup bytes. These properties do not measure register use or occupancy.
K=36,608 is divisible by 32: Q257 covers a partial query tile, not a partial
key tile. The explicit buffer plans are 1,520,435,260 bytes (primary),
1,520,517,180 (tail) and 1,528,299,580 (MPS control); these include both arms
and shared inputs/diagnostics, but exclude opaque MPS private workspace.

Build and runtime guards pass with zero sampled swap growth and zero final
owned live processes. Runtime peak sampled group footprints are below 1.551 GB,
minimum cache-aware headroom is 20,517,928,960 bytes (29.857516% of physical
memory) and pressure free is at least 78%. Fifteen-percent floors, caps and
deadlines remain enforced. File-cache
reclaimability is an estimate; sampled guards do not certify instantaneous
peaks. Only one bounded benchmark process runs at a time.

The scoped source/raw-record adversary verdict is ROBUST for the native
selected-query improvement and parity. Physical load/compute overlap and the
cause of the gain remain unproved: source lookahead can be rescheduled by the
compiler. The direct MPS result rejects replacement of the practical baseline.
Keep the kernel probe-only; no full block/model/Studio/generation ran.

Evidence base: `/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/kv-prefetch-20261021/`.
`source-preflight-v1.json`, `build-launch-v1.json` and per-case invocation/log/
guard files retain exact lineage. The old register-K/V source pin for the MPS
bridge was stale; preflight rejected it before launch. The linked connected
bridge object/source instead match the later query-geometry receipt; Crystal
sources were rebuilt at their current pins. The full upstream fixture manifest
and 749,731,840-byte output are hashed before loading the selected rows.

CPU-only DoD, with `E` set to the evidence base above:
`/opt/homebrew/opt/python@3.11/bin/python3.11 -B "$E/summarize-prefetch-v1.py" --verify`.
Observed exit 0: raw receipt, pinned hashes, run records, medians and summary
recompute exactly. The verifier derives medians from measured arms rather than
trusting decision records; it checks source/object/binary/fixture pins, strict
results, negative controls, waits, tracked allocations and sampled guards.
`crystal tool format --check examples/wan_s2v_bounded_attention_benchmark.cr`
and `git diff --check` also pass. SHA256 pins:

- Verifier: `09b7caaa25c4d6e1aeac220c9a0e9b5764290f1229d054cfba0240dd282d6093`.
- Raw receipt: `9b96c43dfa00dd386622997709734ee48da53a77e10ab4eafe7fe0268f5ed448`.
- Summary: `0cd9a221136ad0bf5519f5da8a0460f095638ac5687e889254d75187fa67d274`.

These local hashes identify retained evidence; they are not independent
attestation. The reproducible check remains conditional on the recorded
fixture and source provenance and on availability of the external artifacts.

### Relevant public fal research, checked 2026-10-21

The [FLUX 3 Max landing page](https://fal.ai/flux-3-max) reports an average
12.45 s for a 10 s 16:9 clip at 720p/768p, versus 126.44 s for its base model.
It names post-training, inference optimization and systems engineering, but
does not disclose the hardware/count, step count or precision for that timing.
This is a vendor result for another checkpoint/workload, not a Wan/Metal target
derived from comparable hardware or a same-trajectory experiment.

The [H3 Max infrastructure article](https://blog.fal.ai/h3-max-built-with-fal-inference-and-training/)
explicitly attributes model work to fewer sampling steps and cheaper steps;
its request-execution charts use GB200s. It separates warm execution from
queueing/cold starts. The [Ideogram v4 technical article](https://blog.fal.ai/serving-sub-second-ideogram-v4-without-quality-loss/)
describes FP4, fused GEMM epilogues, quantization-aware distillation, guidance
distillation from two branches to one, and few-step distribution matching.
Those are documented Ideogram methods, not established FLUX 3 Max internals.
The [epilogue tutorial](https://blog.fal.ai/crafting-efficient-kernels-with-epilogue-fusion/)
contains BF16 CUTLASS examples and gate/up interleaving during weight packing.
Wan's FFN uses GELU, so its concrete pattern must be checked before adaptation.

The [Ulysses overlap experiment](https://blog.fal.ai/ulysses-unbound-experiments-in-communication-computation-overlap/)
reports about 23–25% lower pre-attention latency but only about 3% end-to-end
gain on its multi-B200 workload. Its communication overlap is a different
transport boundary from single-GPU fragment lookahead. [FlashPack](https://github.com/fal-ai/flashpack)
is public loading code; improving cold weight loading does not imply faster
warm denoising. CUDA/NVFP4/NVLink implementations cannot be copied into Metal.

Next controlled discriminator: measure the strongest attention path in the
bounded full block, including packing and surrounding operations, before
choosing another kernel rewrite. Fusion/packing must preserve the current
strict gate and include their preparation cost. A few-step/distilled checkpoint
or guidance change is a separate quality-qualified branch, not an exemption
from the existing intermediate tolerance. No portable fal speed multiplier is
claimed. Refresh numerical/performance evidence after scoped source/binary,
input/fixture, geometry/dtype, device/toolchain/runtime, queue/resource or guard
changes, or artifact loss; refresh public research when its sources change.

## Full-block prepared-K/V MPS attribution contract, 2026-10-21

Measure one real-weight block-0 forward at B=1, N=36,608, C=5,120,
H=40, D=128, FFN=13,824 and text K=512. This CAUTION diagnostic uses
BF16-resident parameters, F32 operator math, lifetime-aliased scratch and
MPSGraph SDPA self-attention with 128-query chunks and head-major K/V.
Text cross-attention remains the existing native tiled path with 320 groups
per segment. Weight loading, plan creation and synthetic input fill are setup;
Q/K/V projections, norms, RoPE, a fresh K/V layout transform on every encode,
all attention chunks, output copies, cross-attention and FFN are charged to the
complete arm. No full model, audio injector, Studio or video generation runs.

The smallest probe extension is `--comparison=profile-replay --prepared-kv
--candidate-self-chunk=128 --reuse-block-scratch --phase-roots --pairs=1`.
Both A/A arms select the same prepared-K/V path. Only B collects the existing
asynchronous root intervals; A supplies unprofiled complete-arm controls.
No new model-path wait, command boundary, activation readback or MetalBuffer
allocation is admitted. Existing terminal diagnostics read compact metrics
only. Root phase unions are overlapping diagnostics, not additive kernel costs.
Pair differences cannot establish an algorithmic speedup because both arms
compute the same graph and B adds observation overhead.

DoD: build a fresh release probe from pinned sources/objects; qualify the
prepared-K/V path on the pinned upstream real-width short-block fixture and
retain the full-key attention oracle qualification. Under the full-size run,
A/A outputs must be bit-identical and pass the unchanged finite
`2e-5+2e-5*abs(reference)` gate. A +1 and NaN control must reject exactly those
two outputs in the existing single control wait. All arms retain one terminal
wait and zero tracked arm-time MetalBuffer allocations. Phase root counts must
match the unprofiled graph. Short upstream fixtures, full-key attention checks
and full-block self-parity remain distinct evidence scopes; this does not
create a full-spatial upstream block oracle.

Strongest failure modes: preparation omitted from the timed interval, stale
packed K/V, an unintended explicit QK/softmax/PV backend, scratch alias damage,
or overlapping root intervals misattributed as kernel costs. Check live
source/binary routing, upstream intermediate parity, exact A/A comparison,
per-encode preparation metadata and the validated root-profile summary. The
current probe validates individual intervals internally but exports their
summary only. Choose a fusion target
only when its measured parent phase and source-level memory/dispatch pattern
support it; isolate the candidate before any speedup claim.

Builds and GPU probes run serially. The existing cache-aware headroom estimate
and pressure query retain 15% floors, with admission charging the whole owned
process cap, a 64 MiB swap-growth stop and monotonic deadlines. Compute the
new memory plan before launch, including prepared K/V and the heuristic opaque
MPS reserve; reject insufficient headroom without shrinking accounting to
force admission. Stop after the first valid attribution unless variance or a
failed control needs a discriminating follow-up. Evidence and build outputs
stay under `/opt/mldata_ssd/cogni-ml/video-scene-20261019/`; rollback removes
the probe/spec opt-ins. Production defaults and numerical tolerances remain.

### Full-block observations and selected fusion experiment

At parent `9de71191`, fresh release builds used seven freshly compiled native
bridges, Crystal 1.22.0/LLVM 23.1.3, Apple clang 21.0.0 and macOS 26.6.2 on
M2 Max with 64 GiB physical memory. The prepared-K/V plan estimates
11,194,526,228 B including its heuristic 1 GiB opaque reserve. Its single
shared prepared pair occupies 1,499,463,680 B and is refreshed inside every arm.
The new probe flag selects this path in both profile-replay arms; it changes
no production default, arithmetic kernel or model-path synchronization.

One warmup per arm and one measured A/A pair completed. Wall times were
18.456799/20.191059 s for warm A/B and **22.392418/22.336865 s** for measured
A/B. Measured A/B summed GPU root durations were 22.349545/22.274099 s.
Host encode times were 19.783769/19.761098 s; these include submission
backpressure and are not isolated CPU compute costs. The warmup drift and
same-configuration comparison establish neither a stable latency distribution
nor a speedup. All preparation, including K/V layout conversion, was charged.

Measured B's 624 roots have a 22.274733 s envelope, 22.274099 s interval union
and zero observed overlap. Warm B has the same broad ranking. These are
coarse source-bound phases, not individual kernel times:

| Phase | Measured union, s | Share of reported root union |
| --- | ---: | ---: |
| Self preparation: norm, modulation, projections, Q/K norm and RoPE | 1.263346 | 5.67% |
| Fresh K/V packing, self-attention, output projection and residual | 17.209883 | 77.26% |
| Text cross-attention, projections and residual | 1.642163 | 7.37% |
| FFN | 2.158707 | 9.69% |

The phase shares describe this no-overlap observation only. In general, root
and phase intervals may overlap and their unions must not be added. The large
self-attention phase contains 286 SDPA query chunks as well as surrounding
work; the profile does not isolate transpose cost or imply that its entire
17.21 s is fusible. Individual root intervals are not exported by this probe.

Both warm and measured block comparisons are bit-exact across all
187,432,960 output values: zero tolerance violations and nonfinite values.
The finite +1/NaN control rejects exactly two differences, two tolerance
violations and one nonfinite value in its existing single wait. All four arms
retain 624 command segments, one terminal wait, zero tracked arm-time
MetalBuffer allocations, zero inter-operator waits and zero activation
readbacks. No full model, video generation or profiler runs in this experiment.

The separate retained upstream scopes remain explicit:

- Prepared chunk1 and padded-tail chunk3 each pass 15 examples with zero
  failures/errors/pending. Each reports all 24 strict real-width short-block
  checkpoint comparisons passing, with bit-exact default/repeat/chained replay
  and zero tracked encode allocations. The parent fixture hash is
  `fb7943d637bc8a6853707cd2ed284595e6744c4368083c36993fd1dd26d78da9`;
  first-batch projection changes no weights and executes no fresh upstream run.
- The rebuilt full-Q/K36608 attention probe passes eight full-output upstream
  checks (two warm and six measured), max absolute error `1.8328428e-6`, max
  tolerance ratio `0.08037424`, and bit-exact agreement between K/V layouts.
  The unchanged finite strict gate remains `2e-5+2e-5*abs(reference)`.
  Its AB/BA/AB complete-attention savings are +0.420292, +0.972046 and
  -1.461543 s. One losing pair rejects layout promotion despite a lower median.
- Full-size block A/A is a distinct same-path check. There is no full-spatial
  upstream block oracle, end-to-end quality claim or evidence of physical
  load/compute overlap. Python performs orchestration and CPU-only evidence
  checks; no PyTorch runtime participates in the native measured path.

The first full-block attempt rejects admission before spawning: cache-aware
headroom 20,464,107,520 B is below the 23,192,823,398 B required for the whole
12 GiB process cap plus 15% physical reserve. The later attempt starts with
27,569,373,184 B, including 17,622,253,568 B of file-backed pages. Accounting
is printed free plus file-backed, with speculative pages included once.
Across its 124 samples, peak owned process footprint is 11,235,903,240 B,
minimum headroom 16,695,230,464 B (24.2948% physical), minimum pressure-free
63%, swap growth zero and final owned process count zero. File-backed bytes
remain a reclaimability estimate; sampling cannot exclude between-poll spikes.

The selected next **proposed** fusion is a K/V preparation epilogue. Preserve
the existing F32 K/V projections, K bias and K RMSNorm. Fold the existing K
RoPE arithmetic into the required head-major K write, and add V's BF16 bias
while writing head-major V. This removes standalone K RoPE and V bias
read/write passes: 2,998,927,360 B of logical traffic and two dispatches at
this shape. It preserves the required K/V transpose reads/writes and
1,499,463,680 B of prepared storage; this is not direct MPS projection into
head-major buffers. No time saving is inferred from byte counts.

The future opt-in probe must preserve F32 arithmetic order, per-encode refresh,
buffer lifetimes, existing command boundaries and the single terminal wait.
Validate packed K/V and downstream checkpoints against identical upstream
inputs. Then require unchanged strict full-block parity and repeatable
AB/BA/AB complete-arm improvement with all preparation charged. Reject on a
nonfinite value, any tolerance violation, new wait/allocation or nonrepeatable
timing. FFN bias+GELU is the simpler fallback (4,048,551,936 B logical traffic
and one dispatch removed); its measured parent is only 2.158707 s. Neither
fusion is implemented or promoted by this diagnostic commit.

Correlated MiniQuad source/math and raw-log adversary review is ROBUST within
these evidence scopes. Evidence is under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/full-block-mps-20261021/`.
`source-pins-v2.json`, native/build receipts, exact invocations, raw logs and
guards identify the qualified runs; `verification-v1.json` contains the
CPU-only evidence recomputation from pinned logs, not a replay of GPU output
arrays. `raw-evidence-receipt-v1.json` pins 66 artifacts;
SHA256 `143a7118654561487eb08524f8d4bc323d7afc9f5f83fd287e9d93cba43e8f53`.
The full-key binary digest is a post-run snapshot; its fresh build invocation
and native object receipts are retained. Failed admission and helper parsing
attempts are explicitly excluded in the receipt.

CPU-only refresh, using an output path that does not already exist:

```sh
evidence_root=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/full-block-mps-20261021
/opt/homebrew/opt/python@3.11/bin/python3.11 "$evidence_root/verify-full-block-v1.py" "$evidence_root/verification-refresh.json"
```

Expected signal: `passed: true`, all pinned source/object/binary/fixture hashes
match, strict upstream and exact A/A checks pass, controls reject corruption
and guards remain within their limits. `git diff --check` and Crystal format
checks also pass for the changed probe/spec. Refresh after scoped source,
object/binary, weight/fixture, geometry/dtype, device/toolchain/runtime,
queue/resource or guard changes, or artifact loss. Rollback removes the new
probe/spec opt-ins and this attribution record; runtime defaults stay intact.
