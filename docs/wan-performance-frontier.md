# Wan S2V performance frontier

The active target is exact GPU inference at B=1, N=36,608, C=5,120,
H=40, D=128, text context M=512 and FFN width F=13,824. This analysis
separates executable operation counts, measured latency and unmeasured ideas.
The numerical gate stays `2e-5 + 2e-5*abs(reference)`, with finite operands.
No new model-path host waits, activation readbacks or encode-time Metal-buffer
allocations are admitted. Experimental defaults remain off until their stated
whole-block gate passes.

Current bounded continuation (2026-10-22): the deep Quad audit separates
packed-model memory admission from custom-GEMM throughput. A source-bound,
paired predecoded-weight control gives 0.905788 s dense versus 0.057411 s
direct MPS (15.78x), with all 61,931,520 outputs bit-exact to the saved
same-format upstream oracle. Neither compute kernel changed. Prioritize a
packed-weight consumer using one shared F32 stage and existing MPS; custom
GEMM scheduling is a separate research route. See the final audit and results.

The opt-in rounded-codebook tiled
GEMM preserves same-format upstream projection results, including all
61,931,520 outputs of the selected 4480-row arm. It is rejected for speed:
3.519166 s median GPU time versus 0.060227 s for decode -> MPS -> bias, a
58.43x slowdown. The fresh predecoded control confirms a large gap in the
custom dense schedule even without decode. Decoder-only micro-optimization
cannot make that unchanged schedule competitive; this does not reject all
fused layouts, which can change reuse and data movement together.
Positional records still provide 4.5202 logical bits per
weight at the loosest tested policy; same-format parity does not remove the
previously measured quantization loss against BF16. No packed full-model or
video-quality admission follows. See the final contract and results below.

Earlier bounded attention result: native F32 key-shard attention with compensated PV
accumulation reduces matched full-block time by 17.79% and 18.02% in two fresh
processes with opposite warmup orders. All eight pairs favor the candidate and
all 187,432,960 output values pass the unchanged tolerance. This qualifies the
opt-in block probe on this host; substantial absolute host drift remains, and
neither full-model quality nor stable end-to-end latency is established.
See the final section for evidence and limits. Full generation remains stopped.

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

### K/V RoPE and V-bias packing fusion contract (2026-10-21)

At parent `11be5cc7`, the admitted baseline separately applies MPS projection,
V bias, K RMSNorm, K RoPE and head-major K/V packing. The candidate is an
opt-in performance rewrite (CAUTION): preserve F32 projection/accumulation,
K bias/RMSNorm and Q RoPE, then compute the identical K rotation and V bias
while writing fresh prepared K/V. No reduced precision, new wait, command
boundary, arm-time MetalBuffer allocation or production default is admitted.

The block flag requires a B=1 head-major MPS attention workspace, prepared MPS
linear workspace and execution chain; combining it with precomputed RoPE is
rejected before dispatch. Trace-only stores may preserve existing logical
checkpoints but must be absent from the measured no-trace arm. Prepared buffers
remain caller-owned, disjoint and refreshed inside every encode, including
changed inputs and completed-chain replay. Rollback is the default false flag.

The falsifier is any nonfinite output, violation of the unchanged finite
`2e-5+2e-5*abs(reference)` gate, lost bit parity against the equivalent native
path, stale packed K/V, new synchronization/allocation, or a failing resource
guard. Focused direct-pack tests cover mixed-sign F32/BF16 bias, multiple
heads, tail lengths, B>1 rejection and changed positions/K/V/bias. Prepared chunk1
and padded-tail chunk3 must retain all short-block upstream checkpoints, exact
repeat/chained/replay results and the existing finite/NaN negative control.
The full-spatial block has a separate exact native A/B check, not a complete
upstream block oracle or video-quality qualification.

DoD: serial guarded release builds and focused specs, then
`wan_s2v_full_block_benchmark --comparison=kv-prep-fusion
--candidate-self-chunk=128 --reuse-block-scratch --pairs=3 --weights=...`
under `wan_memory_guard.py --memory-accounting=file-cache
--max-footprint-mib=12288 --max-swap-growth-mib=64 --timeout-sec=600`.
Expected signal: exact full-output parity, one terminal wait per arm, zero
tracked encode allocations, all three complete-arm AB/BA/AB savings positive
and improved median. Failure of the timing gate leaves the candidate opt-in;
it does not justify loosening correctness or memory limits. The 15% physical
reserve applies throughout. No full model, Studio or generation is in scope.

### K/V preparation fusion result (2026-10-21)

Implemented `WanS2VBlock.encode(..., fuse_kv_preparation: true)` and the
corresponding audio-block forwarding flag. The default remains false. One
native dispatch now rotates normalized K, adds the original F32/BF16 V bias
and writes both head-major destinations. Q, projection precision, K
bias/RMSNorm, reduction order, attention geometry and command boundaries are
unchanged. The no-trace path omits the original row-major K/V stores; optional
trace stores preserve the existing logical checkpoints. The prepared pair is
separate from block scratch, refreshed on every encode and counted once in
the memory plan. This removes two native dispatches and 2,998,927,360 bytes of
logical activation traffic at the tested shape; it does not reduce the
allocated scratch budget or establish a measured speedup.

Current v4 release checks pass 17 examples each with MPS chunk rows 1 and 3.
These include 24 strict retained-upstream block checkpoints, bit-exact
baseline/repeat/chained/completed-chain replay, and direct preparation checks
for three geometries, both bias dtypes and all four trace modes with refreshed
K/V/bias/positions. The D=128 case exercises the actual 22/21/21 RoPE split.
A GPU comparison of the prepared K/V pair detects separately seeded one-bit
defects. Invalid resources are rejected before the first native dispatch.
The upstream fixture is a projection of batch 0 from the pinned B=2 reference;
no fresh upstream model execution was performed.

The complete native block comparison uses the real block-0 BF16 weights with
F32 activations/accumulation, B=1, S=36,608, C=5,120, H=40, D=128,
FFN=13,824 and text context 512. Inputs are synthetic. Both arms use chunk128
MPSGraph SDPA, head-major K/V, cross groups 320 and compact lifetime-aliased
scratch. Layout preparation is inside every measured block encode. Phase-root
profiling is disabled and output validation is outside the timed arms.

| Pair/order | Separate preparation (s) | Fused preparation (s) | Wall-time saving |
| --- | ---: | ---: | ---: |
| 0 / AB | 11.512571 | 11.899208 | -3.358% |
| 1 / BA | 13.867772 | 12.258352 | +11.605% |
| 2 / AB | 16.752843 | 19.808156 | -18.238% |

The timing gate fails: two pairs regress and the median paired saving is
-3.358%. All six measured arms slow down in chronological order, so the sign
also follows arm order. The lower arm-wise B median is not causal evidence of
an improvement. Keep the flag opt-in; neither a speedup nor a repeatable
regression intrinsic to the fused kernel is established. A useful next timing
probe must distinguish this shared drift from preparation cost, rather than
repeat the same unqualified comparison.

Warmup and all three measured pairs are bit-exact over all 187,432,960 output
values, finite and within the unchanged `2e-5+2e-5*abs(reference)` gate. The
finite/NaN corruption control is rejected (2 bit differences, 2 tolerance
violations, 1 nonfinite). Every arm retains 624 command segments, one terminal
wait, zero tracked encode MetalBuffer allocations, zero inter-operator host
waits and zero activation readbacks. This is full-spatial native A/B parity,
not a full-spatial upstream block oracle or a video-quality test.

The successful run finishes under the original 12 GiB process-tree cap and
15% policy: sampled peak footprint 10.686 GiB, minimum cache-aware headroom
15.138 GiB (floor 9.6 GiB), minimum `memory_pressure -Q` 63%, zero swap growth
and no remaining owned children. The guard includes printed free pages plus
file-backed pages once; reclaimability is estimated and sampled telemetry
cannot exclude between-poll spikes. The 11,194,526,228-byte plan includes the
1,499,463,680-byte prepared pair and 1 GiB opaque MPS reserve.

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/kv-prep-fusion-20261021/`.
Qualified inputs are `source-pins-v4.json`, `crystal-build-receipt-v4.json`,
`upstream-chunk1-v3.*`, `upstream-chunk3-v3.*` and `full-block-pairs-v2.*`.
The v4 build verifies the seven unchanged native objects against the parent
receipt before linking. Earlier build/test setup failures and
`full-block-pairs-v1` (prepared buffers incorrectly owned by block scratch)
are excluded. The corrected harness gives the pair separate ownership without
relaxing any overlap check or changing its byte budget. Correlated MiniQuad
source/math and raw-log review is ROBUST for bounded correctness and the
sampled resource contract, VULNERABLE for a speedup claim.

CPU-only evidence refresh, using a new output path:

```sh
evidence_root=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/kv-prep-fusion-20261021
/opt/homebrew/opt/python@3.11/bin/python3.11 "$evidence_root/verify-kv-prep-fusion-v1.py" "$evidence_root/verification-refresh.json"
```

Expected signal: correctness evidence passes and the timing gate is false.
The completed `verification-v1.json` refresh exits 0 with those signals;
SHA-256 `a581130d929f0a2a0768d8cf0f1e3c77ff6e82c0cb6bb79bbb1d297ca390891f`.
This recomputes raw log metrics and checks local hashes; it does not replay
GPU arrays or authenticate the original execution. Refresh after scoped
source/object/binary, fixture/checkpoint, geometry/dtype, device/toolchain,
queue/resource or guard changes, or artifact loss. Rollback is the default
false flag; removing the optional implementation and focused probe restores
the parent surface. Full model, Studio and generation remain outside scope.

### K/V preparation timing drift diagnostic contract (2026-10-21)

The fusion experiment's chronological slowdown confounds a feature-specific
latency conclusion. Investigate it before changing production scheduling or
promoting the opt-in flag. This is a benchmark-only CAUTION change; rollback
removes the diagnostic modes and leaves the production default false.

Competing explanations and discriminators:

- Same-work A/A with identical prepared-K/V geometry can reproduce an ordering
  effect without fusion. Balanced AB/BA/BA/AB slots and explicit monotonic arm
  times expose this confound; an A/A skew does not identify its physical cause.
- Passive process CPU, GC, page-in/fault, resident and footprint snapshots
  outside arm timing distinguish increased host work from elapsed blocking.
  `encode_cpu_seconds` is a legacy name for host elapsed time, not CPU usage.
  Missing resource counters must stay unknown. Do not force collection.
- An isolated full-shape native K-RoPE, V-bias and head-major packing A/B probe
  measures preparation separately. Use the exact production kernels, source
  bias dtype and one terminal wait per arm, GPU-refresh inputs outside timing,
  and balance measured slots. A microbenchmark saving cannot promote a full
  block or video speed claim.

DoD: build source-pinned release diagnostics with verified existing native
objects; run a CPU-positive sensor control; run strict short retained-upstream
checks; execute guarded isolated preparation and same-work full-block A/A
diagnostics. The GPU comparison must scan every K/V or block output, remain
bit-exact/finite within `2e-5+2e-5*abs(reference)`, and reject a finite/NaN
corruption control. Preserve one terminal wait, zero inter-operator waits,
zero activation readbacks and zero tracked MetalBuffer encode allocations.
The 15% cache-aware reserve, process cap and swap-growth stop remain active.
No full model, Studio or video generation is authorized by this diagnostic.

The strongest failure mode is attributing shared chronological drift to the
candidate or to GC/thermal state without discriminating telemetry. A/A and
the isolated probe are guard checks against that conclusion. Private MPS
allocation, device contention and clocks remain unmeasured unless a later
bounded probe explicitly instruments them. Sensor checks, hashes and log
validators qualify only their executed properties, not a physical diagnosis.

### K/V preparation drift: measured separation (2026-10-21)

At parent `0146438e`, the diagnostics change benchmarks only. A source-pinned
release build reuses seven native objects after verifying their source/object
hashes. The CPU-positive sensor reports 99.758 ms user CPU in a 100 ms busy
loop; missing-group, decreasing-counter and signed-gauge controls pass. The
short retained-upstream suite passes 17 examples at each chunk size 1/3,
including 24 strict checkpoints, exact prepared K/V and replay checks. This
uses the saved B=2 fixture projected to batch 0; it is not a fresh upstream run.

The isolated native probe uses S=36,608, C=5,120, H=40, D=128, source BF16 V
bias and F32 arithmetic. A encodes the production bias, K RoPE and head-major
K/V transpose; B encodes the fused preparation. Each fresh process measures
AB/BA/BA/AB after a warmup; reverse the warmup across the two processes.

| Isolated preparation | Median A wall | Median B wall | Median paired saving |
| --- | ---: | ---: | ---: |
| Before full-block A/A | 19.040 ms | 8.931 ms | 53.094% |
| After full-block A/A | 20.420 ms | 8.879 ms | 56.202% |

All warmup/four-pair comparisons scan 374,865,920 K/V values and are bit-exact
and finite in both processes. Finite/NaN controls reject 2 differences,
2 tolerance violations and 1 nonfinite value. The arms use 3 versus 1 native
dispatch, one root/terminal wait, zero tracked encode buffer allocations,
zero inter-operator waits and zero activation readbacks. The explicit arena
is 5,998,304,788 bytes. Input refresh/copy and the roughly 3 GB comparison scan
are completed outside arm timing; they may affect later device/cache state.
The saving is about 10-12 ms, not evidence of a full-block or video speedup.

The same-work full-block control has fusion off in both arms: prepared K/V,
chunk rows 128, cross-attention groups 320, compact scratch and distinct arm
outputs/chunk temporaries. Warmup order is BA; measured order is AB/BA/BA/AB.
Chronological measured wall times are 11.818, 12.907, 13.947, 16.959, 19.848,
22.393, 30.374 and 38.388 s. The corresponding GPU-root duration sum rises
11.793 -> 38.250 s, and encode host elapsed rises 10.144 -> 32.433 s.
Process user+system CPU is only 0.268-1.114 s (1.7-3.0% of arm wall time).
The first four measured arms have no GC collection, page-in or disk read;
one collection and one page-in/53,248-byte read occur in arm five. These
counters do not explain the onset. The encoded work requires no fusion for
this slowdown; the old `encode_cpu_seconds` field cannot identify CPU work.

Warmup and all four full-block comparisons scan 187,432,960 outputs exactly
and finitely; the finite/NaN control rejects 2/2/1. Every arm retains 624
segments, one terminal wait, zero tracked encode allocations, zero
inter-operator waits and zero activation readbacks. Sampled peak footprint
is 11,472,062,240 bytes under the 12 GiB cap; minimum cache-aware headroom is
15,265,300,480 bytes, pressure-free minimum 63%, zero swap growth and no
remaining owned child. The 15% floor is unchanged. These are sampled limits,
not exclusion of between-poll spikes or a physical GPU-residency certificate.

The optional same-work phase replay completes with the existing root profiler
enabled only in B, after the timed A/A control. It is a separate three-pair
AB/BA/AB diagnostic, not a performance comparison. Across its three measured
B arms, the median phase intervals and median per-arm shares are:

| Root phase | Median interval | Median share of root union |
| --- | ---: | ---: |
| Self preparation | 0.907 s | 5.71% |
| Self-attention, output projection and residual | 12.028 s | 76.29% |
| Cross-attention, output projection and residual | 1.270 s | 7.96% |
| FFN | 1.627 s | 9.95% |

Each measured B arm has 624 roots. Root duration sum equals root union in this
capture; the envelope exceeds the union by only 0.550-0.643 ms. This weakens
simple starvation between roots as an explanation for this capture. A root
interval can include internal stalls or other device activity; these are not
exclusive kernel busy times, and phase unions are not generally additive.
Measured arms report no process GC collection, page-in or disk read. The
self-attention phase includes output projection and residual, so it does not
yet identify an individual slow operator. All four complete output scans are
bit-exact/finite and the negative control rejects 2/2/1. Sampled peak footprint
is 11,472,881,416 bytes, minimum cache-aware headroom 15,903,768,576 bytes,
pressure-free minimum 63%, zero swap growth and no remaining owned child.

Two earlier phase attempts stop on unavailable guard telemetry before emitting
arms and are excluded. The first also lacks a successful guard cleanup record;
a later PID/PGID absence snapshot is retained separately, not presented as a
successful guard cleanup. After these failures, an external observer wraps
only the guard's error logger to record exception causes. It forwards to the
unchanged logger and guard entry point without changing limits, deadlines or
success-path sampling. The completed run's exception trace is empty; the
cause of the earlier telemetry failures remains unknown.

Correlated MiniQuad source/math and raw-log review is ROBUST for the bounded
correctness, timing separation and sampled resource evidence, VULNERABLE for
physical drift diagnosis or a full-block speedup claim. Keep fusion default
off. The next useful discriminator is to separate self-attention from its
projection/residual and measure aggregate host elapsed time around the
existing native enqueue/commit calls, alongside current post-wait root
intervals. Preserve the command graph and terminal wait count; distinguish
hidden API blocking from additional CPU work before changing scheduling.

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/kv-prep-drift-20261021/`.
Qualified records are `source-pins-v1.json`, `crystal-build-receipt-v1.json`,
`build-v1.*`, `host-sensor-v1.*`, `upstream-chunk1-v1.*`,
`upstream-chunk3-v1.*`, `prep-before-v1.*`, `full-block-control-v1.*`,
`prep-after-v1.*` and `full-block-phase-observed-v1.*`. The phase observer is
`guard-error-observer-v1.py`; failed `full-block-phase-replay-v1/v2.*` and
`phase-v1-cleanup-check.json` remain explicit excluded-attempt evidence.
Refresh after scoped source/object/binary, fixture/checkpoint, geometry/dtype,
device/toolchain, queue/resource or guard changes, or artifact loss. Short
retained-upstream checks and full-shape native A/B equality do not establish
a full-spatial upstream oracle, full-model gain or video quality. No full
model, Studio or video generation was run.

CPU-only evidence refresh with a new exclusive output path:

```sh
evidence_root=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/kv-prep-drift-20261021
/opt/homebrew/opt/python@3.11/bin/python3.11 -B "$evidence_root/verify-kv-prep-drift-v1.py" --verify --phase-run full-block-phase-observed-v1 --output "$evidence_root/verification-refresh.json"
```

Expected signal: `passed_core_evidence`, `passed=true`, phase replay `passed`;
physical-cause, full-block treatment-effect and default-promotion claims stay
false. The executed `verification-v2.json` exits 0 and checks 270 pinned source
files, seven core guards plus the phase guard, binary/native/fixture hashes,
all complete comparisons and corruption controls. SHA-256:
`f6236222b21ddd781fb36c636f214671c89aa922dc74d24c37f6fdef53529cf6`.
Chronology is reconstructed from full-block monotonic intervals, not warmup
print order; preparation uses validated execution slots. Synthetic checks
reject wrong execution order and overlapping arm intervals. Earlier verifier
attempts produced no receipt; `verification-v1.json` has an incorrect
descriptive A/A pair-order field and is superseded by v2. The refresh checks
recorded properties and local hashes; it neither replays GPU arrays nor
authenticates the original execution. Removing these benchmark diagnostics
leaves the existing production implementation and its default false flag.

### Self-attention/API attribution contract (2026-10-21)

At parent `0a5fb0bb`, the combined self-attention/output-projection/residual
bucket accounts for roughly 76% of the observed full-block root intervals.
Host encode elapsed grows while process CPU remains small. Competing causes
are blocking or descheduling inside framework calls, actual host computation,
and longer GPU service intervals. Existing measurements do not distinguish
these or identify the physical cause of chronological drift.

This CAUTION diagnostic adds opt-in metadata and API timers only. Split at an
existing root boundary without adding `commitAndContinue`: the current root
contains the final attention output copy and the later projection/residual,
so that shared tail must be named explicitly instead of assigned to pure
attention. Preserve legacy unsplit profiling where no such boundary exists.
Measure existing MPSGraph encode, internal continuation and native projection
calls with wall and owner-thread CPU time. Nested calls require inclusive and
exclusive accounting; their inclusive totals must never be added together.
Wall minus thread CPU includes waiting and thread descheduling, and does not
prove a particular GPU wait, thermal limit or exclusive kernel busy time.

DoD: source-pinned release builds with rebuilt affected native objects; CPU
positive/nested/invalid-metadata controls; strict short retained-upstream
checks at unchanged `2e-5+2e-5*abs(reference)`; a bounded full-shape same-work
profile-on/off replay with complete bit-exact/finite output comparisons and
finite/NaN corruption controls. Expected invariants: unchanged root count,
one existing terminal wait, zero explicit inter-operator completion waits, zero arm activation
readbacks and zero tracked encode-time MetalBuffer allocations. Keep the 15%
cache-aware reserve, process cap, swap-growth stop and monotonic deadline.
No full model, Studio or video generation is included.

The strongest failure modes are a misleading phase label, nested timer double
counting, profiler overhead masquerading as a speed change, and a thread-local
collector leaking into another chain. Controls must cover these boundaries.
This task establishes attribution, not a scheduling optimization or default
promotion. Rollback disables the diagnostic flags; the existing tensor graph,
precision and fusion default remain unchanged. Refresh evidence after scoped
source/object/binary, fixture, geometry/dtype, device/runtime, queue/resource,
collector/guard changes or artifact loss.

### Self-attention/API attribution result (2026-10-21)

The opt-in `--api-profile` replay preserves the tensor graph and existing
continuations. Native collectors use fixed per-chain storage and owner-thread
wall/CPU clocks; disabled collectors take no clock samples. Disabled hooks
still initialize a scope and check host state; their cost has not been measured
against the pre-instrumentation binary. Graph and linear bridges also link
without the optional command-chain object through weak no-op
hooks. A completed short fixture needs no artificial attention boundary;
the full-shape replay splits before the already open projection root. That
root still contains attention tail work, including its output copy, disclosed as
`self_projection_residual_with_attention_tail`. No command is added to obtain
this split. Missing, empty, duplicate or inconsistent snapshots fail closed.

The v4 release build rebuilds all three affected native objects, verifies four
reused objects and pins 409 source files. CPU controls pass 17 examples plus
two native positive calls, the null-pointer control and the standalone optional
bridge link. Short real-width retained-upstream checks pass 19 examples each
at chunk rows 1/3, including 24 strict intermediate checkpoints at unchanged
finite `2e-5+2e-5*abs(reference)`, exact K/V controls and exact API-profiled
output. These reuse the retained first-batch upstream oracle; they are not a
fresh upstream execution or a full-spatial upstream comparison.

The bounded full block uses real block-0 BF16 weights with F32 operator math,
36,608 tokens, width 5,120, 40 heads of width 128, prepared K/V, query chunk
128, cross-attention groups 320 and fusion off. There are two warmup arms and
three measured profile-off/profile-on pairs in AB/BA/AB order. The following
are the three profile-on arms' GPU root-interval unions in seconds:

| Region | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Self preparation | 0.737 | 0.746 | 0.949 |
| Self-attention | 8.987 | 9.286 | 12.579 |
| Self projection/residual, including attention tail | 0.370 | 0.388 | 0.561 |
| Cross-attention/projection/residual | 0.876 | 0.911 | 1.088 |
| FFN | 1.339 | 1.470 | 2.031 |

Self-attention accounts for a median 73.01% of each arm's complete root union;
the projection/residual tail is 3.03%. These are asynchronous root intervals,
not exclusive kernel busy times or directly additive latency components.
The envelope exceeds the root union by only 0.397-0.429 ms; this does not
measure stalls within roots. Profile-on block wall times are 12.341, 12.823
and 17.274 s. Chronological slowdown persists, so this replay does not estimate
profiling overhead or establish a treatment speedup.

Host API timers localize the large elapsed spans:

| API entry phase and operation | Calls per arm | Median wall total | Median owner-thread CPU total |
| --- | ---: | ---: | ---: |
| Self-attention: MPSGraph encode, inclusive | 286 | 17.018 ms | 11.974 ms |
| Self-attention: nested `commitAndContinue` | 47 | 0.462 ms | 0.312 ms |
| Cross phase: `commitAndContinue` | 572 | 11,063.198 ms | 5.317 ms |

Graph encode wall ranges from 7.321 to 37.070 ms. Its inclusive row already
contains the nested continuation row; exclusive wall median is 16.556 ms.
The cross-labelled continuation wall ranges from 10.709 to 14.804 s, with
5.272-16.491 ms owner-thread CPU. The longest single such call per arm is
0.732-0.927 s. Thus the large host encode elapsed is inside the existing
superclass continuation call, rather than CPU Graph construction. The phase
label identifies when the host enters the call: prior self-attention can still be
queued, so it cannot attribute the elapsed wait exclusively to cross-attention.
Queue backpressure is a surviving hypothesis; locks and thread descheduling
are not separated by these clocks. Wall minus CPU is not certified GPU wait.
Measured profile-on arms record no GC collection, page-in or disk read.

Warmup plus three pairs scan all 187,432,960 output values each: zero bit
differences, tolerance violations or nonfinite values. The finite/NaN negative
control is rejected with 2 bit differences, 2 tolerance violations and 1
nonfinite value. Every arm retains 624 command roots, one existing terminal
wait, zero tracked encode-time MetalBuffer allocations, no explicit
inter-operator completion wait and no timed activation readback. Existing
framework API calls can still block as measured above. Output comparison
happens after the existing wait, outside arm timing.

The 12 GiB process cap, 15% cache-aware reserve, 64 MiB swap-growth stop and
600 s monotonic deadline pass. Sampled peak footprint is 10.685 GiB; minimum
cache-aware headroom is 13.900 GiB (21.72% physical), pressure-free minimum
62%, swap growth zero and final owned live processes zero. This uses printed
free plus file-backed pages; speculative pages are not counted twice.

The next useful lever is the actual long-key attention GPU work. Changing
CPU Graph encode alone cannot explain this block's seconds of latency.
A follow-up continuation discriminator should associate slow calls with the
previously submitted root indices and owner-thread stacks, without adding
waits. It must not promise to recover all continuation wall time: GPU work
already overlaps host enqueue, and many continuations may only expose the
cost of draining earlier work. Fusion defaults remain unchanged and off.

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/api-wait-split-20261021/`.
Qualified records are `source-pins-v4.json`, both v4 build receipts,
`build-v4.*`, `cpu-controls-v2.*`, `upstream-chunk1-v1.*`,
`upstream-chunk3-v1.*` and `full-block-api-v1.*`. Earlier build v1/v2 and
CPU control v1 failures are excluded; build v3 is superseded by coherent v4.
All runtime GPU probes completed successfully. Invocation records retain
exact guarded commands and SSD temporary/cache paths. No full model, Studio
or video generation was run.

Correlated MiniQuad source/math/raw-log review is ROBUST for these bounded
numeric, timing/accounting and sampled resource claims, VULNERABLE for a
physical wait cause or speedup claim. A CPU adversary found that the first
external verifier accepted three impossible nested-metadata forms that the
Crystal validator already rejected. The hardened verifier rejects zero-count
nonzero nested times, commit child counts exceeding call counts and nested
times exceeding their commit totals. A positive profile and all three
malformed controls pass their expected outcomes; the root also repeats the
three attacks against a copied actual raw profile. Runtime logs and code are
unchanged. Earlier verifier reports are superseded by the v2 final report
below for this structural validation boundary; this is not independent
runtime replication or attestation.

CPU-only evidence refresh with a new exclusive output path:

```sh
evidence_root=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/api-wait-split-20261021
/opt/homebrew/opt/python@3.11/bin/python3.11 -B "$evidence_root/verify-api-wait-v1.py" --verify --output "$evidence_root/verification-refresh.json"
```

Expected signal: `verified=true`, `source_files=409`, `timed_arms=8` and
`whole_output_comparisons=4`. The executed `verification-final-v2.json` exits
0; SHA-256 `765d7f5de1f6142381f6017ae738ca95de5e8c3a6ca0690285b907b3f33a70d0`.
Verifier SHA-256 is
`0e33b9c6c71b0baa2131105790ddfba25a930bf4d795afbf9074c2d63cdbbae2`.
It requires the pinned measurement parent to remain an ancestor and every
scoped source/object/binary/fixture byte hash to match; the current verification
HEAD is reported separately. This checks retained properties, not fresh GPU
outputs or authenticated execution. Refresh runtime evidence after the decay
triggers in the contract above.

### Continuation/root correlation contract (2026-10-21)

At parent `de7350ab`, long continuation calls are observed in the host cross
phase while earlier self-attention roots may still be outstanding. Aggregate
wall/owner-thread CPU clocks cannot distinguish queue admission backpressure,
framework locks and CPU descheduling. This CAUTION diagnostic preserves the
tensor graph, precision, root segmentation and single existing terminal wait.

Add a default-off bounded continuation trace: entry/exit host timestamps in a
documented GPU-compatible clock domain, owner-thread CPU spans, phase/nesting,
and current-root identities/statuses. Join these with existing root GPU times
and completion callbacks after the terminal wait. A tracked root is not thereby
submitted; callback delivery is not the instant of GPU completion. A separate
owned-process stack sample may identify the blocking framework call, but is
diagnostic and cannot establish an unprofiled speedup. No activation readback,
additional completion wait, GPU event, command or scheduled callback is added.

H1 predicts long continuation calls overlap earlier submitted GPU intervals
and that root admission/completion progress bounds the backlog. H2 predicts
owner-thread stacks in a host lock unrelated to queue admission. H3 predicts
descheduling without a stable blocking stack. Temporal overlap alone cannot
choose H1. Fail closed on unknown clocks, missing/overflowed records, invalid
root identities, timestamp/order contradictions or partial sampling evidence.

DoD: source-pinned release build; positive and malformed CPU trace controls;
retained short upstream checks at chunk rows 1/3 with the unchanged finite
`2e-5+2e-5*abs(reference)` tolerance; bounded full-shape profile replay with
complete bit/finite output checks and unchanged root/wait/allocation ledgers;
analyzed raw per-call/root timeline and, if available, owned-thread stack
evidence. Preserve the 12 GiB process cap, 15% cache-aware reserve, 64 MiB
swap-growth stop and 600 s monotonic deadline. No full model, Studio or video
generation. Rollback disables the trace/profile flags. Refresh after scoped
source/build, fixture, geometry/dtype, runtime/device, queue/resource, clock,
sampler/collector or guard changes, or evidence loss. The strongest guard is
that instrumentation leaves full outputs and command counts unchanged while
rejecting a deliberately broken timeline; a correlation remains provisional
until a stack or controlled backlog discriminator supports its mechanism.

### Continuation/root correlation results (2026-10-21)

The bounded probe supports queue-admission backpressure inside
`commitAndContinue`. The bridge creates the default Metal queue with
`newCommandQueue` (`src/ml/metal/bridge.mm:355`). Apple documents a limit of
64 uncompleted command buffers for
[makeCommandQueue()](https://developer.apple.com/documentation/metal/mtldevice/makecommandqueue()).
The installed MPS header says that continuation commits the current root and
creates the next buffer on that queue. This is a mechanism explanation for
the sampled window, not proof of every call's cause or a speedup claim.

The release v3 binary retains BF16 resident block-0 weights and F32 math at
N36608/C5120/H40/D128, with prepared K/V refreshed inside every timed arm.
Profile replay preserves the graph, chunk128 geometry and cross groups320.
Each traced B arm records 623 continuation calls and 624 roots. Three
measured AB/BA/AB pairs give the following slow-call observations:

| Pair | Calls over 1 ms | Their host wall sum | Their owner-thread CPU sum | First slow call wall |
| --- | ---: | ---: | ---: | ---: |
| 0 | 202 | 14.599701 s | 4.258461 ms | 1.020266 s |
| 1 | 307 | 14.556389 s | 6.118294 ms | 0.948036 s |
| 2 | 522 | 17.225644 s | 10.855296 ms | 1.106499 s |

All 1,031 measured calls over 1 ms occur in the host cross phase and enter
with 63 earlier callback completions still unobserved. The current outgoing
root makes a 64-root admission candidate. For 1,030 calls, 63 earlier GPU
intervals have not ended; pair1/serial338 has 62 because one GPU interval
ended before its callback was observed. That call takes 5.416 ms with
19.250 us owner CPU. Callback backlog must not be equated with GPU backlog.
The first slow call in every measured arm is serial64/root63; its owner CPU
is only 36.042--46.541 us.

A separate diagnostic run starts `/usr/bin/sample PID 4 1` at the first
measured-B marker. PID27875 and native owner TID335982823 match its process
and thread headers. All 3,385 owner-thread samples show this ordered stack:

```text
CogniTrackedMPSCommandBuffer commitAndContinue
  AGXG14XFamilyCommandQueue commandBuffer
    _MTLCommandBuffer initWithQueue:retainedReferences:synchronousDebugMode:
      _dispatch_semaphore_wait_slow
        semaphore_wait_trap
```

This directly locates the sampled blocking path at acquisition of the next
Metal command buffer. The 63-prior-root boundary and documented queue limit
support queue-capacity backpressure; the sample does not establish an exact
submission instant, the semaphore's private implementation, or every call's
cause. Sampling perturbs execution and is excluded from speed evidence. Its
wrapper finish timestamp is an upper bound taken after process completion,
not the end of the sample window; wrapper monotonic timestamps are never
subtracted from Mach/GPU timestamps.

The host phase name is not exclusive GPU-work attribution. Earlier
self-attention GPU intervals overlap 12.568602/12.460594/14.493401 s of the
slow cross-phase calls in the three measured arms. Adding that host wall
time to the earlier GPU work would count the same elapsed time twice.
Measured B root unions are 16.951891/16.873181/19.532099 s; their envelopes
contain only 0.476743/0.501359/0.621149 ms outside those intervals. These
are inter-root coverage gaps, not hardware utilization or a measurement of
idle cycles inside a root. The self-attention root range remains the largest
GPU range: 12.462699--14.495810 s. The next lever is its actual GPU work;
raising queue capacity alone does not remove that work. Do not infer a
latency benefit from this diagnostic A/B replay or its host timing drift.

All four full-block comparisons check 187,432,960 values and pass bit-exact
and finite equality. The separate sampled replay also passes both complete
comparisons. Both retained-upstream runs, chunk1 and padded-tail chunk3,
pass 19 examples and 24 strict checkpoints each at the unchanged finite
`2e-5+2e-5*abs(reference)` tolerance. These are short real-width upstream
fixtures; the full-spatial block uses synthetic activations. No full-spatial
upstream, full-model or video-quality claim follows. CPU controls pass
24 examples, the native positive/three malformed controls, optional linking,
and the external positive/nine malformed timeline/sample controls.

Every runtime arm retains 624 roots, one existing terminal wait, zero tracked
encode MetalBuffer allocations, zero inter-operator host waits and zero
activation readbacks. Trace extraction uses the existing completed wait.
There are no added GPU commands, explicit waits, events or callbacks. Trace
enable allocates a fixed CPU record buffer and acquires the existing tracker
lock before encoding; metadata snapshots use that lock after completion.
The measured continuation/completion paths add no lock acquisition, but
enabled status reads can extend existing lock holds. Disabled trace state
and atomic checks also have a cost; zero overhead is not claimed.

Full and sampled guards pass the unchanged 12 GiB process-tree cap, 15%
cache-aware floor, 64 MiB swap-growth stop and 600 s deadline. Sampled peak
footprints are 10.682985/10.690065 GiB; minimum cache-aware headroom is
13.694763/15.325394 GiB (21.3981%/23.9459% physical). Swap growth is zero
and final owned live-process counts are zero. Printed free plus file-backed
pages count cache once; speculative pages are not added a second time.
Sampling does not exclude spikes between observations.

Earlier evidence is retained without promotion: v2/chunk3 rejects an overly
strict cross-clock diagnostic enclosure; v3 adds an enable-time resolution
guard and a 2 us timestamp-quantization bound with a rejecting control. This
does not change tensor tolerances. The first v3 full-block attempt fails
guard telemetry when `ps` times out and cleanup identity disappears; it is
not a successful full run. A fresh process listing independently confirms
the owned PID/group absent before the v4 retry. The retry changes no source,
binary, guard, limits or geometry. Verifier report versions before v4 are
superseded: root review corrected a sample TID join taken from another
process and a duplicate event-count key; controls now reject a stray TID
token and reversed sampled stack. No runtime evidence was rewritten.

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/continuation-correlation-20261021/`.
Qualified files are `source-pins-v3.json`, both v3 build receipts,
`build-trace-v3.*`, `cpu-controls-v3.*`, `upstream-chunk1-v3.*`,
`upstream-chunk3-v3.*`, `full-block-trace-v4.*` and
`full-block-trace-sampled-v4.*`, including retry receipts and the raw sample.
The corrected frozen CPU verifier SHA-256 is
`09d75dbf30800973f697e4ce36ff651d491041a83ad4469cd80b831d4c0a572c`.
It binds all 412 scoped source files, native objects, release binaries,
fixtures, launchers, guard settings and raw evidence. It requires the pinned
measurement parent to remain an ancestor and the original frontier document
snapshot to remain an exact prefix. Append-only reporting is allowed; it
does not attest authentic execution or establish a mechanism automatically.

CPU-only refresh, choosing a new exclusive output path:

```sh
evidence_root=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/continuation-correlation-20261021
/opt/homebrew/opt/python@3.11/bin/python3.11 -B "$evidence_root/verify-continuation-correlation-v1.py" --controls-only --output "$evidence_root/controls-refresh.json"
/opt/homebrew/opt/python@3.11/bin/python3.11 -B "$evidence_root/verify-continuation-correlation-v1.py" --verify --output "$evidence_root/verification-refresh.json"
```

Expected signals: controls accept the positive trace/sample and reject nine
malformed controls; verification reports `verified=true`, `source_files=412`,
`measured_B_arms=3` and `trace_events_per_measured_B=623`. Root execution of
both commands passes in `verifier-controls-root-v1.json` and
`verified-report-root-v1.json`. Refresh runtime evidence at the contract's
decay triggers. Rollback disables the diagnostic flags; queue capacity,
production defaults, tensor math and segmentation remain unchanged.

Final correlated MiniQuad review is ROBUST for the scoped instrumentation,
retained checks, continuation/root correlation and sampled wait-path claim.
It is VULNERABLE if generalized to a cause for every slow call. The separate
sample cannot attribute individual calls in the unsampled run, and the
64-buffer mechanism has no direct semaphore-capacity measurement. Independent
recomputation within the same agent lineage agrees with the raw counts; it
is not an independent runtime replication. Crystal formatting and
`git diff --check` pass for the final implementation.

## Independent-head locality experiment (2026-10-21)

Status: HYPOTHESIS, opt-in only. Measurement parent is `b9002577`.
The current full-key F32 MPS path handles all 40 heads per query chunk of
128 rows. Its prepared K/V pair is 1,499,463,680 bytes; one head's pair is
37,486,592 bytes. The previous continuation probe found queue admission
backpressure during outstanding GPU work, but did not establish a faster
attention implementation.

The candidate partitions independent heads into groups, binds contiguous
BHTD K/V views by buffer offset at setup, and traverses query chunks within
each group. GPU kernels gather Q into contiguous group scratch, zero padded
query rows, and scatter context into the original BQHD output. No keys are
omitted; F32 storage, scale, full-key SDPA and downstream block math remain.
Smaller K/V working sets might improve locality. This is an inferred
opportunity, not a measured cache or bandwidth claim. Extra graph calls,
gather/scatter traffic, reduced occupancy and hidden workspace are opposing
costs. A different graph shape can also change F32 rounding.

This is a CAUTION performance experiment. Defaults remain unchanged.
The admissible corridor is B=1, positive group size dividing the head count,
prepared head-major K/V, prebound aliasing offset views and setup-only plans.
All groups share one compiled executable; only K/V views and feed bindings
are distinct. Graph compilation is not repeated per group.
Reject unsupported shapes, explicit QK-softmax-PV grouping, overlapping or
undersized buffers, integer overflow and nonaliasing views. Existing chain
lifetime rules apply; release must follow the existing terminal boundary.
There are no added events, host waits, callbacks, encode-time Metal buffers
or activation readbacks. Rollback selects the original all-head workspace.

Falsifier/DoD: first run retained real-width upstream block checkpoints and
selected-query full-K native comparisons with finite tolerance
`2e-5 + 2e-5 * abs(reference)`, qualified by finite +1 and NaN controls.
Exercise nonzero head offsets, padded query tails and workspace replay.
Compare group endpoints at Qchunk=1024 and an all-head same-chunk control
against the production Qchunk=128 baseline on Q=8192, including K/V
preparation and gather/scatter. Reject failed numerical/alias/resource gates.
Only a surviving candidate proceeds to the real-weight full block at
Q=K=36608, with compact scratch and balanced AB/BA/BA/AB pairs. Reverse
warmup in a fresh process if chronological drift makes a win ambiguous.
The optimization must improve completed full-block latency, not only call
count, encoding time or a kernel proxy. Nonrepeatable timing leaves it off.

All runs use the unchanged file-cache-aware memory guard with a 15% physical
reserve, swap growth <=64 MiB, bounded process footprint and timeout.
Full-block cap is 12 GiB/600 seconds; selected-query cap is <=5 GiB/120
seconds. No complete model or video generation is authorized by this probe.
Commands, source/binary/fixture pins and raw receipts are retained under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/head-grouping-20261021/`.
Results decay on source, SDK/driver, hardware, dtype, shape, execution order
or guard changes; refresh the same retained oracle and balanced experiment.

Outcome: the opt-in implementation and bounded numerical checks are VERIFIED;
a repeatable speedup and default promotion are not established. Native setup
compiles the first grouped plan once and creates K/V offset views for the
remaining groups, retaining the backing buffers and executable. The reported
`distinct_executable_count=1` is a source-backed structural property, not
independent native runtime introspection. The all-head default retains its
original path. Grouping changes no precision or key coverage. It is not an
LTP/WBA transformation or an approximate/sparse attention method.

Retained-upstream block tests pass 20 examples each at G1/C1, G4/C3 and the
unchanged G40/C128 default, with 24 reused-scratch checkpoints per configuration.
Replay/default/chained outputs are bit-exact, with one existing terminal wait
per chain and zero tracked encode MetalBuffer allocations. Grouped admission
rejects invalid divisors, missing prepared K/V, unsupported graph mode, aliases
and undersized scratch before preparing pipelines. Full-K selected-query checks
cover Q=128, a nonzero-offset padded tail of 257 rows, and Q=8192. The retained
oracle's maximum observed absolute difference in the G4/Q8192 control is
1.1473894e-6, with zero violations at the unchanged finite tensor tolerance.
Finite +1 and NaN controls reject the seeded defects. Changed K/V salts change
the output; fresh and already-used workspaces agree bit-exactly after refresh.
These are comparisons with pinned retained upstream fixtures, not a fresh
upstream execution or a full-spatial upstream block/quality comparison.

The completed full-block run `full-block-g4-ba-v3` uses real BF16 block0 weights,
F32 math, synthetic GPU inputs, B1/Q=K36608/H40/D128 and compact scratch on
Apple M2 Max. A is G40/C128; B is G4/C1024. Both refresh the same full prepared
K/V pair inside each timed block. Cross-attention grouping stays at 320 and
K/V preparation fusion stays disabled in both arms. Warmup is BA and the four
measured orders are AB, BA, BA, AB:

| Pair | Order | A wall seconds | B wall seconds | Paired latency reduction |
| --- | --- | ---: | ---: | ---: |
| 0 | AB | 20.152372 | 17.916411 | 11.10% |
| 1 | BA | 20.372410 | 18.078557 | 11.26% |
| 2 | BA | 22.121236 | 18.968733 | 14.25% |
| 3 | AB | 23.473078 | 23.254743 | 0.93% |

Median paired latency reduction is 11.1774%; separate arm medians are
21.246823 s and 18.523645 s (12.8169% ratio-of-medians reduction). The paired
statistic and ratio of medians are different quantities. All five full-output
comparisons, including warmup, match 187,432,960 values bit-exactly and pass
finite checks. Every arm has one existing terminal wait, zero inter-operator
host waits/readbacks and zero tracked encode MetalBuffer allocations. Segment
count changes from 624 to 637. Group size and query chunk both change, so this
does not isolate a grouping-only speedup or prove a cache/bandwidth mechanism.
Root GPU intervals include command-buffer gaps; they are not kernel busy time.

The successful full-block guard finishes with rc 0 and zero owned processes.
Peak sampled footprint is 11,354,572,528 bytes (10.575 GiB), minimum cache-aware
headroom is 15,768,748,032 bytes (22.9465%), pressure-free minimum is 63%, and
swap growth is zero. The unchanged outer guard enforces a 12 GiB footprint cap,
15% physical reserve and 600 s deadline. File-backed pages count once; their
reclaimability and spikes between samples remain limitations of the guard.

Timing remains VULNERABLE to chronological drift: both arms slow during the
completed run and the final pair nearly loses the benefit. Standalone G4/Q8192
first improves by about 5%, then regresses by about 7.9% in a later control;
short Q=128/257 is slower due to padded work. The all-head C1024 control is
slower than C128. Its separate control arm always runs after A/B, so it cannot
establish a causal locality benefit. Do not enable grouping automatically or
extrapolate this run to diffusion steps, end-to-end generation or all devices.

Four incomplete stages are excluded from qualified timing: the first all-head
primitive (`primitive-g40-q8192-ab-v2`), G5/Q8192 (`primitive-g5-q8192-ab-v3`),
the initial full-block AB attempt (`full-block-g4-ab-v3`) and the fresh AB
replication (`full-block-g4-ab-repeat-v3`). Each hits the existing one-second
`ps` telemetry timeout. The first full-block and G5 cleanup also lose the
leader identity; separate successful process listings confirm their owned
PID/group absent, recorded in their `cleanup-absence.json` files. This is not
guard-verified reaping. The fresh AB replication stops before timing arms
and completes guard cleanup. No timeout, memory floor or tensor tolerance is
weakened; incomplete arms do not become successful runs. G5 has no qualified
completed result and remains unselected.

Evidence under the experiment root includes `source-pins-v2.json` and
`source-pins-v3.json` (413 source files each), native/release build receipts,
per-stage invocation/log/guard/error files, `completed-full-block-ba-v1.json`
and `verify-head-groups-v1.py`. V3 changes the primitive probe's replay
diagnostic; production, benchmark and block-test sources are unchanged from
the V2 numerical runs. The archived V2 probe is `probe-source-v2.cr`.
The full-block release binary SHA-256 is
`4e927b3ca9c0942df0d24d48a20b9f6512a8fbec788a768afa70a3563e8c9205`.
The retained short fixture SHA-256 is
`fb7943d637bc8a6853707cd2ed284595e6744c4368083c36993fd1dd26d78da9`;
the full-K manifest SHA-256 is
`7586da395e4acedcd667639c91350119135183a0ed692fa994faeb683c6b850a`.
Build receipts bind the native objects, SDK/compiler, source and release
binaries. Invocation receipts record the bounded commands without raw
environment dumps. Refresh the curated evidence index with a new exclusive
output path:

```sh
evidence_root=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/head-grouping-20261021
/opt/homebrew/opt/python@3.11/bin/python3.11 -B "$evidence_root/verify-head-groups-v1.py" "$evidence_root/verification-refresh.json"
```

Expected: `verified=true` for the stated bounded scope,
`stable_speedup_verified=false` and `default_promotion_admitted=false`.
The verifier rehashes sources, objects, binaries and fixtures, checks numerical
controls and guard receipts, recomputes the balanced full-block pairs, and
rejects both incomplete full-block runs as negative controls. It indexes local
evidence; it does not attest authentic execution or prove a causal mechanism.
The next discriminating signal is a completed fresh AB/BA replication with
unchanged guard settings and acceptable timing stability. Resolve the telemetry
failure separately before further heavy runs; no default change is admitted.

Root refresh passes in `verified-head-groups-root-v1.json`; verifier SHA-256 is
`a1481eed40b26669d966e106e403624dd0a7f8b1f37f5f6ad06ae34a14fef830`.
Final correlated MiniQuad review is ROBUST for the scoped implementation,
retained numerical checks and one completed comparison; a repeatable speedup
remains VULNERABLE. The review and verifier use the same runtime evidence and
are not independent replications. Crystal formatting and `git diff --check`
pass. No production default changes and no full generation is run.

### Process telemetry repair and full-block replication (2026-10-21)

Completed transition: replace the guard's whole-system `ps -axo` listing with a
single process-group query and repeat the existing G40/C128 versus G4/C1024
full-block comparison. The observed failure was the one-second `ps` timeout;
its internal kernel or scheduling cause is not established. Targeting the
query is a hypothesis to test, not a reason to accept missing telemetry.

This is a CAUTION change to containment instrumentation. Preserve the one-second
subprocess cap, 0.5-second sampling interval, waitable unreaped leader pin,
fail-closed cleanup, 15% reserve, footprint cap and swap-growth limit. Reject
foreign, duplicate, malformed or absent group rows. Propagate the existing
runtime deadline to the group query, without applying an expired run deadline
to cleanup. The macOS `ps(1)` and `compat(5)` manpages define modern `-g` group
selection; explicitly select `COMMAND_MODE=unix2003` for that subprocess only.
Rollback is the telemetry-only diff; no attention source or default changes.

The strongest failure mode is omission of a live descendant or an exited
leader, causing false quiescence or loss of safe cleanup. DoD: CPU guard specs
must reject seeded selector/timeout/deadline/parse failures and preserve real
zombie-leader, TERM-resistant-descendant and unrelated-session controls. An
interleaved CPU membership probe must match targeted and full listings of a
known tree. Then retained-upstream block specs and two fresh full-block runs
with opposite warmup orders must finish under the unchanged guard. Compare
all full outputs at the unchanged finite `2e-5+2e-5*abs(reference)` tolerance;
record whole-block latency, one existing terminal wait, zero inter-operator
waits/readbacks and zero tracked encode MetalBuffer allocations. A failed or
incomplete run is excluded from timing; no stable-speedup or default-promotion
claim is admitted merely because one run improves. Evidence is retained under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/ps-telemetry-20261021/`.

The repair uses `/bin/ps -g <PGID> -o pid=,pgid=,stat=` and forwards the runtime
deadline to the initial group query and the ESRCH membership refresh. Cleanup
still has its own bounded queries and retains the unreaped leader identity pin.
No signaling, memory-accounting, threshold, polling or GPU code changes. Errors,
timeouts and empty/foreign/duplicate/truncated output still stop the guard;
unverified quiescence cannot become a successful completion.

The new CPU falsifiers first fail against the old implementation (four failures
and four errors), then all 38 guard tests pass. Existing live-child, zombie-leader,
TERM-resistant descendant, unrelated-session, telemetry-outage and cleanup
controls remain green. In a CPU-only probe with a legacy-mode parent, modern
targeted and full listings match the known PID/zombie-class set in all 30
interleaved pairs while an unrelated session creates and reaps children. The
zombie-only and exited-leader/live-descendant cases also agree. Targeted `ps`
median is 3.491688 ms versus 42.293395 ms for the full listing. This descriptive
CPU result does not establish a causal full-block speedup from telemetry alone.
The initial membership probe v1 fails in its own zombie-only cleanup control;
it is retained and excluded. Corrected v2 preserves the guard unchanged, skips
the zombie-only signal using the pinned snapshot and verifies owned-process
absence after cleanup.

Two new native full-block processes complete with opposite warmup orders and
the same balanced measured order AB/BA/BA/AB. A is G40/C128 and B is G4/C1024;
both retain 320 cross-attention groups per segment, compact shared scratch,
the same real block-0 BF16 weights, synthetic GPU-filled inputs and full-key F32
math. K/V preparation remains inside each timed block. Phase-root and API
profiling are disabled.

| Warmup | A median wall seconds | B median wall seconds | Median paired reduction | B wins |
| --- | ---: | ---: | ---: | ---: |
| AB | 13.548086 | 11.619343 | 11.5871% | 4/4 |
| BA | 11.244238 | 10.351370 | 6.3081% | 3/4 |

The AB-run paired reductions are 0.2663%, 13.6182%, 15.1316% and 9.5560%; the
BA-run reductions are 5.9355%, 6.6806%, 9.2210% and -1.1654%. Arm medians and
the median of pair reductions are different statistics. Opposite warmups now
complete and both run medians favor B, but chronology drifts and one pair loses.
A stable gain and default promotion remain unverified; G40/C128 stays default.
This comparison changes both head-group and query-chunk geometry and does not
isolate a grouping/cache mechanism. Earlier guard-aborted timings stay excluded.

All ten complete-output comparisons (two warmups plus eight measured pairs)
are finite and bit-exact on 187,432,960 values each. Both short retained-upstream
configurations pass 20 examples and 24 reused-scratch checkpoints. The separate
full-K/Q8192 candidate check passes at maximum oracle absolute error 1.1473894e-6,
including changed-K/V workspace replay and finite +1/NaN rejection controls.
The full-block +1/NaN controls also reject. Tensor tolerance remains finite
`2e-5+2e-5*abs(reference)`. This re-executes native checks against retained
upstream fixtures; it is not a fresh upstream runtime, full-spatial upstream
block comparison, real-activation comparison or video-quality assessment.

All 20 timed arms retain one existing terminal wait, zero tracked encode
MetalBuffer allocations, zero inter-operator waits and zero activation readbacks.
No new GPU synchronization is introduced. In the two full runs all 453 targeted
queries succeed: medians 2.7480/2.8918 ms and maxima 15.3790/11.1849 ms. The
unchanged 12 GiB/600 s guards finish with zero owned live processes. Peak sampled
footprint is 10.681322 GiB, minimum cache-aware headroom 19.9987%, minimum
pressure-free 54%, and swap growth zero. File-backed pages include speculative
pages once; sampled guards do not bound between-poll spikes or guarantee cache
reclaimability. No full model, Studio or generation runs.

Reproduction from the task worktree (CPU-only evidence refresh):

```sh
TMPDIR=/opt/mldata_ssd/cogni-ml/video-scene-20261019/tmp PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 -B spec/wan_memory_guard_spec.py
/opt/homebrew/opt/python@3.11/bin/python3.11 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/ps-telemetry-20261021/verify-replication-v2.py /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/ps-telemetry-20261021/verified-replication-refresh.json
```

The verifier requires a new output path. `replication-pins-v1.json` binds 413
scoped source files (only the guard differs from parent `32fc1459`), unchanged
release binaries/build receipts and retained fixtures. Its SHA256 is
`395d779058190045be6a9400f7c92b175aded8f71d692418f1153a03cccd870d`.
The completed root receipt `verified-replication-root-v2.json` has SHA256
`1ed543735015f1e20cb1ef8e39695fba6662601067e53b89859f0b94f295d8c3`;
the verifier has SHA256
`4fa6a6aacf23c76c25cfea38e556061ef53e382454f630688ce216d82cac81fc`.
It binds 35 raw receipts by hash and checks actual monotonic arm chronology,
pair IDs and measured-record order, geometry, numerical controls, resource
thresholds, live-process quiescence and ps receipts. Both old incomplete
full-block attempts are rejected. `default_promotion_admitted:false` is an
explicit scope policy, not a derived proof of performance stability. The v1
verifier/receipt is retained but superseded: MiniQuad found that swapping A-arm
pair IDs between two BA pairs and reordering their JSON records could change
the derived paired median to 5.9060% without rejection. The v2 analyzer and
verifier bind pair IDs to timestamp positions as well as labels, and require
measured record order to match time order. Original raw measurements are
unchanged; no GPU rerun was needed for this evidence-verification repair.

Final correlated MiniQuad review is ROBUST for the scoped telemetry repair,
completed runs and corrected receipt validation. Three in-memory controls are
rejected: wrong arm geometry, cross-pair relabeling/reordering and same-ID
measured-record reordering. `replication-v2-adversary-controls.json` binds the
verifier, analyzer and root report; its SHA256 is
`12133ea127eee1b614fc8bf27ce2dc86db2bdf71859279cef308c08c9e3d7dc9`.
Review reuses the same raw measurements and is not an independent hardware
replication. A stable-speedup/default-promotion claim remains VULNERABLE.

Next signal: quantify chronological drift and isolate head grouping from chunk
geometry before selecting a production default. Refresh this evidence after
scoped source/binary/fixture, ps/OS/SDK/driver, device/dtype/shape, order,
queue/resource, collector/guard changes or artifact loss. Rollback reverts only
this telemetry/test/documentation change; the existing attention opt-in remains.

### Head-group and query-chunk factorial comparison (2026-10-21)

The previous comparison coupled head grouping and query chunk size. The new
opt-in `--baseline-self-chunk=N` isolates G40 versus G4 at C128 and C1024 on the
native full block. It is accepted only in head-groups mode for 128/256/512/1024;
the baseline default remains 128. GPU operators, two output buffers, shared
compact scratch and prepared K/V are unchanged. Rollback removes this harness
option and its documentation; production attention defaults are unchanged.

The falsifier is a direct equal-chunk pair where G4 is no faster, or a
geometry-dependent reversal. Execute four fresh processes in chunk order
128/1024/1024/128 with warmups AB/BA/AB/BA; within each process A=G40 and B=G4,
both at the declared chunk, followed by AB/BA/BA/AB measured pairs. This
counterbalances process order and warmup, but does not establish a causal
chunk-only speedup across processes or eliminate nonlinear host drift.

DoD: the old binary must reject the new baseline option; the new plan-only path
must report both actual chunk geometries and reject unsupported values or use
outside head-groups mode before GPU setup. Native retained-upstream probes
must cover all four attention configurations at full K, including workspace
reuse and finite +1/NaN rejection. Four complete guarded full-block runs must
compare all output values at finite `2e-5+2e-5*abs(reference)`, retain one
existing terminal wait per arm, zero inter-operator waits/readbacks and zero
tracked encode MetalBuffer allocations. Bind pairs to actual monotonic
chronology, not JSON labels alone. Seed geometry and pair-order violations
against the result verifier. A guard-aborted run is excluded, not retried with
weaker protection.

The strongest risks are opaque MPS scratch at G40/C1024 and host drift that
mimics a grouping effect. Preserve the 12 GiB/600 s full-block cap, 5 GiB/120 s
primitive cap, cache-aware 15% reserve, one-second ps timeout and 64 MiB swap
growth limit. No model, Studio, full generation, precision change, new GPU
synchronization or default promotion is admitted. Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/head-group-chunk-factorial-20261022/`.

All four fresh processes complete at the same frozen source and release binary,
using official block-0 BF16 weights, F32 math and synthetic GPU inputs at
N36608/H40/D128/C5120/FFN13824. K/V preparation is charged inside every timed
arm. Cross-attention stays native tiled at groups320, and diagnostics are off.
The 128/1024/1024/128 process order and opposite warmups are preserved:

| Chunk | Warmup | G40 geometric mean (s) | G4 geometric mean (s) | Paired latency reduction |
| --- | --- | ---: | ---: | ---: |
| 128 | AB | 22.122403 | 17.440045 | 21.1657% |
| 1024 | BA | 27.746835 | 18.573550 | 33.0607% |
| 1024 | AB | 26.576518 | 17.859068 | 32.8013% |
| 128 | BA | 20.266889 | 16.316973 | 19.4895% |

Each process contains four AB/BA/BA/AB measured pairs after two warmup arms.
The primary grouping contrast is the mean of `log(T_G40/T_G4)` within each
process, then an equal-weight mean of its two process contrasts per chunk.
This gives 20.3320% latency reduction at C128 and 32.9311% at C1024. The
corresponding pooled geometric arm means are 21.174331/16.869166 s and
27.155372/18.212806 s. All 16 measured pairs favor G4; within-process pair
reductions range from 15.4559% to 27.5191% at C128 and 31.0462% to 34.9603%
at C1024. The eight pairs per chunk are repeated observations in two processes,
not eight independent hardware replications.

The descriptive interaction is `theta_1024-theta_128 = 0.17214774`, or a
1.1878533 ratio of G40/G4 time ratios. Grouping has a larger observed relative
benefit at C1024. Absolute G4/C128 time is lower in this matrix, but the chunk
contrast spans processes and nonlinear host drift remains possible. Neither a
causal cache explanation nor a preferred production chunk follows. The next
discriminator is C128 versus C1024 with G4 fixed in the same process, including
the complete block and opposite warmups. Default promotion remains excluded.

All 20 complete-output comparisons pass finite
`2e-5+2e-5*abs(reference)` on 187,432,960 values each. At C128, all ten
comparisons have maximum absolute error 2.4795532e-5 and maximum normalized
error 0.70172626; these outputs are not bit-exact. At C1024 all ten comparisons
are bit-exact. Each process rejects the finite +1/NaN corruption control.
Four short native retained-upstream configurations each pass 20 examples and
24 reused-scratch checkpoints. Two full-K/Q8192 probes cover all four
geometries, changed-K/V workspace replay and finite +1/NaN controls, with
maximum candidate oracle absolute error 1.1473894e-6. The oracle is retained
upstream fixture data, not a fresh upstream runtime. Full-block arm agreement
does not establish full-spatial upstream agreement or video quality.

All 40 timed arms retain one existing terminal wait and zero inter-operator
host waits, activation readbacks or tracked encode MetalBuffer allocations.
All four 12 GiB/600 s guards complete with zero owned live processes; all
1,538 targeted ps queries succeed. Peak sampled footprint is 10.684984 GiB,
minimum cache-aware reserve 20.2566%, minimum pressure-free 54%, and swap
growth is zero. The initial C128/AB v1 attempt is retained and excluded: it
was blocked before spawning at 20.807 GiB headroom versus the unchanged
21.6 GiB startup requirement. Its v2 replacement starts only after recovery.
File-backed pages count speculative pages once; sampled guards do not bound
between-poll spikes or guarantee cache reclaimability. No full model, Studio
or video generation runs.

`verify-factorial-v3.report.json` binds all ten completed GPU stages, the
excluded startup attempt, plan/selection controls and 413 scoped source files
to parent `8b2675e54429d02cb0de71a32d22321e7e522d19`. The modified harness is
bound through the release build receipt; the twelve reused build artifacts,
four fixtures and fourteen launch dependencies are rehashed. Its SHA256 is
`95f8c8f925684b229726332b3d556ecb5b7400073b40ea65e13b525cd19f359a`;
`verify-factorial-v3.report.attestation.json` binds that report, the verifier,
controls and raw-receipt manifest. The frozen verifier SHA256 is
`d074bc68cceabbd41d4434f68c6e2fc11e09f843ed18fe1778e2fe31e4d37ee0`,
and `factorial-pins-v1.json` is
`0805adc6225ba8e75b7dbfa613507059b37d76f5f28fe26c0af5c9f796269f6f`.

Correlated MiniQuad review is ROBUST for the harness selection, completed
local comparisons and receipt validation. Both AB/BA positive fixtures pass;
all five in-memory attacks reject: wrong matched chunk, cross-pair
relabel/reorder, same-ID raw-arm reorder, swapped process geometry and a
strict-pass summary with normalized error above one.
`verify-factorial-controls-v5.report.json` has SHA256
`3cc031f68e43338db2c684f658574f85427cd3ea0a642e802300c78a8bc5385b`.
Early verifier schema-mismatch attempts and superseded reports are retained;
they do not change raw measurements. Production/default promotion, a causal
chunk mechanism and full-generation performance remain unsupported.

CPU-only refresh, using a fresh output filename, rechecks source ancestry,
scoped bytes, build/fixture/guard receipts, chronology, numerical results and
the captured controls without GPU execution:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/head-group-chunk-factorial-20261022/verify-factorial-v1.py /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/head-group-chunk-factorial-20261022/verified-factorial-refresh.json
```

Refresh after scoped source/build/fixture, OS/SDK/driver/device, dtype/shape,
order, queue/resource, collector/guard changes or artifact loss. A later
documentation commit is admissible only while the pinned parent remains an
ancestor and every scoped byte still matches. Sampling and retained-fixture
limitations above remain part of the certificate.

### Fixed-G4 query-chunk comparison (2026-10-22)

Run contract: compare A=G4/C128 with B=G4/C1024 inside each complete-block
process, first with AB warmup, then BA in a fresh sequential process. Each
process retains four measured AB/BA/BA/AB pairs. The previous benchmark fixes
baseline heads at 40; add a head-groups-only baseline override, size its Q/A/B
tiles for the actual head group, and report actual geometry. Defaults stay
G40/C128, and GPU kernels, F32 operator math and synchronization stay unchanged.
Rollback removes only this benchmark selection change and its documentation.

The candidate hypothesis is lower complete-block latency at C128 with G4 fixed;
the prior cross-process absolute times are only motivation. A reversal with
opposite warmup, no repeatable paired direction, or strict numerical failure
rejects the corresponding performance claim. The measured value includes K/V
preparation, projections, attention, FFN and residual on real block-0 BF16
weights with synthetic GPU inputs, not video-generation quality or throughput.

DoD: capture old-binary rejection; validate both warmup plans, independent
baseline/candidate geometry, unchanged default, invalid head divisors and use
outside head-groups mode before GPU setup. Recheck G4/C128 and G4/C1024 against
retained upstream fixtures at short/full-K shapes, including changed-K/V replay
and finite +1/NaN rejection. Both guarded full-block processes must finish with
finite 2e-5+2e-5*abs(reference) over every output value, one existing terminal
wait per arm, zero inter-operator waits/readbacks/tracked encode MetalBuffer
allocations, and no diagnostics in timed arms. Bind pair IDs to monotonic
chronology and challenge geometry/order/numerical-summary validation.

Keep 12 GiB/600 s per full block process, 5 GiB/120 s per full-K probe, cache-aware
15% reserve, one-second ps timeout and 64 MiB swap-growth limit. Speculative
pages count once in free plus file-backed headroom. Below-budget startup or any
runtime guard failure stops the stage; retain incomplete attempts and exclude
their timings. No full model, Studio, generation, lower precision, new GPU
synchronization or production-default promotion. Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/g4-chunk-pairs-20261022/`.

Results at parent `792ae4ea`, with the scoped benchmark source SHA256
`80b9aab09d78d5f86bcacc408f9b21adf6ea3ace3994672cf16baf2c64eab918`:

| Warmup | G4/C128 geometric full-block seconds | G4/C1024 geometric full-block seconds | Observed C128 latency reduction |
| --- | ---: | ---: | ---: |
| AB | 12.028588 | 12.535343 | 4.0426% |
| BA | 22.458574 | 27.382974 | 17.9834% |

For each pair, the contrast is `log(T_C1024/T_C128)`. Average within each
process, then give the two process means equal weight: theta is
`0.11975747119096586`, giving an observed 11.2864% latency reduction at C128.
Five of eight pairs favor C128; individual reductions range from -2.1713% to
29.3191%. Measured arm times drift from 10.7537 to 32.5024 s. Both process means
favor C128, but this is not a stable speedup estimate, proof of a warmup-order
effect, or eight independent host replications. Retain C128 as the next
experimental baseline; defaults remain unchanged. An identical-arm control is
the next discriminator for host/queue drift before promoting a stable chunk
winner. C1024 includes a final 768-row query tail padded to 1024; its cost is
part of the full-block treatment, not an isolated head-group or kernel effect.

All ten complete-output comparisons pass the unchanged finite
`2e-5+2e-5*abs(reference)` check over 187,432,960 values each. They are not
bit-exact: maximum absolute error is 2.4795532e-5, normalized error 0.7017164.
Both retained-upstream short native checks pass 20 examples and trace 24
checkpoints. Both full-K probes compare Q8192/K36608/H40/D128 with retained
upstream outputs; candidate maximum errors are 1.1399388e-6 at C128 and
1.1473894e-6 at C1024. Changed-K/V replay, reuse of the original workspaces and
finite +1/NaN rejection pass. No fresh upstream runtime or full-spatial
upstream-block qualification follows from these fixtures.

The 20 timed arms retain one existing terminal wait each and zero
inter-operator waits, activation readbacks or tracked encode MetalBuffer
allocations. K/V preparation stays inside every arm. The two query workspaces
occupy 7,077,888 bytes together: 786,432 for C128 and 6,291,456 for C1024. Both
stay resident, so this experiment does not compare process peak memory between
chunks. All full-block guards finish normally: sampled peak footprint
10.526399 GiB, minimum cache-aware reserve 24.9160%, zero swap growth. The
12 GiB/600 s process budget and 15% reserve are unchanged; file-backed
headroom remains an estimate, and sampling does not bound between-poll spikes.

The first release build failed on nullable baseline-head inference; the typed
assignment fixed it before any measured GPU stage. Its failed receipt is
retained and excluded. Selection checks pass five valid plans and five invalid
configurations; default plan output is byte-identical to the previous binary.
`root-paired-analysis-v1.json` independently derives the contrasts directly
from raw arm records. Stage argv, build/fixture/guard pins and all raw receipts
remain under the evidence root above.

Correlated MiniQuad is ROBUST for scoped source/receipt/chronology/numerical
integrity, VULNERABLE for a stable or causal speedup claim. Positive AB/BA
fixtures pass; all six CPU mutations reject: wrong grouping, wrong chunk,
cross-pair relabel/reorder, same-ID raw-arm reorder, strict-pass normalized
error above one and mismatched stage argv. The final canonical Python 3.11
report is `verify-g4-chunk-pairs-v5.report.json`, SHA256
`c74a086768f13cd06c7a8298e27d0a14546cb4a8c89fb0171211468fa9791aa2`.
The controls-v5 report has SHA256
`d467868efe03ae37ed8727f067769536d3e17dcab366b76e858bc0076d8ee853`.
Earlier verifier schema/interpreter mismatches and their superseded reports
remain retained and excluded from the final certificate.

CPU-only refresh rechecks the scoped source ancestry/bytes, build/fixture/guard
pins, selection, chronology, numerical receipts and adversarial controls;
choose a fresh output path because reports are exclusive:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/g4-chunk-pairs-20261022/verify-g4-chunk-pairs-v1.py /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/g4-chunk-pairs-20261022/verify-g4-chunk-pairs-refresh-UNIQUE.report.json
```

Refresh after scoped source/build/fixture, OS/SDK/driver/device, dtype/shape,
order, queue/resource, collector/guard changes or artifact loss. A subsequent
documentation commit is admissible while the pinned parent stays an ancestor
and all scoped bytes match. This refresh authenticates existing results; a new
performance environment needs fresh paired measurements. Sampling, synthetic
inputs and retained upstream fixtures remain part of the evidence boundary.


## Identical-arm G4/C128 drift control, 2026-10-22

At source parent `9543964d23915a39ed67560281be8802a85ec17e`, reuse the
unchanged full-block binary/source from the fixed-G4 chunk experiment. Its
benchmark source SHA256 remains
`80b9aab09d78d5f86bcacc408f9b21adf6ea3ace3994672cf16baf2c64eab918`.
Both A and B explicitly select G4/C128, shared head-major K/V refreshed inside
every encode, lifetime-aliased block scratch and native tiled cross-attention
at 320 groups per segment. Real block-0 BF16 weights, F32 computation,
B1/N36608/H40/D128, context 512 and FFN 13824 stay fixed. Inputs and all
intermediate activations stay on GPU in the measured path. No kernel,
precision, production default or synchronization code changes.

The falsifier is an observable difference between identical treatments or a
time trend that survives counterbalanced pair order. The predeclared v3
contract keeps four measured pairs AB/BA/BA/AB after one warmup pair, with
warmup AB in one fresh process and BA in another. Each arm has a separate
native attention workspace/graph identity, equal-size query scratch and output
binding. Geometry and resource policy match; executable identity is not shared
between arms. Both processes run sequentially, with an interrupted/excluded
attempt between them, so warmup order is confounded with process and host
state. There is no predeclared stability cutoff or independent-host sample.

DoD: both accepted processes finish under the unchanged 12 GiB/600 s,
15%-of-physical-memory cache-aware guard and 64 MiB swap-growth cap; guard
coverage includes every native arm interval and its final sample has zero live
owned processes. Require ten complete-output comparisons to pass the original
finite `2e-5+2e-5*abs(reference)` gate over 187,432,960 values each, unchanged
GPU residency/explicit synchronization counters, retained-upstream short and
full-key qualification, direct raw-row drift arithmetic and hostile receipt
mutations. Missing final coverage or an exit receipt excludes an attempt;
complete native output alone is insufficient.

| Warmup | A geometric seconds | B geometric seconds | B relative to A | Last four / first four |
| --- | ---: | ---: | ---: | ---: |
| AB | 13.415658 | 13.378366 | -0.27797% | +13.51451% |
| BA | 12.903537 | 12.747967 | -1.20564% | +12.34978% |

For each process, average the four `log(T_B/T_A)` contrasts, then weight the
two process means equally. The resulting theta is
`-0.007456627887890185`, or **-0.74289% B relative to A** using
`100*expm1(theta)`. This is a descriptive identical-treatment contrast, not
an optimization. The same calculation for chronological second/first slots
is +3.94218% and +3.12516%, or +3.53286% with equal process weights. Individual
pairs are not independent host replications. These observations do not define
a calibrated noise bound, prove a warmup-order effect, or supply a bias
correction for the earlier chunk comparison.

The sixteen measured blocks span 11.310708-14.653707 s. Within-process
population CV is 7.15570%/6.62414%; last/first endpoint changes are
+19.00612%/+17.93433%. The first/last-four geometric comparison in the table
uses all eight observations per process, rather than selecting endpoints.
The existing sum of completed command-root GPU intervals shows
+13.49633%/+12.36186% late/early changes and an equal-process label contrast
of -0.72347%. The phase investigation below corrects the previous envelope
label: `gs_mps_chain_finish_wait` sums each root's end-minus-start timestamp.
This is not summed kernel execution time or proof of a GPU frequency/thermal
cause. The immutable A/A receipts retain their earlier field labels; the
numerical observations are unchanged.

Existing host observations record zero page-ins and zero process disk-read
bytes during all sixteen measured arms. Each process records one GC collection
and roughly 1.1 MiB of cumulative tracked GC allocation during these arms.
Actual process user+system time totals 5.395013/3.579420 s, versus
91.461990/87.797338 s of synchronous encode/enqueue elapsed host time and
15.994888/15.036571 s of terminal waiting. Cold file reads and CPU computation
are not supported as the dominant explanation for this measured drift.
Unobserved host contention, GPU frequency and queue/resource behavior remain
competing hypotheses; this control does not separate them. The next useful
probe attributes identical-arm drift to existing block phases/continuation
intervals and matches host/device conditions before another performance
promotion. Do not repeat C128/C1024 tuning solely from absolute cross-process
seconds.

All ten accepted full-output comparisons are bit-exact, with zero nonfinite
values, tolerance violations, maximum absolute error and normalized error.
All twenty arms, including warmups, retain one existing terminal wait and zero
inter-operator host waits, activation readbacks or tracked encode MetalBuffer
allocations. Both accepted guards finish with no live owned processes:
sampled peak footprint 10.525621 GiB, minimum cache-aware reserve 23.7881%,
zero swap growth and 493 successful ps queries. The one-second ps cap and
15% floor remain unchanged. File-backed headroom is a reclaimability estimate;
sampling does not bound between-poll spikes.

The fresh native short check passes 20 examples and 24 traced checkpoints
against the retained upstream fixture. Full-key Q8192/K36608/H40/D128 maximum
absolute error is 1.1399388e-6, normalized error 0.05249706. Changed-K/V replay,
reuse of the original workspace and finite +1/NaN rejection also pass. These
are retained-upstream short-block and full-key checks, plus native/native
full-spatial equality; they are not a fresh upstream runtime, full-spatial
upstream block oracle, full model run or video-quality assessment.

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/g4-c128-aa-drift-20261022/`.
The v3 contract/pins retain the completed AB-v2 and replacement BA-v3. AB-v1
was blocked before native launch at the original admission threshold. BA-v2
has complete native comparisons, but observer coverage ends about 50 s before
the last native arm and lacks a guard completion/launcher exit receipt; its
native timings are excluded. A later bounded process snapshot finds no
matching process, which does not repair the missing interval. The replacement
starts its unchanged bounded guard in a persistent separate supervisor
session with redirected output. The lost observer's cause is unknown;
SIGTERM/SIGINT cleanup support does not protect against abrupt external
termination. No existing raw files or receipts are repaired or overwritten.

`derive-root-aa-v3.py` independently derives the arithmetic and selected
row/coverage checks in `root-aa-drift-v3.json`. Earlier analyzer versions are
retained: v2 stopped before writing a report on a GC field-name mismatch; v3
uses the actual `gc_collection_count_delta` field. The companion receipt
verifier supplies complete lineage/guard checks; root arithmetic alone does
not certify the launch chain. No full generation is run and defaults stay
unchanged. Rollback removes only this documentation update.

Final `verify-g4-c128-aa-drift-v3.py` SHA256 is
`b685765b1635732fa32dab9a3536ad79132a75080c3b833bcaf813aabaa6493f`.
Its `verify-g4-c128-aa-drift-v3.report.json` and the root's separate CPU-only
precommit refresh have the same SHA256
`0a166c614ef10bdfdde3246e143a5ba7aca8670241974812a33f5e75e18b2133`.
The verifier checks 413 scoped source hashes, 15 launch dependencies, 12
reused-build artifacts, four fixture hashes, contracts, chronology and complete
guard coverage. The controls-v3 report SHA256 is
`d4334dd24627e845b707fc6380da5055cdbd20a4b6bdbbdbb0ad03256a343178`:
both valid warmup fixtures pass and all twelve mutations reject, including
forged early guard completion after all native results have been written.
Root report SHA256 is
`32b7bb93916709dad1d732ea4db06b672fd02f8b8e7a2a3c1416ea5b5006f33e`;
its independently derived contrasts and temporal drift match the verifier.
Correlated MiniQuad verdict: ROBUST for the scoped source/receipt, numerical,
chronology and descriptive arithmetic checks; a stable or causal performance
claim remains VULNERABLE. Agreement between reviewers is not an additional
independent measurement.

CPU-only refresh authenticates existing evidence without rerunning GPU work;
choose an unused output path because reports are exclusive:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/g4-c128-aa-drift-20261022/verify-g4-c128-aa-drift-v3.py /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/g4-c128-aa-drift-20261022/verify-g4-c128-aa-drift-refresh-UNIQUE.report.json
```

Refresh integrity checks on source/build/fixture/collector/guard changes or
artifact loss. Documentation-only commits preserve the certificate while the
pinned source parent remains an ancestor and all scoped bytes match. Changing
OS/SDK/driver/device, dtype/shape/order, queue/resource policy or host conditions
requires new matched measurements before transferring timing conclusions.
On a failed refresh, reopen dependent claims and keep defaults unchanged.

## G4/C128 phase and continuation drift contract, 2026-10-22

At parent `49cadb40`, the previous identical-arm control exposes temporal
growth without phase attribution. Existing diagnostics admit only G40 profile
replay and instrument only B. Add a benchmark-only `--profile-identical-arms`
adapter restricted to equal head-group/chunk geometry in `head-groups` mode.
Reuse existing root ranges, API wall/owner-thread CPU timers and fixed-capacity
continuation traces on both arms. No kernels, precision, command boundaries,
queue policy or waits change. This is a CAUTION diagnostic; rollback disables
the flag or removes the benchmark adapter.

DoD: CPU-only malformed-option controls reject unequal geometry and unsupported
modes before GPU setup; source-pinned release build and retained-upstream short
and full-key gates pass. Two sequential G4/C128 full-block processes use warmup
AB/BA and four measured AB/BA/BA/AB pairs. Both arms must retain one terminal
wait, zero inter-operator waits/readbacks/encode MetalBuffer allocations and
pass complete finite output comparisons at the original 2e-5+2e-5*abs(reference)
gate. This opt-in identical-arm diagnostic additionally requires bit-exact A/A
output; existing unprofiled modes retain their acceptance policy. Require
complete guard/native coverage under the unchanged 12 GiB/600 s,
15% cache-aware floor and 64 MiB swap-growth cap. No full generation.

Discriminate phase-root interval growth from host continuation wall time and
owner-thread CPU, then correlate slow calls with prior GPU intervals and
callback backlog. Phase unions can overlap and are not additive; a host cross
phase wait may expose outstanding self-attention work. This is the strongest
attribution failure to guard against. Profiled timing is diagnostic and cannot
establish an unprofiled speedup, thermal cause or hardware utilization. Preserve
incomplete attempts explicitly and do not repair missing receipts.

### Observed phase and continuation drift

The release build changes only the benchmark adapter, group-aware expected
Graph-encode count and symmetric phase-root count validation. G4 means four
heads per group: ten groups times 286 query chunks produce 2,860 Graph encodes,
not the 286 expected by the earlier G40-only diagnostic. Both arms now use the
same existing diagnostics. No engine, kernel, command-boundary, queue-policy
or precision changes accompany this measurement.

Both fresh processes finish with ten fully profiled arms each, including
warmups. The sixteen measured block durations span 10.027263-11.170112 s.
Last-four/first-four geometric host latency grows **2.81687% / 5.56178%** for
AB/BA warmup processes. The corresponding completed-root GPU interval sums
grow **2.77858% / 5.52162%**. These are sequential process clusters, not
independent host replications or a causal warmup-order comparison. The earlier
unprofiled 12-13% drift is not reproduced at its former magnitude; this is not
evidence that profiling fixes it or that the earlier cause is identified.

The table shows authoritative named phase interval unions. Seconds are
arithmetic means of the first and last four measured arms; percentage changes
use the predeclared geometric comparison.

| Phase | AB early / late seconds | AB change | BA early / late seconds | BA change |
| --- | ---: | ---: | ---: | ---: |
| Self preparation | 0.582244 / 0.688059 | +18.32288% | 0.683504 / 0.733270 | +7.28358% |
| Self attention | 7.542541 / 7.571027 | +0.37916% | 7.560000 / 7.889627 | +4.35919% |
| Self projection/residual with attention tail | 0.212428 / 0.228014 | +7.37822% | 0.223729 / 0.247230 | +10.65073% |
| Cross attention/projection/residual | 0.781275 / 0.799986 | +2.43706% | 0.790524 / 0.829524 | +4.95445% |
| FFN | 1.090741 / 1.204706 | +10.57225% | 1.182352 / 1.315898 | +11.47429% |

Self attention remains the largest absolute cost, about 7.5-7.9 s. It does not
explain the first process's drift: self preparation and FFN grow more there.
In the second process, self attention contributes the largest absolute phase
increment, while FFN again grows proportionally faster. There is no single
phase-only explanation common to both processes. Phase unions may overlap in
general; these particular traces report zero root overlap. The largest
measured difference between overall envelope and interval union is 1.028 ms.
Neither short gaps nor long root intervals measure hardware utilization or
prove that kernels continuously execute inside a root.

Each arm records 881 completed roots and 880 continuation events. In every
measured arm the first call longer than 1 ms is serial 64, with 63 earlier
tracked roots whose GPU end and completion observation both follow call
entry. This repeatable boundary is consistent with bounded queue pressure;
the trace does not expose hardware queue occupancy, a driver wait reason or
the cause of the time trend. Callback observation is not GPU completion time.

Across measured arms, mean continuation wall-minus-owner-thread-CPU time is
8.89858 / 9.15996 s per block. This proxy includes driver waiting or owner
descheduling; it is not an additional cost to add to GPU execution. Self
Graph encoding excluding nested continuation calls averages only
0.061007 / 0.065940 s. Inclusive Graph time already contains its nested
continuations and must not be counted again as command-construction work.

For this source-bound block only, infer closure-root ranges from continuation
entry phases and the explicit phase boundaries. Phase 3 maps to
`self_projection_residual_with_attention_tail`, not a pure projection root;
the last terminal root belongs to FFN. Reconstructed unions match every
authoritative phase union in all sixteen measured arms. This checked mapping
supports interval correlation, not kernel ownership. Cross-phase host
continuation spans overlap earlier self-attention roots for a mean
1.64057 / 1.68223 s, the self-attention/projection tail for
0.22020 / 0.23546 s and cross-phase roots for 0.55615 / 0.56794 s.
Charging the whole 2.41331 / 2.48206 s cross-phase wall-minus-CPU proxy to
cross-attention would misattribute outstanding self-stage work.

Measured native observations report zero page-ins, process disk-read bytes and
GC collections in both processes. Actual user+system CPU totals
2.752584 / 2.849925 s across eight measured arms, while each arm spends roughly
nine seconds inside synchronous encode/enqueue. These observations do not
support cold reads, GC pauses or busy CPU graph construction as the dominant
drift explanation. Frequency, bandwidth/resource contention and driver
scheduling remain hypotheses, not measured causes.

All ten full-output comparisons are bit-exact and finite over 187,432,960
values each. All twenty arms retain one existing terminal wait and zero
inter-operator waits, activation readbacks or tracked encode MetalBuffer
allocations. Fresh native qualification against retained upstream fixtures
passes 20 short-block examples with 24 intermediate checkpoints and full-key
Q8192/K36608/H40/D128. Candidate maximum absolute error stays 1.1399388e-6,
normalized error 0.05249706; changed-K/V workspace reuse and finite +1/NaN
controls pass. This is neither a fresh upstream runtime nor a full-spatial
upstream block or video-quality comparison. Pure profiler specs pass 22
examples; five invalid CLI configurations reject before GPU setup, and both
legacy plan-mode compatibility controls pass.

Both guards finish successfully with coverage through the final native sample
and zero live owned processes. Sampled peak footprint is 10.530870 GiB,
minimum cache-aware reserve 29.66325%, and swap growth zero. The 12 GiB/600 s
budget, 15% floor, 64 MiB swap-growth cap and bounded ps telemetry remain
unchanged. Sampling does not bound between-poll spikes. No full generation,
model service or Studio is started.

Decision: keep production defaults unchanged. The largest absolute performance
lever remains GPU self-attention. For drift attribution, next separate the
self-preparation/FFN GPU work from device-state effects under matched host
conditions. Reducing CPU enqueue time cannot be credited with recovering the
roughly nine seconds already overlapping submitted GPU work.

Evidence is external under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/g4-c128-phase-drift-20261022/`.
`phase-contract-v1.json` freezes the experiment before GPU work, while
`phase-pins-v2.json` binds 413 source files, 12 reused build artifacts, four
fixtures and 16 launch dependencies. The source-pinned release binary SHA256
is `06c0aef3d320aac3e5dea3103d0dac5b30aa208190d06c5e7be74124ca553f50`.
`verify-phase-drift-v1.py` authenticates this lineage, raw stage files, guard
coverage, chronology, geometry, API counts, trace reconstruction and numerical
gates. Its SHA256 is
`1772035f8570eb905d222e7616ef9055c8da0cc77f3011bbb4d5d15a8bf2bf35`;
canonical `phase-drift-v1.report.json` SHA256 is
`4ee14efcd939ad7f67a9001f97e0fccbe9fef7bc993b3326f9a66e9c12223bbe`.
`phase-drift-root-precommit-v1.report.json` reproduces the canonical report
except its creation time. `root-crosscheck-v1.json` confirms all 30 raw-file
digests and the separate arithmetic agree with that refresh.
The six CPU semantic attacks reject missing trace/root data, wrong geometry,
premature guard finish and finite/nonfinite comparison failures. Root's
separate `root-phase-arithmetic-v1.py` and read-only correlated MiniQuad review
independently recompute the phase changes and interval overlaps. The scoped
verdict is ROBUST for these observations; a common drift mechanism or an
explanation of the earlier unprofiled drift remains VULNERABLE.

An initial launcher failed before creating a native invocation: a pin-check
loop overwrote the requested stage name. Its supervisor receipts and
`prelaunch-failure-v1.json` remain preserved and hash-bound by pins v2; it is
excluded from GPU timing and memory samples. The frozen contract also retains
the inherited `secondary_gpu_envelope` label. Source inspection corrects that
field to a completed-root duration sum; neither contract nor raw observations
are rewritten, and the numerical comparisons are unchanged.

CPU-only refresh, using an unused report path, launches no GPU work:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/g4-c128-phase-drift-20261022/verify-phase-drift-v1.py --output phase-drift-refresh-UNIQUE.report.json
```

Compare the refreshed raw-file digests with the canonical report. Scoped
source/build/fixture/collector/guard changes or artifact loss require integrity
refresh; documentation-only commits preserve the certificate while the pinned
parent is an ancestor and scoped bytes match. OS/SDK/driver/device, dtype,
shape/order, queue/resource policy or host-state changes require new matched
measurements before transferring timing conclusions. A failed refresh reopens
dependent claims and leaves defaults unchanged. Rollback removes only the
opt-in benchmark adapter and its documentation.

### Smaller independent head groups at fixed C128 (2026-10-22)

Design-sealed experiment, parent `2d6eca5b`: compare G4/C128 with G1/C128
and, if the first candidate fails to improve, G2/C128. Heads are independent;
each retains all 36,608 keys, the same F32 SDPA scale and no mask/dropout.
Only the existing opt-in workspace's head grouping changes. The selected-query
probe gains an explicit baseline group argument, default 40, so a G4 baseline
does not silently fall back to G40 or change the baseline scratch accounting.
No engine, kernel, queue, dtype, barrier, host wait or allocation policy changes
are admitted. Rollback removes the probe option and restores the experiment
settings; production defaults remain unchanged during qualification.

Prediction: a smaller K/V group may lower attention latency enough to outweigh
more gather/SDPA/scatter calls. A cache mechanism is a hypothesis; these probes
do not measure cache hits. Falsifiers are a failed finite unchanged upstream
gate, stale K/V replay, a new encode allocation/wait/readback, guard violation,
or a complete-block regression after balanced warmup/order controls. Do not
infer a block win from selected queries or older profiled timings.

DoD: release-build the probe; reject invalid baseline divisors before fixture
or Metal access; check default and G4 CPU memory plans; compare Q8192/full K
and a nonzero-offset padded Q257 tail with the pinned upstream oracle, including
finite +1/NaN rejection and fresh/reused changed-K/V checks. Run retained
upstream block checkpoints at the candidate group. Only a qualified candidate
advances to two fresh complete-block processes with AB and BA warmups and four
measured pairs in AB,BA,BA,AB order, using real BF16 block weights and F32 math.
All 187,432,960 output values must pass the original finite tolerance. Speed
acceptance requires at least 5% paired geometric latency reduction in each
process, both pair-order strata favoring the candidate, and no individual pair
regression beyond 2%. Diagnostic collectors stay off in timed blocks.

Use unchanged cache-aware 15% reserve and pressure checks, 64 MiB swap-growth
limit, 5 GiB/120 s primitive and 12 GiB/600 s full-block caps. One owned heavy
process runs at a time. No full model, Studio or video generation. Retain frozen
source/build/oracle/command pins and complete guard lifetimes in
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/smaller-head-groups-20261022/`.
Evidence decays on scoped source, artifact, hardware/OS/SDK/driver, shape, dtype,
queue policy or experiment-order changes; refresh matched native runs before
transferring a speed claim. This is ordinary independent-head scheduling,
without approximation or an LTP/WBA claim.

### Native STEEL F32 attention discriminator (2026-10-22)

The smaller-head probes reject further subdivision: at Q8192/full K36608,
G1 and G2 lose to G4 in all three paired samples. Neither advances to the
complete-block gate. This changes the next experiment from scheduling the same
MPS executable to a different native tiled implementation.

The candidate ports the MIT-licensed MLX STEEL attention specialization for
F32, D128, BQ32/BK16, four SIMD groups. Freeze the upstream revision, license
and transitive shader sources before adapting them. Use the existing native
Metal compiler in safe math mode, with no MLX runtime or reduced precision in
the measured path. Preserve full K/V, scale, unmasked noncausal attention and
F32 intermediate/accumulation types. GPU barriers and ordered dispatches are
permitted; no additional host wait, activation readback, or encode-time
MetalBuffer allocation is permitted. No speed or occupancy claim follows from
the source's tile size alone.

The first discriminator is the same pinned full-key upstream fixture at Q128,
Q8192 and the offset Q257 tail, with NaN-poisoned outputs, finite +1/NaN
negative controls and fresh/reused changed-K/V tests. The candidate must pass
the existing finite `2e-5+2e-5*abs(reference)` gate against both upstream and
G4/C128, and beat G4 in all three Q8192 pairs before full-block integration.
Do not hide layout/preparation work outside a timed arm. Strongest failure
modes are a stride/parameter ABI mismatch, an unwritten tail, changed reduction
error on real projected inputs, or a long dispatch missing the existing bounded
submission policy. Validate capability and bound native dispatch work before
large allocations.

Only a surviving primitive advances to the real BF16-weight full block with
F32 activations. Use the preceding two-process AB/BA warmup, balanced four-pair,
5%-per-process speed gate and original finite numerical tolerance. Intermediate
upstream checkpoints remain required. Keep the cache-aware 15% reserve,
64 MiB swap-growth limit, one heavy process and bounded footprint/deadline.
Production defaults remain unchanged until all gates pass. Rollback removes
the opt-in candidate and leaves the existing MPS workspace intact. Full model
and video generation remain stopped. Frozen external evidence lives under
`wan-self-attention/steel-f32-20261022/`; scoped source/oracle/build/driver,
geometry, dtype or queue-policy changes require refreshed matched evidence.

The first native specialization passes Q128/Q8192 and the Q257 offset tail,
including unchanged finite upstream comparisons, NaN/+1 rejection and changed
K/V fresh/reused checks, but loses all three Q8192 timing pairs to G4/C128
(2.292-2.304 s versus 1.763-1.786 s, including K/V preparation). It does not
advance to the complete block. The next single discriminator is BQ32/BK32,
WM4/WN2: eight SIMD groups keep Q in registers, reduce shared storage to
18,436 bytes and halve the full-key loop count from 2,288 to 1,144. The score
exchange adds barriers, so this is a hypothesis rather than a predicted win.
BQ64/WM4 is rejected by upstream's `TQ == 1` static assertion. Keep the same
80-group dispatch bound, strict numerical/negative/replay controls, Q8192
three-pair primitive gate and complete-block gate. Do not sweep further tile
shapes without new evidence about the limiting resource.

The Q-register/BK32 variant also loses all three Q8192 pairs (2.200-2.273 s
versus 1.760-1.800 s), while passing the unchanged upstream and replay gates.
It does not advance. Source review identifies a separate scheduling mismatch:
the native flattened grid traverses all forty heads within each query tile,
whereas MPS G4 traverses all query chunks within each four-head group. Their
nominal K/V working sets are respectively 1,499,463,680 and 149,946,368 bytes;
these byte counts do not establish cache residency or traffic.

The next discriminator changes only this native tile-index permutation to
four-head groups. BQ32/BK32/WM4/WN2, per-row arithmetic/reduction, all key
coverage, buffers, 80-group dispatch bound and chain policy remain unchanged.
Exhaustive CPU index controls must prove a bijection for Q128, Q8192 and the
Q257 padded tail, including dispatch boundaries. Repeat the same strict
upstream/negative/changed-KV controls and three-pair primitive gate before any
complete-block integration. A failure rejects this schedule; no cache or
bandwidth mechanism is inferred from latency alone. Rollback selects the
preceding frozen shader; production remains untouched. This is a loop-order
experiment rather than another tile-geometry sweep.

The four-head native permutation also fails the primitive speed gate:
2.143-2.161 s versus 1.748-1.766 s in all three Q8192 pairs. Numerical,
negative and fresh/reused changed-K/V gates pass. Cross-process differences
between native permutations do not isolate a cache effect. No native STEEL
candidate is admitted to the complete block.

The next MPS discriminator keeps G4 and tests C256 against C128. Earlier
balanced complete-block comparisons cover C128 versus C1024, but do not
interpolate kernel performance at C256. This is a different balance of
query-row parallelism and per-call work with the existing F32 SDPA executable;
neither fewer calls nor an intermediate chunk size implies a win. Keep K/V
preparation and all gather/scatter work inside each arm. The same Q128,
Q8192 and offset Q257 upstream/negative/changed-KV controls, all-three primitive
speed gate, 15% reserve and two-process complete-block gate apply. Reject a
slower primitive before any full-block run. No engine implementation or
production default changes are needed for this experiment.

C256 also fails the primitive gate: Q8192 paired wall times are
1.7053/1.7086/1.7080 s versus C128 1.7059/1.7171/1.7058 s. The third pair
loses, and the aggregate difference is below 0.5%. Q128 and the offset Q257
tail pass numerical/replay checks but are slower. No complete-block run is
admitted for this candidate.

The next distinct discriminator continues the native BQ32/BK32 online softmax
across 2,048-key shards. For each four-head group, visit all query tiles before
advancing to the next key shard. Keep unnormalized F32 output, row maxima and
row sums in preallocated GPU buffers between ordered dispatches; normalize only
after the final shard. Each row retains ascending key order and the preceding
kernel's arithmetic. The nominal active G4 K/V slice falls from 149,946,368 to
8,388,608 bytes; cache residency and a speed benefit remain hypotheses.

The first falsifier is bit identity with the frozen unsharded native v4 shader,
including changed K/V and reused workspaces, in addition to the unchanged
finite upstream gate. A split must neither reload normalized output nor merge
independently normalized contexts. Bound each dispatch by the preceding
80-group full-key work budget, preserve query/key tails, and allocate the two
row-state arrays at setup. The same terminal wait, zero encode allocations,
15% reserve, primitive three-pair gate and complete-block gate still apply.
Any bit discrepancy or primitive loss rejects this formulation. Rollback
selects the frozen unsharded shader; no production implementation changes.

The 2,048-key continuation passes bit identity with v4 for Q128, Q8192 and
the offset Q257 tail, including fresh/reused changed K/V. Against current
MPS G4/C128, Q8192 wins all three pairs: 1.4925/1.4988/1.4928 s versus
1.7752/1.7901/1.7869 s. Q128 and Q257 are slower because the fixed 180 roots
dominate short inputs. This admits only the long-query complete-block probe;
it does not justify selecting the native path for short queries.

Use a probe-only subclass of the block's existing workspace type. It retains
the inherited unused MPS plans and their conservative memory reserve, but
reports zero executed Graph executables for the native arm. The native path
uses the same in-chain K/V transpose as its primitive probe; both arms include
preparation. Allocate 11,714,560 additional row-state bytes at setup. Restrict
the benchmark option to unprofiled G4/C128 comparisons and retain the preceding
strict upstream checkpoint gate, compact scratch, balanced two-process speed
gate and 12 GiB/600 s process cap. No production type or default changes.
The callback adapter records the active chain before yielding to a fused K/V
writer, so a partially encoded failing callback cannot bypass release guards;
the timed comparison uses ordinary preparation on both arms.

### Compensated native long-key attention (2026-10-22)

The first continuation shader (v5) fails the real-weight complete-block warmup:
14,162 of 187,432,960 final values exceed the unchanged tolerance. Its passing
selected-query upstream fixture is insufficient. A diagnostic-only invocation
compares attention before projection on the GPU, consuming A's saved attention
before B overwrites the aliased final output. It finds 92,928 attention
violations. The callback adds no host wait or allocation; diagnostic comparisons
are excluded from normal timed arms. This localizes the failure before the
output projection without reading activations back to the CPU.

Two isolated discriminators are rejected. Translating V by its first row
(v6), although an exact-arithmetic identity, fails the strict Q8192 upstream
fixture. Compensating only the online-softmax denominator (v7) leaves 92,878
attention violations and 14,101 final-output violations. Neither advances to
normal complete-block timing. Retain these failures rather than widening the
tolerance or attributing them to downstream amplification alone.

The surviving shader (v8) compensates the **PV numerator**. Form each BK32
contribution in an additional F32 output fragment, then Kahan-merge it into the
running output with a second correction fragment. Scale output and correction
by the same online-maximum factor and fold the correction before each shard
spill or final normalization. QK, exponentials, denominator, input types and
tile/shard/head scheduling stay unchanged. This adds 32 F32 scalar slots per
thread, with no additional GPU buffers, dispatches, host waits or encode-time
allocations. It is ordinary numerical stabilization, without an LTP/WBA claim.

The full-size diagnostic now passes all 187,432,960 attention values, maximum
absolute difference `9.775162e-6`, maximum tolerance-normalized error
`0.26414144`. The final output also passes every value, maximum absolute
difference `2.0980835e-5`, maximum normalized error `0.2650215`. This isolates
PV accumulation as the main numerical cause on this input; it does not prove
universal stability of uncompensated inner MMA products or all model layers.

The Q8192/full-K36608 primitive passes the pinned upstream, finite/+1/NaN and
changed-K/V fresh/reused controls. Its three paired times, including K/V
preparation, are 1.5075/1.5069/1.5073 s versus MPS G4/C128
1.7401/1.7526/1.7267 s. The nonzero-offset padded Q257 tail also passes.
Retained real-width upstream block checkpoints pass with ordinary and fused
K/V preparation: seven examples, zero failures/errors per invocation, with
repeat/chained/replay and fused K/V checks. This retained upstream block has
short spatial length; no full-spatial real-weight upstream oracle was run.
The full-size comparisons below use the currently qualified MPS path.

Two unprofiled fresh processes use opposite warmups, four measured pairs each
in AB,BA,BA,AB order, BF16 weights, F32 activations/math, compact scratch and
ordinary K/V preparation inside **both** timed arms. A is MPS G4/C128; B is the
native 2,048-key-shard candidate. Inputs are fixed synthetic activations with
real block-0 weights. Each warmup and measured pair compares every
final value using finite `2e-5 + 2e-5*abs(reference)`; all pass. The injected
finite error and NaN are rejected. No root/API collector is enabled.

| Warmup | A paired seconds | B paired seconds | Geometric time reduction | AB / BA geometric B:A |
| --- | --- | --- | --- | --- |
| AB | 11.2439, 12.6471, 15.3274, 18.0915 | 10.2304, 10.3111, 10.8036, 15.8035 | 17.7911% | 0.89151 / 0.75807 |
| BA | 29.3237, 26.2276, 22.5117, 22.1842 | 25.2888, 21.8626, 18.7452, 16.7369 | 18.0231% | 0.80662 / 0.83313 |

All eight pairs favor B; both processes exceed the predeclared 5% geometric
reduction, both order strata favor B, and no pair regresses. Absolute wall time
drifts strongly within and between processes. Balanced ordering gives a scoped
relative result, not a quiet-host or stable-latency certificate. Read-only host
inspection observed unrelated compilation/indexing; `pmset` recorded no thermal
warning, which does not exclude GPU throttling or establish the drift's cause.

Every timed arm reports one terminal wait, zero encode-time MetalBuffer
allocations, zero inter-operator host waits and zero activation readbacks.
Numerical reductions/readback run outside timed arms. Peak sampled group
footprint is 10.533 / 10.467 GiB, minimum cache-aware headroom
22.189% / 22.176%, minimum pressure-free 66% / 64%, swap growth zero. Both
12 GiB / 600 s guards finish with exit zero and no live group members; maximum
sample gaps are 0.607 / 1.989 s. Sampled guards do not prevent between-poll spikes.
No model, Studio or full video generation was started.

The [checked-in shader](../examples/support/kernels/wan_steel_f32_d128.metal)
is byte-identical to the measured external v8 input, SHA-256
`b3e27825b3723d80789e6b73acecff0a2a4e66380691d18dce827a3cf3d8664f`.
Its [provenance and MIT license](../examples/support/kernels/wan_steel_f32_d128.md)
keep the MLX revision and local modifications explicit. Historical receipts
name the external file; packaging it does not claim another measurement.
Keep the native adapter opt-in: short queries lose to MPS, the probe retains
unused inherited MPS plans, and a production pipeline cache/release policy,
other blocks and full-model quality remain outside this certificate.

DoD evidence is under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/steel-f32-20261022/`:
`build-intent-v7.json` / `build-receipt-v7.json`,
`primitive-pv-compensation-mps-q8192-v1`,
`primitive-pv-compensation-mps-q257-v1`,
`upstream-pv-compensation-short-v1`,
`upstream-pv-compensation-short-fused-v1`,
`full-block-pv-compensation-diagnostic-v1`,
`full-block-pv-compensation-ab-v1`, and `full-block-pv-compensation-ba-v1`.
Each stage retains command, stdout, exit, telemetry and complete guard lifetime.
The final report verifies frozen source/build/oracle/shader/command pins as
well as numerical, ordering, synchronization, speed and resource predicates.
Its name is `pv-compensation-results-v2.report.json`, SHA-256
`7aae57fb0c7f71806d211bc0fe44b69b3471ba1d6808cf6be8c240f40caa61dd`.
The CPU-only verifier, `verify-pv-compensation-results-v2.py`, has SHA-256
`612a2e9ebfe5bd4e67cae3a82f7202037cca8341c7a4b384d772ccfd4689bcc2`.
Refresh with:

```sh
python3 /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-self-attention/steel-f32-20261022/verify-pv-compensation-results-v2.py --check-only
```

Expected signal is `RESULT ROBUST qualification_gate_passed=True` and exit zero.
V1's exact report refresh exposed Python 3.11/3.15 differences in the last bit
of the derived geometric mean. V2 uses `math.fsum`, passes exact refresh on both
interpreters, and preserves all raw measurements, full-precision gates and
tensor tolerances. Neither report nor historical evidence was overwritten.
The verifier reconstructs historical source bytes; separately check current
scoped sources against the v7 intent before transferring this result to a new
checkout. Hashes establish file identity, not runtime attestation or independent
recomputation of the emitted GPU metrics.

For a new matched timing series, first rerun the upstream/primitive gates, then
run the built full-block benchmark under the existing cache-aware 15% guard
with a unique log path, `--max-footprint-mib=12288`,
`--max-swap-growth-mib=64`, `--timeout-sec=600`, and:

```sh
wan_s2v_full_block_benchmark \
  --weights=/path/to/wan-official.weights.safetensors \
  --comparison=head-groups --baseline-self-chunk=128 \
  --baseline-heads-per-group=4 --candidate-self-chunk=128 \
  --candidate-heads-per-group=4 --pairs=4 --reuse-block-scratch \
  --candidate-steel-source=examples/support/kernels/wan_steel_f32_d128.metal \
  --warmup-order=AB
```

Repeat in a fresh process with `--warmup-order=BA`. Pin/check the bytes of the
file actually compiled, not just its path. Do not enable diagnostics in timing
runs. Expected signal is exit zero, all finite strict comparisons, unchanged
wait/allocation counts, complete clean guards and the same predeclared speed
gate. The MiniQuad adversary verdict is ROBUST for the measured probe scope;
the reviews are same-lineage and correlated, supported by source/hash checks
and executed numerical/negative/paired controls. OS/SDK/driver/device, shader,
geometry, dtype or resource/queue-policy changes require fresh numerical and
matched timing evidence. Scoped artifact/source loss requires integrity refresh;
a failure reopens dependent claims. Rollback removes this opt-in adapter and
its packaged source; existing engine defaults remain unchanged.

## Reduced-step short-clip experiment (2026-10-22, in progress)

The requested comparison is 8/12/16 complete UniPC schedules against a
40-step native control. This is a quality/performance experiment; it does not
change engine defaults or relax the existing numerical tolerances. Recompute
the shifted schedule for every count. Truncating the first N steps of the
40-step schedule is rejected. Compare every sigma, timestep, guidance,
converted prediction, corrected sample and predicted sample to the pinned
upstream scheduler at each requested count, including 40.

Use a separate frozen experimental runner because the historical pooled
runner admits only five seconds at 40 steps. The short contract is two seconds,
32 delivered frames at 16 fps, 256x448 canvas/delivery, 36 audio-conditioning
frames, nine noisy latent frames at 32x56, plus one reference latent frame.
The VAE decodes 37 frames, drops the four motion-prefix frames, and delivers
the first 32 of the remaining 33. All counts share the same exact reference,
audio, text tokens, initial-noise bytes, checkpoint files, CFG 4.5, shift 3,
BF16 weight storage, F32 computation and Metal/MPS backends. Input decoding,
resizing and schedule setup are preprocessing; all neural inference stays on
GPU with the existing terminal completion/readback boundary.

CAUTION: the highest-risk confounds are an incomplete schedule, mismatched
temporal geometry, different random inputs, numerical drift mistaken for a
step-count effect, and host queueing mistaken for completed GPU time. Keep the
historical production admission checks intact. The separate experiment must
identify its source snapshot and the two existing finite native audio/stack
drift ledgers; the native 40-step result is a relative control, not a claim of
full upstream-model parity. Preserve the exact algorithm as the default.

Admission first requires a header-only pool/state/reserve estimate and the
unchanged cache-aware 15%-of-physical-memory floor, pressure check, process cap,
swap-growth limit and stop deadline. Run at most one owned heavy process. No
Studio or full five-second generation is authorized by this experiment.
Rollback removes the external snapshot and the opt-in experiment only.

DoD: fresh pinned-oracle/native scheduler gates at 8/12/16/40 with zero pending
examples and the unchanged `2e-5 + 2e-5*abs(reference)` finite gate; four
completed videos with identical input hashes; timings measured through actual
terminal completion, separated from setup/encoding/export; complete clean
guard lifetimes; and blinded review of lip sync, identity, motion and temporal
artifacts. Pixel similarity to 40 steps alone is not perceptual quality.
Expected denoiser-only upper bounds from call count are 5x, 3.33x and 2.5x;
end-to-end savings must be measured. One clip/one noise realization cannot
establish a general quality threshold. Missing memory or outputs leaves this
experiment in progress, with no speed or quality result. Refresh after source,
geometry, input, checkpoint, scheduler, backend, device or runtime changes.

### Quantized-weight memory branch (2026-10-22, feasibility only)

The BF16 experiment cannot currently be admitted: the sampled cache-aware
available memory was 20.85 GiB, while the DiT weight pool alone is 30.18 GiB.
The user cannot free the requested background memory and asked about QBit
Gaussian conditional reconstruction or Q4/Q5 instead. Keep the reduced-step
comparison as the return frontier; no model was launched.

A fresh header-only calculation covers all 1,248 native DiT bindings and all
242 T5 tensors. The proposed policy quantizes only rank-two block/injector
linear weights, retaining token/position embeddings, time/conditioning/output
projections, convolutions, modulation, norms and biases in original BF16.
That covers 98.381% of DiT elements and 81.510% of T5 elements.

| Proposed resident storage | DiT weight pool, GiB | T5 weight pool, GiB |
| --- | --- | --- |
| Existing BF16 | 30.181 | 10.582 |
| Q4_K, 4.5 bits/quantized weight | 8.843 | 4.383 |
| Q5_K, 5.5 bits/quantized weight | 10.698 | 4.922 |
| QBit p4, 4.25 bits/quantized weight | 8.379 | 4.248 |
| QBit p5, 5.25 bits/quantized weight | 10.234 | 4.787 |

These estimates include block scales/moments, protected BF16 tensors and
per-buffer page rounding. They exclude a new container's metadata, any future
outlier sidecars, activation/scratch storage, retained inputs, MPS expansion
buffers and opaque framework allocations. Serial phases alias their backing;
DiT and T5 weights are not added together. Quantizing DiT alone leaves T5 as
another possible peak. None of these rows is a memory-admission receipt.
Evidence is `quantization-header-estimates.json` in the experiment evidence
directory above, SHA-256
`2a1bbfb0ad8fddf21f99cc4491da482dd8ed2070acbc276119f6d86f7311a3f3`.

Native video weights currently accept only F32/BF16. The reusable MPS F32
matrix-expansion stage is an integration seam for GPU packed-weight decoding;
it does not yet accept QBit/Q4/Q5. QBit's local scalar codec has the p4/p5
conditional-mean tables and eight-byte F32 mean/sigma metadata per block.
The codec's local p4 minimum is not a fundamental QBit limit. Keep activation
quantization as a separate branch: the weight reduction is the first target,
and prior attention precision gates are not waived.

Next falsifier: one real full-size projection/block, with identical resident
inputs, comparing BF16 against GPU-decoded Q4/Q5/QBit weights. Measure output
error and complete GPU time, including decode, before choosing a format.
Kernel parity must retain its existing tolerances against the same decoded
weights; lossy model quality is a separate experiment and must not be presented
as exact BF16/upstream parity. Preserve the BF16 control and the 15% guard.

### Packed projection probe contract (2026-10-22, design sealed)

The next admitted experiment is one real checkpoint projection,
`blocks.0.ffn.0.weight` (13824 x 5120), with its original bias and identical
resident inputs across BF16, canonical Q4_K, QBit p4 and QBit p5. Use the
short-clip token count (4480 rows) for performance; qualify whether inputs are
captured activations or controlled synthetic values explicitly. No complete
block/model/video inference or activation quantization is admitted here.

CAUTION rollback is deletion of the opt-in benchmark and decode helper;
existing weight types, loaders and engine defaults remain unchanged. Offline
checkpoint conversion is preprocessing and excluded from the projection
timing. Timed arms contain GPU weight expansion/decode followed immediately by
the same existing F32 MPS projection and bias operation, using one persistent
F32 weight stage. Do not add per-operator waits, activation readbacks, or
encode-time MetalBuffer allocations. Each independent sample has its existing
terminal completion boundary. Warm all four arms, rotate their order in both
directions, and report completed GPU and host wall time, not encode time alone.

DoD: independently prepared packed payloads and torch-MPS reference traces;
GPU unpack and projection comparison at the unchanged finite
`2e-5 + 2e-5*abs(reference)` gate; a seeded corruption that the comparator
rejects; full-size timing including decode; explicit packed/stage/activation
bytes; and clean sampled guard lifetimes with a 15% cache-aware physical
reserve and 64 MiB swap-growth stop. Only one owned heavy process may run.
Source/fixture/binary/command hashes and explicit memory budgets bind evidence.
The strongest failure mode is a valid format decoder incorrectly promoted to
quality parity: report quantized-vs-BF16 distortion separately, never widen
the decoder/kernel gate. Another falsifier is decode overhead erasing the
storage benefit in completed projection time. A memory-only result is valid
but cannot be called acceleration. Any failed gate keeps the path opt-in and
unqualified. Refresh for weight/input/geometry/kernel/backend/runtime changes.

### Packed projection results (2026-10-22)

The bounded comparison completed on Apple M2 Max (64 GiB). The real BF16
checkpoint matrix is `blocks.0.ffn.0.weight`, `[13824, 5120]`, with its original
bias. The four upstream-captured, seeded FFN input rows were periodically tiled
on GPU to 4480 rows. This exercises the short-clip projection geometry with
only four distinct inputs; it is not a capture from a generated video.

All four labels below describe **weight storage**, not GEMM arithmetic.
Every timed projection expands/decodes the entire matrix on GPU into the same
persistent 270 MiB F32 stage, then invokes the same F32 MPS matmul and bias.
Offline archive packing, disk IO, pipeline compilation, reference loading,
and validation are outside the timed arms. Projection time includes decode.

| Stored weights | Matrix MiB | Decode GPU ms | Decode + projection GPU ms | Completed wall ms | Output relative L2 vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 135.00 | 1.111 | 56.536 | 56.792 | 0 |
| Q4_K | 37.97 | 2.246 | 57.578 | 57.842 | 4.632% |
| QBit p4 | 35.86 | 2.305 | 57.625 | 57.896 | 11.642% |
| QBit p5 | 44.30 | 2.259 | 57.070 | 57.313 | 6.965% |

Times are medians of eight measured samples per format and mode, after two
warmup rounds. Both forward and reverse rotations give each format two
appearances at each of the four order positions. Projection GPU ranges were
56.202-56.564 / 56.892-57.898 / 56.977-57.896 / 56.698-57.514 ms in table order.
Every paired quantized/BF16 projection was slower in this run. The packed
payloads are smaller; this probe does not establish net model-resident memory
savings or faster projection through the unpack-to-F32 path. Do not rank the
small timing differences between quantized formats as durable.
The common F32 stage is additional memory, not part of the packed matrix size.

Before and after timing, each format's 70,778,880 decoded weights and
61,931,520 projection outputs matched its independently prepared same-format
torch-MPS reference **bit for bit**. The finite `2e-5 + 2e-5*abs(reference)`
gate was unchanged. All six invalid encode requests were rejected, and the
comparator detected both a seeded +1 error and a NaN. There were no
inter-operator host waits, activation readbacks, or encode-time MetalBuffer
allocations. The existing terminal completion boundary remains per sample.

Those exactness checks do not erase lossy compression error. Relative L2 in
the table compares each quantized projection to the native BF16 control;
cosines are 0.99892745 / 0.99365403 / 0.99770158 for Q4_K / p4 / p5. Maximum
absolute errors are 0.458864 / 5.601690 / 4.144031. Q4_K had lower output error
than either QBit variant on these four inputs, while being smaller than p5.
The QBit producer also matched the existing public Crystal codec byte for byte
on 3072 real blocks at the beginning, middle, and end of the matrix per
precision; all 256 LUT entries per precision matched exactly. Gaussian
conditional reconstruction is implemented as specified, but this result does
not establish that its distribution assumptions fit Wan weights or activations.

The native guard exited cleanly: sampled peak group footprint 1.579 GiB,
minimum cache-aware physical headroom 25.973 GiB, minimum pressure-free 80%,
and zero swap growth. Explicit resident probe buffers were 1,679,990,844 bytes
(including all four stored matrices, references and metrics); opaque MPS
storage is excluded from that explicit count. The guard retained the 15%
reserve, 4 GiB process-group budget, 64 MiB swap-growth stop, and 240 s timeout.
The 0.5 s sampling interval does not certify an instantaneous memory peak.
Oracle preparation and the offline codec check also finished under their
separate guards. No model, Studio, full block, or video generation was run.

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-quantized-projection-20261022`.
`summary-v1.json` (SHA-256
`b9b6ad2466c02dd07ce5c2701cb77f6f03986146505649b4c38605d12a5d85a5`)
indexes the measurements, numerical checks and guard lifetimes. The successful
fixture is `fixture-4480-v3/manifest.json` (SHA-256
`0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703`).
`build-intent-v2.json`, `oracle-intent-v3.json`, and `native-intent-v1.json`
bind exact commands, source/object/binary hashes, inputs and budgets before
execution. `native-completion-v1.json` confirms exit 0, unchanged input hashes,
and the output/guard hashes. `qbit-producer-check-intent-v2.json` and its log
bind the existing-codec comparison. Earlier failed fixture preparation attempts
produced no timing evidence and were corrected before this run.

Correlated Mini-Quad source and run-level review returned ROBUST for this
bounded probe. The strongest surviving limitations are the four distinct
activation rows, sampled public-codec comparison (1.111% of blocks per
precision), and sampled memory telemetry. The result does not certify
model/video quality, instantaneous peak memory, or a general timing ranking.

Rerun the qualified native binary with the guard, from the worktree root:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B examples/support/wan_memory_guard.py \
  --memory-accounting=file-cache --max-footprint-mib=4096 \
  --max-swap-growth-mib=64 --timeout-sec=240 \
  --log=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-quantized-projection-20261022/native-guard-new.jsonl \
  -- /opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-quantized-projection-20261022/probe-v2 \
  --fixture=/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-quantized-projection-20261022/fixture-4480-v3 \
  --rounds=8
```

Decision: retain BF16 as the exact control and keep packed decoding opt-in;
engine weight types, loaders and defaults are unchanged. Q4_K is the strongest
memory/error candidate of this measured set, not an approved video-quality
format. Before model integration, test several projections with genuinely
captured denoising activations, including outliers and multiple timesteps;
keep compression distortion separate from same-format kernel parity. For
QBit, distinguish an empirical-distribution mismatch from activation-sensitive
error before changing centroids or adding an outlier residual. Any fused
packed GEMM must beat this complete decode-plus-GEMM control and re-pass the
same-format gate. The reduced-diffusion-step experiment remains the return
frontier after memory admission and model-quality checks; full generation
remains outside the current authorization.

### Gaussian-prefix diagnosis contract (2026-10-22)

The accepted continuation is to measure compression error on genuine denoising
activations at multiple timesteps. The scoped inventory found only scheduler
recurrence traces and the original noise/image/audio/text input pack, not saved
DiT activations or trajectory latents. Reaching later timesteps requires the
preceding full DiT evaluations; the current BF16 model is still outside the
admitted memory budget. Do not relabel the four seeded block rows as trajectory
data or start complete generation to obtain them.

Before changing the weight loader, admit a smaller discriminating experiment
on the existing real FFN matrix and four controlled input rows. Compare weight
error, standardized tail occupancy/error, and projection error for the original
formats. For QBit p4/p5, keep the payload fixed while replacing Gaussian prefix
centroids with the arithmetic mean of full-code levels, restore only original
weights with |z| >= 3 or 4, and independently reassign the original normalized
weights to the nearest existing Gaussian centroid. The last three arms are
diagnostic interventions, not production codecs or equal-storage quality claims.
Integrate the standard-normal scalar error for the existing intervals as a
separate mathematical control: conditional means optimize those fixed cells,
not every possible low-bit partition or the activation-weighted matrix loss.

CAUTION rollback removes the opt-in diagnostics and leaves the existing
benchmark, codecs, loaders and defaults unchanged. DoD: a release native probe
with independently prepared MPS reference outputs, existing finite
`2e-5 + 2e-5*abs(reference)` parity for every tested projection, positive error
controls, GPU-only reconstruction/reduction/projection within each arm, no
inter-operator host waits or activation readbacks, and guarded execution with
the existing 15% cache-aware reserve and 64 MiB swap-growth stop. Record source,
fixture, binary and command identity. Never infer video quality, a new codec's
storage cost or speed from ideal tail repair or these four input rows. The
strongest attribution failure is mistaking a useful oracle intervention for an
implementable equal-bit codec. Refresh for changed weights, inputs or kernels.

### Gaussian-prefix diagnosis result and adaptive-packing correction (2026-10-22)

The opt-in native diagnostic and separately prepared MPS oracle completed on
the same real 13824 x 5120 FFN matrix. All four original decoded matrices and
all twelve original/intervention projections are bit-exact to their same-format
references at the unchanged finite gate. GPU reductions have a full-matrix
known-answer control, one-ULP threshold-boundary controls, and seeded +1/NaN
rejection. Each native arm reconstructs, reduces and projects on GPU without an
inter-operator host wait, activation readback, or encode-time MetalBuffer
allocation. This diagnostic makes no timing claim.

Gaussian conditional means reduce weight squared error versus arithmetic means
of the retained full-code levels by 33.49% at p4 and 13.82% at p5. The centroid
formula is therefore not the observed failure. Even a standard-normal source
has relative scalar RMS errors of 13.31% and 7.41% for these fixed p4/p5 prefix
cells. Conditional means optimize reconstruction within those fixed cells;
they do not make the cells an optimal low-bit partition or minimize the
activation-weighted projection loss.

Only 0.2331% of weights have standardized magnitude at least 3, yet they account
for 33.38%/43.12% of p4/p5 weight squared error. Restoring those weights reduces
projection squared error by 54.55%/63.37% on the four distinct test rows. The
p5 restored-tail output relative L2 is 4.215%, versus uniform p5 6.965% and
Q4_K 4.632%. Original BF16-derived weights are available free to this diagnostic
intervention; these numbers do not establish a real codec's storage or speed.
The raw fourth standardized moment is 2.787, so rare extreme sensitivity does
not justify calling the overall weight distribution heavy-tailed.

The user correctly identified a missing comparison: fixed p4/p5 for an entire
matrix does not test error-dependent dynamic QBit precision selection. The
uniform-format ranking above is not a verdict on adaptive QBit. Locate and
verify the existing selector and physical layout before reusing them for Wan
weights; keep KV/state adaptation, weight packing, and activation-sensitive
quality selection separate. Scalar weight error can improve while projection
error worsens, as the p5 Gaussian/arithmetic control also demonstrates.

Evidence root: external `wan-gaussian-diagnostics-20261022/` under the same
video evidence base. `summary-v1.json` SHA256
`ad6eaadbb788c4d8c5b01b4999113b7aab62d8aebdacffa87ebd52ff6e8753ed`
indexes `build-v4`, `oracle-v1`, `native-v4`, immutable `fixture-v1`, and the
source/binary/fixture-bound intents and unchanged-input completion receipts.
Native explicit buffers total 1,979,917,500 bytes; sampled process peak is
1.858 GiB, minimum cache-aware headroom 32.67 GiB, minimum pressure-free
reading 77%, and swap growth zero. All owned process groups finished normally.
Correlated Mini-Quad review returned ROBUST for this fixture's same-format
parity and statistics, VULNERABLE for model-quality generalization. There are
four distinct seeded input rows, repeated 1120 times, not denoising trajectory
activations. Loaders and defaults remain unchanged; no full generation ran.

### Reuse the existing adaptive-weight falsifier (2026-10-22)

Live source confirms the user's recollection: `bin/qwen35_q6_adaptive_weight_probe.cr`
is an adaptive per-256-value weight-policy prototype. It selects p4 when
`max(abs(decoded-original))/block_sigma <= 0.20`, otherwise p5 at `<= 0.10`,
otherwise original Q6. It forecasts sizes and tests mixed operator outputs;
it does not write or consume a production mixed-tier weight container.
The original Qwen default selected only Q6, and the tested compressed policies
failed their quality gates (`docs/qwen-qbit-cache-frontier.md`, adaptive weight
probe section). This is a transfer hypothesis, not a validated Wan policy.

The next bounded falsifier applies that exact selector to Wan's existing
Gaussian p4/p5 candidates, falling back to the original BF16-derived weights.
Two explicit diagnostic policies use thresholds 0.50/0.50 and 1.00/0.50.
These are selector parameters, not changes to the unchanged same-format
upstream parity gate. Selection, reconstruction, reductions and projection
stay on GPU; tier bytes and projections are checked against an independent
offline MPS reference, with inclusive-boundary, zero-sigma and nonfinite
selector controls. No compact mixed payload is built or timed. Charge the old
weight prototype's four metadata bytes per block in the size forecast, but do
not claim it accounts for a complete future address/index layout.

DoD: release build, exact tier agreement, finite projection parity at
`2e-5 + 2e-5*abs(reference)`, selector controls, zero encode allocations and
inter-operator host waits, and the same 15% reserve/64 MiB swap-growth guard.
The strongest failure is interpreting an error-threshold policy or a storage
forecast as an implemented, quality-approved packed model. Rollback removes
only these opt-in diagnostics; production codecs/loaders/defaults stay fixed.

The bounded GPU transfer passed: all 276480 selected tiers agree exactly with
the MPS oracle in every policy, and all three projections are bit-exact to
their same-format references. Twelve selector controls per policy cover
adjacent Float32 threshold values, p4 precedence, zero sigma and nonfinite
inputs. The twelve original/ablation arms and corruption/statistics controls
also re-pass in the same release executable. There are no inter-operator host
waits, activation readbacks, or tracked encode allocations in the native path.
The offline oracle has host waits and exports tier vectors and four output
rows to immutable reference files; it is outside that native-path claim.
Correlated Mini-Quad review returned ROBUST for selector transfer and
same-fixture parity, VULNERABLE for model-quality generalization, compact-format
completeness and speed.

| p4 / p5 thresholds | p4 blocks | p5 blocks | BF16 blocks | Forecast bits/weight | Output relative L2 vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.20 / 0.10 | 0 | 0 | 276480 | 16.125 | 0.000% |
| 0.50 / 0.50 | 60049 | 130495 | 85936 | 8.499 | 4.168% |
| 1.00 / 0.50 | 186715 | 3829 | 85936 | 8.041 | 6.153% |

The original selector preserves this matrix exactly by rejecting compression.
The 0.50/0.50 policy reduces error compared with uniform p5 (6.965%), but
its forecast costs 8.499 rather than 5.25 bits/weight. The whole-block BF16
escape applies to 31.082% of blocks. This is evidence for a size/error tradeoff
on four inputs, not an equal-byte win over p5 or Q4_K (4.5 bits/weight), an
implemented mixed format, or model/video quality. Next discriminate smaller
normalization groups, intermediate precision tiers, or sparse escapes at a
matched complete storage budget, then validate on real denoising activations.

External `adaptive-summary-v1.json` SHA256
`73aa0d27ef76cc981dff524655d2ddfa7fd8f05dceb3e5b83d6a38888f7f9efd`
indexes the source-bound `build-v6`, `adaptive-oracle-v1`, and
`adaptive-native-v6` receipts plus immutable `adaptive-fixture-v1`.
Native sampled peak footprint is 1.858 GiB, minimum cache-aware headroom
29.944 GiB, minimum pressure-free reading 78%, swap growth zero, and no owned
process remains. No full model or video generation ran. Native manifest
admission also binds both oracle scripts to current source hashes; the command
receipts establish their execution, rather than relying on manifest assertions.

CPU integrity/guard/result refresh (does not run a model):

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-gaussian-diagnostics-20261022/summarize-adaptive-v1.py
```

The fixture-aware result decays after relevant source, weight, activation,
shape or backend changes. Repeat guarded numerical checks before transferring
it; quality/generalization needs new trajectory inputs, and speed needs an
actual packed consumer and matched complete-path timings.

### Positional-correction contract (2026-10-22)

Implemented opt-in experiment: consume the pinned real Wan FFN matrix
and the existing four seeded input rows. For each 256/512/1024-value packing
tile, consider Gaussian p4 and p5 plus every weight whose normalized absolute
reconstruction error exceeds the declared policy, and an original-BF16 escape.
Normalization remains per 256 values. Choose the smallest complete tile cost
under each fixed error policy; ties prefer BF16, then p5, then p4, so a sidecar
must be strictly cheaper than the next viable precision. A common prefix-offset
table, tile header,
position widths, values and alignment all count. Wider packing tiles are an
experiment in metadata amortization, not a change to quantization statistics.

The bounded raw-buffer representation is a UInt32 byte-offset table (tiles+1)
and four-byte-aligned variable records. Shape, tile size and the reconstruction
LUTs are external descriptors; this is not a standalone file container. Report
blob-plus-offset bytes explicitly, excluding any future global container header
or separately resident LUTs. A record starts with a four-byte header:
precision byte (4/5/16), value-mode byte (0 BF16 absolute / 1 F16 delta), and
UInt16 correction count. Quantized records retain all original mean/sigma and
bit planes, followed by increasing local positions (UInt8 for 256, UInt16 for
512/1024) and two-byte values. Raw BF16 records have no corrections. Padding
is zero. Selected positions and base precision are identical between the two
value modes. F16 rounding may miss the reconstruction policy; report this
separately instead of changing the mask, size or parity gate.

Build plans, scan offsets, pack records, reconstruct and project on GPU with
caller-owned buffers and one existing terminal completion boundary per arm.
The packing workspace may reserve worst-case capacity: distinguish its actual
allocation from the logical packed length. Do not infer model memory admission
from logical bytes alone. Initially decode to the existing reusable F32 stage;
tile-fused GEMM and model loader integration remain outside this slice.

CAUTION rollback removes only opt-in codec/probe additions. DoD: release build;
exact layout/selection agreement with an independently prepared oracle;
same-format decoded-weight and projection parity at the unchanged finite
`2e-5 + 2e-5*abs(reference)` gate; complete byte accounting; malformed-header,
offset, capacity, position, ordering and nonfinite controls; seeded comparator
failure; zero tracked encode allocations/inter-operator host waits/activation
readbacks; guarded native and oracle runs with the 15% cache-aware reserve,
64 MiB swap-growth limit and explicit process/time budgets. Keep compression
distortion versus original BF16 separate from kernel parity. Timings include
decode plus projection and exclude offline packing, with matched BF16 control;
packing cost is reported separately if measured. No full generation is allowed.

The strongest failure is an apparently compact format whose sidecars, capacity
or reconstruction rounding are omitted, or whose scalar policy is promoted to
video quality. Correlated Mini-Quad review must attack these boundaries. Evidence
decays with relevant source/weights/inputs/layout/backend changes. The return
frontier remains short-fragment diffusion-step quality/speed testing after
model memory admission and trajectory-quality checks.

#### Positional results and scope

The GPU packer, independent offline MPS oracle and native consumer agree
bit-for-bit for plans, offsets, packed bytes, all 70,778,880 reconstructed
weights and all 61,931,520 projection outputs in each of 18 arms. The native
projection executes 4480 rows, using four distinct seeded upstream rows tiled
1120 times on GPU. This exercises the real 13824 x 5120 weight matrix and
full projection shape; it does not provide denoising-trajectory coverage.
The offline oracle exports references outside the timed native path.

The table reports BF16 replacements. Equal-size F16-delta arms choose exactly
the same positions, tiers and bytes; their projection relative-L2 differences
are below 0.000006 percentage points. No F16 post-rounding policy misses occur
on this fixture. Relative L2 versus original BF16 is compression distortion,
separate from the unchanged same-format implementation parity gate.

| Packing values | Policy | Blob + offsets bytes | Bits/weight | Corrections | Projection relative L2 vs BF16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.10 | 67,321,612 | 7.609232 | 6,082,928 | 2.276470% |
| 256 | 0.20 | 53,942,720 | 6.097041 | 2,014,579 | 3.551548% |
| 256 | 0.50 | 41,531,368 | 4.694210 | 459,420 | 6.954429% |
| 512 | 0.10 | 71,883,764 | 8.124883 | 6,082,300 | 2.275129% |
| 512 | 0.20 | 54,434,996 | 6.152682 | 1,742,908 | 3.285557% |
| 512 | 0.50 | 40,544,924 | 4.582714 | 459,430 | 6.954487% |
| 1024 | 0.10 | 71,330,804 | 8.062383 | 6,082,300 | 2.275129% |
| 1024 | 0.20 | 53,895,064 | 6.091655 | 1,727,077 | 3.264850% |
| 1024 | 0.50 | 39,991,964 | 4.520214 | 459,430 | 6.954487% |

Wider records amortize headers/offsets but require two-byte positions rather
than one. They therefore lose at policy 0.10, where many corrections are
needed. At policy 0.50, 1024-value records save 3.71% against 256-value records.
Normalization groups stay 256. A fixed scalar policy does not fix projection
error, and this result does not beat the earlier Q4_K size/error point
(4.5 bits/weight, 4.632% projection relative L2). No format is promoted for
model/video quality. F16 deltas show no measured equal-byte quality advantage.

Logical compression is also not deployed resident-memory compression. The
probe reserves 142,663,680 bytes of blob capacity plus offsets, maintains the
270 MiB F32 weight stage and retains source/oracle/workspace buffers. Its
explicit arena is 2,211,073,156 bytes, excluding opaque MPS plan internals.
An offline compact artifact, exact-size resident allocation and a packed
consumer remain separate integration work.

Eleven individual GPU controls pass: seeded +1/NaN comparator failures,
threshold equality, strict raw-cost ties, zero sigma, alignment padding,
stale valid bytes after a failed pack, insufficient capacity, and malformed
precision/offset/position/order. Four host argument guards reject. Failed
packing invalidates decoding even if a previous valid blob remains resident.
Every native numerical arm uses one terminal wait, zero inter-operator waits,
zero activation readbacks and zero tracked encode-time MetalBuffer allocations.

#### Paired cost of the staged consumer

Four selected policy-0.50 arms have four pack-only samples and four matched
decode/projection versus BF16-expand/projection pairs each, after individual
warmups. Both pair orders occur (ABBA then BAAB); validation is outside each
timed command. Reported segment medians include the existing terminal wait;
packing is separate from decode/projection. These are small, single-process
paired observations, not whole-block or end-to-end model timings.

| Packing / corrections | Pack wall ms | Decode + projection wall ms | Paired BF16 wall ms | Median paired slowdown |
| --- | ---: | ---: | ---: | ---: |
| 256 / BF16 | 9.875 | 63.840 | 58.114 | 9.017% |
| 512 / BF16 | 7.932 | 61.910 | 56.850 | 7.806% |
| 1024 / BF16 | 7.219 | 59.904 | 58.627 | 2.081% |
| 512 / F16 delta | 8.308 | 61.255 | 57.836 | 5.474% |

All sixteen wall-time pairs favor BF16. Two 1024-value GPU-elapsed pairs are
slightly faster for the candidate, although its GPU median is slower; the
smallest paired effect is noisy. Cross-arm ordering is fixed, so comparing these
arm medians does not establish a tile-size speedup. Smaller packed-source
reads alone do not remove the shared F32 stage write/read, output traffic or
the 317 billion MACs at 4480 rows. At the 1024/0.50 point, the source payload
shrinks from 141.56 MB BF16 to 39.99 MB, while both paths still write 283.12 MB
of F32 weights and the consumer reads that stage. This motivates tiled
decode-and-multiply without the global intermediate, not a bandwidth-speedup
claim for the current path.

Correlated Mini-Quad review is ROBUST for bounded layout/selection/parity and
the reported paired observations; model quality, deployed memory reduction
and a general speedup remain unestablished. Defaults/loaders remain unchanged.
No full generation ran. Guarded native sampled peak is 2.095 GiB, minimum
cache-aware headroom 31.422 GiB, pressure-free minimum 76%, swap growth zero,
and no owned processes remain. All build/oracle/native commands exit zero,
with unchanged prelaunch-pinned inputs and authenticated completion receipts.

External evidence directory:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-positional-20261022/`.
`summary-v3.json` SHA256
`481f555253e5341f54cc4e10c105f7ea8ddb10b1ba6ae80d904a506b34eca896`
indexes `build-v2`, `oracle-v1`, `native-v2`, immutable `fixture-v1`, per-control
results and all sixty timing segments. Earlier native-v1 results are exploratory
after the probe's source comment/default-fixture-path refresh; v2 binds the
final source bytes. The fixture manifest SHA256 is
`d77557f7d01798f91bbc3ad765f0147e33c0702e4cdb735f849b9641a0aaee3f`.

CPU-only source/receipt/guard/result refresh (choose a fresh output filename):

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-positional-20261022/verify.py \
  --build v2 --oracle v1 --native v2 --output summary-refresh-UNIQUE.json
```

The indexed release-build and native commands are in the summary and their
intent receipts. Native invocation uses `--fixture=.../fixture-4480-v3`,
`--positional-fixture=.../fixture-v1` and `--rounds=4`, under
`wan_memory_guard.py --memory-accounting=file-cache
--max-footprint-mib=4096 --max-swap-growth-mib=64 --timeout-sec=600`.
The guard's fixed reserve is 15%; the explicit budget is a sampled stop limit,
not a memory reservation.
Refresh on relevant source, inputs, layout or backend changes; rerun matched
timings on host/runtime/device changes. The return frontier is reduced-step
short-video testing only after packed-model admission and trajectory checks.

### Quantized arithmetic falsifier contract (2026-10-22)

Active bounded hypothesis: eliminate the full decoded-weight intermediate,
then test whether factoring the canonical Gaussian mean/scale avoids useful
per-weight work without failing the existing F32 gate. For each normalization
group, real arithmetic gives
`sum(x*w) = mean*sum(x) + sigma*sum(x*L[code]) + sum(x*correction)`.
The decoder instead rounds `mean + sigma*L[code]` per weight, so the real-number
identity is not an F32 equivalence certificate. BF16 absolute replacements use
the actual rounded base in any delta; F16-delta mode retains its separately
rounded reconstruction. Raw BF16 records retain a dense fallback.

Before a tiled production consumer, add an opt-in native diagnostic over the
four distinct seeded input rows, all 13824 outputs and all 5120 channels.
Use the immutable positional fixture's 1024-value records at policies .20/.50,
both correction modes, including the mixed p4/p5 and p4-only arms. Compare:

1. The oracle's decoded F32 weights with a native F32 dot reduction control.
2. Fused per-weight reconstruction and dot with the same native reduction.
3. Code-domain mean/scale factoring with the actual positional corrections.

An optional fourth diagnostic computes the 16/32-entry affine reconstruction
table once per normalization group, in the decoder's exact F32 multiply-then-add
order, then gathers by code before the same dot reduction. This removes
repeated affine reconstruction without requiring a real-number reassociation.
The factored route, in contrast, also omits the dense per-weight F32 rounding
residual unless it is explicitly represented; sparse escape deltas do not
make that route bit-exact.

Every output is compared on GPU against its independent same-format MPS
oracle at finite `2e-5 + 2e-5*abs(reference)`; compare candidate/control directly
to localize a reduction failure. A failing experimental arm is reported as a
refuted route, never admitted by widening tolerance. Positive +1 and NaN
comparator controls must reject. One terminal completion per diagnostic arm,
zero inter-operator waits/activation readbacks/tracked encode allocations;
preallocate before encode. Uploads and terminal scalar metrics are outside
the native chain. Keep the sampled 15% reserve, 64 MiB swap cap, 4 GiB process
budget and a bounded deadline. No full model or generation.

This CAUTION change is isolated to the probe and removable without changing
production kernels/loaders/defaults. DoD is a guarded release build and a
source/fixture-bound native report that discriminates the three paths,
including any rejected path. Strongest failure: numerical reassociation or
incorrect sparse-replacement semantics masquerading as a codec improvement.
Four-row diagnostic timing cannot establish 4480-row GEMM performance; a
matched tiled consumer comparison remains necessary for any speed claim.

#### Exact codebook arithmetic and candidate limits

Let `fl` denote F32 rounding and let the canonical decoder produce
`D_g[c] = fl(mean_g + fl(sigma_g * L[c]))` for each 256-weight normalization
group. A local table of the 16/32 possible `D_g` values costs 64/128 bytes.
Gathering from that table preserves the base weight bits when the decoder's
explicit multiply-then-add order and disabled contraction are retained.
Apply absolute BF16 replacements or separately rounded F16 deltas afterward,
then use the same multiply/reduction schedule as the control. This is still
logical dequantization, but it can occur inside a consumer without a global
283,115,520-byte F32 weight stage. Building the table amortizes 256 affine
reconstructions into 16/32 entries per group; it does not reduce the GEMM's
multiply/accumulate count or establish a throughput improvement.

Factoring `mean*sum(x) + sigma*sum(x*L)` instead changes arithmetic. The exact
real-valued difference from the current decoded base is the dense per-code
residual `R_g[c] = D_g[c] - (mean_g + sigma_g*L[c])`. Sparse outlier sidecars
do not represent that dense residual. The first stored group already has
nonzero residuals at every p4/p5 level (16/32), with maximum absolute values
`9.090745933804101e-10` / `6.272422581332648e-10`. These tiny weight errors
and the changed dot reduction still require the unchanged projection gate.
The real-arithmetic mean component has rank at most 20 for K=5120, but that
does not make the nonlinear coded component low rank.

A genuinely code-domain real dot can accumulate weighted code buckets
`H[c] = sum(x_i where code_i==c)`, then compute `sum(D[c]*H[c])` and apply
corrections. That avoids reconstructing each weight, but changes F32 sum order
and adds bucket clearing, scatter/reduction and possible contention. No Metal
histogram implementation or performance result is admitted here.

Exact rational analysis of the pinned finite F32 Gaussian LUTs confirms global
complement antisymmetry. Their Walsh bases have exactly 8/16 nonzero terms,
all odd-cardinality masks; no additional terms vanish. A direct signed-feature
expansion therefore needs 2048/4096 signed terms per 256-weight group, versus
256 direct LUT-weighted terms, before parity-feature generation. It is not the
ordinary 4/5 integer-bitplane dot. This count is a reason to prioritize the
local rounded codebook, not a proof that every bitwise algorithm is slower.

Primary references:
[FLUTE](https://arxiv.org/abs/2407.10960) fuses GPU lookup-table dequantization
with matrix multiplication and reorders weights for hardware fragments;
[T-MAC](https://arxiv.org/abs/2407.00088) replaces low-bit CPU dot products with
activation-table lookups over bitplanes. Their device, batch, reduction and
precision choices do not establish Metal/Wan speed or our unchanged parity.
In particular, do not copy lower-precision tables/partial accumulations or
discard Gaussian basis terms under this exact-path contract.

Tiny-LUT CPU evidence is `lut_math.py` / `lut-math-v1.json` in
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-dot-20261022/`.
Report SHA256:
`b4ce9160e51e43726c18ee82a829c15191faecb9807a4e7d031e6c92f53507fa`.
It uses exact rational coefficients, checks every reconstructed LUT entry,
binds LUT/packed inputs to the source manifest and performs no GPU work.

#### Native arithmetic falsifier result

`examples/wan_qbit_dot_probe.cr` executes the four distinct seeded upstream
input rows at the actual 5120 -> 13824 projection dimensions. It consumes the
immutable positional records at tile1024, thresholds .20/.50, both correction
modes; .20 includes mixed p4/p5 records and .50 is p4-only. No PyTorch executes
in the native path. The reference is the saved same-format MPS projection of
the identical weights, inputs and bias, not the original BF16 projection.

Each arm checks 55,296 output values per method, totaling 221,184 per method
over four arms. All four methods have zero nonfinite outputs and zero
violations of finite `2e-5 + 2e-5*abs(reference)` against the MPS oracle:

| Threshold / correction mode | Native dense vs MPS violations | Fused decode vs dense bit differences | Local codebook vs fused bit differences | Factored vs dense bit differences / tolerance violations |
| --- | ---: | ---: | ---: | ---: |
| .20 / BF16 replacement | 0 | 0 | 0 | 43,367 / 0 |
| .20 / F16 delta | 0 | 0 | 0 | 43,363 / 0 |
| .50 / BF16 replacement | 0 | 0 | 0 | 43,311 / 0 |
| .50 / F16 delta | 0 | 0 | 0 | 43,286 / 0 |

The native reducer differs in low bits from MPS even for the dense control;
it remains within the unchanged gate. The fused reconstruction and rounded
local codebook are bit-exact against their same-reduction controls. Factoring
changes output bits but is not refuted by this bounded fixture; this is not an
identity over F32 or evidence about unseen denoising activations. Raw BF16
record handling exists but is not exercised by these selected fixture arms.

+1/NaN positive comparator controls reject in every arm. All explicit buffers
are allocated before encoding (322.586 MiB, including the diagnostic dense
control); encode-time tracked MetalBuffer allocations, device activation
readbacks and inter-operator host waits are zero. Each arm encodes all four
paths and GPU comparisons before one terminal completion. Source-level checks
and source-bound receipts support these scoped statements; the buffer counter
does not instrument opaque runtime allocations.

Definitive build-v2/native-v2 and CPU verification exit zero with unchanged
prelaunch-pinned files (233 build, 325 native). Native-v1 stopped before GPU
execution on an incorrect nesting of `geometry.projection_rows_saved`; the
fixed source was rebuilt and rerun as v2. Native sampled peak is 399.423 MiB,
minimum cache-aware headroom 32.152 GiB, minimum pressure-free 78%, swap growth
zero, and no owned processes remain. The guard retains its 15% reserve,
4 GiB process-tree budget, 64 MiB swap-growth limit and 300-second deadline.

External evidence:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-dot-20261022/`.
`summary-v1.json` SHA256
`639e02d9b6923cd46bfdbda358341ba988837d840590432ddd1fcab09a93807d`
binds the receipts, guard samples, all gates and exact-LUT math report. Native
binary SHA256 is
`6bceb973d24a8d8559a0d65447d129a323feb6cd8fcb2417ba2b06072c5a8e62`.
Probe source SHA256 is
`89bffc1575564dfc2e8de7c82659995c940835ceb457e8b2cb25f95965026b73`.

CPU-only source/receipt/result refresh (fresh output name):

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-dot-20261022/verify.py \
  --build v2 --native v2 --output summary-refresh-UNIQUE.json
```

The summary exposes rejected gates as data; probe exit zero alone does not
admit an experimental path. For this run all sixteen MPS gates are true.
Correlated Mini-Quad verdict is ROBUST for these bounded projection gates and
same-reduction output equivalence; elementwise full-weight equivalence, unseen
activations, raw fallback coverage and throughput are not established by this
projection-output test.
Refresh after relevant source/fixture/decoder/backend changes. No timing,
4480-row GEMM, full block, resident-model savings or video quality was measured.
Next discriminator: a tiled fused consumer using rounded local codebooks,
reusing decoded fragments across input rows, measured against staged MPS at
4480 rows with the same numerical gate and a guarded process budget. Preserve
the exact control when exploring the factored variant.

### Local-codebook tiled GEMM contract (2026-10-22)

CAUTION, opt-in only: integrate the rounded per-normalization-group codebook
with F32 simdgroup matrix multiplication. The existing decoder-to-MPS path
remains the control and all production defaults stay unchanged. Rollback is
removal of the new isolated kernel/wrapper/probe on this worktree.

The first falsifier is the full 5120-channel reduction at four distinct seeded
upstream rows and all 13824 output channels, across the four tile1024 arms
(.20/.50, BF16 replacement/F16 delta). The kernel must preserve canonical
reconstruction, ordered positional corrections and raw BF16 records, using
F32 operands, accumulators and output. No F16 operand staging or tolerance
change is admitted. A same-geometry dense-F32 tiled control separates decoder
effects from matrix-reduction effects. Synthetic controls exercise raw records,
correction semantics, row tails, malformed records and a comparator +1/NaN.
The consumer rejects a malformed record by poisoning its entire 8-channel
output tile and setting status[7]; the synthetic check expects 152 nonfinite
outputs for 19 rows when the first record is corrupted, with the ninth channel
remaining finite. This is intentionally a tile-wide failure boundary.

Only after numerical qualification, compare the complete fused projection
against positional decode -> F32 MPS GEMM -> bias at 4480 rows in one process,
including both warmup orders and ABBA/BAAB measurements. Each timed invocation
owns one command-buffer boundary and terminal completion; comparator kernels,
uploads, pipeline preparation and fixture validation stay outside timing.
Report wall, GPU and encode time and every trial, not just a best sample.
The full-shape inputs repeat four seeded rows and do not establish video
trajectory quality or representative activation locality.

Implementation plan and DoD:

1. `src/ml/video/qbit_codebook_linear.cr` and
   `src/ml/video/kernels/qbit_codebook_linear.metal`: caller-owned fixed buffers,
   local rounded codebooks and shared decoded fragments across input rows;
   bounded row ranges preserve the K traversal. No new waits, activation
   readbacks or tracked MetalBuffer allocations while encoding.
2. `examples/wan_qbit_codebook_gemm_probe.cr`: pinned source/positional manifests,
   GPU reconstruction/projection comparisons and positive rejection controls,
   then guarded, counterbalanced full-projection timings. The strongest risk
   is a changed reduction passing a weak local control but missing upstream;
   finite per-element `2e-5 + 2e-5*abs(reference)` remains mandatory.
3. A source-bound guarded release build and parity/timing reports must cover
   the new source files and preserve the existing cache-aware 15% reserve,
   pressure-free floor 15%, 4 GiB process-tree budget and 64 MiB swap-growth
   stop. Scratch, build cache and evidence remain under `/opt/mldata_ssd`.

No model loading, full block or generation is part of this contract. Logical
packed bytes, retained diagnostic capacity and measured process memory must
be reported separately. A numerical rejection or slower complete consumer is
a measured result; it does not justify precision relaxation or default enablement.

### Local-codebook tiled GEMM result (2026-10-22)

Implementation: `QBitCodebookLinear` computes 16 rows by 8 output channels
with two F32 simdgroup accumulators, ascending K8 fragments and one rounded
codebook per canonical 256-value normalization group. The decoded 8x8 B
fragment is reused across both row tiles. BF16 replacements, separately
rounded F16 delta additions and raw BF16 records preserve their decoder
semantics. The caller owns buffers, status and terminal completion. The
experimental module is not wired into production defaults.

Guarded release build-v2, parity-v2 and timing-v2 complete with unchanged
source, binary and fixture pins. At four distinct seeded upstream input rows,
all four selected tile1024 policies (.20/.50, correction modes 0/1) have zero
bit differences against their saved same-format upstream MPS output oracle.
All 70,778,880 reconstructed weights per arm match the independent decoder
oracle exactly. Fused and same-geometry dense GEMM also match exactly. Current
four-row MPS launches have shape-dependent rounding differences against the
saved 4480-row oracle, but pass the unchanged finite
`2e-5 + 2e-5*abs(reference)` gate (maximum ratio 0.303).

The selected .50/mode0 arm additionally qualifies all 61,931,520 outputs at
4480 rows, with zero bit differences for fused, dense and staged MPS paths
against the saved oracle. These inputs repeat four seeded rows 1120 times;
the weights and projection geometry are real Wan, but the activations are not
captured from a denoising trajectory. Synthetic 19x256 -> 9 controls cover
p4/p5/raw records, both correction modes, positions 0/127/255, row/output tails
and tile-wide malformed-record rejection. Comparator +1/NaN controls reject.

Full consumer timings include status clear, all dispatches, bias, command and
encoder creation, and one terminal completion. Preparation, uploads,
comparators and correctness checks are outside timing. Fused and dense paths
use 70 row-range dispatches on one command buffer; the MPS control decodes the
whole weight matrix then projects and adds bias on that same boundary.
Both warmup orders precede the following ABBA/BAAB measurements (A=fused,
B=decode/MPS). Times are milliseconds; all samples are retained:

| Position | Pair | Order | Route | GPU ms | Wall ms | Encode ms |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 0 | 0 | ABBA | fused | 3518.005667 | 3518.285041 | 0.068041 |
| 1 | 0 | ABBA | decode/MPS | 61.078750 | 66.607000 | 0.064667 |
| 2 | 1 | ABBA | decode/MPS | 59.025375 | 59.349666 | 0.084625 |
| 3 | 1 | ABBA | fused | 3520.326626 | 3520.633542 | 0.082292 |
| 4 | 2 | BAAB | decode/MPS | 59.375624 | 64.462250 | 0.076583 |
| 5 | 2 | BAAB | fused | 3532.482209 | 3532.782667 | 0.069792 |
| 6 | 3 | BAAB | fused | 3517.693958 | 3518.199458 | 0.200416 |
| 7 | 3 | BAAB | decode/MPS | 61.261416 | 67.584708 | 0.061916 |

Median GPU times are 3.519166 s fused and 0.060227 s staged (58.43x ratio).
Individual paired GPU slowdowns range from 57.42x to 59.64x. Median wall times
are 3.519459 s and 0.065535 s. One same-geometry dense diagnostic takes
0.905182 s GPU time; it is not a repeated dense-versus-MPS benchmark. These
results reject a speedup for this implementation rather than quantized GEMM
in general. CPU encode times are tens to hundreds of microseconds and do not
explain the seconds of GPU time.

Source inspection identifies costly hypotheses, not measured cycle attribution:
483,840 threadgroups each traverse 640 K8 fragments; packed decoding and record
validation repeat for each 16-row tile (280 row tiles). Quantized fragments
perform 2612 threadgroup barriers versus 1281 for the dense control, and sparse
corrections use a binary search for each reconstructed value. The poor dense
diagnostic motivates a paired dense-tiled versus direct-MPS comparison on the
same predecoded weights before changing the custom schedule. Removing only
decoder instructions has not been shown sufficient. If that split confirms
the baseline gap, test larger row/output tiles at unchanged K reduction and
parity, then compare on-chip codebook/correction staging with the dense control.
Do not attribute
the gap to one instruction or promise bandwidth savings without that test.

All valid status words remain zero. Native encoding reports zero tracked
MetalBuffer allocations, zero intermediate host waits and zero device
activation readbacks; small metrics/status are read only after the caller's
terminal completion. The timed arena explicitly retains 1,207,689,300 bytes,
including a 270 MiB F32 stage and a 270 MiB oracle matrix for controls.
Logical blob-plus-offset storage is 39,991,964 bytes (4.520214 bpw); it is not
measured deployed resident storage and excludes global container metadata.
The original-weight projection error at this policy remains about 6.95%
relative L2 on the existing fixture. Same-format parity does not qualify that
lossy path against original BF16 or establish video quality.

The sampled timing guard peaks at 1.248 GiB for the owned process group,
minimum cache-aware headroom is 28.605 GiB and pressure-free reading 77%.
Swap growth is zero and no owned process remains. The 4 GiB process budget,
15% reserve/floor, 64 MiB swap-growth stop and 900 s deadline were retained;
sampled telemetry cannot exclude between-sample spikes. No full model, block
or generation was launched.

Evidence is external under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-codebook-gemm-20261022/`:

- `summary-v2.json`, SHA256
  `7fc5055c4746d5372731182bdb65a99a5695641855111ca46dd63c0eb7d46ff7`;
- release binary SHA256
  `b5ee8c0eb3274e1368a9c432664e7df21967a19978ecdeb0e0329ff204fe0065`;
- kernel SHA256
  `84f17155b062e87da940ef55a3b7c2bb9611688ef254f4d780ec205abe237cbf`;
- native probe SHA256
  `5fc3ac5e3ed2680f9b0e14e64b86b8744184f85f91f6285fac32a07b5df4d31b`.

The receipts pin all four fixture arms, transitive sources, link objects,
compiler and binary before and after each guarded run. A CPU-only refresh
rechecks actual receipt hashes, current pinned bytes, numerical records,
sample ordering, status words and guard telemetry:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -B \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-codebook-gemm-20261022/verify.py \
  --build v2 --parity v2 --timing v2 --output summary-refresh-UNIQUE.json
```

This command belongs to the historical probe hash above. The later opt-in
`dense-timing` addition changes that pinned probe, so the old live-source
refresh now rejects it as stale. Preserve its receipts as historical evidence;
use the final dense-control verifier for the current probe bytes.

Correlated Mini-Quad verdict is ROBUST for same-format parity and this measured
complete-path regression, VULNERABLE for other timing arms, real activations,
cycle attribution, resident-model admission and video quality. Refresh after
relevant source, layout, LUT, weight, input, shape or backend changes.

### Quantized GEMM introspection and compute-isolation contract (2026-10-22)

This audit revisits the execution boundary before another kernel rewrite. The
three reviewers inspected separate geometry/reuse, representation/exactness,
and benchmark/model-integration surfaces. Their reviews share a model lineage;
source inspection, arithmetic and executed checks carry the evidence, not
agreement. Mini-Quad remains required at each material transition: make a
prediction, challenge it, run the smallest discriminator, then attack its scope.

**The capacity problem does not require a fused GEMM first.**
[`WanS2VMPSLinearStage`](../src/ml/video/metal_linear_mps.cr) already shares one
F32 expansion stage across 15 fixed block/audio plans. Its largest matrix at
the current dimensions occupies 283,115,520 bytes (270 MiB), independent of
the number of layers. Packed weights could replace the resident BF16 weight
pool while each projection decodes into this stage and uses MPS. This remains
a proposed model integration: `WanT5.mps_projection` currently always expands
a separate F32/BF16 source and rejects source/stage aliasing. It needs a
validated preexpanded entry point, packed loading and per-binding identity;
the current helper is not already a packed-model mode. Preserve serial stage
ownership, ordered GPU decode/use, failure propagation and completion lifetime.
Count all packed bindings, metadata, protected tensors, scratch, T5 and opaque
MPS allocations before model admission. A 270 MiB stage is not a reason to
retain the entire original BF16 pool as well, once the lossy packed model has
its own quality admission. Keep the original-weight exact control as a
separate run rather than silently treating same-format parity as that gate.

**The small fused tile repeats work that a large GEMM should amortize.**
At M=4480, N=13824, K=5120, the 16x8 output mapping launches
`(4480/16)*(13824/8) = 483840` threadgroups. Each traverses 640 K8 fragments,
with two F32 matrix-MACs per fragment. It reconstructs
`280*70778880 = 19818086400` B values and logically requests 158.54 GB of
F32 activation loads across output tiles. These are source-level operations,
not DRAM measurements. Cache/compiler effects and the four repeated input
rows matter. Current record validation, code extraction and sparse correction
lookup repeat per row tile. The measured .50/mode0 arm has only p4 records,
459,430 corrections among 70,778,880 weights (0.65% density). It invokes a
correction search per decoded coefficient; empty lists require no binary-search
loop comparisons. Rounded codebook expansion is amortized per canonical
256-value group within each threadgroup, then repeated across the 280 row
tiles; it is not a fresh multiply/add per weight. Counts do not establish
which instructions dominate GPU time.

The projection performs 634,178,764,800 counted FLOPs. The idealized F32
operand/output footprint is 622,592,000 bytes, giving about 1018.61 FLOPs/byte
if each operand were transferred once. This is a reuse opportunity, not
achieved arithmetic intensity or proof of a compute bottleneck. The measured
complete-path effective rates are about 0.18 TFLOP/s fused and 10.53 TFLOP/s
decode/MPS; the one-shot dense diagnostic is about 0.70 TFLOP/s. A cache or
barrier hypothesis still needs a discriminating measurement.

**Archive layout and execution layout need not be identical.**
The current positional packer copies canonical p4/p5 bitplanes unchanged.
A GPU execution layout that is lossless relative to the selected QBit decoder
outputs could block outputs in K-major order, reduce bit extraction and make
eight-output fragments contiguous. Account for conversion,
offsets, corrections, raw records, and any simultaneous archive copy. Larger
row/output tiles and K32 staging can preserve ascending F32 K8 accumulations;
they are candidates, not established speedups. Existing larger GGUF kernels
provide scheduling examples but use half operands/rounding, so do not satisfy
this numerical contract unchanged. A temporary shared correction index is a
latency/occupancy trade; a persistent dense map also consumes model memory.
Their merits cannot be decided from correction density alone.

[`FLUTE`](https://arxiv.org/html/2407.10960v2) offers relevant mechanisms:
offline weight reordering and vectorized/duplicated lookup tables. Its cited
2-4x kernel gains target batches below 32, not M=4480; its FP16 global partial
reduction is not admitted by our F32 contract. Borrow layout ideas and measure
them on Metal rather than importing CUDA performance claims. Mean/scale
factoring and direct bitplane arithmetic still have the previously recorded
nonlinear-LUT and F32 reassociation falsifiers.

**Validation can be amortized only with enforceable weight ownership.**
A raw mutable `MetalBuffer` plus a status word does not certify future uses.
A prepared handle would need to bind blob, offsets, LUTs, geometry, layout and
generation, prevent mutation/alias reuse through completion, and order
validation/use on the same command chain. Consumers must retain fail-closed
certificate/status checks. Without that boundary, keep extent/header/padding,
finite-value and sorted-unique-correction validation. This is a possible
future design, not a license to remove the current checks.

**Kernel parity, weight selection and video quality have different probes.**
For weight error E and activation matrix X, linear output error is `X*E^T`
and its squared Frobenius norm is `tr(E*(X^T*X)*E^T)`. Repeating four rows
1120 times leaves X's rank at most four; it multiplies the same calibration
Gram matrix rather than adding activation coverage. The current ~6.95%
relative L2 error versus original BF16 therefore cannot qualify a denoising
trajectory. Real early/middle/late-step and CFG-branch activations should guide
precision/corrections after memory admission; keep strict same-format
implementation parity separate. The earlier Q4_K result had lower projection
error at a smaller size than QBit p5 on these inputs, so compare it at matched
complete storage/runtime budgets rather than privileging the more elaborate
codec. No quality ranking beyond these inputs follows.

The return frontier remains 8/12/16 versus 40 diffusion steps after safe model
admission. Step reduction changes the algorithm and does not reduce resident
weight memory. Conversely, weight compression alone does not remove long-key
self-attention or denoising calls. N=36608 whole-block measurements and this
M=4480 single-projection experiment must not be combined into a video ETA.

**Next discriminator, before changing either compute kernel:** run the new opt-in
`dense-timing` mode in the existing probe. Decode the selected arm once outside
timing; compare dense F32 and direct MPS with identical resident F32 weights,
input, bias, status overhead and command boundaries. Warm both orders and use
four ABBA/BAAB pairs. Qualify full outputs against the pinned upstream oracle
before and after timing, including +1/NaN comparator controls. Each timed arm
retains one terminal completion, zero inter-operator waits/readbacks and zero
tracked encode-buffer allocations. Use a 4 GiB owned-group limit, 15%
cache-aware reserve/floor, 64 MiB swap-growth stop and 300 s runtime deadline;
scratch/build/evidence stay on the external SSD. No full model or generation.
Rollback is omission of the new opt-in mode; production defaults are untouched.
DoD is a source/binary/fixture-bound independent CPU verifier accepting all
numerical, order, timing and guard records. Reject missing records and finite
or NaN corruptions. If dense remains much slower, retain MPS as the practical
consumer and investigate custom F32 scheduling separately; if competitive,
isolate packed decode cost. Neither outcome alone authorizes a model-quality
or whole-video acceleration claim.

### Predecoded dense versus direct MPS: paired result (2026-10-22)

The guarded release build-v2 and dense-timing-v2 pass the contract above.
Both routes consume the same retained F32 stage, F32 input and bias. Decode,
full-output comparisons and MPS-plan creation occur outside the timed arms.
Direct MPS borrows the live Metal command buffer/encoder; the bridge does not
submit or wait internally. Each route retains its existing one terminal wait.
The dense route keeps the current 16x8 output tile, K8 accumulation and 70
dispatch segments; this experiment changes neither compute kernel.

| Median, four measured samples per route | Dense F32 | Direct MPS |
| --- | ---: | ---: |
| GPU interval | 905.788 ms | 57.411 ms |
| CPU encoding | 0.043 ms | 0.065 ms |
| Wall interval including terminal completion | 906.168 ms | 58.560 ms |
| Counted projection FLOPs / GPU interval | 0.700 TFLOP/s | 11.046 TFLOP/s |

The ratio of GPU medians is 15.7774x. All four adjacent counterbalanced pairs
favor MPS: dense/MPS ratios 15.4588, 15.6339, 16.2140 and 15.9186.
ABBA and BAAB strata give ratios of medians 15.5461x and 16.0663x;
the direction survives both orders, without establishing host-independent
latency. Eight warmup invocations precede these eight measured invocations.
No profiler is active. Encoding is tens of microseconds here and does not
explain the roughly 848 ms GPU-interval gap. The GPU interval covers the whole
command; it does not attribute cycles to individual instructions.

Before timing, all 70,778,880 decoded weights and all 61,931,520 outputs per
route match their saved same-format upstream oracles bit-for-bit. After
timing, dense/upstream, MPS/upstream and dense/MPS comparisons again have zero
bit differences, nonfinite values or tolerance violations. The unchanged
finite `2e-5 + 2e-5*abs(reference)` gate and GPU +1/NaN comparator controls
pass. Synthetic raw/p4/p5, both correction modes, tails and malformed-record
controls also pass. Tracked encode-time Metal-buffer allocations,
inter-operator host waits and activation readbacks are zero. Status/metrics
are read only after terminal completion, outside the measured encode path.

The runtime guard records 28 samples, a 1.274 GiB process-group peak,
19.977 GiB minimum cache-aware headroom, minimum pressure-free reading 76%,
zero swap growth and zero owned processes remaining. The 4 GiB group cap,
15% reserve/floor, 64 MiB swap-growth stop and 300 s deadline are retained;
build uses a 240 s deadline. Sampled guards cannot exclude between-sample
spikes. All scratch, binaries and receipts are external. No full model, block,
Studio or generation was launched.

Evidence is external under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-dense-control-20261022/`:

- `summary-v2.json`, SHA256
  `992ac74212e16c8da5f26043a0aad096e4f865264b3540d7fceb370092f11735`;
- binary SHA256
  `ea77ef8ab576fb1865d4fb9ab5086ddd91d74e785c85cf1f48886ff046a4fdff`;
- native probe SHA256
  `ef0ecc9d561db918453811853df89b3c10322e0f41e565f2a49031839bff3508`;
- independent CPU verifier SHA256
  `8677d35cc3bc5b9b5e369a62db3225041ed931831da9af34fe81b43e73fd5d10`.

V1 numerical records passed, but its receipt verifier rejected a fixture file
shared by several declared arms. The verifier also needed to recognize the
external-SSD symlink for the executable path. Preserve the rejected V1
receipt/verifier; V2 pins the corrected verifier before a fresh build/run.
Only declared shared roles and the canonical executable alias are admitted;
other command arguments, binary digests and fixture claims remain exact.
Ten CPU negative controls reject nonfinite metrics, over-counts, missing
records, failed parity, absent +1/NaN detection, altered arguments/digests and
unclaimed shared roles. This repairs provenance checks without weakening
numerical or resource gates. CPU-only refresh (choose a new output basename):

```sh
TMPDIR=/opt/mldata_ssd/cogni-ml/video-scene-20261019/build/wan-qbit-dense-control-20261022 \
  /opt/homebrew/opt/python@3.11/bin/python3.11 -B \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-qbit-dense-control-20261022/verify.py \
  --version v2 --output summary-refresh-UNIQUE.json
```

Correlated Mini-Quad verdict: ROBUST for the scoped paired observation and
same-format numerical parity. The tested dense schedule is slow even after
decode is removed; it needs better reuse/scheduling before decoder-only work
can make this route competitive. This does not assign cycles to barriers,
loads or codebooks, nor prove a lower bound for a redesigned fused layout.
The strongest falsifier is a provenance-clean matched rerun that materially
collapses this gap. Four repeated seeded rows remain a throughput fixture,
not trajectory-quality coverage.

Next bounded move: introduce and test a validated preexpanded MPS entry point
on one packed projection, including decode, serial stage reuse, binding
identity and failure propagation. Keep the existing helper/control unchanged.
Only after that passes should a separate memory plan count complete packed
storage, protected tensors, T5, shared scratch and opaque MPS resources for
model admission. Custom lossless execution repacking/larger F32 tiles remain
separate candidates. Reduced-step 8/12/16 versus 40 video is still the return
frontier after memory and trajectory-quality admission. Refresh numerical
evidence after source, layout, LUT, weights, inputs, geometry or backend
changes; refresh paired timing on the current host. No model-quality,
deployed-memory or whole-video speed claim follows from this projection.
