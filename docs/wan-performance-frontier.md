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
the supplied command in a new session. Admission requires `memory_pressure -Q`
at least 15% and free-plus-speculative physical bytes at least 15% of RAM plus
the requested footprint budget. During execution it samples the process group's
summed `ri_phys_footprint`, physical headroom and system swap growth. Missing
telemetry stops the owned group; a monotonic timeout and wrapper SIGINT/SIGTERM
also clean it up. The unreaped session leader pins the group's identity until
cleanup. This covers the group, not descendants that deliberately escape with
`setsid` or `setpgid`, and cannot prevent a between-sample allocation spike.
Timeout initiates termination, rather than bounding wrapper return: an
uninterruptible kernel/driver task may delay reaping even after SIGKILL.

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
Free-plus-speculative pages are a conservative reserve, not a measurement of
all reclaimable RAM. A rejected launch does not establish OOM or a Wan leak;
inactive, file-backed and compressed pages are separate accounting categories.

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
   key/value tile within an eight-query SIMD group. Sharing one tile across
   two or four such groups could reduce repeated loads. K=32, D=128 and two
   F32 tiles already need 32 KiB of threadgroup storage; double buffering needs
   64 KiB. Check device limits and occupancy before implementation. Preserve
   each row's ascending key reduction, then check bitwise parity to the old
   native kernel, strict full-shape upstream parity and the real full block.
   The existing native tiled reduction is not automatically a qualified
   replacement for the MPSGraph long-key path.
3. **FFN bias/GELU epilogue.** A post-MPS bias-plus-GELU kernel could remove one
   full FFN-intermediate read/write, about 3.77 GiB of logical traffic per block,
   and one dispatch, while retaining the MPS dot-product reduction. GELU already
   aliases the FFN-input backing, so this is not an additional buffer saving.
   Preserve the rounded F32 bias result and the current GELU operation order;
   coefficient/FMA changes still require bitwise intermediate and block checks.
   Traces must continue to expose the pre-GELU value when requested.
4. **BF16/F32 GEMM and paired projections.** Q/K/V are currently separate
   linear calls. Packing projections or using native mixed-storage GEMM may
   improve input reuse and avoid expansion, but it changes matrix shape or the
   reduction tree. BF16-to-F32 conversion is exact; a new accumulation tree is
   not necessarily bitwise equal. Qualify each shape before replacing MPS.

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
