# Joint INT4 range and exception fitting for a Wan projection

Status: verified bounded experiment (2026-10-22 local, 2026-10-23 UTC evidence).
Joint range/exception fitting improves this affine representation at unchanged
storage cost, but Q4_K retains lower observed distortion. No speed winner or
model-default promotion is admitted. This opt-in experiment changes neither
model defaults nor checkpoint files.

## Question and controls

Can choosing affine INT4 ranges together with sparse BF16 replacements reduce
projection distortion at the complete Q4_K storage price? The previous
[min/max experiment](wan-affine-budget-benchmark.md) repairs residuals only
after fixing the base range. An extreme endpoint can be represented exactly
while consuming most of the range and leaving inliers with little resolution;
largest-residual selection alone does not repair that cause.

The pinned real tensor is `blocks.0.ffn.0.weight`, N=13,824 and K=5,120.
There are 70,778,880 source BF16 values and 276,480 affine groups. M=4 is the
diagnostic shape. M=4480 repeats the same four seeded source rows 1,120 times
for a real projection size; these are not captured video activations.

Every package contains a 39,813,120-byte payload plus a common 16-byte
shape/format descriptor. Scale/offset parameters, directories, patch positions
and BF16 values, alignment and padding all count. Shared F32 bias is 55,296
bytes separately. No sidecar is hidden from this budget. All compressed routes
still use a reusable 283,115,520-byte F32 stage before the shared MPS plan.

For each sparse affine package, the complete payload is:

| Component | Bytes |
| --- | ---: |
| 276,480 groups: 128 code bytes + F32 scale + F32 offset | 37,601,280 |
| 69,120 record counts, u32 | 276,480 |
| 69,121 record offsets, u32 | 276,484 |
| 414,719 patches: u16 local position + original BF16 value | 1,658,876 |
| Total payload | 39,813,120 |

Patch positions are sorted within each 1024-value record. These are absolute
BF16 replacements, not F16 residuals. The fit metadata is reusable setup
scratch, not a persisted sidecar: 48 bytes/group, or 13,271,040 bytes. The
complete packing workspace is 355,010,000 bytes; it is released before timing.
The probe's explicit buffer sum during packing is 2,344,659,944 bytes at
M=4480, including the ten-arm qualification fixtures. Opaque MPS storage is
excluded from that sum and included in the process-footprint guard.

## Fixed search and allocation policies

For each 256-value group, test endpoint trim ranks in this exact order:
`(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)`. Duplicate values count as
separate order statistics. The range defines F32 scale `(upper-lower)/15`
and offset `lower`; quantization and reconstruction simulate the serialized
parameters and separate F32 operations, with contraction disabled on GPU.
Largest residual ties prefer lower local indices. Loss sums unpatched F32
squared residuals through a fixed reduction tree with strides 128 down to 1.
Candidate-loss ties prefer lower candidate IDs.

- `joint_quota_weight`: independently choose the best one- and two-patch
  ranges. Rank F32 loss improvements to upgrade exactly 138,239 groups.
  Every group gets one patch; upgraded groups get two.
- `joint_global_weight`: choose the range fitted for two local patches, then
  rank all residuals of those selected bases globally for 414,719 patches.
  A group can receive zero patches or more than two.
- `joint_global_activation_f0/f1`: use the same global policy with residual
  scores multiplied by the training-fold F32 mean squared input. Fold 0 uses
  rows 0/1; fold 1 uses rows 2/3. Evaluate each on the opposite held-out rows.

These policies are finite search heuristics. Uniform quotas can waste patches,
while fitting a base for two local corrections can be suboptimal when its
eventual global allocation differs. The activation score is a diagonal proxy
that omits cross-channel covariance. Neither policy is claimed to solve a
globally optimal affine fit or budget allocation problem.

The six previous arms remain controls: Q4_K, global min/max sparse repair,
two min/max activation folds, K-channel exceptions and N-row exceptions.
All affine arms use the existing sparse/K-channel/N-row decoders. Joint modes
emit the unchanged sparse wire, with sorted positions and absolute BF16
replacements. No new decode representation or synchronization is introduced.

## Admission and verification

The source-bound launcher keeps joint evidence under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-joint-affine-20261023`
and build/cache files under the corresponding `build/wan-joint-affine-20261023`
directory. Every invocation has exclusive receipts and cannot overwrite an
earlier run. It retains the prior six hash-pinned bridge objects and records
their existing ancestry limits; source snapshots bind each new native build.
Only the T5 MPS linear bridge has complete source/object build ancestry in
the reused receipt. The other objects are pinned binaries with current source
hashes recorded, not rebuilt objects. The prior build receipt is
`wan-qbit-mps-stage-20261022-v2/build-v2-intent.json`, SHA-256
`4cf673fd89d04112bc72de3e26e98d85111876f8f9b698042df2341aa2f6bca4`.

Run from the owned `codex/wan-cross-segments` worktree:

```sh
python3 -B examples/support/wan_affine_budget_experiment.py cpu-spec joint-v3 --joint
python3 -B examples/support/wan_affine_budget_experiment.py build v2 --joint
python3 -B examples/support/wan_affine_budget_experiment.py spec joint-v2 --joint
python3 -B examples/support/wan_affine_budget_experiment.py oracle r4-v2 --rows=4 --joint
python3 -B examples/support/wan_affine_budget_experiment.py probe r4-v2 --build-tag=v2 --fixture-tag=r4-v2 --joint
python3 -B examples/support/wan_affine_budget_experiment.py oracle r4480-v1 --rows=4480 --joint
python3 -B examples/support/wan_affine_budget_experiment.py probe r4480-v1 --build-tag=v2 --fixture-tag=r4480-v1 --joint
python3 -B examples/support/wan_affine_budget_experiment.py probe r4480-v2 --build-tag=v2 --fixture-tag=r4480-v1 --joint
```

These are the executed successful receipt tags; use fresh tags to reproduce
them because the launcher refuses overwrites. Native binary SHA-256 is bound
by `build-v2.completion.json` and both full-size probe receipts.

The guard keeps a cache-aware reserve of 15% of physical memory, a 4 GiB
process-group footprint ceiling, a 64 MiB swap-growth stop, 240 seconds for
builds and 300 seconds for other invocations. File-backed pages are an
estimated reclaimable cache, not a guaranteed clean reservation.

The independent CPU serializer checks every packed byte against native GPU
packing. Independent CPU decode followed by offline PyTorch MPS `F.linear`
produces the same-format upstream reference. Native measured execution and
packing use only cogni-ml/Metal/MPS; PyTorch stays outside those intervals.
Require finite decoded weights and all projection outputs to pass the
unchanged `2e-5 + 2e-5*abs(reference)` gate before and after timing. Output
distortion against original BF16 is a separate lossy quality metric.

Twenty paired rounds give one complete forward and one reverse rotation of
all ten arms, following both warmup orders. Measure decode-only and complete
decode -> shared F32 stage -> MPS -> bias GPU intervals separately. Each
sample has one terminal wait; admit no additional in-chain host waits,
activation readbacks, or tracked Metal-buffer allocations. One-time GPU fit
and packing cost is reported separately from resident inference cost.

Tiny controls cover stretched tails, mixed tails, constants, deterministic
ties, exact budgets, monotone masked losses, quota limitations and held-out
isolation. Existing malformed-wire NaN/status rejection and comparator
finite/NaN corruption controls remain required. No full block, model, Studio,
or generation is launched. Results expire on a relevant source, payload,
input, geometry or backend change. No video-quality claim follows from a
four-row projection fixture.

## Results

### Exact execution and resource bounds

The final CPU suite passes 22 tests; the final Metal suite passes 5 examples
with no pending examples. The native constant-group control checks complete
quota/global payloads, both activation folds and deterministic zero-score ties.
Alias rejection and malformed-wire poisoning/sticky-status controls pass.

At M=4 and in both fresh M=4480 processes, all nine native GPU packs match
the independent CPU payloads byte for byte. All ten arms have bit-exact
decoded-weight and same-format upstream projection results both before and
after timing: 70,778,880 decoded values per arm and 61,931,520 projection
outputs per arm at M=4480. The original BF16 control also passes bit-exactly.
There are zero nonfinite values or strict-tolerance violations in those
checks. Injected finite and NaN corruption is rejected. This establishes
execution of each reconstructed matrix, not BF16-equivalent accuracy.

Each full-size process records 40 warmups and 400 measured samples across the
two timing modes. All timed samples have one terminal wait, zero inter-op host
waits, zero activation readbacks and zero tracked encode-time Metal-buffer
allocations. Native pack intervals also have one terminal wait and no readback
or tracked Metal-buffer allocation; byte checks occur after those intervals.
Independent reference generation and qualification are outside measured native
execution. Neither the full model nor a full block or generation was launched.

| Probe | Peak process-group footprint, GiB | Minimum estimated headroom, GiB | Minimum pressure-free % | Swap growth |
| --- | ---: | ---: | ---: | ---: |
| M=4 | 1.423 | 18.258 | 77 | 0 |
| M=4480, process 1 | 2.198 | 17.054 | 74 | 0 |
| M=4480, process 2 | 2.197 | 15.419 | 71 | 0 |

The 15% physical reserve is 9.6 GiB on this 64 GiB host. All successful
invocations return zero with unchanged bound sources. These remain sampled
guards with estimated reclaimable file cache; they cannot prevent a spike
between samples or reserve that cache for this process.

### Distortion and allocation

The table reports pooled relative L2 of the projection output against
original BF16. It is a descriptive lossy metric, separate from the unchanged
same-format tolerance. The four unique input rows are seeded diagnostic
inputs; M=4480 repeats them and adds no independent quality examples.

| Equal-size arm | Output relative L2 vs BF16, % | Weight relative L2 vs BF16, % |
| --- | ---: | ---: |
| `q4_k` | 4.63165 | 7.02861 |
| `sparse_weight` | 7.06763 | 10.64988 |
| `sparse_activation_f0` | 6.98856 | 10.84598 |
| `sparse_activation_f1` | 6.89344 | 10.85101 |
| `kchannel_weight` | 7.33323 | 10.97370 |
| `nrow_weight` | 6.98030 | 10.49539 |
| `joint_quota_weight` | 6.08900 | 9.17345 |
| `joint_global_weight` | 6.04116 | 9.10943 |
| `joint_global_activation_f0` | 6.00145 | 9.52308 |
| `joint_global_activation_f1` | 5.95922 | 9.54083 |

The uncalibrated global policy reduces output relative L2 by 14.52% relative
to the old min/max sparse policy at identical package size. It remains
1.40951 percentage points above Q4_K on these inputs. Its global allocation
gives 63,507 groups zero patches, 123,584 one, 85,555 two and 3,834 more than
two; the maximum is 166. The quota arm instead forces one patch in 138,241
groups and two in 138,239. This demonstrates the allocation distinction but
does not establish that the global heuristic is optimal.

Pooled calibration scores must not select the activation arms: both improve
their training rows but regress on held-out rows relative to the uncalibrated
global arm. Values below are means of individual row-relative L2, not pooled
L2, so their aggregation differs from the preceding table.

| Calibration fold | Training rows: uncalibrated -> calibrated, % | Held-out rows: uncalibrated -> calibrated, % |
| --- | ---: | ---: |
| f0, train 0/1, hold out 2/3 | 6.41236 -> 6.05232 | 5.72935 -> 5.96267 |
| f1, train 2/3, hold out 0/1 | 5.72935 -> 5.30782 | 6.41236 -> 6.68757 |

Reject activation-fit promotion on these folds. Four seeded inputs cannot
establish video quality or a general error distribution.

### Timing and drift

Each candidate visits every position twice and appears before Q4_K in ten
rounds and after it in ten. Ratios pair by process, mode and round; these are
medians of ratios, not ratios of separate medians. The complete interval is
decode -> reusable F32 stage -> MPS -> bias, including unpacking.

| Arm | Complete paired ratio to Q4_K, process 1 / 2 | Decode-only GPU median ms, process 1 / 2 |
| --- | ---: | ---: |
| `q4_k` | 1.0000 / 1.0000 | 2.4166 / 2.9841 |
| `sparse_weight` | 1.0096 / 1.0101 | 3.4867 / 4.6925 |
| `sparse_activation_f0` | 1.0237 / 1.0085 | 3.7709 / 4.9449 |
| `sparse_activation_f1` | 1.0112 / 1.0120 | 3.4271 / 4.8429 |
| `kchannel_weight` | 1.0345 / 1.0255 | 4.6798 / 6.8374 |
| `nrow_weight` | 0.9931 / 0.9850 | 1.1756 / 1.1780 |
| `joint_quota_weight` | 1.0360 / 1.0157 | 3.4459 / 4.5906 |
| `joint_global_weight` | 1.0147 / 1.0193 | 3.5081 / 4.9791 |
| `joint_global_activation_f0` | 1.0127 / 1.0159 | 3.5830 / 5.0299 |
| `joint_global_activation_f1` | 1.0112 / 1.0180 | 3.3515 / 4.9122 |

Every candidate's complete-path paired ratio range crosses 1.0 in both
full-size runs. For `joint_global_weight`, ranges are 0.9297-1.1792 and
0.8035-1.2494, with only 5/20 and 4/20 pairs faster than Q4_K. Observed
absolute medians also drift strongly across fresh processes: Q4_K is
71.3282 -> 140.7880 ms and `joint_global_weight` is
73.0570 -> 145.0703 ms. These are run-specific observations, not stable
throughput estimates. No cause for the drift is established here.

There is no verified complete-path speed winner, small slowdown estimate or
latency-equivalence claim. N-row's fast decode and lower ratio medians do
not establish an equal-quality winner: its BF16 distortion is worse than
Q4_K. At M=4, the new quota/global routes take 4.1534/4.1728 ms versus
3.0872 ms Q4_K, and win zero of twenty pairs; the N-row route takes 1.9497 ms
but loses on distortion. A small-M decode result does not establish a
full-size projection speedup.

One-time native GPU fit/packing on the M=4 run costs 89.675 ms for quota,
149.455 ms for global, and 154.094/152.368 ms for the activation folds;
old min/max sparse packing costs 88.677 ms. The Q4_K input payload is pinned
and its pack cost is not measured. These setup costs are separate from
resident execution and do not claim a faster offline packer. The persistent
270 MiB F32 stage is still required by every measured compressed route.

### Evidence and next falsifier

The source-bound receipts and logs live in the external evidence directory
above. Successful final receipts are `cpu-spec-joint-v3`, `spec-joint-v2`,
`build-v2`, `oracle-r4-v2`, `oracle-r4480-v1`, `probe-r4-v2`,
`probe-r4480-v1` and `probe-r4480-v2`. The independently checked aggregate is
`aggregate-v1.json`, SHA-256
`249626a37ce0b67703eec2c52e669b5d451a338b9dc764802107f5f60e35a834`;
its script hash and fixture manifest hash are embedded. Receipts bind the
commands, source closure, binary, fixture and stdout/stderr/guard hashes.
Full arrays and generated binaries stay outside Git.

Retained failed diagnostics are `cpu-spec-joint-v1` (an over-specific expected
candidate ID; a different narrower candidate validly minimized the same loss)
and `oracle-r4-v1` (missing parent directory for the quota upgrade-list file).
The test now checks the loss argmin and repaired outlier; the writer creates
the directory. Both final suites and the rebuilt final source pass. No native
timing used the failed oracle output.

The evidence adversary returns ROBUST for equal bytes, bounded same-format
execution and observed guard compliance; broader quality/generalization or
small speed claims remain VULNERABLE. The decision is to keep this opt-in
and retain Q4_K as the stronger tested accuracy control.

A structural hypothesis for the remaining error gap is range granularity:
Q4_K represents eight independent 32-value ranges within its 256-value block,
using eight six-bit scale/min pairs in twelve bytes plus F16 super-scales;
the affine competitor uses one range per 256 values plus sparse patches.
See [`gemm_q4k.metal`](../src/ml/gguf/kernels/gemm_q4k.metal) and
[`dequant.cr`](../src/ml/gguf/dequant.cr). An alternative cause is the
finite six-range search and its decoupling from final global allocation.
The next discriminating probe is a Q4_K-wire ablation tying its eight local
ranges to one shared fitted range while retaining the same 144-byte block
and re-quantizing its codes. Compare BF16 distortion and its own same-format
parity before further timing. That ablation is proposed, not implemented;
it can test range granularity without claiming a new production format.
