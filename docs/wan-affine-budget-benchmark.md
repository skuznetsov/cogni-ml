# Equal-budget Wan projection quantization

The bounded comparison is complete. Q4_K retains lower measured distortion
than the tested affine INT4 exception schemes at the same physical package
size. No complete-projection speed winner is established. No production
defaults, model checkpoints or full-generation paths changed.

The native GPU packer, decoder and projection match their independently
serialized, same-format upstream MPS references exactly. That establishes
correct execution of the compressed representations; it does not establish
equivalence to original BF16 weights or acceptable video quality.

## Frozen scope and implementation

The selected checkpoint tensor is `blocks.0.ffn.0.weight`, shape
`[13824, 5120]`, with 70,778,880 BF16 values. The measurement geometry is
M=4480, with a diagnostic at M=4. Inputs are four distinct seeded upstream
fixture rows, repeated 1120 times for M=4480. They are not captured video
activations or 4480 independent quality examples.

The opt-in implementation consists of:

- [`AffineExceptionWeights`](../src/ml/video/affine_exception_weights.cr)
  and its [Metal kernels](../src/ml/video/kernels/affine_exception_weights.metal):
  GPU-only packing, deterministic selection, validation and decode.
- [Native projection probe](../examples/wan_affine_budget_benchmark.cr)
  and [diagnostics](../examples/support/wan_affine_budget_metrics.cr).
- [Independent CPU serializer and offline MPS oracle](../examples/support/wan_affine_budget_reference.py),
  [CPU tests](../spec/wan_affine_budget_reference_spec.py), and
  [native API/invalid-wire tests](../spec/affine_exception_weights_spec.cr).
- [Guarded, source-bound runner](../examples/support/wan_affine_budget_experiment.py).

Adaptivity runs only during packing. A preallocated workspace is reused for
the five native packs and released before projection timing. The inference
chain is decode -> reusable F32 weight stage -> native MPS GEMM -> F32 bias.
It does not contain PyTorch. The independent oracle runs separately with
MPS fallback disabled; its time is excluded from the native measurements.

The fixed numerical gate is finite operands and
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`.
The comparator tests that inequality directly in F32; its reported error
ratio is diagnostic only. Distortion against original BF16 is separate.

## Equal physical budget

Every arm has a **39,813,120-byte payload (4.5 bits per weight)** and an actual
16-byte external shape/format descriptor, for **39,813,136 bytes per package**.
All directories, positions, values, alignment and padding are included.
The common F32 bias is another 55,296 bytes. No selector index file used by
the independent oracle is needed by the runtime decoder.

The descriptor is four little-endian U32 values: magic `0x31425157`, format
ID, N and K. Format IDs are 0 for Q4_K, 1 for sparse replacements, 2 for input
channels and 3 for output rows.

Affine groups contain 256 values: 128 nibble bytes, F32 scale and F32 lower
endpoint, totaling 136 bytes. Even positions use the low nibble. The base
uses min/max scale `(max-min)/15`, with `floor(normalized+0.5)` and clamping
to 0..15. Decode uses a separate F32 multiply and add, with contraction off.
No clipping or refit after exception selection is performed in this experiment.

| Arm | Included payload components | Selected exceptions |
|---|---|---:|
| Q4_K | Canonical Q4_K data and parameters: 39,813,120 B | None |
| Sparse replacements, three selectors | Affine base 37,601,280 B; U32 counts 276,480 B; U32 offsets 276,484 B; patches 1,658,876 B | 414,719 |
| Input channels (K) | Affine base 37,601,280 B; U16 map 10,240 B; padded U16 IDs 160 B; BF16 sidecar 2,184,192 B; zero padding 17,248 B | 79 of 5120 |
| Output rows (N) | U32 offsets 55,300 B; affine/BF16 rows 39,752,000 B; zero padding 5,820 B | 286 of 13824 |

Sparse records cover 1024 values. Each patch is a U16 local position plus an
absolute original BF16 replacement, sorted by position. Selection maximizes
F32 squared reconstruction residual, or that residual times mean squared
calibration input. Ties select the lower flat index. Eight GPU radix passes
select a deterministic threshold without host feedback.

Input-channel scores sum F32 squared residuals over N; output-row scores sum
over K. Both use sequential F32 accumulation and lower-index ties. K-channel
BF16 overrides are absolute replacements in the weight stage. An N-row is
either 2720 bytes of affine groups or 10240 bytes of original BF16, identified
by its directory span. No additive correction GEMM is required.

Two sparse activation folds swap calibration and held-out rows:
fold 0 trains on rows 0/1 and tests 2/3; fold 1 trains on 2/3 and tests 0/1.
The weight-only arms have no calibration rows and treat all four as held out.
They still do not provide representative video-quality coverage.

## Correctness and resource evidence

The release build passed, as did **10 CPU tests** and **3 native tests**.
The native tests include malformed sparse offsets producing NaN/status
rejection and a valid follow-up decode preserving sticky error history.

At M=4 and in both fresh-process M=4480 runs:

- All five GPU-packed payloads match every word of the independent CPU
  serialization, including metadata and padding.
- Every arm's 70,778,880 decoded weights is bit-exact to its independent
  decoder. Every projection is bit-exact to its same-shape MPS oracle,
  including all 61,931,520 outputs per arm at M=4480.
- Checks before and after timing report zero tolerance violations and zero
  nonfinite values. Native expansion/projection of original BF16 also
  matches its saved upstream control exactly.
- An injected +1 error and NaN are rejected: two tolerance violations and
  one nonfinite value. These controls are outside measured intervals.
- Every measured invocation has one terminal wait, zero inter-operator host
  waits, zero activation readbacks and zero tracked encode-time Metal-buffer
  allocations. Packing also has a separate terminal wait, outside inference.

Each timing mode has two opposite warmup orders, then 12 balanced rounds:
six forward rotations and six reverse rotations. Every arm occupies every
position twice. Complete projection and decode-only timings are separate.
Same-round ratios are descriptive pairs, not simultaneous measurements.

At M=4480, explicit buffers during packing total 2,168,976,344 bytes,
including the 338,578,880-byte reusable packing workspace, all six payloads,
validation references and diagnostic buffers. The shared F32 weight stage is
283,115,520 bytes (270 MiB). These are probe-resident bytes, not the minimum
storage of a future one-arm inference path. MPS storage is opaque and included
in the process guard, rather than the explicit-buffer sum.

The guard enforces 15% cache-aware physical headroom, a 4 GiB process-group
footprint cap, a 64 MiB swap-growth stop, 240 seconds for build and 300 seconds
for each run. Admission includes the footprint cap plus the 15% reserve.
File-backed memory is an estimate of reclaimable capacity, not guaranteed
immediate free memory. Sampling cannot prevent a between-sample spike.

| Native run | Peak process-group footprint | Minimum estimated headroom | Minimum pressure-free | Observed swap growth |
|---|---:|---:|---:|---:|
| M4 diagnostic | 1.258 GiB | 19.735 GiB | 77% | 0 |
| M4480 run 1 | 2.037 GiB | 20.039 GiB | 76% | 0 |
| M4480 run 2 | 2.039 GiB | 19.958 GiB | 76% | 0 |

No full block, model, Studio or generation was launched.

## Measured latency

GPU medians below include validation/decode, the F32 stage, MPS GEMM and bias
for the projection columns. Packing is measured once per arm and excluded
from inference. Q4_K packing is inherited from the pinned fixture and has no
new packing timing.

| Arm | M4480 projection run 1, ms | Decode run 1, ms | M4480 projection run 2, ms | Decode run 2, ms | GPU pack runs 1/2, ms |
|---|---:|---:|---:|---:|---:|
| Q4_K | 59.9755 | 2.2911 | 221.6797 | 5.7146 | Not remeasured |
| Sparse, weight score | 60.9321 | 3.4927 | 232.0196 | 8.5553 | 86.123 / 86.035 |
| Sparse, activation fold 0 | 60.7297 | 3.3890 | 236.9714 | 8.6312 | 78.755 / 78.314 |
| Sparse, activation fold 1 | 61.1161 | 3.6284 | 222.0870 | 8.7442 | 79.036 / 78.800 |
| Input channels | 61.7809 | 4.5359 | 220.1679 | 11.0294 | 11.564 / 11.333 |
| Output rows | 58.5951 | 1.1498 | 205.6494 | 1.8502 | 7.114 / 7.198 |

Run 1 has a narrow absolute range, but every candidate's matched-round ratio
range crosses Q4_K's time. Output rows win 11/12 pairs there, with ratios
0.9322..1.0467; in run 2 they win only 6/12, with ratios 0.7611..1.1733.
Q4_K itself spans 78.379..368.621 ms in run 2, versus 58.397..63.414 ms in
run 1. The source state, binary and fixtures are unchanged. The cause of this
nonstationarity is not localized. Do not pool the runs, rank their medians
as reliable speedups or attribute the drift to a specific external process.

At M4, output rows take 2.004 ms versus Q4_K's 3.135 ms and win all 12 pairs;
their decoder is cheaper in this diagnostic. This does not establish a
throughput advantage at M4480 or acceptable quantization quality.

## Distortion against original BF16

These numbers are lossy projection errors on the four source rows, expressed
as percentages. Pooled relative L2 weights rows by their reference energy;
the per-row figures are independently normalized. They are not parity passes.

| Arm | Pooled output relative L2 | Row 0 | Row 1 | Row 2 | Row 3 |
|---|---:|---:|---:|---:|---:|
| Q4_K | 4.6316% | 4.8805% | 4.9386% | 4.4631% | 4.3337% |
| Sparse, weight score | 7.0676% | 7.4870% | 7.4876% | 6.8064% | 6.6243% |
| Sparse, activation fold 0 | 6.9886% | 7.1496% | 7.1787% | 6.9460% | 6.7480% |
| Sparse, activation fold 1 | 6.8934% | 7.6238% | 7.5651% | 6.3488% | 6.2084% |
| Input channels | 7.3332% | 7.7553% | 7.7685% | 7.0399% | 6.9069% |
| Output rows | 6.9803% | 7.4011% | 7.3887% | 6.7113% | 6.5528% |

The activation folds expose calibration sensitivity: fold 0's mean held-out
row error is 6.8470%; fold 1's is 7.5944%. Fold 1's pooled 6.8934% must not
be presented as held-out accuracy. Q4_K has lower error on every source row
than every alternative in this fixture. No video-quality acceptance threshold
has been tested or inferred from these projection errors.

## Decision and next falsifier

Retain Q4_K as the control. The measured alternatives do not qualify for
promotion. This rejects this particular min/max group-256 base plus these
selectors as an improvement on this fixture; it does not reject all affine
INT4 schemes, channel exceptions or adaptive packing algorithms.

A plausible limitation is fitting the base range before selecting exceptions.
An extreme value can widen a min/max range while having zero residual itself;
repairing the largest residuals later cannot recover the lost base resolution.
This is a mathematical reason to test joint range fitting and exceptions,
not evidence that a particular new fitter will beat Q4_K.

The next bounded falsifier is an equal-budget pack-time comparison of clipped
or refitted affine bases, selecting exceptions jointly with the inlier range.
It should retain the exact byte budget, independent decoder, deterministic
ties and unchanged same-format gate, then compare held-out distortion and the
complete native route. Smaller groups or mixed group sizes are additional
packing candidates, not admitted performance claims. No new runtime policy
or full-generation authorization follows from this proposal.

All arms here use the same capacity. This comparison does not measure the
benefit of fitting a packed full model in memory, avoiding paging or avoiding
weight transfers. That capacity benefit remains a separate reason to pursue
quantization. The F32 stage also remains a separate optimization target.

## Reproduction and evidence

The experiment ran on Apple M2 Max, 64 GiB unified memory, macOS 26.6.2.
The local date is 2026-10-22; the evidence directory uses 2026-10-23 UTC.
The base revision is `620495ed119284fb9238e9300002c9f7304ca5ff`; intents record
the working source hashes and completion records confirm `changed_files=[]`.

From the repository root, the executed runner commands were:

```sh
python3 -B examples/support/wan_affine_budget_experiment.py cpu-spec v1
python3 -B examples/support/wan_affine_budget_experiment.py build v3
python3 -B examples/support/wan_affine_budget_experiment.py spec v1
python3 -B examples/support/wan_affine_budget_experiment.py oracle r4-v2 --rows=4
python3 -B examples/support/wan_affine_budget_experiment.py oracle r4480-v1 --rows=4480
python3 -B examples/support/wan_affine_budget_experiment.py probe r4-v1 --build-tag=v3 --fixture-tag=r4-v2 --rounds=12
python3 -B examples/support/wan_affine_budget_experiment.py probe r4480-v1 --build-tag=v3 --fixture-tag=r4480-v1 --rounds=12
python3 -B examples/support/wan_affine_budget_experiment.py probe r4480-v2 --build-tag=v3 --fixture-tag=r4480-v1 --rounds=12
```

All eight commands returned zero. Tags are exclusive; choose new tags when
repeating commands. The runner verifies that a probe uses a successful oracle
completion bound to the exact produced manifest and current reference sources.
It verifies the successful build's binary and source hashes before invocation.
Failed early builds and the superseded first M4 oracle are retained as history
but are not qualification evidence.

Evidence root:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-affine-budget-20261023/`.
The pinned source fixture is
`../wan-quantized-projection-20261022/fixture-4480-v3/` relative to that root.
`sources-v3/` archives the measured sources. Six reused bridge objects are
hash-pinned to the prior object receipt
`../wan-qbit-mps-stage-20261022-v2/build-v2-intent.json`.
Only the T5 MPS linear bridge has full source-to-object ancestry established
by that receipt; the other objects are binary-pinned, with current source
hashes recorded but no newly proven source-to-object ancestry. This limits
build-lineage attribution, not the same-binary arm-to-arm comparison.

Key SHA-256 bindings:

| Artifact | SHA-256 |
|---|---|
| Native `probe-v3` binary | `d4df542fce10896cebacfa41d6b69126c1f3e2e8bbc2e500f96984757e088e31` |
| M4480 fixture manifest | `ea8e03dc06e918c7a0907cf6d8c4376c965600f072480904e321e1f92fe644a6` |
| Build v3 completion | `7f474d420eb5781b5511fa3e87ce128ad50408f8b478bc74a52590a97e82acdc` |
| Native spec completion | `251bfae7446be0204d8c3f9cd22b754b1415ddb3bb50c967d3687b42a65c5f03` |
| M4480 oracle completion | `d7c1ca3d6d819d88b0b5ac48c8309782ed320e4c50de36affa15e7ce08c458c8` |
| M4480 run 1 completion | `9476b59b7410405e061cd5bd34166ebdb14738af33b9879e751406bc324fb262` |
| M4480 run 2 completion | `745db9523557c9d9be7aec331b0acf700261d6fe0d2508238463d7599206b3e4` |
| Aggregation script `aggregate-v1.py` | `fc6bf99bcda1d3865351ea0d2038a443feff414b0a575e7da9a903c11b7ab37b` |
| Aggregated metrics `aggregate-v1.json` | `ff7ebb1ff58b9f30818db84b4b4725d84b57b7355e206750bb78d30d25a0c4c4` |

The aggregation independently checks receipt/log hashes, strict parity and
per-invocation counters before summarizing timing pairs and quality rows.
The mini-Quad final audit is ROBUST for the bounded same-format correctness
and guard claims. It admits neither a speed winner nor video-quality approval.
Refresh these results after changes to source, compiler/runtime, linked
objects, device/OS, payload, input distribution, geometry or oracle. Timing
also requires fresh qualification when host scheduling changes. On refresh
failure keep production defaults unchanged and retain the last control.
