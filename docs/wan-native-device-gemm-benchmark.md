# Native F32 GEMM with direct device matrix loads

Status: bounded comparison completed (2026-10-23). On the Apple
M2 Max, the opt-in direct-load variant reduces the tested large projection
from about 282 ms to 90 ms. MPS remains faster at about 55 ms. No model,
precision or production default changes. This follows
[the shared-panel reuse experiment](wan-native-gemm-reuse-benchmark.md),
starting at `23968df1c54725e882cfad6d31aa8d382b5905e8` on
`codex/wan-cross-segments`.

## Change and safety boundary

`Q4KDeltaLinear::GemmVariant::B64N32S4DeviceF32`, selected with
`candidate-b64n32-sg4-device-f32`, is dense-only. The full-tile pipeline uses
device-address-space `simdgroup_load` for F32 A[M,K] and transposed B[N,K].
It keeps M64/N32/K32, four SIMD groups, the B64 fragment reuse schedule,
ascending K8 matrix multiply-accumulates and the F32 bias epilogue. Operand
panels and their per-K threadgroup barriers are removed; an output-only
shared panel and one epilogue barrier remain. Compiled static threadgroup
storage is 8,192 bytes, versus 12,288 for legacy and the safe staged tail.
Pipeline compilation and validation occur in setup, outside encoding.

The host partitions output into complete direct tiles, a right fringe over
complete rows, and a bottom fringe over every column. These regions are
disjoint and exhaustive. Bounds-masked staged loads handle the fringes;
there are no masked out-of-bounds device SIMD matrix loads. All dispatches
share the caller's encoder. M4480/N13824/K5120 uses one direct dispatch
with 432 by 70 threadgroups and no tail. M4 uses one staged bottom dispatch
with 432 groups and no direct dispatch: its timing is a fallback control.
Other shapes need at most three dispatches. Packed `linear` and fused
benchmark selection reject this dense-only selector before encoding.

This CAUTION experiment retains legacy native and staged MPS rollback.
The falsifier was whether eliminating staging lowers paired latency despite
duplicate device reads. Strongest failure modes were tail memory safety,
overlapping writes, changed arithmetic and timing-order effects. The gate
remains finite `abs(actual-ref) <= 2e-5 + 2e-5*abs(ref)`; no precision change
or relaxed tolerance is admitted. These measurements do not isolate direct
loads from the earlier fragment/layout improvement because the timed native
control is legacy, rather than a fourth matched BT32 arm.

## Dense timings

The real Wan projection is `blocks.0.ffn.0.weight`, N=13,824 and K=5,120.
Legacy, candidate and MPS consume the same GPU-decoded Q4_K F32 coefficients,
input and bias. Decode is outside dense timing. M4 has four unique seeded
non-video input rows; M4480 repeats them 1,120 times. This is an execution
probe, not representative video quality or a BF16 quality-equivalence test.

Each fresh process uses two warmup rounds and 16 measured rounds, with two
samples per arm per round. Rotated ABC+CBA palindromes cover all six
rotation/direction orders. The second half changes base direction, and B
starts with the opposite direction via `--reverse-first`. Sixteen rounds
give unequal rotation counts; the summary retains actual order/half/rotation
groups. Processes are not pooled. GPU medians in milliseconds:

| Geometry / process | Legacy native | Direct-load candidate | MPS |
| --- | ---: | ---: | ---: |
| M4, staged fallback only | 3.10277 | 3.05521 | 0.79588 |
| M4480 A | 281.63781 | 89.55925 | 54.91767 |
| M4480 B, opposite initial order | 281.51931 | 89.69533 | 55.14894 |

The paired statistic is the median of same-round candidate/control ratios,
using each arm's two-sample mean:

| Geometry / process | Candidate / legacy median (range) | Candidate / MPS median (range) |
| --- | ---: | ---: |
| M4, staged fallback | 0.98547 (0.96892-0.99469) | 3.81666 (3.73742-3.89577) |
| M4480 A | 0.31650 (0.31263-0.31936) | 1.61947 (1.59917-1.64545) |
| M4480 B | 0.31715 (0.31470-0.32063) | 1.61727 (1.59486-1.64149) |

The large candidate is about 3.15 times faster than legacy, with about
68.3% less GPU latency. Both halves and every order/rotation group preserve
the direction in each process; even the largest individual candidate/legacy
round ratio is 0.32064. Candidate/legacy half medians are 0.31650/0.31664
in A and 0.31776/0.31676 in B. Every candidate/MPS round remains above one.
The M4 saving belongs to the inherited staged-tail schedule and does not
demonstrate direct-load acceleration. The large native gain does not
establish a hardware cause, model-wide latency or packed-weight performance.
The remaining MPS gap is material.

## Verification and resource controls

The final native suite passes 19 examples, including direct-only, tail-only
and mixed core/right/bottom cases with non-dyadic heterogeneous data, scalar
GPU FMA and MPS controls, poisoned outputs and 16 trailing canaries. It also
retains existing malformed-record, signed-zero, status, alias, size and
allocation guards. Direct-load shapes include M64/N32/K256, M65/N35/K512,
M72/N40/K256 and M129/N72/K512; M4/N35/K256 tests fallback. Canaries guard
the output suffix; they are not a general proof of all memory safety.

Independent upstream PyTorch MPS references are reused unchanged from the
parent experiment with CPU fallback disabled. PyTorch is outside every
timed native process. Full coefficients and outputs are checked before and
after timing, with finite/NaN corruption controls. Same-format parity does
not establish original-BF16 quantization quality. All 36 parity metric
groups have zero nonfinite values and tolerance violations. M4480 A and B
are bit-exact for full coefficients and all three full projection outputs
before/after timing. M4 candidate and legacy are bit-exact with each other;
their differences from upstream/MPS have maximum absolute error
`2.193451e-5` and maximum tolerance ratio `0.33169478`, inside the unchanged
gate. Injected finite and NaN corruption is rejected, and the final observed
GPU status is zero in all three probes.

All 324 timing samples, including warmups, retain one existing terminal wait
and zero new
inter-operator host waits, activation readbacks or tracked explicit
encode-time `ML::MetalBuffer` allocations. Diagnostic readbacks are outside
timing. Local GPU barriers are not host waits. Opaque MPS/compiler resources
are excluded from the explicit buffer counter but included in sampled
process accounting. No global F32 weight-stage reduction is claimed by this
dense experiment; the 270 MiB decoded matrix remains resident.

The guard includes reclaimable file-backed cache in estimated headroom,
without double-counting speculative pages. It preserves the 15% physical
reserve and pressure floor, 4 GiB process-group cap, 64 MiB system swap-growth
stop and deadlines. Polling cannot exclude between-sample spikes, and file
cache reclaimability remains an estimate. Across the three native probes,
sampled process-group footprint peaks at 2.46706 GiB, minimum estimated
cache-aware headroom is 18.16843 GiB, pressure-free is at least 71%, and
system swap growth is zero. No full model, block, Studio or
generation is launched. Temporary/build/cache/evidence files remain under
`/opt/mldata_ssd`.

## Reproduction and lineage

Let `L=examples/support/wan_q4k_delta_experiment.py`, run as `python3 -B $L`
from this worktree. Each invocation requires a fresh exclusive tag.

| Arguments | Observed signal |
| --- | --- |
| `spec dense-device-native-v2` | 19 examples, zero failures/errors/pending |
| `build dense-device-v1` | Source-bound release build |
| `analysis-spec dense-device-analysis-v2` | 32 tests, OK |

Probe arguments are `probe TAG --build-tag dense-device-v1 --fixture-tag
FIXTURE --compare-gemm --gemm-variant=candidate-b64n32-sg4-device-f32`.

| Tag | Fixture | Extra flag |
| --- | --- | --- |
| `dense-device-r4-v1` | `fused-r4-v1` | None |
| `dense-device-a4480-v1` | `fused-r4480-v1` | None |
| `dense-device-b4480-v1` | `fused-r4480-v1` | `--reverse-first` |

Evidence base:
`/opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-q4k-delta-20261023`.
M4/M4480 manifest SHA256:
`36b796df8ff172609cc78e75078d11dbf60446e82f3e4861fada4a2e76542891` /
`204432b92773cf54b9db9482d1216e533e249a2e3d26a9ae244c2902860a5af3`.

Build intent SHA256:
`7f702f14ee27e2a2fe3bbf878d963da97856d621c9bcf094cd7fca370ea9d272`.
Final native-suite completion SHA256:
`310c1653d12095cb9e2295604cec839e78f8143459b5fddc4250ae1f5bd4686a`.
Final analyzer-suite completion SHA256:
`4f9417f8cd1ffc1865fe9d40af950d3777c778bd499f33e1f73b52e23f3adb45`.

The guarded strict analyzer consumes all three source-bound probe receipts,
logs and guard samples. `analysis-native-device-v1` returns zero with
`changed_files: []`. External `summary-native-device-v1.json` SHA256:
`55db527c28da55ebb8c07fedaa47eab4670595bc245ae80105bee70dca20784c`.
Analysis intent / completion SHA256:
`27b77e1360677ff33a0eff7c24eadc950d40e69ccafda07f432510eb11f1c82d` /
`f57a3459ca4369aba471550bed947e6ff5a0e802e96444a12005ed65efd9b9b4`.
The exact guarded command is in the intent. Its pinned wrapper is
`build/wan-q4k-delta-20261023/analyze-native-device-v1.py` under the external
experiment base; run it with `python3 -B`, using a fresh output prefix for
repeats. It refuses to overwrite evidence.

Red receipts are retained. The first native test build found an invalid
matcher invocation, corrected in v2 before GPU qualification. The first
analyzer suite found an incorrect M4 expected metadata field: direct columns
retain the floor-aligned N even when direct rows/dispatches are zero. Only
the expected field changes for v2; kernel, wrapper, benchmark and analyzer
executable remain unchanged. Standalone `metal` lacked its optional toolchain;
a guarded safe-math `newLibraryWithSource` compile verified device loads and
transposed B on this machine, followed by actual native GPU tests.

All successful launcher receipts must have `changed_files: []`. Exact build
archives bind probe source pins; complete oracle subsets and external
fixture/object/binary pins remain mandatory. Negative analyzer tests reject
changed geometry, partition, pipeline resources, historical archive mutation
and symlink redirection. This proves local byte consistency, not signed
provenance against coordinated rewriting of all artifacts. Six native
objects and the ggml quantizer are pinned and reused unchanged; only the T5
linear bridge has complete historical source/object ancestry, while five
other bridges have current pins without reconstructed historical lineage.
The prior nine CPU reference tests are inherited, not rerun in this slice.

## Decision and refresh boundary

Keep MPS as the faster control and retain legacy production selection. The
new direct-load path stays opt-in. A packed Q4_K/delta consumer needs its
own full cost and numerical comparison; the dense result does not transfer
automatically. Optional pipeline preparation is only qualified on the
tested device/OS/SDK. Refresh after source/dispatch, input/geometry,
payload/sidecar, compiler/native object, device/OS/SDK, numerical-policy or
timing-schedule changes. Archived measurements remain scoped to their
captured bytes.

Correlated Mini-Quad: ROBUST for the bounded direct/tail execution, parity,
failure controls, observed resource guards and two-process native speedup;
VULNERABLE for portability, representative quality, causal hardware
attribution, model-wide benefits or superiority to MPS. The adversary audit
independently recomputes raw parity/timing/guard totals and verifies the
receipt chain, but same-lineage agreement does not diversify model risk.
