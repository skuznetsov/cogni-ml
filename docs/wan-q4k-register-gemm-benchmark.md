# Q4_K sparse-delta GEMM with register reconstruction

Status: bounded implementation/parity checks pass; speed hypothesis rejected
(2026-10-23). Starting at
`a116784f1a817f818414dc4be998bf790c9949ca` on `codex/wan-cross-segments`.
The target is complete packed projection latency and elimination of the
candidate's global decoded F32 weight stage. No model or default changes.
This follows the [direct-device dense F32 experiment](wan-native-device-gemm-benchmark.md).

## Change and falsifier

The opt-in `candidate-b64n32-sg4-device-q4k` selector uses M64/N32/K32 and
four SIMD groups. Complete output tiles load F32 activations directly from
device memory and reconstruct canonical Q4_K plus sparse F16 corrections
inside the B matrix fragments. A small exact ID matrix, loaded with the
same matrix API, supplies opaque element coordinates; no fixed lane-to-matrix
formula is assumed. This association needs qualification for the compiled
backend and device. It is not a future-hardware portability promise.

The candidate preserves separate F32 coefficient reconstruction and addition
of half-rounded deltas, the corrected-format addition of zero for unmatched
positions, ascending K8 matrix MACs and the existing F32 bias epilogue.
Corrections are checked once per output-channel/256-K block in each output
row tile, preserving duplicate/nonfinite block isolation and sticky status.
Disjoint right and bottom fringes retain masked staged loads. Dense controls
delegate the already-qualified direct-device F32 core and its safe tail.

The CAUTION falsifier is whether removing operand panels improves complete
packed projection time despite duplicate coefficient decoding. Register
reconstruction occurs across row-half SIMD groups and M tiles; fewer resident
bytes alone do not prove lower kernel latency. Compare full packed GEMM/bias
with block-validated decode -> reused F32 stage -> MPS -> bias, including
decode in the control. Keep legacy and staged MPS as rollback.

## Scope and resource accounting

The real Wan matrix is `blocks.0.ffn.0.weight`, N=13,824 and K=5,120.
M4 uses four seeded non-video input rows; M4480 repeats those rows 1,120 times.
The unchanged independent upstream MPS oracle reconstructs and projects the
same packed matrix. Same-format parity does not establish original-BF16
equivalence, representative video quality or full-model performance.
The retained fixture reports projection relative-L2 loss against BF16 of
4.63165%, 4.57702%, 4.44768% and 4.30013% for 0/1/4/8 corrections on those
four unique rows. These are inherited descriptive quality measurements,
not new video-quality results or part of the strict implementation gate.

The coefficient gate remains finite
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`; tolerances and formats
are unchanged. Ordinary timed samples retain one existing terminal wait,
zero inter-operator host waits, activation readbacks or tracked explicit
encode-time `ML::MetalBuffer` allocations. Diagnostic checks are outside
timing. Local GPU barriers are not host synchronization. Opaque backend
allocations are outside the explicit counter and inside sampled accounting.

For this 70,778,880-weight matrix, packed storage including the 16-byte
descriptor is:

| Corrections per 256 weights | Bytes | MiB | Bits per weight |
| --- | ---: | ---: | ---: |
| 0 | 39,813,136 | 37.96877 | 4.500002 |
| 1 | 40,642,576 | 38.75978 | 4.593752 |
| 4 | 43,130,896 | 41.13283 | 4.875002 |
| 8 | 46,448,656 | 44.29689 | 5.250002 |

The candidate binds zero global decoded F32 weight-stage bytes; the staged
control needs 283,115,520 bytes (270 MiB). The combined diagnostic retains
that control stage and oracle/output buffers, so it cannot measure deployed
process-footprint savings. Original BF16 weights alone occupy 135 MiB.

The launcher keeps cache-aware 15% physical reserve and pressure floor,
4 GiB process-group cap, 64 MiB system swap-growth stop and bounded deadlines.
File-backed cache reclaimability is an estimate; sampled guards cannot
exclude between-sample peaks. Builds, caches, fixtures and raw evidence stay
under `/opt/mldata_ssd`. No full model, block, Studio or generation is run.

## Verification and timings

Apple M2 Max, 64 GiB unified memory. M4480/N13824/K5120 uses one direct
dispatch, with no fringe. M4 uses only the staged tail and does not qualify
the register core's performance. GPU medians include reconstruction, GEMM
and bias in both arms; the MPS control includes full decode on every sample.

| Format | M4 staged MPS / fused, ms | M4480 A staged MPS / fused, ms | M4480 B staged MPS / fused, ms |
| --- | ---: | ---: | ---: |
| Q4_K | 3.24646 / 4.41294 | 58.00217 / 555.20260 | 61.28333 / 556.34748 |
| +1 delta | 3.34250 / 4.86360 | 58.07331 / 593.46319 | 61.25994 / 594.35367 |
| +4 deltas | 3.39052 / 5.79613 | 58.08033 / 768.36752 | 61.36163 / 770.45600 |
| +8 deltas | 4.92592 / 7.06085 | 59.70050 / 932.68902 | 63.01529 / 935.93948 |

Paired fused/control GPU ratio medians, computed from each round before
aggregation, are 9.57133x/9.09536x for Q4_K, 10.21756x/9.69004x for one
delta, 13.23598x/12.54577x for four, and 15.61705x/14.84813x for eight
in M4480 A/B. Every measured round loses. M4 paired ratios also lose:
1.35821x, 1.45656x, 1.70878x and 1.43302x respectively.

Dense controls remeasure direct F32 native/MPS at 88.52850/54.82242 ms
in A and 97.88306/60.56052 ms in B; M4 is 3.03204/0.77117 ms.
The approximately 10% control drift is preserved rather than pooling the
two processes. Packed fused medians vary by less than 0.4% between A/B.
There is no matched old packed-kernel arm in this slice: cross-session
numbers cannot establish an improvement or regression against that kernel.

The guarded `spec packed-device-native-v2` passes 23 examples with no
failures, errors or pending examples. It covers full/tail/mixed partitions,
K256/512 and all 0/1/4/8 correction quotas, all four SIMD groups' coordinate
round trips, exact fragment coefficients across K=248/256, signed zero,
malformed-block/neighbor isolation, sticky status and pre-encode alias/size
checks. Coordinate scratch is fenced before epilogue reuse. Output poison
and suffix canaries provide bounded coverage, not universal memory-safety
proof. `analysis-spec packed-device-analysis-v2` passes 35 tests, including
rejection of altered packed/delegated-dense pipelines, geometry, resource
metadata, dispatch partitions and broken receipt/archive ancestry.

Three native probes and strict receipt analysis pass. Across 168 same-format
parity metric groups plus six BF16-control groups, there are zero nonfinite
values and zero tolerance violations. Full M4480 coefficients and outputs,
including fused/staged comparisons, are bit-exact before and after timing
in both processes. M4's maximum tolerance ratio is 0.33293873, maximum
absolute error 2.336502e-5, within the unchanged relative-plus-absolute gate.
Injected finite and NaN errors are rejected, and terminal GPU status is zero
in all three probes. Independent upstream fixtures and their complete
source pins are reused unchanged; no new PyTorch run or timed PyTorch path
was introduced.

All 636 timing samples (576 measured and 60 warmups) retain one terminal
wait and zero inter-operator host waits, activation readbacks or tracked
explicit encode-buffer allocations. Existing CPU inspection of terminal
status words occurs after completion, outside the GPU interval; zero
activation readbacks does not mean zero CPU inspection of status. Sampled
process-group footprint peaks at 2.02469 GiB; minimum estimated cache-aware
headroom is 23.43906 GiB, pressure-free at
least 70%, system swap growth zero. The 15% reserve/floor, 4 GiB group cap,
64 MiB swap-growth stop and deadlines remain unchanged. These are sampled
observations with the accounting limits above.

The first native spec attempt rejected `fragment` as a reserved Metal
identifier; renaming the new locals/parameters to `matrix_fragment` made
compilation pass without changing arithmetic. The first analyzer suite
exposed old synthetic F32 fixtures missing the newly required generic
dense-memory field. Only those test fixtures were corrected; the validator
and numerical gates were retained. Both failed receipts remain external.

## Decision and remaining question

Keep `candidate-b64n32-sg4-device-q4k` opt-in and keep staged MPS as the
faster control. The requested no-global-stage implementation is established
for these inputs, but it does not solve the speed target. The full-tile
kernel uses 10,368 bytes compiled static threadgroup memory, versus 14,464
for its masked tail; it has no full decoded-weight stage. Coefficients are
still reconstructed in registers: this is not arithmetic on packed bits
without reconstruction.

Source inspection identifies repeated coordinate loads and coefficient
decoding for each K8 fragment, row-half SIMD group and M tile. These are
candidate causes, not measured hardware bottlenecks. Capturing coordinates
once is a separate falsifier, and the size of the observed gap does not
justify assuming it will close that gap. Another separate candidate is a
bounded rolling decoded panel feeding MPS, to preserve its throughput while
reducing the 270 MiB control stage. Neither follow-up is implemented or
measured here. Reduced binding inventory remains a valid benefit even
though this latency experiment is red.

Correlated Mini-Quad source/raw-evidence review is overall VULNERABLE because
the acceleration objective is falsified. It is ROBUST for bounded
parity, observed contracts/guards, stage-binding elimination and rejection
of the speed claim. It is VULNERABLE for backend portability, representative
video/BF16 quality, causal hardware attribution, deployed process savings
and model-wide speed or memory benefits.

## Evidence lineage

External root: `/opt/mldata_ssd/cogni-ml/video-scene-20261019/`.
Evidence is under `evidence/wan-q4k-delta-20261023/`; build products and
temporary files are under `build/wan-q4k-delta-20261023/`.
All successful completions have return code zero and `changed_files: []`.
All 248 live build source pins still match the captured source state.
The build archive is `sources-packed-device-v1/` under the evidence folder.

| Receipt or artifact | SHA256 |
| --- | --- |
| `spec-packed-device-native-v2.completion.json` | `c9708deb19209192a89ef4d2198f6381bfdab6ba3044b04ec78227ff5aaeea13` |
| `analysis-spec-packed-device-analysis-v2.completion.json` | `c15f44ab70818d451a915a24f75404324912ef18644f9bba66fd4b6b32a2f9f2` |
| `build-packed-device-v1.intent.json` | `110caa85f1ca6acba14500453ab2e40daa7b9e726868134a40fd63bdac5bc6e9` |
| `build-packed-device-v1.completion.json` | `158a0bba6ce2aca5cce43be0680ad54eb0706bb200eeb9d98b000699c53b2e1e` |
| `probe-packed-device-r4-v1.completion.json` | `c6d99c2d7a0e201334dff142edb0872d1013266b4a127a8a6fd7814a20e1b093` |
| `probe-packed-device-a4480-v1.completion.json` | `2a3fceca73895f38317208701fc1ce831afe9536ed6a14ef0575929024b160ea` |
| `probe-packed-device-b4480-v1.completion.json` | `ea748c7b38f3c36d97ae272b4a3452f2e9bc1f7c6f09510aabb135833889a7e3` |
| `analysis-packed-device-v1.completion.json` | `1525ca0082730b5cbdc2510810b07d6fd63b225a2c6af2d33bc18f8701d217cd` |
| `summary-packed-device-v1.json` | `cc016510afc022266f189bca7d372c4be29d2a1b03bdff1bd03439032f7ee3ed` |
| `probe-packed-device-v1` binary (build folder) | `dff2df0d409a4c42109d710f07b0fc02a247eb333fac056123e79f5a61abecc1` |

Inherited M4/M4480 fixture manifest hashes are respectively
`36b796df8ff172609cc78e75078d11dbf60446e82f3e4861fada4a2e76542891`
and `204432b92773cf54b9db9482d1216e533e249a2e3d26a9ae244c2902860a5af3`.
The six reused native objects retain the pinned
`evidence/wan-qbit-mps-stage-20261022-v2/build-v2-intent.json` receipt,
SHA256 `4cf673fd89d04112bc72de3e26e98d85111876f8f9b698042df2341aa2f6bca4`.
Only the T5 linear bridge has complete historical source/object ancestry;
the other five objects have current exact pins without a reconstructed
historical compilation chain. The quantizer is pinned, not rebuilt here.

## Reproduction and refresh boundary

Use `examples/support/wan_q4k_delta_experiment.py` with fresh exclusive tags
for `spec`, `analysis-spec`, `build` and `probe --compare-fused
--gemm-variant=candidate-b64n32-sg4-device-q4k`, then strict
`examples/support/wan_q4k_delta_results.py` analysis. M4 and two fresh M4480
processes retain the existing balanced paired schedule, with opposite initial
orders in the M4480 pair. Full pre/post upstream and fused/staged comparisons,
finite/NaN corruption controls and terminal GPU status remain mandatory.

Executed launcher tags were `spec packed-device-native-v2`,
`analysis-spec packed-device-analysis-v2`, `build packed-device-v1`,
and the following probe arguments:

```text
probe packed-device-r4-v1 --build-tag packed-device-v1 --fixture-tag fused-r4-v1 --rows 4 --rounds 16 --compare-fused --gemm-variant=candidate-b64n32-sg4-device-q4k
probe packed-device-a4480-v1 --build-tag packed-device-v1 --fixture-tag fused-r4480-v1 --rows 4480 --rounds 16 --compare-fused --gemm-variant=candidate-b64n32-sg4-device-q4k
probe packed-device-b4480-v1 --build-tag packed-device-v1 --fixture-tag fused-r4480-v1 --rows 4480 --rounds 16 --compare-fused --gemm-variant=candidate-b64n32-sg4-device-q4k --reverse-first
```

The source-pinned external `analyze-packed-device-v1.py` wrapper ran
`examples/support/wan_q4k_delta_results.py` over these three tags under
`wan_memory_guard.py --memory-accounting=file-cache --max-footprint-mib=4096
--max-swap-growth-mib=64 --timeout-sec=300`. It refuses to overwrite existing
analysis evidence. Reproduction requires new tags and a new wrapper output
prefix, not overwriting these receipts.

Refresh qualification after kernel, coordinate mapping, codec, fixture,
geometry, compiler/backend or hardware changes; refresh timings on the
current host. Model-wide admission and representative quality remain open.
