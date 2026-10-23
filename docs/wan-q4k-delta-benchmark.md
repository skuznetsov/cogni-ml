# Q4_K with sparse F16 delta corrections

Status: VERIFIED within the bounded projection experiment (2026-10-23).
Sparse corrections reduce BF16-relative distortion on the tested tensor and
inputs, with higher packed storage. The current corrected decoder does not
accelerate the complete projection. Production defaults and checkpoints are
unchanged.

## Scope and representation

The real tensor is `blocks.0.ffn.0.weight`: N=13,824, K=5,120,
70,778,880 BF16 weights. Every arm shares the byte-identical pinned canonical
ggml Q4_K payload, without refitting its scales, minima, or codes. The control
has no corrections; candidates select 1, 4, or 8 corrections per 256-value
block. These fixed quotas cover 0.390625%, 1.5625%, and 3.125% of weights.
They are a bounded packing heuristic, not a globally optimal or
activation-calibrated allocation.

Each block selects the largest absolute F32 reconstruction errors against
the original BF16 weights; equal errors prefer the lower local index.
Corrections are stored in descending error rank. Each record uses one U8
local position and one little-endian F16 delta, without padding, offsets,
or per-block counts. The delta is `F16(F32(original - decoded_base))`;
reconstruction performs a separate `F32(decoded_base + F32(delta))`.
Canonical Q4_K scale/minimum reconstruction and arithmetic order remain
unchanged, with contraction disabled. A common 16-byte descriptor contains
four little-endian U32 values: `[0x31425157, format_id, N, K]`, where the
format ID is zero for the base and `0x100 + correction_quota` otherwise.

M=4 contains four unique seeded source inputs, identical to the prior
projection experiment. M=4480 repeats them 1,120 times to exercise the larger
execution geometry. Neither fixture uses captured video activations;
repetition adds no independent quality evidence.

## Complete storage and distortion

All sizes below include the canonical payload, positions, F16 deltas, and
16-byte descriptor. There are 276,480 blocks; the canonical payload is
39,813,120 bytes, and each correction per block adds 829,440 bytes.

| Corrections per block | Sidecar bytes | Complete bytes | MiB | Bits/weight | Growth vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 (Q4_K) | 0 | 39,813,136 | 37.96877 | 4.500002 | 0% |
| 1 | 829,440 | 40,642,576 | 38.75978 | 4.593752 | 2.0833% |
| 4 | 3,317,760 | 43,130,896 | 41.13283 | 4.875002 | 8.3333% |
| 8 | 6,635,520 | 46,448,656 | 44.29689 | 5.250002 | 16.6667% |

Original BF16 weights occupy 141,557,760 bytes (135 MiB); the largest
corrected package remains 3.05 times smaller. The common 55,296-byte F32 bias
is separate. All arms reuse the same 283,115,520-byte (270 MiB) F32 decoded
stage, followed by the same MPS projection and bias. This is not a fused
quantized GEMM, and packed capacity savings do not establish model-wide
resident-memory savings. Source/expected weights and outputs are diagnostic
buffers rather than per-arm package bytes.

These descriptive lossy metrics compare the independent reconstructed
weights and four-row projection with original BF16. They are separate from
the unchanged same-format implementation gate.

| Corrections per block | Weight relative L2 | Projection relative L2 | Projection RMSE | Projection max absolute error |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7.02861% | 4.63165% | 0.0660111 | 0.458865 |
| 1 | 6.94894% | 4.57702% | 0.0652325 | 0.408526 |
| 4 | 6.74457% | 4.44768% | 0.0633891 | 0.415784 |
| 8 | 6.50725% | 4.30013% | 0.0612862 | 0.399553 |

K=1/4/8 reduce output relative L2 by 1.18%/3.97%/7.16% relative to Q4_K.
K=8 reduces weight relative L2 by 7.42%, removing 14.28% of squared weight
error while correcting 3.125% of weights. Most squared reconstruction error
therefore remains outside these selected positions. Aggregate RMSE improves
monotonically here, but maximum output error does not: K=4 exceeds K=1.
Native GPU full-weight and per-source-row distortion checks independently
confirm the trend. This does not establish perceptual or whole-model quality,
and no lossy arm passes the original-BF16 exactness threshold.

## Native latency

The timed path is entirely native cogni-ml: GPU corrected decode into the
reused F32 stage, existing MPS projection, and bias. All correction selection
and sidecar construction also run on GPU, in a separate setup interval.
PyTorch MPS with CPU fallback disabled supplies an independent offline
reference; reference exports and scalar/CPU wire diagnostics are outside the
measured native process.

Each mode has two opposite warmups and sixteen measured rounds, rotating
four arm positions and reversing the second half. A second fresh M=4480
process reverses the initial warmup and measured order. Medians below are
GPU milliseconds for complete decode -> F32 stage -> MPS -> bias, with
sixteen samples per cell.

| Geometry / process | Q4_K | +1 delta | +4 deltas | +8 deltas |
| --- | ---: | ---: | ---: | ---: |
| M=4 smoke | 3.09700 | 3.25225 | 4.35002 | 8.10585 |
| M=4480 A | 57.66921 | 57.83448 | 58.91102 | 62.67004 |
| M=4480 B, opposite initial order | 57.74969 | 57.86144 | 58.89075 | 62.78829 |

Paired baseline/candidate ratios below 1 mean the candidate is slower:

| Candidate | A median [min, max] | B median [min, max] |
| --- | --- | --- |
| +1 | 0.99344 [0.98200, 1.01157] | 0.99763 [0.98861, 1.00820] |
| +4 | 0.97812 [0.96135, 0.99406] | 0.98279 [0.97164, 0.99439] |
| +8 | 0.91845 [0.90430, 0.93730] | 0.91880 [0.91277, 0.93371] |

K=4/8 are slower in every measured full-size pair in both processes; K=8
adds about 8.8% to full-path GPU latency. The K=1 range crosses 1 in both
runs: no stable small difference or equivalence is established. No candidate
is promoted as a speed winner.

Decode-only GPU medians for Q4_K/K1/K4/K8 are 2.38823/2.48744/3.51652/7.26698
ms in A and 2.37344/2.48131/3.51625/7.26642 ms in B. Full-size median command
encoding ranges from 0.01092 to 0.02831 ms. One-time GPU correction packing
is 11.70946/11.09058/15.31725 ms in A and 15.33062/10.67663/15.35779 ms in B
for K1/K4/K8; these single setup observations are excluded from inference
intervals and are not a packing-speed comparison.

The current decoder redundantly validates K records and their duplicate
positions for every output weight, including an O(K squared) check. Thus
latency describes this implementation, not an inherent sparse-format cost
or lower bound. A possible next execution experiment is block-shared record
validation with identical failure semantics and arithmetic. Removing the
F32 stage requires a separate competitive fused decode/GEMM experiment.

## Verification and resource boundary

The 9 CPU reference tests, 12 receipt-analyzer tests, and 6 native Metal
examples pass, with zero native pending examples. The native GPU sidecars
match every independently produced byte at M=4 and in both M=4480 processes.
All four arms' full decoded weights and projection outputs before and after
timing are bit-exact with their independent same-format references. This
covers all 70,778,880 decoded weights and all 55,296 / 61,931,520 output
values per comparison. The original-BF16 native control also passes.

The finite implementation gate remains
`abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`, with finite operands.
Injected finite/NaN corruption produces exactly two bit differences, two
tolerance violations, and one nonfinite value and is rejected. Focused
controls cover deterministic ties, positions 0/255, F16 rounding, separate
F32 addition, duplicate/nonfinite records, invalid geometry/buffer aliases,
and sticky status. The analyzer rejects incomplete loss metrics, changed
pins, invalid schedules, waits, counters, and guard breaches. Its initial
v1 spec incorrectly expected the permissive corruption-metric helper to
reject a nonfinite loss; v2 mutates the actual loss-parity validator instead,
without weakening production validation.

All timed samples have one existing terminal wait, zero inter-operator host
waits, zero activation readbacks, and zero tracked explicit
`ML::MetalBuffer` allocations during encoding. This counter does not observe
MPS-private resources or compiler-private storage. `packing_workspace_bytes:
0` means no separate explicitly allocated global packing-workspace buffer;
the packer has private arrays which may use compiler-managed scratch. Opaque
resources remain covered only by the sampled process guard.

The guard retains a file-cache-aware 15% physical-memory reserve/pressure
floor, a 4 GiB process-group footprint cap, a 64 MiB system swap-growth stop,
and 240 s build / 300 s other-command deadlines. Native probe peak sampled
footprint is 1.78048 GiB, minimum estimated cache-aware headroom 21.70114 GiB,
pressure-free at least 73%, and swap growth zero. Oracle construction peaks
at 1.77213 GiB with minimum estimated headroom 12.78004 GiB and zero swap
growth. File-backed memory is a reclaimability estimate, not guaranteed
immediately available memory; guard sampling cannot exclude between-poll
spikes or deliberately escaped process groups. The reserve was not weakened.
No full model, block, Studio, or video generation was launched.

## Reproduction and provenance

Working source started at `a68f9ab5` on `codex/wan-cross-segments`. Host:
Apple M2 Max, macOS 26.6.2, Xcode 27.1 (27A9275). All build/cache/evidence files
are outside Git under `/opt/mldata_ssd/cogni-ml/video-scene-20261019`.
Let `L=examples/support/wan_q4k_delta_experiment.py`; successful DoD commands
were:

| Command after `python3 -B $L` | Expected and observed signal |
| --- | --- |
| `cpu-spec green-v1` | 9 tests, OK |
| `analysis-spec v2` | 12 tests, OK |
| `spec v2` | 6 examples, zero failures/errors/pending |
| `build v2` | Successful source-bound release binary |
| `oracle r4-v1 --rows 4` | Independent four-row reference |
| `oracle r4480-v1 --rows 4480` | Independent repeated-row reference |
| `probe r4-v2 --build-tag v2 --fixture-tag r4-v1 --rounds 16` | Exact sidecar bytes, full parity, corruption rejection |
| `probe a4480-v2 --build-tag v2 --fixture-tag r4480-v1 --rounds 16` | Same checks, sixteen full-size rounds |
| `probe b4480-v2 --build-tag v2 --fixture-tag r4480-v1 --rounds 16 --reverse-first` | Same checks, opposite initial order |

Every command returns zero with `changed_files: []`. Tags are exclusive;
reproduction requires fresh tags and matching build/fixture flags. A cosmetic
Crystal formatting change invalidated v1 native receipts for current bytes;
only rebuilt/rerun native v2 is used here. The reference producer and v1
oracles did not change. Receipt analysis was executed under the same guard:

```sh
python3 -B examples/support/wan_memory_guard.py \
  --memory-accounting=file-cache --max-footprint-mib=4096 \
  --max-swap-growth-mib=64 --timeout-sec=300 \
  --log="$EVIDENCE/analysis-v2.guard.jsonl" -- \
  python3 -B examples/support/wan_q4k_delta_results.py "$EVIDENCE" \
  r4-v2 a4480-v2 b4480-v2 > "$EVIDENCE/summary-v2.json"
```

Here `EVIDENCE` is the external base plus
`evidence/wan-q4k-delta-20261023`. Analysis returns zero with unchanged
analyzer/spec/guard pins; intent/completion receipts capture the command and
output hashes. Do not overwrite existing evidence files when reproducing.

| Completion receipt | SHA256 |
| --- | --- |
| `cpu-spec-green-v1` | `574b37fa930a30ceff51d132ccbc509579d4916f64b0e2ab84ed849fc49053be` |
| `analysis-spec-v2` | `905f292a7824ece5b897cac93e398cb6841643dd5bcba206f586683631e60576` |
| `spec-v2` | `ac8053f0261147ba90d0480a7488785be03303c09940627553e0cf2e356a0e7e` |
| `build-v2` | `023dd1d646cf90779a621a1ab43cbd7b6169d8fa463d3944c92561b3d1908913` |
| `oracle-r4-v1` | `96153fb1c9dc4e1289e2e98a974a4549a8a1631dc8943a714e1af0644ed61bc3` |
| `oracle-r4480-v1` | `a45e37418a483956f60777f9ff1df22ba863119de0db440a35f772c4a89413f7` |
| `probe-r4-v2` | `c9638bb051b50c5d66b3ed7d35233be3f9b35fd8f8a7d9db4d64e714795844d8` |
| `probe-a4480-v2` | `a01e1d46b62157f23787b748b36e86b7789ddd5e4314b797178d3aa678650ef4` |
| `probe-b4480-v2` | `138b9c859df82f03aef2be4cf813c6eca52f00606dac6b18c1af1aa221767ceb` |
| `analysis-v2` | `9e171bc37a9054dd1e2924592f13e83ee2f08177e37f1d07e57ceca723be295d` |

Final `summary-v2.json` SHA256:
`506e43e85c24eaa9a3f09ab60c837e997840a1e291730a6fbbebbb181bb5b667`.
Native binary SHA256:
`f08274a37435effd32b82f16185bfe748c8175a6458bef38a2046b4c50fa27f1`.
M4/M4480 reference manifest SHA256 values respectively:
`36b796df8ff172609cc78e75078d11dbf60446e82f3e4861fada4a2e76542891`,
`204432b92773cf54b9db9482d1216e533e249a2e3d26a9ae244c2902860a5af3`.
The inherited source fixture manifest is
`0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703`;
its BF16 weight hash is
`cd1037a3eb9c98cf5baed570e6825e7bb2463165563707cf7f1c3b367afca557`.
Canonical Q4_K payload SHA256 is
`418e4e461b5cc931b9af01e4f3c216f0da61e4fc02087c7deab425fe2e15d737`.

The canonical payload comes from pinned ggml revision
`7e4c0a96880dae4fc4268ad441f8a6446bd5460a`; small independent CPU wire controls
and full native decoding are rechecked, but its quantizer is not rebuilt.
The release build reuses six native objects pinned by prior
`wan-qbit-mps-stage-20261022-v2/build-v2-intent.json` SHA256
`4cf673fd89d04112bc72de3e26e98d85111876f8f9b698042df2341aa2f6bca4`.
Only the T5 linear bridge has complete inherited source/object build ancestry;
the other five objects have byte/current-source pins but incomplete historical
build ancestry. This qualifies observed bytes, not a fresh native-object
rebuild. Build sources are archived; analyzer/spec source pins are independent
of compiler inputs and bound by `analysis-spec-v2` and `analysis-v2` receipts.

## Decision, rollback, and continuation

The correlated Mini-Quad verdict is ROBUST for bounded same-format parity,
serialized bytes, measured distortion/latency, and observed guards. Broader
quality, global optimality, and sparse-format speed-limit claims remain
VULNERABLE and are not admitted. Keep all additions opt-in; rollback removes
the new packer/decoder, probe/reference/analyzer/spec files and documentation
links without rewriting checkpoints or changing model defaults.

For quality, the next discriminator is held-out nonrepeated Wan activations
and more projection types, comparing global/activation-aware correction
allocation with fixed quotas at equal complete bytes. For execution, first
isolate the redundant record-validation cost without changing wire format,
failure handling, or arithmetic. A competitive fused decode/GEMM without the
270 MiB stage remains a separate open frontier. Refresh on relevant source,
payload, input/geometry, compiler/native-object, device/OS/SDK, or numerical
policy changes before consuming this evidence.
