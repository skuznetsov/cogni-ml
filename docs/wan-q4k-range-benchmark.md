# Same-wire Q4_K range-count ablation for a Wan projection

Status: VERIFIED within the bounded projection experiment (2026-10-23).
Eight local ranges reduce BF16-relative output error by 31.27% against one
tied range at identical bytes under the matched fitter. No latency winner
is established. Model defaults and checkpoints are unchanged.

## Question and controls

Does eight-range fitting explain part of the gap between canonical Q4_K and
the previous one-range affine INT4 representation at the same total bytes?
The two new native GPU packers use one identical min/max rule with eight
32-value groups (`q4_k_minmax8`) or one 256-value group (`q4_k_minmax1`).
The unchanged pinned ggml payload (`q4_k`) is a third control. Its weighted
iterative fitter differs from this simple min/max fitter. Only the comparison
between the two new arms isolates range count under the specified rule;
comparison to ggml cannot attribute the entire gap to group size.

The selected real BF16 tensor is `blocks.0.ffn.0.weight`: N=13,824, K=5,120,
70,778,880 values. M=4 contains four unique seeded upstream inputs. M=4480
repeats them 1,120 times to exercise the larger projection geometry, without
adding independent quality examples or real-video activations.

Each arm has 276,480 canonical Q4_K blocks, each exactly 144 bytes:

| Component per block | Bytes |
| --- | ---: |
| F16 super-scale `d` and minimum super-scale `dmin` | 4 |
| Eight six-bit scale/min pairs | 12 |
| 256 four-bit codes | 128 |
| Total | 144 |

The shared-range arm repeats its one fitted pair eight times in the existing
fields. Retaining those bytes deliberately holds the wire format, storage,
decoder and execution geometry fixed. It does not claim that one range needs
all twelve parameter bytes in a newly designed format.

Payload size is 39,813,120 bytes per arm, plus an identical 16-byte external
shape/format descriptor: 39,813,136 bytes per complete package. The shared
55,296-byte F32 bias is separate. There are no correction sidecars, directories
or hidden padding. All arms reuse the same 283,115,520-byte (270 MiB) F32
decoded-weight stage and MPS projection plan. Opaque MPS allocations are
included in the process guard rather than the explicit buffer sum.

## Pinned fitting rule

For each local group choose `lo=min(0,min(weights))`,
`hi=max(0,max(weights))`, raw scale `(hi-lo)/15` and offset `-lo`.
For the shared arm broadcast its one pair to all eight wire fields. Choose
`d=F16(max(raw_scale)/63)` and `dmin=F16(max(offset)/63)` with round-to-nearest
even. Round raw-scale/d and offset/dmin with `floor(x+0.5)` and clamp to 0..63.
Quantize each nibble against the *serialized, reconstructed* `d*scale` and
`dmin*minimum`, with `floor((weight+dmin*minimum)/(d*scale)+0.5)` clamped to
0..15. Every intermediate is F32 and GPU contraction is disabled.

A zero denominator gives zero fields/codes. Zero-valued super-scales are
canonical positive zero. Nonfinite inputs, nonfinite range arithmetic,
F16 overflow and positive super-scale underflow are rejected. Native packing
sets sticky status and initializes the invalid block to zero bytes; the
caller must reject that pack after completion. A valid zero block has zero
status. This is an offline packer contract, not a new production loader.

The finite same-format gate stays `2e-5 + 2e-5*abs(reference)`. BF16-relative
weight/output distortion is a separate lossy measurement and cannot relax
that gate. This experiment does not assume a globally optimal range fit.

## Verification and measurement contract

An independent CPU serializer supplies the two expected candidate payloads.
An independent CPU wire reader decodes all arms, including pinned ggml;
upstream PyTorch `F.linear` on MPS with CPU fallback disabled supplies the
projection references. CPU fixture work and reference copies are outside the
measured process. The timed path is native cogni-ml throughout.

Every new native GPU pack must match every expected payload byte. The existing
`PackedWeightDecode.q4_k` decodes every arm; decoded weights and full projection
outputs are checked before and after timing. The BF16 expansion/projection is
also qualified against its saved upstream result. Tiny specs cover exact wire
controls and invalid inputs; injected finite/NaN output corruption must fail
the comparator. The fixture and source/object hashes bind each receipt.

After two opposite warmups, twelve measured rounds rotate three arm positions
with both forward/reverse orders. A second fresh full-size process reverses
the first warmup and measured ordering. Report complete decode -> shared F32
stage -> MPS -> bias, and decode-only intervals separately. One-time GPU
packing has its own interval; byte validation is outside it. GPU completion
and reduced diagnostic reads are terminal. No inter-operator host wait,
activation readback or tracked encode-time Metal-buffer allocation is admitted.

The existing sampled guard retains a 15% cache-aware headroom/pressure floor,
4 GiB process-group footprint cap, 64 MiB system swap-growth stop, and 240s
build/300s execution deadlines. Reclaimable file-cache headroom is an estimate,
not a memory reservation or protection from inter-poll spikes. No full model,
block, Studio or video generation is allowed by this experiment.

## Evidence and results

### Numerical results

The 13 CPU reference tests, 8 receipt-analyzer tests and 5 native Metal
examples pass, with zero native pending examples. Both candidate GPU packs
match all 39,813,120 expected bytes at M=4 and in both fresh M=4480 processes.
All arms' full decoded-weight and projection comparisons before and after
timing are bit-exact against the independent same-format references. This
covers 70,778,880 decoded weights and all 55,296 / 61,931,520 output values per
comparison. The unchanged finite tolerance also passes. The original BF16
projection control passes; injected finite/NaN corruption produces exactly
two bit differences, two tolerance violations and one nonfinite value.

The independent wire decoder also reproduces all 70,778,880 pinned ggml
decoded weights exactly. Exact six-bit high/low-field controls, half-rounding
boundaries, positive/negative constants, canonical signed-zero handling,
invalid geometry/buffer aliasing, nonfinite values, half overflow/underflow
and sticky error status are exercised by the focused specs. The analyzer
checks complete metric counts, both warmup and measured schedules, receipt /
binary / fixture / producer lineage, the corruption control and actual guard
limits and samples. Its positive synthetic receipt initially had schema/hash
fixture mistakes in v2/v3; those failed attempts are retained. The corrected
fixture passes in v4 without weakening validation.

These lossy metrics use the four unique seeded source rows at M=4, separately
from exact same-format execution parity:

| Arm | Weight relative L2 vs BF16 | Projection relative L2 vs BF16 |
| --- | ---: | ---: |
| Canonical ggml Q4_K | 7.02861% | 4.63165% |
| Matched min/max, eight ranges | 7.71828% | 5.08291% |
| Matched min/max, one tied range | 11.06873% | 7.39585% |

Eight versus one reduces weight relative L2 by 30.27% and output relative L2
by 31.27% at the same package size, improving every one of the four source
rows. This supports local range granularity for this tensor and fitter.
Canonical ggml remains more accurate; its weighted iterative fit is a
competing explanation for the remaining difference. Repeating the four rows
at M=4480 gives the same conclusion but adds no independent quality evidence.
Neither whole-model fidelity nor perceptual video quality is established.

### Native latency and memory

Medians below are GPU milliseconds for complete decode -> shared F32 stage
-> MPS -> bias. Each cell has 12 measured observations; A/B use fresh processes
and opposite initial warmup / measured ordering.

| Geometry / process | ggml Q4_K | Eight ranges | One tied range |
| --- | ---: | ---: | ---: |
| M=4 smoke | 3.0858 | 3.0881 | 3.0939 |
| M=4480 A | 60.6435 | 61.8793 | 60.1917 |
| M=4480 B | 59.9058 | 60.8419 | 60.1302 |

Paired one/eight GPU ratios are median 0.98095, range 0.93485–1.05999 in A,
and median 0.99021, range 0.94861–1.06053 in B. Eight/ggml ratios are median
1.00666, range 0.96086–1.07664 in A, and median 1.01974, range
0.94263–1.08789 in B. All ranges cross 1: these runs establish neither a
latency winner nor equivalence. All arms use the identical decoder and MPS
geometry; tying fields does not specialize execution or reduce traffic.

Decode-only medians for ggml/eight/one are 2.2661/2.2702/2.2725 ms in A and
2.2848/2.3240/2.2998 ms in B. Full-size median command encoding is
0.0465–0.0521 ms. One-time eight/one GPU packing is 13.1808/6.5764 ms in A
and 9.2788/6.7160 ms in B; these are single setup observations per process,
not a packing-speed benchmark. Packing is excluded from inference intervals
and no ggml packing interval is measured. The aggregate retains GPU, wall and
encoding min/median/max values and all paired ratios.

All timed samples report one terminal wait and zero inter-operator host waits,
activation readbacks and tracked encode-time Metal-buffer allocations. Native
probe process-group footprint peaks at 1.60784 GiB, minimum sampled cache-aware
headroom is 16.62109 GiB, pressure-free is at least 73%, and sampled swap
growth is zero. Across all final DoD commands including oracle construction,
the footprint maximum is 1.75083 GiB and minimum estimated headroom is
16.40425 GiB. The 15% / 4 GiB / 64 MiB guards and deadlines remain unchanged.
Sampling cannot exclude between-poll spikes. The reused 270 MiB F32 stage is
still required: smaller persistent weights do not remove its traffic.

### Reproduction and evidence pins

Working source started at `f5db3be8` on `codex/wan-cross-segments`. Host is
Apple M2 Max, macOS 26.6.2 (25G83), Xcode 27.1 (27A9275), Crystal 1.22.0 /
LLVM 23.1.3. The launcher stores all cache, build and evidence files under
`/opt/mldata_ssd/cogni-ml/video-scene-20261019`, without adding binary fixtures
to Git. Let `L=examples/support/wan_q4k_range_experiment.py`; the observed DoD
commands and expected signals are:

| Command after `python3 -B $L` | Expected and observed signal |
| --- | --- |
| `cpu-spec v2` | 13 tests, OK |
| `analysis-spec v4` | 8 tests, OK |
| `spec v1` | 5 examples, zero failures/errors/pending |
| `build v1` | Successful release binary |
| `oracle r4-v2 --rows 4` | Independent M4 fixture, exact ggml wire control |
| `oracle r4480-v2 --rows 4480` | Independent M4480 fixture, exact ggml wire control |
| `probe m4-v1 --build-tag v1 --fixture-tag r4-v2` | Exact bytes, parity, corruption detection, 12 paired rounds |
| `probe m4480-a-v1 --build-tag v1 --fixture-tag r4480-v2` | Same complete checks at M4480 |
| `probe m4480-b-v1 --build-tag v1 --fixture-tag r4480-v2 --reverse-first` | Same checks with opposite initial order |

Every command returns zero with an empty `changed_files` list. Tags are
exclusive: reproduction needs fresh tags and matching build/fixture flags.
The strict read-only receipt audit was also executed:

```sh
python3 -B examples/support/wan_q4k_range_results.py \
  /opt/mldata_ssd/cogni-ml/video-scene-20261019/evidence/wan-q4k-range-20261023 \
  m4-v1 m4480-a-v1 m4480-b-v1
```

Evidence directory is `evidence/wan-q4k-range-20261023` under that external
base. Completion receipt SHA256 values:

| Receipt | SHA256 |
| --- | --- |
| `cpu-spec-v2` | `717f99755a4b174c0765ac92c9c797e030ed4a549cc4d3278d53ca421be06fdc` |
| `analysis-spec-v4` | `89259db198837c44321ade028672ba8d1401a4a2c855b69915f295f1ab56c632` |
| `spec-v1` | `07beb34d14dcbf24d276ebf2c6a7fe6b630c7bca104ef0c859264486296bf5c8` |
| `build-v1` | `4fe03aaacbaeeab1e93ae5b76192817b46439653e9d21ac29926fba1810f7444` |
| `oracle-r4-v2` | `ce6ec836af60a1f2013b567a1d67fc2fca9be9bf2176a7db32ad91caa5c96b8b` |
| `oracle-r4480-v2` | `b965ca12d13b0ad5f71c61792ca89b834f576126e1c16b28b8a507c8994108eb` |
| `probe-m4-v1` | `afbeeb62327b6f08d774ad0217eab1dfce6711b7553d1ae7a8dd38a3d472bf11` |
| `probe-m4480-a-v1` | `6dc77eb455f48484232d5cd32bd49e51d24049e309c97b983cee2080a61a7f3b` |
| `probe-m4480-b-v1` | `d4b76fd90a18947976604089af8a558c8867f414c5f3a551098bc4a57634cd37` |

Final `summary-v1.json` SHA256 is
`e218bd65b9c8a558187c918b6a83eefef3b8a080f1dd116799a17c77d2b2f2d5`.
Native binary SHA256 is
`09bf1cdd9807f36ca86909f44977ce500d3856bb7f86c7008aa7e8a3979cd88b`.
M4/M4480 manifest SHA256 values are respectively
`656d50f73efc0d7810ef5b877bff5299a3e47034a387e0dc96eaf093dffc8ee4`
and `c5e024d6e625cefa9f2dbb2d1597432a837b2e33857912f35ccc5a41e7680e3c`.
The source fixture manifest is
`0dba2df508f72764837c23d8c5743a6d687e2d1b6104406e38325031084c3703`;
its weight is `cd1037a3eb9c98cf5baed570e6825e7bb2463165563707cf7f1c3b367afca557`.

The canonical payload is inherited from pinned ggml revision
`7e4c0a96880dae4fc4268ad441f8a6446bd5460a`; its full decoder control is
rechecked here, while the ggml quantizer is not rebuilt. The release build
reuses six native objects pinned by prior `wan-qbit-mps-stage-20261022-v2`
`build-v2-intent.json` SHA256
`4cf673fd89d04112bc72de3e26e98d85111876f8f9b698042df2341aa2f6bca4`.
Only the T5 linear bridge has complete inherited source/object build ancestry;
the other five objects have byte pins and current source pins but incomplete
build ancestry. This experiment qualifies the observed binary and bytes,
not a fresh native-object toolchain rebuild. The launchers bind source hashes
and archive build sources; the analyzer/spec are independently pinned by the
final `analysis-spec-v4` receipt, outside native compiler inputs.

### Decision and continuation

Mini-Quad verdict is ROBUST for this bounded range-count comparison,
same-format execution, package accounting and observed guards. Broader
video-quality and small latency/equivalence claims remain VULNERABLE and are
not admitted. Keep canonical ggml Q4_K as the accuracy control; do not promote
the simpler min/max fit or change model defaults. Local ranges recover much
of the matched one-range error without consuming extra package bytes or
changing the execution graph. They do not accelerate the common graph.

The next quality discriminator is to repeat this fixed-byte comparison on
captured, disjoint Wan activations and additional projection types before
choosing a model-wide packing policy. For execution, the separate open
question remains a competitive fused decode/GEMM that removes the shared
F32 stage; this ablation does not establish such a kernel. No full model,
block, Studio or video generation was launched. Rollback removes the opt-in
files without rewriting weights. Refresh after relevant source, payload,
input/geometry, compiler/native-object, OS/SDK/device or numerical-policy
changes before consuming these results.
