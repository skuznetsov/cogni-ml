# PixelUMM bounded decoder-block frontier, 2026-10-26

Status: VERIFIED within the one-block fixture scope below. A native resident
BF16 block passes the unchanged finite `atol=rtol=2e-5` gate and takes a median
1.569623 seconds at T6/H44/W80 on this M2 Max. No complete decoder, diffusion
or video generation ran. The oracle is an adapted upstream-semantics PyTorch
MPS implementation, not execution of the compiled upstream CUDA Flex kernel.

## Source and weights

The upstream checkout is clean at `911cea6458a89c2663513ff7099561ec190dd491`.
Its `hymm/models/vae/pixelumm_decoder/model.py` SHA-256 is
`bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec`.
The inventory is pinned to Hugging Face revision
`c67f2c52b3fbe0193cbc202769bbe5cb7264807b`, repository `tencent/Hy-Nomas`, path
`VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors`.

The checkpoint is 31,553,868,864 bytes. Its advertised LFS SHA-256 is
`0bbfe8fd44bd2ebf1059fe65079caba50fd6f52ea8e1b5b35393e3050b9b8378`;
the complete file was not downloaded and its full-file hash is unverified.
The observed 75,192-byte header SHA-256 is
`7e7a92cfe84d54c0da3a132099a9472766ae0034042515a0f76b6198bbd09d11`.
The extractor checks the pinned header, dtype, shapes, extents and exact
ownership before fetching bounded HTTP 206 ranges; its receipt pins chunk
hashes. All 645 stored tensors are F32.

| Scope | Tensors | F32 payload bytes | BF16 runtime bytes |
| --- | ---: | ---: | ---: |
| Decoder, including global projections | 476 | 27,887,173,632 | 13,943,586,816 |
| Encoder, outside this decode task | 169 | 3,666,620,032 | Not loaded |
| Decoder layer zero | 13 | 771,785,728 | 385,892,864 |

Only layer zero's weights were downloaded. Its deterministic F32 subset file
is 771,787,128 bytes, SHA-256
`ae82bb41c121a955493f5b1502dc512ac8aaba05a6cec67e98edeaedfe720485`.
The complete decoder needs 12.99 GiB of BF16 weights; one layer uses 368.016
MiB. Header compatibility alone is not numerical qualification.

## Geometry and memory admission

The architecture has 36 layers, D4096, F12288, 32 query heads, eight KV heads,
head dimension 128 and RMS epsilon 1e-6. Q/K have independent 64-element
T/HW RMS gains. RoPE splits 64/32/32 across T/H/W, theta 1e6/1e4/1e4.
The preset uses BF16, one decoder step, 128-pixel local tiles and a temporal
window of ten latents with overlap five.

The saved DiT latent `[1,32,6,22,40]` implies decoder T6/H44/W80, J21,120.
Eight-token spatial tiles are ordered by tile H/W, time, within-tile H/W.
Border neighborhoods shift inward to three by three spatial tiles; every time
position remains visible. The final height tile is four tokens. The 60 tiles
use actual Q/K extents 384/3456, 384/2880 or 192/2880. Permutation SHA-256
(little-endian I32):
`4e81c6026e99ee2c17c176c4394682ca632602c198c3065ad311a83a3839cf33`.

A global H32/J21,120 F32 score tensor would be 57,094,963,200 bytes (53.17 GiB).
It was never allocated. Physically local K/V gathers, GQA head `query_head/4`
and exact unpadded extents bound one score panel to 169,869,312 bytes (162 MiB).
Six reusable F32 attention panels total 465,567,744 bytes. The global-attention
fallback is rejected before allocation when J>2048.

| Full native block inventory | Bytes |
| --- | ---: |
| Owned workspace without captures | 2,469,764,068 |
| Owned workspace including attention capture | 2,642,779,108 |
| Borrowed weights/input/RoPE tables | 580,534,784 |
| Total named buffers in measured run | 3,223,313,892 |
| Maximum sampled process-group footprint | 4,808,345,760 |
| Minimum sampled cache-aware headroom | 28,193,882,112 |

All GPU runs use the cache-aware guard with 10% of 64 GiB physical memory as
reserve, an explicit process cap, a 64 MiB system swap-growth stop and a
deadline. File-backed cache includes speculative pages and is counted once;
reclaimability estimates and sampling do not bound between-poll spikes.
The native full run used an 8 GiB/900 second cap and zero swap growth. Its
3.00 GiB named buffers and 4.48 GiB sampled footprint include diagnostic
storage. The separately run full reference peaked at 6,966,036,544 sampled
bytes under 12 GiB/900 seconds, also with zero swap growth.

Prior measured Qwen S1280 and compact DiT named inventories plus all decoder
BF16 weights total 36,685,117,252 bytes (34.17 GiB), before PixelUMM workspace,
pixels or opaque allocations. Adding only the reserve needs 43,557,064,926
bytes of available headroom. Their simultaneous residency is not admitted by
the observed snapshot. Phase-separated complete decoding remains unqualified.

## Numerical findings and selected implementation

Risk: CAUTION, a new prepared inference path. The diagnostic module is isolated
from existing DiT, Qwen and production routing. Rollback is to omit the probe.

The first RMS failure had identical statistics and BF16 pre-gain values but a
different gain result. Materializing BF16 pre-gain before a separate gain graph
makes normalization and Q/K/V projections bitwise exact. Split Q/K statistics
also require the upstream MPS reduction geometry. Separate legacy
`reverseSquareRootWithTensor` graphs match the selector present in the installed
PyTorch 2.6 wheel. Safe Metal math alone left failing hashes unchanged; precise
sqrt/division did not improve the gate and was rejected. These are scoped
backend compatibility findings, not portable inverse-root identities.

Saved identical Q/K/V replay is bitwise exact with canonical 32-head geometry;
eager four-head grouping has 74 tolerance violations. The native fused
four-head path has 525 violations and is not qualified. The selected path
preserves 32 heads and materializes F32 QK, scale, softmax and PV handoffs.
QK/PV use rank-three batched matmul; softmax retains rank four/axis three,
matching the pinned PyTorch MPS decomposition. Metal SiLU requires no change.

Graphs, weights and all scratch remain resident through the existing terminal
completion boundary. Immutable index tables cannot double as RMS statistics;
statistics use dedicated disjoint buffers. Only mutable storage with disjoint
live ranges is reused. In-flight session reuse and release are rejected.

## Observed qualification and timing

Inputs are controlled standard-normal BF16 hidden states, seed 20261026, with
actual pretrained layer-zero BF16 weights. They are not full-decoder hidden
states. A reduced-width CPU layer also matches the pinned upstream Flex
implementation and dense mask at the unchanged tolerance.

| Native report | Geometry | Compared boundaries | Result |
| --- | --- | --- | --- |
| `native-small-v13.json` | T2/H32/W32, J2048 | All 26 stages, first/last requests | Bitwise exact, finite, zero tolerance violations |
| `native-full-v1.json` | T6/H44/W80, J21,120 | Attention and output, first/last requests | Bitwise exact, finite, zero tolerance violations |

Small v13 qualifies the final formatted source, with zero warmups and one timed
request; it is not a warmed benchmark. Older small v12 timings are historical.
Full v1 uses one gate request, one warmup and three resident timed requests:

| Measurement | Seconds |
| --- | ---: |
| Preparation, excluding fixture parsing/hashing | 0.393273167 |
| First gate request | 1.582957500 |
| Warmup | 1.592550792 |
| Timed totals | 1.568477208 / 1.569622542 / 1.570300250 |
| Median timed total | 1.569622542 |
| Median encode time | 0.331452417 |
| Median GPU elapsed time | 1.569149458 |

Encoding overlaps GPU execution; encode and GPU elapsed times must not be
added. Each full request has 122 command segments, exactly one terminal wait,
zero named Metal allocations during encode and zero preterminal activation
readbacks. Only first/last requests read two captures after completion. Opaque
MPS allocations are outside the named counter. PyTorch is outside the native
measured path. Fixture loading, table construction and terminal diagnostic
export are outside the GPU neural-math claim.

## Evidence, reproduction and refresh

The curated index is
`.landmark/evidence/2026-10-26-hy-nomas-pixelumm-block.json`. It pins source and
binary manifests, report bytes and guard receipts. Small v13 report SHA-256:
`5d9605826d55b363d3646683fbc8a6f8eae3539b471a24326afad5d1e99effd1`.
Full v1 report SHA-256:
`8a1bd99ef3a5ac7fda42dbc57ccd36cc8bad916c2b2f08eb95f224215b21dc85`.
Artifacts live under `/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/`.
Large subsets, fixtures, builds and logs are excluded from Git. Exact executed
argv is in `logs/*.argv.json`; fresh runs require unused report/log names.

Observed Definition of Done on the source state pinned in that index:

```sh
python3 -B -m unittest spec/hy_nomas_pixelumm_checkpoint_test.py spec/hy_nomas_pixelumm_reference_test.py
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/build pixelumm_decoder_block_probe spec_pixelumm_decoder_block
```

The Python suites pass 18 tests; the guarded build/spec passes seven CPU native
admission examples. Those cover partial tiles, inward border shifts, temporal
visibility, fallback, table coverage and pre-Metal budget rejection. The saved
small v13/full v1 guarded native argv actually executed with passing numerical,
one-wait and allocation/readback gates. All provenance digests were refreshed
against final formatted source. The guard is part of these DoD executions;
the two commands alone do not qualify memory admission or GPU numerics.

Correlated Mini-Quad source/oracle/adversary review is ROBUST within this scope,
not independent replication. Report/guard hashes close the mutable-result gap
found in final review; receipts are not hermetic build attestation. Full
36-layer decode, other inputs, compiled CUDA Flex parity, pixels, end-to-end
latency and quality remain unqualified.

Refresh after source/checkpoint/fixture, toolchain/device/OS/backend, geometry,
precision, neighborhood semantics, ownership/lifetime or guard changes, or loss
of external artifacts. Next useful gate: a memory-admitted phase-separated
decoder assembly with shared workspace and sequential block comparisons before
complete generation.

## Two sequential blocks and BF16 startup cache, 2026-10-27

Status: VERIFIED within the two-block fixture scope below. The preceding
single-block qualification remains historical evidence at its pinned source
state. This extension admits actual decoder layers zero and one at the same
small and full-J geometries. It does not admit complete decoding or generation.

The shared `Workspace` owns transient core and attention/RMS storage once,
plus a persistent output bank and requested diagnostic captures for each slot.
Block one's input is exactly block zero's GPU output, without a copy. Both
blocks submit to one command chain, with no intermediate wait. Aggregate
unique-byte admission precedes GPU allocation and large payload reads. A host
mutex covers validation, native submission and state publication; it adds no
GPU wait. All slots must be prepared before submission. Wrong slot order,
cross-chain submission, premature reads/reuse/release and incomplete failed
requests are fenced until the existing terminal completion boundary.

This is CAUTION work because cross-plan buffer lifetimes change. The strongest
failure is overwriting block zero's retained output or consuming stale scratch.
The full-J replay changes the initial input and poisons mutable shared storage
with NaNs after the previous terminal wait. It excludes weights, index tables,
the exponent, output banks and captures. Both blocks still match the alternate
upstream result. Poisoning and the diagnostic input upload are outside encode;
the replay adds no synchronization inside either block. The standalone API is
preserved and its original small fixture passes on the rebuilt binary. Rollback
is to omit the pair probe and use the standalone plan.

| Native report | Geometry | Compared boundaries | Result |
| --- | --- | --- | --- |
| `native-pair-small-final-v2.json` | T2/H32/W32, J2048 | All 26 stages of both blocks, first/last requests | All 52 stages bitwise exact and finite |
| `native-pair-full-v1.json` | T6/H44/W80, J21,120 | Attention/output of both blocks, first/last and alternate poisoned replay | All four boundaries bitwise exact and finite |
| `native-single-regression-v1.json` | T2/H32/W32, J2048 | Original standalone block, all 26 stages, first/last | Bitwise exact and finite |

All comparisons retain `atol=rtol=2e-5`, with zero violations. Inputs are
synthetic BF16 hidden states, seeds 20261026 and 20261027; block one receives
block zero's computed output in both native and reference paths. The reference
uses the pinned upstream layer semantics adapted to PyTorch MPS, not compiled
CUDA Flex. All 26 BF16 weight tensors also match the independently produced
pair fixtures by exact bytes. Complete decoder hidden states and pixel quality
remain unqualified.

The full pair has one gate, one warmup, three timed requests and one untimed
different-input poisoned replay. Timed totals are 3.169756624 / 3.174909918 /
3.167769917 seconds; median **3.169756624 seconds**. Median GPU elapsed is
3.167306417 seconds and median encode time is 1.926368708 seconds. Encoding
overlaps GPU work; those times must not be added. Preparation is 3.112975
seconds including streamed payload authentication and upload, excluding
metadata parsing. This is not a controlled loading-time comparison or video
throughput measurement.

Every full request uses 243 command segments, exactly one terminal wait,
zero named Metal allocations during encode and zero preterminal activation
readbacks. Four captures are read only after completion on the three compared
requests. Named storage totals **3,955,236,836 bytes**: 2,988,809,188 owned and
966,427,648 borrowed. Two separate workspaces with equivalent captures own
5,285,558,216 bytes; sharing saves **2,296,749,028 bytes (2.14 GiB)** of owned
storage. This is allocation accounting, not a measured speedup. Opaque MPS
allocations are outside the named counters.

The full native guard retains the 10% cache-aware floor, 8 GiB footprint cap,
900-second timeout and 64 MiB swap-growth stop. Sampled group footprint reaches
5,443,013,176 bytes; sampled headroom stays at least 24,304,386,048 bytes and
swap does not grow. File-backed memory is a reclaimability estimate;
sampling is not a peak bound or a reservation. Reference generation uses the
same floor with a separate 9.5 GiB footprint cap. Only one GPU job runs at a
time.

### BF16 cache scope and integrity

Only source blocks zero and one have been downloaded and converted. The two
F32 weight payloads total 1,543,571,456 bytes; the BF16 cache payload is
**771,785,728 bytes (0.719 GiB)**. Its 26 tensors match upstream's own
`Tensor.to(torch.bfloat16)` conversion on MPS as raw uint16 words. Conversion
is a one-time startup operation. Native preparation reads the resulting BF16
bytes directly into GPU buffers, without an intermediate F32 weight copy or
runtime cast.

The cache manifest binds the pinned HF revision/path/header and upstream model
source, each selected subset and HTTP-range receipt, all tensor shapes/extents
and SHA-256 digests, and the conversion method. The final validator checks
actual receipt/subset chunks and cache bytes. The complete checkpoint's
advertised LFS digest remains unverified. The complete decoder's BF16 weight
payload would be 13,943,586,816 bytes (12.99 GiB); that cache has not been built.
Half-sized payload and removal of the conversion are established; a loading
speedup needs paired startup measurements with controlled cache state.

### Reproduction and next boundary

The new curated index is
`.landmark/evidence/2026-10-27-hy-nomas-pixelumm-two-block.json`. It pins the
final sources, binaries, fixture/cache manifests and payloads, reports and
guard receipts under the same external artifact root. Exact executed arguments
are in `logs/pair-*.argv.json`; use fresh report/log names for a new execution.
The failed `pair-native-build-final-v2` attempted a canonical path containing
spaces, which this Makefile does not support; v3 uses the external SSD's
`/opt/mldata_ssd` alias and passes. No numerical failure occurred in that build.

Observed DoD on the source state pinned in the new index:

```sh
python3 -B -m unittest spec/hy_nomas_pixelumm_checkpoint_test.py spec/hy_nomas_pixelumm_reference_test.py spec/hy_nomas_pixelumm_bf16_cache_test.py
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/build pixelumm_decoder_block_probe pixelumm_decoder_two_block_probe spec_pixelumm_decoder_block spec_pixelumm_two_block_probe
```

The Python suites pass 30 tests; native CPU suites pass 10 admission and seven
fixture/cache examples. The final cache validator and the saved guarded
small/full/standalone GPU argv all execute successfully. Those guarded GPU
executions, not the commands above alone, establish numerical and memory scope.
Correlated Mini-Quad source/oracle/adversary review is ROBUST within this
bounded contract. Final review checks report/argv/guard/source/binary hashes,
fixture bindings, the poisoned replay and lifetime guards. It is not independent
replication or a hermetic build attestation.

Refresh after source/checkpoint/fixture, toolchain/device/OS/backend, geometry,
precision, ownership/lifetime or guard changes, or lost external artifacts.
The next boundary is a memory-admitted larger sequential decoder prefix with
shared workspace and real decoder inputs, before complete decoding. No claim
about full generation quality or latency follows from this pair.
