# PixelUMM bounded decoder-block frontier, 2026-10-27

Current frontier: eight consecutive pretrained decoder blocks, zero through
seven, with shared GPU scratch and an authenticated BF16 cache. Small J2048
matches all 208 stages; full T6/H44/W80, J21,120 matches all 16 attention/output
boundaries, including different-input replay after poisoning mutable scratch.
Every compared word is finite and bitwise exact at unchanged
`atol=rtol=2e-5`. The eight-block resident median is 13.597055 seconds on this
M2 Max (three samples, 12.660567-14.520376 seconds). These are bounded block
measurements, not full-decoder or video latency. See the eight-block section
below for evidence, memory and limits. The oracle adapts upstream semantics
on PyTorch MPS; compiled upstream CUDA Flex equivalence is unqualified.

## Historical one-block qualification, 2026-10-26

The following one-block measurements and weight inventory describe the
earlier scope. The two-, four- and eight-block sections extend it explicitly.

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

## Four-block qualification, 2026-10-27

Status: VERIFIED within the saved four-block fixture scope. Actual decoder
layers zero through three run consecutively on GPU at T6/H44/W80, J21,120,
with unchanged finite `atol=rtol=2e-5` and a cache-aware 10% memory floor.
The existing native Workspace already supports this slot count; no native
GPU math, kernel, bridge or command-chain implementation changed here.

The probe and oracle now admit exactly two or four blocks. They authenticate
all 52 four-block weights, per-owner captures and immediate predecessor
edges. Each block receives the preceding block's actual GPU output buffer.
Four retained output banks and separate diagnostic captures survive reuse of
the common transient core/RMS/attention scratch. The loader allocates distinct
weight/input/table/output buffers and checks handle uniqueness. This evidence
does not cover arbitrary caller-supplied Workspace aliases or recovery after
draining an intentionally incomplete prefix; the runner submits all four slots.

The strongest anticipated failure was accepting four metadata owners while
executing only two blocks, or routing later inputs/captures to block zero or
one. Five binary preflight negatives reject incorrect block-two and block-three
predecessors, a missing fourth-block weight, an altered fourth-block receipt
owner and an aggregate budget one byte below admission. They reject before
payload authentication and GPU allocation. CPU tests exercise actual fixture
construction and exact four-slot budget accounting. Full-J alternate-input
output hashes differ for every block, then all eight replay boundaries match
after NaN poisoning of mutable shared scratch. Immutable tables, weights,
retained outputs and diagnostic capture storage are excluded from poisoning.

### Observed numerical, timing and memory results

Small T2/H32/W32 compares all 104 stages on the first and last requests.
Full J compares attention and output for all four blocks on the first request,
last timed request and poisoned different-input replay. Every compared tensor
is finite and **bitwise exact**: zero different words, zero failures and zero
maximum absolute error. Inputs are controlled synthetic BF16 hidden states
(seeds 20261026/20261027), not complete-decoder hidden states.

The full resident chain's timed samples after one warmup are
**6.488500125, 6.393219958 and 6.420842958 seconds**; median
**6.420842958 seconds**. Median GPU elapsed time is 6.417016500 seconds.
Preparation takes 5.855484250 seconds including streamed payload authentication
and upload, excluding metadata parsing. This is not a controlled loading-time
comparison, a kernel speedup or a video-throughput measurement.

All six full requests use 485 command segments, exactly **one terminal wait**,
zero named MetalBuffer allocations during encode and zero preterminal
activation readbacks. The first request also rejects wrong slot order and
in-flight release/read/reuse. Opaque MPS allocations are outside named counters;
one chain is not one command buffer. Diagnostics read captures only after the
terminal wait, and scratch poisoning/input replacement occur between requests.

Unique named storage is **5,419,082,724 bytes (5.05 GiB)**: 3,680,869,348 owned
and 1,738,213,376 borrowed. Four independent workspaces with equivalent
attention captures would own 10,571,116,432 bytes. Sharing saves
**6,890,247,084 bytes (6.42 GiB)** of owned storage by accounting, without a
separate performance comparison. Full native sampled process-group footprint
reaches **7,119,492,232 bytes (6.63 GiB)**; sampled cache-aware headroom stays
at least **24,283,217,920 bytes (22.62 GiB)** and swap does not grow. The guard
retains a 10% floor, 8 GiB footprint cap, 900-second timeout and 64 MiB
swap-growth stop. Oracle runs retain the same floor with an 11 GiB cap;
their maximum sampled footprint is 10,177,808,712 bytes. File-backed memory
is a reclaimability estimate, and polling is neither a peak bound nor a
memory reservation. GPU jobs run sequentially.

### Four-block BF16 cache and reproducibility

Only source layers zero through three are downloaded. New layers two and
three were fetched as pinned HTTP ranges, not as a full checkpoint. The new
cache contains **52 BF16 tensors, 1,543,571,456 bytes (1.44 GiB)**, half the
four F32 payloads. Every raw uint16 word matches upstream's own MPS BF16 cast
and independently exported fixture weights. Native preparation uploads these
authenticated BF16 bytes directly, without runtime F32 staging or conversion.
The full checkpoint hash remains unverified, and a full-decoder BF16 cache
has not been built. No cache-controlled startup speedup is claimed.

The curated index is
`.landmark/evidence/2026-10-27-hy-nomas-pixelumm-four-block.json`. Exact executed
argv, source/binary pins, range receipts, fixture/cache bytes, reports and
memory-guard logs remain under
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026`. Use fresh output/report/log
names when repeating a run; a rejected preflight may leave an empty report.

Observed CPU/build DoD commands:

```sh
python3 -B -m unittest spec/hy_nomas_pixelumm_checkpoint_test.py spec/hy_nomas_pixelumm_reference_test.py spec/hy_nomas_pixelumm_bf16_cache_test.py
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/build pixelumm_decoder_block_probe pixelumm_decoder_two_block_probe pixelumm_decoder_four_block_probe spec_pixelumm_decoder_block spec_pixelumm_two_block_probe
```

The final Python suites pass **34 tests**; native CPU suites pass **11 admission
and 11 fixture/cache examples**. The final validator authenticates both old
26-tensor and new 52-tensor caches. Guarded GPU executions of the saved small,
full, poisoned-replay, old pair and standalone argv all pass; the old pair and
standalone compare all 52 and 26 stages respectively. Those executions, not
the CPU/build commands alone, establish numerical and memory scope.
Correlated Mini-Quad final source/oracle/adversary verdict is ROBUST for this
bounded contract. Final review maps all 232 sampled four-block comparison
digests to their correct base/alternate fixtures and checks report, guard and
receipt bindings; it is not independent replication or hermetic attestation.
Pre-execution sidecars pin the executed binary and listed source files; the
four-block entry point and Makefile have closure hashes only. Build logs retain
the recipe, but do not retroactively attest those two files at compilation.

Risk remains CAUTION for future changes to sequential lifetimes, source ranges
or inventories. Rollback: omit the four-block entry point and retain the old
pair/standalone probes. Refresh after source/checkpoint/fixture, toolchain,
device/OS/backend, geometry, precision, ownership/lifetime or guard changes,
or lost external artifacts. The next gate is a memory-admitted larger decoder
prefix and real decoder inputs before complete decoding. No full-generation
quality or latency claim follows from these four blocks.

## Eight-block qualification, 2026-10-27

Observed gates pass for actual decoder layers zero through seven. The change
extends only the bounded probe, cache/oracle orchestration and validation;
native GPU math, kernel, bridge, command chain and Workspace API are unchanged.
The four-block entry point remains the rollback path. This is CAUTION work:
wrong later-owner edges, overwritten scratch or incomplete diagnostics could
otherwise produce a misleading pass.

The eight-slot loader authenticates all 104 weights, source receipts and
immediate predecessor edges before large payload authentication or GPU upload.
Each output bank is the next block's actual input GPU buffer. Common
core/RMS/local-attention scratch is shared; all eight output banks and per-owner
attention captures remain distinct. The runner submits the entire prefix on
one chain. Arbitrary caller-supplied aliases and partial-prefix cancellation
recovery are outside this qualification.

### Numerical and lifetime gates

| Gate | Geometry | Captures per checked request | Result |
| --- | --- | ---: | --- |
| Small, first/last requests | T2/H32/W32, J2048 | 208, all 26 stages per block | Finite, bitwise exact |
| Full, first/last timed requests | T6/H44/W80, J21,120 | 16 attention/output boundaries | Finite, bitwise exact |
| Full, different-input poisoned replay | T6/H44/W80, J21,120 | 16 attention/output boundaries | Finite, bitwise exact |

All 464 compared tensors have zero different words, zero tolerance failures
and zero maximum absolute error at unchanged `atol=rtol=2e-5`. Seeds
20261026/20261027 create different synthetic BF16 initial hidden states; every
replay block's expected output hash differs from baseline. Native replay is
compared against its matching alternate fixture. RoPE tables and all weights
are identical. This is an adapted upstream-semantics MPS oracle, not compiled
CUDA Flex, actual decoder latent inputs or image/video-quality evidence.

The first full request also rejects last-slot-before-first-slot execution and
in-flight release, read and reuse. All six full requests have 969 command
segments, **one terminal wait**, zero named MetalBuffer allocations during
encode and zero preterminal activation readbacks. Intermediate states remain
on GPU. Postterminal diagnostics read captures; input replacement and NaN
poisoning run between completed requests. No new inference synchronization was
introduced. Startup cache conversion remains a separate operation with its
existing synchronization boundaries. Opaque MPS resources are outside named
allocation counters; one chain is not one command buffer.

Mini-Quad review exposed a weak capture gate: an incomplete small fixture
could pass by comparing only the retained tensors. Eight-block fixtures now
require exactly 208 small stages or 16 full boundaries, with contiguous record
coverage and no trailing bytes. Only this fixture schema receives an 8 GiB
payload bound; pair/four fixtures and weight caches retain 4 GiB. Small/full
payloads are 6,841,110,528/6,050,025,472 bytes, external to Git.

Eight compiled-CLI negatives reject wrong block-six/seven predecessors, a
missing block-seven weight, altered cache owner, a small-fixture aggregate
budget one byte below admission, an omitted stage, wrong schema and malformed
record ranges. They use valid metadata hashes with deliberately invalid global
payload hashes: the expected validation error occurs before payload-hash
failure, no Metal-init marker appears and the fresh report stays empty.
Failed-preflight allocation/read counters are not exported, so these are
ordering checks rather than measured zero-allocation claims. Separate CPU
checks cover exact full-grid accounting, full-grid budget minus one byte,
missing owner metadata, wrong capture inventory and file bounds.

### Timing and memory

After one warmup the resident eight-block samples are **12.660567374,
13.597054874 and 14.520376250 seconds**; median **13.597054874 seconds**.
Median GPU elapsed time is 13.594401834 seconds. The increasing three-sample
series is retained as observed variability; no stable speedup is established.
Preparation takes 11.320959917 seconds, including streamed authentication,
upload and graph preparation, excluding metadata parsing. A cache-controlled
loading comparison and complete video latency remain unmeasured.

| Full eight-block inventory | Bytes |
| --- | ---: |
| Owned shared workspace with captures | 5,064,989,668 |
| Borrowed input, RoPE tables and BF16 weights | 3,281,784,832 |
| Unique named total | 8,346,774,500 (7.77 GiB) |
| Maximum sampled process-group footprint | 10,129,972,472 (9.43 GiB) |
| Minimum sampled cache-aware headroom | 21,021,835,264 (19.58 GiB) |

Eight independent owned workspaces with equivalent captures would use
21,142,232,864 bytes. Sharing saves **16,077,243,196 bytes (14.97 GiB)** by
accounting; no timed independent-workspace comparison was run. The 8 GiB
named-buffer ceiling has 243,160,092 bytes left and is distinct from the
system reserve. Full native uses an 11 GiB process cap, full upstream MPS
uses 17 GiB, both with a **10% physical-memory floor**, 900-second deadline
and 64 MiB swap-growth stop. Native swap growth is zero. File-backed cache is
included once in headroom; reclaimability and sampled footprints are not
reservations or between-sample peak guarantees. GPU jobs run sequentially.

### BF16 cache, reproduction and remaining boundary

Only decoder layers zero through seven are downloaded as pinned authenticated
HTTP ranges; this is not the complete 31,553,868,864-byte checkpoint. The new
cache holds **104 BF16 tensors, 3,087,142,912 bytes (2.875 GiB)**. All raw
uint16 words match upstream's own MPS BF16 conversion and all three separately
exported fixtures. Native load uploads these BF16 bytes directly, without an
intermediate F32 weight copy or runtime conversion. The complete checkpoint
hash and complete-decoder cache remain unverified/unbuilt.

Observed CPU/build DoD commands:

```sh
python3 -B -m unittest spec/hy_nomas_pixelumm_checkpoint_test.py spec/hy_nomas_pixelumm_reference_test.py spec/hy_nomas_pixelumm_bf16_cache_test.py
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/build pixelumm_decoder_eight_block_probe pixelumm_decoder_four_block_probe pixelumm_decoder_two_block_probe spec_pixelumm_decoder_block spec_pixelumm_two_block_probe
```

The suites pass **37 Python tests, 12 admission examples and 15 fixture/cache
examples**. Guarded small/full/poisoned replay gates pass. The rebuilt old
four- and two-block probes also match their saved 104-/52-stage fixtures, and
the final validator authenticates all three 26-/52-/104-tensor caches. These
executed GPU gates, not the CPU/build commands alone, establish numerical scope.

The curated index is
`.landmark/evidence/2026-10-27-hy-nomas-pixelumm-eight-block.json`. Exact argv,
pre-execution source/binary pins, build-object/toolchain pins, range receipts,
payload hashes, reports and memory guards live under
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026`. All three entry points
and the Makefile were pinned before the build. Source/binary receipts are not
hermetic build attestation. Final correlated Mini-Quad native/oracle/adversary
review is ROBUST within this eight-block fixture scope. The external evidence
index run checked 134 payload/source files and exited zero; its argv, output and
guard receipts are linked separately in the packet, without a recursive packet
self-hash. This review does not establish independent replication.

Refresh after source, weight, fixture, backend/device/toolchain/OS, geometry,
precision, ownership/lifetime or guard changes, or lost external artifacts.
Next gate: a fresh memory-admitted larger prefix and real decoder inputs before
complete decoding. No complete decoder, diffusion, Studio or generation ran.
