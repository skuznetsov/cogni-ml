# PixelUMM bounded decoder-block frontier, 2026-10-27

Current frontier: eight consecutive pretrained decoder blocks, zero through
seven, with shared GPU scratch, an authenticated BF16 cache and an opt-in F32
FFN graph mode qualified across two/four/eight blocks. Small J2048 matches all
208 stages of the eight-block prefix; full T6/H44/W80, J21,120 matches all 16
attention/output boundaries, including different-input replay after poisoning
mutable scratch. Every compared word is finite and bitwise exact at unchanged
`atol=rtol=2e-5`. New paired eight-block medians are 12.580/12.931 seconds for
BF16 graph feeds versus 10.067/10.451 seconds for F32 FFN graph operands on
this M2 Max, with higher sampled memory. The default remains BF16. See the
sequential F32 qualification below for timing drift, memory and limits. These
are bounded block measurements, not complete-decoder or video latency. The
saved oracle adapts upstream semantics on PyTorch MPS; compiled upstream CUDA
Flex equivalence is unqualified.

## F32 FFN sequential qualification contract, 2026-10-27

The earlier single-block speed result alone did not admit sequential F32 FFN
plans. This slice forwards one default-off mode to every owner of the existing
shared workspace; it changes no arithmetic kernel, handoff, encode or wait site.
Two, then four and eight blocks were admitted after each preceding memory and
numerical gate passed. Check every available saved upstream stage at J2048 and every
attention/output boundary at full J21120. Different-input replay must overwrite
NaN-poisoned mutable scratch and match each owner's alternate output.

This is CAUTION work: opaque MPSGraph temporaries can multiply across retained
plans even when named storage is unchanged. Process-tree caps and the cache-aware
10% physical floor, 64 MiB swap-growth stop and deadlines guard each serial job.
Use separate BF16/F32 processes in both orders for full-J timings. A regression,
missing comparison, mixed slot mode, early activation read, named encode
allocation, extra terminal wait or guard stop rejects promotion. Rollback omits
`--ffn-f32`; BF16 remains the default. Qualification commands are the guarded
release two/four/eight-block probes with authenticated existing fixtures/cache,
plus `make spec_pixelumm_decoder_block spec_pixelumm_two_block_probe` using the
external build directory. DoD requires finite results at unchanged
`atol=rtol=2e-5`, all ownership/lifetime negatives, one terminal wait and the
declared process budget. No full decoder or generation is authorized here.

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

## Full-J phase profiling qualification, 2026-10-27

One resident pretrained block zero is profiled at T6/H44/W80 (J21,120).
The strongest next optimization candidate is **FFN matrix work**. Resident
unprofiled encode-plus-terminal-wait medians are approximately **1.59-1.68
seconds per block**, including GPU diagnostic capture copies but excluding
preparation and postterminal comparisons. This is not complete-decoder latency
or seconds of video. This profiling-only slice promotes no arithmetic change;
the separately qualified opt-in FFN path is recorded below.

### FFN follow-up contract

The follow-up CAUTION slice isolates the pretrained block-zero gate/up/down
projections before changing the connected decoder. The baseline is the current
resident BF16 MPSGraph linear step. Candidates must consume the same saved
BF16 post-attention-normalized input (gate/up), or the same BF16 product input
(down), and the same authenticated weights. Initial qualification uses saved
upstream J2048 captures and full K4096/K12288; larger-M timing must identify
whether its rows are real full-J activations or GPU-repeated saved rows.

The falsifier is any nonfinite value or failure at the existing
`atol=rtol=2e-5`, including BF16 rounding boundaries. Alternative operand loads,
partial-sum grouping, casts and gate/up fusion may change those boundaries;
no speed result overrides a failed numerical gate. Timings include required
per-request conversions and packing. Resident alternative weight storage is
reported separately, including its preparation and memory cost. All neural
work stays on GPU, with comparisons only after the sole terminal wait.

DoD: build and run the bounded projection probe under the existing cache-aware
10% physical reserve, process-footprint cap, 64 MiB swap-growth stop and
deadline; qualify its comparator with a seeded defect; compare all three
projections against the saved upstream captures and the current MPS control.
A passing candidate then needs balanced resident pairs and a connected full-J
block comparison before promotion. The rollback is to retain the unchanged
MPS default and keep any rejected candidate confined to the diagnostic probe.
No complete decoder, diffusion, Studio or generation belongs to this slice.

The CAUTION contract preserves arithmetic, BF16 boundaries, graph geometry,
captures and existing command submission/wait sites. The old unprofiled FFI
entry point remains available; the Crystal API defaults to `profile: false`.
All full-J requests in this earlier profiling-only qualification retain 122
command roots and one terminal wait. The later F32 FFN path uses 128 roots.
Equal root counts do not establish identical off/on continuation placement.

Opt-in, preallocated CPU markers delimit input normalization, QKV, QK
normalization/RoPE, local attention, output projection, residual/post-norm,
gate/up, activation/product, down and final residual. Existing API collector
IDs retain their original meanings; the PixelUMM report explicitly maps the
used coarse buckets to preparation, local attention, output/residual/norm and
FFN. Graph scopes measure host wall/thread CPU, including nested continuation
calls. They are not GPU kernel timestamps. Root observations may register
tracking metadata but must not create, submit or wait for command roots.

The report preserves completed-root sum, union, envelope and overlap separately
from host API inclusive/nested/exclusive accounting. Phase boundary roots can
contain both phases; per-phase candidate sums cannot be added as exclusive
costs. The bounded Instruments attach gate waits on CPU before any request
encoding/timing; it adds no GPU wait. Profile snapshots reject before the
request's existing terminal wait.

### Timings and attribution

Three controls without Instruments use balanced in-process off/on pairs: one
warmup per arm and four alternating-order timed pairs. The first two also
record CPU continuation chronology; the third enables API profiling only.

| Control | Off median, s | On median, s | Individual paired deltas |
| --- | ---: | ---: | --- |
| Off-on initial order, continuation trace | 1.683734 | 1.686453 | -0.38% to +0.33% |
| On-off initial order, continuation trace | 1.589550 | 1.591180 | -0.15% to +0.42% |
| Off-on initial order, API only | 1.641722 | 1.637838 | -2.29% to +1.79% |

These controls measure enabled profiling against the patched disabled path.
The Graph-scope hook is also invoked when disabled; no pre-patch binary
comparison was run. Process drift and the wider API-only pair spread preclude
zero-overhead or speedup claims. Instruments timing is kept separate.

The Instruments capture contains four requests, 488 command-buffer submissions
and 780 Graph encoding intervals. XML references are resolved per table;
exact-PID Compute rows are joined to ordered submissions. The two profiled
requests provide 244 native root timestamps: maximum envelope-duration
difference and common-clock-offset spread are both about 42 ns, within the
unchanged 2 us telemetry alignment check. Off-arm root placement relies on
chronological grouping and counts, not native per-root validation.

An initial broad-channel join failed that alignment check on two Fragment
rows with conflicting inherited command-buffer identities. The corrected
analysis retains those rejected rows and restricts to Compute, consistent
with this native encoder having no rendering pipeline. It also excludes
17,771 foreign GPU rows and 1,549 foreign shader rows. This telemetry repair
does not change numerical tolerances.

| Root-local matrix timeline candidates | Gate request, s | Timed request, s |
| --- | ---: | ---: |
| QKV candidate roots 2-4 | 0.161088 | 0.161076 |
| Output-projection boundary root 115 | 0.105842 | 0.106195 |
| FFN candidate roots 118-120 | 0.946366 | 0.949459 |

These are unions of **sampled `NDArrayMatrixMultiply`-labelled shader timeline
intervals**, clipped to target Compute root windows. Shader rows have no
command-buffer IDs: attribution uses PID and temporal clipping after the
root join. The FFN candidates account for 59.9%/59.4% of the requests' target
Compute interval unions (1.579923/1.599647 s). This is non-exclusive sampled
interval attribution, not exclusive kernel time, cycles, occupancy, or a
predicted speedup. Root 118 also touches residual/post-norm; root 120 crosses
gate/up, activation and down. Exact subprojection attribution is unavailable.

Across eight traced timed arms, native attention-only roots 6-114 have
interval unions of **0.317-0.330 s**. Including boundary candidates 5-115
gives 0.434-0.453 s, overlapping the output-projection candidate. The FFN
aggregate, including activation and final residual, has definite roots
119-121 at 0.651-0.713 s and candidates 118-121 at 0.966-1.047 s. These are
command-root interval bounds, a different measurement from shader samples.

Each traced attention phase has 110 continuation calls. Their wall-time sums
are **0.319-0.340 s**, versus **0.00070-0.00109 s** of owner-thread CPU.
Union intersections with already-submitted roots cover **98.1-98.5%** of that
wall time. The longest recurring calls overlap earlier QKV work. This shows
little owner-thread CPU while prior GPU roots execute, consistent with queued
work/backpressure. It does not distinguish queue admission, driver waits,
scheduler delay or other off-thread causes. Wall minus thread CPU is not an
exact GPU-cost measurement.

### Numerical, allocation and memory gates

Six indexed reports cover **48 requests and 288 tensor comparisons**. Small
J2048 requests compare all 26 stages; full J compares both attention/output
boundaries on every request, including warmups and both profile arms. All
words are finite and bitwise exact against the saved adapted-upstream MPS
fixtures and between arms, at unchanged `atol=rtol=2e-5`. The oracle was not
rerun this turn; it is not compiled CUDA Flex or real decoder latent input.

Full J records the expected 60 attention tiles and **195 Graph calls**:
9 preparation, 180 local attention, 3 output/residual/norm and 3 FFN.
Every request has one terminal wait, zero named encode allocations and zero
preterminal activation reads. The first request's in-flight release guard and
every profiled request's prewait snapshot guard reject as expected. Opaque MPS
resources are outside named allocation counters. Postterminal capture reads
are diagnostics; no activation is read on CPU in the measured inference path.

One full-J block has 3,223,313,892 unique named bytes (3.00 GiB):
2,642,779,108 owned and 580,534,784 borrowed. Full standalone runs use an
8 GiB process cap; the Instruments process group uses 12 GiB. Across the six
native/report runs, maximum sampled group footprint is 7,135,402,368 bytes
(6.65 GiB, including the profiler) and minimum sampled cache-aware headroom
is 26,855,391,232 bytes (25.01 GiB). Swap growth is zero. Jobs remain serial,
with a **10% physical floor**, 64 MiB swap-growth stop and deadlines.
File cache is included once; polling is not a true peak guarantee or a memory
reservation. No complete decoder, Studio, diffusion or generation ran.

### Verification, reproduction and next falsifier

Executed final build/CPU command:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/build pixelumm_decoder_block_probe spec_pixelumm_block_profile spec_pixelumm_decoder_block spec_pixelumm_two_block_probe
```

Release build, 28 profile/API/trace examples, 12 admission examples and
15 fixture/cache examples pass. The final binary also passes the small
all-stage off/on GPU gate. Core native/profile/support source hashes match
all indexed full-J run receipts; CLI attach/report handling, Makefile and a
spec comment evolved afterward. Historical generic executables were
overwritten by later builds; their hashes remain pinned but their bytes are
not archived. The latest checked executable is preserved separately as
`build/pixelumm_decoder_block_probe-profile-v3`. These receipts are not
hermetic build attestation.

The captured Instruments executable hash exactly matches the later API-only
run's pre-execution receipt, whose core-source pins match this checkout.
The capture has no separate pre-capture source receipt: this cross-run byte
identity links the observation to the source-pinned run, without proving a
pre-capture source state or a hermetic source-to-binary build.

The curated index is
`.landmark/evidence/2026-10-27-hy-nomas-pixelumm-profile.json`. Exact argv,
pre-execution source/binary pins, guards, reports, analysis scripts, XML
exports and the retained Instruments trace live under
`/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026`. The index checks report
comparisons, fixture/report hashes, all sampled guard limits and live core
source pins; packet shape alone does not prove those runtime claims.
Correlated Mini-Quad review is ROBUST within the bounded profile and the
qualified FFN-candidate claim, not independent replication.

Next falsifier: isolate gate/up/down at their real dense BF16 geometry against
MPS, first preserving BF16 rounding/reduction and checking upstream boundaries.
Only promote an alternative or fusion after paired block timings improve.
Activation/residual fusion alone cannot address the measured large matrix
candidate. Roll back profiling by omitting `profile: true` / `--profile-order`;
this retains the patched disabled Graph hook. Refresh after source, fixture,
weight, device/OS/toolchain/backend, geometry, precision, capture, lifetime or
guard changes, or loss of external evidence.

## Earlier single-block F32 FFN graph qualification, 2026-10-27

The measured FFN lever is MPSGraph operand typing. For gate/up/down only,
`ffn_f32: true` casts the resident BF16 input and weight feeds to F32 graph
tensors before matmul, then retains the existing BF16 output cast. The
standalone probe exposes `--ffn-f32`; the Plan option is fixed at preparation.
Q/K/V/output projections, activation/residual stages and caller-owned BF16
storage are unchanged. Graph types do not certify backend physical arithmetic
or explain MPS's kernel selection. The default remains false; the old C create
symbol forwards to mode zero, and the new entry point rejects modes outside
0/1 before inspecting handles or allocating resources.

### Projection falsifiers

The diagnostic probe compares current BF16 MPSGraph, explicit F32 operands,
BF16-to-F16-to-F32 graph operands, and a native F16-operand K8 MMA candidate.
All consume the same pretrained weights and saved upstream BF16 activations,
with full K4096 for gate/up and K12288 for down. Four alternating-order timed
rounds follow one warmup. Per-request casts are included; upload, compilation,
poison, row repetition, comparisons and terminal readback are excluded.

| J2048 projection | BF16 root union, ms | F32 root union, ms | Native F16 tolerance failures |
| --- | ---: | ---: | ---: |
| gate | 31.104 | 18.756 | 149 |
| up | 32.351 | 19.490 | 108 |
| down | 31.721 | 19.521 | 288 |

The measured intervals in these three small-J reports do not overlap other
observed root lifetimes. These are command-root unions, not shader cycle
measurements. All three MPS arms match saved BF16 output words exactly; the
native F16 candidate fails the unchanged `atol=rtol=2e-5` and remains confined
to diagnostics. GPU comparison coverage, injected-NaN detection and an
independent postterminal CPU comparator are checked. F16 roundtrip counters
also detect a seeded underflow and overflow. Actual operand losses, including
tiny nonzero-to-zero values, are recorded; they do not alone establish the
cause of native reduction differences.

A larger-M projection report repeats the J2048 activation rows cyclically on
GPU to M21120. Its BF16/F32 root intervals overlap diagnostic work, so those
timings are retained but disqualified from an exclusive speed comparison.
This repeated-row probe is not the connected full-J block used below.

### Connected full-J result and memory tradeoff

Serial process order is BF16, F32, F32, BF16. Each process prepares one resident
block0 plan at T6/H44/W80, J21120, runs a numerical gate and one warmup, then
three timed requests. Separate processes avoid retaining two plans and their
opaque MPS workspaces at once; these are balanced serial pairs, not
same-process pairs.

| Pair order | BF16 request median, s | F32 request median, s | Latency reduction |
| --- | ---: | ---: | ---: |
| BF16 then F32 | 1.709523 | 1.207000 | 29.40% |
| F32 then BF16 | 1.592808 | 1.194168 | 25.03% |

Time includes native encoding, the existing terminal wait and GPU capture
copies. It excludes preparation/compilation/upload and postterminal CPU
comparisons. Retain the BF16 drift between pairs; the result is a 25-29%
prepared-request reduction in this bounded fixture, not exclusive FFN kernel
time, complete-decoder latency or seconds of video.

Across the four matrix reports, 16 attention/output comparisons from the first
and last requests are finite and bitwise exact against saved adapted
upstream-semantics PyTorch MPS captures. Other requests retain captures but
skip CPU comparison. The final rebuilt binary separately passes 52 comparisons
over all 26 J2048 stages and four full-J attention/output comparisons. Fixtures
use seeded synthetic inputs, not real decoder latents; no fresh upstream or
compiled CUDA Flex run is claimed.

Each connected request has one terminal wait, zero named encode allocations
and zero preterminal activation readbacks. Existing encode/submit/wait sites
are unchanged, but observed backend command segments increase from 122 to 128
at full J and 13 to 16 at small J. Execution topology is not identical.

Full-J named storage stays 3,223,313,892 bytes (3.00 GiB). The four paired runs
sample process-group footprints of approximately 4.80-4.90 GiB for BF16 and
6.44 GiB for F32. This cost is outside named-buffer accounting and is material
to later multi-block admission. All serial guards preserve the cache-aware
10% physical floor, 8 GiB full-block/4 GiB projection caps, 64 MiB swap-growth
stop and deadlines. Sampled swap growth is zero; polling does not guarantee
between-sample peaks or reserve opaque MPS memory.

### Verification, rollback and next gate

Executed release build and regression commands:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/build pixelumm_decoder_block_probe pixelumm_ffn_projection_probe
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/build spec_pixelumm_decoder_block spec_pixelumm_two_block_probe spec_pixelumm_block_profile spec_pixelumm_ffn_projection
```

All 62 examples pass: 15 admission/precision/legacy-ABI, 15 fixture/cache,
28 profile/API/trace and four independent CPU-comparator checks. GPU runs are
separately guarded and indexed in
`.landmark/evidence/2026-10-27-hy-nomas-pixelumm-ffn-f32.json`. The external
validator recomputes fixture payload hashes, compares saved stage hashes,
checks all sampled guards, source/binary receipts and balanced medians. The
repository has no `scripts/validate-landmark-graph.sh`; no structural-validator
success is implied. Same-lineage Mini-Quad review is correlated, not
independent replication. Heavy artifacts and preserved executables stay on SSD.

Rollback is to omit `ffn_f32: true` / `--ffn-f32`; rejected native/F16 paths
cannot be selected by a decoder Plan. Before default promotion, qualify this
mode on sequential blocks with shared scratch, all available upstream output
boundaries, and an admitted process-footprint budget. The earlier eight-block
BF16 qualification does not qualify the new mode. Refresh after source,
fixture/weights, device/OS/toolchain/backend, geometry, precision, ownership,
captures, guards or lost external artifacts. No full decoder, diffusion,
Studio or generation ran in this slice.

## Sequential F32 FFN qualification, 2026-10-27

The serial Session now forwards `ffn_f32: true` to every Plan and verifies the
selected mode for every slot. Two-, four- and eight-block probes accept
`--ffn-f32` and report the actual per-slot modes. The shared workspace, direct
GPU output-to-input handoffs, buffer ownership, arithmetic kernels and native
encode/submit/wait sites are unchanged. Both modes retain BF16 weights and
inter-block outputs; the option changes only gate/up/down graph operand types.
This is ordinary workspace reuse, with no LTP/WBA claim.

### Numerical and lifecycle gates

Nineteen serial guarded processes execute 105 connected prefix requests.
CPU comparisons after the terminal wait cover 54 requests: the numerical gate
and final timed request of each process, plus every full-J alternate-input
replay. Other requests retain GPU captures but skip CPU comparisons. All 1160
capture comparisons are finite, bitwise equal and pass unchanged
`atol=rtol=2e-5`. Small J2048 inventories contain 52/104/208 stages for
two/four/eight blocks. Full J21120 inventories contain 4/8/16 attention/output
boundaries. Each full process also replaces the input, poisons mutable scratch
with NaNs and matches every block's distinct alternate output.

Every request records one terminal wait, zero named buffer allocations during
encode and zero activation reads before that wait. Every process rejects
wrong slot order, in-flight release, early compare and early reuse. Backend
segments differ between modes: full-J counts are 243/485/969 for BF16 versus
255/509/1017 for F32. Small F32 counts are respectively [33,33,32], [67,67,66]
and [135,135,134] across gate/warmup/timed requests. The validator initially
rejected an assumed fixed small-J count; it now records those observations
without claiming a cause or identical execution topology. Wait/read/allocation
checks remain strict.

The oracle is the existing saved pretrained BF16 reference on seeded synthetic
inputs, produced by the adapted upstream-semantics PyTorch MPS path. Payload
hashes, complete ordered capture inventories and per-capture hashes were
rechecked. This slice does not rerun upstream, qualify CUDA Flex or use real
decoder latents. PyTorch is absent from the measured native requests.

### Balanced timings and retained drift

A denotes BF16 graph feeds and B denotes F32 FFN graph operands. Each full-J
process prepares one resident prefix, runs a numerical gate and one warmup,
then three timed requests. ABBA executes A1/B1/B2/A2; the four-block reverse
control executes B1/A1/A2/B2. Each row lists the corresponding two per-process
medians. Separate processes bound retained opaque resources; these are not
same-process pairs or a quiet-GPU guarantee.

| Blocks / order | BF16 medians, s | F32 medians, s | Paired latency reductions |
| --- | ---: | ---: | ---: |
| 2 / ABBA | 3.144104, 3.142410 | 2.353582, 2.352883 | 25.14%, 25.12% |
| 4 / ABBA | 6.282318, 8.974073 | 4.789127, 6.799661 | 23.77%, 24.23% |
| 4 / BAAB | 6.283388, 6.285215 | 5.229961, 4.737358 | 16.77%, 24.63% |
| 8 / ABBA | 12.579933, 12.930832 | 10.067220, 10.451315 | 19.97%, 19.18% |

The four-block ABBA baseline drifts by about 43%; keep those samples rather
than replacing them with the reverse control. The reverse control also favors
F32 in both pair directions but does not establish one stable reduction or a
cause for the drift. Slow requests show changes in both encode wall time and
the sum of observed GPU command-root elapsed intervals. Those intervals are
not exclusive shader execution time. No driver-wait, thermal or background-GPU
cause is established.

Timing includes native encoding, the existing terminal wait and GPU capture
copies. Preparation/compilation/upload, scratch poisoning and postterminal CPU
comparison are excluded. The improvement applies to these prepared captured
prefix requests, not complete decoder or video latency.

### Memory, verification and next boundary

| Full-J prefix | Named bytes, both modes | Sampled BF16 footprint, GiB | Sampled F32 footprint, GiB |
| --- | ---: | ---: | ---: |
| 2 | 3,955,236,836 | 5.07-5.23 | 6.71-7.03 |
| 4, including reverse control | 5,419,082,724 | 6.63-6.79 | 8.27-8.43 |
| 8 | 8,346,774,500 | 9.43 | 11.07 |

Named storage does not include opaque MPS allocations. Do not extrapolate
these sampled footprints linearly to all 36 blocks. Every process is serially
guarded by the cache-aware 10% physical floor and pressure check, a process-tree
cap, 64 MiB swap-growth stop and 900-second deadline. Small-J caps are 6/8/12
GiB and full-J caps are 10/14/22 GiB for two/four/eight blocks. Minimum sampled
cache-aware headroom across this matrix is 13.85 GiB; sampled swap growth is
zero. Guard polling does not establish a between-sample peak bound.

Executed release build and regression command:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/build pixelumm_decoder_two_block_probe pixelumm_decoder_four_block_probe pixelumm_decoder_eight_block_probe spec_pixelumm_decoder_block spec_pixelumm_two_block_probe
```

All 30 examples pass: 15 admission/precision/legacy-ABI and 15 fixture/cache
examples. External `tmp/ffn_serial_case_v1.py` records the exact guarded commands
for the small gates and full ABBA/BAAB matrices. The executed external check is:

```sh
python3 -B /opt/mldata_ssd/cogni-ml/hy-nomas-pixelumm-20261026/tmp/validate_ffn_serial_v1.py
```

It exits zero, checking complete capture coverage, 105 requests, source and
binary pins, fixture/cache payload hashes, report/receipt/guard command binding,
declared serial order, and all sampled guard limits. Its output indexes 134
external artifacts. Receipts are not hermetic build attestation. Existing
reports use exclusive creation; reproductions need fresh report names.
The curated index is
`.landmark/evidence/2026-10-27-hy-nomas-pixelumm-ffn-serial.json`; heavy logs,
payloads, scripts and preserved executables remain on the external SSD.

Rollback omits `ffn_f32: true` / `--ffn-f32`. The default remains false. This
closes the earlier sequential two/four/eight-block gate only. Before wider or
default use, admit the full decoder's retained resources and verify a larger
prefix on real decoder inputs; the remaining layers and global projections
are unqualified. Only layers 0-7 have been downloaded for this decoder scope;
the complete checkpoint hash remains unverified. Refresh after scoped source,
fixture/weights, backend/device/toolchain/OS, geometry/precision, ownership,
capture or guard changes, or loss of evidence. No full decoder, diffusion,
Studio or generation ran in this slice.
