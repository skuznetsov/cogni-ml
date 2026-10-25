# Hy-Nomas native Metal frontier

Document status (2026-10-25): the opt-in natural-prompt Qwen26 prefix, GPU crop,
actual-weight text refiner/time embedding and pretrained DiT blocks 0 to 7 now
pass a connected upstream gate at full B1/V5280/T1280/J6560. The bounded MPS
adapter excludes padded attention rows and explicitly zeros their results;
CUDA/Magi padding semantics remain unqualified. Local DiT weights cover these
eight blocks (256 tensors) plus 42 refiner/time tensors; the full checkpoint is
not downloaded. Earlier synthetic-input, short-input and workspace controls
retain their recorded scopes. A header-only planner accounts for complete DiT
parameters and serial residency, but execution beyond eight blocks, full-stack
peaks, video quality and model-wide performance remain open. This composition
probe establishes correctness, not a speedup.
Upstream: Tencent-Hunyuan/Hy-Nomas at
`911cea6458a89c2663513ff7099561ec190dd491` (Apache-2.0 source).

## Qualified slice: eight BF16-cache blocks with real conditioning

Status: VERIFIED within this MPS-adapter composition scope. The live
Qwen26/crop/refiner chain feeds pretrained blocks 0 through 7 at
B1/V5280/T1280/J6560. Keep the existing
23-valid-text-row MPS attention adapter, full-row projections/residuals,
BF16-resident parameters, F32 arithmetic and finite `atol=rtol=2e-5`.

Risk: CAUTION (new checkpoint identities, shared scratch and output lifetime).
Rollback: omit the opt-in eight-block probe; the four-block qualification
remains the control. Authenticate the four additional named checkpoint subsets
and qualify all native BF16 cache words against upstream GPU conversion before
neural execution. Admit aggregate named bytes before Metal allocation. The
strongest failure is a wrong successor or overwritten earlier output bank.

DoD: focused count/admission/lifetime/oracle tests, then a fresh pinned-source
MPS eight-block oracle and guarded native live chain. Compare conditioning and
all sixteen block outputs on full J; require exact repeat after scratch poison,
a finite wrong-last-block control, zero tracked encode allocations/preterminal
activation reads and one terminal wait per request. Keep the cache-aware 10%
reserve, bounded process caps, 64 MiB swap-growth stop and 600 s deadlines.
No full generation, quality or speed claim. Refresh on source/cache/reference,
geometry, precision, ownership, toolchain, OS or device changes.

Observed: the fresh pinned upstream modules execute all eight actual blocks,
168 attention query chunks (21 per block, C256/G16), the full 1280-row refiner
and scalar timestep 500. The oracle consumes authenticated saved Qwen state
26; native execution regenerates all 27 BF16 prefix boundaries from the same
natural prompt, then passes the GPU crop directly to the refiner. The projected
text, time vector and all sixteen block outputs match upstream raw exact in
both native requests: 18 captures / 110,102,528 F32 values per request, zero
bit differences, tolerance failures or maximum absolute error. The repeat
remains exact after NaN poisoning 116 unique nonconstant scratch/output buffers.
All thirteen tensor spans from the prior four-block fixture remain byte-exact.

The eight blocks share intermediate scratch and one prepared MPS workspace;
their sixteen output banks remain distinct. Every request has zero tracked
encode-time MetalBuffer allocations and preterminal activation reads, one
terminal wait and 47 postterminal diagnostic reads (27 Qwen states, crop/mask,
18 consumer outputs). In-flight producer, crop and consumer releases are
rejected. A finite block-3-for-block-4 control preserves blocks 0--3 and fails
both block-4 outputs; a block-6-for-block-7 control preserves blocks 0--6 and
fails both block-7 outputs. A wrong-padding control preserves the producer and
refiner but fails both block-0 outputs. Later fault propagation stays finite.

New source subsets 4--7 contain their exact 32 named F32 tensors under the
pinned HF revision/header; the full checkpoint LFS digest remains declared,
not verified. All 128 new BF16 tensor payloads match upstream conversion word
for word during native GPU conversion and direct BF16 reload. Each resident
block cache contains 302,113,792 BF16 payload bytes; all eight total
2,416,910,336 bytes. Inference block caches need no F32 weight staging or
conversion dispatches. Retained setup F32 refiner staging remains counted.

The new focused target passes 11 Crystal examples; the previous four-block
target passes 14 examples on the widened serial implementation. Sixteen new
CPU Python tests and six live pre-Metal CLI rejections pass. A separately
written streamed audit authenticates all eight F32 subsets/manifests, all
eight BF16 cache files, 12 upstream source files, producer code and selected
reference spans. Both requests' native-reported capture hashes match those
spans. This is correlated source/artifact review, not a second neural oracle
or a separately recorded native payload. MiniQuad verdict: ROBUST in scope.

Aggregate native named bytes: 19,099,905,356; consumer: 6,361,095,684. The
positive sampled process peak is 20,339,880,776 bytes (18.94 GiB); its minimum
cache-aware headroom is 22,184,656,896 bytes. Across the native neural runs,
the minimum headroom is 21,386,215,424 bytes (19.92 GiB). Upstream sampled
peak is 9,593,507,128 bytes. Admitted neural runs preserve 10% reserve,
22 GiB native / 9 GiB upstream process caps, 64 MiB swap-growth stops and
600 s deadlines, without observed swap growth or guard abort. Sampling does
not bound instantaneous peaks. A first pre-Metal CLI admission attempt hit
its undersized 256 MiB cap while reading the existing Qwen diagnostic
snapshot; it is excluded and the six CLI controls pass with 1 GiB caps.
Two new test-only build failures were repaired before the passing spec run.

Reproduce with `make spec_hy_nomas_conditioned_eight`,
`make spec_hy_nomas_conditioned_four`,
`python3 -B -m unittest discover -s spec -p 'hy_nomas_block47*test.py'`,
`python3 -B spec/hy_nomas_conditioned_eight_reference_test.py` and
`make hy_nomas_conditioned_eight_probe`. Use the external guard, SSD
`BUILD_DIR` and exact arguments indexed by
`/opt/mldata_ssd/cogni-ml/hy-nomas-conditioned-eight-20261025/logs/conditioned-eight-receipt-v1.json`.
Repository evidence: `.landmark/evidence/2026-10-25-hy-nomas-conditioned-eight.json`.
Operator-pinned immutable local artifacts remain a trust assumption; copied
companion object ancestry is opaque. Setup I/O/uploads and terminal diagnostic
reads use CPU. Initial video input is synthetic and upstream ND RoPE tables
are uploaded. This does not qualify CUDA/Magi padding semantics, BF16
activations, the full diffusion stack, video quality or generation speed.
Next falsifier: separately budget a longer real-conditioning block chain
before attempting full-stack/video execution; preserve the same geometry,
precision and finite tolerance gates.

## Qualified slice: real masked conditioning through four full-J blocks

Status: VERIFIED within this MPS-adapter composition scope. The live native
natural-prompt Qwen26 GPU crop feeds the actual-weight refiner/time embedding
and pretrained blocks 0 to 3 in one command chain. Keep
B1/V5280/T1280/J6560 tensor geometry, including 1257 padded text positions.
Only the first 23 text positions are valid. Upstream range construction excludes
padded queries and keys. The bounded MPS adapter explicitly zero-fills omitted
query results; this does not prove the CUDA/Magi operator's padding behavior.
Projections and residuals still operate on all 1280 rows. A compact J5303 replay
cannot establish this full padded-tensor boundary.

Risk: CAUTION (mask semantics, buffer lifetime and composition). Rollback:
omit the new opt-in conditioning probe and retain the qualified synthetic
four-block replay. Defaults, tolerances and arithmetic remain unchanged.
Use prefix-only attention on the existing full buffers for B1, zero padded
attention on GPU every request, and share the scalar-time modulation buffer
only through the existing explicit alias permission. Reject arbitrary masks,
multiple batches and unsupported segmentation before encoding this new path.

DoD: focused admission/mask/lifetime tests, a fresh pinned-source MPS refiner
and four-block oracle, and a guarded native live Qwen26/crop/refiner/four-block
chain. Compare projected text, time vector and eight image/text block outputs
at unchanged finite `atol=rtol=2e-5`. A repeat after scratch poisoning must match
exactly; wrong-padding and wrong-successor controls must be detected. Require
zero tracked encode MetalBuffer allocations/preterminal activation reads and
one terminal wait per request. Source/cache/reference identities and aggregate
named bytes must pass before Metal. Use cache-aware 10% reserve, bounded process
caps, 64 MiB swap-growth stop and 600 s deadline. No full generation, video quality
or performance claim. Refresh on source/fixture/geometry/precision, ownership,
toolchain, OS or device changes; missing or altered evidence rejects admission.

Observed: a fresh pinned-source upstream run evaluates the full 1280-row
refiner, scalar timestep 500 and four actual blocks. Its MPS adapter executes
84 attention query chunks (21 per block, C256/G16). A poisoned padded Q/K/V
suffix leaves the valid-prefix result unchanged. The oracle consumes saved
authenticated Qwen state 26, while native execution reruns all 26 Qwen blocks
from the same natural prompt. All 27 BF16 prefix states, the GPU crop and mask
pass on both positive requests and both negative-control requests.

The projected text, time vector and eight block outputs (56,363,008 F32 values
per positive request) match upstream raw exact: zero tolerance failures, bit
differences or maximum absolute error. The second request remains exact after
NaN poisoning 108 unique nonconstant refiner/DiT scratch buffers. GPU handoff
uses the real projected text and scalar vec; expected conditioning is never
uploaded as an inference input. All blocks share scratch/MPS workspace, keep
distinct output banks and use the explicit shared-modulation alias permission.

Each request has zero tracked MetalBuffer allocations during encode, zero
preterminal activation readbacks and one terminal wait. The 39 postterminal
diagnostic reads cover 27 Qwen states, crop plus mask and 10 consumer outputs.
In-flight release of producer, crop and consumer is rejected. A finite wrong-
padding control keeps the producer/refiner exact but fails both block-0
outputs; substituting block-2 weights for block 3 preserves all predecessors
and fails both block-3 outputs. Thirteen focused Crystal examples, eight CPU
Python tests and six live pre-Metal CLI admission rejections pass. A separately
written streamed byte/count audit authenticates all selected reference spans,
their native digests, the four BF16 caches, source subsets/manifests and 12
upstream source files. It is not a second neural implementation. Correlated
MiniQuad source/runtime review returns ROBUST within this scope.

Aggregate native named buffers: 17,676,492,108 bytes. Consumer named buffers:
4,937,682,436 bytes, including full-row refiner scratch, four BF16 block caches,
BF16 refiner parameters and retained setup F32 refiner staging. Native positive
sampled peak: 18,857,406,248 bytes (17.56 GiB); minimum cache-aware headroom:
22,427,516,928 bytes. Upstream sampled peak: 6,165,564,440 bytes. All admitted
runs preserve the 10% reserve, process caps, 64 MiB swap-growth stop and 600 s
deadline with no observed swap growth or abort. Sampling does not bound
between-poll peaks; this diagnostic replay does not establish generation speed.

Reproduce the focused checks with `make spec_hy_nomas_conditioned_four` and
`python3 -B spec/hy_nomas_conditioned_four_reference_test.py`; build with
`make hy_nomas_conditioned_four_probe`. Use an SSD `BUILD_DIR`, the external
memory guard and exact pinned oracle/native arguments indexed by
`/opt/mldata_ssd/cogni-ml/hy-nomas-conditioned-four-20261025/logs/conditioned-four-receipt-v1.json`.
Repository evidence: `.landmark/evidence/2026-10-25-hy-nomas-conditioned-four.json`.
Operator-pinned immutable local artifacts remain a trust assumption; copied
companion object ancestry is opaque. Setup file I/O/uploads and terminal
diagnostics use CPU. Initial video input is synthetic, and upstream ND RoPE
tables are uploaded rather than newly generated natively. No full diffusion,
model-wide residency, CUDA/Magi or BF16-activation parity, quality or speed
claim. The subsequent eight-block extension is qualified above; execution
beyond that boundary still needs separate admission before a full-stack or
video trial.

## Qualified slice: four consecutive BF16-cache blocks at full J

Status: VERIFIED within this four-block replay scope, 2026-10-25. Actual
pretrained blocks 0 to 3 pass B1/V5280/T1280/C2048/H16/D128/FF8192 (J6560),
upstream ND RoPE theta 256, axes [16,56,56], grid [6,22,40], and F32 neural
arithmetic/MPS SDPA C256/G16. Native parameters are physically BF16; upstream
uses BF16-rounded F32 parameters. Starting hidden inputs are deterministic
synthetic MPS tensors, not the full conditioning pipeline. Retain only each
block's image/text outputs; the prior two-block RoPE gate remains separate.
All four blocks share intermediate scratch and one MPS workspace, with distinct
output banks and direct GPU-only successor inputs.

Risk: CAUTION (new checkpoint identities and extended resource lifetime).
Rollback: omit the opt-in four-block probe and retain the two-block path; the
serial extension is one reversible commit. No kernel arithmetic/default changes.
Before execution, authenticate source/cache identities and aggregate named
bytes, root owners before allocation, and reject malformed/short budgets.
The strongest failure is a wrong successor or overwritten output bank; seeded
wrong-block controls and a scratch-poisoned exact repeat must expose it.

DoD: focused workspace/admission/oracle tests, then fresh guarded upstream and
native four-block probes. All eight outputs must be finite and pass unchanged
`atol=rtol=2e-5`; native repeated output bits must match after scratch poisoning.
Require zero tracked encode allocations/preterminal activation reads, one
terminal wait per request, no interblock waits, and exact new BF16 cache words.
Use the cache-aware 10% floor, 64 MiB swap-growth stop, 600 s deadline and bounded
per-process caps calculated before runs. No full generation or speed claim.
Refresh on source/cache/fixture, geometry/precision, lifetime, toolchain, OS or
device changes; reject or narrow the qualification if an authenticated artifact
changes or disappears. Reread live memory before any later GPU run.

Observed: a fresh pinned-source upstream run executes four actual blocks and
104 attention query chunks. All eight native outputs, 53,739,520 F32 values per
request, match upstream raw exact: zero bit differences, tolerance failures or
maximum absolute error. The second request remains exact after NaN poisoning
all 60 unique nonconstant reusable scratch buffers. A separately written
streamed audit authenticates the reference spans, all four cache files and
128 exact BF16 tensors (604,227,584 words), nine upstream source files and the
producer/input recipe. The six inputs and four block-0/1 output spans also
match the prior qualified two-block fixture exactly. This audit is not a second
neural implementation.

Blocks 2/3 were extracted as exact 32-tensor F32 checkpoint subsets from their
named, nonadjacent ranges under the pinned HF revision/header. Native GPU
conversion and direct BF16 reload each match their independent fresh MPS BF16
oracles word for word. Each block has 302,113,792 BF16 payload bytes. The full
checkpoint's declared LFS digest remains unverified because only subsets were
downloaded. All four inference caches use zero source F32 staging/conversion
dispatches; setup conversion has its own terminal waits outside inference.

Each native request has zero tracked encode-time MetalBuffer allocations and
preterminal activation readbacks, no snapshot copies or interblock waits, one
terminal wait, and eight postterminal diagnostic readbacks. Named buffers total
3,555,779,072 bytes: BF16 weights 1,208,455,168, inputs 59,162,624, scratch
1,967,505,920 and MPS workspace 320,655,360. Owners are rooted before Metal
allocation and retained until completion; failed/incomplete drains retain
resources. Serial preflight validates every successor before appending work;
a late rejected view does not bind an unused chain and can retry on a fresh
chain. The registry getter returns a detached Array for the bounded two-to-four
cardinality; external Tuple-specific callers would need adjustment.

Substituting block 0 at position 2 preserves blocks 0/1 exactly, then produces
10,809,018 image and 2,621,397 text tolerance failures at position 2. Substituting
block 2 at position 3 preserves blocks 0/1/2 exactly, then produces 10,810,014
image and 2,621,388 text failures at position 3. Both controls stay finite and
make no repeat claim. The classification predicate rejects missing/duplicate
captures, nonfinite values, earlier failure or an intact fault output.

Twenty-four focused Crystal examples and 30 CPU Python tests pass. Eight live
CLI controls reject a short budget, incorrect reference digest, swapped new
caches, simultaneous control flags, existing report, widened tolerance, bad
sidecar digest or wrong source order before Metal initialization. The native
sampled process peak is 3,958,524,040 bytes with minimum cache-aware headroom
34,235,318,272 bytes; upstream peaks at 6,153,505,696 with headroom at least
30,634,344,448 bytes. All admitted runs preserve the 10% reserve, 6 GiB native/
10 GiB upstream neural caps, 64 MiB swap-growth stop and 600 s deadline, with
zero observed swap growth or guard abort. Sampled guards do not bound
instantaneous peaks or establish a speedup or causal memory improvement.

Evidence: `.landmark/evidence/2026-10-25-hy-nomas-four-block.json` and
`/opt/mldata_ssd/cogni-ml/hy-nomas-four-block-20261025/logs/four-block-receipt-v1.json`.
Correlated MiniQuad source/runtime review plus numerical controls: ROBUST within
the stated scope. Operator-pinned fixtures and immutable local files remain
trust assumptions; copied companion object ancestry is opaque. Setup I/O and
postterminal diagnostics use CPU. No full generation, quality, full-stack fit,
CUDA/Magi, BF16 activation or performance claim. Next falsifier: a bounded
actual conditioning-to-four-block replay before extending residency further.

## Qualified slice: full-J RoPE and two-block output gate

Status: VERIFIED within the same-input two-block scope, 2026-10-25. Actual
pretrained blocks 0 to 1 pass B1/V5280/T1280/C2048/H16/D128/FF8192 (J6560), using
upstream ND RoPE with theta 256, axes [16,56,56] and grid [6,22,40]. This is a
block probe with deterministic synthetic GPU hidden inputs. Parameters remain
physically BF16 in the native owner and BF16-rounded F32 in upstream; neural
math and attention remain F32. No production arithmetic or default changes.

DoD: a fresh pinned-source upstream MPS run and the native connected graph must
agree at finite `atol=rtol=2e-5` for Q/K after video RoPE and image/text outputs
of both blocks. Retain only these eight checkpoints, not all eighty. Block-0
outputs feed block 1 on GPU, scratch/workspace are shared, and all diagnostic
copies precede overwrite in the same command chain. A poisoned repeated native
request must reproduce the first request exactly, with zero tracked encode
allocations/preterminal activation reads and one terminal wait per request.
Wrong RoPE and wrong-successor controls must be detected. Authenticate exact
cache identities, reference inventory/digests and aggregate named-byte budget
before Metal; reject an insufficient budget and malformed fixtures.

Risk: CAUTION (larger geometry, positional ordering and shared GPU lifetime).
Rollback: omit the new opt-in probe; retain the qualified short replay. Root
owners before allocation and retain them if terminal completion is unknown.
Use the cache-aware 10% reserve, 64 MiB swap-growth stop and 600 s deadline.
Set bounded process caps from named buffers, retained selected tensors and MPS
headroom before a run; the external guard covers opaque allocations by sampling.
No full generation, full checkpoint download, speed or physical-peak claim.

DoD commands: build/run the new `hy_nomas_full_j_probe` under the memory guard;
run its focused native admission tests and Python oracle inventory tests.
Expected signal: eight upstream comparisons and eight exact repeat checks pass;
negative controls reject the tested faults without widening either tolerance.
Refresh on scoped source/cache/fixture, geometry/precision, lifetime, toolchain,
OS or device changes.

Observed: a fresh pinned-source upstream run and both native requests agree
on all eight selected captures, 70,123,520 F32 values per request, with zero
raw-bit differences, tolerance failures and maximum absolute error. Repeating
after NaN poisoning all 56 unique nonconstant reusable scratch buffers remains
exact. The fixture holds six inputs and eight captures (339,656,704 bytes),
instead of retaining all eighty intermediate tensors. Only four Q/K buffers
need extra GPU snapshots (173,015,040 bytes); the four output banks are already
distinct and are read only after completion. Block-0 outputs feed
block 1 directly on GPU through the existing shared scratch/MPS workspace.

Each request has zero tracked encode-time MetalBuffer allocations and
preterminal activation reads, four diagnostic GPU copy dispatches, one terminal
wait and eight postterminal reads. The owner is rooted before its first Metal
allocation and retained through completion; failed or incomplete drains retain
resources. Native BF16 caches have zero F32 source staging/conversion dispatches.
Named buffers total 3,017,087,488 bytes, including BF16 weights 604,227,584,
scratch 1,860,026,880, MPS workspace 320,655,360, inputs 59,162,624 and the four
RoPE snapshots. These are named-byte counts, not physical memory measurements.

The identity-RoPE negative produces finite Q/K and output failures. Substituting
block 0 for the successor preserves all block-0 captures exactly but causes
10,808,815 image and 2,621,402 text output tolerance failures in block 1. Negative
modes report no repeat claim. Nine focused Crystal examples and eight Python
checks pass. Seven CLI controls reject a short budget, swapped cache identities,
widened tolerance, bad sidecar digest, truncated sidecar, bad reference digest
or existing report before Metal initialization. A separate streamed hash/count
audit binds each selected native SHA to its exact fixture span on both requests,
all nine pinned upstream source files, the producer and both qualified caches.
This audit is not another neural implementation.

The positive native run samples a 3,400,582,968-byte process peak, minimum
cache-aware headroom 34,521,726,976 bytes and pressure-free minimum 90%. The fresh
upstream run samples 5,158,063,152 bytes, minimum headroom 30,819,106,816 bytes
and pressure-free minimum 88%. All bounded runs retain the 10% reserve,
6 GiB native/8 GiB upstream caps (3 GiB small checks), 64 MiB swap-growth stop
and 600 s deadline, with zero observed swap growth and no guard abort. Sampling
does not bound instantaneous peaks or establish a causal memory improvement.

An initial native run completed comparisons but could not publish its report:
Crystal rejects mode `wx`. That run is excluded. Report publication now closes
a sibling private tempfile before an atomic no-replace hard link; the focused
test covers complete contents, refusal to overwrite and temporary-name cleanup.

Correlated MiniQuad verdict: ROBUST for this scope. The upstream block class is
pinned and unchanged, with chunked PyTorch MPS F32 SDPA as the labeled attention
surrogate. The oracle generates ND RoPE on MPS; native consumes the saved table
and qualifies its application, not a new native ND table generator. Setup file
I/O/uploads and postterminal streamed diagnostics use CPU. Neural execution and
block handoff stay on GPU, with no added inference synchronization. Caller-pinned
fixture SHA remains a trust boundary; receipt provenance is not hostile-producer
authentication, and local files must remain immutable during use. Companion
objects retain pinned bytes with opaque source ancestry; the build is not
hermetic. No full conditioning, full model, CUDA/Magi/BF16-activation, speed,
video generation or quality claim follows.

Exact guarded arguments, source/artifact pins and excluded attempts are in
`/opt/mldata_ssd/cogni-ml/hy-nomas-full-j-20261025/logs/full-j-receipt-v1.json`;
the curated index is `.landmark/evidence/2026-10-25-hy-nomas-full-j.json`.
Next: extend authenticated BF16 cache coverage beyond the two local blocks and
qualify bounded N-block execution before full conditioning or generation.

## Qualified slice: two-block BF16 source-cache replay

Status: VERIFIED within the B1/V6/T23/C2048/H16/D128/FF8192 saved-input scope,
2026-10-25. The existing two-block owner optionally consumes authenticated BF16
derivatives for actual blocks 0 and 1. It preserves the separate-workspace
control, shared-workspace replay and repeat, with direct GPU hidden-state
handoff and GPU diagnostic copies before scratch reuse. Block 1's new native
derivative passes its source/reload/repeat gates against a fresh independent
MPS BF16 weight conversion. Block 0 reuses its previously qualified derivative
and retained conversion oracle; the receipt binds that producer to its original
source version. No neural upstream block was freshly rerun in this slice.

DoD: all 64 resident weight tensors match independent MPS BF16 oracles word for
word; all 80 intermediate captures per request pass finite `atol=rtol=2e-5`
against the authenticated saved two-block upstream run. Shared/repeat outputs
must match the separate-workspace control exactly. BF16 plans must have zero
F32 raw staging and conversion dispatches, zero tracked encode allocations and
preterminal activation reads, and one terminal wait per request. Reject wrong
block identities/digests/dtypes and insufficient aggregate budgets before
Metal; a wrong-successor control must expose block-1 output differences while
block 0 still passes. No activation CPU round trip during neural execution.

Risk: CAUTION (source identity and shared GPU lifetime). Root owners before
allocations; release only after the existing terminal drain; retain resources
if completion is unknown. Use the cache-aware 10% reserve, 4 GiB sampled probe
cap, 64 MiB swap-growth stop and 600 s deadline (3 GiB for small checks/builds).
Rollback: omit the optional BF16-source replay and retain the F32-source path.
No full model, full-J, speedup, physical peak reduction or generation claim.

Observed: all 64 resident weight tensors (302,113,792 BF16 words) match their
independent MPS conversions exactly. All three requests reproduce all 80 saved
upstream captures (3,186,688 F32 values per request) with zero raw-bit
differences, tolerance failures and maximum absolute error. Shared results and
repeat also match the separate-workspace control exactly after poisoning all
56 unique reusable nonconstant scratch buffers. Native parameters are physically
BF16; neural math and attention remain F32. The saved upstream control uses
BF16-rounded parameter values in F32 storage with F32 MPS math/SDPA.

| Two-block loader | BF16 destinations | F32 raw staging | Control named peak | Source GPU casts |
| --- | ---: | ---: | ---: | ---: |
| Original F32 source | 604,227,584 | 1,208,455,168 | 2,032,759,040 | 64 |
| BF16 source caches | 604,227,584 | 0 | 824,303,872 | 0 |

The shared phase uses 727,380,096 named bytes, including 12,746,752 diagnostic
snapshot bytes. This is exact buffer arithmetic: 1,208,455,168 bytes (1.125 GiB)
of cold F32 staging and 64 setup conversion dispatches are removed. The existing
109,670,528-byte short-input scratch/workspace reuse is preserved. No timing
speedup or physical-peak reduction follows from these figures.

Each request has zero tracked encode-time MetalBuffer allocations and
preterminal activation reads, one terminal wait and 80 postterminal activation
diagnostics. All 64 raw weight-byte comparisons run after the completed control,
before its plans are released. Owners are rooted before their first allocation
and retained through terminal completion. Checkpoint byte I/O, cache creation
and postterminal diagnostics run on CPU; neural execution and block handoff
stay on GPU. There is no new inference synchronization.

Eighteen focused Crystal examples and eleven Python checks pass. Six live CLI
controls reject swapped cache/oracle inventories, a bad digest, a self-oracle,
a one-byte-short budget and an existing report before Metal initialization.
Cross-role hardlinks and identical file digests are rejected too. The wrong
successor control keeps block 0 exact but exposes 12,281 image and 47,102 text
output tolerance failures in block 1. The original F32-source serial replay
also passes all three requests. A separately written streamed-byte audit binds
the two cache files to all 64 oracle tensor hashes, original F32 files/manifests,
producer scripts/reports and retained upstream reference.

The two-block native run samples a 931,726,704-byte process peak, minimum
cache-aware headroom of 37,414,289,408 bytes and zero swap growth. The fresh
block-1 conversion oracle samples 2,586,789,352 bytes. All bounded runs preserve
the 10% reserve and their process/swap/deadline limits. Samples do not prove
instantaneous bounds; no physical-peak comparison is claimed.

Build with `make BUILD_DIR=FRESH_SSD_DIR hy_nomas_bf16_serial_probe`; run the
focused suite with `make BUILD_DIR=FRESH_SSD_DIR spec_hy_nomas_bf16_cache`.
The opt-in CLI requires the reference/conditioning, ordered cache0/cache1 and
oracle0/oracle1 files with trusted SHA-256 pins, a fresh report and aggregate
named-byte budget. Exact guarded arguments and build/source pins are in
`/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-serial-20261025/logs/serial-receipt-v1.json`;
the curated index is `.landmark/evidence/2026-10-25-hy-nomas-bf16-serial.json`.
Correlated MiniQuad verdict: ROBUST for this scope. CLI oracle pins remain
caller-trusted; the external receipt supplies provenance, not hostile-producer
authentication. Local files must remain immutable during use. Companion object
bytes are pinned but their source ancestry remains opaque; the build is not
hermetic. Refresh on scoped source, checkpoint/fixture, precision/geometry,
loader/lifetime, compiler/SDK/OS/device or artifact changes. The full-J positional/
output extension is qualified above; next is cache coverage beyond these two
blocks. Full-stack composition remains open.

## Qualified slice: single-block GPU-produced BF16 source cache

Status: VERIFIED within the one-block saved-input scope, 2026-10-25. The
already-local authenticated block 0 uses the existing native GPU F32-to-BF16
kernel. Its completed BF16 buffers are serialized as a fresh safetensors
derivative outside inference and reloaded directly into BF16 destinations.
The original source and saved-conditioning paths remain controls. No inference
default or numerical policy changes, full checkpoint download, or full
generation are admitted.

DoD: all 32 source names/shapes and 151,056,896 BF16 words match an independent
upstream MPS conversion exactly; all 40 short-block intermediates pass the
authenticated upstream finite `atol=rtol=2e-5` gate and raw-F32 cache/control
equivalence. The BF16 reload must have zero F32 raw staging, no named allocation
or activation readback during encode, and one terminal wait per request. Cache
publication follows successful source, reload and repeat gates and terminal
owner cleanup; it never overwrites a file. The private staging directory is
removed on qualification failure. A report is a separate index to the already
qualified cache; a failed report write does not revoke its numerical result.
Authentication, truncated/corrupt payload, incompatible dtype/layout, incomplete
or failed completion, and insufficient budget must fail closed. Cache creation
and byte serialization are setup diagnostics, not inference operations.

Risk: CAUTION (derived checkpoint serialization and GPU resource lifetime).
The strongest failure is publishing or releasing a buffer still borrowed by
GPU work; the owner must remain rooted until the existing terminal drain, and
an incomplete drain retains its resources. Guard each actual probe/oracle with
a 4 GiB sampled process cap, cache-aware 10% reserve, 64 MiB swap-growth stop
and 600 s deadline; builds/small tests use 3 GiB. Rollback is to omit the opt-in
cache diagnostic and keep the F32-source loader. No whole-pipeline speed or
memory-peak claim follows from a one-block result.

Observed: a fresh independent upstream PyTorch 2.8.0 MPS F32-to-BF16 conversion
and native conversion agree on all 32 names/shapes and 151,056,896 BF16 words.
The source, cache reload and cache repeat requests each reproduce all 40 saved
upstream intermediates (1,593,344 F32 values) with zero raw-bit differences,
zero finite-tolerance failures and maximum absolute error zero. Cache/control
differences are also zero. These are authenticated retained upstream captures,
not a fresh upstream block invocation. Native weights are physically BF16;
neural math/attention remain F32. The upstream intermediate control uses
BF16-rounded parameter values in F32 storage and F32 MPS math/SDPA.

| One-block phase | BF16 destinations | F32 raw staging | Total named bytes | GPU casts |
| --- | ---: | ---: | ---: | ---: |
| Original F32 source | 302,113,792 | 604,227,584 | 1,016,509,568 | 32 |
| BF16 cache reload/repeat | 302,113,792 | 0 | 412,281,984 | 0 |

The native cache is 302,117,424 bytes including its header; parameter payload
is 302,113,792 bytes (288.118 MiB). The named setup reduction is 604,227,584
bytes (576.236 MiB). Each request observes zero tracked encode-time buffer
allocations and preterminal activation reads, one terminal wait, and 40
postterminal activation reads. Exact raw weight-byte inspection and file I/O
are postterminal/setup diagnostics. No extra inference synchronization or
CPU neural arithmetic is introduced. No timing speedup is measured here.

The final native run's sampled process peak is 1,110,476,384 bytes, minimum
cache-aware headroom 38,003,752,960 bytes, minimum pressure-free value 91%, and
observed swap growth zero. The independent conversion oracle peaks at
2,589,197,920 sampled bytes. Both preserve the 4 GiB cap and 10% reserve;
sampling is not an instantaneous peak proof or a reservation. Thirteen focused
Crystal examples and seven Python checks pass. Live CLI controls reject a
one-byte-short budget and wrong oracle digest before Metal initialization,
and preserve an existing cache. Unit falsifiers cover malformed/truncated
files, dtype/name errors, incomplete/failed terminals, equal-shape source/target
permutation, failed qualification cleanup and no-clobber publication.

Build with `make BUILD_DIR=FRESH_SSD_DIR hy_nomas_bf16_cache_probe`; use
`make BUILD_DIR=FRESH_SSD_DIR spec_hy_nomas_bf16_cache` for the native checks.
The CLI requires authenticated reference/conditioning and caller-pinned BF16
oracle, fresh cache/report paths and an aggregate named-byte budget. The
external receipt binds the oracle file, its producer report/script, original
source/manifest and retained reference. Companion objects retain authenticated
bytes with opaque source ancestry; the build is not hermetic. Correlated
MiniQuad verdict is ROBUST within this scope. Evidence:
`.landmark/evidence/2026-10-25-hy-nomas-bf16-cache.json` and
`/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-cache-20261025/logs/cache-receipt-v2.json`.
Refresh on source/checkpoint, fixture, precision, geometry, toolchain/device,
loader/lifetime or artifact changes. Treat local source/cache files as immutable
while authenticated consumers run. The two-block BF16-source gate is now
qualified above, including the later selected-capture full-J positional/output
gate. No full-stack cache, fit, performance or generation claim.

## Qualified slice: full-DiT metadata residency accounting

Status: VERIFIED for authenticated header inventory and named-byte arithmetic,
2026-10-25. No model or Metal device is initialized by this diagnostic. The
252,904-byte pinned header covers 2,268 F32 tensors and 34,213,738,368 payload
bytes. All 54 native block binding sets match the header. Projected BF16 storage
is 17,106,869,184 bytes (15.932 GiB), including all camera/action, refiner,
input, timestep and final-layer parameters. The whole checkpoint payload and
its declared LFS digest remain unverified; only the header is authenticated here.

The current loader retains F32 conversion sources through terminal completion
alongside BF16 destinations. The following estimates retain distinct outputs
for every block, one common scratch set and one MPS workspace. N=54 is an
unimplemented extension of the two-block owner, not an executable allocation
plan. Long geometry is B1/V5280/T1280/C2048/H16/D128/FF8192, with compact
attention query chunk 256 and 16 heads per group.

| Scenario | Cold named bytes | Warm named bytes | Scratch/workspace bytes |
| --- | ---: | ---: | ---: |
| V6/T23, two actual blocks | 1,923,088,512 | 714,633,344 | 110,145,664 |
| J6560, two blocks | 4,052,527,616 | 2,844,072,448 | 2,180,682,240 |
| J6560, all DiT parameters and 54 distinct output pairs | 56,354,907,456 | 22,141,169,088 | 4,975,137,280 |

Cold minus warm is exactly the retained F32 source payload. A separately
reported heuristic adds three maximum score tiles plus 256 MiB: 590,872,576
bytes at J6560. The all-parameter estimates then become 53.035 GiB cold and
21.171 GiB warm. Neither value bounds opaque/queued MPS allocations. Execution
workspace for camera/history/KV, refiner/input/final layers, scheduler/latents,
Qwen and PixelUMM remains excluded; all checkpoint parameters are counted.
The cold estimate does not fit the observed 31.137 GiB cache-aware headroom
after the 10% physical reserve. This snapshot is not reserved capacity or
runtime admission, and a DiT-only warm comparison does not admit the pipeline.

Diagnostic snapshots are separate from these totals: the 40 captures per block
use 12,746,752 bytes for two short blocks, 2,967,666,688 bytes for two J6560
blocks, and 80,127,000,576 bytes for 54 J6560 blocks. They must not be confused
with all 55 native scratch buffers. A future two-bank output policy could save
52 output pairs, 2,794,455,040 bytes (2.603 GiB), at N=54; current fixed bindings,
alias checks and exactly-two-block APIs do not implement it.

Download plan: the previously qualified block-0/block-1 and text/time subsets
cover 106 tensors and 1,791,766,528 source payload bytes. Their paths and sizes
are observed again; their payload hashes are not rerun in this metadata slice.
The remaining 2,162 tensors require 32,421,971,840 F32 payload bytes (30.195 GiB),
before transfer/container overhead. The two remaining source-file ranges are
`[252912,968270320)` and `[2176725488,33630679920)`. No download was performed.
A complete derived BF16 cache would contain 15.932 GiB of parameter payload;
that cache does not yet exist. Qwen weights are already local separately;
PixelUMM residency and the full pipeline download plan remain open.

DoD passed: `make spec_hy_nomas_residency` with the authenticated external
header (eight examples, zero failures/errors/pending); a freshly built native
planner report with zero tracked GPU allocations; and a separately constructed
Python closed-form/header audit, including the prior short-run receipt. CLI
controls reject a same-length tampered header and a comparison budget one byte
below the selected estimate. Correlated MiniQuad review is ROBUST for this
metadata scope; no new neural numerical comparison, speedup or physical peak
was measured. Builds/checks retain a 3 GiB sampled cap, cache-aware 10% reserve,
64 MiB swap-growth stop and 600 s deadline. The existing numerical tolerance
and inference execution path are unchanged.

Build with `make BUILD_DIR=FRESH_SSD_DIR hy_nomas_residency_plan`, then run the
binary with `--header PINNED_HEADER_JSON`. `--estimate-budget-bytes N` is only
a comparison; it does not replace any runtime memory guard. Evidence and exact
guarded arguments are indexed in
`.landmark/evidence/2026-10-25-hy-nomas-residency.json` and
`/opt/mldata_ssd/cogni-ml/hy-nomas-residency-20261025/logs/residency-receipt-v1.json`.
Rollback: omit this diagnostic; no inference/default change is admitted.
Refresh on checkpoint/header, scratch/capture/attention geometry, loader
lifetime or accounting changes; reread live telemetry before any later probe.

The two-block cache gate above now qualifies removal of 1,208,455,168 bytes
of F32 staging with the unchanged upstream intermediate gate. Next falsifier:
expand positional/output coverage and cache coverage beyond those blocks.
Keep setup/cache I/O outside inference, with no new waits or
activation readbacks. The 31.864 GiB full-DiT cold staging removal remains a
modeled extension; no whole-stack cache or runtime fit is implemented here.

## Qualified slice: distinct pretrained blocks 0 to 1 with shared scratch

Status: VERIFIED within the saved-conditioning F32-math control below,
2026-10-25. Two distinct pretrained modules match all 80 upstream intermediates
(3,186,688 F32 values). Both independent plans and two repeated shared-workspace
requests have zero raw-F32 differences against this capture. Every comparison
also passes the unchanged finite `atol=rtol=2e-5` gate. The input is the preceding
authenticated refiner output: 23 text tokens, one timestep vector and six seeded
hidden video tokens with saved RoPE. This replay does not rerun Qwen/refiner or
exercise the video input projection. Setup uploads these saved operands; the
two sequential blocks, weight conversion, attention and snapshot copies execute
on GPU.

The new bounded extractor authenticates the complete pinned header and fetches
only the 604,227,584-byte `double_blocks.1.` F32 payload. Its subset file is
604,231,488 bytes, SHA-256
`026568ecff75c5c4a4d771d313469a9245eb115c34f9117af7ac21290369651c`.
Both actual subset files and manifests are hashed before native preparation.
The source is tencent/Hy-Nomas revision
`bac719a80c8d02f91c485fca804bb8221e149b70`; the declared LFS digest of the
34,213,991,280-byte whole checkpoint is not verified. Trusted local files must
remain immutable from authentication through use. Native weight storage is
BF16 with F32 math/attention; the pinned upstream MPS adapter holds BF16-rounded
weight values in F32 parameter storage. This qualifies that declared control,
not CUDA/Magi or full BF16 activation execution.

The shared owner uses one intermediate scratch set, one MPS workspace and
distinct final-output banks. GPU snapshots preserve all 40 checkpoints per block
before reuse. Each shared request first poisons all 56 unique nonconstant scratch
buffers with NaNs and then emits 80 snapshot copies; both repeats remain exact
against the independent plans and each other. Each request has one terminal wait,
zero tracked encode-time MetalBuffer allocations and zero pre-terminal activation
readbacks, followed by 80 diagnostic reads. Diagnostic poisoning/copies are
extra probe work and establish no throughput benefit.

The strongest identity falsifier replaces successor weights with block 0 while
retaining the true block-1 oracle. Block 0 remains exact; 38 block-1 checkpoints
fail the unchanged numerical gate, including both final outputs. A separate
production preflight guard now rejects equal-shape/equal-dtype weight-map
rebinding that would disagree with the block's cached Linear/RMS resources,
before workspace allocation or dispatch. Disabling only this guard in an
external test runner makes its rejection test fail as expected.

Scratch plus workspace falls from 219,816,192 to 110,145,664 named bytes: a saving
of 109,670,528 bytes (104.59 MiB, 49.9%) for these operands. Both phases are
admitted before the first GPU allocation. The independent-control estimate is
2,032,759,040 named bytes including both raw conversion stages and destinations;
bounded host reference/capture arrays bring the estimate to 2,112,358,660 bytes.
Conversion staging is released after the control terminal, so the lower shared
phase total is not an isolated measure of reuse. Opaque MPS/runtime allocations
are excluded from these estimates and named counters. A rooted owner and chain
lease retain staging, plans and borrowed resources until terminal completion;
incomplete drains retain them rather than freeing in-flight resources.

DoD passed: 14 CPU subset/oracle tests; `make spec_hy_nomas` with the authenticated
admission and timestep fixtures (60 examples: 56 passed, zero failures/errors,
four pre-existing optional external-oracle cases pending); a fresh upstream
two-module capture; positive `hy_nomas_pretrained_serial_probe`; and
`--successor-block0-control`, whose success means the intentional mismatch was
detected. An independent byte audit checks all 3,251,712 fixture values are finite,
all 40 block-0 checkpoints equal the prior capture and all 40 successor checkpoints
differ from block 0. The one-byte budget test rejects before any tracked GPU
allocation/readback. Fresh bridge objects and the native executable were built
under the guard. Correlated MiniQuad review is bounded by these root-run checks.

All runs use a file-cache-aware 10% host reserve, 64 MiB swap-growth stop and
at most a 600 s deadline. Builds/small tests have a 3 GiB sampled process cap; the upstream
and two-block probes have 4 GiB. Across the three oracle/positive/negative runs,
the highest sampled footprint is 2,454,849,288 bytes, minimum cache-aware headroom
34,913,026,048 bytes and observed swap growth zero. Sampling can miss short peaks;
file-backed cache is a reclaimability estimate. The native admission also leaves
10% of its 4 GiB process estimate unused; this is separate from the physical host
reserve and is not a bound on opaque allocations.

Evidence and exact argument vectors are indexed in
`.landmark/evidence/2026-10-25-hy-nomas-pretrained-serial.json` and
`/opt/mldata_ssd/cogni-ml/hy-nomas-pretrained-chain-20261025/logs/serial-receipt-v1.json`.
Build target: `make BUILD_DIR=FRESH_SSD_DIR hy_nomas_pretrained_serial_probe`.
The executable requires the external memory guard; use new report/log paths
when reproducing the receipt commands. Rollback: omit this opt-in probe and retain
the single-block/independent controls. Full generation, full-stack residency,
long-query performance and quality claims remain outside this slice. Refresh
after scoped source, weights/input, precision, toolchain/SDK/OS/device, lifetime
changes or evidence loss/hash mismatch.

## Qualified slice: resident Qwen conditioning through the actual text refiner

Status: VERIFIED within the one-prompt F32-math control below, 2026-10-25.
The qualified natural tram prompt supplies
BF16 `[1,1280,3584]` conditioning and an authenticated mask with 23 valid
prefix rows. The selected upstream preset uses `single_refiner`: masked
context pooling, separate learned timestep/context embeddings, an input
projection and two gated attention/MLP refiner blocks. A single linear
projection is not an admitted substitute. Actual `time_in` weights provide
the DiT vector independently of the refiner timestep embedder.

CAUTION: extend only the opt-in diagnostic path. The implemented
composition compacts valid prefix rows and casts BF16 to F32 on GPU, runs the
complete refiner with BF16-resident checkpoint weights and declared F32 math,
then feeds a six-video-token, single-segment native block. All 42 refiner/time
tensors come from a bounded subset of the pinned distilled checkpoint. The
full masked upstream refiner and compact control must first agree on valid
rows at the unchanged finite gate. Compaction is rejected for nonbinary,
empty or non-prefix masks; multi-segment, camera, guidance and full generation
remain outside admission. Full BF16 activation or CUDA FlashAttention parity
is not implied by this MPS F32 control.

Strongest failure: padding compaction changes pooling/reduction order, or a
BF16 attention boundary promotes a small F32 difference into a failed gate.
DoD: CPU subset/fixture/admission rejection tests, a fresh pinned upstream
full-versus-compact capture, and a guarded native comparison of every saved
refiner/block intermediate on the same conditioning. The connected Qwen
producer, crop, refiner and block must allocate zero named GPU buffers and
read zero activations while encoding, with one terminal wait. All resources
must survive until that wait; early release and budget underflow must reject.
Keep `atol=rtol=2e-5`, cache-aware 10% reserve, sampled process cap, 64 MiB
swap-growth stop and bounded deadline. Commands and observed scope will be
recorded with the resulting evidence. If any gate fails, preserve the first
mismatch and narrow the qualification rather than relaxing tolerances.
Rollback: omit the new opt-in conditioning/block arguments; existing Qwen
and block defaults remain the controls. Scratch and downloads stay under
`/opt/mldata_ssd/cogni-ml/hy-nomas-text-conditioning-20261025`.

### Connected result and numerical discriminators

The fresh pinned upstream capture compares the full masked 1280-row refiner
with its 23-row compact control at 45 checkpoints; every comparison passes
`abs(error) <= 2e-5 + 2e-5*abs(reference)`. The native saved-conditioning control
and connected Qwen producer each pass all 53 refiner and 40 DiT intermediate
comparisons. Both use actual checkpoint weights, resident in BF16 with F32
math and attention. The source is tencent/Hy-Nomas revision
`bac719a80c8d02f91c485fca804bb8221e149b70`, not synthetic refiner weights.

Two same-input probes isolated and repaired execution-order differences:

- Upstream creates a timestep-independent 128-element frequency basis in CPU
  F32 before copying it to GPU. A fixed 512-byte basis reproduces those source
  constants; runtime phase/trigonometry stay on GPU. Identical F32 phases still
  differed with ordinary Metal `sin/cos` by up to `6.247125566e-5`. Explicit
  `precise::sin/cos` match all 2048 output words at eight tested timesteps,
  including fractional and unseen values. Safe math mode alone did not select
  accurate transcendental functions.
- On exact upstream refiner residual operands, the common gated add differed
  in 487 words (maximum `3.814697265625e-6`). A refiner-local kernel with FP
  contraction disabled matches all 47,104 words. It preserves upstream's
  separate F32 gate multiply and residual add without another dispatch,
  buffer or wait. The previously failing downstream text FC2 intermediate now
  passes the unchanged gate. Replaying that FC2 alone on its exact upstream
  activation already matched all 47,104 words, excluding a local FC2 defect
  for that input.

The final connected run keeps all 27 Qwen hidden boundaries and the crop
raw-BF16 exact (134,313,984 plus 4,587,520 words), and the 1280-entry mask exact.
The refiner/DiT are **finite-gate**, not bit-exact, qualifications: the refiner
MLP FC2 still has a maximum absolute difference of `0.000186920166015625`,
within the existing relative allowance. Every saved intermediate passes;
small differences are not hidden by checking only the final output.

Named admission is 14,707,308,492 bytes. The connected guarded command sampled
a 15,748,213,632-byte peak (14.667 GiB), at least 27,880,472,576 bytes of
cache-aware headroom and zero swap growth. The unchanged guard enforces a 10%
physical reserve, 15 GiB sampled process cap, 64 MiB swap-growth stop and 600 s
deadline. It cannot exclude between-sample spikes. There is one terminal wait,
zero tracked encode-time MetalBuffer allocations, zero pre-terminal activation
readbacks and 122 post-terminal diagnostic reads. Opaque MPS allocations are
outside the named counters. The 3.834 s GPU interval is a single correctness
run including Qwen and setup conversions, not a throughput benchmark or video
latency estimate.

A session-level chain lease is registered before the first prepared-weight
conversion. A strong session root keeps raw conversion staging and prepared
plans alive through partial encode failures or an incomplete drain. Release
rejects until terminal completion; it performs no extra wait. Without this
owner, downstream plan roots would not protect PreparedWeights' raw staging.
The standalone projection replay has the same strong ownership boundary. An
incomplete drain retains its resources until process exit; reusable recovery
inside a long-lived caller is not qualified by this top-level probe.
Trusted local checkpoint files must remain immutable between authentication,
opening and use; the existing path checks are not an adversarial concurrent
replacement guarantee.

Verification: `make spec_hy_nomas` reported 57 examples: 53 passed, zero failures
or errors, and four pending external-fixture probes; the new timestep fixture
was enabled.
Twelve CPU subset/oracle admission tests passed. The saved-conditioning and
connected probes both passed at unchanged tolerances. Correlated MiniQuad
review and root-run falsifiers support ROBUST for this serial one-prompt
scope; they do not establish independent CUDA parity or all-input behavior.
Evidence and exact reproduction arguments are indexed in
`.landmark/evidence/2026-10-25-hy-nomas-text-conditioning.json` and the external
receipt under `logs/conditioning-receipt-v1.json` in the scratch root above.
Refresh on source, input/mask/timestep, checkpoint, precision, toolchain/SDK,
OS/device, scheduling/lifetime changes or loss/hash mismatch of the evidence.

## Scope and design laws

The user has temporarily redirected video porting to Hy-Nomas. The immediate
slice is one native dual-stream DiT block on a single segment. Text/video
modulation, Q/K normalization, video RoPE, joint attention and feed-forward must
follow the pinned upstream execution order. Camera conditioning is a separate
post-block module and remains outside this first slice. Unsupported
history/mask/conditioning modes must fail at admission rather than silently
execute an unmasked substitute.

GPU command encoding must consume resident inputs and reusable scratch. No
activation readbacks, inter-operation host waits or GPU-buffer allocations in
the block's encode method are admitted. A diagnostic runner may perform its
existing terminal command wait and read all saved intermediate buffers after
completion. Setup may upload inputs, weights and positional constants. Neural
operations, including any source-dtype conversion, execute on GPU.

Use existing native Metal/MPS operations and strict checkpoint admission.
Checkpoint bindings must validate names, shapes, dtype and memory budget before
allocating. F32 controls and original BF16 residency are separate representations;
INT8/FP8 support is not implied by BF16/F32 admission. Source-file I/O and GPU
uploads at setup are not a CPU inference/offload path.
Use `HyNomasDualStreamBlock.weight_plan` for model-specific admission before
generic preparation. Checkpoint bytes and headers must be trusted and immutable
for the lifetime of the plan and prepared weights. The loader's metadata checks
detect ordinary file replacement; they are not an adversarial content-integrity
guarantee against in-place rewrites that restore file metadata.

The finite numerical gate remains `abs(error) <= 2e-5 + 2e-5*abs(reference)`.
Do not relax it after observing a mismatch. The oracle must invoke pinned
upstream code; any CUDA-only adapter must be enumerated. Oracle time is excluded
from native measurements. Synthetic fixtures establish parity with the declared
F32 math/order control, not video quality or real checkpoint fidelity. The MPS
adapter keeps Q/K/V, attention and its output in F32. Pinned Magi attention
instead casts F32 Q/K/V to BF16 and converts the returned result to the original
dtype when necessary. The wrapper alone does not establish the kernel's output
dtype. This
first slice does not claim numeric parity with that CUDA BF16 runtime, even when
the source weights are stored as BF16. The new `encode_mps` control certifies the
declared MPS BF16 boundary policy, not full BF16 activation execution or CUDA
FlashAttention reductions. Normalization epsilon is fixed to upstream's `1e-6`.

## Rejected and guard-only surface

Full model/Studio/video generation remains disabled for this slice. Audio-driven
video, lip-sync, trained reference-image conditioning, diffusion scheduling,
PixelUMM decoding, general integrated text conditioning, camera/action modules,
cross-segment KV-cache behavior,
INT8/FP8 inference and model-wide performance are unimplemented boundaries.
Text tokenization and the selected 26-block encoder prefix now have a separate
opt-in, one-natural-prompt qualification in `qwen25-hy-nomas-frontier.md`.
The one-prompt, short-input refiner/DiT handoff is qualified above; general
pipeline integration remains unimplemented.
Public RTX 4090 timings are author-reported CUDA results, not Metal predictions.
No GPL community-port code is imported into this Apache-source implementation.

## Active slice: BF16 boundaries and checkpoint weights

Status: actual-weight F32 and BF16-boundary controls pass through the opt-in MPS
backend, 2026-10-24. The original native backend's BF16 failures remain open;
preserve its existing F32-control default.
An explicit opt-in BF16-boundary control rounds concatenated F32 Q/K/V to
BF16 with round-to-nearest-even, compute attention in F32, then round its output
to BF16 before the F32 projection. The output round-trip is an explicit control
assumption until the exact Magi kernel is executed or its contract is evidenced.
Fuse these conversions into existing packing
and splitting dispatches where possible. This emulates pinned Magi storage
boundaries; it does not emulate or certify CUDA FlashAttention's internal
reduction or BF16-activation execution throughout the block.

Falsifier: exact same-input boundary tests must expose a missing conversion or
incorrect tie rounding. Full-block intermediate comparisons retain the finite
`2e-5 + 2e-5*abs(reference)` gate. If upstream/native F32 differences cross a
BF16 rounding midpoint, retain that mismatch and isolate attention on identical
Q/K/V; do not increase tolerances or silently label the boundary path exact.

For real weights, enumerate the official distilled checkpoint at an immutable
Hugging Face revision and materialize only the 32 tensors of `double_blocks.0`.
Validate exact keys, shapes, source dtypes, byte ranges and source identity before
publishing a local subset. Keep F32 source weights and a GPU-converted BF16
residency control distinct. No full-checkpoint, full-model or decoder allocation
is admitted by this slice. Setup file extraction is not neural CPU compute.

The generic checkpoint planner retains its default rejection of F32 sources in
BF16-resident mode. An explicit `allow_f32_to_bf16` setup option may admit only
F32-to-BF16 conversion, with raw F32 staging included in the pre-allocation
budget. Conversion uses a GPU rounding kernel in the caller's existing encoder;
original BF16 tensors continue to copy directly to their final buffers. F16
sources remain rejected in BF16-resident mode even when the option is enabled.

DoD: guarded focused specs establish deterministic rounding on crafted values,
unchanged F32-control results, upstream comparisons for the boundary variant,
and actual one-block weights against the pinned class. Report any failed gate
as an open frontier. One terminal completion/readback per diagnostic variant,
zero encode-time device allocations, the 15% floor and existing process/swap
budgets remain mandatory. Rollback is removal of the new opt-in mode and probe;
the existing F32-control API remains the baseline.

## Verification plan and rollback

Risk: CAUTION (new inference implementation); isolated branch
`codex/hy-nomas-native`, based on Wan commit `e22fc2ba`.
Existing work and inference defaults remain untouched. Rollback is omission of
the new opt-in files; do not reset unrelated work.

1. Audit exact upstream block equations, conditioning and state-dict names.
   Consumers: new `src/ml/video/hy_nomas_block.cr`, its kernels and tests only.
   Inventory: `rg --files src/ml/video spec examples`; no Hy-Nomas implementation
   exists at the starting revision.
2. Add a pinned-source upstream fixture producer and native specs that first
   fail without the new block. Include nonzero modulation, both streams,
   invalid shapes/configurations and resident-weight types. Admit only one
   fully valid unmasked segment with no history, camera or action windows.
3. Implement only the new block and checkpoint bindings, reusing shared engine
   APIs. Preallocate diagnostic scratch. Keep an explicit unsupported-mode gate.
4. Run bounded upstream/native parity and repeated native encode checks under
   the existing cache-aware memory guard. Record every intermediate mismatch,
   terminal wait boundary, memory peak and scope of any timing claim.

DoD: generate bounded upstream fixtures on MPS; compile and run the focused
Crystal spec with fresh Metal/MPS bridge objects; check format and the exact diff.
Expected signal: no nonfinite values or numerical violations, strict negative
admission tests pass, and no model-path synchronization/readback/allocation.

The strongest failure modes are reordered text/video tokens or RoPE channels,
incorrect modulation/gates, BF16 dtype confusion and silently missing temporal
masks. Intermediate upstream
checkpoints and deliberate invalid-mode tests are the discriminators.

All runtime probes use a 15% physical-memory reserve/floor, cache-aware admission,
a bounded process-group footprint, a swap-growth stop and a deadline. Preserve
guard failure as an incomplete measurement, never evidence of acceleration.
Temporary fixtures, objects and logs belong under `/opt/mldata_ssd/cogni-ml/`.

The external oracle JSON is trusted test input. Capture its SHA-256 immediately
after trusted fresh generation and pass it separately in
`HY_NOMAS_FIXTURE_SHA256`. Do not take this digest from mutable metadata inside
the fixture. The native harness hashes one read of the JSON and parses those
same bytes; the supplied digest also binds expected tensors and sidecar hashes.
Checkpoint bytes remain subject to the immutable-file precondition above.

## Reproduction commands

Run from the feature checkout on macOS Metal. Use an existing PyTorch MPS
environment with safetensors; the environment used here is the external
DRAMA-reference venv. It is an oracle dependency, not part of native execution.
Choose a fresh `probe_root`: the oracle refuses to replace existing outputs.

```sh
probe_root=/opt/mldata_ssd/cogni-ml/hy-nomas-native-repro
probe_upstream=/opt/mldata_ssd/cogni-ml/video-scene-20261019/upstream/hy-nomas-20261024
probe_python=/opt/mldata_ssd/cogni-ml/drama3-fish-reference-20261015/venv/bin/python
mkdir -p "$probe_root"/build "$probe_root"/tmp "$probe_root"/crystal-cache "$probe_root"/fixtures "$probe_root"/logs

guarded() {
  probe_log=$1
  shift
  python3 examples/support/wan_memory_guard.py \
    --max-footprint-mib 4096 --max-swap-growth-mib 64 --timeout-sec 600 \
    --memory-accounting file-cache --log "$probe_root/logs/$probe_log.jsonl" -- "$@"
}

guarded oracle-tiny env TMPDIR="$probe_root/tmp" PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$probe_python" spec/support/hy_nomas_reference.py \
  --upstream "$probe_upstream" --output "$probe_root/fixtures/tiny.json"
guarded oracle-width env TMPDIR="$probe_root/tmp" PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$probe_python" spec/support/hy_nomas_reference.py \
  --upstream "$probe_upstream" --output "$probe_root/fixtures/width.json" --real-width
shasum -a 256 "$probe_root/fixtures/tiny.json" "$probe_root/fixtures/width.json"

guarded bridges make BUILD_DIR="$probe_root/build" \
  "$probe_root/build/bridge.o" "$probe_root/build/metal_mps_command_chain.o" "$probe_root/build/metal_stage_io.o" \
  "$probe_root/build/wan_t5_mps_bridge.o" "$probe_root/build/wan_t5_rms_mps_bridge.o" \
  "$probe_root/build/hy_nomas_mps_reductions.o"
probe_link="$probe_root/build/bridge.o $probe_root/build/metal_mps_command_chain.o $probe_root/build/metal_stage_io.o $probe_root/build/wan_t5_mps_bridge.o $probe_root/build/wan_t5_rms_mps_bridge.o $probe_root/build/hy_nomas_mps_reductions.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -lc++ -fuse-ld=/usr/bin/ld"
guarded build env TMPDIR="$probe_root/tmp" CRYSTAL_CACHE_DIR="$probe_root/crystal-cache" \
  crystal build spec/hy_nomas_block_spec.cr -o "$probe_root/build/hy_nomas_block_spec" \
  --error-trace --link-flags="$probe_link"
```

For each of `tiny` and `width`, set `probe_fixture_digest` to its corresponding
digest printed immediately after generation, then run:

```sh
probe_case=tiny
probe_fixture_digest=REPLACE_WITH_TRUSTED_GENERATION_DIGEST
guarded "native-$probe_case" env TMPDIR="$probe_root/tmp" \
  HY_NOMAS_BLOCK_FIXTURE="$probe_root/fixtures/$probe_case.json" \
  HY_NOMAS_FIXTURE_SHA256="$probe_fixture_digest" \
  HY_NOMAS_REPORT="$probe_root/logs/$probe_case-comparisons.txt" \
  "$probe_root/build/hy_nomas_block_spec"
crystal tool format --check src/ml/video/hy_nomas_block.cr spec/hy_nomas_block_spec.cr
git diff --check
```

Without an external fixture, the parity example is pending; a green local
unit-only run cannot establish upstream-control parity. Do not combine the two
GPU runs or run a model alongside them merely because each has its own guard.

## Historical synthetic result, commit 840c3a58

At the previous feature commit, the focused spec passed
**6 examples, 0 failures/errors/pending** for each
geometry. Both have 6 video and 3 text tokens. The broadcast case has B=2;
the per-token modulation case has B=1. They use synthetic nonzero weights,
not a downloaded checkpoint.

| Geometry | C / heads / head dim / FFN | Checkpoint comparisons | Compared values | Maximum absolute error | Maximum gate ratio |
|---|---|---:|---:|---:|---:|
| Tiny | 64 / 2 / 32 / 256 | 444 | 282,240 | 9.536743e-7 | 0.028442943 |
| Nomas-G widths | 2048 / 16 / 128 / 8192 | 444 | 9,031,680 | 5.9604645e-6 | 0.1444688 |

Each geometry compares 37 named intermediates in the first block, a second
application consuming the first block's resident outputs, and a seeded
video-perturbed application, for both F32 and BF16 weight storage in both
modulation cases. A repeated first encode verifies scratch overwrite/reuse.
Video perturbation must change both stream outputs; the block cannot pass as
an identity. Shape/mode/dtype/budget/alias gates, nonfinite/corruption detection,
and the trusted-fixture-digest negative control pass. The BF16 control rounds
weights to BF16, then uses F32 neural math in both implementations.

All four native encodes plus preparation share one command/encoder per variant,
with one terminal wait and subsequent diagnostic reads. The tracked encode
MetalBuffer allocation delta is zero. There is no inter-operation activation
readback or explicit completion wait in the admitted block. The native measured
path has no PyTorch dependency.

At Nomas-G widths the resident block weights occupy 604,227,584 bytes as F32 or
302,113,792 bytes as BF16. Three diagnostic scratch sets occupy 15,435,264 bytes
for broadcast or 8,922,624 bytes for per-token modulation. These numbers do not
include the rest of the model, encoder, decoder or video-length activations.

Diagnostic GPU command spans on Apple M2 Max were 58.516/40.279 ms for F32 and
154.233/116.229 ms for BF16 (broadcast/per-token). These include **four encodes
and weight preparation** and are single observations from a correctness build,
with only nine tokens per item. They are neither per-block steady-state
benchmarks nor video-speed estimates. The shared baseline linear kernel uses
a serial F32 K reduction; no MPS GEMM speedup has been established. BF16's
smaller residency does not imply a faster typed-load kernel in this probe.

Final guarded oracle runs sampled at most 3,527.395 MiB process-group footprint;
final native runs sampled at most 2,096.705 MiB. Minimum cache-aware headroom
was 25.385% for the oracle and 27.872% for native. All final runs exited zero
with no swap growth. The 4 GiB sampled cap, 15% floor, 64 MiB swap-growth stop
and 600 s deadline were retained. Sampled peaks can miss between-sample spikes.
No full model, Studio or video generation was run.

Correlated MiniQuad/adversary review is **ROBUST within the declared F32-control
scope**, assuming trusted fixture digests and immutable checkpoint files. It
does not certify Magi BF16 runtime parity, quality, full-model fit or speed.
The review caught the CUDA BF16 attention distinction and a configurable-epsilon
admission expansion; the latter now rejects non-default epsilon before
allocation. The fixture mutation control closes the inspected false-pass path.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-native-20261024/`.
Final receipts are `logs/oracle-{tiny,real-width}-guard-v2.jsonl`,
`logs/native-tiny-guard-v4.jsonl`, `logs/native-real-width-guard-v2.jsonl`,
`logs/tiny-comparisons-v4.txt` and `logs/real-width-comparisons-v2.txt`.
Build receipt: `logs/build-parity-guard-v5.jsonl`; source is this feature commit.
Toolchain: Crystal 1.22.0, Apple clang 21.0.0, macOS 26.6.2, SDK 27.0;
oracle PyTorch 2.8.0 MPS with fallback disabled. Format and exact-diff checks pass.

SHA-256 pins:

| Artifact relative to evidence root | SHA-256 |
|---|---|
| `fixtures/tiny-v2.json` | `51849dfabe68280ab186e4a13594261fbae45ae39d38a165cee7a133cc68b978` |
| `fixtures/real-width-v2.json` | `13b50759bb75784b40eaec75ca2b8952020807f64fc53a60b7b05ac5789acd71` |
| `build/hy_nomas_block_spec` | `92b6f085ed9bdc1dd8d7d4b4ea383320eba3e68e751d28f5f66716ba0361a7a5` |
| `build/bridge.o` | `16613762decfabc1d44d6615ef0af95986fdccafc07ad93d9f007c5440a055f3` |
| `build/metal_mps_command_chain.o` | `c2ba6e89f532848235e6710eb36e245a117665d6f6b5d0fbbba5656a2f5e0a86` |
| `build/metal_stage_io.o` | `7ee1def8e387edea8213c848441d40ac18ee9ec69f398423dce197d78244370a` |

The oracle's per-source and sidecar-weight pins are retained in each fixture's
manifest. Regenerate rather than edit reference arrays after a source change.

## Actual-checkpoint and boundary-control result, 2026-10-24

The official distilled safetensors file at Hugging Face revision
`bac719a80c8d02f91c485fca804bb8221e149b70`, path
`DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors`, stores F32 weights.
The extractor validates the pinned metadata, bounded complete header, exact
32-key block inventory, shapes/dtypes and contiguous nonoverlapping ranges.
It streams only the block's 604,227,584 payload bytes via an exact HTTP 206 range
`968270320-1572497903`, then publishes a fresh subset and provenance manifest.
There is no tensor decoding or CPU weight conversion during extraction.
The resulting block has 151,056,896 parameters. Its F32 residency is
604,227,584 bytes and GPU-rounded BF16 residency is 302,113,792 bytes.

Source authentication relies on HTTPS at the immutable repository revision.
The declared LFS source hash is
`73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70`;
the complete 34,213,991,280-byte source file was not downloaded or independently
hashed. The observed block payload hash is
`4711e53c3b9ab29abe552f25b95d3cea6f9d75df74e098d7a6acc880befae28b`.
The independently checked local subset-file digest is pinned below. These are
different scopes of evidence; a matching local digest does not verify the
unfetched parts of the upstream checkpoint.

The oracle loads these exact 32 tensors into the pinned upstream class at
C2048/H16/D128/F8192. Both cases still use 6 video and 3 text tokens; this is
real width and actual weights, not representative video sequence length.
F32 storage and GPU-rounded BF16 storage each use F32 neural arithmetic.

| Fixture / mode | Spec result | Comparison records | Failed records | Max absolute error | Max gate ratio |
|---|---|---:|---:|---:|---:|
| Tiny synthetic, F32 math | 8 examples, 0 failures/errors/pending | 488 | 0 | 9.536743e-7 | 0.028442943 |
| Actual block, F32 math | 8 examples, 0 failures/errors/pending | 488 | 0 | 2.4795532e-5 | 0.8565121 |
| Tiny synthetic, BF16 boundaries | 7 examples, 1 numerical failure | 488 | 7 | 0.00048828125 | 22.381378 |
| Actual block, BF16 boundaries | 8 examples, 1 numerical failure | 488 | 289 | 0.125 | 529.8862 |

Each fixture covers four variants: broadcast B=2 and per-token modulation B=1,
each with F32 and BF16 resident weights. Each variant compares 40 named
intermediates for the first, chained and video-perturbed applications, plus
raw and post-boundary attention replays on identical upstream Q/K/V:
`4 * (40 * 3 + 2) = 488`. A repeated first encode checks scratch overwrite.
The tiny BF16 result preceded the additional source-mode unit test; its
488 comparison records use the same gates. A failed comparison record may
contain many violating values; counts here are not counts of bad elements.

Both final F32-control fixtures pass and retain zero tracked encode-time
MetalBuffer allocations. Original BF16 sources retain direct-copy preparation;
actual F32 sources require 604,227,584 bytes of staging for either residency
control. The conversion, four block encodes and attention replay share one
diagnostic command with one terminal completion/readback per variant. No new
production inter-operation wait or readback is added. The diagnostic command
times include all this work and are not per-block benchmarks or video estimates.
The separate checkpoint suite passes 20 examples, including exact rounding ties,
mixed source residency, conversion chaining and pre-allocation budget failures.
The extractor's 11 offline tests and oracle structure/device self-tests pass.

### BF16 failure is retained

Actual-weight BF16-boundary failures per variant are 81/76/75/57 records
(broadcast BF16/F32 residency, then per-token BF16/F32 residency). The first
broadcast/BF16-resident application's upstream/native F32 projections,
normalization and RoPE pass the original gate, while the rounded joint Q/K/V
do not. One captured query value illustrates the discontinuity:

```text
native before round:    0.7675781   (F32 bits 3f448000)
upstream before round:  0.7675783   (F32 bits 3f448003)
native after round:     0.765625
upstream after round:   0.76953125
```

This is an F32 difference crossing a BF16 midpoint. The literal tie tests pass;
changing rounding or removing the boundary would conceal the tested behavior.
Same-input raw F32 attention replay passes all four actual-weight variants
(maximum absolute error 1.4305115e-6). After the output boundary, the
broadcast/F32-resident replay has one violating element, index 10120:
`-0.140625` versus `-0.13964844`, delta `0.0009765625`. The other three rounded
replays pass. Thus implementation numerical differences exist inside attention
as well as before Q/K/V packing; the replay alone does not identify their local
cause. The isolated diagnostics below narrow that attribution. Chained block
and final output gates also fail; these cannot be dismissed as harmless tiny
intermediate differences. No generated
video or quality impact has been tested. BF16-boundary mode stays experimental
and default-off; no Magi/CUDA runtime parity has been established.

### Memory, provenance and evidence

Two early actual-weight oracle attempts hit the unchanged sampled 4 GiB cap
(4106.488 and 4153.456 MiB, guard exit 76); neither is a successful measurement.
Meta initialization and per-parameter GPU BF16 rounding remove duplicate retained
weight copies. A third attempt exposed the `mps` versus `mps:0` device guard
distinction and exited 1; the corrected guard accepts an unspecified MPS index
but rejects CPU or a mismatched explicit index. Its device self-test uses no GPU.

Final oracle F32/BF16-boundary sampled peaks are 3074.222/3508.597 MiB, with
minimum cache-aware headroom 27.328%/25.205%. Final native actual-weight
F32/BF16-boundary peaks are 2789.034/2772.065 MiB, with minimum headroom
23.157%/24.391%. All have zero observed swap growth. The native BF16 run exits 1
for its numerical gate, not for a memory guard stop. The 15% reserve/floor,
4 GiB sampled cap, 64 MiB swap-growth stop and 600 s deadline remain unchanged.
Sampling can miss between-sample peaks. No full model, Studio or video runs.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024/`.
Final receipts: `logs/oracle-real-f32-v4.jsonl`,
`logs/oracle-real-bf16-v1.jsonl`, `logs/native-{tiny,real}-f32-v6.jsonl`,
`logs/native-real-bf16-v6.jsonl`, `logs/native-tiny-bf16-v5.jsonl`,
their matching `*-comparisons.txt` reports and `logs/checkpoint-v3.jsonl`.
Extraction: `logs/extract-block0-v2.jsonl`. Final build receipts:
`logs/build-hy-v6.jsonl`, `logs/build-checkpoint-v3.jsonl`.
The previous evidence root's three bridge objects are reused unchanged with
the SHA-256 pins in the historical table above. Toolchain and device are unchanged.

| Artifact relative to current evidence root | SHA-256 |
|---|---|
| `weights/distilled-block0-f32.safetensors` | `6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1` |
| `fixtures/tiny-f32-v2.json` | `6c2a31852f2e39a7f20017040dcd54ccbbb6043d80b417bf90099bbe7fda4afa` |
| `fixtures/tiny-bf16-v1.json` | `3e1b9d014d02d71a1b38d4a9ea4eccd1a8f28f3824500d4f6f7ac691f67de0d6` |
| `fixtures/real-f32-v4.json` | `f889e7b7726285e8baf0054a7f23652d7004351ef12fd9ab23a72e79a2596f25` |
| `fixtures/real-bf16-v1.json` | `1fb14da66820ca061a87ec28024881de3165301d3993adb944cb45f275bc72e1` |
| `build/hy_nomas_block_spec` | `512f92dca5f75f9049851dfa33e3aef6576eeba63a871549c072e2d310214a6e` |
| `build/video_checkpoint_spec` | `ae8c2b1518814e763fa391e2b20626a7066b34af2d4200d6b1e21591327bf6bf` |

Correlated MiniQuad/adversary review is ROBUST for actual-checkpoint loading
and F32-math control under the trusted producer/digest and immutable-file
preconditions. It is not an independent CUDA comparison. The BF16-boundary
parity gate remains failed. The review added an explicit source-mode gate:
actual-weight runs must supply `HY_NOMAS_CHECKPOINT_SHA256` separately from
`HY_NOMAS_FIXTURE_SHA256`; a generic synthetic fixture cannot satisfy it.
Metadata presence alone does not establish provenance.

### Reproduce the actual-weight slice

Reuse the `guarded` function, bridge build and link flags above, but use a fresh
directory **under** `/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024/` for
`probe_root` (the extractor deliberately restricts its output to that root).
Create its `weights`, `fixtures`, `logs`, `tmp`, `build` and `crystal-cache`
directories. Do not overwrite existing fixtures or receipts.

```sh
guarded extract python3 spec/support/hy_nomas_checkpoint_subset.py \
  "$probe_root/weights/block0.safetensors" --max-payload-bytes 604227584
shasum -a 256 "$probe_root/weights/block0.safetensors"
probe_checkpoint_digest=REPLACE_WITH_INDEPENDENTLY_CHECKED_SUBSET_DIGEST

guarded oracle-actual-f32 env TMPDIR="$probe_root/tmp" PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$probe_python" spec/support/hy_nomas_reference.py \
  --upstream "$probe_upstream" --output "$probe_root/fixtures/actual-f32.json" \
  --real-width --weights "$probe_root/weights/block0.safetensors" \
  --weights-sha256 "$probe_checkpoint_digest" --stage-telemetry
shasum -a 256 "$probe_root/fixtures/actual-f32.json"
probe_fixture_digest=REPLACE_WITH_TRUSTED_GENERATION_DIGEST
guarded native-actual-f32 env TMPDIR="$probe_root/tmp" \
  HY_NOMAS_BLOCK_FIXTURE="$probe_root/fixtures/actual-f32.json" \
  HY_NOMAS_FIXTURE_SHA256="$probe_fixture_digest" \
  HY_NOMAS_CHECKPOINT_SHA256="$probe_checkpoint_digest" \
  HY_NOMAS_REPORT="$probe_root/logs/actual-f32-comparisons.txt" \
  "$probe_root/build/hy_nomas_block_spec"

guarded build-checkpoint env TMPDIR="$probe_root/tmp" CRYSTAL_CACHE_DIR="$probe_root/crystal-cache" \
  crystal build spec/video_checkpoint_spec.cr -o "$probe_root/build/video_checkpoint_spec" \
  --error-trace --link-flags="$probe_link"
guarded checkpoint env TMPDIR="$probe_root/tmp" "$probe_root/build/video_checkpoint_spec"
PYTHONDONTWRITEBYTECODE=1 python3 spec/hy_nomas_checkpoint_subset_test.py
PYTHONDONTWRITEBYTECODE=1 "$probe_python" spec/support/hy_nomas_reference.py --structure-self-test
PYTHONDONTWRITEBYTECODE=1 "$probe_python" spec/support/hy_nomas_reference.py --device-match-self-test
```

Generate a separate fresh oracle fixture with
`--attention-precision bf16-boundaries` and repeat the native invocation with
that fixture's independently captured digest to reproduce the failing gate.
An exit 1 must remain a failure; do not ignore it in automation or treat the
successful F32 control as certification of this precision mode.

## Pilot-frame performance gate, 2026-10-24

No end-to-end video speed has been measured. The native implementation is one
dual-stream block, not the full sampler/text encoder/PixelUMM pipeline. Its
nine-token correctness fixture must not be extrapolated to video throughput.

At pinned upstream revision `911cea6458a89c2663513ff7099561ec190dd491`,
`configs/experiment/inference.yaml` selects 8 Euler steps, 54 DiT blocks and
guidance scales of 1.0. The default final segment has one DiT branch per step
and no cache-write forward (`hymm/pipeline/causal_ar.py`, guidance selection,
denoise loop and `should_cache_chunk`). A non-final segment or preserved cache
adds one whole-stack cache-write pass. A 24-frame, 24-fps prefix uses 6 latent
frames and, with the preset PixelUMM grid, 22 by 40 spatial tokens: 5,280 video
plus 1,280 text tokens. This grid decodes to 704 by 1280; verify actual output
dimensions rather than labeling the result 720p.

For the user's ceiling of 120 wall seconds per video second:

```text
T_other + sum(block-call times) <= 120 seconds
mean block-call time <= (120 - T_other) / (8 * 54)
```

The zero-overhead ceiling is 277.8 ms per block, or 246.9 ms with the ninth
cache-write pass. `T_other` includes conditioning, scheduler, stack overhead,
PixelUMM decoding and output handling. Exceeding this ceiling at representative
geometry falsifies the target; passing one block is only a necessary condition.
Cold request-to-first-pixel and warmed prefix time need separate measurements.

With J=6,560, D=2,048 and FF=8,192, dense projection arithmetic is
`2*J*(4*D^2 + 2*D*FF)`, plus two batch-row modulation projections
`4*B*D*(6*D)` and QK/PV `4*J^2*D`: 1.012983 TFLOP per block. The 432 calls
therefore need at least 3.647 TFLOP/s sustained if the whole 120-second budget
were available to this arithmetic; conditioning and decoding make the actual
requirement higher. This is a necessary throughput bound, not a speed result.

The current named-checkpoint scratch formula requires 1.682 GiB at this
geometry with the default broadcast modulation, or about 2.03 GiB in the
separate per-token mode, before inputs, weights and opaque runtime workspace.
Default conditioning already projects one row per stream and broadcasts it;
there is no default per-token modulation GEMM redundancy to remove. The 54
BF16 block weight sets alone require about 15.19 GiB, excluding embeddings and
other models. Independent MPS workspaces additionally retain about 404.3 MiB of
conversion/attention/RMS stages each, or about 21.32 GiB over 54 blocks. Independent
named scratch sets would add 90.84 GiB. Both scratch lifetime reuse and shared
serial stages are prerequisites for a whole-stack implementation; shared
staging alone is insufficient. Reusing the expansion stage saves allocation
space but still performs the conversion for each projection call.

A target-shape probe needs a
cache-aware residency preflight and compact attention/scratch before launch;
the existing 4 GiB probe cap and 15% host reserve remain in force. A full pilot
also requires the missing pipeline components. The current request excludes
full generation; a future pilot must be a separately scoped run.

The opt-in MPS backend now replaces scalar-output projection reductions and the
serial per-key attention loop. Its nine-token diagnostic command is still not a
video benchmark. Next candidates are compact grouped/chunked MPS attention,
shared serial weight/RMS stages and liveness-based scratch reuse. At J=6,560,
one F32 `[B,H,J,J]` score-sized tensor is 2.565 GiB; the graph's opaque peak can
exceed this, so a complete dense attention graph plus full block scratch is not
admitted under the 4 GiB probe cap. Reducing this workspace is needed before a
representative full-block measurement. Any attention grouping/chunking must
repeat the same-input and chained BF16 gates: a different graph shape may select
different reductions even though heads are mathematically independent.
Remove repeated conditioning work only where the upstream input invariant proves
the rows are identical. A shared conversion stage saves capacity but still
expands each weight matrix: fourteen linears write about 576 MiB of F32 staging
per block call. This is another candidate after target-shape phase timings.
Quantization or fewer diffusion steps are separate quality-changing experiments;
they are not implied by the present unchanged-tolerance arithmetic work.

## MPS reduction intervention frontier, 2026-10-24 (initial falsifier)

The initial falsifier was a native MPS projection/QK/softmax/PV/SDPA control on the
same trusted frozen operands. The existing failing report remains the anchor:
raw arithmetic passes, but small reduction differences can straddle BF16
midpoints. The intervention must retain the original native report and failed
gates, use unchanged finite and bitwise rounded gates, preserve input identities,
allocate plans and scratch before encoding, and use one terminal wait per case.
The completed intervention is recorded below; no default, weight policy or
full-generation scope changes.

Risk is CAUTION: changing graph rank, layout or materialized targets may select
different MPS arithmetic. Standalone QK and frozen-score softmax must therefore
remain distinct from internal SDPA intermediates. The custom online recurrence
and prescaled paths remain explicitly identified controls. Rollback is omission
of the opt-in test controls. DoD is a fresh guarded tiny and actual-weight
comparison report with per-operation outcomes, unchanged native controls,
zero tracked encode buffer allocations, one terminal wait, and cache-aware
memory remaining above 15%. A parity claim is limited to the operation/case
inventory that actually passes; the diagnostic's original failing exit is
retained until all its comparison records pass.

## Same-input attention diagnostics, 2026-10-24

This test-only slice freezes operands from the preceding trusted upstream
fixture. It separates local arithmetic differences from propagation through
earlier operations. It changes no production kernel, rounding rule, tolerance,
default or command schedule. Risk is CAUTION because an incorrect intervention
could misattribute the failed BF16 gate. The guard is exact operand lineage,
bitwise reference reproduction and a factored-native control. Rollback is
omission of `spec/hy_nomas_attention_probe_spec.cr` and its two support files.

The producer invokes pinned upstream RMSNorm/RoPE and PyTorch 2.8.0 MPS math;
PyTorch is only the oracle. Both oracle and native neural operations run on GPU.
The actual-weight geometry remains C2048/H16/D128/F8192 with **6 video and 3
text tokens**, not a representative video length. Each fixture has the same
four broadcast/per-token and F32/BF16-resident weight variants. BF16 residency
means source F32 to BF16 to F32 compute, with conversions on GPU.

Each variant contains 47 isolated operations: biased, unbiased and explicit-bias
versions of eight projections; Q/K RMSNorm and video RoPE; QK, softmax and PV
under three scaling placements; canonical/explicit MPS attention controls;
and frozen-score/probability interventions. Frozen Q/K/V must be exactly the
BF16-RNE round-trip of the original concatenated upstream operands. Layouts,
scales, weight policy, source mode and independent fixture/checkpoint digests
are checked before GPU work. Corruption controls reject malformed tensors,
changed digests, incorrect layouts/scales, missing variants, source-mode
substitution and broken boundary lineage.

All operations and GPU rounding for a variant share one command with **one
terminal wait**. Readbacks occur after that wait. Inputs remain bitwise unchanged
and encode performs zero tracked MetalBuffer allocations. A factored native
QK plus online reduction is bitwise identical to unchanged production attention
in all four actual and all four tiny variants, validating this intervention
within the tested geometry. These are diagnostic command times, not performance
benchmarks.

### Numerical result and causal limits

The diagnostic DoD is a complete, trusted same-input report with unchanged
finite gates, faithful native controls and retained failing BF16 records. It
does **not** require or establish BF16 parity. Five unit controls pass; the
fixture example completes and intentionally exits 1 for numerical failures.

| Actual-weight check | Observed result |
|---|---|
| Raw F32 comparisons | 188/188 pass; max absolute error `2.861023e-6`, max gate ratio `0.0698829` |
| Reproduction of preceding upstream captures | 68/68 records bitwise identical |
| Factored native versus production attention | 4/4 records bitwise identical |
| Rounded diagnostic records | 101/188 fail bitwise equality; 83 also fail the original finite gate |
| Canonical attention output, after BF16 rounding | Two differing elements across four variants; one fails the finite gate |

The bitwise BF16 diagnostic is stricter than the unchanged finite gate. The
101 records include alternative arithmetic paths and counterfactual rounding
sites; they are **not 101 unique live model-boundary failures**. Tiny controls
have 13/188 failing rounded records. No failed gate is waived.
The 188-record counts cover the 47 named operations per variant. Eight additional
raw/rounded report rows compare the factored and frozen-QK interventions with
canonical attention; those overlapping controls are not included in that count.

On identical projection inputs, upstream `linear(x,w,b)` and
`linear(x,w,None)+b` are bitwise equal across all 32 base projection slots
(442,368 values). Native no-bias projections still differ in low F32 bits,
consistent with the different native FMA-lane reduction and MPS GEMM arithmetic;
bias placement does not explain the tested drift. Among normal projections,
28 V values cross a BF16 bin at the actual Q/K/V boundary, 14 exceeding the
finite post-rounding gate. Other projection rounding probes occur before
Q/K RMSNorm/RoPE or after attention and are counterfactual boundaries.

Same-input final Q/K RMSNorm/RoPE probes have zero BF16 bin differences across
16 records. This does not clear chained Q/K: incoming projection differences
are deliberately removed by these frozen-input tests and can propagate in a
complete block.

Two interventions distinguish the canonical attention failures:

- **Broadcast/F32, index 10120:** native raw output is exactly
  `-0.14013671875`, a BF16 midpoint, and rounds to `-0.140625`. Canonical MPS
  gives approximately `-0.14013669`, rounding to `-0.13964844`. Replacing only
  QK scores with frozen explicit-MPS postscale scores in the same native online
  reducer restores the canonical BF16 value. The entire frozen-QK variant has
  zero BF16 differences from canonical MPS here. QK arithmetic is causally
  sufficient to explain this point under these inputs; RNE itself is correct.
- **Broadcast/BF16 weights, index 9949:** freezing MPS QK does not restore the
  canonical BF16 value. Native PV using probabilities returned by the actual
  MPS math-SDPA operator does restore it. This implicates arithmetic between
  scores and output in addition to QK. It does not isolate softmax alone:
  the private MPS probability control includes its internal QK and softmax.
  Frozen-QK online reduction also introduces other BF16 mismatches in this
  variant, so it is not a general replacement or fix.

Canonical SDPA, explicit postscale attention and private MPS math-SDPA expected
outputs are bitwise equal for all four variants (110,592 values per path).
The standalone frozen-score online reference is a custom recurrence control,
not the canonical upstream attention algorithm. No CUDA/Magi execution,
quality comparison, production fix or speedup is claimed.

Correlated MiniQuad source/fixture/report review is ROBUST for this bounded
diagnostic under the trusted-producer/digest and immutable-weight assumptions,
and VULNERABLE as evidence for a general BF16 fix. Cross-path expected-output
equality was independently checked from the pinned fixture during that review;
the compact report verifier below checks report completion and control counts,
not that additional fixture-array comparison.

The next discriminator is to compare the native projection/QK reductions with
the corresponding MPS GEMM/math path, then isolate normalization and PV with
frozen actual MPS probabilities. Preserve the original F32 control and failed
BF16 gates when evaluating a candidate change.

### Safety and pinned evidence

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-attention-20261024/`.
The same 15% physical reserve/floor, cache-aware file-cache accounting, sampled
4 GiB process-group cap, 64 MiB swap-growth stop and 600 s deadline are retained.
Jobs ran serially, with no full model, Studio or video generation.

| Guard receipt under `logs/` | Sampled peak MiB | Minimum cache-aware headroom | Swap growth MiB | Exit |
|---|---:|---:|---:|---:|
| `oracle-tiny-v1.jsonl` | 229.033 | 36.184% | 0 | 0 |
| `oracle-real-v1.jsonl` | 1752.237 | 30.992% | 0 | 0 |
| `native-tiny-v4.jsonl` | 19.954 | 35.091% | 0 | 1, numerical gate |
| `native-real-v2.jsonl` | 2738.659 | 27.507% | 0 | 1, numerical gate |

Sampling can miss between-sample peaks. Source state is parent `bd0bf23a` plus
the three test files in this slice. Toolchain/device and the three bridge
objects remain unchanged from the historical controls above. The final actual
report uses the final native binary; the tiny v4 report precedes only cosmetic
report-field renaming. `logs/native-controls-v4.jsonl` records five passing unit
controls and one pending GPU example with no external fixture.

| Artifact relative to evidence root | SHA-256 |
|---|---|
| `fixtures/tiny-v1.json` | `736864935ca2abc91d69609a03a823a32b7beb55ff2c09c9a95ea7d34982f6ea` |
| `fixtures/real-v1.json` | `481450208e45da4e648538c30855e5919538485e2eacf80e92773dddcc27d937` |
| `build/hy_nomas_attention_probe_spec` | `69c5be3d8597fd61c3f577b1a83662f9f7490c20148098f9a8b6754e6046cec2` |
| `logs/real-v2-comparisons.jsonl` | `46cee80de912bd891c3a645945bfe8f58cf7ff6017fce576b223a4209ece1086` |
| `logs/verification-v1.json` | `0547b5b5eb9eb9db000ab4e5b4d8b5f809a54c3d4987e36dff30e2b98a86be92` |

Actual source fixture/checkpoint pins are the preceding `real-bf16-v1.json`
and actual F32 subset digests, respectively. The tiny source fixture digest is
`3e1b9d014d02d71a1b38d4a9ea4eccd1a8f28f3824500d4f6f7ac691f67de0d6`;
its F32 weight-sidecar digest is
`c16f797818810ae066ed62733813a2adc36f1868b1edf6cc4f24f16f5ee37102`.

### Reproduce the diagnostic slice

Reuse the `guarded` function and pinned upstream/Python paths above, with a
fresh `probe_root` under `/opt/mldata_ssd/cogni-ml/` and its usual subdirectories.
Use freshly built bridges and `probe_link` from the build recipe above, or the
unchanged, digest-checked historical objects. Never run GPU jobs concurrently.

```sh
guarded build-attention env TMPDIR="$probe_root/tmp" CRYSTAL_CACHE_DIR="$probe_root/crystal-cache" \
  crystal build spec/hy_nomas_attention_probe_spec.cr \
  -o "$probe_root/build/hy_nomas_attention_probe_spec" --error-trace --link-flags="$probe_link"
PYTHONDONTWRITEBYTECODE=1 "$probe_python" spec/support/hy_nomas_attention_reference.py --structure-self-test
guarded attention-unit-controls env TMPDIR="$probe_root/tmp" "$probe_root/build/hy_nomas_attention_probe_spec"

probe_source_root=/opt/mldata_ssd/cogni-ml/hy-nomas-bf16-20261024
probe_source_digest=1fb14da66820ca061a87ec28024881de3165301d3993adb944cb45f275bc72e1
probe_checkpoint_digest=6810bf7cd7f9a8ac4d66240eee514637a7ccaa47dc9de02e8bc26ae4543530d1
guarded attention-oracle-actual env TMPDIR="$probe_root/tmp" PYTHONDONTWRITEBYTECODE=1 \
  PYTORCH_ENABLE_MPS_FALLBACK=0 "$probe_python" spec/support/hy_nomas_attention_reference.py \
  --fixture "$probe_source_root/fixtures/real-bf16-v1.json" --fixture-sha256 "$probe_source_digest" \
  --weights "$probe_source_root/weights/distilled-block0-f32.safetensors" \
  --weights-sha256 "$probe_checkpoint_digest" --upstream "$probe_upstream" \
  --output "$probe_root/fixtures/attention-actual.json"
shasum -a 256 "$probe_root/fixtures/attention-actual.json"
probe_fixture_digest=REPLACE_WITH_TRUSTED_GENERATION_DIGEST
guarded attention-native-actual env TMPDIR="$probe_root/tmp" \
  HY_NOMAS_ATTENTION_SOURCE=actual \
  HY_NOMAS_ATTENTION_FIXTURE="$probe_root/fixtures/attention-actual.json" \
  HY_NOMAS_ATTENTION_FIXTURE_SHA256="$probe_fixture_digest" \
  HY_NOMAS_ATTENTION_WEIGHTS="$probe_source_root/weights/distilled-block0-f32.safetensors" \
  HY_NOMAS_ATTENTION_WEIGHTS_SHA256="$probe_checkpoint_digest" \
  HY_NOMAS_ATTENTION_REPORT="$probe_root/logs/attention-actual-comparisons.jsonl" \
  "$probe_root/build/hy_nomas_attention_probe_spec"
```

Expected native exit is **1**, with four summaries and a final
`status=numerical_gate_failed` record. Missing completion, runtime/guard failure
or changed provenance is not an acceptable diagnostic result. Do not mask the
exit in parity automation. This separate diagnostic check verifies completion
and the scoped result; it does not turn failed BF16 parity into a pass:

```sh
python3 - "$probe_root/logs/attention-actual-comparisons.jsonl" \
  "$probe_fixture_digest" "$probe_checkpoint_digest" "$probe_source_digest" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
manifest, complete = rows[0], rows[-1]
assert manifest['phase'] == 'manifest' and manifest['source_mode'] == 'actual'
assert manifest['fixture_sha256'] == sys.argv[2]
assert manifest['checkpoint_sha256'] == sys.argv[3]
assert manifest['source_fixture_sha256'] == sys.argv[4]
assert (manifest['gate_atol'], manifest['gate_rtol']) == (2e-5, 2e-5)
assert manifest['rounded_gate'] == 'bitwise' and not manifest['production_changes']
raw = [r for r in rows if r['phase'] == 'raw' and 'kind' in r]
rounded = [r for r in rows if r['phase'] == 'rounded' and 'kind' in r]
assert len(raw) == len(rounded) == 188
assert all(r['stats']['violations'] == 0 for r in raw)
assert sum(r['stats']['bitwise_mismatches'] > 0 for r in rounded) == 101
assert sum(r['stats']['violations'] > 0 for r in rounded) == 83
for phase, count in [('reference_reproduction', 68), ('bitwise_control', 4)]:
    controls = [r for r in rows if r['phase'] == phase]
    assert len(controls) == count
    assert all(r['stats']['bitwise_mismatches'] == 0 for r in controls)
summaries = [r for r in rows if r['phase'] == 'summary']
assert len(summaries) == 4
assert all(r['operations'] == 47 and r['terminal_waits'] == 1 and
           r['encode_metal_buffer_allocations'] == 0 for r in summaries)
assert complete == dict(phase='complete', cases=4,
                        failed_comparison_records=101, status='numerical_gate_failed')
print('Diagnostic complete; raw F32 controls pass; BF16 parity remains failed')
PY
```

For the tiny replay, substitute its source fixture/sidecar and trusted digests
above, use fresh output paths and set `HY_NOMAS_ATTENTION_SOURCE=synthetic`.

Refresh this diagnosis after changes to operands/checkpoint, upstream or private
MPS operator semantics, geometry/dtype, native arithmetic, producer/validator,
compiler, bridge, runtime/device or command scheduling. Artifact loss requires
regeneration rather than trusting the summary counts.

## Prepared MPS block result, 2026-10-24

The new `HyNomasDualStreamBlock#encode_mps` is an explicit opt-in backend. It
keeps activations and BF16/F32 weights resident, reuses setup-time plans and
scratch, and preserves the existing native encoder and its recorded failures.
The measured path is Crystal/Metal/MPSGraph; PyTorch is only the pinned fixture
oracle. Risk remains CAUTION because graph rank, layout, constants and operation
order affect reduction arithmetic. Rollback is to omit `hy_nomas_block_mps.cr`
and its explicit backend; no default or tolerance was changed.

### Causal controls and implementation

Same-input MPS projection and QK controls remove the native reduction drift:
96 raw projection records and 12 raw QK records, plus their corresponding
rounded records, are bitwise equal to the declared MPS references on actual
weights. Sixteen raw softmax and sixteen raw PV controls also match bitwise,
as do their rounded records. Canonical MPS
attention matches its oracle. Alternative scaling placements are retained as
separate experiments; they are not promoted merely because their raw F32 error
fits the finite gate. The original diagnostic still exits 1 on known native and
alternative-path BF16 failures. This expected diagnostic failure is not a failed
test of the new opt-in full block.

Matching projections/attention alone did not close chained BF16 parity. Tiny
upstream/native F32 differences before a BF16 midpoint altered later bins. Each
intervention first ran on identical frozen operands; only the matching variant
was then inserted into the block:

- Non-affine LayerNorm follows the pinned PyTorch 2.8 single-row reduction and
  variance arithmetic; eight frozen raw comparisons are bitwise equal.
- Modulation and gated residual preserve separate products/additions instead of
  FMA contraction; sixteen frozen comparisons are bitwise equal.
- SiLU and tanh-GELU use setup-prepared MPSGraph plans with the pinned PyTorch
  operation order, constant rank and graph shape. SiLU has four same-input
  bitwise controls, repeated on one chain; eight frozen GELU controls match.
- Q/K RMS uses the existing native square / MPSGraph mean / native normalization
  path with fixed scratch; sixteen raw/rounded controls match. Video RoPE keeps
  adjacent-pair products/additions separate; eight raw/rounded controls match.

Prepared F32 projection stages accept BF16 resident weights via GPU conversion.
There are fourteen fixed linear bindings per block. The MPS workspace owns its
weight, head-major attention and RMS scratch; it is strongly rooted until an
explicit release after the registered chain drains. Projection hooks require
block preflight and obtain the encoder from that exact chain. Rejected early
release, direct projection before preflight and RoPE after release are negative
checks. A separate command-chain fix prevents a rejected pre-wait call from
being mistaken for completed GPU work (`345378c0`).

MPSGraph encodes use nonblocking execution descriptors. Encoder values are
refreshed after graph pauses/resumes. The runtime gate asserts zero tracked
encode-time `MetalBuffer` allocations and exactly one terminal wait per variant;
all diagnostic reads occur afterward. This does not claim zero opaque MPSGraph
allocation or zero internal driver/queue backpressure.

### Final verification and evidence

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-mps-reductions-20261024/`.
The block binary is `build/hy_nomas_block_spec-v11`, compiled from the committed
Crystal source with the fresh command-chain, Hy-Nomas reductions/activation and
Wan T5 linear/RMS bridge objects. Core bridge/stage objects are from the earlier
`hy-nomas-native-20261024/build/` root. The current source/bridge revisions, trusted
checkpoint and fixture digests must be retained together.

Both the actual C2048/H16/D128/FF8192 checkpoint and the tiny configuration pass
the BF16-boundary fixture: eight examples each, four cases (broadcast/per-token
modulation crossed with BF16/F32 weight residency), 536 finite comparisons per
fixture, no violations. Each case encodes a first block, a second block consuming
its resident output, a repeat of the first bindings and perturbed video input.
For actual weights all 480 saved first/chained/perturbed checkpoint arrays are
raw-F32 bitwise equal; all 72 selected BF16 boundary arrays have zero bit
mismatches. This is evidence for these fixtures, not a universal bitwise promise.
Receipts: `logs/block-{real,tiny}-v11-guard.jsonl` and corresponding
`-comparisons.txt` reports. The unchanged finite gate is still
`2e-5 + 2e-5*abs(reference)` and the BF16 boundary gate is still bitwise.

Frozen RMS/RoPE evidence is in `logs/attention-real-v5-comparisons.jsonl`;
same-input projection/QK evidence is in `logs/{tiny,real}-v1-comparisons.jsonl`.
The SiLU reference producer validates source fixture identity and extracts only
the four frozen modulation inputs; it does not load a checkpoint or instantiate
a model. Its references are `pointwise-{tiny,real}-v1.json` (SHA-256 below).
Native same-input SiLU controls and serial plan reuse are in
`logs/pointwise-real-v6-comparisons.jsonl` and its guard receipt.

```text
pointwise-tiny-v1.json d8a9070b87ca4f99ca47ef7695514fc3d9ea6fe567c374b3e05211dab70be2a1
pointwise-real-v1.json 8184c4c0e6e3293d0ed0b947f092c88d9c4fb08ef2834ffe87fe13ed774b90af
```

To replay the completed MPS block slice, use the guarded build/link commands
above including `wan_t5_rms_mps_bridge.o`. Run each trusted F32 or BF16-boundary
fixture serially with `HY_NOMAS_BLOCK_MPS=1`,
`HY_NOMAS_NORM_CONTROLS=1`, `HY_NOMAS_MODULATION_CONTROLS=1`, and
`HY_NOMAS_ACTIVATION_CONTROLS=1`, plus the independently pinned fixture/checkpoint
digests and a fresh `HY_NOMAS_REPORT`. Native-default regression uses
`HY_NOMAS_BLOCK_MPS=0 HY_NOMAS_ACTIVATION_CONTROLS=0` on an F32 fixture. Setting
activation graph controls without the MPS chain is rejected. The same-input
attention diagnostic retains its expected failing exit and original records.

Verdict: correlated MiniQuad ROBUST for the supported `encode_mps` fixture
contract, binding/lifetime guards and successful-path synchronization. This
does not certify CUDA/Magi output dtype or FlashAttention reduction, all-BF16
activations, camera/history/streaming, video quality, target-shape performance or
the full pipeline. Refresh after scoped source, operand/checkpoint, graph rank or
layout, precision policy, bridge/compiler, PyTorch/MPS/OS/device or scheduling
changes. Missing evidence requires reproduction; a new failure reopens this gate.

The final F32 fixtures also pass through `encode_mps`: eight examples and 536
finite comparisons each for actual and tiny weights. All 480 saved checkpoint
arrays per fixture are raw-F32 bitwise equal. The original native-default F32
actual-weight regression still passes eight examples and 488 comparisons under
the same finite gate. Reports/receipts are
`logs/block-{real,tiny}-f32-v11-*` and `logs/block-native-f32-v11-*`.
The producer's offline structure self-test, Crystal format checks and exact-diff
whitespace check pass. The one final formatting-only change to
`spec/support/hy_nomas_modulation_controls.cr` does not alter its operations.

The final five block runs completed with zero swap growth, sampled process-group
footprints below the 4 GiB cap (maximum 3541.144 MiB) and minimum cache-aware host
headroom of 24.532%, above the unchanged 15% floor. Sampling cannot exclude
between-poll peaks; file-backed cache is an estimate of reclaimable memory,
not guaranteed immediately free pages. Build/probe receipts retain these guard
limitations. macOS 26.6.2, Apple M2 Max, Crystal 1.22.1, PyTorch 2.8.0 oracle.
No full checkpoint/model/pipeline was allocated. The nine-token command timings
include repeated blocks and diagnostics with different native/MPS controls;
they are not paired block throughput measurements or evidence for video speed.

## Compact attention slice, 2026-10-24 (explicit opt-in)

`HyNomasCompactAttentionMPSPlan` and its native bridge pack K/V once per encode
into fixed BHTD buffers. Fixed smaller Q/output stages are reused serially. Each
chunk runs the same rank-4 QK -> epsilon -> scale -> softmax -> PV graph. All
bindings, offset views, graphs and execution descriptors are prepared before
encoding. The canonical dense plan remains the default/control; rollback is to
omit the new workspace keywords (`attention_query_chunk` and
`attention_heads_per_group`, both zero by default).

Every query graph has exactly C rows. The last window begins at `T-C` when a
short tail would otherwise remain: independent attention rows are recomputed
and overwritten on the GPU. This preserves the prepared graph shape, at the
cost of at most C-1 redundant rows. No mask, causal, camera or history semantics
are introduced. MPS command roots are refreshed after graph encoding before
subsequent native gather/scatter work. Explicit release requires the latest
chain to be drained; abandoned compact plans deliberately stay strongly
retained until release/process exit.

The workspace rejects known reduction counterexamples before device allocation:
G must equal H; partial chunks require B=1 and C>=32. These are guards against
observed failures, not a proof for every admitted dimension or activation.
The general standalone constructor retains grouped/multi-batch/short-chunk
variants for diagnostics only. No compact default is promoted.

### Numerical falsifiers and measured scope

Changing graph shape can change MPS reduction order. G4 at T129/T257 and C4 at
T9 have BF16-RNE counterexamples under the unchanged gate. Adding a B2/T65/H16/
D128/C32/G16 test found another: max F32 error `5.9604645e-8`, but 17 BF16 bins
differ. That partial multi-batch case is now rejected by workspace admission.
Its failed receipt is retained as `logs/compact-overlap-v6-f32-*`.
`HY_NOMAS_COMPACT_GROUP_DIAGNOSTIC=1` explicitly reruns these counterexamples;
this mode must fail the original gates, not become a passing qualification.

Final standalone controls use full-index avalanche inputs rather than a short
periodic pattern, changed GPU-resident Q/K/V, repeated encodes in one chain,
a second completed-chain reuse, end canaries, binding/alias/size preflight and
premature-release rejection. F32 and BF16-rounded inputs each pass 19 examples
and 48 dense-versus-compact comparisons under the unchanged finite and exact
BF16-bin gates (`logs/compact-overlap-v7-{f32,bf16}-*`). In each input policy,
45/48 comparisons are also raw-F32 exact. The three T65/C32 first/changed/reuse
records have raw-F32 drift up to `5.2154064e-8` with zero finite violations or
BF16-bin differences. The B2/H3 partial-tail positive case exercises offsets
without qualifying partial multi-batch use in the block workspace. The guard
and retained plan-count checks also pass. Full block fixture runs also
exercise the workspace constructor directly: grouped and partial short/multi-batch
requests must reject before any tracked MetalBuffer or lifetime root is added.

The independent synthetic oracle uses PyTorch 2.8.0 MPS, git
`a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`, with CPU fallback disabled. Producer
`spec/support/hy_nomas_compact_attention_reference.py` SHA-256 is
`d5f5a4d8ac0a75f719470e532d57ca0a45b6a5cabe03bc399cf6c787647a670f`.
The verified [upstream Attention.mm](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/mps/operations/Attention.mm)
has SHA-256
`167ee89bddb489c154b5d5ca72cf042745443f557d624fc48175f2d29f45a351`.
The producer/consumer pin manifests, binary outputs, runtime/source provenance,
input formula, shapes and per-record byte ranges before device allocation.

For both F32 and BF16-rounded Q/K/V, C128/G16 matches the full upstream T257
output (526,336 values) and three selected 128-row outputs with full K/V length
6,560 (786,432 values, starts 0/3,200/6,432). These are raw-F32 bitwise equal.
The latter is a selected-row reference, not a full-Q6560 upstream graph or an
end-to-end model oracle. PyTorch is used only to create fixtures; the measured
native path does not contain PyTorch.

The block fixture matrix covers actual C2048/H16/D128/FF8192 checkpoint weights
and the tiny configuration, both F32 and BF16-boundary policies, both weight
residencies, broadcast/per-token modulation, chained outputs and perturbed
inputs. Integration uses C=T=9 (G16 actual/G2 tiny); it proves the workspace
binding/lifetime integration, not a representative long-query block speed.
Every finite comparison retains `2e-5 + 2e-5*abs(reference)` and selected BF16
boundaries require exact bits. Each of the four final compact block runs passes
eight examples, all 536 finite comparisons and all 480 raw-F32 checkpoint arrays;
the two BF16 fixtures also match all 72 selected boundary arrays exactly. A dense
MPS tiny-F32 default-path regression passes the same eight examples/536 comparisons.
Encode paths assert zero tracked MetalBuffer
allocations and one terminal wait per independent case. This does not bound
opaque MPSGraph/driver allocations or hidden API stalls.

### Native attention timing and memory geometry

Release-mode, GPU-filled BF16-rounded inputs, H16/D128, warm plans, four serial
paired repetitions in AB/BA/AB/BA order. Timing includes `plan.encode` through
one terminal wait, including K/V packing and query gather/output scatter;
setup/compilation, input fill and a separate GPU diagnostic are outside it.
Only the diagnostic's five counters are copied to the CPU after completion.
No synchronization was added inside production attention encode.

| Build/run | Tokens | Arm A | Arm B | Median wall A/B (ms) | Median GPU A/B (ms) |
| --- | ---: | --- | --- | ---: | ---: |
| Initial | 1,024 | C128 | canonical dense | 3.260 / 2.557 | 2.654 / 2.150 |
| Initial | 6,560 | C128 | C256 | 66.784 / 61.589 | 66.403 / 61.227 |
| Initial | 6,560 | C256 | C512 | 61.479 / 62.646 | 61.121 / 62.318 |
| Initial | 6,560 | C256 A/A | C256 A/A | 61.475 / 61.586 | 61.018 / 61.133 |
| Makefile | 1,024 | C128 | canonical dense | 2.808 / 2.271 | 2.489 / 1.930 |
| Makefile | 6,560 | C128 | C256 | 68.467 / 62.277 | 68.013 / 61.846 |
| Makefile | 6,560 | C256 | C512 | 62.920 / 63.581 | 62.286 / 62.961 |
| Makefile | 6,560 | C256 A/A | C256 A/A | 65.778 / 65.371 | 62.332 / 62.355 |

C256 reduces observed long-input median wall time by 7.8% versus C128 in the
initial run and 9.0% after a fresh Makefile build, with the same sign in all
eight pairs. The initial C256 A/A within-pair wall ratios range from 0.9922 to
1.0058. The Makefile A/A has wall outliers (ratios 0.8273 to 1.0445), while GPU
ratios remain 0.9955 to 1.0055. One arm takes 83.5 ms wall with 62.7 ms reported
GPU time. This is evidence of latency outside the reported GPU span; it does
not isolate host scheduling from queue/API delays. Both runs are retained.
C512 encodes faster on the CPU but takes longer on the GPU; its small advantage
or disadvantage in wall time is vulnerable to these outliers. No default is
selected from that margin. These are local attention measurements, not
full-block/video throughput.
The short dense control is faster; compact attention buys a bounded score
shape for long sequences, not a universal speedup over dense MPS.
All arm comparisons have zero nonfinite values, finite violations and BF16
bin mismatches, and maximum absolute error zero over every output value.
The long-input full-output parity chain is transitive: C128 versus C256 in one
run, then C256 versus C512 in a separate run on the same generated inputs.
Independent upstream evidence for C128 is limited to the selected rows above.

| J=6,560, B1/H16/D128 | Named stages (MiB) | Largest score (MiB) | Graph calls |
| --- | ---: | ---: | ---: |
| Dense | 205.0 | 2,626.5625 | 1 |
| C128/G16 | 104.5 | 51.25 | 52 |
| C256/G16 | 106.5 | 102.5 | 26 |
| C512/G16 | 110.5 | 205.0 | 13 |

Named stages are `2*B*H*(T+C)*D*4`; score shape is `G*C*T*4` bytes. They are not
opaque peak-memory bounds. The benchmark preflight estimates named buffers,
three score-sized reserves per arm and a 256 MiB opaque reserve; it rejects the
J6560 dense comparison before Arena creation under the 4 GiB estimate. A fit
is only a geometry estimate. The external guard is still mandatory for every
run and is not enforced by the benchmark executable itself.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-compact-attention-20261024/`.
`logs/*-argv.json` stores exact final commands; fixture pins and source/object/
binary hashes are indexed by `logs/final-receipt-v1.json`. Positive runs and
expected-failing counterexamples have separate outputs and guard logs. Every
reported run uses the unchanged sampled 4 GiB maximum (smaller caps for small
probes/builds), cache-aware 15% host floor, 64 MiB system swap-growth stop and
600 s deadline. The five final block runs (`logs/constructor-v4-*`) have a
maximum sampled group footprint of 3412.112 MiB, minimum cache-aware headroom
19.379% and zero swap growth. Sampling can miss short peaks; reclaimable
file-backed cache is an estimate, not guaranteed immediately free pages.
The fresh Makefile release target also builds successfully under the guard;
its four benchmark runs use the exact final benchmark source and bridge.

Rebuild the benchmark with the Makefile target below inside the same external
guard, with TMPDIR and CRYSTAL_CACHE_DIR on the SSD:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-compact-rebuild \
  /opt/mldata_ssd/cogni-ml/hy-compact-rebuild/hy_nomas_attention_benchmark
```

Run through `examples/support/wan_memory_guard.py` with
`--memory-accounting file-cache --max-footprint-mib 4096`,
`--max-swap-growth-mib 64 --timeout-sec 600 --log FRESH`. Pass the benchmark
`--tokens=6560 --query-chunk=128 --compare-query-chunk=256`,
`--input-bf16-boundary --repeats=4`. Use `--compare-dense` instead of the second
chunk for the T1024 control.
`make spec_hy_nomas` includes the standalone and pinned-fixture consumer specs;
the upstream consumer is pending unless all trusted fixture pins are supplied.

## Active slice: shared serial scratch and MPS workspace

Status: verified for the bounded serial workspace and fixture matrix below,
2026-10-24. Correlated MiniQuad verdict: ROBUST within this scope.
Risk: CAUTION (asynchronous shared storage and fixed MPSGraph bindings).
Scope is exactly two serial block applications, not full generation. Only the
32 block-0 tensors are downloaded; the pinned upstream chain oracle applies
that same block twice, not pretrained blocks 0 and 1.

One owner allocates the common intermediate scratch and one set of MPS
linear, RMS and attention staging/plans. Two distinct image/text output banks
keep the preceding block's outputs outside the next block's scratch view.
Per-block weights and resource records are registered at setup; graph-bound
linear/attention/activation buffers and geometry must remain identical.
RMS reduction graphs retain fixed square/mean buffers, with validated norm
weights selected per block. No within-block aliasing, arithmetic change,
additional GPU wait, or forced submission boundary is admitted. The existing
separate-scratch/workspace path remains the control.

The strongest failure is a second live command chain overwriting common
storage, or release of an owner while its graph work is in flight. A single
workspace chain guard must reject that use before any dispatch or active
resource switch. Early release, unregistered blocks, changed bindings, output
alias and repeated/new completed-chain reuse are required negative controls.
Caller-owned input/weight Arenas retain their existing lifetime contract.
Shared owner cleanup drains only at the caller's existing terminal boundary,
releases plans first, clears private borrowed maps without freeing their
buffers, then releases the owning Arena. Rollback removes this opt-in path;
no existing default or numerical tolerance changes.

DoD: fresh bounded builds and the pinned F32/BF16 block fixture consumers must
compare both applications' intermediate checkpoints using the unchanged
`2e-5 + 2e-5*abs(reference)` gate and designated exact BF16 boundaries. Small
fixture-only GPU snapshots preserve first-block checkpoints before reuse;
these diagnostic copies add no waits and are excluded from timing probes.
Native paired two-block probes compare separate and shared storage with one
terminal wait and zero encode-time MetalBuffer allocations. The small
different-weight control requires equal completed root counts. Representative
benchmarks report automatic MPS root segmentation as an observation, rather
than forcing extra submission boundaries to make it deterministic. The sampled process-group cap remains 4 GiB, cache-aware host
headroom 15%, swap-growth stop 64 MiB and timeout 600 seconds. Large geometry
is admitted only if its explicit plan fits; no full generation is authorized.

Implementation history: falsify the missing serial API; add dynamic RMS weight
validation, registered shared-plan bindings and the private scratch owner;
exercise fixture parity and lifecycle negatives; then measure bounded paired
probes and attack evidence/claims. Evidence expires on relevant source,
fixture, geometry, toolchain, bridge, OS/device or scheduling changes.

Probe-driven refinements: the combined four-variant BF16 fixture reported zero
numerical failures but its sampled process footprint reached 4.216 GiB, so the
external guard rejected the run. It is not an admitted passing run. Execute
each case/resident-dtype variant in its own fresh guarded process; retain the
same complete oracle matrix and all numerical gates. The v1 benchmark's third
J1024 process also refuted deterministic root-count stability for an identical
shared path: its first measured request had five roots, and the next request
triggered the root-count assertion. Preserve that failed receipt, report all
observed root counts, and continue to require exact output hashes, zero encode
allocations and one terminal wait. This changes telemetry admission only;
there is no inference or submission change.

### Serial workspace results

Freshly linked tests pass under the existing guards. Each F32-math and
BF16-boundary pinned oracle is consumed in four isolated processes:
`broadcast/BF16`, `broadcast/F32`, `per_token/BF16`, `per_token/F32`.
`HY_NOMAS_FIXTURE_VARIANT` validates the exact schema/case/residency map before
Metal initialization, reports selection as `1/4`, and retains the single
digest-verified JSON parse. Unset selection keeps the original complete matrix.
The eight runs each pass nine examples and 216 finite comparisons (1728 total).
All 640 saved serial first/second checkpoint arrays are raw-F32 exact; the 120
selected BF16 boundary arrays, including 48 serial arrays, also match exactly.
Other native arithmetic controls retain small within-gate drift: this is not
an all-1728-bitwise-exact claim. All tolerances and trusted source pins are
unchanged. The oracle uses short inputs and actual upstream ND RoPE, not J6560.

The focused different-weight successor test passes after GPU NaN poisoning of
all unique nonconstant serial scratch/output buffers before each of two
requests. All 40 second-block checkpoints match the separate-workspace control
at raw F32 bits. Different projection and RMS weights catch stale per-block
weight selection. Binding mutation, unregistered blocks, overlapping chains,
output aliases, early release and setup cleanup have rejecting controls. Reuse
covers fresh completed chains with the same fixed input buffer identities;
there is no input rebinding or concurrent-use API. The complete Makefile suite
passes 43 examples, zero failures/errors and four expected external-oracle
pending cases; the block fixture pending case is discharged by the eight
isolated runs above. This does not discharge the other external oracle scopes.

J1024 performance uses B1/image768/text256/C2048/H16/D128/FF8192, C256/G16,
BF16-resident actual block-0 weights, BF16-boundary attention, and synthetic
GPU inputs/RoPE. Four fresh processes run separate/shared/shared/separate;
each has one warmup and four measured two-block requests. All 16 measured
output hash pairs are identical across both modes. Medians over eight samples
per mode on Apple M2 Max, macOS 26.6.2, SDK 27.0, Crystal 1.22.1:

| Two-block measurement | Separate | Shared |
| --- | ---: | ---: |
| Encode + terminal wait | 34.288 ms | 34.920 ms |
| CPU encoding | 1.459 ms | 2.037 ms |
| Native GPU chain elapsed | 33.066 ms | 33.149 ms |
| Named resident buffers, bytes | 1,150,658,564 | 739,370,500 |
| Scratch + workspace, bytes | 839,353,344 | 428,065,280 |

This probe shows 411,288,064 fewer named resident bytes, not a speedup: shared
wall time is 1.0184 times separate. Shared admission performs extra CPU binding
checks; the sample does not isolate their exact cost. Separate roots are 5 in
all eight samples; shared roots are 5 in seven and 6 in one. Automatic
segmentation is observational. No forced commit or added wait is used to make
it stable. Native GPU elapsed is a whole-chain measure, not an individual
kernel cost. Timers exclude setup and post-terminal CPU output hashing.

At J6560 (image5280/text1280, same channels and C256/G16), one shared workspace
passes one warmup and four two-block requests. Median encode + terminal wait
is 275.237 ms, CPU encoding 3.977 ms, native GPU elapsed 273.181 ms. Output
hashes remain exact; each measured chain has 56 roots, one terminal wait and
zero tracked encode-time MetalBuffer allocations. The scratch owner uses
1,860,026,880 bytes, including two additional final-output buffers, and the
single MPS workspace uses 320,655,360 bytes: 2,180,682,240 bytes (2.031 GiB)
combined. Two independent scratch/workspace sets would require 4,253,885,440
bytes (3.962 GiB), calculated from the same plans; that large separate arm is
not executed under the 4 GiB cap. The performance probe applies block 0 twice
and uses synthetic theta-10000 RoPE, not upstream ND theta-256 RoPE. It is not
a full-J upstream, 54-block, diffusion or video-throughput qualification.

Maximum sampled process-group footprint / minimum cache-aware headroom:

| Admitted runs | Peak bytes | Minimum headroom | System swap growth |
| --- | ---: | ---: | ---: |
| Eight isolated fixture gates | 2,424,391,648 | 30.359% | 0 |
| Four J1024 processes | 1,823,016,400 | 34.273% | 0 |
| Shared J6560 | 3,419,833,856 | 31.857% | 0 |

The sampled 4 GiB process cap, cache-aware 15% floor, 64 MiB system swap-growth
stop and 600-second deadline remain intact. Existing system swap is nonzero;
zero growth is not zero swap. Sampling can miss short peaks; file-backed cache
is a reclaimability estimate. Named-buffer savings are plan/accounting facts;
sampled footprint does not establish an equally large physical peak reduction.
No full model or video generation ran.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-serial-workspace-20261024/`.
`logs/final-receipt-v1.json` binds current source/binary/object hashes, trusted
fixture/checkpoint pins, exact guarded commands, outputs, guards and retained
failures. `logs/serial-analysis-v1.json` checks complete selector coverage,
cross-mode hashes, gates, memory accounting and benchmark medians.
Rebuild with SSD-backed TMPDIR/CRYSTAL_CACHE_DIR through the same external
guard and Makefile target `hy_nomas_serial_benchmark`. Default geometry is
J1024; compare `--mode=separate` and `--mode=shared` in ABBA order. Add `--large`
only to shared mode; `--preflight-only` checks pins/geometry before Metal
initialization. `make spec_hy_nomas` includes the serial ownership test. Replay
each fixture selector using its recorded argv and trusted digests; do not
substitute new digests from mutable metadata. Refresh after scoped source,
fixture, geometry, toolchain, bridge/SDK/OS/device or schedule changes, or
artifact loss. Rollback omits the new serial wrapper/benchmark and uses
separate workspaces; existing inference defaults remain unchanged.

## Previous slice: representative one-block profiling

Status: verified for the bounded one-block probe, 2026-10-24; correlated
MiniQuad verdict ROBUST within this scope. Parent `45cfa8f5` supplied the
standalone compact attention control. The new benchmark uses B1, image/text rows
5280/1280, C2048/H16/D128/FF8192, broadcast modulation, actual block-0 checkpoint
weights converted from F32 to BF16 on GPU, and explicit compact C256/G16
attention with `BF16BoundaryControl`. Inputs use deterministic GPU-generated
values and synthetic unit-circle RoPE with theta 10000. This is not upstream's
ND RoPE (theta 256, axes [16,56,56], grid [6,22,40]); it is a performance and
repeatability probe, not a full-J upstream output or video-quality oracle.

Risk: CAUTION (performance instrumentation across asynchronous command roots).
The strongest failure is attributing a root shared by two phases entirely to
one phase, or changing the submission schedule to obtain cleaner timings.
Root-count observations and completed timestamp snapshots do not pause,
continue, submit or wait for GPU work. Sampling
the cursor can lock/register CPU tracker metadata, so observer-off arms alone
provide clean block timings. An optional trusted CPU diagnostic observer
marks validation, modulation, QKV, joint packing, attention, attention
projection/residual, image FFN and text FFN. Markers denote phase starts;
validation is CPU-only and complete is an endpoint. For GPU-producing phases,
the inclusive root range from the start cursor to the end cursor supplies
candidate attribution; roots crossing a marker retain all candidate phase
names. GPU intervals of shared roots are not exact phase
costs, and sums are not substituted for interval unions/envelopes.

Implementation: optional markers in `hy_nomas_block.cr` / `hy_nomas_block_mps.cr`,
read-only lifecycle-bound cursors/snapshots in `metal_mps_command_chain.cr`, and
CPU-only candidate mapping in `hy_nomas_block_profile.cr`. Dispatch boundaries,
fixed bindings, numerical policies and inference defaults are unchanged. A
trusted observer must only inspect CPU metadata; if it throws after work was
appended, callers must drain before releasing resources.

The fresh release Makefile benchmark uses one prepared workspace, one warmup,
and two ABBA groups (on/off/off/on), four paired comparisons. Each measured
sample starts a fresh chain, encodes one block, then performs its sole terminal
wait. Chain creation is reported separately. GPU output capture/comparison
commands run symmetrically outside block timers, with their own terminal waits;
only six UInt32 counters (24 bytes) are read to CPU. This adds no internal block
wait, neural activation readback or forced profiling submission boundary.

Observed medians on Apple M2 Max, macOS 26.6.2, SDK 27.0, Crystal 1.22.1:

| Measurement | Observer off | Observer on |
| --- | ---: | ---: |
| Encode + terminal wait | 137.626 ms | 137.886 ms |
| CPU command encoding | 1.853 ms | 1.787 ms |
| Terminal wait wall time | 135.795 ms | 135.987 ms |
| GPU root-interval union | 136.704 ms | 136.992 ms |
| GPU root envelope | 136.720 ms | 137.008 ms |

The first warmup block took 392.865 ms, including 225.348 ms of encoding; this
excludes checkpoint loading, workspace construction and graph preparation.
All eight measured outputs match the warmup at raw F32 bits over 13,434,880
values per sample, with zero nonfinite, finite-gate or BF16-bin violations.
Every sample has 28 roots, one block terminal wait and zero tracked encode-time
MetalBuffer allocations. This does not establish zero opaque driver allocation.
Observed paired on-minus-off wall deltas range from -56 to +349 microseconds;
four pairs do not establish general negligible/zero observer overhead.

Root attribution in all four observed samples:

| Roots | Candidate phases | Median interval duration |
| --- | --- | ---: |
| 0 | modulation, QKV, joint packing, attention | 25.497 ms |
| 1..25 | attention only | 57.623 ms summed per sample |
| 26 | attention, projection/residual, image FFN | 27.189 ms |
| 27 | image FFN, text FFN | 26.536 ms |

The 57.623 ms is the median of each sample's sum over its 25 attention-only
roots, roughly 42% of the observer-off GPU union. Boundary roots retain multiple
labels, and intervals can include stalls. This is not an exact kernel-cost or
QKV/FFN breakdown; all-candidate sums overlap and cannot be added. CPU encoding
is small relative to GPU execution, so encoding micro-optimizations are not the
main measured lever.

Guarded DoD passed: `hy_nomas_block_profile_spec.cr` has three CPU examples;
`video_mps_command_chain_spec.cr` has three lifecycle examples; freshly linked
`hy_nomas_block_spec.cr` passes eight examples / 536 finite comparisons for each
of the pinned real F32 and BF16 fixtures. Both block runs link the exact seven
fresh bridge objects used by the benchmark. All 480 canonical saved checkpoint
arrays per fixture are raw-F32 exact; the BF16 fixture also matches 72 selected
boundary arrays. Alternative native modulation controls retain their small
within-gate differences; do not call all 536 comparisons bitwise exact. The
upstream full-block fixtures have six image and three text rows, not J6560.

Preflight estimates 3,737,058,840 bytes, including loader staging, score reserves
and a 256 MiB opaque-allocation heuristic. It is not a guaranteed peak bound.
The benchmark's maximum sampled group footprint is 3,408,070,024 bytes
(3.174 GiB); minimum cache-aware headroom is 41.718%; swap growth is zero.
Across the two fresh fixture gates, the maximum sampled group footprint is
3,576,301,992 bytes and minimum headroom 23.295%, with zero swap growth. All
runs use the external 15% cache-aware floor, sampled 4 GiB cap (smaller for
builds/tests), 64 MiB system swap-growth stop and 600 s deadline. Sampling can
miss short peaks; file-backed cache is a reclaimability estimate. The executable
does not enforce these runtime limits itself. Retained wrapper arrays grow with
`--groups`; this evidence covers two groups under the guard, not arbitrary runs.

An initial build was stopped for system swap growth of about 204 MiB while its
own group used about 84 MiB. The guard was not weakened; a read-only stability
check preceded retry. A Crystal digest API error, an external test-bundle require
path error and a test-only borrowed-encoder comparison error were corrected;
their failed receipts remain alongside the passing runs.

Memory audit: diagnostic scratch has 55 distinct buffers totaling
1,806,287,360 bytes (1.682 GiB). All 14 projections already share one 96 MiB
F32 weight-expansion stage. Independent per-block scratch would consume about
90.84 GiB for 54 blocks, before weights/workspaces. The next necessary step is
sharing scratch/workspace across serial blocks under an explicit lifetime
contract; current retained checkpoints and fixed graph bindings prohibit blind
aliasing. Omitting expansion for F32-resident weights is a separate possible
control, not a solution for the target BF16 residency.

Conditional arithmetic: `432 * 137.626 ms = 59.4545 s` for isolated block-0-
equivalent calls at these operands/geometry, each with a terminal wait. This
passes the necessary 277.8 ms mean-block ceiling discussed above, but does not
measure a 54-block stack, eight diffusion steps or seconds per video second.
Other blocks, real inputs/ND RoPE, scheduler, text encoding, camera/history and
PixelUMM remain outside this timing.

Evidence root: `/opt/mldata_ssd/cogni-ml/hy-nomas-full-block-profile-20261024/`.
`logs/final-receipt-v1.json` indexes source/binary/object hashes, trusted fixture
pins, exact guarded commands, outputs, guard samples and failures. The derived
breakdown is `logs/benchmark-analysis-v1.json`. Rebuild under the existing guard
with SSD-backed TMPDIR and CRYSTAL_CACHE_DIR:

```sh
make BUILD_DIR=/opt/mldata_ssd/cogni-ml/hy-block-rebuild \
  /opt/mldata_ssd/cogni-ml/hy-block-rebuild/hy_nomas_block_benchmark
```

Run through `examples/support/wan_memory_guard.py` with
`--memory-accounting file-cache --max-footprint-mib 4096`,
`--max-swap-growth-mib 64 --timeout-sec 600 --log FRESH`, then the benchmark
`--groups=2`. `--preflight-only` validates source pins and geometry before Metal
initialization. Exact fixture-gate environments are in the receipt's argv files;
retain the trusted fixture/checkpoint digests rather than accepting new ones
from mutable metadata. `make spec_hy_nomas` includes the new CPU profile spec.

Rollback: omit the optional observer and remove the new benchmark; no inference
default, numerical policy, buffer binding or scheduling change is admitted.
Evidence decays on source, fixture, geometry, compiler/SDK/OS/device, numerical
policy or host-scheduling changes. No full model or full generation is admitted.

## Continuation

Active goal: native Hy-Nomas port. The current bounded composition qualifies
the natural-prompt 26-block CogniQwen prefix, GPU crop, actual two-block text
refiner/time embedding and one short-input DiT block on the same chain.
The separate saved-input replay now qualifies distinct actual DiT blocks 0 to 1
with shared scratch/workspace; it does not extend the live Qwen chain itself.
The qualified results at the top of this document and
`docs/qwen25-hy-nomas-frontier.md` define its input and precision limits.
The next DiT frontier is recorded below. Wan performance work remains
at `docs/wan-performance-frontier.md`, commit
`e22fc2ba`; no Wan default has been changed. Confirmed frontier: the actual
checkpoint block passes the declared F32 and BF16-boundary MPS controls with
both weight residencies and chained GPU outputs. Compact attention is explicit
opt-in; C256/G16 is the fastest tested long-query candidate, with selected-row
upstream parity and full-output native cross-chunk parity on the exercised
inputs. Known grouping/short-chunk/multi-batch counterexamples remain rejected
by workspace admission and retained as failed diagnostics. No general shape or
activation qualification follows from the measured matrix.

The shared serial wrapper reuses intermediate scratch and one MPS workspace
with distinct final-output banks. The real block-0-to-block-1 saved-input probe
passes all 80 upstream intermediates and raw-F32 independent/poison/repeat gates.
Its 109,670,528-byte scratch/workspace saving is a short-geometry result. Earlier
J6560 block-0-twice timing remains 275.237 ms warm median wall and 2.031 GiB named
scratch/workspace; independent sets would use 3.962 GiB by calculation. J1024
ABBA shows no speedup. Neither timing qualifies a real 54-block stack.
The authenticated DiT parameter/block-workspace plan is now recorded at the
top; complete pipeline residency remains open. BF16 source caches now remove
cold F32 staging in the actual two-block saved-input replay: all 64 weight
tensors and 80 captures per request are raw exact with one terminal wait.
The selected-capture J6560 positional/output extension is now qualified at the
top of this document. Next: qualify cache coverage beyond the two local blocks.
The 64 tensors from blocks 0/1 plus 42 refiner/time tensors are local; full-model weights are
not downloaded. Retain the independent plans and single-block path as controls.
Real upstream ND RoPE application and the selected full-J outputs now pass the
bounded synthetic-input gate. Full conditioning, stack composition, exact
CUDA/Magi output dtype and BF16 activation semantics remain unresolved before
pipeline quality claims.
Standing authority remains bounded GPU probes. The new connected conditioning
slice explicitly uses a cache-aware 10% floor and sampled 15 GiB process cap;
the short saved-input two-block oracle/probe uses 4 GiB, and full-J upstream/
native probes use 8/6 GiB; small controls use 3 GiB.
All retain the 64 MiB swap-growth stop and 600 s deadline.
Earlier block-profile limits remain historical.
Full generation is outside this slice. Refresh evidence after source,
fixture, geometry, numerical policy, compiler, bridge, SDK/OS/device or scheduling
changes, or artifact loss.
