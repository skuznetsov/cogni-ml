# Qwen2.5-VL text decoder for Hy-Nomas

Status: one natural text-only video prompt passes native preparation,
embedding, the opt-in fed-RMS 26-block prefix and GPU conditioning crop against
a fresh official MPS BF16 capture, 2026-10-25. The full 1388-token input has
134,313,984 raw-bit-exact values across 27 hidden boundaries. Its selected
1280-token suffix has 4,587,520 exact BF16 values and an exact Int32 mask.
The unchanged finite gate is `atol=rtol=2e-5`. Execution retains one terminal
wait, zero pre-terminal activation reads and zero tracked encode allocations.

The legacy default is unchanged. Its earlier S4/S32 prefix qualifications and
S1280 rejection remain historical controls. The S1280 investigation localized
the first divergence to block-0 post-attention RMS square statistics; matched
constant/fed controls identified an operand-source-sensitive route on that
saved input. A subsequent same-binary block-0 pair qualified all 17 stages;
the fresh prefix run now qualifies all 27 retained hidden boundaries. The
52 individual RMS intermediates are not retained by the shared workspace.
The earlier repeated-token S1280 fed capture remains a separate qualification:
123,863,040 exact values across the same 27 boundaries.

The adapter, fed option and new prompt/crop path remain opt-in. Qualification
is limited to the inputs and boundaries below. A subsequent opt-in composition
now passes the actual Hy-Nomas text refiner/time embedding and one six-video-token
DiT block on the same GPU chain for this prompt; see
`hy-nomas-native-frontier.md` and
`.landmark/evidence/2026-10-25-hy-nomas-text-conditioning.json` for its separate
finite-gate scope. Multimodal positions, all-input parity, full DiT and video
generation remain unqualified.

## Natural video prompt and GPU conditioning crop: qualified on one prompt

CAUTION, 2026-10-25: qualify one distinct natural prompt through the pinned
Hy-Nomas `li-dit-encode-video-json` template and Qwen2.5-VL tokenizer. The
previous probe accepted prepared IDs; its repeated S1280 fixture could not establish
prompt preparation or output cropping. Reuse native byte-BPE with the exact
Qwen2.5 pretokenizer and fail closed on unsupported tokenizer assets. Preserve
the default Qwen3.5 pretokenizer. CPU tokenization and source-file I/O precede
GPU encoding; no neural operation or activation crosses to CPU before drain.

The original constructor discovers the user-header crop using a 256-token
prepass on `a photo of a cat`. Video tokenization then right-truncates/pads to
`1280 + crop_start`, not 1280. Forward the complete system/user context with
three identical position axes `0..Sraw-1`, including pad positions, through
embedding and all 26 blocks. Hy-Nomas calls the inner text model, bypassing
the outer multimodal position builder. Crop only the final selected hidden
output and mask. Prepare contiguous GPU destination buffers before encoding;
append exact-byte copies to the existing command root. Rollback is omitting
the new prompt/crop option; saved explicit-ID probes and RMS defaults remain.

DoD: CPU differentials must match rendered UTF-8, IDs, right padding,
truncation, the discovered crop and positions against mechanically extracted
upstream template/tokenization methods and the pinned tokenizer. Reject
changed asset hashes and a wrong crop/input binding. Test one-byte-short crop
admission before allocation. Capture a fresh official MPS BF16 oracle for the
natural prompt, then run the native tokenizer, embedding, fed 26-block prefix
and conditioning crop sequentially against it. Compare all 27 raw hidden
boundaries and the cropped final hidden/mask at unchanged `atol=rtol=2e-5`,
reporting raw-word differences separately. Require one terminal wait, zero
pre-wait activation reads, zero tracked encode allocations, and in-flight
release rejection. Use file-cache-aware 10% admission, 15 GiB sampled process
cap, 64 MiB swap-growth stop and a 600-second deadline. No full generation.

The strongest failure is apparently correct prompt IDs with a wrong upstream
boundary: cropped inputs would discard system context, reset positions, and
change every later hidden state. Authenticate the complete raw request and
crop metadata, and compare the GPU crop with the authenticated raw suffix.
Bind claims to the actual Transformers version and source hashes; local
4.57.3 is not the upstream dependency pin 4.57.1. A failure retains its first
boundary and leaves this route opt-in without widening numerical tolerances.

### Observed result

The prompt is `A yellow tram turns through a wet downtown intersection at dusk
as pedestrians wait under umbrellas.` with no trailing newline. Native CPU
preparation matches the independently produced rendering, token IDs, mask,
positions and crop. The raw width is 1388: 108 system/header tokens plus 1280
conditioning positions. There are 131 visible raw tokens and 23 visible suffix
tokens. All raw positions, including padding, retain their original indices.
The selected checkpoint and tokenizer revision is
`eb6ba55b40ada64977147b52e50a4145ee89ac4f`.

The fresh oracle runs official Transformers 4.57.3 decoder layers 0 through 25
on MPS in BF16. Pinned Hy-Nomas template/tokenization methods are mechanically
extracted from source revision `911cea6458a89c2663513ff7099561ec190dd491`;
the complete Hy-Nomas model is not instantiated. The source-derived selected
state is `hidden_states[-3]` of its 28-layer text model, without final norm.
This is not a claim about the upstream dependency pin 4.57.1 or CUDA reductions.

All 27 native raw boundaries pass: 134,313,984 BF16 words, zero tolerance
failures and zero differing raw words. The final GPU crop passes 4,587,520
BF16 words and 1280 mask entries, also exactly. Native preparation supplies the
actual uploaded arrays; the reference IDs are an equality check, not the input
source. Two preallocated byte copies follow the prefix on the same command
root. There is one terminal wait, zero tracked MetalBuffer allocations during
encoding, zero reads through the three instrumented MetalBuffer read APIs
before drain, and 29 diagnostic reads after it. Both prefix and crop plan
release attempts before completion are rejected.

Aggregate named admission is 12,738,809,672 bytes under a 13 GiB cap. The crop
adds 9,185,712 bytes: 5552 raw-mask bytes, 9,175,040 hidden-suffix bytes and
5120 cropped-mask bytes. The native GPU elapsed interval is 2.796 seconds;
encoding is 2.039 seconds and terminal waiting 0.965 seconds. These intervals
overlap and must not be summed as GPU time. The entire guarded native command
takes 25.907 seconds, including setup and comparison; this is a correctness
run, not a speedup or video-throughput benchmark.

The independent external guards preserve the file-cache-aware 10% floor,
15 GiB sampled process cap, 64 MiB swap-growth stop and 600-second deadline.
The oracle/native sampled peaks are 14,714,783,832/13,744,023,416 bytes;
minimum cache-aware headroom is 12,254,314,496/18,184,142,848 bytes. Both finish
with exit zero and zero observed swap growth. These sampled observations do
not reserve memory or bound between-sample spikes.

The focused Crystal suite reports 56 examples, zero failures and three existing
pending cases requiring the external `llama-tokenize` utility; 20 Python tests
pass, including
the pinned prompt integration. Seven CPU native/upstream differentials match
all 18 fields: five nonblank inputs cover Russian/English, decomposed Unicode,
newlines, a special token and right truncation from 1914 to 1388 tokens. Two
separate tests exercise the original empty/whitespace semantics by bypassing
only the fixture helper's file-reader guard; the ordinary helper correctly
rejects those inputs. Normal/cpu-only type checks, strict Clang compilation,
release linking, formatting and diff checks pass.

Correlated MiniQuad review is ROBUST for this prompt and the documented serial
caller contract. Borrowed source and destination handles must remain valid and
unmodified until drain; the API does not mechanically prevent direct buffer
mutation or certify a source buffer's producer-chain identity. The runner obeys
those preconditions. The generic blit bridge and two pinned reused companion
objects have no new checked ABI or hermetic rebuild claim. All-input parity,
multimodal/batched inputs, the DiT handoff, generation and default promotion
remain outside this certificate.

Evidence: `fixtures/prefix-natural-tram-v1.{json,bf16le}`,
`logs/{oracle,native}-prefix-natural-tram-v1.*` and
`logs/natural-prompt-receipt-v1.json` under
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024`, indexed by
`.landmark/evidence/2026-10-25-qwen25-natural-prompt.json`.
Refresh after scoped source, prompt/tokenizer/template, checkpoint, geometry,
precision, compiler/SDK/OS/device, scheduling/lifetime changes, or artifact
loss/hash mismatch. Next falsifier: admit the actual Hy-Nomas text projection
and its weights, then consume this resident conditioning in a bounded DiT
block and compare intermediate outputs with upstream. No full generation.

## Block-0 GPU-fed RMS integration: qualified on the saved S1280 input

CAUTION, 2026-10-25: qualify the saved S1280 block with one prepared immutable
F32 exponent `2.0` shared by both RMS statistics steps. The prior connected
RMS/gain probe is exact on the saved residual, but that does not establish
equivalence after block graph compilation or serial scalar-scratch reuse.
Keep the legacy constructor/default and use an explicit opt-in block path.
Account its four named bytes before allocation, retain it until the terminal
drain, and reject null, wrong-size/value/device, or overlapping native feeds.
Rollback is disabling the opt-in path; prefix defaults remain unchanged.

DoD: build the focused specs and `qwen25_pretrained_block_probe`, then run
legacy and `--rms-power-fed` controls sequentially on the same authenticated
block-0 input, checkpoint and 17-stage oracle. Require the candidate's full
stage/output gate at unchanged `atol=rtol=2e-5`, report raw BF16 differences,
and directly compare normalized scratch and the retained post-RMS inverse
root with the official fixture. The older RMS traces remain negative controls
when their statistics algorithm differs from the candidate. Require one
terminal wait, zero pre-terminal reads and zero tracked encode allocations.
Admit the explicit named footprint plus opaque runtime budget under the
file-cache-aware 10% floor, 64 MiB swap-growth stop and bounded deadline.
No prefix or full generation is admitted by this slice. A failure preserves
the first differing stage and leaves the candidate opt-in.

### Observed result

The fresh same-binary pair uses checkpoint revision
`eb6ba55b40ada64977147b52e50a4145ee89ac4f`, the retained authenticated S1280
prefix and block-0 stage fixtures, and unchanged `atol=rtol=2e-5`. The candidate
passes all 17 stages: 116,326,400 BF16 words, zero numerical failures and zero
differing raw words. Its final 4,587,520-word hidden output also matches the
independently retained prefix state 1 exactly. The input RMS output after gain
is included in the 17-stage comparison. Post-attention normalized scratch
(4,587,520 BF16 values) and inverse root (1,280 F32 values) match their official
captures raw-bit exactly. Input inverse root is overwritten by shared scalar
scratch reuse and is not separately retained or compared here.

The legacy control exits one and reproduces the earlier first divergence:
post-attention RMS after gain has 27 tolerance failures and 28 differing raw
words; final hidden output has 596 failures/differing words. The candidate's
two older standalone RMS traces retain square statistics and remain visible
negative controls. They do not gate the fed candidate: its normalized scratch
and inverse root are compared directly against the official fixture. This
keeps trace-algorithm disagreement separate from production parity.

Both runs use one terminal wait, zero pre-terminal activation reads and zero
tracked encode-time MetalBuffer allocations, with 29 diagnostic reads after
the wait. In-flight release is rejected. The new ABI rejects null, wrong-size
and wrong-value exponent feeds in native probes; same-device/shared-storage
and full-disjointness checks are present in source but not independently
exercised by those three probes. One plan-private immutable four-byte buffer
is retained for both RMS sites. Candidate named admission is 854,129,924 bytes,
exactly four bytes above the legacy control; 19 prepared block steps and the
encode implementation are unchanged. The old constructor ABI and PrefixPlan
defaults remain unchanged.

The file-cache-aware 10% reserve, 5 GiB sampled process cap, 64 MiB swap-growth
stop and 180-second deadline remain intact. Candidate sampled peak is
2,076,641,416 bytes, minimum cache-aware headroom 27,374,583,808 bytes and
minimum pressure-free percentage 74; observed swap growth is zero with no
guard abort. The legacy sampled peak is 1,927,071,688 bytes. These observations
are not memory reservations or instantaneous peak bounds.

Thirty-five focused Crystal examples, normal/cpu-only no-codegen, strict Clang,
release linking, formatting and diff checks pass on the final source. Correlated
MiniQuad review is ROBUST for this saved-input block/lifecycle scope. The oracle
was retained and hash-authenticated, not recaptured in this slice. Companion
objects have pinned bytes but opaque source ancestry, so this is not a hermetic
rebuild. No speedup, all-input, full-prefix or generation claim follows.

Evidence: `logs/native-block0-rms-{legacy,fed}-v1.*`,
`logs/block0-rms-analysis-v1.json` and
`logs/block0-fed-rms-receipt-v1.json` under
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024`, indexed by
`.landmark/evidence/2026-10-25-qwen25-block0-fed-rms.json`.
The next falsifier was to separately admit the fed option in the 26-block prefix,
including per-plan exponent accounting and shared-workspace lifecycle, then
compare all retained hidden states at S1280 under the same finite gate and
10% reserve. The qualified prefix result below discharges that saved-input
falsifier; the legacy rejection remains a negative control.
Refresh this block certificate on scoped source, fixture/input, precision or
geometry, toolchain/SDK/OS/device, execution/lifecycle changes, artifact loss
or hash mismatch.

## S1280 prefix GPU-fed RMS integration: qualified on the saved input

CAUTION, 2026-10-25: extend the qualified opt-in block path to every block in
the 26-block prefix. Keep the shared stage/scratch workspace and each retained
hidden output unchanged. Each block owns its prepared immutable four-byte
exponent, shared by that block's two RMS sites; admit all 104 additional named
bytes before allocating any prefix resources. Preserve the legacy default and
its authenticated numerical rejection as a negative control. Rollback is
disabling the prefix option.

DoD: first reject a one-byte-short fed prefix budget in a CPU falsifier, then
build the native prefix probe and compare embedding plus every retained hidden
state against the authenticated saved S1280 upstream fixture. Keep
`atol=rtol=2e-5`, report raw BF16 differences separately, and require all 26
blocks to bind the fed path. Encode embedding and the entire prefix in one
chain: one terminal wait, no pre-terminal activation reads or tracked encode
allocations. Exercise premature release rejection before the same terminal
wait. Source-inspect shared scratch lifetime and serial ordering through a
correlated MiniQuad review. Run only after file-cache-aware admission with a
10% reserve, the prior 15 GiB sampled process cap, 64 MiB swap-growth stop and
600-second deadline. An admission or numerical failure retains its exact
boundary; do not widen tolerances or run full generation.

### Observed result

The freshly built native candidate exits zero under the guard and compares
all 27 saved boundaries: embedding and outputs of blocks 0--25. Each boundary
has 4,587,520 BF16 words; all 123,863,040 words match exactly, with zero
tolerance failures, zero raw differences and max_abs zero. The retained oracle
is authenticated by manifest SHA256
`b78616cc1dc3b1deb1215b06ecc49fbf60e3b38477f24aec2a0ccd60da745d3a`
and sidecar SHA256
`59404b4664b1f5d5313c8719565324da2dd77d03c24b14d5278c8d66be5f3bd2`;
it was not recaptured in this slice. The earlier legacy prefix rejection uses
the same fixture but an older binary, so it is a historical negative rather
than a fresh same-binary prefix pair or an isolated causal experiment.

The runner checks all 26 block options and reports 52 configured RMS sites.
Numerical coverage is the 27 retained hidden boundaries, not 52 independently
captured RMS intermediates. Each plan owns a four-byte immutable exponent for
its two RMS sites. Total named admission is 12,681,448,808 bytes, including
104 exponent bytes. Shared stage/scratch is 311,301,120 bytes; retained hidden
outputs are a separate 238,551,040 bytes. Their 549,852,160-byte sum remains
the existing `workspace_bytes` budget and `shared_workspace_bytes` report
field; the new split fields distinguish shared storage from retained outputs.
Exact-fit and one-byte-short CPU admission checks pass for one and 26 blocks.

The prefix has one terminal wait, zero pre-terminal activation reads, zero
tracked encode-time MetalBuffer allocations and 27 post-wait diagnostic reads.
Premature prefix release is rejected before that same wait. Block arithmetic,
native encode steps and serial shared-workspace ordering are unchanged.
The observed GPU interval is 2.650 seconds; this single correctness run is not
a speedup benchmark. The complete guarded process, including authentication,
weight loading and diagnostics, takes 30.203 seconds.

The sampled process-group peak is 13,596,551,152 bytes under the 15 GiB cap;
minimum cache-aware headroom is 17,448,779,776 bytes and minimum pressure-free
percentage is 58. Observed swap growth is zero and there is no guard abort.
The 10% reserve, 64 MiB swap-growth stop and 600-second deadline are intact.
Sampling and file-backed reclaimability do not establish a reservation or
hard instantaneous peak bound.

Thirty-six focused Crystal examples, normal/cpu-only no-codegen, strict Clang,
release linking and format/diff checks pass. Correlated MiniQuad source and
artifact review returns ROBUST within the declared saved-input scope. Reused companion
objects have pinned bytes but opaque source ancestry; this is not a hermetic
build. The repeated-token, right-padded S1280 fixture does not qualify natural
prompts, tokenizer/cropping, multimodal positions, batching or full generation.

Evidence: `logs/native-prefix-s1280-rms-fed-v1.*` and
`logs/prefix-fed-rms-receipt-v1.json` under
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024`, indexed by
`.landmark/evidence/2026-10-25-qwen25-prefix-fed-rms.json`.
Next falsifier: compare fed hidden states on a distinct natural text-only
prompt with authenticated upstream tokenization, position/mask and cropping
semantics before connecting the Hy-Nomas text-conditioning pipeline. Keep the
fed route opt-in; no full generation. Refresh on scoped source, fixture/input,
precision/geometry, toolchain/SDK/OS/device, execution/lifecycle changes,
artifact loss or hash mismatch.

## Pretrained prefix slice

The remainder records earlier slices and their then-open falsifiers. The
qualified opt-in saved-S1280 boundary is stated above; legacy rejections
below remain useful historical controls.

This slice connects the pinned pretrained BF16 checkpoint and compares all
26 selected decoder outputs with the official Transformers MPS decoder.
The bounded right-padded S4/S32 checks below establish its tested numerical
boundary independently of the earlier constructor-weight qualification.
Full 1280-token prompt
qualification, tokenizer/cropping integration and video generation remain
separate boundaries.

All 26 blocks' weights must be resident before encoding. Their exact BF16
payload is 12,119,005,184 bytes. The 1,089,994,752-byte embedding source table
stays on disk; only the selected BF16 rows are uploaded during preparation.
Use a shared stage/scratch workspace for ordered block execution and retain
each block's hidden output for comparison after one terminal drain. Admit the
aggregate named buffers before allocation, plus an explicit sampled process
budget for opaque MPS/runtime resources. The guard remains file-cache-aware,
with a default 15% physical-memory reserve and a 64 MiB swap-growth stop.
The user authorized a per-run floor of 10% or 7.5%; these bounded pretrained
runs explicitly select 10%. The accepted override range is [7.5%, 100%], and
the exact decimal value applies to both startup and runtime checks. Other
process, swap, containment and deadline checks remain unchanged. Its sampling
does not reserve memory or prove a hard instantaneous peak bound.

CAUTION: the strongest new failures are a wrong checkpoint/key remapping,
reuse of shared scratch across unordered chains, overwriting a retained hidden
output, and a BF16 discrepancy that accumulates across blocks. Falsifiers must
cover malformed checkpoint metadata and out-of-range token IDs, a workspace
budget one byte below the required size, premature release/second-chain reuse,
and per-layer hidden comparisons at the unchanged finite gate. Source shard
hashes, tensor inventory and fixture/runtime provenance must be retained.
No full model, vision encoder, LM head, last-two-block or video run is admitted.
Rollback is removal of the opt-in prefix adapter, loader and focused runner.

### Historical legacy S1280 check: rejected at block-0 post-attention RMSNorm

The bounded probe extends only the sequence geometry: 1,280 query rows,
1,260 visible tokens (the authenticated S32 case's 20 visible IDs repeated
63 times), 20 right-padding IDs and position offset 7. It has 18 unique
embedding rows. This is a deterministic numerical stress case, not a natural
language prompt or tokenizer-integration qualification. The exact aggregate
native named admission is 12,681,448,704 bytes, including the
549,852,160-byte combined shared stage/scratch and retained-output budget
and all 26 resident BF16 blocks.

Run the official producer and native prefix sequentially, each under the
file-cache-aware guard with an explicit 10% floor, 15 GiB sampled process cap,
64 MiB swap-growth stop and 600-second deadline. At 64 GiB physical memory,
startup therefore requires 21.4 GiB cache-aware headroom. The producer's
linear estimate and the native named count exclude some quadratic attention
and runtime resources: one F32 [1,28,1280,1280] score tensor is 183,500,800
bytes. Neither estimate is a peak-memory guarantee. Stop the owned probe on
any guard violation; do not lower the floor or run both producers together.

DoD: guarded official capture exits zero; its freshly observed manifest digest
authenticates the native fixture; guarded `qwen25_pretrained_prefix` compares
embedding plus all 26 hidden outputs at unchanged `atol=rtol=2e-5`, reports
one terminal wait, zero tracked encode allocations and no pre-terminal
activation readbacks. Report raw BF16 differing words separately. A failure
preserves the report and first failing state for localization, leaving S4/S32
qualification intact. No video, full generation or additional host wait is
admitted. External inputs, command records, fixtures and reports live under
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024` with fresh S1280 names.

Initial outcome: `oracle-s1280-v1` exits zero; authenticated manifest SHA256
is `b78616cc1dc3b1deb1215b06ecc49fbf60e3b38477f24aec2a0ccd60da745d3a`.
`native-prefix-s1280-v1` compares all 27 states and rejects state 1: 596 of
4,587,520 BF16 words differ, all outside the unchanged gate, with max_abs
0.00390625. Embedding is raw-bit exact. Error then accumulates, reaching
3,965,336 failures and max_abs 28 at state 26. The native encode still has
one terminal wait, zero tracked allocations and no pre-terminal activation
reads. This is a measured rejection, not an S1280 qualification.

Both sampled guards complete without a limit violation or observed swap
growth. Oracle/native sampled peaks are 14,736,099,344 / 13,606,266,912 bytes;
minimum cache-aware headroom is 17,976,426,496 / 19,213,680,640 bytes.
The indexed one-block diagnostic now supports layers 0--25 and optional input
or post-attention RMS traces, preserving byte/digest limits and one terminal
wait. `oracle-block0-s1280-v1` reproduces the full prefix's state 1 bit for bit.
`native-block0-s1280-v1` is raw-BF16 exact through input norm, Q/K/V, RoPE,
attention, output projection and attention residual. Its first failing stage
is `layers.0.post_attention_layernorm`: 28 differing BF16 words, 27 finite
failures, max_abs 0.001953125. Its final 596 differing words reproduce the
full-prefix state-1 failure. An input-norm-only internal trace cannot explain
the later post-attention norm; it is not evidence for that norm's arithmetic.
The subsequent post-attention trace compares actual official variance,
epsilon addition, inverse root, F32 scaling and BF16 rounding on that same
authenticated residual.

`oracle-block0-s1280-post-v1` again reproduces state 1 exactly and captures
the actual post-attention norm, not a substituted formula. Its authenticated
manifest SHA256 is
`d29c8a6bf457f0d848fa29c4dc734c806c54aee063db7c641032460811c331b9`.
`native-block0-s1280-post-v1` reports 670 differing F32 variance rows with
max_abs 2.2351741790771484e-8; the same rows differ after epsilon addition.
Precise inverse root differs in 371 rows, normalized BF16 in 50 words,
the gained norm in 28 words, and the final hidden output in 596 words.
The separately compiled precise trace and actual production scratch match
bit for bit for both inverse root and normalized BF16. Thus the diagnostic
reproduces the production mismatch; its extra outputs are not evidence of
an entirely different normalized result. Direct pre-root production statistics
had not yet been captured in that run, so its intermediate attribution retained
that limitation; the isolated reduction probe below now exposes the direct
production helper's variance-plus-epsilon target.

An SSD-only control replacing `squareWithTensor` with `power(x, F32(2))`
produces the same variance, inverse-root and normalized hashes and the same
596-word hidden discrepancy (`native-block0-s1280-post-power-v1`). Operator
spelling alone is refuted for this case. An SSD-only materialized-square
control (`native-block0-s1280-post-materialized-v1`) then separates the
square graph from mean/epsilon addition, reusing existing SiLU scratch before
its later write. It also produces the same variance, inverse-root, normalized
hashes and all stage discrepancy counts. Both controls preserve named bytes,
one terminal wait and zero tracked encode allocations or pre-terminal reads.
Materialization at this boundary is therefore also refuted as a repair for
this fixture; neither experimental bridge is adopted.
The updated indexed runner also passes the old layer-24 S32/input-site fixture
with all 17 stages raw-BF16 exact (`native-block24-s32-indexed-regression-v1`).
The final focused CPU producer suite passes 29 tests, and Crystal formatting
passes. Production math remains unchanged; the diagnostic extension and
measured rejection completed that slice, while S1280 parity was still open.

Correlated MiniQuad review returns ROBUST for fixture binding, indexed/site
selection, lifecycle constraints and localization; it does not qualify S1280
parity or establish an exact reduction-order cause. At that boundary the next falsifier was an
isolated same-input F32 square/mean capture against the actual production
statistics graph, distinguishing square values from the reduction itself
before changing arithmetic. Its measured continuation is below. The existing production-root and normalized-BF16
matches constrain that probe, but do not expose pre-root production variance.
Refresh these claims on a source, fixture/checkpoint, geometry, precision,
compiler/SDK/OS/device or buffer-lifecycle change. Curated commands, hashes,
guard summaries and limitations are indexed in
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024/logs/prefix-s1280-receipt-v1.json`;
the index is not itself proof. No generation or performance claim follows.

### Active RMS square/reduction localization

Admit only the authenticated block-0 S1280 post-attention residual and pinned
RMS gain. Distinguish BF16-to-F32 cast/squares from the last-axis reduction,
and observe the exact production statistics helper's output before inverse
root overwrites it. An extra-output diagnostic graph alone cannot establish
that helper's arithmetic. Operator-spelling and materialization controls above
remain refuted for their tested formulation.

The first source-grounded control holds the rank, positive last-axis index,
inputs and output target fixed and compares the current nil compilation
descriptor with explicit `MPSGraphOptimizationLevel0`. PyTorch sets level 0
on its graph execution descriptor; the native precompiled executable accepts
the descriptor at compilation instead. This only tests the optimization-level
factor, not equality of the graph and executable execution routes. SDK headers
describe level 1 as the property's default, but do not specify nil descriptor
semantics. Graph options and compilation optimization levels are distinct.

DoD: an actual official RMSNorm pass-through capture authenticates identical
input/gain bytes, F32 cast/squares and existing RMS stages; a native cogni-ml
probe compares these tensors and records the direct production statistics
output. A distinguishable first operation or explicitly bounded unknown must
remain after testing reduction geometry/order hypotheses. Keep the finite
`atol=rtol=2e-5` gate and raw-word diagnostics separate. Both runs use the
file-cache-aware 10% floor, 64 MiB swap-growth stop and 180-second deadline;
admit no more than 256 MiB native named buffers and a 2 GiB sampled process
cap for the isolated RMS probe. Recheck the budget before any broader probe.
Use one terminal wait per run with no pre-terminal activation reads or tracked
encode allocations. No full block prefix, full generation or production
arithmetic change is admitted by this diagnostic slice. Rollback is removal
of the opt-in diagnostic surface; preserve the existing prefix implementation.

The isolated official run `oracle-rms-only-block0-post-v2` consumes the saved
residual and only the 7,168-byte pretrained RMS gain, instantiating the official
norm rather than a decoder layer. Its eight actual pass-through stage captures
include the F32 cast and square. All five previously captured RMS intermediates
and the gained output reproduce the old block trace bit for bit. The fresh
manifest SHA256 is
`c068b70a4a2c0020eb2b42cb639cc82e33c457758c43a237b474eaf18b0b3ca7`.
A first launch rejected missing optional epsilon metadata before GPU setup;
configuration-bound epsilon validation corrected that harness incompatibility.
It is not numerical evidence.

`native-rms-reduction-block0-post-v1` localizes an earlier raw discrepancy:

| Stage | Differing F32 words | Maximum absolute difference |
| --- | ---: | ---: |
| BF16-to-F32 input | 0 / 4,587,520 | 0 |
| Separately bound square | 4,465,460 / 4,587,520 | 3.814697265625e-6 |
| Materialized-square mean | 670 / 1,280 | 2.2351741790771484e-8 |
| Variance plus epsilon | 670 / 1,280 | 2.2351741790771484e-8 |

All four F32 comparisons pass the unchanged finite gate. Raw discrepancy
already begins at the square operation; it cannot yet be attributed to mean
alone. The direct production helper's variance-plus-epsilon output is raw-bit
identical to the materialized-square diagnostic and to the explicit level-0
control. Its target is the actual production graph with no extra outputs;
direct production variance before epsilon is not separately exported.
Neither this finite pass nor the level-0 negative qualifies the full S1280
prefix. The level-0 test does not reproduce Torch's graph-encode route.

Pinned PyTorch source at runtime commit
`a1cb3cc05d46d198467bebbb6e8fba50a325d4e7` routes
`pow.Tensor_Scalar` through `binaryOpScalar` and `binaryOpTensor` to MPSGraph
`powerWithPrimaryTensor:secondaryTensor:`; native RMS uses `squareWithTensor`.
This is a source-grounded candidate for the square-value difference, not a
proof of MPSGraph's closed internal approximation mechanism. The existing
constant-exponent native power control remains a negative for its tested
formulation. A same-input control must also account for the scalar feed and
execution route before equating the two power graphs.

Admit a second isolated native control using the authenticated upstream square
panel as a prepared GPU input. Keep the materialized reduction graph's shape,
axis, two targets and execution route fixed. If mean and epsilon now match,
the square values account for their prior difference in this fixture; otherwise
reduction/epsilon remains a separate source. Eight retained outputs and one
additional input require 64,256,000 native named bytes, still within the same
256 MiB admission and 2 GiB sampled process cap. A separate upstream arithmetic
control compares `x.pow(2)` with `x*x` on that exact residual, preserving a sole
terminal wait; it is diagnostic alternate arithmetic, not a replacement for
the official RMSNorm capture or a production change.

`native-rms-reduction-block0-post-v2` tests that discriminator. With the same
native reduction graph fed the authenticated upstream squares, both variance
and variance-plus-epsilon have zero differing words across all 1,280 rows.
Their hashes are respectively
`0bef526fa01e4fb2883c62207777f698518d55cceb741d4b670d857f672811c3`
and `a0574cfd1144285003d5a939957d4efcc261bc5330c654e59c8ed1a036316d61`.
The native-square arm still reproduces the prior 670-row difference, and the
direct production target still equals that arm. Within this fixed-input test,
square values account for the observed statistics delta; no separate mean or
epsilon discrepancy is observed on identical square inputs. This establishes
an operation boundary, not a particular internal power implementation or a
mathematical proof about all possible reduction inputs.

The official RMS-only run and native v1/v2 guards all finish without a limit
violation or observed swap growth. Sampled process peaks are respectively
1,520,306,120 / 121,537,352 / 252,838,872 bytes; minimum cache-aware headroom
is 19,248,758,784 / 28,304,490,496 / 27,735,572,480 bytes. Each native run
reports one terminal wait, zero tracked encode allocations and no pre-terminal
activation reads. V2 has eight diagnostic readbacks after the wait and uses
64,256,000 named bytes. The guard is sampled and does not reserve memory or
certify an instantaneous peak. Timings here qualify no performance claim.

The separate upstream expression control `oracle-rms-pow-mul-block0-post-v1`
then reproduces both sides on the same uploaded BF16 residual. F32 `x.pow(2)`
and its last-axis mean are raw-bit exact to the official RMS capture; F32
`x*x` and its mean are raw-bit exact to the native square and mean hashes.
All 4,465,460 unequal squares are 1--2 ULP above the power result; the 670
unequal means are above it by at most four ULP. The control is alternate
arithmetic, not a substituted oracle. It has one explicit terminal wait,
131,344,384 estimated MPS bytes, a 1,456,620,936-byte sampled process peak,
26,218,217,472-byte minimum cache-aware headroom and zero observed swap growth.

This closes the fixed-input localization slice: cast is exact; square versus
power paths produce different F32 square values; materialized mean and epsilon
match on identical upstream squares. The production helper's target matches
the native-square arm, but the upstream-square input is fed to a separately
compiled materialized reduction graph. No test bypasses the square inside the
fused production graph. Neither that counterfactual nor the closed MPSGraph
power algorithm is established. At that slice the full S1280 prefix remained
rejected and production arithmetic was unchanged.

Focused validation passes 32 Python producer tests and seven Crystal fixture,
comparator, tamper/snapshot and budget examples, plus normal/cpu-only Crystal
no-codegen, strict C++ warnings, release link, formatting and diff whitespace
checks. Correlated MiniQuad review is ROBUST for the authenticated fixed-input
localization and lifecycle claims, with the fused-graph boundary above. It
establishes no performance or broad-input parity claim. The next falsifier is
a GPU-fed exponent power control matching Torch's scalar feed, distinguishing
it from the previously refuted constant-exponent formulation. Only after a
bounded native repair should block 0 and then the 26-block prefix be rechecked;
the larger checks need their own memory admission. Commands, source/runtime,
fixtures, binary/object pins, observed guards, retained failures and review
scope are indexed in
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024/logs/rms-reduction-receipt-v1.json`.
Refresh on changed scoped source, checkpoint/fixture/input, precision/geometry,
compiler/SDK/OS/device, execution route or buffer lifecycle, or evidence loss.

### GPU-fed exponent falsifier: exact materialized control

The isolated control keeps the authenticated S1280 residual and existing
square/reduction outputs. Compare two native F32 power panels with the same
primary-input rank, power operator, target layout, compilation and encode
route. The secondary operand is either constant 2.0 or a prepared F32 GPU
buffer containing 2.0, both shape `[1]` with F32 bits `0x40000000`. Apply the
same materialized mean/epsilon graph to each panel. This varies only the
effective-F32 operand source. Pinned Torch lowering wraps integer 2, feeds
an Int64 `[1]` scalar and casts it to the common F32 dtype before power.
A third diagnostic arm reproduces that operand/cast topology with a prepared
eight-byte Int64 buffer. It retains the native precompiled-executable route;
it does not reproduce Torch's graph-level encode API. An
`aten::pow.Tensor_Scalar` trace alone does not establish the MPS lowering.

Prediction: if a dynamic exponent is sufficient in this native route, its
square/mean/epsilon hashes will match the authenticated official RMS capture.
The matched constant arm distinguishes an operand-source effect from the
historical negative in a different fused graph. If both arms still match the
native square, the Int64-fed/cast arm distinguishes the remaining operand
topology from an execution-route difference. If all miss, pivot rather than
changing tolerances or asserting a closed-backend cause. An exact panel alone
does not qualify a fused production replacement or the full prefix.

The two panels and four scalar-row outputs add 36,720,640 bytes, plus a
four-byte exponent buffer. The Int64-fed panel, two scalar-row outputs and
eight-byte exponent add another 18,360,328 bytes: 119,336,972 native named
bytes in total at S1280/D3584 (316 bytes at the tiny S2/D4 CPU geometry).
The one-byte-short budget is an explicit rejection case.

Admit only this opt-in diagnostic extension: at most 256 MiB named buffers,
a 2 GiB sampled process cap, file-cache-aware 10% floor, 64 MiB swap-growth
stop and 180-second deadline. Preserve one terminal wait, no pre-terminal
activation reads and zero tracked encode allocations. DoD is a fresh guarded
report with all three operand routes, authenticated comparisons, unchanged
baseline hashes, and CPU budget/fixture/comparator checks. Rollback removes
the diagnostic extension; production RMS arithmetic remains unchanged. No
full block, 26-block prefix or generation is admitted by this small budget.

Observed `native-rms-gpu-exponent-block0-post-v1` exits zero under that guard:

| Prepared native power operand | Square raw differences | Mean raw differences | Mean + epsilon raw differences |
| --- | ---: | ---: | ---: |
| F32 constant `[1]`, value 2 | 4,465,460 / 4,587,520 | 670 / 1,280 | 670 / 1,280 |
| F32 GPU buffer `[1]`, value 2 | 0 / 4,587,520 | 0 / 1,280 | 0 / 1,280 |
| Int64 GPU buffer `[1]`, value 2, cast to F32 | 0 / 4,587,520 | 0 / 1,280 | 0 / 1,280 |

Both fed arms reproduce the authenticated official square, variance and
variance-plus-epsilon SHA256 values exactly. The constant arm reproduces the
old native square/mean/epsilon hashes. All eight baseline output hashes are
unchanged, including the direct production target. The F32 matched pair is
enough to distinguish the operand source on this input; the Int64 cast is not
required for this observed parity. The native precompiled-executable route is
therefore capable of matching these materialized upstream stages. This is
not a proof of MPSGraph's internal constant folding or power implementation,
nor of equivalence across other inputs or fused target graphs.

The report preserves unchanged `atol=rtol=2e-5`; every finite comparison
passes. Aggregate raw parity remains false because the negative constant and
production-square controls intentionally remain in the report. The plan has
12 prepared executable steps, 119,336,972 named bytes, one terminal wait,
17 post-wait output reads, zero tracked encode allocations and no
pre-terminal activation reads. The sampled process peak is 369,050,704 bytes;
minimum file-cache-aware headroom is 19,853,262,848 bytes, with no guard
violation or observed swap growth. Sampling does not establish an
instantaneous peak or reserve memory.

Eight focused Crystal fixture/comparator/budget examples pass, along with
normal/cpu-only no-codegen checks, strict Clang compilation, release linking,
formatting and diff checks. This slice changes only the opt-in diagnostic;
full RMS normalization/gain and block 0 are not executed. The next falsifier
is a bounded GPU-fed-exponent replacement in the fused production-statistics
target on the same residual, then normalization/gain parity, before any
separately admitted block/prefix run. The dynamic exponent must be prepared,
retained and immutable across encoding, with no per-step allocation or host
wait. The exact operand/cast topology is source-pinned; the actual Torch
scalar allocator branch and closed compiler internals remain unobserved.
Correlated MiniQuad review is ROBUST for this fixed-input discriminator and
bounded lifecycle only. Source/build/fixture/run pins, observed guard and
review limits are indexed in
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024/logs/rms-gpu-exponent-receipt-v1.json`
and `.landmark/evidence/2026-10-25-qwen25-rms-gpu-exponent.json`. Refresh on
scoped source, input/fixture, geometry/precision, toolchain/SDK/OS/device,
execution/lifecycle changes, or evidence loss/hash mismatch.

### Connected RMS and gain falsifier: exact saved-input candidate

Admit a bounded three-arm native diagnostic on the same authenticated S1280
post-attention residual and BF16 gain vector. Arm A keeps the production
single-target square/mean/epsilon graph. Arm B changes only its power operand
to a prepared immutable F32 GPU buffer `[1]`, value 2. Both continue through
the same precise Metal root/normalization kernels and BF16 gain graph, with
distinct retained outputs for epsilon, root, normalized F32/BF16 and gain.
Arm C uses B's single-target statistics helper with the production in-place
scalar scratch and non-trace BF16 normalization kernel before the same gain.
It checks whether observation stores or scalar aliasing affect the result.

The falsifier is per-stage upstream comparison at unchanged `atol=rtol=2e-5`,
with raw-word differences reported separately. Arm A is a retained negative
control; B and C must be assessed individually, including their common
BF16 outputs. Load only the norm vector, authenticating the bytes against
the fixture's tensor digest before uploading them. Do not upload block weights.
No intermediate square panel or statistics trace target is added.

This CAUTION change is opt-in; production block arithmetic stays unchanged.
Its rollback removes the new diagnostic entry point and runner. The 13 outputs
and three inputs require 100,958,212 named bytes at S1280/D3584 (228 bytes at
S2/D4). Reject a one-byte-short budget before allocating Metal buffers. Keep
the 256 MiB named cap, 2 GiB sampled process cap, file-cache-aware 10% floor,
64 MiB swap-growth stop and 180-second deadline. DoD: focused CPU validation,
strict native compilation, and a fresh guarded connected report preserving
12 ordered steps, one terminal wait, no pre-terminal activation reads and
zero tracked encode allocations. A failure must retain its report. This
slice admits no block/prefix execution or full generation; exact local parity
does not establish broad-input equivalence. Refresh on scoped source, fixture,
precision/geometry, toolchain/device, execution/lifecycle or evidence changes.

Observed `native-connected-rms-block0-post-v1` exits zero. The candidate's
single-output statistics graph feeds the same precise root and normalization
kernels and BF16 gain helper without an intermediate activation readback.

| Retained output | Legacy trace raw differences | GPU-fed trace raw differences | GPU-fed production-style raw differences |
| --- | ---: | ---: | ---: |
| Mean + epsilon F32 (1,280 rows) | 670 | 0 | Overwritten by root |
| Inverse-root F32 (1,280 rows) | 371 | 0 | 0 |
| Normalized F32 (4,587,520 values) | 1,276,635 | 0 | Not retained |
| Normalized BF16 (4,587,520 values) | 50 | 0 | 0 |
| Gained BF16 (4,587,520 values) | 28 | 0 | 0 |

Both candidate arms pass the unchanged finite `2e-5` gate and raw-bit gates
individually. Their three common outputs are also raw-bit identical. Legacy
BF16 normalization fails the finite gate at 50 values; gain fails at 27
values while 28 words differ. A tolerance pass is therefore still distinct
from raw parity. Its epsilon hash reproduces the retained production control.
The selected gain's exact 7,168 copied bytes match the fixture's tensor digest;
the bounded reader does not map, read or upload the full shard.

The plan executes 12 ordered steps using 100,958,212 named bytes, one terminal
wait, 13 post-wait output reads, zero tracked encode allocations and no
pre-terminal activation reads. The guard observes a 393,217,104-byte process
peak, minimum file-cache-aware headroom of 19,062,308,864 bytes, no swap growth
and no limit violation. The floor stays 10%, with the same 2 GiB process cap,
64 MiB swap stop and 180-second deadline. These are sampled observations,
not instantaneous peak or memory-reservation guarantees.

Fourteen focused Crystal tests pass, covering exact/one-byte-short budgets,
authenticated fixture loading, selective gain mutation/extent rejection and
finite/raw comparator separation. Normal/cpu-only no-codegen, strict Clang,
release linking, formatting and diff checks pass. The retained source and
binary pins identify the observed run; pre-existing companion objects have
opaque source ancestry and this is not a hermetic rebuild. No speedup claim
is made from the correctness run.
Correlated MiniQuad review is ROBUST for this fixed-input connected diagnostic
and its lifecycle. The F32 feed supplies Torch's post-cast value, without
reproducing its Int64 feed/cast topology. The caller-supplied RMS manifest
digest is preserved in the pinned launch record.

The subsequent opt-in block integration above closes the both-RMS block-0
falsifier on the saved input. The subsequently qualified prefix recheck above
closes that saved-S1280 continuation. This earlier connected diagnostic's
source/build/input/run pins and review bounds are indexed in
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024/logs/rms-connected-receipt-v1.json`
and `.landmark/evidence/2026-10-25-qwen25-connected-rms.json`.

### Implemented boundary and observed checks

`Qwen25VLWeights` validates the pinned config, shard index, BF16 tensor shapes,
offset bounds and selected inventory before uploading source bytes. The probe
also requires the upstream tensor digests and hashes the same bounded chunks
copied into each GPU weight buffer, rejecting a mismatch before encoding. The four
downloaded safetensors shards cover the embedding and 312 selected block
tensors; their SHA256 values match the checkpoint's LFS identities. The full
embedding table is not resident: preparation selects unique raw BF16 rows,
and `Qwen25VLEmbeddingPlan` gathers the sequence on the GPU. Duplicate and
reordered IDs preserve the exact BF16 words.

`Qwen25VLPrefixPlan` shares the 16 non-output stages, seven BF16 scratch
bindings and one F32 RMS-statistics binding across ordered blocks, retaining
a distinct hidden output for each block. Its aggregate admission includes
borrowed weights and source buffers. At 1280 tokens, the stage/scratch
workspace is 549,852,160 bytes instead of 26 separate 320,476,160-byte
workspaces; this is a static byte count, not a
measured full-prefix footprint. The workspace enforces block order and rejects
duplicate encoding, another in-flight chain and release before the terminal
drain. Host calls remain serial.

The official producer uses all 26 resident pretrained layers, GPU embedding,
official RoPE/masking and official SDPA, captures 27 BF16 hidden states, then
performs one explicit terminal synchronization and packed readback. The native
runner connects GPU gather directly to the 26-block chain, records zero tracked
named-buffer allocations during encoding and requires exactly one terminal
wait. It makes 27 diagnostic readbacks only after that wait; there are no
pre-terminal activation readbacks. The native fixture consumer requires a
caller-supplied trusted manifest SHA256, captured from the known successful
guarded oracle execution. Internal provenance fields alone are not an
authentication boundary. Manifest parsing uses the bytes whose digest was
checked; comparisons use a bounded authenticated sidecar snapshot, and GPU
embedding source rows are uploaded from the exact packed bytes verified.

Observed bounded checks under
`/opt/mldata_ssd/cogni-ml/qwen25-pretrained-20261024`:

- `logs/native-rms-regression-v3.*`: 42 native examples pass without skips,
  including GPU gather,
  ordered shared-workspace execution, lifecycle/admission falsifiers and
  comparator/digest attacks. A first/last tensor hash mismatch restores live
  MetalBuffer counts and bytes to the pre-upload baseline. The four earlier
  constructor-weight fixtures
  pass all 102 standalone stage comparisons and 70 shared-workspace
  comparisons with zero absolute error. Separate pretrained prefix evidence
  is listed below. The raw-word comparator distinguishes signed zero from
  numerical equality while leaving the finite tolerance gate unchanged.
- `logs/cpu-qwen-rms-final-v2.out`: 92 CPU tests pass: 22 constructor-oracle,
  13 pretrained-prefix-producer, 13 block/RMS-producer and 44 memory-guard
  checks. Strict native syntax checking
  with `-Wall -Wextra -Werror`, Crystal formatting and the release probe build
  (`logs/build-rms-production-v1.*`, refreshed probes in
  `logs/native-rms-regression-v3.*`) pass. The producer validates derived `head_dim`
  and all config metadata before allocating the resident prefix; pinned
  Transformers configuration omits an explicit `head_dim` attribute.
  `logs/guard-floor-final-v1.*` passes 44 guard CPU tests, including exact
  fractional floor arithmetic and fail-closed malformed telemetry.
- `logs/resource-two-block-v1.*` and `logs/resource-two-block.json`: a separate
  official two-pretrained-block resource probe completes with one explicit
  terminal synchronization. Its final self-reported footprint is
  1,451,411,328 bytes. It establishes neither numerical parity nor the
  26-block peak.
- `logs/oracle-s4-v9.*` and `logs/native-prefix-s4-raw-v1.*`: the full selected
  pretrained prefix passes at four tokens. The native embedding and all 26
  block outputs have zero absolute error and zero differing BF16 words, with
  one terminal wait, zero tracked encode
  allocations and zero pre-terminal activation readbacks. The trusted
  `fixtures/prefix-s4.json` digest is
  `c399cc50ef8a2435998fbd8c5dfc4121b4c414d538e33a023e4cd51225ca2398`.
- `logs/oracle-s32-v1.*` produces the official 32-token right-padded prefix
  with position offset 7. Its trusted `fixtures/prefix-s32.json` digest is
  `40c665e0ce5903f43411d39a46379e8dda64ddbe41bd058f15552c877ccba96e`.
  The historical `logs/native-prefix-s32-v1.*` compares states 0--24 with
  zero absolute error, then
  rejects state 25 (892 failures of 114,688, maximum absolute error 0.125)
  and state 26 (1,779 failures, maximum absolute error 0.125). It preserves
  one terminal wait, zero tracked encode allocations and zero pre-terminal
  activation readbacks. The repaired `logs/native-prefix-s32-rms-v1.*`
  passes all 27 captured states with zero absolute error. Fresh
  `logs/native-prefix-s32-raw-v1.*` also reports zero differing BF16 words
  in every state. The unchanged finite gate is not widened; numeric `passed`
  and the diagnostic raw-word equality count are separate signals.
- The fresh native raw-word S4/S32 runs peak at 12,468,626,792 and
  12,527,560,256 sampled group bytes, with minimum cache-aware headroom
  20,975,714,304 and 20,342,177,792 bytes and pressure-free percentages
  at least 63. Both guards exit zero with no observed swap growth. All 54
  state comparisons across the two runs have zero differing BF16 words.
  Each run has one terminal wait, zero tracked encode allocations,
  zero pre-terminal activation reads and 27 post-terminal diagnostic reads.
- Both successful official producers use a 12.5 GiB sampled process cap,
  explicit 10% floor, 64 MiB swap-growth stop and 600-second deadline. S32
  peaks at 13,320,652,576 sampled group bytes, below the 13,421,772,800-byte
  cap, with zero observed swap growth. This small sampled margin is not a
  reservation or a hard peak guarantee. At 64 GiB, startup requires at
  least 18.9 GiB cache-aware headroom for that budget and floor.
- Historical oracle attempts v1--v5 and v7 stop before child execution;
  v6 fails checkpoint path preparation and v8 fails post-forward config
  metadata access. Earlier `native-gather-v3`, `native-final-v5` and
  `build-native-v5` are harness/compile failures. They establish no numerical
  parity. The initial CPU final-v1 attempt lacked the pinned Torch/Transformers
  interpreter, and native-rms-regression-v1 omitted two GPU opt-in flags;
  successful reruns above replace those incomplete checks.

### First pretrained mismatch and repair

The official standalone layer-24 producer reproduces S32 captured state 25
bit exactly before admitting its intermediate fixture. Its pass-through
TorchDispatch observer captures the actual official RMS operations rather
than substituting a copied formula. Trusted `fixtures/block24-s32-v3.json`
SHA256 is
`4aaad9f784d86afbea6cd29868869af8889bd3b4b4bf907b0066d192d5b81960`.

`logs/native-block24-v2.*` isolates the first discrepancy: MPSGraph inverse
square root differs by one F32 ULP in eight rows. Nineteen normalized BF16
words in row 30 cross a rounding boundary, then 16 weighted RMS values and
892 layer-output values fail the unchanged gate. MPSGraph square root plus
division produces the same discrepancy (`native-block24-sqrt-v1`), rejecting
that graph-only substitution as a fix. A remaining one-ULP variance/add-epsilon
difference in row 24 does not cross the downstream tested rounding boundary;
F32 moments are not claimed bit exact.

The independent precise Metal branch (`native-block24-precise-v1`) makes
inverse root, normalized F32 and normalized BF16 bit exact while preserving
the failing legacy production path as a causal control. Production now uses
MPS moments followed by safe-math Metal `1.0f / precise::sqrt(x)`, F32 scaling
and explicit BF16 round-to-nearest-even, then the existing BF16 gain multiply.
Both RMS norms use this path. The added F32 scratch is only four bytes per
token, shared between the two norms and all ordered blocks. At 32 tokens this
adds 128 named bytes. There are 19 prepared graph/Metal steps per block,
compared with 15 in the initial implementation; no new host wait or tracked
encode allocation is added.

`logs/native-block24-rms-production-v1.*` exits successfully: all 17 declared
BF16 stages and the layer output have zero absolute error. Production scratch
matches the independent precise trace bit exactly; the legacy graph trace
remains a separately reported negative control and is not a production gate.
Single correctness-probe timings do not establish a performance comparison.

Next falsifier: before extending to Hy-Nomas's 1280-token prompt, declare a
separate bounded long-prefix memory plan and compare all retained hidden
states against the official decoder on identical tokens/masks/positions.
Preserve one terminal wait, zero tracked encode allocations and no
pre-terminal activation readback. Tokenization, prompt templates, cropping,
DiT integration and video generation remain separate boundaries.

Use the pinned interpreter under
`/opt/mldata_ssd/cogni-ml/drama3-fish-reference-20261015/venv/bin/python`,
`PYTORCH_ENABLE_MPS_FALLBACK=0`, offline checkpoint access and SSD-backed
`TMPDIR`/compiler caches. The bounded oracle command is:

```sh
python3 -B examples/support/wan_memory_guard.py \
  --memory-accounting file-cache --runtime-floor-pct 10 \
  --max-footprint-mib 12800 \
  --max-swap-growth-mib 64 --timeout-sec 600 --log FRESH_GUARD.jsonl -- \
  PINNED_PYTHON -B examples/support/qwen25_pretrained_reference.py \
  --checkpoint-root CHECKPOINT_DIRECTORY \
  --checkpoint-receipt CHECKPOINT_DOWNLOAD_RECEIPT \
  --output NEW_PREFIX_MANIFEST \
  --token-ids 151644,872,151645,0 --attention-mask 1,1,1,0 \
  --position-offset 0
```

Build with
`make EXTERNAL_BUILD_DIRECTORY/qwen25_pretrained_prefix BUILD_DIR=EXTERNAL_BUILD_DIRECTORY`.
The native CLI requires
`--checkpoint-root`, `--reference`, `--reference-sha256`, a fresh `--report`
and a positive `--max-named-bytes` admission budget. Preserve the guard's
explicitly selected 10% runtime floor, 64 MiB swap-growth stop and deadline
for both commands; omission of the floor flag retains the default 15%.
For the focused native suite, create a fresh existing report directory first,
enable `COGNI_QWEN25_EMBEDDING_GPU=1` and `COGNI_QWEN25_WEIGHTS_GPU=1`,
and supply the four trusted constructor-weight fixtures as in the prior
qualification. External `logs/prefix-continuation-receipt-v3.json` indexes the
current source, fixtures, binaries, guard outcomes, rejected controls and
bounded qualification. The v2 receipt is historical and precedes the floor
override and current S4/S32 results. An index is not itself proof of its claims.

## Contract and source

Hy-Nomas source revision `911cea6458a89c2663513ff7099561ec190dd491`
selects Qwen/Qwen2.5-VL-7B-Instruct with BF16 precision, right padding,
attention masks, 1280 text tokens, `hidden_state_skip_layer=2` and
`apply_final_norm=false`. The Hugging Face hidden-state tuple contains the
embedding output followed by decoder-layer outputs, with the last output
normalized by the stack's final RMSNorm. For 28 layers,
`hidden_states[-3]` is therefore the output of block 25. Only blocks 0--25
are needed; neither the last two blocks nor the final RMSNorm, vision tower
or language-model head contribute to that selected state.

This slice consumes resident BF16 embeddings, additive causal/padding mask,
F32 positions and rotary inverse frequencies, and resident BF16 decoder
weights. It reuses CogniQwen's Metal buffers and the engine's serial MPS
command chain. Qwen3.5's hybrid forward pass and weights are not substitutes
for Qwen2.5-VL's biased Q/K/V, split-half rotary embedding and grouped-query
causal attention. Quantized CogniQwen kernels remain a later qualification.

All neural operations execute on GPU. Preparation may upload source bytes,
mask values and positional constants. Encoding must neither allocate named
Metal buffers nor read activations nor wait for GPU completion. Distinct
blocks are linked through resident output/input buffers in one command
chain. The caller owns the sole terminal wait and diagnostic readback.
Prepared plans and their bound buffers remain retained until that wait.
Serial attempts to reuse a plan on another in-flight chain and premature
release fail closed. Concurrent host calls are unsupported; there is no lock.
Borrowed bindings must not be overwritten or released in a way that races
scheduled consumers before the terminal drain. An ordered earlier GPU producer
on the same chain may populate the next block's input. Generic MetalBuffer
write/release APIs do not enforce this caller precondition. The standalone
diagnostic block retains every declared stage with a per-plan byte budget.
The active prefix adapter adds shared scratch and aggregate named-buffer
admission; it does not account for MPS-internal allocations, which remain
subject to the external sampled process guard. Its bounded pretrained
qualification is described above; longer prefixes remain unqualified.

## Qualification and falsifiers

The unchanged finite gate is `abs(error) <= 2e-5 + 2e-5*abs(reference)`.
Compare input normalization, biased Q/K/V, rotary outputs, attention output,
output projection, both residuals, post-attention normalization and all FFN
stages. The oracle must execute the official Transformers decoder on MPS;
locally copied equations cannot establish upstream parity. Record exact
source hashes, runtime versions, dtype, mask and attention implementation.
BF16 storage alone does not prove equivalence of reduction order, fused
operations or SDPA. A mismatch must stop promotion and identify the first
divergent stage; never widen tolerances to make a path pass.

Use one-layer and two-layer fixtures with different successor weights, a
nontrivial position offset, right padding, and a mutation that proves the
comparator rejects an incorrect value. A small synthetic model proves only the
tested geometry and data. One four-token block at 3584/18944/28Q/4KV/128-head
geometry now passes too. The pretrained S4/S32 prefix qualification above
extends this constructor-weight evidence. Longer sequences, tokenization,
prompt templates, cropping and integration with the DiT remain separate
qualification boundaries. The admitted masks are right
padded and every query row has at least one allowed key. The finite BF16 0/min
native bias is qualified only for that class; arbitrary and fully masked rows
are not qualified.

CAUTION: numerical/lifetime changes are isolated to new opt-in files. The
strongest failure modes are crossed BF16 rounding boundaries, incorrect GQA
ordering, accidentally using one block's weights twice, and early freeing of
in-flight resources. Guards validate geometry, exact bindings, storage,
device, lengths and disjointness before encoding. Rollback is removal of this
opt-in adapter and its focused build target; existing model paths stay intact.

DoD: run the Python oracle tests and guarded native specs using
`examples/support/wan_memory_guard.py --memory-accounting file-cache`, a
bounded process footprint, bounded swap growth, and the default 15% reserve
or an explicitly authorized per-run floor (10% for the pretrained probes).
Expected signals: every declared intermediate passes the finite gate,
invalid/alias bindings fail before encoding, no named-buffer allocation is
observed during encoding, and wait count changes from zero to one only at
the terminal drain. Record tested scope and residual mismatches below.
No full generation, Studio or full-model allocation is admitted by this slice.
Temporary artifacts and compiler caches live under
`/opt/mldata_ssd/cogni-ml/qwen25-hy-nomas-20261024`.

## Rejected numerical route

The first one-graph candidate failed the unchanged gate on a tiny official
MPS BF16 fixture: input RMSNorm and biased Q/K/V matched exactly, but Q RoPE
failed 16/96 elements (maximum absolute error 0.0078125). Both rotary cos/sin
outputs matched exactly, rejecting trigonometric values as that first cause.
A two-block fixture with non-unit RMS gains diverged earlier, at input RMSNorm
(23/96 elements, maximum 0.0078125). Explicit F32--BF16--F32 cast round trips
inside the same graph did not change either result. Such casts are not an
empirically valid precision fence in this tested compilation. Evidence:
`logs/native-roundtrip.out` and `logs/native-roundtrip.jsonl` under the artifact
root. Materializing source BF16 operation boundaries in caller-owned buffers
between asynchronous executables eliminated the RMSNorm and RoPE mismatches.
The next first divergence was attention (29/96 elements, maximum 0.00390625),
recorded in `logs/native-materialized.out`. An all-F32 QK/softmax/PV graph does
not reproduce the general BF16 MPS SDPA path. Matching its typed graph sequence
closed that mismatch without a host wait or wider tolerance. Both padded test
geometries dispatch to PyTorch's general MPSGraph SDPA, not its vector shader.
That initial block encoded 15 prepared executables, retaining seven BF16
boundary scratch buffers as part of its admitted workspace. The later precise
RMS repair described above adds four prepared steps and one shared F32
statistics scratch binding.

## Observed qualification

The official oracle uses PyTorch 2.8.0 and Transformers 4.57.3 on MPS BF16,
with CPU fallback disabled. Hy-Nomas declares Transformers 4.57.1; the producer
records the mismatch and hashes the installed source. Its selected Qwen2.5-VL,
RoPE, activation, mask and SDPA math is byte-equivalent to the pinned 4.57.1
source; no equivalence for unrelated library code is claimed. Geometry comes
from Qwen/Qwen2.5-VL-7B-Instruct revision
`eb6ba55b40ada64977147b52e50a4145ee89ac4f`. These are official constructor
weights, not pretrained checkpoint weights. Every fixture uses four tokens,
mask `[1,1,1,0]`, and position offset 7.

| Fixture | Layers | Dimensions D/FF/Q/KV/HD | RMS gains | BF16 stage comparisons |
| --- | ---: | --- | --- | ---: |
| `tiny-1layer.json` | 1 | 24/48/2/1/12 | unit | 17 |
| `tiny-2layer.json` | 2, distinct weights | 24/48/2/1/12 | unit | 34 |
| `tiny-2layer-nonunit.json` | 2, distinct weights | 24/48/2/1/12 | nonunit | 34 |
| `real7b-s4.json` | 1 | 3584/18944/28/4/128 | unit | 17 |

All 102 stage comparisons pass with zero absolute error at the unchanged
finite gate. Those constructor-stage reports do not record raw-word equality;
the fresh pretrained prefix reports above do. Each request reports one terminal
wait and zero tracked named
MetalBuffer allocations during encoding. The two-block tests pass the exact
resident output object to the successor without CPU readback. Invalid bindings,
overlap, workspace budget, early release and sequential second-chain reuse are
exercised; the comparator rejects a one-ULP mutation and nonfinite values.
Python fixture/sidecar tests pass 22 examples. Crystal CPU configuration,
sidecar and comparator checks pass four examples. Native compilation passes
`-Wall -Wextra -Werror`. Correlated MiniQuad review: ROBUST for this bounded,
serial diagnostic scope, not for arbitrary inputs or production prefix use.

The guarded real-geometry oracle peaks at 2,565,818,240 sampled process-group
bytes; the native Makefile run peaks at 829,557,304 bytes. Minimum cache-aware
headroom is 23,983,620,096 and 26,133,790,720 bytes respectively; memory-pressure
free percentages stay at least 71 and 75. Swap growth is zero. These are sampled
observations, not opaque MPS peak bounds. The external 3 GiB process-group cap,
15% reserve, 64 MiB swap-growth stop and 600-second deadline remain intact.
No speedup or full-generation throughput claim follows from these tests.

Evidence under the artifact root: `logs/native-sdpa-general-v2.*` (one-layer and
nonunit two-layer gate), `logs/native-real7b.*` (Makefile target: six examples,
including unit two-layer and real geometry), and `logs/real7b-s4-oracle.*`.
Source, fixture, bridge-object and runtime pins are indexed in
`logs/final-receipt-v1.json`. The Crystal consumer treats environment-supplied
fixtures as trusted input; it does not authenticate their upstream provenance.
Verify their recorded hashes before using a rerun as an oracle qualification.
The producer pins and validates source/runtime identity and sidecar contents.
The final fresh rebuild, `logs/native-final-clean-v2.*`, passes all eight
Crystal examples and all four fixtures together. The earlier clean-run attempt
used a mistyped fixture directory and failed file lookup; it is retained as a
harness error, not numerical evidence. Crystal's Makefile spec runner deletes
its temporary executable after exit, so the receipt pins its compiled bridge
objects, source and command rather than claiming a retained executable hash.

Reproduce only under the guard with SSD-backed TMPDIR/CRYSTAL_CACHE_DIR:

```sh
python3 -B spec/qwen25_hy_nomas_reference_spec.py
python3 -B examples/support/wan_memory_guard.py \
  --memory-accounting file-cache --max-footprint-mib 3072 \
  --max-swap-growth-mib 64 --timeout-sec 600 --log FRESH.jsonl -- \
  env COGNI_QWEN25_REFERENCES=TRUSTED_FIXTURE_PATHS \
      COGNI_QWEN25_REPORT_DIR=EXTERNAL_REPORT_DIRECTORY \
  make spec_qwen25 BUILD_DIR=EXTERNAL_BUILD_DIRECTORY
```

`COGNI_QWEN25_REFERENCES` is a colon-separated list of the recorded fixture
paths. Oracle production uses the pinned interpreter and exact arguments in
the receipt; the real-geometry BF16 sidecar is 466,875,424 bytes including
weights, source inputs and expected stages. No fixture or generated binary is
added to Git. The next discriminator is the admitted pretrained 26-block
oracle/native comparison described above, followed by the Hy-Nomas
prompt/cropping bridge.

Evidence decays when upstream source, source dtype, attention policy,
geometry, compiler, OS/MPS runtime, weights or buffer lifecycle changes;
refresh the corresponding oracle and guarded native tests before promotion.
