# Qwen QBit Cache Frontier

Document status: active design-sealed slice

Current frontier: a default-off p7 transport/restore experiment for recurrent
Qwen cache state plus a bounded ClickHouse HTTP storage/read client and an
exact-full-prompt native-runtime route. A non-session cache hit may now restore
directly into configured adaptive GPU KV owners; misses retain the ordinary
Float32 owner. Session requests also retain Float32 unless both an adaptive
resident-KV map and the explicit `QWEN35_QBIT_ADAPTIVE_SESSION=1` gate are set.
It may emit revision-0 ClickHouse Native blocks whose QBit column is already
bit-transposed, validate an ordered multi-block Native response, decode logical
records that cross response-block boundaries directly into prepared Metal state
buffers, attach exact KV from a separate bounded artifact, compare a complete
QBit cache-hit corridor against the matched INT8 artifact in alternating order,
measure one isolated temporary MergeTree, and construct a versioned admission
envelope for the split recurrent-QBit/exact-KV state. The runtime route is
strictly additive: it may read or write only when explicitly configured and it
must preserve the existing local prompt-cache and ordinary-prefill behavior on
ordinary misses, rejections, cache transport failures, or rejected writes. A
caller-selected explicit checkpoint is rollback authority, so its rejection,
admission failure, or transport failure aborts instead of silently selecting a
different state. System and Metal failures also abort rather than risk a second
heavy attempt.

Active slice: widen the default-off runtime route from exact-full-prompt lookup
to longest-token-prefix restore plus suffix replay. New writes may additionally
publish session checkpoints. A checkpoint is either a full QBit/exact-KV anchor
or a cumulative exact token delta rooted at one immutable anchor. Delta restore
loads the anchor once and deterministically replays the stored token suffix;
it does not add quantized recurrent-state differences. Explicit rollback may
select an earlier checkpoint only when its session identity and token prefix
match the caller-authoritative transcript. A fresh ordinary-session prefill may
also retain a recurrent-only rollback point near the assistant boundary, reuse
the exact live KV prefix, and replay only the re-tokenized boundary suffix when
materializing its first exact anchor.

The default-off session route may additionally defer only the CPU QBit encode
and manifest-last ClickHouse publication of a newly materialized full anchor to
one bounded scheduler-backed execution context. The request still captures one
immutable host snapshot while it owns the exact Metal state, returns the
preallocated checkpoint identity with an explicit pending marker, and never
queues a second snapshot. The next generation and runtime close wait for the
pending publication before consuming or releasing its durability boundary.
Small token-delta checkpoints remain synchronous.

The explicit `QWEN35_QBIT_ADAPTIVE_SESSION=1` gate now composes the adaptive
resident-KV map with synchronous session restore and publication. The first
system-plus-user message boundary becomes one immutable compact root; the
completed assistant boundary and later checkpoints are exact cumulative token
deltas. A hit restores that root directly into adaptive Metal owners and
replays the exact suffix in causal chunks no larger than 64 tokens. A remainder
of one is split across the first 65 tokens so replay cannot accidentally enter
the unsupported single-token Float32 decode route. A suffix containing only one
token is rejected explicitly because no causal multi-token chunk can represent
it. Adaptive session anchors do not use the asynchronous Float32 snapshot
writer; enabling both routes fails before inference.

This is deliberately simpler than periodic anchor renewal. The checkpoint
schema admits at most 4096 cumulative exact-delta tokens and depth 8; each
request is further limited by `max_seq`, root-prefix length, and its generation
budget. Reaching either schema bound fails closed because rebuilding a compact
root from already approximated recurrent state would compound error. A future
renewal path must recompute from caller-authoritative transcript tokens before
it can widen this boundary.

The cache/replay promotion card for this slice is:

- Window: a session request with both an adaptive resident-KV map and the
  explicit adaptive-session gate.
- Transport corridor: transcript and exact tokens -> immutable ClickHouse
  anchor -> adaptive Metal owners -> bounded exact suffix replay -> new delta
  checkpoint.
- Legal move: replace persistent Float32 KV ownership only; live DeltaNet state
  remains uncompressed and recurrent state is compressed only for persistence.
- Boundary safety: model, tokenizer, template, ABI, artifact codec, checkpoint,
  transcript boundary, token prefix, and manifest certificates all remain
  mandatory and fail closed.
- Lexicographic potential: persistent Float32 KV owners, peak admitted source
  bytes, replayed suffix tokens, then end-to-end wall time.
- Recompute safety: only the initial root is computed from exact transcript
  tokens; no arithmetic state delta and no restored approximate recurrent state
  becomes a new anchor source.
- Dual frame: exact token/checkpoint history is the rollback authority while
  adaptive QBit is only the memory representation of full-attention KV.
- Local certificate: clean seed/restore/continue/rollback counters plus top-1,
  top-2, meaning, ECS, ownership, memory-pressure, and swap observations.

Guarded Qwen3.8-27B Q4_K_M evidence on Apple M2 Max (2026-08-26) closed the
512- and 2048-token seed/baseline/restore/continue/rollback lifecycle. At
`max_seq=2048`, the compact root had 286 tokens, the longest measured suffix
replay had 1657 tokens, and the continued prompt reached 1943 tokens. The
baseline and seed both emitted token ids `[66793, 12, 21]` (`checkpoint-6`);
cold restore emitted `restored-7`, continuation emitted `checkpoint-7`, and
rollback emitted `rollback-3`. Every hit reported sole adaptive ownership and
exact token conservation. Checkpoint writes took 3.714-16.005 ms. The largest
observed RSS was 1,485,750,272 bytes, peak footprint was 2,243,942,304 bytes,
and every guarded phase reported zero swaps.

The 2048 ClickHouse tables contained one p4/BF16 adaptive root, ten unique
checkpoint rows, maximum depth 7, and no second full anchor. Active compressed
parts totalled 44,264,745 bytes: 34,557,019 recurrent, 9,698,964 KV, and about
9 KiB of manifest, prefix, and checkpoint metadata. The longest cold restore
took 14,981.473 ms versus 13,291.594 ms for the matched full prefill, so this
slice establishes compact persistence and rollback reliability, not a long-
suffix speedup. In the aligned 829-token quality probe, exact and adaptive both
emitted `Their sum is 95.` and reached EOS: top-1 was `8/8`, exact top-1 was in
the adaptive top-2 at `7/7`, and token ECS mean/minimum was `1.0/1.0`. Ranked
and unordered top-2 agreement was only `9/14` with maximum winning-logit delta
`1.2506447`; the representation preserves this measured greedy trajectory and
meaning but is not logit-lossless.

Separate guarded 3072- and 4096-token stages then closed the same lifecycle
with a 35% free-memory floor, a 24 GiB process-tree cap, and zero swaps. In the
4096 stage the compact root contained 507 tokens. Seed, cold restore,
continuation to a 3986-token prompt, and rollback all retained exact
transcript/checkpoint boundaries, sole adaptive ownership, and zero failure
counters. The store held one compact root and eleven immutable checkpoint
rows, including two legal branches, with maximum depth eight and no replacement
full anchor. Active ClickHouse parts used 51,698,587 compressed bytes
(49.30 MiB) for KV, recurrent state, manifests, prefix index, and checkpoints.

This wider stage is a compactness result, not an acceleration result. At the
matched 3986-token boundary the adaptive continuation took 45,948.137 ms
versus 31,130.034 ms for exact full prefill, or 47.6% longer in this sample.
`/usr/bin/time -l` reported a 1,000,477,352-byte adaptive peak footprint versus
6,778,360,824 bytes for the matched exact run, about 6.78x lower, but that
counter does not fully attribute Metal, wired, and compressor memory on Apple
unified memory. ClickHouse lookup was 157.848 ms and checkpoint write was
7.623 ms; nearly all of the 44,992.266 ms restore interval was compact-root
restore plus replay of the 3479-token suffix. The 64-token causal bound makes
that suffix 55 replay calls. A previously observed non-finite K/V result at a
96-token span prevents widening this bound by assumption.

A guarded replay-width falsifier on 2026-08-26 tested an 80-token diagnostic
candidate before exposing any production runtime switch. The matched control
used a 3173-token suffix at a 3243-token full boundary: 64-token replay produced 50
chunks, preserved `Their sum is 95.` and EOS, scored top-1 `8/8`, ranked top-2
`12/14`, top-2 set overlap `12/14`, exact-top-1 coverage `7/7`, and output-row
ECS mean/minimum `1.0/1.0`. Free replay took 39,452.520 ms and the complete
cold-hit-to-first-token corridor took 40,197.678 ms. Corrected full-boundary
accounting measured 425,066,496 raw KV bytes versus 112,908,288 resident bytes,
or `3.7647x` density. The guarded run reported zero swaps.

The same suffix partitions into 40 model calls at width 80, but the original
monolithic adaptive-attention dispatch failed closed with device status 59
before adaptive resident-state publication and before producing a quality or
latency result. The exact source artifact may already exist at that point. The
aborted run also reported zero swaps. Width 80 therefore remains rejected as a
hardware attention dispatch; a shorter call count is not a speed result when
the candidate cannot publish a valid resident cache.

A follow-up boundary profile kept the qualified width-64 route and added no
commit, wait, or device synchronization. For the same 3173-token suffix, exact
single-span replay took 24,943.127 ms, identically chunked exact replay took
34,089.049 ms, and adaptive replay took 36,377.014 ms. Existing command waits
accounted for virtually the whole per-call boundary: host encode was about
5--9 ms, finalization/commit/publication stayed below 0.1 ms, and GPU wait grew
from about 0.48 to 0.98 seconds per adaptive call. This attributes the dominant
cost to repeating model work at the fixed-64 call shape, not to ClickHouse,
cache publication, or host-side QBit packing.

The admitted optimization separates the two widths. Model-level replay now
uses one span up to the qualified 4096-token session boundary, while each full-
attention layer internally encodes attention and K/V packing as ordered
sub-dispatches of at most 64 tokens in the same command buffer. One reservation
and one tail marker still publish the complete span atomically. A Metal spec
falsified the old wide behavior at 65 tokens (cosine `0.9994193414` versus the
32+33 packed reference), then passed after internal splitting with identical
canonical K/V payloads and output cosine above `0.9999999`.

The guarded Qwen3.8-27B candidate completed the 3173-token suffix in one model
call. Against the profiled fixed-64 run, free replay fell from 36,377.014 to
33,544.593 ms (7.8%), cold-hit-to-first-token from 36,436.992 to 33,607.913 ms
(7.8%), and forced replay from 38,857.517 to 33,042.061 ms (15.0%). Peak RSS
fell from 2,793,586,688 to 2,285,928,448 bytes and both runs reported zero
swaps. Both emitted `Their sum is 95.` and EOS with top-1 `8/8`, ranked top-2
`12/14`, top-2 set overlap `12/14`, exact-top-1 coverage `7/7`, output-row ECS
mean/minimum `1.0/1.0`, and consistent sole resident ownership across all 16
full-attention layers. These measurements qualify the split-width route only
through the existing 4096-token session limit on M2 Max; they do not qualify a
hardware attention dispatch wider than 64.

A fresh build then repeated the guarded run without a replay-width override,
confirming the production default selected one 3173-token model span. Free
replay was 34,493.090 ms, cold-hit-to-first-token was 34,568.357 ms, forced
replay was 34,056.372 ms, peak RSS was 1,977,303,040 bytes, and swaps remained
zero. The same quality vector and all 16 resident-owner checks passed.

The next admitted prefill optimization removes redundant adaptive row-metadata
loads when the immutable K/V plans already prove that every row in a layer has
the same tier. `Plan#uniform_tier` records that invariant. Matching uniform P4
or BF16 K/V plans use direct base or sidecar addressing inside the Metal
prefill kernel; mixed plans, P5/F32 plans, planless prepared artifacts, and
mismatched K/V tiers retain the generic metadata path. Setting
`QWEN35_ADAPTIVE_UNIFORM_PREFILL_OFF=1` also selects the generic path for a
matched diagnostic control. The cache format, resident bytes, exact checkpoint
history, and DeltaNet ownership are unchanged.

A focused plan spec passed `9/9`; the Metal resident-KV suite passed `14/14`,
including a uniform-BF16 CPU-reference parity row with cosine above
`0.9999999` and maximum delta below `2e-4`. A guarded A/B/A Qwen3.8-27B row at
3,220 full tokens then compared the generic control, the uniform loader, and a
second generic control. The candidate's free replay was 24,847.812 ms versus
26,985.614 and 28,686.299 ms for the controls; forced replay was 25,185.206 ms
versus 26,503.942 and 27,780.730 ms. Against the faster control observation,
this is a measured 7.9% free-replay and 5.0% forced-replay reduction. All three
rows emitted `Their sum is 95.` with EOS, top-1 `8/8`, ranked top-2 `12/14`,
top-2 set overlap `12/14`, exact-top-1 coverage `7/7`, and output-row ECS
mean/minimum `1.0/1.0`. Resident KV remained 112,107,520 bytes versus
422,051,840 raw bytes (`3.7647x`), and no resource guard fired. Absolute exact
baselines varied materially across the A/B/A order, so this qualifies the
bounded loader mechanism on M2 Max, not a stable throughput or cross-hardware
claim.

An aligned 3226-token in-memory representation probe separately preserved
`Their sum is 95.`, EOS, top-1 `8/8`, ranked top-2 `11/14`, top-2 set overlap
`11/14`, exact-top-1 coverage `7/7`, and output-row ECS mean/minimum `1.0/1.0`.
Its raw live KV was 418,381,824 bytes versus 112,316,416 adaptive resident
bytes, or `3.7250x` density for the qualified mixed map. This probe reconstructs
the same adaptive representation from an exact artifact; it is compositional
quality evidence, not a second direct ClickHouse session trace. Its
418,382,876-byte (399.0 MiB) exact control artifact correctly failed the
independent 128 MiB transport cap, while the current adaptive session path
stores its root directly in compact form and passed the separate 4096 lifecycle
above.

Bounded context: local `.qkv` state artifacts and an explicitly configured
ClickHouse HTTP endpoint. Background part merges are a separate storage context
and never establish cache visibility or admission.

## Resident KV experimental slice

This began as a separate default-off experiment. Its all-layer restore path now
composes with the durable cache for non-session native-runtime hits and for the
separately gated adaptive-session route; the resident representation itself
remains experimental. It admits uniform p4/p5 payloads and a canonical adaptive
payload
whose affine block is exactly one `(token, KV head)` vector. Its Metal attention
decode consumes p4 bases plus optional p5/BF16/F32 sidecars directly without
materializing an intermediate Float32 KV cache. The Qwen3.8 path can now replace
the persistent Float32 K/V owner for every configured full-attention layer
during prefill and direct synchronous decode.

- Qwen3.8-27B has 64 layers. The 16 full-attention layers are zero-based
  `3,7,...,63`; the other 48 layers are recurrent/DeltaNet. Only the 16
  token-linear K/V owners participate in resident adaptive compression.
- Active DeltaNet/recurrent state remains uncompressed. QBit recurrent-state
  compression remains a save/restore transport concern because the active
  state has fixed size rather than token-linear growth.
- The required falsifier is parity between fused Metal attention and the CPU
  reference over the *same decoded p4/p5 values*. A seeded plane-bit mutation
  must change the comparison result, proving that the parity check is live.
- Automatic row/head/age tier selection, hot tails, state fork/copy, periodic
  exact root renewal, speculative/asynchronous adaptive decode, and non-GQA6
  shapes are guard-only follow-ups. Default-off adaptive snapshot/writeback and
  synchronous session checkpoint routing are now admitted only through their
  explicit runtime gates. Memory-ratio measurements from this experiment do
  not establish an eightfold production context increase.

Bounded synthetic evidence on Apple M2 Max (2026-08-23):

| Format | Resident KV size vs F32 | QBit/F32 time at 8192 tokens |
| --- | ---: | ---: |
| p4 | 7.53x smaller | 0.94x-1.04x |
| p5 | 6.10x smaller | 1.03x-1.13x |

The parity spec uses the Qwen3.8 GQA shape of 24 query heads and 4 KV heads.
The latency probe uses one KV head to bound allocation and isolate scaling. Its
F32 comparator is the current generic attention kernel, not a hypothetical
GQA-specialized F32 kernel, so these timings establish feasibility rather than
the final production overhead. The probe fails closed on output divergence; its
latest p4/p5 runs had maximum differences of `8.85e-9` and `5.59e-9` against
the current F32 kernel over the same decoded values.

### Retire-then-pack quality gate

The real-model quality falsifier models a conservative streaming order:

1. one bounded prefill chunk, or one decode row, runs exact attention and writes
   ordinary K/V;
2. after that work has completed, its `(token, KV head, 256)` K/V rows are
   quantized and reconstructed;
3. the next chunk or decode row observes the reconstructed older values.

`bin/qwen35_qbit_kv_quality_probe.cr` implements this order without changing
production routing. The exact oracle and QBit variants use identical explicit
chunk boundaries, but only the QBit variants roundtrip each completed chunk
before the next one. Decode rows retire in the same order. The probe then
compares teacher-forced full logits plus free greedy continuation. The roundtrip
restores Float32 values into the existing cache, so it isolates model quality
but does **not** demonstrate live-memory compactness or GPU pack cost. Together
with the resident-attention parity probe, it is evidence for the numeric values
a future fused resident route would consume.

One guarded Qwen3.8-27B Q4_K_M stress slice on Apple M2 Max (2026-08-23) used an
eight-token retire chunk and covered explanatory text, Crystal code, and
arithmetic. All prompts crossed at least two retire boundaries.

| Format | Greedy common prefix | Retire-order top-1 | Minimum decode-logit cosine | Maximum decode-logit delta |
| --- | ---: | ---: | ---: | ---: |
| p4 | 21/24 | 23/24 | 0.955567 | 5.088198 |
| p5 | 21/24 | 23/24 | 0.961276 | 3.694828 |

Both formats preserved the prompt-boundary prediction on all three rows. That
prediction already observes compressed older chunks but, by retire-after-use
definition, is computed before the final chunk itself is packed. The first
teacher-forced decode comparison observes the packed final chunk. Both formats
diverged after a five-token common prefix on the arithmetic row and scored
`7/8` across this retire-order sequence. Repeating that row with a
32-token retire chunk produced the same first divergence. Uniform p4 and p5 are
therefore rejected as production defaults by this bounded accumulated-error
slice. The earlier whole-prompt-retire p5 `38/38` result was a lower-bound
diagnostic: it did not expose later prefill chunks to compressed older values.

Cosine/MSE cannot select the adaptive tier by itself. The code row had the
lowest p4 logit cosine in the earlier six-prompt lower-bound slice yet preserved
all tested top-1 tokens, while arithmetic diverged at a higher cosine. The next
admissible policy is a fixed-index p4 base plus optional refinement planes and
bounded escape metadata, calibrated by layer/head sensitivity and
teacher-forced/free-run gates.

This calibration probe still allocates and writes a full-capacity Float32 KV
cache. The later default-off runtime integration instead packs each completed
chunk directly into its persistent resident owner while only current K/V scratch
is transient. This earlier roundtrip result does not itself establish that
ownership change.

### Adaptive resident row layout

The next default-off slice uses one canonical, directly indexed representation
for every semantic `(token, KV head, 256)` row:

- a dense p4 block is always present;
- one aligned 8-byte metadata entry stores a tier and sidecar byte offset;
- tier `p5` stores only the fifth 32-byte plane in the compact sidecar;
- tier `BF16` stores one 512-byte replacement row;
- tier `F32` stores one 1024-byte exact replacement row.

The metadata entry makes lookup O(1) in the fused attention kernel without a
variable-width scan or rank structure. Sidecar offsets must be canonical,
monotonic, aligned, non-overlapping, and consume the payload exactly; malformed
tiers or offsets fail before Metal admission. This costs 8 metadata bytes per
row, reducing pure-p4 density from `136` to `144` bytes per 256 values, or from
`7.53x` to `7.11x` versus F32.

The default-off codec and fused GQA6 Metal reader implement this layout. The
codec keeps one canonical payload with zero-copy base/metadata/sidecar views;
validation rejects non-canonical offsets, incomplete or trailing sidecars,
invalid tiers, malformed p4 rows, and non-finite replacements. The Metal
kernel reconstructs only the bounded 16-token attention tile. A mixed-tier
parity spec measured cosine `1.0` and maximum CPU-versus-Metal delta
`3.73e-08` across the 16-token kernel tile boundary; it did not benchmark
adaptive decode throughput.

The real-model calibration hook deliberately starts with one tier per
full-attention layer while the wire format remains row-addressable. On the
known arithmetic counterexample with eight-token retirement, p4 plus a BF16
escape only for layer 51 restored `8/8` free-running and `8/8` retire-order
top-1 parity at `5.8182x` KV compactness. The same map also preserved all eight
tokens on the text and Crystal-code rows, giving `24/24` over the original
three-prompt slice. A p5 refinement of layer 51 still failed the arithmetic row.

That map is calibration evidence, not a universal policy. With a 32-token
retirement chunk, the single-layer map failed at the same fifth token. A
coarser four-BF16-layer map (`27,43,47,51`) closed the tested arithmetic row for
both chunk sizes, but density fell to `3.7647x`. This falsifies hard-coding
layer 51 independently of retirement policy and keeps automatic selection
default-off. The next density move is row/head/age-sensitive calibration, not
adding more global layer escapes.

### Position-aware quality vector

The quality probe now reports top-1, top-2, and token embedding cosine at each
teacher-forced position. Both executions consume the same exact prefix at a
compared position, so the expected and compressed token IDs are aligned before
their vectors are compared. Exact token matches score `1.0`; mismatches use the
cosine between the corresponding Qwen `output.weight` rows. This output-space
token ECS measures whether the compressed choice remains close to the exact
choice in the model's token decision representation. It is not a calibrated
human semantic score.

Free-running token positions are deliberately not assigned this ECS after the
first insertion, deletion, or replacement. Their BPE positions may then denote
different grammatical roles even when the generated sentence keeps the same
meaning. Free-running quality therefore retains a separate whole-response
semantic assessment.

A guarded Qwen3.8-27B Q4_K_M slice used 64-token generation limits, eight-token
retirement, and three one-sentence tasks: arithmetic, sky scattering, and
Crystal `Array#map`. `adaptive` is the coarse p4 default with BF16 layers
`27,43,47,51`; `p4` is the rejected uniform control.

| Policy | Logical KV density | Retire top-1 | Exact top-1 in candidate top-2 | Ranked top-2 slots | Top-2 set overlap | Token ECS mean | ECS mismatches | Meaning preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform p4 | 7.5294x | 111/118 | 114/115 | 202/230 | 214/230 | 0.948830 | 7/118 | 3/3 |
| adaptive | 3.7647x | 115/118 | 115/115 | 214/230 | 220/230 | 0.976423 | 3/118 | 3/3 |

The vector exposes two different cases that common-prefix equality hides. In
the arithmetic answer, uniform p4 selected `arrive` instead of `obtain` while
keeping the correct result and explanation. In the sky answer, both policies
omitted `the`, shifting the next teacher-forced comparison to `the` versus
`molecules` even though the free sentence remained correct. The raw token ECS
for these rows was low, so ECS alone would misclassify harmless rephrasing or a
token omission. The admitted evaluation order is therefore semantic failure,
exact top-1 absence from candidate top-2, position-aligned ECS, then bytes; no
single local proxy can override a whole-response semantic failure.

This corpus supports the coarse adaptive map over uniform p4 on the measured
local signals, but it is too small and manually assessed to promote either a
universal selector or a numeric ECS threshold. The next calibration gate must
expand prompts, lengths, and retirement policies and replace manual response
classification with a repeatable semantic judge before policy widening.

### Row-local selector falsifier

The quality-only probe can also choose the cheapest p4, p5, or BF16 tier whose
maximum reconstruction residual, normalized by the row standard deviation,
fits `--selected-max-error`. It reports the resulting tier histogram. The
selector runs after each semantic row exists and roundtrips it into the
diagnostic Float32 cache; it neither changes resident allocation nor estimates
the cost of a fused GPU selector.

On a guarded 16-token sky-scattering slice, the row-local proxy failed the
promotion test:

| Policy | Logical KV density | Retire top-1 | Ranked top-2 slots | Token ECS mean | Meaning preserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| coarse BF16 layers `27,43,47,51` | 3.7647x | 15/16 | 28/30 | 0.934414 | yes |
| selected max error `1.5` | 2.2036x | 16/16 | 30/30 | 1.000000 | yes |
| selected max error `3.0` | 3.8487x | 14/16 | 26/30 | 0.876178 | yes |

The strict selector preserves the tested local decisions only by retaining
many BF16 rows. At comparable density, it is worse than the coarse layer map.
Therefore row-local normalized reconstruction error is rejected as a sole tier
selector: value-space error does not encode downstream model sensitivity. The
next KISS calibration boundary is a static `(layer, K/V, KV head)` sensitivity
map, validated by the same top-1/top-2/ECS/meaning vector.

This is also the shape compatible with the existing immutable resident plan:
tier and sidecar offsets are fixed before the K/V values exist. A truly
value-dependent or age-dependent online selector still requires a separate
two-pass compaction, dynamic arena, or hot-tail repack design and remains
guard-only.

### Static KV-head sensitivity calibration

The quality-only probe now accepts static head overrides such as
`--adaptive-map 51:k0=bf16` in addition to whole-layer overrides. The address is
`(full-attention layer, K/V side, KV head)`; a head override wins over its layer
tier, and the layer tier wins over the p4 default. Qwen3.8-27B has four KV heads
of 256 values in each of its 16 full-attention layers, so this map is fixed by
model geometry and remains compatible with the existing immutable resident
plan. It is a calibration mechanism only; the resident runtime does not yet
consume this map.

A guarded three-prompt slice used 64-token generation limits and 32-token
retirement. The prompts covered arithmetic, sky scattering, and Crystal
`Array#map`. The table aggregates only position-aligned teacher-forced metrics;
whole-response meaning was checked separately.

| Policy | Logical KV density | Retire top-1 | Exact top-1 in candidate top-2 | Ranked top-2 slots | Top-2 set overlap | Token ECS mean | Meaning preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all BF16 attention KV | 1.5610x | 131/131 | 128/128 | 256/256 | 256/256 | 1.000000 | 3/3 |
| BF16 layers `27,43,47,51` | 3.7647x | 128/131 | 128/128 | 242/256 | 248/256 | 0.982879 | 3/3 |
| BF16 `51:k0` | 6.9189x | 127/131 | 128/128 | 235/256 | 243/256 | 0.974869 | 3/3 |
| uniform p4 | 7.5294x | 126/131 | 128/128 | 231/256 | 241/256 | 0.970086 | 3/3 |

The `51:k0` escape recovered one arithmetic top-1 decision and four ranked
top-2 slots over uniform p4 for about eight percent more logical KV bytes. It
did not improve the sky or code prompts, whereas the four-layer map recovered
more aggregate quality at lower density. Tier refinement was also not monotone
in token decisions: on the short arithmetic slice, adding `43:k0` to the
working `51:k0` escape regressed to the uniform-p4 decision vector. Static head
sensitivity is therefore useful calibration evidence, not a universal default
or a proof that independently good head escapes compose.

That promotion gate is now implemented as the deterministic long-response
corpus below. It rejects this head escape and narrows the next resident-GPU
trial to the coarse four-layer map. It still does not justify dynamic
allocation or an online selector.

Mutable row replacement and compressed active DeltaNet state remain guard-only.
The later default-off runtime slice owns compact K/V immediately after each
successful prefill command; this calibration corpus alone does not promote its
tier map to a production default.

### Held-out long-response quality gate

`scripts/qwen_qbit_quality_corpus.py` runs the existing quality probe one case
at a time under `run_safe.sh` and keeps the quality result as a vector rather
than a scalar score. The frozen JSONL manifest contains four task-level
oracles: an inventory ledger, a Crystal pipeline trace, an Earth-seasons causal
chain, and a transaction-recovery protocol. Each compressed response and the
exact-F32 response must satisfy the same required facts and minimum length. If
the exact response fails, the case is an invalid baseline rather than evidence
against QBit. Exact and compressed responses must also reach their own EOS;
the compressed free-run is not capped at the shorter exact-response length.
Score-only replay requires the logged generation limit to match the manifest,
preventing an older or truncated log from being accepted.

The valid exact responses generated 155--313 tokens each, 967 tokens in total.
The teacher-forced top-2 comparison covered 963 positions. The aggregate
held-out vector was:

| Policy | Logical KV density | Meaning | Retire top-1 | Ranked top-2 slots | Top-2 set overlap | Exact top-1 in candidate top-2 | Token ECS mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 layers `27,43,47,51` | 3.7647x | 4/4 | 927/967 | 1682/1926 | 1740/1926 | 952/963 | 0.963158 |
| BF16 `51:k0` | 6.9189x | 3/4 | 913/967 | 1583/1926 | 1667/1926 | 951/963 | 0.951736 |
| uniform p4 | 7.5294x | 3/4 | 914/967 | 1588/1926 | 1667/1926 | 950/963 | 0.951670 |

The head escape is therefore rejected for promotion. It consumed about nine
percent more payload than uniform p4 while preserving the meaning of the same
three cases; its aggregate local metrics were effectively tied and mostly a
little worse. The coarse four-layer map is the only measured policy that
preserved all four task-level meanings and it led every aggregate local metric,
but at roughly twice the payload of p4. It is the next default-off resident-GPU
candidate, not a production default.

The corpus also falsifies treating ECS or top-2 coverage as the semantic gate.
On the inventory task, p4 and `51:k0` kept the exact top-1 inside their top-2 at
every measured position and had ECS about `0.9955`, yet an early free-running
divergence produced the wrong ledger states and final count. ECS remains useful
for locating token drift, but the task oracle owns the earlier acceptance
coordinate.

This certificate is deliberately narrow. The oracles are regular-expression
fact checks over four English responses, not a general semantic judge, and can
miss a contradiction that also repeats the required facts. During exact-
baseline qualification and adversary review, patterns were widened for obvious
inflections or semantic equivalents and a false 60-word boundary was removed;
generation budgets were also raised after 128-token truncation. The same
repaired rules and EOS guard were then applied to every policy, but the judge
was not perfectly preregistered. These are long responses, not restored
multi-turn sessions. The diagnostic probe still
roundtrips into Float32 storage, so resident packing, GPU-only execution,
throughput, and restore/checkpoint behavior remain outside this gate.

### Resident GPU held-out replay

`bin/qwen35_adaptive_resident_kv_quality_probe.cr` reruns the same frozen
four-case corpus through the actual resident adaptive-KV corridor. The exact
Float32 oracle state is released before a policy is evaluated. Each policy then
uses two fresh states: one independent free run and one teacher-forced exact
trajectory. Neither state is snapshotted or reconstructed through a Float32 KV
cache. After prefill and after every decode token, the probe requires all 16
full-attention layers to have packed adaptive owners, no Float32 K/V owner, and
the expected published `cache_len`. Corpus `--resident` mode rejects a record
without that execution proof.

The coarse map is the only policy admitted to this bounded resident replay:

| Execution path | Logical KV density | Meaning | Retire top-1 | Ranked top-2 slots | Top-2 set overlap | Exact top-1 in candidate top-2 | Token ECS mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| diagnostic Float32 roundtrip | 3.7647x | 4/4 | 927/967 | 1682/1926 | 1740/1926 | 952/963 | 0.963158 |
| resident GPU packed KV | 3.7647x | 4/4 | 930/967 | 1702/1926 | 1764/1926 | 956/963 | 0.964905 |

Every exact and resident free response reached its own EOS. All four resident
records reported `resident_layers=16`, an empty
`resident_f32_owner_layers`, and a consistent packed prefix. Density includes
the resident allocation capacity and padding; its absolute byte count must not
be compared directly with the earlier diagnostic's populated-prefix byte
count. The small metric improvement over the diagnostic roundtrip is measured,
but is not evidence that resident execution is generally more accurate.

This closes the missing execution-path falsifier, not the production frontier.
The four English regex oracles can still miss a contradiction that repeats the
required facts, and their rules were repaired after earlier outputs were
observed. These are long single responses rather than restored multi-turn
sessions. Runs were ordered, the exact baseline paid cold Metal setup, and the
probe's decode timers include state cleanup, so they do not provide an A/B
throughput certificate. The coarse map also remains red against the separate
single-prompt `0.05` winning-logit guard below. That guard is retained as a
local drift diagnostic; response meaning, EOS, top-1/top-2, and ECS jointly own
the held-out admission decision.

The KISS decision is therefore narrow: retain
`p4;27=bf16,43=bf16,47=bf16,51=bf16` as a default-off resident policy for the
direct synchronous path. Production defaulting, native/asynchronous routing,
restore/checkpoint quality, longer multi-turn sessions, and quiet-host speed
remain separate gates.

### Append-only device packing slice

The next bounded default-off slice now removes the host encoder from the
resident-cache construction path without changing production `LayerState`
ownership. A host-built tier plan fixes every row's metadata and canonical
sidecar offset before allocation. The resident owner then allocates exact
base/metadata/sidecar capacity once and appends a contiguous K/V chunk directly
from temporary Float32 Metal buffers. Its visible length advances only after
both device pack dispatches complete, their shared failure word remains clean,
and a final device encoder writes the completion marker. No persistent Float32
KV buffer is retained by this owner.

One 32-lane SIMD group packs one 256-value `(token, KV head)` row. It computes
moments, writes the dense p4 base, and writes the selected p5/BF16/F32 sidecar
into its preplanned location. Non-finite inputs or moments, invalid tiers, and
non-finite p4/p5 reconstruction bounds fail closed without advancing the live
prefix. A failed row range may be overwritten by a later clean retry. The plan
is immutable to callers, and diagnostic snapshots reassemble only the live
prefix and run the ordinary strict host validator.

GPU moments use deterministic Float32 SIMD reduction, while the diagnostic CPU
encoder uses Float64 accumulation. Byte identity is therefore not an admitted
invariant. The required invariant is canonical structural validation plus
bounded decoded-value and attention parity. A 19-token mixed-tier test appends
`7 + 12` tokens, crosses the 16-token attention tile, and compares the packed
snapshot with the CPU codec. On Apple M2 Max it passed with decoded-value delta
below `2e-5`, attention cosine above `0.9999999`, and attention maximum delta
below `2e-4`. A separate NaN injection left `cache_len == 0`, then a finite
retry produced a strictly valid snapshot.

A bounded synthetic quiet-host probe (no model weights) measured the current
synchronous API, including K and V dispatches, command completion, and the
failure-status read:

| Retired tokens | p4 pack time | mixed25 pack time | p4 input throughput | mixed25 input throughput |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.249 ms | 0.215 ms | 0.245 GiB/s | 0.283 GiB/s |
| 32 | 0.197 ms | 0.191 ms | 1.237 GiB/s | 1.278 GiB/s |
| 128 | 0.189 ms | 0.183 ms | 5.164 GiB/s | 5.344 GiB/s |

Here `mixed25` is a synthetic repeatable plan with one BF16 row per four rows;
it reproduces the `3.765x` representation density of the coarse four-layer
calibration but is not a learned selector. Pure p4 density is `7.111x` after
metadata. At live contexts from 80 to 1280 tokens, adaptive attention ranged
from `0.764x` to `1.152x` the generic F32 comparator time over the same decoded
values, with maximum output delta below `6e-8`; these small timings are noisy
feasibility evidence, not a production speed claim.

Still rejected in this slice:

- production `LayerState` routing or simultaneous full-capacity F32 and packed
  ownership;
- mutable age-based re-tiering or sidecar compaction;
- prompt-cache fork/snapshot/rollback semantics for the packed owner;
- Qwen3.5 GQA4, non-256 head dimensions, or asynchronous publication of a
  prefix before pack completion.

The next integration seam is the existing fused full/recurrent prefill command:
exact attention consumes the current temporary chunk, then the same command
packs retired K/V rows for the next chunk. Production promotion additionally
requires packed-aware decode/prefill consumers and an explicit rejection or
implementation of fork/snapshot paths; the synthetic append owner alone does
not establish immediate compactness for the shipped runtime.

### Mixed packed-history prefill slice

The bounded resident owner now closes the command-ordering seam without yet
changing production `LayerState`. Its GQA6/head-dim-256 prefill kernel computes
each current query over two representations:

- positions before the current chunk are decoded directly from resident
  p4/p5/BF16/F32 rows;
- the causal portion of the current chunk is read exactly from caller-owned
  temporary Float32 K/V buffers.

Six query heads share each reconstructed KV tile. Attention runs before both
device pack dispatches in one command buffer. A shared failure word covers the
attention and pack kernels; `cache_len` advances only after command completion
and a final device kernel replaces a clean zero with a non-zero completion
marker. An untouched zero is failure, so a command that never executes cannot
publish a prefix. A non-finite query, gate, K, or V likewise leaves the
attempted prefix invisible and allows the same destination rows to be retried.

A deterministic 19-token mixed-tier spec exercised two chunks (`7 + 12`). The
first chunk used only exact current K/V; the second consumed the packed first
chunk plus its exact causal tail. Both outputs matched a CPU reference built
from the same decoded history with cosine above `0.9999999` and maximum delta
below `2e-4`. A separate non-finite-query case kept `cache_len == 0` and a clean
retry succeeded. All temporary and resident Metal buffers returned to the
pre-test live-byte baseline.

This proved the representation and ordering seam before runtime ownership was
changed. The bounded integration below now gives every configured
full-attention layer that owner; single-token decode consumes and extends the
same packed owners. Advanced asynchronous/speculative decode remains a separate
frontier.

### All-full-attention runtime prefill and decode integration

The default-off Qwen3.8 experiment accepts either the legacy one-layer pair
`QWEN35_ADAPTIVE_RESIDENT_KV_LAYER` plus
`QWEN35_ADAPTIVE_RESIDENT_KV_TIER`, or an all-layer map such as
`QWEN35_ADAPTIVE_RESIDENT_KV_MAP=p4;27=bf16,43=bf16,47=bf16,51=bf16`.
The map assigns its default tier to every full-attention layer and then applies
explicit layer overrides. Mixing selector forms, duplicate overrides,
recurrent-layer overrides, unsupported geometry, or a second Float32 owner
fails before inference. Allocation is transactional across all selected layers.

Prefill keeps the intermediate hidden state and every temporary Q/K/V row on
Metal. Fused full-attention-plus-recurrent groups append to one caller-owned
command; layer 63 uses the standalone full-attention encoder because it has no
recurrent successor. Every cache appends its completion marker at the true
command tail. After successful command completion, all pending caches are
validated before any `cache_len` is published, so a failure cannot expose a
partially advanced layer set.

Direct synchronous decode passes one adaptive encoder per selected layer into
the existing whole-token Metal wave. Each full-attention layer consumes its
packed prefix plus exact current K/V, then packs the current row in the same
command. Group publication again validates every layer before advancing any of
them. Position mismatch, duplicate ownership, a declined Metal wave, and
unknown command outcome fail closed. `Qwen35NativeRuntime` now admits the same
owner only after a strict non-session QBit hit. `forward_hidden`, fork/copy,
snapshots, checkpoint swapping, tail clearing, session routing, and advanced
asynchronous/speculative helpers remain outside this admitted corridor.

A guarded Qwen3.8-27B Q4_K_M all-BF16 run used `8 + 4` prefill tokens and 32
teacher-forced decode comparisons. It retained `32/32` top-1 and `64/64`
ranked top-2 slots, with top-1 and second-token ECS `1.0`; maximum winning-logit
delta was `0.0038605`, below the unchanged `0.05` guard. Every one of the 16
caches published 44 rows with no Float32 alias. Resident K/V used 3,694,592
bytes instead of 5,767,168 bytes (`1.561x`).

The deeper coarse map (`p4` plus BF16 layers `27,43,47,51`) is structurally
stable but remains red against this local winning-logit guard. On the same
32-step diagnostic it retained
`32/32` top-1, `58/64` ranked top-2 slots, top-1 ECS `1.0`, mean second-token
ECS `0.8512`, and `3.7647x` density, but maximum winning-logit delta reached
`1.6252`. A p5 base improved ranked top-2 only to `59/64` and mean second-token
ECS to `0.8815` while reducing density to `3.4595x`; it did not close the
quality gate. These ordered timings show roughly 7-8% decode overhead but are
not quiet-host throughput evidence.

The later resident held-out replay above preserved all four task meanings with
the same coarse map. This does not erase the local red signal: the two gates
measure different failure surfaces, and the `0.05` guard remains useful for
detecting concentrated logit drift. The all-layer runtime mechanism and coarse
map are therefore admitted only as a default-off bounded experiment. More
aggressive p4/p5 maps remain diagnostic calibration inputs; production
promotion requires broader session, lifecycle, and performance evidence rather
than relaxation of either gate.

## Admitted surface

- A default-off probe may encode recurrent Float32 records in independent
  affine blocks, quantize normalized values with the 256-level Gaussian
  Lloyd-Max quantizer, retain the most-significant 8, 7, or 6 code planes, and
  reconstruct each prefix with its Gaussian conditional mean.
- The probe may report payload size, CPU encode/decode time, next-token logit
  error, and free-running token parity on a local model.
- A default-off resident-KV probe may encode one 256-value semantic row as a
  fixed p4 base plus canonical tier/offset metadata and an optional p5, BF16, or
  F32 sidecar. After strict validation, its Qwen3.8 GQA6 Metal kernel may decode
  those rows directly inside a bounded attention tile.
- A default-off append-only resident owner may preplan exact K/V base,
  metadata, and sidecar capacity and pack a contiguous temporary Float32 Metal
  chunk directly into the next canonical rows. It may publish the longer live
  prefix only after both pack dispatches complete and their final device
  completion marker is present. Diagnostic snapshots may read and validate
  only that live prefix.
- A default-off GQA6/head-dim-256 resident prefill probe may compute causal
  attention over a packed immutable prefix plus an exact temporary Float32
  current chunk, then pack that chunk later in the same command buffer. The
  longer prefix may become visible only after attention and both packers leave
  a clean shared status and the final device completion marker is present.
- A default-off Qwen3.8 runtime experiment may replace any configured set of
  full-attention layers' persistent Float32 KV buffers with resident owners
  during multi-token prefill and synchronous single-token decode. An all-layer
  map applies a default tier and explicit per-layer overrides. Group publication
  belongs to the existing shared command tail and must validate every cache
  before publishing any cache. Advanced async, native-runtime,
  hidden-state-only, and state-copy/persistence operations must fail closed.
- The real-model quality probe may apply an explicitly supplied tier per
  full-attention layer to calibrate the finer row-addressable representation.
  Such maps are diagnostic inputs, not runtime policy.
- Eight-plane reconstruction is the full-code reference. It is not lossless
  relative to the original Float32 state.
- A diagnostic Native writer may batch complete 1024-code tiles and append an
  all-zero eighth plane for the p7 representation required by
  `QBit(Int8, 1024)`. This is a column-layout operation, not a second bit
  transpose.
- A strict Native parser may validate the exact eight-column schema,
  contiguous record/tile ordering, tail counts, and zero p7 LSB for either one
  block or an ordered sequence. A block boundary may split a logical record
  only after a full tile; global tile and destination offsets remain exact.
- A diagnostic Metal kernel may consume that plane-major Native block in one
  upload, fuse p7 untranspose, conditional-centroid lookup, affine
  reconstruction, and write all recurrent records into existing Float32 state
  buffers in one command buffer. Live KV remains exact and outside the QBit
  column.
- A diagnostic `clickhouse local` probe may insert one bounded p7 Native block
  into an isolated temporary MergeTree, inspect `system.parts`, read the rows
  back in canonical order, and require an exact byte-for-byte Native round trip.
  The temporary part is measurement evidence only; it is not a durable cache.
- A matched diagnostic probe may store recurrent QBit plus an exact raw-KV
  artifact in two parts and the complete recurrent-INT8 artifact in one part.
  It reports logical, compressed-data, and on-disk bytes separately and requires
  byte-exact round trips for all three inputs.
- The model probe may accept a bounded external multi-block response only with
  an explicit expected `cache_id`. Prompt/model/token hashes remain the
  responsibility of the surrounding versioned cache envelope.
- A complete diagnostic corridor may read recurrent Native QBit plus exact KV,
  structurally validate both inputs, restore prepared Metal state, and execute
  the first post-restore forward. When the matched INT8 input is present, the
  probe alternates execution order as ABBA and reports paired deltas.
- A versioned QBit cache envelope may bind the state runtime, model, tokenizer,
  explicit chat-template identity, prompt and token hashes, state ABI, codec,
  exact-known-span validation result, and cached next token. Admission requires
  the complete per-layer recurrent/KV record set, exact declared record byte
  sizes and positions, and a block-framing-independent logical digest of all
  QBit columns. V1/V2 KV payloads must remain full-sized. V3 may instead carry
  exactly `prefix_len` complete KV rows from each original `max_seq` record;
  arbitrary truncation remains invalid.
- A successful strict admission may retain the parsed zero-copy views as a
  process-local certificate while their backing byte slices remain immutable.
  Repeated restores from that certificate need not reparse or rehash the same
  bytes.
- The internal ClickHouse client may publish one random 256-bit generation by
  inserting recurrent QBit and exact KV first, then inserting its manifest as
  the commit marker. A failed artifact insert leaves no visible manifest;
  incomplete generations are reclaimed later by TTL.
- Lookups must match both the narrow `UInt64 cache_id` and the full 256-bit
  lookup identity, select one completed generation, bound every streamed HTTP
  response, and run strict envelope/artifact admission on the first process
  hit. An optional byte-bounded resident admission cache may reuse that result
  only after rechecking the current manifest; it is disabled by default.
- Default response limits are 128 MiB for each artifact and 192 MiB combined,
  with a 512 MiB hard configuration ceiling. After reading recurrent state, the
  client uses the authenticated KV size from the manifest to reject an excessive
  combined allocation before it starts the second large response.
- Runtime lookup identity may contain only facts known before inference:
  runtime/model/tokenizer/template identity, rendered prompt and token hashes,
  exact prompt length, state ABI, `max_seq`, and QBit layout. Cached next-token
  and validation-certificate fields are outcomes; they must be admitted from
  the manifest and then checked against the request tokens before restore.
- The first runtime slice admits exact full-prompt hits only. It may synchronously
  write back a freshly prefetched state after a miss, using one-step
  exact-known-span validation derived from `prompt_tokens + next_token`. The
  route is disabled by default and has an explicit bounded TTL.
- A session checkpoint lookup must match the caller-owned rendered transcript
  boundary and the exact token prefix independently. Text equality alone is
  insufficient because Qwen generation prompts contain control tokens that are
  not reproduced by rendering a completed assistant message.
- A fresh ordinary-session prefill may capture only recurrent Metal state eight
  tokens before the generation-prompt boundary. After generation, the fast
  anchor path is admitted only when re-tokenizing the exact completed transcript
  preserves every token through that capture point. It reuses the exact live KV
  prefix, swaps in the recurrent checkpoint, sequentially replays the short
  completed-boundary suffix, and zeroes unused KV rows before publication. It
  never allocates a second full KV cache.
- Early token divergence, missing capture, unsupported Metal weights or state
  ownership, an explicit `QWEN35_QBIT_EXACT_ANCHOR_FAST_OFF=1`, and any state
  restored from QBit or the local prompt cache retain the conservative full
  sequential anchor rebuild. In particular, periodic anchor renewal from a p7
  QBit restore stays sequential so approximation is never compounded through a
  captured recurrent checkpoint.
- Descendants store only the cumulative exact token suffix, parent/anchor
  identities, transcript boundary hash, and certificate. The referenced anchor
  artifact retains the next-token validation; delta metadata does not invent a
  cached next token.
- Session checkpoint chains are immutable and branchable. Delta depth is
  bounded to eight, and the checkpoint schema admits at most 4096 cumulative
  adaptive-session replay tokens. The runtime further limits replay by
  `max_seq`, root-prefix length, and generation budget. Crossing either schema
  limit currently fails closed. A future compact-root renewal must recompute
  exact caller-authoritative transcript tokens rather than promote restored
  approximate recurrent state. Session identities are stored only as SHA-256
  hashes.
- Restore always targets a fresh state. Without an explicit checkpoint, a miss,
  transport failure, admission rejection, or validation/shape restore error
  discards that state before the existing local cache or ordinary prefill route
  is attempted. The same failure for an explicit checkpoint propagates so the
  caller cannot silently leave its selected rollback state. A system or Metal
  execution failure also releases the partial state, but propagates instead of
  attempting another heavy allocation. A write-back failure is observable in
  cache stats but must not fail or alter the generation already in progress.
  Unexpected system failures still propagate.
- With an explicit adaptive resident-KV environment map, a non-session Metal
  QBit hit may prepare sole adaptive owners, restore recurrent state and exact
  live KV into them, and continue through the synchronous prefill/decode wave.
  `adaptive_hits` reports only a successfully restored state that actually owns
  adaptive KV. A miss prepares the ordinary Float32 state so existing writeback
  remains valid. A session request uses adaptive owners only when the resident
  map and `QWEN35_QBIT_ADAPTIVE_SESSION=1` are both explicit; otherwise it uses
  the Float32 route. The adaptive session stores one immutable compact root and
  exact cumulative token deltas, and aborts rather than falling back after a
  Metal mutation begins.
- A validation/shape rejection discards the candidate and may use the ordinary
  prefill fallback. Once adaptive restore enters a Metal decode/pack operation,
  any exception is promoted to a system restore failure: the candidate is
  released and the request aborts instead of attempting a second heavy state.
- Async anchor publication is separately opt-in and single-flight: one active
  immutable host snapshot is the complete queue capacity. It never transfers a
  live Metal buffer to the worker, never overlaps QBit encode with the next
  inference request, and preserves artifact -> manifest -> checkpoint order.
  The returned result marks its checkpoint as pending until publication, while
  cumulative success/failure, capture, commit, pending, and wait metrics remain
  observable. The next request and `close` flush the worker; a background
  failure is reported before the next inference and by `close`.

## Rejected surface

- No production prompt-cache default or compatibility promise is added in this
  slice. Envelope schema v1 and the ClickHouse table schema are internal
  boundaries and may still change before production promotion.
- No claim that scalar MSE implies autoregressive parity.
- No fixed layer escape map as production policy, enabled-by-default
  compactness, adaptive decode speedup, or eightfold production context-window
  claim is admitted. Immediate compact ownership is established for the
  explicitly configured full-attention layers, including the bounded all-layer
  Qwen3.8 experiment; aggressive p4/p5 maps remain diagnostic rather than a
  production quality result.
- No claim that ClickHouse background merges are on the cache-hit critical
  path: newly inserted rows must remain readable before a part merge completes.
- No native ClickHouse TCP packet framing/compression, automatic retry policy,
  CUDA decoder, or enabled-by-default cache route in this slice. The runtime
  uses bounded HTTP requests and synchronous inserts only when explicitly
  configured.
- No unbounded or multi-entry background write queue, cross-`max_seq` restore,
  or partially restored state reuse is admitted in the active runtime slice.
- No live Metal state, GGUF weights, tokenizer, or mutable transcript storage
  may be owned by the background writer. It receives only a bounded immutable
  host snapshot and copied checkpoint metadata.
- An adaptive non-session native-runtime hit is read-only with respect to
  durable state and does not snapshot or write back its extended prefix. The
  separately gated adaptive-session route may publish its certified initial
  compact root and later exact token deltas. Periodic compact-root renewal and
  state fork/copy remain rejected until they can preserve exact transcript
  authority without promoting approximate recurrent state.
- A pending checkpoint identity is not a process-crash recovery record. Until a
  successful explicit or automatic flush, termination may leave that identity
  unpublished and callers must not treat it as durable.
- No claim that the recurrent rollback path accelerates periodic anchors rebuilt
  after a QBit restore. Those anchors remain synchronous and fully sequential.
- No arithmetic recurrent-state delta chain is admitted. Applying successive
  p7 differences would compound approximation error and make rollback depend
  on chain length. Session deltas contain exact token ids and are recomputed
  from a bounded full anchor instead.
- No cache artifact becomes transcript authority. Message content and the
  selected rollback point remain caller-owned; cache metadata may bind only a
  session hash, checkpoint identity, token hashes, and exact replay tokens.
- `always rollback` means every non-expired committed checkpoint whose anchor
  is still retained and validates. It is not an infinite-retention guarantee.
- The manifest certificate is a deterministic integrity binding, not an HMAC,
  signature, or trust proof for bytes supplied by an untrusted store. A first
  strict cold admission must still verify the artifacts. Mutable backing bytes
  invalidate a process-local `Admission`.
- A fast kernel or compact Native block alone is not evidence that the complete
  cold-hit path is faster.
- Recurrent-only QBit part bytes must not be compared with a full INT8 artifact
  that also contains live KV and its envelope. A storage win requires matched
  artifact boundaries or an explicitly itemized recurrent-plus-KV total.

## Guard-only future

- A trusted ClickHouse transport/storage certificate that can safely amortize
  the first logical digest, tied to the full lookup identity and returned
  artifact checksums rather than to a truncated row key alone.
- Background checkpoint compaction and anchor renewal that preserve every
  still-retained rollback boundary.
- ClickHouse storage using fixed-size tiles and independently readable bit
  planes.
- Progressive fetch or fallback from 6/7 planes to the full 8-plane code.
- Row/head/age-sensitive KV tier selection, mutable compressed-row replacement,
  non-GQA6 kernels, lifecycle integration, and production-default promotion.

## Design laws

- Persist block mean and standard deviation; fail closed on non-finite input.
- Pack planes most-significant first and keep the format deterministic.
- Compare 6/7 planes against the 8-plane quantized reference and against the
  current BF16/INT8 cache routes; do not compare only with raw Float32 size.
- Treat token/logit parity and end-to-end cold-hit latency as the value. Payload
  bytes and scalar MSE are supporting proxies.

## Falsifier roster

- Known Gaussian cells reconstruct symmetrically and share one centroid within
  each retained prefix.
- Payload sizes equal the declared block layout for full and tail blocks.
- On deterministic Gaussian-like data, MSE must not increase when precision is
  widened from 6 to 7 to 8 planes.
- On real cache state, report first token divergence and continuation parity;
  any divergence keeps the precision experimental.
- Adaptive row payloads must reject non-canonical or out-of-bounds sidecars,
  invalid tiers, malformed p4 moments, non-finite replacements, and trailing
  bytes before Metal admission. Mixed-tier fused attention must match the CPU
  reference across at least one internal tile boundary. A layer map that loses
  top-1 parity when prompt or retirement policy changes must not be hard-coded.
- A later ClickHouse gate must measure insert visibility separately from
  background merge duration and full cold-hit restore latency.
- The local physical-storage gate must fail if ClickHouse changes the Native
  bytes, returns more than the expected rows, or reports an impossible part
  size. Logical Native bytes, `data_compressed_bytes`, `bytes_on_disk`, and
  response bytes are separate metrics and must not be substituted for one
  another.
- Revision-0 Native bytes must be accepted by ClickHouse and reproduce all
  metadata plus the exact seven retained bit-plane subcolumns. Missing rows,
  malformed plane sizes, mixed tile widths, or unsupported precision must fail
  before any state becomes admissible.
- Metal p7 reconstruction must match the CPU reference on constant, tail,
  sign-boundary, and extreme-code blocks, then retain real-model continuation
  parity. The end-to-end gate is `Native read + Metal restore + continuation`,
  not kernel time alone.
- A multi-block response must reject skipped tiles, a non-final partial tile,
  record reappearance, mixed QBit widths, unexpected cache identity, and an
  input above the probe's bounded size before state admission.
- A complete-state latency comparison must use matched source state, exact
  first post-restore token checks, prepared states, and alternating QBit/INT8
  order. Sequential format-wide runs are rejected as an order-sensitive proxy.
- Envelope admission must fail on model/tokenizer/template or prompt-token
  identity changes, missing or duplicate state records, wrong ABI byte sizes or
  positions, logical recurrent corruption, exact-KV corruption, and mutation of
  any certificate-covered manifest field.
- The logical recurrent digest must remain identical when ClickHouse legally
  reblocks the same ordered rows; raw transport-byte hashes are not a valid
  substitute for logical artifact identity.
- Failure before the final manifest insert must not publish a generation.
  Duplicate or partial artifact rows, an invalid generation id, an unsafe SQL
  identifier, a malformed manifest, and any oversized response must fail closed
  before state admission.
- Two entries with the same request-known lookup identity but different cached
  next tokens or validation hashes must map to the same lookup key. The manifest
  selected by that key must still fail if its certificate is malformed or its
  validation hash is not exactly the hash of `prompt_tokens + next_token`.
- Without a caller-selected explicit checkpoint, injected ClickHouse miss,
  malformed admission, transport exception, and validation/shape restore
  exception must all reach the matched ordinary-prefill result without
  consuming the partially restored state. The same failures against an
  explicit checkpoint must abort rather than silently choose another state. An
  injected system/Metal restore exception must release the candidate and abort
  without retry. An injected write failure after prefill must preserve the
  generated token and increment only the write-failure counter.
- Longest-prefix lookup must reject a row with the right model but the wrong
  tokenizer, template, state ABI, `max_seq`, token hash, or claimed prefix
  length. A truncated `UInt64` cache id is never sufficient admission evidence.
- Restoring a full anchor and replaying the request suffix must produce the same
  next token as an ordinary full prefill. The measured corridor includes index
  lookup, artifact read, Metal restore, suffix replay, and first continuation.
- A delta checkpoint must bind its session hash, parent checkpoint, immutable
  anchor generation and certificate, anchor prefix hash, exact cumulative token
  suffix, child prefix hash, and completed-transcript text boundary. Mutation
  of any field or token must fail before state becomes reusable.
- Explicit rollback must reject a checkpoint from another session and a
  checkpoint whose child token sequence is not a prefix of the supplied
  caller-authoritative transcript. Branches from one anchor remain distinct.
- Delta depth and replay tokens are bounded. The current adaptive-session route
  fails closed at either bound and leaves previous committed checkpoints
  readable. Any future replacement root must be recomputed from exact
  caller-authoritative transcript tokens before publication.
- A captured exact-anchor rollback point must contain recurrent buffers only,
  with no second KV allocation. Unsupported weights or CPU-owned debug state
  must not enter the fast route, and early BPE divergence must select the full
  sequential fallback.
- Rewinding and replaying an exact completed-transcript boundary must match a
  full sequential baseline for the checkpoint next token/logit and the next
  continuation token/logit. Unused live-KV tail rows must be zero before the
  artifact is encoded.
- Async enqueue must return while encode/publication is blocked, reject a second
  job, expose exactly one pending item, and make flush/close wait for the first
  completion. Publication failure must not become a successful write or a
  silently durable checkpoint.
- The async request result must identify its checkpoint as pending. A request
  that depends on the in-flight row must wait before lookup/inference, and cold
  restore after flush must retain the same checkpoint certificate and
  continuation parity as the synchronous route.
- The pending-checkpoint barrier is session-scoped, not global. A request for
  another session, or a sessionless request, must record exactly zero added
  wait while a publication is in flight; a request that resolves the pending row
  itself must record a non-zero wait. A continuation of the pending session must
  still reach its own chain rather than fork, which holds only because enqueue
  drains the previous job before claiming the single-flight slot.
- A failed publication must be reported once and must not become fatal to
  unrelated requests: the drain at the next enqueue records the write failure
  and returns normally, and the retained error surfaces through the explicit
  durability barrier. Teardown must complete even when it reports that failure —
  after close the runtime rejects further barriers as closed and a repeated
  close is silent.
- The measured async corridor must report synchronous host-capture time,
  response latency, background encode/commit time, immediate-next-action wait,
  peak pending jobs, and free-memory floor. Moving latency to the next request
  or exceeding the one-snapshot memory bound falsifies the acceleration claim.

## Stop rules

- Stop widening the artifact format if the pure codec does not survive the
  deterministic tests.
- Do not implement a device decoder until at least one real-model row shows a
  useful size/parity trade-off.
- Do not attribute cache latency to ClickHouse merges unless a measured read is
  actually blocked on a merge.
- Stop promotion if p7 cold-hit latency does not beat recurrent BF16 or is not
  competitive with recurrent INT8 after storage read, validation, upload,
  restore, and first continuation are recomputed together.
- Stop promoting delta replay as a speedup if total restore plus replay does not
  beat full prefill. A compactness-only route may remain experimental when its
  quality, storage, memory-pressure, and rollback gates pass, but the measured
  latency trade-off must stay explicit. Stop all delta promotion if retained
  bytes per rollback boundary do not fall below full-snapshot storage. Do not
  hide anchor cost or checkpoint-index bytes.

## Host resource safety

- QBit verification is limited to `scripts/qwen_qbit_safe_check.sh`. The script
  accepts no caller-provided spec paths and therefore cannot silently widen
  into the full Crystal/Metal suite.
- Heavy Gemma/Qwen specs and model probes must run through `scripts/run_safe.sh`.
  Its macOS system-memory floor is enabled by default at 12%; setting it to zero
  is an explicit unsafe opt-out.
- The in-process spec watchdog uses the same default system-memory floor, so a
  direct `crystal spec` remains pressure-bounded even when the outer wrapper is
  accidentally omitted. RSS remains a secondary signal because Metal, wired,
  and compressor pages share Apple unified memory and are not fully attributed
  to the child RSS.
- Do not use the full suite as a QBit completion proxy. The focused codec,
  Native layout, tail, malformed-input, and Metal parity falsifiers are the
  relevant gate; broader model families add resource pressure without closing
  this frontier.
- After any watchdog reboot or pressure termination, stop model work until the
  panic report and current host headroom are inspected. A user-space watchdog
  reduces risk but cannot guarantee recovery once the kernel scheduler is
  already starved.

The 2026-08-15 full-suite attempt violated the pre-existing guarded-run
boundary and ended in a watchdog panic. The panic reported no watchdogd
check-ins for 92 seconds, compressor segments at 100% (`BAD`), 76 swapfiles,
and only 908 free 16 KiB pages. `crystal-run-spec.tmp` was the largest sampled
process at 3.49 GB RSS, illustrating why RSS alone was not an adequate guard.
The report does not isolate Metal pipeline cache bytes from buffers, compiler
state, other processes, or VM compressor churn. `MTLDevice.currentAllocatedSize`
is now exposed for future attribution; it is diagnostic evidence, not a reboot
prevention mechanism.

## Measured evidence (2026-08-14)

All model rows used the embedded chat renderer, greedy continuation, recurrent
block size 1024, and exact ClickHouse p6/p7 centroid bit patterns. Payload sizes
include raw KV records but exclude a future QBit artifact envelope.

| Model / prompt gate | Codec | Bytes | Raw ratio | Existing INT8 delta | Free run | Teacher forced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 9B, sky, 32 tokens | p7 | 20,017,664 | 32.77% | -7.39% | 32/32 | 31/31 |
| Qwen3.5 9B, sky, 32 tokens | p6 | 18,371,072 | 30.08% | -15.01% | 1/32 | 29/31 |
| Qwen3.8 27B, sky, 16 tokens | p7 | 51,404,032 | 29.60% | -8.46% | 16/16 | 15/15 |
| Qwen3.8 27B, sky, 16 tokens | p6 | 46,501,120 | 26.78% | -17.19% | 16/16 | 15/15 |

The 9B p7 gate also preserved 32/32 free-running and 31/31 teacher-forced
top-1 tokens on JSON and Python prompts. The 9B p6 counterexample means p6 is
not near-lossless and cannot be promoted from this evidence. P7 is the first
candidate for a wider prompt/cursor matrix.

The current scalar CPU implementation is a negative latency result. On Qwen3.8
27B, p7 encode/decode took 2,398/1,191 ms, followed by 13 ms raw Float32
restore. The current recurrent INT8 artifact encoded in 248 ms. A direct Metal
plane decoder is therefore a prerequisite; the CPU route must not enter the
cache-hit path.

### Direct Native-to-Metal gate (2026-08-15)

A release build on the Apple M2 Max exercised Qwen3.8 27B with a five-token
plain-text prompt, seven timed restores after one cold restore, and a prepared
Metal state. The p7 route parsed the actual revision-zero Native block, uploaded
that block once per restore, decoded all 96 recurrent records in one command
buffer, restored exact live KV, and then continued generation.

| Route | Artifact / payload bytes | Cold restore | Median prepared restore |
| --- | ---: | ---: | ---: |
| recurrent BF16 artifact | 86,838,556 | 10.112 ms | 8.556 ms |
| recurrent INT8 artifact | 47,768,476 | 8.687 ms | 6.064 ms |
| Native p7 recurrent + raw KV | 48,646,236 | 10.426 ms | 6.887 ms |

Native p7 retained 16/16 free-running and 15/15 teacher-forced top-1 tokens;
the largest matched top-1 logit delta was 0.292534. The 40,257,628-byte Native
recurrent block encoded in 18.080 ms and its strict parser took 4.477 ms. P7 is
19.5% faster than BF16 at the median, but 13.6% slower than INT8 and 1.84%
larger than the full INT8 state before transport compression. This is a useful
experimental cache route, not a promotion result. QBit's repeated metadata and
all-zero eighth plane need ClickHouse protocol/part compression to establish a
storage-size win.

ClickHouse 26.8.1.1 accepted all 38,304 rows, reconstructed a total value count
of 39,223,296, and observed a zero eighth stream for every row. An ordered
`SELECT ... FORMAT Native` re-emitted a byte-identical 40,257,628-byte block,
which closes the writer/parser representation gap. Warm local reads that hashed
all seven retained plane subcolumns took 10-13 ms. Those server-side timings do
not include a future TCP client, response materialization, raw-KV lookup, or
first-token forward pass, so the complete cold-hit promotion gate remains open.

### Real ClickHouse physical-storage gate (2026-08-15)

The guarded Qwen3.8 27B probe was repeated after the host reboot with the same
11-token chat prompt and 16-token continuation. It produced a
40,257,628-byte recurrent p7 Native block, 8,388,608 raw live-KV bytes, and a
47,768,476-byte recurrent-INT8 artifact. P7 again retained 16/16 free-running
and 15/15 teacher-forced top-1 tokens. Prepared restore medians in this repeat
were 8.230 ms for p7 and 8.469 ms for INT8; this reverses the earlier 13.6%
p7 loss, so their small in-memory difference is host/order sensitive and must
not be promoted without an interleaved benchmark.

`scripts/qwen_qbit_clickhouse_probe.sh` inserted that exact Native block into a
bounded temporary MergeTree under ClickHouse 26.7.1.1315. The active part used
34,580,274 compressed data bytes and 34,582,213 bytes on disk, 14.10% and
14.10% below the logical recurrent Native block respectively. Adding all live
KV bytes without assuming any KV compression gives 42,968,882 bytes, 10.05%
below the logical INT8 artifact. This is a conservative mixed-boundary estimate,
not a matched ClickHouse-vs-ClickHouse result; the matched gate below supersedes
it for storage decisions.

Five exact single-block Native reads measured 29 ms first and 28 ms median.
Together with the same run's 5.474 ms parser and 8.230 ms prepared restore,
this gives a 41.704 ms lower bound for `recurrent read -> parse -> restore`.
It still excludes KV lookup/validation and the first continuation token.

The default ClickHouse read split the same response at MergeTree block
boundaries and measured 19-21 ms across five warm reads. Raising
`preferred_block_size_bytes` to the bounded input limit coalesced it back into
the writer's single 38,304-row block and restored exact byte identity, but cost
about 8 ms median. This is a new transport falsifier: production should parse
and restore validated multi-block responses instead of paying for forced
coalescing.

### Natural multi-block restore gate (2026-08-15)

The stream parser now accepts the five naturally emitted Native blocks without
copying them into a synthetic single block. It proves global record contiguity
across block boundaries and gives each chunk a checked destination offset. The
Metal route uploads each source block once and submits all record chunks in one
command buffer; exact live KV still comes from the snapshot envelope.

On the same Qwen3.8 27B 11-token chat prompt, the 40,258,119-byte natural
response parsed as five blocks in 5.790 ms after a separately reported 6.336 ms
file read. Prepared multi-block restore measured 9.551 ms median versus 8.487 ms
for recurrent INT8 in the same run, a 12.5% restore-latency exchange for the
previously measured storage compactness. It retained 16/16 free-running and
15/15 teacher-forced top-1 tokens with the same 0.773716 maximum matched-logit
delta as the source cache run.

Using the measured natural ClickHouse read median of 20 ms gives a 35.341 ms
lower bound for `recurrent read -> parse -> restore`, about 6.36 ms below the
forced-single-block 41.704 ms bound. This is not an end-to-end cache-hit SLA:
it excludes prompt-envelope lookup and validation, exact KV retrieval, TCP
client framing, and the first continuation-token forward pass. The explicit
`cache_id` check is only a row-selection guard; the production envelope must
still validate model, tokenizer/template, ABI, codec, and prompt-token hashes.

### Matched complete-state physical gate (2026-08-15)

One guarded Qwen3.8 27B run exported all comparison inputs from the same
11-token chat state: a 40,257,628-byte recurrent p7 Native block, an
8,389,396-byte exact raw-KV artifact including its snapshot envelope, and a
47,768,476-byte complete recurrent-INT8 artifact containing the same raw KV.
P7 retained 16/16 free-running and 15/15 teacher-forced top-1 tokens; its
maximum matched-logit delta was 0.773716.

`scripts/qwen_qbit_clickhouse_matched_probe.sh` stored recurrent QBit and exact
KV as two physical parts, while the complete INT8 artifact used one part. The
blob columns both used `String CODEC(LZ4)`, and every input survived an exact
ClickHouse round trip.

| Complete state | Logical bytes | Compressed data bytes | Bytes on disk | Parts |
| --- | ---: | ---: | ---: | ---: |
| recurrent p7 QBit + exact KV | 48,647,024 | 36,047,499 | 36,049,812 | 2 |
| recurrent INT8 + exact KV | 47,768,476 | 38,470,261 | 38,470,759 | 1 |

The QBit layout is 1.84% larger before ClickHouse compression but 6.293%
smaller on disk after including the second-part overhead. This closes the
matched physical compactness question for this one state distribution. It does
not prove a general compression ratio: prompt length, KV share, model, QBit
precision, ClickHouse version/settings, and part size can change the result.
The complete diagnostic latency corridor is measured below. Production envelope
lookup and validation remain open.

### Complete warm cache-hit corridor gate (2026-08-15)

The guarded Qwen3.8 27B probe re-read the exact recurrent QBit, raw-KV, and
complete INT8 outputs above. Both routes used prepared Metal state and the same
cached first token. Seven samples alternated QBit/INT8 order as ABBA; every
first post-restore token matched token id 198.

| Warm in-process phase | QBit + exact KV | Recurrent INT8 + exact KV |
| --- | ---: | ---: |
| local artifact read | 8.148 + 1.316 ms | 11.556 ms |
| parse + structural validation | 7.909 ms | 0.065 ms |
| prepared Metal restore | 9.828 ms | 8.913 ms |
| first post-restore forward | 69.405 ms | 68.825 ms |
| complete median | 94.833 ms | 89.541 ms |

The paired median QBit-minus-INT8 delta was +8.473 ms; the ratio of format
medians was 1.0591, so QBit was 5.9% slower in this diagnostic corridor while
remaining 6.293% smaller on disk. The paired pre-forward cache-ready delta was
+8.726 ms. QBit's maximum first-post-restore logit delta was 0.578329 versus
0.182318 for INT8; both retained the exact expected token.

The matched ClickHouse probe independently measured warm server-side medians of
20 ms for recurrent QBit, 5 ms for exact KV, and 27 ms for the complete INT8
blob. Replacing, rather than adding to, the local file-read phases with those
unpaired server timings gives a diagnostic estimate of 110.369 ms for QBit and
104.985 ms for INT8, or about a 5.1% QBit latency cost. This is not a production
SLA: the ClickHouse samples were not interleaved with model execution and a
future TCP client may change the balance.

The dominant format-specific gap is strict Native parsing and validation, not
Metal reconstruction or the first forward. QBit reads fewer physical bytes and
was about 2 ms faster in both the ClickHouse and local-file read comparisons;
its 7.844 ms parse/validation disadvantage consumes that gain. The next safe
latency move is to eliminate redundant validation through a versioned trusted
envelope or fuse validation with restore, without weakening malformed-stream
rejection.

### Versioned admission-envelope gate (2026-08-15)

The internal v1 envelope now derives the narrow ClickHouse `cache_id` from a
full lookup identity and retains all identity fields for collision-safe
post-lookup comparison. It additionally binds the complete state ABI, exact KV
artifact, recurrent logical content and shape, exact-known-span validation, and
cached next token. A successful `Admission` retains the already parsed views so
subsequent in-process restores can reuse the validation result without hashing
the same immutable buffers again.

The real 38,304-row Qwen3.8 recurrent artifact was hashed in both its canonical
single-block form and the natural five-block ClickHouse response. Their raw
file SHA-256 values differ, but their logical digest is the same:
`0de2...d98b`. In a guarded release probe, strict parsing took 5.430/5.388 ms
and the logical digest took 15.602/14.503 ms for the one/five-block forms. A
fresh final run measured 6.488/4.552 ms parse and 14.835/13.990 ms digest;
the difference is ordinary local timing variance, not a format change.
This digest cost was not included in the earlier 94.833 ms complete QBit
corridor. It is a material first-cold-admission cost, not a free optimization;
eliminating it requires a trusted transport/storage certificate or validation
fused with data consumption, not merely trusting the self-described manifest.

### Bounded ClickHouse HTTP client gate (2026-08-15)

The internal client now owns three generation-scoped MergeTree tables:
recurrent QBit rows, one exact-KV blob, and a manifest. It inserts both artifact
tables synchronously before publishing the manifest. Readers first select one
unexpired manifest by the narrow cache id plus the full lookup key, then fetch
only that random 256-bit generation. TTL cleans up expired committed and orphan
artifact generations in the background; merges are not a visibility barrier.

Focused fake-transport tests cover manifest-last ordering, failure before
publication, strict first admission, misses without artifact reads, malformed
or oversized manifests, safe SQL identifiers, invalid generation ids, per-file
and combined allocation bounds, streamed reads, and resident admission reuse.
The guarded QBit suite passed 34
examples with zero failures or errors; four Metal-only examples were pending on
the unavailable device.

A real HTTP/SQL gate against local ClickHouse 26.7.1.1315 then stored the natural
five-block Qwen3.8 artifact: 40,258,119 recurrent Native bytes plus an 8,389,396
byte exact-KV artifact. The synchronous three-insert save took 85.193 ms; the
first strict lookup, including both artifact reads, parsing, exact-KV SHA-256,
and reblocking-stable recurrent digest, took 80.150 ms. Rechecking the manifest
and reusing the immutable process-local admission took 2.115 ms. Active parts
occupied 34,605,799 recurrent bytes, 1,468,224 KV bytes, and 2,017 manifest
bytes on disk (36,076,040 total). This is 6.22% below the matched 38,470,759-byte
INT8 part from the earlier complete-state gate, not a general compression SLA.

The resident result is deliberately byte-bounded and disabled by default. It
does not make the first cold admission trusted, and it retains the artifact
buffers while resident. Runtime integration therefore needs an explicit memory
budget and a miss/rejection fallback before promotion.

### Incremental session checkpoint gate (2026-08-15)

A guarded Qwen3.8 27B run exercised a five-action transcript at 323/512 prompt
tokens, a sixth cold restore, and a rollback branch from checkpoint three.
Every model process started with at least 82% free system memory. Explicit
checkpoint lookup, QBit anchor admission, exact suffix replay, delta write, and
rollback all completed without rejection, transport failure, restore failure,
or write failure. A matched no-cache action-five run produced the same three
token ids as the checkpoint chain. After hardening the exceptional cleanup
path, a fresh one-state-at-a-time repeat measured 13,819.210 ms for the full
anchor and 3,208.706 ms for its first cold continuation, again with 82% free
memory and zero cache failures.

The first full anchor exposed and closed two correctness falsifiers. Whole-text
BPE tokenization was not assumed to be append-stable, and the Qwen generation
prompt's hidden `<think>...</think>` suffix was not treated as part of the
caller transcript. The runtime now re-tokenizes the completed transcript and
requires both text-boundary and exact token-prefix equality at lookup. The
token-parallel anchor rebuild then produced a non-finite layer-1 ConvState on
this boundary; QBit rejected it before publication. Full anchors therefore use
a sequential exact-boundary prefill unless the fresh recurrent rollback gate
below proves the reusable live-KV prefix and replays the divergent boundary
suffix. QBit encoding remains the fail-closed non-finite publication guard.

| Cold-process phase | Prompt tokens | Reused / replayed | Generate total | Checkpoint write |
| --- | ---: | ---: | ---: | ---: |
| full anchor, action 1 | 79 | 0 / 0 | 13,350.687 ms | 2,529.422 ms |
| delta, action 2 | 140 | 80 / 60 | 3,057.119 ms | 3.648 ms |
| delta, action 5 | 323 | 80 / 243 | 4,319.855 ms | 3.624 ms |
| cold restore, action 6 | 350 | 80 / 270 | 4,600.911 ms | 3.496 ms |
| rollback checkpoint 3 | 229 | 80 / 149 | 4,091.331 ms | 3.953 ms |
| matched no-cache action 5 | 323 | 0 / 0 | 4,792.591 ms | none |

The action-five cold checkpoint path was 9.9% faster than its matched full
prefill in this single local sample. This is not a latency SLA: the fixed
80-token anchor means the advantage shrinks as replay grows. Periodic anchors
rebuilt after a QBit restore are still synchronous and fully sequential;
only fresh ordinary-session anchors admit the default-off background publication
path measured below.

ClickHouse 26.7.1.1315 stored the one full anchor in 32.98 MiB of recurrent
QBit parts plus 10.26 MiB of exact-KV parts. Seven checkpoint rows, including
the restore and rollback branches, occupied 6.42 KiB compressed versus
18.16 KiB uncompressed. Individual JSON envelopes grew from 1,529 bytes for
the anchor checkpoint to 2,909 bytes for the deepest measured delta. Thus the
incremental metadata is compact; retained full anchors, not MergeTree merges or
delta rows, dominate storage and materialization cost.

### Exact-anchor recurrent rollback gate (2026-08-15)

A final guarded Qwen3.8 27B A/B used the same 63-token first-action prompt,
three-token continuation, model, ClickHouse instance, and process-level memory
floor. Both routes emitted token ids `[66793, 12, 16]` (`checkpoint-1`) and
completed with 82% free system memory and zero lookup, admission, restore,
transport, or write failures.

| Fresh first-anchor route | Boundary replay | Exact-anchor materialization | Generate total | Checkpoint write |
| --- | ---: | ---: | ---: | ---: |
| recurrent rollback | 9 tokens | 849.761 ms | 6,749.225 ms | 2,514.663 ms |
| full sequential fallback | full boundary | 6,055.130 ms | 11,749.099 ms | 2,455.675 ms |

For this sample, recurrent rollback made exact-anchor materialization 7.1x
faster (85.97% less time) and reduced the complete request by 42.56%. This is a
scoped fresh-session result, not a general session-cache SLA: synchronous
ClickHouse encoding/write still costs about 2.5 seconds here, and periodic
anchors renewed from p7-restored state intentionally retain full sequential
materialization. The evidence decays when the model, tokenizer, chat template,
prefill/checkpoint kernels, or Metal state-ownership route changes.

### Async anchor publication gate (2026-08-15)

A guarded Qwen3.8 27B sync/async A/B used the same 63-token prompt, model,
ClickHouse instance, process-level memory floor, and four-token generation
limit. Both routes emitted token ids `[66793, 12, 16]` (`checkpoint-1`). The
async route returned the preallocated checkpoint identity as pending after
capturing an immutable host snapshot; the smoke then drained it explicitly to
separate response latency from publication cost.

| Fresh first-anchor route | Generate response | Host snapshot | QBit + ClickHouse | Explicit drain |
| --- | ---: | ---: | ---: | ---: |
| synchronous publication | 6,215.919 ms | included | 2,439.522 ms | none |
| asynchronous publication | 3,537.305 ms | 38.670 ms | 2,345.053 ms | 2,343.060 ms |

Moving publication off the response boundary reduced this first-action sample
by 2,678.614 ms (43.1%) without changing generated tokens. It did not eliminate
the work: this diagnostic drained immediately, and the runtime must also drain
before the next generation or close. Natural idle-time overlap and sustained
multi-session throughput remain unmeasured; the single-flight capacity is one.

The second action restored the anchor and emitted the same token ids
`[66793, 12, 17]` as the synchronous route. A third action in a new process hit
the same chain with zero cache errors: total generation was 3,029.360 ms,
including 2,673.500 ms of restore. The resulting checkpoint depths were
`0 -> 1 -> 2`, with one full anchor and two compact exact-token deltas.

ClickHouse stored the full anchor in 32.98 MiB of recurrent QBit parts plus
10.18 MiB of exact-KV parts, about 43.16 MiB compressed in total. The three
checkpoint rows occupied 4.31 KiB compressed versus 6.34 KiB uncompressed.
An initial 1 GiB ClickHouse memory ceiling rejected the large exact-KV read
safely; a 2 GiB server ceiling inside a 3 GiB process-tree guard completed the
A/B while the host retained at least 84% free memory. These figures are a
bounded local gate, not a storage-ratio, latency, or server-sizing SLA.

### Session-scoped async barrier gate (2026-08-15)

`bin/qwen35_qbit_session_barrier_probe.cr` covers the two paths the unit specs
cannot reach: which requests the pending-checkpoint barrier stops, and what
teardown does after a publication fails. Both need a real model and a real
ClickHouse instance, so the probe runs under `scripts/run_safe.sh`.

The discriminator is wait accounting. `async_checkpoint_wait_time` advances only
inside a drain, so a request that must not be serialized has to show an exactly
zero delta, and a request that depends on the in-flight row has to show a
positive one. This is an accounting identity rather than a timing threshold, so
it does not race.

The interleave phase runs two sessions plus a sessionless request against one
runtime (Qwen3.8 27B, `max_seq` 512):

| Step | Pending publication | Added wait | Outcome |
| --- | --- | ---: | --- |
| `a1` | enqueues A | 0.001 ms | checkpoint returned pending |
| `sessionless` | A in flight | 0.000 ms | completed on the writer thread, unwaited |
| `b1` | enqueues B | 0.001 ms | checkpoint returned pending |
| `a2` | B in flight | 0.000 ms | hit A's own chain, 77 prefix tokens reused |
| `b2` | resolves B | 3,115.165 ms | waited for its own row, then hit |
| `a3` | none | 0.000 ms | hit, 77 prefix tokens reused |

Final accounting was 2 enqueued, 2 completed, 0 pending, 0 write failures, and
all five published parent links matched in ClickHouse (`a2 -> a1`, `a3 -> a2`,
`b2 -> b1`, with `a1` and `b1` rootless). `a2` is the load-bearing row: it
skipped the barrier while another session's publication occupied the slot and
still continued its own chain instead of forking a new anchor.

The failure phase drops the checkpoints table after schema creation, so reads
resolve and the terminal insert fails inside the writer thread. `a1` enqueued a
doomed publication and returned normally; `b1`'s enqueue drained that failure,
recorded it as a write failure, and also returned normally; the sessionless
request added zero wait while publication was failing. The explicit durability
barrier then raised twice — once as `flush failed`, once as
`publication failed` — and stopped. Close reported no further failure because
the errors had already been consumed, the runtime rejected a later barrier as
closed, and a repeated close was silent, so teardown completed.

Both phases pass unchanged on Qwen3.5-0.8B, Qwen3.5-9B, and Qwen3.8-27B. This
is a bounded local gate on one host; it fixes the ordering contract, not
throughput under real concurrency, and the single-flight capacity is still one.

### ClickHouse boundary probe

Local ClickHouse 26.8.1.1 on the Apple M2 Max was tested with incompressible
`String CODEC(NONE)` payloads to isolate part movement from QBit quality:

- a 50 MiB part inserted in 42-48 ms;
- reading and hashing one 50 MiB row took 22-23 ms with the filesystem cache
  warm;
- `OPTIMIZE FINAL` merged eight 8 MiB parts (64 MiB) in 98 ms;
- `OPTIMIZE FINAL` merged eight 50 MiB parts (400 MiB) in 649 ms.

These are local forced-merge throughput probes, not a server scheduling SLA. An
inserted row was visible while eight parts still existed, so a background merge
is not part of cache-hit latency. On this host the measured 50 MiB read was
roughly 50 times shorter than scalar p7 decode.

A second probe used the intended tile shape directly: one row per 1024 codes in
`QBit(Int8, 1024)`, 8192 tiles per inserted part, and eight active parts. Random
full p8 codes occupied 64.51 MiB and merged in 150 ms. Clearing the code LSB to
represent p7 made the eighth plane all-zero; ClickHouse compressed the eight
parts to 56.51 MiB, merged them in 104 ms, and read+hashed the seven retained
plane subcolumns in 27 ms. This validates the compact physical direction and
selective-plane read path.

The SQL array-construction route is a negative result: constructing arrays and casting
them to QBit took 116-156 ms per 8 MiB logical part because ClickHouse had to
transpose the codes. Cogni-ml now writes the pre-transposed revision-zero Native
QBit streams, sends them through the bounded HTTP client, and restores the same
columnar representation directly on Metal. Native TCP framing/compression
remains unimplemented and may be benchmarked later; it is not required for the
default-off HTTP route. Runtime miss/fallback wiring remains open.

Production storage should batch all tiles for one or more cache keys per insert
to avoid part-count pressure. Background merges should perform cleanup and
compaction, never admission or visibility.

### Direct cold artifact to adaptive resident KV gate (2026-08-24)

`QwenQBitStateSnapshot.restore_admitted_native_stream_into_adaptive` is the
smallest restore corridor that avoids recreating a persistent Float32 KV owner.
It decodes the admitted p7 recurrent Native stream directly into the prepared
Metal recurrent buffers, then uploads one attention layer's exact live K/V
prefix to the adaptive GPU packer in fixed 512-token chunks. The two temporary
shared buffers are released before the next chunk; for Qwen3.8 geometry their
combined payload is bounded near 6 MiB instead of growing with the full context.
All 16 attention layers finish with adaptive ownership only. Validation of
identity, record completeness, shape, positions, and empty target ownership
happens before publication; after a device error the caller must discard the
target state.

A guarded Qwen3.8-27B Q4_K_M multi-turn row used an 829-token checkpoint, a
34-token follow-up, and the default-off
`p4;27=bf16,43=bf16,47=bf16,51=bf16` map:

| Phase or quality signal | Measured result |
| --- | ---: |
| exact anchor prefill | 9,515.721 ms |
| synchronous checkpoint serialization | 15,650.038 ms |
| first / second in-memory admitted restore | 64.147 / 35.015 ms |
| adaptive follow-up replay | 1,470.185 ms |
| adaptive response decode | 754.545 ms |
| source snapshot / cold payload | 276,430,848 / 148,917,368 bytes (`1.8563x`) |
| recurrent Native / exact live-KV / total payload | 40,257,628 / 108,659,740 / 148,917,368 bytes |
| raw / resident live attention KV | 108,658,688 / 30,046,208 bytes (`3.6164x`) |
| top-1 / exact top-1 covered by restored top-2 | `8/8` / `7/7` |
| ranked top-2 / unordered overlap | `9/14` / `9/14` |
| output-row token ECS mean/minimum | `1.0 / 1.0` |

Exact and restored free runs both reached EOS and emitted the identical sentence
`Their sum is 95.`. The differing runner-up candidates therefore remain an
explicit distribution-drift warning, not a response-level failure. Meaning,
greedy top-1, exact-top-1 coverage, and ECS form the admission gate; ranked and
unordered top-2 parity remain separately visible diagnostics.

The restore numbers begin after artifacts are already present and parsed in
memory. They exclude filesystem and ClickHouse lookup/read, so they are not
full cold-hit latency. The exact live-KV artifact is still raw Float32 on disk,
and the synchronous 15.7-second recurrent QBit encoding is not on an acceptable
response boundary; background publication remains the intended dual frame.
`Qwen35NativeRuntime`, checkpoint renewal from an adaptive owner, ClickHouse
composition, and packed adaptive KV persistence remain fail-closed or open.
These are single guarded rows; replay/decode variance is not a throughput SLA.

### ClickHouse cold admission to adaptive resident KV gate (2026-08-24)

The same probe can now publish the p7 recurrent stream plus exact V3 live-prefix
KV artifact through the bounded ClickHouse store, construct a fresh store with
its resident admission cache disabled, perform longest-prefix lookup and strict
admission, and restore the returned views directly into adaptive GPU owners.
The lookup also proves the expected 829-token anchor and 34-token suffix replay
boundary before Metal state mutation. A focused negative spec rejects a V3 KV
payload whose live-row count differs from the envelope `prefix_len`.

Two guarded Qwen3.8-27B Q4_K_M rows used the default-off
`p4;27=bf16,43=bf16,47=bf16,51=bf16` map and an isolated local ClickHouse HTTP
server:

| Anchor | Exact prefill | Serialize + publish | Cold lookup + admit | Prepare + adaptive restore | Suffix replay / first token | Full cold hit to first token | Live KV density |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 445 tokens | 8,341.259 ms | 2,510.841 ms | 223.146 ms | 49.518 ms | 739.484 ms | 1,012.148 ms | 3.4975x |
| 829 tokens | 10,094.536 ms | 2,633.929 ms | 327.683 ms | 59.923 ms | 787.860 ms | 1,175.466 ms | 3.6164x |

Both exact and restored 829-token runs reached EOS and emitted `Their sum is
95.` with top-1 `8/8`, exact top-1 covered by restored top-2 `7/7`, and
output-row token ECS `1.0`. Ranked and unordered top-2 agreement remained
`9/14`, so the existing runner-up distribution warning is unchanged. The full
cold boundary is about `8.6x` shorter than the matched exact prefill on this
single host row; it excludes process/model startup and is not a throughput SLA.
ClickHouse background merges were not required for visibility and are outside
the lookup critical path measured here.

This measurement composed strict durable admission with direct adaptive
restore, leaving NativeRuntime ownership as the next gate. The following slice
now closes that gate only for non-session hits: ordinary F32 miss fallback and
session checkpoint/snapshot routes remain fail-closed until adaptive snapshots
have their own certificate.

### NativeRuntime non-session adaptive cache-hit gate (2026-08-24)

`Qwen35NativeRuntime` now selects adaptive ownership only when all of these are
true: a QBit cache is configured, the request is non-session, the backend is
Metal, a strict admission hit exists, and an adaptive resident-KV environment
selector is present. The existing adaptive restore primitive requires every
full-attention layer, so a partial map cannot become a successful adaptive hit.
Ordinary misses explicitly suppress adaptive allocation and retain Float32 KV;
this preserves snapshot-based writeback. Session lookups and checkpoint renewal
also remain Float32. Adaptive hits skip writeback because adaptive snapshot
capture is still rejected.

The restore has two failure classes. Admission/shape failures happen before
device mutation and may fall back to ordinary prefill. Once recurrent decode or
KV packing begins on Metal, failures become `AdaptiveRestoreDeviceError`; the
runtime releases the candidate and aborts instead of allocating a second heavy
state. The public QBit statistics add `adaptive_hits`, incremented from actual
state ownership rather than from the environment selector.

A guarded Qwen3.8-27B Q4_K_M exact-prompt seed/hit used `max_seq=64`, eight
generated tokens, separate processes, the coarse adaptive map, and the local
ClickHouse server. Seed and hit both emitted token ids
`[21,11,220,22,11,220,23,11]` and text `6, 7, 8,`. The hit reported
`adaptive_hits=1`, `hits=1`, and zero restore failures. A second empty table
started with the same adaptive environment and reported `misses=1`, `writes=1`,
and zero failures, proving that the miss state remained snapshot-capable F32.
The focused regression set passed `53 examples, 0 failures, 0 errors, 1
pending`.

The short prompt measured about 6.74 seconds for the adaptive hit versus 7.68
seconds for the seed prefill, only a roughly 1.14x observed improvement with
cold process/kernel setup and no paired quiet-host design. This is an ownership
and lifecycle gate, not a speed claim. The larger direct cold-corridor row above
remains the relevant latency feasibility evidence.

### Compact adaptive miss, writeback, and cold-hit gate (2026-08-24)

The next bounded slice supersedes the earlier Float32-miss restriction for
non-session requests. When the Metal adaptive tier map is enabled, a cache miss
now creates sole adaptive KV owners before prefill, so full-attention KV is
packed as it is produced rather than first accumulating in Float32. Session
requests and requests without the adaptive selector keep the raw path. DeltaNet
recurrent state remains uncompressed in live GPU memory because its size is
fixed; writeback captures and QBit-encodes it only for the cold artifact.

Writeback reuses the existing CQKV V3 envelope and ClickHouse tables. It stores
the canonical resident K/V payloads directly, without expanding them to
Float32, and records the exact tier layout in an adaptive artifact codec. Cache
lookup therefore cannot alias two tier layouts. Cold restore decodes recurrent
records into their final Metal buffers and copies the canonical packed K/V into
fresh sole adaptive owners. K and V are both validated and copied before the
prefix length is published. A rejected or partially restored candidate is
discarded; raw and session artifacts retain their byte-for-byte cache identity.

A guarded Qwen3.8-27B Q4_K_M run used `max_seq=4096`, a 2,172-token prompt,
eight generated tokens, separate baseline/seed/hit processes, and
`p4;27=bf16,43=bf16,47=bf16,51=bf16`. The baseline took 24,540.372 ms, adaptive
seed 28,599.536 ms, and cold adaptive hit 6,048.205 ms: about 4.06x faster than
the matched full prefill. The hit restored all 2,172 tokens with no suffix
replay; lookup took 314.146 ms and state restore 95.599 ms. All three runs
emitted `Their sum is 95.` with identical token ids.

The live raw KV represented 284,688,384 bytes and its canonical adaptive
payload 75,621,404 bytes, or `3.7647x` density for this conservative four-BF16-
layer map. ClickHouse used 107,809,078 bytes on disk for the complete KV,
recurrent, manifest, and prefix-index rows. The save-side host guard accounted
156,893,184 bytes of recurrent Float32 source plus 75,621,404 bytes of already
compact KV, 232,514,588 bytes total, and stayed below its 256 MiB boundary. This
does not establish an eightfold window: the all-P4 control is about `7.1x`, while
the currently qualified quality map is `3.7647x`.

The aligned quality probe preserved exact top-1 `8/8`, ranked top-2 `12/14`,
top-2 set overlap `12/14`, exact-top-1 coverage `7/7`, and output-row embedding
cosine similarity mean/minimum `1.0/1.0`; both paths reached EOS with identical
meaning. Maximum top-2 logit delta was `1.0112152`, so response preservation is
not a logit-parity claim. The compact snapshot unit gate separately proves that
the canonical payload survives snapshot/restore byte-for-byte; together these
are compositional evidence for the same resident representation and compute
path, not a direct top-2 trace inside the cold NativeRuntime process.

The implementation intentionally added no new container, table, or checkpoint
graph. At this 2026-08-24 stage, remaining guard-only surfaces were adaptive
session checkpoints,
longer-than-this-row save peak memory, asynchronous/speculative decoding, and
production-default policy. The long ClickHouse probe also depends on sending
large prefix SQL in the HTTP request body; URL query transport hits the server's
header limit at this prefix length.

### Streaming recurrent writeback peak gate (2026-08-24)

Adaptive writeback no longer captures all recurrent Float32 records and retains
all encoded QBit records before building the Native body. It now reads one Conv
or SSM owner, encodes it, appends one legal Native block, and retains no
encoder-owned reference to the per-record source or QBit objects when that
append returns. The final Native HTTP body still remains resident until
ClickHouse accepts it. This keeps the existing envelope, tables, record schema,
and logical artifact digest; it does not introduce a new checkpoint format.

A release-mode allocation probe used 96 recurrent records with the exact
Qwen3.8 source total of 156,893,184 bytes. The old full-snapshot route reached
319,389,696 bytes maximum RSS; the per-record route reached 184,745,984 bytes,
a 42.2% reduction. Internal encode time was 783.272 versus 781.544 ms. The
logical SHA-256 was identical. Multiple Native headers increased the 40.26 MiB
body by 11,684 bytes, or 0.029%. The probe used equal-sized, zero-valued
synthetic records to isolate allocation and serialization ownership; it is not
a real per-record shape distribution, full model RSS, compression-quality, or
throughput measurement.

ClickHouse accepted a two-record, five-tile concatenated Native test with all
4,000 values and both record identities. The real 2,172-token Qwen3.8 seed then
inserted 38,304 recurrent QBit tile rows covering 39,223,296 values and all 96
recurrent record identities. Compressed data across the isolated cache tables
remained 107,809,078 bytes. In separate guarded model processes, seed generation
took 32,189.108 ms with 3,454.824 ms attributed to writeback; the cold hit took
5,579.489 ms, including 327.929 ms lookup and 94.149 ms restore. Both emitted
`Their sum is 95.` with identical token ids and no cache, transport, restore, or
write failures. This single-host row confirms lifecycle compatibility, not a
new throughput SLA.

At this checkpoint, the remaining save-side allocation target was the final
accumulated Native HTTP body and its transport copy. The next bounded-streaming
slice below closes that target. Session checkpoints remain raw, the 256 MiB
logical source guard remains in force, and the existing top-1, top-2, ECS, EOS,
and meaning boundaries are unchanged.

### Bounded HTTP recurrent writeback gate (2026-08-24)

Adaptive non-session writeback now sends each recurrent Native block directly
to ClickHouse under HTTP backpressure. The request body retains at most one
captured Float32 record and one encoded block; it no longer accumulates the
approximately 40 MiB Native body. The schema, envelope, logical digest, compact
KV artifact, read path, and raw session path are unchanged. Exact KV is fully
validated before the first recurrent record is captured or uploaded. The
manifest remains the commit marker, so a transport or late cross-artifact
validation failure can leave only invisible TTL-governed rows.

On the same synthetic 96-record, 156,893,184-byte recurrent source, maximum
process RSS fell from 159,563,776 to 15,941,632 bytes, a 90.0% reduction. Both
paths produced 40,269,312 Native bytes and logical SHA-256
`80e0be2535a904a351da4eda3096051934110de6a32a425cfd13d0743e9189ba`.
Internal encode time was 490.304 versus 515.717 ms, a 5.2% cost in this one
synthetic run. This isolates serialization ownership; it is not whole-model
unified-memory or throughput evidence.

An actual chunked HTTP insert was admitted by ClickHouse before the model run.
The guarded 2,172-token Qwen3.8 seed then completed in 25,243.030 ms with
2,724.428 ms writeback. A separate cold hit completed in 5,449.876 ms, including
319.774 ms lookup and 93.424 ms restore, with all 2,172 prefix tokens reused and
no suffix replay. Both emitted `Their sum is 95.` with ids
`[33645,2542,369,220,24,20,13]` and zero cache failures. ClickHouse contained
38,304 recurrent rows, 39,223,296 values, 96 record identities, and 107,809,078
compressed bytes, matching the accumulated-body lifecycle.

The 2,172-token prefix lookup also used the pre-existing working-tree change
that sends large read-only SQL through the HTTP body instead of the URL. That
separate transport fix is not part of this write-side commit or promoted by
this evidence.

For this transport-only slice, exact token identity makes the positionwise ECS
mean/minimum `1.0/1.0` and preserves the sentence meaning. The runtime smoke
does not expose fresh runner-up logits, so the applicable top-2 evidence remains
the earlier aligned resident-quality result (`12/14` ranked and set overlap,
maximum top-2 logit delta `1.0112152`), not a new top-2 measurement. At this
checkpoint KV upload still owned its compact byte artifact; the next slice
closes that write-side allocation. Cold reads remain buffered.

### Bounded adaptive-KV writeback gate (2026-08-24)

Adaptive non-session writeback now frames and validates the existing CQKV V3
artifact one K or V record at a time. It snapshots only the selected resident
base/sidecar, produces its canonical payload, and releases that record before
pulling the next one. The wire bytes, SHA-256, envelope, cache identity, tables,
and cold read path are unchanged. This is bounded host staging, not direct
Metal-to-HTTP zero-copy: one selected base/sidecar plus one canonical payload
can still coexist transiently.

The compact KV stream is uploaded first, recurrent Native rows second, and the
manifest and prefix index last. A malformed or interrupted stream can therefore
leave TTL-governed orphan rows, but cannot publish an admissible generation.
The one-shot bodies are deliberately not retried; callers must capture a fresh
body. Raw/session checkpoints keep their existing format and ordering.

A release synthetic probe used the qualified Qwen3.8 long-context KV artifact
size of 75,621,404 bytes. The accumulated path reached 288,030,720 bytes maximum
RSS; the one-record path reached 15,384,576 bytes, an 18.7x reduction. Both
emitted exactly 75,621,404 bytes with SHA-256
`38f63faf4fbe03d5402e70ef1e1065f6b893692d5c5014dc5384c8b50a73c709`;
the largest retained record payload was 2,363,136 bytes. This isolates
serialization ownership rather than whole-model unified-memory pressure.

An isolated ClickHouse/Qwen3.8-27B Q4_K_M seed and cold hit then exercised the
new branch at `max_seq=64` under
`p4;27=bf16,43=bf16,47=bf16,51=bf16`. Seed writeback took 2,516.971 ms. The hit
looked up the entry in 95.606 ms, restored it in 23.265 ms, reused all 38 prompt
tokens, and emitted the same ids `[21,11,220,22]` and text `6, 7` with zero
cache failures. ClickHouse stored one 1,324,060-byte KV payload plus 38,304
recurrent rows covering 39,223,296 values and 96 recurrent identities.

Focused stream/envelope/resident tests passed, and 130 applicable QBit/Metal
examples passed in resource-isolated processes with one optional model-backed
pending. The legacy async-writer concurrency example remains incompatible with
Crystal 1.21 when `Isolated#join` is called from a raw `Thread`
(`Thread#scheduler nil`); it is outside this KV diff and was reproduced alone.
At this stage, direct device streaming, buffered cold reads, adaptive session
checkpoints, and production-default policy remained guard-only.

### File-backed adaptive cold-read gate (2026-08-25)

Adaptive non-session artifact responses now flow from the ClickHouse HTTP body
into an already-unlinked temporary file under the existing byte caps. The file
is mapped read-only, strict envelope admission operates on zero-copy slices,
and the admission retains a typed owner for the mapping lifetime. The schema,
wire bytes, manifest-last authority, cache identity, retry policy, and GPU
restore boundary are unchanged. Raw artifacts and session anchors deliberately
remain on the previous in-memory transport path.

This is bounded managed-heap staging, not direct ClickHouse-to-Metal streaming.
Integrity and layout validation still scan the complete file-backed artifacts,
and the host must have temporary disk space for both responses. A short write,
oversize response, digest/layout failure, mmap failure, or disk error fails
before an admission is returned. The directory entry is removed before the
HTTP read begins, so normal completion and handled failure leave no named
spool. An abrupt process loss in the narrow interval between tempfile creation
and unlink can strand that one generated name. The descriptor closes
immediately after mmap; an explicit backing owner retains the mapping until the
admission is finalized and then unmaps it. Callers must retain the admission
while using its zero-copy views.

The focused transport suite passed 27 examples, including adaptive streaming,
post-GC mapping lifetime, no named spool during the write, truncation, combined
budget rejection before the KV request, and an explicit raw-path preservation
check. The applicable CPU regression partition passed 72 examples with no
failures or pending cases.

An isolated Qwen3.8-27B Q4_K_M seed and three separate cold-hit processes used
`max_seq=64` and `p4;27=bf16,43=bf16,47=bf16,51=bf16`. Seed writeback took
2,409.128 ms. Cold lookup took 180.660, 120.926, and 114.186 ms (median
120.926 ms); restore took 28.640, 27.760, and 29.542 ms. Every hit reused all
38 prompt tokens with no suffix replay or cache failure and emitted the same
ids `[21,11,220,22]` and text `6, 7`. Exact token identity preserves
positionwise ECS at `1.0`; fresh top-2 logits were not captured. The earlier
single buffered lookup at 95.606 ms is only a historical reference, not a
matched latency comparison, so the read overhead is not yet promoted as a
stable percentage.

At this stage, direct network-to-device streaming, adaptive session
checkpoints, response spool pooling, retries, and production-default policy
remained separate slices.

### Matched adaptive cold-read sink A/B (2026-08-25)

A sink-only ABBA probe compared two release binaries built from `b635b72f`.
Both used the same model, 2,172-token prompt, adaptive tier map, ClickHouse
generation, large-query POST framing, `HTTPTransport#post_into`, response byte
caps, parsing, admission, restore, and eight-token decode. The temporary control
changed only `MappedResponse.fetch`: one binary retained the two responses in
managed `IO::Memory` storage, while the other used the committed immediately
unlinked tempfile plus read-only mmap route. A separate mmap hit warmed the
host caches before the measured `buffered, mmap, mmap, buffered` order.

The logical response body was 115,879,032 bytes: 40,257,628 Native recurrent
bytes plus 75,621,404 compact KV bytes. Mean maximum process RSS was
771,555,328 bytes for the buffered control and 607,870,976 bytes for mmap, a
163,684,352-byte or 156.1 MiB reduction (`-21.2%`). The two ranges did not
overlap: buffered measured 756,056,064 and 787,054,592 bytes; mmap measured
607,633,408 and 608,108,544 bytes. Mean macOS peak memory footprint was
814,731,680 versus 535,040,248 bytes (`-34.3%`). The larger footprint delta is
consistent with the buffered control's transient `IO::Memory` plus retained
byte copy; it is not an additional artifact-size claim.

Mean lookup was 333.994 ms buffered and 311.965 ms mmap, corresponding to about
330.9 and 354.2 MiB/s over the logical response bytes. Mean restore was 96.691
and 96.144 ms. Mean generation was 4,623.154 and 4,360.316 ms, and mean wall
time was 5.015 and 4.650 seconds. Both mmap observations were faster, but two
samples per arm under normal desktop background load are insufficient for a
stable latency or throughput percentage. All four hits reused 2,172 tokens,
replayed no suffix, reported `hits=1`, `adaptive_hits=1`, and zero failures,
and emitted ids `[33645,2542,369,220,24,20,13]` and `Their sum is 95.` Exact
token identity gives positionwise ECS mean/minimum `1.0/1.0`; fresh top-2
logits were not captured.

This is a process-cold but host-cache-warm measurement. `/usr/bin/time -l`
reported zero block input and output operations in every measured process, so
it does not measure physical cold-disk throughput. Mapped pages still consume
OS file-cache/unified-memory capacity and require temporary disk space; process
RSS and peak footprint therefore do not prove the same reduction in total
system unified memory. The seed wrote one complete generation in 2,811.152 ms,
and ClickHouse held 107,809,078 compressed bytes, but its initial clean-commit
prefix lookup exposed the known large-query URL limit with `HTTP 500: Field
value too long`. The transport fix now sends body-free SQL in the POST body for
both heap and file response sinks while keeping `input()` payloads in the body
and their SQL in the URL. A socket-backed query larger than 128 KiB fails on
`b635b72f` with `HTTP 414: URI Too Long` and passes with the fix; the complete
ClickHouse-cache spec passes `30/30`. A real ClickHouse 26.7.1 probe also
accepted a 163,852-byte body query and returned `1\n`. This closes the HTTP
framing gate for the measured 2,172-token corridor without changing the
separate ClickHouse query-size or candidate-count limits.

A fresh release binary built from a detached `7211fc03` worktree then repeated
the complete long-prefix lifecycle against a new ClickHouse table. The guarded
adaptive seed used 128 filler records, produced exactly 2,172 prompt tokens,
completed generation in 30,118.824 ms, and wrote one clean miss in 2,720.971
ms. A separate process restored all 2,172 tokens with no suffix replay or
writeback; generation took 7,315.455 ms, including 335.572 ms lookup and
102.201 ms restore. Seed and hit both emitted ids
`[33645,2542,369,220,24,20,13]` and `Their sum is 95.`. All rejection,
transport, restore, and write-failure counters remained zero. ClickHouse held
38,304 recurrent rows plus one KV, manifest, and prefix row, totaling
107,809,078 compressed bytes. Exact aligned tokens imply positionwise ECS
mean/minimum `1.0/1.0`; the runtime smoke still does not capture fresh top-2
logits. This is a fresh-process lifecycle gate, not a physical-cold-disk,
stable-throughput, or total-unified-memory measurement.

### Cache-engine contract

- The internal envelope makes cache keys content-addressed over model,
  tokenizer/chat-template, engine/state ABI, prompt tokens, QBit block/precision,
  and artifact codec version. The client checks this full identity after the
  narrow `UInt64` ClickHouse lookup.
- Insert every tile of one artifact in one batch under a fresh generation and
  persist expected tile count plus an artifact digest. Publish its manifest
  last. Restore fails closed on missing, duplicate, or mismatched tiles; it
  never waits for `FINAL` or a merge.
- Store recurrent state as 1024-code QBit tiles with per-tile mean, sigma, value
  count, record kind, layer, and tile ordinal. Keep live KV chunks separately
  because their length follows the cached sequence rather than the recurrent
  tile grid.
- Filter expiration at read time. TTL merges reclaim space asynchronously and
  are not an authority for whether a stale row is admissible.
- Send pre-transposed Native streams and read only retained plane subcolumns.
  SQL Array-to-QBit casts remain a diagnostic fallback, not the serving route.

### Adaptive prefill tile occupancy gate (2026-08-26)

A bounded synthetic Metal probe now separates adaptive attention-plus-pack from
pack-only work at the Qwen3.8 GQA6 shape (`24` query heads, `4` KV heads,
head dimension `256`, and a `64`-token append). Pack-only cost was only
`0.18--0.26 ms` in the stable rows, while a 3,072-token prefix took roughly
`24--31 ms`; BF16 packing is therefore not the dominant residual prefill cost.

The same release probe compared compile-time attention tiles under a 2 GiB
process-tree cap, a 35% system-memory headroom gate, and a quiet-host gate. For
the seven samples starting at prefix 3,072 and advancing by 64 tokens,
tile `15` measured `23.958/21.002/27.193 ms` for uniform
P4/BF16/F32, versus tile `16` A/A observations of
`26.403--26.350/24.240--23.879/31.152--31.180 ms`. Tile `24` was slower than
the baseline. Tile `12` was faster in isolation but failed the end-to-end gate:
its extra tile reductions did not reduce long-session replay. It remains a
rejected diagnostic, not a production setting.

Two guarded tile-16 baselines and two tile-15 observations then replayed the
same 2,391-token suffix into a 3,220-token Qwen3.8 session. Tile `15` measured
free replay `21,237.714/24,814.303 ms` and forced replay
`21,094.244/22,251.629 ms`. Tile `16` measured
`25,776.402/26,234.501 ms` and `26,520.454/27,574.286 ms`; against the faster
baseline and the slower candidate, the conservative reductions were
`3.7%/15.2%`. The candidate exact chunked controls were
`16,553.551/18,864.194 ms`, while the baselines measured
`16,764.830/23,078.128 ms`; that control spread is large, so no stable
throughput percentage is claimed beyond these bounded, non-overlapping replay
observations.

Both candidate observations preserved top-1 `8/8`, exact-top-1 coverage in
candidate top-2 `7/7`, ECS mean/minimum `1.0/1.0`, EOS, resident ownership,
`3.7647x` logical KV density, and the exact text `Their sum is 95.`. Ranked and
unordered top-2 overlap repeated at `11/14`, one slot below both tile-16
baselines (`12/14`), which is reported as numerical distribution drift from
the changed online-softmax reduction grouping rather than hidden by the
identical greedy trajectory.

The focused auto tile-15 and forced tile-16 Metal routes each passed `14/14`
with CPU-reference cosine `1.0` and maximum delta `3.72529e-08`. The complete
resource-isolated QBit/Metal suite, now including the admission policy checks,
passed `139` examples with zero failures or errors and one optional model-backed
pending example. The earlier intermittent Crystal 1.21 raw-thread async-writer
error did not reproduce in this run.

The host admits tile `15` only when Metal reports the measured `Apple M2 Max`;
every other or unknown device fails closed to tile `16`. The two variants use
distinct pipeline-cache keys, and `QWEN35_ADAPTIVE_GQA6_TILE=15|16` is an
explicit benchmark override. This does not change cache bytes, checkpoints,
publication, rollback, or resident memory. Other Apple GPUs, larger contexts,
stable multi-prompt throughput, and exact runner-up-logit preservation remain
open.

A dedicated command-buffer timestamp probe now separates the standalone
adaptive decode dispatch from host allocation, transfer, readback, and wall
noise. In a guarded tile-15/tile-16/tile-16/tile-15 sequence at 2,048 live
tokens, with 15 timed samples per process and deterministic cache inputs, P4
GPU medians were `6.409/6.340/6.365/6.400 ms`; mixed-25% BF16 medians were
`6.154/6.091/6.067/6.049 ms`. Averaging the paired observations makes tile
`15` approximately `0.8%` slower for P4 and `0.4%` slower for mixed-25%, not
faster. CPU-reference maximum deltas remained below `8e-08`, and the focused
resident Metal suite passed `14/14` with a required nonzero GPU interval.

This refutes standalone decode as the source of the session-level tile-15
gain. The admitted M2 Max policy remains supported by the earlier prefill and
end-to-end replay evidence, but its mechanism is now scoped to the prefill or
system scheduling corridor rather than generalized to decode. Metal
`GPUStartTime/GPUEndTime` measure the completed command-buffer interval; they
are not occupancy, bandwidth, or instruction counters. Hardware-counter
sampling remains rejected for this slice because it would add a separate
capability and can perturb scheduling without answering the current boundary.

The same existing timestamp seam now measures the complete adaptive prefill
command directly: causal attention over the packed prefix, K/V packing of the
current chunk, and the completion finalizer. The probe uses the public
encode/finalize/publication boundary instead of widening the runtime API or
changing a kernel. It continues to report wall time and pack-only wall time,
but no longer presents their difference as an attention measurement.

In a guarded tile-15/tile-16/tile-16/tile-15 sequence, each fresh process used
the same deterministic inputs, a fixed 3,072-token snapshot, and a 64-token
chunk. Every one of the nine timed commands restored that immutable snapshot
into a fresh cache; a separate scratch cache warmed the prefill path before
sampling. P4 GPU medians were `19.752/24.978/25.235/20.004 ms`; BF16 medians
were `16.738/22.721/22.998/17.080 ms`; F32 medians were
`22.613/30.272/31.125/22.908 ms`. The paired means make tile `15` faster by
approximately `20.8%`, `26.0%`, and `25.9%`, respectively. Every tile-15
observation was below every matching tile-16 observation.

Each probe output records the effective Metal device, selected tile, fixed RNG
seed, and fixed-snapshot mode. The synthetic prefix deliberately repeats one
deterministic 64-token K/V chunk; this is an attribution payload, not a model
activation distribution.

The final release probe completed its default fixed-snapshot sweep at prefixes
512, 1,536, and 3,072. Focused tile-15 and tile-16 resident suites each passed
`14/14` with CPU-reference cosine `1.0` and maximum delta `3.72529e-08`; the
complete resource-isolated QBit/Metal suite passed `139` examples with zero
failures or errors and one optional model-backed pending example.

This closes the earlier attribution gap at command-buffer granularity: tile
selection changes only the attention pipeline, while the K/V packers and
finalizer in the same command are identical. The measurement therefore
supports prefill attention as the source of the bounded tile-15 gain and
rejects host-side packing subtraction as the explanation. It still does not
prove the occupancy mechanism, isolate individual encoder counters, or widen
the result beyond Apple M2 Max, this GQA6 shape, the tested prefix range, and
the existing end-to-end quality certificate.

### Real-KV fused prefill profile (2026-08-27)

The existing default-off `QWEN35_PREFILL_BOUNDARY_PROFILE=1` boundary now
reports the completed command buffer's `gpu_ms` through
`GPUStartTime/GPUEndTime`. The ordinary path still uses the original separate
commit and wait calls. The diagnostic interval covers the complete model
prefill command, including full attention, recurrent work, adaptive K/V
packing, the finalizer, and any output blit; it is not a per-kernel occupancy or
bandwidth counter. `gpu_ms` is nested inside `submit_wait_ms`; the two fields
must not be added. A no-op command reports `gpu_ms=0.0 gpu_timed=false` instead
of treating unavailable positive timestamps as an execution failure.

A Qwen3.8 session probe then rebuilt the same deterministic 829-token
model-produced snapshot recipe and replayed the same 2,391-token suffix in a guarded
tile-15/tile-16/tile-16/tile-15 order. Each process produced two independent
adaptive trajectories with 16 resident attention caches. Raw adaptive GPU
means were `23,227.136 ms` for tile 15 and `28,203.145 ms` for tile 16; medians
were `22,960.803/28,125.776 ms`, with a nominal `17.6%` reduction between the
means.

That raw result is not admitted as causal evidence. The tile-independent exact
replay controls drifted in the same direction and by a larger fraction:
`17,824.924 ms` beside tile 15 versus `22,152.505 ms` beside tile 16, a
`19.5%` difference. Per-process adaptive/exact ratios averaged `1.3090` for
tile 15 and `1.2703` for tile 16. The strict quiet-host gate had already waited
600 seconds and failed closed on an unrelated, seven-hour CPU-bound process;
the bounded fallback runs used low priority, a 24 GiB tree cap, 79--84% free
system memory, and no recorded thermal warning. They are controlled busy-host
observations, not a quiet-host certificate.

All four rows completed with zero swaps, roughly `7.67--7.71 GB` peak memory
footprint, consistent resident ownership, and `3.7647x` logical KV density.
They preserved top-1 `8/8`, exact-top-1 coverage `7/7`, ECS mean/minimum
`1.0/1.0`, EOS, meaning, and `Their sum is 95.`. Tile 15 repeated ranked and
set top-2 `11/14`; tile 16 repeated `12/14`, matching the already reported
reduction-order drift. The focused state/resident suite passed `23/23`, and
the complete resource-isolated QBit/Metal suite passed `139` examples with no
failures or errors and one optional model-backed pending example.

The profiler seam is therefore verified, but this real-KV run did not close
steady-state tile attribution. Its required falsifier was the same ABBA on a
quiet host with a 64-token replay chunk, giving a naturally growing resident
owner across many commands while preserving the snapshot recipe, positions,
chunk sequence, exact controls, quality coordinates, and resource guards. The
next section records that completed falsifier; this subsection remains the
historical busy-host result.

### Quiet-host 64-token real-KV replay (2026-08-30)

The release session probe replayed a 2,119-token real-model suffix from the
same deterministic 829-token Qwen3.8-27B Q4_K_M snapshot. Replay used one
7-token chunk followed by 33 64-token chunks, so each adaptive trajectory grew
and reused one resident owner across 34 commands. The guarded order was
tile-15/tile-16/tile-16/tile-15. Relevant source was commit `3362535e`; the
probe arguments were `--filler 48 --suffix-filler 128 --replay-chunk 64
--attribute-replay --gen 8`. Every process independently required less
than 50% CPU from any unrelated process and less than 100% aggregate unrelated
CPU before launch, waited for up to 600 seconds instead of modifying unrelated
workloads, required more than 35% free system memory, used a 24 GiB
process-tree cap, and had a 600-second timeout. The four rows began with
84--85% free memory, waited 16/69/25/99 seconds for a quiet window, and exited
cleanly after 70--75 seconds.

In ABBA order, exact chunked replay wall times were
`21,181.982/21,314.055/21,142.459/21,300.848 ms`; adaptive replay times were
`21,326.282/23,251.760/23,232.472/21,184.013 ms`. Tile 15 therefore averaged
`21,255.148 ms` adaptive replay beside a `21,241.415 ms` exact control, an
overhead of `13.733 ms` or `0.065%`. Tile 16 averaged `23,242.116 ms` beside a
`21,228.257 ms` exact control, an overhead of `2,013.860 ms` or `9.487%`.
The exact-control means differed by only `13.158 ms` (`0.062%`), while tile 15
reduced adaptive wall time by `8.549%` relative to tile 16.

Summed command-buffer GPU intervals independently preserved the result. Tile
15 averaged `20,964.582 ms` adaptive GPU time beside `21,030.207 ms` exact,
while tile 16 averaged `22,927.578 ms` beside `20,982.301 ms` exact. The
normalized adaptive GPU overhead was `-0.312%` for tile 15 and `9.271%` for
tile 16; tile 15 reduced adaptive GPU time by `8.562%`. The agreement between
wall and GPU intervals rejects host-side replay bookkeeping as the explanation
for this ABBA separation.

All four rows preserved top-1 `8/8`, ranked top-2 `11/14`, top-2 set overlap
`11/14`, exact-top-1 coverage in candidate top-2 `7/7`, token ECS
mean/minimum `1.0/1.0`, EOS, meaning, and the exact text
`Their sum is 95.`. Each row retained 16 resident attention layers,
`102,637,568` resident live-KV bytes, and `3.7647x` logical KV density.

This closes the steady-state real-KV attribution gap for the already admitted
Apple M2 Max Qwen3.8 GQA6 tile-15 policy. It does not identify occupancy or
cache behavior through hardware counters, and it does not widen the result to
other GPUs, models, prompts, chunk shapes, or larger contexts. Those remain
separate falsifiers rather than reasons to tune this already closed shape.

### Product-shaped 8K coding and prefill safety frontier (2026-08-30)

The quality probe now accepts a complete prompt from `--prompt-file`, reports
the effective prefill row and layer-group budgets, and emits exact/resident
prefill and decode times in its structured record. A separate scorer extracts
the generated Crystal source, runs the public and sealed neighboring specs in
fresh project copies, and treats that product result as the capability gate.
Top-1, top-2, ECS, density, ownership, and timing remain diagnostic coordinates
rather than substitutes for the external tests.

The sealed prompt renders to 7,718 Qwen3.8 tokens. Its exact and adaptive
continuations both produced the intended `Math.max` upper bound. The extracted
sources had the same SHA-256 and each passed all four external specs.

Three guarded adaptive attempts separated two failure mechanisms without an
OOM or reboot. One 7,718-row layer group per command let the first resident
prefill complete, but a repeated prefill eventually grew individual commands
to `17.1/11.5/12.4 s` and macOS rejected the third as `Impacting
Interactivity`. Capping resident rows at 2,048 and using two layer groups per
command reduced completed GPU intervals to roughly `1.6--5.6 s`; an A/B probe
that first used a distinct 7,718-row exact scratch geometry was safely stopped
by the 35% free-memory guard at 32%. Giving exact and resident sides the same
2,048-row geometry removed that extra peak and retained 54--56% free memory,
but sustained adaptive work still reached the interactivity guard after
completed commands of roughly `2.4--5.3 s`.

One-group rotation alone was not sufficient: a fourth guarded run reduced most
exact commands to roughly `0.95--1.9 s`, but continuous queue occupancy still
hit `Impacting Interactivity` after about 53 seconds. The smallest separating
move was therefore a 50 ms idle window after each completed and published
long-prefill command, before the next command buffer is created. It changes no
cache bytes or arithmetic. The safe default applies only when large-row command
rotation occurs. `QWEN35_PREFILL_APPEND_COOLDOWN_MS=0` removes only the idle
window for timing comparisons; `QWEN35_PREFILL_APPEND_MAX_GROUPS=0` restores
the historical single-command behavior.

The fresh 8K acceptance run completed in about 230 seconds under the strict
quiet-host preflight, 35% free-memory floor, 24 GiB process-tree limit, and
900-second timeout. A live sample retained 56% system memory and the host
returned to 87% afterward. The longest adaptive completed command was
`2.703 s` wall / `2.700 s` GPU. All 16 full-attention layers had adaptive owners,
no Float32 KV owner existed, and every cache published the expected 7,785 rows.
The 1,073,741,824-byte raw-capacity comparison used 285,212,672 resident bytes,
or `3.7647x` logical density.

The 68-token adaptive continuation matched exact top-1 at `68/68`, matched the
entire free-running text including EOS, covered exact top-1 in candidate top-2
at `67/67`, and retained token ECS mean/minimum `1.0/1.0`. Ranked top-2 and set
overlap were only `121/134`; this divergence remains a diagnostic warning even
though it did not change the generated program. Exact/adaptive prefill measured
`68.424/92.134 s`. Exact/adaptive decode measured `5.144/25.192 s`, about
`13.02/2.66 tok/s`; compact decode is therefore functional but still about
`4.90x` slower on this long-context row. The result promotes 8K host safety and
the measured coding task, not general coding quality or speed.

The 16K gate is no longer blocked by missing 8K evidence, but it remains a
separate guarded falsifier. Its prerequisites are the same sole-owner and
external-spec gates plus a capacity-sized memory forecast. Lowering the memory
guard, hiding the teacher-forced pass, treating exact-only output as adaptive
evidence, or trading away the 50 ms safety window before a measured replacement
is explicitly rejected.

### Split-K long-context decode frontier (2026-08-30)

The resident one-token decoder now uses a two-stage split-K reduction when the
K/V plans prove one uniform P4 or BF16 tier and the visible context contains at
least 256 tokens. Stage 1 partitions context into 64-token blocks while keeping
the existing GQA6 sharing: one decoded K/V tile serves six query heads. Each
block emits an online-softmax summary `{maximum, denominator, numerator}`;
stage 2 combines those summaries with a stable log-sum-exp reduction and applies
the existing gate. The exact current K/V row is still packed only after
attention. Multi-token prefill, mixed tiers, P5/F32 plans, and short contexts
retain the previous serial kernel. `QWEN35_ADAPTIVE_SPLITK=0` is the complete
runtime rollback.

The common 256-token threshold is deliberately conservative. In a fresh
boundary probe, P4 split-K GPU time was `0.608/0.616 ms` at 129/257 visible
tokens versus serial `1.042/1.925 ms`. BF16 was slower at 129 tokens
(`0.561` versus `0.370 ms`) but faster at 257 (`0.570` versus `0.695 ms`). A
tier-specific crossover rule would recover one short P4 interval but add policy
complexity; the common 256 threshold rejects that trade until a product result
requires it.

At 8K, fixed-snapshot P4 observations fell from `21.092--21.144 ms` to
`2.749--3.250 ms`, a non-overlapping `6.49--7.69x` reduction. One BF16 control
fell from `21.252 ms` to `2.014 ms` (`10.55x`), but lacks the replicated ABBA
strength of the P4 row. The kernel reads the same compact bytes; the gain comes
from replacing one long sequential context traversal with independent context
blocks that expose enough parallel work to overlap decode arithmetic and memory
latency. It is not an additional compression ratio.

The fresh product-shaped run kept the prior 7,718-token prompt, 68-token
continuation, coarse P4/BF16 layer map, and 8,192-token capacity. Exact/adaptive
decode measured `5.172/5.291 s` for 67 decode steps, or `12.95/12.66 tok/s`.
Adaptive decode is therefore `4.76x` faster than the previous `25.192 s` row
and only `2.30%` slower than its adjacent exact control. Exact and adaptive
generated identical 177-byte Crystal sources with SHA-256
`3494ecf7843f68ecc7261d75fc23e520a39a8fef50d1ce5d6d961f50b70c2d46`;
both passed four external specs. Top-1 remained `68/68`, exact-top-1 coverage
`67/67`, ECS mean/minimum `1.0/1.0`, EOS and full text matched, while ranked
top-2 and set overlap remained the pre-existing `121/134` warning. All 16
attention caches stayed resident, no Float32 KV owner appeared, and every cache
published 7,785 rows.

Split-K adds one shared, process-cached scratch set per distinct cache capacity
rather than one allocation per layer. The measured model has 24 query heads,
not 48. At 8,192 capacity and 64-token blocks the scratch set is
`3,170,304` bytes (about `3.02 MiB`) for partial numerators, maxima, and
denominators; at 16,384 capacity it doubles to `6,340,608` bytes (about
`6.05 MiB`). Scratch is sized from immutable cache capacity, not current
length, so growth across context-block boundaries reuses the same three buffers
instead of retaining one size-keyed pool entry per boundary. A regression
crosses the 256/257 boundary and requires three cache hits with no new misses.
This scratch is not included in the `285,212,672` resident KV payload metric;
counting it conservatively changes effective logical density from `3.7647x` to
about `3.7233x`. Cache payload bytes, checkpoints, restore, and ClickHouse
serialization are unchanged.

One guarded product attempt stopped safely at macOS `Impacting Interactivity`
during exact/adaptive prefill before the new decode phase. A later run with the
same 35% memory floor, 24 GiB process-tree cap, and a 100 ms benchmark cooldown
completed. That is evidence for keeping the existing host guards, not for
changing the admitted 50 ms default. Cross-device speed, all-BF16 product
sessions, live-16K runtime behavior, and harder coding quality remain open
falsifiers.

### Guarding the measured 16K-capacity prefill seam (2026-08-30)

A first guarded 16K-capacity run used the established 7,718-token coding prompt,
one layer group per command, and a 100 ms cooldown. Exact prefill completed, but
adaptive prefill was rejected by macOS as `Impacting Interactivity`. Reducing
the row chunk from 2,048 to 1,024 did not fix the failure: the profiled exact
path still stopped after about 64 seconds at the transition from the chunk
starting at row 5,120 to the next chunk. Completed heavy commands were roughly
`0.58--1.47 s`, so the evidence refuted a single oversized command as the sole
cause.

The trace exposed a narrower scheduling bug. In-chunk command rotation slept
after a completed command only when more layers remained. The final command of
each chunk was committed and waited, but the outer chunk loop immediately
started the next heavy command. The compositor window therefore did not cover
chunk boundaries even though the documented policy said it covered every
successive long-prefill command. The fix reuses the configured cooldown at that
boundary only when the completed chunk actually submitted shared-command GPU
work. It does not sleep after the final chunk or when the explicit group limit
is zero. The automatic policy also skips rows below 1,024; an explicit positive
group override intentionally opts smaller chunks into the same policy.

The same 1,024-row, one-group, 100 ms reproduction then completed all exact,
free-running adaptive, and teacher-forced adaptive phases under the 35% memory
floor and 24 GiB process-tree cap, exiting successfully in about 233 seconds.
Exact/resident/forced prefill measured `89.159/103.122/104.999 s`; exact/free
decode measured `5.383/5.387 s`. Exact and adaptive produced identical text and
EOS, matched top-1 `68/68`, covered exact top-1 `67/67`, and retained ECS
mean/minimum `1.0/1.0`. Ranked top-2 was `120/134`. All 16 attention layers had
adaptive owners, no Float32 owner appeared, cache publication was consistent,
and exact plus adaptive outputs each passed four external Crystal specs.

This is a 16K-*capacity* certificate, not a 16K-live-context certificate. The
probe allocated 16,384 rows but published only 7,785. Capacity accounting was
`2,147,483,648` raw F32 bytes versus `570,425,344` adaptive payload bytes
(`3.7647x`); including the corrected `6,340,608`-byte split-K scratch gives
about `3.7233x`. A prompt that actually renders to roughly 16,000 tokens, plus a
fresh split-K-on/serial-rollback pair, remains the next separately guarded
falsifier. This single run also does not promote a 16K speed claim.

Like the existing Qwen35 Metal scratch paths, this reuse assumes one in-flight
model wave per scratch namespace. A future concurrent multi-queue serving path
must provide a lane/session namespace before it may overlap adaptive decode
commands; this patch does not claim or introduce that wider concurrency model.

### Live-16K split-K acceptance on Apple M2 Max (2026-09-20)

The capacity-only boundary above has now been crossed with a prompt that renders
to 16,109 chat tokens. With a 16,384-token capacity and 68 generated tokens, the
published adaptive prefix reached 16,176 tokens in every admitted row. The fixed
map remained `p4;27=bf16,43=bf16,47=bf16,51=bf16`: all 16 full-attention layers
had adaptive owners, no Float32 KV owner appeared, and cache publication was
consistent according to the probe checks. The 2,147,483,648-byte raw F32
capacity used 570,425,344 adaptive payload bytes (`3.7647x`), excluding the
already-accounted 6,340,608-byte shared split-K scratch.

A fresh-process split-K-on/serial-off/serial-off/split-K-on ABBA explicitly set
`QWEN35_ADAPTIVE_SPLITK=1/0/0/1`; the JSON does not echo this environment
selector, so the launch manifest and mode-named logs preserve that provenance.
The four rows kept the model, prompt SHA-256
`379fbc737a615917a4df5cf083098127af9c0163348b8823688994067990edb3`, map,
1,024-row prefill chunks, one append group, 100 ms cooldown, and 68-token
continuation fixed. Every process began with 71--72% system memory free, used
the current 30% runtime floor and 24 GiB process-tree cap, exited zero, and
produced the same 68 tokens. Split-K decode measured `5.199/5.227 s`; serial
decode measured `47.789/47.811 s`. Median throughput was therefore
`12.85` versus `1.40 tok/s` over 67 timed post-prefill decode calls, a `9.17x`
speedup or 89.09% reduction in adaptive decode time. Median resident prefill
differed by only -0.72%; prefill plus free decode improved by 14.53%. The
split-K rows were also 16.09% faster than their adjacent exact-decode controls,
but that comparison is secondary to the paired rollback.

Quality and ownership gates were invariant across all four rows: the
free-running common prefix and top-1 were `68/68`. On the teacher trajectory,
exact top-1 remained covered by adaptive top-2 at `67/67`, ranked top-2 and set
overlap were `118/134`, output-weight embedding cosine mean/minimum were
`1.0/1.0`, and no cosine mismatch appeared. These are bounded trajectory
diagnostics, not a general semantic-quality score. The generated source was
truncated before EOS at the 68-token limit, so this promotes live-16K runtime,
cache ownership, bounded top-1/ECS trajectory agreement, and split-K speed on
this Apple M2 Max/model/map combination. It does not promote a complete
coding-task pass, general model quality, concurrent serving, another device, or
another cache map. A longer continuation plus an external Crystal spec remains
the product-quality falsifier.

Verdict: ROBUST for the declared single-device live-16K split-K boundary. Keep
split-K enabled for admitted uniform P4/BF16 one-token decode and preserve
`QWEN35_ADAPTIVE_SPLITK=0` as the serial rollback. Evidence:
`/private/tmp/qwen_qbit_16k_live_split_on.log`,
`/private/tmp/qwen_qbit_16k_live_split_off.log`,
`/private/tmp/qwen_qbit_16k_live_split_off_b2.log`, and
`/private/tmp/qwen_qbit_16k_live_split_on_a2.log`. The release probe was built
from source revision `730ce6001d2db807ec8b51dc606b668dac1424ed` and had
SHA-256 `8dc217fdc840c4715c576f24e4a2a452cbbb4fb3fd7f02556ef8314617b4ab32`.
Refresh on source or probe binary, model/prompt/map, Metal compiler, device/OS,
safety policy, or evidence loss.

### CogniGraph bounded exact-prefill enqueue frontier (2026-08-30)

CogniGraph now has a default-off, depth-one-or-two submission corridor for the
exact Qwen prefill path. `QWEN35_COGNIGRAPH_PREFILL_MAX_INFLIGHT=1|2` reserves a
lease before any command encoding or host mutation, creates every command from
one explicit Metal queue, retains a private scratch arena until terminal GPU
completion, and reuses a slot only after the FIFO-oldest command has completed.
Caller-side encoding aborts and command construction, submit, wait, or cancel
failures drain all known work and poison the local queue. Zero or an unset
variable preserves the previous path.

The drain is a resource and ordering barrier, not a transactional rollback of
model state already written by an earlier completed command. As with other
Metal inference failures after state mutation, the caller must discard that
sequence state rather than retry from it.

This first slice deliberately rejects adaptive QBit KV, checkpoints, boundary
profiling, CPU-only execution, and disabled/unavailable resident Metal command
routes. Adaptive caches still expose one publication ledger, so overlapping
them before that ledger becomes per-flight would make completion ownership
ambiguous. The exact slice is the lifecycle certificate for that later change,
not evidence that adaptive enqueue is already safe.

The nine-example focused queue suite covers FIFO reservation, depth-two reuse,
foreign lease rejection, cancellation, construction/submit/wait/discard failure
draining, retained-resource release, and native same-queue Metal ordering.
CPU-only generation also builds.
A Qwen3.5-9B integration probe preserves prefill top-1/logit and the next append
top-1/logit. On the local Qwen3.8-27B Q4_K_M model, a 1,024-token run submitted
14 command buffers, reached a measured maximum pending depth of two, completed
all 14, and matched baseline top-1 plus top-1 logit within `1e-4` in two paired
observations.

Against the historical no-idle timing frame (`cooldown=0`), one noisy pair was
effectively flat: the queued path was `6,050.61 ms` versus `6,015.53 ms`
(`0.58%` slower). Against the admitted 50 ms compositor-window policy, two
interleaved pairs measured `6,032.32 ms` queued versus `6,861.24 ms` baseline,
an `828.92 ms` or `12.08%` reduction, with parity in both pairs. This is a
bounded recovery of host cooldown overhead at 1,024 tokens, not a general GPU
kernel speedup or a long-context safety promotion. Live 8K/16K watchdog behavior
and repeated quiet-host throughput remain open.

The LTP/WBA trigger is an exact resident prefill group that would otherwise
commit, wait, cool down, and only then encode its successor. The transport
corridor is one Metal queue plus a FIFO lease and its private scratch arena. A
legal move is `reserve -> write/encode -> enqueue/commit -> await oldest ->
release/publish`; its boundary invariants are exact output/state parity, no
mutable scratch alias across live leases, same-queue command order, bounded
depth, and terminal draining before resource reuse. The recomputed potential is
`(semantic failure, resource alias, undrained work, interactivity failure,
sync/cooldown wall)`, in that order. The legacy zero-depth path is the dual
frame. The move is not promotable to adaptive QBit until a per-flight cache
publication certificate preserves those same higher-priority coordinates.

### CogniGraph adaptive-QBit publication frontier (2026-08-30)

This slice is admitted behind the existing default-off depth-one-or-two
CogniGraph prefill corridor. Every adaptive cache now carries an ordered
per-command publication ledger, so reservation, completion, failure cleanup,
and visible-prefix publication no longer share one mutable pending slot.

Admitted surface for the implementation is deliberately narrow:

- one serialized model invocation, one explicit Metal command queue, and FIFO
  completion; a pending suffix rejects a command from any other queue;
- adaptive full-attention prefill appends whose token ranges are disjoint and
  reserved before any encoder writes them;
- one per-command ticket per selected cache, containing the exact command,
  status buffer, start row, and token count;
- duplicate-free group validation across all selected layers before any ticket
  for that command advances a visible cache length;
- depth zero as the unchanged synchronous rollback frame.

The visible prefix remains `cache_len`; reservations extend a separate ordered
tail. A second command may reserve only at that tail. Completion may publish
only the FIFO-oldest ticket and only after both the Metal command and its true
tail marker succeed. Snapshot, release, synchronous append, and decode remain
illegal while any ticket is unpublished. A failed or indeterminate command
poisons the local submission corridor. After the queue has cancelled or
drained every possible writer, one checked discard operation releases the
complete unpublished suffix in reverse order without advancing the visible
prefix. The enclosing failed multi-flight model sequence remains non-retryable
and must be discarded by its caller; this does not forbid a fresh isolated
single-ticket cache append after its failed reservation has been removed.

Rejected or not certified in this slice: default enablement, out-of-order
publication, concurrent sessions sharing a cache, adaptive decode overlap,
checkpointing, boundary profiling, recovery from a failed submitted command,
and a general scheduler abstraction. A second Metal queue is rejected while a
suffix is pending, but callers must still serialize access to one model state;
the other cases require separate certificates.

The lifecycle falsifier shows that two commands can reserve adjacent ranges on
the same cache while `cache_len` remains unchanged, that the second cannot
publish before the first, and that foreign-command finalization/cancellation is
rejected. A second test uses two caches and two completed commands, failing
each cache position in turn: neither cache becomes visible, the successor
remains blocked, and a checked suffix discard restores snapshot/release
eligibility without leaking Metal buffers. The full adaptive resident suite
also rejects a duplicate cache group before publication and a second Metal
queue before it can encode a successor reservation. It passes 17/17, and the
policy keeps checkpoint and boundary-profiling routes rejected.

Performance is vector-valued: full wall time, GPU time, host encode/wait time,
peak/resident bytes, watchdog survival, cache ownership/publication, generated
meaning, top-1, ranked top-2, and ECS remain separate coordinates. Lower
command-boundary wall time cannot compensate for a regression in any earlier
semantic, ownership, or safety coordinate. A later speed promotion therefore
requires a guarded paired run and retains the synchronous adaptive route as the
dual frame.

The measured Qwen3.8-27B Q4_K_M pp1024 row used the coarse
`p4;27=bf16,43=bf16,47=bf16,51=bf16` map, a 35% free-memory floor, and a 24 GiB
process-tree cap. Four interleaved pairs preserved final top-1 and its logit
within `1e-4`; depth two was faster in all four pairs, with mean wall time
falling from `7,099.60 ms` to `6,633.17 ms` (`6.57%`). A separate graph-on/off
quality pair produced byte-identical exact and resident token sequences,
teacher top-2 rows, top-1 counts, and ECS. The resident result itself was
`28/32` teacher-forced top-1, `57/62` top-2 set overlap, and ECS `0.882048` for
this prompt; the semantically weaker compressed sentence was identical with
the graph off, so it is a coarse-map quality limitation rather than a scheduler
regression.

This is not adaptive-attention overlap yet. The current adaptive full-attention
route still flushes its command and reads the hidden output before continuing;
the observed `max_pending=2` comes from adjacent recurrent/full-attention
handoff groups. The slice therefore makes bounded CogniGraph usable with an
adaptive session and removes measured scheduler wall time, but it does not
claim to eliminate the remaining adaptive host boundary. That boundary,
tier-specialized pack/decode, and decode-chain command fragmentation are the
next independent performance falsifiers.

### Adaptive final-head resident handoff frontier (2026-08-30)

The first adaptive boundary reduction is intentionally narrower than a general
cross-layer handoff. When `QWEN35_PREFILL_TOP1_ADAPTIVE_RESIDENT=1`, the final
adaptive full-attention layer writes its completed hidden rows into an
invocation-owned Metal buffer. After the existing command completion, status
validation, and KV publication boundary, output RMSNorm and the fused Q6/Q8
top-1 projection consume the last row in place and read back only the token id
and score. The ordinary hidden readback route remains the default and rollback.

This keeps the existing cache semantics unchanged: it neither overlaps two
adaptive commands nor defers publication, and the buffer cannot outlive the
enclosing prefill call. Static preflight certifies the output hidden/norm
dimensions before state mutation. If the selected-row head is nevertheless
rejected after publication, the route materializes only the completed final
hidden row and uses the ordinary exact head; it never reruns the decoder body.
A focused Metal spec compares a selected second row against an independent CPU
RMSNorm plus quantized-head oracle and rejects negative or out-of-range offsets.
The complete forward suite passes `24/24`, and CPU-only generation builds with
the route compiled out.

On the guarded Qwen3.8-27B Q4_K_M pp1024 row, an unmeasured warmup followed by
`off/on/on/off` preserved the final top-1 id, score, and all 16 published cache
lengths. The profile reduced the final adaptive hidden readback from 20 MiB to
zero. This removes the host transfer, not the invocation-owned 20 MiB resident
buffer itself. Warm means were `7,725.79 ms` off and `7,610.65 ms` on (`1.49%`),
but the
per-observation spread is large enough that this is not a speed certificate.
A separate eight-token quality pair preserved the exact text, top-1 `8/8`,
ranked top-2 `14/14`, exact-top-1 coverage `7/7`, ECS mean/minimum `1.0/1.0`,
16 resident owners, no Float32 KV owner, and consistent cache publication.

The value claim is therefore limited to removing a measured transfer while
preserving the tested boundary. Timing, transfer bytes, token quality, cache
ownership, and publication remain separate coordinates. This is ordinary
resident dataflow, not an LTP/WBA promotion. Default enablement needs a stable
end-to-end wall-time win; a wider adaptive-to-recurrent handoff additionally
needs an explicit lifetime certificate for every in-flight output buffer.

### Adaptive final-head command append frontier (2026-08-30)

A source re-read narrowed the next boundary. Intermediate adaptive
full-attention layers that are followed by recurrent runs already use the
existing fused full-to-recurrent Metal helper and keep their hidden handoff in
the caller-owned command. The remaining final boundary was smaller: the last
adaptive layer completed and published its cache, then output RMSNorm and the
resident top-1 head ran in a second command.

`QWEN35_PREFILL_TOP1_ADAPTIVE_APPEND=1`, layered on the existing default-off
`QWEN35_PREFILL_TOP1_ADAPTIVE_RESIDENT=1` gate, appends that RMSNorm and fused
top-1 projection to the last adaptive command. The id/value outputs are
invocation-owned buffers. The adaptive cache finalizer is still encoded after
the head, so successful command completion and every cache tail marker remain
the publication certificate. Scratch belongs to the active CogniGraph lease
arena when one exists. Unset or `0` preserves the separate-head command as the
exact rollback.

The focused Metal contract proves that the encoder neither commits its caller
command nor changes the selected-row result relative to an independent CPU
RMSNorm plus quantized-head oracle; it also rejects an already completed
command. The full forward suite passes `25/25`, the adaptive resident lifecycle
suite passes `17/17`, and CPU-only generation builds. A separate final-layer
policy regression proves that earlier standalone adaptive layers do not consume
the append descriptor; only the last model layer may encode the head. A real
Qwen3.8 depth-one CogniGraph smoke reached both append route markers with zero
profiled Metal syncs. The old resident-head profile reported one sync; the
appended profile reported zero.

The guarded Qwen3.8-27B Q4_K_M pp1024 A/B used the coarse
`p4;27=bf16,43=bf16,47=bf16,51=bf16` map, a 12% memory-pressure floor, and a
24 GiB process-tree cap. Four interleaved pairs preserved top-1 and its logit
within `1e-4`. Means were `7,533.81 ms` for the separate command and
`7,533.07 ms` for the append, only `0.73 ms`; the append won one of four pairs.
This proves no wall-time speedup. The structural command reduction is retained
only as a default-off composition seam for future GPU-resident decode work.

An eight-token quality run preserved exact text, top-1 `8/8`, ranked and set
top-2 `14/14`, exact-top-1 coverage `7/7`, and ECS mean/minimum `1.0/1.0`.
All 16 attention layers remained adaptive owners, no Float32 KV owner appeared,
and cache lengths were consistent. These metrics certify the tested boundary;
they do not turn the noisy timing row into a speed claim.

This is ordinary same-command fusion, not LTP/WBA. The rollback is the existing
separate resident-head command. Default promotion remains rejected until a
downstream resident consumer removes a material synchronization or a repeated
paired measurement establishes end-to-end value.

### Adaptive pack specialization falsifier (2026-08-30)

The generic adaptive pack kernel computes five bit planes before selecting the
row tier, although P4 consumes only four planes and BF16/F32 use exact sidecar
storage. A temporary candidate split uniform P4 and BF16 into dedicated Metal
entry points and host pipelines. Focused two-append tests established exact
payload parity for those uniform plans before timing.

The bounded A/B/B/A result rejected the extra pipelines. With 64 source tokens,
the complete K pack, V pack, and status finalizer GPU interval was
`0.026/0.030 ms` for generic P4 and `0.028/0.024 ms` for the candidate. Generic
BF16 measured `0.027/0.031 ms`; the candidate measured `0.028/0.024 ms`. The
128-token rows also overlapped: generic P4 was `0.027/0.028 ms` versus
`0.026/0.025 ms`, and generic BF16 was `0.029/0.031 ms` versus
`0.029/0.026 ms`. Wall-clock medians crossed in both directions. These are
microsecond-scale differences inside a command that is already a small fraction
of the measured long-prefix attention cost.

The candidate was therefore reverted. In addition to its unverified value, a
uniform entry point would bypass the generic kernel's per-row tier/metadata
validation and would need a wider corruption and mixed-tier certificate before
promotion. The canonical kernel, pipeline set, payload format, and default
runtime behavior remain unchanged.

The useful diagnostic seam is retained: `append_from_metal` can optionally
return the completed pack command's GPU interval, and the model-free probe now
covers uniform BF16 as well as P4 and mixed tiers. Omitting the pointer preserves
the existing synchronous API. The complete adaptive resident suite passes
`17/17`, its measured append interval is positive on Apple M2 Max, the final
generic probe runs all three tier modes, and CPU-only generation builds.

This is ordinary kernel profiling, not LTP/WBA. The value coordinate is
end-to-end latency, not removal of one arithmetic loop in isolation. The next
QBit acceleration target remains attention/dequantization dataflow and tile
occupancy, where the measured GPU time is material.

### SIMD P4 dequantization falsifier (2026-08-30)

The next bounded candidate attacked repeated uniform-P4 loads inside adaptive
prefill and split-K stage one. In the canonical loader every value reads its
row `mean` and `sigma`, and every eight neighboring values address the same
plane byte. A temporary compile-time Metal variant made each 32-lane SIMD group
load the row header once and each plane byte once per eight lanes, then shared
them with `simd_shuffle`. It added no barrier, changed no cache byte or buffer
binding, and left BF16 and mixed-tier arithmetic unchanged.

The transformation was numerically legal but operationally bad. A focused
Metal contract matched the independent CPU reference, canonical serial output,
and split-K output within the established bounds, while the appended packed
payload remained byte-identical. The model-free product-shaped probe then ran
fresh-process A/B/B/A at prefix 3,072, chunk 64, tile 15, and nine repetitions.
Canonical P4 completed the prefill, K/V pack, and finalizer command in
`18.991/18.988 ms`; the SIMD variant required `30.712/30.734 ms`, a repeatable
regression of about `61.8%`. The unchanged BF16 control moved with host noise,
but the P4 regression was stable in both candidate positions and disappeared
in both rollback positions.

The exact microarchitectural cause is not established. Cross-lane shuffle
latency, lower instruction-level parallelism, and effective caching of the
canonical repeated addresses are plausible explanations, not verified facts.
The decisive result is the completed GPU interval: reducing the logical load
count did not reduce product-shaped work. The experimental policy, source
variants, pipelines, probe label, and parity extension were all reverted.

This is ordinary kernel falsification, not LTP/WBA. The canonical loader is the
dual frame and remains unchanged. A future attention candidate must reduce
actual instructions or memory transactions without cross-lane exchange and
must first beat this same product-shaped GPU interval before any model-backed
quality or default-promotion work.

### Register-local t4 adaptive dequantization (2026-08-31)

The successor candidate applies the useful part of llama.cpp's four-value
dequantization pattern without copying its cache format or using cross-lane
exchange. Four aligned adjacent values stay in one thread's registers.
Uniform P4 reads the row header once and one byte from each plane; uniform BF16
loads four adjacent values. The packed representation, tier metadata, sidecar,
threadgroup tile, softmax, pack/finalize path, publication, and host buffer ABI
are unchanged.

The default policy is deliberately route- and device-scoped. On Apple M2 Max,
unset `QWEN35_ADAPTIVE_DEQUANT_T4` selects t4 for uniform multi-token P4/BF16
prefill and for uniform BF16 split-K stage one. Uniform P4 split-K remains on
the scalar source because its isolated timings crossed. One-token serial
attention, non-P4/BF16 and mixed-tier rows, and other Metal devices remain
scalar by default. `QWEN35_ADAPTIVE_DEQUANT_T4=1` forces t4 for eligible routes
as an experimental override; `=0` is the exact scalar rollback.

A common tile-fill helper covers both prefill and split-K stage one, so the
candidate does not duplicate four K/V traversal implementations. The exact
override policy fails closed. Focused Metal coverage compares canonical serial,
t4 serial, and t4 split-K against the independent CPU attention reference,
requires the existing cosine and maximum-error bounds, and requires byte-for-
byte identical packed K/V after append. The fallback tile-16 variant passes the
same contract. The complete adaptive resident suite passes `17/17` with the
gate off and `17/17` with it on.

On Apple M2 Max, the post-refactor fresh-process A/B/B/A row used uniform P4
and BF16, prefix 3,072, chunk 64, tile 15, and ten repetitions. The complete
prefill, K/V pack, and finalizer GPU interval was `18.974/19.015 ms` for scalar
P4 versus `14.027/14.028 ms` for t4, a `26.2%` reduction by pair means. BF16
measured `22.982/22.326 ms` scalar versus `17.083/17.408 ms` t4, a `23.9%`
reduction. An isolated 256-prefix A/B/B/A also favored t4, but earlier
multi-prefix short rows had much larger variance, so no general short-context
speed claim follows from that point.

A paired guarded Qwen3.8-27B Q4_K_M resident replay used the coarse
`p4;27=bf16,43=bf16,47=bf16,51=bf16` map and the frozen Crystal pipeline task.
The scalar and t4 reports were identical, including response text, EOS,
top-1 `151/155`, ranked top-2 `274/308`, top-2 set overlap `282/308`, exact
top-1 coverage `154/154`, ECS `0.976445`, all 16 resident owners, no Float32 KV
owner, consistent publication, and `3.7647x` logical density.

Two paired real-model prompt-processing checks then measured total wall time,
not only the local Metal interval. At pp512, scalar averaged `4134.82 ms` and
t4 `4065.41 ms` (`1.68%` lower); at pp1024, scalar averaged `7534.44 ms` and
t4 `7340.85 ms` (`2.57%` lower). T4 won all `8/8` paired repetitions, with the
same top-1 token and final logit within `1e-4` in every pair. These are bounded
single-device prompt-processing results, not a claim that the whole engine is
25% faster.

The one-token serial route was screened separately but not promoted. Fresh-
process scalar/T4 rows at prefixes 64, 128, and 192 reduced the complete
adaptive attention command on both uniform P4 and BF16. Five guarded paired
Qwen3.8-27B Q4_K_M replays requested 64 generated tokens and reached EOS after
29. T4 won four pairs and the paired median was `0.48%` faster, but one pair
regressed `3.93%` and the five-run arithmetic mean was `0.18%` slower. All ten
runs produced the same token IDs and text; each reported top-1 `29/29`,
ranked/set top-2 `55/56`, exact-top1 coverage `28/28`, ECS mean/minimum
`1.0/1.0`, all 16 adaptive owners, no Float32 owner, consistent publication,
and `3.7647x` logical density on this one prompt/model/device/29-token trace.
The strong local command win therefore does not
justify changing the serial default; explicit `=1` remains available for
experiments.

An isolated uniform BF16 split-K A/B/B/A at prefix 8,192 and chunk one measured
scalar `2.000/1.825 ms` versus t4 `1.860/1.468 ms`, about `13.0%` lower by pair
means. Uniform P4 split-K crossed (`2.764/2.067 ms` scalar versus
`1.546/3.486 ms` t4), so it is explicitly excluded from automatic selection.

This evidence admits a narrow Apple M2 Max default for the measured uniform
prefill and BF16 split-K corridors. Cross-device occupancy, mixed-tier
execution, broader prompt distributions, and production-scale speed remain
open. This is ordinary register-local kernel optimization, not LTP/WBA.

### BF16 fused split-K stage2 weight traversal (2026-08-31)

The long-context split-K reducer used to evaluate the same softmax rescaling
weight once for the normalization sum and again for each of the eight output
dimensions owned by a SIMD lane. The fused traversal keeps the existing
global-maximum pass, computes one weight per block and lane, and updates the
normalization plus all eight dimension accumulators together. Global-max
normalization, ascending block order for every sum, post-normalization gating,
scratch layout, host bindings, cache bytes, and publication remain unchanged.

A same-binary legacy/auto/auto/legacy screen on Apple M2 Max used prefix
8,192, chunk one, tile 15, the default automatic BF16 T4 stage1, and ten
repetitions. The complete adaptive attention, K/V pack, and finalizer GPU
interval measured `2.481/2.490 ms` with exact legacy rollback versus
`1.797/1.807 ms` with automatic fused stage2, about `27.5%` lower by pair
means. A scalar-stage1 control remained positive at about `22.0%` lower. P4
did not reproduce the earlier apparent gain: its pair mean was about `6.6%`
slower fused in the scoped rerun, so automatic P4 admission was rejected.
These numbers include both split-K stages and cache append work; they are not
isolated stage2 timings or a claim about whole-engine decode speed.

The adaptive resident Metal suite now exercises an 8,191-token packed prefix
rather than only the 255-token split-K threshold. It covers scalar and T4
stage1, fused and legacy stage2, matches the independent CPU attention
reference and serial Metal path within the established cosine and maximum-error
bounds, preserves byte-identical packed K/V, and reuses capacity-sized scratch;
the policy plus resident suites pass `24/24`. A guarded Qwen3.8-27B Q4_K_M
check with a 360-token prompt crossed the live split-K threshold using the
mixed default route: legacy P4 plus fused BF16. It preserved top-1 `2/2`,
ranked/set top-2 `2/2`, ECS mean/minimum `1.0/1.0`, all 16 adaptive owners, no
Float32 KV owner, consistent cache publication, and `3.7647x` logical density.

Automatic fused stage2 is admitted only for uniform BF16 on the exact
`Apple M2 Max` device name. P4 and other devices retain the legacy reducer.
`QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED=0` is the exact runtime rollback, while
`1` is an explicit experimental force switch; malformed configured values fail
closed. Register pressure and the relative benefit may change with GPU
generation, compiler, head dimension, block count, or tile. Cross-device
timing and a direct whole-model decode A/B remain open. This is ordinary loop
fusion, not LTP/WBA.

### Prefix-only adaptive pack quantization (2026-08-31)

The resident format stores only the four most-significant QBit code planes for
P4, BF16, and F32 rows, and five planes for P5 rows. The canonical pack kernel
previously resolved all eight code bits with a seven-step search before
discarding the unused suffix. The prefix-only variant instead searches the
exact P4 or P5 group boundary in three or four steps and constructs only the
stored planes. It keeps the generic per-row tier and metadata validation,
sidecar layout, status/finalizer, cache bytes, and publication boundary.

This is a successor to the rejected dedicated-pipeline experiment above, not a
reversal of its evidence. It uses the same generic entry point and does not
bypass mixed-tier validation. A focused mixed P4/P5/BF16/F32 Metal contract
forces legacy and prefix-only compilation and requires byte-for-byte identical
K/V snapshots. The policy suite passes `8/8`, the full adaptive resident suite
passes `18/18`, and CPU-only generation builds.

Fresh-process A/B/B/A screens on Apple M2 Max measured the isolated 64-token
pack command about `41.8%` lower for P4 and `52.5%` lower for BF16. At prefix
8,192 and chunk one, pack GPU time was about `44.6%` lower for P4 and `50.7%`
lower for BF16. The complete BF16 adaptive attention, pack, and finalizer
interval was only about `2.1%` lower by pair means. The corresponding P4
complete-command row was noisy and did not establish a win, so P4 is excluded
from automatic selection despite its faster isolated pack step.

Unset `QWEN35_ADAPTIVE_PACK_PREFIX_QUANT` therefore selects prefix-only packing
only for uniform BF16 on exact `Apple M2 Max`. P4, P5, F32, nonuniform per-row
plans, and other devices remain on the canonical seven-step source by default.
Admission is evaluated independently for K and V, so a uniform-BF16 plan may
use the prefix source even when its peer plan is nonuniform. `0` is the exact
legacy rollback and `1` is an explicit experimental force switch;
malformed configured values fail closed. A guarded 360-token Qwen3.8-27B
Q4_K_M check preserved top-1 `2/2`, ranked/set top-2 `2/2`, ECS mean/minimum
`1.0/1.0`, all 16 adaptive owners, no Float32 owner, consistent publication,
and `3.7647x` logical density.

The admitted value is a bounded reduction in pack and one 8K BF16 adaptive
command interval. Whole-model decode, end-to-end generation, cross-device
occupancy, and broader tier distributions remain open. This is ordinary
quantizer specialization, not LTP/WBA.

### Rejected explicit vec4 Q4_K dequantization (2026-08-31)

A temporary compile-time variant replaced four adjacent scalar Q4_K byte loads
with `uchar4` loads and retained four dequantized values in `float4`. It was
default-off, restricted to exact Apple M2 Max and four-byte-aligned weight
offsets, and covered both the plain B64 gate route and the fused H16
up-plus-SwiGLU route. It changed neither the GGUF layout nor any public runtime
contract.

A real Qwen3.5-9B Q4_K_M parity contract compared every Float32 gate bit and
every Float16 fused up-plus-SwiGLU bit with the scalar source and passed. The
candidate was nevertheless slower in the product-shaped pre-gate: seven
interleaved pp64 pairs measured `515.65 ms` scalar versus `520.28 ms` vec4 on
average, about a `0.90%` regression, with scalar winning five pairs. Final
top-1 and its logit within `1e-4` matched in all seven pairs. Profiling assigned
`27.07%` of logical matmul-weight traffic to recurrent Q4_H16 FFN up/gate and
`34.96%` to total Q4 FFN up/gate, so this was not a cold-route result.

The explicit vector implementation, policy, and tests were removed. No 27B
pp2048 run was admitted after the smaller product-shaped pre-gate failed. A
plausible explanation is that Metal already coalesces the scalar loads and the
explicit vector form increases register pressure, but no counter evidence
promotes that explanation to a finding. The next Q4 FFN experiment must first
measure the actual plain gate plus fused up/SwiGLU command interval and then
demonstrate local work reduction as well as end-to-end parity. This is an
ordinary rejected kernel specialization, not LTP/WBA.

### Exact-route fused Q4 FFN timing boundary (2026-08-31)

The attribution binary can now measure the current default B64 FFN corridor
directly: one shared F32-to-F16 input conversion, a Float32 gate projection,
and the fused H16 up-plus-SwiGLU projection. It reports host submit-and-wait and
the Metal completed-command GPU interval separately. The diagnostic fails
closed unless the dimensions, raw Q4_K sizes, dispatch counts, and current
gate/up/down B64 route policy match. It also rejects ADDNORM-H16 input,
tensor-gate, exact-rowpack, scratch-off, and other route-changing
configurations. Its validation mode compares every Float32 gate bit and every
Float16 activation
bit against the current unfused GPU pair plus standalone SwiGLU route; the
focused real-model Metal contract passes.

On Apple M2 Max, two guarded Qwen3.8-27B Q4_K_M screens measured the exact
`5120 -> 17408`, batch-2048 shape after two warmups and across seven samples.
The first submit-and-wait/GPU p50 was `95.783/94.185 ms`; the final route-aware
binary measured `86.699/85.203 ms` for the same command. The roughly `9.5%`
absolute GPU-p50 drift occurred without a kernel candidate, so future A/B
decisions must be interleaved in one process. There are 64 same-shape pairs,
but even the final `5452.989 ms` serial estimate is multiplication, not a
traced whole-model interval. It must not be read as total prefill time or as an
engine speedup.

This is a kernel-candidate pre-gate, not an independent correctness oracle or
hardware-counter profiler. The comparison reference shares the current Metal
Q4 implementation, and the timestamps do not expose occupancy, register
pressure, cache traffic, or individual dispatch intervals. A candidate must
therefore keep a separate current-source pipeline, preserve full-buffer parity,
win a stable interleaved local GPU-timing screen, and then survive a
product-shaped wall-time and output-parity run. This instrumentation and the
next scheduling probes are ordinary Metal optimization, not LTP/WBA.

### Rejected final B64 gate barrier elision (2026-08-31)

A temporary separate-source candidate removed only the final input-loop
`threadgroup_barrier` from the plain B64 Q4_K gate kernel. Initial,
inter-iteration, SIMD-group, and output-staging barriers stayed intact, and
normal route selection was unchanged. Full-buffer bit parity passed on a real
Qwen3.5-9B Q4_K_M weight for a complete batch-64 tile and a batch-120 tail.

The exact-route probe then ran current and candidate pipelines in alternating
AB/BA order in one process on Apple M2 Max. For the `4096 -> 12288`,
batch-2048 corridor, two warmups and ten measured pairs produced current and
candidate GPU p50s of `45.649 ms` and `45.870 ms`. From the reported p50s the
candidate was about `0.48%` slower and won only `4/10` pairs. Same-process
ordering limits slow drift, but this is still a
noise-sized ten-pair result. The raw pairs and temporary candidate artifact
were not retained, so the row is not a reproducible performance certificate.
Its wrong-sign p50 and minority paired wins are sufficient to reject promotion
on this one shape; no 27B product-shaped run was admitted.

The candidate kernel, diagnostic route, tests, and CLI switches were removed.
The current barrier remains. Exact parity was a safety coordinate rather than
evidence of value; interleaved GPU time was the local value coordinate and it
rejected the change. Future synchronization work must remove a larger,
independently justified unit of work instead of retrying this single-barrier
hypothesis. This was ordinary kernel scheduling, not LTP/WBA.

### Rejected compile-time P4 prefill tier folding (2026-08-31)

A temporary separate-source adaptive-attention pipeline replaced the runtime
uniform-tier selector with a compile-time P4 constant. It changed no cache
layout, arithmetic, publication boundary, or mixed-tier fallback. A focused
Metal contract required bit-identical P4/BF16 outputs and serialized K/V bytes
across prefill and split-K fallback cases and passed; the policy contract also
passed.

The performance evidence did not survive a stronger ordering falsifier. The
route-shaped screen used 24 query heads, 4 KV heads, head dimension 256, a
3,072-token packed P4 prefix, and a 64-token chunk on Apple M2 Max. Simple
same-process AB/BA runs initially appeared `4.6-7.0%` faster, but paired wins
degraded from `10/10` to `16/20` and then `12/20`. A symmetric ten-block
ABBA/BAAB run, averaging two commands per variant inside each block, measured
generic/static GPU medians `16.771/19.465 ms`: the static source was about
`16.06%` slower and won only `3/10` blocks. All compared outputs stayed
bit-identical.

The source patch, route policy, tests, and temporary probe were removed; the
generic uniform loader remains. The reversal does not establish a universal
compiler law, but it rejects automatic promotion on the measured target and
shows that a positive median without stable paired wins is not enough. The
next kernel candidate must remove independently identified work rather than
depending on branch folding. This was ordinary compile-time specialization,
not LTP/WBA.

### Rejected automatic P4 prefix-only pack widening (2026-08-31)

The existing prefix-only pack source was forced on for uniform P4 at an
8,192-token prefix and one appended token. It reliably reduced the isolated
pack GPU median from `0.068--0.072 ms` to `0.036 ms`, confirming that its
three-step P4 search removes real local work relative to the generic seven-step
quantizer.

That local result did not survive the full-operation value boundary. A first
ten-repeat fresh-process A/B/B/A screen had complete attention, pack, and
finalizer GPU medians ranging from `1.875` to `3.298 ms`, too noisy to decide.
A second bounded screen temporarily raised the per-process median sample count
to 100. Its generic/prefix/prefix/generic complete GPU medians were
`1.992/1.860/1.878/1.881 ms`: one adjacent pair improved about `6.6%`, while
the other improved only about `0.2%`. More importantly, the full wall medians
were `2.650/2.599/2.669/2.614 ms`; generic and prefix pair means were both
approximately `2.63 ms`, with the candidate slightly slower.

The temporary repeat-limit change was removed. P4 therefore remains on the
generic pack source by default, while the existing explicit force switch stays
available for attribution. Uniform BF16 retains its independently established
automatic prefix-only route. This rejects treating isolated pack throughput as
end-to-end value; it does not reject a future fused pack/finalize path that
removes a larger fraction of the command. This was ordinary kernel
specialization, not LTP/WBA.

### Rejected adjacent-subblock Q4_K metadata reuse (2026-08-31)

A temporary default-off B64 source variant paired only logical Q4_K subblocks
`(0,1), (2,3), ..., (14,15)`. Each pair shared scale/min extraction while
retaining the current Q-byte offsets, low/high-nibble mask, shared-memory
locations, barriers, and output representation. Current and candidate sources
had distinct cached pipelines and explicit same-process selection; production
default behavior was unchanged.

The safety contract passed. A real Qwen3.5-9B Q4_K_M fixture exercised both
nibbles, both metadata families, multiple 256-value blocks including the
`(14,15)` wrap, a complete batch-64 tile, and a batch-65 tail. The test compared
every Float32 gate bit and every Float16 fused up/SwiGLU activation bit against
the current route and passed `1/1`.

The acceleration gate failed in both order strata. On Apple M2 Max, the exact
`4096 -> 12288`, batch-64 gate-plus-fused-up command was warmed five times and
measured in ten ABBA plus ten BAAB blocks, with two samples per variant averaged
inside each block. ABBA current/candidate medians were `1.705812/1.900917 ms`:
the candidate regressed `10.264%` and won only `3/10` blocks. BAAB medians were
`1.828229/1.830000 ms`, a `0.097%` regression, with `6/10` wins. Both missed the
predeclared `>=3%` and `>=8/10` thresholds despite bit-exact output.

The candidate, tests, and transient ABBA option were removed; no batch-2048 or
27B escalation was admitted after the small exact-route gate failed. The likely
trade is less metadata arithmetic for half as many active weight-loader lanes
and two serialized subblock stores per even lane, but hardware counters did not
attribute the regression. The bounded conclusion is simply that metadata-only
pairing is not a useful acceleration on this measured corridor. This was
ordinary Metal scheduling, not LTP/WBA.

### Operator-scoped Q4_K x16 decode routing (2026-08-31)

The pre-existing `simd_mv_q4k_f32_x16` kernel processes two output rows per
simdgroup. A global force-on experiment on Qwen3.8-27B suggested roughly a 3%
decode gain, but a first FFN gate/up-only default recovered only about 0.8%.
Inspection found that the hot decode wave had not carried each projection's
`QuantWeight` route metadata into `encode_gemv`, so the existing tag selector
could not separate full-attention from recurrent projections. The decode path
now carries this existing metadata without changing weight bytes, cache layout,
command boundaries, or kernel arithmetic.

The useful policy was found by falsification rather than logical traffic alone.
Relative to the FFN gate/up default, x16 on full-attention Q/K/V/output was flat:
`63.483/63.483 ms/token`, with only `3/6` candidate wins. Recurrent/DeltaNet Q4
QKV, gate, alpha, beta, and output routes measured `64.024/63.414 ms/token`, a
`0.963%` throughput gain with `5/6` wins. Enabling all non-FFN projection tags
measured a `1.134%` gain and `6/6`, but the null full-attention ablation excludes
those operators from the automatic policy.

The final default requires an immutable typed capability issued only when both
measured GGUF identity strings match (`general.name=Qwen_Qwen3.8 27B` and
`general.basename=Qwen_Qwen3.8`), then applies exact tag-plus-shape contracts
for Qwen FFN gate/up and the five recurrent projection families. Missing,
conflicting, Qwen3.6, and future model identities fail closed. Against complete rollback,
a guarded adaptive-QBit Qwen3.8-27B Q4_K_M run at prompt 256 and generation 12
measured `64.313/62.676 ms/token`: a `2.545%` time reduction or `2.612%`
throughput gain, with default winning all six alternating pairs. Token IDs were
identical, maximum selected top-1 logit delta was `0.00108242`, every adaptive
full-attention cache reached the expected live length, and the process exited
normally. The ordinary KV top-1 route measured `60.544/58.803 ms/token`, a
`2.875%` time reduction or `2.960%` throughput gain, also `6/6`. Both used a
`35%` system-memory floor and `24576 MiB` process-tree cap.

A separate short default/rollback quality pair closed the richer semantic gate.
Both modes emitted the same eight token IDs and text, matched top-1 `8/8`,
ranked and set-overlap top-2 `14/14`, exact-top1 coverage `7/7`, and ECS
mean/minimum `1.0/1.0`. Both retained all 16 adaptive owners, no Float32 owner,
consistent publication, and `3.7647x` logical density. Cold exact-prefill time
was intentionally excluded from the speed claim because the first process paid
Metal source compilation.

This is a one-device, short synthetic-token certificate. The ordinary timing
harness does not independently compare generated traces, quiet-host gating was
not required. The identity certificate is intentionally narrow and can withhold
the optimization from a semantically identical repack; this is safer than
silently extending a performance default to an unmeasured model. The adaptive
run supplies the semantic and cache-publication check. `QWEN35_Q4K_GEMV_X16=0` remains the exact rollback;
`=1` remains a global experimental force switch. Full-attention and FFN-down
stay on their prior routes. This is ordinary shape- and operator-scoped kernel
routing, not LTP/WBA.

### Rejected Q4_K x16 fused-add FFN-down routing (2026-08-31)

A temporary two-row-per-simdgroup Q4_K fused-add kernel was restricted to the
typed Qwen3.8-27B `17408 -> 5120` FFN-down route at batch one. Random,
zero-input, and alternating-scale cases compared it with both the current GPU
kernel and CPU `QuantMatmul`. Candidate/current maximum absolute error stayed
below `7.2e-7`, CPU cosine stayed above `0.9999999999976`, and zero input
returned the residual bit-exactly. A guarded same-process Apple M2 Max probe
measured current/candidate GPU p50 `0.161208/0.153083 ms`, a `5.040%` local
reduction, with candidate wins `9/10` ABBA and `10/10` BAAB.

The value did not survive recomputation at the full decode boundary. A new
generic `--compare-only` mode skips unrelated standalone wall/profile runs
before the paired environment gate. With 32 greedy top-1 tokens and 16
interleaved pairs, current/candidate means were `2170.67/2156.33 ms`: only a
`0.660%` candidate reduction and `11/16` wins. Body-only means were
`2170.91/2166.85 ms`: `0.187%` and `9/16` wins. Both miss the `>=80%` paired-win
gate. A separate adaptive quality pair preserved identical eight-token output,
top-1 `8/8`, top-2 `14/14`, ECS `1.0`, all 16 cache owners, and consistent
publication. A longer run was excluded because both branches thermally or
allocator-wise degraded by more than 2x before the paired section.

The raw paired samples and temporary candidate source were not retained, so
these measurements are a bounded rejection record rather than an independently
reproducible performance certificate.

The candidate kernel, selector, policy test, and probe were removed; the
current FFN-down fused-add kernel remains. Local GPU timing established the
mechanism but not product value. The compare-only harness improvement stays as
the reusable evidence tool. This was ordinary kernel specialization, not
LTP/WBA.

### Rejected lane-preserving Q4_K metadata broadcast (2026-08-31)

The next Q4_K B64 experiment preserved both adjacent SIMD loader lanes. Each
lane still loaded its own quantized payload and wrote its own 16 dequantized
values, while only the even lane decoded the common scale/min bytes and sent
them to the odd lane with `simd_shuffle`. This avoided the idle-lane mechanism
of the earlier paired-dequantization candidate. The source specialization was
compile-time only, default off, and left Q offsets, nibble masks, barriers,
shared-memory layout, and the production path unchanged.

The guarded safety contract passed on a real Qwen3.5-9B Q4_K_M fixture. It
compared every Float32 gate output bit and every Float16 fused up/SwiGLU output
bit for the exact batch-64 B64 route and reported `1/1` green.

Performance rejected the candidate before any larger escalation. On Apple M2
Max, exact `4096 -> 12288`, batch 64, five warmups, and ten same-process ABBA
blocks measured current/candidate completed-command GPU p50
`1.638875/1.722854 ms`. The candidate was `5.124%` slower and won `0/10`
blocks. The predeclared gate required at least `3%` improvement and `8/10`
wins in both ABBA and BAAB, so the decisive first-stratum failure triggered the
fail-fast rule: BAAB, larger batches, and Qwen3.8-27B product runs were not
executed.

The likely trade is a small reduction in scale/min arithmetic for an added SIMD
shuffle dependency, but hardware counters did not identify the cause. The
bounded result is that this lane-preserving broadcast is slower on the measured
corridor. The source variant, selector, focused test extension, and transient
ABBA option were removed. This was ordinary SIMD scheduling, not LTP/WBA.

### Rejected Q6_K NSG4 layout for recurrent QKV (2026-08-31)

A metadata-only inventory of the local Qwen3.8-27B Q4_K_M GGUF found a useful
unmeasured split: its 48 recurrent `attn_qkv` tensors all have shape
`5120 -> 10240`, but 24 are Q6_K and 24 are Q4_K. The Q4 half is covered by the
new operator-scoped x16 route; the Q6 half remains on the current
`NSG=2, NR0=1` kernel. Earlier NSG4 experiments were global or involved Q6
FFN-down/head shapes, so this exact QKV shape justified one bounded test.

The probe reused the existing alternate-layout pipeline and compared the
current `2x1` launch with `NSG=4, NR0=1`; production routing was never changed.
Both variants passed the existing full-vector `1e-3` maximum-difference guard
against the current matmul path. On Apple M2 Max, a real Qwen3.8-27B Q6_K QKV
weight at batch 1 was warmed five times and measured in ten ABBA blocks.
Current/candidate completed-command p50 was `0.467938/0.465458 ms`, only
`0.530%` improvement, and the candidate won `6/10` blocks.

That misses the predeclared `>=3%` and `>=8/10` local gate. The absolute samples
also shifted from roughly `0.47` to `0.38 ms` during the short run, reinforcing
that a sub-percent median is not robust evidence. BAAB and whole-decode runs
were therefore skipped by fail-fast. The transient probe was removed and no
Q6 production selector changed. This result rejects another launch-geometry
retry, not a future Q6 kernel that removes a larger unit of dequantization,
traffic, or synchronization work. This was ordinary dispatch tuning, not
LTP/WBA.

### Rejected adaptive decode chain2 host overlap (2026-08-31)

A temporary fixed-length diagnostic submitted two adaptive decode tokens on one
named Metal queue. It retained fresh scratch per token, GPU token-ID handoff,
FIFO cache reservations, and ordered wait, validation, and publication. The
candidate matched two serial steps exactly: token IDs, the next token's top-2
continuation, and every adaptive cache length were identical. This did not cover
EOS, grammar, tool, cancellation, or other early-stop semantics.

After one warmup, ten interleaved serial/chain2 pairs at prompt 0 and generation
8 measured means `497.332/492.095 ms`, p50 `498.739/492.069 ms`, and throughput
`16.040/16.258 token/s`. Chain2 reduced wall time by only `1.053%` and won
`7/10` pairs, missing the predeclared `>=3%` and stable-win gate. An earlier
cold comparison that appeared about `93%` faster was excluded because only the
serial branch paid Metal source compilation.

The small warmed gain cannot justify two fresh scratch sets and the larger
failure contract. Recurrent state mutates in place, so a submitted-token failure
would require stop-aware rollback or poisoning the entire advanced state; the
narrow parity result is not a production stopping certificate. The prototype,
cache-tail helper, tests, and probe were removed. Synchronous one-token adaptive
decode remains the default, and the next search targets removal of GPU work
inside a token rather than another command-boundary rearrangement. This was
ordinary bounded pipelining, not LTP/WBA.

### Rejected adaptive split-K chunk 32/128 retune (2026-08-31)

A guarded fixed-snapshot screen tested the existing split-K block-size override
at an 8,192-token prefix, one appended token, and ten repetitions. Fresh Apple
M2 Max processes kept automatic T4 and prefix-pack policy, the 35% free-memory
floor, and the 24,576 MiB tree cap. Complete attention, pack, and finalizer GPU
medians for P4 were `3.087/1.862/2.872 ms` at chunks `32/64/128`. The existing
64-token block was therefore about 40% faster than 32 and 35% faster than 128
on the dominant P4 tier.

BF16 medians were `1.960/1.745/1.709 ms`. The 128-token result is only about
2.1% below 64 in an unpaired process-level screen, while the measured product
map has four BF16 and twelve P4 attention layers. That is insufficient evidence
for a separate BF16 policy branch. No source route changed; chunk 64 remains the
default and `QWEN35_ADAPTIVE_SPLITK_CHUNK` remains available for diagnostics.
This rejects a simple geometry retune, not a future algorithm that removes
summary traffic or dequantization work. It is ordinary split-K tuning, not
LTP/WBA.

### Rejected adaptive split-K FP16 partial output (2026-08-31)

A temporary explicit-only source variant kept the split-K online-softmax
statistics and accumulation in FP32 but stored the stage-one `partial_o`
summary in FP16 before stage two reloaded it. At the measured 24-head,
256-dimensional, 64-token-block geometry this would reduce split-K scratch
from `3,170,304` to `1,597,440` bytes at 8K capacity and from `6,340,608` to
`3,194,880` bytes at 16K capacity. It also removes about 3 MiB of stage-one
write plus stage-two read traffic per attention layer at 8K.

The existing 8,191-token Metal contract passed with its strict cosine and
maximum-difference bounds and retained byte-identical packed K/V. Performance,
however, rejected promotion. An initial four-process P4/BF16 screen contained
one slow FP32 BF16 observation and therefore appeared favorable. A second
BF16-only A/B/B/A screen reproduced FP32 at `1.762/1.763 ms` and FP16 at
`1.763/1.770 ms` for the complete attention, pack, and finalizer GPU interval.
The candidate was effectively neutral and about 0.2% slower by pair means.

P4 was also unstable and did not establish a gain. The source variant, policy,
scratch split, and test extension were removed. The result does not reject FP16
scratch as a compactness option for a future multi-flight design, but current
scratch is already small and shared; halving it does not accelerate one-token
decode on this M2 Max corridor. This was ordinary intermediate-format tuning,
not LTP/WBA.

### Rejected Q6_K batch-one ILP2 pipeline (2026-08-31)

The production adaptive-decode profile showed that quantized weight reads, not
adaptive attention, dominate the remaining one-token interval. Q6_K FFN-down
alone accounts for a large repeated corridor, so a temporary kernel unrolled
the 256-value block loop by two and accumulated the two blocks independently.
This preserved the GGUF layout and total bytes while testing whether a single
dependent accumulation chain was hiding useful instruction-level parallelism.

The candidate passed the full-vector `1e-3` maximum-difference guard. A guarded
same-process Apple M2 Max test used the real Qwen3.8-27B Q4_K_M weights, five
warmups, ten alternating ABBA/BAAB-style blocks per Q6_K shape, and required at
least 3% p50 improvement with at least 8/10 candidate wins. The dominant
`17408 -> 5120` FFN-down shape regressed from `0.7386` to `0.7410 ms`
(`-0.321%`, `3/10` wins). The output head changed from `3.1073` to `3.0935 ms`
(`+0.447%`, `8/10`), recurrent QKV from `0.2804` to `0.2764 ms`
(`+1.447%`, `7/10`), and the small projection from `0.1809` to `0.1780 ms`
(`+1.639%`, `6/10`). None cleared the gate.

The experiment therefore rejects block-loop ILP as the next performance lever:
it does not reduce the dominant Q6_K byte stream, and the primary FFN-down
corridor became slightly slower. The temporary kernel, pipeline, and diagnostic
route were removed. A future Q6_K candidate must change the representation or
eliminate a larger execution boundary, with quality and memory gates stated
before integration. This was ordinary kernel scheduling, not LTP/WBA.

### Rejected 256-value Gaussian P4/P5 compression of Q6_K weights (2026-08-31)

An offline CPU-only probe tested whether the existing Gaussian QBit codec could
reduce Qwen3.8 Q6_K weight traffic before investing in a Metal format or
decoder. It sampled 256 evenly spaced rows from the real
`blk.0.ffn_down.weight` tensor (`17408 -> 5120`) and reconstructed every native
256-value Q6_K block as Gaussian P4 and P5. The forecast charged four bytes of
record metadata in addition to payload: 140 bytes for P4, 172 for P5, and 214
for a native-Q6 escape, versus the original 210-byte Q6_K block.

The original conservative policy admitted no compressed blocks. Across 17,408
sampled blocks, the mean maximum-residual-to-block-standard-deviation ratios
were `1.0068` for P4 and `0.5605` for P5; the `P4 <= 0.20`, `P5 <= 0.10`
policy selected 100% native Q6 and therefore grew by `1.9%`. All-P5 forecast
`1.221x` compression, but five deterministic operator inputs produced cosine
between `0.9851` and `0.99978`, and one changed the sampled-row top-2 set.

A bounded threshold sweep found no near-lossless operating point. A P5-only
threshold of `0.35` forecast only `1.072x` compression and still reduced the
worst cosine to `0.998997`. A threshold of `0.75` forecast `1.162x`
compression but reduced the worst cosine to `0.997995`. Adding P4 at a `0.50`
threshold reached `1.123x`, with worst cosine `0.998494`. Sampled-row top-1 and
top-2 happened to remain stable for those mixed policies, but these are
operator proxies, not token, semantic, or ECS measurements; the predeclared
`>=1.12x`, cosine `>=0.99999`, exact ordered-top-2 gate was not met.
An independent 32-row rerun of the `0.50/0.50` policy forecast `1.127x` but
also swapped sampled-row top-1 on both random-uniform activation families.

The likely structural cause is that one mean and standard deviation over 256
values discard the smaller-scale structure already represented inside Q6_K.
The result rejects this specific single-moment Gaussian block format before any
Metal work. It does not reject a future weight format that preserves sub-block
scales or explicitly codes a bounded residual. The offline probe is retained as
a reproducible quality/size falsifier. This was ordinary approximate weight
representation research, not LTP/WBA.

### Rejected Q6_K recurrent producer/conv-shift fusion (2026-08-31)

A temporary batch-one Metal kernel fused the exact recurrent QKV Q6_K GEMV
producer with the existing conv-state shift, convolution, SiLU, and Q/K/V split.
It removed the 10,240-float intermediate and one consumer dispatch on each of
the 24 Q6_K recurrent layers, a static reduction of about 1.97 MB of
intermediate write-plus-read traffic per decoded token. The production route
remained opt-in while a direct operator seam compared it with the current Q6_K
GEMV followed by the already-fused conv/shift consumer.

The guarded Apple M2 Max probe used a real `5120 -> 10240` Qwen3.8-27B Q6_K
recurrent tensor. Random, zero-input, and hostile alternating-scale cases had
exactly zero maximum difference for Q, K, V, and final conv-state. Five warmups
and ten four-command blocks then measured current/candidate completed-command
GPU p50 `0.30017/0.30192 ms` in ABBA and `0.30388/0.30271 ms` in BAAB. The
candidate therefore changed sign by order and stayed within about `0.6%` of the
baseline, far below the predeclared `>=3%` promotion gate.

Lane 0 must perform the scalar conv/shift after the SIMD reduction while the
other 31 lanes are idle, which plausibly consumes the dispatch and tiny-buffer
saving. The measurement does not establish that mechanism, but it decisively
rejects this implementation before whole-model integration. The temporary
kernel, selector, and probe were removed. A future recurrent fusion needs a
larger shared unit or parallel consumer mapping, not another lane-0 scalar tail.
This was ordinary producer-consumer fusion, not LTP/WBA.

### Rejected native-granularity subscale-P5 compression of Q6_K weights (2026-08-31)

The retained offline weight probe was extended with a second representation
falsifier that preserves Q6_K's 16-value grouping granularity, but not its
native signed scale bytes. Each 256-value block is requantized to signed
five-bit values, sixteen new unsigned scale codes, and one FP32 master scale.
The fixed record is 180 bytes versus 210 bytes for native Q6_K, a theoretical
`1.1667x` reduction.

All-subscale-P5 preserved sampled-row ordered top-2 on five deterministic
activation families, but this proxy did not imply near-lossless arithmetic.
On 256 evenly sampled rows of the real `blk.0.ffn_down.weight`, output cosine
ranged from `0.998450` to `0.999946`. A one-bit tier bitmap and native-Q6
escape policy was then screened by the maximum residual divided by native
block standard deviation. Threshold `0.08` retained cosine between `0.999920`
and `0.999998`, but compressed only `1.005x`. Threshold `0.11` compressed
`1.140x` and still preserved sampled-row ordered top-2, but cosine fell to
`0.999225--0.999953`.

An independent 32-row, different-seed run at threshold `0.11` reproduced the
size result (`1.142x`) and weakened the numerical result further: cosine was
`0.998954--0.999611`. The tested selector therefore cannot satisfy the
predeclared joint gate of at least `1.12x` compression, cosine at least
`0.99999`, and exact ordered top-2. Sampled rows and synthetic activations are
only operator proxies, not token top-1/top-2 or ECS; this limitation weakens a
promotion claim and cannot rescue a representation that already fails its
local numerical gate.

No Metal format or decoder was implemented. The probe remains a reproducible
negative falsifier. The next admissible Q6_K compression candidate must be
bit-exact or use a selector calibrated against real hidden activations and the
token/ECS boundary. In particular, preserving sub-block scale granularity by
itself is insufficient when the sixth value bit is discarded. This was
ordinary approximate weight representation research, not LTP/WBA.

### Rejected exact sparse-bitplane compression of Q6_K weights (2026-08-31)

A bit-exact follow-up tested whether one of Q6_K's six value bitplanes could be
stored as a majority bit plus sparse exception indices. The candidate retains
the other five bitplanes densely and preserves the native sixteen scale bytes
and FP16 `d` unchanged. Its compressed record is `180 + k` bytes for `k`
exceptions versus 210 native bytes; a one-bit type bitmap selects compressed
records or native escape. A block can therefore save space only for `k <= 29`.

The CPU probe extracts all 256 unsigned Q6 codes using the production Q6_K
layout, chooses the sparsest plane and polarity, validates sorted in-range
exception indices, reconstructs the dropped bit, and fails closed unless every
code is identical. On 256 sampled rows of real `blk.0.ffn_down.weight`, all
17,408 blocks reconstructed exactly, but none had `k <= 29`: 51 blocks were in
`k=64--95` and 17,357 were in `k=96--128`. The type bitmap made the forecast
slightly larger than native Q6_K (`0.999405x` native-over-forecast).

The same falsifier on 256 sampled rows of the real Q6_K recurrent
`blk.1.attn_qkv.weight` reproduced the result: zero of 5,120 blocks compressed,
17 had `k=64--95`, and 5,103 had `k=96--128`. This independently attacks the
possibility that FFN-down alone has unusually dense bitplanes. The result is a
representation failure, not a decoder-performance result: no Metal format was
implemented because the byte gate failed before timing was admissible.

The exact reconstruction logic is retained as a reproducible distribution
falsifier. The negative result rules out majority-plus-UInt8-exception coding
of one raw Q6 bitplane on the measured tensors. It does not rule out structured
multi-block entropy coding for disk storage, but such a format would add random
access and decode costs and is not currently justified for resident weights.
This was ordinary lossless representation screening, not LTP/WBA.

### Rejected vectorized Q6_K large-GEMM epilogue (2026-08-31)

A temporary Metal change replaced the scalar FP32 output loop in the
large-batch Q6_K GEMM and residual-add kernels with `float4` loads and
stores on complete 64-row tiles. It retained the existing threadgroup spill,
the historical FP16 rounding before the FP32 write, and the scalar tail for
partial row tiles. The change therefore targeted only epilogue instruction
count; it did not reduce Q6_K weight traffic, dequantization, matrix work,
threadgroup memory, or command boundaries.

The existing Q6_K batch-16 parity test compiled both candidate kernels and
executed the non-add path, reproducing cosine `1.0` and maximum absolute
difference `0.0048918724` against the CPU reference. A guarded
candidate/baseline/baseline/candidate diagnostic then measured the standalone
Qwen3.5-9B Q6_K `12288 -> 4096` FFN-down shape at batch 64. Candidate
versus adjacent baseline completed-command medians were `2.470/2.357 ms` and
`2.307/2.266 ms`: regressions of about `4.8%` and `1.8%`. The unchanged Q4_K
FFN-down row was retained as a noise control; its first pair moved materially,
while the second pair was nearly equal, so an earlier unbalanced apparent gain
was rejected rather than promoted.

The temporary kernel change was removed. The attribution harness retains a
`--name-filter` option so future probes can exclude unrelated large operators;
this is especially important because an unfiltered large-batch run includes the
output head and can exceed the useful watchdog-safe scope. The result rejects
the non-add vectorized epilogue on the measured route and gives no reason to
widen the same transformation to the product residual-add route. A future
large-batch Q6_K change must reduce the weight/dequantization corridor or a
larger execution boundary, not only rearrange its final stores. This was
ordinary kernel optimization, not LTP/WBA.

### Qwen3.8 structured forced-span decode validation (2026-08-31)

The already-default forced-span route inside opt-in constrained tool decoding
was revalidated against the real Qwen3.8-27B Q4_K_M model. The tested edit-mode
schema produced 39 output tokens, of which 12 were transported through exact
deterministic spans. Four sequential fresh-process pairs were run in both
candidate-first and rollback-first order. Candidate versus
`QWEN35_CONSTRAINED_FORCE_SPAN_OFF=1` decode wall times were
`2073.4/2157.0`, `2065.6/2157.0`, `2063.3/2094.1`, and
`2003.4/2088.7 ms`. The candidate won all four pairs; paired speedups were
`4.03%`, `4.43%`, `1.49%`, and `4.26%` (mean `3.55%`).

Every pair emitted the same 39 token IDs and the same parsed
`edit_mode(mode=safe,dry_run=true)` JSON. The model's full Q6_K output head is
about 994.6 MiB of logical row traffic per unconstrained token at
`5120 -> 248320`; constrained allowed-ID selection collapses that scan, while
forced spans additionally batch exact body state updates and remove
intermediate head/synchronization boundaries.

The reusable suite was then hardened to create its log directory, use this
repository's guarded runner, alternate pair order by repetition, and fail
closed unless candidate and rollback have identical token IDs and canonical
parsed JSON. It additionally compares each parsed call with the exact expected
function and arguments, requires positive forced-span coverage in the candidate
and zero coverage under the kill switch, and rejects unbalanced repetition
counts. It now rebuilds by default, fingerprints the content of the Crystal
compiler inputs and Metal bridge, records the source revision and binary digest,
and runs both candidates from a minimal allowlist environment. Explicit
`REBUILD=0` reuse is labelled `prebuilt-explicit`, rather than presented as a
fresh-build certificate. Synthetic token mismatch, malformed JSON,
valid-but-wrong tool-call, and inherited-env checks were all rejected or
isolated before the model gate.

The balanced Qwen3.8 gate covered four schema configurations: required enum and
boolean fields, a required open string plus bounded integer, an optional-field
variant, and a two-tool choice. The required and optional read schemas selected
the same output trace, so this is four grammar configurations but only three
distinct emitted tool-call traces. All eight fresh-process pairs preserved
exact token IDs and parsed JSON. Decode speedups were `3.02%..4.22%`, mean
`3.90%` in the preliminary gate. The final fresh-build/minimal-environment run
measured `3.780%..4.306%`, mean `4.023%`; fresh-process request speedups,
including unchanged prefill and model setup, were `0.917%..2.038%`, mean
`1.407%`. It used source revision
`c454977658518c419d1d21e6ffd0e5f988efbad8`, source-input SHA-256
`8c9c97fef37119d90cf6fb79357b1c8d8fc70cf647815911b28cf141baa56983`,
and freshly built binary SHA-256
`18a517e522e6bcf84714c6733aacf6ff48d9bf268099225823ff568f7667d706`.
All runs exited zero under the 24 GiB
process-tree cap and 35% free-memory floor; the host was deliberately not
quiet-gated.

This closes the requested multi-schema ABBA gate, but remains a scoped
structured-decode certificate rather than a general Qwen3.8 inference claim.
Only 8 or 12 output tokens were transported through forced spans, the corpus is
small, and the total-request gain is materially lower than the decode-only
gain. The feature therefore remains default-on only inside the experimental
constrained structured mode, with the existing kill switch. The next widening
gate is a real CrystalBall tool task or an exact extension that increases
deterministic-span coverage without crossing a free-form or choice boundary.

### Rejected ordinary-F32 GQA6 split-K sharing (2026-08-31)

A temporary Qwen3.8 specialization changed ordinary F32 split-K attention from
one 32-thread group per query head and context block to one 192-thread group per
KV head and block. Six SIMD groups shared a 16-token F32 K/V tile while keeping
the existing `{partial_m, partial_l, partial_o}` ABI and stage-2 reduction. The
candidate was exact-shape and default-off; adaptive QBit attention was excluded
because its existing GQA6 kernel already shares K/V rows across the six query
heads.

The direct Apple M2 Max operator gate used the real Qwen3.8 geometry
`24 query heads / 4 KV heads / head_dim 256` at cache lengths
`127, 128, 129, 255, 256, 257`, with NaN guard rows after the visible prefix.
The guarded spec run passed all three examples. Candidate versus generic maximum differences were
`0` for `partial_m`, `7.6293945e-6` for `partial_l`, `1.4305115e-6` for
`partial_o`, and at most `1.1175871e-8` for final output. Candidate and generic
outputs both stayed within `4.4703484e-8` of the CPU reference.

The whole-model paired gate rejected the speed claim. On Qwen3.8-27B Q4_K_M,
eight greedy decode tokens after a 1,024-token prefill measured baseline versus
candidate mean `473.63/500.58 ms` and median `472.11/481.74 ms`; the candidate
won only one of three pairs. At a 2,048-token prefill, baseline versus candidate
mean was `495.18/504.29 ms` and median was `485.04/487.29 ms`; the candidate won
two of three pairs but remained slower in both aggregate coordinates. Both runs
used one warmup, three interleaved repetitions, a 24 GiB process-tree cap, and a
35% free-memory floor.

The likely cost is the combination of 192-thread coordination, a smaller
16-token tile, and additional threadgroup barriers. That explanation remains a
hypothesis; the paired wall regression is the decision boundary. The temporary
kernel, selector, and test probe were removed. Longer ordinary-F32 contexts are
not a strong retry target because the intended long-context product route is
adaptive QBit, where GQA6 sharing is already implemented. Reopen only with a
materially different synchronization or layout argument. This was ordinary
kernel specialization, not LTP/WBA.

### Rejected per-layer full-attention preparation CogniGraph (2026-08-31)

A temporary default-off decode route used the existing `ComputeGraph` scheduler
for Q/K/V projection, Q/gate split, Q/K head RMSNorm, and Q/K RoPE. Exact buffer
access declarations compiled the eight operations into four dependency waves:
Q/K/V projection; Q/gate split plus K norm; Q norm plus K RoPE; and Q RoPE. A
fail-closed runtime certificate required eight operations, four waves, three
barriers, and maximum wave width three. Adaptive QBit and the Q8 K/V dual-GEMV
route remained on the existing serial path.

The candidate was reachable and completed a guarded real Qwen3.8-27B smoke,
but it did not accelerate product decode. With 16 greedy tokens, one warmup,
and eight interleaved pairs, serial baseline versus graph mean was
`900.61/902.20 ms`; median was `901.31/901.30 ms`; the graph won only `3/8`
pairs. The run used a 24 GiB process-tree cap and 35% free-memory floor without
waiting for WindowServer to become quiet.

The weak result matches the static ceiling. Across all 16 full-attention
layers, the real Qwen3.8 tensors contain 115,998,720 bytes/token
(`110.625 MiB`) of K+V projection
weights that could theoretically hide under the much larger Q projection. That
is only about `0.734%` of the current 15,078 MiB/token logical matmul stream.
Graph construction and dependency analysis also occurred once per layer and
token, while the concurrent dispatches still contended for unified-memory
bandwidth. These are mechanism explanations; paired whole-decode wall is the
decision boundary.

No numerical-parity promotion claim is made because the candidate failed the
speed gate before retention was admissible. The temporary route was removed.
Reopen this scheduling frame only if graph bindings can be compiled and reused,
or if direct host instrumentation shows that eliminating a larger encoder or
command boundary can exceed the `3%` product gate. This was ordinary concurrent
dispatch scheduling, not LTP/WBA.

### Rejected one-cut exact Q6_K output-head norm pruning (2026-08-31)

An offline falsifier tested an exact two-pass output-head scheme on real hidden
rows from Qwen3.8-27B Q4_K_M. The first pass reads a block-aligned prefix of
every Q6_K vocabulary row. A small seed set is evaluated completely, while the
remaining suffix contribution is bounded by Cauchy-Schwarz using one precomputed
Float32 suffix norm per row. A row is discarded only when its upper bound cannot
enter the exact ordered top-2. The probe also evaluates an unrealistically free
oracle threshold using the true exact second score, so a weak seed incumbent
cannot explain a rejection.

The probe exactly scanned the real `248320 x 5120` Q6_K output head for three
consecutive hidden rows. CPU Float64 ordered top-2 matched the normal routed
full-vector helper for all three samples, every discarded row satisfied the
bound, and every survivor scan reproduced the exact ordered top-2. At cuts after 5 and 10 of 20
Q6_K blocks, all 248,320 rows survived. At 15 blocks, the best sample pruned only
`1.907%` of rows and another pruned none. At 18 blocks, survivor counts were
`3,639`, `49,570`, and `2`; after charging the prefix, survivor suffixes, and a
four-byte sidecar per row, output-head savings were `9.758%`, `7.909%`, and
`9.905%`. The seed and oracle survivor sets were identical at every cut.

The measured output head is about `6.6%` of current decode wall, so a `3%`
whole-decode target requires at least `45.455%` head savings before dispatch,
scratch, compaction, and conservative floating-point-bound costs. The best
minimum optimistic head saving was only `7.909%`, a whole-decode ceiling of
about `0.522%`; even the best sample reached only about `0.654%`. A production
Metal implementation is therefore not admissible for this scheme.

This is a robust rejection of one block-aligned prefix pass plus a per-row L2
suffix bound on the measured model and hidden rows. Three positions are not a
general corpus, but the free oracle threshold and the factor-of-5.7 miss against
the required head saving make seed tuning an invalid retry. The result does not
rule out materially tighter multi-stage bounds, constrained vocabulary
frontiers, or approximate/learned retrieval. The retained probe is an exact
geometry and economics falsifier. The routed helper can silently fall back to
CPU, so no fused-Metal top-2 route certificate is claimed; a conservative
Float32 Metal bound would need additional upward inflation and cannot improve
the byte ceiling. This was ordinary exact pruning research, not LTP/WBA.

### Resident decision-tail fusion for constrained tool decoding (2026-09-01)

When a deterministic token span ends immediately before a finite grammar
choice, the constrained Qwen controller can now consume the whole known span
and append an allowed-ID output head to the same adaptive prefill command. The
selected token is returned but deliberately remains unconsumed, preserving the
ordinary next-token state boundary. The route requires the existing adaptive
resident final-head policy and a supported Q6_K output tensor; every other
configuration uses the exact previous two-step fallback. The dedicated rollback
is `QWEN35_CONSTRAINED_TOKEN_STAGE_DECISION_TAIL_OFF=1`.

The caller-owned Metal helper rejects committed commands, empty or out-of-range
allowed-ID sets, invalid row offsets, and undersized buffers before encoding.
It allocates its intermediates through the active prefill scratch Arena, while
the existing adaptive command retains commit, completion, cache-finalization,
and result-visibility ownership. The complete forward suite passed 29 examples
on Apple M2 Max. That suite also disables the fused allowed-token head on a
real Qwen3.8 adaptive state and verifies that the exact full-logit fallback
selects the same non-global winner while advancing the live cache once. A
direct diagnostic Qwen3.8-27B run observed two
`adaptive_final_resident_top1_allowed_appended` route hits, removed two separate
decoder waves and synchronizations, and emitted the same 39 token IDs and typed
tool call as rollback.

Two independent balanced ABBA suites then covered required enum/boolean,
required string/integer, optional-field, and two-tool schemas. All 16 pairs
preserved exact token IDs and expected canonical parsed tool calls. Candidate
decision-tail coverage was one to four choices per call and rollback coverage
was zero. The first suite measured mean paired constrained-decode speedup
`5.412%` (range `-10.126..11.845%`); the second, on a materially slower host
interval, measured `6.206%` (range `-4.976..12.987%`). Across all 16 equally
weighted pairs the mean was `5.809%`. The corresponding combined mean
fresh-process request improvement was only about `0.035%`, because the unchanged
275-token prefill dominated and varied substantially. The prebuilt binary SHA-256
was `f156c527a16f8d6bda0f843158f53571f5cc0b9b617277ec2bb24a4a4e953b88` and
the content fingerprint was
`ac230c09ab5cbe6a96761a2d4924212e4377a4f00dc8cb997518097a9dad4bd3`.

A third fresh-build suite exercised the now-enforced threshold directly. It
again preserved all eight token-ID and typed-call pairs, reported candidate
decision-tail coverage `1..4` and zero rollback coverage, and passed the 3%
gate with mean paired constrained-decode improvement `6.094%` (range
`-16.700..12.921%`). Its mean total-request result was `-7.656%` (range
`-56.791..1.995%`) because one candidate cold-prefill row dominated the small
sample. The run is retained at
`/private/tmp/qwen35_span_suite_decision_tail_current_gate_20260901_63354`;
its source-input SHA-256 was
`006ff7288d67eb97a4b7c7a55a98e05c99b0368bfb88094fd107ae2d3c8e419a`
and binary SHA-256 was
`ec4af3a4028f04d2eab9ff48a958194971e8c119d56ae16ae0e23d96be4a09fd`.
The subsequent unsupported-head exact-fallback patch does not alter the
Q6_K fused route exercised by that suite.

This admits a scoped incremental constrained-decode win on Qwen3.8-27B with
adaptive resident QBit KV, not a whole-request or unconstrained-text speed
claim. Per-pair timing remained noisy and sometimes negative, while both
independent aggregate decode results cleared the predeclared 3% gate. The
optimization is ordinary grammar-certified scheduling and output-head fusion,
not LTP/WBA.

The suite now enforces that economics boundary instead of merely reporting it:
`MIN_DECODE_SPEEDUP_PCT` defaults to `3.0`, and a lower aggregate paired decode
mean exits non-zero after semantic parity and route-activation checks. Replaying
the two retained pair tables through the current summarizer passed at `5.412%`
and `6.206%`; a synthetic zero-mean table failed closed. The original ABBA logs
predate this script-level enforcement, so their recorded source fingerprint is
retained rather than relabeled as if the stronger gate had produced them.

**decision:** Keep resident decision tails default-on only inside the
token-option constrained tool-call route. Preserve exact rollback with
`QWEN35_CONSTRAINED_TOKEN_STAGE_DECISION_TAIL_OFF=1`. Require a CrystalBall
task-level run or a wider exact schema corpus before broadening the performance
claim.

### Eight-value P4 split-K loader on Apple M2 Max (2026-09-01)

The uniform-P4 split-K stage now loads eight adjacent dequantized values at a
time. Each group shares four bitplane-byte loads and one mean/sigma header pair,
while retaining the existing FP32 tile, dot-product order, softmax reduction,
current-token path, and adaptive cache representation. The specialization is
automatic only for the exact `Apple M2 Max` device name and only inside the
already admitted one-token, uniform-P4 split-K route. Other devices retain the
portable loader. `QWEN35_ADAPTIVE_P4_SPLITK_T8=0` is the exact rollback;
`=1` remains an explicit research override.

A protected 8K integration contract compared serial, scalar split-K, T4,
legacy split-K, and T8 against the independent CPU reference. It passed with
the same output tolerances and byte-identical persisted K/V payloads. The T8
case deliberately enabled both T4 and T8, proving that policy selects one
canonical T8 pipeline instead of composing incompatible loaders. The full
guarded QBit suite passed `149` examples with zero failures or errors and one
optional model-backed case pending.

The performance falsifier uses fresh restored cache state for every sample,
prewarms both variants, and alternates order inside one process. At an 8,192
token P4 prefix, ten pairs measured mean wall time `2.687 -> 2.330 ms`
(`13.291%` lower, `8/10` wins) and mean GPU time `2.098 -> 1.615 ms`
(`23.010%` lower, `10/10` wins). At 16,384 tokens, ten pairs measured wall
`4.825 -> 3.879 ms` (`19.608%` lower, `10/10`) and GPU
`4.156 -> 3.247 ms` (`21.867%` lower, `10/10`). Both passed the probe's
predeclared `>=3%` and `>=8/10` wall-and-GPU gate under the 35% free-memory
floor.

A separate real Qwen3.8-27B Q4_K_M semantic pair used a 905-token chat prompt,
eight generated tokens, and adaptive map
`p4;27=bf16,43=bf16,47=bf16,51=bf16`. T8 off/on produced identical exact and
resident token IDs, text, top-1/top-2 coverage, ECS, cache ownership, density,
and consistency metrics. Its single resident free-decode row improved only
about `1.2%` and the forced row was noisy, so it is semantic evidence rather
than a product-speed certificate. With the current map only 12 of 16 full
attention layers use P4; no 13--23% whole-inference claim follows from the
isolated command result.

**Adversary:** The speed certificate is one Apple M2 Max, uniform P4,
head-dimension 256, one-token split-K, and two long contexts. Explicit T8 on
another device or tile remains experimental. Future changes to head dimension,
8-value alignment, uniform-tier routing, bitplane layout, Metal compilation, or
device naming invalidate the certificate. The 8K numerical test and 16K timing
test do not establish arbitrary mixed-tier or cross-device behavior.

**Value proxy:** Reduced plane/header loads explain the candidate, but paired
complete-command wall and GPU time plus numerical and persisted-payload parity
are the admission boundary. Whole-model latency remains a separate coordinate.

**LTP/WBA:** Not claimed. This is ordinary exact-layout kernel specialization
with an explicit portable-loader rollback.

**decision:** Enable the T8 loader only for exact `Apple M2 Max`, uniform P4,
one-token split-K. Keep `QWEN35_ADAPTIVE_P4_SPLITK_T8=0` as rollback and require
new paired and numerical evidence before widening device, tile, tier, or shape
scope.

### Eight-value BF16 split-K loader on Apple M2 Max (2026-09-01)

The uniform-BF16 split-K stage now reconstructs eight adjacent BF16 values from
one aligned `uint4` load. The row stride is 512 bytes and the eight-value
traversal advances by 16 bytes. The host also checks the actual K and V sidecar
base addresses before selecting the specialized pipeline; an unexpected
unaligned buffer falls back to the portable T4 loader. The FP32 tile,
dot-product and softmax order, exact-F32 current-token path, and persisted cache
format are unchanged. Automatic selection is limited to the exact
`Apple M2 Max` device name. `QWEN35_ADAPTIVE_BF16_SPLITK_T8=0` is the rollback;
`=1` remains an aligned-buffer research override on other devices.

The first candidate used two four-value loads. It passed the 8K gate narrowly,
but at 16K its GPU interval improved only `2.508%` with `7/10` wins, below the
predeclared `>=3%` and `>=8/10` requirements. That route was rejected rather
than rescued with more samples. The replacement uses one 128-bit load and
separates the even and odd BF16 halves with integer shifts and masks before
bit-casting to Float32.

A protected 8K contract compared the portable and 128-bit loaders against the
independent CPU reference, verified the actual sidecar base alignment, retained
the exact-F32 current token, and required byte-identical persisted K/V payloads.
It passed. The complete protected adaptive-QBit set then passed `145` examples
with zero failures, errors, or pending cases under a 4 GiB process-tree cap and
35% free-memory floor.

The same-process falsifier prewarmed both pipelines, restored fresh cache state
for every sample, and alternated order for ten pairs. At an 8,192-token BF16
prefix, wall improved `2.492 -> 2.207 ms` (`11.436%`, `8/10` wins) and GPU time
improved `1.706 -> 1.456 ms` (`14.655%`, `8/10`). At 16,384 tokens, wall
improved `3.761 -> 3.007 ms` (`20.051%`, `8/10`) and GPU time improved
`2.734 -> 2.013 ms` (`26.374%`, `8/10`). Both contexts passed the unchanged
wall-and-GPU gate under the 35% free-memory floor.

A separate real Qwen3.8-27B Q4_K_M semantic pair used a 1,055-token prompt,
eight generated tokens, a 4,096-token capacity, and map
`p4;27=bf16,43=bf16,47=bf16,51=bf16`. BF16 T8 off/on produced identical exact
and resident token IDs and text. Both runs retained top-1 `7/8`, ranked top-2
`11/14`, top-2 overlap `13/14`, exact-top-1 coverage `7/7`, ECS `0.888779`, all
16 resident owners, no F32 owner, consistent cache state, and `3.7647x` density.
The single timing pair was order-confounded and is semantic evidence only.

For the measured map, the four BF16 layers account for about `60.29%` of the
adaptive payload, while the vectorized BF16 sidecars themselves account for
`47.06%`. Those byte shares explain why the kernel can matter, but they are not
whole-model speed predictions: the loader does not reduce stored bytes and
does not change packing, stage two, P4 layers, recurrent layers, or weight
traffic.

**Adversary:** The speed certificate is bounded to one Apple M2 Max, uniform
BF16, head dimension 256, one-token split-K, and two long contexts. The runtime
alignment guard closes the local vector-load precondition, but cross-device
compiler behavior and other head dimensions remain unproven. Exactly `8/10`
wins at both contexts clears the declared gate but leaves less noise margin than
the percentage deltas suggest. No whole-inference acceleration is claimed.

**Value proxy:** A 128-bit load and BF16 byte share are mechanism coordinates.
Independent numerical and payload parity plus paired complete-command wall and
GPU time are the admission boundary; the real-model pair establishes semantics,
not product speed.

**LTP/WBA:** Not claimed. This is ordinary exact-layout kernel specialization
with a portable T4 rollback.

**decision:** Enable the aligned BF16 T8 loader only for exact `Apple M2 Max`
inside uniform-BF16 one-token split-K. Preserve
`QWEN35_ADAPTIVE_BF16_SPLITK_T8=0`; widen only after new alignment, numerical,
and paired performance evidence.

### Combined P4+BF16 T8 whole-decode boundary (2026-09-01)

`bin/qwen35_adaptive_t8_decode_probe.cr` measures both T8 loaders through the
complete Qwen3.8 decode step without reloading weights between sides. It creates
two identically prefilled adaptive states, performs one unmeasured matched warm
step, then advances both states over the same ten-token forced greedy
trajectory. Every pair receives the same input token and position. Execution
order alternates AB/BA; `--candidate-first` also reverses warmup and the first
measured pair. Any top-1, logit, cache-length, owner, or Float32-owner mismatch
fails before a performance result is admitted.

The tracked release probe is fail-closed for non-finite logits and invalid token
IDs. It clears and restores every adaptive decode-route override, fixes split-K
to `1`, minimum context to `256`, and chunk size to `64`, and proves the exact
branch preconditions over the live candidate buffers before timing. The tested
map reported twelve P4 T8 owners, four aligned BF16 T8 owners, and no Float32 KV
owners on the exact Apple M2 Max route. This avoids a hot-path counter that
would perturb the corridor while still closing the deterministic branch.

At 1,675 prompt tokens, the final release run measured means
`74.955 -> 78.687 ms` (`-4.979%`, `5/10`) and essentially equal medians
`68.582 -> 68.646 ms`. This failed the predeclared `>=3%` and `>=8/10` gate.
The loader kernels remain correct there, but their work is too small relative
to the 48 recurrent layers, weight projections, and output head for a material
corridor claim.

At 8,327 tokens, two fresh guarded release processes with opposite initial
order preserved the same ten output IDs and exact observed top-1 logits. The
baseline-first run measured means `82.858 -> 81.100 ms` (`2.121%`, `8/10`) and
medians `76.728 -> 73.756 ms` (`3.87%`); it failed the predeclared mean gate.
The candidate-first run measured means `88.702 -> 75.911 ms` (`14.420%`,
`10/10`) and medians `79.726 -> 75.386 ms` (`5.44%`); it passed. Both runs used
the 24 GiB process-tree cap and 35% free-memory floor without quiet-host
waiting. The repeated positive median signal is useful diagnostic evidence,
but one pass in two processes is not a promotion-grade speed certificate.

**Adversary:** The product evidence is still one Apple M2 Max, one 27B model,
one repeated source prompt, ten matched positions per process, and the current
coarse adaptive map. Individual wall samples contain scheduler outliers; the
opposite-order repetition and median agreement narrow but do not remove host
noise, and the predeclared mean gate was unstable. The short-context failure
also proves that isolated 11--26% attention kernel gains must not be presented
as universal token-throughput gains. The state checks cover matched top-1
trajectory, finite logits, adaptive ownership, and published cache length; they
do not prove bytewise KV or recurrent-state identity.

**Value proxy:** Isolated split-K wall/GPU time explains the mechanism. Matched
full-token wall with exact trajectory, logit, owner, and cache publication is
the product boundary. Context length is part of the certificate.

**LTP/WBA:** Not claimed. This is ordinary exact-layout load specialization
measured through a stateful full-model decode corridor.

**decision:** Keep both exact-M2-Max T8 defaults and their independent rollback
flags on their existing isolated-kernel evidence. Report no material
`forward_top1` corridor win near 1.7K and only a provisional `3.9--5.4%` median
signal near 8.3K; do not promote a whole-generation speed claim. Re-run this
same-process probe across more prompts whenever decode scheduling, recurrent
kernels, adaptive tier maps, or model/device identity changes.

### Paired K/V pack encoder refutation (2026-09-01)

A temporary route encoded the independent K and V adaptive-pack dispatches in
one ordinary Metal compute encoder instead of opening two encoders. It changed
neither kernels, payload bytes, cache publication, nor command-buffer count.
The protected resident-QBit set passed `19/19`; its numerical row reported
cosine `1.0` and maximum absolute delta `3.72529e-8`.

The same-process probe prewarmed both routes, restored identical cache state,
and alternated order over ten pairs at an 8,320-token prefix. Uniform P4
regressed from `1.826` to `1.943 ms` mean wall time (`-6.432%`, `6/10` wins)
and from `1.228` to `1.264 ms` GPU time. Uniform BF16 improved wall time only
from `2.253` to `2.207 ms` (`2.025%`, `7/10`) and therefore missed the declared
`>=3%`, `>=8/10` promotion gate. Its GPU mean improved `3.374%`, but only
`4/10` individual GPU pairs won, confirming high measurement variance rather
than a stable product boundary.

**Adversary:** Fewer encoder creations are only mechanism evidence. They do not
guarantee lower command latency, and the measured setup cost is too small and
noisy relative to pack, attention, finalization, and host scheduling.

**decision:** Reject and remove the paired-encoder policy and route. Do not
retry this submission-only optimization unless command-encoding attribution
shows a materially larger host bottleneck or the dispatches can be fused into
one kernel.

### Resident batched verifier-head append result (2026-09-01)

The experiment removed one real submit/wait boundary from the existing
resident verifier, but it did not materially reduce product wall time. Ten
interleaved Qwen3.8-27B pairs preserved every top-1 ID and logit, adaptive cache
length, and the next decoded token. The appended route reduced verifier-head
synchronizations from one to zero, yet averaged only `0.222%` faster and won
`6/10` pairs. This fails the predeclared `>=3%` and `>=8/10` promotion gate.

**retained:** a caller-owned batched RMSNorm/Q6_K top-1 encoder. It leaves the
command uncommitted, validates input, weight, and output extents, and requires
the per-flight scratch Arena explicitly in its API. A direct Metal contract test
matches the existing separate-command path row for row. A second test encodes
two equal-shape commands on distinct queues before either completes, proves
their Arena scratch handles are disjoint, and matches both results.

**removed:** the CPU product flag and adaptive verifier wiring. A synchronization
count is mechanism evidence, not the performance objective; keeping the extra
policy and route after the wall-clock falsifier would add complexity without
user-visible value.

Invalid shape, buffer, environment, and committed-command requests fail before
encoding or allocating Arena scratch.

**next boundary:** a truly deferred verifier requires private speculative state
and fail-closed publication before wait. Do not approximate that boundary with
a thread wrapper, wider CogniGraph flight depth, early cache publication, or
adaptive token batching. The next experiment must first prove that cancellation
or failure leaves canonical recurrent and KV state unchanged.

**LTP/WBA:** not claimed. This was ordinary command-buffer fusion with an exact
separate-command comparison.

### Explicit vector threadgroup stores are not a speed path

A temporary compile-time split-K variant replaced each four scalar threadgroup
writes in the T8 loader with one explicit `float4` store. Both pipelines were
prewarmed, every sample restored the same cache prefix, and ten pairs alternated
order in one process at prefix 8,320.

P4 regressed from `1.624` to `1.645 ms` wall (`-1.302%`, `2/10` wins) and
from `1.030` to `1.041 ms` GPU (`-1.096%`, `5/10`). BF16 changed from
`1.969` to `1.955 ms` wall (`+0.752%`, `4/10`) while GPU regressed from
`1.315` to `1.317 ms` (`-0.165%`, `4/10`). Both missed the unchanged
`>=3%`, `>=8/10` wall-and-GPU gate under the 35% free-memory floor and 4 GiB
tree cap. The candidate was removed.

The source spelling of a vector write is not evidence of fewer generated
instructions. The Metal compiler may already coalesce the scalar form, and the
explicit pointer cast can instead constrain scheduling. Keep the existing
scalar-source T8 stores and require compiler-level evidence before revisiting
this store-shape experiment. The next candidate must remove work or bytes that
remain visible after compiler optimization.

### P4 T8 admits the existing fused stage-two reducer at long context (2026-09-02)

A two-SIMD-group reducer was tested as an alternative to the existing legacy
and fused stage-two kernels. It improved over legacy at the 8K product shape,
but it did not beat the already-shipped fused reducer: the matched chunk-512
screen measured `70.074 -> 70.334 ms` (`-0.371%`, `3/10` wins). The SG2 runtime
route was therefore removed. Its useful result is negative evidence that more
parallel lanes do not offset the extra barrier and shared state here.

An initial product-map screen changed the P4 and BF16 T8 loaders together with
P4 stage two. Its pooled `3.954%` result was therefore rejected as confounded
promotion evidence. The corrected `--compare-stage2 --resident-map p4` probe
held T8 enabled in both branches and used 16 uniform-P4 owners so the global
legacy override could not also change BF16. Baseline forced legacy stage two;
candidate used the automatic fused policy. Two fresh, opposite-order processes
used the same 8,305-token prompt, chunk 512, one append group, a 100 ms
compositor cooldown, pooled scratch, and the GC guard. They measured
`75.769 -> 71.600 ms` (`+5.502%`, `10/10`) and `84.538 -> 80.477 ms`
(`+4.803%`, `9/10`). Pooled means improved from `80.153` to `76.039 ms`, or
`5.134%` over 20 isolated pairs with `19/20` wins. Every matched step preserved
its top-1 token and observed logit. These are complete `forward_top1` wall
times, not an isolated stage-two or whole-session percentage.

The quality probe now supports a repeated prompt with a recorded SHA-256 and
reads tokenizer metadata without mapping the model tensors a second time. On
prompt hash `927649b8afec3a46f328274f88462da89c00e47762ae00b0125b95b3e6226e88`
(`11,495` chat tokens), fresh chunk-512 and chunk-1024 runs produced identical
64-token F32 trajectories and identical teacher-forced adaptive metrics:
`59/64` top-1, `107/126` ranked top-2, the exact token covered by adaptive top-2
at `63/63` positions, and output-embedding ECS mean `0.938380`. Both runs kept
all 16 adaptive owners, no Float32 owner, consistent cache publication, and a
`3.7647x` logical cache ratio. Free adaptive generation diverged after four
tokens, but both continuations selected KV quantization and gave a coherent
safety explanation. Neither reached EOS in the 64-token window, so this is a
bounded meaning check rather than a general semantic-equivalence certificate.

Chunk 1024 was faster on this same prompt: F32/adaptive/teacher-forced prefill
was `141.407/157.588/157.506 s`, versus `191.461/177.939/177.739 s` at chunk
512. Chunk 512 remains a guarded watchdog profile, not a throughput claim.

Automatic P4 fused stage two is now admitted only on exact `Apple M2 Max`, for
a uniform P4 row plan, when the live prefix is at least `6,144` and the P4 T8
loader is actually enabled. This reuses the existing T8 live-prefix boundary;
capacity alone cannot admit the route, and `QWEN35_ADAPTIVE_P4_SPLITK_T8=0`
also suppresses automatic P4 fusion. Existing automatic BF16 behavior is
unchanged. `QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED=0` remains the exact legacy
rollback for all tiers, while `=1` is an explicit experimental force switch.

**Adversary:** The positive isolated timing evidence covers one M2 Max, one
Qwen3.8-27B quant, a uniform-P4 timing map, one repeated prompt family, and 20
decode positions. The mixed production-map quality run is a separate
certificate, not timing attribution. The older isolated P4 result was negative
without this T8 corridor, so the policy must not widen below 6,144 tokens, to
non-T8 P4, or to another device. The repeated prompt is intentionally a
long-context stressor rather than a natural coding session, and the free
answers did not reach EOS. Re-run the matched timing and quality gates after
changes to T8, stage-one summaries, the fused reducer, cache layout,
compiler/runtime, device, or model.

**LTP/WBA:** Not claimed. This is ordinary context-gated kernel selection with
the legacy reducer as a fail-safe runtime frame.

### Split-K chunk alignment does not improve complete-token decode (2026-09-02)

The cache-only probe now supports an explicit same-process split-K chunk A/B.
It prewarms both pipelines, restores the same snapshot before every sample,
alternates order over ten pairs, compares the complete adaptive command's wall
and GPU intervals, and rejects any output delta above `1e-4`.

For uniform P4 at prefix 8,192, chunk `64 -> 60` aligned each block with four
tile-15 passes and reproduced a local win twice: wall improved by `3.937%` and
`4.218%`, GPU interval by `5.581%` and `5.232%`, with `10/10` wins in both
runs. The same setting regressed at prefix 6,144 and 4,096. BF16 at 8,256 was
inconsistent: one row passed at `3.238%` wall and `4.524%` GPU, while its repeat
fell to `1.259%` and `1.987%`.

That local P4 result did not survive the complete model corridor. Four guarded
real-prompt attempts using prefill chunks 128, 256, 512, and 1,024 all stopped
at the unchanged native 120-second Metal command watchdog before a decode A/B
could be collected. The probe therefore gained `--synthetic-prefix`: it
allocates the real 27B state, restores canonical zero-valued adaptive payloads
through the ordinary strict snapshot validator, sets every layer position, and
then runs matched `forward_top1` steps. It exercises real model weights,
recurrent state, adaptive attention, append, publication, logits, and top-1,
but it is explicitly not semantic-quality evidence for a natural prompt.
The probe records that boundary as `semantic_quality_valid=false`, keeps
baseline and candidate state/buffer ownership independent, advances the
caller-owned layer positions after every successful decode, and verifies both
position/cache-length agreement and the absence of pending cache reservations.

At 8,256 live synthetic tokens, the intended tier-selective P4 `60` / BF16
`64` route measured `-0.062%` and `+0.035%` in opposite-order processes, both
with `6/10` candidate wins. A current-source explicit all-tier `64 -> 60`
control measured `+1.256%` with `5/10` wins and `+0.119%` with `7/10` wins.
All four runs preserved every top-1 token, published cache length, sole adaptive
ownership, and the `1e-4` logit bound. None met the `>=3%`, `>=8/10` gate.

**Adversary:** Kernel block alignment and isolated adaptive-command time are
mechanism coordinates, not token latency. Twelve adaptive P4 layers are only
part of sixteen attention owners and sixty-four model layers; the complete
token is dominated by work untouched by this knob. Synthetic zero state also
cannot certify natural-prompt quality, although Metal execution cost depends on
the same shape, tier, and live-prefix layout.

**decision:** Keep the production default at split-K chunk 64 and remove the
temporary automatic tier-specific policy. Retain both probes as reproducible
diagnostics. Do not retry nearby chunk sizes without a new whole-token ceiling;
the next candidate must remove work or bytes across a larger decode boundary.

### Concurrent full-attention Q/K/V fan-out is not a material prefill win (2026-09-04)

An opt-in M2 Max experiment converted the normalized activation to the existing
shared H16 staging buffer, inserted the required buffer barrier, and then issued
the three independent Q/K/V GEMMs through a concurrent Metal compute pass. The
serial production path and candidate produced bitwise-identical terminal logits:
ordered top-2 matched, cosine was `1.0`, and maximum absolute error was `0.0`.

A guarded same-process ABBA falsifier used Qwen3.8-27B Q4_K_M, F16 KV, one
warmup pair, four measured pairs, the unchanged 35% free-memory floor, and a
24 GiB process-tree cap. Candidate/baseline throughput was `139.84/138.86`
tok/s at pp256 and `139.11/138.52` tok/s at pp2048. Paired-median gains were
only `+0.35%` and `+0.52%`, far below the `3%` promotion threshold.

**Adversary:** A concurrent encoder does not create additional arithmetic
capacity. Each large quantized GEMM already occupies the GPU, so overlapping
three sibling projections mostly changes scheduling rather than removing work
or memory traffic. The exact positive timing percentages are host-sensitive;
the robust conclusion is only that neither boundary showed a material gain.

**decision:** Remove the experimental route, env policy, test, and probe. Do not
retry Q/K/V fan-out concurrency without a new mechanism that reduces GEMM work
or bytes. Continue with a dominant matmul/kernel dataflow candidate rather than
more dispatch-only overlap.

### Dual-band SG8 Q4 prefill tile (2026-09-04)

The accepted Q4_K H16 kernel keeps the established eight-SIMD-group launch but
lets each group accumulate two independent 16-row bands. A dequantized 64x32
weight tile is therefore reused across 128 activation rows. Compared with the
B64 route, the threadgroup count is halved while threadgroup storage rises from
16 to 24 KiB and each lane retains a second accumulator band.

Automatic admission is deliberately exact: Apple M2 Max, pp256 or larger,
complete 128-row tiles, input width 5,120, and an observed Qwen3.8 output width
of 1,024, 6,144, 10,240, 12,288, or 17,408. `QWEN35_Q4K_H16_B128_SG8=0`
restores B64; `=1` permits wider exact-tile experiments.

A guarded same-process production-auto/rollback ABBA on Qwen3.8-27B Q4_K_M
measured mean throughput gains of `+4.70/+2.64/+5.81/+5.34%` at
pp256/512/1024/2048; paired medians were `+5.11/+3.87/+4.00/+6.39%`.
Every full terminal-logit vector was bitwise equal, ordered top-2 matched, and
the automatic route recorded 173 regular plus 63 fused SwiGLU calls per pass.

The mathematically compatible downstream widening was separately falsified.
Explicit admission added exactly 32 calls, matching the model's 32 Q4
`17408 -> 5120` recurrent FFN-down tensors, but measured `-0.44%` at pp256
and only `+1.78%` at pp2048 versus the accepted auto route. The default-off
H16 output-projection corridor was not active and is not attributed to this
delta. Wider exact shapes remain experiment-only.

Current strict split-process screens against llama.cpp `7e4c0a968` found native
ahead at every pp256-2048 point both with the entire SG8 route forced and with
it disabled. That establishes a bounded directional lead under the matched
token/KV/terminal-logit contract, not a stable percentage attributable to this
kernel: the separate processes showed substantial host drift. The active
llama.cpp M2 Max route is its 64x32, four-SIMD-group fallback; its Metal tensor
path is rejected by the current pre-M5 device gate even though the runtime
reports Metal 4 support.

**Adversary:** Weight-tile reuse does not guarantee a win when the second
accumulator band reduces occupancy. The direct downstream failure is the
counterexample. This certificate is limited to one M2 Max, model quantization,
and measured shape set. Re-run the ABBA after kernel arithmetic, H16 staging,
projection geometry, Metal compiler/runtime, or device changes.

**decision:** Keep the exact producer-side auto gate and immediate rollback.
Do not generalize by output width, and do not report threadgroup-count reduction
as the speedup. Continue at recurrent FFN-down/projection with a new
layout or fusion that removes different work.

### Transposed Q4 FFN-down tile is exact but not faster (2026-09-04)

The recurrent Q4 `17408 -> 5120` FFN-down corridor was tested with the reuse
direction reversed: one 64-row H16 activation tile was shared across two
64-output weight bands. The exact-shape kernel used eight SIMD groups, 256
threads, and 24 KiB of threadgroup memory. It reduced staged activation bytes
per paired weight tile, while retaining two independent output accumulator
bands.

A guarded same-process candidate/rollback ABBA used Qwen3.8-27B Q4_K_M, F16
KV, two measured pairs, the unchanged 35% free-memory floor, and a 24 GiB
process-tree cap. The route executed exactly 32 times per pass and produced
bitwise-identical full terminal logits: ordered top-2 matched, cosine was
`1.0`, and maximum absolute error was `0.0`. Candidate/rollback throughput was
`171.73/173.16 tok/s` at pp256 (`-0.83%`) and `140.29/139.18 tok/s` at pp2048
(`+0.79%`, paired median `+1.40%`).

**Adversary:** Reusing the input tile is not free. The second weight band adds
matrix fragments and accumulator state, which can lower occupancy or increase
register pressure. The boundary results are well below the 3% promotion gate;
the exact positive pp2048 percentage is not stable speed evidence.

**decision:** Remove the kernel, policy, route, and spec. Do not retry the same
output-wide tile without a mechanism that demonstrably reduces register state
or changes the arithmetic/dataflow ceiling.

### Tensor Q4 layout repair does not justify M2 admission (2026-09-04)

The experimental `simd_mm_q4k_tensor_f32out` kernel described its cooperative
tensor as `(K tile, output rows)` but populated threadgroup memory in the
opposite order. Matching llama.cpp's `row * K_tile + k` layout repaired the
route: a Qwen3.8-27B full-terminal-logit comparison changed from a top-2
failure to bitwise equality at every measured prompt size.

The corrected route still lost to the accepted SG8-B128 kernel on M2 Max.
Candidate/baseline throughput deltas were `-4.82%`, `+0.83%`, `-3.63%`, and
`-6.29%` at pp256/512/1024/2048. The pp512 result is noise-sized and does not
offset the three regressions. Current llama.cpp likewise keeps tensor ops out
of its pre-M5 default route.

**Adversary:** Successful compilation and exact output do not establish that
the hardware executes this tensor formulation efficiently. The comparison is
limited to one M2 Max and the Qwen3.8 Q4_K_M gate/up corridor; it says nothing
about newer Apple GPUs where llama.cpp admits the tensor path.

**decision:** Keep the layout correction and the existing explicit opt-in, but
do not enable Tensor Q4 automatically on M2 Max. Continue from the faster
SG8-B128 baseline and require a new dataflow mechanism before retesting.

### Sixteen-SIMD-group B256 Q4 reuse is exact but slower (2026-09-04)

A temporary recurrent FFN-down experiment reused each dequantized Q4 weight
tile across 256 activation rows. The kernel launched 16 SIMD groups (512
threads), split into two concurrent B128 accumulator sets, and used 24 KiB of
threadgroup memory. Admission was default-off and restricted to Apple M2 Max,
complete 256-row batches, and the exact `17408 -> 5120` Q4 shape.

The device accepted the 512-thread launch (`maxTotalThreadsPerThreadgroup=704`).
A bounded synthetic comparison produced bitwise-identical output for all
16,384 F32 elements. A guarded full-model candidate/rollback comparison then
recorded exactly 24 direct recurrent Q4 FFN-down route hits versus zero in the
rollback, with bitwise-identical terminal logits, ordered top-2, and cosine
`1.0`. Throughput was `167.72/175.10 tok/s` at pp256 (`-4.22%`) and
`164.74/166.64 tok/s` at pp2048 (`-1.14%`).

**Adversary:** The candidate halves weight-tile loads for its admitted rows,
but doubles the active thread and accumulator front. The measured regression at
both boundaries shows that theoretical bandwidth reduction is not the product
objective; occupancy, register pressure, and full-threadgroup barriers erase
the saving on this M2 Max route. The 24-hit scope excludes the separate fused
down-add path and must not be misreported as every recurrent FFN-down call.

**decision:** Remove the kernel, policy, route, spec, and probe. Do not retry
this 512-thread geometry. A future B256 experiment must retain the established
256-thread occupancy, for example by processing two B128 bands serially, and
must beat the current SG8-B128 wall-time boundary rather than only reduce
modelled bytes.

### Runtime P4 nibble words are exact but below the promotion gate (2026-09-20)

The canonical adaptive P4 base stores four 32-byte bitplanes after the 8-byte
row header. A runtime-only alternative was tested at the same 136 bytes per
row: 32 little-endian `UInt32` words, each holding eight adjacent four-bit
prefixes with the first value in the least-significant nibble. The canonical
snapshot layout remains unchanged. An exact CPU codec proves both directions,
all 16 nibble values, row/group boundaries, non-aliasing, and byte-identical
mixed P4/P5/BF16/F32 artifact reconstruction.

A model-free Metal falsifier assigned one 32-thread SIMD-group to each P4 row.
Both arms read identical headers, reconstructed all 256 values with the same
centroid arithmetic, reduced the same Float32 row checksum, and wrote one
output. The baseline loaded one byte from each of four canonical planes; the
candidate loaded one packed `UInt32`. Completed-command GPU intervals used ten
ABBA/BAAB pairs after three balanced warmups at 4 KV heads:

- 6,144 tokens / 24,576 rows: `0.141292 -> 0.137667 ms`, `+2.63%`, 10/10 wins;
- 8,192 tokens / 32,768 rows: `0.181917 -> 0.177792 ms`, `+2.32%`, 10/10 wins;
- 16,384 tokens / 65,536 rows: `0.347500 -> 0.339042 ms`, `+2.49%`, 10/10 wins.

Every row checksum was bitwise identical with zero mismatch and zero maximum
absolute error. The guarded process began with 74% free system memory, kept a
30% runtime floor and 1 GiB process-tree cap, used at most 18,350,080 bytes of
probe GPU buffers, and exited zero. None of the three boundaries met the
predeclared `>=3%` median gate, despite stable directionality.

**Adversary:** This microbenchmark is favorable to the candidate: it isolates
the loader and excludes attention score/value work, softmax, pack conversion,
snapshot inversion, and the rest of a token. Therefore a sub-3% isolated gain
cannot support a 3% whole-kernel or whole-token claim. The stable 10/10 win is
evidence that the layout reduces a small local cost, not that changing the
resident cache representation is worthwhile.

**decision:** Keep the canonical plane-major resident and snapshot format; do
not wire the nibble-word layout into Metal on its own. Retain the exact CPU
codec and model-free probe as oracles. Reopen only if a broader transformation
also removes metadata, dequantization, or tile traffic and independently clears
the 3% full adaptive-attention gate. Evidence:
`/private/tmp/qwen_qbit_p4_runtime_layout_abba.log`, SHA-256
`43047ef7cc123c6bd15593e00accf918e19114c274ff7fbc45ffb751a3684979`.
Probe source SHA-256:
`452cda6e90b741b64a8f9cdb19e3b8233896f4179e6b8986a5680a14d7b242bd`.
Refresh on layout/loader, compiler/toolchain, device, timing method, gate, or
loss of the recorded evidence.

### P4 direct-QK is exact and useful, but remains opt-in (2026-09-20)

The P4 T8 split-K stage1 path was given a second execution geometry. Six SIMD
groups own K rows in strides of six; every lane dequantizes eight values,
accumulates all six GQA query heads, reduces with `simd_sum`, and writes the
scores into the existing threadgroup tile. The V path, canonical adaptive-cache
bytes, snapshot format, publication protocol, launch size, and threadgroup
allocation are unchanged. `QWEN35_ADAPTIVE_P4_SPLITK_DIRECT_QK=1` is an
explicit default-off switch and is effective only for uniform P4 with the P4
T8 loader.

Model-free resident tests cover an 8K prefix plus visible tile tails 1, 6, 15,
and 16. The direct path matches the scalar oracle, retains byte-identical K/V
payloads, and the complete 20-example resident suite passes. In the isolated
adaptive-attention command, two ten-pair runs improved completed-command wall
time by `9.56--14.44%` at 8K and `20.35--20.54%` at 16K; maximum output drift
was `4.3e-7`, with exact payload equality.

The same-process full-model `forward_top1` falsifier is the product boundary.
It loaded the model once, used independent baseline/candidate states, alternated
AB/BA order, held P4/BF16 T8 and fused stage2 constant, and toggled only
direct-QK on the 12 P4 owners. Pooled across two opposite-order ten-pair runs:

- 8K synthetic prefix: `70.003 -> 68.275 ms`, `+2.47%`, 18/20 wins;
- 16K synthetic prefix: `78.139 -> 75.828 ms`, `+2.96%`, 20/20 wins.

One run at each boundary exceeded 3%, while the opposite-order run did not;
the pooled results therefore miss the predeclared `>=3%` full-token gate. A
separate real 401-token prompt-prefill run kept top-1 identical for all ten
decode steps with maximum logit drift `7.6e-6`; it was speed-neutral/regressive
(`-0.38%`), as expected below the long-context regime. Synthetic-zero-prefix
runs establish route timing and baseline/candidate numerical agreement, not
semantic quality.

**Adversary:** Removing the shared K tile is a large local win, but attention is
only part of a token. Six live query accumulators can also increase register
pressure. Directionality is strong at long context, yet a threshold-straddling
result must not be promoted by selecting only the passing repetitions.

**decision:** Keep the exact direct-QK implementation and product A/B probe as
a default-off experimental route. Do not enable it automatically yet. Reopen
automatic admission only when a changed kernel/compiler/device or a broader
transformation produces a repeated pooled full-token gain above 3% without
weakening top-1/logit/cache checks. Evidence logs and SHA-256:
`/private/tmp/qwen_direct_qk_full_8k.log`
`654421923d56fb04e8d68146b5a841c9b93042aaf7c9a7901ad1875306480c3d`,
`/private/tmp/qwen_direct_qk_full_8k_repeat.log`
`e87da725a5a9608d4b238bc7550d55046a290bb33dc31b62335d8cbf11dc3ad5`,
`/private/tmp/qwen_direct_qk_full_16k.log`
`afa48b83c25396ed8efa9550bb3d1c15c66bf6fc369e1694da42033264542b00`,
`/private/tmp/qwen_direct_qk_full_16k_repeat.log`
`cda23650d35553241ebdc5406180aa727bd474f578e44b14ccf29034dd9f7b4c`,
and `/private/tmp/qwen_direct_qk_full_real_prefix.log`
`b7f6e32eeb27cd71619422dfcaf7ae9df2eee81e524fe0729f0d67a1be39166e`.
Refresh on kernel/policy/probe, cache map, model, compiler/toolchain, device/OS,
timing method, safety policy, gate, or evidence loss.

### Contiguous shared-V accelerates P4 stage one but remains opt-in (2026-09-20)

Historical note: this section records the pre-decoupling implementation, where
the only contiguous-V source variant also enabled direct-QK. It is retained as
the evidence lineage for that experiment; the later shared-K decoupling section
supersedes its policy requirement.

True direct-V was rejected as the first move. Six GQA query heads use different
softmax probabilities, so removing the shared V tile would require either six
dequantizations of each V row, roughly six live output accumulators per lane,
or additional global/threadgroup scratch and synchronization. Instead, the
bounded candidate keeps one shared dequantized V tile and changes only lane
ownership: each lane accumulates eight contiguous output dimensions through
two `float4` loads, then scatters them into the existing canonical partial
output layout. Cache bytes, snapshot format, launch geometry, threadgroup
allocation, barriers, fused stage two, and publication are unchanged.

At this historical source state,
`QWEN35_ADAPTIVE_P4_SPLITK_V_CONTIGUOUS=1` was fail-closed, default-off, and
effective only with uniform P4, the P4 T8 loader, and
`QWEN35_ADAPTIVE_P4_SPLITK_DIRECT_QK=1`. AIR contains real four-wide
threadgroup loads; it is not only a source-level rewrite. The 8K random-nonzero
resident oracle and visible tails 1/6/15/16 match the scalar reference with
cosine above `0.9999999`, maximum absolute error `3.73e-8`, and byte-identical
K/V payloads. The combined policy/resident suite passes `32/32` examples.

The isolated adaptive-attention A/B strongly favors contiguous V:

- 8K: wall `2.059 -> 1.714 ms` (`+16.74%`, 9/10) and completed-GPU
  `1.497 -> 1.131 ms` (`+24.43%`, 10/10);
- 16K: wall `2.737 -> 2.229 ms` (`+18.56%`, 10/10) and completed-GPU
  `2.077 -> 1.537 ms` (`+25.97%`, 10/10).

All isolated pairs preserved exact K/V bytes and reported zero output delta.
With direct-QK held on in both full-model arms, V-only pooled means were about
`+2.19%` at 8K (37/40 wins) and `+4.68%` at 16K (40/40). The more useful
product bundle compares legacy P4 T8 stage one with direct-QK plus contiguous
V. Two opposite-order 20-pair runs produced:

- 8K: `68.554 -> 65.645 ms` (`+4.24%`) and
  `69.874 -> 67.060 ms` (`+4.03%`), 40/40 wins;
- 16K: `77.285 -> 71.629 ms` (`+7.32%`) and
  `78.388 -> 72.603 ms` (`+7.38%`), 40/40 wins.

Those long-context product timings use a synthetic zero-valued prefix. They
prove shape/scheduling benefit, not semantic quality. A proposed automatic
M2-Max admission was therefore attacked with a real nonzero 6,511-token chat
prefix, two independent states, and alternating decode order. The first four
top-1 choices agreed, but sample 3 reached a top-1-logit delta of `3.2424927e-4`
and violated the predeclared `1e-4` gate. The run failed closed; the automatic
policy was removed rather than weakening the threshold after observing it.

The preserved timing logs use probe schema `v6`. The final source advances the
reporting schema to `v7` only to label the comparison as an explicit
forced-off/forced-on experiment and to reject underfilled real-prefix attempts
before model prefill; the measured kernel, routing knobs, and timing fields are
unchanged. Treat the logs as evidence for the recorded source slice, not as a
fresh run of the final reporting binary.

**Adversary:** the vector loads reduce instruction count, not shared-memory
bytes or V arithmetic, and their extra live accumulators may alter register
occupancy. The local nonzero oracle is strong for attention layout, while the
real-prefix product failure shows that small per-layer reorder error can grow
across 64 layers. Synthetic top-1 agreement cannot overrule that boundary.

**decision:** retain contiguous shared-V and the combined product probe as an
explicit experiment. Do not enable either direct-QK or contiguous V
automatically. Reopen admission only after a real-prefix long-context run
passes the existing logit/top-1 checks and repeated timing gate; top-2 and ECS
should be added before any broader quality claim. Evidence and SHA-256:

- isolated: `/private/tmp/qwen_v_contiguous_isolated.log`,
  `b3d7feba3ba63ebe38daa1d6b7da9555f18715685246d85c08d798d54bfdf755`;
- V-only 8K: `/private/tmp/qwen_v_contiguous_full_8k.log`,
  `c08d045389046480b0eb5abee7f2333e45a2981fbab3b48feb74b8b4d59c02d2`,
  and `_repeat.log`,
  `756f3067923f7fb4e43fdf8f56d91052e585a9ad998e5582f543186ac0c70ef0`;
- V-only 16K: `/private/tmp/qwen_v_contiguous_full_16k.log`,
  `3b1a5dc57cea7c4c0d31634e32761f1ce563f9745e291403d783b7673d3c1d8d`,
  and `_repeat.log`,
  `46861b5b5533b1d0c91c8691163abb2180812e1aab61acfd70d5d52d49ae384e`;
- stage-one bundle 8K:
  `/private/tmp/qwen_p4_stage1_bundle_full_8k.log`,
  `1826007160dc194d2ac68d35478e0d28e405072a69b829153b09ca91f0e454c5`,
  and `_repeat.log`,
  `142d1208edfa860f9be69e23b57e89c58ece012eeb461bdc0d412b9b59e38a6a`;
- stage-one bundle 16K:
  `/private/tmp/qwen_p4_stage1_bundle_full_16k.log`,
  `ee8b37ee747631fa3b61e096466e771aad8e6ae398760a7a142a842d115f7d9c`,
  and `_repeat.log`,
  `dbdbe9dd8fc1500be712998940fa8d4cdbc02a5939c701dbea4b0f77e2d0815d`;
- rejected real-prefix admission:
  `/private/tmp/qwen_p4_stage1_auto_real_6k5.log`,
  `7337aa3b90684a2bfa1efc8fad476f9a8cb6e84113fd9e2961a50cbb927e02dc`.

Refresh on kernel/policy/probe, cache map, model/tokenizer/template,
compiler/Metal toolchain, device/OS, timing method, threshold, or evidence loss.

### Real-prefix top-2 attribution isolates direct-QK drift (2026-09-20)

Probe schema `qwen-adaptive-t8-decode-ab-v10` adds an explicit quality-only
mode for real prompt prefixes. It creates two independent states, obtains the
prefill-boundary top two from one full-logits prefill per state, and then
self-feeds the baseline and candidate trajectories independently. The gate is
fail-closed on early EOS or an incomplete requested sample count. While inputs
remain aligned, it requires identical ranked top two plus first-logit,
second-logit, and margin deltas no greater than `1e-4`; it also requires the
entire generated common prefix. Once inputs diverge, same-input logit
comparisons are explicitly invalid rather than being compared across different
histories; the divergence itself is recorded as the failing certificate.
Boundary, warmup, and measured positions are all retained in the final JSON.

The guarded contiguous-V-only run used the real 6,511-token chat prefix and 32
measured decode samples. It completed all 34 quality positions (prefill
boundary, warmup, and 32 samples), with 68/68 ranked top-two matches, 34/34
exact top-one and top-two coverage, identical 34-token continuations, and zero
first-logit, second-logit, and margin delta at every position. The quality gate
passed. It started and ended with 73% free memory under `run_safe.sh`, a 30%
runtime floor, 24 GiB process-tree cap, and 900-second timeout. The final
quality run reported `87.971 -> 86.395 ms` (`+1.792%`, 26/32 wins), but timing admission is
deliberately invalid in this mode because exact top-two collection changes the
execution path.

Attribution against the same prompt is unusually strong. The previously
captured direct-QK-only and direct-QK-plus-contiguous-V runs have the same
per-position drift sequence: maximum first-logit delta `0.0028953552`, maximum
second-logit delta `0.0024356842`, maximum margin delta `0.0025596619`, and
respectively 19, 15, and 18 aligned positions above `1e-4`. Both still preserve
68/68 ranked top-two choices and the same generated text. Since the V-only run
has zero delta while adding V to direct-QK does not change the direct-QK drift,
the observed real-prefix numeric deviation is attributable to direct-QK in
this tested trajectory, not to contiguous-V.

These certificates have deliberate limits. The quality prefill uses the
full-logits path, not the production top1-only path. The route certificate
proves policy eligibility, not actual pipeline execution. ECS is computed from
the static `output.weight` rows and is a token-decision proxy only; it is 1.0
here by construction because every compared token ID is identical, and no
semantic task was scored. Diagnostic quality-mode timing is not promotable.

**Adversary:** one prompt, one device, and 34 positions cannot establish broad
semantic equivalence or a production speedup. At that source state, the exact V
result also did not make the combined route safe because policy still required
direct-QK,
whose strict numeric gate fails. Matching top-two ranks can hide materially
different logits, which is why the independent numeric thresholds remain part
of the gate.

**decision:** contiguous-V is ROBUST for this bounded real-prefix quality
certificate. Direct-QK and the combined stage-one bundle remain VULNERABLE for
strict numeric promotion and stay default-off. Do not auto-admit either route.
The next falsifier is to decouple contiguous-V from direct-QK, then compare
legacy shared-K plus contiguous-V against the legacy route on both real-prefix
quality and repeated synthetic long-context timing.

Evidence and SHA-256:

- final V-only v10:
  `/private/tmp/qwen_p4_v_contiguous_real_6k5_top2_ecs_v10_final.log`,
  `aaffb7f3caac11f0f2c29ab23261fc4637fae2bd8bc22f7b3688792b76f155e5`;
- direct-QK-only v9:
  `/private/tmp/qwen_p4_direct_qk_real_6k5_top2_ecs_v9.log`,
  `a023c617b3b5b2c4ba6f780cc824841e7dd080523dd090e4753b1249ff473061`;
- direct-QK plus contiguous-V v9:
  `/private/tmp/qwen_p4_stage1_real_6k5_top2_ecs_v9.log`,
  `dec20d8917db77824fda0b91416d42517e50796ccc859b5f057b41e041309d48`;
- final release probe binary:
  `/private/tmp/qwen35_adaptive_t8_decode_probe_top2_ecs_v10_final`,
  `b11460af04109dcbbeec9feeafb9927a8809da5e97566da255b2a13d597d258f`.

Refresh on probe or quality semantics, kernel/policy/cache layout, model,
tokenizer/template/prompt, compiler/Metal toolchain, device/OS, thresholds,
safety policy, or evidence loss.

### Shared-K contiguous-V decoupling clears the 16K product gate (2026-09-20)

Contiguous-V no longer requires direct-QK. The Metal kernel already separated
the two transformations: direct-QK controls K materialization and score
accumulation, while contiguous-V controls only lane ownership during V
accumulation. The policy now admits explicit contiguous-V for uniform P4 T8
with legacy shared-K, and source selection retains four distinct combinations:
legacy, direct-QK only, contiguous-V only, and direct-QK plus contiguous-V.
Cache bytes, snapshot format, threadgroup allocation, barriers, stage two, and
publication are unchanged. The feature remains fail-closed and default-off.

The resident regression matrix preserves both the new shared-K plus
contiguous-V route and the existing direct-QK plus contiguous-V route. The 8K
random-nonzero oracle and visible tails 1/6/15/16 pass for both combinations,
with byte-identical K/V payloads. The model-free isolated attention probe also
holds direct-QK off in both arms. It measured:

- 8K: wall `2.568 -> 2.178 ms` (`+15.20%`, 10/10) and completed-GPU
  `2.041 -> 1.634 ms` (`+19.91%`, 10/10);
- 16K: wall `3.369 -> 2.652 ms` (`+21.30%`, 10/10) and completed-GPU
  `2.746 -> 2.002 ms` (`+27.08%`, 10/10).

The guarded real-prefix quality run used a 6,511-token chat prefix, independent
self-fed states, and 32 measured samples. Its route certificate reported zero
direct-QK owners and 12 contiguous-V P4 owners. All 34 positions, including
the prefill boundary and warmup, retained identical ranked top two, first and
second logits, margins, token IDs, and generated text: 68/68 ranked top-two
matches, 34/34 exact top-one/top-two coverage, and zero numeric delta. The
quality gate passed. Timing from this full-top-two path is diagnostic only.

Two opposite-order 20-pair full-token runs then separated the long-context
speed boundary:

- 8K: `95.161 -> 93.151 ms` (`+2.11%`, 17/20) and
  `83.469 -> 81.350 ms` (`+2.54%`, 15/20). Both preserve identical output but
  fail the declared 3% and 16/20-wins product gate.
- 16K: `88.540 -> 83.897 ms` (`+5.24%`, 20/20) and
  `85.870 -> 82.221 ms` (`+4.25%`, 20/20). Both pass the product timing gate
  with identical output.

**Adversary:** the 16K timing states use a synthetic zero-valued prefix, the
real-prefix quality certificate covers one prompt and 34 positions, and the
route certificate proves policy eligibility rather than executed-pipeline
identity. The extra contiguous accumulators can also change register pressure
on other devices. Positive isolated timing cannot promote the 8K product row,
which missed the predeclared gate in both orders.

**decision:** the decoupled shared-K plus contiguous-V route is ROBUST for the
bounded correctness and 16K performance certificates on Apple M2 Max. Keep it
explicit and default-off while locating a stable automatic crossover; do not
enable it for 8K from these rows. The next falsifier is a repeated full-token
context sweep around the crossover, followed by a matched production-boundary
quality check before any automatic policy change. Evidence and SHA-256:

- isolated attention:
  `/private/tmp/qwen_p4_v_contiguous_legacy_qk_isolated.log`,
  `b96494776016fd72bad20be26e255a3e95a985a3f36232f4d7d5b393701de629`;
- real-prefix quality:
  `/private/tmp/qwen_p4_v_contiguous_legacy_qk_real_6k5_top2.log`,
  `64e353b639f41875caa869c7aec0e6941c814a3715afa90e04f089cae79dbbd8`;
- 8K full-token AB/BA:
  `/private/tmp/qwen_p4_v_contiguous_legacy_qk_fulltoken_8k_ab.log`,
  `5eb4977f96a702b7981eb72681852489874a00d1e314cc4bd2e53cf0aaeab0b8`,
  and `/private/tmp/qwen_p4_v_contiguous_legacy_qk_fulltoken_8k_ba.log`,
  `405396db833a2c5312de3a85efec98a81567e00ba1c4d21b62c39c87f2ac8fcb`;
- 16K full-token AB/BA:
  `/private/tmp/qwen_p4_v_contiguous_legacy_qk_fulltoken_16k_ab.log`,
  `f3a4c9df84e68cce4f4c499524322c1b8b7554e45bc04c11c3194271c31b2966`,
  and `/private/tmp/qwen_p4_v_contiguous_legacy_qk_fulltoken_16k_ba.log`,
  `a0383d50c43d6a01badd680f74e92b89d7b9ca88dae6bb93edce607d8b274c54`;
- release probe binary:
  `/private/tmp/qwen35_adaptive_t8_decode_probe_v_only_decoupled`,
  `422ee26211f67109798355edf9206e657a17f6f03c146d3ef4e73c0ab447799c`.

Refresh on kernel/source selection, policy/probe/cache layout, model or prompt,
compiler/Metal toolchain, device/OS, timing method, thresholds, safety policy,
or evidence loss.

### Contiguous-V crossover and production-prefill certificate (2026-09-20)

The repeated full-token crossover sweep kept direct-QK disabled and compared
legacy shared-K against shared-K plus contiguous-V with the same uniform P4 T8
route. Each row used 20 alternating-order pairs and retained identical output:

- 10K (`10,239` seeded tokens): `+2.25%` with 17/20 wins and `+2.69%` with
  18/20 wins; both fail the declared 3% product gate;
- 12K (`12,287` seeded tokens): `+3.36%` with 20/20 wins and `+3.02%` with
  19/20 wins; both pass, but the second row is only 0.02 percentage points
  above the threshold;
- 14K (`14,335` seeded tokens): `+3.80%` with 20/20 wins and `+3.84%` with
  19/20 wins; both pass with more separation from the threshold.

The first real 14K quality attempt used a 512-token prefill chunk and failed
closed with Metal `Impacting Interactivity` before comparison. Changing only
the probe's effective prefill chunk to 256 completed the same 14,311-token
prefix. The full-logits diagnostic retained 68/68 ranked top-two matches,
34/34 exact top-one/top-two positions, zero first/second-logit and margin
deltas, identical generated text, and ECS 1.0.

That full-logits path was not the production prefill boundary, so the probe now
also provides `--quality-top2-production-prefill`. This mode obtains the
boundary token from `prefill_tokens_top1`, records that top two is unavailable
there, fails the quality gate on a boundary ID or logit mismatch, and uses
`forward_top2` only for the subsequent independent warmup and free-run steps.
It does not change the inference engine or any Metal kernel.

The guarded production-boundary run at 14,311 prompt tokens reported 12
contiguous-V P4 owners and zero direct-QK owners. Baseline and candidate
production-prefill top one were both token `332` at logit `22.17311`, with
zero delta. The following 33 top-two positions retained 66/66 ranked matches,
33/33 exact top-one/top-two coverage, zero second-logit and margin deltas,
identical 34-token free runs, and ECS 1.0. The quality gate passed. Timing from
this full-top-two run remains diagnostic and is explicitly invalid for product
admission.

**Adversary:** one prompt and one device do not prove general semantic
equivalence; the 12K speed row barely clears the threshold; the 14K timing
prefix (`14,335`) and real quality prefix (`14,311`) do not identify a unique
automatic integer cutoff; and the first 512-token-chunk run hit the watchdog.
The probe certificate separates boundary provenance honestly, but it cannot
manufacture a boundary top two that the production API does not return.

**decision:** the shared-K contiguous-V route is ROBUST for this bounded 14K
production-prefill quality certificate and the repeated 14K synthetic timing
certificate on Apple M2 Max. Keep automatic admission unchanged in this slice.
Choose and falsify an explicit conservative cutoff in a separate policy change
with its own unset/force/kill-switch tests and an auto-mode route certificate.

Evidence and SHA-256:

- crossover 10K AB/BA:
  `/private/tmp/qwen_p4_v_contiguous_crossover_10k_ab.log`,
  `6e62c990aa4005212dbe558064a19e9b18103fd151bd015266894a57b3229340`,
  and `/private/tmp/qwen_p4_v_contiguous_crossover_10k_ba.log`,
  `246df003992e4d8ddb3c2b5fa5c4f785f7f3e1fee01893c83619065e9c1f9aec`;
- crossover 12K AB/BA:
  `/private/tmp/qwen_p4_v_contiguous_crossover_12k_ab.log`,
  `bab19aab8db5536f895961aff74b74ffd2392c7f6d6316204f5775bd31817914`,
  and `/private/tmp/qwen_p4_v_contiguous_crossover_12k_ba.log`,
  `93271b8c6f54175c7166f89c7297112f81501b2ea3144103b938c7b004501cb8`;
- crossover 14K AB/BA:
  `/private/tmp/qwen_p4_v_contiguous_crossover_14k_ab.log`,
  `d67ed786cb39d3d0aeed0853321b6bedeee5bc13aa6258c1c630741c9b0534ef`,
  and `/private/tmp/qwen_p4_v_contiguous_crossover_14k_ba.log`,
  `84928459ddb67fce9e033662e3fe546546e38ace0beaa6011ded6ab00c052270`;
- failed 512-chunk real 14K attempt:
  `/private/tmp/qwen_p4_v_contiguous_real_14k_top2.log`,
  `fc81a2e60e72cead688d9d8cdb3bcd8d7c6916c7798f32675b1076cc3f5f929c`;
- completed 256-chunk full-logits diagnostic:
  `/private/tmp/qwen_p4_v_contiguous_real_14k_top2_chunk256.log`,
  `ec0d7d8de375c6b7c17a4a9acf9b97b4a02af2d04aa4cf3ac9143ad234fc4919`;
- production-prefill smoke and 14K quality runs:
  `/private/tmp/qwen_p4_v_contiguous_prod_prefill_smoke_400_v11.log`,
  `4604205b8658be8d321fa6c8cb47688df0bed38cd8ee80609d1b60fc97e0ad19`,
  and `/private/tmp/qwen_p4_v_contiguous_prod_prefill_14k_top2.log`,
  `7bb7ee976a26d93706b8b1ec382a5c9be7bce0698ed1ee7ff19128bf18cb3dbb`.

Refresh on probe or prefill-boundary semantics, kernel/source selection, policy,
cache layout, model/prompt, compiler/Metal toolchain, device/OS, timing method,
thresholds, safety policy, or evidence loss.

### Automatic contiguous-V admission at the 14K boundary (2026-09-20)

The policy now admits contiguous shared-V accumulation automatically only when
all of the following hold:

- the device name is exactly `Apple M2 Max`;
- the adaptive attention tier is uniformly P4;
- the P4 T8 split-K route is active; and
- the visible context, `packed_len + 1`, is at least `14,336` tokens.

The boundary is deliberately above the barely passing 12K crossover rows and
matches the repeatedly passing 14K timing point. The addition uses `Int64`
arithmetic for the visible-token check and rejects a negative packed length.
`QWEN35_ADAPTIVE_P4_SPLITK_V_CONTIGUOUS=0` is the immediate rollback;
`=1` remains an explicit experiment override, but cannot bypass the uniform-P4
or P4-T8 prerequisites. Malformed or blank overrides fail closed.

Probe schema `qwen-adaptive-t8-decode-ab-v12` adds
`--compare-v-contiguous-auto`. Its baseline explicitly sets the option to zero,
while its candidate leaves the option unset. At the
first admitted point (`14,335` packed, `14,336` visible), the actual pipeline
trace selected 132 legacy P4 T8 stage-one dispatches for the baseline, 132
`p4_t8_v_contiguous` dispatches for the automatic candidate, and 88 BF16 T8
dispatches shared across both states. The route certificate reported all 12 P4
owners on contiguous-V, all four BF16 owners on the BF16 route, and zero
direct-QK owners. Ten alternating full-token pairs retained identical output
and zero logit delta; the candidate won 10/10 with a diagnostic `+4.150%` mean
improvement. This synthetic row proves automatic route selection and bounded
timing only; it is explicitly non-semantic and is not a new product-speed
certificate by itself.

The semantic certificate composes with the immediately preceding production
prefill result: the automatic policy selects the same unchanged contiguous-V
kernel whose explicit-on route retained the production boundary token/logit,
66/66 subsequent ranked top-two choices, all numeric values, the 34-token free
trajectory, and ECS 1.0 at a 14,311-token real prompt. Unit tests separately
pin the exact off/on boundary, device restriction, base prerequisites,
kill-switch, force override, overflow-safe large length, and malformed or
negative inputs.

Three fresh real-prefill attempts do not strengthen that semantic composition.
One three-state run reached its final resident prefill and then failed closed
after about 574 seconds; two two-state automatic-mode attempts, with prefill
chunks 256 and 128, failed during the first baseline prefill after about 158
and 163 seconds. All three failures were Metal `Impacting Interactivity` before
the automatic decode route was selected. Therefore they are evidence that the
current long-prefill validation workload remains watchdog-sensitive, not
evidence against contiguous-V or its admission predicate. Reducing the chunk
from 256 to 128 did not separate the failure, so that simple duration-only
hypothesis is refuted for this host state.

**Adversary:** the automatic route has direct actual-pipeline evidence at the
boundary but not a fresh completed real-prefill auto-vs-off run. The semantic
argument depends on the unchanged-kernel bridge to the explicit-on production
certificate. Exact device-name admission excludes other Apple GPUs by design;
one model, one prompt, and one host cannot establish a general policy. The
single 10-pair automatic timing row cannot replace the repeated AB/BA crossover
matrix. The current executable has no default metallib, so a session that
crosses the boundary after already using the legacy P4 pipeline can pay one
lazy source compilation for the contiguous-V variant. A follow-up fresh-process
API profile with the host system shader cache retained measured that exact lazy
variant at `0.486 ms` for source-library creation, `0.006 ms` for function
lookup, and `0.227 ms` for pipeline-state creation: `0.719 ms` total, versus a
subsequent candidate decode mean of `74.729 ms/token`. This rules out a material
14K threshold cliff in the measured warm-system-cache state. It does not
measure a first-ever compile after source/toolchain cache invalidation.

The earlier source-addressed metallib falsifier remains decisive for the
implementation choice: across 30 used pipelines, file loading plus
pipeline-state creation cost `338.058 ms`, while the warmed source route cost
`18.278 ms`; genuinely cold source setup was `661.875 ms`. A synchronous
prewarm only relocates that cold work, and precompiling every possible variant
would tax sessions that never reach the 14K route. Keep both metallib and
per-request prewarm out of production unless a future cache-cold, all-in
request discriminator shows a net win.

**decision:** the exact-M2-Max, uniform-P4, P4-T8 admission predicate is ROBUST
within this bounded evidence composition. Enable it automatically at 14,336
visible tokens, retain explicit zero as rollback, and keep direct-QK disabled.
The claim remains VULNERABLE if widened to another device, tier mixture, model,
kernel/toolchain, or broad semantic equivalence. Long real-prefill watchdog
stability also remains open and must not be presented as closed by this change.

Additional evidence and SHA-256:

- automatic boundary route trace:
  `/private/tmp/qwen_p4_v_contiguous_auto_boundary_route_v12.log`,
  `5fa6b947868815d9edaf535dd87e0c886f7a534851a68c970671c3c5a7419335`;
- automatic lazy-pipeline profile with retained system shader cache:
  `/private/tmp/qwen_p4_v_contiguous_auto_cold_profile.log`,
  `e05b80e7cffb1058a1f1acc7030bbda5d716130abf33a407a4626cfbcc95b34b`;
- failed three-state long-prefill attempt:
  `/private/tmp/qwen_p4_v_contiguous_auto_14k_baseline.log`,
  `c69482b58a43f3db1518162785865c5332a68093734579002c7fd76694e30735`;
- failed two-state chunk-256 and chunk-128 attempts:
  `/private/tmp/qwen_p4_v_contiguous_auto_14k_prod_quality.log`,
  `0da7a45ff541cc976817ae048ec086f44d474b14ff66ef149b467c9a34ce06f6`,
  and `/private/tmp/qwen_p4_v_contiguous_auto_14k_prod_quality_chunk128.log`,
  `99a52a9ecedeacf8b4d2712ff9b10c9bd67a8aba36f6e929cf53f2f8a7c37d94`.

Refresh on policy or kernel/source selection, probe semantics, cache layout,
model/prompt, compiler/Metal toolchain, device/OS identity, timing method,
thresholds, watchdog behavior, default-library availability, safety policy, or
evidence loss.

### Split-K stage-two SIMD scalar broadcast is rejected (2026-09-20)

The fused split-K stage-two kernel repeats the same maximum scan, exponential
weight, and normalization sum in every SIMD lane. A default-off source variant
made lane zero compute those scalar values and broadcast them with
`simd_shuffle`, while preserving each lane's output dimensions and the
ascending `partial_o` accumulation order.

A model-free Apple M2 Max probe timed 32 dispatches per completed command and
alternated baseline/candidate order across ten pairs at 96, 224, and 256
split-K blocks. The outputs were bitwise identical, finite, and reported zero
status in every row. Nevertheless, the candidate regressed median GPU time by
`18.49%`, `20.85%`, and `22.74%`; it won only `0/10`, `2/10`, and `2/10`
pairs. A preceding independent run showed the same direction, with regressions
of `18--22%` and at most one win per row. The added per-block shuffle and
lane-divergent scalar ownership cost more than the redundant lane-local
arithmetic on this device/compiler.

**decision:** the mathematical transformation is correct but the optimization
is BROKEN for Apple M2 Max performance. The production source, policy, cache
keys, and specs were restored to their pre-experiment state. Do not add the
runtime option or advance this route to full-token testing. Reopen only after a
different mechanism removes the per-block shuffle or new device/toolchain
evidence changes the tradeoff.

Evidence:

- `/private/tmp/qwen_splitk_stage2_uniform_scalar_abba.log`,
  `c2c321b9b247ddd23a89704a3225dfce7a60200ac8e92f46ffb36cd361b2e9d2`;
- `/private/tmp/qwen35_splitk_stage2_uniform_scalar_probe.cr`,
  `7dc7015354bac1f2785b7409519e13e327979f722b6a0b28aae56d81f607f6d4`;
- probe executable,
  `bc6fdea8ba2a2b7b4746d58e89df5218b6a13892272c4c44855b65929f01adf0`.

Refresh on stage-two arithmetic or layout, Metal compiler/toolchain, device,
timing method, or evidence loss.

### BF16 contiguous-V is correct but not promotable (2026-09-20)

The accepted P4 contiguous-V lane ownership is representation-independent
after V has been materialized in the shared FP32 tile. A default-off BF16 T8
variant therefore reused that exact branch without changing cache bytes,
dequantization, score reduction, or stage-two layout. The focused 8K resident
spec passed on Apple M2 Max; baseline, BF16 T8, and BF16 T8 contiguous-V kept
the same canonical K/V payloads and stayed within the existing numerical
oracle. Both timing probes reported `max_output_delta=0` in every pair.

The performance evidence did not survive repetition:

- at 8K, candidate wall time regressed by `0.91%`; completed GPU time improved
  by `5.16%`, but only `7/10` GPU pairs and `4/10` wall pairs won;
- the first 16K run showed `+2.27%` wall and `+12.06%` GPU with `8/10` and
  `7/10` wins respectively, still below the predeclared gate;
- the repeated 16K run reversed the apparent gain: wall regressed by `20.47%`,
  GPU improved only `1.25%`, and wins fell to `2/10` and `5/10`.

**decision:** correctness is ROBUST, but the performance claim is BROKEN on
this device/toolchain. The four BF16 owners do not justify a new production
pipeline variant without a stable local crossover. The policy, source
variants, probe switch, and spec expansion were removed. Reopen only with a
lower-noise batched kernel discriminator or changed compiler/device evidence;
do not infer a whole-token benefit from the noisy first 16K mean.

Refresh on V ownership, BF16 materialization, Metal compiler/toolchain,
device, timing method, or evidence loss.

### P4 T8 row-stride indexing is correct but not promotable (2026-09-20)

The P4 T8 split-K tile loader derives `position_in_tile` and `d` from a
vector index using division and remainder by `head_dim`. A default-off
Qwen3.8-specific variant replaced that arithmetic with the exact fixed-shape
mapping for `head_dim=256`, 192 threads, and six SIMD groups:
`d=(thread_index&31)<<3` and
`position_in_tile=(thread_index>>5)+6*k`. It changed neither cache bytes nor
the dequantization, reduction, or stage-two accumulation order.

Two same-process AB/BA sequences covered 6K, 14K, and 16K visible prefixes,
ten pairs per row. Every pair retained canonical K/V bytes and reported
`max_output_delta=0`. The first sequence failed the predeclared `>=3%` and
`>=8/10` wall-and-GPU gate at every prefix: the 6K favorable means lacked
pairwise support (`+5.87%` wall, `+8.97%` GPU; `7/10`, `4/10` wins), 14K was neutral
(`+0.17%`, `+0.34%`; `5/10`, `5/10`), and 16K remained below threshold
(`+2.97%`, `+1.87%`; `6/10`, `7/10`). The independently repeated rows were
also unstable: 6K regressed by `12.31%` wall and `18.72%` GPU, while apparent
14K/16K mean gains had only `5/10` and at most `6/10` wins.

**decision:** correctness is ROBUST, but the performance claim is BROKEN on
Apple M2 Max with this compiler/toolchain. The candidate showed no stable
measured benefit, so an extra shape-specific pipeline is not justified;
whether the compiler strength-reduced the arithmetic or loader/dequantization
work dominated remains unresolved. The production source, policy, probe
option, and spec expansion were removed. Reopen only if generated Metal
assembly or a lower-noise batched discriminator identifies a persistent
integer-division cost on a changed compiler/device; do not promote from a
favorable mean without pairwise stability.

Repeat evidence:

- `/private/tmp/qwen_p4_t8_row_stride_6k.log`,
  `144b71b08744daa59798586a682da0b70379e9dd22ef1e74282b42ed27b19d21`;
- `/private/tmp/qwen_p4_t8_row_stride_14k.log`,
  `17aad7060c961b5b3b359c2fb699555b38adcb141c5bacd5bca179dfe78c533f`;
- `/private/tmp/qwen_p4_t8_row_stride_16k.log`,
  `fb5e6b7f03f6fed2736c113dcd772dc4d6dedad678871ccc99bc502719fe4a4a`.

Refresh on T8 tile ownership, head shape, Metal compiler/toolchain, device,
timing method, or evidence loss.

### Packed-prefix P4 T8 tile specialization is rejected (2026-09-20)

A temporary default-off P4 T8 helper removed the per-vector exact-current-row
branch only for K/V tiles proven wholly inside the immutable packed prefix.
Mixed tail tiles retained the existing loader, including the exact FP32 current
row. The experiment ran on top of the admitted contiguous-V route and changed
neither the cache representation nor the floating-point accumulation order.

Two same-process AB/BA sequences covered 14K and 16K visible prefixes with ten
pairs per row. Every pair retained canonical K/V bytes and reported
`max_output_delta=0`, but no row met the predeclared `>=3%` and `>=8/10`
wall-and-GPU gate. The first sequence regressed at 14K by `1.43%` wall and
`1.40%` GPU (`4/10`, `3/10` wins), and at 16K by `9.67%` wall and `14.85%`
GPU (`4/10`, `4/10`). The independent repeat showed only unstable mean gains:
14K improved by `2.86%` wall and `4.09%` GPU with `5/10` and `6/10` wins;
16K improved by `1.07%` and `1.82%` with `4/10` and `6/10` wins.

**decision:** correctness is ROBUST, but the performance claim is BROKEN on
Apple M2 Max with this compiler/toolchain. The source helper, pipeline variant,
policy, probe option, and spec were removed. Together with the rejected BF16
contiguous-V and P4 row-stride variants, this closes the current split-K loader
micro-optimization corridor: the next candidate must remove a larger data
movement or dispatch boundary. No compiler or cache mechanism is inferred
without hardware-counter or generated-code evidence.

Repeat evidence:

- `/private/tmp/qwen_p4_packed_tile_14k.log`,
  `0b9937ef0ac430882b461bccf3523b05196b941ce3f52451ecce399c514741a2`;
- `/private/tmp/qwen_p4_packed_tile_16k.log`,
  `1ead68c190a0b44e56670dad5ea9478acf54345fbc9c75ca7eac53a80ff20dc8`.

Refresh on packed-prefix layout, exact-tail ownership, P4 T8 loader or
contiguous-V implementation, Metal compiler/toolchain, device, timing method,
gate, or evidence loss.

### Q4 x16 activation shuffle is rejected (2026-09-20)

The admitted Q4 x16 decode kernel lets both 16-lane halves of a SIMD group
read the same activation fragments while they process different output rows.
A temporary default-off variant kept the existing weight loads, arithmetic,
reduction tree, and row ownership, but loaded each activation `float4` only in
the lower half and used eight SIMD shuffles per Q4 block to feed both rows.

The release build and Metal source compilation passed, but the first guarded
whole-body discriminator rejected the mechanism decisively. On Apple M2 Max
with Qwen3.8-27B Q4_K_M, prompt zero, eight decode tokens, one warmup, and six
interleaved pairs, the production x16 route measured `455.04 ms` p50
(`17.58 tok/s`) while shared-X measured `994.69 ms` (`8.04 tok/s`). The
candidate lost all six pairs and more than doubled latency. The run completed
under a `24576 MiB` process-tree cap and `30%` free-memory floor.

**decision:** the performance claim is BROKEN on this device/toolchain. The
candidate kernel and selector were removed immediately; no semantic-quality
escalation was justified after the fail-fast performance gate. The measured
result is consistent with shuffle dependencies and register pressure costing
far more than repeated cache-served activation loads, but no hardware counter
attributes the cause. Do not retry activation broadcast/staging for this x16
route without generated-code or counter evidence that changes that tradeoff.

Refresh on x16 row ownership, activation layout, Metal compiler/toolchain,
device, or availability of lower-level resource counters.

### Q5_K half-SIMD row ownership is rejected (2026-09-20)

The recurrent QKV path still used the classic Q5_K GEMV mapping where a full
32-lane SIMD group owns one output row. A temporary default-off variant used
the newer half-SIMD idea from llama.cpp's extended matvec family: each 16-lane
half owned an independent output row, dequantized one contiguous 16-value
chunk per lane, and reduced over 16 lanes. The Q5_K bytes, F32 input/output
ABI, and dispatch thread count were unchanged.

The correctness discriminator passed on a real Q5_K `4096 -> 8192` tensor:
the Metal result versus the CPU reference had cosine `1.0` and maximum
absolute delta `5.9604645e-7`. The guarded Qwen3.8-27B body-only A/B then used
prompt zero, eight decode tokens, one warmup, and six interleaved pairs. The
production route measured `469.42 ms` average and `469.60 ms` p50
(`17.04 tok/s`); the half-SIMD candidate measured `473.18 ms` average,
`473.17 ms` p50, and `16.91 tok/s`. Production won `5/6` pairs. The run
completed with `71%`
free memory under the `24576 MiB` process-tree cap and `30%` memory floor.

**decision:** bounded numerical correctness is ROBUST, but the performance
claim is BROKEN on Apple M2 Max with this compiler/toolchain. Narrower
reduction and doubled row ownership did not repay the alternative Q5_K
dequantization/data-access path. The kernel, pipeline, and selector were
removed; do not promote this mapping or infer a wider llama.cpp-style ext-GEMV
benefit from its structural similarity. Reopen only with generated-code or
counter evidence that changes the bottleneck.

Refresh on Q5_K dequantization, recurrent QKV shape or quantization, Metal
compiler/toolchain, device, timing method, or evidence loss.

### Q4_K x16 dual gate/up plus SwiGLU is rejected (2026-09-20)

A temporary default-off decode kernel fused the two admitted x16 Q4_K FFN
gate/up GEMVs with the following SwiGLU operation. Each 16-lane SIMD half
retained the production row ownership and reduction tree, while the candidate
loaded the shared normalized activation once, streamed both weight matrices,
and wrote only the final F32 activation. This removed the two intermediate
gate/up arrays and the standalone SwiGLU dispatch without changing the Q4_K
weight representation or downstream FFN-down route.

The Metal source and release build passed. The first guarded product-level
performance falsifier used Qwen3.8-27B Q4_K_M on Apple M2 Max, prompt zero,
eight body-only decode tokens, one warmup, and six interleaved pairs. Production
measured `477.51 ms` average, `479.04 ms` p50, and `16.70 tok/s`; the fused
candidate measured `473.47 ms` average, `474.87 ms` p50, and `16.85 tok/s`.
The candidate won `4/6` pairs, for only about `0.85%` average and `0.88%` p50
improvement. The run completed with `71%` free memory under the `24576 MiB`
process-tree cap and `30%` memory floor.

**decision:** the predeclared `>=3%` product gate is BROKEN on this
device/toolchain. Although the direction was mildly positive, unchanged Q4_K
weight traffic dominates the removed cache-served activation traffic and
dispatch. The temporary kernel, pipeline, and selector were removed. Semantic
parity was intentionally not promoted or claimed after the fail-fast
performance rejection. This closes the current decode gate/up/SwiGLU fusion
corridor; reopen only if a changed layout removes material weight traffic or
lower-level evidence shows launch/intermediate traffic has become dominant.

Refresh on Q4_K layout, FFN shape, x16 row ownership, Metal compiler/toolchain,
device, timing method, admission gate, or evidence loss.

### H16 split-K partial output is rejected (2026-09-20)

A temporary default-off adaptive-attention variant stored the split-K stage-one
`partial_o` scratch in H16 for uniform P4 owners and converted it back to F32
inside the fused stage-two reduction. The experiment was restricted to Apple
M2 Max, the admitted P4 T8 route, visible prefixes of at least 6,144 tokens,
and the current fused-stage-two plus contiguous-V configuration. BF16 owners,
the resident cache format, logits ABI, and the baseline route were unchanged.

The focused 8K one-layer GPU falsifier passed (`1 example, 0 failures`), with
cosine greater than `0.9999999` and maximum output difference below `2e-4`.
The guarded 16K synthetic-prefix product discriminator then used ten
interleaved pairs in both orders. AB measured `81.592/79.880 ms` baseline
versus candidate (`+2.099%`, `7/10` wins); mirrored BA measured
`81.672/80.261 ms` (`+1.728%`, `7/10`). Output token IDs and text were
identical, but the largest observed top-one logit delta was
`6.389618e-4`.

**decision:** the predeclared `>=3%` and `>=8/10` performance gate is BROKEN
in both orders. Halving only the P4 `partial_o` scratch traffic does not repay
the H16 conversion cost strongly enough, and it introduces measurable numeric
drift. The runtime, policy, kernel, probe, and spec changes were fully removed.
A real-prefix top-two/ECS run was intentionally skipped after the fail-fast
speed rejection; therefore no semantic-quality claim is made. Reopen only if
a new representation removes the conversion boundary or a wider design
eliminates the partial-output round trip rather than merely narrowing it.

Repeat evidence:

- `/private/tmp/qwen_partial_h16_16k_ab.log`,
  `9685a95853ead88ed57013fbc7c4b6acd7dcbef98f327a1f4ef812ba86d7dd3c`;
- `/private/tmp/qwen_partial_h16_16k_ba.log`,
  `41353ed72e9ce96ce5510b86ceae3e6e9c0ca5ef009004d7bac3d91fcc3753be`.

Refresh on split-K scratch representation, stage-one/stage-two fusion boundary,
P4/BF16 ownership, Metal compiler/toolchain, device, timing method, gate, or
evidence loss.

### Ordered direct-QK reduction is rejected (2026-09-20)

The experimental P4 direct-QK route avoids shared K materialization, but its
SIMD tree reduction changes F32 accumulation order and fails the strict
real-prefix logit gate. A temporary variant split each lane's eight products
into the same two four-wide contributions used by the shared-K route, then had
all lanes broadcast those contributions in source-lane order. This reproduced
the legacy sequence of 64 additions without adding scratch or a threadgroup
barrier; only lane zero published the resulting score.

The release build and Metal source compilation passed. In the guarded
model-free 8K adaptive-attention discriminator, canonical K/V bytes remained
unchanged and maximum output delta was exactly zero, confirming that the
ordered reduction removed the local numeric difference. Performance failed
decisively: baseline/candidate mean wall time was `2.756/3.342 ms`
(`-21.249%`, `2/10` candidate wins), while completed-GPU time was
`1.896/2.353 ms` (`-24.141%`, `2/10`). An earlier diagnostic run showed the
same direction and no candidate GPU wins.

**decision:** exact legacy arithmetic is attainable, but serializing the 64
contributions through SIMD shuffles destroys the direct-QK speed mechanism.
The performance claim is BROKEN on Apple M2 Max with this compiler/toolchain;
the temporary kernel change was removed and no full-model or real-prefix run
was justified. Keep the existing tree-reduced direct-QK route default-off. Do
not retry an exact-order gather unless generated code or a new reduction
primitive can preserve order without the long shuffle dependency chain.

Evidence: `/private/tmp/qwen_direct_qk_ordered_8k.log`, SHA-256
`71947dfd15997621a075e43563a23897913ee49756b58887e3a6dde4dabf03a1`.
Refresh on direct-QK reduction geometry, Metal compiler/toolchain, device,
timing method, gate, or evidence loss.

### Paired-block P4 split-K is rejected (2026-09-20)

A temporary default-off P4 T8 variant kept the admitted 64-token split-K
block width but assigned two adjacent blocks to one 384-thread threadgroup.
The twelve SIMD groups processed the two blocks concurrently, merged their
online-softmax summaries in ascending block order inside threadgroup memory,
and emitted one global partial instead of two. The route was restricted to
Apple M2 Max, tile 15, contiguous-V, direct-QK off, and a 64-token chunk. Its
32,256-byte threadgroup footprint stayed 512 bytes below the 32 KiB limit.

The policy spec passed (`13 examples, 0 failures`). Focused Metal correctness
covered both an 8K prefix and visible lengths `1,6,15,16`, including the empty
second slot of an odd block pair; both examples passed. Canonical K/V bytes
were unchanged, and the paired 8K timing discriminator observed maximum
output delta `1.9e-7`.

The guarded model-free 8K fixed-snapshot A/B used ten interleaved pairs.
Baseline/candidate mean wall time was `2.334/3.002 ms` (`-28.619%`, `0/10`
candidate wins). Completed-GPU time was `1.378/1.684 ms` (`-22.158%`, `1/10`
wins). The run completed with `70%` free memory under a `4096 MiB` process
cap and `30%` memory floor.

**decision:** correctness is ROBUST in the bounded synthetic scope, but the
performance claim is BROKEN on Apple M2 Max with this toolchain. Halving the
number of global summaries and stage-two inputs does not repay doubling the
threadgroup width and consuming nearly the full threadgroup-memory budget.
The candidate failed before any 27B-model escalation, so all runtime, policy,
probe, and spec changes were removed. This rejects the current same-threadgroup
two-block design, not every hierarchical merge. Reopen only if a smaller
cooperative group or a different intermediate representation reduces global
partial traffic without the 384-thread/32 KiB occupancy cost.

Refresh on split-K group geometry, threadgroup-memory layout, Metal
compiler/toolchain, device, timing method, promotion gate, or evidence loss.

### Centroid-space P4 QK is rejected (2026-09-20)

A temporary P4 T8 split-K variant tested the exact algebraic factorization
`Q * (mean + sigma * centroid) = mean * sum(Q) + sigma * sum(Q * centroid)`.
Each SIMD lane decoded eight centroid values once and accumulated them against
all six GQA queries; the row affine transform was applied only after the
256-value reduction. This removed F32 K-tile materialization from the key path
and avoided per-value mean/sigma reconstruction, while leaving the canonical
cache, V path, softmax, stage two, and host ABI unchanged. The current-token
tail remained exact F32.

The focused 8K Metal resident spec compiled the source and passed its numerical
and cache-byte checks. The guarded model-free fixed-snapshot discriminator then
ran ten interleaved pairs at prefix 8,191. The candidate retained canonical K/V
bytes and had maximum output delta `1.9e-7`, but mean wall time regressed from
`2.725` to `3.132 ms` (`-14.956%`, `1/10` wins) and completed-GPU time regressed
from `2.250` to `2.546 ms` (`-13.179%`, `1/10`). The 16K boundary was skipped
after the predeclared 8K fail-fast gate failed.

**decision:** the algebra is numerically sound in the bounded test, but the
performance route is BROKEN on Apple M2 Max with this compiler/toolchain. The
six query sums and compressed-domain accumulators increase live register and
reduction pressure enough to outweigh the eliminated affine work and K-tile
traffic. The temporary kernel change was fully removed before any 27B-model or
semantic-quality run. Do not retry centroid-space QK with six simultaneous
query accumulators; reopen only with compiler evidence or a different geometry
that bounds register lifetime and first clears the same 8K gate.

Refresh on P4 QK ownership, accumulator geometry, Metal compiler/toolchain,
device, timing method, promotion gate, or evidence loss.

### H16 P4 value-tile staging is rejected (2026-09-20)

A temporary long-context P4 T8 variant kept the admitted shared-K and
contiguous-V schedule but stored packed V rows in H16 inside the transient
threadgroup tile. The visible current-token V row stayed on its exact F32
path. The canonical adaptive cache, K path, softmax, F32 split-K partials,
stage two, and host ABI were unchanged.

The release probe and both Metal pipelines compiled. A guarded model-free
fixed-snapshot discriminator then ran ten interleaved pairs at prefix 16,383
under a 4 GiB process-tree cap and 30% free-memory floor. Canonical K/V bytes
matched and maximum output delta was `1.521e-5`. Baseline/candidate mean wall
time was `3.393/3.692 ms` (`-8.794%`, `4/10` candidate wins), while completed
GPU time was `2.643/2.877 ms` (`-8.835%`, `3/10`).

**decision:** bounded numerical behavior is acceptable, but the performance
claim is BROKEN on Apple M2 Max with this compiler/toolchain. K still requires
the full F32 threadgroup allocation, so H16 V staging does not improve the
kernel's static shared-memory footprint; its conversions instead add work to
the hot accumulation loop. The temporary kernel, runtime selector, and probe
seam were removed. Do not retry value-tile narrowing unless a new schedule
also removes or splits the F32 K allocation, or compiler/counter evidence
shows a changed resource regime.

Refresh on K/V tile lifetime or allocation, P4 loader, contiguous-V schedule,
Metal compiler/toolchain, device, timing method, promotion gate, or evidence
loss.

### Split-K stage attribution closes the stage-two corridor (2026-09-20)

A model-free diagnostic measured the admitted Apple M2 Max P4 split-K stages
with the same deterministic buffers and exact production geometry: tile 15,
T8 dequantization, shared K, contiguous V, a 64-token block, 24 query heads,
four KV heads, head dimension 256, and fused stage two. Each reported GPU
interval amortized 16 dispatches; ten samples were collected in rotating
stage-one/stage-two/combined order. The combined path encoded both stages in
one command buffer, while the isolated rows used separate command buffers.

At 8K, median stage one/stage two/combined GPU time was
`0.43955/0.03976/0.52103 ms`; stage two was `8.30%` of the sum of isolated
stages and `7.63%` of the combined interval. At 16K the corresponding result
was `0.83628/0.07491/0.97945 ms`, or `8.22%` and `7.65%`. Outputs were finite,
the device status word remained zero, and a repeat made immediately before the
recorded run gave the same conclusion (`7.55%` at 8K and `7.79%` at 16K by
combined-interval attribution).

The combined interval was `7.5--8.7%` larger than the sum of independently
timed stages. Therefore the separate rows are diagnostic, not additive product
timing, and the excess is not assigned to either kernel. Both attribution
frames nevertheless put stage two below 10% and stage one near 92% of the
visible split-K work.

**decision:** the hypothesis that fused stage two remains a first-order
optimization target is BROKEN for the measured P4 8K/16K regime. Even perfect
removal has only about an 8% local ceiling, while any realizable rewrite saves
less. Keep the admitted fused reducer, but stop spending optimization cycles
on reducer micro-variants. The next adaptive-attention work must remove a
material stage-one boundary: P4 K/V dequantization, tile traffic, query/value
ownership, or the production command/dataflow around stage one. This does not
claim a whole-model percentage or transfer the attribution to BF16 or another
device.

Evidence:

- `/private/tmp/qwen35_splitk_stage_attribution_probe.cr`,
  `536e320730db213bab41a41ad4006efa094cb77d9c2c9e3a048ca830fa795435`;
- `/private/tmp/qwen_splitk_stage_attribution.log`,
  `4687ae8bc66ee7ceea7374476888ed6a15a4080980e7ae07d03d9f8609e3f87f`.

Refresh on split-K geometry, stage fusion, P4 loader or value ownership,
Metal compiler/toolchain, device, timing method, or evidence loss.

### P4 MMA8 H16 stage one is rejected (2026-09-20)

A temporary default-off split-K stage-one kernel used 8x8 simdgroup matrices
for the exact Qwen3.8 GQA6 shape. It padded each six-query KV-head group to
eight H16 query rows, dequantized packed P4 K/V into H16 tiles, retained F32
online-softmax state and F32 split-K partials, and left the canonical adaptive
cache unchanged. Admission was restricted to Apple M2 Max, uniform P4,
24 query heads, four KV heads, head dimension 256, chunk 64, one-token decode,
and a visible prefix of at least 6,144 tokens.

A separately captured model-free probe reported 12,576 bytes of static
threadgroup memory, execution width 32, and a maximum 896 threads for its MMA8
pipeline. With query conversion included, isolated stage one improved by
`22.76%` at 8K and `18.13%` at 16K, with all ten pairs won at each length and
maximum output delta near `1e-6`. This proves the local matrix mechanism, not
the integrated route's product speed or strict logit quality.

The guarded product discriminator did not promote the route. On the real
Qwen3.8-27B Q4_K_M model with a 6,667-token production-prefilled prefix and
32 alternating decode pairs, baseline/candidate mean full-token time was
`68.545/66.671 ms` (`+2.734%`, `30/32` candidate wins). This is diagnostic
only because quality mode deliberately marks its timing gate invalid. The
number is below the declared `>=3%` threshold, but this run cannot certify
timing admission. All 33 aligned decode steps retained top-one and ordered
top-two IDs, token ECS was `1.0`, and the free trajectories shared all 34
observed tokens. However, the declared
`1e-4` numeric gate failed: maximum top-one, top-two, and margin deltas reached
`0.023448944`, `0.011590958`, and `0.011857986` respectively.

**decision:** the local matrix mechanism is promising, but this H16 Q/K/V
formulation is BROKEN for strict promotion and does not establish a promotable
product-speed result. Matching token IDs, top two, text, and ECS does not
override the reproducible logit drift. The temporary kernel, converter,
policy, probe, and spec wiring were removed. Preserve the architectural
lesson—eight-row padded GQA6 MMA can materially reduce stage-one cost—but do
not reintroduce this mixed-precision route unless a new correction or dataflow
preserves the existing numeric contract and independently clears end-to-end
timing.

Evidence: `/private/tmp/qwen35_adaptive_p4_mma8_probe_v4.log`, SHA-256
`16fe0e497e12f11b13beec70db05ad432897b87f12198693a3561b07faa0d94c`;
`/private/tmp/qwen35_mma8_real_prefix_quality.log`, SHA-256
`d8ec917bb0798582e83c5cc8f31642956a6c9c35d683dfce70e71b9c9544acfe`.
Refresh on arithmetic precision or correction scheme, split-K ownership,
Metal matrix primitives/compiler, device, quality contract, timing method, or
evidence loss.

### Affine-centroid P4 V-MMA is rejected (2026-09-20)

A temporary default-off split-K stage-one variant used the exact P4 identity
`sum(p * V) = sum(p * sigma * centroid) + sum(p * mean)`. K materialization,
QK accumulation, online softmax, row means, probability-times-sigma, the exact
current-token contribution, and the split-K partial ABI remained F32. Only the
fixed P4 centroids crossed H16 and an 8x16-by-16x8 simdgroup matrix multiply.
The canonical adaptive cache bytes and publication protocol were unchanged.
Admission was restricted to Apple M2 Max, uniform P4 T8, tile 16, chunk 64,
one-token decode, and at least 8,192 visible tokens; direct-QK and contiguous-V
were excluded.

The model-free falsifier initially found and corrected a cumulative-mean bug.
After the fix, tile tails 1/6/15/16 and nonfinite status bits passed. Three
fresh protected runs kept the maximum output delta at `7.7724457e-5`. Isolated
stage one improved by `6.03--8.55%` at 8K with at least five of six pairs won,
and by `10.48--12.46%` at 16K with all six pairs won. This established a local
kernel opportunity within the declared `1e-4` bound, not product suitability.

The integrated real-model gate falsified promotion. A production-prefilled
8,201-token Qwen3.8-27B Q4_K_M run selected the exact affine pipeline 132
times, covering all 12 P4 owners while the four BF16 owners remained on their
existing route. The baseline and candidate retained all 22 ordered top-two
IDs, 11/11 top-one IDs, a 12/12 common free trajectory, identical text, and
token ECS `1.0`. Nevertheless, the strict numeric gate failed: maximum
top-one, top-two, and margin deltas were `0.0029258728`, `0.002571106`, and
`0.002527237`, respectively. The quality run's timing is diagnostic rather
than promotable, but it also moved in the wrong direction: complete-token
means were `112.867/114.536 ms` (`-1.478%`, only `4/10` candidate wins).

**decision:** the affine decomposition is mathematically useful and the local
MMA kernel is ROBUST within its model-free scope, but this H16-centroid route
is BROKEN for product promotion. Exact token identity, top two, text, and ECS
do not override either the strict logit failure or the disappearance of the
local speedup in the complete graph. The kernel, policy, pipeline identity,
probe mode, and specs were removed. Do not retry the same F32-K/H16-centroid
arrangement merely with a different tile or threshold. Reopen only for a new
arithmetic/dataflow that predicts a material full-token win and preserves the
numeric contract before product escalation.

Evidence:

- `/private/tmp/qwen35_adaptive_p4_v_affine_mma_probe_v4_safe.log`, SHA-256
  `f977a0cf30964eb13f69466c873df4b4cfc202ca0d63a711590d959e6ab0b5ec`;
- `/private/tmp/qwen35_adaptive_p4_v_affine_mma_probe_v4_safe_r2.log`, SHA-256
  `0491e14e11447fe4b9632402c43883cf7f05caeff8982b37bd17e1f074b0900d`;
- `/private/tmp/qwen35_adaptive_p4_v_affine_mma_probe_v4_safe_r3.log`, SHA-256
  `7e08631dc091be48412145d06752b7552aeac4d46efe5725dfc39bd2cd467db7`;
- `/private/tmp/qwen35_affine_mma_quality_8k_smoke.log`, SHA-256
  `4b5a4c381fca1b4f82ba8b34a9f7f90b2e74336eb57586aea06d923eff110712`.

Refresh only if centroid precision/correction, matrix accumulation semantics,
stage-one ownership, compiler/toolchain, target device, quality contract,
full-token timing method, or retained evidence changes.

### One-token 8K adaptive decode budget (2026-09-20)

The existing generator/profile path measured one greedy decode step after a
real 8,197-token Qwen3.8-27B Q4_K_M prefill. The adaptive map was
`p4;27=bf16,43=bf16,47=bf16,51=bf16`; split-K used chunk 64, tile 15, the
admitted P4/BF16 T8 routes, and fused stage two. The adaptive decode was one
Metal command buffer and one Metal synchronization, with no CPU-fallback
matvecs. Its completed GPU interval was `68.72 ms`; host wall time was
`81.8 ms`, of which profiling recorded `8.53 ms` of wave encoding and
`69.45 ms` in the synchronous wait. Prefill took `127.49 s` and is outside
this decode interval.

The profile counted `15,078.22 MiB` of logical matmul weight traffic. The
largest named corridor was recurrent FFN up/gate at `4,590.00 MiB`
(`30.44%`, 96 Q4_K GEMVs). Recurrent FFN down/add contributed another
`2,820.94 MiB` across Q4_K and Q6_K (`18.71%`). Other large corridors were
full-attention FFN up/gate (`10.15%`), the top-one head (`6.60%`), recurrent
projection `5120x10240` (`11.01%` across Q4_K/Q6_K), and recurrent output
projection (`5.37%`).

This is a traffic budget, not a per-phase GPU timing decomposition. The
profile timestamps only the complete command buffer; its encoder trace is
host encoding time. Consequently `30.44%` recurrent-FFN up/gate traffic does
not imply `30.44%` wall or GPU time. The next discriminator must measure the
whole-token delta from removing or replacing recurrent FFN work before any
new kernel is admitted. Treat such a skip experiment only as an upper bound:
it changes hidden activations and cannot certify exact subtractive timing,
quality, or a production optimization. A new route must still predict and
clear a `>=3%` complete-token gate with the normal numeric, top-one, top-two,
trajectory, text, and ECS checks.

**decision:** stop inferring the next optimization from adaptive-attention
microbenchmarks alone. Recurrent FFN is the largest observed logical-weight
corridor and therefore the next falsifier target, but it is not yet a proven
GPU-time bottleneck. Preserve the one-command adaptive publication boundary;
do not split it merely to obtain prettier phase timings.

Evidence: `/private/tmp/qwen35_adaptive_8k_one_token_profile.log`, SHA-256
`ace2f25e53343e6f86ebce1b824e7b5c410129af960e3933512e1e162a763fb0`,
source revision `8f5182483e02b5118bfc68744de5aca7223dbaba`.

Refresh on model or quantization, adaptive map, prefix length, Metal route or
compiler, profile semantics, device, command-buffer composition, or evidence
loss.

### Recurrent-FFN skip falsifier (2026-09-20)

A temporary diagnostic reused the existing `skip_recurrent_ffn` execution
path with two independent ordinary Metal states. Both arms used one command
buffer (`QWEN35_WAVE_CHUNK_LAYERS=0`), were warmed before measurement, and
then ran ten alternating baseline/skip pairs. This was deliberately not an
adaptive-KV or quality run: removing all 48 recurrent FFNs changes hidden
activations and the generated trajectories diverged after the first step.

The baseline/skip wall means were `59.606/35.665 ms`; the `23.942 ms`
difference appeared in all ten pairs. Completed Metal GPU means were
`56.779/32.956 ms`; the `23.823 ms` difference also appeared in all ten
pairs. Wall medians were `59.673/35.694 ms`, and GPU medians were
`57.085/33.080 ms`. The probe exited normally with no device error.

This establishes that recurrent FFN is a material optimization corridor on
this Qwen3.8-27B Q4_K_M / Apple M2 Max configuration. It does not establish
an exact additive cost: the skipped arm changes downstream values and work,
and this short-context ordinary state is not the 8K adaptive product route.
Comparing the `23.823 ms` difference with the separately measured `68.72 ms`
8K adaptive GPU interval yields only a rough ceiling (about `34.7%` of that
interval, or an impossible-perfect-removal ceiling near `53.1%` speedup).
Neither number is an attribution or an achievable performance claim.

**decision:** recurrent FFN is now a measured first-order candidate rather
than a traffic-only hypothesis. The next admitted implementation must preserve
the normal arithmetic and target a genuinely new `up/gate/down` mechanism;
do not repeat rejected shared-X, dual-SwiGLU, adjacent-metadata, NR2/B2,
parallel-gate/up, conversion-only H16, or sequential B32/B64 variants. Before
production edits, identify the actual Q4_K/Q6_K routes used by the exact
`5120x17408` and `17408x5120` operators and predict a complete-token gain.
The normal `>=3%` full-token and numeric/trajectory/text/ECS gates remain in
force.

Evidence: `/private/tmp/qwen35_recurrent_ffn_skip_probe.log`, SHA-256
`b88c500f6f0e9bb6138b672a2ed3b458a81e8696cf0421287b1dbcdd9b9ddb45`,
source revision `48bfbf051941696674b01697f49a6af2c48e7881`. The temporary probe source
was removed after the run.

Refresh on model or quantization, device, Metal compiler, recurrent-FFN
routing, command-buffer composition, skip semantics, profile timing semantics,
or evidence loss.

### Direct-QK pinned-fixture semantic smoke gate (2026-09-21)

The direct-QK plus contiguous-V bundle remains default-off. Its complete-token
gain was material at long context, but a real-prefix run exceeded the existing
`1e-4` logit-delta contract even though token IDs, ordered top two, text, and
token ECS remained unchanged. The numeric gate is not weakened or redefined.

A separate opt-in probe mode, `--semantic-coding-quality`, now supports a
bounded product-semantic falsifier. It requires at least 256 requested decode
positions so all three pinned Crystal fixtures can reach natural EOS. A record
is structurally admissible only when both independent arms reach EOS at the
same final output position, retain the complete self-fed token trajectory and
text, match ordered top two at every aligned step, and report the exact
Qwen3.8-27B / Apple M2 Max direct-QK plus contiguous-V owner configuration.
The scorer then runs both generated answers against the pinned `lower_bound`,
`stable_unique`, and `merge_ranges` external Crystal specs. It requires one
7K--9K context and one context at or above the route's 14,336-token boundary.

The scorer pins the three repository fixture paths, `src/answer.cr`, and the
SHA-256 of each `check.cr`. It binds every record to the exact long-prompt file
and requires that file to end with the pinned task prompt. Probe records and
the full suite are validated before generated Crystal is executed. That
execution is still not sandboxed; generated source must be inspected before
the CLI is invoked.

This is deliberately a smoke certificate, not semantic equivalence. ECS is
reported because it is part of the established diagnostic vocabulary, but it
is redundant when the compared token IDs are identical. Numeric deltas and
the unchanged `1e-4` tolerance remain visible, the report separately states
whether the strict numeric gate passed, and `admission_eligible` remains false
even if all three fixtures pass. A future promotion decision needs broader
task coverage plus a separate balanced timing refresh.

The first live `lower_bound` run exposed an integration bug after producing a
passing semantic record: the terminal `elsif` still evaluated the old numeric
gate whenever the semantic predicate itself was true. The semantic mode now
owns that terminal branch; the legacy numeric-only branch is unchanged. The
release binary SHA-256 after the fix is
`45cb31554b7842d51cc6263cb875355c52317664bae53d11962071c3b73fe243`.

Three guarded Qwen3.8-27B runs then exercised prompts of 7,679, 11,226, and
14,735 rendered tokens. `lower_bound` and `stable_unique` reached aligned EOS
with identical complete trajectories and text, exact ordered top two at all
101 and 93 quality steps, and token ECS `1.0`. Their candidate diagnostic
means were `4.47%` and `6.44%` below baseline, but these quality-mode runs are
not a balanced speed gate. The unchanged strict numeric gate failed, with
maximum margin deltas of `0.010690689` and `0.09759712`.

`merge_ranges` also kept all 195 emitted IDs, text, top-one decisions, and EOS
identical, but one of 194 runner-up tokens changed at sample 163: baseline
`"\n"` versus candidate `".to"`. Ordered top-two agreement was therefore
`387/388`, set overlap fell to one on that step, and the producer correctly
returned failure. The two runner-up output embeddings have ECS
`-0.017498802741005993`, so this was not a close embedding-space substitute.
The suite scorer rejected the manifest before creating an output directory or
executing generated Crystal.

Manual source inspection allowed an independent external-spec diagnostic.
The shared identical baseline/candidate sources passed `lower_bound` and
`stable_unique` (`2 examples, 0 failures` each). The shared `merge_ranges`
source failed (`2 examples, 2 failures`) because the generated implementation
merged integer-adjacent, non-overlapping ranges through `cur_end + 1`. This is
a shared baseline model failure, not a direct-QK regression, but it
independently prevents a coding-quality pass.

Evidence logs and SHA-256 values:

- `/private/tmp/qwen_direct_qk_lower_bound_semantic_v2.log`,
  `4ec417bbec51004f0c7dffec6253c908dbde82c042e6e800777c952b39404ab2`;
- `/private/tmp/qwen_direct_qk_stable_unique_semantic.log`,
  `5621436947d14b9be3ed2c8d34028068c48451d97bd0705c07daad1b61577b2d`;
- `/private/tmp/qwen_direct_qk_merge_ranges_semantic.log`,
  `8aed056c28995995623fd28c4e815857ad1b00424bc08fbaf3238e9ccb13751d`;
- `/private/tmp/qwen_direct_qk_semantic_manifest.json`,
  `d186af0487b8c0466e2f59292853f10f64f90cbf58e737b627b642856a2da2c0`;
- `/private/tmp/qwen_direct_qk_runnerup_ecs.log`,
  `447f7a728b97c50f0cbfc8ff726df112624a6c5a33360def1782e1b2c318b622`;
- `/private/tmp/qwen_direct_qk_external_smoke/{lower_bound,stable_unique,merge_ranges}/qbit_external_spec.log`,
  `be64d4b6deaf1524ff1eab49a9247e3144a2b630b0ee83d757a3ae6d3c9e2417`,
  `7112bf5514241f2dfb142e40735378595c108e8c54a758093684f71bba9268cd`,
  and `b842f3f7ac63b77929ad51fb8774799710f86ee53d3a91479780a395e28569a0`.

**decision:** the live gate is ROBUST and fail-closed in this bounded scope;
direct-QK coding promotion is BROKEN by the declared three-fixture contract.
Keep direct-QK default-off. Do not weaken exact top-two after seeing the result
or describe the identical greedy answers as broad coding equivalence. Reopen
only with a new arithmetic-preserving direct-QK formulation or a separately
declared, independently justified greedy-only product contract; either route
still needs broader tasks and a quiet balanced timing refresh.

Refresh on fixture or oracle changes, prompt construction, probe JSON schema,
EOS accounting, route ownership/policy, numeric tolerance, tokenizer/template,
model, device, or external scorer execution semantics.
