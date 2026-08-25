# Qwen QBit Cache Frontier

Document status: active design-sealed slice

Current frontier: a default-off p7 transport/restore experiment for recurrent
Qwen cache state plus a bounded ClickHouse HTTP storage/read client and an
exact-full-prompt native-runtime route. A non-session cache hit may now restore
directly into configured adaptive GPU KV owners; misses and session requests
retain the ordinary Float32 owner.
It may emit revision-0 ClickHouse Native blocks whose QBit column is already
bit-transposed, validate an ordered multi-block Native response, decode logical
records that cross response-block boundaries directly into prepared Metal state
buffers, attach exact KV from a separate bounded artifact, compare a complete
QBit cache-hit corridor against the matched INT8 artifact in alternating order,
measure one isolated temporary MergeTree, and construct a versioned admission
envelope for the split recurrent-QBit/exact-KV state. The runtime route is
strictly additive: it may read or write only when explicitly configured and it
must preserve the existing local prompt-cache and ordinary-prefill behavior on
every miss, rejection, cache transport failure, or rejected write. System and
Metal failures abort instead of risking a second heavy attempt.

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

Bounded context: local `.qkv` state artifacts and an explicitly configured
ClickHouse HTTP endpoint. Background part merges are a separate storage context
and never establish cache visibility or admission.

## Resident KV experimental slice

This began as a separate default-off experiment. Its all-layer restore path now
composes with the durable cache only for non-session native-runtime hits; the
resident representation itself remains experimental. It admits uniform p4/p5
payloads and a canonical adaptive payload
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
- Automatic row/head/age tier selection, hot tails, state fork/copy, adaptive
  snapshot/writeback, session checkpoint routing, speculative/asynchronous
  decode, and non-GQA6 shapes are guard-only follow-ups. Memory-ratio
  measurements from this experiment do not establish an eightfold production
  context increase.

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
  bounded to eight and replay to 512 tokens; crossing either limit requires a
  new full anchor. Session identities are stored only as SHA-256 hashes.
- Restore always targets a fresh state. A miss, transport failure, admission
  rejection, or validation/shape restore error discards that state before the
  existing local cache or ordinary prefill route is attempted. A system or
  Metal execution failure also releases the partial state, but propagates
  instead of attempting another heavy allocation. A write-back failure is
  observable in cache stats but must not fail or alter the generation already
  in progress. Unexpected system failures still propagate.
- With an explicit adaptive resident-KV environment map, a non-session Metal
  QBit hit may prepare sole adaptive owners, restore recurrent state and exact
  live KV into them, and continue through the synchronous prefill/decode wave.
  `adaptive_hits` reports only a successfully restored state that actually owns
  adaptive KV. A miss prepares the ordinary Float32 state so existing writeback
  remains valid. Session requests always use the Float32 restore route.
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
- An adaptive native-runtime hit is read-only with respect to durable state.
  It does not snapshot or write back its extended prefix. Session checkpoints,
  anchor renewal, state fork/copy, and tail clearing continue on the Float32
  route until adaptive snapshots have their own admission certificate.
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
- Injected ClickHouse miss, malformed admission, transport exception, and
  validation/shape restore exception must all reach the matched
  ordinary-prefill result without consuming the partially restored state. An
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
- Delta depth and replay tokens are bounded. Crossing either bound writes a new
  full anchor; failure to publish that anchor leaves the previous committed
  checkpoints readable.
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
- Stop delta promotion if total restore plus replay does not beat full prefill,
  or if retained bytes per rollback boundary do not fall below full-snapshot
  storage. Do not hide anchor cost or checkpoint-index bytes.

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

The implementation intentionally adds no new container, table, or checkpoint
graph. Remaining guard-only surfaces are adaptive session checkpoints,
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
Direct device streaming, buffered cold reads, adaptive session checkpoints,
and production-default policy remain guard-only.

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
