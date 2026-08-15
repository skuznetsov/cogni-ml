# Qwen QBit Cache Frontier

Document status: active design-sealed slice

Current frontier: a default-off p7 transport/restore experiment for recurrent
Qwen cache state plus a bounded ClickHouse HTTP storage/read client and an
exact-full-prompt native-runtime route.
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
match the caller-authoritative transcript.

Bounded context: local `.qkv` state artifacts and an explicitly configured
ClickHouse HTTP endpoint. Background part merges are a separate storage context
and never establish cache visibility or admission.

## Admitted surface

- A default-off probe may encode recurrent Float32 records in independent
  affine blocks, quantize normalized values with the 256-level Gaussian
  Lloyd-Max quantizer, retain the most-significant 8, 7, or 6 code planes, and
  reconstruct each prefix with its Gaussian conditional mean.
- The probe may report payload size, CPU encode/decode time, next-token logit
  error, and free-running token parity on a local model.
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
  the complete per-layer recurrent/KV record set, exact record byte sizes and
  positions, exact KV bytes, and a block-framing-independent logical digest of
  all QBit columns.
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
- A full session anchor is rebuilt at the exact completed-transcript token
  boundary with the conservative sequential prefill route. Descendants store
  only the cumulative exact token suffix, parent/anchor identities, transcript
  boundary hash, and certificate. The referenced anchor artifact retains the
  next-token validation; delta metadata does not invent a cached next token.
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

## Rejected surface

- No production prompt-cache default or compatibility promise is added in this
  slice. Envelope schema v1 and the ClickHouse table schema are internal
  boundaries and may still change before production promotion.
- No claim that scalar MSE implies autoregressive parity.
- No claim that ClickHouse background merges are on the cache-hit critical
  path: newly inserted rows must remain readable before a part merge completes.
- No native ClickHouse TCP packet framing/compression, automatic retry policy,
  CUDA decoder, or enabled-by-default cache route in this slice. The runtime
  uses bounded HTTP requests and synchronous inserts only when explicitly
  configured.
- No background write queue, cross-`max_seq` restore, or partially restored
  state reuse is admitted in the active runtime slice.
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
a sequential exact-boundary prefill, while normal generation and suffix replay
retain their accelerated paths.

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
80-token anchor means the advantage shrinks as replay grows, and the first
exact anchor is deliberately expensive. Periodic anchors are currently
synchronous; background materialization remains outside the admitted surface.

ClickHouse 26.7.1.1315 stored the one full anchor in 32.98 MiB of recurrent
QBit parts plus 10.26 MiB of exact-KV parts. Seven checkpoint rows, including
the restore and rollback branches, occupied 6.42 KiB compressed versus
18.16 KiB uncompressed. Individual JSON envelopes grew from 1,529 bytes for
the anchor checkpoint to 2,909 bytes for the deepest measured delta. Thus the
incremental metadata is compact; retained full anchors, not MergeTree merges or
delta rows, dominate storage and materialization cost.

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
