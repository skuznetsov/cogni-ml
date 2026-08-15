# Qwen QBit Cache Frontier

Document status: implementation-bounded

Current frontier: a default-off p7 transport/restore experiment for recurrent
Qwen cache state plus a bounded local ClickHouse physical-storage/read gate.
It may emit revision-0 ClickHouse Native blocks whose QBit column is already
bit-transposed, validate an ordered multi-block Native response, decode logical
records that cross response-block boundaries directly into prepared Metal state
buffers, and measure one isolated temporary MergeTree.

Bounded context: local `.qkv` state artifacts. ClickHouse storage and background
part merges are a separate transport/storage context.

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
- The model probe may accept a bounded external multi-block response only with
  an explicit expected `cache_id`. Prompt/model/token hashes remain the
  responsibility of the surrounding versioned cache envelope.

## Rejected surface

- No production prompt-cache codec, artifact version, manifest value, default,
  or compatibility promise is added in this slice.
- No claim that scalar MSE implies autoregressive parity.
- No claim that ClickHouse background merges are on the cache-hit critical
  path: newly inserted rows must remain readable before a part merge completes.
- No production ClickHouse client, TCP packet framing/compression, durable QBit
  artifact version, CUDA decoder, or default cache route in this slice.
- A fast kernel or compact Native block alone is not evidence that the complete
  cold-hit path is faster.
- Recurrent-only QBit part bytes must not be compared with a full INT8 artifact
  that also contains live KV and its envelope. A storage win requires matched
  artifact boundaries or an explicitly itemized recurrent-plus-KV total.

## Guard-only future

- A versioned recurrent QBit artifact codec with direct device decode.
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
not a matched ClickHouse-vs-ClickHouse result: the KV storage envelope and the
physical INT8 part remain to be measured.

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

The SQL insertion route is a negative result: constructing arrays and casting
them to QBit took 116-156 ms per 8 MiB logical part because ClickHouse had to
transpose the codes. Cogni-ml now writes the pre-transposed revision-zero Native
QBit streams and restores the same columnar representation directly on Metal.
Production TCP framing/compression remains unimplemented and must be benchmarked
before promotion. Multi-block Native response handling is implemented only in
the bounded diagnostic/state-restore path, not as a production cache client.

Production storage should batch all tiles for one or more cache keys per insert
to avoid part-count pressure. Background merges should perform cleanup and
compaction, never admission or visibility.

### Cache-engine contract

- Make cache keys content-addressed over model, tokenizer/chat-template,
  engine/state ABI, prompt tokens, cache policy, and artifact codec version.
- Insert every tile of one artifact in one batch and persist expected tile
  count plus an artifact digest. Restore fails closed on missing, duplicate, or
  mismatched tiles; it never waits for `FINAL` or a merge.
- Store recurrent state as 1024-code QBit tiles with per-tile mean, sigma, value
  count, record kind, layer, and tile ordinal. Keep live KV chunks separately
  because their length follows the cached sequence rather than the recurrent
  tile grid.
- Filter expiration at read time. TTL merges reclaim space asynchronously and
  are not an authority for whether a stale row is admissible.
- Send pre-transposed Native streams and read only retained plane subcolumns.
  SQL Array-to-QBit casts remain a diagnostic fallback, not the serving route.
