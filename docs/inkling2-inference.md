# Inkling 2 text inference

The native Crystal engine supports the [Inkling 2 Mini IQ4_NL GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)
and the [TML-d24-varloop-research IQ4_NL GGUF](https://huggingface.co/ggml-org/TML-d24-varloop-research-GGUF).
It runs one text sequence at a time. The fixed-depth Metal path reads
quantized weights from the mapped GGUF and keeps intermediate activations,
attention, convolutions, loop controllers, and MoE on the device. By default,
prefill runs up to 32 prompt tokens together per layer visit. The same engine
reuses its GPU workspace across tiles, generated tokens, and core passes.

```sh
make build/inkling2_generate
./build/inkling2_generate --model /path/to/Inkling-2-Mini-IQ4_NL.gguf \
  --raw --prompt 'The capital of France is' --tokens 8
./build/inkling2_generate --model /path/to/TML-d24-varloop-research-IQ4_NL.gguf \
  --raw --prompt 'The capital of France is' --tokens 8 --passes 10
./build/inkling2_generate --model /path/to/TML-d24-varloop-research-IQ4_NL.gguf \
  --raw --prompt 'The capital of France is' --tokens 2 \
  --passes 10 --stop-delta 0.15 --trace-loops
```

Pass `--staged-metal` to select token-by-token prefill on the staged Metal
backend or `--cpu` to select the CPU backend. For a CPU-only build,
run `crystal build -Dcpu_only --release bin/inkling2_generate -o inkling2_generate`.
Use the raw mode for literal completions or omit `--raw` for the supported
single-user chat template.

The Mini checkpoint uses three passes. The research checkpoint defaults to
three and accepts a fixed depth from one to ten. In each case the
prelude and epilogue apply six layers once, and each pass through the core
reuses the middle twelve layers. There are 48 layer visits per token at three
passes and 132 at ten. Each visit has its own KV and convolution history. More
passes also raise the cache cost, especially in the global-attention layers.

Every core pass starts from the prelude's output. The second and subsequent
passes read a message bank formed by the previous pass's completed core blocks
and terminal state. `--trace-loops` reports the relative L2 change between
successive banks after RMS normalization as
`norm(B_current - B_previous) / max(norm(B_previous), 1e-6)`.

The optional research-only `--stop-delta` uses `--passes` as a cap of at
least three. A token exits after two successive changes at or below the
specified positive threshold. If a later token reaches a deeper pass, the
engine runs the missing core passes for earlier tokens in causal order
to populate that pass's KV and convolution state. The epilogue retains
the exit chosen for each earlier token. With tracing enabled,
`backfilled_core_passes` counts this deferred work. Backfilling can defer
compute to a later token, and stored continuations add memory overhead.
The threshold measures movement in the latent and is an experimental policy
for choosing depth.

Thinking Machines Lab reports that the research checkpoint improves through
ten passes on N-Hop Lookup. Its aggregate no-CoT time horizon is roughly flat
past three. [Release evaluation](https://thinkingmachines.ai/news/inkling-2-mini/#no-cot-capabilities)

In a local research IQ4_NL sweep at `47cbfe40`, the same 100 lookup tasks
scored 20, 22, 21, and 23 correct at 3, 4, 5, and 10 passes, respectively.
Always answering A scored 20. An explicit key/value rendering improved
the one-hop pilot. On its held-out set of 40 tasks at 3–6 hops, the four
depths scored 14, 13, 12, and 12 correct. These runs measure sensitivity
to the local prompt formats. The published evaluation uses its own tasks
and inference setup.

## Metal throughput on Apple M2 Max

The November 23, 2026 comparison used cogni-ml `872baf18` and llama.cpp
`e6d624b9` with the same research IQ4_NL GGUF, three passes, FP32 KV,
the same prompt and 48 forced decode IDs, and the same prefill tile
boundaries. Flash Attention was disabled in llama.cpp. Cogni-ml used
`INKLING2_PREFILL_ATTENTION=ordered` and `INKLING2_MOE_PREFILL_ROUTES=32`
with explicit prefill tiles. Ordinary CLI defaults are a 32-token tile,
`regular` attention, and 16 expert routes per tile.

Both engines returned one full-vocabulary logit row at the end of
prefill and one after each scalar decode step. Wall time excludes model
load, warmup, and token selection. Prefill rates count prompt tokens and
decode rates count the 48 forced inputs, including the step across the
1,024-token sliding-window boundary on the long fixture.

| Prompt + decode | Prefill tiles | Cogni-ml prefill (tok/s) | llama.cpp prefill (tok/s) | Cogni-ml scalar (tok/s) | llama.cpp scalar (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 188 + 48 | 128 + 60 | 199.77 (1) | 242.03 (3) | 28.46 (1) | 26.84 (1) |
| 1,022 + 48 | 512 + 510 | 351.62\* | 332.18\* | 25.88 (1) | 25.35 (3) |

Parentheses give the number of quiet intervals out of three warmed
repeats for each phase. Rates are medians of those intervals. The
quiet-host check requires three GPU utilization readings at or below
30% before each phase, then sampled background CPU below one core
total and half a core per process, with gaps below one second.
The asterisk marks the long prefill medians over all three repeats
per engine. Background load left one quiet cogni-ml interval and zero
quiet llama.cpp intervals there, so those prefill rates are diagnostic.
The top-ranked IDs agreed at 47 of 49 positions for the 188-token
N-Hop fixture and 49 of 49 for the 1,022-token fixture.

In a separate warmed run at the same cogni-ml revision, `block4` processed
40.23 tok/s after an 81-token prose prompt and 37.54 tok/s after the
1,022-token prompt. Each rate is the median of three quiet baseline
intervals over 48 decode rows. The benchmark supplies every input ID
in advance, includes greedy top-1 scans, and excludes prefill.
In actual generation, n-gram throughput depends on draft availability
and acceptance.

## Fixed-depth Metal execution contract

Scalar decode submits one Metal command buffer per token. Prefill evaluates a
bounded tile of consecutive tokens together at every layer visit and reads
one logit row for the whole prompt. The same per-visit KV and convolution
histories continue into decode. Each new query reads old history and the
causal part of its tile before the tile can overwrite the sliding cache.
Intermediate buffers stay on the GPU across the prelude, repeated core,
and epilogue. With fixed depth, the CPU submits the work and reads the
requested logit rows.

The resident Metal backend accepts explicit tile sizes from 1 through 128
and the wider sizes 256 and 512. The default is 32. All sizes use float32
staging by default. The explicit 256- and 512-row modes keep final tiles
of fewer than 16 rows at the same boundary as the 128-row mode.

Select `--prefill-tile auto` or pass `auto_prefill_tile: true` to
`Inkling2Engine.new` to use up to 128 rows on a fresh sequence. The same
cap applies to short appends. When the engine already has a live prefix and
one `prefill` call appends at least 512 tokens, that call uses tiles of up to
512 rows. The CLI prefills one fresh sequence. Use consecutive `prefill`
calls on the same engine to exercise the wider KV append path. Automatic
sizing requires resident Metal and is mutually exclusive with an explicit
tile in the engine constructor.

Tiles from 16 through 512 rows use grouped expert GEMM on Apple Silicon
when at most 256 experts are present. The GPU groups routes by expert for
the IQ4_NL and Q8_0 gate/up and down GEMMs. Gate/up applies SiTU in its
epilogue. The down output returns to the original token and routing rank
for the weighted sum. The tile height is a runtime dispatch parameter.
The Metal pipeline cache keys kernels by function name. Wider tiles grow
the reusable scratch buffers as needed.

Set `INKLING2_MOE_PREFILL_ROUTES=32` to try two expert tile widths in the
grouped path. The GPU router sends tiles with up to 16 routes to the M16
GEMMs and denser tiles to the M32 GEMMs. It writes the dispatch sizes from
the map counts. Both widths keep FP32 staging and accumulation. The mode
adds one routing pipeline and four GEMM pipelines to the warmed cache.
On an M2 Max with the research IQ4_NL GGUF at three passes, appending a
512-token tile after a 510-token prefix improved from 225.0 to 232.5
tokens/s in two quiet A/B pairs. Logits and KV matched bitwise.
`INKLING2_MOE_PREFILL_ROUTES=16` is the default.

Set `INKLING2_MOE_PREFILL_ROUTES=16_indirect` to launch the M16 gate/up
and down GEMMs using the actual tile count. The routing kernel writes both
Metal dispatch grids into GPU scratch and reuses the existing FP32 M16
GEMMs. Warming this mode after direct M16 adds one routing pipeline.

The resident path fuses Q/K/V/R preparation. Eligible Q8_0 attention
projections share a dispatch. Eligible F32 loop controllers compute
gain, forget, and their epilogue together in scalar decode and prefill.
Scalar IQ4_NL MoE fuses gate/up with SiTU and reuses activation loads.

Decode attention with more than 64 visible keys partitions the key
range and combines the softmax partials before applying the output gate.
The optimized split-K kernel computes scores once per partition and reuses
them for PV. Its sliding-window loop walks contiguous spans of the KV ring
in logical order, preserving the FP32 summation order across wrap.

Set `INKLING2_PREFILL_ATTENTION=scores` to cache FP32 scores during
multirow resident prefill. Set it to `shared_kv` to share each K/V load
across two adjacent query rows and their three query heads per KV head.
The default `regular` mode is the reference for bitwise comparisons.
The experimental kernels require 128 head channels, 16 relative channels,
and at most 1,024 visible keys in every query row of the tile. The shared
kernel also requires three query heads per KV head. Other shapes and
decode use the existing attention dispatch.

The prefill kernels keep the zero-logit sink and the reference FP32
accumulation order. Both modes size their threadgroup score storage at
dispatch.

Set `INKLING2_PREFILL_ATTENTION=ordered` to batch queries against packed
FP32 K and V. On layouts with three query heads per KV head, QK shares
the packed K loads across all three heads. QK keeps the lane-local additions
and SIMD reduction of `scores`. One kernel computes softmax and PV, starting
with the zero-logit sink and visiting keys in ascending order. Packing,
QK, and softmax/PV reuse GPU scratch across visits. This path handles tiles
of 64–512 rows in the first 1,024 positions with an unwrapped cache,
a vector-aligned gate, 128 head channels, 16 relative channels, and up to
24 query heads and eight KV heads. Other tiles use `scores` when eligible
and regular attention otherwise.

The fixed-depth resident Metal engine also exposes
`decode_block4(StaticArray[id0, id1, id2, id3])` for forced continuations.
It evaluates all four input IDs in one command, returns the unpadded
next-token logits after each input, and advances the sequence by four.
Each row uses the decode reduction order. Tokens routed to the same
expert share packed weight loads. Eligible Q8_0 projections reuse
activation reads across rows, and the attention output projection can
apply its static convolution and commit raw history in the same kernel.
Compare the returned logits and retained F32 KV with four `forward` calls
from the same prefix to test the block path independently of prefill
arithmetic.

The staged backend remains available for adaptive depth, traces, and
checkpoint layouts outside the resident kernel bounds. Backend selection
occurs before the sequence starts. Validate resident execution against
the staged backend with the same forced IDs, including a cache wrap and
a continuation after prefill. The real-checkpoint gate compares every
logit and each visit's retained KV with 3 and 10 passes.

## Four-token Metal verification

The verifier takes three-line token fixtures. The header holds
`prompt_tokens vocab_size decode_steps` in that order. The next line holds
the prompt IDs, and the last line holds the forced inputs followed by the
next known token. Use the same prefix and forced inputs for both decode
paths.

```sh
make build/inkling2_verify_bench
./build/inkling2_verify_bench --model /path/to/research-IQ4_NL.gguf \
  --case /path/to/lookup-3.txt --case /path/to/lookup-10.txt \
  --mode verify --scenario all
./build/inkling2_verify_bench --model /path/to/research-IQ4_NL.gguf \
  --case /path/to/lookup-3.txt --passes 3 --mode bench --pairs 2
```

The default corpus scenario uses the fixture as supplied. The other two
scenarios cross the 64-key attention split and the 1,024-token sliding
window. The verifier compares raw float32 logit bytes at every output row
and after one follow-up scalar step. It checks the position and raw K/V
bytes at every layer visit after prefill, the first block, and the final
block. A differing byte makes the command fail.

Timing uses fixed prefill tiles of up to 128 tokens and resets to the same
prefix before each lane. Each timed lane copies all logit rows. Warmup and
prefill are outside the timer. The JSONL reports elapsed time and Metal
submissions for both lane orders. This timing measures blocks with four
known inputs. Speculative generation also depends on how many rows it
can commit per block.

## Greedy n-gram verification

Use `inkling2_generate --ngram` with fixed-depth resident Metal to accelerate
exact repeats. The decoder looks for a matching suffix of six to eight token
IDs in its history. When the match supplies three candidate IDs, it verifies
them in one block with the exact pending token. It commits the matching
prefix and takes the next ID from the target model.

Sliding KV and convolution histories reserve up to four extra physical rows.
The logical attention window is unchanged. After a mismatch the decoder
shortens the logical prefix, and later steps overwrite rejected rows. This
requires no GPU cache copy or replay. The decoder takes a scalar step
whenever it finds fewer than three candidates. If verification commits
fewer than three rows, the rest of the completion uses scalar decode.

On the research IQ4_NL checkpoint at 3 and 10 passes, two repeating
continuations accepted all 24 proposed tokens over eight blocks each.
The story and six-hop lookup continuations produced no drafts.
Greedy output matched scalar decode in these cases and on a mixed
counter-record continuation with partial acceptance. In an earlier
three-pass M2 Max comparison at `7dc3116c`, the six-fixture mixed workload
rose from 26.12 to 28.14 committed decode tok/s (+7.75%) with n-gram drafting.
The five fixtures with matched prefill and decode brackets gained 0.89%
in combined throughput. Timings included rejected verification rows,
rewinds, and scalar fallback. This result depends on repeat frequency.

## Greedy 3→10 depth speculation

The experimental `inkling2_generate --depth-speculation --passes 10`
path targets the variable-loop research checkpoint on resident Metal.
It drafts up to three candidate IDs after the exact pending token. For
each of the four inputs it saves the prelude state and third-pass message
bank on the GPU, then feeds passes 4–10 with a single exact decode tile.
The research layout uses a 144 KiB private Metal buffer for the four-row
checkpoint. The draft epilogue has its own KV and convolution histories,
which are populated during prefill and scalar fallback. At a mismatch
or EOS, each history retains the prefix accepted by the target. The four
reserved ring rows preserve the previous live prefix while the verifier
runs. The flag is exclusive with n-gram drafting and requires fixed depth.

The parity gate compares the raw bytes of all four float32 logit rows
against serial depth 10. It hashes each visit's live KV bytes and positions
against serial depth 10 for the target and serial depth 3 for the draft
epilogue. It repeats the logit and KV checks after the next scalar row for
each possible commit length. The 62-token prefix crosses the 64-key
attention dispatch boundary. The 1,026-token prefix crosses the 1,028-row
physical ring during the block while keeping a 1,024-token logical window.

Timing measures natural greedy completions and counts committed rows
of decode. Wall time includes draft, verification, rollback, replay, and
scalar tails. Each pair starts from the same prefix and checks the exact
generated token IDs. Prefill and decode are timed separately.

```sh
make build/inkling2_depth_bench
./build/inkling2_depth_bench --model /path/to/research-IQ4_NL.gguf \
  --case /path/to/four-row-fixture.txt --mode verify --scenario physical_wrap
./build/inkling2_depth_bench --model /path/to/research-IQ4_NL.gguf \
  --case /path/to/lookup-10.txt --case /path/to/story-10.txt --mode bench --pairs 4
```

On the M2 Max with research IQ4_NL weights and prefill tiles of up to
128 tokens, the original `47cbfe40` runs produced the following decode
results after warmup.
Both intervals of each retained pair had sampled background CPU below
one full core, with samples less than one second apart. The speedup is
the median of the paired ratios.

| Prompt | Prompt tokens | Committed decode rows | Discarded / verified rows | Pairs | Serial 10, tok/s | 3→10, tok/s | Decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated sentence | 211 | 32 | 0 / 32 | 2 | 6.80 | 8.87 | 1.30× |
| Repeated logs | 272 | 32 | 0 / 32 | 2 | 6.37 | 8.53 | 1.34× |
| Counter records | 187 | 32 | 0 / 32 | 2 | 6.81 | 7.93 | 1.16× |
| Six-hop lookup | 188 | 16 | 0 / 16 | 2 | 6.98 | 9.14 | 1.31× |
| Short story | 46 | 16 | 2 / 16 | 1 | 6.91 | 7.77 | 1.12× |

The story pair ran speculation first. Background CPU spikes excluded its
serial-first pairs. The other four prompts each had a retained pair in
each order. All generated IDs matched serial depth 10. Prefill timing
varied with order. For the six-hop case, the combined prefill and decode
ratios were 0.89× with the serial lane first and 1.18× with speculation
first.

## Checkpoints used for verification

| Checkpoint | Hugging Face revision | SHA-256 |
| --- | --- | --- |
| Inkling-2-Mini-IQ4_NL.gguf | `db61c5b492a6a034d986328d28274af0a439aa6c` | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |
| TML-d24-varloop-research-IQ4_NL.gguf | `618c48f1e3c54094333e3e652a92b1d99c52e34d` | `e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2` |

The GGUF layout uses IQ4_NL for the early routed experts, Q8_0 for dense
projections and the last four layers' experts, and F32 for the loop controllers
and router. The loader checks loop metadata and tensor layout and extents
before decode.

## Historical Metal baselines

### Staged Metal MoE

On an Apple M2 Max with the research checkpoint, the Metal MoE path matched
the native engine at `9fdba2dd` on all 256 top-ranked tokens. The comparison
replayed the same token histories and checked all 200,058 float32 logits at
each sampled position. The largest absolute logit difference was `6.99e-5`.

Both builds warmed the first 16 prompt tokens and reset the sequence before
timing prefill and 31 decode steps separately. The table gives the median of
three runs per depth for the 46-token story prompt and one run per depth for
the 188-token six-hop lookup prompt. The host had background CPU load during
these runs.

| Prompt tokens | Passes | Prefill before → after (tok/s) | Decode before → after (tok/s) |
| --- | ---: | ---: | ---: |
| 46 | 3 | 6.78 → 6.55 | 5.32 → 5.15 |
| 46 | 10 | 1.77 → 2.10 | 1.63 → 1.85 |
| 188 | 3 | 4.13 → 5.27 | 3.49 → 3.81 |
| 188 | 10 | 1.33 → 1.55 | 1.16 → 1.24 |

A separate eight-token decode trace counted 337 → 241 synchronous Metal
submissions per token at 3 passes and 1,093 → 829 at 10 passes. In an
adaptive smoke test with a 10-pass limit and Δ = 0.15, the engine backfilled
16 passes and generated `Paris.`.

### Initial resident Metal parity

The resident path matched the staged engine at all 68 output positions
across the 46-token story and 188-token six-hop lookup prefixes, with 3 and
10 passes on the research checkpoint. Each run compared all 200,058 logits
at the end of prefill and after each of 16 identical decode steps. The
maximum absolute logit difference was `1.01e-4`. Cache row counts and
positions agreed at every layer visit after prefill and after decode. The
largest absolute differences were `5.70e-4` for K and `5.20e-4` for V.
The focused Metal specs also check an overflowing sliding window, continued
decode after prefill, and cache reuse after reset.

### Matched prefill tiles and Q8_0 projections

A repeat on the 188-token six-hop prefix used six matching Metal tiles
of 32, 32, 32, 32, 32, and 28 tokens in both engines. At 3 and 10 passes
the original resident path achieved 59.90 and 20.82 tok/s, and the
pinned llama.cpp reference achieved 127.96 and 39.87 tok/s. Giving
llama.cpp a single 188-token batch raised its throughput by about 2.1×.

The M2 Max GPU profile at 3 passes attributed 46% of the GPU command
time to MoE, 43% to attention projections, and 5.5% to attention and KV
work. This diagnostic split compute encoders at stage boundaries.

On Apple Silicon, Q8_0 projections with at least 16 input rows use
SIMD matrix multiplication. Each threadgroup stages dequantized weight
blocks and activations in float32 and accumulates a 32-by-32 output tile
in float32. In an interleaved A/B run on a busy host, median prefill
time fell from 4.97 to 3.59 seconds at 3 passes and from 11.78 to
9.28 seconds at 10 passes. Across both depths the largest absolute
logit difference was `8.48e-5` and the largest absolute K or V
difference was `7.43e-4`. All 34 top-ranked tokens matched, as did
all cache row counts and positions after prefill and forced decode.

The final matched-tile comparison reached 82.88 and 30.29 tok/s for
cogni-ml at 3 and 10 passes. The local llama.cpp checkout at
`e6d624b9` reached 141.11 and 48.25 tok/s. Its 10-pass graph required
increasing the node reserve from 22,528 to 44,032 in the benchmark build.

### Grouped expert GEMM

The initial grouped kernels bucketed routes on the GPU for 16–32-token
prefill tiles with up to 256 experts. The IQ4_NL and Q8_0 gate/up and down
GEMMs shared the route map and each dequantized weight tile across up to
16 routes for one expert.

On the same 188-token research-checkpoint case and six tiles, an interleaved
A/B comparison on the M2 Max gave the following medians of three timed runs
per path. The comparison kept the Q8_0 projection path active on both sides.

| Passes | Per-route matvec (tok/s) | Grouped GEMM (tok/s) | Throughput gain |
| ---: | ---: | ---: | ---: |
| 3 | 80.42 | 92.95 | 15.6% |
| 10 | 29.98 | 33.92 | 13.1% |

All 17 top-ranked tokens at each depth matched across 200,058 logits per
position. Cache row counts and positions matched at every layer visit
after prefill and after the 16 forced decode steps. At 3 passes, maximum
absolute differences were `1.00e-4` for logits, `4.79e-4` for K and
`3.34e-4` for V. At 10 passes, they were `6.72e-4`, `3.65e-3` and
`4.75e-3`, respectively.

## Verification

On Apple M2 Max, the CLI completed `The capital of France is` with `Paris.`.
For fixed depth, the full-vocabulary comparison below used English and Russian
prefixes for Mini and the English prefix for the research checkpoint. It
included continuation tokens and used float32 attention caches.

| Checkpoint and depth | Matching top-ranked token | Highest per-position logit RMSE |
| --- | ---: | ---: |
| Mini, 3 passes | 20 / 20 | 6.18 × 10⁻⁶ |
| Research, 10 passes | 8 / 8 | 5.10 × 10⁻⁶ |

The reference was
[llama.cpp at `29b52d39`](https://github.com/ggml-org/llama.cpp/tree/29b52d398b294509746ec0a3f28948a3107c0213).
Its 10-pass run used a larger graph allocation to fit 132 layer visits.

In the adaptive example above, the six evaluated positions exited after
4, 4, 6, 6, 9, and 7 passes. Each reported delta matched the corresponding
fixed-depth trace. Including 16 deferred passes, the core ran 52 times
compared with 60 for fixed depth. At a threshold of 0.08, the last two
positions reached the cap and deferred work brought the total back to 60.

The Metal check for `872baf18` compared the block4 output fusion with
the unfused path on 1,022 prompt tokens and 48 forced decode steps at
three passes. With Metal API Validation enabled, all 49 full-vocabulary
logit rows and live KV at 13 checkpoints matched bitwise across 48
layer visits. The check included retaining one through four rows of a
block, rollback across the physical ring boundary, and the next scalar step.

Run the focused CPU and Metal specs from the repository root.

```sh
crystal spec -Dcpu_only spec/inkling2_*_spec.cr
make build/bridge.o
MTL_DEBUG_LAYER=1 crystal spec spec/inkling2_*_spec.cr \
  --link-flags="-fuse-ld=/usr/bin/ld $PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
```
