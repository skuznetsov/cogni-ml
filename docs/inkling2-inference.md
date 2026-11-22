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

## Fixed-depth Metal execution contract

Fixed-depth decode submits one Metal command per token. Prefill evaluates a
bounded tile of consecutive tokens together at every layer visit and reads
logits for the last token. The same per-visit KV and convolution histories
continue into decode. Each new query reads old history and the causal part
of its tile before the tile can overwrite the sliding cache.

Use `--prefill-tile 64` or `--prefill-tile 128` to test wider tiles with the
same float32 staging as the 32-token default. This option accepts 1 through
128 rows and requires the resident Metal backend. Tiles from 16 through
128 rows use grouped expert GEMM on Apple Silicon when at most 256 experts
are present. The tile height is a runtime dispatch parameter. The Metal
pipeline cache keys compiled kernels by function name. Default tile promotion
requires warmed wall time measurements under the same conditions at 3 and
10 passes.

For a single query with more than 64 visible keys, attention splits the
history across four SIMD groups per head and combines their softmax
partials before applying the output gate.

The staged backend remains available for adaptive depth, traces, and
checkpoint layouts outside the resident kernel bounds. Backend selection
occurs before the sequence starts. Validate resident execution against
the staged backend with the same forced IDs, including a cache wrap and
a continuation after prefill. The real-checkpoint gate compares every
logit and each visit's retained KV with 3 and 10 passes.

## Checkpoints used for verification

| Checkpoint | Hugging Face revision | SHA-256 |
| --- | --- | --- |
| Inkling-2-Mini-IQ4_NL.gguf | `db61c5b492a6a034d986328d28274af0a439aa6c` | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |
| TML-d24-varloop-research-IQ4_NL.gguf | `618c48f1e3c54094333e3e652a92b1d99c52e34d` | `e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2` |

The GGUF layout uses IQ4_NL for the early routed experts, Q8_0 for dense
projections and the last four layers' experts, and F32 for the loop controllers
and router. The loader checks loop metadata and tensor layout and extents
before decode.

## Native Metal MoE comparison

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

## Resident Metal comparison

The resident path matched the staged engine at all 68 output positions
across the 46-token story and 188-token six-hop lookup prefixes, with 3 and
10 passes on the research checkpoint. Each run compared all 200,058 logits
at the end of prefill and after each of 16 identical decode steps. The
maximum absolute logit difference was `1.01e-4`. Cache row counts and
positions agreed at every layer visit after prefill and after decode. The
largest absolute differences were `5.70e-4` for K and `5.20e-4` for V.
The focused Metal specs also check an overflowing sliding window, continued
decode after prefill, and cache reuse after reset.

On an Apple M2 Max, both backends used this same GGUF, float32 KV,
and a 256-token context. For each case, cogni-ml and llama.cpp ran
consecutively, with the same 16 forced decode inputs. Each figure is the
median of three timed repetitions after warmup. The host had background
CPU load. Cogni-ml used tiles of up to 32 tokens and llama.cpp used one
batch per prompt.

| Prompt tokens | Passes | Cogni-ml prefill (tok/s) | llama.cpp prefill (tok/s) | Cogni-ml decode (tok/s) | llama.cpp decode (tok/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 46 | 3 | 61.38 | 146.38 | 15.23 | 18.70 |
| 46 | 10 | 18.98 | 48.72 | 5.73 | 7.49 |
| 188 | 3 | 52.52 | 303.74 | 15.98 | 26.89 |
| 188 | 10 | 16.72 | 81.49 | 6.04 | 6.47 |

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

On Apple Silicon, resident prefill tiles of 16–32 tokens with up to 256
experts bucket the selected routes on the GPU. The IQ4_NL and Q8_0 gate/up
and down GEMMs share the route map and reuse each dequantized weight tile
across up to 16 routes for one expert. The kernels scatter their output
back to the original token and top-k rank before the weighted sum.

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

Two local direct-answer lookup examples with five and nine hops scored zero
correct at both fixed depths, three and ten. The last prompt token's delta
decreased on both examples.

Run the focused CPU and Metal specs from the repository root.

```sh
crystal spec -Dcpu_only spec/inkling2_*_spec.cr
make build/bridge.o
crystal spec spec/inkling2_*_spec.cr \
  --link-flags="-fuse-ld=/usr/bin/ld $PWD/build/bridge.o -framework Metal -framework Foundation -lc++"
```
