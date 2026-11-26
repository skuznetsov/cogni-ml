# Inkling 2 Mini serving

## Run on Apple Silicon

```sh
make build/inkling2_serve
./build/inkling2_serve --model /path/to/Inkling-2-Mini-IQ4_NL.gguf
```

Set the client's base URL to `http://127.0.0.1:8080/v1` and its model to
`inkling-2-mini`. The server accepts any client key when
`COGNI_INKLING2_API_KEY` is unset. Set that variable on the server and use the
same bearer key in the client to enable authentication. Set the listen address,
port, context, or advertised model ID with `--host`, `--port`, `--context`,
and `--model-id`. The default context is 16,384 tokens, shared by prompt and
output.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
with client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="none",
    max_completion_tokens=256,
    stream=True,
) as stream:
    for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Defaults

The generator and server select the fast prefill profile on the supported
three-pass Metal layout. It uses 2,048-row outer tiles, vector Flash attention,
FP32 MMA loop controllers, and grouped MoE with FP16 staging. Persistent KV and
matrix accumulation stay in FP32. The profile uses a different reduction
order and staging precision from the ordered reference. Select
`--profile ordered` for the ordered FP32 prefill path.

Requests that omit `temperature` use the same greedy path as `temperature=0`.
On three-pass Metal, this path uses GPU top-1 and verifies exact n-gram drafts
when it finds a repeated continuation. The verifier commits the prefix
accepted by the target and falls back to scalar decode after a low-acceptance
block. Use `--no-ngram` to force scalar greedy decode. A positive temperature
uses full target logits with the requested `top_p` and optional seed.

### Native DSpark

Pass `--dspark` to the server or the standalone generator to use the Inkling
2 Mini DSpark checkpoint on fixed three-pass resident Metal.

```sh
./build/inkling2_serve --model /path/to/Inkling-2-Mini-IQ4_NL.gguf \
  --dspark /path/to/mtp.safetensors
```

Place the draft checkpoint's `config.json` alongside `mtp.safetensors`.
DSpark uses the same HTTP endpoints and replaces n-gram drafting in this
mode. Requests with positive temperature or a generation limit below
eight tokens use the scalar path. Other requests start with two scalar
steps, then verify blocks of up to four rows. The controller tracks draft
and verification time against the cost of the confirmed rows in scalar
decode. It retains savings worth up to two scalar steps and returns to
scalar for the rest of the request when the accumulated loss exceeds the
cost of two scalar steps. Only target-confirmed tokens are emitted.
Prefill and accepted decode tokens supply up to 1,023 recent context
positions to the draft model.

Quiet-host M2 Max measurements used three passes, the research IQ4_NL
target, and 49 greedy output tokens per run. All DSpark runs produced the
same token IDs as scalar decode. Decode timing excludes prefill and
includes rejected work and fallback.

| Prompt | Tokens | Scalar (tok/s) | DSpark (tok/s) |
| --- | ---: | ---: | ---: |
| Python | 104 | 34.82 | 44.00 |
| Python | 1,022 | 30.38 | 32.22 |
| Prose | 81 | 34.66 | 32.48 |

The short prompts report medians from two quiet pairs each. The long
Python result comes from one quiet pair.

## Memory

The server maps one GGUF and reuses Metal scratch buffers and KV storage
across requests. Reset restores the sequence to position zero while
retaining the allocated capacity. The fast profile retained 482 MiB of
scratch at the 2,048-row limit on M2 Max. Global KV capacity grows in
powers of two up to `--context`. For example, the default context can
reserve 16,384 rows once the total sequence exceeds 8,192 tokens.

In a single-server run on M2 Max, an 8,192-token prompt warmed the KV cache.
The physical footprint then settled around 2,018 MiB during 256 identical
48-token prompts with four generated tokens each, alternating JSON and SSE.
System swap counters stayed flat during the run.

## HTTP contract

| Method and path | Response |
| --- | --- |
| `GET /v1/models` | The loaded model |
| `GET /v1/models/{id}` | The model description |
| `POST /v1/chat/completions` | Assistant text from a message history |
| `POST /v1/completions` | Completion of a string or one array of token IDs |

Both completion endpoints return JSON by default and Server-Sent Events
for `stream: true`. Streams end with `[DONE]` and can include final token
usage with `stream_options.include_usage`. The chat path renders text parts,
accepts system, developer, user, and assistant messages, and counts sampled
reasoning tokens in completion usage and limits. The checkpoint's default
reasoning effort is `high`, and it can emit reasoning tokens even with
`reasoning_effort="none"`. The server loads one GGUF and runs one sequence at a time.
Each request starts with a fresh sequence.

The API scope is text and one choice. Requests for tool calls, multimodal
output, or JSON-schema output are rejected. Configure harnesses to use
Chat Completions or legacy text Completions. See the official request and
response shapes in the
[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Loopback measurements

On an M2 Max, one warm server completed fixed token-ID prompts with the
Mini IQ4_NL GGUF, default settings, and 49 generated tokens.

| Prompt tokens | Time to first token | Decode after first token | Requests |
| ---: | ---: | ---: | ---: |
| 1,024 | 1.75 s | 29.6 tok/s | 1 |
| 8,192 | 13.25 s | 25.8 tok/s | 3, median |

Time to first token includes the HTTP request and prompt evaluation. These
are loopback observations under normal host load. Engine prefill benchmarks
are in [Inkling 2 inference](inkling2-inference.md#fast-prefill-profile).

### Greedy decode with n-gram drafts

On the same M2 Max, the Mini IQ4_NL GGUF ran at three passes with fast
prefill. The OpenAI SDK streamed completions at `temperature=0` with and
without `--no-ngram`. These are medians of four requests per setting with
one warm server at a time. Decode timing starts at the first visible token
and runs through the finish event, including any verification and fallback.

| Continuation | Prompt tokens | Generated tokens | `--no-ngram` | Default n-gram |
| --- | ---: | ---: | ---: | ---: |
| Prose | 81 | 49 | 33.91 tok/s | 33.79 tok/s |
| Repeated text | 104 | 97 | 33.86 tok/s | 53.65 tok/s |
| Repeated text across the 1,024-token window | 1,022 | 97 | 29.99 tok/s | 50.68 tok/s |

For each prompt, both settings returned identical text, completion token
counts, and finish reasons. The repeated-text pairs ran under normal host
load. The two prose pairs that passed the quiet-host check also ran at
about 34 tok/s in both settings.

## Verification gates

- Compare the fast profile with the measured snapshot on the same weight file, prompt IDs and depth. Check final logits, active KV and convolution histories. Exercise the 1,024-token ring boundary and the 2,048-row outer tile, including successive requests with different lengths.
- Check that the ordered override retains the previous resident path. Keep the fast profile separable at construction so a regression has a runtime rollback.
- Exercise model discovery, streaming and nonstreaming chat, raw prompts, stop sequences, request errors and sequence reset. Use the unmodified OpenAI SDK with a loopback base URL on the real model.
- Attribute measured speed to prompt length, backend, and accepted draft count. A stream counts only committed tokens. The greedy reference verifies the same tokens even when the draft is rejected.
