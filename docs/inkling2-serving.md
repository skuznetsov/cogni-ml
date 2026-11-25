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

Temperature defaults to zero. The greedy path uses GPU top-1 and verifies
exact n-gram drafts when it finds a repeated continuation. The verifier
commits the prefix accepted by the target and falls back to scalar decode
after a low-acceptance block. Use `--no-ngram` to force scalar greedy decode.
A positive temperature uses full target logits and the requested
`top_p` and optional seed.

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

## Verification gates

- Compare the fast profile with the measured snapshot on the same weight file, prompt IDs and depth. Check final logits, active KV and convolution histories. Exercise the 1,024-token ring boundary and the 2,048-row outer tile, including successive requests with different lengths.
- Check that the ordered override retains the previous resident path. Keep the fast profile separable at construction so a regression has a runtime rollback.
- Exercise model discovery, streaming and nonstreaming chat, raw prompts, stop sequences, request errors and sequence reset. Use the unmodified OpenAI SDK with a loopback base URL on the real model.
- Attribute measured speed to prompt length, backend, and accepted draft count. A stream counts only committed tokens. The greedy reference verifies the same tokens even when the draft is rejected.
