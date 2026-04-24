# Qwen 3.5 Prompt KV Cache

Status: design contract, not implemented.

## Goal

Reuse expensive prompt prefixes by storing the prompt text, token ids, and the
post-prefill Qwen decode state. A later session can find an exact cache hit by
`session_id` or by prompt/model hash, restore the state, and continue decoding
without replaying the prefix.

This is different from routed long-context KV offload:

- **Prompt KV cache**: exact prefix reuse. Lookup by identity/hash. No quality
  trade-off.
- **Warm KV offload / routed recall**: optional future long-context memory.
  Lookup by sketch/vector relevance. Requires quality evaluation.

## Why This Matters

Current Qwen native prefill is token-by-token decode. A 64-token prompt benchmark
measured about `44 tok/s`, while llama.cpp prefill was about `449 tok/s` on the
same local run. Decode is already roughly competitive, so replaying repeated
system/tool prompts is one of the largest avoidable costs.

The existing `Qwen35CPU::State#fork` / `copy_from!` primitives prove that the
state can already be deep-copied across CPU arrays and Metal buffers. The missing
piece is durable serialization plus an index.

## Cache Key

A cache entry is valid only when all compatibility fields match:

- `model_id`: stable model identity, preferably `sha256(GGUF metadata + tensor inventory)`.
- `model_path` and `model_mtime_ns`: diagnostic only, not sufficient by itself.
- `tokenizer_id`: tokenizer/model hash; must match token ids exactly.
- `runtime_id`: cogni-ml cache format version and Qwen state layout version.
- `quant_id`: GGUF quant/type inventory hash.
- `prompt_hash`: `sha256("qwen35-prompt-v1\\0" + token_count:u32le + token_ids:i32le[] + prompt bytes)`.
- `token_hash`: `sha256("qwen35-token-v1\\0" + prefix_len:u32le + token_ids[0...prefix_len]:i32le[])`.
- `prefix_len`: number of tokens represented by the restored state.
- `max_seq`: allocated state capacity; restore target must be compatible.

Recommended primary lookup order:

1. `(session_id, turn_id or prefix_seq)` for active session resume.
2. `(model_id, tokenizer_id, prompt_hash, prefix_len)` for exact cross-session prompt reuse.
3. Longest-prefix match by `(model_id, tokenizer_id, token_hash, prefix_len DESC)`; replay only the suffix.

## What Must Be Serialized

For Qwen 3.5/3.6 `arch=qwen35`, a prompt state is not just KV cache.

Per layer:

- Full-attention layers:
  - `k_cache` / `k_cache_buf`
  - `v_cache` / `v_cache_buf`
  - valid positions `0...prefix_len`
- Recurrent layers:
  - `conv_state` / `conv_state_buf`
  - `ssm_state` / `ssm_state_buf`
- Metadata:
  - layer index
  - state kind
  - dtype and byte order
  - logical shape
  - valid token range

The serializer should read from whichever backend owns the state:

- If a `MetalBuffer` exists, copy bytes from shared `contents` or `read_bytes`.
- Otherwise copy from the CPU `Array(Float32)`.

The restore path should prefer GPU-resident buffers when Metal is available.

## Artifact Format

Use a local binary artifact for large state bytes and PostgreSQL for searchable
metadata. Do not store multi-megabyte state payloads directly in hot query rows
unless a benchmark proves TOAST overhead is acceptable for the target size.

Current local file layout:

```text
.cache/cogni-ml/qwen35-kv-cache/
  manifest.jsonl
  artifacts/
    <sha256(model_id + tokenizer_id)[0..16]>/
      <prefix_len>-<prompt_hash>.qkv
```

Current artifact header:

```text
magic:      "CQKV"
version:    u32 = 1
max_seq:    u32
n_layer:    u32
record_count:u32
positions:  n_layer * u32
records:    repeated { layer:u32, kind:u8, storage_mode:u8, reserved:u16, byte_len:u64, payload:bytes }
```

The `.qkv` artifact intentionally contains only state layout bytes. The
manifest owns model/tokenizer/session/prompt compatibility metadata and stores
the artifact SHA-256. Restore validates the SHA before decoding the artifact.
Default dtype for current Qwen state is `f32`. `f16`/quantized KV is a separate
format version because it changes numerical restore semantics.

## PostgreSQL Metadata Schema

`pg_sorted_heap` is useful for exact lookup and ordered session scans. Keep the
large artifact external first; store the artifact path and hashes in PG.

The checked-in migration is `docs/sql/qwen35_prompt_cache_pg_sorted_heap.sql`.
The same schema can be generated at runtime with
`ML::GGUF::Qwen35PromptCache.pg_sorted_heap_schema_sql`, and the parameterized
upsert statement with `ML::GGUF::Qwen35PromptCache.pg_insert_sql`.

```sql
CREATE EXTENSION IF NOT EXISTS pg_sorted_heap;

CREATE TABLE IF NOT EXISTS qwen35_prompt_cache (
    cache_id           bigserial PRIMARY KEY,
    runtime_id         text NOT NULL,
    session_id         text NOT NULL,
    turn_id            text,
    model_id           text NOT NULL,
    tokenizer_id       text NOT NULL,
    prompt_hash        text NOT NULL,
    token_hash         text NOT NULL,
    prefix_len         integer NOT NULL CHECK (prefix_len >= 0),
    max_seq            integer NOT NULL CHECK (max_seq >= prefix_len),
    layer_count        integer NOT NULL CHECK (layer_count > 0),
    artifact_path      text NOT NULL,
    artifact_sha256    text NOT NULL CHECK (length(artifact_sha256) = 64),
    artifact_byte_size bigint NOT NULL CHECK (artifact_byte_size >= 0),
    state_byte_size    bigint NOT NULL CHECK (state_byte_size >= 0),
    created_at_unix    bigint NOT NULL,
    prompt_preview     text
) USING sorted_heap;

CREATE UNIQUE INDEX IF NOT EXISTS qwen35_prompt_cache_exact_idx
    ON qwen35_prompt_cache (model_id, tokenizer_id, prompt_hash, prefix_len);

CREATE INDEX IF NOT EXISTS qwen35_prompt_cache_session_idx
    ON qwen35_prompt_cache (session_id, turn_id, prefix_len, created_at_unix DESC);

CREATE INDEX IF NOT EXISTS qwen35_prompt_cache_prefix_idx
    ON qwen35_prompt_cache (model_id, tokenizer_id, token_hash, prefix_len DESC);
```

If the installed `pg_sorted_heap` build uses the legacy `clustered_heap` table AM
name, use that name in the migration. The current public docs show `USING
sorted_heap`; older regression SQL still contains `USING clustered_heap`.

## Lookup Flow

The generator path is opt-in:

```bash
QWEN35_PROMPT_CACHE=1 crystal run bin/qwen35_generate.cr -- "The capital of France is" 8
```

Optional controls:

- `QWEN35_PROMPT_CACHE_ROOT`: cache root; defaults to `~/.cache/cogni-ml/qwen35-kv-cache`.
- `QWEN35_SESSION_ID` / `QWEN35_TURN_ID`: metadata for session grouping.
- `QWEN35_PROMPT_CACHE_PREVIEW=1`: store decoded prompt preview; off by default to avoid leaking raw prompts into the manifest.

For greedy generation, the CLI stores the state before the last prompt token.
On a hit it restores the longest prefix and replays at least the last prompt
token, which regenerates the next-token logits without mutating a completed
prompt state twice.

```text
input prompt
  -> tokenize
  -> compute prompt_hash/token_hash
  -> query exact prompt hash first
  -> verify model/tokenizer/state_format/prefix_len
  -> verify artifact sha256
  -> deserialize state into Qwen35CPU::State
  -> continue decode from pos=prefix_len
```

On an exact miss with a prefix hit:

```text
input prompt
  -> choose longest compatible token_hash prefix
  -> verify artifact sha256
  -> deserialize prefix state
  -> replay tokens[prefix_len...prompt_len]
  -> continue decode from pos=prompt_len
```

On a miss:

```text
input prompt
  -> normal prefill
  -> serialize post-prefill state
  -> insert metadata row
  -> continue decode
```

## Correctness Gates

A cache restore is valid only if these pass:

1. For a fixed prompt and next token, `forward_top1` after restore equals
   `forward_top1` after live prefill: same token id and logit tolerance `1e-4`.
2. Full-logit cosine after restore is `>= 0.999999` on at least one test prompt.
3. Restored buffers do not alias the source state buffers.
4. Artifact hash mismatch fails closed and recomputes prefill.
5. Model/tokenizer/hash mismatch fails closed and recomputes prefill.
6. Longest-prefix restore plus suffix replay matches live full prefill.

## Implementation Slices

1. `Qwen35StateSnapshot` in memory:
   - convert `State` to typed byte records
   - restore records into a new `State`
   - spec: live prefill vs snapshot restore next-token equality
   - status: implemented in `src/ml/gguf/qwen35_state_snapshot.cr`

2. File artifact:
   - write/read `.qkv` with header and SHA-256
   - spec: round-trip artifact equality and fail-closed corrupt artifact
   - status: implemented in `src/ml/gguf/qwen35_state_snapshot.cr`

3. Local index without PG:
   - JSONL or SQLite-style manifest for development
   - avoids coupling correctness work to DB setup
   - status: implemented in `src/ml/gguf/qwen35_prompt_cache.cr`

4. PostgreSQL adapter:
   - optional shard or CLI adapter controlled by env DSN
   - no hard dependency for core `cogni-ml`
   - migration SQL for `pg_sorted_heap`
   - status: SQL generation and migration implemented; live DB execution deferred

5. Longest-prefix cache:
   - token rolling hash chain
   - choose longest compatible prefix, replay only suffix
   - status: implemented for local manifest with exact `token_hash`

6. Routed warm-tier KV blocks:
   - separate project after exact prompt cache is verified
   - use FlashHadamard/sketch search only with eval harness

## Risk Notes

- Restoring only full-attention KV is incorrect for Qwen; recurrent `conv_state`
  and `ssm_state` are required.
- Cross-model cache reuse must be forbidden unless model/tokenizer hashes match.
- Storing raw prompt text may leak private data. Make prompt text optional and
  store `prompt_preview` only if explicitly enabled.
- Do not use approximate sketch retrieval for exact prompt reuse.
- Large `bytea` rows may work, but external artifacts keep PG lookup latency and
  table maintenance predictable.
