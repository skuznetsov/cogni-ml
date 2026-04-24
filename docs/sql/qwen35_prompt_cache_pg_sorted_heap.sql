-- Qwen 3.5/3.6 exact prompt-cache metadata for pg_sorted_heap.
--
-- State bytes stay in external .qkv artifacts. PostgreSQL stores only lookup
-- metadata and SHA-256 compatibility checks.

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
