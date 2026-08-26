# Bounded multi-turn quality and latency probe for cold QBit state restored
# directly into adaptive resident GPU KV.

require "json"
require "option_parser"

require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_native_runtime"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_proposal_route"
require "../src/ml/gguf/qwen35_qbit_runtime_cache"
require "../src/ml/gguf/qwen_qbit_quality_metrics"
require "../src/ml/gguf/qwen_qbit_state_snapshot"

DEFAULT_QWEN38_SESSION_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_SESSION_RESIDENT_MAP = "p4;27=bf16,43=bf16,47=bf16,51=bf16"

record SessionTokenECS,
  mean : Float64,
  minimum : Float64,
  compared : Int32,
  exact_count : Int32,
  candidate_count : Int32

private def release_session_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  ML::Metal::Device.synchronize if ML::Metal::Device.available?
  state.layers.each do |layer|
    layer.k_cache_buf.try(&.release)
    layer.v_cache_buf.try(&.release)
    layer.conv_state_buf.try(&.release)
    layer.ssm_state_buf.try(&.release)
    layer.adaptive_kv.try do |cache|
      begin
        cache.release
      rescue ex
        STDERR.puts "adaptive cache cleanup skipped: #{ex.message}"
      end
    end
    layer.k_cache_buf = nil
    layer.v_cache_buf = nil
    layer.conv_state_buf = nil
    layer.ssm_state_buf = nil
    layer.adaptive_kv = nil
  end
end

private def with_session_adaptive_env(map : String?, &)
  keys = [
    "QWEN35_ADAPTIVE_RESIDENT_KV_LAYER",
    "QWEN35_ADAPTIVE_RESIDENT_KV_TIER",
    "QWEN35_ADAPTIVE_RESIDENT_KV_MAP",
    "QWEN35_PREFILL_APPEND_CMD_OFF",
    "QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF",
    "QWEN35_PREFILL_FUSE_FULL_REC_OFF",
    "QWEN35_FULL_PREFILL_CHUNK_OFF",
    "QWEN35_PREFILL_REC_RUN_OFF",
    "QWEN35_PREFILL_CHUNK_OFF",
    "QWEN35_PREFILL_CHUNK_SIZE",
    "QWEN35_PREFILL_FINAL_CHUNK_OFF",
    "QWEN35_PREFILL_LONG_SUFFIX_OFF",
  ]
  old = keys.to_h { |key| {key, ENV[key]?} }
  keys.each { |key| ENV.delete(key) }
  ENV["QWEN35_ADAPTIVE_RESIDENT_KV_MAP"] = map if map
  yield
ensure
  old.try &.each do |key, value|
    if value
      ENV[key] = value
    else
      ENV.delete(key)
    end
  end
end

private def stop_session_token?(tokenizer : ML::GGUF::Qwen35Tokenizer,
                                token_id : Int32) : Bool
  token_id == tokenizer.eos_id || token_id == tokenizer.pad_id ||
    tokenizer.token_to_id["<|im_end|>"]? == token_id
end

private def verify_session_resident!(state : ML::GGUF::Qwen35CPU::State,
                                     hp : ML::GGUF::Qwen35Hparams,
                                     expected_tokens : Int32) : Nil
  unless state.adaptive_kv_layer_indices == hp.full_attention_layers
    raise "cold adaptive restore does not own every full-attention layer"
  end
  hp.full_attention_layers.each do |layer_index|
    layer = state.layers[layer_index]
    if layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf
      raise "cold adaptive restore retained a Float32 KV owner at layer #{layer_index}"
    end
    cache = layer.adaptive_kv.not_nil!
    unless cache.cache_len == expected_tokens
      raise "cold adaptive restore layer #{layer_index} has #{cache.cache_len} tokens, expected #{expected_tokens}"
    end
  end
end

private def session_token_ecs(output_weight : ML::GGUF::QuantWeight,
                              exact_ids : Array(Int32),
                              candidate_ids : Array(Int32)) : SessionTokenECS
  compared = Math.min(exact_ids.size, candidate_ids.size)
  raise "session token ECS requires at least one aligned position" if compared == 0

  cache = {} of Int32 => Array(Float32)
  total = 0.0_f64
  minimum = 1.0_f64
  compared.times do |index|
    exact_id = exact_ids[index]
    candidate_id = candidate_ids[index]
    cosine = if exact_id == candidate_id
               1.0_f64
             else
               exact = cache[exact_id]? || begin
                 row = ML::GGUF::Qwen35CPU.embedding_lookup(output_weight, exact_id)
                 cache[exact_id] = row
                 row
               end
               candidate = cache[candidate_id]? || begin
                 row = ML::GGUF::Qwen35CPU.embedding_lookup(output_weight, candidate_id)
                 cache[candidate_id] = row
                 row
               end
               ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(exact, candidate)
             end
    total += cosine
    minimum = Math.min(minimum, cosine)
  end
  SessionTokenECS.new(
    total / compared,
    minimum,
    compared.to_i32,
    exact_ids.size.to_i32,
    candidate_ids.size.to_i32,
  )
end

private def session_meaning_preserved?(text : String) : Bool
  normalized = text.downcase
  normalized.includes?("95") && normalized.includes?("sum")
end

private def session_filler(repetitions : Int32) : String
  String.build do |io|
    repetitions.times do |index|
      marker = index + 1
      io << "Archive marker " << marker << " is neutral context and does not change the variables.\n"
    end
    io << "Remember the private variables alpha = 37 and beta = 58."
  end
end

private def prefill_session_suffix_top1(
  weights : ML::GGUF::Qwen35Weights,
  token_ids : Array(Int32),
  start_pos : Int32,
  state : ML::GGUF::Qwen35CPU::State,
  chunks : Array(Int32),
) : {Int32, Float32}
  offset = 0
  chunks[0...-1].each do |chunk_size|
    ML::GGUF::Qwen35CPU.prefill_tokens(
      weights,
      token_ids[offset, chunk_size],
      start_pos + offset,
      state,
    )
    offset += chunk_size
  end
  final_chunk = chunks[-1]
  ML::GGUF::Qwen35CPU.prefill_tokens_top1(
    weights,
    token_ids[offset, final_chunk],
    start_pos + offset,
    state,
  )
end

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_QWEN38_SESSION_MODEL
resident_map = DEFAULT_SESSION_RESIDENT_MAP
filler_repetitions = 48_i32
suffix_filler_repetitions = 0_i32
replay_chunk_tokens = ML::GGUF::Qwen35NativeRuntime::ADAPTIVE_SESSION_REPLAY_CHUNK_TOKENS
n_gen = 48_i32
attribute_replay = false
clickhouse_endpoint = nil.as(String?)
clickhouse_table_prefix = "qwen_adaptive_session_restore"

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_adaptive_session_restore_probe [options]"
  parser.on("--model PATH", "Qwen GGUF path") { |value| model_path = value }
  parser.on("--resident-map MAP", "Resident adaptive tier map") { |value| resident_map = value }
  parser.on("--filler N", "Neutral anchor filler repetitions") { |value| filler_repetitions = value.to_i32 }
  parser.on("--suffix-filler N", "Neutral replay-suffix filler repetitions") { |value| suffix_filler_repetitions = value.to_i32 }
  parser.on("--replay-chunk N", "Model replay chunk size: legacy 64, diagnostic 80, or production 4096") { |value| replay_chunk_tokens = value.to_i32 }
  parser.on("--attribute-replay", "Compare single-span and identically chunked exact replay") { attribute_replay = true }
  parser.on("--gen N", "Maximum response tokens") { |value| n_gen = value.to_i32 }
  parser.on("--clickhouse URL", "Persist and cold-read the admitted state through ClickHouse") { |value| clickhouse_endpoint = value }
  parser.on("--table-prefix NAME", "Isolated ClickHouse table prefix") { |value| clickhouse_table_prefix = value }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

raise "model does not exist: #{model_path}" unless File.file?(model_path)
raise "--resident-map cannot be empty" if resident_map.strip.empty?
raise "--filler must be within 0..256" unless filler_repetitions.in?(0..256)
raise "--suffix-filler must be within 0..256" unless suffix_filler_repetitions.in?(0..256)
raise "--replay-chunk must be 64, 80, or 4096" unless replay_chunk_tokens.in?(64, 80, 4096)
raise "--gen must be within 8..128" unless n_gen.in?(8..128)
unless clickhouse_table_prefix.matches?(/\A[A-Za-z_][A-Za-z0-9_]*\z/)
  raise "--table-prefix is not a safe ClickHouse identifier"
end
raise "Metal is unavailable" unless ML::GGUF::Qwen35Metal.available?

startup_started = Time.instant
gguf = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path)
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
startup_ms = (Time.instant - startup_started).total_milliseconds
hp = weights.hparams

anchor_messages = [
  ML::GGUF::Qwen35Chat::Message.new(
    "system",
    "Track the supplied variables exactly and answer the later arithmetic request concisely.",
  ),
  ML::GGUF::Qwen35Chat::Message.new("user", session_filler(filler_repetitions)),
  ML::GGUF::Qwen35Chat::Message.new(
    "assistant",
    "Noted. Alpha is 37 and beta is 58; the neutral archive markers do not alter them.",
  ),
]
full_messages = anchor_messages + [
  ML::GGUF::Qwen35Chat::Message.new(
    "user",
    String.build do |io|
      if suffix_filler_repetitions > 0
        io << session_filler(suffix_filler_repetitions) << '\n'
      end
      io << "Reply with one short sentence containing the integer result and the words 'their sum'. What is alpha + beta?"
    end,
  ),
]
anchor_text = ML::GGUF::Qwen35Chat.render(
  anchor_messages,
  add_generation_prompt: false,
  enable_thinking: false,
)
full_text = ML::GGUF::Qwen35Chat.render(
  full_messages,
  add_generation_prompt: true,
  enable_thinking: false,
)
anchor_ids = tokenizer.encode(anchor_text)
full_ids = tokenizer.encode(full_text)
raise "anchor tokenization is empty" if anchor_ids.empty?
unless full_ids.size > anchor_ids.size && full_ids[0, anchor_ids.size] == anchor_ids
  raise "rendered multi-turn anchor is not an exact token prefix"
end
suffix_ids = full_ids[anchor_ids.size, full_ids.size - anchor_ids.size]
replay_chunks = ML::GGUF::Qwen35NativeRuntime.adaptive_session_replay_chunks(
  suffix_ids.size.to_i32,
  replay_chunk_tokens,
)
max_seq = full_ids.size + n_gen + 1
if max_seq > ML::GGUF::QwenQBitSessionCheckpoint::MAX_REPLAY_TOKENS
  raise "probe max_seq exceeds the qualified 4096-token adaptive-session boundary"
end

snapshot = nil.as(ML::GGUF::Qwen35StateSnapshot::Snapshot?)
exact_ids = [] of Int32
exact_top2 = [] of ML::GGUF::QwenQBitQualityMetrics::Top2
exact_prefill_ms = 0.0_f64
exact_replay_ms = 0.0_f64
exact_decode_ms = 0.0_f64
exact_ended = false
exact_first = -1_i32
exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
begin
  with_session_adaptive_env(nil) do
    ML::GGUF::Qwen35CPU.prepare_state_metal!(exact_state, hp)
    prefill_started = Time.instant
    ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, anchor_ids, 0, exact_state)
    exact_prefill_ms = (Time.instant - prefill_started).total_milliseconds
  end
  exact_state.layers.each { |layer| layer.position = anchor_ids.size.to_i32 }
  snapshot = ML::GGUF::Qwen35StateSnapshot.capture(exact_state)

  with_session_adaptive_env(nil) do
    replay_started = Time.instant
    exact_first, _exact_first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights,
      suffix_ids,
      anchor_ids.size.to_i32,
      exact_state,
    )
    exact_replay_ms = (Time.instant - replay_started).total_milliseconds
  end
  decode_started = Time.instant
  token = exact_first
  pos = full_ids.size.to_i32
  while exact_ids.size < n_gen
    if stop_session_token?(tokenizer, token)
      exact_ended = true
      break
    end
    exact_ids << token
    break if exact_ids.size >= n_gen
    first, first_logit, second, second_logit = ML::GGUF::Qwen35CPU.forward_top2(
      weights,
      token,
      pos,
      exact_state,
    )
    exact_top2 << ML::GGUF::QwenQBitQualityMetrics::Top2.new(
      first,
      first_logit,
      second,
      second_logit,
    )
    token = first
    pos += 1
  end
  exact_decode_ms = (Time.instant - decode_started).total_milliseconds
ensure
  release_session_state!(exact_state)
end
raise "exact multi-turn response did not produce tokens" if exact_ids.empty?
raise "exact multi-turn response did not produce top-2 steps" if exact_top2.empty?
exact_text = tokenizer.decode(exact_ids)
source_snapshot = snapshot.not_nil!

# This optional control keeps model, anchor, suffix, chunk plan, and exact F32
# ownership constant. Its delta from the single-span exact replay therefore
# attributes chunk-shape plus command publication cost without perturbing the
# production adaptive path with profiler-inserted synchronization points.
exact_chunked_restore_ms = 0.0_f64
exact_chunked_replay_ms = 0.0_f64
exact_chunked_boundary_first = -1_i32
if attribute_replay
  with_session_adaptive_env(nil) do
    restore_started = Time.instant
    exact_chunked_state = ML::GGUF::Qwen35StateSnapshot.restore(source_snapshot, hp)
    exact_chunked_restore_ms = (Time.instant - restore_started).total_milliseconds
    begin
      replay_started = Time.instant
      exact_chunked_boundary_first, _exact_chunked_boundary_logit = prefill_session_suffix_top1(
        weights,
        suffix_ids,
        anchor_ids.size.to_i32,
        exact_chunked_state,
        replay_chunks,
      )
      exact_chunked_replay_ms = (Time.instant - replay_started).total_milliseconds
    ensure
      release_session_state!(exact_chunked_state)
    end
  end
  unless exact_chunked_boundary_first == exact_ids.first
    raise "identically chunked exact replay changed the boundary top-1 token"
  end
end

state_abi = ML::GGUF::QwenQBitCacheEnvelope.state_abi(hp, max_seq)
model_id = ML::GGUF::Qwen35ProposalRoute.model_id(model_path)
tokenizer_id = ML::GGUF::Qwen35ProposalRoute.tokenizer_id(model_id, tokenizer)
cache_context = ML::GGUF::QwenQBitCacheEnvelope::Context.new(
  model_id: model_id,
  tokenizer_id: tokenizer_id,
  template_id: ML::GGUF::QwenQBitCacheEnvelope.template_id(tokenizer.chat_template),
  prompt_hash: ML::GGUF::Qwen35PromptCache.prompt_hash(anchor_ids, anchor_text),
  token_hash: ML::GGUF::Qwen35PromptCache.token_hash(anchor_ids),
  prefix_len: anchor_ids.size.to_i32,
  max_seq: max_seq,
  layer_count: state_abi.layer_count,
  qbit_block_size: ML::GGUF::Qwen35QBitRuntimeCache::BLOCK_SIZE,
  qbit_precision: ML::GGUF::Qwen35QBitRuntimeCache::PRECISION,
  validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
  validation_steps: 1,
  validation_hash: ML::GGUF::Qwen35QBitRuntimeCache.validation_hash(anchor_ids, exact_ids.first),
  next_token_id: exact_ids.first,
  state_abi: state_abi,
)
effective_cache_id = ML::GGUF::QwenQBitCacheEnvelope.cache_id(cache_context)
serialize_started = Time.instant
encoded_state = ML::GGUF::QwenQBitStateSnapshot.encode(
  source_snapshot,
  block_size: 1024,
  precision: 7,
)
native_bytes = ML::GGUF::QwenQBitStateSnapshot.encode_native_recurrent(
  encoded_state,
  effective_cache_id,
)
kv_snapshot = ML::GGUF::Qwen35StateSnapshot::Snapshot.new(
  source_snapshot.max_seq,
  source_snapshot.layer_count,
  Array(Int32).new(source_snapshot.layer_count, anchor_ids.size.to_i32),
  source_snapshot.records.select { |record| record.kind.k_cache? || record.kind.v_cache? },
)
exact_artifact_bytes = ML::GGUF::Qwen35StateSnapshot.encode_artifact_bytes(
  kv_snapshot,
  artifact_live_kv_tokens: anchor_ids.size.to_i32,
)
serialize_ms = (Time.instant - serialize_started).total_milliseconds

clickhouse_save_ms = 0.0_f64
clickhouse_lookup_admit_ms = 0.0_f64
admission = nil.as(ML::GGUF::QwenQBitCacheEnvelope::Admission?)
if endpoint = clickhouse_endpoint
  config = ML::GGUF::QwenQBitClickHouseCache::Config.new(
    endpoint: endpoint,
    table_prefix: clickhouse_table_prefix,
    connect_timeout: 2.seconds,
    read_timeout: 180.seconds,
    write_timeout: 180.seconds,
    max_recurrent_bytes: 160_i64 * 1024 * 1024,
    max_kv_bytes: 128_i64 * 1024 * 1024,
    max_total_artifact_bytes: 256_i64 * 1024 * 1024,
    resident_admission_bytes: 0_i64,
  )
  save_store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config)
  save_store.create_schema
  save_started = Time.instant
  saved = save_store.save(cache_context, native_bytes, exact_artifact_bytes, ttl: 1.hour)
  clickhouse_save_ms = (Time.instant - save_started).total_milliseconds
  unless saved.entry.cache_id == effective_cache_id
    raise "ClickHouse publication changed the cache identity"
  end

  cold_store = ML::GGUF::QwenQBitClickHouseCache::Store.new(config)
  lookup_started = Time.instant
  admission = cold_store.lookup_longest_prefix(
    ML::GGUF::QwenQBitCacheEnvelope.prefix_context(cache_context),
    full_ids,
  )
  admitted = admission || raise("ClickHouse cold lookup missed the published session anchor")
  replay = ML::GGUF::Qwen35QBitRuntimeCache.replay_plan(
    admitted.entry,
    full_ids,
    tokenizer.vocab.size.to_i32,
  )
  unless replay.prefix_len == anchor_ids.size && replay.replayed_tokens == suffix_ids.size && !replay.cached_next_token?
    raise "ClickHouse admission produced the wrong suffix replay boundary"
  end
  clickhouse_lookup_admit_ms = (Time.instant - lookup_started).total_milliseconds
else
  native_stream = ML::GGUF::QwenQBitNativeBlock.parse_stream(native_bytes)
  exact_artifact = ML::GGUF::Qwen35StateSnapshot.decode_artifact_encoded_bytes(
    exact_artifact_bytes,
    copy_payloads: false,
  )
  synthetic_entry = ML::GGUF::QwenQBitCacheEnvelope.build(
    cache_context,
    native_bytes,
    exact_artifact_bytes,
  )
  admission = ML::GGUF::QwenQBitCacheEnvelope::Admission.new(
    synthetic_entry,
    native_stream,
    exact_artifact,
  )
end
admitted_state = admission.not_nil!

free_ids = [] of Int32
free_restore_ms = 0.0_f64
free_replay_ms = 0.0_f64
free_decode_ms = 0.0_f64
free_prepare_ms = 0.0_f64
free_ended = false
forced_restore_ms = 0.0_f64
forced_replay_ms = 0.0_f64
forced_boundary_first = -1_i32
forced_top2 = [] of ML::GGUF::QwenQBitQualityMetrics::Top2
resident_live_bytes = 0_i64
with_session_adaptive_env(resident_map) do
  free_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  begin
    prepare_started = Time.instant
    ML::GGUF::Qwen35CPU.prepare_state_metal!(free_state, hp)
    free_prepare_ms = (Time.instant - prepare_started).total_milliseconds
    restore_started = Time.instant
    ML::GGUF::QwenQBitStateSnapshot.restore_admitted_native_stream_into_adaptive(
      admitted_state.native_stream,
      admitted_state.exact_artifact,
      admitted_state.entry.cache_id,
      hp,
      free_state,
    )
    free_restore_ms = (Time.instant - restore_started).total_milliseconds
    verify_session_resident!(free_state, hp, anchor_ids.size.to_i32)
    replay_started = Time.instant
    free_first, _free_first_logit = prefill_session_suffix_top1(
      weights,
      suffix_ids,
      anchor_ids.size.to_i32,
      free_state,
      replay_chunks,
    )
    free_replay_ms = (Time.instant - replay_started).total_milliseconds
    verify_session_resident!(free_state, hp, full_ids.size.to_i32)
    resident_live_bytes = hp.full_attention_layers.sum(0_i64) do |layer_index|
      free_state.layers[layer_index].adaptive_kv.not_nil!.live_compressed_bytes
    end

    decode_started = Time.instant
    token = free_first
    pos = full_ids.size.to_i32
    while free_ids.size < n_gen
      if stop_session_token?(tokenizer, token)
        free_ended = true
        break
      end
      free_ids << token
      break if free_ids.size >= n_gen
      token, _logit = ML::GGUF::Qwen35CPU.forward_top1(weights, token, pos, free_state)
      pos += 1
    end
    free_decode_ms = (Time.instant - decode_started).total_milliseconds
  ensure
    release_session_state!(free_state)
  end

  forced_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  begin
    ML::GGUF::Qwen35CPU.prepare_state_metal!(forced_state, hp)
    restore_started = Time.instant
    ML::GGUF::QwenQBitStateSnapshot.restore_admitted_native_stream_into_adaptive(
      admitted_state.native_stream,
      admitted_state.exact_artifact,
      admitted_state.entry.cache_id,
      hp,
      forced_state,
    )
    forced_restore_ms = (Time.instant - restore_started).total_milliseconds
    replay_started = Time.instant
    forced_boundary_first, _forced_boundary_logit = prefill_session_suffix_top1(
      weights,
      suffix_ids,
      anchor_ids.size.to_i32,
      forced_state,
      replay_chunks,
    )
    forced_replay_ms = (Time.instant - replay_started).total_milliseconds
    verify_session_resident!(forced_state, hp, full_ids.size.to_i32)

    exact_top2.each_with_index do |_expected, index|
      first, first_logit, second, second_logit = ML::GGUF::Qwen35CPU.forward_top2(
        weights,
        exact_ids[index],
        full_ids.size + index,
        forced_state,
      )
      forced_top2 << ML::GGUF::QwenQBitQualityMetrics::Top2.new(
        first,
        first_logit,
        second,
        second_logit,
      )
    end
  ensure
    release_session_state!(forced_state)
  end
end

free_text = tokenizer.decode(free_ids)
ecs = session_token_ecs(weights.output, exact_ids, free_ids)
top1_matches = forced_boundary_first == exact_ids.first ? 1_i32 : 0_i32
ranked_top2_matches = 0_i32
top2_overlap = 0_i32
exact_top1_covered = 0_i32
max_first_logit_delta = 0.0_f32
exact_top2.each_with_index do |expected, index|
  comparison = ML::GGUF::QwenQBitQualityMetrics.compare_top2(expected, forced_top2[index])
  top1_matches += 1 if expected.first_id == forced_top2[index].first_id
  ranked_top2_matches += comparison.ranked_matches
  top2_overlap += comparison.set_overlap
  exact_top1_covered += 1 if comparison.exact_top1_covered
  max_first_logit_delta = Math.max(max_first_logit_delta, comparison.first_logit_delta)
end
top1_count = exact_top2.size + 1
ranked_top2_count = exact_top2.size * 2
raw_live_kv_bytes = hp.full_attention_layers.size.to_i64 * 2_i64 *
                    full_ids.size.to_i64 * hp.n_head_kv.to_i64 *
                    hp.head_dim.to_i64 * sizeof(Float32)
cold_payload_bytes = native_bytes.size.to_i64 + exact_artifact_bytes.size.to_i64
exact_meaning = session_meaning_preserved?(exact_text)
free_meaning = session_meaning_preserved?(free_text)
top1_rate = top1_matches.to_f64 / top1_count
ranked_top2_rate = ranked_top2_matches.to_f64 / ranked_top2_count
top2_overlap_rate = top2_overlap.to_f64 / ranked_top2_count
exact_top1_coverage = exact_top1_covered.to_f64 / exact_top2.size
# A changed runner-up is distribution drift worth reporting, but it is not a
# response failure when the exact top-1 remains covered, the greedy trajectory
# and meaning agree, and token embeddings remain close.
top2_distribution_stable = ranked_top2_rate >= 0.75 && top2_overlap_rate >= 0.75
quality_pass = exact_ended && free_ended && exact_meaning && free_meaning &&
               top1_rate >= 0.80 && exact_top1_coverage >= 0.80 &&
               ecs.mean >= 0.90
cold_hit_to_first_token_ms = clickhouse_lookup_admit_ms + free_prepare_ms + free_restore_ms + free_replay_ms
serialize_and_publish_ms = serialize_ms + clickhouse_save_ms
exact_chunking_overhead_ms = attribute_replay ? exact_chunked_replay_ms - exact_replay_ms : 0.0_f64
adaptive_route_overhead_ms = attribute_replay ? free_replay_ms - exact_chunked_replay_ms : 0.0_f64

puts "qwen35_adaptive_session_restore_probe"
puts "  model=#{model_path}"
puts "  resident_map=#{resident_map.inspect} anchor_filler=#{filler_repetitions} suffix_filler=#{suffix_filler_repetitions} requested_gen=#{n_gen}"
puts "  anchor_tokens=#{anchor_ids.size} suffix_tokens=#{suffix_ids.size} full_tokens=#{full_ids.size} max_seq=#{max_seq} replay_chunk_tokens=#{replay_chunk_tokens} replay_chunk_count=#{replay_chunks.size}"
puts "  startup_ms=#{startup_ms.round(3)} exact_prefill_ms=#{exact_prefill_ms.round(3)} exact_replay_ms=#{exact_replay_ms.round(3)} exact_decode_ms=#{exact_decode_ms.round(3)}"
puts "  attribution_enabled=#{attribute_replay} exact_chunked_restore_ms=#{exact_chunked_restore_ms.round(3)} exact_chunked_replay_ms=#{exact_chunked_replay_ms.round(3)} exact_chunking_overhead_ms=#{exact_chunking_overhead_ms.round(3)} adaptive_route_overhead_ms=#{adaptive_route_overhead_ms.round(3)}"
puts "  serialize_ms=#{serialize_ms.round(3)} clickhouse_save_ms=#{clickhouse_save_ms.round(3)} serialize_and_publish_ms=#{serialize_and_publish_ms.round(3)} cold_native_bytes=#{native_bytes.size} cold_exact_kv_bytes=#{exact_artifact_bytes.size} cold_payload_bytes=#{cold_payload_bytes}"
puts "  clickhouse_enabled=#{!clickhouse_endpoint.nil?} clickhouse_lookup_admit_ms=#{clickhouse_lookup_admit_ms.round(3)} free_prepare_ms=#{free_prepare_ms.round(3)} free_restore_ms=#{free_restore_ms.round(3)} free_replay_ms=#{free_replay_ms.round(3)} cold_hit_to_first_token_ms=#{cold_hit_to_first_token_ms.round(3)}"
puts "  free_decode_ms=#{free_decode_ms.round(3)} forced_restore_ms=#{forced_restore_ms.round(3)} forced_replay_ms=#{forced_replay_ms.round(3)}"
puts "  top1=#{top1_matches}/#{top1_count} ranked_top2=#{ranked_top2_matches}/#{ranked_top2_count} top2_overlap=#{top2_overlap}/#{ranked_top2_count} exact_top1_covered=#{exact_top1_covered}/#{exact_top2.size} ecs=#{ecs.mean.round(6)}"
puts "  raw_live_kv_bytes=#{raw_live_kv_bytes} resident_live_kv_bytes=#{resident_live_bytes} density=#{(raw_live_kv_bytes.to_f64 / resident_live_bytes).round(4)}x"
puts "  exact_ended=#{exact_ended} free_ended=#{free_ended} exact_meaning=#{exact_meaning} free_meaning=#{free_meaning} top2_distribution_stable=#{top2_distribution_stable} quality_pass=#{quality_pass}"
puts "  exact_text=#{exact_text.inspect}"
puts "  free_text=#{free_text.inspect}"

json = JSON.build do |builder|
  builder.object do
    builder.field "schema", "qwen-adaptive-session-restore-v1"
    builder.field "resident_map", resident_map
    builder.field "anchor_filler_repetitions", filler_repetitions
    builder.field "suffix_filler_repetitions", suffix_filler_repetitions
    builder.field "anchor_tokens", anchor_ids.size
    builder.field "suffix_tokens", suffix_ids.size
    builder.field "full_tokens", full_ids.size
    builder.field "replay_chunk_tokens", replay_chunk_tokens
    builder.field "replay_chunk_sizes", replay_chunks
    builder.field "requested_gen", n_gen
    builder.field "exact_text", exact_text
    builder.field "candidate_text", free_text
    builder.field "exact_ended_with_eos", exact_ended
    builder.field "candidate_ended_with_eos", free_ended
    builder.field "exact_meaning_preserved", exact_meaning
    builder.field "candidate_meaning_preserved", free_meaning
    builder.field "top1_matches", top1_matches
    builder.field "top1_count", top1_count
    builder.field "ranked_top2_matches", ranked_top2_matches
    builder.field "ranked_top2_count", ranked_top2_count
    builder.field "top2_set_overlap", top2_overlap
    builder.field "top2_set_overlap_count", ranked_top2_count
    builder.field "top2_distribution_stable", top2_distribution_stable
    builder.field "exact_top1_covered", exact_top1_covered
    builder.field "exact_top1_covered_count", exact_top2.size
    builder.field "token_ecs_mean", ecs.mean
    builder.field "token_ecs_min", ecs.minimum
    builder.field "token_ecs_compared", ecs.compared
    builder.field "exact_token_count", ecs.exact_count
    builder.field "candidate_token_count", ecs.candidate_count
    builder.field "max_first_logit_delta", max_first_logit_delta
    builder.field "startup_ms", startup_ms
    builder.field "exact_prefill_ms", exact_prefill_ms
    builder.field "replay_attribution_enabled", attribute_replay
    builder.field "exact_chunked_restore_ms", exact_chunked_restore_ms
    builder.field "exact_chunked_replay_ms", exact_chunked_replay_ms
    builder.field "exact_chunking_overhead_ms", exact_chunking_overhead_ms
    builder.field "adaptive_route_overhead_ms", adaptive_route_overhead_ms
    builder.field "serialize_ms", serialize_ms
    builder.field "clickhouse_save_ms", clickhouse_save_ms
    builder.field "serialize_and_publish_ms", serialize_and_publish_ms
    builder.field "clickhouse_enabled", !clickhouse_endpoint.nil?
    builder.field "clickhouse_lookup_admit_ms", clickhouse_lookup_admit_ms
    builder.field "free_prepare_ms", free_prepare_ms
    builder.field "free_restore_ms", free_restore_ms
    builder.field "free_replay_ms", free_replay_ms
    builder.field "cold_hit_to_first_token_ms", cold_hit_to_first_token_ms
    builder.field "free_decode_ms", free_decode_ms
    builder.field "forced_restore_ms", forced_restore_ms
    builder.field "forced_replay_ms", forced_replay_ms
    builder.field "cold_native_bytes", native_bytes.size
    builder.field "cold_exact_kv_bytes", exact_artifact_bytes.size
    builder.field "cold_payload_bytes", cold_payload_bytes
    builder.field "source_snapshot_bytes", source_snapshot.byte_size
    builder.field "raw_live_kv_bytes", raw_live_kv_bytes
    builder.field "resident_live_kv_bytes", resident_live_bytes
    builder.field "resident_density", raw_live_kv_bytes.to_f64 / resident_live_bytes
    builder.field "full_attention_layers", hp.full_attention_layers.size
    builder.field "resident_layers", hp.full_attention_layers.size
    builder.field "resident_f32_owner_layers" do
      builder.array { }
    end
    builder.field "resident_cache_consistent", true
    builder.field "quality_pass", quality_pass
  end
end
puts "QBIT_SESSION_RESTORE_JSON=#{json}"

gguf.close
exit 1 unless quality_pass
