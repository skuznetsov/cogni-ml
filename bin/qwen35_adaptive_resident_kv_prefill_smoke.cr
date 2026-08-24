require "json"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen_qbit_quality_metrics"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL_PATH      = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
MODEL_PATH              = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
PROMPT                  = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 9821_i32, 13_i32]
CONTINUATION            = [11751_i32, 13_i32, 198_i32, 760_i32]
MAX_DECODE_TOKENS       = 512
DECODE_TOKENS           = (ENV["QWEN35_ADAPTIVE_DECODE_TOKENS"]? || "1").to_i32
REQUIRED_SEQ            = PROMPT.size + CONTINUATION.size + DECODE_TOKENS
MAX_SEQ                 = REQUIRED_SEQ > 16 ? REQUIRED_SEQ : 16
DEFAULT_MAX_LOGIT_DELTA = 0.05_f32
MAX_LOGIT_DELTA         = (ENV["QWEN35_ADAPTIVE_PREFILL_MAX_LOGIT_DELTA"]? || DEFAULT_MAX_LOGIT_DELTA.to_s).to_f32
DIAGNOSTIC_OVERRIDE     = MAX_LOGIT_DELTA != DEFAULT_MAX_LOGIT_DELTA
ADAPTIVE_MAP            = ENV["QWEN35_ADAPTIVE_RESIDENT_KV_SMOKE_MAP"]? ||
                          "p4;27=bf16,43=bf16,47=bf16,51=bf16"

private def release_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  ML::Metal::Device.synchronize
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

private def token_ecs(weights : ML::GGUF::Qwen35Weights,
                      expected_id : Int32,
                      candidate_id : Int32) : Float64
  return 1.0_f64 if expected_id == candidate_id
  expected = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, expected_id)
  candidate = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, candidate_id)
  ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(expected, candidate)
end

private def with_adaptive_env(layer_index : Int32?, tier : String?, map : String? = nil, &)
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
  ]
  old = keys.to_h { |key| {key, ENV[key]?} }
  keys.each { |key| ENV.delete(key) }
  ENV["QWEN35_ADAPTIVE_RESIDENT_KV_LAYER"] = layer_index.to_s if layer_index
  ENV["QWEN35_ADAPTIVE_RESIDENT_KV_TIER"] = tier if tier
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

unless DECODE_TOKENS.in?(1..MAX_DECODE_TOKENS)
  raise "adaptive decode token count must be within 1..#{MAX_DECODE_TOKENS}"
end
unless MAX_LOGIT_DELTA.finite? && MAX_LOGIT_DELTA > 0
  raise "adaptive logit delta guard must be finite and positive"
end
if DIAGNOSTIC_OVERRIDE && ENV["QWEN35_ADAPTIVE_ALLOW_LOGIT_DELTA_OVERRIDE"]? != "1"
  raise "adaptive logit delta override requires QWEN35_ADAPTIVE_ALLOW_LOGIT_DELTA_OVERRIDE=1"
end
raise "model does not exist: #{MODEL_PATH}" unless File.file?(MODEL_PATH)
raise "Metal is unavailable" unless ML::GGUF::Qwen35Metal.available?

baseline_state = nil.as(ML::GGUF::Qwen35CPU::State?)
adaptive_state = nil.as(ML::GGUF::Qwen35CPU::State?)
begin
  load_started = Time.instant
  weights = ML::GGUF::Qwen35Weights.from_gguf(MODEL_PATH)
  load_ms = (Time.instant - load_started).total_milliseconds
  hp = weights.hparams

  baseline_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: MAX_SEQ)
  baseline_prefill_started = Time.instant
  baseline_top = 0_i32
  baseline_logit = 0.0_f32
  baseline_append_top = 0_i32
  baseline_append_logit = 0.0_f32
  baseline_decode_tops = [] of Int32
  baseline_decode_logits = [] of Float32
  baseline_decode_second_tops = [] of Int32
  baseline_decode_second_logits = [] of Float32
  with_adaptive_env(nil, nil) do
    ML::GGUF::Qwen35CPU.prepare_state_metal!(baseline_state.not_nil!, hp)
    baseline_top, baseline_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights, PROMPT, 0, baseline_state.not_nil!,
    )
    baseline_append_top, baseline_append_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights, CONTINUATION, PROMPT.size.to_i32, baseline_state.not_nil!,
    )
  end
  baseline_prefill_ms = (Time.instant - baseline_prefill_started).total_milliseconds
  baseline_decode_started = Time.instant
  baseline_decode_input = baseline_append_top
  with_adaptive_env(nil, nil) do
    DECODE_TOKENS.times do |step|
      top, logit, second, second_logit = ML::GGUF::Qwen35CPU.forward_top2(
        weights, baseline_decode_input,
        (PROMPT.size + CONTINUATION.size + step).to_i32,
        baseline_state.not_nil!,
      )
      baseline_decode_tops << top
      baseline_decode_logits << logit
      baseline_decode_second_tops << second
      baseline_decode_second_logits << second_logit
      baseline_decode_input = top
    end
  end
  baseline_decode_ms = (Time.instant - baseline_decode_started).total_milliseconds
  baseline_full_kv_bytes = hp.full_attention_layers.sum(0_i64) do |layer_index|
    layer = baseline_state.not_nil!.layers[layer_index]
    layer.k_cache_buf.not_nil!.size + layer.v_cache_buf.not_nil!.size
  end
  release_state!(baseline_state.not_nil!)

  adaptive_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: MAX_SEQ)
  adaptive_prefill_started = Time.instant
  adaptive_top = 0_i32
  adaptive_logit = 0.0_f32
  adaptive_append_top = 0_i32
  adaptive_append_logit = 0.0_f32
  adaptive_decode_tops = [] of Int32
  adaptive_decode_logits = [] of Float32
  adaptive_decode_second_tops = [] of Int32
  adaptive_decode_second_logits = [] of Float32
  first_cache_len = 0_i32
  with_adaptive_env(nil, nil, ADAPTIVE_MAP) do
    ML::GGUF::Qwen35CPU.prepare_state_metal!(adaptive_state.not_nil!, hp)
    adaptive_top, adaptive_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights, PROMPT, 0, adaptive_state.not_nil!,
    )
    first_cache_len = adaptive_state.not_nil!.layers[hp.full_attention_layers.first].adaptive_kv.not_nil!.cache_len
    adaptive_append_top, adaptive_append_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights, CONTINUATION, PROMPT.size.to_i32, adaptive_state.not_nil!,
    )
  end
  adaptive_prefill_ms = (Time.instant - adaptive_prefill_started).total_milliseconds

  adaptive_caches = hp.full_attention_layers.map do |layer_index|
    adaptive_state.not_nil!.layers[layer_index].adaptive_kv.not_nil!
  end
  logit_delta = (adaptive_logit - baseline_logit).abs
  append_logit_delta = (adaptive_append_logit - baseline_append_logit).abs
  raise "adaptive prefill top-1 mismatch: #{adaptive_top} != #{baseline_top}" unless adaptive_top == baseline_top
  unless logit_delta.finite? && logit_delta <= MAX_LOGIT_DELTA
    raise "adaptive prefill logit delta #{logit_delta} exceeds #{MAX_LOGIT_DELTA}"
  end
  unless adaptive_append_top == baseline_append_top
    raise "adaptive packed-history top-1 mismatch: #{adaptive_append_top} != #{baseline_append_top}"
  end
  unless append_logit_delta.finite? && append_logit_delta <= MAX_LOGIT_DELTA
    raise "adaptive packed-history logit delta #{append_logit_delta} exceeds #{MAX_LOGIT_DELTA}"
  end
  raise "adaptive first prefill published #{first_cache_len} tokens, expected #{PROMPT.size}" unless first_cache_len == PROMPT.size
  prefill_cache_len = PROMPT.size + CONTINUATION.size
  adaptive_caches.each do |cache|
    raise "adaptive append published #{cache.cache_len} tokens, expected #{prefill_cache_len}" unless cache.cache_len == prefill_cache_len
  end
  hp.full_attention_layers.each do |layer_index|
    layer = adaptive_state.not_nil!.layers[layer_index]
    if layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf
      raise "adaptive prefill allocated a second F32 KV owner at layer #{layer_index}"
    end
  end

  adaptive_decode_started = Time.instant
  with_adaptive_env(nil, nil, ADAPTIVE_MAP) do
    DECODE_TOKENS.times do |step|
      input = step == 0 ? baseline_append_top : baseline_decode_tops[step - 1]
      top, logit, second, second_logit = ML::GGUF::Qwen35CPU.forward_top2(
        weights, input,
        (PROMPT.size + CONTINUATION.size + step).to_i32,
        adaptive_state.not_nil!,
      )
      adaptive_decode_tops << top
      adaptive_decode_logits << logit
      adaptive_decode_second_tops << second
      adaptive_decode_second_logits << second_logit

      expected_cache_len = PROMPT.size + CONTINUATION.size + step + 1
      adaptive_caches.each do |cache|
        unless cache.cache_len == expected_cache_len
          raise "adaptive decode step #{step} published #{cache.cache_len} tokens, expected #{expected_cache_len}"
        end
      end
      hp.full_attention_layers.each do |layer_index|
        layer = adaptive_state.not_nil!.layers[layer_index]
        if layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf
          raise "adaptive decode step #{step} allocated a second F32 KV owner at layer #{layer_index}"
        end
      end
    end
  end
  adaptive_decode_ms = (Time.instant - adaptive_decode_started).total_milliseconds

  baseline_decode_top = baseline_decode_tops.last
  baseline_decode_logit = baseline_decode_logits.last
  adaptive_decode_top = adaptive_decode_tops.last
  adaptive_decode_logit = adaptive_decode_logits.last
  decode_logit_delta = (adaptive_decode_logit - baseline_decode_logit).abs
  max_decode_logit_delta = 0.0_f32
  decode_logit_delta_sum = 0.0_f64
  decode_top1_matches = 0_i32
  decode_top2_ranked_matches = 0_i32
  decode_top2_set_overlap = 0_i32
  decode_top1_ecs = [] of Float64
  decode_second_ecs = [] of Float64
  first_decode_mismatch_step = nil.as(Int32?)
  DECODE_TOKENS.times do |step|
    delta = (adaptive_decode_logits[step] - baseline_decode_logits[step]).abs
    raise "adaptive decode logit delta is non-finite at step #{step}" unless delta.finite?
    max_decode_logit_delta = delta if delta > max_decode_logit_delta
    decode_logit_delta_sum += delta
    if adaptive_decode_tops[step] == baseline_decode_tops[step]
      decode_top1_matches += 1
    else
      first_decode_mismatch_step ||= step
    end
    exact_top2 = ML::GGUF::QwenQBitQualityMetrics::Top2.new(
      baseline_decode_tops[step], baseline_decode_logits[step],
      baseline_decode_second_tops[step], baseline_decode_second_logits[step],
    )
    adaptive_top2 = ML::GGUF::QwenQBitQualityMetrics::Top2.new(
      adaptive_decode_tops[step], adaptive_decode_logits[step],
      adaptive_decode_second_tops[step], adaptive_decode_second_logits[step],
    )
    top2_comparison = ML::GGUF::QwenQBitQualityMetrics.compare_top2(exact_top2, adaptive_top2)
    decode_top2_ranked_matches += top2_comparison.ranked_matches
    decode_top2_set_overlap += top2_comparison.set_overlap
    decode_top1_ecs << token_ecs(weights, exact_top2.first_id, adaptive_top2.first_id)
    decode_second_ecs << token_ecs(weights, exact_top2.second_id, adaptive_top2.second_id)
  end
  mean_decode_logit_delta = decode_logit_delta_sum / DECODE_TOKENS
  mean_decode_top1_ecs = decode_top1_ecs.sum / DECODE_TOKENS
  min_decode_top1_ecs = decode_top1_ecs.min
  mean_decode_second_ecs = decode_second_ecs.sum / DECODE_TOKENS
  min_decode_second_ecs = decode_second_ecs.min
  expected_cache_len = PROMPT.size + CONTINUATION.size + DECODE_TOKENS
  adaptive_caches.each do |cache|
    raise "adaptive append published #{cache.cache_len} tokens, expected #{expected_cache_len}" unless cache.cache_len == expected_cache_len
  end

  adaptive_bytes = adaptive_caches.sum(0_i64, &.compressed_bytes)
  payload = JSON.build do |json|
    json.object do
      json.field "model", File.basename(MODEL_PATH)
      json.field "adaptive_map", ADAPTIVE_MAP
      json.field "adaptive_layers", hp.full_attention_layers.size
      json.field "prompt_tokens", PROMPT.size
      json.field "append_tokens", CONTINUATION.size
      json.field "decode_tokens", DECODE_TOKENS
      json.field "max_seq", MAX_SEQ
      json.field "max_logit_delta_guard", MAX_LOGIT_DELTA
      json.field "diagnostic_logit_delta_override", DIAGNOSTIC_OVERRIDE
      json.field "first_cache_tokens", first_cache_len
      json.field "cache_tokens", adaptive_caches.first.cache_len
      json.field "baseline_top1", baseline_top
      json.field "adaptive_top1", adaptive_top
      json.field "baseline_append_top1", baseline_append_top
      json.field "adaptive_append_top1", adaptive_append_top
      json.field "baseline_decode_top1", baseline_decode_top
      json.field "adaptive_decode_top1", adaptive_decode_top
      json.field "baseline_decode_top1_sequence", baseline_decode_tops
      json.field "adaptive_decode_top1_sequence", adaptive_decode_tops
      json.field "baseline_decode_top2_sequence", baseline_decode_second_tops
      json.field "adaptive_decode_top2_sequence", adaptive_decode_second_tops
      json.field "decode_top1_matches", decode_top1_matches
      json.field "decode_top2_ranked_matches", decode_top2_ranked_matches
      json.field "decode_top2_set_overlap", decode_top2_set_overlap
      json.field "mean_decode_top1_ecs", mean_decode_top1_ecs
      json.field "min_decode_top1_ecs", min_decode_top1_ecs
      json.field "mean_decode_second_ecs", mean_decode_second_ecs
      json.field "min_decode_second_ecs", min_decode_second_ecs
      json.field "first_decode_mismatch_step", first_decode_mismatch_step
      json.field "baseline_logit", baseline_logit
      json.field "adaptive_logit", adaptive_logit
      json.field "logit_delta", logit_delta
      json.field "baseline_append_logit", baseline_append_logit
      json.field "adaptive_append_logit", adaptive_append_logit
      json.field "append_logit_delta", append_logit_delta
      json.field "baseline_decode_logit", baseline_decode_logit
      json.field "adaptive_decode_logit", adaptive_decode_logit
      json.field "decode_logit_delta", decode_logit_delta
      json.field "max_decode_logit_delta", max_decode_logit_delta
      json.field "mean_decode_logit_delta", mean_decode_logit_delta
      json.field "baseline_full_kv_bytes", baseline_full_kv_bytes
      json.field "adaptive_full_kv_bytes", adaptive_bytes
      json.field "full_kv_compression", baseline_full_kv_bytes.to_f64 / adaptive_bytes
      json.field "load_ms", load_ms.round(3)
      json.field "baseline_prefill_ms", baseline_prefill_ms.round(3)
      json.field "adaptive_prefill_ms", adaptive_prefill_ms.round(3)
      json.field "baseline_decode_ms", baseline_decode_ms.round(3)
      json.field "adaptive_decode_ms", adaptive_decode_ms.round(3)
    end
  end
  puts "ADAPTIVE_PREFILL_SMOKE_JSON=#{payload}"
  if mismatch = first_decode_mismatch_step
    raise "adaptive packed decode top-1 mismatch at step #{mismatch}: #{adaptive_decode_tops[mismatch]} != #{baseline_decode_tops[mismatch]}"
  end
  unless max_decode_logit_delta <= MAX_LOGIT_DELTA
    raise "adaptive packed decode maximum logit delta #{max_decode_logit_delta} exceeds #{MAX_LOGIT_DELTA}"
  end
ensure
  release_state!(baseline_state.not_nil!) if baseline_state
  release_state!(adaptive_state) if adaptive_state
end
