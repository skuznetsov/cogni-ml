require "json"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
MODEL_PATH         = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
MAX_SEQ            = 16
PROMPT             = [760_i32, 6511_i32, 314_i32, 9338_i32, 369_i32, 279_i32, 9821_i32, 13_i32]
CONTINUATION       = [11751_i32, 13_i32, 198_i32, 760_i32]
MAX_LOGIT_DELTA    = (ENV["QWEN35_ADAPTIVE_PREFILL_MAX_LOGIT_DELTA"]? || "0.05").to_f32

private def release_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  ML::Metal::Device.synchronize
  state.layers.each do |layer|
    layer.k_cache_buf.try(&.release)
    layer.v_cache_buf.try(&.release)
    layer.conv_state_buf.try(&.release)
    layer.ssm_state_buf.try(&.release)
    layer.adaptive_kv.try(&.release)
    layer.k_cache_buf = nil
    layer.v_cache_buf = nil
    layer.conv_state_buf = nil
    layer.ssm_state_buf = nil
    layer.adaptive_kv = nil
  end
end

private def with_adaptive_env(layer_index : Int32?, tier : String?, &)
  keys = [
    "QWEN35_ADAPTIVE_RESIDENT_KV_LAYER",
    "QWEN35_ADAPTIVE_RESIDENT_KV_TIER",
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

raise "model does not exist: #{MODEL_PATH}" unless File.file?(MODEL_PATH)
raise "Metal is unavailable" unless ML::GGUF::Qwen35Metal.available?

baseline_state = nil.as(ML::GGUF::Qwen35CPU::State?)
adaptive_state = nil.as(ML::GGUF::Qwen35CPU::State?)
begin
  load_started = Time.instant
  weights = ML::GGUF::Qwen35Weights.from_gguf(MODEL_PATH)
  load_ms = (Time.instant - load_started).total_milliseconds
  hp = weights.hparams
  selected_layer = hp.full_attention_layers.first

  baseline_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: MAX_SEQ)
  baseline_started = Time.instant
  baseline_top = 0_i32
  baseline_logit = 0.0_f32
  baseline_append_top = 0_i32
  baseline_append_logit = 0.0_f32
  baseline_decode_top = 0_i32
  baseline_decode_logit = 0.0_f32
  with_adaptive_env(nil, nil) do
    ML::GGUF::Qwen35CPU.prepare_state_metal!(baseline_state.not_nil!, hp)
    baseline_top, baseline_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights, PROMPT, 0, baseline_state.not_nil!,
    )
    baseline_append_top, baseline_append_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights, CONTINUATION, PROMPT.size.to_i32, baseline_state.not_nil!,
    )
    baseline_decode_top, baseline_decode_logit = ML::GGUF::Qwen35CPU.forward_top1(
      weights, baseline_append_top, (PROMPT.size + CONTINUATION.size).to_i32,
      baseline_state.not_nil!,
    )
  end
  baseline_ms = (Time.instant - baseline_started).total_milliseconds
  baseline_selected_bytes = 2_i64 * baseline_state.not_nil!.layers[selected_layer].k_cache_buf.not_nil!.size
  release_state!(baseline_state.not_nil!)

  adaptive_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: MAX_SEQ)
  adaptive_started = Time.instant
  adaptive_top = 0_i32
  adaptive_logit = 0.0_f32
  adaptive_append_top = 0_i32
  adaptive_append_logit = 0.0_f32
  adaptive_decode_top = 0_i32
  adaptive_decode_logit = 0.0_f32
  first_cache_len = 0_i32
  with_adaptive_env(selected_layer, "bf16") do
    ML::GGUF::Qwen35CPU.prepare_state_metal!(adaptive_state.not_nil!, hp)
    adaptive_top, adaptive_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights, PROMPT, 0, adaptive_state.not_nil!,
    )
    first_cache_len = adaptive_state.not_nil!.layers[selected_layer].adaptive_kv.not_nil!.cache_len
    adaptive_append_top, adaptive_append_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
      weights, CONTINUATION, PROMPT.size.to_i32, adaptive_state.not_nil!,
    )
    adaptive_decode_top, adaptive_decode_logit = ML::GGUF::Qwen35CPU.forward_top1(
      weights, adaptive_append_top, (PROMPT.size + CONTINUATION.size).to_i32,
      adaptive_state.not_nil!,
    )
  end
  adaptive_ms = (Time.instant - adaptive_started).total_milliseconds

  selected_state = adaptive_state.not_nil!.layers[selected_layer]
  cache = selected_state.adaptive_kv.not_nil!
  logit_delta = (adaptive_logit - baseline_logit).abs
  append_logit_delta = (adaptive_append_logit - baseline_append_logit).abs
  decode_logit_delta = (adaptive_decode_logit - baseline_decode_logit).abs
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
  unless adaptive_decode_top == baseline_decode_top
    raise "adaptive packed decode top-1 mismatch: #{adaptive_decode_top} != #{baseline_decode_top}"
  end
  unless decode_logit_delta.finite? && decode_logit_delta <= MAX_LOGIT_DELTA
    raise "adaptive packed decode logit delta #{decode_logit_delta} exceeds #{MAX_LOGIT_DELTA}"
  end
  raise "adaptive first prefill published #{first_cache_len} tokens, expected #{PROMPT.size}" unless first_cache_len == PROMPT.size
  expected_cache_len = PROMPT.size + CONTINUATION.size + 1
  raise "adaptive append published #{cache.cache_len} tokens, expected #{expected_cache_len}" unless cache.cache_len == expected_cache_len
  if selected_state.k_cache || selected_state.v_cache || selected_state.k_cache_buf || selected_state.v_cache_buf
    raise "adaptive prefill allocated a second F32 KV owner"
  end

  adaptive_bytes = cache.compressed_bytes
  payload = JSON.build do |json|
    json.object do
      json.field "model", File.basename(MODEL_PATH)
      json.field "selected_layer", selected_layer
      json.field "tier", "bf16"
      json.field "prompt_tokens", PROMPT.size
      json.field "append_tokens", CONTINUATION.size
      json.field "first_cache_tokens", first_cache_len
      json.field "cache_tokens", cache.cache_len
      json.field "baseline_top1", baseline_top
      json.field "adaptive_top1", adaptive_top
      json.field "baseline_append_top1", baseline_append_top
      json.field "adaptive_append_top1", adaptive_append_top
      json.field "baseline_decode_top1", baseline_decode_top
      json.field "adaptive_decode_top1", adaptive_decode_top
      json.field "baseline_logit", baseline_logit
      json.field "adaptive_logit", adaptive_logit
      json.field "logit_delta", logit_delta
      json.field "baseline_append_logit", baseline_append_logit
      json.field "adaptive_append_logit", adaptive_append_logit
      json.field "append_logit_delta", append_logit_delta
      json.field "baseline_decode_logit", baseline_decode_logit
      json.field "adaptive_decode_logit", adaptive_decode_logit
      json.field "decode_logit_delta", decode_logit_delta
      json.field "baseline_selected_kv_bytes", baseline_selected_bytes
      json.field "adaptive_selected_kv_bytes", adaptive_bytes
      json.field "selected_layer_compression", baseline_selected_bytes.to_f64 / adaptive_bytes
      json.field "load_ms", load_ms.round(3)
      json.field "baseline_prefill_ms", baseline_ms.round(3)
      json.field "adaptive_prefill_ms", adaptive_ms.round(3)
    end
  end
  puts "ADAPTIVE_PREFILL_SMOKE_JSON=#{payload}"
ensure
  release_state!(baseline_state.not_nil!) if baseline_state
  release_state!(adaptive_state) if adaptive_state
end
