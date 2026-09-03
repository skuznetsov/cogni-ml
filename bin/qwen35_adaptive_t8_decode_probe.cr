# Same-process product decode A/B for adaptive T8 and fused-stage2 routes.
#
# Both states follow the same forced greedy trajectory. The execution order is
# alternated per position so model loading, prefill, and token drift cannot be
# mistaken for a loader speedup.
#
# `--synthetic-prefix` restores a canonical zero-valued adaptive cache into the
# real model state. It isolates long-context decode shape and scheduling without
# the quadratic prefill cost; it is performance/correctness evidence, not a
# semantic-quality substitute for a real prompt.

require "json"
require "option_parser"
require "digest/sha256"

require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_QWEN38_MODEL       = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_RESIDENT_MAP       = "p4;27=bf16,43=bf16,47=bf16,51=bf16"
PRODUCT_PREFILL_CHUNK_SIZE = ENV.fetch("QWEN35_ADAPTIVE_PROBE_PREFILL_CHUNK_SIZE", "512").to_i32
PRODUCT_APPEND_MAX_GROUPS  =   1
PRODUCT_APPEND_COOLDOWN_MS = 100
T8_ENV_KEYS                = [
  "QWEN35_ADAPTIVE_P4_SPLITK_T8",
  "QWEN35_ADAPTIVE_BF16_SPLITK_T8",
]
ADAPTIVE_ENV_KEYS = [
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
  "QWEN35_PREFILL_APPEND_MAX_GROUPS",
  "QWEN35_PREFILL_APPEND_COOLDOWN_MS",
  "QWEN35_PREFILL_GC_GUARD_OFF",
  "QWEN35_SCRATCH_OFF",
  "QWEN35_ADAPTIVE_SPLITK",
  "QWEN35_ADAPTIVE_SPLITK_MIN_CTX",
  "QWEN35_ADAPTIVE_SPLITK_CHUNK",
  "QWEN35_ADAPTIVE_GQA6_TILE",
  "QWEN35_ADAPTIVE_DEQUANT_T4",
  "QWEN35_ADAPTIVE_UNIFORM_PREFILL_OFF",
  "QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED",
  "QWEN35_ADAPTIVE_PACK_PREFIX_QUANT",
] + T8_ENV_KEYS

RELEASE_BUILD = {% if flag?(:release) %} true {% else %} false {% end %}

raise "QWEN35_ADAPTIVE_PROBE_PREFILL_CHUNK_SIZE must be positive" unless PRODUCT_PREFILL_CHUNK_SIZE > 0

record DecodeSample,
  index : Int32,
  order : String,
  input_id : Int32,
  output_id : Int32,
  baseline_ms : Float64,
  candidate_ms : Float64,
  logit_delta : Float32

record RouteCertificate,
  p4_t8_owners : Int32,
  bf16_t8_owners : Int32,
  packed_len : Int32

private def with_adaptive_probe_env(resident_map : String,
                                    candidate : Bool,
                                    compare_stage2 : Bool,
                                    compare_splitk_chunk : Bool,
                                    &)
  old = ADAPTIVE_ENV_KEYS.to_h { |key| {key, ENV[key]?} }
  ADAPTIVE_ENV_KEYS.each { |key| ENV.delete(key) }
  ENV["QWEN35_ADAPTIVE_RESIDENT_KV_MAP"] = resident_map
  ENV["QWEN35_PREFILL_CHUNK_SIZE"] = PRODUCT_PREFILL_CHUNK_SIZE.to_s
  ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = PRODUCT_APPEND_MAX_GROUPS.to_s
  ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"] = PRODUCT_APPEND_COOLDOWN_MS.to_s
  ENV["QWEN35_PREFILL_GC_GUARD_OFF"] = "0"
  ENV["QWEN35_SCRATCH_OFF"] = "0"
  ENV["QWEN35_ADAPTIVE_SPLITK"] = "1"
  ENV["QWEN35_ADAPTIVE_SPLITK_MIN_CTX"] = "256"
  ENV["QWEN35_ADAPTIVE_UNIFORM_PREFILL_OFF"] = "0"
  if compare_splitk_chunk
    T8_ENV_KEYS.each { |key| ENV[key] = "1" }
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "1"
    ENV["QWEN35_ADAPTIVE_GQA6_TILE"] = "15"
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = candidate ? "60" : "64"
  elsif compare_stage2
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = "64"
    T8_ENV_KEYS.each { |key| ENV[key] = "1" }
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "0" unless candidate
  else
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = "64"
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "0"
    value = candidate ? "1" : "0"
    T8_ENV_KEYS.each { |key| ENV[key] = value }
  end
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

private def verify_finite_top1!(label : String,
                                token_id : Int32,
                                logit : Float32,
                                vocab_size : Int32) : Nil
  unless token_id >= 0 && token_id < vocab_size
    raise "#{label} top-1 token #{token_id} is outside vocabulary size #{vocab_size}"
  end
  raise "#{label} top-1 logit is not finite" unless logit.finite?
end

private def verify_candidate_t8_route!(state : ML::GGUF::Qwen35CPU::State,
                                       hp : ML::GGUF::Qwen35Hparams,
                                       resident_map : String,
                                       compare_stage2 : Bool,
                                       compare_splitk_chunk : Bool) : RouteCertificate
  device_name = ML::Metal::Device.instance.name
  p4_owners = 0_i32
  bf16_owners = 0_i32
  packed_len = -1_i32

  with_adaptive_probe_env(resident_map, true, compare_stage2, compare_splitk_chunk) do
    hp.full_attention_layers.each do |layer_index|
      cache = state.layers[layer_index].adaptive_kv
      raise "adaptive KV is missing layer #{layer_index}" unless cache
      cache.with_snapshot_buffers do |_k_base, k_sidecar, _v_base, v_sidecar, k_plan, v_plan, cache_len|
        packed_len = cache_len if packed_len < 0
        raise "adaptive T8 route owners have different prefix lengths" unless packed_len == cache_len
        tier = k_plan.uniform_tier
        unless tier && v_plan.uniform_tier == tier
          raise "adaptive T8 route requires a uniform K/V tier at layer #{layer_index}"
        end
        unless ML::GGUF::QwenQBitAdaptiveMetalPolicy.decode_splitk?(
                 cache_len, 1, true,
                 ENV["QWEN35_ADAPTIVE_SPLITK"]?,
                 ENV["QWEN35_ADAPTIVE_SPLITK_MIN_CTX"]?,
               )
          raise "adaptive split-K route is inactive at layer #{layer_index}"
        end

        case tier
        when ML::GGUF::QwenQBitAdaptiveKV::Tier::P4
          if compare_splitk_chunk && ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"]? != "60"
            raise "candidate split-K chunk 60 route is inactive at layer #{layer_index}"
          end
          unless ML::GGUF::QwenQBitAdaptiveMetalPolicy.p4_splitk_t8?(
                   device_name, cache_len, ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"]?,
                 )
            raise "P4 T8 route is inactive at layer #{layer_index}"
          end
          if compare_stage2 && !ML::GGUF::QwenQBitAdaptiveMetalPolicy.splitk_stage2_fused?(
               device_name, true, false, cache_len,
               ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"]?,
               ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"]?,
             )
            raise "automatic P4 fused stage2 route is inactive at layer #{layer_index}"
          end
          p4_owners += 1
        when ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16
          if compare_splitk_chunk && ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"]? != "60"
            raise "candidate split-K chunk 60 route is inactive at layer #{layer_index}"
          end
          unless !k_sidecar.contents.null? && k_sidecar.contents.address % 16_u64 == 0_u64 &&
                 !v_sidecar.contents.null? && v_sidecar.contents.address % 16_u64 == 0_u64
            raise "BF16 T8 sidecar alignment is invalid at layer #{layer_index}"
          end
          unless ML::GGUF::QwenQBitAdaptiveMetalPolicy.bf16_splitk_t8?(
                   device_name, cache_len, ENV["QWEN35_ADAPTIVE_BF16_SPLITK_T8"]?,
                 )
            raise "BF16 T8 route is inactive at layer #{layer_index}"
          end
          bf16_owners += 1
        else
          raise "adaptive T8 probe found unsupported tier #{tier} at layer #{layer_index}"
        end
      end
    end
  end

  unless p4_owners > 0 && p4_owners + bf16_owners == hp.full_attention_layers.size
    raise "adaptive T8 route coverage is incomplete: P4=#{p4_owners}, BF16=#{bf16_owners}"
  end
  if compare_stage2
    raise "stage2 comparison requires every adaptive owner to use P4" unless bf16_owners == 0
  else
    raise "T8 comparison requires both P4 and BF16 owners" unless bf16_owners > 0
  end
  RouteCertificate.new(p4_owners, bf16_owners, packed_len)
end

private def release_state!(state : ML::GGUF::Qwen35CPU::State?) : Nil
  return unless state
  ML::GGUF::Qwen35CPU.release_state_metal!(state) if ML::Metal::Device.available?
end

private def verify_resident_state!(state : ML::GGUF::Qwen35CPU::State,
                                   hp : ML::GGUF::Qwen35Hparams,
                                   expected_tokens : Int32) : Nil
  unless state.adaptive_kv_layer_indices == hp.full_attention_layers
    raise "adaptive KV does not own every full-attention layer"
  end

  f32_owners = hp.full_attention_layers.select do |layer_index|
    layer = state.layers[layer_index]
    !!(layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf)
  end
  raise "adaptive KV has Float32 owners at layers #{f32_owners}" unless f32_owners.empty?

  hp.full_attention_layers.each do |layer_index|
    cache = state.layers[layer_index].adaptive_kv
    raise "adaptive KV is missing layer #{layer_index}" unless cache
    cache.with_snapshot_buffers do |_k_base, _k_sidecar, _v_base, _v_sidecar, _k_plan, _v_plan, cache_len|
      unless cache_len == expected_tokens
        raise "adaptive KV layer #{layer_index} has #{cache_len} tokens, expected #{expected_tokens}"
      end
    end
  end

  state.layers.each_with_index do |layer, layer_index|
    unless layer.position == expected_tokens
      raise "layer #{layer_index} position is #{layer.position}, expected #{expected_tokens}"
    end
  end
end

private def verify_independent_states!(baseline : ML::GGUF::Qwen35CPU::State,
                                       candidate : ML::GGUF::Qwen35CPU::State,
                                       hp : ML::GGUF::Qwen35Hparams) : Nil
  raise "baseline and candidate state objects alias" if baseline.same?(candidate)
  baseline.layers.each_with_index do |baseline_layer, layer_index|
    candidate_layer = candidate.layers[layer_index]
    raise "baseline and candidate layer #{layer_index} alias" if baseline_layer.same?(candidate_layer)
    if hp.full_attention?(layer_index)
      baseline_cache = baseline_layer.adaptive_kv
      candidate_cache = candidate_layer.adaptive_kv
      unless baseline_cache && candidate_cache
        raise "adaptive KV is missing layer #{layer_index}"
      end
      if baseline_cache.same?(candidate_cache)
        raise "baseline and candidate adaptive KV layer #{layer_index} alias"
      end
    else
      if (baseline_conv = baseline_layer.conv_state_buf) && baseline_conv.same?(candidate_layer.conv_state_buf)
        raise "baseline and candidate recurrent conv layer #{layer_index} alias"
      end
      if (baseline_ssm = baseline_layer.ssm_state_buf) && baseline_ssm.same?(candidate_layer.ssm_state_buf)
        raise "baseline and candidate recurrent SSM layer #{layer_index} alias"
      end
    end
  end
end

private def seed_synthetic_adaptive_prefix!(state : ML::GGUF::Qwen35CPU::State,
                                            hp : ML::GGUF::Qwen35Hparams,
                                            prefix_tokens : Int32) : Nil
  raise "synthetic prefix must be positive" unless prefix_tokens > 0
  prefix_rows64 = prefix_tokens.to_i64 * hp.n_head_kv
  raise "synthetic prefix row count exceeds Int32" if prefix_rows64 > Int32::MAX
  prefix_rows = prefix_rows64.to_i32

  hp.full_attention_layers.each do |layer_index|
    cache = state.layers[layer_index].adaptive_kv
    raise "adaptive KV is missing layer #{layer_index}" unless cache

    k_plan = nil.as(ML::GGUF::QwenQBitAdaptiveKV::Plan?)
    v_plan = nil.as(ML::GGUF::QwenQBitAdaptiveKV::Plan?)
    cache.with_snapshot_buffers do |_k_base, _k_sidecar, _v_base, _v_sidecar, current_k_plan, current_v_plan, cache_len|
      raise "synthetic prefix target is not empty at layer #{layer_index}" unless cache_len == 0
      k_plan = current_k_plan
      v_plan = current_v_plan
    end

    k = ML::GGUF::QwenQBitAdaptiveKV.empty_encoded(k_plan.not_nil!, prefix_rows)
    v = ML::GGUF::QwenQBitAdaptiveKV.empty_encoded(v_plan.not_nil!, prefix_rows)
    ML::GGUF::QwenQBitAdaptiveResidentKV.restore_snapshot!(cache, k, v, prefix_tokens)
  end

  state.layers.each { |layer| layer.position = prefix_tokens }
  verify_resident_state!(state, hp, prefix_tokens)
end

private def prepare_state(weights : ML::GGUF::Qwen35Weights,
                          tokens : Array(Int32),
                          max_seq : Int32,
                          resident_map : String,
                          candidate : Bool,
                          compare_stage2 : Bool,
                          compare_splitk_chunk : Bool,
                          synthetic_prefix : Int32)
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  begin
    first_id = -1_i32
    first_logit = Float32::NAN
    with_adaptive_probe_env(resident_map, candidate, compare_stage2, compare_splitk_chunk) do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams)
      if synthetic_prefix > 0
        seed_synthetic_adaptive_prefix!(state, weights.hparams, synthetic_prefix)
        first_id = tokens.last
        first_logit = 0.0_f32
      else
        first_id, first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
          weights, tokens, 0, state,
        )
      end
    end
    expected_tokens = synthetic_prefix > 0 ? synthetic_prefix : tokens.size.to_i32
    state.layers.each { |layer| layer.position = expected_tokens }
    verify_resident_state!(state, weights.hparams, expected_tokens)
    {state, first_id, first_logit}
  rescue ex
    release_state!(state)
    raise ex
  end
end

private def decode_step(weights : ML::GGUF::Qwen35Weights,
                        state : ML::GGUF::Qwen35CPU::State,
                        token_id : Int32,
                        position : Int32,
                        resident_map : String,
                        candidate : Bool,
                        compare_stage2 : Bool,
                        compare_splitk_chunk : Bool)
  output_id = -1_i32
  output_logit = Float32::NAN
  elapsed_ms = 0.0_f64
  with_adaptive_probe_env(resident_map, candidate, compare_stage2, compare_splitk_chunk) do
    started = Time.instant
    output_id, output_logit = ML::GGUF::Qwen35CPU.forward_top1(
      weights, token_id, position, state,
    )
    state.layers.each { |layer| layer.position = position + 1 }
    elapsed_ms = (Time.instant - started).total_milliseconds
  end
  {output_id, output_logit, elapsed_ms}
end

private def median(values : Array(Float64)) : Float64
  raise "median requires at least one value" if values.empty?
  sorted = values.sort
  midpoint = sorted.size // 2
  sorted.size.odd? ? sorted[midpoint] : (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
end

model_path = DEFAULT_QWEN38_MODEL
resident_map = DEFAULT_RESIDENT_MAP
prompt = "Explain one safe optimization for a compressed GPU KV cache."
prompt_file = nil.as(String?)
sample_count = 10
requested_max_seq = 0
prompt_repeats = 1
chat_mode = true
candidate_first = false
compare_stage2 = false
compare_splitk_chunk = false
synthetic_prefix = 0

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_adaptive_t8_decode_probe [options] [prompt]"
  parser.on("--model PATH", "Qwen GGUF path") { |value| model_path = value }
  parser.on("--prompt-file PATH", "Read the complete prompt from PATH") { |value| prompt_file = value }
  parser.on("--samples N", "Matched measured decode positions (default: 10)") { |value| sample_count = value.to_i }
  parser.on("--max-seq N", "Cache capacity; 0 selects the minimum") { |value| requested_max_seq = value.to_i }
  parser.on("--repeat-prompt N", "Repeat prompt text before rendering (default: 1)") { |value| prompt_repeats = value.to_i }
  parser.on("--resident-map MAP", "Adaptive resident tier map") { |value| resident_map = value }
  parser.on("--compare-stage2", "Hold T8 on and compare legacy stage2 with automatic policy") { compare_stage2 = true }
  parser.on("--compare-splitk-chunk", "Compare explicit split-K chunks 64 and 60") { compare_splitk_chunk = true }
  parser.on("--synthetic-prefix N", "Restore a zero-valued adaptive prefix for decode-only timing") { |value| synthetic_prefix = value.to_i }
  parser.on("--raw", "Do not render the Qwen chat template") { chat_mode = false }
  parser.on("--candidate-first", "Run candidate first during warmup and the first measured pair") { candidate_first = true }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

raise "--prompt-file cannot be combined with a positional prompt" if prompt_file && !ARGV.empty?
if path = prompt_file
  raise "prompt file does not exist: #{path}" unless File.file?(path)
  prompt = File.read(path)
elsif !ARGV.empty?
  prompt = ARGV.join(" ")
end

raise "model does not exist: #{model_path}" unless File.file?(model_path)
raise "--samples must be at least 10" unless sample_count >= 10
raise "--repeat-prompt must be positive" unless prompt_repeats > 0
raise "--max-seq cannot be negative" if requested_max_seq < 0
raise "--synthetic-prefix cannot be negative" if synthetic_prefix < 0
raise "resident map cannot be empty" if resident_map.strip.empty?
raise "select only one comparison" if compare_stage2 && compare_splitk_chunk
if synthetic_prefix > 0 && !compare_splitk_chunk
  raise "--synthetic-prefix currently requires --compare-splitk-chunk"
end
raise "Metal is unavailable" unless ML::GGUF::Qwen35Metal.available?

tokenizer_gguf = ML::GGUF::GGUFFile.new(model_path, mmap_tensors: false)
weights = nil.as(ML::GGUF::Qwen35Weights?)
baseline_state = nil.as(ML::GGUF::Qwen35CPU::State?)
candidate_state = nil.as(ML::GGUF::Qwen35CPU::State?)

begin
  tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(tokenizer_gguf, model_path)
  vocab_size = tokenizer.vocab.size.to_i32
  weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
  hp = weights.hparams
  raise "probe requires Qwen3.8 head dimension 256" unless hp.head_dim == 256
  device_name = ML::Metal::Device.instance.name
  raise "probe requires exact Apple M2 Max" unless device_name == "Apple M2 Max"

  repeated_prompt = Array.new(prompt_repeats, prompt).join("\n\n")
  model_prompt = chat_mode ? ML::GGUF::Qwen35Chat.render_user_prompt(repeated_prompt, enable_thinking: false) : repeated_prompt
  tokens = tokenizer.encode(model_prompt)
  raise "prompt encoded to zero tokens" if tokens.empty?
  prefix_tokens = synthetic_prefix > 0 ? synthetic_prefix : tokens.size.to_i32
  minimum_max_seq = prefix_tokens + sample_count + 2
  max_seq = requested_max_seq == 0 ? minimum_max_seq : requested_max_seq
  raise "prefix plus warmup and samples exceeds --max-seq" if max_seq < minimum_max_seq

  baseline_state, baseline_first_id, baseline_first_logit = prepare_state(
    weights, tokens, max_seq, resident_map, false, compare_stage2, compare_splitk_chunk, synthetic_prefix.to_i32,
  )
  candidate_state, candidate_first_id, candidate_first_logit = prepare_state(
    weights, tokens, max_seq, resident_map, true, compare_stage2, compare_splitk_chunk, synthetic_prefix.to_i32,
  )
  verify_independent_states!(baseline_state, candidate_state, hp)
  unless baseline_first_id == candidate_first_id
    raise "prefill top-1 mismatch: #{baseline_first_id} != #{candidate_first_id}"
  end
  semantic_quality_valid = synthetic_prefix == 0
  if semantic_quality_valid
    verify_finite_top1!("baseline prefill", baseline_first_id, baseline_first_logit, vocab_size)
    verify_finite_top1!("candidate prefill", candidate_first_id, candidate_first_logit, vocab_size)
  elsif baseline_first_id < 0 || baseline_first_id >= vocab_size
    raise "synthetic seed token is outside the vocabulary: #{baseline_first_id}"
  end
  prefill_logit_delta = (baseline_first_logit - candidate_first_logit).abs
  if semantic_quality_valid && prefill_logit_delta > 1e-4_f32
    raise "prefill logit mismatch: #{prefill_logit_delta}"
  end
  route_certificate = verify_candidate_t8_route!(
    candidate_state, hp, resident_map, compare_stage2, compare_splitk_chunk,
  )

  input_id = baseline_first_id
  position = prefix_tokens.to_i32
  baseline_warm_id = -1_i32
  baseline_warm_logit = Float32::NAN
  candidate_warm_id = -1_i32
  candidate_warm_logit = Float32::NAN
  if candidate_first
    candidate_warm_id, candidate_warm_logit, _ = decode_step(
      weights, candidate_state, input_id, position, resident_map, true, compare_stage2, compare_splitk_chunk,
    )
    baseline_warm_id, baseline_warm_logit, _ = decode_step(
      weights, baseline_state, input_id, position, resident_map, false, compare_stage2, compare_splitk_chunk,
    )
  else
    baseline_warm_id, baseline_warm_logit, _ = decode_step(
      weights, baseline_state, input_id, position, resident_map, false, compare_stage2, compare_splitk_chunk,
    )
    candidate_warm_id, candidate_warm_logit, _ = decode_step(
      weights, candidate_state, input_id, position, resident_map, true, compare_stage2, compare_splitk_chunk,
    )
  end
  unless baseline_warm_id == candidate_warm_id
    raise "warmup top-1 mismatch: #{baseline_warm_id} != #{candidate_warm_id}"
  end
  verify_finite_top1!("baseline warmup", baseline_warm_id, baseline_warm_logit, vocab_size)
  verify_finite_top1!("candidate warmup", candidate_warm_id, candidate_warm_logit, vocab_size)
  warm_logit_delta = (baseline_warm_logit - candidate_warm_logit).abs
  raise "warmup logit mismatch: #{warm_logit_delta}" if warm_logit_delta > 1e-4_f32
  verify_resident_state!(baseline_state, hp, (prefix_tokens + 1).to_i32)
  verify_resident_state!(candidate_state, hp, (prefix_tokens + 1).to_i32)
  input_id = baseline_warm_id
  position += 1

  samples = [] of DecodeSample
  output_ids = [] of Int32
  sample_count.times do |index|
    baseline_id = -1_i32
    baseline_logit = Float32::NAN
    baseline_ms = 0.0_f64
    candidate_id = -1_i32
    candidate_logit = Float32::NAN
    candidate_ms = 0.0_f64
    run_candidate_first = candidate_first == index.even?
    order = run_candidate_first ? "BA" : "AB"

    if run_candidate_first
      candidate_id, candidate_logit, candidate_ms = decode_step(
        weights, candidate_state, input_id, position, resident_map, true, compare_stage2, compare_splitk_chunk,
      )
      baseline_id, baseline_logit, baseline_ms = decode_step(
        weights, baseline_state, input_id, position, resident_map, false, compare_stage2, compare_splitk_chunk,
      )
    else
      baseline_id, baseline_logit, baseline_ms = decode_step(
        weights, baseline_state, input_id, position, resident_map, false, compare_stage2, compare_splitk_chunk,
      )
      candidate_id, candidate_logit, candidate_ms = decode_step(
        weights, candidate_state, input_id, position, resident_map, true, compare_stage2, compare_splitk_chunk,
      )
    end

    unless baseline_id == candidate_id
      raise "measured top-1 mismatch at sample #{index}: #{baseline_id} != #{candidate_id}"
    end
    verify_finite_top1!("baseline sample #{index}", baseline_id, baseline_logit, vocab_size)
    verify_finite_top1!("candidate sample #{index}", candidate_id, candidate_logit, vocab_size)
    logit_delta = (baseline_logit - candidate_logit).abs
    raise "measured logit mismatch at sample #{index}: #{logit_delta}" if logit_delta > 1e-4_f32
    expected_tokens = (prefix_tokens + index + 2).to_i32
    verify_resident_state!(baseline_state, hp, expected_tokens)
    verify_resident_state!(candidate_state, hp, expected_tokens)

    samples << DecodeSample.new(index.to_i32, order, input_id, baseline_id, baseline_ms, candidate_ms, logit_delta)
    output_ids << baseline_id
    input_id = baseline_id
    position += 1
  end

  baseline_values = samples.map(&.baseline_ms)
  candidate_values = samples.map(&.candidate_ms)
  baseline_mean = baseline_values.sum / baseline_values.size
  candidate_mean = candidate_values.sum / candidate_values.size
  improvement_pct = (baseline_mean - candidate_mean) / baseline_mean * 100.0
  wins = samples.count { |sample| sample.candidate_ms < sample.baseline_ms }
  gate_passed = RELEASE_BUILD && improvement_pct >= 3.0 && wins >= 8
  prompt_sha256 = Digest::SHA256.hexdigest(repeated_prompt.to_slice)
  comparison = compare_splitk_chunk ? "splitk_chunk" : (compare_stage2 ? "p4_stage2" : "t8_loaders")
  state_source = synthetic_prefix > 0 ? "synthetic_zero_prefix" : "real_prompt_prefill"
  baseline_stage2 = compare_splitk_chunk ? "automatic" : "legacy"
  candidate_stage2 = (compare_stage2 || compare_splitk_chunk) ? "automatic" : "legacy"

  puts "qwen35_adaptive_t8_decode_probe"
  puts "  release_build=#{RELEASE_BUILD} device=#{device_name.inspect} comparison=#{comparison} state_source=#{state_source} prompt_sha256=#{prompt_sha256}"
  puts "  semantic_quality_valid=#{semantic_quality_valid}"
  puts "  baseline_stage2=#{baseline_stage2} candidate_stage2=#{candidate_stage2}"
  puts "  baseline_splitk_chunk=64 candidate_splitk_chunk=#{compare_splitk_chunk ? 60 : 64}"
  puts "  prompt_tokens=#{tokens.size} seeded_prefix_tokens=#{prefix_tokens} prompt_repeats=#{prompt_repeats} samples=#{sample_count} max_seq=#{max_seq} full_attention_layers=#{hp.full_attention_layers.size}"
  puts "  prefill_chunk_size=#{PRODUCT_PREFILL_CHUNK_SIZE} append_max_groups=#{PRODUCT_APPEND_MAX_GROUPS} append_cooldown_ms=#{PRODUCT_APPEND_COOLDOWN_MS} pooled_scratch=true gc_guard=true"
  puts "  automatic_t8_min_context=#{ML::GGUF::QwenQBitAdaptiveMetalPolicy::SPLITK_T8_MIN_CONTEXT}"
  puts "  candidate_t8_route_owners=p4:#{route_certificate.p4_t8_owners},bf16:#{route_certificate.bf16_t8_owners} packed_len=#{route_certificate.packed_len}"
  puts "  baseline_mean_ms=#{baseline_mean.round(3)} candidate_mean_ms=#{candidate_mean.round(3)} improvement_pct=#{improvement_pct.round(3)} wins=#{wins}/#{sample_count} gate=#{gate_passed ? "PASS" : "FAIL"}"
  puts "  baseline_median_ms=#{median(baseline_values).round(3)} candidate_median_ms=#{median(candidate_values).round(3)} ratio=#{(baseline_mean / candidate_mean).round(5)}x"
  puts "  output_ids=#{output_ids.join(',')} text=#{tokenizer.decode(output_ids).inspect}"

  payload = JSON.build do |json|
    json.object do
      json.field "schema", "qwen-adaptive-t8-decode-ab-v4"
      json.field "comparison", comparison
      json.field "state_source", state_source
      json.field "semantic_quality_valid", semantic_quality_valid
      json.field "baseline_stage2", baseline_stage2
      json.field "candidate_stage2", candidate_stage2
      json.field "release_build", RELEASE_BUILD
      json.field "device", device_name
      json.field "model", File.basename(model_path)
      json.field "prompt_sha256", prompt_sha256
      json.field "resident_map", resident_map
      json.field "effective_splitk", true
      json.field "effective_splitk_min_context", 256
      json.field "baseline_splitk_chunk", 64
      json.field "candidate_splitk_chunk", compare_splitk_chunk ? 60 : 64
      json.field "automatic_t8_min_context", ML::GGUF::QwenQBitAdaptiveMetalPolicy::SPLITK_T8_MIN_CONTEXT
      json.field "candidate_p4_t8_owners", route_certificate.p4_t8_owners
      json.field "candidate_bf16_t8_owners", route_certificate.bf16_t8_owners
      json.field "prompt_tokens", tokens.size
      json.field "seeded_prefix_tokens", prefix_tokens
      json.field "prompt_repeats", prompt_repeats
      json.field "candidate_first", candidate_first
      json.field "samples", sample_count
      json.field "max_seq", max_seq
      json.field "prefill_chunk_size", PRODUCT_PREFILL_CHUNK_SIZE
      json.field "prefill_append_max_groups", PRODUCT_APPEND_MAX_GROUPS
      json.field "prefill_append_cooldown_ms", PRODUCT_APPEND_COOLDOWN_MS
      json.field "pooled_scratch", true
      json.field "prefill_gc_guard", true
      json.field "baseline_mean_ms", baseline_mean
      json.field "candidate_mean_ms", candidate_mean
      json.field "baseline_median_ms", median(baseline_values)
      json.field "candidate_median_ms", median(candidate_values)
      json.field "improvement_pct", improvement_pct
      json.field "ratio", baseline_mean / candidate_mean
      json.field "candidate_wins", wins
      json.field "gate_passed", gate_passed
      json.field "prefill_logit_delta", prefill_logit_delta
      json.field "warm_logit_delta", warm_logit_delta
      json.field "output_ids", output_ids
      json.field "pairs" do
        json.array do
          samples.each do |sample|
            json.object do
              json.field "index", sample.index
              json.field "order", sample.order
              json.field "input_id", sample.input_id
              json.field "output_id", sample.output_id
              json.field "baseline_ms", sample.baseline_ms
              json.field "candidate_ms", sample.candidate_ms
              json.field "logit_delta", sample.logit_delta
            end
          end
        end
      end
    end
  end
  puts "QBIT_T8_DECODE_JSON=#{payload}"
ensure
  release_state!(candidate_state)
  release_state!(baseline_state)
  weights.try(&.close)
  tokenizer_gguf.close
end
