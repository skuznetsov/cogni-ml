# Same-process product decode A/B for the adaptive P4 and BF16 T8 loaders.
#
# Both states follow the same forced greedy trajectory. The execution order is
# alternated per position so model loading, prefill, and token drift cannot be
# mistaken for a loader speedup.

require "json"
require "option_parser"
require "digest/sha256"

require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_QWEN38_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_RESIDENT_MAP = "p4;27=bf16,43=bf16,47=bf16,51=bf16"
T8_ENV_KEYS          = [
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

private def with_adaptive_t8_env(resident_map : String, enabled : Bool, &)
  old = ADAPTIVE_ENV_KEYS.to_h { |key| {key, ENV[key]?} }
  ADAPTIVE_ENV_KEYS.each { |key| ENV.delete(key) }
  ENV["QWEN35_ADAPTIVE_RESIDENT_KV_MAP"] = resident_map
  ENV["QWEN35_PREFILL_CHUNK_SIZE"] = ML::GGUF::Qwen35CPU.prefill_chunk_size(true).to_s
  ENV["QWEN35_ADAPTIVE_SPLITK"] = "1"
  ENV["QWEN35_ADAPTIVE_SPLITK_MIN_CTX"] = "256"
  ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = "64"
  ENV["QWEN35_ADAPTIVE_UNIFORM_PREFILL_OFF"] = "0"
  value = enabled ? "1" : "0"
  T8_ENV_KEYS.each { |key| ENV[key] = value }
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
                                       resident_map : String) : RouteCertificate
  device_name = ML::Metal::Device.instance.name
  p4_owners = 0_i32
  bf16_owners = 0_i32
  packed_len = -1_i32

  with_adaptive_t8_env(resident_map, true) do
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
          unless ML::GGUF::QwenQBitAdaptiveMetalPolicy.p4_splitk_t8?(
                   device_name, cache_len, ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"]?,
                 )
            raise "P4 T8 route is inactive at layer #{layer_index}"
          end
          p4_owners += 1
        when ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16
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

  unless p4_owners > 0 && bf16_owners > 0 && p4_owners + bf16_owners == hp.full_attention_layers.size
    raise "adaptive T8 route coverage is incomplete: P4=#{p4_owners}, BF16=#{bf16_owners}"
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
    unless cache.cache_len == expected_tokens
      raise "adaptive KV layer #{layer_index} has #{cache.cache_len} tokens, expected #{expected_tokens}"
    end
  end
end

private def prepare_state(weights : ML::GGUF::Qwen35Weights,
                          tokens : Array(Int32),
                          max_seq : Int32,
                          resident_map : String,
                          t8_enabled : Bool)
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  begin
    first_id = -1_i32
    first_logit = Float32::NAN
    with_adaptive_t8_env(resident_map, t8_enabled) do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams)
      first_id, first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
        weights, tokens, 0, state,
      )
    end
    verify_resident_state!(state, weights.hparams, tokens.size.to_i32)
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
                        t8_enabled : Bool)
  output_id = -1_i32
  output_logit = Float32::NAN
  elapsed_ms = 0.0_f64
  with_adaptive_t8_env(resident_map, t8_enabled) do
    started = Time.instant
    output_id, output_logit = ML::GGUF::Qwen35CPU.forward_top1(
      weights, token_id, position, state,
    )
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

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_adaptive_t8_decode_probe [options] [prompt]"
  parser.on("--model PATH", "Qwen GGUF path") { |value| model_path = value }
  parser.on("--prompt-file PATH", "Read the complete prompt from PATH") { |value| prompt_file = value }
  parser.on("--samples N", "Matched measured decode positions (default: 10)") { |value| sample_count = value.to_i }
  parser.on("--max-seq N", "Cache capacity; 0 selects the minimum") { |value| requested_max_seq = value.to_i }
  parser.on("--repeat-prompt N", "Repeat prompt text before rendering (default: 1)") { |value| prompt_repeats = value.to_i }
  parser.on("--resident-map MAP", "Adaptive resident tier map") { |value| resident_map = value }
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
raise "resident map cannot be empty" if resident_map.strip.empty?
raise "Metal is unavailable" unless ML::GGUF::Qwen35Metal.available?

tokenizer_gguf = ML::GGUF::GGUFFile.new(model_path)
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
  minimum_max_seq = tokens.size + sample_count + 2
  max_seq = requested_max_seq == 0 ? minimum_max_seq : requested_max_seq
  raise "prompt plus warmup and samples exceeds --max-seq" if max_seq < minimum_max_seq

  baseline_state, baseline_first_id, baseline_first_logit = prepare_state(
    weights, tokens, max_seq, resident_map, false,
  )
  candidate_state, candidate_first_id, candidate_first_logit = prepare_state(
    weights, tokens, max_seq, resident_map, true,
  )
  unless baseline_first_id == candidate_first_id
    raise "prefill top-1 mismatch: #{baseline_first_id} != #{candidate_first_id}"
  end
  verify_finite_top1!("baseline prefill", baseline_first_id, baseline_first_logit, vocab_size)
  verify_finite_top1!("candidate prefill", candidate_first_id, candidate_first_logit, vocab_size)
  prefill_logit_delta = (baseline_first_logit - candidate_first_logit).abs
  raise "prefill logit mismatch: #{prefill_logit_delta}" if prefill_logit_delta > 1e-4_f32
  route_certificate = verify_candidate_t8_route!(candidate_state, hp, resident_map)

  input_id = baseline_first_id
  position = tokens.size.to_i32
  baseline_warm_id = -1_i32
  baseline_warm_logit = Float32::NAN
  candidate_warm_id = -1_i32
  candidate_warm_logit = Float32::NAN
  if candidate_first
    candidate_warm_id, candidate_warm_logit, _ = decode_step(
      weights, candidate_state, input_id, position, resident_map, true,
    )
    baseline_warm_id, baseline_warm_logit, _ = decode_step(
      weights, baseline_state, input_id, position, resident_map, false,
    )
  else
    baseline_warm_id, baseline_warm_logit, _ = decode_step(
      weights, baseline_state, input_id, position, resident_map, false,
    )
    candidate_warm_id, candidate_warm_logit, _ = decode_step(
      weights, candidate_state, input_id, position, resident_map, true,
    )
  end
  unless baseline_warm_id == candidate_warm_id
    raise "warmup top-1 mismatch: #{baseline_warm_id} != #{candidate_warm_id}"
  end
  verify_finite_top1!("baseline warmup", baseline_warm_id, baseline_warm_logit, vocab_size)
  verify_finite_top1!("candidate warmup", candidate_warm_id, candidate_warm_logit, vocab_size)
  warm_logit_delta = (baseline_warm_logit - candidate_warm_logit).abs
  raise "warmup logit mismatch: #{warm_logit_delta}" if warm_logit_delta > 1e-4_f32
  verify_resident_state!(baseline_state, hp, (tokens.size + 1).to_i32)
  verify_resident_state!(candidate_state, hp, (tokens.size + 1).to_i32)
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
        weights, candidate_state, input_id, position, resident_map, true,
      )
      baseline_id, baseline_logit, baseline_ms = decode_step(
        weights, baseline_state, input_id, position, resident_map, false,
      )
    else
      baseline_id, baseline_logit, baseline_ms = decode_step(
        weights, baseline_state, input_id, position, resident_map, false,
      )
      candidate_id, candidate_logit, candidate_ms = decode_step(
        weights, candidate_state, input_id, position, resident_map, true,
      )
    end

    unless baseline_id == candidate_id
      raise "measured top-1 mismatch at sample #{index}: #{baseline_id} != #{candidate_id}"
    end
    verify_finite_top1!("baseline sample #{index}", baseline_id, baseline_logit, vocab_size)
    verify_finite_top1!("candidate sample #{index}", candidate_id, candidate_logit, vocab_size)
    logit_delta = (baseline_logit - candidate_logit).abs
    raise "measured logit mismatch at sample #{index}: #{logit_delta}" if logit_delta > 1e-4_f32
    expected_tokens = (tokens.size + index + 2).to_i32
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

  puts "qwen35_adaptive_t8_decode_probe"
  puts "  release_build=#{RELEASE_BUILD} device=#{device_name.inspect} prompt_sha256=#{prompt_sha256}"
  puts "  prompt_tokens=#{tokens.size} prompt_repeats=#{prompt_repeats} samples=#{sample_count} max_seq=#{max_seq} full_attention_layers=#{hp.full_attention_layers.size}"
  puts "  automatic_t8_min_context=#{ML::GGUF::QwenQBitAdaptiveMetalPolicy::SPLITK_T8_MIN_CONTEXT}"
  puts "  candidate_t8_route_owners=p4:#{route_certificate.p4_t8_owners},bf16:#{route_certificate.bf16_t8_owners} packed_len=#{route_certificate.packed_len}"
  puts "  baseline_mean_ms=#{baseline_mean.round(3)} candidate_mean_ms=#{candidate_mean.round(3)} improvement_pct=#{improvement_pct.round(3)} wins=#{wins}/#{sample_count} gate=#{gate_passed ? "PASS" : "FAIL"}"
  puts "  baseline_median_ms=#{median(baseline_values).round(3)} candidate_median_ms=#{median(candidate_values).round(3)} ratio=#{(baseline_mean / candidate_mean).round(5)}x"
  puts "  output_ids=#{output_ids.join(',')} text=#{tokenizer.decode(output_ids).inspect}"

  payload = JSON.build do |json|
    json.object do
      json.field "schema", "qwen-adaptive-t8-decode-ab-v1"
      json.field "release_build", RELEASE_BUILD
      json.field "device", device_name
      json.field "model", File.basename(model_path)
      json.field "prompt_sha256", prompt_sha256
      json.field "resident_map", resident_map
      json.field "effective_splitk", true
      json.field "effective_splitk_min_context", 256
      json.field "effective_splitk_chunk", 64
      json.field "automatic_t8_min_context", ML::GGUF::QwenQBitAdaptiveMetalPolicy::SPLITK_T8_MIN_CONTEXT
      json.field "candidate_p4_t8_owners", route_certificate.p4_t8_owners
      json.field "candidate_bf16_t8_owners", route_certificate.bf16_t8_owners
      json.field "prompt_tokens", tokens.size
      json.field "prompt_repeats", prompt_repeats
      json.field "candidate_first", candidate_first
      json.field "samples", sample_count
      json.field "max_seq", max_seq
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
