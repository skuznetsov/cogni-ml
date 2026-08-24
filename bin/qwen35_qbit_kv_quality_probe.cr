# Default-off quality falsifier for retire-then-pack uniform and adaptive QBit
# KV semantics.
#
# Each prefill chunk is consumed exactly, then its completed KV rows are
# quantized and reconstructed into the existing Float32 cache before the next
# chunk. Decode rows use the same retire-then-pack order. This models the values
# a future resident QBit attention kernel would observe without changing
# production cache ownership or routing.

require "option_parser"

require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen_qbit_kv_quality"

DEFAULT_QWEN38_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"

record QBitQualityPolicy,
  label : String,
  precision : Int32?,
  adaptive_default : ML::GGUF::QwenQBitAdaptiveKV::Tier?,
  adaptive_layer_tiers : Hash(Int32, ML::GGUF::QwenQBitAdaptiveKV::Tier)

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_QWEN38_MODEL
prompt = "Explain in one sentence why the sky appears blue."
n_gen = 4
requested_max_seq = 0
precisions = [4, 5] of Int32
adaptive_maps = [] of String
adaptive_sweep = nil.as(String?)
chat_mode = true
retire_chunk = 8

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_qbit_kv_quality_probe [options] [prompt]"
  parser.on("--model PATH", "Qwen GGUF path") { |value| model_path = value }
  parser.on("--gen N", "Generated tokens including the exact prefill token (default: 4)") { |value| n_gen = value.to_i }
  parser.on("--max-seq N", "Cache capacity; 0 selects prompt+gen+1 (default: 0)") { |value| requested_max_seq = value.to_i }
  parser.on("--precisions LIST", "Comma-separated QBit planes (default: 4,5)") do |value|
    precisions = value.split(',').map(&.to_i32)
  end
  parser.on("--no-uniform", "Skip uniform p4/p5 variants") { precisions.clear }
  parser.on("--adaptive-map MAP", "Add p4-default layer escapes, e.g. 3=f32,7=bf16") do |value|
    adaptive_maps << value
  end
  parser.on("--adaptive-sweep TIER", "Try one p5, bf16, or f32 escape layer at a time") do |value|
    adaptive_sweep = value
  end
  parser.on("--retire-chunk N", "Completed prefill rows packed per chunk (default: 8)") { |value| retire_chunk = value.to_i }
  parser.on("--raw", "Do not render the Qwen chat template") { chat_mode = false }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end
prompt = ARGV.join(" ") unless ARGV.empty?

raise "model does not exist: #{model_path}" unless File.file?(model_path)
raise "--gen must be at least 2" unless n_gen >= 2
raise "--max-seq cannot be negative" if requested_max_seq < 0
raise "no quality variant selected" if precisions.empty? && adaptive_maps.empty? && adaptive_sweep.nil?
raise "--retire-chunk must be positive" unless retire_chunk > 0
precisions.each do |precision|
  raise "precision must be p4 or p5" unless precision == 4 || precision == 5
end
raise "duplicate precisions" unless precisions.uniq.size == precisions.size

def adaptive_tier(value : String) : ML::GGUF::QwenQBitAdaptiveKV::Tier
  case value.downcase
  when "p4"   then ML::GGUF::QwenQBitAdaptiveKV::Tier::P4
  when "p5"   then ML::GGUF::QwenQBitAdaptiveKV::Tier::P5
  when "bf16" then ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16
  when "f32"  then ML::GGUF::QwenQBitAdaptiveKV::Tier::F32
  else
    raise "adaptive tier must be p4, p5, bf16, or f32"
  end
end

def adaptive_policy(spec : String) : QBitQualityPolicy
  layer_tiers = {} of Int32 => ML::GGUF::QwenQBitAdaptiveKV::Tier
  spec.split(',').each do |entry|
    layer_text, tier_text = entry.split('=', 2)
    raise "adaptive map entries must use LAYER=TIER" unless layer_text && tier_text
    layer = layer_text.to_i32
    raise "duplicate adaptive layer #{layer}" if layer_tiers.has_key?(layer)
    layer_tiers[layer] = adaptive_tier(tier_text)
  end
  raise "adaptive map cannot be empty" if layer_tiers.empty?
  QBitQualityPolicy.new("adaptive[#{spec}]", nil,
    ML::GGUF::QwenQBitAdaptiveKV::Tier::P4, layer_tiers)
end

def release_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  {% unless flag?(:cpu_only) %}
    ML::Metal::Device.synchronize if ML::Metal::Device.available?
  {% end %}
  state.layers.each do |layer|
    layer.k_cache_buf.try(&.release)
    layer.v_cache_buf.try(&.release)
    layer.conv_state_buf.try(&.release)
    layer.ssm_state_buf.try(&.release)
    layer.k_cache_buf = nil
    layer.v_cache_buf = nil
    layer.conv_state_buf = nil
    layer.ssm_state_buf = nil
  end
end

def top1(logits : Array(Float32)) : {Int32, Float32}
  value = logits.max
  {logits.index(value).not_nil!.to_i32, value}
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64
  aa = 0.0_f64
  bb = 0.0_f64
  a.each_with_index do |value, i|
    x = value.to_f64
    y = b[i].to_f64
    dot += x * y
    aa += x * x
    bb += y * y
  end
  dot / (Math.sqrt(aa) * Math.sqrt(bb))
end

def max_delta(a : Array(Float32), b : Array(Float32)) : Float32
  a.each_with_index.max_of { |value, i| (value - b[i]).abs }
end

def add_stats(a : ML::GGUF::QwenQBitKVQuality::Stats,
              b : ML::GGUF::QwenQBitKVQuality::Stats) : ML::GGUF::QwenQBitKVQuality::Stats
  ML::GGUF::QwenQBitKVQuality::Stats.new(
    a.raw_bytes + b.raw_bytes,
    a.payload_bytes + b.payload_bytes,
    a.blocks + b.blocks,
  )
end

def roundtrip_policy!(state : ML::GGUF::Qwen35CPU::State,
                      hp : ML::GGUF::Qwen35Hparams,
                      max_seq : Int32,
                      start_pos : Int32,
                      token_count : Int32,
                      policy : QBitQualityPolicy) : ML::GGUF::QwenQBitKVQuality::Stats
  if precision = policy.precision
    ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
      state.layers, hp.full_attention_layers, max_seq,
      hp.n_head_kv, hp.head_dim, start_pos, token_count, precision,
    )
  else
    ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span_adaptive!(
      state.layers, hp.full_attention_layers, max_seq,
      hp.n_head_kv, hp.head_dim, start_pos, token_count,
      policy.adaptive_default.not_nil!, policy.adaptive_layer_tiers,
    )
  end
end

def prefill_chunked(weights : ML::GGUF::Qwen35Weights,
                    token_ids : Array(Int32),
                    max_seq : Int32,
                    chunk_size : Int32,
                    policy : QBitQualityPolicy?)
  hp = weights.hparams
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
  stats = ML::GGUF::QwenQBitKVQuality::Stats.new(0_i64, 0_i64, 0_i64)
  roundtrip_ms = 0.0_f64
  first_id = -1_i32
  first_logit = Float32::NAN
  offset = 0
  started = Time.instant

  begin
    while offset < token_ids.size
      count = Math.min(chunk_size, token_ids.size - offset)
      final_chunk = offset + count == token_ids.size
      if final_chunk
        first_id, first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
          weights, token_ids[offset, count], offset, state,
        )
      else
        ML::GGUF::Qwen35CPU.prefill_tokens(weights, token_ids[offset, count], offset, state)
      end

      if selected = policy
        roundtrip_started = Time.instant
        retired = roundtrip_policy!(state, hp, max_seq, offset, count, selected)
        roundtrip_ms += (Time.instant - roundtrip_started).total_milliseconds
        stats = add_stats(stats, retired)
      end
      offset += count
    end

    snapshot = ML::GGUF::Qwen35StateSnapshot.capture(state)
    elapsed_ms = (Time.instant - started).total_milliseconds
    {snapshot, first_id, first_logit, elapsed_ms, roundtrip_ms, stats}
  ensure
    release_state!(state)
  end
end

startup_started = Time.instant
gguf = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path)
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
startup_ms = (Time.instant - startup_started).total_milliseconds
hp = weights.hparams
raise "quality probe requires Qwen3.8 head dimension 256" unless hp.head_dim == 256

model_prompt = chat_mode ? ML::GGUF::Qwen35Chat.render_user_prompt(prompt, enable_thinking: false) : prompt
tokens = tokenizer.encode(model_prompt)
raise "prompt encoded to zero tokens" if tokens.empty?
minimum_max_seq = tokens.size + n_gen + 1
max_seq = requested_max_seq == 0 ? minimum_max_seq : requested_max_seq
raise "prompt plus continuation exceeds --max-seq" if max_seq < minimum_max_seq

snapshot, first_id, first_logit, prefill_ms, _exact_roundtrip_ms, _exact_stats = prefill_chunked(
  weights, tokens, max_seq, retire_chunk, nil,
)

# Exact greedy continuation and its full-logit teacher path form the oracle.
exact_state = ML::GGUF::Qwen35StateSnapshot.restore(snapshot, hp)
exact_ids = [first_id] of Int32
exact_logits = [] of Array(Float32)
begin
  (n_gen - 1).times do |step|
    break if exact_ids[-1] == tokenizer.eos_id
    logits = ML::GGUF::Qwen35CPU.forward(weights, exact_ids[-1], tokens.size + step, exact_state)
    exact_logits << logits
    exact_ids << top1(logits)[0]
  end
ensure
  release_state!(exact_state)
end

puts "qwen35_qbit_kv_quality_probe"
puts "  model=#{model_path}"
puts "  prompt=#{prompt.inspect} chat=#{chat_mode} prompt_tokens=#{tokens.size} requested_gen=#{n_gen} observed_gen=#{exact_ids.size} max_seq=#{max_seq} retire_chunk=#{retire_chunk}"
puts "  layers=#{hp.n_layer} full_attention_layers=#{hp.full_attention_layers.size} n_head_kv=#{hp.n_head_kv} head_dim=#{hp.head_dim}"
puts "  startup_ms=#{startup_ms.round(3)} prefill_ms=#{prefill_ms.round(3)} exact_first_id=#{first_id} exact_first_logit=#{first_logit.round(6)}"
puts "  exact_ids=#{exact_ids.join(',')} exact_text=#{tokenizer.decode(exact_ids).inspect}"

policies = precisions.map do |precision|
  QBitQualityPolicy.new("p#{precision}", precision, nil,
    {} of Int32 => ML::GGUF::QwenQBitAdaptiveKV::Tier)
end
adaptive_maps.each { |spec| policies << adaptive_policy(spec) }
if sweep_name = adaptive_sweep
  sweep_tier = adaptive_tier(sweep_name)
  raise "adaptive sweep tier must refine or replace p4" if sweep_tier.p4?
  hp.full_attention_layers.each do |layer|
    policies << QBitQualityPolicy.new(
      "adaptive[#{layer}=#{sweep_name.downcase}]", nil,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4, {layer => sweep_tier},
    )
  end
end

policies.each do |policy|
  quantized_snapshot, qbit_first_id, qbit_first_logit, qbit_prefill_ms, prefix_quantize_ms, prefix_stats = prefill_chunked(
    weights, tokens, max_seq, retire_chunk, policy,
  )

  free_state = ML::GGUF::Qwen35StateSnapshot.restore(quantized_snapshot, hp)
  free_ids = [qbit_first_id] of Int32
  free_append_stats = ML::GGUF::QwenQBitKVQuality::Stats.new(0_i64, 0_i64, 0_i64)
  begin
    exact_logits.size.times do |step|
      break if free_ids[-1] == tokenizer.eos_id
      pos = tokens.size + step
      next_id, _logit = ML::GGUF::Qwen35CPU.forward_top1(weights, free_ids[-1], pos, free_state)
      appended = roundtrip_policy!(free_state, hp, max_seq, pos, 1, policy)
      free_append_stats = add_stats(free_append_stats, appended)
      free_ids << next_id
    end
  ensure
    release_state!(free_state)
  end

  forced_state = ML::GGUF::Qwen35StateSnapshot.restore(quantized_snapshot, hp)
  forced_matches = 0
  min_logit_cosine = 1.0_f64
  max_logit_delta = 0.0_f32
  forced_append_stats = ML::GGUF::QwenQBitKVQuality::Stats.new(0_i64, 0_i64, 0_i64)
  begin
    exact_logits.size.times do |step|
      pos = tokens.size + step
      logits = ML::GGUF::Qwen35CPU.forward(weights, exact_ids[step], pos, forced_state)
      predicted_id, _predicted_logit = top1(logits)
      forced_matches += 1 if predicted_id == exact_ids[step + 1]
      min_logit_cosine = Math.min(min_logit_cosine, cosine(exact_logits[step], logits))
      max_logit_delta = Math.max(max_logit_delta, max_delta(exact_logits[step], logits))
      appended = roundtrip_policy!(forced_state, hp, max_seq, pos, 1, policy)
      forced_append_stats = add_stats(forced_append_stats, appended)
    end
  ensure
    release_state!(forced_state)
  end

  common_prefix = exact_ids.each_with_index.take_while { |id, i| free_ids[i]? == id }.size
  boundary_top1_match = qbit_first_id == first_id
  total_top1_matches = forced_matches + (boundary_top1_match ? 1 : 0)
  total_top1_count = exact_logits.size + 1
  prefix_ratio = prefix_stats.raw_bytes.to_f64 / prefix_stats.payload_bytes
  append_ratio = forced_append_stats.payload_bytes > 0 ? forced_append_stats.raw_bytes.to_f64 / forced_append_stats.payload_bytes : Float64::NAN
  puts "  #{policy.label} prefix_raw_bytes=#{prefix_stats.raw_bytes} prefix_payload_bytes=#{prefix_stats.payload_bytes} prefix_ratio=#{prefix_ratio.round(4)}x qbit_prefill_ms=#{qbit_prefill_ms.round(3)} prefix_cpu_roundtrip_ms=#{prefix_quantize_ms.round(3)}"
  puts "    boundary_top1_match=#{boundary_top1_match} qbit_first_id=#{qbit_first_id} qbit_first_logit=#{qbit_first_logit.round(6)} retire_order_top1=#{total_top1_matches}/#{total_top1_count}"
  puts "    free_prefix=#{common_prefix}/#{exact_ids.size} free_ids=#{free_ids.join(',')} free_text=#{tokenizer.decode(free_ids).inspect}"
  puts "    forced_top1=#{forced_matches}/#{exact_logits.size} min_logit_cosine=#{min_logit_cosine.round(9)} max_logit_delta=#{max_logit_delta.round(6)} appended_ratio=#{append_ratio.round(4)}x"
  puts "    append_payload_bytes=#{forced_append_stats.payload_bytes} free_append_payload_bytes=#{free_append_stats.payload_bytes}"
  GC.collect
end

gguf.close
