# Default-off quality falsifier for retire-then-pack uniform and adaptive QBit
# KV semantics.
#
# Each prefill chunk is consumed exactly, then its completed KV rows are
# quantized and reconstructed into the existing Float32 cache before the next
# chunk. Decode rows use the same retire-then-pack order. This models the values
# a future resident QBit attention kernel would observe without changing
# production cache ownership or routing.

require "json"
require "option_parser"

require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_state_snapshot"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen_qbit_kv_quality"
require "../src/ml/gguf/qwen_qbit_quality_metrics"

DEFAULT_QWEN38_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"

record QBitQualityPolicy,
  label : String,
  precision : Int32?,
  adaptive_default : ML::GGUF::QwenQBitAdaptiveKV::Tier?,
  adaptive_layer_tiers : Hash(Int32, ML::GGUF::QwenQBitAdaptiveKV::Tier),
  adaptive_head_tiers : Hash(ML::GGUF::QwenQBitKVQuality::HeadCoordinate, ML::GGUF::QwenQBitAdaptiveKV::Tier),
  selected_max_error : Float64?

record TokenECSPosition,
  position : Int32,
  expected_id : Int32,
  candidate_id : Int32,
  ecs : Float64

record TokenECSSummary,
  positions : Array(TokenECSPosition),
  mean : Float64,
  min : Float64,
  mismatch_count : Int32,
  mismatch_mean : Float64?,
  mismatch_min : Float64?

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_QWEN38_MODEL
prompt = "Explain in one sentence why the sky appears blue."
n_gen = 4
requested_max_seq = 0
precisions = [4, 5] of Int32
adaptive_maps = [] of String
selected_max_errors = [] of Float64
adaptive_sweep = nil.as(String?)
chat_mode = true
retire_chunk = 8
emit_json = true
summary_only = false

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_qbit_kv_quality_probe [options] [prompt]"
  parser.on("--model PATH", "Qwen GGUF path") { |value| model_path = value }
  parser.on("--gen N", "Generated tokens including the exact prefill token (default: 4)") { |value| n_gen = value.to_i }
  parser.on("--max-seq N", "Cache capacity; 0 selects prompt+gen+1 (default: 0)") { |value| requested_max_seq = value.to_i }
  parser.on("--precisions LIST", "Comma-separated QBit planes (default: 4,5)") do |value|
    precisions = value.split(',').map(&.to_i32)
  end
  parser.on("--no-uniform", "Skip uniform p4/p5 variants") { precisions.clear }
  parser.on("--adaptive-map MAP", "Add p4-default layer/head escapes, e.g. 27=bf16,27:k0=p4") do |value|
    adaptive_maps << value
  end
  parser.on("--adaptive-sweep TIER", "Try one p5, bf16, or f32 escape layer at a time") do |value|
    adaptive_sweep = value
  end
  parser.on("--selected-max-error X", "Add row-local p4/p5/BF16 selector with normalized max-error bound") do |value|
    selected_max_errors << value.to_f64
  end
  parser.on("--retire-chunk N", "Completed prefill rows packed per chunk (default: 8)") { |value| retire_chunk = value.to_i }
  parser.on("--no-json", "Suppress per-policy JSON; keep the human-readable quality summary") { emit_json = false }
  parser.on("--summary-only", "Print one compact quality line per policy and suppress JSON") { summary_only = true }
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
raise "no quality variant selected" if precisions.empty? && adaptive_maps.empty? && adaptive_sweep.nil? && selected_max_errors.empty?
raise "--retire-chunk must be positive" unless retire_chunk > 0
precisions.each do |precision|
  raise "precision must be p4 or p5" unless precision == 4 || precision == 5
end
raise "duplicate precisions" unless precisions.uniq.size == precisions.size
selected_max_errors.each do |bound|
  raise "selected max-error bound must be finite and positive" unless bound.finite? && bound > 0.0
end
raise "duplicate selected max-error bounds" unless selected_max_errors.uniq.size == selected_max_errors.size

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
  head_tiers = {} of ML::GGUF::QwenQBitKVQuality::HeadCoordinate => ML::GGUF::QwenQBitAdaptiveKV::Tier
  spec.split(',').each do |entry|
    target_text, tier_text = entry.split('=', 2)
    raise "adaptive map entries must use TARGET=TIER" unless target_text && tier_text
    tier = adaptive_tier(tier_text)
    if match = /\A(-?\d+):([kKvV])(\d+)\z/.match(target_text)
      layer = match[1].to_i32
      side = match[2].downcase == "k" ? ML::GGUF::QwenQBitKVQuality::KVSide::K : ML::GGUF::QwenQBitKVQuality::KVSide::V
      head = match[3].to_i32
      coordinate = ML::GGUF::QwenQBitKVQuality::HeadCoordinate.new(layer, side, head)
      raise "duplicate adaptive head #{target_text}" if head_tiers.has_key?(coordinate)
      head_tiers[coordinate] = tier
    else
      layer = target_text.to_i32
      raise "duplicate adaptive layer #{layer}" if layer_tiers.has_key?(layer)
      layer_tiers[layer] = tier
    end
  end
  raise "adaptive map cannot be empty" if layer_tiers.empty? && head_tiers.empty?
  QBitQualityPolicy.new("adaptive[#{spec}]", nil,
    ML::GGUF::QwenQBitAdaptiveKV::Tier::P4, layer_tiers, head_tiers, nil)
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

def token_ecs_summary(embedding_weight : ML::GGUF::QuantWeight,
                      expected_ids : Array(Int32),
                      candidate_ids : Array(Int32),
                      embedding_cache : Hash(Int32, Array(Float32))) : TokenECSSummary
  raise "token ECS requires identical non-empty positions" unless expected_ids.size == candidate_ids.size && !expected_ids.empty?
  aligned = expected_ids.size

  positions = Array(TokenECSPosition).new(aligned)
  total = 0.0_f64
  minimum = 1.0_f64
  mismatch_total = 0.0_f64
  mismatch_minimum = 1.0_f64
  mismatch_count = 0_i32

  aligned.times do |position|
    expected_id = expected_ids[position]
    candidate_id = candidate_ids[position]
    ecs = if expected_id == candidate_id
            1.0_f64
          else
            expected_embedding = embedding_cache[expected_id]? || begin
              embedding = ML::GGUF::Qwen35CPU.embedding_lookup(embedding_weight, expected_id)
              embedding_cache[expected_id] = embedding
              embedding
            end
            candidate_embedding = embedding_cache[candidate_id]? || begin
              embedding = ML::GGUF::Qwen35CPU.embedding_lookup(embedding_weight, candidate_id)
              embedding_cache[candidate_id] = embedding
              embedding
            end
            ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(expected_embedding, candidate_embedding)
          end

    total += ecs
    minimum = Math.min(minimum, ecs)
    if expected_id != candidate_id
      mismatch_count += 1
      mismatch_total += ecs
      mismatch_minimum = Math.min(mismatch_minimum, ecs)
    end
    positions << TokenECSPosition.new(position.to_i32, expected_id, candidate_id, ecs)
  end

  TokenECSSummary.new(
    positions,
    total / aligned,
    minimum,
    mismatch_count,
    mismatch_count > 0 ? mismatch_total / mismatch_count : nil,
    mismatch_count > 0 ? mismatch_minimum : nil,
  )
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
                      policy : QBitQualityPolicy,
                      counts : ML::GGUF::QwenQBitKVQuality::TierCounts) : ML::GGUF::QwenQBitKVQuality::Stats
  if precision = policy.precision
    ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span!(
      state.layers, hp.full_attention_layers, max_seq,
      hp.n_head_kv, hp.head_dim, start_pos, token_count, precision,
    )
  elsif max_error = policy.selected_max_error
    ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span_selected!(
      state.layers, hp.full_attention_layers, max_seq,
      hp.n_head_kv, hp.head_dim, start_pos, token_count, max_error, counts,
    )
  else
    ML::GGUF::QwenQBitKVQuality.roundtrip_layers_span_adaptive!(
      state.layers, hp.full_attention_layers, max_seq,
      hp.n_head_kv, hp.head_dim, start_pos, token_count,
      policy.adaptive_default.not_nil!, policy.adaptive_layer_tiers,
      policy.adaptive_head_tiers,
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
  tier_counts = ML::GGUF::QwenQBitKVQuality::TierCounts.new
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
        retired = roundtrip_policy!(state, hp, max_seq, offset, count, selected, tier_counts)
        roundtrip_ms += (Time.instant - roundtrip_started).total_milliseconds
        stats = add_stats(stats, retired)
      end
      offset += count
    end

    snapshot = ML::GGUF::Qwen35StateSnapshot.capture(state)
    elapsed_ms = (Time.instant - started).total_milliseconds
    {snapshot, first_id, first_logit, elapsed_ms, roundtrip_ms, stats, tier_counts}
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

snapshot, first_id, first_logit, prefill_ms, _exact_roundtrip_ms, _exact_stats, _exact_tier_counts = prefill_chunked(
  weights, tokens, max_seq, retire_chunk, nil,
)

# Exact greedy continuation and its full-logit teacher path form the oracle.
exact_state = ML::GGUF::Qwen35StateSnapshot.restore(snapshot, hp)
exact_ids = [first_id] of Int32
exact_logits = [] of Array(Float32)
exact_top2s = [] of ML::GGUF::QwenQBitQualityMetrics::Top2
begin
  (n_gen - 1).times do |step|
    break if exact_ids[-1] == tokenizer.eos_id
    logits = ML::GGUF::Qwen35CPU.forward(weights, exact_ids[-1], tokens.size + step, exact_state)
    top2 = ML::GGUF::QwenQBitQualityMetrics.top2(logits)
    exact_logits << logits
    exact_top2s << top2
    exact_ids << top2.first_id
  end
ensure
  release_state!(exact_state)
end
raise "exact continuation ended before a top-2 decode step" if exact_top2s.empty?

puts "qwen35_qbit_kv_quality_probe"
puts "  model=#{model_path}"
puts "  prompt=#{prompt.inspect} chat=#{chat_mode} prompt_tokens=#{tokens.size} requested_gen=#{n_gen} observed_gen=#{exact_ids.size} max_seq=#{max_seq} retire_chunk=#{retire_chunk}"
puts "  layers=#{hp.n_layer} full_attention_layers=#{hp.full_attention_layers.size} n_head_kv=#{hp.n_head_kv} head_dim=#{hp.head_dim}"
puts "  startup_ms=#{startup_ms.round(3)} prefill_ms=#{prefill_ms.round(3)} exact_first_id=#{first_id} exact_first_logit=#{first_logit.round(6)}"
puts "  exact_ids=#{exact_ids.join(',')} exact_text=#{tokenizer.decode(exact_ids).inspect}"

policies = precisions.map do |precision|
  QBitQualityPolicy.new("p#{precision}", precision, nil,
    {} of Int32 => ML::GGUF::QwenQBitAdaptiveKV::Tier,
    {} of ML::GGUF::QwenQBitKVQuality::HeadCoordinate => ML::GGUF::QwenQBitAdaptiveKV::Tier,
    nil)
end
adaptive_maps.each { |spec| policies << adaptive_policy(spec) }
selected_max_errors.each do |bound|
  policies << QBitQualityPolicy.new(
    "selected[max_error=#{bound}]", nil, nil,
    {} of Int32 => ML::GGUF::QwenQBitAdaptiveKV::Tier,
    {} of ML::GGUF::QwenQBitKVQuality::HeadCoordinate => ML::GGUF::QwenQBitAdaptiveKV::Tier,
    bound,
  )
end
if sweep_name = adaptive_sweep
  sweep_tier = adaptive_tier(sweep_name)
  raise "adaptive sweep tier must refine or replace p4" if sweep_tier.p4?
  hp.full_attention_layers.each do |layer|
    policies << QBitQualityPolicy.new(
      "adaptive[#{layer}=#{sweep_name.downcase}]", nil,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4, {layer => sweep_tier},
      {} of ML::GGUF::QwenQBitKVQuality::HeadCoordinate => ML::GGUF::QwenQBitAdaptiveKV::Tier,
      nil,
    )
  end
end

policies.each do |policy|
  quantized_snapshot, qbit_first_id, qbit_first_logit, qbit_prefill_ms, prefix_quantize_ms, prefix_stats, prefix_tier_counts = prefill_chunked(
    weights, tokens, max_seq, retire_chunk, policy,
  )

  free_state = ML::GGUF::Qwen35StateSnapshot.restore(quantized_snapshot, hp)
  free_ids = [qbit_first_id] of Int32
  free_top2s = [] of ML::GGUF::QwenQBitQualityMetrics::Top2
  free_append_stats = ML::GGUF::QwenQBitKVQuality::Stats.new(0_i64, 0_i64, 0_i64)
  free_tier_counts = ML::GGUF::QwenQBitKVQuality::TierCounts.new
  begin
    exact_logits.size.times do |step|
      break if free_ids[-1] == tokenizer.eos_id
      pos = tokens.size + step
      free_first_id, free_first_logit, free_second_id, free_second_logit = ML::GGUF::Qwen35CPU.forward_top2(
        weights, free_ids[-1], pos, free_state,
      )
      top2 = ML::GGUF::QwenQBitQualityMetrics::Top2.new(
        free_first_id, free_first_logit, free_second_id, free_second_logit,
      )
      free_top2s << top2
      appended = roundtrip_policy!(free_state, hp, max_seq, pos, 1, policy, free_tier_counts)
      free_append_stats = add_stats(free_append_stats, appended)
      free_ids << top2.first_id
    end
  ensure
    release_state!(free_state)
  end

  forced_state = ML::GGUF::Qwen35StateSnapshot.restore(quantized_snapshot, hp)
  forced_matches = 0
  min_logit_cosine = 1.0_f64
  max_logit_delta = 0.0_f32
  forced_top2s = [] of ML::GGUF::QwenQBitQualityMetrics::Top2
  top2_ranked_matches = 0_i32
  top2_set_overlap = 0_i32
  top2_set_matches = 0_i32
  exact_top1_covered = 0_i32
  exact_top2_covered = 0_i32
  top2_rescues = 0_i32
  first_top2_rank_mismatch_step = nil.as(Int32?)
  max_top2_logit_delta = 0.0_f32
  max_top2_margin_delta = 0.0_f32
  min_exact_top2_margin = Float32::INFINITY
  forced_append_stats = ML::GGUF::QwenQBitKVQuality::Stats.new(0_i64, 0_i64, 0_i64)
  forced_tier_counts = ML::GGUF::QwenQBitKVQuality::TierCounts.new
  begin
    exact_logits.size.times do |step|
      pos = tokens.size + step
      logits = ML::GGUF::Qwen35CPU.forward(weights, exact_ids[step], pos, forced_state)
      candidate_top2 = ML::GGUF::QwenQBitQualityMetrics.top2(logits)
      exact_top2 = exact_top2s[step]
      comparison = ML::GGUF::QwenQBitQualityMetrics.compare_top2(exact_top2, candidate_top2)
      forced_top2s << candidate_top2
      top1_matches = candidate_top2.first_id == exact_ids[step + 1]
      forced_matches += 1 if top1_matches
      top2_ranked_matches += comparison.ranked_matches
      top2_set_overlap += comparison.set_overlap
      top2_set_matches += 1 if comparison.set_overlap == 2
      exact_top1_covered += 1 if comparison.exact_top1_covered
      exact_top2_covered += 1 if comparison.exact_top2_covered
      top2_rescues += 1 if !top1_matches && comparison.exact_top1_covered
      first_top2_rank_mismatch_step ||= step.to_i32 unless comparison.ranked_matches == 2
      max_top2_logit_delta = Math.max(max_top2_logit_delta,
        Math.max(comparison.first_logit_delta, comparison.second_logit_delta))
      max_top2_margin_delta = Math.max(max_top2_margin_delta, comparison.margin_delta)
      min_exact_top2_margin = Math.min(min_exact_top2_margin, exact_top2.margin)
      min_logit_cosine = Math.min(min_logit_cosine, cosine(exact_logits[step], logits))
      max_logit_delta = Math.max(max_logit_delta, max_delta(exact_logits[step], logits))
      appended = roundtrip_policy!(forced_state, hp, max_seq, pos, 1, policy, forced_tier_counts)
      forced_append_stats = add_stats(forced_append_stats, appended)
    end
  ensure
    release_state!(forced_state)
  end

  common_prefix = exact_ids.each_with_index.take_while { |id, i| free_ids[i]? == id }.size
  boundary_top1_match = qbit_first_id == first_id
  total_top1_matches = forced_matches + (boundary_top1_match ? 1 : 0)
  total_top1_count = exact_logits.size + 1
  forced_ids = [qbit_first_id] + forced_top2s.map(&.first_id)
  embedding_cache = {} of Int32 => Array(Float32)
  teacher_token_ecs = token_ecs_summary(weights.output, exact_ids, forced_ids, embedding_cache)
  prefix_ratio = prefix_stats.raw_bytes.to_f64 / prefix_stats.payload_bytes
  append_ratio = forced_append_stats.payload_bytes > 0 ? forced_append_stats.raw_bytes.to_f64 / forced_append_stats.payload_bytes : Float64::NAN
  exact_text = tokenizer.decode(exact_ids)
  free_text = tokenizer.decode(free_ids)
  if summary_only
    puts "QBIT_QUALITY_SUMMARY policy=#{policy.label.inspect} ratio=#{prefix_ratio.round(4)}x top1=#{total_top1_matches}/#{total_top1_count} ranked_top2=#{top2_ranked_matches}/#{2 * exact_top2s.size} top2_overlap=#{top2_set_overlap}/#{2 * exact_top2s.size} exact_top1_covered=#{exact_top1_covered}/#{exact_top2s.size} ecs=#{teacher_token_ecs.mean.round(6)} max_logit_delta=#{max_logit_delta.round(6)} free_prefix=#{common_prefix}/#{exact_ids.size} free_text=#{free_text.inspect}"
    GC.collect
    next
  end
  puts "  #{policy.label} prefix_raw_bytes=#{prefix_stats.raw_bytes} prefix_payload_bytes=#{prefix_stats.payload_bytes} prefix_ratio=#{prefix_ratio.round(4)}x qbit_prefill_ms=#{qbit_prefill_ms.round(3)} prefix_cpu_roundtrip_ms=#{prefix_quantize_ms.round(3)}"
  puts "    boundary_top1_match=#{boundary_top1_match} qbit_first_id=#{qbit_first_id} qbit_first_logit=#{qbit_first_logit.round(6)} retire_order_top1=#{total_top1_matches}/#{total_top1_count}"
  puts "    free_prefix=#{common_prefix}/#{exact_ids.size} free_ids=#{free_ids.join(',')} free_text=#{free_text.inspect}"
  puts "    forced_top1=#{forced_matches}/#{exact_logits.size} min_logit_cosine=#{min_logit_cosine.round(9)} max_logit_delta=#{max_logit_delta.round(6)} appended_ratio=#{append_ratio.round(4)}x"
  puts "    teacher_top2_ranked=#{top2_ranked_matches}/#{2 * exact_top2s.size} teacher_top2_set_overlap=#{top2_set_overlap}/#{2 * exact_top2s.size} teacher_top2_set_matches=#{top2_set_matches}/#{exact_top2s.size} exact_top1_covered=#{exact_top1_covered}/#{exact_top2s.size} exact_top2_covered=#{exact_top2_covered}/#{exact_top2s.size} top2_rescues=#{top2_rescues} first_top2_rank_mismatch_step=#{first_top2_rank_mismatch_step} min_exact_top2_margin=#{min_exact_top2_margin.round(6)} max_top2_logit_delta=#{max_top2_logit_delta.round(6)} max_top2_margin_delta=#{max_top2_margin_delta.round(6)}"
  puts "    teacher_token_ecs_basis=output.weight teacher_token_ecs_mean=#{teacher_token_ecs.mean.round(6)} teacher_token_ecs_min=#{teacher_token_ecs.min.round(6)} teacher_token_ecs_mismatches=#{teacher_token_ecs.mismatch_count} teacher_token_ecs_mismatch_mean=#{teacher_token_ecs.mismatch_mean.try(&.round(6))} teacher_token_ecs_mismatch_min=#{teacher_token_ecs.mismatch_min.try(&.round(6))}"
  teacher_token_ecs.positions.each do |row|
    next if row.expected_id == row.candidate_id
    puts "      teacher_token_ecs_position=#{row.position} expected=#{tokenizer.decode([row.expected_id]).inspect}(#{row.expected_id}) candidate=#{tokenizer.decode([row.candidate_id]).inspect}(#{row.candidate_id}) ecs=#{row.ecs.round(6)}"
  end
  puts "    append_payload_bytes=#{forced_append_stats.payload_bytes} free_append_payload_bytes=#{free_append_stats.payload_bytes}"
  if policy.selected_max_error
    puts "    prefix_tiers=p4:#{prefix_tier_counts.p4},p5:#{prefix_tier_counts.p5},bf16:#{prefix_tier_counts.bf16} forced_append_tiers=p4:#{forced_tier_counts.p4},p5:#{forced_tier_counts.p5},bf16:#{forced_tier_counts.bf16} free_append_tiers=p4:#{free_tier_counts.p4},p5:#{free_tier_counts.p5},bf16:#{free_tier_counts.bf16}"
  end
  unless emit_json
    GC.collect
    next
  end
  payload = JSON.build do |json|
    json.object do
      json.field "schema", "qwen-qbit-quality-v1"
      json.field "model", File.basename(model_path)
      json.field "prompt", prompt
      json.field "policy", policy.label
      json.field "retire_chunk", retire_chunk
      json.field "exact_text", exact_text
      json.field "candidate_text", free_text
      json.field "exact_ids", exact_ids
      json.field "candidate_ids", free_ids
      json.field "free_common_prefix", common_prefix
      json.field "retire_order_top1_matches", total_top1_matches
      json.field "retire_order_top1_count", total_top1_count
      json.field "teacher_top2_ranked_matches", top2_ranked_matches
      json.field "teacher_top2_ranked_count", 2 * exact_top2s.size
      json.field "teacher_top2_set_overlap", top2_set_overlap
      json.field "teacher_top2_set_overlap_count", 2 * exact_top2s.size
      json.field "teacher_top2_set_matches", top2_set_matches
      json.field "teacher_top2_steps", exact_top2s.size
      json.field "teacher_exact_top1_covered", exact_top1_covered
      json.field "teacher_exact_top2_covered", exact_top2_covered
      json.field "teacher_top2_rescues", top2_rescues
      json.field "teacher_token_ecs_mean", teacher_token_ecs.mean
      json.field "teacher_token_ecs_min", teacher_token_ecs.min
      json.field "teacher_token_ecs_mismatch_count", teacher_token_ecs.mismatch_count
      json.field "teacher_token_ecs_mismatch_mean", teacher_token_ecs.mismatch_mean
      json.field "teacher_token_ecs_mismatch_min", teacher_token_ecs.mismatch_min
      json.field "teacher_token_ecs_basis", "output.weight"
      json.field "first_top2_rank_mismatch_step", first_top2_rank_mismatch_step
      json.field "min_exact_top2_margin", min_exact_top2_margin
      json.field "max_top2_logit_delta", max_top2_logit_delta
      json.field "max_top2_margin_delta", max_top2_margin_delta
      json.field "min_logit_cosine", min_logit_cosine
      json.field "max_logit_delta", max_logit_delta
      json.field "prefix_raw_bytes", prefix_stats.raw_bytes
      json.field "prefix_payload_bytes", prefix_stats.payload_bytes
      json.field "prefix_ratio", prefix_ratio
      json.field "selected_max_error", policy.selected_max_error
      json.field "prefix_tier_counts" do
        json.object do
          json.field "p4", prefix_tier_counts.p4
          json.field "p5", prefix_tier_counts.p5
          json.field "bf16", prefix_tier_counts.bf16
          json.field "f32", prefix_tier_counts.f32
        end
      end
      json.field "forced_append_tier_counts" do
        json.object do
          json.field "p4", forced_tier_counts.p4
          json.field "p5", forced_tier_counts.p5
          json.field "bf16", forced_tier_counts.bf16
          json.field "f32", forced_tier_counts.f32
        end
      end
      json.field "free_append_tier_counts" do
        json.object do
          json.field "p4", free_tier_counts.p4
          json.field "p5", free_tier_counts.p5
          json.field "bf16", free_tier_counts.bf16
          json.field "f32", free_tier_counts.f32
        end
      end
      json.field "teacher_token_ecs_mismatches" do
        json.array do
          teacher_token_ecs.positions.each do |row|
            next if row.expected_id == row.candidate_id
            json.object do
              json.field "position", row.position
              json.field "expected_id", row.expected_id
              json.field "candidate_id", row.candidate_id
              json.field "expected_piece", tokenizer.decode([row.expected_id])
              json.field "candidate_piece", tokenizer.decode([row.candidate_id])
              json.field "ecs", row.ecs
            end
          end
        end
      end
      json.field "exact_top2_ids" do
        json.array do
          exact_top2s.each do |row|
            json.array do
              json.number row.first_id
              json.number row.second_id
            end
          end
        end
      end
      json.field "candidate_teacher_top2_ids" do
        json.array do
          forced_top2s.each do |row|
            json.array do
              json.number row.first_id
              json.number row.second_id
            end
          end
        end
      end
      json.field "candidate_free_top2_ids" do
        json.array do
          free_top2s.each do |row|
            json.array do
              json.number row.first_id
              json.number row.second_id
            end
          end
        end
      end
    end
  end
  puts "QBIT_QUALITY_JSON=#{payload}"
  GC.collect
end

gguf.close
