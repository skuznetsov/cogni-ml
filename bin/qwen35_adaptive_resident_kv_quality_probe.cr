# Held-out quality probe for the real resident adaptive-QBit GPU KV path.
#
# The exact oracle uses the normal Float32 KV owner. Each resident policy gets
# two fresh states: one for an independent greedy response and one for the
# exact teacher trajectory. Resident states are never snapshotted or expanded
# through an intermediate Float32 KV cache.

require "json"
require "option_parser"

require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen_qbit_quality_metrics"

DEFAULT_QWEN38_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_RESIDENT_MAP = "p4;27=bf16,43=bf16,47=bf16,51=bf16"

record ResidentTokenECS,
  mean : Float64,
  min : Float64,
  mismatch_count : Int32,
  mismatch_mean : Float64?,
  mismatch_min : Float64?

private def release_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
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

private def with_adaptive_env(map : String?, &)
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

private def f32_owner_layers(state : ML::GGUF::Qwen35CPU::State,
                             hp : ML::GGUF::Qwen35Hparams) : Array(Int32)
  hp.full_attention_layers.select do |layer_index|
    layer = state.layers[layer_index]
    !!(layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf)
  end
end

private def verify_resident_state!(state : ML::GGUF::Qwen35CPU::State,
                                   hp : ML::GGUF::Qwen35Hparams,
                                   expected_tokens : Int32) : Nil
  unless state.adaptive_kv_layer_indices == hp.full_attention_layers
    raise "resident adaptive KV does not own every full-attention layer"
  end
  owners = f32_owner_layers(state, hp)
  raise "resident adaptive KV has Float32 owners at layers #{owners}" unless owners.empty?
  hp.full_attention_layers.each do |layer_index|
    cache = state.layers[layer_index].adaptive_kv
    raise "resident adaptive KV is missing layer #{layer_index}" unless cache
    unless cache.cache_len == expected_tokens
      raise "resident adaptive KV layer #{layer_index} has #{cache.cache_len} tokens, expected #{expected_tokens}"
    end
  end
end

private def prepare_prefill(weights : ML::GGUF::Qwen35Weights,
                            tokens : Array(Int32),
                            max_seq : Int32,
                            resident_map : String?)
  hp = weights.hparams
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  first_id = -1_i32
  first_logit = Float32::NAN
  started = Time.instant
  begin
    with_adaptive_env(resident_map) do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      first_id, first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
        weights, tokens, 0, state,
      )
    end
    verify_resident_state!(state, hp, tokens.size.to_i32) if resident_map
    {state, first_id, first_logit, (Time.instant - started).total_milliseconds}
  rescue ex
    release_state!(state)
    raise ex
  end
end

private def token_ecs_summary(output_weight : ML::GGUF::QuantWeight,
                              expected_ids : Array(Int32),
                              candidate_ids : Array(Int32)) : ResidentTokenECS
  unless !expected_ids.empty? && expected_ids.size == candidate_ids.size
    raise "resident token ECS requires identical non-empty positions"
  end
  cache = {} of Int32 => Array(Float32)
  total = 0.0_f64
  minimum = 1.0_f64
  mismatch_total = 0.0_f64
  mismatch_minimum = 1.0_f64
  mismatch_count = 0_i32

  expected_ids.each_with_index do |expected_id, index|
    candidate_id = candidate_ids[index]
    ecs = if expected_id == candidate_id
            1.0_f64
          else
            expected = cache[expected_id]? || begin
              row = ML::GGUF::Qwen35CPU.embedding_lookup(output_weight, expected_id)
              cache[expected_id] = row
              row
            end
            candidate = cache[candidate_id]? || begin
              row = ML::GGUF::Qwen35CPU.embedding_lookup(output_weight, candidate_id)
              cache[candidate_id] = row
              row
            end
            ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(expected, candidate)
          end
    total += ecs
    minimum = Math.min(minimum, ecs)
    if expected_id != candidate_id
      mismatch_count += 1
      mismatch_total += ecs
      mismatch_minimum = Math.min(mismatch_minimum, ecs)
    end
  end

  ResidentTokenECS.new(
    total / expected_ids.size,
    minimum,
    mismatch_count,
    mismatch_count > 0 ? mismatch_total / mismatch_count : nil,
    mismatch_count > 0 ? mismatch_minimum : nil,
  )
end

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_QWEN38_MODEL
prompt = "Explain in one sentence why the sky appears blue."
n_gen = 128
requested_max_seq = 0
resident_maps = [] of String
chat_mode = true

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_adaptive_resident_kv_quality_probe [options] [prompt]"
  parser.on("--model PATH", "Qwen GGUF path") { |value| model_path = value }
  parser.on("--gen N", "Maximum generated tokens including prefill top-1") { |value| n_gen = value.to_i }
  parser.on("--max-seq N", "Cache capacity; 0 selects prompt+gen+1") { |value| requested_max_seq = value.to_i }
  parser.on("--resident-map MAP", "Resident tier map; may be repeated") { |value| resident_maps << value }
  parser.on("--raw", "Do not render the Qwen chat template") { chat_mode = false }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end
prompt = ARGV.join(" ") unless ARGV.empty?
resident_maps << DEFAULT_RESIDENT_MAP if resident_maps.empty?

raise "model does not exist: #{model_path}" unless File.file?(model_path)
raise "--gen must be at least 2" unless n_gen >= 2
raise "--max-seq cannot be negative" if requested_max_seq < 0
raise "resident map cannot be empty" if resident_maps.any?(&.strip.empty?)
raise "duplicate resident maps" unless resident_maps.uniq.size == resident_maps.size
raise "Metal is unavailable" unless ML::GGUF::Qwen35Metal.available?

startup_started = Time.instant
gguf = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path)
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
startup_ms = (Time.instant - startup_started).total_milliseconds
hp = weights.hparams
raise "resident quality probe requires Qwen3.8 head dimension 256" unless hp.head_dim == 256

model_prompt = chat_mode ? ML::GGUF::Qwen35Chat.render_user_prompt(prompt, enable_thinking: false) : prompt
tokens = tokenizer.encode(model_prompt)
raise "prompt encoded to zero tokens" if tokens.empty?
minimum_max_seq = tokens.size + n_gen + 1
max_seq = requested_max_seq == 0 ? minimum_max_seq : requested_max_seq
raise "prompt plus continuation exceeds --max-seq" if max_seq < minimum_max_seq

exact_state, exact_first_id, exact_first_logit, exact_prefill_ms = prepare_prefill(
  weights, tokens, max_seq, nil,
)
exact_ids = [exact_first_id] of Int32
exact_top2s = [] of ML::GGUF::QwenQBitQualityMetrics::Top2
exact_decode_started = Time.instant
begin
  (n_gen - 1).times do |step|
    break if exact_ids[-1] == tokenizer.eos_id
    first, first_logit, second, second_logit = ML::GGUF::Qwen35CPU.forward_top2(
      weights, exact_ids[-1], tokens.size + step, exact_state,
    )
    exact_top2s << ML::GGUF::QwenQBitQualityMetrics::Top2.new(
      first, first_logit, second, second_logit,
    )
    exact_ids << first
  end
ensure
  release_state!(exact_state)
end
exact_decode_ms = (Time.instant - exact_decode_started).total_milliseconds
raise "exact continuation ended before a top-2 decode step" if exact_top2s.empty?
exact_text = tokenizer.decode(exact_ids)

puts "qwen35_adaptive_resident_kv_quality_probe"
puts "  model=#{model_path}"
puts "  prompt=#{prompt.inspect} chat=#{chat_mode} prompt_tokens=#{tokens.size} requested_gen=#{n_gen} observed_gen=#{exact_ids.size} max_seq=#{max_seq}"
puts "  layers=#{hp.n_layer} full_attention_layers=#{hp.full_attention_layers.size} n_head_kv=#{hp.n_head_kv} head_dim=#{hp.head_dim}"
puts "  startup_ms=#{startup_ms.round(3)} exact_prefill_ms=#{exact_prefill_ms.round(3)} exact_decode_ms=#{exact_decode_ms.round(3)} exact_first_id=#{exact_first_id} exact_first_logit=#{exact_first_logit.round(6)}"
puts "  exact_ids=#{exact_ids.join(',')} exact_text=#{exact_text.inspect}"

resident_maps.each do |resident_map|
  free_state, resident_first_id, resident_first_logit, resident_prefill_ms = prepare_prefill(
    weights, tokens, max_seq, resident_map,
  )
  free_ids = [resident_first_id] of Int32
  free_top2s = [] of ML::GGUF::QwenQBitQualityMetrics::Top2
  resident_bytes = hp.full_attention_layers.sum(0_i64) do |layer_index|
    free_state.layers[layer_index].adaptive_kv.not_nil!.compressed_bytes
  end
  raw_bytes = hp.full_attention_layers.size.to_i64 * 2_i64 * max_seq.to_i64 *
              hp.n_head_kv.to_i64 * hp.head_dim.to_i64 * sizeof(Float32)
  free_decode_started = Time.instant
  begin
    with_adaptive_env(resident_map) do
      (n_gen - 1).times do |step|
        break if free_ids[-1] == tokenizer.eos_id
        first, first_logit, second, second_logit = ML::GGUF::Qwen35CPU.forward_top2(
          weights, free_ids[-1], tokens.size + step, free_state,
        )
        free_top2s << ML::GGUF::QwenQBitQualityMetrics::Top2.new(
          first, first_logit, second, second_logit,
        )
        free_ids << first
        verify_resident_state!(free_state, hp, (tokens.size + step + 1).to_i32)
      end
    end
  ensure
    release_state!(free_state)
  end
  free_decode_ms = (Time.instant - free_decode_started).total_milliseconds

  forced_state, forced_first_id, forced_first_logit, forced_prefill_ms = prepare_prefill(
    weights, tokens, max_seq, resident_map,
  )
  forced_ids = [forced_first_id] of Int32
  forced_top2s = [] of ML::GGUF::QwenQBitQualityMetrics::Top2
  forced_matches = 0_i32
  top2_ranked_matches = 0_i32
  top2_set_overlap = 0_i32
  exact_top1_covered = 0_i32
  exact_top2_covered = 0_i32
  top2_rescues = 0_i32
  first_top2_rank_mismatch_step = nil.as(Int32?)
  max_top2_logit_delta = 0.0_f32
  max_top2_margin_delta = 0.0_f32
  min_exact_top2_margin = Float32::INFINITY
  final_f32_owner_layers = [] of Int32
  resident_cache_consistent = false
  forced_decode_started = Time.instant
  begin
    with_adaptive_env(resident_map) do
      exact_top2s.each_with_index do |exact_top2, step|
        first, first_logit, second, second_logit = ML::GGUF::Qwen35CPU.forward_top2(
          weights, exact_ids[step], tokens.size + step, forced_state,
        )
        candidate_top2 = ML::GGUF::QwenQBitQualityMetrics::Top2.new(
          first, first_logit, second, second_logit,
        )
        forced_top2s << candidate_top2
        forced_ids << first
        comparison = ML::GGUF::QwenQBitQualityMetrics.compare_top2(exact_top2, candidate_top2)
        top1_match = first == exact_ids[step + 1]
        forced_matches += 1 if top1_match
        top2_ranked_matches += comparison.ranked_matches
        top2_set_overlap += comparison.set_overlap
        exact_top1_covered += 1 if comparison.exact_top1_covered
        exact_top2_covered += 1 if comparison.exact_top2_covered
        top2_rescues += 1 if !top1_match && comparison.exact_top1_covered
        first_top2_rank_mismatch_step ||= step.to_i32 unless comparison.ranked_matches == 2
        max_top2_logit_delta = Math.max(max_top2_logit_delta,
          Math.max(comparison.first_logit_delta, comparison.second_logit_delta))
        max_top2_margin_delta = Math.max(max_top2_margin_delta, comparison.margin_delta)
        min_exact_top2_margin = Math.min(min_exact_top2_margin, exact_top2.margin)
        verify_resident_state!(forced_state, hp, (tokens.size + step + 1).to_i32)
      end
      final_f32_owner_layers = f32_owner_layers(forced_state, hp)
      resident_cache_consistent = hp.full_attention_layers.all? do |layer_index|
        forced_state.layers[layer_index].adaptive_kv.not_nil!.cache_len ==
          tokens.size + exact_top2s.size
      end
    end
  ensure
    release_state!(forced_state)
  end
  forced_decode_ms = (Time.instant - forced_decode_started).total_milliseconds

  boundary_top1_match = resident_first_id == exact_first_id
  total_top1_matches = forced_matches + (boundary_top1_match ? 1 : 0)
  total_top1_count = exact_top2s.size + 1
  common_prefix = exact_ids.each_with_index.take_while { |id, index| free_ids[index]? == id }.size
  teacher_token_ecs = token_ecs_summary(weights.output, exact_ids, forced_ids)
  free_text = tokenizer.decode(free_ids)
  policy = "resident[#{resident_map}]"
  density = raw_bytes.to_f64 / resident_bytes

  puts "  #{policy} density=#{density.round(4)}x resident_prefill_ms=#{resident_prefill_ms.round(3)} forced_prefill_ms=#{forced_prefill_ms.round(3)} free_decode_ms=#{free_decode_ms.round(3)} forced_decode_ms=#{forced_decode_ms.round(3)}"
  puts "    retire_order_top1=#{total_top1_matches}/#{total_top1_count} ranked_top2=#{top2_ranked_matches}/#{2 * exact_top2s.size} top2_overlap=#{top2_set_overlap}/#{2 * exact_top2s.size} exact_top1_covered=#{exact_top1_covered}/#{exact_top2s.size} ecs=#{teacher_token_ecs.mean.round(6)}"
  puts "    free_prefix=#{common_prefix}/#{exact_ids.size} free_ids=#{free_ids.join(',')} free_text=#{free_text.inspect}"

  payload = JSON.build do |json|
    json.object do
      json.field "schema", "qwen-qbit-quality-v1"
      json.field "execution_mode", "resident_gpu"
      json.field "model", File.basename(model_path)
      json.field "prompt", prompt
      json.field "policy", policy
      json.field "resident_map", resident_map
      json.field "exact_text", exact_text
      json.field "candidate_text", free_text
      json.field "exact_ids", exact_ids
      json.field "candidate_ids", free_ids
      json.field "exact_ended_with_eos", exact_ids[-1] == tokenizer.eos_id
      json.field "candidate_ended_with_eos", free_ids[-1] == tokenizer.eos_id
      json.field "free_common_prefix", common_prefix
      json.field "retire_order_top1_matches", total_top1_matches
      json.field "retire_order_top1_count", total_top1_count
      json.field "teacher_top2_ranked_matches", top2_ranked_matches
      json.field "teacher_top2_ranked_count", 2 * exact_top2s.size
      json.field "teacher_top2_set_overlap", top2_set_overlap
      json.field "teacher_top2_set_overlap_count", 2 * exact_top2s.size
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
      json.field "prefix_raw_bytes", raw_bytes
      json.field "prefix_payload_bytes", resident_bytes
      json.field "prefix_ratio", density
      json.field "full_attention_layers", hp.full_attention_layers.size
      json.field "resident_layers", hp.full_attention_layers.size
      json.field "resident_f32_owner_layers" do
        json.array do
          final_f32_owner_layers.each { |layer_index| json.number layer_index }
        end
      end
      json.field "resident_cache_consistent", resident_cache_consistent
      json.field "resident_cache_tokens", tokens.size + exact_top2s.size
      json.field "resident_capacity_tokens", max_seq
      json.field "resident_first_logit", resident_first_logit
      json.field "forced_first_logit", forced_first_logit
      json.field "resident_prefill_ms", resident_prefill_ms
      json.field "forced_prefill_ms", forced_prefill_ms
      json.field "free_decode_ms", free_decode_ms
      json.field "forced_decode_ms", forced_decode_ms
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
    end
  end
  puts "QBIT_QUALITY_JSON=#{payload}"
  GC.collect
end

gguf.close
