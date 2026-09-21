# Offline, CPU-only falsifier for re-encoding sampled Qwen3.8 Q4_K FFN
# gate/up rows as llama.cpp IQ3_S or IQ3_XXS blocks.
#
# Build with a current llama.cpp libggml-base, for example:
#   crystal build bin/qwen35_q4_iq3_weight_probe.cr -o /tmp/qwen35_q4_iq3_weight_probe \
#     --link-flags="/path/to/llama.cpp/build/bin/libggml-base.dylib -Wl,-rpath,/path/to/llama.cpp/build/bin"
#
# This probe never changes the GGUF file and does not implement a production
# resident format or Metal decoder. Its operator top-1/top-2 checks are only a
# cheap rejection gate before any GPU work is admissible.

require "option_parser"
require "../src/ml/gguf/reader"

lib LibGGMLBase
  fun iq3xs_init_impl(grid_size : Int32)
  fun iq3xs_free_impl(grid_size : Int32)
  fun quantize_row_iq3_xxs_ref(x : Float32*, y : UInt8*, k : Int64)
  fun quantize_row_iq3_s_ref(x : Float32*, y : UInt8*, k : Int64)
  fun dequantize_row_iq3_xxs(x : UInt8*, y : Float32*, k : Int64)
  fun dequantize_row_iq3_s(x : UInt8*, y : Float32*, k : Int64)
end

QK                       =             256
NATIVE_Q4_BYTES          =             144
IQ3_S_BYTES              =             110
IQ3_XXS_BYTES            =              98
DEFAULT_ROWS             =              64
DEFAULT_SEED             = 0x4f31_93a7_u64
REQUIRED_WHOLE_TOKEN_PCT =         3.0_f64
GATE_UP_CORRIDOR_PCT     =       30.44_f64
NUMERIC_COSINE_GATE      =     0.99999_f64

record Activation, name : String, values : Array(Float32)

def supported_q4_weight_name?(name : String) : Bool
  !/\Ablk\.\d+\.ffn_(gate|up)\.weight\z/.match(name).nil?
end

def sampled_rows(out_dim : Int32, requested : Int32) : Array(Int32)
  raise ArgumentError.new("rows must be positive") if requested <= 0
  count = Math.min(out_dim, requested)
  raise ArgumentError.new("tensor has no output rows") if count <= 0
  return [0_i32] if count == 1

  Array(Int32).new(count) do |index|
    ((index.to_i64 * (out_dim.to_i64 - 1_i64)) / (count.to_i64 - 1_i64)).to_i32
  end
end

def build_activations(in_dim : Int32, seed : UInt64) : Array(Activation)
  first_rng = Random.new(seed)
  second_rng = Random.new(seed ^ 0x9e37_79b9_u64)
  denominator = Math.max(in_dim - 1, 1).to_f64

  [
    Activation.new("random_uniform_a", Array(Float32).new(in_dim) { ((first_rng.next_float * 2.0) - 1.0).to_f32 }),
    Activation.new("random_uniform_b", Array(Float32).new(in_dim) { ((second_rng.next_float * 2.0) - 1.0).to_f32 }),
    Activation.new("ramp", Array(Float32).new(in_dim) { |i| ((i.to_f64 * 2.0 / denominator) - 1.0).to_f32 }),
    Activation.new("sinusoid", Array(Float32).new(in_dim) { |i| Math.sin((i.to_f64 + 1.0) * 0.017).to_f32 }),
    Activation.new("alternating", Array(Float32).new(in_dim) do |i|
      sign = i.even? ? 1.0 : -1.0
      (sign * (1.0 + (i % 17).to_f64 / 17.0)).to_f32
    end),
  ]
end

def parse_thresholds(value : String) : Array(Float64)
  thresholds = value.split(',').map do |part|
    parsed = part.strip.to_f64
    unless parsed.finite? && parsed >= 0.0
      raise ArgumentError.new("thresholds must be finite and non-negative")
    end
    parsed
  end
  raise ArgumentError.new("at least one threshold is required") if thresholds.empty?
  thresholds.uniq.sort
end

def block_std(values : Array(Float32)) : Float64
  mean = values.sum(0.0_f64) { |value| value.to_f64 } / values.size.to_f64
  variance = values.sum(0.0_f64) do |value|
    delta = value.to_f64 - mean
    delta * delta
  end / values.size.to_f64
  Math.sqrt(variance)
end

def max_residual_ratio(reference : Array(Float32), candidate : Array(Float32)) : Float64
  standard_deviation = block_std(reference)
  max_error = 0.0_f64
  reference.each_with_index do |value, index|
    error = (value.to_f64 - candidate[index].to_f64).abs
    max_error = error if error > max_error
  end
  return 0.0_f64 if standard_deviation == 0.0 && max_error == 0.0
  return Float64::INFINITY if standard_deviation == 0.0
  max_error / standard_deviation
end

def iq3_roundtrip(values : Array(Float32), kind : Symbol) : Array(Float32)
  bytes = case kind
          when :iq3_s   then IQ3_S_BYTES
          when :iq3_xxs then IQ3_XXS_BYTES
          else               raise ArgumentError.new("unsupported IQ3 kind #{kind}")
          end
  packed = Bytes.new(bytes, 0_u8)
  output = Array(Float32).new(QK, 0.0_f32)
  case kind
  when :iq3_s
    LibGGMLBase.quantize_row_iq3_s_ref(values.to_unsafe, packed.to_unsafe, QK.to_i64)
    LibGGMLBase.dequantize_row_iq3_s(packed.to_unsafe, output.to_unsafe, QK.to_i64)
  when :iq3_xxs
    LibGGMLBase.quantize_row_iq3_xxs_ref(values.to_unsafe, packed.to_unsafe, QK.to_i64)
    LibGGMLBase.dequantize_row_iq3_xxs(packed.to_unsafe, output.to_unsafe, QK.to_i64)
  end
  raise ArgumentError.new("IQ3 roundtrip produced non-finite values") unless output.all?(&.finite?)
  output
end

def dot_block(values : Array(Float32), activation : Array(Float32), offset : Int32) : Float64
  sum = 0.0_f64
  QK.times do |i|
    sum += values[i].to_f64 * activation[offset + i].to_f64
  end
  sum
end

def top_two(values : Array(Float64)) : Array(Int32)
  indices = Array(Int32).new(values.size) { |index| index.to_i32 }
  indices.sort! do |left, right|
    left_value = values[left]
    right_value = values[right]
    if left_value > right_value
      -1
    elsif left_value < right_value
      1
    else
      left <=> right
    end
  end
  values.size <= 2 ? indices : indices[0, 2]
end

def cosine(reference : Array(Float64), candidate : Array(Float64)) : Float64
  dot = 0.0_f64
  reference_norm = 0.0_f64
  candidate_norm = 0.0_f64
  reference.each_with_index do |value, index|
    other = candidate[index]
    dot += value * other
    reference_norm += value * value
    candidate_norm += other * other
  end
  if reference_norm == 0.0 || candidate_norm == 0.0
    return reference_norm == 0.0 && candidate_norm == 0.0 ? 1.0_f64 : 0.0_f64
  end
  dot / Math.sqrt(reference_norm * candidate_norm)
end

def print_metrics(policy : String,
                  activation : Activation,
                  sampled_rows : Array(Int32),
                  reference : Array(Float64),
                  candidate : Array(Float64)) : {Float64, Bool}
  similarity = cosine(reference, candidate)
  native_top2 = top_two(reference)
  candidate_top2 = top_two(candidate)
  ordered_top2 = native_top2 == candidate_top2
  puts "activation=#{activation.name} policy=#{policy} cosine=#{similarity.round(9)} " +
       "top1_identity=#{native_top2[0] == candidate_top2[0]} ordered_top2=#{ordered_top2} " +
       "native_top2_rows=#{native_top2.map { |i| sampled_rows[i] }.join(',')} " +
       "candidate_top2_rows=#{candidate_top2.map { |i| sampled_rows[i] }.join(',')}"
  {similarity, ordered_top2}
end

default_model_path = "#{ENV["HOME"]? || "."}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
model_path = ENV["QWEN35_MODEL"]? || default_model_path
requested_tensor = nil.as(String?)
requested_rows = DEFAULT_ROWS
thresholds = [0.40_f64, 0.50_f64, 0.60_f64, 0.65_f64, 0.70_f64, 0.80_f64, 1.00_f64, 1.20_f64]
seed = DEFAULT_SEED

begin
  OptionParser.parse do |parser|
    parser.banner = "Usage: qwen35_q4_iq3_weight_probe [options]"
    parser.on("--model=PATH", "GGUF file") { |value| model_path = value }
    parser.on("--tensor=NAME", "exact Q4_K FFN gate/up tensor") { |value| requested_tensor = value }
    parser.on("--rows=N", "evenly sampled output rows (default: #{DEFAULT_ROWS})") { |value| requested_rows = value.to_i }
    parser.on("--thresholds=LIST", "IQ3 max residual/std thresholds") { |value| thresholds = parse_thresholds(value) }
    parser.on("--seed=N", "deterministic activation seed") { |value| seed = value.to_u64 }
    parser.on("-h", "--help", "show this help") do
      puts parser
      exit 0
    end
  end

  raise ArgumentError.new("rows must be positive") if requested_rows <= 0
  gguf = nil.as(ML::GGUF::GGUFFile?)
  LibGGMLBase.iq3xs_init_impl(256)
  LibGGMLBase.iq3xs_init_impl(512)
  begin
    gguf = ML::GGUF::GGUFFile.new(model_path)
    tensor = if name = requested_tensor
               raise ArgumentError.new("tensor #{name} is not a supported Q4_K weight") unless supported_q4_weight_name?(name)
               gguf.not_nil!.tensor(name) || raise ArgumentError.new("tensor not found: #{name}")
             else
               gguf.not_nil!.tensors.find do |candidate|
                 supported_q4_weight_name?(candidate.name) && candidate.type == ML::GGUF::TensorType::Q4_K
               end || raise ArgumentError.new("no Q4_K blk.<layer>.ffn_gate/up.weight tensor found")
             end

    raise ArgumentError.new("selected tensor #{tensor.name} is not Q4_K") unless tensor.type == ML::GGUF::TensorType::Q4_K
    raise ArgumentError.new("selected tensor is not rank-2") unless tensor.dims.size == 2
    in_dim = tensor.dims[0].to_i64
    out_dim = tensor.dims[1].to_i64
    raise ArgumentError.new("input dimension is not divisible by #{QK}") unless in_dim > 0 && in_dim % QK == 0
    raise ArgumentError.new("invalid output dimension") unless out_dim > 0
    raise ArgumentError.new("dimensions exceed Int32") if in_dim > Int32::MAX || out_dim > Int32::MAX

    blocks_per_row = in_dim // QK
    expected_bytes = out_dim * blocks_per_row * NATIVE_Q4_BYTES
    unless tensor.data_bytes == expected_bytes
      raise ArgumentError.new("GGUF data_bytes=#{tensor.data_bytes} disagrees with Q4_K shape bytes=#{expected_bytes}")
    end
    raw = gguf.not_nil!.read_tensor_raw(tensor)
    raise ArgumentError.new("raw tensor size mismatch") unless raw.size.to_i64 == expected_bytes

    rows = sampled_rows(out_dim.to_i32, requested_rows)
    activations = build_activations(in_dim.to_i32, seed)
    total_blocks = rows.size.to_i64 * blocks_per_row
    native_bytes = total_blocks * NATIVE_Q4_BYTES
    bitmap_bytes = (total_blocks + 7_i64) // 8_i64

    native_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    iq3_s_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    iq3_xxs_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    adaptive_s_outputs = Array.new(thresholds.size) do
      Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    end
    adaptive_xxs_outputs = Array.new(thresholds.size) do
      Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    end
    adaptive_s_counts = Array(Int64).new(thresholds.size, 0_i64)
    adaptive_xxs_counts = Array(Int64).new(thresholds.size, 0_i64)
    s_ratio_sum = 0.0_f64
    s_ratio_max = 0.0_f64
    xxs_ratio_sum = 0.0_f64
    xxs_ratio_max = 0.0_f64

    rows.each_with_index do |row, sampled_index|
      row_native = Array(Float64).new(activations.size, 0.0_f64)
      row_s = Array(Float64).new(activations.size, 0.0_f64)
      row_xxs = Array(Float64).new(activations.size, 0.0_f64)
      row_adaptive_s = Array.new(thresholds.size) { Array(Float64).new(activations.size, 0.0_f64) }
      row_adaptive_xxs = Array.new(thresholds.size) { Array(Float64).new(activations.size, 0.0_f64) }

      blocks_per_row.times do |block|
        block_offset = (row.to_i64 * blocks_per_row * NATIVE_Q4_BYTES + block.to_i64 * NATIVE_Q4_BYTES).to_i
        native = ML::GGUF::Dequant.dequantize(raw[block_offset, NATIVE_Q4_BYTES], ML::GGUF::TensorType::Q4_K, QK)
        iq3_s = iq3_roundtrip(native, :iq3_s)
        iq3_xxs = iq3_roundtrip(native, :iq3_xxs)
        s_ratio = max_residual_ratio(native, iq3_s)
        xxs_ratio = max_residual_ratio(native, iq3_xxs)
        s_ratio_sum += s_ratio
        xxs_ratio_sum += xxs_ratio
        s_ratio_max = s_ratio if s_ratio > s_ratio_max
        xxs_ratio_max = xxs_ratio if xxs_ratio > xxs_ratio_max

        activations.each_with_index do |activation, activation_index|
          activation_offset = block.to_i32 * QK
          native_dot = dot_block(native, activation.values, activation_offset)
          s_dot = dot_block(iq3_s, activation.values, activation_offset)
          xxs_dot = dot_block(iq3_xxs, activation.values, activation_offset)
          row_native[activation_index] += native_dot
          row_s[activation_index] += s_dot
          row_xxs[activation_index] += xxs_dot
          thresholds.each_with_index do |threshold, threshold_index|
            row_adaptive_s[threshold_index][activation_index] += s_ratio <= threshold ? s_dot : native_dot
            row_adaptive_xxs[threshold_index][activation_index] += xxs_ratio <= threshold ? xxs_dot : native_dot
          end
        end

        thresholds.each_with_index do |threshold, threshold_index|
          adaptive_s_counts[threshold_index] += 1 if s_ratio <= threshold
          adaptive_xxs_counts[threshold_index] += 1 if xxs_ratio <= threshold
        end
      end

      activations.each_index do |activation_index|
        native_outputs[activation_index][sampled_index] = row_native[activation_index]
        iq3_s_outputs[activation_index][sampled_index] = row_s[activation_index]
        iq3_xxs_outputs[activation_index][sampled_index] = row_xxs[activation_index]
        thresholds.each_index do |threshold_index|
          adaptive_s_outputs[threshold_index][activation_index][sampled_index] = row_adaptive_s[threshold_index][activation_index]
          adaptive_xxs_outputs[threshold_index][activation_index][sampled_index] = row_adaptive_xxs[threshold_index][activation_index]
        end
      end
    end

    puts "probe=qwen35_q4_iq3_weight_probe mode=offline_cpu_only"
    puts "tensor=#{tensor.name} type=#{tensor.type} dims=#{tensor.dims.join('x')} sampled_rows=#{rows.size}/#{out_dim} blocks=#{total_blocks}"
    puts "scope=reencode_dequantized_q4_values; no_source_float_weights no_metal no_model_generation"
    puts "ranking_scope=sampled_output_rows_only; top1_top2_are_operator_proxies_not_token_or_ECS_metrics"
    puts "whole_token_model=recurrent_gate_up_logical_byte_share_#{GATE_UP_CORRIDOR_PCT}% required_ideal_saving_#{REQUIRED_WHOLE_TOKEN_PCT}%"
    puts "residual_ratio=max_abs(q4-iq3)/q4_block_std iq3_s_mean=#{(s_ratio_sum / total_blocks).round(9)} iq3_s_max=#{s_ratio_max.round(9)} iq3_xxs_mean=#{(xxs_ratio_sum / total_blocks).round(9)} iq3_xxs_max=#{xxs_ratio_max.round(9)}"

    policies = [
      {"all_iq3_s", IQ3_S_BYTES.to_i64 * total_blocks, total_blocks, iq3_s_outputs},
      {"all_iq3_xxs", IQ3_XXS_BYTES.to_i64 * total_blocks, total_blocks, iq3_xxs_outputs},
    ]
    thresholds.each_with_index do |threshold, index|
      s_count = adaptive_s_counts[index]
      xxs_count = adaptive_xxs_counts[index]
      policies << {"adaptive_iq3_s_t#{threshold}", bitmap_bytes + s_count * IQ3_S_BYTES + (total_blocks - s_count) * NATIVE_Q4_BYTES, s_count, adaptive_s_outputs[index]}
      policies << {"adaptive_iq3_xxs_t#{threshold}", bitmap_bytes + xxs_count * IQ3_XXS_BYTES + (total_blocks - xxs_count) * NATIVE_Q4_BYTES, xxs_count, adaptive_xxs_outputs[index]}
    end

    policies.each do |policy, forecast_bytes, compressed_blocks, outputs|
      compression = native_bytes.to_f64 / forecast_bytes.to_f64
      compressed_pct = 100.0 * compressed_blocks.to_f64 / total_blocks.to_f64
      ideal_saving = GATE_UP_CORRIDOR_PCT * (1.0 - forecast_bytes.to_f64 / native_bytes.to_f64)
      metric_rows = activations.each_with_index.map do |activation, activation_index|
        print_metrics(policy, activation, rows, native_outputs[activation_index], outputs[activation_index])
      end.to_a
      minimum_cosine = metric_rows.min_of(&.[0])
      top2_match_count = metric_rows.count { |metrics| metrics[1] }
      ordered_top2 = top2_match_count == metric_rows.size
      numeric_gate = minimum_cosine >= NUMERIC_COSINE_GATE && ordered_top2
      byte_gate = ideal_saving >= REQUIRED_WHOLE_TOKEN_PCT
      puts "policy_summary=#{policy} compressed_blocks=#{compressed_blocks}/#{total_blocks} compressed_pct=#{compressed_pct.round(6)} " +
           "forecast_bytes=#{forecast_bytes} native_bytes=#{native_bytes} " +
           "native_over_forecast=#{compression.round(9)} ideal_whole_token_saving_pct=#{ideal_saving.round(6)} " +
           "min_cosine=#{minimum_cosine.round(9)} ordered_top2_matches=#{top2_match_count}/#{metric_rows.size} " +
           "ordered_top2_all=#{ordered_top2} byte_gate=#{byte_gate} " +
           "numeric_gate=#{numeric_gate} metal_candidate=#{byte_gate && numeric_gate}"
    end
  ensure
    gguf.try(&.close)
    LibGGMLBase.iq3xs_free_impl(256)
    LibGGMLBase.iq3xs_free_impl(512)
  end
rescue ex
  STDERR.puts "error: #{ex.message}"
  exit 2
end
