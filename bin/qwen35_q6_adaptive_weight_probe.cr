# Offline, CPU-only falsifier for a sampled Qwen3.8 Q6_K FFN-down weight row set.
#
# This probe forecasts a variable-record representation. It does not implement
# or exercise a production weight format: each selected 256-value block is
# charged payload bytes (P4=136, P5=168, native Q6=210) plus four forecast
# metadata bytes. No model runner, Metal command buffer, or scheduling path is
# involved.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"

QK                      =             256
NATIVE_Q6_BYTES         =             210
P4_PAYLOAD_BYTES        =             136
P5_PAYLOAD_BYTES        =             168
FORECAST_METADATA_BYTES =               4
DEFAULT_ROWS            =             256
DEFAULT_P4_THRESHOLD    =        0.20_f64
DEFAULT_P5_THRESHOLD    =        0.10_f64
DEFAULT_SEED            = 0x51f0_3a95_u64

record Activation, name : String, values : Array(Float32)

def ffn_down_name?(name : String) : Bool
  !/\Ablk\.\d+\.ffn_down\.weight\z/.match(name).nil?
end

def parse_nonnegative_float(value : String, option : String) : Float64
  parsed = value.to_f64
  raise ArgumentError.new("#{option} must be finite and non-negative") unless parsed.finite? && parsed >= 0.0
  parsed
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
  random_a = Array(Float32).new(in_dim) { ((first_rng.next_float * 2.0) - 1.0).to_f32 }
  random_b = Array(Float32).new(in_dim) { ((second_rng.next_float * 2.0) - 1.0).to_f32 }

  denominator = Math.max(in_dim - 1, 1).to_f64
  ramp = Array(Float32).new(in_dim) do |index|
    ((index.to_f64 * 2.0 / denominator) - 1.0).to_f32
  end

  sinusoid = Array(Float32).new(in_dim) do |index|
    Math.sin((index.to_f64 + 1.0) * 0.017).to_f32
  end

  alternating = Array(Float32).new(in_dim) do |index|
    sign = index.even? ? 1.0 : -1.0
    (sign * (1.0 + (index % 17).to_f64 / 17.0)).to_f32
  end

  [
    Activation.new("random_uniform_a", random_a),
    Activation.new("random_uniform_b", random_b),
    Activation.new("ramp", ramp),
    Activation.new("sinusoid", sinusoid),
    Activation.new("alternating", alternating),
  ]
end

def dequantize_q6_block(raw : Bytes) : Array(Float32)
  raise ArgumentError.new("native Q6_K block is not #{NATIVE_Q6_BYTES} bytes") unless raw.size == NATIVE_Q6_BYTES
  values = ML::GGUF::Dequant.dequantize(raw, ML::GGUF::TensorType::Q6_K, QK)
  raise ArgumentError.new("native Q6_K block dequantized to non-finite values") unless values.all? { |value| value.finite? }
  values
end

def reconstruct_gaussian(values : Array(Float32), precision : Int32) : Array(Float32)
  encoded = ML::GGUF::QwenQBitGaussianCodec.encode(values, QK, precision)
  expected_payload = precision == 4 ? P4_PAYLOAD_BYTES : P5_PAYLOAD_BYTES
  unless encoded.payload.size == expected_payload
    raise ArgumentError.new("unexpected P#{precision} Gaussian payload size #{encoded.payload.size}; expected #{expected_payload}")
  end
  ML::GGUF::QwenQBitGaussianCodec.decode(encoded)
end

def block_std(values : Array(Float32)) : Float64
  mean = values.sum(0.0_f64) { |value| value.to_f64 } / values.size.to_f64
  variance = values.sum(0.0_f64) do |value|
    delta = value.to_f64 - mean
    delta * delta
  end / values.size.to_f64
  Math.sqrt(variance)
end

def max_residual_ratio(original : Array(Float32), candidate : Array(Float32), standard_deviation : Float64) : Float64
  max_error = 0.0_f64
  original.each_with_index do |value, index|
    error = (value.to_f64 - candidate[index].to_f64).abs
    max_error = error if error > max_error
  end
  return 0.0_f64 if standard_deviation == 0.0 && max_error == 0.0
  return Float64::INFINITY if standard_deviation == 0.0
  max_error / standard_deviation
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

def print_metrics(policy : String, activation : Activation, sampled_rows : Array(Int32), reference : Array(Float64), candidate : Array(Float64))
  max_abs = 0.0_f64
  sum_squared = 0.0_f64
  reference_sum_squared = 0.0_f64
  max_relative = 0.0_f64
  reference.each_with_index do |value, index|
    difference = (candidate[index] - value).abs
    max_abs = difference if difference > max_abs
    sum_squared += difference * difference
    reference_sum_squared += value * value
    relative = difference / Math.max(value.abs, 1.0e-12_f64)
    max_relative = relative if relative > max_relative
  end

  sample_count = reference.size.to_f64
  nrmse = if reference_sum_squared == 0.0
            sum_squared == 0.0 ? 0.0_f64 : Float64::INFINITY
          else
            Math.sqrt(sum_squared / sample_count) / Math.sqrt(reference_sum_squared / sample_count)
          end
  native_top2 = top_two(reference)
  candidate_top2 = top_two(candidate)
  ordered_top2 = native_top2 == candidate_top2
  set_top2 = native_top2.sort == candidate_top2.sort
  native_top1 = native_top2[0]
  candidate_top1 = candidate_top2[0]
  native_top2_rows = native_top2.map { |index| sampled_rows[index] }
  candidate_top2_rows = candidate_top2.map { |index| sampled_rows[index] }

  puts "activation=#{activation.name} policy=#{policy} cosine=#{cosine(reference, candidate).round(9)} " +
       "max_abs=#{max_abs.round(9)} max_relative=#{max_relative.round(9)} nrmse=#{nrmse.round(9)} " +
       "top1_identity=#{native_top1 == candidate_top1} ordered_top2=#{ordered_top2} top2_set=#{set_top2} " +
       "native_top1_row=#{sampled_rows[native_top1]} candidate_top1_row=#{sampled_rows[candidate_top1]} " +
       "native_top2_rows=#{native_top2_rows.join(",")} candidate_top2_rows=#{candidate_top2_rows.join(",")}"
end

default_model_path = "#{ENV["HOME"]? || "."}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
model_path = ENV["QWEN35_MODEL"]? || default_model_path
requested_tensor = nil.as(String?)
requested_rows = DEFAULT_ROWS
p4_threshold = DEFAULT_P4_THRESHOLD
p5_threshold = DEFAULT_P5_THRESHOLD
seed = DEFAULT_SEED

begin
  OptionParser.parse do |parser|
    parser.banner = "Usage: crystal run bin/qwen35_q6_adaptive_weight_probe.cr -- [options]"
    parser.on("--model PATH", "GGUF file (default: QWEN35_MODEL or the Qwen3.8 cache path)") { |value| model_path = value }
    parser.on("--tensor NAME", "exact blk.<layer>.ffn_down.weight tensor") { |value| requested_tensor = value }
    parser.on("--rows N", "evenly sampled output rows (default: #{DEFAULT_ROWS})") { |value| requested_rows = value.to_i }
    parser.on("--p4-threshold X", "P4 max residual/std threshold (default: #{DEFAULT_P4_THRESHOLD})") { |value| p4_threshold = parse_nonnegative_float(value, "--p4-threshold") }
    parser.on("--p5-threshold X", "P5 max residual/std threshold (default: #{DEFAULT_P5_THRESHOLD})") { |value| p5_threshold = parse_nonnegative_float(value, "--p5-threshold") }
    parser.on("--seed N", "deterministic activation seed (default: #{DEFAULT_SEED})") { |value| seed = value.to_u64 }
    parser.on("-h", "--help", "show this help") do
      puts parser
      exit 0
    end
  end

  raise ArgumentError.new("rows must be positive") if requested_rows <= 0

  gguf = nil.as(ML::GGUF::GGUFFile?)
  begin
    gguf = ML::GGUF::GGUFFile.new(model_path)
    tensor = if name = requested_tensor
               raise ArgumentError.new("tensor #{name} is not an FFN-down tensor") unless ffn_down_name?(name)
               gguf.not_nil!.tensor(name) || raise ArgumentError.new("tensor not found: #{name}")
             else
               gguf.not_nil!.tensors.find do |candidate|
                 ffn_down_name?(candidate.name) && candidate.type == ML::GGUF::TensorType::Q6_K
               end || raise ArgumentError.new("no Q6_K blk.<layer>.ffn_down.weight tensor found")
             end

    raise ArgumentError.new("selected tensor #{tensor.name} is not Q6_K") unless tensor.type == ML::GGUF::TensorType::Q6_K
    raise ArgumentError.new("selected tensor #{tensor.name} is not rank-2") unless tensor.dims.size == 2
    raise ArgumentError.new("selected tensor #{tensor.name} has invalid dimensions #{tensor.dims}") if tensor.dims.any? { |dimension| dimension <= 0 }

    in_dim = tensor.dims[0].to_i64
    out_dim = tensor.dims[1].to_i64
    raise ArgumentError.new("input dimension #{in_dim} is not divisible by #{QK}") unless in_dim % QK == 0
    raise ArgumentError.new("input or output dimension is too large") if in_dim > Int32::MAX || out_dim > Int32::MAX
    blocks_per_row = in_dim // QK
    expected_bytes = out_dim * blocks_per_row * NATIVE_Q6_BYTES
    raise ArgumentError.new("tensor byte count overflow") if expected_bytes < 0
    raise ArgumentError.new("GGUF data_bytes=#{tensor.data_bytes} disagrees with Q6_K shape bytes=#{expected_bytes}") unless tensor.data_bytes == expected_bytes
    raise ArgumentError.new("tensor offset exceeds signed address range") if tensor.offset > Int64::MAX.to_u64
    file_size = File.size(model_path).to_i64
    tensor_end = gguf.not_nil!.data_offset + tensor.offset.to_i64 + expected_bytes
    raise ArgumentError.new("tensor data range exceeds GGUF file size") unless gguf.not_nil!.data_offset >= 0 && tensor_end >= gguf.not_nil!.data_offset && tensor_end <= file_size

    raw = gguf.not_nil!.read_tensor_raw(tensor)
    raise ArgumentError.new("raw tensor bytes=#{raw.size} disagrees with Q6_K shape bytes=#{expected_bytes}") unless raw.size.to_i64 == expected_bytes

    rows = sampled_rows(out_dim.to_i32, requested_rows)
    activations = build_activations(in_dim.to_i32, seed)
    total_blocks = rows.size.to_i64 * blocks_per_row
    native_bytes = total_blocks * NATIVE_Q6_BYTES
    native_record_bytes = total_blocks * (NATIVE_Q6_BYTES + FORECAST_METADATA_BYTES)
    all_p4_bytes = total_blocks * (P4_PAYLOAD_BYTES + FORECAST_METADATA_BYTES)
    all_p5_bytes = total_blocks * (P5_PAYLOAD_BYTES + FORECAST_METADATA_BYTES)

    native_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    p4_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    p5_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    adaptive_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }

    adaptive_p4 = 0_i64
    adaptive_p5 = 0_i64
    adaptive_q6 = 0_i64
    adaptive_bytes = 0_i64
    p4_ratio_sum = 0.0_f64
    p5_ratio_sum = 0.0_f64
    p4_ratio_max = 0.0_f64
    p5_ratio_max = 0.0_f64
    zero_std_blocks = 0_i64

    rows.each_with_index do |row, sampled_index|
      row_native_sums = Array(Float64).new(activations.size, 0.0_f64)
      row_p4_sums = Array(Float64).new(activations.size, 0.0_f64)
      row_p5_sums = Array(Float64).new(activations.size, 0.0_f64)
      row_adaptive_sums = Array(Float64).new(activations.size, 0.0_f64)

      blocks_per_row.times do |block|
        block_offset = (row.to_i64 * blocks_per_row * NATIVE_Q6_BYTES + block.to_i64 * NATIVE_Q6_BYTES).to_i
        native_block = dequantize_q6_block(raw[block_offset, NATIVE_Q6_BYTES])
        p4_block = reconstruct_gaussian(native_block, 4)
        p5_block = reconstruct_gaussian(native_block, 5)
        standard_deviation = block_std(native_block)
        zero_std_blocks += 1 if standard_deviation == 0.0
        p4_ratio = max_residual_ratio(native_block, p4_block, standard_deviation)
        p5_ratio = max_residual_ratio(native_block, p5_block, standard_deviation)
        p4_ratio_sum += p4_ratio
        p5_ratio_sum += p5_ratio
        p4_ratio_max = p4_ratio if p4_ratio > p4_ratio_max
        p5_ratio_max = p5_ratio if p5_ratio > p5_ratio_max

        adaptive_kind = if p4_ratio <= p4_threshold
                          adaptive_p4 += 1
                          adaptive_bytes += P4_PAYLOAD_BYTES + FORECAST_METADATA_BYTES
                          :p4
                        elsif p5_ratio <= p5_threshold
                          adaptive_p5 += 1
                          adaptive_bytes += P5_PAYLOAD_BYTES + FORECAST_METADATA_BYTES
                          :p5
                        else
                          adaptive_q6 += 1
                          adaptive_bytes += NATIVE_Q6_BYTES + FORECAST_METADATA_BYTES
                          :q6
                        end

        activations.each_with_index do |activation, activation_index|
          activation_offset = block * QK
          native_sum = 0.0_f64
          p4_sum = 0.0_f64
          p5_sum = 0.0_f64
          adaptive_sum = 0.0_f64
          value_index = 0
          while value_index < QK
            input = activation.values[activation_offset + value_index].to_f64
            native_value = native_block[value_index].to_f64
            p4_value = p4_block[value_index].to_f64
            p5_value = p5_block[value_index].to_f64
            adaptive_value = case adaptive_kind
                             when :p4 then p4_value
                             when :p5 then p5_value
                             else          native_value
                             end
            native_sum += input * native_value
            p4_sum += input * p4_value
            p5_sum += input * p5_value
            adaptive_sum += input * adaptive_value
            value_index += 1
          end
          row_native_sums[activation_index] += native_sum
          row_p4_sums[activation_index] += p4_sum
          row_p5_sums[activation_index] += p5_sum
          row_adaptive_sums[activation_index] += adaptive_sum
        end
      end

      activations.each_index do |activation_index|
        native_outputs[activation_index][sampled_index] = row_native_sums[activation_index]
        p4_outputs[activation_index][sampled_index] = row_p4_sums[activation_index]
        p5_outputs[activation_index][sampled_index] = row_p5_sums[activation_index]
        adaptive_outputs[activation_index][sampled_index] = row_adaptive_sums[activation_index]
      end
    end

    puts "probe=qwen35_q6_adaptive_weight_probe mode=offline_cpu_only"
    puts "tensor=#{tensor.name} type=#{tensor.type} dims=#{tensor.dims.join("x")} sampled_rows=#{rows.size}/#{out_dim} first_row=#{rows.first} last_row=#{rows.last}"
    puts "thresholds=p4:#{p4_threshold} p5:#{p5_threshold} seed=#{seed} activations=#{activations.size}"
    puts "forecast_warning=payload_plus_4_metadata_per_256_value_block; this is a forecast, not an implemented production layout"
    puts "memory_model=GGUF raw mmap view plus one native Q6_K block and two Gaussian reconstructions at a time"
    puts "ranking_scope=sampled_output_rows_only; top1_top2_are_operator_proxies_not_token_or_ECS_metrics"
    puts "residual_ratio=max_abs(original-candidate)/original_block_std p4_mean=#{(p4_ratio_sum / total_blocks).round(9)} p4_max=#{p4_ratio_max.round(9)} p5_mean=#{(p5_ratio_sum / total_blocks).round(9)} p5_max=#{p5_ratio_max.round(9)} zero_std_blocks=#{zero_std_blocks}"
    puts "native_baseline=raw_q6_payload_bytes=#{native_bytes} variable_record_q6_forecast_bytes=#{native_record_bytes}"
    puts "policy=all_p4 blocks=#{total_blocks} p4=#{total_blocks} p5=0 q6=0 forecast_bytes=#{all_p4_bytes} native_bytes=#{native_bytes} forecast_over_native_payload=#{(all_p4_bytes.to_f64 / native_bytes.to_f64).round(9)} forecast_over_native_record=#{(all_p4_bytes.to_f64 / native_record_bytes.to_f64).round(9)} native_over_forecast=#{(native_bytes.to_f64 / all_p4_bytes.to_f64).round(9)}"
    puts "policy=all_p5 blocks=#{total_blocks} p4=0 p5=#{total_blocks} q6=0 forecast_bytes=#{all_p5_bytes} native_bytes=#{native_bytes} forecast_over_native_payload=#{(all_p5_bytes.to_f64 / native_bytes.to_f64).round(9)} forecast_over_native_record=#{(all_p5_bytes.to_f64 / native_record_bytes.to_f64).round(9)} native_over_forecast=#{(native_bytes.to_f64 / all_p5_bytes.to_f64).round(9)}"
    puts "policy=adaptive blocks=#{total_blocks} p4=#{adaptive_p4} p5=#{adaptive_p5} q6=#{adaptive_q6} forecast_bytes=#{adaptive_bytes} native_bytes=#{native_bytes} forecast_over_native_payload=#{(adaptive_bytes.to_f64 / native_bytes.to_f64).round(9)} forecast_over_native_record=#{(adaptive_bytes.to_f64 / native_record_bytes.to_f64).round(9)} native_over_forecast=#{(native_bytes.to_f64 / adaptive_bytes.to_f64).round(9)}"

    activations.each_with_index do |activation, activation_index|
      print_metrics("all_p4", activation, rows, native_outputs[activation_index], p4_outputs[activation_index])
      print_metrics("all_p5", activation, rows, native_outputs[activation_index], p5_outputs[activation_index])
      print_metrics("adaptive", activation, rows, native_outputs[activation_index], adaptive_outputs[activation_index])
    end
  ensure
    gguf.try do |file|
      file.close
    end
  end
rescue ex
  STDERR.puts "error: #{ex.message}"
  exit 2
end
