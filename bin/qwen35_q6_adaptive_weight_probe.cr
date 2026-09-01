# Offline, CPU-only falsifier for sampled Qwen3.8 Q6_K weight rows.
#
# This probe forecasts several weight representations. It does not implement
# or exercise a production format: Gaussian records charge payload plus four
# metadata bytes, while subscale-P5 and exact sparse-bitplane Q6 report their
# own layouts. No model runner, Metal command buffer, or scheduling path is
# involved.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen_qbit_gaussian_codec"

QK                             =             256
NATIVE_Q6_BYTES                =             210
SUBSCALE_P5_BYTES              =             180
SPARSE_BITPLANE_BASE_BYTES     =             180
SPARSE_BITPLANE_MAX_EXCEPTIONS =              29
P4_PAYLOAD_BYTES               =             136
P5_PAYLOAD_BYTES               =             168
FORECAST_METADATA_BYTES        =               4
DEFAULT_ROWS                   =             256
DEFAULT_P4_THRESHOLD           =        0.20_f64
DEFAULT_P5_THRESHOLD           =        0.10_f64
DEFAULT_SUBSCALE_P5_THRESHOLD  =        0.11_f64
DEFAULT_SEED                   = 0x51f0_3a95_u64

record Activation, name : String, values : Array(Float32)

def ffn_down_name?(name : String) : Bool
  !/\Ablk\.\d+\.ffn_down\.weight\z/.match(name).nil?
end

def supported_q6_weight_name?(name : String) : Bool
  ffn_down_name?(name) || !/\Ablk\.\d+\.attn_qkv\.weight\z/.match(name).nil?
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

# Extract the unsigned six-bit codes in dequantized element order. Native Q6_K
# maps those codes to signed values by subtracting 32; scales and d remain
# byte-identical in the sparse-bitplane candidate.
def q6_codes(raw : Bytes) : Array(UInt8)
  raise ArgumentError.new("native Q6_K block is not #{NATIVE_Q6_BYTES} bytes") unless raw.size == NATIVE_Q6_BYTES

  codes = Array(UInt8).new(QK, 0_u8)
  output_base = 0
  2.times do |half|
    ql_base = half * 64
    qh_base = 128 + half * 32
    32.times do |lane|
      ql0 = raw[ql_base + lane]
      ql1 = raw[ql_base + lane + 32]
      qh = raw[qh_base + lane]
      codes[output_base + lane] = ((ql0 & 0x0f_u8) | (((qh >> 0) & 3_u8) << 4)).to_u8
      codes[output_base + lane + 32] = ((ql1 & 0x0f_u8) | (((qh >> 2) & 3_u8) << 4)).to_u8
      codes[output_base + lane + 64] = ((ql0 >> 4) | (((qh >> 4) & 3_u8) << 4)).to_u8
      codes[output_base + lane + 96] = ((ql1 >> 4) | (((qh >> 6) & 3_u8) << 4)).to_u8
    end
    output_base += 128
  end
  codes
end

def pack_q6_codes(codes : Array(UInt8)) : Bytes
  raise ArgumentError.new("Q6_K packing requires exactly #{QK} codes") unless codes.size == QK

  packed = Bytes.new(192, 0_u8)
  output_base = 0
  2.times do |half|
    ql_base = half * 64
    qh_base = 128 + half * 32
    32.times do |lane|
      q1 = codes[output_base + lane]
      q2 = codes[output_base + lane + 32]
      q3 = codes[output_base + lane + 64]
      q4 = codes[output_base + lane + 96]
      raise ArgumentError.new("Q6_K code exceeds six bits") if (q1 | q2 | q3 | q4) > 63_u8

      packed[ql_base + lane] = (q1 & 0x0f_u8) | ((q3 & 0x0f_u8) << 4)
      packed[ql_base + lane + 32] = (q2 & 0x0f_u8) | ((q4 & 0x0f_u8) << 4)
      packed[qh_base + lane] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
    end
    output_base += 128
  end
  packed
end

# Drop the sparsest of six bitplanes, retain the other five densely, and code
# deviations from the dropped plane's majority bit as sorted UInt8 indices.
# This simulates the fail-closed decoder and proves semantic Q6 code identity;
# the unchanged 18-byte scales+d suffix is charged in the fixed 180-byte base.
def analyze_sparse_bitplane_q6(raw : Bytes) : {Int32, Int32, UInt8}
  codes = q6_codes(raw)
  best_plane = 0
  best_default = 0_u8
  best_exceptions = QK + 1

  6.times do |plane|
    ones = codes.count { |code| ((code >> plane) & 1_u8) == 1_u8 }
    default_bit = ones <= QK - ones ? 0_u8 : 1_u8
    exceptions = Math.min(ones, QK - ones)
    if exceptions < best_exceptions
      best_plane = plane
      best_default = default_bit
      best_exceptions = exceptions
    end
  end

  plane_mask = (1_u8 << best_plane)
  reconstructed = codes.map do |code|
    (code & ~plane_mask) | (best_default << best_plane)
  end
  exception_indices = Array(UInt8).new(best_exceptions)
  codes.each_with_index do |code, index|
    bit = (code >> best_plane) & 1_u8
    exception_indices << index.to_u8 if bit != best_default
  end
  raise ArgumentError.new("sparse Q6 exception count mismatch") unless exception_indices.size == best_exceptions

  previous = -1
  exception_indices.each do |encoded_index|
    index = encoded_index.to_i
    raise ArgumentError.new("sparse Q6 exception index is not strictly increasing") unless index > previous && index < QK
    reconstructed[index] ^= plane_mask
    previous = index
  end
  raise ArgumentError.new("sparse Q6 reconstruction changed a quantized code") unless reconstructed == codes
  raise ArgumentError.new("sparse Q6 reconstruction changed packed value bytes") unless pack_q6_codes(reconstructed) == raw[0, 192]
  raise ArgumentError.new("sparse Q6 metadata suffix is truncated") unless raw[192, 18].size == 18

  {best_exceptions, best_plane, best_default}
end

def reconstruct_gaussian(values : Array(Float32), precision : Int32) : Array(Float32)
  encoded = ML::GGUF::QwenQBitGaussianCodec.encode(values, QK, precision)
  expected_payload = precision == 4 ? P4_PAYLOAD_BYTES : P5_PAYLOAD_BYTES
  unless encoded.payload.size == expected_payload
    raise ArgumentError.new("unexpected P#{precision} Gaussian payload size #{encoded.payload.size}; expected #{expected_payload}")
  end
  ML::GGUF::QwenQBitGaussianCodec.decode(encoded)
end

# Requantize one native Q6_K block to a fixed-width symmetric 5-bit format
# while retaining Q6_K's 16-value scale granularity. The candidate layout is:
# 256 signed 5-bit values (160 B), 16 unsigned scale bytes, and one F32 master
# scale (4 B). This is an offline reconstruction model, not a production ABI.
def reconstruct_subscale_p5(values : Array(Float32)) : Array(Float32)
  raise ArgumentError.new("subscale P5 requires exactly #{QK} values") unless values.size == QK

  ideal_scales = Array(Float32).new(QK // 16, 0.0_f32)
  ideal_scales.size.times do |group|
    base = group * 16
    minimum = values[base].to_f64
    maximum = minimum
    1.upto(15) do |offset|
      value = values[base + offset].to_f64
      minimum = value if value < minimum
      maximum = value if value > maximum
    end
    ideal_scales[group] = Math.max(maximum / 15.0_f64, -minimum / 16.0_f64).to_f32
  end

  max_scale = ideal_scales.max
  return Array(Float32).new(QK, 0.0_f32) if max_scale == 0.0_f32

  # The scale bytes are unsigned magnitudes; signed weights carry the sign.
  # F32 is intentional: real Q6 blocks can require a subnormal FP16 master.
  master = max_scale / 127.0_f32
  raise ArgumentError.new("subscale P5 master scale is not finite and positive") unless master.finite? && master > 0.0_f32

  reconstructed = Array(Float32).new(QK, 0.0_f32)
  ideal_scales.each_with_index do |ideal, group|
    scale_code = (ideal / master).round.to_i.clamp(1, 127)
    scale = master * scale_code.to_f32
    base = group * 16
    16.times do |offset|
      quant = (values[base + offset] / scale).round.to_i.clamp(-16, 15)
      reconstructed[base + offset] = scale * quant.to_f32
    end
  end
  reconstructed
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
subscale_p5_threshold = DEFAULT_SUBSCALE_P5_THRESHOLD
seed = DEFAULT_SEED

begin
  OptionParser.parse do |parser|
    parser.banner = "Usage: crystal run bin/qwen35_q6_adaptive_weight_probe.cr -- [options]"
    parser.on("--model PATH", "GGUF file (default: QWEN35_MODEL or the Qwen3.8 cache path)") { |value| model_path = value }
    parser.on("--tensor NAME", "exact Q6_K FFN-down or recurrent QKV tensor") { |value| requested_tensor = value }
    parser.on("--rows N", "evenly sampled output rows (default: #{DEFAULT_ROWS})") { |value| requested_rows = value.to_i }
    parser.on("--p4-threshold X", "P4 max residual/std threshold (default: #{DEFAULT_P4_THRESHOLD})") { |value| p4_threshold = parse_nonnegative_float(value, "--p4-threshold") }
    parser.on("--p5-threshold X", "P5 max residual/std threshold (default: #{DEFAULT_P5_THRESHOLD})") { |value| p5_threshold = parse_nonnegative_float(value, "--p5-threshold") }
    parser.on("--subscale-p5-threshold X", "subscale P5 max residual/std threshold (default: #{DEFAULT_SUBSCALE_P5_THRESHOLD})") { |value| subscale_p5_threshold = parse_nonnegative_float(value, "--subscale-p5-threshold") }
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
               raise ArgumentError.new("tensor #{name} is not a supported Q6_K weight") unless supported_q6_weight_name?(name)
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
    all_subscale_p5_bytes = total_blocks * SUBSCALE_P5_BYTES
    all_p4_bytes = total_blocks * (P4_PAYLOAD_BYTES + FORECAST_METADATA_BYTES)
    all_p5_bytes = total_blocks * (P5_PAYLOAD_BYTES + FORECAST_METADATA_BYTES)

    native_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    subscale_p5_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    adaptive_subscale_p5_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    p4_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    p5_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }
    adaptive_outputs = Array(Array(Float64)).new(activations.size) { Array(Float64).new(rows.size, 0.0_f64) }

    adaptive_p4 = 0_i64
    adaptive_p5 = 0_i64
    adaptive_q6 = 0_i64
    adaptive_bytes = 0_i64
    adaptive_subscale_p5 = 0_i64
    adaptive_subscale_q6 = 0_i64
    adaptive_subscale_bytes = (total_blocks + 7_i64) // 8_i64
    subscale_p5_ratio_sum = 0.0_f64
    subscale_p5_ratio_max = 0.0_f64
    p4_ratio_sum = 0.0_f64
    p5_ratio_sum = 0.0_f64
    p4_ratio_max = 0.0_f64
    p5_ratio_max = 0.0_f64
    zero_std_blocks = 0_i64
    sparse_bitplane_bitmap_bytes = (total_blocks + 7_i64) // 8_i64
    sparse_bitplane_bytes = sparse_bitplane_bitmap_bytes
    sparse_bitplane_compressed = 0_i64
    sparse_bitplane_escapes = 0_i64
    sparse_bitplane_histogram = Array(Int64).new(129, 0_i64)
    sparse_bitplane_plane_counts = Array(Int64).new(6, 0_i64)
    sparse_bitplane_default_one = 0_i64

    rows.each_with_index do |row, sampled_index|
      row_native_sums = Array(Float64).new(activations.size, 0.0_f64)
      row_subscale_p5_sums = Array(Float64).new(activations.size, 0.0_f64)
      row_adaptive_subscale_p5_sums = Array(Float64).new(activations.size, 0.0_f64)
      row_p4_sums = Array(Float64).new(activations.size, 0.0_f64)
      row_p5_sums = Array(Float64).new(activations.size, 0.0_f64)
      row_adaptive_sums = Array(Float64).new(activations.size, 0.0_f64)

      blocks_per_row.times do |block|
        block_offset = (row.to_i64 * blocks_per_row * NATIVE_Q6_BYTES + block.to_i64 * NATIVE_Q6_BYTES).to_i
        raw_block = raw[block_offset, NATIVE_Q6_BYTES]
        exceptions, sparse_plane, sparse_default = analyze_sparse_bitplane_q6(raw_block)
        sparse_bitplane_histogram[exceptions] += 1
        sparse_bitplane_plane_counts[sparse_plane] += 1
        sparse_bitplane_default_one += 1 if sparse_default == 1_u8
        if exceptions <= SPARSE_BITPLANE_MAX_EXCEPTIONS
          sparse_bitplane_compressed += 1
          sparse_bitplane_bytes += SPARSE_BITPLANE_BASE_BYTES + exceptions
        else
          sparse_bitplane_escapes += 1
          sparse_bitplane_bytes += NATIVE_Q6_BYTES
        end

        native_block = dequantize_q6_block(raw_block)
        subscale_p5_block = reconstruct_subscale_p5(native_block)
        p4_block = reconstruct_gaussian(native_block, 4)
        p5_block = reconstruct_gaussian(native_block, 5)
        standard_deviation = block_std(native_block)
        subscale_p5_ratio = max_residual_ratio(native_block, subscale_p5_block, standard_deviation)
        subscale_p5_ratio_sum += subscale_p5_ratio
        subscale_p5_ratio_max = subscale_p5_ratio if subscale_p5_ratio > subscale_p5_ratio_max
        adaptive_subscale_block = if subscale_p5_ratio <= subscale_p5_threshold
                                    adaptive_subscale_p5 += 1
                                    adaptive_subscale_bytes += SUBSCALE_P5_BYTES
                                    subscale_p5_block
                                  else
                                    adaptive_subscale_q6 += 1
                                    adaptive_subscale_bytes += NATIVE_Q6_BYTES
                                    native_block
                                  end
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
          subscale_p5_sum = 0.0_f64
          adaptive_subscale_p5_sum = 0.0_f64
          p4_sum = 0.0_f64
          p5_sum = 0.0_f64
          adaptive_sum = 0.0_f64
          value_index = 0
          while value_index < QK
            input = activation.values[activation_offset + value_index].to_f64
            native_value = native_block[value_index].to_f64
            subscale_p5_value = subscale_p5_block[value_index].to_f64
            adaptive_subscale_p5_value = adaptive_subscale_block[value_index].to_f64
            p4_value = p4_block[value_index].to_f64
            p5_value = p5_block[value_index].to_f64
            adaptive_value = case adaptive_kind
                             when :p4 then p4_value
                             when :p5 then p5_value
                             else          native_value
                             end
            native_sum += input * native_value
            subscale_p5_sum += input * subscale_p5_value
            adaptive_subscale_p5_sum += input * adaptive_subscale_p5_value
            p4_sum += input * p4_value
            p5_sum += input * p5_value
            adaptive_sum += input * adaptive_value
            value_index += 1
          end
          row_native_sums[activation_index] += native_sum
          row_subscale_p5_sums[activation_index] += subscale_p5_sum
          row_adaptive_subscale_p5_sums[activation_index] += adaptive_subscale_p5_sum
          row_p4_sums[activation_index] += p4_sum
          row_p5_sums[activation_index] += p5_sum
          row_adaptive_sums[activation_index] += adaptive_sum
        end
      end

      activations.each_index do |activation_index|
        native_outputs[activation_index][sampled_index] = row_native_sums[activation_index]
        subscale_p5_outputs[activation_index][sampled_index] = row_subscale_p5_sums[activation_index]
        adaptive_subscale_p5_outputs[activation_index][sampled_index] = row_adaptive_subscale_p5_sums[activation_index]
        p4_outputs[activation_index][sampled_index] = row_p4_sums[activation_index]
        p5_outputs[activation_index][sampled_index] = row_p5_sums[activation_index]
        adaptive_outputs[activation_index][sampled_index] = row_adaptive_sums[activation_index]
      end
    end

    puts "probe=qwen35_q6_adaptive_weight_probe mode=offline_cpu_only"
    puts "tensor=#{tensor.name} type=#{tensor.type} dims=#{tensor.dims.join("x")} sampled_rows=#{rows.size}/#{out_dim} first_row=#{rows.first} last_row=#{rows.last}"
    puts "thresholds=p4:#{p4_threshold} p5:#{p5_threshold} subscale_p5:#{subscale_p5_threshold} seed=#{seed} activations=#{activations.size}"
    puts "forecast_warning=Gaussian records include 4 metadata bytes per 256 values; subscale P5 uses its separately reported fixed/grouped layout; neither is a production ABI"
    puts "memory_model=GGUF raw mmap view plus bounded native and reconstructed 256-value blocks"
    puts "ranking_scope=sampled_output_rows_only; top1_top2_are_operator_proxies_not_token_or_ECS_metrics"
    puts "residual_ratio=max_abs(original-candidate)/original_block_std p4_mean=#{(p4_ratio_sum / total_blocks).round(9)} p4_max=#{p4_ratio_max.round(9)} p5_mean=#{(p5_ratio_sum / total_blocks).round(9)} p5_max=#{p5_ratio_max.round(9)} zero_std_blocks=#{zero_std_blocks}"
    puts "subscale_p5_residual_ratio=mean:#{(subscale_p5_ratio_sum / total_blocks).round(9)} max:#{subscale_p5_ratio_max.round(9)}"
    puts "native_baseline=raw_q6_payload_bytes=#{native_bytes} variable_record_q6_forecast_bytes=#{native_record_bytes}"
    puts "policy=all_subscale_p5 blocks=#{total_blocks} forecast_bytes=#{all_subscale_p5_bytes} native_bytes=#{native_bytes} native_over_forecast=#{(native_bytes.to_f64 / all_subscale_p5_bytes.to_f64).round(9)} layout=fixed_5bit_values_plus_16_scale_bytes_plus_f32_master"
    puts "policy=adaptive_subscale_p5 blocks=#{total_blocks} p5=#{adaptive_subscale_p5} q6=#{adaptive_subscale_q6} forecast_bytes=#{adaptive_subscale_bytes} native_bytes=#{native_bytes} native_over_forecast=#{(native_bytes.to_f64 / adaptive_subscale_bytes.to_f64).round(9)} layout=one_bit_tier_bitmap_plus_grouped_fixed_records"
    sparse_ratio = native_bytes.to_f64 / sparse_bitplane_bytes.to_f64
    sparse_escape_rate = sparse_bitplane_escapes.to_f64 / total_blocks.to_f64
    sparse_histogram_bands = [
      sparse_bitplane_histogram[0..15].sum,
      sparse_bitplane_histogram[16..29].sum,
      sparse_bitplane_histogram[30..63].sum,
      sparse_bitplane_histogram[64..95].sum,
      sparse_bitplane_histogram[96..128].sum,
    ]
    sparse_metal_gate = sparse_ratio >= 1.12 && sparse_escape_rate <= 0.10
    puts "policy=exact_q6_sparse_bitplane blocks=#{total_blocks} compressed=#{sparse_bitplane_compressed} q6_escapes=#{sparse_bitplane_escapes} escape_rate=#{sparse_escape_rate.round(9)} forecast_bytes=#{sparse_bitplane_bytes} native_bytes=#{native_bytes} native_over_forecast=#{sparse_ratio.round(9)} exact_reconstruction=true metal_gate=#{sparse_metal_gate} layout=one_bit_type_bitmap_plus_180_byte_base_plus_uint8_exceptions"
    puts "exact_q6_sparse_bitplane_histogram=k0_15:#{sparse_histogram_bands[0]} k16_29:#{sparse_histogram_bands[1]} k30_63:#{sparse_histogram_bands[2]} k64_95:#{sparse_histogram_bands[3]} k96_128:#{sparse_histogram_bands[4]} selected_planes=#{sparse_bitplane_plane_counts.join(",")} default_zero=#{total_blocks - sparse_bitplane_default_one} default_one=#{sparse_bitplane_default_one}"
    puts "policy=all_p4 blocks=#{total_blocks} p4=#{total_blocks} p5=0 q6=0 forecast_bytes=#{all_p4_bytes} native_bytes=#{native_bytes} forecast_over_native_payload=#{(all_p4_bytes.to_f64 / native_bytes.to_f64).round(9)} forecast_over_native_record=#{(all_p4_bytes.to_f64 / native_record_bytes.to_f64).round(9)} native_over_forecast=#{(native_bytes.to_f64 / all_p4_bytes.to_f64).round(9)}"
    puts "policy=all_p5 blocks=#{total_blocks} p4=0 p5=#{total_blocks} q6=0 forecast_bytes=#{all_p5_bytes} native_bytes=#{native_bytes} forecast_over_native_payload=#{(all_p5_bytes.to_f64 / native_bytes.to_f64).round(9)} forecast_over_native_record=#{(all_p5_bytes.to_f64 / native_record_bytes.to_f64).round(9)} native_over_forecast=#{(native_bytes.to_f64 / all_p5_bytes.to_f64).round(9)}"
    puts "policy=adaptive blocks=#{total_blocks} p4=#{adaptive_p4} p5=#{adaptive_p5} q6=#{adaptive_q6} forecast_bytes=#{adaptive_bytes} native_bytes=#{native_bytes} forecast_over_native_payload=#{(adaptive_bytes.to_f64 / native_bytes.to_f64).round(9)} forecast_over_native_record=#{(adaptive_bytes.to_f64 / native_record_bytes.to_f64).round(9)} native_over_forecast=#{(native_bytes.to_f64 / adaptive_bytes.to_f64).round(9)}"

    activations.each_with_index do |activation, activation_index|
      print_metrics("all_subscale_p5", activation, rows, native_outputs[activation_index], subscale_p5_outputs[activation_index])
      print_metrics("adaptive_subscale_p5", activation, rows, native_outputs[activation_index], adaptive_subscale_p5_outputs[activation_index])
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
