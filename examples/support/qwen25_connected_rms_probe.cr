require "./qwen25_rms_statistics_probe"

# Bounded preparation and post-drain diagnostics only. No CPU neural path.
module Qwen25ConnectedRMSProbe
  extend self

  alias Output = NamedTuple(arm: String, suffix: String, dtype: String,
    nbytes: Int64, row_width: Int64)
  alias Gain = NamedTuple(path: String, raw: Bytes, tensor_sha256: String,
    header_sha256: String, shard_nbytes: Int64)

  def output_layout(tokens : Int32, dim : Int32) : Array(Output)
    unless 0 < tokens <= 1280 && 0 < dim <= 3584
      raise ArgumentError.new("connected RMS geometry outside bounded range")
    end
    outputs = [] of Output
    {"legacy_trace", "fed_trace", "fed_production"}.each do |arm|
      suffixes = arm == "fed_production" ? %w(rsqrt_f32 normalized_bf16 gained_output_bf16) : %w(variance_plus_eps_f32 rsqrt_f32 normalized_f32 normalized_bf16 gained_output_bf16)
      suffixes.each do |suffix|
        scalar = suffix == "variance_plus_eps_f32" || suffix == "rsqrt_f32"
        f32 = suffix.ends_with?("_f32")
        outputs << {arm: arm, suffix: suffix, dtype: f32 ? "float32" : "bfloat16",
                    nbytes: tokens.to_i64 * (scalar ? 1_i64 : dim.to_i64) * (f32 ? 4 : 2),
                    row_width: scalar ? 1_i64 : dim.to_i64}
      end
    end
    outputs
  end

  def required_named_bytes(tokens : Int32, dim : Int32) : Int64
    outputs = output_layout(tokens, dim)
    tokens.to_i64 * dim * 2 + dim.to_i64 * 2 + 4 + outputs.sum { |output| output[:nbytes] }
  end

  def admit_named_bytes!(tokens : Int32, dim : Int32, budget : Int64) : Int64
    required = required_named_bytes(tokens, dim)
    unless 0 < budget <= Qwen25RMSStatisticsProbe::MAX_NAMED_BYTES && required <= budget
      raise ArgumentError.new("connected RMS named-buffer budget rejected: requires #{required} bytes, cap #{budget}")
    end
    required
  end

  # Read just the selected vector, without mapping or uploading the full shard.
  # Its digest is authenticated by both pinned source/reference manifests.
  def load_gain(model_dir : String, metadata : JSON::Any, dim : Int32) : Gain
    unless 0 < dim <= 3584 && metadata["key"].as_s == Qwen25RMSStatisticsProbe::NORM_WEIGHT_KEY &&
           metadata["dtype"].as_s == "BF16" && metadata["shape"].as_a.map(&.as_i) == [dim] &&
           metadata["nbytes"].as_i64 == dim.to_i64 * 2
      raise ArgumentError.new("connected RMS gain metadata mismatch")
    end
    expected_sha = Qwen25RMSStatisticsProbe.checked_sha256(metadata["tensor_sha256"].as_s, "gain tensor")
    path = Qwen25RMSStatisticsProbe.contained_sibling(File.realpath(model_dir), metadata["shard"].as_s)
    File.open(path, "rb") do |io|
      shard_size = io.size
      raise ArgumentError.new("gain shard lacks header") unless shard_size >= 8
      length = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
      unless 0 < length <= 16_u64 * 1024 * 1024 && length <= (shard_size - 8).to_u64
        raise ArgumentError.new("gain shard header outside bounded range")
      end
      header = Bytes.new(length.to_i)
      io.read_fully(header)
      record = JSON.parse(String.new(header))[metadata["key"].as_s]
      unless record["dtype"].as_s == "BF16" && record["shape"].as_a.map(&.as_i) == [dim]
        raise ArgumentError.new("gain shard tensor dtype/shape mismatch")
      end
      offsets = record["data_offsets"].as_a.map(&.as_i64)
      unless offsets.size == 2 && 0 <= offsets[0] <= offsets[1] &&
             offsets[1] - offsets[0] == dim.to_i64 * 2 &&
             offsets[1] <= shard_size - 8 - length.to_i64
        raise ArgumentError.new("gain tensor extent outside shard bounds")
      end
      io.seek(8 + length.to_i64 + offsets[0])
      raw = Bytes.new(dim * 2)
      io.read_fully(raw)
      digest = Digest::SHA256.hexdigest(raw)
      raise ArgumentError.new("gain tensor digest mismatch") unless digest == expected_sha && io.size == shard_size
      return {path: path, raw: raw, tensor_sha256: digest,
              header_sha256: Digest::SHA256.hexdigest(header), shard_nbytes: shard_size}
    end
  end

  def compare_bf16(name : String, reference : Bytes, actual : Bytes,
                   row_width : Int64) : Qwen25RMSStatisticsProbe::FloatComparison
    unless reference.size == actual.size && reference.size > 0 && reference.size.even? && row_width > 0
      raise ArgumentError.new("BF16 comparison requires equal nonempty words and positive row width")
    end
    count = (reference.size // 2).to_i64
    raw_unequal = 0_i64
    failures = 0_i64
    maximum = 0.0
    raw_samples = [] of Qwen25RMSStatisticsProbe::RawSample
    failure_samples = [] of Qwen25RMSStatisticsProbe::RawSample
    count.times do |index|
      offset = index.to_i * 2
      reference_word = reference[offset].to_u32 | (reference[offset + 1].to_u32 << 8)
      actual_word = actual[offset].to_u32 | (actual[offset + 1].to_u32 << 8)
      expected = (reference_word << 16).unsafe_as(Float32).to_f64
      value = (actual_word << 16).unsafe_as(Float32).to_f64
      unequal = reference_word != actual_word
      raw_unequal += 1 if unequal
      finite = expected.finite? && value.finite?
      delta = (value - expected).abs
      maximum = finite ? Math.max(maximum, delta) : Float64::MAX
      failed = !finite || delta > Qwen25RMSStatisticsProbe::ATOL + Qwen25RMSStatisticsProbe::RTOL * expected.abs
      failures += 1 if failed
      if (unequal && raw_samples.size < 8) || (failed && failure_samples.size < 8)
        sample = {index: index, row: index // row_width,
                  actual: value.to_s, reference: expected.to_s,
                  actual_bits: actual_word.to_s(16).rjust(4, '0'),
                  reference_bits: reference_word.to_s(16).rjust(4, '0')}
        raw_samples << sample if unequal && raw_samples.size < 8
        failure_samples << sample if failed && failure_samples.size < 8
      end
    end
    {name: name, elements: count, finite_failures: failures,
     raw_unequal: raw_unequal, max_abs: maximum,
     first_raw_unequal: raw_samples, first_finite_failures: failure_samples,
     actual_sha256: Digest::SHA256.hexdigest(actual),
     reference_sha256: Digest::SHA256.hexdigest(reference)}
  end

  def compare(output : Output, expected : Bytes, actual : Bytes) : Qwen25RMSStatisticsProbe::FloatComparison
    raise ArgumentError.new("connected RMS output length mismatch") unless expected.size == output[:nbytes] && actual.size == output[:nbytes]
    name = "#{output[:arm]}.#{output[:suffix]}"
    if output[:dtype] == "float32"
      Qwen25RMSStatisticsProbe.compare_f32(name, expected, actual, output[:row_width])
    else
      compare_bf16(name, expected, actual, output[:row_width])
    end
  end
end
