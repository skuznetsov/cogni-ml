require "json"
require "./support/hy_nomas_cache_qbit"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasCacheQBitProbe
  extend self

  ROW_VALUES   = HyNomasCacheQBit::ROW_VALUES
  ROWS         = 8
  COUNT        = ROW_VALUES * ROWS
  HEADER_DELTA = 0.5_f32

  def inputs : Array(Float32)
    values = Array(Float32).new(COUNT, 0.0_f32)

    # Constant row: the zero-sigma p4 representation must be lossless.
    ROW_VALUES.times { |i| values[i] = 0.25_f32 }

    # Nonconstant p4 row. The probe flips this row's first stored plane bit
    # after packing to prove that the GPU decoder consumes the bitplane bytes.
    ROW_VALUES.times do |i|
      values[ROW_VALUES + i] = i.even? ? 1.0_f32 : -1.0_f32
    end

    # Fixed CLT samples selected to land safely inside p5's experimental
    # quality band while p4 fails its max-error bound.
    state = 8042_u32
    ROW_VALUES.times do |i|
      total = 0.0_f32
      12.times do
        state = ((state.to_u64 &* 1_664_525_u64 &+ 1_013_904_223_u64) & 0xffff_ffff_u64).to_u32
        unit = ((state >> 8) & 0x00ff_ffff_u32).to_f32 / 16_777_216.0_f32
        total += unit - 0.5_f32
      end
      values[2 * ROW_VALUES + i] = total
    end

    # Eight equal high values create normalized tail values close to the p8
    # centroid while p5 exceeds its max-error bound.
    8.times { |i| values[3 * ROW_VALUES + i] = 1.0_f32 }

    # A finite outlier deliberately exceeds the p8 error gate and must escape
    # as exact F32 without being counted as an invalid row.
    values[4 * ROW_VALUES] = 20.0_f32

    # Nonfinite values are exact-escaped and reported invalid for the export
    # gate; their original F32 bit patterns must survive the roundtrip.
    values[5 * ROW_VALUES + 17] = Float32::INFINITY
    values[5 * ROW_VALUES + 18] = Float32::NAN

    # This non-binary constant catches mean drift from repeated F32 summation.
    ROW_VALUES.times { |i| values[6 * ROW_VALUES + i] = 0.1_f32 }
    # Bitwise constant detection and direct zero-sigma decode must preserve -0.
    ROW_VALUES.times { |i| values[7 * ROW_VALUES + i] = -0.0_f32 }
    values
  end

  def raw_code(normalized : Float32, levels : Array(Float32)) : UInt8
    magnitude = normalized.abs
    low = 0
    high = levels.size - 1
    while low < high
      mid = (low + high) // 2
      boundary = (levels[mid] + levels[mid + 1]) * 0.5_f32
      if magnitude <= boundary
        high = mid
      else
        low = mid + 1
      end
    end
    normalized < 0.0_f32 ? (255 - low).to_u8 : low.to_u8
  end

  def decode_reference(header : HyNomasCacheQBit::RowHeader,
                       diagnostics : HyNomasCacheQBit::PackedDiagnostics,
                       row : Int32,
                       within : Int32) : Float32
    return exact_f32(diagnostics, header, within) if header.tier == 32_u32

    raw = 0_u8
    header.tier.to_i32.times do |plane|
      bit_index = 127 - (within ^ 7)
      byte_index = bit_index // 8
      bit_mask = 1_u8 << (bit_index % 8)
      byte_offset = header.payload_offset.to_i + plane * 16 + byte_index
      if (diagnostics.payload[byte_offset] & bit_mask) != 0
        raw |= 1_u8 << (7 - plane)
      end
    end
    centroid = ML::GGUF::QwenQBitGaussianCodec.reconstruct_raw_code(raw, header.tier.to_i32)
    header.mean + header.sigma * centroid
  end

  def exact_f32(diagnostics : HyNomasCacheQBit::PackedDiagnostics,
                header : HyNomasCacheQBit::RowHeader,
                within : Int32) : Float32
    offset = header.payload_offset.to_i + within * sizeof(Float32)
    bits = diagnostics.payload[offset].to_u32 |
           (diagnostics.payload[offset + 1].to_u32 << 8) |
           (diagnostics.payload[offset + 2].to_u32 << 16) |
           (diagnostics.payload[offset + 3].to_u32 << 24)
    bits.unsafe_as(Float32)
  end

  def policy_errors(values : Array(Float32), offset : Int32,
                    header : HyNomasCacheQBit::RowHeader,
                    precision : Int32,
                    levels : Array(Float32)) : {Float64, Float64}
    max_error = 0.0_f64
    squared_error = 0.0_f64
    ROW_VALUES.times do |i|
      z = (values[offset + i] - header.mean) / header.sigma
      raw = raw_code(z, levels)
      centroid = ML::GGUF::QwenQBitGaussianCodec.reconstruct_raw_code(raw, precision)
      error = (z - centroid).abs.to_f64
      max_error = Math.max(max_error, error)
      squared_error += error * error
    end
    {max_error, Math.sqrt(squared_error / ROW_VALUES)}
  end

  def expected_tier(values : Array(Float32), offset : Int32,
                    header : HyNomasCacheQBit::RowHeader,
                    levels : Array(Float32)) : UInt32
    return 4_u32 if header.sigma == 0.0_f32

    {
      {4, 0.50_f64, 0.12_f64},
      {5, 0.25_f64, 0.06_f64},
      {8, 0.10_f64, 0.02_f64},
    }.each do |precision, max_limit, rms_limit|
      max_error, rms_error = policy_errors(values, offset, header, precision, levels)
      return precision.to_u32 if max_error <= max_limit && rms_error <= rms_limit
    end
    32_u32
  end

  def assert_reference!(values : Array(Float32), restored : Array(Float32),
                        diagnostics : HyNomasCacheQBit::PackedDiagnostics,
                        levels : Array(Float32)) : Nil
    ROWS.times do |row|
      header = diagnostics.headers[row]
      input_offset = row * ROW_VALUES
      ROW_VALUES.times do |within|
        expected = decode_reference(header, diagnostics, row, within)
        actual = restored[input_offset + within]
        if header.tier == 32_u32
          unless actual.unsafe_as(UInt32) == expected.unsafe_as(UInt32)
            raise "F32 fallback changed row #{row}, value #{within}"
          end
          unless actual.unsafe_as(UInt32) == values[input_offset + within].unsafe_as(UInt32)
            raise "F32 escape did not preserve source bits at row #{row}, value #{within}"
          end
        else
          unless actual.finite? && expected.finite?
            raise "Quantized row #{row} produced a nonfinite value"
          end
          tolerance = 2e-5_f32 + 2e-5_f32 * expected.abs
          unless (actual - expected).abs <= tolerance
            raise "GPU decoder differs from Gaussian CPU reference at row #{row}, value #{within}"
          end
        end
      end

      next if row == 0 || row == 1 || header.tier == 32_u32
      expected = expected_tier(values, input_offset, header, levels)
      unless header.tier == expected
        raise "GPU selected p#{header.tier} but CPU selector expected p#{expected} for row #{row}"
      end
      # Zero-sigma constants are checked bitwise below; normalized error is
      # undefined for them and the lossless representation is exact.
      next if header.sigma == 0.0_f32

      max_limit, rms_limit = case header.tier
                             when 4_u32 then {0.50_f64, 0.12_f64}
                             when 5_u32 then {0.25_f64, 0.06_f64}
                             when 8_u32 then {0.10_f64, 0.02_f64}
                             else            raise "Unexpected quantized tier #{header.tier}"
                             end
      max_error = 0.0_f64
      squared_error = 0.0_f64
      ROW_VALUES.times do |within|
        sigma = header.sigma.to_f64
        error = (values[input_offset + within].to_f64 - restored[input_offset + within].to_f64).abs / sigma
        max_error = Math.max(max_error, error)
        squared_error += error * error
      end
      rms_error = Math.sqrt(squared_error / ROW_VALUES)
      unless max_error <= max_limit && rms_error <= rms_limit
        raise "Selected p#{header.tier} row exceeds its experimental normalized error limits"
      end
    end
  end
end

raise "Metal unavailable" unless ML::Metal::Device.init!

static_shape_rejected = false
begin
  HyNomasCacheQBit::Plan.named_bytes(127_i32, 1_i32)
rescue ex : ArgumentError
  static_shape_rejected = true
end
raise "Static byte admission accepted a non-128 row shape" unless static_shape_rejected

cursor_overflow_rejected = false
begin
  HyNomasCacheQBit::Plan.named_bytes(1_073_741_824_i32, 1_i32)
rescue ex : ArgumentError
  cursor_overflow_rejected = true
end
raise "Static byte admission accepted a uint32 payload cursor overflow" unless cursor_overflow_rejected

values = HyNomasCacheQBitProbe.inputs
buffer = ML::MetalBuffer.from_array(values)
binding = HyNomasCacheQBit::Binding.new("synthetic_cache", buffer, values.size.to_i32)

binding_shape_rejected = false
begin
  HyNomasCacheQBit::Plan.new([
    HyNomasCacheQBit::Binding.new("bad_shape", buffer, values.size.to_i32 - 1),
  ])
rescue ex : ArgumentError
  binding_shape_rejected = true
end
raise "Plan accepted a binding whose count is not divisible by 128" unless binding_shape_rejected

alias_rejected = false
begin
  HyNomasCacheQBit::Plan.new([
    HyNomasCacheQBit::Binding.new("first_alias", buffer, values.size.to_i32),
    HyNomasCacheQBit::Binding.new("second_alias", buffer, values.size.to_i32),
  ])
rescue ex : ArgumentError
  alias_rejected = true
end
raise "Plan accepted aliased cache bindings" unless alias_rejected

plan = HyNomasCacheQBit::Plan.new([binding])
chain = ML::Metal::MPSCommandChain.new
before_allocations = ML::MetalBuffer.hy_text_probe_allocations
before_readbacks = ML::MetalBuffer.hy_text_probe_readbacks
before_waits = chain.wait_count

begin
  plan.encode_mutation_control(chain, 0, 1, HyNomasCacheQBitProbe::HEADER_DELTA)
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  raise "QBit encode allocated Metal buffers, read activations, or waited" unless encode_allocations == 0 && encode_readbacks == 0 && chain.wait_count == before_waits && before_waits == 0

  chain.commit_and_wait
  raise "QBit diagnostic requires one terminal wait" unless chain.wait_count == 1

  report = plan.report_after_drain(chain)
  tiers = report["tier_counts"]
  expected_tiers = {"p4" => 4_i64, "p5" => 1_i64, "p8" => 1_i64, "f32" => 2_i64}
  expected_tiers.each do |tier, count|
    raise "Expected #{count} rows in #{tier}, got #{tiers[tier].as_i64}" unless tiers[tier].as_i64 == count
  end
  raise "Expected one invalid row" unless report["invalid_rows"].as_i64 == 1
  raise "Unexpected exact byte count" unless report["exact_bytes"].as_i64 == HyNomasCacheQBitProbe::COUNT.to_i64 * sizeof(Float32)

  invalid_rejected = false
  begin
    HyNomasCacheQBit::Plan.validate_report!(report)
  rescue ex : ArgumentError
    invalid_rejected = true
  end
  raise "Invalid cache rows were not rejected by the export gate" unless invalid_rejected

  diagnostics = plan.diagnostics_after_drain(chain)
  restored = buffer.read(HyNomasCacheQBitProbe::COUNT.to_i32)
  levels = ML::GGUF::QwenQBitGaussianCodec.positive_levels_for_setup
  HyNomasCacheQBitProbe.assert_reference!(values, restored, diagnostics, levels)

  constant_offset = 6 * HyNomasCacheQBitProbe::ROW_VALUES
  constant_header = diagnostics.headers[6]
  raise "Non-binary constant row did not select zero-sigma p4" unless constant_header.tier == 4_u32 && constant_header.sigma == 0.0_f32
  HyNomasCacheQBitProbe::ROW_VALUES.times do |within|
    input_bits = values[constant_offset + within].unsafe_as(UInt32)
    restored_bits = restored[constant_offset + within].unsafe_as(UInt32)
    raise "Non-binary constant row changed source bits" unless input_bits == restored_bits
  end
  negative_zero_offset = 7 * HyNomasCacheQBitProbe::ROW_VALUES
  raise "Negative zero row did not use zero-sigma p4" unless diagnostics.headers[7].tier == 4_u32 && diagnostics.headers[7].sigma == 0.0_f32
  HyNomasCacheQBitProbe::ROW_VALUES.times do |within|
    input_bits = values[negative_zero_offset + within].unsafe_as(UInt32)
    restored_bits = restored[negative_zero_offset + within].unsafe_as(UInt32)
    raise "Negative zero constant row changed source bits" unless input_bits == restored_bits
  end

  header_mutated = diagnostics.headers[0].mean == values[0] + HyNomasCacheQBitProbe::HEADER_DELTA
  raise "Header mutation positive control did not reach GPU decode" unless header_mutated && restored[0] == values[0] + HyNomasCacheQBitProbe::HEADER_DELTA

  bitplane_header = diagnostics.headers[1]
  raise "Bitplane mutation row unexpectedly escaped" if bitplane_header.tier == 32_u32
  original_z = (values[HyNomasCacheQBitProbe::ROW_VALUES] - bitplane_header.mean) / bitplane_header.sigma
  original_code = HyNomasCacheQBitProbe.raw_code(original_z, levels)
  expected_original_bit = (original_code & 0x80_u8) != 0
  changed_bit = (diagnostics.payload[bitplane_header.payload_offset.to_i + 15] & 0x01_u8) != 0
  raise "Bitplane mutation positive control did not change the selected bit" if changed_bit == expected_original_bit
  raise "Bitplane mutation did not change decoded output" if restored[HyNomasCacheQBitProbe::ROW_VALUES] == values[HyNomasCacheQBitProbe::ROW_VALUES]

  HyNomasCacheQBitProbe::ROWS.times do |row|
    next if row == 5 # The nonfinite exact-escape row is explicitly invalid.

    HyNomasCacheQBitProbe::ROW_VALUES.times do |within|
      unless restored[row * HyNomasCacheQBitProbe::ROW_VALUES + within].finite?
        raise "A finite cache row produced a nonfinite output"
      end
    end
  end

  logical = report["logical_packed_bytes"].as_i64
  expected_logical = HyNomasCacheQBitProbe::ROWS.to_i64 * HyNomasCacheQBit::HEADER_BYTES + diagnostics.payload.size
  raise "Packed byte report does not match full headers and payload" unless logical == expected_logical
  raise "Resident bytes disagree with static admission" unless report["resident_bytes"].as_i64 == HyNomasCacheQBit::Plan.named_bytes(HyNomasCacheQBitProbe::COUNT.to_i32, 1)

  puts(JSON.build do |json|
    json.object do
      json.field "schema", "hy-nomas-cache-qbit-probe-v1"
      json.field "report", report
      json.field "encode_named_allocations", encode_allocations
      json.field "encode_readbacks", encode_readbacks
      json.field "waits_before_terminal", before_waits
      json.field "terminal_waits", chain.wait_count
      json.field "static_shape_rejected", static_shape_rejected
      json.field "cursor_overflow_rejected", cursor_overflow_rejected
      json.field "binding_shape_rejected", binding_shape_rejected
      json.field "alias_rejected", alias_rejected
      json.field "cpu_decoder_parity", true
      json.field "f32_escape_exact", true
      json.field "constant_lossless", true
      json.field "nonfinite_rejected_before_export", invalid_rejected
      json.field "header_mutation_detected", header_mutated
      json.field "bitplane_mutation_detected", changed_bit != expected_original_bit
      json.field "nonbinary_constant_lossless", true
      json.field "negative_zero_constant_lossless", true
      json.field "finite_outputs", true
    end
  end)
ensure
  if chain.completed?
    plan.release_after_drain(chain)
    buffer.release
  end
end
