require "spec"
require "../src/ml/video/qbit_positional_mps_linear"
require "./support/video_allocation_counter"

private def qbit_pos_mps_bf16_bits(value : Float32) : UInt16
  (value.unsafe_as(UInt32) >> 16).to_u16
end

private def qbit_pos_mps_bf16_value(value : Float32) : Float32
  (qbit_pos_mps_bf16_bits(value).to_u32 << 16).unsafe_as(Float32)
end

private def qbit_pos_mps_raw_records(
  arena : ML::Video::Arena, prefix : String, weights : Array(Float32),
) : {offsets: ML::MetalBuffer, blob: ML::MetalBuffer}
  unless weights.size > 0 && weights.size % 256 == 0
    raise ArgumentError.new("Raw positional fixture requires whole 256-value tiles")
  end

  tiles = weights.size // 256
  offsets = arena.alloc("#{prefix}_offsets", tiles + 1)
  blob = arena.alloc("#{prefix}_blob", tiles * 516 // 4)
  offset_words = offsets.contents.as(UInt32*)
  blob_bytes = blob.contents.as(UInt8*)
  tiles.times do |tile|
    offset_words[tile] = (tile * 516).to_u32
    record = blob_bytes + tile * 516
    record[0] = 16_u8
    record[1] = 0_u8
    record[2] = 0_u8
    record[3] = 0_u8
    256.times do |within|
      bits = qbit_pos_mps_bf16_bits(weights[tile * 256 + within])
      record[4 + within * 2] = (bits & 0xff_u16).to_u8
      record[5 + within * 2] = (bits >> 8).to_u8
    end
  end
  offset_words[tiles] = (tiles * 516).to_u32
  {offsets: offsets, blob: blob}
end

private def qbit_pos_mps_bias_bf16(
  arena : ML::Video::Arena, name : String, values : Array(Float32),
) : ML::MetalBuffer
  buffer = arena.alloc_weight(name, values.size, ML::Video::WeightDType::BF16)
  bytes = Bytes.new(values.size * sizeof(UInt16))
  values.each_with_index do |value, index|
    bits = qbit_pos_mps_bf16_bits(value)
    bytes[index * 2] = (bits & 0xff_u16).to_u8
    bytes[index * 2 + 1] = (bits >> 8).to_u8
  end
  buffer.write_bytes(bytes.to_unsafe, bytes.size)
  buffer
end

private def qbit_pos_mps_set_status(buffer : ML::MetalBuffer, values : Array(UInt32)) : Nil
  raise ArgumentError.new("Status fixture must contain eight words") unless values.size == 8
  words = buffer.contents.as(UInt32*)
  values.each_with_index { |value, index| words[index] = value }
end

private def qbit_pos_mps_status(buffer : ML::MetalBuffer) : Array(UInt32)
  words = buffer.contents.as(UInt32*)
  Array(UInt32).new(ML::Video::QBitPositionalWeights::STATUS_WORDS) do |index|
    words[index]
  end
end

private def qbit_pos_mps_expected(
  input : Array(Float32), weights : Array(Float32), bias : Array(Float32),
  rows : Int32, input_dim : Int32, output_dim : Int32, apply_bias : Bool,
) : Array(Float32)
  output = Array(Float32).new(rows * output_dim, 0_f32)
  rows.times do |row|
    output_dim.times do |column|
      sum = 0_f32
      input_dim.times do |k|
        sum += input[row * input_dim + k] * weights[column * input_dim + k]
      end
      sum += bias[column] if apply_bias
      output[row * output_dim + column] = sum
    end
  end
  output
end

private def qbit_pos_mps_assert_close(
  name : String, actual : Array(Float32), expected : Array(Float32),
) : Nil
  actual.size.should eq(expected.size)
  violations = 0
  worst = 0_f64
  actual.each_with_index do |value, index|
    target = expected[index]
    ratio = (value - target).abs.to_f64 /
            (2e-5_f64 + 2e-5_f64 * target.abs.to_f64)
    violations += 1 unless value.finite? && target.finite? && ratio <= 1.0
    worst = Math.max(worst, ratio)
  end
  violations.should(eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}")
end

private def qbit_pos_mps_write_zero_luts(
  lut4 : ML::MetalBuffer, lut5 : ML::MetalBuffer,
) : Nil
  lut4.write(Array(Float32).new(256, 0_f32))
  lut5.write(Array(Float32).new(256, 0_f32))
end

private def qbit_pos_mps_guard_sentinels(
  stage : ML::MetalBuffer, status : ML::MetalBuffer, stage_count : Int32,
  &assertion : ML::Metal::ComputeEncoder -> Nil
) : Nil
  expected_stage = Array(Float32).new(stage_count, 19.25_f32)
  stage.write(expected_stage)
  expected_status = Array(UInt32).new(8) { |index| 0x6a10_0020_u32 + index.to_u32 }
  qbit_pos_mps_set_status(status, expected_status)

  command = nil.as(ML::Metal::CommandBuffer?)
  begin
    active = ML::Metal::CommandBuffer.new
    command = active
    encoder = ML::Metal::ComputeEncoder.new(active)
    assertion.call(encoder)
    encoder.end_encoding
    active.commit
    active.wait

    stage.read(stage_count).map(&.unsafe_as(UInt32)).should(
      eq(expected_stage.map(&.unsafe_as(UInt32))), "preflight failure must not decode into the stage")
    qbit_pos_mps_status(status).should(
      eq(expected_status), "preflight failure must not mutate caller status")
  ensure
    if active = command
      active.discard unless active.committed?
    end
  end
end

describe "QBit positional MPS linear consumer" do
  it "reuses one stage and plan for raw BF16 A/B/A with F32, BF16, and disabled bias" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!

    rows = 4
    input_dim = 256
    output_dim = 2
    matrix_count = input_dim * output_dim
    arena = ML::Video::Arena.new
    plan = nil.as(ML::Video::WanT5MPSLinearPlan?)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      ML::Video::QBitPositionalMPSLinear.prepare!

      input_values = Array(Float32).new(rows * input_dim) do |index|
        row = index // input_dim
        k = index % input_dim
        (((k + row * 3) % 8).to_i32 - 4).to_f32 / 8_f32
      end
      weights_a = Array(Float32).new(matrix_count, 0_f32)
      weights_b = Array(Float32).new(matrix_count, 0_f32)
      125.times do |k|
        weights_a[k] = 0.5_f32
        weights_a[input_dim + k] = -0.5_f32
        weights_b[k] = -0.5_f32
        weights_b[input_dim + k] = 0.5_f32
      end
      f32_bias_values = [0.25_f32, -0.125_f32]
      bf16_bias_values = [0.125_f32, -0.25_f32]

      input = arena.alloc("input", rows * input_dim)
      input.write(input_values)
      input_bias = arena.alias_f32("input_bias", "input")
      raw_a = qbit_pos_mps_raw_records(arena, "raw_a", weights_a)
      raw_b = qbit_pos_mps_raw_records(arena, "raw_b", weights_b)
      lut4 = arena.alloc("lut4", 256)
      lut5 = arena.alloc("lut5", 256)
      qbit_pos_mps_write_zero_luts(lut4, lut5)
      bias_f32 = arena.alloc("bias_f32", output_dim)
      bias_f32.write(f32_bias_values)
      bias_bf16 = qbit_pos_mps_bias_bf16(arena, "bias_bf16", bf16_bias_values)
      stage = arena.alloc("weight_stage", matrix_count)
      output = arena.alloc("projected", rows * output_dim)
      status = arena.alloc("status", ML::Video::QBitPositionalWeights::STATUS_WORDS)
      qbit_pos_mps_set_status(status, Array(UInt32).new(8, 0_u32))
      outputs = Array(ML::MetalBuffer).new(6) do |index|
        arena.alloc("snapshot_#{index}", rows * output_dim)
      end

      cfg = ML::Video::WanT5Config.new(vocab: 2, dim: input_dim, ffn_dim: output_dim,
        heads: 1, layers: 1, batches: 1, tokens: rows)
      plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, stage, output)

      before_allocations = ML::MetalBuffer.video_spec_allocations
      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw_a[:offsets], raw_a[:blob], lut4, lut5, bias_f32, output, status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, outputs[0], rows * output_dim)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw_b[:offsets], raw_b[:blob], lut4, lut5, bias_bf16, output, status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::BF16, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, outputs[1], rows * output_dim)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw_a[:offsets], raw_a[:blob], lut4, lut5, bias_f32, output, status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, outputs[2], rows * output_dim)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw_a[:offsets], raw_a[:blob], lut4, lut5, bias_bf16, output, status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::BF16, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, outputs[3], rows * output_dim)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw_a[:offsets], raw_a[:blob], lut4, lut5, bias_f32, output, status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, plan.not_nil!, stage, false)
      ML::Video::MetalOps.copy(encoder, output, outputs[4], rows * output_dim)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw_a[:offsets], raw_a[:blob], lut4, lut5, input_bias, output, status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, outputs[5], rows * output_dim)
      encoder.end_encoding
      ML::MetalBuffer.video_spec_allocations.should(eq(before_allocations),
        "positional decode and preexpanded MPS encode must allocate no Metal buffers")
      active.commit
      active.wait

      expected_a_f32 = qbit_pos_mps_expected(input_values, weights_a, f32_bias_values,
        rows, input_dim, output_dim, true)
      expected_b_bf16 = qbit_pos_mps_expected(input_values, weights_b,
        bf16_bias_values.map { |value| qbit_pos_mps_bf16_value(value) },
        rows, input_dim, output_dim, true)
      expected_a_bf16 = qbit_pos_mps_expected(input_values, weights_a,
        bf16_bias_values.map { |value| qbit_pos_mps_bf16_value(value) },
        rows, input_dim, output_dim, true)
      expected_a_no_bias = qbit_pos_mps_expected(input_values, weights_a, f32_bias_values,
        rows, input_dim, output_dim, false)
      expected_a_input_bias = qbit_pos_mps_expected(input_values, weights_a,
        [input_values[0], input_values[1]], rows, input_dim, output_dim, true)
      actual_a0 = outputs[0].read(rows * output_dim)
      actual_b = outputs[1].read(rows * output_dim)
      actual_a1 = outputs[2].read(rows * output_dim)
      actual_a_bf16 = outputs[3].read(rows * output_dim)
      actual_a_no_bias = outputs[4].read(rows * output_dim)
      actual_a_input_bias = outputs[5].read(rows * output_dim)
      qbit_pos_mps_assert_close("A/F32 bias", actual_a0, expected_a_f32)
      qbit_pos_mps_assert_close("B/BF16 bias", actual_b, expected_b_bf16)
      qbit_pos_mps_assert_close("A/BF16 bias", actual_a_bf16, expected_a_bf16)
      qbit_pos_mps_assert_close("A/bias disabled", actual_a_no_bias, expected_a_no_bias)
      qbit_pos_mps_assert_close("A/input also used as read-only bias",
        actual_a_input_bias, expected_a_input_bias)
      actual_a0.map(&.unsafe_as(UInt32)).should(eq(actual_a1.map(&.unsafe_as(UInt32))),
        "A must reproduce bit-for-bit after B overwrites the shared stage")
      actual_a0.map(&.unsafe_as(UInt32)).should_not(eq(actual_b.map(&.unsafe_as(UInt32))),
        "B must replace A in the shared stage")
      qbit_pos_mps_status(status).should(eq(Array(UInt32).new(8, 0_u32)))
    ensure
      if active = command
        active.discard unless active.committed?
      end
      plan.try(&.release)
      arena.release
    end
  end

  it "fails closed on malformed records, preserves status errors, and poisons pack failures" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!

    rows = 4
    input_dim = 256
    output_dim = 2
    matrix_count = input_dim * output_dim
    arena = ML::Video::Arena.new
    plan = nil.as(ML::Video::WanT5MPSLinearPlan?)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      ML::Video::QBitPositionalMPSLinear.prepare!
      input_values = Array(Float32).new(rows * input_dim, 0.5_f32)
      weights = Array(Float32).new(matrix_count, 0_f32)
      bias_values = [0.25_f32, -0.125_f32]
      input = arena.alloc("input", rows * input_dim)
      input.write(input_values)
      raw = qbit_pos_mps_raw_records(arena, "valid_raw", weights)
      malformed_blob = arena.alloc("malformed_blob", matrix_count * 516 // 256 // 4)
      malformed_blob.copy_from(raw[:blob], raw[:blob].size)
      malformed_blob.contents.as(UInt8*)[0] = 6_u8
      lut4 = arena.alloc("lut4", 256)
      lut5 = arena.alloc("lut5", 256)
      qbit_pos_mps_write_zero_luts(lut4, lut5)
      bias = arena.alloc("bias", output_dim)
      bias.write(bias_values)
      stage = arena.alloc("stage", matrix_count)
      output = arena.alloc("projected", rows * output_dim)
      malformed_output = arena.alloc("malformed_output", rows * output_dim)
      valid_output = arena.alloc("valid_output", rows * output_dim)
      poisoned5_output = arena.alloc("poisoned5_output", rows * output_dim)
      poisoned6_output = arena.alloc("poisoned6_output", rows * output_dim)
      sticky_status = arena.alloc("sticky_status", ML::Video::QBitPositionalWeights::STATUS_WORDS)
      poisoned5_status = arena.alloc("poisoned5_status", ML::Video::QBitPositionalWeights::STATUS_WORDS)
      poisoned6_status = arena.alloc("poisoned6_status", ML::Video::QBitPositionalWeights::STATUS_WORDS)
      qbit_pos_mps_set_status(sticky_status, Array(UInt32).new(8, 0_u32))
      status5 = Array(UInt32).new(8, 0_u32); status5[5] = 1_u32
      status6 = Array(UInt32).new(8, 0_u32); status6[6] = 1_u32
      qbit_pos_mps_set_status(poisoned5_status, status5)
      qbit_pos_mps_set_status(poisoned6_status, status6)

      cfg = ML::Video::WanT5Config.new(vocab: 2, dim: input_dim, ffn_dim: output_dim,
        heads: 1, layers: 1, batches: 1, tokens: rows)
      plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, stage, output)

      before_allocations = ML::MetalBuffer.video_spec_allocations
      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw[:offsets], malformed_blob, lut4, lut5, bias, output, sticky_status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, malformed_output, rows * output_dim)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw[:offsets], raw[:blob], lut4, lut5, bias, output, sticky_status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, valid_output, rows * output_dim)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw[:offsets], raw[:blob], lut4, lut5, bias, output, poisoned5_status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, poisoned5_output, rows * output_dim)
      ML::Video::QBitPositionalMPSLinear.project(encoder, input,
        raw[:offsets], raw[:blob], lut4, lut5, bias, output, poisoned6_status,
        rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, plan.not_nil!, stage)
      ML::Video::MetalOps.copy(encoder, output, poisoned6_output, rows * output_dim)
      encoder.end_encoding
      ML::MetalBuffer.video_spec_allocations.should(eq(before_allocations),
        "malformed and valid positional decode must not allocate Metal buffers")
      active.commit
      active.wait

      bad = malformed_output.read(rows * output_dim)
      rows.times { |row| bad[row * output_dim].nan?.should be_true }
      rows.times { |row| bad[row * output_dim + 1].finite?.should be_true }
      qbit_pos_mps_assert_close("valid decode after malformed record",
        valid_output.read(rows * output_dim), bias_values + bias_values + bias_values + bias_values)
      qbit_pos_mps_status(sticky_status)[7].should eq(1_u32)
      qbit_pos_mps_status(sticky_status)[5].should eq(0_u32)
      qbit_pos_mps_status(sticky_status)[6].should eq(0_u32)
      poisoned5 = poisoned5_output.read(rows * output_dim)
      poisoned6 = poisoned6_output.read(rows * output_dim)
      poisoned5.all?(&.nan?).should be_true
      poisoned6.all?(&.nan?).should be_true
      qbit_pos_mps_status(poisoned5_status)[5].should eq(1_u32)
      qbit_pos_mps_status(poisoned5_status)[7].should eq(2_u32)
      qbit_pos_mps_status(poisoned6_status)[6].should eq(1_u32)
      qbit_pos_mps_status(poisoned6_status)[7].should eq(2_u32)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      plan.try(&.release)
      arena.release
    end
  end

  it "rejects shape, capacity, plan, and alias errors before decode changes stage or status" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!

    rows = 4
    input_dim = 256
    output_dim = 2
    matrix_count = input_dim * output_dim
    arena = ML::Video::Arena.new
    plans = [] of ML::Video::WanT5MPSLinearPlan
    no_copy_aliases = [] of ML::MetalBuffer
    begin
      ML::Video::QBitPositionalMPSLinear.prepare!
      input = arena.alloc("input", 4096)
      input.write(Array(Float32).new(rows * input_dim, 0_f32))
      raw = qbit_pos_mps_raw_records(arena, "raw", Array(Float32).new(matrix_count, 0_f32))
      lut4 = arena.alloc("lut4", 256)
      lut5 = arena.alloc("lut5", 256)
      qbit_pos_mps_write_zero_luts(lut4, lut5)
      bias = arena.alloc("bias", 8)
      bias.write(Array(Float32).new(8, 0_f32))
      stage = arena.alloc("stage", matrix_count)
      output = arena.alloc("output", rows * output_dim)
      status = arena.alloc("status", ML::Video::QBitPositionalWeights::STATUS_WORDS)
      cfg = ML::Video::WanT5Config.new(vocab: 2, dim: input_dim, ffn_dim: output_dim,
        heads: 1, layers: 1, batches: 1, tokens: rows)
      plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, stage, output)
      plans << plan

      small_input = arena.alloc("small_input", rows * input_dim - 1)
      small_stage = arena.alloc("small_stage", matrix_count - 1)
      small_output = arena.alloc("small_output", rows * output_dim - 1)
      small_bias = arena.alloc("small_bias", 1)
      small_status = arena.alloc("small_status", ML::Video::QBitPositionalWeights::STATUS_WORDS - 1)
      small_offsets = arena.alloc("small_offsets", 2)
      small_blob = arena.alloc_weight("small_blob", 1, ML::Video::WeightDType::BF16)
      small_lut = arena.alloc("small_lut", 255)

      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status, rows - 1, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, small_input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, small_stage)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, small_output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, small_bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage, false)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, small_status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, small_offsets, raw[:blob],
            lut4, lut5, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], small_blob,
            lut4, lut5, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            small_lut, lut5, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, small_lut, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end

      stage_input_alias = ML::MetalBuffer.wrap_no_copy(input.contents, input.size)
      no_copy_aliases << stage_input_alias
      qbit_pos_mps_guard_sentinels(stage_input_alias, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage_input_alias)
        end
      end
      bias_stage_alias = arena.alias_f32("bias_stage_alias", "stage")
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias_stage_alias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end

      status_input_alias = arena.alias_f32("status_input_alias", "input")
      qbit_pos_mps_guard_sentinels(stage, status_input_alias, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status_input_alias, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      status_bias_alias = arena.alias_f32("status_bias_alias", "bias")
      qbit_pos_mps_guard_sentinels(stage, status_bias_alias, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status_bias_alias, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end
      status_output_alias = arena.alias_f32("status_output_alias", "output")
      qbit_pos_mps_guard_sentinels(stage, status_output_alias, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status_output_alias, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, plan, stage)
        end
      end

      offsets_owner = arena.alloc("offsets_alias_owner", 8)
      offsets_owner.contents.as(UInt32*)[0] = 0_u32
      offsets_owner.contents.as(UInt32*)[1] = 516_u32
      offsets_owner.contents.as(UInt32*)[2] = 1032_u32
      output_offsets_alias = arena.alias_f32("output_offsets_alias", "offsets_alias_owner")
      output_offsets_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, stage, output_offsets_alias)
      plans << output_offsets_plan
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input,
            output_offsets_alias, raw[:blob], lut4, lut5, bias, output_offsets_alias, status,
            rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, output_offsets_plan, stage)
        end
      end

      output_blob_alias = arena.alias_f32("output_blob_alias", "raw_blob")
      output_blob_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, stage, output_blob_alias)
      plans << output_blob_plan
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input,
            raw[:offsets], output_blob_alias, lut4, lut5, bias, output_blob_alias, status,
            rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, output_blob_plan, stage)
        end
      end

      output_lut4_alias = arena.alias_f32("output_lut4_alias", "lut4")
      output_lut4_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, stage, output_lut4_alias)
      plans << output_lut4_plan
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input,
            raw[:offsets], raw[:blob], output_lut4_alias, lut5, bias, output_lut4_alias, status,
            rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, output_lut4_plan, stage)
        end
      end

      output_lut5_alias = arena.alias_f32("output_lut5_alias", "lut5")
      output_lut5_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, stage, output_lut5_alias)
      plans << output_lut5_plan
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input,
            raw[:offsets], raw[:blob], lut4, output_lut5_alias, bias, output_lut5_alias, status,
            rows, input_dim, output_dim, 256, ML::Video::WeightDType::F32, output_lut5_plan, stage)
        end
      end

      wrong_stage = arena.alloc("wrong_stage", matrix_count)
      wrong_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, wrong_stage, output)
      plans << wrong_plan
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        expect_raises(ArgumentError) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, wrong_plan, stage)
        end
      end

      released_plan = ML::Video::WanT5MPSLinearPlan.new(cfg, input_dim, output_dim,
        input, stage, output)
      released_plan.release
      qbit_pos_mps_guard_sentinels(stage, status, matrix_count) do |encoder|
        error = expect_raises(Exception) do
          ML::Video::QBitPositionalMPSLinear.project(encoder, input, raw[:offsets], raw[:blob],
            lut4, lut5, bias, output, status, rows, input_dim, output_dim, 256,
            ML::Video::WeightDType::F32, released_plan, stage)
        end
        error.message.to_s.should(contain("released"))
      end
    ensure
      no_copy_aliases.each(&.release)
      plans.each(&.release)
      arena.release
    end
  end
end
