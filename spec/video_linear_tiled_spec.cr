require "spec"
require "json"
require "../src/ml/video/metal_linear_tiled"
require "../src/ml/video/metal_ops"
require "./support/video_allocation_counter"

private def video_linear_tiled_bf16(value : Float32) : Float32
  (value.unsafe_as(UInt32) & 0xffff0000_u32).unsafe_as(Float32)
end

private def video_linear_tiled_resident(
  arena : ML::Video::Arena, name : String, values : Array(Float32),
  dtype : ML::Video::WeightDType,
) : ML::MetalBuffer
  case dtype
  when ML::Video::WeightDType::F32
    arena.upload(name, values)
  when ML::Video::WeightDType::BF16
    buffer = arena.alloc_weight(name, values.size, dtype)
    bytes = Bytes.new(values.size * sizeof(UInt16))
    values.each_with_index do |value, index|
      bits = (value.unsafe_as(UInt32) >> 16).to_u16
      bytes[index * 2] = (bits & 0xff).to_u8
      bytes[index * 2 + 1] = (bits >> 8).to_u8
    end
    buffer.write_bytes(bytes.to_unsafe, bytes.size)
    buffer
  else
    raise ArgumentError.new("Unsupported tiled linear storage dtype")
  end
end

private def video_linear_tiled_fill_bf16_pattern(
  buffer : ML::MetalBuffer, element_count : Int32,
) : Nil
  pattern = StaticArray[0x3f80_u16, 0xbf80_u16, 0x3f00_u16, 0xbf00_u16,
    0x3e80_u16, 0xbe80_u16, 0x3f20_u16, 0xbf20_u16]
  bytes = Bytes.new(element_count * sizeof(UInt16))
  element_count.times do |index|
    bits = pattern[index % pattern.size]
    bytes[index * 2] = (bits & 0xff).to_u8
    bytes[index * 2 + 1] = (bits >> 8).to_u8
  end
  buffer.write_bytes(bytes.to_unsafe, bytes.size)
end

private def video_linear_tiled_dtype(name : String) : ML::Video::WeightDType
  case name
  when "f32"  then ML::Video::WeightDType::F32
  when "bf16" then ML::Video::WeightDType::BF16
  else             raise ArgumentError.new("Invalid fixture dtype #{name}")
  end
end

private def video_linear_tiled_values(value : JSON::Any) : Array(Float32)
  value.as_a.map(&.as_f.to_f32)
end

private def video_linear_tiled_assert_close(
  name : String, actual : Array(Float32), expected : Array(Float32),
) : Nil
  actual.size.should eq(expected.size), name
  violations = 0
  worst = 0_f64
  actual.each_with_index do |value, index|
    target = expected[index]
    ratio = (value - target).abs.to_f64 / (2e-5_f64 + 2e-5_f64 * target.abs.to_f64)
    violations += 1 unless value.finite? && target.finite? && ratio <= 1.0
    worst = Math.max(worst, ratio)
  end
  violations.should eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}"
end

describe ML::Video::MetalLinearTiled do
  it "matches the scalar GPU primitive for mixed storage and all 8x8 tails" do
    rows, in_dim, out_dim = 19, 19, 13
    input = Array(Float32).new(rows * in_dim) do |i|
      (((i * 17) % 37 - 18) * 0.03125).to_f32
    end
    weight = Array(Float32).new(out_dim * in_dim) do |i|
      (((i * 11) % 29 - 14) * 0.015625).to_f32
    end
    bias = Array(Float32).new(out_dim) { |i| ((i - 6) * 0.0625).to_f32 }

    {
      {ML::Video::WeightDType::F32, ML::Video::WeightDType::F32, ML::Video::WeightDType::F32},
      {ML::Video::WeightDType::F32, ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16},
      {ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16, ML::Video::WeightDType::F32},
      {ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16, ML::Video::WeightDType::BF16},
      {ML::Video::WeightDType::BF16, ML::Video::WeightDType::F32, ML::Video::WeightDType::F32},
      {ML::Video::WeightDType::BF16, ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16},
      {ML::Video::WeightDType::BF16, ML::Video::WeightDType::BF16, ML::Video::WeightDType::F32},
      {ML::Video::WeightDType::BF16, ML::Video::WeightDType::BF16, ML::Video::WeightDType::BF16},
    }.each_with_index do |(input_dtype, weight_dtype, bias_dtype), case_index|
      arena = ML::Video::Arena.new
      begin
        ML::Video::MetalLinearTiled.prepare!
        x = video_linear_tiled_resident(arena, "x", input, input_dtype)
        w = video_linear_tiled_resident(arena, "w", weight, weight_dtype)
        b = video_linear_tiled_resident(arena, "b", bias, bias_dtype)
        x_f32 = arena.upload("x_f32", input_dtype.bf16? ? input.map { |v| video_linear_tiled_bf16(v) } : input)
        w_f32 = arena.upload("w_f32", weight_dtype.bf16? ? weight.map { |v| video_linear_tiled_bf16(v) } : weight)
        b_f32 = arena.upload("b_f32", bias_dtype.bf16? ? bias.map { |v| video_linear_tiled_bf16(v) } : bias)
        full = arena.alloc("full", rows * out_dim)
        split = arena.alloc("split", rows * out_dim)
        scalar = arena.alloc("scalar", rows * out_dim)

        allocations = ML::MetalBuffer.video_spec_allocations
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        ML::Video::MetalLinearTiled.linear(encoder, x, w, b, full, rows, in_dim, out_dim,
          input_dtype, weight_dtype, bias_dtype)
        ML::Video::MetalLinearTiled.linear_range(encoder, x, w, b, split, rows, in_dim,
          out_dim, 0, 8, input_dtype, weight_dtype, bias_dtype)
        ML::Video::MetalLinearTiled.linear_range(encoder, x, w, b, split, rows, in_dim,
          out_dim, 8, 8, input_dtype, weight_dtype, bias_dtype)
        ML::Video::MetalLinearTiled.linear_range(encoder, x, w, b, split, rows, in_dim,
          out_dim, 16, 3, input_dtype, weight_dtype, bias_dtype)
        ML::Video::MetalOps.linear(encoder, x_f32, w_f32, b_f32, scalar, rows, in_dim, out_dim)
        encoder.end_encoding
        command.commit
        command.wait

        ML::MetalBuffer.video_spec_allocations.should eq(allocations),
          "encode allocated buffers in dtype case #{case_index}"
        full_values = full.read(rows * out_dim)
        split_values = split.read(rows * out_dim)
        scalar_values = scalar.read(rows * out_dim)
        full_values.map(&.unsafe_as(UInt32)).should eq(split_values.map(&.unsafe_as(UInt32))),
          "full/ranged result differs in dtype case #{case_index}"
        video_linear_tiled_assert_close("tiled/scalar case #{case_index}", full_values, scalar_values)
      ensure
        arena.release
      end
    end
  end

  it "rejects invalid row ranges and unsafe aliases before dispatch" do
    arena = ML::Video::Arena.new
    begin
      x = arena.upload("x", Array(Float32).new(19 * 5, 1_f32))
      w = arena.upload("w", Array(Float32).new(7 * 5, 1_f32))
      b = arena.upload("b", Array(Float32).new(7, 0_f32))
      alias_weight = arena.upload("alias_weight", Array(Float32).new(7 * 7, 1_f32))
      output = arena.alloc("output", 19 * 7)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      expect_raises(ArgumentError) do
        ML::Video::MetalLinearTiled.linear_range(encoder, x, w, b, output,
          19, 5, 7, 1, 8)
      end
      expect_raises(ArgumentError) do
        ML::Video::MetalLinearTiled.linear_range(encoder, x, w, b, output,
          19, 5, 7, 8, 3)
      end
      expect_raises(ArgumentError) do
        ML::Video::MetalLinearTiled.linear_range(encoder, x, w, b, output,
          19, 5, 7, 16, 4)
      end
      expect_raises(ArgumentError, "Unsafe video buffer alias") do
        ML::Video::MetalLinearTiled.linear_range(encoder, output, alias_weight, b, output,
          19, 7, 7, 0, 19)
      end
      encoder.end_encoding
      command.discard
    ensure
      arena.release
    end
  end

  it "continues bounded row segments asynchronously and waits only at the caller boundary" do
    rows, in_dim, out_dim = 19, 5, 7
    arena = ML::Video::Arena.new
    begin
      x = arena.upload("x", Array(Float32).new(rows * in_dim) { |i| (i % 13).to_f32 / 16_f32 })
      w = arena.upload("w", Array(Float32).new(out_dim * in_dim) { |i| (i % 11).to_f32 / 32_f32 })
      b = arena.upload("b", Array(Float32).new(out_dim) { |i| i.to_f32 / 8_f32 })
      full = arena.alloc("full", rows * out_dim)
      bounded = arena.alloc("bounded", rows * out_dim)
      ML::Video::MetalLinearTiled.prepare!

      full_command = ML::Metal::CommandBuffer.new
      full_encoder = ML::Metal::ComputeEncoder.new(full_command)
      ML::Video::MetalLinearTiled.linear(full_encoder, x, w, b, full, rows, in_dim, out_dim)
      full_encoder.end_encoding
      full_command.commit
      full_command.wait

      chain = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.video_spec_allocations
      ML::Video::MetalLinearTiled.linear_bounded(chain, x, w, b, bounded,
        rows, in_dim, out_dim, rows_per_segment: 8)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      chain.wait_count.should eq(0)
      chain.commit_and_wait
      chain.wait_count.should eq(1)
      chain.segment_count.should eq(3)
      chain.release

      full.read(rows * out_dim).map(&.unsafe_as(UInt32)).should eq(
        bounded.read(rows * out_dim).map(&.unsafe_as(UInt32)))
    ensure
      arena.release
    end
  end

  it "profiles a bounded Wan-width F32-accumulating projection against the scalar path" do
    unless ENV["VIDEO_LINEAR_TILED_COST_PROBE"]? == "1"
      pending! "Set VIDEO_LINEAR_TILED_COST_PROBE=1 to allocate the 135 MiB synthetic BF16 projection weights"
    end
    rows = (ENV["VIDEO_LINEAR_TILED_COST_ROWS"]? || "8").to_i
    repeats = (ENV["VIDEO_LINEAR_TILED_COST_REPEATS"]? || "1").to_i
    in_dim, out_dim = 5120, 13_824
    unless rows > 0 && rows <= 64 && rows % 8 == 0
      raise ArgumentError.new("Cost probe rows must be an 8-aligned value in 8..64")
    end
    unless repeats >= 1 && repeats <= 8
      raise ArgumentError.new("Cost probe repeats must be in 1..8")
    end

    arena = ML::Video::Arena.new
    begin
      ML::Video::MetalLinearTiled.prepare!
      x = arena.upload("x", Array(Float32).new(rows * in_dim) { |i| ((i % 31) - 15).to_f32 / 32_f32 })
      weight = arena.alloc_weight("weight_bf16", out_dim * in_dim, ML::Video::WeightDType::BF16)
      video_linear_tiled_fill_bf16_pattern(weight, out_dim * in_dim)
      bias = arena.upload("bias", Array(Float32).new(out_dim, 0_f32))
      tiled = arena.alloc("tiled", rows * out_dim)
      scalar = arena.alloc("scalar", rows * out_dim)

      measure_tiled = -> do
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        allocations = ML::MetalBuffer.video_spec_allocations
        ML::Video::MetalLinearTiled.linear_range(encoder, x, weight, bias, tiled,
          rows, in_dim, out_dim, 0, rows, weight_dtype: ML::Video::WeightDType::BF16)
        encoder.end_encoding
        ML::MetalBuffer.video_spec_allocations.should eq(allocations), "tiled cost encode allocated buffers"
        command.commit_and_wait_gpu_elapsed_seconds
      end
      measure_scalar = -> do
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        allocations = ML::MetalBuffer.video_spec_allocations
        ML::Video::MetalOps.linear(encoder, x, weight, bias, scalar, rows, in_dim, out_dim,
          weight_dtype: ML::Video::WeightDType::BF16)
        encoder.end_encoding
        ML::MetalBuffer.video_spec_allocations.should eq(allocations), "scalar cost encode allocated buffers"
        command.commit_and_wait_gpu_elapsed_seconds
      end

      tiled_samples = [] of Float64
      scalar_samples = [] of Float64
      repeats.times do |sample_index|
        if sample_index.even?
          tiled_samples << measure_tiled.call
          scalar_samples << measure_scalar.call
        else
          scalar_samples << measure_scalar.call
          tiled_samples << measure_tiled.call
        end
      end

      tiled_values = tiled.read(rows * out_dim)
      scalar_values = scalar.read(rows * out_dim)
      video_linear_tiled_assert_close("cost probe tiled/scalar", tiled_values, scalar_values)

      tiled_seconds = video_linear_tiled_median(tiled_samples)
      scalar_seconds = video_linear_tiled_median(scalar_samples)

      flops = 2_i64 * rows * in_dim * out_dim
      timing = String.build do |io|
        io << "video_linear_tiled_cost rows=" << rows << " in=" << in_dim << " out=" << out_dim
        io << " bf16_weight_bytes=" << weight.size << " f32_flops=" << flops
        io << " repeats=" << repeats
        io << " tiled_samples_seconds=" << tiled_samples.join(",")
        io << " scalar_samples_seconds=" << scalar_samples.join(",")
        io << " tiled_gpu_seconds=" << tiled_seconds << " scalar_gpu_seconds=" << scalar_seconds
        io << " speedup=" << scalar_seconds / tiled_seconds
      end
      puts timing
    ensure
      arena.release
    end
  end

  it "matches the pinned PyTorch MPS fixture for all independent storage dtypes" do
    fixture_path = ENV["VIDEO_LINEAR_TILED_MPS_FIXTURE"]?
    unless fixture_path
      pending! "Set VIDEO_LINEAR_TILED_MPS_FIXTURE from spec/support/video_linear_tiled_reference.py"
    end
    fixture = JSON.parse(File.read(fixture_path))
    fixture["schema"].as_s.should eq("video-linear-tiled-mps-v2")
    fixture["upstream"]["oracle_device"].as_s.should eq("mps")
    fixture["upstream"]["operation"].as_s.should eq("torch.nn.functional.linear")
    fixture["upstream"]["torch_version"].as_s.starts_with?("2.8.0").should be_true
    fixture["upstream"]["mps_fallback_disabled"].as_bool.should be_true
    groups = fixture["groups"].as_a
    groups.size.should eq(3)
    expected_shapes = [{19, 19, 13}, {3, 5120, 5}, {3, 13824, 5}]
    groups.each_with_index do |group, group_index|
      shape = group["shape"]
      rows, in_dim, out_dim = shape["rows"].as_i, shape["in_dim"].as_i, shape["out_dim"].as_i
      {rows, in_dim, out_dim}.should eq(expected_shapes[group_index])
      input_values = video_linear_tiled_values(group["input"])
      weight_values = video_linear_tiled_values(group["weight"])
      bias_values = video_linear_tiled_values(group["bias"])
      cases = group["cases"].as_a
      cases.size.should eq(8)
      dtype_triples = [] of Tuple(String, String, String)

      cases.each_with_index do |item, case_index|
        expected = video_linear_tiled_values(item["output"])
        input_dtype_name = item["input_dtype"].as_s
        weight_dtype_name = item["weight_dtype"].as_s
        bias_dtype_name = item["bias_dtype"].as_s
        dtype_triples << {input_dtype_name, weight_dtype_name, bias_dtype_name}
        input_dtype = video_linear_tiled_dtype(input_dtype_name)
        weight_dtype = video_linear_tiled_dtype(weight_dtype_name)
        bias_dtype = video_linear_tiled_dtype(bias_dtype_name)

        arena = ML::Video::Arena.new
        begin
          ML::Video::MetalLinearTiled.prepare!
          x = video_linear_tiled_resident(arena, "x", input_values, input_dtype)
          w = video_linear_tiled_resident(arena, "w", weight_values, weight_dtype)
          b = video_linear_tiled_resident(arena, "b", bias_values, bias_dtype)
          output = arena.alloc("output", rows * out_dim)
          command = ML::Metal::CommandBuffer.new
          encoder = ML::Metal::ComputeEncoder.new(command)
          ML::Video::MetalLinearTiled.linear(encoder, x, w, b, output, rows, in_dim, out_dim,
            input_dtype, weight_dtype, bias_dtype)
          encoder.end_encoding
          command.commit
          command.wait
          video_linear_tiled_assert_close("MPS fixture group #{group_index} case #{case_index}",
            output.read(rows * out_dim), expected)
        ensure
          arena.release
        end
      end
      dtype_triples.uniq.size.should eq(8), "MPS fixture group #{group_index} must cover 8 unique dtype triples"
    end
  end
end

private def video_linear_tiled_median(values : Array(Float64)) : Float64
  ordered = values.sort
  middle = ordered.size // 2
  if ordered.size.odd?
    ordered[middle]
  else
    (ordered[middle - 1] + ordered[middle]) / 2_f64
  end
end
