require "spec"
require "json"
require "../src/ml/video/wan_t5"
require "../src/ml/video/metal_linear_tiled"
require "./support/video_allocation_counter"

private def video_linear_mps_values(value : JSON::Any) : Array(Float32)
  value.as_a.map(&.as_f.to_f32)
end

private def video_linear_mps_bf16(value : Float32) : Float32
  (value.unsafe_as(UInt32) & 0xffff0000_u32).unsafe_as(Float32)
end

private def video_linear_mps_resident(
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
    raise ArgumentError.new("Unsupported MPS projection storage dtype")
  end
end

private def video_linear_mps_dtype(name : String) : ML::Video::WeightDType
  case name
  when "f32"  then ML::Video::WeightDType::F32
  when "bf16" then ML::Video::WeightDType::BF16
  else             raise ArgumentError.new("Invalid MPS fixture dtype #{name}")
  end
end

private def video_linear_mps_assert_close(
  name : String, actual : Array(Float32), expected : Array(Float32),
) : Nil
  actual.size.should(eq(expected.size), name)
  violations = 0
  worst = 0_f64
  actual.each_with_index do |value, index|
    target = expected[index]
    ratio = (value - target).abs.to_f64 / (2e-5_f64 + 2e-5_f64 * target.abs.to_f64)
    violations += 1 unless value.finite? && target.finite? && ratio <= 1.0
    worst = Math.max(worst, ratio)
  end
  violations.should(eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}")
end

private def video_linear_mps_median(values : Array(Float64)) : Float64
  ordered = values.sort
  middle = ordered.size // 2
  if ordered.size.odd?
    ordered[middle]
  else
    (ordered[middle - 1] + ordered[middle]) / 2_f64
  end
end

private def video_linear_mps_fill_cost_weight(buffer : ML::MetalBuffer, in_dim : Int32, out_dim : Int32) : Nil
  pattern = StaticArray[0x3f80_u16, 0xbf80_u16, 0x3f00_u16, 0xbf00_u16,
    0x3e80_u16, 0xbe80_u16, 0x3f20_u16, 0xbf20_u16]
  count = in_dim * out_dim
  bytes = Bytes.new(count * sizeof(UInt16))
  count.times do |index|
    column = index // in_dim
    k = index % in_dim
    bits = pattern[(k + column * 3) % pattern.size]
    bytes[index * 2] = (bits & 0xff).to_u8
    bytes[index * 2 + 1] = (bits >> 8).to_u8
  end
  buffer.write_bytes(bytes.to_unsafe, bytes.size)
end

describe "Wan T5 persistent MPS projection" do
  it "matches the pinned MPS fixture and reuses one F32 weight stage across BF16/F32 replacements" do
    fixture_path = ENV["VIDEO_LINEAR_TILED_MPS_FIXTURE"]?
    unless fixture_path
      pending! "Set VIDEO_LINEAR_TILED_MPS_FIXTURE to video-linear-tiled-mps.json"
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
      rows = shape["rows"].as_i
      in_dim = shape["in_dim"].as_i
      out_dim = shape["out_dim"].as_i
      {rows, in_dim, out_dim}.should eq(expected_shapes[group_index])
      x_values = video_linear_mps_values(group["input"])
      w_values = video_linear_mps_values(group["weight"])
      b_values = video_linear_mps_values(group["bias"])
      cases = group["cases"].as_a
      cases.size.should eq(8)
      cases.map do |item|
        {item["input_dtype"].as_s, item["weight_dtype"].as_s, item["bias_dtype"].as_s}
      end.uniq.size.should(eq(8), "fixture group #{group_index} must cover all dtype triples")

      selected = [] of {JSON::Any, ML::Video::WeightDType, ML::Video::WeightDType}
      cases.each do |item|
        next unless item["input_dtype"].as_s == "f32"
        weight_dtype = video_linear_mps_dtype(item["weight_dtype"].as_s)
        bias_dtype = video_linear_mps_dtype(item["bias_dtype"].as_s)
        selected << {item, weight_dtype, bias_dtype}
      end
      selected.size.should(eq(4), "group #{group_index} must select F32-input/F32-output cases")
      ordered = [] of {JSON::Any, ML::Video::WeightDType, ML::Video::WeightDType}
      { {"f32", "f32"}, {"bf16", "f32"}, {"f32", "bf16"}, {"bf16", "bf16"} }.each do |weight_name, bias_name|
        matches = selected.select do |(item, _, _)|
          item["weight_dtype"].as_s == weight_name && item["bias_dtype"].as_s == bias_name
        end
        matches.size.should(eq(1), "fixture is missing dtype case #{weight_name}/#{bias_name}")
        ordered << matches.first
      end
      selected = ordered

      alternate_weight_values = w_values.map_with_index do |value, index|
        quantized = video_linear_mps_bf16(value)
        index % 3 == 0 ? -quantized : quantized
      end
      alternate_weight_values.map(&.unsafe_as(UInt32)).should_not(
        eq(w_values.map { |value| video_linear_mps_bf16(value).unsafe_as(UInt32) }),
        "weight replacement probe requires distinct BF16 tensors")

      arena = ML::Video::Arena.new
      plan = nil.as(ML::Video::WanT5MPSLinearPlan?)
      command = nil.as(ML::Metal::CommandBuffer?)
      begin
        ML::Video::WanT5.prepare!
        ML::Video::MetalLinearTiled.prepare!
        cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
          layers: 1, batches: 1, tokens: rows)
        cfg.validate!

        x = arena.upload("x", x_values)
        weight_f32 = arena.upload("weight_f32", w_values)
        weight_bf16 = video_linear_mps_resident(arena, "weight_bf16", w_values,
          ML::Video::WeightDType::BF16)
        weight_bf16_alternate = video_linear_mps_resident(arena, "weight_bf16_alternate",
          alternate_weight_values, ML::Video::WeightDType::BF16)
        bias_f32 = arena.upload("bias_f32", b_values)
        bias_bf16 = video_linear_mps_resident(arena, "bias_bf16", b_values,
          ML::Video::WeightDType::BF16)
        bias_f32_rows = arena.upload("bias_f32_rows", Array(Float32).new(rows * out_dim) do |index|
          b_values[index % out_dim]
        end)
        bias_bf16_rows = arena.upload("bias_bf16_rows", Array(Float32).new(rows * out_dim) do |index|
          video_linear_mps_bf16(b_values[index % out_dim])
        end)
        stage = arena.alloc("weight_stage_f32", in_dim * out_dim)
        projected = arena.alloc("projected", rows * out_dim)
        biased = arena.alloc("biased", rows * out_dim)
        mps_outputs = Array(ML::MetalBuffer).new(4) do |index|
          arena.alloc("mps_#{index}", rows * out_dim)
        end
        scalar_outputs = Array(ML::MetalBuffer).new(4) do |index|
          arena.alloc("scalar_#{index}", rows * out_dim)
        end
        tiled_outputs = Array(ML::MetalBuffer).new(4) do |index|
          arena.alloc("tiled_#{index}", rows * out_dim)
        end
        replacement_outputs = Array(ML::MetalBuffer).new(3) do |index|
          arena.alloc("replacement_#{index}", rows * out_dim)
        end
        plan = ML::Video::WanT5MPSLinearPlan.new(cfg, in_dim, out_dim, x, stage, projected)

        allocations = ML::MetalBuffer.video_spec_allocations
        active = ML::Metal::CommandBuffer.new
        command = active
        encoder = ML::Metal::ComputeEncoder.new(active)
        selected.each_with_index do |(item, weight_dtype, bias_dtype), case_index|
          weight = weight_dtype.bf16? ? weight_bf16 : weight_f32
          bias = bias_dtype.bf16? ? bias_bf16 : bias_f32
          bias_rows = bias_dtype.bf16? ? bias_bf16_rows : bias_f32_rows
          expected = video_linear_mps_values(item["output"])

          ML::Video::WanT5.mps_projection(encoder, x, weight, projected, rows,
            in_dim, out_dim, weight_dtype, plan.not_nil!, stage)
          ML::Video::MetalOps.add(encoder, projected, bias_rows, biased, rows * out_dim)
          ML::Video::MetalOps.copy(encoder, biased, mps_outputs[case_index], rows * out_dim)

          ML::Video::MetalOps.linear(encoder, x, weight, bias, scalar_outputs[case_index],
            rows, in_dim, out_dim, weight_dtype: weight_dtype, bias_dtype: bias_dtype)
          ML::Video::MetalLinearTiled.linear(encoder, x, weight, bias, tiled_outputs[case_index],
            rows, in_dim, out_dim, ML::Video::WeightDType::F32, weight_dtype, bias_dtype)

          # Retain the expected result in the fixture itself; the buffer is read
          # only after the command's one terminal completion boundary.
          expected.size.should eq(rows * out_dim)
        end

        # One plan and one F32 stage receive A, then a distinct BF16 tensor B,
        # then A again on the same ordered command. The saved A outputs must be
        # bit-identical, while B must visibly replace the staged matrix.
        ML::Video::WanT5.mps_projection(encoder, x, weight_bf16, projected, rows,
          in_dim, out_dim, ML::Video::WeightDType::BF16, plan.not_nil!, stage)
        ML::Video::MetalOps.copy(encoder, projected, replacement_outputs[0], rows * out_dim)
        ML::Video::WanT5.mps_projection(encoder, x, weight_bf16_alternate, projected, rows,
          in_dim, out_dim, ML::Video::WeightDType::BF16, plan.not_nil!, stage)
        ML::Video::MetalOps.copy(encoder, projected, replacement_outputs[1], rows * out_dim)
        ML::Video::WanT5.mps_projection(encoder, x, weight_bf16, projected, rows,
          in_dim, out_dim, ML::Video::WeightDType::BF16, plan.not_nil!, stage)
        ML::Video::MetalOps.copy(encoder, projected, replacement_outputs[2], rows * out_dim)
        encoder.end_encoding
        ML::MetalBuffer.video_spec_allocations.should(eq(allocations),
          "MPS projection encode must not allocate Metal buffers")
        active.commit
        active.wait

        selected.each_with_index do |(item, _weight_dtype, _bias_dtype), case_index|
          expected = video_linear_mps_values(item["output"])
          video_linear_mps_assert_close("MPS fixture group #{group_index} case #{case_index}",
            mps_outputs[case_index].read(rows * out_dim), expected)
          video_linear_mps_assert_close("scalar fixture group #{group_index} case #{case_index}",
            scalar_outputs[case_index].read(rows * out_dim), expected)
          video_linear_mps_assert_close("tiled fixture group #{group_index} case #{case_index}",
            tiled_outputs[case_index].read(rows * out_dim), expected)
        end
        before_replacement = replacement_outputs[0].read(rows * out_dim)
        alternate_replacement = replacement_outputs[1].read(rows * out_dim)
        after_replacement = replacement_outputs[2].read(rows * out_dim)
        before_replacement.all?(&.finite?).should be_true
        alternate_replacement.all?(&.finite?).should be_true
        after_replacement.all?(&.finite?).should be_true
        before_replacement.map(&.unsafe_as(UInt32)).should(
          eq(after_replacement.map(&.unsafe_as(UInt32))),
          "reloading the first BF16 matrix must restore the exact output")
        before_replacement.map(&.unsafe_as(UInt32)).should_not(
          eq(alternate_replacement.map(&.unsafe_as(UInt32))),
          "the second BF16 matrix must replace the staged matrix")
      ensure
        if active = command
          active.discard unless active.committed?
        end
        plan.try(&.release)
        arena.release
      end
    end
  end

  it "profiles full-width BF16 expansion plus MPS projection against the tiled kernel" do
    unless ENV["VIDEO_LINEAR_MPS_COST_PROBE"]? == "1"
      pending! "Set VIDEO_LINEAR_MPS_COST_PROBE=1 to allocate the full-width projection probe"
    end
    rows = (ENV["VIDEO_LINEAR_MPS_COST_ROWS"]? || "64").to_i
    repeats = (ENV["VIDEO_LINEAR_MPS_COST_REPEATS"]? || "4").to_i
    budget_mib = (ENV["VIDEO_LINEAR_MPS_COST_BUDGET_MIB"]? || "1024").to_i
    in_dim, out_dim = 5120, 13_824
    unless {64, 512, 2048}.includes?(rows)
      raise ArgumentError.new("Cost probe rows must be 64, 512, or 2048")
    end
    unless repeats >= 1 && repeats <= 8
      raise ArgumentError.new("Cost probe repeats must be in 1..8")
    end
    unless budget_mib >= 512 && budget_mib <= 8192
      raise ArgumentError.new("Cost probe memory cap must be in 512..8192 MiB")
    end

    matrix_count = in_dim.to_i64 * out_dim
    output_count = rows.to_i64 * out_dim
    planned_bytes = matrix_count * 2 + matrix_count * 4 +
                    rows.to_i64 * in_dim * 4 + 3_i64 * output_count * 4 + out_dim.to_i64 * 4
    budget_bytes = budget_mib.to_i64 * 1024 * 1024
    unless planned_bytes > 0 && planned_bytes <= budget_bytes
      raise ArgumentError.new("Cost probe buffers exceed the explicit memory cap")
    end

    arena = ML::Video::Arena.new
    plan = nil.as(ML::Video::WanT5MPSLinearPlan?)
    begin
      ML::Video::WanT5.prepare!
      ML::Video::MetalLinearTiled.prepare!
      cfg = ML::Video::WanT5Config.new(vocab: 8, dim: 8, ffn_dim: 16, heads: 1,
        layers: 1, batches: 1, tokens: rows)
      x = arena.upload("x", Array(Float32).new(rows * in_dim) do |index|
        (((index * 17) % 37 - 18) * 0.03125).to_f32
      end)
      weight = arena.alloc_weight("weight_bf16", matrix_count.to_i32, ML::Video::WeightDType::BF16)
      video_linear_mps_fill_cost_weight(weight, in_dim, out_dim)
      bias = arena.upload("bias", Array(Float32).new(out_dim, 0_f32))
      stage = arena.alloc("weight_stage_f32", matrix_count.to_i32)
      projected = arena.alloc("mps_output", output_count.to_i32)
      tiled = arena.alloc("tiled_output", output_count.to_i32)
      bias_rows = arena.upload("bias_rows", Array(Float32).new(output_count.to_i32, 0_f32))
      plan = ML::Video::WanT5MPSLinearPlan.new(cfg, in_dim, out_dim, x, stage, projected)

      measure_mps = -> do
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        allocations = ML::MetalBuffer.video_spec_allocations
        ML::Video::WanT5.mps_projection(encoder, x, weight, projected, rows,
          in_dim, out_dim, ML::Video::WeightDType::BF16, plan.not_nil!, stage)
        ML::Video::MetalOps.add(encoder, projected, bias_rows, projected, output_count.to_i32)
        encoder.end_encoding
        ML::MetalBuffer.video_spec_allocations.should(eq(allocations),
          "MPS cost encode allocated Metal buffers")
        command.commit_and_wait_gpu_elapsed_seconds
      end
      measure_tiled = -> do
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        allocations = ML::MetalBuffer.video_spec_allocations
        ML::Video::MetalLinearTiled.linear_range(encoder, x, weight, bias, tiled,
          rows, in_dim, out_dim, 0, rows,
          ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16, ML::Video::WeightDType::F32)
        encoder.end_encoding
        ML::MetalBuffer.video_spec_allocations.should(eq(allocations),
          "tiled cost encode allocated Metal buffers")
        command.commit_and_wait_gpu_elapsed_seconds
      end

      mps_samples = [] of Float64
      tiled_samples = [] of Float64
      repeats.times do |sample_index|
        if sample_index.even?
          mps_samples << measure_mps.call
          tiled_samples << measure_tiled.call
        else
          tiled_samples << measure_tiled.call
          mps_samples << measure_mps.call
        end
      end
      video_linear_mps_assert_close("full-width MPS/tiled", projected.read(output_count.to_i32),
        tiled.read(output_count.to_i32))

      flops = 2_i64 * rows * in_dim * out_dim
      timing = String.build do |io|
        io << "video_linear_mps_cost rows=" << rows << " in=" << in_dim << " out=" << out_dim
        io << " resident_bf16_weight_bytes=" << weight.size
        io << " reusable_f32_weight_stage_bytes=" << stage.size
        io << " explicit_buffer_bytes=" << planned_bytes << " budget_mib=" << budget_mib
        io << " f32_flops=" << flops << " repeats=" << repeats
        io << " mps_samples_seconds=" << mps_samples.join(",")
        io << " tiled_samples_seconds=" << tiled_samples.join(",")
        io << " mps_gpu_seconds=" << video_linear_mps_median(mps_samples)
        io << " tiled_gpu_seconds=" << video_linear_mps_median(tiled_samples)
        io << " tiled_over_mps=" << video_linear_mps_median(tiled_samples) / video_linear_mps_median(mps_samples)
        io << " mps_timing_includes_bf16_to_f32_expansion=true"
        io << " mps_timing_includes_bias_add=true"
      end
      puts timing
    ensure
      plan.try(&.release)
      arena.release
    end
  end
end
