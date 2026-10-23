require "spec"
require "../src/ml/video/q4k_delta_linear"
require "../src/ml/video/metal_linear_mps"
require "./support/video_allocation_counter"

alias Q4KDeltaLinear = ML::Video::Q4KDeltaLinear

private def q4kl_store_u16(bytes : Bytes, offset : Int32, value : UInt16) : Nil
  bytes[offset] = (value & 0xff_u16).to_u8
  bytes[offset + 1] = (value >> 8).to_u8
end

private def q4kl_load_f32(buffer : ML::MetalBuffer, index : Int32) : Float32
  buffer.contents.as(Float32*)[index]
end

private def q4kl_test_payload(in_dim : Int32, out_dim : Int32) : Bytes
  blocks_per_row = in_dim // 256
  bytes = Bytes.new(out_dim * blocks_per_row * 144, 0_u8)
  out_dim.times do |column|
    blocks_per_row.times do |block|
      offset = (column * blocks_per_row + block) * 144
      q4kl_store_u16(bytes, offset, 0x2e66_u16)
      q4kl_store_u16(bytes, offset + 2, 0_u16)
      12.times { |index| bytes[offset + 4 + index] = 1_u8 }
      128.times do |index|
        low = (index * 7 + column * 3 + block) % 16
        high = (index * 11 + column + block * 5) % 16
        bytes[offset + 16 + index] = (low | (high << 4)).to_u8
      end
    end
  end
  bytes
end

private def q4kl_positions(patches : Int32, block : Int32, rank : Int32) : UInt8
  case patches
  when 1
    block.even? ? 0_u8 : 255_u8
  when 4
    {0_u8, 64_u8, 192_u8, 255_u8}[rank]
  when 8
    {0_u8, 32_u8, 64_u8, 96_u8, 128_u8, 160_u8, 224_u8, 255_u8}[rank]
  else
    raise ArgumentError.new("No correction positions for patch count #{patches}")
  end
end

private def q4kl_test_sidecar(in_dim : Int32, out_dim : Int32, patches : Int32) : Bytes
  return Bytes[0xa5_u8] if patches == 0
  blocks = out_dim * (in_dim // 256)
  bytes = Bytes.new(blocks * patches * 3, 0_u8)
  blocks.times do |block|
    patches.times do |rank|
      offset = (block * patches + rank) * 3
      bytes[offset] = q4kl_positions(patches, block, rank)
      delta = rank.odd? ? 0xbc00_u16 : 0x3a00_u16
      q4kl_store_u16(bytes, offset + 1, delta)
    end
  end
  bytes
end

private def q4kl_assert_close(name : String, actual : ML::MetalBuffer,
                              expected : ML::MetalBuffer, count : Int32) : Nil
  actual_values = actual.contents.as(Float32*)
  expected_values = expected.contents.as(Float32*)
  violations = 0
  worst = 0_f64
  count.times do |index|
    value = actual_values[index]
    target = expected_values[index]
    ratio = (value - target).abs.to_f64 /
            (2e-5_f64 + 2e-5_f64 * target.abs.to_f64)
    violations += 1 unless value.finite? && target.finite? && ratio <= 1.0
    worst = Math.max(worst, ratio)
  end
  violations.should(eq(0), "#{name}: violations=#{violations}, worst normalized error=#{worst}")
end

private def q4kl_release(buffers : Array(ML::MetalBuffer)) : Nil
  buffers.each(&.release)
end

private def q4kl_run_gemm_case(rows : Int32, in_dim : Int32, patches : Int32) : Nil
  out_dim = 35
  weight_count = in_dim * out_dim
  output_count = rows * out_dim
  input_values = Array(Float32).new(rows * in_dim) do |index|
    ((index * 17 % 101).to_f32 - 50_f32) / 97_f32
  end
  bias_values = Array(Float32).new(out_dim) do |column|
    ((column * 13 % 17).to_f32 - 8_f32) / 113_f32
  end
  payload_bytes = q4kl_test_payload(in_dim, out_dim)
  sidecar_bytes = q4kl_test_sidecar(in_dim, out_dim, patches)
  buffers = [] of ML::MetalBuffer
  plan = nil.as(ML::Video::WanT5MPSLinearPlan?)
  command = nil.as(ML::Metal::CommandBuffer?)
  begin
    input = ML::MetalBuffer.new(input_values.size.to_i64 * sizeof(Float32))
    buffers << input
    input.write(input_values)
    payload = ML::MetalBuffer.new(payload_bytes.size.to_i64)
    buffers << payload
    payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
    sidecar = ML::MetalBuffer.new(sidecar_bytes.size.to_i64)
    buffers << sidecar
    sidecar.write_bytes(sidecar_bytes.to_unsafe, sidecar_bytes.size)
    bias = ML::MetalBuffer.new(bias_values.size.to_i64 * sizeof(Float32))
    buffers << bias
    bias.write(bias_values)
    status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    buffers << status
    status.contents.as(UInt32*).value = 0_u32

    decoded = ML::MetalBuffer.new(weight_count.to_i64 * sizeof(Float32))
    buffers << decoded
    fused = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32))
    buffers << fused
    dense = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32))
    buffers << dense
    scalar = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32))
    buffers << scalar
    mps_stage = ML::MetalBuffer.new(weight_count.to_i64 * sizeof(Float32))
    buffers << mps_stage
    mps = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32))
    buffers << mps

    cfg = ML::Video::WanT5Config.new(vocab: 8, dim: in_dim, ffn_dim: out_dim,
      heads: in_dim // 256, layers: 1, batches: 1, tokens: rows)
    cfg.validate!
    plan = ML::Video::WanT5MPSLinearPlan.new(cfg, in_dim, out_dim,
      input, mps_stage, mps)

    active = ML::Metal::CommandBuffer.new
    command = active
    encoder = ML::Metal::ComputeEncoder.new(active)
    allocation_count = ML::MetalBuffer.video_spec_allocations
    Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, decoded, status,
      weight_count, patches)
    Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused, status,
      rows, in_dim, out_dim, patches)
    Q4KDeltaLinear.dense_linear(encoder, input, decoded, bias, dense,
      rows, in_dim, out_dim)
    ML::Video::MetalOps.linear(encoder, input, decoded, bias, scalar,
      rows, in_dim, out_dim)
    ML::MetalBuffer.video_spec_allocations.should(eq(allocation_count),
      "Q4_K fused, dense, and probe encodes must not allocate Metal buffers")
    ML::Video::MetalLinearMPS.project(encoder, input, decoded, bias, mps,
      rows, in_dim, out_dim, ML::Video::WeightDType::F32,
      ML::Video::WeightDType::F32, plan.not_nil!, mps_stage)
    encoder.end_encoding
    active.commit_and_wait
    active.completed_successfully?.should be_true
    status.contents.as(UInt32*).value.should eq(0_u32)

    q4kl_assert_close("fused rows=#{rows} K=#{in_dim} patches=#{patches}", fused, scalar, output_count)
    q4kl_assert_close("dense rows=#{rows} K=#{in_dim} patches=#{patches}", dense, scalar, output_count)
    q4kl_assert_close("MPS rows=#{rows} K=#{in_dim} patches=#{patches}", mps, scalar, output_count)
  ensure
    plan.try(&.release)
    if active = command
      active.discard unless active.committed?
    end
    q4kl_release(buffers)
  end
end

describe "Q4_K sparse-delta fused linear" do
  it "validates GEMM geometry and the cached four-simdgroup pipeline contract" do
    {1, 19, 65}.each do |rows|
      {256, 512}.each do |in_dim|
        Q4KDeltaLinear.checked_geometry(rows, in_dim, 35).should eq({rows, in_dim, 35})
      end
    end
    {0, -1}.each do |rows|
      expect_raises(ArgumentError) { Q4KDeltaLinear.checked_geometry(rows, 256, 35) }
    end
    {0, 255, 257}.each do |in_dim|
      expect_raises(ArgumentError) { Q4KDeltaLinear.checked_geometry(1, in_dim, 35) }
    end
    {0, -1}.each do |out_dim|
      expect_raises(ArgumentError) { Q4KDeltaLinear.checked_geometry(1, 256, out_dim) }
    end
    expect_raises(ArgumentError) do
      Q4KDeltaLinear.checked_geometry(Int32::MAX, 256, 1)
    end
    expect_raises(ArgumentError) do
      Q4KDeltaLinear.checked_geometry(1, 256, Int32::MAX)
    end
    {2, 3, 9}.each do |patches|
      expect_raises(ArgumentError) { Q4KDeltaLinear.sidecar_bytes(256, patches) }
    end
    Q4KDeltaLinear.sidecar_bytes(256, 0).should eq(1_i64)
    {0, 1, 4, 8}.each do |patches|
      Q4KDeltaLinear.sidecar_bytes(512, patches).should be > 0
    end

    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    Q4KDeltaLinear.linear_pipeline.thread_execution_width.should eq(32)
    Q4KDeltaLinear.linear_pipeline.max_total_threads_per_threadgroup.should be >= 128
    Q4KDeltaLinear.static_threadgroup_memory_bytes.should be <= 32768
  end

  it "matches the separately decoded scalar FMA oracle and MPS across M/N/K tails and patch quotas" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!
    ML::Video::MetalLinearMPS.prepare!

    # These cover every admitted M tail, N=35, both supported K depths, all
    # quotas, and local correction positions 0/255. K=0 patches still binds
    # the required one-byte ignored sidecar.
    [
      {1, 256, 0},
      {19, 256, 1},
      {65, 256, 4},
      {1, 512, 8},
      {19, 512, 0},
      {65, 512, 1},
    ].each do |rows, in_dim, patches|
      q4kl_run_gemm_case(rows, in_dim, patches)
    end
  end

  it "preserves exact decoded F32 coefficient rounding" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!

    payload_bytes = Bytes.new(144, 0_u8)
    q4kl_store_u16(payload_bytes, 0, 0x2e66_u16)
    payload_bytes[4] = 1_u8
    payload_bytes[16] = 3_u8
    sidecar_bytes = Bytes[0_u8, 0x9a_u8, 0x39_u8]
    payload = ML::MetalBuffer.new(144_i64)
    sidecar = ML::MetalBuffer.new(3_i64)
    decoded = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
      sidecar.write_bytes(sidecar_bytes.to_unsafe, sidecar_bytes.size)
      status.contents.as(UInt32*).value = 0_u32
      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, decoded, status, 256, 1)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should eq(0_u32)
      q4kl_load_f32(decoded, 0).should eq(1.0001220703125_f32)
      q4kl_load_f32(decoded, 1).should eq(0_f32)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      {payload, sidecar, decoded, status}.each(&.release)
    end
  end

  it "preserves signed-zero decode semantics for unmatched correction records" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!

    payload_bytes = Bytes.new(144, 0_u8)
    q4kl_store_u16(payload_bytes, 0, 0x8000_u16)
    payload_bytes[4] = 1_u8
    payload = ML::MetalBuffer.new(144_i64)
    base_sidecar = ML::MetalBuffer.new(1_i64)
    corrected_sidecar = ML::MetalBuffer.new(3_i64)
    base_decoded = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    corrected_decoded = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    base_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    corrected_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
      base_sidecar.contents.as(UInt8*).value = 0xa5_u8
      corrected_sidecar.contents.as(UInt8*)[0] = 255_u8
      corrected_sidecar.contents.as(UInt8*)[1] = 0_u8
      corrected_sidecar.contents.as(UInt8*)[2] = 0_u8
      base_status.contents.as(UInt32*).value = 0_u32
      corrected_status.contents.as(UInt32*).value = 0_u32

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDeltaLinear.decode_probe(encoder, payload, base_sidecar, base_decoded,
        base_status, 256, 0)
      Q4KDeltaLinear.decode_probe(encoder, payload, corrected_sidecar,
        corrected_decoded, corrected_status, 256, 1)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      base_status.contents.as(UInt32*).value.should eq(0_u32)
      corrected_status.contents.as(UInt32*).value.should eq(0_u32)
      base_decoded.contents.as(UInt32*)[0].should eq(0x8000_0000_u32)
      corrected_decoded.contents.as(UInt32*)[0].should eq(0_u32)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      {payload, base_sidecar, corrected_sidecar, base_decoded,
       corrected_decoded, base_status, corrected_status}.each(&.release)
    end
  end

  it "isolates duplicate, NaN, and Inf lists per output-channel K block in fused GEMM" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!

    rows = 1
    in_dim = 512
    out_dim = 2
    input_values = Array(Float32).new(in_dim) do |index|
      ((index * 17 % 101).to_f32 - 50_f32) / 97_f32
    end
    bias_values = [0.125_f32, -0.0625_f32]
    payload_bytes = q4kl_test_payload(in_dim, out_dim)
    valid_bytes = q4kl_test_sidecar(in_dim, out_dim, 4)
    duplicate_bytes = valid_bytes.dup
    duplicate_bytes[3] = duplicate_bytes[0]
    nan_bytes = valid_bytes.dup
    nan_bytes[4] = 0x00_u8
    nan_bytes[5] = 0x7e_u8
    inf_bytes = valid_bytes.dup
    inf_bytes[4] = 0x00_u8
    inf_bytes[5] = 0x7c_u8

    { {"duplicate", duplicate_bytes}, {"NaN", nan_bytes}, {"Inf", inf_bytes} }.each do |label, malformed_bytes|
      buffers = [] of ML::MetalBuffer
      command = nil.as(ML::Metal::CommandBuffer?)
      begin
        input = ML::MetalBuffer.new(in_dim.to_i64 * sizeof(Float32)); buffers << input
        input.write(input_values)
        payload = ML::MetalBuffer.new(payload_bytes.size.to_i64); buffers << payload
        payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
        sidecar = ML::MetalBuffer.new(malformed_bytes.size.to_i64); buffers << sidecar
        sidecar.write_bytes(malformed_bytes.to_unsafe, malformed_bytes.size)
        bias = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << bias
        bias.write(bias_values)
        decoded = ML::MetalBuffer.new((out_dim * in_dim).to_i64 * sizeof(Float32)); buffers << decoded
        scalar = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << scalar
        fused = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << fused
        fused.contents.as(Float32*)[0] = 1234_f32
        fused.contents.as(Float32*)[1] = -1234_f32
        probe_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << probe_status
        fused_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << fused_status
        probe_status.contents.as(UInt32*).value = 0_u32
        fused_status.contents.as(UInt32*).value = 0_u32

        active = ML::Metal::CommandBuffer.new
        command = active
        encoder = ML::Metal::ComputeEncoder.new(active)
        Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, decoded,
          probe_status, out_dim * in_dim, 4)
        ML::Video::MetalOps.linear(encoder, input, decoded, bias, scalar,
          rows, in_dim, out_dim)
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused,
          fused_status, rows, in_dim, out_dim, 4)
        encoder.end_encoding
        active.commit_and_wait
        active.completed_successfully?.should be_true
        probe_status.contents.as(UInt32*).value.should(eq(1_u32), "#{label} probe status")
        fused_status.contents.as(UInt32*).value.should(eq(1_u32), "#{label} fused status")

        256.times do |index|
          q4kl_load_f32(decoded, index).should(eq(0_f32), "#{label} malformed channel 0 block")
        end
        256.upto(out_dim * in_dim - 1) do |index|
          q4kl_load_f32(decoded, index).finite?.should be_true
        end
        q4kl_load_f32(decoded, 256).should_not eq(0_f32) # channel 0, K block 1
        q4kl_load_f32(decoded, 512).should_not eq(0_f32) # channel 1, K block 0
        q4kl_assert_close("#{label} malformed fused output", fused, scalar, out_dim)
      ensure
        if active = command
          active.discard unless active.committed?
        end
        q4kl_release(buffers)
      end
    end

    # A pre-existing sticky error must not short-circuit valid output work.
    buffers = [] of ML::MetalBuffer
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      input = ML::MetalBuffer.new(in_dim.to_i64 * sizeof(Float32)); buffers << input
      input.write(input_values)
      payload = ML::MetalBuffer.new(payload_bytes.size.to_i64); buffers << payload
      payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
      sidecar = ML::MetalBuffer.new(valid_bytes.size.to_i64); buffers << sidecar
      sidecar.write_bytes(valid_bytes.to_unsafe, valid_bytes.size)
      bias = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << bias
      bias.write(bias_values)
      decoded = ML::MetalBuffer.new((out_dim * in_dim).to_i64 * sizeof(Float32)); buffers << decoded
      scalar = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << scalar
      fused = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << fused
      fused.contents.as(Float32*)[0] = 1234_f32
      fused.contents.as(Float32*)[1] = -1234_f32
      decode_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << decode_status
      fused_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << fused_status
      decode_status.contents.as(UInt32*).value = 0_u32
      fused_status.contents.as(UInt32*).value = 0x40_u32

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, decoded,
        decode_status, out_dim * in_dim, 4)
      ML::Video::MetalOps.linear(encoder, input, decoded, bias, scalar,
        rows, in_dim, out_dim)
      Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused,
        fused_status, rows, in_dim, out_dim, 4)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      decode_status.contents.as(UInt32*).value.should eq(0_u32)
      fused_status.contents.as(UInt32*).value.should eq(0x40_u32)
      q4kl_assert_close("pre-existing sticky status still computes", fused, scalar, out_dim)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      q4kl_release(buffers)
    end
  end

  it "rejects aliases and undersized buffers before changing poison output or status" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    input = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    payload = ML::MetalBuffer.new(144_i64)
    sidecar = ML::MetalBuffer.new(1_i64)
    bias = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    output = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    tiny_input = ML::MetalBuffer.new(255_i64 * sizeof(Float32))
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      output.contents.as(Float32*).value = 19.25_f32
      status.contents.as(UInt32*).value = 0x6a10_0020_u32
      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, tiny_input, payload, sidecar, bias, output,
          status, 1, 256, 1, 0)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, input,
          status, 1, 256, 1, 0)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, output, output,
          status, 1, 256, 1, 0)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, payload, bias, output,
          status, 1, 256, 1, 0)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, bias, bias, output,
          status, 1, 256, 1, 0)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, output,
          payload, 1, 256, 1, 0)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, output,
          output, 1, 256, 1, 0)
      end
      output.contents.as(Float32*).value.should eq(19.25_f32)
      status.contents.as(UInt32*).value.should eq(0x6a10_0020_u32)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      output.contents.as(Float32*).value.should eq(19.25_f32)
      status.contents.as(UInt32*).value.should eq(0x6a10_0020_u32)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      {input, payload, sidecar, bias, output, status, tiny_input}.each(&.release)
    end
  end
end
