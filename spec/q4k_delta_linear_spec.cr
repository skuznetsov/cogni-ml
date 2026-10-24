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

private def q4kl_run_gemm_case(rows : Int32, in_dim : Int32, patches : Int32,
                               variant : Q4KDeltaLinear::GemmVariant = Q4KDeltaLinear::GemmVariant::Legacy,
                               dense_variant : Q4KDeltaLinear::GemmVariant? = nil,
                               out_dim : Int32 = 35) : Nil
  selected_dense_variant = dense_variant || variant
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
    output_guard_count = 16
    fused = ML::MetalBuffer.new((output_count + output_guard_count).to_i64 * sizeof(Float32))
    buffers << fused
    fused_values = fused.contents.as(Float32*)
    (output_count + output_guard_count).times do |index|
      fused_values[index] = 765_432.125_f32
    end
    dense = ML::MetalBuffer.new((output_count + output_guard_count).to_i64 * sizeof(Float32))
    buffers << dense
    dense_values = dense.contents.as(Float32*)
    (output_count + output_guard_count).times do |index|
      dense_values[index] = -1_234_567.25_f32
    end
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
    if selected_dense_variant == Q4KDeltaLinear::GemmVariant::B64N32S4DeviceF32
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused,
          status, rows, in_dim, out_dim, patches, selected_dense_variant)
      end
    end
    Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, decoded, status,
      weight_count, patches)
    Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused, status,
      rows, in_dim, out_dim, patches, variant)
    Q4KDeltaLinear.dense_linear(encoder, input, decoded, bias, dense,
      rows, in_dim, out_dim, selected_dense_variant)
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
    output_guard_count.times do |index|
      dense_values[output_count + index].should(eq(-1_234_567.25_f32),
        "dense output guard changed at #{index} for rows=#{rows} N=#{out_dim}")
      fused_values[output_count + index].should(eq(765_432.125_f32),
        "fused output guard changed at #{index} for rows=#{rows} N=#{out_dim}")
    end
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
    Q4KDeltaLinear.variant_name.should eq("legacy")
    Q4KDeltaLinear.variant_name(Q4KDeltaLinear::GemmVariant::B128N32S4K32).should eq("candidate-b128n32-sg4-tk32")
    Q4KDeltaLinear.variant_name(Q4KDeltaLinear::GemmVariant::B64N32S4K32BT).should eq("candidate-b64n32-sg4-tk32-bt")
    Q4KDeltaLinear.variant_name(Q4KDeltaLinear::GemmVariant::B64N32S4K64BT).should eq("candidate-b64n32-sg4-tk64-bt")
    device_variant = Q4KDeltaLinear::GemmVariant::B64N32S4DeviceF32
    Q4KDeltaLinear.variant_name(device_variant).should eq("candidate-b64n32-sg4-device-f32")
    Q4KDeltaLinear.tile_m(device_variant).should eq(64)
    Q4KDeltaLinear.tile_n(device_variant).should eq(32)
    Q4KDeltaLinear.tile_k(device_variant).should eq(32)
    Q4KDeltaLinear.threadgroup_threads(device_variant).should eq(128)
    expect_raises(ArgumentError) { Q4KDeltaLinear.linear_pipeline(device_variant) }
    device_q4k_variant = Q4KDeltaLinear::GemmVariant::B64N32S4DeviceQ4K
    Q4KDeltaLinear.variant_name(device_q4k_variant).should eq("candidate-b64n32-sg4-device-q4k")
    Q4KDeltaLinear.tile_m(device_q4k_variant).should eq(64)
    Q4KDeltaLinear.tile_n(device_q4k_variant).should eq(32)
    Q4KDeltaLinear.tile_k(device_q4k_variant).should eq(32)
    Q4KDeltaLinear.threadgroup_threads(device_q4k_variant).should eq(128)
    {
      {Q4KDeltaLinear::GemmVariant::Legacy, 64, 32, 32},
      {Q4KDeltaLinear::GemmVariant::B128N32S4K32, 128, 32, 32},
      {Q4KDeltaLinear::GemmVariant::B64N32S4K32BT, 64, 32, 32},
      {Q4KDeltaLinear::GemmVariant::B64N32S4K64BT, 64, 32, 64},
      {device_q4k_variant, 64, 32, 32},
    }.each do |variant, tile_m, tile_n, tile_k|
      Q4KDeltaLinear.tile_m(variant).should eq(tile_m)
      Q4KDeltaLinear.tile_n(variant).should eq(tile_n)
      Q4KDeltaLinear.tile_k(variant).should eq(tile_k)
      Q4KDeltaLinear.threadgroup_threads(variant).should eq(128)
    end

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
    Q4KDeltaLinear.dense_dispatch_partition(4, 256, 35).should eq({
      direct_rows: 0, direct_columns: 32, direct_groups_x: 1, direct_groups_y: 0,
      right_tail_groups: 0, bottom_tail_groups: 2,
      direct_dispatches: 0, staged_tail_dispatches: 1,
    })
    Q4KDeltaLinear.dense_dispatch_partition(4_480, 256, 256).should eq({
      direct_rows: 4_480, direct_columns: 256, direct_groups_x: 8, direct_groups_y: 70,
      right_tail_groups: 0, bottom_tail_groups: 0,
      direct_dispatches: 1, staged_tail_dispatches: 0,
    })
    Q4KDeltaLinear.dense_dispatch_partition(65, 256, 35).should eq({
      direct_rows: 64, direct_columns: 32, direct_groups_x: 1, direct_groups_y: 1,
      right_tail_groups: 1, bottom_tail_groups: 2,
      direct_dispatches: 1, staged_tail_dispatches: 2,
    })
    Q4KDeltaLinear.dense_dispatch_partition(65, 5120, 35,
      device_q4k_variant).should eq({
      direct_rows: 64, direct_columns: 32, direct_groups_x: 1, direct_groups_y: 1,
      right_tail_groups: 1, bottom_tail_groups: 2,
      direct_dispatches: 1, staged_tail_dispatches: 2,
    })
    Q4KDeltaLinear.linear_dispatch_partition(4, 256, 13_824,
      device_q4k_variant).should eq({
      direct_rows: 0, direct_columns: 13_824, direct_groups_x: 432, direct_groups_y: 0,
      right_tail_groups: 0, bottom_tail_groups: 432,
      direct_dispatches: 0, staged_tail_dispatches: 1,
    })
    Q4KDeltaLinear.linear_dispatch_partition(4_480, 256, 13_824,
      device_q4k_variant).should eq({
      direct_rows: 4_480, direct_columns: 13_824, direct_groups_x: 432, direct_groups_y: 70,
      right_tail_groups: 0, bottom_tail_groups: 0,
      direct_dispatches: 1, staged_tail_dispatches: 0,
    })
    Q4KDeltaLinear.linear_dispatch_partition(65, 5120, 35,
      device_q4k_variant).should eq({
      direct_rows: 64, direct_columns: 32, direct_groups_x: 1, direct_groups_y: 1,
      right_tail_groups: 1, bottom_tail_groups: 2,
      direct_dispatches: 1, staged_tail_dispatches: 2,
    })
    expect_raises(ArgumentError) do
      Q4KDeltaLinear.dense_dispatch_partition(1, 255, 35)
    end
    expect_raises(ArgumentError) do
      Q4KDeltaLinear.dense_dispatch_partition(1, 256, 35,
        Q4KDeltaLinear::GemmVariant::Legacy)
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
    {
      Q4KDeltaLinear::GemmVariant::Legacy,
      Q4KDeltaLinear::GemmVariant::B128N32S4K32,
      Q4KDeltaLinear::GemmVariant::B64N32S4K32BT,
      Q4KDeltaLinear::GemmVariant::B64N32S4K64BT,
      device_q4k_variant,
    }.each do |variant|
      Q4KDeltaLinear.linear_pipeline(variant).thread_execution_width.should eq(32)
      Q4KDeltaLinear.linear_pipeline(variant).max_total_threads_per_threadgroup.should be >= 128
      Q4KDeltaLinear.dense_pipeline(variant).thread_execution_width.should eq(32)
      Q4KDeltaLinear.dense_pipeline(variant).max_total_threads_per_threadgroup.should be >= 128
      Q4KDeltaLinear.static_threadgroup_memory_bytes(variant).should be <= 32768
      Q4KDeltaLinear.dense_static_threadgroup_memory_bytes(variant).should be <= 32768
      if variant == Q4KDeltaLinear::GemmVariant::B64N32S4K64BT
        Q4KDeltaLinear.static_threadgroup_memory_bytes(variant).should eq(26752_i64)
        Q4KDeltaLinear.dense_static_threadgroup_memory_bytes(variant).should eq(24576_i64)
      end
      if variant == device_q4k_variant
        Q4KDeltaLinear.static_threadgroup_memory_bytes(variant).should eq(10_368_i64)
        Q4KDeltaLinear.dense_pipeline(variant).name.should eq(
          "video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_device_f32")
        Q4KDeltaLinear.dense_static_threadgroup_memory_bytes(variant).should eq(8_192_i64)
        Q4KDeltaLinear.dense_staged_tail_pipeline(variant).name.should eq(
          "video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_device_f32_tail")
        Q4KDeltaLinear.dense_staged_tail_static_threadgroup_memory_bytes(variant).should eq(12_288_i64)
      end
    end
    direct = Q4KDeltaLinear.dense_pipeline(device_variant)
    direct.name.should eq("video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_device_f32")
    direct.thread_execution_width.should eq(32)
    direct.max_total_threads_per_threadgroup.should be >= 128
    Q4KDeltaLinear.dense_static_threadgroup_memory_bytes(device_variant).should eq(8_192_i64)
    tail = Q4KDeltaLinear.dense_staged_tail_pipeline(device_variant)
    tail.name.should eq("video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_device_f32_tail")
    tail.thread_execution_width.should eq(32)
    tail.max_total_threads_per_threadgroup.should be >= 128
    Q4KDeltaLinear.dense_staged_tail_static_threadgroup_memory_bytes(device_variant).should eq(12_288_i64)

    packed = Q4KDeltaLinear.linear_pipeline(device_q4k_variant)
    packed.name.should eq("video_q4k_delta_linear_f32_candidate_b64n32_sg4_device_q4k")
    packed.thread_execution_width.should eq(32)
    packed.max_total_threads_per_threadgroup.should be >= 128
    Q4KDeltaLinear.linear_staged_tail_pipeline(device_q4k_variant).name.should eq(
      "video_q4k_delta_linear_f32_candidate_b64n32_sg4_device_q4k_tail")
    Q4KDeltaLinear.linear_staged_tail_static_threadgroup_memory_bytes(device_q4k_variant).should be <= 32_768_i64
  end

  it "runs the opt-in 128x32 schedule across row-tile boundaries and Q4_K patch quotas" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!
    ML::Video::MetalLinearMPS.prepare!

    # N=35 exercises the column tail. Rows straddle the candidate M=128 tile,
    # and K=256/512 covers one and two Q4_K blocks with every patch quota.
    [
      {127, 256, 0},
      {128, 256, 1},
      {129, 512, 4},
      {257, 512, 8},
    ].each do |rows, in_dim, patches|
      q4kl_run_gemm_case(rows, in_dim, patches,
        Q4KDeltaLinear::GemmVariant::B128N32S4K32)
    end
  end

  it "runs the transposed-B 64x32 schedule across row tails and Q4_K patch quotas" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!
    ML::Video::MetalLinearMPS.prepare!

    # N=35 exercises the column tail. Rows cover both sides of the M=64 tile
    # and a second tile; K and patch quotas match the shared Q4_K path.
    [
      {63, 256, 0},
      {64, 256, 1},
      {65, 512, 4},
      {129, 512, 8},
    ].each do |rows, in_dim, patches|
      q4kl_run_gemm_case(rows, in_dim, patches,
        Q4KDeltaLinear::GemmVariant::B64N32S4K32BT)
    end
  end

  it "runs the transposed-B TK64 schedule across row tails and Q4_K patch quotas" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!
    ML::Video::MetalLinearMPS.prepare!

    # N=35 exercises the column tail. Rows straddle the M=64 tile and cross
    # into a second tile; K and patches cover one/two blocks and every quota.
    [
      {63, 256, 0},
      {64, 256, 1},
      {65, 512, 4},
      {129, 512, 8},
    ].each do |rows, in_dim, patches|
      q4kl_run_gemm_case(rows, in_dim, patches,
        Q4KDeltaLinear::GemmVariant::B64N32S4K64BT)
    end
  end

  it "runs the direct-device F32 core and staged fringes with poisoned outputs and canaries" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!
    ML::Video::MetalLinearMPS.prepare!

    device_variant = Q4KDeltaLinear::GemmVariant::B64N32S4DeviceF32
    # The first case is one direct dispatch. The second is tail-only. The
    # remaining cases combine direct and staged groups, including M/N values
    # that are multiples of eight but not of the 64x32 tile.
    [
      {64, 256, 0, 32},
      {4, 256, 1, 35},
      {65, 512, 4, 35},
      {72, 256, 8, 40},
      {129, 512, 0, 72},
    ].each do |rows, in_dim, patches, out_dim|
      q4kl_run_gemm_case(rows, in_dim, patches,
        Q4KDeltaLinear::GemmVariant::Legacy, device_variant, out_dim)
    end
  end

  it "runs direct Q4_K tiles and staged fringes for every sparse-delta quota" do
    pending! "Requires macOS Metal Performance Shaders" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!
    ML::Video::MetalLinearMPS.prepare!

    variant = Q4KDeltaLinear::GemmVariant::B64N32S4DeviceQ4K
    # Full core, tail-only, and mixed right/bottom fringes. Rows and columns
    # include multiples of eight that do not align to the 64x32 direct tile.
    [
      {64, 256, 0, 32},
      {4, 256, 1, 35},
      {65, 512, 4, 35},
      {72, 256, 8, 40},
      {129, 512, 1, 72},
    ].each do |rows, in_dim, patches, out_dim|
      q4kl_run_gemm_case(rows, in_dim, patches, variant, nil, out_dim)
    end
  end

  it "round-trips all 64 coordinates in each SIMD group and materializes both Q4_K N halves at K=248/256" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    coordinate_map = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    coordinate_command = nil.as(ML::Metal::CommandBuffer?)
    begin
      active = ML::Metal::CommandBuffer.new
      coordinate_command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDeltaLinear.coordinate_mapping_probe(encoder, coordinate_map)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      mapped = coordinate_map.contents.as(Float32*)
      4.times do |simdgroup|
        64.times do |coordinate|
          expected = (coordinate // 8 * 100 + coordinate % 8).to_f32
          index = simdgroup * 64 + coordinate
          mapped[index].should(eq(expected), "coordinate round trip SIMD group #{simdgroup} at #{coordinate}")
        end
      end
    ensure
      if active = coordinate_command
        active.discard unless active.committed?
      end
      coordinate_map.release
    end

    {0, 1, 4, 8}.each do |patches|
      in_dim = 512
      out_dim = 32
      weight_count = in_dim * out_dim
      payload_bytes = q4kl_test_payload(in_dim, out_dim)
      sidecar_bytes = q4kl_test_sidecar(in_dim, out_dim, patches)
      buffers = [] of ML::MetalBuffer
      command = nil.as(ML::Metal::CommandBuffer?)
      begin
        payload = ML::MetalBuffer.new(payload_bytes.size.to_i64); buffers << payload
        payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
        sidecar = ML::MetalBuffer.new(sidecar_bytes.size.to_i64); buffers << sidecar
        sidecar.write_bytes(sidecar_bytes.to_unsafe, sidecar_bytes.size)
        expected = ML::MetalBuffer.new(weight_count.to_i64 * sizeof(Float32)); buffers << expected
        actual = ML::MetalBuffer.new(1024_i64 * sizeof(Float32)); buffers << actual
        status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << status
        status.contents.as(UInt32*).value = 0_u32

        active = ML::Metal::CommandBuffer.new
        command = active
        encoder = ML::Metal::ComputeEncoder.new(active)
        Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, expected,
          status, weight_count, patches)
        Q4KDeltaLinear.q4_fragment_probe(encoder, payload, sidecar, actual,
          status, patches)
        encoder.end_encoding
        active.commit_and_wait
        active.completed_successfully?.should be_true
        status.contents.as(UInt32*).value.should eq(0_u32)
        expected_bits = expected.contents.as(UInt32*)
        actual_bits = actual.contents.as(UInt32*)
        [248, 256].each_with_index do |k_origin, origin_index|
          4.times do |simdgroup|
            2.times do |cb|
              8.times do |local_k|
                8.times do |local_n|
                  channel = (simdgroup // 2) * 16 + cb * 8 + local_n
                  index = origin_index * 512 + simdgroup * 128 + cb * 64 + local_k * 8 + local_n
                  expected_index = channel * in_dim + k_origin + local_k
                  actual_bits[index].should(eq(expected_bits[expected_index]),
                    "Q4_K fragment patches=#{patches} SIMD group=#{simdgroup} cb=#{cb} channel=#{channel} K=#{k_origin + local_k}")
                end
              end
            end
          end
        end
      ensure
        if active = command
          active.discard unless active.committed?
        end
        q4kl_release(buffers)
      end
    end
  end

  it "preserves signed-zero coefficient bits through the direct fragment loader" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!

    in_dim = 512
    out_dim = 32
    weight_count = in_dim * out_dim
    rows = 64
    output_count = rows * out_dim
    input_values = Array(Float32).new(rows * in_dim, 1_f32)
    bias_values = Array(Float32).new(out_dim, 0_f32)
    payload_bytes = Bytes.new(out_dim * (in_dim // 256) * 144, 0_u8)
    (out_dim * (in_dim // 256)).times do |block|
      offset = block * 144
      q4kl_store_u16(payload_bytes, offset, 0x8000_u16)
      q4kl_store_u16(payload_bytes, offset + 2, 0_u16)
      12.times { |index| payload_bytes[offset + 4 + index] = 1_u8 }
    end

    {0, 1}.each do |patches|
      sidecar_bytes = patches == 0 ? Bytes[0xa5_u8] : Bytes.new(out_dim * 2 * 3, 0_u8)
      buffers = [] of ML::MetalBuffer
      command = nil.as(ML::Metal::CommandBuffer?)
      begin
        payload = ML::MetalBuffer.new(payload_bytes.size.to_i64); buffers << payload
        payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
        sidecar = ML::MetalBuffer.new(sidecar_bytes.size.to_i64); buffers << sidecar
        sidecar.write_bytes(sidecar_bytes.to_unsafe, sidecar_bytes.size)
        expected = ML::MetalBuffer.new(weight_count.to_i64 * sizeof(Float32)); buffers << expected
        actual = ML::MetalBuffer.new(1024_i64 * sizeof(Float32)); buffers << actual
        input = ML::MetalBuffer.new(input_values.size.to_i64 * sizeof(Float32)); buffers << input
        input.write(input_values)
        bias = ML::MetalBuffer.new(bias_values.size.to_i64 * sizeof(Float32)); buffers << bias
        bias.write(bias_values)
        scalar = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32)); buffers << scalar
        fused = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32)); buffers << fused
        status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << status
        status.contents.as(UInt32*).value = 0_u32
        fused_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << fused_status
        fused_status.contents.as(UInt32*).value = 0_u32

        active = ML::Metal::CommandBuffer.new
        command = active
        encoder = ML::Metal::ComputeEncoder.new(active)
        Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, expected,
          status, weight_count, patches)
        Q4KDeltaLinear.q4_fragment_probe(encoder, payload, sidecar, actual,
          status, patches)
        ML::Video::MetalOps.linear(encoder, input, expected, bias, scalar,
          rows, in_dim, out_dim)
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused,
          fused_status, rows, in_dim, out_dim, patches,
          Q4KDeltaLinear::GemmVariant::B64N32S4DeviceQ4K)
        encoder.end_encoding
        active.commit_and_wait
        active.completed_successfully?.should be_true
        status.contents.as(UInt32*).value.should eq(0_u32)
        fused_status.contents.as(UInt32*).value.should eq(0_u32)
        q4kl_assert_close("signed-zero direct Q4_K core patches=#{patches}",
          fused, scalar, output_count)

        expected_bits = expected.contents.as(UInt32*)
        actual_bits = actual.contents.as(UInt32*)
        sign_bits = patches == 0 ? 0x8000_0000_u32 : 0_u32
        [248, 256].each_with_index do |k_origin, origin_index|
          4.times do |simdgroup|
            2.times do |cb|
              8.times do |local_k|
                8.times do |local_n|
                  channel = (simdgroup // 2) * 16 + cb * 8 + local_n
                  index = origin_index * 512 + simdgroup * 128 + cb * 64 + local_k * 8 + local_n
                  expected_index = channel * in_dim + k_origin + local_k
                  expected_bits[expected_index].should eq(sign_bits)
                  actual_bits[index].should eq(sign_bits)
                end
              end
            end
          end
        end
      ensure
        if active = command
          active.discard unless active.committed?
        end
        q4kl_release(buffers)
      end
    end
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
    input = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    bias = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    base_output = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    corrected_output = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    bt_base_output = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    bt_corrected_output = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    tk64_base_output = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    tk64_corrected_output = ML::MetalBuffer.new(sizeof(Float32).to_i64)
    base_gemm_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    corrected_gemm_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    bt_base_gemm_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    bt_corrected_gemm_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    tk64_base_gemm_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    tk64_corrected_gemm_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
      base_sidecar.contents.as(UInt8*).value = 0xa5_u8
      corrected_sidecar.contents.as(UInt8*)[0] = 255_u8
      corrected_sidecar.contents.as(UInt8*)[1] = 0_u8
      corrected_sidecar.contents.as(UInt8*)[2] = 0_u8
      base_status.contents.as(UInt32*).value = 0_u32
      corrected_status.contents.as(UInt32*).value = 0_u32
      input.contents.as(Float32*)[0] = 1_f32
      1.upto(255) { |index| input.contents.as(Float32*)[index] = 0_f32 }
      bias.contents.as(Float32*).value = 0_f32
      base_gemm_status.contents.as(UInt32*).value = 0_u32
      corrected_gemm_status.contents.as(UInt32*).value = 0_u32
      bt_base_gemm_status.contents.as(UInt32*).value = 0_u32
      bt_corrected_gemm_status.contents.as(UInt32*).value = 0_u32
      tk64_base_gemm_status.contents.as(UInt32*).value = 0_u32
      tk64_corrected_gemm_status.contents.as(UInt32*).value = 0_u32

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDeltaLinear.decode_probe(encoder, payload, base_sidecar, base_decoded,
        base_status, 256, 0)
      Q4KDeltaLinear.decode_probe(encoder, payload, corrected_sidecar,
        corrected_decoded, corrected_status, 256, 1)
      Q4KDeltaLinear.linear(encoder, input, payload, base_sidecar, bias,
        base_output, base_gemm_status, 1, 256, 1, 0,
        Q4KDeltaLinear::GemmVariant::B128N32S4K32)
      Q4KDeltaLinear.linear(encoder, input, payload, corrected_sidecar, bias,
        corrected_output, corrected_gemm_status, 1, 256, 1, 1,
        Q4KDeltaLinear::GemmVariant::B128N32S4K32)
      Q4KDeltaLinear.linear(encoder, input, payload, base_sidecar, bias,
        bt_base_output, bt_base_gemm_status, 1, 256, 1, 0,
        Q4KDeltaLinear::GemmVariant::B64N32S4K32BT)
      Q4KDeltaLinear.linear(encoder, input, payload, corrected_sidecar, bias,
        bt_corrected_output, bt_corrected_gemm_status, 1, 256, 1, 1,
        Q4KDeltaLinear::GemmVariant::B64N32S4K32BT)
      Q4KDeltaLinear.linear(encoder, input, payload, base_sidecar, bias,
        tk64_base_output, tk64_base_gemm_status, 1, 256, 1, 0,
        Q4KDeltaLinear::GemmVariant::B64N32S4K64BT)
      Q4KDeltaLinear.linear(encoder, input, payload, corrected_sidecar, bias,
        tk64_corrected_output, tk64_corrected_gemm_status, 1, 256, 1, 1,
        Q4KDeltaLinear::GemmVariant::B64N32S4K64BT)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      base_status.contents.as(UInt32*).value.should eq(0_u32)
      corrected_status.contents.as(UInt32*).value.should eq(0_u32)
      base_gemm_status.contents.as(UInt32*).value.should eq(0_u32)
      corrected_gemm_status.contents.as(UInt32*).value.should eq(0_u32)
      bt_base_gemm_status.contents.as(UInt32*).value.should eq(0_u32)
      bt_corrected_gemm_status.contents.as(UInt32*).value.should eq(0_u32)
      tk64_base_gemm_status.contents.as(UInt32*).value.should eq(0_u32)
      tk64_corrected_gemm_status.contents.as(UInt32*).value.should eq(0_u32)
      base_decoded.contents.as(UInt32*)[0].should eq(0x8000_0000_u32)
      corrected_decoded.contents.as(UInt32*)[0].should eq(0_u32)
      q4kl_load_f32(base_output, 0).should eq(0_f32)
      q4kl_load_f32(corrected_output, 0).should eq(0_f32)
      q4kl_load_f32(bt_base_output, 0).should eq(0_f32)
      q4kl_load_f32(bt_corrected_output, 0).should eq(0_f32)
      q4kl_load_f32(tk64_base_output, 0).should eq(0_f32)
      q4kl_load_f32(tk64_corrected_output, 0).should eq(0_f32)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      {payload, base_sidecar, corrected_sidecar, base_decoded,
       corrected_decoded, base_status, corrected_status, input, bias,
       base_output, corrected_output, base_gemm_status,
       corrected_gemm_status, bt_base_output, bt_corrected_output,
       bt_base_gemm_status, bt_corrected_gemm_status, tk64_base_output,
       tk64_corrected_output, tk64_base_gemm_status,
       tk64_corrected_gemm_status}.each(&.release)
    end
  end

  it "keeps malformed Q4_K blocks local in the direct-device core and preserves sticky status" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDeltaLinear.prepare!
    ML::Video::MetalOps.prepare!

    rows = 64
    in_dim = 512
    out_dim = 32
    weight_count = in_dim * out_dim
    output_count = rows * out_dim
    input_values = Array(Float32).new(rows * in_dim, 1_f32)
    bias_values = Array(Float32).new(out_dim, 0.125_f32)
    payload_bytes = q4kl_test_payload(in_dim, out_dim)
    malformed_bytes = q4kl_test_sidecar(in_dim, out_dim, 4)
    # Duplicate one correction index in channel 0, block 0. Every other
    # channel/block remains independently valid.
    malformed_bytes[3] = malformed_bytes[0]

    buffers = [] of ML::MetalBuffer
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      input = ML::MetalBuffer.new(input_values.size.to_i64 * sizeof(Float32)); buffers << input
      input.write(input_values)
      payload = ML::MetalBuffer.new(payload_bytes.size.to_i64); buffers << payload
      payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
      sidecar = ML::MetalBuffer.new(malformed_bytes.size.to_i64); buffers << sidecar
      sidecar.write_bytes(malformed_bytes.to_unsafe, malformed_bytes.size)
      bias = ML::MetalBuffer.new(bias_values.size.to_i64 * sizeof(Float32)); buffers << bias
      bias.write(bias_values)
      decoded = ML::MetalBuffer.new(weight_count.to_i64 * sizeof(Float32)); buffers << decoded
      scalar = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32)); buffers << scalar
      guard_count = 8
      fused = ML::MetalBuffer.new((output_count + guard_count).to_i64 * sizeof(Float32)); buffers << fused
      fused_values = fused.contents.as(Float32*)
      (output_count + guard_count).times { |index| fused_values[index] = 765_432.125_f32 }
      decode_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << decode_status
      fused_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << fused_status
      decode_status.contents.as(UInt32*).value = 0_u32
      fused_status.contents.as(UInt32*).value = 0x40_u32

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, decoded,
        decode_status, weight_count, 4)
      ML::Video::MetalOps.linear(encoder, input, decoded, bias, scalar,
        rows, in_dim, out_dim)
      Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused,
        fused_status, rows, in_dim, out_dim, 4,
        Q4KDeltaLinear::GemmVariant::B64N32S4DeviceQ4K)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      decode_status.contents.as(UInt32*).value.should eq(1_u32)
      fused_status.contents.as(UInt32*).value.should eq(0x41_u32)

      decoded_values = decoded.contents.as(Float32*)
      256.times do |k|
        decoded_values[k].should eq(0_f32)
      end
      decoded_values[256].should_not eq(0_f32) # Channel 0, next Q4_K block.
      decoded_values[512].should_not eq(0_f32) # Neighbor channel, first block.
      q4kl_assert_close("malformed direct-device Q4_K core", fused, scalar, output_count)
      guard_count.times do |index|
        fused_values[output_count + index].should eq(765_432.125_f32)
      end
    ensure
      if active = command
        active.discard unless active.committed?
      end
      q4kl_release(buffers)
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
        candidate_fused = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << candidate_fused
        bt_candidate_fused = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << bt_candidate_fused
        tk64_candidate_fused = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << tk64_candidate_fused
        fused.contents.as(Float32*)[0] = 1234_f32
        fused.contents.as(Float32*)[1] = -1234_f32
        candidate_fused.contents.as(Float32*)[0] = 1234_f32
        candidate_fused.contents.as(Float32*)[1] = -1234_f32
        bt_candidate_fused.contents.as(Float32*)[0] = 1234_f32
        bt_candidate_fused.contents.as(Float32*)[1] = -1234_f32
        tk64_candidate_fused.contents.as(Float32*)[0] = 1234_f32
        tk64_candidate_fused.contents.as(Float32*)[1] = -1234_f32
        probe_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << probe_status
        fused_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << fused_status
        candidate_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << candidate_status
        bt_candidate_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << bt_candidate_status
        tk64_candidate_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << tk64_candidate_status
        probe_status.contents.as(UInt32*).value = 0_u32
        fused_status.contents.as(UInt32*).value = 0_u32
        candidate_status.contents.as(UInt32*).value = 0_u32
        bt_candidate_status.contents.as(UInt32*).value = 0_u32
        tk64_candidate_status.contents.as(UInt32*).value = 0_u32

        active = ML::Metal::CommandBuffer.new
        command = active
        encoder = ML::Metal::ComputeEncoder.new(active)
        Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, decoded,
          probe_status, out_dim * in_dim, 4)
        ML::Video::MetalOps.linear(encoder, input, decoded, bias, scalar,
          rows, in_dim, out_dim)
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused,
          fused_status, rows, in_dim, out_dim, 4)
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias,
          candidate_fused, candidate_status, rows, in_dim, out_dim, 4,
          Q4KDeltaLinear::GemmVariant::B128N32S4K32)
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias,
          bt_candidate_fused, bt_candidate_status, rows, in_dim, out_dim, 4,
          Q4KDeltaLinear::GemmVariant::B64N32S4K32BT)
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias,
          tk64_candidate_fused, tk64_candidate_status, rows, in_dim, out_dim, 4,
          Q4KDeltaLinear::GemmVariant::B64N32S4K64BT)
        encoder.end_encoding
        active.commit_and_wait
        active.completed_successfully?.should be_true
        probe_status.contents.as(UInt32*).value.should(eq(1_u32), "#{label} probe status")
        fused_status.contents.as(UInt32*).value.should(eq(1_u32), "#{label} fused status")
        candidate_status.contents.as(UInt32*).value.should(eq(1_u32), "#{label} candidate status")
        bt_candidate_status.contents.as(UInt32*).value.should(eq(1_u32), "#{label} transposed-B candidate status")
        tk64_candidate_status.contents.as(UInt32*).value.should(eq(1_u32), "#{label} TK64 transposed-B candidate status")

        256.times do |index|
          q4kl_load_f32(decoded, index).should(eq(0_f32), "#{label} malformed channel 0 block")
        end
        256.upto(out_dim * in_dim - 1) do |index|
          q4kl_load_f32(decoded, index).finite?.should be_true
        end
        q4kl_load_f32(decoded, 256).should_not eq(0_f32) # channel 0, K block 1
        q4kl_load_f32(decoded, 512).should_not eq(0_f32) # channel 1, K block 0
        q4kl_assert_close("#{label} malformed fused output", fused, scalar, out_dim)
        q4kl_assert_close("#{label} malformed candidate output", candidate_fused, scalar, out_dim)
        q4kl_assert_close("#{label} malformed transposed-B candidate output", bt_candidate_fused, scalar, out_dim)
        q4kl_assert_close("#{label} malformed TK64 transposed-B candidate output", tk64_candidate_fused, scalar, out_dim)
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
      candidate_fused = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << candidate_fused
      bt_candidate_fused = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << bt_candidate_fused
      tk64_candidate_fused = ML::MetalBuffer.new(out_dim.to_i64 * sizeof(Float32)); buffers << tk64_candidate_fused
      fused.contents.as(Float32*)[0] = 1234_f32
      fused.contents.as(Float32*)[1] = -1234_f32
      candidate_fused.contents.as(Float32*)[0] = 1234_f32
      candidate_fused.contents.as(Float32*)[1] = -1234_f32
      bt_candidate_fused.contents.as(Float32*)[0] = 1234_f32
      bt_candidate_fused.contents.as(Float32*)[1] = -1234_f32
      tk64_candidate_fused.contents.as(Float32*)[0] = 1234_f32
      tk64_candidate_fused.contents.as(Float32*)[1] = -1234_f32
      decode_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << decode_status
      fused_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << fused_status
      candidate_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << candidate_status
      bt_candidate_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << bt_candidate_status
      tk64_candidate_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64); buffers << tk64_candidate_status
      decode_status.contents.as(UInt32*).value = 0_u32
      fused_status.contents.as(UInt32*).value = 0x40_u32
      candidate_status.contents.as(UInt32*).value = 0x40_u32
      bt_candidate_status.contents.as(UInt32*).value = 0x40_u32
      tk64_candidate_status.contents.as(UInt32*).value = 0x40_u32

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDeltaLinear.decode_probe(encoder, payload, sidecar, decoded,
        decode_status, out_dim * in_dim, 4)
      ML::Video::MetalOps.linear(encoder, input, decoded, bias, scalar,
        rows, in_dim, out_dim)
      Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, fused,
        fused_status, rows, in_dim, out_dim, 4)
      Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias,
        candidate_fused, candidate_status, rows, in_dim, out_dim, 4,
        Q4KDeltaLinear::GemmVariant::B128N32S4K32)
      Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias,
        bt_candidate_fused, bt_candidate_status, rows, in_dim, out_dim, 4,
        Q4KDeltaLinear::GemmVariant::B64N32S4K32BT)
      Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias,
        tk64_candidate_fused, tk64_candidate_status, rows, in_dim, out_dim, 4,
        Q4KDeltaLinear::GemmVariant::B64N32S4K64BT)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      decode_status.contents.as(UInt32*).value.should eq(0_u32)
      fused_status.contents.as(UInt32*).value.should eq(0x40_u32)
      candidate_status.contents.as(UInt32*).value.should eq(0x40_u32)
      bt_candidate_status.contents.as(UInt32*).value.should eq(0x40_u32)
      tk64_candidate_status.contents.as(UInt32*).value.should eq(0x40_u32)
      q4kl_assert_close("pre-existing sticky status still computes", fused, scalar, out_dim)
      q4kl_assert_close("pre-existing sticky status candidate still computes", candidate_fused, scalar, out_dim)
      q4kl_assert_close("pre-existing sticky status transposed-B candidate still computes", bt_candidate_fused, scalar, out_dim)
      q4kl_assert_close("pre-existing sticky status TK64 transposed-B candidate still computes", tk64_candidate_fused, scalar, out_dim)
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
        Q4KDeltaLinear.linear(encoder, tiny_input, payload, sidecar, bias, output,
          status, 1, 256, 1, 0, Q4KDeltaLinear::GemmVariant::B128N32S4K32)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, tiny_input, payload, sidecar, bias, output,
          status, 1, 256, 1, 0, Q4KDeltaLinear::GemmVariant::B64N32S4K32BT)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, tiny_input, payload, sidecar, bias, output,
          status, 1, 256, 1, 0, Q4KDeltaLinear::GemmVariant::B64N32S4K64BT)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, tiny_input, payload, sidecar, bias, output,
          status, 1, 256, 1, 0, Q4KDeltaLinear::GemmVariant::B64N32S4DeviceQ4K)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, input,
          status, 1, 256, 1, 0)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, input,
          status, 1, 256, 1, 0, Q4KDeltaLinear::GemmVariant::B128N32S4K32)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, input,
          status, 1, 256, 1, 0, Q4KDeltaLinear::GemmVariant::B64N32S4K32BT)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, input,
          status, 1, 256, 1, 0, Q4KDeltaLinear::GemmVariant::B64N32S4K64BT)
      end
      expect_raises(ArgumentError) do
        Q4KDeltaLinear.linear(encoder, input, payload, sidecar, bias, input,
          status, 1, 256, 1, 0, Q4KDeltaLinear::GemmVariant::B64N32S4DeviceQ4K)
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
