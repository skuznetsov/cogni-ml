require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/q4k_delta_weights"

alias Q4KDelta = ML::Video::Q4KDeltaWeights

private def q4kd_store_u16(bytes : Bytes, offset : Int32, value : UInt16) : Nil
  bytes[offset] = (value & 0xff_u16).to_u8
  bytes[offset + 1] = (value >> 8).to_u8
end

private def q4kd_load_u16(bytes : Bytes, offset : Int32) : UInt16
  bytes[offset].to_u16 | (bytes[offset + 1].to_u16 << 8)
end

private def q4kd_input(count : Int32, values : Hash(Int32, UInt16)) : Bytes
  bytes = Bytes.new(count * 2, 0_u8)
  values.each { |index, bits| q4kd_store_u16(bytes, index * 2, bits) }
  bytes
end

private def q4kd_zero_payload(blocks : Int32 = 1) : Bytes
  Bytes.new(blocks * Q4KDelta::Q4_K_BLOCK_BYTES, 0_u8)
end

private def q4kd_rounding_payload : Bytes
  # d=half(0x2e66)=0.0999755859375, scale[0]=1, q[0]=3.
  # The canonical decoder therefore returns 0.2999267578125 at index zero.
  bytes = q4kd_zero_payload
  bytes[0] = 0x66_u8
  bytes[1] = 0x2e_u8
  bytes[4] = 1_u8
  bytes[16] = 3_u8
  bytes
end

private def q4kd_sidecar_bytes(buffer : ML::MetalBuffer, size : Int32) : Bytes
  Bytes.new(buffer.contents.as(UInt8*), size)
end

private def q4kd_output(buffer : ML::MetalBuffer, index : Int32) : Float32
  buffer.contents.as(Float32*)[index]
end

private def q4kd_encode_decode(command : ML::Metal::CommandBuffer,
                               source : ML::MetalBuffer,
                               payload : ML::MetalBuffer,
                               sidecar : ML::MetalBuffer,
                               status : ML::MetalBuffer,
                               destination : ML::MetalBuffer,
                               count : Int32,
                               patches : Int32) : Nil
  encoder = ML::Metal::ComputeEncoder.new(command)
  Q4KDelta.pack(encoder, source, payload, sidecar, status, count, patches)
  Q4KDelta.decode(encoder, payload, sidecar, destination, status, count, patches)
  encoder.end_encoding
end

describe Q4KDelta do
  it "reports fixed sidecar geometry and admits only whole blocks and supported K values" do
    Q4KDelta::BLOCK_ELEMENTS.should eq(256)
    Q4KDelta::Q4_K_BLOCK_BYTES.should eq(144)
    Q4KDelta::STATUS_WORDS.should eq(1)
    {1, 4, 8}.each do |patches|
      Q4KDelta.sidecar_bytes(256, patches).should eq((patches * 3).to_i64)
      Q4KDelta.sidecar_bytes(512, patches).should eq((patches * 6).to_i64)
    end
    {0, -256, 255, 257}.each do |count|
      expect_raises(ArgumentError) { Q4KDelta.sidecar_bytes(count, 1) }
    end
    {0, 2, 3, 9}.each do |patches|
      expect_raises(ArgumentError) { Q4KDelta.sidecar_bytes(256, patches) }
    end
  end
end

describe "Q4_K sparse F16 delta native packing and decoding" do
  it "selects exactly K by descending absolute error with lower-index ties and fixed per-block strides" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDelta.prepare!

    # Block zero has ties at errors 2 and 1, including the final local index.
    # Block one exercises a separate fixed-K sidecar stride and the same edge.
    input_bytes = q4kd_input(512, {
              0 => 0x4000_u16,
              2 => 0xc000_u16,
              5 => 0x3f80_u16,
            255 => 0xbf80_u16,
      256 + 0   => 0x3f80_u16,
      256 + 128 => 0xc000_u16,
      256 + 255 => 0x4040_u16,
    })
    source = ML::MetalBuffer.new(input_bytes.size.to_i64)
    payload_bytes = q4kd_zero_payload(2)
    payload = ML::MetalBuffer.new(payload_bytes.size.to_i64)
    sidecar_1 = ML::MetalBuffer.new(Q4KDelta.sidecar_bytes(512, 1))
    sidecar_4 = ML::MetalBuffer.new(Q4KDelta.sidecar_bytes(512, 4))
    sidecar_8 = ML::MetalBuffer.new(Q4KDelta.sidecar_bytes(512, 8))
    status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    destination = ML::MetalBuffer.new(512_i64 * sizeof(Float32))
    destination_1 = ML::MetalBuffer.new(512_i64 * sizeof(Float32))
    destination_8 = ML::MetalBuffer.new(512_i64 * sizeof(Float32))
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      source.write_bytes(input_bytes.to_unsafe, input_bytes.size)
      payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
      status.contents.as(UInt32*).value = 0_u32

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      allocations = ML::MetalBuffer.video_spec_allocations
      Q4KDelta.pack(encoder, source, payload, sidecar_1, status, 512, 1)
      Q4KDelta.pack(encoder, source, payload, sidecar_4, status, 512, 4)
      Q4KDelta.pack(encoder, source, payload, sidecar_8, status, 512, 8)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should eq(0_u32)

      # K=1: largest absolute error, local index zero.
      q4kd_sidecar_bytes(sidecar_1, 6).should eq(Bytes[
        0_u8, 0x00_u8, 0x40_u8, 255_u8, 0x00_u8, 0x42_u8,
      ])
      # K=4: rank ties by lower local index. Block one starts exactly 12 bytes later.
      q4kd_sidecar_bytes(sidecar_4, 24).should eq(Bytes[
        0_u8, 0x00_u8, 0x40_u8, 2_u8, 0x00_u8, 0xc0_u8,
        5_u8, 0x00_u8, 0x3c_u8, 255_u8, 0x00_u8, 0xbc_u8,
        255_u8, 0x00_u8, 0x42_u8, 128_u8, 0x00_u8, 0xc0_u8,
        0_u8, 0x00_u8, 0x3c_u8, 1_u8, 0x00_u8, 0x00_u8,
      ])
      # K=8 adds zero-error positions in ascending-index order after nonzero errors.
      q4kd_sidecar_bytes(sidecar_8, 48)[0, 24].should eq(Bytes[
        0_u8, 0x00_u8, 0x40_u8, 2_u8, 0x00_u8, 0xc0_u8,
        5_u8, 0x00_u8, 0x3c_u8, 255_u8, 0x00_u8, 0xbc_u8,
        1_u8, 0x00_u8, 0x00_u8, 3_u8, 0x00_u8, 0x00_u8,
        4_u8, 0x00_u8, 0x00_u8, 6_u8, 0x00_u8, 0x00_u8,
      ])

      decode_command = ML::Metal::CommandBuffer.new
      command = decode_command
      decode_encoder = ML::Metal::ComputeEncoder.new(decode_command)
      Q4KDelta.decode(decode_encoder, payload, sidecar_1, destination_1, status, 512, 1)
      Q4KDelta.decode(decode_encoder, payload, sidecar_4, destination, status, 512, 4)
      Q4KDelta.decode(decode_encoder, payload, sidecar_8, destination_8, status, 512, 8)
      decode_encoder.end_encoding
      decode_command.commit_and_wait
      decode_command.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should eq(0_u32)
      q4kd_output(destination, 0).should eq(2.0_f32)
      q4kd_output(destination, 2).should eq(-2.0_f32)
      q4kd_output(destination, 5).should eq(1.0_f32)
      q4kd_output(destination, 255).should eq(-1.0_f32)
      q4kd_output(destination, 256 + 255).should eq(3.0_f32)
      q4kd_output(destination, 256 + 128).should eq(-2.0_f32)
      q4kd_output(destination, 256 + 0).should eq(1.0_f32)
      q4kd_output(destination, 256 + 1).should eq(0.0_f32)
      q4kd_output(destination_1, 0).should eq(2.0_f32)
      q4kd_output(destination_1, 2).should eq(0.0_f32)
      q4kd_output(destination_1, 256 + 255).should eq(3.0_f32)
      q4kd_output(destination_8, 2).should eq(-2.0_f32)
      q4kd_output(destination_8, 255).should eq(-1.0_f32)
      q4kd_output(destination_8, 256 + 128).should eq(-2.0_f32)
      q4kd_output(destination_8, 256 + 255).should eq(3.0_f32)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      {source, payload, sidecar_1, sidecar_4, sidecar_8, status, destination,
       destination_1, destination_8}.each(&.release)
    end
  end

  it "stores a rounded nonzero F16 delta and adds it to the canonical F32 decode in one dispatch" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDelta.prepare!

    input_bytes = q4kd_input(256, {0 => 0x3f80_u16})
    packed_bytes = q4kd_rounding_payload
    source = ML::MetalBuffer.new(input_bytes.size.to_i64)
    payload = ML::MetalBuffer.new(packed_bytes.size.to_i64)
    sidecar = ML::MetalBuffer.new(Q4KDelta.sidecar_bytes(256, 1))
    status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    destination = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      source.write_bytes(input_bytes.to_unsafe, input_bytes.size)
      payload.write_bytes(packed_bytes.to_unsafe, packed_bytes.size)
      status.contents.as(UInt32*).value = 0_u32
      active = ML::Metal::CommandBuffer.new
      command = active
      q4kd_encode_decode(active, source, payload, sidecar, status, destination, 256, 1)
      active.commit_and_wait
      active.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should eq(0_u32)
      q4kd_sidecar_bytes(sidecar, 3).should eq(Bytes[0_u8, 0x9a_u8, 0x39_u8])
      # 0.2999267578125 + half(0x399a)=0.7001953125.
      q4kd_output(destination, 0).should eq(1.0001220703125_f32)
      q4kd_output(destination, 1).should eq(0.0_f32)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      {source, payload, sidecar, status, destination}.each(&.release)
    end
  end
end

describe "Q4_K sparse F16 delta guards" do
  it "rejects invalid geometry, K, undersized buffers, and every alias including status" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDelta.prepare!

    source = ML::MetalBuffer.new(512_i64)
    payload = ML::MetalBuffer.new(144_i64)
    sidecar = ML::MetalBuffer.new(12_i64)
    status = ML::MetalBuffer.new(4_i64)
    destination = ML::MetalBuffer.new(1024_i64)
    small_source = ML::MetalBuffer.new(511_i64)
    small_payload = ML::MetalBuffer.new(143_i64)
    small_sidecar = ML::MetalBuffer.new(11_i64)
    small_status = ML::MetalBuffer.new(3_i64)
    small_destination = ML::MetalBuffer.new(1023_i64)
    shared = ML::MetalBuffer.new(2048_i64)
    released = ML::MetalBuffer.new(512_i64)
    released.release
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    ended = false
    begin
      status.contents.as(UInt32*).value = 0_u32
      {0, -256, 255, 257}.each do |count|
        expect_raises(ArgumentError) { Q4KDelta.pack(encoder, source, payload, sidecar, status, count, 4) }
        expect_raises(ArgumentError) { Q4KDelta.decode(encoder, payload, sidecar, destination, status, count, 4) }
      end
      {0, 2, 3, 9}.each do |patches|
        expect_raises(ArgumentError) { Q4KDelta.pack(encoder, source, payload, sidecar, status, 256, patches) }
        expect_raises(ArgumentError) { Q4KDelta.decode(encoder, payload, sidecar, destination, status, 256, patches) }
      end
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, small_source, payload, sidecar, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, source, small_payload, sidecar, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, source, payload, small_sidecar, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, source, payload, sidecar, small_status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, payload, sidecar, small_destination, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, small_payload, sidecar, destination, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, payload, small_sidecar, destination, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, payload, sidecar, destination, small_status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, released, payload, sidecar, status, 256, 4) }

      # Each pairwise alias is rejected, including aliases through status.
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, shared, shared, sidecar, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, shared, payload, shared, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, shared, payload, sidecar, shared, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, source, shared, shared, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, source, shared, sidecar, shared, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.pack(encoder, source, payload, shared, shared, 256, 4) }

      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, shared, shared, destination, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, shared, sidecar, shared, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, shared, sidecar, destination, shared, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, payload, shared, shared, status, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, payload, shared, destination, shared, 256, 4) }
      expect_raises(ArgumentError) { Q4KDelta.decode(encoder, payload, sidecar, shared, shared, 256, 4) }
      encoder.end_encoding
      ended = true
      command.discard
    ensure
      encoder.end_encoding unless ended
      command.discard unless command.committed?
      {source, payload, sidecar, status, destination, small_source, small_payload,
       small_sidecar, small_status, small_destination, shared}.each(&.release)
    end
  end

  it "zeroes failed pack sidecars, reports nonfinite and F16-overflow inputs, and keeps status sticky" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDelta.prepare!

    source_bytes = q4kd_input(256, {0 => 0x7fc0_u16})   # BF16 NaN
    overflow_bytes = q4kd_input(256, {0 => 0x4780_u16}) # finite 65536; not representable in F16
    source = ML::MetalBuffer.new(512_i64)
    payload = ML::MetalBuffer.new(144_i64)
    bad_source_sidecar = ML::MetalBuffer.new(12_i64)
    overflow_sidecar = ML::MetalBuffer.new(12_i64)
    base_nan_sidecar = ML::MetalBuffer.new(12_i64)
    status = ML::MetalBuffer.new(4_i64)
    valid_source = ML::MetalBuffer.new(512_i64)
    valid_sidecar = ML::MetalBuffer.new(12_i64)
    valid_destination = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    zero_payload = q4kd_zero_payload
    nan_payload = q4kd_zero_payload
    nan_payload[0] = 0x00_u8
    nan_payload[1] = 0x7c_u8
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      payload.write_bytes(zero_payload.to_unsafe, zero_payload.size)
      source.write_bytes(source_bytes.to_unsafe, source_bytes.size)
      status.contents.as(UInt32*).value = 0_u32
      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDelta.pack(encoder, source, payload, bad_source_sidecar, status, 256, 4)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should_not eq(0_u32)
      q4kd_sidecar_bytes(bad_source_sidecar, 12).should eq(Bytes.new(12, 0_u8))

      # A finite BF16 source can still overflow when its delta is narrowed to F16.
      source.write_bytes(overflow_bytes.to_unsafe, overflow_bytes.size)
      overflow_command = ML::Metal::CommandBuffer.new
      command = overflow_command
      overflow_encoder = ML::Metal::ComputeEncoder.new(overflow_command)
      Q4KDelta.pack(overflow_encoder, source, payload, overflow_sidecar, status, 256, 4)
      overflow_encoder.end_encoding
      overflow_command.commit_and_wait
      overflow_command.completed_successfully?.should be_true
      q4kd_sidecar_bytes(overflow_sidecar, 12).should eq(Bytes.new(12, 0_u8))
      status.contents.as(UInt32*).value.should_not eq(0_u32)

      # A nonfinite canonical base rejects the whole pack block as well.
      payload.write_bytes(nan_payload.to_unsafe, nan_payload.size)
      source.write_bytes(Bytes.new(512, 0_u8).to_unsafe, 512)
      base_command = ML::Metal::CommandBuffer.new
      command = base_command
      base_encoder = ML::Metal::ComputeEncoder.new(base_command)
      Q4KDelta.pack(base_encoder, source, payload, base_nan_sidecar, status, 256, 4)
      base_encoder.end_encoding
      base_command.commit_and_wait
      base_command.completed_successfully?.should be_true
      q4kd_sidecar_bytes(base_nan_sidecar, 12).should eq(Bytes.new(12, 0_u8))
      status.contents.as(UInt32*).value.should_not eq(0_u32)

      # Sticky status does not poison valid sidecar output or decoded values.
      valid_input = q4kd_input(256, {0 => 0x3f80_u16})
      valid_source.write_bytes(valid_input.to_unsafe, valid_input.size)
      payload.write_bytes(zero_payload.to_unsafe, zero_payload.size)
      valid_command = ML::Metal::CommandBuffer.new
      command = valid_command
      valid_encoder = ML::Metal::ComputeEncoder.new(valid_command)
      Q4KDelta.pack(valid_encoder, valid_source, payload, valid_sidecar, status, 256, 4)
      Q4KDelta.decode(valid_encoder, payload, valid_sidecar, valid_destination, status, 256, 4)
      valid_encoder.end_encoding
      valid_command.commit_and_wait
      valid_command.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should_not eq(0_u32)
      q4kd_sidecar_bytes(valid_sidecar, 12)[0, 3].should eq(Bytes[0_u8, 0x00_u8, 0x3c_u8])
      q4kd_output(valid_destination, 0).should eq(1.0_f32)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      {source, payload, bad_source_sidecar, overflow_sidecar, base_nan_sidecar,
       status, valid_source, valid_sidecar, valid_destination}.each(&.release)
    end
  end

  it "rejects duplicate or nonfinite sidecar records as a zero-output block and sets sticky status" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KDelta.prepare!

    payload_bytes = q4kd_zero_payload
    duplicate_records = Bytes[
      7_u8, 0x00_u8, 0x3c_u8,
      7_u8, 0x00_u8, 0x3c_u8,
      9_u8, 0x00_u8, 0x3c_u8,
      11_u8, 0x00_u8, 0x3c_u8,
    ]
    nonfinite_records = Bytes[
      7_u8, 0x00_u8, 0x7c_u8,
      8_u8, 0x00_u8, 0x3c_u8,
      9_u8, 0x00_u8, 0x3c_u8,
      10_u8, 0x00_u8, 0x3c_u8,
    ]
    payload = ML::MetalBuffer.new(payload_bytes.size.to_i64)
    duplicate_sidecar = ML::MetalBuffer.new(duplicate_records.size.to_i64)
    nonfinite_sidecar = ML::MetalBuffer.new(nonfinite_records.size.to_i64)
    status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    destination = ML::MetalBuffer.new(256_i64 * sizeof(Float32))
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      payload.write_bytes(payload_bytes.to_unsafe, payload_bytes.size)
      duplicate_sidecar.write_bytes(duplicate_records.to_unsafe, duplicate_records.size)
      nonfinite_sidecar.write_bytes(nonfinite_records.to_unsafe, nonfinite_records.size)
      status.contents.as(UInt32*).value = 0_u32

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KDelta.decode(encoder, payload, duplicate_sidecar, destination, status, 256, 4)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should_not eq(0_u32)
      256.times { |index| q4kd_output(destination, index).should eq(0.0_f32) }

      status.contents.as(UInt32*).value = 0x80_u32
      nonfinite_command = ML::Metal::CommandBuffer.new
      command = nonfinite_command
      nonfinite_encoder = ML::Metal::ComputeEncoder.new(nonfinite_command)
      Q4KDelta.decode(nonfinite_encoder, payload, nonfinite_sidecar, destination, status, 256, 4)
      nonfinite_encoder.end_encoding
      nonfinite_command.commit_and_wait
      nonfinite_command.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should eq(0x81_u32)
      256.times { |index| q4kd_output(destination, index).should eq(0.0_f32) }
    ensure
      if active = command
        active.discard unless active.committed?
      end
      {payload, duplicate_sidecar, nonfinite_sidecar, status, destination}.each(&.release)
    end
  end
end
