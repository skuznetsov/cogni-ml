require "spec"
require "../src/ml/video/q4k_range_weights"

alias Q4KRangePacker = ML::Video::Q4KRangeWeights

private def q4kr_store_bf16(bytes : Bytes, index : Int32, bits : UInt16) : Nil
  bytes[index * 2] = (bits & 0xff_u16).to_u8
  bytes[index * 2 + 1] = (bits >> 8).to_u8
end

private def q4kr_repeated_input(value : UInt16, blocks : Int32 = 1) : Bytes
  bytes = Bytes.new(blocks * Q4KRangePacker::BLOCK_ELEMENTS * 2, 0_u8)
  (blocks * Q4KRangePacker::BLOCK_ELEMENTS).times do |index|
    q4kr_store_bf16(bytes, index, value)
  end
  bytes
end

private def q4kr_concat(inputs : Array(Bytes)) : Bytes
  total = inputs.sum(&.size)
  combined = Bytes.new(total, 0_u8)
  cursor = 0
  inputs.each do |input|
    combined[cursor, input.size].copy_from(input)
    cursor += input.size
  end
  combined
end

private def q4kr_ranges_input(values : Array(UInt16)) : Bytes
  raise ArgumentError.new("Expected eight BF16 range values") unless values.size == 8
  bytes = Bytes.new(Q4KRangePacker::BLOCK_ELEMENTS * 2, 0_u8)
  Q4KRangePacker::BLOCK_ELEMENTS.times do |index|
    q4kr_store_bf16(bytes, index, values[index // 32])
  end
  bytes
end

private def q4kr_payload_block(payload : ML::MetalBuffer, block_index : Int32 = 0) : Bytes
  address = payload.contents.as(UInt8*) + block_index * Q4KRangePacker::Q4_K_BLOCK_BYTES
  Bytes.new(address, Q4KRangePacker::Q4_K_BLOCK_BYTES)
end

private def q4kr_scale_code(block : Bytes, index : Int32) : UInt8
  if index < 4
    block[4 + index] & 0x3f_u8
  else
    (block[8 + index] & 0x0f_u8) | ((block[index] >> 2) & 0x30_u8)
  end
end

private def q4kr_minimum_code(block : Bytes, index : Int32) : UInt8
  if index < 4
    block[8 + index] & 0x3f_u8
  else
    (block[8 + index] >> 4) | ((block[4 + index] >> 2) & 0x30_u8)
  end
end

describe Q4KRangePacker do
  it "reports the bounded same-format payload geometry and rejects invalid counts" do
    Q4KRangePacker::STATUS_WORDS.should eq(1)
    Q4KRangePacker.payload_bytes(256).should eq(144_i64)
    Q4KRangePacker.payload_bytes(512).should eq(288_i64)

    {0, -256, 255}.each do |count|
      expect_raises(ArgumentError) { Q4KRangePacker.payload_bytes(count) }
    end
  end
end

describe "Q4_K range-native packing" do
  it "matches hand-computed all-zero, positive-constant, and negative-constant blocks" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KRangePacker.prepare!

    input_bytes = q4kr_concat([q4kr_repeated_input(0x0000_u16),
                               q4kr_repeated_input(0x8000_u16),
                               q4kr_repeated_input(0x4000_u16),
                               q4kr_repeated_input(0xc000_u16)])
    original = ML::MetalBuffer.new(input_bytes.size.to_i64)
    payload = ML::MetalBuffer.new(Q4KRangePacker.payload_bytes(1024))
    status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      original.write_bytes(input_bytes.to_unsafe, input_bytes.size)
      status.contents.as(UInt32*).value = 0_u32

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KRangePacker.pack(encoder, original, payload, status, 1024, 1)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should eq(0_u32)

      q4kr_payload_block(payload, 0).should eq(Bytes.new(144, 0_u8))
      q4kr_payload_block(payload, 1).should eq(Bytes.new(144, 0_u8))

      positive = Bytes.new(144, 0_u8)
      positive[0] = 0x56_u8
      positive[1] = 0x18_u8
      4.times { |index| positive[4 + index] = 0xff_u8 }
      4.times { |index| positive[12 + index] = 0x0f_u8 }
      (16...144).each { |index| positive[index] = 0xff_u8 }
      q4kr_payload_block(payload, 2).should eq(positive)

      negative = Bytes.new(144, 0_u8)
      negative[0] = 0x56_u8
      negative[1] = 0x18_u8
      negative[2] = 0x10_u8
      negative[3] = 0x28_u8
      (4...16).each { |index| negative[index] = 0xff_u8 }
      q4kr_payload_block(payload, 3).should eq(negative)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      original.release
      payload.release
      status.release
    end
  end

  it "uses one shared fit for range count 1 and distinct local fits for range count 8" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KRangePacker.prepare!

    input_bytes = q4kr_ranges_input([0x3f80_u16, 0x4000_u16, 0x4040_u16, 0x4080_u16,
                                     0x40a0_u16, 0x40c0_u16, 0x40e0_u16, 0x4100_u16])
    original = ML::MetalBuffer.new(input_bytes.size.to_i64)
    shared_payload = ML::MetalBuffer.new(Q4KRangePacker.payload_bytes(256))
    local_payload = ML::MetalBuffer.new(Q4KRangePacker.payload_bytes(256))
    status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      original.write_bytes(input_bytes.to_unsafe, input_bytes.size)
      status.contents.as(UInt32*).value = 0_u32
      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      Q4KRangePacker.pack(encoder, original, shared_payload, status, 256, 1)
      Q4KRangePacker.pack(encoder, original, local_payload, status, 256, 8)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true
      status.contents.as(UInt32*).value.should eq(0_u32)

      shared = q4kr_payload_block(shared_payload)
      local = q4kr_payload_block(local_payload)
      shared_codes = Array(UInt8).new(8) { |index| q4kr_scale_code(shared, index) }
      local_codes = Array(UInt8).new(8) { |index| q4kr_scale_code(local, index) }
      shared_codes.uniq.size.should eq(1)
      local_codes.uniq.size.should eq(8)
      8.times do |index|
        q4kr_minimum_code(shared, index).should eq(0_u8)
        q4kr_minimum_code(local, index).should eq(0_u8)
      end
    ensure
      if active = command
        active.discard unless active.committed?
      end
      original.release
      shared_payload.release
      local_payload.release
      status.release
    end
  end

  it "covers F16 underflow/overflow, nonfinite failure, and an exact six-bit rounding tie" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KRangePacker.prepare!

    edge_input = q4kr_concat([q4kr_repeated_input(0x37ec_u16),
                              q4kr_repeated_input(0x37ed_u16),
                              q4kr_repeated_input(0x4c80_u16),
                              q4kr_repeated_input(0x7fc0_u16)])
    edge_original = ML::MetalBuffer.new(edge_input.size.to_i64)
    edge_payload = ML::MetalBuffer.new(Q4KRangePacker.payload_bytes(1024))
    edge_status = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    tie_input = q4kr_ranges_input([0x34f0_u16, 0x37ed_u16, 0_u16, 0_u16,
                                   0_u16, 0_u16, 0_u16, 0_u16])
    tie_original = ML::MetalBuffer.new(tie_input.size.to_i64)
    tie_payload = ML::MetalBuffer.new(Q4KRangePacker.payload_bytes(256))
    commands = [] of ML::Metal::CommandBuffer
    begin
      edge_original.write_bytes(edge_input.to_unsafe, edge_input.size)
      edge_status.contents.as(UInt32*).value = 0_u32
      edge_command = ML::Metal::CommandBuffer.new
      commands << edge_command
      edge_encoder = ML::Metal::ComputeEncoder.new(edge_command)
      Q4KRangePacker.pack(edge_encoder, edge_original, edge_payload, edge_status, 1024, 1)
      edge_encoder.end_encoding
      edge_command.commit_and_wait
      edge_command.completed_successfully?.should be_true

      edge_status.contents.as(UInt32*).value.should_not eq(0_u32)
      q4kr_payload_block(edge_payload, 0).should eq(Bytes.new(144, 0_u8))
      valid_boundary = q4kr_payload_block(edge_payload, 1)
      valid_boundary[0].should eq(1_u8)
      valid_boundary[1].should eq(0_u8)
      valid_boundary.should_not eq(Bytes.new(144, 0_u8))
      q4kr_payload_block(edge_payload, 2).should eq(Bytes.new(144, 0_u8))
      q4kr_payload_block(edge_payload, 3).should eq(Bytes.new(144, 0_u8))

      tie_original.write_bytes(tie_input.to_unsafe, tie_input.size)
      tie_command = ML::Metal::CommandBuffer.new
      commands << tie_command
      tie_encoder = ML::Metal::ComputeEncoder.new(tie_command)
      Q4KRangePacker.pack(tie_encoder, tie_original, tie_payload, edge_status, 256, 8)
      tie_encoder.end_encoding
      tie_command.commit_and_wait
      tie_command.completed_successfully?.should be_true

      # A later valid pack must preserve an earlier sticky error word.
      edge_status.contents.as(UInt32*).value.should_not eq(0_u32)
      tie_block = q4kr_payload_block(tie_payload)
      tie_block[0].should eq(1_u8)
      tie_block[1].should eq(0_u8)
      q4kr_scale_code(tie_block, 0).should eq(1_u8)
      q4kr_scale_code(tie_block, 1).should eq(32_u8)
      q4kr_scale_code(tie_block, 2).should eq(0_u8)
      q4kr_minimum_code(tie_block, 0).should eq(0_u8)
      tie_block[16, 32].should eq(Bytes.new(32, 0xf8_u8))
      tie_block[48, 96].should eq(Bytes.new(96, 0_u8))
    ensure
      commands.each { |active| active.discard unless active.committed? }
      edge_original.release
      edge_payload.release
      edge_status.release
      tie_original.release
      tie_payload.release
    end
  end
end

describe "Q4_K range pack preflight" do
  it "rejects bad geometry, unsupported ranges, undersized storage, and all buffer aliases" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    Q4KRangePacker.prepare!

    input = ML::MetalBuffer.new(512_i64)
    payload = ML::MetalBuffer.new(144_i64)
    status = ML::MetalBuffer.new(4_i64)
    small_input = ML::MetalBuffer.new(511_i64)
    small_payload = ML::MetalBuffer.new(143_i64)
    small_status = ML::MetalBuffer.new(3_i64)
    shared = ML::MetalBuffer.new(512_i64)
    shared_small = ML::MetalBuffer.new(144_i64)
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    ended = false
    begin
      status.contents.as(UInt32*).value = 0_u32
      expect_raises(ArgumentError) { Q4KRangePacker.pack(encoder, input, payload, status, 255, 1) }
      expect_raises(ArgumentError) { Q4KRangePacker.pack(encoder, input, payload, status, 256, 2) }
      expect_raises(ArgumentError) { Q4KRangePacker.pack(encoder, small_input, payload, status, 256, 1) }
      expect_raises(ArgumentError) { Q4KRangePacker.pack(encoder, input, small_payload, status, 256, 1) }
      expect_raises(ArgumentError) { Q4KRangePacker.pack(encoder, input, payload, small_status, 256, 1) }

      expect_raises(ArgumentError) { Q4KRangePacker.pack(encoder, shared, shared, status, 256, 1) }
      expect_raises(ArgumentError) { Q4KRangePacker.pack(encoder, shared, payload, shared, 256, 1) }
      expect_raises(ArgumentError) { Q4KRangePacker.pack(encoder, input, shared_small, shared_small, 256, 1) }

      encoder.end_encoding
      ended = true
      command.discard
    ensure
      encoder.end_encoding unless ended
      command.discard unless command.committed?
      {input, payload, status, small_input, small_payload, small_status, shared, shared_small}.each(&.release)
    end
  end
end
