require "spec"
require "../src/ml/video/affine_exception_weights"

alias AffineExceptionCodec = ML::Video::AffineExceptionWeights

private def ax_store_u16(bytes : Bytes, offset : Int32, value : UInt16) : Nil
  2.times do |index|
    bytes[offset + index] = ((value >> (index * 8)) & 0xff_u16).to_u8
  end
end

private def ax_store_u32(bytes : Bytes, offset : Int32, value : UInt32) : Nil
  4.times do |index|
    bytes[offset + index] = ((value >> (index * 8)) & 0xff_u32).to_u8
  end
end

private def ax_expected_joint_payload(geometry : AffineExceptionCodec::Geometry,
                                      quota : Bool) : Bytes
  expected = Bytes.new(geometry.payload_bytes.to_i32, 0_u8)
  offset_bits = (-2.5_f32).unsafe_as(UInt32)
  geometry.groups_256.times do |group|
    base = group * AffineExceptionCodec::GROUP_BYTES
    ax_store_u32(expected, base + 128, 0_u32)
    ax_store_u32(expected, base + 132, offset_bits)
  end

  patch_cursor = 0
  geometry.records_1024.times do |record|
    ax_store_u32(expected, geometry.sparse_offsets_offset.to_i32 + record * 4, patch_cursor.to_u32)
    positions = [] of Int32
    4.times do |in_record|
      group = record * 4 + in_record
      if quota
        positions << in_record * 256
        positions << in_record * 256 + 1 if group < 127
      else
        first = group * 256
        finish = Math.min(first + 256, geometry.patch_count)
        if first < finish
          (first...finish).each do |flat_index|
            positions << in_record * 256 + flat_index - first
          end
        end
      end
    end

    ax_store_u32(expected, geometry.sparse_counts_offset.to_i32 + record * 4, positions.size.to_u32)
    positions.each do |local_position|
      address = geometry.sparse_patches_offset.to_i32 + patch_cursor * 4
      ax_store_u16(expected, address, local_position.to_u16)
      ax_store_u16(expected, address + 2, 0xc020_u16)
      patch_cursor += 1
    end
  end
  raise "Expected joint payload patch count mismatch" unless patch_cursor == geometry.patch_count
  ax_store_u32(expected, geometry.sparse_offsets_offset.to_i32 + geometry.records_1024 * 4,
    patch_cursor.to_u32)
  expected
end

describe ML::Video::AffineExceptionWeights::Geometry do
  it "matches the pinned Q4_K byte budget and sparse exception counts" do
    geometry = ML::Video::AffineExceptionWeights::Geometry.new(13_824, 5_120)

    geometry.weights.should eq(70_778_880_i64)
    geometry.payload_bytes.should eq(39_813_120_i64)
    geometry.base_bytes.should eq(37_601_280_i64)
    geometry.records_1024.should eq(69_120)
    geometry.groups_256.should eq(276_480)
    geometry.patch_count.should eq(414_719)
    geometry.k_channel_count.should eq(79)
    geometry.n_row_count.should eq(286)
    geometry.n_padding_bytes.should eq(5_820_i64)
    geometry.joint_upgrade_count.should eq(138_239)
    geometry.fit_bytes.should eq(13_271_040_i64)
    geometry.candidate_capacity.should eq(276_480)
    geometry.status_words.should eq(8)
    geometry.status_error_words.should eq([0, 1, 2, 3])

    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_SPARSE_WEIGHT).should eq(414_719)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_SPARSE_ACTIVATION).should eq(414_719)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_K_CHANNEL).should eq(79)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_N_ROW).should eq(286)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_JOINT_QUOTA_WEIGHT).should eq(414_719)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_JOINT_GLOBAL_WEIGHT).should eq(414_719)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_JOINT_GLOBAL_ACTIVATION).should eq(414_719)
  end

  it "rejects geometries that cannot be represented by the fixed wire layout" do
    expect_raises(ArgumentError) do
      ML::Video::AffineExceptionWeights::Geometry.new(4, 255)
    end

    expect_raises(ArgumentError) do
      ML::Video::AffineExceptionWeights::Geometry.new(4, 256)
    end

    geometry = ML::Video::AffineExceptionWeights::Geometry.new(13_824, 5_120)
    expect_raises(ArgumentError) do
      geometry.target_count(99)
    end
  end
end

describe "Affine exception joint sparse packing" do
  it "matches exact zero-score quota and global tie payloads" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!

    geometry = AffineExceptionCodec::Geometry.new(256, 256)
    geometry.patch_count.should eq(383)
    geometry.joint_upgrade_count.should eq(127)
    AffineExceptionCodec.prepare!

    buffers = [] of ML::MetalBuffer
    workspace = nil.as(AffineExceptionCodec::Workspace?)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      original = ML::MetalBuffer.new(geometry.weights * 2)
      buffers << original
      original.contents.as(UInt16*).to_slice(geometry.weights.to_i32).fill(0xc020_u16)

      calibration = ML::MetalBuffer.new(geometry.k.to_i64 * 4 * sizeof(Float32))
      buffers << calibration
      calibration_values = calibration.contents.as(Float32*).to_slice(geometry.k * 4)
      calibration_values.size.times do |index|
        calibration_values[index] = ((index % 29) - 14).to_f32 / 16.0_f32
      end

      payload_quota = ML::MetalBuffer.new(geometry.payload_bytes)
      payload_global = ML::MetalBuffer.new(geometry.payload_bytes)
      payload_fold0 = ML::MetalBuffer.new(geometry.payload_bytes)
      payload_fold1 = ML::MetalBuffer.new(geometry.payload_bytes)
      {payload_quota, payload_global, payload_fold0, payload_fold1}.each do |payload|
        buffers << payload
        payload.contents.as(UInt8*).to_slice(geometry.payload_bytes.to_i32).fill(0xa5_u8)
      end

      status = ML::MetalBuffer.new(geometry.status_words.to_i64 * sizeof(UInt32))
      buffers << status
      status.contents.as(UInt32*).to_slice(geometry.status_words).fill(0_u32)
      workspace = AffineExceptionCodec::Workspace.new(geometry)

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      AffineExceptionCodec.pack(encoder, original, calibration, payload_quota, workspace.not_nil!, status,
        geometry.n, geometry.k, AffineExceptionCodec::MODE_JOINT_QUOTA_WEIGHT)
      AffineExceptionCodec.pack(encoder, original, calibration, payload_global, workspace.not_nil!, status,
        geometry.n, geometry.k, AffineExceptionCodec::MODE_JOINT_GLOBAL_WEIGHT)
      AffineExceptionCodec.pack(encoder, original, calibration, payload_fold0, workspace.not_nil!, status,
        geometry.n, geometry.k, AffineExceptionCodec::MODE_JOINT_GLOBAL_ACTIVATION, 0)
      AffineExceptionCodec.pack(encoder, original, calibration, payload_fold1, workspace.not_nil!, status,
        geometry.n, geometry.k, AffineExceptionCodec::MODE_JOINT_GLOBAL_ACTIVATION, 1)
      encoder.end_encoding
      active.commit_and_wait
      active.completed_successfully?.should be_true

      status_words = status.contents.as(UInt32*).to_slice(geometry.status_words)
      status_words.each { |word| word.should eq(0_u32) }

      expected_quota = ax_expected_joint_payload(geometry, true)
      expected_global = ax_expected_joint_payload(geometry, false)
      Bytes.new(payload_quota.contents.as(UInt8*), geometry.payload_bytes.to_i32).should eq(expected_quota)
      Bytes.new(payload_global.contents.as(UInt8*), geometry.payload_bytes.to_i32).should eq(expected_global)
      Bytes.new(payload_fold0.contents.as(UInt8*), geometry.payload_bytes.to_i32).should eq(expected_global)
      Bytes.new(payload_fold1.contents.as(UInt8*), geometry.payload_bytes.to_i32).should eq(expected_global)
      Bytes.new(payload_fold0.contents.as(UInt8*), geometry.payload_bytes.to_i32).should eq(
        Bytes.new(payload_fold1.contents.as(UInt8*), geometry.payload_bytes.to_i32))
    ensure
      if active = command
        active.discard unless active.committed?
      end
      workspace.try(&.release)
      buffers.each(&.release)
    end
  end
end

describe "Affine exception decode preflight" do
  it "rejects packed-payload and writable-status aliasing before encoding" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!

    geometry = AffineExceptionCodec::Geometry.new(256, 256)
    AffineExceptionCodec.prepare!
    shared = ML::MetalBuffer.new(geometry.payload_bytes)
    stage = ML::MetalBuffer.new(geometry.weights * sizeof(Float32))
    shared.contents.as(UInt8*).to_slice(geometry.payload_bytes.to_i32).fill(0xa5_u8)
    stage.contents.as(Float32*).to_slice(geometry.weights.to_i32).fill(19.25_f32)

    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    encoding_ended = false
    begin
      rejected = false
      begin
        AffineExceptionCodec.decode(encoder, shared, stage, shared, geometry.n, geometry.k,
          AffineExceptionCodec::MODE_SPARSE_WEIGHT)
      rescue ArgumentError
        rejected = true
      end
      encoder.end_encoding
      encoding_ended = true
      command.commit_and_wait
      rejected.should be_true
      command.completed_successfully?.should be_true
      shared.contents.as(UInt8*).to_slice(geometry.payload_bytes.to_i32).should eq(
        Bytes.new(geometry.payload_bytes.to_i32, 0xa5_u8))
      stage_values = stage.contents.as(Float32*).to_slice(geometry.weights.to_i32)
      stage_values[0].should eq(19.25_f32)
      stage_values[stage_values.size - 1].should eq(19.25_f32)
    ensure
      encoder.end_encoding unless encoding_ended
      if command.committed?
        command.wait unless command.completed?
      else
        command.discard
      end
      shared.release
      stage.release
    end
  end
end

describe "Affine exception sparse decoder validation" do
  it "poisons malformed input while preserving the sticky error count on a later valid decode" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!

    geometry = AffineExceptionCodec::Geometry.new(256, 256)
    payload = nil.as(ML::MetalBuffer?)
    stage = nil.as(ML::MetalBuffer?)
    status = nil.as(ML::MetalBuffer?)
    command = nil.as(ML::Metal::CommandBuffer?)
    begin
      AffineExceptionCodec.prepare!
      payload = ML::MetalBuffer.new(geometry.payload_bytes)
      stage = ML::MetalBuffer.new(geometry.weights * sizeof(Float32))
      status = ML::MetalBuffer.new(geometry.status_words.to_i64 * sizeof(UInt32))

      records = geometry.records_1024
      counts_offset = geometry.sparse_counts_offset.to_i
      offsets_offset = geometry.sparse_offsets_offset.to_i
      patches_offset = geometry.sparse_patches_offset.to_i
      malformed = Bytes.new(geometry.payload_bytes.to_i, 0_u8)
      last_offset = offsets_offset + records * 4
      malformed[last_offset] = ((geometry.patch_count + 1) & 0xff).to_u8
      malformed[last_offset + 1] = (((geometry.patch_count + 1) >> 8) & 0xff).to_u8
      malformed[last_offset + 2] = 0_u8
      malformed[last_offset + 3] = 0_u8
      payload.write_bytes(malformed.to_unsafe, malformed.size)
      stage.write(Array(Float32).new(geometry.weights.to_i32, 19.25_f32))
      status_words = status.contents.as(UInt32*)
      geometry.status_words.times { |index| status_words[index] = 0_u32 }

      active = ML::Metal::CommandBuffer.new
      command = active
      encoder = ML::Metal::ComputeEncoder.new(active)
      decode_modes = [
        AffineExceptionCodec::MODE_SPARSE_WEIGHT,
        AffineExceptionCodec::MODE_JOINT_QUOTA_WEIGHT,
        AffineExceptionCodec::MODE_JOINT_GLOBAL_WEIGHT,
        AffineExceptionCodec::MODE_JOINT_GLOBAL_ACTIVATION,
      ]
      decode_modes.each do |mode|
        AffineExceptionCodec.decode(encoder, payload, stage, status, geometry.n, geometry.k, mode)
      end
      encoder.end_encoding
      active.commit
      active.wait
      status_words = status.contents.as(UInt32*)
      status_words[AffineExceptionCodec::STATUS_DECODE_ERROR].should be > 0_u32
      stage.read(1)[0].nan?.should be_true
      sticky_errors = status_words[AffineExceptionCodec::STATUS_DECODE_ERROR]

      valid = Bytes.new(geometry.payload_bytes.to_i, 0_u8)
      4.times do |byte|
        valid[counts_offset + byte] = ((geometry.patch_count >> (byte * 8)) & 0xff).to_u8
      end
      records.times do |record|
        offset = record == 0 ? 0 : geometry.patch_count
        offset_address = offsets_offset + record * 4
        valid[offset_address] = (offset & 0xff).to_u8
        valid[offset_address + 1] = ((offset >> 8) & 0xff).to_u8
        valid[offset_address + 2] = ((offset >> 16) & 0xff).to_u8
        valid[offset_address + 3] = ((offset >> 24) & 0xff).to_u8
      end
      final_offset = offsets_offset + records * 4
      4.times do |byte|
        valid[final_offset + byte] = ((geometry.patch_count >> (byte * 8)) & 0xff).to_u8
      end
      geometry.patch_count.times do |position|
        address = patches_offset + position * 4
        valid[address] = (position & 0xff).to_u8
        valid[address + 1] = ((position >> 8) & 0xff).to_u8
      end
      payload.write_bytes(valid.to_unsafe, valid.size)

      second = ML::Metal::CommandBuffer.new
      command = second
      encoder = ML::Metal::ComputeEncoder.new(second)
      decode_modes.each do |mode|
        AffineExceptionCodec.decode(encoder, payload, stage, status, geometry.n, geometry.k, mode)
      end
      encoder.end_encoding
      second.commit
      second.wait
      stage.read(1)[0].should eq(0.0_f32)
      status_words = status.contents.as(UInt32*)
      status_words[AffineExceptionCodec::STATUS_DECODE_ERROR].should eq(sticky_errors)
    ensure
      if active = command
        active.discard unless active.committed?
      end
      payload.try(&.release)
      stage.try(&.release)
      status.try(&.release)
    end
  end
end
