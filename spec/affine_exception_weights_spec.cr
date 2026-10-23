require "spec"
require "../src/ml/video/affine_exception_weights"

alias AffineExceptionCodec = ML::Video::AffineExceptionWeights

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
    geometry.status_words.should eq(8)
    geometry.status_error_words.should eq([0, 1, 2, 3])

    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_SPARSE_WEIGHT).should eq(414_719)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_SPARSE_ACTIVATION).should eq(414_719)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_K_CHANNEL).should eq(79)
    geometry.target_count(ML::Video::AffineExceptionWeights::MODE_N_ROW).should eq(286)
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
      AffineExceptionCodec.decode(encoder, payload, stage, status, geometry.n, geometry.k,
        AffineExceptionCodec::MODE_SPARSE_WEIGHT)
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
      AffineExceptionCodec.decode(encoder, payload, stage, status, geometry.n, geometry.k,
        AffineExceptionCodec::MODE_SPARSE_WEIGHT)
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
