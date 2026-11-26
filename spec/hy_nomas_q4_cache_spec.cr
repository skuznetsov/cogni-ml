require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_q4_cache"

describe ML::Video::HyNomasQ4Cache do
  it "decodes all eight Q4_K local ranges directly into BF16 on the existing chain" do
    scales = [1, 2, 3, 4, 5, 16, 32, 63]
    mins = [63, 32, 16, 5, 4, 3, 2, 1]
    packed = Bytes.new(144, 0_u8)
    IO::ByteFormat::LittleEndian.encode(0x2c00_u16, packed[0, 2]) # d = 1/16
    IO::ByteFormat::LittleEndian.encode(0x2800_u16, packed[2, 2]) # min = 1/32
    4.times do |i|
      packed[4 + i] = (scales[i] | ((scales[i + 4] >> 4) << 6)).to_u8
      packed[8 + i] = (mins[i] | ((mins[i + 4] >> 4) << 6)).to_u8
      packed[12 + i] = ((scales[i + 4] & 15) | ((mins[i + 4] & 15) << 4)).to_u8
    end
    128.times { |i| packed[16 + i] = ((i % 16) | ((15 - i % 16) << 4)).to_u8 }
    source = ML::MetalBuffer.new(144_i64)
    output = ML::MetalBuffer.new(512_i64)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      source.write_bytes(packed.to_unsafe, packed.size)
      ML::Video::HyNomasQ4Cache.prepare_decode!
      chain = ML::Metal::MPSCommandChain.new
      before = ML::MetalBuffer.video_spec_allocations
      chain.encoder.set_pipeline(ML::Video::HyNomasQ4Cache.q4_pipeline)
        .set_buffer(source, 0).set_buffer(output, 1, ML::Metal::BufferAccess::Write)
        .set_value(256_u32, 2).dispatch_1d(256)
      ML::MetalBuffer.video_spec_allocations.should eq(before)
      chain.wait_count.should eq(0)
      chain.commit_and_wait
      chain.wait_count.should eq(1)
      # Postterminal diagnostics only; never used as an inference handoff.
      actual = Bytes.new(512)
      actual.to_unsafe.copy_from(output.contents.as(UInt8*), 512)
      256.times do |i|
        group, rem = i // 64, i % 64
        local = 2 * group + (rem >= 32 ? 1 : 0)
        byte = packed[16 + group * 32 + rem % 32]
        q = rem < 32 ? byte & 15 : byte >> 4
        value = scales[local].to_f32 / 16 * q.to_f32 - mins[local].to_f32 / 32
        bits = value.unsafe_as(UInt32)
        expected = ((bits &+ 0x7fff_u32 &+ ((bits >> 16) & 1_u32)) >> 16).to_u16
        IO::ByteFormat::LittleEndian.decode(UInt16, actual[i * 2, 2]).should eq(expected)
      end
    ensure
      chain.try(&.commit_and_wait)
      chain.try(&.release)
      source.release
      output.release
    end
  end
end
