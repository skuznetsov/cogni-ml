require "spec"
require "json"
require "../examples/support/hy_nomas_probe_instrumentation"
require "../examples/support/hy_nomas_q4_dit_io"

private def with_q4_finite_checkpoint(&)
  path = File.tempname("hy-nomas-q4-finite", ".safetensors")
  source = nil.as(ML::Video::CheckpointFile?)
  begin
    header = JSON.build do |json|
      json.object do
        [
          {name: "finite_f32", dtype: "F32", shape: [2_i64], first: 0_i64, last: 8_i64},
          {name: "special_f32", dtype: "F32", shape: [3_i64], first: 8_i64, last: 20_i64},
          {name: "overflow_f32", dtype: "F32", shape: [1_i64], first: 20_i64, last: 24_i64},
          {name: "special_bf16", dtype: "BF16", shape: [3_i64], first: 24_i64, last: 30_i64},
        ].each do |tensor|
          json.field tensor[:name] do
            json.object do
              json.field "dtype", tensor[:dtype]
              json.field "shape", tensor[:shape]
              json.field "data_offsets", [tensor[:first], tensor[:last]]
            end
          end
        end
      end
    end
    header += " " * ((8 - header.bytesize % 8) % 8)
    File.open(path, "wb") do |io|
      io.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      io << header
      [1.25_f32, -2.5_f32].each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
      [0x7fc0_0000_u32, 0x7f80_0000_u32, 0xff80_0000_u32].each do |bits|
        io.write_bytes(bits, IO::ByteFormat::LittleEndian)
      end
      io.write_bytes(0x7f7f_ffff_u32, IO::ByteFormat::LittleEndian)
      [0x7fc1_u16, 0x7f80_u16, 0xff80_u16].each do |bits|
        io.write_bytes(bits, IO::ByteFormat::LittleEndian)
      end
    end
    source = ML::Video::CheckpointFile.new(path)
    yield source
  ensure
    source.try(&.close)
    File.delete(path) if File.exists?(path)
  end
end

private def with_q4_finite_chain(source : ML::Video::CheckpointFile,
                                 selected : Array(String), &)
  sizes = {
    "finite_f32"   => 2_i32,
    "special_f32"  => 3_i32,
    "overflow_f32" => 1_i32,
    "special_bf16" => 3_i32,
  }
  bindings = selected.map do |name|
    ML::Video::WeightBinding.new("test.#{name}", name, [sizes[name].to_i64])
  end
  plan = ML::Video::WeightPlan.new([source], bindings, 1024_i64 * 1024,
    reserve_bytes: HyNomasQ4DiTIO::BF16_FINITE_STATUS_BYTES,
    resident_dtype: ML::Video::WeightDType::BF16,
    require_all: false, allow_f32_to_bf16: true)
  arena = ML::Video::Arena.new
  prepared = nil.as(ML::Video::PreparedWeights?)
  checker = nil.as(HyNomasQ4DiTIO::BF16FiniteCheck?)
  chain = nil.as(ML::Metal::MPSCommandChain?)
  begin
    prepared = ML::Video::PreparedWeights.new(plan, arena)
    checker = HyNomasQ4DiTIO::BF16FiniteCheck.new(plan, arena)
    chain = ML::Metal::MPSCommandChain.new
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks
    prepared.encode(chain.encoder)
    checker.encode(chain)
    ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
    ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks)
    chain.wait_count.should eq(0)
    chain.commit_and_wait
    chain.wait_count.should eq(1)
    yield checker, chain, arena
  ensure
    chain.try(&.commit_and_wait)
    checker.try(&.release)
    prepared.try(&.release)
    chain.try(&.release)
    arena.release
  end
end

describe HyNomasQ4DiTIO::BF16FiniteCheck do
  it "reserves one scalar and detects F32 specials, F32-to-BF16 overflow, and BF16 specials after one shared-chain wait" do
    pending! "Requires macOS Metal" unless ML::Metal::Device.init!
    HyNomasQ4DiTIO::BF16_FINITE_STATUS_BYTES.should eq(sizeof(UInt32).to_i64)

    with_q4_finite_checkpoint do |source|
      with_q4_finite_chain(source, ["finite_f32"]) do |checker, chain, _arena|
        checker.validate_after_terminal!(chain)
      end

      with_q4_finite_chain(source, ["special_f32"]) do |checker, chain, _arena|
        expect_raises(ArgumentError, /NaN or infinity/) do
          checker.validate_after_terminal!(chain)
        end
      end

      with_q4_finite_chain(source, ["overflow_f32"]) do |checker, chain, arena|
        expect_raises(ArgumentError, /NaN or infinity/) do
          checker.validate_after_terminal!(chain)
        end
        # The source F32 is finite. Assert the actual rounded resident is
        # positive BF16 infinity, so this case proves the gate scans the
        # post-conversion destination rather than only the source bytes.
        arena.buffers["test.overflow_f32"].contents.as(UInt16*).value.should eq(0x7f80_u16)
      end

      with_q4_finite_chain(source, ["special_bf16"]) do |checker, chain, _arena|
        expect_raises(ArgumentError, /NaN or infinity/) do
          checker.validate_after_terminal!(chain)
        end
      end
    end
  end
end
