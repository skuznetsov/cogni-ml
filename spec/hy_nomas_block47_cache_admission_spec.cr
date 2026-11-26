require "spec"
require "../examples/support/hy_nomas_block47_cache_probe"

private def with_block47_oracle_file(dtype : String, index : Int32, &)
  width = dtype == "BF16" ? 2 : 4
  header = JSON.build do |json|
    json.object do
      32.times do |slot|
        json.field "double_blocks.#{index}.weight#{slot}" do
          json.object do
            json.field "dtype", dtype
            json.field "shape", [1]
            json.field "data_offsets", [slot * width, (slot + 1) * width]
          end
        end
      end
    end
  end
  temporary = File.tempfile("hy-block47-admission", ".safetensors")
  file = nil.as(ML::Video::CheckpointFile?)
  begin
    temporary.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    temporary << header
    temporary.write(Bytes.new(32 * width))
    temporary.flush
    file = ML::Video::CheckpointFile.new(temporary.path)
    yield file
  ensure
    file.try(&.close)
    temporary.close
    temporary.delete
  end
end

describe HyNomasBlock47CacheProbe do
  it "rejects a short conversion budget or unadmitted block before file access and Metal" do
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    {4, 5, 6, 7}.each do |index|
      expect_raises(ArgumentError, /budget mismatch/) do
        HyNomasBlock47CacheProbe::Owner.new("missing", "a" * 64, "missing2", "b" * 64,
          index, HyNomasBlock47CacheProbe::SOURCE_BYTES + HyNomasBlock47CacheProbe::DESTINATION_BYTES - 1)
      end
    end
    expect_raises(ArgumentError, /index/) do
      HyNomasBlock47CacheProbe::Owner.new("missing", "a" * 64, "missing2", "b" * 64, 8, 1024_i64**3)
    end
    ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
  end

  it "requires oracle index, BF16 dtype and the complete source inventory before Metal" do
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    {4, 5, 6, 7}.each do |index|
      with_block47_oracle_file("F32", index) do |source|
        bindings = source.tensors.values.map { |info| ML::Video::WeightBinding.new(info.name, info.name, info.shape) }
        plan = ML::Video::WeightPlan.new([source], bindings, 1024_i64,
          resident_dtype: ML::Video::WeightDType::BF16, allow_f32_to_bf16: true, require_all: true)
        with_block47_oracle_file("BF16", index) do |oracle|
          HyNomasBlock47CacheProbe.validate_oracle!(plan, oracle)
        end
        with_block47_oracle_file("BF16", index == 4 ? 7 : 4) do |wrong|
          expect_raises(ArgumentError, /inventory mismatch/) { HyNomasBlock47CacheProbe.validate_oracle!(plan, wrong) }
        end
        expect_raises(ArgumentError, /inventory mismatch/) { HyNomasBlock47CacheProbe.validate_oracle!(plan, source) }
      end
    end
    ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
  end
end
