require "spec"
require "../examples/support/hy_nomas_bf16_cache_probe"

private def with_cache_admission_file(dtype : String, block_index : Int32 = 0, &)
  header = JSON.build do |json|
    json.object do
      32.times do |i|
        json.field "double_blocks.#{block_index}.w#{i}" do
          json.object do
            json.field "dtype", dtype
            json.field "shape", [1]
            width = dtype == "BF16" ? 2 : 4
            json.field "data_offsets", [i * width, (i + 1) * width]
          end
        end
      end
    end
  end
  temporary = File.tempfile("hy-cache-admission", ".safetensors")
  checkpoint = nil.as(ML::Video::CheckpointFile?)
  begin
    temporary.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    temporary << header
    temporary.write(Bytes.new(32 * (dtype == "BF16" ? 2 : 4)))
    temporary.flush
    checkpoint = ML::Video::CheckpointFile.new(temporary.path)
    yield checkpoint
  ensure
    checkpoint.try(&.close)
    temporary.close
    File.delete(temporary.path)
  end
end

describe HyNomasBF16CacheProbe do
  it "keeps a failed cache qualification private and removes its staging" do
    target = File.tempname("failed-hy-cache", ".safetensors")
    staged = ""
    expect_raises(Exception, /injected consumer failure/) do
      HyNomasBF16CacheProbe.qualify_cache!(target) do |path|
        staged = path
        File.write(path, "source verified only")
        raise "injected consumer failure"
        true
      end
    end
    File.exists?(target).should be_false
    File.exists?(staged).should be_false
    Dir.exists?(File.dirname(staged)).should be_false
  end

  it "publishes a cache only after qualification and never replaces a target" do
    target = File.tempname("qualified-hy-cache", ".safetensors")
    staged = ""
    begin
      HyNomasBF16CacheProbe.qualify_cache!(target) do |path|
        staged = path
        File.exists?(target).should be_false
        File.write(path, "qualified")
        true
      end
      File.read(target).should eq("qualified")
      Dir.exists?(File.dirname(staged)).should be_false
      called = false
      expect_raises(ArgumentError, /refusing to replace/) do
        HyNomasBF16CacheProbe.qualify_cache!(target) do |_|
          called = true
          true
        end
      end
      called.should be_false
      File.read(target).should eq("qualified")
    ensure
      File.delete(target) if File.exists?(target)
    end
  end

  it "rejects a false qualification result even when a staged file exists" do
    target = File.tempname("rejected-hy-cache", ".safetensors")
    staged = ""
    expect_raises(Exception, /qualification failed/) do
      HyNomasBF16CacheProbe.qualify_cache!(target) do |path|
        staged = path
        File.write(path, "unqualified")
        false
      end
    end
    File.exists?(target).should be_false
    Dir.exists?(File.dirname(staged)).should be_false
  end

  it "rejects an insufficient aggregate phase budget before Metal" do
    with_cache_admission_file("F32") do |source|
      bindings = source.tensors.values.map do |info|
        ML::Video::WeightBinding.new(info.name, info.name, info.shape)
      end
      plan = ML::Video::WeightPlan.new([source], bindings, 1024_i64,
        resident_dtype: ML::Video::WeightDType::BF16, require_all: true, allow_f32_to_bf16: true)
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      expect_raises(ArgumentError, /named peak .* exceeds budget 1/) do
        HyNomasBF16CacheProbe.admit!(plan, 1_i64)
      end
      ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
    end
  end

  it "rejects same-shaped weights from another block and F32 oracle storage before Metal" do
    with_cache_admission_file("F32") do |source|
      bindings = source.tensors.values.map do |info|
        ML::Video::WeightBinding.new(info.name, info.name, info.shape)
      end
      plan = ML::Video::WeightPlan.new([source], bindings, 1024_i64,
        resident_dtype: ML::Video::WeightDType::BF16, require_all: true, allow_f32_to_bf16: true)
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      with_cache_admission_file("BF16", block_index: 1) do |wrong|
        expect_raises(ArgumentError, /exactly the 32 block-0 tensors/) do
          HyNomasBF16CacheProbe.validate_oracle!(plan, wrong)
        end
      end
      expect_raises(ArgumentError, /dtype mismatch/) do
        HyNomasBF16CacheProbe.validate_oracle!(plan, source)
      end
      ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
    end
  end
end
