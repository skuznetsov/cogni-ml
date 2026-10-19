require "spec"
require "../src/ml/video/checkpoint"
require "./support/video_allocation_counter"

private def with_video_checkpoint(header : String, payload : Bytes = Bytes.empty, &block : String ->)
  file = File.tempfile("video-checkpoint", ".safetensors")
  begin
    file.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    file << header
    file.write(payload)
    file.flush
    yield file.path
  ensure
    path = file.path
    file.close
    File.delete(path) if File.exists?(path)
  end
end

private def checkpoint_header(dtype = "F32", shape = "[2]", offsets = "[0,8]")
  %({"w":{"dtype":"#{dtype}","shape":#{shape},"data_offsets":#{offsets}}})
end

describe ML::Video::CheckpointFile do
  it "admits a scalar and preserves the original descriptor without CPU decoding" do
    with_video_checkpoint(checkpoint_header(shape: "[]", offsets: "[0,4]"), Bytes[0, 0, 128, 63]) do |path|
      file = ML::Video::CheckpointFile.new(path)
      begin
        file.tensors["w"].shape.should eq([] of Int64)
        binding = ML::Video::WeightBinding.new("weight", "w", [] of Int64)
        plan = ML::Video::WeightPlan.new([file], [binding], 32_i64, 8_i64)
        plan.destination_bytes.should eq(4)
        plan.staging_bytes.should eq(4)
        plan.peak_bytes.should eq(16)
      ensure
        file.close
      end
    end
  end

  it "rejects enormous header prefixes before allocating the claimed length" do
    file = File.tempfile("video-invalid-header")
    begin
      file.write_bytes(UInt64::MAX, IO::ByteFormat::LittleEndian)
      file << "{}"
      file.flush
      expect_raises(ArgumentError, /header length/) { ML::Video::CheckpointFile.new(file.path) }
    ensure
      path = file.path
      file.close
      File.delete(path)
    end
  end

  it "rejects ambiguous JSON keys, nested duplicates, and trailing objects" do
    headers = [
      %({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}),
      %({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"data_offsets":[0,8]}}),
      %({"__metadata__":{"key":"one","key":"two"},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}),
      checkpoint_header + " {}",
    ]
    headers.each do |header|
      with_video_checkpoint(header, Bytes.new(8)) do |path|
        expect_raises(Exception) { ML::Video::CheckpointFile.new(path) }
      end
    end
  end

  it "rejects negative, overflowing, zero and mismatched shapes and invalid ranges" do
    headers = [
      checkpoint_header(shape: "[-2,-1]"),
      checkpoint_header(shape: "[9223372036854775807,2]"),
      checkpoint_header(shape: "[0]"),
      checkpoint_header(shape: "[3]"),
      checkpoint_header(offsets: "[-1,7]"),
      checkpoint_header(offsets: "[8,0]"),
      checkpoint_header(offsets: "[4,12]"),
      checkpoint_header(offsets: "[0,8,8]"),
      %({"w":{"dtype":"F32","data_offsets":[0,8]}}),
      checkpoint_header(dtype: "F8_E4M3"),
      %({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"overlap":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
      checkpoint_header(shape: "[1]", offsets: "[0,4]"),
    ]
    headers.each do |header|
      with_video_checkpoint(header, Bytes.new(8)) do |path|
        expect_raises(Exception) { ML::Video::CheckpointFile.new(path) }
      end
    end
  end

  it "rejects missing tensors, wrong shapes, duplicate destinations and insufficient budgets before GPU allocation" do
    with_video_checkpoint(checkpoint_header, Bytes.new(8)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      begin
        binding = ML::Video::WeightBinding.new("weight", "w", [2_i64])
        allocations = ML::MetalBuffer.video_spec_allocations
        expect_raises(ArgumentError, /budget/) { ML::Video::WeightPlan.new([file], [binding], 15_i64) }
        expect_raises(ArgumentError, /budget/) { ML::Video::WeightPlan.new([file], [binding], 16_i64, 1_i64) }
        wrong = ML::Video::WeightBinding.new("weight", "w", [1_i64, 2_i64])
        expect_raises(ArgumentError, /shape mismatch/) { ML::Video::WeightPlan.new([file], [wrong], 1024_i64) }
        wrong_dtype = ML::Video::WeightBinding.new("weight", "w", [2_i64], ML::GGUF::SafeTensorDType::BF16)
        expect_raises(ArgumentError, /dtype mismatch/) { ML::Video::WeightPlan.new([file], [wrong_dtype], 1024_i64) }
        missing = ML::Video::WeightBinding.new("weight", "missing", [2_i64])
        expect_raises(ArgumentError, /Missing/) { ML::Video::WeightPlan.new([file], [missing], 1024_i64) }
        expect_raises(ArgumentError, /Duplicate/) { ML::Video::WeightPlan.new([file], [binding, binding], 1024_i64) }
        expect_raises(ArgumentError, /across shards/) { ML::Video::WeightPlan.new([file, file], [binding], 1024_i64) }
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      ensure
        file.close
      end
    end
  end

  it "rejects a checkpoint changed after metadata admission" do
    with_video_checkpoint(checkpoint_header, Bytes.new(8)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      begin
        File.open(path, "ab") { |io| io << "x" }
        binding = ML::Video::WeightBinding.new("weight", "w", [2_i64])
        expect_raises(ArgumentError, /changed/) { ML::Video::WeightPlan.new([file], [binding], 1024_i64) }
      ensure
        file.close
      end
    end
  end

  it "returns defensive snapshots of admitted metadata and the weight plan" do
    with_video_checkpoint(checkpoint_header, Bytes.new(8)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      begin
        binding = ML::Video::WeightBinding.new("weight", "w", [2_i64])
        plan = ML::Video::WeightPlan.new([file], [binding], 1024_i64)
        binding.shape[0] = 99_i64
        file.tensors["w"].shape[0] = 99_i64
        file.tensors.clear
        file.tensors["w"].shape.should eq([2_i64])
        entries = plan.entries
        entries[0].info.shape[0] = 99_i64
        entries[0].binding.shape[0] = 99_i64
        entries.clear
        plan.entries.size.should eq(1)
        plan.entries[0].info.shape.should eq([2_i64])
        plan.entries[0].binding.shape.should eq([2_i64])
      ensure
        file.close
      end
    end
  end

  it "rejects forged descriptors and undersized destination slices before a raw copy" do
    with_video_checkpoint(checkpoint_header, Bytes.new(8)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      begin
        info = file.tensors["w"]
        destination = Bytes.new(8, 0x7f_u8)
        expect_raises(ArgumentError, /tensor copy/) { file.copy_raw(info, destination[0, 4]) }
        forged = ML::Video::CheckpointTensor.new("w", info.dtype, info.shape, 0_i64, 16_i64, 4_i64)
        expect_raises(ArgumentError, /descriptor/) { file.copy_raw(forged, destination) }
        destination.should eq(Bytes.new(8, 0x7f_u8))
      ensure
        file.close
      end
    end
  end
end

describe ML::Video::PreparedWeights do
  it "rejects an occupied target or unreserved arena before allocating loader buffers" do
    unless ML::Metal::Device.init!
      pending "Metal required"
      next
    end
    with_video_checkpoint(checkpoint_header, Bytes.new(8)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      arena = ML::Video::Arena.new
      begin
        arena.alloc("occupied", 2)
        allocations = ML::MetalBuffer.video_spec_allocations
        binding = ML::Video::WeightBinding.new("weight", "w", [2_i64])
        plan = ML::Video::WeightPlan.new([file], [binding], 1024_i64)
        expect_raises(ArgumentError, /reserve/) { ML::Video::PreparedWeights.new(plan, arena) }
        occupied = ML::Video::WeightBinding.new("occupied", "w", [2_i64])
        plan = ML::Video::WeightPlan.new([file], [occupied], 1024_i64, 8_i64)
        expect_raises(ArgumentError, /already exists/) { ML::Video::PreparedWeights.new(plan, arena) }
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        arena.buffers.keys.should eq(["occupied"])
      ensure
        arena.release
        file.close
      end
    end
  end

  it "converts BF16/F16/F32 on GPU and chains linear in the same encoder without allocating during encode" do
    unless ML::Metal::Device.init!
      pending "Metal required"
      next
    end
    header = %({"bf":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"half":{"dtype":"F16","shape":[2,2],"data_offsets":[8,16]},"bias":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}})
    raw = IO::Memory.new
    [0x3f80_u16, 0xc000_u16, 0x3f00_u16, 0x4080_u16,
     0x3c00_u16, 0xc000_u16, 0x3800_u16, 0x4400_u16].each { |word| raw.write_bytes(word, IO::ByteFormat::LittleEndian) }
    [0.25_f32, -0.5_f32].each { |value| raw.write_bytes(value, IO::ByteFormat::LittleEndian) }
    with_video_checkpoint(header, raw.to_slice) do |path|
      file = ML::Video::CheckpointFile.new(path)
      arena = ML::Video::Arena.new
      weights = nil.as(ML::Video::PreparedWeights?)
      begin
        bindings = [
          ML::Video::WeightBinding.new("w", "bf", [2_i64, 2_i64]),
          ML::Video::WeightBinding.new("half", "half", [2_i64, 2_i64]),
          ML::Video::WeightBinding.new("bias", "bias", [2_i64]),
        ]
        plan = ML::Video::WeightPlan.new([file], bindings, 1024_i64, 16_i64)
        weights = ML::Video::PreparedWeights.new(plan, arena)
        x = arena.upload("x", [3_f32, -1_f32])
        y = arena.alloc("y", 2)
        allocations = ML::MetalBuffer.video_spec_allocations
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        2.times do
          weights.encode(encoder)
          ML::Video::MetalOps.linear(encoder, x, arena["w"], arena["bias"], y, 1, 2, 2)
        end
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        encoder.end_encoding
        command.commit_and_wait
        arena["w"].read(4).should eq([1_f32, -2_f32, 0.5_f32, 4_f32])
        arena["half"].read(4).should eq([1_f32, -2_f32, 0.5_f32, 4_f32])
        y.read(2).should eq([5.25_f32, -3_f32])
        weights.release
        expect_raises(ArgumentError, /released/) { weights.encode(encoder) }
      ensure
        weights.try(&.release)
        arena.release
        file.close
      end
    end
  end
end
