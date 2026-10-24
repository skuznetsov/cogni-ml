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

private class FailingVideoCheckpoint < ML::Video::CheckpointFile
  @copies = 0

  def copy_raw(info : ML::Video::CheckpointTensor, destination : Bytes) : Nil
    @copies += 1
    raise IO::Error.new("Injected checkpoint copy failure") if @copies == 2
    super
  end
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

  it "plans original BF16 residency without staging and rejects incomplete or multiply bound manifests" do
    header = %({"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}})
    with_video_checkpoint(header, Bytes.new(6)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      begin
        bindings = [
          ML::Video::WeightBinding.new("weight", "w", [2_i64]),
          ML::Video::WeightBinding.new("bias", "b", [1_i64]),
        ]
        allocations = ML::MetalBuffer.video_spec_allocations
        plan = ML::Video::WeightPlan.new([file], bindings, 14_i64, 8_i64,
          resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        plan.resident_dtype.should eq(ML::Video::WeightDType::BF16)
        plan.destination_bytes.should eq(6)
        plan.staging_bytes.should eq(0)
        plan.peak_bytes.should eq(14)
        expect_raises(ArgumentError, /budget/) do
          ML::Video::WeightPlan.new([file], bindings, 13_i64, 8_i64, resident_dtype: ML::Video::WeightDType::BF16)
        end
        expect_raises(ArgumentError, /complete/) do
          ML::Video::WeightPlan.new([file], bindings[0, 1], 64_i64, resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        end
        duplicate = ML::Video::WeightBinding.new("another", "w", [2_i64])
        expect_raises(ArgumentError, /source/) do
          ML::Video::WeightPlan.new([file], [bindings[0], duplicate], 64_i64, resident_dtype: ML::Video::WeightDType::BF16)
        end
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      ensure
        file.close
      end
    end
  end

  it "rejects F32 and F16 sources in native BF16 mode rather than quantizing them" do
    {"F32" => 8, "F16" => 4}.each do |dtype, bytes|
      with_video_checkpoint(checkpoint_header(dtype: dtype, offsets: "[0,#{bytes}]"), Bytes.new(bytes)) do |path|
        file = ML::Video::CheckpointFile.new(path)
        begin
          binding = ML::Video::WeightBinding.new("weight", "w", [2_i64])
          allocations = ML::MetalBuffer.video_spec_allocations
          expect_raises(ArgumentError, /original BF16/) do
            ML::Video::WeightPlan.new([file], [binding], 64_i64, resident_dtype: ML::Video::WeightDType::BF16)
          end
          ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        ensure
          file.close
        end
      end
    end
  end

  it "admits F32-to-BF16 only by opt-in and budgets only converted source staging" do
    header = %({"f32_a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"bf16":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]},"f32_b":{"dtype":"F32","shape":[2],"data_offsets":[12,20]}})
    with_video_checkpoint(header, Bytes.new(20)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      begin
        bindings = [
          ML::Video::WeightBinding.new("a", "f32_a", [2_i64]),
          ML::Video::WeightBinding.new("b", "bf16", [2_i64]),
          ML::Video::WeightBinding.new("c", "f32_b", [2_i64]),
        ]
        allocations = ML::MetalBuffer.video_spec_allocations
        expect_raises(ArgumentError, /original BF16/) do
          ML::Video::WeightPlan.new([file], bindings, 128_i64,
            resident_dtype: ML::Video::WeightDType::BF16)
        end
        expect_raises(ArgumentError, /budget/) do
          ML::Video::WeightPlan.new([file], bindings, 31_i64, 4_i64,
            resident_dtype: ML::Video::WeightDType::BF16,
            allow_f32_to_bf16: true)
        end
        plan = ML::Video::WeightPlan.new([file], bindings, 32_i64, 4_i64,
          resident_dtype: ML::Video::WeightDType::BF16,
          allow_f32_to_bf16: true)
        plan.allow_f32_to_bf16.should be_true
        plan.destination_bytes.should eq(12)
        plan.staging_bytes.should eq(16)
        plan.peak_bytes.should eq(32)
        plan.entries.map(&.raw_offset).should eq([0_i64, 0_i64, 8_i64])
        expect_raises(ArgumentError, /BF16 residency/) do
          ML::Video::WeightPlan.new([file], [bindings[0]], 128_i64,
            allow_f32_to_bf16: true)
        end
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      ensure
        file.close
      end
    end

    with_video_checkpoint(checkpoint_header(dtype: "F16", offsets: "[0,4]"), Bytes.new(4)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      begin
        binding = ML::Video::WeightBinding.new("weight", "w", [2_i64])
        allocations = ML::MetalBuffer.video_spec_allocations
        expect_raises(ArgumentError, /requires F32 source/) do
          ML::Video::WeightPlan.new([file], [binding], 64_i64,
            resident_dtype: ML::Video::WeightDType::BF16,
            allow_f32_to_bf16: true)
        end
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      ensure
        file.close
      end
    end
  end
end

describe ML::Video::PreparedWeights do
  it "rolls back partial BF16 setup while preserving caller-owned buffers" do
    header = %({"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}})
    with_video_checkpoint(header, Bytes.new(6)) do |path|
      file = FailingVideoCheckpoint.new(path)
      arena = ML::Video::Arena.new
      begin
        original = arena.upload("existing", [7_f32])
        bindings = [
          ML::Video::WeightBinding.new("weight", "w", [2_i64]),
          ML::Video::WeightBinding.new("bias", "b", [1_i64]),
        ]
        plan = ML::Video::WeightPlan.new([file.as(ML::Video::CheckpointFile)], bindings, 10_i64, 4_i64,
          resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        baseline = ML::MetalBuffer.stats
        expect_raises(IO::Error, /Injected/) { ML::Video::PreparedWeights.new(plan, arena) }
        rolled_back = ML::MetalBuffer.stats
        rolled_back[:live_buffers].should eq(baseline[:live_buffers])
        rolled_back[:live_bytes].should eq(baseline[:live_bytes])
        arena.buffers.keys.should eq(["existing"])
        original.valid?.should be_true
        original.read(1).should eq([7_f32])
        arena.dtype("existing").should eq(ML::Video::WeightDType::F32)
        # Rollback must remove dtype entries as well as buffers.
        arena.alloc("weight", 1)
        arena.dtype("weight").should eq(ML::Video::WeightDType::F32)
      ensure
        arena.release
        file.close
      end
    end
  end

  it "rejects released native BF16 destinations before encoding any work" do
    with_video_checkpoint(checkpoint_header(dtype: "BF16", offsets: "[0,4]"), Bytes.new(4)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      arena = ML::Video::Arena.new
      prepared = nil.as(ML::Video::PreparedWeights?)
      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      begin
        bindings = [ML::Video::WeightBinding.new("weight", "w", [2_i64])]
        plan = ML::Video::WeightPlan.new([file], bindings, 4_i64,
          resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        prepared = ML::Video::PreparedWeights.new(plan, arena)
        arena["weight"].release
        expect_raises(ArgumentError, /resident video buffer/) { prepared.encode(encoder) }
      ensure
        encoder.end_encoding
        command.discard
        prepared.try(&.release)
        arena.release
        file.close
      end
    end
  end

  it "rejects a replaced destination before either BF16 validation or F32 decode dispatch" do
    {ML::Video::WeightDType::F32, ML::Video::WeightDType::BF16}.each do |dtype|
      with_video_checkpoint(checkpoint_header(dtype: "BF16", offsets: "[0,4]"), Bytes.new(4)) do |path|
        file = ML::Video::CheckpointFile.new(path)
        arena = ML::Video::Arena.new
        prepared = nil.as(ML::Video::PreparedWeights?)
        baseline = ML::MetalBuffer.stats
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        begin
          bindings = [ML::Video::WeightBinding.new("weight", "w", [2_i64])]
          plan = ML::Video::WeightPlan.new([file], bindings, 12_i64,
            resident_dtype: dtype, require_all: true)
          prepared = ML::Video::PreparedWeights.new(plan, arena)
          original = arena["weight"]
          replacement = ML::MetalBuffer.new(8_i64)
          arena.buffers["weight"] = replacement
          allocations = ML::MetalBuffer.video_spec_allocations
          expect_raises(ArgumentError, /destination was replaced/) { prepared.encode(encoder) }
          ML::MetalBuffer.video_spec_allocations.should eq(allocations)
          prepared.release
          arena.release
          original.valid?.should be_false
          replacement.valid?.should be_false
          released = ML::MetalBuffer.stats
          released[:live_buffers].should eq(baseline[:live_buffers])
          released[:live_bytes].should eq(baseline[:live_bytes])
        ensure
          encoder.end_encoding
          command.discard
          prepared.try(&.release)
          arena.release
          file.close
        end
      end
    end
  end

  it "counts replaced or deleted typed owners against the checkpoint reserve before allocation" do
    {false, true}.each do |replace|
      with_video_checkpoint(checkpoint_header(dtype: "BF16", offsets: "[0,4]"), Bytes.new(4)) do |path|
        file = ML::Video::CheckpointFile.new(path)
        arena = ML::Video::Arena.new
        baseline = ML::MetalBuffer.stats
        replacement = nil.as(ML::MetalBuffer?)
        begin
          original = arena.alloc_weight("hidden", 2, ML::Video::WeightDType::BF16)
          if replace
            replacement = ML::MetalBuffer.new(8_i64)
            arena.buffers["hidden"] = replacement
          else
            arena.buffers.delete("hidden")
          end
          binding = ML::Video::WeightBinding.new("weight", "w", [2_i64])
          reserve = replace ? 8_i64 : 0_i64
          plan = ML::Video::WeightPlan.new([file], [binding], 64_i64, reserve,
            resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
          allocations = ML::MetalBuffer.video_spec_allocations
          expect_raises(ArgumentError, /reserve/) { ML::Video::PreparedWeights.new(plan, arena) }
          ML::MetalBuffer.video_spec_allocations.should eq(allocations)
          arena.release
          original.valid?.should be_false
          replacement.try { |buffer| buffer.valid?.should be_false }
          released = ML::MetalBuffer.stats
          released[:live_buffers].should eq(baseline[:live_buffers])
          released[:live_bytes].should eq(baseline[:live_bytes])
        ensure
          arena.release
          file.close
        end
      end
    end
  end

  it "rejects a deleted typed destination before allocating F32 staging" do
    with_video_checkpoint(checkpoint_header, Bytes.new(8)) do |path|
      file = ML::Video::CheckpointFile.new(path)
      arena = ML::Video::Arena.new
      begin
        original = arena.alloc_weight("weight", 2, ML::Video::WeightDType::BF16)
        arena.buffers.delete("weight")
        binding = ML::Video::WeightBinding.new("weight", "w", [2_i64])
        plan = ML::Video::WeightPlan.new([file], [binding], 64_i64, 4_i64)
        allocations = ML::MetalBuffer.video_spec_allocations
        expect_raises(ArgumentError, /already exists/) { ML::Video::PreparedWeights.new(plan, arena) }
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        original.valid?.should be_true
      ensure
        arena.release
        file.close
      end
    end
  end

  it "loads raw BF16 directly into final GPU buffers and chains projections without a parameter mirror" do
    header = %({"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}})
    raw = IO::Memory.new
    [0x3f80_u16, 0xc000_u16, 0x3f00_u16, 0x4080_u16, 0x3e80_u16, 0xbf00_u16].each do |word|
      raw.write_bytes(word, IO::ByteFormat::LittleEndian)
    end
    with_video_checkpoint(header, raw.to_slice) do |path|
      file = ML::Video::CheckpointFile.new(path)
      arena = ML::Video::Arena.new
      prepared = nil.as(ML::Video::PreparedWeights?)
      begin
        bindings = [
          ML::Video::WeightBinding.new("w", "w", [2_i64, 2_i64]),
          ML::Video::WeightBinding.new("b", "b", [2_i64]),
        ]
        plan = ML::Video::WeightPlan.new([file], bindings, 36_i64, 24_i64,
          resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        before_setup = ML::MetalBuffer.video_spec_allocations
        prepared = ML::Video::PreparedWeights.new(plan, arena)
        ML::MetalBuffer.video_spec_allocations.should eq(before_setup + 2)
        arena.buffers.values.sum(0_i64, &.size).should eq(12_i64)
        arena.dtype("w").should eq(ML::Video::WeightDType::BF16)
        arena.dtype("b").should eq(ML::Video::WeightDType::BF16)
        Bytes.new(arena["w"].contents.as(UInt8*), 8).should eq(raw.to_slice[0, 8])
        Bytes.new(arena["b"].contents.as(UInt8*), 4).should eq(raw.to_slice[8, 4])
        x = arena.upload("x", [3_f32, -1_f32])
        y = arena.alloc("y", 2)
        z = arena.alloc("z", 2)
        allocations = ML::MetalBuffer.video_spec_allocations
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        2.times do
          prepared.encode(encoder)
          ML::Video::MetalOps.linear(encoder, x, arena["w"], arena["b"], y, 1, 2, 2,
            weight_dtype: arena.dtype("w"), bias_dtype: arena.dtype("b"))
          ML::Video::MetalOps.linear(encoder, y, arena["w"], arena["b"], z, 1, 2, 2,
            weight_dtype: arena.dtype("w"), bias_dtype: arena.dtype("b"))
        end
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        encoder.end_encoding
        command.commit_and_wait
        y.read(2).should eq([5.25_f32, -3_f32])
        z.read(2).should eq([11.5_f32, -9.875_f32])
        prepared.release
        arena["w"].valid?.should be_true
        expect_raises(ArgumentError, /released/) { prepared.encode(encoder) }
      ensure
        prepared.try(&.release)
        arena.release
        file.close
      end
    end
  end

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

  it "rounds opt-in F32 sources to BF16 with ties-to-even and chains projections in one encoder" do
    unless ML::Metal::Device.init!
      pending "Metal required"
      next
    end
    header = %({"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]},"rounds":{"dtype":"F32","shape":[8],"data_offsets":[24,56]},"direct":{"dtype":"BF16","shape":[2],"data_offsets":[56,60]}})
    raw = IO::Memory.new
    [0x3f800000_u32, 0x00000000_u32, 0x00000000_u32, 0x3f800000_u32,
     0x00000000_u32, 0x00000000_u32,
     0x3f808000_u32, 0x3f808001_u32, 0xbf808000_u32,
     0x3f818000_u32, 0xbf818000_u32,
     0x80000000_u32, 0x7f800000_u32, 0x7f812345_u32].each do |word|
      raw.write_bytes(word, IO::ByteFormat::LittleEndian)
    end
    [0x3f80_u16, 0xc000_u16].each { |word| raw.write_bytes(word, IO::ByteFormat::LittleEndian) }
    with_video_checkpoint(header, raw.to_slice) do |path|
      file = ML::Video::CheckpointFile.new(path)
      arena = ML::Video::Arena.new
      weights = nil.as(ML::Video::PreparedWeights?)
      command = nil.as(ML::Metal::CommandBuffer?)
      encoder = nil.as(ML::Metal::ComputeEncoder?)
      encoder_ended = false
      begin
        bindings = [
          ML::Video::WeightBinding.new("w", "w", [2_i64, 2_i64]),
          ML::Video::WeightBinding.new("b", "b", [2_i64]),
          ML::Video::WeightBinding.new("rounds", "rounds", [8_i64]),
          ML::Video::WeightBinding.new("direct", "direct", [2_i64]),
        ]
        plan = ML::Video::WeightPlan.new([file], bindings, 112_i64, 24_i64,
          resident_dtype: ML::Video::WeightDType::BF16,
          allow_f32_to_bf16: true, require_all: true)
        plan.staging_bytes.should eq(56)
        plan.peak_bytes.should eq(112)
        weights = ML::Video::PreparedWeights.new(plan, arena)
        Bytes.new(arena["direct"].contents.as(UInt8*), 4).should eq(raw.to_slice[56, 4])
        x = arena.upload("x", [3_f32, -1_f32])
        y = arena.alloc("y", 2)
        z = arena.alloc("z", 2)
        allocations = ML::MetalBuffer.video_spec_allocations
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command.not_nil!)
        weights.encode(encoder.not_nil!)
        ML::Video::MetalOps.linear(encoder.not_nil!, x, arena["w"], arena["b"], y, 1, 2, 2,
          weight_dtype: ML::Video::WeightDType::BF16,
          bias_dtype: ML::Video::WeightDType::BF16)
        ML::Video::MetalOps.linear(encoder.not_nil!, y, arena["w"], arena["b"], z, 1, 2, 2,
          weight_dtype: ML::Video::WeightDType::BF16,
          bias_dtype: ML::Video::WeightDType::BF16)
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        encoder.not_nil!.end_encoding
        encoder_ended = true
        command.not_nil!.commit_and_wait
        y.read(2).should eq([3_f32, -1_f32])
        z.read(2).should eq([3_f32, -1_f32])
        rounded_bytes = Bytes.new(16)
        arena["rounds"].read_bytes(rounded_bytes.to_unsafe, rounded_bytes.size)
        rounded = Array(UInt16).new(8) do |i|
          IO::ByteFormat::LittleEndian.decode(UInt16, rounded_bytes[i * 2, 2])
        end
        rounded.should eq([0x3f80_u16, 0x3f81_u16, 0xbf80_u16,
                           0x3f82_u16, 0xbf82_u16, 0x8000_u16,
                           0x7f80_u16, 0x7fc1_u16])
      ensure
        encoder.try(&.end_encoding) unless encoder_ended
        command.try { |active| active.discard unless active.committed? }
        weights.try(&.release)
        arena.release
        file.close
      end
    end
  end
end
