require "spec"
require "digest/sha256"
require "../src/ml/video/bf16_weight_cache"
require "./support/video_allocation_counter"

private class BF16WeightCacheTestChain
  getter wait_count : Int32
  getter commit_calls = 0
  @complete : Bool
  @fail_commit : Bool

  def initialize(@complete : Bool, @wait_count : Int32, @fail_commit : Bool = false)
  end

  def completed? : Bool
    @complete
  end

  def commit_and_wait : Nil
    @commit_calls += 1
    raise "injected native completion error" if @fail_commit
  end
end

private def with_bf16_cache_checkpoint(header : String, payload : Bytes, &block : String ->)
  file = File.tempfile("video-bf16-source", ".safetensors")
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

private def with_bf16_cache_target(&block : String ->)
  file = File.tempfile("video-bf16-target", ".safetensors")
  path = file.path
  file.close
  File.delete(path)
  begin
    yield path
  ensure
    File.delete(path) if File.exists?(path) || File.symlink?(path)
  end
end

private def bf16_cache_header : String
  %({"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}})
end

private def bf16_cache_plan(file : ML::Video::CheckpointFile) : ML::Video::WeightPlan
  bindings = [
    ML::Video::WeightBinding.new("weight_w", "w", [2_i64]),
    ML::Video::WeightBinding.new("weight_b", "b", [1_i64]),
  ]
  ML::Video::WeightPlan.new([file], bindings, 64_i64,
    resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
end

describe ML::Video::BF16WeightCache do
  it "rejects an incomplete or non-single-wait chain before arena access or file creation" do
    payload = Bytes[0x80, 0x3f, 0x00, 0x40, 0x00, 0x3f]
    with_bf16_cache_checkpoint(bf16_cache_header, payload) do |source_path|
      file = ML::Video::CheckpointFile.new(source_path)
      begin
        plan = bf16_cache_plan(file)
        arena = ML::Video::Arena.allocate
        source_sha = "a" * 64
        with_bf16_cache_target do |target|
          allocations = ML::MetalBuffer.video_spec_allocations
          incomplete = BF16WeightCacheTestChain.new(false, 1)
          expect_raises(ArgumentError) do
            ML::Video::BF16WeightCache.write_after_terminal!(target, plan, arena, incomplete, source_sha)
          end
          incomplete.commit_calls.should eq(0)
          File.exists?(target).should be_false

          undrained = BF16WeightCacheTestChain.new(true, 0)
          expect_raises(ArgumentError) do
            ML::Video::BF16WeightCache.write_after_terminal!(target, plan, arena, undrained, source_sha)
          end
          undrained.commit_calls.should eq(0)
          File.exists?(target).should be_false
          ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        end
      ensure
        file.close
      end
    end
  end

  it "calls terminal result validation once without changing the recorded wait count" do
    payload = Bytes[0x80, 0x3f, 0x00, 0x40, 0x00, 0x3f]
    with_bf16_cache_checkpoint(bf16_cache_header, payload) do |source_path|
      file = ML::Video::CheckpointFile.new(source_path)
      begin
        plan = bf16_cache_plan(file)
        arena = ML::Video::Arena.allocate
        with_bf16_cache_target do |target|
          chain = BF16WeightCacheTestChain.new(true, 1, true)
          allocations = ML::MetalBuffer.video_spec_allocations
          expect_raises(Exception, /native completion error/) do
            ML::Video::BF16WeightCache.write_after_terminal!(target, plan, arena, chain, "a" * 64)
          end
          chain.commit_calls.should eq(1)
          chain.wait_count.should eq(1)
          File.exists?(target).should be_false
          ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        end
      ensure
        file.close
      end
    end
  end

  it "rejects partial source bindings even when the plan omitted require_all" do
    header = %({"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}})
    payload = Bytes[0x80, 0x3f, 0x00, 0x40, 0x00, 0x3f]
    with_bf16_cache_checkpoint(header, payload) do |source_path|
      file = ML::Video::CheckpointFile.new(source_path)
      begin
        binding = ML::Video::WeightBinding.new("weight_w", "w", [2_i64])
        plan = ML::Video::WeightPlan.new([file], [binding], 64_i64,
          resident_dtype: ML::Video::WeightDType::BF16)
        arena = ML::Video::Arena.allocate
        with_bf16_cache_target do |target|
          chain = BF16WeightCacheTestChain.new(true, 1)
          expect_raises(ArgumentError, /complete|inventory|binding/) do
            ML::Video::BF16WeightCache.write_after_terminal!(target, plan, arena, chain, "a" * 64)
          end
          chain.commit_calls.should eq(1)
          File.exists?(target).should be_false
        end
      ensure
        file.close
      end
    end
  end

  it "authenticates only pinned, regular, complete BF16 safetensors files" do
    payload = Bytes[0x80, 0x3f, 0x00, 0x40, 0x00, 0x3f]
    with_bf16_cache_checkpoint(bf16_cache_header, payload) do |path|
      expected = Digest::SHA256.hexdigest(File.read(path).to_slice)
      checkpoint = ML::Video::BF16WeightCache.authenticate(path, expected)
      begin
        checkpoint.tensors.keys.sort.should eq(["b", "w"])
        checkpoint.tensors.values.each { |tensor| tensor.dtype.bf16?.should be_true }
      ensure
        checkpoint.close
      end
      expect_raises(ArgumentError, /SHA-256|hash|digest/i) do
        ML::Video::BF16WeightCache.authenticate(path, "0" * 64)
      end
    end
  end

  it "rejects a correctly pinned truncated payload and an F32 cache" do
    header = %({"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})
    with_bf16_cache_checkpoint(header, Bytes[0x80, 0x3f]) do |truncated|
      digest = Digest::SHA256.hexdigest(File.read(truncated).to_slice)
      expect_raises(ArgumentError) { ML::Video::BF16WeightCache.authenticate(truncated, digest) }
    end

    f32_header = %({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})
    with_bf16_cache_checkpoint(f32_header, Bytes[0, 0, 128, 63]) do |path|
      digest = Digest::SHA256.hexdigest(File.read(path).to_slice)
      expect_raises(ArgumentError, /BF16/) { ML::Video::BF16WeightCache.authenticate(path, digest) }
    end
  end

  it "rejects a symlink even when its target has the pinned bytes" do
    payload = Bytes[0x80, 0x3f, 0x00, 0x40, 0x00, 0x3f]
    with_bf16_cache_checkpoint(bf16_cache_header, payload) do |source|
      digest = Digest::SHA256.hexdigest(File.read(source).to_slice)
      with_bf16_cache_target do |link|
        File.symlink(source, link)
        expect_raises(ArgumentError, /regular|symlink|symbolic/i) do
          ML::Video::BF16WeightCache.authenticate(link, digest)
        end
      end
    end
  end

  it "writes original names and BF16 bytes only after a successful terminal wait" do
    unless ML::Metal::Device.init!
      pending "Metal required"
      next
    end

    payload = Bytes[0x80, 0x3f, 0x00, 0x40, 0x00, 0x3f]
    with_bf16_cache_checkpoint(bf16_cache_header, payload) do |source_path|
      file = ML::Video::CheckpointFile.new(source_path)
      arena = ML::Video::Arena.new
      begin
        plan = bf16_cache_plan(file)
        w = arena.alloc_weight("weight_w", 2, ML::Video::WeightDType::BF16)
        b = arena.alloc_weight("weight_b", 1, ML::Video::WeightDType::BF16)
        payload.each_with_index do |byte, index|
          if index < 4
            w.contents.as(UInt8*)[index] = byte
          else
            b.contents.as(UInt8*)[index - 4] = byte
          end
        end

        with_bf16_cache_target do |target|
          chain = BF16WeightCacheTestChain.new(true, 1)
          source_sha = "a" * 64
          file_sha = ML::Video::BF16WeightCache.write_after_terminal!(target, plan, arena, chain, source_sha)
          chain.commit_calls.should eq(1)
          chain.wait_count.should eq(1)
          file_sha.should eq(Digest::SHA256.hexdigest(File.read(target).to_slice))
          File.open(target, "rb") do |io|
            header_size = io.read_bytes(UInt64, IO::ByteFormat::LittleEndian)
            (header_size % 8).should eq(0)
          end

          cached = ML::Video::BF16WeightCache.authenticate(target, file_sha)
          begin
            cached.tensors.keys.sort.should eq(["b", "w"])
            cached.tensors["w"].shape.should eq([2_i64])
            cached.tensors["b"].shape.should eq([1_i64])
            cached.tensors.values.each { |tensor| tensor.dtype.bf16?.should be_true }
            cached_payload = Bytes.new(payload.size)
            cached.tensors.keys.sort.each do |name|
              info = cached.tensors[name]
              bytes = Bytes.new(info.data_bytes.to_i)
              cached.copy_raw(info, bytes)
              offset = name == "w" ? 0 : 4
              cached_payload[offset, bytes.size].copy_from(bytes)
            end
            cached_payload.should eq(payload)
          ensure
            cached.close
          end

          original = File.read(target)
          second = BF16WeightCacheTestChain.new(true, 1)
          expect_raises(Exception) do
            ML::Video::BF16WeightCache.write_after_terminal!(target, plan, arena, second, source_sha)
          end
          second.commit_calls.should eq(1)
          second.wait_count.should eq(1)
          File.read(target).should eq(original)
        end
      ensure
        arena.release
        file.close
      end
    end
  end

  it "serializes same-shaped swapped destinations under the original source names" do
    unless ML::Metal::Device.init!
      pending "Metal required"
      next
    end

    header = %({"alpha":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"beta":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}})
    alpha_bytes = Bytes[0x80, 0x3f, 0x00, 0x40]
    beta_bytes = Bytes[0x00, 0x40, 0x80, 0x40]
    payload = Bytes.new(8)
    payload[0, 4].copy_from(alpha_bytes)
    payload[4, 4].copy_from(beta_bytes)

    with_bf16_cache_checkpoint(header, payload) do |source_path|
      file = ML::Video::CheckpointFile.new(source_path)
      arena = ML::Video::Arena.new
      begin
        # Equal shapes make a source/target-name swap structurally admissible.
        # The destination named alpha contains source beta, and vice versa.
        bindings = [
          ML::Video::WeightBinding.new("alpha", "beta", [2_i64]),
          ML::Video::WeightBinding.new("beta", "alpha", [2_i64]),
        ]
        plan = ML::Video::WeightPlan.new([file], bindings, 32_i64,
          resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
        alpha_destination = arena.alloc_weight("alpha", 2, ML::Video::WeightDType::BF16)
        beta_destination = arena.alloc_weight("beta", 2, ML::Video::WeightDType::BF16)
        alpha_bytes.each_with_index { |byte, index| beta_destination.contents.as(UInt8*)[index] = byte }
        beta_bytes.each_with_index { |byte, index| alpha_destination.contents.as(UInt8*)[index] = byte }

        with_bf16_cache_target do |target|
          chain = BF16WeightCacheTestChain.new(true, 1)
          digest = ML::Video::BF16WeightCache.write_after_terminal!(target, plan, arena, chain, "a" * 64)
          cached = ML::Video::BF16WeightCache.authenticate(target, digest)
          begin
            cached.tensors.keys.sort.should eq(["alpha", "beta"])
            cached.tensors["alpha"].shape.should eq([2_i64])
            cached.tensors["beta"].shape.should eq([2_i64])
            cached_alpha = Bytes.new(4)
            cached_beta = Bytes.new(4)
            cached.copy_raw(cached.tensors["alpha"], cached_alpha)
            cached.copy_raw(cached.tensors["beta"], cached_beta)
            cached_alpha.should eq(alpha_bytes)
            cached_beta.should eq(beta_bytes)
          ensure
            cached.close
          end
        end
      ensure
        arena.release
        file.close
      end
    end
  end
end
