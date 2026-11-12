require "spec"
require "../src/ml/gguf/decoder"

# CPU memory with the same buffer interface used by KVCache.
class ML::MetalBuffer
  @fixture_memory = Bytes.empty

  def initialize(@size : Int64, @storage_mode : StorageMode = StorageMode::Shared)
    @valid = true
    @fixture_memory = Bytes.new(@size.to_i)
  end

  def contents : Pointer(Void)
    @fixture_memory.to_unsafe.as(Void*)
  end

  def read(count : Int32) : Array(Float32)
    Array(Float32).new(count) { |index| contents.as(Float32*)[index] }
  end
end

describe ML::GGUF::KVCache do
  it "retains earlier K and V rows when appending and writes transposed V" do
    cache = ML::GGUF::KVCache.new(2, 1, 2)
    cache.append([1_f32, 2_f32], [11_f32, 12_f32])
    cache.append([3_f32, 4_f32], [13_f32, 14_f32])

    cache.k_buf.read(4).should eq([1_f32, 2_f32, 3_f32, 4_f32])
    cache.v_buf.read(4).should eq([11_f32, 12_f32, 13_f32, 14_f32])
    cache.v_t_buf.read(4).should eq([11_f32, 13_f32, 12_f32, 14_f32])
    cache.position.should eq(2)
  end
end
