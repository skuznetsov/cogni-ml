require "spec"
require "../src/ml/gguf/qwen35_metal"

# Invalid placeholders exercise accounting without allocating or initializing Metal.
class ML::MetalBuffer
  def self.inventory_placeholder(size : Int64)
    buffer = allocate
    buffer.init_inventory_placeholder(size)
    buffer
  end

  protected def init_inventory_placeholder(@size : Int64)
    @handle = Pointer(Void).null
    @storage_mode = StorageMode::Shared
    @owned = false
    @counted = false
    @valid = false
  end
end

module ML::GGUF::Qwen35Metal::Scratch
  def self.with_inventory_spec(&)
    saved, saved_s = @@pool, @@pool_s
    begin
      @@pool = {} of {Symbol, Int64} => ML::MetalBuffer
      @@pool_s = {} of {String, Int64} => ML::MetalBuffer
      yield
    ensure
      @@pool, @@pool_s = saved, saved_s
    end
  end

  def self.seed_inventory_spec(tag : Symbol | String, bytes : Int64)
    buffer = ML::MetalBuffer.inventory_placeholder(bytes)
    if tag.is_a?(Symbol)
      @@pool[{tag, bytes}] = buffer
    else
      @@pool_s[{tag, bytes}] = buffer
    end
  end
end

class ML::Metal::ComputePipeline
  def self.inventory_placeholder
    new("inventory_placeholder", Pointer(Void).null)
  end
end

class ML::Metal::PipelineCache
  def self.with_inventory_spec(&)
    saved = @@cache
    begin
      @@cache = Hash(String, ML::Metal::ComputePipeline).new
      yield
    ensure
      @@cache = saved
    end
  end
end

class ML::Metal::Device
  def self.inventory_spec_identity
    @@instance.try &.object_id
  end
end

describe "Qwen read-only memory inventory" do
  it "counts exact-size entries, separates tag namespaces, and returns detached data" do
    scratch = ML::GGUF::Qwen35Metal::Scratch
    scratch.with_inventory_spec do
      original_counters = scratch.stats
      scratch.memory_inventory[:retained_bytes].should eq(0)
      scratch.seed_inventory_spec(:x, 64_i64)
      scratch.seed_inventory_spec(:x, 64_i64)
      scratch.seed_inventory_spec(:x, 128_i64)
      scratch.seed_inventory_spec("x", 32_i64)
      stats = scratch.memory_inventory
      stats[:entries].should eq(3)
      stats[:retained_bytes].should eq(224)
      stats[:by_tag]["symbol:x"].should eq({entries: 2, bytes: 192_i64})
      stats[:by_tag]["string:x"].should eq({entries: 1, bytes: 32_i64})
      stats[:by_tag].clear
      scratch.memory_inventory[:entries].should eq(3)
      scratch.stats.should eq(original_counters)
    end
  end

  it "counts only cached pipeline keys without compiling anything" do
    cache = ML::Metal::PipelineCache
    cache.with_inventory_spec do
      fake = ML::Metal::ComputePipeline.inventory_placeholder
      2.times { cache.get("inventory_a") { fake } }
      cache.get("inventory_b") { fake }
      cache.entry_count.should eq(2)
    end
  end

  it "does not initialize Metal merely to inspect allocated bytes" do
    identity = ML::Metal::Device.inventory_spec_identity
    value = ML::Metal::Device.current_allocated_size_if_initialized
    ML::Metal::Device.inventory_spec_identity.should eq(identity)
    value.should be_nil if identity.nil?
  end

  it "emits qualified counters only when explicitly enabled, without initializing Metal" do
    previous = ENV["QWEN35_MEMORY_TRACE"]?
    output = IO::Memory.new
    identity = ML::Metal::Device.inventory_spec_identity
    expected_entries = ML::GGUF::Qwen35Metal::Scratch.memory_inventory[:entries]
    begin
      ENV.delete("QWEN35_MEMORY_TRACE")
      ML::GGUF::Qwen35Metal.trace_memory("spec", 0, 4, output: output)
      output.empty?.should be_true
      ENV["QWEN35_MEMORY_TRACE"] = "1"
      ML::GGUF::Qwen35Metal.trace_memory("spec", 0, 4, output: output)
      row = JSON.parse(output.to_s)
      row["event"].as_s.should eq("qwen_memory")
      row["scratch_entries"].as_i.should eq(expected_entries)
      row["metal_allocated_bytes"].raw.should be_nil if identity.nil?
      ML::Metal::Device.inventory_spec_identity.should eq(identity)
    ensure
      if previous
        ENV["QWEN35_MEMORY_TRACE"] = previous
      else
        ENV.delete("QWEN35_MEMORY_TRACE")
      end
    end
  end
end
