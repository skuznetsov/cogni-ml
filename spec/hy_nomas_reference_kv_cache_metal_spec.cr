require "spec"
require "../src/ml/video/hy_nomas_reference_kv_cache"
require "./support/video_allocation_counter"

HY_NOMAS_REFERENCE_KV_CACHE_OVERWRITE_SOURCE = <<-METAL
#include <metal_stdlib>
using namespace metal;

kernel void hy_nomas_reference_kv_cache_spec_overwrite(
    device float *buffer [[buffer(0)]], constant uint &count [[buffer(1)]],
    constant float &value [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) buffer[id] = value;
}
METAL

private def hy_nomas_reference_kv_cache_roundtrip(value : Float32) : Float32
  bits = value.unsafe_as(UInt32)
  magnitude = bits & 0x7fff_ffff_u32
  if magnitude > 0x7f80_0000_u32
    nan_bf16 = (bits >> 16) | 0x40_u32
    (nan_bf16 << 16).unsafe_as(Float32)
  else
    rounded = bits + 0x7fff_u32 + ((bits >> 16) & 1_u32)
    (rounded & 0xffff_0000_u32).unsafe_as(Float32)
  end
end

private def hy_nomas_reference_kv_cache_values : Tuple(Array(Float32), Array(Float32))
  key = [
    1.00390625_f32, 1.01171875_f32, -1.00390625_f32, -1.01171875_f32,
    0.1001_f32, -0.3003_f32, 3.14159_f32, -9.125_f32,
    91.0_f32, 92.0_f32, 93.0_f32, 94.0_f32,
    95.0_f32, 96.0_f32, 97.0_f32, 98.0_f32,
  ]
  value = [
    -2.00390625_f32, -2.01171875_f32, 0.00390625_f32, 0.0078125_f32,
    0.7007_f32, 1.2345_f32, -3.14159_f32, 8.875_f32,
    -81.0_f32, -82.0_f32, -83.0_f32, -84.0_f32,
    -85.0_f32, -86.0_f32, -87.0_f32, -88.0_f32,
  ]
  {key, value}
end

private def hy_nomas_reference_kv_cache_overwrite!(
  chain : ML::Metal::MPSCommandChain, pipeline : ML::Metal::ComputePipeline,
  buffer : ML::MetalBuffer, count : Int32, sentinel : Float32,
) : Nil
  chain.encoder
    .set_pipeline(pipeline)
    .set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
    .set_value(count.to_u32, 1)
    .set_value(sentinel, 2)
    .dispatch_1d(count)
end

private def hy_nomas_reference_kv_cache_bits(values : Array(Float32)) : Array(UInt32)
  values.map(&.unsafe_as(UInt32))
end

describe ML::Video::HyNomasReferenceKVCache::Plan do
  it "rejects invalid source extents and aliases before allocating cache buffers" do
    short_key = ML::MetalBuffer.from_array(Array.new(15, 0.0_f32))
    full_value = ML::MetalBuffer.from_array(Array.new(16, 0.0_f32))
    alias_buffer = ML::MetalBuffer.from_array(Array.new(16, 1.0_f32))
    begin
      allocations_before_extent = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /exact-size shared F32/) do
        ML::Video::HyNomasReferenceKVCache::Plan.new(short_key, full_value,
          storage_rows: 4, active_rows: 2, heads: 2, head_dim: 2)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_extent)

      allocations_before_alias = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /must not alias/) do
        ML::Video::HyNomasReferenceKVCache::Plan.new(alias_buffer, alias_buffer,
          storage_rows: 4, active_rows: 2, heads: 2, head_dim: 2)
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_alias)
    ensure
      short_key.release
      full_value.release
      alias_buffer.release
    end
  end

  it "stores the active physical BF16 pair before source overwrite and reads it back without allocation or waiting" do
    key_values, value_values = hy_nomas_reference_kv_cache_values
    key = ML::MetalBuffer.from_array(key_values)
    value = ML::MetalBuffer.from_array(value_values)
    plan = nil.as(ML::Video::HyNomasReferenceKVCache::Plan?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      plan = ML::Video::HyNomasReferenceKVCache::Plan.new(key, value,
        storage_rows: 4, active_rows: 2, heads: 2, head_dim: 2)
      plan.not_nil!.shape.should eq({2, 2, 2})
      plan.not_nil!.resident_bytes.should eq(96_i64)
      chain = ML::Metal::MPSCommandChain.new

      expect_raises(ArgumentError, /must be stored before read/) do
        plan.not_nil!.read(chain.not_nil!)
      end
      plan.not_nil!.validate_store!(chain.not_nil!)
      allocations_before_encode = ML::MetalBuffer.video_spec_allocations
      plan.not_nil!.store(chain.not_nil!)
      expect_raises(ArgumentError, /single-write/) do
        plan.not_nil!.store(chain.not_nil!)
      end

      overwrite = ML::Metal::ComputePipeline.new(
        "hy_nomas_reference_kv_cache_spec_overwrite",
        HY_NOMAS_REFERENCE_KV_CACHE_OVERWRITE_SOURCE)
      source_count = plan.not_nil!.geometry.source_elements
      hy_nomas_reference_kv_cache_overwrite!(chain.not_nil!, overwrite,
        key, source_count, -777.0_f32)
      hy_nomas_reference_kv_cache_overwrite!(chain.not_nil!, overwrite,
        value, source_count, 888.0_f32)
      plan.not_nil!.read(chain.not_nil!)
      # Repeat reads are permitted; the second pair still comes from the
      # physical BF16 cache after the original F32 sources were overwritten.
      plan.not_nil!.read(chain.not_nil!)

      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_encode)
      chain.not_nil!.wait_count.should eq(0)
      expect_raises(ArgumentError, /Drain.*before release/) do
        plan.not_nil!.release
      end

      chain.not_nil!.commit_and_wait
      chain.not_nil!.wait_count.should eq(1)
      active_key = key_values[0, plan.not_nil!.geometry.cache_elements]
      active_value = value_values[0, plan.not_nil!.geometry.cache_elements]
      expected_key_bits = hy_nomas_reference_kv_cache_bits(active_key.map { |x| hy_nomas_reference_kv_cache_roundtrip(x) })
      expected_value_bits = hy_nomas_reference_kv_cache_bits(active_value.map { |x| hy_nomas_reference_kv_cache_roundtrip(x) })
      hy_nomas_reference_kv_cache_bits(plan.not_nil!.key_read.read(plan.not_nil!.geometry.cache_elements)).should eq(expected_key_bits)
      hy_nomas_reference_kv_cache_bits(plan.not_nil!.value_read.read(plan.not_nil!.geometry.cache_elements)).should eq(expected_value_bits)
      key.read(source_count).should eq(Array.new(source_count, -777.0_f32))
      value.read(source_count).should eq(Array.new(source_count, 888.0_f32))

      plan.not_nil!.release
      plan = nil
    ensure
      if pending = chain
        pending.commit_and_wait unless pending.completed?
      end
      plan.try(&.release)
      chain.try(&.release)
      key.release
      value.release
    end
  end

  it "allows a physical-cache read on a new chain only after the store chain drains" do
    key_values, value_values = hy_nomas_reference_kv_cache_values
    key = ML::MetalBuffer.from_array(key_values)
    value = ML::MetalBuffer.from_array(value_values)
    plan = nil.as(ML::Video::HyNomasReferenceKVCache::Plan?)
    store_chain = nil.as(ML::Metal::MPSCommandChain?)
    read_chain = nil.as(ML::Metal::MPSCommandChain?)
    blocked_chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      plan = ML::Video::HyNomasReferenceKVCache::Plan.new(key, value,
        storage_rows: 4, active_rows: 2, heads: 2, head_dim: 2)
      store_chain = ML::Metal::MPSCommandChain.new
      plan.not_nil!.store(store_chain.not_nil!)
      blocked_chain = ML::Metal::MPSCommandChain.new
      expect_raises(ArgumentError, /cannot span in-flight command chains/) do
        plan.not_nil!.read(blocked_chain.not_nil!)
      end
      blocked_chain.not_nil!.commit_and_wait
      blocked_chain.not_nil!.release

      store_chain.not_nil!.commit_and_wait
      store_chain.not_nil!.release
      read_chain = ML::Metal::MPSCommandChain.new
      allocations_before_read = ML::MetalBuffer.video_spec_allocations
      plan.not_nil!.read(read_chain.not_nil!)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_read)
      read_chain.not_nil!.commit_and_wait

      active_key = key_values[0, plan.not_nil!.geometry.cache_elements]
      expected_key_bits = hy_nomas_reference_kv_cache_bits(active_key.map { |x| hy_nomas_reference_kv_cache_roundtrip(x) })
      hy_nomas_reference_kv_cache_bits(plan.not_nil!.key_read.read(plan.not_nil!.geometry.cache_elements)).should eq(expected_key_bits)
      plan.not_nil!.release
      plan = nil
    ensure
      if pending = store_chain
        pending.commit_and_wait unless pending.completed?
      end
      if pending = read_chain
        pending.commit_and_wait unless pending.completed?
      end
      plan.try(&.release)
      store_chain.try(&.release)
      read_chain.try(&.release)
      blocked_chain.try(&.release)
      key.release
      value.release
    end
  end
end
