{% unless flag?(:trellis2_sparse_self_attention_rope_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse self-attention RoPE allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/self_attention_rope"
require "../spec_helper"
require "json"

private def measured_sparse_self_attention_rope_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 packed sparse self-attention RoPE allocation" do
  it "allocates one fresh packed QKV payload without a coordinate or phase plane" do
    point_count = 16_384_i32
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      offset = row * 4
      coordinates[offset + 1] = row % 1_024
      coordinates[offset + 2] = row // 1_024
    end
    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      1,
      {1_024, 16, 1}
    )
    projected_channels = 18_i32
    qkv_values = Array(Float32).new(point_count * projected_channels, 0.0_f32)
    point_count.times do |row|
      offset = row * projected_channels
      12.times { |index| qkv_values[offset + index] = (index + 1).to_f32 / 8.0_f32 }
      6.times { |index| qkv_values[offset + 12 + index] = (row + index).to_f32 }
    end
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(
      ML::Sparse::TensorCPU.new(
        ML::Tensor.from_array(
          qkv_values,
          ML::Shape.new(point_count, projected_channels)
        ),
        map
      ),
      1
    )

    output = uninitialized ML::Sparse::SelfAttentionQKVCPU
    allocated = measured_sparse_self_attention_rope_bytes do
      output = ML::Sparse::TensorCPU.apply_self_attention_rope(qkv)
    end
    output_payload_bytes = point_count.to_i64 * projected_channels.to_i64 * 4_i64

    output.coordinate_map.same?(map).should be_true
    output.feature(0, 2, 0, 0).should eq(0.0_f32)
    output.feature(point_count - 1, 2, 0, 5).should eq(
      (point_count - 1 + 5).to_f32
    )
    allocated.should be >= output_payload_bytes
    allocated.should be < output_payload_bytes + 8_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-self-attention-rope-output-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "output_payload_bytes", output_payload_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
