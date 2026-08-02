{% unless flag?(:trellis2_sparse_qk_rms_norm_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse Q/K RMS-normalization allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/self_attention_qk_rms_norm"
require "../spec_helper"
require "json"

private def measured_sparse_qk_rms_norm_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 packed sparse Q/K RMS-normalization allocation" do
  it "allocates one fresh packed QKV payload" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    points_per_batch = 1_024_i32
    point_count = batch_size * points_per_batch
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      offset = row * 4
      coordinates[offset] = row // points_per_batch
      coordinates[offset + 1] = row % points_per_batch
    end
    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      batch_size,
      {points_per_batch, 1, 1}
    )
    qkv_values = Array(Float32).new(point_count * 3, 0.0_f32)
    point_count.times do |row|
      qkv_values[row * 3] = 2.0_f32
      qkv_values[row * 3 + 1] = -3.0_f32
      qkv_values[row * 3 + 2] = row.to_f32
    end
    flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        qkv_values,
        ML::Shape.new(point_count, 3_i32)
      ),
      map
    )
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(flat, 1)
    q_gamma = ML::Tensor.from_array(
      [0.5_f32],
      ML::Shape.new(1_i32, 1_i32)
    )
    k_gamma = ML::Tensor.from_array(
      [2.0_f32],
      ML::Shape.new(1_i32, 1_i32)
    )

    output = uninitialized ML::Sparse::SelfAttentionQKVCPU
    allocated = measured_sparse_qk_rms_norm_bytes do
      output = ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
        qkv,
        q_gamma,
        k_gamma
      )
    end
    output_payload_bytes = point_count.to_i64 * 3_i64 * 4_i64

    output.coordinate_map.same?(map).should be_true
    output.feature(0, 0, 0, 0).should eq(0.5_f32)
    output.feature(0, 1, 0, 0).should eq(-2.0_f32)
    output.feature(point_count - 1, 2, 0, 0).should eq(
      (point_count - 1).to_f32
    )
    allocated.should be >= output_payload_bytes
    allocated.should be < output_payload_bytes + 8_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-qk-rms-norm-output-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "output_payload_bytes", output_payload_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
