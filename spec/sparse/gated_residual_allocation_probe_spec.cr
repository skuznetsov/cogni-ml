{% unless flag?(:trellis2_sparse_gated_residual_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse gated residual allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse"
require "../spec_helper"
require "json"

private def measured_sparse_gated_residual_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 sparse gated residual allocation" do
  it "allocates one output payload without copying inputs, gate, or batch map" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    points_per_batch = 1_024_i32
    point_count = batch_size * points_per_batch
    channels = 64_i32
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
    residual = ML::Sparse::TensorCPU.new(
      ML::Tensor.zeros(point_count, channels),
      map
    )
    update = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(point_count, channels),
      map
    )
    gate_msa = ML::Tensor.ones(batch_size, channels)

    output = uninitialized ML::Sparse::TensorCPU
    allocated = measured_sparse_gated_residual_bytes do
      output = ML::Sparse::TensorCPU.apply_gated_residual(
        residual,
        update,
        gate_msa
      )
    end
    output_payload_bytes = point_count.to_i64 * channels.to_i64 * 4_i64

    output.coordinate_map.same?(map).should be_true
    output.feature(0, 0).should eq(1.0_f32)
    output.feature(point_count - 1, channels - 1).should eq(1.0_f32)
    allocated.should be >= output_payload_bytes
    allocated.should be < output_payload_bytes + 64_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-gated-residual-output-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "input_payload_bytes", output_payload_bytes
        json.field "input_payload_count", 2
        json.field "gate_payload_bytes", batch_size.to_i64 * channels * 4_i64
        json.field "batch_map_payload_bytes", point_count.to_i64 * 4_i64
        json.field "output_payload_bytes", output_payload_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
