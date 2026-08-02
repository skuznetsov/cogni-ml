{% unless flag?(:trellis2_sparse_adaptive_layer_norm_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse adaptive layer norm allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/adaptive_layer_norm"
require "../spec_helper"
require "json"

private def measured_sparse_adaln_allocation_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

private def maximum_sparse_adaln_map : ML::Sparse::CoordinateMap3D
  point_count = ML::Sparse::CoordinateMap3D::MAX_POINTS
  coordinates = Array(Int32).new(point_count * 4, 0_i32)
  point_count.times do |row|
    offset = row * 4
    coordinates[offset + 1] = row % 1024
    coordinates[offset + 2] = (row // 1024) % 64
  end
  ML::Sparse::CoordinateMap3D.new(coordinates, 1, {1024, 64, 1})
end

describe "TRELLIS.2 bounded sparse adaptive layer norm allocation" do
  it "allocates one feature payload without a second input or output copy" do
    point_count = ML::Sparse::CoordinateMap3D::MAX_POINTS
    channels = 64_i32
    map = maximum_sparse_adaln_map
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(point_count, channels),
      map
    )
    scale = ML::Tensor.zeros(1, channels)
    shift = ML::Tensor.zeros(1, channels)

    output = uninitialized ML::Sparse::TensorCPU
    allocated = measured_sparse_adaln_allocation_bytes do
      output = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        input,
        scale,
        shift
      )
    end
    output_payload_bytes = point_count.to_i64 * channels * 4_i64

    output.coordinate_map.same?(map).should be_true
    allocated.should be >= output_payload_bytes
    allocated.should be < output_payload_bytes + 64_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-adaptive-layer-norm-output-copy-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "input_payload_bytes", output_payload_bytes
        json.field "output_payload_bytes", output_payload_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end

  it "does not duplicate the coordinate batch-broadcast map" do
    point_count = ML::Sparse::CoordinateMap3D::MAX_POINTS
    channels = 1_i32
    map = maximum_sparse_adaln_map
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(point_count, channels),
      map
    )
    scale = ML::Tensor.zeros(1, channels)
    shift = ML::Tensor.zeros(1, channels)

    output = uninitialized ML::Sparse::TensorCPU
    allocated = measured_sparse_adaln_allocation_bytes do
      output = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        input,
        scale,
        shift
      )
    end
    output_payload_bytes = point_count.to_i64 * 4_i64

    output.coordinate_map.same?(map).should be_true
    allocated.should be >= output_payload_bytes
    allocated.should be < output_payload_bytes + 64_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-adaptive-layer-norm-map-copy-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "batch_map_payload_bytes", point_count.to_i64 * 4_i64
        json.field "output_payload_bytes", output_payload_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
