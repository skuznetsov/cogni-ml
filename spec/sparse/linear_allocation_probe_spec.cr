{% unless flag?(:trellis2_sparse_linear_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse linear allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/linear"
require "../spec_helper"
require "json"

private def freeze_probe_linear(layer : ML::NN::Linear) : Nil
  layer.weight.requires_grad = false
  layer.bias.try { |bias| bias.requires_grad = false }
end

private def measured_allocation_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 bounded sparse linear allocation" do
  it "does not materialize a full input copy" do
    point_count = ML::Sparse::CoordinateMap3D::MAX_POINTS
    input_channels = 64_i32
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      offset = row * 4
      coordinates[offset + 1] = row % 1024
      coordinates[offset + 2] = (row // 1024) % 64
    end
    map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {1024, 64, 1})
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(point_count, input_channels),
      map
    )
    layer = ML::NN::Linear.new(
      input_channels,
      1,
      device: ML::Tensor::Device::CPU
    )
    freeze_probe_linear(layer)

    output = uninitialized ML::Sparse::TensorCPU
    allocated = measured_allocation_bytes do
      output = ML::Sparse::TensorCPU.apply_linear(input, layer)
    end
    input_payload_bytes = point_count.to_i64 * input_channels * 4_i64

    output.point_count.should eq(point_count)
    allocated.should be < input_payload_bytes // 4_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-linear-input-copy-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "input_payload_bytes", input_payload_bytes
        json.field "output_payload_bytes", point_count.to_i64 * 4_i64
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end

  it "does not materialize a second full output copy" do
    point_count = ML::Sparse::CoordinateMap3D::MAX_POINTS
    output_channels = 64_i32
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      offset = row * 4
      coordinates[offset + 1] = row % 1024
      coordinates[offset + 2] = (row // 1024) % 64
    end
    map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {1024, 64, 1})
    input = ML::Sparse::TensorCPU.new(ML::Tensor.ones(point_count, 1), map)
    layer = ML::NN::Linear.new(
      1,
      output_channels,
      device: ML::Tensor::Device::CPU
    )
    freeze_probe_linear(layer)

    output = uninitialized ML::Sparse::TensorCPU
    allocated = measured_allocation_bytes do
      output = ML::Sparse::TensorCPU.apply_linear(input, layer)
    end
    output_payload_bytes = point_count.to_i64 * output_channels * 4_i64

    output.channels.should eq(output_channels)
    allocated.should be < output_payload_bytes + 1_i64 * 1024_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-linear-output-copy-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "input_payload_bytes", point_count.to_i64 * 4_i64
        json.field "output_payload_bytes", output_payload_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
