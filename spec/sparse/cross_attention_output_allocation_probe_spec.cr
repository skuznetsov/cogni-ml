{% unless flag?(:trellis2_sparse_cross_attention_output_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse cross-attention output allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "json"
require "../../src/ml/sparse/cross_attention_output"
require "../spec_helper"

private def measured_sparse_cross_attention_output_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 bounded sparse cross-attention output allocation" do
  it "allocates only one final flat output payload" do
    point_count = ML::Sparse::CoordinateMap3D::MAX_POINTS
    channels = 6_i32
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      offset = row * 4
      coordinates[offset + 1] = row % 1024
      coordinates[offset + 2] = (row // 1024) % 64
    end
    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      1,
      {1024, 64, 1}
    )
    query = ML::Sparse::TensorCPU.new(
      ML::Tensor.zeros(point_count, channels),
      map
    )
    context = ML::Tensor.zeros(
      1,
      1,
      channels,
      device: ML::Tensor::Device::CPU
    )
    plan = ML::Sparse::CrossAttentionPlanCPU.preflight(
      query,
      context,
      2
    )
    pre_output = ML::Sparse::TensorCPU.new(
      ML::Tensor.zeros(point_count, channels),
      map
    )
    to_out = ML::NN::Linear.new(
      channels,
      channels,
      device: ML::Tensor::Device::CPU
    )
    to_out.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    to_out.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    to_out.weight.requires_grad = false
    to_out.bias.not_nil!.requires_grad = false

    ML::Sparse::TensorCPU.apply_cross_attention_output(
      pre_output,
      plan,
      to_out
    )
    output = uninitialized ML::Sparse::TensorCPU
    allocated = measured_sparse_cross_attention_output_bytes do
      output = ML::Sparse::TensorCPU.apply_cross_attention_output(
        pre_output,
        plan,
        to_out
      )
    end

    output_payload_bytes = point_count.to_i64 * channels.to_i64 * 4_i64
    output.coordinate_map.same?(map).should be_true
    output.feature(0, 0).should eq(0.0_f32)
    output.feature(point_count - 1, channels - 1).should eq(0.0_f32)
    allocated.should be >= output_payload_bytes
    allocated.should be < output_payload_bytes + 32_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-cross-attention-output-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "output_payload_bytes", output_payload_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
