{% unless flag?(:trellis2_sparse_cross_attention_pre_output_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse cross-attention pre-output allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "json"
require "../../src/ml/sparse/cross_attention_pre_output"
require "../spec_helper"

private def measured_sparse_cross_attention_pre_output_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 bounded sparse cross-attention pre-output allocation" do
  it "allocates one flat output payload and one reusable score row" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    # Keep the conservative logical score admission below the 1 MiB planner
    # cap while leaving it far above the physical one-row scratch footprint.
    points_per_batch = 16_i32
    point_count = batch_size * points_per_batch
    channels = 6_i32
    context_length = 64_i32
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
    query = ML::Sparse::TensorCPU.new(
      ML::Tensor.zeros(point_count, channels),
      map
    )
    context = ML::Tensor.zeros(
      batch_size,
      context_length,
      channels,
      device: ML::Tensor::Device::CPU
    )
    to_q = ML::NN::Linear.new(
      channels,
      channels,
      device: ML::Tensor::Device::CPU
    )
    to_kv = ML::NN::Linear.new(
      channels,
      channels * 2,
      device: ML::Tensor::Device::CPU
    )
    {to_q, to_kv}.each do |layer|
      layer.weight.data.cpu_data.not_nil!.fill(0.0_f32)
      layer.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
      layer.weight.requires_grad = false
      layer.bias.not_nil!.requires_grad = false
    end
    projection = ML::Sparse::CrossAttentionProjectionCPU.project(
      query,
      context,
      2,
      to_q,
      to_kv
    )
    gamma = ML::Tensor.ones(2, 3, device: ML::Tensor::Device::CPU)
    normalized = ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
      projection,
      gamma,
      gamma
    )
    ML::Sparse::TensorCPU.apply_cross_attention_pre_output(normalized)

    output = uninitialized ML::Sparse::TensorCPU
    allocated = measured_sparse_cross_attention_pre_output_bytes do
      output = ML::Sparse::TensorCPU.apply_cross_attention_pre_output(normalized)
    end
    output.coordinate_map.same?(map).should be_true
    output.feature(0, 0).should eq(0.0_f32)
    output.feature(point_count - 1, channels - 1).should eq(0.0_f32)

    output_payload_bytes = point_count.to_i64 * channels.to_i64 * 4_i64
    score_row_bytes = context_length.to_i64 * 4_i64
    composed_payload_bytes = output_payload_bytes + score_row_bytes
    allocated.should be >= composed_payload_bytes
    allocated.should be < composed_payload_bytes + 32_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-cross-attention-pre-output-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "output_payload_bytes", output_payload_bytes
        json.field "score_row_bytes", score_row_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
