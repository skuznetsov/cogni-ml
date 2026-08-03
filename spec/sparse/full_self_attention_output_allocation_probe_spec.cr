{% unless flag?(:trellis2_sparse_full_self_attention_output_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse full self-attention output allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/full_self_attention_output"
require "../spec_helper"
require "json"

private def measured_sparse_full_attention_output_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 sparse full self-attention output allocation" do
  it "adds only the final C-channel output payload to the bounded composition" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    points_per_batch = 64_i32
    point_count = batch_size * points_per_batch
    channels = 6_i32
    projected_channels = channels * 3
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
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.zeros(point_count, channels),
      map
    )
    to_qkv = ML::NN::Linear.new(
      channels,
      projected_channels,
      device: ML::Tensor::Device::CPU
    )
    to_qkv.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    to_qkv.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    to_qkv.weight.requires_grad = false
    to_qkv.bias.not_nil!.requires_grad = false
    to_out = ML::NN::Linear.new(
      channels,
      channels,
      device: ML::Tensor::Device::CPU
    )
    to_out.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    to_out.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    to_out.weight.requires_grad = false
    to_out.bias.not_nil!.requires_grad = false
    q_gamma = ML::Tensor.ones(1, channels)
    k_gamma = ML::Tensor.ones(1, channels)

    output = uninitialized ML::Sparse::TensorCPU
    allocated = measured_sparse_full_attention_output_bytes do
      output = ML::Sparse::TensorCPU.apply_full_self_attention_output(
        input,
        to_qkv,
        to_out,
        1,
        q_gamma: q_gamma,
        k_gamma: k_gamma,
        use_rope: true
      )
    end
    packed_payload_bytes =
      point_count.to_i64 * projected_channels.to_i64 * 4_i64
    output_payload_bytes = point_count.to_i64 * channels.to_i64 * 4_i64
    score_row_bytes = points_per_batch.to_i64 * 4_i64
    composed_payload_bytes =
      3_i64 * packed_payload_bytes + 2_i64 * output_payload_bytes + score_row_bytes

    output.coordinate_map.same?(map).should be_true
    output.feature(0, 0).should eq(0.0_f32)
    output.feature(point_count - 1, channels - 1).should eq(0.0_f32)
    allocated.should be >= composed_payload_bytes
    allocated.should be < composed_payload_bytes + 32_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-full-self-attention-output-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "packed_payload_bytes", packed_payload_bytes
        json.field "packed_payload_count", 3
        json.field "output_payload_bytes", output_payload_bytes
        json.field "output_payload_count", 2
        json.field "score_row_bytes", score_row_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
