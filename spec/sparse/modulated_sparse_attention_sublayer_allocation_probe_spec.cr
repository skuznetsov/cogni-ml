{% unless flag?(:trellis2_sparse_modulated_sublayer_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "modulated sparse attention sublayer allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/adaptive_layer_norm"
require "../../src/ml/sparse/full_self_attention_output"
require "../../src/ml/sparse/gated_residual"
require "../spec_helper"
require "json"

private def measured_modulated_sparse_attention_sublayer_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 modulated sparse attention sublayer allocation" do
  it "bounds the sequential CPU reference by its named feature payloads" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    points_per_batch = 64_i32
    point_count = batch_size * points_per_batch
    channels = 8_i32
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
    residual = ML::Sparse::TensorCPU.new(
      ML::Tensor.zeros(point_count, channels),
      map
    )
    scale = ML::Tensor.zeros(batch_size, channels)
    shift = ML::Tensor.zeros(batch_size, channels)
    gate = ML::Tensor.ones(batch_size, channels)
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
    allocated = measured_modulated_sparse_attention_sublayer_bytes do
      normalized = ML::Sparse::TensorCPU.apply_adaptive_layer_norm(
        residual,
        scale,
        shift
      )
      attention = ML::Sparse::TensorCPU.apply_full_self_attention_output(
        normalized,
        to_qkv,
        to_out,
        1,
        q_gamma: q_gamma,
        k_gamma: k_gamma,
        use_rope: true
      )
      output = ML::Sparse::TensorCPU.apply_gated_residual(
        residual,
        attention,
        gate
      )
    end

    feature_payload_bytes = point_count.to_i64 * channels.to_i64 * 4_i64
    score_row_bytes = points_per_batch.to_i64 * 4_i64
    # adaptive output + QKV + Q/K-normalized QKV + RoPE QKV + attention
    # context + output projection + gated residual = 13 C-channel payloads.
    named_payload_bytes = 13_i64 * feature_payload_bytes + score_row_bytes

    output.coordinate_map.same?(map).should be_true
    output.feature(0, 0).should eq(0.0_f32)
    output.feature(point_count - 1, channels - 1).should eq(0.0_f32)
    allocated.should be >= named_payload_bytes
    allocated.should be < named_payload_bytes + 64_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-modulated-sparse-attention-sublayer-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "feature_payload_bytes", feature_payload_bytes
        json.field "feature_payload_count", 13
        json.field "score_row_bytes", score_row_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
