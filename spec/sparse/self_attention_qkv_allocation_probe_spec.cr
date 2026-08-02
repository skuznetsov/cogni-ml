{% unless flag?(:trellis2_sparse_self_attention_qkv_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse self-attention QKV allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/self_attention_qkv"
require "../spec_helper"
require "json"

private def freeze_sparse_qkv_probe(layer : ML::NN::Linear) : Nil
  layer.weight.requires_grad = false
  layer.bias.try { |bias| bias.requires_grad = false }
end

private def measured_sparse_qkv_allocation_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 bounded sparse self-attention QKV allocation" do
  it "allocates one projected payload without input or reshape copies" do
    point_count = ML::Sparse::CoordinateMap3D::MAX_POINTS
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      offset = row * 4
      coordinates[offset + 1] = row % 1024
      coordinates[offset + 2] = (row // 1024) % 64
    end
    map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {1024, 64, 1})
    input = ML::Sparse::TensorCPU.new(ML::Tensor.ones(point_count, 1), map)
    layer = ML::NN::Linear.new(1, 3, device: ML::Tensor::Device::CPU)
    freeze_sparse_qkv_probe(layer)

    output = uninitialized ML::Sparse::SelfAttentionQKVCPU
    allocated = measured_sparse_qkv_allocation_bytes do
      output = ML::Sparse::TensorCPU.apply_self_attention_qkv(input, layer, 1)
    end
    input_payload_bytes = point_count.to_i64 * 4_i64
    output_payload_bytes = point_count.to_i64 * 3_i64 * 4_i64

    output.shape.should eq({1, 3, 1, 1})
    output.coordinate_map.same?(map).should be_true
    allocated.should be < output_payload_bytes + 64_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-self-attention-qkv-copy-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "input_payload_bytes", input_payload_bytes
        json.field "output_payload_bytes", output_payload_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
