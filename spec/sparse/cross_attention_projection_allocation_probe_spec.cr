{% unless flag?(:trellis2_sparse_cross_attention_projection_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse cross-attention projection allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/cross_attention_projection"
require "../spec_helper"
require "json"

private def measured_sparse_cross_attention_projection_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

private def freeze_sparse_cross_attention_projection_probe(
  layer : ML::NN::Linear,
) : Nil
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
end

describe "TRELLIS.2 bounded sparse cross-attention projection allocation" do
  it "allocates only physical flat Q and dense K/V payloads" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    points_per_nonempty_batch = 2_048_i32
    nonempty_batch_count = batch_size // 2
    point_count = nonempty_batch_count * points_per_nonempty_batch
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      local_row = row % points_per_nonempty_batch
      offset = row * 4
      coordinates[offset] = (row // points_per_nonempty_batch) * 2
      coordinates[offset + 1] = local_row % 1_024
      coordinates[offset + 2] = local_row // 1_024
    end
    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      batch_size,
      {1_024, 2, 1}
    )
    query = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(point_count, 1, device: ML::Tensor::Device::CPU),
      map
    )
    context = ML::Tensor.ones(
      batch_size,
      1,
      1,
      device: ML::Tensor::Device::CPU
    )
    to_q = ML::NN::Linear.new(1, 1, device: ML::Tensor::Device::CPU)
    to_kv = ML::NN::Linear.new(1, 2, device: ML::Tensor::Device::CPU)
    freeze_sparse_cross_attention_projection_probe(to_q)
    freeze_sparse_cross_attention_projection_probe(to_kv)

    projection = uninitialized ML::Sparse::CrossAttentionProjectionCPU
    allocated = measured_sparse_cross_attention_projection_bytes do
      projection = ML::Sparse::CrossAttentionProjectionCPU.project(
        query,
        context,
        1,
        to_q,
        to_kv
      )
    end

    physical_query_elements = point_count.to_i64
    padded_query_elements = batch_size.to_i64 *
                            points_per_nonempty_batch.to_i64
    projection_bytes = projection.plan.projection_bytes
    padding_bytes = (padded_query_elements - physical_query_elements) * 4_i64
    projection.query_feature_shape.should eq({point_count, 1})
    projection.query_batch_slice(1).size.should eq(0)
    allocated.should be < projection_bytes + padding_bytes // 2_i64

    context.cpu_data.not_nil![0] = Float32::NAN
    to_q.weight.requires_grad = true
    rejected_allocation = measured_sparse_cross_attention_projection_bytes do
      begin
        ML::Sparse::CrossAttentionProjectionCPU.project(
          query,
          context,
          1,
          to_q,
          to_kv,
          max_projection_bytes: projection_bytes - 1_i64
        )
      rescue ML::Sparse::SparseTensorBudgetError
      end
    end
    rejected_allocation.should be < 4_i64 * 1_024_i64

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-cross-attention-projection-allocation-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "physical_query_elements", physical_query_elements
        json.field "padded_query_elements", padded_query_elements
        json.field "logical_projection_bytes", projection_bytes
        json.field "allocation_bytes", allocated
        json.field "rejected_allocation_bytes", rejected_allocation
      end
    end
    puts measurement
  end
end
