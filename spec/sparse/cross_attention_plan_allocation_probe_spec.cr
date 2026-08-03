{% unless flag?(:trellis2_sparse_cross_attention_plan_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse cross-attention plan allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/cross_attention_plan"
require "../spec_helper"
require "json"

private def measured_sparse_cross_attention_plan_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 bounded sparse cross-attention plan allocation" do
  it "admits a ragged exact score ceiling without padding or payload allocation" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    points_per_nonempty_batch = 128_i32
    nonempty_batch_count = batch_size // 2
    point_count = nonempty_batch_count * points_per_nonempty_batch
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      batch = (row // points_per_nonempty_batch) * 2
      offset = row * 4
      coordinates[offset] = batch
      coordinates[offset + 1] = row % points_per_nonempty_batch
    end
    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      batch_size,
      {points_per_nonempty_batch, 1, 1}
    )
    channels = 64_i32
    heads = 64_i32
    query = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(point_count, channels),
      map
    )
    context = ML::Tensor.ones(batch_size, 1_i32, 1_i32)

    plan = uninitialized ML::Sparse::CrossAttentionPlanCPU
    allocated = measured_sparse_cross_attention_plan_bytes do
      plan = ML::Sparse::CrossAttentionPlanCPU.preflight(query, context, heads)
    end

    padded_query_elements = batch_size.to_i64 *
                            points_per_nonempty_batch.to_i64 *
                            channels.to_i64
    plan.score_bytes.should eq(
      ML::Sparse::CrossAttentionPlanCPU::MAX_SCORE_BYTES
    )
    plan.query_projection_elements.should eq(
      point_count.to_i64 * channels.to_i64
    )
    plan.query_projection_elements.should be < padded_query_elements
    plan.query_batch_slice(1).size.should eq(0)
    allocated.should be < 4_i64 * 1024_i64

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-cross-attention-plan-allocation-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "physical_query_elements", plan.query_projection_elements
        json.field "padded_query_elements", padded_query_elements
        json.field "logical_score_bytes", plan.score_bytes
        json.field "logical_projection_bytes", plan.projection_bytes
        json.field "logical_work_elements", plan.work_elements
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
