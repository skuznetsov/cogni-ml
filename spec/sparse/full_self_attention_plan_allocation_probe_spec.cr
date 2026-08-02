{% unless flag?(:trellis2_sparse_full_self_attention_plan_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse full-attention plan allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/full_self_attention_plan"
require "../spec_helper"
require "json"

private def measured_sparse_full_attention_plan_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 bounded sparse full-attention plan allocation" do
  it "admits the exact score ceiling without materializing score or output payloads" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    points_per_batch = 64_i32
    point_count = batch_size * points_per_batch
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
    flat = ML::Sparse::TensorCPU.new(ML::Tensor.ones(point_count, 3), map)
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(flat, 1)

    plan = uninitialized ML::Sparse::FullSelfAttentionPlanCPU
    allocated = measured_sparse_full_attention_plan_bytes do
      plan = ML::Sparse::FullSelfAttentionPlanCPU.build(qkv)
    end

    plan.score_bytes.should eq(ML::Sparse::FullSelfAttentionPlanCPU::MAX_SCORE_BYTES)
    plan.output_bytes.should eq(point_count.to_i64 * 4_i64)
    allocated.should be < 4_i64 * 1024_i64
    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-full-attention-plan-allocation-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "logical_score_bytes", plan.score_bytes
        json.field "logical_output_bytes", plan.output_bytes
        json.field "allocation_bytes", allocated
      end
    end
    puts measurement
  end
end
