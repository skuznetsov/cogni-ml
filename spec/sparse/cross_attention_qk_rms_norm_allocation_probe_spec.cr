{% unless flag?(:trellis2_sparse_cross_attention_qk_rms_norm_allocation_probe) %}
  {% skip_file %}
{% end %}

{% unless flag?(:darwin) && flag?(:cpu_only) %}
  {% raise "sparse cross-attention Q/K normalization allocation probe requires Darwin CPU-only execution" %}
{% end %}

require "../../src/ml/sparse/cross_attention_qk_rms_norm"
require "../spec_helper"
require "json"

private def measured_cross_qk_rms_norm_bytes(& : ->) : Int64
  GC.collect
  before = GC.stats.total_bytes.to_i64
  yield
  GC.stats.total_bytes.to_i64 - before
end

describe "TRELLIS.2 sparse cross-attention Q/K normalization allocation" do
  it "owns only physical Q and dense K while retaining V zero-copy" do
    batch_size = ML::Sparse::CoordinateMap3D::MAX_BATCH_SIZE
    channels = 256_i32
    context_length = 64_i32
    point_count = batch_size // 2
    coordinates = Array(Int32).new(point_count * 4, 0_i32)
    point_count.times do |row|
      offset = row * 4
      coordinates[offset] = row * 2
      coordinates[offset + 1] = row
    end
    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      batch_size,
      {point_count, 1, 1}
    )
    query = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(point_count, channels, device: ML::Tensor::Device::CPU),
      map
    )
    context = ML::Tensor.ones(
      batch_size,
      context_length,
      1,
      device: ML::Tensor::Device::CPU
    )
    to_q = ML::NN::Linear.new(
      channels,
      channels,
      device: ML::Tensor::Device::CPU
    )
    to_kv = ML::NN::Linear.new(
      1,
      2 * channels,
      device: ML::Tensor::Device::CPU
    )
    {to_q, to_kv}.each do |layer|
      layer.weight.requires_grad = false
      layer.bias.not_nil!.requires_grad = false
    end
    projection = ML::Sparse::CrossAttentionProjectionCPU.project(
      query,
      context,
      1,
      to_q,
      to_kv
    )
    gamma = ML::Tensor.ones(1, channels, device: ML::Tensor::Device::CPU)

    output = uninitialized ML::Sparse::CrossAttentionQKNormalizedCPU
    allocated = measured_cross_qk_rms_norm_bytes do
      output = ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
        projection,
        gamma,
        gamma
      )
    end

    key_elements = projection.plan.context_kv_elements // 2_i64
    value_bytes = key_elements * 4_i64
    output.source_projection.same?(projection).should be_true
    output.query_features_copy.size.should eq(point_count * channels)
    output.normalized_bytes.should eq(
      (projection.plan.query_projection_elements + key_elements) * 4_i64
    )
    # The fixed margin is smaller than either a padded-Q copy (32 KiB) or the
    # dense V payload (4 MiB) in this probe. This scopes the observation to
    # cumulative managed allocation traffic; it is not backing-pointer, RSS,
    # native-allocation, or peak-memory evidence.
    allocated.should be < output.normalized_bytes + 4_i64 * 1_024_i64

    gamma.cpu_data.not_nil![0] = Float32::NAN
    rejected_allocation = measured_cross_qk_rms_norm_bytes do
      begin
        ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
          projection,
          gamma,
          gamma,
          max_normalized_bytes: output.normalized_bytes - 1_i64
        )
      rescue ML::Sparse::SparseTensorBudgetError
      end
    end
    rejected_allocation.should be < 4_i64 * 1_024_i64

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "trellis2-sparse-cross-attention-qk-rms-norm-allocation-v1"
        json.field "scope", "darwin-local-cpu-only-manual"
        json.field "gc_total_bytes_kind", "cumulative-allocation-traffic"
        json.field "physical_query_rows", point_count
        json.field "padded_query_rows", batch_size
        json.field "normalized_qk_bytes", output.normalized_bytes
        json.field "borrowed_value_bytes", value_bytes
        json.field "allocation_bytes", allocated
        json.field "rejected_allocation_bytes", rejected_allocation
      end
    end
    puts measurement
  end
end
