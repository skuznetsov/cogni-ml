require "json"
require "../../src/ml/sparse/cross_attention_plan"
require "../spec_helper"

private def sparse_cross_attention_plan_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json"
  )))
end

private def sparse_cross_attention_plan_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_cross_attention_plan_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if hash = payload.as_h?
    if values = hash["values"]?
      return sparse_cross_attention_plan_f32(values, output)
    end
  elsif array = payload.as_a?
    array.each { |entry| sparse_cross_attention_plan_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_cross_attention_plan_query(
  fixture : JSON::Any,
) : ML::Sparse::TensorCPU
  input = fixture["input"]
  point_count = input["coordinates"].as_a.size.to_i32
  channels = input["channels"].as_i.to_i32
  map = ML::Sparse::CoordinateMap3D.new(
    sparse_cross_attention_plan_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  ML::Sparse::TensorCPU.new(
    ML::Tensor.zeros(point_count, channels, device: ML::Tensor::Device::CPU),
    map
  )
end

private def sparse_cross_attention_plan_context(
  fixture : JSON::Any,
) : ML::Tensor
  input = fixture["input"]
  values = sparse_cross_attention_plan_f32(input["context"])
  ML::Tensor.from_array(
    values,
    ML::Shape.new(
      input["batch_size"].as_i.to_i32,
      input["context_length"].as_i.to_i32,
      input["context_channels"].as_i.to_i32
    )
  )
end

class SparseCrossAttentionPlanTensorCPUOverride < ML::Sparse::TensorCPU
end

class SparseCrossAttentionPlanContextOverride < ML::Tensor
end

class SparseCrossAttentionPlanMapOverride < ML::Sparse::CoordinateMap3D
end

describe ML::Sparse::CrossAttentionPlanCPU do
  it "preserves the pinned flat sparse query layout against dense uniform context" do
    fixture = sparse_cross_attention_plan_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-cross-attention-seam-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )

    query = sparse_cross_attention_plan_query(fixture)
    context = sparse_cross_attention_plan_context(fixture)
    plan = ML::Sparse::CrossAttentionPlanCPU.preflight(query, context, 2)

    plan.batch_size.should eq(3)
    plan.point_count.should eq(4)
    plan.query_channels.should eq(16)
    plan.context_channels.should eq(5)
    plan.context_length.should eq(3)
    plan.num_heads.should eq(2)
    plan.head_dim.should eq(8)
    plan.max_query_length.should eq(2)
    plan.query_batch_slice(0).should eq(ML::Sparse::BatchSlice.new(0, 2))
    plan.query_batch_slice(1).should eq(ML::Sparse::BatchSlice.new(2, 2))
    plan.query_batch_slice(2).should eq(ML::Sparse::BatchSlice.new(2, 4))
    4.times.map { |row| plan.query_batch_index(row) }.to_a.should eq([0, 0, 2, 2])
    plan.coordinate_map.same?(query.coordinate_map).should be_true

    plan.score_elements.should eq(24_i64)
    plan.score_bytes.should eq(96_i64)
    plan.query_projection_elements.should eq(64_i64)
    plan.context_kv_elements.should eq(288_i64)
    plan.projection_elements.should eq(352_i64)
    plan.projection_bytes.should eq(1_408_i64)
    plan.output_elements.should eq(64_i64)
    plan.output_bytes.should eq(256_i64)
    plan.query_projection_mac_elements.should eq(1_024_i64)
    plan.context_kv_projection_mac_elements.should eq(1_440_i64)
    plan.attention_mac_elements.should eq(384_i64)
    plan.output_projection_mac_elements.should eq(1_024_i64)
    plan.work_elements.should eq(3_872_i64)
  end

  it "accepts exact caller caps and rejects one unit below each cap" do
    fixture = sparse_cross_attention_plan_fixture
    query = sparse_cross_attention_plan_query(fixture)
    context = sparse_cross_attention_plan_context(fixture)
    exact = ML::Sparse::CrossAttentionPlanCPU.preflight(
      query,
      context,
      2,
      max_score_bytes: 96_i64,
      max_projection_bytes: 1_408_i64,
      max_work_elements: 3_872_i64
    )
    exact.max_score_bytes.should eq(96_i64)
    exact.max_projection_bytes.should eq(1_408_i64)
    exact.max_work_elements.should eq(3_872_i64)

    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 96 bytes.*limit is 95/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        query,
        context,
        2,
        max_score_bytes: 95_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 1408 bytes.*limit is 1407/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        query,
        context,
        2,
        max_projection_bytes: 1_407_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 3872.*limit is 3871/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        query,
        context,
        2,
        max_work_elements: 3_871_i64
      )
    end
  end

  it "rejects attempts to widen process-local ceilings" do
    fixture = sparse_cross_attention_plan_fixture
    query = sparse_cross_attention_plan_query(fixture)
    context = sparse_cross_attention_plan_context(fixture)

    expect_raises(ML::Sparse::SparseTensorBudgetError, /score byte budget must be in 1/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        query,
        context,
        2,
        max_score_bytes: ML::Sparse::CrossAttentionPlanCPU::MAX_SCORE_BYTES + 1_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorBudgetError, /projection byte budget must be in 1/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        query,
        context,
        2,
        max_projection_bytes: ML::Sparse::CrossAttentionPlanCPU::MAX_PROJECTION_BYTES + 1_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorBudgetError, /work budget must be in 1/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        query,
        context,
        2,
        max_work_elements: ML::Sparse::CrossAttentionPlanCPU::MAX_WORK_ELEMENTS + 1_i64
      )
    end
  end

  it "rejects malformed dense context before any projection boundary" do
    fixture = sparse_cross_attention_plan_fixture
    query = sparse_cross_attention_plan_query(fixture)
    context = sparse_cross_attention_plan_context(fixture)

    batch_mismatch = ML::Tensor.zeros(2, 3, 5, device: ML::Tensor::Device::CPU)
    expect_raises(ML::Sparse::SparseTensorError, /context batch size 2 does not match sparse batch size 3/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(query, batch_mismatch, 2)
    end

    empty_context = ML::Tensor.zeros(3, 0, 5, device: ML::Tensor::Device::CPU)
    expect_raises(ML::Sparse::SparseTensorError, /context length must be positive/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(query, empty_context, 2)
    end

    rank_two = ML::Tensor.zeros(3, 5, device: ML::Tensor::Device::CPU)
    expect_raises(ML::Sparse::SparseTensorError, /rank 3 \[B, L, C\]/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(query, rank_two, 2)
    end

    strided_context = ML::Tensor.zeros(
      3, 5, 3, device: ML::Tensor::Device::CPU
    ).transpose
    strided_context.shape.should eq(context.shape)
    strided_context.contiguous?.should be_false
    expect_raises(ML::Sparse::SparseTensorError, /context must be contiguous/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(query, strided_context, 2)
    end

    expect_raises(ML::Sparse::SparseTensorError, /head count must be positive/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(query, context, 0)
    end
    expect_raises(ML::Sparse::SparseTensorError, /must be divisible by heads 3/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(query, context, 3)
    end
  end

  it "rejects work and byte caps without reading mutable context payloads" do
    fixture = sparse_cross_attention_plan_fixture
    query = sparse_cross_attention_plan_query(fixture)
    context = sparse_cross_attention_plan_context(fixture)
    context.cpu_data.not_nil![0] = Float32::NAN

    expect_raises(ML::Sparse::SparseTensorBudgetError, /score budget/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        query,
        context,
        2,
        max_score_bytes: 95_i64
      )
    end
    plan = ML::Sparse::CrossAttentionPlanCPU.preflight(query, context, 2)
    plan.score_bytes.should eq(96_i64)
  end

  it "requires canonical sparse query, coordinate map, and context runtime types" do
    fixture = sparse_cross_attention_plan_fixture
    query = sparse_cross_attention_plan_query(fixture)
    context = sparse_cross_attention_plan_context(fixture)
    inherited_query = SparseCrossAttentionPlanTensorCPUOverride.new(
      ML::Tensor.zeros(4, 16, device: ML::Tensor::Device::CPU),
      query.coordinate_map
    )
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(inherited_query, context, 2)
    end

    inherited_context = SparseCrossAttentionPlanContextOverride.new(
      context.shape,
      device: ML::Tensor::Device::CPU
    )
    expect_raises(ML::Sparse::SparseTensorError, /base Tensor context/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(query, inherited_context, 2)
    end

    input = fixture["input"]
    inherited_map = SparseCrossAttentionPlanMapOverride.new(
      sparse_cross_attention_plan_i32(input["coordinates"]),
      input["batch_size"].as_i.to_i32,
      {
        input["spatial_shape"][0].as_i.to_i32,
        input["spatial_shape"][1].as_i.to_i32,
        input["spatial_shape"][2].as_i.to_i32,
      }
    )
    query_with_inherited_map = ML::Sparse::TensorCPU.new(
      ML::Tensor.zeros(4, 16, device: ML::Tensor::Device::CPU),
      inherited_map
    )
    expect_raises(ML::Sparse::SparseTensorError, /base CoordinateMap3D/) do
      ML::Sparse::CrossAttentionPlanCPU.preflight(
        query_with_inherited_map,
        context,
        2
      )
    end
  end
end
