require "json"
require "../../src/ml/sparse/full_self_attention_plan"
require "../spec_helper"

private def sparse_full_attention_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_full_self_attention_plan_cpu_v1.json"
  )))
end

private def flatten_sparse_full_attention_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if array = payload.as_a?
    array.each { |entry| flatten_sparse_full_attention_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_full_attention_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_full_attention_view(fixture : JSON::Any) : ML::Sparse::SelfAttentionQKVCPU
  input = fixture["input"]
  attention = fixture["attention"]
  point_count = input["coordinates"].as_a.size.to_i32
  num_heads = attention["num_heads"].as_i.to_i32
  head_dim = attention["head_dim"].as_i.to_i32
  projected_channels = 3_i32 * num_heads * head_dim
  map = ML::Sparse::CoordinateMap3D.new(
    sparse_full_attention_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  flat = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      flatten_sparse_full_attention_f32(input["qkv_features"]),
      ML::Shape.new(point_count, projected_channels)
    ),
    map
  )
  ML::Sparse::SelfAttentionQKVCPU.new(flat, num_heads)
end

class SparseFullAttentionMapOverride < ML::Sparse::CoordinateMap3D
  def sequence_lengths : Array(Int32)
    [999_i32]
  end
end

class SparseFullAttentionQKVOverride < ML::Sparse::SelfAttentionQKVCPU
end

describe ML::Sparse::FullSelfAttentionPlanCPU do
  it "matches the pinned block-diagonal full-attention budget contract" do
    fixture = sparse_full_attention_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-full-self-attention-plan-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["sources"]["sparse_full_attention"]["sha256"].as_s.should eq(
      "bee0c32089f060c8136292f41a2cc7a952a8679a8b6d113d14c772f2c681e520"
    )
    provenance["sources"]["sparse_attention_modules"]["sha256"].as_s.should eq(
      "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf"
    )
    provenance["sources"]["dense_full_attention"]["sha256"].as_s.should eq(
      "64c43354780dcbc3dcf7612ac5e53d6e21c2081234ea63cd329a77f4185dadfc"
    )
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["device"].as_s.should eq("cpu")
    provenance["network"].as_s.should eq("none")
    provenance["weights"].as_s.should eq("synthetic")
    provenance["upstream_sparse_backend_executed"].as_bool.should be_false

    contract = fixture["contract"]
    contract["reference_kind"].as_s.should eq(
      "source-bound explicit block-diagonal CPU reference"
    )
    contract["not_backend_parity"].as_bool.should be_true
    contract["sequence_lengths"].as_a.map(&.as_i).should eq([2_i64, 0_i64, 3_i64, 1_i64])
    contract["row_order_preserved_checked"].as_bool.should be_true

    plan = ML::Sparse::FullSelfAttentionPlanCPU.build(
      sparse_full_attention_view(fixture)
    )
    plan.batch_size.should eq(4)
    plan.point_count.should eq(6)
    plan.num_heads.should eq(2)
    plan.head_dim.should eq(2)
    plan.max_batch_length.should eq(3)
    plan.score_elements.should eq(28_i64)
    plan.score_bytes.should eq(112_i64)
    plan.output_elements.should eq(24_i64)
    plan.output_bytes.should eq(96_i64)
    plan.attention_mac_elements.should eq(112_i64)
    output = fixture["output"]
    output["features_f32le_sha256"].as_s.should eq(
      "ce294683d1ec7f8fe46bd45066871a5c3bfeaaf257256f7d24478e700dd12112"
    )
    output["batch_isolation_checked"].as_bool.should be_true
    output["stable_softmax_checked"].as_bool.should be_true
  end

  it "uses per-batch squares instead of a global point-count square" do
    plan = ML::Sparse::FullSelfAttentionPlanCPU.build(
      sparse_full_attention_view(sparse_full_attention_fixture)
    )
    plan.score_elements.should eq(2_i64 * (2_i64 ** 2 + 0_i64 ** 2 + 3_i64 ** 2 + 1_i64 ** 2))
    plan.score_elements.should_not eq(2_i64 * 6_i64 ** 2)
  end

  it "accepts the exact caller budget and rejects one byte below it" do
    view = sparse_full_attention_view(sparse_full_attention_fixture)
    exact = ML::Sparse::FullSelfAttentionPlanCPU.build(view, 112_i64)
    exact.max_score_bytes.should eq(112_i64)

    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 112 bytes.*limit is 111/) do
      ML::Sparse::FullSelfAttentionPlanCPU.build(view, 111_i64)
    end
  end

  it "admits empty and singleton layouts with conservative logical accounting" do
    empty_map = ML::Sparse::CoordinateMap3D.new(
      [] of Int32,
      3,
      {1, 1, 1}
    )
    empty_flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 12_i32)),
      empty_map
    )
    empty_view = ML::Sparse::SelfAttentionQKVCPU.new(empty_flat, 2)
    empty_plan = ML::Sparse::FullSelfAttentionPlanCPU.build(empty_view)
    empty_plan.score_bytes.should eq(0_i64)
    empty_plan.output_bytes.should eq(0_i64)
    empty_plan.max_batch_length.should eq(0_i32)

    singleton_map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      2,
      {1, 1, 1}
    )
    singleton_flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        Array(Float32).new(12, 1.0_f32),
        ML::Shape.new(1_i32, 12_i32)
      ),
      singleton_map
    )
    singleton_plan = ML::Sparse::FullSelfAttentionPlanCPU.build(
      ML::Sparse::SelfAttentionQKVCPU.new(singleton_flat, 2)
    )
    singleton_plan.score_elements.should eq(2_i64)
    singleton_plan.score_bytes.should eq(8_i64)
  end

  it "rejects the default quadratic ceiling without allocating a score matrix" do
    point_count = 513_i32
    coordinates = Array(Int32).new(point_count * 4) do |index|
      row = index // 4
      case index % 4
      when 0 then 0_i32
      when 1 then row.to_i32
      else        0_i32
      end
    end
    map = ML::Sparse::CoordinateMap3D.new(coordinates, 1, {1024, 1, 1})
    flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        Array(Float32).new(point_count * 3, 1.0_f32),
        ML::Shape.new(point_count, 3_i32)
      ),
      map
    )
    view = ML::Sparse::SelfAttentionQKVCPU.new(flat, 1)

    expect_raises(ML::Sparse::SparseTensorBudgetError, /quadratic score budget/) do
      ML::Sparse::FullSelfAttentionPlanCPU.build(view)
    end
  end

  it "rejects attempts to widen the process-local score ceiling" do
    view = sparse_full_attention_view(sparse_full_attention_fixture)
    expect_raises(ML::Sparse::SparseTensorBudgetError, /must be in 1/) do
      ML::Sparse::FullSelfAttentionPlanCPU.build(
        view,
        ML::Sparse::FullSelfAttentionPlanCPU::MAX_SCORE_BYTES + 1_i64
      )
    end
  end

  it "requires canonical QKV and coordinate-map runtime types" do
    fixture = sparse_full_attention_fixture
    view = sparse_full_attention_view(fixture)
    inherited_view = SparseFullAttentionQKVOverride.new(
      view.flat_projection,
      view.num_heads
    )
    expect_raises(ML::Sparse::SparseTensorError, /base SelfAttentionQKVCPU/) do
      ML::Sparse::FullSelfAttentionPlanCPU.build(inherited_view)
    end

    input = fixture["input"]
    map = SparseFullAttentionMapOverride.new(
      sparse_full_attention_i32(input["coordinates"]),
      input["batch_size"].as_i.to_i32,
      {
        input["spatial_shape"][0].as_i.to_i32,
        input["spatial_shape"][1].as_i.to_i32,
        input["spatial_shape"][2].as_i.to_i32,
      }
    )
    flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        Array(Float32).new(72, 1.0_f32),
        ML::Shape.new(6_i32, 12_i32)
      ),
      map
    )
    expect_raises(ML::Sparse::SparseTensorError, /base CoordinateMap3D/) do
      ML::Sparse::FullSelfAttentionPlanCPU.build(
        ML::Sparse::SelfAttentionQKVCPU.new(flat, 2)
      )
    end
  end
end
