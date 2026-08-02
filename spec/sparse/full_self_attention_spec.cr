require "json"
require "../../src/ml/sparse/full_self_attention"
require "../spec_helper"

private SPARSE_FULL_ATTENTION_TOLERANCE = 2.0e-5_f32

private def sparse_full_attention_executor_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_full_self_attention_plan_cpu_v1.json"
  )))
end

private def flatten_sparse_full_attention_executor_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if array = payload.as_a?
    array.each do |entry|
      flatten_sparse_full_attention_executor_f32(entry, output)
    end
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_full_attention_executor_i32(
  payload : JSON::Any,
) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_full_attention_executor_view(
  fixture : JSON::Any,
) : ML::Sparse::SelfAttentionQKVCPU
  input = fixture["input"]
  attention = fixture["attention"]
  point_count = input["coordinates"].as_a.size.to_i32
  num_heads = attention["num_heads"].as_i.to_i32
  head_dim = attention["head_dim"].as_i.to_i32
  map = ML::Sparse::CoordinateMap3D.new(
    sparse_full_attention_executor_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  flat = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      flatten_sparse_full_attention_executor_f32(input["qkv_features"]),
      ML::Shape.new(point_count, 3_i32 * num_heads * head_dim)
    ),
    map
  )
  ML::Sparse::SelfAttentionQKVCPU.new(flat, num_heads)
end

class SparseFullAttentionTensorCPUOverride < ML::Sparse::TensorCPU
end

describe ML::Sparse::TensorCPU do
  it "matches the pinned block-diagonal dense CPU reference" do
    fixture = sparse_full_attention_executor_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-full-self-attention-plan-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    fixture["provenance"]["upstream_sparse_backend_executed"].as_bool.should be_false
    fixture["contract"]["not_backend_parity"].as_bool.should be_true
    fixture["output"]["features_f32le_sha256"].as_s.should eq(
      "ce294683d1ec7f8fe46bd45066871a5c3bfeaaf257256f7d24478e700dd12112"
    )
    qkv = sparse_full_attention_executor_view(fixture)
    before = qkv.features_copy

    output = ML::Sparse::TensorCPU.apply_full_self_attention(qkv)
    expected = flatten_sparse_full_attention_executor_f32(
      fixture["output"]["features"]
    )

    output.coordinate_map.same?(qkv.coordinate_map).should be_true
    output.point_count.should eq(qkv.point_count)
    output.channels.should eq(qkv.channels)
    output.max_feature_bytes.should eq(qkv.max_feature_bytes)
    output.features_copy.size.should eq(expected.size)
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, SPARSE_FULL_ATTENTION_TOLERANCE)
    end
    qkv.features_copy.should eq(before)
  end

  it "preserves empty and singleton batches without cross-batch work" do
    empty_map = ML::Sparse::CoordinateMap3D.new(
      [] of Int32,
      3,
      {1, 1, 1}
    )
    empty_flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 6_i32)),
      empty_map
    )
    empty_output = ML::Sparse::TensorCPU.apply_full_self_attention(
      ML::Sparse::SelfAttentionQKVCPU.new(empty_flat, 1)
    )
    empty_output.coordinate_map.same?(empty_map).should be_true
    empty_output.point_count.should eq(0)
    empty_output.channels.should eq(2)
    empty_output.features_copy.should be_empty

    singleton_map = ML::Sparse::CoordinateMap3D.new(
      [1, 0, 0, 0] of Int32,
      3,
      {1, 1, 1}
    )
    singleton_flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [1.0_f32, -2.0_f32, 3.0_f32, 4.0_f32, 17.0_f32, -19.0_f32],
        ML::Shape.new(1_i32, 6_i32)
      ),
      singleton_map
    )
    singleton_output = ML::Sparse::TensorCPU.apply_full_self_attention(
      ML::Sparse::SelfAttentionQKVCPU.new(singleton_flat, 1)
    )
    singleton_output.coordinate_map.same?(singleton_map).should be_true
    singleton_output.features_copy.should eq([17.0_f32, -19.0_f32])
  end

  it "isolates non-empty batch slices and ignores a trailing empty batch" do
    map = ML::Sparse::CoordinateMap3D.new(
      [
        0, 0, 0, 0,
        0, 1, 0, 0,
        1, 2, 0, 0,
        1, 3, 0, 0,
      ] of Int32,
      3,
      {4, 1, 1}
    )
    flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [
          0.0_f32, 0.0_f32, 1.0_f32,
          0.0_f32, 0.0_f32, 3.0_f32,
          0.0_f32, 0.0_f32, 101.0_f32,
          0.0_f32, 0.0_f32, 103.0_f32,
        ],
        ML::Shape.new(4_i32, 3_i32)
      ),
      map
    )

    output = ML::Sparse::TensorCPU.apply_full_self_attention(
      ML::Sparse::SelfAttentionQKVCPU.new(flat, 1)
    )

    output.coordinate_map.same?(map).should be_true
    output.features_copy.should eq(
      [2.0_f32, 2.0_f32, 102.0_f32, 102.0_f32]
    )
  end

  it "rejects singleton F32 dot overflow instead of shortcutting to V" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [Float32::MAX, 2.0_f32, 1.0_f32],
        ML::Shape.new(1_i32, 3_i32)
      ),
      map
    )

    expect_raises(ML::Sparse::SparseTensorError, /score.*finite/) do
      ML::Sparse::TensorCPU.apply_full_self_attention(
        ML::Sparse::SelfAttentionQKVCPU.new(flat, 1)
      )
    end
  end

  it "applies the caller score budget before numerical execution" do
    qkv = sparse_full_attention_executor_view(
      sparse_full_attention_executor_fixture
    )
    before = qkv.features_copy

    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 112 bytes.*limit is 111/
    ) do
      ML::Sparse::TensorCPU.apply_full_self_attention(qkv, 111_i64)
    end
    qkv.features_copy.should eq(before)
  end

  it "requires the base TensorCPU receiver" do
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseFullAttentionTensorCPUOverride.apply_full_self_attention(
        sparse_full_attention_executor_view(
          sparse_full_attention_executor_fixture
        )
      )
    end
  end
end
