require "json"
require "digest/sha256"
require "../../src/ml/sparse/self_attention_qkv"
require "../spec_helper"

private def sparse_qkv_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_self_attention_qkv_cpu_v1.json"
  )))
end

private def sparse_qkv_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_qkv_f32(payload : JSON::Any) : Array(Float32)
  flatten_sparse_qkv_f32(payload, [] of Float32)
end

private def flatten_sparse_qkv_f32(
  payload : JSON::Any,
  output : Array(Float32),
) : Array(Float32)
  if array = payload.as_a?
    array.each { |entry| flatten_sparse_qkv_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_qkv_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def freeze_sparse_qkv(layer : ML::NN::Linear) : Nil
  layer.weight.requires_grad = false
  layer.bias.try { |bias| bias.requires_grad = false }
end

private def assign_sparse_qkv(
  layer : ML::NN::Linear,
  weight : Indexable(Float32),
  bias : Indexable(Float32),
) : Nil
  weight_data = layer.weight.data.cpu_data.not_nil!
  weight.each_with_index { |value, index| weight_data[index] = value }
  bias_data = layer.bias.not_nil!.data.cpu_data.not_nil!
  bias.each_with_index { |value, index| bias_data[index] = value }
end

class SparseQKVFlatOverride < ML::Sparse::TensorCPU
  def initialize(features : ML::Tensor, map : ML::Sparse::CoordinateMap3D)
    super(features, map)
  end
end

class SparseQKVReceiverOverride < ML::Sparse::TensorCPU
  def self.==(other : ML::Sparse::TensorCPU.class) : Bool
    true
  end
end

describe ML::Sparse::SelfAttentionQKVCPU do
  it "matches the pinned SparseMultiHeadAttention QKV projection and reshape" do
    fixture = sparse_qkv_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-self-attention-qkv-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    sources = provenance["sources"]
    sources["sparse_attention_modules"]["sha256"].as_s.should eq(
      "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf"
    )
    sources["sparse_basic"]["sha256"].as_s.should eq(
      "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
    )
    sources["sparse_modulated"]["sha256"].as_s.should eq(
      "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2"
    )
    sources["structured_flow"]["sha256"].as_s.should eq(
      "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021"
    )
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["weights"].as_s.should eq("synthetic")
    provenance["sparse_backend"].as_s.should eq("none")

    contract = fixture["contract"]
    contract["consumer"].as_s.should eq(
      "ModulatedSparseTransformerCrossBlock.self_attn"
    )
    contract["reshape_reuses_projected_storage"].as_bool.should be_true
    contract["logical_feature_shape"].as_a.map(&.as_i).should eq(
      [3_i64, 3_i64, 2_i64, 2_i64]
    )
    contract["sparse_shape"].as_a.map(&.as_i).should eq(
      [2_i64, 3_i64, 2_i64, 2_i64]
    )

    input = fixture["input"]
    coordinates = sparse_qkv_i32(input["coordinates"])
    features = sparse_qkv_f32(input["features"])
    weights = sparse_qkv_f32(fixture["attention"]["qkv_weight"])
    bias = sparse_qkv_f32(fixture["attention"]["qkv_bias"])
    expected = sparse_qkv_f32(fixture["output"]["logical_qkv_features"])
    sparse_qkv_f32le_sha256(features).should eq(
      input["features_f32le_sha256"].as_s
    )
    sparse_qkv_f32le_sha256(weights).should eq(
      fixture["attention"]["qkv_weight_f32le_sha256"].as_s
    )
    sparse_qkv_f32le_sha256(bias).should eq(
      fixture["attention"]["qkv_bias_f32le_sha256"].as_s
    )
    sparse_qkv_f32le_sha256(expected).should eq(
      fixture["output"]["projected_features_f32le_sha256"].as_s
    )

    map = ML::Sparse::CoordinateMap3D.new(coordinates, 2, {3, 3, 2})
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(features, ML::Shape.new(3_i32, 4_i32)),
      map,
      256_i64
    )
    layer = ML::NN::Linear.new(4, 12, device: ML::Tensor::Device::CPU)
    assign_sparse_qkv(layer, weights, bias)
    freeze_sparse_qkv(layer)

    output = ML::Sparse::TensorCPU.apply_self_attention_qkv(sparse, layer, 2)

    output.shape.should eq({2, 3, 2, 2})
    output.feature_shape.should eq({3, 3, 2, 2})
    output.point_count.should eq(3)
    output.channels.should eq(4)
    output.num_heads.should eq(2)
    output.head_dim.should eq(2)
    output.coordinate_map.same?(map).should be_true
    output.flat_projection.channels.should eq(12)
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, 1e-6)
    end
    3.times do |row|
      3.times do |component|
        2.times do |head|
          2.times do |channel|
            flat_index = row * 12 + component * 4 + head * 2 + channel
            output.feature(row, component, head, channel).should be_close(
              expected[flat_index],
              1e-6
            )
          end
        end
      end
    end

    layer.weight.data.cpu_data.not_nil![0] = -99.0_f32
    layer.bias.not_nil!.data.cpu_data.not_nil![0] = -99.0_f32
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, 1e-6)
    end
  end

  it "is a zero-copy logical view over an existing flat projection" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    flat = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 12), map)
    view = ML::Sparse::SelfAttentionQKVCPU.new(flat, 2)

    view.flat_projection.same?(flat).should be_true
    view.shape.should eq({1, 3, 2, 2})
    view.feature_shape.should eq({1, 3, 2, 2})
  end

  it "supports the upstream bias-free and empty sparse configuration" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 2, {1, 1, 1})
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 2_i32)),
      map
    )
    layer = ML::NN::Linear.new(
      2,
      6,
      bias: false,
      device: ML::Tensor::Device::CPU
    )
    freeze_sparse_qkv(layer)

    output = ML::Sparse::TensorCPU.apply_self_attention_qkv(sparse, layer, 1)
    output.shape.should eq({2, 3, 1, 2})
    output.features_copy.should be_empty
    output.coordinate_map.same?(map).should be_true
  end

  it "fails closed on malformed logical layouts" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    flat = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 12), map)
    expect_raises(ML::Sparse::SparseTensorError, /head count must be positive/) do
      ML::Sparse::SelfAttentionQKVCPU.new(flat, 0)
    end

    malformed = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 10), map)
    expect_raises(ML::Sparse::SparseTensorError, /divisible by three/) do
      ML::Sparse::SelfAttentionQKVCPU.new(malformed, 2)
    end

    wrong_heads = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 15), map)
    expect_raises(ML::Sparse::SparseTensorError, /channels 5.*heads 2/) do
      ML::Sparse::SelfAttentionQKVCPU.new(wrong_heads, 2)
    end
  end

  it "fails closed before projection on an invalid self-attention contract" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 4), map)

    wrong_input = ML::NN::Linear.new(3, 12, device: ML::Tensor::Device::CPU)
    freeze_sparse_qkv(wrong_input)
    expect_raises(ML::Sparse::SparseTensorError, /input channels 4.*in_features 3/) do
      ML::Sparse::TensorCPU.apply_self_attention_qkv(sparse, wrong_input, 2)
    end

    wrong_output = ML::NN::Linear.new(4, 8, device: ML::Tensor::Device::CPU)
    freeze_sparse_qkv(wrong_output)
    expect_raises(ML::Sparse::SparseTensorError, /out_features 8.*expected 12/) do
      ML::Sparse::TensorCPU.apply_self_attention_qkv(sparse, wrong_output, 2)
    end

    correct = ML::NN::Linear.new(4, 12, device: ML::Tensor::Device::CPU)
    freeze_sparse_qkv(correct)
    expect_raises(ML::Sparse::SparseTensorError, /channels 4.*heads 3/) do
      ML::Sparse::TensorCPU.apply_self_attention_qkv(sparse, correct, 3)
    end
  end

  it "keeps the existing bounded linear ceiling explicit" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {1, 1, 1})
    exact = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 85_i32)),
      map
    )
    exact_layer = ML::NN::Linear.new(
      85,
      255,
      device: ML::Tensor::Device::CPU
    )
    freeze_sparse_qkv(exact_layer)
    exact_output = ML::Sparse::TensorCPU.apply_self_attention_qkv(
      exact,
      exact_layer,
      5
    )
    exact_output.shape.should eq({1, 3, 5, 17})

    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 86_i32)),
      map
    )
    layer = ML::NN::Linear.new(86, 258, device: ML::Tensor::Device::CPU)
    freeze_sparse_qkv(layer)

    expect_raises(ML::Sparse::SparseTensorBudgetError, /3C <= 256/) do
      ML::Sparse::TensorCPU.apply_self_attention_qkv(sparse, layer, 2)
    end
  end

  it "rejects inherited view projections and receiver dispatch" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    inherited = SparseQKVFlatOverride.new(ML::Tensor.ones(1, 12), map)
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU projection/) do
      ML::Sparse::SelfAttentionQKVCPU.new(inherited, 2)
    end

    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)
    layer = ML::NN::Linear.new(1, 3, device: ML::Tensor::Device::CPU)
    freeze_sparse_qkv(layer)
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseQKVReceiverOverride.apply_self_attention_qkv(sparse, layer, 1)
    end
  end

  it "checks every logical index independently" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    view = ML::Sparse::SelfAttentionQKVCPU.new(
      ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 12), map),
      2
    )

    expect_raises(IndexError, /component -1/) { view.feature(0, -1, 0, 0) }
    expect_raises(IndexError, /component 3/) { view.feature(0, 3, 0, 0) }
    expect_raises(IndexError, /head 2/) { view.feature(0, 0, 2, 0) }
    expect_raises(IndexError, /channel 2/) { view.feature(0, 0, 0, 2) }
    expect_raises(IndexError, /row 1/) { view.feature(1, 0, 0, 0) }
  end
end
