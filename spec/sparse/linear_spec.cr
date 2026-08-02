require "json"
require "digest/sha256"
require "../../src/ml/sparse/linear"
require "../spec_helper"

private def sparse_linear_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_linear_cpu_v1.json"
  )))
end

private def sparse_linear_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_linear_f32(payload : JSON::Any) : Array(Float32)
  if payload.as_a.first?.try(&.as_a?)
    payload.as_a.flat_map do |row|
      row.as_a.map { |value| value.as_f.to_f32 }
    end
  else
    payload.as_a.map { |value| value.as_f.to_f32 }
  end
end

private def sparse_linear_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def freeze_sparse_linear(layer : ML::NN::Linear) : Nil
  layer.weight.requires_grad = false
  layer.bias.try { |bias| bias.requires_grad = false }
end

private def assign_sparse_linear(
  layer : ML::NN::Linear,
  weight : Indexable(Float32),
  bias : Indexable(Float32)? = nil,
) : Nil
  weight_data = layer.weight.data.cpu_data.not_nil!
  weight.each_with_index { |value, index| weight_data[index] = value }
  if expected_bias = bias
    bias_data = layer.bias.not_nil!.data.cpu_data.not_nil!
    expected_bias.each_with_index { |value, index| bias_data[index] = value }
  end
end

class SparseLinearReceiverOverride < ML::Sparse::TensorCPU
  def self.==(other : ML::Sparse::TensorCPU.class) : Bool
    true
  end

  def initialize(
    features : Array(Float32),
    map : ML::Sparse::CoordinateMap3D,
    point_count : Int32,
    channels : Int32,
    budget : Int64,
  )
    features[0] = Float32::NAN unless features.empty?
    super(features, map, point_count, channels, budget)
  end
end

describe ML::Sparse::TensorCPU do
  it "matches the pinned asymmetric SparseLinear oracle" do
    fixture = sparse_linear_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-linear-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    sources = provenance["sources"]
    sources["sparse_basic"]["sha256"].as_s.should eq(
      "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
    )
    sources["sparse_linear"]["sha256"].as_s.should eq(
      "733264d356556108bfbbf31ba757b4d29adde2d126f22a362288848c4c023ce2"
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

    input = fixture["input"]
    coordinates = sparse_linear_i32(input["coordinates"])
    features = sparse_linear_f32(input["features"])
    weights = sparse_linear_f32(fixture["linear"]["weight"])
    bias = sparse_linear_f32(fixture["linear"]["bias"])
    expected = sparse_linear_f32(fixture["output"]["features"])
    sparse_linear_f32le_sha256(features).should eq(
      input["features_f32le_sha256"].as_s
    )
    sparse_linear_f32le_sha256(weights).should eq(
      fixture["linear"]["weight_f32le_sha256"].as_s
    )
    sparse_linear_f32le_sha256(bias).should eq(
      fixture["linear"]["bias_f32le_sha256"].as_s
    )
    sparse_linear_f32le_sha256(expected).should eq(
      fixture["output"]["features_f32le_sha256"].as_s
    )

    map = ML::Sparse::CoordinateMap3D.new(coordinates, 2, {3, 3, 2})
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(features, ML::Shape.new(3_i32, 2_i32)),
      map,
      64_i64
    )
    layer = ML::NN::Linear.new(
      2,
      3,
      device: ML::Tensor::Device::CPU
    )
    assign_sparse_linear(layer, weights, bias)
    freeze_sparse_linear(layer)

    output = ML::Sparse::TensorCPU.apply_linear(sparse, layer)

    output.point_count.should eq(3)
    output.channels.should eq(3)
    output.coordinate_map.same?(map).should be_true
    output.max_feature_bytes.should eq(64_i64)
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, 1e-6)
    end

    layer.weight.data.cpu_data.not_nil![0] = -99.0_f32
    layer.bias.not_nil!.data.cpu_data.not_nil![0] = -99.0_f32
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, 1e-6)
    end
  end

  it "supports frozen bias-free layers and empty sparse values" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 2, {4, 4, 4})
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 2_i32)),
      map
    )
    layer = ML::NN::Linear.new(
      2,
      3,
      bias: false,
      device: ML::Tensor::Device::CPU
    )
    assign_sparse_linear(
      layer,
      [0.25_f32, -0.5_f32, 1.5_f32, 2.0_f32, -2.0_f32, 0.75_f32]
    )
    freeze_sparse_linear(layer)

    output = ML::Sparse::TensorCPU.apply_linear(sparse, layer)
    output.features_copy.should be_empty
    output.channels.should eq(3)
    output.coordinate_map.same?(map).should be_true
  end

  it "rejects trainable parameters, uninitialized input, and mismatched dimensions" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 2), map)
    trainable = ML::NN::Linear.new(
      2,
      1,
      device: ML::Tensor::Device::CPU
    )
    expect_raises(ML::Sparse::SparseTensorError, /frozen graphless parameters/) do
      ML::Sparse::TensorCPU.apply_linear(sparse, trainable)
    end

    freeze_sparse_linear(trainable)
    raw = ML::Sparse::TensorCPU.allocate
    expect_raises(ML::Sparse::SparseTensorError, /initialized sparse value/) do
      ML::Sparse::TensorCPU.apply_linear(raw, trainable)
    end

    mismatched = ML::NN::Linear.new(
      3,
      1,
      device: ML::Tensor::Device::CPU
    )
    freeze_sparse_linear(mismatched)
    expect_raises(ML::Sparse::SparseTensorError, /input channels 2.*in_features 3/) do
      ML::Sparse::TensorCPU.apply_linear(sparse, mismatched)
    end
  end

  it "rejects inherited subclass receivers before constructing output" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)
    layer = ML::NN::Linear.new(
      1,
      1,
      device: ML::Tensor::Device::CPU
    )
    freeze_sparse_linear(layer)

    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseLinearReceiverOverride.apply_linear(sparse, layer)
    end
  end

  it "fails closed on output channel and byte budgets" do
    empty_map = ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {1, 1, 1})
    empty = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 1_i32)),
      empty_map
    )
    exact = ML::NN::Linear.new(
      1,
      256,
      device: ML::Tensor::Device::CPU
    )
    freeze_sparse_linear(exact)
    ML::Sparse::TensorCPU.apply_linear(empty, exact).channels.should eq(256)

    too_wide = ML::NN::Linear.new(
      1,
      257,
      device: ML::Tensor::Device::CPU
    )
    freeze_sparse_linear(too_wide)
    expect_raises(ML::Sparse::SparseTensorBudgetError, /channel count 257/) do
      ML::Sparse::TensorCPU.apply_linear(empty, too_wide)
    end

    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0, 0, 1, 0, 0] of Int32,
      1,
      {2, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(2, 1), map, 8_i64)
    layer = ML::NN::Linear.new(
      1,
      2,
      device: ML::Tensor::Device::CPU
    )
    freeze_sparse_linear(layer)
    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 16 bytes, limit is 8/) do
      ML::Sparse::TensorCPU.apply_linear(sparse, layer)
    end
  end

  it "rejects non-finite parameters and non-finite computed outputs" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 1), map)
    layer = ML::NN::Linear.new(
      1,
      1,
      device: ML::Tensor::Device::CPU
    )
    freeze_sparse_linear(layer)
    layer.weight.data.cpu_data.not_nil![0] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorError, /weight\[0\].*finite/) do
      ML::Sparse::TensorCPU.apply_linear(sparse, layer)
    end

    layer.weight.data.cpu_data.not_nil![0] = Float32::MAX
    layer.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::MAX
    expect_raises(ML::Sparse::SparseTensorError, /output\[0\].*finite/) do
      ML::Sparse::TensorCPU.apply_linear(sparse, layer)
    end
  end
end
