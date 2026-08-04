require "json"
require "digest/sha256"
require "../../src/ml/sparse/mlp_first_stage"
require "../spec_helper"

private MLP_FIRST_STAGE_TOLERANCE      = 5e-5_f32
private MLP_FIRST_STAGE_FIXTURE_SHA256 =
  "8fc03ff5300983e627f720c946c9669aff38b08cc074559869f82fadfbd2ae54"

private def sparse_mlp_first_stage_fixture : JSON::Any
  path = File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_mlp_first_stage_cpu_v1.json"
  )
  payload = File.read(path)
  Digest::SHA256.hexdigest(payload.to_slice).should eq(
    MLP_FIRST_STAGE_FIXTURE_SHA256
  )
  JSON.parse(payload)
end

private def sparse_mlp_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map { |row| row.as_a.map { |value| value.as_i.to_i32 } }
end

private def sparse_mlp_f32(payload : JSON::Any) : Array(Float32)
  payload.as_a.flat_map do |row|
    if nested = row.as_a?
      nested.map { |value| value.as_f.to_f32 }
    else
      [row.as_f.to_f32]
    end
  end
end

private def sparse_mlp_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def sparse_mlp_assign(
  layer : ML::NN::Linear,
  weight : Indexable(Float32),
  bias : Indexable(Float32),
) : Nil
  weight_data = layer.weight.data.cpu_data.not_nil!
  weight_data.each_index { |index| weight_data[index] = weight[index] }
  bias_data = layer.bias.not_nil!.data.cpu_data.not_nil!
  bias_data.each_index { |index| bias_data[index] = bias[index] }
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
end

private def sparse_mlp_scalar_reference(
  features : Indexable(Float32),
  weight : Indexable(Float32),
  bias : Indexable(Float32),
  point_count : Int32,
  channels : Int32,
  hidden_channels : Int32,
) : Array(Float32)
  output = Array(Float32).new(point_count.to_i * hidden_channels.to_i)
  point_count.times do |row|
    hidden_channels.times do |hidden_channel|
      sum = 0.0_f32
      channels.times do |channel|
        sum += features[row * channels + channel] *
               weight[hidden_channel * channels + channel]
      end
      projected = sum + bias[hidden_channel]
      cubic = projected * projected * projected
      argument = 0.7978845608028654_f32 *
                 (projected + 0.044715_f32 * cubic)
      output << 0.5_f32 * projected *
                (1.0_f32 + Math.tanh(argument.to_f64).to_f32)
    end
  end
  output
end

class SparseMLPFirstStageReceiverOverride < ML::Sparse::TensorCPU
  def self.==(other : ML::Sparse::TensorCPU.class) : Bool
    true
  end
end

class SparseMLPFirstStageInputOverride < ML::Sparse::TensorCPU
  def initialize(
    features : Array(Float32),
    map : ML::Sparse::CoordinateMap3D,
    point_count : Int32,
    channels : Int32,
    budget : Int64,
  )
    super(features, map, point_count, channels, budget)
  end
end

describe "TRELLIS.2 sparse MLP first projection and tanh GELU" do
  it "matches the source-pinned SparseFeedForwardNet first stage" do
    fixture = sparse_mlp_first_stage_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-mlp-first-stage-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["device"].as_s.should eq("cpu")
    provenance["dtype"].as_s.should eq("float32")
    provenance["network"].as_s.should eq("none")
    provenance["sparse_backend"].as_s.should eq("none")
    provenance["generator_sha256"].as_s.should eq(
      "d7e60a7c4f3c95f8ac83ee2abb3ae35afc35906f5427c7623d9afcd353abd1ef"
    )

    sources = provenance["sources"]
    sources["basic"]["sha256"].as_s.should eq(
      "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
    )
    sources["config"]["sha256"].as_s.should eq(
      "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b"
    )
    sources["blocks"]["sha256"].as_s.should eq(
      "622e7c5374976c053fb96151c706356b44244e241afab180e0fbdde5da6770d9"
    )
    sources["nonlinearity"]["sha256"].as_s.should eq(
      "bc0b7f1f57be9682d0cbbf024a55a3783a80d70c5d0e7f4bede84d13e10430bf"
    )
    sources["linear"]["sha256"].as_s.should eq(
      "733264d356556108bfbbf31ba757b4d29adde2d126f22a362288848c4c023ce2"
    )
    sources["modulated"]["sha256"].as_s.should eq(
      "fab9838c79b5fa9cbc6055c4a958f5a8e6f394f94e1691140be022caab7078d2"
    )
    configs = provenance["production_slat_configs"]
    configs["img2shape_512"]["sha256"].as_s.should eq(
      "6989e77f8b5ff4eb524522649e7708bee56526544f5d059f55760fcc5567d388"
    )
    configs["img2shape_ft1024"]["sha256"].as_s.should eq(
      "310f9588a6d3ebc7c036b1bb5be79e96343ff232cc9c5627e0d590f101949da0"
    )
    configs["imgshape2tex_512"]["sha256"].as_s.should eq(
      "a344cef8feca45a4efc2201c53e772ebd77b97f9aff1b1e91328576ab6f3e1c6"
    )
    configs["imgshape2tex_ft1024"]["sha256"].as_s.should eq(
      "df727c8b2bcd6fc592e4feb0489ddec57c73f2f4fdb5b4028ded8648d6d37057"
    )

    contract = fixture["contract"]
    contract["upstream_call"].as_s.should eq(
      "SparseFeedForwardNet.forward -> mlp[0] -> mlp[1]"
    )
    contract["activation"].as_s.should eq(
      "SparseGELU(approximate=\"tanh\")"
    )
    contract["hidden_rule"].as_s.should eq("int(C * 5.3334)")
    contract["production_model_channels"].as_i.should eq(1_536_i64)
    contract["production_mlp_ratio"].as_f.should eq(5.3334_f64)
    (contract["production_model_channels"].as_i.to_f64 *
      contract["production_mlp_ratio"].as_f).to_i64.should eq(8_192_i64)
    contract["tail_projection_executed"].as_bool.should be_true
    fixture["tail_linear"]["output_features_f32le_sha256"].as_s.should eq(
      "20460a290d9110f532c6e59b66500a34a3f1b8d1d3a002135811c4944615d722"
    )

    input = fixture["input"]
    coordinates = sparse_mlp_i32(input["coordinates"])
    features = sparse_mlp_f32(input["features"])
    weight = sparse_mlp_f32(fixture["linear"]["weight"])
    bias = sparse_mlp_f32(fixture["linear"]["bias"])
    expected = sparse_mlp_f32(fixture["output"]["features"])
    sparse_mlp_f32le_sha256(features).should eq(input["features_f32le_sha256"].as_s)
    sparse_mlp_f32le_sha256(weight).should eq(fixture["linear"]["weight_f32le_sha256"].as_s)
    sparse_mlp_f32le_sha256(bias).should eq(fixture["linear"]["bias_f32le_sha256"].as_s)
    sparse_mlp_f32le_sha256(expected).should eq(fixture["output"]["features_f32le_sha256"].as_s)
    scalar = sparse_mlp_scalar_reference(
      features,
      weight,
      bias,
      input["point_count"].as_i.to_i32,
      input["channels"].as_i.to_i32,
      fixture["linear"]["out_channels"].as_i.to_i32
    )
    expected.each_with_index { |value, index| value.should be_close(scalar[index], MLP_FIRST_STAGE_TOLERANCE) }

    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      input["batch_size"].as_i.to_i32,
      {
        input["spatial_shape"][0].as_i.to_i32,
        input["spatial_shape"][1].as_i.to_i32,
        input["spatial_shape"][2].as_i.to_i32,
      }
    )
    sparse = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        features,
        ML::Shape.new(input["point_count"].as_i.to_i32, input["channels"].as_i.to_i32)
      ),
      map,
      input["max_feature_bytes"].as_i64
    )
    layer = ML::NN::Linear.new(
      input["channels"].as_i.to_i32,
      fixture["linear"]["out_channels"].as_i.to_i32,
      device: ML::Tensor::Device::CPU
    )
    sparse_mlp_assign(layer, weight, bias)
    input_before = sparse.features_copy
    weight_before = layer.weight.data.cpu_data.not_nil!.dup
    bias_before = layer.bias.not_nil!.data.cpu_data.not_nil!.dup

    output = ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(sparse, layer)
    actual = output.features_copy
    actual.size.should eq(expected.size)
    actual.each_with_index { |value, index| value.should be_close(expected[index], MLP_FIRST_STAGE_TOLERANCE) }
    output.coordinate_map.same?(map).should be_true
    output.channels.should eq(fixture["linear"]["out_channels"].as_i)
    output.production_width?.should be_false
    output.max_feature_bytes.should eq(input["max_feature_bytes"].as_i64)
    sparse.features_copy.should eq(input_before)
    layer.weight.data.cpu_data.not_nil!.should eq(weight_before)
    layer.bias.not_nil!.data.cpu_data.not_nil!.should eq(bias_before)
  end

  it "accepts C=48 and rejects C=49 before reading poisoned payloads" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    valid = ML::Sparse::TensorCPU.new(ML::Tensor.zeros(1, 48), map)
    valid_layer = ML::NN::Linear.new(48, 256, device: ML::Tensor::Device::CPU)
    valid_layer.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    valid_layer.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    valid_layer.weight.requires_grad = false
    valid_layer.bias.not_nil!.requires_grad = false
    valid_output = ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(valid, valid_layer)
    valid_output.channels.should eq(256)
    valid_output.features_copy.all?(&.finite?).should be_true

    poisoned = SparseMLPFirstStageInputOverride.new(
      [Float32::NAN] * 49,
      map,
      1_i32,
      49_i32,
      1_i64
    )
    layer = ML::NN::Linear.new(49, 261, device: ML::Tensor::Device::CPU)
    expect_raises(ML::Sparse::SparseTensorBudgetError, /input channel count/) do
      ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(poisoned, layer)
    end
  end

  it "rejects production carriers before payload or parameter reads" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    production = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array(
        [1.0_f32, 2.0_f32],
        ML::Shape.new(1_i32, 2_i32)
      ),
      map,
      8_i64
    )
    layer = ML::NN::Linear.new(2, 16, device: ML::Tensor::Device::CPU)
    layer.weight.data.cpu_data.not_nil![0] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorError, /bounded carrier/) do
      ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(production, layer)
    end
  end

  it "enforces the exact frozen affine contract and output/work budgets" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 3), map, 1_i64 * 1024_i64)

    wrong_width = ML::NN::Linear.new(3, 15, device: ML::Tensor::Device::CPU)
    wrong_width.weight.requires_grad = false
    wrong_width.bias.not_nil!.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /output channels.*int\(C \* 5\.3334\)/) do
      ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(sparse, wrong_width)
    end

    no_bias = ML::NN::Linear.new(3, 16, bias: false, device: ML::Tensor::Device::CPU)
    no_bias.weight.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /requires a bias/) do
      ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(sparse, no_bias)
    end

    trainable = ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU)
    expect_raises(ML::Sparse::SparseTensorError, /frozen graphless/) do
      ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(sparse, trainable)
    end

    expect_raises(ML::Sparse::SparseTensorBudgetError, /output features require 64 bytes.*limit is 16/) do
      ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(
        ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 3), map, 16_i64),
        ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU),
      )
    end

    bounded_layer = ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU)
    bounded_layer.weight.requires_grad = false
    bounded_layer.bias.not_nil!.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorBudgetError, /work would require 48 MAC elements.*limit is 47/) do
      ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(
        sparse,
        bounded_layer,
        max_work_elements: 47_i64,
      )
    end
  end

  it "handles empty trailing batches and rejects non-finite outputs" do
    empty_map = ML::Sparse::CoordinateMap3D.new(
      [] of Int32,
      3,
      {1, 1, 1}
    )
    empty = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 3_i32)),
      empty_map,
      1024_i64
    )
    empty_layer = ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU)
    empty_layer.weight.requires_grad = false
    empty_layer.bias.not_nil!.requires_grad = false
    empty_output = ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(empty, empty_layer)
    empty_output.features_copy.should eq([] of Float32)
    empty_output.coordinate_map.same?(empty_map).should be_true

    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    finite = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 3), map)
    overflowing = ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU)
    overflowing.weight.data.cpu_data.not_nil!.fill(Float32::MAX)
    overflowing.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    overflowing.weight.requires_grad = false
    overflowing.bias.not_nil!.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /output\[0\].*finite/) do
      ML::Sparse::TensorCPU.apply_mlp_first_projection_gelu(finite, overflowing)
    end
  end

  it "rejects receiver spoofing before executing the operation" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.zeros(1, 3), map)
    layer = ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU)
    layer.weight.requires_grad = false
    layer.bias.not_nil!.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseMLPFirstStageReceiverOverride.apply_mlp_first_projection_gelu(sparse, layer)
    end
  end
end
