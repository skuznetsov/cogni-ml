require "json"
require "digest/sha256"
require "../../src/ml/sparse/mlp_first_stage"
require "../../src/ml/sparse/gated_residual"
require "../spec_helper"

private MLP_FIRST_STAGE_TOLERANCE      = 5e-5_f32
private MLP_FIRST_STAGE_FIXTURE_SHA256 =
  "7cda9277e2b68374fe6f9671cc785cf7b060b2dbd9038173b8a52d0f7b67c600"

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

private def sparse_mlp_i32le_sha256(values : Indexable(Int32)) : String
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

private def sparse_mlp_scalar_tail_reference(
  hidden : Indexable(Float32),
  weight : Indexable(Float32),
  bias : Indexable(Float32),
  point_count : Int32,
  hidden_channels : Int32,
  output_channels : Int32,
) : Array(Float32)
  output = Array(Float32).new(point_count.to_i * output_channels.to_i)
  point_count.times do |row|
    output_channels.times do |output_channel|
      sum = 0.0_f32
      hidden_channels.times do |hidden_channel|
        sum += hidden[row * hidden_channels + hidden_channel] *
               weight[output_channel * hidden_channels + hidden_channel]
      end
      output << sum + bias[output_channel]
    end
  end
  output
end

private def sparse_mlp_scalar_gated_residual_reference(
  residual : Indexable(Float32),
  update : Indexable(Float32),
  gate : Indexable(Float32),
  coordinates : Indexable(Int32),
  point_count : Int32,
  channels : Int32,
) : Array(Float32)
  output = Array(Float32).new(point_count.to_i * channels.to_i)
  point_count.times do |row|
    batch = coordinates[row * 4]
    channels.times do |channel|
      index = row * channels + channel
      output << residual[index] + update[index] * gate[batch * channels + channel]
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
      "cogni-ml/trellis2/sparse-mlp-oracle/v3"
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
      "8d5a205d0c2f91776c332b64db2d9d8b01c94b6e5e17cbeb60d0be31fddc6477"
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
      "SparseFeedForwardNet.forward -> mlp[0] -> mlp[1] -> mlp[2]"
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
    contract["tail_contract"].as_s.should eq("biased frozen Linear(H,C)")
    contract["full_output"].as_s.should eq(
      "SparseFeedForwardNet output after mlp[2]"
    )
    contract["total_work_rule"].as_s.should eq("2 * N * C * H")
    fixture["tail_linear"]["weight_f32le_sha256"].as_s.should eq(
      "9dd3d574395578e1e3c253a7364fc3692f909d2da05aea0c0ea8d938c60f9c23"
    )
    fixture["tail_linear"]["bias_f32le_sha256"].as_s.should eq(
      "c07c5a6e283880bf6a3eed8c52d591f0879d600d6ebf51f0b2b8f173e9a057e6"
    )

    input = fixture["input"]
    coordinates = sparse_mlp_i32(input["coordinates"])
    features = sparse_mlp_f32(input["features"])
    weight = sparse_mlp_f32(fixture["linear"]["weight"])
    bias = sparse_mlp_f32(fixture["linear"]["bias"])
    tail_weight = sparse_mlp_f32(fixture["tail_linear"]["weight"])
    tail_bias = sparse_mlp_f32(fixture["tail_linear"]["bias"])
    expected = sparse_mlp_f32(fixture["output"]["features"])
    full_expected = sparse_mlp_f32(fixture["full_output"]["features"])
    sparse_mlp_i32le_sha256(coordinates).should eq(
      input["coordinates_i32le_sha256"].as_s
    )
    sparse_mlp_f32le_sha256(features).should eq(input["features_f32le_sha256"].as_s)
    sparse_mlp_f32le_sha256(weight).should eq(fixture["linear"]["weight_f32le_sha256"].as_s)
    sparse_mlp_f32le_sha256(bias).should eq(fixture["linear"]["bias_f32le_sha256"].as_s)
    sparse_mlp_f32le_sha256(expected).should eq(fixture["output"]["features_f32le_sha256"].as_s)
    sparse_mlp_f32le_sha256(tail_weight).should eq(
      fixture["tail_linear"]["weight_f32le_sha256"].as_s
    )
    sparse_mlp_f32le_sha256(tail_bias).should eq(
      fixture["tail_linear"]["bias_f32le_sha256"].as_s
    )
    sparse_mlp_f32le_sha256(full_expected).should eq(
      fixture["full_output"]["features_f32le_sha256"].as_s
    )
    scalar = sparse_mlp_scalar_reference(
      features,
      weight,
      bias,
      input["point_count"].as_i.to_i32,
      input["channels"].as_i.to_i32,
      fixture["linear"]["out_channels"].as_i.to_i32
    )
    expected.each_with_index { |value, index| value.should be_close(scalar[index], MLP_FIRST_STAGE_TOLERANCE) }
    tail_scalar = sparse_mlp_scalar_tail_reference(
      expected,
      tail_weight,
      tail_bias,
      input["point_count"].as_i.to_i32,
      fixture["linear"]["out_channels"].as_i.to_i32,
      fixture["tail_linear"]["out_channels"].as_i.to_i32
    )
    full_expected.each_with_index do |value, index|
      value.should be_close(tail_scalar[index], MLP_FIRST_STAGE_TOLERANCE)
    end

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

  it "matches the complete SparseFeedForwardNet output through mlp[2]" do
    fixture = sparse_mlp_first_stage_fixture
    input = fixture["input"]
    coordinates = sparse_mlp_i32(input["coordinates"])
    features = sparse_mlp_f32(input["features"])
    first_weight = sparse_mlp_f32(fixture["linear"]["weight"])
    first_bias = sparse_mlp_f32(fixture["linear"]["bias"])
    tail_weight = sparse_mlp_f32(fixture["tail_linear"]["weight"])
    tail_bias = sparse_mlp_f32(fixture["tail_linear"]["bias"])
    expected = sparse_mlp_f32(fixture["full_output"]["features"])

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
    first = ML::NN::Linear.new(
      input["channels"].as_i.to_i32,
      fixture["linear"]["out_channels"].as_i.to_i32,
      device: ML::Tensor::Device::CPU
    )
    second = ML::NN::Linear.new(
      fixture["linear"]["out_channels"].as_i.to_i32,
      fixture["tail_linear"]["out_channels"].as_i.to_i32,
      device: ML::Tensor::Device::CPU
    )
    sparse_mlp_assign(first, first_weight, first_bias)
    sparse_mlp_assign(second, tail_weight, tail_bias)

    input_before = sparse.features_copy
    first_weight_before = first.weight.data.cpu_data.not_nil!.dup
    first_bias_before = first.bias.not_nil!.data.cpu_data.not_nil!.dup
    second_weight_before = second.weight.data.cpu_data.not_nil!.dup
    second_bias_before = second.bias.not_nil!.data.cpu_data.not_nil!.dup

    output = ML::Sparse::TensorCPU.apply_mlp(sparse, first, second)
    actual = output.features_copy
    actual.size.should eq(expected.size)
    actual.each_with_index do |value, index|
      value.should be_close(expected[index], MLP_FIRST_STAGE_TOLERANCE)
    end
    output.coordinate_map.same?(map).should be_true
    output.channels.should eq(fixture["tail_linear"]["out_channels"].as_i)
    output.production_width?.should be_false
    output.max_feature_bytes.should eq(input["max_feature_bytes"].as_i64)
    sparse.features_copy.should eq(input_before)
    first.weight.data.cpu_data.not_nil!.should eq(first_weight_before)
    first.bias.not_nil!.data.cpu_data.not_nil!.should eq(first_bias_before)
    second.weight.data.cpu_data.not_nil!.should eq(second_weight_before)
    second.bias.not_nil!.data.cpu_data.not_nil!.should eq(second_bias_before)
  end

  it "matches the source-pinned final gate_mlp broadcast and residual" do
    fixture = sparse_mlp_first_stage_fixture
    input = fixture["input"]
    contract = fixture["contract"]
    contract["modulated_final"].as_s.should eq(
      "x = residual + gate_mlp[batch] * SparseFeedForwardNet(mlp_input)"
    )
    contract["gate_broadcast"].as_s.should eq(
      "SparseTensor.__elemwise__ [B,C] -> batch_boardcast_map -> [N,C]"
    )
    contract["gate_shape"].as_a.map(&.as_i).should eq([3_i64, 3_i64])
    fixture["residual_input"]["coordinate_object_reused"].as_bool.should be_true
    fixture["final_output"]["coordinate_object_reused"].as_bool.should be_true
    coordinates = sparse_mlp_i32(input["coordinates"])
    mlp_features = sparse_mlp_f32(input["features"])
    residual_features = sparse_mlp_f32(fixture["residual_input"]["features"])
    first_weight = sparse_mlp_f32(fixture["linear"]["weight"])
    first_bias = sparse_mlp_f32(fixture["linear"]["bias"])
    tail_weight = sparse_mlp_f32(fixture["tail_linear"]["weight"])
    tail_bias = sparse_mlp_f32(fixture["tail_linear"]["bias"])
    gate_values = sparse_mlp_f32(fixture["gate_mlp"]["features"])
    expected = sparse_mlp_f32(fixture["final_output"]["features"])

    sparse_mlp_f32le_sha256(residual_features).should eq(
      fixture["residual_input"]["features_f32le_sha256"].as_s
    )
    sparse_mlp_f32le_sha256(gate_values).should eq(
      fixture["gate_mlp"]["features_f32le_sha256"].as_s
    )
    sparse_mlp_f32le_sha256(expected).should eq(
      fixture["final_output"]["features_f32le_sha256"].as_s
    )
    contract["batch_broadcast_map"].as_a.map(&.as_i).should eq(
      [0_i64, 0_i64, 1_i64, 1_i64]
    )

    point_count = input["point_count"].as_i.to_i32
    channels = input["channels"].as_i.to_i32
    scalar = sparse_mlp_scalar_gated_residual_reference(
      residual_features,
      sparse_mlp_f32(fixture["full_output"]["features"]),
      gate_values,
      coordinates,
      point_count,
      channels
    )
    expected.each_with_index do |value, index|
      value.should be_close(scalar[index], MLP_FIRST_STAGE_TOLERANCE)
    end

    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      input["batch_size"].as_i.to_i32,
      {
        input["spatial_shape"][0].as_i.to_i32,
        input["spatial_shape"][1].as_i.to_i32,
        input["spatial_shape"][2].as_i.to_i32,
      }
    )
    mlp_input = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(mlp_features, ML::Shape.new(point_count, channels)),
      map,
      input["max_feature_bytes"].as_i64
    )
    residual = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(residual_features, ML::Shape.new(point_count, channels)),
      map,
      input["max_feature_bytes"].as_i64
    )
    first = ML::NN::Linear.new(
      channels,
      fixture["linear"]["out_channels"].as_i.to_i32,
      device: ML::Tensor::Device::CPU
    )
    second = ML::NN::Linear.new(
      fixture["linear"]["out_channels"].as_i.to_i32,
      channels,
      device: ML::Tensor::Device::CPU
    )
    sparse_mlp_assign(first, first_weight, first_bias)
    sparse_mlp_assign(second, tail_weight, tail_bias)
    gate = ML::Tensor.from_array(
      gate_values,
      ML::Shape.new(input["batch_size"].as_i.to_i32, channels)
    )

    mlp_before = mlp_input.features_copy
    residual_before = residual.features_copy
    gate_before = gate.cpu_data.not_nil!.dup
    mlp_output = ML::Sparse::TensorCPU.apply_mlp(mlp_input, first, second)
    output = ML::Sparse::TensorCPU.apply_gated_residual(residual, mlp_output, gate)
    actual = output.features_copy

    actual.size.should eq(expected.size)
    actual.each_with_index do |value, index|
      value.should be_close(expected[index], MLP_FIRST_STAGE_TOLERANCE)
    end
    output.coordinate_map.same?(map).should be_true
    output.production_width?.should be_false
    output.max_feature_bytes.should eq(input["max_feature_bytes"].as_i64)
    mlp_input.features_copy.should eq(mlp_before)
    residual.features_copy.should eq(residual_before)
    gate.cpu_data.not_nil!.should eq(gate_before)
  end

  it "rejects complete-MLP bounds before poisoned payload or parameter reads" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    poisoned = SparseMLPFirstStageInputOverride.new(
      [Float32::NAN] * 49,
      map,
      1_i32,
      49_i32,
      1_i64
    )
    first = ML::NN::Linear.new(49, 261, device: ML::Tensor::Device::CPU)
    second = ML::NN::Linear.new(261, 49, device: ML::Tensor::Device::CPU)
    first.weight.data.cpu_data.not_nil![0] = Float32::NAN
    first.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::NAN
    second.weight.data.cpu_data.not_nil![0] = Float32::NAN
    second.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorBudgetError, /input channel count/) do
      ML::Sparse::TensorCPU.apply_mlp(poisoned, first, second)
    end

    low_work_input = SparseMLPFirstStageInputOverride.new(
      [Float32::NAN] * 3,
      map,
      1_i32,
      3_i32,
      4_i64 * 1024_i64
    )
    low_work_first = ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU)
    low_work_second = ML::NN::Linear.new(16, 3, device: ML::Tensor::Device::CPU)
    low_work_first.weight.data.cpu_data.not_nil![0] = Float32::NAN
    low_work_second.weight.data.cpu_data.not_nil![0] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorBudgetError, /work would require 96 MAC elements.*limit is 95/) do
      ML::Sparse::TensorCPU.apply_mlp(
        low_work_input,
        low_work_first,
        low_work_second,
        max_work_elements: 95_i64,
      )
    end
  end

  it "enforces the exact biased frozen H-to-C tail contract" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    sparse = ML::Sparse::TensorCPU.new(ML::Tensor.ones(1, 3), map)
    first = ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU)
    first.weight.requires_grad = false
    first.bias.not_nil!.requires_grad = false

    wrong_width = ML::NN::Linear.new(16, 2, device: ML::Tensor::Device::CPU)
    wrong_width.weight.requires_grad = false
    wrong_width.bias.not_nil!.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /tail output channels.*input channels/) do
      ML::Sparse::TensorCPU.apply_mlp(sparse, first, wrong_width)
    end

    no_bias = ML::NN::Linear.new(16, 3, bias: false, device: ML::Tensor::Device::CPU)
    no_bias.weight.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /tail.*requires a bias/) do
      ML::Sparse::TensorCPU.apply_mlp(sparse, first, no_bias)
    end

    trainable = ML::NN::Linear.new(16, 3, device: ML::Tensor::Device::CPU)
    expect_raises(ML::Sparse::SparseTensorError, /tail.*frozen graphless/) do
      ML::Sparse::TensorCPU.apply_mlp(sparse, first, trainable)
    end
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

    full_first = ML::NN::Linear.new(2, 10, device: ML::Tensor::Device::CPU)
    full_second = ML::NN::Linear.new(10, 2, device: ML::Tensor::Device::CPU)
    full_first.weight.data.cpu_data.not_nil![0] = Float32::NAN
    expect_raises(ML::Sparse::SparseTensorError, /bounded carrier/) do
      ML::Sparse::TensorCPU.apply_mlp(production, full_first, full_second)
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

    empty_tail_layer = ML::NN::Linear.new(16, 3, device: ML::Tensor::Device::CPU)
    empty_tail_layer.weight.requires_grad = false
    empty_tail_layer.bias.not_nil!.requires_grad = false
    empty_full_output = ML::Sparse::TensorCPU.apply_mlp(
      empty,
      empty_layer,
      empty_tail_layer,
    )
    empty_full_output.features_copy.should eq([] of Float32)
    empty_full_output.channels.should eq(3)
    empty_full_output.coordinate_map.same?(empty_map).should be_true

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

    finite_first = ML::NN::Linear.new(3, 16, device: ML::Tensor::Device::CPU)
    finite_first.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    finite_first.bias.not_nil!.data.cpu_data.not_nil!.fill(1.0_f32)
    finite_first.weight.requires_grad = false
    finite_first.bias.not_nil!.requires_grad = false
    overflowing_tail = ML::NN::Linear.new(16, 3, device: ML::Tensor::Device::CPU)
    overflowing_tail.weight.data.cpu_data.not_nil!.fill(Float32::MAX)
    overflowing_tail.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    overflowing_tail.weight.requires_grad = false
    overflowing_tail.bias.not_nil!.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /output\[0\].*finite/) do
      ML::Sparse::TensorCPU.apply_mlp(
        finite,
        finite_first,
        overflowing_tail,
      )
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

    second = ML::NN::Linear.new(16, 3, device: ML::Tensor::Device::CPU)
    second.weight.requires_grad = false
    second.bias.not_nil!.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseMLPFirstStageReceiverOverride.apply_mlp(sparse, layer, second)
    end
  end
end
