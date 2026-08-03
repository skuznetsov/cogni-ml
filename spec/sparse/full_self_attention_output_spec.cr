require "json"
require "digest/sha256"
require "../../src/ml/sparse/full_self_attention_output"
require "../spec_helper"

private SPARSE_FULL_ATTENTION_OUTPUT_TOLERANCE = 5.0e-5_f32
private SPARSE_FULL_ATTENTION_OUTPUT_DIGESTS   = {
  to_out_weight: "a040058c3bf0d0eec832fe412011d85a30e0a0bd6108447109a55a400d7b6b92",
  to_out_bias:   "ca50355c11f3f48babbb8e611341624b9f1931c87b26caf2934ef4df5fbc2178",
  pre_output:    "fa270952088de696aab6c8cc4fe78eac08660a96a76b711dcd489b24d7f3b8cb",
  final_output:  "2c2feca9cdb8981cc2a5ce9dc1ec0e1640be4673531889c6ec71376e79f05588",
}

private def sparse_full_attention_output_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_full_self_attention_output_cpu_v1.json"
  )))
end

private def flatten_sparse_full_attention_output_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if array = payload.as_a?
    array.each do |entry|
      flatten_sparse_full_attention_output_f32(entry, output)
    end
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_full_attention_output_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def sparse_full_attention_output_input(
  fixture : JSON::Any,
) : ML::Sparse::TensorCPU
  input = fixture["input"]
  channels = fixture["attention"]["channels"].as_i.to_i32
  coordinates = input["coordinates"].as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
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
  ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      flatten_sparse_full_attention_output_f32(input["features"]),
      ML::Shape.new(input["coordinates"].as_a.size.to_i32, channels)
    ),
    map
  )
end

private def sparse_full_attention_output_qkv(
  fixture : JSON::Any,
) : ML::NN::Linear
  attention = fixture["attention"]
  channels = attention["channels"].as_i.to_i32
  layer = ML::NN::Linear.new(
    channels,
    channels * 3,
    device: ML::Tensor::Device::CPU
  )
  weight = flatten_sparse_full_attention_output_f32(attention["qkv_weight"])
  weight_data = layer.weight.data.cpu_data.not_nil!
  weight.each_with_index { |value, index| weight_data[index] = value }
  bias = flatten_sparse_full_attention_output_f32(attention["qkv_bias_values"])
  bias_data = layer.bias.not_nil!.data.cpu_data.not_nil!
  bias.each_with_index { |value, index| bias_data[index] = value }
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def sparse_full_attention_output_projection(
  fixture : JSON::Any,
) : ML::NN::Linear
  attention = fixture["attention"]
  channels = attention["channels"].as_i.to_i32
  layer = ML::NN::Linear.new(
    channels,
    channels,
    device: ML::Tensor::Device::CPU
  )
  weight = flatten_sparse_full_attention_output_f32(attention["to_out_weight"])
  weight_data = layer.weight.data.cpu_data.not_nil!
  weight.each_with_index { |value, index| weight_data[index] = value }
  bias = flatten_sparse_full_attention_output_f32(attention["to_out_bias_values"])
  bias_data = layer.bias.not_nil!.data.cpu_data.not_nil!
  bias.each_with_index { |value, index| bias_data[index] = value }
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def sparse_full_attention_output_gamma(
  fixture : JSON::Any,
  name : String,
) : ML::Tensor
  attention = fixture["attention"]
  ML::Tensor.from_array(
    flatten_sparse_full_attention_output_f32(attention[name]),
    ML::Shape.new(
      attention["num_heads"].as_i.to_i32,
      attention["head_dim"].as_i.to_i32
    )
  )
end

private def freeze_sparse_full_attention_output_linear(
  layer : ML::NN::Linear,
) : ML::NN::Linear
  layer.weight.requires_grad = false
  layer.bias.try { |bias| bias.requires_grad = false }
  layer
end

class SparseFullAttentionOutputTensorCPUOverride < ML::Sparse::TensorCPU
end

describe "TRELLIS.2 sparse full self-attention output composition" do
  it "matches the source-bound asymmetric biased final projection" do
    fixture = sparse_full_attention_output_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-full-self-attention-output-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_sparse_full_self_attention_output.py"
    )
    provenance["final_output_projection"].as_s.should eq(
      "frozen asymmetric biased Linear"
    )
    contract = fixture["contract"]
    contract["final_order"].as_a.map(&.as_s).should eq([
      "to_qkv",
      "packed [N, 3, H, D] view",
      "Q/K RMS normalization",
      "3D RoPE",
      "block-diagonal full attention",
      "flatten heads to [N, C]",
      "biased to_out Linear(C, C)",
    ])
    contract["final_boundary"].as_s.should eq("post-output projection")
    contract["final_coordinate_object_reused"].as_bool.should be_true
    contract["output_projection_non_identity"].as_bool.should be_true
    contract["qkv_bias_independent_from_to_out_bias"].as_bool.should be_true

    attention = fixture["attention"]
    attention["to_out_bias"].as_bool.should be_true
    {
      {"to_out_weight", "to_out_weight_f32le_sha256", :to_out_weight},
      {"to_out_bias_values", "to_out_bias_f32le_sha256", :to_out_bias},
    }.each do |values_name, digest_name, expected_name|
      values = flatten_sparse_full_attention_output_f32(attention[values_name])
      sparse_full_attention_output_f32le_sha256(values).should eq(
        SPARSE_FULL_ATTENTION_OUTPUT_DIGESTS[expected_name]
      )
      attention[digest_name].as_s.should eq(
        SPARSE_FULL_ATTENTION_OUTPUT_DIGESTS[expected_name]
      )
    end
    stages = fixture["stages"]
    {
      {"attention_output", :pre_output},
      {"final_output", :final_output},
    }.each do |stage_name, expected_name|
      stage = stages[stage_name]
      values = flatten_sparse_full_attention_output_f32(stage["features"])
      sparse_full_attention_output_f32le_sha256(values).should eq(
        SPARSE_FULL_ATTENTION_OUTPUT_DIGESTS[expected_name]
      )
      stage["features_f32le_sha256"].as_s.should eq(
        SPARSE_FULL_ATTENTION_OUTPUT_DIGESTS[expected_name]
      )
    end

    input = sparse_full_attention_output_input(fixture)
    before = input.features_copy
    to_qkv = sparse_full_attention_output_qkv(fixture)
    to_out = sparse_full_attention_output_projection(fixture)
    q_gamma = sparse_full_attention_output_gamma(fixture, "q_gamma")
    k_gamma = sparse_full_attention_output_gamma(fixture, "k_gamma")
    output = ML::Sparse::TensorCPU.apply_full_self_attention_output(
      input,
      to_qkv,
      to_out,
      2,
      q_gamma: q_gamma,
      k_gamma: k_gamma,
      use_rope: true
    )
    expected = flatten_sparse_full_attention_output_f32(
      stages["final_output"]["features"]
    )
    output.shape.should eq({3, 16})
    output.coordinate_map.same?(input.coordinate_map).should be_true
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, SPARSE_FULL_ATTENTION_OUTPUT_TOLERANCE)
    end
    output.features_copy.should_not eq(
      flatten_sparse_full_attention_output_f32(
        stages["attention_output"]["features"]
      )
    )
    input.features_copy.should eq(before)

    owned = output.features_copy
    to_qkv.weight.data.cpu_data.not_nil![0] = Float32::NAN
    to_out.weight.data.cpu_data.not_nil![0] = Float32::NAN
    to_out.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::NAN
    output.features_copy.should eq(owned)
  end

  it "requires the exact frozen biased C-to-C upstream boundary before QKV" do
    fixture = sparse_full_attention_output_fixture
    input = sparse_full_attention_output_input(fixture)
    channels = fixture["attention"]["channels"].as_i.to_i32
    poisoned_qkv = sparse_full_attention_output_qkv(fixture)
    poisoned_qkv.weight.data.cpu_data.not_nil![0] = Float32::NAN

    wrong_shape = freeze_sparse_full_attention_output_linear(
      ML::NN::Linear.new(channels, channels + 1, device: ML::Tensor::Device::CPU)
    )
    expect_raises(ML::Sparse::SparseTensorError, /Linear\(16, 16\)/) do
      ML::Sparse::TensorCPU.apply_full_self_attention_output(
        input, poisoned_qkv, wrong_shape, 2
      )
    end

    biasless = freeze_sparse_full_attention_output_linear(
      ML::NN::Linear.new(
        channels,
        channels,
        bias: false,
        device: ML::Tensor::Device::CPU
      )
    )
    expect_raises(ML::Sparse::SparseTensorError, /biased to_out/) do
      ML::Sparse::TensorCPU.apply_full_self_attention_output(
        input, poisoned_qkv, biasless, 2
      )
    end

    trainable = ML::NN::Linear.new(
      channels,
      channels,
      device: ML::Tensor::Device::CPU
    )
    expect_raises(ML::Sparse::SparseTensorError, /frozen/) do
      ML::Sparse::TensorCPU.apply_full_self_attention_output(
        input, poisoned_qkv, trainable, 2
      )
    end
  end

  it "keeps qkv_bias independent from the required to_out bias on empty input" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 2, {1, 1, 1})
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 4_i32)),
      map
    )
    to_qkv = freeze_sparse_full_attention_output_linear(
      ML::NN::Linear.new(
        4,
        12,
        bias: false,
        device: ML::Tensor::Device::CPU
      )
    )
    to_out = freeze_sparse_full_attention_output_linear(
      ML::NN::Linear.new(4, 4, device: ML::Tensor::Device::CPU)
    )

    output = ML::Sparse::TensorCPU.apply_full_self_attention_output(
      input, to_qkv, to_out, 2
    )
    output.shape.should eq({2, 4})
    output.features_copy.should be_empty
    output.coordinate_map.same?(map).should be_true
  end

  it "validates final parameters even when the sparse input is empty" do
    # Final storage/finiteness validation is intentionally leaf-local: after
    # score admission and context construction, before final-output allocation.
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {1, 1, 1})
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 4_i32)),
      map
    )
    to_qkv = freeze_sparse_full_attention_output_linear(
      ML::NN::Linear.new(4, 12, device: ML::Tensor::Device::CPU)
    )
    to_out = freeze_sparse_full_attention_output_linear(
      ML::NN::Linear.new(4, 4, device: ML::Tensor::Device::CPU)
    )
    to_out.weight.data.cpu_data.not_nil![0] = Float32::NAN

    expect_raises(ML::Sparse::SparseTensorError, /weight\[0\] must be finite/) do
      ML::Sparse::TensorCPU.apply_full_self_attention_output(
        input, to_qkv, to_out, 2
      )
    end

    to_out.weight.data.cpu_data.not_nil![0] = 0.0_f32
    to_out.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::INFINITY

    expect_raises(ML::Sparse::SparseTensorError, /bias\[0\] must be finite/) do
      ML::Sparse::TensorCPU.apply_full_self_attention_output(
        input, to_qkv, to_out, 2
      )
    end
  end

  it "preserves score-budget admission before parameter arithmetic" do
    fixture = sparse_full_attention_output_fixture
    input = sparse_full_attention_output_input(fixture)
    to_qkv = sparse_full_attention_output_qkv(fixture)
    to_out = sparse_full_attention_output_projection(fixture)
    to_qkv.weight.data.cpu_data.not_nil![0] = Float32::NAN
    to_out.weight.data.cpu_data.not_nil![0] = Float32::NAN

    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 64 bytes.*limit is 63/
    ) do
      ML::Sparse::TensorCPU.apply_full_self_attention_output(
        input, to_qkv, to_out, 2, max_score_bytes: 63_i64
      )
    end
  end

  it "requires the base TensorCPU receiver" do
    fixture = sparse_full_attention_output_fixture
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseFullAttentionOutputTensorCPUOverride.apply_full_self_attention_output(
        sparse_full_attention_output_input(fixture),
        sparse_full_attention_output_qkv(fixture),
        sparse_full_attention_output_projection(fixture),
        2
      )
    end
  end
end
