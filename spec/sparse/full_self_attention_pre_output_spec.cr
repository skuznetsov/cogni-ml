require "json"
require "digest/sha256"
require "../../src/ml/sparse/full_self_attention_pre_output"
require "../spec_helper"

private SPARSE_FULL_ATTENTION_PRE_OUTPUT_STAGE_TOLERANCE = 1.0e-5_f32
private SPARSE_FULL_ATTENTION_PRE_OUTPUT_TOLERANCE       = 5.0e-5_f32
private SPARSE_FULL_ATTENTION_PRE_OUTPUT_DIGESTS         = {
  input_features:   "29e41c0d939409a1d5de4c94f252e3a5f9d37bda8341998e83ddf1603cfee092",
  coordinates:      "c6afe4c9dc3de690827ffac56ec827597b8602b7257099c50f2073e8c8854a5b",
  qkv_weight:       "f0dc3161db1263eaf342e2f44fb5fa9ebad90ffa4e1c1140499b59508b4815c9",
  qkv_bias:         "2f49dc419c4a876ab71d9412d43d28ef5b5db5e533c18ab50746e5f0dbcac9f0",
  q_gamma:          "029be74caff11a210de58cc0b4de872dc283dae215dd1480ade78c11ca10602b",
  k_gamma:          "9081e77f5cdf26979c0c6e9dad34a812849f1ec911b1e7e46d6cd07ef68449a0",
  projected_qkv:    "3f78eb64b012dd57b89f300ec7c8ec2efe488c6b44df35f3f59ac4c24b67a1f8",
  normalized_qkv:   "864c046a79f8e177cceeb7f543ac86072ccef6b4175ccf940382f4996e1b5a68",
  roped_qkv:        "ae00c0095e77350644b25d4aca43f1dae24bbdce613ec0f813e2d87965e62341",
  attention_output: "fa270952088de696aab6c8cc4fe78eac08660a96a76b711dcd489b24d7f3b8cb",
}
private SPARSE_FULL_ATTENTION_PRE_OUTPUT_SOURCE_DIGESTS = {
  sparse_full_attention:    "bee0c32089f060c8136292f41a2cc7a952a8679a8b6d113d14c772f2c681e520",
  sparse_attention_modules: "cfa99afda24e5840118814e80cefae783423d01d47e6322fe967412aff11f6cf",
  sparse_rope:              "0525164901c3f1c885e961b747c856b654a4d6840882f34827677fb017dcec00",
  sparse_basic:             "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78",
  sparse_config:            "6a9cb44608829cb2c11591685282959928c6081c5bc659687aa8395765c5f91b",
  dense_full_attention:     "64c43354780dcbc3dcf7612ac5e53d6e21c2081234ea63cd329a77f4185dadfc",
  structured_flow:          "76454ead55d112214e36db8de5e9b3d1d4128f05d25256fb6c581b4c1a588021",
}

private def sparse_full_attention_pre_output_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_full_self_attention_output_cpu_v1.json"
  )))
end

private def flatten_sparse_full_attention_pre_output_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if array = payload.as_a?
    array.each do |entry|
      flatten_sparse_full_attention_pre_output_f32(entry, output)
    end
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_full_attention_pre_output_i32(
  payload : JSON::Any,
) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_full_attention_pre_output_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def sparse_full_attention_pre_output_i32le_sha256(
  values : Indexable(Int32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def assert_sparse_full_attention_pre_output_f32_digest(
  payload : JSON::Any,
  digest : JSON::Any,
  expected : String,
) : Nil
  values = flatten_sparse_full_attention_pre_output_f32(payload)
  sparse_full_attention_pre_output_f32le_sha256(values).should eq(expected)
  digest.as_s.should eq(expected)
end

private def sparse_full_attention_pre_output_linear(
  fixture : JSON::Any,
) : ML::NN::Linear
  attention = fixture["attention"]
  channels = attention["channels"].as_i.to_i32
  weights = flatten_sparse_full_attention_pre_output_f32(
    attention["qkv_weight"]
  )
  bias = flatten_sparse_full_attention_pre_output_f32(
    attention["qkv_bias_values"]
  )
  layer = ML::NN::Linear.new(
    channels,
    channels * 3,
    device: ML::Tensor::Device::CPU
  )
  weight_data = layer.weight.data.cpu_data.not_nil!
  weights.each_with_index { |value, index| weight_data[index] = value }
  bias_data = layer.bias.not_nil!.data.cpu_data.not_nil!
  bias.each_with_index { |value, index| bias_data[index] = value }
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def sparse_full_attention_pre_output_input(
  fixture : JSON::Any,
) : ML::Sparse::TensorCPU
  input = fixture["input"]
  channels = fixture["attention"]["channels"].as_i.to_i32
  coordinates = sparse_full_attention_pre_output_i32(input["coordinates"])
  point_count = input["coordinates"].as_a.size.to_i32
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
      flatten_sparse_full_attention_pre_output_f32(input["features"]),
      ML::Shape.new(point_count, channels)
    ),
    map
  )
end

private def sparse_full_attention_pre_output_gamma(
  fixture : JSON::Any,
  name : String,
) : ML::Tensor
  attention = fixture["attention"]
  ML::Tensor.from_array(
    flatten_sparse_full_attention_pre_output_f32(attention[name]),
    ML::Shape.new(
      attention["num_heads"].as_i.to_i32,
      attention["head_dim"].as_i.to_i32
    )
  )
end

private def sparse_full_attention_pre_output_manual(
  input : ML::Sparse::TensorCPU,
  linear : ML::NN::Linear,
  num_heads : Int32,
  q_gamma : ML::Tensor?,
  k_gamma : ML::Tensor?,
  use_rope : Bool,
) : ML::Sparse::TensorCPU
  qkv = ML::Sparse::TensorCPU.apply_self_attention_qkv(
    input,
    linear,
    num_heads
  )
  if q = q_gamma
    qkv = ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
      qkv,
      q,
      k_gamma.not_nil!
    )
  end
  if use_rope
    qkv = ML::Sparse::TensorCPU.apply_self_attention_rope(qkv)
  end
  ML::Sparse::TensorCPU.apply_full_self_attention(qkv)
end

class SparseFullAttentionPreOutputTensorCPUOverride < ML::Sparse::TensorCPU
end

describe "TRELLIS.2 sparse full self-attention pre-output composition" do
  it "matches the source-bound upstream self/full forward boundary" do
    fixture = sparse_full_attention_pre_output_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-full-self-attention-output-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["repository"].as_s.should eq("microsoft/TRELLIS.2")
    provenance["python_version"].as_s.should eq("3.11.9")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["weights"].as_s.should eq("synthetic")
    provenance["upstream_forward_executed"].as_bool.should be_true
    provenance["upstream_sparse_backend_executed"].as_bool.should be_false
    provenance["upstream_dense_reference_executed"].as_bool.should be_true
    provenance["output_projection"].as_s.should eq("identity substitution")
    SPARSE_FULL_ATTENTION_PRE_OUTPUT_SOURCE_DIGESTS.each do |name, digest|
      provenance["sources"][name.to_s]["sha256"].as_s.should eq(digest)
    end
    contract = fixture["contract"]
    contract["consumer"].as_s.should eq(
      "SparseMultiHeadAttention.forward self/full path"
    )
    contract["order"].as_a.map(&.as_s).should eq([
      "to_qkv",
      "packed [N, 3, H, D] view",
      "Q/K RMS normalization",
      "3D RoPE",
      "block-diagonal full attention",
    ])
    contract["boundary"].as_s.should eq("pre-output projection")
    contract["not_sparse_backend_parity"].as_bool.should be_true
    contract["qk_rms_norm"].as_bool.should be_true
    contract["use_rope"].as_bool.should be_true
    contract["sequence_lengths"].as_a.map(&.as_i).should eq([2_i64, 0_i64, 2_i64])

    fixture_input = fixture["input"]
    fixture_input["batch_size"].as_i.should eq(3_i64)
    fixture_input["spatial_shape"].as_a.map(&.as_i).should eq(
      [3_i64, 3_i64, 3_i64]
    )
    coordinates = sparse_full_attention_pre_output_i32(
      fixture_input["coordinates"]
    )
    sparse_full_attention_pre_output_i32le_sha256(coordinates).should eq(
      SPARSE_FULL_ATTENTION_PRE_OUTPUT_DIGESTS[:coordinates]
    )
    fixture_input["coordinates_i32le_sha256"].as_s.should eq(
      SPARSE_FULL_ATTENTION_PRE_OUTPUT_DIGESTS[:coordinates]
    )
    assert_sparse_full_attention_pre_output_f32_digest(
      fixture_input["features"],
      fixture_input["features_f32le_sha256"],
      SPARSE_FULL_ATTENTION_PRE_OUTPUT_DIGESTS[:input_features]
    )
    attention = fixture["attention"]
    attention["channels"].as_i.should eq(16_i64)
    attention["num_heads"].as_i.should eq(2_i64)
    attention["head_dim"].as_i.should eq(8_i64)
    attention["qkv_bias"].as_bool.should be_true
    attention["rope_freq"].as_a.map(&.as_f).should eq([1.0, 10000.0])
    attention["scale"].as_f.should be_close(1.0 / Math.sqrt(8.0), 1.0e-15)
    {
      {"qkv_weight", "qkv_weight_f32le_sha256", :qkv_weight},
      {"qkv_bias_values", "qkv_bias_f32le_sha256", :qkv_bias},
      {"q_gamma", "q_gamma_f32le_sha256", :q_gamma},
      {"k_gamma", "k_gamma_f32le_sha256", :k_gamma},
    }.each do |values_name, digest_name, expected_name|
      assert_sparse_full_attention_pre_output_f32_digest(
        attention[values_name],
        attention[digest_name],
        SPARSE_FULL_ATTENTION_PRE_OUTPUT_DIGESTS[expected_name]
      )
    end
    stages = fixture["stages"]
    {
      {"projected_qkv", :projected_qkv},
      {"normalized_qkv", :normalized_qkv},
      {"roped_qkv", :roped_qkv},
      {"attention_output", :attention_output},
    }.each do |stage_name, expected_name|
      stage = stages[stage_name]
      assert_sparse_full_attention_pre_output_f32_digest(
        stage["features"],
        stage["features_f32le_sha256"],
        SPARSE_FULL_ATTENTION_PRE_OUTPUT_DIGESTS[expected_name]
      )
    end

    input = sparse_full_attention_pre_output_input(fixture)
    before = input.features_copy
    linear = sparse_full_attention_pre_output_linear(fixture)
    q_gamma = sparse_full_attention_pre_output_gamma(fixture, "q_gamma")
    k_gamma = sparse_full_attention_pre_output_gamma(fixture, "k_gamma")
    num_heads = fixture["attention"]["num_heads"].as_i.to_i32

    qkv = ML::Sparse::TensorCPU.apply_self_attention_qkv(input, linear, num_heads)
    qkv.shape.should eq({3, 3, 2, 8})
    qkv.feature_shape.should eq({4, 3, 2, 8})
    expected_projected = flatten_sparse_full_attention_pre_output_f32(
      stages["projected_qkv"]["features"]
    )
    sparse_full_attention_pre_output_f32le_sha256(expected_projected).should eq(
      stages["projected_qkv"]["features_f32le_sha256"].as_s
    )
    qkv.features_copy.zip(expected_projected).each do |actual, expected|
      actual.should be_close(expected, SPARSE_FULL_ATTENTION_PRE_OUTPUT_STAGE_TOLERANCE)
    end

    normalized = ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
      qkv,
      q_gamma,
      k_gamma
    )
    expected_normalized = flatten_sparse_full_attention_pre_output_f32(
      stages["normalized_qkv"]["features"]
    )
    normalized.features_copy.zip(expected_normalized).each do |actual, expected|
      actual.should be_close(expected, SPARSE_FULL_ATTENTION_PRE_OUTPUT_STAGE_TOLERANCE)
    end

    roped = ML::Sparse::TensorCPU.apply_self_attention_rope(normalized)
    expected_roped = flatten_sparse_full_attention_pre_output_f32(
      stages["roped_qkv"]["features"]
    )
    roped.features_copy.zip(expected_roped).each do |actual, expected|
      actual.should be_close(expected, SPARSE_FULL_ATTENTION_PRE_OUTPUT_STAGE_TOLERANCE)
    end

    output = ML::Sparse::TensorCPU.apply_full_self_attention_pre_output(
      input,
      linear,
      num_heads,
      q_gamma: q_gamma,
      k_gamma: k_gamma,
      use_rope: true
    )
    expected_output = flatten_sparse_full_attention_pre_output_f32(
      stages["attention_output"]["features"]
    )
    sparse_full_attention_pre_output_f32le_sha256(expected_output).should eq(
      stages["attention_output"]["features_f32le_sha256"].as_s
    )
    output.coordinate_map.same?(input.coordinate_map).should be_true
    output.shape.should eq({3, 16})
    output.features_copy.zip(expected_output).each do |actual, expected|
      actual.should be_close(expected, SPARSE_FULL_ATTENTION_PRE_OUTPUT_TOLERANCE)
    end
    input.features_copy.should eq(before)
  end

  it "keeps normalization and RoPE independently optional" do
    fixture = sparse_full_attention_pre_output_fixture
    input = sparse_full_attention_pre_output_input(fixture)
    linear = sparse_full_attention_pre_output_linear(fixture)
    q_gamma = sparse_full_attention_pre_output_gamma(fixture, "q_gamma")
    k_gamma = sparse_full_attention_pre_output_gamma(fixture, "k_gamma")
    num_heads = fixture["attention"]["num_heads"].as_i.to_i32

    [false, true].each do |use_norm|
      [false, true].each do |use_rope|
        q = use_norm ? q_gamma : nil
        k = use_norm ? k_gamma : nil
        actual = ML::Sparse::TensorCPU.apply_full_self_attention_pre_output(
          input,
          linear,
          num_heads,
          q_gamma: q,
          k_gamma: k,
          use_rope: use_rope
        )
        expected = sparse_full_attention_pre_output_manual(
          input,
          linear,
          num_heads,
          q,
          k,
          use_rope
        )
        actual.features_copy.should eq(expected.features_copy)
      end
    end
  end

  it "rejects a half-configured Q/K normalizer before projection" do
    fixture = sparse_full_attention_pre_output_fixture
    input = sparse_full_attention_pre_output_input(fixture)
    before = input.features_copy
    linear = sparse_full_attention_pre_output_linear(fixture)
    q_gamma = sparse_full_attention_pre_output_gamma(fixture, "q_gamma")
    k_gamma = sparse_full_attention_pre_output_gamma(fixture, "k_gamma")

    expect_raises(ML::Sparse::SparseTensorError, /must be provided together/) do
      ML::Sparse::TensorCPU.apply_full_self_attention_pre_output(
        input,
        linear,
        2,
        q_gamma: q_gamma
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /must be provided together/) do
      ML::Sparse::TensorCPU.apply_full_self_attention_pre_output(
        input,
        linear,
        2,
        k_gamma: k_gamma
      )
    end
    input.features_copy.should eq(before)
  end

  it "propagates the bounded full-attention score admission" do
    fixture = sparse_full_attention_pre_output_fixture
    input = sparse_full_attention_pre_output_input(fixture)
    linear = sparse_full_attention_pre_output_linear(fixture)
    # A projection attempt would reject this parameter first. The score-budget
    # error therefore proves composition admission happens before projection.
    linear.weight.data.cpu_data.not_nil![0] = Float32::NAN

    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require 64 bytes.*limit is 63/
    ) do
      ML::Sparse::TensorCPU.apply_full_self_attention_pre_output(
        input,
        linear,
        2,
        max_score_bytes: 63_i64
      )
    end
  end

  it "supports an empty sparse input without optional transforms" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 2, {1, 1, 1})
    input = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 4_i32)),
      map
    )
    linear = ML::NN::Linear.new(
      4,
      12,
      bias: false,
      device: ML::Tensor::Device::CPU
    )
    linear.weight.requires_grad = false

    output = ML::Sparse::TensorCPU.apply_full_self_attention_pre_output(
      input,
      linear,
      2
    )
    output.shape.should eq({2, 4})
    output.features_copy.should be_empty
    output.coordinate_map.same?(map).should be_true
  end

  it "requires the base TensorCPU receiver" do
    fixture = sparse_full_attention_pre_output_fixture
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseFullAttentionPreOutputTensorCPUOverride.apply_full_self_attention_pre_output(
        sparse_full_attention_pre_output_input(fixture),
        sparse_full_attention_pre_output_linear(fixture),
        2
      )
    end
  end
end
