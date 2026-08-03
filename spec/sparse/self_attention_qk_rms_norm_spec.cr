require "json"
require "digest/sha256"
require "../../src/ml/sparse/self_attention_qk_rms_norm"
require "../../src/ml/sparse/full_self_attention"
require "../spec_helper"

private SPARSE_QK_RMS_NORM_TOLERANCE = 2.0e-5_f32

private def sparse_qk_rms_norm_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_self_attention_qk_rms_norm_cpu_v1.json"
  )))
end

private def flatten_sparse_qk_rms_norm_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if array = payload.as_a?
    array.each { |entry| flatten_sparse_qk_rms_norm_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_qk_rms_norm_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_qk_rms_norm_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def sparse_qk_rms_norm_inputs(
  fixture : JSON::Any,
) : Tuple(
  ML::Sparse::SelfAttentionQKVCPU,
  ML::Tensor,
  ML::Tensor,
)
  input = fixture["input"]
  point_count = input["coordinates"].as_a.size.to_i32
  num_heads = input["num_heads"].as_i.to_i32
  head_dim = input["head_dim"].as_i.to_i32
  map = ML::Sparse::CoordinateMap3D.new(
    sparse_qk_rms_norm_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  flat = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      flatten_sparse_qk_rms_norm_f32(input["qkv_features"]),
      ML::Shape.new(point_count, 3_i32 * num_heads * head_dim)
    ),
    map
  )
  q_gamma = ML::Tensor.from_array(
    flatten_sparse_qk_rms_norm_f32(input["q_gamma"]),
    ML::Shape.new(num_heads, head_dim)
  )
  k_gamma = ML::Tensor.from_array(
    flatten_sparse_qk_rms_norm_f32(input["k_gamma"]),
    ML::Shape.new(num_heads, head_dim)
  )
  {ML::Sparse::SelfAttentionQKVCPU.new(flat, num_heads), q_gamma, k_gamma}
end

class SparseQKRMSNormTensorCPUOverride < ML::Sparse::TensorCPU
end

describe ML::Sparse::TensorCPU do
  it "matches the pinned D=3 upstream Q/K fixture exactly" do
    fixture = sparse_qk_rms_norm_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-self-attention-qk-rms-norm-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    fixture["provenance"]["upstream_sparse_qk_normalizer_executed"]
      .as_bool.should be_true
    fixture["contract"]["not_attention_backend_parity"].as_bool.should be_true
    fixture["output"]["qkv_features_f32le_sha256"].as_s.should eq(
      "69ea07a3d7796b315b613139bb85f5114ea302eca35d6b84c65dfb69ae7b9e6d"
    )
    qkv, q_gamma, k_gamma = sparse_qk_rms_norm_inputs(fixture)
    before = qkv.features_copy

    output = ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
      qkv,
      q_gamma,
      k_gamma
    )
    expected = flatten_sparse_qk_rms_norm_f32(
      fixture["output"]["qkv_features"]
    )

    output.coordinate_map.same?(qkv.coordinate_map).should be_true
    output.shape.should eq(qkv.shape)
    output.max_feature_bytes.should eq(qkv.max_feature_bytes)
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, SPARSE_QK_RMS_NORM_TOLERANCE)
    end
    sparse_qk_rms_norm_f32le_sha256(output.features_copy).should eq(
      fixture["output"]["qkv_features_f32le_sha256"].as_s
    )
    qkv.features_copy.should eq(before)

    row_width = 3 * qkv.channels
    qkv.point_count.times do |row|
      qkv.channels.times do |channel|
        index = row * row_width + 2 * qkv.channels + channel
        output.features_copy[index].should eq(before[index])
      end
    end
    output.feature(0, 1, 1, 0).should eq(0.0_f32)
    output.feature(1, 0, 0, 0).should eq(0.0_f32)

    plan = ML::Sparse::FullSelfAttentionPlanCPU.build(output)
    plan.point_count.should eq(output.point_count)
    attention = ML::Sparse::TensorCPU.apply_full_self_attention(output)
    attention.coordinate_map.same?(qkv.coordinate_map).should be_true
    attention.features_copy.each(&.finite?.should(be_true))
  end

  it "normalizes D=1 by sign without mixing heads or V" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [2.0_f32, -3.0_f32, 0.0_f32, 4.0_f32, 7.0_f32, 8.0_f32],
        ML::Shape.new(1_i32, 6_i32)
      ),
      map
    )
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(flat, 2)
    q_gamma = ML::Tensor.from_array(
      [0.5_f32, -2.0_f32],
      ML::Shape.new(2_i32, 1_i32)
    )
    k_gamma = ML::Tensor.from_array(
      [3.0_f32, 0.25_f32],
      ML::Shape.new(2_i32, 1_i32)
    )

    output = ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
      qkv,
      q_gamma,
      k_gamma
    )

    output.features_copy.should eq(
      [0.5_f32, 2.0_f32, 0.0_f32, 0.25_f32, 7.0_f32, 8.0_f32]
    )
  end

  it "preserves an all-empty packed value for the local CPU policy" do
    map = ML::Sparse::CoordinateMap3D.new(
      [] of Int32,
      3,
      {1, 1, 1}
    )
    flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [] of Float32,
        ML::Shape.new(0_i32, 6_i32)
      ),
      map
    )
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(flat, 1)
    gamma = ML::Tensor.from_array(
      [1.0_f32, 1.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    )

    output = ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
      qkv,
      gamma,
      gamma
    )

    output.coordinate_map.same?(map).should be_true
    output.shape.should eq({3_i32, 3_i32, 1_i32, 2_i32})
    output.features_copy.should be_empty
    ML::Sparse::TensorCPU.apply_full_self_attention(output)
      .features_copy.should be_empty
  end

  it "rejects malformed, noncontiguous, and non-finite gamma values" do
    qkv, q_gamma, k_gamma = sparse_qk_rms_norm_inputs(
      sparse_qk_rms_norm_fixture
    )
    wrong_shape = ML::Tensor.from_array(
      [1.0_f32, 1.0_f32, 1.0_f32],
      ML::Shape.new(1_i32, 3_i32)
    )
    expect_raises(ML::Sparse::SparseTensorError, /q_gamma shape.*\[2, 3\]/) do
      ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
        qkv,
        wrong_shape,
        k_gamma
      )
    end

    noncontiguous = ML::Tensor.from_array(
      [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 6.0_f32],
      ML::Shape.new(3_i32, 2_i32)
    ).transpose
    expect_raises(ML::Sparse::SparseTensorError, /q_gamma.*contiguous/) do
      ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
        qkv,
        noncontiguous,
        k_gamma
      )
    end

    nonfinite = ML::Tensor.from_array(
      [1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32, Float32::NAN, 1.0_f32],
      ML::Shape.new(2_i32, 3_i32)
    )
    expect_raises(ML::Sparse::SparseTensorError, /k_gamma\[4\].*finite/) do
      ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
        qkv,
        q_gamma,
        nonfinite
      )
    end
  end

  it "rejects non-finite normalized output before returning a value" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    flat = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [1.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32,
         4.0_f32, 5.0_f32, 6.0_f32],
        ML::Shape.new(1_i32, 9_i32)
      ),
      map
    )
    qkv = ML::Sparse::SelfAttentionQKVCPU.new(flat, 1)
    huge_gamma = ML::Tensor.from_array(
      [Float32::MAX, 1.0_f32, 1.0_f32],
      ML::Shape.new(1_i32, 3_i32)
    )
    unit_gamma = ML::Tensor.from_array(
      [1.0_f32, 1.0_f32, 1.0_f32],
      ML::Shape.new(1_i32, 3_i32)
    )

    expect_raises(ML::Sparse::SparseTensorError, /normalized Q output.*finite/) do
      ML::Sparse::TensorCPU.apply_self_attention_qk_rms_norm(
        qkv,
        huge_gamma,
        unit_gamma
      )
    end
  end

  it "requires the base TensorCPU receiver" do
    qkv, q_gamma, k_gamma = sparse_qk_rms_norm_inputs(
      sparse_qk_rms_norm_fixture
    )
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      SparseQKRMSNormTensorCPUOverride.apply_self_attention_qk_rms_norm(
        qkv,
        q_gamma,
        k_gamma
      )
    end
  end
end
