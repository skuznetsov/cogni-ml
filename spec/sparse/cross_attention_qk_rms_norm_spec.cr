require "json"
require "../../src/ml/sparse/cross_attention_qk_rms_norm"
require "../spec_helper"

private CROSS_QK_RMS_NORM_TOLERANCE = 2.0e-5_f32

private def cross_qk_rms_norm_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json"
  )))
end

private def cross_qk_rms_norm_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if hash = payload.as_h?
    if values = hash["values"]?
      return cross_qk_rms_norm_f32(values, output)
    end
  elsif array = payload.as_a?
    array.each { |entry| cross_qk_rms_norm_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def cross_qk_rms_norm_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def cross_qk_rms_norm_linear(
  fixture : JSON::Any,
  weight_name : String,
  bias_name : String,
) : ML::NN::Linear
  weight = fixture["parameters"][weight_name]
  shape = weight["shape"].as_a
  layer = ML::NN::Linear.new(
    shape[1].as_i.to_i32,
    shape[0].as_i.to_i32,
    device: ML::Tensor::Device::CPU
  )
  cross_qk_rms_norm_f32(weight).each_with_index do |value, index|
    layer.weight.data.cpu_data.not_nil![index] = value
  end
  cross_qk_rms_norm_f32(fixture["parameters"][bias_name]).each_with_index do |value, index|
    layer.bias.not_nil!.data.cpu_data.not_nil![index] = value
  end
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def cross_qk_rms_norm_inputs(
  fixture : JSON::Any,
) : Tuple(ML::Sparse::CrossAttentionProjectionCPU, ML::Tensor, ML::Tensor)
  input = fixture["input"]
  point_count = input["coordinates"].as_a.size.to_i32
  channels = input["channels"].as_i.to_i32
  coordinate_map = ML::Sparse::CoordinateMap3D.new(
    cross_qk_rms_norm_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  query = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      cross_qk_rms_norm_f32(fixture["stages"]["norm2_output"]),
      ML::Shape.new(point_count, channels)
    ),
    coordinate_map
  )
  context = ML::Tensor.from_array(
    cross_qk_rms_norm_f32(input["context"]),
    ML::Shape.new(
      input["batch_size"].as_i.to_i32,
      input["context_length"].as_i.to_i32,
      input["context_channels"].as_i.to_i32
    )
  )
  projection = ML::Sparse::CrossAttentionProjectionCPU.project(
    query,
    context,
    input["heads"].as_i.to_i32,
    cross_qk_rms_norm_linear(fixture, "to_q_weight", "to_q_bias"),
    cross_qk_rms_norm_linear(fixture, "to_kv_weight", "to_kv_bias")
  )
  q_gamma = ML::Tensor.from_array(
    cross_qk_rms_norm_f32(fixture["parameters"]["q_gamma"]),
    ML::Shape.new(projection.num_heads, projection.head_dim)
  )
  k_gamma = ML::Tensor.from_array(
    cross_qk_rms_norm_f32(fixture["parameters"]["k_gamma"]),
    ML::Shape.new(projection.num_heads, projection.head_dim)
  )
  {projection, q_gamma, k_gamma}
end

private def cross_qk_rms_norm_d1_inputs : Tuple(
  ML::Sparse::CrossAttentionProjectionCPU,
  ML::Tensor,
  ML::Tensor,
)
  coordinate_map = ML::Sparse::CoordinateMap3D.new(
    [0, 0, 0, 0] of Int32,
    1,
    {1, 1, 1}
  )
  query = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      [2.0_f32, -3.0_f32],
      ML::Shape.new(1_i32, 2_i32)
    ),
    coordinate_map
  )
  context = ML::Tensor.zeros(1, 1, 2, device: ML::Tensor::Device::CPU)
  to_q = ML::NN::Linear.new(2, 2, device: ML::Tensor::Device::CPU)
  to_q.weight.data.cpu_data.not_nil!.fill(0.0_f32)
  to_q.weight.data.cpu_data.not_nil![0] = 1.0_f32
  to_q.weight.data.cpu_data.not_nil![3] = 1.0_f32
  to_q.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
  to_kv = ML::NN::Linear.new(2, 4, device: ML::Tensor::Device::CPU)
  to_kv.weight.data.cpu_data.not_nil!.fill(0.0_f32)
  [
    0.0_f32,
    4.0_f32,
    7.0_f32,
    8.0_f32,
  ].each_with_index do |value, index|
    to_kv.bias.not_nil!.data.cpu_data.not_nil![index] = value
  end
  {to_q, to_kv}.each do |layer|
    layer.weight.requires_grad = false
    layer.bias.not_nil!.requires_grad = false
  end
  projection = ML::Sparse::CrossAttentionProjectionCPU.project(
    query,
    context,
    2,
    to_q,
    to_kv
  )
  q_gamma = ML::Tensor.from_array(
    [0.5_f32, -2.0_f32],
    ML::Shape.new(2_i32, 1_i32)
  )
  k_gamma = ML::Tensor.from_array(
    [3.0_f32, 0.25_f32],
    ML::Shape.new(2_i32, 1_i32)
  )
  {projection, q_gamma, k_gamma}
end

describe ML::Sparse::CrossAttentionQKNormalizedCPU do
  it "matches the pinned upstream cross Q/K normalization without padding Q or copying V" do
    fixture = cross_qk_rms_norm_fixture
    contract = fixture["cross_attention_qk_rms_norm_contract"]
    contract["upstream_qk_rms_normalizers_executed"].as_bool.should be_true
    projection, q_gamma, k_gamma = cross_qk_rms_norm_inputs(fixture)
    before_query = projection.query_features_copy
    before_kv = projection.context_kv_features_copy
    before_value = [] of Float32
    projection.plan.batch_size.times do |batch|
      projection.plan.context_length.times do |token|
        projection.num_heads.times do |head|
          projection.head_dim.times do |channel|
            before_value << projection.context_kv_feature(
              batch,
              token,
              1_i32,
              head,
              channel
            )
          end
        end
      end
    end
    before_q_gamma = q_gamma.cpu_data.not_nil!.dup
    before_k_gamma = k_gamma.cpu_data.not_nil!.dup

    output = ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
      projection,
      q_gamma,
      k_gamma
    )
    expected = fixture["cross_attention_qk_rms_norm"]
    expected_query = cross_qk_rms_norm_f32(expected["query"])
    expected_key = cross_qk_rms_norm_f32(expected["key"])
    expected_value = cross_qk_rms_norm_f32(expected["value"])

    output.coordinate_map.same?(projection.coordinate_map).should be_true
    output.plan.same?(projection.plan).should be_true
    output.query_feature_shape.should eq({4, 2, 8})
    output.key_value_feature_shape.should eq({3, 3, 2, 8})
    output.query_batch_slice(1).should eq(ML::Sparse::BatchSlice.new(2, 2))
    output.query_features_copy.size.should eq(4 * 2 * 8)
    output.query_features_copy.zip(expected_query).each do |actual, wanted|
      actual.should be_close(wanted, CROSS_QK_RMS_NORM_TOLERANCE)
    end
    output.key_features_copy.zip(expected_key).each do |actual, wanted|
      actual.should be_close(wanted, CROSS_QK_RMS_NORM_TOLERANCE)
    end
    output.value_features_copy.should eq(before_value)
    output.value_features_copy.zip(expected_value).each do |actual, wanted|
      actual.should be_close(wanted, CROSS_QK_RMS_NORM_TOLERANCE)
    end
    output.value_feature(2, 2, 1, 7).should eq(
      projection.context_kv_feature(2, 2, 1, 1, 7)
    )

    projection.query_features_copy.should eq(before_query)
    projection.context_kv_features_copy.should eq(before_kv)
    q_gamma.cpu_data.not_nil!.should eq(before_q_gamma)
    k_gamma.cpu_data.not_nil!.should eq(before_k_gamma)
  end

  it "normalizes D=1 by sign without mixing heads and keeps zero K zero" do
    projection, q_gamma, k_gamma = cross_qk_rms_norm_d1_inputs

    output = ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
      projection,
      q_gamma,
      k_gamma
    )

    output.query_features_copy.should eq([0.5_f32, 2.0_f32])
    output.key_features_copy.should eq([0.0_f32, 0.25_f32])
    output.value_features_copy.should eq([7.0_f32, 8.0_f32])
    output.source_projection.same?(projection).should be_true
    output.normalized_bytes.should eq(16_i64)
  end

  it "pins the 1e-12 denominator floor for nonzero sub-epsilon Q and K" do
    map = ML::Sparse::CoordinateMap3D.new(
      [0, 0, 0, 0] of Int32,
      1,
      {1, 1, 1}
    )
    query = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        [5.0e-13_f32],
        ML::Shape.new(1_i32, 1_i32)
      ),
      map
    )
    context = ML::Tensor.from_array(
      [-2.5e-13_f32],
      ML::Shape.new(1_i32, 1_i32, 1_i32)
    )
    to_q = ML::NN::Linear.new(1, 1, device: ML::Tensor::Device::CPU)
    to_kv = ML::NN::Linear.new(1, 2, device: ML::Tensor::Device::CPU)
    to_q.weight.data.cpu_data.not_nil![0] = 1.0_f32
    to_q.bias.not_nil!.data.cpu_data.not_nil![0] = 0.0_f32
    to_kv.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    to_kv.weight.data.cpu_data.not_nil![0] = 1.0_f32
    to_kv.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    {to_q, to_kv}.each do |layer|
      layer.weight.requires_grad = false
      layer.bias.not_nil!.requires_grad = false
    end
    projection = ML::Sparse::CrossAttentionProjectionCPU.project(
      query,
      context,
      1,
      to_q,
      to_kv
    )
    gamma = ML::Tensor.ones(1, 1, device: ML::Tensor::Device::CPU)

    output = ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
      projection,
      gamma,
      gamma
    )

    output.query_feature(0, 0, 0).should be_close(0.5_f32, 1.0e-7_f32)
    output.key_feature(0, 0, 0, 0).should be_close(-0.25_f32, 1.0e-7_f32)
    output.value_feature(0, 0, 0, 0).should eq(0.0_f32)
  end

  it "rejects the local byte budget before reading gamma values" do
    projection, q_gamma, k_gamma = cross_qk_rms_norm_inputs(
      cross_qk_rms_norm_fixture
    )
    q_gamma.cpu_data.not_nil![0] = Float32::NAN
    required_bytes = (
      projection.plan.query_projection_elements +
      projection.plan.context_kv_elements // 2_i64
    ) * 4_i64

    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /require #{required_bytes} bytes.*limit is #{required_bytes - 1_i64}/
    ) do
      ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
        projection,
        q_gamma,
        k_gamma,
        max_normalized_bytes: required_bytes - 1_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /q_gamma\[0\].*finite/) do
      ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
        projection,
        q_gamma,
        k_gamma
      )
    end
  end

  it "rejects malformed, noncontiguous, and non-finite gamma values" do
    projection, q_gamma, k_gamma = cross_qk_rms_norm_inputs(
      cross_qk_rms_norm_fixture
    )
    wrong_shape = ML::Tensor.ones(1, 16, device: ML::Tensor::Device::CPU)
    expect_raises(ML::Sparse::SparseTensorError, /q_gamma shape.*\[2, 8\]/) do
      ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
        projection,
        wrong_shape,
        k_gamma
      )
    end

    noncontiguous = ML::Tensor.ones(
      8,
      2,
      device: ML::Tensor::Device::CPU
    ).transpose
    expect_raises(ML::Sparse::SparseTensorError, /q_gamma.*contiguous/) do
      ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
        projection,
        noncontiguous,
        k_gamma
      )
    end

    k_gamma.cpu_data.not_nil![3] = Float32::INFINITY
    expect_raises(ML::Sparse::SparseTensorError, /k_gamma\[3\].*finite/) do
      ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
        projection,
        q_gamma,
        k_gamma
      )
    end
  end

  it "rejects non-finite Q before K and before publishing a result" do
    projection, q_gamma, k_gamma = cross_qk_rms_norm_inputs(
      cross_qk_rms_norm_fixture
    )
    q_gamma.cpu_data.not_nil!.fill(Float32::MAX)
    k_gamma.cpu_data.not_nil!.fill(Float32::MAX)

    expect_raises(ML::Sparse::SparseTensorError, /normalized Q output.*finite/) do
      ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
        projection,
        q_gamma,
        k_gamma
      )
    end

    q_gamma.cpu_data.not_nil!.fill(1.0_f32)
    expect_raises(ML::Sparse::SparseTensorError, /normalized K output.*finite/) do
      ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
        projection,
        q_gamma,
        k_gamma
      )
    end
  end

  it "checks logical normalized Q/K indices" do
    projection, q_gamma, k_gamma = cross_qk_rms_norm_d1_inputs
    output = ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
      projection,
      q_gamma,
      k_gamma
    )

    expect_raises(IndexError, /query row 1 is out of bounds/) do
      output.query_feature(1, 0, 0)
    end
    expect_raises(IndexError, /context token 1 is out of bounds/) do
      output.key_feature(0, 1, 0, 0)
    end
    expect_raises(IndexError, /head 2 is out of bounds/) do
      output.key_feature(0, 0, 2, 0)
    end
  end
end
