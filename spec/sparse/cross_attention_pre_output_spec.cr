require "json"
require "../../src/ml/sparse/cross_attention_qk_rms_norm"
require "../spec_helper"

private CROSS_PRE_OUTPUT_TOLERANCE = 2.0e-5_f32

private def cross_pre_output_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json"
  )))
end

private def cross_pre_output_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if hash = payload.as_h?
    if values = hash["values"]?
      return cross_pre_output_f32(values, output)
    end
  elsif array = payload.as_a?
    array.each { |entry| cross_pre_output_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def cross_pre_output_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def cross_pre_output_linear(
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
  cross_pre_output_f32(weight).each_with_index do |value, index|
    layer.weight.data.cpu_data.not_nil![index] = value
  end
  cross_pre_output_f32(fixture["parameters"][bias_name]).each_with_index do |value, index|
    layer.bias.not_nil!.data.cpu_data.not_nil![index] = value
  end
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def cross_pre_output_normalized(
  fixture : JSON::Any,
) : ML::Sparse::CrossAttentionQKNormalizedCPU
  input = fixture["input"]
  point_count = input["coordinates"].as_a.size.to_i32
  channels = input["channels"].as_i.to_i32
  coordinate_map = ML::Sparse::CoordinateMap3D.new(
    cross_pre_output_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  query = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      cross_pre_output_f32(fixture["stages"]["norm2_output"]),
      ML::Shape.new(point_count, channels)
    ),
    coordinate_map
  )
  context = ML::Tensor.from_array(
    cross_pre_output_f32(input["context"]),
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
    cross_pre_output_linear(fixture, "to_q_weight", "to_q_bias"),
    cross_pre_output_linear(fixture, "to_kv_weight", "to_kv_bias")
  )
  q_gamma = ML::Tensor.from_array(
    cross_pre_output_f32(fixture["parameters"]["q_gamma"]),
    ML::Shape.new(projection.num_heads, projection.head_dim)
  )
  k_gamma = ML::Tensor.from_array(
    cross_pre_output_f32(fixture["parameters"]["k_gamma"]),
    ML::Shape.new(projection.num_heads, projection.head_dim)
  )
  ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
    projection,
    q_gamma,
    k_gamma
  )
end

private def cross_pre_output_extreme_normalized(
  production : Bool = false,
  empty_query : Bool = false,
) : ML::Sparse::CrossAttentionQKNormalizedCPU
  coordinate_map = ML::Sparse::CoordinateMap3D.new(
    empty_query ? [] of Int32 : [0, 0, 0, 0] of Int32,
    2,
    {1, 1, 1}
  )
  query_tensor = ML::Tensor.from_array(
    empty_query ? [] of Float32 : [1.0_f32],
    ML::Shape.new(empty_query ? 0_i32 : 1_i32, 1_i32)
  )
  query = if production
            ML::Sparse::TensorCPU.production(
              query_tensor,
              coordinate_map,
              128_i64
            )
          else
            ML::Sparse::TensorCPU.new(query_tensor, coordinate_map)
          end
  context = ML::Tensor.from_array(
    [1.0_f32, -1.0_f32, 1.0_f32, -1.0_f32],
    ML::Shape.new(2_i32, 2_i32, 1_i32)
  )
  to_q = ML::NN::Linear.new(
    1,
    1,
    device: ML::Tensor::Device::CPU
  )
  to_q.weight.data.cpu_data.not_nil![0] = 1.0_f32
  to_q.bias.not_nil!.data.cpu_data.not_nil![0] = 0.0_f32
  to_kv = ML::NN::Linear.new(
    1,
    2,
    device: ML::Tensor::Device::CPU
  )
  to_kv.weight.data.cpu_data.not_nil![0] = 1.0_f32
  to_kv.weight.data.cpu_data.not_nil![1] = -2.0_f32
  to_kv.bias.not_nil!.data.cpu_data.not_nil![0] = 0.0_f32
  to_kv.bias.not_nil!.data.cpu_data.not_nil![1] = 5.0_f32
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
  gamma = ML::Tensor.from_array(
    [100.0_f32],
    ML::Shape.new(1_i32, 1_i32)
  )
  ML::Sparse::CrossAttentionQKNormalizedCPU.normalize(
    projection,
    gamma,
    gamma
  )
end

describe "TRELLIS.2 sparse cross-attention pre-output" do
  it "matches the source-pinned ragged-query/dense-context backend boundary" do
    fixture = cross_pre_output_fixture
    normalized = cross_pre_output_normalized(fixture)
    before_query = normalized.query_features_copy
    before_key = normalized.key_features_copy
    before_value = normalized.value_features_copy
    output = ML::Sparse::TensorCPU.apply_cross_attention_pre_output(normalized)
    expected = cross_pre_output_f32(fixture["cross_attention_pre_output"])

    output.coordinate_map.same?(normalized.coordinate_map).should be_true
    output.production_width?.should be_false
    output.features_copy.size.should eq(expected.size)
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, CROSS_PRE_OUTPUT_TOLERANCE)
    end
    normalized.query_batch_slice(1).should eq(ML::Sparse::BatchSlice.new(2, 2))
    output.feature(2, 0).finite?.should be_true
    normalized.query_features_copy.should eq(before_query)
    normalized.key_features_copy.should eq(before_key)
    normalized.value_features_copy.should eq(before_value)
  end

  it "uses stable F32 softmax for extreme logits and skips an empty batch" do
    normalized = cross_pre_output_extreme_normalized
    output = ML::Sparse::TensorCPU.apply_cross_attention_pre_output(normalized)

    output.feature(0, 0).should be_close(3.0_f32, 1.0e-4_f32)
    normalized.query_batch_slice(1).should eq(ML::Sparse::BatchSlice.new(1, 1))
    output.coordinate_map.same?(normalized.coordinate_map).should be_true
  end

  it "preserves production provenance and rejects one-byte-low score/work budgets" do
    normalized = cross_pre_output_extreme_normalized(production: true)
    output = ML::Sparse::TensorCPU.apply_cross_attention_pre_output(normalized)

    output.production_width?.should be_true
    output.max_feature_bytes.should eq(128_i64)
    output.coordinate_map.same?(normalized.coordinate_map).should be_true

    expect_raises(ML::Sparse::SparseTensorBudgetError) do
      ML::Sparse::TensorCPU.apply_cross_attention_pre_output(
        normalized,
        max_score_bytes: normalized.plan.score_row_bytes - 1_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorBudgetError) do
      ML::Sparse::TensorCPU.apply_cross_attention_pre_output(
        normalized,
        max_work_elements: normalized.plan.attention_mac_elements - 1_i64
      )
    end
  end

  it "returns an empty carrier without score scratch for an all-empty query" do
    normalized = cross_pre_output_extreme_normalized(empty_query: true)
    output = ML::Sparse::TensorCPU.apply_cross_attention_pre_output(
      normalized,
      max_score_bytes: 1_i64
    )

    output.features_copy.should be_empty
    output.coordinate_map.same?(normalized.coordinate_map).should be_true
  end
end
