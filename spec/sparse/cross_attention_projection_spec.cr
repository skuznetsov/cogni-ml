require "json"
require "../../src/ml/sparse/cross_attention_projection"
require "../spec_helper"

private CROSS_ATTENTION_PROJECTION_TOLERANCE = 5.0e-5_f32

private def sparse_cross_attention_projection_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json"
  )))
end

private def sparse_cross_attention_projection_i32(
  payload : JSON::Any,
) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_cross_attention_projection_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if hash = payload.as_h?
    if values = hash["values"]?
      return sparse_cross_attention_projection_f32(values, output)
    end
  elsif array = payload.as_a?
    array.each { |entry| sparse_cross_attention_projection_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def sparse_cross_attention_projection_query(
  fixture : JSON::Any,
) : ML::Sparse::TensorCPU
  input = fixture["input"]
  values = sparse_cross_attention_projection_f32(
    fixture["stages"]["norm2_output"]
  )
  point_count = input["coordinates"].as_a.size.to_i32
  channels = input["channels"].as_i.to_i32
  map = ML::Sparse::CoordinateMap3D.new(
    sparse_cross_attention_projection_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(values, ML::Shape.new(point_count, channels)),
    map
  )
end

private def sparse_cross_attention_projection_context(
  fixture : JSON::Any,
) : ML::Tensor
  input = fixture["input"]
  ML::Tensor.from_array(
    sparse_cross_attention_projection_f32(input["context"]),
    ML::Shape.new(
      input["batch_size"].as_i.to_i32,
      input["context_length"].as_i.to_i32,
      input["context_channels"].as_i.to_i32
    )
  )
end

private def sparse_cross_attention_projection_linear(
  fixture : JSON::Any,
  weight_name : String,
  bias_name : String,
) : ML::NN::Linear
  weight_payload = fixture["parameters"][weight_name]
  weight_shape = weight_payload["shape"].as_a
  output_channels = weight_shape[0].as_i.to_i32
  input_channels = weight_shape[1].as_i.to_i32
  layer = ML::NN::Linear.new(
    input_channels,
    output_channels,
    device: ML::Tensor::Device::CPU
  )

  sparse_cross_attention_projection_f32(weight_payload).each_with_index do |value, index|
    layer.weight.data.cpu_data.not_nil![index] = value
  end
  sparse_cross_attention_projection_f32(
    fixture["parameters"][bias_name]
  ).each_with_index do |value, index|
    layer.bias.not_nil!.data.cpu_data.not_nil![index] = value
  end
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def sparse_cross_attention_projection_run(
  fixture : JSON::Any,
) : Tuple(
  ML::Sparse::CrossAttentionProjectionCPU,
  ML::Sparse::TensorCPU,
  ML::Tensor,
  ML::NN::Linear,
  ML::NN::Linear,
)
  input = fixture["input"]
  query = sparse_cross_attention_projection_query(fixture)
  context = sparse_cross_attention_projection_context(fixture)
  to_q = sparse_cross_attention_projection_linear(
    fixture,
    "to_q_weight",
    "to_q_bias"
  )
  to_kv = sparse_cross_attention_projection_linear(
    fixture,
    "to_kv_weight",
    "to_kv_bias"
  )
  projection = ML::Sparse::CrossAttentionProjectionCPU.project(
    query,
    context,
    input["heads"].as_i.to_i32,
    to_q,
    to_kv
  )
  {projection, query, context, to_q, to_kv}
end

describe ML::Sparse::CrossAttentionProjectionCPU do
  it "matches the source-bound flat Q and dense K/V projections" do
    fixture = sparse_cross_attention_projection_fixture
    projection, query, context, to_q, to_kv =
      sparse_cross_attention_projection_run(fixture)
    expected = fixture["cross_attention_projections"]

    projection.coordinate_map.same?(query.coordinate_map).should be_true
    projection.plan.coordinate_map.same?(query.coordinate_map).should be_true
    projection.query_feature_shape.should eq({4, 16})
    projection.context_kv_feature_shape.should eq({3, 3, 32})
    projection.query_batch_slice(0).should eq(ML::Sparse::BatchSlice.new(0, 2))
    projection.query_batch_slice(1).should eq(ML::Sparse::BatchSlice.new(2, 2))
    projection.query_batch_slice(2).should eq(ML::Sparse::BatchSlice.new(2, 4))

    wanted_query = sparse_cross_attention_projection_f32(expected["query"])
    projection.query_features_copy.size.should eq(wanted_query.size)
    projection.query_features_copy.zip(wanted_query).each do |actual, wanted|
      actual.should be_close(wanted, CROSS_ATTENTION_PROJECTION_TOLERANCE)
    end
    wanted_kv = sparse_cross_attention_projection_f32(expected["context_kv"])
    projection.context_kv_features_copy.size.should eq(wanted_kv.size)
    projection.context_kv_features_copy.zip(wanted_kv).each do |actual, wanted|
      actual.should be_close(wanted, CROSS_ATTENTION_PROJECTION_TOLERANCE)
    end

    projection.query_feature(3, 1, 7).should be_close(
      wanted_query[3 * 16 + 1 * 8 + 7],
      CROSS_ATTENTION_PROJECTION_TOLERANCE
    )
    projection.context_kv_feature(2, 2, 1, 1, 7).should be_close(
      wanted_kv[(((2 * 3 + 2) * 2 + 1) * 2 + 1) * 8 + 7],
      CROSS_ATTENTION_PROJECTION_TOLERANCE
    )

    owned_query = projection.query_features_copy
    owned_query[0] = 99.0_f32
    projection.query_feature(0, 0, 0).should_not eq(99.0_f32)
    owned_kv = projection.context_kv_features_copy
    owned_kv[0] = 99.0_f32
    projection.context_kv_feature(0, 0, 0, 0, 0).should_not eq(99.0_f32)

    query.features_copy.should eq(
      sparse_cross_attention_projection_f32(fixture["stages"]["norm2_output"])
    )
    context.cpu_data.not_nil!.should eq(
      sparse_cross_attention_projection_f32(fixture["input"]["context"])
    )
    to_q.weight.data.cpu_data.not_nil!.should eq(
      sparse_cross_attention_projection_f32(fixture["parameters"]["to_q_weight"])
    )
    to_q.bias.not_nil!.data.cpu_data.not_nil!.should eq(
      sparse_cross_attention_projection_f32(fixture["parameters"]["to_q_bias"])
    )
    to_kv.weight.data.cpu_data.not_nil!.should eq(
      sparse_cross_attention_projection_f32(fixture["parameters"]["to_kv_weight"])
    )
    to_kv.bias.not_nil!.data.cpu_data.not_nil!.should eq(
      sparse_cross_attention_projection_f32(fixture["parameters"]["to_kv_bias"])
    )
    to_q.weight.requires_grad?.should be_false
    to_kv.weight.requires_grad?.should be_false
  end

  it "runs preflight before reading mutable payloads or parameters" do
    fixture = sparse_cross_attention_projection_fixture
    query = sparse_cross_attention_projection_query(fixture)
    context = sparse_cross_attention_projection_context(fixture)
    context.cpu_data.not_nil![0] = Float32::NAN
    to_q = ML::NN::Linear.new(16, 16, device: ML::Tensor::Device::CPU)
    to_kv = ML::NN::Linear.new(5, 32, device: ML::Tensor::Device::CPU)

    expect_raises(ML::Sparse::SparseTensorBudgetError, /require 1408 bytes.*limit is 1407/) do
      ML::Sparse::CrossAttentionProjectionCPU.project(
        query,
        context,
        2,
        to_q,
        to_kv,
        max_projection_bytes: 1_407_i64
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /context\[0\] must be finite/) do
      ML::Sparse::CrossAttentionProjectionCPU.project(
        query,
        context,
        2,
        to_q,
        to_kv
      )
    end
  end

  it "rejects non-source projection parameters before output allocation" do
    fixture = sparse_cross_attention_projection_fixture
    query = sparse_cross_attention_projection_query(fixture)
    context = sparse_cross_attention_projection_context(fixture)
    to_q = sparse_cross_attention_projection_linear(
      fixture,
      "to_q_weight",
      "to_q_bias"
    )
    to_kv = sparse_cross_attention_projection_linear(
      fixture,
      "to_kv_weight",
      "to_kv_bias"
    )

    to_q.weight.requires_grad = true
    expect_raises(ML::Sparse::SparseTensorError, /frozen graphless to_q parameters/) do
      ML::Sparse::CrossAttentionProjectionCPU.project(query, context, 2, to_q, to_kv)
    end
    to_q.weight.requires_grad = false

    biasless = ML::NN::Linear.new(
      5,
      32,
      bias: false,
      device: ML::Tensor::Device::CPU
    )
    biasless.weight.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /requires a biased to_kv/) do
      ML::Sparse::CrossAttentionProjectionCPU.project(query, context, 2, to_q, biasless)
    end

    wrong_shape = ML::NN::Linear.new(5, 16, device: ML::Tensor::Device::CPU)
    wrong_shape.weight.requires_grad = false
    wrong_shape.bias.not_nil!.requires_grad = false
    expect_raises(ML::Sparse::SparseTensorError, /requires to_kv Linear\(5, 32\)/) do
      ML::Sparse::CrossAttentionProjectionCPU.project(query, context, 2, to_q, wrong_shape)
    end

    to_kv.weight.data.cpu_data.not_nil![0] = Float32::INFINITY
    expect_raises(ML::Sparse::SparseTensorError, /to_kv weight\[0\] must be finite/) do
      ML::Sparse::CrossAttentionProjectionCPU.project(query, context, 2, to_q, to_kv)
    end
  end

  it "rejects non-finite arithmetic before publishing a result" do
    map = ML::Sparse::CoordinateMap3D.new([0, 0, 0, 0], 1, {1, 1, 1})
    query = ML::Sparse::TensorCPU.new(
      ML::Tensor.ones(1, 1, device: ML::Tensor::Device::CPU),
      map
    )
    context = ML::Tensor.ones(1, 1, 1, device: ML::Tensor::Device::CPU)
    to_q = ML::NN::Linear.new(1, 1, device: ML::Tensor::Device::CPU)
    to_kv = ML::NN::Linear.new(1, 2, device: ML::Tensor::Device::CPU)
    to_q.weight.data.cpu_data.not_nil![0] = Float32::MAX
    to_q.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::MAX
    to_q.weight.requires_grad = false
    to_q.bias.not_nil!.requires_grad = false
    to_kv.weight.requires_grad = false
    to_kv.bias.not_nil!.requires_grad = false

    expect_raises(ML::Sparse::SparseTensorError, /Q output\[0\] must be finite/) do
      ML::Sparse::CrossAttentionProjectionCPU.project(
        query,
        context,
        1,
        to_q,
        to_kv
      )
    end
  end

  it "keeps the ordinary constructor bounded despite production authority" do
    map = ML::Sparse::CoordinateMap3D.new([0, 0, 0, 0], 1, {1, 1, 1})
    expect_raises(ML::Sparse::SparseTensorError, /channel count must be in 1\.\.256/) do
      ML::Sparse::TensorCPU.new(
        ML::Tensor.zeros(1, 1_536, device: ML::Tensor::Device::CPU),
        map
      )
    end
  end

  it "checks logical projection indices" do
    projection, _, _, _, _ = sparse_cross_attention_projection_run(
      sparse_cross_attention_projection_fixture
    )
    expect_raises(IndexError, /query row 4 is out of bounds/) do
      projection.query_feature(4, 0, 0)
    end
    expect_raises(IndexError, /context token 3 is out of bounds/) do
      projection.context_kv_feature(0, 3, 0, 0, 0)
    end
    expect_raises(IndexError, /component 2 is out of bounds/) do
      projection.context_kv_feature(0, 0, 2, 0, 0)
    end
  end
end
