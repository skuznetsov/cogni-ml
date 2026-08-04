require "json"
require "digest/sha256"
require "../../src/ml/sparse/cross_attention"
require "../spec_helper"

private CROSS_ATTENTION_TOLERANCE    = 5.0e-5_f32
private CROSS_ATTENTION_FINAL_DIGEST = "afa01c08b92d7750e1d0453fcd29b6049600c95d5c147f584141a2a5fa15b635"

private def combined_cross_attention_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json"
  )))
end

private def combined_cross_attention_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if hash = payload.as_h?
    if values = hash["values"]?
      return combined_cross_attention_f32(values, output)
    end
  elsif array = payload.as_a?
    array.each { |entry| combined_cross_attention_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def combined_cross_attention_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def combined_cross_attention_digest(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def combined_cross_attention_linear(
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
  combined_cross_attention_f32(weight).each_with_index do |value, index|
    layer.weight.data.cpu_data.not_nil![index] = value
  end
  combined_cross_attention_f32(fixture["parameters"][bias_name])
    .each_with_index do |value, index|
      layer.bias.not_nil!.data.cpu_data.not_nil![index] = value
    end
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  layer
end

private def combined_cross_attention_inputs(fixture : JSON::Any)
  input = fixture["input"]
  point_count = input["coordinates"].as_a.size.to_i32
  channels = input["channels"].as_i.to_i32
  coordinate_map = ML::Sparse::CoordinateMap3D.new(
    combined_cross_attention_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  query = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      combined_cross_attention_f32(fixture["stages"]["norm2_output"]),
      ML::Shape.new(point_count, channels)
    ),
    coordinate_map
  )
  context = ML::Tensor.from_array(
    combined_cross_attention_f32(input["context"]),
    ML::Shape.new(
      input["batch_size"].as_i.to_i32,
      input["context_length"].as_i.to_i32,
      input["context_channels"].as_i.to_i32
    )
  )
  heads = input["heads"].as_i.to_i32
  head_dim = input["head_dim"].as_i.to_i32
  {
    query:   query,
    context: context,
    heads:   heads,
    to_q:    combined_cross_attention_linear(
      fixture,
      "to_q_weight",
      "to_q_bias"
    ),
    to_kv: combined_cross_attention_linear(
      fixture,
      "to_kv_weight",
      "to_kv_bias"
    ),
    q_gamma: ML::Tensor.from_array(
      combined_cross_attention_f32(fixture["parameters"]["q_gamma"]),
      ML::Shape.new(heads, head_dim)
    ),
    k_gamma: ML::Tensor.from_array(
      combined_cross_attention_f32(fixture["parameters"]["k_gamma"]),
      ML::Shape.new(heads, head_dim)
    ),
    to_out: combined_cross_attention_linear(
      fixture,
      "to_out_weight",
      "to_out_bias"
    ),
  }
end

private def freeze_combined_cross_attention_linear(
  layer : ML::NN::Linear,
) : ML::NN::Linear
  layer.weight.requires_grad = false
  layer.bias.try { |bias| bias.requires_grad = false }
  layer
end

class CombinedCrossAttentionTensorCPUOverride < ML::Sparse::TensorCPU
end

describe "TRELLIS.2 combined sparse cross-attention" do
  it "composes the source-pinned normalized cross-attention stages exactly" do
    fixture = combined_cross_attention_fixture
    inputs = combined_cross_attention_inputs(fixture)
    before_query = inputs[:query].features_copy
    before_context = inputs[:context].cpu_data.not_nil!.dup
    before_to_q_weight = inputs[:to_q].weight.data.cpu_data.not_nil!.dup
    before_to_q_bias = inputs[:to_q].bias.not_nil!.data.cpu_data.not_nil!.dup
    before_to_kv_weight = inputs[:to_kv].weight.data.cpu_data.not_nil!.dup
    before_to_kv_bias = inputs[:to_kv].bias.not_nil!.data.cpu_data.not_nil!.dup
    before_q_gamma = inputs[:q_gamma].cpu_data.not_nil!.dup
    before_k_gamma = inputs[:k_gamma].cpu_data.not_nil!.dup
    before_to_out_weight = inputs[:to_out].weight.data.cpu_data.not_nil!.dup
    before_to_out_bias = inputs[:to_out].bias.not_nil!.data.cpu_data.not_nil!.dup

    output = ML::Sparse::TensorCPU.apply_cross_attention(
      inputs[:query],
      inputs[:context],
      inputs[:heads],
      inputs[:to_q],
      inputs[:to_kv],
      inputs[:q_gamma],
      inputs[:k_gamma],
      inputs[:to_out]
    )
    expected = combined_cross_attention_f32(
      fixture["stages"]["cross_attention_output"]
    )

    output.coordinate_map.same?(inputs[:query].coordinate_map).should be_true
    output.point_count.should eq(inputs[:query].point_count)
    output.channels.should eq(inputs[:query].channels)
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, CROSS_ATTENTION_TOLERANCE)
    end
    combined_cross_attention_digest(expected).should eq(
      CROSS_ATTENTION_FINAL_DIGEST
    )

    inputs[:query].features_copy.should eq(before_query)
    inputs[:context].cpu_data.not_nil!.should eq(before_context)
    inputs[:to_q].weight.data.cpu_data.not_nil!.should eq(before_to_q_weight)
    inputs[:to_q].bias.not_nil!.data.cpu_data.not_nil!.should eq(before_to_q_bias)
    inputs[:to_kv].weight.data.cpu_data.not_nil!.should eq(before_to_kv_weight)
    inputs[:to_kv].bias.not_nil!.data.cpu_data.not_nil!.should eq(before_to_kv_bias)
    inputs[:q_gamma].cpu_data.not_nil!.should eq(before_q_gamma)
    inputs[:k_gamma].cpu_data.not_nil!.should eq(before_k_gamma)
    inputs[:to_out].weight.data.cpu_data.not_nil!.should eq(before_to_out_weight)
    inputs[:to_out].bias.not_nil!.data.cpu_data.not_nil!.should eq(before_to_out_bias)
  end

  it "rejects every one-under plan cap before reading poisoned parameters" do
    fixture = combined_cross_attention_fixture
    inputs = combined_cross_attention_inputs(fixture)
    plan = ML::Sparse::CrossAttentionPlanCPU.preflight(
      inputs[:query],
      inputs[:context],
      inputs[:heads]
    )
    inputs[:to_q].weight.data.cpu_data.not_nil![0] = Float32::NAN

    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /score budget would require #{plan.score_bytes} bytes/
    ) do
      ML::Sparse::TensorCPU.apply_cross_attention(
        inputs[:query], inputs[:context], inputs[:heads],
        inputs[:to_q], inputs[:to_kv], inputs[:q_gamma], inputs[:k_gamma],
        inputs[:to_out],
        max_score_bytes: plan.score_bytes - 1_i64
      )
    end
    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /projections would require #{plan.projection_bytes} bytes/
    ) do
      ML::Sparse::TensorCPU.apply_cross_attention(
        inputs[:query], inputs[:context], inputs[:heads],
        inputs[:to_q], inputs[:to_kv], inputs[:q_gamma], inputs[:k_gamma],
        inputs[:to_out],
        max_projection_bytes: plan.projection_bytes - 1_i64
      )
    end
    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /work would require #{plan.work_elements} MAC elements/
    ) do
      ML::Sparse::TensorCPU.apply_cross_attention(
        inputs[:query], inputs[:context], inputs[:heads],
        inputs[:to_q], inputs[:to_kv], inputs[:q_gamma], inputs[:k_gamma],
        inputs[:to_out],
        max_total_work_elements: plan.work_elements - 1_i64
      )
    end
  end

  it "rejects an inherited receiver before entering any leaf" do
    fixture = combined_cross_attention_fixture
    inputs = combined_cross_attention_inputs(fixture)
    inputs[:to_q].weight.data.cpu_data.not_nil![0] = Float32::NAN

    expect_raises(
      ML::Sparse::SparseTensorError,
      /base TensorCPU receiver/
    ) do
      CombinedCrossAttentionTensorCPUOverride.apply_cross_attention(
        inputs[:query], inputs[:context], inputs[:heads],
        inputs[:to_q], inputs[:to_kv], inputs[:q_gamma], inputs[:k_gamma],
        inputs[:to_out]
      )
    end
  end

  it "preserves an empty production carrier and still validates parameters" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {1, 1, 1})
    query = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 1_i32)),
      map,
      16_i64
    )
    context = ML::Tensor.from_array(
      [0.5_f32],
      ML::Shape.new(1_i32, 1_i32, 1_i32)
    )
    to_q = freeze_combined_cross_attention_linear(
      ML::NN::Linear.new(1, 1, device: ML::Tensor::Device::CPU)
    )
    to_kv = freeze_combined_cross_attention_linear(
      ML::NN::Linear.new(1, 2, device: ML::Tensor::Device::CPU)
    )
    to_out = freeze_combined_cross_attention_linear(
      ML::NN::Linear.new(1, 1, device: ML::Tensor::Device::CPU)
    )
    gamma = ML::Tensor.from_array([1.0_f32], ML::Shape.new(1_i32, 1_i32))

    output = ML::Sparse::TensorCPU.apply_cross_attention(
      query, context, 1, to_q, to_kv, gamma, gamma, to_out
    )
    output.features_copy.should be_empty
    output.coordinate_map.same?(map).should be_true
    output.production_width?.should be_true
    output.max_feature_bytes.should eq(16_i64)

    to_out.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::INFINITY
    expect_raises(ML::Sparse::SparseTensorError, /bias\[0\] must be finite/) do
      ML::Sparse::TensorCPU.apply_cross_attention(
        query, context, 1, to_q, to_kv, gamma, gamma, to_out
      )
    end
  end
end
