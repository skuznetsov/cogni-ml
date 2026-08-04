require "json"
require "digest/sha256"
require "../../src/ml/sparse/cross_attention_output"
require "../spec_helper"

private CROSS_ATTENTION_OUTPUT_TOLERANCE = 5.0e-5_f32
private CROSS_ATTENTION_OUTPUT_DIGESTS   = {
  pre_output:   "c001e63fc83d92ed96bff8ecdc1a0751db5567a24fd94d6c937eafb2298212f1",
  final_output: "afa01c08b92d7750e1d0453fcd29b6049600c95d5c147f584141a2a5fa15b635",
}

private def cross_attention_output_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_cross_attention_seam_cpu_v1.json"
  )))
end

private def cross_attention_output_f32(
  payload : JSON::Any,
  output = [] of Float32,
) : Array(Float32)
  if hash = payload.as_h?
    if values = hash["values"]?
      return cross_attention_output_f32(values, output)
    end
  elsif array = payload.as_a?
    array.each { |entry| cross_attention_output_f32(entry, output) }
  else
    output << payload.as_f.to_f32
  end
  output
end

private def cross_attention_output_i32(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def cross_attention_output_f32le_sha256(
  values : Indexable(Float32),
) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def cross_attention_output_fixture_values(
  fixture : JSON::Any,
) : Tuple(
  ML::Sparse::TensorCPU,
  ML::Sparse::CrossAttentionPlanCPU,
  ML::NN::Linear,
)
  input = fixture["input"]
  point_count = input["coordinates"].as_a.size.to_i32
  channels = input["channels"].as_i.to_i32
  map = ML::Sparse::CoordinateMap3D.new(
    cross_attention_output_i32(input["coordinates"]),
    input["batch_size"].as_i.to_i32,
    {
      input["spatial_shape"][0].as_i.to_i32,
      input["spatial_shape"][1].as_i.to_i32,
      input["spatial_shape"][2].as_i.to_i32,
    }
  )
  query = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      cross_attention_output_f32(fixture["stages"]["norm2_output"]),
      ML::Shape.new(point_count, channels)
    ),
    map
  )
  context = ML::Tensor.from_array(
    cross_attention_output_f32(input["context"]),
    ML::Shape.new(
      input["batch_size"].as_i.to_i32,
      input["context_length"].as_i.to_i32,
      input["context_channels"].as_i.to_i32
    )
  )
  plan = ML::Sparse::CrossAttentionPlanCPU.preflight(
    query,
    context,
    input["heads"].as_i.to_i32
  )
  pre_output = ML::Sparse::TensorCPU.new(
    ML::Tensor.from_array(
      cross_attention_output_f32(fixture["cross_attention_pre_output"]),
      ML::Shape.new(point_count, channels)
    ),
    map
  )
  weight = fixture["parameters"]["to_out_weight"]
  layer = ML::NN::Linear.new(
    channels,
    channels,
    device: ML::Tensor::Device::CPU
  )
  cross_attention_output_f32(weight).each_with_index do |value, index|
    layer.weight.data.cpu_data.not_nil![index] = value
  end
  cross_attention_output_f32(fixture["parameters"]["to_out_bias"])
    .each_with_index do |value, index|
      layer.bias.not_nil!.data.cpu_data.not_nil![index] = value
    end
  layer.weight.requires_grad = false
  layer.bias.not_nil!.requires_grad = false
  {pre_output, plan, layer}
end

private def freeze_cross_attention_output_linear(
  layer : ML::NN::Linear,
) : ML::NN::Linear
  layer.weight.requires_grad = false
  layer.bias.try { |bias| bias.requires_grad = false }
  layer
end

class CrossAttentionOutputTensorCPUOverride < ML::Sparse::TensorCPU
end

describe "TRELLIS.2 sparse cross-attention output projection" do
  it "matches the pinned asymmetric biased to_out stage" do
    fixture = cross_attention_output_fixture
    pre_output, plan, to_out = cross_attention_output_fixture_values(fixture)
    before_input = pre_output.features_copy
    before_weight = to_out.weight.data.cpu_data.not_nil!.dup
    before_bias = to_out.bias.not_nil!.data.cpu_data.not_nil!.dup

    cross_attention_output_f32le_sha256(before_input).should eq(
      CROSS_ATTENTION_OUTPUT_DIGESTS[:pre_output]
    )
    expected = cross_attention_output_f32(
      fixture["stages"]["cross_attention_output"]
    )
    cross_attention_output_f32le_sha256(expected).should eq(
      CROSS_ATTENTION_OUTPUT_DIGESTS[:final_output]
    )

    output = ML::Sparse::TensorCPU.apply_cross_attention_output(
      pre_output,
      plan,
      to_out
    )

    output.coordinate_map.same?(pre_output.coordinate_map).should be_true
    output.point_count.should eq(pre_output.point_count)
    output.channels.should eq(pre_output.channels)
    output.max_feature_bytes.should eq(pre_output.max_feature_bytes)
    output.features_copy.zip(expected).each do |actual, wanted|
      actual.should be_close(wanted, CROSS_ATTENTION_OUTPUT_TOLERANCE)
    end
    output.features_copy.should_not eq(before_input)
    pre_output.features_copy.should eq(before_input)
    to_out.weight.data.cpu_data.not_nil!.should eq(before_weight)
    to_out.bias.not_nil!.data.cpu_data.not_nil!.should eq(before_bias)

    owned = output.features_copy
    to_out.weight.data.cpu_data.not_nil![0] = Float32::NAN
    to_out.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::NAN
    output.features_copy.should eq(owned)
  end

  it "rejects one-under byte and total-work caps before poisoned parameters" do
    fixture = cross_attention_output_fixture
    pre_output, plan, to_out = cross_attention_output_fixture_values(fixture)
    to_out.weight.data.cpu_data.not_nil![0] = Float32::NAN

    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /output would require #{plan.output_bytes} bytes/
    ) do
      ML::Sparse::TensorCPU.apply_cross_attention_output(
        pre_output,
        plan,
        to_out,
        max_output_bytes: plan.output_bytes - 1_i64
      )
    end
    expect_raises(
      ML::Sparse::SparseTensorBudgetError,
      /work would require #{plan.work_elements} MAC elements/
    ) do
      ML::Sparse::TensorCPU.apply_cross_attention_output(
        pre_output,
        plan,
        to_out,
        max_total_work_elements: plan.work_elements - 1_i64
      )
    end
  end

  it "requires the exact frozen biased C-to-C upstream boundary" do
    fixture = cross_attention_output_fixture
    pre_output, plan, _ = cross_attention_output_fixture_values(fixture)
    channels = pre_output.channels

    wrong_shape = freeze_cross_attention_output_linear(
      ML::NN::Linear.new(
        channels,
        channels + 1,
        device: ML::Tensor::Device::CPU
      )
    )
    expect_raises(ML::Sparse::SparseTensorError, /Linear\(16, 16\)/) do
      ML::Sparse::TensorCPU.apply_cross_attention_output(
        pre_output,
        plan,
        wrong_shape
      )
    end

    biasless = freeze_cross_attention_output_linear(
      ML::NN::Linear.new(
        channels,
        channels,
        bias: false,
        device: ML::Tensor::Device::CPU
      )
    )
    expect_raises(ML::Sparse::SparseTensorError, /biased to_out/) do
      ML::Sparse::TensorCPU.apply_cross_attention_output(
        pre_output,
        plan,
        biasless
      )
    end

    trainable = ML::NN::Linear.new(
      channels,
      channels,
      device: ML::Tensor::Device::CPU
    )
    expect_raises(ML::Sparse::SparseTensorError, /frozen/) do
      ML::Sparse::TensorCPU.apply_cross_attention_output(
        pre_output,
        plan,
        trainable
      )
    end
  end

  it "rejects a mismatched plan and inherited receiver override" do
    fixture = cross_attention_output_fixture
    pre_output, plan, to_out = cross_attention_output_fixture_values(fixture)
    duplicate_map = ML::Sparse::CoordinateMap3D.new(
      cross_attention_output_i32(fixture["input"]["coordinates"]),
      fixture["input"]["batch_size"].as_i.to_i32,
      {4, 4, 4}
    )
    duplicate = ML::Sparse::TensorCPU.new(
      ML::Tensor.from_array(
        pre_output.features_copy,
        ML::Shape.new(pre_output.point_count, pre_output.channels)
      ),
      duplicate_map
    )

    expect_raises(ML::Sparse::SparseTensorError, /does not match its plan/) do
      ML::Sparse::TensorCPU.apply_cross_attention_output(
        duplicate,
        plan,
        to_out
      )
    end
    expect_raises(ML::Sparse::SparseTensorError, /base TensorCPU receiver/) do
      CrossAttentionOutputTensorCPUOverride.apply_cross_attention_output(
        pre_output,
        plan,
        to_out
      )
    end
  end

  it "preserves production provenance and validates empty-input parameters" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {1, 1, 1})
    query = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 1_i32)),
      map,
      16_i64
    )
    context = ML::Tensor.zeros(1, 1, 1, device: ML::Tensor::Device::CPU)
    plan = ML::Sparse::CrossAttentionPlanCPU.preflight(query, context, 1)
    pre_output = ML::Sparse::TensorCPU.production(
      ML::Tensor.from_array([] of Float32, ML::Shape.new(0_i32, 1_i32)),
      map,
      16_i64
    )
    to_out = freeze_cross_attention_output_linear(
      ML::NN::Linear.new(1, 1, device: ML::Tensor::Device::CPU)
    )

    output = ML::Sparse::TensorCPU.apply_cross_attention_output(
      pre_output,
      plan,
      to_out
    )
    output.production_width?.should be_true
    output.features_copy.should be_empty
    output.coordinate_map.same?(map).should be_true

    to_out.bias.not_nil!.data.cpu_data.not_nil![0] = Float32::INFINITY
    expect_raises(ML::Sparse::SparseTensorError, /bias\[0\] must be finite/) do
      ML::Sparse::TensorCPU.apply_cross_attention_output(
        pre_output,
        plan,
        to_out
      )
    end
  end
end
