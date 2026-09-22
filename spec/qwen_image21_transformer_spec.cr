require "./spec_helper"
require "../src/ml/gguf/qwen_image21_transformer"

private def qwen_image21_transformer_f32_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32)
  raw = Bytes.new(values.size * 4)
  raw.to_unsafe.copy_from(values.to_unsafe.as(Pointer(UInt8)), raw.size)
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
end

private def qwen_image21_transformer_matrix(out_dim : Int32, in_dim : Int32, phase : Int32)
  Array(Float32).new(out_dim * in_dim) do |index|
    (((index * 19 + phase * 11) % 37) - 18).to_f32 / 43.0_f32
  end
end

private def qwen_image21_transformer_fixture(include_layer : Bool = true)
  block_config = ML::GGUF::QwenImage21BlockConfig.new(
    hidden_dim: 6,
    heads: 1,
    head_dim: 6,
    intermediate_dim: 3,
    axes_dims: StaticArray[2, 2, 2],
  )
  config = ML::GGUF::QwenImage21TransformerConfig.new(
    input_dim: 4,
    output_dim: 4,
    context_dim: 5,
    time_input_dim: 4,
    block: block_config,
  )
  weight = ->(out_dim : Int32, in_dim : Int32, phase : Int32) do
    qwen_image21_transformer_f32_weight(
      qwen_image21_transformer_matrix(out_dim, in_dim, phase), out_dim, in_dim
    )
  end
  layer = ML::GGUF::QwenImage21BlockWeights.new(
    weight.call(6, 6, 9),
    weight.call(6, 6, 10),
    weight.call(6, 6, 11),
    weight.call(6, 6, 12),
    [1.0_f32, 0.9_f32, 1.1_f32, 0.8_f32, 1.2_f32, 0.95_f32],
    [0.85_f32, 1.05_f32, 0.9_f32, 1.15_f32, 0.8_f32, 1.1_f32],
    weight.call(6, 6, 13),
    weight.call(6, 3, 14),
  )
  weights = ML::GGUF::QwenImage21TransformerWeights.new(
    img_in: weight.call(6, 4, 1),
    modulation: weight.call(24, 6, 2),
    norm_out_linear: weight.call(6, 6, 3),
    proj_out: weight.call(4, 6, 4),
    timestep_linear_1: weight.call(6, 4, 5),
    timestep_linear_2: weight.call(6, 6, 6),
    text_in_layer: weight.call(6, 5, 7),
    text_out_layer: weight.call(6, 6, 8),
    text_norm: [0.10_f32, -0.08_f32, 0.04_f32, 0.0_f32, 0.12_f32],
    layers: include_layer ? [layer] : [] of ML::GGUF::QwenImage21BlockWeights,
  )
  image_latents = Array(Float32).new(8 * 4) { |i| (((i * 13) % 31) - 15).to_f32 / 29.0_f32 }
  encoder_hidden = Array(Float32).new(4 * 5) { |i| (((i * 7) % 29) - 14).to_f32 / 23.0_f32 }
  {
    config:         config,
    weights:        weights,
    image_latents:  image_latents,
    encoder_hidden: encoder_hidden,
    img_shapes:     [StaticArray[1, 2, 2], StaticArray[1, 2, 2]],
    img_mask:       [false, true, false, false, true],
    encoder_valid:  [true, true, false, true],
  }
end

describe ML::GGUF::QwenImage21TransformerCPU do
  it "builds exact block boundaries, centered positions, and padding validity" do
    layout = ML::GGUF::QwenImage21TransformerCPU.build_layout(
      [false, true, false, false, true],
      [StaticArray[1, 2, 2], StaticArray[1, 2, 2]],
      4,
      [true, true, false, true],
    )

    layout.image_pad_mask.should eq([false, true, true, true, true, false, false, true, true, true, true])
    layout.image_ids.should eq([-1, 0, 0, 0, 0, -1, -1, 1, 1, 1, 1])
    layout.target_token_mask.should eq([false, false, false, false, false, false, false, true, true, true, true])
    layout.positions.should eq([
      StaticArray[0, 0, 0],
      StaticArray[1, -1, -1], StaticArray[1, -1, 0],
      StaticArray[1, 0, -1], StaticArray[1, 0, 0],
      StaticArray[3, 3, 3], StaticArray[4, 4, 4],
      StaticArray[5, -1, -1], StaticArray[5, -1, 0],
      StaticArray[5, 0, -1], StaticArray[5, 0, 0],
    ])
    layout.key_valid.should eq([true, true, true, true, true, false, true, true, true, true, true])
  end

  it "keeps adjacent image slots as separate attention blocks" do
    layout = ML::GGUF::QwenImage21TransformerCPU.build_layout(
      [true, true, true],
      [StaticArray[1, 2, 2], StaticArray[1, 2, 2], StaticArray[1, 2, 2]],
      2,
    )

    layout.image_ids.should eq([
      0, 0, 0, 0,
      1, 1, 1, 1,
      2, 2, 2, 2,
    ])
  end

  it "matches an independent PyTorch oracle of the upstream outer equations" do
    fixture = qwen_image21_transformer_fixture
    result = ML::GGUF::QwenImage21TransformerCPU.forward(
      fixture[:image_latents],
      fixture[:encoder_hidden],
      0.625_f32,
      fixture[:img_shapes],
      fixture[:img_mask],
      fixture[:weights],
      fixture[:config],
      encoder_hidden_states_mask: fixture[:encoder_valid],
    )

    # Golden values are produced by
    # spec/support/qwen_image21_transformer_torch_reference.py.
    expected = [
      -0.163389266_f32, -0.165539145_f32, -0.167689055_f32, -0.976779819_f32,
      0.446061552_f32, 0.449024439_f32, 0.451987207_f32, -0.394000858_f32,
      -0.582533300_f32, -0.583961070_f32, -0.585388720_f32, -0.992484748_f32,
      -0.250272632_f32, -0.253261924_f32, -0.256251305_f32, 0.939634562_f32,
      0.484646797_f32, 0.487706274_f32, 0.490765750_f32, -0.308147579_f32,
      -0.472115546_f32, -0.476630419_f32, -0.481145322_f32, -0.098826125_f32,
      0.479212373_f32, 0.483974218_f32, 0.488736093_f32, -0.147444978_f32,
      -0.597488463_f32, -0.599148691_f32, -0.600808680_f32, -0.889214993_f32,
      -0.253141552_f32, -0.255884439_f32, -0.258627385_f32, 0.915194273_f32,
      0.535951376_f32, 0.538969159_f32, 0.541986942_f32, -0.177267179_f32,
      0.383459717_f32, 0.386444211_f32, 0.389428675_f32, -0.658058763_f32,
    ]
    result.output.zip(expected).each do |actual, reference|
      actual.should be_close(reference, 3e-5_f32)
    end
  end

  it "keeps prefix output timestep-independent under causal conditioning" do
    fixture = qwen_image21_transformer_fixture(include_layer: false)
    first = ML::GGUF::QwenImage21TransformerCPU.forward(
      fixture[:image_latents], fixture[:encoder_hidden], 0.125_f32,
      fixture[:img_shapes], fixture[:img_mask], fixture[:weights], fixture[:config],
      encoder_hidden_states_mask: fixture[:encoder_valid],
    )
    second = ML::GGUF::QwenImage21TransformerCPU.forward(
      fixture[:image_latents], fixture[:encoder_hidden], 0.875_f32,
      fixture[:img_shapes], fixture[:img_mask], fixture[:weights], fixture[:config],
      encoder_hidden_states_mask: fixture[:encoder_valid],
    )

    prefix_values = 7 * fixture[:config].output_dim
    first.output.first(prefix_values).should eq(second.output.first(prefix_values))
    first.output.last(4 * fixture[:config].output_dim).should_not eq(
      second.output.last(4 * fixture[:config].output_dim)
    )
  end
end
