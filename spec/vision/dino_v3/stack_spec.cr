require "../../../src/ml/vision/dino_v3/stack"
require "../../spec_helper"

private def dino_v3_stack_identity(size : Int32) : Array(Float32)
  values = Array(Float32).new(size * size, 0.0_f32)
  size.times { |index| values[index * size + index] = 1.0_f32 }
  values
end

private def dino_v3_stack_parameters(
  config : ML::Vision::DinoV3::BlockConfig,
) : ML::Vision::DinoV3::BlockParameters
  width = config.hidden_size
  intermediate = config.intermediate_size
  zeros = Array.new(width, 0.0_f32)
  ones = Array.new(width, 1.0_f32)
  ML::Vision::DinoV3::BlockParameters.new(
    config,
    norm1_weight: ones.dup,
    norm1_bias: zeros.dup,
    q_weight: dino_v3_stack_identity(width),
    q_bias: zeros.dup,
    k_weight: dino_v3_stack_identity(width),
    v_weight: dino_v3_stack_identity(width),
    v_bias: zeros.dup,
    o_weight: dino_v3_stack_identity(width),
    o_bias: zeros.dup,
    layer_scale1: ones.dup,
    norm2_weight: ones.dup,
    norm2_bias: zeros.dup,
    up_weight: dino_v3_stack_identity(intermediate),
    up_bias: Array.new(intermediate, 0.0_f32),
    down_weight: dino_v3_stack_identity(intermediate),
    down_bias: zeros.dup,
    layer_scale2: ones.dup
  )
end

private def dino_v3_stack_block(
  register_tokens : Int32 = 0,
) : ML::Vision::DinoV3::BlockCPU
  config = ML::Vision::DinoV3::BlockConfig.new(
    hidden_size: 4,
    intermediate_size: 4,
    num_attention_heads: 1,
    num_register_tokens: register_tokens,
    layer_norm_eps: 1.0e-5_f32
  )
  ML::Vision::DinoV3::BlockCPU.new(dino_v3_stack_parameters(config))
end

private def dino_v3_stack_input : ML::Tensor
  ML::Tensor.from_array(
    [-0.5_f32, 0.25_f32, 0.75_f32, -1.0_f32, 0.5_f32, -0.25_f32, 1.0_f32, 0.125_f32],
    ML::Shape.new([1, 2, 4])
  )
end

private def dino_v3_stack_rope : ML::Tensor
  ML::Tensor.from_array([1.0_f32, 1.0_f32, 1.0_f32, 1.0_f32], ML::Shape.new([1, 4]))
end

describe ML::Vision::DinoV3::BlockStackCPU do
  it "composes block outputs and applies final normalization once" do
    first = dino_v3_stack_block
    second = dino_v3_stack_block
    stack = ML::Vision::DinoV3::BlockStackCPU.new([first, second])
    input = dino_v3_stack_input
    rope = dino_v3_stack_rope

    first_output = first.forward_with_trace(input, rope, rope).block_output
    expected = second.forward_with_trace(first_output, rope, rope).extractor_final
    actual = stack.forward(input, rope, rope)

    stack.depth.should eq(2)
    actual.shape.should eq(expected.shape)
    actual.to_a.should eq(expected.to_a)
  end

  it "keeps layer paths bounded to the admitted depth" do
    stack = ML::Vision::DinoV3::BlockStackCPU.new([
      dino_v3_stack_block,
      dino_v3_stack_block,
    ])

    stack.runtime_path.should eq("model.layer")
    stack.serialized_state_dict_prefix(0).should eq("model.layer.0")
    stack.serialized_state_dict_prefix(1).should eq("model.layer.1")
    stack.layer_index_from_serialized_prefix("model.layer.1").should eq(1)

    expect_raises(ML::Vision::DinoV3::StackError, /depth/) do
      stack.serialized_state_dict_prefix(2)
    end
    expect_raises(ML::Vision::DinoV3::StackError, /depth/) do
      stack.layer_index_from_serialized_prefix("model.layer.2")
    end
    expect_raises(ML::Vision::DinoV3::ConfigError, /serialized layer prefix/) do
      stack.layer_index_from_serialized_prefix("layer.0")
    end
  end

  it "rejects empty, oversized, and structurally mismatched stacks" do
    expect_raises(ML::Vision::DinoV3::StackError, /at least one/) do
      ML::Vision::DinoV3::BlockStackCPU.new([] of ML::Vision::DinoV3::BlockCPU)
    end

    expect_raises(ML::Vision::DinoV3::StackBudgetError, /maximum/) do
      ML::Vision::DinoV3::BlockStackCPU.new(Array.new(5) { dino_v3_stack_block })
    end

    expect_raises(ML::Vision::DinoV3::StackError, /compatible/) do
      ML::Vision::DinoV3::BlockStackCPU.new([
        dino_v3_stack_block,
        dino_v3_stack_block(1),
      ])
    end
  end

  it "copies the layer collection and rejects a non-pinned runtime path" do
    layers = [dino_v3_stack_block, dino_v3_stack_block]
    stack = ML::Vision::DinoV3::BlockStackCPU.new(layers)
    layers.clear
    stack.depth.should eq(2)
    stack.layers.clear
    stack.depth.should eq(2)

    expect_raises(ML::Vision::DinoV3::ConfigError, /pinned/) do
      ML::Vision::DinoV3::BlockStackCPU.new(
        [dino_v3_stack_block],
        runtime_path: "model.model.layer"
      )
    end
  end
end
