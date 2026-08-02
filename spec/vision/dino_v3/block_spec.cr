require "json"
require "digest/sha256"

# Synthetic one-block execution boundary; real checkpoint configuration and
# model-scale execution remain deliberately outside this spec.
require "../../../src/ml/vision/dino_v3/block"
require "../../spec_helper"

private def dino_v3_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def dino_v3_payload_values(payload : JSON::Any) : Array(Float32)
  payload["values"].as_a.map { |value| value.as_f.to_f32 }
end

private def dino_v3_tensor_from_payload(payload : JSON::Any) : ML::Tensor
  shape = ML::Shape.new(payload["shape"].as_a.map { |value| value.as_i.to_i32 })
  ML::Tensor.from_array(dino_v3_payload_values(payload), shape)
end

private def dino_v3_flat_index(index : JSON::Any, shape : JSON::Any) : Int32
  coordinates = index.as_a.map { |value| value.as_i.to_i32 }
  dimensions = shape.as_a.map { |value| value.as_i.to_i32 }
  raise "probe rank mismatch" unless coordinates.size == dimensions.size
  flat = 0_i32
  dimensions.each_with_index do |dimension, axis|
    coordinate = coordinates[axis]
    raise "probe index out of bounds" unless 0 <= coordinate < dimension
    flat = flat * dimension + coordinate
  end
  flat
end

private def dino_v3_assert_probes(payload : JSON::Any, context : String) : Nil
  values = dino_v3_payload_values(payload)
  probes = payload["probes"].as_a
  probes.size.should eq(4), "#{context} probe completeness"
  probes.each do |probe|
    flat = dino_v3_flat_index(probe["index"], payload["shape"])
    values[flat].should eq(probe["value"].as_f.to_f32),
      "#{context} probe #{probe["index"]}"
  end
end

private def dino_v3_lattice(
  count : Int32,
  multiplier : Int32,
  offset : Int32,
  modulus : Int32,
  divisor : Float32,
) : Array(Float32)
  Array(Float32).new(count) do |index|
    (((index.to_i64 * multiplier + offset) % modulus) - modulus // 2).to_f32 / divisor
  end
end

private def dino_v3_parameters : Hash(String, Array(Float32))
  {
    "norm1_weight" => [0.73_f32, 1.11_f32, 0.89_f32, 1.27_f32, 0.97_f32, 0.81_f32, 1.19_f32, 0.67_f32],
    "norm1_bias"   => [-0.23_f32, 0.17_f32, 0.05_f32, -0.31_f32, 0.29_f32, -0.07_f32, 0.13_f32, -0.19_f32],
    "q_weight"     => dino_v3_lattice(64, 19, 2, 47, 113.0_f32),
    "q_bias"       => dino_v3_lattice(8, 11, 7, 31, 37.0_f32),
    "k_weight"     => dino_v3_lattice(64, 23, 3, 53, 127.0_f32),
    "v_weight"     => dino_v3_lattice(64, 29, 5, 59, 109.0_f32),
    "v_bias"       => dino_v3_lattice(8, 13, 1, 29, 41.0_f32),
    "o_weight"     => dino_v3_lattice(64, 31, 9, 61, 131.0_f32),
    "o_bias"       => dino_v3_lattice(8, 17, 4, 37, 43.0_f32),
    "layer_scale1" => [0.37_f32, 0.43_f32, 0.59_f32, 0.71_f32, 0.83_f32, 0.47_f32, 0.65_f32, 0.91_f32],
    "norm2_weight" => [1.21_f32, 0.79_f32, 1.07_f32, 0.93_f32, 1.31_f32, 0.69_f32, 1.17_f32, 0.87_f32],
    "norm2_bias"   => [0.11_f32, -0.21_f32, 0.09_f32, 0.27_f32, -0.15_f32, 0.03_f32, -0.29_f32, 0.19_f32],
    "up_weight"    => dino_v3_lattice(96, 37, 4, 67, 151.0_f32),
    "up_bias"      => dino_v3_lattice(12, 19, 3, 41, 47.0_f32),
    "down_weight"  => dino_v3_lattice(96, 41, 8, 71, 139.0_f32),
    "down_bias"    => dino_v3_lattice(8, 23, 6, 43, 53.0_f32),
    "layer_scale2" => [0.41_f32, 0.53_f32, 0.67_f32, 0.79_f32, 0.61_f32, 0.89_f32, 0.73_f32, 0.97_f32],
  }
end

private def dino_v3_block_parameters(
  config : ML::Vision::DinoV3::BlockConfig,
  values : Hash(String, Array(Float32)) = dino_v3_parameters,
) : ML::Vision::DinoV3::BlockParameters
  ML::Vision::DinoV3::BlockParameters.new(
    config,
    norm1_weight: values["norm1_weight"],
    norm1_bias: values["norm1_bias"],
    q_weight: values["q_weight"],
    q_bias: values["q_bias"],
    k_weight: values["k_weight"],
    v_weight: values["v_weight"],
    v_bias: values["v_bias"],
    o_weight: values["o_weight"],
    o_bias: values["o_bias"],
    layer_scale1: values["layer_scale1"],
    norm2_weight: values["norm2_weight"],
    norm2_bias: values["norm2_bias"],
    up_weight: values["up_weight"],
    up_bias: values["up_bias"],
    down_weight: values["down_weight"],
    down_bias: values["down_bias"],
    layer_scale2: values["layer_scale2"]
  )
end

private def dino_v3_input : Array(Float32)
  dino_v3_lattice(56, 17, 5, 43, 19.0_f32)
end

private def dino_v3_assert_payload(
  actual : ML::Tensor,
  payload : JSON::Any,
  absolute : Float32,
  relative : Float32,
  context : String,
) : Nil
  expected_shape = payload["shape"].as_a.map { |value| value.as_i.to_i32 }
  actual.shape.to_a.should eq(expected_shape), "#{context} shape"
  expected_values = dino_v3_payload_values(payload)
  expected_values.size.should eq(actual.numel), "#{context} flattened size"
  actual_values = actual.to_a
  max_ratio = 0.0_f32
  max_index = 0
  actual_values.each_with_index do |value, index|
    value.finite?.should be_true, "#{context}[#{index}] finite"
    delta = (value - expected_values[index]).abs
    limit = absolute + relative * expected_values[index].abs
    ratio = delta / Math.max(limit, 1.0e-30_f32)
    if ratio > max_ratio
      max_ratio = ratio
      max_index = index
    end
  end
  max_ratio.should be <= 1.0_f32,
    "#{context} max_ratio=#{max_ratio} index=#{max_index} " \
    "actual=#{actual_values[max_index]} expected=#{expected_values[max_index]}"
end

describe ML::Vision::DinoV3::BlockCPU do
  it "matches every pinned one-block boundary and extractor final norm" do
    fixture_path = File.join(
      __DIR__,
      "../../fixtures/trellis2/dino_v3_block_cpu_v1.json"
    )
    fixture = JSON.parse(File.read(fixture_path))
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/dino-v3-block-oracle/v1"
    )

    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["transformers_version"].as_s.should eq("5.8.1")
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["numpy_version"].as_s.should eq("2.1.3")
    provenance["transformers_modeling_sha256"].as_s.should eq(
      "6073b7665eea50fb2260d86984011e6af8ed68cbd37ae57a6095e5e90e0eea34"
    )
    provenance["transformers_config_sha256"].as_s.should eq(
      "9a13d3c9ea8020aaed7057db28a7ffe0ffd9bb094a6178261bcc45ed04e9bbfc"
    )
    provenance["trellis_extractor_sha256"].as_s.should eq(
      "12530b23e8b6a2cc6b87d8cd01922c7b0085199a365b731f19dd0e7ef4919150"
    )
    provenance["model_reference"].as_s.should eq(
      "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    provenance["model_revision"].as_s.should eq(
      "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    )
    provenance["weights"].as_s.should contain("synthetic only")
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["real_model_config_status"].as_s.should eq(
      "unavailable/gated/not_used"
    )
    provenance["real_model_config"].as_s.should contain("not loaded")

    object_path = fixture["object_path"]
    object_path["root_layer_attribute_present"].as_bool.should be_false
    object_path["root_model_layer_attribute_present"].as_bool.should be_true
    object_path["state_dict_layer_prefix"].as_s.should eq("model.layer.0")
    object_path["state_dict_layer_key_count"].as_i.should eq(17)
    fixture["context_bridge"]["root_path"].as_s.should eq(".layer absent")
    fixture["context_bridge"]["encoder_path"].as_s.should eq(".model.layer present")
    fixture["context_bridge"]["state_dict_prefix"].as_s.should eq("model.layer.0")

    attention_backend = fixture["attention_backend"]
    attention_backend["transformers_default"].as_s.should eq("sdpa")
    attention_backend["oracle"].as_s.should eq("eager")

    config = fixture["config"]
    {
      "image_size"          => 32_i64,
      "patch_size"          => 16_i64,
      "num_channels"        => 3_i64,
      "hidden_size"         => 8_i64,
      "intermediate_size"   => 12_i64,
      "num_hidden_layers"   => 1_i64,
      "num_attention_heads" => 2_i64,
      "num_register_tokens" => 2_i64,
    }.each do |field, expected|
      config[field].as_i.should eq(expected), "config #{field}"
    end
    config["rope_theta"].as_f.should eq(100.0)
    config["layer_norm_eps"].as_f.should eq(1.0e-5)
    config["hidden_act"].as_s.should eq("gelu")
    config["query_bias"].as_bool.should be_true
    config["key_bias"].as_bool.should be_false
    config["value_bias"].as_bool.should be_true
    config["proj_bias"].as_bool.should be_true
    config["mlp_bias"].as_bool.should be_true
    config["layerscale_value"].as_f.should eq(1.0)
    config["attention_dropout"].as_f.should eq(0.0)
    config["drop_path_rate"].as_f.should eq(0.0)
    config["use_gated_mlp"].as_bool.should be_false
    config["attn_implementation"].as_s.should eq("eager")
    config["training"].as_bool.should be_false
    config["dtype"].as_s.should eq("float32")
    config["device"].as_s.should eq("cpu")

    tolerance = fixture["tolerance"]
    absolute = tolerance["absolute"].as_f.to_f32
    relative = tolerance["relative"].as_f.to_f32
    absolute.should eq(5.0e-5_f32)
    relative.should eq(5.0e-5_f32)
    tolerance["extractor_final_layer_norm_eps"].as_f.should eq(1.0e-5)
    tolerance["comparison"].as_s.should eq(
      "bounded absolute/relative tolerance; not byte identity"
    )

    expected_order = fixture["expected_order"].as_a.map(&.as_s)
    expected_order.should eq([
      "input", "norm1", "q_heads", "k_heads", "v_heads", "q_rope", "k_rope",
      "scores", "probabilities", "attention_context", "output_projection",
      "layer_scale1", "first_residual", "norm2", "mlp_up", "exact_gelu",
      "mlp_down", "layer_scale2", "block_output", "extractor_final",
    ])
    expected = fixture["expected"]
    expected_order.each do |name|
      payload = expected[name]
      payload["values"].as_a.size.should eq(
        payload["shape"].as_a.reduce(1_i64) { |count, dim| count * dim.as_i }
      ), "#{name} full flattened payload"
      dino_v3_assert_probes(payload, name)
      dino_v3_f32le_sha256(dino_v3_payload_values(payload)).should eq(
        payload["f32le_sha256"].as_s
      ), "#{name} fixture byte hash"
    end

    input_payload = fixture["inputs"]["input"]
    input_values = dino_v3_input
    dino_v3_f32le_sha256(input_values).should eq(input_payload["f32le_sha256"].as_s)
    input_values.size.should eq(input_payload["values"].as_a.size)
    input = ML::Tensor.from_array(
      input_values,
      ML::Shape.new(input_payload["shape"].as_a.map { |value| value.as_i.to_i32 })
    )
    dino_v3_f32le_sha256(input.to_a).should eq(input_payload["f32le_sha256"].as_s)
    rope_cos = dino_v3_tensor_from_payload(fixture["inputs"]["rope_cos"])
    rope_sin = dino_v3_tensor_from_payload(fixture["inputs"]["rope_sin"])
    rope_cos_sha = fixture["inputs"]["rope_cos"]["f32le_sha256"].as_s
    rope_sin_sha = fixture["inputs"]["rope_sin"]["f32le_sha256"].as_s
    {"rope_cos", "rope_sin"}.each do |name|
      payload = fixture["inputs"][name]
      dino_v3_f32le_sha256(dino_v3_payload_values(payload)).should eq(
        payload["f32le_sha256"].as_s
      ), "#{name} fixture byte hash"
      dino_v3_assert_probes(payload, name)
    end
    fixture["inputs"]["dummy_image"]["shape"].as_a.map(&.as_i).should eq([1, 3, 32, 32])

    parameter_values = dino_v3_parameters
    parameter_section = fixture["parameters"]
    parameter_order = parameter_section["order"].as_a.map(&.as_s)
    parameter_order.should eq(parameter_values.keys)
    parameter_bytes = [] of Float32
    parameter_order.each do |name|
      values = parameter_values[name]
      payload = parameter_section["tensors"][name]
      values.should eq(dino_v3_payload_values(payload)), "#{name} deterministic values"
      dino_v3_f32le_sha256(values).should eq(payload["f32le_sha256"].as_s), "#{name} hash"
      parameter_bytes.concat(values)
    end
    parameter_sha = dino_v3_f32le_sha256(parameter_bytes)
    parameter_sha.should eq(parameter_section["f32le_sha256"].as_s)

    block_config = ML::Vision::DinoV3::BlockConfig.new(
      hidden_size: config["hidden_size"].as_i.to_i32,
      intermediate_size: config["intermediate_size"].as_i.to_i32,
      num_attention_heads: config["num_attention_heads"].as_i.to_i32,
      num_register_tokens: config["num_register_tokens"].as_i.to_i32,
      layer_norm_eps: config["layer_norm_eps"].as_f.to_f32,
      hidden_act: config["hidden_act"].as_s,
      query_bias: config["query_bias"].as_bool,
      key_bias: config["key_bias"].as_bool,
      value_bias: config["value_bias"].as_bool,
      proj_bias: config["proj_bias"].as_bool,
      mlp_bias: config["mlp_bias"].as_bool,
      attention_dropout: config["attention_dropout"].as_f.to_f32,
      drop_path_rate: config["drop_path_rate"].as_f.to_f32,
      use_gated_mlp: config["use_gated_mlp"].as_bool,
      attention_backend: config["attn_implementation"].as_s,
      training: config["training"].as_bool,
      extractor_final_layer_norm_eps: tolerance["extractor_final_layer_norm_eps"].as_f.to_f32
    )
    block_config.hidden_act.should eq("gelu")
    block_config.attention_backend.should eq("eager")
    block_parameters = dino_v3_block_parameters(block_config, parameter_values)
    trace = ML::Vision::DinoV3::BlockCPU.new(block_parameters).forward_with_trace(
      input,
      rope_cos,
      rope_sin
    )
    trace.parameter_f32le_sha256.should eq(parameter_sha)
    trace.trellis_extractor_sha256.should eq(
      provenance["trellis_extractor_sha256"].as_s
    )
    trace.source_revision.should eq(provenance["commit"].as_s)
    trace.transformers_modeling_sha256.should eq(
      provenance["transformers_modeling_sha256"].as_s
    )
    trace.transformers_config_sha256.should eq(
      provenance["transformers_config_sha256"].as_s
    )

    actual = {
      "input"             => trace.input,
      "norm1"             => trace.norm1,
      "q_heads"           => trace.q_heads,
      "k_heads"           => trace.k_heads,
      "v_heads"           => trace.v_heads,
      "q_rope"            => trace.q_rope,
      "k_rope"            => trace.k_rope,
      "scores"            => trace.scores,
      "probabilities"     => trace.probabilities,
      "attention_context" => trace.attention_context,
      "output_projection" => trace.output_projection,
      "layer_scale1"      => trace.layer_scale1,
      "first_residual"    => trace.first_residual,
      "norm2"             => trace.norm2,
      "mlp_up"            => trace.mlp_up,
      "exact_gelu"        => trace.exact_gelu,
      "mlp_down"          => trace.mlp_down,
      "layer_scale2"      => trace.layer_scale2,
      "block_output"      => trace.block_output,
      "extractor_final"   => trace.extractor_final,
    }
    actual.keys.should eq(expected_order)
    expected_order.each do |name|
      dino_v3_assert_payload(actual[name], expected[name], absolute, relative, name)
    end

    prefix_tokens = 1 + block_config.num_register_tokens
    q_before = trace.q_heads.to_a
    q_after = trace.q_rope.to_a
    k_before = trace.k_heads.to_a
    k_after = trace.k_rope.to_a
    token_count = input.shape[1]
    q_patch_changed = false
    k_patch_changed = false
    block_config.num_attention_heads.times do |head|
      prefix_tokens.times do |token|
        block_config.head_dim.times do |dimension|
          index = (head * token_count + token) * block_config.head_dim + dimension
          q_after[index].should eq(q_before[index]), "query prefix #{index}"
          k_after[index].should eq(k_before[index]), "key prefix #{index}"
        end
      end
      (prefix_tokens...token_count).each do |token|
        block_config.head_dim.times do |dimension|
          index = (head * token_count + token) * block_config.head_dim + dimension
          q_patch_changed ||= q_after[index] != q_before[index]
          k_patch_changed ||= k_after[index] != k_before[index]
        end
      end
    end
    q_patch_changed.should be_true
    k_patch_changed.should be_true
    dino_v3_f32le_sha256(input.to_a).should eq(input_payload["f32le_sha256"].as_s)
    dino_v3_f32le_sha256(rope_cos.to_a).should eq(rope_cos_sha)
    dino_v3_f32le_sha256(rope_sin.to_a).should eq(rope_sin_sha)
    current_parameter_values = [] of Float32
    parameter_order.each { |name| current_parameter_values.concat(parameter_values[name]) }
    dino_v3_f32le_sha256(current_parameter_values).should eq(parameter_sha)
  end

  it "fails closed on malformed geometry, retained parameter mutation, and non-finite math" do
    expect_raises(ML::Vision::DinoV3::BlockError, /head dimension/) do
      ML::Vision::DinoV3::BlockConfig.new(
        hidden_size: 12,
        intermediate_size: 24,
        num_attention_heads: 2,
        num_register_tokens: 2,
        layer_norm_eps: 1.0e-5_f32
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /exact GELU/) do
      ML::Vision::DinoV3::BlockConfig.new(
        hidden_size: 8, intermediate_size: 12, num_attention_heads: 2,
        num_register_tokens: 2, layer_norm_eps: 1.0e-5_f32,
        hidden_act: "gelu_new"
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /no key bias/) do
      ML::Vision::DinoV3::BlockConfig.new(
        hidden_size: 8, intermediate_size: 12, num_attention_heads: 2,
        num_register_tokens: 2, layer_norm_eps: 1.0e-5_f32,
        key_bias: true
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /zero attention dropout/) do
      ML::Vision::DinoV3::BlockConfig.new(
        hidden_size: 8, intermediate_size: 12, num_attention_heads: 2,
        num_register_tokens: 2, layer_norm_eps: 1.0e-5_f32,
        attention_dropout: 0.1_f32
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /gated MLP/) do
      ML::Vision::DinoV3::BlockConfig.new(
        hidden_size: 8, intermediate_size: 12, num_attention_heads: 2,
        num_register_tokens: 2, layer_norm_eps: 1.0e-5_f32,
        use_gated_mlp: true
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /eager attention/) do
      ML::Vision::DinoV3::BlockConfig.new(
        hidden_size: 8, intermediate_size: 12, num_attention_heads: 2,
        num_register_tokens: 2, layer_norm_eps: 1.0e-5_f32,
        attention_backend: "sdpa"
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /evaluation mode/) do
      ML::Vision::DinoV3::BlockConfig.new(
        hidden_size: 8, intermediate_size: 12, num_attention_heads: 2,
        num_register_tokens: 2, layer_norm_eps: 1.0e-5_f32,
        training: true
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /must equal 1e-5/) do
      ML::Vision::DinoV3::BlockConfig.new(
        hidden_size: 8, intermediate_size: 12, num_attention_heads: 2,
        num_register_tokens: 2, layer_norm_eps: 1.0e-5_f32,
        extractor_final_layer_norm_eps: 1.0e-6_f32
      )
    end

    config = ML::Vision::DinoV3::BlockConfig.new(
      hidden_size: 8,
      intermediate_size: 12,
      num_attention_heads: 2,
      num_register_tokens: 2,
      layer_norm_eps: 1.0e-5_f32
    )
    values = dino_v3_parameters
    parameters = dino_v3_block_parameters(config, values)
    block = ML::Vision::DinoV3::BlockCPU.new(parameters)
    valid_input = ML::Tensor.from_array(
      dino_v3_input,
      ML::Shape.new(1_i32, 7_i32, 8_i32)
    )
    valid_cos = ML::Tensor.ones(4, 4, device: ML::Tensor::Device::CPU)
    valid_sin = ML::Tensor.zeros(4, 4, device: ML::Tensor::Device::CPU)

    expect_raises(ML::Vision::DinoV3::BlockError, /token count/) do
      block.forward_with_trace(
        ML::Tensor.zeros(1, 65, 8, device: ML::Tensor::Device::CPU),
        valid_cos,
        valid_sin
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /patch count 33/) do
      block.forward_with_trace(
        ML::Tensor.zeros(1, 36, 8, device: ML::Tensor::Device::CPU),
        ML::Tensor.ones(33, 4, device: ML::Tensor::Device::CPU),
        ML::Tensor.zeros(33, 4, device: ML::Tensor::Device::CPU)
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /rope cos must have shape/) do
      block.forward_with_trace(
        valid_input,
        ML::Tensor.ones(3, 4, device: ML::Tensor::Device::CPU),
        valid_sin
      )
    end
    expect_raises(ML::Vision::DinoV3::BlockError, /must be contiguous/) do
      block.forward_with_trace(
        ML::Tensor.zeros(1, 8, 7, device: ML::Tensor::Device::CPU).transpose,
        valid_cos,
        valid_sin
      )
    end

    nonfinite_input_values = dino_v3_input
    nonfinite_input_values[9] = Float32::NAN
    expect_raises(ML::Vision::DinoV3::BlockError, /input\[9\] must be finite/) do
      block.forward_with_trace(
        ML::Tensor.from_array(nonfinite_input_values, ML::Shape.new(1_i32, 7_i32, 8_i32)),
        valid_cos,
        valid_sin
      )
    end
    nonfinite_rope = Array(Float32).new(16, 1.0_f32)
    nonfinite_rope[5] = Float32::NAN
    expect_raises(ML::Vision::DinoV3::BlockError, /rope cos\[5\] must be finite/) do
      block.forward_with_trace(
        valid_input,
        ML::Tensor.from_array(nonfinite_rope, ML::Shape.new(4_i32, 4_i32)),
        valid_sin
      )
    end

    original_q = values["q_weight"][0]
    values["q_weight"][0] = Float32::NAN
    expect_raises(ML::Vision::DinoV3::BlockError, /query weight\[0\] must be finite/) do
      block.forward_with_trace(valid_input, valid_cos, valid_sin)
    end
    values["q_weight"][0] = original_q

    original_output_bias = values["o_bias"][0]
    original_layer_scale = values["layer_scale1"][0]
    values["o_bias"][0] = Float32::MAX
    values["layer_scale1"][0] = Float32::MAX
    expect_raises(
      ML::Vision::DinoV3::BlockError,
      /intermediate (arithmetic overflow|.*must be finite)/
    ) do
      block.forward_with_trace(valid_input, valid_cos, valid_sin)
    end
    values["o_bias"][0] = original_output_bias
    values["layer_scale1"][0] = original_layer_scale
  end

  it "derives the patch-only RoPE boundary when register tokens are absent" do
    config = ML::Vision::DinoV3::BlockConfig.new(
      hidden_size: 8,
      intermediate_size: 12,
      num_attention_heads: 2,
      num_register_tokens: 0,
      layer_norm_eps: 1.0e-5_f32
    )
    block = ML::Vision::DinoV3::BlockCPU.new(
      dino_v3_block_parameters(config)
    )
    input = ML::Tensor.from_array(
      dino_v3_input.first(40),
      ML::Shape.new(1_i32, 5_i32, 8_i32)
    )
    trace = block.forward_with_trace(
      input,
      ML::Tensor.zeros(4, 4, device: ML::Tensor::Device::CPU),
      ML::Tensor.ones(4, 4, device: ML::Tensor::Device::CPU)
    )
    trace.block_output.shape.to_a.should eq([1, 5, 8])
    trace.extractor_final.shape.to_a.should eq([1, 5, 8])
    before = trace.q_heads.to_a
    after = trace.q_rope.to_a
    config.num_attention_heads.times do |head|
      config.head_dim.times do |dimension|
        prefix_index = head * 5 * config.head_dim + dimension
        after[prefix_index].should eq(before[prefix_index])
        patch_index = prefix_index + config.head_dim
        after[patch_index].should_not eq(before[patch_index])
      end
    end
  end
end
