require "json"
require "digest/sha256"

# This is a planned production boundary.  It intentionally remains RED until
# src/ml/vision/dino_v3/block.cr is supplied by the owning implementation task.
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
      payload["probes"].as_a.size.should eq(4), "#{name} probe completeness"
      payload["probes"].as_a.each do |probe|
        probe["index"].as_a.size.should eq(payload["shape"].as_a.size), "#{name} probe rank"
      end
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
      layer_norm_eps: config["layer_norm_eps"].as_f.to_f32
    )
    block_parameters = ML::Vision::DinoV3::BlockParameters.new(
      block_config,
      norm1_weight: parameter_values["norm1_weight"],
      norm1_bias: parameter_values["norm1_bias"],
      q_weight: parameter_values["q_weight"],
      q_bias: parameter_values["q_bias"],
      k_weight: parameter_values["k_weight"],
      v_weight: parameter_values["v_weight"],
      v_bias: parameter_values["v_bias"],
      o_weight: parameter_values["o_weight"],
      o_bias: parameter_values["o_bias"],
      layer_scale1: parameter_values["layer_scale1"],
      norm2_weight: parameter_values["norm2_weight"],
      norm2_bias: parameter_values["norm2_bias"],
      up_weight: parameter_values["up_weight"],
      up_bias: parameter_values["up_bias"],
      down_weight: parameter_values["down_weight"],
      down_bias: parameter_values["down_bias"],
      layer_scale2: parameter_values["layer_scale2"]
    )
    trace = ML::Vision::DinoV3::BlockCPU.new(block_parameters).forward_with_trace(
      input,
      rope_cos,
      rope_sin
    )
    trace.parameter_f32le_sha256.should eq(parameter_sha)
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
  end
end
