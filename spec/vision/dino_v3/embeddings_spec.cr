require "json"
require "digest/sha256"
require "../../../src/ml/vision/dino_v3/embeddings"
require "../../spec_helper"

private def dino_v3_f32le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def dino_v3_truncated_1e4le_sha256(values : Indexable(Float32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    quantized = (value * 10_000.0_f32).trunc.to_i32
    IO::ByteFormat::LittleEndian.encode(quantized, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

private def dino_v3_lcg_f32(count : Int32, seed : UInt32) : Array(Float32)
  state = seed
  Array(Float32).new(count) do
    state = (
      (state.to_u64 * 1_664_525_u64 + 1_013_904_223_u64) & 0xffff_ffff_u64
    ).to_u32
    signed = ((state >> 8) % 2001_u32).to_i32 - 1000
    signed.to_f32 / 997.0_f32
  end
end

private def dino_v3_input(edge : Int32) : ML::Tensor
  tensor = ML::Tensor.new(
    ML::Shape.new(1_i32, 3_i32, edge, edge),
    device: ML::Tensor::Device::CPU
  )
  values = tensor.cpu_data.not_nil!
  3.times do |channel|
    edge.times do |y|
      edge.times do |x|
        offset = channel * edge * edge + y * edge + x
        values[offset] = (
          ((5 * channel + 3 * y + 7 * x) % 33) - 16
        ).to_f32 / 16.0_f32
      end
    end
  end
  tensor
end

private def dino_v3_parameters(
  config : ML::Vision::DinoV3::EmbeddingConfig,
) : ML::Vision::DinoV3::EmbeddingParameters
  patch = config.patch_size
  channels = config.num_channels
  hidden = config.hidden_size
  weight = Array(Float32).new(hidden * channels * patch * patch) do |index|
    x = index
    kernel_x = x % patch
    x //= patch
    kernel_y = x % patch
    x //= patch
    channel = x % channels
    output = x // channels
    (
      (11 * output + 7 * channel + 5 * kernel_y + 3 * kernel_x) % 17 - 8
    ).to_f32 / 64.0_f32
  end
  bias = Array(Float32).new(hidden) { |i| (i - 8).to_f32 / 32.0_f32 }
  cls = Array(Float32).new(hidden) { |i| (i - 8).to_f32 / 16.0_f32 }
  registers = Array(Float32).new(config.num_register_tokens * hidden) do |i|
    (i - 32).to_f32 / 32.0_f32
  end
  ML::Vision::DinoV3::EmbeddingParameters.new(
    config,
    weight,
    bias,
    cls,
    registers
  )
end

private def dino_v3_parameters_sha256(
  parameters : ML::Vision::DinoV3::EmbeddingParameters,
) : String
  values = parameters.patch_weight + parameters.patch_bias +
           parameters.cls_token + parameters.register_tokens
  dino_v3_f32le_sha256(values)
end

private def dino_v3_assert_probes(
  tensor : ML::Tensor,
  probes : Array(JSON::Any),
  tolerance : Float32,
  context : String,
) : Nil
  probes.each do |probe|
    index = probe["index"].as_a.map(&.as_i.to_i32)
    tensor[index].should be_close(
      probe["value"].as_f.to_f32,
      tolerance
    ), "#{context} probe #{index}"
  end
end

describe ML::Vision::DinoV3::EmbeddingCPU do
  it "matches pinned synthetic Transformers embeddings and dynamic RoPE at 512 and 1024" do
    fixture_path = File.join(
      __DIR__,
      "../../fixtures/trellis2/dino_v3_embeddings_cpu_v1.json"
    )
    fixture = JSON.parse(File.read(fixture_path))
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/dino-v3-embeddings-oracle/v1"
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
    provenance["extractor"].as_s.should contain(
      "/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/image_feature_extractor.py"
    )
    provenance["model_reference"].as_s.should eq(
      "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    provenance["transformers_license"].as_s.should eq("Apache-2.0")
    provenance["generator"].as_s.should eq(
      "tools/trellis2_oracle/export_dino_v3_embeddings.py"
    )
    provenance["weights"].as_s.should contain("synthetic only")
    provenance["device"].as_s.should eq("cpu")
    provenance["network"].as_s.should eq("none")
    provenance["real_model_config"].as_s.should contain("unadmitted")
    compatibility = fixture["compatibility"]
    compatibility["pinned_upstream_layer_path"].as_s.should eq("model.layer")
    compatibility["transformers_5_8_1_layer_path"].as_s.should eq(
      "model.model.layer"
    )
    compatibility["pinned_upstream_path_available_on_instance"].as_bool.should be_false
    compatibility["transformers_5_8_1_path_available_on_instance"].as_bool.should be_true
    compatibility["status"].as_s.should contain("blocked for full encoder")

    synthetic = fixture["synthetic_config"]
    synthetic["head_dim"].as_i.should eq(8)
    synthetic["training"].as_bool.should be_false
    config = ML::Vision::DinoV3::EmbeddingConfig.new(
      patch_size: synthetic["patch_size"].as_i.to_i32,
      num_channels: synthetic["num_channels"].as_i.to_i32,
      hidden_size: synthetic["hidden_size"].as_i.to_i32,
      num_attention_heads: synthetic["num_attention_heads"].as_i.to_i32,
      num_register_tokens: synthetic["num_register_tokens"].as_i.to_i32,
      rope_theta: synthetic["rope_theta"].as_f.to_f32
    )
    parameters = dino_v3_parameters(config)
    parameter_sha256 = dino_v3_parameters_sha256(parameters)
    parameter_sha256.should eq(fixture["parameters"]["f32le_sha256"].as_s)
    embedding = ML::Vision::DinoV3::EmbeddingCPU.new(parameters)

    cases = fixture["cases"].as_a
    cases.size.should eq(2)
    cases.map { |test_case| test_case["name"].as_s }.should eq(
      ["synthetic_512", "synthetic_1024"]
    )
    expected_input_shapes = {
      "synthetic_512"  => [1, 3, 512, 512],
      "synthetic_1024" => [1, 3, 1024, 1024],
    }
    cases.each do |test_case|
      name = test_case["name"].as_s
      input_shape = test_case["input"]["shape"].as_a.map(&.as_i.to_i32)
      edge = input_shape[2]
      input_shape.should eq(expected_input_shapes[name]), "#{name} input shape metadata"
      input = dino_v3_input(edge)
      input_sha256 = dino_v3_f32le_sha256(input.cpu_data.not_nil!)
      input_sha256.should eq(test_case["input"]["f32le_sha256"].as_s)

      result = embedding.forward(input)
      result.source_revision.should eq(
        "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
      )
      result.transformers_modeling_sha256.should eq(
        provenance["transformers_modeling_sha256"].as_s
      )
      result.transformers_config_sha256.should eq(
        provenance["transformers_config_sha256"].as_s
      )
      result.config.hidden_size.should eq(config.hidden_size)
      result.config.num_attention_heads.should eq(config.num_attention_heads)
      result.config.num_register_tokens.should eq(config.num_register_tokens)
      result.parameter_f32le_sha256.should eq(parameter_sha256)

      {
        "patches"     => result.patches,
        "embeddings"  => result.embeddings,
        "coordinates" => result.patch_coordinates,
      }.each do |boundary, tensor|
        expected = test_case["expected"][boundary]
        tensor.shape.to_a.should eq(
          expected["shape"].as_a.map(&.as_i.to_i32)
        ), "#{name} #{boundary} shape"
        dino_v3_f32le_sha256(tensor.cpu_data.not_nil!).should eq(
          expected["f32le_sha256"].as_s
        ), "#{name} #{boundary} bytes"
        dino_v3_assert_probes(
          tensor,
          expected["probes"].as_a,
          0.0_f32,
          "#{name} #{boundary}"
        )
      end

      {
        "rope_cos" => result.rope_cos,
        "rope_sin" => result.rope_sin,
      }.each do |boundary, tensor|
        expected = test_case["expected"][boundary]
        tensor.shape.to_a.should eq(
          expected["shape"].as_a.map(&.as_i.to_i32)
        ), "#{name} #{boundary} shape"
        dino_v3_truncated_1e4le_sha256(tensor.cpu_data.not_nil!).should eq(
          expected["truncated_1e4le_sha256"].as_s
        ), "#{name} #{boundary} truncated bytes"
        dino_v3_assert_probes(
          tensor,
          expected["probes"].as_a,
          2.0e-6_f32,
          "#{name} #{boundary}"
        )
      end

      dino_v3_f32le_sha256(input.cpu_data.not_nil!).should eq(
        input_sha256
      ), "#{name} source immutability"
      dino_v3_parameters_sha256(parameters).should eq(
        parameter_sha256
      ), "#{name} parameter immutability"
    end
  end

  it "matches non-binary Float32 Conv2d probes within the declared tolerance" do
    fixture_path = File.join(
      __DIR__,
      "../../fixtures/trellis2/dino_v3_embeddings_cpu_v1.json"
    )
    test_case = JSON.parse(File.read(fixture_path))["float_stress_case"]
    test_case["name"].as_s.should eq("non_binary_conv2d_512")
    config_json = test_case["config"]
    {
      "patch_size"          => 16_i64,
      "num_channels"        => 3_i64,
      "hidden_size"         => 4_i64,
      "num_attention_heads" => 1_i64,
      "num_register_tokens" => 2_i64,
    }.each do |field, expected|
      config_json[field].as_i.should eq(expected), "stress config #{field}"
    end
    config_json["rope_theta"].as_f.should eq(100.0)
    test_case["input"]["shape"].as_a.map(&.as_i.to_i32).should eq(
      [1, 3, 512, 512]
    )
    test_case["input"]["seed"].as_s.should eq("0x13579bdf")
    test_case["parameters"]["seeds"].as_a.map(&.as_s).should eq(
      ["0x2468ace0", "0x10203040", "0x50607080", "0x90abcdef"]
    )
    config = ML::Vision::DinoV3::EmbeddingConfig.new(
      patch_size: config_json["patch_size"].as_i.to_i32,
      num_channels: config_json["num_channels"].as_i.to_i32,
      hidden_size: config_json["hidden_size"].as_i.to_i32,
      num_attention_heads: config_json["num_attention_heads"].as_i.to_i32,
      num_register_tokens: config_json["num_register_tokens"].as_i.to_i32,
      rope_theta: config_json["rope_theta"].as_f.to_f32
    )
    input_values = dino_v3_lcg_f32(3 * 512 * 512, 0x13579bdf_u32)
    input = ML::Tensor.from_array(
      input_values,
      ML::Shape.new(1_i32, 3_i32, 512_i32, 512_i32)
    )
    dino_v3_f32le_sha256(input_values).should eq(
      test_case["input"]["f32le_sha256"].as_s
    )
    parameters = ML::Vision::DinoV3::EmbeddingParameters.new(
      config,
      dino_v3_lcg_f32(4 * 3 * 16 * 16, 0x2468ace0_u32),
      dino_v3_lcg_f32(4, 0x10203040_u32),
      dino_v3_lcg_f32(4, 0x50607080_u32),
      dino_v3_lcg_f32(2 * 4, 0x90abcdef_u32)
    )
    dino_v3_parameters_sha256(parameters).should eq(
      test_case["parameters"]["f32le_sha256"].as_s
    )
    result = ML::Vision::DinoV3::EmbeddingCPU.new(parameters).forward(input)
    result.parameter_f32le_sha256.should eq(
      test_case["parameters"]["f32le_sha256"].as_s
    )
    tolerance = test_case["expected"]["absolute_tolerance"].as_f.to_f32
    tolerance.should eq(5.0e-5_f32)
    expected_probe_indices = {
      "patches" => [
        [0, 0, 0],
        [0, 0, 3],
        [0, 1, 1],
        [0, 31, 2],
        [0, 32, 3],
        [0, 255, 0],
        [0, 511, 1],
        [0, 1023, 3],
      ],
      "embeddings" => [
        [0, 0, 0],
        [0, 1, 0],
        [0, 2, 3],
        [0, 3, 0],
        [0, 4, 1],
        [0, 1026, 3],
      ],
    }
    {
      "patches"    => result.patches,
      "embeddings" => result.embeddings,
    }.each do |boundary, tensor|
      expected = test_case["expected"][boundary]
      tensor.shape.to_a.should eq(expected["shape"].as_a.map(&.as_i.to_i32))
      expected["probes"].as_a.map do |probe|
        probe["index"].as_a.map(&.as_i.to_i32)
      end.should eq(expected_probe_indices[boundary])
      dino_v3_assert_probes(
        tensor,
        expected["probes"].as_a,
        tolerance,
        "float stress #{boundary}"
      )
    end
  end

  it "fails closed on invalid configuration, buffers, geometry, and model-scale CPU work" do
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /patch size must be 16/) do
      ML::Vision::DinoV3::EmbeddingConfig.new(
        patch_size: 14,
        num_channels: 3,
        hidden_size: 16,
        num_attention_heads: 2,
        num_register_tokens: 4,
        rope_theta: 100.0_f32
      )
    end
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /power of two/) do
      ML::Vision::DinoV3::EmbeddingConfig.new(
        patch_size: 16,
        num_channels: 3,
        hidden_size: 12,
        num_attention_heads: 2,
        num_register_tokens: 4,
        rope_theta: 100.0_f32
      )
    end
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /theta must be finite/) do
      ML::Vision::DinoV3::EmbeddingConfig.new(
        patch_size: 16,
        num_channels: 3,
        hidden_size: 16,
        num_attention_heads: 2,
        num_register_tokens: 4,
        rope_theta: Float32::NAN
      )
    end

    config = ML::Vision::DinoV3::EmbeddingConfig.new(
      patch_size: 16,
      num_channels: 3,
      hidden_size: 16,
      num_attention_heads: 2,
      num_register_tokens: 4,
      rope_theta: 100.0_f32
    )
    parameters = dino_v3_parameters(config)
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /patch weight/) do
      ML::Vision::DinoV3::EmbeddingParameters.new(
        config,
        parameters.patch_weight[0...-1],
        parameters.patch_bias,
        parameters.cls_token,
        parameters.register_tokens
      )
    end

    nonfinite_bias = parameters.patch_bias.dup
    nonfinite_bias[0] = Float32::NAN
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /must be finite/) do
      ML::Vision::DinoV3::EmbeddingParameters.new(
        config,
        parameters.patch_weight,
        nonfinite_bias,
        parameters.cls_token,
        parameters.register_tokens
      )
    end
    original_bias = parameters.patch_bias[0]
    parameters.patch_bias[0] = Float32::NAN
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /must be finite/) do
      parameters.validate!
    end
    parameters.patch_bias[0] = original_bias

    embedding = ML::Vision::DinoV3::EmbeddingCPU.new(parameters)
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /512 or 1024/) do
      embedding.forward(ML::Tensor.zeros(1, 3, 256, 256, device: ML::Tensor::Device::CPU))
    end
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /rank 4/) do
      embedding.forward(
        ML::Tensor.zeros(3, 512, 512, device: ML::Tensor::Device::CPU)
      )
    end
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /batch size 1/) do
      embedding.forward(
        ML::Tensor.zeros(2, 3, 256, 256, device: ML::Tensor::Device::CPU)
      )
    end
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /must be square/) do
      embedding.forward(
        ML::Tensor.zeros(1, 3, 512, 256, device: ML::Tensor::Device::CPU)
      )
    end
    nonfinite_input = ML::Tensor.zeros(
      1,
      3,
      512,
      512,
      device: ML::Tensor::Device::CPU
    )
    nonfinite_input.cpu_data.not_nil![0] = Float32::NAN
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /must be contiguous/) do
      embedding.forward(nonfinite_input.transpose)
    end
    expect_raises(ML::Vision::DinoV3::EmbeddingError, /input\[0\] must be finite/) do
      embedding.forward(nonfinite_input)
    end

    model_config = ML::Vision::DinoV3::EmbeddingConfig.new(
      patch_size: 16,
      num_channels: 3,
      hidden_size: 1024,
      num_attention_heads: 16,
      num_register_tokens: 4,
      rope_theta: 100.0_f32
    )
    model_parameters = ML::Vision::DinoV3::EmbeddingParameters.new(
      model_config,
      Array(Float32).new(1024 * 3 * 16 * 16, 0.0_f32),
      Array(Float32).new(1024, 0.0_f32),
      Array(Float32).new(1024, 0.0_f32),
      Array(Float32).new(4 * 1024, 0.0_f32)
    )
    model_embedding = ML::Vision::DinoV3::EmbeddingCPU.new(model_parameters)
    expect_raises(ML::Vision::DinoV3::EmbeddingBudgetError, /multiply-add/) do
      model_embedding.forward(
        ML::Tensor.zeros(1, 3, 512, 512, device: ML::Tensor::Device::CPU)
      )
    end
  end

  it "derives prefix length from the configuration instead of assuming four registers" do
    config = ML::Vision::DinoV3::EmbeddingConfig.new(
      patch_size: 16,
      num_channels: 3,
      hidden_size: 4,
      num_attention_heads: 1,
      num_register_tokens: 0,
      rope_theta: 100.0_f32
    )
    parameters = dino_v3_parameters(config)
    result = ML::Vision::DinoV3::EmbeddingCPU.new(parameters).forward(
      dino_v3_input(512)
    )
    result.patches.shape.to_a.should eq([1, 1024, 4])
    result.embeddings.shape.to_a.should eq([1, 1025, 4])
    4.times do |feature|
      result.embeddings[0, 0, feature].should eq(parameters.cls_token[feature])
      result.embeddings[0, 1, feature].should eq(result.patches[0, 0, feature])
    end
  end
end
