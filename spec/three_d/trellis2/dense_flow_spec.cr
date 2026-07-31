require "json"
require "../../spec_helper"
require "../../../src/ml/gguf/safetensors"

private def dense_flow_tensor(
  tensors : ML::GGUF::SafetensorsFile,
  payload : JSON::Any,
) : ML::Tensor
  name = payload["tensor"].as_s
  info = tensors.tensor(name) || raise "missing dense-flow oracle tensor #{name}"
  info.dtype.should eq(ML::GGUF::SafeTensorDType::F32), "#{name} dtype"
  info.shape.should eq(payload["shape"].as_a.map(&.as_i64)), "#{name} manifest shape"
  ML::Tensor.from_array(
    tensors.read_tensor_f32(info),
    ML::Shape.new(info.shape.map(&.to_i32))
  )
end

private def dense_flow_load_linear(
  linear : ML::NN::Linear,
  tensors : ML::GGUF::SafetensorsFile,
  parameters : Hash(String, JSON::Any),
  prefix : String,
) : Nil
  linear.weight.data.cpu_data.not_nil!.replace(
    dense_flow_tensor(tensors, parameters["#{prefix}.weight"]).to_a
  )
  linear.weight.requires_grad = false
  if bias = linear.bias
    bias.data.cpu_data.not_nil!.replace(
      dense_flow_tensor(tensors, parameters["#{prefix}.bias"]).to_a
    )
    bias.requires_grad = false
  end
end

private def dense_flow_load_parameters(
  stage : ML::ThreeD::Trellis2::DenseFlowStageCPU,
  tensors : ML::GGUF::SafetensorsFile,
  parameters : Hash(String, JSON::Any),
) : Nil
  dense_flow_load_linear(stage.input_layer, tensors, parameters, "input_layer")
  dense_flow_load_linear(
    stage.conditioning.timestep.first_linear,
    tensors,
    parameters,
    "timestep.first_linear"
  )
  dense_flow_load_linear(
    stage.conditioning.timestep.second_linear,
    tensors,
    parameters,
    "timestep.second_linear"
  )
  dense_flow_load_linear(
    stage.conditioning.modulation.linear,
    tensors,
    parameters,
    "top_modulation.linear"
  )

  block = stage.block
  block.modulation.data.cpu_data.not_nil!.replace(
    dense_flow_tensor(tensors, parameters["block.modulation"]).to_a
  )
  block.norm2.weight.data.cpu_data.not_nil!.replace(
    dense_flow_tensor(tensors, parameters["block.norm2.weight"]).to_a
  )
  block.norm2.bias.data.cpu_data.not_nil!.replace(
    dense_flow_tensor(tensors, parameters["block.norm2.bias"]).to_a
  )
  block.self_attn.q_rms_norm.gamma.data.cpu_data.not_nil!.replace(
    dense_flow_tensor(tensors, parameters["block.self_attn.q_rms_norm.gamma"]).to_a
  )
  block.self_attn.k_rms_norm.gamma.data.cpu_data.not_nil!.replace(
    dense_flow_tensor(tensors, parameters["block.self_attn.k_rms_norm.gamma"]).to_a
  )
  block.cross_attn.q_rms_norm.gamma.data.cpu_data.not_nil!.replace(
    dense_flow_tensor(tensors, parameters["block.cross_attn.q_rms_norm.gamma"]).to_a
  )
  block.cross_attn.k_rms_norm.gamma.data.cpu_data.not_nil!.replace(
    dense_flow_tensor(tensors, parameters["block.cross_attn.k_rms_norm.gamma"]).to_a
  )
  dense_flow_load_linear(block.self_attn.to_qkv, tensors, parameters, "block.self_attn.to_qkv")
  dense_flow_load_linear(block.self_attn.to_out, tensors, parameters, "block.self_attn.to_out")
  dense_flow_load_linear(block.cross_attn.to_q, tensors, parameters, "block.cross_attn.to_q")
  dense_flow_load_linear(block.cross_attn.to_kv, tensors, parameters, "block.cross_attn.to_kv")
  dense_flow_load_linear(block.cross_attn.to_out, tensors, parameters, "block.cross_attn.to_out")
  dense_flow_load_linear(block.mlp.fc1, tensors, parameters, "block.mlp.mlp.0")
  dense_flow_load_linear(block.mlp.fc2, tensors, parameters, "block.mlp.mlp.2")
  dense_flow_load_linear(stage.out_layer, tensors, parameters, "out_layer")
end

private def dense_flow_assert_close(
  actual : ML::Tensor,
  expected : ML::Tensor,
  name : String,
  absolute : Float32,
  relative : Float32,
) : Nil
  actual.shape.should eq(expected.shape), "#{name} shape"
  max_ratio = 0.0_f32
  max_abs = 0.0_f32
  max_index = 0
  expected_values = expected.to_a
  actual.to_a.each_with_index do |value, index|
    value.finite?.should be_true, "#{name}[#{index}] must be finite"
    delta = (value - expected_values[index]).abs
    limit = absolute + relative * expected_values[index].abs
    ratio = delta / Math.max(limit, 1e-30_f32)
    if ratio > max_ratio
      max_ratio = ratio
      max_abs = delta
      max_index = index
    end
  end
  max_ratio.should be <= 1.0_f32,
    "#{name} max_abs=#{max_abs} max_ratio=#{max_ratio} index=#{max_index} " \
    "actual=#{actual.to_a[max_index]} expected=#{expected_values[max_index]}"
end

private def dense_flow_stage(config : JSON::Any) : ML::ThreeD::Trellis2::DenseFlowStageCPU
  ML::ThreeD::Trellis2::DenseFlowStageCPU.new(
    resolution: config["resolution"].as_i.to_i32,
    in_channels: config["in_channels"].as_i.to_i32,
    model_channels: config["model_channels"].as_i.to_i32,
    context_channels: config["context_channels"].as_i.to_i32,
    out_channels: config["out_channels"].as_i.to_i32,
    num_heads: config["num_heads"].as_i.to_i32,
    mlp_ratio: config["mlp_ratio"].as_f.to_f32,
    frequency_dim: config["frequency_dim"].as_i.to_i32,
    block_eps: config["block_eps"].as_f.to_f32,
    final_eps: config["final_eps"].as_f.to_f32,
    device: ML::Tensor::Device::CPU
  )
end

describe ML::ThreeD::Trellis2::DenseFlowStageCPU do
  it "matches one unified pinned dense-flow forward at every named boundary" do
    fixture_path = File.join(__DIR__, "../../fixtures/trellis2/dense_flow_cpu_v1.json")
    fixture = JSON.parse(File.read(fixture_path))
    fixture["schema"].as_s.should eq("cogni-ml/trellis2/dense-flow-oracle/v1")
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    config = fixture["config"]
    config["device"].as_s.should eq("cpu")
    config["dtype"].as_s.should eq("float32")
    config["num_blocks"].as_i.should eq(1)
    config["pe_mode"].as_s.should eq("rope")
    config["share_mod"].as_bool.should be_true
    config["qk_rms_norm"].as_bool.should be_true
    config["qk_rms_norm_cross"].as_bool.should be_true
    config["context_length"].as_i.should_not eq(config["voxel_count"].as_i)
    config["in_channels"].as_i.should_not eq(config["out_channels"].as_i)

    tensors = ML::GGUF::SafetensorsFile.new(
      File.join(File.dirname(fixture_path), fixture["tensor_file"].as_s)
    )
    begin
      oracle_metadata = JSON.parse(tensors.metadata["oracle"])
      oracle_metadata["upstream_commit"].as_s.should eq(fixture["provenance"]["commit"].as_s)
      oracle_metadata["schema"].as_s.should eq(fixture["schema"].as_s)
      referenced = [] of String
      {"inputs", "parameters", "expected"}.each do |section|
        fixture[section].as_h.each_value { |payload| referenced << payload["tensor"].as_s }
      end
      referenced.sort.should eq(tensors.tensors.map(&.name).sort)

      stage = dense_flow_stage(config)
      dense_flow_load_parameters(stage, tensors, fixture["parameters"].as_h)
      inputs = fixture["inputs"]
      trace = stage.forward_with_trace(
        dense_flow_tensor(tensors, inputs["voxels"]),
        dense_flow_tensor(tensors, inputs["timesteps"]),
        dense_flow_tensor(tensors, inputs["context"])
      )
      trace["conditioning.mod"].to_a.should_not eq(trace["block.combined_mod"].to_a)
      absolute = fixture["tolerance"]["absolute"].as_f.to_f32
      relative = fixture["tolerance"]["relative"].as_f.to_f32
      expected = fixture["expected"].as_h
      trace.keys.sort.should eq(expected.keys.sort)
      expected.each do |name, payload|
        dense_flow_assert_close(
          trace[name],
          dense_flow_tensor(tensors, payload),
          name,
          absolute,
          relative
        )
      end
    ensure
      tensors.close
    end
  end

  it "preserves ij coordinates and the exact NCDHW-token-NCDHW inverse layout" do
    fixture_path = File.join(__DIR__, "../../fixtures/trellis2/dense_flow_cpu_v1.json")
    fixture = JSON.parse(File.read(fixture_path))
    config = fixture["config"]
    tensors = ML::GGUF::SafetensorsFile.new(
      File.join(File.dirname(fixture_path), fixture["tensor_file"].as_s)
    )
    begin
      stage = dense_flow_stage(config)
      dense_flow_load_parameters(stage, tensors, fixture["parameters"].as_h)
      inputs = fixture["inputs"]
      voxels = dense_flow_tensor(tensors, inputs["voxels"])
      trace = stage.forward_with_trace(
        voxels,
        dense_flow_tensor(tensors, inputs["timesteps"]),
        dense_flow_tensor(tensors, inputs["context"])
      )
      trace["coordinates"].to_a.should eq([
        0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
        1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
      ].map(&.to_f32))

      batch = config["batch"].as_i.to_i32
      input_channels = config["in_channels"].as_i.to_i32
      output_channels = config["out_channels"].as_i.to_i32
      voxel_count = config["voxel_count"].as_i.to_i32
      voxel_values = voxels.to_a
      flattened = trace["flattened"].to_a
      batch.times do |b|
        voxel_count.times do |position|
          input_channels.times do |channel|
            flattened[(b * voxel_count + position) * input_channels + channel].should eq(
              voxel_values[(b * input_channels + channel) * voxel_count + position]
            )
          end
        end
      end

      output_tokens = trace["output_tokens"].to_a
      output = trace["output"].to_a
      batch.times do |b|
        voxel_count.times do |position|
          output_channels.times do |channel|
            output[(b * output_channels + channel) * voxel_count + position].should eq(
              output_tokens[(b * voxel_count + position) * output_channels + channel]
            )
          end
        end
      end
    ensure
      tensors.close
    end
  end

  it "rejects unsupported geometry, devices, scale, and logical tensors before math" do
    expect_raises(ArgumentError, /CPU-only/) do
      ML::ThreeD::Trellis2::DenseFlowStageCPU.new(
        resolution: 2, in_channels: 3, model_channels: 16,
        context_channels: 5, out_channels: 4, num_heads: 2,
        device: ML::Tensor::Device::GPU
      )
    end
    expect_raises(ArgumentError, /dense-flow parameter budget/) do
      ML::ThreeD::Trellis2::DenseFlowStageCPU.new(
        resolution: 16, in_channels: 8, model_channels: 1536,
        context_channels: 1024, out_channels: 8, num_heads: 12,
        mlp_ratio: 5.3334_f32, frequency_dim: 256,
        device: ML::Tensor::Device::CPU
      )
    end

    stage = ML::ThreeD::Trellis2::DenseFlowStageCPU.new(
      resolution: 2, in_channels: 3, model_channels: 16,
      context_channels: 5, out_channels: 4, num_heads: 2,
      device: ML::Tensor::Device::CPU
    )
    valid_t = ML::Tensor.zeros(2)
    valid_context = ML::Tensor.zeros(2, 3, 5)
    expect_raises(ArgumentError, /rank 5/) do
      stage.forward(ML::Tensor.zeros(2, 3), valid_t, valid_context)
    end
    expect_raises(ArgumentError, /voxel input/) do
      stage.forward(ML::Tensor.zeros(2, 3, 2, 2, 1), valid_t, valid_context)
    end
    expect_raises(ArgumentError, /timesteps/) do
      stage.forward(ML::Tensor.zeros(2, 3, 2, 2, 2), ML::Tensor.zeros(1), valid_context)
    end
    expect_raises(ArgumentError, /context/) do
      stage.forward(ML::Tensor.zeros(2, 3, 2, 2, 2), valid_t, ML::Tensor.zeros(2, 3, 4))
    end
    expect_raises(ArgumentError, /contiguous/) do
      stage.forward(ML::Tensor.zeros(2, 3, 2, 2, 2).transpose, valid_t, valid_context)
    end
    expect_raises(ArgumentError, /finite/) do
      stage.forward(
        ML::Tensor.from_array(
          [Float32::NAN] + Array(Float32).new(47, 0.0_f32),
          ML::Shape.new(2_i32, 3_i32, 2_i32, 2_i32, 2_i32)
        ),
        valid_t,
        valid_context
      )
    end

    bounded = ML::ThreeD::Trellis2::DenseFlowStageCPU.new(
      resolution: 2, in_channels: 3, model_channels: 16,
      context_channels: 5, out_channels: 4, num_heads: 2,
      max_logical_tensor_bytes: 128_i64,
      device: ML::Tensor::Device::CPU
    )
    expect_raises(ArgumentError, /logical tensor budget/) do
      bounded.forward(
        ML::Tensor.zeros(2, 3, 2, 2, 2),
        valid_t,
        valid_context
      )
    end

    stage.input_layer.weight.data.cpu_data.not_nil![0] = Float32::NAN
    expect_raises(ArgumentError, /dense-flow parameter.*finite/) do
      stage.forward(
        ML::Tensor.zeros(2, 3, 2, 2, 2),
        valid_t,
        valid_context
      )
    end
  end

  it "keeps the complete one-block inference parameter surface graphless" do
    stage = ML::ThreeD::Trellis2::DenseFlowStageCPU.new(
      resolution: 2, in_channels: 3, model_channels: 16,
      context_channels: 5, out_channels: 4, num_heads: 2,
      device: ML::Tensor::Device::CPU
    )
    stage.parameters.should_not be_empty
    stage.parameters.each { |parameter| parameter.requires_grad?.should be_false }
    stage.parameters.sum(0_i64) { |parameter| parameter.data.numel.to_i64 }.should eq(
      stage.parameter_elements
    )
  end
end
