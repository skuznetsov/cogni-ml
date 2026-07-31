require "json"
require "../../spec_helper"
require "../../../src/ml/gguf/safetensors"

private def oracle_tensor(
  tensors : ML::GGUF::SafetensorsFile,
  payload : JSON::Any,
) : ML::Tensor
  name = payload["tensor"].as_s
  info = tensors.tensor(name) || raise "missing oracle tensor #{name}"
  info.dtype.should eq(ML::GGUF::SafeTensorDType::F32), "#{name} dtype"
  declared_shape = payload["shape"].as_a.map(&.as_i64)
  info.shape.should eq(declared_shape), "#{name} manifest shape"
  shape = ML::Shape.new(info.shape.map(&.to_i32))
  ML::Tensor.from_array(tensors.read_tensor_f32(info), shape)
end

private def load_linear(
  linear : ML::NN::Linear,
  tensors : ML::GGUF::SafetensorsFile,
  parameters : Hash(String, JSON::Any),
  prefix : String,
) : Nil
  linear.weight.data.cpu_data.not_nil!.replace(
    oracle_tensor(tensors, parameters["#{prefix}.weight"]).to_a
  )
  linear.weight.requires_grad = false
  if bias = linear.bias
    bias.data.cpu_data.not_nil!.replace(
      oracle_tensor(tensors, parameters["#{prefix}.bias"]).to_a
    )
    bias.requires_grad = false
  end
end

private def assert_close(
  actual : ML::Tensor,
  expected : ML::Tensor,
  name : String,
  absolute : Float32 = 2e-5_f32,
  relative_tolerance : Float32 = 2e-5_f32,
) : Nil
  actual.shape.should eq(expected.shape), "#{name} shape"
  actual_values = actual.to_a
  expected_values = expected.to_a
  max_abs = 0.0_f32
  max_rel = 0.0_f32
  max_ratio = 0.0_f32
  max_index = 0
  actual_values.each_with_index do |value, index|
    value.finite?.should be_true, "#{name}[#{index}] must be finite"
    delta = (value - expected_values[index]).abs
    relative_error = delta / Math.max(expected_values[index].abs, 1e-6_f32)
    limit = absolute + relative_tolerance * expected_values[index].abs
    ratio = delta / Math.max(limit, 1e-30_f32)
    if delta > max_abs
      max_abs = delta
      max_index = index
    end
    max_rel = relative_error if relative_error > max_rel
    max_ratio = ratio if ratio > max_ratio
  end
  max_ratio.should be <= 1.0_f32,
    "#{name} max_abs=#{max_abs} max_rel=#{max_rel} max_ratio=#{max_ratio} index=#{max_index} " \
    "actual=#{actual_values[max_index]} expected=#{expected_values[max_index]}"
end

describe ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock do
  it "matches the pinned stage-representative CPU formula at named boundaries" do
    fixture_path = File.join(
      __DIR__,
      "../../fixtures/trellis2/shared_modulated_cross_block_cpu_v1.json"
    )
    fixture = JSON.parse(File.read(fixture_path))
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/shared-modulated-cross-block-oracle/v1"
    )
    fixture["provenance"]["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    config = fixture["config"]
    parameters = fixture["parameters"].as_h
    absolute = fixture["tolerance"]["absolute"].as_f.to_f32
    relative = fixture["tolerance"]["relative"].as_f.to_f32
    config["device"].as_s.should eq("cpu")
    config["dtype"].as_s.should eq("float32")
    config["share_mod"].as_bool.should be_true
    config["use_rope"].as_bool.should be_true
    config["qk_rms_norm"].as_bool.should be_true
    config["qk_rms_norm_cross"].as_bool.should be_true
    tensors = ML::GGUF::SafetensorsFile.new(
      File.join(File.dirname(fixture_path), fixture["tensor_file"].as_s)
    )

    begin
      tensors.metadata["upstream_commit"]?.should be_nil
      JSON.parse(tensors.metadata["oracle"])["upstream_commit"].as_s.should eq(
        fixture["provenance"]["commit"].as_s
      )
      referenced_names = [] of String
      {"inputs", "parameters", "expected"}.each do |section|
        fixture[section].as_h.each_value do |payload|
          referenced_names << payload["tensor"].as_s
        end
      end
      referenced_names.sort.should eq(tensors.tensors.map(&.name).sort)

      block = ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
        channels: config["channels"].as_i.to_i32,
        context_channels: config["context_channels"].as_i.to_i32,
        num_heads: config["heads"].as_i.to_i32,
        mlp_ratio: config["mlp_ratio"].as_f.to_f32,
        eps: config["eps"].as_f.to_f32,
        device: ML::Tensor::Device::CPU
      )

      block.modulation.data.cpu_data.not_nil!.replace(
        oracle_tensor(tensors, parameters["modulation"]).to_a
      )
      block.norm2.weight.data.cpu_data.not_nil!.replace(
        oracle_tensor(tensors, parameters["norm2.weight"]).to_a
      )
      block.norm2.bias.data.cpu_data.not_nil!.replace(
        oracle_tensor(tensors, parameters["norm2.bias"]).to_a
      )
      block.self_attn.q_rms_norm.gamma.data.cpu_data.not_nil!.replace(
        oracle_tensor(tensors, parameters["self_attn.q_rms_norm.gamma"]).to_a
      )
      block.self_attn.k_rms_norm.gamma.data.cpu_data.not_nil!.replace(
        oracle_tensor(tensors, parameters["self_attn.k_rms_norm.gamma"]).to_a
      )
      block.cross_attn.q_rms_norm.gamma.data.cpu_data.not_nil!.replace(
        oracle_tensor(tensors, parameters["cross_attn.q_rms_norm.gamma"]).to_a
      )
      block.cross_attn.k_rms_norm.gamma.data.cpu_data.not_nil!.replace(
        oracle_tensor(tensors, parameters["cross_attn.k_rms_norm.gamma"]).to_a
      )

      load_linear(block.self_attn.to_qkv, tensors, parameters, "self_attn.to_qkv")
      load_linear(block.self_attn.to_out, tensors, parameters, "self_attn.to_out")
      load_linear(block.cross_attn.to_q, tensors, parameters, "cross_attn.to_q")
      load_linear(block.cross_attn.to_kv, tensors, parameters, "cross_attn.to_kv")
      load_linear(block.cross_attn.to_out, tensors, parameters, "cross_attn.to_out")
      load_linear(block.mlp.fc1, tensors, parameters, "mlp.mlp.0")
      load_linear(block.mlp.fc2, tensors, parameters, "mlp.mlp.2")

      inputs = fixture["inputs"]
      inputs["x"]["shape"].as_a.map(&.as_i).should eq([
        config["batch"].as_i,
        config["length"].as_i,
        config["channels"].as_i,
      ])
      inputs["mod"]["shape"].as_a.map(&.as_i).should eq([
        config["batch"].as_i,
        config["channels"].as_i * 6,
      ])
      inputs["context"]["shape"].as_a.map(&.as_i).should eq([
        config["batch"].as_i,
        config["context_length"].as_i,
        config["context_channels"].as_i,
      ])
      phases = oracle_tensor(tensors, inputs["phases"])
      head_dim = config["channels"].as_i.to_i32 // config["heads"].as_i.to_i32
      phases.shape.should eq(
        ML::Shape.new(
          config["length"].as_i.to_i32,
          head_dim // 2,
          2_i32
        )
      )
      phase_values = phases.to_a
      config["length"].as_i.to_i32.times do |position|
        last_pair = (position * (head_dim // 2) + (head_dim // 2 - 1)) * 2
        phase_values[last_pair].should eq(1.0_f32)
        phase_values[last_pair + 1].should eq(0.0_f32)
      end
      trace = block.forward_with_trace(
        ML::Autograd::Variable.new(oracle_tensor(tensors, inputs["x"]), requires_grad: false),
        ML::Autograd::Variable.new(oracle_tensor(tensors, inputs["mod"]), requires_grad: false),
        ML::Autograd::Variable.new(oracle_tensor(tensors, inputs["context"]), requires_grad: false),
        phases
      )

      expected = fixture["expected"].as_h
      expected.each do |name, payload|
        actual = trace[name]? || raise "missing trace tensor #{name}"
        assert_close(
          actual,
          oracle_tensor(tensors, payload),
          name,
          absolute,
          relative
        )
      end
    ensure
      tensors.close
    end
  end

  it "rejects unverified modes and malformed geometry before projection" do
    expect_raises(ArgumentError, /requires qk_rms_norm=true/) do
      ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
        channels: 6,
        context_channels: 4,
        num_heads: 1,
        qk_rms_norm: false,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /requires use_rope=true/) do
      ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
        channels: 6,
        context_channels: 4,
        num_heads: 1,
        use_rope: false,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /requires qk_rms_norm_cross=true/) do
      ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
        channels: 6,
        context_channels: 4,
        num_heads: 1,
        qk_rms_norm_cross: false,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /only shared modulation/) do
      ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
        channels: 6,
        context_channels: 4,
        num_heads: 1,
        share_mod: false,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /overflow shared modulation width/) do
      ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
        channels: Int32::MAX,
        context_channels: 1,
        num_heads: 1,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /dense CPU oracle parameter budget/) do
      ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
        channels: 4096,
        context_channels: 4096,
        num_heads: 32,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /dense CPU oracle parameter budget/) do
      ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
        channels: 232_000_000,
        context_channels: Int32::MAX,
        num_heads: 1,
        mlp_ratio: 9.0_f32,
        device: ML::Tensor::Device::CPU
      )
    end

    block = ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
      channels: 6,
      context_channels: 4,
      num_heads: 1,
      device: ML::Tensor::Device::CPU
    )
    input = ML::Autograd::Variable.new(
      ML::Tensor.zeros(1, 2, 6),
      requires_grad: false
    )
    context = ML::Autograd::Variable.new(
      ML::Tensor.zeros(1, 3, 4),
      requires_grad: false
    )
    phases = ML::Tensor.from_array(
      Array(Float32).new(2 * 3 * 2) { |index| index.even? ? 1.0_f32 : 0.0_f32 },
      ML::Shape.new(2_i32, 3_i32, 2_i32)
    )

    expect_raises(ArgumentError, /block modulation/) do
      block.forward_with_trace(
        input,
        ML::Autograd::Variable.new(ML::Tensor.zeros(1, 35), requires_grad: false),
        context,
        phases
      )
    end
    expect_raises(ArgumentError, /block context/) do
      block.forward_with_trace(
        input,
        ML::Autograd::Variable.new(ML::Tensor.zeros(1, 36), requires_grad: false),
        ML::Autograd::Variable.new(ML::Tensor.zeros(1, 3, 5), requires_grad: false),
        phases
      )
    end
    expect_raises(ArgumentError, /RoPE phases are required/) do
      block.forward_with_trace(
        input,
        ML::Autograd::Variable.new(ML::Tensor.zeros(1, 36), requires_grad: false),
        context
      )
    end
  end

  it "keeps the inference-only parameter surface graphless" do
    block = ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
      channels: 6,
      context_channels: 4,
      num_heads: 1,
      device: ML::Tensor::Device::CPU
    )
    parameters = [
      block.modulation,
      block.norm2.weight,
      block.norm2.bias,
      block.self_attn.q_rms_norm.gamma,
      block.self_attn.k_rms_norm.gamma,
      block.cross_attn.q_rms_norm.gamma,
      block.cross_attn.k_rms_norm.gamma,
    ]
    [
      block.self_attn.to_qkv,
      block.self_attn.to_out,
      block.cross_attn.to_q,
      block.cross_attn.to_kv,
      block.cross_attn.to_out,
      block.mlp.fc1,
      block.mlp.fc2,
    ].each { |linear| parameters.concat(linear.parameters) }

    parameters.each { |parameter| parameter.requires_grad?.should be_false }
  end

  it "reads shape-preserving strided inputs in logical order" do
    block = ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
      channels: 6,
      context_channels: 4,
      num_heads: 1,
      device: ML::Tensor::Device::CPU
    )
    input_base = ML::Tensor.from_array(
      Array(Float32).new(36) { |index| -0.4_f32 + index.to_f32 * 0.031_f32 },
      ML::Shape.new(1_i32, 6_i32, 6_i32)
    )
    context_base = ML::Tensor.from_array(
      Array(Float32).new(16) { |index| 0.3_f32 - index.to_f32 * 0.027_f32 },
      ML::Shape.new(1_i32, 4_i32, 4_i32)
    )
    input_view = input_base.transpose
    context_view = context_base.transpose
    modulation = ML::Autograd::Variable.new(
      ML::Tensor.from_array(
        Array(Float32).new(36) { |index| 0.02_f32 - index.to_f32 * 0.001_f32 },
        ML::Shape.new(1_i32, 36_i32)
      ),
      requires_grad: false
    )
    phases = ML::Tensor.from_array(
      Array(Float32).new(6 * 3 * 2) do |index|
        index % 2 == 0 ? 1.0_f32 : 0.0_f32
      end,
      ML::Shape.new(6_i32, 3_i32, 2_i32)
    )
    input_before = input_base.to_a
    context_before = context_base.to_a

    strided = block.forward_with_trace(
      ML::Autograd::Variable.new(input_view, requires_grad: false),
      modulation,
      ML::Autograd::Variable.new(context_view, requires_grad: false),
      phases
    )["output"]
    dense = block.forward_with_trace(
      ML::Autograd::Variable.new(input_view.contiguous, requires_grad: false),
      modulation,
      ML::Autograd::Variable.new(context_view.contiguous, requires_grad: false),
      phases
    )["output"]

    assert_close(strided, dense, "strided output", 1e-6_f32, 1e-6_f32)
    input_base.to_a.should eq(input_before)
    context_base.to_a.should eq(context_before)
  end
end
