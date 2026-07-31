require "json"
require "../../../src/ml/three_d/trellis2/conditioning"
require "../../../src/ml/gguf/safetensors"
require "../../spec_helper"

private def conditioning_tensor(
  tensors : ML::GGUF::SafetensorsFile,
  payload : JSON::Any,
) : ML::Tensor
  name = payload["tensor"].as_s
  info = tensors.tensor(name) || raise "missing conditioning tensor #{name}"
  info.dtype.should eq(ML::GGUF::SafeTensorDType::F32), "#{name} dtype"
  declared_shape = payload["shape"].as_a.map(&.as_i64)
  info.shape.should eq(declared_shape), "#{name} manifest shape"
  ML::Tensor.from_array(
    tensors.read_tensor_f32(info),
    ML::Shape.new(info.shape.map(&.to_i32))
  )
end

private def conditioning_load_linear(
  linear : ML::NN::Linear,
  tensors : ML::GGUF::SafetensorsFile,
  parameters : Hash(String, JSON::Any),
  prefix : String,
) : Nil
  weight = conditioning_tensor(tensors, parameters["#{prefix}.weight"])
  linear.weight.data.cpu_data.not_nil!.replace(weight.to_a)
  linear.weight.requires_grad = false
  if bias = linear.bias
    value = conditioning_tensor(tensors, parameters["#{prefix}.bias"])
    bias.data.cpu_data.not_nil!.replace(value.to_a)
    bias.requires_grad = false
  end
end

private def conditioning_assert_close(
  actual : ML::Tensor,
  expected : ML::Tensor,
  name : String,
  absolute : Float32,
  relative : Float32,
) : Nil
  actual.shape.should eq(expected.shape), "#{name} shape"
  actual_values = actual.to_a
  expected_values = expected.to_a
  actual_values.each_with_index do |value, index|
    value.finite?.should be_true, "#{name}[#{index}] must be finite"
    delta = (value - expected_values[index]).abs
    limit = absolute + relative * expected_values[index].abs
    delta.should be <= limit,
      "#{name}[#{index}] actual=#{value} expected=#{expected_values[index]} delta=#{delta} limit=#{limit}"
  end
end

describe ML::ThreeD::Trellis2::TimestepConditioningCPU do
  it "matches the pinned CPU formula at named boundaries" do
    fixture_path = File.join(__DIR__, "../../fixtures/trellis2/conditioning_cpu_v1.json")
    fixture = JSON.parse(File.read(fixture_path))
    fixture["schema"].as_s.should eq("cogni-ml/trellis2/conditioning-oracle/v1")
    fixture["provenance"]["commit"].as_s.should eq("75fbf0183001ed9876c8dbb35de6b68552ee08bd")
    config = fixture["config"]
    parameters = fixture["parameters"].as_h
    absolute = fixture["tolerance"]["absolute"].as_f.to_f32
    relative = fixture["tolerance"]["relative"].as_f.to_f32
    config["device"].as_s.should eq("cpu")
    config["dtype"].as_s.should eq("float32")
    config["frequency_dim"].as_i.should eq(256)
    config["head_dim"].as_i.should eq(128)
    config["spatial_dim"].as_i.should eq(3)
    config["max_period"].as_f.should eq(10000.0)
    config["rope_low"].as_f.should eq(1.0)
    config["rope_high"].as_f.should eq(10000.0)
    config["coordinate_count"].as_i.should_not eq(config["batch"].as_i)

    tensors = ML::GGUF::SafetensorsFile.new(
      File.join(File.dirname(fixture_path), fixture["tensor_file"].as_s)
    )
    begin
      oracle_metadata = JSON.parse(tensors.metadata["oracle"])
      oracle_metadata["upstream_commit"].as_s.should eq("75fbf0183001ed9876c8dbb35de6b68552ee08bd")
      oracle_metadata["schema"].as_s.should eq(fixture["schema"].as_s)
      referenced_names = [] of String
      {"inputs", "parameters", "expected"}.each do |section|
        fixture[section].as_h.each_value do |payload|
          referenced_names << payload["tensor"].as_s
        end
      end
      referenced_names.sort.should eq(tensors.tensors.map(&.name).sort)
      condition = ML::ThreeD::Trellis2::TimestepConditioningCPU.new(
        channels: config["channels"].as_i.to_i32,
        rotary_head_dim: config["head_dim"].as_i.to_i32,
        frequency_dim: config["frequency_dim"].as_i.to_i32,
        max_period: config["max_period"].as_f.to_f32,
        spatial_dim: config["spatial_dim"].as_i.to_i32,
        rope_low: config["rope_low"].as_f.to_f32,
        rope_high: config["rope_high"].as_f.to_f32,
        device: ML::Tensor::Device::CPU
      )
      conditioning_load_linear(
        condition.timestep.first_linear,
        tensors,
        parameters,
        "timestep.first_linear"
      )
      conditioning_load_linear(
        condition.timestep.second_linear,
        tensors,
        parameters,
        "timestep.second_linear"
      )
      conditioning_load_linear(
        condition.modulation.linear,
        tensors,
        parameters,
        "modulation.linear"
      )

      inputs = fixture["inputs"]
      timesteps = conditioning_tensor(tensors, inputs["timesteps"])
      coordinates = conditioning_tensor(tensors, inputs["coordinates"])
      trace = condition.forward_with_trace(timesteps)
      expected = fixture["expected"].as_h
      ["t_freq", "first_linear", "first_silu", "t_emb", "top_silu", "mod"].each do |name|
        conditioning_assert_close(
          trace[name],
          conditioning_tensor(tensors, expected[name]),
          name,
          absolute,
          relative
        )
      end

      phases_trace = condition.rotary.forward_with_trace(coordinates)
      ["frequencies", "angles", "phases"].each do |name|
        conditioning_assert_close(
          phases_trace[name],
          conditioning_tensor(tensors, expected[name]),
          name,
          absolute,
          relative
        )
      end
      trace["mod"].shape.should eq(
        ML::Shape.new(config["batch"].as_i.to_i32, 6_i32 * config["channels"].as_i.to_i32)
      )
      phases_trace["phases"].shape.should eq(
        ML::Shape.new(
          config["coordinate_count"].as_i.to_i32,
          config["head_dim"].as_i.to_i32 // 2,
          2_i32
        )
      )
      phases_trace["frequencies"].shape.should eq(ML::Shape.new(21_i32))
      angles = phases_trace["angles"].to_a
      angle_width = 3 * 21
      angles[angle_width].should eq(1.0_f32)
      angles[angle_width + 21].should eq(2.0_f32)
      angles[angle_width + 42].should eq(3.0_f32)
      phases = phases_trace["phases"].to_a
      config["coordinate_count"].as_i.to_i32.times do |position|
        last_pair = (position * 64 + 63) * 2
        phases[last_pair].should eq(1.0_f32)
        phases[last_pair + 1].should eq(0.0_f32)
      end
    ensure
      tensors.close
    end
  end

  it "supports odd frequency dimensions with exact zero padding" do
    condition = ML::ThreeD::Trellis2::TimestepEmbedderCPU.new(
      channels: 3,
      frequency_dim: 5,
      device: ML::Tensor::Device::CPU
    )
    condition.first_linear.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    condition.first_linear.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    condition.second_linear.weight.data.cpu_data.not_nil!.fill(0.0_f32)
    condition.second_linear.bias.not_nil!.data.cpu_data.not_nil!.fill(0.0_f32)
    trace = condition.forward_with_trace(
      ML::Tensor.from_array([0.25_f32, -0.75_f32], ML::Shape.new(2_i32))
    )
    trace["t_freq"].shape.should eq(ML::Shape.new(2_i32, 5_i32))
    values = trace["t_freq"].to_a
    values[4].should eq(0.0_f32)
    values[9].should eq(0.0_f32)
  end

  it "rejects malformed and non-finite inputs before math" do
    condition = ML::ThreeD::Trellis2::TimestepConditioningCPU.new(
      channels: 4,
      rotary_head_dim: 8,
      device: ML::Tensor::Device::CPU
    )
    expect_raises(ArgumentError, /rank-1/) do
      condition.forward(ML::Tensor.zeros(2, 1))
    end
    expect_raises(ArgumentError, /finite/) do
      condition.forward(ML::Tensor.from_array([Float32::NAN], ML::Shape.new(1_i32)))
    end
    expect_raises(ArgumentError, /rank-2/) do
      condition.rotary.forward(ML::Tensor.zeros(2))
    end
    expect_raises(ArgumentError, /coordinate width/) do
      condition.rotary.forward(ML::Tensor.zeros(2, 2))
    end
    expect_raises(ArgumentError, /finite/) do
      condition.rotary.forward(
        ML::Tensor.from_array(
          [0.0_f32, Float32::INFINITY, 0.0_f32],
          ML::Shape.new(1_i32, 3_i32)
        )
      )
    end
    expect_raises(ArgumentError, /at least one coordinate/) do
      condition.rotary.forward(ML::Tensor.new(0, 3))
    end
  end

  it "rejects unsupported devices and model-scale budgets before allocation" do
    expect_raises(ArgumentError, /CPU-only/) do
      ML::ThreeD::Trellis2::TimestepConditioningCPU.new(
        channels: 4,
        rotary_head_dim: 8,
        device: ML::Tensor::Device::GPU
      )
    end
    expect_raises(ArgumentError, /parameter budget/) do
      ML::ThreeD::Trellis2::TimestepConditioningCPU.new(
        channels: 1536,
        rotary_head_dim: 128,
        frequency_dim: 256,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /parameter arithmetic overflow|parameter budget/) do
      ML::ThreeD::Trellis2::TimestepConditioningCPU.new(
        channels: Int32::MAX,
        rotary_head_dim: 128,
        frequency_dim: Int32::MAX,
        device: ML::Tensor::Device::CPU
      )
    end
    expect_raises(ArgumentError, /phase output budget/) do
      ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU.new(
        head_dim: 128,
        dim: 3,
        max_output_bytes: 128_i64
      ).forward(ML::Tensor.zeros(5, 3))
    end
    expect_raises(ArgumentError, /no greater than/) do
      ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU.new(
        head_dim: 128,
        max_output_bytes: ML::ThreeD::Trellis2::ConditioningCPU::MAX_REFERENCE_TENSOR_BYTES + 1_i64
      )
    end
    expect_raises(ArgumentError, /even/) do
      ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU.new(head_dim: 7)
    end
    expect_raises(ArgumentError, /frequency dimension must be positive/) do
      ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU.new(head_dim: 4, dim: 3)
    end
    expect_raises(ArgumentError, /output budget/) do
      ML::ThreeD::Trellis2::ConditioningCPU.ensure_tensor_budget!(
        Int32::MAX,
        Int32::MAX,
        "hostile activation"
      )
    end
    expect_raises(ArgumentError, /finite and positive/) do
      ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU.new(
        head_dim: 8,
        rope_high: Float32::NAN
      )
    end
    expect_raises(ArgumentError, /representable in F32/) do
      ML::ThreeD::Trellis2::TimestepEmbedderCPU.new(
        channels: 2,
        frequency_dim: 256,
        max_period: 1e-40_f32
      )
    end
    expect_raises(ArgumentError, /representable in F32/) do
      ML::ThreeD::Trellis2::RotaryPositionEmbedderCPU.new(
        head_dim: 128,
        rope_low: 1e30_f32,
        rope_high: 1e-30_f32
      )
    end
  end

  it "keeps parameters graphless and reads strided coordinates logically" do
    condition = ML::ThreeD::Trellis2::TimestepConditioningCPU.new(
      channels: 4,
      rotary_head_dim: 8,
      device: ML::Tensor::Device::CPU
    )
    parameters = condition.parameters
    parameters.each { |parameter| parameter.requires_grad?.should be_false }
    base = ML::Tensor.from_array(
      Array(Float32).new(15) { |index| (index.to_f32 - 5.0_f32) * 0.17_f32 },
      ML::Shape.new(3_i32, 5_i32)
    )
    strided = condition.rotary.forward(base.transpose)
    dense = condition.rotary.forward(base.transpose.contiguous)
    conditioning_assert_close(strided, dense, "strided phases", 1e-6_f32, 1e-6_f32)
  end

  it "bridges generated conditioning into the admitted dense block shapes" do
    condition = ML::ThreeD::Trellis2::TimestepConditioningCPU.new(
      channels: 128,
      rotary_head_dim: 128,
      device: ML::Tensor::Device::CPU
    )
    timesteps = ML::Tensor.from_array([0.0_f32, 0.5_f32], ML::Shape.new(2_i32))
    coordinates = ML::Tensor.from_array(
      [
        0.0_f32, 0.0_f32, 0.0_f32,
        1.0_f32, 2.0_f32, 3.0_f32,
        4.0_f32, 5.0_f32, 6.0_f32,
      ],
      ML::Shape.new(3_i32, 3_i32)
    )
    modulation = ML::Autograd::Variable.new(condition.forward(timesteps), requires_grad: false)
    phases = condition.rotary.forward(coordinates)
    block = ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
      channels: 128,
      context_channels: 5,
      num_heads: 1,
      mlp_ratio: 1.0_f32,
      device: ML::Tensor::Device::CPU
    )
    output = block.forward(
      ML::Autograd::Variable.new(ML::Tensor.zeros(2, 3, 128), requires_grad: false),
      modulation,
      ML::Autograd::Variable.new(ML::Tensor.zeros(2, 2, 5), requires_grad: false),
      phases
    ).data

    output.shape.should eq(ML::Shape.new(2_i32, 3_i32, 128_i32))
    output.to_a.all?(&.finite?).should be_true
  end
end
