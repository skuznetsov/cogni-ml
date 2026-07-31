require "../../spec_helper"
require "../../../src/ml/three_d/trellis2/padded_dense_flow"

private def padded_stage_profile(
  voxel_buckets : Array(Int32) = [8_i32, 64_i32],
  context_buckets : Array(Int32) = [3_i32, 4_i32],
) : ML::ThreeD::Trellis2::DenseStageResourceProfile
  ML::ThreeD::Trellis2::DenseStageResourceProfile.new(
    id: "padded-dense-flow-tiny",
    in_channels: 3,
    model_channels: 16,
    context_channels: 5,
    out_channels: 4,
    num_heads: 2,
    mlp_hidden_channels: 64,
    frequency_dim: 256,
    batch_buckets: [1_i32],
    voxel_buckets: voxel_buckets,
    context_buckets: context_buckets
  )
end

private def padded_stage_contract(
  voxel_buckets : Array(Int32) = [8_i32, 64_i32],
  context_buckets : Array(Int32) = [3_i32, 4_i32],
  cache_owner : String = "padded-stage-owner",
  source_digest : String = "1" * 64,
  device_family : String = "cpu-oracle",
  compiler_abi : String = "t2n2d0-v1",
  weight_format : String = "f32-reference",
  accumulation_dtype : ML::DType = ML::DType::F32,
  activation_mode : String = "silu-gelu",
  attention_mode : String = "self-cross-qkrms",
) : ML::ThreeD::Trellis2::DenseDeviceResourceContract
  abi = ML::ThreeD::Trellis2::DenseKernelABI.new(
    source_digest: source_digest,
    device_family: device_family,
    compiler_abi: compiler_abi,
    weight_format: weight_format,
    accumulation_dtype: accumulation_dtype,
    activation_mode: activation_mode,
    normalization_mode: "layer-rms",
    attention_mode: attention_mode,
    rope_mode: "realpair-3d",
    mask_mode: "right-valid-trim",
    layout_mode: "ncdhw-cubic"
  )
  ML::ThreeD::Trellis2::DenseDeviceResourceContract.new(
    profiles: [padded_stage_profile(voxel_buckets, context_buckets)],
    kernel_variants: ["padded-stage"],
    dtypes: [ML::DType::F32],
    cache_owner: cache_owner,
    kernel_abi: abi,
    max_axis_padding_ratio: 16.0,
    max_single_tensor_bytes: 64_i64 * 1024_i64 * 1024_i64,
    max_declared_activation_bytes: 512_i64 * 1024_i64 * 1024_i64
  )
end

private def padded_ledger(
  contract : ML::ThreeD::Trellis2::DenseDeviceResourceContract,
) : ML::ThreeD::Trellis2::BoundedKernelKeyLedger
  ML::ThreeD::Trellis2::BoundedKernelKeyLedger.new(
    contract,
    contract.theoretical_kernel_key_count
  )
end

private def padded_stage(
  resolution : Int32 = 2,
  context_length : Int32 = 3,
  max_logical_tensor_bytes : Int64 = 64_i64 * 1024_i64 * 1024_i64,
) : ML::ThreeD::Trellis2::DenseFlowStageCPU
  stage = ML::ThreeD::Trellis2::DenseFlowStageCPU.new(
    resolution: resolution,
    in_channels: 3,
    model_channels: 16,
    context_channels: 5,
    out_channels: 4,
    num_heads: 2,
    frequency_dim: 256,
    max_logical_tensor_bytes: max_logical_tensor_bytes,
    device: ML::Tensor::Device::CPU
  )
  fill_nonzero_parameters(stage.parameters)
  stage
end

private def fill_nonzero_parameters(parameters : Array(ML::Autograd::Variable)) : Nil
  parameters.each_with_index do |parameter, parameter_index|
    values = parameter.data.cpu_data.not_nil!
    values.each_index do |index|
      magnitude = 0.01_f32 + ((parameter_index * 3 + index) % 7).to_f32 * 0.005_f32
      values[index] = ((parameter_index + index).even? ? magnitude : -magnitude)
    end
    parameter.requires_grad = false
  end
end

private def fill_nonzero_block_parameters(
  block : ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock,
) : Nil
  parameters = [] of ML::Autograd::Variable
  parameters << block.modulation
  parameters << block.norm2.weight
  parameters << block.norm2.bias
  parameters << block.self_attn.q_rms_norm.gamma
  parameters << block.self_attn.k_rms_norm.gamma
  parameters << block.cross_attn.q_rms_norm.gamma
  parameters << block.cross_attn.k_rms_norm.gamma
  [
    block.self_attn.to_qkv,
    block.self_attn.to_out,
    block.cross_attn.to_q,
    block.cross_attn.to_kv,
    block.cross_attn.to_out,
    block.mlp.fc1,
    block.mlp.fc2,
  ].each { |linear| parameters.concat(linear.parameters) }
  fill_nonzero_parameters(parameters)
end

private def assert_padded_close(actual : ML::Tensor, expected : ML::Tensor, name : String) : Nil
  actual.shape.should eq(expected.shape), "#{name} shape"
  actual.to_a.each_with_index do |value, index|
    value.finite?.should be_true, "#{name}[#{index}] must be finite"
    value.should be_close(expected.to_a[index], 2e-5_f32)
  end
end

describe ML::ThreeD::Trellis2::PaddedDenseFlowStageCPU do
  it "right-pads N=8 to N=64 and S=3 to S=4, then trims exact logical output" do
    stage = padded_stage
    contract = padded_stage_contract(
      voxel_buckets: [64_i32],
      context_buckets: [4_i32],
      cache_owner: "padded-pad-owner"
    )
    request = ML::ThreeD::Trellis2::DenseActivationRequest.new(
      "padded-dense-flow-tiny", 1, 8, 3, ML::DType::F32
    )
    padded = ML::ThreeD::Trellis2::PaddedDenseFlowStageCPU.new(
      stage, contract, request, padded_ledger(contract)
    )
    voxels = ML::Tensor.from_array(
      Array(Float32).new(24) { |index| -0.2_f32 + index.to_f32 * 0.017_f32 },
      ML::Shape.new(1_i32, 3_i32, 2_i32, 2_i32, 2_i32)
    )
    timesteps = ML::Tensor.from_array([0.37_f32], ML::Shape.new(1_i32))
    context = ML::Tensor.from_array(
      Array(Float32).new(15) { |index| 0.11_f32 - index.to_f32 * 0.013_f32 },
      ML::Shape.new(1_i32, 3_i32, 5_i32)
    )

    expected = stage.forward(voxels, timesteps, context)
    trace = padded.forward_with_trace(voxels, timesteps, context)

    trace["output"].shape.should eq(ML::Shape.new(1_i32, 4_i32, 2_i32, 2_i32, 2_i32))
    assert_padded_close(trace["output"], expected, "trimmed output")
    trace["padding.phases"].shape.should eq(ML::Shape.new(64_i32, 4_i32, 2_i32))
    phase_values = trace["padding.phases"].to_a
    (8...64).each do |position|
      4.times do |pair|
        offset = (position * 4 + pair) * 2
        phase_values[offset].should eq(1.0_f32)
        phase_values[offset + 1].should eq(0.0_f32)
      end
    end
  end

  it "keeps an already-bucketed N=8/S=4 path numerically identical" do
    stage = padded_stage
    contract = padded_stage_contract(cache_owner: "padded-noop-owner")
    request = ML::ThreeD::Trellis2::DenseActivationRequest.new(
      "padded-dense-flow-tiny", 1, 8, 4, ML::DType::F32
    )
    padded = ML::ThreeD::Trellis2::PaddedDenseFlowStageCPU.new(
      stage, contract, request, padded_ledger(contract)
    )
    voxels = ML::Tensor.from_array(
      Array(Float32).new(24) { |index| 0.03_f32 + index.to_f32 * 0.009_f32 },
      ML::Shape.new(1_i32, 3_i32, 2_i32, 2_i32, 2_i32)
    )
    timesteps = ML::Tensor.from_array([0.63_f32], ML::Shape.new(1_i32))
    context = ML::Tensor.from_array(
      Array(Float32).new(20) { |index| -0.09_f32 + index.to_f32 * 0.011_f32 },
      ML::Shape.new(1_i32, 4_i32, 5_i32)
    )

    assert_padded_close(
      padded.forward(voxels, timesteps, context),
      stage.forward(voxels, timesteps, context),
      "no-op output"
    )
  end

  it "proves the unmasked physical control differs from the valid-length path" do
    block = ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
      channels: 6,
      context_channels: 4,
      num_heads: 1,
      device: ML::Tensor::Device::CPU
    )
    fill_nonzero_block_parameters(block)
    x = ML::Tensor.from_array(
      [0.2_f32, -0.1_f32, 0.3_f32, 0.4_f32, -0.5_f32, 0.6_f32,
       3.0_f32, 2.0_f32, -4.0_f32, 1.0_f32, 5.0_f32, -2.0_f32,
       91.0_f32, -73.0_f32, 61.0_f32, -55.0_f32, 49.0_f32, -43.0_f32],
      ML::Shape.new(1_i32, 3_i32, 6_i32)
    )
    context = ML::Tensor.from_array(
      [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32,
       -0.4_f32, 0.3_f32, -0.2_f32, 0.1_f32,
       77.0_f32, -66.0_f32, 55.0_f32, -44.0_f32],
      ML::Shape.new(1_i32, 3_i32, 4_i32)
    )
    phases = ML::Tensor.from_array(
      Array(Float32).new(3 * 3 * 2) { |index| index % 2 == 0 ? 1.0_f32 : 0.0_f32 },
      ML::Shape.new(3_i32, 3_i32, 2_i32)
    )
    modulation = ML::Tensor.zeros(1, 36)

    unmasked = block.forward_with_trace(
      ML::Autograd::Variable.new(x, requires_grad: false),
      ML::Autograd::Variable.new(modulation, requires_grad: false),
      ML::Autograd::Variable.new(context, requires_grad: false),
      phases
    )["output"].to_a.first(12)
    masked = block.forward_with_trace(
      ML::Autograd::Variable.new(x, requires_grad: false),
      ML::Autograd::Variable.new(modulation, requires_grad: false),
      ML::Autograd::Variable.new(context, requires_grad: false),
      phases,
      valid_voxel_tokens: 2,
      valid_context_tokens: 2
    )["output"].to_a.first(12)
    unmasked.should_not eq(masked)
  end

  it "masks adversarial self/cross pad rows and zeroes padded queries after bias" do
    block = ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
      channels: 6,
      context_channels: 4,
      num_heads: 1,
      device: ML::Tensor::Device::CPU
    )
    fill_nonzero_block_parameters(block)
    modulation = ML::Tensor.zeros(1, 36)
    logical_x = ML::Tensor.from_array(
      [0.2_f32, -0.1_f32, 0.3_f32, 0.4_f32, -0.5_f32, 0.6_f32,
       3.0_f32, 2.0_f32, -4.0_f32, 1.0_f32, 5.0_f32, -2.0_f32],
      ML::Shape.new(1_i32, 2_i32, 6_i32)
    )
    padded_x = ML::Tensor.from_array(
      logical_x.to_a + [91.0_f32, -73.0_f32, 61.0_f32, -55.0_f32, 49.0_f32, -43.0_f32],
      ML::Shape.new(1_i32, 3_i32, 6_i32)
    )
    logical_context = ML::Tensor.from_array(
      [0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32,
       -0.4_f32, 0.3_f32, -0.2_f32, 0.1_f32],
      ML::Shape.new(1_i32, 2_i32, 4_i32)
    )
    padded_context = ML::Tensor.from_array(
      logical_context.to_a + [77.0_f32, -66.0_f32, 55.0_f32, -44.0_f32],
      ML::Shape.new(1_i32, 3_i32, 4_i32)
    )
    logical_phases = ML::Tensor.from_array(
      Array(Float32).new(2 * 3 * 2) { |index| index % 2 == 0 ? 1.0_f32 : 0.0_f32 },
      ML::Shape.new(2_i32, 3_i32, 2_i32)
    )
    padded_phases = ML::Tensor.from_array(
      logical_phases.to_a + Array(Float32).new(3 * 2) { |index| index.even? ? 1.0_f32 : 0.0_f32 },
      ML::Shape.new(3_i32, 3_i32, 2_i32)
    )

    logical = block.forward_with_trace(
      ML::Autograd::Variable.new(logical_x, requires_grad: false),
      ML::Autograd::Variable.new(modulation, requires_grad: false),
      ML::Autograd::Variable.new(logical_context, requires_grad: false),
      logical_phases
    )
    padded = block.forward_with_trace(
      ML::Autograd::Variable.new(padded_x, requires_grad: false),
      ML::Autograd::Variable.new(modulation, requires_grad: false),
      ML::Autograd::Variable.new(padded_context, requires_grad: false),
      padded_phases,
      valid_voxel_tokens: 2,
      valid_context_tokens: 2
    )

    padded_output = padded["output"].to_a
    logical_output = logical["output"].to_a
    padded_output.first(12).each_with_index do |value, index|
      value.should be_close(logical_output[index], 2e-5_f32)
    end
    padded_output.last(6).each { |value| value.should eq(0.0_f32) }
    padded["self_attention"].to_a.last(6).each { |value| value.should eq(0.0_f32) }
    padded["cross_attention"].to_a.last(6).each { |value| value.should eq(0.0_f32) }
    padded_output.each { |value| value.finite?.should be_true }
  end

  it "rejects inconsistent plans and lengths before executing math" do
    stage = padded_stage
    contract = padded_stage_contract(cache_owner: "padded-input-owner")
    request = ML::ThreeD::Trellis2::DenseActivationRequest.new(
      "padded-dense-flow-tiny", 1, 8, 3, ML::DType::F32
    )
    voxels = ML::Tensor.zeros(1, 3, 2, 2, 2)
    timesteps = ML::Tensor.zeros(1)
    context = ML::Tensor.zeros(1, 3, 5)
    padded = ML::ThreeD::Trellis2::PaddedDenseFlowStageCPU.new(
      stage, contract, request, padded_ledger(contract)
    )
    padded.forward(voxels, timesteps, context).shape.should eq(
      ML::Shape.new(1_i32, 4_i32, 2_i32, 2_i32, 2_i32)
    )

    short_context = ML::Tensor.zeros(1, 2, 5)
    expect_raises(ArgumentError, /logical context length/) do
      padded.forward(voxels, timesteps, short_context)
    end

    expect_raises(ArgumentError, /rank 5/) do
      padded.forward(ML::Tensor.zeros(1, 3, 2, 2), timesteps, context)
    end

    block = ML::ThreeD::Trellis2::SharedModulatedTransformerCrossBlock.new(
      channels: 6,
      context_channels: 4,
      num_heads: 1,
      device: ML::Tensor::Device::CPU
    )
    phases = ML::Tensor.from_array(
      Array(Float32).new(2 * 3 * 2) { |index| index.even? ? 1.0_f32 : 0.0_f32 },
      ML::Shape.new(2_i32, 3_i32, 2_i32)
    )
    expect_raises(ArgumentError, /valid length/) do
      block.forward_with_trace(
        ML::Autograd::Variable.new(ML::Tensor.zeros(1, 2, 6), requires_grad: false),
        ML::Autograd::Variable.new(ML::Tensor.zeros(1, 36), requires_grad: false),
        ML::Autograd::Variable.new(ML::Tensor.zeros(1, 2, 4), requires_grad: false),
        phases,
        valid_voxel_tokens: 0,
        valid_context_tokens: 2
      )
    end
  end

  it "rejects padded tensors above the stage cap before materialization" do
    stage = padded_stage(max_logical_tensor_bytes: 1024_i64)
    contract = padded_stage_contract(
      voxel_buckets: [64_i32],
      context_buckets: [4_i32],
      cache_owner: "padded-cap-owner"
    )
    request = ML::ThreeD::Trellis2::DenseActivationRequest.new(
      "padded-dense-flow-tiny", 1, 8, 3, ML::DType::F32
    )
    ledger = padded_ledger(contract)
    expect_raises(ArgumentError, /padded tensor .*exceeds stage max_logical_tensor_bytes/) do
      ML::ThreeD::Trellis2::PaddedDenseFlowStageCPU.new(
        stage, contract, request, ledger
      )
    end
    ledger.size.should eq(0)
  end

  it "rejects a stale contract ledger before admission" do
    stage = padded_stage
    contract = padded_stage_contract(cache_owner: "padded-identity-owner")
    stale_contract = padded_stage_contract(
      cache_owner: "padded-identity-owner",
      source_digest: "2" * 64
    )
    request = ML::ThreeD::Trellis2::DenseActivationRequest.new(
      "padded-dense-flow-tiny", 1, 8, 3, ML::DType::F32
    )
    stale_ledger = padded_ledger(stale_contract)
    expect_raises(ArgumentError, /not declared by this contract/) do
      ML::ThreeD::Trellis2::PaddedDenseFlowStageCPU.new(
        stage, contract, request, stale_ledger
      )
    end
    stale_ledger.size.should eq(0)

    foreign_contract = padded_stage_contract(cache_owner: "padded-foreign-owner")
    foreign_ledger = padded_ledger(foreign_contract)
    expect_raises(ArgumentError, /not declared by this contract/) do
      ML::ThreeD::Trellis2::PaddedDenseFlowStageCPU.new(
        stage, contract, request, foreign_ledger
      )
    end
    foreign_ledger.size.should eq(0)
  end

  it "rejects every non-oracle ABI variant before ledger admission" do
    stage = padded_stage
    request = ML::ThreeD::Trellis2::DenseActivationRequest.new(
      "padded-dense-flow-tiny", 1, 8, 3, ML::DType::F32
    )
    cases = [
      {
        label:              "BF16 accumulation",
        device_family:      "cpu-oracle",
        compiler_abi:       "t2n2d0-v1",
        weight_format:      "f32-reference",
        accumulation_dtype: ML::DType::BF16,
        activation_mode:    "silu-gelu",
        attention_mode:     "self-cross-qkrms",
      },
      {
        label:              "activation mode",
        device_family:      "cpu-oracle",
        compiler_abi:       "t2n2d0-v1",
        weight_format:      "f32-reference",
        accumulation_dtype: ML::DType::F32,
        activation_mode:    "relu",
        attention_mode:     "self-cross-qkrms",
      },
      {
        label:              "attention mode",
        device_family:      "cpu-oracle",
        compiler_abi:       "t2n2d0-v1",
        weight_format:      "f32-reference",
        accumulation_dtype: ML::DType::F32,
        activation_mode:    "silu-gelu",
        attention_mode:     "self-only",
      },
      {
        label:              "weight format",
        device_family:      "cpu-oracle",
        compiler_abi:       "t2n2d0-v1",
        weight_format:      "bf16-reference",
        accumulation_dtype: ML::DType::F32,
        activation_mode:    "silu-gelu",
        attention_mode:     "self-cross-qkrms",
      },
      {
        label:              "device family",
        device_family:      "metal-gpu",
        compiler_abi:       "t2n2d0-v1",
        weight_format:      "f32-reference",
        accumulation_dtype: ML::DType::F32,
        activation_mode:    "silu-gelu",
        attention_mode:     "self-cross-qkrms",
      },
      {
        label:              "compiler ABI",
        device_family:      "cpu-oracle",
        compiler_abi:       "t2n2d0-v2",
        weight_format:      "f32-reference",
        accumulation_dtype: ML::DType::F32,
        activation_mode:    "silu-gelu",
        attention_mode:     "self-cross-qkrms",
      },
    ]
    cases.each_with_index do |variant, index|
      contract = padded_stage_contract(
        cache_owner: "padded-abi-#{index}",
        device_family: variant[:device_family],
        compiler_abi: variant[:compiler_abi],
        weight_format: variant[:weight_format],
        accumulation_dtype: variant[:accumulation_dtype],
        activation_mode: variant[:activation_mode],
        attention_mode: variant[:attention_mode]
      )
      ledger = padded_ledger(contract)
      expect_raises(ArgumentError, /exact CPU oracle ABI/) do
        ML::ThreeD::Trellis2::PaddedDenseFlowStageCPU.new(
          stage, contract, request, ledger
        )
      end
      ledger.size.should eq(0), variant[:label]
    end
  end
end
