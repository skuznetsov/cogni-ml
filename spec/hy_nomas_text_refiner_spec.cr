require "spec"
require "../src/ml/video/hy_nomas_text_refiner"

describe ML::Video::HyNomasTextRefinerMPSPlan do
  it "keeps all BF16 prelude controls opt-in and preserves input-only casts" do
    ML::Video::HyNomasTextRefinerMPSPlan::DEFAULT_PRELUDE_INPUT_BF16_CONTROL.should be_false
    ML::Video::HyNomasTextRefinerMPSPlan::DEFAULT_PRELUDE_LINEAR_BF16_CONTROL.should be_false
    ML::Video::HyNomasTextRefinerMPSPlan::DEFAULT_PRELUDE_SILU_BF16_CONTROL.should be_false
    ML::Video::HyNomasTextRefinerMPSPlan::DEFAULT_PRELUDE_SECOND_LINEAR_BF16_CONTROL.should be_false
    ML::Video::HyNomasTextRefinerMPSPlan::DEFAULT_PRELUDE_ADD_BF16_CONTROL.should be_false
    ML::Video::HyNomasTextRefinerMPSPlan::DEFAULT_BLOCK_BOUNDARIES_BF16_CONTROL.should be_false
    ML::Video::HyNomasTextRefinerMPSPlan::DEFAULT_QKV_BF16_CONTROL.should be_false
    ML::Video::HyNomasMPSActivation::UnaryPlan::DEFAULT_BF16_CONTROL.should be_false

    source = ML::Video::HyNomasTextRefinerMPSPlan::SOURCE
    source.includes?("constant uint &prelude_input_bf16_control [[buffer(4)]]").should be_true
    source.includes?("constant uint &prelude_input_bf16_control [[buffer(3)]]").should be_true
    source.includes?("sum = hy_text_roundtrip_bf16(sum);").should be_true
    source.includes?("float denominator = hy_text_roundtrip_bf16((float)valid_rows);").should be_true
    source.includes?("mean[channel] = hy_text_roundtrip_bf16(sum / denominator);").should be_true
    source.includes?("mean[channel] = sum / (float)valid_rows;").should be_true
    source.includes?("hy_text_roundtrip_bf16(precise::cos(phase))").should be_true
    source.includes?("hy_text_roundtrip_bf16(precise::sin(phase))").should be_true
    source.includes?("embedding[index] = precise::cos(phase);").should be_true
    source.includes?("embedding[index + half_dim] = precise::sin(phase);").should be_true
  end

  it "requires connected norm1 boundaries before QKV BF16 output rounding" do
    ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, true, true, true, true, true)
    expect_raises(ArgumentError, /QKV BF16 control requires the block-boundary BF16 control/) do
      ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, true, true, true, false, true)
    end
    ML::Video::HyNomasTextRefinerMPSPlan::QKV_LINEAR_NAMES.should eq(["block0.qkv", "block1.qkv"])
  end

  it "requires each prelude precision control to include its preceding boundary" do
    ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(false, false, false)
    ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, false)
    ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, true)
    ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, true, true, true)
    ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, true, true, true, true)

    expect_raises(ArgumentError, /Block-boundary BF16 control requires all five prelude BF16 controls/) do
      ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, true, true, false, true)
    end
    expect_raises(ArgumentError, /input BF16 control/) do
      ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(false, true, false)
    end
    expect_raises(ArgumentError, /linear BF16 control/) do
      ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, false, true)
    end
    expect_raises(ArgumentError, /second-linear BF16 control requires the SiLU BF16 control/) do
      ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, false, true, false)
    end
    expect_raises(ArgumentError, /add BF16 control requires the second-linear BF16 control/) do
      ML::Video::HyNomasTextRefinerMPSPlan.validate_precision_controls!(true, true, true, false, true)
    end
  end

  it "limits post-bias BF16 rounding to the four first prelude projections" do
    names = ML::Video::HyNomasTextRefinerMPSPlan::PRELUDE_FIRST_LINEAR_NAMES
    names.should eq(["refiner_timestep_mlp1", "context_linear1", "input_embed", "time_in.mlp1"])
    {"refiner_timestep", "context_embed", "time_in.mlp2",
     "block0.adaLN", "block0.mlp_fc1", "block1.adaLN", "block1.mlp_fc1"}.each do |name|
      names.includes?(name).should be_false
    end

    source = ML::Video::MetalLinearMPS::SOURCE
    source.includes?("constant uint &round_output_bf16 [[buffer(6)]]").should be_true
    source.includes?("video_linear_mps_roundtrip_bf16(value)").should be_true
  end

  it "limits BF16 SiLU execution to the three prelude activation nodes" do
    names = ML::Video::HyNomasTextRefinerMPSPlan::PRELUDE_SILU_NAMES
    names.should eq(["refiner_timestep_silu", "context_silu", "time_in.silu"])
    {"block0.condition_silu", "block0.mlp_silu",
     "block1.condition_silu", "block1.mlp_silu"}.each do |name|
      names.includes?(name).should be_false
    end
  end

  it "limits BF16 rounding to the three second prelude linears and c addition" do
    names = ML::Video::HyNomasTextRefinerMPSPlan::PRELUDE_SECOND_LINEAR_NAMES
    names.should eq(["refiner_timestep", "context_embed", "time_in.mlp2"])
    {"refiner_timestep_mlp1", "context_linear1", "input_embed", "time_in.mlp1",
     "block0.adaLN", "block0.mlp_fc1", "block1.adaLN", "block1.mlp_fc1"}.each do |name|
      names.includes?(name).should be_false
    end

    source = ML::Video::HyNomasTextRefinerMPSPlan::SOURCE
    source.includes?("kernel void hy_nomas_text_c_add_bf16_f32(").should be_true
    source.includes?("c[index] = hy_text_roundtrip_bf16(timestep[index] + context[index]);").should be_true
  end

  it "limits the cumulative BF16 control to both condition SiLUs, adaLN projections, and norm1 affine outputs" do
    ML::Video::HyNomasTextRefinerMPSPlan::BLOCK_BOUNDARY_CONDITION_SILU_NAMES.should eq(["block0.condition_silu", "block1.condition_silu"])
    ML::Video::HyNomasTextRefinerMPSPlan::BLOCK_BOUNDARY_ADALN_NAMES.should eq(["block0.adaLN", "block1.adaLN"])

    source = ML::Video::HyNomasTextRefinerMPSPlan::SOURCE
    source.includes?("constant uint &round_output_bf16 [[buffer(9)]]").should be_true
    source.includes?("value = hy_text_roundtrip_bf16(value);").should be_true
    source.includes?("values[index] = values[index] * weight + bias;").should be_false
  end

  it "binds the exact 42-tensor F32 upstream subset without renaming source keys" do
    bindings = ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings
    bindings.size.should eq(42)
    bindings.all? { |binding| binding.target == binding.source }.should be_true
    bindings.all? { |binding| binding.dtype == ML::GGUF::SafeTensorDType::F32 }.should be_true
    bindings.sum do |binding|
      binding.shape.reduce(1_i64) { |count, dimension| count * dimension } * 4_i64
    end.should eq(583_311_360_i64)

    names = bindings.map(&.source).to_set
    names.includes?("txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight").should be_true
    names.includes?("txt_in.individual_token_refiner.blocks.1.adaLN_modulation.1.bias").should be_true
    names.includes?("txt_in.t_embedder.mlp.0.weight").should be_true
    names.includes?("time_in.mlp.2.weight").should be_true
    names.any?(&.includes?("self_attn_q_norm")).should be_false
  end

  it "admits only binary, nonempty contiguous valid prefixes beginning at token zero" do
    mask = Array(Int32).new(1280) { |index| index < 23 ? 1 : 0 }
    ML::Video::HyNomasTextRefinerMPSPlan.valid_tokens(mask).should eq(23)

    invalid = mask.dup
    invalid[0] = 0
    expect_raises(ArgumentError, /token zero/) do
      ML::Video::HyNomasTextRefinerMPSPlan.valid_tokens(invalid)
    end
    invalid = mask.dup
    invalid[24] = 1
    expect_raises(ArgumentError, /contiguous/) do
      ML::Video::HyNomasTextRefinerMPSPlan.valid_tokens(invalid)
    end
    invalid = mask.dup
    invalid[1] = 2
    expect_raises(ArgumentError, /binary/) do
      ML::Video::HyNomasTextRefinerMPSPlan.valid_tokens(invalid)
    end
    expect_raises(ArgumentError, /valid token/) do
      ML::Video::HyNomasTextRefinerMPSPlan.valid_tokens(Array(Int32).new(1280, 0))
    end
    expect_raises(ArgumentError, /exactly 1280/) do
      ML::Video::HyNomasTextRefinerMPSPlan.valid_tokens([1, 0])
    end
  end

  it "keeps named-buffer accounting unchanged for prelude precision controls" do
    mask = Array(Int32).new(1280) { |index| index < 23 ? 1 : 0 }
    bytes = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask)
    bytes.activations.should eq(8_610_820_i64)
    bytes.linear_stage.should eq(67_108_864_i64)
    bytes.attention_stage.should eq(1_507_328_i64)
    bytes.total.should eq(77_227_012_i64)
    ML::Video::HyNomasTextRefinerMPSPlan.admit_named_bytes!(mask, bytes.total).should eq(bytes)
    expect_raises(ArgumentError, /budget/) do
      ML::Video::HyNomasTextRefinerMPSPlan.admit_named_bytes!(mask, bytes.total - 1)
    end
    expect_raises(ArgumentError, /seg_pos_emb_num=0/) do
      ML::Video::HyNomasTextRefinerMPSPlan.admit_named_bytes!(mask, bytes.total, 1)
    end

    full = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(Array(Int32).new(1280, 1))
    (full.total > bytes.total).should be_true
  end
end
