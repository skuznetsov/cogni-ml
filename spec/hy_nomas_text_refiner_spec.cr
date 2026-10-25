require "spec"
require "../src/ml/video/hy_nomas_text_refiner"

describe ML::Video::HyNomasTextRefinerMPSPlan do
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

  it "accounts for compact F32 activations, the shared linear stage, and attention staging" do
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
