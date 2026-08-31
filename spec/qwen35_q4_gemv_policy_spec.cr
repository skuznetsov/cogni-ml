require "./spec_helper"
require "../src/ml/gguf/qwen35_metal"

describe "Q4_K x16 default route policy" do
  it "enables only the measured Qwen3.8 FFN and recurrent routes" do
    policy = ML::GGUF::Qwen35Metal
    capability = ML::GGUF::Q4GemvX16Capability::Qwen38

    policy.q4_gemv_x16_default_route?("qwen35:blk.0.ffn_gate.weight", 5120, 17408, capability).should be_true
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.ffn_up.weight", 5120, 17408, capability).should be_true
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.attn_qkv.weight", 5120, 10240, capability).should be_true
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.attn_gate.weight", 5120, 6144, capability).should be_true
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.ssm_alpha.weight", 5120, 48, capability).should be_true
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.ssm_beta.weight", 5120, 48, capability).should be_true
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.ssm_out.weight", 6144, 5120, capability).should be_true

    policy.q4_gemv_x16_default_route?("qwen35:blk.0.ffn_down.weight", 17408, 5120, capability).should be_false
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.ffn_up.weight", 4096, 12288, capability).should be_false
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.attn_q.weight", 5120, 12288, capability).should be_false
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.attn_k.weight", 5120, 1024, capability).should be_false
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.attn_v.weight", 5120, 1024, capability).should be_false
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.attn_output.weight", 6144, 5120, capability).should be_false
    policy.q4_gemv_x16_default_route?("qwen35:blk.0.attn_qkv.weight", 4096, 8192, capability).should be_false
  end

  it "requires the exact Qwen3.8 model profile while preserving Gemma" do
    policy = ML::GGUF::Qwen35Metal
    unknown = ML::GGUF::Q4GemvX16Capability::Unknown

    policy.q4_gemv_x16_default_route?("qwen35:blk.0.ffn_up.weight", 5120, 17408, unknown).should be_false
    policy.q4_gemv_x16_default_route?("gemma4:blk.0.ffn_up.weight", 3840, 15360, unknown).should be_true
    policy.q4_gemv_x16_default_route?(nil, 5120, 17408, unknown).should be_false
  end

  it "issues the optimization profile only for the measured GGUF identity" do
    weights = ML::GGUF::Qwen35Weights
    qwen38 = ML::GGUF::Q4GemvX16Capability::Qwen38
    unknown = ML::GGUF::Q4GemvX16Capability::Unknown

    weights.q4_gemv_x16_capability_for("Qwen_Qwen3.8 27B", "Qwen_Qwen3.8").should eq(qwen38)
    weights.q4_gemv_x16_capability_for("Qwen3.6-27B", "Qwen3.6-27B").should eq(unknown)
    weights.q4_gemv_x16_capability_for("Qwen_Qwen3.8 27B", "Qwen3.8").should eq(unknown)
    weights.q4_gemv_x16_capability_for(nil, "Qwen_Qwen3.8").should eq(unknown)
  end

  it "keeps unprofiled quantized weights fail-closed" do
    qw = ML::GGUF::QuantWeight.new(Bytes.empty, ML::GGUF::TensorType::F32, 1, 1)

    qw.q4_gemv_x16_capability.should eq(ML::GGUF::Q4GemvX16Capability::Unknown)
  end
end
