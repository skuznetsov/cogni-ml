require "spec"
require "../src/ml/llm/llama_ffi"

describe "ML::LLM::LlamaFFI ABI" do
  it "matches llama.cpp b9960 by-value parameter struct sizes" do
    sizeof(ML::LLM::LlamaFFI::LlamaModelParams).should eq(72)
    sizeof(ML::LLM::LlamaFFI::LlamaContextParams).should eq(160)
  end

  it "matches the b9960 context parameter field offsets that shifted" do
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @n_rs_seq).should eq(16)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @n_outputs_max).should eq(20)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @n_threads).should eq(24)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @ctx_type).should eq(32)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @samplers).should eq(136)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @ctx_other).should eq(152)
  end
end
