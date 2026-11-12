require "spec"
require "../src/ml/llm/llama_ffi"

describe "ML::LLM::LlamaFFI b9960 ABI" do
  it "matches the context-parameter struct layout expected by b9960" do
    sizeof(ML::LLM::LlamaFFI::LlamaContextParams).should eq(160)
  end
end
