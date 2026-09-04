require "spec"
require "../src/ml/llm/llama_ffi"

describe "ML::LLM::LlamaFFI ABI" do
  it "matches the ggml cache type values used by llama context parameters" do
    ML::LLM::LlamaFFI::GgmlType::F32.value.should eq(0)
    ML::LLM::LlamaFFI::GgmlType::F16.value.should eq(1)
  end

  it "matches llama.cpp b10434 enum values used in by-value parameters" do
    ML::LLM::LlamaFFI::LlamaSplitMode::Tensor.value.should eq(3)
    ML::LLM::LlamaFFI::LlamaLoadMode::Auto.value.should eq(-1)
    ML::LLM::LlamaFFI::LlamaLoadMode::None.value.should eq(0)
    ML::LLM::LlamaFFI::LlamaLoadMode::Mmap.value.should eq(1)
    ML::LLM::LlamaFFI::LlamaLoadMode::Mlock.value.should eq(2)
    ML::LLM::LlamaFFI::LlamaLoadMode::MmapMlock.value.should eq(3)
    ML::LLM::LlamaFFI::LlamaLoadMode::DirectIO.value.should eq(4)
  end

  it "matches llama.cpp b10434 by-value parameter struct sizes" do
    sizeof(ML::LLM::LlamaFFI::LlamaModelParams).should eq(72)
    sizeof(ML::LLM::LlamaFFI::LlamaContextParams).should eq(160)
  end

  it "matches the b10434 model parameter field offsets" do
    offsetof(ML::LLM::LlamaFFI::LlamaModelParams, @n_gpu_layers).should eq(16)
    offsetof(ML::LLM::LlamaFFI::LlamaModelParams, @split_mode).should eq(20)
    offsetof(ML::LLM::LlamaFFI::LlamaModelParams, @load_mode).should eq(24)
    offsetof(ML::LLM::LlamaFFI::LlamaModelParams, @main_gpu).should eq(28)
    offsetof(ML::LLM::LlamaFFI::LlamaModelParams, @tensor_split).should eq(32)
    offsetof(ML::LLM::LlamaFFI::LlamaModelParams, @vocab_only).should eq(64)
    offsetof(ML::LLM::LlamaFFI::LlamaModelParams, @load_mtp).should eq(69)
  end

  it "matches the b10434 context parameter field offsets" do
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @n_rs_seq).should eq(16)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @n_outputs_max).should eq(20)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @n_outputs_max_per_seq).should eq(24)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @n_threads).should eq(28)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @n_threads_batch).should eq(32)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @ctx_type).should eq(36)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @flash_attn_type).should eq(52)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @cb_eval).should eq(88)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @samplers).should eq(136)
    offsetof(ML::LLM::LlamaFFI::LlamaContextParams, @ctx_other).should eq(152)
  end
end
