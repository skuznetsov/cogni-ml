require "spec"
require "../src/ml/llm/llama"

module LoadProbe
  MLOCK = [] of Bool
end

alias LlamaModelParams = ML::LLM::LlamaFFI::LlamaModelParams
alias LlamaModel = ML::LLM::LlamaFFI::LlamaModel

fun llama_model_default_params : LlamaModelParams
  params = LlamaModelParams.new
  params.use_mlock = false
  params
end

fun llama_model_load_from_file(path : LibC::Char*, params : LlamaModelParams) : LlamaModel
  LoadProbe::MLOCK << params.use_mlock
  Pointer(Void).null.as(LlamaModel)
end

describe ML::LLM::Model do
  it "requests locked mmap pages by default" do
    expect_raises(Exception, "Failed to load model: fixture.gguf") do
      ML::LLM::Model.new("fixture.gguf", n_gpu_layers: 0)
    end
    LoadProbe::MLOCK.should eq([true])
  end
end
