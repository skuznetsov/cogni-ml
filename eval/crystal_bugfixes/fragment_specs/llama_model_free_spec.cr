require "spec"
require "../src/ml/llm/llama"

module FreeProbe
  class_property calls : Int32 = 0
end

fun llama_model_free(handle : ML::LLM::LlamaFFI::LlamaModel) : Void
  FreeProbe.calls += 1
end

class ML::LLM::Model
  def initialize(*, regression_handle : LlamaFFI::LlamaModel)
    @handle = regression_handle
    @vocab = Pointer(Void).null.as(LlamaFFI::LlamaVocab)
    @path = ""
    @n_ctx_train = @n_embd = @n_layers = @n_heads = 0
    @size_bytes = @n_params = 0_u64
    @description = ""
  end
end

describe ML::LLM::Model do
  it "releases a native model at most once across repeated finalization" do
    model = ML::LLM::Model.new(regression_handle: Pointer(Void).new(1_u64).as(ML::LLM::LlamaFFI::LlamaModel))
    model.finalize
    model.finalize
    FreeProbe.calls.should eq(1)
  end
end
