require "spec"
require "../src/ml/llm/llama"

module BackendProbe
  EVENTS = [] of Symbol
end

fun llama_backend_init : Void
  BackendProbe::EVENTS << :init
end

fun llama_backend_free : Void
  BackendProbe::EVENTS << :free
end

describe ML::LLM do
  it "initializes and frees the backend only once per lifecycle" do
    ML::LLM.init
    ML::LLM.init
    ML::LLM.cleanup
    ML::LLM.cleanup
    ML::LLM.init
    ML::LLM.cleanup
    BackendProbe::EVENTS.should eq([:init, :free, :init, :free])
  end
end
