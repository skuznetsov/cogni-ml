require "spec"
require "../src/ml/llm/llama"

class ML::LLM::Model
  def initialize(*, regression_seed_probe : Bool)
    @handle = Pointer(Void).null.as(LlamaFFI::LlamaModel)
    @vocab = Pointer(Void).null.as(LlamaFFI::LlamaVocab)
    @path = ""
    @n_ctx_train = @n_embd = @n_layers = @n_heads = 0
    @size_bytes = @n_params = 0_u64
    @description = ""
  end

  def create_context(n_ctx : Int32 = 0, n_batch : Int32 = 512,
                     n_ubatch : Int32 = 0, n_threads : Int32 = 0, flash_attn : Bool = true,
                     embeddings : Bool = false)
    Context.new(self, regression_sampler_probe: true)
  end

  def tokenize(text : String, add_bos : Bool = true)
    [] of Int32
  end
end

class ML::LLM::Context
  getter regression_seeds = [] of UInt32

  def initialize(@model : Model, *, regression_sampler_probe : Bool)
    @handle = Pointer(Void).null.as(LlamaFFI::LlamaContext)
    @sampler = nil
    @n_ctx = @n_batch = 0_u32
  end

  def setup_sampler(temperature : Float32 = 0.8_f32, top_k : Int32 = 40,
                    top_p : Float32 = 0.95_f32, min_p : Float32 = 0.05_f32,
                    seed : UInt32 = LlamaFFI::LLAMA_DEFAULT_SEED)
    @regression_seeds << seed
  end

  def reset
  end

  # Stop after observing the generation entry points' sampler setup.
  def eval(tokens : Array(Int32))
    false
  end
end

describe ML::LLM::Generator do
  it "preserves the configured sampler seed when generation resets the sampler" do
    model = ML::LLM::Model.new(regression_seed_probe: true)
    generator = ML::LLM::Generator.new(model, sampler_seed: 42_u32)
    generator.generate("hello")
    generator.ask("hello")
    generator.context.regression_seeds.should eq([42_u32, 42_u32, 42_u32, 42_u32])
  end
end
