require "spec"
require "../src/ml/gguf/nomic_bert"

class ML::GGUF::NomicBertMoE(B)
  def initialize(@backend : B, *, regression_head_dim : Int32)
    @dim = @head_dim = regression_head_dim
    @n_heads = @max_seq_len = @vocab_size = 1
    @n_layers = @n_experts = @n_experts_used = @moe_every_n = @ffn_dim = 0
    @rope_theta = 10_000_f32
    @rope_cos = [0_f32, 1_f32]
    @rope_sin = [1_f32, 0_f32]
  end

  def rotate_for_regression(values : Array(Float32))
    apply_rope!(values, 1, 0)
  end
end

describe ML::GGUF::NomicBertMoE do
  it "rotates corresponding dimensions from opposite halves of a NeoX head" do
    model = ML::GGUF::NomicBertMoE(ML::GGUF::F32Backend).new(ML::GGUF::F32Backend.new, regression_head_dim: 4)
    values = [99_f32, 1_f32, 2_f32, 3_f32, 4_f32, 88_f32]
    model.rotate_for_regression(values)
    values.should eq([99_f32, -3_f32, 2_f32, 1_f32, 4_f32, 88_f32])
  end
end
