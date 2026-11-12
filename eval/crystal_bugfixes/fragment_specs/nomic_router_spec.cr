require "spec"
require "../src/ml/gguf/nomic_bert"

class ML::GGUF::NomicBertMoE(B)
  def initialize(@backend : B, *, regression_experts : Int32)
    @dim = @head_dim = @n_heads = @max_seq_len = @vocab_size = @n_experts_used = @ffn_dim = 1
    @n_layers = @moe_every_n = 0
    @n_experts = regression_experts
    @rope_theta = 10_000_f32
  end

  def route_for_regression(weights : LayerWeights)
    moe_ffn([1_f32], 1, weights).first
  end
end

private def f32_weights(*values : Float32)
  floats = values.to_a
  bytes = Bytes.new(floats.size * sizeof(Float32))
  floats.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * sizeof(Float32), sizeof(Float32)])
  end
  ML::GGUF::QuantWeight.new(bytes, ML::GGUF::TensorType::F32, floats.size, 1)
end

describe ML::GGUF::NomicBertMoE do
  it "preserves the selected expert's probability from the full softmax" do
    model = ML::GGUF::NomicBertMoE(ML::GGUF::F32Backend).new(ML::GGUF::F32Backend.new, regression_experts: 2)
    zero = f32_weights(0_f32)
    unused = [0_f32]
    weights = ML::GGUF::NomicBertMoE::LayerWeights.new(
      zero, unused, zero, unused, unused, unused, unused, unused,
      gate_w: [0_f32, Math.log(0.25).to_f32],
      expert_up_w: f32_weights(1_f32, 9_f32),
      expert_down_w: f32_weights(1_f32, 9_f32),
    )
    expected = 0.8_f32 * ML::GGUF::F32Backend.new.gelu(1_f32)
    model.route_for_regression(weights).should be_close(expected, 1e-5_f32)
  end
end
