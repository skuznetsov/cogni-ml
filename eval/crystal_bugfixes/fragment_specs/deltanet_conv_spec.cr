require "spec"
require "../src/ml/gguf/qwen35_cpu"

struct ML::GGUF::Qwen35Hparams
  def initialize(*, regression_kernel : Int32)
    @arch = "qwen35"
    @n_layer = @n_embd = @n_ff = @n_head = @n_head_kv = @head_dim = 1
    @context_length = @full_attention_interval = 2
    @vocab_size = 1
    @rms_eps = 1e-6_f32
    @rope_freq_base = 10_000_f32
    @rope_dim_count = 0
    @rope_sections = [0, 0, 0, 0]
    @ssm_conv_kernel = regression_kernel
    @ssm_state_size = @ssm_group_count = @ssm_time_step_rank = @ssm_inner_size = 1
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

describe ML::GGUF::Qwen35CPU do
  it "applies channel-major GGUF convolution coefficients in a recurrent layer" do
    hp = ML::GGUF::Qwen35Hparams.new(regression_kernel: 2)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: 2)
    one = f32_weights(1_f32)
    zero = f32_weights(0_f32)
    weights = ML::GGUF::Qwen35RecurrentWeights.new(
      [1_f32], f32_weights(1_f32, 1_f32, 1_f32), one,
      [0_f32], zero, zero, [0_f32, 1_f32, 0_f32, 1_f32, 0_f32, 1_f32],
      [0_f32], [1_f32], one, [1_f32], zero, zero, zero,
    )
    ML::GGUF::Qwen35CPU.forward_recurrent_layer([1_f32], 0, weights, state.layers[0], hp, 2)
    # With zero prior history, the last coefficient of each channel drives its
    # output. Positive k and v must write positive state through the delta rule.
    state.layers[0].ssm_state.not_nil!.first.should be > 0.3_f32
  end
end
