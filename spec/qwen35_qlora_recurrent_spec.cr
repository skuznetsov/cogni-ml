require "./spec_helper"
require "../src/ml/gguf/qwen35_qlora_recurrent"
require "../src/ml/gguf/qwen35_cpu"

private def qwen35_qlora_recurrent_hparams : ML::GGUF::Qwen35Hparams
  file = File.tempfile("qwen35-qlora-recurrent-hparams")
  path = file.path
  file.write("GGUF".to_slice)
  file.write_bytes(3_u32, IO::ByteFormat::LittleEndian)
  file.write_bytes(0_i64, IO::ByteFormat::LittleEndian)
  file.write_bytes(0_i64, IO::ByteFormat::LittleEndian)
  file.close

  gguf = ML::GGUF::GGUFFile.new(path, mmap_tensors: false)
  begin
    metadata = gguf.metadata
    metadata["qwen35.block_count"] = 1_i32
    metadata["qwen35.embedding_length"] = 4_i32
    metadata["qwen35.feed_forward_length"] = 3_i32
    metadata["qwen35.context_length"] = 12_i32
    metadata["qwen35.attention.head_count"] = 2_i32
    metadata["qwen35.attention.head_count_kv"] = 1_i32
    metadata["qwen35.attention.key_length"] = 2_i32
    metadata["qwen35.attention.value_length"] = 2_i32
    metadata["qwen35.attention.layer_norm_rms_epsilon"] = 1.0e-5_f32
    metadata["qwen35.full_attention_interval"] = 2_i32
    metadata["qwen35.rope.freq_base"] = 10_000.0_f32
    metadata["qwen35.rope.dimension_count"] = 2_i32
    sections = Array(ML::GGUF::Value).new
    sections << 1_i32
    sections << 0_i32
    sections << 0_i32
    sections << 0_i32
    metadata["qwen35.rope.dimension_sections"] = sections
    metadata["qwen35.ssm.conv_kernel"] = 3_i32
    metadata["qwen35.ssm.state_size"] = 2_i32
    metadata["qwen35.ssm.group_count"] = 1_i32
    metadata["qwen35.ssm.time_step_rank"] = 2_i32
    metadata["qwen35.ssm.inner_size"] = 4_i32

    ML::GGUF::Qwen35Hparams.new(gguf)
  ensure
    gguf.close
    File.delete(path) if File.exists?(path)
  end
end

private def qwen35_qlora_recurrent_f32_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32) : ML::GGUF::QuantWeight
  raw = Bytes.new(values.size * sizeof(Float32))
  raw.to_unsafe.copy_from(values.to_unsafe.as(Pointer(UInt8)), raw.size)
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
end

private def qwen35_qlora_recurrent_matrix(out_dim : Int32, in_dim : Int32, phase : Int32) : Array(Float32)
  Array(Float32).new(out_dim * in_dim) do |i|
    ((((i * 11 + phase * 5) % 19) - 9).to_f32 / 31.0_f32) + 0.002_f32
  end
end

private def qwen35_qlora_recurrent_weights : ML::GGUF::Qwen35RecurrentWeights
  hidden = 4
  qkv_dim = 8
  heads = 2
  state = 2
  conv_kernel = 3
  n_ff = 3
  ML::GGUF::Qwen35RecurrentWeights.new(
    [1.05_f32, 0.93_f32, 1.11_f32, 0.87_f32],
    qwen35_qlora_recurrent_f32_weight(qwen35_qlora_recurrent_matrix(qkv_dim, hidden, 1), qkv_dim, hidden),
    qwen35_qlora_recurrent_f32_weight(qwen35_qlora_recurrent_matrix(hidden, hidden, 2), hidden, hidden),
    [-0.35_f32, -0.27_f32],
    qwen35_qlora_recurrent_f32_weight(qwen35_qlora_recurrent_matrix(heads, hidden, 3), heads, hidden),
    qwen35_qlora_recurrent_f32_weight(qwen35_qlora_recurrent_matrix(heads, hidden, 4), heads, hidden),
    Array(Float32).new(qkv_dim * conv_kernel) { |i| (((i * 7) % 13) - 6).to_f32 / 21.0_f32 },
    [0.1_f32, -0.05_f32],
    [1.1_f32, 0.9_f32],
    qwen35_qlora_recurrent_f32_weight(qwen35_qlora_recurrent_matrix(hidden, heads * state, 5), hidden, heads * state),
    [0.89_f32, 1.02_f32, 1.13_f32, 0.95_f32],
    qwen35_qlora_recurrent_f32_weight(qwen35_qlora_recurrent_matrix(n_ff, hidden, 6), n_ff, hidden),
    qwen35_qlora_recurrent_f32_weight(qwen35_qlora_recurrent_matrix(n_ff, hidden, 7), n_ff, hidden),
    qwen35_qlora_recurrent_f32_weight(qwen35_qlora_recurrent_matrix(hidden, n_ff, 8), hidden, n_ff),
  )
end

private def qwen35_qlora_recurrent_loss(
  block : ML::GGUF::Qwen35QLoRA::RecurrentBlock,
  input : Array(Float32), grad : Array(Float32),
) : Float64
  block.forward(input, retain_tape: false).each_with_index.sum(0.0_f64) do |value, i|
    value.to_f64 * grad[i].to_f64
  end
end

describe ML::GGUF::Qwen35QLoRA::RecurrentBlock do
  it "runs a complete sample from the zero recurrent state" do
    hp = qwen35_qlora_recurrent_hparams
    lw = qwen35_qlora_recurrent_weights
    input = [
      0.3_f32, -0.7_f32, 0.2_f32, 0.9_f32,
      -0.5_f32, 0.4_f32, 0.8_f32, -0.1_f32,
      0.6_f32, 0.1_f32, -0.4_f32, 0.7_f32,
    ]

    actual = ML::GGUF::Qwen35QLoRA::RecurrentBlock.new(lw, hp).forward(input)
    actual.size.should eq(input.size)
    actual.all?(&.finite?).should be_true

    {% if flag?(:cpu_only) %}
      state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: hp.context_length)
      expected = [] of Float32
      3.times do |row|
        expected.concat(ML::GGUF::Qwen35CPU.forward_recurrent_layer(
          input[row * hp.n_embd, hp.n_embd], row.to_i32, lw, state.layers[0], hp, hp.context_length,
        ))
      end
      actual.zip(expected).each { |value, reference| value.should be_close(reference, 3.0e-4_f32) }
    {% end %}
  end

  it "propagates input gradients through the complete frozen recurrent block" do
    hp = qwen35_qlora_recurrent_hparams
    block = ML::GGUF::Qwen35QLoRA::RecurrentBlock.new(qwen35_qlora_recurrent_weights, hp)
    input = [
      0.3_f32, -0.7_f32, 0.2_f32, 0.9_f32,
      -0.5_f32, 0.4_f32, 0.8_f32, -0.1_f32,
      0.6_f32, 0.1_f32, -0.4_f32, 0.7_f32,
    ]
    grad = [
      0.25_f32, -0.5_f32, 0.75_f32, 0.1_f32,
      -0.2_f32, 0.3_f32, 0.4_f32, -0.6_f32,
      0.7_f32, -0.1_f32, 0.2_f32, 0.5_f32,
    ]

    block.forward(input)
    actual = block.backward(grad)
    step = 1.0e-3_f32
    tolerance = 1.2e-2_f32
    input.each_index do |index|
      original = input[index]
      input[index] = original + step
      plus = qwen35_qlora_recurrent_loss(block, input, grad)
      input[index] = original - step
      minus = qwen35_qlora_recurrent_loss(block, input, grad)
      input[index] = original
      numerical = ((plus - minus) / (2.0_f32 * step)).to_f32
      actual[index].should be_close(numerical, tolerance)
    end
  end

  it "supports tape opt-out, explicit release, and input/gradient shape validation" do
    hp = qwen35_qlora_recurrent_hparams
    block = ML::GGUF::Qwen35QLoRA::RecurrentBlock.new(qwen35_qlora_recurrent_weights, hp)
    input = Array(Float32).new(hp.n_embd * 2, 0.25_f32)

    expect_raises(ArgumentError) { block.forward([1.0_f32]) }
    block.forward(input, retain_tape: false)
    expect_raises(ArgumentError) { block.backward(Array(Float32).new(input.size, 1.0_f32)) }
    block.forward(input)
    expect_raises(ArgumentError) { block.backward([1.0_f32]) }
    block.release_tape
    expect_raises(ArgumentError) { block.backward(Array(Float32).new(input.size, 1.0_f32)) }
  end
end

describe ML::GGUF::Qwen35CPU, "recurrent QLoRA generation adapter" do
  it "uses batch prefill and scalar decode with the stock zero-state semantics" do
    hp = qwen35_qlora_recurrent_hparams
    lw = qwen35_qlora_recurrent_weights
    input = [
      0.3_f32, -0.7_f32, 0.2_f32, 0.9_f32,
      -0.5_f32, 0.4_f32, 0.8_f32, -0.1_f32,
      0.6_f32, 0.1_f32, -0.4_f32, 0.7_f32,
    ]
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: hp.context_length)
    actual = ML::GGUF::Qwen35CPU.qlora_recurrent(input, lw, state.layers[0], hp, hp.context_length)

    {% if flag?(:cpu_only) %}
      reference_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: hp.context_length)
      expected = [] of Float32
      3.times do |row|
        expected.concat(ML::GGUF::Qwen35CPU.forward_recurrent_layer(
          input[row * hp.n_embd, hp.n_embd], row.to_i32, lw, reference_state.layers[0], hp, hp.context_length,
        ))
      end
      actual.zip(expected).each { |value, reference| value.should be_close(reference, 1.0e-6_f32) }
    {% else %}
      actual.size.should eq(input.size)
      actual.all?(&.finite?).should be_true
    {% end %}

    actual_step = ML::GGUF::Qwen35CPU.qlora_recurrent(
      input[0, hp.n_embd], lw, ML::GGUF::Qwen35CPU::LayerState.new, hp, hp.context_length,
    )
    reference_step = ML::GGUF::Qwen35CPU.forward_recurrent_layer(
      input[0, hp.n_embd], 0, lw, ML::GGUF::Qwen35CPU::LayerState.new, hp, hp.context_length,
    )
    actual_step.zip(reference_step).each { |value, reference| value.should eq(reference) }
  end
end
