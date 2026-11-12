require "./spec_helper"
require "../src/ml/gguf/qwen35_qlora_block"
require "../src/ml/gguf/qwen35_cpu"

private def qlora_tiny_hparams : ML::GGUF::Qwen35Hparams
  file = File.tempfile("qwen35-qlora-hparams")
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
    metadata["qwen35.full_attention_interval"] = 1_i32
    metadata["qwen35.rope.freq_base"] = 10_000.0_f32
    metadata["qwen35.rope.dimension_count"] = 2_i32
    sections = Array(ML::GGUF::Value).new
    sections << 1_i32
    sections << 0_i32
    sections << 0_i32
    sections << 0_i32
    metadata["qwen35.rope.dimension_sections"] = sections
    metadata["qwen35.ssm.conv_kernel"] = 2_i32
    metadata["qwen35.ssm.state_size"] = 2_i32
    metadata["qwen35.ssm.group_count"] = 1_i32
    metadata["qwen35.ssm.time_step_rank"] = 1_i32
    metadata["qwen35.ssm.inner_size"] = 2_i32

    ML::GGUF::Qwen35Hparams.new(gguf)
  ensure
    gguf.close
    File.delete(path) if File.exists?(path)
  end
end

private def qlora_f32_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32) : ML::GGUF::QuantWeight
  raw = Bytes.new(values.size * sizeof(Float32))
  raw.to_unsafe.copy_from(values.to_unsafe.as(Pointer(UInt8)), raw.size)
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
end

private def qlora_matrix(out_dim : Int32, in_dim : Int32, phase : Int32) : Array(Float32)
  Array(Float32).new(out_dim * in_dim) do |i|
    ((((i * 13 + phase * 7) % 23) - 11).to_f32 / 37.0_f32) + 0.003_f32
  end
end

private def qlora_tiny_weights : ML::GGUF::Qwen35FullAttnWeights
  hidden = 4
  q_dim = 4
  kv_dim = 2
  n_ff = 3
  ML::GGUF::Qwen35FullAttnWeights.new(
    [1.05_f32, 0.93_f32, 1.11_f32, 0.87_f32],
    qlora_f32_weight(qlora_matrix(2 * q_dim, hidden, 1), 2 * q_dim, hidden),
    [1.1_f32, 0.9_f32],
    qlora_f32_weight(qlora_matrix(kv_dim, hidden, 2), kv_dim, hidden),
    [0.91_f32, 1.08_f32],
    qlora_f32_weight(qlora_matrix(kv_dim, hidden, 3), kv_dim, hidden),
    qlora_f32_weight(qlora_matrix(hidden, q_dim, 4), hidden, q_dim),
    [0.89_f32, 1.02_f32, 1.13_f32, 0.95_f32],
    qlora_f32_weight(qlora_matrix(n_ff, hidden, 5), n_ff, hidden),
    qlora_f32_weight(qlora_matrix(n_ff, hidden, 6), n_ff, hidden),
    qlora_f32_weight(qlora_matrix(hidden, n_ff, 7), hidden, n_ff),
  )
end

private def qlora_loss(output : Array(Float32), grad : Array(Float32)) : Float64
  output.each_with_index.sum(0.0_f64) { |value, i| value.to_f64 * grad[i].to_f64 }
end

private def qlora_with_cpu_reference(&block : -> Nil) : Nil
  keys = [
    "QWEN35_FULL_LAYER_FUSE_OFF",
    "QWEN35_FULL_ATTN_FUSE_OFF",
    "QWEN35_ATTN_FUSE_OFF",
    "QWEN35_ATTN_CPU",
    "QWEN35_FFN_FUSE_OFF",
    "QWEN35_BATCH_OFF",
  ]
  previous = {} of String => String?
  keys.each do |key|
    previous[key] = ENV[key]?
    ENV[key] = "1"
  end
  begin
    yield
  ensure
    previous.each do |key, value|
      if value
        ENV[key] = value
      else
        ENV.delete(key)
      end
    end
  end
end

describe ML::GGUF::Qwen35QLoRA::Block do
  it "matches the frozen full-attention reference for batched prefill and incremental decode" do
    hp = qlora_tiny_hparams
    lw = qlora_tiny_weights
    block = ML::GGUF::Qwen35QLoRA::Block.new(lw, hp, 17_u64)
    input = [
      0.3_f32, -0.7_f32, 0.2_f32, 0.9_f32,
      -0.5_f32, 0.4_f32, 0.8_f32, -0.1_f32,
      0.6_f32, 0.1_f32, -0.4_f32, 0.7_f32,
    ]

    batched_state = ML::GGUF::Qwen35QLoRA::DecodeState.new
    batched = block.prefill(input, batched_state)
    batched_state.position.should eq(3)
    batched_state.key_cache.size.should eq(3 * hp.n_head_kv * hp.head_dim)
    batched.all?(&.finite?).should be_true

    # The existing CPU implementation is the frozen base reference. LoRA B
    # starts at zero, so batched and incremental paths must preserve it.
    cpu_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: hp.context_length)
    reference = [] of Float32
    incremental_state = ML::GGUF::Qwen35QLoRA::DecodeState.new
    incremental = [] of Float32
    qlora_with_cpu_reference do
      3.times do |row|
        token = input[row * hp.n_embd, hp.n_embd]
        actual = block.decode(token, incremental_state)
        expected = ML::GGUF::Qwen35CPU.forward_full_attn_layer(
          token, row.to_i32, lw, cpu_state.layers[0], hp, hp.context_length,
        )
        incremental.concat(actual)
        reference.concat(expected)

        cached = (row + 1) * hp.n_head_kv * hp.head_dim
        cpu_keys = cpu_state.layers[0].k_cache.not_nil!
        cpu_values = cpu_state.layers[0].v_cache.not_nil!
        cached.times do |i|
          (incremental_state.key_cache[i] - cpu_keys[i]).abs.should be < 2.0e-5_f32
          (incremental_state.value_cache[i] - cpu_values[i]).abs.should be < 2.0e-5_f32
        end
      end
      incremental_state.position.should eq(3)

      incremental.zip(reference).each { |actual, expected| actual.should be_close(expected, 2.0e-4_f32) }
      batched.zip(reference).each { |actual, expected| actual.should be_close(expected, 2.0e-4_f32) }
    end
  end

  it "matches finite differences for Q/K/V/O and all SwiGLU adapters" do
    hp = qlora_tiny_hparams
    block = ML::GGUF::Qwen35QLoRA::Block.new(qlora_tiny_weights, hp, 91_u64)
    input = [
      0.3_f32, -0.7_f32, 0.2_f32, 0.9_f32,
      -0.5_f32, 0.4_f32, 0.8_f32, -0.1_f32,
      0.6_f32, 0.1_f32, -0.4_f32, 0.7_f32,
    ]
    output_grad = Array(Float32).new(input.size) { |i| (((i * 7) % 13) - 6).to_f32 / 11.0_f32 }

    block.zero_grad
    block.forward(input)
    block.backward(output_grad)

    block.projections.each_with_index do |projection, projection_index|
      selected = 0
      projection.grad_b.size.times do |i|
        selected = i if projection.grad_b[i].abs > projection.grad_b[selected].abs
      end
      analytic = projection.grad_b[selected]
      analytic.abs.should be > 1.0e-6_f32
      original = projection.b[selected]
      epsilon = 2.0e-3_f32

      projection.b[selected] = original + epsilon
      plus = qlora_loss(block.forward(input), output_grad)
      projection.b[selected] = original - epsilon
      minus = qlora_loss(block.forward(input), output_grad)
      projection.b[selected] = original

      numeric = ((plus - minus) / (2.0_f64 * epsilon.to_f64)).to_f32
      tolerance = 3.0e-3_f32 + analytic.abs * 0.04_f32
      (numeric - analytic).abs.should be < tolerance
    end
  end
end
