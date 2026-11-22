require "./spec_helper"
require "../src/ml/gguf/inkling2_weights"
require "../src/ml/gguf/inkling2_matmul"
require "../src/ml/gguf/inkling2_attention"
require "../src/ml/gguf/inkling2_metal_view"
require "../src/ml/gguf/inkling2_resident_attention"
require "../src/ml/gguf/inkling2_resident_math"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  private record ResidentProjectedRow,
    q : Array(Float32),
    k : Array(Float32),
    v : Array(Float32),
    q_rel : Array(Float32),
    gate : Array(Float32),
    sink : Array(Float32),
    dynamic_k : Array(Float32)?,
    dynamic_v : Array(Float32)?

  private def resident_test_config(
    *,
    sliding : Bool,
    n_head : Int32,
    n_head_kv : Int32,
    head_dim : Int32,
    rel_dim : Int32,
    window : Int32,
    max_seq : Int32,
  ) : ML::GGUF::Inkling2Config
    n_embd = n_head * head_dim
    ML::GGUF::Inkling2Config.new(
      n_layer: 1_i32,
      n_embd: n_embd,
      n_head: n_head,
      n_head_kv: n_head_kv,
      head_dim: head_dim,
      rel_dim: rel_dim,
      rel_extent: 4_i32,
      sliding_window: window,
      conv_size: 2_i32,
      log_scale_floor: 2_i32,
      log_scale_alpha: 0.17_f32,
      qk_clip: 0.35_f32,
      rel_clip: 0.9_f32,
      rms_eps: 1e-5_f32,
      n_expert: 1_i32,
      n_expert_used: 1_i32,
      expert_dim: 1_i32,
      expert_scale: 1_f32,
      situ_beta: 1_f32,
      situ_linear_beta: 1_f32,
      res_block_size: 1_i32,
      vocab_size: 2_i32,
      padded_vocab_size: 2_i32,
      context_length: max_seq,
      core_start: 0_i32,
      core_end: 1_i32,
      passes: 3_i32,
      gate_rank: 1_i32,
      logit_softcap: 30_f32,
      varying: false,
      sliding_pattern: [sliding],
    )
  end

  private def resident_f32_weight(label : String, out_dim : Int32, in_dim : Int32, values : Array(Float32)) : ML::GGUF::QuantWeight
    raise "bad resident test matrix size for #{label}" unless values.size == out_dim * in_dim
    bytes = Bytes.new(values.size * sizeof(Float32), 0_u8)
    bytes.to_unsafe.as(Float32*).copy_from(values.to_unsafe, values.size)
    ML::GGUF::QuantWeight.new(bytes, ML::GGUF::TensorType::F32, out_dim, in_dim, "resident-test:#{label}")
  end

  private def resident_matrix(rows : Int32, columns : Int32, salt : Int32) : Array(Float32)
    Array(Float32).new(rows * columns) do |index|
      value = ((index * 7 + salt * 11) % 19 - 9).to_f32
      value * 0.025_f32
    end
  end

  private def resident_attention_weights(config : ML::GGUF::Inkling2Config) : ML::GGUF::Inkling2AttentionWeights
    q_width = config.n_head * config.head_dim
    kv_width = config.n_head_kv * config.head_dim
    relative_extent = config.sliding?(0) ? config.sliding_window : config.rel_extent
    output = Array(Float32).new(config.n_embd * q_width, 0_f32)
    Math.min(config.n_embd, q_width).times { |index| output[index * q_width + index] = 1_f32 }
    dynamic_k = if config.sliding?(0)
                  nil
                else
                  resident_f32_weight("dynamic-k", config.conv_size * config.n_head_kv, config.n_embd,
                    resident_matrix(config.conv_size * config.n_head_kv, config.n_embd, 8))
                end
    dynamic_v = if config.sliding?(0)
                  nil
                else
                  resident_f32_weight("dynamic-v", config.conv_size * config.n_head_kv, config.n_embd,
                    resident_matrix(config.conv_size * config.n_head_kv, config.n_embd, 9))
                end
    relative_projection = resident_matrix(config.n_head * config.rel_dim, config.n_embd, 6)
      .map { |value| value * 25_f32 }

    ML::GGUF::Inkling2AttentionWeights.new(
      q: resident_f32_weight("q", q_width, config.n_embd, resident_matrix(q_width, config.n_embd, 1)),
      k: resident_f32_weight("k", kv_width, config.n_embd, resident_matrix(kv_width, config.n_embd, 2)),
      v: resident_f32_weight("v", kv_width, config.n_embd, resident_matrix(kv_width, config.n_embd, 3)),
      output: resident_f32_weight("output", config.n_embd, q_width, output),
      gate: resident_f32_weight("gate", q_width, config.n_embd, resident_matrix(q_width, config.n_embd, 4)),
      sink: resident_f32_weight("sink", config.n_head, config.n_embd, resident_matrix(config.n_head, config.n_embd, 5)),
      rel_proj: resident_f32_weight("relative", config.n_head * config.rel_dim, config.n_embd,
        relative_projection),
      q_clip: Array(Float32).new(q_width) { |index| 0.7_f32 + (index % 4).to_f32 * 0.1_f32 },
      rel_bias: Array(Float32).new(relative_extent * config.rel_dim) { |index| (index % 5 - 2).to_f32 * 0.09_f32 },
      rel_clip: Array(Float32).new(config.n_head * config.rel_dim) { |index| 0.8_f32 + (index % 3).to_f32 * 0.1_f32 },
      k_conv: resident_matrix(kv_width, config.conv_size, 10),
      v_conv: resident_matrix(kv_width, config.conv_size, 11),
      post_conv: Array(Float32).new(config.n_embd * config.conv_size, 0_f32),
      k_conv_proj: dynamic_k,
      v_conv_proj: dynamic_v,
    )
  end

  private def resident_projected_rows(
    weights : ML::GGUF::Inkling2AttentionWeights,
    matmul : ML::GGUF::Inkling2CPUMatmul,
    inputs : Array({Array(Float32), Array(Float32), Array(Float32)}),
  ) : Array(ResidentProjectedRow)
    inputs.map do |q_in, k_in, v_in|
      jobs = [
        {weights.q, q_in}, {weights.k, k_in}, {weights.v, v_in},
        {weights.rel_proj, q_in}, {weights.gate, q_in}, {weights.sink, q_in},
      ]
      if k_projection = weights.k_conv_proj
        jobs << {k_projection, k_in}
        jobs << {weights.v_conv_proj.not_nil!, v_in}
      end
      projections = matmul.multiply_many(jobs)
      ResidentProjectedRow.new(
        projections[0], projections[1], projections[2], projections[3], projections[4], projections[5],
        weights.k_conv_proj ? projections[6] : nil,
        weights.v_conv_proj ? projections[7] : nil,
      )
    end
  end

  private def resident_inputs(config : ML::GGUF::Inkling2Config, rows : Int32) : Array({Array(Float32), Array(Float32), Array(Float32)})
    Array({Array(Float32), Array(Float32), Array(Float32)}).new(rows) do |row|
      make = ->(salt : Int32) do
        Array(Float32).new(config.n_embd) do |column|
          (((row + 2) * 13 + column * 5 + salt * 7) % 23 - 11).to_f32 * 0.06_f32
        end
      end
      {make.call(1), make.call(2), make.call(3)}
    end
  end

  private def resident_flatten(rows : Array(ResidentProjectedRow), &block : ResidentProjectedRow -> Array(Float32)) : Array(Float32)
    result = [] of Float32
    rows.each { |row| result.concat(yield row) }
    result
  end

  private def resident_shared_view(values : Array(Float32), buffers : Array(ML::MetalBuffer)) : ML::GGUF::Inkling2View
    buffer = ML::MetalBuffer.from_array(values)
    buffers << buffer
    ML::GGUF::Inkling2View.new(buffer)
  end

  private def resident_run_tile(
    attention : ML::GGUF::Inkling2ResidentAttention,
    start : Int32,
    projected : Array(ResidentProjectedRow),
    weights : ML::GGUF::Inkling2AttentionWeights,
    config : ML::GGUF::Inkling2Config,
    *,
    decode_block : Bool = false,
  ) : Array(Float32)
    rows = projected.size.to_i32
    q_width = config.n_head * config.head_dim
    output_count = rows * q_width
    buffers = [] of ML::MetalBuffer
    scratch = ML::GGUF::Inkling2ResidentScratch.new
    output = ML::MetalBuffer.new(output_count.to_i64 * sizeof(Float32), ML::StorageMode::Shared)
    begin
      q = resident_shared_view(resident_flatten(projected) { |row| row.q }, buffers)
      k = resident_shared_view(resident_flatten(projected) { |row| row.k }, buffers)
      v = resident_shared_view(resident_flatten(projected) { |row| row.v }, buffers)
      q_rel = resident_shared_view(resident_flatten(projected) { |row| row.q_rel }, buffers)
      gate = resident_shared_view(resident_flatten(projected) { |row| row.gate }, buffers)
      sink = resident_shared_view(resident_flatten(projected) { |row| row.sink }, buffers)
      q_clip = resident_shared_view(weights.q_clip, buffers)
      rel_clip = resident_shared_view(weights.rel_clip, buffers)
      relative_bias = resident_shared_view(weights.rel_bias, buffers)
      k_taps = resident_shared_view(weights.k_conv, buffers)
      v_taps = resident_shared_view(weights.v_conv, buffers)
      dynamic_k = if projected.first.dynamic_k
                    resident_shared_view(resident_flatten(projected) { |row| row.dynamic_k.not_nil! }, buffers)
                  end
      dynamic_v = if projected.first.dynamic_v
                    resident_shared_view(resident_flatten(projected) { |row| row.dynamic_v.not_nil! }, buffers)
                  end

      command = ML::Metal::CommandBuffer.new
      encoder = ML::Metal::ComputeEncoder.new(command)
      attended = nil.as(ML::GGUF::Inkling2View?)
      begin
        attended = attention.encode(encoder, scratch, start, rows, q, k, v, q_rel, gate, sink,
          q_clip, rel_clip, relative_bias, k_taps, v_taps, dynamic_k, dynamic_v,
          decode_block: decode_block)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, attended.not_nil!, ML::GGUF::Inkling2View.new(output), output_count)
      ensure
        encoder.end_encoding
      end
      command.commit_and_wait
      attention.finish
      output.read(output_count)
    ensure
      scratch.close
      buffers.each(&.release)
      output.release
    end
  end

  private def resident_encode_convolution_tile(
    convolution : ML::GGUF::Inkling2ResidentConvolution,
    start : Int32,
    input_values : Array(Float32),
    tap_values : Array(Float32),
  ) : Nil
    buffers = [] of ML::MetalBuffer
    scratch = ML::GGUF::Inkling2ResidentScratch.new
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    begin
      input = resident_shared_view(input_values, buffers)
      taps = resident_shared_view(tap_values, buffers)
      begin
        convolution.encode(encoder, scratch, start, input_values.size.to_i32, input, taps)
      ensure
        encoder.end_encoding
      end
      command.commit_and_wait
    ensure
      scratch.close
      buffers.each(&.release)
    end
  end

  private def resident_run_convolution_tile(
    convolution : ML::GGUF::Inkling2ResidentConvolution,
    start : Int32,
    input_values : Array(Float32),
    tap_values : Array(Float32),
  ) : Array(Float32)
    count = input_values.size.to_i32
    buffers = [] of ML::MetalBuffer
    scratch = ML::GGUF::Inkling2ResidentScratch.new
    output = ML::MetalBuffer.new(count.to_i64 * sizeof(Float32), ML::StorageMode::Shared)
    command = ML::Metal::CommandBuffer.new
    encoder = ML::Metal::ComputeEncoder.new(command)
    begin
      input = resident_shared_view(input_values, buffers)
      taps = resident_shared_view(tap_values, buffers)
      begin
        result = convolution.encode(encoder, scratch, start, count, input, taps)
        ML::GGUF::Inkling2ResidentMath.copy(encoder, result, ML::GGUF::Inkling2View.new(output), count)
      ensure
        encoder.end_encoding
      end
      command.commit_and_wait
      output.read(count)
    ensure
      scratch.close
      buffers.each(&.release)
      output.release
    end
  end

  if ML::Metal::Device.init!
    describe ML::GGUF::Inkling2ResidentAttention do
      it "matches sliding attention across tile boundaries, ring wrap, oversized batches, reset, and KV snapshots" do
        config = resident_test_config(sliding: true, n_head: 4_i32, n_head_kv: 2_i32,
          head_dim: 3_i32, rel_dim: 1_i32, window: 3_i32, max_seq: 16_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 9_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, true, 16_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 16_i32)
        expected = [] of Float32

        inputs.each_with_index do |(q_in, k_in, v_in), position|
          output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul, q_in, k_in, v_in, position.to_i32)
          output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
        end

        actual = resident_run_tile(gpu, 0_i32, projected[0, 5], weights, config)
        actual.concat(resident_run_tile(gpu, 5_i32, projected[5, 4], weights, config))
        actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 3e-4_f32 }

        3.times do |chronological|
          key, value, position = cpu.key_value_at(chronological.to_i32)
          snapshot_key, snapshot_value, snapshot_position = gpu.key_value_at(position, 9_i32)
          snapshot_position.should eq position
          key.size.times { |channel| (snapshot_key[channel] - key[channel]).abs.should be < 3e-4_f32 }
          value.size.times { |channel| (snapshot_value[channel] - value[channel]).abs.should be < 3e-4_f32 }
        end
        expect_raises(IndexError) { gpu.key_value_at(4_i32, 5_i32) }

        gpu.reset
        cpu.reset
        reset_expected = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
          inputs[0][0], inputs[0][1], inputs[0][2], 0_i32)
        reset_actual = resident_run_tile(gpu, 0_i32, [projected[0]], weights, config)
        reset_actual.size.times do |index|
          (reset_actual[index] - reset_expected[index] * Math.sqrt(config.conv_size + 1).to_f32).abs.should be < 3e-4_f32
        end
      ensure
        gpu.try(&.close)
      end

      it "preserves the old full-attention prefix when cache growth crosses 256 rows" do
        config = resident_test_config(sliding: false, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 1_i32, rel_dim: 1_i32, window: 4_i32, max_seq: 300_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 258_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, false, 300_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, false, 300_i32)
        expected = [] of Float32

        inputs.each_with_index do |(q_in, k_in, v_in), position|
          output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul, q_in, k_in, v_in, position.to_i32)
          expected << output[0] * Math.sqrt(config.conv_size + 1).to_f32
        end

        actual = resident_run_tile(gpu, 0_i32, projected[0, 256], weights, config)
        actual.concat(resident_run_tile(gpu, 256_i32, projected[256, 2], weights, config))
        actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 4e-4_f32 }

        {0_i32, 127_i32, 255_i32, 257_i32}.each do |position|
          key, value, actual_position = gpu.key_value_at(position, 258_i32)
          expected_key, expected_value, expected_position = cpu.key_value_at(position)
          actual_position.should eq position
          expected_position.should eq position
          (key[0] - expected_key[0]).abs.should be < 3e-4_f32
          (value[0] - expected_value[0]).abs.should be < 3e-4_f32
        end
      ensure
        gpu.try(&.close)
      end

      it "matches split-K decode after 64 keys through a retained sliding ring" do
        config = resident_test_config(sliding: true, n_head: 2_i32, n_head_kv: 1_i32,
          head_dim: 2_i32, rel_dim: 2_i32, window: 70_i32, max_seq: 90_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 80_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, true, 90_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 90_i32)
        expected = [] of Float32

        inputs.each_with_index do |(q_in, k_in, v_in), position|
          output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
            q_in, k_in, v_in, position.to_i32)
          output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
        end

        actual = [] of Float32
        projected.each_with_index do |row, position|
          actual.concat(resident_run_tile(gpu, position.to_i32, [row], weights, config))
        end
        actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 5e-4_f32 }

        {10_i32, 42_i32, 79_i32}.each do |position|
          snapshot_key, snapshot_value, snapshot_position = gpu.key_value_at(position, 80_i32)
          expected_key, expected_value, expected_position = cpu.key_value_at(position - 10)
          snapshot_position.should eq position
          expected_position.should eq position
          snapshot_key.size.times { |channel| (snapshot_key[channel] - expected_key[channel]).abs.should be < 3e-4_f32 }
          snapshot_value.size.times { |channel| (snapshot_value[channel] - expected_value[channel]).abs.should be < 3e-4_f32 }
        end
        expect_raises(IndexError) { gpu.key_value_at(9_i32, 80_i32) }
        expect_raises(IndexError) { gpu.key_value_at(9_i32, 10_i32) }
      ensure
        gpu.try(&.close)
      end

      it "matches serial attention across the 64-key split threshold in a four-row decode block" do
        config = resident_test_config(sliding: false, n_head: 2_i32, n_head_kv: 1_i32,
          head_dim: 2_i32, rel_dim: 2_i32, window: 70_i32, max_seq: 70_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 66_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        blocked = ML::GGUF::Inkling2ResidentAttention.new(config, false, 70_i32)
        serial = ML::GGUF::Inkling2ResidentAttention.new(config, false, 70_i32)
        begin
          # Both paths share a 62-row prefix. The following four rows expose
          # 63, 64, 65, and 66 keys, so the block crosses the >64 split-K gate.
          resident_run_tile(blocked, 0_i32, projected[0, 62], weights, config)
          resident_run_tile(serial, 0_i32, projected[0, 62], weights, config)

          expected = [] of Float32
          4.times do |row|
            expected.concat(resident_run_tile(serial, 62 + row, [projected[62 + row]], weights, config))
          end
          actual = resident_run_tile(blocked, 62_i32, projected[62, 4], weights, config, decode_block: true)
          actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 2e-6_f32 }

          {62_i32, 63_i32, 64_i32, 65_i32}.each do |position|
            block_key, block_value, _block_position = blocked.key_value_at(position, 66_i32)
            serial_key, serial_value, _serial_position = serial.key_value_at(position, 66_i32)
            block_key.size.times { |channel| (block_key[channel] - serial_key[channel]).abs.should be < 2e-6_f32 }
            block_value.size.times { |channel| (block_value[channel] - serial_value[channel]).abs.should be < 2e-6_f32 }
          end
        ensure
          blocked.close
          serial.close
        end
      end

      it "matches serial attention when a four-row decode block crosses the 1024-slot KV ring" do
        config = resident_test_config(sliding: true, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 1_i32, rel_dim: 1_i32, window: 1024_i32, max_seq: 1026_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        inputs = resident_inputs(config, 1026_i32)
        projected = resident_projected_rows(weights, matmul, inputs)
        blocked = ML::GGUF::Inkling2ResidentAttention.new(config, true, 1026_i32)
        serial = ML::GGUF::Inkling2ResidentAttention.new(config, true, 1026_i32)
        begin
          resident_run_tile(blocked, 0_i32, projected[0, 1022], weights, config)
          resident_run_tile(serial, 0_i32, projected[0, 1022], weights, config)

          expected = [] of Float32
          4.times do |row|
            expected.concat(resident_run_tile(serial, 1022 + row, [projected[1022 + row]], weights, config))
          end
          actual = resident_run_tile(blocked, 1022_i32, projected[1022, 4], weights, config, decode_block: true)
          actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 3e-6_f32 }

          {2_i32, 1023_i32, 1024_i32, 1025_i32}.each do |position|
            block_key, block_value, _block_position = blocked.key_value_at(position, 1026_i32)
            serial_key, serial_value, _serial_position = serial.key_value_at(position, 1026_i32)
            (block_key[0] - serial_key[0]).abs.should be < 2e-6_f32
            (block_value[0] - serial_value[0]).abs.should be < 2e-6_f32
          end
        ensure
          blocked.close
          serial.close
        end
      end

      it "preserves 128-channel split-K results across a four-row decode block and a reserved KV ring" do
        config = resident_test_config(sliding: true, n_head: 2_i32, n_head_kv: 1_i32,
          head_dim: 128_i32, rel_dim: 16_i32, window: 1024_i32, max_seq: 1026_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        pattern = resident_projected_rows(weights, matmul, resident_inputs(config, 23_i32))
        projected = Array(ResidentProjectedRow).new(1026) { |row| pattern[row % pattern.size] }
        blocked = ML::GGUF::Inkling2ResidentAttention.new(config, true, 1026_i32, rollback_rows: 4_i32)
        serial = ML::GGUF::Inkling2ResidentAttention.new(config, true, 1026_i32)
        begin
          resident_run_tile(blocked, 0_i32, projected[0, 1022], weights, config)
          resident_run_tile(serial, 0_i32, projected[0, 1022], weights, config)

          expected = [] of Float32
          4.times do |row|
            expected.concat(resident_run_tile(serial, 1022 + row, [projected[1022 + row]], weights, config))
          end
          actual = resident_run_tile(blocked, 1022_i32, projected[1022, 4], weights, config, decode_block: true)
          actual.map(&.unsafe_as(UInt32)).should eq(expected.map(&.unsafe_as(UInt32)))

          {2_i32, 1023_i32, 1024_i32, 1025_i32}.each do |position|
            block_key, block_value, block_position = blocked.key_value_at(position, 1026_i32)
            serial_key, serial_value, serial_position = serial.key_value_at(position, 1026_i32)
            block_position.should eq(serial_position)
            block_key.map(&.unsafe_as(UInt32)).should eq(serial_key.map(&.unsafe_as(UInt32)))
            block_value.map(&.unsafe_as(UInt32)).should eq(serial_value.map(&.unsafe_as(UInt32)))
          end
        ensure
          blocked.close
          serial.close
        end
      end

      it "rewinds a speculative sliding block without losing the accepted branch across cache wrap" do
        config = resident_test_config(sliding: true, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 1_i32, rel_dim: 1_i32, window: 3_i32, max_seq: 23_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        original_inputs = resident_inputs(config, 23_i32)
        alternate_inputs = resident_inputs(config, 2_i32).map do |(q_in, k_in, v_in)|
          {q_in.map { |value| value * 1.7_f32 + 0.1_f32 },
           k_in.map { |value| value * -0.8_f32 + 0.2_f32 },
           v_in.map { |value| value * 1.3_f32 - 0.15_f32 }}
        end
        original_projected = resident_projected_rows(weights, matmul, original_inputs)
        alternate_projected = resident_projected_rows(weights, matmul, alternate_inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, true, 23_i32, rollback_rows: 4_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 23_i32)
        expected = [] of Float32

        begin
          # The speculative rows 19..22 cross the seven-slot physical KV ring
          # while preserving the accepted logical window of three rows.
          resident_run_tile(gpu, 0_i32, original_projected[0, 19], weights, config)
          resident_run_tile(gpu, 19_i32, original_projected[19, 4], weights, config)
          gpu.rewind(21_i32)

          21.times do |position|
            q_in, k_in, v_in = original_inputs[position]
            output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
              q_in, k_in, v_in, position.to_i32)
            output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
          end
          2.times do |row|
            q_in, k_in, v_in = alternate_inputs[row]
            output = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
              q_in, k_in, v_in, (21 + row).to_i32)
            output.each { |value| expected << value * Math.sqrt(config.conv_size + 1).to_f32 }
          end

          actual = resident_run_tile(gpu, 21_i32, [alternate_projected[0]], weights, config)
          # This rewind is within the reserve and before the current end, but
          # it crosses the latest appended tile's start and would revisit
          # history already changed by the accepted branch row.
          expect_raises(ArgumentError) { gpu.rewind(18_i32) }
          actual.concat(resident_run_tile(gpu, 22_i32, [alternate_projected[1]], weights, config))
          2.times do |row|
            q_width = config.n_head * config.head_dim
            q_width.times do |channel|
              (actual[row * q_width + channel] - expected[(21 + row) * q_width + channel]).abs.should be < 4e-5_f32
            end
          end

          {20_i32, 21_i32, 22_i32}.each_with_index do |position, chronological|
            actual_key, actual_value, actual_position = gpu.key_value_at(position, 23_i32)
            expected_key, expected_value, expected_position = cpu.key_value_at(chronological.to_i32)
            actual_position.should eq position
            expected_position.should eq position
            (actual_key[0] - expected_key[0]).abs.should be < 4e-5_f32
            (actual_value[0] - expected_value[0]).abs.should be < 4e-5_f32
          end
        ensure
          gpu.close
        end
      end

      it "retains a partial prefix across four scalar speculative writes over ring wrap" do
        config = resident_test_config(sliding: true, n_head: 1_i32, n_head_kv: 1_i32,
          head_dim: 1_i32, rel_dim: 1_i32, window: 3_i32, max_seq: 23_i32)
        weights = resident_attention_weights(config)
        matmul = ML::GGUF::Inkling2CPUMatmul.new
        original_inputs = resident_inputs(config, 23_i32)
        alternate_inputs = resident_inputs(config, 1_i32).map do |(q_in, k_in, v_in)|
          {q_in.map { |value| value * 1.7_f32 + 0.1_f32 },
           k_in.map { |value| value * -0.8_f32 + 0.2_f32 },
           v_in.map { |value| value * 1.3_f32 - 0.15_f32 }}
        end
        original_projected = resident_projected_rows(weights, matmul, original_inputs)
        alternate_projected = resident_projected_rows(weights, matmul, alternate_inputs)
        gpu = ML::GGUF::Inkling2ResidentAttention.new(config, true, 23_i32, rollback_rows: 4_i32)
        cpu = ML::GGUF::Inkling2AttentionState.new(config, true, 23_i32)

        begin
          resident_run_tile(gpu, 0_i32, original_projected[0, 19], weights, config)
          expect_raises(ArgumentError) { gpu.begin_speculation(18_i32) }
          gpu.begin_speculation(19_i32)
          4.times do |offset|
            resident_run_tile(gpu, 19_i32 + offset, [original_projected[19 + offset]], weights, config)
          end
          expect_raises(ArgumentError) { gpu.rewind(22_i32) }
          expect_raises(ArgumentError) do
            resident_run_tile(gpu, 23_i32, [original_projected[22]], weights, config)
          end
          expect_raises(ArgumentError) { gpu.finish_speculation(18_i32) }
          gpu.finish_speculation(21_i32)

          21.times do |position|
            q_in, k_in, v_in = original_inputs[position]
            ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul,
              q_in, k_in, v_in, position.to_i32)
          end
          q_in, k_in, v_in = alternate_inputs[0]
          expected = ML::GGUF::Inkling2Attention.forward(weights, config, cpu, matmul, q_in, k_in, v_in, 21_i32)
            .map { |value| value * Math.sqrt(config.conv_size + 1).to_f32 }
          actual = resident_run_tile(gpu, 21_i32, [alternate_projected[0]], weights, config)
          actual.size.times { |index| (actual[index] - expected[index]).abs.should be < 4e-5_f32 }

          {19_i32, 20_i32, 21_i32}.each_with_index do |position, chronological|
            actual_key, actual_value, actual_position = gpu.key_value_at(position, 22_i32)
            expected_key, expected_value, expected_position = cpu.key_value_at(chronological.to_i32)
            actual_position.should eq position
            expected_position.should eq position
            (actual_key[0] - expected_key[0]).abs.should be < 4e-5_f32
            (actual_value[0] - expected_value[0]).abs.should be < 4e-5_f32
          end
        ensure
          gpu.close
        end
      end

      it "supports a bounded partial speculation span on standalone convolution" do
        convolution = ML::GGUF::Inkling2ResidentConvolution.new(1_i32, 2_i32, rollback_rows: 4_i32)
        taps = [0.65_f32, -0.25_f32]
        inputs = Array(Float32).new(12) { |index| (index % 11 - 5).to_f32 * 0.07_f32 }
        replacement = -0.31_f32

        begin
          resident_run_convolution_tile(convolution, 0_i32, inputs[0, 8], taps)
          expect_raises(ArgumentError) { convolution.begin_speculation(7_i32) }
          convolution.begin_speculation(8_i32)
          4.times do |offset|
            resident_run_convolution_tile(convolution, 8_i32 + offset, [inputs[8 + offset]], taps)
          end
          expect_raises(ArgumentError) { convolution.rewind(11_i32) }
          expect_raises(ArgumentError) do
            resident_run_convolution_tile(convolution, 12_i32, [replacement], taps)
          end
          expect_raises(ArgumentError) { convolution.finish_speculation(7_i32) }
          convolution.finish_speculation(10_i32)

          actual = resident_run_convolution_tile(convolution, 10_i32, [replacement], taps)
          expected = (replacement + inputs[9] * taps[0] + replacement * taps[1]) / Math.sqrt(3_f32).to_f32
          (actual[0] - expected).abs.should be < 2e-6_f32
        ensure
          convolution.close
        end
      end

      it "keeps standalone convolution rewind inside the latest appended tile" do
        convolution = ML::GGUF::Inkling2ResidentConvolution.new(1_i32, 2_i32, rollback_rows: 4_i32)
        taps = [0.65_f32, -0.25_f32]
        inputs = Array(Float32).new(68) { |index| (index % 11 - 5).to_f32 * 0.07_f32 }
        begin
          resident_encode_convolution_tile(convolution, 0_i32, inputs[0, 64], taps)
          resident_encode_convolution_tile(convolution, 64_i32, inputs[64, 4], taps)
          convolution.rewind(65_i32)
          resident_encode_convolution_tile(convolution, 65_i32, [0.31_f32], taps)

          expect_raises(ArgumentError) { convolution.rewind(62_i32) }
        ensure
          convolution.close
        end
      end
    end
  end
{% end %}
