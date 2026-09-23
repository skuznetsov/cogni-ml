require "./spec_helper"
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_weights"

private def qwen_image21_resident_f32_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32)
  raw = Bytes.new(values.size * sizeof(Float32))
  raw.to_unsafe.copy_from(values.to_unsafe.as(Pointer(UInt8)), raw.size)
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::F32, out_dim, in_dim)
end

private def qwen_image21_resident_bf16_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32)
  raw = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    bits = value.unsafe_as(UInt32) >> 16
    raw[index * 2] = (bits & 0xff).to_u8
    raw[index * 2 + 1] = (bits >> 8).to_u8
  end
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::BF16, out_dim, in_dim)
end

private def qwen_image21_resident_matrix(out_dim : Int32, in_dim : Int32, phase : Int32)
  Array(Float32).new(out_dim * in_dim) do |index|
    (((index * 17 + phase * 13) % 29) - 14).to_f32 / 37.0_f32
  end
end

private def qwen_image21_resident_q8_weight(out_dim : Int32, in_dim : Int32)
  raw = Bytes.new(out_dim * (in_dim // 32) * 34)
  out_dim.times do |row|
    (in_dim // 32).times do |block|
      offset = (row * (in_dim // 32) + block) * 34
      scale_bits = case (row + block) % 4
                   when 0 then 0x3800_u16 # IEEE binary16 0.5
                   when 1 then 0x3e00_u16 # IEEE binary16 1.5
                   when 2 then 0x3a00_u16 # IEEE binary16 0.75
                   else        0x4000_u16 # IEEE binary16 2.0
                   end
      raw[offset] = (scale_bits & 0xff).to_u8
      raw[offset + 1] = (scale_bits >> 8).to_u8
      32.times do |column|
        raw[offset + 2 + column] = (((row * 7 + block * 11 + column * 3) % 19) - 9).to_i8.unsafe_as(UInt8)
      end
    end
  end
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::Q8_0, out_dim, in_dim)
end

private def qwen_image21_resident_time_embedding(timesteps : Array(Float32), dim : Int32)
  half = dim // 2
  output = Array(Float32).new(timesteps.size * dim, 0.0_f32)
  timesteps.each_with_index do |timestep, row|
    half.times do |index|
      frequency = Math.exp(-Math.log(10_000.0_f64) * index / half)
      angle = 1000.0_f64 * timestep * frequency
      output[row * dim + index] = Math.cos(angle).to_f32
      output[row * dim + half + index] = Math.sin(angle).to_f32
    end
  end
  output
end

private def qwen_image21_resident_fixture
  config = ML::GGUF::QwenImage21BlockConfig.new(
    hidden_dim: 6,
    heads: 1,
    head_dim: 6,
    intermediate_dim: 5,
    axes_dims: StaticArray[2, 2, 2],
  )
  weights = ML::GGUF::QwenImage21BlockWeights.new(
    qwen_image21_resident_f32_weight(qwen_image21_resident_matrix(6, 6, 1), 6, 6),
    qwen_image21_resident_f32_weight(qwen_image21_resident_matrix(6, 6, 2), 6, 6),
    qwen_image21_resident_f32_weight(qwen_image21_resident_matrix(6, 6, 3), 6, 6),
    qwen_image21_resident_f32_weight(qwen_image21_resident_matrix(6, 6, 4), 6, 6),
    [1.0_f32, 0.9_f32, 1.1_f32, 0.8_f32, 1.2_f32, 0.95_f32],
    [0.85_f32, 1.05_f32, 0.9_f32, 1.15_f32, 0.8_f32, 1.1_f32],
    qwen_image21_resident_f32_weight(qwen_image21_resident_matrix(10, 6, 5), 10, 6),
    qwen_image21_resident_f32_weight(qwen_image21_resident_matrix(6, 5, 6), 6, 5),
  )
  {config, weights}
end

describe ML::GGUF::QwenImage21MetalBlock do
  it "limits automatic Q8 register reuse to validated device and batch policy" do
    q8 = ML::GGUF::QwenImage21MetalQ8
    q8.register_reuse_enabled?("Apple M2 Max", 255, nil).should be_false
    q8.register_reuse_enabled?("Apple M2 Max", 256, nil).should be_true
    q8.register_reuse_enabled?("Apple M3 Max", 513, nil).should be_false
    q8.register_reuse_enabled?("Apple M2 Max", 513, "0").should be_false
    q8.register_reuse_enabled?("Apple M3 Max", 16, "1").should be_true
    q8.register_reuse_enabled?("Apple M2 Max", 256, "true").should be_false

    q8.kernel_name("Apple M2 Max", 256, nil).should eq("qi21_q8_0_register_reuse_matmul")
    q8.kernel_name("Apple M2 Max", 255, nil).should eq("qi21_q8_0_batch_matmul")
    q8.kernel_name("Apple M3 Max", 513, nil).should eq("qi21_q8_0_batch_matmul")
  end

  {% unless flag?(:cpu_only) %}
    it "preserves exact Q8_0 results across batch, output, and block tails" do
      pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
      prior_reuse = ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"]?
      begin
        cases = [{1, 32, 64}, {7, 3, 96}, {8, 32, 64}, {9, 17, 160},
                 {16, 32, 64}, {17, 31, 96}, {32, 32, 64}, {64, 32, 64},
                 {256, 32, 64}]
        cases.each do |batch, out_dim, in_dim|
          weight = qwen_image21_resident_q8_weight(out_dim, in_dim)
          input = Array(Float32).new(batch * weight.in_dim) do |index|
            (((index * 13) % 47) - 23).to_f32 / 37.0_f32
          end
          input_buf = ML::MetalBuffer.from_array(input)
          output_bytes = batch.to_i64 * weight.out_dim * sizeof(Float32)
          reference_buf = ML::MetalBuffer.new(output_bytes)
          baseline_buf = ML::MetalBuffer.new(output_bytes)
          register_reuse_buf = ML::MetalBuffer.new(output_bytes)
          begin
            command = ML::Metal::CommandBuffer.new
            encoder = ML::Metal::ComputeEncoder.new(command)
            ML::GGUF::Qwen35Metal.encode_matmul_to_buffer(encoder, weight, input_buf, reference_buf, batch).should be_true
            ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"] = "0"
            ML::GGUF::QwenImage21MetalQ8.encode_matmul_to_buffer(encoder, weight, input_buf, baseline_buf, batch).should be_true
            ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"] = "1"
            ML::GGUF::QwenImage21MetalQ8.encode_matmul_to_buffer(encoder, weight, input_buf, register_reuse_buf, batch).should be_true
            encoder.end_encoding
            command.commit
            command.wait

            register_reuse_buf.read(batch * weight.out_dim).should eq(baseline_buf.read(batch * weight.out_dim))
            register_reuse_buf.read(batch * weight.out_dim).zip(reference_buf.read(batch * weight.out_dim)).each do |value, reference|
              value.should be_close(reference, 1e-4_f32)
            end
          ensure
            input_buf.release
            reference_buf.release
            baseline_buf.release
            register_reuse_buf.release
          end
        end
      ensure
        if prior_reuse
          ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"] = prior_reuse
        else
          ENV.delete("QWEN_IMAGE21_Q8_REGISTER_REUSE")
        end
      end
    end

    it "preserves Q8_0 NaN propagation across paired output channels" do
      pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
      batch = 9
      weight = qwen_image21_resident_q8_weight(3, 96)
      input = Array(Float32).new(batch * weight.in_dim) do |index|
        (((index * 13) % 47) - 23).to_f32 / 37.0_f32
      end
      input[3 * weight.in_dim + 37] = Float32::NAN
      input_buf = ML::MetalBuffer.from_array(input)
      baseline_buf = ML::MetalBuffer.new(batch.to_i64 * weight.out_dim * sizeof(Float32))
      register_reuse_buf = ML::MetalBuffer.new(batch.to_i64 * weight.out_dim * sizeof(Float32))
      prior_reuse = ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"]?
      begin
        command = ML::Metal::CommandBuffer.new
        encoder = ML::Metal::ComputeEncoder.new(command)
        ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"] = "0"
        ML::GGUF::QwenImage21MetalQ8.encode_matmul_to_buffer(encoder, weight, input_buf, baseline_buf, batch).should be_true
        ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"] = "1"
        ML::GGUF::QwenImage21MetalQ8.encode_matmul_to_buffer(encoder, weight, input_buf, register_reuse_buf, batch).should be_true
        encoder.end_encoding
        command.commit
        command.wait

        baseline = baseline_buf.read(batch * weight.out_dim)
        register_reuse = register_reuse_buf.read(batch * weight.out_dim)
        register_reuse.zip(baseline).each do |value, expected|
          value.nan?.should eq(expected.nan?)
          value.should eq(expected) unless expected.nan?
        end
      ensure
        if prior_reuse
          ENV["QWEN_IMAGE21_Q8_REGISTER_REUSE"] = prior_reuse
        else
          ENV.delete("QWEN_IMAGE21_Q8_REGISTER_REUSE")
        end
        input_buf.release
        baseline_buf.release
        register_reuse_buf.release
      end
    end
  {% end %}

  it "requires an ordered image-only target suffix for the resident cache" do
    weight = qwen_image21_resident_bf16_weight([1.0_f32], 1, 1)
    input = ML::GGUF::QwenImage21MetalResidentInput.new(
      [1.0_f32, 2.0_f32, 3.0_f32], [0.0_f32], [0.0_f32, 1.0_f32],
      [0, -1, -2, -3], [false, false, true, true],
      weight, weight, weight, weight, weight,
    )
    input.image_target_suffix?(2).should be_true
    input.source_rows[2] = 0
    input.image_target_suffix?(2).should be_false
    input.source_rows[2] = -2
    input.target_mask[1] = true
    input.image_target_suffix?(2).should be_false
  end

  it "certifies raw prefix inputs while ignoring changed target and target timestep" do
    _, block = qwen_image21_resident_fixture
    image_weight = qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 2, 1), 6, 2)
    time_weight = qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 6, 2), 6, 6)
    modulation = qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(24, 6, 3), 24, 6)
    scale = qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 6, 4), 6, 6)
    output = qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(2, 6, 5), 2, 6)
    weights = ML::GGUF::QwenImage21TransformerWeights.new(
      image_weight, modulation, scale, output, time_weight, time_weight,
      time_weight, time_weight, Array(Float32).new(6, 1.0_f32), [block],
    )
    layout = ML::GGUF::QwenImage21TransformerCPU.build_layout(
      [false, true, true], [StaticArray[1, 2, 2], StaticArray[1, 2, 2]], 2,
    )
    image = Array(Float32).new(16) { |index| index.to_f32 / 19.0_f32 }
    encoder = Array(Float32).new(12) { |index| index.to_f32 / 11.0_f32 }
    projected_text = Array(Float32).new(12) { |index| index.to_f32 / 13.0_f32 }
    time = qwen_image21_resident_time_embedding([0.25_f32, 0.0_f32], 6)
    source_rows = [0] + (1..8).map { |index| -index }
    input = ML::GGUF::QwenImage21MetalResidentInput.new(
      image, projected_text, time, source_rows, layout.target_token_mask,
      image_weight, time_weight, time_weight, modulation, scale,
    )
    certificate = ML::GGUF::QwenImage21MetalRawPrefixCertificate.new(
      image, encoder, input, layout, weights, 5,
    )
    certificate.compatible?(image, encoder, input, layout, weights, 5).should be_true
    ML::GGUF::QwenImage21MetalRawPrefixCertificate.prefix_closed?(layout, 5).should be_true

    changed_target = image.dup
    changed_target[8] += 1.0_f32
    changed_time = time.dup
    changed_time[0] += 1.0_f32
    changed_input = ML::GGUF::QwenImage21MetalResidentInput.new(
      changed_target, projected_text, changed_time, source_rows,
      layout.target_token_mask, image_weight, time_weight, time_weight,
      modulation, scale,
    )
    certificate.compatible?(changed_target, encoder, changed_input, layout, weights, 5).should be_true

    changed_condition = image.dup
    changed_condition[0] += 1.0_f32
    certificate.compatible?(changed_condition, encoder, input, layout, weights, 5).should be_false
    changed_encoder = encoder.dup
    changed_encoder[0] += 1.0_f32
    certificate.compatible?(image, changed_encoder, input, layout, weights, 5).should be_false
    changed_projected_text = projected_text.dup
    changed_projected_text[0] += 1.0_f32
    changed_text_input = ML::GGUF::QwenImage21MetalResidentInput.new(
      image, changed_projected_text, time, source_rows, layout.target_token_mask,
      image_weight, time_weight, time_weight, modulation, scale,
    )
    certificate.compatible?(image, encoder, changed_text_input, layout, weights, 5).should be_false
    changed_zero = time.dup
    changed_zero[6] += 1.0_f32
    changed_zero_input = ML::GGUF::QwenImage21MetalResidentInput.new(
      image, projected_text, changed_zero, source_rows, layout.target_token_mask,
      image_weight, time_weight, time_weight, modulation, scale,
    )
    certificate.compatible?(image, encoder, changed_zero_input, layout, weights, 5).should be_false
    changed_positions = layout.positions.dup
    changed_positions[0] = StaticArray[9, 9, 9]
    changed_layout = ML::GGUF::QwenImage21TokenLayout.new(
      layout.image_pad_mask, layout.image_ids, layout.target_token_mask,
      changed_positions, layout.key_valid,
    )
    certificate.compatible?(image, encoder, input, changed_layout, weights, 5).should be_false
    colliding_ids = layout.image_ids.dup
    colliding_ids[5] = layout.image_ids[1]
    colliding_layout = ML::GGUF::QwenImage21TokenLayout.new(
      layout.image_pad_mask, colliding_ids, layout.target_token_mask,
      layout.positions, layout.key_valid,
    )
    ML::GGUF::QwenImage21MetalRawPrefixCertificate.prefix_closed?(colliding_layout, 5).should be_false
    replacement_image_weight = qwen_image21_resident_bf16_weight(
      qwen_image21_resident_matrix(6, 2, 1), 6, 2,
    )
    changed_weights = ML::GGUF::QwenImage21TransformerWeights.new(
      replacement_image_weight, modulation, scale, output, time_weight, time_weight,
      time_weight, time_weight, weights.text_norm, [block],
    )
    certificate.compatible?(image, encoder, input, layout, changed_weights, 5).should be_false
  end

  it "keeps every intermediate resident while matching the exact block" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    config, weights = qwen_image21_resident_fixture
    token_count = 4
    hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
      (((index * 11) % 23) - 11).to_f32 / 19.0_f32
    end
    modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
      (((index * 7) % 31) - 15).to_f32 / 101.0_f32
    end
    positions = [
      StaticArray[0, 0, 0],
      StaticArray[1, 1, 1],
      StaticArray[2, -1, 0],
      StaticArray[2, 0, 0],
    ]
    image_ids = [-1, -1, 0, 0]
    key_valid = [true, true, false, true]

    expected = ML::GGUF::QwenImage21BlockCPU.forward(
      hidden, token_count, modulation, positions, image_ids, weights, config,
      key_valid: key_valid,
    )
    actual = ML::GGUF::QwenImage21MetalBlock.forward(
      hidden, token_count, modulation, positions, image_ids, weights, config,
      key_valid: key_valid,
    )

    actual.hidden.zip(expected).each do |value, reference|
      value.should be_close(reference, 3e-4_f32)
    end
    actual.stats.command_buffers.should eq(1)
    actual.stats.projection_dispatches.should eq(6)
    actual.stats.intermediate_readbacks.should eq(0)
    actual.stats.final_readbacks.should eq(1)
  end

  it "keeps a layer stack in one command buffer and one final readback" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    config, weights = qwen_image21_resident_fixture
    token_count = 4
    hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
      (((index * 19) % 31) - 15).to_f32 / 23.0_f32
    end
    modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
      (((index * 5) % 37) - 18).to_f32 / 113.0_f32
    end
    positions = [
      StaticArray[0, 0, 0],
      StaticArray[1, 1, 1],
      StaticArray[2, -1, 0],
      StaticArray[2, 0, 0],
    ]
    image_ids = [-1, -1, 0, 0]

    first = ML::GGUF::QwenImage21BlockCPU.forward(
      hidden, token_count, modulation, positions, image_ids, weights, config
    )
    expected = ML::GGUF::QwenImage21BlockCPU.forward(
      first, token_count, modulation, positions, image_ids, weights, config
    )
    actual = ML::GGUF::QwenImage21MetalBlock.forward_layers(
      hidden, token_count, modulation, positions, image_ids,
      [weights, weights], config,
    )

    actual.hidden.zip(expected).each do |value, reference|
      value.should be_close(reference, 5e-4_f32)
    end
    actual.stats.command_buffers.should eq(1)
    actual.stats.projection_dispatches.should eq(12)
    actual.stats.intermediate_readbacks.should eq(0)
    actual.stats.final_readbacks.should eq(1)
  end

  it "attributes diagnostic GPU phases without changing the block result" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    config, weights = qwen_image21_resident_fixture
    token_count = 4
    hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
      (((index * 19) % 31) - 15).to_f32 / 23.0_f32
    end
    modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
      (((index * 5) % 37) - 18).to_f32 / 113.0_f32
    end
    positions = [StaticArray[0, 0, 0], StaticArray[1, 1, 1],
                 StaticArray[2, -1, 0], StaticArray[2, 0, 0]]
    image_ids = [-1, -1, 0, 0]
    ordinary = ML::GGUF::QwenImage21MetalBlock.forward_layers(
      hidden, token_count, modulation, positions, image_ids, [weights], config,
    )

    prior_profile = ENV["QWEN_IMAGE21_PROFILE"]?
    begin
      ENV["QWEN_IMAGE21_PROFILE"] = "phases"
      profiled = ML::GGUF::QwenImage21MetalBlock.forward_layers(
        hidden, token_count, modulation, positions, image_ids, [weights], config,
      )
      profiled.hidden.zip(ordinary.hidden).each do |value, reference|
        value.should be_close(reference, 1e-5_f32)
      end
      profiled.stats.command_buffers.should be > 1
      phases = profiled.stats.phase_gpu_ms.not_nil!
      phases["q_projection"].should be > 0.0
      phases["k_projection"].should be > 0.0
      phases["v_projection"].should be > 0.0
      phases["attention"].should be > 0.0
      phases["ffn_projection"].should be > 0.0
      phases["cache_copy"]?.should be_nil
      profiled.stats.intermediate_readbacks.should eq(0)
    ensure
      if prior_profile
        ENV["QWEN_IMAGE21_PROFILE"] = prior_profile
      else
        ENV.delete("QWEN_IMAGE21_PROFILE")
      end
    end
  end

  it "preserves resident build and cache-hit outputs in diagnostic phase mode" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    block_config, block_weights = qwen_image21_resident_fixture
    config = ML::GGUF::QwenImage21TransformerConfig.new(2, 2, 6, 6, block_config)
    weights = ML::GGUF::QwenImage21TransformerWeights.new(
      qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 2, 1), 6, 2),
      qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(24, 6, 2), 24, 6),
      qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 6, 3), 6, 6),
      qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(2, 6, 4), 2, 6),
      qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 6, 5), 6, 6),
      qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 6, 6), 6, 6),
      qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 6, 7), 6, 6),
      qwen_image21_resident_bf16_weight(qwen_image21_resident_matrix(6, 6, 8), 6, 6),
      Array(Float32).new(6, 1.0_f32), [block_weights],
    )
    image = Array(Float32).new(16) { |index| (index - 8).to_f32 / 19.0_f32 }
    changed_image = image.dup
    changed_image[8] += 0.125_f32
    encoder = Array(Float32).new(12) { |index| (index - 6).to_f32 / 11.0_f32 }
    shapes = [StaticArray[1, 2, 2], StaticArray[1, 2, 2]]
    mask = [false, true, true]
    backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
    ordinary_stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    profiled_stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    prior_profile = ENV["QWEN_IMAGE21_PROFILE"]?
    begin
      ENV["QWEN_IMAGE21_PROFILE"] = "1"
      ordinary_build = ML::GGUF::QwenImage21TransformerCPU.forward(
        image, encoder, 0.25_f32, shapes, mask, weights, config,
        backend: backend, layer_stack_backend: ordinary_stack,
      )
      ordinary_hit = ML::GGUF::QwenImage21TransformerCPU.forward(
        changed_image, encoder, 0.75_f32, shapes, mask, weights, config,
        backend: backend, layer_stack_backend: ordinary_stack,
      )
      ENV["QWEN_IMAGE21_PROFILE"] = "phases"
      profiled_build = ML::GGUF::QwenImage21TransformerCPU.forward(
        image, encoder, 0.25_f32, shapes, mask, weights, config,
        backend: backend, layer_stack_backend: profiled_stack,
      )
      build_stats = profiled_stack.last_stats.not_nil!
      profiled_hit = ML::GGUF::QwenImage21TransformerCPU.forward(
        changed_image, encoder, 0.75_f32, shapes, mask, weights, config,
        backend: backend, layer_stack_backend: profiled_stack,
      )
      hit_stats = profiled_stack.last_stats.not_nil!
      profiled_build.output.zip(ordinary_build.output).each do |value, reference|
        value.should be_close(reference, 1e-5_f32)
      end
      profiled_hit.output.zip(ordinary_hit.output).each do |value, reference|
        value.should be_close(reference, 1e-5_f32)
      end
      profiled_stack.prefix_cache_builds.should eq(1)
      profiled_stack.prefix_cache_hits.should eq(1)
      build_stats.phase_gpu_ms.not_nil!["attention"].should be > 0.0
      build_stats.phase_gpu_ms.not_nil!["input"].should be > 0.0
      hit_stats.phase_gpu_ms.not_nil!["cache_copy"].should be > 0.0
      hit_stats.image_projection_rows.should eq(4)
      hit_stats.command_buffers.should be > 1
    ensure
      ordinary_stack.close
      profiled_stack.close
      if prior_profile
        ENV["QWEN_IMAGE21_PROFILE"] = prior_profile
      else
        ENV.delete("QWEN_IMAGE21_PROFILE")
      end
    end
  end

  it "keeps the final normalization and BF16 output projection in the resident stack command" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    config, weights = qwen_image21_resident_fixture
    token_count = 4
    hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
      (((index * 19) % 31) - 15).to_f32 / 23.0_f32
    end
    modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
      (((index * 5) % 37) - 18).to_f32 / 113.0_f32
    end
    scales = Array(Float32).new(token_count * config.hidden_dim) do |index|
      (((index * 7) % 19) - 9).to_f32 / 127.0_f32
    end
    positions = [
      StaticArray[0, 0, 0],
      StaticArray[1, 1, 1],
      StaticArray[2, -1, 0],
      StaticArray[2, 0, 0],
    ]
    image_ids = [-1, -1, 0, 0]
    output_dim = 4
    output_weight = qwen_image21_resident_bf16_weight(
      qwen_image21_resident_matrix(output_dim, config.hidden_dim, 7),
      output_dim,
      config.hidden_dim,
    )

    expected_hidden = [weights, weights].reduce(hidden) do |state, layer|
      ML::GGUF::QwenImage21BlockCPU.forward(
        state, token_count, modulation, positions, image_ids, layer, config
      )
    end
    normalized = expected_hidden.dup
    ML::GGUF::F32Backend.new.layer_norm!(
      normalized,
      token_count,
      config.hidden_dim,
      Array(Float32).new(config.hidden_dim, 1.0_f32),
      Array(Float32).new(config.hidden_dim, 0.0_f32),
    )
    normalized.size.times { |index| normalized[index] *= 1.0_f32 + scales[index] }
    expected = ML::GGUF::F32Backend.new.matmul(
      normalized, token_count, output_weight, Array(Float32).new(output_dim, 0.0_f32)
    )

    actual = ML::GGUF::QwenImage21MetalBlock.forward_layers_projected(
      hidden, token_count, modulation, positions, image_ids,
      [weights, weights], config, scales, output_weight,
    )

    actual.output.zip(expected).each do |value, reference|
      value.should be_close(reference, 7e-4_f32)
    end
    actual.stats.command_buffers.should eq(1)
    actual.stats.projection_dispatches.should eq(13)
    actual.stats.intermediate_readbacks.should eq(0)
    actual.stats.final_readbacks.should eq(1)
  end

  it "reuses cached prefix K/V while recomputing only a changed target suffix" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    config, weights = qwen_image21_resident_fixture
    token_count = 4
    prefix_tokens = 2
    hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
      (((index * 19) % 31) - 15).to_f32 / 23.0_f32
    end
    changed = hidden.dup
    (prefix_tokens * config.hidden_dim...changed.size).each do |index|
      changed[index] += ((index % 5) - 2).to_f32 / 17.0_f32
    end
    modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
      (((index * 5) % 37) - 18).to_f32 / 113.0_f32
    end
    changed_modulation = modulation.dup
    (prefix_tokens * 4 * config.hidden_dim...changed_modulation.size).each do |index|
      changed_modulation[index] += ((index % 7) - 3).to_f32 / 211.0_f32
    end
    positions = [
      StaticArray[0, 0, 0],
      StaticArray[1, 1, 1],
      StaticArray[2, -1, 0],
      StaticArray[2, 0, 0],
    ]
    image_ids = [-1, -1, 0, 0]
    layers = [weights, weights]

    stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    begin
      stack.forward_layers(
        hidden, token_count, modulation, positions, image_ids,
        layers, config, nil, prefix_tokens,
      )
      actual = stack.forward_layers(
        changed, token_count, changed_modulation, positions, image_ids,
        layers, config, nil, prefix_tokens,
      )
      expected = layers.reduce(changed) do |state, layer|
        ML::GGUF::QwenImage21BlockCPU.forward(
          state, token_count, changed_modulation, positions, image_ids, layer, config
        )
      end

      actual.zip(expected).each do |value, reference|
        value.should be_close(reference, 8e-4_f32)
      end
      stack.prefix_cache_builds.should eq(1)
      stack.prefix_cache_hits.should eq(1)
      stack.last_stats.not_nil!.active_tokens.should eq(token_count - prefix_tokens)

      changed_prefix = changed.dup
      changed_prefix[0] += 0.25_f32
      rebuilt = stack.forward_layers(
        changed_prefix, token_count, changed_modulation, positions, image_ids,
        layers, config, nil, prefix_tokens,
      )
      rebuilt_expected = layers.reduce(changed_prefix) do |state, layer|
        ML::GGUF::QwenImage21BlockCPU.forward(
          state, token_count, changed_modulation, positions, image_ids, layer, config
        )
      end
      rebuilt.zip(rebuilt_expected).each do |value, reference|
        value.should be_close(reference, 8e-4_f32)
      end
      stack.prefix_cache_builds.should eq(2)
      stack.prefix_cache_hits.should eq(1)
      stack.last_stats.not_nil!.active_tokens.should eq(token_count)
    ensure
      stack.close
    end
  end

  it "preserves mixed text and image attention with a masked key across cache rebuilds and hits" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    config, weights = qwen_image21_resident_fixture
    token_count = 6
    prefix_tokens = 4
    hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
      (((index * 23 + 7) % 43) - 21).to_f32 / 31.0_f32
    end
    changed_target = hidden.dup
    (prefix_tokens * config.hidden_dim...changed_target.size).each do |index|
      changed_target[index] += ((index % 7) - 3).to_f32 / 29.0_f32
    end
    changed_prefix = changed_target.dup
    changed_prefix[0] += 0.25_f32
    modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
      (((index * 11 + 3) % 47) - 23).to_f32 / 131.0_f32
    end
    changed_modulation = modulation.dup
    (prefix_tokens * 4 * config.hidden_dim...changed_modulation.size).each do |index|
      changed_modulation[index] += ((index % 5) - 2).to_f32 / 173.0_f32
    end
    positions = Array(StaticArray(Int32, 3)).new(token_count) do |index|
      StaticArray[index, index % 3, index % 2]
    end
    image_ids = [-1, -1, 0, 0, 1, 1]
    key_valid = [true, false, true, false, true, true]
    layers = [weights, weights]
    stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    begin
      built = stack.forward_layers(
        hidden, token_count, modulation, positions, image_ids,
        layers, config, key_valid, prefix_tokens,
      )
      built_expected = layers.reduce(hidden) do |state, layer|
        ML::GGUF::QwenImage21BlockCPU.forward(
          state, token_count, modulation, positions, image_ids, layer, config,
          key_valid: key_valid,
        )
      end
      built.zip(built_expected).each do |value, reference|
        value.should be_close(reference, 8e-4_f32)
      end
      stack.last_stats.not_nil!.command_buffers.should eq(1)
      stack.last_stats.not_nil!.intermediate_readbacks.should eq(0)

      hit = stack.forward_layers(
        changed_target, token_count, changed_modulation, positions, image_ids,
        layers, config, key_valid, prefix_tokens,
      )
      hit_expected = layers.reduce(changed_target) do |state, layer|
        ML::GGUF::QwenImage21BlockCPU.forward(
          state, token_count, changed_modulation, positions, image_ids, layer, config,
          key_valid: key_valid,
        )
      end
      hit.zip(hit_expected).each do |value, reference|
        value.should be_close(reference, 8e-4_f32)
      end
      stack.prefix_cache_builds.should eq(1)
      stack.prefix_cache_hits.should eq(1)
      stack.last_stats.not_nil!.active_tokens.should eq(token_count - prefix_tokens)
      stack.last_stats.not_nil!.command_buffers.should eq(1)
      stack.last_stats.not_nil!.intermediate_readbacks.should eq(0)

      rebuilt = stack.forward_layers(
        changed_prefix, token_count, changed_modulation, positions, image_ids,
        layers, config, key_valid, prefix_tokens,
      )
      rebuilt_expected = layers.reduce(changed_prefix) do |state, layer|
        ML::GGUF::QwenImage21BlockCPU.forward(
          state, token_count, changed_modulation, positions, image_ids, layer, config,
          key_valid: key_valid,
        )
      end
      rebuilt.zip(rebuilt_expected).each do |value, reference|
        value.should be_close(reference, 8e-4_f32)
      end
      stack.prefix_cache_builds.should eq(2)
      stack.prefix_cache_hits.should eq(1)
      stack.last_stats.not_nil!.active_tokens.should eq(token_count)
      stack.last_stats.not_nil!.command_buffers.should eq(1)
      stack.last_stats.not_nil!.intermediate_readbacks.should eq(0)
    ensure
      stack.close
    end
  end

  it "rejects a key mask that leaves an attention row without valid keys" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    config, weights = qwen_image21_resident_fixture

    expect_raises(ArgumentError, /attention row has no valid keys/) do
      ML::GGUF::QwenImage21MetalBlock.forward(
        Array(Float32).new(config.hidden_dim, 0.0_f32),
        1,
        Array(Float32).new(4 * config.hidden_dim, 0.0_f32),
        [StaticArray[0, 0, 0]],
        [-1],
        weights,
        config,
        key_valid: [false],
      )
    end
  end

  it "matches the hybrid reference for one real mixed-quant block" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed resident check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?

    model = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = model.block_config
      token_count = 4
      hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
        (((index * 17 + 11) % 257) - 128).to_f32 / 193.0_f32
      end
      modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
        (((index * 13 + 7) % 101) - 50).to_f32 / 401.0_f32
      end
      positions = [
        StaticArray[0, 0, 0],
        StaticArray[1, 1, 1],
        StaticArray[2, -1, 0],
        StaticArray[2, 0, 0],
      ]
      image_ids = [-1, -1, 0, 0]
      key_valid = [true, true, false, true]
      projection_backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)

      expected = ML::GGUF::QwenImage21BlockCPU.forward(
        hidden, token_count, modulation, positions, image_ids,
        model.layers[0], config,
        key_valid: key_valid,
        backend: projection_backend,
      )
      actual = ML::GGUF::QwenImage21MetalBlock.forward(
        hidden, token_count, modulation, positions, image_ids,
        model.layers[0], config,
        key_valid: key_valid,
      )

      max_abs = 0.0_f64
      dot = 0.0_f64
      expected_norm = 0.0_f64
      actual_norm = 0.0_f64
      expected.each_with_index do |reference, index|
        value = actual.hidden[index]
        max_abs = Math.max(max_abs, (reference - value).abs)
        dot += reference.to_f64 * value
        expected_norm += reference.to_f64 ** 2
        actual_norm += value.to_f64 ** 2
      end
      cosine = dot / Math.sqrt(expected_norm * actual_norm)
      STDERR.puts "qwen_image21_resident_block_parity max_abs=#{max_abs} cosine=#{cosine}"

      projection_backend.metal_projection_count.should eq(6)
      actual.stats.command_buffers.should eq(1)
      actual.stats.intermediate_readbacks.should eq(0)
      actual.hidden.all?(&.finite?).should be_true
      max_abs.should be < 2.0e-3
      cosine.should be > 0.999999
    ensure
      model.close
    end
  end

  it "preserves a real mixed-quant block across the Q8_0 batch route" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed Q8_0 batch check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?

    model = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    prior_route = ENV["QWEN_IMAGE21_Q8_BATCH"]?
    begin
      config = model.block_config
      tokens = 17
      hidden = Array(Float32).new(tokens * config.hidden_dim) do |index|
        (((index * 17 + 11) % 257) - 128).to_f32 / 193.0_f32
      end
      modulation = Array(Float32).new(tokens * 4 * config.hidden_dim) do |index|
        (((index * 13 + 7) % 101) - 50).to_f32 / 401.0_f32
      end
      positions = Array(StaticArray(Int32, 3)).new(tokens) { |index| StaticArray[0, index, 0] }
      image_ids = Array(Int32).new(tokens, 0)
      ENV["QWEN_IMAGE21_Q8_BATCH"] = "0"
      reference = ML::GGUF::QwenImage21MetalBlock.forward(
        hidden, tokens, modulation, positions, image_ids, model.layers[0], config,
      )
      ENV.delete("QWEN_IMAGE21_Q8_BATCH")
      candidate = ML::GGUF::QwenImage21MetalBlock.forward(
        hidden, tokens, modulation, positions, image_ids, model.layers[0], config,
      )
      candidate.hidden.zip(reference.hidden).each do |value, expected|
        value.should be_close(expected, 1e-4_f32)
      end
      candidate.stats.command_buffers.should eq(1)
      candidate.stats.intermediate_readbacks.should eq(0)
    ensure
      if prior_route
        ENV["QWEN_IMAGE21_Q8_BATCH"] = prior_route
      else
        ENV.delete("QWEN_IMAGE21_Q8_BATCH")
      end
      model.close
    end
  end

  it "executes the complete real layer stack with one command buffer and readback" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed resident stack check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?

    model = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = model.transformer_config
      hidden = Array(Float32).new(4 * config.input_dim) do |index|
        (((index * 23 + 5) % 97) - 48).to_f32 / 127.0_f32
      end
      reference_backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
      expected = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden,
        [] of Float32,
        0.5_f32,
        [StaticArray[1, 2, 2]],
        [true],
        model.transformer_weights,
        config,
        backend: reference_backend,
      )

      outer_backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
      stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
      actual = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden,
        [] of Float32,
        0.5_f32,
        [StaticArray[1, 2, 2]],
        [true],
        model.transformer_weights,
        config,
        backend: outer_backend,
        layer_stack_backend: stack,
      )

      max_abs = 0.0_f64
      dot = 0.0_f64
      expected_norm = 0.0_f64
      actual_norm = 0.0_f64
      expected.output.each_with_index do |reference, index|
        value = actual.output[index]
        max_abs = Math.max(max_abs, (reference - value).abs)
        dot += reference.to_f64 * value
        expected_norm += reference.to_f64 ** 2
        actual_norm += value.to_f64 ** 2
      end
      cosine = dot / Math.sqrt(expected_norm * actual_norm)
      stats = stack.last_stats.not_nil!
      STDERR.puts "qwen_image21_resident_stack_parity max_abs=#{max_abs} cosine=#{cosine}"

      stack.invocations.should eq(1)
      stack.resident_head_invocations.should eq(1)
      stack.resident_input_invocations.should eq(1)
      stats.command_buffers.should eq(1)
      stats.projection_dispatches.should eq(model.layers.size * 6 + 6)
      stats.intermediate_readbacks.should eq(0)
      stats.final_readbacks.should eq(1)
      actual.output.all?(&.finite?).should be_true
      max_abs.should be < 5.0e-2
      cosine.should be > 0.99999
      stack.close
    ensure
      model.close
    end
  end

  it "assembles mixed image and text inputs and timestep rows on the resident path" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed resident input check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?

    model = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    begin
      original = model.transformer_config
      config = ML::GGUF::QwenImage21TransformerConfig.new(
        original.input_dim, original.output_dim, original.context_dim,
        original.time_input_dim, original.block, false,
      )
      hidden = Array(Float32).new(8 * config.input_dim) do |index|
        (((index * 23 + 5) % 97) - 48).to_f32 / 127.0_f32
      end
      encoder = Array(Float32).new(2 * config.context_dim) do |index|
        (((index * 31 + 9) % 127) - 63).to_f32 / 173.0_f32
      end
      shapes = [StaticArray[1, 2, 2], StaticArray[1, 2, 2]]
      mask = [true, false, true]
      expected = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden, encoder, 0.625_f32, shapes, mask,
        model.transformer_weights, config,
        backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
      )
      legacy_stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
      legacy = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden, encoder, 0.625_f32, shapes, mask,
        model.transformer_weights, config,
        backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(
          strict: false, resident_input: false,
        ),
        layer_stack_backend: legacy_stack,
      )
      legacy_stack.close
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
      actual = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden, encoder, 0.625_f32, shapes, mask,
        model.transformer_weights, config,
        backend: backend, layer_stack_backend: stack,
      )

      max_abs = expected.output.zip(actual.output).max_of { |reference, value| (reference - value).abs }
      legacy_abs = legacy.output.zip(actual.output).max_of { |reference, value| (reference - value).abs }
      STDERR.puts "qwen_image21_resident_input_parity max_abs=#{max_abs}"
      legacy_abs.should be < 1.0e-5
      stack.resident_input_invocations.should eq(1)
      stack.last_stats.not_nil!.projection_dispatches.should eq(model.layers.size * 6 + 6)
      stack.last_stats.not_nil!.command_buffers.should eq(1)
      stack.last_stats.not_nil!.intermediate_readbacks.should eq(0)
      stack.last_stats.not_nil!.final_readbacks.should eq(1)
      max_abs.should be < 1.0e-2
    ensure
      stack.close
      model.close
    end
  end

  it "matches one real block on mixed image and text attention" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed mixed block check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    model = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = model.transformer_config.block
      layout = ML::GGUF::QwenImage21TransformerCPU.build_layout(
        [true, false, true], [StaticArray[1, 2, 2], StaticArray[1, 2, 2]], 2,
      )
      hidden = Array(Float32).new(layout.token_count * config.hidden_dim) do |index|
        (((index * 19 + 7) % 101) - 50).to_f32 / 151.0_f32
      end
      modulation = Array(Float32).new(layout.token_count * 4 * config.hidden_dim) do |index|
        (((index * 7 + 3) % 67) - 33).to_f32 / 301.0_f32
      end
      expected = ML::GGUF::QwenImage21BlockCPU.forward(
        hidden, layout.token_count, modulation, layout.positions, layout.image_ids,
        model.layers[0], config, key_valid: layout.key_valid,
        backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
      )
      f32_expected = ML::GGUF::QwenImage21BlockCPU.forward(
        hidden, layout.token_count, modulation, layout.positions, layout.image_ids,
        model.layers[0], config, key_valid: layout.key_valid)
      backend_abs = expected.zip(f32_expected).max_of { |reference, value| (reference - value).abs }
      actual = ML::GGUF::QwenImage21MetalBlock.forward(
        hidden, layout.token_count, modulation, layout.positions, layout.image_ids,
        model.layers[0], config, key_valid: layout.key_valid)
      max_abs = expected.zip(actual.hidden).max_of { |reference, value| (reference - value).abs }
      STDERR.puts "qwen_image21_mixed_block_parity max_abs=#{max_abs} backend_abs=#{backend_abs}"
      backend_abs.should be < 0.25
      max_abs.should be < 0.25
    ensure
      model.close
    end
  end

  it "selects the causal zero-timestep prefix row on the GPU" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed timestep row check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?
    model = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    begin
      config = model.transformer_config
      image_input = Array(Float32).new(8 * config.input_dim) do |index|
        (((index * 19 + 3) % 101) - 50).to_f32 / 149.0_f32
      end
      encoder = Array(Float32).new(config.context_dim, 0.0_f32)
      shapes = [StaticArray[1, 2, 2], StaticArray[1, 2, 2]]
      mask = [true, true]
      layout = ML::GGUF::QwenImage21TransformerCPU.build_layout(mask, shapes, 1)
      expected = ML::GGUF::QwenImage21TransformerCPU.forward(
        image_input, encoder, 0.75_f32, shapes, mask,
        model.transformer_weights, config,
        backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
      )
      actual = stack.forward_resident_input(
        image_input, Array(Float32).new(config.hidden_dim, 0.0_f32),
        qwen_image21_resident_time_embedding([0.75_f32, 0.0_f32], config.time_input_dim),
        mask, layout, model.transformer_weights, config,
      ).not_nil!
      max_abs = expected.output.zip(actual).max_of { |reference, value| (reference - value).abs }
      STDERR.puts "qwen_image21_resident_causal_rows_parity max_abs=#{max_abs}"
      stack.resident_input_invocations.should eq(1)
      stack.last_stats.not_nil!.intermediate_readbacks.should eq(0)
      max_abs.should be < 1.0e-2
    ensure
      stack.close
      model.close
    end
  end

  it "reuses a real 32-layer text prefix across changed target and timestep inputs" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed prefix cache check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?

    model = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    begin
      config = model.transformer_config
      hidden = Array(Float32).new(4 * config.input_dim) do |index|
        (((index * 29 + 3) % 113) - 56).to_f32 / 149.0_f32
      end
      changed = hidden.map_with_index do |value, index|
        value + (((index * 7) % 17) - 8).to_f32 / 997.0_f32
      end
      encoder_hidden = Array(Float32).new(config.context_dim) do |index|
        (((index * 31 + 9) % 127) - 63).to_f32 / 173.0_f32
      end
      shapes = [StaticArray[1, 2, 2]]
      mask = [false, true]
      outer_backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)

      ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden,
        encoder_hidden,
        0.25_f32,
        shapes,
        mask,
        model.transformer_weights,
        config,
        backend: outer_backend,
        layer_stack_backend: stack,
      )
      expected = ML::GGUF::QwenImage21TransformerCPU.forward(
        changed,
        encoder_hidden,
        0.75_f32,
        shapes,
        mask,
        model.transformer_weights,
        config,
        backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
      )
      actual = ML::GGUF::QwenImage21TransformerCPU.forward(
        changed,
        encoder_hidden,
        0.75_f32,
        shapes,
        mask,
        model.transformer_weights,
        config,
        backend: outer_backend,
        layer_stack_backend: stack,
      )

      max_abs = 0.0_f64
      dot = 0.0_f64
      expected_norm = 0.0_f64
      actual_norm = 0.0_f64
      expected.output.each_with_index do |reference, index|
        value = actual.output[index]
        max_abs = Math.max(max_abs, (reference - value).abs)
        dot += reference.to_f64 * value
        expected_norm += reference.to_f64 ** 2
        actual_norm += value.to_f64 ** 2
      end
      cosine = dot / Math.sqrt(expected_norm * actual_norm)
      stats = stack.last_stats.not_nil!
      STDERR.puts "qwen_image21_prefix_cache_parity max_abs=#{max_abs} cosine=#{cosine}"

      stack.prefix_cache_builds.should eq(1)
      stack.prefix_cache_hits.should eq(1)
      stack.resident_head_invocations.should eq(2)
      stack.resident_input_invocations.should eq(2)
      stats.active_tokens.should eq(4)
      stats.command_buffers.should eq(1)
      stats.projection_dispatches.should eq(model.layers.size * 6 + 6)
      stats.intermediate_readbacks.should eq(0)
      stats.final_readbacks.should eq(1)
      actual.output.all?(&.finite?).should be_true
      max_abs.should be < 5.0e-2
      cosine.should be > 0.99999

      changed_encoder = encoder_hidden.dup
      changed_encoder[0] += 0.125_f32
      ML::GGUF::QwenImage21TransformerCPU.forward(
        changed, changed_encoder, 0.75_f32, shapes, mask,
        model.transformer_weights, config,
        backend: outer_backend, layer_stack_backend: stack,
      )
      stack.prefix_cache_builds.should eq(2)
      stack.prefix_cache_hits.should eq(1)
      stack.resident_input_invocations.should eq(3)
    ensure
      stack.close
      model.close
    end
  end

  it "invalidates a resident prefix when condition-image latents change" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the condition-image cache check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalBlock.available?

    model = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    full_stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    begin
      config = model.transformer_config
      hidden = Array(Float32).new(8 * config.input_dim) do |index|
        (((index * 19 + 3) % 101) - 50).to_f32 / 149.0_f32
      end
      changed_target = hidden.dup
      changed_target[4 * config.input_dim] += 0.125_f32
      changed_condition = changed_target.dup
      changed_condition[0] += 0.125_f32
      encoder = Array(Float32).new(2 * config.context_dim) do |index|
        (((index * 31 + 9) % 127) - 63).to_f32 / 173.0_f32
      end
      shapes = [StaticArray[1, 2, 2], StaticArray[1, 2, 2]]
      mask = [false, true, true]
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)

      ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden, encoder, 0.25_f32, shapes, mask, model.transformer_weights, config,
        backend: backend, layer_stack_backend: stack,
      )
      stack.last_stats.not_nil!.image_projection_rows.should eq(8)
      expected = ML::GGUF::QwenImage21TransformerCPU.forward(
        changed_target, encoder, 0.75_f32, shapes, mask,
        model.transformer_weights, config,
        backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false),
      )
      full_resident = ML::GGUF::QwenImage21TransformerCPU.forward(
        changed_target, encoder, 0.75_f32, shapes, mask,
        model.transformer_weights, config,
        backend: backend, layer_stack_backend: full_stack,
      )
      actual = ML::GGUF::QwenImage21TransformerCPU.forward(
        changed_target, encoder, 0.75_f32, shapes, mask,
        model.transformer_weights, config,
        backend: backend, layer_stack_backend: stack,
      )
      stack.prefix_cache_builds.should eq(1)
      stack.prefix_cache_hits.should eq(1)
      stack.last_stats.not_nil!.active_tokens.should eq(4)
      stack.last_stats.not_nil!.image_projection_rows.should eq(4)
      actual.output.zip(expected.output).max_of { |value, reference| (value - reference).abs }.should be < 5.0e-2
      # Nine full rows use the GGUF GEMM route by default; four cached rows use GEMV.
      # Raising QWEN35_GEMM_BATCH_THRESHOLD to 16 makes this comparison exact.
      same_matmul_route = (ENV["QWEN35_GEMM_BATCH_THRESHOLD"]?.try(&.to_i?) || 8) >= actual.layout.token_count
      parity_limit = same_matmul_route ? 1.0e-4 : 5.0e-3
      actual.output.zip(full_resident.output).max_of { |value, reference| (value - reference).abs }.should be < parity_limit

      ML::GGUF::QwenImage21TransformerCPU.forward(
        changed_condition, encoder, 0.75_f32, shapes, mask,
        model.transformer_weights, config,
        backend: backend, layer_stack_backend: stack,
      )
      stack.prefix_cache_builds.should eq(2)
      stack.prefix_cache_hits.should eq(1)
      stack.last_stats.not_nil!.image_projection_rows.should eq(8)
      stack.resident_input_invocations.should eq(3)
    ensure
      full_stack.close
      stack.close
      model.close
    end
  end
end
