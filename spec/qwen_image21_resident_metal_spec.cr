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
      stack.resident_input_invocations.should eq(0)
      stats.active_tokens.should eq(4)
      stats.command_buffers.should eq(1)
      stats.projection_dispatches.should eq(model.layers.size * 6 + 1)
      stats.intermediate_readbacks.should eq(0)
      stats.final_readbacks.should eq(1)
      actual.output.all?(&.finite?).should be_true
      max_abs.should be < 5.0e-2
      cosine.should be > 0.99999
    ensure
      stack.close
      model.close
    end
  end
end
