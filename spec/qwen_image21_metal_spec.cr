require "./spec_helper"
require "../src/ml/gguf/qwen_image21_weights"
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_flow_match"
require "../src/ml/gguf/qwen_image21_conditioning_bundle"

private def qwen_image21_metal_bf16_weight(values : Array(Float32), out_dim : Int32, in_dim : Int32)
  raw = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    bits = value.unsafe_as(UInt32) >> 16
    raw[index * 2] = (bits & 0xff).to_u8
    raw[index * 2 + 1] = (bits >> 8).to_u8
  end
  ML::GGUF::QuantWeight.new(raw, ML::GGUF::TensorType::BF16, out_dim, in_dim)
end

private def qwen_image21_metal_normalize_text(
  input : Array(Float32), rows : Int32, dim : Int32,
  weight : Array(Float32), eps : Float32,
) : Array(Float32)
  output = Array(Float32).new(input.size, 0.0_f32)
  rows.times do |row|
    offset = row * dim
    mean_square = 0.0_f64
    dim.times { |column| mean_square += input[offset + column].to_f64 ** 2 }
    inv_rms = 1.0_f64 / Math.sqrt(mean_square / dim + eps)
    dim.times do |column|
      output[offset + column] = (
        input[offset + column] * inv_rms * (weight[column] + 1.0_f32)
      ).to_f32
    end
  end
  output
end

# Isolates the new top-level BF16 route while retaining the already-verified
# mixed-quant Metal projections inside each transformer block.
private class QwenImage21CPUReferenceBF16Backend
  include ML::GGUF::ComputeBackend

  def initialize
    @cpu = ML::GGUF::F32Backend.new
    @metal = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
  end

  def matmul(x : Array(Float32), rows : Int32, qw : ML::GGUF::QuantWeight,
             bias : Array(Float32)) : Array(Float32)
    if qw.type.bf16?
      @cpu.matmul(x, rows, qw, bias)
    else
      @metal.matmul(x, rows, qw, bias)
    end
  end

  def layer_norm!(x : Array(Float32), n_pos : Int32, dim : Int32,
                  w : Array(Float32), b : Array(Float32)) : Nil
    @cpu.layer_norm!(x, n_pos, dim, w, b)
  end

  def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32) : Nil
    @cpu.softmax_row!(scores, offset, len)
  end

  def gelu(x : Float32) : Float32
    @cpu.gelu(x)
  end

  def dot(a : Array(Float32), a_off : Int32, b : Array(Float32),
          b_off : Int32, len : Int32) : Float32
    @cpu.dot(a, a_off, b, b_off, len)
  end
end

describe ML::GGUF::QwenImage21MetalProjectionBackend do
  it "matches the CPU reference for a BF16 batch projection" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?
    weight = qwen_image21_metal_bf16_weight(
      Array(Float32).new(20) { |index| (((index * 7) % 13) - 6).to_f32 / 5.0_f32 },
      4,
      5,
    )
    input = Array(Float32).new(15) do |index|
      (((index * 11) % 17) - 8).to_f32 / 7.0_f32
    end
    bias = [0.25_f32, -0.5_f32, 0.75_f32, -1.0_f32]
    expected = ML::GGUF::F32Backend.new.matmul(input, 3, weight, bias)
    backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)

    actual = backend.matmul(input, 3, weight, bias)

    actual.zip(expected).each do |value, reference|
      value.should be_close(reference, 1e-5_f32)
    end
    backend.metal_projection_count.should eq(1)
    backend.bf16_projection_count.should eq(1)
  end

  it "keeps chained BF16 text and timestep projections in one command each" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?
    cpu = ML::GGUF::F32Backend.new
    backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
    weight = ->(out_dim : Int32, in_dim : Int32, phase : Int32) do
      qwen_image21_metal_bf16_weight(
        Array(Float32).new(out_dim * in_dim) do |index|
          (((index * 7 + phase * 5) % 23) - 11).to_f32 / 13.0_f32
        end,
        out_dim,
        in_dim,
      )
    end
    rows = 2
    input = Array(Float32).new(rows * 3) do |index|
      (((index * 11) % 17) - 8).to_f32 / 9.0_f32
    end
    first = weight.call(4, 3, 1)
    second = weight.call(4, 4, 2)
    modulation = weight.call(8, 4, 3)
    scale = weight.call(4, 4, 4)

    text_hidden = cpu.matmul(input, rows, first, Array(Float32).new(4, 0.0_f32))
    text_hidden.map! { |value| cpu.gelu(value) }
    expected_text = cpu.matmul(text_hidden, rows, second, Array(Float32).new(4, 0.0_f32))
    actual_text = backend.project_text_layers(input, rows, first, second).not_nil!

    time_hidden = cpu.matmul(input, rows, first, Array(Float32).new(4, 0.0_f32))
    time_hidden.map! { |value| value / (1.0_f32 + Math.exp(-value)) }
    time_hidden = cpu.matmul(time_hidden, rows, second, Array(Float32).new(4, 0.0_f32))
    time_hidden.map! { |value| value / (1.0_f32 + Math.exp(-value)) }
    expected_modulation = cpu.matmul(
      time_hidden, rows, modulation, Array(Float32).new(8, 0.0_f32)
    )
    expected_scale = cpu.matmul(time_hidden, rows, scale, Array(Float32).new(4, 0.0_f32))
    actual_time = backend.project_timestep_layers(
      input, rows, first, second, modulation, scale
    ).not_nil!

    actual_text.zip(expected_text).each do |value, reference|
      value.should be_close(reference, 1e-4_f32)
    end
    actual_time[0].zip(expected_modulation).each do |value, reference|
      value.should be_close(reference, 1e-4_f32)
    end
    actual_time[1].zip(expected_scale).each do |value, reference|
      value.should be_close(reference, 1e-4_f32)
    end
    backend.metal_projection_count.should eq(6)
    backend.bf16_projection_count.should eq(6)
    backend.fused_outer_command_count.should eq(2)
  end

  it "keeps finite saturated GELU activations finite in the fused BF16 text chain" do
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?
    first = qwen_image21_metal_bf16_weight([12.0_f32, -12.0_f32, 20.0_f32, -20.0_f32], 4, 1)
    identity = qwen_image21_metal_bf16_weight(
      [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32,
       0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32,
       0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32,
       0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32],
      4,
      4,
    )
    expected = [12.0_f32, 0.0_f32, 20.0_f32, 0.0_f32]

    actual = ML::GGUF::QwenImage21MetalBF16.project_text_layers(
      [1.0_f32], 1, first, identity
    ).not_nil!

    actual.count(&.finite?).should eq(actual.size)
    actual.zip(expected).each do |value, reference|
      value.should be_close(reference, 1e-4_f32)
    end
  end

  it "keeps fused real Qwen3-VL text projection finite against the separated Metal route" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    bundle_path = ENV["QWEN_IMAGE21_CONDITIONING"]?
    pending!("set QWEN_IMAGE21_GGUF and QWEN_IMAGE21_CONDITIONING for real text projection") unless path && File.file?(path) && bundle_path && File.file?(bundle_path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    conditioning = ML::GGUF::QwenImage21ConditioningBundle.load(bundle_path.not_nil!)
    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.transformer_config
      rows = conditioning.encoder_hidden_states.size // config.context_dim
      normalized = qwen_image21_metal_normalize_text(
        conditioning.encoder_hidden_states, rows, config.context_dim,
        weights.text_norm, config.block.eps,
      )
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
      first = backend.matmul(
        normalized, rows, weights.text_in_layer,
        Array(Float32).new(config.hidden_dim, 0.0_f32),
      )
      cpu = ML::GGUF::F32Backend.new
      first.map! { |value| cpu.gelu(value) }
      separated = backend.matmul(
        first, rows, weights.text_out_layer,
        Array(Float32).new(config.hidden_dim, 0.0_f32),
      )
      fused = ML::GGUF::QwenImage21MetalBF16.project_text_layers(
        normalized, rows, weights.text_in_layer, weights.text_out_layer,
      ).not_nil!

      fused_finite = fused.count(&.finite?)
      separated_finite = separated.count(&.finite?)
      max_abs = 0.0_f64
      if fused_finite == fused.size && separated_finite == separated.size
        fused.zip(separated).each do |actual, expected|
          max_abs = Math.max(max_abs, (actual - expected).abs)
        end
      end
      fused.size.should eq(rows * config.hidden_dim)
      separated.size.should eq(fused.size)
      separated_finite.should eq(separated.size)
      fused_finite.should eq(fused.size)
      max_abs.should be < 1.0e-2
    ensure
      weights.close
    end
  end

  it "matches the CPU reference for one real mixed-quant transformer block" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed parity check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.block_config
      token_count = 2
      hidden = Array(Float32).new(token_count * config.hidden_dim) do |index|
        (((index * 17 + 11) % 257) - 128).to_f32 / 193.0_f32
      end
      modulation = Array(Float32).new(token_count * 4 * config.hidden_dim) do |index|
        (((index * 13 + 7) % 101) - 50).to_f32 / 401.0_f32
      end
      positions = [StaticArray[0, 0, 0], StaticArray[1, 1, 1]]
      image_ids = [-1, 0]

      cpu = ML::GGUF::QwenImage21BlockCPU.forward(
        hidden, token_count, modulation, positions, image_ids,
        weights.layers[0], config,
      )
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true)
      metal = ML::GGUF::QwenImage21BlockCPU.forward(
        hidden, token_count, modulation, positions, image_ids,
        weights.layers[0], config,
        backend: backend,
      )

      max_abs = 0.0_f64
      dot = 0.0_f64
      cpu_norm = 0.0_f64
      metal_norm = 0.0_f64
      cpu.each_with_index do |expected, index|
        actual = metal[index]
        max_abs = Math.max(max_abs, (expected - actual).abs)
        dot += expected.to_f64 * actual
        cpu_norm += expected.to_f64 ** 2
        metal_norm += actual.to_f64 ** 2
      end
      cosine = dot / Math.sqrt(cpu_norm * metal_norm)
      STDERR.puts "qwen_image21_block_parity max_abs=#{max_abs} cosine=#{cosine}"

      backend.metal_projection_count.should eq(6)
      metal.all?(&.finite?).should be_true
      max_abs.should be < 1.0e-2
      cosine.should be > 0.999999
    ensure
      weights.close
    end
  end

  it "executes the complete 32-block outer transformer with a real text prefix" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed outer-forward check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.transformer_config
      hidden = Array(Float32).new(4 * config.input_dim) do |index|
        (((index * 23 + 5) % 97) - 48).to_f32 / 127.0_f32
      end
      encoder_hidden = Array(Float32).new(config.context_dim) do |index|
        (((index * 31 + 9) % 103) - 51).to_f32 / 137.0_f32
      end
      expected = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden,
        encoder_hidden,
        0.5_f32,
        [StaticArray[1, 2, 2]],
        [false, true],
        weights.transformer_weights,
        config,
        backend: QwenImage21CPUReferenceBF16Backend.new,
      )
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
      started = Time.instant
      result = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden,
        encoder_hidden,
        0.5_f32,
        [StaticArray[1, 2, 2]],
        [false, true],
        weights.transformer_weights,
        config,
        backend: backend,
      )
      elapsed = Time.instant - started
      STDERR.puts "qwen_image21_outer_forward seconds=#{elapsed.total_seconds} metal_projections=#{backend.metal_projection_count}"

      result.output.size.should eq(5 * config.output_dim)
      result.output.all?(&.finite?).should be_true
      result.layout.target_token_mask.should eq([false, true, true, true, true])
      max_abs = 0.0_f64
      dot = 0.0_f64
      expected_norm = 0.0_f64
      result_norm = 0.0_f64
      result.output.zip(expected.output).each do |value, reference|
        max_abs = Math.max(max_abs, (value - reference).abs)
        dot += value.to_f64 * reference
        expected_norm += reference.to_f64 ** 2
        result_norm += value.to_f64 ** 2
      end
      cosine = dot / Math.sqrt(expected_norm * result_norm)
      STDERR.puts "qwen_image21_outer_bf16_parity max_abs=#{max_abs} cosine=#{cosine}"
      max_abs.should be < 1e-4
      cosine.should be > 0.999999
      backend.metal_projection_count.should eq(32 * 6 + 8)
      backend.bf16_projection_count.should eq(8)
      backend.fused_outer_command_count.should eq(1)
    ensure
      weights.close
    end
  end

  it "runs two configured FlowMatch steps through all 32 real blocks" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed denoising check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.transformer_config
      initial = Array(Float32).new(4 * config.input_dim) do |index|
        (((index * 29 + 7) % 101) - 50).to_f32 / 131.0_f32
      end
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
      started = Time.instant
      result = ML::GGUF::QwenImage21LatentDenoiser.run(
        initial,
        [] of Float32,
        [] of Float32,
        [StaticArray[1, 2, 2]],
        [] of Bool,
        weights.transformer_weights,
        config,
        num_inference_steps: 2,
        backend: backend,
      )
      elapsed = Time.instant - started
      STDERR.puts "qwen_image21_two_step_denoise seconds=#{elapsed.total_seconds} metal_projections=#{backend.metal_projection_count}"

      result.transformer_evaluations.should eq(2)
      result.latents.size.should eq(initial.size)
      result.latents.all?(&.finite?).should be_true
      result.latents.should_not eq(initial)
      backend.metal_projection_count.should eq(2 * (32 * 6 + 6))
      backend.bf16_projection_count.should eq(12)
      backend.fused_outer_command_count.should eq(2)
    ensure
      weights.close
    end
  end

  it "keeps real Qwen3-VL text-to-image denoising finite on the default Metal route" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    bundle_path = ENV["QWEN_IMAGE21_CONDITIONING"]?
    pending!("set QWEN_IMAGE21_GGUF and QWEN_IMAGE21_CONDITIONING for real prompt integration") unless path && File.file?(path) && bundle_path && File.file?(bundle_path)
    pending!("the unsafe fused text route was explicitly enabled") if ENV["QWEN_IMAGE21_FUSED_TEXT"]? == "1"
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    conditioning = ML::GGUF::QwenImage21ConditioningBundle.load(bundle_path.not_nil!)
    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
    begin
      result = ML::GGUF::QwenImage21LatentDenoiser.run(
        conditioning.initial_target_latents,
        [] of Float32,
        conditioning.encoder_hidden_states,
        conditioning.img_shapes,
        conditioning.encoder_img_mask,
        weights.transformer_weights,
        weights.transformer_config,
        num_inference_steps: 2,
        encoder_hidden_states_mask: conditioning.encoder_hidden_states_mask,
        backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true),
        layer_stack_backend: stack,
      )
      result.transformer_evaluations.should eq(2)
      result.latents.size.should eq(conditioning.initial_target_latents.size)
      result.latents.all?(&.finite?).should be_true
      result.latents.should_not eq(conditioning.initial_target_latents)
    ensure
      stack.close
      weights.close
    end
  end
end
