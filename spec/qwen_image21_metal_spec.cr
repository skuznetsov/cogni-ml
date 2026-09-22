require "./spec_helper"
require "../src/ml/gguf/qwen_image21_weights"
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_flow_match"

describe ML::GGUF::QwenImage21MetalProjectionBackend do
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

  it "executes the complete 32-block outer transformer on a minimum valid target" do
    path = ENV["QWEN_IMAGE21_GGUF"]?
    pending!("set QWEN_IMAGE21_GGUF to run the model-backed outer-forward check") unless path && File.file?(path)
    pending!("Metal is unavailable") unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

    weights = ML::GGUF::QwenImage21Weights.from_gguf(path.not_nil!)
    begin
      config = weights.transformer_config
      hidden = Array(Float32).new(4 * config.input_dim) do |index|
        (((index * 23 + 5) % 97) - 48).to_f32 / 127.0_f32
      end
      backend = ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: false)
      started = Time.instant
      result = ML::GGUF::QwenImage21TransformerCPU.forward(
        hidden,
        [] of Float32,
        0.5_f32,
        [StaticArray[1, 2, 2]],
        [true],
        weights.transformer_weights,
        config,
        backend: backend,
      )
      elapsed = Time.instant - started
      STDERR.puts "qwen_image21_outer_forward seconds=#{elapsed.total_seconds} metal_projections=#{backend.metal_projection_count}"

      result.output.size.should eq(4 * config.output_dim)
      result.output.all?(&.finite?).should be_true
      result.layout.target_token_mask.all?.should be_true
      backend.metal_projection_count.should be >= 32 * 6
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
      backend.metal_projection_count.should be >= 2 * 32 * 6
    ensure
      weights.close
    end
  end
end
