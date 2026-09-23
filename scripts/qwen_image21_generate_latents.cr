#!/usr/bin/env crystal

# Run the native GGUF Metal DiT on a real Qwen3-VL prompt-conditioning bundle.
# The companion reference-side scripts prepare the bundle and decode this
# output. Diffusers never executes the transformer in this path.
require "../src/ml/gguf/qwen_image21_conditioning_bundle"
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_weights"
require "../src/ml/gguf/qwen_image21_flow_match"

private def write_latent_bundle(
  output_dir : String,
  conditioning : ML::GGUF::QwenImage21ConditioningBundle,
  latents : Array(Float32),
  steps : Int32,
  gguf_path : String,
) : Nil
  expected = conditioning.latent_height * conditioning.latent_width * 64
  raise ArgumentError.new("denoiser output size mismatch") unless latents.size == expected
  raise ArgumentError.new("denoiser produced non-finite latents") unless latents.all?(&.finite?)

  Dir.mkdir_p(output_dir)
  payload_path = File.join(output_dir, "qwen_image21_latents.bin")
  manifest_path = File.join(output_dir, "qwen_image21_latents.json")
  if File.exists?(payload_path) || File.exists?(manifest_path)
    raise ArgumentError.new("output bundle already exists in #{output_dir}")
  end

  File.open(payload_path, "w") do |io|
    latents.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
  end
  manifest = JSON.build do |json|
    json.object do
      json.field "format", "qwen-image21-latents-v1"
      json.field "model_id", "Qwen/Qwen-Image-2.1"
      json.field "layout", "tokens_hwc"
      json.field "channels", 64
      json.field "latent_height", conditioning.latent_height
      json.field "latent_width", conditioning.latent_width
      json.field "image_height", conditioning.image_height
      json.field "image_width", conditioning.image_width
      json.field "dtype", "float32-le"
      json.field "scaling", "diffusers_normalized"
      json.field "payload", "qwen_image21_latents.bin"
      json.field "payload_bytes", latents.size * sizeof(Float32)
      json.field "model_revision", conditioning.model_revision
      json.field "seed", conditioning.seed
      json.field "prompt", conditioning.prompt
      json.field "denoising_steps", steps
      json.field "dit_gguf", File.expand_path(gguf_path)
    end
  end
  File.write(manifest_path, manifest)
  puts "latent_bundle=#{manifest_path} tokens=#{conditioning.latent_height * conditioning.latent_width} steps=#{steps}"
end

gguf_path = ARGV[0]? || abort "usage: crystal run scripts/qwen_image21_generate_latents.cr -- MODEL.gguf CONDITIONING.json OUTPUT_DIR [STEPS=40]"
conditioning_path = ARGV[1]? || abort "missing CONDITIONING.json"
output_dir = ARGV[2]? || abort "missing OUTPUT_DIR"
steps = ARGV[3]?.try(&.to_i) || 40
abort "steps must be in 2..100" unless steps >= 2 && steps <= 100
abort "GGUF file not found: #{gguf_path}" unless File.file?(gguf_path)
abort "Metal backend unavailable" unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

conditioning = ML::GGUF::QwenImage21ConditioningBundle.load(conditioning_path)
timing = ENV["QWEN_IMAGE21_TIMING"]? == "1"
load_started = Time.instant
model = ML::GGUF::QwenImage21Weights.from_gguf(gguf_path)
stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
loaded_at = Time.instant
begin
  config = model.transformer_config
  puts "denoising prompt=#{conditioning.prompt.inspect} image=#{conditioning.image_width}x#{conditioning.image_height} seed=#{conditioning.seed} steps=#{steps}"
  result = ML::GGUF::QwenImage21LatentDenoiser.run(
    conditioning.initial_target_latents,
    [] of Float32,
    conditioning.encoder_hidden_states,
    conditioning.img_shapes,
    conditioning.encoder_img_mask,
    model.transformer_weights,
    config,
    steps,
    encoder_hidden_states_mask: conditioning.encoder_hidden_states_mask,
    backend: ML::GGUF::QwenImage21MetalProjectionBackend.new(strict: true),
    layer_stack_backend: stack,
  )
  denoised_at = Time.instant
  raise "wrong number of transformer evaluations" unless result.transformer_evaluations == steps
  write_latent_bundle(output_dir, conditioning, result.latents, steps, gguf_path)
  if timing
    written_at = Time.instant
    puts "timing model_load_ms=#{(loaded_at - load_started).total_milliseconds.round(3)} " \
         "denoise_ms=#{(denoised_at - loaded_at).total_milliseconds.round(3)} " \
         "bundle_write_ms=#{(written_at - denoised_at).total_milliseconds.round(3)}"
  end
ensure
  stack.close
  model.close
end
