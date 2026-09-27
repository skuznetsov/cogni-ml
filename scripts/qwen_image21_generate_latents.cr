#!/usr/bin/env crystal

# Run the native GGUF Metal DiT on a real Qwen3-VL prompt-conditioning bundle.
# The companion reference-side scripts prepare the bundle and decode this
# output. Diffusers never executes the transformer in this path.
require "digest/sha256"
require "../src/ml/gguf/qwen_image21_conditioning_bundle"
require "../src/ml/gguf/qwen_image21_metal"
require "../src/ml/gguf/qwen_image21_weights"
require "../src/ml/gguf/qwen_image21_flow_match"

private def write_latent_bundle(
  output_dir : String,
  conditioning : ML::GGUF::QwenImage21ConditioningBundle,
  latents : Array(Float32),
  steps : Int32,
  solver : ML::GGUF::QwenImage21FlowMatchSolver,
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
      json.field "conditioning_payload_sha256", conditioning.conditioning_payload_sha256
      json.field "seed", conditioning.seed
      json.field "prompt", conditioning.prompt
      json.field "denoising_steps", steps
      json.field "flow_match_solver", solver.label
      json.field "dit_gguf", File.expand_path(gguf_path)
    end
  end
  File.write(manifest_path, manifest)
  puts "latent_bundle=#{manifest_path} tokens=#{conditioning.latent_height * conditioning.latent_width} steps=#{steps}"
end

private def resolve_new_step_latent_directory(path : String?, output_dir : String) : String?
  return nil unless path
  abort "QWEN_IMAGE21_STEP_LATENT_DIR must not be empty" if path.empty?

  expanded = File.expand_path(path)
  output_path = File.expand_path(output_dir)
  abort "step latent directory must differ from output directory" if expanded == output_path
  if File.exists?(expanded) || Dir.exists?(expanded) || File.symlink?(expanded)
    abort "step latent directory already exists: #{expanded}"
  end

  parent = File.dirname(expanded)
  abort "step latent directory parent does not exist: #{parent}" unless Dir.exists?(parent)
  expanded
end

private def create_step_latent_directory(path : String?) : Nil
  return unless path

  begin
    Dir.mkdir(path)
  rescue error : File::Error
    abort "could not create new step latent directory #{path}: #{error.message}"
  end
end

private def write_step_latent_snapshot(
  directory : String,
  index : Int32,
  timestep : Float32,
  latents : Array(Float32),
) : Nil
  filename = "step-#{index.to_s.rjust(3, '0')}.bin"
  path = File.join(directory, filename)
  if File.exists?(path) || Dir.exists?(path) || File.symlink?(path)
    raise ArgumentError.new("step latent snapshot already exists: #{path}")
  end

  bytes = IO::Memory.new(latents.size * sizeof(Float32))
  latents.each { |value| bytes.write_bytes(value, IO::ByteFormat::LittleEndian) }
  payload = bytes.to_slice
  File.open(path, "w") { |file| file.write(payload) }
  sha256 = Digest::SHA256.hexdigest(payload)
  puts "denoising_step_latent index=#{index} timestep=#{timestep} " \
       "sha256=#{sha256} bytes=#{payload.size} file=#{path}"
end

gguf_path = ARGV[0]? || abort "usage: crystal run scripts/qwen_image21_generate_latents.cr -- MODEL.gguf CONDITIONING.json OUTPUT_DIR [STEPS=40]"
conditioning_path = ARGV[1]? || abort "missing CONDITIONING.json"
output_dir = ARGV[2]? || abort "missing OUTPUT_DIR"
steps = ARGV[3]?.try(&.to_i) || 40
abort "steps must be in 2..100" unless steps >= 2 && steps <= 100
step_latent_dir = resolve_new_step_latent_directory(
  ENV["QWEN_IMAGE21_STEP_LATENT_DIR"]?, output_dir,
)
solver = begin
  ML::GGUF::QwenImage21FlowMatchSolver.parse(ENV["QWEN_IMAGE21_SOLVER"]? || "euler")
rescue error : ArgumentError
  abort "invalid QWEN_IMAGE21_SOLVER: #{error.message}"
end
abort "GGUF file not found: #{gguf_path}" unless File.file?(gguf_path)
abort "Metal backend unavailable" unless ML::GGUF::QwenImage21MetalProjectionBackend.available?

conditioning = ML::GGUF::QwenImage21ConditioningBundle.load(conditioning_path)
create_step_latent_directory(step_latent_dir)
if directory = step_latent_dir
  puts "step_latent_snapshot_format directory=#{directory} layout=tokens_hwc " \
       "channels=64 latent_height=#{conditioning.latent_height} " \
       "latent_width=#{conditioning.latent_width} dtype=float32-le " \
       "scaling=diffusers_normalized"
end
timing = ENV["QWEN_IMAGE21_TIMING"]? == "1"
step_timing = ENV["QWEN_IMAGE21_STEP_TIMING"]? == "1"
step_hashes = ENV["QWEN_IMAGE21_STEP_HASHES"]? == "1"
if step_timing && ENV["QWEN_IMAGE21_PROFILE"]?.nil?
  # Profile the existing single-command route so the per-step report can
  # include device elapsed time without enabling diagnostic phase splits.
  ENV["QWEN_IMAGE21_PROFILE"] = "1"
end
profile_mode = ENV["QWEN_IMAGE21_PROFILE"]?
load_started = Time.instant
model = ML::GGUF::QwenImage21Weights.from_gguf(gguf_path)
stack = ML::GGUF::QwenImage21MetalLayerStackBackend.new
loaded_at = Time.instant
begin
  config = model.transformer_config
  puts "denoising prompt=#{conditioning.prompt.inspect} image=#{conditioning.image_width}x#{conditioning.image_height} seed=#{conditioning.seed} steps=#{steps} solver=#{solver.label}"
  previous_cache_builds = 0
  previous_cache_hits = 0
  step_latent_hash_observer = if step_hashes
                                ->(index : Int32, timestep : Float32, sha256 : String) {
                                  puts "denoising_step_hash index=#{index} timestep=#{timestep} sha256=#{sha256}"
                                  nil
                                }
                              else
                                nil
                              end
  step_latent_snapshot_observer = if directory = step_latent_dir
                                    ->(index : Int32, timestep : Float32, latents : Array(Float32)) {
                                      write_step_latent_snapshot(directory, index, timestep, latents)
                                    }
                                  else
                                    nil
                                  end
  step_observer = if step_timing
                    ->(index : Int32, sigma : Float32, timestep : Float32, elapsed : Time::Span) {
                      cache_builds = stack.prefix_cache_builds
                      cache_hits = stack.prefix_cache_hits
                      cache_mode = if cache_builds > previous_cache_builds
                                     "build"
                                   elsif cache_hits > previous_cache_hits
                                     "hit"
                                   else
                                     "none"
                                   end
                      previous_cache_builds = cache_builds
                      previous_cache_hits = cache_hits
                      stats = stack.last_stats
                      layer_fields = if stats
                                       " layer_stack_invocations=#{stack.invocations}" \
                                       " active_tokens=#{stats.active_tokens}" \
                                       " image_projection_rows=#{stats.image_projection_rows}" \
                                       " command_buffers=#{stats.command_buffers}" \
                                       " projection_dispatches=#{stats.projection_dispatches}" \
                                       " intermediate_readbacks=#{stats.intermediate_readbacks}" \
                                       " final_readbacks=#{stats.final_readbacks}" \
                                       " buffer_prepare_ms=#{stats.buffer_prepare_ms.try(&.round(3)) || "unavailable"}" \
                                       " readback_ms=#{stats.readback_ms.try(&.round(3)) || "unavailable"}"
                                     else
                                       " layer_stats=unavailable"
                                     end
                      gpu_ms = stats.try(&.gpu_command_ms)
                      gpu_field = if profile_mode == "phases"
                                    " gpu_phase_split_sum_ms=#{gpu_ms.try(&.round(3)) || "unavailable"}"
                                  else
                                    " gpu_command_ms=#{gpu_ms.try(&.round(3)) || "unavailable"}"
                                  end
                      puts "denoising_step index=#{index} sigma=#{sigma} timestep=#{timestep}" \
                           " elapsed_ms=#{elapsed.total_milliseconds.round(3)}" \
                           " cache=#{cache_mode} cache_builds=#{cache_builds} cache_hits=#{cache_hits}" \
                           "#{layer_fields}#{gpu_field}"
                      nil
                    }
                  else
                    nil
                  end
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
    step_observer: step_observer,
    step_latent_hash_observer: step_latent_hash_observer,
    step_latent_snapshot_observer: step_latent_snapshot_observer,
    solver: solver,
  )
  denoised_at = Time.instant
  raise "wrong number of transformer evaluations" unless result.transformer_evaluations == steps
  write_latent_bundle(output_dir, conditioning, result.latents, steps, solver, gguf_path)
  if timing || step_timing
    written_at = Time.instant
    puts "timing model_load_ms=#{(loaded_at - load_started).total_milliseconds.round(3)} " \
         "denoise_ms=#{(denoised_at - loaded_at).total_milliseconds.round(3)} " \
         "bundle_write_ms=#{(written_at - denoised_at).total_milliseconds.round(3)}"
  end
  ML::GGUF::QwenImage21MetalAttentionRouteDiagnostics.write_summary
ensure
  stack.close
  model.close
end
