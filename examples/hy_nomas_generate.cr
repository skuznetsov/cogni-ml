require "option_parser"
require "./support/hy_nomas_generation"
require "./support/hy_nomas_decode_latent_admission"

input_path = ""
input_sha = ""
output_dir = ""
budget_mib = 22_528_i64
dit_weights = HyNomasGeneration::DiTWeights::BF16
steps = HyNomasGeneration::DEFAULT_STEPS
plan_only = false
capture_decoder = false
conditioned_inputs = false
resolution = "512x288"
cinematic_path = ""
cinematic_sha = ""
decoder_cache_path = ""
decoder_cache_sha = ""
OptionParser.parse do |p|
  p.banner = "Bounded native Hy-Nomas preview: 5 seconds at 24 fps"
  p.on("--inputs PATH", "Authenticated fixed preview inputs") { |v| input_path = v }
  p.on("--inputs-sha SHA", "Trusted input metadata identity") { |v| input_sha = v }
  p.on("--output-dir PATH", "Existing empty result directory") { |v| output_dir = v }
  p.on("--budget-mib N", "Aggregate named buffers per phase, including shared latent") { |v| budget_mib = v.to_i64 }
  p.on("--dit-weights MODE", "bf16 (default control) or q4-k (experimental); PixelUMM stays BF16") { |v| dit_weights = HyNomasGeneration.parse_dit_weights(v) }
  p.on("--steps N", "DiT Euler steps (BF16: 1..128; Q4: 1..8; default 8)") { |v| steps = v.to_i32 }
  p.on("--plan-only", "CPU-only metadata and memory admission") { plan_only = true }
  p.on("--capture-decoder", "Export completed DiT latent and raw decoder values for upstream comparison") { capture_decoder = true }
  p.on("--conditioned-inputs", "Admit a v2 default-size or opt-in v3 high-resolution bundle with an authenticated Qwen crop") { conditioned_inputs = true }
  p.on("--resolution SIZE", "512x288 (default) or opt-in 768x416; high resolution requires conditioned v3 inputs") { |v| resolution = v }
  p.on("--cinematic-cache PATH", "Opt-in authenticated Cinematic BF16 cache lineage; camera/action off") { |v| cinematic_path = v }
  p.on("--cinematic-cache-sha SHA", "Trusted Cinematic lineage SHA-256") { |v| cinematic_sha = v }
  p.on("--decoder-cache PATH", "Opt-in authenticated PixelUMM BF16 cache manifest") { |v| decoder_cache_path = v }
  p.on("--decoder-cache-sha SHA", "Trusted PixelUMM cache manifest SHA-256") { |v| decoder_cache_sha = v }
end
decoder_cache_identity = HyNomasLatentDecodeAdmission.resolve_decoder_cache!(decoder_cache_path, decoder_cache_sha)
cinematic = HyNomasCinematic.load!(cinematic_path, cinematic_sha,
  bf16: dit_weights == HyNomasGeneration::DiTWeights::BF16,
  camera_conditioning: false, action_conditioning: false)
geometry = HyNomasGeneration::SpatialGeometry.parse(resolution)
steps = HyNomasGeneration.validate_steps!(steps, dit_weights)
raise ArgumentError.new("--inputs and --inputs-sha are required") if input_path.empty? || input_sha.empty?
unless (16_384..24_576).includes?(budget_mib)
  raise ArgumentError.new("named budget must be within 16..24 GiB; use the external 10% memory guard")
end
budget = budget_mib * 1024 * 1024
request_start = Time.monotonic
inputs = HyNomasGeneration::Inputs.new(input_path, input_sha,
  conditioned: conditioned_inputs, geometry: geometry)
runtime_config = geometry.dit_config
admission = dit_weights == HyNomasGeneration::DiTWeights::BF16 ? HyNomasGeneration.authenticate_bf16_index(runtime_config, cinematic: cinematic) : nil
q4_payload_bytes = admission ? admission.payload_bytes : HyNomasGeneration.authenticate_weight_index(runtime_config).caches.sum(&.payload_bytes)
metadata = HyNomasGeneration::DecoderMetadata.new(inputs, budget,
  capture_raw: capture_decoder, cache_identity: decoder_cache_identity)
dit_named = HyNomasGeneration.dit_resident_weight_bytes(dit_weights,
  HyNomasGeneration.stage_bytes, q4_payload_bytes) + HyNomasGeneration.dit_reserve_bytes(inputs, steps, runtime_config)
raise ArgumentError.new("DiT named aggregate exceeds budget") if dit_named > budget
preflight_seconds = (Time.monotonic - request_start).total_seconds
HyNomasGeneration.emit("memory_plan", dit_weights: HyNomasGeneration.dit_weights_name(dit_weights), steps: steps, dit_named_bytes: dit_named,
  dit_variant: cinematic ? "cinematic-active-blocks" : "original", dit_weight_index_sha256: cinematic.try(&.sha256) || HyNomasGeneration::INDEX_SHA,
  decoder_weight_cache_sha256: metadata.cache.sha256,
  decoder_named_bytes: metadata.named_bytes, named_budget_bytes: budget,
  resolution: "#{geometry.pixel_width}x#{geometry.pixel_height}", latent_shape: metadata.config.latent_shape,
  dit_video_tokens: runtime_config.video_tokens,
  pixelumm_tokens: metadata.config.decoder_config.tokens, pixelumm_windows: metadata.config.window_count,
  preflight_seconds: preflight_seconds,
  phase_handoff: "one drained chain, latent retained on GPU, weights released before next phase")
if plan_only
  metadata.close
  exit
end
unless Path[output_dir].absolute? && File.directory?(output_dir) && Dir.empty?(output_dir)
  raise ArgumentError.new("output directory must be an existing empty absolute directory")
end
output_dir = File.realpath(output_dir)
unless output_dir.starts_with?(File.realpath("/opt/mldata_ssd") + "/")
  raise ArgumentError.new("preview outputs must be on /opt/mldata_ssd")
end
inputs.file.verify_payload!
shared = ML::Video::Arena.new
latent = inputs.upload!("input.latent", shared)
dit = nil.as(HyNomasGeneration::DiT?)
decoder = nil.as(HyNomasGeneration::Decoder?)
chain = nil.as(ML::Metal::MPSCommandChain?)
timings = {} of String => Float64
diagnostics = {} of String => String
timings["preflight_seconds"] = preflight_seconds
metrics = {} of String => Int64
start = Time.monotonic
begin
  setup = Time.monotonic
  dit = HyNomasGeneration::DiT.new(inputs, latent, budget, dit_weights, steps: steps, admission: admission, cinematic: cinematic)
  raise "DiT preflight and allocation admission disagree" unless dit.named_bytes == dit_named
  timings["dit_setup_seconds"] = (Time.monotonic - setup).total_seconds
  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  run = Time.monotonic
  dit.encode(chain)
  metrics["dit_encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
  metrics["dit_encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
  metrics["dit_preterminal_waits"] = chain.wait_count.to_i64
  raise "DiT encode introduced allocation/readback/wait" unless metrics.values.all?(&.zero?)
  chain.commit_and_wait
  timings["dit_wall_seconds"] = (Time.monotonic - run).total_seconds
  timings["dit_gpu_seconds"] = chain.gpu_elapsed_seconds
  metrics["dit_segments"] = chain.segment_count.to_i64
  metrics["dit_waits"] = chain.wait_count.to_i64
  HyNomasGeneration.emit("dit_completed", wall_seconds: timings["dit_wall_seconds"], gpu_seconds: chain.gpu_elapsed_seconds)
  if capture_decoder
    path = File.join(output_dir, "latent.f32le")
    HyNomasGeneration.write_fresh_output(path) { |f| f.write(Bytes.new(latent.contents.as(UInt8*), latent.size.to_i32)) }
    diagnostics["latent"] = path
    diagnostics["latent_sha256"] = PixelUMMBlockProbe.sha_file(path)
  end
  dit.release
  dit = nil
  chain.release
  chain = nil
  # This is the explicit memory phase boundary. The final latent never leaves
  # its original GPU allocation. No block, window or diffusion-step wait exists.
  setup = Time.monotonic
  decoder = HyNomasGeneration::Decoder.new(inputs, latent, metadata)
  timings["decoder_setup_seconds"] = (Time.monotonic - setup).total_seconds
  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  run = Time.monotonic
  decoder.encode(chain)
  metrics["decoder_encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
  metrics["decoder_encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
  metrics["decoder_preterminal_waits"] = chain.wait_count.to_i64
  unless metrics["decoder_encode_allocations"] == 0 && metrics["decoder_encode_readbacks"] == 0 && chain.wait_count == 0
    raise "decoder encode introduced allocation/readback/wait"
  end
  chain.commit_and_wait
  timings["decoder_wall_seconds"] = (Time.monotonic - run).total_seconds
  timings["decoder_gpu_seconds"] = chain.gpu_elapsed_seconds
  metrics["decoder_segments"] = chain.segment_count.to_i64
  metrics["decoder_waits"] = chain.wait_count.to_i64
  HyNomasGeneration.emit("decoder_completed", wall_seconds: timings["decoder_wall_seconds"], gpu_seconds: chain.gpu_elapsed_seconds)
  if capture_decoder
    raw = decoder.raw_output
    path = File.join(output_dir, "decoder-raw.f32le")
    HyNomasGeneration.write_fresh_output(path) { |f| f.write(Bytes.new(raw.contents.as(UInt8*), raw.size.to_i32)) }
    diagnostics["raw_pixels"] = path
    diagnostics["raw_pixels_sha256"] = PixelUMMBlockProbe.sha_file(path)
    diagnostics["scope"] = "opt-in completed-buffer exports after the existing phase waits; raw capture is a GPU copy before display conversion"
  end
  pixels = decoder.output
  path = File.join(output_dir, "pixels.f32le")
  HyNomasGeneration.write_fresh_output(path) { |f| f.write(Bytes.new(pixels.contents.as(UInt8*), pixels.size.to_i32)) }
  timings["total_seconds"] = (Time.monotonic - start).total_seconds
  # Include preflight authentication and input upload as well as setup/compute.
  # The historical total_seconds field starts after those earlier stages.
  timings["request_seconds"] = (Time.monotonic - request_start).total_seconds
  receipt = {schema: "cogni_ml.hy_nomas_native_preview/v1", upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
             prompt: inputs.prompt, inputs: inputs.file.path, inputs_sha256: inputs.file.sha256,
             qwen_valid_tokens: inputs.valid_text_tokens, compact_text_tokens: inputs.compact_text_tokens,
             qwen_input: "authenticated saved upstream state26 crop; native live text encoding excluded",
             dit_weights: HyNomasGeneration.dit_weights_name(dit_weights),
             dit_quantization_experimental: dit_weights == HyNomasGeneration::DiTWeights::Q4K,
             precision: dit_weights == HyNomasGeneration::DiTWeights::BF16 ? "BF16 weights and boundary rounding, F32 native arithmetic/MPS accumulation" : "Q4_K DiT block weights decoded on GPU into one reusable BF16 stage; BF16 DiT globals, PixelUMM and Euler boundary rounding; F32 native arithmetic/MPS accumulation",
             dit_weight_index_sha256: cinematic.try(&.sha256) || HyNomasGeneration::INDEX_SHA, decoder_weight_cache_sha256: metadata.cache.sha256,
             dit_variant: cinematic ? "cinematic-active-blocks" : "original",
             cinematic_lineage: cinematic.try(&.lineage_path), cinematic_lineage_sha256: cinematic.try(&.lineage_sha256),
             seed: 42, frames: metadata.config.pixel_frames, width: geometry.pixel_width,
             height: geometry.pixel_height, fps: 24, steps: steps, shift: 5, cfg: 1,
             camera_conditioning: false, action_conditioning: false,
             pixelumm: {steps: 1, window_latents: 10, overlap_latents: 5, starts: [0, 5, 10, 15, 20]},
             handoffs: "GPU latent retained across one explicit drained weight-phase boundary",
             output: path, output_bytes: pixels.size, output_sha256: PixelUMMBlockProbe.sha_file(path),
             dit_named_bytes: dit_named, decoder_named_bytes: metadata.named_bytes,
             metric_scope: "allocation/readback counters cover instrumented MetalBuffer methods; opaque MPS resources and direct terminal pixel-file export are excluded",
             timings: timings, metrics: metrics, diagnostics: diagnostics, full_upstream_video_parity: false}
  HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) { |f| f.puts(receipt.to_pretty_json) }
  HyNomasGeneration.emit("preview_pixels_saved", output: path, total_seconds: timings["total_seconds"])
ensure
  chain.try(&.commit_and_wait)
  decoder.try(&.release)
  dit.try(&.release)
  chain.try(&.release)
  shared.release
  metadata.close
end
