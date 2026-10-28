require "option_parser"
require "./support/hy_nomas_generation"

input_path = ""
input_sha = ""
output_dir = ""
budget_mib = 22_528_i64
dit_weights = HyNomasGeneration::DiTWeights::BF16
plan_only = false
OptionParser.parse do |p|
  p.banner = "Bounded native Hy-Nomas preview: 120 frames, 512x288, 24 fps"
  p.on("--inputs PATH", "Authenticated fixed preview inputs") { |v| input_path = v }
  p.on("--inputs-sha SHA", "Trusted input metadata identity") { |v| input_sha = v }
  p.on("--output-dir PATH", "Existing empty result directory") { |v| output_dir = v }
  p.on("--budget-mib N", "Aggregate named buffers per phase, including shared latent") { |v| budget_mib = v.to_i64 }
  p.on("--dit-weights MODE", "bf16 (default control) or q4-k (experimental); PixelUMM stays BF16") { |v| dit_weights = HyNomasGeneration.parse_dit_weights(v) }
  p.on("--plan-only", "CPU-only metadata and memory admission") { plan_only = true }
end
raise ArgumentError.new("--inputs and --inputs-sha are required") if input_path.empty? || input_sha.empty?
unless (16_384..24_576).includes?(budget_mib)
  raise ArgumentError.new("named budget must be within 16..24 GiB; use the external 10% memory guard")
end
budget = budget_mib * 1024 * 1024
request_start = Time.monotonic
inputs = HyNomasGeneration::Inputs.new(input_path, input_sha)
admission = dit_weights == HyNomasGeneration::DiTWeights::BF16 ? HyNomasGeneration.authenticate_bf16_index : nil
q4_payload_bytes = admission ? admission.payload_bytes : HyNomasGeneration.authenticate_weight_index.caches.sum(&.payload_bytes)
metadata = HyNomasGeneration::DecoderMetadata.new(inputs, budget)
dit_named = HyNomasGeneration.dit_resident_weight_bytes(dit_weights,
  HyNomasGeneration.stage_bytes, q4_payload_bytes) + HyNomasGeneration.dit_reserve_bytes(inputs)
raise ArgumentError.new("DiT named aggregate exceeds budget") if dit_named > budget
preflight_seconds = (Time.monotonic - request_start).total_seconds
HyNomasGeneration.emit("memory_plan", dit_weights: HyNomasGeneration.dit_weights_name(dit_weights), dit_named_bytes: dit_named,
  decoder_named_bytes: metadata.named_bytes, named_budget_bytes: budget,
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
timings["preflight_seconds"] = preflight_seconds
metrics = {} of String => Int64
start = Time.monotonic
begin
  setup = Time.monotonic
  dit = HyNomasGeneration::DiT.new(inputs, latent, budget, dit_weights, admission: admission)
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
  pixels = decoder.output
  path = File.join(output_dir, "pixels.f32le")
  HyNomasGeneration.write_fresh_output(path) { |f| f.write(Bytes.new(pixels.contents.as(UInt8*), pixels.size.to_i32)) }
  timings["total_seconds"] = (Time.monotonic - start).total_seconds
  # Include preflight authentication and input upload as well as setup/compute.
  # The historical total_seconds field starts after those earlier stages.
  timings["request_seconds"] = (Time.monotonic - request_start).total_seconds
  receipt = {schema: "cogni_ml.hy_nomas_native_preview/v1", upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
             prompt: inputs.prompt, inputs: inputs.file.path, inputs_sha256: inputs.file.sha256,
             qwen_input: "authenticated saved upstream state26 crop; native live text encoding excluded",
             dit_weights: HyNomasGeneration.dit_weights_name(dit_weights),
             dit_quantization_experimental: dit_weights == HyNomasGeneration::DiTWeights::Q4K,
             precision: dit_weights == HyNomasGeneration::DiTWeights::BF16 ? "BF16 weights and boundary rounding, F32 native arithmetic/MPS accumulation" : "Q4_K DiT block weights decoded on GPU into one reusable BF16 stage; BF16 DiT globals, PixelUMM and Euler boundary rounding; F32 native arithmetic/MPS accumulation",
             dit_weight_index_sha256: HyNomasGeneration::INDEX_SHA, decoder_weight_cache_sha256: HyNomasGeneration::PIXEL_CACHE_SHA,
             seed: 42, frames: 120, width: 512, height: 288, fps: 24, steps: 8, shift: 5, cfg: 1,
             camera_conditioning: false, action_conditioning: false,
             pixelumm: {steps: 1, window_latents: 10, overlap_latents: 5, starts: [0, 5, 10, 15, 20]},
             handoffs: "GPU latent retained across one explicit drained weight-phase boundary",
             output: path, output_bytes: pixels.size, output_sha256: PixelUMMBlockProbe.sha_file(path),
             dit_named_bytes: dit_named, decoder_named_bytes: metadata.named_bytes,
             metric_scope: "allocation/readback counters cover instrumented MetalBuffer methods; opaque MPS resources and direct terminal pixel-file export are excluded",
             timings: timings, metrics: metrics, full_upstream_video_parity: false}
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
