require "option_parser"
require "./support/hy_nomas_sequence"

input_path = ""
input_sha = ""
output_dir = ""
budget_mib = 33_792_i64
segments = 4
plan_only = false
dit_only = false
text_cache_policy = HyNomasSequence::HISTORY_TEXT_POLICY
cinematic_path = ""
cinematic_sha = ""
OptionParser.parse do |p|
  p.banner = "Experimental GPU Hy-Nomas continuation: two through six linked 5-second segments, 512x288 at 24 fps"
  p.on("--inputs PATH", "Authenticated continuation input bundle") { |v| input_path = v }
  p.on("--inputs-sha SHA", "Trusted continuation manifest identity") { |v| input_sha = v }
  p.on("--output-dir PATH", "Existing empty result directory on the external scratch SSD") { |v| output_dir = v }
  p.on("--budget-mib N", "Named buffers at the larger weight phase; external 10% guard is mandatory") { |v| budget_mib = v.to_i64 }
  p.on("--segments N", "Two through six linked segments; default four (20 seconds)") { |v| segments = v.to_i32 }
  p.on("--dit-only", "Qualification: export the latent bank after the DiT terminal wait") { dit_only = true }
  p.on("--legacy-text-per-chunk", "Control only: repeat scene text in every 5-second chunk") do
    text_cache_policy = ML::Video::HyNomasContinuationHistory::TextCachePolicy::PerChunk
  end
  p.on("--cinematic-cache PATH", "Opt-in authenticated Cinematic BF16 cache lineage") { |v| cinematic_path = v }
  p.on("--cinematic-cache-sha SHA", "Trusted Cinematic lineage SHA-256") { |v| cinematic_sha = v }
  p.on("--plan-only", "CPU-only authentication and aggregate admission") { plan_only = true }
end
HyNomasSequence.validate_segments!(segments)
raise ArgumentError.new("--inputs and --inputs-sha are required") if input_path.empty? || input_sha.empty?
unless (24_576..49_152).includes?(budget_mib)
  raise ArgumentError.new("sequence named budget must be within 24..48 GiB")
end
budget = budget_mib * 1024 * 1024
request_start = Time.monotonic
inputs = HyNomasGeneration::TimelineInputs.new(File.realpath(input_path), input_sha)
inputs.verify_payload!
inputs.base.file.verify_payload!
HyNomasSequence.validate_shared_scene_prompt!(inputs.base.prompt)
cinematic = HyNomasCinematic.load!(cinematic_path, cinematic_sha,
  bf16: true, camera_conditioning: false, action_conditioning: false)
weight_identity = HyNomasSequence.weight_identity(cinematic)
admission = HyNomasGeneration.authenticate_bf16_index(cinematic: cinematic)
dit_peak = HyNomasSequence.peak_named_bytes(inputs, segments, text_cache_policy)
dit_steady = dit_peak - HyNomasSequence.old_attention_bytes(inputs.base)
raise ArgumentError.new("sequence peak named buffers exceed budget") unless dit_peak <= budget
latent_frames = segments * 30
pixel_frames = latent_frames * 4
metadata = dit_only ? nil : HyNomasGeneration::DecoderMetadata.new(inputs.base, budget, total_latent_frames: latent_frames)
# A shorter decoder consumes gathered prefixes. Their destinations are already
# counted by DecoderMetadata; retain both full source banks until its chain drains.
decoder_peak = metadata.try do |decoder_metadata|
  retained_sources = segments < 6 ? inputs.bytes(%w(input.latent_bank decoder.noise)) : 0_i64
  decoder_metadata.named_bytes + retained_sources
end
raise ArgumentError.new("decoder source banks and prefix buffers exceed budget") if decoder_peak && decoder_peak > budget
preflight_seconds = (Time.monotonic - request_start).total_seconds
HyNomasGeneration.emit("sequence_memory_plan", segments: segments, steps_per_segment: 4,
  dit_peak_named_bytes: dit_peak, dit_steady_named_bytes: dit_steady,
  history_named_bytes: HyNomasSequence.history_bytes(inputs.base, segments, text_cache_policy),
  decoder_named_bytes: metadata.try(&.named_bytes), decoder_peak_named_bytes: decoder_peak,
  frames: pixel_frames, named_budget_bytes: budget,
  cache_precision: "F32 native post-RoPE K/V; no additional BF16 rounding",
  dit_variant: weight_identity[:dit_variant],
  dit_weight_index_sha256: weight_identity[:dit_weight_index_sha256],
  cinematic_lineage: weight_identity[:cinematic_lineage],
  cinematic_lineage_sha256: weight_identity[:cinematic_lineage_sha256],
  text_cache_policy: text_cache_policy.once_per_scene? ? "once_per_scene" : "per_chunk",
  cache_writer: "separate noisy forward after four denoise forwards; sigma 0.206897, t 206.897",
  full_upstream_video_parity: false)
if plan_only
  metadata.try(&.close)
  exit
end
unless Path[output_dir].absolute? && File.directory?(output_dir) && Dir.empty?(output_dir)
  raise ArgumentError.new("output directory must be an existing empty absolute directory")
end
output_dir = File.realpath(output_dir)
unless output_dir.starts_with?(File.realpath("/opt/mldata_ssd") + "/")
  raise ArgumentError.new("sequence outputs must be on /opt/mldata_ssd")
end
timeline_arena = ML::Video::Arena.new
current_arena = ML::Video::Arena.new
decoder_inputs = ML::Video::Arena.new
dit = nil.as(HyNomasSequence::DiT?)
decoder = nil.as(HyNomasGeneration::Decoder?)
chain = nil.as(ML::Metal::MPSCommandChain?)
prefixes = [] of ML::Video::HyNomasTimelineOps::PrefixPlan
timings = {"preflight_seconds" => preflight_seconds}
metrics = {} of String => Int64
diagnostics = {} of String => String
request_error : Exception? = nil
cleanup_errors = [] of String
begin
  timeline = inputs.upload!("input.latent_bank", timeline_arena)
  latent = current_arena.alloc("input.latent", 32 * 30 * 9 * 16)
  setup = Time.monotonic
  dit = HyNomasSequence::DiT.new(inputs, timeline, latent, budget, segments, admission,
    text_cache_policy, cinematic: cinematic)
  raise "sequence preflight/allocation admission disagree" unless dit.named_bytes == dit_steady
  timings["dit_setup_seconds"] = (Time.monotonic - setup).total_seconds
  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  run = Time.monotonic
  dit.encode(chain)
  metrics["dit_encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
  metrics["dit_encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
  metrics["dit_preterminal_waits"] = chain.wait_count.to_i64
  raise "sequence encode introduced allocation/readback/wait" unless metrics.values.all?(&.zero?)
  chain.commit_and_wait
  timings["dit_wall_seconds"] = (Time.monotonic - run).total_seconds
  timings["dit_gpu_seconds"] = chain.gpu_elapsed_seconds
  metrics["dit_segments"] = chain.segment_count.to_i64
  metrics["dit_waits"] = chain.wait_count.to_i64
  path = File.join(output_dir, "latent-bank.f32le")
  HyNomasGeneration.write_fresh_output(path) { |f| f.write(Bytes.new(timeline.contents.as(UInt8*), timeline.size.to_i32)) }
  diagnostics["latent_bank"] = path
  diagnostics["latent_bank_sha256"] = PixelUMMBlockProbe.sha_file(path)
  HyNomasGeneration.emit("sequence_dit_completed", wall_seconds: timings["dit_wall_seconds"],
    gpu_seconds: chain.gpu_elapsed_seconds, clean_latent_frames: segments * 30)
  dit.release
  dit = nil
  chain.release
  chain = nil
  current_arena.release
  if decoder_metadata = metadata
    # Exactly the existing drained weight-phase handoff. The clean timeline
    # retains its GPU allocation; noise is uploaded only after DiT is freed.
    setup = Time.monotonic
    noise = inputs.upload!("decoder.noise", decoder_inputs)
    decoder_latent = timeline
    decoder_noise = noise
    if segments < 6
      latent_prefix = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
        channels: 32, total_frames: 180, prefix_frames: latent_frames, height: 9, width: 16)
      noise_prefix = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
        channels: 3, total_frames: 720, prefix_frames: pixel_frames, height: 288, width: 512)
      decoder_latent = decoder_inputs.alloc("decoder.latent_prefix", latent_prefix.prefix_elements)
      decoder_noise = decoder_inputs.alloc("decoder.noise_prefix", noise_prefix.prefix_elements)
      prefixes << ML::Video::HyNomasTimelineOps::PrefixPlan.new(latent_prefix,
        source: timeline, destination: decoder_latent)
      prefixes << ML::Video::HyNomasTimelineOps::PrefixPlan.new(noise_prefix,
        source: noise, destination: decoder_noise)
    end
    decoder = HyNomasGeneration::Decoder.new(inputs.base, decoder_latent, decoder_metadata, timeline_noise: decoder_noise)
    timings["decoder_setup_seconds"] = (Time.monotonic - setup).total_seconds
    chain = ML::Metal::MPSCommandChain.new
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks
    run = Time.monotonic
    prefixes.each do |prefix|
      prefix.encode!(chain)
    end
    decoder.encode(chain)
    metrics["decoder_encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
    metrics["decoder_encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
    metrics["decoder_preterminal_waits"] = chain.wait_count.to_i64
    unless metrics["decoder_encode_allocations"] == 0 && metrics["decoder_encode_readbacks"] == 0 && chain.wait_count == 0
      raise "long decoder encode introduced allocation/readback/wait"
    end
    chain.commit_and_wait
    timings["decoder_wall_seconds"] = (Time.monotonic - run).total_seconds
    timings["decoder_gpu_seconds"] = chain.gpu_elapsed_seconds
    metrics["decoder_segments"] = chain.segment_count.to_i64
    metrics["decoder_waits"] = chain.wait_count.to_i64
    prefixes.each(&.release_after_drain!(chain))
    pixels = decoder.output
    path = File.join(output_dir, "pixels.f32le")
    HyNomasGeneration.write_fresh_output(path) { |f| f.write(Bytes.new(pixels.contents.as(UInt8*), pixels.size.to_i32)) }
    diagnostics["pixels"] = path
    diagnostics["pixels_sha256"] = PixelUMMBlockProbe.sha_file(path)
    HyNomasGeneration.emit("sequence_pixels_saved", output: path, frames: pixel_frames,
      decoder_wall_seconds: timings["decoder_wall_seconds"])
  end
  timings["request_seconds"] = (Time.monotonic - request_start).total_seconds
  receipt = {schema: "cogni_ml.hy_nomas_native_continuation/v1",
             upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
             inputs: inputs.file.path, inputs_sha256: inputs.file.sha256,
             prompt: inputs.base.prompt, qwen_valid_tokens: inputs.base.valid_text_tokens,
             qwen_input: "authenticated saved upstream state26 crop; native live text encoding excluded",
             conditioning: "one shared scene-level prompt; tagged action tails and per-segment prompts rejected",
             segments: segments, frames: dit_only ? 0 : pixel_frames, clean_latent_frames: latent_frames,
             latent_bank_shape: [1, 32, 180, 9, 16], width: 512, height: 288, fps: 24,
             steps_per_segment: 4, shift: 5, cfg: 1,
             dit_variant: weight_identity[:dit_variant],
             dit_weight_index_sha256: weight_identity[:dit_weight_index_sha256],
             cinematic_lineage: weight_identity[:cinematic_lineage],
             cinematic_lineage_sha256: weight_identity[:cinematic_lineage_sha256],
             cache_precision: "F32 exact native K/V boundary; upstream cache is BF16",
             text_cache_policy: text_cache_policy.once_per_scene? ? "once_per_scene" : "per_chunk",
             cache_writer: {sigma: 0.206897, timestep: 206.897, separate_noisy_forward: true},
             history: text_cache_policy.once_per_scene? ? "all prior chunk video rows plus the first chunk's scene text, layer-specific, no truncation" : "all prior chunk video/text rows, layer-specific, no truncation",
             noise_scope: "authenticated MPS-generated fixed banks; stock upstream CPU RNG value parity is false",
             pixelumm: {steps: 1, window_latents: 10, overlap_latents: 5,
                        global_windows: metadata.try(&.config.window_count)},
             phase_handoff: "one drained DiT chain then one decoder chain; shared GPU timeline",
             dit_peak_named_bytes: dit_peak, dit_steady_named_bytes: dit_steady,
             decoder_named_bytes: metadata.try(&.named_bytes), decoder_peak_named_bytes: decoder_peak,
             timings: timings, metrics: metrics, diagnostics: diagnostics,
             metric_scope: "instrumented MetalBuffer methods; opaque MPS resources and terminal exports excluded",
             full_upstream_video_parity: false, continuation_experimental: true}
  HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) { |f| f.puts(receipt.to_pretty_json) }
rescue ex
  request_error = ex
ensure
  may_release = true
  chain.try do |active_chain|
    unless active_chain.drained?
      begin
        active_chain.commit_and_wait
      rescue ex
        cleanup_errors << "terminal wait: #{ex.class}: #{ex.message}"
      end
    end
    may_release = active_chain.drained?
    unless may_release
      cleanup_errors << "terminal wait returned without proving GPU drain"
      # A failed wait has not proved that encoded work is finished. Keep every
      # owner strongly reachable and do not release any of its GPU buffers.
      HyNomasSequence.retain_in_flight do
        inputs.object_id
        metadata.try(&.object_id)
        dit.try(&.object_id)
        decoder.try(&.object_id)
        active_chain.object_id
        decoder_inputs.object_id
        current_arena.object_id
        timeline_arena.object_id
        prefixes.object_id
        nil
      end
    end
  end
  if may_release
    actions = [] of Proc(Nil)
    chain.try do |active_chain|
      prefixes.each do |prefix|
        if prefix.encoded? && !prefix.released?
          actions << -> { prefix.release_after_drain!(active_chain) }
        end
      end
    end
    decoder.try do |active_decoder|
      actions << -> { active_decoder.release }
    end
    dit.try do |active_dit|
      actions << -> { active_dit.release }
    end
    chain.try do |active_chain|
      actions << -> { active_chain.release }
    end
    actions << -> { decoder_inputs.release }
    actions << -> { current_arena.release }
    actions << -> { timeline_arena.release }
    metadata.try do |active_metadata|
      actions << -> { active_metadata.close }
    end
    cleanup_errors.concat(HyNomasSequence.release_actions(actions))
  end
end
cleanup_errors.each { |message| STDERR.puts("sequence cleanup: #{message}") }
if error = request_error
  raise error
end
unless cleanup_errors.empty?
  raise "sequence cleanup failed: #{cleanup_errors.join("; ")}"
end
