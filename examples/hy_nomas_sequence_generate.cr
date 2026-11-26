require "option_parser"
require "./support/hy_nomas_sequence"
require "./support/hy_nomas_reference_encoder"
require "./support/hy_nomas_decode_latent_admission"

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
decoder_cache_path = ""
decoder_cache_sha = ""
reference_image_inputs_path = ""
reference_image_inputs_sha = ""
reference_encoder_path = ""
reference_encoder_sha = ""
reference_action_inputs_path = ""
reference_action_inputs_sha = ""
reference_action_mask_path = ""
reference_action_mask_sha = ""
allow_unqualified_reference = false
source_bf16 = false
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
  p.on("--decoder-cache PATH", "Opt-in authenticated PixelUMM BF16 cache manifest") { |v| decoder_cache_path = v }
  p.on("--decoder-cache-sha SHA", "Trusted PixelUMM cache manifest SHA-256") { |v| decoder_cache_sha = v }
  p.on("--reference-image-inputs PATH", "Authenticated prepared RGB8 reference image manifest") { |v| reference_image_inputs_path = v }
  p.on("--reference-image-inputs-sha SHA", "Trusted reference image manifest SHA-256") { |v| reference_image_inputs_sha = v }
  p.on("--reference-encoder PATH", "Experimental native BF16 DCAE encoder safetensors") { |v| reference_encoder_path = v }
  p.on("--reference-encoder-sha SHA", "Trusted native DCAE encoder SHA-256") { |v| reference_encoder_sha = v }
  p.on("--reference-action-inputs PATH", "Separate authenticated T1 action conditioning; requires two-segment reference initialization") { |v| reference_action_inputs_path = v }
  p.on("--reference-action-inputs-sha SHA", "Trusted T1 action conditioning manifest SHA-256") { |v| reference_action_inputs_sha = v }
  p.on("--reference-action-mask PATH", "Authenticated upstream action offsets and video windows") { |v| reference_action_mask_path = v }
  p.on("--reference-action-mask-sha SHA", "Trusted action-mask profile SHA-256") { |v| reference_action_mask_sha = v }
  p.on("--allow-unqualified-reference", "Explicitly opt in to experimental image-conditioned continuation") { allow_unqualified_reference = true }
  p.on("--reference-source-bf16", "Experimental source-BF16 DiT and history attention for two reference segments") { source_bf16 = true }
  p.on("--plan-only", "CPU-only authentication and aggregate admission") { plan_only = true }
end
decoder_cache_identity = HyNomasLatentDecodeAdmission.resolve_decoder_cache!(decoder_cache_path, decoder_cache_sha)
HyNomasSequence.validate_segments!(segments)
raise ArgumentError.new("--inputs and --inputs-sha are required") if input_path.empty? || input_sha.empty?
unless (24_576..49_152).includes?(budget_mib)
  raise ArgumentError.new("sequence named budget must be within 24..48 GiB")
end
budget = budget_mib * 1024 * 1024
request_start = Time.monotonic
reference_requested = HyNomasReferenceEncoder.validate_cli!(
  reference_image_inputs_path, reference_image_inputs_sha,
  reference_encoder_path, reference_encoder_sha,
  allow_unqualified: allow_unqualified_reference, plan_only: plan_only,
  cinematic_requested: !cinematic_path.empty? || !cinematic_sha.empty?,
  legacy_text: !text_cache_policy.once_per_scene?)
unless reference_action_inputs_path.empty? == reference_action_inputs_sha.empty?
  raise ArgumentError.new("reference action requires both --reference-action-inputs and --reference-action-inputs-sha")
end
reference_action_requested = !reference_action_inputs_path.empty?
unless reference_action_mask_path.empty? == reference_action_mask_sha.empty?
  raise ArgumentError.new("reference action masks require both --reference-action-mask and --reference-action-mask-sha")
end
if !reference_action_mask_path.empty? && !reference_action_requested
  raise ArgumentError.new("reference action masks require separate action conditioning")
end
if reference_action_requested && (!reference_requested || segments != 2)
  raise ArgumentError.new("reference action requires exactly two segments and reference image initialization")
end
reference_history_plan = reference_requested ? HyNomasReferenceEncoder::ReferenceHistoryPlan.for(segments) : nil
reference_inputs = nil.as(ML::Video::HyNomasReferenceImage::Inputs?)
reference_session = nil.as(HyNomasReferenceEncoder::Session?)
if reference_requested
  reference_inputs = ML::Video::HyNomasReferenceImage::Inputs.new(
    ML::Video::HyNomasReferenceImage.canonical_manifest_argument!(reference_image_inputs_path), reference_image_inputs_sha,
    reference_encoder_sha)
end
inputs = HyNomasGeneration::TimelineInputs.new(File.realpath(input_path), input_sha)
inputs.verify_payload!
inputs.base.file.verify_payload!
HyNomasSequence.validate_shared_scene_prompt!(inputs.base.prompt)
reference_action_inputs = nil.as(HyNomasGeneration::Inputs?)
reference_action = nil.as(ML::Video::HyNomasReferenceAction?)
if reference_action_requested
  action_inputs = HyNomasGeneration::Inputs.new(File.realpath(reference_action_inputs_path), reference_action_inputs_sha,
    conditioned: true)
  action_inputs.file.verify_payload!
  reference_action_inputs = action_inputs
  action_profile = if reference_action_mask_path.empty?
                     nil
                   else
                     ML::Video::HyNomasActionConditioning::Profile.from_file(
                       File.realpath(reference_action_mask_path), reference_action_mask_sha,
                       conditioning_manifest_sha256: action_inputs.file.sha256,
                       conditioning_payload_sha256: action_inputs.file.document["payload_sha256"].as_s,
                       prompt: action_inputs.prompt, valid_text_tokens: action_inputs.valid_text_tokens)
                   end
  reference_action = ML::Video::HyNomasReferenceAction.new(run_segments: segments,
    conditioning_manifest_sha256: action_inputs.file.sha256,
    conditioning_payload_sha256: action_inputs.file.document["payload_sha256"].as_s,
    prompt_sha256: Digest::SHA256.hexdigest(action_inputs.prompt),
    valid_text_tokens: action_inputs.valid_text_tokens, action_conditioning: action_profile)
end
cinematic = HyNomasCinematic.load!(cinematic_path, cinematic_sha,
  bf16: true, camera_conditioning: false, action_conditioning: false)
weight_identity = HyNomasSequence.weight_identity(cinematic)
admission = HyNomasGeneration.authenticate_bf16_index(cinematic: cinematic)
reference_latent_bytes = reference_requested ? HyNomasSequence::REFERENCE_LATENT_BYTES : 0_i64
HyNomasSequence.validate_reference_action!(reference_action, reference_action_inputs,
  caption_prompt: inputs.base.prompt,
  segments: segments, reference_latent_bytes: reference_latent_bytes,
  text_cache_policy: text_cache_policy, cinematic: cinematic)
dit_peak = HyNomasSequence.peak_named_bytes(inputs, segments, text_cache_policy,
  reference_latent_bytes: reference_latent_bytes, reference_action: reference_action,
  source_bf16: source_bf16)
dit_steady = dit_peak - HyNomasSequence.old_attention_bytes(inputs.base, reference_action: reference_action)
raise ArgumentError.new("sequence peak named buffers exceed budget") unless dit_peak <= budget
latent_frames = segments * 30
pixel_frames = latent_frames * 4
metadata = dit_only ? nil : HyNomasGeneration::DecoderMetadata.new(inputs.base, budget,
  total_latent_frames: latent_frames, cache_identity: decoder_cache_identity)
# A shorter decoder consumes gathered prefixes. Their destinations are already
# counted by DecoderMetadata; retain both full source banks until its chain drains.
decoder_peak = metadata.try do |decoder_metadata|
  retained_sources = segments < 6 ? inputs.bytes(%w(input.latent_bank decoder.noise)) : 0_i64
  decoder_metadata.named_bytes + retained_sources
end
raise ArgumentError.new("decoder source banks and prefix buffers exceed budget") if decoder_peak && decoder_peak > budget
preflight_seconds = (Time.monotonic - request_start).total_seconds
reference_encoder_peak_named_bytes = reference_requested ? HyNomasReferenceEncoder::ResidencyPlan.for(HyNomasReferenceEncoder::FIXED_CFG).named_bytes : nil
emit_memory_plan = -> do
  HyNomasGeneration.emit("sequence_memory_plan", segments: segments, steps_per_segment: 4,
    dit_peak_named_bytes: dit_peak, dit_steady_named_bytes: dit_steady,
    history_named_bytes: HyNomasSequence.history_bytes(inputs.base, segments, text_cache_policy,
      reference_action: reference_action, source_bf16: source_bf16),
    decoder_named_bytes: metadata.try(&.named_bytes), decoder_peak_named_bytes: decoder_peak,
    frames: pixel_frames, named_budget_bytes: budget,
    cache_precision: source_bf16 ? "F32 storage of source-BF16 post-RoPE K/V" : "F32 native post-RoPE K/V; no additional BF16 rounding",
    source_bf16: source_bf16,
    dit_variant: weight_identity[:dit_variant],
    dit_weight_index_sha256: weight_identity[:dit_weight_index_sha256],
    cinematic_lineage: weight_identity[:cinematic_lineage],
    cinematic_lineage_sha256: weight_identity[:cinematic_lineage_sha256],
    text_cache_policy: text_cache_policy.once_per_scene? ? "once_per_scene" : "per_chunk",
    text_conditioning_mode: reference_action.try(&.action_conditioning) ? "reference_caption_then_masked_action" : (reference_action ? "reference_caption_then_action" : "shared_scene"),
    reference_action: reference_action.try(&.identity),
    segment_text_tokens: reference_action.try(&.text_tokens(inputs.base.valid_text_tokens)),
    segment_cache_text: reference_action.try(&.cache_text),
    reference_initialization: reference_requested,
    reference_manifest: reference_inputs.try(&.manifest_path),
    reference_manifest_sha256: reference_inputs.try(&.manifest_sha256),
    reference_encoder: reference_requested ? File.realpath(reference_encoder_path) : nil,
    reference_encoder_sha256: reference_requested ? reference_encoder_sha : nil,
    reference_encoder_peak_named_bytes: reference_encoder_peak_named_bytes,
    reference_encoder_memory_scope: reference_requested ? HyNomasReferenceEncoder::MPS_MEMORY_SCOPE : nil,
    reference_latent_bytes: reference_latent_bytes,
    reference_parity_status: reference_requested ? "experimental continuation; full upstream cache/decoder/video parity not established" : nil,
    reference_generated_segment_indices: reference_history_plan.try(&.generated_segment_indices),
    reference_generated_cache_writer_segment_indices: reference_history_plan.try(&.generated_cache_writer_segment_indices),
    reference_generated_duration_seconds: reference_history_plan.try(&.generated_duration_seconds),
    reference_total_timeline_duration_seconds: reference_history_plan.try(&.total_timeline_duration_seconds),
    cache_writer: reference_requested ? "technical T0 DCAE reference initialization without denoising; generated T1 onward uses regular denoise" : "separate noisy forward after four denoise forwards; sigma 0.206897, t 206.897",
    full_upstream_video_parity: false)
end
if plan_only
  begin
    if reference_requested
      reference_session = HyNomasReferenceEncoder::Session.new(reference_inputs.not_nil!,
        reference_encoder_path, reference_encoder_sha, budget)
    end
    emit_memory_plan.call
  ensure
    begin
      reference_session.try(&.close!)
    ensure
      metadata.try(&.close)
    end
  end
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
  if reference_requested
    reference_session = HyNomasReferenceEncoder::Session.new(reference_inputs.not_nil!,
      reference_encoder_path, reference_encoder_sha, budget)
  end
  emit_memory_plan.call
  if reference = reference_session
    setup = Time.monotonic
    reference.prepare!
    timings["reference_encoder_setup_seconds"] = (Time.monotonic - setup).total_seconds
    chain = ML::Metal::MPSCommandChain.new
    run = Time.monotonic
    reference.encode!(chain)
    reference_metrics = reference.metrics.not_nil!
    metrics["reference_encoder_encode_allocations"] = reference_metrics[:allocations]
    metrics["reference_encoder_encode_readbacks"] = reference_metrics[:readbacks]
    metrics["reference_encoder_preterminal_waits"] = reference_metrics[:preterminal_waits]
    chain.commit_and_wait
    timings["reference_encoder_wall_seconds"] = (Time.monotonic - run).total_seconds
    timings["reference_encoder_gpu_seconds"] = chain.gpu_elapsed_seconds
    metrics["reference_encoder_segments"] = chain.segment_count.to_i64
    metrics["reference_encoder_waits"] = chain.wait_count.to_i64
    reference.finish_encoder_phase!(chain)
    chain.release
    chain = nil
  end

  timeline = inputs.upload!("input.latent_bank", timeline_arena)
  latent = current_arena.alloc("input.latent", 32 * 30 * 9 * 16)
  setup = Time.monotonic
  reference_latent = reference_session.try(&.latent_for_dit)
  dit = HyNomasSequence::DiT.new(inputs, timeline, latent, budget, segments, admission,
    text_cache_policy, cinematic: cinematic, reference_latent: reference_latent,
    reference_action: reference_action, reference_action_inputs: reference_action_inputs,
    source_bf16: source_bf16)
  raise "sequence preflight/allocation admission disagree" unless dit.named_bytes == dit_steady
  timings["dit_setup_seconds"] = (Time.monotonic - setup).total_seconds
  chain = ML::Metal::MPSCommandChain.new
  allocations = ML::MetalBuffer.hy_text_probe_allocations
  readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  run = Time.monotonic
  reference_session.try(&.bind_dit_chain!(chain))
  dit.encode(chain)
  metrics["dit_encode_allocations"] = ML::MetalBuffer.hy_text_probe_allocations - allocations
  metrics["dit_encode_readbacks"] = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
  metrics["dit_preterminal_waits"] = chain.wait_count.to_i64
  HyNomasReferenceEncoder.validate_dit_encode_metrics!(metrics)
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
  reference_session.try(&.release_after_dit_drain!(chain))
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
             conditioning: if profile = reference_action.try(&.action_conditioning)
               profile.sequential_actions? ? "caption-only T0 with Subject1 Alex and Subject2 Resume; authenticated sequential T1 Action1/Action2 offsets and temporal attention masks" : "caption-only T0 with Subject1 Alex; authenticated T1 Action1 offsets and temporal attention mask"
             else
               reference_action ? "caption-only reference T0 then separately authenticated T1 action text; tagged action tails rejected" : "one shared scene-level prompt; tagged action tails and per-segment prompts rejected"
             end,
             text_conditioning_mode: reference_action.try(&.action_conditioning) ? "reference_caption_then_masked_action" : (reference_action ? "reference_caption_then_action" : "shared_scene"),
             reference_action: reference_action.try(&.identity),
             segment_text_tokens: reference_action.try(&.text_tokens(inputs.base.valid_text_tokens)),
             segment_cache_text: reference_action.try(&.cache_text),
             segments: segments, frames: dit_only ? 0 : pixel_frames, clean_latent_frames: latent_frames,
             reference_initialization: reference_requested,
             reference_history: reference_requested ? "T0 is a technical DCAE image-reference initialization with one cache writer and no denoising; generated segments start at T1" : nil,
             technical_reference_segment_index: reference_requested ? 0 : nil,
             technical_reference_latent_frames: reference_requested ? 30 : nil,
             generated_segment_start_index: reference_requested ? 1 : 0,
             generated_segment_indices: reference_history_plan.try(&.generated_segment_indices) || (0...segments).to_a,
             generated_segment_count: reference_history_plan.try(&.generated_segment_indices.size) || segments,
             generated_latent_frames: reference_requested ? (segments - 1) * 30 : latent_frames,
             generated_duration_seconds: reference_history_plan.try(&.generated_duration_seconds) || segments * 5,
             pixel_output_duration_seconds: dit_only ? 0 : segments * 5,
             pixel_output_contains_technical_reference: reference_requested && !dit_only,
             generated_cache_writers: reference_history_plan.try do |history|
               {scope: "generated segments only", segment_indices: history.generated_cache_writer_segment_indices,
                writer_count: history.generated_cache_writer_segment_indices.size,
                denoise_forwards_before_each: 4,
                separate_noisy_forward: true, sigma: 0.206897, timestep: 206.897,
                final_generated_segment_index: history.final_generated_segment_index,
                final_generated_segment_has_writer: false}
             end,
             reference_manifest: reference_inputs.try(&.manifest_path),
             reference_manifest_sha256: reference_inputs.try(&.manifest_sha256),
             reference_image_sha256: reference_inputs.try(&.image_sha256),
             reference_payload_sha256: reference_inputs.try(&.payload_sha256),
             reference_encoder_sha256: reference_requested ? reference_encoder_sha : nil,
             reference_encoder_peak_named_bytes: reference_session.try(&.named_bytes),
             reference_encoder_memory_scope: reference_requested ? HyNomasReferenceEncoder::MPS_MEMORY_SCOPE : nil,
             reference_latent_bytes: reference_latent_bytes,
             reference_parity_status: reference_requested ? "experimental continuation; full upstream cache/decoder/video parity not established" : nil,
             latent_bank_shape: [1, 32, 180, 9, 16], width: 512, height: 288, fps: 24,
             steps_per_segment: 4, shift: 5, cfg: 1,
             dit_variant: weight_identity[:dit_variant],
             dit_weight_index_sha256: weight_identity[:dit_weight_index_sha256],
             cinematic_lineage: weight_identity[:cinematic_lineage],
             cinematic_lineage_sha256: weight_identity[:cinematic_lineage_sha256],
             cache_precision: source_bf16 ? "F32 storage of source-BF16 post-RoPE K/V" : "F32 exact native K/V boundary; upstream cache is BF16",
             source_bf16: source_bf16,
             text_cache_policy: text_cache_policy.once_per_scene? ? "once_per_scene" : "per_chunk",
             cache_writer: {kind: reference_requested ? "technical_t0_reference_init" : "separate_noisy_forward",
                            scope: reference_requested ? "technical reference segment T0 only" : "each nonfinal generated segment",
                            segment_indices: reference_requested ? [0] : (0...(segments - 1)).to_a,
                            sigma: 0.206897, timestep: 206.897,
                            separate_noisy_forward: true,
                            denoise_forwards: reference_requested ? 0 : 4,
                            writer_forwards: 1},
             history: reference_action ? "T0 video and caption text history plus fresh T1 action text; layer-specific, no truncation" : (text_cache_policy.once_per_scene? ? "all prior chunk video rows plus the first chunk's scene text, layer-specific, no truncation" : "all prior chunk video/text rows, layer-specific, no truncation"),
             noise_scope: "authenticated MPS-generated fixed banks; stock upstream CPU RNG value parity is false",
             pixelumm: {steps: 1, window_latents: 10, overlap_latents: 5,
                        global_windows: metadata.try(&.config.window_count)},
             phase_handoff: reference_requested ? "one drained DCAE encoder chain, one drained DiT chain, then one decoder chain; BF16 reference latent retained only through the DiT drain" : "one drained DiT chain then one decoder chain; shared GPU timeline",
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
        reference_session.try(&.object_id)
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
    reference_session.try do |active_reference|
      actions << -> { active_reference.close! }
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
