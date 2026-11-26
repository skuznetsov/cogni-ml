require "option_parser"
require "./support/pixelumm_saved_latent"

# A fixed, authenticated lifecycle experiment; no prompts, media export or full
# generation. Both arms decode the same saved latent after an optional DiT
# prelude, then repeat on the same decoder owners after one terminal wait.
alias Probe = PixelUMMAfterDiTProbe
mode = ""
output_dir = ""
plan_only = false
OptionParser.parse do |p|
  p.on("--mode MODE", "standalone or after-dit") { |v| mode = v }
  p.on("--output-dir PATH", "Existing empty external-SSD result directory") { |v| output_dir = v }
  p.on("--plan-only", "Authenticate fixtures and budget without initializing Metal") { plan_only = true }
end
raise ArgumentError.new("mode must be standalone or after-dit") unless {"standalone", "after-dit"}.includes?(mode)
unless plan_only
  unless Path[output_dir].absolute? && File.directory?(output_dir) && Dir.empty?(output_dir) &&
         File.realpath(output_dir).starts_with?(File.realpath("/opt/mldata_ssd") + "/")
    raise ArgumentError.new("result directory must be an existing empty external-SSD directory")
  end
  output_dir = File.realpath(output_dir)
end
request_start = Time.instant
input_path = Probe.authenticate!(Probe::INPUTS, Probe::INPUT_SHA)
latent_path = Probe.authenticate!(Probe::LATENT, Probe::LATENT_SHA)
Probe.authenticate!(Probe::ORACLE_REPORT, Probe::ORACLE_REPORT_SHA)
oracle_path = Probe.authenticate!(Probe::ORACLE_PIXELS, Probe::ORACLE_SHA)
geometry = HyNomasGeneration::SpatialGeometry::CANDIDATE_768
inputs = HyNomasGeneration::Inputs.new(input_path, Probe::INPUT_SHA, conditioned: true, geometry: geometry)
raise "saved decoder noise identity mismatch" unless inputs.records["decoder.noise"].sha256 == Probe::NOISE_SHA
inputs.file.verify_payload!
metadata = HyNomasGeneration::DecoderMetadata.new(inputs, Probe::BUDGET, capture_raw: false)
config = metadata.config
raise "saved latent size mismatch" unless File.size(latent_path) == config.latent_elements * sizeof(Float32)
raise "upstream display size mismatch" unless File.size(oracle_path) == config.output_elements * sizeof(Float32)
cinematic = HyNomasCinematic.load!(Probe::CINEMATIC, Probe::CINEMATIC_SHA,
  bf16: true, camera_conditioning: false, action_conditioning: false)
admission = HyNomasGeneration.authenticate_bf16_index(geometry.dit_config, cinematic: cinematic)
# A saved source is retained in both phases. The cold display snapshot exists
# only in the decoder phase, and replaces the old optional raw-capture buffer.
source_bytes = config.latent_elements * sizeof(Float32)
dit_named = HyNomasGeneration.dit_resident_weight_bytes(HyNomasGeneration::DiTWeights::BF16,
  HyNomasGeneration.stage_bytes, admission.payload_bytes) +
            HyNomasGeneration.dit_reserve_bytes(inputs, 4, geometry.dit_config) + source_bytes
decoder_named = metadata.named_bytes + source_bytes + config.output_elements * sizeof(Float32)
raise "aggregate experiment buffers exceed the 24 GiB named budget" if {dit_named, decoder_named}.max > Probe::BUDGET
HyNomasGeneration.emit("lifecycle_memory_plan", mode: mode, dit_named_bytes: dit_named,
  decoder_named_bytes: decoder_named, named_budget_bytes: Probe::BUDGET,
  reserve_pct: 10, raw_capture: false, resolution: "768x416")
if plan_only
  metadata.close
  exit
end
raw_latent = File.read(latent_path)
raise "saved latent changed during read" unless Digest::SHA256.hexdigest(raw_latent) == Probe::LATENT_SHA
raw_latent.to_unsafe.as(Float32*).to_slice((raw_latent.bytesize // sizeof(Float32)).to_i32).each do |v|
  raise "non-finite saved latent" unless v.finite?
end
shared = ML::Video::Arena.new
source_latent = shared.alloc("saved_source", config.latent_elements.to_i32)
source_latent.write_bytes(raw_latent.to_slice.to_unsafe, raw_latent.bytesize)
latent = inputs.upload!("input.latent", shared)
snapshot_arena = ML::Video::Arena.new
decoder = nil.as(HyNomasGeneration::Decoder?)
dit = nil.as(HyNomasGeneration::DiT?)
chains = [] of ML::Metal::MPSCommandChain
passes = [] of NamedTuple(label: String, wall_seconds: Float64, encode_seconds: Float64,
  terminal_wait_seconds: Float64, gpu_seconds: Float64, segments: Int32, waits: Int32,
  encode_allocations: Int64, encode_readbacks: Int64, preterminal_waits: Int32)
timings = {} of String => Float64
begin
  if mode == "after-dit"
    setup = Time.instant
    dit = HyNomasGeneration::DiT.new(inputs, latent, Probe::BUDGET - source_bytes,
      HyNomasGeneration::DiTWeights::BF16, steps: 4, admission: admission, cinematic: cinematic)
    timings["dit_setup_seconds"] = (Time.instant - setup).total_seconds
    chain = ML::Metal::MPSCommandChain.new
    chains << chain
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks
    start = Time.instant
    dit.encode(chain)
    unless ML::MetalBuffer.hy_text_probe_allocations == allocations &&
           ML::MetalBuffer.hy_text_probe_readbacks == readbacks && chain.wait_count == 0
      raise "DiT prelude introduced an allocation/readback/wait"
    end
    chain.commit_and_wait
    timings["dit_wall_seconds"] = (Time.instant - start).total_seconds
    timings["dit_gpu_seconds"] = chain.gpu_elapsed_seconds
    dit.release
    dit = nil
    chain.release
    # No tensor read, export, or hash between the prelude and decoder setup.
    HyNomasGeneration.emit("lifecycle_dit_completed", wall_seconds: timings["dit_wall_seconds"])
  end
  cold_output = snapshot_arena.alloc("cold_display", config.output_elements.to_i32)
  setup = Time.instant
  decoder = HyNomasGeneration::Decoder.new(inputs, latent, metadata)
  timings["decoder_setup_seconds"] = (Time.instant - setup).total_seconds
  owner_ids = decoder.diagnostic_owner_ids
  poison = ML::Metal::ComputePipeline.new("pixelumm_decoder_replay_poison", {{ read_file("#{__DIR__}/support/pixelumm_decoder_replay.metal") }})
  {"cold", "warm"}.each_with_index do |label, index|
    if index == 1
      decoder.diagnostic_restart!
      # Completion/wait counters remain observable after chain release. Match
      # ordinary request cleanup instead of retaining completed root resources.
      chains.last.release
      sleep 1.second
    end
    chain = ML::Metal::MPSCommandChain.new
    chains << chain
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks
    start = Time.instant
    # Both arms use the same GPU source copy and complete-output poison. The
    # timed warm request cannot pass by exporting the previous display output.
    ML::Video::MetalOps.copy(chain.encoder, source_latent, latent, config.latent_elements.to_i32)
    e = chain.encoder
    e.set_pipeline(poison).set_buffer(decoder.output, 0, ML::Metal::BufferAccess::Write)
      .set_value(config.output_elements.to_u32, 1)
    e.dispatch_1d(config.output_elements.to_i32)
    decoder.encode(chain)
    if index == 0
      ML::Video::MetalOps.copy(chain.encoder, decoder.output, cold_output, config.output_elements.to_i32)
    end
    encode_seconds = (Time.instant - start).total_seconds
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
    preterminal_waits = chain.wait_count
    unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
      raise "decoder replay introduced an allocation/readback/wait"
    end
    wait_start = Time.instant
    chain.commit_and_wait
    terminal_wait_seconds = (Time.instant - wait_start).total_seconds
    wall_seconds = (Time.instant - start).total_seconds
    raise "decoder owners changed during replay" unless decoder.diagnostic_owner_ids == owner_ids
    passes << {label: label, wall_seconds: wall_seconds, encode_seconds: encode_seconds,
               terminal_wait_seconds: terminal_wait_seconds, gpu_seconds: chain.gpu_elapsed_seconds,
               segments: chain.segment_count, waits: chain.wait_count, encode_allocations: encode_allocations,
               encode_readbacks: encode_readbacks, preterminal_waits: preterminal_waits}
    HyNomasGeneration.emit("lifecycle_decoder_completed", label: label, wall_seconds: wall_seconds,
      gpu_seconds: chain.gpu_elapsed_seconds)
  end
  # All tensor exports and hashing are deferred until both terminal waits have
  # completed. No diagnostic CPU tensor work sits between the measured passes.
  cold_result = Probe.export!(File.join(output_dir, "cold-display.f32le"), cold_output)
  warm_result = Probe.export!(File.join(output_dir, "warm-display.f32le"), decoder.output)
  receipt = {schema: "cogni_ml.pixelumm_after_dit_probe/v1", mode: mode,
             upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
             input_sha256: Probe::INPUT_SHA, latent_sha256: Probe::LATENT_SHA, noise_sha256: Probe::NOISE_SHA,
             oracle_report_sha256: Probe::ORACLE_REPORT_SHA, oracle_display_sha256: Probe::ORACLE_SHA,
             decoder_cache_sha256: HyNomasGeneration::PIXEL_CACHE_SHA,
             cinematic_lineage_sha256: Probe::CINEMATIC_SHA, geometry: "768x416", frames: 120,
             raw_capture: false, named_budget_bytes: Probe::BUDGET, dit_named_bytes: dit_named,
             decoder_named_bytes: decoder_named, reused_owner_ids: owner_ids, passes: passes,
             timings: timings, request_seconds: (Time.instant - request_start).total_seconds,
             outputs: {cold: cold_result, warm: warm_result}, atol: 2e-5, rtol: 2e-5,
             resource_scope: "same live decoder owners within each cell; identical construction, values and geometry across fresh processes",
             measurement_scope: "GPU latent restore and output poison in both passes; cold also snapshots display on GPU; one terminal wait per request; all exports after both passes",
             upstream_scope: "authenticated saved source-MPS PixelUMM oracle; stock CUDA/Flex and full DiT trajectory parity excluded",
             counter_scope: "instrumented MetalBuffer methods only; opaque MPS allocations are guarded externally"}
  HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) { |f| f.puts(receipt.to_pretty_json) }
ensure
  # Failed completion must not free buffers that may still be referenced by the
  # GPU. Exit the bounded subprocess immediately and let the OS reclaim them.
  begin
    chains.each(&.commit_and_wait)
    decoder.try(&.release)
    dit.try(&.release)
    chains.each(&.release)
    snapshot_arena.release
    shared.release
    metadata.close
  rescue ex
    STDERR.puts("lifecycle cleanup could not establish terminal completion: #{ex.message}")
    Process.exit(1)
  end
end
