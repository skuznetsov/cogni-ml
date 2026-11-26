require "option_parser"
require "./support/pixelumm_saved_latent"
require "./support/pixelumm_recovery_timing"

# Three identical saved-latent decodes on one set of GPU owners. Only the idle
# interval after the warmup / first measurement changes. No DiT is constructed.
module PixelUMMRecoveryProbe
  lib Host
    fun thermal_state = gs_pixelumm_recovery_thermal_state : Int32
  end

  def self.thermal_state : Int32
    state = Host.thermal_state
    raise "invalid advisory NSProcessInfo thermal state" unless (0..3).includes?(state)
    state
  end

  alias Pass = NamedTuple(label: String, output_index: Int32,
    requested_recovery_seconds: Int32, observed_recovery_seconds: Float64,
    observed_recovery_adjusted_seconds: Float64, observed_recovery_wall_seconds: Float64,
    started_unix_ms: Int64, finished_unix_ms: Int64,
    started_monotonic_ns: UInt64, finished_monotonic_ns: UInt64,
    thermal_before: Int32, thermal_after: Int32, wall_seconds: Float64,
    encode_seconds: Float64, terminal_wait_seconds: Float64,
    gpu_seconds: Float64, segments: Int32, waits: Int32,
    encode_allocations: Int64, encode_readbacks: Int64, preterminal_waits: Int32)
end

alias Probe = PixelUMMAfterDiTProbe
order = ""
output_dir = ""
plan_only = false
OptionParser.parse do |p|
  p.on("--recovery-order ORDER", "short-long or long-short after one warmup") { |v| order = v }
  p.on("--output-dir PATH", "Existing empty external-SSD result directory") { |v| output_dir = v }
  p.on("--plan-only", "Authenticate fixtures and budget without initializing Metal") { plan_only = true }
end
raise ArgumentError.new("recovery order must be short-long or long-short") unless {"short-long", "long-short"}.includes?(order)
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
source_bytes = config.latent_elements * sizeof(Float32)
decoder_named = metadata.named_bytes + source_bytes + 3_i64 * config.output_elements * sizeof(Float32)
raise "aggregate experiment buffers exceed the 24 GiB named budget" if decoder_named > Probe::BUDGET
HyNomasGeneration.emit("recovery_memory_plan", recovery_order: order, decoder_named_bytes: decoder_named,
  named_budget_bytes: Probe::BUDGET, frames: 120, geometry: "768x416", dit_constructed: false)
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
snapshots = Array(ML::MetalBuffer).new(3) { |i| shared.alloc("display_#{i}", config.output_elements.to_i32) }
decoder = nil.as(HyNomasGeneration::Decoder?)
chains = [] of ML::Metal::MPSCommandChain
passes = [] of PixelUMMRecoveryProbe::Pass
setup_seconds = 0.0
begin
  setup = Time.instant
  decoder = HyNomasGeneration::Decoder.new(inputs, latent, metadata)
  setup_seconds = (Time.instant - setup).total_seconds
  owner_ids = decoder.diagnostic_owner_ids
  poison = ML::Metal::ComputePipeline.new("pixelumm_decoder_replay_poison", {{ read_file("#{__DIR__}/support/pixelumm_decoder_replay.metal") }})
  labels = ["warmup"] + order.split('-')
  prior_terminal = nil.as(UInt64?)
  prior_terminal_adjusted = nil.as(UInt64?)
  clock = PixelUMMRecoveryTiming::Clock.new
  labels.each_with_index do |label, index|
    requested_recovery = label == "warmup" ? 0 : (label == "short" ? 1 : 120)
    if index > 0
      decoder.diagnostic_restart!
      chains.last.release
      HyNomasGeneration.emit("recovery_idle_started", label: label, seconds: requested_recovery,
        started_unix_ms: Time.utc.to_unix_ms)
      PixelUMMRecoveryTiming.wait_until(clock, prior_terminal.not_nil!, requested_recovery)
    end
    chain = ML::Metal::MPSCommandChain.new
    chains << chain
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    readbacks = ML::MetalBuffer.hy_text_probe_readbacks
    thermal_before = PixelUMMRecoveryProbe.thermal_state
    started_unix_ms = Time.utc.to_unix_ms
    start = Time.instant
    started_monotonic_ns = clock.now_ns
    started_adjusted_ns = clock.adjusted_now_ns
    observed_recovery = prior_terminal.try { |previous| (started_monotonic_ns - previous) / 1e9 } || 0.0
    observed_recovery_adjusted = prior_terminal_adjusted.try { |previous| (started_adjusted_ns - previous) / 1e9 } || 0.0
    observed_recovery_wall = passes.empty? ? 0.0 : (started_unix_ms - passes.last[:finished_unix_ms]) / 1000.0
    upper_recovery = label == "short" ? 5.0 : 125.0
    HyNomasGeneration.emit("recovery_interval_observed", label: label,
      observed_recovery_seconds: observed_recovery, observed_recovery_adjusted_seconds: observed_recovery_adjusted,
      observed_recovery_wall_seconds: observed_recovery_wall, requested_recovery_seconds: requested_recovery)
    if index > 0 && !(requested_recovery.to_f64..upper_recovery).includes?(observed_recovery)
      raise "actual recovery interval #{observed_recovery}s is outside #{requested_recovery}..#{upper_recovery}s"
    end
    ML::Video::MetalOps.copy(chain.encoder, source_latent, latent, config.latent_elements.to_i32)
    e = chain.encoder
    e.set_pipeline(poison).set_buffer(decoder.output, 0, ML::Metal::BufferAccess::Write)
      .set_value(config.output_elements.to_u32, 1)
    e.dispatch_1d(config.output_elements.to_i32)
    decoder.encode(chain)
    # Identical GPU snapshot work in warmup and both measured requests.
    ML::Video::MetalOps.copy(chain.encoder, decoder.output, snapshots[index], config.output_elements.to_i32)
    encode_seconds = (Time.instant - start).total_seconds
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks
    preterminal_waits = chain.wait_count
    unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
      raise "decoder replay introduced an allocation/readback/wait"
    end
    wait_start = Time.instant
    chain.commit_and_wait
    finished = Time.instant
    finished_monotonic_ns = clock.now_ns
    finished_adjusted_ns = clock.adjusted_now_ns
    prior_terminal = finished_monotonic_ns
    prior_terminal_adjusted = finished_adjusted_ns
    finished_unix_ms = Time.utc.to_unix_ms
    thermal_after = PixelUMMRecoveryProbe.thermal_state
    terminal_wait_seconds = (finished - wait_start).total_seconds
    wall_seconds = (finished - start).total_seconds
    raise "decoder owners changed during recovery replay" unless decoder.diagnostic_owner_ids == owner_ids
    passes << {label: label, output_index: index, requested_recovery_seconds: requested_recovery,
               observed_recovery_seconds: observed_recovery, started_unix_ms: started_unix_ms,
               observed_recovery_adjusted_seconds: observed_recovery_adjusted,
               observed_recovery_wall_seconds: observed_recovery_wall,
               finished_unix_ms: finished_unix_ms, started_monotonic_ns: started_monotonic_ns,
               finished_monotonic_ns: finished_monotonic_ns, thermal_before: thermal_before, thermal_after: thermal_after,
               wall_seconds: wall_seconds, encode_seconds: encode_seconds,
               terminal_wait_seconds: terminal_wait_seconds, gpu_seconds: chain.gpu_elapsed_seconds,
               segments: chain.segment_count, waits: chain.wait_count, encode_allocations: encode_allocations,
               encode_readbacks: encode_readbacks, preterminal_waits: preterminal_waits}
    HyNomasGeneration.emit("recovery_decoder_completed", label: label, wall_seconds: wall_seconds,
      gpu_seconds: chain.gpu_elapsed_seconds, thermal_before: thermal_before, thermal_after: thermal_after,
      finished_unix_ms: finished_unix_ms)
  end
  outputs = labels.map_with_index do |label, index|
    result = Probe.export!(File.join(output_dir, "#{label}-display.f32le"), snapshots[index])
    {label: label, output_index: index, path: result[:path], sha256: result[:sha256],
     upstream_bitwise_equal: result[:upstream_bitwise_equal], exported_unix_ms: Time.utc.to_unix_ms}
  end
  receipt = {schema: "cogni_ml.pixelumm_recovery_probe/v1", recovery_order: order,
             recovery_clock: "CLOCK_MONOTONIC_RAW(4)",
             upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
             input_sha256: Probe::INPUT_SHA, latent_sha256: Probe::LATENT_SHA, noise_sha256: Probe::NOISE_SHA,
             oracle_report_sha256: Probe::ORACLE_REPORT_SHA, oracle_display_sha256: Probe::ORACLE_SHA,
             decoder_cache_sha256: HyNomasGeneration::PIXEL_CACHE_SHA, geometry: "768x416", frames: 120,
             raw_capture: false, dit_constructed: false, named_budget_bytes: Probe::BUDGET,
             decoder_named_bytes: decoder_named, reused_owner_ids: owner_ids, passes: passes,
             timings: {decoder_setup_seconds: setup_seconds}, outputs: outputs,
             request_seconds: (Time.instant - request_start).total_seconds, atol: 2e-5, rtol: 2e-5,
             resource_scope: "same live decoder owners within each process; identical construction across fresh processes",
             measurement_scope: "one warmup then two measured requests; identical GPU restore, poison and snapshot work; exports after all three terminal waits",
             thermal_scope: "advisory NSProcessInfo thermalState at request boundaries; frequency requires the separate powermetrics observer",
             upstream_scope: "authenticated saved source-MPS PixelUMM oracle; stock CUDA/Flex and full DiT trajectory parity excluded",
             counter_scope: "instrumented MetalBuffer methods only; opaque MPS allocations are guarded externally"}
  HyNomasGeneration.write_fresh_output(File.join(output_dir, "receipt.json")) { |f| f.puts(receipt.to_pretty_json) }
ensure
  begin
    chains.each(&.commit_and_wait)
    decoder.try(&.release)
    chains.each(&.release)
    shared.release
    metadata.close
  rescue ex
    STDERR.puts("recovery cleanup could not establish terminal completion: #{ex.message}")
    Process.exit(1)
  end
end
