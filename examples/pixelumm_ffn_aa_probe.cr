require "option_parser"
require "./support/pixelumm_ffn_aa_probe"

alias A = PixelUMMFFNAaProbe
alias C = PixelUMMFFNAaContract
alias I = PixelUMMFFNIsolatedProbe
alias B = PixelUMMFFNWeightReuseProbe

reference = ""
reference_sha = ""
weight_cache = ""
report = ""
OptionParser.parse do |parser|
  parser.banner = "Bounded native J5760 connected FFN baseline A/A; no generation"
  parser.on("--reference PATH", "Pinned J5760 windows fixture") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted fixture manifest digest") { |value| reference_sha = value }
  parser.on("--weight-cache PATH", "Pinned BF16 cache manifest") { |value| weight_cache = value }
  parser.on("--report PATH", "Fresh absolute JSON under the external volume") { |value| report = value }
  parser.invalid_option { |value| raise ArgumentError.new("unknown FFN A/A option: #{value}") }
end
raise ArgumentError.new("unexpected FFN A/A positional argument") unless ARGV.empty?
unless reference.starts_with?("/") && weight_cache.starts_with?("/") && reference_sha == A::REFERENCE_SHA256
  raise ArgumentError.new("FFN A/A requires the pinned J5760 fixture and absolute input paths")
end
unless report.starts_with?("/") && report.ends_with?(".json") && File.directory?(File.dirname(report)) &&
       !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("FFN A/A report must be a fresh absolute JSON path")
end
report = File.join(File.realpath(File.dirname(report)), File.basename(report))
raise ArgumentError.new("FFN A/A report must use external scratch") unless report.starts_with?("/Volumes/")

source_paths = [
  __FILE__, "examples/support/pixelumm_ffn_aa_probe.cr",
  "examples/support/pixelumm_ffn_aa_contract.cr", "examples/support/pixelumm_ffn_aa_bridge.mm",
  "examples/support/pixelumm_ffn_dense.metal", "examples/support/pixelumm_ffn_isolated_probe.cr",
  "examples/support/pixelumm_ffn_weight_reuse_probe.cr", "examples/support/wan_memory_guard.py",
  "src/ml/video/pixelumm_decoder_block_mps_bridge.mm",
  "src/ml/video/metal_mps_command_chain.cr", "src/ml/video/metal_mps_command_chain.mm",
]
source_hashes = source_paths.to_h { |path| {path, I.sha_file(path)} }
reference = File.realpath(reference)
weight_cache = File.realpath(weight_cache)
fixture = B::Fixture.new(reference, reference_sha)
fixture.verify_payload!
cache = B::WeightCache.new(weight_cache, fixture)
schedule = C.schedule
timings = Array(I::StageTiming?).new(schedule.size, nil)
memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
memory.emit("before_ffn_aa_owner")
owner = nil.as(A::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
wait_attempted = false
begin
  active = A::Owner.new(fixture, cache)
  owner = active
  memory.emit("ffn_aa_preencode", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: active.named_bytes)
  command = ML::Metal::MPSCommandChain.new
  chain = command
  command.enable_api_profile!(phase_roots: false)
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
  encode_started = Time.instant

  # No fixture reads, hashing, telemetry, GC diagnostics, or readbacks here.
  # GPU comparison cadence is fixed and explicitly outside each timed root set.
  schedule.each do |sample|
    timings[sample.index] = I.timed_stage(command, "connected_ffn") do
      active.encode!(command, sample.capture_raw)
    end
    active.copy_canonical!(command) if sample.index == 0
    active.compare_sample!(command, sample.index)
  end
  active.controls!(command)

  encode_seconds = (Time.instant - encode_started).total_seconds
  allocation_delta = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  raise "FFN A/A encoding allocated named buffers or read back tensors" unless allocation_delta == 0 && encode_readbacks == 0
  raise "FFN A/A introduced an intermediate wait" unless command.wait_count == 0
  wait_attempted = true
  terminal_started = Time.instant
  command.commit_and_wait
  terminal_seconds = (Time.instant - terminal_started).total_seconds
  active.postwait!(command)
  roots = command.completed_root_timings
  metrics = active.metrics(command)
  total_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  counts = active.counts
  raise "FFN A/A changed operation count" unless counts == [18_u64, 18_u64, 18_u64, 18_u64, 18_u64, 18_u64, 2_u64]
  raise "FFN A/A requires one packed post-terminal readback" unless total_readbacks == 1

  samples = schedule.map do |sample|
    timing = timings[sample.index].not_nil!
    gpu = I.root_timing(timing, roots)
    {index: sample.index, warmup: sample.warmup, capture_raw: sample.capture_raw,
     mode: "baseline", first_root: timing[:first_root], end_root: timing[:end_root],
     encode_wall_seconds: timing[:encode_wall_seconds],
     gpu_root_union_seconds: gpu[:gpu_root_union_seconds],
     outside_root_overlap_seconds: gpu[:outside_root_overlap_seconds],
     isolated_root_timing: gpu[:isolated_root_timing]}
  end
  timed = samples.reject(&.[:warmup])
  timing_pass = samples.all?(&.[:isolated_root_timing])
  numerical_failure = nil.as(String?)
  begin
    C.validate_metrics!(metrics)
  rescue error
    numerical_failure = error.message || error.class.name
  end
  summary = C.summarize(timed.map(&.[:gpu_root_union_seconds]))
  api = command.api_profile
  result = {
    schema: "cogni_ml.pixelumm_ffn_aa_probe/v1",
    scope: "native connected FFN A/A under fixed GPU-comparator cadence; no full generation",
    geometry: {tokens: 5760, hidden: 4096, ffn: 12288, grid: [10, 18, 32]},
    fixture: {path: File.realpath(reference), metadata_sha256: fixture.sha256,
              payload_sha256: fixture.payload_sha256,
              weight_cache_metadata_sha256: cache.tensor_file.sha256,
              weight_cache_payload_sha256: cache.tensor_file.document["payload_sha256"],
              input_window: "w0", block: 0, normalized_input: fixture.windows["w0"].captures["block0"]["post_attn_norm"].sha256,
              input_provenance: "saved_initial_noise_conditioning_upstream_stem",
              model_inference: false, torch_version: fixture.document["torch_version"],
              upstream_revision: B::UPSTREAM_REVISION, upstream_model_sha256: B::MODEL_SHA256},
    comparison: {atol: B::ATOL, rtol: B::RTOL},
    protocol: {warmups: 2, timed_samples: 16, identical_plan_and_buffers: true,
               timed_path: "separate gate/up F32-compute MPSGraph -> BF16 SiLU -> BF16 product -> F32-compute down MPSGraph",
               correctness_capture_copies: "raw gate/up in sample 0 warmup only",
               normal_checks_per_sample: 3, raw_gate_up_checks_after_loop: 2, positive_nan_controls: 1,
               cpu_diagnostics_between_samples: false, cpu_readback_between_samples: false,
               gpu_checks_between_samples: true, no_candidate_or_promotion: true,
               memory_accounting: "external cache-aware guard; file-backed pages counted once; reserve 10 percent"},
    execution: {named_bytes: active.named_bytes, resources: active.resources,
                encoding_named_allocation_delta: allocation_delta, encoding_readbacks: encode_readbacks,
                total_readbacks: total_readbacks, wait_count: command.wait_count,
                encode_loop_wall_seconds: encode_seconds, terminal_wait_wall_seconds: terminal_seconds,
                total_root_count: roots.size, operation_counts: counts},
    samples: samples, summary: summary,
    numerical_metrics: metrics, numerical_failure: numerical_failure,
    root_timings: roots.map { |root| {sequence: root.sequence, gpu_start_seconds: root.gpu_start_seconds,
                                      gpu_end_seconds: root.gpu_end_seconds, completed: root.completed, has_error: root.has_error} },
    api_profile: {phase_count: api.phases.size}, source_sha256: source_hashes,
    numerical_pass: numerical_failure.nil?, timing_isolation_pass: timing_pass,
    pass: numerical_failure.nil? && timing_pass,
  }
  fd = LibC.open(report, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
  raise "FFN A/A report reservation failed" if fd < 0
  io = IO::FileDescriptor.new(fd)
  begin
    io << result.to_json << '\n'
    io.flush
  ensure
    io.close
  end
  puts({report: report, pass: result[:pass], summary: summary,
        numerical_failure: numerical_failure, timing_isolation_pass: timing_pass}.to_json)
  memory.emit("ffn_aa_postwait", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: active.named_bytes)
  raise "FFN A/A numerical/isolation gate failed; raw report retained" unless result[:pass]
ensure
  if active_chain = chain
    if !active_chain.completed? && !wait_attempted
      wait_attempted = true
      active_chain.commit_and_wait
    end
    if active_chain.completed?
      if active_owner = owner
        active_owner.postwait!(active_chain) unless active_owner.lifecycle.terminal_wait_succeeded?
        active_owner.release_after_terminal!(active_chain)
      end
      active_chain.release
    end
  elsif active_owner = owner
    active_owner.release_unsubmitted!
  end
end
