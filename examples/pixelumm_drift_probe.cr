require "option_parser"
require "./support/pixelumm_drift_probe"

alias B = PixelUMMFFNWeightReuseProbe
alias G = PixelUMMGCProbe
alias D = PixelUMMDriftProbe

reference = reference_sha256 = weight_cache = report = ""
OptionParser.parse do |parser|
  parser.banner = "Bounded PixelUMM phase/API drift probe; no generation."
  parser.on("--reference PATH", "Pinned upstream two-window fixture") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA256") { |value| reference_sha256 = value }
  parser.on("--weight-cache PATH", "Pinned BF16 cache manifest") { |value| weight_cache = value }
  parser.on("--report PATH", "Fresh absolute JSON report under /Volumes") { |value| report = value }
end
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?

def canonical_file?(path : String) : Bool
  !path.empty? && Path[path].absolute? && File.file?(path) &&
    !File.symlink?(path) && File.realpath(path) == path
end

unless canonical_file?(reference) && canonical_file?(weight_cache) &&
       PixelUMMBlockProbe.valid_digest?(reference_sha256) && Path[report].absolute? &&
       File.directory?(File.dirname(report)) && File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       File.extname(report) == ".json" && !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("reference/cache must be canonical absolute files and report must be fresh under /Volumes")
end
G.require_boehm!
B.preflight_budget(B::MAX_NAMED_BYTES)
fixture = B::Fixture.new(reference, reference_sha256)
cache = B::WeightCache.new(weight_cache, fixture)
schedule = D.schedule
memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
owner = nil.as(B::Owner?)
observing = completed = false
requests = Array(D::Request).new(schedule.size)
markers = Array(Slice(PixelUMMDecoderBlockMPSFFI::PhaseMarker)).new(4) do
  Slice(PixelUMMDecoderBlockMPSFFI::PhaseMarker).new(11) { PixelUMMDecoderBlockMPSFFI::PhaseMarker.new }
end
result = nil.as(JSON::Any?)

begin
  setup_started = Time.instant
  active = B::Owner.new(fixture, cache, B::MAX_NAMED_BYTES, memory,
    expected_request_count: schedule.size, emit_request_memory: false)
  owner = active
  resources = active.named_resources_json
  handles = active.native_plan_handles.dup
  setup_seconds = (Time.instant - setup_started).total_seconds
  G.start_observer!
  observing = true
  loop_started = G::Point.new
  schedule.each do |trial|
    measured = D.run_request(active, trial, markers)
    requests << measured
    if trial[:index] % 8 == 7 || trial[:index] == 1 || trial[:index] == schedule.size - 1
      STDERR.puts({prefix: "PIXELUMM_DRIFT_PROGRESS", index: trial[:index], profile: trial[:profile],
                   gpu_seconds: measured[:gpu_elapsed_seconds], body_encode_seconds: measured[:body_encode_seconds],
                   thermal_state: measured[:host_points][4].thermal_state}.to_json)
    end
  end
  loop_finished = G::Point.new
  G.stop_observer!
  observing = false
  active.release_after_terminal!
  completed = true
  result = JSON.parse({
    schema: "cogni_ml.pixelumm_drift_probe/v1",
    scope: "Standalone J5760, two windows and blocks 0/1, ordinary GC, baseline BF16; no head or full generation.",
    geometry: {grid: [10, 18, 32], tokens: B::TOKENS, hidden: B::HIDDEN, ffn: B::FFN},
    fixture: {path: fixture.path, metadata_sha256: fixture.sha256,
              payload_sha256: fixture.payload_sha256, upstream_revision: B::UPSTREAM_REVISION,
              model_sha256: B::MODEL_SHA256,
              input_x_sha256: B::WINDOW_NAMES.to_h { |name| {name, fixture.windows[name].inputs["x"].sha256} },
              input_source: "authenticated saved initial-noise conditioning, not generated DiT latents"},
    weight_cache: {path: cache.tensor_file.path, metadata_sha256: cache.tensor_file.sha256,
                   payload_sha256: cache.tensor_file.document["payload_sha256"]},
    topology: {plan_order: [0, 1, 2, 3], schedule: "window-major", native_plan_handles: handles,
               all_native_modes: "baseline", shared_scratch: true, stage_conversions: 0},
    comparison: {atol: B::ATOL, rtol: B::RTOL, capture_order: B::CAPTURE_ORDER,
                 correctness: "24 captures per correctness request, GPU comparison, poison and NaN positive control",
                 timing: "four GPU output checks per request, no intermediate capture copies"},
    observation: {phase_roots: false, instrumentation: "existing native phase/API/continuation collectors",
                  profile_control: "ABBA/BAAB, A=off B=on; both ordinary GC; CPU profile extraction follows terminal wait",
                  attribution: "shared root candidates and overlapping root intervals are not additive phase GPU work",
                  api_nesting: "graph wall includes nested continuation calls; use exclusive graph time or keep separate",
                  clocks: "phase markers CLOCK_MONOTONIC relative per plan; continuation mach ticks, correlate by root sequence",
                  terminal_wait: "existing commit_and_wait envelope includes encoder closure; not pure blocked time",
                  thermal_state: "NSProcessInfo advisory enum 0 nominal/1 fair/2 serious/3 critical, not measured GPU frequency",
                  observer: "Boehm stats at existing host boundaries, no forced collections",
                  reports_deferred: true, full_generation_executed: false},
    memory: {named_bytes: active.budget[:total_bytes], resources: resources,
             unused_resident_f32_stage_bytes: active.budget[:stage_bytes],
             guard_required: "16 GiB footprint cap, 10% file-cache reserve, 64 MiB swap growth, 600 s timeout",
             setup_and_release_events: memory.events},
    setup_seconds: setup_seconds, observed_loop: G.delta(loop_started, loop_finished),
    schedule: {correctness_requests: 2, warmup_requests: 4, timed_requests: D::TIMED_REQUESTS,
               native_requests: schedule.size, requests: requests.map { |request| D.request_json(request) }},
    validation: {all_numerical_metrics_passed: true, positive_nan_control_passed: true,
                 one_terminal_wait_and_packed_readback_per_request: true,
                 zero_named_encode_allocations_and_preterminal_readbacks: true,
                 full_generation_executed: false, production_promoted: false},
  }.to_json)
ensure
  begin
    G.stop_observer! if observing
  ensure
    if owner && owner.not_nil!.lifecycle.release_allowed?
      owner.not_nil!.release_unsubmitted!
    end
  end
end
raise "incomplete drift probe" unless completed
File.write(report, JSON.build(indent: "  ") { |json| result.not_nil!.to_json(json) })
puts({prefix: "PIXELUMM_DRIFT_PROBE", report: report, requests: requests.size}.to_json)
