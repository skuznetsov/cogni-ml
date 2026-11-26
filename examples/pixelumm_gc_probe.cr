require "option_parser"
require "./support/pixelumm_gc_probe"

alias B = PixelUMMFFNWeightReuseProbe
alias G = PixelUMMGCProbe

reference = reference_sha256 = weight_cache = report = ""
OptionParser.parse do |parser|
  parser.banner = "Bounded PixelUMM GC/allocation probe; no generation."
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
schedule = G.schedule
memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
owner = nil.as(B::Owner?)
observing = false
completed = false

alias TrialResult = NamedTuple(trial: G::Trial, primary: G::Request, tail: G::Request,
  started: G::Point, finished: G::Point)
trials = Array(TrialResult).new(schedule.size)
result = nil.as(JSON::Any?)

begin
  setup_started = Time.instant
  active = B::Owner.new(fixture, cache, B::MAX_NAMED_BYTES, memory,
    expected_request_count: schedule.size * 2, emit_request_memory: false)
  owner = active
  resources = active.named_resources_json
  handles = active.native_plan_handles.dup
  setup_seconds = (Time.instant - setup_started).total_seconds
  G.start_observer!
  observing = true
  loop_started = G::Point.new
  schedule.each do |trial|
    started = G::Point.new
    primary = G.run_request(active, G.entry(trial), trial[:policy])
    # A matched ordinary-GC tail follows every arm, including the last trial.
    # It exposes collection debt without injecting allocations or GC.collect.
    tail = G.run_request(active, G.entry(trial, tail: true), G::NORMAL)
    finished = G::Point.new
    trials << {trial: trial, primary: primary, tail: tail, started: started, finished: finished}
  end
  loop_finished = G::Point.new
  G.stop_observer!
  observing = false
  G.require_boehm!
  active.release_after_terminal!
  completed = true

  result = JSON.parse({
    schema: "cogni_ml.pixelumm_gc_probe/v1",
    scope: "Standalone J5760 two-window/two-block saved initial-noise BF16 baseline, not full generation or a production GC policy qualification.",
    geometry: {grid: [10, 18, 32], tokens: B::TOKENS, hidden: B::HIDDEN, ffn: B::FFN},
    fixture: {path: fixture.path, metadata_sha256: fixture.sha256,
              payload_sha256: fixture.payload_sha256, upstream_revision: B::UPSTREAM_REVISION,
              model_sha256: B::MODEL_SHA256,
              input_x_sha256: B::WINDOW_NAMES.to_h { |name| {name, fixture.windows[name].inputs["x"].sha256} },
              input_source: "saved initial-noise preview conditioning through the authenticated upstream stem; not final generated DiT latents"},
    weight_cache: {path: cache.tensor_file.path, metadata_sha256: cache.tensor_file.sha256,
                   payload_sha256: cache.tensor_file.document["payload_sha256"]},
    topology: {plan_order: [0, 1, 2, 3], schedule: "window-major", all_native_modes: "baseline",
               native_plan_handles: handles, shared_scratch: true, stage_conversions: 0,
               unused_resident_f32_stage_bytes: active.budget[:stage_bytes]},
    comparison: {atol: B::ATOL, rtol: B::RTOL, capture_order: B::CAPTURE_ORDER,
                 correctness: "24 captures per primary correctness request; GPU comparator, poisoned resources and NaN positive control",
                 timing: "four outputs per native request, no intermediate capture copies"},
    observation: {runtime: "Crystal #{Crystal::VERSION} / Boehm", observer_abi_bytes: sizeof(G::Observer::Counters),
                  scope: "GC.disable from before chain creation through the existing sole terminal wait; GC.enable in ensure before metrics/release",
                  total_allocations: "cumulative Boehm allocated-byte deltas for this diagnostic path, not allocation counts, production allocation volume or Objective-C/Metal allocation bytes",
                  event_timing: "collector envelope, rendezvous-inclusive pause, fully-stopped interval are overlapping windows; do not add",
                  cumulative_maxima: "raw Point maxima are cumulative, not subtractable per-phase maxima",
                  phase_attribution: "completed event intervals are charged at their closing boundary",
                  forced_collections_or_artificial_pressure_in_gpu_run: false,
                  request_json_deferred_until_after_all_trials: true,
                  per_request_memory_json_disabled: true,
                  normal_tail_after_every_primary: true,
                  trial_interval: "primary start through the point after its normal tail; outer loop totals also include inter-trial gaps",
                  gc_debt_limit: "one fixed normal tail exposes immediate carryover, not an exhaustive repayment of all future collection debt",
                  cpu_stats_at_existing_phase_boundaries_only: true,
                  observer_overhead: "GC stats and callback ownership queries acquire Boehm CPU locks in both arms; no new GPU synchronization"},
    memory: {named_bytes: active.budget[:total_bytes], resources: resources,
             categories_additive: false,
             guard_required: "16 GiB footprint cap, 10% file-cache reserve, 64 MiB swap growth, 600 s timeout",
             setup_and_release_events: memory.events},
    setup_seconds: setup_seconds,
    observed_loop: G.delta(loop_started, loop_finished),
    initial_point: loop_started, final_point: loop_finished,
    schedule: {correctness_trials: 2, warmup_trials: 4, timed_trials: G::TIMED_TRIALS,
               native_requests: schedule.size * 2, timed_order: "ABBA/BAAB repeated eight times; A=normal B=scoped_disabled",
               trials: trials.map do |item|
                 {index: item[:trial][:index], policy: item[:trial][:policy], kind: item[:trial][:kind],
                  inclusive: G.delta(item[:started], item[:finished]),
                  primary: G.request_json(item[:primary]), tail: G.request_json(item[:tail])}
               end},
    validation: {all_numerical_metrics_passed: true, positive_nan_control_passed: true,
                 one_terminal_wait_and_packed_readback_per_request: true,
                 zero_named_encode_allocations_and_preterminal_readbacks: true,
                 gc_enabled_after_every_scope: true, observer_events_paired: true,
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

raise "incomplete GC probe" unless completed
File.write(report, JSON.build(indent: "  ") { |json| result.not_nil!.to_json(json) })
puts({prefix: "PIXELUMM_GC_PROBE", report: report, trials: trials.size,
      requests: trials.size * 2, named_bytes: B.preflight_budget(B::MAX_NAMED_BYTES)[:total_bytes]}.to_json)
