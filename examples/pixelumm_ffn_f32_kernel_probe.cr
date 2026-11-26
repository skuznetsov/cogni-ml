require "option_parser"
require "digest/sha256"
require "./support/pixelumm_ffn_f32_kernel_probe"

alias B = PixelUMMFFNIsolatedProbe
alias P = PixelUMMFFNF32KernelProbe

reference = ""
reference_sha = ""
report = ""
projection = "gate"
rows = 21120
reverse_order = false
reuse_b = false
maximum_named_bytes = 4_i64 * 1024 * 1024 * 1024

OptionParser.parse do |parser|
  parser.banner = "Isolated exact-F32 PixelUMM FFN GEMM kernel diagnostic; no generation."
  parser.on("--reference PATH", "Authenticated true J2048 or full-J FFN fixture metadata") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted fixture metadata SHA256") { |value| reference_sha = value }
  parser.on("--report PATH", "Fresh absolute JSON report under /Volumes") { |value| report = value }
  parser.on("--projection NAME", "Projection to probe: gate, up, or down") { |value| projection = value }
  parser.on("--rows N", "Exact fixture row count, 2048 or 21120; no repetition") { |value| rows = value.to_i32 }
  parser.on("--reverse-order", "Reverse both warmups and all measured requests for the paired order cell") { reverse_order = true }
  parser.on("--reuse-b", "Select the opt-in exact-F32 kernel that reuses B fragments across row blocks") { reuse_b = true }
  parser.on("--max-named-bytes N", "Pre-read exact named MetalBuffer cap, at most 4 GiB") { |value| maximum_named_bytes = value.to_i64 }
end

unless ARGV.empty? && {"gate", "up", "down"}.includes?(projection) && {2048, 21120}.includes?(rows)
  raise ArgumentError.new("invalid F32 FFN probe arguments")
end
unless Path[reference].absolute? && !report.empty? && Path[report].absolute? &&
       File.directory?(File.dirname(report)) &&
       File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       File.extname(report) == ".json" && !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("fixture must be absolute and report must be fresh under /Volumes")
end

# Reject an over-budget request before fixture metadata or tensor payload is
# read and before the Metal device is initialized.
preflight_named_bytes = P.validate_budget!(projection, rows, maximum_named_bytes)
shape = P.projection(projection)
dispatch_shape = P.geometry(rows, shape[:k], shape[:n])
candidate_kernel_name = reuse_b ? P::REUSE_B_KERNEL : P::BASE_KERNEL
candidate_variant = reuse_b ? "reuse_b" : "baseline"
schedule = P.schedule(reverse_order)
unless schedule.size == P::REQUESTS && schedule.count { |entry| entry[:warmup] } == 2 &&
       schedule.count { |entry| entry[:arm] == "mps.f32" && !entry[:warmup] } == 4 &&
       schedule.count { |entry| entry[:arm] == "native.f32" && !entry[:warmup] } == 4
  raise "F32 FFN order schedule is not balanced"
end

fixture = B::Fixture.new(reference, reference_sha)
unless fixture.metadata.tokens == rows
  raise ArgumentError.new("F32 FFN row count must equal the saved fixture grid; repetition is forbidden")
end

memory = PixelUMMMemoryProbe::Recorder.system(STDERR)
owner = nil.as(P::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
terminal_wait_attempted = false
terminal_wait_succeeded = false
setup_seconds = 0.0
timed_samples = [] of NamedTuple(index: Int32, arm: String, warmup: Bool, timing: B::StageTiming)
comparator_enqueued = [] of Int32
metrics = nil.as(Array(B::Metric)?)
roots = nil.as(Array(ML::Video::MPSCommandRootTiming)?)
allocations_before = 0_i64
readbacks_before = 0_i64
preterminal_allocations = 0_i64
preterminal_readbacks = 0_i64
postterminal_readbacks = 0_i64
terminal_wait_seconds = 0.0
run_errors = [] of String
validation_errors = [] of String

begin
  setup_started = Time.instant
  owner = P::Owner.new(fixture, projection, rows, maximum_named_bytes, memory, candidate_kernel_name)
  setup_seconds = (Time.instant - setup_started).total_seconds
  unless owner.not_nil!.named_bytes == preflight_named_bytes
    raise "owner named-buffer extent differs from pre-read budget"
  end
  chain = ML::Metal::MPSCommandChain.new
  owner.not_nil!.preencode!

  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
  schedule.each do |request|
    owner.not_nil!.poison!(chain.not_nil!)
    timing = B.timed_stage(chain.not_nil!, request[:arm]) do
      owner.not_nil!.encode!(chain.not_nil!, request[:arm])
    end
    timed_samples << {index: request[:index], arm: request[:arm],
                      warmup: request[:warmup], timing: timing}
    owner.not_nil!.compare!(chain.not_nil!, request[:index])
    comparator_enqueued << request[:index]
    memory.emit("postrequest", slot: request[:index], request: request[:index],
      phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: owner.not_nil!.named_bytes)
  end
  owner.not_nil!.positive_control!(chain.not_nil!)

  preterminal_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  if preterminal_allocations != 0
    validation_errors << "named MetalBuffer allocations occurred after encoding began"
  end
  if preterminal_readbacks != 0
    validation_errors << "GPU readback occurred before the sole terminal wait"
  end
  if chain.not_nil!.wait_count != 0
    validation_errors << "command chain waited before the terminal boundary"
  end

  wait_started = Time.instant
  terminal_wait_attempted = true
  chain.not_nil!.commit_and_wait
  terminal_wait_seconds = (Time.instant - wait_started).total_seconds
  terminal_wait_succeeded = chain.not_nil!.wait_count == 1 && chain.not_nil!.completed?
  raise "F32 FFN probe did not complete its sole terminal wait" unless terminal_wait_succeeded
  owner.not_nil!.postwait!(chain.not_nil!)
  metrics = owner.not_nil!.metrics(chain.not_nil!)
  postterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  roots = chain.not_nil!.completed_root_timings
rescue error
  run_errors << "#{error.class}: #{error.message}"
  if active_chain = chain
    unless terminal_wait_attempted
      begin
        terminal_wait_attempted = true
        wait_started = Time.instant
        active_chain.commit_and_wait
        terminal_wait_seconds = (Time.instant - wait_started).total_seconds
        terminal_wait_succeeded = active_chain.wait_count == 1 && active_chain.completed?
        if terminal_wait_succeeded && (active_owner = owner)
          active_owner.postwait!(active_chain) unless active_owner.lifecycle.terminal_wait_succeeded?
          metrics ||= active_owner.metrics(active_chain)
          postterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
        end
        roots ||= active_chain.completed_root_timings if terminal_wait_succeeded
      rescue drain_error
        run_errors << "terminal drain: #{drain_error.class}: #{drain_error.message}"
        terminal_wait_succeeded = false
      end
    end
  end
ensure
  if active_owner = owner
    active_chain = chain
    if terminal_wait_succeeded && active_chain
      begin
        active_owner.postwait!(active_chain) unless active_owner.lifecycle.terminal_wait_succeeded?
        active_owner.release_after_terminal!(active_chain)
      rescue release_error
        run_errors << "terminal resource release: #{release_error.class}: #{release_error.message}"
        P::FailureQuarantine.retain(active_owner, active_chain) unless active_owner.lifecycle.release_allowed?(active_chain.completed?, active_chain.wait_count)
      end
    elsif !active_owner.lifecycle.attempted?
      begin
        active_owner.release_unsubmitted!
      rescue release_error
        run_errors << "unsubmitted resource release: #{release_error.class}: #{release_error.message}"
      end
    elsif active_chain
      P::FailureQuarantine.retain(active_owner, active_chain)
    end
  end
  if active_chain = chain
    if terminal_wait_succeeded && active_chain.drained?
      begin
        active_chain.release
        memory.emit("after_chain_release", phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: 0_i64)
      rescue release_error
        run_errors << "command-chain release: #{release_error.class}: #{release_error.message}"
        P::FailureQuarantine.retain_chain(active_chain)
      end
    elsif !terminal_wait_attempted
      # A chain with no submission still needs an explicit terminal state
      # before release; this is the sole wait, not a per-arm synchronization.
      begin
        terminal_wait_attempted = true
        active_chain.commit_and_wait
        terminal_wait_succeeded = active_chain.wait_count == 1 && active_chain.completed?
        active_chain.release if terminal_wait_succeeded
      rescue release_error
        run_errors << "empty command-chain drain: #{release_error.class}: #{release_error.message}"
        P::FailureQuarantine.retain_chain(active_chain)
      end
    else
      P::FailureQuarantine.retain_chain(active_chain)
    end
  end
end

expected_elements = dispatch_shape[:output_elements].to_u32
if completed = metrics
  if completed.size != P::METRIC_SLOTS
    validation_errors << "GPU comparator returned the wrong metric slot count"
  else
    comparator_enqueued.each do |slot|
      metric = completed[slot]
      if metric[:elements] != expected_elements
        validation_errors << "GPU comparator extent mismatch in request slot #{slot}"
      end
      if metric[:failures] != 0 || metric[:nonfinite] != 0
        arm = schedule[slot][:arm]
        validation_errors << "#{arm} output failed GPU BF16 comparison in request slot #{slot}"
      end
    end
    if comparator_enqueued.size != P::REQUESTS
      validation_errors << "not all ten request comparisons were enqueued"
    end
    control = completed[P::REQUESTS]
    unless control[:failures] == 1 && control[:nonfinite] == 1 && control[:elements] == expected_elements
      validation_errors << "forced-NaN GPU comparator control did not report exactly one nonfinite failure"
    end
  end
else
  validation_errors << "GPU comparator metrics were unavailable after terminal processing"
end

root_by_slot = {} of Int32 => JSON::Any
if completed_roots = roots
  timed_samples.each do |sample|
    begin
      gpu = P.checked_root_timing(sample[:timing], completed_roots)
      root_by_slot[sample[:index]] = JSON.parse({
        gpu_root_union_seconds:       gpu[:gpu_root_union_seconds],
        outside_root_overlap_seconds: gpu[:outside_root_overlap_seconds],
        isolated_root_timing:         gpu[:isolated_root_timing],
      }.to_json)
      # Warmups retain their raw overlap evidence but do not contribute to
      # latency estimates or the predeclared measured-root isolation gate.
      if !sample[:warmup] && !gpu[:isolated_root_timing]
        validation_errors << "non-isolated root timing in measured request slot #{sample[:index]}"
      end
    rescue timing_error
      root_by_slot[sample[:index]] = JSON.parse({timing_error: "#{timing_error.class}: #{timing_error.message}"}.to_json)
      validation_errors << "root timing failed in request slot #{sample[:index]}"
    end
  end
else
  validation_errors << "completed root timing summaries were unavailable"
end

sample_reports = schedule.map do |request|
  timed = timed_samples.find { |sample| sample[:index] == request[:index] }
  metric = metrics.try { |values| values[request[:index]]? }
  timing_json = if timed
                  timing = timed[:timing]
                  JSON.parse({stage: timing[:stage], first_root: timing[:first_root], end_root: timing[:end_root],
                              encode_wall_seconds: timing[:encode_wall_seconds], gpu: root_by_slot[request[:index]]?}.to_json)
                end
  JSON.parse({index: request[:index], arm: request[:arm], warmup: request[:warmup],
              comparator_enqueued: comparator_enqueued.includes?(request[:index]),
              timing: timing_json, metric: metric ? P.metric_json(metric.not_nil!) : nil}.to_json)
end

arm_summaries = ["mps.f32", "native.f32"].map do |arm|
  selected = timed_samples.select { |sample| sample[:arm] == arm && !sample[:warmup] }
  values = selected.compact_map do |sample|
    root = root_by_slot[sample[:index]]?
    root.try { |entry| entry["gpu_root_union_seconds"].as_f? }
  end.compact
  {arm: arm, measured_requests: selected.size,
   median_gpu_root_union_seconds: values.size == 4 ? P.median(values) : nil,
   min_gpu_root_union_seconds: values.min?, max_gpu_root_union_seconds: values.max?,
   all_roots_isolated: selected.size == 4 && selected.all? do |sample|
     root_by_slot[sample[:index]]?.try { |entry| entry["isolated_root_timing"].as_bool } == true
   end}
end
mps_median = arm_summaries[0][:median_gpu_root_union_seconds]
native_median = arm_summaries[1][:median_gpu_root_union_seconds]
single_cell_improvement = if mps_median && native_median && mps_median.not_nil! > 0.0
                            1.0 - native_median.not_nil! / mps_median.not_nil!
                          end

if terminal_wait_succeeded && chain && chain.not_nil!.wait_count != 1
  validation_errors << "terminal wait count was not exactly one"
end
if postterminal_readbacks != 1 && metrics
  validation_errors << "expected exactly one post-wait scalar-metric readback"
end

status = run_errors.empty? && validation_errors.empty? ? "passed" : "failed"
both_arms_isolated = arm_summaries.all? { |arm| arm[:all_roots_isolated] }
timing_eligible = status == "passed" && both_arms_isolated
result = JSON.parse({
  schema:  P::SCHEMA,
  request: {projection: projection, rows: rows, input_width: shape[:k], output_width: shape[:n],
            order_cell: reverse_order ? "reverse" : "forward",
            fixture_tokens: fixture.metadata.tokens, fixture_metadata_sha256: fixture.sha256,
            fixture_payload_sha256: fixture.metadata.payload_sha256,
            input_record_sha256: owner.try(&.input_record.sha256),
            weight_record_sha256: owner.try(&.weight_record.sha256),
            expected_record_sha256: owner.try(&.expected_record.sha256)},
  candidate: {kernel: candidate_kernel_name, variant: candidate_variant,
              source_sha256: Digest::SHA256.hexdigest(P::NATIVE_SOURCE),
              tile_m: P::TM, tile_n: P::TN, tile_k: P::TK,
              threadgroup: dispatch_shape[:threads], grid: dispatch_shape[:groups],
              native_execution_width: owner.try(&.native_execution_width),
              native_max_threads_per_threadgroup: owner.try(&.native_max_threads),
              native_static_threadgroup_bytes: owner.try(&.native_static_threadgroup_bytes),
              operand_arithmetic: "BF16 bit expansion to F32 in threadgroup; F32 simdgroup operands/accumulators; ascending K32 panels and K8 MMA calls; final BF16 round-to-nearest-even"},
  baseline: {mode: "mps.f32", projection_bridge_mode: 1,
             operation_identity: "separate diagnostic builder mirroring production F32 FFN projection operations",
             matmul_output_data_type_raw: owner.try(&.product_type),
             comparator_source_sha256: Digest::SHA256.hexdigest(B::SOURCE)},
  schedule:           schedule.map { |entry| {index: entry[:index], arm: entry[:arm], warmup: entry[:warmup]} },
  samples:            sample_reports,
  summaries:          arm_summaries,
  timing_eligibility: {eligible: timing_eligible, status: status,
                       both_arms_isolated: both_arms_isolated,
                       scope: "single-cell medians are eligible only when the complete probe passes and both timed arms have isolated roots"},
  single_cell_native_improvement_fraction: single_cell_improvement,
  timing_scope:                            "GPU root-union intervals attributed by per-arm root markers; includes encoded root work and is not exclusive shader time",
  execution:                               {named_bytes: preflight_named_bytes, resources: P.resources(projection, rows),
              maximum_named_bytes: maximum_named_bytes, setup_seconds: setup_seconds,
              named_buffer_allocations_after_encoding_started: preterminal_allocations,
              readbacks_before_terminal_wait: preterminal_readbacks,
              readbacks_after_terminal_wait: postterminal_readbacks,
              terminal_wait_attempted: terminal_wait_attempted,
              terminal_wait_succeeded: terminal_wait_succeeded,
              terminal_wait_count: chain.try(&.wait_count),
              terminal_wait_seconds: terminal_wait_seconds,
              root_summary_count: roots.try(&.size),
              root_timings: roots.try { |items| items.map do |root|
                {sequence: root.sequence, completed: root.completed, has_error: root.has_error,
                 gpu_start_seconds: root.gpu_start_seconds, gpu_end_seconds: root.gpu_end_seconds}
              end },
              memory_events: memory.events},
  gpu_comparison: {tolerance_atol: B::ATOL, tolerance_rtol: B::RTOL,
                   metric_slots: metrics.try(&.map { |metric| P.metric_json(metric) }),
                   positive_control: metrics.try { |values| P.metric_json(values[P::REQUESTS]) }},
  errors:  {execution: run_errors, validation: validation_errors},
  summary: {status: status, production_promoted: false,
            interpretation: "One order cell supplies diagnostic evidence only; compare both forward and reverse cells before connected-followup eligibility."},
}.to_json)

begin
  P.write_report_new!(report, result.to_pretty_json)
rescue write_error
  run_errors << "report write: #{write_error.class}: #{write_error.message}"
  raise write_error
end

puts result["summary"].to_json
unless run_errors.empty? && validation_errors.empty?
  raise "F32 FFN kernel probe failed; detailed arm metrics and errors were written to #{report}"
end
