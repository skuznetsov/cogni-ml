require "option_parser"
require "process"
require "./support/pixelumm_ffn_stage_timing_contract"

alias T = PixelUMMFFNStageTimingContract
alias G = PixelUMMFFNGraphProbe
alias A = PixelUMMFFNAaProbe
alias C = PixelUMMFFNAaContract
alias B = PixelUMMFFNWeightReuseProbe
alias I = PixelUMMFFNIsolatedProbe

reference = ""
reference_sha = ""
weight_cache = ""
parity_report = ""
parity_report_sha = ""
guard_log = ""
report = ""
OptionParser.parse do |parser|
  parser.banner = "Diagnostic-only J5760 FFN stage timing; require a passed stage_outputs parity report and the sampled 8 GiB/10% external memory guard; no generation or promotion"
  parser.on("--reference PATH", "Pinned J5760 windows fixture") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted fixture manifest digest") { |value| reference_sha = value }
  parser.on("--weight-cache PATH", "Pinned BF16 block-weight cache") { |value| weight_cache = value }
  parser.on("--parity-report PATH", "Successful stage_outputs graph correctness report") { |value| parity_report = value }
  parser.on("--parity-report-sha256 SHA", "Trusted parity report file digest") { |value| parity_report_sha = value }
  parser.on("--guard-log PATH", "Live log for the enclosing wan_memory_guard.py process") { |value| guard_log = value }
  parser.on("--report PATH", "Fresh absolute JSON report path under /Volumes") { |value| report = value }
  parser.invalid_option { |value| raise ArgumentError.new("unknown FFN stage timing option: #{value}") }
end

raise ArgumentError.new("unexpected FFN stage timing positional argument") unless ARGV.empty?
unless reference.starts_with?("/") && weight_cache.starts_with?("/") &&
       parity_report.starts_with?("/") && guard_log.starts_with?("/") &&
       reference_sha == A::REFERENCE_SHA256 && parity_report_sha.match(/\A[0-9a-f]{64}\z/)
  raise ArgumentError.new("FFN timing requires pinned absolute inputs, parity report hash, and external guard log")
end

def canonical_file!(path : String, label : String) : String
  unless path.starts_with?("/") && File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
    raise ArgumentError.new("#{label} must be a canonical absolute file")
  end
  path
end

reference = canonical_file!(reference, "reference fixture")
weight_cache = canonical_file!(weight_cache, "weight cache")
parity_report = canonical_file!(parity_report, "stage-output parity report")
guard_log = canonical_file!(guard_log, "external memory guard log")
unless report.starts_with?("/") && report.ends_with?(".json") &&
       File.directory?(File.dirname(report)) &&
       File.realpath(File.dirname(report)).starts_with?("/Volumes/") &&
       !File.exists?(report) && !File.symlink?(report)
  raise ArgumentError.new("FFN timing report must be a fresh absolute JSON path under /Volumes")
end
report = File.join(File.realpath(File.dirname(report)), File.basename(report))

def free_memory_percent! : Int32
  output = IO::Memory.new
  error = IO::Memory.new
  status = Process.run("memory_pressure", ["-Q"], output: output, error: error)
  raise ArgumentError.new("10% free-memory preflight could not read memory_pressure -Q") unless status.success?
  match = output.to_s.match(/System-wide memory free percentage:\s*(\d+)%/)
  value = match.try(&.[1]).try(&.to_i?) ||
          raise ArgumentError.new("10% free-memory preflight could not parse memory_pressure output")
  raise ArgumentError.new("10% free-memory preflight requires more than 10% free") unless value > 10
  value
end

def guard_records!(path : String) : Array(JSON::Any)
  records = File.read(path).lines.reject(&.empty?).map do |line|
    JSON.parse(line)
  rescue
    raise ArgumentError.new("external memory guard log contains invalid JSONL")
  end
  T.validate_guard_records!(records, Process.pid.to_i64)
  records
end

def clock_anchor! : JSON::Any
  before = PixelUMMMemoryProbe.guard_monotonic_ns ||
           raise ArgumentError.new("Mach absolute time is unavailable for observer alignment")
  unix = Time.utc.to_unix_ns.to_i64
  after = PixelUMMMemoryProbe.guard_monotonic_ns ||
          raise ArgumentError.new("Mach absolute time is unavailable for observer alignment")
  T.validate_clock_anchor!(before, unix, after)
  JSON.parse({implementation: "mach_absolute_time()",
              monotonic_before_ns: before, unix_ns: unix,
              monotonic_after_ns: after}.to_json)
end

def current_source_hashes! : Hash(String, String)
  paths = T::SOURCE_PATHS.map do |path|
    raise ArgumentError.new("parity source is missing or a symlink: #{path}") unless File.file?(path) && !File.symlink?(path)
    File.realpath(path)
  end
  raise ArgumentError.new("parity source inventory resolves to duplicates") unless paths.uniq.size == paths.size
  paths.to_h { |path| {path, G.sha_file(path)} }
end

def reported_source_hashes!(document : JSON::Any) : Hash(String, String)
  raw = document["source_sha256"].as_h
  raise ArgumentError.new("parity report source hash inventory size differs") unless raw.size == T::SOURCE_PATHS.size
  raw.each_with_object({} of String => String) do |(path, digest), normalized|
    resolved = File.realpath(path)
    raise ArgumentError.new("parity report source paths resolve to duplicates") if normalized.has_key?(resolved)
    normalized[resolved] = digest.as_s
  end
rescue error : ArgumentError
  raise error
rescue error
  raise ArgumentError.new("parity report source inventory is malformed: #{error.message || error.class.name}")
end

def fresh_report!(path : String, document) : Nil
  fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
  raise "FFN stage timing report reservation failed" if fd < 0
  io = IO::FileDescriptor.new(fd)
  begin
    io << document.to_json << '\n'
    io.flush
  ensure
    io.close
  end
end

guard_records = guard_records!(guard_log)
guard_limit = T.validate_guard_records!(guard_records, Process.pid.to_i64)
guard_started = guard_records.find { |record| record["event"].as_s == "started" } ||
                raise ArgumentError.new("external memory guard start record is missing")

parity_digest = G.sha_file(parity_report)
unless parity_digest == parity_report_sha
  raise ArgumentError.new("stage-output parity report SHA-256 differs from the trusted pin")
end
parity_document = JSON.parse(File.read(parity_report))
source_hashes = current_source_hashes!
reported_hashes = reported_source_hashes!(parity_document)
T.validate_source_hashes!(reported_hashes, source_hashes)

fixture = B::Fixture.new(reference, reference_sha)
fixture.verify_payload!
cache = B::WeightCache.new(weight_cache, fixture)
normalized_input_sha = fixture.windows["w0"].captures["block0"]["post_attn_norm"].sha256
T.validate_stage_outputs_parity_report!(parity_document, source_hashes,
  File.realpath(reference), fixture.sha256, fixture.payload_sha256,
  cache.tensor_file.sha256, cache.tensor_file.document["payload_sha256"].as_s,
  normalized_input_sha)

free_percent_before = free_memory_percent!
baseline = nil.as(A::Owner?)
candidate = nil.as(G::Owner?)
chain = nil.as(ML::Metal::MPSCommandChain?)
wait_attempted = false
report_written = false
failure = nil.as(String?)
metrics = [] of G::Metric
metric_failures = [] of String
baseline_counts = [] of UInt64
candidate_counts = [] of UInt64
candidate_operation_counts = {} of String => UInt64
stage_timings = Array(I::StageTiming).new(T::SCHEDULE.size)
sample_timings = [] of T::SampleTiming
raw_roots = [] of ML::Video::MPSCommandRootTiming
profile_json = [] of JSON::Any
allocations = -1_i64
preterminal_readbacks = -1_i64
postterminal_readbacks = -1_i64
wait_count = 0_i32
free_percent_after = -1
named_bytes = 0_i64
resources = {} of String => Int64
before_anchor = JSON.parse("null")
after_anchor = JSON.parse("null")
api_profile_pass = false

begin
  active_baseline = A::Owner.new(fixture, cache)
  baseline = active_baseline
  active_candidate = G::Owner.new(active_baseline, G::Topology::StageOutputs)
  candidate = active_candidate
  named_bytes = active_candidate.named_bytes
  resources = active_candidate.resources
  raise ArgumentError.new("FFN timing named-buffer cap exceeded") unless named_bytes > 0 && named_bytes <= T::NAMED_BUFFER_CAP_BYTES
  free_percent_after = free_memory_percent!

  command = ML::Metal::MPSCommandChain.new
  chain = command
  command.enable_api_profile!(phase_roots: false)
  command.set_api_profile_phase!(5)
  allocation_start = ML::MetalBuffer.hy_text_probe_allocations
  readback_start = ML::MetalBuffer.hy_text_probe_readbacks

  before_anchor = clock_anchor!
  active_baseline.encode!(command, true)
  active_baseline.copy_canonical!(command)

  T.validate_schedule!(T::SCHEDULE)
  T::SCHEDULE.each_with_index do |entry, index|
    timing = I.timed_stage(command, entry[:label]) do
      if entry[:mode] == "A"
        T::REPEATS.times { active_baseline.encode!(command, false) }
      else
        T::REPEATS.times { active_candidate.encode_output_only!(command) }
      end
    end
    stage_timings << timing

    product_source = entry[:mode] == "A" ? "work.up" : "graph.product"
    active_candidate.compare!(command, product_source, "expected.product",
      (index * 2).to_i32, C::PRODUCT_ELEMENTS)
    active_candidate.compare!(command, "work.down", "expected.down",
      (index * 2 + 1).to_i32, C::DOWN_ELEMENTS)
  end

  active_baseline.controls!(command)
  allocations = ML::MetalBuffer.hy_text_probe_allocations - allocation_start
  preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
  wait_attempted = true
  command.commit_and_wait
  after_anchor = clock_anchor!

  active_baseline.postwait!(command)
  active_candidate.postwait!(command)
  metrics = active_baseline.metrics(command)
  postterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readback_start
  wait_count = command.wait_count
  baseline_counts = active_baseline.counts
  candidate_counts = active_candidate.counts
  candidate_operation_counts = G.operation_count_report(candidate_counts)
  raw_roots = command.completed_root_timings
  sample_timings = T.build_sample_timings(stage_timings, raw_roots)
  api_profile = command.api_profile
  T.validate_ffn_api_profile!(api_profile)
  profile_json = T.api_profile_json(api_profile)
  api_profile_pass = true
  metric_failures = T.validate_metrics!(metrics)

  execution_failures = [] of String
  begin
    T.validate_baseline_counts!(baseline_counts)
  rescue error : ArgumentError
    execution_failures << (error.message || "baseline counts invalid")
  end
  begin
    T.validate_candidate_counts!(candidate_counts)
  rescue error : ArgumentError
    execution_failures << (error.message || "candidate counts invalid")
  end
  execution_failures << "named buffer cap exceeded" unless named_bytes > 0 && named_bytes <= T::NAMED_BUFFER_CAP_BYTES
  execution_failures << "named GPU buffers were allocated during encode" unless allocations == 0
  execution_failures << "readback occurred before terminal wait" unless preterminal_readbacks == 0
  execution_failures << "expected exactly one packed post-wait readback" unless postterminal_readbacks == 1
  execution_failures << "expected exactly one terminal wait" unless wait_count == 1 && command.completed?
  execution_failures << "local memory pressure preflight did not remain above 10%" unless free_percent_before > 10 && free_percent_after > 10

  roots_json = raw_roots.map do |root|
    JSON.parse({sequence: root.sequence, completed: root.completed,
      has_error: root.has_error, gpu_start_seconds: root.gpu_start_seconds,
      gpu_end_seconds: root.gpu_end_seconds}.to_json)
  end
  sample_json = sample_timings.map { |timing| JSON.parse(timing.to_json) }
  metric_rows = metrics.map_with_index do |metric, index|
    name = if index < 20
      stage = index.even? ? "product" : "down"
      "#{T::SCHEDULE[index // 2][:label]}.#{stage}_vs_saved"
    elsif index < 54
      "unused_#{index}"
    else
      {54 => "raw_gate_vs_saved", 55 => "raw_up_vs_saved", 56 => "injected_nan_control"}[index]
    end
    JSON.parse({slot: index, name: name, failures: metric[:failures],
      different_words: metric[:different_words], nonfinite: metric[:nonfinite],
      elements: metric[:elements]}.to_json)
  end
  correctness_pass = metric_failures.empty?
  execution_pass = execution_failures.empty?
  timing_pass = sample_timings.size == T::SCHEDULE.size &&
                sample_timings.all? { |sample| sample[:outside_root_overlap_seconds] == 0.0 }
  result = {
    schema: T::SCHEMA,
    scope: "diagnostic-only J5760 FFN baseline-vs-stage_outputs timed samples; one chain; no generation",
    pass: correctness_pass && execution_pass && timing_pass && api_profile_pass,
    correctness_pass: correctness_pass,
    execution_pass: execution_pass,
    timing_isolation_pass: timing_pass,
    production_promoted: false,
    full_generation_executed: false,
    frequency_matched: false,
    performance_promoted: false,
    frequency_claim: "none; correlate with the external power observer at the fixed frequency gate before comparing arms",
    protocol: {
      tokens: C::TOKENS, hidden: C::HIDDEN, ffn: C::FFN,
      window: "w0", block: 0,
      candidate_topology: "stage_outputs",
      candidate_graph_sequence: "output-only BF16 gate/up/silu/product front graph followed by the separate production F32 down graph",
      repeat_count_per_sample: T::REPEATS,
      terminal_waits: 1,
      api_profile_enabled: true,
      api_phase_roots_enabled: false,
      api_profile_phase: "ffn",
      sample_order: T::SCHEDULE,
      comparator_slots: {timed_comparisons: "0..19", unused_zero_coverage: "20..53",
        saved_raw_controls: "54..55", injected_nan_control: 56},
      no_generation: true,
      no_promotion: true,
    },
    parity: {
      path: parity_report, sha256: parity_digest,
      schema: parity_document["schema"].as_s,
      candidate_topology: parity_document["protocol"]["candidate_topology"].as_s,
      fixture_path: File.realpath(reference),
      fixture_metadata_sha256: fixture.sha256,
      fixture_payload_sha256: fixture.payload_sha256,
      weight_cache_metadata_sha256: cache.tensor_file.sha256,
      weight_cache_payload_sha256: cache.tensor_file.document["payload_sha256"].as_s,
      atol: B::ATOL, rtol: B::RTOL,
      source_sha256: source_hashes,
    },
    clock_anchors: {before_encode: before_anchor, after_terminal_wait: after_anchor},
    schedule: sample_json,
    gpu_roots: roots_json,
    api_profile: profile_json,
    api_profile_semantics: "graph_encode wall/thread-CPU totals include nested continuation spans; preserve nested and exclusive fields separately and do not add inclusive parent and child durations",
    memory_guard: {
      log_path: guard_log,
      limits: guard_limit,
      started: guard_started,
      policy: "sampled 8 GiB footprint, 10 percent runtime floor, 64 MiB swap-growth cap, at most 600 seconds; not a reservation and cannot prevent between-sample spikes",
    },
    execution: {
      named_bytes: named_bytes, maximum_named_bytes: T::NAMED_BUFFER_CAP_BYTES,
      named_resources: resources,
      free_memory_percent_before_allocate: free_percent_before,
      free_memory_percent_after_allocate: free_percent_after,
      encoding_named_allocation_delta: allocations,
      preterminal_readbacks: preterminal_readbacks,
      total_readbacks_after_wait: postterminal_readbacks,
      wait_count: wait_count,
      baseline_operation_counts: baseline_counts,
      expected_baseline_operation_counts: T::BASELINE_COUNTS,
      candidate_graph_operation_counts: candidate_operation_counts,
      expected_candidate_graph_operation_counts: G.operation_count_report(T::CANDIDATE_COUNTS),
      api_profile_pass: api_profile_pass,
      metrics: metric_rows,
      metric_failures: metric_failures,
      execution_failures: execution_failures,
    },
  }
  fresh_report!(report, result)
  report_written = true
  puts({report: report, pass: result[:pass], correctness_pass: correctness_pass,
    execution_pass: execution_pass, timing_isolation_pass: timing_pass,
    frequency_matched: false}.to_json)
  raise "FFN stage timing execution/correctness/isolation gate failed; report retained" unless result[:pass]
rescue error
  failure = error.message || error.class.name
  raise error
ensure
  if active_chain = chain
    if !active_chain.completed? && !wait_attempted
      wait_attempted = true
      active_chain.commit_and_wait
    end
    if active_chain.completed?
      if active_candidate = candidate
        active_candidate.postwait!(active_chain) unless active_candidate.lifecycle.terminal_wait_succeeded?
        active_candidate.release_after_terminal!(active_chain)
      end
      if active_baseline = baseline
        active_baseline.postwait!(active_chain) unless active_baseline.lifecycle.terminal_wait_succeeded?
        active_baseline.release_after_terminal!(active_chain)
      end
      active_chain.release
    end
  else
    candidate.try(&.release_unsubmitted!)
    baseline.try(&.release_unsubmitted!)
  end
  STDERR.puts("PIXELUMM_FFN_STAGE_TIMING_INCOMPLETE #{failure || "probe did not reach its report"}") unless report_written
end
