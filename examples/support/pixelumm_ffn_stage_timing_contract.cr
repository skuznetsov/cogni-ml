require "json"
require "set"
require "./pixelumm_ffn_graph_probe"

module PixelUMMFFNStageTimingContract
  extend self

  alias G = PixelUMMFFNGraphProbe
  alias A = G::A
  alias B = G::B
  alias C = G::C
  alias I = G::I
  alias Metric = G::Metric
  alias StageEntry = NamedTuple(label: String, mode: String, warmup: Bool)
  alias SampleTiming = NamedTuple(
    label: String,
    mode: String,
    warmup: Bool,
    repeats: Int32,
    first_root: Int32,
    end_root: Int32,
    encode_wall_seconds: Float64,
    gpu_root_union_seconds: Float64,
    first_gpu_start_seconds: Float64,
    last_gpu_end_seconds: Float64,
    gpu_root_count: Int32,
    outside_root_overlap_seconds: Float64,
  )

  SCHEMA = "cogni_ml.pixelumm_ffn_stage_timing_probe/v1"
  REPEATS = 128
  METRIC_SLOTS = 57
  NAMED_BUFFER_CAP_BYTES = A::MAX_NAMED_BYTES
  EXTERNAL_GUARD_FOOTPRINT_BYTES = 8_i64 * 1024 * 1024 * 1024
  EXTERNAL_GUARD_SWAP_BYTES = 64_i64 * 1024 * 1024
  EXTERNAL_GUARD_MIN_FREE_PERCENT = 10
  EXTERNAL_GUARD_MAX_TIMEOUT_SECONDS = 600
  MACH_ANCHOR_MAX_BRACKET_NS = 1_000_000_i64

  SOURCE_PATHS = [
    "examples/pixelumm_ffn_graph_probe.cr",
    "examples/support/pixelumm_ffn_graph_probe.cr",
    "examples/support/pixelumm_ffn_graph_bridge.mm",
    "examples/support/pixelumm_ffn_aa_bridge.mm",
    "examples/support/pixelumm_ffn_aa_probe.cr",
    "examples/support/pixelumm_ffn_aa_contract.cr",
    "examples/support/pixelumm_ffn_isolated_probe.cr",
    "examples/support/pixelumm_ffn_weight_reuse_probe.cr",
    "examples/support/wan_memory_guard.py",
    "examples/support/pixelumm_ffn_dense.metal",
    "src/ml/video/pixelumm_decoder_block_mps.cr",
    "src/ml/video/pixelumm_decoder_block_mps_bridge.mm",
    "src/ml/video/metal_mps_command_chain.cr",
    "src/ml/video/metal_mps_command_chain.mm",
    "src/ml/metal/bridge.mm",
  ]

  SCHEDULE = [
    {label: "A_warm", mode: "A", warmup: true},
    {label: "B_warm", mode: "B", warmup: true},
    {label: "A_1", mode: "A", warmup: false},
    {label: "B_1", mode: "B", warmup: false},
    {label: "B_2", mode: "B", warmup: false},
    {label: "A_2", mode: "A", warmup: false},
    {label: "B_3", mode: "B", warmup: false},
    {label: "A_3", mode: "A", warmup: false},
    {label: "A_4", mode: "A", warmup: false},
    {label: "B_4", mode: "B", warmup: false},
  ]

  BASELINE_COUNTS = Array(UInt64).new(6, 641_u64) + [2_u64]
  CANDIDATE_COUNTS = [640_u64, 640_u64, 0_u64, 0_u64, 0_u64]

  def validate_schedule!(schedule : Array(StageEntry)) : Nil
    unless schedule == SCHEDULE && schedule.count { |entry| entry[:mode] == "A" } == 5 &&
           schedule.count { |entry| entry[:mode] == "B" } == 5 &&
           schedule.count { |entry| entry[:warmup] } == 2
      raise ArgumentError.new("FFN timing schedule differs from the fixed 10-sample A/B order")
    end
  end

  def validate_baseline_counts!(counts : Array(UInt64)) : Nil
    raise ArgumentError.new("FFN baseline operation counts differ from 641 calls and two raw copies") unless counts == BASELINE_COUNTS
  end

  def validate_candidate_counts!(counts : Array(UInt64)) : Nil
    raise ArgumentError.new("FFN stage-output counts differ from 640 output-only front/down calls") unless counts == CANDIDATE_COUNTS
  end

  def validate_metrics!(metrics : Array(Metric)) : Array(String)
    raise ArgumentError.new("FFN stage timing metric vector must contain exactly 57 slots") unless metrics.size == METRIC_SLOTS
    failures = [] of String
    product_elements = C::PRODUCT_ELEMENTS.to_u32
    down_elements = C::DOWN_ELEMENTS.to_u32
    10.times do |sample|
      product = metrics[sample * 2]
      down = metrics[sample * 2 + 1]
      failures << "sample #{sample} product comparator coverage/failure mismatch: #{product}" unless
        product[:elements] == product_elements && product[:failures] == 0 && product[:nonfinite] == 0
      failures << "sample #{sample} down comparator coverage/failure mismatch: #{down}" unless
        down[:elements] == down_elements && down[:failures] == 0 && down[:nonfinite] == 0
    end

    (20...54).each do |slot|
      metric = metrics[slot]
      unless metric == {failures: 0_u32, different_words: 0_u32,
                        nonfinite: 0_u32, elements: 0_u32}
        failures << "unused comparator slot #{slot} is nonzero: #{metric}"
      end
    end

    expected_gate = {failures: 0_u32, different_words: metrics[54][:different_words],
                     nonfinite: 0_u32, elements: (C::TOKENS * C::FFN).to_u32}
    expected_up = {failures: 0_u32, different_words: metrics[55][:different_words],
                   nonfinite: 0_u32, elements: (C::TOKENS * C::FFN).to_u32}
    failures << "gate saved-stage control failed: #{metrics[54]}" unless metrics[54] == expected_gate
    failures << "up saved-stage control failed: #{metrics[55]}" unless metrics[55] == expected_up
    expected_nan = {failures: 1_u32, different_words: 0_u32,
                    nonfinite: 1_u32, elements: down_elements}
    failures << "injected-NaN control mismatch: #{metrics[56]}" unless metrics[56] == expected_nan
    failures
  end

  def validate_stage_boundaries!(timings : Array(I::StageTiming)) : Nil
    raise ArgumentError.new("FFN stage timing count differs from the fixed schedule") unless timings.size == SCHEDULE.size
    timings.each_with_index do |timing, index|
      expected = SCHEDULE[index]
      unless timing[:stage] == expected[:label] && timing[:first_root] >= 0 &&
             timing[:end_root] > timing[:first_root] &&
             timing[:encode_wall_seconds].finite? && timing[:encode_wall_seconds] > 0.0
        raise ArgumentError.new("FFN timing boundary does not match schedule entry #{index}")
      end
      if index > 0 && timings[index - 1][:end_root] > timing[:first_root]
        raise ArgumentError.new("FFN sample root ranges are out of order or overlap")
      end
    end
  end

  def build_sample_timings(timings : Array(I::StageTiming),
                           roots : Array(ML::Video::MPSCommandRootTiming)) : Array(SampleTiming)
    validate_stage_boundaries!(timings)
    unless !roots.empty? && roots.map(&.sequence) == (0...roots.size).to_a &&
           roots.all? { |root| root.completed && !root.has_error &&
             root.gpu_start_seconds.finite? && root.gpu_end_seconds.finite? &&
             root.gpu_start_seconds >= 0.0 && root.gpu_end_seconds > root.gpu_start_seconds }
      raise ArgumentError.new("FFN GPU root inventory is incomplete, failed, or non-finite")
    end

    timings.map_with_index do |timing, index|
      entry = SCHEDULE[index]
      summary = root_summary(timing, roots)
      unless summary[:isolated_root_timing] && summary[:outside_root_overlap_seconds] == 0.0 &&
             summary[:gpu_root_union_seconds].finite? && summary[:gpu_root_union_seconds] > 0.0
        raise ArgumentError.new("FFN sample #{entry[:label]} overlaps a root outside its timed range")
      end
      selected = roots.select { |root| timing[:first_root] <= root.sequence < timing[:end_root] }
      raise ArgumentError.new("FFN sample #{entry[:label]} has no GPU roots") if selected.empty?
      {
        label: entry[:label], mode: entry[:mode], warmup: entry[:warmup], repeats: REPEATS,
        first_root: timing[:first_root], end_root: timing[:end_root],
        encode_wall_seconds: timing[:encode_wall_seconds],
        gpu_root_union_seconds: summary[:gpu_root_union_seconds],
        first_gpu_start_seconds: selected.min_of(&.gpu_start_seconds),
        last_gpu_end_seconds: selected.max_of(&.gpu_end_seconds),
        gpu_root_count: selected.size.to_i32,
        outside_root_overlap_seconds: summary[:outside_root_overlap_seconds],
      }
    end
  end

  def validate_clock_anchor!(monotonic_before_ns : Int64, unix_ns : Int64,
                             monotonic_after_ns : Int64) : Nil
    unless monotonic_before_ns > 0 && unix_ns > 0 && monotonic_after_ns >= monotonic_before_ns &&
           monotonic_after_ns - monotonic_before_ns <= MACH_ANCHOR_MAX_BRACKET_NS
      raise ArgumentError.new("FFN clock anchor is invalid or wider than one millisecond")
    end
  end

  def validate_source_hashes!(reported : Hash(String, String),
                              current : Hash(String, String)) : Nil
    unless !current.empty? && reported.size == current.size &&
           reported.keys.to_set == current.keys.to_set
      raise ArgumentError.new("stage-output parity source inventory differs from current sources")
    end
    current.each do |path, digest|
      unless valid_sha256?(digest) && reported[path]? == digest
        raise ArgumentError.new("stage-output parity source hash differs: #{path}")
      end
    end
  end

  # The outer sampled guard starts this process in a dedicated session. Verify
  # its live admission record and identity; this checks the stated envelope,
  # not a reservation or protection from spikes between samples.
  def validate_guard_records!(records : Array(JSON::Any), process_id : Int64) : JSON::Any
    limits = records.select { |record| member(record, "event").as_s == "limits" }
    started = records.select { |record| member(record, "event").as_s == "started" }
    unless limits.size == 1 && started.size == 1 &&
           records.none? { |record| {"blocked", "launch_failed", "stopping", "finished", "cleanup_failed", "cleanup_incomplete"}.includes?(member(record, "event").as_s) }
      raise ArgumentError.new("external memory guard has no unique active admission/identity record")
    end
    limit = limits.first
    start = started.first
    timeout = integer(member(limit, "timeout_sec"), "guard timeout")
    unless integer(member(limit, "max_footprint_bytes"), "guard max footprint") == EXTERNAL_GUARD_FOOTPRINT_BYTES &&
           integer(member(limit, "max_swap_growth_bytes"), "guard max swap") == EXTERNAL_GUARD_SWAP_BYTES &&
           integer(member(limit, "runtime_floor_pct"), "guard runtime floor") == EXTERNAL_GUARD_MIN_FREE_PERCENT &&
           member(limit, "runtime_floor_pct_exact").as_s == "10" &&
           timeout > 0 && timeout <= EXTERNAL_GUARD_MAX_TIMEOUT_SECONDS
      raise ArgumentError.new("external memory guard limits differ from the 8 GiB/10%/64 MiB/600 s envelope")
    end
    unless integer(member(start, "pid"), "guard child pid") == process_id &&
           integer(member(start, "pgid"), "guard child pgid") == process_id
      raise ArgumentError.new("external memory guard PID/PGID does not own this probe process")
    end
    limit
  end

  def validate_stage_outputs_parity_report!(document : JSON::Any,
                                            current_source_hashes : Hash(String, String),
                                            fixture_path : String,
                                            fixture_metadata_sha256 : String,
                                            fixture_payload_sha256 : String,
                                            weight_cache_metadata_sha256 : String,
                                            weight_cache_payload_sha256 : String,
                                            normalized_input_sha256 : String) : Nil
    begin
      validate_parity_report_inner!(document, current_source_hashes, fixture_path,
        fixture_metadata_sha256, fixture_payload_sha256,
        weight_cache_metadata_sha256, weight_cache_payload_sha256,
        normalized_input_sha256)
    rescue error : ArgumentError
      raise error
    rescue error
      raise ArgumentError.new("malformed stage-output parity report: #{error.message || error.class.name}")
    end
  end

  def api_profile_json(profile : ML::Video::MPSAPIProfile) : Array(JSON::Any)
    profile.phases.map do |phase|
      operations = {} of String => JSON::Any
      phase.operations.each do |name, stats|
        operations[name] = JSON.parse({count: stats.count, wall_total_ns: stats.wall_total_ns,
          wall_max_ns: stats.wall_max_ns, thread_cpu_total_ns: stats.thread_cpu_total_ns,
          thread_cpu_max_ns: stats.thread_cpu_max_ns, nested_count: stats.nested_count,
          nested_wall_total_ns: stats.nested_wall_total_ns,
          nested_thread_cpu_total_ns: stats.nested_thread_cpu_total_ns,
          exclusive_wall_total_ns: stats.exclusive_wall_total_ns,
          exclusive_thread_cpu_total_ns: stats.exclusive_thread_cpu_total_ns}.to_json)
      end
      JSON.parse({phase_id: phase.phase_id, phase: phase.name, operations: operations}.to_json)
    end
  end

  def validate_ffn_api_profile!(profile : ML::Video::MPSAPIProfile) : Nil
    ffn = profile.phases.find { |phase| phase.phase_id == 5 } ||
          raise ArgumentError.new("MPS API profile omits the FFN phase")
    unless ffn.operations["graph_encode"].count > 0 &&
           ffn.operations["commit_and_continue"].count > 0 &&
           profile.phases.all? { |phase| phase.phase_id == 5 || phase.operations.values.all?(&.count.zero?) }
      raise ArgumentError.new("MPS API calls were not attributed solely to the FFN phase")
    end
  end

  private def validate_parity_report_inner!(document : JSON::Any,
                                            current_source_hashes : Hash(String, String),
                                            fixture_path : String,
                                            fixture_metadata_sha256 : String,
                                            fixture_payload_sha256 : String,
                                            weight_cache_metadata_sha256 : String,
                                            weight_cache_payload_sha256 : String,
                                            normalized_input_sha256 : String) : Nil
    string!(document, "schema", G::SCHEMA)
    bool!(document, "pass", true)
    bool!(document, "probe_completed", true)
    bool!(document, "numerical_pass", true)
    bool!(document, "execution_pass", true)
    bool!(document, "production_promoted", false)
    bool!(document, "full_generation_executed", false)
    bool!(document, "correctness_only", true)

    fixture = member(document, "fixture")
    {
      {"path", fixture_path},
      {"metadata_sha256", fixture_metadata_sha256},
      {"payload_sha256", fixture_payload_sha256},
      {"weight_cache_metadata_sha256", weight_cache_metadata_sha256},
      {"weight_cache_payload_sha256", weight_cache_payload_sha256},
      {"input_window", "w0"},
      {"input_provenance", "saved_initial_noise_conditioning_upstream_stem"},
      {"upstream_revision", B::UPSTREAM_REVISION},
      {"upstream_model_sha256", B::MODEL_SHA256},
    }.each { |key, expected| string!(fixture, key, expected) }
    unless integer(member(fixture, "block"), "fixture block") == 0
      raise ArgumentError.new("stage-output parity report names the wrong fixture block")
    end
    bool!(fixture, "model_inference", false)
    string!(fixture, "normalized_input_sha256", normalized_input_sha256)

    comparison = member(document, "comparison")
    unless float(member(comparison, "atol"), "comparison atol") == B::ATOL &&
           float(member(comparison, "rtol"), "comparison rtol") == B::RTOL
      raise ArgumentError.new("stage-output parity tolerance differs from 2e-5/2e-5")
    end

    protocol = member(document, "protocol")
    unless integer(member(protocol, "tokens"), "protocol tokens") == C::TOKENS &&
           integer(member(protocol, "hidden"), "protocol hidden") == C::HIDDEN &&
           integer(member(protocol, "ffn"), "protocol ffn") == C::FFN
      raise ArgumentError.new("stage-output parity geometry differs from J5760 FFN")
    end
    string!(protocol, "candidate_topology", "stage_outputs")
    bool!(protocol, "stage_outputs_are_mandatory_targets", true)
    bool!(protocol, "output_only_and_captured_graphs_compiled_separately", true)
    bool!(protocol, "output_only_down_compared_to_saved_and_native_baseline", true)
    bool!(protocol, "output_only_product_compared_to_saved_and_native_baseline_before_capture", true)
    bool!(protocol, "output_only_front_stages_compared_before_capture", true)
    bool!(protocol, "captured_stages_compared_to_saved_and_native_baseline", true)
    bool!(protocol, "no_generation", true)
    bool!(protocol, "no_promotion", true)
    unless integer(member(protocol, "candidate_timing_samples"), "candidate parity timing samples").zero? &&
           integer(member(protocol, "baseline_timing_samples"), "baseline parity timing samples").zero? &&
           integer(member(protocol, "terminal_waits"), "parity terminal waits") == 1 &&
           integer(member(protocol, "comparator_readback_passes"), "parity comparator readbacks") == 1 &&
           integer(member(protocol, "encoded_named_buffer_allocations_allowed"), "parity encoded allocations").zero? &&
           integer(member(protocol, "preterminal_readbacks_allowed"), "parity preterminal readbacks").zero?
      raise ArgumentError.new("stage-output parity execution protocol differs from the fixed correctness probe")
    end
    unless member(protocol, "output_only_front_stage_outputs").as_a.map(&.as_s) ==
           ["graph.gate", "graph.up", "graph.silu", "graph.product"]
      raise ArgumentError.new("stage-output parity report omits a mandatory front-stage output")
    end

    expected_counts = G.operation_count_report(G.expected_candidate_counts(G::Topology::StageOutputs))
    unless json_integer_map(member(protocol, "expected_candidate_graph_operation_counts")) == expected_counts &&
           member(protocol, "candidate_graph_operation_count_names").as_a.map(&.as_s) == G::GRAPH_OPERATION_NAMES
      raise ArgumentError.new("stage-output parity expected operation inventory differs")
    end

    execution = member(document, "execution")
    unless integer(member(execution, "named_bytes"), "parity named bytes") > 0 &&
           integer(member(execution, "named_bytes"), "parity named bytes") <= NAMED_BUFFER_CAP_BYTES &&
           integer(member(execution, "maximum_named_bytes"), "parity named cap") == NAMED_BUFFER_CAP_BYTES &&
           integer(member(execution, "encoding_named_allocation_delta"), "parity encoding allocation delta").zero? &&
           integer(member(execution, "preterminal_readbacks"), "parity preterminal readbacks").zero? &&
           integer(member(execution, "total_readbacks_after_wait"), "parity postwait readbacks") == 1 &&
           integer(member(execution, "wait_count"), "parity wait count") == 1
      raise ArgumentError.new("stage-output parity execution counters do not pass")
    end
    unless member(execution, "baseline_operation_counts").as_a.map(&.as_i64) == [1_i64, 1, 1, 1, 1, 1, 2] &&
           json_integer_map(member(execution, "candidate_graph_operation_counts")) == expected_counts
      raise ArgumentError.new("stage-output parity operation counts differ from the fixed graph paths")
    end

    validate_report_stages!(member(document, "stages"))

    reported_sources = {} of String => String
    raw_sources = member(document, "source_sha256").as_h
    raise ArgumentError.new("stage-output parity source map is incomplete") unless raw_sources.size == SOURCE_PATHS.size
    raw_sources.each do |path, value|
      resolved = File.realpath(path)
      raise ArgumentError.new("stage-output parity source paths resolve more than once") if reported_sources.has_key?(resolved)
      reported_sources[resolved] = value.as_s
    end
    validate_source_hashes!(reported_sources, current_source_hashes)
  end

  private def validate_report_stages!(value : JSON::Any) : Nil
    rows = value.as_a
    names = G::STAGE_NAMES
    counts = G.expected_stage_counts(G::Topology::StageOutputs)
    raise ArgumentError.new("stage-output parity report has incomplete stage coverage") unless rows.size == names.size
    rows.each_with_index do |row, index|
      string!(row, "name", names[index])
      expected = counts[index].to_i64
      unless integer(member(row, "expected_elements"), "expected stage coverage") == expected &&
             integer(member(row, "elements"), "observed stage coverage") == expected &&
             bool_value(member(row, "passed"), "stage pass")
        raise ArgumentError.new("stage-output parity coverage/pass mismatch for #{names[index]}")
      end
      if index == names.size - 1
        unless integer(member(row, "failures"), "positive control failures") == 1 &&
               integer(member(row, "different_words"), "positive control word differences").zero? &&
               integer(member(row, "nonfinite"), "positive control non-finite") == 1
          raise ArgumentError.new("stage-output parity positive control did not inject exactly one non-finite failure")
        end
      elsif !integer(member(row, "failures"), "stage failures").zero? ||
            !integer(member(row, "nonfinite"), "stage non-finite").zero?
        raise ArgumentError.new("stage-output parity numerical stage failed for #{names[index]}")
      end
    end
  end

  private def root_summary(timing : I::StageTiming,
                           roots : Array(ML::Video::MPSCommandRootTiming)) : NamedTuple(gpu_root_union_seconds: Float64, outside_root_overlap_seconds: Float64, isolated_root_timing: Bool)
    {% unless flag?(:cpu_only) %}
      I.root_timing(timing, roots)
    {% else %}
      selected = roots.select { |root| timing[:first_root] <= root.sequence < timing[:end_root] }
      unless !selected.empty? && selected.all? { |root| root.completed && !root.has_error &&
             root.gpu_start_seconds.finite? && root.gpu_end_seconds.finite? &&
             root.gpu_start_seconds > 0.0 && root.gpu_end_seconds >= root.gpu_start_seconds }
        raise ArgumentError.new("isolated FFN timed root is missing or failed")
      end
      union = interval_union(selected.map { |root| {root.gpu_start_seconds, root.gpu_end_seconds} })
      outside = roots.reject { |root| timing[:first_root] <= root.sequence < timing[:end_root] }
      intersections = [] of Tuple(Float64, Float64)
      selected.each do |root|
        outside.each do |other|
          start = Math.max(root.gpu_start_seconds, other.gpu_start_seconds)
          finish = Math.min(root.gpu_end_seconds, other.gpu_end_seconds)
          intersections << {start, finish} if finish > start
        end
      end
      overlap = interval_union(intersections)
      raise ArgumentError.new("isolated FFN timed root union is empty") unless union > 0.0
      {gpu_root_union_seconds: union, outside_root_overlap_seconds: overlap,
       isolated_root_timing: overlap == 0.0}
    {% end %}
  end

  private def interval_union(intervals : Array(Tuple(Float64, Float64))) : Float64
    return 0.0 if intervals.empty?
    sorted = intervals.sort_by(&.[0])
    start, finish = sorted.first
    total = 0.0
    sorted.skip(1).each do |left, right|
      if left <= finish
        finish = Math.max(finish, right)
      else
        total += finish - start
        start, finish = left, right
      end
    end
    total + finish - start
  end

  private def json_integer_map(value : JSON::Any) : Hash(String, UInt64)
    value.as_h.transform_values do |entry|
      count = integer(entry, "operation count")
      raise ArgumentError.new("operation count is negative") if count < 0
      count.to_u64
    end
  end

  private def valid_sha256?(value : String) : Bool
    !!value.match(/\A[0-9a-f]{64}\z/)
  end

  private def member(value : JSON::Any, key : String) : JSON::Any
    value.as_h[key]? || raise ArgumentError.new("missing JSON field #{key}")
  end

  private def integer(value : JSON::Any, label : String) : Int64
    value.as_i64
  rescue
    raise ArgumentError.new("#{label} must be an integer")
  end

  private def float(value : JSON::Any, label : String) : Float64
    value.as_f
  rescue
    raise ArgumentError.new("#{label} must be a finite number")
  end

  private def bool_value(value : JSON::Any, label : String) : Bool
    value.as_bool
  rescue
    raise ArgumentError.new("#{label} must be a boolean")
  end

  private def string!(value : JSON::Any, key : String, expected : String) : Nil
    observed = member(value, key).as_s
    raise ArgumentError.new("#{key} differs from the pinned value") unless observed == expected
  rescue error : ArgumentError
    raise error
  rescue
    raise ArgumentError.new("#{key} must be the pinned string")
  end

  private def bool!(value : JSON::Any, key : String, expected : Bool) : Nil
    observed = bool_value(member(value, key), key)
    raise ArgumentError.new("#{key} differs from the pinned value") unless observed == expected
  end
end
