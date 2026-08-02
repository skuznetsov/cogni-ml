module ML::Metal::Wba
  enum Verdict
    Refuted
    CandidateDescent
    ProfileOnly
    InsufficientEvidence
    Incomparable
  end

  # A local optimization is eligible for comparison only when its transport,
  # safety boundary, and recomputation obligations are explicit.
  record Card,
    window_or_trigger : String,
    transport_corridor : String,
    legal_move : String,
    boundary_safety : String,
    recompute_safety : String,
    dual_frame : String,
    local_certificate : String do
    def initialize(
      @window_or_trigger : String,
      @transport_corridor : String,
      @legal_move : String,
      @boundary_safety : String,
      @recompute_safety : String,
      @dual_frame : String,
      @local_certificate : String,
    )
      {
        "window_or_trigger"  => @window_or_trigger,
        "transport_corridor" => @transport_corridor,
        "legal_move"         => @legal_move,
        "boundary_safety"    => @boundary_safety,
        "recompute_safety"   => @recompute_safety,
        "dual_frame"         => @dual_frame,
        "local_certificate"  => @local_certificate,
      }.each do |name, value|
        raise ArgumentError.new("#{name} must not be empty") if value.empty?
      end
    end
  end

  record ResourceEnvelope,
    pipeline_entries : Int32,
    pipeline_capacity : Int32,
    pipeline_compile_attempts : Int64,
    pipeline_compile_budget : Int64,
    peak_live_bytes : Int64,
    peak_live_budget_bytes : Int64,
    memory_pressure_events : Int32,
    padding_waste_ppm : Int32,
    padding_waste_limit_ppm : Int32 do
    def initialize(
      @pipeline_entries : Int32,
      @pipeline_capacity : Int32,
      @pipeline_compile_attempts : Int64,
      @pipeline_compile_budget : Int64,
      @peak_live_bytes : Int64,
      @peak_live_budget_bytes : Int64,
      @memory_pressure_events : Int32,
      @padding_waste_ppm : Int32,
      @padding_waste_limit_ppm : Int32,
    )
      {
        "pipeline_entries"          => @pipeline_entries.to_i64,
        "pipeline_capacity"         => @pipeline_capacity.to_i64,
        "pipeline_compile_attempts" => @pipeline_compile_attempts,
        "pipeline_compile_budget"   => @pipeline_compile_budget,
        "peak_live_bytes"           => @peak_live_bytes,
        "peak_live_budget_bytes"    => @peak_live_budget_bytes,
        "memory_pressure_events"    => @memory_pressure_events.to_i64,
        "padding_waste_ppm"         => @padding_waste_ppm.to_i64,
        "padding_waste_limit_ppm"   => @padding_waste_limit_ppm.to_i64,
      }.each do |name, value|
        raise ArgumentError.new("#{name} must be non-negative") if value < 0
      end
      raise ArgumentError.new("pipeline_capacity must be positive") unless @pipeline_capacity > 0
      unless @pipeline_compile_budget > 0
        raise ArgumentError.new("pipeline_compile_budget must be positive")
      end
      unless @peak_live_budget_bytes > 0
        raise ArgumentError.new("peak_live_budget_bytes must be positive")
      end
      unless @padding_waste_ppm <= 1_000_000 && @padding_waste_limit_ppm <= 1_000_000
        raise ArgumentError.new("padding waste ppm values must not exceed 1000000")
      end
    end

    def violations : Array(String)
      violations = [] of String
      if @pipeline_entries > @pipeline_capacity
        violations << "pipeline_capacity_exceeded"
      end
      if @pipeline_compile_attempts > @pipeline_compile_budget
        violations << "pipeline_compile_budget_exceeded"
      end
      if @peak_live_bytes > @peak_live_budget_bytes
        violations << "peak_live_budget_exceeded"
      end
      if @memory_pressure_events > 0
        violations << "memory_pressure_events=#{@memory_pressure_events}"
      end
      if @padding_waste_ppm > @padding_waste_limit_ppm
        violations << "padding_waste_limit_exceeded"
      end
      violations
    end

    def policy_identity : {Int32, Int64, Int64, Int32}
      {
        @pipeline_capacity,
        @pipeline_compile_budget,
        @peak_live_budget_bytes,
        @padding_waste_limit_ppm,
      }
    end
  end

  record Potential,
    parity_failures : Int32,
    resource_limit_violations : Int32,
    active_window_gpu_ns : Int64,
    tied_dominant_routes : Int32,
    conflict_or_sync_count : Int32,
    boundary_transfer_bytes : Int64,
    remaining_work_bytes : Int64 do
    def values : Array(Int64)
      [
        @parity_failures.to_i64,
        @resource_limit_violations.to_i64,
        @active_window_gpu_ns,
        @tied_dominant_routes.to_i64,
        @conflict_or_sync_count.to_i64,
        @boundary_transfer_bytes,
        @remaining_work_bytes,
      ]
    end

    def descends_from?(baseline : Potential) : Bool
      candidate_values = values
      baseline_values = baseline.values
      candidate_values.each_with_index do |value, index|
        other = baseline_values[index]
        return true if value < other
        return false if value > other
      end
      false
    end
  end

  record CorridorObservation,
    card : Card,
    corridor_id : String,
    profile_id : String,
    bucket_id : String,
    active_window : String,
    active_window_gpu_ns : Int64,
    wall_ns : Int64,
    tied_dominant_routes : Int32,
    sync_count : Int32,
    boundary_transfer_bytes : Int64,
    remaining_work_bytes : Int64,
    parity_failures : Int32,
    resources : ResourceEnvelope do
    def initialize(
      @card : Card,
      @corridor_id : String,
      @profile_id : String,
      @bucket_id : String,
      @active_window : String,
      @active_window_gpu_ns : Int64,
      @wall_ns : Int64,
      @tied_dominant_routes : Int32,
      @sync_count : Int32,
      @boundary_transfer_bytes : Int64,
      @remaining_work_bytes : Int64,
      @parity_failures : Int32,
      @resources : ResourceEnvelope,
    )
      {
        "corridor_id"   => @corridor_id,
        "profile_id"    => @profile_id,
        "bucket_id"     => @bucket_id,
        "active_window" => @active_window,
      }.each do |name, value|
        if value.empty?
          raise ArgumentError.new("#{name} must not be empty")
        end
      end
      {
        "active_window_gpu_ns"    => @active_window_gpu_ns,
        "wall_ns"                 => @wall_ns,
        "tied_dominant_routes"    => @tied_dominant_routes.to_i64,
        "sync_count"              => @sync_count.to_i64,
        "boundary_transfer_bytes" => @boundary_transfer_bytes,
        "remaining_work_bytes"    => @remaining_work_bytes,
        "parity_failures"         => @parity_failures.to_i64,
      }.each do |name, value|
        raise ArgumentError.new("#{name} must be non-negative") if value < 0
      end
    end

    def identity : {String, String, String, String}
      {@corridor_id, @profile_id, @bucket_id, @active_window}
    end

    def potential : Potential
      Potential.new(
        parity_failures: @parity_failures,
        resource_limit_violations: @resources.violations.size,
        active_window_gpu_ns: @active_window_gpu_ns,
        tied_dominant_routes: @tied_dominant_routes,
        conflict_or_sync_count: @sync_count,
        boundary_transfer_bytes: @boundary_transfer_bytes,
        remaining_work_bytes: @remaining_work_bytes
      )
    end
  end

  record PairedCertificate, pairs : Int32, candidate_wins : Int32 do
    def initialize(@pairs : Int32, @candidate_wins : Int32)
      raise ArgumentError.new("pairs must be positive") unless @pairs > 0
      unless 0 <= @candidate_wins <= @pairs
        raise ArgumentError.new("candidate_wins must be within 0..pairs")
      end
    end

    def majority? : Bool
      @candidate_wins > @pairs // 2
    end
  end

  class Analysis
    getter verdict : Verdict
    getter baseline : CorridorObservation
    getter candidate : CorridorObservation?
    getter certificate : PairedCertificate?
    getter reasons : Array(String)

    def initialize(
      @verdict : Verdict,
      @baseline : CorridorObservation,
      @candidate : CorridorObservation?,
      @certificate : PairedCertificate?,
      reasons : Array(String),
    )
      @reasons = reasons.dup
    end
  end

  module Atlas
    extend self

    # Pure structural classifier. Observations and paired counts are
    # caller-supplied and remain untrusted until a later telemetry boundary
    # binds them to immutable samples and a qualified host/runtime.

    def profile(observation : CorridorObservation) : Analysis
      Analysis.new(
        Verdict::ProfileOnly,
        observation,
        nil,
        nil,
        ["missing_paired_certificate"]
      )
    end

    def compare(
      baseline : CorridorObservation,
      candidate : CorridorObservation,
      certificate : PairedCertificate,
    ) : Analysis
      unless baseline.identity == candidate.identity
        return Analysis.new(
          Verdict::Incomparable,
          baseline,
          candidate,
          certificate,
          ["corridor_identity_mismatch"]
        )
      end

      unless baseline.resources.policy_identity == candidate.resources.policy_identity
        return Analysis.new(
          Verdict::Incomparable,
          baseline,
          candidate,
          certificate,
          ["resource_policy_mismatch"]
        )
      end

      unless baseline.card == candidate.card
        return Analysis.new(
          Verdict::Incomparable,
          baseline,
          candidate,
          certificate,
          ["corridor_card_mismatch"]
        )
      end

      if certificate.pairs < 3
        return Analysis.new(
          Verdict::InsufficientEvidence,
          baseline,
          candidate,
          certificate,
          ["at_least_three_pairs_required"]
        )
      end

      if baseline.active_window_gpu_ns == 0 || candidate.active_window_gpu_ns == 0
        return Analysis.new(
          Verdict::InsufficientEvidence,
          baseline,
          candidate,
          certificate,
          ["positive_gpu_timing_required"]
        )
      end

      if baseline.wall_ns == 0 || candidate.wall_ns == 0
        return Analysis.new(
          Verdict::InsufficientEvidence,
          baseline,
          candidate,
          certificate,
          ["positive_full_corridor_wall_required"]
        )
      end

      baseline_reasons = [] of String
      if baseline.parity_failures > 0
        baseline_reasons << "baseline_parity_failures=#{baseline.parity_failures}"
      end
      baseline.resources.violations.each do |reason|
        baseline_reasons << "baseline_#{reason}"
      end
      unless baseline_reasons.empty?
        return Analysis.new(
          Verdict::InsufficientEvidence,
          baseline,
          candidate,
          certificate,
          baseline_reasons
        )
      end

      reasons = [] of String
      if candidate.parity_failures > 0
        reasons << "parity_failures=#{candidate.parity_failures}"
      end
      reasons.concat(candidate.resources.violations)
      unless candidate.wall_ns < baseline.wall_ns && certificate.majority?
        reasons << "full_corridor_wall_did_not_descend"
      end
      unless candidate.potential.descends_from?(baseline.potential)
        reasons << "lexicographic_potential_did_not_descend"
      end

      unless reasons.empty?
        return Analysis.new(
          Verdict::Refuted,
          baseline,
          candidate,
          certificate,
          reasons
        )
      end

      Analysis.new(
        Verdict::CandidateDescent,
        baseline,
        candidate,
        certificate,
        ["paired_full_corridor_descent"]
      )
    end
  end
end
