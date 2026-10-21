# Pure CPU model for opt-in MPS API timing snapshots. The native collector
# records inclusive API wall/thread-CPU spans and explicitly nested spans;
# callers must not add inclusive parent and child durations together.
module ML::Video
  module MPSAPIProfileSchema
    UNASSIGNED_PHASE = 0
    PHASE_NAMES = {
      0 => "unassigned",
      1 => "self_prepare",
      2 => "self_attention",
      3 => "self_projection_residual",
      4 => "cross_attention_projection_residual",
      5 => "ffn",
    }
    OPERATION_NAMES = {
      0 => "graph_encode",
      1 => "commit_and_continue",
      2 => "t5_linear_encode",
    }
    PHASE_COUNT = PHASE_NAMES.size
    OPERATION_COUNT = OPERATION_NAMES.size
  end

  module MPSAPIPhaseBoundary
    # Root counts name the currently open root as `root_count - 1`. This split
    # assigns that shared tail to the following phase and requires at least one
    # earlier root to complete the attention-side interval.
    def self.precurrent_root_end_count(root_count : Int32,
                                       phase_start_count : Int32) : Int32
      unless root_count > 0 && phase_start_count >= 0 && phase_start_count < root_count
        raise ArgumentError.new("MPS phase split has invalid root counts")
      end
      end_count = root_count - 1
      unless end_count > phase_start_count
        raise ArgumentError.new("MPS phase split has no completed attention root")
      end
      end_count
    end
  end

  # One native snapshot for a (phase, operation) pair. All time fields are
  # integer nanoseconds to preserve the native counters without conversion.
  struct MPSAPITimingStats
    getter count : UInt64
    getter wall_total_ns : UInt64
    getter wall_max_ns : UInt64
    getter thread_cpu_total_ns : UInt64
    getter thread_cpu_max_ns : UInt64
    getter nested_count : UInt64
    getter nested_wall_total_ns : UInt64
    getter nested_thread_cpu_total_ns : UInt64
    getter exclusive_wall_total_ns : UInt64
    getter exclusive_thread_cpu_total_ns : UInt64

    def initialize(@count : UInt64,
                   @wall_total_ns : UInt64,
                   @wall_max_ns : UInt64,
                   @thread_cpu_total_ns : UInt64,
                   @thread_cpu_max_ns : UInt64,
                   @nested_count : UInt64,
                   @nested_wall_total_ns : UInt64,
                   @nested_thread_cpu_total_ns : UInt64,
                   @exclusive_wall_total_ns : UInt64,
                   @exclusive_thread_cpu_total_ns : UInt64)
    end

    def validate!(operation_id : Int32) : Nil
      unless MPSAPIProfileSchema::OPERATION_NAMES.has_key?(operation_id)
        raise ArgumentError.new("MPS API profile has an unknown operation")
      end

      if @count == 0
        unless {@wall_total_ns, @wall_max_ns, @thread_cpu_total_ns,
                @thread_cpu_max_ns, @nested_count, @nested_wall_total_ns,
                @nested_thread_cpu_total_ns, @exclusive_wall_total_ns,
                @exclusive_thread_cpu_total_ns}.all?(&.zero?)
          raise ArgumentError.new("Empty MPS API profile entry has non-zero timing data")
        end
        return
      end

      unless @wall_max_ns > 0 && @thread_cpu_max_ns > 0 &&
             @wall_max_ns <= @wall_total_ns &&
             @thread_cpu_max_ns <= @thread_cpu_total_ns &&
             (operation_id == 0 || @nested_count <= @count)
        raise ArgumentError.new("MPS API profile has inconsistent count or maximum")
      end

      nested_count_empty = @nested_count.zero?
      unless nested_count_empty == @nested_wall_total_ns.zero? &&
             nested_count_empty == @nested_thread_cpu_total_ns.zero?
        raise ArgumentError.new("MPS API profile has inconsistent nested count and timing totals")
      end

      if operation_id == 0
        unless @nested_wall_total_ns <= @wall_total_ns &&
               @nested_thread_cpu_total_ns <= @thread_cpu_total_ns &&
               @exclusive_wall_total_ns == @wall_total_ns - @nested_wall_total_ns &&
               @exclusive_thread_cpu_total_ns == @thread_cpu_total_ns - @nested_thread_cpu_total_ns
          raise ArgumentError.new("MPSGraph API profile has inconsistent nested accounting")
        end
      elsif operation_id == 1
        unless @nested_wall_total_ns <= @wall_total_ns &&
               @nested_thread_cpu_total_ns <= @thread_cpu_total_ns &&
               @exclusive_wall_total_ns == @wall_total_ns &&
               @exclusive_thread_cpu_total_ns == @thread_cpu_total_ns
          raise ArgumentError.new("MPS commit API profile has inconsistent nested accounting")
        end
      else
        unless @nested_count.zero? && @nested_wall_total_ns.zero? && @nested_thread_cpu_total_ns.zero? &&
               @exclusive_wall_total_ns == @wall_total_ns &&
               @exclusive_thread_cpu_total_ns == @thread_cpu_total_ns
          raise ArgumentError.new("Non-graph MPS API profile has unexpected nested accounting")
        end
      end
    end
  end

  struct MPSAPIProfileSample
    getter phase_id : Int32
    getter operation_id : Int32
    getter stats : MPSAPITimingStats

    def initialize(@phase_id : Int32, @operation_id : Int32,
                   @stats : MPSAPITimingStats)
    end
  end

  struct MPSAPIPhaseProfile
    getter phase_id : Int32
    getter name : String
    getter operations : Hash(String, MPSAPITimingStats)

    def initialize(@phase_id : Int32, @name : String,
                   @operations : Hash(String, MPSAPITimingStats))
    end
  end

  struct MPSAPIProfile
    UNASSIGNED_PHASE = MPSAPIProfileSchema::UNASSIGNED_PHASE
    PHASE_NAMES = MPSAPIProfileSchema::PHASE_NAMES
    OPERATION_NAMES = MPSAPIProfileSchema::OPERATION_NAMES
    PHASE_COUNT = MPSAPIProfileSchema::PHASE_COUNT
    OPERATION_COUNT = MPSAPIProfileSchema::OPERATION_COUNT

    getter phases : Array(MPSAPIPhaseProfile)

    private def initialize(@phases : Array(MPSAPIPhaseProfile))
    end

    # Full-block profile-replay has a fixed backend and geometry. Keep this
    # check at that boundary instead of treating a structurally complete but
    # empty or partially instrumented matrix as evidence of API timing.
    def validate_full_block_counts!(expected_graph_calls : UInt64,
                                    expected_linear_calls : UInt64,
                                    expected_commit_calls : UInt64) : Nil
      raise ArgumentError.new("Expected full-block API call counts must be positive") unless
        expected_graph_calls > 0 && expected_linear_calls > 0 && expected_commit_calls > 0

      attention = @phases.find { |phase| phase.phase_id == 2 } ||
                  raise ArgumentError.new("MPS API profile has no self-attention phase")
      attention_graph_count = attention.operations["graph_encode"].count
      unless attention_graph_count == expected_graph_calls
        raise ArgumentError.new("MPS API profile has an unexpected self-attention Graph call count")
      end

      graph_calls = 0_u64
      linear_calls = 0_u64
      commit_calls = 0_u64
      @phases.each do |phase|
        graph_calls += phase.operations["graph_encode"].count
        linear_calls += phase.operations["t5_linear_encode"].count
        commit_calls += phase.operations["commit_and_continue"].count
      end
      unless graph_calls == expected_graph_calls && linear_calls == expected_linear_calls &&
             commit_calls == expected_commit_calls
        raise ArgumentError.new("MPS API profile call counts do not match the full-block geometry")
      end
    end

    # Require one successful snapshot for every known phase/operation pair.
    # Zero counts are valid only when every timing field is also zero.
    def self.build(samples : Array(MPSAPIProfileSample)) : self
      expected = MPSAPIProfile::PHASE_COUNT * MPSAPIProfile::OPERATION_COUNT
      unless samples.size == expected
        raise ArgumentError.new("MPS API profile is missing phase/operation snapshots")
      end

      indexed = {} of Tuple(Int32, Int32) => MPSAPITimingStats
      samples.each do |sample|
        unless MPSAPIProfile::PHASE_NAMES.has_key?(sample.phase_id) &&
               MPSAPIProfile::OPERATION_NAMES.has_key?(sample.operation_id)
          raise ArgumentError.new("MPS API profile contains an unknown phase or operation")
        end
        key = {sample.phase_id, sample.operation_id}
        raise ArgumentError.new("MPS API profile contains a duplicate snapshot") if indexed.has_key?(key)
        sample.stats.validate!(sample.operation_id)
        indexed[key] = sample.stats
      end

      phases = Array(MPSAPIPhaseProfile).new(MPSAPIProfile::PHASE_COUNT)
      MPSAPIProfile::PHASE_NAMES.keys.sort.each do |phase_id|
        operations = {} of String => MPSAPITimingStats
        MPSAPIProfile::OPERATION_NAMES.keys.sort.each do |operation_id|
          stats = indexed[{phase_id, operation_id}]? ||
                  raise ArgumentError.new("MPS API profile is missing a phase/operation snapshot")
          operations[MPSAPIProfile::OPERATION_NAMES[operation_id]] = stats
        end
        graph = operations["graph_encode"]
        commit = operations["commit_and_continue"]
        unless graph.nested_count == commit.nested_count &&
               graph.nested_wall_total_ns == commit.nested_wall_total_ns &&
               graph.nested_thread_cpu_total_ns == commit.nested_thread_cpu_total_ns
          raise ArgumentError.new("MPSGraph and commit nested snapshots disagree")
        end
        phases << MPSAPIPhaseProfile.new(phase_id,
          MPSAPIProfile::PHASE_NAMES[phase_id], operations)
      end
      unless samples.any? { |sample| sample.stats.count > 0 }
        raise ArgumentError.new("MPS API profile contains no measured API calls")
      end
      new(phases)
    end
  end
end
