require "set"

# Pure CPU validation and interval accounting for asynchronous MPS command roots.
# The numbers describe overlapping command-buffer GPU intervals, not additive
# stage costs. Phase ownership is by contiguous root sequence, while temporal
# intervals may overlap across phases.
module ML::Video
  struct MPSCommandRootTiming
    getter sequence : Int32
    getter completed : Bool
    getter has_error : Bool
    getter gpu_start_seconds : Float64
    getter gpu_end_seconds : Float64

    def initialize(@sequence : Int32, @completed : Bool, @has_error : Bool,
                   @gpu_start_seconds : Float64, @gpu_end_seconds : Float64)
    end
  end

  # Half-open root sequence range [first_root, end_root).
  struct MPSCommandPhaseRange
    getter name : String
    getter first_root : Int32
    getter end_root : Int32

    def initialize(@name : String, @first_root : Int32, @end_root : Int32)
    end
  end

  struct MPSCommandPhaseProfile
    # Bound covers Float64 subtraction and sequential summation of the same
    # root durations used by the native aggregate. It scales with root count.
    private FLOAT64_EPSILON = 2.2204460492503131e-16

    getter root_count : Int32
    getter root_duration_seconds : Float64
    getter envelope_seconds : Float64
    getter union_seconds : Float64
    getter overlap_seconds : Float64
    getter phase_union_seconds : Hash(String, Float64)

    private def initialize(@root_count : Int32,
                           @root_duration_seconds : Float64,
                           @envelope_seconds : Float64,
                           @union_seconds : Float64,
                           @overlap_seconds : Float64,
                           @phase_union_seconds : Hash(String, Float64))
    end

    def self.build(roots : Array(MPSCommandRootTiming),
                   phases : Array(MPSCommandPhaseRange),
                   native_duration_sum_seconds : Float64) : self
      raise ArgumentError.new("MPS phase profile requires command roots") if roots.empty?
      raise ArgumentError.new("MPS phase profile requires phase ranges") if phases.empty?
      unless native_duration_sum_seconds.finite? && native_duration_sum_seconds > 0.0
        raise ArgumentError.new("MPS native GPU duration sum is invalid")
      end

      intervals = Array(Tuple(Float64, Float64)).new(roots.size)
      root_duration_sum = 0.0
      roots.each_with_index do |root, index|
        unless root.sequence == index
          raise ArgumentError.new("MPS root summaries are missing, duplicated, or out of sequence")
        end
        raise ArgumentError.new("MPS root #{index} did not complete") unless root.completed
        raise ArgumentError.new("MPS root #{index} reported an error") if root.has_error
        unless root.gpu_start_seconds.finite? && root.gpu_end_seconds.finite? &&
               root.gpu_start_seconds >= 0.0 && root.gpu_end_seconds > root.gpu_start_seconds
          raise ArgumentError.new("MPS root #{index} has invalid GPU timestamps")
        end

        duration = root.gpu_end_seconds - root.gpu_start_seconds
        root_duration_sum += duration
        intervals << {root.gpu_start_seconds, root.gpu_end_seconds}
      end

      raise ArgumentError.new("MPS root-duration sum is invalid") unless root_duration_sum.finite? && root_duration_sum > 0.0
      allowed_rounding = FLOAT64_EPSILON * 8.0 * (roots.size + 1).to_f64 *
                         {1.0, root_duration_sum.abs, native_duration_sum_seconds.abs}.max
      if (root_duration_sum - native_duration_sum_seconds).abs > allowed_rounding
        raise ArgumentError.new("MPS root-duration sum differs from native aggregate")
      end

      cursor = 0
      phase_names = Set(String).new
      phase_unions = Hash(String, Float64).new
      phases.each do |phase|
        if phase.name.empty? || phase_names.includes?(phase.name)
          raise ArgumentError.new("MPS phase names must be non-empty and unique")
        end
        phase_names << phase.name
        unless phase.first_root == cursor && phase.end_root > phase.first_root &&
               phase.end_root <= roots.size
          raise ArgumentError.new("MPS phase ranges must cover each root exactly once in sequence")
        end
        phase_unions[phase.name] = interval_union(intervals[phase.first_root...phase.end_root])
        cursor = phase.end_root
      end
      unless cursor == roots.size
        raise ArgumentError.new("MPS phase ranges do not cover every command root")
      end

      root_union = interval_union(intervals)
      envelope_start = roots.min_of(&.gpu_start_seconds)
      envelope_end = roots.max_of(&.gpu_end_seconds)
      envelope = envelope_end - envelope_start
      overlap = root_duration_sum - root_union
      if overlap < -allowed_rounding
        raise ArgumentError.new("MPS interval union exceeds summed root durations")
      end
      overlap = 0.0 if overlap < 0.0

      new(roots.size.to_i32, root_duration_sum, envelope, root_union,
        overlap, phase_unions)
    end

    private def self.interval_union(intervals : Array(Tuple(Float64, Float64))) : Float64
      raise ArgumentError.new("MPS interval union requires intervals") if intervals.empty?
      ordered = intervals.sort_by(&.[0])
      first_start, current_end = ordered.first
      total = 0.0
      ordered.skip(1).each do |start_time, end_time|
        if start_time <= current_end
          current_end = end_time if end_time > current_end
        else
          total += current_end - first_start
          first_start = start_time
          current_end = end_time
        end
      end
      total + (current_end - first_start)
    end
  end
end
