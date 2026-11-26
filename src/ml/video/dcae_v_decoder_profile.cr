require "set"
require "./metal_mps_command_chain"

module ML::Video
  record DCAEVDecoderOperationTiming,
    name : String,
    category : String,
    stage : String,
    encode_wall_seconds : Float64,
    start_root_count : Int32,
    end_root_count : Int32

  # CPU-only interpretation of DCAE-V encode markers and existing MPS command
  # root timestamps. A marker's endpoint roots are candidates because a root
  # may contain work from either side. Candidate sums are upper bounds on
  # assignable root duration and can double-count shared roots; definite sums
  # include only roots whose marker candidates all belong to one operation or
  # group. Neither is an exclusive kernel-time measurement.
  struct DCAEVDecoderProfile
    getter operations : Array(DCAEVDecoderOperationTiming)
    getter root_candidates : Array(Array(String))
    getter operation_candidate_root_duration_sum_seconds : Hash(String, Float64)
    getter operation_definite_root_duration_sum_seconds : Hash(String, Float64)
    getter group_candidate_root_duration_sum_seconds : Hash(String, Float64)
    getter group_definite_root_duration_sum_seconds : Hash(String, Float64)
    getter roots : Array(MPSCommandRootTiming)
    getter root_profile : MPSCommandPhaseProfile
    getter api_profile : MPSAPIProfile?
    getter api_profile_unassigned : MPSAPIPhaseProfile?

    private def initialize(@operations : Array(DCAEVDecoderOperationTiming),
                           @root_candidates : Array(Array(String)),
                           @operation_candidate_root_duration_sum_seconds : Hash(String, Float64),
                           @operation_definite_root_duration_sum_seconds : Hash(String, Float64),
                           @group_candidate_root_duration_sum_seconds : Hash(String, Float64),
                           @group_definite_root_duration_sum_seconds : Hash(String, Float64),
                           @roots : Array(MPSCommandRootTiming),
                           @root_profile : MPSCommandPhaseProfile,
                           @api_profile : MPSAPIProfile?,
                           @api_profile_unassigned : MPSAPIPhaseProfile?)
    end

    def self.build(operations : Array(DCAEVDecoderOperationTiming),
                   roots : Array(MPSCommandRootTiming),
                   native_duration_sum_seconds : Float64,
                   api_profile : MPSAPIProfile? = nil) : self
      raise ArgumentError.new("DCAE-V decoder profile requires operations") if operations.empty?
      raise ArgumentError.new("DCAE-V decoder profile requires command roots") if roots.empty?

      names = Set(String).new
      previous_root_count = 1
      operations.each do |operation|
        unless !operation.name.empty? && !operation.category.empty? && !operation.stage.empty? &&
               !names.includes?(operation.name)
          raise ArgumentError.new("DCAE-V decoder profile operation names and labels must be non-empty and unique")
        end
        names << operation.name
        unless operation.encode_wall_seconds.finite? && operation.encode_wall_seconds >= 0.0 &&
               operation.start_root_count == previous_root_count &&
               operation.end_root_count >= operation.start_root_count &&
               operation.end_root_count <= roots.size
          raise ArgumentError.new("DCAE-V decoder profile operation clocks or root cursors are invalid")
        end
        previous_root_count = operation.end_root_count
      end
      unless operations.first.start_root_count == 1 && operations.last.end_root_count == roots.size.to_i32
        raise ArgumentError.new("DCAE-V decoder profile markers must cover the fresh chain through its terminal root")
      end

      whole_chain = MPSCommandPhaseProfile.build(roots,
        [MPSCommandPhaseRange.new("whole_chain", 0, roots.size.to_i32)],
        native_duration_sum_seconds)

      operation_names = operations.map(&.name)
      operation_groups = operations.to_h do |operation|
        {operation.name, "#{operation.stage}/#{operation.category}"}
      end
      root_candidates = Array(Array(String)).new(roots.size) { [] of String }
      operations.each do |operation|
        first_root = operation.start_root_count - 1
        last_root = operation.end_root_count - 1
        unless 0 <= first_root <= last_root < roots.size
          raise ArgumentError.new("DCAE-V decoder profile operation names a root outside the completed chain")
        end
        (first_root..last_root).each do |index|
          root_candidates[index] << operation.name
        end
      end
      if root_candidates.any?(&.empty?)
        raise ArgumentError.new("DCAE-V decoder profile leaves a command root without an operation candidate")
      end

      operation_candidate = operation_names.to_h { |name| {name, 0.0} }
      operation_definite = operation_names.to_h { |name| {name, 0.0} }
      groups = operation_groups.values.uniq
      group_candidate = groups.to_h { |group| {group, 0.0} }
      group_definite = groups.to_h { |group| {group, 0.0} }
      roots.each_with_index do |root, index|
        duration = root.gpu_end_seconds - root.gpu_start_seconds
        candidates = root_candidates[index]
        candidates.each { |name| operation_candidate[name] += duration }
        operation_definite[candidates.first] += duration if candidates.size == 1

        group_candidates = candidates.map { |name| operation_groups[name] }.uniq
        group_candidates.each { |group| group_candidate[group] += duration }
        group_definite[group_candidates.first] += duration if group_candidates.size == 1
      end

      api_unassigned = api_profile.try do |profile|
        profile.phases.find { |phase| phase.phase_id == MPSAPIProfile::UNASSIGNED_PHASE } ||
          raise ArgumentError.new("DCAE-V decoder MPS API profile is missing its unassigned phase")
      end

      new(operations, root_candidates, operation_candidate, operation_definite,
        group_candidate, group_definite, roots, whole_chain, api_profile,
        api_unassigned)
    end
  end

  # Mutable encode-time observer used only when the caller opts in. Root
  # cursors observe the current chain but do not pause, continue, submit, wait,
  # or read buffers. GPU timestamps are consumed only by #build after the
  # caller's existing terminal wait.
  class DCAEVDecoderProfileRecorder
    @operations = [] of DCAEVDecoderOperationTiming
    @chain : ML::Metal::MPSCommandChain? = nil

    def measure(chain : ML::Metal::MPSCommandChain,
                name : String, category : String, stage : String,
                &operation : -> Nil) : Nil
      if previous = @chain
        unless previous.same?(chain)
          raise ArgumentError.new("DCAE-V decoder profile cannot span command chains")
        end
      else
        @chain = chain
      end

      start_root_count = chain.observed_root_count
      started = Time.instant
      operation.call
      encode_wall_seconds = (Time.instant - started).total_seconds
      end_root_count = chain.observed_root_count
      @operations << DCAEVDecoderOperationTiming.new(name, category, stage,
        encode_wall_seconds, start_root_count, end_root_count)
    end

    def build(chain : ML::Metal::MPSCommandChain) : DCAEVDecoderProfile
      unless @chain.try(&.same?(chain)) && chain.completed? && chain.wait_count == 1
        raise ArgumentError.new("DCAE-V decoder profile requires its chain's existing terminal wait")
      end
      api_profile = chain.api_profile_enabled? ? chain.api_profile : nil
      DCAEVDecoderProfile.build(@operations, chain.completed_root_timings,
        chain.gpu_elapsed_seconds, api_profile)
    end
  end
end
