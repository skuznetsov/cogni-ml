require "./mps_command_phase_profile"

module ML::Video
  record HyNomasBlockPhaseMarker,
    name : String,
    elapsed_seconds : Float64,
    root_count : Int32

  # CPU-only interpretation of optional start markers on one fresh command
  # chain containing exactly one block. The current root can include both
  # sides of a marker: candidate phase names are conservative attribution,
  # never an additive breakdown of GPU kernel costs.
  struct HyNomasBlockPhaseProfile
    MARKER_NAMES = {"validate", "modulation", "qkv", "joint_pack", "attention",
                    "attention_projection_residual", "image_ffn", "text_ffn", "complete"}

    getter phase_cpu_seconds : Hash(String, Float64)
    getter root_candidates : Array(Array(String))
    getter root_profile : MPSCommandPhaseProfile

    private def initialize(@phase_cpu_seconds : Hash(String, Float64),
                           @root_candidates : Array(Array(String)),
                           @root_profile : MPSCommandPhaseProfile)
    end

    def self.build(markers : Array(HyNomasBlockPhaseMarker),
                   roots : Array(MPSCommandRootTiming),
                   native_duration_sum_seconds : Float64) : self
      unless markers.map(&.name) == MARKER_NAMES.to_a
        raise ArgumentError.new("Hy-Nomas phase markers must have the complete source order")
      end
      raise ArgumentError.new("Hy-Nomas phase profile requires command roots") if roots.empty?
      previous_time = 0.0
      previous_count = 1
      markers.each do |marker|
        unless marker.elapsed_seconds.finite? && marker.elapsed_seconds >= previous_time &&
               marker.root_count >= previous_count && marker.root_count <= roots.size
          raise ArgumentError.new("Hy-Nomas phase markers have invalid time or root cursors")
        end
        previous_time = marker.elapsed_seconds
        previous_count = marker.root_count
      end
      unless markers.first.root_count == 1 && markers[1].root_count == 1 &&
             markers.last.root_count == roots.size
        raise ArgumentError.new("Hy-Nomas phase profile requires a fresh chain and CPU-only validation")
      end

      cpu = {} of String => Float64
      candidates = Array(Array(String)).new(roots.size) { [] of String }
      markers.each_cons_pair do |start_marker, end_marker|
        name = start_marker.name
        cpu[name] = end_marker.elapsed_seconds - start_marker.elapsed_seconds
        # Validation appends no GPU work; complete is only an end sentinel.
        # Merely observing the pre-existing root cannot assign it to validation.
        next if name == "validate"
        ((start_marker.root_count - 1)..(end_marker.root_count - 1)).each do |root|
          candidates[root] << name
        end
      end
      if candidates.any?(&.empty?)
        raise ArgumentError.new("Hy-Nomas phase markers leave an unattributed command root")
      end
      whole = MPSCommandPhaseProfile.build(roots,
        [MPSCommandPhaseRange.new("whole_block", 0, roots.size.to_i32)], native_duration_sum_seconds)
      new(cpu, candidates, whole)
    end
  end
end
