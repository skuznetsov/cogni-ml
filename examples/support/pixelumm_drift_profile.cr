require "json"
require "../../src/ml/video/pixelumm_block_profile"
require "../../src/ml/video/mps_continuation_trace"

# CPU-only interpretation of four sequential PixelUMM block plans. Marker
# roots remain candidates on both sides of each phase and plan boundary;
# they are not exclusive per-phase GPU timings.
module PixelUMMDriftProfile
  alias Marker = ML::Video::PixelUMMBlockPhaseMarker

  class Profile
    PLAN_LABELS = [
      "window0/block0", "window0/block1",
      "window1/block0", "window1/block1",
    ]
    PHASE_NAMES              = ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES.to_a[0...-1]
    API_PHASE_PIXELUMM_NAMES = {
      0 => "unassigned",
      1 => "self_prepare",
      2 => "local_attention",
      3 => "projection_residual",
      4 => "unused_cross_attention",
      5 => "ffn",
    }

    getter markers : Array(Array(Marker))
    getter roots : Array(ML::Video::MPSCommandRootTiming)
    getter phase_wall_seconds : Hash(String, Float64)
    getter phase_thread_cpu_seconds : Hash(String, Float64)
    getter root_candidates : Array(Array(String))
    getter phase_candidate_root_duration_sum_seconds : Hash(String, Float64)
    getter phase_definite_root_duration_sum_seconds : Hash(String, Float64)
    getter root_profile : ML::Video::MPSCommandPhaseProfile
    getter api_profile : ML::Video::MPSAPIProfile
    getter trace : ML::Video::MPSContinuationTrace

    def initialize(@markers : Array(Array(Marker)),
                   @roots : Array(ML::Video::MPSCommandRootTiming),
                   native_duration_sum : Float64,
                   @api_profile : ML::Video::MPSAPIProfile,
                   @trace : ML::Video::MPSContinuationTrace)
      validate_markers!
      validate_api_and_trace!

      @phase_wall_seconds = {} of String => Float64
      @phase_thread_cpu_seconds = {} of String => Float64
      PHASE_NAMES.each do |phase|
        PLAN_LABELS.each do |plan|
          label = "#{plan}/#{phase}"
          @phase_wall_seconds[label] = 0.0
          @phase_thread_cpu_seconds[label] = 0.0
        end
      end

      @root_candidates = Array(Array(String)).new(@roots.size) { [] of String }
      @markers.each_with_index do |plan_markers, plan_index|
        plan_markers.each_cons_pair do |start_marker, end_marker|
          label = "#{PLAN_LABELS[plan_index]}/#{start_marker.name}"
          @phase_wall_seconds[label] = end_marker.elapsed_seconds - start_marker.elapsed_seconds
          @phase_thread_cpu_seconds[label] = end_marker.thread_cpu_seconds - start_marker.thread_cpu_seconds
          ((start_marker.root_count - 1)..(end_marker.root_count - 1)).each do |root_index|
            @root_candidates[root_index] << label
          end
        end
      end

      # The first open root can also contain poison/metric setup. The final
      # marker still names the open root, where post-body comparator dispatches
      # can begin; any continuation roots after it are also comparator candidates.
      @root_candidates.first << "request_prelude"
      final_comparator_start = @markers.last.last.root_count - 1
      (final_comparator_start...@roots.size).each do |root_index|
        @root_candidates[root_index] << "request_final_comparators"
      end
      if @root_candidates.any?(&.empty?)
        raise ArgumentError.new("PixelUMM drift profile leaves a command root without a candidate")
      end

      @phase_candidate_root_duration_sum_seconds = {} of String => Float64
      @phase_definite_root_duration_sum_seconds = {} of String => Float64
      all_phase_labels = phase_labels
      phase_label_set = {} of String => Bool
      all_phase_labels.each do |label|
        phase_label_set[label] = true
        @phase_candidate_root_duration_sum_seconds[label] = 0.0
        @phase_definite_root_duration_sum_seconds[label] = 0.0
      end
      @roots.each_with_index do |root, index|
        duration = root.gpu_end_seconds - root.gpu_start_seconds
        candidates = @root_candidates[index]
        candidates.each do |label|
          if phase_label_set.has_key?(label)
            @phase_candidate_root_duration_sum_seconds[label] += duration
          end
        end
        phase_candidates = candidates.select { |label| phase_label_set.has_key?(label) }
        if phase_candidates.size == 1
          @phase_definite_root_duration_sum_seconds[phase_candidates.first] += duration
        end
      end

      @root_profile = ML::Video::MPSCommandPhaseProfile.build(@roots,
        [ML::Video::MPSCommandPhaseRange.new("whole_chain", 0, @roots.size.to_i32)],
        native_duration_sum)
    end

    private def phase_labels : Array(String)
      PHASE_NAMES.flat_map do |phase|
        PLAN_LABELS.map { |plan| "#{plan}/#{phase}" }
      end
    end

    private def validate_markers! : Nil
      unless @markers.size == PLAN_LABELS.size
        raise ArgumentError.new("PixelUMM drift profile requires exactly four block plans")
      end
      raise ArgumentError.new("PixelUMM drift profile requires command roots") if @roots.empty?

      expected_names = ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES.to_a
      previous_plan_root = 1
      @markers.each_with_index do |plan_markers, plan_index|
        unless plan_markers.size == expected_names.size && plan_markers.map(&.name) == expected_names
          raise ArgumentError.new("PixelUMM drift plan #{plan_index} has incomplete or reordered markers")
        end
        unless plan_markers.first.elapsed_seconds == 0.0 &&
               plan_markers.first.thread_cpu_seconds == 0.0 &&
               plan_markers.first.root_count == previous_plan_root
          raise ArgumentError.new("PixelUMM drift plan #{plan_index} does not start at zero clocks and the adjacent root cursor")
        end

        previous_wall = 0.0
        previous_cpu = 0.0
        previous_root = previous_plan_root
        plan_markers.each do |marker|
          unless marker.elapsed_seconds.finite? && marker.elapsed_seconds >= previous_wall &&
                 marker.thread_cpu_seconds.finite? && marker.thread_cpu_seconds >= previous_cpu &&
                 marker.root_count >= previous_root && marker.root_count >= 1 &&
                 marker.root_count <= @roots.size
            raise ArgumentError.new("PixelUMM drift plan #{plan_index} has invalid marker clocks or root cursors")
          end
          previous_wall = marker.elapsed_seconds
          previous_cpu = marker.thread_cpu_seconds
          previous_root = marker.root_count
        end
        previous_plan_root = plan_markers.last.root_count
      end
    end

    private def validate_api_and_trace! : Nil
      unless @trace.roots.size == @roots.size &&
             @trace.info.root_count == @roots.size.to_u64 &&
             @trace.events.size == @roots.size - 1
        raise ArgumentError.new("PixelUMM drift API/continuation trace does not cover the complete chain")
      end
      @roots.each_with_index do |root, index|
        traced = @trace.roots[index]
        unless traced.sequence == index &&
               (traced.gpu_start_seconds - root.gpu_start_seconds).abs <= ML::Video::MPSContinuationTrace::CLOCK_ROUNDING_SECONDS &&
               (traced.gpu_end_seconds - root.gpu_end_seconds).abs <= ML::Video::MPSContinuationTrace::CLOCK_ROUNDING_SECONDS
          raise ArgumentError.new("PixelUMM drift root timings disagree with the continuation trace at sequence #{index}")
        end
      end

      graph_counts = {} of Int32 => UInt64
      commit_counts = {} of Int32 => UInt64
      linear_counts = {} of Int32 => UInt64
      @api_profile.phases.each do |phase|
        graph_counts[phase.phase_id] = phase.operations["graph_encode"].count
        commit_counts[phase.phase_id] = phase.operations["commit_and_continue"].count
        linear_counts[phase.phase_id] = phase.operations["t5_linear_encode"].count
      end
      attention_graph_count = graph_counts[2]? ||
                              raise ArgumentError.new("PixelUMM drift API profile is missing self-attention Graph counts")

      unless graph_counts.size == API_PHASE_PIXELUMM_NAMES.size &&
             graph_counts[0]? == 0_u64 && graph_counts[1]? == 36_u64 &&
             attention_graph_count > 0 && attention_graph_count % 12_u64 == 0 &&
             graph_counts[3]? == 12_u64 && graph_counts[4]? == 0_u64 &&
             graph_counts[5]? == 12_u64 &&
             graph_counts.values.sum == 60_u64 + attention_graph_count &&
             commit_counts[0]? == 0_u64 && commit_counts[4]? == 0_u64 &&
             linear_counts.values.all?(&.zero?)
        raise ArgumentError.new("PixelUMM drift API Graph/linear counts do not match four separate-gate/up block plans")
      end

      @trace.validate_api_profile!(@api_profile)
    end

    private def write_markers(json : JSON::Builder,
                              markers : Array(Marker)) : Nil
      json.array do
        markers.each do |marker|
          json.object do
            json.field "name", marker.name
            json.field "elapsed_seconds", marker.elapsed_seconds
            json.field "thread_cpu_seconds", marker.thread_cpu_seconds
            json.field "root_count", marker.root_count
          end
        end
      end
    end

    private def write_timing_values(json : JSON::Builder,
                                    values : Hash(String, Float64)) : Nil
      json.object do
        phase_labels.each do |label|
          json.field label, values[label]
        end
      end
    end

    private def write_api_profile(json : JSON::Builder) : Nil
      json.object do
        json.field "source_schema", "MPSAPIProfile WAN phase IDs"
        json.field "phases" do
          json.array do
            @api_profile.phases.each do |phase|
              json.object do
                json.field "source_phase_id", phase.phase_id
                json.field "source_phase_name", phase.name
                json.field "pixelumm_phase", API_PHASE_PIXELUMM_NAMES[phase.phase_id]
                json.field "operations" do
                  json.object do
                    ML::Video::MPSAPIProfile::OPERATION_NAMES.keys.sort.each do |operation_id|
                      name = ML::Video::MPSAPIProfile::OPERATION_NAMES[operation_id]
                      stats = phase.operations[name]
                      json.field name do
                        json.object do
                          json.field "count", stats.count
                          json.field "wall_total_ns", stats.wall_total_ns
                          json.field "wall_max_ns", stats.wall_max_ns
                          json.field "thread_cpu_total_ns", stats.thread_cpu_total_ns
                          json.field "thread_cpu_max_ns", stats.thread_cpu_max_ns
                          json.field "nested_count", stats.nested_count
                          json.field "nested_wall_total_ns", stats.nested_wall_total_ns
                          json.field "nested_thread_cpu_total_ns", stats.nested_thread_cpu_total_ns
                          json.field "exclusive_wall_total_ns", stats.exclusive_wall_total_ns
                          json.field "exclusive_thread_cpu_total_ns", stats.exclusive_thread_cpu_total_ns
                        end
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
    end

    private def write_root_profile(json : JSON::Builder) : Nil
      json.object do
        json.field "root_count", @root_profile.root_count
        json.field "root_duration_seconds", @root_profile.root_duration_seconds
        json.field "envelope_seconds", @root_profile.envelope_seconds
        json.field "union_seconds", @root_profile.union_seconds
        json.field "overlap_seconds", @root_profile.overlap_seconds
        json.field "gap_seconds", {@root_profile.envelope_seconds - @root_profile.union_seconds, 0.0}.max
        json.field "phase_union_seconds" do
          @root_profile.phase_union_seconds.to_json(json)
        end
      end
    end

    def to_json(json : JSON::Builder) : Nil
      json.object do
        json.field "schema", "cogni_ml.pixelumm_drift_profile/v1"
        json.field "markers_by_plan" do
          json.array do
            @markers.each_with_index do |plan_markers, index|
              json.object do
                json.field "plan", PLAN_LABELS[index]
                json.field "markers" do
                  write_markers(json, plan_markers)
                end
              end
            end
          end
        end
        json.field "phase_wall_seconds" do
          write_timing_values(json, @phase_wall_seconds)
        end
        json.field "phase_thread_cpu_seconds" do
          write_timing_values(json, @phase_thread_cpu_seconds)
        end
        json.field "phase_candidate_root_duration_sum_seconds" do
          write_timing_values(json, @phase_candidate_root_duration_sum_seconds)
        end
        json.field "phase_definite_root_duration_sum_seconds" do
          write_timing_values(json, @phase_definite_root_duration_sum_seconds)
        end
        json.field "root_candidate_duration_semantics",
          "Candidate sums can count a root for multiple phases. Definite sums use roots with one marker candidate but are still root GPU durations, not exclusive phase work; root intervals may overlap."
        json.field "gpu_phase_interpretation",
          "Marker phases identify CPU-side encode spans. Candidate roots can straddle phases and GPU roots can overlap; no per-phase value is exclusive GPU work."
        json.field "roots" do
          json.array do
            @roots.each_with_index do |root, index|
              json.object do
                json.field "sequence", root.sequence
                json.field "start_seconds", root.gpu_start_seconds
                json.field "end_seconds", root.gpu_end_seconds
                json.field "candidates", @root_candidates[index]
              end
            end
          end
        end
        json.field "root_profile" do
          write_root_profile(json)
        end
        json.field "api_profile" do
          write_api_profile(json)
        end
        json.field "continuation_trace" do
          json.object do
            json.field "info", @trace.info
            json.field "events", @trace.events
            json.field "roots", @trace.roots
          end
        end
      end
    end
  end
end
