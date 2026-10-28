require "json"
require "./mps_api_profile"
require "./mps_command_phase_profile"

module ML::Video
  record PixelUMMBlockPhaseMarker,
    name : String,
    elapsed_seconds : Float64,
    thread_cpu_seconds : Float64,
    root_count : Int32

  # CPU-only interpretation of native phase markers for one PixelUMM decoder
  # block. A marker names the start of a phase; `complete` is the terminal
  # sentinel. GPU root candidates include both roots adjacent to each marker,
  # since an already-open root may contain work from both phases.
  struct PixelUMMBlockPhaseProfile
    MARKER_NAMES = {"input_norm", "qkv_projection", "qk_norm_rope", "attention",
                    "output_projection", "residual_post_norm", "ffn_gate_up",
                    "ffn_activation", "ffn_down", "final_residual", "complete"}

    getter markers : Array(PixelUMMBlockPhaseMarker)
    getter phase_wall_seconds : Hash(String, Float64)
    getter phase_thread_cpu_seconds : Hash(String, Float64)
    # For each root, list all phases whose marker interval can include it.
    getter root_candidates : Array(Array(String))
    # These are per-phase sums of root timestamp durations, not additive GPU
    # work. Candidate sums double-count roots shared across a boundary; even
    # unique-root sums can overlap in GPU time with roots assigned elsewhere.
    getter phase_candidate_root_duration_sum_seconds : Hash(String, Float64)
    getter phase_definite_root_duration_sum_seconds : Hash(String, Float64)
    getter roots : Array(MPSCommandRootTiming)
    getter tile_count : Int32
    getter shared_gate_up : Bool
    getter fused_softmax_pv : Bool
    getter check_softmax_pv_f32 : Bool
    # Whole-block root totals, interval union, envelope, and overlap remain the
    # additive-safe summary. Individual roots may overlap temporally.
    getter root_profile : MPSCommandPhaseProfile
    getter api_profile : MPSAPIProfile

    private def initialize(@markers : Array(PixelUMMBlockPhaseMarker),
                           @phase_wall_seconds : Hash(String, Float64),
                           @phase_thread_cpu_seconds : Hash(String, Float64),
                           @root_candidates : Array(Array(String)),
                           @phase_candidate_root_duration_sum_seconds : Hash(String, Float64),
                           @phase_definite_root_duration_sum_seconds : Hash(String, Float64),
                           @roots : Array(MPSCommandRootTiming),
                           @tile_count : Int32,
                           @root_profile : MPSCommandPhaseProfile,
                           @api_profile : MPSAPIProfile,
                           @shared_gate_up : Bool,
                           @fused_softmax_pv : Bool,
                           @check_softmax_pv_f32 : Bool)
    end

    def self.build(markers : Array(PixelUMMBlockPhaseMarker),
                   roots : Array(MPSCommandRootTiming),
                   native_duration_sum_seconds : Float64,
                   api_profile : MPSAPIProfile,
                   tile_count : Int32,
                   shared_gate_up : Bool = false,
                   fused_softmax_pv : Bool = false,
                   check_softmax_pv_f32 : Bool = false) : self
      unless markers.map(&.name) == MARKER_NAMES.to_a
        raise ArgumentError.new("PixelUMM phase markers must have the complete source order")
      end
      raise ArgumentError.new("PixelUMM phase profile requires command roots") if roots.empty?
      raise ArgumentError.new("PixelUMM attention tile count must be positive") unless tile_count > 0
      if check_softmax_pv_f32 && shared_gate_up
        raise ArgumentError.new("PixelUMM direct F32 softmax/PV check cannot use shared gate/up")
      end

      previous_wall = 0.0
      previous_thread_cpu = 0.0
      previous_root_count = 1
      markers.each do |marker|
        unless marker.elapsed_seconds.finite? && marker.elapsed_seconds >= 0.0 &&
               marker.elapsed_seconds >= previous_wall &&
               marker.thread_cpu_seconds.finite? && marker.thread_cpu_seconds >= 0.0 &&
               marker.thread_cpu_seconds >= previous_thread_cpu &&
               marker.root_count >= previous_root_count &&
               marker.root_count >= 1 && marker.root_count <= roots.size
          raise ArgumentError.new("PixelUMM phase markers have invalid time or root cursors")
        end
        previous_wall = marker.elapsed_seconds
        previous_thread_cpu = marker.thread_cpu_seconds
        previous_root_count = marker.root_count
      end
      unless markers.first.elapsed_seconds == 0.0 &&
             markers.first.thread_cpu_seconds == 0.0 &&
             markers.first.root_count == 1 &&
             markers.last.root_count == roots.size
        raise ArgumentError.new("PixelUMM phase markers require a fresh root and terminal cursor")
      end

      validate_api_profile!(api_profile, roots.size, tile_count, shared_gate_up,
        fused_softmax_pv, check_softmax_pv_f32)

      phase_names = MARKER_NAMES.to_a[0...-1]
      wall_seconds = {} of String => Float64
      thread_cpu_seconds = {} of String => Float64
      phase_names.each do |name|
        wall_seconds[name] = 0.0
        thread_cpu_seconds[name] = 0.0
      end

      root_candidates = Array(Array(String)).new(roots.size) { [] of String }
      markers.each_cons_pair do |start_marker, end_marker|
        phase_name = start_marker.name
        wall_seconds[phase_name] = end_marker.elapsed_seconds - start_marker.elapsed_seconds
        thread_cpu_seconds[phase_name] = end_marker.thread_cpu_seconds - start_marker.thread_cpu_seconds
        # Both endpoint roots remain candidates. MPS command roots can carry
        # work from either side of a CPU marker; this does not split GPU work.
        ((start_marker.root_count - 1)..(end_marker.root_count - 1)).each do |root_index|
          root_candidates[root_index] << phase_name
        end
      end
      if root_candidates.any?(&.empty?)
        raise ArgumentError.new("PixelUMM phase markers leave an unattributed command root")
      end

      candidate_duration_sums = {} of String => Float64
      definite_duration_sums = {} of String => Float64
      phase_names.each do |name|
        candidate_duration_sums[name] = 0.0
        definite_duration_sums[name] = 0.0
      end
      roots.each_with_index do |root, index|
        duration = root.gpu_end_seconds - root.gpu_start_seconds
        candidates = root_candidates[index]
        candidates.each do |phase_name|
          candidate_duration_sums[phase_name] += duration
        end
        if candidates.size == 1
          definite_duration_sums[candidates.first] += duration
        end
      end

      whole_block = MPSCommandPhaseProfile.build(roots,
        [MPSCommandPhaseRange.new("whole_block", 0, roots.size.to_i32)],
        native_duration_sum_seconds)

      new(markers, wall_seconds, thread_cpu_seconds, root_candidates,
        candidate_duration_sums, definite_duration_sums, roots, tile_count,
        whole_block, api_profile, shared_gate_up, fused_softmax_pv,
        check_softmax_pv_f32)
    end

    def attention_graph_topology : String
      return "f32_softmax_pv_dual_path_check" if @check_softmax_pv_f32
      @fused_softmax_pv ? "separate_qk_fused_softmax_pv" : "separate_qk_separate_softmax_pv"
    end

    # Keep the shared WAN API phase IDs visible as source coordinates while
    # labeling their PixelUMM interpretation explicitly in each serialized row.
    def to_json(json : JSON::Builder) : Nil
      json.object do
        json.field "schema", "cogni_ml.pixelumm_block_phase_profile/v1"
        json.field "attention_tile_count", @tile_count
        json.field "fused_softmax_pv", @fused_softmax_pv
        json.field "check_softmax_pv_f32", @check_softmax_pv_f32
        json.field "timing_eligible", !@check_softmax_pv_f32
        json.field "attention_graph_topology", attention_graph_topology
        json.field "ffn_graph_topology",
          @shared_gate_up ? "shared_f32_gate_up" : "separate_gate_up"
        json.field "markers" do
          json.array do
            @markers.each do |marker|
              json.object do
                json.field "name", marker.name
                json.field "elapsed_seconds", marker.elapsed_seconds
                json.field "thread_cpu_seconds", marker.thread_cpu_seconds
                json.field "root_count", marker.root_count
              end
            end
          end
        end
        json.field "phase_wall_seconds" do
          write_phase_values(json, @phase_wall_seconds)
        end
        json.field "phase_thread_cpu_seconds" do
          write_phase_values(json, @phase_thread_cpu_seconds)
        end
        json.field "root_candidates" do
          @root_candidates.to_json(json)
        end
        json.field "phase_candidate_root_duration_sum_seconds" do
          write_phase_values(json, @phase_candidate_root_duration_sum_seconds)
        end
        json.field "phase_definite_root_duration_sum_seconds" do
          write_phase_values(json, @phase_definite_root_duration_sum_seconds)
        end
        json.field "phase_root_duration_sum_semantics",
          "Root timestamp durations, not GPU work; candidate sums may double-count shared roots and root intervals may overlap."
        json.field "root_profile" do
          json.object do
            json.field "root_count", @root_profile.root_count
            json.field "root_duration_seconds", @root_profile.root_duration_seconds
            json.field "envelope_seconds", @root_profile.envelope_seconds
            json.field "union_seconds", @root_profile.union_seconds
            json.field "overlap_seconds", @root_profile.overlap_seconds
            json.field "phase_union_seconds" do
              @root_profile.phase_union_seconds.to_json(json)
            end
          end
        end
        json.field "roots" do
          json.array do
            @roots.each_with_index do |root, index|
              json.object do
                json.field "sequence", root.sequence
                json.field "completed", root.completed
                json.field "has_error", root.has_error
                json.field "gpu_start_seconds", root.gpu_start_seconds
                json.field "gpu_end_seconds", root.gpu_end_seconds
                json.field "duration_seconds", root.gpu_end_seconds - root.gpu_start_seconds
                json.field "candidate_phases" do
                  @root_candidates[index].to_json(json)
                end
              end
            end
          end
        end
        json.field "api_profile" do
          json.object do
            json.field "source_schema", "MPSAPIProfile WAN phase IDs"
            json.field "phases" do
              json.array do
                @api_profile.phases.each do |phase|
                  json.object do
                    json.field "pixelumm_phase", pixelumm_api_phase_name(phase.phase_id)
                    json.field "source_phase_id", phase.phase_id
                    json.field "source_phase_name", phase.name
                    json.field "operations" do
                      json.object do
                        MPSAPIProfile::OPERATION_NAMES.keys.sort.each do |operation_id|
                          operation_name = MPSAPIProfile::OPERATION_NAMES[operation_id]
                          stats = phase.operations[operation_name]
                          json.field operation_name do
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
      end
    end

    private def write_phase_values(json : JSON::Builder, values : Hash(String, Float64)) : Nil
      json.object do
        MARKER_NAMES.to_a[0...-1].each do |name|
          json.field name, values[name]
        end
      end
    end

    private def pixelumm_api_phase_name(phase_id : Int32) : String
      case phase_id
      when 0 then "unassigned"
      when 1 then "input_norm_qkv_projection_qk_norm_rope"
      when 2 then "local_attention"
      when 3 then "output_projection_residual_post_norm"
      when 4 then "unused_cross_attention"
      when 5 then "ffn"
      else
        raise ArgumentError.new("PixelUMM API profile has an unknown source phase")
      end
    end

    private def self.validate_api_profile!(profile : MPSAPIProfile,
                                           root_count : Int32,
                                           tile_count : Int32,
                                           shared_gate_up : Bool,
                                           fused_softmax_pv : Bool,
                                           check_softmax_pv_f32 : Bool) : Nil
      # Reuse the existing MPS API phase IDs without applying the Wan-specific
      # full-block validator: PixelUMM phase 2 is its two-, three-, or
      # four-per-tile local attention graph, and phase 4 is unused.
      expected_graph_calls = {
        0 => 0_u64,
        1 => 9_u64, # input norm, QKV, Q/K norm statistics and inverse
        2 => (check_softmax_pv_f32 ? 4_u64 : (fused_softmax_pv ? 2_u64 : 3_u64)) * tile_count.to_u64,
        3 => 3_u64,                            # output projection, post norm and gain
        4 => 0_u64,                            # PixelUMM has no cross-attention phase
        5 => (shared_gate_up ? 2_u64 : 3_u64), # shared/separate gate-up plus down projection
      }

      graph_calls = 0_u64
      continuation_calls = 0_u64
      linear_calls = 0_u64
      expected_graph_calls.each do |phase_id, expected_count|
        phase = profile.phases.find { |item| item.phase_id == phase_id } ||
                raise ArgumentError.new("PixelUMM API profile is missing phase #{phase_id}")
        graph = phase.operations["graph_encode"]?
        raise ArgumentError.new("PixelUMM API profile is missing Graph timing") unless graph
        unless graph.count == expected_count
          raise ArgumentError.new("PixelUMM API Graph counts do not match the block phase inventory")
        end
        graph_calls += graph.count
      end

      profile.phases.each do |phase|
        continuation = phase.operations["commit_and_continue"]?
        linear = phase.operations["t5_linear_encode"]?
        unless continuation && linear
          raise ArgumentError.new("PixelUMM API profile is missing operation timing")
        end
        continuation_calls += continuation.count
        linear_calls += linear.count
      end

      expected_total = (shared_gate_up ? 14_u64 : 15_u64) +
                       (check_softmax_pv_f32 ? 4_u64 : (fused_softmax_pv ? 2_u64 : 3_u64)) * tile_count.to_u64
      unless graph_calls == expected_total
        raise ArgumentError.new("PixelUMM API profile has an unexpected total Graph call count")
      end
      unless linear_calls == 0
        raise ArgumentError.new("PixelUMM API profile contains a T5 linear call")
      end
      unless continuation_calls == (root_count - 1).to_u64
        raise ArgumentError.new("PixelUMM API continuation count does not match command roots")
      end
    end
  end
end
