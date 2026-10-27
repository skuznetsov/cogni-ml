require "json"
require "./pixelumm_block_profile"

module ML::Video
  record PixelUMMDecoderBlockPhaseTiming,
    slot : Int32,
    tile_count : Int32,
    markers : Array(PixelUMMBlockPhaseMarker),
    phase_wall_seconds : Hash(String, Float64),
    phase_thread_cpu_seconds : Hash(String, Float64)

  # CPU-only interpretation of the block marker arrays plus the final global
  # head interval. Marker root cursors are chain-global, while each block's
  # wall/CPU clocks restart at zero. Root candidates deliberately retain the
  # open root on both sides of a marker or block boundary.
  struct PixelUMMDecoderPhaseProfile
    PHASE_NAMES              = PixelUMMBlockPhaseProfile::MARKER_NAMES.to_a[0...-1]
    API_PHASE_PIXELUMM_NAMES = {
      0 => "unassigned_head",
      1 => "self_prepare",
      2 => "local_attention",
      3 => "projection_residual",
      4 => "unused_cross_attention",
      5 => "ffn",
    }

    getter blocks : Array(PixelUMMDecoderBlockPhaseTiming)
    getter aggregate_phase_wall_seconds : Hash(String, Float64)
    getter aggregate_phase_thread_cpu_seconds : Hash(String, Float64)
    getter head_start_root_count : Int32
    getter head_end_root_count : Int32
    getter root_candidates : Array(Array(String))
    getter phase_candidate_root_duration_sum_seconds : Hash(String, Float64)
    getter phase_definite_root_duration_sum_seconds : Hash(String, Float64)
    getter roots : Array(MPSCommandRootTiming)
    getter root_profile : MPSCommandPhaseProfile
    getter api_profile : MPSAPIProfile

    private def initialize(@blocks : Array(PixelUMMDecoderBlockPhaseTiming),
                           @aggregate_phase_wall_seconds : Hash(String, Float64),
                           @aggregate_phase_thread_cpu_seconds : Hash(String, Float64),
                           @head_start_root_count : Int32,
                           @head_end_root_count : Int32,
                           @root_candidates : Array(Array(String)),
                           @phase_candidate_root_duration_sum_seconds : Hash(String, Float64),
                           @phase_definite_root_duration_sum_seconds : Hash(String, Float64),
                           @roots : Array(MPSCommandRootTiming),
                           @root_profile : MPSCommandPhaseProfile,
                           @api_profile : MPSAPIProfile)
    end

    def self.build(block_markers : Array(Array(PixelUMMBlockPhaseMarker)),
                   tile_counts : Array(Int32),
                   head_start_root_count : Int32,
                   head_end_root_count : Int32,
                   roots : Array(MPSCommandRootTiming),
                   native_duration_sum_seconds : Float64,
                   api_profile : MPSAPIProfile) : self
      unless !block_markers.empty? && block_markers.size == tile_counts.size
        raise ArgumentError.new("PixelUMM decoder profile requires one positive tile count per block")
      end
      raise ArgumentError.new("PixelUMM decoder profile requires command roots") if roots.empty?
      tile_counts.each do |count|
        raise ArgumentError.new("PixelUMM decoder attention tile counts must be positive") unless count > 0
      end

      blocks = Array(PixelUMMDecoderBlockPhaseTiming).new(block_markers.size)
      aggregate_wall = PHASE_NAMES.to_h { |name| {name, 0.0} }
      aggregate_cpu = PHASE_NAMES.to_h { |name| {name, 0.0} }
      previous_block_end = 1
      labels = [] of String
      block_markers.each_with_index do |markers, slot|
        validate_markers!(markers, roots.size, previous_block_end, slot)
        previous_block_end = markers.last.root_count

        wall = {} of String => Float64
        cpu = {} of String => Float64
        markers.each_cons_pair do |start_marker, end_marker|
          phase = start_marker.name
          wall_delta = end_marker.elapsed_seconds - start_marker.elapsed_seconds
          cpu_delta = end_marker.thread_cpu_seconds - start_marker.thread_cpu_seconds
          wall[phase] = wall_delta
          cpu[phase] = cpu_delta
          aggregate_wall[phase] += wall_delta
          aggregate_cpu[phase] += cpu_delta
          labels << "block_#{slot}/#{phase}"
        end

        blocks << PixelUMMDecoderBlockPhaseTiming.new(slot.to_i32,
          tile_counts[slot], markers, wall, cpu)
      end

      unless head_start_root_count == previous_block_end &&
             head_start_root_count >= 1 && head_end_root_count >= head_start_root_count &&
             head_end_root_count == roots.size
        raise ArgumentError.new("PixelUMM decoder head root cursors do not continue the final block through all roots")
      end
      labels << "head"
      validate_api_profile!(api_profile, blocks.size, tile_counts, roots.size)

      root_candidates = Array(Array(String)).new(roots.size) { [] of String }
      blocks.each do |block|
        block.markers.each_cons_pair do |start_marker, end_marker|
          label = "block_#{block.slot}/#{start_marker.name}"
          candidate_roots(start_marker.root_count, end_marker.root_count).each do |root_index|
            root_candidates[root_index] << label
          end
        end
      end
      candidate_roots(head_start_root_count, head_end_root_count).each do |root_index|
        root_candidates[root_index] << "head"
      end
      if root_candidates.any?(&.empty?)
        raise ArgumentError.new("PixelUMM decoder profile leaves a command root without a candidate phase")
      end

      candidate_duration_sums = labels.to_h { |label| {label, 0.0} }
      definite_duration_sums = labels.to_h { |label| {label, 0.0} }
      roots.each_with_index do |root, index|
        duration = root.gpu_end_seconds - root.gpu_start_seconds
        candidates = root_candidates[index]
        candidates.each { |label| candidate_duration_sums[label] += duration }
        definite_duration_sums[candidates.first] += duration if candidates.size == 1
      end

      whole_chain = MPSCommandPhaseProfile.build(roots,
        [MPSCommandPhaseRange.new("whole_chain", 0, roots.size.to_i32)],
        native_duration_sum_seconds)

      new(blocks, aggregate_wall, aggregate_cpu, head_start_root_count,
        head_end_root_count, root_candidates, candidate_duration_sums,
        definite_duration_sums, roots, whole_chain, api_profile)
    end

    private def self.validate_markers!(markers : Array(PixelUMMBlockPhaseMarker),
                                       root_count : Int32,
                                       expected_start_root_count : Int32,
                                       slot : Int32) : Nil
      unless markers.map(&.name) == PixelUMMBlockPhaseProfile::MARKER_NAMES.to_a
        raise ArgumentError.new("PixelUMM decoder block #{slot} markers have incomplete or reordered phases")
      end

      previous_wall = 0.0
      previous_cpu = 0.0
      previous_root = expected_start_root_count
      markers.each do |marker|
        unless marker.elapsed_seconds.finite? && marker.elapsed_seconds >= previous_wall &&
               marker.thread_cpu_seconds.finite? && marker.thread_cpu_seconds >= previous_cpu &&
               marker.root_count >= previous_root && marker.root_count >= 1 &&
               marker.root_count <= root_count
          raise ArgumentError.new("PixelUMM decoder block #{slot} marker clocks or root cursors are invalid")
        end
        previous_wall = marker.elapsed_seconds
        previous_cpu = marker.thread_cpu_seconds
        previous_root = marker.root_count
      end

      unless markers.first.elapsed_seconds == 0.0 &&
             markers.first.thread_cpu_seconds == 0.0 &&
             markers.first.root_count == expected_start_root_count
        raise ArgumentError.new("PixelUMM decoder block #{slot} must start at zero clocks and the contiguous root cursor")
      end
    end

    # The marker's root count names the currently open root as count - 1.
    # Keep both endpoint roots as candidates, matching the single-block
    # profile's treatment of roots that can span a CPU marker.
    private def self.candidate_roots(start_root_count : Int32,
                                     end_root_count : Int32) : Range(Int32, Int32)
      (start_root_count - 1)..(end_root_count - 1)
    end

    private def self.validate_api_profile!(profile : MPSAPIProfile,
                                           block_count : Int32,
                                           tile_counts : Array(Int32),
                                           root_count : Int32) : Nil
      expected_graph_calls = {
        0 => 5_u64,
        1 => 9_u64 * block_count.to_u64,
        2 => 3_u64 * tile_counts.sum(&.to_u64),
        3 => 3_u64 * block_count.to_u64,
        4 => 0_u64,
        5 => 3_u64 * block_count.to_u64,
      }

      graph_calls = 0_u64
      continuation_calls = 0_u64
      linear_calls = 0_u64
      expected_graph_calls.each do |phase_id, expected_count|
        phase = profile.phases.find { |item| item.phase_id == phase_id } ||
                raise ArgumentError.new("PixelUMM decoder API profile is missing phase #{phase_id}")
        graph = phase.operations["graph_encode"]?
        raise ArgumentError.new("PixelUMM decoder API profile is missing Graph timing") unless graph
        unless graph.count == expected_count
          raise ArgumentError.new("PixelUMM decoder API Graph counts do not match blocks and head")
        end
        graph_calls += graph.count
      end

      profile.phases.each do |phase|
        continuation = phase.operations["commit_and_continue"]?
        linear = phase.operations["t5_linear_encode"]?
        unless continuation && linear
          raise ArgumentError.new("PixelUMM decoder API profile is missing operation timing")
        end
        continuation_calls += continuation.count
        linear_calls += linear.count
      end

      expected_total = 5_u64 + 15_u64 * block_count.to_u64 +
                       3_u64 * tile_counts.sum(&.to_u64)
      unless graph_calls == expected_total && linear_calls == 0_u64 &&
             continuation_calls == (root_count - 1).to_u64
        raise ArgumentError.new("PixelUMM decoder API totals do not match the integrated geometry")
      end
    end

    def to_json(json : JSON::Builder) : Nil
      json.object do
        json.field "schema", "cogni_ml.pixelumm_decoder_phase_profile/v1"
        json.field "block_count", @blocks.size
        json.field "blocks" do
          json.array do
            @blocks.each do |block|
              json.object do
                json.field "slot", block.slot
                json.field "attention_tile_count", block.tile_count
                json.field "markers" do
                  write_markers(json, block.markers)
                end
                json.field "phase_wall_seconds" do
                  write_phase_values(json, block.phase_wall_seconds)
                end
                json.field "phase_thread_cpu_seconds" do
                  write_phase_values(json, block.phase_thread_cpu_seconds)
                end
              end
            end
          end
        end
        json.field "aggregate_phase_wall_seconds" do
          write_phase_values(json, @aggregate_phase_wall_seconds)
        end
        json.field "aggregate_phase_thread_cpu_seconds" do
          write_phase_values(json, @aggregate_phase_thread_cpu_seconds)
        end
        json.field "head_root_range" do
          json.object do
            json.field "start_root_count", @head_start_root_count
            json.field "end_root_count", @head_end_root_count
            json.field "candidate_root_indices_inclusive" do
              json.array do
                json.number(@head_start_root_count - 1)
                json.number(@head_end_root_count - 1)
              end
            end
          end
        end
        json.field "root_candidates" do
          @root_candidates.to_json(json)
        end
        json.field "phase_candidate_root_duration_sum_seconds" do
          write_labeled_values(json, @phase_candidate_root_duration_sum_seconds)
        end
        json.field "phase_definite_root_duration_sum_seconds" do
          write_labeled_values(json, @phase_definite_root_duration_sum_seconds)
        end
        json.field "phase_root_duration_sum_semantics",
          "Candidate root-duration sums are non-additive: shared roots may appear in multiple phase candidates and root GPU intervals may overlap; they are not per-phase GPU work."
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
          write_api_profile(json)
        end
      end
    end

    private def write_markers(json : JSON::Builder,
                              markers : Array(PixelUMMBlockPhaseMarker)) : Nil
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

    private def write_phase_values(json : JSON::Builder,
                                   values : Hash(String, Float64)) : Nil
      json.object do
        PHASE_NAMES.each do |name|
          json.field name, values[name]
        end
      end
    end

    private def write_labeled_values(json : JSON::Builder,
                                     values : Hash(String, Float64)) : Nil
      json.object do
        @blocks.each do |block|
          PHASE_NAMES.each do |phase|
            label = "block_#{block.slot}/#{phase}"
            json.field label, values[label]
          end
        end
        json.field "head", values["head"]
      end
    end

    private def write_api_profile(json : JSON::Builder) : Nil
      json.object do
        json.field "source_schema", "MPSAPIProfile WAN phase IDs"
        json.field "phases" do
          json.array do
            @api_profile.phases.each do |phase|
              json.object do
                json.field "pixelumm_phase", API_PHASE_PIXELUMM_NAMES[phase.phase_id]
                json.field "source_phase_id", phase.phase_id
                json.field "source_phase_name", phase.name
                json.field "operations" do
                  json.object do
                    MPSAPIProfile::OPERATION_NAMES.keys.sort.each do |operation_id|
                      name = MPSAPIProfile::OPERATION_NAMES[operation_id]
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
  end
end
