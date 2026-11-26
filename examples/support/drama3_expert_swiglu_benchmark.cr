# Diagnostic-only interception and receipts for the routed and dense SwiGLU
# benchmark. Activations and comparisons remain in Metal buffers.
require "json"
require "set"
require "../../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::Audio::Drama3
    class WeightStore
      def expert_swiglu_tensor_names : Hash(UInt64, String)
        names = {} of UInt64 => String
        @tensors.each { |name, tensor| names[tensor.object_id] = name }
        names
      end
    end
  end

  module Drama3ExpertSwiGLU
    include ML::Audio::Drama3
    alias Context = Resident::Context
    alias Buffer = Resident::Buffer
    alias SwiGLUResult = NamedTuple(output: Buffer, gate: Buffer?, up: Buffer?)
    alias Signature = Tuple(String, Int32, Int32, Int32, Int32, Int32)

    RECEIPT_SOURCE = {{ read_file("#{__DIR__}/drama3_dense_fusion.metal") }}
    SELECTED_ROWS = [0, 29, 59] of Int32
    FAMILIES = ["routed", "shared"] of String

    record InputVariation, name : String, family : String, kind : String,
      from_frame : Int32, to_frame : Int32, reference : Buffer, actual : Buffer

    class Projection
      getter frame : Int32
      getter family : String
      getter gate_name : String
      getter up_name : String
      getter gate_matrix : Matrix
      getter up_matrix : Matrix
      getter input : Buffer
      getter ids : Buffer?
      getter candidate_activation : Buffer
      getter candidate_gate : Buffer
      getter candidate_up : Buffer
      property reference_gate : Buffer?
      property reference_up : Buffer?
      property reference_activation : Buffer?

      def initialize(@frame, @family, @gate_name, @up_name, @gate_matrix, @up_matrix,
                     @input, @ids, @candidate_activation, @candidate_gate, @candidate_up)
        @reference_gate = nil
        @reference_up = nil
        @reference_activation = nil
      end

      def layer : Int32
        pieces = @gate_name.split('.')
        raise "unexpected SwiGLU matrix name #{@gate_name}" unless pieces.size > 3 &&
          pieces[0] == "backbone" && pieces[1] == "layers"
        pieces[2].to_i
      end

      def ranks : Int32
        @ids.try(&.count) || 1
      end

      def groups : Int32
        return 0 unless @gate_matrix.scales
        @gate_matrix.cols // @gate_matrix.group_size
      end

      def effective_group_size : Int32
        @gate_matrix.scales ? @gate_matrix.group_size : 0
      end

      def signature : Signature
        {@family, @gate_matrix.rows, @gate_matrix.cols, @gate_matrix.bits,
         effective_group_size, ranks}
      end

      def eligible : Bool
        return false unless @gate_matrix.rows == @up_matrix.rows && @gate_matrix.cols == @up_matrix.cols
        return false unless @input.count == @gate_matrix.cols
        return false unless @candidate_gate.count == @gate_matrix.rows * ranks &&
          @candidate_up.count == @up_matrix.rows * ranks &&
          @candidate_activation.count == @gate_matrix.rows * ranks

        case @family
        when "routed"
          @gate_matrix.bits == @up_matrix.bits && @gate_matrix.group_size == @up_matrix.group_size &&
            @gate_matrix.group_size == 64 && @gate_matrix.bits.in?(4, 8) && groups.in?(1..32) &&
            ranks.in?(1..32) && @gate_matrix.rows.to_i64 * ranks <= Int32::MAX.to_i64 &&
            !!@gate_matrix.scales && !!@gate_matrix.biases && !!@up_matrix.scales && !!@up_matrix.biases &&
            @gate_matrix.tensor.dtype == "U32" && @up_matrix.tensor.dtype == "U32" &&
            @gate_matrix.tensor.shape == @up_matrix.tensor.shape
        when "shared"
          @gate_matrix.tensor.dtype == "BF16" && @up_matrix.tensor.dtype == "BF16" &&
            @gate_matrix.scales.nil? && @up_matrix.scales.nil? &&
            @gate_matrix.biases.nil? && @up_matrix.biases.nil? &&
            @gate_matrix.expert == 0 && @up_matrix.expert == 0 &&
            @gate_matrix.tensor.shape.size == 2 && @up_matrix.tensor.shape.size == 2
        else
          false
        end
      end

      def metadata : JSON::Any
        JSON.parse({frame: @frame, layer: layer, family: @family,
          gate_matrix: @gate_name, up_matrix: @up_name,
          gate_rows: @gate_matrix.rows, gate_cols: @gate_matrix.cols,
          gate_bits: @gate_matrix.scales ? @gate_matrix.bits : nil,
          up_bits: @up_matrix.scales ? @up_matrix.bits : nil,
          gate_dtype: @gate_matrix.tensor.dtype, up_dtype: @up_matrix.tensor.dtype,
          group_size: @gate_matrix.scales ? @gate_matrix.group_size : nil,
          groups: groups, ranks: ranks,
          eligible: eligible, reduction: @family == "shared" ? "tree" : "native",
          effective_threads: 64, input_values: @input.count,
          id_values: @ids.try(&.count) || 0,
          candidate_gate_values: @candidate_gate.count,
          candidate_up_values: @candidate_up.count,
          candidate_activation_values: @candidate_activation.count,
          capture_bytes: (@input.count + (@ids.try(&.count) || 0) +
            @candidate_gate.count + @candidate_up.count + @candidate_activation.count) * 4}.to_json)
      end
    end

    class Capture
      property frame : Int32 = 0
      property active : Bool = false
      getter records = [] of Projection
      @names : Hash(UInt64, String)
      @initial_expert_dispatches : Int64
      @initial_dense_dispatches : Int64

      def initialize(@context : Context, store : WeightStore)
        @names = store.expert_swiglu_tensor_names
        @initial_expert_dispatches = @context.expert_swiglu_fused_dispatches
        @initial_dense_dispatches = @context.dense_swiglu_fused_dispatches
        raise "capture requires the frozen candidate selector" unless @context.expert_swiglu_fusion_enabled
        raise "capture requires 64 expert threads" unless @context.expert_threads == 64
        raise "capture requires packed expert rows disabled" if @context.expert_packed_rows_enabled
        raise "capture cannot use a recycling workspace" if @context.workspace_active?
      end

      private def snapshot(value : Buffer) : Buffer
        target = @context.alloc(value.count)
        raise "capture snapshot aliases source" if target.buffer.handle.address == value.buffer.handle.address
        @context.copy(value, target)
        target
      end

      private def assert_eligible!(family : String, gate_matrix : Matrix, up_matrix : Matrix,
                                   input : Buffer, ids : Buffer?) : Nil
        ranks = ids.try(&.count) || 1
        common = gate_matrix.rows == up_matrix.rows && gate_matrix.cols == up_matrix.cols &&
          input.count == gate_matrix.cols && @context.expert_threads == 64
        family_eligible = case family
                          when "routed"
                            groups = gate_matrix.cols // gate_matrix.group_size
                            gate_matrix.bits == up_matrix.bits && gate_matrix.group_size == up_matrix.group_size &&
                              gate_matrix.group_size == 64 && gate_matrix.bits.in?(4, 8) && groups.in?(1..32) &&
                              ranks.in?(1..32) && gate_matrix.rows.to_i64 * ranks <= Int32::MAX.to_i64 &&
                              !!gate_matrix.scales && !!gate_matrix.biases && !!up_matrix.scales && !!up_matrix.biases &&
                              gate_matrix.tensor.dtype == "U32" && up_matrix.tensor.dtype == "U32" &&
                              gate_matrix.tensor.shape == up_matrix.tensor.shape && ids != nil
                          when "shared"
                            gate_matrix.tensor.dtype == "BF16" && up_matrix.tensor.dtype == "BF16" &&
                              gate_matrix.scales.nil? && up_matrix.scales.nil? &&
                              gate_matrix.biases.nil? && up_matrix.biases.nil? &&
                              gate_matrix.expert == 0 && up_matrix.expert == 0 &&
                              gate_matrix.tensor.shape.size == 2 && up_matrix.tensor.shape.size == 2 && ids.nil?
                          else
                            false
                          end
        raise "ineligible #{family} SwiGLU capture for #{gate_matrix.rows}x#{gate_matrix.cols} dtype=#{gate_matrix.tensor.dtype} bits=#{gate_matrix.bits} ranks=#{ranks}" unless common && family_eligible
      end

      private def capture_result(frame : Int32, family : String, gate_matrix : Matrix, up_matrix : Matrix,
                                 input : Buffer, ids : Buffer?, result : SwiGLUResult,
                                 before_graph : Hash(String, Int64), before_reads : Hash(String, Int64),
                                 expected_dispatch : Int64) : SwiGLUResult
        gate_name = @names[gate_matrix.tensor.object_id]? || raise "unnamed SwiGLU gate matrix"
        up_name = @names[up_matrix.tensor.object_id]? || raise "unnamed SwiGLU up matrix"
        expected_family = family == "routed" ? ".switch_mlp." : ".shared_expert."
        raise "unexpected gate matrix family #{gate_name}" unless gate_name.starts_with?("backbone.layers.") && gate_name.includes?(expected_family) && gate_name.ends_with?("gate_proj.weight")
        raise "unexpected up matrix family #{up_name}" unless up_name.starts_with?("backbone.layers.") && up_name.includes?(expected_family) && up_name.ends_with?("up_proj.weight")
        raise "gate/up layer mismatch" unless gate_name.split('.')[2] == up_name.split('.')[2]
        raise "SwiGLU capture omitted gate projection" unless gate = result[:gate]
        raise "SwiGLU capture omitted up projection" unless up = result[:up]
        saved_input, saved_ids = snapshot(input), ids.try { |value| snapshot(value) }
        saved_output, saved_gate, saved_up = snapshot(result[:output]), snapshot(gate), snapshot(up)
        ranks = ids.try(&.count) || 1
        raise "SwiGLU capture result shape mismatch" unless saved_gate.count == gate_matrix.rows * ranks &&
          saved_up.count == up_matrix.rows * ranks && saved_output.count == gate_matrix.rows * ranks
        actual_dispatch = family == "routed" ? @context.expert_swiglu_fused_dispatches : @context.dense_swiglu_fused_dispatches
        raise "#{family} SwiGLU fell back instead of dispatching the candidate" unless actual_dispatch == expected_dispatch + 1
        raise "capture added a wait or read" unless @context.graph_stats == before_graph && @context.readbacks == before_reads
        @records << Projection.new(frame, family, gate_name, up_name, gate_matrix, up_matrix,
          saved_input, saved_ids, saved_output, saved_gate, saved_up)
        result
      end

      def intercept_expert(gate_matrix : Matrix, up_matrix : Matrix, input : Buffer, ids : Buffer,
                           & : -> SwiGLUResult) : SwiGLUResult
        intercept("routed", gate_matrix, up_matrix, input, ids) { yield }
      end

      def intercept_dense(gate_matrix : Matrix, up_matrix : Matrix, input : Buffer,
                          & : -> SwiGLUResult) : SwiGLUResult
        intercept("shared", gate_matrix, up_matrix, input, nil) { yield }
      end

      private def intercept(family : String, gate_matrix : Matrix, up_matrix : Matrix, input : Buffer,
                            ids : Buffer?, & : -> SwiGLUResult) : SwiGLUResult
        raise "capture intercepted an inactive row" unless @active
        raise "capture exceeded selected row set" unless SELECTED_ROWS.includes?(@frame)
        assert_eligible!(family, gate_matrix, up_matrix, input, ids)
        before_graph, before_reads = @context.graph_stats.dup, @context.readbacks.dup
        expected = family == "routed" ? @context.expert_swiglu_fused_dispatches : @context.dense_swiglu_fused_dispatches
        @active = false
        result = yield
        capture_result(@frame, family, gate_matrix, up_matrix, input, ids, result,
          before_graph, before_reads, expected)
      ensure
        @active = true
      end

      def assert_inventory! : Nil
        raise "selected SwiGLU rows are missing" unless @records.map(&.frame).uniq.sort == SELECTED_ROWS
        SELECTED_ROWS.each do |row|
          FAMILIES.each do |family|
            expected_gate = Set(String).new
            expected_up = Set(String).new
            40.times do |layer|
              prefix = "backbone.layers.#{layer}.mlp."
              middle = family == "routed" ? "switch_mlp" : "shared_expert"
              expected_gate << "#{prefix}#{middle}.gate_proj.weight"
              expected_up << "#{prefix}#{middle}.up_proj.weight"
            end
            observed = @records.select { |record| record.frame == row && record.family == family }
            raise "selected row #{row} did not capture exactly 40 #{family} SwiGLU calls" unless observed.size == 40
            gates = observed.map(&.gate_name).to_set
            ups = observed.map(&.up_name).to_set
            raise "selected row #{row} missing #{family} gate #{(expected_gate - gates).first?}" unless gates == expected_gate
            raise "selected row #{row} missing #{family} up #{(expected_up - ups).first?}" unless ups == expected_up
          end
        end
        raise "captured SwiGLU candidate was ineligible" unless @records.all?(&.eligible)
        expected = SELECTED_ROWS.size.to_i64 * 40
        expert_count = @context.expert_swiglu_fused_dispatches - @initial_expert_dispatches
        dense_count = @context.dense_swiglu_fused_dispatches - @initial_dense_dispatches
        raise "routed SwiGLU candidate dispatch count #{expert_count} < #{expected}" if expert_count < expected
        raise "shared SwiGLU candidate dispatch count #{dense_count} < #{expected}" if dense_count < expected
      end

      def inventory : JSON::Any
        assert_inventory!
        JSON.parse({selected_rows: SELECTED_ROWS, total_records: @records.size,
          per_row_counts: SELECTED_ROWS.to_h do |row|
            {row.to_s, FAMILIES.to_h { |family| {family, @records.count { |record| record.frame == row && record.family == family}} }}
          end,
          matrix_names_per_row: SELECTED_ROWS.to_h do |row|
            {row.to_s, FAMILIES.to_h do |family|
              {family, @records.select { |record| record.frame == row && record.family == family }.map { |record| {gate: record.gate_name, up: record.up_name} }.sort_by(&.[:gate])}
            end}
          end,
          eligible_records: @records.count(&.eligible), signatures: @records.map(&.signature).uniq.sort_by(&.to_a.join(":")),
          captured_routed_candidate_dispatches: @records.count { |record| record.family == "routed" },
          captured_shared_candidate_dispatches: @records.count { |record| record.family == "shared" },
          total_routed_candidate_dispatches_since_context_creation: @context.expert_swiglu_fused_dispatches - @initial_expert_dispatches,
          total_shared_candidate_dispatches_since_context_creation: @context.dense_swiglu_fused_dispatches - @initial_dense_dispatches}.to_json)
      end

      def input_variations : Array(InputVariation)
        grouped = {} of Tuple(Int32, String, String) => Projection
        @records.each { |record| grouped[{record.frame, record.family, record.gate_name}] = record }
        variations = [] of InputVariation
        [{0, 29}, {29, 59}].each do |from, to|
          grouped.select { |key, _| key[0] == from }.each do |key, reference|
            actual = grouped[{to, key[1], key[2]}]? || raise "missing matching #{key[1]} input for #{key[2]} row #{to}"
            raise "varying-input shape mismatch for #{key[2]}" unless reference.signature == actual.signature
            variations << InputVariation.new(key[2], key[1], "input", from, to, reference.input, actual.input)
            if reference.ids && actual.ids
              variations << InputVariation.new(key[2], key[1], "ids", from, to,
                reference.ids.not_nil!, actual.ids.not_nil!)
            end
          end
        end
        variations
      end
    end
  end

  module ML::Audio::Drama3::Resident
    class Context
      property expert_swiglu_capture : Drama3ExpertSwiGLU::Capture?

      def enable_expert_swiglu_timeline! : Nil
        raise "timeline transition requires a drained queue" unless graph_stats["pending"] == 0
        @timeline = true
      end

      def experts_swiglu(gate_matrix : ML::Audio::Drama3::Matrix, up_matrix : ML::Audio::Drama3::Matrix,
                         input : Buffer, ids : Buffer, capture_projections : Bool = false) : Drama3ExpertSwiGLU::SwiGLUResult
        capture = @expert_swiglu_capture
        return previous_def unless capture && capture.active
        capture.intercept_expert(gate_matrix, up_matrix, input, ids) do
          previous_def(gate_matrix, up_matrix, input, ids, capture_projections: true)
        end
      end

      def matvec_swiglu(gate_matrix : ML::Audio::Drama3::Matrix, up_matrix : ML::Audio::Drama3::Matrix,
                        input : Buffer, dense_reduction : String? = nil,
                        capture_projections : Bool = false) : Drama3ExpertSwiGLU::SwiGLUResult
        capture = @expert_swiglu_capture
        return previous_def unless capture && capture.active
        capture.intercept_dense(gate_matrix, up_matrix, input) do
          previous_def(gate_matrix, up_matrix, input, dense_reduction, capture_projections: true)
        end
      end
    end
  end
{% end %}
