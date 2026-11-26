# Diagnostic-only interception and receipt support for the expert down-sum
# benchmark. Captured operands and results are copied on Metal; no activation
# values are returned to the host.
require "json"
require "set"
require "../../src/ml/audio/drama3/generator_resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::Audio::Drama3
    class WeightStore
      def expert_down_sum_tensor_names : Hash(UInt64, String)
        names = {} of UInt64 => String
        @tensors.each { |name, tensor| names[tensor.object_id] = name }
        names
      end
    end
  end

  module Drama3ExpertDownSum
    include ML::Audio::Drama3
    alias Context = Resident::Context
    alias Buffer = Resident::Buffer
    alias DownSum = NamedTuple(output: Buffer, down: Buffer?)
    alias Signature = Tuple(Int32, Int32, Int32, Int32, Int32)

    RECEIPT_SOURCE = {{ read_file("#{__DIR__}/drama3_dense_fusion.metal") }}
    SELECTED_ROWS = [0, 29, 59] of Int32

    record InputVariation, name : String, kind : String, from_frame : Int32, to_frame : Int32,
      reference : Buffer, actual : Buffer

    class DownProjection
      getter frame : Int32
      getter name : String
      getter matrix : Matrix
      getter input : Buffer
      getter ids : Buffer
      getter weights : Buffer
      getter candidate_down : Buffer
      getter candidate_output : Buffer
      property reference_down : Buffer?
      property reference_output : Buffer?

      def initialize(@frame, @name, @matrix, @input, @ids, @weights,
                     @candidate_down, @candidate_output)
        @reference_down = nil
        @reference_output = nil
      end

      def layer : Int32
        pieces = @name.split('.')
        raise "unexpected down matrix name #{@name}" unless pieces.size > 3 && pieces[0] == "backbone" && pieces[1] == "layers"
        pieces[2].to_i
      end

      def groups : Int32
        @matrix.cols // @matrix.group_size
      end

      def ranks : Int32
        @ids.count
      end

      def eligible : Bool
        @matrix.group_size == 64 && @matrix.bits.in?(4, 8) && groups.in?(1..32) &&
          64 % groups == 0 && ranks.in?(1..8) && groups * ranks <= 256 &&
          @input.count == @matrix.cols.to_i64 * ranks && @weights.count == ranks
      end

      def signature : Signature
        {@matrix.rows, @matrix.cols, @matrix.bits, @matrix.group_size, ranks}
      end

      def metadata : JSON::Any
        JSON.parse({frame: @frame, layer: layer, name: @name,
          matrix_rows: @matrix.rows, matrix_cols: @matrix.cols, dtype: @matrix.tensor.dtype,
          bits: @matrix.bits, group_size: @matrix.group_size, groups: groups, ranks: ranks,
          eligible: eligible, reduction: "native", effective_threads: 64,
          input_values: @input.count, id_values: @ids.count, weight_values: @weights.count,
          candidate_down_values: @candidate_down.count, candidate_output_values: @candidate_output.count,
          capture_bytes: (@input.count + @ids.count + @weights.count + @candidate_down.count + @candidate_output.count) * 4}.to_json)
      end
    end

    class Capture
      property frame : Int32 = 0
      property active : Bool = false
      getter records = [] of DownProjection
      @names : Hash(UInt64, String)
      @initial_fused_dispatches : Int64

      def initialize(@context : Context, store : WeightStore)
        @names = store.expert_down_sum_tensor_names
        @initial_fused_dispatches = @context.expert_down_sum_fused_dispatches
        raise "capture requires the frozen candidate selector" unless @context.expert_down_sum_fusion_enabled
        raise "capture requires 64 expert threads" unless @context.expert_threads == 64
        raise "capture requires packed rows disabled" if @context.expert_packed_rows_enabled
        raise "capture cannot use a recycling workspace" if @context.workspace_active?
      end

      private def snapshot(value : Buffer) : Buffer
        target = @context.alloc(value.count)
        raise "capture snapshot aliases source" if target.buffer.handle.address == value.buffer.handle.address
        @context.copy(value, target)
        target
      end

      private def assert_eligible!(matrix : Matrix, input : Buffer, ids : Buffer, weights : Buffer,
                                   reduction : String, capture_down : Bool) : Nil
        groups = matrix.cols // matrix.group_size
        unless matrix.group_size == 64 && matrix.bits.in?(4, 8) && groups.in?(1..32) && 64 % groups == 0 &&
               ids.count.in?(1..8) && groups * ids.count <= 256 && @context.expert_threads == 64 &&
               reduction == "native" && capture_down && input.count == matrix.cols.to_i64 * ids.count &&
               weights.count == ids.count
          raise "ineligible expert down-sum capture for #{matrix.rows}x#{matrix.cols} #{matrix.bits}-bit groups=#{groups} ranks=#{ids.count}"
        end
      end

      # Snapshot all inputs before invoking the production method because the
      # backbone may immediately mutate or recycle its activation buffers.
      def intercept(matrix : Matrix, input : Buffer, ids : Buffer, weights : Buffer,
                    reduction : String, capture_down : Bool, & : -> DownSum) : DownSum
        raise "capture intercepted an inactive row" unless @active
        raise "capture exceeded the selected row set" unless SELECTED_ROWS.includes?(@frame)
        assert_eligible!(matrix, input, ids, weights, reduction, capture_down)
        name = @names[matrix.tensor.object_id]? || raise "unnamed expert down matrix"
        raise "unexpected matrix family #{name}" unless name.starts_with?("backbone.layers.") &&
          name.includes?(".switch_mlp.down_proj.")
        before_graph, before_reads = @context.graph_stats.dup, @context.readbacks.dup
        before_fused = @context.expert_down_sum_fused_dispatches
        saved_input, saved_ids, saved_weights = snapshot(input), snapshot(ids), snapshot(weights)
        @active = false
        result = yield
        raise "expert down-sum fell back instead of dispatching the candidate" unless
          @context.expert_down_sum_fused_dispatches == before_fused + 1
        saved_down = result[:down] || raise "candidate capture omitted per-rank down output"
        saved_output = snapshot(result[:output])
        saved_down = snapshot(saved_down)
        expected_down_values = matrix.rows.to_i64 * ids.count
        raise "candidate capture result shape mismatch" unless saved_output.count == matrix.rows && saved_down.count == expected_down_values
        raise "capture added a wait or read" unless @context.graph_stats == before_graph && @context.readbacks == before_reads
        @records << DownProjection.new(@frame, name, matrix, saved_input, saved_ids, saved_weights,
          saved_down, saved_output)
        result
      ensure
        @active = true
      end

      def assert_inventory! : Nil
        expected = Set(String).new
        40.times { |layer| expected << "backbone.layers.#{layer}.mlp.switch_mlp.down_proj.weight" }
        raise "selected expert-down rows are missing" unless @records.map(&.frame).uniq.sort == SELECTED_ROWS
        SELECTED_ROWS.each do |row|
          observed = @records.select { |record| record.frame == row }.map(&.name)
          raise "selected row #{row} did not capture exactly 40 real down matrices" unless observed.size == 40 && observed.uniq.size == 40
          missing = expected.reject { |name| observed.includes?(name) }.to_a.sort
          raise "selected row #{row} missing down matrix #{missing.first}" unless missing.empty?
          unexpected = observed.reject { |name| expected.includes?(name) }
          raise "selected row #{row} captured unexpected down matrix #{unexpected.first}" unless unexpected.empty?
        end
        raise "captured down projection was not eligible" unless @records.all?(&.eligible)
        expected_dispatches = SELECTED_ROWS.size.to_i64 * expected.size
        actual_dispatches = @context.expert_down_sum_fused_dispatches - @initial_fused_dispatches
        raise "candidate dispatch count #{actual_dispatches} is below the #{expected_dispatches} captured calls" unless
          actual_dispatches >= expected_dispatches
      end

      def inventory : JSON::Any
        assert_inventory!
        JSON.parse({selected_rows: SELECTED_ROWS,
          total_records: @records.size,
          per_row_counts: SELECTED_ROWS.to_h { |row| {row.to_s, @records.count { |record| record.frame == row }} },
          unique_matrix_names_per_row: SELECTED_ROWS.to_h do |row|
            {row.to_s, @records.select { |record| record.frame == row }.map(&.name).uniq.sort}
          end,
          eligible_records: @records.count(&.eligible),
          signatures: @records.map(&.signature).uniq.sort_by { |entry| entry.to_a.join(":") },
          captured_candidate_dispatches: @records.size,
          total_candidate_dispatches_since_context_creation: @context.expert_down_sum_fused_dispatches - @initial_fused_dispatches}.to_json)
      end

      def input_variations : Array(InputVariation)
        grouped = {} of Tuple(Int32, String) => DownProjection
        @records.each { |record| grouped[{record.frame, record.name}] = record }
        variations = [] of InputVariation
        [{0, 29}, {29, 59}].each do |from, to|
          grouped.select { |key, _| key[0] == from }.each do |key, reference|
            name = key[1]
            actual = grouped[{to, name}]? || raise "missing matching down input for #{name} row #{to}"
            raise "varying-input shape mismatch for #{name}" unless reference.signature == actual.signature
            {"down_input" => {reference.input, actual.input},
             "selected_ids" => {reference.ids, actual.ids},
             "selected_weights" => {reference.weights, actual.weights}}.each do |kind, buffers|
              variations << InputVariation.new(name, kind, from, to, buffers[0], buffers[1])
            end
          end
        end
        variations
      end
    end
  end

  module ML::Audio::Drama3::Resident
    class Context
      property expert_down_sum_capture : Drama3ExpertDownSum::Capture?

      # Enable the existing Metal timeline only after ordinary generation and
      # diagnostic reference work have drained, so generation policy is intact.
      def enable_expert_down_sum_timeline! : Nil
        raise "timeline transition requires a drained queue" unless graph_stats["pending"] == 0
        @timeline = true
      end

      def experts_down_sum(matrix : ML::Audio::Drama3::Matrix, input : Buffer, ids : Buffer, weights : Buffer,
                           reduction : String = "native", capture_down : Bool = false) : Drama3ExpertDownSum::DownSum
        capture = @expert_down_sum_capture
        return previous_def unless capture && capture.active
        # Device tracing is intentionally independent of backbone MoE tracing,
        # so selected production calls arrive with capture_down=false. Force the
        # diagnostic output only for those calls while the interceptor is active.
        capture.intercept(matrix, input, ids, weights, reduction, true) do
          previous_def(matrix, input, ids, weights, reduction, true)
        end
      end
    end
  end
{% end %}
