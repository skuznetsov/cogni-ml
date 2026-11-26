# Diagnostic-only interception of production projections. No inference selector,
# reduction, allocation policy or synchronization is changed by this require.
require "json"
require "../../src/ml/audio/drama3/generator_resident"
require "./drama3_decode_replay_coverage"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::Audio::Drama3
    class WeightStore
      def decode_replay_tensor_names : Hash(UInt64, String)
        names = {} of UInt64 => String
        @tensors.each { |name, tensor| names[tensor.object_id] = name }
        names
      end
    end
  end

  module Drama3DecodeReplay
    include ML::Audio::Drama3
    alias Context = Resident::Context
    alias Buffer = Resident::Buffer
    RECEIPT_SOURCE = {{ read_file("#{__DIR__}/drama3_dense_fusion.metal") }}
    SCOPE = "dense_expert_projections_only"

    class Projection
      getter frame : Int32, operation : String, name : String, family : String
      getter matrix : Matrix, input : Buffer, ids : Buffer?, expected : Buffer
      getter reduction : String, rank : Int32, per_rank_input : Bool

      def initialize(@frame, @operation, @name, @family, @matrix, @input, @ids,
                     @expected, @reduction, @rank, @per_rank_input)
      end

      def encode(ctx : Context) : Buffer
        case @operation
        when "matvec"  then ctx.matvec(@matrix, @input, dense_reduction: @reduction)
        when "expert"  then ctx.expert(@matrix, @input, @ids, @rank, @reduction)
        when "experts" then ctx.experts(@matrix, @input, @ids.not_nil!, @per_rank_input, @reduction)
        else                raise "unknown replay operation #{@operation}"
        end
      end

      def kernel : String
        return "dr3_experts" if @operation == "experts"
        return "dr3_expert" if @matrix.scales
        return "dr3_dense_selected" if @operation == "expert" && @ids
        "dr3_dense"
      end

      def metadata : JSON::Any
        JSON.parse({frame: @frame, operation: @operation, name: @name, family: @family,
                    kernel: kernel, rows: @matrix.rows, cols: @matrix.cols,
                    dtype: @matrix.tensor.dtype, bits: @matrix.scales ? @matrix.bits : 0, group_size: @matrix.group_size,
                    reduction: @matrix.scales && @operation == "matvec" ? "native" : @reduction,
                    requested_dense_reduction: @operation == "matvec" ? @reduction : nil,
                    threadgroup_width: 64, rank: @rank, per_rank_input: @per_rank_input,
                    input_values: @input.count, output_values: @expected.count,
                    selected_experts: @ids.try(&.count) || 0,
                    snapshot_bytes: (@input.count.to_i64 + @expected.count + (@ids.try(&.count) || 0)) * 4}.to_json)
      end
    end

    record Arm, command_index : Int64, outputs : Array(Buffer), encode_seconds : Float64

    class Capture
      property frame : Int32 = 0
      property active : Bool = false
      getter records = [] of Projection
      @names : Hash(UInt64, String)
      @omitted_dispatches = Hash(String, Int64).new(0_i64)

      def initialize(@context : Context, store : WeightStore)
        @names = store.decode_replay_tensor_names
        raise "decode replay requires graph depth two" unless @context.graph_stats["depth"] == 2
        raise "decode replay requires native 64-thread experts" unless @context.expert_threads == 64 && !@context.expert_packed_rows_enabled
        raise "decode replay requires tree dense reduction" unless @context.dense_reduction == "tree"
      end

      def observe_dispatch(name : String) : Nil
        @omitted_dispatches[name] += 1 if @active
      end

      def self.family(name : String) : String
        if name.starts_with?("backbone.layers.")
          if name.includes?(".linear_attn.")
            {"qkv", "z", "b", "a"}.each do |role|
              return "backbone.linear_attention.#{role}" if name.ends_with?(".in_proj_#{role}.weight")
            end
            return "backbone.linear_attention.out" if name.ends_with?(".out_proj.weight")
            raise "unclassified linear projection #{name}"
          end
          return "backbone.full_attention" if name.includes?(".self_attn.")
          return "backbone.experts.down" if name.includes?(".switch_mlp.down_proj.")
          return "backbone.experts.gate_up" if name.includes?(".switch_mlp.")
          return "backbone.shared_expert" if name.includes?(".shared_expert.")
          return "backbone.router_and_shared_gate" if name.ends_with?(".mlp.gate.weight") || name.includes?(".shared_expert_gate.")
        elsif name == "lm_head.weight"
          return "main.head"
        elsif name.starts_with?("tower.")
          return "tower.ffn.w1_w3" if name.includes?(".feed_forward.w1.") || name.includes?(".feed_forward.w3.")
          return "tower.ffn.w2" if name.includes?(".feed_forward.w2.")
          return "tower.attention" if name.includes?(".attention.")
          return "tower.eh_proj" if name.includes?(".eh_proj.")
          return "tower.heads" if name.includes?(".codebook_heads.")
        end
        raise "unclassified projection #{name}"
      end

      private def snapshot(value : Buffer) : Buffer
        raise "capture cannot use a recycling workspace" if @context.workspace_active?
        target = @context.alloc(value.count)
        raise "capture snapshot aliases source" if target.buffer.handle.address == value.buffer.handle.address
        @context.copy(value, target)
        target
      end

      # Snapshot before invoking the original method: callers may mutate returned
      # projections in place immediately after this wrapper returns (e.g. RoPE).
      def projection(operation : String, matrix : Matrix, input : Buffer, ids : Buffer?,
                     rank : Int32, reduction : String, per_rank_input : Bool, & : -> Buffer) : Buffer
        raise "inactive capture interception" unless @active
        @active = false
        before_graph, before_reads = @context.graph_stats, @context.readbacks.dup
        name = @names[matrix.tensor.object_id]? || raise "unnamed checkpoint projection"
        family = Capture.family(name)
        raise "unsupported expert reduction" if matrix.scales && operation != "matvec" && reduction != "native"
        raise "unsupported dense reduction" if !matrix.scales && reduction != "tree"
        saved_input, saved_ids = snapshot(input), ids.try { |selection| snapshot(selection) }
        output = yield
        saved_output = snapshot(output)
        raise "capture added a wait or read" unless @context.graph_stats == before_graph && @context.readbacks == before_reads
        @records << Projection.new(@frame, operation, name, family, matrix, saved_input,
          saved_ids, saved_output, reduction, rank, per_rank_input)
        output
      ensure
        @active = true
      end

      def summary : JSON::Any
        JSON.parse({scope: SCOPE, records: @records.size,
                    frames: @records.map(&.frame).uniq.sort, families: @records.map(&.family).uniq.sort,
                    snapshot_bytes: @records.sum { |record| record.metadata["snapshot_bytes"].as_i64 },
                    observed_omitted_generic_dispatches: @omitted_dispatches,
                    unobserved_direct_paths: ["full_attention", "embedding", "codebook_sum", "other direct encoders"],
                    main_head_frame_zero_omitted: true, activation_readbacks: 0,
                    full_decode_coverage_claimed: false}.to_json)
      end

      private def compare(reference : Buffer, actual : Buffer, counters : Buffer) : Nil
        raise "receipt shape" unless reference.count == actual.count
        @context.dispatch("dr3_dense_four_compare", RECEIPT_SOURCE, [reference, actual, counters],
          [reference.count.to_u64], reference.count)
      end

      private def read_receipt(counters : Buffer) : Array(UInt32)
        pointer = counters.buffer.contents.as(UInt32*) + counters.offset // 4
        @context.readbacks["terminal"] += 8_i64
        [pointer[0], pointer[1]]
      end

      private def pair(records : Array(Projection), frame : Int32, family : String, index : Int32,
                       warmups : Int32, repeats : Int32, negative : Tuple(Buffer, Buffer, Buffer),
                       negative_control : Bool) : Int64
        before, reads_before = @context.graph_stats, @context.readbacks.dup
        raise "replay requires drained queue" unless before["pending"] == 0 && before["retained_bytes"] == 0
        order = index.even? ? ["a", "b"] : ["b", "a"]
        arms = {} of String => Arm
        order.each do |name|
          command_index = @context.graph_stats["submitted"]
          started = Time.instant
          outputs = [] of Buffer
          repeats.times { records.each { |record| outputs << record.encode(@context) } }
          encode_seconds = (Time.instant - started).total_seconds
          @context.submit
          raise "unexpected early read/wait" unless @context.readbacks == reads_before && @context.graph_stats["completion_waits"] == before["completion_waits"]
          arms[name] = Arm.new(command_index, outputs, encode_seconds)
        end
        raise "expected two replay commands pending" unless @context.graph_stats["pending"] == 2
        counters = @context.alloc(2)
        @context.dispatch("dr3_dense_four_receipt_clear", RECEIPT_SOURCE, [counters], [] of UInt64, 2)
        values = 0_i64
        arms.each_value do |arm|
          arm.outputs.each_with_index do |output, ordinal|
            compare(records[ordinal % records.size].expected, output, counters)
          end
        end
        values = arms["a"].outputs.sum(&.count.to_i64)
        if negative_control
          @context.dispatch("dr3_dense_four_receipt_clear", RECEIPT_SOURCE, [negative[2]], [] of UInt64, 2)
          compare(negative[0], negative[1], negative[2])
        end
        @context.finish
        counts = read_receipt(counters)
        negative_counts = negative_control ? read_receipt(negative[2]) : nil
        raise "replay bit/nonfinite mismatch #{counts}" unless counts == [0_u32, 0_u32]
        raise "negative control failed #{negative_counts}" if negative_control && negative_counts != [2_u32, 1_u32]
        after = @context.graph_stats
        raise "unexpected diagnostic queue accounting" unless after["submitted"] == before["submitted"] + 3 &&
                                                                 after["completed"] == before["completed"] + 3 &&
                                                                 after["completion_waits"] == before["completion_waits"] + 3 &&
                                                                 after["pending"] == 0 && after["retained_bytes"] == 0
        receipt_bytes = negative_control ? 16 : 8
        raise "unexpected replay readback" unless @context.readbacks.fetch("terminal", 0_i64) == reads_before.fetch("terminal", 0_i64) + receipt_bytes &&
                                                  @context.readbacks.reject("terminal") == reads_before.reject("terminal")
        puts({event: "decode_replay_pair", frame: frame, family: family, pair: index, warmup: index < warmups,
              repeats: repeats, order: order, command_indices: arms.transform_values(&.command_index),
              host_encode_seconds: arms.transform_values(&.encode_seconds), records: records.size,
              values_per_arm: values, bit_mismatches: counts[0], nonfinite_pairs: counts[1],
              negative_control: negative_counts, terminal_receipt_bytes: receipt_bytes,
              graph_before: before, graph_after: after, readbacks_before: reads_before,
              readbacks_after: @context.readbacks}.to_json)
        values
      end

      def run(pairs : Int32 = 7, warmups : Int32 = 2, repeats : Int32 = 1) : Nil
        @active = false
        raise "replay bounds" unless pairs.in?(3..21) && warmups.in?(1..4) && repeats.in?(1..8)
        raise "capture rows incomplete" unless @records.map(&.frame).uniq.sort == [0, 29, 59]
        raise "capture empty" if @records.empty?
        metadata = @records.map(&.metadata)
        Drama3DecodeReplay.assert_coverage!(metadata)
        Drama3DecodeReplay.assert_homogeneous!(metadata)
        groups = @records.group_by { |record| {record.frame, record.family} }
        families = @records.map(&.family).uniq.sort
        # Frame zero starts after its head; a one-call placeholder is not timed.
        frames = [0, 29, 59]
        keys = frames.flat_map { |frame| families.map { |family| {frame, family} } }
        missing = keys.reject { |key| groups.has_key?(key) }
        raise "replay family missing #{missing}" unless missing.all? { |key| key == {0, "main.head"} }
        keys = keys.reject { |key| missing.includes?(key) }
        puts({event: "decode_replay_invocation", format: 1, scope: SCOPE, pairs: pairs, warmups: warmups,
              repeats: repeats, frames: frames, families: families, omitted_cells: missing,
              records: @records.size, covered_values: @records.sum(&.expected.count.to_i64)}.to_json)
        negative = {@context.upload([1_f32, 2_f32]), @context.upload([1.25_f32, Float32::NAN]), @context.alloc(2)}
        # Upload copies are separate setup work, not part of any timed arm.
        @context.finish
        @context.decode_replay_enable_timeline!
        total = 0_i64
        (pairs + warmups).times do |index|
          order = index.even? ? keys : keys.reverse
          order.each_with_index do |key, ordinal|
            total += pair(groups[key], key[0], key[1], index, warmups, repeats, negative, index == 0 && ordinal == 0)
          end
        end
        puts({event: "decode_replay_summary", scope: SCOPE, pairs_completed: keys.size * (pairs + warmups),
              verified_values_per_arm: total, graph: @context.graph_stats, readbacks: @context.readbacks,
              activation_readbacks: 0, full_decode_coverage_claimed: false}.to_json)
      end
    end
  end

  module ML::Audio::Drama3::Resident
    class Context
      property decode_replay_capture : Drama3DecodeReplay::Capture?

      # New flights consult this field when created. Capture has already drained;
      # timestamps are read after the existing waits on replay commands only.
      def decode_replay_enable_timeline! : Nil
        raise "timeline transition requires a drained queue" unless graph_stats["pending"] == 0
        @timeline = true
      end

      def dispatch(name : String, source : String, bindings : Array(Buffer), params : Array(UInt64), count : Int32, group : Int32 = 64) : Nil
        @decode_replay_capture.try(&.observe_dispatch(name))
        previous_def
      end

      def matvec(matrix : Matrix, x : Buffer, dense_reduction : String? = nil) : Buffer
        capture = @decode_replay_capture
        return previous_def unless capture && capture.active
        capture.projection("matvec", matrix, x, nil, 0, dense_reduction || @dense_reduction, false) { previous_def }
      end

      def expert(matrix : Matrix, x : Buffer, ids : Buffer? = nil, rank : Int32 = 0, reduction : String = "native") : Buffer
        capture = @decode_replay_capture
        return previous_def unless capture && capture.active
        capture.projection("expert", matrix, x, ids, rank, reduction, false) { previous_def }
      end

      def experts(matrix : Matrix, x : Buffer, ids : Buffer, per_rank_input : Bool = false, reduction : String = "native") : Buffer
        capture = @decode_replay_capture
        return previous_def unless capture && capture.active
        capture.projection("experts", matrix, x, ids, 0, reduction, per_rank_input) { previous_def }
      end

      def matvec_four(matrices : Tuple(Matrix, Matrix, Matrix, Matrix), x : Buffer,
                      dense_reduction : String? = nil, fused : Bool = false) : Tuple(Buffer, Buffer, Buffer, Buffer)
        raise "uncovered fused four projection" if fused && @decode_replay_capture.try(&.active)
        previous_def
      end

      def matvec_pair(matrices : Tuple(Matrix, Matrix), x : Buffer,
                      dense_reduction : String? = nil, fused : Bool = false) : Tuple(Buffer, Buffer)
        raise "uncovered fused pair projection" if fused && @decode_replay_capture.try(&.active)
        previous_def
      end
    end
  end
{% end %}
